#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import zlib from "node:zlib";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ONE_SECOND_TOLERANCE_SECONDS = 0.011;
const DEFAULT_HISTORY_SECONDS = 4.5;
const MAXIMUM_HISTORY_SECONDS = 30.0;
const MAXIMUM_COLLISION_TOLERANCE_METERS = 0.5;
const CONTACT_EPSILON_METERS = 1e-4;
const GAP_BISECTION_ITERATIONS = 30;
const EVIDENCE_SCHEMA = "paper_figure_evidence";
const EVIDENCE_VERSION = 1;
const RENDERER_SCRIPT = path.join(path.dirname(fileURLToPath(import.meta.url)), "render_paper_figure_panels.py");
const ROAD_STYLE_BY_TYPE = Object.freeze({ 0: "lane", 1: "road_line", 2: "edge" });
const FALLBACK_ROAD_STYLE = "yellow_line";

const CHUNK_TYPES = Object.freeze({
    float32: { bytes: 4, read: (buffer, offset) => buffer.readFloatLE(offset) },
    int32: { bytes: 4, read: (buffer, offset) => buffer.readInt32LE(offset) },
    int16: { bytes: 2, read: (buffer, offset) => buffer.readInt16LE(offset) },
    uint8: { bytes: 1, read: (buffer, offset) => buffer.readUInt8(offset) },
});

function fail(message) {
    throw new Error(message);
}

function usage() {
    return [
        "Usage:",
        "  node scripts/export_regents_failure_figures.mjs [--evidence-only] [--history-seconds 3] [--brake-seconds N] [--warning-seconds N] [--collision-tolerance-meters N] <source.html|source.replay.zlib> [output-directory]",
        "",
        "The source must be a genuine target failure with an avoided braking candidate and",
        "a dangerous, finite TTC sample at least 1.0 second before collision.",
    ].join("\n");
}

function parseCliArguments(cliArguments) {
    const positionalArguments = [];
    let evidenceOnly = false;
    let historySeconds = DEFAULT_HISTORY_SECONDS;
    let brakeSeconds = null;
    let warningSeconds = null;
    let collisionToleranceMeters = CONTACT_EPSILON_METERS;
    for (let argumentIndex = 0; argumentIndex < cliArguments.length; argumentIndex++) {
        const argument = cliArguments[argumentIndex];
        if (argument === "--evidence-only") {
            evidenceOnly = true;
            continue;
        }
        if (argument === "--history-seconds") {
            argumentIndex++;
            if (argumentIndex >= cliArguments.length) {
                fail("--history-seconds requires a numeric value");
            }
            historySeconds = Number(cliArguments[argumentIndex]);
            continue;
        }
        if (argument.startsWith("--history-seconds=")) {
            historySeconds = Number(argument.slice("--history-seconds=".length));
            continue;
        }
        if (argument === "--brake-seconds") {
            argumentIndex++;
            if (argumentIndex >= cliArguments.length) {
                fail("--brake-seconds requires a numeric value");
            }
            brakeSeconds = Number(cliArguments[argumentIndex]);
            continue;
        }
        if (argument.startsWith("--brake-seconds=")) {
            brakeSeconds = Number(argument.slice("--brake-seconds=".length));
            continue;
        }
        if (argument === "--warning-seconds") {
            argumentIndex++;
            if (argumentIndex >= cliArguments.length) {
                fail("--warning-seconds requires a numeric value");
            }
            warningSeconds = Number(cliArguments[argumentIndex]);
            continue;
        }
        if (argument === "--collision-tolerance-meters") {
            argumentIndex++;
            if (argumentIndex >= cliArguments.length) {
                fail("--collision-tolerance-meters requires a numeric value");
            }
            collisionToleranceMeters = Number(cliArguments[argumentIndex]);
            continue;
        }
        if (argument.startsWith("--warning-seconds=")) {
            warningSeconds = Number(argument.slice("--warning-seconds=".length));
            continue;
        }
        if (argument.startsWith("-") && argument !== "-h" && argument !== "--help") {
            fail(`Unknown option: ${argument}`);
        }
        positionalArguments.push(argument);
    }
    if (!(historySeconds > 0) || !Number.isFinite(historySeconds) || historySeconds > MAXIMUM_HISTORY_SECONDS) {
        fail(`History duration must be finite and in (0, ${MAXIMUM_HISTORY_SECONDS}] seconds`);
    }
    if (brakeSeconds !== null && (
        !(brakeSeconds > 0) || !Number.isFinite(brakeSeconds) || brakeSeconds > MAXIMUM_HISTORY_SECONDS
    )) {
        fail(`Brake lead time must be finite and in (0, ${MAXIMUM_HISTORY_SECONDS}] seconds`);
    }
    if (warningSeconds !== null && (
        !(warningSeconds > 0) || !Number.isFinite(warningSeconds) || warningSeconds > MAXIMUM_HISTORY_SECONDS
    )) {
        fail(`Warning lead time must be finite and in (0, ${MAXIMUM_HISTORY_SECONDS}] seconds`);
    }
    if (!(collisionToleranceMeters >= 0) || collisionToleranceMeters > MAXIMUM_COLLISION_TOLERANCE_METERS) {
        fail(`Collision tolerance must be in [0, ${MAXIMUM_COLLISION_TOLERANCE_METERS}] meters`);
    }
    return { positionalArguments, evidenceOnly, historySeconds, brakeSeconds, warningSeconds, collisionToleranceMeters };
}

function product(dimensions) {
    let elementCount = 1;
    for (const dimension of dimensions) {
        if (!Number.isInteger(dimension) || dimension < 0) {
            fail(`Invalid chunk dimension: ${dimension}`);
        }
        elementCount *= dimension;
        if (!Number.isSafeInteger(elementCount)) {
            fail("Chunk shape exceeds JavaScript's safe integer range");
        }
    }
    return elementCount;
}

function extractCompressedPayload(sourcePath, sourceBytes) {
    if (!sourcePath.endsWith(".html")) {
        return sourceBytes;
    }

    const html = sourceBytes.toString("utf8");
    const payloadPattern = /<script[^>]*class="payload-chunk"[^>]*>([\s\S]*?)<\/script>/g;
    const compressedParts = [];
    let payloadMatch;
    while ((payloadMatch = payloadPattern.exec(html)) !== null) {
        const encodedPart = payloadMatch[1].replace(/\s+/g, "");
        if (encodedPart.length === 0 || encodedPart.length % 4 === 1) {
            fail("Replay HTML contains an invalid base64 payload chunk");
        }
        compressedParts.push(Buffer.from(encodedPart, "base64"));
    }
    if (compressedParts.length === 0) {
        fail("Replay HTML contains no payload-chunk script elements");
    }
    return Buffer.concat(compressedParts);
}

function decodeReplay(sourcePath) {
    const sourceBytes = fs.readFileSync(sourcePath);
    const compressedPayload = extractCompressedPayload(sourcePath, sourceBytes);
    let payload;
    try {
        payload = zlib.inflateSync(compressedPayload);
    } catch (error) {
        fail(`Unable to inflate replay payload: ${error.message}`);
    }

    if (payload.length < 4) {
        fail("Replay payload is truncated before its header length");
    }
    const headerLength = payload.readUInt32LE(0);
    if (headerLength <= 1 || 4 + headerLength > payload.length) {
        fail(`Replay header length ${headerLength} is outside the payload`);
    }

    let header;
    try {
        header = JSON.parse(payload.subarray(4, 4 + headerLength).toString("utf8"));
    } catch (error) {
        fail(`Replay header is not valid JSON: ${error.message}`);
    }
    const dataStart = 4 + headerLength + ((-(4 + headerLength)) & 3);
    if (dataStart > payload.length || typeof header.chunks !== "object" || header.chunks === null) {
        fail("Replay header has invalid chunk metadata");
    }

    const chunkViews = new Map();
    for (const [chunkName, chunkMetadata] of Object.entries(header.chunks)) {
        const type = CHUNK_TYPES[chunkMetadata.dtype];
        if (!type || !Array.isArray(chunkMetadata.shape)) {
            fail(`Chunk ${chunkName} has an unsupported dtype or shape`);
        }
        const elementCount = product(chunkMetadata.shape);
        if (elementCount * type.bytes !== chunkMetadata.nbytes) {
            fail(`Chunk ${chunkName} byte count does not match its shape`);
        }
        if (!Number.isInteger(chunkMetadata.offset) || chunkMetadata.offset < 0) {
            fail(`Chunk ${chunkName} has an invalid offset`);
        }
        const chunkStart = dataStart + chunkMetadata.offset;
        const chunkEnd = chunkStart + chunkMetadata.nbytes;
        if (chunkStart < dataStart || chunkEnd > payload.length) {
            fail(`Chunk ${chunkName} extends beyond the replay payload`);
        }
        chunkViews.set(chunkName, {
            name: chunkName,
            shape: chunkMetadata.shape,
            dtype: chunkMetadata.dtype,
            elementCount,
            get(flatIndex) {
                if (!Number.isInteger(flatIndex) || flatIndex < 0 || flatIndex >= elementCount) {
                    fail(`Chunk ${chunkName} index ${flatIndex} is out of bounds`);
                }
                return type.read(payload, chunkStart + flatIndex * type.bytes);
            },
        });
    }

    return {
        sourceBytes,
        payload,
        header,
        chunk(chunkName) {
            const chunkView = chunkViews.get(chunkName);
            if (!chunkView) {
                fail(`Replay is missing required chunk ${chunkName}`);
            }
            return chunkView;
        },
    };
}

function validateInteger(value, label, minimum, maximum) {
    if (!Number.isInteger(value) || value < minimum || value > maximum) {
        fail(`${label}=${value} is outside [${minimum}, ${maximum}]`);
    }
}

function readAgent(replay, frameIndex, agentIndex) {
    const { header } = replay;
    validateInteger(frameIndex, "frame index", 0, header.frames - 1);
    validateInteger(agentIndex, "agent index", 0, header.agent_cap - 1);
    const agentFloat = replay.chunk("agent_f32");
    const agentInteger = replay.chunk("agent_i32");
    if (agentFloat.shape.length !== 3 || agentInteger.shape.length !== 3) {
        fail("Agent chunks must have three dimensions");
    }
    const floatFieldCount = agentFloat.shape[2];
    const integerFieldCount = agentInteger.shape[2];
    if (floatFieldCount < 12 || integerFieldCount < 10) {
        fail("Agent chunks do not contain the expected fields");
    }
    const floatBase = (frameIndex * header.agent_cap + agentIndex) * floatFieldCount;
    const integerBase = (frameIndex * header.agent_cap + agentIndex) * integerFieldCount;
    if (agentInteger.get(integerBase + 2) !== 1) {
        return null;
    }
    return {
        index: agentIndex,
        id: agentInteger.get(integerBase),
        type: agentInteger.get(integerBase + 1),
        active: agentInteger.get(integerBase + 3) === 1,
        stopped: agentInteger.get(integerBase + 4) === 1,
        x: agentFloat.get(floatBase),
        y: agentFloat.get(floatBase + 1),
        z: agentFloat.get(floatBase + 2),
        heading: agentFloat.get(floatBase + 3),
        length: agentFloat.get(floatBase + 4),
        width: agentFloat.get(floatBase + 5),
        speed: agentFloat.get(floatBase + 6),
        acceleration: agentFloat.get(floatBase + 8),
    };
}

function agentFromSnapshot(snapshot) {
    if (!snapshot || Number(snapshot.valid) !== 1) {
        fail("Collision debug data is missing a valid agent snapshot");
    }
    return {
        index: Number(snapshot.index),
        id: Number(snapshot.index),
        type: Number(snapshot.type),
        active: Boolean(snapshot.active),
        stopped: Boolean(snapshot.stopped),
        x: Number(snapshot.x),
        y: Number(snapshot.y),
        z: Number(snapshot.z),
        heading: Number(snapshot.heading),
        length: Number(snapshot.length),
        width: Number(snapshot.width),
        speed: Math.hypot(Number(snapshot.vx), Number(snapshot.vy)),
        vx: Number(snapshot.vx),
        vy: Number(snapshot.vy),
        acceleration: 0,
    };
}

function decodeRoadGeometry(replay) {
    const roadPoints = replay.chunk("road_points");
    const roadLengths = replay.chunk("road_lengths");
    const roadTypes = replay.chunk("road_types");
    const roadElementIndices = replay.chunk("road_element_indices");
    if (roadPoints.shape.length !== 2 || roadPoints.shape[1] !== 2) {
        fail("road_points must have shape [point_count, 2]");
    }
    if (
        roadLengths.elementCount !== roadTypes.elementCount
        || roadLengths.elementCount !== roadElementIndices.elementCount
    ) {
        fail("Road metadata chunks have inconsistent lengths");
    }

    const polylines = [];
    const byElementIndex = new Map();
    let pointOffset = 0;
    for (let polylineIndex = 0; polylineIndex < roadLengths.elementCount; polylineIndex++) {
        const pointCount = roadLengths.get(polylineIndex);
        if (!Number.isInteger(pointCount) || pointCount <= 0 || pointOffset + pointCount > roadPoints.shape[0]) {
            fail(`Road polyline ${polylineIndex} has an invalid point count`);
        }
        const points = [];
        for (let pointIndex = 0; pointIndex < pointCount; pointIndex++) {
            const flatPointIndex = (pointOffset + pointIndex) * 2;
            points.push({ x: roadPoints.get(flatPointIndex), y: roadPoints.get(flatPointIndex + 1) });
        }
        pointOffset += pointCount;
        const polyline = {
            type: roadTypes.get(polylineIndex),
            elementIndex: roadElementIndices.get(polylineIndex),
            points,
        };
        polylines.push(polyline);
        byElementIndex.set(polyline.elementIndex, points);
    }
    if (pointOffset !== roadPoints.shape[0]) {
        fail("Road polyline lengths do not consume the complete road_points chunk");
    }
    return { polylines, byElementIndex };
}

function orientedBoxCorners(agent, widthExpansionMeters = 0) {
    const halfLength = agent.length / 2;
    const halfWidth = (agent.width + widthExpansionMeters) / 2;
    const cosine = Math.cos(agent.heading);
    const sine = Math.sin(agent.heading);
    const localCorners = [
        { x: halfLength, y: halfWidth },
        { x: -halfLength, y: halfWidth },
        { x: -halfLength, y: -halfWidth },
        { x: halfLength, y: -halfWidth },
    ];
    return localCorners.map((corner) => ({
        x: agent.x + corner.x * cosine - corner.y * sine,
        y: agent.y + corner.x * sine + corner.y * cosine,
    }));
}

function crossProduct(origin, first, second) {
    return (first.x - origin.x) * (second.y - origin.y) - (first.y - origin.y) * (second.x - origin.x);
}

function lineIntersection(segmentStart, segmentEnd, clipStart, clipEnd) {
    const segmentX = segmentEnd.x - segmentStart.x;
    const segmentY = segmentEnd.y - segmentStart.y;
    const clipX = clipEnd.x - clipStart.x;
    const clipY = clipEnd.y - clipStart.y;
    const denominator = segmentX * clipY - segmentY * clipX;
    if (Math.abs(denominator) < 1e-12) {
        return segmentEnd;
    }
    const fraction = ((clipStart.x - segmentStart.x) * clipY - (clipStart.y - segmentStart.y) * clipX)
        / denominator;
    return { x: segmentStart.x + fraction * segmentX, y: segmentStart.y + fraction * segmentY };
}

function intersectConvexPolygons(subjectPolygon, clipPolygon) {
    let output = subjectPolygon.slice();
    for (let clipIndex = 0; clipIndex < clipPolygon.length; clipIndex++) {
        const clipStart = clipPolygon[clipIndex];
        const clipEnd = clipPolygon[(clipIndex + 1) % clipPolygon.length];
        const input = output;
        output = [];
        if (input.length === 0) {
            break;
        }
        let segmentStart = input[input.length - 1];
        for (const segmentEnd of input) {
            const endInside = crossProduct(clipStart, clipEnd, segmentEnd) >= -1e-9;
            const startInside = crossProduct(clipStart, clipEnd, segmentStart) >= -1e-9;
            if (endInside) {
                if (!startInside) {
                    output.push(lineIntersection(segmentStart, segmentEnd, clipStart, clipEnd));
                }
                output.push(segmentEnd);
            } else if (startInside) {
                output.push(lineIntersection(segmentStart, segmentEnd, clipStart, clipEnd));
            }
            segmentStart = segmentEnd;
        }
    }
    return output;
}

function boxGapMeters(firstAgent, secondAgent) {
    if (polygonsOverlap(firstAgent, secondAgent)) {
        return 0;
    }
    let lowMeters = 0;
    let highMeters = MAXIMUM_COLLISION_TOLERANCE_METERS;
    const grown = (agent, growthMeters) => ({ ...agent, length: agent.length + growthMeters, width: agent.width + growthMeters });
    if (!polygonsOverlap(grown(firstAgent, highMeters), grown(secondAgent, highMeters))) {
        return Infinity;
    }
    for (let iteration = 0; iteration < GAP_BISECTION_ITERATIONS; iteration++) {
        const middleMeters = (lowMeters + highMeters) / 2;
        if (polygonsOverlap(grown(firstAgent, middleMeters), grown(secondAgent, middleMeters))) {
            highMeters = middleMeters;
        } else {
            lowMeters = middleMeters;
        }
    }
    return highMeters;
}

function polygonsOverlap(firstAgent, secondAgent) {
    return intersectConvexPolygons(orientedBoxCorners(firstAgent), orientedBoxCorners(secondAgent)).length >= 3;
}

function samplePolylineByDistance(points, distanceMeters) {
    if (points.length === 0) {
        fail("Cannot sample an empty path");
    }
    if (points.length === 1) {
        return {
            x: points[0].x + distanceMeters * Math.cos(points[0].heading),
            y: points[0].y + distanceMeters * Math.sin(points[0].heading),
            heading: points[0].heading,
        };
    }

    let traveledMeters = 0;
    let segmentStartHeading = points[0].heading;
    for (let pointIndex = 0; pointIndex < points.length - 1; pointIndex++) {
        const start = points[pointIndex];
        const end = points[pointIndex + 1];
        const deltaX = end.x - start.x;
        const deltaY = end.y - start.y;
        const segmentMeters = Math.hypot(deltaX, deltaY);
        if (traveledMeters + segmentMeters < distanceMeters) {
            traveledMeters += segmentMeters;
            if (segmentMeters > 1e-6) {
                segmentStartHeading = end.heading;
            }
            continue;
        }
        const fraction = segmentMeters > 1e-6
            ? Math.max(0, Math.min(1, (distanceMeters - traveledMeters) / segmentMeters))
            : 0;
        const segmentEndHeading = segmentMeters > 1e-6 ? end.heading : segmentStartHeading;
        const headingDelta = Math.atan2(
            Math.sin(segmentEndHeading - segmentStartHeading),
            Math.cos(segmentEndHeading - segmentStartHeading),
        );
        return {
            x: start.x + fraction * deltaX,
            y: start.y + fraction * deltaY,
            heading: segmentStartHeading + fraction * headingDelta,
        };
    }

    const lastIndex = points.length - 1;
    const finalStart = points[lastIndex - 1];
    const finalEnd = points[lastIndex];
    const deltaX = finalEnd.x - finalStart.x;
    const deltaY = finalEnd.y - finalStart.y;
    const finalHeading = Math.hypot(deltaX, deltaY) > 1e-6
        ? Math.atan2(deltaY, deltaX)
        : finalEnd.heading;
    const overshootMeters = Math.max(0, distanceMeters - traveledMeters);
    return {
        x: finalEnd.x + overshootMeters * Math.cos(finalHeading),
        y: finalEnd.y + overshootMeters * Math.sin(finalHeading),
        heading: finalHeading,
    };
}

function observedRail(replay, agentIndex, startFrame, collisionFrame, collisionSnapshot) {
    const points = [];
    for (let frameIndex = startFrame; frameIndex < collisionFrame; frameIndex++) {
        const agent = readAgent(replay, frameIndex, agentIndex);
        if (agent) {
            points.push({ x: agent.x, y: agent.y, heading: agent.heading });
        }
    }
    points.push({
        x: Number(collisionSnapshot.x),
        y: Number(collisionSnapshot.y),
        heading: Number(collisionSnapshot.heading),
    });
    return points;
}

function agentHistory(replay, agentIndex, endFrame, historySeconds, dtSeconds, endSnapshot = null) {
    const historyFrameCount = Math.ceil(historySeconds / dtSeconds);
    const startFrame = Math.max(0, endFrame - historyFrameCount);
    const states = [];
    for (let frameIndex = startFrame; frameIndex <= endFrame; frameIndex++) {
        const agent = frameIndex === endFrame && endSnapshot
            ? agentFromSnapshot(endSnapshot)
            : readAgent(replay, frameIndex, agentIndex);
        if (agent) {
            states.push({ frameIndex, agent });
        }
    }
    if (states.length < 2) {
        fail(`Agent ${agentIndex} has insufficient history ending at frame ${endFrame}`);
    }
    return states;
}

function velocityAt(replay, frameIndex, agentIndex, dtSeconds) {
    const neighboringFrame = frameIndex < replay.header.frames - 1 ? frameIndex + 1 : frameIndex - 1;
    const currentAgent = readAgent(replay, frameIndex, agentIndex);
    const neighboringAgent = readAgent(replay, neighboringFrame, agentIndex);
    if (!currentAgent || !neighboringAgent) {
        fail(`Cannot estimate velocity for agent ${agentIndex} at frame ${frameIndex}`);
    }
    const elapsedSeconds = (neighboringFrame - frameIndex) * dtSeconds;
    return {
        vx: (neighboringAgent.x - currentAgent.x) / elapsedSeconds,
        vy: (neighboringAgent.y - currentAgent.y) / elapsedSeconds,
    };
}

function appendUniquePoint(points, point) {
    const previous = points[points.length - 1];
    if (!previous || Math.hypot(point.x - previous.x, point.y - previous.y) > 1e-6) {
        points.push(point);
    }
}

function capturedRouteFromAgent(roadGeometry, routeElementIndices, targetAgent) {
    const routePoints = [];
    for (const routeElementIndex of routeElementIndices) {
        const elementPoints = roadGeometry.byElementIndex.get(Number(routeElementIndex)) || [];
        for (const point of elementPoints) {
            appendUniquePoint(routePoints, point);
        }
    }
    if (routePoints.length < 2) {
        return [];
    }

    let bestProjection = null;
    for (let pointIndex = 0; pointIndex < routePoints.length - 1; pointIndex++) {
        const start = routePoints[pointIndex];
        const end = routePoints[pointIndex + 1];
        const deltaX = end.x - start.x;
        const deltaY = end.y - start.y;
        const lengthSquared = deltaX * deltaX + deltaY * deltaY;
        if (lengthSquared < 1e-9) {
            continue;
        }
        const fraction = Math.max(0, Math.min(
            1,
            ((targetAgent.x - start.x) * deltaX + (targetAgent.y - start.y) * deltaY) / lengthSquared,
        ));
        const x = start.x + fraction * deltaX;
        const y = start.y + fraction * deltaY;
        const distanceSquared = (targetAgent.x - x) ** 2 + (targetAgent.y - y) ** 2;
        if (!bestProjection || distanceSquared < bestProjection.distanceSquared) {
            bestProjection = { pointIndex, x, y, distanceSquared };
        }
    }
    if (!bestProjection) {
        return [];
    }
    return [
        { x: bestProjection.x, y: bestProjection.y },
        ...routePoints.slice(bestProjection.pointIndex + 1),
    ];
}

function polylinePrefix(points, distanceMeters) {
    if (points.length === 0) {
        return [];
    }
    const prefix = [points[0]];
    let traveledMeters = 0;
    for (let pointIndex = 0; pointIndex < points.length - 1; pointIndex++) {
        const start = points[pointIndex];
        const end = points[pointIndex + 1];
        const segmentMeters = Math.hypot(end.x - start.x, end.y - start.y);
        if (traveledMeters + segmentMeters < distanceMeters) {
            prefix.push(end);
            traveledMeters += segmentMeters;
            continue;
        }
        const fraction = segmentMeters > 1e-6
            ? Math.max(0, Math.min(1, (distanceMeters - traveledMeters) / segmentMeters))
            : 0;
        prefix.push({
            x: start.x + fraction * (end.x - start.x),
            y: start.y + fraction * (end.y - start.y),
        });
        return prefix;
    }
    return prefix;
}

function straightTtcSeconds(target, targetVelocity, hitter, hitterVelocity, lateralBufferMeters, horizonSeconds, dtSeconds) {
    const stepCount = Math.floor(horizonSeconds / dtSeconds);
    for (let stepIndex = 0; stepIndex <= stepCount; stepIndex++) {
        const elapsedSeconds = stepIndex * dtSeconds;
        const projectedTarget = {
            ...target,
            x: target.x + targetVelocity.vx * elapsedSeconds,
            y: target.y + targetVelocity.vy * elapsedSeconds,
        };
        const projectedHitter = {
            ...hitter,
            x: hitter.x + hitterVelocity.vx * elapsedSeconds,
            y: hitter.y + hitterVelocity.vy * elapsedSeconds,
        };
        const overlap = intersectConvexPolygons(
            orientedBoxCorners(projectedTarget, 2 * lateralBufferMeters),
            orientedBoxCorners(projectedHitter),
        );
        if (overlap.length >= 3) {
            return elapsedSeconds;
        }
    }
    return Infinity;
}

function recomputeWarningSample(replay, detectionSamples, collisionFrame, targetIndex, hitterIndex, stepsBack, dtSeconds) {
    if (detectionSamples.length === 0) {
        fail("Replay has no stored detection samples to take the lateral buffer and danger threshold from");
    }
    const nearestSample = detectionSamples.reduce((best, sample) => (
        Math.abs(Number(sample.steps_back) - stepsBack) < Math.abs(Number(best.steps_back) - stepsBack) ? sample : best
    ));
    const lateralBufferMeters = Number(nearestSample.lateral_buffer_meters);
    const dangerThresholdSeconds = Number(nearestSample.danger_threshold_seconds);
    const detectionFrame = collisionFrame - stepsBack;
    const target = readAgent(replay, detectionFrame, targetIndex);
    const hitter = readAgent(replay, detectionFrame, hitterIndex);
    if (!target || !hitter) {
        fail(`Target or hitter is absent at the requested warning frame ${detectionFrame}`);
    }
    const ttcSeconds = straightTtcSeconds(
        target,
        velocityAt(replay, detectionFrame, targetIndex, dtSeconds),
        hitter,
        velocityAt(replay, detectionFrame, hitterIndex, dtSeconds),
        lateralBufferMeters,
        dangerThresholdSeconds,
        dtSeconds,
    );
    if (!Number.isFinite(ttcSeconds)) {
        fail(`No straight-TTC conflict within ${dangerThresholdSeconds} s at ${stepsBack} steps before collision`);
    }
    return {
        steps_back: stepsBack,
        dangerous: 1,
        danger_threshold_seconds: dangerThresholdSeconds,
        straight_ttc_seconds: ttcSeconds,
        route_ttc_seconds: Number.NaN,
        lateral_buffer_meters: lateralBufferMeters,
        recomputed_with_parameters_from_steps_back: Number(nearestSample.steps_back),
    };
}

function collectEvidence(
    replay, roadGeometry, historySeconds, requestedBrakeSeconds, requestedWarningSeconds, collisionToleranceMeters,
) {
    const { header } = replay;
    if (!Number.isInteger(header.frames) || header.frames <= 0) {
        fail("Replay header has an invalid frame count");
    }
    if (!Number.isInteger(header.agent_cap) || header.agent_cap <= 0) {
        fail("Replay header has an invalid agent capacity");
    }
    const avoidability = header.avoidability_debug;
    if (!avoidability || !avoidability.collision || !avoidability.classification) {
        fail("Replay has no avoidability collision analysis");
    }
    const classification = avoidability.classification;
    if (Number(classification.genuine_target_failure) !== 1) {
        fail(
            "Scenario is not a genuine target failure "
            + `(genuine_target_failure=${classification.genuine_target_failure ?? "missing"})`,
        );
    }
    if (Number(classification.unavoidable) === 1 || Number(classification.adversary_forced) === 1) {
        fail("Scenario classification conflicts with genuine target failure requirements");
    }

    const constants = avoidability.constants || {};
    const dtSeconds = Number(constants.dt);
    const brakingDecelerationMps2 = Number(constants.braking_deceleration);
    const tBrakeSeconds = Number(classification.t_brake);
    if (!(dtSeconds > 0) || !Number.isFinite(dtSeconds)) {
        fail(`Invalid replay dt: ${constants.dt}`);
    }
    if (!(brakingDecelerationMps2 > 0) || !Number.isFinite(brakingDecelerationMps2)) {
        fail(`Invalid braking deceleration: ${constants.braking_deceleration}`);
    }
    if (!(tBrakeSeconds > 0) || !Number.isFinite(tBrakeSeconds)) {
        fail(`Genuine failure has no positive braking time: ${classification.t_brake}`);
    }

    const collision = avoidability.collision;
    const collisionFrame = Number(collision.collision_timestep) - Number(header.init_step || 0);
    const targetIndex = Number(collision.target_agent_index);
    const hitterIndex = Number(collision.collision_adversary_index);
    validateInteger(collisionFrame, "collision frame", 0, header.frames - 1);
    validateInteger(targetIndex, "target agent index", 0, header.agent_cap - 1);
    validateInteger(hitterIndex, "hitter agent index", 0, header.agent_cap - 1);
    if (targetIndex === hitterIndex) {
        fail("Collision target and hitter indices must differ");
    }

    const targetAtCollision = agentFromSnapshot(collision.target);
    const hitterAtCollision = agentFromSnapshot(collision.adversary);
    const collisionSnapshotGapMeters = boxGapMeters(targetAtCollision, hitterAtCollision);
    if (collisionSnapshotGapMeters > collisionToleranceMeters) {
        fail(
            `Collision snapshots are ${collisionSnapshotGapMeters.toFixed(3)} m apart `
            + `(tolerance ${collisionToleranceMeters} m; see --collision-tolerance-meters)`,
        );
    }

    const candidates = Array.isArray(avoidability.candidates) ? avoidability.candidates : [];
    const avoidedCandidate = candidates.find((candidate) => {
        const candidateSeconds = Number(candidate.steps_back) * dtSeconds;
        return Number(candidate.avoided) === 1
            && Math.abs(candidateSeconds - tBrakeSeconds) <= dtSeconds / 2 + 1e-6;
    });
    if (!avoidedCandidate) {
        fail("Replay has no avoided braking candidate matching t_brake");
    }
    const renderedBrakeSeconds = requestedBrakeSeconds ?? tBrakeSeconds;
    if (renderedBrakeSeconds + dtSeconds / 2 < tBrakeSeconds) {
        fail(
            `Requested braking at ${renderedBrakeSeconds} s is later than the stored minimum `
            + `${tBrakeSeconds} s avoiding intervention`,
        );
    }
    const brakingStepsBack = Math.round(renderedBrakeSeconds / dtSeconds);
    if (Math.abs(brakingStepsBack * dtSeconds - renderedBrakeSeconds) > dtSeconds / 2 + 1e-6) {
        fail(`Requested braking time ${renderedBrakeSeconds} s cannot be represented by replay frames`);
    }
    const brakingStartFrame = collisionFrame - brakingStepsBack;
    validateInteger(brakingStartFrame, "braking start frame", 0, collisionFrame - 1);

    const detectionSamples = Array.isArray(avoidability.detection_samples)
        ? avoidability.detection_samples
        : [];
    let warningSample;
    let warningLeadSeconds;
    if (requestedWarningSeconds === null) {
        const eligibleDetectionSamples = detectionSamples
            .map((sample) => ({ sample, leadSeconds: Number(sample.steps_back) * dtSeconds }))
            .filter(({ sample, leadSeconds }) => (
                Number(sample.dangerous) === 1
                && Number.isFinite(leadSeconds)
                && leadSeconds >= 1.0 - ONE_SECOND_TOLERANCE_SECONDS
            ))
            .sort((first, second) => (
                Math.abs(first.leadSeconds - 1.0) - Math.abs(second.leadSeconds - 1.0)
            ));
        if (eligibleDetectionSamples.length === 0) {
            fail("Replay has no stored dangerous detection sample at least 1.0 second before collision");
        }
        warningSample = eligibleDetectionSamples[0].sample;
        warningLeadSeconds = eligibleDetectionSamples[0].leadSeconds;
    } else {
        const warningStepsBack = Math.round(requestedWarningSeconds / dtSeconds);
        if (Math.abs(warningStepsBack * dtSeconds - requestedWarningSeconds) > dtSeconds / 2 + 1e-6) {
            fail(`Requested warning time ${requestedWarningSeconds} s cannot be represented by replay frames`);
        }
        validateInteger(collisionFrame - warningStepsBack, "requested warning frame", 0, collisionFrame - 1);
        const storedSample = detectionSamples.find((sample) => Number(sample.steps_back) === warningStepsBack);
        if (storedSample && Number(storedSample.dangerous) !== 1) {
            fail(`Stored detection sample at ${requestedWarningSeconds} s before collision is not dangerous`);
        }
        warningSample = storedSample ?? recomputeWarningSample(
            replay, detectionSamples, collisionFrame, targetIndex, hitterIndex, warningStepsBack, dtSeconds,
        );
        warningLeadSeconds = warningStepsBack * dtSeconds;
    }
    const ttcOptions = [
        { mode: "straight", seconds: Number(warningSample.straight_ttc_seconds) },
        { mode: "route", seconds: Number(warningSample.route_ttc_seconds) },
    ].filter((option) => Number.isFinite(option.seconds) && option.seconds > 0);
    if (ttcOptions.length === 0) {
        fail("The selected detection sample has no finite, positive TTC");
    }
    ttcOptions.sort((first, second) => first.seconds - second.seconds);
    const selectedTtc = ttcOptions[0];
    const dangerThresholdSeconds = Number(warningSample.danger_threshold_seconds);
    if (!(dangerThresholdSeconds > selectedTtc.seconds)) {
        fail(
            `The selected TTC ${selectedTtc.seconds} is not below danger threshold ${dangerThresholdSeconds}`,
        );
    }
    const detectionStepsBack = Number(warningSample.steps_back);
    const detectionFrame = collisionFrame - detectionStepsBack;
    validateInteger(detectionFrame, "one-second detection frame", 0, collisionFrame - 1);

    const targetAtBrakeStart = readAgent(replay, brakingStartFrame, targetIndex);
    if (!targetAtBrakeStart) {
        fail("Target is absent at the braking start frame");
    }
    const targetBrakeRail = observedRail(
        replay,
        targetIndex,
        brakingStartFrame,
        collisionFrame,
        collision.target,
    );
    const targetStopTimeSeconds = targetAtBrakeStart.speed / brakingDecelerationMps2;
    const brakingElapsedSeconds = brakingStepsBack * dtSeconds;
    const appliedBrakingSeconds = Math.min(brakingElapsedSeconds, targetStopTimeSeconds);
    const targetDistanceAtCollision = targetAtBrakeStart.speed * appliedBrakingSeconds
        - 0.5 * brakingDecelerationMps2 * appliedBrakingSeconds ** 2;
    const targetPointAtCollision = samplePolylineByDistance(targetBrakeRail, targetDistanceAtCollision);
    const targetBrakingAtCollision = {
        ...targetAtBrakeStart,
        x: targetPointAtCollision.x,
        y: targetPointAtCollision.y,
        heading: targetPointAtCollision.heading,
        speed: Math.max(0, targetAtBrakeStart.speed - brakingDecelerationMps2 * appliedBrakingSeconds),
        acceleration: -brakingDecelerationMps2,
    };
    if (polygonsOverlap(targetBrakingAtCollision, hitterAtCollision)) {
        fail("Reconstructed avoided candidate still overlaps the original hitter at nominal impact time");
    }

    const brakingTargetPath = [];
    for (let rolloutStep = 0; rolloutStep <= brakingStepsBack; rolloutStep++) {
        const elapsedSeconds = rolloutStep * dtSeconds;
        const brakingSeconds = Math.min(elapsedSeconds, targetStopTimeSeconds);
        const distanceMeters = targetAtBrakeStart.speed * brakingSeconds
            - 0.5 * brakingDecelerationMps2 * brakingSeconds ** 2;
        brakingTargetPath.push(samplePolylineByDistance(targetBrakeRail, distanceMeters));
    }
    const observedTargetPathFromBrake = observedRail(
        replay,
        targetIndex,
        brakingStartFrame,
        collisionFrame,
        collision.target,
    );
    const hitterPathFromBrake = observedRail(
        replay,
        hitterIndex,
        brakingStartFrame,
        collisionFrame,
        collision.adversary,
    );

    const targetAtDetection = readAgent(replay, detectionFrame, targetIndex);
    const hitterAtDetection = readAgent(replay, detectionFrame, hitterIndex);
    if (!targetAtDetection || !hitterAtDetection) {
        fail("Target or hitter is absent at the one-second detection frame");
    }
    const targetVelocity = velocityAt(replay, detectionFrame, targetIndex, dtSeconds);
    const hitterVelocity = velocityAt(replay, detectionFrame, hitterIndex, dtSeconds);
    let targetProjectionPath;
    if (selectedTtc.mode === "route") {
        const route = capturedRouteFromAgent(
            roadGeometry,
            avoidability.target_route_lane_indices || [],
            targetAtDetection,
        );
        targetProjectionPath = polylinePrefix(route, targetAtDetection.speed * selectedTtc.seconds);
        if (targetProjectionPath.length < 2) {
            fail("The selected route TTC has no usable target route geometry");
        }
    } else {
        targetProjectionPath = [
            { x: targetAtDetection.x, y: targetAtDetection.y },
            {
                x: targetAtDetection.x + targetVelocity.vx * selectedTtc.seconds,
                y: targetAtDetection.y + targetVelocity.vy * selectedTtc.seconds,
            },
        ];
    }
    const targetProjectionEnd = targetProjectionPath[targetProjectionPath.length - 1];
    const hitterProjectionPath = [
        { x: hitterAtDetection.x, y: hitterAtDetection.y },
        {
            x: hitterAtDetection.x + hitterVelocity.vx * selectedTtc.seconds,
            y: hitterAtDetection.y + hitterVelocity.vy * selectedTtc.seconds,
        },
    ];
    const hitterProjectionEnd = hitterProjectionPath[1];
    const projectedTarget = { ...targetAtDetection, ...targetProjectionEnd };
    const projectedHitter = { ...hitterAtDetection, ...hitterProjectionEnd };
    const lateralBufferMeters = Number(warningSample.lateral_buffer_meters);
    if (!(lateralBufferMeters >= 0) || !Number.isFinite(lateralBufferMeters)) {
        fail(`Invalid lateral safety buffer: ${warningSample.lateral_buffer_meters}`);
    }
    const projectedOverlap = intersectConvexPolygons(
        orientedBoxCorners(projectedTarget, 2 * lateralBufferMeters),
        orientedBoxCorners(projectedHitter),
    );

    const recordedApproachStartFrame = detectionFrame;
    const observedTargetApproach = observedRail(
        replay,
        targetIndex,
        recordedApproachStartFrame,
        collisionFrame,
        collision.target,
    );
    const observedHitterApproach = observedRail(
        replay,
        hitterIndex,
        recordedApproachStartFrame,
        collisionFrame,
        collision.adversary,
    );
    const targetHistoryToCollision = agentHistory(
        replay,
        targetIndex,
        collisionFrame,
        historySeconds,
        dtSeconds,
        collision.target,
    );
    const hitterHistoryToCollision = agentHistory(
        replay,
        hitterIndex,
        collisionFrame,
        historySeconds,
        dtSeconds,
        collision.adversary,
    );
    const targetHistoryToDetection = agentHistory(
        replay,
        targetIndex,
        detectionFrame,
        historySeconds,
        dtSeconds,
    );
    const hitterHistoryToDetection = agentHistory(
        replay,
        hitterIndex,
        detectionFrame,
        historySeconds,
        dtSeconds,
    );

    return {
        avoidability,
        classification,
        constants,
        collision,
        collisionFrame,
        collisionSnapshotGapMeters,
        collisionToleranceMeters,
        targetIndex,
        hitterIndex,
        targetAtCollision,
        hitterAtCollision,
        collisionOverlap: intersectConvexPolygons(
            orientedBoxCorners(targetAtCollision),
            orientedBoxCorners(hitterAtCollision),
        ),
        tBrakeSeconds,
        renderedBrakeSeconds: brakingStepsBack * dtSeconds,
        usesStoredMinimumBrake: brakingStepsBack === Number(avoidedCandidate.steps_back),
        brakingDecelerationMps2,
        brakingStepsBack,
        brakingStartFrame,
        targetAtBrakeStart,
        targetBrakingAtCollision,
        brakingTargetPath,
        observedTargetPathFromBrake,
        hitterPathFromBrake,
        warningSample,
        warningLeadSeconds,
        detectionFrame,
        targetAtDetection,
        hitterAtDetection,
        selectedTtc,
        dangerThresholdSeconds,
        lateralBufferMeters,
        targetProjectionPath,
        hitterProjectionPath,
        projectedTarget,
        projectedHitter,
        projectedOverlap,
        observedTargetApproach,
        observedHitterApproach,
        historySeconds,
        targetHistoryToCollision,
        hitterHistoryToCollision,
        targetHistoryToDetection,
        hitterHistoryToDetection,
    };
}

function safeScenarioName(sourcePath) {
    const basename = path.basename(sourcePath);
    const scenarioMatch = basename.match(/^(scenario_\d+)/);
    if (scenarioMatch) {
        return scenarioMatch[1];
    }
    return basename.replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "replay";
}

function serializeAgent(agent) {
    return {
        index: agent.index,
        x: agent.x,
        y: agent.y,
        heading: agent.heading,
        length: agent.length,
        width: agent.width,
    };
}

function buildFigureEvidence(replay, roadGeometry, evidence) {
    const dtSeconds = Number(evidence.constants.dt);
    const historyFrameCount = Math.ceil(evidence.historySeconds / dtSeconds);
    const firstFrame = Math.max(0, Math.min(evidence.detectionFrame - historyFrameCount, evidence.brakingStartFrame));
    const collisionSnapshots = new Map([
        [evidence.targetIndex, evidence.targetAtCollision],
        [evidence.hitterIndex, evidence.hitterAtCollision],
    ]);
    const frames = [];
    for (let frameIndex = firstFrame; frameIndex <= evidence.collisionFrame; frameIndex++) {
        const agents = [];
        for (let agentIndex = 0; agentIndex < replay.header.agent_cap; agentIndex++) {
            const agent = frameIndex === evidence.collisionFrame && collisionSnapshots.has(agentIndex)
                ? { ...collisionSnapshots.get(agentIndex), index: agentIndex }
                : readAgent(replay, frameIndex, agentIndex);
            if (agent) {
                agents.push(serializeAgent(agent));
            }
        }
        frames.push(agents);
    }
    return {
        schema: EVIDENCE_SCHEMA,
        version: EVIDENCE_VERSION,
        dt_seconds: dtSeconds,
        history_frame_count: historyFrameCount,
        target_index: evidence.targetIndex,
        hitter_index: evidence.hitterIndex,
        first_frame: firstFrame,
        collision_frame: evidence.collisionFrame,
        detection_frame: evidence.detectionFrame,
        braking_start_frame: evidence.brakingStartFrame,
        roads: roadGeometry.polylines.map((polyline) => ({
            style: ROAD_STYLE_BY_TYPE[polyline.type] ?? FALLBACK_ROAD_STYLE,
            x: polyline.points.map((point) => point.x),
            y: polyline.points.map((point) => point.y),
        })),
        frames,
        braking_target_states: evidence.brakingTargetPath.map((point) => ({
            x: point.x,
            y: point.y,
            heading: point.heading,
        })),
        prediction: {
            ttc_mode: evidence.selectedTtc.mode,
            ttc_seconds: evidence.selectedTtc.seconds,
            lateral_buffer_meters: evidence.lateralBufferMeters,
            target_path: evidence.targetProjectionPath.map((point) => ({ x: point.x, y: point.y })),
            hitter_path: evidence.hitterProjectionPath.map((point) => ({ x: point.x, y: point.y })),
            projected_target: serializeAgent(evidence.projectedTarget),
            projected_hitter: serializeAgent(evidence.projectedHitter),
            overlap_polygon: evidence.projectedOverlap,
        },
    };
}

function runRenderer(evidencePath, outputDirectory) {
    const pythonExecutable = process.env.PYTHON || "python3";
    const result = spawnSync(pythonExecutable, [RENDERER_SCRIPT, evidencePath, outputDirectory], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "pipe"],
    });
    if (result.error || result.status !== 0) {
        fail(`Renderer failed: ${(result.error?.message || result.stderr || `exit status ${result.status}`).trim()}`);
    }
    return result.stdout.split("\n").filter((line) => line.length > 0);
}

function writeOutputs(sourcePath, outputDirectory, replay, figureEvidence, evidence, evidenceOnly) {
    fs.mkdirSync(outputDirectory, { recursive: true });
    const evidencePath = path.join(outputDirectory, "figure_evidence.json");
    fs.writeFileSync(evidencePath, `${JSON.stringify(figureEvidence)}\n`);
    const writtenFiles = [evidencePath];
    if (!evidenceOnly) {
        writtenFiles.push(...runRenderer(evidencePath, outputDirectory));
    }

    const metadata = {
        schema_version: 2,
        source_path: path.resolve(sourcePath),
        source_sha256: crypto.createHash("sha256").update(replay.sourceBytes).digest("hex"),
        scenario_id: replay.header.scenario_id,
        map_name: replay.header.map_name,
        collision_timestep: evidence.collision.collision_timestep,
        collision_frame: evidence.collisionFrame,
        dt_seconds: Number(evidence.constants.dt),
        target_agent_index: evidence.targetIndex,
        collision_adversary_index: evidence.hitterIndex,
        collision_snapshot_gap_meters: evidence.collisionSnapshotGapMeters,
        collision_tolerance_meters: evidence.collisionToleranceMeters,
        classification: {
            genuine_target_failure: 1,
            adversary_forced: 0,
            unavoidable: 0,
        },
        braking_counterfactual: {
            rendered_seconds_before_collision: evidence.renderedBrakeSeconds,
            minimum_avoiding_seconds_before_collision: evidence.tBrakeSeconds,
            uses_stored_minimum_candidate: evidence.usesStoredMinimumBrake,
            deceleration_mps2: evidence.brakingDecelerationMps2,
            avoided: true,
        },
        early_warning: {
            seconds_before_collision: evidence.warningLeadSeconds,
            steps_back: Number(evidence.warningSample.steps_back),
            dangerous: true,
            selected_ttc_mode: evidence.selectedTtc.mode,
            selected_ttc_seconds: evidence.selectedTtc.seconds,
            straight_ttc_seconds: Number(evidence.warningSample.straight_ttc_seconds),
            route_ttc_seconds: Number(evidence.warningSample.route_ttc_seconds),
            danger_threshold_seconds: evidence.dangerThresholdSeconds,
            lateral_buffer_meters: evidence.lateralBufferMeters,
            recomputed_with_parameters_from_steps_back:
                evidence.warningSample.recomputed_with_parameters_from_steps_back ?? null,
        },
        rendering: {
            method: "figure_evidence.json rendered by scripts/render_paper_figure_panels.py (matplotlib)",
            history_seconds: evidence.historySeconds,
        },
        outputs: writtenFiles.map((filePath) => path.basename(filePath)),
    };
    const metadataPath = path.join(outputDirectory, "figure_metadata.json");
    fs.writeFileSync(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`);
    writtenFiles.push(metadataPath);
    return writtenFiles;
}

function main() {
    const {
        positionalArguments, evidenceOnly, historySeconds, brakeSeconds, warningSeconds, collisionToleranceMeters,
    } = parseCliArguments(
        process.argv.slice(2),
    );
    const sourceArgument = positionalArguments[0];
    if (!sourceArgument || sourceArgument === "--help" || sourceArgument === "-h") {
        process.stdout.write(`${usage()}\n`);
        process.exit(sourceArgument ? 0 : 2);
    }
    const sourcePath = path.resolve(sourceArgument);
    if (!fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isFile()) {
        fail(`Replay source does not exist or is not a file: ${sourcePath}`);
    }
    const scenarioName = safeScenarioName(sourcePath);
    const outputDirectory = path.resolve(
        positionalArguments[1] || path.join("paper_figures", "regents_failure", scenarioName),
    );

    const replay = decodeReplay(sourcePath);
    const roadGeometry = decodeRoadGeometry(replay);
    const evidence = collectEvidence(
        replay, roadGeometry, historySeconds, brakeSeconds, warningSeconds, collisionToleranceMeters,
    );
    const figureEvidence = buildFigureEvidence(replay, roadGeometry, evidence);
    const writtenFiles = writeOutputs(sourcePath, outputDirectory, replay, figureEvidence, evidence, evidenceOnly);
    process.stdout.write(`Validated genuine failure ${replay.header.scenario_id}\n`);
    process.stdout.write(`Wrote ${writtenFiles.length} files to ${outputDirectory}\n`);
    for (const filePath of writtenFiles) {
        process.stdout.write(`  ${filePath}\n`);
    }
    if (evidenceOnly) {
        process.stdout.write(`Rendering skipped by --evidence-only; run: python3 ${RENDERER_SCRIPT} ${outputDirectory}/figure_evidence.json\n`);
    }
}

try {
    main();
} catch (error) {
    process.stderr.write(`error: ${error.message}\n`);
    process.exit(1);
}
