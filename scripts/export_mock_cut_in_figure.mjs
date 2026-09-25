#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const DT_SECONDS = 0.1;
const SIMULATION_SECONDS = 10;
const HISTORY_SECONDS = 4.5;
const WARNING_LEAD_SECONDS = 1.5;
const BRAKING_DECELERATION_MPS2 = 5;
const POST_IMPACT_CHECK_SECONDS = 2;
const TTC_HORIZON_SECONDS = 4;
const TTC_STEP_SECONDS = 0.05;
const DANGER_THRESHOLD_SECONDS = 2;
const LATERAL_BUFFER_METERS = 0.2;

const LANE_WIDTH_METERS = 3.6;
const LANE_COUNT = 3;
const EGO_LANE_INDEX = 1;
const VEHICLE_LENGTH_METERS = 4.6;
const VEHICLE_WIDTH_METERS = 1.9;

const TARGET_START_X_METERS = -28;
const TARGET_SPEED_MPS = 14;
const HITTER_START_X_METERS = 2;
const HITTER_SPEED_MPS = 10;
const CUT_IN_START_SECONDS = 4.0;
const CUT_IN_DURATION_SECONDS = 3.2;
const CONTEXT_VEHICLES = Object.freeze([
    { startX: -24, laneIndex: 2, speedMps: 10 },
    { startX: 23, laneIndex: 2, speedMps: 10.5 },
    { startX: -20, laneIndex: 0, speedMps: 13 },
]);

const TARGET_AGENT_INDEX = 0;
const HITTER_AGENT_INDEX = 1;
const ROAD_START_X_METERS = -80;
const ROAD_END_X_METERS = 160;
const EVIDENCE_SCHEMA = "paper_figure_evidence";
const EVIDENCE_VERSION = 1;
const RENDERER_SCRIPT = path.join(path.dirname(fileURLToPath(import.meta.url)), "render_paper_figure_panels.py");

function fail(message) {
    throw new Error(message);
}

function formatNumber(value, fractionDigits = 1) {
    return Number(value).toFixed(fractionDigits).replace(/\.0+$/, "");
}

function laneCenterY(laneIndex) {
    return (laneIndex - EGO_LANE_INDEX) * LANE_WIDTH_METERS;
}

function orientedBoxCorners(vehicle, widthExpansionMeters = 0) {
    const halfLength = VEHICLE_LENGTH_METERS / 2;
    const halfWidth = VEHICLE_WIDTH_METERS / 2 + widthExpansionMeters / 2;
    const cosHeading = Math.cos(vehicle.heading);
    const sinHeading = Math.sin(vehicle.heading);
    return [
        [halfLength, halfWidth], [halfLength, -halfWidth], [-halfLength, -halfWidth], [-halfLength, halfWidth],
    ].map(([longitudinal, lateral]) => ({
        x: vehicle.x + longitudinal * cosHeading - lateral * sinHeading,
        y: vehicle.y + longitudinal * sinHeading + lateral * cosHeading,
    }));
}

function polygonsOverlap(firstPolygon, secondPolygon) {
    for (const polygon of [firstPolygon, secondPolygon]) {
        for (let edgeIndex = 0; edgeIndex < polygon.length; edgeIndex++) {
            const start = polygon[edgeIndex];
            const end = polygon[(edgeIndex + 1) % polygon.length];
            const axis = { x: start.y - end.y, y: end.x - start.x };
            const firstProjections = firstPolygon.map((point) => point.x * axis.x + point.y * axis.y);
            const secondProjections = secondPolygon.map((point) => point.x * axis.x + point.y * axis.y);
            if (Math.max(...firstProjections) <= Math.min(...secondProjections)
                || Math.max(...secondProjections) <= Math.min(...firstProjections)) {
                return false;
            }
        }
    }
    return true;
}

function clipPolygon(subjectPolygon, clipPolygonPoints) {
    let output = subjectPolygon;
    for (let edgeIndex = 0; edgeIndex < clipPolygonPoints.length; edgeIndex++) {
        const edgeStart = clipPolygonPoints[edgeIndex];
        const edgeEnd = clipPolygonPoints[(edgeIndex + 1) % clipPolygonPoints.length];
        const inside = (point) => (edgeEnd.x - edgeStart.x) * (point.y - edgeStart.y)
            - (edgeEnd.y - edgeStart.y) * (point.x - edgeStart.x) >= 0;
        const intersect = (first, second) => {
            const edgeDx = edgeEnd.x - edgeStart.x;
            const edgeDy = edgeEnd.y - edgeStart.y;
            const segmentDx = second.x - first.x;
            const segmentDy = second.y - first.y;
            const ratio = (edgeDx * (first.y - edgeStart.y) - edgeDy * (first.x - edgeStart.x))
                / (edgeDy * segmentDx - edgeDx * segmentDy);
            return { x: first.x + ratio * segmentDx, y: first.y + ratio * segmentDy };
        };
        const input = output;
        output = [];
        for (let pointIndex = 0; pointIndex < input.length; pointIndex++) {
            const current = input[pointIndex];
            const previous = input[(pointIndex + input.length - 1) % input.length];
            if (inside(current)) {
                if (!inside(previous)) {
                    output.push(intersect(previous, current));
                }
                output.push(current);
            } else if (inside(previous)) {
                output.push(intersect(previous, current));
            }
        }
    }
    return output;
}

function smoothstepProgress(timeSeconds) {
    const progress = Math.min(1, Math.max(0, (timeSeconds - CUT_IN_START_SECONDS) / CUT_IN_DURATION_SECONDS));
    return progress * progress * (3 - 2 * progress);
}

function smoothstepRate(timeSeconds) {
    const progress = (timeSeconds - CUT_IN_START_SECONDS) / CUT_IN_DURATION_SECONDS;
    if (progress <= 0 || progress >= 1) {
        return 0;
    }
    return 6 * progress * (1 - progress) / CUT_IN_DURATION_SECONDS;
}

function hitterState(timeSeconds) {
    const lateralSpanMeters = laneCenterY(EGO_LANE_INDEX + 1) - laneCenterY(EGO_LANE_INDEX);
    const lateralVelocityMps = -lateralSpanMeters * smoothstepRate(timeSeconds);
    return {
        x: HITTER_START_X_METERS + HITTER_SPEED_MPS * timeSeconds,
        y: laneCenterY(EGO_LANE_INDEX + 1) - lateralSpanMeters * smoothstepProgress(timeSeconds),
        heading: Math.atan2(lateralVelocityMps, HITTER_SPEED_MPS),
        vx: HITTER_SPEED_MPS,
        vy: lateralVelocityMps,
    };
}

function targetState(timeSeconds, brakeStartSeconds = Infinity) {
    const cruiseSeconds = Math.min(timeSeconds, brakeStartSeconds);
    const brakingSeconds = Math.min(
        Math.max(0, timeSeconds - brakeStartSeconds),
        TARGET_SPEED_MPS / BRAKING_DECELERATION_MPS2,
    );
    const speedMps = TARGET_SPEED_MPS - BRAKING_DECELERATION_MPS2 * brakingSeconds;
    return {
        x: TARGET_START_X_METERS + TARGET_SPEED_MPS * (cruiseSeconds + brakingSeconds)
            - 0.5 * BRAKING_DECELERATION_MPS2 * brakingSeconds * brakingSeconds,
        y: laneCenterY(EGO_LANE_INDEX),
        heading: 0,
        vx: speedMps,
        vy: 0,
    };
}

function contextStates(timeSeconds) {
    return CONTEXT_VEHICLES.map((vehicle) => ({
        x: vehicle.startX + vehicle.speedMps * timeSeconds,
        y: laneCenterY(vehicle.laneIndex),
        heading: 0,
    }));
}

function straightProjection(vehicle, horizonSeconds) {
    return {
        x: vehicle.x + vehicle.vx * horizonSeconds,
        y: vehicle.y + vehicle.vy * horizonSeconds,
        heading: vehicle.heading,
    };
}

function straightTtcSeconds(target, hitter) {
    const stepCount = Math.round(TTC_HORIZON_SECONDS / TTC_STEP_SECONDS);
    for (let stepIndex = 0; stepIndex <= stepCount; stepIndex++) {
        const horizonSeconds = stepIndex * TTC_STEP_SECONDS;
        const projectedTarget = orientedBoxCorners(straightProjection(target, horizonSeconds), LATERAL_BUFFER_METERS * 2);
        const projectedHitter = orientedBoxCorners(straightProjection(hitter, horizonSeconds));
        if (polygonsOverlap(projectedTarget, projectedHitter)) {
            return horizonSeconds;
        }
    }
    return Infinity;
}

function vehiclesOverlap(firstVehicle, secondVehicle) {
    return polygonsOverlap(orientedBoxCorners(firstVehicle), orientedBoxCorners(secondVehicle));
}

function buildEvidence() {
    const frameCount = Math.round(SIMULATION_SECONDS / DT_SECONDS);
    let impactFrame = -1;
    for (let frameIndex = 0; frameIndex <= frameCount; frameIndex++) {
        if (vehiclesOverlap(targetState(frameIndex * DT_SECONDS), hitterState(frameIndex * DT_SECONDS))) {
            impactFrame = frameIndex;
            break;
        }
    }
    if (impactFrame < 0) {
        fail("Mock scenario produced no collision");
    }
    const impactSeconds = impactFrame * DT_SECONDS;
    const historyFrameCount = Math.round(HISTORY_SECONDS / DT_SECONDS);
    if (impactFrame < historyFrameCount) {
        fail(`Impact at ${impactSeconds} s leaves less than ${HISTORY_SECONDS} s of history`);
    }
    const detectionFrame = impactFrame - Math.round(WARNING_LEAD_SECONDS / DT_SECONDS);
    const detectionSeconds = detectionFrame * DT_SECONDS;
    for (let frameIndex = 0; frameIndex <= frameCount; frameIndex++) {
        for (const context of contextStates(frameIndex * DT_SECONDS)) {
            if (vehiclesOverlap(context, targetState(frameIndex * DT_SECONDS))
                || vehiclesOverlap(context, hitterState(frameIndex * DT_SECONDS))) {
                fail(`Context vehicle overlaps a key vehicle at frame ${frameIndex}`);
            }
        }
    }

    const postImpactFrame = impactFrame + Math.round(POST_IMPACT_CHECK_SECONDS / DT_SECONDS);
    let minimumBrakingGapMeters = Infinity;
    for (let frameIndex = detectionFrame; frameIndex <= postImpactFrame; frameIndex++) {
        const timeSeconds = frameIndex * DT_SECONDS;
        const brakingTarget = targetState(timeSeconds, detectionSeconds);
        const hitter = hitterState(timeSeconds);
        if (vehiclesOverlap(brakingTarget, hitter)) {
            fail(`Braking from t = -${WARNING_LEAD_SECONDS} s still collides at frame ${frameIndex}`);
        }
        const gapMeters = hitter.x - brakingTarget.x - VEHICLE_LENGTH_METERS;
        minimumBrakingGapMeters = Math.min(minimumBrakingGapMeters, gapMeters);
    }

    const targetAtDetection = targetState(detectionSeconds);
    const hitterAtDetection = hitterState(detectionSeconds);
    const ttcSeconds = straightTtcSeconds(targetAtDetection, hitterAtDetection);
    if (!(ttcSeconds < DANGER_THRESHOLD_SECONDS)) {
        fail(`Straight TTC ${ttcSeconds} s at detection is not below ${DANGER_THRESHOLD_SECONDS} s`);
    }
    const projectedTarget = straightProjection(targetAtDetection, ttcSeconds);
    const projectedHitter = straightProjection(hitterAtDetection, ttcSeconds);
    const projectedOverlap = clipPolygon(
        orientedBoxCorners(projectedHitter),
        orientedBoxCorners(projectedTarget, LATERAL_BUFFER_METERS * 2).reverse(),
    );
    if (projectedOverlap.length < 3) {
        fail("Projected footprints do not overlap at the reported TTC");
    }

    const firstFrame = Math.max(0, detectionFrame - historyFrameCount);
    const frames = [];
    for (let frameIndex = firstFrame; frameIndex <= impactFrame; frameIndex++) {
        const timeSeconds = frameIndex * DT_SECONDS;
        const vehicles = [targetState(timeSeconds), hitterState(timeSeconds), ...contextStates(timeSeconds)];
        frames.push(vehicles.map((vehicle, vehicleIndex) => serializeVehicle(vehicle, vehicleIndex)));
    }
    const brakingTargetStates = [];
    for (let frameIndex = detectionFrame; frameIndex <= impactFrame; frameIndex++) {
        const { x, y, heading } = targetState(frameIndex * DT_SECONDS, detectionSeconds);
        brakingTargetStates.push({ x, y, heading });
    }
    const figureEvidence = {
        schema: EVIDENCE_SCHEMA,
        version: EVIDENCE_VERSION,
        dt_seconds: DT_SECONDS,
        history_frame_count: historyFrameCount,
        target_index: TARGET_AGENT_INDEX,
        hitter_index: HITTER_AGENT_INDEX,
        first_frame: firstFrame,
        collision_frame: impactFrame,
        detection_frame: detectionFrame,
        braking_start_frame: detectionFrame,
        roads: roadPolylines(),
        frames,
        braking_target_states: brakingTargetStates,
        prediction: {
            ttc_mode: "straight",
            ttc_seconds: ttcSeconds,
            lateral_buffer_meters: LATERAL_BUFFER_METERS,
            target_path: [targetAtDetection, projectedTarget].map(({ x, y }) => ({ x, y })),
            hitter_path: [hitterAtDetection, projectedHitter].map(({ x, y }) => ({ x, y })),
            projected_target: serializeVehicle(projectedTarget, TARGET_AGENT_INDEX),
            projected_hitter: serializeVehicle(projectedHitter, HITTER_AGENT_INDEX),
            overlap_polygon: projectedOverlap,
        },
    };
    return { figureEvidence, impactSeconds, ttcSeconds, minimumBrakingGapMeters };
}

function serializeVehicle(vehicle, vehicleIndex) {
    return {
        index: vehicleIndex,
        x: vehicle.x,
        y: vehicle.y,
        heading: vehicle.heading,
        length: VEHICLE_LENGTH_METERS,
        width: VEHICLE_WIDTH_METERS,
    };
}

function roadPolylines() {
    const straightLine = (style, y) => ({ style, x: [ROAD_START_X_METERS, ROAD_END_X_METERS], y: [y, y] });
    const roads = [];
    for (let laneIndex = 0; laneIndex < LANE_COUNT; laneIndex++) {
        roads.push(straightLine("lane", laneCenterY(laneIndex)));
    }
    for (let laneIndex = 1; laneIndex < LANE_COUNT; laneIndex++) {
        roads.push(straightLine("road_line", laneCenterY(laneIndex) - LANE_WIDTH_METERS / 2));
    }
    roads.push(straightLine("edge", laneCenterY(0) - LANE_WIDTH_METERS / 2));
    roads.push(straightLine("edge", laneCenterY(LANE_COUNT - 1) + LANE_WIDTH_METERS / 2));
    return roads;
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

function main() {
    const evidenceOnly = process.argv.includes("--evidence-only");
    const outputArgument = process.argv.slice(2).find((argument) => argument !== "--evidence-only");
    const outputDirectory = path.resolve(outputArgument || path.join("paper_figures", "mock_cut_in"));
    fs.mkdirSync(outputDirectory, { recursive: true });
    const { figureEvidence, impactSeconds, ttcSeconds, minimumBrakingGapMeters } = buildEvidence();
    const evidencePath = path.join(outputDirectory, "figure_evidence.json");
    fs.writeFileSync(evidencePath, `${JSON.stringify(figureEvidence)}\n`);
    const writtenFiles = [evidencePath, ...(evidenceOnly ? [] : runRenderer(evidencePath, outputDirectory))];
    const metadata = {
        schema_version: 2,
        provenance: "synthetic mock; analytic kinematics, not replay or simulator data",
        scenario: "three-lane road; slower vehicle ahead in the adjacent lane cuts into the target lane",
        dt_seconds: DT_SECONDS,
        vehicle_size_meters: [VEHICLE_LENGTH_METERS, VEHICLE_WIDTH_METERS],
        target_speed_mps: TARGET_SPEED_MPS,
        hitter_speed_mps: HITTER_SPEED_MPS,
        cut_in_duration_seconds: CUT_IN_DURATION_SECONDS,
        collision_time_seconds: impactSeconds,
        braking_counterfactual: {
            seconds_before_collision: WARNING_LEAD_SECONDS,
            deceleration_mps2: BRAKING_DECELERATION_MPS2,
            avoided: true,
            minimum_gap_meters: Number(minimumBrakingGapMeters.toFixed(3)),
        },
        early_warning: {
            seconds_before_collision: WARNING_LEAD_SECONDS,
            selected_ttc_mode: "straight",
            selected_ttc_seconds: ttcSeconds,
            danger_threshold_seconds: DANGER_THRESHOLD_SECONDS,
            lateral_buffer_meters: LATERAL_BUFFER_METERS,
        },
        rendering: {
            method: "figure_evidence.json rendered by scripts/render_paper_figure_panels.py (matplotlib)",
            history_seconds: HISTORY_SECONDS,
        },
        outputs: writtenFiles.map((filePath) => path.basename(filePath)),
    };
    fs.writeFileSync(path.join(outputDirectory, "figure_metadata.json"), `${JSON.stringify(metadata, null, 2)}\n`);
    process.stdout.write(`Collision at t=${formatNumber(impactSeconds)} s, TTC at warning `
        + `${formatNumber(ttcSeconds, 2)} s, braking min gap ${formatNumber(minimumBrakingGapMeters, 2)} m\n`);
    process.stdout.write(`Wrote mock cut-in evidence${evidenceOnly ? "" : " and panels"} to ${outputDirectory}\n`);
}

try {
    main();
} catch (error) {
    process.stderr.write(`error: ${error.message}\n`);
    process.exit(1);
}
