#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const DT_SECONDS = 0.1;
const SIMULATION_SECONDS = 10;
const DEFAULT_HISTORY_SECONDS = 4.5;
const BRAKE_MARGIN_SECONDS = 0.2;
const WARNING_TO_BRAKE_SECONDS = 0.8;
const BRAKE_LEAD_SEARCH_MAX_SECONDS = 4;
const BRAKING_DECELERATION_MPS2 = 5;
const POST_IMPACT_CHECK_SECONDS = 2;
const REACTION_TIME_SECONDS = 1.0;
const DANGER_TTC_MARGIN_SECONDS = 0.1;
const DANGER_TTC_MAX_PROJECTION_STEPS = 92;
const LATERAL_BUFFER_BASE_METERS = 0.2;
const LATERAL_BUFFER_RESPONSE_TIME_SECONDS = 0.0;
const LATERAL_BUFFER_DECELERATION_MPS2 = 0.8;
const LATERAL_BUFFER_MAX_METERS = 2.0;

const LANE_WIDTH_METERS = 3.6;
const LANE_COUNT = 3;
const EGO_LANE_INDEX = 1;
const VEHICLE_LENGTH_METERS = 4.6;
const VEHICLE_WIDTH_METERS = 1.9;

const KMH_PER_MPS = 3.6;
const MOCK_PRESETS = Object.freeze({
    cut_in: {
        outputName: "mock_cut_in",
        description: "three-lane road; slower vehicle ahead in the adjacent lane drifts into the target lane",
        targetStartX: -28,
        targetSpeedMps: 14,
        hitterStartX: 2,
        hitterSpeedMps: 10,
        cutInStartSeconds: 4.0,
        cutInDurationSeconds: 3.2,
        contextVehicles: [
            { startX: -24, laneIndex: 2, speedMps: 10 },
            { startX: 23, laneIndex: 2, speedMps: 10.5 },
            { startX: -20, laneIndex: 0, speedMps: 13 },
        ],
    },
    cut_in_2: {
        outputName: "mock_cut_in_2",
        description: "three-lane road; target at 30 km/h, vehicle at 10 km/h in the adjacent lane cuts in ahead of it",
        targetStartX: -30,
        targetSpeedMps: 30 / KMH_PER_MPS,
        hitterStartX: -30 + (30 / KMH_PER_MPS) * 4.0 + 24 - (10 / KMH_PER_MPS) * 4.0,
        hitterSpeedMps: 10 / KMH_PER_MPS,
        cutInStartSeconds: 4.0,
        cutInDurationSeconds: 5.0,
        historySeconds: 7.0,
        contextVehicles: [],
    },
});
const DEFAULT_PRESET = "cut_in";

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

function smoothstepProgress(scenario, timeSeconds) {
    const progress = Math.min(1, Math.max(0, (timeSeconds - scenario.cutInStartSeconds) / scenario.cutInDurationSeconds));
    return progress * progress * (3 - 2 * progress);
}

function smoothstepRate(scenario, timeSeconds) {
    const progress = (timeSeconds - scenario.cutInStartSeconds) / scenario.cutInDurationSeconds;
    if (progress <= 0 || progress >= 1) {
        return 0;
    }
    return 6 * progress * (1 - progress) / scenario.cutInDurationSeconds;
}

function hitterState(scenario, timeSeconds) {
    const lateralSpanMeters = laneCenterY(EGO_LANE_INDEX + 1) - laneCenterY(EGO_LANE_INDEX);
    const lateralVelocityMps = -lateralSpanMeters * smoothstepRate(scenario, timeSeconds);
    return {
        x: scenario.hitterStartX + scenario.hitterSpeedMps * timeSeconds,
        y: laneCenterY(EGO_LANE_INDEX + 1) - lateralSpanMeters * smoothstepProgress(scenario, timeSeconds),
        heading: Math.atan2(lateralVelocityMps, scenario.hitterSpeedMps),
        vx: scenario.hitterSpeedMps,
        vy: lateralVelocityMps,
    };
}

function targetState(scenario, timeSeconds, brakeStartSeconds = Infinity) {
    const cruiseSeconds = Math.min(timeSeconds, brakeStartSeconds);
    const brakingSeconds = Math.min(
        Math.max(0, timeSeconds - brakeStartSeconds),
        scenario.targetSpeedMps / BRAKING_DECELERATION_MPS2,
    );
    const speedMps = scenario.targetSpeedMps - BRAKING_DECELERATION_MPS2 * brakingSeconds;
    return {
        x: scenario.targetStartX + scenario.targetSpeedMps * (cruiseSeconds + brakingSeconds)
            - 0.5 * BRAKING_DECELERATION_MPS2 * brakingSeconds * brakingSeconds,
        y: laneCenterY(EGO_LANE_INDEX),
        heading: 0,
        vx: speedMps,
        vy: 0,
    };
}

function contextStates(scenario, timeSeconds) {
    return scenario.contextVehicles.map((vehicle) => ({
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

function lateralSafetyBufferMeters(target, hitter) {
    const leftX = -Math.sin(target.heading);
    const leftY = Math.cos(target.heading);
    const signedLateralMeters = (hitter.x - target.x) * leftX + (hitter.y - target.y) * leftY;
    const relativeLateralMps = (hitter.vx - target.vx) * leftX + (hitter.vy - target.vy) * leftY;
    const intrusionMps = Math.abs(signedLateralMeters) > 1e-6
        ? Math.max(0, -relativeLateralMps * Math.sign(signedLateralMeters))
        : 0;
    const bufferMeters = LATERAL_BUFFER_BASE_METERS + intrusionMps * LATERAL_BUFFER_RESPONSE_TIME_SECONDS
        + intrusionMps * intrusionMps / (2 * LATERAL_BUFFER_DECELERATION_MPS2);
    return Math.min(LATERAL_BUFFER_MAX_METERS, bufferMeters);
}

function dangerThresholdSeconds(target) {
    return REACTION_TIME_SECONDS + Math.hypot(target.vx, target.vy) / BRAKING_DECELERATION_MPS2
        + DANGER_TTC_MARGIN_SECONDS;
}

function straightTtcSeconds(target, hitter, lateralBufferMeters, thresholdSeconds) {
    for (let stepIndex = 0; stepIndex < DANGER_TTC_MAX_PROJECTION_STEPS; stepIndex++) {
        const horizonSeconds = stepIndex * DT_SECONDS;
        if (horizonSeconds >= thresholdSeconds) {
            return Infinity;
        }
        const projectedTarget = orientedBoxCorners(straightProjection(target, horizonSeconds), lateralBufferMeters * 2);
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

function minimumBrakingGapMeters(scenario, brakeFrame, postImpactFrame) {
    const brakeSeconds = brakeFrame * DT_SECONDS;
    let gapMeters = Infinity;
    for (let frameIndex = brakeFrame; frameIndex <= postImpactFrame; frameIndex++) {
        const timeSeconds = frameIndex * DT_SECONDS;
        const brakingTarget = targetState(scenario, timeSeconds, brakeSeconds);
        const hitter = hitterState(scenario, timeSeconds);
        if (vehiclesOverlap(brakingTarget, hitter)) {
            return -Infinity;
        }
        gapMeters = Math.min(gapMeters, hitter.x - brakingTarget.x - VEHICLE_LENGTH_METERS);
    }
    return gapMeters;
}

function buildEvidence(scenario) {
    const frameCount = Math.round(SIMULATION_SECONDS / DT_SECONDS);
    let impactFrame = -1;
    for (let frameIndex = 0; frameIndex <= frameCount; frameIndex++) {
        if (vehiclesOverlap(targetState(scenario, frameIndex * DT_SECONDS), hitterState(scenario, frameIndex * DT_SECONDS))) {
            impactFrame = frameIndex;
            break;
        }
    }
    if (impactFrame < 0) {
        fail("Mock scenario produced no collision");
    }
    const impactSeconds = impactFrame * DT_SECONDS;
    const historySeconds = scenario.historySeconds ?? DEFAULT_HISTORY_SECONDS;
    const historyFrameCount = Math.round(historySeconds / DT_SECONDS);
    if (impactFrame < historyFrameCount) {
        fail(`Impact at ${impactSeconds} s leaves less than ${historySeconds} s of history`);
    }
    for (let frameIndex = 0; frameIndex <= frameCount; frameIndex++) {
        for (const context of contextStates(scenario, frameIndex * DT_SECONDS)) {
            if (vehiclesOverlap(context, targetState(scenario, frameIndex * DT_SECONDS))
                || vehiclesOverlap(context, hitterState(scenario, frameIndex * DT_SECONDS))) {
                fail(`Context vehicle overlaps a key vehicle at frame ${frameIndex}`);
            }
        }
    }

    const postImpactFrame = impactFrame + Math.round(POST_IMPACT_CHECK_SECONDS / DT_SECONDS);
    const maximumLeadSteps = Math.min(impactFrame, Math.round(BRAKE_LEAD_SEARCH_MAX_SECONDS / DT_SECONDS));
    let latestAvoidingLeadSteps = -1;
    for (let leadSteps = 1; leadSteps <= maximumLeadSteps; leadSteps++) {
        if (minimumBrakingGapMeters(scenario, impactFrame - leadSteps, postImpactFrame) > -Infinity) {
            latestAvoidingLeadSteps = leadSteps;
            break;
        }
    }
    if (latestAvoidingLeadSteps < 0) {
        fail(`No braking start within ${BRAKE_LEAD_SEARCH_MAX_SECONDS} s avoids the collision`);
    }
    const brakeLeadSteps = latestAvoidingLeadSteps + Math.round(BRAKE_MARGIN_SECONDS / DT_SECONDS);
    const warningLeadSteps = brakeLeadSteps + Math.round(WARNING_TO_BRAKE_SECONDS / DT_SECONDS);
    const brakeFrame = impactFrame - brakeLeadSteps;
    const brakeSeconds = brakeFrame * DT_SECONDS;
    const detectionFrame = impactFrame - warningLeadSteps;
    const detectionSeconds = detectionFrame * DT_SECONDS;
    if (detectionFrame < 0) {
        fail("Warning frame precedes the start of the mock scenario");
    }
    const brakingGapMeters = minimumBrakingGapMeters(scenario, brakeFrame, postImpactFrame);
    if (brakingGapMeters === -Infinity) {
        fail(`Braking ${brakeLeadSteps} steps before impact still collides`);
    }

    const targetAtDetection = targetState(scenario, detectionSeconds);
    const hitterAtDetection = hitterState(scenario, detectionSeconds);
    const lateralBufferMeters = lateralSafetyBufferMeters(targetAtDetection, hitterAtDetection);
    const thresholdSeconds = dangerThresholdSeconds(targetAtDetection);
    const ttcSeconds = straightTtcSeconds(targetAtDetection, hitterAtDetection, lateralBufferMeters, thresholdSeconds);
    if (!(ttcSeconds < thresholdSeconds)) {
        fail(`No straight-TTC conflict below ${thresholdSeconds.toFixed(2)} s at ${warningLeadSteps} steps before impact`);
    }
    const projectedTarget = straightProjection(targetAtDetection, ttcSeconds);
    const projectedHitter = straightProjection(hitterAtDetection, ttcSeconds);
    const projectedOverlap = clipPolygon(
        orientedBoxCorners(projectedHitter),
        orientedBoxCorners(projectedTarget, lateralBufferMeters * 2).reverse(),
    );
    if (projectedOverlap.length < 3) {
        fail("Projected footprints do not overlap at the reported TTC");
    }

    const firstFrame = Math.max(0, detectionFrame - historyFrameCount);
    const frames = [];
    for (let frameIndex = firstFrame; frameIndex <= impactFrame; frameIndex++) {
        const timeSeconds = frameIndex * DT_SECONDS;
        const vehicles = [
            targetState(scenario, timeSeconds), hitterState(scenario, timeSeconds), ...contextStates(scenario, timeSeconds),
        ];
        frames.push(vehicles.map((vehicle, vehicleIndex) => serializeVehicle(vehicle, vehicleIndex)));
    }
    const brakingTargetStates = [];
    for (let frameIndex = brakeFrame; frameIndex <= impactFrame; frameIndex++) {
        const { x, y, heading } = targetState(scenario, frameIndex * DT_SECONDS, brakeSeconds);
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
        braking_start_frame: brakeFrame,
        roads: roadPolylines(),
        frames,
        braking_target_states: brakingTargetStates,
        prediction: {
            ttc_mode: "straight",
            ttc_seconds: ttcSeconds,
            lateral_buffer_meters: lateralBufferMeters,
            target_path: [targetAtDetection, projectedTarget].map(({ x, y }) => ({ x, y })),
            hitter_path: [hitterAtDetection, projectedHitter].map(({ x, y }) => ({ x, y })),
            projected_target: serializeVehicle(projectedTarget, TARGET_AGENT_INDEX),
            projected_hitter: serializeVehicle(projectedHitter, HITTER_AGENT_INDEX),
            overlap_polygon: projectedOverlap,
        },
    };
    return {
        figureEvidence,
        impactSeconds,
        latestAvoidingBrakeSeconds: latestAvoidingLeadSteps * DT_SECONDS,
        brakeLeadSeconds: brakeLeadSteps * DT_SECONDS,
        warningLeadSeconds: warningLeadSteps * DT_SECONDS,
        ttcSeconds,
        thresholdSeconds,
        lateralBufferMeters,
        brakingGapMeters,
    };
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

function parseCliArguments(cliArguments) {
    let evidenceOnly = false;
    let presetName = DEFAULT_PRESET;
    const positionalArguments = [];
    for (let argumentIndex = 0; argumentIndex < cliArguments.length; argumentIndex++) {
        const argument = cliArguments[argumentIndex];
        if (argument === "--evidence-only") {
            evidenceOnly = true;
            continue;
        }
        if (argument === "--preset") {
            argumentIndex++;
            presetName = cliArguments[argumentIndex];
            continue;
        }
        if (argument.startsWith("-")) {
            fail(`Unknown option: ${argument}`);
        }
        positionalArguments.push(argument);
    }
    if (!Object.hasOwn(MOCK_PRESETS, presetName)) {
        fail(`Unknown preset ${presetName}; expected one of ${Object.keys(MOCK_PRESETS).join(", ")}`);
    }
    return { evidenceOnly, presetName, outputArgument: positionalArguments[0] };
}

function main() {
    const { evidenceOnly, presetName, outputArgument } = parseCliArguments(process.argv.slice(2));
    const scenario = MOCK_PRESETS[presetName];
    const outputDirectory = path.resolve(outputArgument || path.join("paper_figures", scenario.outputName));
    fs.mkdirSync(outputDirectory, { recursive: true });
    const evidence = buildEvidence(scenario);
    const { figureEvidence, impactSeconds, ttcSeconds, thresholdSeconds, lateralBufferMeters, brakingGapMeters } = evidence;
    const evidencePath = path.join(outputDirectory, "figure_evidence.json");
    fs.writeFileSync(evidencePath, `${JSON.stringify(figureEvidence)}\n`);
    const writtenFiles = [evidencePath, ...(evidenceOnly ? [] : runRenderer(evidencePath, outputDirectory))];
    const metadata = {
        schema_version: 2,
        provenance: "synthetic mock; analytic kinematics, not replay or simulator data",
        preset: presetName,
        scenario: scenario.description,
        dt_seconds: DT_SECONDS,
        vehicle_size_meters: [VEHICLE_LENGTH_METERS, VEHICLE_WIDTH_METERS],
        target_speed_mps: scenario.targetSpeedMps,
        hitter_speed_mps: scenario.hitterSpeedMps,
        cut_in_duration_seconds: scenario.cutInDurationSeconds,
        collision_time_seconds: impactSeconds,
        braking_counterfactual: {
            seconds_before_collision: evidence.brakeLeadSeconds,
            latest_avoiding_seconds_before_collision: evidence.latestAvoidingBrakeSeconds,
            deceleration_mps2: BRAKING_DECELERATION_MPS2,
            avoided: true,
            minimum_gap_meters: Number(brakingGapMeters.toFixed(3)),
        },
        early_warning: {
            seconds_before_collision: evidence.warningLeadSeconds,
            selected_ttc_mode: "straight",
            selected_ttc_seconds: ttcSeconds,
            danger_threshold_seconds: thresholdSeconds,
            lateral_buffer_meters: lateralBufferMeters,
        },
        rendering: {
            method: "figure_evidence.json rendered by scripts/render_paper_figure_panels.py (matplotlib)",
            history_seconds: scenario.historySeconds ?? DEFAULT_HISTORY_SECONDS,
        },
        outputs: writtenFiles.map((filePath) => path.basename(filePath)),
    };
    fs.writeFileSync(path.join(outputDirectory, "figure_metadata.json"), `${JSON.stringify(metadata, null, 2)}\n`);
    process.stdout.write(`${presetName}: collision at t=${formatNumber(impactSeconds)} s; `
        + `latest avoiding brake -${formatNumber(evidence.latestAvoidingBrakeSeconds)} s, `
        + `brake -${formatNumber(evidence.brakeLeadSeconds)} s, warning -${formatNumber(evidence.warningLeadSeconds)} s; `
        + `TTC ${formatNumber(ttcSeconds, 2)} s < ${formatNumber(thresholdSeconds, 2)} s; `
        + `braking gap ${formatNumber(brakingGapMeters, 2)} m\n`);
    process.stdout.write(`Wrote mock cut-in evidence${evidenceOnly ? "" : " and panels"} to ${outputDirectory}\n`);
}

try {
    main();
} catch (error) {
    process.stderr.write(`error: ${error.message}\n`);
    process.exit(1);
}
