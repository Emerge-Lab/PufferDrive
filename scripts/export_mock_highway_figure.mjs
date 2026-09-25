#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const PANEL_WIDTH_PX = 1800;
const PANEL_HEIGHT_PX = 1200;
const TRIPTYCH_WIDTH_PX = PANEL_WIDTH_PX * 3;
const PLOT_LEFT_PX = 58;
const PLOT_TOP_PX = 158;
const PLOT_WIDTH_PX = 1684;
const PLOT_HEIGHT_PX = 846;
const ROAD_TOP_PX = 300;
const ROAD_BOTTOM_PX = 850;
const VEHICLE_LENGTH_PX = 118;
const VEHICLE_WIDTH_PX = 54;

const COLORS = Object.freeze({
    background: "#ffffff",
    surface: "#f8fafc",
    road: "#f1f5f9",
    text: "#111827",
    mutedText: "#4b5563",
    border: "#cbd5e1",
    lane: "#94a3b8",
    roadEdge: "#475569",
    ego: "#d55e00",
    conflict: "#0072b2",
    context: "#9ca3af",
    danger: "#c81e1e",
    safe: "#15803d",
    projection: "#6d28d9",
    schematic: "#6d28d9",
});

function escapeXml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&apos;");
}

function formatNumber(value, fractionDigits = 1) {
    return Number(value).toFixed(fractionDigits).replace(/\.0$/, "");
}

function pointsAttribute(points) {
    return points.map((point) => `${formatNumber(point.x)},${formatNumber(point.y)}`).join(" ");
}

function vehicleCorners(vehicle, expansionPx = 0) {
    const halfLength = VEHICLE_LENGTH_PX / 2;
    const halfWidth = VEHICLE_WIDTH_PX / 2 + expansionPx;
    const cosHeading = Math.cos(vehicle.heading);
    const sinHeading = Math.sin(vehicle.heading);
    return [
        { longitudinal: halfLength, lateral: halfWidth },
        { longitudinal: halfLength, lateral: -halfWidth },
        { longitudinal: -halfLength, lateral: -halfWidth },
        { longitudinal: -halfLength, lateral: halfWidth },
    ].map(({ longitudinal, lateral }) => ({
        x: vehicle.x + longitudinal * cosHeading - lateral * sinHeading,
        y: vehicle.y + longitudinal * sinHeading + lateral * cosHeading,
    }));
}

function vehicleHeading(start, end) {
    return Math.atan2(end.y - start.y, end.x - start.x);
}

function withHeadings(points) {
    return points.map((point, pointIndex) => {
        const start = pointIndex === points.length - 1 ? points[pointIndex - 1] : point;
        const end = pointIndex === points.length - 1 ? point : points[pointIndex + 1];
        return { ...point, heading: vehicleHeading(start, end) };
    });
}

function renderVehicle(vehicle, color, options = {}) {
    const corners = vehicleCorners(vehicle, options.expansionPx || 0);
    const forwardX = Math.cos(vehicle.heading);
    const forwardY = Math.sin(vehicle.heading);
    const leftX = -forwardY;
    const leftY = forwardX;
    const frontCenter = {
        x: vehicle.x + forwardX * VEHICLE_LENGTH_PX * 0.42,
        y: vehicle.y + forwardY * VEHICLE_LENGTH_PX * 0.42,
    };
    const frontMarker = [
        { x: frontCenter.x + forwardX * 13, y: frontCenter.y + forwardY * 13 },
        { x: frontCenter.x + leftX * 11, y: frontCenter.y + leftY * 11 },
        { x: frontCenter.x - leftX * 11, y: frontCenter.y - leftY * 11 },
    ];
    const dash = options.dash ? ` stroke-dasharray="${options.dash}"` : "";
    const marker = options.hideFrontMarker
        ? ""
        : `<polygon points="${pointsAttribute(frontMarker)}" fill="#ffffff" opacity="0.72"/>`;
    return `<g opacity="${options.opacity ?? 1}">`
        + `<polygon points="${pointsAttribute(corners)}" fill="${options.outline ? "none" : color}" `
        + `fill-opacity="${options.fillOpacity ?? 0.92}" stroke="${color}" stroke-width="${options.strokeWidth || 3}"${dash}/>`
        + marker + `</g>`;
}

function renderPolyline(points, color, options = {}) {
    const dash = options.dash ? ` stroke-dasharray="${options.dash}"` : "";
    const marker = options.marker ? ` marker-end="url(#${options.marker})"` : "";
    return `<polyline points="${pointsAttribute(points)}" fill="none" stroke="${color}" `
        + `stroke-width="${options.width || 6}" stroke-linecap="round" stroke-linejoin="round" `
        + `opacity="${options.opacity ?? 1}"${dash}${marker}/>`;
}

function renderHistory(states, color) {
    const elements = [];
    const finalIndex = states.length - 1;
    for (let stateIndex = 1; stateIndex < states.length; stateIndex++) {
        const opacity = 0.10 + 0.34 * stateIndex / finalIndex;
        elements.push(renderPolyline([states[stateIndex - 1], states[stateIndex]], color, {
            width: 5,
            opacity,
        }));
    }
    for (let stateIndex = 0; stateIndex < finalIndex; stateIndex++) {
        const opacity = 0.08 + 0.34 * stateIndex / finalIndex;
        elements.push(renderVehicle(states[stateIndex], color, {
            opacity,
            fillOpacity: 0.7,
            strokeWidth: 2,
            hideFrontMarker: true,
        }));
    }
    return elements.join("\n");
}

function panelDefinitions(prefix) {
    return `<defs>`
        + `<clipPath id="${prefix}-clip"><rect x="${PLOT_LEFT_PX}" y="${PLOT_TOP_PX}" `
        + `width="${PLOT_WIDTH_PX}" height="${PLOT_HEIGHT_PX}" rx="14"/></clipPath>`
        + `<marker id="${prefix}-ego-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" `
        + `orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="${COLORS.ego}"/></marker>`
        + `<marker id="${prefix}-conflict-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" `
        + `orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="${COLORS.conflict}"/></marker>`
        + `</defs>`;
}

function renderHeader(letter, title, subtitle, badgeText, badgeColor) {
    const badgeWidth = Math.max(360, badgeText.length * 14 + 50);
    return `<rect width="${PANEL_WIDTH_PX}" height="${PANEL_HEIGHT_PX}" fill="${COLORS.background}"/>`
        + `<rect x="1.5" y="1.5" width="${PANEL_WIDTH_PX - 3}" height="${PANEL_HEIGHT_PX - 3}" rx="18" `
        + `fill="none" stroke="${COLORS.border}" stroke-width="3"/>`
        + `<rect x="58" y="42" width="66" height="66" rx="14" fill="${COLORS.text}"/>`
        + `<text x="91" y="88" text-anchor="middle" font-size="38" font-weight="800" fill="#ffffff">${letter}</text>`
        + `<text x="148" y="75" font-size="38" font-weight="750" fill="${COLORS.text}">${escapeXml(title)}</text>`
        + `<text x="148" y="112" font-size="24" fill="${COLORS.mutedText}">${escapeXml(subtitle)}</text>`
        + `<rect x="${PANEL_WIDTH_PX - badgeWidth - 58}" y="49" width="${badgeWidth}" height="48" rx="24" `
        + `fill="${badgeColor}" fill-opacity="0.10" stroke="${badgeColor}" stroke-width="2"/>`
        + `<text x="${PANEL_WIDTH_PX - 58 - badgeWidth / 2}" y="81" text-anchor="middle" font-size="21" `
        + `font-weight="750" fill="${badgeColor}">${escapeXml(badgeText)}</text>`
        + `<rect x="${PLOT_LEFT_PX}" y="${PLOT_TOP_PX}" width="${PLOT_WIDTH_PX}" height="${PLOT_HEIGHT_PX}" `
        + `rx="14" fill="${COLORS.surface}" stroke="${COLORS.border}" stroke-width="2"/>`;
}

function renderHighway() {
    const laneHeight = (ROAD_BOTTOM_PX - ROAD_TOP_PX) / 4;
    const elements = [
        `<rect x="${PLOT_LEFT_PX}" y="${ROAD_TOP_PX}" width="${PLOT_WIDTH_PX}" `
            + `height="${ROAD_BOTTOM_PX - ROAD_TOP_PX}" fill="${COLORS.road}"/>`,
        `<line x1="${PLOT_LEFT_PX}" y1="${ROAD_TOP_PX}" x2="${PLOT_LEFT_PX + PLOT_WIDTH_PX}" `
            + `y2="${ROAD_TOP_PX}" stroke="${COLORS.roadEdge}" stroke-width="4"/>`,
        `<line x1="${PLOT_LEFT_PX}" y1="${ROAD_BOTTOM_PX}" x2="${PLOT_LEFT_PX + PLOT_WIDTH_PX}" `
            + `y2="${ROAD_BOTTOM_PX}" stroke="${COLORS.roadEdge}" stroke-width="4"/>`,
    ];
    for (let laneIndex = 1; laneIndex < 4; laneIndex++) {
        const y = ROAD_TOP_PX + laneIndex * laneHeight;
        elements.push(`<line x1="${PLOT_LEFT_PX}" y1="${formatNumber(y)}" `
            + `x2="${PLOT_LEFT_PX + PLOT_WIDTH_PX}" y2="${formatNumber(y)}" `
            + `stroke="${COLORS.lane}" stroke-width="2" stroke-dasharray="12 11"/>`);
    }
    return elements.join("\n");
}

function renderContextVehicles() {
    const contextVehicles = [
        { x: 350, y: 369, heading: 0 },
        { x: 690, y: 644, heading: 0 },
        { x: 1580, y: 369, heading: 0 },
    ];
    return contextVehicles.map((vehicle) => renderVehicle(vehicle, COLORS.context, {
        opacity: 0.68,
        fillOpacity: 0.7,
        hideFrontMarker: true,
    })).join("\n");
}

function renderScaleBar() {
    return `<g><line x1="84" y1="974" x2="234" y2="974" stroke="${COLORS.text}" stroke-width="5"/>`
        + `<line x1="84" y1="966" x2="84" y2="982" stroke="${COLORS.text}" stroke-width="4"/>`
        + `<line x1="234" y1="966" x2="234" y2="982" stroke="${COLORS.text}" stroke-width="4"/>`
        + `<text x="159" y="960" text-anchor="middle" font-size="21" font-weight="600" fill="${COLORS.text}">10 m</text></g>`;
}

function legendEntry(x, y, color, label, outline = false) {
    return `<rect x="${x}" y="${y - 18}" width="42" height="27" rx="5" fill="${outline ? "none" : color}" `
        + `fill-opacity="0.92" stroke="${color}" stroke-width="3"${outline ? ' stroke-dasharray="10 7"' : ""}/>`
        + `<text x="${x + 56}" y="${y + 4}" font-size="22" fill="${COLORS.text}">${escapeXml(label)}</text>`;
}

function renderLegend(includeProjection = false) {
    const entries = [
        legendEntry(80, 1080, COLORS.ego, "Ego vehicle"),
        legendEntry(350, 1080, COLORS.conflict, "Cut-in vehicle"),
    ];
    if (includeProjection) {
        entries.push(legendEntry(650, 1080, COLORS.projection, "Projected footprint", true));
    }
    return `<g>${entries.join("\n")}</g>`;
}

function renderOverlap(center, label) {
    return `<circle cx="${center.x}" cy="${center.y}" r="25" fill="${COLORS.danger}" fill-opacity="0.15" `
        + `stroke="${COLORS.danger}" stroke-width="5"/>`
        + `<text x="${center.x - 34}" y="${center.y - 31}" text-anchor="end" font-size="23" `
        + `font-weight="700" fill="${COLORS.danger}">${escapeXml(label)}</text>`;
}

function renderFooter(text) {
    return `<text x="1720" y="1130" text-anchor="end" font-size="20" fill="${COLORS.mutedText}">`
        + `${escapeXml(text)}</text>`;
}

const egoCollisionHistory = withHeadings([
    { x: 350, y: 781 }, { x: 520, y: 781 }, { x: 690, y: 781 }, { x: 860, y: 781 },
    { x: 1030, y: 781 }, { x: 1200, y: 781 }, { x: 1450, y: 781 },
]);
const conflictCollisionHistory = withHeadings([
    { x: 820, y: 644 }, { x: 930, y: 644 }, { x: 1040, y: 653 }, { x: 1150, y: 693 },
    { x: 1260, y: 746 }, { x: 1360, y: 776 }, { x: 1450, y: 781 },
]);
const egoBrakingPath = withHeadings([
    { x: 860, y: 781 }, { x: 1010, y: 781 }, { x: 1125, y: 781 }, { x: 1210, y: 781 }, { x: 1260, y: 781 },
]);

function renderPanelA() {
    const prefix = "mock-a";
    const body = [renderHeader(
        "A",
        "Adjacent-lane cut-in collision",
        "Schematic highway scenario · nominal impact time",
        "MOCK SCENARIO · NOT REPLAY DATA",
        COLORS.schematic,
    )];
    body.push(`<g clip-path="url(#${prefix}-clip)">`, renderHighway());
    body.push(renderHistory(egoCollisionHistory, COLORS.ego));
    body.push(renderHistory(conflictCollisionHistory, COLORS.conflict));
    body.push(renderContextVehicles());
    body.push(renderVehicle(egoCollisionHistory.at(-1), COLORS.ego));
    body.push(renderVehicle(conflictCollisionHistory.at(-1), COLORS.conflict));
    body.push(renderOverlap({ x: 1450, y: 781 }, "Collision overlap"));
    body.push(renderScaleBar(), `</g>`, renderLegend());
    body.push(renderFooter("Faded footprints: preceding 3.0 s · vehicle begins one lane away"));
    return { prefix, definitions: panelDefinitions(prefix), body: body.join("\n") };
}

function renderPanelB() {
    const prefix = "mock-b";
    const body = [renderHeader(
        "B",
        "Earlier ego braking",
        "Constant deceleration from t = −1.5 s",
        "COLLISION AVOIDED",
        COLORS.safe,
    )];
    body.push(`<g clip-path="url(#${prefix}-clip)">`, renderHighway());
    body.push(renderHistory(egoCollisionHistory, COLORS.ego));
    body.push(renderHistory(conflictCollisionHistory, COLORS.conflict));
    body.push(renderContextVehicles());
    body.push(renderPolyline(egoBrakingPath, COLORS.ego, {
        width: 8,
        opacity: 0.94,
        marker: `${prefix}-ego-arrow`,
    }));
    body.push(renderVehicle(egoCollisionHistory.at(-1), COLORS.ego, {
        outline: true,
        dash: "12 9",
        opacity: 0.72,
        hideFrontMarker: true,
    }));
    body.push(renderVehicle(egoBrakingPath.at(-1), COLORS.ego));
    body.push(renderVehicle(conflictCollisionHistory.at(-1), COLORS.conflict));
    body.push(renderScaleBar(), `</g>`, renderLegend());
    body.push(renderFooter("Dashed orange footprint: unbraked ego at nominal impact time"));
    return { prefix, definitions: panelDefinitions(prefix), body: body.join("\n") };
}

function renderPanelC() {
    const prefix = "mock-c";
    const egoCurrent = egoCollisionHistory[1];
    const conflictCurrent = conflictCollisionHistory[1];
    const egoPast = egoCollisionHistory.slice(0, 2);
    const conflictPast = conflictCollisionHistory.slice(0, 2);
    const egoProjection = [egoCurrent, { x: 1450, y: 781, heading: 0 }];
    const conflictProjection = [conflictCurrent, { x: 1450, y: 781, heading: conflictCollisionHistory.at(-1).heading }];
    const body = [renderHeader(
        "C",
        "Conflict visible 2.5 s earlier",
        "Cut-in vehicle is ahead in the adjacent lane",
        "PREDICTED CONFLICT",
        COLORS.danger,
    )];
    body.push(`<g clip-path="url(#${prefix}-clip)">`, renderHighway());
    body.push(renderHistory(egoPast, COLORS.ego));
    body.push(renderHistory(conflictPast, COLORS.conflict));
    body.push(renderContextVehicles());
    body.push(renderPolyline(egoProjection, COLORS.ego, {
        width: 7,
        opacity: 0.82,
        dash: "13 10",
        marker: `${prefix}-ego-arrow`,
    }));
    body.push(renderPolyline(conflictProjection, COLORS.conflict, {
        width: 7,
        opacity: 0.82,
        dash: "13 10",
        marker: `${prefix}-conflict-arrow`,
    }));
    body.push(renderVehicle(egoProjection.at(-1), COLORS.ego, {
        outline: true,
        dash: "12 9",
        opacity: 0.78,
        hideFrontMarker: true,
    }));
    body.push(renderVehicle(conflictProjection.at(-1), COLORS.conflict, {
        outline: true,
        dash: "12 9",
        opacity: 0.78,
        hideFrontMarker: true,
    }));
    body.push(renderVehicle(egoCurrent, COLORS.ego));
    body.push(renderVehicle(conflictCurrent, COLORS.conflict));
    body.push(renderOverlap({ x: 1450, y: 781 }, "Predicted overlap"));
    body.push(renderScaleBar(), `</g>`, renderLegend(true));
    body.push(renderFooter("Dashed paths and footprints: constant-velocity schematic projection"));
    return { prefix, definitions: panelDefinitions(prefix), body: body.join("\n") };
}

function completeSvg(width, height, definitions, body) {
    return `<?xml version="1.0" encoding="UTF-8"?>\n`
        + `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" `
        + `viewBox="0 0 ${width} ${height}" role="img">\n`
        + `<style>text { font-family: Arial, Helvetica, sans-serif; }</style>\n`
        + `${definitions}\n${body}\n</svg>\n`;
}

function validatePngDimensions(pngPath, expectedWidth, expectedHeight) {
    const pngBytes = fs.readFileSync(pngPath);
    if (pngBytes.length < 24 || pngBytes.subarray(0, 8).toString("hex") !== "89504e470d0a1a0a") {
        throw new Error(`${pngPath} is not a valid PNG`);
    }
    const width = pngBytes.readUInt32BE(16);
    const height = pngBytes.readUInt32BE(20);
    if (width !== expectedWidth || height !== expectedHeight) {
        throw new Error(`${pngPath} has dimensions ${width}x${height}; expected ${expectedWidth}x${expectedHeight}`);
    }
}

function rasterizeSvg(svgPath, pngPath, width, height) {
    const chromeCandidates = [process.env.CHROME_PATH, "/usr/bin/google-chrome", "google-chrome"].filter(Boolean);
    let lastError = "Chrome executable not found";
    for (const chromePath of chromeCandidates) {
        const result = spawnSync(chromePath, [
            "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
            "--force-device-scale-factor=1", `--window-size=${width},${height}`,
            `--screenshot=${pngPath}`, pathToFileURL(path.resolve(svgPath)).href,
        ], { encoding: "utf8", timeout: 30000 });
        if (!result.error && result.status === 0 && fs.existsSync(pngPath)) {
            validatePngDimensions(pngPath, width, height);
            return;
        }
        lastError = result.error?.message || result.stderr || `exit status ${result.status}`;
        if (result.error?.code !== "ENOENT") {
            break;
        }
    }
    throw new Error(`Unable to rasterize ${path.basename(svgPath)}: ${lastError.trim()}`);
}

function main() {
    const svgOnly = process.argv.includes("--svg-only");
    const outputArgument = process.argv.slice(2).find((argument) => argument !== "--svg-only");
    const outputDirectory = path.resolve(
        outputArgument || path.join("paper_figures", "mock_highway_one_lane_cut_in"),
    );
    fs.mkdirSync(outputDirectory, { recursive: true });
    const panels = [renderPanelA(), renderPanelB(), renderPanelC()];
    const panelNames = ["panel_a_mock_collision", "panel_b_mock_braking", "panel_c_mock_prediction"];
    for (let panelIndex = 0; panelIndex < panels.length; panelIndex++) {
        const panel = panels[panelIndex];
        const svgPath = path.join(outputDirectory, `${panelNames[panelIndex]}.svg`);
        const pngPath = path.join(outputDirectory, `${panelNames[panelIndex]}.png`);
        fs.writeFileSync(svgPath, completeSvg(PANEL_WIDTH_PX, PANEL_HEIGHT_PX, panel.definitions, panel.body));
        if (!svgOnly) {
            rasterizeSvg(svgPath, pngPath, PANEL_WIDTH_PX, PANEL_HEIGHT_PX);
        }
    }
    const triptychSvgPath = path.join(outputDirectory, "figure_abc.svg");
    const triptychPngPath = path.join(outputDirectory, "figure_abc.png");
    const definitions = panels.map((panel) => panel.definitions).join("\n");
    const body = panels.map((panel, panelIndex) => (
        `<g transform="translate(${panelIndex * PANEL_WIDTH_PX},0)">${panel.body}</g>`
    )).join("\n");
    fs.writeFileSync(triptychSvgPath, completeSvg(TRIPTYCH_WIDTH_PX, PANEL_HEIGHT_PX, definitions, body));
    if (!svgOnly) {
        rasterizeSvg(triptychSvgPath, triptychPngPath, TRIPTYCH_WIDTH_PX, PANEL_HEIGHT_PX);
    }
    const metadata = {
        schema_version: 1,
        provenance: "synthetic schematic; not generated from replay, simulation, or measured data",
        scenario: "four-lane highway; ego travels straight; vehicle initially ahead in the adjacent lane cuts into ego lane",
        history_seconds: 3,
        footprint_interval_seconds: 0.5,
        braking: {
            model: "illustrative constant-deceleration trajectory",
            starts_seconds_before_nominal_impact: 1.5,
        },
        prediction: {
            model: "illustrative constant-velocity projection",
            seconds_before_nominal_impact: 2.5,
        },
        panel_dimensions_px: [PANEL_WIDTH_PX, PANEL_HEIGHT_PX],
        triptych_dimensions_px: [TRIPTYCH_WIDTH_PX, PANEL_HEIGHT_PX],
    };
    fs.writeFileSync(path.join(outputDirectory, "figure_metadata.json"), `${JSON.stringify(metadata, null, 2)}\n`);
    process.stdout.write(`Wrote mock highway figure set to ${outputDirectory}\n`);
    if (svgOnly) {
        process.stdout.write("PNG rasterization skipped by --svg-only\n");
    }
}

try {
    main();
} catch (error) {
    process.stderr.write(`error: ${error.message}\n`);
    process.exit(1);
}
