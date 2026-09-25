#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";

const PANEL_WIDTH_PX = 1800;
const TRIPTYCH_WIDTH_PX = 5400;
const PLOT_LEFT_PX = 58;
const PLOT_TOP_PX = 158;
const PLOT_WIDTH_PX = 1684;
const PLOT_HEIGHT_PX = 846;
const TRIPTYCH_PLOT_WIDTH_PX = 2 * PANEL_WIDTH_PX + PLOT_LEFT_PX + PLOT_WIDTH_PX - PLOT_LEFT_PX;

function collectSvgPaths(directoryPath) {
    const svgPaths = [];
    const entries = fs.readdirSync(directoryPath, { withFileTypes: true });
    entries.sort((first, second) => first.name.localeCompare(second.name));
    for (const entry of entries) {
        const entryPath = path.join(directoryPath, entry.name);
        if (entry.isDirectory()) {
            svgPaths.push(...collectSvgPaths(entryPath));
            continue;
        }
        if (entry.isFile() && entry.name.endsWith(".svg")) {
            svgPaths.push(entryPath);
        }
    }
    return svgPaths;
}

function replaceSvgViewport(svg, sourceWidth, outputWidth) {
    const rootTagMatch = svg.match(/<svg\b[^>]*>/);
    if (!rootTagMatch) {
        throw new Error("SVG has no root element");
    }
    let rootTag = rootTagMatch[0];
    rootTag = rootTag.replace(/\swidth="[^"]*"/, ` width="${outputWidth}"`);
    rootTag = rootTag.replace(/\sheight="[^"]*"/, ` height="${PLOT_HEIGHT_PX}"`);
    rootTag = rootTag.replace(
        /\sviewBox="[^"]*"/,
        ` viewBox="${PLOT_LEFT_PX} ${PLOT_TOP_PX} ${outputWidth} ${PLOT_HEIGHT_PX}"`,
    );
    if (!rootTag.includes("viewBox=")) {
        rootTag = rootTag.replace(">", ` viewBox="${PLOT_LEFT_PX} ${PLOT_TOP_PX} ${outputWidth} ${PLOT_HEIGHT_PX}">`);
    }
    if (sourceWidth !== PANEL_WIDTH_PX && sourceWidth !== TRIPTYCH_WIDTH_PX) {
        throw new Error(`Unsupported SVG width ${sourceWidth}`);
    }
    return svg.replace(rootTagMatch[0], rootTag);
}

function main() {
    const sourceDirectory = path.resolve(process.argv[2] || "paper_figures");
    if (!fs.existsSync(sourceDirectory) || !fs.statSync(sourceDirectory).isDirectory()) {
        throw new Error(`Paper figure directory does not exist: ${sourceDirectory}`);
    }
    const temporaryDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "pufferdrive-label-free-"));
    const jobs = [];
    for (const svgPath of collectSvgPaths(sourceDirectory)) {
        const svg = fs.readFileSync(svgPath, "utf8");
        const widthMatch = svg.match(/<svg\b[^>]*\swidth="(\d+)"/);
        if (!widthMatch) {
            throw new Error(`Cannot determine SVG width: ${svgPath}`);
        }
        const sourceWidth = Number(widthMatch[1]);
        const outputWidth = sourceWidth === PANEL_WIDTH_PX ? PLOT_WIDTH_PX : TRIPTYCH_PLOT_WIDTH_PX;
        const withoutText = svg.replace(/<text\b[^>]*>[\s\S]*?<\/text>/g, "");
        const labelFreeSvg = replaceSvgViewport(withoutText, sourceWidth, outputWidth);
        const relativePath = path.relative(sourceDirectory, svgPath);
        const temporarySvgPath = path.join(temporaryDirectory, relativePath);
        fs.mkdirSync(path.dirname(temporarySvgPath), { recursive: true });
        fs.writeFileSync(temporarySvgPath, labelFreeSvg);
        jobs.push({
            source_svg: svgPath,
            temporary_svg: temporarySvgPath,
            output_png: svgPath.replace(/\.svg$/, ".png"),
            width: outputWidth,
            height: PLOT_HEIGHT_PX,
        });
    }
    const manifestPath = path.join(temporaryDirectory, "manifest.json");
    fs.writeFileSync(manifestPath, `${JSON.stringify(jobs, null, 2)}\n`);
    process.stdout.write(`${manifestPath}\n`);
}

try {
    main();
} catch (error) {
    process.stderr.write(`error: ${error.message}\n`);
    process.exit(1);
}
