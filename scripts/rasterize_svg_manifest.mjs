#!/usr/bin/env node

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { spawn } from "node:child_process";
import { pathToFileURL } from "node:url";

const MAX_CONCURRENT_CHROME_PROCESSES = 4;
const RASTER_TIMEOUT_MS = 60000;
const OUTPUT_POLL_INTERVAL_MS = 250;

function validatePng(pngPath, expectedWidth, expectedHeight, earliestMtimeMs) {
    if (!fs.existsSync(pngPath)) {
        return false;
    }
    const stats = fs.statSync(pngPath);
    if (stats.mtimeMs < earliestMtimeMs || stats.size < 24) {
        return false;
    }
    const pngBytes = fs.readFileSync(pngPath);
    if (pngBytes.subarray(0, 8).toString("hex") !== "89504e470d0a1a0a") {
        return false;
    }
    return pngBytes.readUInt32BE(16) === expectedWidth
        && pngBytes.readUInt32BE(20) === expectedHeight;
}

function rasterize(job, jobIndex, jobCount) {
    return new Promise((resolve, reject) => {
        const profileDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "pufferdrive-chrome-"));
        const startTimeMs = Date.now();
        const chrome = spawn("google-chrome", [
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            `--user-data-dir=${profileDirectory}`,
            `--window-size=${job.width},${job.height}`,
            `--screenshot=${job.output_png}`,
            pathToFileURL(job.temporary_svg).href,
        ], { stdio: "ignore" });
        let settled = false;
        const finish = (error = null) => {
            if (settled) {
                return;
            }
            settled = true;
            clearInterval(pollInterval);
            clearTimeout(timeout);
            if (chrome.exitCode === null) {
                chrome.kill("SIGTERM");
            }
            fs.rmSync(profileDirectory, { recursive: true, force: true });
            if (error) {
                reject(error);
                return;
            }
            process.stdout.write(`[${jobIndex + 1}/${jobCount}] ${job.output_png}\n`);
            resolve();
        };
        const pollInterval = setInterval(() => {
            try {
                if (validatePng(job.output_png, job.width, job.height, startTimeMs)) {
                    finish();
                }
            } catch {
                // Chrome can expose the file before its final bytes are flushed.
            }
        }, OUTPUT_POLL_INTERVAL_MS);
        const timeout = setTimeout(() => {
            finish(new Error(`Rasterization timed out: ${job.temporary_svg}`));
        }, RASTER_TIMEOUT_MS);
        chrome.once("error", (error) => finish(error));
        chrome.once("exit", () => {
            if (!settled && validatePng(job.output_png, job.width, job.height, startTimeMs)) {
                finish();
            }
        });
    });
}

async function main() {
    const manifestPath = process.argv[2];
    if (!manifestPath) {
        throw new Error("Usage: node scripts/rasterize_svg_manifest.mjs <manifest.json>");
    }
    const jobs = JSON.parse(fs.readFileSync(path.resolve(manifestPath), "utf8"));
    if (!Array.isArray(jobs) || jobs.length === 0) {
        throw new Error("Manifest must contain at least one raster job");
    }
    let nextJobIndex = 0;
    async function worker() {
        while (nextJobIndex < jobs.length) {
            const jobIndex = nextJobIndex;
            nextJobIndex++;
            await rasterize(jobs[jobIndex], jobIndex, jobs.length);
        }
    }
    const workerCount = Math.min(MAX_CONCURRENT_CHROME_PROCESSES, jobs.length);
    await Promise.all(Array.from({ length: workerCount }, () => worker()));
}

main().catch((error) => {
    process.stderr.write(`error: ${error.message}\n`);
    process.exit(1);
});
