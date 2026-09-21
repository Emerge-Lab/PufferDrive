#!/usr/bin/env python
"""Render the SPDZONE1 speed zones of PufferDrive map .bin files into one self-contained HTML inspector.

    python scripts/visualize_speed_zones.py --bin-dir ~/ordnung/data/CARLA/puffer_bins_zones
    python scripts/visualize_speed_zones.py --bin-dir <dir> --output zones.html --fragment   # no <html>/<body> wrapper
    python scripts/visualize_speed_zones.py --bin-dir <dir> --draw-seed 3    # overlay one C-side random limit draw
"""

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "data_utils"))
from mirror_map_bin import read_bin  # noqa: E402

LANE_TYPE_MAX = 9
ROAD_EDGE_TYPES = (20, 21, 22)
COORD_DECIMALS = 1
MPS_TO_KMH = 3.6


def _polyline(road):
    return [[round(x, COORD_DECIMALS), round(y, COORD_DECIMALS)] for x, y in zip(road["x"], road["y"])]


def _bbox_diagonal_m(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def sample_limit_draw(bin_file: pathlib.Path, seed: int, delta_mps: float, min_mps: float, max_mps: float) -> dict:
    """Effective per-lane limits (m/s) after one episode draw of the C sampler, keyed by road element id."""
    from pufferlib.ocean.drive.drive import Drive

    env = Drive(
        map_dir=str(bin_file),
        num_maps=1,
        num_agents=32,
        allow_map_subset=True,
        scenario_length=300,
        resample_frequency=1_000_000,
        termination_mode=0,
        report_interval=1,
        speed_limit_random_prob=1.0,
        speed_limit_random_delta_mps=delta_mps,
        speed_limit_random_min_mps=min_mps,
        speed_limit_random_max_mps=max_mps,
    )
    env.reset(seed=seed)
    roads = env.get_state()[0]["road_elements"]
    env.close()
    return {road["id"]: road["speed_limit"] for road in roads}


def collect_map(bin_file: pathlib.Path, draw: dict | None = None) -> dict:
    data = read_bin(bin_file)
    lanes, edges = [], []
    for road in data["roads"]:
        if 0 <= road["type"] <= LANE_TYPE_MAX:
            lanes.append(
                {
                    "id": road["id"],
                    "zone": road["speed_zone_idx"],
                    "kmh": round(road["speed_limit"] * MPS_TO_KMH, 1),
                    "kmh_draw": None if draw is None else round(draw[road["id"]] * MPS_TO_KMH, 1),
                    "length": round(road["length"], 1),
                    "pts": _polyline(road),
                }
            )
        elif road["type"] in ROAD_EDGE_TYPES:
            edges.append(_polyline(road))

    zones = {}
    for lane in lanes:
        if lane["zone"] < 0:
            continue
        zone = zones.setdefault(
            lane["zone"], {"idx": lane["zone"], "lanes": 0, "length": 0.0, "kmh": set(), "kmh_draw": set(), "pts": []}
        )
        zone["lanes"] += 1
        if lane["kmh_draw"] is not None:
            zone["kmh_draw"].add(lane["kmh_draw"])
        zone["length"] += lane["length"]
        zone["kmh"].add(lane["kmh"])
        zone["pts"].extend(lane["pts"])
    zone_rows = []
    for zone in sorted(zones.values(), key=lambda z: z["idx"]):
        xs = [p[0] for p in zone["pts"]]
        ys = [p[1] for p in zone["pts"]]
        zone_rows.append(
            {
                "idx": zone["idx"],
                "lanes": zone["lanes"],
                "length": round(zone["length"], 0),
                "extent": round(_bbox_diagonal_m(zone["pts"]), 0),
                "kmh": sorted(zone["kmh"]),
                "kmh_draw": sorted(zone["kmh_draw"]),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            }
        )
    return {
        "name": bin_file.stem.split("__")[-1],
        "file": bin_file.name,
        "has_zone_section": data["has_zone_section"],
        "lanes": lanes,
        "edges": edges,
        "zones": zone_rows,
    }


STYLE = """
:root {
  --ground: #F2F3F1; --panel: #FAFAF8; --line: #D9DCD8; --text: #1C2126; --muted: #6E7882;
  --accent: #C9552B; --accent-soft: rgba(201, 85, 43, 0.12); --edge: #C3C7C2; --junction: #9AA1A8;
  --row-hover: #ECEEEA; --tooltip: #1C2126; --tooltip-text: #F2F3F1; --lane-l: 44%;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #141719; --panel: #1B1F22; --line: #2C3237; --text: #E4E8EA; --muted: #8B95A0;
    --accent: #E2764B; --accent-soft: rgba(226, 118, 75, 0.18); --edge: #33393E; --junction: #5C656E;
    --row-hover: #23282C; --tooltip: #E4E8EA; --tooltip-text: #141719; --lane-l: 62%;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --ground: #141719; --panel: #1B1F22; --line: #2C3237; --text: #E4E8EA; --muted: #8B95A0;
  --accent: #E2764B; --accent-soft: rgba(226, 118, 75, 0.18); --edge: #33393E; --junction: #5C656E;
  --row-hover: #23282C; --tooltip: #E4E8EA; --tooltip-text: #141719; --lane-l: 62%;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0; background: var(--ground); color: var(--text);
  font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; font-size: 13px; line-height: 1.4;
}
.app { display: grid; grid-template-rows: auto 1fr; grid-template-columns: 1fr 340px; height: 100vh; min-height: 560px; }
header {
  grid-column: 1 / -1; display: flex; align-items: center; gap: 20px; padding: 10px 16px;
  border-bottom: 1px solid var(--line); background: var(--panel); flex-wrap: wrap;
}
header h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: 0.01em; white-space: nowrap; }
.tabs { display: flex; gap: 4px; flex-wrap: wrap; }
.tabs button, .seg button {
  font: inherit; color: var(--text); background: transparent; border: 1px solid var(--line); border-radius: 4px;
  padding: 4px 10px; cursor: pointer;
}
.tabs button[aria-pressed="true"], .seg button[aria-pressed="true"] { background: var(--accent-soft); border-color: var(--accent); }
.tabs button:focus-visible, .seg button:focus-visible, .zone-table tr:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.seg { display: flex; }
.seg button:first-child { border-radius: 4px 0 0 4px; }
.seg button:last-child { border-radius: 0 4px 4px 0; margin-left: -1px; }
.controls { display: flex; align-items: center; gap: 14px; margin-left: auto; color: var(--muted); }
.controls label { display: flex; align-items: center; gap: 5px; cursor: pointer; }
.stage { position: relative; overflow: hidden; }
canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
.hint { position: absolute; left: 12px; bottom: 10px; color: var(--muted); font-size: 12px; pointer-events: none; }
.legend { position: absolute; right: 12px; top: 10px; display: flex; flex-direction: column; gap: 3px; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; pointer-events: none; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
.tooltip {
  position: absolute; pointer-events: none; background: var(--tooltip); color: var(--tooltip-text);
  padding: 6px 9px; border-radius: 4px; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px;
  white-space: nowrap; transform: translate(12px, 12px); line-height: 1.5;
}
.tooltip[hidden] { display: none; }
aside { border-left: 1px solid var(--line); background: var(--panel); display: flex; flex-direction: column; min-height: 0; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
.stat .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
.stat .v { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 16px; font-variant-numeric: tabular-nums; }
.hist { padding: 10px 14px 6px; border-bottom: 1px solid var(--line); }
.hist .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.hist svg { width: 100%; height: 70px; display: block; overflow: visible; }
.hist text { fill: var(--muted); font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 9px; }
.hist rect { fill: var(--muted); opacity: 0.55; }
.table-wrap { overflow: auto; flex: 1; min-height: 0; }
.zone-table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
.zone-table th { position: sticky; top: 0; background: var(--panel); text-align: right; font-weight: 500; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; padding: 8px 10px 6px; border-bottom: 1px solid var(--line); }
.zone-table th:first-child, .zone-table td:first-child { text-align: left; }
.zone-table td { text-align: right; padding: 5px 10px; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px; border-bottom: 1px solid var(--line); }
.zone-table tr { cursor: pointer; }
.zone-table tr:hover td { background: var(--row-hover); }
.zone-table tr[aria-selected="true"] td { background: var(--accent-soft); }
.zone-table i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; vertical-align: -1px; margin-right: 6px; }
.note { padding: 8px 14px; color: var(--muted); font-size: 11.5px; border-top: 1px solid var(--line); }
@media (max-width: 860px) { .app { grid-template-columns: 1fr; grid-template-rows: auto 1fr 320px; } aside { border-left: 0; border-top: 1px solid var(--line); } }
@media (prefers-reduced-motion: no-preference) { .tabs button, .seg button { transition: background 120ms; } }
"""

SCRIPT = r"""
const MAPS = window.__SPEED_ZONE_MAPS__;
const KMH_MIN = 10, KMH_MAX = 130;
const state = { map: 0, mode: "zone", junction: true, edges: true, hover: null, selected: null, view: null };
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const stage = document.querySelector(".stage");

function laneL() { return getComputedStyle(document.documentElement).getPropertyValue("--lane-l").trim() || "44%"; }
function zoneColor(idx) { return `hsl(${(idx * 137.508) % 360} 58% ${laneL()})`; }
function limitColor(kmh) {
  const t = Math.min(1, Math.max(0, (kmh - KMH_MIN) / (KMH_MAX - KMH_MIN)));
  return `hsl(${Math.round(215 - 215 * t)} 70% ${laneL()})`;
}
function laneColor(lane) {
  if (state.mode === "zone") return zoneColor(lane.zone);
  if (state.mode === "draw") return limitColor(lane.kmh_draw ?? lane.kmh);
  return limitColor(lane.kmh);
}
function hasDraw() { return current().lanes.some(l => l.kmh_draw !== null); }
function current() { return MAPS[state.map]; }
function zoneOf(idx) { return current().zones.find(z => z.idx === idx); }

function fitView() {
  const m = current();
  const pts = m.lanes.flatMap(l => l.pts).concat(m.edges.flat());
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of pts) { if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; }
  fitBox([minX, minY, maxX, maxY], 30);
}
function fitBox(box, pad) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const scale = Math.min((w - 2 * pad) / Math.max(1, box[2] - box[0]), (h - 2 * pad) / Math.max(1, box[3] - box[1]));
  state.view = { scale, tx: w / 2 - scale * (box[0] + box[2]) / 2, ty: h / 2 + scale * (box[1] + box[3]) / 2 };
}
function toScreen(p) { const v = state.view; return [v.tx + p[0] * v.scale, v.ty - p[1] * v.scale]; }

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!state.view) fitView();
  draw();
}

function strokePolyline(pts) {
  ctx.beginPath();
  pts.forEach((p, i) => { const s = toScreen(p); if (i === 0) ctx.moveTo(s[0], s[1]); else ctx.lineTo(s[0], s[1]); });
  ctx.stroke();
}

function draw() {
  const m = current();
  const css = getComputedStyle(document.documentElement);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  if (state.edges) {
    ctx.strokeStyle = css.getPropertyValue("--edge"); ctx.lineWidth = 1; ctx.setLineDash([]);
    m.edges.forEach(strokePolyline);
  }
  const focusZone = state.selected ?? (state.hover ? state.hover.zone : null);
  const dimmed = focusZone !== null && focusZone >= 0;
  const base = Math.max(1.4, Math.min(4, 0.9 * state.view.scale));
  if (state.junction) {
    ctx.setLineDash([4, 4]); ctx.lineWidth = base * 0.7;
    ctx.globalAlpha = dimmed ? 0.35 : 0.9;
    for (const l of m.lanes) {
      if (l.zone >= 0) continue;
      ctx.strokeStyle = state.mode === "draw" && l.kmh_draw !== null ? limitColor(l.kmh_draw) : css.getPropertyValue("--junction");
      strokePolyline(l.pts);
    }
  }
  ctx.setLineDash([]);
  for (const lane of m.lanes) {
    if (lane.zone < 0) continue;
    const inFocus = dimmed && lane.zone === focusZone;
    ctx.globalAlpha = dimmed && !inFocus ? 0.22 : 1;
    ctx.lineWidth = inFocus ? base * 1.6 : base;
    ctx.strokeStyle = laneColor(lane);
    strokePolyline(lane.pts);
  }
  if (state.hover) {
    ctx.globalAlpha = 1; ctx.lineWidth = base * 2.2; ctx.strokeStyle = css.getPropertyValue("--accent");
    strokePolyline(state.hover.pts);
  }
  ctx.globalAlpha = 1;
}

function distToSegment(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}
function pickLane(sx, sy) {
  let best = null, bestD = 8;
  for (const lane of current().lanes) {
    if (lane.zone < 0 && !state.junction) continue;
    const scr = lane.pts.map(toScreen);
    for (let i = 1; i < scr.length; i++) {
      const d = distToSegment([sx, sy], scr[i - 1], scr[i]);
      if (d < bestD) { bestD = d; best = lane; }
    }
  }
  return best;
}

function showTooltip(lane, x, y) {
  if (!lane) { tooltip.hidden = true; return; }
  const z = lane.zone >= 0 ? zoneOf(lane.zone) : null;
  const drawText = lane.kmh_draw === null ? "" : `  →  draw ${lane.kmh_draw} km/h`;
  const lines = [`lane ${lane.id}  ·  limit ${lane.kmh} km/h${drawText}  ·  ${lane.length} m`];
  if (z) lines.push(`zone ${z.idx}: ${z.lanes} lanes, extent ${z.extent} m, ${z.length} m of lane`);
  else lines.push("junction lane · no zone (inherits from entry lanes)");
  tooltip.textContent = ""; lines.forEach(t => { const d = document.createElement("div"); d.textContent = t; tooltip.appendChild(d); });
  tooltip.style.left = `${x}px`; tooltip.style.top = `${y}px`; tooltip.hidden = false;
}

let drag = null, moved = false;
canvas.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, tx: state.view.tx, ty: state.view.ty }; moved = false; canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener("pointermove", e => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    state.view.tx = drag.tx + dx; state.view.ty = drag.ty + dy;
    tooltip.hidden = true; requestAnimationFrame(draw); return;
  }
  const lane = pickLane(sx, sy);
  if (lane !== state.hover) { state.hover = lane; requestAnimationFrame(draw); }
  showTooltip(lane, sx, sy);
});
canvas.addEventListener("pointerup", e => {
  canvas.releasePointerCapture(e.pointerId);
  if (!moved) {
    const lane = state.hover;
    selectZone(lane && lane.zone >= 0 ? lane.zone : null, false);
  }
  drag = null;
});
canvas.addEventListener("pointerleave", () => { state.hover = null; tooltip.hidden = true; draw(); });
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const factor = Math.exp(-e.deltaY * 0.0015);
  const v = state.view;
  v.tx = sx - (sx - v.tx) * factor; v.ty = sy - (sy - v.ty) * factor; v.scale *= factor;
  requestAnimationFrame(draw);
}, { passive: false });
window.addEventListener("keydown", e => { if (e.key === "Escape") selectZone(null, false); if (e.key === "f") { fitView(); draw(); } });

function selectZone(idx, zoom) {
  state.selected = idx;
  document.querySelectorAll(".zone-table tr[data-zone]").forEach(tr => tr.setAttribute("aria-selected", String(Number(tr.dataset.zone) === idx)));
  if (idx !== null && zoom) { fitBox(zoneOf(idx).bbox, 60); }
  if (idx !== null) { const row = document.querySelector(`.zone-table tr[data-zone="${idx}"]`); if (row) row.scrollIntoView({ block: "nearest" }); }
  draw();
}

function median(a) { const s = [...a].sort((x, y) => x - y); return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : 0; }

function renderPanel() {
  const m = current();
  const junction = m.lanes.filter(l => l.zone < 0).length;
  const extents = m.zones.map(z => z.extent);
  const stats = [
    ["zones", m.zones.length], ["lanes", m.lanes.length],
    ["junction lanes", junction], ["median extent", `${median(extents)} m`],
    ["min extent", `${extents.length ? Math.min(...extents) : 0} m`], ["max extent", `${extents.length ? Math.max(...extents) : 0} m`],
  ];
  document.getElementById("stats").innerHTML = stats.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");

  const binW = 50, maxExt = Math.max(100, ...extents), nb = Math.ceil(maxExt / binW);
  const counts = new Array(nb).fill(0);
  extents.forEach(x => { counts[Math.min(nb - 1, Math.floor(x / binW))]++; });
  const W = 300, H = 56, peak = Math.max(1, ...counts), bw = W / nb;
  let svg = `<svg viewBox="0 0 ${W} ${H + 14}" preserveAspectRatio="none">`;
  counts.forEach((c, i) => { const h = (c / peak) * H; svg += `<rect x="${(i * bw + 1).toFixed(1)}" y="${(H - h).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${h.toFixed(1)}"></rect>`; });
  for (let i = 0; i <= nb; i += Math.max(1, Math.round(nb / 5))) svg += `<text x="${(i * bw).toFixed(1)}" y="${H + 11}">${i * binW}</text>`;
  svg += "</svg>";
  document.getElementById("hist").innerHTML = `<div class="k">zone extent (bbox diagonal, m) — count per ${binW} m bin</div>${svg}`;

  const rows = [...m.zones].sort((a, b) => a.extent - b.extent).map(z =>
    `<tr data-zone="${z.idx}" tabindex="0" aria-selected="false"><td><i style="background:${zoneColor(z.idx)}"></i>${z.idx}</td><td>${z.kmh.join("/")}</td><td>${z.kmh_draw.length ? z.kmh_draw.join("/") : "–"}</td><td>${z.lanes}</td><td>${z.extent}</td><td>${z.length}</td></tr>`
  ).join("");
  document.getElementById("rows").innerHTML = rows;
  document.querySelectorAll(".zone-table tr[data-zone]").forEach(tr => {
    tr.addEventListener("click", () => selectZone(Number(tr.dataset.zone), true));
    tr.addEventListener("keydown", e => { if (e.key === "Enter") selectZone(Number(tr.dataset.zone), true); });
    tr.addEventListener("mouseenter", () => { state.hover = null; state.selected = state.selected ?? null; tr.classList.add("hot"); });
  });
  renderLegend();
}

function renderLegend() {
  const el = document.getElementById("legend");
  if (state.mode === "limit") {
    const kmhs = [...new Set(current().lanes.filter(l => l.zone >= 0).map(l => l.kmh))].sort((a, b) => a - b);
    el.innerHTML = kmhs.map(k => `<span><i style="background:${limitColor(k)}"></i>${k} km/h</span>`).join("") + `<span><i style="background:var(--junction)"></i>junction lane</span>`;
  } else if (state.mode === "draw") {
    const steps = [10, 30, 50, 70, 90, 110, 130];
    el.innerHTML = steps.map(k => `<span><i style="background:${limitColor(k)}"></i>${k} km/h</span>`).join("") + `<span><i style="background:var(--junction)"></i>junction lane (drawn, min over entries)</span>`;
  } else {
    el.innerHTML = `<span><i style="background:${zoneColor(3)}"></i>one hue per zone</span><span><i style="background:var(--junction)"></i>junction lane (no zone)</span>`;
  }
}

function selectMap(i) {
  state.map = i; state.selected = null; state.hover = null; state.view = null;
  document.querySelectorAll(".tabs button").forEach((b, j) => b.setAttribute("aria-pressed", String(i === j)));
  fitView(); renderPanel(); draw();
}

const tabs = document.querySelector(".tabs");
MAPS.forEach((m, i) => {
  const b = document.createElement("button"); b.textContent = m.name; b.type = "button";
  b.setAttribute("aria-pressed", String(i === 0)); b.addEventListener("click", () => selectMap(i)); tabs.appendChild(b);
});
document.getElementById("draw-btn").hidden = !hasDraw();
if (hasDraw()) {
  state.mode = "draw";
  document.querySelectorAll(".seg button").forEach(x => x.setAttribute("aria-pressed", String(x.dataset.mode === "draw")));
}
document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => {
  state.mode = b.dataset.mode;
  document.querySelectorAll(".seg button").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  renderLegend(); draw();
}));
document.getElementById("junction").addEventListener("change", e => { state.junction = e.target.checked; draw(); });
document.getElementById("edges").addEventListener("change", e => { state.edges = e.target.checked; draw(); });
document.getElementById("fit").addEventListener("click", () => { fitView(); draw(); });
window.addEventListener("resize", resize);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderPanel(); draw(); });
new MutationObserver(() => { renderPanel(); draw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
renderPanel();
resize();
"""


def build_html(maps: list[dict], fragment: bool, draw_note: str) -> str:
    payload = json.dumps(maps, separators=(",", ":"))
    total_zones = sum(len(m["zones"]) for m in maps)
    head = (
        "<title>CARLA Speed Zones</title>\n"
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
        f"<style>{STYLE}</style>\n"
    )
    body = f"""
<div class="app">
  <header>
    <h1>CARLA Speed Zones</h1>
    <div class="tabs" role="tablist" aria-label="Maps"></div>
    <div class="seg" role="group" aria-label="Colour by">
      <button type="button" data-mode="zone" aria-pressed="true">colour by zone</button>
      <button type="button" data-mode="limit" aria-pressed="false">colour by posted limit</button>
      <button type="button" data-mode="draw" aria-pressed="false" id="draw-btn">colour by random draw</button>
    </div>
    <div class="controls">
      <label><input type="checkbox" id="junction" checked> junction lanes</label>
      <label><input type="checkbox" id="edges" checked> road edges</label>
      <button type="button" id="fit">fit (f)</button>
    </div>
  </header>
  <div class="stage">
    <canvas id="map" aria-label="Lane map coloured by speed zone"></canvas>
    <div class="legend" id="legend"></div>
    <div class="tooltip" id="tooltip" hidden></div>
    <div class="hint">drag to pan · wheel to zoom · click a lane to isolate its zone · Esc clears · {len(maps)} maps, {total_zones} zones</div>
  </div>
  <aside>
    <div class="stats" id="stats"></div>
    <div class="hist" id="hist"></div>
    <div class="table-wrap">
      <table class="zone-table">
        <thead><tr><th>zone</th><th>km/h</th><th>draw</th><th>lanes</th><th>extent m</th><th>lane m</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div class="note">{draw_note}</div>
    <div class="note">Zone = parallel lanes + both directions of a road + consecutive road pieces up to the next junction or limit change; zones under the converter's minimum extent were merged through junctions into their smallest same-limit neighbour. Rows sorted by extent.</div>
  </aside>
</div>
<script>window.__SPEED_ZONE_MAPS__ = {payload};</script>
<script>{SCRIPT}</script>
"""
    if fragment:
        return head + body
    return f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n{head}</head>\n<body>{body}</body>\n</html>\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin-dir", required=True, help="Directory of *.bin map files with a SPDZONE1 section")
    parser.add_argument("--output", default=None, help="Output .html (default: <bin-dir>/speed_zones.html)")
    parser.add_argument("--fragment", action="store_true", help="Omit the <html>/<head>/<body> wrapper (artifact publishing)")
    parser.add_argument("--draw-seed", type=int, default=None, help="Overlay one episode draw of the C speed-limit sampler")
    parser.add_argument("--draw-delta-mps", type=float, default=9.72)
    parser.add_argument("--draw-min-mps", type=float, default=1.39)
    parser.add_argument("--draw-max-mps", type=float, default=36.11)
    args = parser.parse_args()

    bin_dir = pathlib.Path(args.bin_dir).expanduser()
    bin_files = sorted(bin_dir.glob("*.bin"))
    if not bin_files:
        sys.exit(f"no .bin files in {bin_dir}")
    draws = {
        f: sample_limit_draw(f, args.draw_seed, args.draw_delta_mps, args.draw_min_mps, args.draw_max_mps) for f in bin_files
    } if args.draw_seed is not None else {}
    maps = [collect_map(f, draws.get(f)) for f in bin_files]
    draw_note = (
        f"Random draw: episode seed {args.draw_seed}, one offset per zone uniform in ±{args.draw_delta_mps * MPS_TO_KMH:.0f} km/h, "
        f"clipped to {args.draw_min_mps * MPS_TO_KMH:.0f}–{args.draw_max_mps * MPS_TO_KMH:.0f} km/h; junction lanes take the min over their entries."
        if args.draw_seed is not None
        else "No random draw overlaid (pass --draw-seed)."
    )
    missing = [m["name"] for m in maps if not m["has_zone_section"]]
    if missing:
        print(f"warning: no SPDZONE1 section in {missing}", file=sys.stderr)
    output = pathlib.Path(args.output) if args.output else bin_dir / "speed_zones.html"
    output.write_text(build_html(maps, args.fragment, draw_note))
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {len(maps)} maps, {sum(len(m['zones']) for m in maps)} zones)")


if __name__ == "__main__":
    main()
