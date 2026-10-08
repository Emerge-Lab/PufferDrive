#!/usr/bin/env python
"""Render PufferDrive map .bin files into one self-contained HTML map inspector (lanes, road lines, road edges),
with click-to-mark: every click drops a numbered mark and appends a line with the bin coordinates and the nearest
road edge / lane to a report you can copy. Optionally overlays the road edges of a second bin directory for comparison.

    python scripts/visualize_map_bins.py --bin-dir ~/ordnung/data/CARLA/puffer_bins_nd_shoulders \\
        --compare-dir ~/ordnung/data/CARLA/puffer_bins_stops_tol01 --output /tmp/map_inspector.html

Keys: 1/2/3 mark category, u undo, c copy report, f fit, Esc clear. Drag pans, wheel zooms.
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "data_utils"))
from mirror_map_bin import read_bin  # noqa: E402

LANE_TYPE_MAX = 9
ROAD_LINE_TYPES = range(10, 20)
ROAD_EDGE_TYPES = (20, 21, 22)


def _polyline(road):
    return [[round(float(x), 2), round(float(y), 2)] for x, y in zip(road["x"], road["y"])]


def _edges(data):
    return [{"id": int(r["id"]), "type": int(r["type"]), "pts": _polyline(r)} for r in data["roads"] if r["type"] in ROAD_EDGE_TYPES]


def collect_map(bin_file: pathlib.Path, compare_file: pathlib.Path | None) -> dict:
    data = read_bin(bin_file)
    lanes, lines = [], []
    for road in data["roads"]:
        if 0 <= road["type"] <= LANE_TYPE_MAX:
            lanes.append(
                {
                    "id": int(road["id"]),
                    "junction": bool(data.get("has_zone_section")) and road.get("speed_zone_idx", 0) < 0,
                    "length": round(float(road["length"]), 1),
                    "pts": _polyline(road),
                }
            )
        elif road["type"] in ROAD_LINE_TYPES:
            lines.append(_polyline(road))
    compare = _edges(read_bin(compare_file)) if compare_file is not None else []
    return {
        "name": bin_file.stem.split("__")[-1],
        "file": bin_file.name,
        "lanes": lanes,
        "lines": lines,
        "edges": _edges(data),
        "compare": compare,
    }


STYLE = """
:root {
  --ground: #F2F3F1; --panel: #FAFAF8; --line: #D9DCD8; --text: #1C2126; --muted: #6E7882;
  --accent: #C9552B; --accent-soft: rgba(201, 85, 43, 0.12); --lane: #B9BEB9; --junction: #D3D7D2; --roadline: #E3E6E2;
  --edge: #C9552B; --median: #8C3D1E; --compare: #E6A23C; --row-hover: #ECEEEA; --tooltip: #1C2126; --tooltip-text: #F2F3F1;
  --cat1: #C9552B; --cat2: #2F6FB3; --cat3: #B07C1A;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #141719; --panel: #1B1F22; --line: #2C3237; --text: #E4E8EA; --muted: #8B95A0;
    --accent: #E2764B; --accent-soft: rgba(226, 118, 75, 0.18); --lane: #4A5159; --junction: #363C42; --roadline: #2A3035;
    --edge: #E2764B; --median: #C45A33; --compare: #E6B45C; --row-hover: #23282C; --tooltip: #E4E8EA; --tooltip-text: #141719;
    --cat1: #E2764B; --cat2: #6FA5E6; --cat3: #D9A43A;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --ground: #141719; --panel: #1B1F22; --line: #2C3237; --text: #E4E8EA; --muted: #8B95A0;
  --accent: #E2764B; --accent-soft: rgba(226, 118, 75, 0.18); --lane: #4A5159; --junction: #363C42; --roadline: #2A3035;
  --edge: #E2764B; --median: #C45A33; --compare: #E6B45C; --row-hover: #23282C; --tooltip: #E4E8EA; --tooltip-text: #141719;
  --cat1: #E2764B; --cat2: #6FA5E6; --cat3: #D9A43A;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body { margin: 0; background: var(--ground); color: var(--text); font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; font-size: 13px; line-height: 1.4; }
.app { display: grid; grid-template-rows: auto 1fr; grid-template-columns: 1fr 360px; height: 100vh; min-height: 560px; }
header { grid-column: 1 / -1; display: flex; align-items: center; gap: 16px; padding: 10px 16px; border-bottom: 1px solid var(--line); background: var(--panel); flex-wrap: wrap; }
header h1 { font-size: 15px; font-weight: 600; margin: 0; white-space: nowrap; }
.tabs { display: flex; gap: 4px; flex-wrap: wrap; }
.tabs button, .seg button, .controls button { font: inherit; color: var(--text); background: transparent; border: 1px solid var(--line); border-radius: 4px; padding: 4px 10px; cursor: pointer; }
.tabs button[aria-pressed="true"], .seg button[aria-pressed="true"] { background: var(--accent-soft); border-color: var(--accent); }
.seg { display: flex; }
.seg button:first-child { border-radius: 4px 0 0 4px; }
.seg button:last-child { border-radius: 0 4px 4px 0; }
.seg button:not(:first-child) { margin-left: -1px; }
.controls { display: flex; align-items: center; gap: 12px; margin-left: auto; color: var(--muted); flex-wrap: wrap; }
.controls label { display: flex; align-items: center; gap: 5px; cursor: pointer; }
.stage { position: relative; overflow: hidden; }
canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
.hint { position: absolute; left: 12px; bottom: 10px; color: var(--muted); font-size: 12px; pointer-events: none; }
.legend { position: absolute; right: 12px; top: 10px; display: flex; flex-direction: column; gap: 3px; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; pointer-events: none; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
.tooltip { position: absolute; pointer-events: none; background: var(--tooltip); color: var(--tooltip-text); padding: 6px 9px; border-radius: 4px; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px; white-space: nowrap; transform: translate(12px, 12px); line-height: 1.5; }
.tooltip[hidden] { display: none; }
aside { border-left: 1px solid var(--line); background: var(--panel); display: flex; flex-direction: column; min-height: 0; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
.stat .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }
.stat .v { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 15px; font-variant-numeric: tabular-nums; }
.marks { overflow: auto; flex: 1; min-height: 0; padding: 6px 0; }
.mark-row { display: grid; grid-template-columns: 26px 1fr 22px; gap: 6px; align-items: start; padding: 5px 14px; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px; cursor: pointer; }
.mark-row:hover { background: var(--row-hover); }
.mark-row .n { display: inline-flex; width: 20px; height: 20px; border-radius: 50%; color: #fff; align-items: center; justify-content: center; font-size: 11px; }
.mark-row .d { color: var(--muted); }
.mark-row button { font: inherit; border: 0; background: transparent; color: var(--muted); cursor: pointer; }
.report { border-top: 1px solid var(--line); padding: 10px 14px; display: flex; flex-direction: column; gap: 6px; }
.report textarea { width: 100%; height: 120px; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; background: var(--ground); color: var(--text); border: 1px solid var(--line); border-radius: 4px; padding: 6px; resize: vertical; }
.report .row { display: flex; gap: 8px; align-items: center; color: var(--muted); font-size: 11.5px; }
.note { padding: 8px 14px; color: var(--muted); font-size: 11.5px; border-top: 1px solid var(--line); }
@media (max-width: 860px) { .app { grid-template-columns: 1fr; grid-template-rows: auto 1fr 360px; } aside { border-left: 0; border-top: 1px solid var(--line); } }
"""

SCRIPT = r"""
const MAPS = window.__MAP_INSPECTOR__.maps;
const META = window.__MAP_INSPECTOR__.meta;
const CATEGORIES = { "1": ["not_drivable", "--cat1", "NOT drivable in CARLA but drivable in the map"],
                     "2": ["missing_drivable", "--cat2", "drivable in CARLA but blocked in the map"],
                     "3": ["other", "--cat3", "other issue"] };
const state = { map: 0, category: "1", hover: null, view: null, compare: true, lines: true, lanes: true, marks: MAPS.map(() => []) };
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const css = () => getComputedStyle(document.documentElement);
const color = v => css().getPropertyValue(v).trim();
function current() { return MAPS[state.map]; }

function fitView() {
  const m = current();
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const pts of [...m.lanes.map(l => l.pts), ...m.edges.map(e => e.pts)]) for (const [x, y] of pts) {
    if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  fitBox([minX, minY, maxX, maxY], 30);
}
function fitBox(box, pad) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const scale = Math.min((w - 2 * pad) / Math.max(1, box[2] - box[0]), (h - 2 * pad) / Math.max(1, box[3] - box[1]));
  state.view = { scale, tx: w / 2 - scale * (box[0] + box[2]) / 2, ty: h / 2 + scale * (box[1] + box[3]) / 2 };
}
function toScreen(p) { const v = state.view; return [v.tx + p[0] * v.scale, v.ty - p[1] * v.scale]; }
function toWorld(sx, sy) { const v = state.view; return [(sx - v.tx) / v.scale, (v.ty - sy) / v.scale]; }

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
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.globalAlpha = 1;
  const base = Math.max(1.2, Math.min(4, 0.8 * state.view.scale));
  if (state.lines) { ctx.strokeStyle = color("--roadline"); ctx.lineWidth = Math.max(0.8, base * 0.5); ctx.setLineDash([]); m.lines.forEach(strokePolyline); }
  if (state.lanes) {
    for (const lane of m.lanes) {
      ctx.setLineDash(lane.junction ? [4, 4] : []);
      ctx.strokeStyle = lane.junction ? color("--junction") : color("--lane"); ctx.lineWidth = base * 0.8;
      strokePolyline(lane.pts);
    }
  }
  ctx.setLineDash([]);
  if (state.compare && m.compare.length) {
    ctx.strokeStyle = color("--compare"); ctx.lineWidth = base * 2.2; ctx.globalAlpha = 0.55;
    m.compare.forEach(e => strokePolyline(e.pts)); ctx.globalAlpha = 1;
  }
  for (const e of m.edges) { ctx.strokeStyle = e.type === 22 ? color("--median") : color("--edge"); ctx.lineWidth = base * 1.1; strokePolyline(e.pts); }
  if (state.hover) { ctx.lineWidth = base * 2.6; ctx.strokeStyle = color("--accent"); strokePolyline(state.hover.pts); }
  for (const mk of state.marks[state.map]) {
    const [sx, sy] = toScreen([mk.x, mk.y]);
    ctx.beginPath(); ctx.arc(sx, sy, 8, 0, 2 * Math.PI); ctx.fillStyle = color(CATEGORIES[mk.cat][1]); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "bold 10px IBM Plex Mono, ui-monospace, monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(String(mk.n), sx, sy);
  }
}

function distToSegment(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}
function polylineDistance(p, pts) {
  let best = Infinity;
  for (let i = 1; i < pts.length; i++) best = Math.min(best, distToSegment(p, pts[i - 1], pts[i]));
  return pts.length === 1 ? Math.hypot(p[0] - pts[0][0], p[1] - pts[0][1]) : best;
}
function candidates() {
  const m = current();
  const items = m.edges.map(e => ({ kind: "edge", id: e.id, type: e.type, pts: e.pts }));
  if (state.compare) items.push(...m.compare.map(e => ({ kind: "old edge", id: e.id, type: e.type, pts: e.pts })));
  if (state.lanes) items.push(...m.lanes.map(l => ({ kind: l.junction ? "junction lane" : "lane", id: l.id, type: null, pts: l.pts, length: l.length })));
  return items;
}
function pick(sx, sy) {
  const p = toWorld(sx, sy), tol = 8 / state.view.scale;
  let best = null, bestD = tol;
  for (const it of candidates()) { const d = polylineDistance(p, it.pts); if (d < bestD) { bestD = d; best = it; } }
  return best;
}
function nearest(p, kind) {
  let best = null, bestD = Infinity;
  for (const it of candidates()) {
    if (!it.kind.includes(kind)) continue;
    const d = polylineDistance(p, it.pts); if (d < bestD) { bestD = d; best = it; }
  }
  return best ? { id: best.id, type: best.type, kind: best.kind, d: bestD } : null;
}
function showTooltip(item, x, y) {
  if (!item) { tooltip.hidden = true; return; }
  const len = item.pts.reduce((acc, p, i) => i ? acc + Math.hypot(p[0] - item.pts[i - 1][0], p[1] - item.pts[i - 1][1]) : 0, 0);
  const text = item.kind.includes("edge") ? `${item.kind} ${item.id} · type ${item.type} · ${item.pts.length} pts · ${len.toFixed(0)} m`
                                         : `${item.kind} ${item.id} · ${len.toFixed(0)} m`;
  tooltip.textContent = text; tooltip.style.left = `${x}px`; tooltip.style.top = `${y}px`; tooltip.hidden = false;
}

function markLine(mk) {
  const parts = [];
  if (mk.edge) parts.push(`${mk.edge.kind} ${mk.edge.id} type ${mk.edge.type} at ${mk.edge.d.toFixed(2)} m`);
  if (mk.oldEdge) parts.push(`old edge ${mk.oldEdge.id} at ${mk.oldEdge.d.toFixed(2)} m`);
  if (mk.lane) parts.push(`${mk.lane.kind} ${mk.lane.id} at ${mk.lane.d.toFixed(2)} m`);
  return `#${mk.n} ${CATEGORIES[mk.cat][0]} at (${mk.x.toFixed(1)}, ${mk.y.toFixed(1)})  [${parts.join("; ")}]`;
}
function report() {
  const m = current(), marks = state.marks[state.map];
  const head = `Map marks for ${m.name} (PufferDrive bin coordinates, ${META.bin_dir}${META.compare_dir ? ", old edges from " + META.compare_dir : ""}):`;
  return [head, ...marks.map(mk => "  " + markLine(mk))].join("\n");
}
function renderPanel() {
  const m = current(), marks = state.marks[state.map];
  const stats = [["lanes", m.lanes.length], ["road edges", m.edges.length], ["old edges", m.compare.length], ["marks", marks.length]];
  document.getElementById("stats").innerHTML = stats.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  document.getElementById("marks").innerHTML = marks.map((mk, i) =>
    `<div class="mark-row" data-i="${i}"><span class="n" style="background:${color(CATEGORIES[mk.cat][1])}">${mk.n}</span><span>${markLine(mk).replace(/^#\d+ /, "")}<br><span class="d">${CATEGORIES[mk.cat][2]}</span></span><button type="button" title="remove" data-rm="${i}">×</button></div>`).join("");
  document.querySelectorAll(".mark-row").forEach(row => row.addEventListener("click", e => {
    const i = Number(row.dataset.i);
    if (e.target.dataset.rm !== undefined) { removeMark(i); return; }
    const mk = marks[i]; fitBox([mk.x - 40, mk.y - 40, mk.x + 40, mk.y + 40], 20); draw();
  }));
  document.getElementById("report").value = report();
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.cat === state.category)));
}
function addMark(x, y) {
  const marks = state.marks[state.map], p = [x, y];
  const edge = nearest(p, "edge"), lane = nearest(p, "lane");
  const own = current().edges.map(e => ({ id: e.id, type: e.type, d: polylineDistance(p, e.pts) })).sort((a, b) => a.d - b.d)[0];
  const old = state.compare ? current().compare.map(e => ({ id: e.id, type: e.type, d: polylineDistance(p, e.pts) })).sort((a, b) => a.d - b.d)[0] : null;
  marks.push({ n: marks.length + 1, cat: state.category, x, y,
               edge: own && own.d < 5 ? { ...own, kind: "edge" } : null,
               oldEdge: old && old.d < 5 ? old : null,
               lane: lane && lane.d < 8 ? lane : null });
  renderPanel(); draw();
}
function removeMark(i) {
  const marks = state.marks[state.map]; marks.splice(i, 1); marks.forEach((mk, k) => mk.n = k + 1); renderPanel(); draw();
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
  const item = pick(sx, sy);
  if (item !== state.hover) { state.hover = item; requestAnimationFrame(draw); }
  showTooltip(item, sx, sy);
  const w = toWorld(sx, sy); document.getElementById("coords").textContent = `bin (${w[0].toFixed(1)}, ${w[1].toFixed(1)})`;
});
canvas.addEventListener("pointerup", e => {
  canvas.releasePointerCapture(e.pointerId);
  if (!moved && e.button === 0) { const rect = canvas.getBoundingClientRect(); const w = toWorld(e.clientX - rect.left, e.clientY - rect.top); addMark(w[0], w[1]); }
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
window.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (CATEGORIES[e.key]) { state.category = e.key; renderPanel(); }
  else if (e.key === "u") { const marks = state.marks[state.map]; if (marks.length) removeMark(marks.length - 1); }
  else if (e.key === "f") { fitView(); draw(); }
  else if (e.key === "c") { copyReport(); }
  else if (e.key === "Escape") { state.hover = null; tooltip.hidden = true; draw(); }
});
function copyReport() {
  const ta = document.getElementById("report"); ta.select();
  const done = () => { document.getElementById("copied").textContent = "copied"; setTimeout(() => { document.getElementById("copied").textContent = ""; }, 1500); };
  if (navigator.clipboard) navigator.clipboard.writeText(ta.value).then(done, () => { document.execCommand("copy"); done(); }); else { document.execCommand("copy"); done(); }
}
function selectMap(i) {
  state.map = i; state.hover = null; state.view = null;
  document.querySelectorAll(".tabs button").forEach((b, j) => b.setAttribute("aria-pressed", String(i === j)));
  fitView(); renderPanel(); draw();
}
const tabs = document.querySelector(".tabs");
MAPS.forEach((m, i) => {
  const b = document.createElement("button"); b.textContent = m.name; b.type = "button";
  b.setAttribute("aria-pressed", String(i === 0)); b.addEventListener("click", () => selectMap(i)); tabs.appendChild(b);
});
document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => { state.category = b.dataset.cat; renderPanel(); }));
document.getElementById("compare").addEventListener("change", e => { state.compare = e.target.checked; draw(); });
document.getElementById("lines").addEventListener("change", e => { state.lines = e.target.checked; draw(); });
document.getElementById("lanes").addEventListener("change", e => { state.lanes = e.target.checked; draw(); });
document.getElementById("fit").addEventListener("click", () => { fitView(); draw(); });
document.getElementById("copy").addEventListener("click", copyReport);
document.getElementById("clear").addEventListener("click", () => { state.marks[state.map] = []; renderPanel(); draw(); });
document.getElementById("legend").innerHTML = `<span><i style="background:var(--edge)"></i>road edge (bin)</span><span><i style="background:var(--median)"></i>median edge</span><span><i style="background:var(--compare)"></i>old edge (compare)</span><span><i style="background:var(--lane)"></i>lane centerline</span><span><i style="background:var(--junction)"></i>junction lane</span>`;
window.addEventListener("resize", resize);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderPanel(); draw(); });
new MutationObserver(() => { renderPanel(); draw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
renderPanel();
resize();
"""


def build_html(maps: list[dict], meta: dict) -> str:
    payload = json.dumps({"maps": maps, "meta": meta}, separators=(",", ":"))
    head = (
        "<title>PufferDrive Map Inspector</title>\n"
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
        f"<style>{STYLE}</style>\n"
    )
    compare_note = f" · old edges: {meta['compare_dir']}" if meta.get("compare_dir") else ""
    body = f"""
<div class="app">
  <header>
    <h1>PufferDrive Map Inspector</h1>
    <div class="tabs" role="tablist" aria-label="Maps"></div>
    <div class="seg" role="group" aria-label="Mark category">
      <button type="button" data-cat="1" aria-pressed="true">1 not drivable</button>
      <button type="button" data-cat="2" aria-pressed="false">2 missing drivable</button>
      <button type="button" data-cat="3" aria-pressed="false">3 other</button>
    </div>
    <div class="controls">
      <label><input type="checkbox" id="lanes" checked> lanes</label>
      <label><input type="checkbox" id="lines" checked> road lines</label>
      <label><input type="checkbox" id="compare" checked> old edges</label>
      <button type="button" id="fit">fit (f)</button>
      <span id="coords"></span>
    </div>
  </header>
  <div class="stage">
    <canvas id="map" aria-label="PufferDrive map with road edges"></canvas>
    <div class="legend" id="legend"></div>
    <div class="tooltip" id="tooltip" hidden></div>
    <div class="hint">drag to pan · wheel to zoom · click to drop a mark (1/2/3 category) · u undo · c copy report · bins: {meta['bin_dir']}{compare_note}</div>
  </div>
  <aside>
    <div class="stats" id="stats"></div>
    <div class="marks" id="marks"></div>
    <div class="report">
      <div class="row"><button type="button" id="copy">copy report (c)</button><button type="button" id="clear">clear marks</button><span id="copied"></span></div>
      <textarea id="report" readonly></textarea>
    </div>
    <div class="note">Marks are in PufferDrive bin coordinates (what the C sim and the obs replays use). Hover shows element ids; a mark lists the nearest bin road edge, the nearest old edge and the nearest lane.</div>
  </aside>
</div>
<script>window.__MAP_INSPECTOR__ = {payload};</script>
<script>{SCRIPT}</script>
"""
    return f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n{head}</head>\n<body>{body}</body>\n</html>\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin-dir", required=True, help="Directory of PufferDrive *.bin map files to inspect")
    parser.add_argument("--compare-dir", default=None, help="Second bin directory whose road edges are overlaid in orange")
    parser.add_argument("--towns", nargs="*", default=None, help="Subset of towns, e.g. Town04 Town06 (default: all bins)")
    parser.add_argument("--output", default=None, help="Output .html (default: <bin-dir>/map_inspector.html)")
    args = parser.parse_args()

    bin_dir = pathlib.Path(args.bin_dir).expanduser()
    compare_dir = pathlib.Path(args.compare_dir).expanduser() if args.compare_dir else None
    bin_files = sorted(bin_dir.glob("*.bin"))
    if args.towns:
        bin_files = [f for f in bin_files if f.stem.split("__")[-1] in set(args.towns)]
    if not bin_files:
        sys.exit(f"no matching .bin files in {bin_dir}")
    maps = []
    for bin_file in bin_files:
        compare_file = compare_dir / bin_file.name if compare_dir and (compare_dir / bin_file.name).exists() else None
        maps.append(collect_map(bin_file, compare_file))
    meta = {"bin_dir": str(bin_dir), "compare_dir": str(compare_dir) if compare_dir else None}
    output = pathlib.Path(args.output) if args.output else bin_dir / "map_inspector.html"
    output.write_text(build_html(maps, meta))
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, {len(maps)} maps)")


if __name__ == "__main__":
    main()
