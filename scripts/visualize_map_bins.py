#!/usr/bin/env python
"""Render PufferDrive map .bin files into one self-contained HTML map inspector (lanes, road lines, road edges),
with click-to-mark: every click drops a numbered mark and appends a line with the bin coordinates and the nearest
road edge / lane to a report you can copy. Optionally overlays the road edges of a second bin directory for comparison.
The view is a 3D orbit camera: right/middle drag tilts and turns the map, a slider exaggerates heights, edges can be
coloured by height, and road edges whose height steps between vertical layers (bridge seams) are listed and ringed.

    python scripts/visualize_map_bins.py --bin-dir ~/ordnung/data/CARLA/puffer_bins_nd_shoulders \\
        --compare-dir ~/ordnung/data/CARLA/puffer_bins_stops_tol01 --output /tmp/map_inspector.html

Keys: 1/2/3 mark category, u undo, c copy report, f fit, t top view, Esc clear. Left drag pans, right/middle drag
rotates, wheel zooms.
"""

import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "data_utils"))
from mirror_map_bin import read_bin  # noqa: E402

LANE_TYPE_MAX = 9
ROAD_LINE_TYPES = range(10, 20)
ROAD_EDGE_TYPES = (20, 21, 22)
SEAM_MIN_DZ_M = 1.5  # a road-edge segment climbing more than this ...
SEAM_MIN_GRADE = 0.5  # ... at a grade steeper than any road steps between vertical layers


def _polyline(road):
    return [
        [round(float(x), 2), round(float(y), 2), round(float(z), 2)] for x, y, z in zip(road["x"], road["y"], road["z"])
    ]


def _seams(edges, kind):
    seams = []
    for edge in edges:
        for (x0, y0, z0), (x1, y1, z1) in zip(edge["pts"], edge["pts"][1:]):
            dz = abs(z1 - z0)
            if dz > SEAM_MIN_DZ_M and dz > SEAM_MIN_GRADE * max(math.hypot(x1 - x0, y1 - y0), 0.05):
                seams.append(
                    {
                        "kind": kind,
                        "id": edge["id"],
                        "x": round((x0 + x1) / 2, 2),
                        "y": round((y0 + y1) / 2, 2),
                        "z0": z0,
                        "z1": z1,
                    }
                )
    return seams


def _z_range(polylines):
    zs = [p[2] for pts in polylines for p in pts]
    return [min(zs), max(zs)] if zs else [0.0, 0.0]


def _edges(data):
    return [
        {"id": int(r["id"]), "type": int(r["type"]), "pts": _polyline(r)}
        for r in data["roads"]
        if r["type"] in ROAD_EDGE_TYPES
    ]


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
    edges = _edges(data)
    return {
        "name": bin_file.stem.split("__")[-1],
        "file": bin_file.name,
        "lanes": lanes,
        "lines": lines,
        "edges": edges,
        "compare": compare,
        "seams": _seams(edges, "edge"),
        "compareSeams": _seams(compare, "old edge"),
        "zrange": _z_range([lane["pts"] for lane in lanes] + [edge["pts"] for edge in edges]),
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
.seams { max-height: 170px; overflow: auto; border-bottom: 1px solid var(--line); padding: 6px 0; }
.seam-row { padding: 4px 14px; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11.5px; cursor: pointer; }
.seam-row:hover { background: var(--row-hover); }
.seam-row.old { color: var(--muted); }
.seam-row.none { color: var(--muted); cursor: default; }
.controls input[type="range"] { width: 90px; accent-color: var(--accent); }
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
const MAX_PITCH = 80 * Math.PI / 180;
const ORBIT_RAD_PER_PX = 0.008;
const FOV_DEG = 50;
const NEAR_FRACTION = 0.05;
const PICK_PX = 8;
const VERTEX_PICK_PX = 12;
const state = { map: 0, category: "1", hover: null, view: null, compare: true, lines: true, lanes: true, heightColor: false, zscale: 1, marks: MAPS.map(() => []) };
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");
const css = () => getComputedStyle(document.documentElement);
const color = v => css().getPropertyValue(v).trim();
function current() { return MAPS[state.map]; }

// Perspective orbit camera: the target (cx, cy, cz) sits at the screen centre, scale is px per metre at the target,
// yaw turns the map about the vertical axis and pitch tilts the camera (0 = top-down).
function cameraDistance(v) { return (canvas.clientHeight / v.scale) / (2 * Math.tan(FOV_DEG * Math.PI / 360)); }
function project(p, v) {
  const ca = Math.cos(v.yaw), sa = Math.sin(v.yaw), cb = Math.cos(v.pitch), sb = Math.sin(v.pitch);
  const rx = p[0] - v.cx, ry = p[1] - v.cy, rz = ((p[2] || 0) - v.cz) * state.zscale;
  const x1 = rx * ca - ry * sa, y1 = rx * sa + ry * ca;
  const dist = cameraDistance(v), depth = dist + y1 * sb - rz * cb;
  if (depth < NEAR_FRACTION * dist) return null;
  const f = v.scale * dist / depth;
  return [canvas.clientWidth / 2 + f * x1, canvas.clientHeight / 2 - f * (y1 * cb + rz * sb)];
}
function toScreen(p) { return project(p, state.view); }
function toWorld(sx, sy, z = 0) {
  const v = state.view, ca = Math.cos(v.yaw), sa = Math.sin(v.yaw), cb = Math.cos(v.pitch), sb = Math.sin(v.pitch);
  const dist = cameraDistance(v), rz = (z - v.cz) * state.zscale;
  const u = (sx - canvas.clientWidth / 2) / (v.scale * dist), w = (canvas.clientHeight / 2 - sy) / (v.scale * dist);
  const denom = w * sb - cb;
  if (Math.abs(denom) < 1e-9) return null;
  const y1 = (rz * (sb + w * cb) - w * dist) / denom;
  const depth = dist + y1 * sb - rz * cb;
  if (depth <= 0) return null;
  const x1 = u * depth;
  return [v.cx + x1 * ca + y1 * sa, v.cy - x1 * sa + y1 * ca, z];
}
function groundOffset(sx, sy) {
  const w = toWorld(sx, sy, state.view.cz);
  return w ? [w[0] - state.view.cx, w[1] - state.view.cy] : null;
}
function allPoints(m) { const out = []; for (const l of m.lanes) out.push(...l.pts); for (const e of m.edges) out.push(...e.pts); return out; }
function fitPoints(points, pad) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const yaw = state.view ? state.view.yaw : 0, pitch = state.view ? state.view.pitch : 0;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, sumZ = 0;
  for (const p of points) {
    if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0]; if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1]; sumZ += p[2] || 0;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = points.length ? sumZ / points.length : 0;
  const ca = Math.cos(yaw), sa = Math.sin(yaw), cb = Math.cos(pitch), sb = Math.sin(pitch);
  let ex = 1, ey = 1;
  for (const p of points) {
    const rx = p[0] - cx, ry = p[1] - cy, rz = ((p[2] || 0) - cz) * state.zscale;
    ex = Math.max(ex, Math.abs(rx * ca - ry * sa)); ey = Math.max(ey, Math.abs((rx * sa + ry * ca) * cb + rz * sb));
  }
  const scale = Math.min((w - 2 * pad) / (2 * ex), (h - 2 * pad) / (2 * ey)) * (1 - 0.35 * sb);
  state.view = { cx, cy, cz, scale, yaw, pitch };
}
function fitView() { fitPoints(allPoints(current()), 30); }
function fitBox(box, pad, z = 0) { fitPoints([[box[0], box[1], z], [box[2], box[1], z], [box[2], box[3], z], [box[0], box[3], z]], pad); }
function orbit(dyaw, dpitch) { const v = state.view; v.yaw += dyaw; v.pitch = Math.max(0, Math.min(MAX_PITCH, v.pitch + dpitch)); }
function topView() { state.view.yaw = 0; state.view.pitch = 0; }
function setZScale(z) { state.zscale = z; document.getElementById("zscale-val").textContent = `${z}×`; }
function viewLabel() { const v = state.view, deg = r => Math.round(r * 180 / Math.PI); return v.pitch < 1e-3 ? "top view" : `yaw ${deg(v.yaw)}° · tilt ${deg(v.pitch)}°`; }

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
  let pen = false;
  for (const p of pts) {
    const s = toScreen(p);
    if (!s) { pen = false; continue; }
    if (pen) ctx.lineTo(s[0], s[1]); else ctx.moveTo(s[0], s[1]);
    pen = true;
  }
  ctx.stroke();
}
function heightColor(t) { return `hsl(${Math.round(230 - 230 * Math.max(0, Math.min(1, t)))}, 85%, 45%)`; }
function strokeByHeight(pts) {
  const [z0, z1] = current().zrange, span = Math.max(1e-6, z1 - z0);
  let prev = toScreen(pts[0]);
  for (let i = 1; i < pts.length; i++) {
    const s = toScreen(pts[i]);
    if (prev && s) {
      ctx.strokeStyle = heightColor(((pts[i - 1][2] + pts[i][2]) / 2 - z0) / span);
      ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(s[0], s[1]); ctx.stroke();
    }
    prev = s;
  }
}
function drawSeam(seam, dashed) {
  const s = toScreen([seam.x, seam.y, (seam.z0 + seam.z1) / 2]);
  if (!s) return;
  ctx.setLineDash(dashed ? [3, 3] : []); ctx.lineWidth = 2; ctx.strokeStyle = color("--accent");
  ctx.beginPath(); ctx.arc(s[0], s[1], 7, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
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
  ctx.lineWidth = base * 1.1;
  for (const e of m.edges) {
    if (state.heightColor) { strokeByHeight(e.pts); continue; }
    ctx.strokeStyle = e.type === 22 ? color("--median") : color("--edge"); strokePolyline(e.pts);
  }
  if (state.hover) { ctx.lineWidth = base * 2.6; ctx.strokeStyle = color("--accent"); strokePolyline(state.hover.pts); }
  m.seams.forEach(s => drawSeam(s, false));
  if (state.compare) m.compareSeams.forEach(s => drawSeam(s, true));
  for (const mk of state.marks[state.map]) {
    const s = toScreen([mk.x, mk.y, mk.z]);
    if (!s) continue;
    const [sx, sy] = s;
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
function screenDistance(sx, sy, pts) {
  let prev = toScreen(pts[0]), best = prev ? Math.hypot(sx - prev[0], sy - prev[1]) : Infinity;
  for (let i = 1; i < pts.length; i++) {
    const s = toScreen(pts[i]);
    if (prev && s) best = Math.min(best, distToSegment([sx, sy], prev, s));
    prev = s;
  }
  return best;
}
function candidates() {
  const m = current();
  const items = m.edges.map(e => ({ kind: "edge", id: e.id, type: e.type, pts: e.pts }));
  if (state.compare) items.push(...m.compare.map(e => ({ kind: "old edge", id: e.id, type: e.type, pts: e.pts })));
  if (state.lanes) items.push(...m.lanes.map(l => ({ kind: l.junction ? "junction lane" : "lane", id: l.id, type: null, pts: l.pts, length: l.length })));
  return items;
}
function pick(sx, sy) {
  let best = null, bestD = PICK_PX;
  for (const it of candidates()) { const d = screenDistance(sx, sy, it.pts); if (d < bestD) { bestD = d; best = it; } }
  return best;
}
function pickVertex(sx, sy) {
  let best = null, bestD = VERTEX_PICK_PX;
  for (const it of candidates()) for (const p of it.pts) { const s = toScreen(p); if (!s) continue; const d = Math.hypot(sx - s[0], sy - s[1]); if (d < bestD) { bestD = d; best = p; } }
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
  const zs = item.pts.map(p => p[2]), zText = `z ${Math.min(...zs).toFixed(1)}..${Math.max(...zs).toFixed(1)} m`;
  const text = item.kind.includes("edge") ? `${item.kind} ${item.id} · type ${item.type} · ${item.pts.length} pts · ${len.toFixed(0)} m · ${zText}`
                                         : `${item.kind} ${item.id} · ${len.toFixed(0)} m · ${zText}`;
  tooltip.textContent = text; tooltip.style.left = `${x}px`; tooltip.style.top = `${y}px`; tooltip.hidden = false;
}

function markLine(mk) {
  const parts = [];
  if (mk.edge) parts.push(`${mk.edge.kind} ${mk.edge.id} type ${mk.edge.type} at ${mk.edge.d.toFixed(2)} m`);
  if (mk.oldEdge) parts.push(`old edge ${mk.oldEdge.id} at ${mk.oldEdge.d.toFixed(2)} m`);
  if (mk.lane) parts.push(`${mk.lane.kind} ${mk.lane.id} at ${mk.lane.d.toFixed(2)} m`);
  return `#${mk.n} ${CATEGORIES[mk.cat][0]} at (${mk.x.toFixed(1)}, ${mk.y.toFixed(1)}, z ${mk.z.toFixed(1)})  [${parts.join("; ")}]`;
}
function report() {
  const m = current(), marks = state.marks[state.map];
  const head = `Map marks for ${m.name} (PufferDrive bin coordinates, ${META.bin_dir}${META.compare_dir ? ", old edges from " + META.compare_dir : ""}):`;
  return [head, ...marks.map(mk => "  " + markLine(mk))].join("\n");
}
function seamLine(s) { return `${s.kind} ${s.id} at (${s.x.toFixed(1)}, ${s.y.toFixed(1)}) z ${s.z0.toFixed(1)} → ${s.z1.toFixed(1)}`; }
function focusSeam(s) {
  state.view.yaw = 0.6; state.view.pitch = 0.95;
  fitBox([s.x - 35, s.y - 35, s.x + 35, s.y + 35], 20, (s.z0 + s.z1) / 2); draw();
}
function renderPanel() {
  const m = current(), marks = state.marks[state.map];
  const stats = [["lanes", m.lanes.length], ["road edges", m.edges.length], ["old edges", m.compare.length], ["marks", marks.length],
                 ["seams", m.seams.length], ["old seams", m.compareSeams.length]];
  document.getElementById("stats").innerHTML = stats.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const seams = [...m.seams, ...(state.compare ? m.compareSeams : [])];
  document.getElementById("seams").innerHTML = seams.length
    ? seams.map((s, i) => `<div class="seam-row${s.kind === "old edge" ? " old" : ""}" data-i="${i}">${seamLine(s)}</div>`).join("")
    : `<div class="seam-row none">no height seams in the bin road edges</div>`;
  document.querySelectorAll(".seam-row[data-i]").forEach(row => row.addEventListener("click", () => focusSeam(seams[Number(row.dataset.i)])));
  document.getElementById("marks").innerHTML = marks.map((mk, i) =>
    `<div class="mark-row" data-i="${i}"><span class="n" style="background:${color(CATEGORIES[mk.cat][1])}">${mk.n}</span><span>${markLine(mk).replace(/^#\d+ /, "")}<br><span class="d">${CATEGORIES[mk.cat][2]}</span></span><button type="button" title="remove" data-rm="${i}">×</button></div>`).join("");
  document.querySelectorAll(".mark-row").forEach(row => row.addEventListener("click", e => {
    const i = Number(row.dataset.i);
    if (e.target.dataset.rm !== undefined) { removeMark(i); return; }
    const mk = marks[i]; fitBox([mk.x - 40, mk.y - 40, mk.x + 40, mk.y + 40], 20, mk.z); draw();
  }));
  document.getElementById("report").value = report();
  document.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.cat === state.category)));
}
function addMark(x, y, z) {
  const marks = state.marks[state.map], p = [x, y];
  const lane = nearest(p, "lane");
  const own = current().edges.map(e => ({ id: e.id, type: e.type, d: polylineDistance(p, e.pts) })).sort((a, b) => a.d - b.d)[0];
  const old = state.compare ? current().compare.map(e => ({ id: e.id, type: e.type, d: polylineDistance(p, e.pts) })).sort((a, b) => a.d - b.d)[0] : null;
  marks.push({ n: marks.length + 1, cat: state.category, x, y, z,
               edge: own && own.d < 5 ? { ...own, kind: "edge" } : null,
               oldEdge: old && old.d < 5 ? old : null,
               lane: lane && lane.d < 8 ? lane : null });
  renderPanel(); draw();
}
function removeMark(i) {
  const marks = state.marks[state.map]; marks.splice(i, 1); marks.forEach((mk, k) => mk.n = k + 1); renderPanel(); draw();
}

let drag = null, moved = false;
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("pointerdown", e => {
  const orbiting = e.button === 1 || e.button === 2 || (e.button === 0 && e.shiftKey);
  const rect = canvas.getBoundingClientRect();
  drag = { mode: orbiting ? "orbit" : "pan", x: e.clientX, y: e.clientY, t0: [state.view.cx, state.view.cy],
           g0: groundOffset(e.clientX - rect.left, e.clientY - rect.top) };
  moved = false; canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", e => {
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    if (drag.mode === "orbit") { orbit(dx * ORBIT_RAD_PER_PX, dy * ORBIT_RAD_PER_PX); drag.x = e.clientX; drag.y = e.clientY; }
    else { const g = groundOffset(sx, sy); if (g && drag.g0) { state.view.cx = drag.t0[0] + drag.g0[0] - g[0]; state.view.cy = drag.t0[1] + drag.g0[1] - g[1]; } }
    tooltip.hidden = true; document.getElementById("coords").textContent = viewLabel(); requestAnimationFrame(draw); return;
  }
  const item = pick(sx, sy);
  if (item !== state.hover) { state.hover = item; requestAnimationFrame(draw); }
  showTooltip(item, sx, sy);
  const w = toWorld(sx, sy); document.getElementById("coords").textContent = `${w ? `bin (${w[0].toFixed(1)}, ${w[1].toFixed(1)})` : "above horizon"} · ${viewLabel()}`;
});
canvas.addEventListener("pointerup", e => {
  canvas.releasePointerCapture(e.pointerId);
  if (!moved && e.button === 0) {
    const rect = canvas.getBoundingClientRect(), sx = e.clientX - rect.left, sy = e.clientY - rect.top;
    const w = pickVertex(sx, sy) || toWorld(sx, sy);
    if (w) addMark(w[0], w[1], w[2] || 0);
  }
  drag = null;
});
canvas.addEventListener("pointerleave", () => { state.hover = null; tooltip.hidden = true; draw(); });
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const sx = e.clientX - rect.left, sy = e.clientY - rect.top;
  const factor = Math.exp(-e.deltaY * 0.0015);
  const v = state.view, before = groundOffset(sx, sy);
  v.scale *= factor;
  const after = groundOffset(sx, sy);
  if (before && after) { v.cx += before[0] - after[0]; v.cy += before[1] - after[1]; }
  requestAnimationFrame(draw);
}, { passive: false });
window.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (CATEGORIES[e.key]) { state.category = e.key; renderPanel(); }
  else if (e.key === "u") { const marks = state.marks[state.map]; if (marks.length) removeMark(marks.length - 1); }
  else if (e.key === "f") { fitView(); draw(); }
  else if (e.key === "t") { topView(); draw(); }
  else if (e.key === "c") { copyReport(); }
  else if (e.key === "Escape") { state.hover = null; tooltip.hidden = true; draw(); }
});
function copyReport() {
  const ta = document.getElementById("report"); ta.select();
  const done = () => { document.getElementById("copied").textContent = "copied"; setTimeout(() => { document.getElementById("copied").textContent = ""; }, 1500); };
  if (navigator.clipboard) navigator.clipboard.writeText(ta.value).then(done, () => { document.execCommand("copy"); done(); }); else { document.execCommand("copy"); done(); }
}
function selectMap(i) {
  state.map = i; state.hover = null;
  document.querySelectorAll(".tabs button").forEach((b, j) => b.setAttribute("aria-pressed", String(i === j)));
  fitView(); renderPanel(); draw();
}
const tabs = document.querySelector(".tabs");
MAPS.forEach((m, i) => {
  const b = document.createElement("button"); b.textContent = m.name; b.type = "button";
  b.setAttribute("aria-pressed", String(i === 0)); b.addEventListener("click", () => selectMap(i)); tabs.appendChild(b);
});
document.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => { state.category = b.dataset.cat; renderPanel(); }));
document.getElementById("compare").addEventListener("change", e => { state.compare = e.target.checked; renderPanel(); draw(); });
document.getElementById("lines").addEventListener("change", e => { state.lines = e.target.checked; draw(); });
document.getElementById("lanes").addEventListener("change", e => { state.lanes = e.target.checked; draw(); });
document.getElementById("height").addEventListener("change", e => { state.heightColor = e.target.checked; draw(); });
document.getElementById("zscale").addEventListener("input", e => { setZScale(Number(e.target.value)); draw(); });
document.getElementById("fit").addEventListener("click", () => { fitView(); draw(); });
document.getElementById("top").addEventListener("click", () => { topView(); draw(); });
document.getElementById("copy").addEventListener("click", copyReport);
document.getElementById("clear").addEventListener("click", () => { state.marks[state.map] = []; renderPanel(); draw(); });
document.getElementById("legend").innerHTML = `<span><i style="background:var(--edge)"></i>road edge (bin)</span><span><i style="background:var(--median)"></i>median edge</span><span><i style="background:var(--compare)"></i>old edge (compare)</span><span><i style="background:var(--lane)"></i>lane centerline</span><span><i style="background:var(--junction)"></i>junction lane</span><span><i style="background:linear-gradient(90deg, hsl(230,85%,45%), hsl(0,85%,45%))"></i>height (low → high)</span><span><i style="border:2px solid var(--accent); background:transparent; height:8px; width:8px; border-radius:50%"></i>height seam in an edge</span>`;
window.addEventListener("resize", resize);
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { renderPanel(); draw(); });
new MutationObserver(() => { renderPanel(); draw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
function applyHashView() {
  const q = new URLSearchParams(location.hash.replace(/^#/, ""));
  if (!q.size) return;
  const mapIdx = MAPS.findIndex(m => m.name === q.get("map"));
  if (mapIdx >= 0) { state.map = mapIdx; document.querySelectorAll(".tabs button").forEach((b, j) => b.setAttribute("aria-pressed", String(mapIdx === j))); }
  if (q.has("z")) { state.zscale = Number(q.get("z")); document.getElementById("zscale").value = state.zscale; document.getElementById("zscale-val").textContent = `${state.zscale}×`; }
  if (q.has("height")) { state.heightColor = q.get("height") === "1"; document.getElementById("height").checked = state.heightColor; }
  const deg = Math.PI / 180;
  state.view = { cx: 0, cy: 0, cz: 0, scale: 1, yaw: Number(q.get("yaw") || 0) * deg, pitch: Math.min(MAX_PITCH, Number(q.get("pitch") || 0) * deg) };
  fitView();
  if (q.has("cx") && q.has("cy")) {
    const r = Number(q.get("r") || 60), cx = Number(q.get("cx")), cy = Number(q.get("cy"));
    fitBox([cx - r, cy - r, cx + r, cy + r], 20, Number(q.get("cz") || 0));
  }
  renderPanel();
}
applyHashView();
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
      <label><input type="checkbox" id="height"> colour by height</label>
      <label>z <input type="range" id="zscale" min="1" max="10" step="1" value="1"> <span id="zscale-val">1×</span></label>
      <button type="button" id="top">top (t)</button>
      <button type="button" id="fit">fit (f)</button>
      <span id="coords"></span>
    </div>
  </header>
  <div class="stage">
    <canvas id="map" aria-label="PufferDrive map with road edges"></canvas>
    <div class="legend" id="legend"></div>
    <div class="tooltip" id="tooltip" hidden></div>
    <div class="hint">left-drag pan · right/middle-drag (or shift+drag) rotate · wheel zoom · t top view · click drops a mark (1/2/3) · u undo · c copy report · view preset via #map=Town05&amp;yaw=30&amp;pitch=55&amp;z=3&amp;height=1&amp;cx=-190&amp;cy=-2&amp;r=60 · bins: {meta["bin_dir"]}{compare_note}</div>
  </div>
  <aside>
    <div class="stats" id="stats"></div>
    <div class="seams" id="seams"></div>
    <div class="marks" id="marks"></div>
    <div class="report">
      <div class="row"><button type="button" id="copy">copy report (c)</button><button type="button" id="clear">clear marks</button><span id="copied"></span></div>
      <textarea id="report" readonly></textarea>
    </div>
    <div class="note">Marks are in PufferDrive bin coordinates (what the C sim and the obs replays use). Hover shows element ids and heights; a mark snaps to the nearest vertex and lists the nearest bin road edge, old edge and lane. Seam rows (edges whose height steps between layers) focus the camera on the spot.</div>
  </aside>
</div>
<script>window.__MAP_INSPECTOR__ = {payload};</script>
<script>{SCRIPT}</script>
"""
    return f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n{head}</head>\n<body>{body}</body>\n</html>\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin-dir", required=True, help="Directory of PufferDrive *.bin map files to inspect")
    parser.add_argument(
        "--compare-dir", default=None, help="Second bin directory whose road edges are overlaid in orange"
    )
    parser.add_argument(
        "--towns", nargs="*", default=None, help="Subset of towns, e.g. Town04 Town06 (default: all bins)"
    )
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
