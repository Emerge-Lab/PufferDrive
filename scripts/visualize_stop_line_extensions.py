"""Render every map's traffic-light stop lines plus their virtual lateral extensions
(RED_LIGHT_LATERAL_EXTENSION_M, read from constants.h) into a standalone HTML page,
and audit the extensions: list every lane an entering-aligned vehicle could cross
outside the painted span -- the only geometry that can trigger a red-light flag.
Findings tagged OWN are the light's own controlled lanes (desired coverage);
CROSS-ARM findings are other lanes and deserve a manual look.

Usage:
  python scripts/visualize_stop_line_extensions.py \
      [--map-dir pufferlib/resources/drive/binaries/carla_hole_fixes] \
      [--output stop_line_audit.html]
"""

import argparse
import glob
import json
import math
import os
import re
import struct

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSTANTS_H = os.path.join(REPO_ROOT, "pufferlib", "ocean", "drive", "constants.h")
Z_BUFFER_M = 4.0
PAINTED_SPAN_TOLERANCE_M = 0.25


def read_extension_from_constants() -> float:
    text = open(CONSTANTS_H).read()
    match = re.search(r"#define\s+RED_LIGHT_LATERAL_EXTENSION_M\s+([0-9.]+)f?", text)
    if not match:
        raise SystemExit(f"RED_LIGHT_LATERAL_EXTENSION_M not found in {CONSTANTS_H}")
    return float(match.group(1))


class BinReader:
    def __init__(self, path):
        self.buf = open(path, "rb").read()
        self.off = 0

    def ints(self, n=1):
        values = struct.unpack_from(f"<{n}i", self.buf, self.off)
        self.off += 4 * n
        return values[0] if n == 1 else list(values)

    def floats(self, n=1):
        values = struct.unpack_from(f"<{n}f", self.buf, self.off)
        self.off += 4 * n
        return values[0] if n == 1 else np.array(values, dtype=np.float32)

    def skip(self, nbytes):
        self.off += nbytes


def is_road_lane(road_type):
    return 0 <= road_type <= 9


def is_road_edge(road_type):
    return 20 <= road_type <= 29


def load_map_bin(path):
    """Mirror of load_map_binary in map_data.h (roads + traffic elements only)."""
    r = BinReader(path)
    num_agents, num_roads, num_traffic, _num_objects = r.ints(), r.ints(), r.ints(), r.ints()
    for _ in range(num_agents):
        _agent_id, _agent_type, trajectory_len = r.ints(), r.ints(), r.ints()
        r.skip(9 * trajectory_len * 4 + trajectory_len * 4)
        route_len = r.ints()
        r.skip(route_len * 4)
        r.ints()  # route_gt_len
        r.floats(3)  # gt goal xyz
        r.ints()  # mark_as_expert
    roads = []
    for i in range(num_roads):
        road_id, road_type, segment_size = r.ints(), r.ints(), r.ints()
        assert road_id == i, f"road id {road_id} != idx {i}"
        x, y, z = r.floats(segment_size), r.floats(segment_size), r.floats(segment_size)
        headings = r.floats(segment_size)
        road = dict(type=road_type, x=x, y=y, z=z, headings=headings)
        if is_road_lane(road_type):
            num_entries = r.ints()
            r.skip(num_entries * 4)
            num_exits = r.ints()
            r.skip(num_exits * 4)
            r.floats(2)  # speed_limit, length
            r.skip(segment_size * 4)  # cum_lengths
        roads.append(road)
    traffic = []
    for i in range(num_traffic):
        traffic_id, tc_type = r.ints(), r.ints()
        assert traffic_id == i, f"traffic id {traffic_id} != idx {i}"
        stop_line = r.floats(6)
        heading = r.floats()
        state_size = r.ints()
        r.skip(state_size * 4)
        num_controlled = r.ints()
        controlled = r.ints(num_controlled) if num_controlled > 0 else []
        if not isinstance(controlled, list):
            controlled = [controlled]
        traffic.append(dict(type=tc_type, stop_line=stop_line, heading=heading, controlled_lanes=controlled))
    return roads, traffic


def segment_intersection(p1, p2, p3, p4):
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2y - (p3[1] - p1[1]) * d2x) / denom
    u = ((p3[0] - p1[0]) * d1y - (p3[1] - p1[1]) * d1x) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t
    return None


def heading_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def export_town(path, extension_m):
    roads, traffic_elements = load_map_bin(path)

    lanes_js, edges_js = [], []
    for i, road in enumerate(roads):
        x, y = np.asarray(road["x"], dtype=float), np.asarray(road["y"], dtype=float)
        if len(x) < 2:
            continue
        points = [[round(float(a), 1), round(float(b), 1)] for a, b in zip(x, y)]
        if is_road_lane(road["type"]):
            lanes_js.append({"id": i, "p": points})
        elif is_road_edge(road["type"]) or road["type"] == 6:
            edges_js.append({"p": points})

    tls_js = []
    for ci, tc in enumerate(traffic_elements):
        if tc["type"] != 1:  # TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT
            continue
        sl = tc["stop_line"]
        line_dx, line_dy = sl[3] - sl[0], sl[4] - sl[1]
        line_len = math.hypot(line_dx, line_dy)
        if line_len <= 0:
            continue
        ux, uy = line_dx / line_len, line_dy / line_len
        mid_x, mid_y = (sl[0] + sl[3]) / 2, (sl[1] + sl[4]) / 2
        mid_z = (sl[2] + sl[5]) / 2
        ext_p1 = (sl[0] - extension_m * ux, sl[1] - extension_m * uy)
        ext_p2 = (sl[3] + extension_m * ux, sl[4] + extension_m * uy)

        hits, seen_lanes = [], set()
        for i, road in enumerate(roads):
            if not is_road_lane(road["type"]) or i in seen_lanes:
                continue
            x, y, z = (np.asarray(road[k], dtype=float) for k in ("x", "y", "z"))
            for j in range(len(x) - 1):
                t = segment_intersection((x[j], y[j]), (x[j + 1], y[j + 1]), ext_p1, ext_p2)
                if t is None:
                    continue
                cross_x = x[j] + t * (x[j + 1] - x[j])
                cross_y = y[j] + t * (y[j + 1] - y[j])
                cross_z = z[j] + t * (z[j + 1] - z[j])
                lateral = (cross_x - mid_x) * ux + (cross_y - mid_y) * uy
                hdiff = abs(heading_diff(float(road["headings"][j]), tc["heading"]))
                entering_aligned = hdiff < math.pi / 2
                outside_painted = abs(lateral) > line_len / 2 + PAINTED_SPAN_TOLERANCE_M
                if entering_aligned and outside_painted:
                    seen_lanes.add(i)
                    hits.append({
                        "lane": i, "x": round(cross_x, 1), "y": round(cross_y, 1),
                        "lat": round(lateral, 1), "hdiff_deg": round(math.degrees(hdiff)),
                        "z_filtered": bool(abs(cross_z - mid_z) > Z_BUFFER_M),
                        "own": bool(i in tc["controlled_lanes"]),
                    })
                    break
        tls_js.append({
            "id": ci,
            "line": [round(float(v), 1) for v in (sl[0], sl[1], sl[3], sl[4])],
            "ext": [round(float(v), 1) for v in (*ext_p1, *ext_p2)],
            "heading": round(float(tc["heading"]), 3),
            "mid": [round(mid_x, 1), round(mid_y, 1)],
            "hits": hits,
        })
    return {"lanes": lanes_js, "edges": edges_js, "tls": tls_js}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", default=os.path.join(REPO_ROOT, "pufferlib/resources/drive/binaries/carla_hole_fixes"))
    parser.add_argument("--output", default="stop_line_audit.html")
    args = parser.parse_args()

    extension_m = read_extension_from_constants()
    print(f"RED_LIGHT_LATERAL_EXTENSION_M = {extension_m} (from constants.h)")

    data = {}
    for path in sorted(glob.glob(os.path.join(args.map_dir, "*.bin"))):
        town = os.path.basename(path).replace("opendrive__", "").replace(".bin", "")
        data[town] = export_town(path, extension_m)
        hit_count = sum(len(tl["hits"]) for tl in data[town]["tls"])
        own_count = sum(1 for tl in data[town]["tls"] for h in tl["hits"] if h["own"])
        print(f"{town}: {len(data[town]['tls'])} lights, {hit_count} findings ({own_count} own-lane)")

    payload = json.dumps(data, separators=(",", ":"), default=float)
    html = PAGE_HEAD + '<script type="application/json" id="mapdata">' + payload + "</script>\n" + PAGE_TAIL
    with open(args.output, "w") as f:
        f.write(html)
    print(f"wrote {args.output} ({len(html) / 1e6:.1f} MB)")


PAGE_HEAD = r"""<title>Stop Line Audit</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {
  --ground: #F7F6F3;
  --panel: #FFFFFF;
  --panel-edge: #E4E1DA;
  --ink: #22262B;
  --ink-soft: #6C6F75;
  --lane: #C8C5BC;
  --edge: #918D82;
  --stopline: #C6303E;
  --extension: #D97E1A;
  --hit: #B0369B;
  --hit-soft: rgba(176,54,155,0.12);
  --accent: #3B6EA8;
  --accent-soft: rgba(59,110,168,0.12);
  --ok: #3E7D4E;
  --scrollbar: #D5D2CA;
}
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #14161A;
    --panel: #1C1F24;
    --panel-edge: #2C3038;
    --ink: #E8E6E1;
    --ink-soft: #9A9DA3;
    --lane: #3A3E45;
    --edge: #575C64;
    --stopline: #E25563;
    --extension: #E8A04C;
    --hit: #D466C0;
    --hit-soft: rgba(212,102,192,0.16);
    --accent: #6E9FD4;
    --accent-soft: rgba(110,159,212,0.16);
    --ok: #6FAE7E;
    --scrollbar: #383C44;
  }
}
:root[data-theme="dark"] {
  --ground: #14161A;
  --panel: #1C1F24;
  --panel-edge: #2C3038;
  --ink: #E8E6E1;
  --ink-soft: #9A9DA3;
  --lane: #3A3E45;
  --edge: #575C64;
  --stopline: #E25563;
  --extension: #E8A04C;
  --hit: #D466C0;
  --hit-soft: rgba(212,102,192,0.16);
  --accent: #6E9FD4;
  --accent-soft: rgba(110,159,212,0.16);
  --ok: #6FAE7E;
  --scrollbar: #383C44;
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", system-ui, sans-serif;
  font-size: 14px;
  overflow: hidden;
}
#app { display: flex; height: 100vh; }
#sidebar {
  width: 300px;
  min-width: 300px;
  background: var(--panel);
  border-right: 1px solid var(--panel-edge);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
#sidebar h1 {
  font-family: Archivo, system-ui, sans-serif;
  font-size: 17px;
  font-weight: 700;
  letter-spacing: 0.01em;
  margin: 0;
  padding: 16px 16px 2px;
}
#sidebar .sub {
  padding: 0 16px 12px;
  color: var(--ink-soft);
  font-size: 12px;
  line-height: 1.5;
  border-bottom: 1px solid var(--panel-edge);
}
.seclabel {
  font-family: Archivo, system-ui, sans-serif;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink-soft);
  padding: 14px 16px 6px;
}
#towns { padding: 0 8px; }
.townrow {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 6px 8px;
  border: 0;
  background: none;
  color: var(--ink);
  font: inherit;
  border-radius: 6px;
  cursor: pointer;
  text-align: left;
}
.townrow:hover { background: var(--accent-soft); }
.townrow.active { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
.townrow .name { flex: 1; }
.townrow .tlcount {
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  color: var(--ink-soft);
}
.badge {
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  font-weight: 500;
  padding: 1px 7px;
  border-radius: 99px;
  background: var(--hit-soft);
  color: var(--hit);
}
#findings { flex: 1; overflow-y: auto; padding: 0 8px 16px; scrollbar-width: thin; scrollbar-color: var(--scrollbar) transparent; }
.finding {
  display: block;
  width: 100%;
  border: 0;
  background: none;
  color: var(--ink);
  font: inherit;
  text-align: left;
  padding: 7px 8px;
  border-radius: 6px;
  cursor: pointer;
  line-height: 1.45;
}
.finding:hover { background: var(--hit-soft); }
.finding.active { background: var(--hit-soft); outline: 1px solid var(--hit); }
.finding .where {
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
}
.finding .meta { font-size: 12px; color: var(--ink-soft); }
.tag {
  display: inline-block;
  font-family: Archivo, system-ui, sans-serif;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.06em;
  padding: 1px 6px;
  border-radius: 4px;
  vertical-align: 1px;
}
.tag.own { background: var(--accent-soft); color: var(--ok); }
.tag.other { background: var(--hit-soft); color: var(--hit); }
.note {
  margin: 8px 16px 0;
  padding: 10px 12px;
  font-size: 12px;
  line-height: 1.55;
  color: var(--ink-soft);
  background: var(--ground);
  border: 1px solid var(--panel-edge);
  border-radius: 8px;
}
#main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
#topbar {
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 10px 18px;
  background: var(--panel);
  border-bottom: 1px solid var(--panel-edge);
  flex-wrap: wrap;
}
#townname {
  font-family: Archivo, system-ui, sans-serif;
  font-size: 15px;
  font-weight: 700;
}
.leg { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--ink-soft); white-space: nowrap; }
.leg .sw { width: 26px; height: 0; border-top: 3px solid; border-radius: 2px; }
.leg .sw.dash { border-top-style: dashed; border-top-width: 2px; }
.leg .sw.ring {
  width: 12px; height: 12px;
  border: 2px solid var(--hit);
  border-radius: 50%;
}
#hint { margin-left: auto; font-size: 12px; color: var(--ink-soft); }
#canvaswrap { flex: 1; position: relative; min-height: 0; }
canvas { position: absolute; inset: 0; width: 100%; height: 100%; cursor: grab; display: block; }
canvas.dragging { cursor: grabbing; }
#scale {
  position: absolute;
  right: 14px;
  bottom: 12px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  color: var(--ink-soft);
  background: var(--panel);
  border: 1px solid var(--panel-edge);
  border-radius: 6px;
  padding: 3px 8px;
}
button:focus-visible, .finding:focus-visible, .townrow:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
@media (max-width: 760px) {
  #app { flex-direction: column; }
  #sidebar { width: 100%; min-width: 0; max-height: 45vh; border-right: 0; border-bottom: 1px solid var(--panel-edge); }
}
</style>
<div id="app">
  <div id="sidebar">
    <h1>Stop Line Audit</h1>
    <div class="sub">Painted stop lines and their 15&thinsp;m virtual extensions across every CARLA bin in <span style="font-family:'IBM Plex Mono',monospace">carla_hole_fixes</span>. Verify no extension reaches a parallel road.</div>
    <div class="seclabel">Towns</div>
    <div id="towns"></div>
    <div class="seclabel">Audit findings <span id="findcount" class="badge"></span></div>
    <div class="note">A finding is a lane an <em>entering-aligned</em> vehicle (&lt;&thinsp;90&deg; off the light heading) could cross <em>outside</em> the painted span &mdash; the only geometry that can flag. <span class="tag own">OWN</span> = the light&rsquo;s own controlled lane (desired coverage). <span class="tag other">CROSS-ARM</span> = any other lane &mdash; inspect these.</div>
    <div id="findings"></div>
  </div>
  <div id="main">
    <div id="topbar">
      <span id="townname"></span>
      <span class="leg"><span class="sw" style="border-color:var(--stopline)"></span>painted stop line</span>
      <span class="leg"><span class="sw dash" style="border-color:var(--extension)"></span>15 m extension</span>
      <span class="leg"><span class="sw" style="border-color:var(--lane)"></span>lane centerline</span>
      <span class="leg"><span class="sw ring"></span>audit finding</span>
      <span id="hint">drag to pan &middot; scroll to zoom &middot; arrows show direction when zoomed</span>
    </div>
    <div id="canvaswrap">
      <canvas id="cv"></canvas>
      <div id="scale"></div>
    </div>
  </div>
</div>
"""

PAGE_TAIL = r"""<script>
const DATA = JSON.parse(document.getElementById('mapdata').textContent);
const TOWNS = Object.keys(DATA);
const EXT_M = 15, TRIGGER_R = 30;
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let town = TOWNS[0];
let cam = { x: 0, y: 0, s: 1 };           // world center + px per meter
let sel = null;                            // {tl, lane} selected finding
const pathCache = {};                      // town -> {lanes: Path2D, edges: Path2D, bounds}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function buildPaths(t) {
  if (pathCache[t]) return pathCache[t];
  const d = DATA[t];
  let minx = 1e18, miny = 1e18, maxx = -1e18, maxy = -1e18;
  const mk = (arr) => {
    const p = new Path2D();
    for (const el of arr) {
      const pts = el.p;
      p.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) p.lineTo(pts[i][0], pts[i][1]);
      for (const [x, y] of pts) {
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
      }
    }
    return p;
  };
  const lanes = mk(d.lanes), edges = mk(d.edges);
  pathCache[t] = { lanes, edges, bounds: [minx, miny, maxx, maxy] };
  return pathCache[t];
}

function fitView() {
  const { bounds } = buildPaths(town);
  const [minx, miny, maxx, maxy] = bounds;
  cam.x = (minx + maxx) / 2; cam.y = (miny + maxy) / 2;
  const w = cv.clientWidth, h = cv.clientHeight;
  cam.s = Math.min(w / (maxx - minx + 80), h / (maxy - miny + 80));
}

function resize() {
  const dpr = devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr;
  cv.height = cv.clientHeight * dpr;
  draw();
}

function draw() {
  const dpr = devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = css('--ground');
  ctx.fillRect(0, 0, w, h);

  // world -> screen: x right, y up (flip)
  ctx.translate(w / 2, h / 2);
  ctx.scale(cam.s, -cam.s);
  ctx.translate(-cam.x, -cam.y);

  const pc = buildPaths(town);
  const px = 1 / cam.s; // one screen pixel in meters
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';

  ctx.strokeStyle = css('--lane');
  ctx.lineWidth = Math.max(0.5, 1.1 * px);
  ctx.stroke(pc.lanes);
  ctx.strokeStyle = css('--edge');
  ctx.lineWidth = Math.max(0.35, 0.9 * px);
  ctx.stroke(pc.edges);

  // viewport in world coords
  const vx0 = cam.x - w / 2 * px, vx1 = cam.x + w / 2 * px;
  const vy0 = cam.y - h / 2 * px, vy1 = cam.y + h / 2 * px;
  const inView = (x, y, m) => x > vx0 - m && x < vx1 + m && y > vy0 - m && y < vy1 + m;

  // lane direction arrows when zoomed in
  if (cam.s > 2.5) {
    ctx.fillStyle = css('--edge');
    for (const ln of DATA[town].lanes) {
      const pts = ln.p, j = Math.floor((pts.length - 1) / 2);
      const [x1, y1] = pts[j], [x2, y2] = pts[j + 1];
      if (!inView(x1, y1, 10)) continue;
      const a = Math.atan2(y2 - y1, x2 - x1), mx = (x1 + x2) / 2, my = (y1 + y2) / 2, r = 4.5 * px;
      ctx.beginPath();
      ctx.moveTo(mx + Math.cos(a) * r * 1.6, my + Math.sin(a) * r * 1.6);
      ctx.lineTo(mx + Math.cos(a + 2.5) * r, my + Math.sin(a + 2.5) * r);
      ctx.lineTo(mx + Math.cos(a - 2.5) * r, my + Math.sin(a - 2.5) * r);
      ctx.fill();
    }
  }

  // selected finding: highlight the crossing lane + trigger radius
  if (sel && sel.town === town) {
    const laneEl = DATA[town].lanes.find(l => l.id === sel.lane);
    if (laneEl) {
      ctx.strokeStyle = css('--hit');
      ctx.lineWidth = Math.max(1.2, 3.2 * px);
      const p = new Path2D();
      p.moveTo(laneEl.p[0][0], laneEl.p[0][1]);
      for (let i = 1; i < laneEl.p.length; i++) p.lineTo(laneEl.p[i][0], laneEl.p[i][1]);
      ctx.stroke(p);
    }
    const tl = DATA[town].tls.find(t => t.id === sel.tl);
    if (tl) {
      ctx.strokeStyle = css('--accent');
      ctx.setLineDash([4 * px, 4 * px]);
      ctx.lineWidth = 1.2 * px;
      ctx.beginPath();
      ctx.arc(tl.mid[0], tl.mid[1], TRIGGER_R, 0, 7);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  // traffic lights: extension (dashed) under painted line (solid)
  for (const tl of DATA[town].tls) {
    if (!inView(tl.mid[0], tl.mid[1], 40)) continue;
    ctx.strokeStyle = css('--extension');
    ctx.setLineDash([2.2, 1.6]);
    ctx.lineWidth = Math.max(0.8, 1.6 * px);
    ctx.beginPath();
    ctx.moveTo(tl.ext[0], tl.ext[1]);
    ctx.lineTo(tl.ext[2], tl.ext[3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = css('--stopline');
    ctx.lineWidth = Math.max(1.2, 2.6 * px);
    ctx.beginPath();
    ctx.moveTo(tl.line[0], tl.line[1]);
    ctx.lineTo(tl.line[2], tl.line[3]);
    ctx.stroke();
    if (cam.s > 1.5) {
      const a = tl.heading, r = 3.2;
      ctx.fillStyle = css('--stopline');
      ctx.beginPath();
      ctx.moveTo(tl.mid[0] + Math.cos(a) * r * 1.8, tl.mid[1] + Math.sin(a) * r * 1.8);
      ctx.lineTo(tl.mid[0] + Math.cos(a + 2.4) * r, tl.mid[1] + Math.sin(a + 2.4) * r);
      ctx.lineTo(tl.mid[0] + Math.cos(a - 2.4) * r, tl.mid[1] + Math.sin(a - 2.4) * r);
      ctx.fill();
    }
    for (const hit of tl.hits) {
      ctx.strokeStyle = css('--hit');
      ctx.lineWidth = Math.max(1.4, 1.8 * px);
      ctx.beginPath();
      ctx.arc(hit.x, hit.y, Math.max(3, 8 * px), 0, 7);
      ctx.stroke();
    }
  }

  const scaleEl = document.getElementById('scale');
  scaleEl.textContent = cam.s >= 1 ? `${cam.s.toFixed(1)} px/m` : `1 px = ${(1 / cam.s).toFixed(0)} m`;
}

// interaction
let dragging = false, lastPt = null;
cv.addEventListener('pointerdown', e => { dragging = true; lastPt = [e.clientX, e.clientY]; cv.classList.add('dragging'); cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointermove', e => {
  if (!dragging) return;
  cam.x -= (e.clientX - lastPt[0]) / cam.s;
  cam.y += (e.clientY - lastPt[1]) / cam.s;
  lastPt = [e.clientX, e.clientY];
  requestAnimationFrame(draw);
});
cv.addEventListener('pointerup', e => { dragging = false; cv.classList.remove('dragging'); });
cv.addEventListener('wheel', e => {
  e.preventDefault();
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left - rect.width / 2;
  const my = e.clientY - rect.top - rect.height / 2;
  const wx = cam.x + mx / cam.s, wy = cam.y - my / cam.s;
  const f = Math.exp(-e.deltaY * 0.0015);
  cam.s = Math.min(60, Math.max(0.02, cam.s * f));
  cam.x = wx - mx / cam.s; cam.y = wy + my / cam.s;
  requestAnimationFrame(draw);
}, { passive: false });

function flyTo(x, y, s) {
  if (reduceMotion) { cam = { x, y, s }; draw(); return; }
  const from = { ...cam }, t0 = performance.now(), dur = 450;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    cam.x = from.x + (x - from.x) * e;
    cam.y = from.y + (y - from.y) * e;
    cam.s = from.s * Math.pow(s / from.s, e);
    draw();
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// sidebar
function setTown(t, keepView) {
  town = t;
  document.getElementById('townname').textContent = t;
  document.querySelectorAll('.townrow').forEach(el => el.classList.toggle('active', el.dataset.town === t));
  if (!keepView) { fitView(); }
  draw();
}

const townsEl = document.getElementById('towns');
for (const t of TOWNS) {
  const d = DATA[t];
  const hits = d.tls.reduce((n, tl) => n + tl.hits.length, 0);
  const b = document.createElement('button');
  b.className = 'townrow';
  b.dataset.town = t;
  b.innerHTML = `<span class="name">${t}</span><span class="tlcount">${d.tls.length} TL</span>` +
    (hits ? `<span class="badge">${hits}</span>` : '');
  b.onclick = () => { sel = null; setTown(t); renderFindings(); };
  townsEl.appendChild(b);
}

const findEl = document.getElementById('findings');
let findingRows = [];
function renderFindings() {
  findEl.innerHTML = '';
  findingRows = [];
  let total = 0;
  for (const t of TOWNS) {
    for (const tl of DATA[t].tls) {
      for (const hit of tl.hits) {
        total++;
        const b = document.createElement('button');
        b.className = 'finding';
        const active = sel && sel.town === t && sel.tl === tl.id && sel.lane === hit.lane;
        if (active) b.classList.add('active');
        b.innerHTML = `<span class="where">${t} &middot; TL ${tl.id} &rarr; lane ${hit.lane}</span> ` +
          `<span class="tag ${hit.own ? 'own' : 'other'}">${hit.own ? 'OWN' : 'CROSS-ARM'}</span><br>` +
          `<span class="meta">crossing at ${hit.lat > 0 ? '+' : ''}${hit.lat} m lateral &middot; ${hit.hdiff_deg}&deg; off light heading${hit.z_filtered ? ' &middot; z-filtered' : ''}</span>`;
        b.onclick = () => {
          sel = { town: t, tl: tl.id, lane: hit.lane };
          if (town !== t) setTown(t, true);
          renderFindings();
          flyTo(hit.x, hit.y, 5);
        };
        findEl.appendChild(b);
        findingRows.push(b);
      }
    }
  }
  document.getElementById('findcount').textContent = total;
}

matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => draw());
new MutationObserver(() => draw()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
addEventListener('resize', resize);

renderFindings();
setTown(TOWNS[0]);
resize();
</script>
"""


if __name__ == "__main__":
    main()
