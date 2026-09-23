"""Failure analysis + visualizations for CARLA leaderboard runs of the PufferDrive co-sim agent
(pufferlib/ocean/cosim/carla/leaderboard_agent.py run with COSIM_WORLD_LOG and COSIM_TELEMETRY set).
Same report shape as analyze_nuplan_cosim.py: score table worst first, failure-category counts,
per-route diagnosis, six-frame top-down strip + speed plot + chase-cam / top-down video.

usage: python scripts/eval/analyze_carla_cosim.py <run_dir> [<run_dir> ...] <report_dir> [--max-inline N] [--no-video]

<run_dir> holds result.json (leaderboard_evaluator --checkpoint), world_log/*.npz, telemetry/*.csv and
carla_view/*.mp4; routes of one run pair with the per-route files in timestamp order.
"""

import argparse
import glob
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import data_utils.mirror_map_bin as mbin
from pufferlib.ocean.cosim.carla_cosim import write_mp4


LIGHT_COLOR = {1: "red", 2: "orange", 3: "limegreen", 4: "0.5", 0: "0.7"}
STOPPED_SPEED_MPS = 0.3
CREEP_SPEED_MPS = 1.0
CLOSE_AGENT_M = 8.0
MAP_CROP_MARGIN_M = 80.0
STRIP_SPAN_M = 40.0
VIDEO_FRAME_STRIDE = 2
VIDEO_FPS = 5
INFRACTION_KEYS = (
    ("collisions_pedestrian", "collision"),
    ("collisions_vehicle", "collision"),
    ("collisions_layout", "collision"),
    ("red_light", "red light"),
    ("stop_infraction", "stop sign"),
    ("outside_route_lanes", "offroad"),
    ("route_dev", "route deviation"),
    ("route_timeout", "timeout"),
    ("vehicle_blocked", "blocked"),
)
TABLE_COLS = [
    "route",
    "town",
    "status",
    "driving_score",
    "route_completion",
    "infraction_penalty",
    "collisions",
    "red_lights",
    "offroad_pct",
    "route_dev",
    "min_speed_infractions",
    "stopped_frac",
    "creep_frac",
    "categories",
]


def draw_box(ax, cx, cy, heading, length, width, color, alpha=0.7):
    c, s = math.cos(heading), math.sin(heading)
    corners = [(length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2)]
    pts = [(cx + c * dx - s * dy, cy + s * dx + c * dy) for dx, dy in corners]
    ax.add_patch(Polygon(pts, closed=True, color=color, alpha=alpha))
    ax.plot([cx, cx + c * length / 2], [cy, cy + s * length / 2], color="k", lw=0.8)


class RouteLog:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.ego = d["ego"]  # step, x, y, heading, speed, accel, goal_cursor
        self.partners = d["partners"]  # step_idx, x, y, heading, length, width, speed, is_walker
        self.lights = d["lights"]  # (T, num_traffic)
        self.route_goals = d["route_goals"]
        self.dense_route = d["dense_route"]
        self.meta = json.loads(str(d["meta"]))
        self.tick_dt = float(self.meta["tick_dt"])
        self.t = self.ego[:, 0] * self.tick_dt
        bin_data = mbin.read_bin(Path(self.meta["town_bin"]))
        lo = self.ego[:, 1:3].min(axis=0) - MAP_CROP_MARGIN_M
        hi = self.ego[:, 1:3].max(axis=0) + MAP_CROP_MARGIN_M
        self.lanes, self.edges = [], []
        for road in bin_data["roads"]:
            xs, ys = np.asarray(road["x"]), np.asarray(road["y"])
            if not len(xs) or xs.max() < lo[0] or xs.min() > hi[0] or ys.max() < lo[1] or ys.min() > hi[1]:
                continue
            (self.lanes if 0 <= road["type"] <= 9 else self.edges).append((xs, ys))
        self.stop_lines = [(t["type"], np.asarray(t["stop_line"]).reshape(2, 3)[:, :2]) for t in bin_data["traffic"]]
        self.partner_rows = {}
        for row in self.partners:
            self.partner_rows.setdefault(int(row[0]), []).append(row)

    def carla_to_bin(self, x, y):
        tx, ty = self.meta["offset"]
        return x + tx, -y + ty

    def step_nearest(self, bx, by):
        return int(np.argmin(np.hypot(self.ego[:, 1] - bx, self.ego[:, 2] - by)))

    def own_light_state(self, i, ahead_m=40.0):
        """state of the nearest light stop line ahead of the ego (within ahead_m, roughly facing it), else None"""
        ex, ey, eh = self.ego[i, 1], self.ego[i, 2], self.ego[i, 3]
        c, s = math.cos(eh), math.sin(eh)
        best, best_d = None, ahead_m
        for j, (kind, line) in enumerate(self.stop_lines):
            if kind != 1 or j >= self.lights.shape[1]:
                continue
            mx, my = line.mean(axis=0)
            dx, dy = mx - ex, my - ey
            along, lateral = dx * c + dy * s, -dx * s + dy * c
            if -3.0 < along < best_d and abs(lateral) < 4.0:
                best, best_d = int(self.lights[i, j]), along
        return best


def draw_scene(ax, log, i, infraction_pts, title=""):
    ex, ey, eh = log.ego[i, 1], log.ego[i, 2], log.ego[i, 3]
    for xs, ys in log.lanes:
        ax.plot(xs, ys, color="0.8", lw=0.7)
    for xs, ys in log.edges:
        ax.plot(xs, ys, color="0.3", lw=0.6)
    for j, (kind, line) in enumerate(log.stop_lines):
        state = int(log.lights[i, j]) if kind == 1 and j < log.lights.shape[1] else 0
        ax.plot(line[:, 0], line[:, 1], color=LIGHT_COLOR.get(state, "0.7"), lw=2.2 if kind == 1 else 1.0)
    ax.plot(log.dense_route[:, 0], log.dense_route[:, 1], "g--", lw=1.2)
    cur = int(log.ego[i, 6])
    ax.scatter(log.route_goals[:, 0], log.route_goals[:, 1], s=18, c="g", zorder=3)
    if 0 <= cur < len(log.route_goals):
        ax.scatter([log.route_goals[cur, 0]], [log.route_goals[cur, 1]], s=60, c="none", edgecolors="g", zorder=3)
    for row in log.partner_rows.get(i, []):
        _, px, py, ph, length, width, _, is_walker = row
        if abs(px - ex) > STRIP_SPAN_M * 1.2 or abs(py - ey) > STRIP_SPAN_M * 1.2:
            continue
        d = math.hypot(px - ex, py - ey)
        color = "tab:red" if d < CLOSE_AGENT_M else ("tab:purple" if is_walker else "tab:blue")
        draw_box(ax, px, py, ph, length, width, color)
    for kind, bx, by in infraction_pts:
        ax.plot([bx], [by], marker="x", color="k" if kind == "collision" else "r", ms=8, mew=2)
    draw_box(ax, ex, ey, eh, 4.9, 2.1, "tab:orange", 0.95)
    ax.set_xlim(ex - STRIP_SPAN_M, ex + STRIP_SPAN_M)
    ax.set_ylim(ey - STRIP_SPAN_M, ey + STRIP_SPAN_M)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def fig_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def parse_infractions(record, log):
    """-> [(category, bin_x, bin_y, message)] for every infraction with a position"""
    out = []
    for key, category in INFRACTION_KEYS:
        for message in record.get("infractions", {}).get(key, []) or []:
            m = re.search(r"x=([-\d.]+), y=([-\d.]+)", message)
            if m:
                bx, by = log.carla_to_bin(float(m.group(1)), float(m.group(2)))
                out.append((category, bx, by, message))
            else:
                out.append((category, float("nan"), float("nan"), message))
    return out


def stall_segments(log):
    """[(start_idx, end_idx)] of maximal runs with speed < STOPPED_SPEED_MPS, longest first"""
    stopped = np.abs(log.ego[:, 4]) < STOPPED_SPEED_MPS
    segs, start = [], None
    for i, s in enumerate(stopped):
        if s and start is None:
            start = i
        if not s and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(stopped) - 1))
    return sorted(segs, key=lambda seg: seg[0] - seg[1])


def diagnose(record, log, infractions):
    scores = record["scores"]
    speed = np.abs(log.ego[:, 4])
    stopped_frac = float((speed < STOPPED_SPEED_MPS).mean())
    creep_frac = float(((speed >= STOPPED_SPEED_MPS) & (speed < CREEP_SPEED_MPS)).mean())
    categories, reasons = [], []
    for seg in stall_segments(log)[:2]:
        dur = log.t[seg[1]] - log.t[seg[0]]
        if dur < 8.0:
            continue
        states = [log.own_light_state(k) for k in range(seg[0], seg[1] + 1)]
        greens = sum(1 for s in states if s == 3) * log.tick_dt * (log.ego[1, 0] - log.ego[0, 0])
        light = "no light ahead" if all(s is None for s in states) else f"own light green for {greens:.0f} s of it"
        reasons.append(f"standstill {dur:.0f} s from t={log.t[seg[0]]:.0f} s ({light})")
    if stopped_frac > 0.4:
        categories.append("stalled")
    elif creep_frac > 0.15:
        categories.append("creeping")
    for category, bx, by, message in infractions:
        categories.append(category)
        if math.isnan(bx):
            reasons.append(f"{category}: {message[:90]}")
            continue
        i = log.step_nearest(bx, by)
        state = log.own_light_state(i)
        reasons.append(
            f"{category} at t={log.t[i]:.0f} s (v={log.ego[i, 4]:.1f} m/s, own light {LIGHT_COLOR.get(state, 'none') if state is not None else 'none'})"
        )
    if scores["score_route"] < 100 and "route deviation" not in categories and "timeout" not in categories:
        reasons.append(f"route completion {scores['score_route']:.0f}%: {record.get('status')}")
    for message in record.get("infractions", {}).get("min_speed_infractions", []) or []:
        m = re.search(r"([\d.]+)%", message)
        if m and float(m.group(1)) < 80:
            categories.append("slow")
            reasons.append(f"min-speed: {message[:70]}")
            break
    counts = {}
    for c in categories:
        counts[c] = counts.get(c, 0) + 1
    return sorted(counts), reasons or ["no infraction"], stopped_frac, creep_frac


def render_route(record, log, chase_mp4, report_dir, tag, make_video):
    infractions = parse_infractions(record, log)
    categories, reasons, stopped_frac, creep_frac = diagnose(record, log, infractions)
    n = len(log.ego)
    picks = {0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1}
    for seg in stall_segments(log)[:1]:
        picks.add((seg[0] + seg[1]) // 2)
    for _, bx, by, _ in infractions[:2]:
        if not math.isnan(bx):
            picks.add(log.step_nearest(bx, by))
    picks = sorted(picks)[:8]
    pts = [(c, bx, by) for c, bx, by, _ in infractions if not math.isnan(bx)]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.8))
    for ax, i in zip(np.atleast_1d(axes), picks):
        draw_scene(ax, log, i, pts, title=f"t={log.t[i]:.0f}s v={log.ego[i, 4]:.1f} m/s goal {int(log.ego[i, 6])}")
    fig.suptitle(
        f"{tag} {log.meta['town']} DS={record['scores']['score_composed']:.1f} RC={record['scores']['score_route']:.0f}%",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(report_dir / "strips" / f"{tag}.png", dpi=70)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.plot(log.t, log.ego[:, 4], color="tab:orange", label="ego speed")
    states = np.array([log.own_light_state(i) or 0 for i in range(n)])
    for state, color in LIGHT_COLOR.items():
        mask = states == state
        if state and mask.any():
            ax.scatter(log.t[mask], np.full(mask.sum(), -0.6), s=6, c=color, marker="s")
    for seg in stall_segments(log):
        if log.t[seg[1]] - log.t[seg[0]] >= 8.0:
            ax.axvspan(log.t[seg[0]], log.t[seg[1]], color="0.9")
    for category, bx, by, _ in infractions:
        if not math.isnan(bx):
            ax.axvline(log.t[log.step_nearest(bx, by)], color="k" if category == "collision" else "r", lw=1.0, ls="--")
    ax.set_xlabel("s")
    ax.set_ylabel("m/s")
    ax.set_ylim(bottom=-1.0)
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(report_dir / "speed" / f"{tag}.png", dpi=70)
    plt.close(fig)

    if chase_mp4 is not None:
        shutil.copy(chase_mp4, report_dir / "videos" / f"{tag}_chase.mp4")
    if make_video:
        frames = []
        for i in range(0, n, VIDEO_FRAME_STRIDE):
            fig, ax = plt.subplots(figsize=(7, 7), dpi=90)
            draw_scene(ax, log, i, pts, title=f"{tag} t={log.t[i]:.0f}s v={log.ego[i, 4]:.1f} m/s")
            frames.append(fig_to_rgb(fig))
            plt.close(fig)
        write_mp4(report_dir / "videos" / f"{tag}_topdown.mp4", frames, fps=VIDEO_FPS)
    infr = record.get("infractions", {})
    return {
        "route": tag,
        "town": log.meta["town"],
        "status": record.get("status"),
        "driving_score": float(record["scores"]["score_composed"]),
        "route_completion": float(record["scores"]["score_route"]),
        "infraction_penalty": float(record["scores"]["score_penalty"]),
        "collisions": sum(
            len(infr.get(k, []) or []) for k in ("collisions_pedestrian", "collisions_vehicle", "collisions_layout")
        ),
        "red_lights": len(infr.get("red_light", []) or []),
        "offroad_pct": sum(
            float(m.group(1))
            for m in (re.search(r"\(([\d.]+)%", s) for s in infr.get("outside_route_lanes", []) or [])
            if m
        ),
        "route_dev": len(infr.get("route_dev", []) or []),
        "min_speed_infractions": len(infr.get("min_speed_infractions", []) or []),
        "stopped_frac": stopped_frac,
        "creep_frac": creep_frac,
        "duration_s": float(log.t[-1]),
        "categories": "|".join(categories) if categories else "none",
        "diagnosis": "; ".join(reasons),
        "has_chase": chase_mp4 is not None,
        "has_topdown": make_video,
    }


def route_section(r, inline):
    tag = r["route"]
    html = [
        f"<h3 id='{tag}'>{tag} &middot; {r['town']} &middot; DS {r['driving_score']:.1f} &middot; RC {r['route_completion']:.0f}% &middot; {r['status']}</h3>",
        f"<p><b>{r['categories']}</b>: {r['diagnosis']}</p>",
    ]
    if inline:
        html.append(f"<img src='strips/{tag}.png' style='max-width:100%'><br><img src='speed/{tag}.png'><br>")
        if r["has_chase"]:
            html.append(f"<video src='videos/{tag}_chase.mp4' controls width='480'></video>")
        if r["has_topdown"]:
            html.append(f"<video src='videos/{tag}_topdown.mp4' controls width='480'></video>")
    return "\n".join(html)


def collect_routes(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "result.json") as result_file:
        result = json.load(result_file)
    records = result["_checkpoint"]["records"]
    logs = sorted(glob.glob(str(run_dir / "world_log" / "*.npz")))
    videos = sorted(glob.glob(str(run_dir / "carla_view" / "*.mp4")))
    if len(logs) != len(records):
        print(
            f"[{run_dir.name}] {len(records)} route records but {len(logs)} world logs; pairing the first {min(len(logs), len(records))}"
        )
    out = []
    for k, (record, log_path) in enumerate(zip(records, logs)):
        tag = f"{run_dir.name}_{record.get('route_id', k)}"
        out.append((record, log_path, videos[k] if k < len(videos) else None, tag))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="run dirs followed by the report dir")
    parser.add_argument("--max-inline", type=int, default=12)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    run_dirs, report_dir = args.paths[:-1], Path(args.paths[-1])
    for sub in ("strips", "speed", "videos"):
        (report_dir / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in run_dirs:
        for record, log_path, chase_mp4, tag in collect_routes(run_dir):
            rows.append(render_route(record, RouteLog(log_path), chase_mp4, report_dir, tag, not args.no_video))
            print(f"{tag}: {rows[-1]['categories']} | {rows[-1]['diagnosis']}")
    df = pd.DataFrame(rows).sort_values("driving_score")
    df.to_csv(report_dir / "routes.csv", index=False)
    category_counts = pd.Series([c for cats in df["categories"] for c in cats.split("|")]).value_counts()
    table = df[TABLE_COLS].round(2).copy()
    table["route"] = table["route"].map(lambda t: f"<a href='#{t}'>{t}</a>")
    html = [
        "<html><head><meta charset='utf-8'><title>PufferDrive CARLA leaderboard report</title>",
        "<style>body{font-family:sans-serif;max-width:1500px;margin:auto} table{border-collapse:collapse;font-size:12px} td,th{border:1px solid #ccc;padding:3px 6px}</style></head><body>",
        f"<h1>PufferDrive CARLA leaderboard &middot; {len(df)} routes &middot; mean driving score {df['driving_score'].mean():.1f} &middot; mean route completion {df['route_completion'].mean():.0f}%</h1>",
        (
            "<p>Orange = ego, blue = vehicles (red within 8 m), purple = pedestrians, green dashed = route with goal dots (current goal circled), "
            "stop lines colored by light state, x = infraction (black collision, red other). Speed plot: grey bands = standstills &ge; 8 s, "
            "squares along the bottom = state of the ego's own next light.</p>"
        ),
        "<h2>Mean metrics</h2>",
        df[
            [
                "driving_score",
                "route_completion",
                "infraction_penalty",
                "collisions",
                "red_lights",
                "stopped_frac",
                "creep_frac",
            ]
        ]
        .mean()
        .to_frame("mean")
        .T.round(3)
        .to_html(index=False),
        "<h2>Failure categories (route counts)</h2>",
        category_counts.to_frame("routes").to_html(),
        "<h2>All routes, worst first</h2>",
        table.to_html(index=False, escape=False),
        f"<h2>Worst {min(args.max_inline, len(df))} routes</h2>",
    ]
    for k, (_, r) in enumerate(df.iterrows()):
        html.append(route_section(r, inline=k < args.max_inline))
    html.append("</body></html>")
    (report_dir / "index.html").write_text("\n".join(html))
    print("failure categories:", category_counts.to_dict())
    print("report:", report_dir / "index.html")


if __name__ == "__main__":
    main()
