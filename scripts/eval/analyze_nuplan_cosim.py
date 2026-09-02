"""Failure analysis + visualizations for a nuPlan closed-loop run of the PufferDrive planner
(pufferlib/ocean/cosim/nuplan). Needs the nuPlan devkit env (same interpreter as the simulation).

usage: python scripts/eval/analyze_nuplan_cosim.py <group_dir> <report_dir> [--workers N] [--max-inline N] [--no-video]

<group_dir> is the run_simulation.py `group` directory (holds simulation/<challenge>/<timestamp>/...).
Writes <report_dir>/index.html (score table sorted worst-first, failure-category counts, per-scenario
diagnosis, the worst --max-inline scenarios inline with a six-frame strip + speed plot + video), plus
strips/, speed/, videos/ and scenarios.csv. Per-scenario work runs in a multiprocessing pool.
"""

import argparse
import glob
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from nuplan.common.actor_state.state_representation import Point2D  # noqa: E402
from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # noqa: E402
from nuplan.planning.simulation.simulation_log import SimulationLog  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nuplan_log_remap  # noqa: E402  logs written on another machine (cluster paths, planner classes)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from pufferlib.ocean.cosim.carla_cosim import write_mp4  # noqa: E402

SCORE_COLS = [
    "score",
    "ego_progress_along_expert_route",
    "ego_is_making_progress",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "no_ego_at_fault_collisions",
    "time_to_collision_within_bound",
    "ego_is_comfortable",
    "speed_limit_compliance",
]
COMFORT_METRICS = ["ego_lon_acceleration", "ego_lon_jerk", "ego_jerk", "ego_lat_acceleration", "ego_yaw_acceleration", "ego_yaw_rate"]
LIGHT_COLOR = {"RED": "red", "YELLOW": "orange", "GREEN": "limegreen", "UNKNOWN": "0.6"}
STOPPED_SPEED_MPS = 0.3
CLOSE_AGENT_M = 8.0
LANE_CHOICE_OFFSET_M = 3.0
VIDEO_FRAME_STRIDE = 2
VIDEO_FPS = 5


def draw_box(ax, cx, cy, heading, length, width, color, alpha=0.7):
    c, s = math.cos(heading), math.sin(heading)
    corners = [(length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2)]
    pts = [(cx + c * dx - s * dy, cy + s * dx + c * dy) for dx, dy in corners]
    ax.add_patch(Polygon(pts, closed=True, color=color, alpha=alpha))
    ax.plot([cx, cx + c * length / 2], [cy, cy + s * length / 2], color="k", lw=0.8)


def draw_scene(ax, sample, scenario, expert_xy, span=40.0, title=""):
    ego = sample.ego_state
    objs = scenario.map_api.get_proximal_map_objects(
        Point2D(ego.center.x, ego.center.y), span * 1.3, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    )
    lights = {str(t.lane_connector_id): t.status.name for t in sample.traffic_light_status}
    for layer, lanes in objs.items():
        for lane in lanes:
            pts = np.array([[p.x, p.y] for p in lane.baseline_path.discrete_path])
            is_connector = layer == SemanticMapLayer.LANE_CONNECTOR
            color = LIGHT_COLOR.get(lights.get(str(lane.id)), "0.8") if is_connector else "0.8"
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.6 if str(lane.id) in lights else 0.7)
    ax.plot(expert_xy[:, 0], expert_xy[:, 1], "g--", lw=1.2)
    for o in sample.observation.tracked_objects.get_agents():
        d = math.hypot(o.center.x - ego.center.x, o.center.y - ego.center.y)
        color = "tab:red" if d < CLOSE_AGENT_M else "tab:blue"
        draw_box(ax, o.center.x, o.center.y, o.center.heading, o.box.length, o.box.width, color)
    for o in sample.observation.tracked_objects.get_static_objects():
        draw_box(ax, o.center.x, o.center.y, o.center.heading, o.box.length, o.box.width, "0.4", 0.5)
    fp = ego.car_footprint
    draw_box(ax, ego.center.x, ego.center.y, ego.center.heading, fp.length, fp.width, "tab:orange", 0.95)
    ax.set_xlim(ego.center.x - span, ego.center.x + span)
    ax.set_ylim(ego.center.y - span, ego.center.y + span)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def fig_to_rgb(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def trajectory_stats(hist, scenario):
    ego = np.array(
        [
            [
                s.ego_state.center.x,
                s.ego_state.center.y,
                s.ego_state.center.heading,
                s.ego_state.dynamic_car_state.speed,
                s.ego_state.dynamic_car_state.center_acceleration_2d.x,
            ]
            for s in hist.data
        ]
    )
    expert = np.array([[s.center.x, s.center.y, s.dynamic_car_state.speed] for s in scenario.get_expert_ego_trajectory()])
    min_dist, min_dist_iter, min_dist_speed = np.inf, 0, 0.0
    for i, s in enumerate(hist.data):
        e = s.ego_state.center
        for o in s.observation.tracked_objects.get_agents():
            d = math.hypot(o.center.x - e.x, o.center.y - e.y)
            if d < min_dist:
                min_dist, min_dist_iter, min_dist_speed = d, i, o.velocity.magnitude()
    lateral = [0.0]  # offset from the expert path, only while the ego is still alongside it (not past its end)
    for x, y in ego[:, :2]:
        d = np.hypot(expert[:, 0] - x, expert[:, 1] - y)
        k = int(np.argmin(d))
        if k < len(expert) - 1:
            lateral.append(float(d[k]))
    stats = dict(
        iterations=len(ego),
        ego_dist_m=float(np.hypot(*np.diff(ego[:, :2], axis=0).T).sum()),
        expert_dist_m=float(np.hypot(*np.diff(expert[:, :2], axis=0).T).sum()),
        ego_max_speed=float(ego[:, 3].max()),
        expert_max_speed=float(expert[:, 2].max()),
        stopped_frac=float((ego[:, 3] < STOPPED_SPEED_MPS).mean()),
        expert_stopped_frac=float((expert[:, 2] < STOPPED_SPEED_MPS).mean()),
        max_lateral_from_expert_m=float(np.max(lateral)),
        min_agent_dist_m=float(min_dist),
        min_agent_dist_iter=int(min_dist_iter),
        min_agent_dist_other_speed=float(min_dist_speed),
        min_accel=float(ego[:, 4].min()),
        max_accel=float(ego[:, 4].max()),
    )
    return ego, expert, stats


def comfort_failures(sim_dir, token):
    """nuPlan's own within-bound flags per comfort sub-metric -> ['ego_lon_jerk range -2.7..5.0', ...]."""
    out = []
    for name in COMFORT_METRICS:
        f = Path(sim_dir) / "metrics" / f"{name}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        rows = df[df["scenario_name"] == token]
        if rows.empty:
            continue
        r = rows.iloc[0]
        flag = [c for c in df.columns if c.endswith("within_bounds_stat_value")]
        if flag and not bool(r[flag[0]]):
            out.append(f"{name} range {float(r[f'min_{name}_stat_value']):.2f}..{float(r[f'max_{name}_stat_value']):.2f}")
    return out


def diagnose(row, stats, comfort):
    """-> (categories, reasons). Categories are the aggregate failure buckets, reasons the per-scenario text."""
    categories, reasons = [], []
    if row["ego_progress_along_expert_route"] < 0.5 and stats["expert_dist_m"] > 5:
        if stats["stopped_frac"] > 0.6:
            categories.append("stalled")
            reasons.append(f"stalled: stopped {stats['stopped_frac']:.0%} of the time (expert {stats['expert_stopped_frac']:.0%})")
        else:
            categories.append("slow progress")
            reasons.append(f"slow/short progress: drove {stats['ego_dist_m']:.0f} m vs expert {stats['expert_dist_m']:.0f} m")
    if row["no_ego_at_fault_collisions"] < 1:
        categories.append("collision")
        reasons.append(f"at-fault collision (closest agent {stats['min_agent_dist_m']:.1f} m at iter {stats['min_agent_dist_iter']})")
    if row["drivable_area_compliance"] < 1:
        categories.append("offroad")
        reasons.append(f"left drivable area (max {stats['max_lateral_from_expert_m']:.1f} m from expert path)")
    if row["driving_direction_compliance"] < 1:
        categories.append("wrong way")
        reasons.append("drove against traffic direction")
    if row["time_to_collision_within_bound"] < 1:
        categories.append("TTC")
        reasons.append(
            f"TTC below bound: closest agent {stats['min_agent_dist_m']:.1f} m (moving {stats['min_agent_dist_other_speed']:.1f} m/s) "
            f"at iter {stats['min_agent_dist_iter']}"
        )
    if row["ego_is_comfortable"] < 1:
        categories.append("comfort")
        reasons.append("uncomfortable: " + ", ".join(comfort or ["sub-metric not identified"]))
    if row["speed_limit_compliance"] < 1:
        categories.append("speeding")
        reasons.append(f"speeding: max {stats['ego_max_speed']:.1f} m/s (expert {stats['expert_max_speed']:.1f})")
    if stats["max_lateral_from_expert_m"] > LANE_CHOICE_OFFSET_M and row["drivable_area_compliance"] >= 1:
        reasons.append(f"lane choice differs from expert (up to {stats['max_lateral_from_expert_m']:.1f} m off the expert path)")
    if stats["ego_dist_m"] > stats["expert_dist_m"] + 5:
        reasons.append(f"drove further than the expert ({stats['ego_dist_m']:.0f} m vs {stats['expert_dist_m']:.0f} m)")
    return categories or ["none"], reasons or ["no metric failed"]


def process_scenario(job):
    sim_dir, token, row, report_dir, make_video = job
    report_dir = Path(report_dir)
    log = nuplan_log_remap.load_log(row["log_path"])
    hist, scenario = log.simulation_history, log.scenario
    ego, expert, stats = trajectory_stats(hist, scenario)
    categories, reasons = diagnose(row, stats, comfort_failures(sim_dir, token))
    n = len(hist.data)
    picks = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]
    if stats["min_agent_dist_iter"] not in picks:
        picks[3] = stats["min_agent_dist_iter"]
    picks = sorted(set(picks))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.8))
    for ax, it in zip(axes, picks):
        draw_scene(ax, hist.data[it], scenario, expert[:, :2], title=f"it {it} v={ego[it, 3]:.1f} m/s")
    fig.suptitle(f"{token} {row['scenario_type']} score={row['score']:.2f}", fontsize=10)
    plt.tight_layout()
    fig.savefig(report_dir / "strips" / f"{token}.png", dpi=70)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 2.2))
    dt = scenario.database_interval
    ax.plot(np.arange(len(ego)) * dt, ego[:, 3], label="ego")
    ax.plot(np.arange(len(expert)) * dt, expert[:, 2], "g--", label="expert")
    ax.set_xlabel("s")
    ax.set_ylabel("m/s")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(report_dir / "speed" / f"{token}.png", dpi=70)
    plt.close(fig)

    if make_video:
        frames = []
        for it in range(0, n, VIDEO_FRAME_STRIDE):
            fig, ax = plt.subplots(figsize=(7, 7), dpi=90)
            draw_scene(ax, hist.data[it], scenario, expert[:, :2], title=f"{row['scenario_type']} it {it} v={ego[it, 3]:.1f} m/s")
            frames.append(fig_to_rgb(fig))
            plt.close(fig)
        write_mp4(report_dir / "videos" / f"{token}.mp4", frames, fps=VIDEO_FPS)
    return {
        "token": token,
        "type": row["scenario_type"],
        "log": row["log_name"],
        "map": scenario.map_api.map_name,
        **{c: float(row[c]) for c in SCORE_COLS},
        **stats,
        "categories": "|".join(categories),
        "diagnosis": "; ".join(reasons),
    }


def collect_jobs(group_dir, report_dir, make_video):
    jobs = []
    for sim_dir in sorted(glob.glob(f"{group_dir}/simulation/*/20*")):
        agg_files = glob.glob(f"{sim_dir}/aggregator_metric/*_weighted_average_metrics_*.csv")
        if not agg_files:
            continue
        agg = pd.read_csv(agg_files[0])
        per_scenario = agg[(agg["scenario_type"] != "final_score") & (agg["scenario"] != agg["scenario_type"])]
        logs = {Path(p).parent.name: p for p in glob.glob(f"{sim_dir}/simulation_log/**/*.msgpack.xz", recursive=True)}
        for _, row in per_scenario.iterrows():
            token = str(row["scenario"])
            if token not in logs:
                continue
            row = row.to_dict()
            row["log_path"] = logs[token]
            jobs.append((sim_dir, token, row, str(report_dir), make_video))
    return jobs


def scenario_section(r, report_dir, inline):
    header = f"<h2 id='{r['token']}'>{r['token']} &middot; {r['type']} &middot; {r['map']} &middot; score {r['score']:.2f}</h2>"
    scores = " &nbsp; ".join(f"{c.replace('_', ' ')}: <b>{r[c]:.2f}</b>" for c in SCORE_COLS[1:])
    facts = (
        f"ego {r['ego_dist_m']:.0f} m / expert {r['expert_dist_m']:.0f} m, max speed {r['ego_max_speed']:.1f} / {r['expert_max_speed']:.1f} m/s, "
        f"stopped {r['stopped_frac']:.0%} / {r['expert_stopped_frac']:.0%}, closest agent {r['min_agent_dist_m']:.1f} m"
    )
    media = (
        f'<img src="strips/{r["token"]}.png" style="max-width:100%"><br><img src="speed/{r["token"]}.png"><br>'
        + (f'<video src="videos/{r["token"]}.mp4" controls width="420"></video>' if (report_dir / "videos" / f"{r['token']}.mp4").exists() else "")
        if inline
        else f'<a href="strips/{r["token"]}.png">frames</a> &middot; <a href="speed/{r["token"]}.png">speed</a>'
        + (f' &middot; <a href="videos/{r["token"]}.mp4">video</a>' if (report_dir / "videos" / f"{r['token']}.mp4").exists() else "")
    )
    return f"{header}<p><b>Diagnosis:</b> {r['diagnosis']}</p><p>{scores}</p><p>{facts}</p>{media}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group_dir")
    ap.add_argument("report_dir")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-inline", type=int, default=40, help="worst scenarios shown inline; the rest get links")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    report_dir = Path(args.report_dir)
    for sub in ("strips", "speed", "videos"):
        (report_dir / sub).mkdir(parents=True, exist_ok=True)
    jobs = collect_jobs(args.group_dir, report_dir, not args.no_video)
    if not jobs:
        raise SystemExit(f"no scenarios with aggregator metrics + simulation logs under {args.group_dir}")
    print(f"[analyze_nuplan_cosim] {len(jobs)} scenarios, {args.workers} workers")
    with Pool(args.workers) as pool:
        rows = pool.map(process_scenario, jobs, chunksize=1)
    df = pd.DataFrame(rows).sort_values("score")
    df.to_csv(report_dir / "scenarios.csv", index=False)

    category_counts = pd.Series([c for cats in df["categories"] for c in cats.split("|")]).value_counts()
    by_type = df.groupby("type")[["score", "ego_progress_along_expert_route", "no_ego_at_fault_collisions", "drivable_area_compliance", "ego_is_comfortable", "speed_limit_compliance"]].mean().sort_values("score")
    table_cols = ["token", "type", "map", "score", "ego_progress_along_expert_route", "no_ego_at_fault_collisions", "drivable_area_compliance", "time_to_collision_within_bound", "ego_is_comfortable", "speed_limit_compliance", "categories"]
    table = df[table_cols].round(2).copy()
    table["token"] = table["token"].map(lambda t: f"<a href='#{t}'>{t}</a>")
    html = [
        "<html><head><meta charset='utf-8'><title>PufferDrive nuPlan closed-loop report</title>",
        "<style>body{font-family:sans-serif;max-width:1500px;margin:auto} table{border-collapse:collapse;font-size:12px} td,th{border:1px solid #ccc;padding:3px 6px}</style></head><body>",
        f"<h1>PufferDrive nuPlan closed-loop &middot; {len(df)} scenarios &middot; mean score {df['score'].mean():.3f}</h1>",
        "<p>Orange = ego, blue = other agents (red within 8 m), green dashed = expert path, lane connectors colored by reported light state.</p>",
        "<h2>Mean metrics</h2>", df[SCORE_COLS].mean().to_frame("mean").T.round(3).to_html(index=False),
        "<h2>Failure categories (scenario counts)</h2>", category_counts.to_frame("scenarios").to_html(),
        "<h2>By scenario type</h2>", by_type.round(3).to_html(),
        "<h2>All scenarios, worst first</h2>", table.to_html(index=False, escape=False),
        f"<h2>Worst {min(args.max_inline, len(df))} scenarios</h2>",
    ]
    for k, (_, r) in enumerate(df.iterrows()):
        html.append(scenario_section(r, report_dir, inline=k < args.max_inline))
    html.append("</body></html>")
    (report_dir / "index.html").write_text("\n".join(html))
    print(df[SCORE_COLS].mean().round(3).to_string())
    print("failure categories:", category_counts.to_dict())
    print("report:", report_dir / "index.html")


if __name__ == "__main__":
    main()
