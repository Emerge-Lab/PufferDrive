"""Render the interactive observation replays (obs_html/<token>.replay.zlib) of a nuPlan run into HTML
pages plus an index.html gallery (same navigator as the self-play eval replays), only for the
scenarios worth looking at.

usage: python scripts/eval/render_obs_html.py <group_dir> [--max-score 0.9] [--metric no_ego_at_fault_collisions ...]
       [--tokens tok1 tok2 ...] [--prune] [--workers N]

Selection: aggregated score < --max-score OR any --metric < 1 OR explicitly listed --tokens.
Pages are named s<score>_<flags>_<scenario_type>_<token>.html so the gallery lists the worst first
(flags: COL at-fault collision, OFF drivable area, DIR driving direction, STALL no progress).
--prune deletes the .replay.zlib of every scenario that was not selected.
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from pufferlib import viz  # noqa: E402

FLAG_METRICS = (
    ("COL", "no_ego_at_fault_collisions"),
    ("OFF", "drivable_area_compliance"),
    ("DIR", "driving_direction_compliance"),
    ("STALL", "ego_is_making_progress"),
)


SCORED_METRICS = (
    ("no_ego_at_fault_collisions", "no at-fault collision", "multiplier"),
    ("drivable_area_compliance", "drivable area", "multiplier"),
    ("driving_direction_compliance", "driving direction", "multiplier"),
    ("ego_is_making_progress", "making progress", "multiplier"),
    ("time_to_collision_within_bound", "time to collision", "weight 5"),
    ("ego_progress_along_expert_route", "progress along expert route", "weight 5"),
    ("speed_limit_compliance", "speed limit", "weight 4"),
    ("ego_is_comfortable", "comfort", "weight 2"),
)
COMFORT_METRICS = ("ego_lon_acceleration", "ego_lon_jerk", "ego_jerk", "ego_lat_acceleration", "ego_yaw_acceleration", "ego_yaw_rate")
COMFORT_BOUNDS = {
    "ego_lon_acceleration": "-4.05..2.40 m/s^2",
    "ego_lon_jerk": "|j| <= 4.13 m/s^3",
    "ego_jerk": "|j| <= 8.37 m/s^3",
    "ego_lat_acceleration": "|a| <= 4.89 m/s^2",
    "ego_yaw_acceleration": "|a| <= 1.93 rad/s^2",
    "ego_yaw_rate": "|w| <= 0.95 rad/s",
}


def scenario_rows(group_dir):
    """token -> aggregator row (score + per-metric values) over every challenge of the run."""
    rows = {}
    for csv in glob.glob(f"{group_dir}/simulation/*/20*/aggregator_metric/*_weighted_average_metrics_*.csv"):
        agg = pd.read_csv(csv)
        per_scenario = agg[(agg["scenario_type"] != "final_score") & (agg["scenario"] != agg["scenario_type"])]
        for _, row in per_scenario.iterrows():
            rows[str(row["scenario"])] = row
    return rows


def comfort_failures(group_dir):
    """token -> ['ego_lon_jerk -2.7..5.0', ...]: nuPlan's own out-of-bound comfort sub-metrics."""
    out = {}
    for sim_dir in glob.glob(f"{group_dir}/simulation/*/20*"):
        for name in COMFORT_METRICS:
            f = Path(sim_dir) / "metrics" / f"{name}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f)
            flag = [c for c in df.columns if c.endswith("within_bounds_stat_value")]
            if not flag:
                continue
            for _, r in df[~df[flag[0]].astype(bool)].iterrows():
                out.setdefault(str(r["scenario_name"]), []).append(
                    f"{name} {float(r[f'min_{name}_stat_value']):.2f}..{float(r[f'max_{name}_stat_value']):.2f} (bound {COMFORT_BOUNDS[name]})"
                )
    return out


def score_panel(token, row, comfort):
    """Fixed panel injected into the viewer: nuPlan score, every scored sub-metric, violated ones highlighted."""
    if row is None:
        return ""
    lines = []
    for metric, label, role in SCORED_METRICS:
        value = float(row[metric])
        color = "#ff6b6b" if value < 1.0 else "#7ed491"
        mark = "&#10007;" if value < 1.0 else "&#10003;"
        lines.append(
            f'<div style="display:flex;justify-content:space-between;gap:16px"><span style="color:{color}">{mark} {label}</span>'
            f'<span style="color:{color}">{value:.2f} <span style="color:#7f8ba0">({role})</span></span></div>'
        )
    for detail in comfort:
        lines.append(f'<div style="color:#ff6b6b;padding-left:14px">&#8627; {detail}</div>')
    return (
        '<details open style="position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:1000;'
        'background:rgba(13,20,32,.93);color:#c4cddc;font:12px/1.5 ui-monospace,monospace;padding:6px 12px;'
        'border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.4);max-width:640px">'
        f'<summary style="cursor:pointer;font-weight:600">nuPlan score {float(row["score"]):.2f} &middot; {row["scenario_type"]} &middot; {token}</summary>'
        + "".join(lines)
        + "</details>"
    )


def render_page(zlib_path, html_path, panel):
    viz.render_interactive_replay_zlib(str(zlib_path), str(html_path))
    if panel:
        html = Path(html_path).read_text()
        Path(html_path).write_text(html.replace("<body>", "<body>" + panel, 1))


def selected_tokens(rows, max_score, metrics):
    tokens = set()
    for token, row in rows.items():
        if row["score"] < max_score or any(row[metric] < 1.0 for metric in metrics):
            tokens.add(token)
    return tokens


def page_name(token, row):
    if row is None:
        return f"{token}.html"
    flags = "".join(f"{flag}_" for flag, metric in FLAG_METRICS if row[metric] < 1.0)
    return f"s{row['score']:.2f}_{flags}{row['scenario_type']}_{token}.html"


def gallery_metrics(row):
    """Failure flags in the names the self-play gallery filters on (rate > 0 = flagged)."""
    if row is None:
        return {}
    return {
        "score": float(row["score"]),
        "at_fault_collision_rate": 1.0 - float(row["no_ego_at_fault_collisions"]),
        "offroad_rate": 1.0 - float(row["drivable_area_compliance"]),
        "collision_rate": 1.0 - float(row["no_ego_at_fault_collisions"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group_dir")
    ap.add_argument("--max-score", type=float, default=1.01)
    ap.add_argument("--metric", action="append", default=[])
    ap.add_argument("--tokens", nargs="*", default=[])
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    obs_dir = Path(args.group_dir) / "obs_html"
    replays = {p.name[: -len(".replay.zlib")]: p for p in obs_dir.glob("*.replay.zlib")}
    if not replays:
        raise SystemExit(f"no obs_html/*.replay.zlib under {args.group_dir}")
    rows = scenario_rows(args.group_dir)
    comfort = comfort_failures(args.group_dir)
    wanted = selected_tokens(rows, args.max_score, args.metric) | set(args.tokens)
    jobs = [(token, replays[token], obs_dir / page_name(token, rows.get(token))) for token in sorted(wanted) if token in replays]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(lambda job: render_page(job[1], job[2], score_panel(job[0], rows.get(job[0]), comfort.get(job[0], []))), jobs))
    file_metrics = {html.name: gallery_metrics(rows.get(token)) for token, _, html in jobs}
    pruned = 0
    if args.prune:
        for token, path in replays.items():
            if token not in wanted:
                path.unlink()
                pruned += 1
    if jobs:
        viz.build_gallery_index(str(obs_dir), file_metrics=file_metrics)
    missing = len(wanted) - len(jobs)
    print(
        f"[render_obs_html] {len(replays)} replays, {len(wanted)} scenarios selected, {len(jobs)} pages rendered"
        f"{f', {missing} selected without a replay' if missing else ''}{f', {pruned} replays pruned' if pruned else ''}"
        f" -> {obs_dir / 'index.html'}"
    )


if __name__ == "__main__":
    main()
