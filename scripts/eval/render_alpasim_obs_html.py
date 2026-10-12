"""Render the interactive observation replays of an AlpaSim run (scripts/kesai/12_alpasim.sh:
<run_dir>/obs_replays/<clipgt>.replay.zlib) into one page per scene plus an index.html gallery (same
navigator as the nuPlan, CARLA and self-play eval replays).

usage: python scripts/eval/render_alpasim_obs_html.py <run_dir> [--workers N]

Pages are named s<score>_<flags>_<clipgt>.html so the gallery lists the worst first (flags: COL at-fault
collision, OFF offroad, CORR left the corridor laterally, PROG progress score < 1); each page carries the
scene's AlpaSim score criteria and per-metric values as a panel, every violated per-step metric linking to the
replay step of its first violation. The page the run wrote (<clipgt>.html) is replaced by the renamed one.
Runs before 2026-10-12 keep everything under <run_dir>/alpagym/; both layouts are read.
"""

import argparse
import html
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from pufferlib import viz  # noqa: E402

FLAG_METRICS = (
    ("COL", "collision_at_fault"),
    ("OFF", "offroad"),
    ("CORR", "left_corridor_laterally"),
)
PROGRESS_FLAG = "PROG"
SCORED_METRICS = (
    ("collision_at_fault", "at-fault collision"),
    ("offroad", "offroad"),
    ("left_corridor_laterally", "left corridor laterally"),
)
UNSCORED_SERIES = (
    ("collision_any", "collision (any)"),
    ("collision_front", "collision front"),
    ("collision_lateral", "collision lateral"),
    ("collision_rear", "collision rear"),
    ("wrong_lane", "wrong lane"),
)
PER_STEP_SERIES = (
    "collision_any",
    "collision_front",
    "collision_lateral",
    "collision_rear",
    "offroad",
    "left_corridor_laterally",
    "wrong_lane",
    "eval_relevant",
    "dist_to_gt_trajectory",
)
ALPASIM_STEP_SECONDS = 0.1


def run_layout(run_dir):
    """-> (obs replay dir, AlpaSim wizard dir) of the tidied layout, else of the pre-2026-10-12 alpagym/ nesting."""
    for base in (run_dir, run_dir / "alpagym"):
        if (base / "obs_replays").is_dir():
            return base / "obs_replays", base / "alpasim" / "wizard_0"
    raise SystemExit(f"no obs_replays/ under {run_dir} or {run_dir / 'alpagym'}")


def scene_rows(wizard_dir):
    """clipgt -> results-summary rollout row (score, status, failure_reason, metrics, score_metrics)."""
    summary_path = wizard_dir / "aggregate" / "results-summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"no {summary_path}: the run has not been scored")
    summary = json.loads(summary_path.read_text())
    rows = {}
    for row in summary["rollouts"]:
        if row["clipgt_id"] in rows:
            raise SystemExit(f"{row['clipgt_id']} has several rollouts; the scene's single replay cannot be paired")
        rows[row["clipgt_id"]] = row
    return rows, summary.get("score_criteria", {})


def max_gt_dist_m(wizard_dir):
    return float(
        yaml.safe_load((wizard_dir / "eval-config.yaml").read_text())["aggregation_modifiers"][
            "max_dist_to_gt_trajectory"
        ]
    )


def scoring_timeline(wizard_dir, row, frames, stop_gt_dist_m):
    """AlpaSim scores from the first eval_relevant step through the first collision, offroad or stop_gt_dist_m
    from the gt trajectory. -> {scored: series -> first replay step > 0 inside that window, after: first one past it,
    stop_step, stop_reasons}. The replay ends with the per-step series, so index i is replay step i - (length - frames)."""
    df = pd.read_parquet(wizard_dir / "rollouts" / row["clipgt_id"] / row["rollout_id"] / "metrics.parquet")
    wide = (
        df[df["name"].isin(PER_STEP_SERIES)].pivot(index="timestamps_us", columns="name", values="values").sort_index()
    )
    wide["collision_at_fault"] = ((wide["collision_front"] > 0) | (wide["collision_lateral"] > 0)).astype(np.float64)
    offset = len(wide) - frames
    relevant = np.flatnonzero(wide["eval_relevant"].to_numpy() > 0)
    start = int(relevant[0]) if relevant.size else len(wide)
    stop_events = {
        "collision": wide["collision_any"].to_numpy() > 0,
        "offroad": wide["offroad"].to_numpy() > 0,
        f">= {stop_gt_dist_m:g} m from the gt trajectory": wide["dist_to_gt_trajectory"].to_numpy() >= stop_gt_dist_m,
    }
    stop_hits = np.flatnonzero(np.logical_or.reduce(list(stop_events.values())) & (np.arange(len(wide)) >= start))
    stop = int(stop_hits[0]) if stop_hits.size else len(wide) - 1
    timeline = {"scored": {}, "after": {}, "stop_step": None, "stop_reasons": []}
    if stop_hits.size:
        timeline["stop_step"] = stop - offset
        timeline["stop_reasons"] = [reason for reason, hits in stop_events.items() if hits[stop]]
    for name in [metric for metric, _ in SCORED_METRICS] + [series for series, _ in UNSCORED_SERIES]:
        hits = np.flatnonzero(wide[name].to_numpy() > 0)
        inside, later = hits[(hits >= start) & (hits <= stop)], hits[hits > stop]
        if inside.size:
            timeline["scored"][name] = int(inside[0]) - offset
        if later.size:
            timeline["after"][name] = int(later[0]) - offset
    return timeline


def step_html(step):
    if step < 0:
        return f"{-step * ALPASIM_STEP_SECONDS:.1f} s before the handover"
    return viz.replay_step_link(step, step * ALPASIM_STEP_SECONDS)


def metric_line(label, value, failed, name, timeline):
    color = "#ff6b6b" if failed else "#7ed491"
    mark = "&#10007;" if failed else "&#10003;"
    detail = ""
    if failed and name in timeline["scored"]:
        detail = f" &middot; {step_html(timeline['scored'][name])}"
    elif not failed and name in timeline["after"]:
        detail = (
            f' <span style="color:#7f8ba0">&middot; {step_html(timeline["after"][name])}, after scoring stopped</span>'
        )
    return (
        f'<div style="display:flex;justify-content:space-between;gap:16px"><span style="color:{color}">{mark} {label}</span>'
        f'<span style="color:{color}">{value}{detail}</span></div>'
    )


def score_panel(row, criteria, timeline):
    """Fixed panel injected into the viewer: AlpaSim score, every score criterion, the unscored per-step metrics,
    each violation linking to its first step in the scored window."""
    metrics, score_metrics = row["metrics"], row["score_metrics"]
    lines = []
    for metric, label in SCORED_METRICS:
        value = float(score_metrics[metric])
        criterion = html.escape(criteria.get(metric, ""))
        lines.append(
            metric_line(
                label, f'{value:.0f} <span style="color:#7f8ba0">({criterion})</span>', value > 0, metric, timeline
            )
        )
    progress_score = float(score_metrics["progress_score"])
    lines.append(
        metric_line(
            "progress score",
            f'{progress_score:.2f} <span style="color:#7f8ba0">(clipped rel {float(score_metrics["progress_clipped_rel"]):.2f}, '
            f"full at 0.8)</span>",
            progress_score < 1.0,
            "progress",
            timeline,
        )
    )
    if timeline["stop_step"] is not None:
        lines.append(
            f'<div style="color:#7f8ba0">scoring stopped at {step_html(timeline["stop_step"])}: '
            f"{', '.join(timeline['stop_reasons'])}</div>"
        )
    for series, label in UNSCORED_SERIES:
        value = float(metrics[series])
        lines.append(
            metric_line(f'<span style="color:#7f8ba0">{label}</span>', f"{value:.0f}", value > 0, series, timeline)
        )
    lines.append(
        f'<div style="color:#7f8ba0">driven {float(metrics["dist_traveled_m"]):.0f} m of gt {float(metrics["gt_dist_traveled_m"]):.0f} m &middot; '
        f"gt trajectory dist {float(metrics['dist_to_gt_trajectory']):.1f} m (lateral {float(metrics['lateral_dist_to_gt_trajectory']):.1f} m)</div>"
    )
    failure = f" &middot; {html.escape(row['failure_reason'])}" if row["failure_reason"] else ""
    return (
        '<details open style="position:fixed;top:8px;left:50%;transform:translateX(-50%);z-index:1000;'
        "background:rgba(13,20,32,.93);color:#c4cddc;font:12px/1.5 ui-monospace,monospace;padding:6px 12px;"
        'border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.4);max-width:640px">'
        f'<summary style="cursor:pointer;font-weight:600">AlpaSim score {float(row["score"]):.2f} &middot; '
        f"{html.escape(row['status'])}{failure} &middot; {html.escape(row['clipgt_id'])}</summary>"
        + "".join(lines)
        + "</details>"
    )


def page_name(clipgt, row):
    if row is None:
        return f"{clipgt}.html"
    flags = [flag for flag, metric in FLAG_METRICS if float(row["score_metrics"][metric]) > 0]
    if float(row["score_metrics"]["progress_score"]) < 1.0:
        flags.append(PROGRESS_FLAG)
    return f"s{float(row['score']):.2f}_" + "".join(f"{flag}_" for flag in flags) + f"{clipgt}.html"


def gallery_metrics(row):
    """Failure flags in the names the self-play gallery filters on (rate > 0 = flagged)."""
    if row is None:
        return {}
    return {
        "score": float(row["score"]),
        "collision_rate": float(row["metrics"]["collision_any"]),
        "at_fault_collision_rate": float(row["score_metrics"]["collision_at_fault"]),
        "offroad_rate": float(row["score_metrics"]["offroad"]),
    }


def render_page(zlib_path, html_path, panel):
    viz.render_interactive_replay_zlib(str(zlib_path), str(html_path))
    if panel:
        page = Path(html_path).read_text()
        Path(html_path).write_text(page.replace("<body>", "<body>" + panel, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="output dir of 12_alpasim.sh (holds obs_replays/ and alpasim/wizard_0/)")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    obs_dir, wizard_dir = run_layout(Path(args.run_dir))
    replays = {p.name[: -len(".replay.zlib")]: p for p in sorted(obs_dir.glob("*.replay.zlib"))}
    if not replays:
        raise SystemExit(f"no {obs_dir}/*.replay.zlib")
    rows, criteria = scene_rows(wizard_dir)
    stop_gt_dist_m = max_gt_dist_m(wizard_dir)
    jobs = [(clipgt, replay, obs_dir / page_name(clipgt, rows.get(clipgt))) for clipgt, replay in replays.items()]

    def render(job):
        clipgt, replay, html_path = job
        row = rows.get(clipgt)
        panel = ""
        if row is not None:
            frames = int(viz.read_replay_zlib(replay, chunk_names=("metrics_f32",))[0]["frames"])
            panel = score_panel(row, criteria, scoring_timeline(wizard_dir, row, frames, stop_gt_dist_m))
        render_page(replay, html_path, panel)
        run_page = obs_dir / f"{clipgt}.html"
        if run_page != html_path and run_page.exists():
            run_page.unlink()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(render, jobs))
    file_metrics = {html_path.name: gallery_metrics(rows.get(clipgt)) for clipgt, _, html_path in jobs}
    viz.build_gallery_index(str(obs_dir), file_metrics=file_metrics)
    unscored = sum(clipgt not in rows for clipgt in replays)
    print(
        f"[render_alpasim_obs_html] {len(jobs)} scene pages rendered"
        f"{f', {unscored} without a score (no panel)' if unscored else ''} -> {obs_dir / 'index.html'}"
    )


if __name__ == "__main__":
    main()
