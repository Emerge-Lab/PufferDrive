"""Render the obs replay pages + index.html gallery of a tidied AlpaSim run's selected scenes, with score panels."""

import argparse
import html
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from render_obs_html import render_page

from opengigaflow import viz


PROGRESS_FLAG = "PROG"
SELECT_MODES = ("imperfect", "failed", "all")
REPLAY_SUFFIX = ".replay.zlib"
GALLERY_INDEX = "index.html"
CLIPGT_PATTERN = re.compile(r"clipgt-[0-9a-f-]{36}$")  # the scene id ends every page and replay name
SCORED_METRICS = (  # (metric, page-name flag, panel label)
    ("collision_at_fault", "COL", "at-fault collision"),
    ("offroad", "OFF", "offroad"),
    ("left_corridor_laterally", "CORR", "left corridor laterally"),
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
    """-> (obs_html/, results-summary.json, metrics_unprocessed.parquet, eval-config.yaml) of a tidied run."""
    obs_dir, details_dir = run_dir / "obs_html", run_dir / "details"
    if not obs_dir.is_dir():
        raise SystemExit(f"no obs_html/ under {run_dir}")
    summary_path = details_dir / "results-summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"no {summary_path}: the run has not been scored or tidied (tidy_alpasim_run.py)")
    return obs_dir, summary_path, details_dir / "metrics_unprocessed.parquet", details_dir / "eval-config.yaml"


def scene_rows(summary_path):
    """-> ({clipgt: results-summary rollout row}, the summary's score criteria)."""
    summary = json.loads(summary_path.read_text())
    rows = {}
    for row in summary["rollouts"]:
        if row["clipgt_id"] in rows:
            raise SystemExit(f"{row['clipgt_id']} has several rollouts; the scene's single replay cannot be paired")
        rows[row["clipgt_id"]] = row
    return rows, summary.get("score_criteria", {})


def max_gt_dist_m(eval_config_path):
    return float(yaml.safe_load(eval_config_path.read_text())["aggregation_modifiers"]["max_dist_to_gt_trajectory"])


def scene_series(metrics_path, rows):
    """clipgt -> its rollout's per-step metrics (PER_STEP_SERIES rows of metrics_unprocessed.parquet)."""
    df = pd.read_parquet(metrics_path, columns=["name", "timestamps_us", "values", "clipgt_id", "rollout_id"])
    df = df[df["name"].isin(PER_STEP_SERIES)]
    series = {}
    for (clipgt, rollout_id), scene_df in df.groupby(["clipgt_id", "rollout_id"]):
        if clipgt in rows and rows[clipgt]["rollout_id"] == rollout_id:
            series[clipgt] = scene_df
    return series


def scoring_timeline(df, frames, stop_gt_dist_m):
    """-> {"scored"/"after": {series: first replay step > 0 in/after the scoring window}, stop_step, stop_reasons}."""
    wide = df.pivot(index="timestamps_us", columns="name", values="values").sort_index()
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
    for name in [metric for metric, _, _ in SCORED_METRICS] + [series for series, _ in UNSCORED_SERIES]:
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
    """Fixed panel injected into the viewer: AlpaSim score, criteria and per-step metrics with their first step."""
    metrics, score_metrics = row["metrics"], row["score_metrics"]
    lines = []
    for metric, _, label in SCORED_METRICS:
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


def selected(row, select):
    """A scene without a summary row never finished scoring and counts as failed."""
    if select == "all" or row is None:
        return True
    return row["status"] == "fail" if select == "failed" else float(row["score"]) < 1.0


def replay_sources(obs_dir):
    """clipgt -> its .replay.zlib, else a page of the scene (the run's own or a rendered one, embedding the replay)."""
    sources = {path.name[: -len(REPLAY_SUFFIX)]: path for path in sorted(obs_dir.glob(f"*{REPLAY_SUFFIX}"))}
    for page in sorted(obs_dir.glob("*.html")):
        match = CLIPGT_PATTERN.search(page.stem)
        if match and match.group(0) not in sources:
            sources[match.group(0)] = page
    return sources


def scene_pages(obs_dir):
    """clipgt -> every page of the scene in obs_dir (the run's <clipgt>.html, earlier renders)."""
    pages = {}
    for page in sorted(obs_dir.glob("*.html")):
        match = CLIPGT_PATTERN.search(page.stem)
        if match:
            pages.setdefault(match.group(0), []).append(page)
    return pages


def page_name(clipgt, row):
    if row is None:
        return f"{clipgt}.html"
    flags = [flag for metric, flag, _ in SCORED_METRICS if float(row["score_metrics"][metric]) > 0]
    if float(row["score_metrics"]["progress_score"]) < 1.0:
        flags.append(PROGRESS_FLAG)
    return f"s{float(row['score']):.2f}_" + "".join(f"{flag}_" for flag in flags) + f"{clipgt}.html"


def gallery_metrics(row):
    """Failure flags in the names the self-play gallery filters on (rate > 0 = flagged)."""
    if row is None:
        return {}
    return {
        "collision_rate": float(row["metrics"]["collision_any"]),
        "at_fault_collision_rate": float(row["score_metrics"]["collision_at_fault"]),
        "offroad_rate": float(row["score_metrics"]["offroad"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="output dir of 7_eval_alpasim.sh (holds obs_html/ and details/)")
    ap.add_argument("--select", choices=SELECT_MODES, default="imperfect", help="scenes that get a page")
    ap.add_argument("--prune", action="store_true", help="delete every .replay.zlib once the pages are written")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()
    obs_dir, summary_path, metrics_path, eval_config_path = run_layout(Path(args.run_dir))
    sources = replay_sources(obs_dir)
    if not sources:
        raise SystemExit(f"no {obs_dir}/*{REPLAY_SUFFIX} or rendered pages")
    rows, criteria = scene_rows(summary_path)
    series = scene_series(metrics_path, rows)
    stop_gt_dist_m = max_gt_dist_m(eval_config_path)
    old_pages = scene_pages(obs_dir)
    jobs = [
        (clipgt, source, obs_dir / page_name(clipgt, rows.get(clipgt)))
        for clipgt, source in sources.items()
        if selected(rows.get(clipgt), args.select)
    ]

    def render(job):
        clipgt, source, html_path = job
        row = rows.get(clipgt)
        panel = ""
        if row is not None:
            frames = int(viz.read_replay_zlib(source, chunk_names=("metrics_f32",))[0]["frames"])
            panel = score_panel(row, criteria, scoring_timeline(series[clipgt], frames, stop_gt_dist_m))
        render_page(source, html_path, panel)
        for page in old_pages.get(clipgt, []):
            if page != html_path:
                page.unlink()  # the run's own page, or a render under a stale name

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(render, jobs))
    rendered = {clipgt for clipgt, _, _ in jobs}
    dropped_count = 0
    for clipgt, pages in old_pages.items():
        for page in pages if clipgt not in rendered else []:
            page.unlink()  # an unselected scene keeps no page
            dropped_count += 1
    file_metrics = {html_path.name: gallery_metrics(rows.get(clipgt)) for clipgt, _, html_path in jobs}
    if jobs:
        viz.build_gallery_index(str(obs_dir), file_metrics=file_metrics)
    pruned_count = 0
    if args.prune:
        for source in sources.values():
            if source.name.endswith(REPLAY_SUFFIX):
                source.unlink()
                pruned_count += 1
    unscored = sum(clipgt not in rows for clipgt in rendered)
    print(
        f"[render_alpasim_obs_html] {len(jobs)} of {len(sources)} scenes paged ({args.select})"
        f"{f', {unscored} without a score (no panel)' if unscored else ''}"
        f"{f', {dropped_count} unselected pages removed' if dropped_count else ''}"
        f"{f', {pruned_count} replays pruned' if pruned_count else ''} -> {obs_dir / GALLERY_INDEX}"
    )


if __name__ == "__main__":
    main()
