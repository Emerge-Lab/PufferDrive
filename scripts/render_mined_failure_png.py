#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from tqdm import tqdm

import pufferlib.mining_viz

sns.set_theme()
plt.rcParams["figure.figsize"] = (5, 5)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

TARGET_COLOR = "#ff1f5b"
ADVERSARIAL_COLOR = "#009ade"
LANE_MARKING_COLOR = "#ffc61e"


ROAD_STYLES = {
    "lane": {"color": "#d7dde3", "width": 0.9, "alpha": 0.72, "linestyle": "solid"},
    "yellow_line": {"color": LANE_MARKING_COLOR, "width": 1.15, "alpha": 0.90, "linestyle": (0, (4, 4))},
    "road_line": {"color": "#8f98a3", "width": 0.75, "alpha": 0.55, "linestyle": (0, (4, 4))},
    "edge": {"color": "#111111", "width": 0.9, "alpha": 0.85, "linestyle": "solid"},
}


def road_style(elem_type):
    elem_type = int(elem_type or 0)
    if 1 <= elem_type <= 3:
        return ROAD_STYLES["lane"]
    if elem_type == 14:
        return ROAD_STYLES["yellow_line"]
    if 11 <= elem_type <= 18:
        return ROAD_STYLES["road_line"]
    if 21 <= elem_type <= 23:
        return ROAD_STYLES["edge"]
    return None


def agent_slots(agent_arrays):
    valid = agent_arrays["valid"]
    seen = {}
    for frame_idx in range(valid.shape[0]):
        for slot_idx in np.flatnonzero(valid[frame_idx]):
            agent_id = int(agent_arrays["id"][frame_idx, slot_idx])
            seen.setdefault(agent_id, slot_idx)
    return seen


def trajectory(agent_arrays, slot_idx):
    valid = agent_arrays["valid"][:, slot_idx]
    frames = np.flatnonzero(valid)
    return {
        "frames": frames,
        "x": agent_arrays["x"][frames, slot_idx].astype(float),
        "y": agent_arrays["y"][frames, slot_idx].astype(float),
    }


def crop_bounds(agent_arrays, focus_slots, pad=28.0):
    xs = []
    ys = []
    for slot_idx in focus_slots:
        if slot_idx is None:
            continue
        tr = trajectory(agent_arrays, slot_idx)
        xs.extend(tr["x"].tolist())
        ys.extend(tr["y"].tolist())
    if not xs:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y, 1.0)
    dynamic_pad = max(pad, 0.25 * span)
    return [min_x - dynamic_pad, max_x + dynamic_pad, min_y - dynamic_pad, max_y + dynamic_pad]


def expand_bounds_to_aspect(bounds, aspect):
    min_x, max_x, min_y, max_y = bounds
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    current = width / height
    if current < aspect:
        target_width = height * aspect
        delta = (target_width - width) / 2
        min_x -= delta
        max_x += delta
    else:
        target_height = width / aspect
        delta = (target_height - height) / 2
        min_y -= delta
        max_y += delta
    return [min_x, max_x, min_y, max_y]


def fixed_target_bounds(agent_arrays, frame_idx, target_slot, half_width=50.0):
    tx = float(agent_arrays["x"][frame_idx, target_slot])
    ty = float(agent_arrays["y"][frame_idx, target_slot])
    return [tx - half_width, tx + half_width, ty - half_width, ty + half_width]


def draw_roads(ax, map_static, bounds):
    min_x, max_x, min_y, max_y = bounds
    for elem in map_static.get("road_elements", []):
        xs = elem.get("x", [])
        ys = elem.get("y", [])
        if len(xs) < 2 or len(ys) < 2:
            continue
        if max(xs) < min_x or min(xs) > max_x or max(ys) < min_y or min(ys) > max_y:
            continue
        style = road_style(elem.get("type", 0))
        if style is None:
            continue
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=style["width"],
            alpha=style["alpha"],
            linestyle=style["linestyle"],
            zorder=1,
        )


def draw_trajectory(ax, xs, ys, color, linewidth, alpha, zorder):
    if len(xs) < 2:
        return
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    alphas = np.linspace(alpha * 0.25, alpha, len(segments))
    colors = [(*matplotlib.colors.to_rgb(color), a) for a in alphas]
    collection = LineCollection(segments, colors=colors, linewidths=linewidth, capstyle="round", zorder=zorder)
    ax.add_collection(collection)


def vehicle_body_polygon(x, y, heading, length, width):
    length = max(float(length), 1.0)
    width = max(float(width), 0.5)
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )
    c = math.cos(heading)
    s = math.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def vehicle_heading_polygon(x, y, heading, length, width):
    length = max(float(length), 1.0)
    width = max(float(width), 0.5)
    tip_x = length * 0.60
    head_len = width * 0.25
    head_half_width = width * 0.20
    local = np.array(
        [
            [tip_x, 0.0],
            [tip_x - head_len, head_half_width],
            [tip_x - head_len, -head_half_width],
        ]
    )
    c = math.cos(heading)
    s = math.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def draw_vehicle(ax, agent_arrays, frame_idx, slot_idx, color, alpha=1.0, edge="#1f2937", zorder=5):
    if slot_idx is None or not agent_arrays["valid"][frame_idx, slot_idx]:
        return
    x = agent_arrays["x"][frame_idx, slot_idx]
    y = agent_arrays["y"][frame_idx, slot_idx]
    heading = agent_arrays["heading"][frame_idx, slot_idx]
    length = agent_arrays["length"][frame_idx, slot_idx]
    width = agent_arrays["width"][frame_idx, slot_idx]
    body = vehicle_body_polygon(
        x,
        y,
        heading,
        length,
        width,
    )
    patch = Polygon(body, closed=True, facecolor=color, edgecolor=edge, linewidth=0.7, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)

    nose = vehicle_heading_polygon(
        agent_arrays["x"][frame_idx, slot_idx],
        agent_arrays["y"][frame_idx, slot_idx],
        agent_arrays["heading"][frame_idx, slot_idx],
        agent_arrays["length"][frame_idx, slot_idx],
        agent_arrays["width"][frame_idx, slot_idx],
    )
    nose_patch = Polygon(
        nose,
        closed=True,
        facecolor=color,
        edgecolor="#111827",
        linewidth=0.3,
        alpha=min(1.0, alpha + 0.10),
        zorder=zorder + 1,
    )
    ax.add_patch(nose_patch)


def render_failure_png(
    replay_path,
    output_path,
    title=None,
    subtitle=None,
    last_n_frames=None,
    target_crop_half_width=None,
):
    replay_bundle = pufferlib.mining_viz.load_compact_replay(replay_path)
    payload = pufferlib.mining_viz._build_render_payload(replay_bundle)
    agent_arrays = replay_bundle["agent_arrays"]
    metadata = payload["metadata"]

    slots = agent_slots(agent_arrays)
    target_slot = next(
        (
            slot_idx
            for slot_idx in range(agent_arrays["valid"].shape[1])
            if np.any(agent_arrays["valid"][:, slot_idx] & agent_arrays["is_target"][:, slot_idx])
        ),
        None,
    )
    final_idx = agent_arrays["valid"].shape[0] - 1
    if target_crop_half_width is not None:
        bounds = fixed_target_bounds(agent_arrays, final_idx, target_slot, half_width=float(target_crop_half_width))
    else:
        focus_slots = list(slots.values())
        bounds = crop_bounds(agent_arrays, focus_slots)
        if bounds is None:
            min_x, min_y, max_x, max_y = payload["bounds"]
            bounds = [min_x, max_x, min_y, max_y]

    figsize = (5, 5)
    bounds = expand_bounds_to_aspect(bounds, figsize[0] / figsize[1])

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    draw_roads(ax, payload["map"], bounds)

    frame_start = 0
    if last_n_frames is not None:
        frame_start = max(0, final_idx - int(last_n_frames) + 1)
    visible_frame_count = final_idx - frame_start + 1
    sample_frames = sorted(set(np.linspace(frame_start, final_idx, min(6, visible_frame_count), dtype=int).tolist()))
    target_color = TARGET_COLOR
    other_color = ADVERSARIAL_COLOR
    other_vehicle_color = ADVERSARIAL_COLOR

    for agent_id, slot_idx in slots.items():
        tr = trajectory(agent_arrays, slot_idx)
        keep = tr["frames"] >= frame_start
        tr_x = tr["x"][keep]
        tr_y = tr["y"][keep]
        if slot_idx == target_slot:
            draw_trajectory(ax, tr_x, tr_y, target_color, 3.2, 0.95, 4)
        else:
            draw_trajectory(ax, tr_x, tr_y, other_color, 1.1, 0.35, 2)

    for frame_idx in sample_frames[:-1]:
        draw_vehicle(ax, agent_arrays, frame_idx, target_slot, target_color, alpha=0.18, edge=target_color, zorder=4)

    for agent_id, slot_idx in slots.items():
        if slot_idx == target_slot:
            continue
        draw_vehicle(ax, agent_arrays, final_idx, slot_idx, other_vehicle_color, alpha=0.72, edge="#006399", zorder=5)
    draw_vehicle(ax, agent_arrays, final_idx, target_slot, target_color, alpha=0.98, edge="#7f1d1d", zorder=7)

    ax.plot([], [], color=target_color, linewidth=3.2, label="Target agent")
    ax.plot([], [], color=other_color, linewidth=1.1, alpha=0.70, label="Adversarial agents")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output_path


def _numeric_series(df, column, default=0.0):
    if column not in df:
        return pd.Series(default, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _run_root(csv_path):
    csv_path = Path(csv_path)
    for parent in [csv_path.parent, *csv_path.parents]:
        if parent.name.startswith("puffer_drive_paper_vs_"):
            return parent
    return csv_path.parent


def _build_replay_index(csv_path):
    root = _run_root(csv_path)
    return {path.name: path for path in root.glob("**/*.replay.zlib")}


def _resolve_replay_path(raw_path, csv_path, episode_id=None, replay_index=None):
    if raw_path and not pd.isna(raw_path):
        path = Path(str(raw_path))
        if path.exists():
            return path

        if not path.is_absolute():
            cwd_candidate = (Path.cwd() / path).resolve()
            if cwd_candidate.exists():
                return cwd_candidate

    filename = f"episode_{int(episode_id):06d}.replay.zlib" if episode_id is not None else None
    if raw_path and not pd.isna(raw_path):
        filename = Path(str(raw_path)).name

    if not filename:
        return None

    if replay_index is not None and filename in replay_index:
        return replay_index[filename]

    csv_dir = Path(csv_path).parent
    search_roots = [csv_dir, *csv_dir.parents]
    for root in search_roots:
        if root.name == "failure_runs" or root == Path.cwd():
            break
        matches = list(root.glob(f"**/{filename}"))
        if matches:
            return matches[0]

    return None


def _run_label(csv_path):
    parts = Path(csv_path).parts
    run_name = next((part for part in parts if part.startswith("puffer_drive_paper_vs_")), Path(csv_path).parent.name)
    opponent = run_name.replace("puffer_drive_paper_vs_", "").split("_drive_paper")[0]
    if opponent == "pdm":
        return "PDM against adversarial traffic"
    return f"{opponent.replace('_', ' ').upper()} against adversarial traffic"


def _subtitle_for_row(row, shown_frames):
    responsibility = max(
        _safe_float(row.get("target_collision_responsibility")),
        _safe_float(row.get("target_hit_responsibility")),
    )
    at_fault = (
        _safe_float(row.get("did_target_have_at_fault_collision")) > 0
        or _safe_float(row.get("target_hit_at_fault_rate")) > 0
    )
    parts = [
        f"episode {int(row.get('episode_id', -1)):06d}",
        f"last {shown_frames} frames",
        f"target responsibility={responsibility:.3f}",
        f"target at fault={'yes' if at_fault else 'no'}",
    ]
    if row.get("map_name") is not None and not pd.isna(row.get("map_name")):
        parts.append(str(row.get("map_name")))
    return " | ".join(parts)


def _filename_for_row(row):
    episode_id = int(row["episode_id"])
    responsibility = max(
        _safe_float(row.get("target_collision_responsibility")),
        _safe_float(row.get("target_hit_responsibility")),
    )
    at_fault = (
        _safe_float(row.get("did_target_have_at_fault_collision")) > 0
        or _safe_float(row.get("target_hit_at_fault_rate")) > 0
    )
    fault_label = "atfault" if at_fault else "notatfault"
    responsibility_label = f"resp{responsibility:.3f}".replace(".", "p")
    return f"episode_{episode_id:06d}_{fault_label}_{responsibility_label}.png"


def _render_job(job):
    replay_path, output_path, title, subtitle, last_n_frames = job
    render_failure_png(
        replay_path,
        output_path,
        title=title,
        subtitle=subtitle,
        last_n_frames=last_n_frames,
        target_crop_half_width=50.0,
    )
    return str(output_path)


def batch_render(
    failure_runs_root,
    output_name="paper_figures",
    responsibility_threshold=0.2,
    last_n_frames=50,
    workers=0,
    limit=None,
):
    csv_paths = sorted(Path(failure_runs_root).glob("**/episodes*.csv"))
    jobs = []
    skipped_missing_replay = 0
    for csv_path in csv_paths:
        episodes_df = pd.read_csv(csv_path)
        replay_index = _build_replay_index(csv_path)
        has_replay = _numeric_series(episodes_df, "has_replay") > 0
        at_fault = (_numeric_series(episodes_df, "did_target_have_at_fault_collision") > 0) | (
            _numeric_series(episodes_df, "target_hit_at_fault_rate") > 0
        )
        responsibility = pd.concat(
            [
                _numeric_series(episodes_df, "target_collision_responsibility"),
                _numeric_series(episodes_df, "target_hit_responsibility"),
            ],
            axis=1,
        ).max(axis=1)
        selected = episodes_df[has_replay & (at_fault | (responsibility > responsibility_threshold))].copy()
        label = _run_label(csv_path)
        total_scenes = len(episodes_df)
        output_dir = csv_path.parent / output_name
        for row in selected.to_dict(orient="records"):
            episode_id = int(row["episode_id"])
            replay_path = _resolve_replay_path(
                row.get("replay_path"),
                csv_path,
                episode_id=episode_id,
                replay_index=replay_index,
            )
            if replay_path is None or not replay_path.exists():
                skipped_missing_replay += 1
                continue
            output_path = output_dir / _filename_for_row(row)
            title = f"{label}, scene {episode_id + 1} of {total_scenes}"
            subtitle = _subtitle_for_row(row, last_n_frames)
            jobs.append((replay_path, output_path, title, subtitle, last_n_frames))

    if limit is not None:
        jobs = jobs[: int(limit)]

    rendered = 0
    failed = []
    if workers and int(workers) > 1 and jobs:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {executor.submit(_render_job, job): job for job in jobs}
            with tqdm(total=len(futures), desc="Rendering paper PNGs") as pbar:
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        future.result()
                        rendered += 1
                    except Exception as exc:
                        failed.append((str(job[0]), str(exc)))
                    pbar.update(1)
    else:
        for job in tqdm(jobs, desc="Rendering paper PNGs"):
            try:
                _render_job(job)
                rendered += 1
            except Exception as exc:
                failed.append((str(job[0]), str(exc)))

    return {
        "csv_count": len(csv_paths),
        "job_count": len(jobs),
        "rendered": rendered,
        "skipped_missing_replay": skipped_missing_replay,
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_path", nargs="?")
    parser.add_argument("output_path", nargs="?")
    parser.add_argument("--title", default=None)
    parser.add_argument("--subtitle", default=None)
    parser.add_argument("--last-n-frames", type=int, default=None)
    parser.add_argument("--target-crop-half-width", type=float, default=None)
    parser.add_argument("--batch-failure-runs", default=None)
    parser.add_argument("--batch-output-name", default="paper_figures")
    parser.add_argument("--responsibility-threshold", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers for batch rendering")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of batch jobs for smoke tests")
    args = parser.parse_args()
    if args.batch_failure_runs:
        summary = batch_render(
            args.batch_failure_runs,
            output_name=args.batch_output_name,
            responsibility_threshold=args.responsibility_threshold,
            last_n_frames=args.last_n_frames or 50,
            workers=args.workers,
            limit=args.limit,
        )
        print(summary)
        return

    if not args.replay_path or not args.output_path:
        parser.error("replay_path and output_path are required unless --batch-failure-runs is used")
    render_failure_png(
        args.replay_path,
        args.output_path,
        title=args.title,
        subtitle=args.subtitle,
        last_n_frames=args.last_n_frames,
        target_crop_half_width=args.target_crop_half_width,
    )


if __name__ == "__main__":
    main()
