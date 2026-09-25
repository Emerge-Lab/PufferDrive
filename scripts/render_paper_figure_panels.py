#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

EVIDENCE_SCHEMA = "paper_figure_evidence"
EVIDENCE_VERSION = 1
PANEL_SIZE_INCHES = 5.0
OUTPUT_DPI = 600
DEFAULT_CROP_HALF_WIDTH_METERS = 50.0
CROP_MARGIN_METERS = 3.0
GHOST_FRAME_COUNT = 6

TARGET_COLOR = "#ff1f5b"
ADVERSARIAL_COLOR = "#009ade"
LANE_MARKING_COLOR = "#ffc61e"
TARGET_EDGE_COLOR = "#7f1d1d"
ADVERSARIAL_EDGE_COLOR = "#006399"
OVERLAP_COLOR = "#111827"
BRAKING_COLOR = "#ff8c1a"
BRAKING_EDGE_COLOR = "#7c3d06"
BRAKE_MARKER_COLOR = BRAKING_COLOR
BRAKE_MARKER_SIZE_POINTS = 6
BRAKE_MARKER_EDGE_WIDTH_POINTS = 0.7
FUTURE_BRAKE_MARKER_ALPHA = 0.6
FUTURE_ORIGINAL_PATH_ALPHA = 0.45
FUTURE_BRAKING_PATH_ALPHA = 0.85
FUTURE_PATH_WIDTH_POINTS = 2.2
FUTURE_PATH_DASH = (0, (4, 3))
TIME_LABEL_FONT_SIZE_POINTS = 11
TIME_LABEL_MARGIN_AXES = 0.03
ORIGINAL_PATH_ALPHA = 0.55

ROAD_STYLES = {
    "lane": {"color": "#d7dde3", "width": 0.9, "alpha": 0.72, "linestyle": "solid"},
    "yellow_line": {"color": LANE_MARKING_COLOR, "width": 1.15, "alpha": 0.90, "linestyle": (0, (4, 4))},
    "road_line": {"color": "#8f98a3", "width": 0.75, "alpha": 0.55, "linestyle": (0, (4, 4))},
    "edge": {"color": "#111111", "width": 0.9, "alpha": 0.85, "linestyle": "solid"},
}

PANEL_NAMES = ("panel_a_collision", "panel_b_braking_counterfactual", "panel_c_early_warning")


def load_evidence(evidence_path):
    evidence = json.loads(Path(evidence_path).read_text())
    if evidence.get("schema") != EVIDENCE_SCHEMA or evidence.get("version") != EVIDENCE_VERSION:
        raise ValueError(f"{evidence_path} is not a {EVIDENCE_SCHEMA} v{EVIDENCE_VERSION} file")
    frame_count = len(evidence["frames"])
    first_frame = evidence["first_frame"]
    for key in ("collision_frame", "detection_frame", "braking_start_frame"):
        if not first_frame <= evidence[key] < first_frame + frame_count:
            raise ValueError(f"{key}={evidence[key]} is outside the exported frame range")
    braking_state_count = evidence["collision_frame"] - evidence["braking_start_frame"] + 1
    if len(evidence["braking_target_states"]) != braking_state_count:
        raise ValueError("braking_target_states does not span braking start to collision")
    for road in evidence["roads"]:
        if road["style"] not in ROAD_STYLES:
            raise ValueError(f"Unknown road style {road['style']!r}")
    return evidence


def agents_at(evidence, frame_idx):
    return evidence["frames"][frame_idx - evidence["first_frame"]]


def agent_state(evidence, frame_idx, agent_index):
    return next((agent for agent in agents_at(evidence, frame_idx) if agent["index"] == agent_index), None)


def history_start_frame(evidence):
    return max(evidence["first_frame"], evidence["collision_frame"] - evidence["history_frame_count"])


def window_frames(evidence, end_frame):
    return list(range(history_start_frame(evidence), end_frame + 1))


def ghost_frames(evidence, end_frame):
    start_frame = history_start_frame(evidence)
    frame_count = evidence["collision_frame"] - start_frame + 1
    shared_frames = np.linspace(
        start_frame, evidence["collision_frame"], min(GHOST_FRAME_COUNT, frame_count), dtype=int
    )
    return sorted(frame_idx for frame_idx in set(shared_frames.tolist()) if frame_idx < end_frame)


def collision_trajectory_bounds(evidence, half_width):
    target_index = evidence["target_index"]
    hitter_index = evidence["hitter_index"]
    xs = []
    ys = []
    frames = set(window_frames(evidence, evidence["collision_frame"]))
    frames |= set(window_frames(evidence, evidence["detection_frame"]))
    for frame_idx in sorted(frames):
        for agent_index in (target_index, hitter_index):
            agent = agent_state(evidence, frame_idx, agent_index)
            if agent is not None:
                xs.append(agent["x"])
                ys.append(agent["y"])
    target = agent_state(evidence, evidence["collision_frame"], target_index)
    hitter = agent_state(evidence, evidence["collision_frame"], hitter_index)
    collision_x = 0.5 * (target["x"] + hitter["x"])
    collision_y = 0.5 * (target["y"] + hitter["y"])
    center_x = _trajectory_axis_center(xs, collision_x, half_width, CROP_MARGIN_METERS)
    center_y = _trajectory_axis_center(ys, collision_y, half_width, CROP_MARGIN_METERS)
    return [center_x - half_width, center_x + half_width, center_y - half_width, center_y + half_width]


def _trajectory_axis_center(values, collision_value, half_width, margin):
    low = float(np.min(values)) - margin
    high = float(np.max(values)) + margin
    if high - low <= 2 * half_width:
        return float(np.clip(collision_value, high - half_width, low + half_width))
    return float(np.clip(0.5 * (low + high), collision_value - half_width, collision_value + half_width))


def draw_roads(ax, roads, bounds):
    min_x, max_x, min_y, max_y = bounds
    for road in roads:
        xs = road["x"]
        ys = road["y"]
        if len(xs) < 2:
            continue
        if max(xs) < min_x or min(xs) > max_x or max(ys) < min_y or min(ys) > max_y:
            continue
        style = ROAD_STYLES[road["style"]]
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


def vehicle_body_polygon(x, y, heading, length, width, width_expansion=0.0):
    length = max(float(length), 1.0)
    width = max(float(width), 0.5) + width_expansion
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


def draw_vehicle(ax, agent, color, alpha=1.0, edge="#1f2937", zorder=5):
    if agent is None:
        return
    body = vehicle_body_polygon(agent["x"], agent["y"], agent["heading"], agent["length"], agent["width"])
    ax.add_patch(Polygon(body, closed=True, facecolor=color, edgecolor=edge, linewidth=0.7, alpha=alpha, zorder=zorder))
    nose = vehicle_heading_polygon(agent["x"], agent["y"], agent["heading"], agent["length"], agent["width"])
    ax.add_patch(
        Polygon(
            nose,
            closed=True,
            facecolor=color,
            edgecolor="#111827",
            linewidth=0.3,
            alpha=min(1.0, alpha + 0.10),
            zorder=zorder + 1,
        )
    )


def draw_vehicle_outline(ax, agent, color, width_expansion=0.0, zorder=6):
    body = vehicle_body_polygon(
        agent["x"], agent["y"], agent["heading"], agent["length"], agent["width"], width_expansion
    )
    ax.add_patch(
        Polygon(
            body,
            closed=True,
            facecolor="none",
            edgecolor=color,
            linewidth=1.1,
            linestyle=(0, (3, 2)),
            alpha=0.95,
            zorder=zorder,
        )
    )


def draw_overlap(ax, polygon):
    if len(polygon) < 3:
        return
    points = np.array([[point["x"], point["y"]] for point in polygon])
    ax.add_patch(Polygon(points, closed=True, facecolor=OVERLAP_COLOR, edgecolor="none", alpha=0.75, zorder=9))


def draw_scene(ax, evidence, end_frame, target_track=None, braking_start_frame=None):
    target_index = evidence["target_index"]
    frames = window_frames(evidence, end_frame)
    if target_track is None:
        target_track = {frame_idx: agent_state(evidence, frame_idx, target_index) for frame_idx in frames}

    other_indices = [evidence["hitter_index"]]
    for agent_index in other_indices:
        states = [agent_state(evidence, frame_idx, agent_index) for frame_idx in frames]
        states = [state for state in states if state is not None]
        draw_trajectory(ax, [s["x"] for s in states], [s["y"] for s in states], ADVERSARIAL_COLOR, 1.1, 0.35, 2)
    if braking_start_frame is None:
        braking_start_frame = end_frame + 1
    pre_brake_states = [
        target_track[frame_idx]
        for frame_idx in frames
        if frame_idx <= braking_start_frame and target_track[frame_idx] is not None
    ]
    draw_trajectory(
        ax, [s["x"] for s in pre_brake_states], [s["y"] for s in pre_brake_states], TARGET_COLOR, 3.2, 0.95, 4
    )
    braking_states = [
        target_track[frame_idx]
        for frame_idx in frames
        if frame_idx >= braking_start_frame and target_track[frame_idx] is not None
    ]
    ax.plot(
        [s["x"] for s in braking_states],
        [s["y"] for s in braking_states],
        color=BRAKING_COLOR,
        linewidth=3.2,
        alpha=0.95,
        solid_capstyle="round",
        zorder=4,
    )

    for frame_idx in ghost_frames(evidence, end_frame):
        ghost_color = BRAKING_COLOR if frame_idx >= braking_start_frame else TARGET_COLOR
        draw_vehicle(ax, target_track[frame_idx], ghost_color, alpha=0.18, edge=ghost_color, zorder=4)
        for agent_index in other_indices:
            draw_vehicle(
                ax,
                agent_state(evidence, frame_idx, agent_index),
                ADVERSARIAL_COLOR,
                alpha=0.18,
                edge=ADVERSARIAL_COLOR,
                zorder=3,
            )

    for agent_index in other_indices:
        draw_vehicle(
            ax,
            agent_state(evidence, end_frame, agent_index),
            ADVERSARIAL_COLOR,
            alpha=0.72,
            edge=ADVERSARIAL_EDGE_COLOR,
            zorder=5,
        )
    if end_frame >= braking_start_frame:
        draw_vehicle(ax, target_track[end_frame], BRAKING_COLOR, alpha=0.98, edge=BRAKING_EDGE_COLOR, zorder=7)
    else:
        draw_vehicle(ax, target_track[end_frame], TARGET_COLOR, alpha=0.98, edge=TARGET_EDGE_COLOR, zorder=7)


def finish_axes(ax, bounds):
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_brake_marker(ax, evidence, alpha):
    brake_start = evidence["braking_target_states"][0]
    ax.plot(
        brake_start["x"],
        brake_start["y"],
        marker="X",
        markersize=BRAKE_MARKER_SIZE_POINTS,
        markerfacecolor=BRAKE_MARKER_COLOR,
        markeredgecolor=BRAKING_EDGE_COLOR,
        markeredgewidth=BRAKE_MARKER_EDGE_WIDTH_POINTS,
        linestyle="none",
        alpha=alpha,
        zorder=10,
    )


def render_panel_a(ax, evidence, bounds):
    draw_roads(ax, evidence["roads"], bounds)
    draw_scene(ax, evidence, evidence["collision_frame"])
    finish_axes(ax, bounds)


def render_panel_b(ax, evidence, bounds):
    target_index = evidence["target_index"]
    collision_frame = evidence["collision_frame"]
    braking_start_frame = evidence["braking_start_frame"]
    observed_target_at_collision = agent_state(evidence, collision_frame, target_index)
    target_track = {}
    for frame_idx in window_frames(evidence, collision_frame):
        observed = agent_state(evidence, frame_idx, target_index)
        if frame_idx < braking_start_frame or observed is None:
            target_track[frame_idx] = observed
            continue
        braked_pose = evidence["braking_target_states"][frame_idx - braking_start_frame]
        target_track[frame_idx] = {**observed, **braked_pose}
    draw_roads(ax, evidence["roads"], bounds)
    draw_scene(ax, evidence, collision_frame, target_track, braking_start_frame)
    draw_vehicle_outline(ax, observed_target_at_collision, TARGET_COLOR)
    original_path = [
        agent_state(evidence, frame_idx, target_index) for frame_idx in range(braking_start_frame, collision_frame + 1)
    ]
    original_path = [state for state in original_path if state is not None]
    ax.plot(
        [state["x"] for state in original_path],
        [state["y"] for state in original_path],
        color=TARGET_COLOR,
        linewidth=1.6,
        linestyle=(0, (3, 2)),
        alpha=ORIGINAL_PATH_ALPHA,
        zorder=8,
    )
    draw_brake_marker(ax, evidence, alpha=1.0)
    finish_axes(ax, bounds)


def render_panel_c(ax, evidence, bounds):
    prediction = evidence["prediction"]
    draw_roads(ax, evidence["roads"], bounds)
    draw_scene(ax, evidence, evidence["detection_frame"])
    target_index = evidence["target_index"]
    future_original = [
        agent_state(evidence, frame_idx, target_index)
        for frame_idx in range(evidence["detection_frame"], evidence["collision_frame"] + 1)
    ]
    future_original = [state for state in future_original if state is not None]
    ax.plot(
        [state["x"] for state in future_original],
        [state["y"] for state in future_original],
        color=TARGET_COLOR,
        linewidth=FUTURE_PATH_WIDTH_POINTS,
        linestyle=FUTURE_PATH_DASH,
        alpha=FUTURE_ORIGINAL_PATH_ALPHA,
        zorder=3,
    )
    braking_path = evidence["braking_target_states"]
    ax.plot(
        [state["x"] for state in braking_path],
        [state["y"] for state in braking_path],
        color=BRAKING_COLOR,
        linewidth=FUTURE_PATH_WIDTH_POINTS,
        linestyle=FUTURE_PATH_DASH,
        alpha=FUTURE_BRAKING_PATH_ALPHA,
        zorder=3.5,
    )
    for path_key, color in (("target_path", TARGET_COLOR), ("hitter_path", ADVERSARIAL_COLOR)):
        path = prediction[path_key]
        ax.plot(
            [point["x"] for point in path],
            [point["y"] for point in path],
            color=color,
            linewidth=1.3,
            linestyle=(0, (3, 2)),
            alpha=0.95,
            zorder=6,
        )
    draw_vehicle_outline(
        ax, prediction["projected_target"], TARGET_COLOR, width_expansion=2 * prediction["lateral_buffer_meters"]
    )
    draw_vehicle_outline(ax, prediction["projected_hitter"], ADVERSARIAL_COLOR)
    draw_overlap(ax, prediction["overlap_polygon"])
    draw_brake_marker(ax, evidence, alpha=FUTURE_BRAKE_MARKER_ALPHA)
    finish_axes(ax, bounds)


PANEL_RENDERERS = (render_panel_a, render_panel_b, render_panel_c)


def new_figure(panel_count):
    fig, axes = plt.subplots(1, panel_count, figsize=(PANEL_SIZE_INCHES * panel_count, PANEL_SIZE_INCHES))
    fig.patch.set_facecolor("#ffffff")
    for ax in np.atleast_1d(axes):
        ax.set_facecolor("#ffffff")
    return fig, np.atleast_1d(axes)


def save_figure(fig, output_stem, fill_figure=False):
    if fill_figure:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    else:
        fig.tight_layout(pad=0.2)
    output_path = output_stem.with_suffix(".png")
    fig.savefig(output_path, bbox_inches=None if fill_figure else "tight", dpi=OUTPUT_DPI)
    plt.close(fig)
    return [output_path]


def panel_time_labels(evidence):
    dt = evidence["dt_seconds"]
    brake_seconds = (evidence["braking_start_frame"] - evidence["collision_frame"]) * dt
    warning_seconds = (evidence["detection_frame"] - evidence["collision_frame"]) * dt
    return (
        "t = 0.0 s",
        f"t = 0.0 s (brake at t = {brake_seconds:.1f} s)",
        f"t = {warning_seconds:.1f} s",
    )


def draw_time_label(ax, label):
    ax.text(
        TIME_LABEL_MARGIN_AXES,
        1 - TIME_LABEL_MARGIN_AXES,
        label.replace("-", "\u2212"),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TIME_LABEL_FONT_SIZE_POINTS,
        color="#111827",
        zorder=20,
    )


def render_evidence(evidence_path, output_dir, crop_half_width, time_labels=False):
    evidence = load_evidence(evidence_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bounds = collision_trajectory_bounds(evidence, crop_half_width)
    labels = panel_time_labels(evidence)
    written = []
    for panel_name, renderer in zip(PANEL_NAMES, PANEL_RENDERERS):
        fig, axes = new_figure(1)
        renderer(axes[0], evidence, bounds)
        if time_labels:
            draw_time_label(axes[0], labels[PANEL_RENDERERS.index(renderer)])
        written += save_figure(fig, output_dir / panel_name, fill_figure=True)
    fig, axes = new_figure(len(PANEL_RENDERERS))
    for ax, renderer in zip(axes, PANEL_RENDERERS):
        renderer(ax, evidence, bounds)
        if time_labels:
            draw_time_label(ax, labels[PANEL_RENDERERS.index(renderer)])
    written += save_figure(fig, output_dir / "figure_abc")
    return written


def main():
    parser = argparse.ArgumentParser(description="Render A/B/C paper panels from a paper_figure_evidence JSON")
    parser.add_argument("evidence_path")
    parser.add_argument("output_dir", nargs="?", default=None)
    parser.add_argument("--crop-half-width", type=float, default=DEFAULT_CROP_HALF_WIDTH_METERS)
    parser.add_argument("--time-labels", action="store_true", help="Stamp each panel with the time it shows")
    args = parser.parse_args()
    if not (args.crop_half_width > 0 and math.isfinite(args.crop_half_width)):
        parser.error("--crop-half-width must be a positive finite number of meters")
    output_dir = args.output_dir or Path(args.evidence_path).parent
    for output_path in render_evidence(args.evidence_path, output_dir, args.crop_half_width, args.time_labels):
        print(output_path)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
