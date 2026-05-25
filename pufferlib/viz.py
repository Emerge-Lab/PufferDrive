"""Bird's Eye View visualization for PufferDrive scenarios using Matplotlib."""

import dataclasses
import weakref
from typing import Optional, Tuple

import math
import re
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
from matplotlib.patches import Circle
import os
import json
import zlib
import base64
import struct

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import compute_effective_road_obs_count


COLORS = {
    "pedestrian": "#2E8B57",
    "cyclist": "#FF8C00",
    "road_line": "#808080",
    "road_edge": "#000000",
    "lane": "#D3D3D3",
    "crosswalk": "#E6C200",
    "speed_bump": "#C90000",
    "stop_sign": "#FF0000",
    "inactive_agent": "#808080",
    "background": "#F5F5F5",
}

TRAFFIC_LIGHT_COLORS = {
    binding.TRAFFIC_CONTROL_STATE_UNKNOWN: "#808080",
    binding.TRAFFIC_CONTROL_STATE_RED: "#FF0000",
    binding.TRAFFIC_CONTROL_STATE_YELLOW: "#FFFF00",
    binding.TRAFFIC_CONTROL_STATE_GREEN: "#00FF00",
    binding.TRAFFIC_CONTROL_STATE_OFF: "#808080",
}

VEHICLE_COLORS = [
    "#681D00",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#9467BD",
    "#8C564B",
    "#D47CBA",
    "#BCBD22",
    "#17BECF",
    "#AEC7E8",
    "#FFBB78",
    "#98DF8A",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#DBDB8D",
    "#9EDAE5",
]

_figure_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_map_cache = {}

MULTI_LANE_FULL_SCORE_TIME = binding.MULTI_LANE_FULL_SCORE_TIME
MULTI_LANE_HALF_SCORE_TIME = binding.MULTI_LANE_HALF_SCORE_TIME
JERK_LONG_VALUES = np.asarray([-15.0, -4.0, 0.0, 4.0], dtype=np.float32)
JERK_LAT_VALUES = np.asarray([-4.0, 0.0, 4.0], dtype=np.float32)
ACCELERATION_VALUES = np.asarray([-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0], dtype=np.float32)
STEERING_VALUES = np.asarray([-0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667], dtype=np.float32)

METRIC_LABELS = [
    "collision",
    "offroad",
    "red_light",
    "stop_sign",
    "reached_goal",
    "lane_dist",
    "lane_angle",
    "comfort_violation",
    "velocity_progress",
    "speed_limit",
    "ADE",
    "progression",
    "at_fault_collision",
    "ttc",
    "ttc_tfl",
    "progress_ratio",
    "multi_lane_time",
    "multi_lane_score",
]


@dataclasses.dataclass
class VizConfig:
    """Visualization config using radius and center for view bounds."""

    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    figsize: Tuple[float, float] = (20.0, 20.0)
    dpi: int = 100
    show_agent_id: bool = True
    show_routes: bool = False
    show_goal: bool = True
    show_sdc_paths: bool = False
    show_trajectories: bool = False
    goal_radius: float = 2.0
    follow_ego: bool = False
    debug_metrics: bool = False
    reuse_figure: bool = True

    def get_bounds(self, scenario) -> Tuple[float, float, float, float]:
        map_corners = scenario.get("map_corners")

        if self.follow_ego:
            ego_agent = scenario.get("agents")[-1]
            cx, cy = ego_agent["sim_x"], ego_agent["sim_y"]
        elif self.center is not None:
            cx, cy = self.center
        elif map_corners and len(map_corners) >= 4:
            cx, cy = (map_corners[0] + map_corners[2]) / 2, (map_corners[1] + map_corners[3]) / 2
        else:
            cx, cy = 0.0, 0.0

        if self.radius is not None:
            r = self.radius
        elif map_corners and len(map_corners) >= 4:
            r = max(map_corners[2] - map_corners[0], map_corners[3] - map_corners[1]) / 2 * 1.02
        else:
            r = 100.0
        return (cx - r, cx + r, cy - r, cy + r)


def get_agent_color(agent_id, is_active=True):
    return COLORS["inactive_agent"] if not is_active else VEHICLE_COLORS[agent_id % len(VEHICLE_COLORS)]


def _traffic_control_kind(control_type):
    control_type = int(control_type)
    if control_type == binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT:
        return "light"
    if control_type == binding.TRAFFIC_CONTROL_TYPE_STOP_SIGN:
        return "stop"
    if control_type == binding.TRAFFIC_CONTROL_TYPE_YIELD_SIGN:
        return "yield"
    return None


def _traffic_light_color(state):
    return TRAFFIC_LIGHT_COLORS.get(int(state), COLORS["inactive_agent"])


def _scale_ratio(numerator, denominator, default=1.0):
    return default if denominator == 0 else float(numerator) / float(denominator)


def _obs_scales(
    env_cfg=None,
    obs_norm_goal_offset_m=100.0,
    obs_norm_xy_offset_m=100.0,
    obs_norm_veh_width_m=10.0,
    obs_norm_veh_length_m=15.0,
    obs_norm_road_seg_length_m=5.0,
    obs_norm_road_seg_width_m=5.0,
):
    env_cfg = env_cfg or {}
    obs_norm_goal_offset_m = float(env_cfg.get("obs_norm_goal_offset_m", obs_norm_goal_offset_m))
    obs_norm_xy_offset_m = float(env_cfg.get("obs_norm_xy_offset_m", obs_norm_xy_offset_m))
    obs_norm_veh_width_m = float(env_cfg.get("obs_norm_veh_width_m", obs_norm_veh_width_m))
    obs_norm_veh_length_m = float(env_cfg.get("obs_norm_veh_length_m", obs_norm_veh_length_m))
    obs_norm_road_seg_length_m = float(env_cfg.get("obs_norm_road_seg_length_m", obs_norm_road_seg_length_m))
    obs_norm_road_seg_width_m = float(env_cfg.get("obs_norm_road_seg_width_m", obs_norm_road_seg_width_m))
    return {
        "obs_norm_goal_offset_m": obs_norm_goal_offset_m,
        "obs_norm_xy_offset_m": obs_norm_xy_offset_m,
        "veh_width_to_position": _scale_ratio(obs_norm_veh_width_m, obs_norm_xy_offset_m),
        "veh_len_to_position": _scale_ratio(obs_norm_veh_length_m, obs_norm_xy_offset_m),
        "goal_to_position": _scale_ratio(obs_norm_goal_offset_m, obs_norm_xy_offset_m),
        "road_length_to_position": _scale_ratio(obs_norm_road_seg_length_m, obs_norm_xy_offset_m),
        "road_width_to_position": _scale_ratio(obs_norm_road_seg_width_m, obs_norm_xy_offset_m),
    }


def _init_fig_ax(config: VizConfig, reuse_key: str = None, with_metrics: bool = False):
    cache_key = f"{reuse_key}_{'metrics' if with_metrics else 'single'}" if reuse_key else None

    if config.reuse_figure and cache_key and cache_key in _figure_cache:
        fig = _figure_cache[cache_key]
        if fig and plt.fignum_exists(fig.number):
            for ax in fig.axes:
                ax.clear()
                ax.set_facecolor(COLORS["background"])
            if with_metrics:
                return fig, fig.axes[0], fig.axes[1]
            return fig, fig.axes[0]

    if with_metrics:
        fig, (ax_main, ax_metrics) = plt.subplots(
            1, 2, figsize=(config.figsize[0] * 1.5, config.figsize[1]), gridspec_kw={"width_ratios": [2, 1]}
        )
    else:
        fig, ax_main = plt.subplots()
        fig.set_size_inches(config.figsize)
        ax_metrics = None

    fig.set_dpi(config.dpi)
    fig.set_facecolor(COLORS["background"])
    ax_main.set_facecolor(COLORS["background"])
    if ax_metrics:
        ax_metrics.set_facecolor(COLORS["background"])

    if config.reuse_figure and cache_key:
        _figure_cache[cache_key] = fig

    if with_metrics:
        return fig, ax_main, ax_metrics
    return fig, ax_main


def _build_road_cache(road_elements):
    lanes, lines, edges = [], [], []
    lane_dict = {}
    for elem in road_elements or []:
        if not isinstance(elem, dict):
            continue
        x, y, t = elem.get("x"), elem.get("y"), elem.get("type", 0)
        if not x or not y:
            continue
        pts = np.column_stack((np.asarray(x), np.asarray(y)))
        if 1 <= t <= 3:
            lanes.append(pts)
            lid = elem.get("id")
            if lid is not None:
                lane_dict[lid] = pts
        elif 11 <= t <= 18:
            lines.append(pts)
        elif 21 <= t <= 23:
            edges.append(pts)
    return {
        "lanes": lanes,
        "lines": lines,
        "edges": edges,
        "lane_dict": lane_dict,
        "collections": None,
    }


def _render_roads(ax, road_cache):
    if not road_cache:
        return
    collections = road_cache.get("collections")
    if collections is None:
        collections = []
        lanes = road_cache.get("lanes") or []
        lines = road_cache.get("lines") or []
        edges = road_cache.get("edges") or []
        if lanes:
            collections.append(LineCollection(lanes, colors=COLORS["lane"], linewidths=0.8, alpha=0.7, zorder=1))
        if lines:
            collections.append(
                LineCollection(
                    lines,
                    colors=COLORS["road_line"],
                    linewidths=0.8,
                    alpha=0.6,
                    linestyles=(0, (5, 5)),
                    zorder=2,
                )
            )
        if edges:
            collections.append(LineCollection(edges, colors=COLORS["road_edge"], linewidths=0.8, alpha=0.8, zorder=2))
        road_cache["collections"] = collections
    for collection in collections:
        ax.add_collection(collection)


def _build_traffic_cache(traffic_elements):
    traffic_lights = []  # (stop_line, states)
    stop_signs = []  # stop_line endpoints
    yield_signs = []  # stop_line endpoints
    for elem in traffic_elements or []:
        if not isinstance(elem, dict):
            continue
        t_type = elem.get("type", binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT)
        sl = elem.get("stop_line")
        if sl is None or len(sl) < 4:
            continue
        kind = _traffic_control_kind(t_type)
        if kind == "light":
            traffic_lights.append({"stop_line": sl, "states": elem.get("states", [])})
        elif kind == "stop":
            stop_signs.append(sl)
        elif kind == "yield":
            yield_signs.append(sl)
    return {
        "traffic_lights": traffic_lights,
        "stop_signs": stop_signs,
        "yield_signs": yield_signs,
    }


def _render_traffic(ax, traffic_cache, timestep):
    if not traffic_cache:
        return
    # Traffic lights — colored by state
    for light in traffic_cache.get("traffic_lights", []):
        sl = light["stop_line"]
        states = light["states"]
        state = int(states[timestep]) if states and len(states) > timestep else 0
        color = _traffic_light_color(state)
        ax.plot([sl[0], sl[3]], [sl[1], sl[4]], color=color, linewidth=3, solid_capstyle="butt", alpha=0.9, zorder=15)

    # Stop signs — red/black striped
    for sl in traffic_cache.get("stop_signs", []):
        ax.plot([sl[0], sl[3]], [sl[1], sl[4]], color="black", linewidth=4, solid_capstyle="butt", alpha=0.9, zorder=15)
        ax.plot(
            [sl[0], sl[3]],
            [sl[1], sl[4]],
            color="#FF0000",
            linewidth=2.5,
            solid_capstyle="butt",
            alpha=0.9,
            zorder=15,
            linestyle=(0, (3, 2)),
        )

    # Yield signs — yellow/black striped
    for sl in traffic_cache.get("yield_signs", []):
        ax.plot([sl[0], sl[3]], [sl[1], sl[4]], color="black", linewidth=4, solid_capstyle="butt", alpha=0.9, zorder=15)
        ax.plot(
            [sl[0], sl[3]],
            [sl[1], sl[4]],
            color="#FFD700",
            linewidth=2.5,
            solid_capstyle="butt",
            alpha=0.9,
            zorder=15,
            linestyle=(0, (3, 2)),
        )


def _render_routes(ax, agents, lane_dict, active_indices):
    if not agents or not lane_dict:
        return

    active_set = set(active_indices or [])
    segments_by_color = {}
    for idx, agent in enumerate(agents):
        if not isinstance(agent, dict) or idx not in active_set:
            continue
        route = agent.get("route", [])
        if not route:
            continue
        color = get_agent_color(agent.get("id", idx))
        segs = segments_by_color.setdefault(color, [])
        for lid in route:
            if lid in lane_dict:
                segs.append(lane_dict[lid])

    for color, segs in segments_by_color.items():
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=2.0, alpha=0.6, linestyles="--", zorder=5))


def _render_agents(ax, agents, active_indices, static_indices, config, px_per_meter):
    if not agents:
        return
    active_set, static_set = set(active_indices or []), set(static_indices or [])
    vehicles = []
    vehicle_lengths = []
    vehicle_widths = []
    vehicle_headings = []
    vehicle_colors = []
    vehicle_edges = []
    text_items = []
    goal_points = []
    goal_colors = []
    ped_patches = []
    cyclist_patches = []
    font_size = max(12, int(px_per_meter / 5))

    for idx, agent in enumerate(agents):
        if idx not in active_set and idx not in static_set:
            continue
        if not agent.get("sim_valid"):
            continue
        x, y = agent.get("sim_x"), agent.get("sim_y")
        if x is None or y is None:
            continue

        agent_type = agent.get("type", 1)
        agent_id = agent.get("id", idx)
        is_active = idx in active_set
        color = get_agent_color(agent_id, is_active)
        edge = "black" if is_active else COLORS["inactive_agent"]

        if agent_type == 1:
            if agent.get("stopped"):
                color = "red"
            length = agent.get("sim_length", 4)
            width = agent.get("sim_width", 2)
            heading = agent.get("sim_heading", 0)

            vehicles.append((x, y))
            vehicle_lengths.append(length)
            vehicle_widths.append(width)
            vehicle_headings.append(heading)
            vehicle_colors.append(color)
            vehicle_edges.append(edge)

            if config.show_agent_id:
                text_items.append((x, y + width, str(agent_id)))

            if config.show_goal and is_active:
                gx, gy = agent.get("goal_position_x"), agent.get("goal_position_y")
                if gx is not None and gy is not None:
                    goal_points.append((gx, gy))
                    goal_colors.append(color)
        elif agent_type == 2:
            ped_patches.append(
                Circle(
                    (x, y),
                    radius=0.5,
                    facecolor=COLORS["pedestrian"],
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.85,
                    zorder=10,
                )
            )
        elif agent_type == 3:
            cyclist_patches.append(
                Circle(
                    (x, y),
                    radius=0.8,
                    facecolor=COLORS["cyclist"],
                    edgecolor="black",
                    linewidth=1.5,
                    alpha=0.85,
                    zorder=10,
                )
            )

    if vehicles:
        centers = np.asarray(vehicles, dtype=float)
        lengths = np.asarray(vehicle_lengths, dtype=float)
        widths = np.asarray(vehicle_widths, dtype=float)
        headings = np.asarray(vehicle_headings, dtype=float)
        cos_h = np.cos(headings)
        sin_h = np.sin(headings)
        half_l = lengths / 2.0
        half_w = widths / 2.0
        base = np.stack(
            (
                np.stack((half_l, half_w), axis=1),
                np.stack((half_l, -half_w), axis=1),
                np.stack((-half_l, -half_w), axis=1),
                np.stack((-half_l, half_w), axis=1),
            ),
            axis=1,
        )
        rot_x = base[:, :, 0] * cos_h[:, None] - base[:, :, 1] * sin_h[:, None]
        rot_y = base[:, :, 0] * sin_h[:, None] + base[:, :, 1] * cos_h[:, None]
        polys = np.stack((rot_x, rot_y), axis=2) + centers[:, None, :]

        ax.add_collection(
            PolyCollection(
                polys,
                facecolors=vehicle_colors,
                edgecolors=vehicle_edges,
                linewidths=0.7,
                alpha=0.8,
                zorder=10,
            )
        )

        dx = lengths * 0.6 * cos_h
        dy = lengths * 0.6 * sin_h
        segments = np.stack((centers, centers + np.stack((dx, dy), axis=1)), axis=1)
        ax.add_collection(LineCollection(segments, colors=vehicle_colors, linewidths=0.7, zorder=11))

        head_len = widths * 0.25
        head_half_width = widths * 0.2
        tip = centers + np.stack((dx, dy), axis=1)
        dir_vec = np.stack((cos_h, sin_h), axis=1)
        perp_vec = np.stack((-sin_h, cos_h), axis=1)
        base_center = tip - dir_vec * head_len[:, None]
        left = base_center + perp_vec * head_half_width[:, None]
        right = base_center - perp_vec * head_half_width[:, None]
        arrows = np.stack((tip, left, right), axis=1)
        ax.add_collection(
            PolyCollection(
                arrows,
                facecolors=vehicle_colors,
                edgecolors="black",
                linewidths=0.3,
                zorder=12,
            )
        )

    if text_items:
        for x, y, text in text_items:
            ax.text(
                x,
                y,
                text,
                fontsize=font_size,
                color="black",
                ha="center",
                va="bottom",
                fontweight="bold",
                zorder=12,
            )

    if goal_points:
        gx, gy = zip(*goal_points)
        ax.scatter(gx, gy, s=20, c=goal_colors, marker="o", zorder=13)
        goal_patches = [Circle((x, y), radius=config.goal_radius) for x, y in goal_points]
        ax.add_collection(
            PatchCollection(
                goal_patches,
                facecolors="none",
                edgecolors=goal_colors,
                linewidths=1.0,
                linestyles="--",
                zorder=13,
            )
        )

    if ped_patches:
        ax.add_collection(PatchCollection(ped_patches, match_original=True))
    if cyclist_patches:
        ax.add_collection(PatchCollection(cyclist_patches, match_original=True))


def _render_paths(ax, scenario):
    """Render SDC planned paths."""
    for idx in range(scenario["active_agent_count"]):
        x = np.array([item["x"] for item in scenario["sdc_paths"][idx]["waypoints"]])
        y = np.array([item["y"] for item in scenario["sdc_paths"][idx]["waypoints"]])
        init_idx = scenario["agents"][scenario["active_agent_indices"][idx]]["closest_path_idx_wp"]
        end_idx = min(init_idx + 20, scenario["sdc_paths"][idx]["num_waypoints"] - 1)
        agent_id = scenario["agents"][scenario["active_agent_indices"][idx]]["id"]
        color = get_agent_color(agent_id, is_active=True)
        ax.scatter(x[init_idx:end_idx], y[init_idx:end_idx], color=color, s=20)


def _render_trajectories(ax, scenario):
    for idx in range(scenario["active_agent_count"]):
        wps = scenario["trajectory_waypoints_global"][idx]["waypoints"]
        x = np.array([item["x"] for item in wps])
        y = np.array([item["y"] for item in wps])
        heading = np.array([item["heading"] for item in wps])
        ax.scatter(x, y, color=np.array([0, 100, 0]) / 255.0, s=20)
        ax.quiver(
            x,
            y,
            np.cos(heading),
            np.sin(heading),
            color=np.array([0, 100, 0]) / 255.0,
            scale_units="xy",  # Use data coordinates for scaling
            scale=1.0,  # A scale of 1.0 means arrows of length (U,V) are plotted as such
            width=0.005,
        )


def _render_debug_metrics_table(ax, agents, active_agent_indices, px_per_meter=10.0):
    """Render a table of per-agent metrics for debugging."""
    font_size = max(10, int(px_per_meter / 5))

    if not agents or not active_agent_indices:
        ax.text(0.5, 0.5, "No active agents", ha="center", va="center", fontsize=font_size)
        ax.axis("off")
        return

    active_set = set(active_agent_indices)

    # Gather metrics for active agents
    metrics_data = []
    for idx, agent in enumerate(agents):
        if idx not in active_set:
            continue
        agent_id = agent["id"]
        vx, vy = agent.get("sim_vx", 0), agent.get("sim_vy", 0)
        speed = np.sqrt(vx**2 + vy**2)
        current_lane_id = agent.get("current_lane_idx", -1)
        metrics = agent.get("metrics_array", [0.0] * len(METRIC_LABELS))
        metrics_data.append(
            {
                "id": agent_id,
                "current_lane": current_lane_id,
                "speed": speed,
                "lane_dist": metrics[5],
                "lane_head": metrics[6],
                "offroad": metrics[1],
                "collision": metrics[0],
                "comfort": metrics[7],
                "red_light": metrics[2],
                "at_fault": metrics[12],
                "ttc": metrics[13],
                "ttc_tfl": metrics[14],
                "progress": metrics[15],
                "ml_time": metrics[16] if len(metrics) > 16 else 0.0,
                "color": get_agent_color(agent_id, is_active=True),
            }
        )

    if not metrics_data:
        ax.text(0.5, 0.5, "No active agents", ha="center", va="center", fontsize=font_size)
        ax.axis("off")
        return

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Remove margins
    ax.margins(0)

    # Table headers
    headers = [
        "ID",
        "Lane",
        "LDist",
        "LHead",
        "Spd",
        "Cmft",
        "Off",
        "Col",
        "Red",
        "AF",
        "TTC",
        "TTC_TFL",
        "Prog",
        "MLt",
    ]
    num_agents = len(metrics_data)
    y_start, y_end = 0.95, 0.05
    row_height = min(0.06, (y_start - y_end) / (num_agents + 2))
    x_positions = np.linspace(0.02, 0.96, len(headers))
    for i, header in enumerate(headers):
        ax.text(x_positions[i], y_start, header, fontsize=font_size + 2, fontweight="bold", va="top")

    for row_idx, data in enumerate(metrics_data):
        y_pos = y_start - (row_idx + 1) * row_height
        ax.text(
            x_positions[0], y_pos, str(data["id"]), fontsize=font_size, color=data["color"], fontweight="bold", va="top"
        )
        ax.text(x_positions[1], y_pos, f"{data['current_lane']:.0f}", fontsize=font_size, va="top")
        ax.text(x_positions[2], y_pos, f"{data['lane_dist']:.2f}", fontsize=font_size, va="top")
        ax.text(x_positions[3], y_pos, f"{data['lane_head']:.2f}", fontsize=font_size, va="top")
        ax.text(x_positions[4], y_pos, f"{data['speed']:.1f}", fontsize=font_size, va="top")
        ax.text(
            x_positions[5],
            y_pos,
            f"{data['comfort']:.1f}",
            fontsize=font_size,
            color="red" if data["comfort"] > 0 else "green",
            va="top",
        )
        ax.text(
            x_positions[6],
            y_pos,
            "+" if data["offroad"] else "-",
            fontsize=font_size,
            color="red" if data["offroad"] else "green",
            va="top",
        )
        ax.text(
            x_positions[7],
            y_pos,
            "+" if data["collision"] else "-",
            fontsize=font_size,
            color="red" if data["collision"] else "green",
            va="top",
        )
        ax.text(
            x_positions[8],
            y_pos,
            "+" if data["red_light"] else "-",
            fontsize=font_size,
            color="red" if data["red_light"] else "green",
            va="top",
        )
        ax.text(
            x_positions[9],
            y_pos,
            "+" if data["at_fault"] else "-",
            fontsize=font_size,
            color="red" if data["at_fault"] else "green",
            va="top",
        )
        ax.text(
            x_positions[10],
            y_pos,
            f"{data['ttc']:.2f}",
            fontsize=font_size,
            color="red" if data["ttc"] < 0.95 else "green",
            va="top",
        )
        ax.text(
            x_positions[11],
            y_pos,
            f"{data['ttc_tfl']:.2f}",
            fontsize=font_size,
            color="red" if data["ttc_tfl"] < 0.95 else "green",
            va="top",
        )
        ax.text(
            x_positions[12],
            y_pos,
            f"{data['progress']:.2f}",
            fontsize=font_size,
            color="green" if data["progress"] > 0.2 else "red",
            va="top",
        )
        ax.text(
            x_positions[13],
            y_pos,
            f"{data['ml_time']:.1f}",
            fontsize=font_size,
            color="red" if data["ml_time"] > MULTI_LANE_FULL_SCORE_TIME else "green",
            va="top",
        )

    ax.set_title("Active Agent Metrics + V-Max", fontsize=font_size + 4, fontweight="bold", pad=10)


def _get_cache_key(reuse_key):
    return reuse_key


def _get_or_build_map_cache(cache_key, scenario):
    if cache_key:
        cache = _map_cache.get(cache_key)
        map_name = scenario.get("map_name")
        if cache and cache.get("map_name") == map_name:
            return cache
        road_cache = _build_road_cache(scenario.get("road_elements", []))
        traffic_cache = _build_traffic_cache(scenario.get("traffic_elements", []))
        cache = {
            "map_name": map_name,
            "road": road_cache,
            "traffic": traffic_cache,
        }
        _map_cache[cache_key] = cache
        return cache

    return {
        "map_name": scenario.get("map_name"),
        "road": _build_road_cache(scenario.get("road_elements", [])),
        "traffic": _build_traffic_cache(scenario.get("traffic_elements", [])),
    }


def plot_simulator_state(
    scenario,
    timestep: int = 0,
    show_trajectories: bool = False,
    simulation_mode: str = None,
    reuse_key: str = None,
) -> np.ndarray:
    """Render simulator state to RGB image array."""
    vis_radius = None if simulation_mode == "gigaflow" or simulation_mode is None else 75.0
    vis_config = VizConfig(radius=vis_radius, show_trajectories=show_trajectories)

    cache_key = _get_cache_key(reuse_key)
    map_cache = _get_or_build_map_cache(cache_key, scenario)

    bounds = vis_config.get_bounds(scenario)
    x_min, x_max, y_min, y_max = bounds

    px_per_meter = min(
        vis_config.figsize[0] * vis_config.dpi / (x_max - x_min),
        vis_config.figsize[1] * vis_config.dpi / (y_max - y_min),
    )

    if vis_config.debug_metrics:
        fig, ax, ax_metrics = _init_fig_ax(vis_config, cache_key, with_metrics=True)
    else:
        fig, ax = _init_fig_ax(vis_config, cache_key, with_metrics=False)
        ax_metrics = None

    ax.set_aspect("equal")
    ax.set_title(
        f"PufferDrive | {scenario.get('dataset_name', '')} | {scenario.get('scenario_id', '')} | t={timestep}",
        fontsize=max(14, int(px_per_meter / 8)),
        fontweight="bold",
    )

    _render_roads(ax, map_cache.get("road"))
    _render_traffic(ax, map_cache.get("traffic"), timestep)
    if vis_config.show_routes:
        _render_routes(
            ax,
            scenario.get("agents", []),
            map_cache.get("road", {}).get("lane_dict"),
            scenario.get("active_agent_indices", []),
        )
    if vis_config.show_sdc_paths:
        _render_paths(ax, scenario)
    if vis_config.show_trajectories and timestep > 0:
        _render_trajectories(ax, scenario)

    _render_agents(
        ax,
        scenario.get("agents", []),
        scenario.get("active_agent_indices", []),
        scenario.get("static_agent_indices", []),
        vis_config,
        px_per_meter,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if vis_config.debug_metrics and ax_metrics:
        _render_debug_metrics_table(
            ax_metrics,
            scenario.get("agents", []),
            scenario.get("active_agent_indices", []),
            px_per_meter=px_per_meter,
        )

    close_fig = not (vis_config.reuse_figure and cache_key)
    return _img_from_fig(fig, close=close_fig)


def _img_from_fig(fig: matplotlib.figure.Figure, close: bool = True) -> np.ndarray:
    fig.subplots_adjust(left=0.01, bottom=0.02, right=1.00, top=0.96)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
    if close:
        plt.close(fig)
    return img


def close_figure(reuse_key: str):
    if not reuse_key:
        return
    for suffix in ("single", "metrics"):
        cache_key = f"{reuse_key}_{suffix}"
        fig = _figure_cache.pop(cache_key, None)
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)
    _map_cache.pop(reuse_key, None)


def unpack_obs(
    obs_flat,
    dynamics_model=None,
    target_type: str = "static",
    reward_conditioning: bool = False,
    num_target_waypoints: int = 5,
    max_partners: int = 16,
    max_lane_segments: int = 16,
    max_boundary_segments: int = 16,
    obs_slots_traffic_controls: int = 16,
    obs_dropout_lane: float = 0.0,
    obs_dropout_boundary: float = 0.0,
    agent_idx: int = 0,
):
    """
    Unpack the flattened observation into ego, map, partner, and traffic-control views.
    Args:
        obs_flat: flattened observation tensor of shape (batch_size, obs_dim) or (obs_dim,)
    Return:
        ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_controls_obs
    """
    obs_flat = np.asarray(obs_flat)
    if obs_flat.ndim == 1:
        obs_flat = obs_flat[None, :]

    ego_dim = binding.EGO_FEATURES

    # Partner obs
    partner_feature_size = binding.PARTNER_FEATURES
    # Road obs
    road_feature_size = binding.ROAD_FEATURES
    # Traffic control obs
    traffic_control_feature_size = binding.TRAFFIC_CONTROL_FEATURES
    lane_segment_count = compute_effective_road_obs_count(max_lane_segments, obs_dropout_lane)
    boundary_segment_count = compute_effective_road_obs_count(max_boundary_segments, obs_dropout_boundary)

    # Target obs
    target_features = binding.STATIC_TARGET_FEATURES if target_type == "static" else binding.DYNAMIC_TARGET_FEATURES
    target_dim = num_target_waypoints * target_features

    # Extract ego state
    ego_state = obs_flat[:, :ego_dim]

    target_start = ego_dim
    if reward_conditioning:
        target_start += binding.NUM_REWARD_COEFS

    target_end = target_start + target_dim
    target_obs = obs_flat[:, target_start:target_end]
    target_obs = target_obs.reshape(-1, num_target_waypoints, target_features)

    # Extract partners
    partners_start = target_end
    partners_end = partners_start + max_partners * partner_feature_size
    partners_obs = obs_flat[:, partners_start:partners_end]
    partners_obs = partners_obs.reshape(-1, max_partners, partner_feature_size)

    # Extract lane elements
    lane_start = partners_end
    lane_end = lane_start + lane_segment_count * road_feature_size
    lane_obs = obs_flat[:, lane_start:lane_end]
    lane_obs = lane_obs.reshape(-1, lane_segment_count, road_feature_size)

    # Extract boundary elements
    boundary_start = lane_end
    boundary_end = boundary_start + boundary_segment_count * road_feature_size
    boundary_obs = obs_flat[:, boundary_start:boundary_end]
    boundary_obs = boundary_obs.reshape(-1, boundary_segment_count, road_feature_size)

    # Extract traffic controls
    traffic_start = boundary_end
    traffic_end = traffic_start + obs_slots_traffic_controls * traffic_control_feature_size
    if obs_slots_traffic_controls > 0:
        traffic_controls_obs = obs_flat[:, traffic_start:traffic_end]
        traffic_controls_obs = traffic_controls_obs.reshape(
            -1, obs_slots_traffic_controls, traffic_control_feature_size
        )
    else:
        traffic_controls_obs = np.zeros((obs_flat.shape[0], 0, traffic_control_feature_size))

    return (
        ego_state[agent_idx],
        target_obs[agent_idx],
        partners_obs[agent_idx],
        lane_obs[agent_idx],
        boundary_obs[agent_idx],
        traffic_controls_obs[agent_idx],
    )


def plot_observation(
    obs,
    dynamics_model=None,
    target_type="static",
    reward_conditioning=False,
    num_target_waypoints=10,
    max_partners=16,
    max_lane_segments=32,
    max_boundary_segments=32,
    obs_slots_traffic_controls=4,
    obs_dropout_lane=0.0,
    obs_dropout_boundary=0.0,
    agent_idx=0,
    obs_norm_goal_offset_m=100.0,
    obs_norm_xy_offset_m=100.0,
    obs_norm_veh_width_m=10.0,
    obs_norm_veh_length_m=15.0,
    obs_norm_road_seg_length_m=5.0,
    obs_norm_road_seg_width_m=5.0,
) -> np.ndarray:
    """Plot observation in ego-centric frame.

    Args:
        obs: flattened observation tensor
        target_type: 0 for goal only, 1 for waypoints only, 2 for both
    """
    fig, ax = plt.subplots(figsize=(20, 20))

    ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_controls_obs = unpack_obs(
        obs,
        target_type=target_type,
        reward_conditioning=reward_conditioning,
        num_target_waypoints=num_target_waypoints,
        max_partners=max_partners,
        max_lane_segments=max_lane_segments,
        max_boundary_segments=max_boundary_segments,
        obs_slots_traffic_controls=obs_slots_traffic_controls,
        obs_dropout_lane=obs_dropout_lane,
        obs_dropout_boundary=obs_dropout_boundary,
        agent_idx=agent_idx,
    )
    scales = _obs_scales(
        obs_norm_goal_offset_m=obs_norm_goal_offset_m,
        obs_norm_xy_offset_m=obs_norm_xy_offset_m,
        obs_norm_veh_width_m=obs_norm_veh_width_m,
        obs_norm_veh_length_m=obs_norm_veh_length_m,
        obs_norm_road_seg_length_m=obs_norm_road_seg_length_m,
        obs_norm_road_seg_width_m=obs_norm_road_seg_width_m,
    )
    target_position_scale = scales["goal_to_position"] if target_type == "static" else 1.0

    ego_speed, ego_width, ego_length, steering_angle, a_long, a_lat, lcenter, lalign, speed_limit, _ = ego_state

    ego_width *= scales["veh_width_to_position"]
    ego_length *= scales["veh_len_to_position"]

    # Ego vehicle at origin
    ax.add_patch(
        mpatches.Rectangle(
            (-ego_length / 2, -ego_width / 2),
            ego_length,
            ego_width,
            facecolor="#0055FF",
            edgecolor="#FFD700",
            linewidth=4,
            alpha=0.9,
            zorder=10,
        )
    )
    # SDC label above the vehicle
    ax.text(
        0,
        ego_width / 2 + 0.03,
        "SDC",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#FFD700",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0055FF", edgecolor="#FFD700", linewidth=1.5),
        zorder=11,
    )

    # Draw target waypoints
    for i in range(target_obs.shape[0]):
        if np.all(target_obs[i] == 0):
            continue
        wp_x = target_obs[i][0] * target_position_scale
        wp_y = target_obs[i][1] * target_position_scale
        if target_type == "static":
            color = "red" if i == 0 else "orange"
            marker = "*" if i == 0 else "o"
            s = 200 if i == 0 else 80
        else:
            color = "magenta"
            marker = "o"
            s = 100
        ax.scatter(wp_x, wp_y, color=color, marker=marker, s=s, zorder=15)

    # Add dynamics info text for JERK model
    ego_info = f"Speed: {ego_speed:.2f}\nLane Centering: {lcenter:.2f}\nLane Align: {lalign:.2f}\nSpeed Limit: {speed_limit:.2f}"

    ego_info += f"\nSteering: {steering_angle:.3f}\naccel_long: {a_long:.2f}\naccel_lat: {a_lat:.2f}"

    ax.text(
        0.02,
        0.98,
        ego_info,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Partner agents
    for i in range(partners_obs.shape[0]):
        if np.all(partners_obs[i] == 0):
            continue
        rel_x, rel_y = partners_obs[i][0], partners_obs[i][1]
        length = partners_obs[i][3] * scales["veh_len_to_position"]
        width = partners_obs[i][4] * scales["veh_width_to_position"]
        heading_cos, heading_sin = partners_obs[i][5], partners_obs[i][6]
        heading = np.arctan2(heading_sin, heading_cos)

        rect = mpatches.Rectangle(
            (-length / 2, -width / 2),
            length,
            width,
            facecolor="gray",
            edgecolor="black",
            linewidth=1,
            alpha=0.6,
            zorder=9,
        )
        rect.set_transform(plt.matplotlib.transforms.Affine2D().rotate(heading).translate(rel_x, rel_y) + ax.transData)
        ax.add_patch(rect)

    # Road elements
    rl2p = scales["road_length_to_position"]
    rw2p = scales["road_width_to_position"]
    count_lane = 0
    for i in range(lane_obs.shape[0]):
        if np.all(lane_obs[i] == 0):
            continue
        count_lane += 1
        rel_x, rel_y = lane_obs[i][0], lane_obs[i][1]
        length, width = lane_obs[i][3] * rl2p, lane_obs[i][4] * rw2p
        dir_cos, dir_sin = lane_obs[i][5], lane_obs[i][6]
        color = "lightgrey"
        ax.scatter(rel_x, rel_y, color=color, s=10, zorder=1)
        ax.plot(
            [rel_x + dir_cos * length / 2, rel_x - dir_cos * length / 2],
            [rel_y + dir_sin * length / 2, rel_y - dir_sin * length / 2],
            color=color,
            linewidth=1,
            zorder=1,
        )

    count_boundary = 0
    for i in range(boundary_obs.shape[0]):
        if np.all(boundary_obs[i] == 0):
            continue
        count_boundary += 1
        rel_x, rel_y = boundary_obs[i][0], boundary_obs[i][1]
        length, width = boundary_obs[i][3] * rl2p, boundary_obs[i][4] * rw2p
        dir_cos, dir_sin = boundary_obs[i][5], boundary_obs[i][6]
        color = "black"
        ax.scatter(rel_x, rel_y, color=color, s=10, zorder=1)
        ax.plot(
            [rel_x + dir_cos * length / 2, rel_x - dir_cos * length / 2],
            [rel_y + dir_sin * length / 2, rel_y - dir_sin * length / 2],
            color=color,
            linewidth=1,
            zorder=1,
        )

    ax.text(
        0.12,
        0.95,
        f"Lanes: {count_lane}\nBoundaries: {count_boundary}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Traffic controls
    for i in range(traffic_controls_obs.shape[0]):
        if np.all(traffic_controls_obs[i] == 0):
            continue
        rel_x1, rel_y1, rel_x2, rel_y2, _, control_type, state = traffic_controls_obs[i]
        control_type = int(control_type)
        if control_type == binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT:
            ax.plot(
                [rel_x1, rel_x2],
                [rel_y1, rel_y2],
                color=_traffic_light_color(state),
                linewidth=2.5,
                solid_capstyle="round",
                alpha=0.9,
                zorder=12,
            )
            continue

        overlay = "#FF0000" if control_type == binding.TRAFFIC_CONTROL_TYPE_STOP_SIGN else "#FFD700"
        ax.plot(
            [rel_x1, rel_x2],
            [rel_y1, rel_y2],
            color="black",
            linewidth=3.5,
            solid_capstyle="round",
            alpha=0.9,
            zorder=12,
        )
        ax.plot(
            [rel_x1, rel_x2],
            [rel_y1, rel_y2],
            color=overlay,
            linewidth=2.2,
            solid_capstyle="round",
            alpha=0.9,
            zorder=13,
            linestyle=(0, (3, 2)),
        )

    ax.axis((-1, 1, -1, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (ego frame)", fontsize=16)
    ax.set_ylabel("Y (ego frame)", fontsize=16)
    ax.set_title("Observation (Ego-Centric View)", fontsize=18, fontweight="bold")
    # ax.grid(True, alpha=0.3)
    return _img_from_fig(fig)


# HTML INTERACTIVE REPLAY
def fill_agents_state(scenario, use_trajectory=False):
    current_agents_data = []
    active_indices = scenario.get("active_agent_indices", [])
    road_elements = scenario.get("road_elements", [])

    for idx, agent in enumerate(scenario.get("agents", [])):
        if not agent.get("sim_valid"):
            continue

        agent_id = agent.get("id", idx)
        is_active = idx in active_indices
        color = "red" if agent.get("stopped", False) else get_agent_color(agent_id, is_active)

        metrics = agent.get("metrics_array", [])
        puffer_metrics = agent.get("puffer_metrics") if isinstance(agent.get("puffer_metrics"), dict) else None
        current_lane_idx = int(agent.get("current_lane_idx", -1))
        current_lane_id = -1
        if 0 <= current_lane_idx < len(road_elements):
            road_elem = road_elements[current_lane_idx]
            if isinstance(road_elem, dict):
                current_lane_id = int(road_elem.get("id", current_lane_idx))

        current_agents_data.append(
            {
                "id": int(agent_id),
                "x": round(float(agent["sim_x"]), 2),
                "y": round(float(agent["sim_y"]), 2),
                "z": round(float(agent.get("sim_z", 0)), 2),
                "h": round(float(agent["sim_heading"]), 3),
                "cl": current_lane_id,
                "l": round(float(agent["sim_length"]), 2),
                "w": round(float(agent["sim_width"]), 2),
                "s": round(float(agent.get("sim_speed", 0)), 2),
                "st": round(float(agent.get("sim_steering", 0)), 3),
                "al": round(float(agent.get("accel_long", 0)), 3),
                "alat": round(float(agent.get("accel_lat", 0)), 3),
                "jl": round(float(agent.get("jerk_long", 0)), 3),
                "jlat": round(float(agent.get("jerk_lat", 0)), 3),
                "c": color,
                "m": [round(float(m), 2) for m in metrics],
            }
        )
        if puffer_metrics:
            current_agents_data[-1]["pf"] = {k: round(float(v), 3) for k, v in puffer_metrics.items()}
        if puffer_metrics or "puffer_score" in agent:
            current_agents_data[-1]["ps"] = round(
                float(agent.get("puffer_score", puffer_metrics.get("score", 0.0) if puffer_metrics else 0.0)), 3
            )

    return current_agents_data


def fill_traffics_state(scenario, timestep):
    current_traffic_data = []
    traffic_elements = scenario.get("traffic_elements", [])
    for elem in traffic_elements or []:
        if not isinstance(elem, dict):
            continue

        t_type = elem.get("type", binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT)
        sl = elem.get("stop_line")
        if sl is None or len(sl) < 4:
            continue

        kind = _traffic_control_kind(t_type)
        if kind == "light":
            states = elem.get("states", [])
            state = int(states[timestep]) if states and len(states) > timestep else 0
            color = _traffic_light_color(state)
            current_traffic_data.append({"type": "light", "stop_line": sl, "c": color})
        elif kind == "stop":
            current_traffic_data.append({"type": "stop", "stop_line": sl, "c": "#FF0000", "c2": "#000000"})
        elif kind == "yield":
            current_traffic_data.append({"type": "yield", "stop_line": sl, "c": "#FFD700", "c2": "#000000"})

    return current_traffic_data


def fill_trajectories(scenario, timestep):
    current_trajectories = []
    if timestep > 0:
        traj_data = scenario.get("trajectory_waypoints_global", [])
        active_count = scenario.get("active_agent_count", 0)

        # On itère seulement sur les agents actifs qui ont des trajectoires
        for idx in range(min(len(traj_data), active_count)):
            waypoints = traj_data[idx].get("waypoints", [])
            pts = []
            for wp in waypoints:
                pts.append([float(wp["x"]), float(wp["y"]), float(wp["heading"])])

            current_trajectories.append(pts)
    return current_trajectories


def _scale_continuous_action(clipped_action, dynamics_model):
    clipped = np.asarray(clipped_action, dtype=np.float32)
    if clipped.size < 2:
        return clipped

    if dynamics_model == "jerk":
        j_long_action = float(clipped[0])
        j_long = j_long_action * (-JERK_LONG_VALUES[0]) if j_long_action < 0.0 else j_long_action * JERK_LONG_VALUES[-1]
        j_lat = float(clipped[1]) * JERK_LAT_VALUES[-1]
        return np.asarray([j_long, j_lat], dtype=np.float32)

    return np.asarray(
        [
            float(clipped[0]) * ACCELERATION_VALUES[-1],
            float(clipped[1]) * STEERING_VALUES[-1],
        ],
        dtype=np.float32,
    )


def _decode_discrete_action(action_value, dynamics_model):
    action_val = int(round(float(action_value)))
    if dynamics_model == "jerk":
        num_lat = len(JERK_LAT_VALUES)
        long_idx = int(np.clip(action_val // num_lat, 0, len(JERK_LONG_VALUES) - 1))
        lat_idx = int(np.clip(action_val % num_lat, 0, len(JERK_LAT_VALUES) - 1))
        raw = np.asarray([long_idx, lat_idx], dtype=np.float32)
        scaled = np.asarray([JERK_LONG_VALUES[long_idx], JERK_LAT_VALUES[lat_idx]], dtype=np.float32)
        labels = ["jerk_long", "jerk_lat"]
        return labels, raw, scaled

    num_steer = len(STEERING_VALUES)
    accel_idx = int(np.clip(action_val // num_steer, 0, len(ACCELERATION_VALUES) - 1))
    steer_idx = int(np.clip(action_val % num_steer, 0, len(STEERING_VALUES) - 1))
    raw = np.asarray([accel_idx, steer_idx], dtype=np.float32)
    scaled = np.asarray([ACCELERATION_VALUES[accel_idx], STEERING_VALUES[steer_idx]], dtype=np.float32)
    labels = ["accel", "steer"]
    return labels, raw, scaled


def _discrete_action_rows(probabilities, selected_action, dynamics_model):
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    rows = []
    for action_idx, probability in enumerate(probs):
        _, _, scaled = _decode_discrete_action(action_idx, dynamics_model)
        if dynamics_model == "jerk":
            detail = f"jerk_long {float(scaled[0]):.3g}, jerk_lat {float(scaled[1]):.3g}"
        else:
            detail = f"accel {float(scaled[0]):.3g}, steer {float(scaled[1]):.3g}"
        rows.append(
            {
                "index": int(action_idx),
                "label": f"{action_idx}: {detail}",
                "probability": float(probability),
                "selected": int(action_idx) == int(selected_action),
            }
        )
    return rows


def _scale_policy_params(action_type, clipped_action, trajectory_scaling_factors, dynamics_model):
    clipped = np.asarray(clipped_action, dtype=np.float32)

    if action_type == "continuous":
        return _scale_continuous_action(clipped, dynamics_model)

    if action_type == "discrete":
        return clipped

    scale_array = np.asarray(trajectory_scaling_factors, dtype=np.float32)
    if scale_array.size != clipped.shape[0]:
        return clipped

    scaled = clipped * scale_array
    if action_type in ("trajectory", "trajectory_frenet"):
        negative_longitudinal = clipped[:2] < 0.0
        scaled[:2] = clipped[:2] * scale_array[:2] * np.where(negative_longitudinal, 2.0, 1.0)
    return scaled


def _policy_param_labels(action_type, num_params, dynamics_model):
    if action_type == "continuous":
        if dynamics_model == "jerk":
            return ["jerk_long", "jerk_lat"][:num_params]
        return ["accel", "steer"][:num_params]
    if action_type == "discrete":
        if dynamics_model == "jerk":
            return ["jerk_long_idx", "jerk_lat_idx"]
        return ["accel_idx", "steer_idx"]
    return [f"p{idx}" for idx in range(num_params)]


def fill_policy_state(
    scenario,
    raw_actions,
    clipped_actions,
    values,
    entropies,
    action_type,
    trajectory_scaling_factors,
    dynamics_model,
    policy_outputs=None,
):
    active_indices = scenario.get("active_agent_indices", [])
    if (
        raw_actions is None
        or clipped_actions is None
        or values is None
        or entropies is None
        or len(active_indices) == 0
    ):
        return {}

    raw_array = np.asarray(raw_actions, dtype=np.float32)
    clipped_array = np.asarray(clipped_actions, dtype=np.float32)
    values_array = np.asarray(values, dtype=np.float32).reshape(-1)
    entropies_array = np.asarray(entropies, dtype=np.float32).reshape(-1)
    if raw_array.ndim == 1:
        raw_array = raw_array.reshape(-1, 1)
    if clipped_array.ndim == 1:
        clipped_array = clipped_array.reshape(-1, 1)

    if (
        raw_array.shape[0] != len(active_indices)
        or clipped_array.shape[0] != len(active_indices)
        or values_array.shape[0] != len(active_indices)
        or entropies_array.shape[0] != len(active_indices)
    ):
        return {}

    policy_state = {}
    agents = scenario.get("agents", [])
    output_array = None
    if policy_outputs is not None and not isinstance(policy_outputs, dict):
        output_array = np.asarray(policy_outputs, dtype=np.float32)
    for i, agent_idx in enumerate(active_indices):
        raw = raw_array[i].astype(np.float32, copy=False)
        clipped = clipped_array[i].astype(np.float32, copy=False)
        extra = {}

        if action_type == "discrete":
            labels, raw_display, scaled = _decode_discrete_action(raw[0], dynamics_model)
            if output_array is not None and output_array.shape[0] == len(active_indices):
                extra["selected_action"] = int(round(float(raw[0])))
                extra["action_probs"] = _discrete_action_rows(output_array[i], extra["selected_action"], dynamics_model)
        else:
            scaled = _scale_policy_params(action_type, clipped, trajectory_scaling_factors, dynamics_model)
            labels = _policy_param_labels(action_type, int(scaled.shape[0]), dynamics_model)
            raw_display = raw
            if action_type == "continuous" and isinstance(policy_outputs, dict):
                means = np.asarray(policy_outputs.get("mean"), dtype=np.float32)
                stds = np.asarray(policy_outputs.get("std"), dtype=np.float32)
                log_probs = np.asarray(policy_outputs.get("log_prob"), dtype=np.float32).reshape(-1)
                if means.shape[0] == len(active_indices) and stds.shape[0] == len(active_indices):
                    extra["density"] = {
                        "labels": labels,
                        "mean": [float(val) for val in means[i].reshape(-1).tolist()],
                        "std": [float(val) for val in stds[i].reshape(-1).tolist()],
                        "log_prob": float(log_probs[i]),
                    }

        agent_id = int(agent_idx)
        if 0 <= agent_idx < len(agents):
            agent_id = int(agents[agent_idx].get("id", agent_idx))

        policy_state[agent_id] = {
            "value": float(values_array[i]),
            "entropy": float(entropies_array[i]),
            "labels": labels,
            "raw": [float(val) for val in raw_display.tolist()],
            "scaled": [float(val) for val in scaled.tolist()],
        }
        policy_state[agent_id].update(extra)

    return policy_state


def extract_obs_frame(obs, scenario, args, timestep, obs_index=0, agent_idx=0, head_north=False):
    ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_controls_obs = unpack_obs(
        obs,
        target_type=args["env"]["target_type"],
        reward_conditioning=args["env"]["reward_conditioning"],
        num_target_waypoints=args["env"]["num_target_waypoints"],
        max_partners=args["env"]["obs_slots_partners"],
        max_lane_segments=args["env"]["obs_slots_lane"],
        max_boundary_segments=args["env"]["obs_slots_boundary"],
        obs_slots_traffic_controls=args["env"]["obs_slots_traffic_controls"],
        obs_dropout_lane=args["env"].get("obs_dropout_lane", 0.0),
        obs_dropout_boundary=args["env"].get("obs_dropout_boundary", 0.0),
        agent_idx=obs_index,
    )
    scales = _obs_scales(args.get("env"))
    target_position_scale = scales["goal_to_position"] if args["env"]["target_type"] == "static" else 1.0

    # --- Rotation Helper ---
    def _rot(x, y):
        """Rotates coordinates 90 degrees CCW if head_north is True."""
        return (-y, x) if head_north else (x, y)

    # --- Parse Ego ---
    ego_speed, ego_width, ego_length, steering_angle, accel_long, accel_lat = ego_state[:6]

    ego_width *= scales["veh_width_to_position"]
    ego_length *= scales["veh_len_to_position"]

    ego_data = {
        "s": round(float(ego_speed), 3),
        "w": round(float(ego_width), 3),
        "l": round(float(ego_length), 3),
        "st": round(float(steering_angle), 3),
        "al": round(float(accel_long), 3),
        "alat": round(float(accel_lat), 3),
    }

    # --- Parse Road Segments ---
    rl2p = scales["road_length_to_position"]
    rw2p = scales["road_width_to_position"]

    def parse_roads(roads):
        res = []
        for r in roads:
            if np.all(r == 0):
                continue
            x, y = r[0], r[1]
            length, width = r[3] * rl2p, r[4] * rw2p
            cos_a, sin_a = r[5], r[6]
            if head_north:
                x_rot, y_rot = _rot(x, y)
                cos_rot, sin_rot = _rot(cos_a, sin_a)
            else:
                x_rot, y_rot = x, y
                cos_rot, sin_rot = cos_a, sin_a
            res.append(
                [
                    round(float(x_rot), 4),
                    round(float(y_rot), 4),
                    round(float(length), 4),
                    round(float(width), 4),
                    round(float(cos_rot), 4),
                    round(float(sin_rot), 4),
                ]
            )
        return res

    # --- Parse Partners ---
    parsed_partners = []
    for p in partners_obs:
        if np.all(p == 0):
            continue

        px, py = _rot(p[0], p[1])
        h = math.atan2(p[6], p[5])

        if head_north:
            h += math.pi / 2
            h = (h + math.pi) % (2 * math.pi) - math.pi

        pl = float(p[3]) * scales["veh_len_to_position"]
        pw = float(p[4]) * scales["veh_width_to_position"]
        parsed_partners.append(
            {
                "x": round(float(px), 3),
                "y": round(float(py), 3),
                "w": round(pw, 3),
                "l": round(pl, 3),
                "h": round(float(h), 3),
                "s": round(float(p[7]), 3),
            }
        )

    # --- Parse Traffic Controls ---
    parsed_traffic_controls = []
    for t in traffic_controls_obs:
        if np.all(t == 0):
            continue
        kind = _traffic_control_kind(t[5])
        if kind is None:
            continue
        x1, y1 = _rot(t[0], t[1])
        x2, y2 = _rot(t[2], t[3])
        parsed_traffic_controls.append(
            {
                "type": kind,
                "x1": round(float(x1), 3),
                "y1": round(float(y1), 3),
                "x2": round(float(x2), 3),
                "y2": round(float(y2), 3),
                "state": int(t[6]),
            }
        )

    # --- Parse Trajectory & GPS ---
    traj_data = []
    if ("trajectory" in args["env"]["action_type"] or args.get("show_trajectories")) and timestep > 0:
        wps = scenario["trajectory_waypoints_local"][agent_idx]["waypoints"]
        for wp in wps:
            wx, wy = _rot(float(wp["x"]) / 70.0, float(wp["y"]) / 70.0)
            traj_data.append({"x": round(wx, 4), "y": round(wy, 4)})

    gps_data = []
    for g in target_obs:
        if np.all(g == 0):
            continue
        gx, gy = _rot(g[0] * target_position_scale, g[1] * target_position_scale)
        gps_data.append([round(float(gx), 3), round(float(gy), 3)])

    return {
        "ego": ego_data,
        "partners": parsed_partners,
        "lanes": parse_roads(lane_obs),
        "bounds": parse_roads(boundary_obs),
        "traffic_controls": parsed_traffic_controls,
        "traj": traj_data,
        "gps": gps_data,
    }


def _pack_replay_binary(header, chunks):
    packed = {}
    blob_parts = []
    offset = 0
    dtype_names = {
        np.dtype(np.float32): "float32",
        np.dtype(np.int32): "int32",
        np.dtype(np.int16): "int16",
        np.dtype(np.uint8): "uint8",
    }
    for name, arr in chunks.items():
        arr = np.ascontiguousarray(arr)
        dtype = dtype_names[arr.dtype]
        raw = arr.tobytes()
        packed[name] = {"dtype": dtype, "shape": list(arr.shape), "offset": offset, "nbytes": len(raw)}
        blob_parts.append(raw)
        offset += len(raw)
        pad = (-offset) % 4
        if pad:
            blob_parts.append(b"\0" * pad)
            offset += pad

    header = dict(header)
    header["chunks"] = packed
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(4 + len(header_bytes))) % 4
    payload = struct.pack("<I", len(header_bytes)) + header_bytes + (b"\0" * pad) + b"".join(blob_parts)
    return base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")


def _generate_compact_interactive_replay(scenario, replay, filename="replay.html", head_north=False):
    road_points = []
    road_lengths = []
    road_types = []
    for elem in scenario.get("road_elements", []) or []:
        if not isinstance(elem, dict):
            continue
        elem_type = int(elem.get("type", 0))
        xs = elem.get("x") or []
        ys = elem.get("y") or []
        if not xs or not ys:
            continue
        if 1 <= elem_type <= 3:
            draw_type = 0
        elif 11 <= elem_type <= 18:
            draw_type = 1
        elif 21 <= elem_type <= 23:
            draw_type = 2
        else:
            continue
        count = min(len(xs), len(ys))
        road_lengths.append(count)
        road_types.append(draw_type)
        for i in range(count):
            road_points.append((float(xs[i]), float(ys[i])))

    traffic_stop_lines = []
    traffic_types = []
    for elem in scenario.get("traffic_elements", []) or []:
        if not isinstance(elem, dict):
            continue
        stop_line = elem.get("stop_line") or [0, 0, 0, 0, 0, 0]
        traffic_stop_lines.append([float(v) for v in stop_line[:6]])
        traffic_types.append(int(elem.get("type", 0)))

    env_cfg = replay["env"]
    scales = _obs_scales(env_cfg)
    lane_count = compute_effective_road_obs_count(env_cfg["obs_slots_lane"], env_cfg.get("obs_dropout_lane", 0.0))
    boundary_count = compute_effective_road_obs_count(
        env_cfg["obs_slots_boundary"], env_cfg.get("obs_dropout_boundary", 0.0)
    )

    chunks = {
        "road_points": np.asarray(road_points or [(0.0, 0.0)], dtype=np.float32),
        "road_lengths": np.asarray(road_lengths or [0], dtype=np.int32),
        "road_types": np.asarray(road_types or [0], dtype=np.int16),
        "traffic_stop_lines": np.asarray(traffic_stop_lines or [[0, 0, 0, 0, 0, 0]], dtype=np.float32),
        "traffic_types": np.asarray(traffic_types or [0], dtype=np.int16),
        "agent_f32": replay["agent_f32"].astype(np.float32, copy=False),
        "agent_i32": replay["agent_i32"].astype(np.int32, copy=False),
        "metrics_f32": replay["metrics_f32"].astype(np.float32, copy=False),
        "puffer_f32": replay["puffer_f32"].astype(np.float32, copy=False),
        "traffic_i16": replay["traffic_i16"].astype(np.int16, copy=False),
        "obs": replay["obs"].astype(np.float32, copy=False),
        "raw_action": replay["raw_action"].astype(np.float32, copy=False),
        "clipped_action": replay["clipped_action"].astype(np.float32, copy=False),
        "value": replay["value"].astype(np.float32, copy=False),
        "entropy": replay["entropy"].astype(np.float32, copy=False),
    }
    if replay.get("policy_probs") is not None:
        chunks["policy_probs"] = replay["policy_probs"].astype(np.float32, copy=False)
    if replay.get("policy_mean") is not None:
        chunks["policy_mean"] = replay["policy_mean"].astype(np.float32, copy=False)
        chunks["policy_std"] = replay["policy_std"].astype(np.float32, copy=False)
        chunks["policy_log_prob"] = replay["policy_log_prob"].astype(np.float32, copy=False)

    metadata = {
        "map_name": scenario.get("map_name", "Unknown"),
        "scenario_id": scenario.get("scenario_id", "Unknown"),
        "target_type": scenario.get("target_type", env_cfg.get("target_type", "static")),
        "active_indices": scenario.get("active_agent_indices", []),
        "frames": int(replay["agent_f32"].shape[0]),
        "agent_cap": int(replay["agent_f32"].shape[1]),
        "traffic_cap": int(replay["traffic_i16"].shape[1]),
        "active_count": int(replay["obs"].shape[1]),
        "obs_dim": int(replay["obs"].shape[2]),
        "action_type": env_cfg.get("action_type", "continuous"),
        "dynamics_model": env_cfg.get("dynamics_model", "classic"),
        "trajectory_scaling_factors": env_cfg.get("trajectory_scaling_factors", []),
        "num_target_waypoints": int(env_cfg["num_target_waypoints"]),
        "reward_conditioning": bool(env_cfg["reward_conditioning"]),
        "max_partners": int(env_cfg["obs_slots_partners"]),
        "lane_count": int(lane_count),
        "boundary_count": int(boundary_count),
        "traffic_obs_count": int(env_cfg["obs_slots_traffic_controls"]),
        "target_features": 3 if env_cfg.get("target_type", "static") == "static" else 5,
        "head_north": bool(head_north),
        "scales": scales,
        "road_polyline_count": len(road_lengths),
        "traffic_static_count": len(traffic_types),
    }
    payload = _pack_replay_binary(metadata, chunks)

    html_template = """
<!DOCTYPE html>
<html data-theme="light">
<head>
    <meta charset="UTF-8">
    <title>PufferDrive Replay</title>
    <style>
        :root { --bg:#f3f4f6; --text:#111827; --panel:rgba(255,255,255,.94); --muted:#6b7280; --road:#c9cdd2; --line:#8a9099; --edge:#222831; --accent:#0b6bcb; --danger:#d71920; --shadow:rgba(15,23,42,.18); }
        [data-theme="dark"] { --bg:#111316; --text:#f3f4f6; --panel:rgba(28,31,36,.94); --muted:#aeb6c2; --road:#3c424a; --line:#6d7683; --edge:#050607; --accent:#48a6ff; --danger:#ff4d55; --shadow:rgba(0,0,0,.45); }
        body { margin:0; overflow:hidden; background:var(--bg); color:var(--text); font-family:system-ui,Segoe UI,sans-serif; user-select:none; }
        canvas { display:block; width:100vw; height:100vh; cursor:crosshair; }
        #ui-layer { position:absolute; inset:0; pointer-events:none; z-index:10; }
        .panel { background:var(--panel); border:1px solid rgba(127,127,127,.18); border-radius:8px; box-shadow:0 8px 28px var(--shadow); pointer-events:auto; backdrop-filter:blur(6px); }
        #loading-overlay { position:absolute; inset:0; z-index:9999; display:flex; align-items:center; justify-content:center; background:var(--bg); color:var(--text); font-size:18px; font-weight:800; }
        #hud-global { position:absolute; top:14px; left:14px; width:230px; padding:12px; }
        #hud-global.collapsed > *:not(h3) { display:none; }
        h3 { margin:0 0 10px 0; padding-bottom:6px; border-bottom:1px solid rgba(127,127,127,.35); color:var(--muted); font-size:12px; letter-spacing:.08em; text-transform:uppercase; }
        .label { margin-top:8px; color:var(--muted); font-size:10px; font-weight:800; letter-spacing:.06em; text-transform:uppercase; }
        .value { color:var(--text); font-size:15px; font-weight:800; overflow-wrap:anywhere; }
        .highlight { color:var(--accent); }
        button, select { border:0; border-radius:8px; padding:8px 10px; background:#23272f; color:white; font-weight:800; cursor:pointer; }
        button:hover { filter:brightness(1.12); }
        input[type=range] { width:320px; accent-color:var(--accent); }
        input[type=number] { width:74px; padding:8px; border:1px solid rgba(127,127,127,.35); border-radius:8px; background:var(--panel); color:var(--text); font-weight:800; }
        #controls { position:absolute; left:50%; bottom:18px; transform:translateX(-50%); padding:10px; display:flex; gap:10px; align-items:center; }
        #btnPlay { min-width:76px; }
        #search-box { position:absolute; right:14px; bottom:82px; display:flex; gap:8px; align-items:center; }
        #hud-telemetry { position:absolute; top:60px; right:14px; width:330px; max-height:calc(100vh - 92px); padding:12px; overflow-y:auto; display:none; border-left:5px solid var(--accent); color:white; background:rgba(18,20,24,.96); }
        #tel-drag-handle { cursor:grab; color:#dbeafe; }
        .tele-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px; }
        .tel-big { font-size:28px; font-family:ui-monospace,Consolas,monospace; font-weight:900; }
        .tel-mono { font-family:ui-monospace,Consolas,monospace; font-weight:800; }
        #obs-container { position:absolute; left:14px; bottom:18px; width:390px; height:390px; display:none; overflow:hidden; border:2px solid var(--accent); border-radius:8px; background:white; }
        #obs-title { position:absolute; top:0; left:0; right:0; z-index:2; padding:7px 10px; background:var(--accent); color:white; font-size:11px; font-weight:900; letter-spacing:.06em; cursor:grab; }
        #obs-canvas { width:100%; height:100%; }
        .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px; margin-top:8px; }
        .item { padding:6px; border:1px solid rgba(255,255,255,.08); border-radius:6px; background:rgba(255,255,255,.05); }
        .name { display:block; color:#aeb6c2; font-size:9px; font-weight:900; text-transform:uppercase; }
        .num { color:#74d99f; font-family:ui-monospace,Consolas,monospace; font-weight:900; font-size:12px; }
        .bar-row { position:relative; min-height:18px; overflow:hidden; border-radius:5px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08); }
        .bar-row.selected { border-color:var(--accent); background:rgba(72,166,255,.16); }
        .bar { position:absolute; inset:0 auto 0 0; background:rgba(72,166,255,.28); }
        .bar-text { position:relative; z-index:1; display:flex; justify-content:space-between; gap:8px; padding:2px 5px; font-size:10px; font-family:ui-monospace,Consolas,monospace; }
        .toggle-header { width:100%; margin-top:10px; padding:6px 0; display:flex; justify-content:space-between; align-items:center; background:transparent; color:#aeb6c2; border:0; border-bottom:1px solid rgba(255,255,255,.14); border-radius:0; font-size:10px; font-weight:900; letter-spacing:.06em; text-transform:uppercase; text-align:left; }
        .toggle-header:hover { filter:brightness(1.18); }
        .toggle-header span:last-child { transition:transform .15s ease; }
        .toggle-header.is-collapsed span:last-child { transform:rotate(-90deg); }
        .toggle-body.is-collapsed { display:none; }
        #crash-overlay { position:absolute; inset:0; z-index:5; display:none; pointer-events:none; background:radial-gradient(circle,transparent 45%,rgba(215,25,32,.38)); }
        #crash-msg { display:none; margin-bottom:10px; padding:7px; border:2px solid var(--danger); color:#ff777d; text-align:center; font-weight:950; }
    </style>
</head>
<body>
    <div id="loading-overlay">Loading replay...</div>
    <div id="crash-overlay"></div>
    <div id="ui-layer">
        <div id="hud-global" class="panel collapsed">
            <h3 onclick="toggleGlobalPanel()">Scenario <span id="globalChevron" style="float:right">&#9656;</span></h3>
            <div class="label">Map</div><div class="value" id="meta-map">-</div>
            <div class="label">ID</div><div class="value" id="meta-id">-</div>
            <div class="label">Step</div><div class="value highlight" id="stepDisplay" style="font-size:30px">0</div>
            <div class="label">Camera</div><div class="value highlight" id="camMode" onclick="toggleCamMode()">Free Roam</div>
            <button onclick="toggleTheme()" style="width:100%; margin-top:10px">Theme</button>
        </div>
        <div id="hud-telemetry" class="panel">
            <div id="crash-msg"></div>
            <h3 id="tel-drag-handle">Agent <span id="tel-id" class="highlight">?</span></h3>
            <div class="tele-row">
                <div><div class="label">Speed</div><div><span id="tel-speed" class="tel-big">0.0</span> km/h</div></div>
                <div><div class="label">Lane</div><div id="tel-lane-top" class="value highlight">-1</div></div>
            </div>
            <div class="tele-row">
                <div><div class="label">Steer</div><div id="tel-st" class="tel-mono">0.0</div></div>
                <div></div>
            </div>
            <div class="tele-row">
                <div><div class="label">Accel L/Lat</div><div class="tel-mono"><span id="tel-al">0</span> / <span id="tel-alat">0</span></div></div>
                <div><div class="label">Jerk L/Lat</div><div class="tel-mono"><span id="tel-jl">0</span> / <span id="tel-jlat">0</span></div></div>
            </div>
            <div class="label">Position (X/Y/H/Lane)</div>
            <div class="tel-mono"><span id="tel-x">0</span>, <span id="tel-y">0</span>, <span id="tel-h">0</span>, <span id="tel-lane">-1</span></div>
            <div class="label">Policy Outputs</div><div id="policy-grid" class="grid"></div>
            <button type="button" class="toggle-header" data-target="puffer-score-body"><span>Puffer Score</span><span>▾</span></button>
            <div id="puffer-score-body" class="toggle-body"><div id="tel-ps" class="tel-mono" style="margin-top:6px;">0.000</div></div>
            <button type="button" class="toggle-header" data-target="puffer-grid"><span>Puffer Metrics</span><span>▾</span></button>
            <div id="puffer-grid" class="grid toggle-body"></div>
            <button type="button" class="toggle-header" data-target="metrics-grid"><span>Metrics</span><span>▾</span></button>
            <div id="metrics-grid" class="grid toggle-body"></div>
        </div>
        <div id="obs-container"><div id="obs-title">EGO-CENTRIC NN OBS</div><canvas id="obs-canvas"></canvas></div>
        <div id="search-box"><input type="number" id="agentSearch" placeholder="ID" onkeydown="if(event.key==='Enter') searchAgent()"><button onclick="searchAgent()" class="panel">Search</button></div>
        <div id="controls" class="panel">
            <button id="btnPlay" onclick="toggle()">PLAY</button>
            <select id="speedSel" onchange="changeSpeed()"><option value="0.25">0.25x</option><option value="1">1x</option><option value="2">2x</option><option value="4" selected>4x</option><option value="8">8x</option></select>
            <input id="sld" type="range" min="0" value="0" step="1">
        </div>
    </div>
    <canvas id="c"></canvas>
    <script>
        const B64_PAYLOAD = "__B64_PAYLOAD__";
        const METRIC_LABELS = __METRIC_LABELS__;
        const VEHICLE_COLORS = __VEHICLE_COLORS__;
        const PUFFER_KEYS = [["score","score"],["no_at_fault","no at fault"],["no_offroad","no offroad"],["no_red_light","no red light"],["making_progress","progress > .2"],["direction_score","direction"],["ttc_puffer_rate","ttc"],["progress_ratio","progress"],["speed_limit_compliance","speed"],["comfort_score","comfort"],["multi_lane_score","multi lane"],["multiplier","multiplier"],["weighted_average","weighted avg"]];
        const ACCEL = [-4,-2.667,-1.333,0,1.333,2.667,4], STEER = [-0.667,-0.5,-0.333,-0.167,0,0.167,0.333,0.5,0.667];
        const JLONG = [-15,-4,0,4], JLAT = [-4,0,4];
        let H, C = {}, paths = {0:new Path2D(),1:new Path2D(),2:new Path2D()}, lastDrawn = -1;
        const c = document.getElementById('c'), ctx = c.getContext('2d');
        const obsC = document.getElementById('obs-canvas'), obsCtx = obsC.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        obsC.width = 390 * dpr; obsC.height = 390 * dpr;
        let step = 0, play = false, speed = 4, lastTick = 0;
        let cam = {x:0,y:0,z:5,drag:false,lx:0,ly:0};
        let followedId = null, isEgoCam = false, darkMode = false;

        function chunk(name) {
            const m = H.chunks[name], start = H.dataStart + m.offset, n = m.nbytes / ({float32:4,int32:4,int16:2,uint8:1}[m.dtype]);
            if (m.dtype === "float32") return new Float32Array(H.buffer, start, n);
            if (m.dtype === "int32") return new Int32Array(H.buffer, start, n);
            if (m.dtype === "int16") return new Int16Array(H.buffer, start, n);
            return new Uint8Array(H.buffer, start, n);
        }
        async function initReplay() {
            const binary = atob(B64_PAYLOAD), bytes = new Uint8Array(binary.length);
            for (let i=0;i<binary.length;i++) bytes[i] = binary.charCodeAt(i);
            const ds = new DecompressionStream('deflate');
            const buf = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer();
            const view = new DataView(buf), headerLen = view.getUint32(0, true);
            H = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
            H.buffer = buf; H.dataStart = 4 + headerLen + ((-(4 + headerLen)) & 3);
            for (const name of Object.keys(H.chunks)) C[name] = chunk(name);
            document.getElementById('meta-map').textContent = String(H.map_name).split('/').pop();
            document.getElementById('meta-id').textContent = H.scenario_id || "-";
            document.getElementById('sld').max = H.frames - 1;
            const first = getFrameAgents(0)[0]; if (first) { cam.x = first.x; cam.y = first.y; }
            document.getElementById('loading-overlay').style.display = 'none';
            window.onresize();
            requestAnimationFrame(() => { buildMapPaths(); draw(true); });
        }
        initReplay().catch(err => { console.error(err); document.getElementById('loading-overlay').textContent = 'Replay load failed. See console.'; });

        function buildMapPaths() {
            paths = {0:new Path2D(),1:new Path2D(),2:new Path2D()};
            let p = 0;
            for (let i=0;i<H.road_polyline_count;i++) {
                const len = C.road_lengths[i], type = C.road_types[i], path = paths[type];
                if (len <= 0) continue;
                path.moveTo(C.road_points[p*2], C.road_points[p*2+1]);
                for (let j=1;j<len;j++) path.lineTo(C.road_points[(p+j)*2], C.road_points[(p+j)*2+1]);
                p += len;
            }
        }
        function colorFor(id, active, stopped) { return stopped ? "red" : (active ? VEHICLE_COLORS[Math.abs(id) % VEHICLE_COLORS.length] : "#808080"); }
        function agentAt(frame, idx) {
            const ib = (frame * H.agent_cap + idx) * 8, fb = (frame * H.agent_cap + idx) * 12;
            if (!C.agent_i32[ib+2]) return null;
            const mb = (frame * H.agent_cap + idx) * 18, pb = (frame * H.agent_cap + idx) * 15;
            const m = Array.from(C.metrics_f32.subarray(mb, mb + 18));
            const pfVals = Array.from(C.puffer_f32.subarray(pb, pb + 15));
            const pf = {}; PUFFER_KEYS.forEach((row, i) => pf[row[0]] = pfVals[i]);
            return {id:C.agent_i32[ib], type:C.agent_i32[ib+1], active:C.agent_i32[ib+3], stopped:C.agent_i32[ib+4], removed:C.agent_i32[ib+5], cl:C.agent_i32[ib+6], slot:C.agent_i32[ib+7], x:C.agent_f32[fb], y:C.agent_f32[fb+1], z:C.agent_f32[fb+2], h:C.agent_f32[fb+3], l:C.agent_f32[fb+4], w:C.agent_f32[fb+5], s:C.agent_f32[fb+6], st:C.agent_f32[fb+7], al:C.agent_f32[fb+8], alat:C.agent_f32[fb+9], jl:C.agent_f32[fb+10], jlat:C.agent_f32[fb+11], c:colorFor(C.agent_i32[ib], C.agent_i32[ib+3], C.agent_i32[ib+4]), m:m, pf:pf, ps:pf.score};
        }
        function getFrameAgents(frame) { const out = []; for (let i=0;i<H.agent_cap;i++) { const a = agentAt(frame, i); if (a) out.push(a); } return out; }
        function findAgent(frame, id) { for (let i=0;i<H.agent_cap;i++) { const a = agentAt(frame, i); if (a && a.id === id) return a; } return null; }
        function trafficAt(frame, idx) {
            const db = (frame * H.traffic_cap + idx) * 3;
            if (!C.traffic_i16[db]) return null;
            const sb = idx * 6, type = C.traffic_types[idx] || C.traffic_i16[db+1], state = C.traffic_i16[db+2];
            return {type, state, stop_line:Array.from(C.traffic_stop_lines.subarray(sb, sb + 6))};
        }
        function trafficColor(t) { return t.state === 1 ? "#ff0000" : t.state === 2 ? "#ffff00" : t.state === 3 ? "#00ff00" : "#888888"; }
        function getColors() { const s = getComputedStyle(document.documentElement); return {bg:s.getPropertyValue('--bg'), road:s.getPropertyValue('--road'), line:s.getPropertyValue('--line'), edge:s.getPropertyValue('--edge'), text:s.getPropertyValue('--text')}; }
        window.onresize = () => { c.width = innerWidth; c.height = innerHeight; draw(true); };
        function toggleTheme(){ darkMode=!darkMode; document.documentElement.setAttribute('data-theme', darkMode?'dark':'light'); draw(true); }
        function toggleGlobalPanel(){ const p=document.getElementById('hud-global'), collapsed=!p.classList.contains('collapsed'); p.classList.toggle('collapsed', collapsed); document.getElementById('globalChevron').innerHTML=collapsed?'&#9656;':'&#9662;'; }
        function toggleCamMode(){ if(followedId !== null){ isEgoCam=!isEgoCam; draw(true); } }
        function searchAgent(){ const id=parseInt(document.getElementById('agentSearch').value); if(!isNaN(id)){ followedId=id; play=false; updateBtn(); draw(true); } }
        document.addEventListener('keydown', e => { if(!H || e.target.tagName === 'INPUT') return; if(e.code === 'Space'){ toggle(); e.preventDefault(); } if(e.code === 'ArrowRight'){ play=false; updateBtn(); step=Math.min(step+1,H.frames-1); draw(true); } if(e.code === 'ArrowLeft'){ play=false; updateBtn(); step=Math.max(step-1,0); draw(true); } if(e.code === 'Escape'){ followedId=null; isEgoCam=false; updateUI(); draw(true); } });
        c.onwheel = e => { e.preventDefault(); cam.z *= Math.exp(-e.deltaY * .001); draw(true); };
        c.onmousedown = e => { if(!H) return; const r=c.getBoundingClientRect(), wx=(e.clientX-r.left-c.width/2)/cam.z+cam.x, wy=(e.clientY-r.top-c.height/2)/-cam.z+cam.y; let hit=null, agents=getFrameAgents(Math.floor(step)); if(!isEgoCam) for(const a of agents) if(Math.hypot(wx-a.x, wy-a.y) < Math.max(a.l,3)){ hit=a.id; break; } if(hit !== null){ followedId=hit; cam.drag=false; } else { followedId=null; isEgoCam=false; cam.drag=true; cam.lx=e.clientX; cam.ly=e.clientY; } draw(true); };
        window.onmouseup = () => cam.drag = false;
        c.onmousemove = e => { if(cam.drag && !isEgoCam){ cam.x -= (e.clientX-cam.lx)/cam.z; cam.y -= (e.clientY-cam.ly)/-cam.z; cam.lx=e.clientX; cam.ly=e.clientY; draw(true); } };
        function dragPanel(handleId, panelId) { const h=document.getElementById(handleId), p=document.getElementById(panelId); let on=false,sx=0,sy=0,sl=0,st=0; h.addEventListener('mousedown', e => { on=true; sx=e.clientX; sy=e.clientY; const r=p.getBoundingClientRect(); sl=r.left; st=r.top; p.style.right='auto'; p.style.bottom='auto'; p.style.left=sl+'px'; p.style.top=st+'px'; }); window.addEventListener('mousemove', e => { if(on){ p.style.left=(sl+e.clientX-sx)+'px'; p.style.top=(st+e.clientY-sy)+'px'; }}); window.addEventListener('mouseup', () => on=false); }
        dragPanel('tel-drag-handle','hud-telemetry'); dragPanel('obs-title','obs-container');
        document.querySelectorAll('.toggle-header').forEach(header => header.addEventListener('click', () => {
            const body = document.getElementById(header.dataset.target);
            if (!body) return;
            const collapsed = !body.classList.contains('is-collapsed');
            body.classList.toggle('is-collapsed', collapsed);
            header.classList.toggle('is-collapsed', collapsed);
        }));

        function decodeObs(frame, slot) {
            if (slot < 0 || slot >= H.active_count) return null;
            const base = (frame * H.active_count + slot) * H.obs_dim, obs = C.obs;
            let p = base, ego = obs.subarray(p, p+10); p += 10;
            if (H.reward_conditioning) p += 17;
            const targetStart = p; p += H.num_target_waypoints * H.target_features;
            const partnersStart = p; p += H.max_partners * 8;
            const lanesStart = p; p += H.lane_count * 7;
            const boundsStart = p; p += H.boundary_count * 7;
            const trafficStart = p;
            const rot = (x,y) => H.head_north ? [-y,x] : [x,y];
            const zero = (off,n) => { for(let i=0;i<n;i++) if(obs[off+i] !== 0) return false; return true; };
            const roads = (start,count) => { const out=[]; for(let i=0;i<count;i++){ const o=start+i*7; if(zero(o,7)) continue; let xy=rot(obs[o],obs[o+1]), cs=H.head_north?rot(obs[o+5],obs[o+6]):[obs[o+5],obs[o+6]]; out.push([xy[0],xy[1],obs[o+3]*H.scales.road_length_to_position,obs[o+4]*H.scales.road_width_to_position,cs[0],cs[1]]); } return out; };
            const partners = []; for(let i=0;i<H.max_partners;i++){ const o=partnersStart+i*8; if(zero(o,8)) continue; let xy=rot(obs[o],obs[o+1]), h=Math.atan2(obs[o+6],obs[o+5]); if(H.head_north) h = ((h + Math.PI/2 + Math.PI) % (2*Math.PI)) - Math.PI; partners.push({x:xy[0],y:xy[1],l:obs[o+3]*H.scales.veh_len_to_position,w:obs[o+4]*H.scales.veh_width_to_position,h:h,s:obs[o+7]}); }
            const gps = []; for(let i=0;i<H.num_target_waypoints;i++){ const o=targetStart+i*H.target_features; if(zero(o,H.target_features)) continue; let scale=H.target_type === "static" ? H.scales.goal_to_position : 1, xy=rot(obs[o]*scale, obs[o+1]*scale); gps.push(xy); }
            const controls = []; for(let i=0;i<H.traffic_obs_count;i++){ const o=trafficStart+i*7; if(zero(o,7)) continue; let a=rot(obs[o],obs[o+1]), b=rot(obs[o+2],obs[o+3]); controls.push({type:obs[o+5], state:obs[o+6], x1:a[0], y1:a[1], x2:b[0], y2:b[1]}); }
            return {ego:{s:ego[0],w:ego[1]*H.scales.veh_width_to_position,l:ego[2]*H.scales.veh_len_to_position,st:ego[3],al:ego[4],alat:ego[5]}, partners, lanes:roads(lanesStart,H.lane_count), bounds:roads(boundsStart,H.boundary_count), gps, traffic_controls:controls};
        }
        function drawObs(frame) {
            const scale = (obsC.width / 2) * 2.2, px = dpr / scale;
            obsCtx.fillStyle = "#fff"; obsCtx.fillRect(0,0,obsC.width,obsC.height);
            obsCtx.save(); obsCtx.translate(obsC.width/2, obsC.height/2); obsCtx.scale(scale, -scale); obsCtx.lineCap = "round";
            obsCtx.strokeStyle="#bbb"; obsCtx.lineWidth=1.5*px; for(const r of frame.lanes){ obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[4]*r[2]/2,r[1]+r[5]*r[2]/2); obsCtx.lineTo(r[0]-r[4]*r[2]/2,r[1]-r[5]*r[2]/2); obsCtx.stroke(); }
            obsCtx.strokeStyle="#333"; obsCtx.lineWidth=3*px; for(const r of frame.bounds){ obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[4]*r[2]/2,r[1]+r[5]*r[2]/2); obsCtx.lineTo(r[0]-r[4]*r[2]/2,r[1]-r[5]*r[2]/2); obsCtx.stroke(); }
            for(const g of frame.gps){ obsCtx.fillStyle="magenta"; obsCtx.beginPath(); obsCtx.arc(g[0],g[1],5*px,0,7); obsCtx.fill(); }
            for(const t of frame.traffic_controls){ obsCtx.strokeStyle = t.type === 1 ? trafficColor({state:t.state}) : (t.type === 2 ? "#cc0000" : "#ffd700"); obsCtx.lineWidth=2.5*px; obsCtx.beginPath(); obsCtx.moveTo(t.x1,t.y1); obsCtx.lineTo(t.x2,t.y2); obsCtx.stroke(); }
            for(const p of frame.partners){ obsCtx.save(); obsCtx.translate(p.x,p.y); obsCtx.rotate(p.h); obsCtx.fillStyle="rgba(136,136,136,.8)"; obsCtx.strokeStyle="#333"; obsCtx.lineWidth=1.5*px; obsCtx.beginPath(); obsCtx.rect(-p.l/2,-p.w/2,p.l,p.w); obsCtx.fill(); obsCtx.stroke(); obsCtx.restore(); }
            if(frame.ego){ obsCtx.save(); if(H.head_north) obsCtx.rotate(Math.PI/2); obsCtx.fillStyle="rgba(0,102,255,.8)"; obsCtx.strokeStyle="#000"; obsCtx.lineWidth=1.5*px; obsCtx.beginPath(); obsCtx.rect(-frame.ego.l/2,-frame.ego.w/2,frame.ego.l,frame.ego.w); obsCtx.fill(); obsCtx.stroke(); obsCtx.restore(); }
            obsCtx.restore();
        }
        function policyFor(frame, agent) {
            if (agent.slot < 0) return "";
            const v = C.value[frame * H.active_count + agent.slot], ent = C.entropy[frame * H.active_count + agent.slot];
            const actionShape = H.chunks.raw_action.shape, actionDims = actionShape.length > 2 ? actionShape[2] : 1;
            const ab = (frame * H.active_count + agent.slot) * actionDims;
            const raw = Array.from(C.raw_action.subarray(ab, ab + actionDims)), clipped = Array.from(C.clipped_action.subarray(ab, ab + actionDims));
            let html = `<div class="item"><span class="name">value</span><span class="num">${v.toFixed(3)}</span></div><div class="item"><span class="name">entropy</span><span class="num">${ent.toFixed(3)}</span></div>`;
            if (H.action_type === "discrete" && C.policy_probs) {
                const n = H.chunks.policy_probs.shape[2], pb = (frame * H.active_count + agent.slot) * n, selected = Math.round(raw[0]);
                for(let i=0;i<n;i++){ const prob=Math.max(0,Math.min(1,C.policy_probs[pb+i])), values=H.dynamics_model==="jerk"?[JLONG[Math.floor(i/JLAT.length)],JLAT[i%JLAT.length]]:[ACCEL[Math.floor(i/STEER.length)],STEER[i%STEER.length]]; html += `<div class="bar-row ${i===selected?'selected':''}"><div class="bar" style="width:${(prob*100).toFixed(2)}%"></div><div class="bar-text"><span>${i}: ${values[0].toFixed(2)}, ${values[1].toFixed(2)}</span><span>${(prob*100).toFixed(1)}%</span></div></div>`; }
                return html;
            }
            let labels = H.action_type === "continuous" ? (H.dynamics_model === "jerk" ? ["jerk_long","jerk_lat"] : ["accel","steer"]) : raw.map((_,i)=>`p${i}`);
            let scaled = clipped.slice();
            if (H.action_type === "continuous") scaled = H.dynamics_model === "jerk" ? [clipped[0] < 0 ? clipped[0] * 15 : clipped[0] * 4, clipped[1] * 4] : [clipped[0] * 4, clipped[1] * .667];
            if (H.action_type.includes("trajectory") && H.trajectory_scaling_factors.length === clipped.length) scaled = clipped.map((x,i)=>x*H.trajectory_scaling_factors[i]*(i<2 && x<0 ? 2 : 1));
            labels.forEach((label,i) => html += `<div class="item"><span class="name">${label}</span><span class="num">${Number(scaled[i]).toFixed(2)} / ${Number(raw[i]).toFixed(2)}</span></div>`);
            if (C.policy_mean) { const mb = ab; labels.forEach((label,i) => html += `<div class="item"><span class="name">mean ${label}</span><span class="num">${C.policy_mean[mb+i].toFixed(3)}</span></div><div class="item"><span class="name">std ${label}</span><span class="num">${C.policy_std[mb+i].toFixed(3)}</span></div>`); html += `<div class="item"><span class="name">log prob</span><span class="num">${C.policy_log_prob[frame * H.active_count + agent.slot].toFixed(3)}</span></div>`; }
            return html;
        }
        function updateUI(agent=null) {
            const f = Math.floor(step); document.getElementById('stepDisplay').textContent = f; document.getElementById('sld').value = f;
            const hud = document.getElementById('hud-telemetry'), obsBox = document.getElementById('obs-container');
            if (followedId === null || !agent) { hud.style.display='none'; obsBox.style.display='none'; document.getElementById('crash-overlay').style.display='none'; document.getElementById('camMode').textContent='Free Roam'; return; }
            hud.style.display='block'; document.getElementById('camMode').textContent = isEgoCam ? 'LOCKED (EGO)' : 'LOCKED (WORLD)';
            for (const [id,val] of [["tel-id",agent.id],["tel-speed",(agent.s*3.6).toFixed(1)],["tel-st",(agent.st*180/Math.PI).toFixed(1)],["tel-al",agent.al.toFixed(2)],["tel-alat",agent.alat.toFixed(2)],["tel-jl",agent.jl.toFixed(2)],["tel-jlat",agent.jlat.toFixed(2)],["tel-x",agent.x.toFixed(1)],["tel-y",agent.y.toFixed(1)],["tel-h",agent.h.toFixed(3)],["tel-lane",agent.cl],["tel-lane-top",agent.cl],["tel-ps",agent.ps.toFixed(3)]]) document.getElementById(id).textContent = val;
            document.getElementById('metrics-grid').innerHTML = agent.m.map((v,i)=>`<div class="item"><span class="name">${METRIC_LABELS[i] || 'M'+i}</span><span class="num">${Number(v).toFixed(2)}</span></div>`).join('');
            document.getElementById('puffer-grid').innerHTML = PUFFER_KEYS.map(([k,n])=>`<div class="item"><span class="name">${n}</span><span class="num">${Number(agent.pf[k]).toFixed(3)}</span></div>`).join('');
            document.getElementById('policy-grid').innerHTML = policyFor(f, agent);
            const warnings = []; if(agent.m[0] === 1) warnings.push("COLLISION"); if(agent.m[1] === 1) warnings.push("OFFROAD"); if(agent.m[2] === 1) warnings.push("RED LIGHT"); if(agent.m[3] === 1) warnings.push("STOP SIGN");
            document.getElementById('crash-overlay').style.display = warnings.length ? 'block' : 'none'; document.getElementById('crash-msg').style.display = warnings.length ? 'block' : 'none'; document.getElementById('crash-msg').innerHTML = warnings.join('<br>');
            const obs = decodeObs(f, agent.slot); if (obs) { obsBox.style.display='block'; drawObs(obs); } else obsBox.style.display='none';
        }
        function draw(force=false) {
            if(!H || (!force && Math.floor(step) === lastDrawn && !play)) return;
            const f = Math.max(0, Math.min(H.frames - 1, Math.floor(step)));
            const target = followedId !== null ? findAgent(f, followedId) : null;
            if (target) { cam.x = target.x; cam.y = target.y; }
            updateUI(target);
            const colors = getColors(); ctx.fillStyle = colors.bg; ctx.fillRect(0,0,c.width,c.height); ctx.save(); ctx.translate(c.width/2,c.height/2); ctx.scale(cam.z,-cam.z); if(isEgoCam && target) ctx.rotate(Math.PI/2 - target.h); ctx.translate(-cam.x,-cam.y);
            ctx.lineCap='round'; ctx.strokeStyle=colors.road; ctx.lineWidth=.5; ctx.stroke(paths[0]); ctx.strokeStyle=colors.line; ctx.setLineDash([1,1]); ctx.stroke(paths[1]); ctx.setLineDash([]); ctx.strokeStyle=colors.edge; ctx.lineWidth=.8; ctx.stroke(paths[2]);
            for(const a of getFrameAgents(f)){ ctx.save(); ctx.translate(a.x,a.y); ctx.rotate(a.h); ctx.fillStyle=a.c; ctx.strokeStyle=darkMode?'#fff':'#111'; ctx.lineWidth=.1; ctx.beginPath(); ctx.rect(-a.l/2,-a.w/2,a.l,a.w); ctx.fill(); ctx.stroke(); ctx.fillStyle='rgba(255,255,0,.55)'; ctx.fillRect(a.l/2-.5,-a.w/2,.5,a.w); ctx.restore(); ctx.save(); ctx.translate(a.x,a.y); if(isEgoCam && target) ctx.rotate(-Math.PI/2 + target.h); else ctx.scale(1,-1); ctx.fillStyle=colors.text; ctx.font='bold '+(14/cam.z)+'px Arial'; ctx.textAlign='center'; ctx.fillText(a.id,0,(isEgoCam && target)?a.w/2+.5:-a.w/2-.5); ctx.restore(); if(a.id === followedId){ ctx.save(); ctx.translate(a.x,a.y); ctx.strokeStyle='#00ff00'; ctx.lineWidth=4/cam.z; ctx.beginPath(); ctx.arc(0,0,Math.max(a.l,a.w)*1.2,0,7); ctx.stroke(); ctx.restore(); } }
            for(let i=0;i<H.traffic_static_count;i++){ const t=trafficAt(f,i); if(!t) continue; const sl=t.stop_line; ctx.lineCap='butt'; if(t.type === 1){ ctx.strokeStyle=trafficColor(t); ctx.lineWidth=Math.max(1.5,3/cam.z); } else { ctx.strokeStyle=t.type === 2 ? '#ff0000' : '#ffd700'; ctx.lineWidth=Math.max(1.2,2.5/cam.z); ctx.setLineDash([6/cam.z,4/cam.z]); } ctx.beginPath(); ctx.moveTo(sl[0],sl[1]); ctx.lineTo(sl[3],sl[4]); ctx.stroke(); ctx.setLineDash([]); }
            ctx.restore(); lastDrawn = f;
        }
        function toggle(){ play=!play; lastTick=performance.now(); updateBtn(); if(play) requestAnimationFrame(loop); }
        function updateBtn(){ document.getElementById('btnPlay').textContent = play ? 'PAUSE' : 'PLAY'; }
        function changeSpeed(){ speed=parseFloat(document.getElementById('speedSel').value); lastTick=performance.now(); }
        function loop(ts){
            if(!play) return;
            const prev = Math.floor(step);
            const dt = Math.min((ts-lastTick)/1000, 0.25);
            lastTick = ts;
            step += dt * speed * 10;
            while(step >= H.frames) step -= H.frames;
            draw(Math.floor(step) !== prev);
            requestAnimationFrame(loop);
        }
        document.getElementById('sld').oninput = e => { step = +e.target.value; play=false; updateBtn(); draw(true); };
    </script>
</body>
</html>
    """
    final_html = (
        html_template.replace("__B64_PAYLOAD__", payload)
        .replace("__METRIC_LABELS__", json.dumps(METRIC_LABELS, separators=(",", ":")))
        .replace("__VEHICLE_COLORS__", json.dumps(VEHICLE_COLORS, separators=(",", ":")))
    )
    with open(filename, "w") as f:
        f.write(final_html)


def generate_interactive_replay(
    scenario,
    agent_history,
    traffic_history,
    trajectory_history,
    all_agents_obs_history,
    policy_history=None,
    filename="replay.html",
    head_north=False,
):
    if isinstance(policy_history, (str, os.PathLike)):
        filename = policy_history
        policy_history = None
    if isinstance(agent_history, dict) and agent_history.get("schema") == "obs_html_compact_v1":
        return _generate_compact_interactive_replay(scenario, agent_history, filename=filename, head_north=head_north)

    # --- 0. COMPRESSION HELPER ---
    def pack_and_compress_data(data, decimals=3):
        # Recursively round all floats to save string space
        def round_floats(o):
            if isinstance(o, float):
                return round(o, decimals)
            if isinstance(o, dict):
                return {k: round_floats(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [round_floats(v) for v in o]
            return o

        # Dump without whitespace
        compact_json = json.dumps(round_floats(data), separators=(",", ":"))

        # Compress using zlib (deflate)
        compressed_bytes = zlib.compress(compact_json.encode("utf-8"))

        # Return as Base64 string for safe HTML embedding
        return base64.b64encode(compressed_bytes).decode("ascii")

    metadata = {
        "map_name": scenario.get("map_name", "Unknown"),
        "scenario_id": scenario.get("scenario_id", "Unknown"),
        "target_type": scenario.get("target_type", "static"),
        "active_indices": str(scenario.get("active_agent_indices", [])),
    }

    # --- 2. MAP DATA ---
    map_data = {"lanes": [], "lines": [], "edges": []}
    for elem in scenario.get("road_elements", []):
        if not isinstance(elem, dict):
            continue
        t = elem.get("type", 0)
        if "x" in elem and "y" in elem:
            pts = [[float(x), float(y)] for x, y in zip(elem["x"], elem["y"])]
            if 1 <= t <= 3:
                map_data["lanes"].append(pts)
            elif 11 <= t <= 18:
                map_data["lines"].append(pts)
            elif 21 <= t <= 23:
                map_data["edges"].append(pts)

    # --- 3. TEMPLATE HTML ---
    html_template = """
<!DOCTYPE html>
<html data-theme="light">
<head>
    <title>PufferDrive Replay XXL</title>
    <style>
        :root {
            --bg: #e8e8e8; --text: #222; --panel-bg: rgba(255,255,255,0.95);
            --road: #bbb; --line: #999; --edge: #333;
            --hud-label: #888; --hud-val: #222;
            --btn-bg: #333; --btn-txt: #fff;
            --shadow: rgba(0,0,0,0.1);
            --accent: #007bff;
        }

        [data-theme="dark"] {
            --bg: #111; --text: #eee; --panel-bg: rgba(30,30,30,0.95);
            --road: #333; --line: #555; --edge: #000;
            --hud-label: #aaa; --hud-val: #fff;
            --btn-bg: #555; --btn-txt: #fff;
            --shadow: rgba(0,0,0,0.5);
            --accent: #3b9eff;
        }

        body { margin: 0; overflow: hidden; background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; user-select: none; transition: background 0.3s, color 0.3s; }
        canvas { display: block; width: 100vw; height: 100vh; cursor: crosshair; }
        #ui-layer { position: absolute; inset: 0; pointer-events: none; z-index: 10; }
        .panel { background: var(--panel-bg); padding: 18px; border-radius: 16px; box-shadow: 0 8px 30px var(--shadow); pointer-events: auto; backdrop-filter: blur(5px); }
        #hud-global { position: absolute; top: 20px; left: 20px; min-width: 220px; }
        #hud-global.collapsed > *:not(h3) { display: none; }
        #hud-global h3 { cursor: pointer; margin-bottom: 0; }
        #hud-global:not(.collapsed) h3 { margin-bottom: 12px; }
        #hud-telemetry {
            position: absolute; top: 80px; right: 20px; width: 320px; max-height: calc(100vh - 120px);
            display: none; overflow-y: auto; border-left: 6px solid var(--accent);
            background: rgba(15, 15, 15, 0.98); color: white; z-index: 20;
        }
        #tel-drag-handle { margin: 0 0 10px 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1.2px; border-bottom: 1px solid #444; padding-bottom: 5px; color: #eee; cursor: grab; }
        #tel-drag-handle:active { cursor: grabbing; }
        #controls { position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%); padding: 12px 30px; border-radius: 50px; display: flex; gap: 20px; align-items: center; z-index: 20; }
        #search-box { position: absolute; bottom: 110px; right: 20px; display: flex; gap: 8px; align-items: center; pointer-events: auto; z-index: 20; }
        #obs-container {
            position: absolute; bottom: 30px; left: 20px; width: 400px; height: 400px;
            background: rgba(255, 255, 255, 0.95); border-radius: 16px; box-shadow: 0 8px 30px var(--shadow);
            display: none; border: 3px solid var(--accent); overflow: hidden; pointer-events: auto;
            backdrop-filter: blur(5px); z-index: 20;
        }
        #obs-canvas { width: 100%; height: 100%; display: block; }
        #obs-title { position: absolute; top: 0; left: 0; width: 100%; padding: 8px 12px; font-size: 11px; font-weight: 900; color: #fff; background: var(--accent); z-index: 2; letter-spacing: 1px; cursor: grab; box-sizing: border-box;}
        #obs-title:active { cursor: grabbing; }
        h3 { margin: 0 0 12px 0; font-size: 16px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--hud-label); border-bottom: 1px solid #444; padding-bottom: 6px; }
        .label { font-size: 11px; color: #888; margin-top: 10px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.4px; }
        .value { font-size: 18px; font-weight: 800; color: var(--hud-val); }
        .highlight { color: var(--accent); }
        .val-speed { font-size: 28px; color: #fff; font-family: 'Courier New', monospace; }
        .val-action { font-size: 18px; color: var(--accent); font-family: 'Courier New', monospace; }
        .val-subtle { font-size: 12px; color: #9eb0bb; font-family: 'Courier New', monospace; }
        #puffer-score-wrap { display: none; margin-top: 12px; }
        #policy-block { display: none; margin-top: 14px; }
        #policy-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }
        .policy-item {
            display: flex; flex-direction: column; gap: 3px; padding: 6px 8px; border-radius: 8px;
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
        }
        .policy-name { color: #aaa; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; }
        .policy-values { display: flex; justify-content: space-between; gap: 8px; font-family: 'Courier New', monospace; }
        .policy-scaled { color: #7ed7ff; font-size: 13px; font-weight: 900; }
        .policy-raw { color: #c8c8c8; font-size: 10px; }
        .policy-hist { grid-column: 1 / -1; display: flex; flex-direction: column; gap: 3px; }
        .policy-row {
            position: relative; min-height: 18px; overflow: hidden; border-radius: 6px;
            background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06);
        }
        .policy-row.selected { border-color: var(--accent); background: rgba(0, 229, 255, 0.12); }
        .policy-bar { position: absolute; inset: 0 auto 0 0; background: rgba(126,215,255,0.24); }
        .policy-row-text {
            position: relative; z-index: 1; display: flex; justify-content: space-between; gap: 8px;
            padding: 2px 6px; color: #c8d3d9; font-family: 'Courier New', monospace; font-size: 10px;
        }
        .policy-row.selected .policy-row-text { color: #fff; font-weight: 900; }
        #puffer-block { display: none; margin-top: 14px; }
        #puffer-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 8px; }
        .puffer-header { grid-column: 1 / -1; margin-top: 10px; border-bottom: 1px solid #444; font-size: 11px; font-weight: 800; color: #888; text-transform: uppercase; padding-bottom: 4px; }
        .puffer-item {
            display: flex; flex-direction: column; gap: 3px; padding: 6px 8px; border-radius: 8px;
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
        }
        .puffer-name { color: #aaa; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; }
        .puffer-val { color: #ffd166; font-family: 'Courier New', monospace; font-weight: 900; font-size: 13px; }
        #metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 8px; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }
        .metric-item { display: flex; flex-direction: column; border-bottom: 1px solid #333; padding-bottom: 4px; }
        .m-name { color: #aaa; font-weight: bold; font-size: 10px; text-transform: uppercase; margin-bottom: 2px; }
        .m-val { color: #00ff88; font-family: 'Courier New', monospace; font-weight: 900; font-size: 15px; }
        .collapsible-section { margin-top: 14px; }
        button.collapsible-header {
            width: 100%; display: flex; align-items: center; justify-content: space-between; gap: 10px;
            padding: 0; margin: 0; background: transparent; color: var(--accent); border: 0; border-bottom: 1px solid #333;
            border-radius: 0; font-size: 11px; font-weight: 800; letter-spacing: 0.4px; text-transform: uppercase; text-align: left;
            transform: none; filter: none;
        }
        button.collapsible-header:hover { transform: none; filter: brightness(1.15); }
        .collapsible-icon { color: #bbb; font-size: 12px; transition: transform 0.15s ease; }
        .collapsible-body { overflow: hidden; }
        .collapsible-section.is-collapsed .collapsible-icon { transform: rotate(-90deg); }
        button { cursor: pointer; padding: 10px 20px; background: var(--btn-bg); color: var(--btn-txt); border: none; border-radius: 25px; font-weight: 800; font-size: 13px; transition: 0.2s; }
        button:hover { transform: scale(1.05); filter: brightness(1.2); }
        select { padding: 8px; border-radius: 20px; font-weight: bold; cursor: pointer; border: none; }
        input[type=range] { width: 280px; cursor: pointer; accent-color: var(--accent); }
        input[type=number] { width: 80px; padding: 12px; border-radius: 15px; border: 2px solid #444; background: var(--panel-bg); color: var(--text); font-weight: bold; text-align: center; outline: none; transition: border-color 0.2s; }
        input[type=number]:focus { border-color: var(--accent); }
        .crash-overlay { position: absolute; inset: 0; background: radial-gradient(circle, transparent 40%, rgba(255,0,0,0.6) 100%); display: none; pointer-events: none; z-index: 999; animation: pulse 0.4s infinite; }
        @keyframes pulse { 0% {opacity: 0.4;} 50% {opacity: 1;} 100% {opacity: 0.4;} }
        #crash-msg { color: #ff3333; font-weight: 950; font-size: 24px; display: none; text-align: center; margin-bottom: 15px; border: 3px solid red; padding: 8px; background: rgba(0,0,0,0.5); }
        #help-hint { position: absolute; bottom: 10px; right: 20px; font-size: 12px; color: var(--hud-label); opacity: 0.6; }
        #loading-overlay { position: absolute; inset: 0; background: var(--bg); color: var(--text); z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center; font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
    <div id="loading-overlay">Unpacking Replay Data...</div>
    <div id="crash-overlay" class="crash-overlay"></div>
    <div id="help-hint">SPACE: Play | ARROWS: Step | ESC: Free | CLICK: Follow | ENTER: Search</div>

    <div id="ui-layer">
        <div id="hud-global" class="panel collapsed">
            <h3 onclick="toggleGlobalPanel()" title="Click to expand/minimize">Scenario Info <span id="globalChevron" style="float:right;">&#9656;</span></h3>
            <div class="label">Map</div> <div class="value" id="meta-map">-</div>
            <div class="label">ID</div> <div class="value small-val" id="meta-id" style="font-size:12px">-</div>
            <hr style="border: 0; border-top: 1px solid #555; margin: 12px 0;">
            <div class="label">Step</div> <div class="value" style="font-size: 32px; color:var(--accent)" id="stepDisplay">0</div>
            <div class="label">Camera Mode</div>
            <div class="value highlight" id="camMode" onclick="toggleCamMode()" title="Click to Toggle World/Ego">Free Roam</div>
            <button onclick="toggleTheme()" style="width:100%; margin-top:15px; font-size:11px">THEME</button>
        </div>

        <div id="hud-telemetry" class="panel">
            <div id="crash-msg">COLLISION</div>
            <h3 id="tel-drag-handle">DRAG | Agent <span id="tel-id" style="color:var(--accent)">?</span></h3>

            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="label" style="margin-top:0;">Speed</div>
                    <div><span class="val-speed" id="tel-speed">0.0</span> <span style="font-size:14px; color:#888">km/h</span></div>
                    <div style="margin-top:6px;">
                        <div class="label" style="margin-top:0;">Steering Angle</div>
                        <div class="val-subtle"><span id="tel-st">0.0</span><span style="margin-left:2px;">°</span></div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="label" style="margin-top:0;">Lane</div>
                    <div class="value highlight" id="tel-lane-top">-1</div>
                </div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 12px;">
                <div>
                    <div class="label" style="margin-top:0;">Accel Long</div>
                    <div class="value highlight" id="tel-al">0.00</div>
                </div>
                <div style="text-align: right;">
                    <div class="label" style="margin-top:0;">Accel Lat</div>
                    <div class="value highlight" id="tel-alat">0.00</div>
                </div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-top: 10px;">
                <div>
                    <div class="label" style="margin-top:0;">Jerk Long</div>
                    <div class="value highlight" id="tel-jl">0.00</div>
                </div>
                <div style="text-align: right;">
                    <div class="label" style="margin-top:0;">Jerk Lat</div>
                    <div class="value highlight" id="tel-jlat">0.00</div>
                </div>
            </div>

            <div id="puffer-score-wrap" class="collapsible-section">
                <button type="button" class="collapsible-header" data-target="puffer-score-body" aria-expanded="true">
                    <span>Puffer Score</span>
                    <span class="collapsible-icon">▾</span>
                </button>
                <div id="puffer-score-body" class="collapsible-body">
                    <div class="value highlight" id="tel-ps" style="margin-top: 8px;">0.000</div>
                </div>
            </div>

            <div id="policy-block" class="collapsible-section">
                <button type="button" class="collapsible-header" data-target="policy-grid" aria-expanded="true">
                    <span>Policy Outputs</span>
                    <span class="collapsible-icon">▾</span>
                </button>
                <div id="policy-grid" class="collapsible-body"></div>
            </div>

            <div id="puffer-block" class="collapsible-section">
                <button type="button" class="collapsible-header" data-target="puffer-grid" aria-expanded="true">
                    <span>Puffer Metrics</span>
                    <span class="collapsible-icon">▾</span>
                </button>
                <div id="puffer-grid" class="collapsible-body"></div>
            </div>

            <div id="metrics-block" class="collapsible-section">
                <button type="button" class="collapsible-header" data-target="metrics-grid" aria-expanded="true">
                    <span>Metrics Table</span>
                    <span class="collapsible-icon">▾</span>
                </button>
                <div id="metrics-grid" class="collapsible-body"></div>
            </div>

            <div class="label" style="margin-top: 15px;">Position (X/Y/H/Lane)</div>
            <div style="font-family: monospace; font-size: 15px; color: #ccc; font-weight: bold;">
                <span id="tel-x">0</span> , <span id="tel-y">0</span> , <span id="tel-h">0</span> , <span id="tel-lane">-1</span>
            </div>
        </div>

        <div id="obs-container">
            <div id="obs-title">DRAG TO MOVE | EGO-CENTRIC NN OBS</div>
            <canvas id="obs-canvas"></canvas>
        </div>

        <div id="search-box">
             <input type="number" id="agentSearch" placeholder="ID" onkeydown="if(event.key==='Enter') searchAgent()">
             <button onclick="searchAgent()" class="panel" style="border-radius:15px; padding: 12px 18px;">Search</button>
        </div>

        <div id="controls" class="panel">
            <button id="btnPlay" onclick="toggle()" style="min-width: 100px; font-size: 16px;">PLAY</button>
            <select id="speedSel" onchange="changeSpeed()">
                <option value="0.5">0.5x</option>
                <option value="1.0">1x</option>
                <option value="2.0">2x</option>
                <option value="4.0" selected>4x</option>
                <option value="8.0">8x</option>
            </select>
            <input id="sld" type="range" min="0" value="0" step="1">
        </div>
    </div>

    <canvas id="c"></canvas>

    <script>
        const B64_PAYLOAD = "__COMPRESSED_PAYLOAD__";
        let MAP, AGENTS, TRAFFIC, TRAJ, META, ALL_OBS, POLICY, HEAD_NORTH;

        const c=document.getElementById('c'), ctx=c.getContext('2d');
        const obsC = document.getElementById('obs-canvas'), obsCtx = obsC.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        obsC.width = 400 * dpr; obsC.height = 400 * dpr;

        let step=0, play=false, speed=4.0, lastTick=0;
        let cam={x:0, y:0, z:5, drag:false, lx:0, ly:0};
        let followedId = null, darkMode = false, isEgoCam = false;
        const collapsedPanels = {
            "policy-grid": false,
            "puffer-score-body": false,
            "puffer-grid": false,
            "metrics-grid": false,
        };
        const METRIC_LABELS = ["collision", "offroad", "red_light", "stop_sign", "reached_goal", "lane_dist", "lane_angle", "comfort_violation", "velocity_progress", "speed_limit", "ADE", "progression", "at_fault_collision", "ttc", "ttc_tfl", "progress_ratio", "multi_lane_time", "multi_lane_score"];
        const PUFFER_MULTIPLIERS = [
            ["no_at_fault", "No At Fault"],
            ["no_offroad", "No Offroad"],
            ["no_red_light", "No Red Light"],
            ["making_progress", "Progress > 0.2"],
            ["direction_score", "Direction"],
            ["multiplier", "Multiplier"]
        ];
        const PUFFER_WEIGHTED = [
            ["ttc_puffer_rate", "TTC (w5)"],
            ["progress_ratio", "Progress (w5)"],
            ["speed_limit_compliance", "Speed Compliance (w4)"],
            ["comfort_score", "Comfort (w2)"],
            ["multi_lane_score", "Multi Lane (w3)"],
            ["weighted_average", "Weighted Avg"]
        ];

        async function initReplay() {
            try {
                const binaryStr = atob(B64_PAYLOAD);
                const bytes = new Uint8Array(binaryStr.length);
                for (let i = 0; i < binaryStr.length; i++) {
                    bytes[i] = binaryStr.charCodeAt(i);
                }

                const ds = new DecompressionStream('deflate');
                const stream = new Blob([bytes]).stream().pipeThrough(ds);
                const decompressedText = await new Response(stream).text();
                const data = JSON.parse(decompressedText);

                MAP = data.map;
                AGENTS = data.agents;
                TRAFFIC = data.traffic;
                TRAJ = data.traj;
                META = data.meta;
                ALL_OBS = data.obs;
                POLICY = data.policy || [];
                HEAD_NORTH = data.head_north;

                document.getElementById('meta-map').innerText = META.map_name.split('binaries/')[1] || META.map_name;
                document.getElementById('meta-id').innerText = META.scenario_id;
                if (AGENTS[0]?.length) { cam.x = AGENTS[0][0].x; cam.y = AGENTS[0][0].y; }
                document.getElementById('sld').max = AGENTS.length - 1;
                document.getElementById('loading-overlay').style.display = 'none';
                window.onresize();
            } catch (err) {
                console.error("Failed to unpack replay data:", err);
                document.getElementById('loading-overlay').innerText = "Error loading replay data. Check console.";
            }
        }
        initReplay();

        const obsTitle = document.getElementById('obs-title');
        const obsCont = document.getElementById('obs-container');
        let isDraggingPiP = false, startX, startY, startLeft, startTop;
        const telHandle = document.getElementById('tel-drag-handle');
        const telCont = document.getElementById('hud-telemetry');
        let isDraggingTel = false, telStartX, telStartY, telStartLeft, telStartTop;

        obsTitle.addEventListener('mousedown', (e) => {
            isDraggingPiP = true;
            startX = e.clientX; startY = e.clientY;
            const rect = obsCont.getBoundingClientRect();
            startLeft = rect.left; startTop = rect.top;
            obsCont.style.bottom = 'auto'; obsCont.style.right = 'auto';
            obsCont.style.left = startLeft + 'px'; obsCont.style.top = startTop + 'px';
        });

        telHandle.addEventListener('mousedown', (e) => {
            isDraggingTel = true;
            telStartX = e.clientX; telStartY = e.clientY;
            const rect = telCont.getBoundingClientRect();
            telStartLeft = rect.left; telStartTop = rect.top;
            telCont.style.right = 'auto';
            telCont.style.left = telStartLeft + 'px';
            telCont.style.top = telStartTop + 'px';
        });

        window.addEventListener('mousemove', (e) => {
            if (isDraggingPiP) {
                obsCont.style.left = (startLeft + e.clientX - startX) + 'px';
                obsCont.style.top = (startTop + e.clientY - startY) + 'px';
            }
            if (isDraggingTel) {
                telCont.style.left = (telStartLeft + e.clientX - telStartX) + 'px';
                telCont.style.top = (telStartTop + e.clientY - telStartY) + 'px';
            }
        });

        window.addEventListener('mouseup', () => {
            isDraggingPiP = false;
            isDraggingTel = false;
        });

        function getColors() {
            const style = getComputedStyle(document.body);
            return {
                bg: style.getPropertyValue('--bg').trim(),
                road: style.getPropertyValue('--road').trim(),
                line: style.getPropertyValue('--line').trim(),
                edge: style.getPropertyValue('--edge').trim(),
                text: style.getPropertyValue('--text').trim()
            };
        }

        window.onresize = () => { c.width=window.innerWidth; c.height=window.innerHeight; draw(); };
        function toggleTheme() { darkMode = !darkMode; document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light'); draw(); }

        function toggleGlobalPanel() {
            const panel = document.getElementById('hud-global');
            const chevron = document.getElementById('globalChevron');
            const collapsed = !panel.classList.contains('collapsed');
            panel.classList.toggle('collapsed', collapsed);
            chevron.innerHTML = collapsed ? '&#9656;' : '&#9662;';
        }

        function setSectionCollapsed(targetId, collapsed) {
            collapsedPanels[targetId] = collapsed;
            const body = document.getElementById(targetId);
            if (!body) return;
            const section = body.closest('.collapsible-section');
            if (!section) return;
            section.classList.toggle('is-collapsed', collapsed);
            body.style.display = collapsed ? 'none' : '';
            const header = section.querySelector('.collapsible-header');
            if (header) {
                header.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            }
        }

        function toggleSection(targetId) {
            setSectionCollapsed(targetId, !collapsedPanels[targetId]);
        }

        function ensureTelemetrySectionOrder() {
            const policyBlock = document.getElementById('policy-block');
            const pufferBlock = document.getElementById('puffer-block');
            const metricsBlock = document.getElementById('metrics-block');
            if (!policyBlock || !pufferBlock || !metricsBlock) return;

            const parent = policyBlock.parentNode;
            if (!parent || parent !== pufferBlock.parentNode || parent !== metricsBlock.parentNode) return;

            if (parent.firstElementChild !== policyBlock) {
                parent.insertBefore(policyBlock, pufferBlock);
            }
            if (policyBlock.nextElementSibling !== pufferBlock) {
                parent.insertBefore(pufferBlock, metricsBlock);
            }
            if (pufferBlock.nextElementSibling !== metricsBlock) {
                parent.appendChild(metricsBlock);
            }
        }

        document.querySelectorAll('.collapsible-header').forEach((header) => {
            header.addEventListener('click', () => toggleSection(header.dataset.target));
        });

        Object.keys(collapsedPanels).forEach((targetId) => setSectionCollapsed(targetId, collapsedPanels[targetId]));
        ensureTelemetrySectionOrder();

        function toggleCamMode() {
            if (followedId !== null) {
                isEgoCam = !isEgoCam;
                updateUI(AGENTS[Math.floor(step)].find(a => a.id === followedId));
                draw();
            }
        }

        function searchAgent() {
            if (!AGENTS) return;
            const id = parseInt(document.getElementById('agentSearch').value);
            if (!isNaN(id)) {
                followedId = id;
                play = false;
                updateBtn();
                draw();
            }
        }

        document.addEventListener('keydown', (e) => {
            if (!AGENTS) return;
            if (e.target.tagName === 'INPUT') return;
            if (e.code === "Space") { toggle(); e.preventDefault(); }
            if (e.code === "ArrowRight") { play = false; updateBtn(); step = Math.min(step + 1, AGENTS.length - 1); draw(); }
            if (e.code === "ArrowLeft") { play = false; updateBtn(); step = Math.max(step - 1, 0); draw(); }
            if (e.code === "Escape") { followedId = null; isEgoCam = false; updateUI(); draw(); }
        });

        c.onwheel = e => { e.preventDefault(); cam.z *= Math.exp(-e.deltaY * 0.001); draw(); };
        c.onmousedown = e => {
            if (!AGENTS) return;
            const r = c.getBoundingClientRect();
            const wx = (e.clientX - r.left - c.width/2)/cam.z + cam.x;
            const wy = (e.clientY - r.top - c.height/2)/-cam.z + cam.y;
            let hit = null;
            const idx = Math.floor(step);
            if (AGENTS[idx] && !isEgoCam) {
                for (let a of AGENTS[idx]) {
                    if (Math.sqrt((wx-a.x)**2 + (wy-a.y)**2) < Math.max(a.l, 3.0)) { hit = a.id; break; }
                }
            }
            if (hit !== null) { followedId = hit; cam.drag = false; }
            else { followedId = null; isEgoCam = false; cam.drag = true; cam.lx = e.clientX; cam.ly = e.clientY; }
            updateUI(); draw();
        };
        window.onmouseup = () => cam.drag = false;
        c.onmousemove = e => { if (cam.drag && !isEgoCam) { cam.x -= (e.clientX-cam.lx)/cam.z; cam.y -= (e.clientY-cam.ly)/-cam.z; cam.lx = e.clientX; cam.ly = e.clientY; draw(); }};

        function drawObs(frame) {
            const zoomLevel = 2.2;
            const bScale = (obsC.width / 2) * zoomLevel;
            const px = dpr / bScale;

            obsCtx.fillStyle = "#ffffff"; obsCtx.fillRect(0, 0, obsC.width, obsC.height);
            obsCtx.save(); obsCtx.translate(obsC.width/2, obsC.height/2); obsCtx.scale(bScale, -bScale);

            obsCtx.lineCap = "round"; obsCtx.strokeStyle = "#bbb"; obsCtx.lineWidth = 1.5 * px;
            if(frame.lanes) frame.lanes.forEach(r => {
                obsCtx.beginPath(); obsCtx.moveTo(r[0] + r[4]*r[2]/2, r[1] + r[5]*r[2]/2);
                obsCtx.lineTo(r[0] - r[4]*r[2]/2, r[1] - r[5]*r[2]/2); obsCtx.stroke();
            });

            obsCtx.strokeStyle = "#333"; obsCtx.lineWidth = 3 * px;
            if(frame.bounds) frame.bounds.forEach(r => {
                obsCtx.beginPath(); obsCtx.moveTo(r[0] + r[4]*r[2]/2, r[1] + r[5]*r[2]/2);
                obsCtx.lineTo(r[0] - r[4]*r[2]/2, r[1] - r[5]*r[2]/2); obsCtx.stroke();
            });

            if (frame.gps && frame.gps.length > 0) {
                const tType = META.target_type || "static";
                frame.gps.forEach((g, i) => {
                    if (g[0] === 0 && g[1] === 0) return;
                    let color, isStar = false, r;
                    if (tType === "static") {
                        color = (i === 0) ? "red" : "orange";
                        isStar = (i === 0);
                        r = (i === 0) ? 8 * px : 4 * px;
                    } else {
                        color = "magenta";
                        r = 5 * px;
                    }

                    obsCtx.fillStyle = color;
                    obsCtx.beginPath();

                    if (isStar) {
                        const spikes = 5, outerRadius = r, innerRadius = r / 2;
                        let rot = Math.PI / 2 * 3;
                        let x = g[0], y = g[1];
                        let step = Math.PI / spikes;
                        obsCtx.moveTo(g[0], g[1] - outerRadius);
                        for (let k = 0; k < spikes; k++) {
                            x = g[0] + Math.cos(rot) * outerRadius;
                            y = g[1] + Math.sin(rot) * outerRadius;
                            obsCtx.lineTo(x, y);
                            rot += step;
                            x = g[0] + Math.cos(rot) * innerRadius;
                            y = g[1] + Math.sin(rot) * innerRadius;
                            obsCtx.lineTo(x, y);
                            rot += step;
                        }
                        obsCtx.closePath();
                    } else {
                        obsCtx.arc(g[0], g[1], r, 0, 2 * Math.PI);
                    }
                    obsCtx.fill();
                });
            }

            if (frame.traffic_controls && frame.traffic_controls.length > 0) {
                const lightColors = {0: "#888888", 1: "#ff0000", 2: "#ffff00", 3: "#00ff00", 4: "#888888"};
                frame.traffic_controls.forEach(t => {
                    if (t.type === "light") {
                        obsCtx.strokeStyle = lightColors[t.state] || "#888";
                        obsCtx.lineWidth = 2.5 * px;
                        obsCtx.beginPath();
                        obsCtx.moveTo(t.x1, t.y1);
                        obsCtx.lineTo(t.x2, t.y2);
                        obsCtx.stroke();
                        return;
                    }

                    obsCtx.strokeStyle = "#000";
                    obsCtx.lineWidth = 4 * px;
                    obsCtx.beginPath();
                    obsCtx.moveTo(t.x1, t.y1);
                    obsCtx.lineTo(t.x2, t.y2);
                    obsCtx.stroke();

                    obsCtx.strokeStyle = t.type === "stop" ? "#cc0000" : "#ffd700";
                    obsCtx.lineWidth = 2.5 * px;
                    obsCtx.setLineDash([6 * px, 4 * px]);
                    obsCtx.beginPath();
                    obsCtx.moveTo(t.x1, t.y1);
                    obsCtx.lineTo(t.x2, t.y2);
                    obsCtx.stroke();
                    obsCtx.setLineDash([]);
                });
            }

            // --- DRAW VEHICLES FIRST ---
            if(frame.partners) frame.partners.forEach(p => {
                obsCtx.save(); obsCtx.translate(p.x, p.y); obsCtx.rotate(p.h);
                obsCtx.fillStyle = "rgba(136, 136, 136, 0.8)";
                obsCtx.strokeStyle = "#333"; obsCtx.lineWidth = 1.5 * px;

                // NO OFFSET NEEDED ANYMORE! C++ already sent the center.
                obsCtx.beginPath(); obsCtx.rect(-p.l/2, -p.w/2, p.l, p.w); obsCtx.fill(); obsCtx.stroke();
                obsCtx.restore();
            });

            if (frame.ego) {
                obsCtx.save();
                if (typeof HEAD_NORTH !== 'undefined' && HEAD_NORTH) {
                    obsCtx.rotate(Math.PI / 2);
                }
                obsCtx.fillStyle = "rgba(0, 102, 255, 0.8)"; // Semi-transparent blue
                obsCtx.strokeStyle = "#000"; obsCtx.lineWidth = 1.5 * px;
                obsCtx.beginPath(); obsCtx.rect(-frame.ego.l/2, -frame.ego.w/2, frame.ego.l, frame.ego.w); obsCtx.fill(); obsCtx.stroke();
                obsCtx.restore();
            }

            obsCtx.restore();
        }

        function renderPufferMetrics(agent) {
            const pufferScoreWrap = document.getElementById('puffer-score-wrap');
            const pufferBlock = document.getElementById('puffer-block');
            const pufferGrid = document.getElementById('puffer-grid');
            const pufferMetrics = agent && agent.pf ? agent.pf : null;
            const hasScore = agent && typeof agent.ps === "number";

            if (!pufferMetrics && !hasScore) {
                pufferScoreWrap.style.display = "none";
                pufferBlock.style.display = "none";
                pufferGrid.innerHTML = "";
                return;
            }

            const livePufferScore = pufferMetrics && pufferMetrics.score !== undefined ? pufferMetrics.score : agent.ps;
            document.getElementById('tel-ps').innerText = Number(livePufferScore).toFixed(3);
            pufferScoreWrap.style.display = "block";

            if (!pufferMetrics) {
                pufferGrid.innerHTML = "";
                pufferBlock.style.display = "none";
                return;
            }

            const renderGroup = (label, list) => {
                const items = list.map(([key, name]) => {
                    const value = pufferMetrics[key];
                    const display = (value === undefined || value === null) ? "-" : Number(value).toFixed(3);
                    return `
                        <div class="puffer-item">
                            <span class="puffer-name">${name}</span>
                            <span class="puffer-val">${display}</span>
                        </div>
                    `;
                }).join('');
                return `
                    <div class="puffer-header">${label}</div>
                    ${items}
                `;
            };

            pufferGrid.innerHTML =
                renderGroup("Multipliers", PUFFER_MULTIPLIERS) +
                renderGroup("Weighted Score Components", PUFFER_WEIGHTED);
            pufferBlock.style.display = "block";
        }

        function updateUI(agent=null) {
            document.getElementById('stepDisplay').innerText = Math.floor(step);
            document.getElementById('sld').value = Math.floor(step);
            const hudTel = document.getElementById('hud-telemetry');
            const obsCont = document.getElementById('obs-container');

            if(followedId !== null && agent) {
                document.getElementById('camMode').innerText = isEgoCam ? "LOCKED (EGO)" : "LOCKED (WORLD)";
                hudTel.style.display = "block";
                document.getElementById('tel-id').innerText = agent.id;
                document.getElementById('tel-speed').innerText = (agent.s * 3.6).toFixed(1);
                document.getElementById('tel-st').innerText = (((agent.st ?? 0) * 180) / Math.PI).toFixed(1);
                document.getElementById('tel-al').innerText = agent.al.toFixed(2);
                document.getElementById('tel-alat').innerText = agent.alat.toFixed(2);
                document.getElementById('tel-jl').innerText = (agent.jl ?? 0).toFixed(2);
                document.getElementById('tel-jlat').innerText = (agent.jlat ?? 0).toFixed(2);
                document.getElementById('tel-x').innerText = agent.x.toFixed(1);
                document.getElementById('tel-y').innerText = agent.y.toFixed(1);
                document.getElementById('tel-h').innerText = agent.h.toFixed(3);
                document.getElementById('tel-lane').innerText = String(agent.cl ?? -1);
                document.getElementById('tel-lane-top').innerText = String(agent.cl ?? -1);

                let currentIdx = Math.floor(step);

                const metricsBlock = document.getElementById('metrics-block');
                const mGrid = document.getElementById('metrics-grid');
                if (agent.m) {
                    const metricItems = agent.m.map((val, i) => `
                        <div class="metric-item">
                            <span class="m-name">${METRIC_LABELS[i] || "M"+(i+1)}</span>
                            <span class="m-val">${typeof val === "number" ? val.toFixed(2) : "inf"}</span>
                        </div>
                    `).join('');
                    metricsBlock.style.display = "block";
                    mGrid.innerHTML = metricItems;
                } else {
                    metricsBlock.style.display = "none";
                    mGrid.innerHTML = "";
                }

                renderPufferMetrics(agent);

                const policyBlock = document.getElementById('policy-block');
                const policyGrid = document.getElementById('policy-grid');
                const currentPolicy = POLICY[currentIdx] ? POLICY[currentIdx][agent.id] : null;

                if (currentPolicy && currentPolicy.labels && currentPolicy.scaled && currentPolicy.raw) {
                    const policyItems = [];
                    if (typeof currentPolicy.value === "number") {
                        policyItems.push(`
                        <div class="policy-item">
                            <span class="policy-name">value</span>
                            <div class="policy-values">
                                <span class="policy-scaled">${currentPolicy.value.toFixed(3)}</span>
                                <span class="policy-raw">critic</span>
                            </div>
                        </div>
                    `);
                    }
                    if (typeof currentPolicy.entropy === "number") {
                        policyItems.push(`
                        <div class="policy-item">
                            <span class="policy-name">entropy</span>
                            <div class="policy-values">
                                <span class="policy-scaled">${currentPolicy.entropy.toFixed(3)}</span>
                                <span class="policy-raw">policy</span>
                            </div>
                        </div>
                    `);
                    }
                    policyItems.push(...currentPolicy.labels.map((label, i) => `
                        <div class="policy-item">
                            <span class="policy-name">${label}</span>
                            <div class="policy-values">
                                <span class="policy-scaled">${Number(currentPolicy.scaled[i]).toFixed(2)}</span>
                                <span class="policy-raw">${Number(currentPolicy.raw[i]).toFixed(2)}</span>
                            </div>
                        </div>
                    `));
                    if (currentPolicy.density) {
                        policyItems.push(...currentPolicy.density.labels.map((label, i) => `
                            <div class="policy-item">
                                <span class="policy-name">mean ${label}</span>
                                <div class="policy-values">
                                    <span class="policy-scaled">${Number(currentPolicy.density.mean[i]).toFixed(3)}</span>
                                    <span class="policy-raw">mu</span>
                                </div>
                            </div>
                            <div class="policy-item">
                                <span class="policy-name">std ${label}</span>
                                <div class="policy-values">
                                    <span class="policy-scaled">${Number(currentPolicy.density.std[i]).toFixed(3)}</span>
                                    <span class="policy-raw">sigma</span>
                                </div>
                            </div>
                        `));
                        policyItems.push(`
                            <div class="policy-item">
                                <span class="policy-name">log_prob(action)</span>
                                <div class="policy-values">
                                    <span class="policy-scaled">${Number(currentPolicy.density.log_prob).toFixed(3)}</span>
                                    <span class="policy-raw">density</span>
                                </div>
                            </div>
                        `);
                    }
                    if (Array.isArray(currentPolicy.action_probs)) {
                        const rows = currentPolicy.action_probs.map((row) => {
                            const probability = Math.max(0, Math.min(1, Number(row.probability) || 0));
                            return `
                                <div class="policy-row ${row.selected ? "selected" : ""}">
                                    <div class="policy-bar" style="width: ${(probability * 100).toFixed(2)}%;"></div>
                                    <div class="policy-row-text">
                                        <span>${row.selected ? "> " : ""}${row.label}</span>
                                        <span>${(probability * 100).toFixed(2)}%</span>
                                    </div>
                                </div>
                            `;
                        }).join('');
                        policyItems.push(`<div class="policy-hist">${rows}</div>`);
                    }
                    policyGrid.innerHTML = policyItems.join('');
                    policyBlock.style.display = "block";
                } else {
                    policyBlock.style.display = "none";
                    policyGrid.innerHTML = "";
                }

                let warnings = [];
                if (agent.m) {
                    if (agent.m[0] === 1) warnings.push("COLLISION");
                    if (agent.m[1] === 1) warnings.push("OFFROAD");
                    if (agent.m[2] === 1) warnings.push("RED LIGHT");
                    if (agent.m[3] === 1) warnings.push("STOP SIGN");
                }

                const crashMsgEl = document.getElementById('crash-msg');
                const crashOverlayEl = document.getElementById('crash-overlay');

                if (warnings.length > 0) {
                    crashOverlayEl.style.display = "block";
                    crashMsgEl.style.display = "block";
                    crashMsgEl.innerHTML = warnings.join("<br>");
                    hudTel.style.borderLeftColor = "red";
                } else {
                    crashOverlayEl.style.display = "none";
                    crashMsgEl.style.display = "none";
                    hudTel.style.borderLeftColor = "var(--accent)";
                }

                if (ALL_OBS[currentIdx] && ALL_OBS[currentIdx][agent.id]) {
                    obsCont.style.display = "block";
                    drawObs(ALL_OBS[currentIdx][agent.id]);
                } else {
                    obsCont.style.display = "none";
                }

            } else {
                document.getElementById('camMode').innerText = "Free Roam";
                hudTel.style.display = "none";
                obsCont.style.display = "none";
                document.getElementById('crash-overlay').style.display = "none";
                document.getElementById('puffer-score-wrap').style.display = "none";
                document.getElementById('puffer-block').style.display = "none";
                document.getElementById('policy-block').style.display = "none";
                document.getElementById('metrics-block').style.display = "none";
                document.getElementById('puffer-grid').innerHTML = "";
                document.getElementById('policy-grid').innerHTML = "";
                document.getElementById('metrics-grid').innerHTML = "";
            }
        }

        function draw() {
            if(!AGENTS) return; // Wait until data unzips

            let idx = Math.floor(step);
            let target = null;
            if(followedId !== null && AGENTS[idx]) target = AGENTS[idx].find(a => a.id === followedId);
            if(target) { cam.x = target.x; cam.y = target.y; }
            updateUI(target);

            const colors = getColors();
            ctx.fillStyle=colors.bg; ctx.fillRect(0,0,c.width,c.height);
            ctx.save();

            ctx.translate(c.width/2, c.height/2);
            ctx.scale(cam.z, -cam.z);

            if (isEgoCam && target) {
                ctx.rotate(Math.PI/2 - target.h);
            }

            ctx.translate(-cam.x, -cam.y);

            ctx.lineCap="round";
            ctx.strokeStyle=colors.road; ctx.lineWidth=0.5; MAP.lanes.forEach(l=>line(l));
            ctx.strokeStyle=colors.line; ctx.setLineDash([1,1]); MAP.lines.forEach(l=>line(l)); ctx.setLineDash([]);
            ctx.strokeStyle=colors.edge; ctx.lineWidth=0.8; MAP.edges.forEach(l=>line(l));

            if(AGENTS[idx]) AGENTS[idx].forEach(a => {
                ctx.save(); ctx.translate(a.x, a.y); ctx.rotate(a.h);
                ctx.fillStyle=a.c; ctx.strokeStyle=darkMode?"#fff":"#111"; ctx.lineWidth=0.1;
                ctx.beginPath(); ctx.rect(-a.l/2, -a.w/2, a.l, a.w); ctx.fill(); ctx.stroke();
                ctx.fillStyle="rgba(255,255,0,0.5)"; ctx.beginPath(); ctx.rect(a.l/2-0.5, -a.w/2, 0.5, a.w); ctx.fill();

                let isBraking = false;
                if(idx > 0 && AGENTS[idx-1]) {
                    let prev = AGENTS[idx-1].find(pa => pa.id === a.id);
                    if(prev && (a.s < prev.s - 0.05)) isBraking = true;
                }
                if(isBraking || a.s < 0.1) {
                    ctx.fillStyle="rgba(255,0,0,0.9)"; ctx.shadowColor="red"; ctx.shadowBlur=10;
                    ctx.beginPath(); ctx.rect(-a.l/2, -a.w/2, 0.2, a.w/3); ctx.fill();
                    ctx.beginPath(); ctx.rect(-a.l/2, a.w/2 - a.w/3, 0.2, a.w/3); ctx.fill();
                    ctx.shadowBlur=0;
                }
                ctx.restore();

                ctx.save(); ctx.translate(a.x, a.y);
                if (isEgoCam && target) {
                    ctx.rotate(-Math.PI/2 + target.h);
                } else {
                    ctx.scale(1, -1);
                }
                ctx.fillStyle = colors.text; ctx.font = "bold " + (14/cam.z) + "px Arial";
                ctx.textAlign="center"; ctx.fillText(a.id, 0, (isEgoCam && target) ? a.w/2 + 0.5 : -a.w/2 - 0.5);
                ctx.restore();

                if(a.id === followedId) {
                    ctx.save(); ctx.translate(a.x, a.y);
                    ctx.strokeStyle = a.c === "red" ? "red" : "#00ff00"; ctx.lineWidth = 4/cam.z;
                    ctx.beginPath(); ctx.arc(0,0,Math.max(a.l,a.w)*1.2, 0, 7); ctx.stroke();
                    ctx.restore();
                }
            });

            if(TRAFFIC[idx]) TRAFFIC[idx].forEach(t => {
                let sl = t.stop_line; if(!sl || sl.length < 6) return;
                ctx.lineCap = "butt";
                if(t.type == 'light') {
                    ctx.strokeStyle = t.c; ctx.lineWidth = Math.max(1.5, 3/cam.z);
                    ctx.beginPath(); ctx.moveTo(sl[0], sl[1]); ctx.lineTo(sl[3], sl[4]); ctx.stroke();
                } else {
                    ctx.strokeStyle = t.c2 || "black"; ctx.lineWidth = Math.max(2, 4/cam.z);
                    ctx.beginPath(); ctx.moveTo(sl[0], sl[1]); ctx.lineTo(sl[3], sl[4]); ctx.stroke();
                    ctx.strokeStyle = t.c; ctx.lineWidth = Math.max(1.2, 2.5/cam.z);
                    ctx.setLineDash([6/cam.z, 4/cam.z]);
                    ctx.beginPath(); ctx.moveTo(sl[0], sl[1]); ctx.lineTo(sl[3], sl[4]); ctx.stroke();
                    ctx.setLineDash([]);
                }
            });
            ctx.restore();
        }

        function line(p){if(p.length<2)return;ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);for(let i=1;i<p.length;i++)ctx.lineTo(p[i][0],p[i][1]);ctx.stroke();}
        function toggle(){ play=!play; lastTick=performance.now(); updateBtn(); if(play) requestAnimationFrame(loop); }
        function updateBtn(){ document.getElementById('btnPlay').innerText=play?"PAUSE":"PLAY"; }
        function changeSpeed() { speed = parseFloat(document.getElementById('speedSel').value); lastTick=performance.now(); }
        function loop(ts){
            if(!play)return;
            const prev = Math.floor(step);
            const dt = Math.min((ts-lastTick)/1000, 0.25);
            lastTick = ts;
            step += dt * speed * 10;
            while(step>=AGENTS.length)step-=AGENTS.length;
            draw();
            requestAnimationFrame(loop);
        }
        document.getElementById('sld').oninput = e => { play=false; updateBtn(); step=+e.target.value; draw(); };
    </script>
</body>
</html>
    """

    # --- 4. PACKAGE, COMPRESS, AND INJECT ---
    master_payload = {
        "map": map_data,
        "agents": agent_history,
        "traffic": traffic_history,
        "traj": trajectory_history,
        "meta": metadata,
        "obs": all_agents_obs_history,
        "policy": policy_history or [],
        "head_north": head_north,
    }

    print("Compressing replay data, this might take a second...")
    compressed_payload = pack_and_compress_data(master_payload, decimals=3)

    try:
        final_html = html_template.replace("__COMPRESSED_PAYLOAD__", compressed_payload)
        with open(filename, "w") as f:
            f.write(final_html)
        print(f"Success! Saved optimized replay to {filename}")
    except Exception as e:
        print(f"Error: {e}")


def build_gallery_index(folder_path="."):
    files = [f for f in os.listdir(folder_path) if f != "index.html" and re.fullmatch(r"(.+)_([0-9]+)\.html", f)]

    if not files:
        print("No matching .html files found in this directory.")
        return

    def sort_key(filename):
        match = re.fullmatch(r"(.+)_([0-9]+)\.html", filename)
        env_map_name = match.group(1)
        global_episode_id = int(match.group(2))
        return (global_episode_id, env_map_name)

    files.sort(key=sort_key)

    # 3. Build the HTML template
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>PufferDrive Replay Gallery</title>
        <style>
            body { margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; font-family: 'Segoe UI', system-ui, sans-serif; background: #111; color: #eee; overflow: hidden; }
            #topbar { padding: 12px 20px; background: #222; display: flex; align-items: center; gap: 15px; border-bottom: 2px solid #007bff; z-index: 100; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            #viewer { flex-grow: 1; border: none; width: 100%; height: 100%; }
            select { padding: 8px 12px; border-radius: 8px; background: #333; color: white; border: 1px solid #555; cursor: pointer; font-weight: bold; font-size: 14px; outline: none;}
            select:focus { border-color: #007bff; }
            button { padding: 8px 16px; border-radius: 8px; background: #007bff; color: white; border: none; cursor: pointer; font-weight: 800; font-size: 13px; text-transform: uppercase; transition: 0.2s;}
            button:hover:not(:disabled) { background: #0056b3; transform: scale(1.05); }
            button:disabled { background: #444; color: #888; cursor: not-allowed; }
            .title { font-weight: 900; font-size: 18px; margin-right: auto; letter-spacing: 1px; color: #fff;}
        </style>
    </head>
    <body>
        <div id="topbar">
            <div class="title">PUFFERDRIVE GALLERY</div>
            <button id="prevBtn" onclick="navigate(-1)">&#9664; Prev</button>
            <select id="fileSelect" onchange="loadSelected()">
                __OPTIONS__
            </select>
            <button id="nextBtn" onclick="navigate(1)">Next &#9654;</button>
        </div>

        <iframe id="viewer" src="__FIRST__"></iframe>

        <script>
            const select = document.getElementById('fileSelect');
            const viewer = document.getElementById('viewer');
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');

            function loadSelected() {
                viewer.src = select.value;
                updateButtons();
            }

            function navigate(dir) {
                let newIdx = select.selectedIndex + dir;
                if (newIdx >= 0 && newIdx < select.options.length) {
                    select.selectedIndex = newIdx;
                    loadSelected();
                }
            }

            function updateButtons() {
                prevBtn.disabled = select.selectedIndex === 0;
                nextBtn.disabled = select.selectedIndex === select.options.length - 1;

                // Return focus to the iframe so your Spacebar/Arrow keys still work!
                viewer.onload = () => viewer.contentWindow.focus();
            }

            updateButtons();
        </script>
    </body>
    </html>
    """

    # 4. Inject the options into the dropdown
    options_html = "\n".join(
        [f'<option value="{f}">{f.replace(".html", "").replace("_", " ")}</option>' for f in files]
    )

    final_html = html_content.replace("__OPTIONS__", options_html).replace("__FIRST__", files[0])

    # 5. Save the file
    index_path = os.path.join(folder_path, "index.html")
    with open(index_path, "w") as f:
        f.write(final_html)
