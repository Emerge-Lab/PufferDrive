"""Bird's Eye View visualization for PufferDrive scenarios using Matplotlib."""

import dataclasses
import weakref
from typing import Optional, Tuple

import math
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
from matplotlib.patches import Circle, Polygon
import os
import json
import zlib
import base64

from pufferlib.ocean.drive import binding


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
    0: "#808080",
    1: "#FF0000",
    2: "#FFFF00",
    3: "#00FF00",
    4: "#FF0000",
    5: "#FFFF00",
    6: "#00FF00",
    7: "#FF6600",
    8: "#FFFF00",
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
        t_type = elem.get("type", 1)
        sl = elem.get("stop_line")
        if sl is None or len(sl) < 4:
            continue
        if t_type == 1:
            traffic_lights.append({"stop_line": sl, "states": elem.get("states", [])})
        elif t_type == 2:
            stop_signs.append(sl)
        elif t_type == 3:
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
        color = TRAFFIC_LIGHT_COLORS.get(state, "#808080")
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


def _render_agents(ax, agents, active_indices, static_indices, config, px_per_meter, use_rear_axle=False):
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
        current_lane_id = agent.get("current_lane_index", -1)
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
    use_rear_axle: bool = False,
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
        use_rear_axle,
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
    dynamics_model: int = 0,
    target_type: str = "static",
    reward_conditioning: bool = False,
    num_target_waypoints: int = 5,
    max_partners: int = 16,
    max_lane_segments: int = 16,
    max_boundary_segments: int = 16,
    max_traffic_lights: int = 16,
    max_stop_signs: int = 10,
    agent_idx: int = 0,
):
    """
    Unpack the flattened observation into the ego state and visible state.
    Args:
        obs_flat: flattened observation tensor of shape (batch_size, obs_dim)
        dynamics_model: 0 for CLASSIC, 1 for JERK
        target_type: 0 for goal only, 1 for waypoints only, 2 for both
    Return:
        ego_state, partners_obs, lane_obs, boundary_obs, traffic_obs, gps_obs, include_goal, include_waypoints
    """
    ego_dim = binding.EGO_FEATURES_JERK if dynamics_model == "jerk" else binding.EGO_FEATURES_CLASSIC

    # Partner obs
    partner_feature_size = binding.PARTNER_FEATURES
    # Road obs
    road_feature_size = binding.ROAD_FEATURES
    # Traffic light obs
    traffic_feature_size = binding.TRAFFIC_LIGHT_FEATURES
    # Stop sign obs
    stop_sign_feature_size = binding.STOP_SIGN_FEATURES

    # Target obs
    target_features = binding.STATIC_TARGET_FEATURES if target_type == "static" else binding.DYNAMIC_TARGET_FEATURES
    target_dim = num_target_waypoints * target_features

    if max_stop_signs > 0:
        ego_dim += 1

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
    lane_end = lane_start + max_lane_segments * road_feature_size
    lane_obs = obs_flat[:, lane_start:lane_end]
    lane_obs = lane_obs.reshape(-1, max_lane_segments, road_feature_size)

    # Extract boundary elements
    boundary_start = lane_end
    boundary_end = boundary_start + max_boundary_segments * road_feature_size
    boundary_obs = obs_flat[:, boundary_start:boundary_end]
    boundary_obs = boundary_obs.reshape(-1, max_boundary_segments, road_feature_size)

    # Extract traffic lights
    traffic_start = boundary_end
    traffic_end = traffic_start + max_traffic_lights * traffic_feature_size
    if max_traffic_lights > 0:
        traffic_obs = obs_flat[:, traffic_start:traffic_end]
        traffic_obs = traffic_obs.reshape(-1, max_traffic_lights, traffic_feature_size)
    else:
        traffic_obs = np.zeros((obs_flat.shape[0], 0, traffic_feature_size))

    # Extract stop signs
    stop_sign_start = traffic_end
    stop_sign_end = stop_sign_start + max_stop_signs * stop_sign_feature_size
    if max_stop_signs > 0:
        stop_sign_obs = obs_flat[:, stop_sign_start:stop_sign_end]
        stop_sign_obs = stop_sign_obs.reshape(-1, max_stop_signs, stop_sign_feature_size)
    else:
        stop_sign_obs = np.zeros((obs_flat.shape[0], 0, stop_sign_feature_size))

    return (
        ego_state[agent_idx],
        target_obs[agent_idx],
        partners_obs[agent_idx],
        lane_obs[agent_idx],
        boundary_obs[agent_idx],
        traffic_obs[agent_idx],
        stop_sign_obs[agent_idx],
    )


def plot_observation(
    obs,
    dynamics_model="classic",
    target_type="static",
    reward_conditioning=False,
    num_target_waypoints=10,
    max_partners=16,
    max_lane_segments=32,
    max_boundary_segments=32,
    max_traffic_lights=4,
    max_stop_signs=4,
    agent_idx=0,
    use_rear_axle=False,
) -> np.ndarray:
    """Plot observation in ego-centric frame.

    Args:
        obs: flattened observation tensor
        dynamics_model: 0 for CLASSIC, 1 for JERK
        target_type: 0 for goal only, 1 for waypoints only, 2 for both
    """
    fig, ax = plt.subplots(figsize=(20, 20))

    ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_obs, stop_sign_obs = unpack_obs(
        obs,
        dynamics_model,
        target_type,
        reward_conditioning,
        num_target_waypoints,
        max_partners,
        max_lane_segments,
        max_boundary_segments,
        max_traffic_lights,
        max_stop_signs,
        agent_idx,
    )

    if dynamics_model == "jerk":
        ego_speed, ego_width, ego_length, steering_angle, a_long, a_lat, lcenter, lalign, speed_limit = ego_state
    else:
        ego_speed, ego_width, ego_length, lcenter, lalign, speed_limit = ego_state

    # Ego vehicle at origin
    ax.add_patch(
        mpatches.Rectangle(
            (-ego_length / 2, -ego_width / 2),
            ego_length,
            ego_width,
            facecolor="blue",
            edgecolor="black",
            linewidth=2,
            alpha=0.7,
            zorder=10,
        )
    )

    # Draw target waypoints
    for i in range(target_obs.shape[0]):
        if np.all(target_obs[i] == 0):
            continue
        wp_x, wp_y = target_obs[i][0], target_obs[i][1]
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

    if dynamics_model == "jerk":
        ego_info += f"\nSteering: {steering_angle:.3f}\na_long: {a_long:.2f}\na_lat: {a_lat:.2f}"

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
        rel_x, rel_y, width, length, heading_cos, heading_sin, speed = partners_obs[i]
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
    count_lane = 0
    for i in range(lane_obs.shape[0]):
        if np.all(lane_obs[i] == 0):
            continue
        count_lane += 1
        rel_x, rel_y, length, width, dir_cos, dir_sin = lane_obs[i]
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
        rel_x, rel_y, length, width, dir_cos, dir_sin = boundary_obs[i]
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

    # Traffic lights (stop lines)
    for i in range(traffic_obs.shape[0]):
        if np.all(traffic_obs[i] == 0):
            continue
        rel_x1, rel_y1, rel_x2, rel_y2, state_normalized = traffic_obs[i]

        if state_normalized == 0:
            state = 4
        elif state_normalized == 1:
            state = 2
        elif state_normalized == 2:
            state = 6
        else:
            state = 0

        ax.plot(
            [rel_x1, rel_x2],
            [rel_y1, rel_y2],
            color=TRAFFIC_LIGHT_COLORS[state],
            linewidth=2.5,
            solid_capstyle="round",
            alpha=0.9,
            zorder=12,
        )

    # Stop signs
    for i in range(stop_sign_obs.shape[0]):
        if np.all(stop_sign_obs[i] == 0):
            continue
        rel_x, rel_y, _ = stop_sign_obs[i]

        radius = 0.02
        angles = np.linspace(0, 2 * np.pi, 9)
        octagon_x = rel_x + radius * np.cos(angles + np.pi / 8)
        octagon_y = rel_y + radius * np.sin(angles + np.pi / 8)
        octagon_points = np.column_stack((octagon_x, octagon_y))
        ax.add_patch(
            plt.Polygon(
                xy=octagon_points,
                alpha=0.9,
                facecolor=COLORS.get("stop_sign", "#808080"),
                edgecolor="red",
                linewidth=1,
                zorder=12,
            )
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

    # Actions
    if use_trajectory:
        raw_actions = scenario.get("ctrl_trajectory_actions", [])
    else:
        raw_actions = scenario.get("actions", [])
    action_map = {}
    if raw_actions and len(raw_actions) == len(active_indices):
        for i, agent_idx in enumerate(active_indices):
            action_map[agent_idx] = raw_actions[i]

    for idx, agent in enumerate(scenario.get("agents", [])):
        if not agent.get("sim_valid"):
            continue

        agent_id = agent.get("id", idx)
        is_active = idx in active_indices

        # Couleur
        if agent.get("stopped", False):
            color = "red"
        else:
            color = get_agent_color(agent_id, is_active)
        req_acc = float(action_map[idx][0]) if idx in action_map else 0.0
        req_str = float(action_map[idx][1]) if idx in action_map else 0.0

        # On arrondit tout pour alléger le JSON final
        current_agents_data.append(
            {
                "id": int(agent_id),
                "x": round(float(agent["sim_x"]), 2),
                "y": round(float(agent["sim_y"]), 2),
                "h": round(float(agent["sim_heading"]), 3),
                "l": round(float(agent["sim_length"]), 2),
                "w": round(float(agent["sim_width"]), 2),
                "s": round(float(agent.get("sim_speed", 0)), 2),
                "st": round(float(agent.get("sim_steering", 0)), 3),
                "c": color,
                # Commandes
                "ra": round(req_acc, 2),
                "rs": round(req_str, 2),
                # Compact metrics array (M1..M18)
                "m": [round(float(m), 2) for m in agent.get("metrics_array")],
            }
        )

    return current_agents_data


def fill_traffics_state(scenario, timestep):
    current_traffic_data = []
    traffic_elements = scenario.get("traffic_elements", [])
    for elem in traffic_elements or []:
        if not isinstance(elem, dict):
            continue

        t_type = elem.get("type", 1)
        sl = elem.get("stop_line")
        if sl is None or len(sl) < 4:
            continue

        if t_type == 1:
            states = elem.get("states", [])
            state = int(states[timestep]) if states and len(states) > timestep else 0
            color = TRAFFIC_LIGHT_COLORS.get(state, "#808080")
            current_traffic_data.append({"type": "light", "stop_line": sl, "c": color})
        elif t_type == 2:
            current_traffic_data.append({"type": "stop", "stop_line": sl, "c": "#FF0000", "c2": "#000000"})
        elif t_type == 3:
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


def extract_obs_frame(obs, scenario, args, timestep, obs_index=0, agent_idx=0, head_north=False):
    ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_obs, stop_sign_obs = unpack_obs(
        obs,
        dynamics_model=args["env"]["dynamics_model"],
        target_type=args["env"]["target_type"],
        reward_conditioning=args["env"]["reward_conditioning"],
        num_target_waypoints=args["env"]["num_target_waypoints"],
        max_partners=args["env"]["max_partner_observations"],
        max_lane_segments=args["env"]["max_lane_segment_observations"],
        max_boundary_segments=args["env"]["max_boundary_segment_observations"],
        max_traffic_lights=args["env"]["max_traffic_light_observations"],
        max_stop_signs=args["env"]["max_stop_sign_observations"],
        agent_idx=obs_index,
    )

    # --- Rotation Helper ---
    def _rot(x, y):
        """Rotates coordinates 90 degrees CCW if head_north is True."""
        return (-y, x) if head_north else (x, y)

    # --- Parse Ego ---
    if args["env"]["dynamics_model"] == "jerk":
        ego_speed, ego_width, ego_length, steering_angle, a_long, a_lat = ego_state[:6]
    else:
        ego_speed, ego_width, ego_length = ego_state[:3]
        steering_angle, a_long, a_lat = 0.0, 0.0, 0.0

    ego_data = {
        "s": round(float(ego_speed), 3),
        "w": round(float(ego_width), 3),
        "l": round(float(ego_length), 3),
        "st": round(float(steering_angle), 3),
        "al": round(float(a_long), 3),
        "alat": round(float(a_lat), 3),
    }

    # --- Parse Road Segments ---
    def parse_roads(roads):
        res = []
        for r in roads:
            if np.all(r == 0):
                continue
            x, y = r[0], r[1]
            length, width = r[2], r[3]
            cos_a, sin_a = r[4], r[5]
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
        h = math.atan2(p[5], p[4])

        if head_north:
            h += math.pi / 2
            h = (h + math.pi) % (2 * math.pi) - math.pi

        parsed_partners.append(
            {
                "x": round(float(px), 3),
                "y": round(float(py), 3),
                "w": round(float(p[2]), 3),
                "l": round(float(p[3]), 3),
                "h": round(float(h), 3),
                "s": round(float(p[6]), 3),
            }
        )

    # --- Parse Traffic Lights ---
    parsed_lights = []
    for t in traffic_obs:
        if np.all(t == 0):
            continue
        lx, ly = _rot(t[0], t[1])
        parsed_lights.append({"x": round(float(lx), 3), "y": round(float(ly), 3), "state": int(t[-1])})

    # --- Parse Stop Signs ---
    parsed_stops = []
    for s in stop_sign_obs:
        if np.all(s == 0):
            continue
        sx, sy = _rot(s[0], s[1])
        parsed_stops.append({"x": round(float(sx), 3), "y": round(float(sy), 3)})

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
        gx, gy = _rot(g[0], g[1])
        gps_data.append([round(float(gx), 3), round(float(gy), 3)])

    return {
        "ego": ego_data,
        "partners": parsed_partners,
        "lanes": parse_roads(lane_obs),
        "bounds": parse_roads(boundary_obs),
        "lights": parsed_lights,
        "stops": parsed_stops,
        "traj": traj_data,
        "gps": gps_data,
    }


def generate_interactive_replay(
    scenario,
    agent_history,
    traffic_history,
    trajectory_history,
    all_agents_obs_history,
    filename="replay.html",
    head_north=False,
    use_rear_axle=False,
):
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

    # --- 1. METADATA ---
    raw_dyn = scenario.get("dynamics_model", 0)
    dyn_str = "Jerk" if raw_dyn == 1 else "Classic"

    metadata = {
        "map_name": scenario.get("map_name", "Unknown"),
        "scenario_id": scenario.get("scenario_id", "Unknown"),
        "dynamics_model": dyn_str,
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

        #hud-telemetry { position: absolute; top: 80px; right: 20px; width: 340px; display: none; border-left: 6px solid var(--accent); background: rgba(15, 15, 15, 0.98); color: white; z-index: 20; }

        #tel-drag-handle { margin: 0 0 12px 0; font-size: 16px; text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 1px solid #444; padding-bottom: 6px; color: #eee; cursor: grab; }
        #tel-drag-handle:active { cursor: grabbing; }

        #controls { position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%); padding: 12px 30px; border-radius: 50px; display: flex; gap: 20px; align-items: center; z-index: 20; }

        #search-box { position: absolute; bottom: 110px; right: 20px; display: flex; gap: 8px; align-items: center; pointer-events: auto; z-index: 20;}

        /* PiP OBSERVATION CONTAINER */
        #obs-container {
            position: absolute; bottom: 30px; left: 20px; width: 400px; height: 400px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 16px; box-shadow: 0 8px 30px var(--shadow);
            display: none; border: 3px solid var(--accent); overflow: hidden; pointer-events: auto;
            backdrop-filter: blur(5px); z-index: 20;
        }
        #obs-canvas { width: 100%; height: 100%; display: block; }
        #obs-title { position: absolute; top: 0; left: 0; width: 100%; padding: 8px 12px; font-size: 11px; font-weight: 900; color: #fff; background: var(--accent); z-index: 2; letter-spacing: 1px; cursor: grab; box-sizing: border-box;}
        #obs-title:active { cursor: grabbing; }

        h3 { margin: 0 0 12px 0; font-size: 16px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--hud-label); border-bottom: 1px solid #444; padding-bottom: 6px; }
        .label { font-size: 12px; color: #888; margin-top: 12px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
        .value { font-size: 18px; font-weight: 800; color: var(--hud-val); }
        .highlight { color: var(--accent); cursor: pointer; }
        .highlight:hover { filter: brightness(1.3); text-decoration: underline; }

        .val-speed { font-size: 36px; color: #fff; font-family: 'Courier New', monospace; }
        .val-action { font-size: 24px; color: var(--accent); font-family: 'Courier New', monospace; }

        .sparkline { width: 100%; height: 40px; margin-top: 8px; background: rgba(0,0,0,0.4); border-radius: 4px; display: block; }

        #metrics-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 12px; background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); }
        .metric-item { display: flex; flex-direction: column; border-bottom: 1px solid #333; padding-bottom: 6px; }
        .m-name { color: #aaa; font-weight: bold; font-size: 11px; text-transform: uppercase; margin-bottom: 2px; }
        .m-val { color: #00ff88; font-family: 'Courier New', monospace; font-weight: 900; font-size: 18px; }

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
        <div id="hud-global" class="panel">
            <h3>Scenario Info</h3>
            <div class="label">Map</div> <div class="value" id="meta-map">-</div>
            <div class="label">ID</div> <div class="value small-val" id="meta-id" style="font-size:12px">-</div>
            <hr style="border: 0; border-top: 1px solid #555; margin: 12px 0;">
            <div class="label">Step</div> <div class="value" style="font-size: 32px; color:var(--accent)" id="stepDisplay">0</div>
            <div class="label">Camera Mode</div>
            <div class="value highlight" id="camMode" onclick="toggleCamMode()" title="Click to Toggle World/Ego">Free Roam</div>
            <button onclick="toggleTheme()" style="width:100%; margin-top:15px; font-size:11px">🌙 THEME</button>
        </div>

        <div id="hud-telemetry" class="panel">
            <div id="crash-msg">⚠ COLLISION ⚠</div>
            <h3 id="tel-drag-handle">☰ DRAG | Agent <span id="tel-id" style="color:var(--accent)">?</span></h3>

            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="label" style="margin-top:0;">Speed</div>
                    <div><span class="val-speed" id="tel-speed">0.0</span> <span style="font-size:14px; color:#888">km/h</span></div>
                </div>
                <div style="text-align: right;">
                    <div class="label" style="margin-top:0;">Req Acc/Str</div>
                    <div class="val-action" style="margin-top:5px;"><span id="tel-ra">0.0</span> <span style="color:#444">/</span> <span id="tel-rs">0.0</span></div>
                </div>
            </div>

            <canvas id="spark-speed" class="sparkline"></canvas>

            <div class="label" style="margin-top: 15px; color: var(--accent); border-bottom: 1px solid #333">Metrics Table</div>
            <div id="metrics-grid"></div>

            <div class="label" style="margin-top: 15px;">Position (X/Y)</div>
            <div style="font-family: monospace; font-size: 15px; color: #ccc; font-weight: bold;"><span id="tel-x">0</span> , <span id="tel-y">0</span></div>
        </div>

        <div id="obs-container">
            <div id="obs-title">☰ DRAG TO MOVE | EGO-CENTRIC NN OBS</div>
            <canvas id="obs-canvas"></canvas>
        </div>

        <div id="search-box">
             <input type="number" id="agentSearch" placeholder="ID" onkeydown="if(event.key==='Enter') searchAgent()">
             <button onclick="searchAgent()" class="panel" style="border-radius:15px; padding: 12px 18px;">🔍</button>
        </div>

        <div id="controls" class="panel">
            <button id="btnPlay" onclick="toggle()" style="min-width: 100px; font-size: 16px;">PLAY</button>
            <select id="speedSel" onchange="changeSpeed()">
                <option value="0.1">0.1x</option>
                <option value="0.25" selected>0.25x</option>
                <option value="0.5" selected>0.5x</option>
                <option value="1.0">1.0x</option>
            </select>
            <input id="sld" type="range" min="0" value="0" step="1">
        </div>
    </div>

    <canvas id="c"></canvas>

    <script>
        // Payload Placeholder
        const B64_PAYLOAD = "__COMPRESSED_PAYLOAD__";

        // Globals (populated after decompression)
        let MAP, AGENTS, TRAFFIC, TRAJ, META, ALL_OBS, HEAD_NORTH, USE_REAR_AXLE;

        const c=document.getElementById('c'), ctx=c.getContext('2d');
        const obsC = document.getElementById('obs-canvas'), obsCtx = obsC.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        obsC.width = 400 * dpr; obsC.height = 400 * dpr;

        const sparkSpeed = document.getElementById('spark-speed');
        sparkSpeed.width = sparkSpeed.clientWidth * dpr; sparkSpeed.height = sparkSpeed.clientHeight * dpr;

        let step=0, play=false, speed=0.5;
        let cam={x:0, y:0, z:5, drag:false, lx:0, ly:0};
        let followedId = null, darkMode = false, isEgoCam = false;

        const METRIC_LABELS = ["collision", "offroad", "red_light", "stop_sign", "reached_goal", "lane_dist", "lane_angle", "comfort_violation", "velocity_progress", "speed_limit", "ADE", "progression", "at_fault_collision", "ttc", "ttc_tfl", "progress_ratio", "multi_lane_time", "multi_lane_score"];

        // --- Data Unpacking ---
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
                HEAD_NORTH = data.head_north;
                USE_REAR_AXLE = data.use_rear_axle;

                document.getElementById('meta-map').innerText = META.map_name.split('binaries/')[1] || META.map_name;
                document.getElementById('meta-id').innerText = META.scenario_id;

                if(AGENTS[0]?.length) { cam.x=AGENTS[0][0].x; cam.y=AGENTS[0][0].y; }
                document.getElementById('sld').max = AGENTS.length-1;

                document.getElementById('loading-overlay').style.display = 'none';
                window.onresize(); // Initial draw

            } catch (err) {
                console.error("Failed to unpack replay data:", err);
                document.getElementById('loading-overlay').innerText = "Error loading replay data. Check console.";
            }
        }

        initReplay();

        // --- Data Unpacking ---
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
                HEAD_NORTH = data.head_north;
                USE_REAR_AXLE = data.use_rear_axle;

                document.getElementById('meta-map').innerText = META.map_name.split('binaries/')[1] || META.map_name;
                document.getElementById('meta-id').innerText = META.scenario_id;

                if(AGENTS[0]?.length) { cam.x=AGENTS[0][0].x; cam.y=AGENTS[0][0].y; }
                document.getElementById('sld').max = AGENTS.length-1;

                document.getElementById('loading-overlay').style.display = 'none';
                window.onresize(); // Initial draw

            } catch (err) {
                console.error("Failed to unpack replay data:", err);
                document.getElementById('loading-overlay').innerText = "Error loading replay data. Check console.";
            }
        }

        initReplay();

        // --- Draggable PiP & Telemetry Window Logic ---
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
                bg: style.getPropertyValue('--bg').trim(), road: style.getPropertyValue('--road').trim(),
                line: style.getPropertyValue('--line').trim(), edge: style.getPropertyValue('--edge').trim(),
                text: style.getPropertyValue('--text').trim()
            };
        }

        window.onresize = () => { c.width=window.innerWidth; c.height=window.innerHeight; draw(); };

        function toggleTheme() { darkMode = !darkMode; document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light'); draw(); }

        function toggleCamMode() {
            if(followedId !== null) {
                isEgoCam = !isEgoCam;
                updateUI(AGENTS[Math.floor(step)].find(a => a.id === followedId));
                draw();
            }
        }

        function searchAgent() {
            if(!AGENTS) return;
            const id = parseInt(document.getElementById('agentSearch').value);
            if(!isNaN(id)) {
                followedId = id; play = false; updateBtn(); draw();
            }
        }

        document.addEventListener('keydown', (e) => {
            if(!AGENTS) return;
            if (e.target.tagName === 'INPUT') return;
            if (e.code === "Space") { toggle(); e.preventDefault(); }
            if (e.code === "ArrowRight") { play=false; updateBtn(); step=Math.min(step+1, AGENTS.length-1); draw(); }
            if (e.code === "ArrowLeft") { play=false; updateBtn(); step=Math.max(step-1, 0); draw(); }
            if (e.code === "Escape") { followedId = null; isEgoCam = false; updateUI(); draw(); }
        });

        c.onwheel = e => { e.preventDefault(); cam.z *= Math.exp(-e.deltaY*0.001); draw(); };
        c.onmousedown = e => {
            if(!AGENTS) return;
            const r = c.getBoundingClientRect();
            const wx = (e.clientX - r.left - c.width/2)/cam.z + cam.x;
            const wy = (e.clientY - r.top - c.height/2)/-cam.z + cam.y;
            let hit = null;
            const idx = Math.floor(step);
            if(AGENTS[idx] && !isEgoCam) {
                for(let a of AGENTS[idx]) {
                    if(Math.sqrt((wx-a.x)**2 + (wy-a.y)**2) < Math.max(a.l, 3.0)) { hit = a.id; break; }
                }
            }
            if(hit !== null) { followedId = hit; cam.drag = false; }
            else { followedId = null; isEgoCam = false; cam.drag = true; cam.lx=e.clientX; cam.ly=e.clientY; }
            updateUI(); draw();
        };
        window.onmouseup = () => cam.drag=false;
        c.onmousemove = e => { if(cam.drag && !isEgoCam){ cam.x-=(e.clientX-cam.lx)/cam.z; cam.y-=(e.clientY-cam.ly)/-cam.z; cam.lx=e.clientX; cam.ly=e.clientY; draw(); }};

        function drawSparkline(canvasCtx, data, color, minVal, maxVal) {
            const w = canvasCtx.canvas.width;
            const h = canvasCtx.canvas.height;
            canvasCtx.clearRect(0, 0, w, h);
            if(data.length < 2) return;

            canvasCtx.beginPath();
            canvasCtx.strokeStyle = color;
            canvasCtx.lineWidth = 2 * dpr;
            canvasCtx.lineCap = "round";
            canvasCtx.lineJoin = "round";

            const range = maxVal - minVal || 1;
            for(let i=0; i<data.length; i++) {
                const x = (i / (data.length - 1)) * w;
                const y = h - ((data[i] - minVal) / range) * h;
                if(i===0) canvasCtx.moveTo(x, y);
                else canvasCtx.lineTo(x, y);
            }
            canvasCtx.stroke();
            canvasCtx.lineTo(w, h);
            canvasCtx.lineTo(0, h);
            canvasCtx.fillStyle = color + "44";
            canvasCtx.fill();
        }

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

            if (frame.lights && frame.lights.length > 0) {
                const lightColors = {1: "#ff0000", 2: "#ffff00", 3: "#00ff00"};
                obsCtx.lineWidth = 1 * px; obsCtx.strokeStyle = "#000";
                frame.lights.forEach(l => {
                    obsCtx.fillStyle = lightColors[l.state] || "#888";
                    obsCtx.beginPath(); obsCtx.arc(l.x, l.y, 0.012, 0, 7); obsCtx.fill(); obsCtx.stroke();
                });
            }
            if (frame.stops && frame.stops.length > 0) {
                obsCtx.fillStyle = "#cc0000"; obsCtx.strokeStyle = "#fff"; obsCtx.lineWidth = 1 * px;
                frame.stops.forEach(s => {
                    obsCtx.beginPath();
                    for (let i=0; i<8; i++) {
                        const angle = i*Math.PI/4 + Math.PI/8;
                        const ptX = s.x + 0.018 * Math.cos(angle), ptY = s.y + 0.018 * Math.sin(angle);
                        if (i===0) obsCtx.moveTo(ptX, ptY); else obsCtx.lineTo(ptX, ptY);
                    }
                    obsCtx.closePath(); obsCtx.fill(); obsCtx.stroke();
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
                document.getElementById('tel-ra').innerText = agent.ra.toFixed(2);
                document.getElementById('tel-rs').innerText = agent.rs.toFixed(2);
                document.getElementById('tel-x').innerText = agent.x.toFixed(1);
                document.getElementById('tel-y').innerText = agent.y.toFixed(1);

                let histLen = 50;
                let startIdx = Math.max(0, Math.floor(step) - histLen);
                let speedData = [];
                for(let i=startIdx; i<=Math.floor(step); i++) {
                    let pastA = AGENTS[i] ? AGENTS[i].find(a => a.id === agent.id) : null;
                    if(pastA) {
                        speedData.push(pastA.s * 3.6);
                    } else {
                        speedData.push(0);
                    }
                }
                drawSparkline(sparkSpeed.getContext('2d'), speedData, '#00ff88', 0, Math.max(50, ...speedData));

                const mGrid = document.getElementById('metrics-grid');
                if (agent.m) {
                    mGrid.innerHTML = agent.m.map((val, i) => `
                        <div class="metric-item">
                            <span class="m-name">${METRIC_LABELS[i] || "M"+(i+1)}</span>
                            <span class="m-val">${val.toFixed(2)}</span>
                        </div>
                    `).join('');
                }

                let warnings = [];
                if (agent.m) {
                    if (agent.m[0] === 1) warnings.push("⚠ COLLISION ⚠");
                    if (agent.m[1] === 1) warnings.push("⚠ OFFROAD ⚠");
                    if (agent.m[2] === 1) warnings.push("⚠ RED LIGHT ⚠");
                    if (agent.m[3] === 1) warnings.push("⚠ STOP SIGN VIOLATION ⚠");
                }

                const crashMsgEl = document.getElementById('crash-msg');
                const crashOverlayEl = document.getElementById('crash-overlay');

                if(warnings.length > 0) {
                    crashOverlayEl.style.display = "block";
                    crashMsgEl.style.display = "block";
                    crashMsgEl.innerHTML = warnings.join("<br>");
                    hudTel.style.borderLeftColor = "red";
                } else {
                    crashOverlayEl.style.display = "none";
                    crashMsgEl.style.display = "none";
                    hudTel.style.borderLeftColor = "var(--accent)";
                }

                let currentIdx = Math.floor(step);
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
                let sl=t.stop_line; if(!sl) return;
                ctx.lineWidth=Math.max(1.5, 3/cam.z); ctx.lineCap="butt";
                if(t.type=='light') {
                    ctx.strokeStyle=t.c; ctx.beginPath(); ctx.moveTo(sl[0],sl[1]); ctx.lineTo(sl[3],sl[4]); ctx.stroke();
                } else {
                    ctx.strokeStyle=t.c2||"black"; ctx.lineWidth=Math.max(2, 4/cam.z);
                    ctx.beginPath(); ctx.moveTo(sl[0],sl[1]); ctx.lineTo(sl[3],sl[4]); ctx.stroke();
                    ctx.strokeStyle=t.c; ctx.lineWidth=Math.max(1.2, 2.5/cam.z); ctx.setLineDash([6/cam.z,4/cam.z]);
                    ctx.beginPath(); ctx.moveTo(sl[0],sl[1]); ctx.lineTo(sl[3],sl[4]); ctx.stroke();
                    ctx.setLineDash([]);
                }
            });
            ctx.restore();
        }

        function line(p){if(p.length<2)return;ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);for(let i=1;i<p.length;i++)ctx.lineTo(p[i][0],p[i][1]);ctx.stroke();}
        function toggle(){ play=!play; updateBtn(); if(play) loop(); }
        function updateBtn(){ document.getElementById('btnPlay').innerText=play?"PAUSE":"PLAY"; }
        function changeSpeed() { speed = parseFloat(document.getElementById('speedSel').value); }
        function loop(){ if(!play)return; step+=speed; if(step>=AGENTS.length)step=0; draw(); requestAnimationFrame(loop); }
        document.getElementById('sld').oninput = e => { step=+e.target.value; draw(); };
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
        "head_north": head_north,
        "use_rear_axle": use_rear_axle,
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
    # Assuming files still start with "map_" based on your example
    files = [f for f in os.listdir(folder_path) if f.startswith("map_") and f.endswith(".html")]

    if not files:
        print("No matching .html files found in this directory.")
        return

    def sort_key(filename):
        # 1. Strip the '.html' extension
        name_no_ext = filename[:-5]

        # 2. Split from the right exactly once
        # e.g., "map_000_000" -> ["map_000", "000"]
        parts = name_no_ext.rsplit("_", 1)

        env_map_name = parts[0]
        global_episode_id = int(parts[1])

        # 3. Sort first by episode ID, then by map name
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
