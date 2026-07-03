"""Bird's Eye View visualization for PufferDrive scenarios using Matplotlib."""

import dataclasses
from typing import Optional, Tuple


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
    "distance_to_collision",
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
    show_goal: bool = True
    goal_radius: float = 2.0

    def get_bounds(self, scenario) -> Tuple[float, float, float, float]:
        map_corners = scenario.get("map_corners")

        if self.center is not None:
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


def _init_fig_ax(config: VizConfig):
    fig, ax_main = plt.subplots()
    fig.set_size_inches(config.figsize)

    fig.set_dpi(config.dpi)
    fig.set_facecolor(COLORS["background"])
    ax_main.set_facecolor(COLORS["background"])

    return fig, ax_main


def _build_road_data(road_elements):
    lanes, lines, edges = [], [], []
    for elem in road_elements or []:
        if not isinstance(elem, dict):
            continue
        x, y, t = elem.get("x"), elem.get("y"), elem.get("type", 0)
        if not x or not y:
            continue
        pts = np.column_stack((np.asarray(x), np.asarray(y)))
        if 1 <= t <= 3:
            lanes.append(pts)
        elif 11 <= t <= 18:
            lines.append(pts)
        elif 21 <= t <= 23:
            edges.append(pts)
    return {
        "lanes": lanes,
        "lines": lines,
        "edges": edges,
    }


def _render_roads(ax, road_data):
    if not road_data:
        return
    lanes = road_data.get("lanes") or []
    lines = road_data.get("lines") or []
    edges = road_data.get("edges") or []
    if lanes:
        ax.add_collection(LineCollection(lanes, colors=COLORS["lane"], linewidths=0.8, alpha=0.7, zorder=1))
    if lines:
        ax.add_collection(
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
        ax.add_collection(LineCollection(edges, colors=COLORS["road_edge"], linewidths=0.8, alpha=0.8, zorder=2))


def _build_traffic_data(traffic_elements):
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


def _render_traffic(ax, traffic_data, timestep):
    if not traffic_data:
        return
    # Traffic lights — colored by state
    for light in traffic_data.get("traffic_lights", []):
        sl = light["stop_line"]
        states = light["states"]
        state = int(states[timestep]) if states and len(states) > timestep else 0
        color = _traffic_light_color(state)
        ax.plot([sl[0], sl[3]], [sl[1], sl[4]], color=color, linewidth=3, solid_capstyle="butt", alpha=0.9, zorder=15)

    # Stop signs — red/black striped
    for sl in traffic_data.get("stop_signs", []):
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
    for sl in traffic_data.get("yield_signs", []):
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
                gx, gy = agent.get("current_goal_x"), agent.get("current_goal_y")
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


def plot_simulator_state(scenario, timestep: int = 0) -> np.ndarray:
    """Render simulator state to RGB image array."""
    vis_config = VizConfig()
    map_data = {
        "map_name": scenario.get("map_name"),
        "road": _build_road_data(scenario.get("road_elements", [])),
        "traffic": _build_traffic_data(scenario.get("traffic_elements", [])),
    }

    bounds = vis_config.get_bounds(scenario)
    x_min, x_max, y_min, y_max = bounds

    px_per_meter = min(
        vis_config.figsize[0] * vis_config.dpi / (x_max - x_min),
        vis_config.figsize[1] * vis_config.dpi / (y_max - y_min),
    )

    fig, ax = _init_fig_ax(vis_config)

    ax.set_aspect("equal")
    ax.set_title(
        f"PufferDrive | {scenario.get('dataset_name', '')} | {scenario.get('scenario_id', '')} | t={timestep}",
        fontsize=max(14, int(px_per_meter / 8)),
        fontweight="bold",
    )

    _render_roads(ax, map_data.get("road"))
    _render_traffic(ax, map_data.get("traffic"), timestep)

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

    return _img_from_fig(fig)


def _img_from_fig(fig: matplotlib.figure.Figure, close: bool = True) -> np.ndarray:
    fig.subplots_adjust(left=0.01, bottom=0.02, right=1.00, top=0.96)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
    if close:
        plt.close(fig)
    return img


def unpack_obs(
    obs_flat,
    goal_regen_mode: str = "finite",
    reward_conditioning: bool = False,
    num_goals: int = 5,
    obs_slots_partners_n: int = 16,
    obs_slots_lane_n: int = 16,
    obs_slots_boundary_n: int = 16,
    obs_slots_traffic_controls_n: int = 16,
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

    if isinstance(goal_regen_mode, int):
        goal_regen_mode = "finite" if goal_regen_mode == binding.GOAL_REGEN_FINITE else "rolling"

    ego_dim = binding.EGO_FEATURES

    # Partner obs
    partner_feature_size = binding.PARTNER_FEATURES
    # Road obs
    lane_feature_size = binding.LANE_FEATURES
    boundary_feature_size = binding.BOUNDARY_FEATURES
    # Traffic control obs
    traffic_control_feature_size = binding.TRAFFIC_CONTROL_FEATURES
    lane_segment_count = compute_effective_road_obs_count(obs_slots_lane_n, obs_dropout_lane)
    boundary_segment_count = compute_effective_road_obs_count(obs_slots_boundary_n, obs_dropout_boundary)

    # Target obs
    goal_features = binding.GOAL_FEATURES
    goal_dim = num_goals * goal_features

    # Extract ego state
    ego_state = obs_flat[:, :ego_dim]

    target_start = ego_dim
    if reward_conditioning:
        target_start += binding.NUM_REWARD_COEFS

    target_end = target_start + goal_dim
    target_obs = obs_flat[:, target_start:target_end]
    target_obs = target_obs.reshape(-1, num_goals, goal_features)

    # Extract partners
    partners_start = target_end
    partners_end = partners_start + obs_slots_partners_n * partner_feature_size
    partners_obs = obs_flat[:, partners_start:partners_end]
    partners_obs = partners_obs.reshape(-1, obs_slots_partners_n, partner_feature_size)

    # Extract lane elements
    lane_start = partners_end
    lane_end = lane_start + lane_segment_count * lane_feature_size
    lane_obs = obs_flat[:, lane_start:lane_end]
    lane_obs = lane_obs.reshape(-1, lane_segment_count, lane_feature_size)

    # Extract boundary elements
    boundary_start = lane_end
    boundary_end = boundary_start + boundary_segment_count * boundary_feature_size
    boundary_obs = obs_flat[:, boundary_start:boundary_end]
    boundary_obs = boundary_obs.reshape(-1, boundary_segment_count, boundary_feature_size)

    # Extract traffic controls
    traffic_start = boundary_end
    traffic_end = traffic_start + obs_slots_traffic_controls_n * traffic_control_feature_size
    if obs_slots_traffic_controls_n > 0:
        traffic_controls_obs = obs_flat[:, traffic_start:traffic_end]
        traffic_controls_obs = traffic_controls_obs.reshape(
            -1, obs_slots_traffic_controls_n, traffic_control_feature_size
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
    goal_regen_mode="finite",
    reward_conditioning=False,
    num_goals=10,
    obs_slots_partners_n=16,
    obs_slots_lane_n=32,
    obs_slots_boundary_n=32,
    obs_slots_traffic_controls_n=4,
    obs_dropout_lane=0.0,
    obs_dropout_boundary=0.0,
    obs_lane_stride=1,
    obs_boundary_stride=1,
    obs_goal_lane_distance=False,
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
        goal_regen_mode: "finite" or "rolling"
    """
    if isinstance(goal_regen_mode, int):
        goal_regen_mode = "finite" if goal_regen_mode == binding.GOAL_REGEN_FINITE else "rolling"

    fig, ax = plt.subplots(figsize=(20, 20))

    ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_controls_obs = unpack_obs(
        obs,
        goal_regen_mode=goal_regen_mode,
        reward_conditioning=reward_conditioning,
        num_goals=num_goals,
        obs_slots_partners_n=obs_slots_partners_n,
        obs_slots_lane_n=obs_slots_lane_n,
        obs_slots_boundary_n=obs_slots_boundary_n,
        obs_slots_traffic_controls_n=obs_slots_traffic_controls_n,
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
    target_position_scale = scales["goal_to_position"]

    ego_speed, ego_width, ego_length, steering_angle, accel_long, accel_lat, lcenter, lalign, speed_limit, _ = ego_state

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
        color = "red" if i == 0 else "orange"
        marker = "*" if i == 0 else "o"
        s = 200 if i == 0 else 80
        ax.scatter(wp_x, wp_y, color=color, marker=marker, s=s, zorder=15)

    # Add dynamics info text for JERK model
    ego_info = f"Speed: {ego_speed:.2f}\nLane Centering: {lcenter:.2f}\nLane Align: {lalign:.2f}\nSpeed Limit: {speed_limit:.2f}"

    ego_info += f"\nSteering: {steering_angle:.3f}\naccel_long: {accel_long:.2f}\naccel_lat: {accel_lat:.2f}"

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
        # idx 7 = goal_dist_abs (0 near goal lane -> 1 far/unreachable); green->red colormap
        color = plt.cm.RdYlGn_r(float(lane_obs[i][7])) if obs_goal_lane_distance else "lightgrey"
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
        f"Lanes: {count_lane}\nBoundaries: {count_boundary}\nStride: {obs_lane_stride}/{obs_boundary_stride}"
        + ("\nLanes: green=near goal -> red=far" if obs_goal_lane_distance else ""),
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
    return base64.b64encode(zlib.compress(payload, level=6)).decode("ascii")


def generate_interactive_replay(scenario, replay, filename="replay.html"):
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
    lane_count = compute_effective_road_obs_count(env_cfg["obs_slots_lane_n"], env_cfg.get("obs_dropout_lane", 0.0))
    boundary_count = compute_effective_road_obs_count(
        env_cfg["obs_slots_boundary_n"], env_cfg.get("obs_dropout_boundary", 0.0)
    )

    # Quantize obs to int16: the obs chunk dominates file size; symmetric scale recovers values within ~S/2.
    obs_f32 = np.asarray(replay["obs"], dtype=np.float32)
    obs_scale = float(np.max(np.abs(obs_f32))) / 32767.0 if obs_f32.size else 1.0
    if obs_scale == 0.0:
        obs_scale = 1.0

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
        "obs": np.round(obs_f32 / obs_scale).astype(np.int16),
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
    for pool_name in ("pool_partner", "pool_lane", "pool_boundary", "pool_traffic"):
        if replay.get(pool_name) is not None:
            chunks[pool_name] = replay[pool_name].astype(np.int16, copy=False)

    # Ghost: logged/expert trajectory bbox per active agent, so policy-vs-log divergence is visible
    # (esp. control_sdc_only, 1 active agent). Frame-aligned: frame f = logged pose at timestep init_step + f.
    # Fields: x, y, heading, length, width; width <= 0 marks frames with no valid logged pose.
    agents = scenario.get("agents", []) or []
    active_indices = scenario.get("active_agent_indices", []) or []
    frame_count = int(replay["agent_f32"].shape[0])
    active_count = int(replay["obs"].shape[1])
    init_step = int(env_cfg.get("init_step", 0))
    ghost = np.zeros((frame_count, max(1, active_count), 5), dtype=np.float32)
    for slot in range(min(active_count, len(active_indices))):
        agent_idx = active_indices[slot]
        if agent_idx < 0 or agent_idx >= len(agents):
            continue
        a = agents[agent_idx]
        lx = np.asarray(a.get("log_trajectory_x") or [], dtype=np.float32)
        ly = np.asarray(a.get("log_trajectory_y") or [], dtype=np.float32)
        lh = np.asarray(a.get("log_heading") or [], dtype=np.float32)
        lv = np.asarray(a.get("log_valid") or [], dtype=np.int32)
        n = min(frame_count, max(0, lx.shape[0] - init_step))
        if n <= 0:
            continue
        window = slice(init_step, init_step + n)
        ghost[:n, slot, 0] = lx[window]
        ghost[:n, slot, 1] = ly[window]
        ghost[:n, slot, 2] = lh[window]
        ghost[:n, slot, 3] = float(a.get("sim_length", 0.0))
        width = float(a.get("sim_width", 0.0))
        ghost[:n, slot, 4] = np.where(lv[window] == 0, 0.0, width) if lv.shape[0] >= init_step + n else width
    chunks["ghost_f32"] = ghost

    goal_regen_mode = scenario.get("goal_regen_mode", env_cfg.get("goal_regen_mode", "finite"))
    if isinstance(goal_regen_mode, int):
        goal_regen_mode = "finite" if goal_regen_mode == binding.GOAL_REGEN_FINITE else "rolling"

    metadata = {
        "map_name": scenario.get("map_name", "Unknown"),
        "scenario_id": scenario.get("scenario_id", "Unknown"),
        "goal_regen_mode": goal_regen_mode,
        "active_indices": scenario.get("active_agent_indices", []),
        "total_agents": int(scenario.get("num_total_agents", replay["agent_f32"].shape[1])),
        "eval_overrides": replay.get("eval_overrides") or {},
        "frames": int(replay["agent_f32"].shape[0]),
        "agent_cap": int(replay["agent_f32"].shape[1]),
        "traffic_cap": int(replay["traffic_i16"].shape[1]),
        "active_count": int(replay["obs"].shape[1]),
        "obs_dim": int(replay["obs"].shape[2]),
        "obs_scale": obs_scale,
        "action_type": env_cfg.get("action_type", "continuous"),
        "dynamics_model": env_cfg.get("dynamics_model", "classic"),
        "num_goals": int(env_cfg["num_goals"]),
        "reward_conditioning": bool(env_cfg["reward_conditioning"]),
        "obs_slots_partners_n": int(env_cfg["obs_slots_partners_n"]),
        "ego_dim": int(binding.EGO_FEATURES),
        "reward_coef_count": int(binding.NUM_REWARD_COEFS),
        "partner_features": int(binding.PARTNER_FEATURES),
        "lane_features": int(binding.LANE_FEATURES),
        "boundary_features": int(binding.BOUNDARY_FEATURES),
        "traffic_features": int(binding.TRAFFIC_CONTROL_FEATURES),
        "lane_count": int(lane_count),
        "boundary_count": int(boundary_count),
        "obs_slots_lane_n": int(env_cfg["obs_slots_lane_n"]),
        "obs_slots_boundary_n": int(env_cfg["obs_slots_boundary_n"]),
        "obs_dropout_lane": float(env_cfg.get("obs_dropout_lane", 0.0)),
        "obs_dropout_boundary": float(env_cfg.get("obs_dropout_boundary", 0.0)),
        "obs_lane_stride": int(env_cfg.get("obs_lane_stride", 1)),
        "obs_boundary_stride": int(env_cfg.get("obs_boundary_stride", 1)),
        "traffic_obs_count": int(env_cfg["obs_slots_traffic_controls_n"]),
        "goal_features": int(binding.GOAL_FEATURES),
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
        :root {
            --bg:#e9ebee; --surface:rgba(255,255,255,.92); --surface-solid:#ffffff; --border:#dcdfe5;
            --text:#181b20; --muted:#6c7484; --field:rgba(108,116,132,.07);
            --accent:#0a66d0; --accent-soft:rgba(10,102,208,.12); --danger:#d6202c;
            --road:#c6cad1; --line:#959ca8; --edge:#2a2e35;
            --shadow:0 1px 2px rgba(22,26,34,.05),0 10px 30px rgba(22,26,34,.10);
            --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
        }
        [data-theme="dark"] {
            --bg:#0d0f12; --surface:rgba(23,26,31,.92); --surface-solid:#171a1f; --border:#2a2f37;
            --text:#e9ebef; --muted:#8c94a4; --field:rgba(140,148,164,.08);
            --accent:#4d9fff; --accent-soft:rgba(77,159,255,.16); --danger:#ff5560;
            --road:#363b43; --line:#5d6573; --edge:#06070a;
            --shadow:0 1px 2px rgba(0,0,0,.5),0 12px 34px rgba(0,0,0,.55);
        }
        * { box-sizing:border-box; }
        body { margin:0; overflow:hidden; background:var(--bg); color:var(--text); font:13px/1.45 system-ui,"Segoe UI",sans-serif; user-select:none; }
        canvas#c { display:block; width:100vw; height:100vh; cursor:crosshair; }
        #ui-layer { position:absolute; inset:0; pointer-events:none; z-index:10; }
        .panel { background:var(--surface); border:1px solid var(--border); border-radius:10px; box-shadow:var(--shadow); pointer-events:auto; backdrop-filter:blur(10px); }
        #loading-overlay { position:absolute; inset:0; z-index:9999; display:flex; flex-direction:column; gap:14px; align-items:center; justify-content:center; background:var(--bg); color:var(--muted); font-size:13px; letter-spacing:.04em; }
        .spinner { width:26px; height:26px; border-radius:50%; border:2px solid var(--border); border-top-color:var(--accent); animation:spin .8s linear infinite; }
        @keyframes spin { to { transform:rotate(360deg); } }
        h3 { margin:0; padding:0 0 8px; border-bottom:1px solid var(--border); color:var(--muted); font-size:10px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; }
        .label { margin-top:8px; color:var(--muted); font-size:10px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }
        .value { font-size:13px; font-weight:600; overflow-wrap:anywhere; }
        .mono { font-family:var(--mono); font-variant-numeric:tabular-nums; }
        .dim { color:var(--muted); }
        .highlight { color:var(--accent); }
        .link { color:var(--accent); cursor:pointer; }
        button, select, input { font:inherit; color:inherit; }
        .btn { border:1px solid var(--border); border-radius:7px; padding:6px 12px; background:var(--surface-solid); color:var(--text); font-size:12px; font-weight:600; cursor:pointer; }
        .btn:hover { border-color:var(--accent); color:var(--accent); }
        .btn.icon { display:flex; align-items:center; justify-content:center; width:36px; height:30px; padding:0; color:var(--accent); }
        select { border:1px solid var(--border); border-radius:7px; padding:5px 8px; background:var(--surface-solid); color:var(--text); font-size:12px; cursor:pointer; }
        input[type=range] { width:300px; accent-color:var(--accent); }
        input[type=number] { width:82px; padding:5px 8px; border:1px solid var(--border); border-radius:7px; background:var(--surface-solid); color:var(--text); font-family:var(--mono); font-size:12px; }
        #hud-global { position:absolute; top:14px; left:14px; width:232px; padding:12px 14px; }
        #hud-global h3 { cursor:pointer; }
        #hud-global.collapsed > *:not(h3) { display:none; }
        #hud-global.collapsed h3 { padding-bottom:0; border-bottom:0; }
        #overrides-body { grid-template-columns:1fr; max-height:34vh; overflow-y:auto; }
        #overrides-body .num { font-size:10px; overflow-wrap:anywhere; text-align:right; }
        /* Agent panel: dark instrument-cluster surface in both themes — scoped variable overrides restyle all children. */
        #hud-telemetry { --border:#4a5468; --muted:#aab3c5; --field:rgba(255,255,255,.07); --accent:#7cbcff; --accent-soft:rgba(124,188,255,.25); position:absolute; top:14px; right:14px; width:372px; max-height:calc(100vh - 90px); padding:12px 14px; overflow-y:auto; display:none; background:rgba(54,62,77,.95); color:#eef1f6; }
        [data-theme="dark"] #hud-telemetry { background:rgba(48,56,70,.95); }
        #tel-drag-handle { display:flex; align-items:center; gap:6px; }
        .cam-chip { margin-left:auto; padding:2px 8px; border:1px solid var(--border); border-radius:10px; background:transparent; color:var(--muted); font-size:9px; font-weight:600; letter-spacing:.06em; text-transform:uppercase; cursor:pointer; }
        .cam-chip:hover { color:var(--accent); border-color:var(--accent); }
        .heat { grid-column:1 / -1; display:grid; gap:3px; margin-top:6px; padding:6px; border:1px solid var(--border); border-radius:7px; background:rgba(5,10,20,.18); }
        .heat-cell { position:relative; display:flex; align-items:center; justify-content:center; height:25px; border-radius:4px; border:1px solid rgba(255,255,255,.10); font-family:var(--mono); font-size:9px; font-weight:700; font-variant-numeric:tabular-nums; box-shadow:inset 0 1px 0 rgba(255,255,255,.14); overflow:hidden; }
        .heat-cell.selected { z-index:1; outline:2px solid #fff; outline-offset:1px; box-shadow:0 0 0 3px var(--accent),0 0 18px rgba(124,188,255,.72),inset 0 0 0 1px rgba(13,20,32,.55); transform:scale(1.04); }
        .heat-cell.selected::after { content:''; position:absolute; top:3px; right:3px; width:5px; height:5px; border-radius:50%; background:#fff; box-shadow:0 0 0 1px rgba(13,20,32,.75); }
        .heat-lab { display:flex; align-items:center; justify-content:center; height:19px; font-family:var(--mono); font-size:9px; color:#c8d1e0; }
        .heat-cap { grid-column:1 / -1; color:var(--muted); font-size:9.5px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }
        #warn-row { display:none; flex-wrap:wrap; gap:6px; margin-top:10px; }
        .warn-chip { padding:3px 9px; border-radius:5px; background:var(--danger); color:#fff; font-size:10px; font-weight:700; letter-spacing:.08em; }
        .speed-block { display:flex; align-items:baseline; gap:6px; margin-top:8px; }
        .speed-num { font-family:var(--mono); font-size:32px; font-weight:600; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
        .speed-unit { color:var(--muted); font-size:11px; font-weight:600; letter-spacing:.06em; }
        .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:5px; margin-top:8px; }
        .item { display:flex; justify-content:space-between; align-items:baseline; gap:6px; padding:5px 8px; border:1px solid var(--border); border-radius:6px; background:var(--field); }
        .name { color:var(--muted); font-size:9.5px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .num { font-family:var(--mono); font-size:11.5px; font-variant-numeric:tabular-nums; }
        .score-num { margin-top:6px; font-family:var(--mono); font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }
        .toggle-header { width:100%; margin-top:12px; padding:6px 0; display:flex; justify-content:space-between; align-items:center; background:transparent; color:var(--muted); border:0; border-bottom:1px solid var(--border); border-radius:0; font-size:10px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; text-align:left; cursor:pointer; }
        .toggle-header:hover { color:var(--text); }
        .toggle-header span:last-child { transition:transform .15s ease; }
        .toggle-header.is-collapsed span:last-child { transform:rotate(-90deg); }
        .toggle-body.is-collapsed { display:none; }
        #controls { position:absolute; left:50%; bottom:16px; transform:translateX(-50%); padding:9px 12px; display:flex; gap:12px; align-items:center; }
        .step-counter { font-size:12px; min-width:90px; text-align:center; }
        #obs-container { position:absolute; left:14px; bottom:16px; width:390px; height:390px; min-width:250px; min-height:250px; max-width:92vw; max-height:86vh; display:none; overflow:hidden; resize:both; border-radius:10px; }
        #obs-title { position:absolute; top:0; left:0; right:0; z-index:2; display:flex; gap:6px; align-items:center; padding:6px 10px; background:var(--surface); border-bottom:1px solid var(--border); color:var(--muted); font-size:10px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; cursor:grab; }
        #obs-title span { flex:1; }
        .obs-tool { padding:3px 8px; border:1px solid var(--border); border-radius:5px; background:transparent; color:var(--muted); font-size:9.5px; font-weight:600; letter-spacing:.05em; cursor:pointer; }
        .obs-tool:hover { color:var(--accent); border-color:var(--accent); }
        #obs-canvas { width:100%; height:100%; background:#fff; }
    </style>
</head>
<body>
    <div id="loading-overlay"><div class="spinner"></div><div id="load-text">Decoding replay&#8230;</div></div>
    <div id="ui-layer">
        <div id="hud-global" class="panel collapsed">
            <h3 onclick="toggleGlobalPanel()">Scenario <span id="globalChevron" style="float:right">&#9656;</span></h3>
            <div class="label">Map</div><div class="value" id="meta-map">-</div>
            <div class="label">Scenario ID</div><div class="value mono" id="meta-id" style="font-size:11px">-</div>
            <div class="label">Agents (active / total)</div><div class="value mono" id="meta-agents">-</div>
            <button type="button" class="toggle-header is-collapsed" id="overrides-header" data-target="overrides-body"><span>Eval overrides</span><span>&#9662;</span></button>
            <div id="overrides-body" class="grid toggle-body is-collapsed"></div>
            <div class="label">Obs Road</div><div class="value" id="meta-obs-road">-</div>
            <button class="btn" onclick="toggleTheme()" style="width:100%; margin-top:12px">Toggle theme</button>
        </div>
        <div id="hud-telemetry" class="panel">
            <h3 id="tel-drag-handle">Agent <span id="tel-id" class="highlight mono">?</span><button type="button" id="camMode" class="cam-chip" onclick="toggleCamMode()">world cam</button></h3>
            <div id="warn-row"></div>
            <div class="speed-block"><span id="tel-speed" class="speed-num">0.0</span><span class="speed-unit">km/h</span></div>
            <div class="grid">
                <div class="item"><span class="name">steer &#176;</span><span class="num" id="tel-st">0.0</span></div>
                <div class="item"><span class="name">lane</span><span class="num" id="tel-lane">-1</span></div>
                <div class="item"><span class="name">accel lon</span><span class="num" id="tel-al">0</span></div>
                <div class="item"><span class="name">accel lat</span><span class="num" id="tel-alat">0</span></div>
                <div class="item"><span class="name">jerk lon</span><span class="num" id="tel-jl">0</span></div>
                <div class="item"><span class="name">jerk lat</span><span class="num" id="tel-jlat">0</span></div>
            </div>
            <div class="label">Position x / y / heading</div>
            <div class="mono dim" style="font-size:11.5px"><span id="tel-x">0</span>, <span id="tel-y">0</span>, <span id="tel-h">0</span></div>
            <div class="label">Policy</div><div id="policy-grid" class="grid"></div>
            <button type="button" class="toggle-header" data-target="puffer-score-body"><span>Puffer score</span><span>&#9662;</span></button>
            <div id="puffer-score-body" class="toggle-body"><div id="tel-ps" class="score-num">0.000</div></div>
            <button type="button" class="toggle-header" data-target="puffer-grid"><span>Puffer metrics</span><span>&#9662;</span></button>
            <div id="puffer-grid" class="grid toggle-body"></div>
            <button type="button" class="toggle-header" data-target="metrics-grid"><span>Metrics</span><span>&#9662;</span></button>
            <div id="metrics-grid" class="grid toggle-body"></div>
        </div>
        <div id="obs-container" class="panel"><div id="obs-title"><span>Ego-centric observation</span><button type="button" class="obs-tool" onclick="resetObsZoom(event)">1x</button><button type="button" id="obsModeBtn" class="obs-tool" onclick="toggleObsMode(event)">BOTH</button><button type="button" class="obs-tool" onclick="toggleObsSize(event)">Expand</button></div><canvas id="obs-canvas"></canvas></div>
        <div id="controls" class="panel">
            <button id="btnPlay" class="btn icon" onclick="toggle()"></button>
            <span class="mono step-counter"><span id="stepNow">0</span><span class="dim"> / </span><span id="stepTotal">0</span></span>
            <input id="sld" type="range" min="0" value="0" step="1">
            <select id="speedSel" onchange="changeSpeed()"><option value="0.25">0.25x</option><option value="1">1x</option><option value="2">2x</option><option value="4" selected>4x</option><option value="8">8x</option></select>
            <input type="number" id="agentSearch" placeholder="agent id" onkeydown="if(event.key==='Enter') searchAgent()">
        </div>
    </div>
    <canvas id="c"></canvas>
    __B64_SCRIPT_TAGS__
    <script>
        // Read base64 chunks from non-executed text/plain tags. Keeping them out of the JS source keeps
        // this script under V8's ~512MiB max source-string length (a giant inline literal fails to compile).
        const B64_CHUNKS = Array.from(document.querySelectorAll('script.b64chunk'), n => n.textContent);
        const METRIC_LABELS = __METRIC_LABELS__;
        const VEHICLE_COLORS = __VEHICLE_COLORS__;
        // Order must match the Log fields written in env_binding.h vec_get_obs_html_frame (15 values).
        const PUFFER_LABELS = ["score","no at fault","no offroad","no red light","progress > .2","direction","ttc","progress ratio","speed limit","comfort","multi lane","wrong way dist","speed violation","multiplier","weighted avg"];
        const ACCEL = [-4,-2.667,-1.333,0,1.333,2.667,4], STEER = [-0.667,-0.5,-0.333,-0.167,0,0.167,0.333,0.5,0.667];
        const JLONG = [-15,-4,0,4], JLAT = [-4,0,4];
        const SVG_PLAY = '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M4.5 2.5v11l9-5.5z" fill="currentColor"/></svg>';
        const SVG_PAUSE = '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>';
        let H, C = {}, F, paths = {0:new Path2D(),1:new Path2D(),2:new Path2D()}, lastDrawn = -1;
        const c = document.getElementById('c'), ctx = c.getContext('2d');
        const obsC = document.getElementById('obs-canvas'), obsCtx = obsC.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        let step = 0, play = false, speed = 4, lastTick = 0;
        let cam = {x:0,y:0,z:5,drag:false,lx:0,ly:0};
        let followedId = null, isEgoCam = false, darkMode = false, showGhost = false;
        let obsZoom = 2.2, obsExpanded = false, obsMode = 2;
        const OBS_MODES = ["ALL","POOL","BOTH"];

        function chunk(name) {
            const m = H.chunks[name], start = H.dataStart + m.offset, n = m.nbytes / ({float32:4,int32:4,int16:2,uint8:1}[m.dtype]);
            if (m.dtype === "float32") return new Float32Array(H.buffer, start, n);
            if (m.dtype === "int32") return new Int32Array(H.buffer, start, n);
            if (m.dtype === "int16") return new Int16Array(H.buffer, start, n);
            return new Uint8Array(H.buffer, start, n);
        }
        function frameMax() { return Math.max(0, (H ? H.frames : 1) - 1); }
        async function initReplay() {
            // fetch() of a data: URI decodes base64 natively — far faster than an atob + per-char copy loop.
            // Payload is split into chunks (each < V8's ~512MiB max string length); decode each and concat the bytes.
            const parts = [];
            for (const b64 of B64_CHUNKS) parts.push(await (await fetch('data:application/octet-stream;base64,' + b64)).blob());
            const compressed = new Blob(parts);
            const ds = new DecompressionStream('deflate');
            const buf = await new Response(compressed.stream().pipeThrough(ds)).arrayBuffer();
            const view = new DataView(buf), headerLen = view.getUint32(0, true);
            H = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
            H.buffer = buf; H.dataStart = 4 + headerLen + ((-(4 + headerLen)) & 3);
            for (const name of Object.keys(H.chunks)) C[name] = chunk(name);
            F = {af:H.chunks.agent_f32.shape[2], ai:H.chunks.agent_i32.shape[2], mf:H.chunks.metrics_f32.shape[2], pf:H.chunks.puffer_f32.shape[2], tf:H.chunks.traffic_i16.shape[2]};
            document.getElementById('meta-map').textContent = String(H.map_name).split('/').pop();
            document.getElementById('meta-id').textContent = H.scenario_id || "-";
            document.getElementById('meta-agents').textContent = H.active_count + ' / ' + H.total_agents;
            showGhost = (H.active_count === 1) && !!(H.chunks && H.chunks.ghost_f32);
            const ov = H.eval_overrides || {}, ovKeys = Object.keys(ov);
            if (ovKeys.length) document.getElementById('overrides-body').innerHTML = ovKeys.map(k=>`<div class="item"><span class="name">${k}</span><span class="num">${ov[k]}</span></div>`).join('');
            else document.getElementById('overrides-header').style.display = 'none';
            document.getElementById('sld').max = frameMax();
            document.getElementById('stepTotal').textContent = frameMax();
            updateBtn();
            const first = getFrameAgents(0)[0]; if (first) { cam.x = first.x; cam.y = first.y; }
            document.getElementById('loading-overlay').style.display = 'none';
            window.onresize();
            requestAnimationFrame(() => { buildMapPaths(); draw(true); });
        }
        initReplay().catch(err => { console.error(err); document.getElementById('load-text').textContent = 'Replay load failed. See console.'; });

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
        function colorFor(id, control_state, stopped) { return control_state === 3 ? "#9b59b6" : (stopped ? "red" : (control_state === 0 ? VEHICLE_COLORS[Math.abs(id) % VEHICLE_COLORS.length] : (control_state === 1 ? "#bfbfbf" : "#404040"))); }
        function rr(x, y, w, h, r) { ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h); }
        function drawAgentBody(a, outline) {
            // Top-down sprites in agent frame (+x forward, rear at -l/2). Detail only when zoomed in enough to see it.
            const l = a.l, w = a.w, detail = cam.z > 2.2;
            ctx.strokeStyle = outline; ctx.lineWidth = .08;
            if (a.type === 2) { ctx.fillStyle = a.c; ctx.beginPath(); ctx.arc(0, 0, Math.max(w, 0.7) / 2, 0, 7); ctx.fill(); ctx.stroke(); return; }
            const rad = Math.min(w*0.30, l*0.10), GLASS = 'rgba(28,40,56,.60)', LITE = 'rgba(255,255,255,.18)';
            ctx.fillStyle = a.c;
            rr(-l/2, -w/2, l, w, rad); ctx.fill(); ctx.stroke();
            if (detail) {
                ctx.fillStyle = LITE; rr(-l*0.26, -w*0.41, l*0.40, w*0.82, w*0.12); ctx.fill();
                ctx.fillStyle = GLASS;
                rr(l*0.10, -w*0.36, l*0.14, w*0.72, w*0.10); ctx.fill();
                rr(-l*0.34, -w*0.33, l*0.09, w*0.66, w*0.08); ctx.fill();
                ctx.fillStyle = 'rgba(255,244,200,.85)';
                rr(l/2 - l*0.05, -w*0.40, l*0.04, w*0.16, w*0.04); ctx.fill();
                rr(l/2 - l*0.05, w*0.24, l*0.04, w*0.16, w*0.04); ctx.fill();
            }
            if (detail && a.al < -0.3) {
                ctx.fillStyle = '#ff2222';
                rr(-l/2, -w*0.42, l*0.05, w*0.18, w*0.04); ctx.fill();
                rr(-l/2, w*0.24, l*0.05, w*0.18, w*0.04); ctx.fill();
            }
        }
        function agentAt(frame, idx) {
            // Light decode (no metric copies) — called for every agent every frame; metrics/puffer read on demand.
            const ib = (frame * H.agent_cap + idx) * F.ai;
            if (!C.agent_i32[ib+2]) return null;
            const fb = (frame * H.agent_cap + idx) * F.af;
            return {idx:idx, id:C.agent_i32[ib], type:C.agent_i32[ib+1], control_state:C.agent_i32[ib+3], stopped:C.agent_i32[ib+4], removed:C.agent_i32[ib+5], cl:C.agent_i32[ib+6], slot:C.agent_i32[ib+7], x:C.agent_f32[fb], y:C.agent_f32[fb+1], h:C.agent_f32[fb+3], l:C.agent_f32[fb+4], w:C.agent_f32[fb+5], s:C.agent_f32[fb+6], st:C.agent_f32[fb+7], al:C.agent_f32[fb+8], alat:C.agent_f32[fb+9], jl:C.agent_f32[fb+10], jlat:C.agent_f32[fb+11], c:colorFor(C.agent_i32[ib], C.agent_i32[ib+3], C.agent_i32[ib+4])};
        }
        function getFrameAgents(frame) { const out = []; for (let i=0;i<H.agent_cap;i++) { const a = agentAt(frame, i); if (a) out.push(a); } return out; }
        function drawGhosts(f) {
            if (!showGhost || !C.ghost_f32) return;
            const N = H.chunks.ghost_f32.shape[1];
            ctx.strokeStyle = '#ff0000'; ctx.fillStyle = 'rgba(255,0,0,.22)'; ctx.lineWidth = .28; ctx.setLineDash([.6,.4]);
            for (let j=0;j<N;j++) { const b=(f*N+j)*5, w=C.ghost_f32[b+4]; if (w <= 0) continue; ctx.save(); ctx.translate(C.ghost_f32[b], C.ghost_f32[b+1]); ctx.rotate(C.ghost_f32[b+2]); ctx.beginPath(); ctx.rect(-C.ghost_f32[b+3]/2, -w/2, C.ghost_f32[b+3], w); ctx.fill(); ctx.stroke(); ctx.restore(); }
            ctx.setLineDash([]);
        }
        function findAgent(frame, id) { for (let i=0;i<H.agent_cap;i++) { const a = agentAt(frame, i); if (a && a.id === id) return a; } return null; }
        function trafficAt(frame, idx) {
            const db = (frame * H.traffic_cap + idx) * F.tf;
            if (!C.traffic_i16[db]) return null;
            const sb = idx * 6, type = C.traffic_types[idx] || C.traffic_i16[db+1], state = C.traffic_i16[db+2];
            return {type, state, stop_line:Array.from(C.traffic_stop_lines.subarray(sb, sb + 6))};
        }
        function trafficColor(t) { return t.state === 1 ? "#ff0000" : t.state === 2 ? "#ffff00" : t.state === 3 ? "#00ff00" : "#888888"; }
        function getColors() { const s = getComputedStyle(document.documentElement); return {bg:s.getPropertyValue('--bg'), road:s.getPropertyValue('--road'), line:s.getPropertyValue('--line'), edge:s.getPropertyValue('--edge'), text:s.getPropertyValue('--text'), accent:s.getPropertyValue('--accent')}; }
        function resizeObsCanvas() {
            const r = obsC.getBoundingClientRect();
            if (r.width <= 0 || r.height <= 0) return;
            const w = Math.max(1, Math.floor(r.width * dpr)), h = Math.max(1, Math.floor(r.height * dpr));
            if (obsC.width !== w || obsC.height !== h) { obsC.width = w; obsC.height = h; draw(true); }
        }
        new ResizeObserver(resizeObsCanvas).observe(document.getElementById('obs-container'));
        window.onresize = () => { c.width = innerWidth; c.height = innerHeight; resizeObsCanvas(); draw(true); };
        function toggleTheme(){ darkMode=!darkMode; document.documentElement.setAttribute('data-theme', darkMode?'dark':'light'); draw(true); }
        function toggleGlobalPanel(){ const p=document.getElementById('hud-global'), collapsed=!p.classList.contains('collapsed'); p.classList.toggle('collapsed', collapsed); document.getElementById('globalChevron').innerHTML=collapsed?'&#9656;':'&#9662;'; }
        function toggleCamMode(){ if(followedId !== null){ isEgoCam=!isEgoCam; draw(true); } }
        function resetObsZoom(e){ if(e) e.stopPropagation(); obsZoom=2.2; draw(true); }
        function toggleObsMode(e){ if(e) e.stopPropagation(); obsMode=(obsMode+1)%OBS_MODES.length; document.getElementById('obsModeBtn').textContent=OBS_MODES[obsMode]; draw(true); }
        function toggleObsSize(e){ if(e) e.stopPropagation(); const p=document.getElementById('obs-container'), b=e ? e.currentTarget : null; obsExpanded=!obsExpanded; p.style.width=obsExpanded?'680px':'390px'; p.style.height=obsExpanded?'680px':'390px'; if(b) b.textContent=obsExpanded?'Collapse':'Expand'; resizeObsCanvas(); draw(true); }
        function searchAgent(){ const id=parseInt(document.getElementById('agentSearch').value); if(!isNaN(id)){ followedId=id; play=false; updateBtn(); draw(true); } }
        document.addEventListener('keydown', e => { if(!H || e.target.tagName === 'INPUT') return; if(e.code === 'Space'){ toggle(); e.preventDefault(); } if(e.code === 'ArrowRight'){ play=false; updateBtn(); step=Math.min(step+1,frameMax()); draw(true); } if(e.code === 'ArrowLeft'){ play=false; updateBtn(); step=Math.max(step-1,0); draw(true); } if(e.code === 'Escape'){ followedId=null; isEgoCam=false; updateUI(); draw(true); } if(e.code === 'KeyG'){ showGhost=!showGhost; draw(true); } });
        c.onwheel = e => { e.preventDefault(); cam.z *= Math.exp(-e.deltaY * .001); draw(true); };
        c.onmousedown = e => { if(!H) return; const r=c.getBoundingClientRect(), wx=(e.clientX-r.left-c.width/2)/cam.z+cam.x, wy=(e.clientY-r.top-c.height/2)/-cam.z+cam.y; let hit=null, agents=getFrameAgents(Math.floor(step)); if(!isEgoCam) for(const a of agents) if(Math.hypot(wx-a.x, wy-a.y) < Math.max(a.l,3)){ hit=a.id; break; } if(hit !== null){ followedId=hit; cam.drag=false; } else { followedId=null; isEgoCam=false; cam.drag=true; cam.lx=e.clientX; cam.ly=e.clientY; } draw(true); };
        window.onmouseup = () => cam.drag = false;
        c.onmousemove = e => { if(cam.drag && !isEgoCam){ cam.x -= (e.clientX-cam.lx)/cam.z; cam.y -= (e.clientY-cam.ly)/-cam.z; cam.lx=e.clientX; cam.ly=e.clientY; draw(true); } };
        obsC.addEventListener('wheel', e => { e.preventDefault(); obsZoom = Math.max(.45, Math.min(8, obsZoom * Math.exp(-e.deltaY * .001))); draw(true); }, {passive:false});
        function dragPanel(handleId, panelId) { const h=document.getElementById(handleId), p=document.getElementById(panelId); let on=false,sx=0,sy=0,sl=0,st=0; h.addEventListener('mousedown', e => { if(e.target.closest('button')) return; on=true; sx=e.clientX; sy=e.clientY; const r=p.getBoundingClientRect(); sl=r.left; st=r.top; p.style.right='auto'; p.style.bottom='auto'; p.style.left=sl+'px'; p.style.top=st+'px'; }); window.addEventListener('mousemove', e => { if(on){ p.style.left=(sl+e.clientX-sx)+'px'; p.style.top=(st+e.clientY-sy)+'px'; }}); window.addEventListener('mouseup', () => on=false); }
        dragPanel('obs-title','obs-container');
        document.querySelectorAll('.obs-tool').forEach(btn => {
            btn.addEventListener('mousedown', e => e.stopPropagation());
            btn.addEventListener('click', e => e.stopPropagation());
        });
        document.querySelectorAll('.toggle-header').forEach(header => header.addEventListener('click', () => {
            const body = document.getElementById(header.dataset.target);
            if (!body) return;
            const collapsed = !body.classList.contains('is-collapsed');
            body.classList.toggle('is-collapsed', collapsed);
            header.classList.toggle('is-collapsed', collapsed);
        }));

        function poolAt(name, frame, slot, idx) {
            if (!C[name] || slot < 0) return 0;
            const n = H.chunks[name].shape[2];
            return C[name][(frame * H.active_count + slot) * n + idx] || 0;
        }
        const POOL_STOPS = [[56,189,248],[34,197,94],[250,204,21],[239,68,68]];
        const HEAT_STOPS = [[40,48,62],[36,99,196],[77,196,255],[235,248,255]];
        function heatColor(t) { t = t < 0 ? 0 : (t > 1 ? 1 : t); const f = t * (HEAT_STOPS.length - 1), i = Math.floor(f), k = f - i, a = HEAT_STOPS[i], b = HEAT_STOPS[Math.min(i + 1, HEAT_STOPS.length - 1)]; return `rgb(${Math.round(a[0]+(b[0]-a[0])*k)},${Math.round(a[1]+(b[1]-a[1])*k)},${Math.round(a[2]+(b[2]-a[2])*k)})`; }
        function poolColor(t) { t = t < 0 ? 0 : (t > 1 ? 1 : t); const f = t * (POOL_STOPS.length - 1), i = Math.floor(f), k = f - i, a = POOL_STOPS[i], b = POOL_STOPS[Math.min(i + 1, POOL_STOPS.length - 1)]; return `rgb(${Math.round(a[0]+(b[0]-a[0])*k)},${Math.round(a[1]+(b[1]-a[1])*k)},${Math.round(a[2]+(b[2]-a[2])*k)})`; }
        function drawPoolLegend(maxN) { const w = 116*dpr, h = 9*dpr, x = obsC.width - w - 12*dpr, y = obsC.height - 20*dpr, grad = obsCtx.createLinearGradient(x, 0, x+w, 0); for (let i=0;i<=10;i++) grad.addColorStop(i/10, poolColor(i/10)); obsCtx.fillStyle = grad; obsCtx.fillRect(x, y, w, h); obsCtx.strokeStyle = "rgba(0,0,0,.45)"; obsCtx.lineWidth = dpr; obsCtx.strokeRect(x, y, w, h); obsCtx.fillStyle = "#111"; obsCtx.font = `bold ${9.5*dpr}px system-ui`; obsCtx.textAlign = "left"; obsCtx.fillText("pool wins  1", x, y - 4*dpr); obsCtx.textAlign = "right"; obsCtx.fillText(maxN, x+w, y - 4*dpr); }
        function selectedGoals(frame, agent) {
            if (!agent || agent.slot < 0) return [];
            const base = (frame * H.active_count + agent.slot) * H.obs_dim, obs = C.obs, Q = H.obs_scale;
            let p = base + H.ego_dim;
            if (H.reward_conditioning) p += H.reward_coef_count;
            const scale = H.scales.obs_norm_goal_offset_m * Q;
            const out = [];
            for (let i=0;i<H.num_goals;i++) {
                const o = p + i * H.goal_features;
                let empty = true;
                for (let j=0;j<H.goal_features;j++) if (obs[o+j] !== 0) empty = false;
                if (empty) continue;
                const rx = obs[o] * scale, ry = obs[o+1] * scale, ch = Math.cos(agent.h), sh = Math.sin(agent.h);
                out.push({x:agent.x + rx * ch - ry * sh, y:agent.y + rx * sh + ry * ch, i:i});
            }
            return out;
        }
        function decodeObs(frame, slot) {
            if (slot < 0 || slot >= H.active_count) return null;
            const base = (frame * H.active_count + slot) * H.obs_dim, obs = C.obs, Q = H.obs_scale, LF = H.lane_features, BF = H.boundary_features, TF = H.traffic_features;
            let p = base; const egoStart = p; p += H.ego_dim;
            if (H.reward_conditioning) p += H.reward_coef_count;
            const targetStart = p; p += H.num_goals * H.goal_features;
            const partnersStart = p; p += H.obs_slots_partners_n * H.partner_features;
            const lanesStart = p; p += H.lane_count * LF;
            const boundsStart = p; p += H.boundary_count * BF;
            const trafficStart = p;
            const rot = (x,y) => [-y,x];
            const zero = (off,n) => { for(let i=0;i<n;i++) if(obs[off+i] !== 0) return false; return true; };
            const roads = (start,count,poolName,feat) => { const out=[]; for(let i=0;i<count;i++){ const o=start+i*feat; if(zero(o,feat)) continue; let xy=rot(obs[o]*Q,obs[o+1]*Q), cs=rot(obs[o+5]*Q,obs[o+6]*Q); out.push([xy[0],xy[1],obs[o+3]*Q*H.scales.road_length_to_position,obs[o+4]*Q*H.scales.road_width_to_position,cs[0],cs[1],poolAt(poolName,frame,slot,i)]); } return out; };
            const partners = []; for(let i=0;i<H.obs_slots_partners_n;i++){ const o=partnersStart+i*H.partner_features; if(zero(o,H.partner_features)) continue; let xy=rot(obs[o]*Q,obs[o+1]*Q), h=Math.atan2(obs[o+6],obs[o+5]); h = ((h + Math.PI/2 + Math.PI) % (2*Math.PI)) - Math.PI; partners.push({x:xy[0],y:xy[1],l:obs[o+3]*Q*H.scales.veh_len_to_position,w:obs[o+4]*Q*H.scales.veh_width_to_position,h:h,s:obs[o+7]*Q,pool:poolAt("pool_partner",frame,slot,i)}); }
            const gps = []; for(let i=0;i<H.num_goals;i++){ const o=targetStart+i*H.goal_features; if(zero(o,H.goal_features)) continue; let scale=H.scales.goal_to_position*Q, xy=rot(obs[o]*scale, obs[o+1]*scale); gps.push(xy); }
            const controls = []; for(let i=0;i<H.traffic_obs_count;i++){ const o=trafficStart+i*TF; if(zero(o,TF)) continue; let a=rot(obs[o]*Q,obs[o+1]*Q), b=rot(obs[o+2]*Q,obs[o+3]*Q); controls.push({type:Math.round(obs[o+5]*Q), state:Math.round(obs[o+6]*Q), x1:a[0], y1:a[1], x2:b[0], y2:b[1], pool:poolAt("pool_traffic",frame,slot,i)}); }
            return {ego:{s:obs[egoStart]*Q,w:obs[egoStart+1]*Q*H.scales.veh_width_to_position,l:obs[egoStart+2]*Q*H.scales.veh_len_to_position,st:obs[egoStart+3]*Q,al:obs[egoStart+4]*Q,alat:obs[egoStart+5]*Q}, partners, lanes:roads(lanesStart,H.lane_count,"pool_lane",LF), bounds:roads(boundsStart,H.boundary_count,"pool_boundary",BF), gps, traffic_controls:controls};
        }
        function drawObs(frame) {
            resizeObsCanvas();
            const scale = (Math.min(obsC.width, obsC.height) / 2) * obsZoom, px = dpr / scale;
            const showAll = obsMode !== 1, showPool = obsMode !== 0, bothMode = obsMode === 2;
            let poolMax = 1;
            for(const r of frame.lanes) if(r[6] > poolMax) poolMax = r[6];
            for(const r of frame.bounds) if(r[6] > poolMax) poolMax = r[6];
            for(const p of frame.partners) if(p.pool > poolMax) poolMax = p.pool;
            for(const t of frame.traffic_controls) if(t.pool > poolMax) poolMax = t.pool;
            const pw = t => (2.0 + 2.4*t)*px;
            obsCtx.fillStyle = "#fff"; obsCtx.fillRect(0,0,obsC.width,obsC.height);
            obsCtx.save(); obsCtx.translate(obsC.width/2, obsC.height/2); obsCtx.scale(scale, -scale); obsCtx.lineCap = "round";
            if(showAll){ obsCtx.strokeStyle=bothMode?"#000":"#bbb"; obsCtx.lineWidth=1.5*px; for(const r of frame.lanes){ obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[4]*r[2]/2,r[1]+r[5]*r[2]/2); obsCtx.lineTo(r[0]-r[4]*r[2]/2,r[1]-r[5]*r[2]/2); obsCtx.stroke(); } }
            if(showAll){ obsCtx.strokeStyle=bothMode?"#000":"#333"; obsCtx.lineWidth=3*px; for(const r of frame.bounds){ obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[4]*r[2]/2,r[1]+r[5]*r[2]/2); obsCtx.lineTo(r[0]-r[4]*r[2]/2,r[1]-r[5]*r[2]/2); obsCtx.stroke(); } }
            if(showPool){ for(const r of frame.lanes.concat(frame.bounds)){ if(r[6] > 0){ obsCtx.strokeStyle=poolColor(r[6]/poolMax); obsCtx.lineWidth=pw(r[6]/poolMax); obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[4]*r[2]/2,r[1]+r[5]*r[2]/2); obsCtx.lineTo(r[0]-r[4]*r[2]/2,r[1]-r[5]*r[2]/2); obsCtx.stroke(); } } }
            for(const g of frame.gps){ obsCtx.fillStyle="magenta"; obsCtx.beginPath(); obsCtx.arc(g[0],g[1],5*px,0,7); obsCtx.fill(); }
            for(const t of frame.traffic_controls){ if(showAll){ obsCtx.strokeStyle = bothMode ? "#000" : (t.type === 1 ? trafficColor({state:t.state}) : (t.type === 2 ? "#cc0000" : "#ffd700")); obsCtx.lineWidth=2.5*px; obsCtx.beginPath(); obsCtx.moveTo(t.x1,t.y1); obsCtx.lineTo(t.x2,t.y2); obsCtx.stroke(); } if(showPool && t.pool > 0){ obsCtx.strokeStyle=poolColor(t.pool/poolMax); obsCtx.lineWidth=pw(t.pool/poolMax)+0.8*px; obsCtx.beginPath(); obsCtx.moveTo(t.x1,t.y1); obsCtx.lineTo(t.x2,t.y2); obsCtx.stroke(); } }
            for(const p of frame.partners){ const win = showPool && p.pool > 0; if(!showAll && !win) continue; obsCtx.save(); obsCtx.translate(p.x,p.y); obsCtx.rotate(p.h); if(showAll){ obsCtx.fillStyle=bothMode?"rgba(0,0,0,.55)":"rgba(136,136,136,.8)"; obsCtx.strokeStyle=bothMode?"#000":"#333"; obsCtx.lineWidth=1.5*px; obsCtx.beginPath(); obsCtx.rect(-p.l/2,-p.w/2,p.l,p.w); obsCtx.fill(); obsCtx.stroke(); } if(win){ obsCtx.strokeStyle=poolColor(p.pool/poolMax); obsCtx.lineWidth=pw(p.pool/poolMax); obsCtx.strokeRect(-p.l/2,-p.w/2,p.l,p.w); } obsCtx.restore(); }
            if(frame.ego){ obsCtx.save(); obsCtx.rotate(Math.PI/2); obsCtx.fillStyle="rgba(0,102,255,.8)"; obsCtx.strokeStyle="#000"; obsCtx.lineWidth=1.5*px; obsCtx.beginPath(); obsCtx.rect(-frame.ego.l/2,-frame.ego.w/2,frame.ego.l,frame.ego.w); obsCtx.fill(); obsCtx.stroke(); obsCtx.restore(); }
            obsCtx.restore();
            if(showPool && poolMax > 1) drawPoolLegend(poolMax);
        }
        let panelKey = null, refs = null, lastWarnKey = "";
        function ensurePanels() {
            // Panel structure is identical across agents/frames — build the DOM once, update textContent per frame.
            const discrete = H.action_type === "discrete" && !!C.policy_probs;
            const actionDims = H.chunks.raw_action.shape.length > 2 ? H.chunks.raw_action.shape[2] : 1;
            const key = (discrete ? 'd' : 'c') + actionDims;
            if (refs && panelKey === key) return;
            panelKey = key;
            const mg = document.getElementById('metrics-grid');
            mg.innerHTML = METRIC_LABELS.map(l=>`<div class="item"><span class="name">${l}</span><span class="num">-</span></div>`).join('');
            const pg = document.getElementById('puffer-grid');
            pg.innerHTML = PUFFER_LABELS.map(l=>`<div class="item"><span class="name">${l}</span><span class="num">-</span></div>`).join('');
            const pol = document.getElementById('policy-grid');
            let html = '<div class="item"><span class="name">value</span><span class="num" data-pol="v">-</span></div><div class="item"><span class="name">entropy</span><span class="num" data-pol="e">-</span></div>';
            let labels = [];
            if (discrete) {
                // Probability heatmap over the 2D action grid (rows x cols), index i = row * cols.length + col.
                const jerk = H.dynamics_model === "jerk", rows = jerk ? JLONG : ACCEL, cols = jerk ? JLAT : STEER;
                html += `<div class="heat" style="grid-template-columns:auto repeat(${cols.length},1fr)">`;
                html += '<div class="heat-lab"></div>' + cols.map(v=>`<div class="heat-lab">${v.toFixed(1)}</div>`).join('');
                for (let r=0;r<rows.length;r++) { html += `<div class="heat-lab">${rows[r].toFixed(1)}</div>`; for (let cI=0;cI<cols.length;cI++) html += '<div class="heat-cell"></div>'; }
                html += `</div><div class="heat-cap">${jerk ? 'jerk_long &#8595; / jerk_lat &#8594;' : 'accel &#8595; / steer &#8594;'}</div>`;
            } else {
                labels = H.action_type === "continuous" ? (H.dynamics_model === "jerk" ? ["jerk_long","jerk_lat"] : ["accel","steer"]) : Array.from({length:actionDims}, (_,i)=>`p${i}`);
                labels.forEach(l => html += `<div class="item"><span class="name">${l}</span><span class="num pol-act">-</span></div>`);
                if (C.policy_mean) { labels.forEach(l => html += `<div class="item"><span class="name">mean ${l}</span><span class="num pol-mean">-</span></div><div class="item"><span class="name">std ${l}</span><span class="num pol-std">-</span></div>`); html += '<div class="item"><span class="name">log prob</span><span class="num" data-pol="lp">-</span></div>'; }
            }
            pol.innerHTML = html;
            refs = {
                metric: [...mg.querySelectorAll('.num')],
                puffer: [...pg.querySelectorAll('.num')],
                polV: pol.querySelector('[data-pol=v]'), polE: pol.querySelector('[data-pol=e]'),
                heat: [...pol.querySelectorAll('.heat-cell')],
                acts: [...pol.querySelectorAll('.pol-act')], means: [...pol.querySelectorAll('.pol-mean')], stds: [...pol.querySelectorAll('.pol-std')],
                polLp: pol.querySelector('[data-pol=lp]'),
                discrete, actionDims,
            };
        }
        function updatePolicy(frame, agent) {
            if (agent.slot < 0) return;
            const s = frame * H.active_count + agent.slot;
            refs.polV.textContent = C.value[s].toFixed(3);
            refs.polE.textContent = C.entropy[s].toFixed(3);
            const ab = s * refs.actionDims;
            if (refs.discrete) {
                const n = refs.heat.length, pb = s * n, selected = Math.round(C.raw_action[ab]);
                let maxP = 1e-9;
                for (let i=0;i<n;i++) maxP = Math.max(maxP, C.policy_probs[pb+i]);
                for (let i=0;i<n;i++){
                    const prob = Math.max(0, Math.min(1, C.policy_probs[pb+i]));
                    const cell = refs.heat[i], t = prob / maxP;
                    cell.style.background = heatColor(t);
                    cell.textContent = prob >= 0.04 || i === selected ? Math.round(prob*100) + '%' : '';
                    cell.style.color = t > 0.6 ? '#0d1420' : '#c4cddc';
                    cell.classList.toggle('selected', i===selected);
                    cell.title = (prob*100).toFixed(1)+'%';
                }
                return;
            }
            for (let i=0;i<refs.acts.length;i++) {
                const raw = C.raw_action[ab+i], clip = C.clipped_action[ab+i];
                let scaled = clip;
                if (H.action_type === "continuous") scaled = H.dynamics_model === "jerk" ? (i===0 ? (clip < 0 ? clip*15 : clip*4) : clip*4) : (i===0 ? clip*4 : clip*.667);
                refs.acts[i].textContent = scaled.toFixed(2) + ' / ' + raw.toFixed(2);
            }
            if (C.policy_mean) { for (let i=0;i<refs.means.length;i++){ refs.means[i].textContent = C.policy_mean[ab+i].toFixed(3); refs.stds[i].textContent = C.policy_std[ab+i].toFixed(3); } refs.polLp.textContent = C.policy_log_prob[s].toFixed(3); }
        }
        function updateUI(agent=null) {
            const f = Math.max(0, Math.min(frameMax(), Math.floor(step)));
            document.getElementById('stepNow').textContent = f; document.getElementById('sld').value = f;
            const hud = document.getElementById('hud-telemetry'), obsBox = document.getElementById('obs-container');
            if (followedId === null || !agent) { hud.style.display='none'; obsBox.style.display='none'; return; }
            hud.style.display='block'; document.getElementById('camMode').textContent = isEgoCam ? 'ego cam' : 'world cam';
            ensurePanels();
            const mb = (f * H.agent_cap + agent.idx) * F.mf, pb = (f * H.agent_cap + agent.idx) * F.pf;
            for (const [id,val] of [["tel-id",agent.id],["tel-speed",(agent.s*3.6).toFixed(1)],["tel-st",(agent.st*180/Math.PI).toFixed(1)],["tel-al",agent.al.toFixed(2)],["tel-alat",agent.alat.toFixed(2)],["tel-jl",agent.jl.toFixed(2)],["tel-jlat",agent.jlat.toFixed(2)],["tel-x",agent.x.toFixed(1)],["tel-y",agent.y.toFixed(1)],["tel-h",agent.h.toFixed(3)],["tel-lane",agent.cl],["tel-ps",C.puffer_f32[pb].toFixed(3)]]) document.getElementById(id).textContent = val;
            for (let i=0;i<refs.metric.length;i++) refs.metric[i].textContent = C.metrics_f32[mb+i].toFixed(2);
            for (let i=0;i<refs.puffer.length;i++) refs.puffer[i].textContent = C.puffer_f32[pb+i].toFixed(3);
            updatePolicy(f, agent);
            const warnings = []; if(C.metrics_f32[mb] === 1) warnings.push("COLLISION"); if(C.metrics_f32[mb+1] === 1) warnings.push("OFFROAD"); if(C.metrics_f32[mb+2] === 1) warnings.push("RED LIGHT"); if(C.metrics_f32[mb+3] === 1) warnings.push("STOP SIGN");
            const warnKey = warnings.join('|'), warnRow = document.getElementById('warn-row');
            if (warnKey !== lastWarnKey) { lastWarnKey = warnKey; warnRow.style.display = warnings.length ? 'flex' : 'none'; warnRow.innerHTML = warnings.map(w=>`<span class="warn-chip">${w}</span>`).join(''); }
            const obs = decodeObs(f, agent.slot); if (obs) { obsBox.style.display='block'; drawObs(obs); } else obsBox.style.display='none';
        }
        function draw(force=false) {
            if(!H) return;
            const f = Math.max(0, Math.min(frameMax(), Math.floor(step)));
            if(!force && f === lastDrawn) return;
            const target = followedId !== null ? findAgent(f, followedId) : null;
            if (target) { cam.x = target.x; cam.y = target.y; }
            updateUI(target);
            const colors = getColors(); ctx.fillStyle = colors.bg; ctx.fillRect(0,0,c.width,c.height); ctx.save(); ctx.translate(c.width/2,c.height/2); ctx.scale(cam.z,-cam.z); if(isEgoCam && target) ctx.rotate(Math.PI/2 - target.h); ctx.translate(-cam.x,-cam.y);
            ctx.lineCap='round'; ctx.strokeStyle=colors.road; ctx.lineWidth=.5; ctx.stroke(paths[0]); ctx.strokeStyle=colors.line; ctx.setLineDash([1,1]); ctx.stroke(paths[1]); ctx.setLineDash([]); ctx.strokeStyle=colors.edge; ctx.lineWidth=.8; ctx.stroke(paths[2]);
            drawGhosts(f);
            for(const a of getFrameAgents(f)){ ctx.save(); ctx.translate(a.x,a.y); ctx.rotate(a.h); drawAgentBody(a, darkMode?'#fff':'#111'); ctx.restore(); ctx.save(); ctx.translate(a.x,a.y); if(isEgoCam && target) ctx.rotate(-Math.PI/2 + target.h); else ctx.scale(1,-1); ctx.fillStyle=colors.text; ctx.font='600 '+(14/cam.z)+'px system-ui'; ctx.textAlign='center'; ctx.fillText(a.id,0,(isEgoCam && target)?a.w/2+.5:-a.w/2-.5); ctx.restore(); if(a.id === followedId){ ctx.save(); ctx.translate(a.x,a.y); ctx.strokeStyle=colors.accent; ctx.lineWidth=3/cam.z; ctx.beginPath(); ctx.arc(0,0,Math.max(a.l,a.w)*1.2,0,7); ctx.stroke(); ctx.restore(); } }
            for(let i=0;i<H.traffic_static_count;i++){ const t=trafficAt(f,i); if(!t) continue; const sl=t.stop_line; ctx.lineCap='butt'; if(t.type === 1){ ctx.strokeStyle=trafficColor(t); ctx.lineWidth=Math.min(1.5,3/cam.z); } else { ctx.strokeStyle=t.type === 2 ? '#ff0000' : '#ffd700'; ctx.lineWidth=Math.min(1.2,2.5/cam.z); ctx.setLineDash([6/cam.z,4/cam.z]); } ctx.beginPath(); ctx.moveTo(sl[0],sl[1]); ctx.lineTo(sl[3],sl[4]); ctx.stroke(); ctx.setLineDash([]); }
            if(target){ for(const g of selectedGoals(f,target)){ const r=Math.max(1.8,8/cam.z); ctx.strokeStyle='#38bdf8'; ctx.fillStyle='rgba(56,189,248,.22)'; ctx.lineWidth=Math.max(.25,2.5/cam.z); ctx.beginPath(); ctx.arc(g.x,g.y,r,0,7); ctx.fill(); ctx.stroke(); } }
            ctx.restore(); lastDrawn = f;
        }
        function toggle(){ play=!play; lastTick=performance.now(); updateBtn(); if(play) requestAnimationFrame(loop); }
        function updateBtn(){ document.getElementById('btnPlay').innerHTML = play ? SVG_PAUSE : SVG_PLAY; }
        function changeSpeed(){ speed=parseFloat(document.getElementById('speedSel').value); lastTick=performance.now(); }
        function loop(ts){
            if(!play) return;
            const dt = Math.min((ts-lastTick)/1000, 0.25);
            lastTick = ts;
            step += dt * speed * 10;
            while(step > frameMax()) step -= frameMax() + 1;
            draw();
            requestAnimationFrame(loop);
        }
        document.getElementById('sld').oninput = e => { step = +e.target.value; play=false; updateBtn(); draw(true); };
    </script>
</body>
</html>
    """
    # Split base64 on 4-char boundaries so each chunk decodes to whole bytes, and emit each as a
    # non-executed <script type="text/plain"> tag. This keeps both every text node and the main JS
    # source under V8's ~512MiB max string length (base64 has no '<', so "</script>" can't appear).
    B64_CHUNK_CHARS = 64 * 1024 * 1024
    chunks = [payload[i : i + B64_CHUNK_CHARS] for i in range(0, len(payload), B64_CHUNK_CHARS)]
    b64_script_tags = "".join('<script type="text/plain" class="b64chunk">' + c + "</script>" for c in chunks)
    final_html = (
        html_template.replace("__B64_SCRIPT_TAGS__", b64_script_tags)
        .replace("__METRIC_LABELS__", json.dumps(METRIC_LABELS, separators=(",", ":")))
        .replace("__VEHICLE_COLORS__", json.dumps(VEHICLE_COLORS, separators=(",", ":")))
    )
    with open(filename, "w") as f:
        f.write(final_html)


def build_gallery_index(folder_path=".", file_metrics=None):
    """Build an index.html navigator for per-episode replay HTMLs in folder_path.

    If `file_metrics` is a dict mapping `<html basename> -> {metric_name: value}`,
    the index also exposes a sort dropdown so the user can flip between sort
    keys (default: `score` ascending — failures bubble to the top). When
    `file_metrics` is None or empty, behaves as before (filename-order
    dropdown, no sort UI).
    """
    files = [f for f in os.listdir(folder_path) if f != "index.html" and f.endswith(".html")]

    if not files:
        print("No matching .html files found in this directory.")
        return

    # Lexicographic sort over the full filename. With the triage_html stem
    # `{map}_{scenario_id}_{scenarios_done:04d}_epoch{e}_step{s}.html`, the
    # zero-padded scenarios_done dominates ordering within a map.
    files.sort()

    metrics_map = file_metrics or {}
    has_metrics = bool(metrics_map)

    # (key, default_direction). Anything in this list with at least one
    # non-null value across files gets a dropdown entry. Default direction
    # is what makes triage-useful values bubble to the top.
    SORT_KEYS = [
        ("score", "asc"),
        ("dnf_rate", "desc"),
        ("episode_return", "asc"),
        ("num_goals_reached", "asc"),
        ("collision_rate", "desc"),
        ("offroad_rate", "desc"),
        ("red_light_violation_rate", "desc"),
        ("total_infractions", "desc"),
        ("total_distance_travelled", "asc"),
        ("episode_length", "asc"),
    ]

    available_keys = []
    if has_metrics:
        present = set()
        for v in metrics_map.values():
            present.update(v.keys())
        for k, d in SORT_KEYS:
            if k in present:
                available_keys.append((k, d))

    metrics_json = json.dumps(metrics_map, separators=(",", ":"))
    defaults_json = json.dumps({k: d for k, d in available_keys}, separators=(",", ":"))

    def make_label(f):
        if not has_metrics or f not in metrics_map:
            return f.replace(".html", "").replace("_", " ")
        bits = [f.replace(".html", "")]
        for k in ("score", "dnf_rate", "num_goals_reached", "episode_return"):
            if k in metrics_map[f]:
                v = metrics_map[f][k]
                bits.append(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
        return "  ·  ".join(bits)

    options_html = "\n".join(f'<option value="{f}" data-name="{f}">{make_label(f)}</option>' for f in files)

    sort_ui = ""
    sort_js = ""
    if has_metrics and available_keys:
        sort_options = "\n".join(
            f'<option value="{k}"{" selected" if k == "score" else ""}>{k}</option>' for k, _ in available_keys
        )
        sort_ui = (
            '<span style="color:#888;font-size:12px;font-weight:bold">SORT</span>'
            f'<select id="sortKey" onchange="onSortKeyChange()">{sort_options}</select>'
            '<select id="sortDir" onchange="resortFiles()">'
            '<option value="asc" selected>asc</option>'
            '<option value="desc">desc</option>'
            "</select>"
        )
        sort_js = (
            (
                "const FILE_METRICS = __METRICS_JSON__;"
                "const SORT_DEFAULTS = __DEFAULTS_JSON__;"
                "const sortKeySel = document.getElementById('sortKey');"
                "const sortDirSel = document.getElementById('sortDir');"
                "function onSortKeyChange() {"
                "  const k = sortKeySel.value;"
                "  if (SORT_DEFAULTS[k]) sortDirSel.value = SORT_DEFAULTS[k];"
                "  resortFiles();"
                "}"
                "function resortFiles() {"
                "  const key = sortKeySel.value;"
                "  const dir = sortDirSel.value;"
                "  const opts = Array.from(select.options);"
                "  opts.sort(function (a, b) {"
                "    const fA = a.getAttribute('data-name');"
                "    const fB = b.getAttribute('data-name');"
                "    const mA = (FILE_METRICS[fA] || {})[key];"
                "    const mB = (FILE_METRICS[fB] || {})[key];"
                "    const nA = (mA === undefined || mA === null) ? -Infinity : mA;"
                "    const nB = (mB === undefined || mB === null) ? -Infinity : mB;"
                "    if (nA === nB) return fA.localeCompare(fB);"
                "    return dir === 'asc' ? nA - nB : nB - nA;"
                "  });"
                "  const current = select.value;"
                "  while (select.firstChild) select.removeChild(select.firstChild);"
                "  opts.forEach(function (o) { select.appendChild(o); });"
                "  select.value = current;"
                "  updateButtons();"
                "}"
                "resortFiles();"
            )
            .replace("__METRICS_JSON__", metrics_json)
            .replace("__DEFAULTS_JSON__", defaults_json)
        )

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
            #fileSelect { flex: 1 1 280px; min-width: 240px; }
        </style>
    </head>
    <body>
        <div id="topbar">
            <div class="title">PUFFERDRIVE GALLERY</div>
            <button id="prevBtn" onclick="navigate(-1)">&#9664; Prev</button>
            __SORT_UI__
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

            __SORT_JS__

            updateButtons();
        </script>
    </body>
    </html>
    """

    final_html = (
        html_content.replace("__OPTIONS__", options_html)
        .replace("__FIRST__", files[0])
        .replace("__SORT_UI__", sort_ui)
        .replace("__SORT_JS__", sort_js)
    )

    # 5. Save the file
    index_path = os.path.join(folder_path, "index.html")
    with open(index_path, "w") as f:
        f.write(final_html)
