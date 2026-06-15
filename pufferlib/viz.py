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
    target_type: str = "static",
    reward_conditioning: bool = False,
    num_target_waypoints: int = 5,
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

    if isinstance(target_type, int):
        target_type = "static" if target_type == binding.TARGET_STATIC else "dynamic"

    ego_dim = binding.EGO_FEATURES

    # Partner obs
    partner_feature_size = binding.PARTNER_FEATURES
    # Road obs
    road_feature_size = binding.ROAD_FEATURES
    # Traffic control obs
    traffic_control_feature_size = binding.TRAFFIC_CONTROL_FEATURES
    lane_segment_count = compute_effective_road_obs_count(obs_slots_lane_n, obs_dropout_lane)
    boundary_segment_count = compute_effective_road_obs_count(obs_slots_boundary_n, obs_dropout_boundary)

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
    partners_end = partners_start + obs_slots_partners_n * partner_feature_size
    partners_obs = obs_flat[:, partners_start:partners_end]
    partners_obs = partners_obs.reshape(-1, obs_slots_partners_n, partner_feature_size)

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
    target_type="static",
    reward_conditioning=False,
    num_target_waypoints=10,
    obs_slots_partners_n=16,
    obs_slots_lane_n=32,
    obs_slots_boundary_n=32,
    obs_slots_traffic_controls_n=4,
    obs_dropout_lane=0.0,
    obs_dropout_boundary=0.0,
    obs_lane_stride=1,
    obs_boundary_stride=1,
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
    if isinstance(target_type, int):
        target_type = "static" if target_type == binding.TARGET_STATIC else "dynamic"

    fig, ax = plt.subplots(figsize=(20, 20))

    ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_controls_obs = unpack_obs(
        obs,
        target_type=target_type,
        reward_conditioning=reward_conditioning,
        num_target_waypoints=num_target_waypoints,
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
    target_position_scale = scales["goal_to_position"] if target_type == "static" else 1.0

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
        f"Lanes: {count_lane}\nBoundaries: {count_boundary}\nStride: {obs_lane_stride}/{obs_boundary_stride}",
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
    return base64.b64encode(zlib.compress(payload, level=3)).decode("ascii")


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
    for pool_name in ("pool_partner", "pool_lane", "pool_boundary", "pool_traffic"):
        if replay.get(pool_name) is not None:
            chunks[pool_name] = replay[pool_name].astype(np.int16, copy=False)

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
        "num_target_waypoints": int(env_cfg["num_target_waypoints"]),
        "reward_conditioning": bool(env_cfg["reward_conditioning"]),
        "obs_slots_partners_n": int(env_cfg["obs_slots_partners_n"]),
        "partner_features": int(binding.PARTNER_FEATURES),
        "lane_count": int(lane_count),
        "boundary_count": int(boundary_count),
        "obs_slots_lane_n": int(env_cfg["obs_slots_lane_n"]),
        "obs_slots_boundary_n": int(env_cfg["obs_slots_boundary_n"]),
        "obs_dropout_lane": float(env_cfg.get("obs_dropout_lane", 0.0)),
        "obs_dropout_boundary": float(env_cfg.get("obs_dropout_boundary", 0.0)),
        "obs_lane_stride": int(env_cfg.get("obs_lane_stride", 1)),
        "obs_boundary_stride": int(env_cfg.get("obs_boundary_stride", 1)),
        "traffic_obs_count": int(env_cfg["obs_slots_traffic_controls_n"]),
        "target_features": 3 if env_cfg.get("target_type", "static") == "static" else 5,
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
        #obs-container { position:absolute; left:14px; bottom:18px; width:390px; height:390px; min-width:250px; min-height:250px; max-width:92vw; max-height:86vh; display:none; overflow:hidden; resize:both; pointer-events:auto; border:2px solid var(--accent); border-radius:8px; background:white; }
        #obs-title { position:absolute; top:0; left:0; right:0; z-index:2; padding:6px 8px; display:flex; gap:6px; align-items:center; background:var(--accent); color:white; font-size:11px; font-weight:900; letter-spacing:.06em; cursor:grab; }
        #obs-title span { flex:1; }
        .obs-tool { padding:4px 7px; border-radius:5px; background:rgba(0,0,0,.28); font-size:10px; }
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
            <div class="label">Obs Road</div><div class="value" id="meta-obs-road">-</div>
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
        <div id="obs-container"><div id="obs-title"><span>EGO-CENTRIC NN OBS</span><button type="button" class="obs-tool" onclick="resetObsZoom(event)">1X</button><button type="button" id="obsModeBtn" class="obs-tool" onclick="toggleObsMode(event)">BOTH</button><button type="button" class="obs-tool" onclick="toggleObsSize(event)">EXPAND</button></div><canvas id="obs-canvas"></canvas></div>
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
        let step = 0, play = false, speed = 4, lastTick = 0;
        let cam = {x:0,y:0,z:5,drag:false,lx:0,ly:0};
        let followedId = null, isEgoCam = false, darkMode = false;
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
            document.getElementById('meta-obs-road').textContent = `L ${H.lane_count}/${H.obs_slots_lane_n} s${H.obs_lane_stride} d${Number(H.obs_dropout_lane).toFixed(2)} | B ${H.boundary_count}/${H.obs_slots_boundary_n} s${H.obs_boundary_stride} d${Number(H.obs_dropout_boundary).toFixed(2)}`;
            document.getElementById('sld').max = frameMax();
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
        function toggleObsSize(e){ if(e) e.stopPropagation(); const p=document.getElementById('obs-container'), b=e ? e.currentTarget : null; obsExpanded=!obsExpanded; p.style.width=obsExpanded?'680px':'390px'; p.style.height=obsExpanded?'680px':'390px'; if(b) b.textContent=obsExpanded?'COLLAPSE':'EXPAND'; resizeObsCanvas(); draw(true); }
        function searchAgent(){ const id=parseInt(document.getElementById('agentSearch').value); if(!isNaN(id)){ followedId=id; play=false; updateBtn(); draw(true); } }
        document.addEventListener('keydown', e => { if(!H || e.target.tagName === 'INPUT') return; if(e.code === 'Space'){ toggle(); e.preventDefault(); } if(e.code === 'ArrowRight'){ play=false; updateBtn(); step=Math.min(step+1,frameMax()); draw(true); } if(e.code === 'ArrowLeft'){ play=false; updateBtn(); step=Math.max(step-1,0); draw(true); } if(e.code === 'Escape'){ followedId=null; isEgoCam=false; updateUI(); draw(true); } });
        c.onwheel = e => { e.preventDefault(); cam.z *= Math.exp(-e.deltaY * .001); draw(true); };
        c.onmousedown = e => { if(!H) return; const r=c.getBoundingClientRect(), wx=(e.clientX-r.left-c.width/2)/cam.z+cam.x, wy=(e.clientY-r.top-c.height/2)/-cam.z+cam.y; let hit=null, agents=getFrameAgents(Math.floor(step)); if(!isEgoCam) for(const a of agents) if(Math.hypot(wx-a.x, wy-a.y) < Math.max(a.l,3)){ hit=a.id; break; } if(hit !== null){ followedId=hit; cam.drag=false; } else { followedId=null; isEgoCam=false; cam.drag=true; cam.lx=e.clientX; cam.ly=e.clientY; } draw(true); };
        window.onmouseup = () => cam.drag = false;
        c.onmousemove = e => { if(cam.drag && !isEgoCam){ cam.x -= (e.clientX-cam.lx)/cam.z; cam.y -= (e.clientY-cam.ly)/-cam.z; cam.lx=e.clientX; cam.ly=e.clientY; draw(true); } };
        obsC.addEventListener('wheel', e => { e.preventDefault(); obsZoom = Math.max(.45, Math.min(8, obsZoom * Math.exp(-e.deltaY * .001))); draw(true); }, {passive:false});
        function dragPanel(handleId, panelId) { const h=document.getElementById(handleId), p=document.getElementById(panelId); let on=false,sx=0,sy=0,sl=0,st=0; h.addEventListener('mousedown', e => { if(e.target.closest('button')) return; on=true; sx=e.clientX; sy=e.clientY; const r=p.getBoundingClientRect(); sl=r.left; st=r.top; p.style.right='auto'; p.style.bottom='auto'; p.style.left=sl+'px'; p.style.top=st+'px'; }); window.addEventListener('mousemove', e => { if(on){ p.style.left=(sl+e.clientX-sx)+'px'; p.style.top=(st+e.clientY-sy)+'px'; }}); window.addEventListener('mouseup', () => on=false); }
        dragPanel('tel-drag-handle','hud-telemetry'); dragPanel('obs-title','obs-container');
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
        function poolAlpha(n) { return Math.min(.58, .22 + n / 140); }
        function selectedGoals(frame, agent) {
            if (!agent || agent.slot < 0) return [];
            const base = (frame * H.active_count + agent.slot) * H.obs_dim, obs = C.obs;
            let p = base + 10;
            if (H.reward_conditioning) p += 17;
            const scale = H.target_type === "static" ? H.scales.obs_norm_goal_offset_m : H.scales.obs_norm_xy_offset_m;
            const out = [];
            for (let i=0;i<H.num_target_waypoints;i++) {
                const o = p + i * H.target_features;
                let empty = true;
                for (let j=0;j<H.target_features;j++) if (obs[o+j] !== 0) empty = false;
                if (empty) continue;
                const rx = obs[o] * scale, ry = obs[o+1] * scale, ch = Math.cos(agent.h), sh = Math.sin(agent.h);
                out.push({x:agent.x + rx * ch - ry * sh, y:agent.y + rx * sh + ry * ch, i:i});
            }
            return out;
        }
        function decodeObs(frame, slot) {
            if (slot < 0 || slot >= H.active_count) return null;
            const base = (frame * H.active_count + slot) * H.obs_dim, obs = C.obs;
            let p = base, ego = obs.subarray(p, p+10); p += 10;
            if (H.reward_conditioning) p += 17;
            const targetStart = p; p += H.num_target_waypoints * H.target_features;
            const partnersStart = p; p += H.obs_slots_partners_n * H.partner_features;
            const lanesStart = p; p += H.lane_count * 7;
            const boundsStart = p; p += H.boundary_count * 7;
            const trafficStart = p;
            const rot = (x,y) => [-y,x];
            const zero = (off,n) => { for(let i=0;i<n;i++) if(obs[off+i] !== 0) return false; return true; };
            const roads = (start,count,poolName) => { const out=[]; for(let i=0;i<count;i++){ const o=start+i*7; if(zero(o,7)) continue; let xy=rot(obs[o],obs[o+1]), cs=rot(obs[o+5],obs[o+6]); out.push([xy[0],xy[1],obs[o+3]*H.scales.road_length_to_position,obs[o+4]*H.scales.road_width_to_position,cs[0],cs[1],poolAt(poolName,frame,slot,i)]); } return out; };
            const partners = []; for(let i=0;i<H.obs_slots_partners_n;i++){ const o=partnersStart+i*H.partner_features; if(zero(o,H.partner_features)) continue; let xy=rot(obs[o],obs[o+1]), h=Math.atan2(obs[o+6],obs[o+5]); h = ((h + Math.PI/2 + Math.PI) % (2*Math.PI)) - Math.PI; partners.push({x:xy[0],y:xy[1],l:obs[o+3]*H.scales.veh_len_to_position,w:obs[o+4]*H.scales.veh_width_to_position,h:h,s:obs[o+7],pool:poolAt("pool_partner",frame,slot,i)}); }
            const gps = []; for(let i=0;i<H.num_target_waypoints;i++){ const o=targetStart+i*H.target_features; if(zero(o,H.target_features)) continue; let scale=H.target_type === "static" ? H.scales.goal_to_position : 1, xy=rot(obs[o]*scale, obs[o+1]*scale); gps.push(xy); }
            const controls = []; for(let i=0;i<H.traffic_obs_count;i++){ const o=trafficStart+i*7; if(zero(o,7)) continue; let a=rot(obs[o],obs[o+1]), b=rot(obs[o+2],obs[o+3]); controls.push({type:obs[o+5], state:obs[o+6], x1:a[0], y1:a[1], x2:b[0], y2:b[1], pool:poolAt("pool_traffic",frame,slot,i)}); }
            return {ego:{s:ego[0],w:ego[1]*H.scales.veh_width_to_position,l:ego[2]*H.scales.veh_len_to_position,st:ego[3],al:ego[4],alat:ego[5]}, partners, lanes:roads(lanesStart,H.lane_count,"pool_lane"), bounds:roads(boundsStart,H.boundary_count,"pool_boundary"), gps, traffic_controls:controls};
        }
        function drawObs(frame) {
            resizeObsCanvas();
            const scale = (Math.min(obsC.width, obsC.height) / 2) * obsZoom, px = dpr / scale;
            const showAll = obsMode !== 1, showPool = obsMode !== 0;
            obsCtx.fillStyle = "#fff"; obsCtx.fillRect(0,0,obsC.width,obsC.height);
            obsCtx.save(); obsCtx.translate(obsC.width/2, obsC.height/2); obsCtx.scale(scale, -scale); obsCtx.lineCap = "round";
            if(showAll){ obsCtx.strokeStyle="#bbb"; obsCtx.lineWidth=1.5*px; for(const r of frame.lanes){ obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[4]*r[2]/2,r[1]+r[5]*r[2]/2); obsCtx.lineTo(r[0]-r[4]*r[2]/2,r[1]-r[5]*r[2]/2); obsCtx.stroke(); } }
            if(showAll){ obsCtx.strokeStyle="#333"; obsCtx.lineWidth=3*px; for(const r of frame.bounds){ obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[4]*r[2]/2,r[1]+r[5]*r[2]/2); obsCtx.lineTo(r[0]-r[4]*r[2]/2,r[1]-r[5]*r[2]/2); obsCtx.stroke(); } }
            if(showPool){
                for(const r of frame.lanes){ if(r[6] > 0){ obsCtx.strokeStyle=`rgba(0,125,145,${poolAlpha(r[6])})`; obsCtx.lineWidth=(obsMode === 1 ? 2.2 : 2.0)*px; obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[4]*r[2]/2,r[1]+r[5]*r[2]/2); obsCtx.lineTo(r[0]-r[4]*r[2]/2,r[1]-r[5]*r[2]/2); obsCtx.stroke(); } }
                for(const r of frame.bounds){ if(r[6] > 0){ obsCtx.strokeStyle=`rgba(200,0,0,${poolAlpha(r[6])})`; obsCtx.lineWidth=(obsMode === 1 ? 2.2 : 2.0)*px; obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[4]*r[2]/2,r[1]+r[5]*r[2]/2); obsCtx.lineTo(r[0]-r[4]*r[2]/2,r[1]-r[5]*r[2]/2); obsCtx.stroke(); } }
            }
            for(const g of frame.gps){ obsCtx.fillStyle="magenta"; obsCtx.beginPath(); obsCtx.arc(g[0],g[1],5*px,0,7); obsCtx.fill(); }
            for(const t of frame.traffic_controls){ if(showPool && t.pool > 0){ obsCtx.strokeStyle=`rgba(0,125,145,${poolAlpha(t.pool)})`; obsCtx.lineWidth=(obsMode === 1 ? 3.2 : 2.4)*px; obsCtx.beginPath(); obsCtx.moveTo(t.x1,t.y1); obsCtx.lineTo(t.x2,t.y2); obsCtx.stroke(); } if(showAll){ obsCtx.strokeStyle = t.type === 1 ? trafficColor({state:t.state}) : (t.type === 2 ? "#cc0000" : "#ffd700"); obsCtx.lineWidth=2.5*px; obsCtx.beginPath(); obsCtx.moveTo(t.x1,t.y1); obsCtx.lineTo(t.x2,t.y2); obsCtx.stroke(); } }
            for(const p of frame.partners){ if(!showAll && !(showPool && p.pool > 0)) continue; obsCtx.save(); obsCtx.translate(p.x,p.y); obsCtx.rotate(p.h); if(showAll){ obsCtx.fillStyle="rgba(136,136,136,.8)"; obsCtx.strokeStyle="#333"; obsCtx.lineWidth=1.5*px; obsCtx.beginPath(); obsCtx.rect(-p.l/2,-p.w/2,p.l,p.w); obsCtx.fill(); obsCtx.stroke(); } if(showPool && p.pool > 0){ obsCtx.strokeStyle=`rgba(0,125,145,${poolAlpha(p.pool)})`; obsCtx.lineWidth=(obsMode === 1 ? 2.4 : 2.0)*px; obsCtx.strokeRect(-p.l/2,-p.w/2,p.l,p.w); } obsCtx.restore(); }
            if(frame.ego){ obsCtx.save(); obsCtx.rotate(Math.PI/2); obsCtx.fillStyle="rgba(0,102,255,.8)"; obsCtx.strokeStyle="#000"; obsCtx.lineWidth=1.5*px; obsCtx.beginPath(); obsCtx.rect(-frame.ego.l/2,-frame.ego.w/2,frame.ego.l,frame.ego.w); obsCtx.fill(); obsCtx.stroke(); obsCtx.restore(); }
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
            labels.forEach((label,i) => html += `<div class="item"><span class="name">${label}</span><span class="num">${Number(scaled[i]).toFixed(2)} / ${Number(raw[i]).toFixed(2)}</span></div>`);
            if (C.policy_mean) { const mb = ab; labels.forEach((label,i) => html += `<div class="item"><span class="name">mean ${label}</span><span class="num">${C.policy_mean[mb+i].toFixed(3)}</span></div><div class="item"><span class="name">std ${label}</span><span class="num">${C.policy_std[mb+i].toFixed(3)}</span></div>`); html += `<div class="item"><span class="name">log prob</span><span class="num">${C.policy_log_prob[frame * H.active_count + agent.slot].toFixed(3)}</span></div>`; }
            return html;
        }
        function updateUI(agent=null) {
            const f = Math.max(0, Math.min(frameMax(), Math.floor(step))); document.getElementById('stepDisplay').textContent = f; document.getElementById('sld').value = f;
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
            const f = Math.max(0, Math.min(frameMax(), Math.floor(step)));
            const target = followedId !== null ? findAgent(f, followedId) : null;
            if (target) { cam.x = target.x; cam.y = target.y; }
            updateUI(target);
            const colors = getColors(); ctx.fillStyle = colors.bg; ctx.fillRect(0,0,c.width,c.height); ctx.save(); ctx.translate(c.width/2,c.height/2); ctx.scale(cam.z,-cam.z); if(isEgoCam && target) ctx.rotate(Math.PI/2 - target.h); ctx.translate(-cam.x,-cam.y);
            ctx.lineCap='round'; ctx.strokeStyle=colors.road; ctx.lineWidth=.5; ctx.stroke(paths[0]); ctx.strokeStyle=colors.line; ctx.setLineDash([1,1]); ctx.stroke(paths[1]); ctx.setLineDash([]); ctx.strokeStyle=colors.edge; ctx.lineWidth=.8; ctx.stroke(paths[2]);
            for(const a of getFrameAgents(f)){ ctx.save(); ctx.translate(a.x,a.y); ctx.rotate(a.h); ctx.fillStyle=a.c; ctx.strokeStyle=darkMode?'#fff':'#111'; ctx.lineWidth=.1; ctx.beginPath(); ctx.rect(-a.l/2,-a.w/2,a.l,a.w); ctx.fill(); ctx.stroke(); ctx.fillStyle='rgba(255,255,0,.55)'; ctx.fillRect(a.l/2-.5,-a.w/2,.5,a.w); ctx.restore(); ctx.save(); ctx.translate(a.x,a.y); if(isEgoCam && target) ctx.rotate(-Math.PI/2 + target.h); else ctx.scale(1,-1); ctx.fillStyle=colors.text; ctx.font='bold '+(14/cam.z)+'px Arial'; ctx.textAlign='center'; ctx.fillText(a.id,0,(isEgoCam && target)?a.w/2+.5:-a.w/2-.5); ctx.restore(); if(a.id === followedId){ ctx.save(); ctx.translate(a.x,a.y); ctx.strokeStyle='#00ff00'; ctx.lineWidth=4/cam.z; ctx.beginPath(); ctx.arc(0,0,Math.max(a.l,a.w)*1.2,0,7); ctx.stroke(); ctx.restore(); } }
            for(let i=0;i<H.traffic_static_count;i++){ const t=trafficAt(f,i); if(!t) continue; const sl=t.stop_line; ctx.lineCap='butt'; if(t.type === 1){ ctx.strokeStyle=trafficColor(t); ctx.lineWidth=Math.max(1.5,3/cam.z); } else { ctx.strokeStyle=t.type === 2 ? '#ff0000' : '#ffd700'; ctx.lineWidth=Math.max(1.2,2.5/cam.z); ctx.setLineDash([6/cam.z,4/cam.z]); } ctx.beginPath(); ctx.moveTo(sl[0],sl[1]); ctx.lineTo(sl[3],sl[4]); ctx.stroke(); ctx.setLineDash([]); }
            if(target){ for(const g of selectedGoals(f,target)){ const r=Math.max(1.8,8/cam.z); ctx.strokeStyle='#38bdf8'; ctx.fillStyle='rgba(56,189,248,.22)'; ctx.lineWidth=Math.max(.25,2.5/cam.z); ctx.beginPath(); ctx.arc(g.x,g.y,r,0,7); ctx.fill(); ctx.stroke(); } }
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
            while(step > frameMax()) step -= frameMax() + 1;
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
            #topbar { padding: 12px 20px; background: #222; display: flex; align-items: center; gap: 12px; border-bottom: 2px solid #007bff; z-index: 100; box-shadow: 0 4px 15px rgba(0,0,0,0.5); flex-wrap: wrap; }
            #viewer { flex-grow: 1; border: none; width: 100%; height: 100%; }
            select { padding: 8px 12px; border-radius: 8px; background: #333; color: white; border: 1px solid #555; cursor: pointer; font-weight: bold; font-size: 13px; outline: none;}
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
