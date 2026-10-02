"""Bird's Eye View visualization for PufferDrive scenarios using Matplotlib."""

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

from pufferlib import replay_format
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import compute_effective_road_obs_count


COLORS = {
    "pedestrian": "#2E8B57",
    "cyclist": "#FF8C00",
    "road_line": "#808080",
    "road_edge": "#000000",
    "lane": "#D3D3D3",
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

VEHICLE_COLORS = replay_format.REPLAY_VIEW_STYLE["vehicle_colors"]

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

PAYLOAD_CHUNK_SIZE = 4 * 1024 * 1024
SIMULATOR_FIGSIZE = (20.0, 20.0)
SIMULATOR_DPI = 100
SIMULATOR_GOAL_RADIUS_METERS = 2.0
SIMULATOR_DEFAULT_RADIUS_METERS = 100.0


def _get_bounds(scenario):
    map_corners = scenario.get("map_corners")
    if map_corners and len(map_corners) >= 4:
        center_x = (map_corners[0] + map_corners[2]) / 2
        center_y = (map_corners[1] + map_corners[3]) / 2
        radius = max(map_corners[2] - map_corners[0], map_corners[3] - map_corners[1]) / 2 * 1.02
        return (center_x - radius, center_x + radius, center_y - radius, center_y + radius)
    radius = SIMULATOR_DEFAULT_RADIUS_METERS
    return (-radius, radius, -radius, radius)


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


def _obs_scales(
    env_cfg=None,
    obs_norm_goal_offset_m=100.0,
    obs_norm_xy_offset_m=100.0,
    obs_norm_veh_width_m=10.0,
    obs_norm_veh_length_m=15.0,
    obs_norm_road_seg_length_m=5.0,
):
    env_cfg = env_cfg or {}
    obs_norm_goal_offset_m = float(env_cfg.get("obs_norm_goal_offset_m", obs_norm_goal_offset_m))
    obs_norm_xy_offset_m = float(env_cfg.get("obs_norm_xy_offset_m", obs_norm_xy_offset_m))
    obs_norm_veh_width_m = float(env_cfg.get("obs_norm_veh_width_m", obs_norm_veh_width_m))
    obs_norm_veh_length_m = float(env_cfg.get("obs_norm_veh_length_m", obs_norm_veh_length_m))
    obs_norm_road_seg_length_m = float(env_cfg.get("obs_norm_road_seg_length_m", obs_norm_road_seg_length_m))
    inverse_xy_scale = None if obs_norm_xy_offset_m == 0 else 1.0 / obs_norm_xy_offset_m
    return {
        "obs_norm_goal_offset_m": obs_norm_goal_offset_m,
        "veh_width_to_position": 1.0 if inverse_xy_scale is None else obs_norm_veh_width_m * inverse_xy_scale,
        "veh_len_to_position": 1.0 if inverse_xy_scale is None else obs_norm_veh_length_m * inverse_xy_scale,
        "goal_to_position": 1.0 if inverse_xy_scale is None else obs_norm_goal_offset_m * inverse_xy_scale,
        "road_length_to_position": 1.0 if inverse_xy_scale is None else obs_norm_road_seg_length_m * inverse_xy_scale,
    }


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


def _render_agents(ax, agents, active_indices, static_indices, px_per_meter):
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
        color = VEHICLE_COLORS[agent_id % len(VEHICLE_COLORS)] if is_active else COLORS["inactive_agent"]
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

            text_items.append((x, y + width, str(agent_id)))

            if is_active:
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
        goal_patches = [Circle((x, y), radius=SIMULATOR_GOAL_RADIUS_METERS) for x, y in goal_points]
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
    road_data = _build_road_data(scenario.get("road_elements", []))
    traffic_data = _build_traffic_data(scenario.get("traffic_elements", []))

    bounds = _get_bounds(scenario)
    x_min, x_max, y_min, y_max = bounds

    px_per_meter = min(
        SIMULATOR_FIGSIZE[0] * SIMULATOR_DPI / (x_max - x_min),
        SIMULATOR_FIGSIZE[1] * SIMULATOR_DPI / (y_max - y_min),
    )

    fig, ax = plt.subplots()
    fig.set_size_inches(SIMULATOR_FIGSIZE)
    fig.set_dpi(SIMULATOR_DPI)
    fig.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    ax.set_aspect("equal")
    ax.set_title(
        f"PufferDrive | {scenario.get('dataset_name', '')} | {scenario.get('scenario_id', '')} | t={timestep}",
        fontsize=max(14, int(px_per_meter / 8)),
        fontweight="bold",
    )

    _render_roads(ax, road_data)
    _render_traffic(ax, traffic_data, timestep)

    _render_agents(
        ax,
        scenario.get("agents", []),
        scenario.get("active_agent_indices", []),
        scenario.get("static_agent_indices", []),
        px_per_meter,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return _img_from_fig(fig)


def _img_from_fig(fig: matplotlib.figure.Figure) -> np.ndarray:
    fig.subplots_adjust(left=0.01, bottom=0.02, right=1.00, top=0.96)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
    plt.close(fig)
    return img


def unpack_obs(
    obs_flat,
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
) -> np.ndarray:
    """Plot observation in ego-centric frame.

    Args:
        obs: flattened observation tensor
    """
    fig, ax = plt.subplots(figsize=(20, 20))

    ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_controls_obs = unpack_obs(
        obs,
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

    # Add dynamics info text for DYNAMICS_MODEL_JERK model
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
    count_lane = 0
    for i in range(lane_obs.shape[0]):
        if np.all(lane_obs[i] == 0):
            continue
        count_lane += 1
        rel_x, rel_y = lane_obs[i][0], lane_obs[i][1]
        length = lane_obs[i][3] * rl2p
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
        length = boundary_obs[i][3] * rl2p
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
    for name, arr in chunks.items():
        arr = np.ascontiguousarray(arr)
        dtype = replay_format.REPLAY_DTYPE_NAMES[arr.dtype]
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
    return zlib.compress(payload, level=3)


def encode_interactive_replay(scenario, replay):
    road_points = []
    road_points_z = []
    road_lengths = []
    road_types = []
    road_ids = []
    for road_idx, elem in enumerate(scenario.get("road_elements", []) or []):
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
        zs = elem.get("z") or [0.0] * count
        if len(zs) < count:
            raise ValueError(f"Road element {road_idx} has {len(zs)} z values for {count} points")
        road_lengths.append(count)
        road_types.append(draw_type)
        road_ids.append(int(elem.get("id", road_idx)))
        for i in range(count):
            road_points.append((float(xs[i]), float(ys[i])))
            road_points_z.append(float(zs[i]))

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

    observations = None
    obs_layout = None
    if replay.get("obs") is not None:
        observations = np.asarray(replay["obs"]).astype(np.float16, copy=False)
        non_finite_entries = np.argwhere(~np.isfinite(observations))
        if len(non_finite_entries):
            frame_idx, slot_idx, column_idx = non_finite_entries[0]
            raise ValueError(
                f"Replay observation is not finite at frame {frame_idx}, slot {slot_idx}, column {column_idx}"
            )
        obs_layout = replay_format.build_obs_layout(replay["obs_layout"])
        if obs_layout["obs_dim"] != observations.shape[-1]:
            raise ValueError(
                f"Replay observation layout sums to {obs_layout['obs_dim']} features, "
                f"but captured observations have {observations.shape[-1]}"
            )

    chunks = {
        "road_points": np.asarray(road_points or [(0.0, 0.0)], dtype=np.float32),
        "road_points_z": np.asarray(road_points_z or [0.0], dtype=np.float32),
        "road_lengths": np.asarray(road_lengths or [0], dtype=np.int32),
        "road_types": np.asarray(road_types or [0], dtype=np.int16),
        "road_ids": np.asarray(road_ids or [0], dtype=np.int32),
        "traffic_stop_lines": np.asarray(traffic_stop_lines or [[0, 0, 0, 0, 0, 0]], dtype=np.float32),
        "traffic_types": np.asarray(traffic_types or [0], dtype=np.int16),
        "agent_f32": replay["agent_f32"].astype(np.float32, copy=False),
        "agent_i32": replay["agent_i32"].astype(np.int32, copy=False),
        "metrics_f32": replay["metrics_f32"].astype(np.float32, copy=False),
        "puffer_f32": replay["puffer_f32"].astype(np.float32, copy=False),
        "traffic_i16": replay["traffic_i16"].astype(np.int16, copy=False),
        "raw_action": replay["raw_action"].astype(np.float32, copy=False),
        "clipped_action": replay["clipped_action"].astype(np.float32, copy=False),
        "value": replay["value"].astype(np.float32, copy=False),
        "entropy": replay["entropy"].astype(np.float32, copy=False),
    }
    if replay.get("goals_f32") is not None:
        chunks["goals_f32"] = replay["goals_f32"].astype(np.float32, copy=False)
    if replay.get("rewards_f32") is not None:
        chunks["rewards_f32"] = replay["rewards_f32"].astype(np.float32, copy=False)
    if replay.get("coefs_f32") is not None:
        chunks["coefs_f32"] = replay["coefs_f32"].astype(np.float32, copy=False)
    if observations is not None:
        chunks["obs"] = observations
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
    expert_indices = [agent_idx for agent_idx, agent in enumerate(agents) if int(agent.get("mark_as_expert", 0)) == 1]
    frame_count = int(replay["agent_f32"].shape[0])
    active_count = int(replay["raw_action"].shape[1])
    init_step = int(env_cfg.get("init_step", 0))
    ghost = np.zeros((frame_count, max(1, active_count), 5), dtype=np.float32)
    ghost_z = np.zeros((frame_count, max(1, active_count)), dtype=np.float32)
    for slot in range(min(active_count, len(active_indices))):
        agent_idx = active_indices[slot]
        if agent_idx < 0 or agent_idx >= len(agents):
            continue
        a = agents[agent_idx]
        lx = np.asarray(a.get("log_trajectory_x") or [], dtype=np.float32)
        ly = np.asarray(a.get("log_trajectory_y") or [], dtype=np.float32)
        lh = np.asarray(a.get("log_heading") or [], dtype=np.float32)
        lv = np.asarray(a.get("log_valid") or [], dtype=np.int32)
        lz = np.asarray(a.get("log_trajectory_z") or [], dtype=np.float32)
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
        if lz.shape[0] >= init_step + n:
            ghost_z[:n, slot] = lz[window]
    chunks["ghost_f32"] = ghost
    chunks["ghost_z_f32"] = ghost_z

    metadata = {
        "map_name": scenario.get("map_name", "Unknown"),
        "scenario_id": scenario.get("scenario_id", "Unknown"),
        "expert_indices": expert_indices,
        "total_agents": int(scenario.get("num_total_agents", replay["agent_f32"].shape[1])),
        "eval_overrides": replay.get("eval_overrides") or {},
        "frames": int(replay["agent_f32"].shape[0]),
        "agent_cap": int(replay["agent_f32"].shape[1]),
        "agent_goal_radius_field": int(binding.AGENT_F32_GOAL_RADIUS_IDX),
        "agent_path_field": int(binding.AGENT_F32_PATH_BASE_IDX),
        "agent_path_sample_count": int(binding.AGENT_F32_PATH_SAMPLES),
        "default_goal_radius_meters": float(env_cfg["goal_radius"]),
        "traffic_cap": int(replay["traffic_i16"].shape[1]),
        "active_count": int(replay["raw_action"].shape[1]),
        "obs_dim": int(observations.shape[2]) if observations is not None else 0,
        "replay_format_version": replay_format.REPLAY_FORMAT_VERSION,
        "obs_layout": obs_layout,
        "obs_norm_m": replay_format.obs_norm_from_env_config(env_cfg),
        "obs_range_m": replay_format.obs_range_from_env_config(env_cfg),
        "dt": float(env_cfg["dt"]),
        "action_type": env_cfg.get("action_type", "continuous"),
        "dynamics_model": env_cfg.get("dynamics_model", "classic"),
        "trajectory_baseline": bool(env_cfg.get("trajectory_baseline", False)),
        "num_goals": int(env_cfg["num_goals"]),
        "reward_conditioning": bool(env_cfg["reward_conditioning"]),
        "obs_slots_partners_n": int(env_cfg["obs_slots_partners_n"]),
        "ego_dim": int(binding.EGO_FEATURES)
        + (int(binding.SPLINE_INTENT_FEATURES) if env_cfg.get("action_type") == "spline" else 0)
        + (
            int(binding.LATTICE_PLAN_FEATURES) + int(bool(env_cfg.get("lattice_oncoming_overtake", False)))
            if env_cfg.get("action_type") == "lattice"
            else 0
        ),
        "reward_coef_count": int(binding.NUM_REWARD_COEFS),
        "partner_features": int(binding.PARTNER_FEATURES),
        "lane_features": int(binding.LANE_FEATURES),
        "boundary_features": int(binding.BOUNDARY_FEATURES),
        "traffic_features": int(binding.TRAFFIC_CONTROL_FEATURES),
        "lane_count": int(lane_count),
        "boundary_count": int(boundary_count),
        "traffic_obs_count": int(env_cfg["obs_slots_traffic_controls_n"]),
        "goal_features": int(binding.GOAL_FEATURES),
        "scales": scales,
        "road_polyline_count": len(road_lengths),
        "traffic_static_count": len(traffic_types),
    }
    return _pack_replay_binary(metadata, chunks)


def _render_interactive_replay_payload(compressed_payload, filename):
    payload = base64.b64encode(compressed_payload).decode("ascii")

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
            --accent:#0a66d0; --danger:#d6202c;
            --road:__MAP_ROAD_LIGHT__; --line:__MAP_LINE_LIGHT__; --edge:__MAP_EDGE_LIGHT__;
            --shadow:0 1px 2px rgba(22,26,34,.05),0 10px 30px rgba(22,26,34,.10);
            --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
        }
        [data-theme="dark"] {
            --bg:#0d0f12; --surface:rgba(23,26,31,.92); --surface-solid:#171a1f; --border:#2a2f37;
            --text:#e9ebef; --muted:#8c94a4; --field:rgba(140,148,164,.08);
            --accent:#4d9fff; --danger:#ff5560;
            --road:__MAP_ROAD_DARK__; --line:__MAP_LINE_DARK__; --edge:__MAP_EDGE_DARK__;
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
        #perturbation-legend { margin-top:10px; padding-top:8px; border-top:1px solid var(--border); }
        .perturbation-key { display:flex; align-items:center; gap:8px; margin-top:5px; color:var(--muted); font-size:10px; font-weight:600; }
        .perturbation-line { width:28px; height:0; border-top:3px solid; }
        .perturbation-line.blindness { border-color:#6d28d9; border-top-style:dashed; }
        .perturbation-line.phantom-braking { border-color:#b45309; }
        /* Agent panel: dark instrument-cluster surface in both themes — scoped variable overrides restyle all children. */
        #hud-telemetry { --border:#4a5468; --muted:#aab3c5; --field:rgba(255,255,255,.07); --accent:#7cbcff; position:absolute; top:14px; right:14px; width:372px; max-height:calc(100vh - 90px); padding:12px 14px; overflow-y:auto; display:none; background:rgba(54,62,77,.95); color:#eef1f6; }
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
        .cam-chip + .cam-chip { margin-left:4px; }
        .cam-chip.on { color:var(--accent); border-color:var(--accent); }
        #hud-telemetry.expanded { width:min(680px, 45vw); }
        #agent-view-box { display:none; position:sticky; top:0; z-index:2; margin:8px 0 4px; padding:6px; border:1px solid var(--border); border-radius:8px; background:#363e4d; }
        #agent-view-canvas { display:block; width:100%; aspect-ratio:16 / 9; border-radius:5px; background:#cfe0f0; }
        .av-tools { display:flex; gap:6px; justify-content:flex-end; margin-top:5px; }
        #view-pill { position:absolute; top:14px; left:50%; transform:translateX(-50%); display:none; padding:5px 12px; border:1px solid var(--border); border-radius:14px; background:var(--surface-solid); color:var(--text); box-shadow:var(--shadow); font-size:11px; font-weight:600; letter-spacing:.04em; white-space:nowrap; }
    </style>
</head>
<body>
    <div id="loading-overlay"><div class="spinner"></div><div id="load-text">Decoding replay&#8230;</div></div>
    <div id="ui-layer">
        <div id="view-pill"></div>
        <div id="hud-global" class="panel collapsed">
            <h3 onclick="toggleGlobalPanel()">Scenario <span id="globalChevron" style="float:right">&#9656;</span></h3>
            <div class="label">Map</div><div class="value" id="meta-map">-</div>
            <div class="label">Scenario ID</div><div class="value mono" id="meta-id" style="font-size:11px">-</div>
            <div class="label">Agents (active / total)</div><div class="value mono" id="meta-agents">-</div>
            <div id="perturbation-legend">
                <div class="label">Active perturbations</div>
                <div class="perturbation-key"><span class="perturbation-line blindness"></span><span>Partner blindness</span></div>
                <div class="perturbation-key"><span class="perturbation-line phantom-braking"></span><span>Phantom braking</span></div>
            </div>
            <button type="button" class="toggle-header is-collapsed" id="overrides-header" data-target="overrides-body"><span>Eval overrides</span><span>&#9662;</span></button>
            <div id="overrides-body" class="grid toggle-body is-collapsed"></div>
            <button class="btn" onclick="toggleTheme()" style="width:100%; margin-top:12px">Toggle theme</button>
        </div>
        <div id="hud-telemetry" class="panel">
            <h3 id="tel-drag-handle">Agent <span id="tel-id" class="highlight mono">?</span><button type="button" id="camMode" class="cam-chip" onclick="toggleCamMode()">world cam</button><button type="button" id="agentViewBtn" class="cam-chip" onclick="toggleAgentView()">agent view</button></h3>
            <div id="agent-view-box"><canvas id="agent-view-canvas"></canvas><div class="av-tools"><button type="button" id="agentViewPresetBtn" class="obs-tool" onclick="cycleAgentViewPreset()">chase</button><button type="button" id="agentViewExpandBtn" class="obs-tool" onclick="toggleAgentViewSize()">expand</button></div></div>
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
            <button type="button" id="reward-header" class="toggle-header" data-target="reward-grid"><span>Reward</span><span>&#9662;</span></button>
            <div id="reward-grid" class="grid toggle-body"></div>
            <button type="button" id="coef-header" class="toggle-header is-collapsed" data-target="coef-grid"><span>Conditioning</span><span>&#9662;</span></button>
            <div id="coef-grid" class="grid toggle-body is-collapsed"></div>
            <button type="button" class="toggle-header" data-target="puffer-score-body"><span>Puffer score</span><span>&#9662;</span></button>
            <div id="puffer-score-body" class="toggle-body"><div id="tel-ps" class="score-num">0.000</div></div>
            <button type="button" class="toggle-header" data-target="puffer-grid"><span>Puffer metrics</span><span>&#9662;</span></button>
            <div id="puffer-grid" class="grid toggle-body"></div>
            <button type="button" class="toggle-header" data-target="metrics-grid"><span>Metrics</span><span>&#9662;</span></button>
            <div id="metrics-grid" class="grid toggle-body"></div>
        </div>
        <div id="obs-container" class="panel"><div id="obs-title"><span id="obs-title-text">Ego-centric observation</span><button type="button" class="obs-tool" onclick="resetObsZoom(event)">1x</button><button type="button" id="obsModeBtn" class="obs-tool" onclick="toggleObsMode(event)">BOTH</button><button type="button" class="obs-tool" onclick="toggleObsSize(event)">Expand</button></div><canvas id="obs-canvas"></canvas></div>
        <div id="controls" class="panel">
            <button id="btnPlay" class="btn icon" onclick="toggle()"></button>
            <span class="mono step-counter"><span id="stepNow">0</span><span class="dim"> / </span><span id="stepTotal">0</span></span>
            <input id="sld" type="range" min="0" value="0" step="1">
            <select id="speedSel" onchange="changeSpeed()"><option value="0.25">0.25x</option><option value="1">1x</option><option value="2">2x</option><option value="4" selected>4x</option><option value="8">8x</option></select>
            <input type="number" id="agentSearch" placeholder="agent id" onkeydown="if(event.key==='Enter') searchAgent()">
        </div>
    </div>
    <canvas id="c"></canvas>
__PAYLOAD_CHUNKS__
    <script>
        const METRIC_LABELS = __METRIC_LABELS__;
        const VEHICLE_COLORS = __VEHICLE_COLORS__;
        const VIEW_STYLE = __REPLAY_VIEW_STYLE__;
        // Order must match the Log fields written in env_binding.h vec_get_obs_html_frame (15 values).
        const PUFFER_LABELS = ["score","no at fault","no offroad","no red light","progress > .2","direction","ttc","progress ratio","speed limit","comfort","multi lane","wrong way dist","speed violation","multiplier","weighted avg"];
        // Order must match the REWARD_COEF_* indices in constants.h.
        const COEF_LABELS = ["goal radius","goal speed","collision","offroad","comfort","lane align","vel align","lane center","center bias","velocity","reverse","stop line","timestep","overspeed","throttle","steer","acc","speed"];
        const REWARD_LABELS = ["collision","offroad","red light","stop sign","goal","lane align","lane center","comfort","velocity","timestep","reverse","overspeed","ADE"];
        const ACCEL = [-4,-2.667,-1.333,0,1.333,2.667,4], STEER = [-0.667,-0.5,-0.333,-0.167,0,0.167,0.333,0.5,0.667];
        const JLONG = [-15,-4,0,4], JLAT = [-4,0,4];
        const DYNAMIC_EXPERT_COLOR = VIEW_STYLE.agent_colors.dynamic_expert;
        const STATIC_AGENT_COLOR = VIEW_STYLE.agent_colors.static;
        const INFRACTION_AGENT_COLOR = VIEW_STYLE.agent_colors.infraction;
        const PARTNER_BLINDNESS_OUTLINE_COLOR = "#6d28d9";
        const PHANTOM_BRAKING_OUTLINE_COLOR = "#b45309";
        const PREDICTED_PATH_COLOR = VIEW_STYLE.predicted_path_color;
        const INFRACTION_METRIC_COUNT = VIEW_STYLE.infraction_metric_count;
        const VIEW_PILL_FLASH_MS = 2500;
        const DEFAULT_GOAL_RADIUS_METERS = 2;
        const SVG_PLAY = '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M4.5 2.5v11l9-5.5z" fill="currentColor"/></svg>';
        const SVG_PAUSE = '<svg viewBox="0 0 16 16" width="13" height="13"><path d="M4 2.5h3v11H4zM9 2.5h3v11H9z" fill="currentColor"/></svg>';
        let H, C = {}, F, paths = {0:new Path2D(),1:new Path2D(),2:new Path2D()}, roadPathsById = new Map(), lastDrawn = -1;
        const c = document.getElementById('c'), ctx = c.getContext('2d');
        const obsC = document.getElementById('obs-canvas'), obsCtx = obsC.getContext('2d');
        const avC = document.getElementById('agent-view-canvas'), avCtx = avC.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        let step = 0, play = false, speed = 4, lastTick = 0;
        let cam = {x:0,y:0,z:5,drag:false,lx:0,ly:0};
        let followedId = null, isEgoCam = false, darkMode = false, showGhost = false, showPredictedPath = false;
        let obsZoom = VIEW_STYLE.obs_panel_default_zoom, obsExpanded = false, obsMode = 2;
        let agentViewOn = false, agentViewPreset = 'chase', agentViewExpanded = false, observedOnly = false;
        let visibleAgentIdx = null, visibleAgentSlots = null, pillFlashUntil = 0;
        let roadSegs = new Float32Array(0), roadSegCount = 0, groundPoints = new Float64Array(0), groundGrid = new Map();
        let obsRowCache = {key:null, row:null}, observedCache = {key:null, value:null};
        const HALF_FLOAT_TABLE = buildHalfFloatTable();
        let expertAgentIndices = new Set();
        const OBS_MODES = ["ALL","POOL","BOTH"];

        function chunk(name) {
            const m = H.chunks[name], start = H.dataStart + m.offset, n = m.nbytes / ({float32:4,int32:4,int16:2,uint8:1,float16:2}[m.dtype]);
            if (m.dtype === "float32") return new Float32Array(H.buffer, start, n);
            if (m.dtype === "int32") return new Int32Array(H.buffer, start, n);
            if (m.dtype === "int16") return new Int16Array(H.buffer, start, n);
            if (m.dtype === "float16") return new Uint16Array(H.buffer, start, n);
            return new Uint8Array(H.buffer, start, n);
        }
        function frameMax() { return Math.max(0, (H ? H.frames : 1) - 1); }
        async function decodeReplayPayload() {
            const nodes = Array.from(document.querySelectorAll('.payload-chunk'));
            if (!nodes.length) throw new Error('Missing replay payload');
            const workerCode = `
let parts = [];
function decodeBase64(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
}
self.onmessage = async event => {
    const message = event.data;
    if (message.type === 'chunk') {
        parts.push(decodeBase64(message.data));
        self.postMessage({type:'progress', done:message.index + 1, total:message.total});
        return;
    }
    if (message.type === 'end') {
        try {
            const stream = new DecompressionStream('deflate');
            const buffer = await new Response(new Blob(parts).stream().pipeThrough(stream)).arrayBuffer();
            parts = null;
            self.postMessage({type:'done', buffer:buffer}, [buffer]);
        } catch (error) {
            self.postMessage({type:'error', message:String(error && error.message ? error.message : error)});
        }
    }
};
`;
            return await new Promise((resolve, reject) => {
                const workerUrl = URL.createObjectURL(new Blob([workerCode], {type:'text/javascript'}));
                const worker = new Worker(workerUrl);
                const loadText = document.getElementById('load-text');
                let chunkIdx = 0;

                function closeWorker() {
                    worker.terminate();
                    URL.revokeObjectURL(workerUrl);
                }
                function sendNext() {
                    if (chunkIdx >= nodes.length) {
                        loadText.textContent = 'Inflating replay...';
                        worker.postMessage({type:'end'});
                        return;
                    }
                    const node = nodes[chunkIdx];
                    worker.postMessage({
                        type:'chunk',
                        data:node.textContent,
                        index:chunkIdx,
                        total:nodes.length,
                    });
                    node.textContent = '';
                    chunkIdx += 1;
                }

                worker.onmessage = event => {
                    const message = event.data;
                    if (message.type === 'progress') {
                        loadText.textContent = 'Reading replay ' + message.done + ' / ' + message.total;
                        setTimeout(sendNext, 0);
                        return;
                    }
                    if (message.type === 'done') {
                        closeWorker();
                        for (const node of nodes) node.remove();
                        resolve(message.buffer);
                        return;
                    }
                    if (message.type === 'error') {
                        closeWorker();
                        reject(new Error(message.message));
                    }
                };
                worker.onerror = event => {
                    closeWorker();
                    reject(new Error(event.message || 'Replay worker failed'));
                };
                sendNext();
            });
        }
        async function initReplay() {
            const buf = await decodeReplayPayload();
            const view = new DataView(buf), headerLen = view.getUint32(0, true);
            H = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
            H.buffer = buf; H.dataStart = 4 + headerLen + ((-(4 + headerLen)) & 3);
            for (const name of Object.keys(H.chunks)) C[name] = chunk(name);
            F = {af:H.chunks.agent_f32.shape[2], ai:H.chunks.agent_i32.shape[2], mf:H.chunks.metrics_f32.shape[2], pf:H.chunks.puffer_f32.shape[2], tf:H.chunks.traffic_i16.shape[2], gf:H.chunks.goals_f32 ? H.chunks.goals_f32.shape[2] : 0, rf:H.chunks.rewards_f32 ? H.chunks.rewards_f32.shape[2] : 0, cf:H.chunks.coefs_f32 ? H.chunks.coefs_f32.shape[2] : 0};
            expertAgentIndices = new Set(H.expert_indices);
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
            requestAnimationFrame(() => { buildMapPaths(); buildGroundGrid(); buildRoadSegments(); draw(true); });
        }
        initReplay().catch(err => { console.error(err); document.getElementById('load-text').textContent = 'Replay load failed. See console.'; });

        function buildMapPaths() {
            paths = {0:new Path2D(),1:new Path2D(),2:new Path2D()};
            roadPathsById = new Map();
            let p = 0;
            for (let i=0;i<H.road_polyline_count;i++) {
                const len = C.road_lengths[i], type = C.road_types[i], path = paths[type], roadId = C.road_ids[i];
                if (len <= 0) continue;
                const roadPath = new Path2D();
                path.moveTo(C.road_points[p*2], C.road_points[p*2+1]);
                roadPath.moveTo(C.road_points[p*2], C.road_points[p*2+1]);
                for (let j=1;j<len;j++) {
                    path.lineTo(C.road_points[(p+j)*2], C.road_points[(p+j)*2+1]);
                    roadPath.lineTo(C.road_points[(p+j)*2], C.road_points[(p+j)*2+1]);
                }
                roadPathsById.set(roadId, roadPath);
                p += len;
            }
        }
        function colorFor(id, isActive, isExpert) {
            if (isActive) return VEHICLE_COLORS[Math.abs(id) % VEHICLE_COLORS.length];
            return isExpert ? DYNAMIC_EXPERT_COLOR : STATIC_AGENT_COLOR;
        }
        function colorForAgent(id, isActive, isExpert, hasInfraction) {
            return hasInfraction ? INFRACTION_AGENT_COLOR : colorFor(id, isActive, isExpert);
        }
        function agentHasInfraction(frame, idx) {
            const metricsBase = (frame * H.agent_cap + idx) * F.mf;
            for (let metricIdx=0;metricIdx<INFRACTION_METRIC_COUNT;metricIdx++) {
                if (C.metrics_f32[metricsBase+metricIdx] > 0) return true;
            }
            return false;
        }
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
        function drawPerturbationOutline(a, color, dashPixels, paddingPixels) {
            const padding = paddingPixels / cam.z;
            ctx.save();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.5 / cam.z;
            ctx.setLineDash(dashPixels.map(length => length / cam.z));
            if (a.type === 2) {
                ctx.beginPath();
                ctx.arc(0, 0, Math.max(a.w, 0.7) / 2 + padding, 0, 7);
            } else {
                rr(-a.l/2-padding, -a.w/2-padding, a.l+2*padding, a.w+2*padding, Math.min(a.w*0.30, a.l*0.10)+padding);
            }
            ctx.stroke();
            ctx.restore();
        }
        function drawPerturbationOutlines(a) {
            if (a.phantomBrakingActive) {
                drawPerturbationOutline(a, PHANTOM_BRAKING_OUTLINE_COLOR, [], 3);
            }
            if (a.partnerBlindnessActive) {
                drawPerturbationOutline(a, PARTNER_BLINDNESS_OUTLINE_COLOR, [6, 4], a.phantomBrakingActive ? 7 : 3);
            }
        }
        function agentAt(frame, idx) {
            // Light decode (no metric copies) — called for every agent every frame; metrics/puffer read on demand.
            const ib = (frame * H.agent_cap + idx) * F.ai;
            if (!C.agent_i32[ib+2]) return null;
            const fb = (frame * H.agent_cap + idx) * F.af;
            const agentType = C.agent_i32[ib+1], isActive = C.agent_i32[ib+3] === 1;
            const isExpert = expertAgentIndices.has(idx);
            const hasInfraction = agentType === 1 && agentHasInfraction(frame, idx);
            const agentColor = colorForAgent(C.agent_i32[ib], isActive, isExpert, hasInfraction);
            const goalRadiusField = H.agent_goal_radius_field;
            const defaultGoalRadius = H.default_goal_radius_meters || DEFAULT_GOAL_RADIUS_METERS;
            const goalRadius = Number.isInteger(goalRadiusField) && goalRadiusField < F.af ? C.agent_f32[fb+goalRadiusField] : defaultGoalRadius;
            return {idx:idx, id:C.agent_i32[ib], type:agentType, cl:C.agent_i32[ib+6], slot:C.agent_i32[ib+7], partnerBlindnessActive:C.agent_i32[ib+8] === 1, phantomBrakingActive:C.agent_i32[ib+9] === 1, x:C.agent_f32[fb], y:C.agent_f32[fb+1], z:C.agent_f32[fb+2], h:C.agent_f32[fb+3], l:C.agent_f32[fb+4], w:C.agent_f32[fb+5], s:C.agent_f32[fb+6], st:C.agent_f32[fb+7], al:C.agent_f32[fb+8], alat:C.agent_f32[fb+9], jl:C.agent_f32[fb+10], jlat:C.agent_f32[fb+11], goalRadius:goalRadius, c:agentColor};
        }
        function getFrameAgents(frame) { const out = []; for (let i=0;i<H.agent_cap;i++) { const a = agentAt(frame, i); if (a) out.push(a); } return out; }
        function drawGhosts(f) {
            if (!showGhost || !C.ghost_f32) return;
            const N = H.chunks.ghost_f32.shape[1];
            ctx.strokeStyle = '#ff0000'; ctx.fillStyle = 'rgba(255,0,0,.22)'; ctx.lineWidth = .28; ctx.setLineDash([.6,.4]);
            for (let j=0;j<N;j++) { const b=(f*N+j)*5, w=C.ghost_f32[b+4]; if (w <= 0 || (visibleAgentSlots && !visibleAgentSlots.has(j))) continue; ctx.save(); ctx.translate(C.ghost_f32[b], C.ghost_f32[b+1]); ctx.rotate(C.ghost_f32[b+2]); ctx.beginPath(); ctx.rect(-C.ghost_f32[b+3]/2, -w/2, C.ghost_f32[b+3], w); ctx.fill(); ctx.stroke(); ctx.restore(); }
            ctx.setLineDash([]);
        }
        function strokeAgentPath(f, slot, width) {
            const base = (f * H.agent_cap + slot) * F.af + H.agent_path_field, n = H.agent_path_sample_count;
            if (C.agent_f32[base] === 0 && C.agent_f32[base+1] === 0) return;
            const path = new Path2D();
            path.moveTo(C.agent_f32[base], C.agent_f32[base+1]);
            for (let k=1;k<n;k++) path.lineTo(C.agent_f32[base+2*k], C.agent_f32[base+2*k+1]);
            ctx.save();
            ctx.strokeStyle = PREDICTED_PATH_COLOR; ctx.globalAlpha = VIEW_STYLE.predicted_path_alpha; ctx.lineWidth = Math.max(width, .1); ctx.lineCap = 'butt';
            ctx.stroke(path);
            ctx.restore();
        }
        function drawPredictedPath(f) {
            // lattice runs show every policy agent's committed plan by default (P toggles it off)
            const lattice = H.action_type === "lattice";
            if (!(lattice ? !showPredictedPath : showPredictedPath) || !(lattice || H.action_type === "spline" || H.trajectory_baseline)) return;
            if (lattice) {
                for (let i=0;i<H.agent_cap;i++) { const a = agentAt(f, i); if (a && (!visibleAgentIdx || visibleAgentIdx.has(i))) strokeAgentPath(f, i, a.w); }
                return;
            }
            const ego = agentAt(f, 0); // EGO_IDX = 0, matches constants.h
            if (!ego || (visibleAgentIdx && !visibleAgentIdx.has(0))) return;
            strokeAgentPath(f, 0, ego.w);
        }
        function findAgent(frame, id) { for (let i=0;i<H.agent_cap;i++) { const a = agentAt(frame, i); if (a && a.id === id) return a; } return null; }
        function trafficAt(frame, idx) {
            const db = (frame * H.traffic_cap + idx) * F.tf;
            if (!C.traffic_i16[db]) return null;
            const sb = idx * 6, type = C.traffic_types[idx] || C.traffic_i16[db+1], state = C.traffic_i16[db+2];
            return {type, state, stop_line:Array.from(C.traffic_stop_lines.subarray(sb, sb + 6))};
        }
        function trafficColor(t) { return VIEW_STYLE.traffic_state_colors[String(t.state)] || VIEW_STYLE.traffic_default_color; }
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
        function resetObsZoom(e){ if(e) e.stopPropagation(); obsZoom=VIEW_STYLE.obs_panel_default_zoom; draw(true); }
        function toggleObsMode(e){ if(e) e.stopPropagation(); obsMode=(obsMode+1)%OBS_MODES.length; document.getElementById('obsModeBtn').textContent=OBS_MODES[obsMode]; draw(true); }
        function toggleObsSize(e){ if(e) e.stopPropagation(); const p=document.getElementById('obs-container'), b=e ? e.currentTarget : null; obsExpanded=!obsExpanded; p.style.width=obsExpanded?'680px':'390px'; p.style.height=obsExpanded?'680px':'390px'; if(b) b.textContent=obsExpanded?'Collapse':'Expand'; resizeObsCanvas(); draw(true); }
        function searchAgent(){ const id=parseInt(document.getElementById('agentSearch').value); if(!isNaN(id)){ followedId=id; play=false; updateBtn(); draw(true); } }
        document.addEventListener('keydown', e => { if(!H || e.target.tagName === 'INPUT') return; if(e.code === 'Space'){ toggle(); e.preventDefault(); } if(e.code === 'ArrowRight'){ play=false; updateBtn(); step=Math.min(step+1,frameMax()); draw(true); } if(e.code === 'ArrowLeft'){ play=false; updateBtn(); step=Math.max(step-1,0); draw(true); } if(e.code === 'Escape'){ followedId=null; isEgoCam=false; observedOnly=false; updateUI(); draw(true); } if(e.code === 'KeyV'){ toggleObservedOnly(); } if(e.code === 'KeyG'){ showGhost=!showGhost; draw(true); } if(e.code === 'KeyP' && (H.action_type === 'spline' || H.action_type === 'lattice' || H.trajectory_baseline)){ showPredictedPath=!showPredictedPath; draw(true); } });
        c.onwheel = e => { e.preventDefault(); cam.z *= Math.exp(-e.deltaY * .001); draw(true); };
        c.onmousedown = e => { if(!H) return; const r=c.getBoundingClientRect(), wx=(e.clientX-r.left-c.width/2)/cam.z+cam.x, wy=(e.clientY-r.top-c.height/2)/-cam.z+cam.y; let hit=null, agents=getFrameAgents(Math.floor(step)); if(!isEgoCam) for(const a of agents) if((!visibleAgentIdx || visibleAgentIdx.has(a.idx)) && Math.hypot(wx-a.x, wy-a.y) < Math.max(a.l,3)){ hit=a.id; break; } if(hit !== null){ followedId=hit; cam.drag=false; } else { followedId=null; isEgoCam=false; observedOnly=false; cam.drag=true; cam.lx=e.clientX; cam.ly=e.clientY; } draw(true); };
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
            // World-frame goals from the replay log, so goals render without captured observations.
            if (!C.goals_f32 || !agent || agent.slot < 0) return [];
            const base = (frame * H.agent_cap + agent.idx) * F.gf, stride = F.gf / H.num_goals, goals = C.goals_f32, out = [];
            for (let i=0;i<H.num_goals;i++) {
                const o = base + i * stride;
                if (goals[o] === 0 && goals[o+1] === 0) continue;
                out.push({x:goals[o], y:goals[o+1], radius:agent.goalRadius});
            }
            return out;
        }
        function strokeLanePath(laneId, color, width) {
            const path = roadPathsById.get(laneId);
            if (!path) return;
            ctx.strokeStyle = color;
            ctx.lineWidth = width;
            ctx.stroke(path);
        }
        function drawCurrentLane(agent, colors) {
            if (!agent || agent.cl < 0) return;
            ctx.save();
            ctx.lineCap = 'round';
            strokeLanePath(agent.cl, 'rgba(255,255,255,.35)', Math.max(.5, 5.0/cam.z));
            strokeLanePath(agent.cl, colors.accent, Math.max(.35, 4.0/cam.z));
            ctx.restore();
        }
        function decodeObs(frame, slot) {
            if (!obsAvailable() || slot < 0 || slot >= H.active_count) return null;
            const obs = obsRow(frame, slot), layout = H.obs_layout, LF = layout.lane_features, BF = layout.boundary_features, TF = layout.traffic_features, PF = layout.partner_features, GF = layout.goal_features;
            const rot = (x,y) => [-y,x];
            const zero = (off,n) => { for(let i=0;i<n;i++) if(obs[off+i] !== 0) return false; return true; };
            const roads = (start,count,poolName,feat) => { const out=[]; for(let i=0;i<count;i++){ const o=start+i*feat; if(zero(o,feat)) continue; let xy=rot(obs[o],obs[o+1]), cs=rot(obs[o+5],obs[o+6]); out.push([xy[0],xy[1],2*obs[o+3]*H.scales.road_length_to_position,cs[0],cs[1],poolAt(poolName,frame,slot,i)]); } return out; };
            const partners = []; for(let i=0;i<layout.partner_count;i++){ const o=layout.partner_start+i*PF; if(zero(o,PF)) continue; let xy=rot(obs[o],obs[o+1]), h=Math.atan2(obs[o+6],obs[o+5]); h = ((h + Math.PI/2 + Math.PI) % (2*Math.PI)) - Math.PI; partners.push({x:xy[0],y:xy[1],l:obs[o+3]*H.scales.veh_len_to_position,w:obs[o+4]*H.scales.veh_width_to_position,h:h,pool:poolAt("pool_partner",frame,slot,i)}); }
            const gps = []; for(let i=0;i<layout.goal_count;i++){ const o=layout.goal_start+i*GF; if(zero(o,GF)) continue; gps.push(rot(obs[o]*H.scales.goal_to_position, obs[o+1]*H.scales.goal_to_position)); }
            const controls = []; for(let i=0;i<layout.traffic_count;i++){ const o=layout.traffic_start+i*TF; if(zero(o,TF)) continue; let a=rot(obs[o],obs[o+1]), b=rot(obs[o+2],obs[o+3]); controls.push({type:Math.round(obs[o+5]), state:Math.round(obs[o+6]), x1:a[0], y1:a[1], x2:b[0], y2:b[1], pool:poolAt("pool_traffic",frame,slot,i)}); }
            return {ego:{w:obs[1]*H.scales.veh_width_to_position,l:obs[2]*H.scales.veh_len_to_position}, partners, lanes:roads(layout.lane_start,layout.lane_count,"pool_lane",LF), bounds:roads(layout.boundary_start,layout.boundary_count,"pool_boundary",BF), gps, traffic_controls:controls};
        }
        function drawObs(frame) {
            resizeObsCanvas();
            const scale = (Math.min(obsC.width, obsC.height) / 2) * obsZoom, px = dpr / scale;
            const showAll = obsMode !== 1, showPool = obsMode !== 0, bothMode = obsMode === 2;
            let poolMax = 1;
            for(const r of frame.lanes) if(r[5] > poolMax) poolMax = r[5];
            for(const r of frame.bounds) if(r[5] > poolMax) poolMax = r[5];
            for(const p of frame.partners) if(p.pool > poolMax) poolMax = p.pool;
            for(const t of frame.traffic_controls) if(t.pool > poolMax) poolMax = t.pool;
            const pw = t => (2.0 + 2.4*t)*px;
            obsCtx.fillStyle = "#fff"; obsCtx.fillRect(0,0,obsC.width,obsC.height);
            obsCtx.save(); obsCtx.translate(obsC.width/2, obsC.height/2); obsCtx.scale(scale, -scale); obsCtx.lineCap = "round";
            if(showAll){ obsCtx.strokeStyle=bothMode?"#000":"#bbb"; obsCtx.lineWidth=1.5*px; for(const r of frame.lanes){ obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[3]*r[2]/2,r[1]+r[4]*r[2]/2); obsCtx.lineTo(r[0]-r[3]*r[2]/2,r[1]-r[4]*r[2]/2); obsCtx.stroke(); } }
            if(showAll){ obsCtx.strokeStyle=bothMode?"#000":"#333"; obsCtx.lineWidth=3*px; for(const r of frame.bounds){ obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[3]*r[2]/2,r[1]+r[4]*r[2]/2); obsCtx.lineTo(r[0]-r[3]*r[2]/2,r[1]-r[4]*r[2]/2); obsCtx.stroke(); } }
            if(showPool){ for(const r of frame.lanes.concat(frame.bounds)){ if(r[5] > 0){ obsCtx.strokeStyle=poolColor(r[5]/poolMax); obsCtx.lineWidth=pw(r[5]/poolMax); obsCtx.beginPath(); obsCtx.moveTo(r[0]+r[3]*r[2]/2,r[1]+r[4]*r[2]/2); obsCtx.lineTo(r[0]-r[3]*r[2]/2,r[1]-r[4]*r[2]/2); obsCtx.stroke(); } } }
            for(const g of frame.gps){ obsCtx.fillStyle="magenta"; obsCtx.beginPath(); obsCtx.arc(g[0],g[1],5*px,0,7); obsCtx.fill(); }
            for(const t of frame.traffic_controls){ if(showAll){ obsCtx.strokeStyle = bothMode ? "#000" : (t.type === 1 ? trafficColor({state:t.state}) : (t.type === 2 ? "#cc0000" : "#ffd700")); obsCtx.lineWidth=2.5*px; obsCtx.beginPath(); obsCtx.moveTo(t.x1,t.y1); obsCtx.lineTo(t.x2,t.y2); obsCtx.stroke(); } if(showPool && t.pool > 0){ obsCtx.strokeStyle=poolColor(t.pool/poolMax); obsCtx.lineWidth=pw(t.pool/poolMax)+0.8*px; obsCtx.beginPath(); obsCtx.moveTo(t.x1,t.y1); obsCtx.lineTo(t.x2,t.y2); obsCtx.stroke(); } }
            for(const p of frame.partners){ const win = showPool && p.pool > 0; if(!showAll && !win) continue; obsCtx.save(); obsCtx.translate(p.x,p.y); obsCtx.rotate(p.h); if(showAll){ obsCtx.fillStyle=bothMode?"rgba(0,0,0,.55)":"rgba(136,136,136,.8)"; obsCtx.strokeStyle=bothMode?"#000":"#333"; obsCtx.lineWidth=1.5*px; obsCtx.beginPath(); obsCtx.rect(-p.l/2,-p.w/2,p.l,p.w); obsCtx.fill(); obsCtx.stroke(); } if(win){ obsCtx.strokeStyle=poolColor(p.pool/poolMax); obsCtx.lineWidth=pw(p.pool/poolMax); obsCtx.strokeRect(-p.l/2,-p.w/2,p.l,p.w); } obsCtx.restore(); }
            if(frame.ego){ obsCtx.save(); obsCtx.rotate(Math.PI/2); obsCtx.fillStyle="rgba(0,102,255,.8)"; obsCtx.strokeStyle="#000"; obsCtx.lineWidth=1.5*px; obsCtx.beginPath(); obsCtx.rect(-frame.ego.l/2,-frame.ego.w/2,frame.ego.l,frame.ego.w); obsCtx.fill(); obsCtx.stroke(); obsCtx.restore(); }
            obsCtx.restore();
            if(showPool && poolMax > 1) drawPoolLegend(poolMax);
        }
        function buildHalfFloatTable() {
            const table = new Float64Array(65536);
            for (let bits=0; bits<65536; bits++) {
                const sign = (bits & 0x8000) ? -1 : 1, exponent = (bits >> 10) & 0x1f, mantissa = bits & 0x3ff;
                if (exponent === 0) table[bits] = sign * Math.pow(2, -14) * (mantissa / 1024);
                else if (exponent === 31) table[bits] = mantissa ? NaN : sign * Infinity;
                else table[bits] = sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
            }
            return table;
        }
        function obsAvailable() { return !!C.obs && H.replay_format_version >= 2 && !!H.obs_layout; }
        function obsRow(frame, slot) {
            const key = frame * H.active_count + slot;
            if (obsRowCache.key === key) return obsRowCache.row;
            const dim = H.obs_dim, base = key * dim, row = new Float64Array(dim);
            for (let i=0;i<dim;i++) row[i] = HALF_FLOAT_TABLE[C.obs[base+i]];
            obsRowCache = {key:key, row:row};
            return row;
        }
        function decodeObservedWorld(frame, agent) {
            if (!agent || agent.slot < 0 || agent.slot >= H.active_count || !obsAvailable()) return null;
            const key = frame * H.agent_cap + agent.idx;
            if (observedCache.key === key) return observedCache.value;
            const row = obsRow(frame, agent.slot), layout = H.obs_layout, norm = H.obs_norm_m;
            const cosH = Math.cos(agent.h), sinH = Math.sin(agent.h);
            const worldX = (lx, ly) => agent.x + cosH * lx - sinH * ly, worldY = (lx, ly) => agent.y + sinH * lx + cosH * ly;
            const occupied = (o, n) => { for (let i=0;i<n;i++) if (row[o+i] !== 0) return true; return false; };
            const relativeZ = value => agent.z + value * (norm.z || 0);
            const segments = (start, count, features, heights) => {
                const out = [];
                for (let i=0;i<count;i++) {
                    const o = start + i * features;
                    if (!occupied(o, features)) continue;
                    const midX = row[o] * norm.xy_offset, midY = row[o+1] * norm.xy_offset, half = row[o+3] * norm.road_seg_length;
                    const startX = midX - row[o+5] * half, startY = midY - row[o+6] * half, endX = midX + row[o+5] * half, endY = midY + row[o+6] * half;
                    out.push([worldX(startX, startY), worldY(startX, startY), worldX(endX, endY), worldY(endX, endY)]);
                    heights.push(relativeZ(row[o+2]));
                }
                return out;
            };
            const decodedPartners = [];
            for (let i=0;i<layout.partner_count;i++) {
                const o = layout.partner_start + i * layout.partner_features;
                if (!occupied(o, layout.partner_features)) continue;
                const lx = row[o] * norm.xy_offset, ly = row[o+1] * norm.xy_offset;
                decodedPartners.push({slot:i, x:worldX(lx, ly), y:worldY(lx, ly), z:relativeZ(row[o+2]), h:agent.h + Math.atan2(row[o+6], row[o+5]), l:row[o+3] * norm.veh_length, w:row[o+4] * norm.veh_width});
            }
            const pairs = [];
            for (let i=0;i<H.agent_cap;i++) {
                const other = i === agent.idx ? null : agentAt(frame, i);
                if (!other) continue;
                for (const partner of decodedPartners) {
                    const distanceM = Math.hypot(partner.x - other.x, partner.y - other.y);
                    if (distanceM <= VIEW_STYLE.partner_match_tolerance_m) pairs.push([distanceM, partner.slot, i]);
                }
            }
            pairs.sort((p, q) => p[0] - q[0] || p[1] - q[1] || p[2] - q[2]);
            const usedSlots = new Set(), usedAgents = new Set(), partnerAgentIdx = [];
            for (const pair of pairs) {
                if (usedSlots.has(pair[1]) || usedAgents.has(pair[2])) continue;
                usedSlots.add(pair[1]); usedAgents.add(pair[2]); partnerAgentIdx.push(pair[2]);
            }
            const stopLines = [];
            for (let i=0;i<layout.traffic_count;i++) {
                const o = layout.traffic_start + i * layout.traffic_features;
                if (!occupied(o, layout.traffic_features)) continue;
                const x0 = row[o] * norm.xy_offset, y0 = row[o+1] * norm.xy_offset, x1 = row[o+2] * norm.xy_offset, y1 = row[o+3] * norm.xy_offset;
                stopLines.push({x0:worldX(x0, y0), y0:worldY(x0, y0), z0:relativeZ(row[o+4]), x1:worldX(x1, y1), y1:worldY(x1, y1), z1:relativeZ(row[o+4]), type:Math.round(row[o+5]), state:Math.round(row[o+6])});
            }
            const laneZ = [], boundZ = [];
            const value = {
                lanes: segments(layout.lane_start, layout.lane_count, layout.lane_features, laneZ),
                bounds: segments(layout.boundary_start, layout.boundary_count, layout.boundary_features, boundZ),
                laneZ: laneZ,
                boundZ: boundZ,
                stopLines: stopLines,
                partnerAgentIdx: partnerAgentIdx,
                unmatchedPartners: decodedPartners.filter(p => !usedSlots.has(p.slot)).map(p => ({x:p.x, y:p.y, z:p.z, h:p.h, l:p.l, w:p.w})),
                partnerBlind: agent.partnerBlindnessActive,
            };
            observedCache = {key:key, value:value};
            return value;
        }
        function setPill(text) {
            if (!text && performance.now() < pillFlashUntil) return;
            const pill = document.getElementById('view-pill');
            pill.textContent = text || '';
            pill.style.display = text ? 'block' : 'none';
        }
        function flashPill(text) {
            pillFlashUntil = 0;
            setPill(text);
            pillFlashUntil = performance.now() + VIEW_PILL_FLASH_MS;
            setTimeout(() => { pillFlashUntil = 0; draw(true); }, VIEW_PILL_FLASH_MS + 10);
        }
        function missingObservationsText() { return C.obs ? 're-render replay: old format' : 'V needs eval.capture_observations=true'; }
        function observedSetFor(frame, target) {
            if (!observedOnly) { setPill(null); return null; }
            if (!obsAvailable()) { setPill(missingObservationsText()); return null; }
            if (!target) { setPill('agent ' + followedId + ' not present'); return null; }
            if (target.slot < 0) { setPill('agent ' + target.id + ' has no observation'); return null; }
            const observed = decodeObservedWorld(frame, target);
            setPill('observed by agent ' + target.id + ' (V)' + (observed.partnerBlind ? ' · partner-blind' : ''));
            return observed;
        }
        function toggleObservedOnly() {
            if (!H) return;
            if (observedOnly) { observedOnly = false; draw(true); return; }
            if (followedId === null) { flashPill('V: select an agent first'); return; }
            if (!obsAvailable()) { flashPill(missingObservationsText()); return; }
            const target = findAgent(Math.max(0, Math.min(frameMax(), Math.floor(step))), followedId);
            if (!target) { flashPill('agent ' + followedId + ' not present'); return; }
            if (target.slot < 0) { flashPill('agent ' + target.id + ' has no observation'); return; }
            observedOnly = true;
            draw(true);
        }
        function drawObservedMap(observed, colors) {
            ctx.lineCap = 'round';
            ctx.strokeStyle = colors.road; ctx.lineWidth = .5; ctx.beginPath();
            for (const s of observed.lanes) { ctx.moveTo(s[0], s[1]); ctx.lineTo(s[2], s[3]); }
            ctx.stroke();
            ctx.strokeStyle = colors.edge; ctx.lineWidth = .8; ctx.beginPath();
            for (const s of observed.bounds) { ctx.moveTo(s[0], s[1]); ctx.lineTo(s[2], s[3]); }
            ctx.stroke();
        }
        function drawObservedStopLines(observed) {
            for (const t of observed.stopLines) {
                ctx.lineCap = 'butt';
                if (t.type === 1) { ctx.strokeStyle = trafficColor(t); ctx.lineWidth = Math.min(1.5, 3/cam.z); }
                else { ctx.strokeStyle = t.type === 2 ? VIEW_STYLE.stop_sign_color : VIEW_STYLE.yield_sign_color; ctx.lineWidth = Math.min(1.2, 2.5/cam.z); ctx.setLineDash([6/cam.z, 4/cam.z]); }
                ctx.beginPath(); ctx.moveTo(t.x0, t.y0); ctx.lineTo(t.x1, t.y1); ctx.stroke(); ctx.setLineDash([]);
            }
        }
        function drawUnmatchedPartners(observed, colors) {
            ctx.save(); ctx.strokeStyle = colors.text; ctx.lineWidth = 1.5/cam.z; ctx.setLineDash([4/cam.z, 3/cam.z]);
            for (const p of observed.unmatchedPartners) { ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.h); ctx.strokeRect(-p.l/2, -p.w/2, p.l, p.w); ctx.restore(); }
            ctx.restore();
        }
        function toggleAgentView() {
            agentViewOn = !agentViewOn;
            document.getElementById('agent-view-box').style.display = agentViewOn ? 'block' : 'none';
            document.getElementById('agentViewBtn').classList.toggle('on', agentViewOn);
            draw(true);
        }
        function cycleAgentViewPreset() {
            agentViewPreset = agentViewPreset === 'chase' ? 'driver' : 'chase';
            document.getElementById('agentViewPresetBtn').textContent = agentViewPreset;
            draw(true);
        }
        function toggleAgentViewSize() {
            agentViewExpanded = !agentViewExpanded;
            document.getElementById('hud-telemetry').classList.toggle('expanded', agentViewExpanded);
            document.getElementById('agentViewExpandBtn').textContent = agentViewExpanded ? 'collapse' : 'expand';
            draw(true);
        }
        function roadPointZ(pointIdx) { return C.road_points_z ? C.road_points_z[pointIdx] : 0; }
        function buildRoadSegments() {
            let total = 0;
            for (let i=0;i<H.road_polyline_count;i++) total += Math.max(0, C.road_lengths[i] - 1);
            roadSegs = new Float32Array(total * 8); roadSegCount = 0;
            let p = 0;
            for (let i=0;i<H.road_polyline_count;i++) {
                const len = C.road_lengths[i], type = C.road_types[i];
                if (len <= 0) continue;
                for (let j=1;j<len;j++) {
                    const o = roadSegCount * 8;
                    roadSegs[o] = C.road_points[(p+j-1)*2]; roadSegs[o+1] = C.road_points[(p+j-1)*2+1]; roadSegs[o+2] = roadPointZ(p+j-1);
                    roadSegs[o+3] = C.road_points[(p+j)*2]; roadSegs[o+4] = C.road_points[(p+j)*2+1]; roadSegs[o+5] = roadPointZ(p+j); roadSegs[o+6] = type;
                    roadSegs[o+7] = elevationBand((roadSegs[o] + roadSegs[o+3]) / 2, (roadSegs[o+1] + roadSegs[o+4]) / 2, (roadSegs[o+2] + roadSegs[o+5]) / 2);
                    roadSegCount++;
                }
                p += len;
            }
        }
        function buildGroundGrid() {
            // lane centrelines only, as the sim does for car heights; road edges and lines sit at other heights
            const lanePoints = [];
            let p = 0;
            for (let i=0;i<H.road_polyline_count;i++) {
                const len = C.road_lengths[i];
                if (len <= 0) continue;
                if (C.road_types[i] === 0) for (let j=0;j<len;j++) lanePoints.push(p + j);
                p += len;
            }
            const cellM = VIEW_STYLE.ground_grid_cell_m;
            groundPoints = new Float64Array(lanePoints.length * 3); groundGrid = new Map();
            for (let i=0;i<lanePoints.length;i++) {
                const x = C.road_points[lanePoints[i]*2], y = C.road_points[lanePoints[i]*2+1];
                groundPoints[i*3] = x; groundPoints[i*3+1] = y; groundPoints[i*3+2] = roadPointZ(lanePoints[i]);
                const key = Math.floor(x / cellM) + ',' + Math.floor(y / cellM);
                if (!groundGrid.has(key)) groundGrid.set(key, []);
                groundGrid.get(key).push(i);
            }
        }
        function groundHeightM(x, y, referenceZ) {
            // average of the nearest lane points on the level closest to referenceZ, so stacked roads never mix
            const cellM = VIEW_STYLE.ground_grid_cell_m, radiusM = VIEW_STYLE.ground_lookup_radius_m, toleranceM = VIEW_STYLE.ground_level_tolerance_m, reach = Math.ceil(radiusM / cellM);
            const cellX = Math.floor(x / cellM), cellY = Math.floor(y / cellM), candidates = [];
            for (let dx=-reach; dx<=reach; dx++) for (let dy=-reach; dy<=reach; dy++) {
                const bucket = groundGrid.get((cellX + dx) + ',' + (cellY + dy));
                if (!bucket) continue;
                for (const i of bucket) {
                    const ddx = groundPoints[i*3] - x, ddy = groundPoints[i*3+1] - y, d2 = ddx * ddx + ddy * ddy;
                    if (d2 < radiusM * radiusM) candidates.push([d2, groundPoints[i*3+2]]);
                }
            }
            if (!candidates.length) return referenceZ;
            let levelZ = referenceZ;
            if (!candidates.some(c => Math.abs(c[1] - referenceZ) <= toleranceM)) {
                let closest = candidates[0];
                for (const c of candidates) if (Math.abs(c[1] - referenceZ) < Math.abs(closest[1] - referenceZ)) closest = c;
                levelZ = closest[1];
            }
            const level = candidates.filter(c => Math.abs(c[1] - levelZ) <= toleranceM).sort((p, q) => p[0] - q[0]).slice(0, VIEW_STYLE.ground_average_point_count);
            let sumZ = 0;
            for (const c of level) sumZ += c[1];
            return sumZ / level.length;
        }
        function elevationBand(x, y, z) {
            const cellM = VIEW_STYLE.ground_grid_cell_m, radiusM = VIEW_STYLE.overpass_search_radius_m, belowZ = z - VIEW_STYLE.overpass_clearance_m, reach = Math.ceil(radiusM / cellM);
            const cellX = Math.floor(x / cellM), cellY = Math.floor(y / cellM);
            for (let dx=-reach; dx<=reach; dx++) for (let dy=-reach; dy<=reach; dy++) {
                const bucket = groundGrid.get((cellX + dx) + ',' + (cellY + dy));
                if (!bucket) continue;
                for (const i of bucket) {
                    const ddx = groundPoints[i*3] - x, ddy = groundPoints[i*3+1] - y;
                    if (ddx * ddx + ddy * ddy <= radiusM * radiusM && groundPoints[i*3+2] <= belowZ) return 1;
                }
            }
            return 0;
        }
        function agentViewCamera(agent, presetName, widthPx, heightPx) {
            const preset = VIEW_STYLE.cameras[presetName];
            if (preset.back_m + preset.ahead_m <= 0) throw new Error('agent view camera needs back_m + ahead_m > 0');
            const headingX = Math.cos(agent.h), headingY = Math.sin(agent.h), baseZ = agent.z || 0;
            const eye = [agent.x - preset.back_m * headingX, agent.y - preset.back_m * headingY, baseZ + preset.eye_z_m];
            const lookAt = [agent.x + preset.ahead_m * headingX, agent.y + preset.ahead_m * headingY, baseZ + preset.target_z_m];
            const delta = [lookAt[0] - eye[0], lookAt[1] - eye[1], lookAt[2] - eye[2]];
            const deltaNorm = Math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
            const forward = [delta[0] / deltaNorm, delta[1] / deltaNorm, delta[2] / deltaNorm];
            const rightNorm = Math.sqrt(forward[1] * forward[1] + forward[0] * forward[0]);
            const right = [forward[1] / rightNorm, -forward[0] / rightNorm, 0];
            const up = [right[1] * forward[2] - right[2] * forward[1], right[2] * forward[0] - right[0] * forward[2], right[0] * forward[1] - right[1] * forward[0]];
            const focalLengthPx = (heightPx / 2) / Math.tan(preset.fovy_deg * Math.PI / 360);
            const horizonYPx = heightPx / 2 - focalLengthPx * (up[0] * headingX + up[1] * headingY) / (forward[0] * headingX + forward[1] * headingY);
            return {eye:eye, forward:forward, right:right, up:up, focalLengthPx:focalLengthPx, widthPx:widthPx, heightPx:heightPx, horizonYPx:horizonYPx};
        }
        function camDepth(view, x, y, z) { return (x - view.eye[0]) * view.forward[0] + (y - view.eye[1]) * view.forward[1] + (z - view.eye[2]) * view.forward[2]; }
        function camProject(view, x, y, z) {
            const dx = x - view.eye[0], dy = y - view.eye[1], dz = z - view.eye[2];
            const cameraX = dx * view.right[0] + dy * view.right[1] + dz * view.right[2];
            const cameraY = dx * view.up[0] + dy * view.up[1] + dz * view.up[2];
            const depthM = dx * view.forward[0] + dy * view.forward[1] + dz * view.forward[2];
            return [view.widthPx / 2 + view.focalLengthPx * cameraX / depthM, view.heightPx / 2 - view.focalLengthPx * cameraY / depthM, depthM];
        }
        function clipPolygonNear(view, points, nearM) {
            const clipped = [];
            for (let i=0;i<points.length;i++) {
                const a = points[i], b = points[(i+1) % points.length], depthA = camDepth(view, a[0], a[1], a[2]), depthB = camDepth(view, b[0], b[1], b[2]);
                if (depthA >= nearM) clipped.push(a);
                if ((depthA >= nearM) === (depthB >= nearM)) continue;
                const t = (nearM - depthA) / (depthB - depthA);
                clipped.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]);
            }
            return clipped.length >= 3 ? clipped : [];
        }
        function clipSegmentNear(view, a, b, nearM) {
            const depthA = camDepth(view, a[0], a[1], a[2]), depthB = camDepth(view, b[0], b[1], b[2]);
            if (depthA < nearM && depthB < nearM) return null;
            if (depthA >= nearM && depthB >= nearM) return [a, b];
            const t = (nearM - depthA) / (depthB - depthA), crossing = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
            return depthA < nearM ? [crossing, b] : [a, crossing];
        }
        function appendClippedPolygon(view, points, nearM) {
            const clipped = clipPolygonNear(view, points, nearM);
            for (let i=0;i<clipped.length;i++) { const p = camProject(view, clipped[i][0], clipped[i][1], clipped[i][2]); if (i === 0) avCtx.moveTo(p[0], p[1]); else avCtx.lineTo(p[0], p[1]); }
            if (clipped.length) avCtx.closePath();
            return clipped.length > 0;
        }
        function appendClippedSegment(view, a, b, nearM) {
            const segment = clipSegmentNear(view, a, b, nearM);
            if (!segment) return false;
            const start = camProject(view, segment[0][0], segment[0][1], segment[0][2]), end = camProject(view, segment[1][0], segment[1][1], segment[1][2]);
            avCtx.moveTo(start[0], start[1]); avCtx.lineTo(end[0], end[1]);
            return true;
        }
        function viewSegments(observed, drawType, target) {
            const out = [], rangeM = VIEW_STYLE.cull_range_m;
            if (observed) {
                const source = drawType === 0 ? observed.lanes : (drawType === 2 ? observed.bounds : []), heights = drawType === 0 ? observed.laneZ : observed.boundZ;
                for (let i=0;i<source.length;i++) { const s = source[i]; if (Math.hypot((s[0] + s[2]) / 2 - target.x, (s[1] + s[3]) / 2 - target.y) <= rangeM) out.push([s[0], s[1], heights[i], s[2], s[3], heights[i], elevationBand((s[0] + s[2]) / 2, (s[1] + s[3]) / 2, heights[i])]); }
                return out;
            }
            for (let i=0;i<roadSegCount;i++) {
                const o = i * 8;
                if (roadSegs[o+6] !== drawType || Math.hypot((roadSegs[o] + roadSegs[o+3]) / 2 - target.x, (roadSegs[o+1] + roadSegs[o+4]) / 2 - target.y) > rangeM) continue;
                out.push([roadSegs[o], roadSegs[o+1], roadSegs[o+2], roadSegs[o+3], roadSegs[o+4], roadSegs[o+5], roadSegs[o+7]]);
            }
            return out;
        }
        function trailRuns(frame, agent) {
            const frames = H.dt ? Math.round(VIEW_STYLE.trail_seconds / H.dt) : 0, runs = [];
            let run = [];
            for (let k=Math.max(0, frame - frames); k<=frame; k++) {
                const ib = (k * H.agent_cap + agent.idx) * F.ai, fb = (k * H.agent_cap + agent.idx) * F.af;
                const alive = C.agent_i32[ib+2] === 1 && C.agent_i32[ib+5] === 0, x = C.agent_f32[fb], y = C.agent_f32[fb+1], z = C.agent_f32[fb+2];
                const jumped = run.length > 0 && Math.hypot(x - run[run.length-1][0], y - run[run.length-1][1]) > VIEW_STYLE.trail_break_distance_m;
                if (!alive || jumped) { if (run.length > 1) runs.push(run); run = []; }
                if (alive) run.push([x, y, z]);
            }
            if (run.length > 1) runs.push(run);
            return runs;
        }
        function loggedFutureRuns(frame, agent) {
            if (!C.ghost_f32 || agent.slot < 0 || agent.slot >= H.chunks.ghost_f32.shape[1]) return [];
            const slots = H.chunks.ghost_f32.shape[1], frames = H.dt ? Math.round(VIEW_STYLE.logged_future_seconds / H.dt) : 0, runs = [];
            let run = [];
            for (let k=frame; k<Math.min(H.frames, frame + frames + 1); k++) {
                const b = (k * slots + agent.slot) * 5;
                if (C.ghost_f32[b+4] <= 0) { if (run.length > 1) runs.push(run); run = []; continue; }
                run.push([C.ghost_f32[b], C.ghost_f32[b+1], C.ghost_z_f32 ? C.ghost_z_f32[k * slots + agent.slot] : 0]);
            }
            if (run.length > 1) runs.push(run);
            return runs;
        }
        function agentPathPoints(frame, agent) {
            const base = (frame * H.agent_cap + agent.idx) * F.af + H.agent_path_field, n = H.agent_path_sample_count;
            if (C.agent_f32[base] === 0 && C.agent_f32[base+1] === 0) return null;
            const out = [];
            for (let k=0;k<n;k++) out.push([C.agent_f32[base+2*k], C.agent_f32[base+2*k+1]]);
            return out;
        }
        function elevatedPath(path, startZ) {
            // each sample takes the road height on the level nearest the previous sample, so bridges keep the car's level
            const out = [];
            let previousZ = startZ;
            for (const point of path) { previousZ = groundHeightM(point[0], point[1], previousZ); out.push([point[0], point[1], previousZ]); }
            return out;
        }
        function resamplePolyline(points, pieceM) {
            const out = [points[0]];
            for (let i=1;i<points.length;i++) {
                const a = points[i-1], b = points[i], pieces = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / pieceM));
                for (let k=1;k<=pieces;k++) { const t = k / pieces; out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]); }
            }
            return out;
        }
        function ribbonQuads(points, halfWidthM) {
            const normals = [];
            for (let i=0;i<points.length;i++) {
                const prev = points[Math.max(0, i-1)], next = points[Math.min(points.length-1, i+1)];
                const dx = next[0] - prev[0], dy = next[1] - prev[1], len = Math.hypot(dx, dy);
                normals.push(len > 0 ? [-dy / len, dx / len] : [0, 0]);
            }
            const quads = [];
            for (let i=1;i<points.length;i++) {
                const a = points[i-1], b = points[i], na = normals[i-1], nb = normals[i];
                if (Math.hypot(b[0] - a[0], b[1] - a[1]) <= 0) continue;
                quads.push([[a[0] + na[0] * halfWidthM, a[1] + na[1] * halfWidthM, a[2]], [b[0] + nb[0] * halfWidthM, b[1] + nb[1] * halfWidthM, b[2]], [b[0] - nb[0] * halfWidthM, b[1] - nb[1] * halfWidthM, b[2]], [a[0] - na[0] * halfWidthM, a[1] - na[1] * halfWidthM, a[2]]]);
            }
            return quads;
        }
        function hexAlpha(hex, alpha) { return 'rgba(' + parseInt(hex.slice(1,3),16) + ',' + parseInt(hex.slice(3,5),16) + ',' + parseInt(hex.slice(5,7),16) + ',' + alpha + ')'; }
        function shadeColor(hex, normal) {
            const light = VIEW_STYLE.box_light_direction, lightNorm = Math.sqrt(light[0]*light[0] + light[1]*light[1] + light[2]*light[2]);
            const shade = VIEW_STYLE.box_ambient + VIEW_STYLE.box_diffuse * Math.max(0, (normal[0]*light[0] + normal[1]*light[1] + normal[2]*light[2]) / lightNorm);
            const channel = offset => Math.min(255, Math.round(parseInt(hex.slice(offset, offset + 2), 16) * shade));
            return 'rgb(' + channel(1) + ',' + channel(3) + ',' + channel(5) + ')';
        }
        function boxHeightM(agentType) { return agentType === 2 ? VIEW_STYLE.box_height_m.pedestrian : (agentType === 3 ? VIEW_STYLE.box_height_m.cyclist : VIEW_STYLE.box_height_m.vehicle); }
        function boxChunks(a, heightM) {
            const fx = Math.cos(a.h), fy = Math.sin(a.h), lx = -Math.sin(a.h), ly = Math.cos(a.h), halfW = a.w / 2, baseZ = a.z || 0;
            const chunkCount = Math.max(1, Math.ceil(a.l / VIEW_STYLE.sort_chunk_length_m)), chunkLen = a.l / chunkCount, out = [];
            const at = (along, side, z) => [a.x + fx * along + lx * side, a.y + fy * along + ly * side, baseZ + z];
            for (let k=0;k<chunkCount;k++) {
                const rear = -a.l / 2 + k * chunkLen, front = rear + chunkLen, mid = (rear + front) / 2;
                const faces = [
                    {corners:[at(rear, halfW, heightM), at(rear, -halfW, heightM), at(front, -halfW, heightM), at(front, halfW, heightM)], normal:[0, 0, 1], center:at(mid, 0, heightM)},
                    {corners:[at(rear, halfW, 0), at(front, halfW, 0), at(front, halfW, heightM), at(rear, halfW, heightM)], normal:[lx, ly, 0], center:at(mid, halfW, heightM / 2)},
                    {corners:[at(front, -halfW, 0), at(rear, -halfW, 0), at(rear, -halfW, heightM), at(front, -halfW, heightM)], normal:[-lx, -ly, 0], center:at(mid, -halfW, heightM / 2)},
                ];
                if (k === 0) faces.push({corners:[at(rear, -halfW, 0), at(rear, halfW, 0), at(rear, halfW, heightM), at(rear, -halfW, heightM)], normal:[-fx, -fy, 0], center:at(rear, 0, heightM / 2)});
                if (k === chunkCount - 1) faces.push({corners:[at(front, halfW, 0), at(front, -halfW, 0), at(front, -halfW, heightM), at(front, halfW, heightM)], normal:[fx, fy, 0], center:at(front, 0, heightM / 2)});
                out.push({centroid:at(mid, 0, heightM / 2), faces:faces});
            }
            return out;
        }
        function agentViewPrimitives(frame, target, observed, view) {
            // painter's list per elevation band (bridge decks after the roads beneath); within a band ground layers sort before cars
            const colors = VIEW_STYLE.agent_view_colors, lineWidths = VIEW_STYLE.agent_view_line_width_px, bias = VIEW_STYLE.agent_view_sort_bias_m;
            const pieceM = VIEW_STYLE.agent_view_piece_length_m, halfLaneM = VIEW_STYLE.lane_surface_width_m / 2, primitives = [];
            const distanceM = (x, y, z) => Math.hypot(x - view.eye[0], y - view.eye[1], z - view.eye[2]);
            const pushLine = (a, b, color, widthPx, sortBias, band) => primitives.push({kind:'line', band:band, key:distanceM((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2) + sortBias, a:a, b:b, stroke:color, strokeWidth:widthPx});
            const bandAt = (a, b) => elevationBand((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
            for (const s of viewSegments(observed, 0, target)) {
                const dx = s[3] - s[0], dy = s[4] - s[1], dz = s[5] - s[2], len = Math.hypot(dx, dy);
                if (len <= 0) continue;
                const nx = -dy / len * halfLaneM, ny = dx / len * halfLaneM, pieces = Math.max(1, Math.ceil(len / pieceM));
                for (let k=0;k<pieces;k++) {
                    const t0 = k / pieces, t1 = (k + 1) / pieces;
                    const ax = s[0] + dx * t0, ay = s[1] + dy * t0, az = s[2] + dz * t0, bx = s[0] + dx * t1, by = s[1] + dy * t1, bz = s[2] + dz * t1;
                    primitives.push({kind:'polygon', band:s[6], key:distanceM((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2) + bias.surface, points:[[ax - nx, ay - ny, az], [bx - nx, by - ny, bz], [bx + nx, by + ny, bz], [ax + nx, ay + ny, az]], fill:colors.road, stroke:colors.road, strokeWidth:dpr});
                }
            }
            for (const lineType of [[1, colors.line, lineWidths.road_line], [2, colors.edge, lineWidths.edge]]) {
                for (const s of viewSegments(observed, lineType[0], target)) {
                    const pieces = Math.max(1, Math.ceil(Math.hypot(s[3] - s[0], s[4] - s[1]) / pieceM));
                    for (let k=0;k<pieces;k++) {
                        const t0 = k / pieces, t1 = (k + 1) / pieces;
                        pushLine([s[0] + (s[3] - s[0]) * t0, s[1] + (s[4] - s[1]) * t0, s[2] + (s[5] - s[2]) * t0], [s[0] + (s[3] - s[0]) * t1, s[1] + (s[4] - s[1]) * t1, s[2] + (s[5] - s[2]) * t1], lineType[1], lineType[2] * dpr, bias.line, s[6]);
                    }
                }
            }
            const controls = observed ? observed.stopLines : [];
            if (!observed) for (let i=0;i<H.traffic_static_count;i++) { const t = trafficAt(frame, i); if (t) controls.push({x0:t.stop_line[0], y0:t.stop_line[1], z0:t.stop_line[2], x1:t.stop_line[3], y1:t.stop_line[4], z1:t.stop_line[5], type:t.type, state:t.state}); }
            for (const t of controls) {
                if (Math.hypot((t.x0 + t.x1) / 2 - target.x, (t.y0 + t.y1) / 2 - target.y) > VIEW_STYLE.cull_range_m) continue;
                const color = t.type === 1 ? trafficColor(t) : (t.type === 2 ? VIEW_STYLE.stop_sign_color : VIEW_STYLE.yield_sign_color);
                pushLine([t.x0, t.y0, t.z0], [t.x1, t.y1, t.z1], color, lineWidths.stop_line * dpr, bias.stop_line, bandAt([t.x0, t.y0, t.z0], [t.x1, t.y1, t.z1]));
            }
            const vertexCount = VIEW_STYLE.goal_polygon_vertex_count;
            for (const g of selectedGoals(frame, target)) {
                const goalZ = groundHeightM(g.x, g.y, target.z), polygon = [];
                for (let k=0;k<vertexCount;k++) polygon.push([g.x + g.radius * Math.cos(k * 2 * Math.PI / vertexCount), g.y + g.radius * Math.sin(k * 2 * Math.PI / vertexCount), goalZ]);
                primitives.push({kind:'polygon', band:elevationBand(g.x, g.y, goalZ), key:distanceM(g.x, g.y, goalZ) + bias.goal, points:polygon, fill:hexAlpha(VIEW_STYLE.goal_color, VIEW_STYLE.goal_fill_alpha), stroke:VIEW_STYLE.goal_color, strokeWidth:dpr});
            }
            const trajectoryWidth = lineWidths.trajectory * dpr;
            for (const run of trailRuns(frame, target)) for (let i=1;i<run.length;i++) pushLine(run[i-1], run[i], hexAlpha(target.c, VIEW_STYLE.trail_alpha), trajectoryWidth, bias.trajectory, bandAt(run[i-1], run[i]));
            for (const run of loggedFutureRuns(frame, target)) for (let i=1;i<run.length;i++) pushLine(run[i-1], run[i], hexAlpha(VIEW_STYLE.logged_future_color, VIEW_STYLE.logged_future_alpha), trajectoryWidth, bias.trajectory, bandAt(run[i-1], run[i]));
            const path = agentPathPoints(frame, target);
            if (path) {
                for (const quad of ribbonQuads(resamplePolyline(elevatedPath(path, target.z), pieceM), target.w / 2)) {
                    const center = [(quad[0][0] + quad[2][0]) / 2, (quad[0][1] + quad[2][1]) / 2, (quad[0][2] + quad[2][2]) / 2];
                    primitives.push({kind:'polygon', band:elevationBand(center[0], center[1], center[2]), key:distanceM(center[0], center[1], center[2]) + bias.trajectory, points:quad, fill:hexAlpha(VIEW_STYLE.predicted_path_color, VIEW_STYLE.predicted_path_alpha)});
                }
            }
            const drawEgo = VIEW_STYLE.cameras[agentViewPreset].draw_ego;
            for (let i=0;i<H.agent_cap;i++) {
                const a = agentAt(frame, i);
                if (!a || (a.idx === target.idx && !drawEgo) || (visibleAgentIdx && !visibleAgentIdx.has(a.idx))) continue;
                if (Math.hypot(a.x - target.x, a.y - target.y) > VIEW_STYLE.cull_range_m) continue;
                const band = elevationBand(a.x, a.y, a.z || 0);
                for (const chunk of boxChunks(a, boxHeightM(a.type))) primitives.push({kind:'box', band:band, key:distanceM(chunk.centroid[0], chunk.centroid[1], chunk.centroid[2]), faces:chunk.faces, color:a.c});
            }
            const heightM = VIEW_STYLE.box_height_m.vehicle;
            for (const p of (observed ? observed.unmatchedPartners : [])) {
                const fx = Math.cos(p.h), fy = Math.sin(p.h), halfL = p.l / 2, halfW = p.w / 2;
                const footprint = [[-halfL, halfW], [halfL, halfW], [halfL, -halfW], [-halfL, -halfW]].map(q => [p.x + fx * q[0] - fy * q[1], p.y + fy * q[0] + fx * q[1]]);
                for (let k=0;k<4;k++) {
                    const a = footprint[k], b = footprint[(k+1) % 4];
                    const band = elevationBand(p.x, p.y, p.z);
                    pushLine([a[0], a[1], p.z], [b[0], b[1], p.z], colors.box_outline, 1.5 * dpr, 0, band);
                    pushLine([a[0], a[1], p.z + heightM], [b[0], b[1], p.z + heightM], colors.box_outline, 1.5 * dpr, 0, band);
                    pushLine([a[0], a[1], p.z], [a[0], a[1], p.z + heightM], colors.box_outline, 1.5 * dpr, 0, band);
                }
            }
            primitives.sort((p, q) => (p.band - q.band) || (q.key - p.key));
            return primitives;
        }
        function drawAgentView(frame, target, observed) {
            const rect = avC.getBoundingClientRect();
            if (rect.width <= 0) return;
            const aspect = VIEW_STYLE.agent_view_aspect, widthPx = Math.max(2, Math.floor(rect.width * dpr)), heightPx = Math.max(2, Math.floor(widthPx * aspect[1] / aspect[0]));
            if (avC.width !== widthPx || avC.height !== heightPx) { avC.width = widthPx; avC.height = heightPx; }
            const view = agentViewCamera(target, agentViewPreset, avC.width, avC.height), colors = VIEW_STYLE.agent_view_colors, nearM = VIEW_STYLE.near_plane_m;
            avCtx.setTransform(1, 0, 0, 1, 0, 0);
            avCtx.fillStyle = colors.ground; avCtx.fillRect(0, 0, avC.width, avC.height);
            avCtx.fillStyle = colors.sky; avCtx.fillRect(0, 0, avC.width, Math.min(Math.max(view.horizonYPx, 0), avC.height));
            avCtx.lineCap = 'round'; avCtx.lineJoin = 'round';
            for (const p of agentViewPrimitives(frame, target, observed, view)) {
                if (p.kind === 'line') {
                    avCtx.beginPath();
                    if (!appendClippedSegment(view, p.a, p.b, nearM)) continue;
                    avCtx.strokeStyle = p.stroke; avCtx.lineWidth = p.strokeWidth; avCtx.stroke();
                    continue;
                }
                if (p.kind === 'polygon') {
                    avCtx.beginPath();
                    if (!appendClippedPolygon(view, p.points, nearM)) continue;
                    avCtx.fillStyle = p.fill; avCtx.fill();
                    if (p.stroke) { avCtx.strokeStyle = p.stroke; avCtx.lineWidth = p.strokeWidth; avCtx.stroke(); }
                    continue;
                }
                for (const face of p.faces) {
                    const toEye = [view.eye[0] - face.center[0], view.eye[1] - face.center[1], view.eye[2] - face.center[2]];
                    if (toEye[0] * face.normal[0] + toEye[1] * face.normal[1] + toEye[2] * face.normal[2] <= 0) continue;
                    avCtx.beginPath();
                    if (!appendClippedPolygon(view, face.corners, nearM)) continue;
                    avCtx.fillStyle = shadeColor(p.color, face.normal); avCtx.fill();
                    avCtx.strokeStyle = colors.box_outline; avCtx.lineWidth = dpr; avCtx.stroke();
                }
            }
        }
        let panelKey = null, refs = null, lastWarnKey = "";
        function ensurePanels() {
            // Panel structure is identical across agents/frames — build the DOM once, update textContent per frame.
            // keyed on captured probs: a discrete policy on the continuous env still records them
            const discrete = !!C.policy_probs && H.action_type !== "lattice";
            const actionDims = H.chunks.raw_action.shape.length > 2 ? H.chunks.raw_action.shape[2] : 1;
            const key = (discrete ? 'd' : 'c') + actionDims;
            if (refs && panelKey === key) return;
            panelKey = key;
            const mg = document.getElementById('metrics-grid');
            mg.innerHTML = METRIC_LABELS.map(l=>`<div class="item"><span class="name">${l}</span><span class="num">-</span></div>`).join('');
            const pg = document.getElementById('puffer-grid');
            pg.innerHTML = PUFFER_LABELS.map(l=>`<div class="item"><span class="name">${l}</span><span class="num">-</span></div>`).join('');
            const rg = document.getElementById('reward-grid');
            const rewardLabels = ["return (cum)","total (step)"].concat(REWARD_LABELS.slice(0, Math.max(0, F.rf - 1)));
            rg.innerHTML = C.rewards_f32 ? rewardLabels.map(l=>`<div class="item"><span class="name">${l}</span><span class="num">-</span></div>`).join('') : '';
            document.getElementById('reward-header').style.display = C.rewards_f32 ? '' : 'none';
            const cg = document.getElementById('coef-grid');
            const coefLabels = COEF_LABELS.slice(0, F.cf);
            cg.innerHTML = C.coefs_f32 ? coefLabels.map(l=>`<div class="item"><span class="name">${l}</span><span class="num">-</span></div>`).join('') : '';
            document.getElementById('coef-header').style.display = C.coefs_f32 ? '' : 'none';
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
                labels = H.action_type === "continuous" ? (H.dynamics_model === "jerk" ? ["jerk_long","jerk_lat"] : ["accel","steer"]) : H.action_type === "lattice" ? ["lat gate","lat cell","lon gate","lon cell","exit slot"] : Array.from({length:actionDims}, (_,i)=>`p${i}`);
                labels.forEach(l => html += `<div class="item"><span class="name">${l}</span><span class="num pol-act">-</span></div>`);
                if (C.policy_mean) { labels.forEach(l => html += `<div class="item"><span class="name">mean ${l}</span><span class="num pol-mean">-</span></div><div class="item"><span class="name">std ${l}</span><span class="num pol-std">-</span></div>`); html += '<div class="item"><span class="name">log prob</span><span class="num" data-pol="lp">-</span></div>'; }
            }
            pol.innerHTML = html;
            refs = {
                metric: [...mg.querySelectorAll('.num')],
                puffer: [...pg.querySelectorAll('.num')],
                rewardCells: [...rg.querySelectorAll('.num')],
                coefCells: [...cg.querySelectorAll('.num')],
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
        function formatCoef(value) {
            if (value === 0) return "0";
            return Math.abs(value) >= 1e-3 ? value.toFixed(4) : value.toExponential(1);
        }
        function updateUI(agent=null) {
            const f = Math.max(0, Math.min(frameMax(), Math.floor(step)));
            document.getElementById('stepNow').textContent = f;
            if (!scrubbing) document.getElementById('sld').value = f;
            const hud = document.getElementById('hud-telemetry'), obsBox = document.getElementById('obs-container');
            if (followedId === null || !agent) { hud.style.display='none'; obsBox.style.display='none'; return; }
            hud.style.display='block'; document.getElementById('camMode').textContent = isEgoCam ? 'ego cam' : 'world cam';
            ensurePanels();
            const mb = (f * H.agent_cap + agent.idx) * F.mf, pb = (f * H.agent_cap + agent.idx) * F.pf;
            for (const [id,val] of [["tel-id",agent.id],["tel-speed",(agent.s*3.6).toFixed(1)],["tel-st",(agent.st*180/Math.PI).toFixed(1)],["tel-al",agent.al.toFixed(2)],["tel-alat",agent.alat.toFixed(2)],["tel-jl",agent.jl.toFixed(2)],["tel-jlat",agent.jlat.toFixed(2)],["tel-x",agent.x.toFixed(1)],["tel-y",agent.y.toFixed(1)],["tel-h",agent.h.toFixed(3)],["tel-lane",agent.cl],["tel-ps",C.puffer_f32[pb].toFixed(3)]]) document.getElementById(id).textContent = val;
            for (let i=0;i<refs.metric.length;i++) refs.metric[i].textContent = C.metrics_f32[mb+i].toFixed(2);
            for (let i=0;i<refs.puffer.length;i++) refs.puffer[i].textContent = C.puffer_f32[pb+i].toFixed(3);
            if (refs.rewardCells.length) {
                // rewards_f32 rows are cumulative Log sums; step value = difference to the previous frame.
                const rb = (f * H.agent_cap + agent.idx) * F.rf, rbPrev = ((f-1) * H.agent_cap + agent.idx) * F.rf;
                const cumulativeReward = i => C.rewards_f32[rb + i];
                const stepReward = i => f > 0 ? cumulativeReward(i) - C.rewards_f32[rbPrev + i] : cumulativeReward(i);
                const formatReward = value => value === 0 ? "0" : (Math.abs(value) >= 1e-4 ? value.toFixed(5) : value.toExponential(1));
                refs.rewardCells[0].textContent = cumulativeReward(0).toFixed(3);
                refs.rewardCells[1].textContent = formatReward(stepReward(0));
                for (let i=2;i<refs.rewardCells.length;i++) refs.rewardCells[i].textContent = formatReward(stepReward(i-1));
            }
            if (refs.coefCells.length) {
                const cb = (f * H.agent_cap + agent.idx) * F.cf;
                for (let i=0;i<refs.coefCells.length;i++) refs.coefCells[i].textContent = formatCoef(C.coefs_f32[cb+i]);
            }
            updatePolicy(f, agent);
            const warnings = []; if(C.metrics_f32[mb] === 1) warnings.push("COLLISION"); if(C.metrics_f32[mb+1] === 1) warnings.push("OFFROAD"); if(C.metrics_f32[mb+2] === 1) warnings.push("RED LIGHT"); if(C.metrics_f32[mb+3] === 1) warnings.push("STOP SIGN");
            const warnKey = warnings.join('|'), warnRow = document.getElementById('warn-row');
            if (warnKey !== lastWarnKey) { lastWarnKey = warnKey; warnRow.style.display = warnings.length ? 'flex' : 'none'; warnRow.innerHTML = warnings.map(w=>`<span class="warn-chip">${w}</span>`).join(''); }
            const obs = decodeObs(f, agent.slot), obsTitle = document.getElementById('obs-title-text');
            if (obs) { obsBox.style.display='block'; obsTitle.textContent='Ego-centric observation'; drawObs(obs); }
            else if (C.obs && agent.slot >= 0) { obsBox.style.display='block'; obsTitle.textContent=missingObservationsText(); obsCtx.fillStyle='#fff'; obsCtx.fillRect(0,0,obsC.width,obsC.height); }
            else obsBox.style.display='none';
        }
        function draw(force=false) {
            if(!H) return;
            const f = Math.max(0, Math.min(frameMax(), Math.floor(step)));
            if(!force && f === lastDrawn) return;
            const target = followedId !== null ? findAgent(f, followedId) : null;
            if (target) { cam.x = target.x; cam.y = target.y; }
            updateUI(target);
            const observed = observedSetFor(f, target);
            visibleAgentIdx = observed ? new Set([target.idx].concat(observed.partnerAgentIdx)) : null;
            visibleAgentSlots = observed ? new Set(Array.from(visibleAgentIdx, i => agentAt(f, i)).filter(a => a && a.slot >= 0).map(a => a.slot)) : null;
            const colors = getColors(); ctx.fillStyle = colors.bg; ctx.fillRect(0,0,c.width,c.height); ctx.save(); ctx.translate(c.width/2,c.height/2); ctx.scale(cam.z,-cam.z); if(isEgoCam && target) ctx.rotate(Math.PI/2 - target.h); ctx.translate(-cam.x,-cam.y);
            if (observed) drawObservedMap(observed, colors);
            else { ctx.lineCap='round'; ctx.strokeStyle=colors.road; ctx.lineWidth=.5; ctx.stroke(paths[0]); ctx.strokeStyle=colors.line; ctx.setLineDash([1,1]); ctx.stroke(paths[1]); ctx.setLineDash([]); ctx.strokeStyle=colors.edge; ctx.lineWidth=.8; ctx.stroke(paths[2]); drawCurrentLane(target, colors); }
            drawGhosts(f);
            drawPredictedPath(f);
            for(const a of getFrameAgents(f)){ if(visibleAgentIdx && !visibleAgentIdx.has(a.idx)) continue; ctx.save(); ctx.translate(a.x,a.y); ctx.rotate(a.h); drawAgentBody(a, darkMode?'#fff':'#111'); drawPerturbationOutlines(a); ctx.restore(); ctx.save(); ctx.translate(a.x,a.y); if(isEgoCam && target) ctx.rotate(-Math.PI/2 + target.h); else ctx.scale(1,-1); ctx.fillStyle=colors.text; ctx.font='600 '+(14/cam.z)+'px system-ui'; ctx.textAlign='center'; ctx.fillText(a.id,0,(isEgoCam && target)?a.w/2+.5:-a.w/2-.5); ctx.restore(); if(a.id === followedId){ ctx.save(); ctx.translate(a.x,a.y); ctx.strokeStyle=colors.accent; ctx.lineWidth=3/cam.z; ctx.beginPath(); ctx.arc(0,0,Math.max(a.l,a.w)*1.2,0,7); ctx.stroke(); ctx.restore(); } }
            if (observed) { drawUnmatchedPartners(observed, colors); drawObservedStopLines(observed); }
            for(let i=0;i<H.traffic_static_count && !observed;i++){ const t=trafficAt(f,i); if(!t) continue; const sl=t.stop_line; ctx.lineCap='butt'; if(t.type === 1){ ctx.strokeStyle=trafficColor(t); ctx.lineWidth=Math.min(1.5,3/cam.z); } else { ctx.strokeStyle=t.type === 2 ? '#ff0000' : '#ffd700'; ctx.lineWidth=Math.min(1.2,2.5/cam.z); ctx.setLineDash([6/cam.z,4/cam.z]); } ctx.beginPath(); ctx.moveTo(sl[0],sl[1]); ctx.lineTo(sl[3],sl[4]); ctx.stroke(); ctx.setLineDash([]); }
            if(target){ for(const g of selectedGoals(f,target)){ ctx.strokeStyle='#38bdf8'; ctx.fillStyle='rgba(56,189,248,.22)'; ctx.lineWidth=Math.max(.25,2.5/cam.z); ctx.beginPath(); ctx.arc(g.x,g.y,g.radius,0,7); ctx.fill(); ctx.stroke(); } }
            ctx.restore(); lastDrawn = f;
            if (agentViewOn && target) drawAgentView(f, target, observed);
        }
        function toggle(){ play=!play; lastTick=performance.now(); updateBtn(); if(play) requestAnimationFrame(loop); }
        function updateBtn(){ document.getElementById('btnPlay').innerHTML = play ? SVG_PAUSE : SVG_PLAY; }
        function changeSpeed(){ speed=parseFloat(document.getElementById('speedSel').value); lastTick=performance.now(); }
        function loop(ts){
            if(!play) return;
            const dt = Math.min((ts-lastTick)/1000, 0.25);
            lastTick = ts;
            step += dt * speed * VIEW_STYLE.replay_frames_per_second;
            while(step > frameMax()) step -= frameMax() + 1;
            draw();
            requestAnimationFrame(loop);
        }
        let scrubbing = false;
        const sld = document.getElementById('sld');
        sld.addEventListener('pointerdown', () => { scrubbing = true; play = false; updateBtn(); });
        window.addEventListener('pointerup', () => { if (scrubbing) { scrubbing = false; draw(true); } });
        sld.oninput = e => { step = +e.target.value; play=false; updateBtn(); draw(true); };
    </script>
</body>
</html>
    """
    payload_chunks = "\n".join(
        f'    <script type="application/octet-stream" class="payload-chunk">'
        f"{payload[chunk_start : chunk_start + PAYLOAD_CHUNK_SIZE]}</script>"
        for chunk_start in range(0, len(payload), PAYLOAD_CHUNK_SIZE)
    )
    map_colors = replay_format.REPLAY_VIEW_STYLE["map_colors"]
    final_html = (
        html_template.replace("__METRIC_LABELS__", json.dumps(METRIC_LABELS, separators=(",", ":")))
        .replace("__VEHICLE_COLORS__", json.dumps(VEHICLE_COLORS, separators=(",", ":")))
        .replace("__REPLAY_VIEW_STYLE__", json.dumps(replay_format.REPLAY_VIEW_STYLE, separators=(",", ":")))
        .replace("__MAP_ROAD_LIGHT__", map_colors["light"]["road"])
        .replace("__MAP_LINE_LIGHT__", map_colors["light"]["line"])
        .replace("__MAP_EDGE_LIGHT__", map_colors["light"]["edge"])
        .replace("__MAP_ROAD_DARK__", map_colors["dark"]["road"])
        .replace("__MAP_LINE_DARK__", map_colors["dark"]["line"])
        .replace("__MAP_EDGE_DARK__", map_colors["dark"]["edge"])
        .replace("__PAYLOAD_CHUNKS__", payload_chunks)
    )
    with open(filename, "w") as f:
        f.write(final_html)


def save_interactive_replay_zlib(scenario, replay, filename):
    compressed_payload = encode_interactive_replay(scenario, replay)
    with open(filename, "wb") as replay_file:
        replay_file.write(compressed_payload)
    return compressed_payload


def render_interactive_replay_zlib(replay_path, filename):
    with open(replay_path, "rb") as replay_file:
        compressed_payload = replay_file.read()
    _render_interactive_replay_payload(compressed_payload, filename)


def build_gallery_index(folder_path=".", file_metrics=None):
    """Build an index.html navigator for per-episode replay HTMLs in folder_path.

    If `file_metrics` is a dict mapping `<html basename> -> {metric_name: value}`,
    the index exposes a filter for each supported infraction present in the
    metrics. Previous/next navigation follows the active filtered list.
    """
    files = [f for f in os.listdir(folder_path) if f != "index.html" and f.endswith(".html")]

    if not files:
        print("No matching .html files found in this directory.")
        return

    files.sort()

    metrics_map = file_metrics or {}
    present_metrics = set()
    for metrics in metrics_map.values():
        present_metrics.update(metrics.keys())

    FAILURE_FILTERS = (
        ("offroad", "offroad_rate", "Off-road"),
        ("collision", "collision_rate", "Collisions"),
        ("atfault", "at_fault_collision_rate", "At-fault collisions"),
        ("redlight", "red_light_violation_rate", "Red-light violations"),
    )
    available_failure_filters = [
        failure_filter for failure_filter in FAILURE_FILTERS if failure_filter[1] in present_metrics
    ]

    failure_flags = {}
    for filename in files:
        metrics = metrics_map.get(filename, {})
        failure_flags[filename] = {
            filter_key: metrics.get(metric_key, 0) > 0 for filter_key, metric_key, _ in FAILURE_FILTERS
        }

    options_html = "\n".join(
        (
            f'<option value="{filename}" data-name="{filename}" '
            + " ".join(
                f'data-{filter_key}="{str(failure_flags[filename][filter_key]).lower()}"'
                for filter_key, _, _ in FAILURE_FILTERS
            )
            + ">"
            f"{filename.removesuffix('.html')}</option>"
        )
        for filename in files
    )

    category_filter_ui = ""
    if available_failure_filters:
        category_buttons = "".join(
            (
                f'<button class="filter-button category-button" type="button" '
                f'data-filter="{filter_key}" aria-pressed="false"'
                f"{' disabled' if failure_count == 0 else ''}>"
                '<span class="selection-mark" aria-hidden="true">&#10003;</span>'
                f'<span class="filter-dot {filter_key}-dot"></span>'
                f"<span>{filter_label}</span>"
                f'<span class="category-count">{failure_count}</span></button>'
            )
            for filter_key, _, filter_label in available_failure_filters
            for failure_count in [sum(flags[filter_key] for flags in failure_flags.values())]
        )
        category_filter_ui = (
            '<nav class="category-bar" aria-label="Replay category">'
            '<div class="category-filters" role="group" '
            'aria-label="Replay category">'
            '<button class="filter-button category-button is-active" type="button" '
            'data-filter="all" aria-pressed="true">'
            '<span class="selection-mark" aria-hidden="true">&#10003;</span>'
            '<span class="filter-dot all-dot"></span><span>All replays</span>'
            f'<span class="category-count">{len(files)}</span></button>'
            f"{category_buttons}"
            "</div>"
            "</nav>"
        )

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>PufferDrive Replay Gallery</title>
    <style>
        :root {
            color-scheme: light;
            --background: #f3f4f6;
            --surface: #ffffff;
            --surface-muted: #f8f9fb;
            --border: #d7dce2;
            --border-strong: #aeb6c2;
            --text: #1f2933;
            --muted: #667085;
            --accent: #2563eb;
            --offroad: #b45309;
            --collision: #b42318;
            --atfault: #7e22ce;
            --redlight: #d92d20;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            height: 100vh;
            overflow: hidden;
            background: var(--background);
            color: var(--text);
            font-family: Arial, system-ui, sans-serif;
        }

        .app-shell {
            display: flex;
            flex-direction: column;
            width: 100%;
            height: 100%;
        }

        .top-header {
            display: grid;
            flex: 0 0 auto;
            grid-template-columns: 170px minmax(0, 1fr) minmax(360px, 450px);
            border-bottom: 1px solid var(--border);
            background: var(--surface);
        }

        .brand {
            display: flex;
            grid-column: 1;
            flex-direction: column;
            justify-content: center;
            padding: 8px 18px;
        }

        .brand-kicker {
            margin-bottom: 4px;
            color: var(--muted);
            font-size: 12px;
            font-weight: 600;
        }

        .brand-title {
            font-size: 20px;
            font-weight: 700;
        }

        .top-section {
            min-width: 0;
            padding: 8px 12px;
            border-left: 1px solid var(--border);
        }

        .browse-section {
            display: flex;
            align-items: center;
            grid-column: 3;
            width: min(100%, 450px);
            justify-self: end;
        }

        button, select {
            min-height: 36px;
            border: 1px solid var(--border);
            border-radius: 4px;
            outline: none;
            color: var(--text);
            font: inherit;
        }

        button {
            cursor: pointer;
        }

        button:focus-visible, select:focus-visible {
            outline: 2px solid var(--accent);
            outline-offset: 1px;
        }

        button:disabled {
            cursor: not-allowed;
            opacity: 0.38;
        }

        .filter-dot {
            flex: 0 0 auto;
            width: 8px;
            height: 8px;
            border-radius: 1px;
            background: var(--muted);
        }

        .all-dot { background: var(--accent); }
        .offroad-dot { background: var(--offroad); }
        .collision-dot { background: var(--collision); }
        .atfault-dot { background: var(--atfault); }
        .redlight-dot { background: var(--redlight); }

        select {
            cursor: pointer;
            width: 100%;
            padding: 0 28px 0 10px;
            background: var(--surface);
            font-size: 12px;
            font-weight: 400;
        }

        select:hover { border-color: var(--border-strong); }

        .replay-picker {
            display: flex;
            flex-direction: column;
        }

        #fileSelect {
            width: 100%;
            min-width: 0;
            border-color: var(--border);
            background: var(--surface);
        }

        .browse-controls {
            display: grid;
            grid-template-columns: minmax(180px, 1fr) auto;
            align-items: end;
            gap: 8px;
        }

        .navigation {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 8px;
        }

        .nav-button {
            min-height: 30px;
            padding: 0 8px;
            background: var(--surface);
            color: var(--text);
            font-size: 11px;
            font-weight: 600;
        }

        .nav-button:not(:disabled):hover {
            border-color: var(--accent);
            background: var(--surface-muted);
        }

        #position {
            min-width: 52px;
            color: var(--muted);
            font-size: 11px;
            font-variant-numeric: tabular-nums;
            font-weight: 600;
            text-align: center;
        }

        .category-bar {
            display: flex;
            grid-column: 2;
            flex: 0 0 auto;
            min-width: 0;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            border-left: 1px solid var(--border);
            background: var(--surface);
        }

        .category-filters {
            display: flex;
            min-width: 0;
            flex-wrap: wrap;
            gap: 7px;
        }

        .category-button {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            min-height: 34px;
            padding: 0 8px;
            border: 2px solid var(--border);
            background: var(--surface);
            color: var(--text);
            font-size: 12px;
            font-weight: 600;
        }

        .category-button:not(:disabled):not(.is-active):hover {
            border-color: var(--border-strong);
            background: var(--surface-muted);
        }

        .category-button.is-active {
            border-color: var(--text);
            background: var(--text);
            color: var(--surface);
        }

        .category-button.is-active .filter-dot {
            outline: 1px solid rgba(255, 255, 255, 0.65);
            outline-offset: 1px;
        }

        .selection-mark {
            display: none;
            font-size: 13px;
            font-weight: 700;
            line-height: 1;
        }

        .category-button.is-active .selection-mark {
            display: inline;
        }

        .category-count {
            display: inline-flex;
            min-width: 21px;
            height: 21px;
            align-items: center;
            justify-content: center;
            padding-inline: 5px;
            border: 1px solid var(--border);
            border-radius: 3px;
            background: var(--surface-muted);
            color: var(--text);
            font-size: 11px;
            font-variant-numeric: tabular-nums;
            line-height: 1;
        }

        .category-button.is-active .category-count {
            border-color: #647080;
            background: #374151;
            color: var(--surface);
        }

        .replay-stage {
            display: flex;
            flex: 1 1 auto;
            flex-direction: column;
            min-height: 0;
            min-width: 0;
            background: var(--background);
        }

        .scenario-header {
            display: flex;
            flex: 0 0 auto;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px 16px;
            min-height: 52px;
            margin: 8px 12px 0;
            padding: 7px 12px;
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-bottom: 1px solid var(--border);
            background: var(--surface);
        }

        .scenario-identity {
            min-width: 0;
            flex: 1 1 280px;
        }

        .scenario-eyebrow {
            display: block;
            margin-bottom: 2px;
            color: var(--muted);
            font-size: 11px;
            font-weight: 600;
        }

        #currentReplayName {
            display: block;
            overflow: hidden;
            font-size: 14px;
            font-weight: 700;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        #currentFailures {
            display: flex;
            flex: 0 1 auto;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 5px;
            margin-left: auto;
        }

        .scenario-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 2px 5px;
            border: 1px solid var(--border);
            border-radius: 3px;
            background: var(--surface-muted);
            color: var(--muted);
            font-size: 10px;
            font-weight: 600;
        }

        .scenario-badge.offroad {
            border-left: 3px solid var(--offroad);
            color: var(--offroad);
        }

        .scenario-badge.collision {
            border-left: 3px solid var(--collision);
            color: var(--collision);
        }

        .scenario-badge.atfault {
            border-left: 3px solid var(--atfault);
            color: var(--atfault);
        }

        .scenario-badge.redlight {
            border-left: 3px solid var(--redlight);
            color: var(--redlight);
        }

        #viewer {
            flex: 1 1 auto;
            width: 100%;
            min-height: 0;
            border: 0;
            background: var(--surface);
        }

        @media (max-width: 1200px) {
            body {
                overflow: auto;
            }

            .app-shell {
                min-height: 100%;
                height: auto;
            }

            .top-header {
                grid-template-columns: 170px minmax(0, 1fr);
            }

            .category-bar {
                grid-column: 2;
            }

            .browse-section {
                grid-column: 1 / -1;
                width: 100%;
                border-top: 1px solid var(--border);
                border-left: 0;
            }

            .category-bar {
                border-left: 1px solid var(--border);
            }

            .replay-stage { min-height: 72vh; }
        }

        @media (max-width: 720px) {
            .top-header { display: flex; flex-direction: column; }
            .brand { padding-block: 16px; }
            .brand-title { font-size: 18px; }
            .top-section { border-top: 1px solid var(--border); border-left: 0; }
            .category-bar { border-top: 1px solid var(--border); border-left: 0; }
            .browse-section { width: 100%; }
            .browse-controls { grid-template-columns: 1fr; }
            .category-bar { padding: 12px; }
            .category-filters { width: 100%; }
            .scenario-header { align-items: flex-start; }
            #currentFailures { margin-left: 0; justify-content: flex-start; }
        }
    </style>
</head>
<body>
    <div class="app-shell">
        <header class="top-header">
            <div class="brand">
                <span class="brand-kicker">PufferDrive</span>
                <span class="brand-title">Replay index</span>
            </div>
            __CATEGORY_FILTER_UI__
            <section class="top-section browse-section">
                <div class="browse-controls">
                    <div class="replay-picker">
                        <select id="fileSelect" aria-label="Replay">
                            __OPTIONS__
                        </select>
                    </div>
                    <nav class="navigation" aria-label="Replay navigation">
                        <button id="prevBtn" class="nav-button" type="button" aria-label="Previous replay">
                            &#8592; Previous
                        </button>
                        <span id="position" aria-live="polite"></span>
                        <button id="nextBtn" class="nav-button" type="button" aria-label="Next replay">
                            Next &#8594;
                        </button>
                    </nav>
                </div>
            </section>
        </header>

        <main class="replay-stage">
            <header class="scenario-header">
                <div class="scenario-identity">
                    <span class="scenario-eyebrow">Selected replay</span>
                    <strong id="currentReplayName"></strong>
                </div>
                <div id="currentFailures" aria-label="Scenario failures"></div>
            </header>
            <iframe id="viewer" src="__FIRST__" title="Replay viewer"></iframe>
        </main>
    </div>

    <script>
        const select = document.getElementById('fileSelect');
        const viewer = document.getElementById('viewer');
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const position = document.getElementById('position');
        const currentReplayName = document.getElementById('currentReplayName');
        const currentFailures = document.getElementById('currentFailures');
        const filterButtons = Array.from(document.querySelectorAll('.filter-button'));
        const allOptions = Array.from(select.options);
        let activeFilter = 'all';

        function optionMatchesActiveFilter(option) {
            if (activeFilter === 'all') return true;
            return option.dataset[activeFilter] === 'true';
        }

        function addFailureBadge(label, failureClass) {
            const badge = document.createElement('span');
            badge.className = `scenario-badge ${failureClass}`;
            const dot = document.createElement('span');
            dot.className = `filter-dot ${failureClass}-dot`;
            badge.append(dot, label);
            currentFailures.appendChild(badge);
        }

        function updateScenarioSummary() {
            const selectedOption = select.options[select.selectedIndex];
            currentFailures.replaceChildren();

            if (!selectedOption) {
                currentReplayName.textContent = 'No replay matches this filter';
                return;
            }

            const filename = selectedOption.dataset.name;
            currentReplayName.textContent = filename.replace(/\\.html$/, '');
            if (selectedOption.dataset.offroad === 'true') addFailureBadge('Off-road', 'offroad');
            if (selectedOption.dataset.collision === 'true') addFailureBadge('Collision', 'collision');
            if (selectedOption.dataset.atfault === 'true') addFailureBadge('At-fault collision', 'atfault');
            if (selectedOption.dataset.redlight === 'true') addFailureBadge('Red-light violation', 'redlight');
            if (!currentFailures.childElementCount) {
                const badge = document.createElement('span');
                badge.className = 'scenario-badge';
                badge.textContent = 'No recorded infraction';
                currentFailures.appendChild(badge);
            }
        }

        function updateControls() {
            const optionCount = select.options.length;
            const selectedIndex = select.selectedIndex;
            prevBtn.disabled = selectedIndex <= 0;
            nextBtn.disabled = selectedIndex < 0 || selectedIndex >= optionCount - 1;
            position.textContent = selectedIndex < 0 ? `0 / ${optionCount}` : `${selectedIndex + 1} / ${optionCount}`;
            updateScenarioSummary();
        }

        function loadSelected() {
            if (select.selectedIndex < 0) {
                viewer.removeAttribute('src');
                updateControls();
                return;
            }
            viewer.src = select.value;
            updateControls();
        }

        function renderOptions(loadReplay) {
            const currentFilename = select.value;
            const matchingOptions = allOptions.filter(optionMatchesActiveFilter);
            select.replaceChildren(...matchingOptions);

            const currentOption = matchingOptions.find(option => option.value === currentFilename);
            if (currentOption) {
                select.value = currentFilename;
            } else {
                select.selectedIndex = matchingOptions.length > 0 ? 0 : -1;
            }

            const selectionChanged = select.value !== currentFilename;
            if (loadReplay && selectionChanged) loadSelected();
            else updateControls();
        }

        function setFilter(filterName) {
            activeFilter = filterName;
            for (const button of filterButtons) {
                const isActive = button.dataset.filter === activeFilter;
                button.classList.toggle('is-active', isActive);
                button.setAttribute('aria-pressed', String(isActive));
            }
            renderOptions(true);
        }

        function navigate(direction) {
            const nextIndex = select.selectedIndex + direction;
            if (nextIndex < 0 || nextIndex >= select.options.length) return;
            select.selectedIndex = nextIndex;
            loadSelected();
        }

        select.addEventListener('change', loadSelected);
        prevBtn.addEventListener('click', () => navigate(-1));
        nextBtn.addEventListener('click', () => navigate(1));
        for (const button of filterButtons) {
            button.addEventListener('click', () => setFilter(button.dataset.filter));
        }
        viewer.addEventListener('load', () => viewer.contentWindow.focus());

        renderOptions(false);
    </script>
</body>
</html>
    """

    final_html = (
        html_content.replace("__OPTIONS__", options_html)
        .replace("__FIRST__", files[0])
        .replace("__CATEGORY_FILTER_UI__", category_filter_ui)
    )

    index_path = os.path.join(folder_path, "index.html")
    with open(index_path, "w") as f:
        f.write(final_html)
