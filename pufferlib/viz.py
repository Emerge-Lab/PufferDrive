"""Bird's Eye View visualization for PufferDrive scenarios using Matplotlib."""

import dataclasses
import weakref
from typing import Optional, Tuple

import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, FancyArrow

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


@dataclasses.dataclass
class VizConfig:
    """Visualization config using radius and center for view bounds."""

    center: Optional[Tuple[float, float]] = None
    radius: Optional[float] = None
    figsize: Tuple[float, float] = (20.0, 20.0)
    dpi: int = 150
    show_agent_id: bool = True
    show_routes: bool = False
    show_goal: bool = True
    show_sdc_paths: bool = False
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
            1, 2, figsize=(config.figsize[0] * 1.3, config.figsize[1]), gridspec_kw={"width_ratios": [3, 1]}
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


def _render_roads(ax, road_elements):
    if not road_elements:
        return
    lanes, lines, edges = [], [], []
    for elem in road_elements:
        if not isinstance(elem, dict):
            continue
        x, y, t = elem.get("x"), elem.get("y"), elem.get("type", 0)
        if not x or not y:
            continue
        pts = np.column_stack((np.array(x), np.array(y)))
        if 1 <= t <= 3:
            lanes.append(pts)
        elif 11 <= t <= 18:
            lines.append(pts)
        elif 21 <= t <= 23:
            edges.append(pts)

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


def _render_traffic(ax, traffic_elements, timestep):
    if not traffic_elements:
        return
    patches, colors = [], []
    for elem in traffic_elements:
        if not isinstance(elem, dict):
            continue
        x, y = elem.get("x"), elem.get("y")
        if x is None or y is None:
            continue
        if elem.get("type", 1) == 1:
            states = elem.get("states", [])
            state = int(states[timestep]) if states and len(states) > timestep else 0
            patches.append(Circle((x, y), radius=0.6))
            colors.append(TRAFFIC_LIGHT_COLORS.get(state, "#808080"))
    if patches:
        ax.add_collection(
            PatchCollection(patches, facecolors=colors, edgecolors="black", linewidths=0.5, alpha=0.9, zorder=15)
        )


def _render_routes(ax, agents, road_elements, active_indices):
    if not agents or not road_elements:
        return
    lane_dict = {}
    for elem in road_elements:
        if isinstance(elem, dict) and elem.get("type") in [1, 2, 3]:
            lid, x, y = elem.get("id"), elem.get("x"), elem.get("y")
            if lid is not None and x and y:
                lane_dict[lid] = np.column_stack((np.array(x), np.array(y)))

    active_set = set(active_indices or [])
    segments_by_color = {}
    for idx, agent in enumerate(agents):
        if not isinstance(agent, dict) or idx not in active_set:
            continue
        route = agent.get("route", [])
        if not route:
            continue
        color = get_agent_color(agent.get("id", idx))
        if color not in segments_by_color:
            segments_by_color[color] = []
        for lid in route:
            if lid in lane_dict:
                segments_by_color[color].append(lane_dict[lid])

    for color, segs in segments_by_color.items():
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=2.0, alpha=0.6, linestyles="--", zorder=5))


def _render_agents(ax, agents, active_indices, static_indices, config, px_per_meter):
    if not agents:
        return
    active_set, static_set = set(active_indices or []), set(static_indices or [])

    for idx, agent in enumerate(agents):
        if idx not in active_set and idx not in static_set:
            continue
        x, y = agent.get("sim_x"), agent.get("sim_y")
        if not agent.get("sim_valid"):
            continue

        agent_type, agent_id = agent.get("type", 1), agent.get("id", idx)
        heading = agent.get("sim_heading", 0)
        length, width = agent.get("sim_length", 4), agent.get("sim_width", 2)
        is_active = idx in active_set
        color = get_agent_color(agent_id, is_active)
        edge = "black" if is_active else COLORS["inactive_agent"]

        if agent_type == 1:  # Vehicle
            if agent["stopped"]:
                color = "red"

            rect = mpatches.Rectangle(
                (-length / 2, -width / 2),
                length,
                width,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.7,
                alpha=0.8,
                zorder=10,
            )
            rect.set_transform(plt.matplotlib.transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData)
            ax.add_patch(rect)

            dx, dy = length * 0.6 * np.cos(heading), length * 0.6 * np.sin(heading)
            ax.add_patch(
                FancyArrow(
                    x,
                    y,
                    dx,
                    dy,
                    width=width * 0.12,
                    head_width=width * 0.4,
                    head_length=width * 0.25,
                    fc=color,
                    ec="black",
                    linewidth=0.3,
                    zorder=11,
                )
            )

            if config.show_agent_id:
                ax.text(
                    x,
                    y + width,
                    str(agent_id),
                    fontsize=max(12, int(px_per_meter / 5)),
                    color="black",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    zorder=12,
                )

            if config.show_goal and is_active:
                gx, gy = agent.get("goal_position_x"), agent.get("goal_position_y")
                if gx is not None and gy is not None:
                    ax.scatter([gx], [gy], s=20, c=[color], marker="o", zorder=13)
                    circle = Circle((gx, gy), radius=config.goal_radius)
                    circle.set_facecolor("none")
                    circle.set_edgecolor(color)
                    circle.set_linestyle("--")
                    ax.add_patch(circle)

        elif agent_type == 2:  # Pedestrian
            ax.add_patch(
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
        elif agent_type == 3:  # Cyclist
            ax.add_patch(
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


def _render_debug_metrics_table(ax, agents, active_agent_indices, px_per_meter=10.0):
    """Render a table of per-agent metrics for debugging."""
    font_size = max(8, int(px_per_meter / 5))

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
        # collision_state = agent.get("collision_state", 0)
        # collision_names = {0: "none", 1: "vehicle", 2: "offroad", 3: "traffic"}
        metrics_data.append(
            {
                "id": agent_id,
                "current_lane": current_lane_id,
                "speed": speed,
                # "collision": collision_names.get(collision_state, "unknown"),
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

    # Table headers
    headers = ["ID", "LaneIdx", "Speed", "Collision"]
    num_agents = len(metrics_data)
    y_start, y_end = 0.95, 0.05
    row_height = min(0.06, (y_start - y_end) / (num_agents + 2))
    x_positions = [0.02, 0.20, 0.45, 0.70]
    for i, header in enumerate(headers):
        ax.text(x_positions[i], y_start, header, fontsize=font_size + 2, fontweight="bold", va="top")

    collision_colors = {"none": "green", "vehicle": "red", "offroad": "orange", "traffic": "red"}
    for row_idx, data in enumerate(metrics_data):
        y_pos = y_start - (row_idx + 1) * row_height
        ax.text(
            x_positions[0], y_pos, str(data["id"]), fontsize=font_size, color=data["color"], fontweight="bold", va="top"
        )
        ax.text(x_positions[1], y_pos, f"{data['current_lane']:.1f}", fontsize=font_size, va="top")
        ax.text(x_positions[2], y_pos, f"{data['speed']:.1f}", fontsize=font_size, va="top")
        ax.text(
            x_positions[3],
            y_pos,
            data["collision"],
            fontsize=font_size,
            color=collision_colors.get(data["collision"], "black"),
            va="top",
        )

    ax.set_title("Active Agent Metrics", fontsize=font_size + 4, fontweight="bold", pad=10)


def plot_simulator_state(scenario, timestep: int = 0, reuse_key: str = None) -> np.ndarray:
    """Render simulator state to RGB image array."""
    vis_config = VizConfig()

    bounds = vis_config.get_bounds(scenario)
    x_min, x_max, y_min, y_max = bounds

    px_per_meter = min(
        vis_config.figsize[0] * vis_config.dpi / (x_max - x_min),
        vis_config.figsize[1] * vis_config.dpi / (y_max - y_min),
    )

    if vis_config.debug_metrics:
        fig, ax, ax_metrics = _init_fig_ax(vis_config, reuse_key, with_metrics=True)
    else:
        fig, ax = _init_fig_ax(vis_config, reuse_key, with_metrics=False)
        ax_metrics = None

    ax.set_aspect("equal")
    ax.set_title(
        f"PufferDrive | {scenario.get('dataset_name', '')} | {scenario.get('scenario_id', '')} | t={timestep}",
        fontsize=max(14, int(px_per_meter / 8)),
        fontweight="bold",
    )

    _render_roads(ax, scenario.get("road_elements", []))
    _render_traffic(ax, scenario.get("traffic_elements", []), timestep)
    if vis_config.show_routes:
        _render_routes(
            ax, scenario.get("agents", []), scenario.get("road_elements", []), scenario.get("active_agent_indices", [])
        )
    if vis_config.show_sdc_paths:
        _render_paths(ax, scenario)

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

    return _img_from_fig(fig)


def _img_from_fig(fig: matplotlib.figure.Figure) -> np.ndarray:
    fig.subplots_adjust(left=0.01, bottom=0.02, right=1.00, top=0.96)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
    plt.close(fig)
    return img


def unpack_obs(obs_flat, dynamics_model=0, target_type=0):
    """
    Unpack the flattened observation into the ego state and visible state.
    Args:
        obs_flat: flattened observation tensor of shape (batch_size, obs_dim)
        dynamics_model: 0 for CLASSIC, 1 for JERK
        target_type: 0 for goal only, 1 for waypoints only, 2 for both
    Return:
        ego_state, partners_obs, road_obs, traffic_obs, gps_obs, include_goal, include_waypoints
    """
    include_goal = target_type in ["goal", "both"]
    include_waypoints = target_type in ["waypoints", "both"]

    if include_goal:
        ego_dim = binding.EGO_FEATURES_JERK if dynamics_model == "jerk" else binding.EGO_FEATURES_CLASSIC
    else:
        ego_dim = (
            binding.EGO_FEATURES_JERK_NO_GOAL if dynamics_model == "jerk" else binding.EGO_FEATURES_CLASSIC_NO_GOAL
        )

    max_partners = binding.MAX_AGENTS_OBSERVATIONS
    partner_feature_size = binding.PARTNER_FEATURES
    max_road_segments = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
    road_feature_size = binding.ROAD_FEATURES
    max_traffic_lights = binding.MAX_TRAFFIC_CONTROLS
    traffic_feature_size = binding.TRAFFIC_CONTROL_FEATURES
    max_gps_objects = binding.MAX_GPS_OBSERVATIONS if include_waypoints else 0
    gps_feature_size = binding.GPS_FEATURES

    # Extract ego state
    ego_state = obs_flat[:, :ego_dim]

    # Extract GPS path (only if waypoints included)
    gps_start = ego_dim
    if include_waypoints:
        gps_end = gps_start + max_gps_objects * gps_feature_size
        gps_obs = obs_flat[:, gps_start:gps_end]
        gps_obs = gps_obs.reshape(-1, max_gps_objects, gps_feature_size)
    else:
        gps_end = gps_start
        gps_obs = np.zeros((obs_flat.shape[0], 0, gps_feature_size))

    # Extract partners
    partners_start = gps_end
    partners_end = partners_start + max_partners * partner_feature_size
    partners_obs = obs_flat[:, partners_start:partners_end]
    partners_obs = partners_obs.reshape(-1, max_partners, partner_feature_size)

    # Extract road elements
    road_start = partners_end
    road_end = road_start + max_road_segments * road_feature_size
    road_obs = obs_flat[:, road_start:road_end]
    road_obs = road_obs.reshape(-1, max_road_segments, road_feature_size)

    # Extract traffic lights
    traffic_start = road_end
    traffic_end = traffic_start + max_traffic_lights * traffic_feature_size
    traffic_obs = obs_flat[:, traffic_start:traffic_end]
    traffic_obs = traffic_obs.reshape(-1, max_traffic_lights, traffic_feature_size)

    return ego_state[0], partners_obs[0], road_obs[0], traffic_obs[0], gps_obs[0], include_goal, include_waypoints


def plot_observation(obs, dynamics_model="classic", target_type="goal") -> np.ndarray:
    """Plot observation in ego-centric frame.

    Args:
        obs: flattened observation tensor
        dynamics_model: 0 for CLASSIC, 1 for JERK
        target_type: 0 for goal only, 1 for waypoints only, 2 for both
    """
    fig, ax = plt.subplots(figsize=(20, 20))

    ego_state, partners_obs, road_obs, traffic_obs, gps_obs, include_goal, include_waypoints = unpack_obs(
        obs, dynamics_model, target_type
    )

    # Unpack ego state based on dynamics model and target_type
    if include_goal:
        if dynamics_model == "jerk":  # JERK model with goal
            goal_x, goal_y, ego_speed, ego_width, ego_length, steering_angle, a_long, a_lat = ego_state
        else:  # CLASSIC model with goal
            goal_x, goal_y, ego_speed, ego_width, ego_length = ego_state
    else:
        goal_x, goal_y = None, None
        if dynamics_model == "jerk":  # JERK model without goal
            ego_speed, ego_width, ego_length, steering_angle, a_long, a_lat = ego_state
        else:  # CLASSIC model without goal
            ego_speed, ego_width, ego_length = ego_state

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

    # Draw goal if included
    if include_goal and goal_x is not None:
        ax.scatter(goal_x, goal_y, color="red", marker="*", s=200, zorder=15, label="Goal")

    # Add dynamics info text for JERK model
    if dynamics_model == "jerk":
        ax.text(
            0.02,
            0.98,
            f"Speed: {ego_speed:.2f}\nSteering: {steering_angle:.3f}\na_long: {a_long:.2f}\na_lat: {a_lat:.2f}",
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
    for i in range(road_obs.shape[0]):
        if np.all(road_obs[i] == 0):
            continue
        rel_x, rel_y, length, width, dir_cos, dir_sin, road_type = road_obs[i]

        if road_type == 0:  # lane
            color = "lightgrey"
        elif road_type == 1:  # line
            color = "grey"
        elif road_type == 2:  # edge
            color = "black"
        else:
            continue

        ax.scatter(rel_x, rel_y, color=color, s=10, zorder=1)
        ax.plot(
            [rel_x + dir_cos * length / 2, rel_x - dir_cos * length / 2],
            [rel_y + dir_sin * length / 2, rel_y - dir_sin * length / 2],
            color=color,
            linewidth=1,
            zorder=1,
        )

    # Traffic lights
    for i in range(traffic_obs.shape[0]):
        if np.all(traffic_obs[i] == 0):
            continue
        rel_x, rel_y, state_normalized = traffic_obs[i]

        if state_normalized == 0:
            state = 4
        elif state_normalized == 1:
            state = 2
        elif state_normalized == 2:
            state = 6
        else:
            state = 0

        ax.add_patch(
            plt.Circle(
                (rel_x, rel_y),
                radius=0.01,
                alpha=0.9,
                facecolor=TRAFFIC_LIGHT_COLORS[state],
                edgecolor="black",
                linewidth=1,
                zorder=12,
            )
        )

    # Plot GPS path (waypoints) if included
    if include_waypoints and gps_obs.shape[0] > 0:
        wp_x_list, wp_y_list = [], []
        for i in range(gps_obs.shape[0]):
            if np.all(gps_obs[i] == 0):
                continue
            rel_x, rel_y, heading_cos, heading_sin = gps_obs[i]
            wp_x_list.append(rel_x)
            wp_y_list.append(rel_y)
            ax.scatter(rel_x, rel_y, color="yellow", s=30, zorder=5, edgecolors="black", linewidths=0.5)
        # Connect waypoints with line
        if len(wp_x_list) > 1:
            ax.plot(wp_x_list, wp_y_list, color="yellow", linewidth=2, alpha=0.7, zorder=4)

    ax.axis((-1, 1, -1, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (ego frame)", fontsize=16)
    ax.set_ylabel("Y (ego frame)", fontsize=16)
    ax.set_title("Observation (Ego-Centric View)", fontsize=18, fontweight="bold")
    # ax.grid(True, alpha=0.3)
    return _img_from_fig(fig)
