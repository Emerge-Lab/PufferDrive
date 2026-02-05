"""Bird's Eye View visualization for PufferDrive scenarios using Matplotlib."""

import dataclasses
import weakref
from typing import Optional, Tuple

import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
from matplotlib.patches import Circle

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
    positions = []
    states_list = []
    for elem in traffic_elements or []:
        if not isinstance(elem, dict):
            continue
        x, y = elem.get("x"), elem.get("y")
        if x is None or y is None:
            continue
        if elem.get("type", 1) == 1:
            positions.append((x, y))
            states_list.append(elem.get("states", []))
    patches = [Circle(pos, radius=0.6) for pos in positions]
    collection = None
    if patches:
        collection = PatchCollection(
            patches, facecolors=COLORS["road_line"], edgecolors="black", linewidths=0.5, alpha=0.9, zorder=15
        )
    return {"states": states_list, "collection": collection, "count": len(states_list)}


def _render_traffic(ax, traffic_cache, timestep):
    if not traffic_cache:
        return
    collection = traffic_cache.get("collection")
    if collection is None:
        return
    states_list = traffic_cache.get("states") or []
    colors = []
    for states in states_list:
        state = int(states[timestep]) if states and len(states) > timestep else 0
        colors.append(TRAFFIC_LIGHT_COLORS.get(state, "#808080"))
    if colors:
        collection.set_facecolor(colors)
        ax.add_collection(collection)


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
        metrics = agent.get("metrics_array", [0] * 10)
        metrics_data.append(
            {
                "id": agent_id,
                "current_lane": current_lane_id,
                "speed": speed,
                "lane_dist": metrics[4],
                "lane_head": metrics[5],
                "offroad": metrics[1],
                "collision": metrics[0],
                "comfort": metrics[6],
                "red_light": metrics[2],
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
    headers = ["ID", "Lane", "LDist", "LHead", "Spd", "Cmft", "Off", "Col", "Red"]
    num_agents = len(metrics_data)
    y_start, y_end = 0.95, 0.05
    row_height = min(0.06, (y_start - y_end) / (num_agents + 2))
    x_positions = [0.02, 0.10, 0.20, 0.32, 0.44, 0.56, 0.66, 0.78, 0.90]
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

    ax.set_title("Active Agent Metrics", fontsize=font_size + 4, fontweight="bold", pad=10)


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


def plot_simulator_state(scenario, timestep: int = 0, reuse_key: str = None) -> np.ndarray:
    """Render simulator state to RGB image array."""
    vis_config = VizConfig()

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


def unpack_obs(obs_flat, dynamics_model=0, target_type="static", reward_conditioning=False, num_target_waypoints=10):
    """
    Unpack the flattened observation into the ego state and visible state.
    Args:
        obs_flat: flattened observation tensor of shape (batch_size, obs_dim)
        dynamics_model: "classic" or "jerk"
        target_type: "static" or "dynamic"
        num_target_waypoints: number of target waypoints
    Return:
        ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_obs
    """
    ego_dim = binding.EGO_FEATURES_JERK if dynamics_model == "jerk" else binding.EGO_FEATURES_CLASSIC

    # Partner obs
    max_partners = binding.MAX_AGENTS_OBSERVATIONS
    partner_feature_size = binding.PARTNER_FEATURES
    # Road obs
    max_lane_segments = binding.MAX_LANE_SEGMENT_OBSERVATIONS
    max_boundary_segments = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
    road_feature_size = binding.ROAD_FEATURES
    # Traffic light obs
    max_traffic_lights = binding.MAX_TRAFFIC_CONTROLS
    traffic_feature_size = binding.TRAFFIC_CONTROL_FEATURES
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

    # Extract road elements
    lane_start = partners_end
    lane_end = lane_start + max_lane_segments * road_feature_size
    lane_obs = obs_flat[:, lane_start:lane_end]
    lane_obs = lane_obs.reshape(-1, max_lane_segments, road_feature_size)

    boundary_start = lane_end
    boundary_end = boundary_start + max_boundary_segments * road_feature_size
    boundary_obs = obs_flat[:, boundary_start:boundary_end]
    boundary_obs = boundary_obs.reshape(-1, max_boundary_segments, road_feature_size)

    # Extract traffic lights
    traffic_start = boundary_end
    traffic_end = traffic_start + max_traffic_lights * traffic_feature_size
    traffic_obs = obs_flat[:, traffic_start:traffic_end]
    traffic_obs = traffic_obs.reshape(-1, max_traffic_lights, traffic_feature_size)

    return (
        ego_state[0],
        target_obs[0],
        partners_obs[0],
        lane_obs[0],
        boundary_obs[0],
        traffic_obs[0],
    )


def plot_observation(
    obs, dynamics_model="classic", target_type="static", reward_conditioning=False, num_target_waypoints=10
) -> np.ndarray:
    """Plot observation in ego-centric frame.

    Args:
        obs: flattened observation tensor
        dynamics_model: "classic" or "jerk"
        target_type: "static" or "dynamic"
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    ego_state, target_obs, partners_obs, lane_obs, boundary_obs, traffic_obs = unpack_obs(
        obs,
        dynamics_model,
        target_type,
        reward_conditioning,
        num_target_waypoints,
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
        color = "red" if i == 0 else "orange"
        marker = "*" if i == 0 else "o"
        s = 200 if i == 0 else 80
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

    ax.axis((-1, 1, -1, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (ego frame)", fontsize=16)
    ax.set_ylabel("Y (ego frame)", fontsize=16)
    ax.set_title("Observation (Ego-Centric View)", fontsize=18, fontweight="bold")
    # ax.grid(True, alpha=0.3)
    return _img_from_fig(fig)
