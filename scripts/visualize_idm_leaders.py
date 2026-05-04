import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding


IDM_BBOX_MARGIN = 0.05
IDM_LOOKAHEAD_TIME = 5.0
IDM_MIN_LOOKAHEAD = 20.0
IDM_MAX_LOOKAHEAD = 80.0
INVALID_POSITION = -10000.0
Z_BUFFER = 4.0

TRAFFIC_LIGHT_COLORS = {
    binding.TRAFFIC_CONTROL_STATE_UNKNOWN: "#808080",
    binding.TRAFFIC_CONTROL_STATE_RED: "#ff0000",
    binding.TRAFFIC_CONTROL_STATE_YELLOW: "#f2c300",
    binding.TRAFFIC_CONTROL_STATE_GREEN: "#00a651",
    binding.TRAFFIC_CONTROL_STATE_OFF: "#808080",
}


def normalize_scenarios(state):
    if isinstance(state, list):
        return state
    return [state]


def point_to_ego_frame(ego, x, y):
    dx = x - ego["sim_x"]
    dy = y - ego["sim_y"]
    heading = ego["sim_heading"]
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return (
        dx * cos_h + dy * sin_h,
        -dx * sin_h + dy * cos_h,
    )


def agent_corners(agent, margin=0.0):
    half_length = 0.5 * agent["sim_length"] + margin
    half_width = 0.5 * agent["sim_width"] + margin
    heading = agent["sim_heading"]
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    corners = []
    for long_sign, lat_sign in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
        x = agent["sim_x"] + long_sign * half_length * cos_h - lat_sign * half_width * sin_h
        y = agent["sim_y"] + long_sign * half_length * sin_h + lat_sign * half_width * cos_h
        corners.append((x, y))
    return corners


def candidate_indices(scenario):
    active = scenario.get("active_agent_indices") or []
    static = scenario.get("static_agent_indices") or []
    return list(active) + list(static)


def traffic_light_controls_lane(traffic, lane_idx):
    controlled_lanes = traffic.get("controlled_lanes") or []
    return lane_idx != -1 and lane_idx in controlled_lanes


def traffic_state_at(traffic, timestep):
    states = traffic.get("states") or []
    if timestep < 0 or timestep >= len(states):
        return binding.TRAFFIC_CONTROL_STATE_OFF
    return int(states[timestep])


def update_best(best, kind, idx, gap, speed):
    if gap < 0.0:
        gap = 0.1
    if best is not None and gap >= best["gap"]:
        return best
    return {
        "kind": kind,
        "idx": idx,
        "gap": max(gap, 0.1),
        "speed": max(speed, 0.0),
    }


def find_sdc_leader(scenario, timestep):
    agents = scenario["agents"]
    active = scenario.get("active_agent_indices") or []
    if not active:
        return None, None

    sdc_idx = active[0]
    ego = agents[sdc_idx]
    ego_speed = max(0.0, float(ego.get("sim_speed_signed", ego.get("sim_speed", 0.0))))
    lookahead = min(max(ego_speed * IDM_LOOKAHEAD_TIME, IDM_MIN_LOOKAHEAD), IDM_MAX_LOOKAHEAD)
    corridor_start = 0.5 * ego["sim_length"] + IDM_BBOX_MARGIN
    corridor_end = corridor_start + lookahead
    corridor_half_width = 0.5 * ego["sim_width"] + IDM_BBOX_MARGIN

    best = None
    for other_idx in candidate_indices(scenario):
        if other_idx == sdc_idx or other_idx < 0 or other_idx >= len(agents):
            continue
        other = agents[other_idx]
        if other.get("removed") or other.get("sim_x") == INVALID_POSITION or not other.get("sim_valid", True):
            continue
        if abs(other.get("sim_z", 0.0) - ego.get("sim_z", 0.0)) > Z_BUFFER:
            continue

        rel = [point_to_ego_frame(ego, x, y) for x, y in agent_corners(other, IDM_BBOX_MARGIN)]
        xs = [p[0] for p in rel]
        ys = [p[1] for p in rel]
        if max(xs) < corridor_start or min(xs) > corridor_end:
            continue
        if max(ys) < -corridor_half_width or min(ys) > corridor_half_width:
            continue

        heading = ego["sim_heading"]
        leader_speed = other.get("sim_vx", 0.0) * math.cos(heading) + other.get("sim_vy", 0.0) * math.sin(heading)
        best = update_best(best, "agent", other_idx, min(xs) - corridor_start, leader_speed)

    ego_lane = int(ego.get("current_lane_idx", -1))
    for traffic_idx, traffic in enumerate(scenario.get("traffic_elements") or []):
        if traffic.get("type") != binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT:
            continue
        if not traffic_light_controls_lane(traffic, ego_lane):
            continue
        if traffic_state_at(traffic, timestep) != binding.TRAFFIC_CONTROL_STATE_RED:
            continue

        stop_line = traffic.get("stop_line") or []
        if len(stop_line) < 6:
            continue
        p1 = point_to_ego_frame(ego, stop_line[0], stop_line[1])
        p2 = point_to_ego_frame(ego, stop_line[3], stop_line[4])
        min_x = min(p1[0], p2[0])
        max_x = max(p1[0], p2[0])
        min_y = min(p1[1], p2[1])
        max_y = max(p1[1], p2[1])
        if max_x < corridor_start or min_x > corridor_end:
            continue
        if max_y < -corridor_half_width or min_y > corridor_half_width:
            continue
        best = update_best(best, "red_light", traffic_idx, 0.5 * (p1[0] + p2[0]) - corridor_start, 0.0)

    corridor = {
        "start": corridor_start,
        "end": corridor_end,
        "half_width": corridor_half_width,
        "sdc_idx": sdc_idx,
    }
    return best, corridor


def draw_roads(ax, scenario):
    for road in scenario.get("road_elements") or []:
        xs = road.get("x") or []
        ys = road.get("y") or []
        if len(xs) < 2:
            continue
        road_type = int(road.get("type", -1))
        if 0 <= road_type <= 9:
            color = "#d0d0d0"
            lw = 1.2
        elif 20 <= road_type <= 29:
            color = "#111111"
            lw = 1.0
        else:
            color = "#888888"
            lw = 0.5
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.75, zorder=1)


def traffic_light_color(state):
    return TRAFFIC_LIGHT_COLORS.get(int(state), "#808080")


def draw_traffic_lights(ax, scenario, timestep, leader=None):
    for traffic_idx, traffic in enumerate(scenario.get("traffic_elements") or []):
        if traffic.get("type") != binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT:
            continue
        stop_line = traffic.get("stop_line") or []
        if len(stop_line) < 6:
            continue

        state = traffic_state_at(traffic, timestep)
        is_leader = leader and leader["kind"] == "red_light" and leader["idx"] == traffic_idx
        ax.plot(
            [stop_line[0], stop_line[3]],
            [stop_line[1], stop_line[4]],
            color="black",
            linewidth=5.0 if is_leader else 3.5,
            solid_capstyle="butt",
            alpha=0.95,
            zorder=13,
        )
        ax.plot(
            [stop_line[0], stop_line[3]],
            [stop_line[1], stop_line[4]],
            color=traffic_light_color(state),
            linewidth=3.2 if is_leader else 2.0,
            solid_capstyle="butt",
            alpha=0.95,
            zorder=14,
        )


def draw_agent(ax, agent, color, label=None, alpha=0.85, zorder=5):
    if agent.get("sim_x") == INVALID_POSITION:
        return
    corners = np.array(agent_corners(agent))
    poly = patches.Polygon(
        corners, closed=True, facecolor=color, edgecolor="black", linewidth=0.6, alpha=alpha, zorder=zorder
    )
    ax.add_patch(poly)
    if label:
        ax.text(agent["sim_x"], agent["sim_y"], label, ha="center", va="center", fontsize=7, zorder=zorder + 1)


def draw_corridor(ax, ego, corridor):
    heading = ego["sim_heading"]
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    start = corridor["start"]
    end = corridor["end"]
    half_w = corridor["half_width"]
    local = [(start, -half_w), (end, -half_w), (end, half_w), (start, half_w)]
    points = []
    for x, y in local:
        wx = ego["sim_x"] + x * cos_h - y * sin_h
        wy = ego["sim_y"] + x * sin_h + y * cos_h
        points.append((wx, wy))
    ax.add_patch(
        patches.Polygon(
            points, closed=True, facecolor="#4c78a8", edgecolor="#1f4e79", alpha=0.16, linewidth=1.0, zorder=3
        )
    )


def plot_scenario_frame(scenario, timestep, leader, corridor, out_path):
    agents = scenario["agents"]
    sdc_idx = corridor["sdc_idx"]
    ego = agents[sdc_idx]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140)
    ax.set_aspect("equal")
    draw_roads(ax, scenario)
    draw_traffic_lights(ax, scenario, timestep, leader)
    draw_corridor(ax, ego, corridor)

    for idx in candidate_indices(scenario):
        if idx < 0 or idx >= len(agents):
            continue
        if idx == sdc_idx:
            continue
        color = "#b7b7b7"
        zorder = 4
        if leader and leader["kind"] == "agent" and leader["idx"] == idx:
            color = "#e45756"
            zorder = 8
        draw_agent(ax, agents[idx], color, label=str(idx), alpha=0.65, zorder=zorder)

    draw_agent(ax, ego, "#2f80ed", label="SDC", alpha=0.95, zorder=10)

    if leader and leader["kind"] == "agent":
        other = agents[leader["idx"]]
        ax.plot(
            [ego["sim_x"], other["sim_x"]], [ego["sim_y"], other["sim_y"]], color="#e45756", linewidth=2.0, zorder=12
        )

    title = f"t={timestep} | SDC speed={ego.get('sim_speed', 0):.2f} m/s"
    if leader is None:
        title += " | leader=None"
    else:
        title += f" | leader={leader['kind']}:{leader['idx']} gap={leader['gap']:.1f}m speed={leader['speed']:.1f}"
    ax.set_title(title)

    radius = 55.0
    ax.set_xlim(ego["sim_x"] - radius, ego["sim_x"] + radius)
    ax.set_ylim(ego["sim_y"] - radius, ego["sim_y"] + radius)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_summary(history, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), dpi=140, sharex=True)
    for env_id, rows in sorted(history.items()):
        timesteps = [r["timestep"] for r in rows]
        gaps = [np.nan if r["leader"] is None else r["leader"]["gap"] for r in rows]
        speeds = [r["sdc_speed"] for r in rows]
        axes[0].plot(timesteps, gaps, marker=".", label=f"env {env_id}")
        axes[1].plot(timesteps, speeds, marker=".", label=f"env {env_id}")

    axes[0].set_ylabel("Leader gap (m)")
    axes[1].set_ylabel("SDC speed (m/s)")
    axes[1].set_xlabel("Timestep")
    axes[0].grid(alpha=0.2)
    axes[1].grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_env(args):
    return Drive(
        map_dir=args.map_dir,
        maps=args.maps,
        num_maps=args.num_maps,
        num_agents=args.num_envs * args.agents_per_env,
        min_agents_per_env=args.agents_per_env,
        max_agents_per_env=args.agents_per_env,
        scenario_length=args.steps + 8,
        resample_frequency=0,
        simulation_mode="gigaflow",
        control_mode="control_vehicles",
        sdc_controller=args.sdc_controller,
        non_sdc_controller=args.non_sdc_controller,
        action_type="discrete",
        dynamics_model="jerk",
        max_lane_segment_observations=8,
        lane_segment_dropout=0.0,
        max_boundary_segment_observations=8,
        boundary_segment_dropout=0.0,
        max_partner_observations=min(16, args.agents_per_env - 1),
        max_traffic_control_observations=4,
        compute_eval_metrics=False,
        seed=args.seed,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize IDM SDC leader selection in Gigaflow.")
    parser.add_argument("--output-dir", type=Path, default=Path("failure_runs/idm_leader_viz"))
    parser.add_argument("--map-dir", default="pufferlib/resources/drive/binaries/carla")
    parser.add_argument("--maps", default=None)
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--agents-per-env", type=int, default=12)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--frame-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sdc-controller", default="idm")
    parser.add_argument("--non-sdc-controller", default="idm")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(args)
    history = {}
    try:
        env.reset(seed=args.seed)
        actions = np.zeros(env.action_space.shape, dtype=env.actions.dtype)
        for timestep in range(args.steps):
            scenarios = normalize_scenarios(env.get_state())
            for env_id, scenario in enumerate(scenarios):
                leader, corridor = find_sdc_leader(scenario, timestep)
                if corridor is None:
                    continue
                sdc = scenario["agents"][corridor["sdc_idx"]]
                history.setdefault(env_id, []).append(
                    {
                        "timestep": timestep,
                        "leader": leader,
                        "sdc_speed": float(sdc.get("sim_speed", 0.0)),
                    }
                )
                if timestep % args.frame_interval == 0:
                    plot_scenario_frame(
                        scenario,
                        timestep,
                        leader,
                        corridor,
                        args.output_dir / f"env_{env_id:02d}_t_{timestep:04d}.png",
                    )
            env.step(actions)
    finally:
        env.close()

    plot_summary(history, args.output_dir / "leader_summary.png")
    print(f"Wrote IDM leader visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
