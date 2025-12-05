"""
Visual validation script for WOSAC collision and offroad detection.

Plots road edges and agent trajectories for a single scenario,
marking collision and offroad events.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from matplotlib.patches import Polygon

from pufferlib.pufferl import load_config, load_env, load_policy
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
from pufferlib.ocean.benchmark.metrics import compute_interaction_features, compute_map_features
from pufferlib.ocean.benchmark.geometry_utils import get_2d_box_corners


def plot_road_edges(ax, road_edge_polylines, scenario_id):
    """Plot road edge polylines for a specific scenario."""
    lengths = road_edge_polylines["lengths"]
    scenario_ids = road_edge_polylines["scenario_id"]
    x = road_edge_polylines["x"]
    y = road_edge_polylines["y"]

    pt_idx = 0
    for i in range(len(lengths)):
        length = lengths[i]
        if scenario_ids[i] == scenario_id:
            poly_x = x[pt_idx : pt_idx + length]
            poly_y = y[pt_idx : pt_idx + length]
            ax.plot(poly_x, poly_y, "k-", linewidth=1, alpha=0.7)
        pt_idx += length


def plot_agent_trajectories(ax, traj, agent_mask, rollout_idx, collisions, offroad, agent_length, agent_width):
    """Plot trajectories as bounding boxes with collision/offroad coloring."""
    x = traj["x"][agent_mask, rollout_idx, :]
    y = traj["y"][agent_mask, rollout_idx, :]
    heading = traj["heading"][agent_mask, rollout_idx, :]
    length = agent_length[agent_mask]
    width = agent_width[agent_mask]
    coll = collisions[agent_mask, rollout_idx, :]
    off = offroad[agent_mask, rollout_idx, :]

    num_agents = x.shape[0]
    num_steps = x.shape[1]

    collision_agents = []
    offroad_agents = []

    for i in range(num_agents):
        has_collision = np.any(coll[i])
        has_offroad = np.any(off[i])
        if has_collision:
            collision_agents.append(i)
        if has_offroad:
            offroad_agents.append(i)

        for t in range(num_steps):
            box = torch.as_tensor(
                [[x[i, t], y[i, t], length[i], width[i], heading[i, t]]],
                dtype=torch.float32,
            )
            corners = get_2d_box_corners(box)[0].cpu().numpy()

            if coll[i, t]:
                facecolor = "red"
                alpha = 0.6
                edgecolor = "red"
            elif off[i, t]:
                facecolor = "orange"
                alpha = 0.4
                edgecolor = "orange"
            else:
                facecolor = plt.cm.viridis(t / num_steps)
                alpha = 0.3
                edgecolor = facecolor

            polygon = Polygon(corners, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.3, alpha=alpha)
            ax.add_patch(polygon)

    return collision_agents, offroad_agents


def main():
    parser = argparse.ArgumentParser(description="Visual validation of collision/offroad detection")
    parser.add_argument("--env", default="puffer_drive")
    parser.add_argument("--output", default="visual_sanity_check.png")
    args = parser.parse_args()

    config = load_config(args.env)
    config["vec"]["backend"] = "PufferEnv"
    config["vec"]["num_envs"] = 1
    config["eval"]["enabled"] = True
    config["eval"]["wosac_num_rollouts"] = 1

    config["env"]["num_agents"] = config["eval"]["wosac_num_agents"]
    config["env"]["init_mode"] = config["eval"]["wosac_init_mode"]
    config["env"]["control_mode"] = config["eval"]["wosac_control_mode"]
    config["env"]["init_steps"] = config["eval"]["wosac_init_steps"]
    config["env"]["goal_behavior"] = config["eval"]["wosac_goal_behavior"]

    config["env"]["map_dir"] = config["eval"]["map_dir"]
    config["env"]["num_maps"] = config["eval"]["num_maps"]
    config["env"]["use_all_maps"] = True

    vecenv = load_env(args.env, config)
    policy = load_policy(config, vecenv, args.env)

    evaluator = WOSACEvaluator(config)
    gt_traj = evaluator.collect_ground_truth_trajectories(vecenv)
    sim_traj = evaluator.collect_simulated_trajectories(config, vecenv, policy)
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()

    scenario_ids = gt_traj["scenario_id"]
    agent_length = agent_state["length"]
    agent_width = agent_state["width"]

    # Compute per-timestep indicators
    num_agents = sim_traj["x"].shape[0]
    eval_mask = np.ones(num_agents, dtype=bool)

    device = torch.device("cpu")

    _, collisions, _ = compute_interaction_features(
        sim_traj["x"],
        sim_traj["y"],
        sim_traj["heading"],
        scenario_ids,
        agent_length,
        agent_width,
        eval_mask,
        device=device,
    )
    _, offroad = compute_map_features(
        sim_traj["x"],
        sim_traj["y"],
        sim_traj["heading"],
        scenario_ids,
        agent_length,
        agent_width,
        road_edge_polylines,
        device=device,
    )

    # Plot each scenario
    unique_scenarios = np.unique(scenario_ids[:, 0])

    for scenario_idx, target_scenario in enumerate(unique_scenarios):
        agent_mask = scenario_ids[:, 0] == target_scenario

        fig, ax = plt.subplots(figsize=(12, 10))

        plot_road_edges(ax, road_edge_polylines, target_scenario)
        collision_agents, offroad_agents = plot_agent_trajectories(
            ax,
            sim_traj,
            agent_mask,
            rollout_idx=0,
            collisions=collisions,
            offroad=offroad,
            agent_length=agent_length,
            agent_width=agent_width,
        )

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Scenario {target_scenario} - Collision/Offroad Detection")

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", alpha=0.6, label="Collision"),
            Patch(facecolor="orange", alpha=0.4, label="Offroad"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Colorbar for time
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Time (normalized)")

        # Summary text
        num_agents_in_scenario = agent_mask.sum()
        summary = f"Agents: {num_agents_in_scenario}\n"
        summary += f"Collisions: {len(collision_agents)} agents ({collision_agents})\n"
        summary += f"Offroad: {len(offroad_agents)} agents ({offroad_agents})"
        ax.text(
            0.02,
            0.98,
            summary,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        # Save with scenario index in filename
        output_base = args.output.rsplit(".", 1)
        if len(output_base) == 2:
            output_path = f"{output_base[0]}_{scenario_idx}.{output_base[1]}"
        else:
            output_path = f"{args.output}_{scenario_idx}"

        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Scenario {target_scenario}: {output_path}")
        print(f"  {summary.replace(chr(10), ', ')}")


if __name__ == "__main__":
    main()
