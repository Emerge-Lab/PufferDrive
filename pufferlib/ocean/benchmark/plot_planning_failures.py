"""
Visual validation script for planning evaluation failures.

Plots trajectories for scenarios where planning accuracy is 0,
showing collision and offroad events. SDC is marked distinctly.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from matplotlib.patches import Polygon

from pufferlib.pufferl import load_config, load_env, load_policy
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator, PlanningEvaluator
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


def plot_agent_trajectories(ax, traj, agent_mask, collisions, offroad, agent_length, agent_width, global_to_eval_idx):
    """Plot trajectories as bounding boxes with collision/offroad coloring."""
    scenario_agent_indices = np.where(agent_mask)[0]

    x = traj["x"][agent_mask, 0, :]
    y = traj["y"][agent_mask, 0, :]
    heading = traj["heading"][agent_mask, 0, :]
    valid = traj["valid"][agent_mask, 0, :].astype(bool)
    ids = traj["id"][agent_mask, 0]
    length = agent_length[agent_mask]
    width = agent_width[agent_mask]

    num_agents = x.shape[0]
    num_steps = x.shape[1]

    collision_agents = []
    offroad_agents = []
    sdc_idx = None

    for i in range(num_agents):
        # Skip agents with no valid timesteps
        if not np.any(valid[i]):
            continue

        is_sdc = ids[i] == -1
        if is_sdc:
            sdc_idx = i
            global_idx = scenario_agent_indices[i]
            eval_idx = global_to_eval_idx.get(global_idx, None)

        for t in range(num_steps):
            # Skip invalid timesteps
            if not valid[i, t]:
                continue

            box = torch.as_tensor(
                [[x[i, t], y[i, t], length[i], width[i], heading[i, t]]],
                dtype=torch.float32,
            )
            corners = get_2d_box_corners(box)[0].cpu().numpy()

            # Check collision/offroad only for SDC
            has_coll = False
            has_off = False
            if is_sdc and eval_idx is not None:
                has_coll = collisions[eval_idx, 0, t]
                has_off = offroad[eval_idx, 0, t]

            # Color priority: collision > offroad > time-based
            if has_coll:
                facecolor = "red"
                alpha = 0.6
                edgecolor = "darkred"
                linewidth = 0.5
                if i not in collision_agents:
                    collision_agents.append(i)
            elif has_off:
                facecolor = "orange"
                alpha = 0.4
                edgecolor = "darkorange"
                linewidth = 0.5
                if i not in offroad_agents:
                    offroad_agents.append(i)
            else:
                facecolor = plt.cm.viridis(t / num_steps)
                alpha = 0.3
                edgecolor = facecolor
                linewidth = 0.3

            # Thicker edges for SDC to distinguish it
            if is_sdc:
                linewidth = 1.5
                edgecolor = "lime" if not (has_coll or has_off) else edgecolor

            polygon = Polygon(corners, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha)
            ax.add_patch(polygon)

    return collision_agents, offroad_agents, sdc_idx


def main():
    parser = argparse.ArgumentParser(description="Visual validation of planning failures")
    parser.add_argument("--env", default="puffer_drive")
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="planning_failure_plots", help="Base output directory")
    args = parser.parse_args()

    # Load config
    config = load_config(args.env)
    if args.load_model_path:
        config["load_model_path"] = args.load_model_path
    config["vec"]["backend"] = "PufferEnv"
    config["vec"]["num_envs"] = 1
    config["eval"]["wosac_num_rollouts"] = 1
    config["env"]["init_steps"] = config["eval"]["wosac_init_steps"]
    config["env"]["map_dir"] = config["eval"]["map_dir"]
    config["env"]["num_maps"] = config["eval"]["num_maps"]
    config["env"]["use_all_maps"] = True

    # Extract model name from path for folder structure
    if args.load_model_path:
        model_name = os.path.basename(args.load_model_path).replace(".pt", "")
    else:
        model_name = "random_policy"

    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    print("Collecting ground truth trajectories...")
    # Collect GT trajectories (control_wosac mode)
    gt_config = config.copy()
    gt_config["env"]["control_mode"] = "control_wosac"
    gt_vecenv = load_env(args.env, gt_config)

    wosac_evaluator = WOSACEvaluator(config)
    gt_trajs = wosac_evaluator.collect_ground_truth_trajectories(gt_vecenv)
    gt_agent_state = gt_vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = gt_vecenv.driver_env.get_road_edge_polylines()

    print("Collecting planning trajectories...")
    # Collect planning trajectories (control_sdc_only mode)
    planning_config = config.copy()
    planning_config["env"]["control_mode"] = "control_sdc_only"
    planning_config["env"]["init_steps"] = config["eval"]["planning_init_steps"]
    planning_config["env"]["goal_behavior"] = config["eval"]["planning_goal_behavior"]
    planning_config["env"]["goal_radius"] = config["eval"]["wosac_goal_radius"]
    planning_vecenv = load_env(args.env, planning_config)
    policy = load_policy(config, planning_vecenv, args.env)

    planning_trajs = wosac_evaluator.collect_simulated_trajectories(planning_config, planning_vecenv, policy)

    print("Combining trajectories...")
    # Combine trajectories
    evaluator = PlanningEvaluator(config)
    combined_trajs, _ = evaluator.combine_trajectories(gt_trajs, planning_trajs)

    print("Computing metrics...")
    # Compute per-scenario metrics
    results = evaluator.compute_metrics(combined_trajs, gt_agent_state, road_edge_polylines, aggregate_results=False)

    # Find failing scenarios (accuracy = 0)
    failing_scenarios = results[results["accuracy"] == 0]

    if len(failing_scenarios) == 0:
        print("No failing scenarios found!")
        return

    print(f"Found {len(failing_scenarios)} failing scenarios")

    # Compute per-timestep collision/offroad for SDCs only
    eval_mask = combined_trajs["id"][:, 0] == -1
    scenario_ids = combined_trajs["scenario_id"]

    # Create mapping from global agent index to eval-only index
    eval_indices = np.where(eval_mask)[0]
    global_to_eval_idx = {global_idx: eval_idx for eval_idx, global_idx in enumerate(eval_indices)}

    _, collisions, _ = compute_interaction_features(
        combined_trajs["x"],
        combined_trajs["y"],
        combined_trajs["heading"],
        scenario_ids,
        gt_agent_state["length"],
        gt_agent_state["width"],
        eval_mask,
        device=config.get("train", {}).get("device", "cuda"),
    )

    _, offroad = compute_map_features(
        combined_trajs["x"][eval_mask],
        combined_trajs["y"][eval_mask],
        combined_trajs["heading"][eval_mask],
        scenario_ids[eval_mask],
        gt_agent_state["length"][eval_mask],
        gt_agent_state["width"][eval_mask],
        road_edge_polylines,
        device=config.get("train", {}).get("device", "cuda"),
    )

    # Plot each failing scenario
    for scenario_id in failing_scenarios.index:
        print(f"Plotting scenario {scenario_id}...")
        agent_mask = scenario_ids[:, 0] == scenario_id

        fig, ax = plt.subplots(figsize=(12, 10))

        plot_road_edges(ax, road_edge_polylines, scenario_id)
        collision_agents, offroad_agents, sdc_idx = plot_agent_trajectories(
            ax,
            combined_trajs,
            agent_mask,
            collisions,
            offroad,
            gt_agent_state["length"],
            gt_agent_state["width"],
            global_to_eval_idx,
        )

        # Fix scaling: compute axis limits from valid trajectory data only
        x_data = combined_trajs["x"][agent_mask, 0, :]
        y_data = combined_trajs["y"][agent_mask, 0, :]
        valid_data = combined_trajs["valid"][agent_mask, 0, :].astype(bool)

        # Only use valid coordinates
        x_valid = x_data[valid_data]
        y_valid = y_data[valid_data]

        if len(x_valid) > 0:
            x_min, x_max = np.min(x_valid), np.max(x_valid)
            y_min, y_max = np.min(y_valid), np.max(y_valid)

            # Add 10% padding
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Scenario {scenario_id} - Planning Failure (Accuracy = 0)")

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="red", alpha=0.6, label="Collision"),
            Patch(facecolor="orange", alpha=0.4, label="Offroad"),
            Patch(facecolor="white", edgecolor="lime", linewidth=2, label="SDC (Ego)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Colorbar for time
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Time (normalized)")

        # Summary text
        num_agents_in_scenario = agent_mask.sum()
        row = failing_scenarios.loc[scenario_id]
        summary = f"Agents: {num_agents_in_scenario}\n"
        summary += f"SDC: agent {sdc_idx}\n"
        summary += f"Collisions: {len(collision_agents)} agents\n"
        summary += f"Offroad: {len(offroad_agents)} agents\n"
        summary += f"Collision rate: {row['collision_indication']:.2f}\n"
        summary += f"Offroad rate: {row['offroad_indication']:.2f}"
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

        output_path = os.path.join(output_dir, f"scenario_{scenario_id}.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {output_path}")

    print(f"\nDone! Generated {len(failing_scenarios)} visualization files in {output_dir}/")


if __name__ == "__main__":
    main()
