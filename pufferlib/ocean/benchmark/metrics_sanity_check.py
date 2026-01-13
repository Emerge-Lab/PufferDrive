"""
Comprehensive evaluation script for Drive environment checkpoints.

Evaluates all .pt checkpoints in a folder using:
1. WOSAC metrics (realism, ADE, likelihood metrics)
2. Collision rates (SDC-only control mode)

Includes baselines for ground truth and random policy.
"""

import argparse
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from pufferlib.pufferl import load_env, load_policy
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
import pufferlib.pytorch


def evaluate_wosac(config, vecenv, policy, policy_name="policy"):
    """Run WOSAC evaluation for a given policy."""
    print(f"Running WOSAC evaluation for {policy_name}...")

    evaluator = WOSACEvaluator(config)

    # Collect ground truth trajectories
    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    # Collect simulated trajectories
    simulated_trajectories = evaluator.collect_simulated_trajectories(config, vecenv, policy)
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()

    results = {}
    for num_gt in [0, 1, 2, 8, 16, 32]:
        modified_sim = replace_rollouts_with_gt(simulated_trajectories, gt_trajectories, num_gt)
        scene_results = evaluator.compute_metrics(gt_trajectories, modified_sim, agent_state, road_edge_polylines)

        results[num_gt] = {
            "ade": scene_results["ade"].mean(),
            "min_ade": scene_results["min_ade"].mean(),
            "likelihood_linear_speed": scene_results["likelihood_linear_speed"].mean(),
            "likelihood_linear_acceleration": scene_results["likelihood_linear_acceleration"].mean(),
            "likelihood_angular_speed": scene_results["likelihood_angular_speed"].mean(),
            "likelihood_angular_acceleration": scene_results["likelihood_angular_acceleration"].mean(),
            "likelihood_distance_to_nearest_object": scene_results["likelihood_distance_to_nearest_object"].mean(),
            "likelihood_time_to_collision": scene_results["likelihood_time_to_collision"].mean(),
            "likelihood_collision_indication": scene_results["likelihood_collision_indication"].mean(),
            "likelihood_distance_to_road_edge": scene_results["likelihood_distance_to_road_edge"].mean(),
            "likelihood_offroad_indication": scene_results["likelihood_offroad_indication"].mean(),
            "realism_meta_score": scene_results["realism_meta_score"].mean(),
        }

    return results


def format_results_table(results):
    lines = [
        "## WOSAC Log-Likelihood Validation Results\n",
        "| GT Rollouts | ADE    | minADE | Linear Speed | Linear Accel | Angular Speed | Angular Accel | Dist Obj | TTC    | Collision | Dist Road | Offroad | Metametric |",
        "|-------------|--------|--------|--------------|--------------|---------------|---------------|----------|--------|-----------|-----------|---------|------------|\n",
    ]

    for num_gt in sorted(results.keys()):
        label = f"{num_gt:2d} (random)" if num_gt == 0 else f"{num_gt:2d} (all GT)" if num_gt == 32 else f"{num_gt:2d}"
        r = results[num_gt]
        lines.append(
            f"| {label:11s} | {r['ade']:6.4f} | {r['min_ade']:6.4f} | {r['likelihood_linear_speed']:12.4f} | "
            f"{r['likelihood_linear_acceleration']:12.4f} | {r['likelihood_angular_speed']:13.4f} | "
            f"{r['likelihood_angular_acceleration']:13.4f} | {r['likelihood_distance_to_nearest_object']:8.4f} | "
            f"{r['likelihood_time_to_collision']:6.4f} | {r['likelihood_collision_indication']:9.4f} | "
            f"{r['likelihood_distance_to_road_edge']:9.4f} | {r['likelihood_offroad_indication']:7.4f} | {r['realism_meta_score']:10.4f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Drive environment checkpoints", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint_dir", type=str, help="Directory containing .pt checkpoint files")
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.csv",
        help="Output CSV file path (default: evaluation_results.csv)",
    )
    parser.add_argument(
        "--num-collision-episodes",
        type=int,
        default=100,
        help="Number of episodes for collision rate evaluation (default: 100)",
    )
    parser.add_argument("--skip-wosac", action="store_true", help="Skip WOSAC evaluation")
    parser.add_argument("--skip-collision", action="store_true", help="Skip collision rate evaluation")
    args = parser.parse_args()

    config = load_config(args.env)
    config["vec"]["backend"] = "PufferEnv"
    config["vec"]["num_envs"] = 1
    config["eval"]["enabled"] = True
    config["eval"]["wosac_num_rollouts"] = 32
    config["env"]["map_dir"] = config["eval"]["map_dir"]
    config["env"]["num_maps"] = config["eval"]["num_maps"]
    config["env"]["use_all_maps"] = True

    config["env"]["num_agents"] = config["eval"]["wosac_num_agents"]
    config["env"]["init_mode"] = config["eval"]["wosac_init_mode"]
    config["env"]["control_mode"] = config["eval"]["wosac_control_mode"]
    config["env"]["init_steps"] = config["eval"]["wosac_init_steps"]
    config["env"]["goal_behavior"] = config["eval"]["wosac_goal_behavior"]

    vecenv = load_env(args.env, config)
    policy = load_policy(config, vecenv, args.env)

    results = run_validation_experiment(config, vecenv, policy)
    print("\n" + format_results_table(results))


if __name__ == "__main__":
    main()
