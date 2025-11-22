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

    # Hardcode environment name
    env_name = "puffer_drive"

    # Find all checkpoint files
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_files = sorted(glob.glob(str(checkpoint_dir / "*.pt")))

    if not checkpoint_files:
        print(f"No .pt files found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint files")

    # Load base config without CLI parsing
    config = load_config_programmatic(env_name)

    # Results storage
    wosac_results = []
    collision_results = []

    # === WOSAC Evaluation ===
    if not args.skip_wosac:
        print("\n" + "=" * 80)
        print("STARTING WOSAC EVALUATION")
        print("=" * 80)

        # Configure for WOSAC
        config["wosac"]["enabled"] = True
        config["vec"]["backend"] = "PufferEnv"
        config["vec"]["num_envs"] = 1
        config["env"]["num_agents"] = config["wosac"]["num_total_wosac_agents"]
        config["env"]["init_mode"] = config["wosac"]["init_mode"]
        config["env"]["control_mode"] = config["wosac"]["control_mode"]
        config["env"]["init_steps"] = config["wosac"]["init_steps"]
        config["env"]["goal_behavior"] = config["wosac"]["goal_behavior"]
        config["env"]["goal_radius"] = config["wosac"]["goal_radius"]

        # Create environment for WOSAC
        vecenv_wosac = load_env(env_name, config)

        # Evaluate ground truth
        try:
            gt_result = evaluate_wosac_ground_truth(config, vecenv_wosac)
            wosac_results.append(gt_result)
        except Exception as e:
            print(f"Error evaluating ground truth: {e}")

        # Evaluate random policy
        try:
            policy_template = load_policy(config, vecenv_wosac, env_name)
            random_result = evaluate_wosac_random(config, vecenv_wosac, policy_template)
            wosac_results.append(random_result)
        except Exception as e:
            print(f"Error evaluating random policy: {e}")

        # Evaluate each checkpoint
        for checkpoint_path in checkpoint_files:
            checkpoint_name = Path(checkpoint_path).stem
            print(f"\nEvaluating checkpoint: {checkpoint_name}")

            try:
                # Load policy
                policy = load_policy(config, vecenv_wosac, env_name)
                state_dict = torch.load(checkpoint_path, map_location=config["train"]["device"])
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                policy.load_state_dict(state_dict)

                # Evaluate
                result = evaluate_wosac(config, vecenv_wosac, policy, policy_name=checkpoint_name)
                wosac_results.append(result)

            except Exception as e:
                print(f"Error evaluating {checkpoint_name}: {e}")

        vecenv_wosac.close()

    # === Collision Rate Evaluation ===
    if not args.skip_collision:
        print("\n" + "=" * 80)
        print("STARTING COLLISION RATE EVALUATION")
        print("=" * 80)

        # Configure for SDC-only collision evaluation
        config["vec"]["backend"] = "PufferEnv"
        config["vec"]["num_envs"] = 1
        config["env"]["num_agents"] = 1
        config["env"]["control_mode"] = "control_sdc_only"
        config["env"]["init_mode"] = "create_all_valid"
        config["env"]["init_steps"] = 10

        # Create environment for collision evaluation
        vecenv_human_replay = load_env(env_name, config)

        # Evaluate random policy
        try:
            random_collision_result = evaluate_collision_rate_random(
                config, vecenv_human_replay, num_episodes=args.num_collision_episodes
            )
            collision_results.append(random_collision_result)
        except Exception as e:
            print(f"Error evaluating random policy collision rate: {e}")

        # Evaluate each checkpoint
        for checkpoint_path in checkpoint_files:
            checkpoint_name = Path(checkpoint_path).stem
            print(f"\nEvaluating checkpoint collision rate: {checkpoint_name}")

            try:
                # Load policy
                policy = load_policy(config, vecenv_human_replay, env_name)
                state_dict = torch.load(checkpoint_path, map_location=config["train"]["device"])
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                policy.load_state_dict(state_dict)

                # Evaluate
                result = evaluate_collision_rate(
                    config,
                    vecenv_human_replay,
                    policy,
                    policy_name=checkpoint_name,
                    num_episodes=args.num_collision_episodes,
                )
                collision_results.append(result)

            except Exception as e:
                print(f"Error evaluating {checkpoint_name} collision rate: {e}")

        vecenv_human_replay.close()

    # === Save Results ===
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Create DataFrames
    if wosac_results:
        df_wosac = pd.DataFrame(wosac_results)
        wosac_output = args.output.replace(".csv", "_wosac.csv")
        df_wosac.to_csv(wosac_output, index=False)
        print(f"\nWOSAC results saved to {wosac_output}")
        print("\nWOSAC Results Summary:")
        print(df_wosac.to_string(index=False))

    if collision_results:
        df_collision = pd.DataFrame(collision_results)
        collision_output = args.output.replace(".csv", "_collision.csv")
        df_collision.to_csv(collision_output, index=False)
        print(f"\nCollision results saved to {collision_output}")
        print("\nCollision Results Summary:")
        print(df_collision.to_string(index=False))

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
