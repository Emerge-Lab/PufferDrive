import argparse
import os
import numpy as np
import pandas as pd
import glob

from pufferlib.pufferl import load_config, load_env, load_policy
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator


def replace_rollouts_with_gt(simulated_traj, gt_traj, num_replacements):
    """Replace first N rollouts with ground truth."""
    modified = {}
    for key in simulated_traj:
        if key in ["x", "y", "z", "heading"]:
            modified[key] = simulated_traj[key].copy()
            modified[key][:, :num_replacements, :] = np.broadcast_to(
                gt_traj[key], (gt_traj[key].shape[0], num_replacements, gt_traj[key].shape[2])
            )
        else:
            modified[key] = simulated_traj[key].copy()
    return modified


def run_evaluation_for_folder(config, args, folder_path, folder_name, policy):
    print(f"Processing folder: {folder_name}")

    # Update config for this folder
    config["env"]["map_dir"] = folder_path
    # Count maps in folder
    num_maps = len(glob.glob(os.path.join(folder_path, "*.bin")))
    config["env"]["num_maps"] = num_maps
    config["env"]["use_all_maps"] = True

    print(f"  Found {num_maps} maps in {folder_path}")

    # Reload environment with new config
    # We need to reload env because map_dir changed
    vecenv = load_env(args.env, config)

    # We can reuse the policy if it's compatible, but load_policy takes vecenv.
    # If we passed policy in, we assume it works.
    # However, let's just use the passed policy.

    evaluator = WOSACEvaluator(config)

    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    print(f"Number of scenarios: {len(np.unique(gt_trajectories['scenario_id']))}")
    print(f"Number of controlled agents: {gt_trajectories['x'].shape[0]}")
    print(f"Number of evaluated agents: {np.sum(gt_trajectories['track_id'] >= 0)}")

    simulated_trajectories = evaluator.collect_simulated_trajectories(config, vecenv, policy)
    agent_state = vecenv.driver_env.get_global_agent_state()
    road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()

    # Evaluate the model and save results
    output_csv = os.path.join(folder_path, f"results_{folder_name}_model.csv")
    if os.path.exists(output_csv):
        print(f"  Skipping, results already exist: {output_csv}")
        vecenv.close()
        return

    print(f"  Computing metrics for model evaluation")
    # Use simulated_trajectories as is (no GT replacement)
    scene_results = evaluator.compute_metrics(
        gt_trajectories, simulated_trajectories, agent_state, road_edge_polylines, aggregate_results=False
    )
    scene_results.to_csv(output_csv)
    print(f"  Saved results to {output_csv}")

    vecenv.close()


def main():
    parser = argparse.ArgumentParser(description="Run large scale WOSAC evaluation")
    parser.add_argument("--env", default="puffer_drive")
    parser.add_argument("--config", default="config/ocean/drive.ini")
    parser.add_argument("--base_dir", default="resources/drive/binaries/validation_10k")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument(
        "--load-model-path", type=str, default="pufferdrive_weights.pt", help="Path to model weights to evaluate"
    )

    # Parse known args to extract shard info, leaving the rest for load_config
    args, unknown_args = parser.parse_known_args()

    # Update sys.argv to remove the args we just parsed, so load_config doesn't choke on them
    # We keep sys.argv[0] and append the unknown args
    import sys

    sys.argv = [sys.argv[0]] + unknown_args

    # Initial config load to get general settings
    config = load_config(args.env)
    config["vec"]["backend"] = "PufferEnv"
    config["vec"]["num_envs"] = 1
    config["eval"]["enabled"] = True
    config["eval"]["wosac_num_rollouts"] = 32

    config["env"]["num_agents"] = config["eval"]["wosac_num_agents"]
    config["env"]["init_mode"] = config["eval"]["wosac_init_mode"]
    config["env"]["control_mode"] = config["eval"]["wosac_control_mode"]
    config["env"]["init_steps"] = config["eval"]["wosac_init_steps"]
    config["env"]["goal_behavior"] = config["eval"]["wosac_goal_behavior"]

    # Load policy once (assuming it doesn't depend on specific map set, just observation space)
    # We need a dummy env to load the policy
    # Let's use the first folder to initialize
    subfolders = sorted([d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))])
    if not subfolders:
        print(f"No subfolders found in {args.base_dir}")
        return

    first_folder = os.path.join(args.base_dir, subfolders[0])
    config["env"]["map_dir"] = first_folder
    config["env"]["num_maps"] = len(glob.glob(os.path.join(first_folder, "*.bin")))
    config["env"]["use_all_maps"] = True

    print("Loading policy...")
    dummy_vecenv = load_env(args.env, config)
    # Pass load-model-path to config for load_policy
    config["load_model_path"] = args.load_model_path
    policy = load_policy(config, dummy_vecenv, args.env)
    dummy_vecenv.close()
    print(f"Policy loaded from {args.load_model_path}.")

    # Sharding logic
    total_folders = len(subfolders)
    folders_per_shard = int(np.ceil(total_folders / args.num_shards))
    start_idx = args.shard_id * folders_per_shard
    end_idx = min((args.shard_id + 1) * folders_per_shard, total_folders)

    shard_folders = subfolders[start_idx:end_idx]
    print(
        f"Shard {args.shard_id}/{args.num_shards}: Processing folders {start_idx} to {end_idx - 1} ({len(shard_folders)} folders)"
    )

    for folder_name in shard_folders:
        folder_path = os.path.join(args.base_dir, folder_name)
        run_evaluation_for_folder(config, args, folder_path, folder_name, policy)


if __name__ == "__main__":
    main()
