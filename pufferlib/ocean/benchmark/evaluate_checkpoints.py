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
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from pufferlib.pufferl import load_env, load_policy, load_config
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

    # Optional sanity check
    if config["wosac"].get("sanity_check", False):
        evaluator._quick_sanity_check(gt_trajectories, simulated_trajectories)

    # Compute metrics
    results = evaluator.compute_metrics(gt_trajectories, simulated_trajectories)

    return {
        "policy": policy_name,
        "ade": results["ade"].mean(),
        "min_ade": results["min_ade"].mean(),
        "likelihood_linear_speed": results["likelihood_linear_speed"].mean(),
        "likelihood_linear_acceleration": results["likelihood_linear_acceleration"].mean(),
        "likelihood_angular_speed": results["likelihood_angular_speed"].mean(),
        "likelihood_angular_acceleration": results["likelihood_angular_acceleration"].mean(),
        "realism_metametric": results["realism_metametric"].mean(),
    }


def evaluate_wosac_ground_truth(config, vecenv):
    """Run WOSAC evaluation using ground truth as the policy (perfect replay)."""
    print("Running WOSAC evaluation for ground truth...")

    evaluator = WOSACEvaluator(config)

    # Collect ground truth trajectories
    gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

    # Use ground truth as simulated (perfect replay)
    # For this, we need to broadcast GT to match rollout dimensions
    simulated_trajectories = {}
    num_rollouts = config["wosac"]["num_rollouts"]

    for key in gt_trajectories:
        if key in ["x", "y", "z", "heading"]:
            # Broadcast ground truth across rollouts
            gt_shape = gt_trajectories[key].shape
            simulated_trajectories[key] = np.broadcast_to(
                gt_trajectories[key], (gt_shape[0], num_rollouts, gt_shape[2])
            ).copy()
        else:
            simulated_trajectories[key] = gt_trajectories[key].copy()

    # Compute metrics
    results = evaluator.compute_metrics(gt_trajectories, simulated_trajectories)

    return {
        "policy": "ground_truth",
        "ade": results["ade"].mean(),
        "min_ade": results["min_ade"].mean(),
        "likelihood_linear_speed": results["likelihood_linear_speed"].mean(),
        "likelihood_linear_acceleration": results["likelihood_linear_acceleration"].mean(),
        "likelihood_angular_speed": results["likelihood_angular_speed"].mean(),
        "likelihood_angular_acceleration": results["likelihood_angular_acceleration"].mean(),
        "realism_metametric": results["realism_metametric"].mean(),
    }


def evaluate_wosac_random(config, vecenv, policy):
    """Run WOSAC evaluation with a random policy."""
    print("Running WOSAC evaluation for random policy...")

    class RandomPolicy:
        """Random action policy wrapper."""

        def __init__(self, action_space):
            self.action_space = action_space

        def forward_eval(self, obs, state):
            batch_size = obs.shape[0]
            # Generate random actions
            if hasattr(self.action_space, "nvec"):  # MultiDiscrete
                actions = torch.tensor(
                    [[np.random.randint(0, n) for n in self.action_space.nvec] for _ in range(batch_size)],
                    dtype=torch.long,
                )
            else:  # Continuous
                actions = torch.tensor(self.action_space.sample()).unsqueeze(0).repeat(batch_size, 1)

            # Return dummy values for value estimate
            value = torch.zeros(batch_size, 1)
            return actions, value

    random_policy = RandomPolicy(vecenv.single_action_space)
    return evaluate_wosac(config, vecenv, random_policy, policy_name="random")


def evaluate_collision_rate(config, vecenv, policy, policy_name="policy", num_episodes=100):
    """Evaluate collision rate in SDC-only control mode."""
    print(f"Running collision rate evaluation for {policy_name}...")

    device = config["train"]["device"]
    num_agents = vecenv.num_agents

    # Initialize LSTM state if needed
    state = {}
    if config["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    # Track collision statistics
    total_steps = 0
    total_collision_count = 0
    total_collisions_per_agent = 0.0
    episodes_completed = 0

    ob, info = vecenv.reset()

    while episodes_completed < num_episodes:
        with torch.no_grad():
            ob_tensor = torch.as_tensor(ob).to(device)

            if hasattr(policy, "forward_eval"):
                logits, value = policy.forward_eval(ob_tensor, state)
            else:
                # Random policy
                if hasattr(vecenv.single_action_space, "nvec"):
                    action = np.array([[np.random.randint(0, n) for n in vecenv.single_action_space.nvec]])
                else:
                    action = vecenv.single_action_space.sample()
                    if isinstance(action, np.ndarray):
                        action = action.reshape(1, -1)
                ob, rewards, terminals, truncations, info = vecenv.step(action)
                total_steps += 1

                # Check for episode completion
                if terminals.any() or truncations.any():
                    episodes_completed += 1
                    # Extract metrics from final info (contains log data)
                    if info:
                        for info_dict in info:
                            if isinstance(info_dict, dict):
                                if "collision_rate" in info_dict:
                                    total_collision_count += int(info_dict["collision_rate"])
                                if "avg_collisions_per_agent" in info_dict:
                                    total_collisions_per_agent += float(info_dict["avg_collisions_per_agent"])
                continue

            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

            # Clip continuous actions
            if isinstance(logits, torch.distributions.Normal):
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

        ob, rewards, terminals, truncations, info = vecenv.step(action)
        total_steps += 1

        # Check for episode completion
        if terminals.any() or truncations.any():
            episodes_completed += 1
            # Extract metrics from final info (contains log data)
            if info:
                for info_dict in info:
                    if isinstance(info_dict, dict):
                        if "collision_rate" in info_dict:
                            total_collision_count += int(info_dict["collision_rate"])
                        if "avg_collisions_per_agent" in info_dict:
                            total_collisions_per_agent += float(info_dict["avg_collisions_per_agent"])

    # Calculate rates
    collision_rate = total_collision_count / num_episodes if num_episodes > 0 else 0
    avg_collisions_per_agent = total_collisions_per_agent / num_episodes if num_episodes > 0 else 0

    return {
        "policy": policy_name,
        "num_episodes": num_episodes,
        "total_steps": total_steps,
        "collision_rate": collision_rate,
        "avg_collisions_per_agent": avg_collisions_per_agent,
    }


def evaluate_collision_rate_random(config, vecenv, num_episodes=100):
    """Evaluate collision rate with random policy."""
    print("Running collision rate evaluation for random policy...")

    class RandomPolicy:
        """Dummy random policy."""

        pass

    random_policy = RandomPolicy()
    return evaluate_collision_rate(config, vecenv, random_policy, policy_name="random", num_episodes=num_episodes)


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

    # Temporarily modify sys.argv to prevent load_config from parsing our arguments
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep only script name

    try:
        config = load_config(env_name)
    finally:
        sys.argv = original_argv

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
