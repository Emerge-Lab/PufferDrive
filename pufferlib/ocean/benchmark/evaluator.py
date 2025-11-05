"""WOSAC evaluation class for PufferDrive."""

import torch
import time
import numpy as np
import pandas as pd
from pprint import pprint
from typing import Dict, Optional
import matplotlib.pyplot as plt

import pufferlib
from pufferlib.ocean.benchmark import metrics
from pufferlib.ocean.benchmark import estimators


class WOSACEvaluator:
    """Evaluates policys on the Waymo Open Sim Agent Challenge (WOSAC) in PufferDrive. Info and links in the readme."""

    def __init__(self, config: Dict):
        self.config = config
        self.num_steps = 91  # Hardcoded for WOSAC (9.1s at 10Hz)
        self.init_steps = config.get("wosac", {}).get("init_steps", 0)
        self.sim_steps = self.num_steps - self.init_steps
        self.show_dashboard = config.get("wosac", {}).get("dashboard", False)
        self.num_rollouts = config.get("wosac", {}).get("num_rollouts", 32)

    def collect_ground_truth_trajectories(self, puffer_env):
        """Collect ground truth data for evaluation.
        Returns:
            trajectories: dict with keys 'x', 'y', 'z', 'heading', 'id'
                        each of shape (num_agents, 1, num_steps) for trajectory data
        """
        return puffer_env.get_ground_truth_trajectories()

    def collect_simulated_trajectories(self, args, puffer_env, policy):
        """Roll out policy in env and collect trajectories.
        Returns:
            trajectories: dict with keys 'x', 'y', 'z', 'heading' each of shape
                (num_agents, num_rollouts, num_steps)
        """

        driver = puffer_env.driver_env
        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]

        trajectories = {
            "x": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "id": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.int32),
        }

        for rollout_idx in range(self.num_rollouts):
            print(f"\rCollecting rollout {rollout_idx + 1}/{self.num_rollouts}...", end="", flush=True)
            obs, info = puffer_env.reset()
            state = {}
            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                )

            for time_idx in range(self.sim_steps):
                # Get global state
                agent_state = driver.get_global_agent_state()
                trajectories["x"][:, rollout_idx, time_idx] = agent_state["x"]
                trajectories["y"][:, rollout_idx, time_idx] = agent_state["y"]
                trajectories["z"][:, rollout_idx, time_idx] = agent_state["z"]
                trajectories["heading"][:, rollout_idx, time_idx] = agent_state["heading"]
                trajectories["id"][:, rollout_idx, time_idx] = agent_state["id"]

                # Step policy
                with torch.no_grad():
                    ob_tensor = torch.as_tensor(obs).to(device)
                    logits, value = policy.forward_eval(ob_tensor, state)
                    action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                    action_np = action.cpu().numpy().reshape(puffer_env.action_space.shape)

                if isinstance(logits, torch.distributions.Normal):
                    action_np = np.clip(action_np, puffer_env.action_space.low, puffer_env.action_space.high)

                obs, _, _, _, _ = puffer_env.step(action_np)

        return trajectories

    def compute_metrics(
        self,
        ground_truth_trajectories: Dict,
        simulated_trajectories: Dict,
    ) -> Dict:
        """Compute realism metrics comparing simulated and ground truth trajectories.

        Args:
            ground_truth_trajectories: Dict with keys ['x', 'y', 'z', 'heading', 'id']
                                Each trajectory has shape (n_agents, n_rollouts, n_steps)
            simulated_trajectories: Dict with same keys plus 'scenario_id'
                                shape (n_agents, n_steps) for trajectories
                                shape (n_agents,) for id
                                list of length n_agents for scenario_id

        Note: z-position currently not used.

        Returns:
            Dictionary with scores per scenario_id
        """
        # Ensure the id order matches exactly for simulated and ground truth
        assert np.array_equal(simulated_trajectories["id"][:, 0:1, 0], ground_truth_trajectories["id"]), (
            "Agent IDs don't match between simulated and ground truth trajectories"
        )

        # Extract trajectories
        sim_x = simulated_trajectories["x"]
        sim_y = simulated_trajectories["y"]
        sim_heading = simulated_trajectories["heading"]
        ref_x = ground_truth_trajectories["x"]
        ref_y = ground_truth_trajectories["y"]
        ref_heading = ground_truth_trajectories["heading"]
        ref_valid = ground_truth_trajectories["valid"]

        breakpoint()

        # Compute features
        # Kinematics-related features
        sim_linear_speed, sim_linear_accel, sim_angular_speed, sim_angular_accel = metrics.compute_kinematic_features(
            sim_x, sim_y, sim_heading
        )

        ref_linear_speed, ref_linear_accel, ref_angular_speed, ref_angular_accel = metrics.compute_kinematic_features(
            ref_x, ref_y, ref_heading
        )

        # Get the log speed (linear and angular) validity. Since this is computed by
        # a delta between steps i-1 and i+1, we verify that both of these are
        # valid (logical and).
        speed_validity, acceleration_validity = metrics.compute_kinematic_validity(ref_valid)

        # Compute realism metrics
        # Average Displacement Error (ADE) and minADE
        # Note: This metric is not included in the scoring meta-metric, as per WOSAC rules.
        ade, min_ade = metrics.compute_displacement_error(sim_x, sim_y, ref_x, ref_y, ref_valid)

        # Log-likelihood metrics
        # Kinematic features log-likelihoods
        linear_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_linear_speed,
            sim_values=sim_linear_speed,
            treat_timesteps_independently=True,
            min_val=0.0,
            max_val=25.0,
            num_bins=10,
            sanity_check=True,
        )

        # Compute final normalized scores
        # Each of these metrics is in [0, 1], higher is better

        # Input: [num_agents, num_steps]
        speed_likelihood = np.exp(
            metrics._reduce_average_with_validity(
                linear_speed_log_likelihood,
                speed_validity[:, 0, :],
                axis=1,  # Average over time steps
            )
        )

        # Get agent IDs and scenario IDs
        agent_ids = ground_truth_trajectories["id"]
        scenario_ids = ground_truth_trajectories["scenario_id"]

        df = pd.DataFrame(
            {
                "agent_id": agent_ids.flatten(),
                "scenario_id": scenario_ids.flatten(),
                "ade": ade,
                "min_ade": min_ade,
                "likelihood_speed": speed_likelihood,
            }
        )

        # Aggregate results per scenario_id
        scene_level_results = df.groupby("scenario_id")[["ade", "min_ade", "likelihood_speed"]].mean()

        print(f"\n Scene-level results:\n")
        print(scene_level_results)

        print(f"\n Full agent-level results:\n")
        print(df)

        return scene_level_results

    def _quick_sanity_check(self, gt_trajectories, simulated_trajectories):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        agent_idx = 0  # Visualize the first agent
        axs[0].set_title(f"Agent ID: {simulated_trajectories['id'][agent_idx, 0][0]}")
        axs[0].scatter(
            simulated_trajectories["x"][agent_idx, :, :],
            simulated_trajectories["y"][agent_idx, :, :],
            color="b",
            alpha=0.1,
            label="Simulated",
        )
        axs[0].scatter(
            gt_trajectories["x"][agent_idx, :, :],
            gt_trajectories["y"][agent_idx, :, :],
            color="g",
            label="Ground Truth",
        )
        axs[0].scatter(
            gt_trajectories["x"][agent_idx, 0, 0],
            gt_trajectories["y"][agent_idx, 0, 0],
            color="purple",
            marker="*",
            s=200,
            label="GT start",
            zorder=5,
            alpha=0.5,
        )
        axs[0].scatter(
            simulated_trajectories["x"][agent_idx, :, 0],
            simulated_trajectories["y"][agent_idx, :, 0],
            color="purple",
            marker="*",
            s=200,
            label="Agent start",
            zorder=5,
            alpha=0.5,
        )
        axs[0].set_xlabel("X Position")
        axs[0].set_ylabel("Y Position")
        axs[0].legend()

        axs[1].set_title(f"Heading timeseries; ID: {simulated_trajectories['id'][agent_idx, 0][0]}")
        time_steps = list(range(self.sim_steps))
        for r in range(self.num_rollouts):
            axs[1].plot(
                time_steps, simulated_trajectories["heading"][agent_idx, r, :], color="b", alpha=0.1, label="Simulated"
            )
        axs[1].plot(time_steps, gt_trajectories["heading"][agent_idx, :, :].T, color="g", label="Ground Truth")
        axs[1].set_xlabel("Time Step")
        plt.savefig("trajectory_comparison.png")
