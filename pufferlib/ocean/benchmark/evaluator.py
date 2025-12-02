"""WOSAC evaluation class for PufferDrive."""

import torch
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import configparser
import os

import pufferlib
from pufferlib.ocean.benchmark import metrics
from pufferlib.ocean.benchmark import estimators


_METRIC_FIELD_NAMES = [
    "linear_speed",
    "linear_acceleration",
    "angular_speed",
    "angular_acceleration",
    "distance_to_nearest_object",
    "time_to_collision",
    "collision_indication",
    "distance_to_road_edge",
    "offroad_indication",
]


class WOSACEvaluator:
    """Evaluates policys on the Waymo Open Sim Agent Challenge (WOSAC) in PufferDrive. Info and links in the readme."""

    def __init__(self, config: Dict):
        self.config = config
        self.num_steps = 91  # Hardcoded for WOSAC (9.1s at 10Hz)
        self.init_steps = config.get("eval", {}).get("wosac_init_steps", 0)
        self.sim_steps = self.num_steps - self.init_steps
        self.num_rollouts = config.get("eval", {}).get("wosac_num_rollouts", 32)
        self.device = config.get("train", {}).get("device", "cuda")

        wosac_metrics_path = os.path.join(os.path.dirname(__file__), "wosac.ini")
        self.metrics_config = configparser.ConfigParser()
        self.metrics_config.read(wosac_metrics_path)

    def _compute_metametric(self, metrics: pd.Series) -> float:
        metametric = 0.0
        for field_name in _METRIC_FIELD_NAMES:
            likelihood_field_name = "likelihood_" + field_name
            weight = self.metrics_config.getfloat(field_name, "metametric_weight")
            metric_score = metrics[likelihood_field_name]
            metametric += weight * metric_score

        weight_sum = sum(self.metrics_config.getfloat(fn, "metametric_weight") for fn in _METRIC_FIELD_NAMES)
        return metametric / weight_sum

    def _get_histogram_params(self, metric_name: str):
        return (
            self.metrics_config.getfloat(metric_name, "histogram.min_val"),
            self.metrics_config.getfloat(metric_name, "histogram.max_val"),
            self.metrics_config.getint(metric_name, "histogram.num_bins"),
            self.metrics_config.getfloat(metric_name, "histogram.additive_smoothing_pseudocount"),
            self.metrics_config.getboolean(metric_name, "independent_timesteps"),
        )

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
        agent_state: Dict,
        road_edge_polylines: Dict,
        aggregate_results: bool = False,
    ) -> Dict:
        """Compute realism metrics comparing simulated and ground truth trajectories.

        Args:
            ground_truth_trajectories: Dict with keys ['x', 'y', 'z', 'heading', 'id', 'scenario_id', 'valid']
            simulated_trajectories: Dict with keys ['x', 'y', 'z', 'heading', 'id']
            agent_state: Dict with length and width of agents.
            road_edge_polylines: Dict with keys ['x', 'y', 'lengths', 'scenario_id']

        Note: z-position currently not used.

        Returns:
            Dictionary with scores per scenario_id
        """
        # Ensure the id order matches exactly for simulated and ground truth
        assert np.array_equal(simulated_trajectories["id"][:, 0:1, 0], ground_truth_trajectories["id"]), (
            "Agent IDs don't match between simulated and ground truth trajectories"
        )

        eval_mask = ground_truth_trajectories["id"][:, 0] >= 0

        # Extract trajectories
        sim_x = simulated_trajectories["x"]
        sim_y = simulated_trajectories["y"]
        sim_heading = simulated_trajectories["heading"]
        ref_x = ground_truth_trajectories["x"]
        ref_y = ground_truth_trajectories["y"]
        ref_heading = ground_truth_trajectories["heading"]
        ref_valid = ground_truth_trajectories["valid"]
        agent_length = agent_state["length"]
        agent_width = agent_state["width"]
        scenario_ids = ground_truth_trajectories["scenario_id"]

        # We evaluate the metrics only for the Tracks to Predict.
        eval_sim_x = sim_x[eval_mask]
        eval_sim_y = sim_y[eval_mask]
        eval_sim_heading = sim_heading[eval_mask]
        eval_ref_x = ref_x[eval_mask]
        eval_ref_y = ref_y[eval_mask]
        eval_ref_heading = ref_heading[eval_mask]
        eval_ref_valid = ref_valid[eval_mask]
        eval_agent_length = agent_length[eval_mask]
        eval_agent_width = agent_width[eval_mask]
        eval_scenario_ids = scenario_ids[eval_mask]

        # Compute features
        # Kinematics-related features
        sim_linear_speed, sim_linear_accel, sim_angular_speed, sim_angular_accel = metrics.compute_kinematic_features(
            eval_sim_x, eval_sim_y, eval_sim_heading
        )

        ref_linear_speed, ref_linear_accel, ref_angular_speed, ref_angular_accel = metrics.compute_kinematic_features(
            eval_ref_x, eval_ref_y, eval_ref_heading
        )

        # Get the log speed (linear and angular) validity. Since this is computed by
        # a delta between steps i-1 and i+1, we verify that both of these are
        # valid (logical and).
        speed_validity, acceleration_validity = metrics.compute_kinematic_validity(ref_valid[eval_mask])

        # Interaction-related features
        sim_signed_distances, sim_collision_per_step, sim_time_to_collision = metrics.compute_interaction_features(
            sim_x, sim_y, sim_heading, scenario_ids, agent_length, agent_width, eval_mask, device=self.device
        )

        ref_signed_distances, ref_collision_per_step, ref_time_to_collision = metrics.compute_interaction_features(
            ref_x,
            ref_y,
            ref_heading,
            scenario_ids,
            agent_length,
            agent_width,
            eval_mask,
            device=self.device,
            valid=ref_valid,
        )

        # Map-based features
        sim_distance_to_road_edge, sim_offroad_per_step = metrics.compute_map_features(
            eval_sim_x,
            eval_sim_y,
            eval_sim_heading,
            eval_scenario_ids,
            eval_agent_length,
            eval_agent_width,
            road_edge_polylines,
            device=self.device,
        )

        ref_distance_to_road_edge, ref_offroad_per_step = metrics.compute_map_features(
            eval_ref_x,
            eval_ref_y,
            eval_ref_heading,
            eval_scenario_ids,
            eval_agent_length,
            eval_agent_width,
            road_edge_polylines,
            device=self.device,
            valid=eval_ref_valid,
        )

        # Compute realism metrics
        # Average Displacement Error (ADE) and minADE
        # Note: This metric is not included in the scoring meta-metric, as per WOSAC rules.
        ade, min_ade = metrics.compute_displacement_error(
            eval_sim_x, eval_sim_y, eval_ref_x, eval_ref_y, eval_ref_valid
        )

        # Log-likelihood metrics
        # Kinematic features log-likelihoods
        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "linear_speed"
        )
        linear_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_linear_speed,
            sim_values=sim_linear_speed,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "linear_acceleration"
        )
        linear_accel_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_linear_accel,
            sim_values=sim_linear_accel,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "angular_speed"
        )
        angular_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_angular_speed,
            sim_values=sim_angular_speed,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "angular_acceleration"
        )
        angular_accel_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_angular_accel,
            sim_values=sim_angular_accel,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "distance_to_nearest_object"
        )
        distance_to_nearest_object_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_signed_distances,
            sim_values=sim_signed_distances,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "time_to_collision"
        )
        time_to_collision_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_time_to_collision,
            sim_values=sim_time_to_collision,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        # Map-based features log-likelihoods
        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "distance_to_road_edge"
        )
        distance_to_road_edge_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_distance_to_road_edge,
            sim_values=sim_distance_to_road_edge,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        speed_likelihood = np.exp(
            metrics._reduce_average_with_validity(
                linear_speed_log_likelihood,
                speed_validity[:, 0, :],
                axis=1,
            )
        )

        accel_likelihood = np.exp(
            metrics._reduce_average_with_validity(
                linear_accel_log_likelihood,
                acceleration_validity[:, 0, :],
                axis=1,
            )
        )

        angular_speed_likelihood = np.exp(
            metrics._reduce_average_with_validity(
                angular_speed_log_likelihood,
                speed_validity[:, 0, :],
                axis=1,
            )
        )

        angular_accel_likelihood = np.exp(
            metrics._reduce_average_with_validity(
                angular_accel_log_likelihood,
                acceleration_validity[:, 0, :],
                axis=1,
            )
        )

        distance_to_nearest_object_likelihood = np.exp(
            metrics._reduce_average_with_validity(
                distance_to_nearest_object_log_likelihood,
                eval_ref_valid[:, 0, :],
                axis=1,
            )
        )

        time_to_collision_likelihood = np.exp(
            metrics._reduce_average_with_validity(
                time_to_collision_log_likelihood,
                eval_ref_valid[:, 0, :],
                axis=1,
            )
        )

        distance_to_road_edge_likelihood = np.exp(
            metrics._reduce_average_with_validity(
                distance_to_road_edge_log_likelihood,
                eval_ref_valid[:, 0, :],
                axis=1,
            )
        )

        # Collision likelihood is computed by aggregating in time. For invalid objects
        # in the logged scenario, we need to filter possible collisions in simulation.
        # `sim_collision_indication` shape: (n_samples, n_objects).

        sim_collision_indication = np.any(np.where(eval_ref_valid, sim_collision_per_step, False), axis=2)
        ref_collision_indication = np.any(np.where(eval_ref_valid, ref_collision_per_step, False), axis=2)

        sim_num_collisions = np.sum(sim_collision_indication, axis=1)
        ref_num_collisions = np.sum(ref_collision_indication, axis=1)

        collision_log_likelihood = estimators.log_likelihood_estimate_scenario_level(
            log_values=ref_collision_indication[:, 0],
            sim_values=sim_collision_indication,
            min_val=0.0,
            max_val=1.0,
            num_bins=2,
            use_bernoulli=True,
        )
        collision_likelihood = np.exp(collision_log_likelihood)

        # Offroad likelihood (same pattern as collision)
        sim_offroad_indication = np.any(np.where(eval_ref_valid, sim_offroad_per_step, False), axis=2)
        ref_offroad_indication = np.any(np.where(eval_ref_valid, ref_offroad_per_step, False), axis=2)

        sim_num_offroad = np.sum(sim_offroad_indication, axis=1)
        ref_num_offroad = np.sum(ref_offroad_indication, axis=1)

        offroad_log_likelihood = estimators.log_likelihood_estimate_scenario_level(
            log_values=ref_offroad_indication[:, 0],
            sim_values=sim_offroad_indication,
            min_val=0.0,
            max_val=1.0,
            num_bins=2,
            use_bernoulli=True,
        )
        offroad_likelihood = np.exp(offroad_log_likelihood)

        # Get agent IDs
        eval_agent_ids = ground_truth_trajectories["id"][eval_mask]

        df = pd.DataFrame(
            {
                "agent_id": eval_agent_ids.flatten(),
                "scenario_id": eval_scenario_ids.flatten(),
                "num_collisions_sim": sim_num_collisions.flatten(),
                "num_collisions_ref": ref_num_collisions.flatten(),
                "num_offroad_sim": sim_num_offroad.flatten(),
                "num_offroad_ref": ref_num_offroad.flatten(),
                "ade": ade,
                "min_ade": min_ade,
                "likelihood_linear_speed": speed_likelihood,
                "likelihood_linear_acceleration": accel_likelihood,
                "likelihood_angular_speed": angular_speed_likelihood,
                "likelihood_angular_acceleration": angular_accel_likelihood,
                "likelihood_distance_to_nearest_object": distance_to_nearest_object_likelihood,
                "likelihood_time_to_collision": time_to_collision_likelihood,
                "likelihood_collision_indication": collision_likelihood,
                "likelihood_distance_to_road_edge": distance_to_road_edge_likelihood,
                "likelihood_offroad_indication": offroad_likelihood,
            }
        )

        scene_level_results = df.groupby("scenario_id")[
            [
                "ade",
                "min_ade",
                "num_collisions_sim",
                "num_collisions_ref",
                "num_offroad_sim",
                "num_offroad_ref",
                "likelihood_linear_speed",
                "likelihood_linear_acceleration",
                "likelihood_angular_speed",
                "likelihood_angular_acceleration",
                "likelihood_distance_to_nearest_object",
                "likelihood_time_to_collision",
                "likelihood_collision_indication",
                "likelihood_distance_to_road_edge",
                "likelihood_offroad_indication",
            ]
        ].mean()

        scene_level_results["realism_meta_score"] = scene_level_results.apply(self._compute_metametric, axis=1)
        scene_level_results["num_agents"] = df.groupby("scenario_id").size()
        scene_level_results = scene_level_results[
            ["num_agents"] + [col for col in scene_level_results.columns if col != "num_agents"]
        ]

        if aggregate_results:
            aggregate_metrics = scene_level_results.mean().to_dict()
            aggregate_metrics["total_num_agents"] = scene_level_results["num_agents"].sum()
            # Convert numpy types to Python native types
            return {k: v.item() if hasattr(v, "item") else v for k, v in aggregate_metrics.items()}
        else:
            print("\n Scene-level results:\n")
            print(scene_level_results)

            print(f"\n Overall realism meta score: {scene_level_results['realism_meta_score'].mean():.4f}")
            print(f"\n Overall minADE: {scene_level_results['min_ade'].mean():.4f}")
            print(f"\n Overall ADE: {scene_level_results['ade'].mean():.4f}")

            # print(f"\n Full agent-level results:\n")
            # print(df)
            return scene_level_results

    def _quick_sanity_check(self, gt_trajectories, simulated_trajectories, agent_idx=None, max_agents_to_plot=10):
        if agent_idx is None:
            agent_indices = range(np.clip(simulated_trajectories["x"].shape[0], 1, max_agents_to_plot))

        else:
            agent_indices = [agent_idx]

        for agent_idx in agent_indices:
            valid_mask = gt_trajectories["valid"][agent_idx, 0, :] == 1
            invalid_mask = ~valid_mask

            last_valid_idx = np.where(valid_mask)[0][-1] if valid_mask.any() else 0
            goal_x = gt_trajectories["x"][agent_idx, 0, last_valid_idx]
            goal_y = gt_trajectories["y"][agent_idx, 0, last_valid_idx]
            goal_radius = 2.0  # Note: Hardcoded here; ideally pass from config

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].set_title(f"Simulated rollouts (x, y) for agent id: {simulated_trajectories['id'][agent_idx, 0][0]}")

            for i in range(self.num_rollouts):
                # Sample random color for each rollout
                color = plt.cm.tab20(i % 20)
                axs[0].scatter(
                    simulated_trajectories["x"][agent_idx, i, :],
                    simulated_trajectories["y"][agent_idx, i, :],
                    alpha=0.1,
                    color=color,
                )

            axs[1].set_title(
                f"Simulated rollouts (x, y) and GT; agent id: {simulated_trajectories['id'][agent_idx, 0][0]}"
            )

            axs[1].scatter(
                simulated_trajectories["x"][agent_idx, :, valid_mask],
                simulated_trajectories["y"][agent_idx, :, valid_mask],
                color="b",
                alpha=0.1,
                zorder=4,
            )

            axs[1].scatter(
                gt_trajectories["x"][agent_idx, 0, valid_mask],
                gt_trajectories["y"][agent_idx, 0, valid_mask],
                color="g",
                label="Ground truth",
                alpha=0.5,
            )

            axs[1].scatter(
                gt_trajectories["x"][agent_idx, 0, 0],
                gt_trajectories["y"][agent_idx, 0, 0],
                color="darkgreen",
                marker="*",
                s=200,
                label="Log start",
                zorder=5,
                alpha=0.5,
            )
            axs[1].scatter(
                simulated_trajectories["x"][agent_idx, :, 0],
                simulated_trajectories["y"][agent_idx, :, 0],
                color="darkblue",
                marker="*",
                s=200,
                label="Agent start",
                zorder=5,
                alpha=0.5,
            )

            circle = plt.Circle(
                (goal_x, goal_y),
                goal_radius,
                color="g",
                fill=False,
                linewidth=2,
                linestyle="--",
                label=f"Goal radius ({goal_radius}m)",
                zorder=0,
            )
            axs[1].add_patch(circle)

            axs[1].set_xlabel("x")
            axs[1].set_ylabel("y")
            axs[1].legend()
            axs[1].set_aspect("equal", adjustable="datalim")

            axs[2].set_title(f"Heading timeseries for agent ID: {simulated_trajectories['id'][agent_idx, 0][0]}")
            time_steps = list(range(self.sim_steps))
            for r in range(self.num_rollouts):
                axs[2].plot(
                    time_steps,
                    simulated_trajectories["heading"][agent_idx, r, :],
                    color="b",
                    alpha=0.1,
                    label="Simulated" if r == 0 else "",
                )
            axs[2].plot(time_steps, gt_trajectories["heading"][agent_idx, 0, :], color="g", label="Ground truth")

            if invalid_mask.any():
                invalid_timesteps = np.where(invalid_mask)[0]
                axs[2].scatter(
                    invalid_timesteps,
                    gt_trajectories["heading"][agent_idx, 0, invalid_mask],
                    color="r",
                    marker="^",
                    s=100,
                    label="Invalid",
                    zorder=6,
                    edgecolors="darkred",
                    linewidths=1,
                )

            axs[2].set_xlabel("Time step")
            axs[2].legend()

            plt.tight_layout()

            plt.savefig(f"trajectory_comparison_agent_{agent_idx}.png")


class HumanReplayEvaluator:
    """Evaluates policies against human replays in PufferDrive."""

    def __init__(self, config: Dict):
        self.config = config
        self.sim_steps = 91 - self.config["env"]["init_steps"]

    def rollout(self, args, puffer_env, policy):
        """Roll out policy in env with human replays. Store statistics.

        In human replay mode, only the SDC (self-driving car) is controlled by the policy
        while all other agents replay their human trajectories. This tests how compatible
        the policy is with (static) human partners.

        Args:
            args: Config dict with train settings (device, use_rnn, etc.)
            puffer_env: PufferLib environment wrapper
            policy: Trained policy to evaluate

        Returns:
            dict: Aggregated metrics including:
                - avg_collisions_per_agent: Average collisions per agent
                - avg_offroad_per_agent: Average offroad events per agent
        """
        import numpy as np
        import torch
        import pufferlib

        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]

        obs, info = puffer_env.reset()
        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        for time_idx in range(self.sim_steps):
            # Step policy
            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, value = policy.forward_eval(ob_tensor, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(puffer_env.action_space.shape)

            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, puffer_env.action_space.low, puffer_env.action_space.high)

            obs, rewards, dones, truncs, info_list = puffer_env.step(action_np)

            if len(info_list) > 0:  # Happens at the end of episode
                results = info_list[0]
                return results
