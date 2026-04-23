"""WOSAC evaluation class for PufferDrive."""

import copy
import torch
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import configparser
import os
import pufferlib

# WOSAC eval
from pufferlib.ocean.benchmark import metrics
from pufferlib.ocean.benchmark import estimators
from pufferlib.ocean.benchmark import interaction_features


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
        self.filter_out_post_done = config.get("eval", {}).get("wosac_filter_out_post_done", True)
        self.device = config.get("train", {}).get("device", "cuda")
        self.eval_mode = config.get("eval", {}).get("wosac_eval_mode", "policy")

        wosac_metrics_path = os.path.join(os.path.dirname(__file__), "wosac.ini")
        self.metrics_config = configparser.ConfigParser()
        self.metrics_config.read(wosac_metrics_path)
        self.mode = "rl"

    def evaluate(self, args, vecenv, policy=None, drop_scene_duplicates=True):
        """Run full WOSAC evaluation with batched iteration over target scenarios.

        Args:
            args: Configuration dictionary
            vecenv: Vectorized environment
            policy: Policy to evaluate
            drop_scene_duplicates: Whether to drop duplicate scenarios

        Returns:
            DataFrame: Full results aggregated by scenario.
        """
        num_target_maps = args["eval"]["wosac_target_scenarios"]
        max_batches = args["eval"].get("wosac_max_batches", 100)

        unique_files_sampled = set()
        combined_results = []

        with tqdm(total=100, desc="Processing batches", unit="%", colour="cyan") as pbar:
            batch_idx = 0
            while batch_idx < max_batches:
                # Resample maps for each batch (except first)
                if batch_idx > 0:
                    vecenv.driver_env.resample_maps()

                # Obtain ground truth trajectories
                gt_trajectories = self.collect_ground_truth_trajectories(vecenv)

                # Collect simulated trajectories
                if policy is not None and self.eval_mode == "policy":
                    simulated_trajectories = self.collect_simulated_trajectories(args, vecenv, policy)
                elif self.eval_mode == "ground_truth":
                    # Create fake simulated trajectories by repeating ground truth
                    simulated_trajectories = gt_trajectories.copy()
                    for key in ["x", "y", "heading", "id"]:
                        simulated_trajectories[key] = np.repeat(
                            gt_trajectories[key], args["eval"]["wosac_num_rollouts"], axis=1
                        )
                    simulated_trajectories["id"] = simulated_trajectories["id"][..., np.newaxis]
                    simulated_trajectories["dones"] = np.zeros_like(simulated_trajectories["x"])
                else:
                    raise ValueError(f"Policy is None or unknown evaluation mode: {self.eval_mode}")

                # Compute metrics for this batch
                agent_state = vecenv.driver_env.get_global_agent_state()
                road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
                batch_results = self.compute_metrics(
                    gt_trajectories,
                    simulated_trajectories,
                    agent_state,
                    road_edge_polylines,
                    aggregate_results=False,
                )

                # Optional: sanity check on first batch
                if args["eval"].get("wosac_sanity_check", False) and batch_idx == 0:
                    self._quick_sanity_check(gt_trajectories, simulated_trajectories)

                # Track coverage
                unique_files_sampled.update(str(s) for s in np.unique(gt_trajectories["scenario_id"]))
                combined_results.append(batch_results)

                # Update progress
                coverage = len(unique_files_sampled) / num_target_maps
                pbar.n = int(coverage * 100)
                pbar.set_postfix({"n": len(unique_files_sampled), "batch": batch_idx + 1})
                pbar.refresh()

                batch_idx += 1

                # Stop if we've covered all target scenarios
                if len(unique_files_sampled) >= num_target_maps:
                    break

            # Check if we didn't reach target coverage
            if len(unique_files_sampled) < num_target_maps:
                print(
                    f"\nWarning: Only covered {len(unique_files_sampled)}/{num_target_maps} scenarios after {batch_idx} batches"
                )

            # Combine batch results into single dataframe
            df_combined = pd.concat(combined_results)

            # Optionally drop duplicate scenarios (keep first occurrence)
            if drop_scene_duplicates:
                initial_count = len(df_combined)
                df_combined = df_combined[~df_combined.index.duplicated(keep="first")]
                dropped = initial_count - len(df_combined)
                if dropped > 0:
                    print(f"\nDropped {dropped} duplicate scenarios.")

            print(f"\nCollected {len(df_combined)} agent records from {batch_idx} batches")

            return df_combined

    def _compute_metametric(self, metrics: pd.Series) -> float:
        metametric = 0.0
        for field_name in _METRIC_FIELD_NAMES:
            likelihood_field_name = "likelihood_" + field_name
            weight = self.metrics_config.getfloat(field_name, "metametric_weight")
            metric_score = metrics[likelihood_field_name]
            metametric += weight * metric_score
        return metametric

    def _get_histogram_params(self, metric_name: str):
        return (
            self.metrics_config.getfloat(metric_name, "histogram.min_val"),
            self.metrics_config.getfloat(metric_name, "histogram.max_val"),
            self.metrics_config.getint(metric_name, "histogram.num_bins"),
            self.metrics_config.getfloat(metric_name, "histogram.additive_smoothing_pseudocount"),
            self.metrics_config.getboolean(metric_name, "independent_timesteps"),
        )

    def _get_eval_mask(self, ground_truth_trajectories: Dict):
        agent_filter = self.config.get("eval", {}).get("wosac_eval_agent_filter", "tracks_to_predict")

        if agent_filter == "tracks_to_predict":
            return ground_truth_trajectories["is_track_to_predict"][:, 0].astype(bool)

        if agent_filter == "all":
            return ground_truth_trajectories["valid"].any(axis=2)[:, 0].astype(bool)

        if agent_filter == "sdc":
            if "is_sdc" not in ground_truth_trajectories:
                raise KeyError("wosac_eval_agent_filter=sdc requires is_sdc metadata")
            return ground_truth_trajectories["is_sdc"][:, 0].astype(bool)

        raise ValueError(
            f"Unknown wosac_eval_agent_filter={agent_filter!r}. "
            "Expected one of: tracks_to_predict, all, sdc."
        )

    def collect_ground_truth_trajectories(self, puffer_env):
        """Collect ground truth data for evaluation.
        Returns:
            trajectories: dict with keys 'x', 'y', 'z', 'heading', 'id'
                        each of shape (num_agents, 1, num_steps) for trajectory data
        """
        return puffer_env.get_ground_truth_trajectories()

    def collect_simulated_trajectories(self, args, puffer_env, policy=None, actions=None):
        """Roll out policy in env and collect trajectories.
        Args:
            args: configuration dictionary
            puffer_env: PufferDrive environment
            policy: policy to evaluate (if None, actions must be provided)
            actions: actions to step the agent (if policy is None). Currently only works with discrete actions and the
                classic dynamics model. Shape: [time, num_agents, 1]

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
            "dones": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.bool_),
        }

        for rollout_idx in tqdm(range(self.num_rollouts), desc="Collecting rollouts", colour="blue"):
            print(f"\rCollecting rollout {rollout_idx + 1}/{self.num_rollouts}...", end="", flush=True)

            obs, info = puffer_env.reset()
            truncations = np.zeros((num_agents,), dtype=bool)
            state = {}

            if args["train"]["use_rnn"] and policy is not None:
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
                trajectories["dones"][:, rollout_idx, time_idx] = truncations

                # Step policy
                if policy is None and actions is not None:
                    human_act_time_index = self.init_steps + time_idx
                    action_np = actions[human_act_time_index, :].copy()

                    # Replace invalid actions (-1) with "do nothing" action
                    # For discrete actions: use action 45 (accel=0, steer=0)
                    # For continuous actions: use [0.0, 0.0]
                    invalid_mask = action_np == -1.0

                    if puffer_env.action_space.__class__.__name__ == "MultiDiscrete":
                        # Discrete action space
                        action_np[invalid_mask] = 45  # Do nothing action

                elif policy is not None:
                    if self.mode == "bc_policy":
                        with torch.no_grad():
                            ob_tensor = torch.as_tensor(obs).to(device)
                            pred_action = policy(ob_tensor, deterministic=True)
                            action_np = pred_action.cpu().numpy().reshape(puffer_env.action_space.shape)
                    else:
                        with torch.no_grad():
                            ob_tensor = torch.as_tensor(obs).to(device)
                            logits, value = policy.forward_eval(ob_tensor, state)
                            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                            action_np = action.cpu().numpy().reshape(puffer_env.action_space.shape)

                        if isinstance(logits, torch.distributions.Normal):
                            action_np = np.clip(action_np, puffer_env.action_space.low, puffer_env.action_space.high)

                obs, rewards, terminals, truncations, infos = puffer_env.step(action_np)

        return trajectories

    def collect_wosac_random_baseline(self, puffer_env):
        """
        Random Baseline from Wosac 2023 paper
        """
        driver = puffer_env.driver_env
        num_agents = puffer_env.observation_space.shape[0]

        trajectories = {
            "x": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "id": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.int32),
            "dones": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.bool_),
        }

        for rollout_idx in range(self.num_rollouts):
            obs, info = puffer_env.reset()

            # Do Initialization
            agent_state = driver.get_global_agent_state()
            trajectories["x"][:, rollout_idx, 0] = agent_state["x"]
            trajectories["y"][:, rollout_idx, 0] = agent_state["y"]
            trajectories["heading"][:, rollout_idx, 0] = agent_state["heading"]
            trajectories["id"][:, rollout_idx, 0] = agent_state["id"]

            # Update using Gaussian:
            samples = np.random.normal(loc=1, scale=0.1, size=(num_agents, self.sim_steps, 3))
            for time_idx in range(1, self.sim_steps):
                dx, dy, d_heading = samples[:, time_idx, 0], samples[:, time_idx, 1], samples[:, time_idx, 2]
                x, y, heading = (
                    trajectories["x"][:, rollout_idx, time_idx - 1],
                    trajectories["y"][:, rollout_idx, time_idx - 1],
                    trajectories["heading"][:, rollout_idx, time_idx - 1],
                )

                cos_h = np.cos(heading)
                sin_h = np.sin(heading)

                x += dx * cos_h - dy * sin_h
                y += dx * sin_h + dy * cos_h
                heading += d_heading

                trajectories["x"][:, rollout_idx, time_idx] = x
                trajectories["y"][:, rollout_idx, time_idx] = y
                trajectories["heading"][:, rollout_idx, time_idx] = heading

        return trajectories

    def compute_metrics(
        self,
        ground_truth_trajectories: Dict,
        simulated_trajectories: Dict,
        agent_state: Dict,
        road_edge_polylines: Dict,
        aggregate_results: bool = False,
        drop_last_scenario: bool = True,
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
        simulated_ids = np.asarray(simulated_trajectories["id"])
        if simulated_ids.ndim == 1:
            simulated_ids = simulated_ids[:, None]
        elif simulated_ids.ndim == 2:
            simulated_ids = simulated_ids[:, 0:1]
        else:
            simulated_ids = simulated_ids[:, 0:1, 0]

        assert np.array_equal(simulated_ids, ground_truth_trajectories["id"]), (
            "Agent IDs don't match between simulated and ground truth trajectories"
        )

        eval_mask = self._get_eval_mask(ground_truth_trajectories)

        # Extract trajectories
        sim_x = simulated_trajectories["x"]
        sim_y = simulated_trajectories["y"]
        sim_heading = simulated_trajectories["heading"]
        sim_dones = simulated_trajectories["dones"]
        ref_x = ground_truth_trajectories["x"]
        ref_y = ground_truth_trajectories["y"]
        ref_heading = ground_truth_trajectories["heading"]
        ref_valid = ground_truth_trajectories["valid"]
        agent_length = agent_state["length"]
        agent_width = agent_state["width"]
        is_vehicle = ground_truth_trajectories["is_vehicle"]
        scenario_ids = ground_truth_trajectories["scenario_id"]

        last_scenario_id = str(scenario_ids[-1][0])

        # We evaluate the metrics only for the Tracks to Predict.
        eval_sim_x = sim_x[eval_mask]
        eval_sim_y = sim_y[eval_mask]
        eval_sim_heading = sim_heading[eval_mask]
        eval_ref_x = ref_x[eval_mask]
        eval_ref_y = ref_y[eval_mask]
        eval_dones = sim_dones[eval_mask]
        eval_ref_heading = ref_heading[eval_mask]
        eval_ref_valid = ref_valid[eval_mask]
        eval_agent_length = agent_length[eval_mask]
        eval_agent_width = agent_width[eval_mask]
        eval_scenario_ids = scenario_ids[eval_mask]
        eval_is_vehicle = is_vehicle[eval_mask]

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

        speed_log_likelihood = metrics._reduce_average_with_validity(
            linear_speed_log_likelihood,
            speed_validity[:, 0, :],
            axis=1,
        )

        accel_log_likelihood = metrics._reduce_average_with_validity(
            linear_accel_log_likelihood,
            acceleration_validity[:, 0, :],
            axis=1,
        )

        angular_speed_log_likelihood = metrics._reduce_average_with_validity(
            angular_speed_log_likelihood,
            speed_validity[:, 0, :],
            axis=1,
        )

        angular_accel_log_likelihood = metrics._reduce_average_with_validity(
            angular_accel_log_likelihood,
            acceleration_validity[:, 0, :],
            axis=1,
        )

        distance_to_nearest_object_log_likelihood = metrics._reduce_average_with_validity(
            distance_to_nearest_object_log_likelihood,
            eval_ref_valid[:, 0, :],
            axis=1,
        )

        # TTC is computed only for vehicles
        ttc_valid = eval_ref_valid & eval_is_vehicle[..., None]
        time_to_collision_log_likelihood = metrics._reduce_average_with_validity(
            time_to_collision_log_likelihood,
            ttc_valid[:, 0, :],
            axis=1,
        )

        distance_to_road_edge_log_likelihood = metrics._reduce_average_with_validity(
            distance_to_road_edge_log_likelihood,
            eval_ref_valid[:, 0, :],
            axis=1,
        )

        # Collision likelihood is computed by aggregating in time. For invalid objects
        # in the logged scenario, we need to filter possible collisions in simulation.
        # `sim_collision_indication` shape: (n_samples, n_objects).

        # Combine validity masks: only count events when ref is valid and agent is not done
        if self.filter_out_post_done:
            active_mask = eval_ref_valid & ~eval_dones  # (n_agents, n_rollouts, n_steps)
        else:
            active_mask = eval_ref_valid  # (n_agents, 1, n_steps)

        # Diagnostic: show average number of active timesteps per rollout
        active_timesteps_per_rollout = np.sum(active_mask, axis=2)  # (n_agents, n_rollouts)
        avg_active_timesteps = np.mean(active_timesteps_per_rollout)

        sim_collision_indication = np.any(np.where(active_mask, sim_collision_per_step, False), axis=2)
        ref_collision_indication = np.any(np.where(active_mask, ref_collision_per_step, False), axis=2)

        sim_num_collisions = np.mean(sim_collision_indication, axis=1)
        ref_num_collisions = np.mean(ref_collision_indication, axis=1)

        collision_log_likelihood = estimators.log_likelihood_estimate_scenario_level(
            log_values=ref_collision_indication[:, 0],
            sim_values=sim_collision_indication,
            min_val=0.0,
            max_val=1.0,
            num_bins=2,
            use_bernoulli=True,
        )

        # Offroad likelihood (same pattern as collision)
        sim_offroad_indication = np.any(np.where(active_mask, sim_offroad_per_step, False), axis=2)
        ref_offroad_indication = np.any(np.where(active_mask, ref_offroad_per_step, False), axis=2)

        sim_num_offroad = np.mean(sim_offroad_indication, axis=1)
        ref_num_offroad = np.mean(ref_offroad_indication, axis=1)

        offroad_log_likelihood = estimators.log_likelihood_estimate_scenario_level(
            log_values=ref_offroad_indication[:, 0],
            sim_values=sim_offroad_indication,
            min_val=0.0,
            max_val=1.0,
            num_bins=2,
            use_bernoulli=True,
        )

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
                "likelihood_linear_speed": speed_log_likelihood,
                "likelihood_linear_acceleration": accel_log_likelihood,
                "likelihood_angular_speed": angular_speed_log_likelihood,
                "likelihood_angular_acceleration": angular_accel_log_likelihood,
                "likelihood_distance_to_nearest_object": distance_to_nearest_object_log_likelihood,
                "likelihood_time_to_collision": time_to_collision_log_likelihood,
                "likelihood_collision_indication": collision_log_likelihood,
                "likelihood_distance_to_road_edge": distance_to_road_edge_log_likelihood,
                "likelihood_offroad_indication": offroad_log_likelihood,
            }
        )

        # Aggregate along agent dimenision: Obtain one score per scenario
        df_scene_level = df.groupby("scenario_id", as_index=True).mean().drop(columns=["agent_id"]).dropna()

        # Exponentiate the averaged log-likelihoods to get final likelihoods
        likelihood_columns = [col for col in df_scene_level.columns if col.startswith("likelihood_")]
        df_scene_level[likelihood_columns] = np.exp(df_scene_level[likelihood_columns])

        df_scene_level["realism_meta_score"] = df_scene_level.apply(self._compute_metametric, axis=1)
        df_scene_level["num_agents_per_scene"] = df.groupby("scenario_id").size()
        df_scene_level = df_scene_level.round(3)

        # Get group summary metrics
        kinematic_metrics = np.mean(
            [
                df_scene_level["likelihood_linear_speed"],
                df_scene_level["likelihood_linear_acceleration"],
                df_scene_level["likelihood_angular_speed"],
                df_scene_level["likelihood_angular_acceleration"],
            ]
        )

        interactive_metrics = np.mean(
            [
                df_scene_level["likelihood_collision_indication"],
                df_scene_level["likelihood_distance_to_nearest_object"],
                df_scene_level["likelihood_time_to_collision"],
            ]
        )

        map_metrics = np.mean(
            [
                df_scene_level["likelihood_distance_to_road_edge"],
                df_scene_level["likelihood_offroad_indication"],
            ]
        )

        df_scene_level["kinematic_metrics"] = kinematic_metrics
        df_scene_level["interactive_metrics"] = interactive_metrics
        df_scene_level["map_based_metrics"] = map_metrics

        # Safety: drop the last scenario (potentially incomplete) from the scene-level results
        if drop_last_scenario and last_scenario_id in df_scene_level.index:
            df_scene_level = df_scene_level.drop(last_scenario_id)

        if aggregate_results:
            # Aggregate over scenarios
            aggregate_metrics = df_scene_level.mean().to_dict()
            aggregate_metrics["total_num_agents"] = df_scene_level["num_agents_per_scene"].sum()
            aggregate_metrics["realism_score_std"] = df_scene_level["realism_meta_score"].std()
            return aggregate_metrics
        else:
            return df_scene_level

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


class PlanningEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.sim_steps = 91 - self.config["env"]["init_steps"]
        self.device = config.get("train", {}).get("device", "cuda")

    @staticmethod
    def _normalize_scenario_ids(scenario_ids):
        scenario_ids = np.asarray(scenario_ids)
        if scenario_ids.ndim == 1:
            return scenario_ids[:, None]
        if scenario_ids.ndim == 2:
            return scenario_ids
        return scenario_ids[:, :, 0]

    def _get_eval_mask(self, combined_trajectories):
        agent_filter = self.config.get("eval", {}).get("planning_eval_agent_filter", "sdc")

        if agent_filter == "sdc":
            if "is_sdc" in combined_trajectories:
                return combined_trajectories["is_sdc"][:, 0].astype(bool)
            return combined_trajectories["id"][:, 0] <= -2

        if agent_filter == "all":
            return combined_trajectories["valid"].any(axis=2)[:, 0].astype(bool)

        if agent_filter == "tracks_to_predict":
            if "is_track_to_predict" not in combined_trajectories:
                raise KeyError("planning_eval_agent_filter=tracks_to_predict requires is_track_to_predict metadata")
            return combined_trajectories["is_track_to_predict"][:, 0].astype(bool)

        raise ValueError(
            f"Unknown planning_eval_agent_filter={agent_filter!r}. "
            "Expected one of: sdc, all, tracks_to_predict."
        )

    @staticmethod
    def _compute_route_progress(sim_x, sim_y, sim_valid, gt_x, gt_y, gt_valid, reached_goal=None, chunk_size=1024):
        """Compute normalized progress along the ground-truth route.

        For each simulated position, find the closest valid point on the GT route
        and report the maximum normalized GT arclength reached during the rollout.
        """
        sim_valid = sim_valid.astype(bool)
        gt_valid = gt_valid.astype(bool)
        num_agents, num_rollouts, _ = sim_x.shape
        route_progress = np.zeros((num_agents, num_rollouts), dtype=np.float32)

        for start in range(0, num_agents, chunk_size):
            end = min(start + chunk_size, num_agents)
            ref_x = gt_x[start:end, 0, :]
            ref_y = gt_y[start:end, 0, :]
            ref_valid = gt_valid[start:end, 0, :]
            cur_sim_x = sim_x[start:end]
            cur_sim_y = sim_y[start:end]
            cur_sim_valid = sim_valid[start:end]

            segment_valid = ref_valid[:, 1:] & ref_valid[:, :-1]
            segment_lengths = np.sqrt(np.diff(ref_x, axis=1) ** 2 + np.diff(ref_y, axis=1) ** 2)
            cumulative = np.concatenate(
                [np.zeros((end - start, 1), dtype=np.float32), np.cumsum(segment_lengths * segment_valid, axis=1)],
                axis=1,
            )
            route_length = cumulative[:, -1]
            valid_route = route_length > 1e-3
            progress_at_ref = np.divide(
                cumulative,
                route_length[:, None],
                out=np.zeros_like(cumulative, dtype=np.float32),
                where=valid_route[:, None],
            )

            dist_sq = (cur_sim_x[..., None] - ref_x[:, None, None, :]) ** 2 + (
                cur_sim_y[..., None] - ref_y[:, None, None, :]
            ) ** 2
            dist_sq = np.where(ref_valid[:, None, None, :], dist_sq, np.inf)
            closest_ref_idx = np.argmin(dist_sq, axis=3)
            progress_samples = np.take_along_axis(
                progress_at_ref[:, None, None, :],
                closest_ref_idx[..., None],
                axis=3,
            )[..., 0]
            progress_samples = np.where(cur_sim_valid, progress_samples, 0.0)
            route_progress[start:end] = np.clip(np.max(progress_samples, axis=2), 0.0, 1.0)

        if reached_goal is not None:
            route_progress = np.where(reached_goal.astype(bool), 1.0, route_progress)
        return route_progress

    @staticmethod
    def _compute_velocities(x, y, valid, seconds_per_step=0.1):
        vx = np.zeros_like(x, dtype=np.float32)
        vy = np.zeros_like(y, dtype=np.float32)
        if x.shape[-1] <= 1:
            return vx, vy

        valid = valid.astype(bool)
        step_valid = valid[..., 1:] & valid[..., :-1]
        vx[..., 1:] = np.where(step_valid, (x[..., 1:] - x[..., :-1]) / seconds_per_step, 0.0)
        vy[..., 1:] = np.where(step_valid, (y[..., 1:] - y[..., :-1]) / seconds_per_step, 0.0)
        vx[..., 0] = vx[..., 1]
        vy[..., 0] = vy[..., 1]
        return vx, vy

    @classmethod
    def _compute_collision_fault_rates(
        cls,
        x,
        y,
        heading,
        valid,
        scenario_ids,
        agent_length,
        agent_width,
        eval_mask,
        device,
        seconds_per_step=0.1,
    ):
        """Classify collision responsibility for evaluated agents.

        Mirrors the simulator heuristic: an evaluated agent is at fault when it
        collides with an object in front while moving toward it. A rear collision
        is the converse case where the other object is moving toward the
        evaluated agent from behind.
        """
        eval_indices = np.where(eval_mask)[0]
        num_eval_agents = len(eval_indices)
        num_rollouts = x.shape[1]
        at_fault_collision = np.zeros((num_eval_agents, num_rollouts), dtype=np.float32)
        rear_collision = np.zeros((num_eval_agents, num_rollouts), dtype=np.float32)
        if num_eval_agents == 0:
            return at_fault_collision, rear_collision

        eval_to_result = {agent_idx: result_idx for result_idx, agent_idx in enumerate(eval_indices)}
        valid = valid.astype(bool)
        scenario_ids = np.asarray(scenario_ids)[:, 0]
        agent_length = np.asarray(agent_length).reshape(-1)
        agent_width = np.asarray(agent_width).reshape(-1)
        vx, vy = cls._compute_velocities(x, y, valid, seconds_per_step=seconds_per_step)

        for scenario_id in np.unique(scenario_ids):
            scenario_mask_np = scenario_ids == scenario_id
            scenario_eval_mask_np = eval_mask[scenario_mask_np]
            if not np.any(scenario_eval_mask_np):
                continue

            agent_indices = np.where(scenario_mask_np)[0]
            scenario_x = x[scenario_mask_np]
            scenario_y = y[scenario_mask_np]
            scenario_heading = heading[scenario_mask_np]
            scenario_valid = valid[scenario_mask_np]
            scenario_vx = vx[scenario_mask_np]
            scenario_vy = vy[scenario_mask_np]
            num_agents = scenario_x.shape[0]

            length = np.broadcast_to(agent_length[agent_indices, None], (num_agents, num_rollouts)).copy()
            width = np.broadcast_to(agent_width[agent_indices, None], (num_agents, num_rollouts)).copy()

            signed_distances = interaction_features.compute_signed_distances(
                center_x=torch.as_tensor(scenario_x, dtype=torch.float32, device=device),
                center_y=torch.as_tensor(scenario_y, dtype=torch.float32, device=device),
                length=torch.as_tensor(length, dtype=torch.float32, device=device),
                width=torch.as_tensor(width, dtype=torch.float32, device=device),
                heading=torch.as_tensor(scenario_heading, dtype=torch.float32, device=device),
                valid=torch.as_tensor(scenario_valid, dtype=torch.bool, device=device),
                evaluated_object_mask=torch.as_tensor(scenario_eval_mask_np, dtype=torch.bool, device=device),
            )
            colliding = (signed_distances < interaction_features.COLLISION_DISTANCE_THRESHOLD).cpu().numpy()

            local_eval_indices = np.where(scenario_eval_mask_np)[0]
            for eval_rank, local_eval_idx in enumerate(local_eval_indices):
                global_eval_idx = agent_indices[local_eval_idx]
                result_idx = eval_to_result[global_eval_idx]

                for other_idx in range(num_agents):
                    if other_idx == local_eval_idx:
                        continue

                    pair_collision = colliding[eval_rank, other_idx]
                    if not np.any(pair_collision):
                        continue

                    dx = scenario_x[other_idx] - scenario_x[local_eval_idx]
                    dy = scenario_y[other_idx] - scenario_y[local_eval_idx]
                    eval_forward_dot = dx * np.cos(scenario_heading[local_eval_idx]) + dy * np.sin(
                        scenario_heading[local_eval_idx]
                    )
                    eval_approach_dot = scenario_vx[local_eval_idx] * dx + scenario_vy[local_eval_idx] * dy
                    at_fault = pair_collision & (eval_forward_dot > 0.0) & (eval_approach_dot > 0.0)

                    dx_reverse = -dx
                    dy_reverse = -dy
                    other_forward_dot = dx_reverse * np.cos(scenario_heading[other_idx]) + dy_reverse * np.sin(
                        scenario_heading[other_idx]
                    )
                    other_approach_dot = scenario_vx[other_idx] * dx_reverse + scenario_vy[other_idx] * dy_reverse
                    rear = pair_collision & (other_forward_dot > 0.0) & (other_approach_dot > 0.0)

                    at_fault_collision[result_idx] = np.maximum(
                        at_fault_collision[result_idx], np.any(at_fault, axis=1).astype(np.float32)
                    )
                    rear_collision[result_idx] = np.maximum(
                        rear_collision[result_idx], np.any(rear, axis=1).astype(np.float32)
                    )

        return at_fault_collision, rear_collision

    def compute_metrics(
        self,
        combined_trajectories,
        agent_state,
        road_edge_polylines,
        aggregate_results: bool = False,
        ground_truth_trajectories=None,
        goal_radius: float | None = None,
        goal_speed: float | None = None,
    ) -> Dict:
        eval_mask = self._get_eval_mask(combined_trajectories)
        x = combined_trajectories["x"]
        y = combined_trajectories["y"]
        heading = combined_trajectories["heading"]
        valid = combined_trajectories["valid"]
        agent_length = agent_state["length"]
        agent_width = agent_state["width"]
        scenario_ids = self._normalize_scenario_ids(combined_trajectories["scenario_id"])

        eval_x = x[eval_mask]
        eval_y = y[eval_mask]
        eval_heading = heading[eval_mask]
        eval_valid = valid[eval_mask]
        eval_agent_length = agent_length[eval_mask]
        eval_agent_width = agent_width[eval_mask]
        eval_scenario_ids = scenario_ids[eval_mask]

        _, collisions_per_step, _ = metrics.compute_interaction_features(
            x, y, heading, scenario_ids, agent_length, agent_width, eval_mask, device=self.device
        )
        at_fault_collision_rate, rear_collision_rate = self._compute_collision_fault_rates(
            x,
            y,
            heading,
            valid,
            scenario_ids,
            agent_length,
            agent_width,
            eval_mask,
            device=self.device,
        )

        _, offroad_per_step = metrics.compute_map_features(
            eval_x,
            eval_y,
            eval_heading,
            eval_scenario_ids,
            eval_agent_length,
            eval_agent_width,
            road_edge_polylines,
            device=self.device,
            agent_chunk_size=self.config.get("eval", {}).get("planning_map_agent_chunk_size"),
        )

        collision_indication = np.any(np.where(eval_valid, collisions_per_step, False), axis=2).astype(float)
        offroad_indication = np.any(np.where(eval_valid, offroad_per_step, False), axis=2).astype(float)
        accuracy = 1.0 - (collision_indication + offroad_indication) + (collision_indication * offroad_indication)

        scene_level_results = {
            "collision_indication": collision_indication.flatten(),
            "at_fault_collision_rate": at_fault_collision_rate.flatten(),
            "rear_collision_rate": rear_collision_rate.flatten(),
            "offroad_indication": offroad_indication.flatten(),
            "accuracy": accuracy.flatten(),
        }

        if ground_truth_trajectories is not None and goal_radius is not None:
            gt_x = ground_truth_trajectories["x"][eval_mask]
            gt_y = ground_truth_trajectories["y"][eval_mask]
            gt_valid = ground_truth_trajectories["valid"][eval_mask]

            final_valid_idx = np.clip(gt_valid.sum(axis=2).astype(int) - 1, 0, gt_valid.shape[2] - 1)
            goal_x = np.take_along_axis(gt_x[:, 0, :], final_valid_idx[:, 0, None], axis=1)[:, 0]
            goal_y = np.take_along_axis(gt_y[:, 0, :], final_valid_idx[:, 0, None], axis=1)[:, 0]

            goal_distance = np.sqrt((eval_x - goal_x[:, None, None]) ** 2 + (eval_y - goal_y[:, None, None]) ** 2)
            linear_speed, _, _, _ = metrics.compute_kinematic_features(eval_x, eval_y, eval_heading)
            speed_threshold = np.inf if goal_speed is None else goal_speed
            reached_goal = np.any(
                np.where(eval_valid, goal_distance <= goal_radius, False) &
                np.where(np.isnan(linear_speed), False, linear_speed <= speed_threshold),
                axis=2,
            ).astype(float)
            route_progress = self._compute_route_progress(
                eval_x,
                eval_y,
                eval_valid,
                gt_x,
                gt_y,
                gt_valid,
                reached_goal=reached_goal,
            )

            scene_level_results["goal_reached"] = reached_goal.flatten()
            scene_level_results["route_progress"] = route_progress.flatten()
            scene_level_results["score"] = (accuracy * reached_goal).flatten()

        scene_level_results = pd.DataFrame(scene_level_results, index=eval_scenario_ids[:, 0])

        if aggregate_results:
            aggregate_metrics = scene_level_results.mean().to_dict()
            aggregate_metrics["num_scenarios"] = scene_level_results.shape[0]
            return {k: v.item() if hasattr(v, "item") else v for k, v in aggregate_metrics.items()}

        print("\n Scene-level results:\n")
        print(scene_level_results)
        return scene_level_results


class Evaluator:
    """Evaluates policies in self_play or human_replay mode, with optional rendering.

    Initializes the eval envs needed based on eval config flags:
    - human_replay_eval: creates sp_env + hr_env
    - render_eval: creates sp_env (if not already created)
    """

    RENDER_FIRST = "first"
    RENDER_RANDOM = "random"
    RENDER_WORST_SCORE = "worst_score"
    RENDER_WORST_COLLISION = "worst_collision"

    def __init__(self, configs, logger=None):
        self.configs = configs
        self.logger = logger
        self.sim_steps = 90
        self.self_play_stats = None
        self.human_replay_stats = None
        self.sp_env = None
        self.hr_env = None
        self.render_env_idx = 0  # Which of the vecenvs to use for rendering
        self.inference_lambda_values = [0.0, 0.01, 0.05, 0.2]
        self.lambda_sweep_results = {}  # {lambda_val: collision_rate}

        self._unpack_eval_configs(configs)

    def _unpack_eval_configs(self, configs):
        eval_config = copy.deepcopy(configs)
        # Create separate evaluation environments based on specified configs
        eval_config["env"]["termination_mode"] = 1  # Important to ensure correct statistics
        backend = eval_config["eval"].get("backend", "PufferEnv")
        eval_config["env"]["map_dir"] = eval_config["eval"]["map_dir"]
        eval_config["env"]["num_maps"] = 10_000  # Validation set
        eval_config["env"]["num_agents"] = eval_config["eval"]["num_eval_agents"]
        eval_config["env"]["episode_length"] = 91  # WOMD scenario length
        eval_config["vec"] = dict(backend=backend, num_envs=1)
        eval_config["env"]["fix_lambdas"] = True
        eval_config["env"]["fix_rewards"] = True  # Fix to the ini file ones for all agents
        eval_config["env"]["lambda_value"] = configs["env"]["lambda_value"]
        eval_config["env"]["obs_partner_noise_speed"] = 0.0
        eval_config["env"]["obs_partner_noise_pos"] = 0.0

        self.hr_eval_config = copy.deepcopy(eval_config)
        self.hr_eval_config["env"]["control_mode"] = "control_sdc_only"
        self.hr_eval_config["env"]["goal_behavior"] = 0  # Remove and terminate at goal
        self.sp_eval_config = copy.deepcopy(eval_config)
        self.sp_eval_config["env"]["control_mode"] = "control_agents"
        self.render_select_mode = self.configs["eval"]["render_select_mode"]
        self.render_sp_rollout = self.configs["eval"]["render_self_play_eval"]
        self.render_hr_rollout = self.configs["eval"]["render_human_replay_eval"]

    def select_render_env(self, env_logs):
        """Select which environment to render based on per-env rollout statistics.
        Args:
            env_logs: List of dicts, one per environment. Each dict contains
                aggregated agent statistics (score, collision_rate, offroad_rate, etc.)
                with 'n' being the number of controlled agents in that env.
                Empty dicts indicate no data was collected for that env.

        Returns:
            int: Index of the environment to render.
        """
        mode = self.render_select_mode
        if mode == self.RENDER_FIRST:
            return 0
        if mode == self.RENDER_RANDOM:
            return np.random.randint(len(env_logs))

        populated = [(i, log) for i, log in enumerate(env_logs) if log]

        if not populated:
            return 0

        if mode == self.RENDER_WORST_SCORE:
            return min(populated, key=lambda x: x[1].get("score", 1.0))[0]
        elif mode == self.RENDER_WORST_COLLISION:
            return max(populated, key=lambda x: x[1].get("collision_rate", 0.0))[0]
        # Add other modes based on desiderata here
        return 0

    def rollout(self, policy, mode="self_play", view_mode=None):
        from pufferlib.ocean.drive.drive import RenderView

        if view_mode is None:
            view_mode = RenderView.FULL_SIM_STATE

        env = self.hr_env if mode == "human_replay" else self.sp_env
        render_eval = self.render_sp_rollout if mode == "self_play" else self.render_hr_rollout
        driver = env.driver_env

        needs_stats_first = render_eval and self.render_select_mode not in (self.RENDER_FIRST, self.RENDER_RANDOM)

        if needs_stats_first:
            env_logs = self._run_rollout(policy, env, per_env_logs=True)
            render_env_idx = self.select_render_env(env_logs)
        else:
            render_env_idx = self.select_render_env([{}] * driver.num_envs)

        env_statistics = self._run_rollout(
            policy,
            env,
            mode,
            render_env_idx if render_eval else None,
            per_env_logs=True,
            view_mode=view_mode,
        )

        if mode == "self_play":
            self.self_play_stats = env_statistics
            self.self_play_stats[0]["render_env_idx"] = render_env_idx
        elif mode == "human_replay":
            self.human_replay_stats = env_statistics
            self.human_replay_stats[0]["render_env_idx"] = render_env_idx

    def _run_rollout(self, policy, env, mode, render_env_idx=None, per_env_logs=False, view_mode=None):
        """Run a single rollout. If render_env_idx is not None, render that env."""
        from pufferlib.ocean.drive.drive import RenderView

        if view_mode is None:
            view_mode = RenderView.FULL_SIM_STATE

        driver = env.driver_env
        num_agents = env.observation_space.shape[0]
        device = self.configs["train"]["device"]

        # Reset environment
        obs, info = env.reset()
        terminals = np.zeros((num_agents, 1), dtype=bool)

        # Initialize RNN state if needed
        state = {}
        if self.configs["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        info_list = []
        for time_idx in range(self.sim_steps):
            if mode == "human_replay" and not terminals[render_env_idx]:
                driver.render(view_mode=view_mode, env_idx=render_env_idx)
            elif mode == "self_play":
                driver.render(view_mode=view_mode, env_idx=render_env_idx)

            # Get action from policy
            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, value = policy.forward_eval(ob_tensor, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(env.action_space.shape)

            # Clip continuous actions to valid range
            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            # Step environment
            obs, rewards, terminals, truncated, info_list = env.step(action_np, per_env_logs=per_env_logs)

            if truncated.all():
                break

        return info_list

    def run_lambda_sweep(self, policy, load_env_fn_from_config):
        """Run human replay rollouts across inference lambda values and collect stats.

        Args:
            policy: The policy to evaluate.
            load_env_fn_from_config: Callable(config) that creates a new hr_env.
        """
        self.lambda_sweep_results = {}
        for lam in self.inference_lambda_values:
            config = copy.deepcopy(self.hr_eval_config)
            config["env"]["fix_lambdas"] = True
            config["env"]["lambda_value"] = lam

            self.hr_env = load_env_fn_from_config(config)
            self.rollout(policy, mode="human_replay")
            self.hr_env.close()

            if self.human_replay_stats is not None:
                self.lambda_sweep_results[lam] = {
                    "collision_rate": self.human_replay_stats.get("collision_rate", 0.0),
                    "score": self.human_replay_stats.get("score", 0.0),
                }
            else:
                self.lambda_sweep_results[lam] = {"collision_rate": 0.0, "score": 0.0}

    def log_lambda_sweep(self, epoch):
        """Log the lambda sweep results as scalar metrics and a seaborn swarmplot to wandb."""
        if not (self.logger and hasattr(self.logger, "wandb") and self.logger.wandb):
            return
        if not self.lambda_sweep_results:
            return
        import wandb

        df = pd.DataFrame(self.lambda_sweep_results).T
        df.index.name = "lambda"

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
        ax.set_title(f"Effect of λ (regularization strength) \n Epoch {epoch}")
        sns.lineplot(data=df, x="lambda", y="collision_rate", marker="s", color="tab:blue", ax=ax)
        ax.set_ylabel("Human-replay collision rate", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_xlabel(r"$λ$")
        ax.set_ylim(0, 0.5)

        ax2 = ax.twinx()
        sns.lineplot(data=df, x="lambda", y="score", marker="o", color="tab:orange", ax=ax2)
        ax2.set_ylabel("Score", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(0, 1)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.logger.wandb.log({"eval/lambda_effect": wandb.Image(fig)})
        plt.close(fig)

    def run_collision_reward_sweep(self, policy, load_env_fn_from_config):
        """Run human replay rollouts across collision reward values and collect stats.

        Args:
            policy: The policy to evaluate.
            load_env_fn_from_config: Callable(config) that creates a new hr_env.
        """
        self.collision_reward_sweep_results = {}
        self.inference_collision_reward_values = [0.1, 0.0, -0.1, -0.3]

        for reward_val in self.inference_collision_reward_values:
            config = copy.deepcopy(self.hr_eval_config)
            config["env"]["fix_rewards"] = True
            config["env"]["reward_vehicle_collision"] = reward_val

            self.hr_env = load_env_fn_from_config(config)
            self.rollout(policy, mode="human_replay")
            self.hr_env.close()

            if self.human_replay_stats is not None:
                self.collision_reward_sweep_results[reward_val] = {
                    "collision_rate": self.human_replay_stats["collision_rate"],
                    "score": self.human_replay_stats.get("score", 0.0),
                }
            else:
                self.collision_reward_sweep_results[reward_val] = {"collision_rate": 0.0, "score": 0.0}

    def log_collision_reward_sweep(self, epoch):
        """Log the collision reward sweep results as a plot to wandb."""
        if not (self.logger and hasattr(self.logger, "wandb") and self.logger.wandb):
            return
        if not self.collision_reward_sweep_results:
            return
        import wandb

        df = pd.DataFrame(self.collision_reward_sweep_results).T
        df.index.name = "collision_reward"

        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
        ax.set_title(f"Effect of collision penalty conditioning\nEpoch {epoch}")
        sns.lineplot(data=df, x="collision_reward", y="collision_rate", marker="s", color="tab:purple", ax=ax)
        ax.set_ylabel("Human-replay collision rate", color="tab:purple")
        ax.tick_params(axis="y")
        ax.set_xlabel("Collision penalty (conditioning value)")
        ax.set_ylim(0, 1.0)

        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.logger.wandb.log({"eval/collision_reward_effect": wandb.Image(fig)})
        plt.close(fig)

    def log_videos(self, eval_mode, epoch):
        """Log all mp4s in local path to wandb after env close has flushed ffmpeg pipes."""
        import os
        import glob

        if not (self.logger and hasattr(self.logger, "wandb") and self.logger.wandb):
            # Still clean up even if not logging
            for p in glob.glob("*.mp4"):
                os.remove(p)
            return

        import wandb

        video_files = glob.glob("*.mp4")
        if not video_files:
            print("Warning: no render videos found in local path")
            return

        render_mode = self.render_select_mode
        for p in video_files:
            scenario_id = os.path.splitext(os.path.basename(p))[0]
            caption = f"scene_{scenario_id}_epoch_{epoch}_select_{render_mode}"
            self.logger.wandb.log({f"render/{eval_mode}": wandb.Video(p, format="mp4", caption=caption)})

        # Clean up
        for p in video_files:
            os.remove(p)

    def collect_stats(self):
        stats = {}

        if self.human_replay_stats is not None:
            populated = [log for log in self.human_replay_stats if log and log.get("n", 0) > 0]
            if populated:
                collisions_per_agent = np.array([log["collisions_per_agent"] for log in populated])
                did_collide = np.array([log["collision_rate"] for log in populated])
                stats["eval/hr_mean_collisions_per_agent"] = float(np.mean(collisions_per_agent))
                stats["eval/hr_mean_did_collide"] = float(np.mean(did_collide))
                stats["eval/hr_score"] = float(np.mean([log["score"] for log in populated]))

        if self.self_play_stats is not None:
            populated = [log for log in self.self_play_stats if log and log.get("n", 0) > 0]
            if populated:
                collisions_per_agent = np.array([log["collisions_per_agent"] for log in populated])
                did_collide = np.array([log["collision_rate"] for log in populated])
                stats["eval/sp_mean_collisions_per_agent"] = float(np.mean(collisions_per_agent))
                stats["eval/sp_mean_did_collide"] = float(np.mean(did_collide))
                stats["eval/sp_score"] = float(np.mean([log["score"] for log in populated]))
                stats["eval/sp_num_agents"] = float(np.mean([log["n"] for log in populated]))

        return stats
