"""WOSAC evaluation class for PufferDrive."""

import copy
import torch
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        self.eval_mode = config.get("eval", {}).get("wosac_eval_mode", "policy")

        wosac_metrics_path = os.path.join(os.path.dirname(__file__), "wosac.ini")
        self.metrics_config = configparser.ConfigParser()
        self.metrics_config.read(wosac_metrics_path)

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

        eval_mask = ground_truth_trajectories["is_track_to_predict"][:, 0]

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
        is_vehicle = ground_truth_trajectories["is_vehicle"]
        scenario_ids = ground_truth_trajectories["scenario_id"]

        last_scenario_id = str(scenario_ids[-1][0])

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

        sim_collision_indication = np.any(np.where(eval_ref_valid, sim_collision_per_step, False), axis=2)
        ref_collision_indication = np.any(np.where(eval_ref_valid, ref_collision_per_step, False), axis=2)

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
        sim_offroad_indication = np.any(np.where(eval_ref_valid, sim_offroad_per_step, False), axis=2)
        ref_offroad_indication = np.any(np.where(eval_ref_valid, ref_offroad_per_step, False), axis=2)

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
        if last_scenario_id in df_scene_level.index:
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


class Evaluator:
    """Evaluates policies in self_play, human_replay, with optional rendering.

    Initializes the eval envs needed based on eval config flags:
    - human_replay_eval: creates hr_env (control_sdc_only)
    - self_play_eval: creates sp_env (control_agents)
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

        self._unpack_eval_configs(configs)

    def _unpack_eval_configs(self, configs):
        from pufferlib.ocean.drive.drive import RenderView

        _VIEW_MODE_MAP = {
            "sim_state": [RenderView.FULL_SIM_STATE],
            "topdown": [RenderView.FULL_SIM_STATE],
            "bev": [RenderView.BEV_AGENT_OBS],
            "agent": [RenderView.BEV_AGENT_OBS],
            "persp": [RenderView.AGENT_PERSP],
            "both": [RenderView.FULL_SIM_STATE, RenderView.AGENT_PERSP],
            "all": [RenderView.FULL_SIM_STATE, RenderView.AGENT_PERSP, RenderView.BEV_AGENT_OBS],
        }
        _VIEW_SUFFIX = {
            RenderView.FULL_SIM_STATE: "sim_state",
            RenderView.AGENT_PERSP: "persp",
            RenderView.BEV_AGENT_OBS: "bev",
        }

        eval_config = copy.deepcopy(configs)
        # Create separate evaluation environments based on specified configs
        eval_config["env"]["termination_mode"] = 0
        backend = eval_config["eval"].get("backend", "PufferEnv")
        eval_config["env"]["map_dir"] = eval_config["eval"]["map_dir"]
        eval_config["env"]["num_agents"] = eval_config["eval"]["num_eval_agents"]
        if self.configs["eval"]["human_replay_eval"]:
            eval_config["env"]["episode_length"] = 91  # WOMD scenario length
        eval_config["vec"] = dict(backend=backend, num_envs=1)

        self.render_sp_rollout = self.configs["eval"]["render_self_play_eval"]
        self.render_hr_rollout = self.configs["eval"]["render_human_replay_eval"]

        view_mode_str = str(self.configs["eval"].get("render_view_mode", "sim_state")).lower().strip('"').strip("'")
        self.render_view_modes = _VIEW_MODE_MAP.get(view_mode_str, [RenderView.FULL_SIM_STATE])
        self.render_view_suffix = _VIEW_SUFFIX

        self.hr_eval_config = copy.deepcopy(eval_config)
        self.hr_eval_config["env"]["control_mode"] = "control_sdc_only"
        self.hr_eval_config["env"]["render_mode"] = (
            0  # primary env: stats only; render envs created per-view in rollout()
        )

        self.sp_eval_config = copy.deepcopy(eval_config)
        self.sp_eval_config["env"]["control_mode"] = "control_agents"
        self.sp_eval_config["env"]["render_mode"] = (
            0  # primary env: stats only; render envs created per-view in rollout()
        )

        self.render_select_mode = self.configs["eval"]["render_select_mode"]

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

    def rollout(self, policy, mode="self_play"):
        """Roll out the given policy in the specified eval env and collect statistics.

        Stats are collected using the primary env (already loaded by pufferl.py).
        If rendering is enabled, each view mode gets its own temporary env with
        render_mode=1, so each view has its own ffmpeg pipe and uniquely named mp4.
        """
        from pufferlib.pufferl import load_env

        if mode == "human_replay":
            env = self.hr_env
            eval_config = self.hr_eval_config
            render_eval = self.render_hr_rollout
        else:  # self_play
            env = self.sp_env
            eval_config = self.sp_eval_config
            render_eval = self.render_sp_rollout
        driver = env.driver_env

        needs_stats_first = render_eval and self.render_select_mode not in (self.RENDER_FIRST, self.RENDER_RANDOM)

        if needs_stats_first:
            env_logs = self._run_rollout(policy, env, per_env_logs=True)
            render_env_idx = self.select_render_env(env_logs)
        else:
            render_env_idx = self.select_render_env([{}] * driver.num_envs)

        # Collect stats from the primary env (no rendering)
        info_list = self._run_rollout(policy, env)
        final_info = info_list[0] if info_list else {}

        # Render each view in its own temporary env so each gets its own ffmpeg pipe
        # and uniquely named mp4 (e.g. {scenario_id}_bev.mp4).
        if render_eval:
            view_modes = self.render_view_modes
            multi_view = len(view_modes) > 1
            for view_mode in view_modes:
                suffix = f"_{self.render_view_suffix[view_mode]}" if multi_view else ""
                render_cfg = copy.deepcopy(eval_config)
                render_cfg["env"]["render_mode"] = 1
                render_env = load_env("puffer_drive", render_cfg)
                try:
                    self._run_rollout(
                        policy,
                        render_env,
                        render_env_idx=render_env_idx,
                        view_mode=view_mode,
                        view_suffix=suffix,
                    )
                finally:
                    render_env.close()

        if mode == "self_play":
            self.self_play_stats = final_info
            self.self_play_stats["render_env_idx"] = render_env_idx
        elif mode == "human_replay":
            self.human_replay_stats = final_info
            self.human_replay_stats["render_env_idx"] = render_env_idx

    def _run_rollout(self, policy, env, render_env_idx=None, per_env_logs=False, view_mode=None, view_suffix=""):
        """Run a single rollout. If render_env_idx is not None, render that env.

        view_mode: RenderView enum to use when rendering (defaults to FULL_SIM_STATE).
        view_suffix: appended to the mp4 filename when logging multiple views, e.g. "_bev".
        """
        from pufferlib.ocean.drive.drive import RenderView

        driver = env.driver_env
        num_agents = env.observation_space.shape[0]
        device = self.configs["train"]["device"]
        if view_mode is None:
            view_mode = RenderView.FULL_SIM_STATE

        # Set video filename suffix in C before the first render call of this rollout.
        # This ensures each view produces a uniquely named mp4 (e.g. {scenario_id}_bev.mp4).
        if render_env_idx is not None and view_suffix:
            driver.set_video_suffix(view_suffix, env_id=render_env_idx)

        # Reset environment
        obs, info = env.reset()

        # Initialize RNN state if needed
        state = {}
        if self.configs["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        info_list = []
        for time_idx in range(self.sim_steps):
            if render_env_idx is not None:
                driver.render(view_mode=view_mode, env_id=render_env_idx)

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
            obs, rewards, dones, truncs, info_list = env.step(action_np, per_env_logs=per_env_logs)

            if truncs.all():
                break

        return info_list

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
        multi_view = len(self.render_view_modes) > 1
        _known_suffixes = {"_sim_state", "_persp", "_bev"}

        for p in video_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            # Extract view suffix from filename if present (e.g. "abc123_bev" → view="bev")
            view_tag = ""
            if multi_view:
                for s in _known_suffixes:
                    if stem.endswith(s):
                        view_tag = s[1:]  # strip leading "_"
                        stem = stem[: -len(s)]
                        break
            scenario_id = stem
            caption = f"scene_{scenario_id}_epoch_{epoch}_select_{render_mode}"
            wandb_key = f"render/{eval_mode}/{view_tag}" if view_tag else f"render/{eval_mode}"
            self.logger.wandb.log({wandb_key: wandb.Video(p, format="mp4", caption=caption)})

        # Clean up
        for p in video_files:
            os.remove(p)

    def log_stats(self):
        if not (self.logger and hasattr(self.logger, "wandb") and self.logger.wandb):
            return

        eval_stats = {}

        if self.human_replay_stats is not None:
            if "collision_rate" in self.human_replay_stats:
                eval_stats["eval/hr_collision_rate"] = self.human_replay_stats["collision_rate"]
            if "score" in self.human_replay_stats:
                eval_stats["eval/hr_score"] = self.human_replay_stats["score"]
        if self.self_play_stats is not None:
            if "collision_rate" in self.self_play_stats:
                eval_stats["eval/sp_collision_rate"] = self.self_play_stats["collision_rate"]
            if "score" in self.self_play_stats:
                eval_stats["eval/sp_score"] = self.self_play_stats["score"]
            if "n" in self.self_play_stats:
                eval_stats["eval/num_agents"] = self.self_play_stats["n"]

        if not eval_stats:
            return

        self.logger.wandb.log(eval_stats)


class SafeEvaluator:
    """Evaluates policies with fixed safe/law-abiding reward conditioning.

    Runs the policy with deterministic reward bounds (reward_randomization=1,
    min=max for each bound) so the conditioning observation vector is fixed to
    safe driving values. Collects metrics over multiple episodes.

    Re-parses the INI config internally so it doesn't need the full training
    args dict — only the env_name, safe_eval config, and device.
    """

    def __init__(self, env_name: str, safe_eval_config: Dict, device="cuda", logger=None):
        self.env_name = env_name
        self.logger = logger
        self.safe_eval_config = safe_eval_config
        self.num_episodes = safe_eval_config.get("num_episodes", 100)
        self.num_agents = safe_eval_config.get("num_agents", 64)
        self.episode_length = safe_eval_config.get("episode_length", 1000)
        if isinstance(device, int):
            device = f"cuda:{device}"
        self.device = device
        self.stats = None

    def _build_eval_env_config(self):
        """Build env config with safe reward conditioning values applied.

        Re-parses the INI file to get a fresh full config, then applies
        safe_eval overrides for env, vec, and reward bounds.
        """
        import re
        from pufferlib.pufferl import load_config

        # Re-parse INI to get full config (env, vec, policy, rnn, etc.)
        import sys

        original_argv = sys.argv
        sys.argv = ["pufferl"]
        try:
            eval_config = load_config(self.env_name)
        finally:
            sys.argv = original_argv

        eval_config["vec"] = dict(backend="PufferEnv", num_envs=1)
        eval_config["train"]["device"] = self.device
        eval_config["env"]["num_agents"] = self.num_agents
        eval_config["env"]["episode_length"] = self.episode_length
        eval_config["env"]["resample_frequency"] = 0
        eval_config["env"]["reward_randomization"] = 1
        eval_config["env"]["reward_conditioning"] = 1

        # Apply safe_eval overrides for map_dir, goal distances, etc.
        for override_key in ("map_dir", "num_maps", "min_goal_distance", "max_goal_distance"):
            if override_key in self.safe_eval_config:
                eval_config["env"][override_key] = self.safe_eval_config[override_key]

        # Discover valid reward bound names from env config
        valid_bounds = set()
        for key in eval_config["env"]:
            m = re.match(r"reward_bound_(.+)_min$", key)
            if m:
                valid_bounds.add(m.group(1))

        # Set min=max for each reward bound to fix the conditioning values
        for key, val in self.safe_eval_config.items():
            if key not in valid_bounds:
                continue
            eval_config["env"][f"reward_bound_{key}_min"] = float(val)
            eval_config["env"][f"reward_bound_{key}_max"] = float(val)

        return eval_config

    def evaluate(self, vecenv, policy):
        """Run evaluation with safe reward conditioning and collect metrics.

        Args:
            vecenv: Vectorized environment (created with safe eval config)
            policy: Trained policy to evaluate

        Returns:
            dict: Averaged metrics over collected episodes
        """
        from collections import defaultdict

        policy.eval()
        num_agents = vecenv.observation_space.shape[0]
        use_rnn = hasattr(policy, "hidden_size")

        ob, _ = vecenv.reset()
        state = {}
        dones = torch.zeros(num_agents, device=self.device)
        prev_rewards = torch.zeros(num_agents, device=self.device)
        if use_rnn:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=self.device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=self.device),
            )

        all_stats = defaultdict(list)
        episodes_collected = 0
        max_steps = (self.num_episodes // max(num_agents, 1) + 2) * self.episode_length

        for step in range(max_steps):
            if episodes_collected >= self.num_episodes:
                break

            with torch.no_grad():
                ob_t = torch.as_tensor(ob).to(self.device)
                if use_rnn:
                    state["reward"] = prev_rewards
                    state["done"] = dones
                logits, value = policy.forward_eval(ob_t, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action = action.cpu().numpy().reshape(vecenv.action_space.shape)

            if isinstance(logits, torch.distributions.Normal):
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

            ob, rewards, terminals, truncations, infos = vecenv.step(action)
            prev_rewards = torch.as_tensor(rewards).float().to(self.device)
            dones = torch.as_tensor(np.maximum(terminals, truncations)).float().to(self.device)

            for entry in infos:
                if isinstance(entry, dict):
                    episodes_collected += int(entry.get("n", 1))
                    for k, v in entry.items():
                        try:
                            float(v)
                            all_stats[k].append(v)
                        except (TypeError, ValueError):
                            pass

        self.stats = {k: float(np.mean(v)) for k, v in all_stats.items() if len(v) > 0}
        return self.stats

    def log_stats(self, global_step=None):
        """Log collected metrics to wandb."""
        if self.stats is None:
            return

        if not (self.logger and hasattr(self.logger, "wandb") and self.logger.wandb):
            return

        payload = {f"eval/safe_{k}": v for k, v in self.stats.items()}
        if global_step is not None:
            payload["train_step"] = global_step
        self.logger.wandb.log(payload)
