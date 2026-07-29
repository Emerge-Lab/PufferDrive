"""Verify that compact C replay capture is deterministic, complete, and frame-aligned."""

import os
import pickle
import zlib

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")
SCENARIO_LENGTH = 400
REPLAY_SEED = 1234
REMOVED_FIELD_IDX = 5
REMOVED_FIELD_UNAVAILABLE = -2


def _make_replay_drive(capture_replay, compute_eval_metrics=True, num_environments=1):
    return Drive(
        num_agents=num_environments,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=SCENARIO_LENGTH,
        termination_mode=0,
        terminate_on_goal=True,
        num_goals=3,
        goal_radius=2.0,
        goal_source="gt",
        eval_mode=1,
        num_eval_scenarios=num_environments,
        eval_map_indices=[0] * num_environments,
        eval_scenario_seeds=[REPLAY_SEED + env_idx for env_idx in range(num_environments)],
        capture_replay=capture_replay,
        compute_eval_metrics=compute_eval_metrics,
    )


def _extract_python_replay_frame(scenario, episode_timestep):
    agent_capacity = len(scenario["agents"] or [])
    traffic_capacity = max(len(scenario["traffic_elements"] or []), 1)
    frame = {
        "agent_f32": np.zeros((agent_capacity, binding.AGENT_F32_FIELDS), dtype=np.float32),
        "agent_i32": np.zeros((agent_capacity, binding.AGENT_I32_FIELDS), dtype=np.int32),
        "metrics_f32": np.zeros((agent_capacity, binding.METRICS_F32_FIELDS), dtype=np.float32),
        "puffer_f32": np.zeros((agent_capacity, binding.SCORE_F32_FIELDS), dtype=np.float32),
        "traffic_i16": np.zeros((traffic_capacity, binding.TRAFFIC_I16_FIELDS), dtype=np.int16),
    }
    active_indices = {agent_idx: active_idx for active_idx, agent_idx in enumerate(scenario["active_agent_indices"])}
    puffer_keys = (
        "score",
        "no_at_fault",
        "no_offroad",
        "no_red_light",
        "making_progress",
        "direction_score",
        "ttc_puffer_rate",
        "progress_ratio",
        "speed_limit_compliance",
        "comfort_score",
        "multi_lane_score",
        "wrong_way_distance",
        "speed_violation_sum",
        "multiplier",
        "weighted_average",
    )
    for agent_idx, agent in enumerate(scenario["agents"] or []):
        frame["agent_f32"][agent_idx] = (
            agent["sim_x"],
            agent["sim_y"],
            agent["sim_z"],
            agent["sim_heading"],
            agent["sim_length"],
            agent["sim_width"],
            agent["sim_speed"],
            agent["sim_steering"],
            agent["accel_long"],
            agent["accel_lat"],
            agent["jerk_long"],
            agent["jerk_lat"],
        )
        frame["agent_i32"][agent_idx] = (
            agent["id"],
            agent["type"],
            agent["sim_valid"],
            agent["active_agent"],
            agent["stopped"],
            REMOVED_FIELD_UNAVAILABLE,
            agent["current_lane_idx"],
            active_indices.get(agent_idx, -1),
        )
        frame["metrics_f32"][agent_idx] = np.asarray(agent["metrics_array"], dtype=np.float32)
        puffer_metrics = agent.get("puffer_metrics")
        if puffer_metrics is not None:
            frame["puffer_f32"][agent_idx] = tuple(puffer_metrics[key] for key in puffer_keys)
    for traffic_idx, traffic in enumerate(scenario["traffic_elements"] or []):
        states = traffic["states"] or []
        state = states[episode_timestep] if episode_timestep < len(states) else 0
        frame["traffic_i16"][traffic_idx] = (1, traffic["type"], state)
    return frame


def test_exact_seed_matches_after_resampling():
    """Resampling the same evaluation seed reproduces the same scenario."""
    env = Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=SCENARIO_LENGTH,
        termination_mode=0,
        terminate_on_goal=True,
        num_goals=3,
        goal_radius=2.0,
        goal_source="gt",
        eval_mode=1,
        num_eval_scenarios=2,
        eval_map_indices=[0, 0],
        eval_scenario_seeds=[REPLAY_SEED, REPLAY_SEED],
        obs_dropout_lane=0.5,
        obs_dropout_boundary=0.5,
    )
    observation_batches = []
    mask_batches = []
    summaries = []
    zero_action = np.zeros_like(env.actions)
    try:
        env.reset(seed=0)
        for _ in range(2):
            observations = []
            masks = []
            for _ in range(SCENARIO_LENGTH):
                observations.append(env.observations.copy())
                masks.append(env.masks.copy())
                *_, infos = env.step(zero_action)
                summaries.extend(info for info in infos if info.get("summary_type") == "evaluation_episode")
            observation_batches.append(np.stack(observations))
            mask_batches.append(np.stack(masks))
    finally:
        env.close()

    assert len(summaries) == 2
    np.testing.assert_array_equal(observation_batches[0], observation_batches[1])
    np.testing.assert_array_equal(mask_batches[0], mask_batches[1])

    numeric_keys = [key for key, value in summaries[0].items() if isinstance(value, (int, float)) and key != "env_slot"]
    np.testing.assert_array_equal(
        np.asarray([summaries[0][key] for key in numeric_keys]),
        np.asarray([summaries[1][key] for key in numeric_keys]),
    )
    assert summaries[0]["seed"] == summaries[1]["seed"] == REPLAY_SEED
    assert summaries[0]["map_name"] == summaries[1]["map_name"]
    assert summaries[0]["scenario_id"] == summaries[1]["scenario_id"]


@pytest.mark.parametrize("compute_eval_metrics", [False, True])
def test_bulk_replay_frame_matches_python_state_extraction(compute_eval_metrics):
    """The compact C replay frame matches the readable Python state."""
    env = _make_replay_drive(
        capture_replay=False,
        compute_eval_metrics=compute_eval_metrics,
        num_environments=2,
    )
    try:
        env.reset(seed=0)
        observed_nonzero_puffer_metrics = False
        for episode_timestep in range(2):
            scenarios = env.get_state()
            agent_capacity = max(len(scenario["agents"] or []) for scenario in scenarios)
            traffic_capacity = max(max(len(scenario["traffic_elements"] or []), 1) for scenario in scenarios)
            bulk_frame = {
                "agent_f32": np.empty(
                    (len(scenarios), agent_capacity, binding.AGENT_F32_FIELDS),
                    dtype=np.float32,
                ),
                "agent_i32": np.empty(
                    (len(scenarios), agent_capacity, binding.AGENT_I32_FIELDS),
                    dtype=np.int32,
                ),
                "metrics_f32": np.empty(
                    (len(scenarios), agent_capacity, binding.METRICS_F32_FIELDS),
                    dtype=np.float32,
                ),
                "puffer_f32": np.empty(
                    (len(scenarios), agent_capacity, binding.SCORE_F32_FIELDS),
                    dtype=np.float32,
                ),
                "traffic_i16": np.empty(
                    (len(scenarios), traffic_capacity, binding.TRAFFIC_I16_FIELDS),
                    dtype=np.int16,
                ),
            }
            binding.vec_get_obs_html_frame(
                env.c_envs,
                bulk_frame["agent_f32"],
                bulk_frame["agent_i32"],
                bulk_frame["metrics_f32"],
                bulk_frame["puffer_f32"],
                bulk_frame["traffic_i16"],
            )
            observed_nonzero_puffer_metrics |= bool(np.any(bulk_frame["puffer_f32"]))
            for env_idx, scenario in enumerate(scenarios):
                expected = _extract_python_replay_frame(scenario, episode_timestep)
                for key, expected_values in expected.items():
                    actual_values = bulk_frame[key][env_idx, : expected_values.shape[0]]
                    if key == "agent_i32":
                        actual_values = np.delete(actual_values, REMOVED_FIELD_IDX, axis=1)
                        expected_values = np.delete(expected_values, REMOVED_FIELD_IDX, axis=1)
                    np.testing.assert_array_equal(actual_values, expected_values)
            env.step(np.zeros_like(env.actions))
        assert observed_nonzero_puffer_metrics == compute_eval_metrics
    finally:
        env.close()


def test_replay_environment_capture_trims_frozen_early_termination():
    """Early termination excludes frozen tail frames from the replay bundle."""
    env = _make_replay_drive(capture_replay=True)
    try:
        env.reset(seed=0)

        def reject_per_step_state_conversion():
            raise AssertionError("Replay capture must not call get_state() on each step")

        env.get_state = reject_per_step_state_conversion
        completed_summary = None
        zero_action = np.zeros_like(env.actions)
        for _ in range(SCENARIO_LENGTH):
            *_, infos = env.step(zero_action)
            completed_summary = next(
                (info for info in infos if info.get("summary_type") == "evaluation_episode"),
                completed_summary,
            )
    finally:
        env.close()

    assert completed_summary is not None
    episode_length = int(completed_summary["episode_length"])
    assert episode_length < SCENARIO_LENGTH

    replay_bundle = pickle.loads(zlib.decompress(completed_summary["replay_environment_bundle"]))

    assert replay_bundle["schema"] == "interactive_replay_environment_v1"
    assert replay_bundle["metadata"]["episode_length"] == episode_length
    assert replay_bundle["metadata"]["map_path"].endswith(".bin")
    assert replay_bundle["frames"]["agent_f32"].shape[0] == episode_length
    assert replay_bundle["frames"]["agent_i32"].shape[0] == episode_length
    assert replay_bundle["frames"]["metrics_f32"].shape[0] == episode_length
    assert replay_bundle["frames"]["puffer_f32"].shape[0] == episode_length
    assert replay_bundle["frames"]["traffic_i16"].shape[0] == episode_length
