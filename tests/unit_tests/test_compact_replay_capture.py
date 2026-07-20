import os
import pickle
import zlib

import numpy as np

from pufferlib.ocean.drive.drive import Drive


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")
SCENARIO_LENGTH = 400
REPLAY_SEED = 1234


def test_replay_environment_capture_trims_frozen_early_termination():
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
        num_target_waypoints=3,
        goal_radius=2.0,
        eval_mode=1,
        num_eval_scenarios=1,
        eval_map_indices=[0],
        eval_scenario_seeds=[REPLAY_SEED],
        capture_replay=True,
    )
    try:
        env.reset(seed=0)
        completed_summary = None
        zero_action = np.zeros_like(env.actions)
        for _ in range(SCENARIO_LENGTH):
            *_, infos = env.step(zero_action)
            completed_summary = next(
                (info for info in infos if info.get("summary_type") == "completed_episode"),
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
