"""Co-sim ObsReplayCapture: its frame call must track Drive.get_obs_html_frame and the replay must render."""

import os

import numpy as np

from pufferlib import viz
from pufferlib.ocean.cosim.obs_replay import ObsReplayCapture
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")
NUM_GOALS = 3
STEP_COUNT = 5


def _fixture_env():
    return Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        goal_source="gt",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=400,
        resample_frequency=1_000_000,
        termination_mode=False,
        terminate_on_goal=False,
        report_interval=1,
        num_goals=NUM_GOALS,
        goal_radius=2.0,
    )


def test_capture_writes_goal_reward_coef_chunks(tmp_path):
    env = _fixture_env()
    obs, _ = env.reset()
    capture = ObsReplayCapture(env, None, tmp_path / "route", max_steps=STEP_COUNT)
    for _ in range(STEP_COUNT):
        actions = env.action_space.sample()
        capture.capture(obs, actions, {}, None)
        obs, *_ = env.step(actions)
    sdc_idx = env.get_state()[0]["active_agent_indices"][0]
    html_path = capture.write(render_html=True)
    env.close()

    assert len(capture) == STEP_COUNT
    assert os.path.getsize(html_path) > 0
    _, chunks = viz.read_replay_zlib(str(tmp_path / "route.replay.zlib"))
    agent_cap = capture.agent_cap
    assert chunks["goals_f32"].shape == (STEP_COUNT, agent_cap, NUM_GOALS * binding.GOAL_XY_FIELDS)
    assert chunks["rewards_f32"].shape == (STEP_COUNT, agent_cap, binding.REWARD_F32_FIELDS)
    assert chunks["coefs_f32"].shape == (STEP_COUNT, agent_cap, binding.NUM_REWARD_COEFS)
    assert np.any(chunks["goals_f32"][0, sdc_idx] != 0.0)
    assert np.any(chunks["coefs_f32"][0, sdc_idx] != 0.0)
