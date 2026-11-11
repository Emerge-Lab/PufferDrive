"""Co-sim forecast goals through the Python binding: agents without a known route get goals along their lane
(Drive.set_agent_lane_goals) and a rollout's goal progress can be put back (Drive.set_agent_goal_progress)."""

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive


MAP_DIR = "pufferlib/resources/drive/binaries/carla"
POLICY_AGENTS = 3
PARTNER_SLOTS = 2
GOAL_SPACING_M = 20.0


@pytest.fixture
def env():
    drive = Drive(
        num_agents=POLICY_AGENTS,
        min_agents_per_env=POLICY_AGENTS,
        max_agents_per_env=POLICY_AGENTS,
        cosim_partner_slots=PARTNER_SLOTS,
        goal_source="external",
        num_maps=1,
        allow_map_subset=True,
        map_dir=MAP_DIR,
        simulation_mode="gigaflow",
        action_type="continuous",
        dynamics_model="jerk",
        scenario_length=300,
        resample_frequency=0,
        termination_mode=0,
        stagger_first_episode=False,
        collision_behavior="ignore",
        offroad_behavior="ignore",
        seed=3,
    )
    drive.reset(seed=3)
    yield drive
    drive.close()


def test_lane_goals_are_placed_ahead_of_spawned_agents(env):
    assert env.get_agent_goal_progress(0) == (0, 0)  # external goal source: spawning assigns no goals
    counts = env.set_agent_lane_goals(np.arange(POLICY_AGENTS), GOAL_SPACING_M)
    assert counts.dtype == np.int32 and counts.shape == (POLICY_AGENTS,)
    assert np.all(counts >= 1) and np.all(counts <= env.num_goals)  # spawned on a lane with road ahead
    for agent_idx in range(POLICY_AGENTS):
        assert env.get_agent_goal_progress(agent_idx) == (0, int(counts[agent_idx]))


def test_lane_goals_survive_a_rollout_of_coasting_agents(env):
    env.set_agent_lane_goals(np.arange(POLICY_AGENTS), GOAL_SPACING_M)
    actions = np.zeros_like(env.actions)
    for _ in range(40):
        env.step(actions)
    for agent_idx in range(POLICY_AGENTS):
        current_goal_idx, goal_count = env.get_agent_goal_progress(agent_idx)
        assert 0 <= current_goal_idx <= goal_count <= env.num_goals
    assert np.all(np.isfinite(env.observations))


def test_goal_progress_can_be_set_back_and_forth(env):
    counts = env.set_agent_lane_goals(np.array([0]), GOAL_SPACING_M)
    goal_count = int(counts[0])
    env.set_agent_goal_progress(0, goal_count)
    assert env.get_agent_goal_progress(0) == (goal_count, goal_count)
    env.set_agent_goal_progress(0, 0)
    assert env.get_agent_goal_progress(0) == (0, goal_count)
    with pytest.raises(ValueError, match="goal_count"):
        env.set_agent_goal_progress(0, goal_count + 1)
    with pytest.raises(ValueError, match="goal_count"):
        env.set_agent_goal_progress(0, -1)
    with pytest.raises(ValueError, match="out of range"):
        env.set_agent_goal_progress(POLICY_AGENTS + PARTNER_SLOTS, 0)


def test_lane_goals_reject_bad_input(env):
    with pytest.raises(ValueError, match="spacing_m"):
        env.set_agent_lane_goals(np.array([0]), 0.0)
    with pytest.raises(ValueError, match="spacing_m"):
        env.set_agent_lane_goals(np.array([0]), float("nan"))
    with pytest.raises(ValueError, match="out of range"):
        env.set_agent_lane_goals(np.array([POLICY_AGENTS + PARTNER_SLOTS]), GOAL_SPACING_M)
    with pytest.raises(ValueError, match="without a pose"):
        env.set_agent_lane_goals(np.array([POLICY_AGENTS]), GOAL_SPACING_M)  # a partner slot nothing was put in
    assert env.set_agent_lane_goals(np.zeros(0, np.int32), GOAL_SPACING_M).shape == (0,)
