"""Test that randomize_respawn produces different agent positions across resets.

Run on the cluster with:
    srun ... python -m pytest tests/test_randomize_respawn.py -v
"""

import numpy as np
import pytest
from pufferlib.ocean.drive.drive import Drive


MAP_DIR = "pufferlib/resources/drive/binaries/carla_data"


def get_agent_positions(env):
    """Extract current agent positions from observations."""
    # Ego obs starts at index 0: sim_x, sim_y are the first features
    # But obs are in ego frame (normalized). Use the C env directly.
    # The simplest proxy: just hash the full observation vector.
    return env.observations.copy()


@pytest.fixture
def env_randomize():
    e = Drive(
        num_agents=8,
        num_maps=2,
        map_dir=MAP_DIR,
        dynamics_model="classic",
        min_agents_per_env=1,
        max_agents_per_env=8,
        init_mode="init_variable_agent_number",
        control_mode="control_vehicles",
        episode_length=300,
        resample_frequency=0,
        randomize_respawn=1,
    )
    e.reset()
    yield e
    e.close()


@pytest.fixture
def env_no_randomize():
    e = Drive(
        num_agents=8,
        num_maps=2,
        map_dir=MAP_DIR,
        dynamics_model="classic",
        min_agents_per_env=1,
        max_agents_per_env=8,
        init_mode="init_variable_agent_number",
        control_mode="control_vehicles",
        episode_length=10,
        resample_frequency=0,
        randomize_respawn=0,
    )
    e.reset()
    yield e
    e.close()


def test_randomize_respawn_produces_different_positions(env_randomize):
    """With randomize_respawn=1, positions should differ after episode reset."""
    env = env_randomize
    actions = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    # Get initial observations
    obs_before = env.observations.copy()

    # Step until episode resets (episode_length=300, or force via resample)
    env.resample_maps()
    obs_after = env.observations.copy()

    # Observations should differ (agents at different positions)
    assert not np.allclose(obs_before, obs_after, atol=1e-6), (
        "Observations should differ after reset with randomize_respawn=1"
    )


def test_no_randomize_same_positions(env_no_randomize):
    """With randomize_respawn=0, positions should be the same after episode reset."""
    env = env_no_randomize
    actions = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

    # Get initial observations
    obs_before = env.observations.copy()

    # Step through the full episode to trigger c_reset
    for _ in range(15):
        env.step(actions)

    # After reset, positions should return to initial state
    # Note: obs won't be exactly the same due to metrics/counters,
    # but the position-related features should match
    obs_after = env.observations.copy()

    # With no randomization, the first few ego features (position-related)
    # should be identical after reset
    # ego features: speed, heading components, goal direction, etc.
    # After a full reset with no randomization, agents return to log_trajectory[0]
    assert np.allclose(obs_before[:, :5], obs_after[:, :5], atol=0.1), (
        "Position features should be similar after reset with randomize_respawn=0"
    )


def test_multiple_resets_produce_variety(env_randomize):
    """Multiple resets with randomize_respawn should produce different positions each time."""
    env = env_randomize
    observations = []

    for _ in range(5):
        env.resample_maps()
        observations.append(env.observations[:, :10].copy())

    # Check that not all resets produce the same observations
    all_same = all(np.allclose(observations[0], obs, atol=1e-6) for obs in observations[1:])
    assert not all_same, "5 resets with randomize_respawn should not all produce identical observations"
