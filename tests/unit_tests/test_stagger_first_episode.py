"""Tests for `stagger_first_episode`: the first episode after an env is created ends at a
random step in [init_step_min_horizon, scenario_length], so the sub-envs of a worker stop
running their episodes in lockstep; every later episode keeps the full scenario_length."""

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive


MAP_DIR = "pufferlib/resources/drive/binaries/carla"
SCENARIO_LENGTH = 120
MIN_HORIZON = 20
# Single-agent sub-envs: one independent first-episode draw per agent.
NUM_ENVS = 16


def _make_env(seed, **overrides):
    kwargs = {
        "num_agents": NUM_ENVS,
        "min_agents_per_env": 1,
        "max_agents_per_env": 1,
        "num_maps": 1,
        "allow_map_subset": True,
        "map_dir": MAP_DIR,
        "simulation_mode": "gigaflow",
        "action_type": "continuous",
        "dynamics_model": "jerk",
        "scenario_length": SCENARIO_LENGTH,
        "init_step_min_horizon": MIN_HORIZON,
        "resample_frequency": 1_000_000,
        "termination_mode": 0,
        "stagger_first_episode": True,
        "seed": seed,
    }
    kwargs.update(overrides)
    return Drive(**kwargs)


def _truncation_ticks(env, num_ticks):
    """Per agent, the ticks at which the env reported a truncation (zero actions, no early resets)."""
    ticks = [[] for _ in range(env.num_agents)]
    actions = np.zeros_like(env.actions)
    for tick in range(1, num_ticks + 1):
        _, _, _, truncations, _ = env.step(actions)
        for agent_idx in np.flatnonzero(truncations):
            ticks[agent_idx].append(tick)
    return ticks


def _first_episode_ends(seed, **overrides):
    env = _make_env(seed, **overrides)
    try:
        env.reset(seed=seed)
        return [agent_ticks[0] for agent_ticks in _truncation_ticks(env, SCENARIO_LENGTH + 1)]
    finally:
        env.close()


def test_first_episode_is_staggered_and_later_episodes_are_full_length():
    env = _make_env(seed=0)
    try:
        env.reset(seed=0)
        ticks = _truncation_ticks(env, 2 * SCENARIO_LENGTH + 2)
    finally:
        env.close()

    first_ends = [agent_ticks[0] for agent_ticks in ticks]
    assert all(MIN_HORIZON <= end <= SCENARIO_LENGTH for end in first_ends), first_ends
    assert len(set(first_ends)) >= 2, first_ends
    # one deferred-reset tick, then a full-length second episode
    for agent_ticks in ticks:
        assert agent_ticks[1] == agent_ticks[0] + SCENARIO_LENGTH + 1, agent_ticks


def test_flag_off_keeps_full_length_first_episode():
    first_ends = _first_episode_ends(seed=0, stagger_first_episode=False)
    assert first_ends == [SCENARIO_LENGTH] * NUM_ENVS


def test_first_episode_ends_are_deterministic_per_seed():
    same_seed_a = _first_episode_ends(seed=7)
    same_seed_b = _first_episode_ends(seed=7)
    other_seed = _first_episode_ends(seed=8)
    assert same_seed_a == same_seed_b
    assert same_seed_a != other_seed


@pytest.mark.parametrize("init_step_min_horizon", [0, SCENARIO_LENGTH + 1])
def test_rejects_horizon_outside_scenario(init_step_min_horizon):
    with pytest.raises(ValueError, match="stagger_first_episode"):
        _make_env(seed=0, init_step_min_horizon=init_step_min_horizon)
