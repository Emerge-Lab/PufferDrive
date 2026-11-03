"""Scripted pedestrian crossings (env.scripted_pedestrian_ratio): config validation, per-env slot
rounding, and a short gigaflow rollout in which walkers appear as static partners and get logged."""

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive

MAP_DIR = "pufferlib/resources/drive/binaries/carla"
AGENTS_PER_ENV = 32


def _kwargs(**overrides):
    kwargs = {
        "num_agents": AGENTS_PER_ENV,
        "min_agents_per_env": AGENTS_PER_ENV,
        "max_agents_per_env": AGENTS_PER_ENV,
        "num_maps": 1,
        "allow_map_subset": True,
        "map_dir": MAP_DIR,
        "simulation_mode": "gigaflow",
        "action_type": "continuous",
        "dynamics_model": "jerk",
        "scenario_length": 300,
        "resample_frequency": 0,
        "termination_mode": 0,
        "stagger_first_episode": False,
        "spawn_initial_speed": 5.0,
        "seed": 3,
    }
    kwargs.update(overrides)
    return kwargs


def test_ratio_validation():
    with pytest.raises(ValueError, match="scripted_pedestrian_ratio"):
        Drive(**_kwargs(scripted_pedestrian_ratio=-0.1, config_only=True))
    with pytest.raises(ValueError, match="scripted_pedestrian_ratio"):
        Drive(**_kwargs(scripted_pedestrian_ratio=0.1, simulation_mode="replay", config_only=True))
    with pytest.raises(ValueError, match="scripted_pedestrian_ratio"):
        Drive(**_kwargs(scripted_pedestrian_ratio=0.1, cosim_partner_slots=4, config_only=True))


def test_slot_rounding_matches_ceil_of_ratio_times_agents():
    env = Drive(**_kwargs(scripted_pedestrian_ratio=0.1, config_only=True))
    assert [env._scripted_pedestrian_slots(n) for n in (1, 10, 11, 30, 150)] == [1, 1, 2, 3, 15]
    assert Drive(**_kwargs(config_only=True))._scripted_pedestrian_slots(150) == 0
    assert env._env_init_kwargs("unused.bin", 30)["scripted_pedestrian_slots"] == 3


def test_rollout_spawns_crossings_as_static_partners():
    env = Drive(**_kwargs(scripted_pedestrian_ratio=0.125))
    try:
        env.reset(seed=3)
        assert env.num_envs == 1
        expected_slots = 4
        state = env.get_global_agent_state(include_static=True)
        assert len(state["x"]) == env.num_agents + expected_slots

        actions = np.zeros_like(env.actions)
        walkers_seen = 0
        crossings_logged = 0.0
        for _ in range(env.scenario_length + 1):
            obs, _, _, _, infos = env.step(actions)
            assert np.all(np.isfinite(obs))
            state = env.get_global_agent_state(include_static=True)
            walker_x = state["x"][env.num_agents :]
            walkers_seen += int(np.sum(walker_x > -9000.0))
            for info in infos:
                crossings_logged += float(info.get("pedestrian_crossings", 0.0))
        assert walkers_seen > 0
        assert crossings_logged > 0.0
    finally:
        env.close()


def _walker_trace(seed, num_steps):
    env = Drive(**_kwargs(scripted_pedestrian_ratio=0.125, seed=seed))
    try:
        env.reset(seed=seed)
        actions = np.zeros_like(env.actions)
        trace = []
        for _ in range(num_steps):
            env.step(actions)
            trace.append(env.get_global_agent_state(include_static=True)["x"][env.num_agents :].copy())
        return np.asarray(trace)
    finally:
        env.close()


def test_walkers_are_deterministic_per_seed():
    first = _walker_trace(5, 120)
    second = _walker_trace(5, 120)
    assert np.array_equal(first, second)
    assert np.any(first > -9000.0)


def test_zero_ratio_adds_no_slots():
    env = Drive(**_kwargs())
    try:
        env.reset(seed=3)
        state = env.get_global_agent_state(include_static=True)
        assert len(state["x"]) == env.num_agents
    finally:
        env.close()
