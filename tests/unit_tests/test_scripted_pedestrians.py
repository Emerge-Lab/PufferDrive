"""Scripted pedestrians (env.scripted_pedestrian_ratio): config validation, per-env slot rounding, and a
short gigaflow rollout in which walkers appear as static partners and get logged."""

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive

MAP_DIR = "pufferlib/resources/drive/binaries/carla"
AGENTS_PER_ENV = 32
CAR_SPEED_MPS = 5.0


def _kwargs(**overrides):
    kwargs = {
        "num_agents": AGENTS_PER_ENV,
        "min_agents_per_env": AGENTS_PER_ENV,
        "max_agents_per_env": AGENTS_PER_ENV,
        "num_maps": 1,
        "allow_map_subset": True,
        "map_dir": MAP_DIR,
        "simulation_mode": "gigaflow",
        "scenario_length": 300,
        "resample_frequency": 0,
        "termination_mode": 0,
        "stagger_first_episode": False,
        "seed": 3,
    }
    kwargs.update(overrides)
    return kwargs


def _launch_cars(env):
    """Walkers appear beside moving cars: give every car CAR_SPEED_MPS along its heading."""
    state = env.get_global_agent_state()
    zeros = np.zeros(env.num_agents, np.float32)
    env.set_agent_states(
        np.arange(env.num_agents, dtype=np.int32),
        state["x"],
        state["y"],
        state["z"],
        state["heading"],
        CAR_SPEED_MPS * np.cos(state["heading"]),
        CAR_SPEED_MPS * np.sin(state["heading"]),
        zeros,
        zeros,
    )


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


def test_rollout_spawns_pedestrians_as_static_partners():
    env = Drive(**_kwargs(scripted_pedestrian_ratio=0.125))
    try:
        env.reset(seed=3)
        _launch_cars(env)
        assert env.num_envs == 1
        expected_slots = 4
        state = env.get_global_agent_state(include_static=True)
        assert len(state["x"]) == env.num_agents + expected_slots

        actions = np.zeros_like(env.actions)
        walkers_seen = 0
        appearances_logged = 0.0
        for _ in range(env.scenario_length + 1):
            obs, _, _, _, infos = env.step(actions)
            assert np.all(np.isfinite(obs))
            state = env.get_global_agent_state(include_static=True)
            walker_x = state["x"][env.num_agents :]
            walkers_seen += int(np.sum(walker_x > -9000.0))
            for info in infos:
                appearances_logged += float(info["pedestrian_crossings"]) + float(info["pedestrian_bystanders"])
        assert walkers_seen > 0
        assert appearances_logged > 0.0
    finally:
        env.close()


def _walker_trace(seed, num_steps):
    env = Drive(**_kwargs(scripted_pedestrian_ratio=0.125, seed=seed))
    try:
        env.reset(seed=seed)
        _launch_cars(env)
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
