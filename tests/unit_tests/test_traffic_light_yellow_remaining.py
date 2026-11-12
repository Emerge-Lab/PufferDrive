"""obs_traffic_light_yellow_remaining: the co-sim setter's timing reaches the traffic-control obs column."""

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

MAP_DIR = "pufferlib/resources/drive/binaries/carla"
YELLOW = binding.TRAFFIC_CONTROL_STATE_YELLOW
RED = binding.TRAFFIC_CONTROL_STATE_RED
YELLOW_MAX_S = 3.0
DT = 0.1


def _make_env(**overrides):
    kwargs = {
        "num_agents": 16,
        "min_agents_per_env": 1,
        "max_agents_per_env": 16,
        "num_maps": 1,
        "allow_map_subset": True,
        "map_dir": MAP_DIR,
        "simulation_mode": "gigaflow",
        "scenario_length": 120,
        "resample_frequency": 1_000_000,
        "termination_mode": 0,
        "seed": 3,
        "dt": DT,
        "traffic_lights_enabled": True,
        "obs_slots_traffic_controls_n": 4,
        "traffic_light_yellow_min_s": 2.0,
        "traffic_light_yellow_max_s": YELLOW_MAX_S,
        "obs_traffic_light_yellow_remaining": True,
    }
    kwargs.update(overrides)
    env = Drive(**kwargs)
    env.reset()
    return env


def _num_traffic(env):
    state = env.get_state()
    scenario = state[0] if isinstance(state, list) else state
    return int(scenario["num_traffic_elements"])


def _light_rows(env):
    """(rows, width) traffic-control rows of every agent that hold a traffic light."""
    slots, width = env.obs_slots_traffic_controls_n, env.traffic_control_features
    end = env.num_obs - env.obs_valid_count_features
    rows = np.asarray(env.recompute_observations())[:, end - slots * width : end].reshape(-1, slots, width)
    return rows[rows[:, :, -2] == binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT]


def test_setter_timing_reaches_obs_column_clipped_to_training_max():
    env = _make_env()
    num_traffic = _num_traffic(env)
    yellow = np.full(num_traffic, YELLOW, np.int32)

    env.set_traffic_light_states(yellow, np.full(num_traffic, 5.0))  # above the training max: clipped to 3 s
    rows = _light_rows(env)
    yellow_rows = rows[rows[:, -1] == YELLOW]
    assert len(yellow_rows) > 0
    assert np.allclose(yellow_rows[:, -3], 1.0)
    assert np.all(rows[rows[:, -1] != YELLOW][:, -3] == 0.0)

    env.set_traffic_light_states(yellow, np.full(num_traffic, 1.0))  # 10 steps of 0.1 s
    rows = _light_rows(env)
    assert np.allclose(rows[rows[:, -1] == YELLOW][:, -3], 1.0 / YELLOW_MAX_S, atol=1e-6)

    env.set_traffic_light_states(np.full(num_traffic, RED, np.int32), np.zeros(num_traffic))
    rows = _light_rows(env)
    assert len(rows) > 0
    assert np.all(rows[:, -3] == 0.0)


def test_setter_rejects_missing_or_invalid_timing_when_observed():
    env = _make_env()
    num_traffic = _num_traffic(env)
    yellow = np.full(num_traffic, YELLOW, np.int32)
    with pytest.raises(ValueError):
        env.set_traffic_light_states(yellow)
    with pytest.raises(ValueError):
        env.set_traffic_light_states(yellow, np.full(num_traffic, np.nan))
    with pytest.raises(ValueError):
        env.set_traffic_light_states(yellow, np.full(num_traffic, -1.0))
    with pytest.raises(ValueError):
        env.set_traffic_light_states(yellow, np.zeros(num_traffic - 1))


def test_flag_off_keeps_the_row_width_and_accepts_timing():
    env = _make_env(obs_traffic_light_yellow_remaining=False)
    num_traffic = _num_traffic(env)
    assert env.traffic_control_features == binding.TRAFFIC_CONTROL_FEATURES
    env.set_traffic_light_states(np.full(num_traffic, YELLOW, np.int32))
    env.set_traffic_light_states(np.full(num_traffic, YELLOW, np.int32), np.full(num_traffic, 1.0))
    assert len(_light_rows(env)) > 0
