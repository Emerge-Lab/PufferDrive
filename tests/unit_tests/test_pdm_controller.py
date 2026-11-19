"""PDM controller: config validation, and PDM-driven agents that follow their routes whatever the policy outputs."""

import numpy as np
import pytest

from opengigaflow.ocean.drive import binding
from opengigaflow.ocean.drive.drive import Drive

STEPS = 150


def make_pdm_env(**overrides):
    kwargs = {
        "num_agents": 16,
        "min_agents_per_env": 16,
        "max_agents_per_env": 16,
        "num_maps": 1,
        "map_dir": "opengigaflow/resources/carla",
        "simulation_mode": "gigaflow",
        "goal_source": "route",
        "sdc_controller": "pdm",
        "non_sdc_controller": "pdm",
        "collision_behavior": "stop",
        "offroad_behavior": "stop",
        "traffic_lights_enabled": False,
        "scenario_length": STEPS + 50,
        "resample_frequency": 0,
        "termination_mode": False,
    }
    kwargs.update(overrides)
    return Drive(**kwargs)


def drive_pdm_agents(action_seed):
    env = make_pdm_env()
    env.reset(seed=1)
    start = env.get_global_agent_state()
    rng = np.random.default_rng(action_seed)
    for _ in range(STEPS):
        env.step(rng.uniform(-1.0, 1.0, size=env.actions.shape).astype(env.actions.dtype))
    end = env.get_global_agent_state()
    still_driving = np.array(env.masks, dtype=bool)
    env.close()
    return start, end, still_driving


def test_pdm_agents_drive_their_routes_and_ignore_policy_actions():
    start, end, still_driving = drive_pdm_agents(action_seed=0)
    _, end_other_actions, _ = drive_pdm_agents(action_seed=1)
    travelled_m = np.hypot(end["x"] - start["x"], end["y"] - start["y"])
    assert still_driving.all()  # a collision or an off-road event would have stopped the agent
    assert np.count_nonzero(travelled_m > 20.0) >= 14
    for key in ("x", "y", "heading"):
        np.testing.assert_array_equal(end[key], end_other_actions[key])


def test_pdm_in_gigaflow_needs_route_goals():
    with pytest.raises(ValueError, match="goal_source 'route'"):
        Drive(simulation_mode="gigaflow", non_sdc_controller="pdm", goal_source="map")


@pytest.mark.parametrize("horizon_s", [0.0, -1.0, float("nan"), float("inf")])
def test_pdm_horizon_must_be_positive_and_finite(horizon_s):
    with pytest.raises(ValueError, match="pdm_horizon_s"):
        Drive(pdm_horizon_s=horizon_s)


def test_non_vehicles_replay_their_logs_next_to_pdm_vehicles():
    env = Drive(config_only=True, simulation_mode="replay", non_sdc_controller="pdm")
    assert env.non_sdc_controller == binding.CONTROLLER_PDM
    assert env.non_vehicle_controller == binding.CONTROLLER_REPLAY
