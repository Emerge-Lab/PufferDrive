"""Config rules of dynamics_model spline_werling / action_type lattice (mirrors lattice.h fail-fast checks)."""

import sys
from unittest.mock import patch

import pytest

import pufferlib
from pufferlib.config_schema import normalize_puffer_drive_config, validate_puffer_drive_config
from pufferlib.pufferl import load_config


def _lattice_args():
    with patch.object(sys, "argv", ["pufferl.py"]):
        return load_config("puffer_drive_spline_werling")


def _validate(args):
    normalized = normalize_puffer_drive_config(args, "test")
    validate_puffer_drive_config(normalized, "test")
    return normalized


def test_spline_werling_config_loads():
    args = _validate(_lattice_args())
    assert args["env"]["dynamics_model"] == "spline_werling"
    assert args["env"]["action_type"] == "lattice"
    assert args["policy"]["action_type"] == "lattice"
    assert args["eval"]["action_selection"] == "sample"
    assert args["env"]["lattice_backup_distances_m"] == [1.0, 2.0, 5.0, 10.0]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("env", "action_type", "continuous"),
        ("policy", "action_type", "discrete"),
        ("env", "dynamics_model", "jerk"),
        ("env", "reset_accel_on_stop", True),
        ("env", "lattice_lat_offsets_m", [-0.9, 0.9]),
        ("env", "lattice_lat_offsets_m", [0.9, 0.0, -0.9]),
        ("env", "lattice_lat_durations_s", [2.4, 3.55]),
        ("env", "lattice_backup_distances_m", [1.0, 14.0]),
        ("env", "lattice_lon_speeds_mps", [0.0, 25.0]),
        ("env", "lattice_low_speed_distances_m", [5.0, 10.0]),
        ("env", "lattice_decision_period_s", 0.25),
        ("eval", "action_selection", "mean"),
    ],
)
def test_invalid_lattice_settings_are_rejected(section, key, value):
    args = _lattice_args()
    args[section][key] = value
    with pytest.raises(pufferlib.APIUsageError):
        _validate(args)


def test_trajectory_training_cannot_use_lattice():
    args = _lattice_args()
    args["trajectory_training"] = True
    with pytest.raises(pufferlib.APIUsageError):
        _validate(args)


def test_goal_exit_mode_needs_goal_lanes():
    args = _lattice_args()
    args["env"]["lattice_exit_mode"] = "goal"
    args["env"]["goal_source"] = "gt"
    args["env"]["simulation_mode"] = "replay"
    with pytest.raises(pufferlib.APIUsageError):
        _validate(args)


def test_dt_01_menus_are_whole_steps():
    args = _lattice_args()
    args["env"]["dt"] = 0.1
    normalized = _validate(args)
    assert normalized["env"]["dt"] == 0.1


def test_oncoming_overtake_config():
    args = _lattice_args()
    args["env"]["lattice_oncoming_overtake"] = True
    args["env"]["reward_oncoming_penalty_frac"] = 5e-4
    args["env"]["reward_wait_penalty_frac"] = 5e-4
    normalized = _validate(args)
    assert normalized["env"]["lattice_oncoming_overtake"] is True


def test_turnaround_config():
    args = _lattice_args()
    assert args["env"]["lattice_turnaround"] is False
    args["env"]["lattice_turnaround"] = True
    normalized = _validate(args)
    assert normalized["env"]["lattice_turnaround"] is True


def test_wait_speed_and_guard_config():
    args = _lattice_args()
    assert args["env"]["reward_wait_full_speed_mps"] == 1.0
    assert args["env"]["reward_wait_guard"] is True
    args["env"].update({"reward_wait_full_speed_mps": 5.0, "reward_wait_penalty_frac": 9.5e-4})
    assert _validate(args)["env"]["reward_wait_full_speed_mps"] == 5.0
    over = _lattice_args()
    over["env"]["reward_wait_penalty_frac"] = 1e-2
    with pytest.raises(pufferlib.APIUsageError):
        _validate(over)
    over["env"]["reward_wait_guard"] = False
    assert _validate(over)["env"]["reward_wait_penalty_frac"] == 1e-2


def test_light_facing_flag_config():
    args = _lattice_args()
    assert args["env"]["obs_light_facing_only"] is False
    args["env"].update({"obs_light_facing_only": True, "lattice_light_in_view": True, "lattice_exit_mode": "policy"})
    normalized = _validate(args)
    assert normalized["env"]["obs_light_facing_only"] is True


def test_overtake_commit_needs_oncoming_overtake():
    args = _lattice_args()
    assert args["env"]["lattice_overtake_commit"] is False
    args["env"].update({"lattice_overtake_commit": True, "lattice_oncoming_overtake": True})
    assert _validate(args)["env"]["lattice_overtake_commit"] is True
    args["env"]["lattice_oncoming_overtake"] = False
    with pytest.raises(pufferlib.APIUsageError, match="lattice_overtake_commit"):
        _validate(args)


def test_wait_guard_uses_bootstrapped_horizon():
    args = _lattice_args()
    assert args["train"]["use_value_bootstrapping"] is True
    args["env"]["reward_wait_penalty_frac"] = 1.05e-3
    with pytest.raises(pufferlib.APIUsageError):
        _validate(args)
    args["train"]["use_value_bootstrapping"] = False
    assert _validate(args)["env"]["reward_wait_penalty_frac"] == 1.05e-3


@pytest.mark.parametrize("speed", [0.0, -1.0])
def test_wait_full_speed_must_be_positive(speed):
    args = _lattice_args()
    args["env"]["reward_wait_full_speed_mps"] = speed
    with pytest.raises(pufferlib.APIUsageError):
        _validate(args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"reward_oncoming_penalty_frac": 5e-4},
        {"lattice_oncoming_overtake": True, "lattice_lat_offsets_m": [-1.5, 0.0, 1.5]},
        {"lattice_oncoming_overtake": True, "reward_oncoming_penalty_frac": 7e-4, "reward_wait_penalty_frac": 7e-4},
    ],
)
def test_invalid_oncoming_overtake_settings_are_rejected(overrides):
    args = _lattice_args()
    args["env"].update(overrides)
    with pytest.raises(pufferlib.APIUsageError):
        _validate(args)
