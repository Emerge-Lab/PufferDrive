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
