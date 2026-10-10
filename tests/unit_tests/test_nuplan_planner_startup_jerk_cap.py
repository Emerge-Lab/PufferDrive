"""Start-up jerk caps map onto the continuous jerk action's asymmetric scaling (brake JERK_LONG[0], accel JERK_LONG[-1])."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

nuplan = pytest.importorskip("nuplan")

from pufferlib.ocean.cosim.nuplan.planner import startup_jerk_action_bounds
from pufferlib.ocean.drive import binding


def test_caps_scale_by_the_env_jerk_range():
    brake_range = abs(float(binding.JERK_LONG[0]))
    accel_range = float(binding.JERK_LONG[-1])
    lo, hi = startup_jerk_action_bounds(2.0, 3.0)
    assert lo == pytest.approx(-3.0 / brake_range)
    assert hi == pytest.approx(2.0 / accel_range)


def test_zero_cap_leaves_that_side_uncapped():
    assert startup_jerk_action_bounds(0.0, 0.0) == (-1.0, 1.0)
    assert startup_jerk_action_bounds(2.0, 0.0)[0] == -1.0
    assert startup_jerk_action_bounds(0.0, 2.0)[1] == 1.0


def test_caps_beyond_the_env_range_saturate_at_the_action_limits():
    assert startup_jerk_action_bounds(100.0, 100.0) == (-1.0, 1.0)
