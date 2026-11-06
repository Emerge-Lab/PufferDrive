"""Forecast rollout helpers of the nuPlan planner's nuplan_tracker mode: step count, slot split, light forecast,
and the goal window snapshot that lets a rollout be undone."""

import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.cosim.goals import RouteGoalWindow, route_goals_from_xy
from pufferlib.ocean.cosim.nuplan_bridge import LIGHT_STATE_GREEN, LIGHT_STATE_RED, LIGHT_STATE_YELLOW
from pufferlib.ocean.cosim.nuplan_rollout import forecast_light_states, rollout_step_count, split_rollout_slots


NUM_GOALS = 3
ROUTE_XY = np.array([[20.0, 0.0], [40.0, 0.0], [60.0, 0.0], [80.0, 0.0], [100.0, 0.0]])


class _FakeGoalEnv:
    def __init__(self):
        self.num_goals = NUM_GOALS
        self.goal_radius = 10.0
        self.window = []
        self.current_goal_idx = 0

    def set_agent_goals(self, agent_idx, gx, gy, gz, gdir_x=None, gdir_y=None):
        self.window = np.column_stack([gx, gy]).tolist()
        self.current_goal_idx = 0

    def get_agent_goal_progress(self, agent_idx):
        return self.current_goal_idx, len(self.window)

    def set_agent_goal_progress(self, agent_idx, current_goal_idx):
        assert 0 <= current_goal_idx <= len(self.window)
        self.current_goal_idx = current_goal_idx


def test_step_count_reaches_the_horizon_without_a_rounding_step():
    assert rollout_step_count(8.0, 0.3) == 27  # 26 steps end at 7.8 s
    assert rollout_step_count(8.1, 0.3) == 27  # 8.1 / 0.3 is 27.000000000000004 in floats
    assert rollout_step_count(0.3, 0.3) == 1
    assert rollout_step_count(0.2, 0.3) == 1
    assert rollout_step_count(8.0, 0.1) == 80
    with pytest.raises(ValueError):
        rollout_step_count(8.0, 0.0)
    with pytest.raises(ValueError):
        rollout_step_count(0.0, 0.3)


def test_nearest_eligible_partners_take_the_policy_slots():
    distance_sq = np.array([400.0, 1.0, 25.0, 100.0, 9.0])
    eligible = np.array([True, False, True, True, True])  # row 1: a pedestrian right next to the ego
    policy_rows, static_rows = split_rollout_slots(distance_sq, eligible, 2)
    assert policy_rows.tolist() == [2, 4]  # the two nearest vehicles
    assert static_rows.tolist() == [0, 1, 3]  # the pedestrian and the vehicles that did not fit
    everything, nothing = split_rollout_slots(distance_sq, eligible, 10)
    assert everything.tolist() == [0, 2, 3, 4] and nothing.tolist() == [1]
    none, all_static = split_rollout_slots(distance_sq, eligible, 0)
    assert none.tolist() == [] and all_static.tolist() == [0, 1, 2, 3, 4]


def test_slot_split_of_an_empty_scene():
    policy_rows, static_rows = split_rollout_slots(np.zeros(0), np.zeros(0, bool), 4)
    assert len(policy_rows) == 0 and len(static_rows) == 0


def test_forecast_lights_hold_except_a_yellow_that_runs_out():
    states = np.array([LIGHT_STATE_GREEN, LIGHT_STATE_RED, LIGHT_STATE_YELLOW, LIGHT_STATE_YELLOW], np.int32)
    remaining = np.array([0.0, 0.0, 0.5, 2.0], np.float32)
    early_states, early_remaining = forecast_light_states(states, remaining, 0.3)
    assert early_states.tolist() == states.tolist()
    np.testing.assert_allclose(early_remaining, [0.0, 0.0, 0.2, 1.7], atol=1e-6)
    late_states, late_remaining = forecast_light_states(states, remaining, 0.6)
    assert late_states.tolist() == [LIGHT_STATE_GREEN, LIGHT_STATE_RED, LIGHT_STATE_RED, LIGHT_STATE_YELLOW]
    np.testing.assert_allclose(late_remaining, [0.0, 0.0, 0.0, 1.4], atol=1e-6)
    assert late_states.dtype == np.int32 and late_remaining.dtype == np.float32
    assert states.tolist()[2] == LIGHT_STATE_YELLOW  # the synced arrays are reused for every rollout step


def test_restore_undoes_the_goals_a_rollout_consumed_in_a_sliding_window():
    env = _FakeGoalEnv()
    window = RouteGoalWindow(env, route_goals_from_xy(ROUTE_XY), sliding=True)
    assert window.sync(0.0, 0.0, 0.0) is True  # first push
    assert window.sync(1.0, 0.0, 0.0) is False  # nothing consumed: the observations stay valid
    env.current_goal_idx = 1  # the first rollout step reaches the first goal: this one counts
    assert window.sync(12.0, 0.0, 0.0) is True
    snapshot = window.snapshot()
    for ego_x in (32.0, 52.0):  # later rollout steps run two more goals down
        env.current_goal_idx = 1
        window.sync(ego_x, 0.0, 0.0)
    assert env.window == [[80.0, 0.0], [100.0, 0.0]]
    window.restore(snapshot)
    assert env.window == [[40.0, 0.0], [60.0, 0.0], [80.0, 0.0]]
    assert (window.window_start, window.current_index, env.current_goal_idx) == (1, 1, 0)


def test_restore_keeps_the_consumed_count_of_a_batch_window():
    env = _FakeGoalEnv()
    window = RouteGoalWindow(env, route_goals_from_xy(ROUTE_XY))
    window.sync(0.0, 0.0, 0.0)
    env.current_goal_idx = 1
    window.sync(12.0, 0.0, 0.0)
    snapshot = window.snapshot()  # one of three goals consumed, window not replaced
    env.current_goal_idx = 3
    window.sync(62.0, 0.0, 0.0)  # the rollout exhausts the window: the next one is pushed
    assert env.window == [[80.0, 0.0], [100.0, 0.0]]
    window.restore(snapshot)
    assert env.window == [[20.0, 0.0], [40.0, 0.0], [60.0, 0.0]]
    assert env.current_goal_idx == 1 and window.window_start == 0


def test_restore_keeps_an_exhausted_route_exhausted():
    env = _FakeGoalEnv()
    window = RouteGoalWindow(env, route_goals_from_xy(ROUTE_XY[:1]), sliding=True)
    window.sync(0.0, 0.0, 0.0)
    env.current_goal_idx = 1  # the only goal is consumed: nothing left to push
    assert window.sync(15.0, 0.0, 0.0) is False
    snapshot = window.snapshot()
    window.restore(snapshot)
    assert env.get_agent_goal_progress(0) == (1, 1)  # still consumed: the goal must not reappear


def test_snapshot_needs_a_pushed_window():
    window = RouteGoalWindow(_FakeGoalEnv(), route_goals_from_xy(ROUTE_XY))
    with pytest.raises(RuntimeError):
        window.snapshot()
