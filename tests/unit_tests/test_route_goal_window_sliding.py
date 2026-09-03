"""RouteGoalWindow: batch windows replace once exhausted; sliding windows refill after every consumed goal."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.cosim.goals import RouteGoalWindow, route_goals_from_xy

NUM_GOALS = 3
GOAL_RADIUS_M = 10.0
ROUTE_XY = np.array([[20.0, 0.0], [40.0, 0.0], [60.0, 0.0], [80.0, 0.0], [100.0, 0.0]])


class _FakeEnv:
    """Records pushed windows; goal consumption is driven by the test through `consume`."""

    def __init__(self):
        self.num_goals = NUM_GOALS
        self.goal_radius = GOAL_RADIUS_M
        self.pushed_windows = []
        self.current_goal_idx = 0
        self.goal_count = 0

    def set_agent_goals(self, agent_idx, gx, gy, gz, gdir_x=None, gdir_y=None):
        self.pushed_windows.append(np.column_stack([gx, gy]).tolist())
        self.current_goal_idx = 0
        self.goal_count = len(gx)

    def get_agent_goal_progress(self, agent_idx):
        return self.current_goal_idx, self.goal_count

    def consume(self):
        self.current_goal_idx += 1


def test_batch_window_replaces_only_when_exhausted():
    env = _FakeEnv()
    window = RouteGoalWindow(env, route_goals_from_xy(ROUTE_XY))
    window.sync(0.0, 0.0, 0.0)
    assert env.pushed_windows == [[[20.0, 0.0], [40.0, 0.0], [60.0, 0.0]]]
    env.consume()
    window.sync(22.0, 0.0, 0.0)
    assert len(env.pushed_windows) == 1
    env.consume()
    env.consume()
    window.sync(62.0, 0.0, 0.0)
    assert env.pushed_windows[-1] == [[80.0, 0.0], [100.0, 0.0]]


def test_sliding_window_refills_after_each_consumed_goal():
    env = _FakeEnv()
    window = RouteGoalWindow(env, route_goals_from_xy(ROUTE_XY), sliding=True)
    window.sync(0.0, 0.0, 0.0)
    env.consume()
    window.sync(22.0, 0.0, 0.0)
    assert env.pushed_windows[-1] == [[40.0, 0.0], [60.0, 0.0], [80.0, 0.0]]
    assert window.window_start == 1 and window.current_index == 1
    env.consume()
    window.sync(42.0, 0.0, 0.0)
    assert env.pushed_windows[-1] == [[60.0, 0.0], [80.0, 0.0], [100.0, 0.0]]
    env.consume()
    window.sync(62.0, 0.0, 0.0)
    assert env.pushed_windows[-1] == [[80.0, 0.0], [100.0, 0.0]]  # route end: the last window stays partial
    env.consume()
    env.consume()
    window.sync(102.0, 0.0, 0.0)
    assert len(env.pushed_windows) == 4  # exhausted at the route end: nothing left to push


def test_sliding_window_keeps_skip_ahead_when_goal_passed_without_consuming():
    env = _FakeEnv()
    window = RouteGoalWindow(env, route_goals_from_xy(ROUTE_XY), sliding=True)
    window.sync(0.0, 0.0, 0.0)
    window.sync(35.0, 0.0, 0.0)  # 15 m past the first goal, never consumed
    assert env.pushed_windows[-1] == [[40.0, 0.0], [60.0, 0.0], [80.0, 0.0]]
