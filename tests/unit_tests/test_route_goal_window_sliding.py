"""RouteGoalWindow: batch windows replace once exhausted; sliding windows refill after every consumed goal."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.cosim.goals import RouteGoalWindow, route_goals_from_target_points, route_goals_from_xy

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


def test_route_goals_first_direction_comes_from_origin():
    goals = np.array([[10.0, 40.0], [10.0, 80.0]])  # left turn: goals up the cross street
    default = route_goals_from_xy(goals)
    np.testing.assert_allclose(default[0, 3:5], [0.0, 0.0])
    with_origin = route_goals_from_xy(goals, origin_xy=(0.0, 0.0))
    np.testing.assert_allclose(with_origin[0, 3:5], [10.0, 40.0])
    np.testing.assert_allclose(with_origin[1, 3:5], [0.0, 40.0])
    np.testing.assert_allclose(with_origin[:, :2], goals)


def test_target_points_become_route_goals_with_dense_route_direction():
    # CARLA frame: straight east 0..100, then a left turn north (CARLA y down -> north is -y)
    dense = [(x, 0.0, 0.0) for x in range(101)] + [(100.0, -y, 0.0) for y in range(1, 41)]
    targets = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (100.0, -40.0, 0.0)]  # start, junction entry, route end
    to_bin = lambda x, y: (x + 5.0, -y)  # the CARLA->bin transform flips y
    goals = route_goals_from_target_points(targets, dense, to_bin, to_bin(0.0, 0.0), skip_radius=10.0)
    assert goals.shape == (2, 5)  # the start point under the ego is dropped
    np.testing.assert_allclose(goals[:, :2], [[105.0, 0.0], [105.0, 40.0]])
    np.testing.assert_allclose(goals[0, 3:5], [1.0, 0.0])  # arriving at the entry: heading east
    np.testing.assert_allclose(goals[1, 3:5], [0.0, 1.0])  # at the end: heading north in the bin frame
    kept = route_goals_from_target_points(targets, dense, to_bin, to_bin(50.0, 0.0), skip_radius=10.0)
    assert kept.shape == (3, 5)  # nothing near the ego: every target point kept


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
