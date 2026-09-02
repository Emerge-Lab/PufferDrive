"""gt_map goal source: one endpoint goal, then route goal windows past the ego once it is cleared."""

import numpy as np
import pytest

nuplan = pytest.importorskip("nuplan")  # noqa: F841

from pufferlib.ocean.cosim import nuplan_bridge as nb  # noqa: E402
from pufferlib.ocean.cosim.goals import RouteGoalWindow  # noqa: E402
from pufferlib.ocean.cosim.nuplan.planner import PufferDrivePlanner  # noqa: E402


class FakeEnv:
    goal_radius = 10.0
    num_goals = 3

    def __init__(self):
        self.pushed = []
        self.cleared = False

    def set_agent_goals(self, agent_idx, gx, gy, gz, gdir_x, gdir_y):
        self.pushed.append(np.column_stack([gx, gy]))

    def get_agent_goal_progress(self, agent_idx):
        count = len(self.pushed[-1])
        return (count if self.cleared else 0), count


def _planner(tmp_path, env):
    planner = PufferDrivePlanner("dummy", str(tmp_path), goal_source="gt_map", goal_spacing=20.0)
    planner._env = env
    planner._transform = nb.NuPlanTransform(1000.0, 2000.0)
    planner._route_centerline = np.column_stack([1000.0 + np.arange(0.0, 201.0), np.full(201, 2000.0)])
    planner._goal_window = RouteGoalWindow(env, np.array([[40.0, 0.0, 0.0, 1.0, 0.0]], np.float32))
    planner._goal_window.sync(0.0, 0.0, 0.0)
    return planner


def test_endpoint_window_holds_until_cleared(tmp_path):
    env = FakeEnv()
    planner = _planner(tmp_path, env)
    assert len(env.pushed) == 1 and np.allclose(env.pushed[0], [[40.0, 0.0]])
    assert not planner._goal_window.exhausted
    planner._goal_window.sync(20.0, 0.0, 0.0)
    assert len(env.pushed) == 1


def test_cleared_endpoint_continues_with_route_goals_ahead(tmp_path):
    env = FakeEnv()
    planner = _planner(tmp_path, env)
    env.cleared = True
    assert planner._goal_window.exhausted
    planner._continue_along_route(1042.0, 2000.0, 42.0, 0.0, 0.0)
    env.cleared = False
    assert len(env.pushed) == 2
    assert np.allclose(env.pushed[1][:, 0], [60.0, 80.0, 100.0])  # >= 42 + radius 10 + margin 5, bin frame
    assert planner._goal_window.window.shape[0] == 3


def test_route_used_up_keeps_last_goal(tmp_path):
    env = FakeEnv()
    planner = _planner(tmp_path, env)
    env.cleared = True
    planner._continue_along_route(1195.0, 2000.0, 195.0, 0.0, 0.0)
    assert len(env.pushed) == 1
