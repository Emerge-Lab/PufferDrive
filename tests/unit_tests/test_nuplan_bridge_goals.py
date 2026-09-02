"""logged_ego_goals: logged ego samples land on the co-directional lane baseline, never on the
oncoming lane, and keep the raw pose when no lane is within the snap radius."""

import math
from types import SimpleNamespace

import numpy as np
import pytest


nuplan = pytest.importorskip("nuplan")
from nuplan.common.actor_state.state_representation import StateSE2

from pufferlib.ocean.cosim import nuplan_bridge as nb


class _StraightBaseline:
    def __init__(self, y, heading):
        self.y, self.heading = y, heading

    def get_nearest_pose_from_position(self, point):
        return StateSE2(point.x, self.y, self.heading)


class _MapApi:
    """Two straight lanes along x: y=0 eastbound, y=3.5 westbound."""

    def get_proximal_map_objects(self, point, radius, layers):
        lanes = [
            SimpleNamespace(baseline_path=_StraightBaseline(0.0, 0.0)),
            SimpleNamespace(baseline_path=_StraightBaseline(3.5, math.pi)),
        ]
        near = [lane for lane in lanes if abs(lane.baseline_path.y - point.y) <= radius]
        return {layers[0]: near, layers[1]: []}


def _scenario(xs, ys):
    states = [SimpleNamespace(center=StateSE2(x, y, 0.0)) for x, y in zip(xs, ys)]
    return SimpleNamespace(get_number_of_iterations=lambda: len(states), get_ego_state_at_iteration=lambda i: states[i])


def test_logged_goals_snap_to_codirectional_lane():
    xs = np.arange(0.0, 61.0, 1.0)
    scenario = _scenario(xs, np.full_like(xs, 1.4))  # eastbound, 1.4 m left of its lane center
    xy, headings, snapped = nb.logged_ego_goals(scenario, _MapApi(), spacing=20.0)
    assert snapped == len(xy) == 3  # 20 m, 40 m, endpoint
    np.testing.assert_allclose(xy[:, 0], [20.0, 40.0, 60.0])
    np.testing.assert_allclose(xy[:, 1], 0.0)  # eastbound lane, never the closer-by-heading westbound one
    np.testing.assert_allclose(headings, 0.0)


def test_logged_goals_keep_raw_pose_without_lane():
    xs = np.arange(0.0, 21.0, 1.0)
    scenario = _scenario(xs, np.full_like(xs, 50.0))
    xy, _, snapped = nb.logged_ego_goals(scenario, _MapApi(), spacing=20.0)
    assert snapped == 0
    np.testing.assert_allclose(xy[:, 1], 50.0)
