"""match_connectors_to_stop_lines: nearest co-directional traffic light by stop-line segment distance."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

nuplan = pytest.importorskip("nuplan")

from pufferlib.ocean.cosim import nuplan_bridge as nb

SOUTH = -np.pi / 2
IDENTITY = nb.NuPlanTransform(0.0, 0.0)
# element 0: 25 m stop line across a seven-lane southbound approach (y = -1), midpoint at x = 12.5
# element 1: 6 m oncoming stop line 8 m away, element 2: cross-street light 5 m away, element 3: stop sign on top of the entry
STOP_LINES = np.array([[0.0, -1.0, 25.0, -1.0], [-5.0, 8.0, 1.0, 8.0], [-5.0, -3.0, -5.0, 3.0], [-1.0, 0.0, 3.0, 0.0]])
HEADINGS = np.array([SOUTH, np.pi / 2, 0.0, SOUTH])
TYPES = np.array([nb.TRAFFIC_TYPE_LIGHT, nb.TRAFFIC_TYPE_LIGHT, nb.TRAFFIC_TYPE_LIGHT, 2], np.int32)


def match(entries, **kwargs):
    return nb.match_connectors_to_stop_lines(entries, IDENTITY, STOP_LINES, HEADINGS, TYPES, **kwargs)


def test_outer_lane_matches_the_wide_stop_line_by_segment_distance():
    assert match({"outer": (1.5, 0.0, SOUTH), "middle": (12.5, 0.0, SOUTH)}) == {"outer": 0, "middle": 0}


def test_oncoming_cross_street_and_stop_sign_elements_are_never_candidates():
    assert match({"c": (-2.0, 4.0, SOUTH)}) == {"c": 0}  # element 1 (oncoming) and 2 (cross) are nearer
    assert match({"far": (-2.0, 40.0, SOUTH)}) == {}


def test_skewed_stop_line_is_kept_within_the_heading_tolerance():
    assert match({"c": (1.5, 0.0, SOUTH - np.radians(60.0))}) == {"c": 0}
    assert match({"c": (1.5, 0.0, SOUTH - np.radians(80.0))}) == {}


def test_max_distance_and_empty_map():
    assert match({"c": (1.5, 14.0, SOUTH)}) == {"c": 0}
    assert match({"c": (1.5, 17.0, SOUTH)}) == {}
    assert match({"c": (1.5, 17.0, SOUTH)}, max_dist_m=19.0) == {"c": 0}
    assert nb.match_connectors_to_stop_lines({"c": (0.0, 0.0, 0.0)}, IDENTITY, STOP_LINES[:0], HEADINGS[:0], TYPES[:0]) == {}


def test_point_to_segment_distance_clamps_to_endpoints():
    seg = np.array([[0.0, 0.0, 10.0, 0.0]])
    np.testing.assert_allclose(nb.point_to_segment_distance(5.0, 3.0, seg), [3.0])
    np.testing.assert_allclose(nb.point_to_segment_distance(-4.0, 3.0, seg), [5.0])
    np.testing.assert_allclose(nb.point_to_segment_distance(14.0, 3.0, seg), [5.0])
