"""symmetric_lane_widths: bin lane widths become twice the distance to the nearer nuPlan boundary."""

import numpy as np
import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import LineString  # noqa: E402

from pufferlib.ocean.cosim import nuplan_bridge as nb  # noqa: E402


class _Polyline:
    def __init__(self, coords):
        self.linestring = LineString(coords)


class _MapLane:
    def __init__(self, baseline, left, right):
        self.baseline_path = _Polyline(baseline)
        self.left_boundary = _Polyline(left)
        self.right_boundary = _Polyline(right)


def _lane_road(tf, utm_points, widths):
    xs, ys = zip(*(tf.loc_to_bin(x, y) for x, y in utm_points))
    return {"type": 2, "S": len(utm_points), "x": tuple(xs), "y": tuple(ys), "widths": tuple(widths)}


def test_parking_strip_lane_shrinks_to_the_narrow_side_and_centered_lane_keeps_its_width():
    tf = nb.NuPlanTransform(1000.0, 2000.0)
    parking = _MapLane([(0, 0), (20, 0)], [(0, 1.7), (20, 1.7)], [(0, -4.3), (20, -4.3)])
    centered = _MapLane([(0, 10), (20, 10)], [(0, 11.5), (20, 11.5)], [(0, 8.5), (20, 8.5)])
    roads = [
        _lane_road(tf, [(0, 0), (10, 0), (20, 0)], [6.0, 6.0, 6.0]),
        _lane_road(tf, [(0, 10), (10, 10), (20, 10)], [3.0, 3.0, 3.0]),
        {"type": 15, "S": 2, "x": (0.0, 1.0), "y": (0.0, 0.0)},
    ]
    patched, unmatched = nb.symmetric_lane_widths(roads, tf, [parking, centered])
    assert (patched, unmatched) == (2, 0)
    np.testing.assert_allclose(roads[0]["widths"], [3.4, 3.4, 3.4])
    np.testing.assert_allclose(roads[1]["widths"], [3.0, 3.0, 3.0])
    assert "widths" not in roads[2]


def test_lane_without_a_nearby_baseline_keeps_its_widths():
    tf = nb.NuPlanTransform(0.0, 0.0)
    lane = _MapLane([(0, 0), (20, 0)], [(0, 1.7), (20, 1.7)], [(0, -4.3), (20, -4.3)])
    roads = [_lane_road(tf, [(0, 50), (20, 50)], [5.5, 5.5]), _lane_road(tf, [(0, 1.0), (20, 1.0)], [6.0, 6.0])]
    patched, unmatched = nb.symmetric_lane_widths(roads, tf, [lane])
    assert (patched, unmatched) == (0, 2)  # 50 m away: no candidate; 1 m off the baseline: beyond the match tolerance
    assert roads[0]["widths"] == (5.5, 5.5) and roads[1]["widths"] == (6.0, 6.0)


def test_width_is_floored_where_the_baseline_touches_a_boundary():
    tf = nb.NuPlanTransform(0.0, 0.0)
    tapered = _MapLane([(0, 0), (20, 0)], [(0, 0.1), (10, 0.1), (10, 1.7), (20, 1.7)], [(0, -3.0), (20, -3.0)])
    roads = [_lane_road(tf, [(0, 0), (20, 0)], [4.0, 4.0])]
    nb.symmetric_lane_widths(roads, tf, [tapered])
    assert roads[0]["widths"][0] == nb.SYMMETRIC_WIDTH_MIN_M
    assert roads[0]["widths"][1] == pytest.approx(3.4)
