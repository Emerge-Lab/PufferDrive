"""roadblock_centroid_goals: route roadblock centroids become goals from the ego's block on, off-polygon
centroids snap to a lane baseline, goals behind the ego or too close together are dropped."""

from types import SimpleNamespace

import numpy as np
import pytest


nuplan = pytest.importorskip("nuplan")

from pufferlib.ocean.cosim import nuplan_bridge as nb


class _BlockMap:
    """Route blocks along x: A [0,40] (ego inside), B [40,60], C [60,70], D an L-bend whose centroid is off-polygon."""

    def __init__(self):
        from shapely.geometry import Polygon

        def block(polygon, lane_xy):
            path = [SimpleNamespace(x=x, y=y) for x, y in lane_xy]
            return SimpleNamespace(polygon=polygon, interior_edges=[SimpleNamespace(baseline_path=SimpleNamespace(discrete_path=path))])

        self.blocks = {
            "A": block(Polygon([(0, 0), (40, 0), (40, 4), (0, 4)]), [(0, 2), (40, 2)]),
            "B": block(Polygon([(40, 0), (60, 0), (60, 4), (40, 4)]), [(40, 2), (60, 2)]),
            "C": block(Polygon([(60, 0), (70, 0), (70, 4), (60, 4)]), [(60, 2), (70, 2)]),
            "D": block(Polygon([(70, 0), (110, 0), (110, 4), (74, 4), (74, 40), (70, 40)]), [(72, 2), (108, 2), (72, 20), (72, 38)]),
        }

    def get_map_object(self, block_id, layer):
        block = self.blocks.get(block_id)
        if block is not None:
            block.id = block_id
        return block


def test_roadblock_centroid_goals_skip_behind_thin_and_snap_off_polygon_centroids():
    map_api = _BlockMap()
    goals = nb.roadblock_centroid_goals(map_api, ["A", "B", "C", "D"], 30.0, 2.0, 0.0, min_spacing=20.0, min_ahead_m=15.0)
    centroid_d = map_api.blocks["D"].polygon.centroid
    assert not map_api.blocks["D"].polygon.contains(centroid_d)
    lane_d = np.array([(72, 2), (108, 2), (72, 20), (72, 38)], dtype=np.float64)
    snapped_d = lane_d[np.argmin(np.hypot(lane_d[:, 0] - centroid_d.x, lane_d[:, 1] - centroid_d.y))]
    # A's centroid (20, 2) is behind the ego, C's (65, 2) is 15 m past B's (50, 2): both dropped
    np.testing.assert_allclose(goals, [[50.0, 2.0], snapped_d])


def test_roadblock_centroid_goals_follow_the_route_through_turns():
    # ego at the end of A facing east; a route that turns back west after B keeps its goals (only the ego
    # block's own centroid is heading-gated), and the repeated block id of a loop is kept as a revisit
    map_api = _BlockMap()
    goals = nb.roadblock_centroid_goals(map_api, ["A", "B", "A", "B"], 30.0, 2.0, 0.0, min_spacing=10.0, min_ahead_m=15.0)
    np.testing.assert_allclose(goals, [[50.0, 2.0], [20.0, 2.0], [50.0, 2.0]])


def test_roadblock_centroid_goals_start_at_the_nearest_block_and_empty_without_blocks():
    goals = nb.roadblock_centroid_goals(_BlockMap(), ["A", "B", "C"], 45.0, 9.0, 0.0, min_spacing=10.0, min_ahead_m=2.0)
    np.testing.assert_allclose(goals, [[50.0, 2.0], [65.0, 2.0]])  # ego beside B: A is never visited
    assert nb.roadblock_centroid_goals(_BlockMap(), ["nope"], 0.0, 0.0, 0.0, min_spacing=20.0, min_ahead_m=15.0).shape == (0, 2)


def test_extend_route_past_loop_cut_appends_the_logged_remainder():
    assert nb.extend_route_past_loop_cut(["1", "2", "3"], ["1", "2", "3", "4", "2", "5"]) == ["1", "2", "3", "4", "2", "5"]
    assert nb.extend_route_past_loop_cut(["9", "1", "2"], ["1", "2", "3"]) == ["9", "1", "2", "3"]  # prepended start fix
    assert nb.extend_route_past_loop_cut(["1", "2"], ["1", "2"]) == ["1", "2"]
    assert nb.extend_route_past_loop_cut([], ["1"]) == []


class _LaneMap:
    """Two-lane road A -> connector B -> road C -> connector D -> road E; only lane 1 (y=6) flows into D."""

    def __init__(self):
        from shapely.geometry import Polygon

        self.blocks, self.lanes = {}, {}

        def add(block_id, x0, x1, lane_ys):
            lanes = []
            for k, y in enumerate(lane_ys):
                path = [SimpleNamespace(x=float(x), y=float(y)) for x in np.arange(x0, x1 + 1e-9, 2.0)]
                lane = SimpleNamespace(
                    id=f"{block_id}{k}",
                    baseline_path=SimpleNamespace(discrete_path=path),
                    outgoing_edges=[],
                    get_roadblock_id=lambda block_id=block_id: block_id,
                )
                lanes.append(lane)
                self.lanes[lane.id] = lane
            self.blocks[block_id] = SimpleNamespace(id=block_id, polygon=Polygon([(x0, 0), (x1, 0), (x1, 8), (x0, 8)]), interior_edges=lanes)

        add("A", 0, 40, [2, 6])
        add("B", 40, 60, [2, 6])
        add("C", 60, 100, [2, 6])
        add("D", 100, 120, [6])
        add("E", 120, 160, [6])
        for src, dst in [("A0", "B0"), ("A1", "B1"), ("B0", "C0"), ("B1", "C1"), ("C1", "D0"), ("D0", "E0")]:
            self.lanes[src].outgoing_edges.append(self.lanes[dst])

    def get_map_object(self, block_id, layer):
        return self.blocks.get(block_id)


def test_roadblock_lane_goals_keep_the_ego_lane_until_the_graph_forces_a_change():
    goals = nb.roadblock_lane_goals(_LaneMap(), list("ABCDE"), 10.0, 2.0, 0.0, min_spacing=20.0, min_ahead_m=5.0)
    # lane 0 through A and B; C's goal moves to lane 1 because only lane 1 flows into D
    np.testing.assert_allclose(goals, [[20, 2], [50, 2], [80, 6], [110, 6], [140, 6]])


def test_roadblock_lane_goals_seed_from_the_co_directional_lane_nearest_the_ego():
    goals = nb.roadblock_lane_goals(_LaneMap(), list("ABCDE"), 10.0, 5.0, 0.0, min_spacing=20.0, min_ahead_m=5.0)
    np.testing.assert_allclose(goals, [[20, 6], [50, 6], [80, 6], [110, 6], [140, 6]])
    # facing west no lane is co-directional: nearest lane, own-block goal behind the ego dropped
    goals = nb.roadblock_lane_goals(_LaneMap(), list("ABCDE"), 10.0, 5.0, np.pi, min_spacing=20.0, min_ahead_m=5.0)
    np.testing.assert_allclose(goals, [[50, 6], [80, 6], [110, 6], [140, 6]])
    assert nb.roadblock_lane_goals(_LaneMap(), ["nope"], 0.0, 0.0, 0.0, min_spacing=20.0, min_ahead_m=5.0).shape == (0, 2)
