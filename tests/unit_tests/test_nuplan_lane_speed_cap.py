"""lane_speed_cap_mps: nuPlan lane limit resolved like the speed_limit_compliance metric, capped only in slow zones."""

from types import SimpleNamespace

import pytest

nuplan = pytest.importorskip("nuplan")

from pufferlib.ocean.cosim import nuplan_bridge as nb  # noqa: E402


class _Map:
    def __init__(self, lane_limit=None, connector_limits=()):
        self.lane = None if lane_limit == "none" else SimpleNamespace(speed_limit_mps=lane_limit)
        self.connector_limits = connector_limits

    def get_one_map_object(self, point, layer):
        return self.lane

    def get_all_map_objects(self, point, layer):
        edges = [SimpleNamespace(speed_limit_mps=v) for v in self.connector_limits]
        return [SimpleNamespace(outgoing_edges=edges[:1], incoming_edges=edges[1:])] if edges else []


def test_slow_lane_is_capped_at_limit_plus_margin():
    assert nb.lane_speed_cap_mps(_Map(lane_limit=6.7), 0, 0, 8.33, 0.5) == pytest.approx(7.2)


def test_fast_lane_and_unknown_limit_are_uncapped():
    assert nb.lane_speed_cap_mps(_Map(lane_limit=13.4), 0, 0, 8.33, 0.0) == 0.0
    assert nb.lane_speed_cap_mps(_Map(lane_limit=None), 0, 0, 8.33, 0.0) == 0.0
    assert nb.lane_speed_cap_mps(_Map(lane_limit="none"), 0, 0, 8.33, 0.0) == 0.0  # off the lane graph


def test_connector_uses_the_max_adjoining_lane_limit():
    assert nb.lane_speed_cap_mps(_Map(lane_limit="none", connector_limits=(6.0, 7.0)), 0, 0, 8.33, 0.0) == pytest.approx(7.0)
    assert nb.lane_speed_cap_mps(_Map(lane_limit="none", connector_limits=(6.0, 11.0)), 0, 0, 8.33, 0.0) == 0.0
    assert nb.lane_speed_cap_mps(_Map(lane_limit="none", connector_limits=(6.0, None)), 0, 0, 8.33, 0.0) == 0.0
