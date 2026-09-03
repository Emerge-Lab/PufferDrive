"""Static nuPlan clutter (cones, poles, barriers) reaches the shadow env only when it stands inside a lane
or lane-connector polygon; intersection corners and crosswalks (drivable area) do not count."""

from enum import Enum
from types import SimpleNamespace

import pytest

nuplan = pytest.importorskip("nuplan")  # noqa: F841  (partner_tracked_objects imports nuPlan datatypes)

from pufferlib.ocean.cosim import nuplan_bridge as nb  # noqa: E402


class FakeType(Enum):
    VEHICLE = 0
    PEDESTRIAN = 1
    BICYCLE = 2
    TRAFFIC_CONE = 3
    GENERIC_OBJECT = 6
    EGO = 7


def _obj(type_name, token, x=0.0):
    return SimpleNamespace(tracked_object_type=FakeType[type_name], track_token=token, center=SimpleNamespace(x=x, y=0.0))


class FakeMap:
    def __init__(self, lane_x, connector_x=()):
        self.lane_x = lane_x
        self.connector_x = set(connector_x)
        self.queries = 0

    def is_in_layer(self, point, layer):
        self.queries += 1
        assert layer.name == "LANE", f"unexpected layer {layer}"
        return point.x in self.lane_x

    def get_all_map_objects(self, point, layer):
        self.queries += 1
        assert layer.name == "LANE_CONNECTOR", f"unexpected layer {layer}"
        return ["connector"] if point.x in self.connector_x else []


def test_moving_agents_kept_static_off_road_dropped():
    objs = [
        _obj("VEHICLE", "v1", 50.0),
        _obj("PEDESTRIAN", "p1", 60.0),
        _obj("BICYCLE", "b1", 70.0),
        _obj("TRAFFIC_CONE", "c_on_lane", 1.0),
        _obj("TRAFFIC_CONE", "c_off", 2.0),
        _obj("GENERIC_OBJECT", "g_off", 3.0),
        _obj("GENERIC_OBJECT", "g_on_connector", 4.0),
        _obj("EGO", "ego", 0.0),
    ]
    map_api = FakeMap(lane_x={1.0}, connector_x={4.0})
    cache = {}
    kept = nb.partner_tracked_objects(objs, map_api, cache)
    assert [o.track_token for o in kept] == ["v1", "p1", "b1", "c_on_lane", "g_on_connector"]
    assert cache == {"c_on_lane": True, "c_off": False, "g_off": False, "g_on_connector": True}
    assert map_api.queries == 7  # lane hit: 1 query; connector hit or miss: 2


def test_static_verdict_is_cached_per_track():
    objs = [_obj("TRAFFIC_CONE", "c1", 1.0), _obj("GENERIC_OBJECT", "g1", 9.0)]
    map_api = FakeMap(lane_x={1.0})
    cache = {}
    nb.partner_tracked_objects(objs, map_api, cache)
    nb.partner_tracked_objects(objs, map_api, cache)
    assert map_api.queries == 3
    assert [o.track_token for o in nb.partner_tracked_objects(objs, map_api, cache)] == ["c1"]
