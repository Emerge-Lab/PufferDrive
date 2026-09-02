"""Static nuPlan clutter (cones, poles, barriers) reaches the shadow env only when it stands on the drivable area."""

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
    def __init__(self, drivable_x):
        self.drivable_x = drivable_x
        self.queries = 0

    def is_in_layer(self, point, layer):
        self.queries += 1
        return point.x in self.drivable_x


def test_moving_agents_kept_static_off_road_dropped():
    objs = [
        _obj("VEHICLE", "v1", 50.0),
        _obj("PEDESTRIAN", "p1", 60.0),
        _obj("BICYCLE", "b1", 70.0),
        _obj("TRAFFIC_CONE", "c_on", 1.0),
        _obj("TRAFFIC_CONE", "c_off", 2.0),
        _obj("GENERIC_OBJECT", "g_off", 3.0),
        _obj("EGO", "ego", 0.0),
    ]
    map_api = FakeMap(drivable_x={1.0})
    cache = {}
    kept = nb.partner_tracked_objects(objs, map_api, cache)
    assert [o.track_token for o in kept] == ["v1", "p1", "b1", "c_on"]
    assert cache == {"c_on": True, "c_off": False, "g_off": False}
    assert map_api.queries == 3


def test_static_verdict_is_cached_per_track():
    objs = [_obj("TRAFFIC_CONE", "c1", 1.0), _obj("GENERIC_OBJECT", "g1", 9.0)]
    map_api = FakeMap(drivable_x={1.0})
    cache = {}
    nb.partner_tracked_objects(objs, map_api, cache)
    nb.partner_tracked_objects(objs, map_api, cache)
    assert map_api.queries == 2
    assert [o.track_token for o in nb.partner_tracked_objects(objs, map_api, cache)] == ["c1"]
