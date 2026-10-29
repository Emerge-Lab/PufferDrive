"""Static nuPlan clutter (cones, poles, barriers) reaches the shadow env only when its box overlaps a lane
or lane-connector polygon; intersection corners and crosswalks (drivable area) do not count."""

from enum import Enum
from types import SimpleNamespace

import pytest

nuplan = pytest.importorskip("nuplan")  # noqa: F841  (partner_tracked_objects imports nuPlan datatypes)
from shapely.geometry import LineString, box  # noqa: E402

from pufferlib.ocean.cosim import nuplan_bridge as nb  # noqa: E402


class FakeType(Enum):
    VEHICLE = 0
    PEDESTRIAN = 1
    BICYCLE = 2
    TRAFFIC_CONE = 3
    BARRIER = 4
    GENERIC_OBJECT = 6
    EGO = 7


def _obj(type_name, token, x=0.0, y=0.0, length=0.4, width=0.4):
    """Axis-aligned box: length along x, width along y."""
    geometry = box(x - length / 2, y - width / 2, x + length / 2, y + width / 2)
    return SimpleNamespace(
        tracked_object_type=FakeType[type_name],
        track_token=token,
        center=SimpleNamespace(x=x, y=y),
        box=SimpleNamespace(length=length, width=width, geometry=geometry),
    )


class ConnectorWithoutPolygon:
    """Devkit lane connector whose scaled-width polygon row is missing: .polygon asserts."""

    def __init__(self, centerline):
        self.centerline = centerline

    @property
    def polygon(self):
        raise AssertionError("Could not find the desired key")


class FakeMap:
    """Devkit proximity semantics: the square query patch is matched against lane polygons but against lane
    connector centerlines."""

    def __init__(self, lane_polygons, connectors=()):
        self.lanes = [SimpleNamespace(polygon=p) for p in lane_polygons]
        self.connectors = [
            c if isinstance(c, ConnectorWithoutPolygon) else SimpleNamespace(polygon=c[0], centerline=c[1])
            for c in connectors
        ]
        self.queries = 0

    def get_proximal_map_objects(self, point, radius, layers):
        self.queries += 1
        patch = box(point.x - radius, point.y - radius, point.x + radius, point.y + radius)
        result = {}
        for layer in layers:
            if layer.name == "LANE":
                result[layer] = [lane for lane in self.lanes if lane.polygon.intersects(patch)]
            else:
                result[layer] = [c for c in self.connectors if c.centerline.intersects(patch)]
        return result


LANE = box(-10.0, -2.0, 0.0, 2.0)  # right lane edge at x = 0
CONNECTOR = box(19.0, -1.0, 21.0, 1.0)
CONNECTOR_CENTERLINE = LineString([(19.0, 0.0), (21.0, 0.0)])


def test_moving_agents_kept_static_off_road_dropped():
    objs = [
        _obj("VEHICLE", "v1", 50.0),
        _obj("PEDESTRIAN", "p1", 60.0),
        _obj("BICYCLE", "b1", 70.0),
        _obj("TRAFFIC_CONE", "c_on_lane", -5.0),
        _obj("GENERIC_OBJECT", "g_protruding", 1.0, length=2.3, width=1.7),  # center 1 m off the lane, box reaches in
        _obj("TRAFFIC_CONE", "c_off", 1.0),
        _obj("BARRIER", "bar_end_reaches", 1.9, length=4.0, width=0.5),  # only the far end overlaps the lane
        _obj("GENERIC_OBJECT", "g_on_connector", 20.0),
        _obj("EGO", "ego", 0.0),
    ]
    map_api = FakeMap([LANE], [(CONNECTOR, CONNECTOR_CENTERLINE)])
    cache = {}
    kept = nb.partner_tracked_objects(objs, map_api, cache)
    assert [o.track_token for o in kept] == [
        "v1",
        "p1",
        "b1",
        "c_on_lane",
        "g_protruding",
        "bar_end_reaches",
        "g_on_connector",
    ]
    assert cache == {
        "c_on_lane": True,
        "g_protruding": True,
        "c_off": False,
        "bar_end_reaches": True,
        "g_on_connector": True,
    }
    assert map_api.queries == 5


def test_static_verdict_is_cached_per_track():
    objs = [_obj("TRAFFIC_CONE", "c1", -1.0), _obj("GENERIC_OBJECT", "g1", 9.0)]
    map_api = FakeMap([LANE])
    cache = {}
    nb.partner_tracked_objects(objs, map_api, cache)
    nb.partner_tracked_objects(objs, map_api, cache)
    assert map_api.queries == 2
    assert [o.track_token for o in nb.partner_tracked_objects(objs, map_api, cache)] == ["c1"]


def test_static_inside_a_connector_away_from_its_centerline_is_kept():
    """A 0.3 m patch around a cone 0.8 m off the centerline never reached the connector (run 390 regression)."""
    objs = [_obj("TRAFFIC_CONE", "c_edge", 20.0, 0.8), _obj("TRAFFIC_CONE", "c_outside", 20.0, 1.5)]
    kept = nb.partner_tracked_objects(objs, FakeMap([], [(CONNECTOR, CONNECTOR_CENTERLINE)]), {})
    assert [o.track_token for o in kept] == ["c_edge"]


def test_connector_without_polygon_row_is_skipped():
    objs = [_obj("TRAFFIC_CONE", "c1", 30.0)]
    map_api = FakeMap([], [ConnectorWithoutPolygon(LineString([(29.0, 0.0), (31.0, 0.0)]))])
    assert nb.partner_tracked_objects(objs, map_api, {}) == []
