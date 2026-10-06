"""Lanelet2 OSM conversion.

- Parse supported lanelets and boundaries.
- Build PufferDrive roads and directed lane links.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Tuple


if __package__:
    from .lanelet2_geometry import (
        arc_lengths,
        centerline,
        clip_polyline,
        detect_utm_epsg,
        headings,
        normalise_boundaries,
        project_epsg_array,
    )
else:
    from lanelet2_geometry import (
        arc_lengths,
        centerline,
        clip_polyline,
        detect_utm_epsg,
        headings,
        normalise_boundaries,
        project_epsg_array,
    )

LANE_SURFACE_STREET = 2
ROAD_LINE_SOLID_SINGLE_WHITE = 12
ROAD_EDGE_BOUNDARY = 21
DRIVABLE_SUBTYPES = frozenset({"road", "highway"})
BOUNDARY_TYPES = frozenset({"line_thin", "line_thick", "road_border", "virtual", "guard_rail", "fence", "wall"})
EDGE_BOUNDARY_TYPES = frozenset({"road_border", "guard_rail", "fence", "wall"})


@dataclass(frozen=True)
class ConversionConfig:
    """Store conversion options in metre-based units."""

    crop: Optional[Tuple[float, float, float, float]] = None
    link_tolerance_m: float = 0.0
    virtual_as_edge: bool = False


@dataclass(frozen=True)
class ProjectionInfo:
    """Record the detected CRS and local projected origin."""

    epsg: int
    origin_easting: float
    origin_northing: float


def _tags(element):
    """Return an OSM element's tags as a key-value mapping."""

    return {tag.get("k", ""): tag.get("v", "") for tag in element.findall("tag")}


def _speed_limit_mps(tags):
    """Normalize a Lanelet2 speed limit to metres per second."""

    raw = tags.get("speed_limit") or tags.get("maxspeed")
    if not raw:
        return 50.0 * 1_000.0 / 3_600.0
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if match is None:
        return 50.0 * 1_000.0 / 3_600.0
    value = float(match.group())
    lowered = raw.lower()
    if "mph" in lowered:
        return value * 1_609.344 / 3_600.0
    if "m/s" in lowered or "mps" in lowered:
        return value
    return value * 1_000.0 / 3_600.0


def _lane_record(points, speed_limit_mps):
    """Build one drivable PufferDrive lane record."""

    cumulative = arc_lengths(points)
    return {
        "id": -1,
        "type": LANE_SURFACE_STREET,
        "S": len(points),
        "x": tuple(point[0] for point in points),
        "y": tuple(point[1] for point in points),
        "z": (0.0,) * len(points),
        "headings": tuple(headings(points)),
        "entry_lanes": [],
        "exit_lanes": [],
        "speed_limit": speed_limit_mps,
        "length": cumulative[-1],
        "cum_lengths": tuple(cumulative),
    }


def _line_record(points, road_type):
    """Build one PufferDrive line or edge record."""

    return {
        "id": -1,
        "type": road_type,
        "S": len(points),
        "x": tuple(point[0] for point in points),
        "y": tuple(point[1] for point in points),
        "z": (0.0,) * len(points),
        "headings": tuple(headings(points)),
    }


def _project_nodes(root):
    """Project all OSM nodes and shift them to a local metre origin."""

    nodes = root.findall("node")
    latitudes = [float(node.get("lat")) for node in nodes]
    longitudes = [float(node.get("lon")) for node in nodes]
    epsg = detect_utm_epsg(latitudes, longitudes)
    eastings, northings = project_epsg_array(latitudes, longitudes, epsg)
    origin_easting = min(eastings)
    origin_northing = min(northings)
    node_xy = {
        int(node.get("id")): (easting - origin_easting, northing - origin_northing)
        for node, easting, northing in zip(nodes, eastings, northings)
    }
    return node_xy, ProjectionInfo(epsg, origin_easting, origin_northing)


def _way_nodes(root):
    """Map each OSM way ID to its ordered node IDs."""

    return {int(way.get("id")): [int(item.get("ref")) for item in way.findall("nd")] for way in root.findall("way")}


def _lanelets(root, node_xy, way_nodes, config):
    """Extract supported drivable lanelets and clipped centerlines."""

    lanelets = []
    for relation in root.findall("relation"):
        tags = _tags(relation)
        if tags.get("type") != "lanelet" or tags.get("subtype", "road") not in DRIVABLE_SUBTYPES:
            continue
        members = {
            item.get("role"): int(item.get("ref")) for item in relation.findall("member") if item.get("type") == "way"
        }
        if "left" not in members or "right" not in members:
            continue
        boundaries = normalise_boundaries(
            way_nodes.get(members["left"], ()),
            way_nodes.get(members["right"], ()),
            node_xy,
        )
        points = clip_polyline(centerline(boundaries, node_xy), config.crop)
        if len(points) < 2:
            continue
        lanelets.append(
            {
                "relation_id": int(relation.get("id")),
                "boundaries": boundaries,
                "points": points,
                "speed_limit_mps": _speed_limit_mps(tags),
            }
        )
    return lanelets


def _connect_shared_endpoints(lanelets, roads):
    """Create directed links between lanelets with shared endpoints."""

    relation_index = {item["relation_id"]: index for index, item in enumerate(lanelets)}
    first_left = {}
    first_right = {}
    for item in lanelets:
        left_ids, right_ids = item["boundaries"]
        if left_ids:
            first_left.setdefault(left_ids[0], []).append(item["relation_id"])
        if right_ids:
            first_right.setdefault(right_ids[0], []).append(item["relation_id"])
    for index, item in enumerate(lanelets):
        left_ids, right_ids = item["boundaries"]
        successors = set(first_left.get(left_ids[-1], ())) if left_ids else set()
        if right_ids:
            successors.update(first_right.get(right_ids[-1], ()))
        successors.discard(item["relation_id"])
        roads[index]["exit_lanes"] = sorted(
            relation_index[item_id] for item_id in successors if item_id in relation_index
        )
    for index, road in enumerate(roads):
        for successor in road["exit_lanes"]:
            roads[successor]["entry_lanes"].append(index)


def _connect_nearby_endpoints(roads, tolerance_m):
    """Connect otherwise unlinked lane endpoints within a distance limit."""

    if tolerance_m <= 0.0:
        return
    tolerance_sq = tolerance_m * tolerance_m
    for index, road in enumerate(roads):
        if road["exit_lanes"]:
            continue
        end_x, end_y = road["x"][-1], road["y"][-1]
        candidates = []
        for successor, candidate in enumerate(roads):
            if successor == index:
                continue
            distance_sq = (candidate["x"][0] - end_x) ** 2 + (candidate["y"][0] - end_y) ** 2
            if distance_sq <= tolerance_sq:
                candidates.append((distance_sq, successor))
        road["exit_lanes"] = [successor for _, successor in sorted(candidates)]
        for successor in road["exit_lanes"]:
            if index not in roads[successor]["entry_lanes"]:
                roads[successor]["entry_lanes"].append(index)


def _append_boundaries(root, roads, node_xy, way_nodes, config):
    """Append supported Lanelet2 boundaries as line or edge records."""

    for way in root.findall("way"):
        way_type = _tags(way).get("type", "")
        if way_type not in BOUNDARY_TYPES:
            continue
        points = clip_polyline(
            [node_xy[node_id] for node_id in way_nodes[int(way.get("id"))] if node_id in node_xy],
            config.crop,
        )
        if len(points) < 2:
            continue
        is_edge = way_type in EDGE_BOUNDARY_TYPES or (way_type == "virtual" and config.virtual_as_edge)
        roads.append(_line_record(points, ROAD_EDGE_BOUNDARY if is_edge else ROAD_LINE_SOLID_SINGLE_WHITE))


def _map_record(roads):
    """Wrap converted roads in the PufferDrive map schema."""

    source_name = b"lanelet2"
    return {
        "agents": [],
        "roads": roads,
        "traffic": [],
        "objects": [],
        "lane_graph": {"n": 0, "lane_ids": (), "distances": ()},
        "scenario_id": source_name.ljust(128, b"\0"),
        "dataset_name": source_name.ljust(32, b"\0"),
        "log_length": 0,
        "log_dt": 0.0,
        "objects_of_interest": (),
        "tracks_to_predict": (),
    }


def convert_map(
    input_path,
    crop=None,
    link_tolerance_m=0.0,
    virtual_as_edge=False,
):
    """Convert a Lanelet2 OSM file and return map and projection data."""

    config = ConversionConfig(
        crop=tuple(crop) if crop is not None else None,
        link_tolerance_m=link_tolerance_m,
        virtual_as_edge=virtual_as_edge,
    )
    root = ET.parse(input_path).getroot()
    node_xy, projection = _project_nodes(root)
    way_nodes = _way_nodes(root)
    lanelets = _lanelets(root, node_xy, way_nodes, config)
    if not lanelets:
        raise ValueError("no drivable Lanelet2 relations were found in the selected area")
    roads = [_lane_record(item["points"], item["speed_limit_mps"]) for item in lanelets]
    _connect_shared_endpoints(lanelets, roads)
    _connect_nearby_endpoints(roads, config.link_tolerance_m)
    _append_boundaries(root, roads, node_xy, way_nodes, config)
    for index, road in enumerate(roads):
        road["id"] = index
        if road["type"] == LANE_SURFACE_STREET:
            road["entry_lanes"] = sorted(set(road["entry_lanes"]))
    return _map_record(roads), projection


def build_map(
    input_path,
    crop=None,
    link_tolerance_m=0.0,
    virtual_as_edge=False,
):
    """Convert a Lanelet2 OSM file and return only the map data."""

    map_data, _ = convert_map(
        input_path,
        crop=crop,
        link_tolerance_m=link_tolerance_m,
        virtual_as_edge=virtual_as_edge,
    )
    return map_data
