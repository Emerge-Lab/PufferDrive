"""Lanelet2 projection and geometry.

- Detect and apply a local UTM projection.
- Build, resample, orient, and crop road geometry.
"""

from __future__ import annotations

import math
from functools import lru_cache


WGS84_EPSG = 4326
GEOMETRY_EPSILON_M = 1e-9
INTERPOLATION_EPSILON_M = 1e-12


def detect_utm_epsg(latitudes, longitudes):
    """Detect the WGS 84 UTM CRS covering the map center."""

    if len(latitudes) == 0 or len(longitudes) == 0 or len(latitudes) != len(longitudes):
        raise ValueError("at least one latitude/longitude pair is required")
    try:
        from pyproj.aoi import AreaOfInterest
        from pyproj.database import query_utm_crs_info
    except ImportError as exc:
        raise RuntimeError("Lanelet2 conversion requires pyproj>=3.0") from exc
    latitude = (min(latitudes) + max(latitudes)) * 0.5
    longitude = (min(longitudes) + max(longitudes)) * 0.5
    matches = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(longitude, latitude, longitude, latitude),
        contains=True,
    )
    if not matches:
        raise ValueError("no WGS 84 UTM CRS covers the Lanelet2 map centre")
    return int(matches[0].code)


@lru_cache(maxsize=None)
def _epsg_transformer(epsg):
    """Create and cache a WGS 84 to projected CRS transformer."""

    try:
        from pyproj import CRS, Transformer
    except ImportError as exc:
        raise RuntimeError("Lanelet2 conversion requires pyproj>=3.0") from exc
    return Transformer.from_crs(CRS.from_epsg(WGS84_EPSG), CRS.from_epsg(epsg), always_xy=True)


def project_epsg_array(latitudes, longitudes, epsg):
    """Project latitude and longitude arrays to metre coordinates."""

    eastings, northings = _epsg_transformer(epsg).transform(longitudes, latitudes)
    return [float(value) for value in eastings], [float(value) for value in northings]


def arc_lengths(points):
    """Return cumulative distances along a polyline."""

    cumulative = [0.0]
    for start, end in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(start, end))
    return cumulative


def resample(points, count):
    """Resample a polyline to an evenly spaced point count."""

    if len(points) == count:
        return list(points)
    if len(points) < 2:
        return list(points) * count
    cumulative = arc_lengths(points)
    if cumulative[-1] <= GEOMETRY_EPSILON_M:
        return [points[0]] * count
    result = []
    segment = 0
    for index in range(count):
        target = cumulative[-1] * index / (count - 1)
        while segment + 1 < len(cumulative) - 1 and cumulative[segment + 1] < target:
            segment += 1
        span = cumulative[segment + 1] - cumulative[segment]
        fraction = (target - cumulative[segment]) / span if span > INTERPOLATION_EPSILON_M else 0.0
        start, end = points[segment], points[segment + 1]
        result.append((start[0] + fraction * (end[0] - start[0]), start[1] + fraction * (end[1] - start[1])))
    return result


def normalise_boundaries(left_ids, right_ids, node_xy):
    """Orient lane boundaries consistently with lane travel direction."""

    left_ids = list(left_ids)
    right_ids = list(right_ids)
    left = [node_xy[node_id] for node_id in left_ids if node_id in node_xy]
    right = [node_xy[node_id] for node_id in right_ids if node_id in node_xy]
    if len(left) < 2 or len(right) < 2:
        return left_ids, right_ids
    left_vector = (left[-1][0] - left[0][0], left[-1][1] - left[0][1])
    right_vector = (right[-1][0] - right[0][0], right[-1][1] - right[0][1])
    if left_vector[0] * right_vector[0] + left_vector[1] * right_vector[1] < 0.0:
        left_ids.reverse()
        left.reverse()
    side = (left[0][0] - right[0][0], left[0][1] - right[0][1])
    if right_vector[0] * side[1] - right_vector[1] * side[0] < 0.0:
        left_ids.reverse()
        right_ids.reverse()
    return left_ids, right_ids


def centerline(boundaries, node_xy):
    """Build a centerline midway between normalized lane boundaries."""

    left_ids, right_ids = boundaries
    left = [node_xy[node_id] for node_id in left_ids if node_id in node_xy]
    right = [node_xy[node_id] for node_id in right_ids if node_id in node_xy]
    if not left or not right:
        return left or right
    count = max(len(left), len(right))
    left = resample(left, count)
    right = resample(right, count)
    return [((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5) for a, b in zip(left, right)]


def headings(points):
    """Calculate the heading at each polyline point in radians."""

    if len(points) == 1:
        return [0.0]
    values = [math.atan2(end[1] - start[1], end[0] - start[0]) for start, end in zip(points, points[1:])]
    return values + [values[-1]]


def clip_polyline(points, crop):
    """Clip a polyline and keep its longest valid fragment."""

    if crop is None:
        return list(points)
    try:
        from shapely.geometry import LineString, box
    except ImportError as exc:
        raise RuntimeError("Lanelet2 conversion requires shapely>=2.0") from exc
    geometry = LineString(points).intersection(box(*crop))
    fragments = [geometry] if geometry.geom_type == "LineString" else list(getattr(geometry, "geoms", ()))
    fragments = [
        fragment
        for fragment in fragments
        if fragment.geom_type == "LineString" and fragment.length > GEOMETRY_EPSILON_M
    ]
    if not fragments:
        return []
    return [(float(x), float(y)) for x, y in max(fragments, key=lambda fragment: fragment.length).coords]
