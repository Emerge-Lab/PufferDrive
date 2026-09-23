"""Lanelet2 binary validation.

- Check geometry, IDs, and directed lane references.
- Return a concise conversion summary.
"""

from __future__ import annotations

import math
from pathlib import Path


if __package__:
    from .lanelet2_conversion import LANE_SURFACE_STREET
    from .mirror_map_bin import read_bin
else:
    from lanelet2_conversion import LANE_SURFACE_STREET
    from mirror_map_bin import read_bin

GEOMETRY_FIELDS = ("x", "y", "z", "headings")


def validate_bin(path):
    """Validate one generated binary and return its map statistics."""

    data = read_bin(path)
    roads = data["roads"]
    lane_count = sum(road["type"] == LANE_SURFACE_STREET for road in roads)
    if lane_count == 0:
        raise ValueError("binary contains no drivable lanes")
    for index, road in enumerate(roads):
        if road["id"] != index:
            raise ValueError(f"road id {road['id']} does not match index {index}")
        if road["S"] < 2:
            raise ValueError(f"road {index} has fewer than two geometry points")
        if not all(math.isfinite(value) for key in GEOMETRY_FIELDS for value in road[key]):
            raise ValueError(f"road {index} contains a non-finite value")
        if road["type"] != LANE_SURFACE_STREET:
            continue
        for lane_id in road["entry_lanes"] + road["exit_lanes"]:
            if not 0 <= lane_id < lane_count:
                raise ValueError(f"lane {index} references non-lane road {lane_id}")
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "roads": len(roads),
        "lanes": lane_count,
        "directed_links": sum(len(road.get("exit_lanes", ())) for road in roads),
        "id_equals_index": True,
    }
