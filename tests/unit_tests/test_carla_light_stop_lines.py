"""light_stop_line_overrides moves a bin light stop line along its lane onto the light's stop waypoint (the
junction entry RunningRedLightTest scores) and leaves lines within tolerance as exported."""

import math
import os
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import data_utils.mirror_map_bin as mbin
from pufferlib.ocean.cosim import carla_bridge as cb


TOWN_BIN = Path(__file__).resolve().parents[2] / "pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin"


def _line_mid(element):
    line = element["stop_line"]
    return 0.5 * (line[0] + line[3]), 0.5 * (line[1] + line[4])


def _light_with_waypoints(transform, elements, along_offsets_m):
    """One light whose stop waypoints sit along_offsets_m along the lane from each element's bin stop line."""
    waypoints = []
    for element, along_m in zip(elements, along_offsets_m):
        heading = element["heading"]
        mx, my = _line_mid(element)
        cx, cy = transform.bin_to_loc(mx + along_m * math.cos(heading), my + along_m * math.sin(heading))
        waypoints.append(
            {
                "x": cx,
                "y": cy,
                "yaw_deg": transform.bin_heading_to_yaw(heading),
                "lane_width": 3.5,
                "backward": [],
                "forward": [],
            }
        )
    return {
        "id": 1,
        "x": waypoints[0]["x"],
        "y": waypoints[0]["y"],
        "trigger_x": 0.0,
        "trigger_y": 0.0,
        "stop_waypoints": waypoints,
    }


def _setup():
    traffic = mbin.read_bin(TOWN_BIN)["traffic"]
    light_elements = [j for j, t in enumerate(traffic) if t["type"] == 1]
    transform = cb.CarlaTransform("Town01", offset=cb.town_offset(TOWN_BIN))
    return traffic, light_elements, transform


def test_line_moves_onto_waypoint_eight_metres_upstream():
    traffic, light_elements, transform = _setup()
    j = light_elements[0]
    light = _light_with_waypoints(transform, [traffic[j]], [-8.0])

    indices, lines = cb.light_stop_line_overrides([light], [[j]], transform, TOWN_BIN)

    assert indices.tolist() == [j]
    heading = traffic[j]["heading"]
    mx, my = _line_mid(traffic[j])
    moved_mid = 0.5 * (lines[0, 0:2] + lines[0, 3:5])
    assert np.allclose(moved_mid, [mx - 8.0 * math.cos(heading), my - 8.0 * math.sin(heading)], atol=1e-3)
    x1, y1, z1, x2, y2, z2 = traffic[j]["stop_line"]
    moved_length = float(np.hypot(lines[0, 3] - lines[0, 0], lines[0, 4] - lines[0, 1]))
    assert math.isclose(moved_length, math.hypot(x2 - x1, y2 - y1), abs_tol=1e-3)
    assert lines[0, 2] == np.float32(z1) and lines[0, 5] == np.float32(z2)


def test_line_within_tolerance_is_left_as_exported():
    traffic, light_elements, transform = _setup()
    j = light_elements[1]
    light = _light_with_waypoints(transform, [traffic[j]], [0.5])

    indices, lines = cb.light_stop_line_overrides([light], [[j]], transform, TOWN_BIN)

    assert len(indices) == 0 and lines.shape == (0, 6)


def test_waypoints_go_to_the_nearest_of_the_lights_elements():
    traffic, light_elements, transform = _setup()
    a, b = light_elements[2], light_elements[3]
    light = _light_with_waypoints(transform, [traffic[a], traffic[b]], [-8.0, 0.3])

    indices, lines = cb.light_stop_line_overrides([light], [[a, b]], transform, TOWN_BIN)

    assert indices.tolist() == [a]
    heading = traffic[a]["heading"]
    mx, my = _line_mid(traffic[a])
    moved_mid = 0.5 * (lines[0, 0:2] + lines[0, 3:5])
    assert np.allclose(moved_mid, [mx - 8.0 * math.cos(heading), my - 8.0 * math.sin(heading)], atol=1e-3)
