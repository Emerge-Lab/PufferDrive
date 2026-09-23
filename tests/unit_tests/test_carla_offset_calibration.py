"""calibrate_town_offset recovers a known bin-vs-CARLA translation from fake driving waypoints."""

import os
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import data_utils.mirror_map_bin as mbin
from pufferlib.ocean.cosim import carla_bridge as cb


TOWN_BIN = Path(__file__).resolve().parents[2] / "pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin"
TRUE_SHIFT = (-0.42, 0.31)  # bin lanes = CARLA lanes + shift, in the bin frame


class _Vec:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Transform:
    def __init__(self, x, y, heading_x, heading_y):
        self.location = _Vec(x, y)
        self._right = _Vec(heading_y, -heading_x)  # right of travel in a y-up frame

    def get_right_vector(self):
        return self._right


class _Waypoint:
    def __init__(self, x, y, heading_x, heading_y):
        self.transform = _Transform(x, y, heading_x, heading_y)
        self.is_junction = False
        self.lane_type = "Driving"


class _Map:
    """Waypoints on the bin's own lane centres, moved by -TRUE_SHIFT and mapped to the CARLA frame."""

    def __init__(self, transform):
        data = mbin.read_bin(TOWN_BIN)
        self.waypoints = []
        for road in data["roads"]:
            if not (0 <= road["type"] <= 9) or len(road["x"]) < 3:
                continue
            xs, ys = np.asarray(road["x"]), np.asarray(road["y"])
            for k in range(1, len(xs) - 1, 4):
                hx, hy = xs[k + 1] - xs[k - 1], ys[k + 1] - ys[k - 1]
                norm = float(np.hypot(hx, hy))
                if norm < 1e-6:
                    continue
                cx, cy = transform.bin_to_loc(xs[k] - TRUE_SHIFT[0], ys[k] - TRUE_SHIFT[1])
                self.waypoints.append(_Waypoint(cx, cy, hx / norm, -hy / norm))  # heading y flips with the frame

    def generate_waypoints(self, spacing_m):
        return self.waypoints


def test_calibration_recovers_translation():
    transform = cb.CarlaTransform("Town01", offset=cb.town_offset(str(TOWN_BIN)))
    offset, residual_before, residual_after = cb.calibrate_town_offset(_Map(transform), transform, str(TOWN_BIN))
    assert residual_before > 0.2
    assert residual_after < 0.02
    np.testing.assert_allclose((offset[0] - transform.tx, offset[1] - transform.ty), TRUE_SHIFT, atol=0.02)
