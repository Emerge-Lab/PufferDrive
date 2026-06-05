"""CARLA <-> PufferDrive coordinate transform + state readback for co-simulation.

PufferDrive's `carla*/opendrive__Town*.bin` maps are OpenDRIVE exports of the
CARLA towns, stored in a y-flipped, per-town-translated frame. Empirically
(road-network alignment + heading match to 1e-3 rad):

    bin_x       =  carla_x + tx
    bin_y       = -carla_y + ty
    bin_heading = -radians(carla_yaw)          # CARLA yaw is degrees, left-handed
    bin_vx      =  carla_vx
    bin_vy      = -carla_vy

`(tx, ty)` is the town's georeference offset; it's recovered at runtime by ICP-
aligning the loaded CARLA map's road points to the bin's lane points (so we never
depend on stale constants). Precomputed values are cached as a fallback.

The PufferDrive env additionally subtracts its own `world_mean` inside
`set_agent_states`/`set_agent_goals`, so the values produced here are in the
*bin* frame (== sim frame + world_mean), matching `get_global_agent_state`.
"""

import math
from pathlib import Path

import numpy as np

import data_utils.mirror_map_bin as _mbin

# Fallback per-town offsets (ICP-computed; runtime recompute preferred).
TOWN_OFFSETS = {
    "Town01": (-204.35, 151.29),
    "Town02": (-93.72, 209.93),
    "Town03": (-43.99, -4.55),
    "Town04": (-5.46, -3.46),
    "Town05": (46.99, 2.95),
    "Town10HD": (8.09, 35.25),
}

_BIN_DIR = Path(__file__).resolve().parents[2] / "resources" / "drive" / "binaries" / "carla"


def bin_path_for_town(town: str) -> str:
    """Path to the OpenDRIVE bin for a CARLA town name (e.g. 'Town01')."""
    return str(_BIN_DIR / f"opendrive__{town}.bin")


def _bin_lane_points(bin_path: str) -> np.ndarray:
    """All lane-polyline (x, y) points from a bin (type 0..9 == lanes)."""
    data = _mbin.read_bin(Path(bin_path))
    pts = []
    for road in data["roads"]:
        if 0 <= road["type"] <= 9:
            pts.extend(zip(road["x"], road["y"]))
    return np.asarray(pts, dtype=np.float64)


def compute_town_offset(carla_map, bin_path: str, sample_m: float = 2.0, iters: int = 10):
    """Recover (tx, ty) by ICP-aligning CARLA road waypoints to the bin lanes,
    with the y-flip reflection fixed. Returns (tx, ty)."""
    wps = carla_map.generate_waypoints(sample_m)
    C = np.array([[w.transform.location.x, w.transform.location.y] for w in wps], dtype=np.float64)
    B = _bin_lane_points(bin_path)
    TC = C * np.array([1.0, -1.0])  # y-flip
    t = B.mean(0) - TC.mean(0)
    for _ in range(iters):
        P = (TC + t)[::2]
        idx = ((P[:, None, 0] - B[None, :, 0]) ** 2 + (P[:, None, 1] - B[None, :, 1]) ** 2).argmin(1)
        t = t + (B[idx] - P).mean(0)
    return float(t[0]), float(t[1])


def town_offset(carla_map, bin_path: str):
    """Exact CARLA<->bin (tx, ty) from the bin's stored centroid when present
    (bin = original - centroid, and CARLA = original y-flipped, so the offset is
    just -centroid_xy). Falls back to ICP alignment for legacy bins that don't
    carry a centroid. The stored-centroid path is exact and deterministic; ICP is
    biased because it aligns the bin's full lane set to CARLA's driving-only
    waypoints, so prefer the stored value."""
    centroid = _mbin.read_bin(Path(bin_path)).get("centroid")
    if centroid is not None:
        return (-float(centroid[0]), -float(centroid[1]))
    return compute_town_offset(carla_map, bin_path)


class CarlaTransform:
    """Bidirectional CARLA <-> PufferDrive-bin-frame transform for one town."""

    def __init__(self, town: str, offset=None):
        self.town = town
        self.tx, self.ty = offset if offset is not None else TOWN_OFFSETS[town]

    # --- CARLA -> bin frame ---
    def loc_to_bin(self, cx, cy):
        return cx + self.tx, -cy + self.ty

    def yaw_to_bin(self, yaw_deg):
        return -math.radians(yaw_deg)

    def vel_to_bin(self, vx, vy):
        return vx, -vy

    def actor_state_to_bin(self, actor):
        """Return (x, y, z, heading, vx, vy) in the bin frame for a CARLA actor."""
        tf = actor.get_transform()
        v = actor.get_velocity()
        bx, by = self.loc_to_bin(tf.location.x, tf.location.y)
        return (bx, by, tf.location.z, self.yaw_to_bin(tf.rotation.yaw), v.x, -v.y)

    # --- bin frame -> CARLA (to teleport the ego back into CARLA) ---
    def bin_to_loc(self, bx, by):
        return bx - self.tx, -(by - self.ty)

    def bin_heading_to_yaw(self, heading_rad):
        return -math.degrees(heading_rad)


# CARLA traffic-light state -> PufferDrive enum (datatypes.h:61-67)
#   UNKNOWN=0 RED=1 YELLOW=2 GREEN=3 OFF=4
def carla_light_to_puffer(state) -> int:
    import carla

    return {
        carla.TrafficLightState.Red: 1,
        carla.TrafficLightState.Yellow: 2,
        carla.TrafficLightState.Green: 3,
        carla.TrafficLightState.Off: 4,
        carla.TrafficLightState.Unknown: 0,
    }.get(state, 0)
