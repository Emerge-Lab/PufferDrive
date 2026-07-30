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

# Corrected per-town offsets
TOWN_OFFSETS = {
    "Town01": (-204.34, 148.75),
    "Town02": (-93.70, 213.06),
    "Town03": (-43.56, -4.60),
    "Town04": (-10.98, -7.49),
    "Town05": (49.04, 0.95),
    "Town10HD": (8.13, 32.98),
}

_BIN_DIR = Path(__file__).resolve().parents[2] / "resources" / "drive" / "binaries" / "carla"


def bin_path_for_town(town: str) -> str:
    """Path to the OpenDRIVE bin for a CARLA town name (e.g. 'Town01')."""
    return str(_BIN_DIR / f"opendrive__{town}.bin")


def _bin_lane_points(bin_path: str) -> np.ndarray:
    """All lane-polyline (x, y) points from a bin (type 0..9 == lanes)."""
    return _bin_lane_points_headings(bin_path)[0]

def _bin_lane_points_headings(bin_path: str):
    """Lane-polyline points and per-point headings from a bin (type 0..9)."""
    data = _mbin.read_bin(Path(bin_path))
    pts, hds = [], []
    for road in data["roads"]:
        if 0 <= road["type"] <= 9:
            pts.extend(zip(road["x"], road["y"]))
            hds.extend(road["headings"])
    return np.asarray(pts, dtype=np.float64), np.asarray(hds, dtype=np.float64)

HEADING_OCTANTS = 8  # ICP heading-match granularity (45 deg buckets)


def compute_town_offset(carla_map, bin_path: str, sample_m: float = 2.0, iters: int = 10):
    """Recover (tx, ty) by ICP-aligning CARLA driving waypoints to the bin
    lanes, with the y-flip reflection fixed. Matches only heading-compatible
    lane points (waypoint travel direction within ~67 deg of the bin lane):
    plain nearest-neighbor ICP can lock onto the opposite-direction lane of a
    two-way road and converge with a stable one-lane lateral bias (observed
    +2.59 m on Town01's east-west roads). Returns (tx, ty)."""
    wps = carla_map.generate_waypoints(sample_m)
    C = np.array([[w.transform.location.x, w.transform.location.y] for w in wps], dtype=np.float64)
    CH = np.array([-math.radians(w.transform.rotation.yaw) for w in wps], dtype=np.float64)
    B, BH = _bin_lane_points_headings(bin_path)
    TC = C * np.array([1.0, -1.0])  # y-flip (headings already flipped via -yaw)

    # Bucket bin points by heading octant; a waypoint in octant o may match
    # octants o-1..o+1 (+-67.5 deg), which excludes the opposite direction.
    bin_oct = np.round(BH / (2.0 * np.pi / HEADING_OCTANTS)).astype(int) % HEADING_OCTANTS
    wp_oct = np.round(CH / (2.0 * np.pi / HEADING_OCTANTS)).astype(int) % HEADING_OCTANTS
    cand = [np.where((bin_oct == (o - 1) % HEADING_OCTANTS)
                     | (bin_oct == o)
                     | (bin_oct == (o + 1) % HEADING_OCTANTS))[0] for o in range(HEADING_OCTANTS)]

    t = B.mean(0) - TC.mean(0)
    sub = slice(None, None, 2)
    P0, PO = TC[sub], wp_oct[sub]
    for _ in range(iters):
        P = P0 + t
        res = np.zeros_like(P)
        for o in range(HEADING_OCTANTS):
            m = PO == o
            if not m.any() or not len(cand[o]):
                continue
            Bo = B[cand[o]]
            idx = ((P[m, None, 0] - Bo[None, :, 0]) ** 2
                   + (P[m, None, 1] - Bo[None, :, 1]) ** 2).argmin(1)
            res[m] = Bo[idx] - P[m]
        t = t + res.mean(0)
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
        """Return (x, y, z, heading, vx, vy, yaw_rate, accel_long) in the bin frame for a CARLA actor.
        yaw_rate/accel_long come from CARLA's own physics (get_angular_velocity/get_acceleration), not
        finite-differenced from a previous PufferDrive state: env.step() may have already overwritten
        this actor's previous state with a throwaway dummy-action rollout by the time PufferDrive would
        otherwise read "previous" state itself."""
        tf = actor.get_transform()
        v = actor.get_velocity()
        av = actor.get_angular_velocity()  # deg/s, world frame (CARLA's rotation convention)
        acc = actor.get_acceleration()  # m/s^2, world frame
        bx, by = self.loc_to_bin(tf.location.x, tf.location.y)
        heading = self.yaw_to_bin(tf.rotation.yaw)
        yaw_rate = -math.radians(av.z)  # mirrored y flips rotation sense, like yaw_to_bin's negation
        accel_x, accel_y = acc.x, -acc.y  # mirror y, like vel_to_bin
        accel_long = accel_x * math.cos(heading) + accel_y * math.sin(heading)
        return (bx, by, tf.location.z, heading, v.x, -v.y, yaw_rate, accel_long)

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
