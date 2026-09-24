"""CARLA <-> PufferDrive coordinate transform + state readback for co-simulation.

PufferDrive's `carla*/opendrive__Town*.bin` maps are OpenDRIVE exports of the
CARLA towns, stored in a y-flipped, per-town-translated frame. Empirically
(road-network alignment + heading match to 1e-3 rad):

    bin_x       =  carla_x + tx
    bin_y       = -carla_y + ty
    bin_heading = -radians(carla_yaw)          # CARLA yaw is degrees, left-handed
    bin_vx      =  carla_vx
    bin_vy      = -carla_vy

`(tx, ty)` is the town's georeference offset -- a hardcoded constant per town
(TOWN_OFFSETS below), not computed at runtime. History: this used to be
recovered on the fly by ICP-aligning the loaded CARLA map's road points to the
bin's lane points, on the theory that computing it fresh beats trusting a
possibly-stale constant. In practice ICP's local gradient descent (start =
centroid delta, refine by nearest-heading-bucket matching) silently settled a
few meters from the true optimum for Town04/Town06 -- close enough that
aggregate alignment stats still looked fine, but consistently off by about a
lane width, so CARLA-synced background agents rendered off-lane in PufferDrive.
The fix that actually worked: a brute-force (tx, ty) grid search minimizing
median nearest-neighbor distance from transformed CARLA waypoints to bin lane
points (coarse then refined).
For Town6, TWO independently geometry-refined Town06 candidates -- (-150.1,
113.35) from the py123d arrow match, and (-150.93, 109.40) -- both scored
dramatically WORSE on a full longest6 Town06 leaderboard sweep than the
original (-150.93, 111.40).

The PufferDrive env additionally subtracts its own `world_mean` inside
`set_agent_states`/`set_agent_goals`, so the values produced here are in the
*bin* frame (== sim frame + world_mean), matching `get_global_agent_state`.
"""

import math
from pathlib import Path

import numpy as np

import data_utils.mirror_map_bin as _mbin

# Verified per-town offsets -- see the module docstring for how these were
# derived and why they're hardcoded instead of computed on the fly.
TOWN_OFFSETS = {
    "Town01": (-204.34, 148.75),
    "Town02": (-93.70, 213.06),
    "Town03": (-43.56, -4.60),
    "Town04": (-13.73, -10.49),
    "Town05": (49.04, 0.95),
    "Town06": (-150.93, 111.40),
    "Town10HD": (8.13, 32.98),
}

_BIN_DIR = Path(__file__).resolve().parents[2] / "resources" / "drive" / "binaries" / "carla"


def bin_path_for_town(town: str) -> str:
    """Path to the OpenDRIVE bin for a CARLA town name (e.g. 'Town01')."""
    return str(_BIN_DIR / f"opendrive__{town}.bin")


def wrap_deg_180(d):
    """Wrap a degree value (e.g. a yaw difference) to (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def _bin_lane_points(bin_path: str) -> np.ndarray:
    """All lane-polyline (x, y) points from a bin (type 0..9 == lanes)."""
    data = _mbin.read_bin(Path(bin_path))
    pts = []
    for road in data["roads"]:
        if 0 <= road["type"] <= 9:
            pts.extend(zip(road["x"], road["y"]))
    return np.asarray(pts, dtype=np.float64)


def town_offset(bin_path: str):
    """CARLA<->bin (tx, ty): the bin's stored centroid when present (bin =
    original - centroid, and CARLA = original y-flipped, so the offset is just
    -centroid_xy)."""
    centroid = _mbin.read_bin(Path(bin_path)).get("centroid")
    if centroid is not None:
        return (-float(centroid[0]), -float(centroid[1]))
    town = Path(bin_path).stem.split("__")[-1]
    return TOWN_OFFSETS[town]


OFFSET_CALIBRATION_MAX_RESIDUAL_M = (
    1.0  # median lane-normal residual left after calibration: larger = wrong bin for this map
)
OFFSET_CALIBRATION_SPACING_M = 10.0
OFFSET_CALIBRATION_MATCH_MAX_M = 2.0  # waypoint-to-bin-lane residuals beyond this are mismatches, not offset


def calibrate_town_offset(carla_map, transform, town_bin):
    """(tx, ty) refined so the bin's lane polylines coincide with CARLA's lane centres.

    The exported bins sit a constant fraction of a metre off the CARLA frame (measured 2026-09-23 with
    the stored offsets: Town01 0.27 m, Town02 0.6 m), and the ego plus every streamed actor inherited it,
    driving 0.44 m right of CARLA's lane centre on Town02 and clipping curbs the shadow env could not see.
    Least squares of the translation over the lane-normal residuals of CARLA's driving waypoints against
    the nearest bin lane segment, two passes (re-matched after the first shift), mismatches beyond
    OFFSET_CALIBRATION_MATCH_MAX_M dropped. Returns (offset, residual_before_m, residual_after_m)."""
    data = _mbin.read_bin(Path(town_bin))
    seg_start, seg_end = [], []
    for road in data["roads"]:
        if not (0 <= road["type"] <= 9) or len(road["x"]) < 2:
            continue
        pts = np.column_stack([road["x"], road["y"]])
        seg_start.append(pts[:-1])
        seg_end.append(pts[1:])
    seg_start = np.vstack(seg_start)
    seg_end = np.vstack(seg_end)
    seg_dir = seg_end - seg_start
    seg_len_sq = np.maximum((seg_dir**2).sum(1), 1e-9)
    points, normals = [], []
    for wp in carla_map.generate_waypoints(OFFSET_CALIBRATION_SPACING_M):
        if wp.is_junction or str(wp.lane_type) != "Driving":
            continue
        loc, right = wp.transform.location, wp.transform.get_right_vector()
        points.append(transform.loc_to_bin(loc.x, loc.y))
        normals.append((right.x, -right.y))  # y flips into the bin frame
    points = np.array(points, dtype=np.float64).reshape(-1, 2)
    normals = np.array(normals, dtype=np.float64).reshape(-1, 2)
    normals /= np.maximum(np.hypot(normals[:, 0], normals[:, 1]), 1e-9)[:, None]

    def normal_residuals(shift):
        residuals = np.empty(len(points))
        for start in range(0, len(points), 256):
            chunk = points[start : start + 256] - shift
            t = (
                (chunk[:, None, 0] - seg_start[None, :, 0]) * seg_dir[None, :, 0]
                + (chunk[:, None, 1] - seg_start[None, :, 1]) * seg_dir[None, :, 1]
            ) / seg_len_sq[None, :]
            t = np.clip(t, 0.0, 1.0)
            foot_x = seg_start[None, :, 0] + t * seg_dir[None, :, 0]
            foot_y = seg_start[None, :, 1] + t * seg_dir[None, :, 1]
            dist = np.hypot(chunk[:, None, 0] - foot_x, chunk[:, None, 1] - foot_y)
            nearest = np.argmin(dist, axis=1)
            rows = np.arange(len(chunk))
            delta = np.column_stack([foot_x[rows, nearest], foot_y[rows, nearest]]) - chunk
            residuals[start : start + 256] = (delta * normals[start : start + 256]).sum(1)
        return residuals

    shift = np.zeros(2)  # bin lanes = CARLA lanes + shift  ->  loc_to_bin must add shift
    residual_before = float(np.median(np.abs(normal_residuals(shift))))
    for _ in range(2):
        residuals = normal_residuals(-shift)
        keep = np.abs(residuals) <= OFFSET_CALIBRATION_MATCH_MAX_M
        n = normals[keep]
        # residual_i = n_i . v  ->  (N^T N) v = N^T r
        shift += np.linalg.lstsq(n, residuals[keep], rcond=None)[0]
    residual_after = float(np.median(np.abs(normal_residuals(-shift))))
    if residual_after > OFFSET_CALIBRATION_MAX_RESIDUAL_M:
        raise RuntimeError(
            f"{Path(town_bin).name}: bin lanes do not match this CARLA map (median lane residual "
            f"{residual_after:.2f} m after offset calibration)"
        )
    return (transform.tx + float(shift[0]), transform.ty + float(shift[1])), residual_before, residual_after


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
        box = actor.bounding_box  # pivot differs per asset (vehicles: ground, walkers: capsule centre); bin z = ground
        ground_z = tf.location.z + box.location.z - box.extent.z
        return (bx, by, ground_z, heading, v.x, -v.y, yaw_rate, accel_long)

    # --- bin frame -> CARLA (to teleport the ego back into CARLA) ---
    def bin_to_loc(self, bx, by):
        return bx - self.tx, -(by - self.ty)

    def bin_heading_to_yaw(self, heading_rad):
        return -math.degrees(heading_rad)


# CARLA traffic-light state -> PufferDrive enum (datatypes.h:61-67)
#   UNKNOWN=0 RED=1 YELLOW=2 GREEN=3 OFF=4
TRAFFIC_LIGHT_STATE_OFF = 4


def carla_light_to_puffer(state) -> int:
    import carla

    return {
        carla.TrafficLightState.Red: 1,
        carla.TrafficLightState.Yellow: 2,
        carla.TrafficLightState.Green: 3,
        carla.TrafficLightState.Off: 4,
        carla.TrafficLightState.Unknown: 0,
    }.get(state, 0)


LIGHT_LANE_MATCH_MAX_DIST_M = 12.0  # measured Town01 controlled-lane stub distance (see below)
LIGHT_LANE_FORWARD_MATCH_MAX_DIST_M = (
    3.0  # forward fallback lands ON the connector; a lane width away is the neighbour's
)
LIGHT_LANE_ALIGN_COS_MIN = (
    0.5  # forward fallback: the lane must run along the waypoint's heading (no crossing connectors)
)
JUNCTION_CLUSTER_M = 60.0  # stop lines this close are treated as one junction (outlier repair, see below)
# An assigned light farther from its junction cluster-mates' lights than this
# multiple of the cluster-mates' own spread is a wrong-junction match.
OUTLIER_SPREAD_MULTIPLE = 2.0
OUTLIER_MIN_SPREAD_M = 5.0  # floor for tiny/degenerate clusters (e.g. two lights a few meters apart)


def _cluster_by_proximity(points, keys, radius_m):
    """Single-linkage clustering of `keys` by `points[key]` proximity <= radius_m.
    Returns {cluster_root_key: [member_keys]}. points: {key: (x, y)}."""
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            if np.hypot(*(np.subtract(points[k1], points[k2]))) <= radius_m:
                root1, root2 = find(k1), find(k2)
                if root1 != root2:
                    parent[root1] = root2
    clusters = {}
    for k in keys:
        clusters.setdefault(find(k), []).append(k)
    return clusters


def _drop_junction_outliers(mapping, lights, transform, stop_centers):
    """(light_idx, element_idx) pairs to remove from `mapping`: elements whose
    assigned light sits far outside where its junction cluster-mates' lights
    actually are -- see map_lights_to_bin's docstring."""
    element_light = {}
    for li, elements in enumerate(mapping):
        for j in elements:
            element_light.setdefault(j, []).append(li)
    element_light = {j: v[0] for j, v in element_light.items() if len(v) == 1}  # single-owner only
    if len(element_light) < 3:
        return set()

    stop_center_of = {j: tuple(stop_centers[j]) for j in element_light}
    clusters = _cluster_by_proximity(stop_center_of, sorted(element_light), JUNCTION_CLUSTER_M)

    to_drop = set()
    for elements in clusters.values():
        if len(elements) < 3:
            continue
        light_pos = {}
        for j in elements:
            loc = lights[element_light[j]].get_location()
            light_pos[j] = np.array(transform.loc_to_bin(loc.x, loc.y))
        for j in elements:
            others = np.array([light_pos[j2] for j2 in elements if j2 != j])
            centroid = others.mean(axis=0)
            dist_to_centroid = float(np.hypot(*(light_pos[j] - centroid)))
            spread = 0.0
            for a in range(len(others)):
                for b in range(a + 1, len(others)):
                    spread = max(spread, float(np.hypot(*(others[a] - others[b]))))
            if dist_to_centroid > OUTLIER_SPREAD_MULTIPLE * max(spread, OUTLIER_MIN_SPREAD_M):
                to_drop.add((element_light[j], j))
    return to_drop


def map_lights_to_bin(lights, transform, town_bin):
    """mapping[i] = list of bin traffic-element indices controlled by lights[i].

    Semantic matching: each of the light's STOP WAYPOINTS is snapped to its bin
    LANE (nearest drivable-lane segment -- stop lines lie ON lane segments, so
    this is a near-zero-distance match), then lane -> element via the bin's
    controlled_lanes. Geometry-only alternatives mis-bind systematically: the
    light's own transform is the pole across the junction, and stop-line-to-
    stop-line matching is off by ~9 m in these bins while adjacent approaches
    are only 15-25 m apart.

    CARLA's stop waypoints sit at the junction ENTRY, up to ~10 m past the
    bin's controlled-lane stub, so each waypoint is walked backward along its
    own lane until a controlled lane resolves (0/36 raw matches without this).
    Where the bin element only lists the junction connectors (Town01: 3 of 36,
    Town02: 4 of 24 approaches, connectors starting 12-15 m past the waypoint),
    the backward walk finds nothing; a fallback then probes forward into the
    junction for a lane within LIGHT_LANE_FORWARD_MATCH_MAX_DIST_M that runs
    along the waypoint's heading (tight radius + alignment: on multi-lane
    approaches the neighbour lane's connector is only a lane width away).

    A second pass then repairs occasional wrong-junction matches: the walk-back
    above resolves per light independently, and for a small number of lights
    per town it lands on a bin element geometrically nowhere near that light
    (measured: Town04 light 0 -> an element 44 m away, Town06 light 5 -> 34 m
    away, while every OTHER approach in each of those same junctions matched
    within its own junction's ~15-20 m span). Elements never collide (two
    lights never claim the same one), so nothing else catches this -- the
    element just silently carries a confidently wrong state forever. Fix:
    cluster single-owner elements by stop-line proximity (one cluster ~= one
    junction), and within each cluster of >=3, drop any element whose
    assigned light sits far outside where the cluster's OTHER lights actually
    are. A dropped element goes unmapped (state stays UNKNOWN) rather than
    guessing a replacement -- no other candidate light contests these, so
    there's nothing to reassign to with confidence.

    Returns (mapping, num_traffic_elements); the state array passed to
    set_traffic_light_states is sized to the bin's element count."""
    import data_utils.mirror_map_bin as mbin

    data = mbin.read_bin(Path(town_bin))
    seg_start, seg_end, seg_lane = [], [], []
    for road in data["roads"]:
        if not (0 <= road["type"] <= 9):
            continue
        xs, ys = np.asarray(road["x"]), np.asarray(road["y"])
        for k in range(len(xs) - 1):
            seg_start.append((xs[k], ys[k]))
            seg_end.append((xs[k + 1], ys[k + 1]))
            seg_lane.append(road["id"])
    seg_start = np.array(seg_start).reshape(-1, 2)
    seg_end = np.array(seg_end).reshape(-1, 2)
    seg_lane = np.array(seg_lane, dtype=int)
    element_of_lane = {}
    for element_idx, t in enumerate(data["traffic"]):
        for lane_id in t.get("controlled_lanes", ()):
            element_of_lane.setdefault(int(lane_id), []).append(element_idx)

    seg_dir = seg_end - seg_start
    seg_dir = seg_dir / np.maximum(np.hypot(seg_dir[:, 0], seg_dir[:, 1]), 1e-9)[:, None]

    def controlling_elements_near(px, py, max_dist_m, heading=None):
        d = seg_end - seg_start
        length_sq = np.maximum((d**2).sum(1), 1e-9)
        t = np.clip(((px - seg_start[:, 0]) * d[:, 0] + (py - seg_start[:, 1]) * d[:, 1]) / length_sq, 0.0, 1.0)
        dist = np.hypot(px - (seg_start[:, 0] + t * d[:, 0]), py - (seg_start[:, 1] + t * d[:, 1]))
        candidate = dist <= max_dist_m
        if heading is not None:
            candidate &= (
                seg_dir[:, 0] * math.cos(heading) + seg_dir[:, 1] * math.sin(heading) >= LIGHT_LANE_ALIGN_COS_MIN
            )
        lane_best = {}
        for j in np.nonzero(candidate)[0]:
            lane_id = int(seg_lane[j])
            if dist[j] < lane_best.get(lane_id, np.inf):
                lane_best[lane_id] = float(dist[j])
        for lane_id in sorted(lane_best, key=lane_best.get):
            controlling = element_of_lane.get(lane_id, [])
            if controlling:
                return controlling
        return []

    probe_steps_m = (0.0, 3.0, 6.0, 10.0, 15.0)
    mapping = []
    for lt in lights:
        element_indices = []
        for wp in lt.get_stop_waypoints():
            if not len(seg_lane):
                continue
            controlling = []
            for step_m in probe_steps_m:
                probes = [wp] if step_m == 0.0 else wp.previous(step_m)
                for probe in probes[:1]:
                    bx, by = transform.loc_to_bin(probe.transform.location.x, probe.transform.location.y)
                    controlling = controlling_elements_near(bx, by, LIGHT_LANE_MATCH_MAX_DIST_M)
                if controlling:
                    break
            for step_m in probe_steps_m[1:] if not controlling else ():  # fallback: forward into the junction
                for probe in (wp.next(step_m) or [])[:1]:
                    bx, by = transform.loc_to_bin(probe.transform.location.x, probe.transform.location.y)
                    heading = transform.yaw_to_bin(probe.transform.rotation.yaw)
                    controlling = controlling_elements_near(bx, by, LIGHT_LANE_FORWARD_MATCH_MAX_DIST_M, heading)
                if controlling:
                    break
            element_indices.extend(controlling)
        mapping.append(sorted(set(element_indices)))

    stop_centers = np.array(
        [
            [0.5 * (t["stop_line"][0] + t["stop_line"][3]), 0.5 * (t["stop_line"][1] + t["stop_line"][4])]
            for t in data["traffic"]
        ]
    ).reshape(-1, 2)
    for light_idx, element_idx in _drop_junction_outliers(mapping, lights, transform, stop_centers):
        mapping[light_idx].remove(element_idx)

    return mapping, len(data["traffic"])


def stop_signs_from_carla(world, carla_map, transform):
    """(lines (K, 6), headings (K,)) in the bin frame, one per CARLA `traffic.stop` actor: the trigger
    volume's centre projected onto its driving lane, spanning the trigger's width across that lane, with
    the lane's travel direction as heading. Feed to Drive.set_stop_signs so the shadow env runs on the
    volumes the leaderboard scores against instead of the bin's exported stop lines (which sit up to
    9 m away on a few Town03/05 approaches and miss four signs)."""
    import carla

    lines, headings = [], []
    for actor in world.get_actors().filter("traffic.stop"):
        actor_transform = actor.get_transform()
        trigger = actor.trigger_volume
        center = actor_transform.transform(trigger.location)
        waypoint = carla_map.get_waypoint(center, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            continue
        forward = waypoint.transform.get_forward_vector()
        lane_yaw = math.atan2(forward.y, forward.x)
        box_yaw = math.radians(actor_transform.rotation.yaw)
        box_x_along_lane = abs(math.cos(box_yaw - lane_yaw)) >= math.cos(math.pi / 4.0)
        half_width = trigger.extent.y if box_x_along_lane else trigger.extent.x
        across_x, across_y = -math.sin(lane_yaw), math.cos(lane_yaw)
        left = transform.loc_to_bin(center.x - half_width * across_x, center.y - half_width * across_y)
        right = transform.loc_to_bin(center.x + half_width * across_x, center.y + half_width * across_y)
        lines.append([left[0], left[1], center.z, right[0], right[1], center.z])
        headings.append(transform.yaw_to_bin(math.degrees(lane_yaw)))
    return np.array(lines, np.float32).reshape(-1, 6), np.array(headings, np.float32)
