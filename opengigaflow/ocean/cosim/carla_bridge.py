"""CARLA <-> OpenGigaFlow bin frame (y mirrored, translated per town) and CARLA map/light/stop-sign readback."""

import math
from pathlib import Path

import numpy as np

import data_utils.mirror_map_bin as _mbin


# bin_x = carla_x + tx, bin_y = -carla_y + ty; found by a brute-force grid search (ICP settled a lane width off)
TOWN_OFFSETS = {
    "Town01": (-204.34, 148.75),
    "Town02": (-93.70, 213.06),
    "Town03": (-43.56, -4.60),
    "Town04": (-13.73, -10.49),
    "Town05": (49.04, 0.95),
    "Town06": (-150.93, 111.40),
    "Town10HD": (8.13, 32.98),
}

DEFAULT_TOWN_BIN_DIR = Path(__file__).resolve().parents[2] / "resources" / "carla"


def bin_path_for_town(town: str, town_bin_dir) -> str:
    """Path to the OpenDRIVE bin for a CARLA town name (e.g. 'Town01') in the folder town_bin_dir."""
    return str(Path(town_bin_dir) / f"opendrive__{town}.bin")


# median lane-normal residual left after calibration: larger = wrong bin for this map
OFFSET_CALIBRATION_MAX_RESIDUAL_M = 1.0
OFFSET_CALIBRATION_SPACING_M = 10.0
OFFSET_CALIBRATION_MATCH_MAX_M = 2.0  # waypoint-to-bin-lane residuals beyond this are mismatches, not offset
OFFSET_CALIBRATION_MAX_Z_OFFSET_M = 5.0  # exported bins sit 0-1.7 m below CARLA's ground; more means a frame mix-up


DRIVING_WAYPOINT_COLUMNS = 5  # x, y, z, right_x, right_y in the CARLA frame


def driving_waypoint_samples(carla_map):
    """(N, 5) [x, y, z, right_x, right_y] of CARLA's non-junction driving waypoints, calibrate_town_offset's input."""
    rows = []
    for wp in carla_map.generate_waypoints(OFFSET_CALIBRATION_SPACING_M):
        if wp.is_junction or str(wp.lane_type) != "Driving":
            continue
        loc, right = wp.transform.location, wp.transform.get_right_vector()
        rows.append((loc.x, loc.y, loc.z, right.x, right.y))
    return np.array(rows, dtype=np.float64).reshape(-1, DRIVING_WAYPOINT_COLUMNS)


def calibrate_town_offset(driving_waypoints, transform, town_bin):
    """(offset, z_offset_m, residual_before_m, residual_after_m): least squares of the translation over the lane-normal
    residuals of driving_waypoint_samples against the bin lanes (two passes), z offset = median bin lane z - CARLA z."""
    data = _mbin.read_bin(Path(town_bin))
    seg_start, seg_end, seg_z_start, seg_z_end = [], [], [], []
    for road in data["roads"]:
        if not (0 <= road["type"] <= 9) or len(road["x"]) < 2:
            continue
        pts = np.column_stack([road["x"], road["y"]])
        heights = np.asarray(road["z"], dtype=np.float64)
        seg_start.append(pts[:-1])
        seg_end.append(pts[1:])
        seg_z_start.append(heights[:-1])
        seg_z_end.append(heights[1:])
    seg_start = np.vstack(seg_start)
    seg_end = np.vstack(seg_end)
    seg_z_start = np.concatenate(seg_z_start)
    seg_z_end = np.concatenate(seg_z_end)
    seg_dir = seg_end - seg_start
    seg_len_sq = np.maximum((seg_dir**2).sum(1), 1e-9)
    samples = np.asarray(driving_waypoints, dtype=np.float64).reshape(-1, DRIVING_WAYPOINT_COLUMNS)
    points = np.array([transform.loc_to_bin(x, y) for x, y in samples[:, :2]], dtype=np.float64).reshape(-1, 2)
    normals = np.column_stack([samples[:, 3], -samples[:, 4]])  # y flips into the bin frame
    normals /= np.maximum(np.hypot(normals[:, 0], normals[:, 1]), 1e-9)[:, None]
    waypoint_z = samples[:, 2]

    def normal_residuals(shift):
        residuals = np.empty(len(points))
        foot_z = np.empty(len(points))
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
            t_nearest = t[rows, nearest]
            foot_z[start : start + 256] = seg_z_start[nearest] + t_nearest * (seg_z_end[nearest] - seg_z_start[nearest])
        return residuals, foot_z

    shift = np.zeros(2)  # bin lanes = CARLA lanes + shift  ->  loc_to_bin must add shift
    residual_before = float(np.median(np.abs(normal_residuals(shift)[0])))
    for _ in range(2):
        residuals, _ = normal_residuals(-shift)
        keep = np.abs(residuals) <= OFFSET_CALIBRATION_MATCH_MAX_M
        n = normals[keep]
        # residual_i = n_i . v  ->  (N^T N) v = N^T r
        shift += np.linalg.lstsq(n, residuals[keep], rcond=None)[0]
    residuals, foot_z = normal_residuals(-shift)
    residual_after = float(np.median(np.abs(residuals)))
    if residual_after > OFFSET_CALIBRATION_MAX_RESIDUAL_M:
        raise RuntimeError(
            f"{Path(town_bin).name}: bin lanes do not match this CARLA map (median lane residual "
            f"{residual_after:.2f} m after offset calibration)"
        )
    keep = np.abs(residuals) <= OFFSET_CALIBRATION_MATCH_MAX_M
    z_offset = float(np.median(foot_z[keep] - waypoint_z[keep]))  # bin z = CARLA z + z_offset
    if not np.isfinite(z_offset) or abs(z_offset) > OFFSET_CALIBRATION_MAX_Z_OFFSET_M:
        raise RuntimeError(
            f"{Path(town_bin).name}: bin lane height is {z_offset:.2f} m off this CARLA map (limit "
            f"{OFFSET_CALIBRATION_MAX_Z_OFFSET_M} m)"
        )
    return (transform.tx + float(shift[0]), transform.ty + float(shift[1])), z_offset, residual_before, residual_after


class CarlaTransform:
    """CARLA <-> bin frame of one town (the sim frame + world_mean, as set_agent_states and get_global_agent_state
    use); bin z = CARLA z + z_offset, so every injected height goes through z_to_bin."""

    def __init__(self, town: str, offset=None, z_offset=0.0):
        self.tx, self.ty = offset if offset is not None else TOWN_OFFSETS[town]
        self.tz = float(z_offset)

    def loc_to_bin(self, cx, cy):
        return cx + self.tx, -cy + self.ty

    def z_to_bin(self, cz):
        return cz + self.tz

    def yaw_to_bin(self, yaw_deg):
        return -math.radians(yaw_deg)

    def state_to_bin(self, x, y, z, yaw_deg, vx, vy, angular_velocity_z_deg_s, accel_x, accel_y, box_z, box_extent_z):
        """(x, y, z, heading, vx, vy, yaw_rate, accel_long) in the bin frame from a CARLA-frame actor state with
        CARLA's own angular velocity (deg/s) and world-frame acceleration; bin z = the bounding box's ground."""
        bx, by = self.loc_to_bin(x, y)
        heading = self.yaw_to_bin(yaw_deg)
        # mirrored y flips rotation sense, like yaw_to_bin's negation
        yaw_rate = -math.radians(angular_velocity_z_deg_s)
        accel_long = accel_x * math.cos(heading) + (-accel_y) * math.sin(heading)  # mirror y, like the velocity
        ground_z = z + box_z - box_extent_z
        return (bx, by, self.z_to_bin(ground_z), heading, vx, -vy, yaw_rate, accel_long)

    def bin_to_loc(self, bx, by):
        return bx - self.tx, -(by - self.ty)

    def bin_heading_to_yaw(self, heading_rad):
        return -math.degrees(heading_rad)


# carla.TrafficLightState name -> OpenGigaFlow light state (UNKNOWN=0 RED=1 YELLOW=2 GREEN=3 OFF=4)
TRAFFIC_LIGHT_STATE_YELLOW = 2
TRAFFIC_LIGHT_STATE_OFF = 4
LIGHT_STATE_BY_NAME = {
    "Red": 1,
    "Yellow": TRAFFIC_LIGHT_STATE_YELLOW,
    "Green": 3,
    "Off": TRAFFIC_LIGHT_STATE_OFF,
    "Unknown": 0,
}
# TrafficLightManipulator (Leaderboard 1 scenarios 7-9) freezes phase times at 1e9 s: the timer then tells nothing
CARLA_LIGHT_TIMER_FROZEN_S = 1.0e6


def carla_yellow_remaining_s(elapsed_s, yellow_time_s):
    """Seconds a CARLA light stays yellow by its own phase timer (get_yellow_time - get_elapsed_time), or None
    when a scenario froze that timer."""
    if float(yellow_time_s) >= CARLA_LIGHT_TIMER_FROZEN_S:
        return None
    return max(float(yellow_time_s) - float(elapsed_s), 0.0)


LIGHT_PROBE_STEPS_M = (0.0, 3.0, 6.0, 10.0, 15.0)  # walk-back (all) / forward (all but 0) probes of map_lights_to_bin
LIGHT_LANE_MATCH_MAX_DIST_M = 12.0  # measured Town01 controlled-lane stub distance
# forward fallback lands ON the connector; a lane width away is the neighbour's
LIGHT_LANE_FORWARD_MATCH_MAX_DIST_M = 3.0
# forward fallback: the lane must run along the waypoint's heading (no crossing connectors)
LIGHT_LANE_ALIGN_COS_MIN = 0.5
LIGHT_STOP_LINE_TOLERANCE_M = 1.0  # exported bin lines within this of the light's stop waypoint stay (median 0.5 m)
ROUTE_STOP_LINE_MARGIN_M = 1.0  # a route crossing a stop line extended by this at both ends drives that line's lane
ROUTE_STOP_LINE_ALIGN_COS_MIN = 0.5  # ... in the element's travel direction (opposing traffic crosses the same line)
TRIGGER_SAMPLE_SPAN_FRACTION = 0.9  # RunningRedLightTest.get_traffic_light_waypoints: avoids adjacent lanes
TRIGGER_SAMPLE_STEP_M = 1.0
JUNCTION_ADVANCE_STEP_M = 0.5
MAX_JUNCTION_ADVANCE_STEPS = 400  # 200 m: a trigger volume farther from its junction is a map error
JUNCTION_CLUSTER_M = 60.0  # stop lines this close are treated as one junction
OUTLIER_SPREAD_MULTIPLE = 2.0  # a light this many cluster-mate spreads from their centroid matched the wrong junction
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


def _drop_junction_outliers(mapping, light_geometry, transform, stop_centers):
    """(light_idx, element_idx) pairs to drop: single-owner elements whose light sits far outside where the lights of
    the other elements of its junction (stop-line cluster of >= 3) are; no other light contests them."""
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
            light = light_geometry[element_light[j]]
            light_pos[j] = np.array(transform.loc_to_bin(light["x"], light["y"]))
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


def light_stop_waypoints(light, carla_map):
    """One waypoint per lane the light's trigger volume spans, advanced to the junction entry: the line
    RunningRedLightTest scores (TrafficLight.get_stop_waypoints sits 10-14 m before it in Town01/02)."""
    import carla

    base_transform = light.get_transform()
    trigger = light.trigger_volume
    lane_waypoints = []
    for x in np.arange(
        -TRIGGER_SAMPLE_SPAN_FRACTION * trigger.extent.x,
        TRIGGER_SAMPLE_SPAN_FRACTION * trigger.extent.x,
        TRIGGER_SAMPLE_STEP_M,
    ):
        wp = carla_map.get_waypoint(base_transform.transform(trigger.location + carla.Location(x=float(x))))
        if not lane_waypoints or lane_waypoints[-1].road_id != wp.road_id or lane_waypoints[-1].lane_id != wp.lane_id:
            lane_waypoints.append(wp)
    stop_waypoints = []
    for wp in lane_waypoints:
        for _ in range(MAX_JUNCTION_ADVANCE_STEPS):
            if wp.is_junction:
                break
            ahead = wp.next(JUNCTION_ADVANCE_STEP_M)
            if not ahead or ahead[0].is_junction:
                break
            wp = ahead[0]
        stop_waypoints.append(wp)
    return stop_waypoints


def light_geometry_from_carla(lights, carla_map):
    """CARLA-frame dicts per light {id, x, y, trigger_x, trigger_y, stop_waypoints: [{x, y, yaw_deg, lane_width,
    backward: [[x, y] per LIGHT_PROBE_STEPS_M behind], forward: [[x, y, yaw_deg] per non-zero step ahead]}]}."""
    geometry = []
    for light in lights:
        location = light.get_location()
        trigger = light.get_transform().transform(light.trigger_volume.location)
        stop_waypoints = []
        for wp in light_stop_waypoints(light, carla_map):
            backward, forward = [], []
            for step_m in LIGHT_PROBE_STEPS_M:
                probes = [wp] if step_m == 0.0 else wp.previous(step_m)
                backward.append(
                    None if not probes else [probes[0].transform.location.x, probes[0].transform.location.y]
                )
            for step_m in LIGHT_PROBE_STEPS_M[1:]:
                probes = wp.next(step_m) or []
                forward.append(
                    None
                    if not probes
                    else [
                        probes[0].transform.location.x,
                        probes[0].transform.location.y,
                        probes[0].transform.rotation.yaw,
                    ]
                )
            stop_waypoints.append(
                {
                    "x": wp.transform.location.x,
                    "y": wp.transform.location.y,
                    "yaw_deg": wp.transform.rotation.yaw,
                    "lane_width": wp.lane_width,
                    "backward": backward,
                    "forward": forward,
                }
            )
        geometry.append(
            {
                "id": light.id,
                "x": location.x,
                "y": location.y,
                "trigger_x": trigger.x,
                "trigger_y": trigger.y,
                "stop_waypoints": stop_waypoints,
            }
        )
    return geometry


def map_lights_to_bin(light_geometry, transform, town_bin):
    """(mapping, bin traffic element count), mapping[i] = bin elements light_geometry[i] controls, matched by lane (the
    light pole stands across the junction): each stop waypoint walks back along its lane, else forward into the
    junction, until a bin lane with controlling elements is in reach; wrong-junction matches are then dropped."""
    data = _mbin.read_bin(Path(town_bin))
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

    mapping = []
    for light in light_geometry:
        element_indices = []
        for stop_waypoint in light["stop_waypoints"]:
            if not len(seg_lane):
                continue
            controlling = []
            for probe in stop_waypoint["backward"]:
                if probe is None:
                    continue
                bx, by = transform.loc_to_bin(probe[0], probe[1])
                controlling = controlling_elements_near(bx, by, LIGHT_LANE_MATCH_MAX_DIST_M)
                if controlling:
                    break
            for probe in stop_waypoint["forward"] if not controlling else ():  # fallback: forward into the junction
                if probe is None:
                    continue
                bx, by = transform.loc_to_bin(probe[0], probe[1])
                heading = transform.yaw_to_bin(probe[2])
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
    for light_idx, element_idx in _drop_junction_outliers(mapping, light_geometry, transform, stop_centers):
        mapping[light_idx].remove(element_idx)

    return mapping, len(data["traffic"])


def route_crosses_stop_line(route_xy, stop_line, heading):
    """True when the polyline route_xy (N, 2) crosses stop_line [x1, y1, z1, x2, y2, z2] (extended by
    ROUTE_STOP_LINE_MARGIN_M at both ends) while running along `heading`, the element's travel direction."""
    route = np.asarray(route_xy, dtype=np.float64).reshape(-1, 2)
    if len(route) < 2:
        return False
    x1, y1, _, x2, y2, _ = stop_line
    line_len = math.hypot(x2 - x1, y2 - y1)
    if line_len <= 0.0:
        return False
    ux, uy = (x2 - x1) / line_len, (y2 - y1) / line_len
    ax, ay = x1 - ROUTE_STOP_LINE_MARGIN_M * ux, y1 - ROUTE_STOP_LINE_MARGIN_M * uy
    bx, by = x2 + ROUTE_STOP_LINE_MARGIN_M * ux, y2 + ROUTE_STOP_LINE_MARGIN_M * uy
    p, q = route[:-1], route[1:]
    d = q - p
    seg_len = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    along = (d[:, 0] * math.cos(heading) + d[:, 1] * math.sin(heading)) / seg_len
    line_side_p = (bx - ax) * (p[:, 1] - ay) - (by - ay) * (p[:, 0] - ax)
    line_side_q = (bx - ax) * (q[:, 1] - ay) - (by - ay) * (q[:, 0] - ax)
    seg_side_a = d[:, 0] * (ay - p[:, 1]) - d[:, 1] * (ax - p[:, 0])
    seg_side_b = d[:, 0] * (by - p[:, 1]) - d[:, 1] * (bx - p[:, 0])
    crossing = ((line_side_p < 0.0) != (line_side_q < 0.0)) & ((seg_side_a < 0.0) != (seg_side_b < 0.0))
    return bool(np.any(crossing & (along >= ROUTE_STOP_LINE_ALIGN_COS_MIN)))


def light_stop_line_overrides(light_geometry, mapping, transform, town_bin, route_xy=None):
    """(element indices (K,), lines (K, 6)): bin light stop lines more than LIGHT_STOP_LINE_TOLERANCE_M along the lane
    from the light's scored stop waypoints, moved along the travel direction onto the most upstream one. With route_xy
    (N, 2, bin frame) only lines the route crosses move: a moved neighbouring line is out of distribution."""
    bin_traffic = _mbin.read_bin(Path(town_bin))["traffic"]
    along_offsets_by_element = {}
    for light, elements in zip(light_geometry, mapping):
        if not elements:
            continue
        mids = {
            j: (
                0.5 * (bin_traffic[j]["stop_line"][0] + bin_traffic[j]["stop_line"][3]),
                0.5 * (bin_traffic[j]["stop_line"][1] + bin_traffic[j]["stop_line"][4]),
            )
            for j in elements
        }
        for waypoint in light["stop_waypoints"]:
            wx, wy = transform.loc_to_bin(waypoint["x"], waypoint["y"])
            j = min(elements, key=lambda e: math.hypot(mids[e][0] - wx, mids[e][1] - wy))
            heading = bin_traffic[j]["heading"]
            along = (wx - mids[j][0]) * math.cos(heading) + (wy - mids[j][1]) * math.sin(heading)
            along_offsets_by_element.setdefault(j, []).append(along)
    indices, lines = [], []
    for j in sorted(along_offsets_by_element):
        shift_m = min(along_offsets_by_element[j])
        if abs(shift_m) <= LIGHT_STOP_LINE_TOLERANCE_M:
            continue
        if route_xy is not None and not route_crosses_stop_line(
            route_xy, bin_traffic[j]["stop_line"], bin_traffic[j]["heading"]
        ):
            continue
        heading = bin_traffic[j]["heading"]
        dx, dy = shift_m * math.cos(heading), shift_m * math.sin(heading)
        x1, y1, z1, x2, y2, z2 = bin_traffic[j]["stop_line"]
        indices.append(j)
        lines.append([x1 + dx, y1 + dy, z1, x2 + dx, y2 + dy, z2])
    return np.array(indices, np.int32), np.array(lines, np.float32).reshape(-1, 6)


def stop_sign_geometry_from_carla(world, carla_map):
    """CARLA-frame dicts per `traffic.stop` actor on a driving lane {id, center, extent, yaw_deg of the trigger volume,
    lane: [x, y, z, yaw_deg] of the driving-lane waypoint under the centre}."""
    import carla

    geometry = []
    for actor in world.get_actors().filter("traffic.stop"):
        actor_transform = actor.get_transform()
        trigger = actor.trigger_volume
        center = actor_transform.transform(trigger.location)
        waypoint = carla_map.get_waypoint(center, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            continue
        forward = waypoint.transform.get_forward_vector()
        lane_location = waypoint.transform.location
        geometry.append(
            {
                "id": actor.id,
                "center": [center.x, center.y, center.z],
                "extent": [trigger.extent.x, trigger.extent.y, trigger.extent.z],
                "yaw_deg": actor_transform.rotation.yaw,
                "lane": [
                    lane_location.x,
                    lane_location.y,
                    lane_location.z,
                    math.degrees(math.atan2(forward.y, forward.x)),
                ],
            }
        )
    return geometry


def stop_sign_lines(stop_sign_geometry, transform):
    """(lines (K, 6), headings (K,)) in the bin frame for Drive.set_stop_signs: each trigger volume's centre on its
    driving lane, spanning the trigger's width across it at the lane's height, heading = the lane's travel direction."""
    lines, headings = [], []
    for sign in stop_sign_geometry:
        center_x, center_y, _ = sign["center"]
        extent_x, extent_y, _ = sign["extent"]
        _, _, lane_z, lane_yaw_deg = sign["lane"]
        lane_yaw = math.radians(lane_yaw_deg)
        box_yaw = math.radians(sign["yaw_deg"])
        box_x_along_lane = abs(math.cos(box_yaw - lane_yaw)) >= math.cos(math.pi / 4.0)
        half_width = extent_y if box_x_along_lane else extent_x
        across_x, across_y = -math.sin(lane_yaw), math.cos(lane_yaw)
        left = transform.loc_to_bin(center_x - half_width * across_x, center_y - half_width * across_y)
        right = transform.loc_to_bin(center_x + half_width * across_x, center_y + half_width * across_y)
        line_z = transform.z_to_bin(lane_z)
        lines.append([left[0], left[1], line_z, right[0], right[1], line_z])
        headings.append(transform.yaw_to_bin(lane_yaw_deg))
    return np.array(lines, np.float32).reshape(-1, 6), np.array(headings, np.float32)
