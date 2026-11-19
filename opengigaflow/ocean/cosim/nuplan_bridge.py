"""nuPlan <-> OpenGigaFlow city-bin frame (translation only) and nuPlan readback; nuplan is imported lazily."""

from pathlib import Path

import numpy as np

import data_utils.mirror_map_bin as _mbin


TRAFFIC_TYPE_LIGHT = 1  # matches the CARLA bins' traffic-element type

# nuPlan TrackedObjectType -> OpenGigaFlow agent type (1=vehicle 2=ped 3=cyclist)
_NUPLAN_AGENT_TYPE = {"VEHICLE": 1, "PEDESTRIAN": 2, "BICYCLE": 3}

# nuPlan TrafficLightStatusType -> light state; training never produces UNKNOWN, so unknown or unreported is GREEN
LIGHT_STATE_RED = 1
LIGHT_STATE_YELLOW = 2
LIGHT_STATE_GREEN = 3
_NUPLAN_LIGHT_STATE = {
    "RED": LIGHT_STATE_RED,
    "YELLOW": LIGHT_STATE_YELLOW,
    "GREEN": LIGHT_STATE_GREEN,
    "UNKNOWN": LIGHT_STATE_GREEN,
}

ROUTE_SEARCH_DEPTH_BLOCKS = 30  # PDM's Dijkstra window over route roadblocks


class NuPlanTransform:
    """Translation-only nuPlan <-> bin-frame transform for one scenario."""

    def __init__(self, origin_x: float, origin_y: float):
        self.ox = float(origin_x)
        self.oy = float(origin_y)

    def loc_to_bin(self, x, y):
        return x - self.ox, y - self.oy

    def bin_to_loc(self, bx, by):
        return bx + self.ox, by + self.oy


def route_lane_ids(map_api, route_roadblock_ids, start_x: float, start_y: float, start_heading: float):
    """[lane ids] of the PDM-style lane-graph route: Dijkstra from the on-route lane under the ego with the smallest
    heading error (never a crossing lane of the same junction) to the last route roadblock."""
    from nuplan.common.actor_state.state_representation import Point2D
    from carl_nuplan.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra

    blocks = _route_blocks(map_api, dict.fromkeys(route_roadblock_ids))
    if not blocks:
        return []
    route_lanes = {lane.id: lane for block in blocks for lane in block.interior_edges}

    def _lane_pts(lane):
        return np.array([[p.x, p.y] for p in lane.baseline_path.discrete_path])

    def _heading_error(lane):
        path = lane.baseline_path.discrete_path
        pts = _lane_pts(lane)
        nearest = int(np.argmin(np.hypot(*(pts - (start_x, start_y)).T)))
        return abs((path[nearest].heading - start_heading + np.pi) % (2.0 * np.pi) - np.pi)

    def _distance(lane):
        return float(np.min(np.hypot(*(_lane_pts(lane) - (start_x, start_y)).T)))

    ego_point = Point2D(start_x, start_y)
    containing = sorted((lane for lane in route_lanes.values() if lane.contains_point(ego_point)), key=_heading_error)
    block_ids = [block.id for block in blocks]
    start_block_idx = block_ids.index(containing[0].get_roadblock_id()) if containing else 0
    # the ego's lane may not connect to the next route block (wrong lane for the turn): its siblings, nearest first
    siblings = sorted(
        (lane for lane in blocks[start_block_idx].interior_edges if lane not in containing), key=_distance
    )
    candidates = containing + siblings or sorted(route_lanes.values(), key=_distance)[:1]
    target_block = blocks[min(len(blocks) - 1, start_block_idx + ROUTE_SEARCH_DEPTH_BLOCKS - 1)]
    path = []
    for start_lane in candidates:
        candidate_path, found = Dijkstra(start_lane, list(route_lanes.keys())).search(target_block)
        if found:
            path = candidate_path
            break
        if len(candidate_path) > len(path):
            path = candidate_path
    return [str(lane.id) for lane in path]


def _route_blocks(map_api, route_roadblock_ids):
    """Route roadblock ids -> [roadblock or roadblock connector] with interior lanes, consecutive repeats merged."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    blocks = []
    for rid in route_roadblock_ids:
        block = map_api.get_map_object(str(rid), SemanticMapLayer.ROADBLOCK) or map_api.get_map_object(
            str(rid), SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        if block is None or not block.interior_edges or (blocks and blocks[-1].id == block.id):
            continue
        blocks.append(block)
    return blocks


def roadblock_centroid_goals(
    map_api, route_roadblock_ids, start_x, start_y, start_heading, min_spacing: float, min_ahead_m: float
):
    """(N, 2) goals in nuPlan map coordinates: route roadblock centroids from the ego's (nearest) roadblock on, its own
    only when min_ahead_m ahead, off-polygon ones moved onto a lane baseline, ones closer than min_spacing thinned."""
    from shapely.geometry import Point

    blocks = _route_blocks(map_api, route_roadblock_ids)
    if not blocks:
        return np.zeros((0, 2))
    ego_point = Point(start_x, start_y)
    start_idx = int(np.argmin([block.polygon.distance(ego_point) for block in blocks]))
    heading_dir = np.array([np.cos(start_heading), np.sin(start_heading)])
    goals = []
    for block_idx in range(start_idx, len(blocks)):
        block = blocks[block_idx]
        centroid = block.polygon.centroid
        goal = np.array([centroid.x, centroid.y], dtype=np.float64)
        if not block.polygon.contains(centroid):
            lane_pts = np.concatenate(
                [[[p.x, p.y] for p in lane.baseline_path.discrete_path] for lane in block.interior_edges]
            )
            goal = lane_pts[int(np.argmin(np.hypot(*(lane_pts - goal).T)))].astype(np.float64)
        if block_idx == start_idx and (goal - (start_x, start_y)) @ heading_dir < min_ahead_m:
            continue
        if goals and np.hypot(*(goal - goals[-1])) < min_spacing:
            continue
        goals.append(goal)
    return np.array(goals, dtype=np.float64).reshape(-1, 2)


def extend_route_past_loop_cut(corrected_ids, raw_ids):
    """[str ids]: CaRL's route correction cuts a looping route for its Dijkstra; goals follow the whole route, so the
    logged remainder after the cut is appended again."""
    corrected = [str(rid) for rid in corrected_ids]
    raw = [str(rid) for rid in raw_ids]
    if not corrected or corrected[-1] not in raw:
        return corrected
    return corrected + raw[raw.index(corrected[-1]) + 1 :]


def logged_ego_boxes(scenario, transform: NuPlanTransform) -> np.ndarray:
    """Human-driven ego per scenario iteration -> (N, 5) float32 [x, y, heading, length, width], bin frame."""
    iteration_count = scenario.get_number_of_iterations()
    boxes = np.zeros((iteration_count, 5), np.float32)
    for i in range(iteration_count):
        state = scenario.get_ego_state_at_iteration(i)
        bx, by = transform.loc_to_bin(float(state.center.x), float(state.center.y))
        footprint = state.car_footprint
        boxes[i] = (bx, by, float(state.center.heading), float(footprint.length), float(footprint.width))
    return boxes


def read_bin_geometry(bin_path: Path) -> dict:
    """Bin-frame traffic elements of a bin: stop_line_centers (K, 2), stop_lines (K, 4) [x1, y1, x2, y2],
    stop_line_headings (K,) [rad], traffic_types (K,) and num_traffic K."""
    data = _mbin.read_bin(Path(bin_path))
    stop_lines = np.array(
        [[t["stop_line"][0], t["stop_line"][1], t["stop_line"][3], t["stop_line"][4]] for t in data["traffic"]],
        dtype=np.float64,
    ).reshape(-1, 4)
    centers = 0.5 * (stop_lines[:, 0:2] + stop_lines[:, 2:4])
    stop_line_headings = np.array([t["heading"] for t in data["traffic"]], dtype=np.float64)
    traffic_types = np.array([t["type"] for t in data["traffic"]], dtype=np.int32)
    return {
        "stop_line_centers": centers,
        "stop_lines": stop_lines,
        "stop_line_headings": stop_line_headings,
        "traffic_types": traffic_types,
        "num_traffic": len(centers),
    }


def fit_translation(src: np.ndarray, ref: np.ndarray, init=None, iterations: int = 50, max_points: int = 2000):
    """(t (2,), median residual [m]): translation-only ICP of src + t onto ref, two samplings of the same curves;
    callers reject large residuals rather than trust a bad fit."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    ref = np.asarray(ref, np.float64).reshape(-1, 2)
    if len(src) < 2 or len(ref) < 2:
        raise ValueError("fit_translation needs at least 2 points in src and ref")
    if len(src) > max_points:
        src = src[np.linspace(0, len(src) - 1, max_points).astype(int)]

    from scipy.spatial import cKDTree

    nearest = cKDTree(ref).query
    t = np.asarray(init, np.float64) if init is not None else ref.mean(0) - src.mean(0)
    for _ in range(iterations):
        dist, j = nearest(src + t)
        step = (ref[j] - (src + t)).mean(0)
        t = t + step
        if float(np.hypot(*step)) < 1e-6:
            break
    dist, _ = nearest(src + t)
    return t, float(np.median(dist))


def read_bin_lane_points(bin_path: Path, max_points: int = 200_000) -> np.ndarray:
    """(N, 2) float64 lane-centerline vertices of a bin (road types 0-9), uniformly subsampled to max_points."""
    data = _mbin.read_bin(Path(bin_path))
    pts = [
        np.stack([np.asarray(r["x"], np.float64), np.asarray(r["y"], np.float64)], axis=1)
        for r in data["roads"]
        if 0 <= r["type"] <= 9
    ]
    if not pts:
        raise ValueError(f"{bin_path}: no lane elements to register against")
    out = np.concatenate(pts)
    if len(out) > max_points:
        out = out[np.linspace(0, len(out) - 1, max_points).astype(int)]
    return out


def coarse_translation_vote(src: np.ndarray, ref: np.ndarray, grid: float = 5.0) -> np.ndarray:
    """Translation between two sparse constellations of the same landmarks: every ref - src pair votes on a grid-m
    cell, the median of the diffs near the winner is a few-metre init for fit_translation."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    ref = np.asarray(ref, np.float64).reshape(-1, 2)
    if len(src) < 3 or len(ref) < 3:
        raise ValueError("coarse_translation_vote needs at least 3 landmarks per side")
    if len(src) > 3000:
        src = src[np.linspace(0, len(src) - 1, 3000).astype(int)]
    if len(ref) > 3000:
        ref = ref[np.linspace(0, len(ref) - 1, 3000).astype(int)]

    d = (ref[None, :, :] - src[:, None, :]).reshape(-1, 2)
    keys = np.round(d / grid).astype(np.int64)
    hashed = keys[:, 0] * 1_000_003 + keys[:, 1]
    vals, counts = np.unique(hashed, return_counts=True)
    t0 = d[hashed == vals[counts.argmax()]].mean(axis=0)
    near = d[np.hypot(d[:, 0] - t0[0], d[:, 1] - t0[1]) < 1.5 * grid]
    return np.median(near, axis=0)


LIGHT_MATCH_MAX_DIST_M = 15.0  # nuPlan connector entries sit up to 14 m before/after the bin stop line
# keeps skewed stop lines, rejects cross-street (90) and oncoming (180)
LIGHT_MATCH_MAX_HEADING_DIFF_RAD = np.radians(75.0)


def point_to_segment_distance(px: float, py: float, segments: np.ndarray) -> np.ndarray:
    """Distance from (px, py) to each segment of a (K, 4) [x1, y1, x2, y2] array."""
    ax, ay, bx, by = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
    dx, dy = bx - ax, by - ay
    length_sq = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / length_sq, 0.0, 1.0)
    return np.hypot(ax + t * dx - px, ay + t * dy - py)


def match_connectors_to_stop_lines(
    connector_entries: dict,
    transform: NuPlanTransform,
    stop_lines: np.ndarray,
    stop_line_headings: np.ndarray,
    traffic_types: np.ndarray,
    max_dist_m: float = LIGHT_MATCH_MAX_DIST_M,
    max_heading_diff_rad: float = LIGHT_MATCH_MAX_HEADING_DIFF_RAD,
) -> dict:
    """{lane_connector_id: bin traffic element idx}: each connector entry pose (x, y, heading in nuPlan map coordinates)
    takes the nearest light stop line within max_dist_m facing its travel direction, measured to the line segment."""
    mapping = {}
    is_light = traffic_types == TRAFFIC_TYPE_LIGHT
    for cid, (x, y, heading) in connector_entries.items():
        bx, by = transform.loc_to_bin(x, y)
        heading_diff = np.abs((stop_line_headings - heading + np.pi) % (2.0 * np.pi) - np.pi)
        candidate = is_light & (heading_diff <= max_heading_diff_rad)
        if not candidate.any():
            continue
        dist = np.where(candidate, point_to_segment_distance(bx, by, stop_lines), np.inf)
        j = int(dist.argmin())
        if dist[j] <= max_dist_m:
            mapping[str(cid)] = j
    return mapping


MOVING_PARTNER_TYPES = ("VEHICLE", "PEDESTRIAN", "BICYCLE")  # the only types the training bins turn into agents
STATIC_PARTNER_TYPES = ("TRAFFIC_CONE", "BARRIER", "CZONE_SIGN", "GENERIC_OBJECT")
# lane connectors match by centerline (lanes by polygon): reach every connector whose polygon can touch the box
STATIC_PARTNER_LANE_QUERY_RADIUS_M = 10.0


def lane_polygon_overlaps(lane, box) -> bool:
    try:
        polygon = lane.polygon
    except AssertionError:  # devkit: lane connector without a scaled-width polygon row, nothing to hit
        return False
    return bool(polygon.intersects(box))


def partner_tracked_objects(tracked_objects, map_api, static_on_lane: dict):
    """Objects the shadow env sees: moving agents, and static clutter only where its box overlaps a lane or lane
    connector polygon (not the whole drivable area); static_on_lane caches the test per track token."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    lane_layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    kept = []
    for obj in tracked_objects:
        type_name = obj.tracked_object_type.name
        if type_name in MOVING_PARTNER_TYPES:
            kept.append(obj)
            continue
        if type_name not in STATIC_PARTNER_TYPES:
            continue
        on_lane = static_on_lane.get(obj.track_token)
        if on_lane is None:
            box = obj.box.geometry
            proximal = map_api.get_proximal_map_objects(
                Point2D(obj.center.x, obj.center.y), STATIC_PARTNER_LANE_QUERY_RADIUS_M, lane_layers
            )
            on_lane = any(lane_polygon_overlaps(lane, box) for layer in lane_layers for lane in proximal[layer])
            static_on_lane[obj.track_token] = on_lane
        if on_lane:
            kept.append(obj)
    return kept


VRU_PARTNER_TYPE_IDS = (_NUPLAN_AGENT_TYPE["PEDESTRIAN"], _NUPLAN_AGENT_TYPE["BICYCLE"])


def pad_vru_partner_sizes(types, lengths, widths, pad_m: float):
    """Pedestrian and bicycle boxes grow by pad_m in length and width (0.8 puts every nuPlan pedestrian
    and bicycle strictly above the 0.8 x 0.8 m training minimum). Other types untouched."""
    vru = np.isin(types, VRU_PARTNER_TYPE_IDS)
    lengths = np.where(vru, lengths + pad_m, lengths).astype(np.float32)
    widths = np.where(vru, widths + pad_m, widths).astype(np.float32)
    return lengths, widths


def nearest_partner_slots(x, y, ego_x: float, ego_y: float, slots: int):
    """Row order for at most `slots` partners: the devkit order when they fit, else nearest to the ego first."""
    if len(x) <= slots:
        return np.arange(len(x))
    distance_sq = (np.asarray(x, np.float64) - ego_x) ** 2 + (np.asarray(y, np.float64) - ego_y) ** 2
    return np.argsort(distance_sq, kind="stable")[:slots]


def tracked_objects_to_arrays(tracked_objects, transform: NuPlanTransform):
    """nuPlan DetectionsTracks objects -> (x, y, heading, vx, vy, types, lengths, widths) arrays in the bin frame."""
    x, y, h, vx, vy, tp, ln, wd = [], [], [], [], [], [], [], []
    for obj in tracked_objects:
        bx, by = transform.loc_to_bin(obj.center.x, obj.center.y)
        v = getattr(obj, "velocity", None)
        x.append(bx)
        y.append(by)
        h.append(float(obj.center.heading))
        vx.append(float(v.x) if v is not None else 0.0)
        vy.append(float(v.y) if v is not None else 0.0)
        tp.append(_NUPLAN_AGENT_TYPE.get(obj.tracked_object_type.name, 1))
        ln.append(float(obj.box.length))
        wd.append(float(obj.box.width))
    return (
        np.array(x, np.float32),
        np.array(y, np.float32),
        np.array(h, np.float32),
        np.array(vx, np.float32),
        np.array(vy, np.float32),
        np.array(tp, np.int32),
        np.array(ln, np.float32),
        np.array(wd, np.float32),
    )


_LIGHT_RESTRICTIVENESS = {LIGHT_STATE_RED: 0, LIGHT_STATE_YELLOW: 1, LIGHT_STATE_GREEN: 2}


def traffic_light_states(
    traffic_light_data, connector_map: dict, num_traffic: int, route_connector_ids=()
) -> np.ndarray:
    """Per-element light states from PlannerInput.traffic_light_data: connectors share stop-line elements, so a route
    element takes its route connector's status, any other the most restrictive reported one; unreported = GREEN."""
    states = np.full(num_traffic, LIGHT_STATE_GREEN, dtype=np.int32)
    on_route = set(str(cid) for cid in route_connector_ids)
    route_elements = {connector_map[cid] for cid in on_route if cid in connector_map}
    for tl in traffic_light_data:
        cid = str(tl.lane_connector_id)
        j = connector_map.get(cid)
        if j is None:
            continue
        state = _NUPLAN_LIGHT_STATE[tl.status.name]
        if cid in on_route:
            states[j] = state
        elif j not in route_elements and _LIGHT_RESTRICTIVENESS[state] < _LIGHT_RESTRICTIVENESS[int(states[j])]:
            states[j] = state
    return states
