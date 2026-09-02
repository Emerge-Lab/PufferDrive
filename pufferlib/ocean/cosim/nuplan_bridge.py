"""nuPlan <-> PufferDrive map conversion + state readback for co-simulation.

Unlike CARLA (see carla_bridge.py), nuPlan is already a right-handed, z-up,
meters, radians frame — no reflection is needed. The only transform is a
translation: nuPlan map coordinates are large (UTM-ish, ~1e5-1e6 m) and the
PufferDrive bins store float32, so each bin is translated by an `origin`
recovered per city (registered against the GPKG map, see `planner.py`'s
`_city_bin_origin`) or read from a stored `centroid`:

    bin_x = nuplan_x - origin_x        bin_y = nuplan_y - origin_y
    bin_heading = nuplan_heading       bin_v = nuplan_v (global frame)

Bins are pre-converted per-city map-only bins (`city_bin_dir`, see
`planner.py`), holding only the road graph (lanes / boundaries / crosswalks /
traffic-light stop lines) — no logged trajectories. Agent states are streamed
from the nuPlan simulation every tick, exactly like the CARLA co-sim streams
from CARLA. Traffic lights are matched geometrically at planner-build time
(`match_connectors_to_stop_lines`) so live traffic-light status
(`PlannerInput.traffic_light_data`) can be written with set_traffic_light_states.

Nothing in this module imports `nuplan` at module scope, so the pure-python bin
writer stays testable from an env without the nuplan-devkit installed.
"""

from pathlib import Path

import numpy as np

import data_utils.mirror_map_bin as _mbin

# PufferDrive road types (datatypes.h): lanes 0-9, road lines 10-19, edges 20-29
LANE_SURFACE_STREET = 2
ROAD_LINE_BROKEN_WHITE = 11
ROAD_EDGE_BOUNDARY = 21
MISC_CROSSWALK = 31
TRAFFIC_TYPE_LIGHT = 1  # matches the CARLA bins' traffic-element type

# nuPlan TrackedObjectType -> PufferDrive agent type (1=vehicle 2=ped 3=cyclist)
_NUPLAN_AGENT_TYPE = {"VEHICLE": 1, "PEDESTRIAN": 2, "BICYCLE": 3}

# nuPlan TrafficLightStatusType -> PufferDrive light enum (datatypes.h: RED=1 YELLOW=2 GREEN=3).
# Training never produces UNKNOWN (0), so an unknown or unreported light is treated as GREEN.
LIGHT_STATE_GREEN = 3
_NUPLAN_LIGHT_STATE = {"RED": 1, "YELLOW": 2, "GREEN": LIGHT_STATE_GREEN, "UNKNOWN": LIGHT_STATE_GREEN}

FAR_AWAY = 1.0e6  # park surplus PufferDrive agents out of observation range
ROUTE_SEARCH_DEPTH_BLOCKS = 30  # PDM's Dijkstra window over route roadblocks

# Env/obs layout the carla_combined gigaflow policy expects at eval time (a
# fallback for shadow_env_kwargs' checkpoint-config adoption, see
# cosim/arch.py -- only used for keys the checkpoint config doesn't set, i.e.
# chiefly the no-checkpoint dummy run). Override per-checkpoint via the
# planner's `env_overrides` when the training config differs.
DEFAULT_ARCH = dict(
    goal_source="external",  # route goal windows are pushed by the planner
    num_goals=3,
    obs_slots_lane_n=80,
    obs_slots_boundary_n=40,
    obs_slots_partners_n=16,
    obs_slots_traffic_controls_n=4,
    obs_range_partner_m=200.0,
    obs_range_road_front_m=200.0,
    obs_range_road_behind_m=40.0,
    obs_range_road_side_m=50.0,
    obs_range_traffic_control_m=100.0,
    obs_norm_xy_offset_m=200.0,
    obs_norm_goal_offset_m=200.0,
    obs_norm_road_seg_length_m=10.0,
    obs_norm_road_seg_width_m=5.0,
    obs_norm_veh_length_m=15.0,
    obs_norm_veh_width_m=10.0,
    reward_conditioning=True,
    goal_speed=20.0,  # reward conditioning: arrive at goals at up to 20 m/s
    goal_radius=6.0,
    dynamics_model="jerk",
    dt=0.1,  # lockstep with nuPlan's 10 Hz planner interval
)


class NuPlanTransform:
    """Translation-only nuPlan <-> bin-frame transform for one scenario."""

    def __init__(self, origin_x: float, origin_y: float):
        self.ox = float(origin_x)
        self.oy = float(origin_y)

    def loc_to_bin(self, x, y):
        return x - self.ox, y - self.oy

    def bin_to_loc(self, bx, by):
        return bx + self.ox, by + self.oy


def _polyline_heading(xy: np.ndarray) -> np.ndarray:
    """Per-point forward-tangent headings; last point repeats."""
    d = np.diff(xy, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    return np.append(h, h[-1] if len(h) else 0.0).astype(np.float32)


def _road_entry(road_id: int, road_type: int, xy: np.ndarray, entry=None, exit_=None, speed_limit=-1.0) -> dict:
    """One road element in mirror_map_bin's dict schema."""
    xy = np.asarray(xy, dtype=np.float32)
    S = len(xy)
    e = {
        "id": road_id,
        "type": road_type,
        "S": S,
        "x": tuple(xy[:, 0].tolist()),
        "y": tuple(xy[:, 1].tolist()),
        "z": tuple([0.0] * S),
        "headings": tuple(_polyline_heading(xy).tolist()),
    }
    if 0 <= road_type <= 9:
        seg = np.hypot(*np.diff(xy, axis=0).T) if S > 1 else np.array([0.0])
        cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)[:S]
        e["entry_lanes"] = tuple(entry or [])
        e["exit_lanes"] = tuple(exit_ or [])
        e["speed_limit"] = float(speed_limit if speed_limit is not None else -1.0)
        e["length"] = float(cum[-1])
        e["cum_lengths"] = tuple(cum.tolist())
    return e


def write_drive_bin(roads, traffic, out_path: Path, scenario_id: str, centroid=(0.0, 0.0, 0.0)):
    """Write a map-only PufferDrive bin (0 agents, 0 objects, empty lane graph),
    like the CARLA town bins. `roads`/`traffic` follow mirror_map_bin's schema."""
    data = {
        "agents": [],
        "roads": roads,
        "traffic": traffic,
        "objects": [],
        "lane_graph": {"n": 0, "lane_ids": (), "distances": ()},
        "scenario_id": scenario_id.encode("utf-8")[:128].ljust(128, b"\0"),
        "dataset_name": b"nuplan".ljust(32, b"\0"),
        "log_length": 0,
        "log_dt": 0.0,
        "objects_of_interest": (),
        "tracks_to_predict": (),
        "centroid": (float(centroid[0]), float(centroid[1]), float(centroid[2])),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _mbin.write_bin(data, out_path)


def expert_route_xy(scenario) -> np.ndarray:
    """(N, 2) logged ego-center trajectory for the whole scenario window, in
    nuPlan map coordinates -- the ground-truth path the expert actually drove.
    Preferred goal source over `route_centerline`: a lane-graph walk only knows
    which roadblocks are on-route, not which lane within a multi-lane block or
    which fork at a split, so it can place goals on a parallel or turning lane;
    the logged trajectory has neither ambiguity. Starts at the scenario's
    initial ego state (iteration 0), so no start-position trim is needed
    (unlike route_centerline, whose first lane can start well behind the ego).
    Duck-typed on `EgoState`-like objects (`.center.x/.y`), not a real nuPlan
    import, so this stays testable without the devkit installed."""
    return np.array(
        [[s.center.x, s.center.y] for s in scenario.get_expert_ego_trajectory()],
        dtype=np.float64,
    ).reshape(-1, 2)


def route_centerline(map_api, route_roadblock_ids, start_x: float, start_y: float, start_heading: float):
    """Lane-graph route through the route roadblocks -> ((N, 2) centerline in
    nuPlan map coordinates, [lane ids along it]), PDM-style: the starting lane
    is the on-route lane under the ego with the smallest heading error (never
    a crossing lane of the same junction), then Dijkstra over the route's lane
    graph to the last roadblock. The first lane is trimmed to the point
    nearest the ego."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from carl_nuplan.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra

    blocks = []
    for rid in dict.fromkeys(route_roadblock_ids):
        block = map_api.get_map_object(str(rid), SemanticMapLayer.ROADBLOCK) or map_api.get_map_object(
            str(rid), SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        if block is not None and block.interior_edges:
            blocks.append(block)
    if not blocks:
        return np.zeros((0, 2)), []
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
    # The ego's own lane may not connect to the next route block (wrong lane for the turn, PDM's
    # route correction keeps the block): fall back to the sibling lanes of its block, nearest first.
    siblings = sorted((lane for lane in blocks[start_block_idx].interior_edges if lane not in containing), key=_distance)
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

    lane_points = [_lane_pts(lane) for lane in path]
    start_idx = int(np.argmin(np.hypot(*(lane_points[0] - (start_x, start_y)).T)))
    lane_points[0] = lane_points[0][start_idx:]
    return np.concatenate(lane_points, axis=0), [str(lane.id) for lane in path]

def indices_along(polyline: np.ndarray, spacing: float) -> np.ndarray:
    """(N, 2) polyline -> vertex indices every `spacing` meters of arc length (+ the endpoint)."""
    if len(polyline) < 2:
        return np.arange(len(polyline))
    seg = np.hypot(*np.diff(polyline, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    indices = [int(np.searchsorted(cum, s)) for s in np.arange(spacing, cum[-1], spacing)]
    indices.append(len(polyline) - 1)
    return np.asarray(indices, dtype=np.int64)


def goals_along(centerline: np.ndarray, spacing: float) -> np.ndarray:
    """(N, 2) polyline -> fixed goal sequence every `spacing` meters (+ endpoint)."""
    return np.asarray(centerline, dtype=np.float64)[indices_along(centerline, spacing)]


GOAL_LANE_SNAP_MAX_DIST_M = 6.0  # same radius as Drive's GOAL_LANE_SNAP_MAX_DIST_M
GOAL_LANE_SNAP_MAX_HEADING_ERROR_RAD = np.pi / 4


def snap_to_lane_center(map_api, x: float, y: float, heading: float):
    """Nearest pose on a lane/lane-connector baseline within the snap radius whose direction agrees with
    `heading` (co-directional, so neither oncoming nor crossing lanes), or None if no lane qualifies."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    point = Point2D(x, y)
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    best_pose, best_dist = None, GOAL_LANE_SNAP_MAX_DIST_M
    for lanes in map_api.get_proximal_map_objects(point, GOAL_LANE_SNAP_MAX_DIST_M, layers).values():
        for lane in lanes:
            pose = lane.baseline_path.get_nearest_pose_from_position(point)
            heading_error = abs((pose.heading - heading + np.pi) % (2.0 * np.pi) - np.pi)
            if heading_error > GOAL_LANE_SNAP_MAX_HEADING_ERROR_RAD:
                continue
            dist = float(np.hypot(pose.x - x, pose.y - y))
            if dist < best_dist:
                best_pose, best_dist = pose, dist
    return best_pose


def logged_ego_goals(scenario, map_api, spacing: float):
    """Logged ego path sampled every `spacing` m (+ endpoint), each sample snapped to the nearest
    co-directional lane center so the expert's exact pose never reaches the policy. Returns
    ((N, 2) xy in nuPlan map coordinates, (N,) headings, snapped count); samples without a lane
    within the snap radius keep the raw logged pose."""
    states = [scenario.get_ego_state_at_iteration(i) for i in range(scenario.get_number_of_iterations())]
    path = np.array([[s.center.x, s.center.y] for s in states], dtype=np.float64)
    headings = np.array([s.center.heading for s in states], dtype=np.float64)
    indices = indices_along(path, spacing)
    xy = path[indices].copy()
    goal_headings = headings[indices].copy()
    snapped_count = 0
    for k in range(len(indices)):
        pose = snap_to_lane_center(map_api, xy[k, 0], xy[k, 1], goal_headings[k])
        if pose is None:
            continue
        xy[k] = (pose.x, pose.y)
        goal_headings[k] = pose.heading
        snapped_count += 1
    return xy, goal_headings, snapped_count


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
    """Read the co-sim-relevant geometry out of an existing PufferDrive bin
    (city_bin_dir map-only bins or pre-converted training-format bins):

      origin            (ox, oy) from the stored centroid, or None if unset
                        (nuPlan map coords are ~1e5-1e6 m, so a (0, 0)/missing
                        centroid means "not stored", never a real origin)
      stop_line_centers (K, 2) bin-frame stop-line midpoints of all traffic
                        elements, for geometric light matching
      num_traffic       K, the size set_traffic_light_states expects
      ego_t0            (x, y, heading) of agent 0 at its first valid log step,
                        or None if the bin has no agents — fallback origin
                        recovery: origin = nuplan_ego_t0_xy - bin_ego_t0_xy
      ego_traj          (N, 2) agent-0 valid trajectory in the bin frame, or
                        None — fallback origin recovery for whole-log bins:
                        fit_translation against the log's UTM ego trajectory
    """
    data = _mbin.read_bin(Path(bin_path))

    centroid = data.get("centroid")
    origin = None
    if centroid is not None and (abs(centroid[0]) > 1.0 or abs(centroid[1]) > 1.0):
        origin = (float(centroid[0]), float(centroid[1]))

    centers = np.array(
        [
            [0.5 * (t["stop_line"][0] + t["stop_line"][3]), 0.5 * (t["stop_line"][1] + t["stop_line"][4])]
            for t in data["traffic"]
        ],
        dtype=np.float64,
    ).reshape(-1, 2)

    ego_t0, ego_traj = None, None
    if data["agents"]:
        cols = data["agents"][0]["cols"]
        valid = np.asarray(cols["valid"]) > 0
        if valid.any():
            xs = np.asarray(cols["x"], np.float64)[valid]
            ys = np.asarray(cols["y"], np.float64)[valid]
            hs = np.asarray(cols["h"], np.float64)[valid]
            ego_t0 = (float(xs[0]), float(ys[0]), float(hs[0]))
            ego_traj = np.stack([xs, ys], axis=1)

    return {
        "origin": origin,
        "stop_line_centers": centers,
        "num_traffic": len(centers),
        "ego_t0": ego_t0,
        "ego_traj": ego_traj,
    }


def fit_translation(src: np.ndarray, ref: np.ndarray, init=None, iterations: int = 50, max_points: int = 2000):
    """Translation-only alignment of two samplings of the SAME physical curve
    (e.g. a bin's centered ego trajectory vs the log's UTM ego trajectory):
    find t minimizing nearest-neighbor distance from src + t onto ref.

    Because the transform is a pure translation and both point sets trace the
    identical path, the centroid-difference initialization lands within meters
    and the ICP iterations converge to sub-centimeter. Returns
    (t (2,) float64, median residual in meters) — callers should reject large
    residuals rather than trust a bad fit."""
    src = np.asarray(src, np.float64).reshape(-1, 2)
    ref = np.asarray(ref, np.float64).reshape(-1, 2)
    if len(src) < 2 or len(ref) < 2:
        raise ValueError("fit_translation needs at least 2 points in src and ref")
    if len(src) > max_points:
        src = src[np.linspace(0, len(src) - 1, max_points).astype(int)]

    try:
        from scipy.spatial import cKDTree

        nearest = cKDTree(ref).query
    except ImportError:  # brute force on a subsample
        if len(ref) > 5000:
            ref = ref[np.linspace(0, len(ref) - 1, 5000).astype(int)]

        def nearest(pts):
            d2 = ((pts[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
            j = d2.argmin(1)
            return np.sqrt(d2[np.arange(len(pts)), j]), j

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
    """(N, 2) float64 lane-centerline vertices of a bin (road types 0-9), the
    bin-side point cloud for whole-city origin registration. Uniformly
    subsampled to max_points."""
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
    """Global translation-only registration of two sparse constellations of the
    SAME landmarks (e.g. a city bin's traffic stop-line centers vs the GPKG
    map's stop polygons): every (ref - src) pair difference votes on a `grid`-m
    cell; the true translation collects one vote per real landmark match while
    mismatched pairs scatter. Returns the median of the diffs near the winning
    cell — a few-meter-accurate init for fit_translation, found with no prior."""
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


def match_connectors_to_stop_lines(
    connector_entries: dict, transform: NuPlanTransform, stop_line_centers: np.ndarray, max_dist_m: float = 10.0
) -> dict:
    """Geometric traffic-light mapping, the runtime replacement for the
    `.tl.json` sidecar (works for any bin source). `connector_entries` maps
    lane_connector_id(str) -> (x, y) entry point in nuPlan map coordinates.
    Returns {lane_connector_id: bin traffic-element idx}, skipping connectors
    with no stop line within max_dist_m."""
    mapping = {}
    if not len(stop_line_centers):
        return mapping
    for cid, (x, y) in connector_entries.items():
        bx, by = transform.loc_to_bin(x, y)
        d2 = (stop_line_centers[:, 0] - bx) ** 2 + (stop_line_centers[:, 1] - by) ** 2
        j = int(d2.argmin())
        if d2[j] <= max_dist_m**2:
            mapping[str(cid)] = j
    return mapping


MOVING_PARTNER_TYPES = ("VEHICLE", "PEDESTRIAN", "BICYCLE")  # the only types the training bins turn into agents
STATIC_PARTNER_TYPES = ("TRAFFIC_CONE", "BARRIER", "CZONE_SIGN", "GENERIC_OBJECT")


def partner_tracked_objects(tracked_objects, map_api, static_on_drivable: dict):
    """Objects the shadow env should see: moving agents always; static clutter (cones, barriers, signs,
    generic objects) only when it stands on the drivable area, where nuPlan scores a collision with it.
    Off-road poles would otherwise fill the nearest-N partner slots as stopped 0.3 m vehicles the policy
    never saw in training. static_on_drivable: track_token -> bool, filled here (statics never move)."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    kept = []
    for obj in tracked_objects:
        type_name = obj.tracked_object_type.name
        if type_name in MOVING_PARTNER_TYPES:
            kept.append(obj)
            continue
        if type_name not in STATIC_PARTNER_TYPES:
            continue
        on_drivable = static_on_drivable.get(obj.track_token)
        if on_drivable is None:
            on_drivable = bool(map_api.is_in_layer(Point2D(obj.center.x, obj.center.y), SemanticMapLayer.DRIVABLE_AREA))
            static_on_drivable[obj.track_token] = on_drivable
        if on_drivable:
            kept.append(obj)
    return kept


def tracked_objects_to_arrays(tracked_objects, transform: NuPlanTransform, first_slot: int = 1):
    """nuPlan DetectionsTracks agents -> (idx, x, y, z, h, vx, vy, types, lengths, widths)
    arrays in the bin frame, filling PufferDrive slots first_slot..N."""
    idx, x, y, z, h, vx, vy, tp, ln, wd = [], [], [], [], [], [], [], [], [], []
    for j, obj in enumerate(tracked_objects):
        bx, by = transform.loc_to_bin(obj.center.x, obj.center.y)
        v = getattr(obj, "velocity", None)
        idx.append(first_slot + j)
        x.append(bx)
        y.append(by)
        z.append(0.0)
        h.append(float(obj.center.heading))
        vx.append(float(v.x) if v is not None else 0.0)
        vy.append(float(v.y) if v is not None else 0.0)
        tp.append(_NUPLAN_AGENT_TYPE.get(obj.tracked_object_type.name, 1))
        ln.append(float(obj.box.length))
        wd.append(float(obj.box.width))
    return (
        np.array(idx, np.int32),
        np.array(x, np.float32),
        np.array(y, np.float32),
        np.array(z, np.float32),
        np.array(h, np.float32),
        np.array(vx, np.float32),
        np.array(vy, np.float32),
        np.array(tp, np.int32),
        np.array(ln, np.float32),
        np.array(wd, np.float32),
    )


_LIGHT_RESTRICTIVENESS = {1: 0, 2: 1, LIGHT_STATE_GREEN: 2}  # RED < YELLOW < GREEN


def traffic_light_states(traffic_light_data, connector_map: dict, num_traffic: int, route_connector_ids=()) -> np.ndarray:
    """PlannerInput.traffic_light_data -> per-element state array for
    set_traffic_light_states. Several lane connectors (straight/left/right
    from one lane) share one stop-line element: an element on the ego's route
    is decided by its route connector alone, any other element by the most
    restrictive reported status. Unreported means GREEN."""
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
