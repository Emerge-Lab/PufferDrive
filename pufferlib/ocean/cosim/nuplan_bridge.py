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

# nuPlan TrafficLightStatusType -> PufferDrive light enum
# (datatypes.h: UNKNOWN=0 RED=1 YELLOW=2 GREEN=3 OFF=4)
_NUPLAN_LIGHT_STATE = {"RED": 1, "YELLOW": 2, "GREEN": 3, "UNKNOWN": 0}

FAR_AWAY = 1.0e6  # park surplus PufferDrive agents out of observation range

# Env/obs layout the carla_combined gigaflow policy expects at eval time (same
# role as CARLA_ARCH in cosim/carla/world_sync.py). Override per-checkpoint via
# the planner's `env_overrides` when the training config differs.
DEFAULT_ARCH = dict(
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
    target_type="static",
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


def _road_entry(road_id: int, road_type: int, xy: np.ndarray,
                entry=None, exit_=None, speed_limit=-1.0) -> dict:
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


def route_centerline(map_api, route_roadblock_ids, start_x: float, start_y: float) -> np.ndarray:
    """Greedy lane walk through the route roadblocks -> (N, 2) centerline in
    nuPlan map coordinates (PDM-style: nearest lane in the first block, then
    prefer connected lanes block to block). Fallback goal source for when the
    scenario has no usable logged trajectory (see `expert_route_xy`)."""
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    blocks = []
    for rid in route_roadblock_ids:
        block = map_api.get_map_object(str(rid), SemanticMapLayer.ROADBLOCK) or map_api.get_map_object(
            str(rid), SemanticMapLayer.ROADBLOCK_CONNECTOR
        )
        if block is not None and block.interior_edges:
            blocks.append(block)
    if not blocks:
        return np.zeros((0, 2))

    def _lane_pts(lane):
        return np.array([[p.x, p.y] for p in lane.baseline_path.discrete_path])

    def _nearest(lanes, x, y):
        return min(lanes, key=lambda l: float(np.min(np.hypot(*(_lane_pts(l) - (x, y)).T))))

    current = _nearest(blocks[0].interior_edges, start_x, start_y)
    path = [current]
    for block in blocks[1:]:
        ids = {l.id for l in block.interior_edges}
        nxt = next((o for o in current.outgoing_edges if o.id in ids), None)
        if nxt is None:  # route gap (e.g. lane-change requirement): jump to nearest
            end = _lane_pts(current)[-1]
            nxt = _nearest(block.interior_edges, end[0], end[1])
        path.append(nxt)
        current = nxt

    lane_points = [_lane_pts(l) for l in path]
    # The first lane's baseline path starts at the lane's own beginning, which
    # is usually behind the ego (ego is partway along it) -> trim to the point
    # nearest the ego so goals_along doesn't place goals behind the vehicle.
    start_idx = int(np.argmin(np.hypot(*(lane_points[0] - (start_x, start_y)).T)))
    lane_points[0] = lane_points[0][start_idx:]
    return np.concatenate(lane_points, axis=0)


def goals_along(centerline: np.ndarray, spacing: float) -> np.ndarray:
    """(N, 2) polyline -> fixed goal sequence every `spacing` meters (+ endpoint)."""
    if len(centerline) < 2:
        return centerline.copy()
    seg = np.hypot(*np.diff(centerline, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    goals = [centerline[int(np.searchsorted(cum, s))] for s in np.arange(spacing, cum[-1], spacing)]
    goals.append(centerline[-1])
    return np.asarray(goals, dtype=np.float64)


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
        [[0.5 * (t["stop_line"][0] + t["stop_line"][3]),
          0.5 * (t["stop_line"][1] + t["stop_line"][4])] for t in data["traffic"]],
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

    return {"origin": origin, "stop_line_centers": centers,
            "num_traffic": len(centers), "ego_t0": ego_t0, "ego_traj": ego_traj}


def fit_translation(src: np.ndarray, ref: np.ndarray, init=None,
                    iterations: int = 50, max_points: int = 2000):
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
    pts = [np.stack([np.asarray(r["x"], np.float64), np.asarray(r["y"], np.float64)], axis=1)
           for r in data["roads"] if 0 <= r["type"] <= 9]
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


def match_connectors_to_stop_lines(connector_entries: dict, transform: NuPlanTransform,
                                   stop_line_centers: np.ndarray, max_dist_m: float = 10.0) -> dict:
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
        if d2[j] <= max_dist_m ** 2:
            mapping[str(cid)] = j
    return mapping


def tracked_objects_to_arrays(tracked_objects, transform: NuPlanTransform, first_slot: int = 1):
    """nuPlan DetectionsTracks agents -> (idx, x, y, z, h, vx, vy, types, lengths, widths)
    arrays in the bin frame, filling PufferDrive slots first_slot..N."""
    idx, x, y, z, h, vx, vy, tp, ln, wd = [], [], [], [], [], [], [], [], [], []
    for j, obj in enumerate(tracked_objects):
        bx, by = transform.loc_to_bin(obj.center.x, obj.center.y)
        v = getattr(obj, "velocity", None)
        idx.append(first_slot + j)
        x.append(bx); y.append(by); z.append(0.0); h.append(float(obj.center.heading))
        vx.append(float(v.x) if v is not None else 0.0)
        vy.append(float(v.y) if v is not None else 0.0)
        tp.append(_NUPLAN_AGENT_TYPE.get(obj.tracked_object_type.name, 1))
        ln.append(float(obj.box.length)); wd.append(float(obj.box.width))
    return (np.array(idx, np.int32), np.array(x, np.float32), np.array(y, np.float32),
            np.array(z, np.float32), np.array(h, np.float32), np.array(vx, np.float32),
            np.array(vy, np.float32), np.array(tp, np.int32),
            np.array(ln, np.float32), np.array(wd, np.float32))


def traffic_light_states(traffic_light_data, connector_map: dict, num_traffic: int) -> np.ndarray:
    """PlannerInput.traffic_light_data -> per-element state array for
    set_traffic_light_states. Unlisted elements stay UNKNOWN (0)."""
    states = np.zeros(num_traffic, dtype=np.int32)
    for tl in traffic_light_data:
        j = connector_map.get(str(tl.lane_connector_id))
        if j is not None:
            states[j] = _NUPLAN_LIGHT_STATE.get(tl.status.name, 0)
    return states
