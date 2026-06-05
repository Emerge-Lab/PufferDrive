"""Generate a synthetic 4-lane bottleneck scenario as a PufferDrive .bin map.

Scene (all in meters, road runs along +x):
  - A 200 m straight road with 4 lanes (lane 0 bottom ... lane 3 top).
  - 20 controllable vehicles spawned at the back (5 per lane).
  - A wall of static 2x2 m cars at x = BOTTLENECK_X (150 m from the far end)
    blocking the bottom 3 lanes; the top lane (lane 3) stays free.
  - Every controllable agent gets 3 replay waypoints: the first in the free
    lane at the bottleneck, the other two past it on the far side.

Waypoints are NOT stored explicitly: in replay mode the C sim samples each
agent's logged trajectory at indices t = (g+1)*(T-1)/num_target_waypoints for
g in 0..num_target_waypoints-1 (drive.h set_active_agents). With T = 200 and
num_target_waypoints = 3 those indices are 66, 132, 199. We therefore author a
per-agent reference trajectory that passes through (spawn -> wp1 -> wp2 -> wp3)
at indices (0, 66, 132, 199); the policy drives the real motion, the trajectory
only fixes the spawn pose and the 3 goals.

Static cars are ordinary VEHICLE agents with mark_as_expert=1 and a constant
trajectory: they are created but not policy-controlled (replayed in place), so
they act as fixed collidable obstacles. They are listed AFTER the 20
controllable agents so a small num_max_agents budget still creates all 20.

Binary layout mirrors load_map_binary() in drive.h (little-endian); lanes carry
speed_limit, length and cum_lengths (data_utils/json_to_bin.py is stale and
omits the last two).

Usage:
    python data_utils/make_bottleneck_map.py --output /workspace/data/bottleneck
"""

import argparse
import struct
from pathlib import Path

import numpy as np

# ── road / scene geometry ─────────────────────────────────────────────────────
ROAD_LENGTH_M = 200.0
# Extra road behind the spawn end so the rearmost cars aren't sitting on the map
# boundary — the corridor keeps going behind where they start. Only the
# lane/edge/divider geometry extends; spawns, bottleneck and goals are unchanged.
ROAD_EXTENSION_M = 40.0
BASE_LANE_WIDTH_M = 3.5
NUM_LANES = 4
FREE_LANE_IDX = NUM_LANES - 1                  # top lane stays open
FREE_LANE_WIDTH_MULT = 1.5                      # the open lane is 1.5x wider
# Per-lane widths; only the free (top) lane is widened.
LANE_WIDTHS_M = [BASE_LANE_WIDTH_M * (FREE_LANE_WIDTH_MULT if i == FREE_LANE_IDX else 1.0)
                 for i in range(NUM_LANES)]
# Anchor the bottom edge so the blocked lanes (0..2) keep their positions; the
# free lane's extra width extends the road upward only. Boundaries run bottom
# edge -> ... -> top edge; lane centers are the midpoints.
ROAD_BOTTOM_Y = -(NUM_LANES / 2.0) * BASE_LANE_WIDTH_M       # -7.0
LANE_BOUNDARIES_Y = [ROAD_BOTTOM_Y]
for _w in LANE_WIDTHS_M:
    LANE_BOUNDARIES_Y.append(LANE_BOUNDARIES_Y[-1] + _w)
ROAD_TOP_Y = LANE_BOUNDARIES_Y[-1]                            # 8.75 with the 1.5x lane
LANE_CENTERS_Y = [(LANE_BOUNDARIES_Y[i] + LANE_BOUNDARIES_Y[i + 1]) / 2.0 for i in range(NUM_LANES)]
FREE_LANE_Y = LANE_CENTERS_Y[FREE_LANE_IDX]

BOTTLENECK_X = ROAD_LENGTH_M - 120.0           # 80 m: 120 m from the end

# ── controllable agents ───────────────────────────────────────────────────────
# 4 rows (one per lane) of 4 cars each = 16 queued cars.
CARS_PER_LANE = 4
SPAWN_LANE_INDICES = list(range(NUM_LANES))     # all 4 lanes -> 4 rows
NUM_CONTROLLED = CARS_PER_LANE * len(SPAWN_LANE_INDICES)  # 16
SPAWN_BACK_X = [2.0 + 10.0 * k for k in range(CARS_PER_LANE)]  # queue: 2,12,22,32
# Each controllable car gets a size drawn (deterministically) from these ranges,
# spanning compact cars up to small trucks. Max width stays under the lane width
# (3.5 m) and the merge clearance to the static blockers.
CAR_LENGTH_RANGE_M = (3.8, 6.0)
CAR_WIDTH_RANGE_M = (1.8, 2.4)
CAR_SIZE_SEED = 0                               # fixed -> reproducible map

# ── static blocker cars ───────────────────────────────────────────────────────
STATIC_SIZE_M = 2.0                             # 2x2 m
# One stopped car parked at the center of each of the bottom 3 lanes; the top
# lane (lane 3) is left free.
STATIC_Y = [LANE_CENTERS_Y[i] for i in range(NUM_LANES - 1)]  # lanes 0,1,2 centers
STATIC_ROWS_X = [BOTTLENECK_X]                  # single car per blocked lane

# ── orientation ───────────────────────────────────────────────────────────────
# True -> traffic flows right-to-left (along -x). Every x is mirrored about the
# road center (x -> ROAD_LENGTH_M - x); headings and velocities are recomputed
# from the mirrored positions, so lanes, agents and goals all stay consistent.
TRAVEL_RIGHT_TO_LEFT = True


def _orient_x(x):
    """Mirror an x coordinate (or array) about the road center when flowing R->L."""
    return (ROAD_LENGTH_M - x) if TRAVEL_RIGHT_TO_LEFT else x


# ── trajectory / episode ──────────────────────────────────────────────────────
T_STEPS = 600                                   # trajectory length == scenario length
DT_SECONDS = 0.1
NUM_TARGET_WAYPOINTS = 3                         # must match the policy's arch key
# True -> agents start at rest. set_start_position seeds an active agent's sim
# velocity from its logged velocity at the init step, so zeroing the spawn-step
# velocity makes the cars begin stopped (heading/goals unchanged; the policy
# accelerates them from a standstill).
SPAWN_STOPPED = True

# road-type constants (datatypes.h)
LANE_FREEWAY = 1
ROAD_EDGE_BOUNDARY = 21
ROAD_LINE_BROKEN_SINGLE_WHITE = 11
VEHICLE = 1

LANE_SAMPLE_SPACING_M = 5.0
METADATA_ID_BYTES = 128
METADATA_DATASET_BYTES = 32


def _waypoint_indices(traj_len: int, num_wp: int) -> list:
    """Indices the C sim samples for replay goals (drive.h set_active_agents)."""
    remaining = traj_len - 1
    return [min((g + 1) * remaining // num_wp, traj_len - 1) for g in range(num_wp)]


def _piecewise_trajectory(control_points, control_indices, traj_len):
    """Linear interpolation of (x,y) through control_points placed at
    control_indices (must start at 0 and end at traj_len-1)."""
    xs = np.empty(traj_len, dtype=np.float32)
    ys = np.empty(traj_len, dtype=np.float32)
    for seg in range(len(control_indices) - 1):
        i0, i1 = control_indices[seg], control_indices[seg + 1]
        (x0, y0), (x1, y1) = control_points[seg], control_points[seg + 1]
        n = i1 - i0
        for k in range(n + 1):
            f = k / n
            xs[i0 + k] = x0 + f * (x1 - x0)
            ys[i0 + k] = y0 + f * (y1 - y0)
    return xs, ys


def _trajectory_kinematics(xs, ys, dt):
    """Headings (tangent) and velocities from a position trajectory."""
    dx = np.gradient(xs).astype(np.float32)
    dy = np.gradient(ys).astype(np.float32)
    headings = np.arctan2(dy, dx).astype(np.float32)
    vxs = (dx / dt).astype(np.float32)
    vys = (dy / dt).astype(np.float32)
    return headings, vxs, vys


def _pack_agent(buf, agent_id, agent_type, xs, ys, heading, vxs, vys,
                length_m, width_m, mark_as_expert):
    traj_len = len(xs)
    zs = np.zeros(traj_len, dtype=np.float32)
    headings = np.full(traj_len, heading, dtype=np.float32) if np.isscalar(heading) else heading
    lengths = np.full(traj_len, length_m, dtype=np.float32)
    widths = np.full(traj_len, width_m, dtype=np.float32)
    heights = np.full(traj_len, 1.5, dtype=np.float32)
    valid = np.ones(traj_len, dtype=np.int32)

    buf.extend(struct.pack("<ii", agent_id, agent_type))
    buf.extend(struct.pack("<i", traj_len))
    for col in (xs.astype(np.float32), ys.astype(np.float32), zs, headings,
                vxs.astype(np.float32), vys.astype(np.float32), lengths, widths, heights):
        buf.extend(col.tobytes())
    buf.extend(valid.tobytes())

    # Dummy route ([-1]) so CONTROL_SDC_ONLY would still see a routed agent 0;
    # harmless in CONTROL_VEHICLES. route_gt_len = 0.
    buf.extend(struct.pack("<ii", 1, -1))   # route_length=1, route=[-1]
    buf.extend(struct.pack("<i", 0))        # route_gt_len

    # Goal stored in the bin = final logged position (replay re-derives the 3
    # waypoints from the trajectory, so this is only a fallback).
    buf.extend(struct.pack("<fff", float(xs[-1]), float(ys[-1]), 0.0))
    buf.extend(struct.pack("<i", int(mark_as_expert)))


def _pack_polyline(buf, road_id, road_type, xs, ys, is_lane):
    seg_len = len(xs)
    zs = np.zeros(seg_len, dtype=np.float32)
    dx = np.gradient(xs).astype(np.float32)
    dy = np.gradient(ys).astype(np.float32)
    headings = np.arctan2(dy, dx).astype(np.float32)

    buf.extend(struct.pack("<ii", road_id, road_type))
    buf.extend(struct.pack("<i", seg_len))
    for col in (xs.astype(np.float32), ys.astype(np.float32), zs, headings):
        buf.extend(col.tobytes())

    if is_lane:
        buf.extend(struct.pack("<i", 0))      # num_entries
        buf.extend(struct.pack("<i", 0))      # num_exits
        buf.extend(struct.pack("<f", 15.0))   # speed_limit (m/s)
        seg = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        cum = np.concatenate([[0.0], np.cumsum(seg)]).astype(np.float32)
        buf.extend(struct.pack("<f", float(cum[-1])))  # length
        buf.extend(cum.tobytes())                       # cum_lengths[S]


def build_bottleneck_bin() -> bytes:
    buf = bytearray()

    controllable = []   # (id, xs, ys, headings, vxs, vys, length_m, width_m)
    wp_indices = _waypoint_indices(T_STEPS, NUM_TARGET_WAYPOINTS)
    size_rng = np.random.default_rng(CAR_SIZE_SEED)

    agent_id = 0
    for lane_idx in SPAWN_LANE_INDICES:
        lane_y = LANE_CENTERS_Y[lane_idx]
        for spawn_x in SPAWN_BACK_X:
            # Reference path: stay in the spawn lane (heading +x at spawn), then
            # merge fully into the free lane BEFORE the static wall so the path
            # passes through the open gap without clipping a blocker car (which
            # remove_bad_trajectories would then delete). After the merge it runs
            # straight through the bottleneck, then fans back to its own lane.
            merge_start = (BOTTLENECK_X - 12.0, lane_y)         # x=38, still in lane
            merge_done = (BOTTLENECK_X - 5.0, FREE_LANE_Y)      # x=45, in free lane
            wp1 = (BOTTLENECK_X, FREE_LANE_Y)                   # free lane @ bottleneck
            wp2 = (BOTTLENECK_X + 60.0, FREE_LANE_Y)            # other side, free lane
            wp3 = (ROAD_LENGTH_M - 10.0, lane_y)               # fan back to own lane
            control_pts = [(spawn_x, lane_y), merge_start, merge_done, wp1, wp2, wp3]
            control_idx = [0, 38, 58] + wp_indices              # 0,38,58,66,132,199
            xs, ys = _piecewise_trajectory(control_pts, control_idx, T_STEPS)
            xs = _orient_x(xs)                                   # flip to R->L if set
            headings, vxs, vys = _trajectory_kinematics(xs, ys, DT_SECONDS)
            if SPAWN_STOPPED:
                vxs[0] = 0.0                                     # start at rest
                vys[0] = 0.0
            length_m = float(size_rng.uniform(*CAR_LENGTH_RANGE_M))
            width_m = float(size_rng.uniform(*CAR_WIDTH_RANGE_M))
            controllable.append((agent_id, xs, ys, headings, vxs, vys, length_m, width_m))
            agent_id += 1

    static = []   # (id, x, y)
    for row_x in STATIC_ROWS_X:
        for sy in STATIC_Y:
            static.append((agent_id, row_x, sy))
            agent_id += 1

    num_agents = len(controllable) + len(static)

    # roads: 4 lanes + 2 edges + 3 dividers
    num_lanes = NUM_LANES
    num_edges = 2
    num_dividers = NUM_LANES - 1
    num_roads = num_lanes + num_edges + num_dividers

    # ── header ──
    buf.extend(struct.pack("<iiii", num_agents, num_roads, 0, 0))

    # ── controllable agents ──
    for (aid, xs, ys, headings, vxs, vys, length_m, width_m) in controllable:
        _pack_agent(buf, aid, VEHICLE, xs, ys, headings, vxs, vys,
                    length_m, width_m, mark_as_expert=0)

    # ── static blocker cars (constant trajectory, marked expert) ──
    static_heading = np.pi if TRAVEL_RIGHT_TO_LEFT else 0.0
    for (aid, sx, sy) in static:
        xs = np.full(T_STEPS, _orient_x(sx), dtype=np.float32)
        ys = np.full(T_STEPS, sy, dtype=np.float32)
        vxs = np.zeros(T_STEPS, dtype=np.float32)
        vys = np.zeros(T_STEPS, dtype=np.float32)
        _pack_agent(buf, aid, VEHICLE, xs, ys, static_heading, vxs, vys,
                    STATIC_SIZE_M, STATIC_SIZE_M, mark_as_expert=1)

    # ── roads ──
    # Road geometry runs the full length plus the spawn-side extension. In the
    # LTR build frame the spawn end is the low-x end (cars start near x=0 and
    # drive +x), so extend the low end behind them; the orient flip then places
    # the extra road just behind where the cars start.
    road_x_min = -ROAD_EXTENSION_M
    n_lane_pts = int((ROAD_LENGTH_M - road_x_min) / LANE_SAMPLE_SPACING_M) + 1
    road_xs = _orient_x(np.linspace(road_x_min, ROAD_LENGTH_M, n_lane_pts).astype(np.float32))
    road_id = 0
    # lanes
    for lane_idx in range(NUM_LANES):
        ys = np.full(n_lane_pts, LANE_CENTERS_Y[lane_idx], dtype=np.float32)
        _pack_polyline(buf, road_id, LANE_FREEWAY, road_xs, ys, is_lane=True)
        road_id += 1
    # outer edges (asymmetric once the free lane is widened)
    for edge_y in (ROAD_BOTTOM_Y, ROAD_TOP_Y):
        ys = np.full(n_lane_pts, edge_y, dtype=np.float32)
        _pack_polyline(buf, road_id, ROAD_EDGE_BOUNDARY, road_xs, ys, is_lane=False)
        road_id += 1
    # lane dividers = the internal lane boundaries
    for d in range(1, NUM_LANES):
        ys = np.full(n_lane_pts, LANE_BOUNDARIES_Y[d], dtype=np.float32)
        _pack_polyline(buf, road_id, ROAD_LINE_BROKEN_SINGLE_WHITE, road_xs, ys, is_lane=False)
        road_id += 1

    assert road_id == num_roads

    # ── lane graph: empty ──
    buf.extend(struct.pack("<i", 0))

    # ── metadata ──
    scenario_id = "bottleneck_4lane"
    dataset = "synthetic"
    buf.extend(scenario_id.encode("utf-8")[:METADATA_ID_BYTES].ljust(METADATA_ID_BYTES, b"\0"))
    buf.extend(dataset.encode("utf-8")[:METADATA_DATASET_BYTES].ljust(METADATA_DATASET_BYTES, b"\0"))
    buf.extend(struct.pack("<i", T_STEPS))     # scenario_length
    buf.extend(struct.pack("<f", DT_SECONDS))  # dt
    buf.extend(struct.pack("<i", 0))           # objects_of_interest count
    buf.extend(struct.pack("<i", 0))           # tracks_to_predict count

    return bytes(buf)


def main():
    parser = argparse.ArgumentParser(description="Generate a 4-lane bottleneck .bin map")
    parser.add_argument("--output", required=True,
                        help="Output directory; the .bin is written inside it")
    parser.add_argument("--name", default="bottleneck_4lane", help="Map file stem")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.name + ".bin")

    binary = build_bottleneck_bin()
    with open(out_path, "wb") as f:
        f.write(binary)

    print(f"wrote {out_path} ({len(binary)} bytes)")
    print(f"  controllable agents: {NUM_CONTROLLED}")
    print(f"  static blocker cars: {len(STATIC_Y) * len(STATIC_ROWS_X)}")
    print(f"  bottleneck at x={BOTTLENECK_X:.0f} m, free lane y={FREE_LANE_Y:.2f}")
    print(f"  replay waypoint indices: {_waypoint_indices(T_STEPS, NUM_TARGET_WAYPOINTS)}")


if __name__ == "__main__":
    main()
