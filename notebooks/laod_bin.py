"""
Inspect PufferDrive .bin map files and print the live observation vector.

Usage:
    python yvonne/laod_bin.py

Loads one Carla bin (gigaflow) and one nuPlan bin (replay), prints a summary
of what's inside each file, resets the Drive env, decodes the first agent's
observation vector, and optionally saves a rendered frame.
"""
import struct
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ------------------------------------------------------------------
# Constants (mirror drive.h / datatypes.h)
# ------------------------------------------------------------------
MAX_SPEED        = 40.0
Z_BUFFER         = 4.0
LANE_DIST_NORM   = 4.0
STEERING_MAX     = 0.667       # STEERING_VALUES[8]
MAX_STOPPED_S    = 60.0
LANE_WIDTH       = 3.7

AGENT_TYPE_NAMES  = {0: "UNKNOWN", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "CYCLIST"}
ROAD_TYPE_NAMES   = {
    0: "LANE_UNKNOWN", 1: "LANE_FREEWAY", 2: "LANE_SURFACE_STREET",
    3: "LANE_BIKE_LANE", 4: "LANE_BUS_LANE",
    10: "ROAD_LINE_UNKNOWN", 11: "ROAD_LINE_BROKEN_SINGLE_WHITE",
    12: "ROAD_LINE_SOLID_SINGLE_WHITE", 20: "ROAD_EDGE_UNKNOWN",
    21: "ROAD_EDGE_BOUNDARY", 22: "ROAD_EDGE_MEDIAN",
    30: "MISC_UNKNOWN", 31: "MISC_CROSSWALK", 32: "MISC_SPEED_BUMP",
}
TRAFFIC_TYPE_NAMES  = {0: "NONE", 1: "TRAFFIC_LIGHT", 2: "STOP_SIGN", 3: "YIELD_SIGN"}
TRAFFIC_STATE_NAMES = {0: "UNKNOWN", 1: "RED", 2: "YELLOW", 3: "GREEN", 4: "OFF"}

# EGO_FEATURES: classic=8, jerk=10 (adds a_long, a_lat)
EGO_FEATURES_CLASSIC = 8
EGO_FEATURES_JERK    = 10
ROAD_FEATURES        = 7
PARTNER_FEATURES     = 8
TRAFFIC_CTRL_FEATURES = 7
STATIC_TARGET_FEATURES  = 3
DYNAMIC_TARGET_FEATURES = 5


# ==================================================================
# Section 1: Binary file parser (mirrors load_map_binary in drive.h)
# ==================================================================

def _ri(f):  return struct.unpack("i", f.read(4))[0]
def _rf(f):  return struct.unpack("f", f.read(4))[0]
def _ri_arr(f, n): return list(struct.unpack(f"{n}i", f.read(4 * n))) if n > 0 else []
def _rf_arr(f, n): return list(struct.unpack(f"{n}f", f.read(4 * n))) if n > 0 else []


def read_bin(path: str) -> dict:
    """Parse a PufferDrive .bin file and return a structured dict.

    Header (4 ints):
        num_total_agents | num_roads | num_traffic | num_objects

    Per-agent record:
        id | type | tlen
        tlen x [x, y, z, heading, vx, vy, length, width, height] (float)
        tlen x valid (int)
        route_length | route[route_length] | route_gt_len
        goal_x | goal_y | goal_z | mark_as_expert

    Per-road record:
        id | type | segment_length
        slen x [x, y, z, headings] (float)
        if is_road_lane (type 0-9):
            num_entries | entry_lanes[] | num_exits | exit_lanes[] | speed_limit

    Per-traffic record:
        id | type | stop_line[6] | heading | state_length | states[] |
        num_controlled_lanes | controlled_lanes[]

    Objects: skipped (9*T floats + T ints per object)

    Lane graph:
        n_lanes | lane_ids[] | lane_lengths[] | distances[n*n]
    """
    data = {"agents": [], "roads": [], "traffic": [], "lane_graph": {}}
    with open(path, "rb") as f:
        num_agents  = _ri(f)
        num_roads   = _ri(f)
        num_traffic = _ri(f)
        num_objects = _ri(f)

        # --- Agents ---
        for _ in range(num_agents):
            agent_id = _ri(f)
            atype    = _ri(f)
            tlen     = _ri(f)
            traj_x   = _rf_arr(f, tlen)
            traj_y   = _rf_arr(f, tlen)
            traj_z   = _rf_arr(f, tlen)
            heading  = _rf_arr(f, tlen)
            vel_x    = _rf_arr(f, tlen)
            vel_y    = _rf_arr(f, tlen)
            length   = _rf_arr(f, tlen)
            width    = _rf_arr(f, tlen)
            height   = _rf_arr(f, tlen)
            valid    = _ri_arr(f, tlen)
            route_len = _ri(f)
            route     = _ri_arr(f, route_len)
            route_gt_len = _ri(f)
            goal_x = _rf(f); goal_y = _rf(f); goal_z = _rf(f)
            expert = _ri(f)
            data["agents"].append({
                "id": agent_id, "type": atype, "tlen": tlen,
                "x": traj_x, "y": traj_y, "z": traj_z,
                "heading": heading, "vx": vel_x, "vy": vel_y,
                "length": length, "width": width, "height": height,
                "valid": valid, "route": route, "route_gt_len": route_gt_len,
                "goal": (goal_x, goal_y, goal_z), "expert": expert,
            })

        # --- Road elements ---
        for _ in range(num_roads):
            road_id = _ri(f)
            rtype   = _ri(f)
            slen    = _ri(f)
            xs = _rf_arr(f, slen)
            ys = _rf_arr(f, slen)
            zs = _rf_arr(f, slen)
            hs = _rf_arr(f, slen)
            if 0 <= rtype <= 9:  # is_road_lane
                n_entries = _ri(f)
                entries   = _ri_arr(f, n_entries)
                n_exits   = _ri(f)
                exits     = _ri_arr(f, n_exits)
                speed_lim = _rf(f)
            else:
                entries = exits = []; speed_lim = 0.0
            data["roads"].append({
                "id": road_id, "type": rtype, "slen": slen,
                "x": xs, "y": ys, "z": zs, "headings": hs,
                "entries": entries, "exits": exits, "speed_limit": speed_lim,
            })

        # --- Traffic control elements ---
        for _ in range(num_traffic):
            traf_id    = _ri(f)
            ttype      = _ri(f)
            stop_line  = _rf_arr(f, 6)
            heading    = _rf(f)
            state_len  = _ri(f)
            states     = _ri_arr(f, state_len)
            n_ctrl     = _ri(f)
            ctrl_lanes = _ri_arr(f, n_ctrl)
            data["traffic"].append({
                "id": traf_id, "type": ttype, "stop_line": stop_line,
                "heading": heading, "states": states, "controlled_lanes": ctrl_lanes,
            })

        # --- Objects (skipped in C, skip here too) ---
        for _ in range(num_objects):
            _ri(f); _ri(f)      # obj_id, obj_type
            T = _ri(f)
            f.read(9 * T * 4 + T * 4)  # 9 float arrays + 1 int array

        # --- Lane graph ---
        n_lanes = _ri(f)
        lane_ids   = _ri_arr(f, n_lanes)
        lane_lens  = _rf_arr(f, n_lanes)
        distances  = _rf_arr(f, n_lanes * n_lanes)
        data["lane_graph"] = {
            "n_lanes": n_lanes, "ids": lane_ids,
            "lengths": lane_lens, "distances": distances,
        }

    return data


def print_bin_summary(path: str, data: dict, label: str = ""):
    """Print a human-readable summary of the bin file contents."""
    agents  = data["agents"]
    roads   = data["roads"]
    traffic = data["traffic"]
    lg      = data["lane_graph"]

    header = f"  [{label}]  {path}"
    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)

    # --- Agents ---
    print(f"\n  AGENTS: {len(agents)} total")
    type_counts: dict = {}
    for a in agents:
        name = AGENT_TYPE_NAMES.get(a["type"], f"?{a['type']}")
        type_counts[name] = type_counts.get(name, 0) + 1
    for name, cnt in sorted(type_counts.items()):
        print(f"    {name}: {cnt}")
    if agents:
        a0 = agents[0]
        valid_steps = sum(a0["valid"])
        print(f"  Trajectory length : {a0['tlen']} steps")
        print(f"  Agent[0] valid    : {valid_steps}/{a0['tlen']}")
        if a0["x"]:
            xs = [x for x, v in zip(a0["x"], a0["valid"]) if v]
            ys = [y for y, v in zip(a0["y"], a0["valid"]) if v]
            if xs:
                print(f"  Agent[0] pos range: x=[{min(xs):.1f}, {max(xs):.1f}]"
                      f"  y=[{min(ys):.1f}, {max(ys):.1f}]")
        print(f"  Agent[0] goal     : ({a0['goal'][0]:.1f}, {a0['goal'][1]:.1f}, {a0['goal'][2]:.1f})")
        print(f"  Expert agents     : {sum(1 for a in agents if a['expert'])}")

    # --- Roads ---
    print(f"\n  ROAD ELEMENTS: {len(roads)}")
    rtype_counts: dict = {}
    for r in roads:
        name = ROAD_TYPE_NAMES.get(r["type"], f"?{r['type']}")
        rtype_counts[name] = rtype_counts.get(name, 0) + 1
    for name, cnt in sorted(rtype_counts.items()):
        print(f"    {name}: {cnt}")
    if roads:
        all_x = [x for r in roads for x in r["x"]]
        all_y = [y for r in roads for y in r["y"]]
        if all_x:
            print(f"  Road bbox         : x=[{min(all_x):.1f}, {max(all_x):.1f}]"
                  f"  y=[{min(all_y):.1f}, {max(all_y):.1f}]")

    # --- Traffic ---
    print(f"\n  TRAFFIC CTRL: {len(traffic)}")
    tc_counts: dict = {}
    for t in traffic:
        name = TRAFFIC_TYPE_NAMES.get(t["type"], f"?{t['type']}")
        tc_counts[name] = tc_counts.get(name, 0) + 1
    for name, cnt in sorted(tc_counts.items()):
        print(f"    {name}: {cnt}")

    # --- Lane graph ---
    print(f"\n  LANE GRAPH    : {lg['n_lanes']} lanes")
    if lg["lengths"]:
        ls = lg["lengths"]
        print(f"  Lane length   : min={min(ls):.1f}  max={max(ls):.1f}  mean={sum(ls)/len(ls):.1f} m")


# ==================================================================
# Section 2: Observation vector decoder
# ==================================================================

def _obs_name_layout(dynamics_model, target_type, num_target_wps,
                     max_partner_obs, obs_lane_segs, obs_boundary_segs,
                     max_traffic_ctrl_obs, reward_conditioning):
    """Return a list of (name, scale_to_raw) tuples matching the obs layout."""
    names = []
    ego_n = EGO_FEATURES_JERK if dynamics_model == "jerk" else EGO_FEATURES_CLASSIC
    # EGO
    names.append(("ego/speed_signed [m/s]",  MAX_SPEED))
    names.append(("ego/width [m]",            None))   # * max_veh_width
    names.append(("ego/length [m]",           None))   # * max_veh_len
    names.append(("ego/steering [rad]",       STEERING_MAX))
    if dynamics_model == "jerk":
        names.append(("ego/a_long [norm]",    1.0))
        names.append(("ego/a_lat [norm]",     1.0))
    names.append(("ego/lane_dist [m]",        LANE_DIST_NORM))
    names.append(("ego/lane_angle_cos",       1.0))
    names.append(("ego/speed_limit [m/s]",    MAX_SPEED))
    names.append(("ego/seconds_stopped [s]",  MAX_STOPPED_S))
    # REWARD COEFS
    if reward_conditioning:
        for c in range(17):
            names.append((f"reward_coef/{c}", 1.0))
    # TARGET
    if target_type == "static":
        for wp in range(num_target_wps):
            names.append((f"wp{wp}/rel_fwd [m]",  None))
            names.append((f"wp{wp}/rel_left [m]", None))
            names.append((f"wp{wp}/dz [m]",       Z_BUFFER))
    else:
        for wp in range(num_target_wps):
            names.append((f"wp{wp}/rel_x [m]",  None))
            names.append((f"wp{wp}/rel_y [m]",  None))
            names.append((f"wp{wp}/dz [m]",     Z_BUFFER))
            names.append((f"wp{wp}/cos_h",      1.0))
            names.append((f"wp{wp}/sin_h",      1.0))
    # PARTNERS
    for p in range(max_partner_obs):
        names.append((f"partner{p}/rel_x [m]",   None))
        names.append((f"partner{p}/rel_y [m]",   None))
        names.append((f"partner{p}/dz [m]",      Z_BUFFER))
        names.append((f"partner{p}/length [m]",  None))
        names.append((f"partner{p}/width [m]",   None))
        names.append((f"partner{p}/cos_h",       1.0))
        names.append((f"partner{p}/sin_h",       1.0))
        names.append((f"partner{p}/speed [m/s]", MAX_SPEED))
    # LANE SEGMENTS
    for s in range(obs_lane_segs):
        for feat in ("x", "y", "z", "len", "width", "cos", "sin"):
            names.append((f"lane{s}/{feat}", 1.0))
    # BOUNDARY SEGMENTS
    for s in range(obs_boundary_segs):
        for feat in ("x", "y", "z", "len", "width", "cos", "sin"):
            names.append((f"bound{s}/{feat}", 1.0))
    # TRAFFIC CTRL
    for t in range(max_traffic_ctrl_obs):
        for feat in ("rx1", "ry1", "rx2", "ry2", "dz", "type", "state"):
            names.append((f"tc{t}/{feat}", 1.0))
    return names


def decode_and_print_obs(obs: np.ndarray, env, label: str = ""):
    """Decode obs[0] from a Drive env and print human-readable values."""
    dm    = env.dynamics_model        # "classic" or "jerk"
    tt    = env.target_type_str       # "static" or "dynamic"
    n_wp  = env.num_target_waypoints
    n_p   = env.max_partner_observations
    n_ls  = env.obs_lane_segment_count
    n_bs  = env.obs_boundary_segment_count
    n_tc  = env.max_traffic_control_observations
    rc    = env.reward_conditioning
    mgl   = env.max_goal_position
    mp    = env.max_position
    mvl   = env.max_veh_len
    mvw   = env.max_veh_width
    mrsl  = env.max_road_segment_length
    mrsw  = env.max_road_segment_width

    print(f"\n{'='*70}")
    print(f"  OBSERVATION VECTOR  [{label}]")
    print(f"{'='*70}")
    print(f"  Total dims : {len(obs)}")
    print(f"  Dynamics   : {dm}  |  Target: {tt}  |  Waypoints: {n_wp}")
    print(f"  Partners   : {n_p}  |  Lane segs: {n_ls}  |  Boundary segs: {n_bs}  |  Traffic ctrl: {n_tc}")

    idx = 0

    # --- EGO ---
    print(f"\n  --- EGO ({EGO_FEATURES_JERK if dm=='jerk' else EGO_FEATURES_CLASSIC} features) ---")
    print(f"    [{idx:3d}] speed_signed     = {obs[idx]*MAX_SPEED:+8.3f} m/s  (raw={obs[idx]:.4f})"); idx += 1
    print(f"    [{idx:3d}] width            = {obs[idx]*mvw:8.3f} m     (raw={obs[idx]:.4f})"); idx += 1
    print(f"    [{idx:3d}] length           = {obs[idx]*mvl:8.3f} m     (raw={obs[idx]:.4f})"); idx += 1
    print(f"    [{idx:3d}] steering_angle   = {obs[idx]*STEERING_MAX:+8.3f} rad   (raw={obs[idx]:.4f})"); idx += 1
    if dm == "jerk":
        print(f"    [{idx:3d}] a_long           = {obs[idx]:+8.4f} (norm)"); idx += 1
        print(f"    [{idx:3d}] a_lat            = {obs[idx]:+8.4f} (norm)"); idx += 1
    print(f"    [{idx:3d}] lane_dist        = {obs[idx]*LANE_DIST_NORM:+8.3f} m     (raw={obs[idx]:.4f})"); idx += 1
    print(f"    [{idx:3d}] lane_angle_cos   = {obs[idx]:+8.4f}        (raw={obs[idx]:.4f})"); idx += 1
    print(f"    [{idx:3d}] speed_limit      = {obs[idx]*MAX_SPEED:8.3f} m/s  (raw={obs[idx]:.4f})"); idx += 1
    print(f"    [{idx:3d}] seconds_stopped  = {obs[idx]*MAX_STOPPED_S:8.3f} s     (raw={obs[idx]:.4f})"); idx += 1

    # --- REWARD COEFS ---
    if rc:
        print(f"\n  --- REWARD COEFS (17) ---")
        for c in range(17):
            print(f"    [{idx:3d}] coef_{c:02d} = {obs[idx]:.4f}"); idx += 1

    # --- TARGET WAYPOINTS ---
    print(f"\n  --- TARGET WAYPOINTS ({n_wp}, type={tt}) ---")
    for wp in range(n_wp):
        if tt == "static":
            fwd  = obs[idx] * mgl
            left = obs[idx+1] * mgl
            dz   = obs[idx+2] * Z_BUFFER
            dist = np.sqrt(fwd**2 + left**2)
            print(f"    WP{wp} [{idx:3d}..{idx+2}] fwd={fwd:+7.2f}m  left={left:+7.2f}m  dz={dz:+.3f}m  |dist|={dist:.2f}m")
            idx += 3
        else:
            rx = obs[idx] * mp; ry = obs[idx+1] * mp
            dz = obs[idx+2] * Z_BUFFER
            ch = obs[idx+3]; sh = obs[idx+4]
            dist = np.sqrt(rx**2 + ry**2)
            print(f"    WP{wp} [{idx:3d}..{idx+4}] x={rx:+7.2f}m  y={ry:+7.2f}m  dz={dz:+.3f}m  |dist|={dist:.2f}m  h=({ch:.3f},{sh:.3f})")
            idx += 5

    # --- PARTNERS ---
    partner_start = idx
    active = []
    for p in range(n_p):
        rx = obs[idx]; ry = obs[idx+1]
        if abs(rx) > 1e-6 or abs(ry) > 1e-6:
            active.append((p, idx, obs[idx:idx+PARTNER_FEATURES].copy()))
        idx += PARTNER_FEATURES
    print(f"\n  --- PARTNERS: {len(active)}/{n_p} occupied slots ---")
    for p_idx, base, pobs in active[:8]:
        rx = pobs[0]*mp; ry = pobs[1]*mp
        dz = pobs[2]*Z_BUFFER
        ln = pobs[3]*mvl; wd = pobs[4]*mvw
        ch = pobs[5]; sh = pobs[6]
        sp = pobs[7]*MAX_SPEED
        dist = np.sqrt(rx**2 + ry**2)
        print(f"    P{p_idx:02d} [{base:3d}] dist={dist:6.1f}m  x={rx:+6.1f}  y={ry:+6.1f}  "
              f"speed={sp:.1f}m/s  size={ln:.1f}x{wd:.1f}m")
    if len(active) > 8:
        print(f"    ... ({len(active)-8} more partners not shown)")

    # --- LANE SEGMENTS ---
    lane_start = idx
    active_lanes = 0
    for s in range(n_ls):
        seg = obs[idx:idx+ROAD_FEATURES]
        if abs(seg[3]) > 1e-6:  # non-zero length
            active_lanes += 1
        idx += ROAD_FEATURES
    print(f"\n  --- LANE SEGMENTS : {active_lanes}/{n_ls} active (indices {lane_start}..{idx-1}) ---")
    # Show a few
    idx2 = lane_start
    shown = 0
    for s in range(n_ls):
        seg = obs[idx2:idx2+ROAD_FEATURES]
        if abs(seg[3]) > 1e-6 and shown < 4:
            x=seg[0]*mp; y=seg[1]*mp; z=seg[2]*Z_BUFFER
            ln=seg[3]*mrsl; wd=seg[4]*mrsw
            dist = np.sqrt(x**2 + y**2)
            print(f"    L{s:02d} [{idx2:3d}] x={x:+6.1f}  y={y:+6.1f}  dist={dist:.1f}m  len={ln:.2f}m  cos={seg[5]:.3f}")
            shown += 1
        idx2 += ROAD_FEATURES
    if active_lanes > 4:
        print(f"    ... ({active_lanes-4} more lanes not shown)")

    # --- BOUNDARY SEGMENTS ---
    bound_start = idx
    active_bounds = 0
    for s in range(n_bs):
        seg = obs[idx:idx+ROAD_FEATURES]
        if abs(seg[3]) > 1e-6:
            active_bounds += 1
        idx += ROAD_FEATURES
    print(f"\n  --- BOUNDARY SEGS : {active_bounds}/{n_bs} active (indices {bound_start}..{idx-1}) ---")
    idx2 = bound_start
    shown = 0
    for s in range(n_bs):
        seg = obs[idx2:idx2+ROAD_FEATURES]
        if abs(seg[3]) > 1e-6 and shown < 4:
            x=seg[0]*mp; y=seg[1]*mp
            ln=seg[3]*mrsl
            dist = np.sqrt(x**2 + y**2)
            print(f"    B{s:02d} [{idx2:3d}] x={x:+6.1f}  y={y:+6.1f}  dist={dist:.1f}m  len={ln:.2f}m  cos={seg[5]:.3f}")
            shown += 1
        idx2 += ROAD_FEATURES
    if active_bounds > 4:
        print(f"    ... ({active_bounds-4} more boundaries not shown)")

    # --- TRAFFIC CTRL ---
    tc_start = idx
    active_tc = 0
    for t in range(n_tc):
        seg = obs[idx:idx+TRAFFIC_CTRL_FEATURES]
        if abs(seg[5]) > 0.5:  # type > 0 => real element
            active_tc += 1
            rx1=seg[0]*mp; ry1=seg[1]*mp
            rx2=seg[2]*mp; ry2=seg[3]*mp
            ttype = int(round(seg[5])); tstate = int(round(seg[6]))
            print(f"  TC{t} [{idx:3d}] type={TRAFFIC_TYPE_NAMES.get(ttype,ttype)}"
                  f"  state={TRAFFIC_STATE_NAMES.get(tstate,tstate)}"
                  f"  p1=({rx1:.1f},{ry1:.1f})  p2=({rx2:.1f},{ry2:.1f})")
        idx += TRAFFIC_CTRL_FEATURES
    if active_tc == 0:
        print(f"\n  --- TRAFFIC CTRL  : 0/{n_tc} active (indices {tc_start}..{idx-1}) ---")

    assert idx == len(obs), f"Obs decode mismatch: consumed {idx}, got {len(obs)}"
    print(f"\n  ✓ All {len(obs)} dimensions accounted for.")


# ==================================================================
# Section 3: Create Drive env, reset, print obs, render
# ==================================================================

def make_env(map_dir, simulation_mode, control_mode, num_maps=1,
             num_agents=8, min_agents_per_env=1, max_agents_per_env=8,
             scenario_length=None, render_mode=None, dynamics_model="jerk"):
    """Thin wrapper: create a Drive env and call reset."""
    from pufferlib.ocean.drive.drive import Drive
    env = Drive(
        map_dir=map_dir,
        num_maps=num_maps,
        num_agents=num_agents,
        min_agents_per_env=min_agents_per_env,
        max_agents_per_env=max_agents_per_env,
        simulation_mode=simulation_mode,
        control_mode=control_mode,
        init_mode="create_all_valid",
        dynamics_model=dynamics_model,
        action_type="discrete",
        target_type="static",
        num_target_waypoints=3,
        max_lane_segment_observations=80,
        max_boundary_segment_observations=80,
        max_partner_observations=16,
        max_traffic_control_observations=4,
        scenario_length=scenario_length,
        reward_conditioning=False,
        render_mode=render_mode,
        seed=42,
        report_interval=1,
    )
    env.reset()
    return env


def try_render(env, label, out_dir="/scratch/yw4142/PufferDrive4/yvonne"):
    """Attempt headless render; print result."""
    if env.render_mode is None:
        print(f"  [render] render_mode=None — skipping.")
        return
    try:
        env.render(env_idx=0, view_mode=0)
        env.render(env_idx=0, view_mode=1)   # BEV
        env.close_client(env_idx=0)
        print(f"  [render] Frame saved to {out_dir}/ (check for .mp4)")
    except Exception as e:
        print(f"  [render] Failed: {e}")


# ==================================================================
# MAIN
# ==================================================================

CARLA_DIR  = "/scratch/ev2237/data/carla"
NUPLAN_DIR = "/scratch/ev2237/data/nuplan/nuplan_full_dir_50"

CARLA_BIN  = os.path.join(CARLA_DIR,  "opendrive__Town01.bin")
NUPLAN_BIN = os.path.join(NUPLAN_DIR, "nuplan__00605dc7-3887-5ff4-9a7f-996d11a57851.bin")


def main():
    # ---- Parse both bin files ----
    print("\n" + "#" * 70)
    print("#  STEP 1: Parse .bin files (C binary format)")
    print("#" * 70)
    carla_data  = read_bin(CARLA_BIN)
    nuplan_data = read_bin(NUPLAN_BIN)

    print_bin_summary(CARLA_BIN,  carla_data,  label="CARLA / gigaflow")
    print_bin_summary(NUPLAN_BIN, nuplan_data, label="nuPlan / replay")

    # ---- Create envs ----
    print("\n\n" + "#" * 70)
    print("#  STEP 2: Create Drive envs + reset")
    print("#" * 70)

    print("\n  Creating Carla env (gigaflow, control_vehicles) ...")
    carla_env = make_env(
        map_dir=CARLA_DIR,
        simulation_mode="gigaflow",
        control_mode="control_vehicles",
        num_maps=1,
        num_agents=4,
        min_agents_per_env=1,
        max_agents_per_env=4,
        scenario_length=300,
        dynamics_model="jerk",
    )
    print(f"  Carla env:  num_envs={carla_env.num_envs}  "
          f"num_agents={carla_env.num_agents}  obs_dim={carla_env.num_obs}")

    print("\n  Creating nuPlan env (replay, control_sdc_only) ...")
    nuplan_env = make_env(
        map_dir=NUPLAN_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        num_maps=1,
        num_agents=4,
        min_agents_per_env=1,
        max_agents_per_env=4,
        scenario_length=201,
        dynamics_model="jerk",
    )
    print(f"  nuPlan env: num_envs={nuplan_env.num_envs}  "
          f"num_agents={nuplan_env.num_agents}  obs_dim={nuplan_env.num_obs}")

    # ---- Print observation vectors ----
    print("\n\n" + "#" * 70)
    print("#  STEP 3: Decode observation vector (agent 0)")
    print("#" * 70)

    carla_obs  = carla_env.observations[0]
    nuplan_obs = nuplan_env.observations[0]

    decode_and_print_obs(carla_obs,  carla_env,  label="CARLA / gigaflow / agent 0")
    decode_and_print_obs(nuplan_obs, nuplan_env, label="nuPlan / replay / agent 0")

    # ---- Render ----
    print("\n\n" + "#" * 70)
    print("#  STEP 4: Render (headless, requires EGL)")
    print("#" * 70)

    for env_name, env in [("carla", carla_env), ("nuplan", nuplan_env)]:
        print(f"  [{env_name}] render_mode={env.render_mode}")
        try_render(env, env_name)

    carla_env.close()
    nuplan_env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
