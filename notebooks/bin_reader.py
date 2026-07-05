"""Deserializer for PufferDrive .bin files.

Binary layout matches the format produced by 123Drive's serialize.py.
Returns plain dicts/numpy arrays — no 123Drive import required.
"""

import struct
import numpy as np


def _read_int_list(buf: bytes, offset: int):
    n, = struct.unpack_from("<i", buf, offset)
    offset += 4
    if n > 0:
        vals = list(struct.unpack_from(f"<{n}i", buf, offset))
        offset += 4 * n
    else:
        vals = []
    return vals, offset


def _read_dynamic_states(buf: bytes, offset: int):
    T, = struct.unpack_from("<i", buf, offset)
    offset += 4
    cols = []
    for _ in range(9):  # x, y, z, heading, vx, vy, length, width, height
        arr = np.frombuffer(buf, dtype=np.float32, count=T, offset=offset).copy()
        cols.append(arr)
        offset += 4 * T
    valid = np.frombuffer(buf, dtype=np.int32, count=T, offset=offset).copy()
    offset += 4 * T
    x, y, z, heading, vx, vy, length, width, height = cols
    return {
        "T": T,
        "position": np.stack([x, y, z], axis=1),
        "heading": heading,
        "velocity": np.stack([vx, vy], axis=1),
        "length": length,
        "width": width,
        "height": height,
        "valid": valid,
    }, offset


def load_bin(path: str) -> dict:
    """Load a PufferDrive .bin file and return a parsed scenario dict.

    Keys:
      agents       — dict[int, dict]  (id -> track dict with dynamic states + route + goal)
      road_map     — dict[int, dict]  (id -> element dict with type, xyz, heading, ...)
      traffic_controls — list[dict]
      objects      — dict[int, dict]
      lane_graph   — dict | None
      metadata     — dict
    """
    with open(path, "rb") as f:
        buf = f.read()

    offset = 0
    num_agents, num_road, num_tcs, num_objects = struct.unpack_from("<iiii", buf, offset)
    offset += 16

    # --- Agents ---
    agents = {}
    for _ in range(num_agents):
        agent_id, agent_type = struct.unpack_from("<ii", buf, offset)
        offset += 8
        states, offset = _read_dynamic_states(buf, offset)
        route, offset = _read_int_list(buf, offset)
        route_gt_len, = struct.unpack_from("<i", buf, offset)
        offset += 4
        goal = struct.unpack_from("<fff", buf, offset)
        offset += 12
        control_state, = struct.unpack_from("<i", buf, offset)
        offset += 4
        agents[agent_id] = {
            **states,
            "type": agent_type,
            "route": route,
            "route_gt_len": route_gt_len,
            "goal": np.array(goal, dtype=np.float32),
            "control_state": control_state,
        }

    # --- Road map ---
    road_map = {}
    for _ in range(num_road):
        elem_id, road_type = struct.unpack_from("<ii", buf, offset)
        offset += 8
        n_pts, = struct.unpack_from("<i", buf, offset)
        offset += 4
        xyz_cols = []
        for _ in range(3):
            col = np.frombuffer(buf, dtype=np.float32, count=n_pts, offset=offset).copy()
            xyz_cols.append(col)
            offset += 4 * n_pts
        heading = np.frombuffer(buf, dtype=np.float32, count=n_pts, offset=offset).copy()
        offset += 4 * n_pts
        xyz = np.stack(xyz_cols, axis=1)
        elem = {"type": road_type, "xyz": xyz, "heading": heading}
        if 0 <= road_type <= 9:  # lane
            entry_lanes, offset = _read_int_list(buf, offset)
            exit_lanes, offset = _read_int_list(buf, offset)
            speed_limit, = struct.unpack_from("<f", buf, offset)
            offset += 4
            elem.update({"entry_lanes": entry_lanes, "exit_lanes": exit_lanes, "speed_limit_mps": speed_limit})
        road_map[elem_id] = elem

    # --- Traffic controls ---
    traffic_controls = []
    for _ in range(num_tcs):
        ctrl_id, ctrl_type = struct.unpack_from("<ii", buf, offset)
        offset += 8
        sl_start = struct.unpack_from("<fff", buf, offset)
        offset += 12
        sl_end = struct.unpack_from("<fff", buf, offset)
        offset += 12
        tc_heading, = struct.unpack_from("<f", buf, offset)
        offset += 4
        states, offset = _read_int_list(buf, offset)
        controlled_lanes, offset = _read_int_list(buf, offset)
        traffic_controls.append({
            "id": ctrl_id,
            "type": ctrl_type,
            "stop_line": np.array([sl_start, sl_end], dtype=np.float32),
            "heading": tc_heading,
            "states": states,
            "controlled_lanes": controlled_lanes,
        })

    # --- Objects ---
    objects = {}
    for _ in range(num_objects):
        obj_id, obj_type = struct.unpack_from("<ii", buf, offset)
        offset += 8
        states, offset = _read_dynamic_states(buf, offset)
        objects[obj_id] = {**states, "type": obj_type}

    # --- Lane graph ---
    n_graph, = struct.unpack_from("<i", buf, offset)
    offset += 4
    if n_graph > 0:
        lane_ids = list(struct.unpack_from(f"<{n_graph}i", buf, offset))
        offset += 4 * n_graph
        lane_lengths = np.frombuffer(buf, dtype=np.float32, count=n_graph, offset=offset).copy()
        offset += 4 * n_graph
        distances = np.frombuffer(buf, dtype=np.float32, count=n_graph * n_graph, offset=offset).copy()
        offset += 4 * n_graph * n_graph
        lane_graph = {"lane_ids": lane_ids, "lane_lengths": lane_lengths, "distances": distances.reshape(n_graph, n_graph)}
    else:
        lane_graph = None

    # --- Metadata ---
    scenario_id = buf[offset:offset + 128].rstrip(b"\0").decode("utf-8", errors="replace")
    offset += 128
    dataset = buf[offset:offset + 32].rstrip(b"\0").decode("utf-8", errors="replace")
    offset += 32
    scenario_length, = struct.unpack_from("<i", buf, offset)
    offset += 4
    dt, = struct.unpack_from("<f", buf, offset)
    offset += 4
    ooi, offset = _read_int_list(buf, offset)
    ttp, offset = _read_int_list(buf, offset)

    metadata = {
        "scenario_id": scenario_id,
        "dataset": dataset,
        "scenario_length": scenario_length,
        "dt": dt,
        "objects_of_interest": ooi,
        "tracks_to_predict": ttp,
    }

    return {
        "agents": agents,
        "road_map": road_map,
        "traffic_controls": traffic_controls,
        "objects": objects,
        "lane_graph": lane_graph,
        "metadata": metadata,
    }
