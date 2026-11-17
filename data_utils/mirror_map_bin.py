"""Read and write OpenGigaFlow map .bin files."""

import struct
from pathlib import Path


TRAFFIC_PHASE_SECTION_TAG = b"TLPHASE1"
LANE_WIDTH_SECTION_TAG = b"LANEWID1"
SPEED_ZONE_SECTION_TAG = b"SPDZONE1"
AGENT_FLOAT_COLUMNS = ("x", "y", "z", "h", "vx", "vy", "len", "wid", "hgt")


def _is_lane(road_type: int) -> bool:
    return 0 <= road_type <= 9


def _read(fmt, f):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def _read_f_array(f, n):
    return struct.unpack(f"<{n}f", f.read(4 * n)) if n else ()


def _read_i_array(f, n):
    return struct.unpack(f"<{n}i", f.read(4 * n)) if n else ()


def _read_i_list(f):
    (count,) = _read("<i", f)
    return _read_i_array(f, count)


def _write_i_list(f, values):
    f.write(struct.pack(f"<i{len(values)}i", len(values), *values))


def read_bin(path: Path) -> dict:
    with open(path, "rb") as f:
        num_agents, num_roads, num_traffic, num_objects = _read("<iiii", f)

        agents = []
        for i in range(num_agents):
            agent_id, agent_type = _read("<ii", f)
            assert agent_id == i, f"agent id {agent_id} != idx {i}"
            (T,) = _read("<i", f)
            cols = {k: _read_f_array(f, T) for k in AGENT_FLOAT_COLUMNS}
            cols["valid"] = _read_i_array(f, T)
            route = _read_i_list(f)
            (route_gt_len,) = _read("<i", f)
            goal = _read("<fff", f)
            (mark_as_expert,) = _read("<i", f)
            agents.append(
                {
                    "id": agent_id,
                    "type": agent_type,
                    "T": T,
                    "cols": cols,
                    "route": route,
                    "route_gt_len": route_gt_len,
                    "goal": goal,
                    "mark_as_expert": mark_as_expert,
                }
            )

        roads = []
        for i in range(num_roads):
            road_id, road_type = _read("<ii", f)
            assert road_id == i, f"road id {road_id} != idx {i}"
            (S,) = _read("<i", f)
            road = {
                "id": road_id,
                "type": road_type,
                "S": S,
                "x": _read_f_array(f, S),
                "y": _read_f_array(f, S),
                "z": _read_f_array(f, S),
                "headings": _read_f_array(f, S),
            }
            if _is_lane(road_type):
                road["entry_lanes"] = _read_i_list(f)
                road["exit_lanes"] = _read_i_list(f)
                (road["speed_limit"],) = _read("<f", f)
                (road["length"],) = _read("<f", f)
                road["cum_lengths"] = _read_f_array(f, S)
            roads.append(road)

        traffic = []
        for i in range(num_traffic):
            traffic_id, traffic_type = _read("<ii", f)
            assert traffic_id == i, f"traffic id {traffic_id} != idx {i}"
            stop_line = _read("<6f", f)
            (heading,) = _read("<f", f)
            states = _read_i_list(f)
            ctrl_lanes = _read_i_list(f)
            traffic.append(
                {
                    "id": traffic_id,
                    "type": traffic_type,
                    "stop_line": stop_line,
                    "heading": heading,
                    "states": states,
                    "controlled_lanes": ctrl_lanes,
                }
            )

        objects = []
        for i in range(num_objects):
            obj_id, obj_type, T = _read("<iii", f)
            raw = f.read((9 * 4 + 4) * T)  # 9 float arrays + 1 int array of length T
            objects.append({"id": obj_id, "type": obj_type, "T": T, "raw": raw})

        (n_lanes_g,) = _read("<i", f)
        lane_ids = _read_i_array(f, n_lanes_g)
        distances = _read_f_array(f, n_lanes_g * n_lanes_g)

        scenario_id = f.read(128)
        dataset_name = f.read(32)
        (log_length,) = _read("<i", f)
        (log_dt,) = _read("<f", f)
        objects_of_interest = _read_i_list(f)
        tracks_to_predict = _read_i_list(f)

        phase_tag = f.read(len(TRAFFIC_PHASE_SECTION_TAG))
        has_phase_section = phase_tag == TRAFFIC_PHASE_SECTION_TAG
        assert has_phase_section or phase_tag == b"", f"unexpected bytes after metadata in {path}"
        for t in traffic:
            t["junction_id"], t["phase_idx"] = _read("<ii", f) if has_phase_section else (-1, -1)

        width_tag = f.read(len(LANE_WIDTH_SECTION_TAG))
        has_width_section = width_tag == LANE_WIDTH_SECTION_TAG
        assert has_width_section or width_tag == b"", f"unexpected bytes after phase section in {path}"
        if has_width_section:
            for r in roads:
                if _is_lane(r["type"]):
                    r["widths"] = _read_f_array(f, r["S"])

        zone_tag = f.read(len(SPEED_ZONE_SECTION_TAG))
        has_zone_section = zone_tag == SPEED_ZONE_SECTION_TAG
        assert has_zone_section or zone_tag == b"", f"unexpected bytes after width section in {path}"
        for r in roads:
            if _is_lane(r["type"]):
                (r["speed_zone_idx"],) = _read("<i", f) if has_zone_section else (-1,)

        trailing = f.read()
        assert not trailing, f"{len(trailing)} unparsed trailing bytes in {path}"

    return {
        "agents": agents,
        "roads": roads,
        "traffic": traffic,
        "objects": objects,
        "lane_graph": {"n": n_lanes_g, "lane_ids": lane_ids, "distances": distances},
        "scenario_id": scenario_id,
        "dataset_name": dataset_name,
        "log_length": log_length,
        "log_dt": log_dt,
        "objects_of_interest": objects_of_interest,
        "tracks_to_predict": tracks_to_predict,
        "has_phase_section": has_phase_section,
        "has_width_section": has_width_section,
        "has_zone_section": has_zone_section,
    }


def write_bin(data: dict, path: Path):
    with open(path, "wb") as f:
        agents, roads, traffic, objects = data["agents"], data["roads"], data["traffic"], data["objects"]
        lane_roads = [r for r in roads if _is_lane(r["type"])]
        f.write(struct.pack("<iiii", len(agents), len(roads), len(traffic), len(objects)))

        for a in agents:
            T = a["T"]
            f.write(struct.pack("<iii", a["id"], a["type"], T))
            f.writelines(struct.pack(f"<{T}f", *a["cols"][k]) for k in AGENT_FLOAT_COLUMNS)
            f.write(struct.pack(f"<{T}i", *a["cols"]["valid"]))
            _write_i_list(f, a["route"])
            f.write(struct.pack("<i3fi", a["route_gt_len"], *a["goal"], a["mark_as_expert"]))

        for r in roads:
            S = r["S"]
            f.write(struct.pack("<iii", r["id"], r["type"], S))
            f.writelines(struct.pack(f"<{S}f", *r[k]) for k in ("x", "y", "z", "headings"))
            if _is_lane(r["type"]):
                _write_i_list(f, r["entry_lanes"])
                _write_i_list(f, r["exit_lanes"])
                f.write(struct.pack("<ff", r["speed_limit"], r["length"]))
                f.write(struct.pack(f"<{S}f", *r["cum_lengths"]))

        for t in traffic:
            f.write(struct.pack("<ii6ff", t["id"], t["type"], *t["stop_line"], t["heading"]))
            _write_i_list(f, t["states"])
            _write_i_list(f, t["controlled_lanes"])

        for o in objects:
            f.write(struct.pack("<iii", o["id"], o["type"], o["T"]))
            f.write(o["raw"])

        n_lanes_g = data["lane_graph"]["n"]
        f.write(struct.pack(f"<i{n_lanes_g}i", n_lanes_g, *data["lane_graph"]["lane_ids"]))
        f.write(struct.pack(f"<{n_lanes_g * n_lanes_g}f", *data["lane_graph"]["distances"]))

        f.write(data["scenario_id"])
        f.write(data["dataset_name"])
        f.write(struct.pack("<if", data["log_length"], data["log_dt"]))
        _write_i_list(f, data["objects_of_interest"])
        _write_i_list(f, data["tracks_to_predict"])
        if data["has_phase_section"]:
            f.write(TRAFFIC_PHASE_SECTION_TAG)
            for t in traffic:
                f.write(struct.pack("<ii", t["junction_id"], t["phase_idx"]))
        if data["has_width_section"]:
            f.write(LANE_WIDTH_SECTION_TAG)
            for r in lane_roads:
                f.write(struct.pack(f"<{r['S']}f", *r["widths"]))
        if data["has_zone_section"]:
            f.write(SPEED_ZONE_SECTION_TAG)
            for r in lane_roads:
                f.write(struct.pack("<i", r["speed_zone_idx"]))
