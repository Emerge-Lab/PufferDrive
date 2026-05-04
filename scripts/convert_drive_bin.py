#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path

from inspect_drive_bins import parse_new


CURRENT_VEHICLE = 1
CURRENT_PEDESTRIAN = 2
CURRENT_CYCLIST = 3
CURRENT_ROAD_LANE = 4
CURRENT_ROAD_LINE = 5
CURRENT_ROAD_EDGE = 6
CURRENT_CROSSWALK = 8
CURRENT_SPEED_BUMP = 9
CURRENT_DRIVEWAY = 10

IDM_EXTENSION_MAGIC = b"PDRV_IDM_EXT_V1\0"


def read_new_full(path):
    summary = parse_new(path)
    data = Path(path).read_bytes()
    return summary, data


class Reader:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.off = 0

    def read(self, fmt):
        size = struct.calcsize("<" + fmt)
        if self.off + size > len(self.data):
            raise EOFError(f"{self.path}: wanted {size} bytes at {self.off}")
        out = struct.unpack_from("<" + fmt, self.data, self.off)
        self.off += size
        return out[0] if len(out) == 1 else out

    def read_bytes(self, n):
        if self.off + n > len(self.data):
            raise EOFError(f"{self.path}: wanted {n} bytes at {self.off}")
        out = self.data[self.off : self.off + n]
        self.off += n
        return out

    def read_array(self, fmt, n):
        if n <= 0:
            return []
        size = struct.calcsize("<" + fmt) * n
        if self.off + size > len(self.data):
            raise EOFError(f"{self.path}: wanted {size} bytes at {self.off}")
        out = struct.unpack_from("<" + fmt * n, self.data, self.off)
        self.off += size
        return list(out)


def parse_new_for_conversion(path):
    r = Reader(path)
    num_agents = r.read("i")
    num_roads = r.read("i")
    num_traffic = r.read("i")
    num_objects = r.read("i")

    agents = []
    for _ in range(num_agents):
        agent = {}
        agent["id"] = r.read("i")
        agent["type"] = r.read("i")
        tlen = r.read("i")
        agent["tlen"] = tlen
        agent["x"] = r.read_array("f", tlen)
        agent["y"] = r.read_array("f", tlen)
        agent["z"] = r.read_array("f", tlen)
        agent["heading"] = r.read_array("f", tlen)
        agent["vx"] = r.read_array("f", tlen)
        agent["vy"] = r.read_array("f", tlen)
        agent["length"] = r.read_array("f", tlen)
        agent["width"] = r.read_array("f", tlen)
        agent["height"] = r.read_array("f", tlen)
        agent["valid"] = r.read_array("i", tlen)
        route_length = r.read("i")
        agent["route"] = r.read_array("i", route_length)
        agent["route_gt_len"] = r.read("i")
        agent["goal"] = (r.read("f"), r.read("f"), r.read("f"))
        agent["mark_as_expert"] = r.read("i")
        agents.append(agent)

    roads = []
    for _ in range(num_roads):
        road = {}
        road["id"] = r.read("i")
        road["type"] = r.read("i")
        slen = r.read("i")
        road["slen"] = slen
        road["x"] = r.read_array("f", slen)
        road["y"] = r.read_array("f", slen)
        road["z"] = r.read_array("f", slen)
        road["heading"] = r.read_array("f", slen)
        if is_new_lane(road["type"]):
            num_entries = r.read("i")
            road["entries"] = r.read_array("i", num_entries)
            num_exits = r.read("i")
            road["exits"] = r.read_array("i", num_exits)
            road["speed_limit"] = r.read("f")
        roads.append(road)

    for _ in range(num_traffic):
        r.read("i")  # id
        r.read("i")  # type
        r.read_array("f", 6)
        r.read("f")  # heading
        state_length = r.read("i")
        r.read_array("i", state_length)
        num_controlled_lanes = r.read("i")
        r.read_array("i", num_controlled_lanes)

    for _ in range(num_objects):
        r.read("i")
        r.read("i")
        tlen = r.read("i")
        r.off += 9 * tlen * struct.calcsize("<f") + tlen * struct.calcsize("<i")

    n_lanes_graph = r.read("i")
    r.read_array("i", n_lanes_graph)
    r.read_array("f", n_lanes_graph)
    r.read_array("f", n_lanes_graph * n_lanes_graph)

    scenario_id = clean_cstr(r.read_bytes(128))
    dataset_name = clean_cstr(r.read_bytes(32))
    log_length = r.read("i")
    log_dt = r.read("f")
    num_objects_of_interest = r.read("i")
    objects_of_interest = r.read_array("i", num_objects_of_interest)
    num_tracks_to_predict = r.read("i")
    tracks_to_predict = r.read_array("i", num_tracks_to_predict)

    return {
        "agents": agents,
        "roads": roads,
        "scenario_id": scenario_id,
        "dataset_name": dataset_name,
        "log_length": log_length,
        "log_dt": log_dt,
        "objects_of_interest": objects_of_interest,
        "tracks_to_predict": tracks_to_predict,
    }


def clean_cstr(raw):
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def is_new_lane(road_type):
    return 0 <= road_type <= 9


def current_road_type(new_type):
    if 0 <= new_type <= 9:
        return CURRENT_ROAD_LANE
    if 10 <= new_type <= 19:
        return CURRENT_ROAD_LINE
    if 20 <= new_type <= 29:
        return CURRENT_ROAD_EDGE
    if new_type == 31:
        return CURRENT_CROSSWALK
    if new_type == 32:
        return CURRENT_SPEED_BUMP
    return CURRENT_DRIVEWAY


def first_valid_scalar(values, valid, default=0.0):
    for value, is_valid in zip(values, valid):
        if is_valid:
            return float(value)
    return float(values[0]) if values else default


def pack_string(text, length):
    raw = text.encode("utf-8")[:length]
    return raw + b"\0" * (length - len(raw))


def write_floats(f, values):
    if values:
        f.write(struct.pack(f"<{len(values)}f", *[float(v) for v in values]))


def write_ints(f, values):
    if values:
        f.write(struct.pack(f"<{len(values)}i", *[int(v) for v in values]))


def write_current_payload(f, data, unique_map_id, sdc_track_index, tracks_to_predict, expert_actions):
    agents = data["agents"]
    roads = data["roads"]

    f.write(pack_string(data["scenario_id"], 16))
    f.write(struct.pack("<i", int(sdc_track_index)))
    f.write(struct.pack("<i", len(tracks_to_predict)))
    write_ints(f, tracks_to_predict)
    f.write(struct.pack("<i", len(agents)))
    f.write(struct.pack("<i", len(roads)))

    for agent in agents:
        tlen = agent["tlen"]
        f.write(struct.pack("<i", int(unique_map_id)))
        f.write(struct.pack("<i", int(agent["type"])))
        f.write(struct.pack("<i", int(agent["id"])))
        f.write(struct.pack("<i", tlen))
        write_floats(f, agent["x"])
        write_floats(f, agent["y"])
        write_floats(f, agent["z"])
        write_floats(f, agent["vx"])
        write_floats(f, agent["vy"])
        write_floats(f, [0.0] * tlen)
        write_floats(f, agent["heading"])
        write_ints(f, agent["valid"])

        action_fill = 0.0 if expert_actions == "zero" else -1.0
        for _ in range(5):
            write_floats(f, [action_fill] * tlen)

        f.write(struct.pack("<f", first_valid_scalar(agent["width"], agent["valid"])))
        f.write(struct.pack("<f", first_valid_scalar(agent["length"], agent["valid"])))
        f.write(struct.pack("<f", first_valid_scalar(agent["height"], agent["valid"])))
        f.write(struct.pack("<fff", *agent["goal"]))
        f.write(struct.pack("<i", int(agent["mark_as_expert"])))

    for road in roads:
        f.write(struct.pack("<i", int(unique_map_id)))
        f.write(struct.pack("<i", current_road_type(road["type"])))
        f.write(struct.pack("<i", int(road["id"])))
        f.write(struct.pack("<i", int(road["slen"])))
        write_floats(f, road["x"])
        write_floats(f, road["y"])
        write_floats(f, road["z"])
        f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
        f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
        f.write(struct.pack("<i", 0))


def convert(input_path, output_path, unique_map_id, sdc_track_index, append_idm_extension, expert_actions):
    data = parse_new_for_conversion(input_path)
    original_bytes = Path(input_path).read_bytes()

    if sdc_track_index is None:
        sdc_track_index = 0

    tracks_to_predict = data["tracks_to_predict"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        write_current_payload(f, data, unique_map_id, sdc_track_index, tracks_to_predict, expert_actions)
        if append_idm_extension:
            f.write(IDM_EXTENSION_MAGIC)
            f.write(struct.pack("<Q", len(original_bytes)))
            f.write(original_bytes)

    return {
        "scenario_id": data["scenario_id"],
        "num_agents": len(data["agents"]),
        "num_roads": len(data["roads"]),
        "tracks_to_predict": tracks_to_predict,
        "sdc_track_index": sdc_track_index,
        "output_size": output_path.stat().st_size,
        "appended_idm_extension": append_idm_extension,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert new Drive binary files to gsp_v0-compatible binaries.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--unique-map-id", type=int, default=0)
    parser.add_argument("--sdc-track-index", type=int, default=None)
    parser.add_argument("--no-idm-extension", action="store_true")
    parser.add_argument("--expert-actions", choices=["missing", "zero"], default="missing")
    args = parser.parse_args()

    summary = convert(
        input_path=args.input,
        output_path=args.output,
        unique_map_id=args.unique_map_id,
        sdc_track_index=args.sdc_track_index,
        append_idm_extension=not args.no_idm_extension,
        expert_actions=args.expert_actions,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
