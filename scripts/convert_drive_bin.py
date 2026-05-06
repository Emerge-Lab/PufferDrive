#!/usr/bin/env python3
import argparse
import math
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


CURRENT_AGENT_TYPES = {1, 2, 3}
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


class Reader:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        self.off = 0

    def remaining(self):
        return len(self.data) - self.off

    def read(self, fmt):
        size = struct.calcsize("<" + fmt)
        if self.off + size > len(self.data):
            raise EOFError(f"{self.path}: wanted {size} bytes at {self.off}, have {self.remaining()}")
        out = struct.unpack_from("<" + fmt, self.data, self.off)
        self.off += size
        return out[0] if len(out) == 1 else out

    def read_bytes(self, n):
        if self.off + n > len(self.data):
            raise EOFError(f"{self.path}: wanted {n} bytes at {self.off}, have {self.remaining()}")
        out = self.data[self.off : self.off + n]
        self.off += n
        return out

    def read_array(self, fmt, n):
        if n <= 0:
            return []
        size = struct.calcsize("<" + fmt) * n
        if self.off + size > len(self.data):
            raise EOFError(f"{self.path}: wanted {size} bytes at {self.off}, have {self.remaining()}")
        out = struct.unpack_from("<" + fmt * n, self.data, self.off)
        self.off += size
        return list(out)

    def skip(self, n):
        if self.off + n > len(self.data):
            raise EOFError(f"{self.path}: wanted skip {n} bytes at {self.off}, have {self.remaining()}")
        self.off += n


def clean_cstr(raw):
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def counts(values):
    out = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def summarize_points(xs, ys):
    if not xs or not ys:
        return None
    return {"n": len(xs), "x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)}


def first_valid_dims(lengths, widths, heights, valid):
    for i, v in enumerate(valid):
        if v:
            return (lengths[i], widths[i], heights[i])
    return None


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
            road["entries"] = r.read_array("i", r.read("i"))
            road["exits"] = r.read_array("i", r.read("i"))
            road["speed_limit"] = r.read("f")
        roads.append(road)

    for _ in range(num_traffic):
        r.read("i")
        r.read("i")
        r.read_array("f", 6)
        r.read("f")
        r.read_array("i", r.read("i"))
        r.read_array("i", r.read("i"))

    for _ in range(num_objects):
        r.read("i")
        r.read("i")
        tlen = r.read("i")
        r.skip(9 * tlen * struct.calcsize("<f") + tlen * struct.calcsize("<i"))

    n_lanes_graph = r.read("i")
    r.read_array("i", n_lanes_graph)
    r.read_array("f", n_lanes_graph)
    r.read_array("f", n_lanes_graph * n_lanes_graph)

    scenario_id = clean_cstr(r.read_bytes(128))
    dataset_name = clean_cstr(r.read_bytes(32))
    log_length = r.read("i")
    log_dt = r.read("f")
    objects_of_interest = r.read_array("i", r.read("i"))
    tracks_to_predict = r.read_array("i", r.read("i"))

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


def parse_new(path):
    data = parse_new_for_conversion(path)
    route_lengths = [len(agent["route"]) for agent in data["agents"]]
    speed_limits = [
        road["speed_limit"]
        for road in data["roads"]
        if "speed_limit" in road and road["speed_limit"] > 0 and math.isfinite(road["speed_limit"])
    ]
    total_exits = sum(len(road.get("exits", [])) for road in data["roads"] if is_new_lane(road["type"]))
    lane_count = sum(1 for road in data["roads"] if is_new_lane(road["type"]))

    return {
        "format": "new",
        "path": str(path),
        "size_bytes": Path(path).stat().st_size,
        "scenario_id": data["scenario_id"],
        "dataset_name": data["dataset_name"],
        "log_length": data["log_length"],
        "log_dt": data["log_dt"],
        "objects_of_interest": data["objects_of_interest"],
        "tracks_to_predict": data["tracks_to_predict"],
        "num_agents": len(data["agents"]),
        "num_roads": len(data["roads"]),
        "agent_type_counts": counts(agent["type"] for agent in data["agents"]),
        "road_type_counts": counts(road["type"] for road in data["roads"]),
        "lane_count": lane_count,
        "avg_lane_exits": total_exits / lane_count if lane_count else 0.0,
        "speed_limit_min_max": (min(speed_limits), max(speed_limits)) if speed_limits else None,
        "route_length_min_max": (min(route_lengths), max(route_lengths)) if route_lengths else None,
        "sample_agents": [
            {
                "id": agent["id"],
                "type": agent["type"],
                "tlen": agent["tlen"],
                "valid_count": sum(1 for v in agent["valid"] if v),
                "route_length": len(agent["route"]),
                "route_head": agent["route"][:8],
                "route_gt_len": agent["route_gt_len"],
                "goal": agent["goal"],
                "mark_as_expert": agent["mark_as_expert"],
                "bbox": summarize_points(agent["x"], agent["y"]),
                "first_heading": next((h for h, v in zip(agent["heading"], agent["valid"]) if v), None),
                "first_dims": first_valid_dims(agent["length"], agent["width"], agent["height"], agent["valid"]),
            }
            for agent in data["agents"][:3]
        ],
        "sample_roads": [
            {
                "id": road["id"],
                "type": road["type"],
                "slen": road["slen"],
                "bbox": summarize_points(road["x"], road["y"]),
                "first_heading": road["heading"][0] if road["heading"] else None,
                "speed_limit": road.get("speed_limit"),
                "entries": road.get("entries", [])[:8],
                "exits": road.get("exits", [])[:8],
            }
            for road in data["roads"][:5]
        ],
    }


def parse_current(path):
    r = Reader(path)
    scenario_id = clean_cstr(r.read_bytes(16))
    sdc_track_index = r.read("i")
    tracks_to_predict = r.read_array("i", r.read("i"))
    num_objects = r.read("i")
    num_roads = r.read("i")
    objects = []
    roads = []

    for idx in range(num_objects + num_roads):
        scenario_int = r.read("i")
        entity_type = r.read("i")
        entity_id = r.read("i")
        size = r.read("i")
        xs = r.read_array("f", size)
        ys = r.read_array("f", size)
        r.read_array("f", size)
        entity = {
            "idx": idx,
            "scenario_int": scenario_int,
            "type": entity_type,
            "id": entity_id,
            "size": size,
            "bbox": summarize_points(xs, ys),
        }
        if entity_type in CURRENT_AGENT_TYPES:
            r.read_array("f", size)
            r.read_array("f", size)
            r.read_array("f", size)
            headings = r.read_array("f", size)
            valid = r.read_array("i", size)
            for _ in range(5):
                r.read_array("f", size)
            entity["valid_count"] = sum(1 for v in valid if v)
            entity["first_heading"] = next((h for h, v in zip(headings, valid) if v), None)
            objects.append(entity)
        else:
            roads.append(entity)
        entity["width"] = r.read("f")
        entity["length"] = r.read("f")
        entity["height"] = r.read("f")
        entity["goal"] = (r.read("f"), r.read("f"), r.read("f"))
        entity["mark_as_expert"] = r.read("i")

    return {
        "format": "current",
        "path": str(path),
        "size_bytes": len(r.data),
        "bytes_remaining": r.remaining(),
        "scenario_id": scenario_id,
        "sdc_track_index": sdc_track_index,
        "tracks_to_predict": tracks_to_predict,
        "num_objects": num_objects,
        "num_roads": num_roads,
        "object_type_counts": counts(o["type"] for o in objects),
        "road_type_counts": counts(e["type"] for e in roads),
        "sample_objects": objects[:3],
        "sample_roads": roads[:5],
    }


def detect_format(path):
    raw = Path(path).read_bytes()
    if len(raw) < 16:
        return "unknown"
    first_four = struct.unpack_from("<4i", raw, 0)
    if all(0 <= x < 100000 for x in first_four) and first_four[0] > 0 and first_four[1] >= 0:
        return "new"
    return "current"


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
    f.write(pack_string(data["scenario_id"], 16))
    f.write(struct.pack("<i", int(sdc_track_index)))
    f.write(struct.pack("<i", len(tracks_to_predict)))
    write_ints(f, tracks_to_predict)
    f.write(struct.pack("<i", len(data["agents"])))
    f.write(struct.pack("<i", len(data["roads"])))

    for agent in data["agents"]:
        tlen = agent["tlen"]
        f.write(struct.pack("<iiii", int(unique_map_id), int(agent["type"]), int(agent["id"]), tlen))
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

    for road in data["roads"]:
        f.write(
            struct.pack("<iiii", int(unique_map_id), current_road_type(road["type"]), int(road["id"]), road["slen"])
        )
        write_floats(f, road["x"])
        write_floats(f, road["y"])
        write_floats(f, road["z"])
        f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
        f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
        f.write(struct.pack("<i", 0))


def convert(
    input_path, output_path, unique_map_id=0, sdc_track_index=None, append_idm_extension=True, expert_actions="missing"
):
    data = parse_new_for_conversion(input_path)
    original_bytes = Path(input_path).read_bytes()
    if sdc_track_index is None:
        sdc_track_index = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        write_current_payload(f, data, unique_map_id, sdc_track_index, data["tracks_to_predict"], expert_actions)
        if append_idm_extension:
            f.write(IDM_EXTENSION_MAGIC)
            f.write(struct.pack("<Q", len(original_bytes)))
            f.write(original_bytes)

    return {
        "scenario_id": data["scenario_id"],
        "num_agents": len(data["agents"]),
        "num_roads": len(data["roads"]),
        "tracks_to_predict": data["tracks_to_predict"],
        "sdc_track_index": sdc_track_index,
        "output_size": output_path.stat().st_size,
        "appended_idm_extension": append_idm_extension,
    }


def convert_one_for_batch(args_tuple):
    input_path, output_dir, file_index, append_idm_extension, expert_actions = args_tuple
    output_path = output_dir / f"map_{file_index:03d}.bin"
    try:
        convert(input_path, output_path, file_index, None, append_idm_extension, expert_actions)
        return input_path.name, output_path.name, True, None
    except Exception as exc:
        return input_path.name, output_path.name, False, str(exc)


def batch_convert(input_dir, output_dir, workers, append_idm_extension=True, expert_actions="missing"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_files = sorted(input_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {input_dir}")

    start_time = time.time()
    success_count = 0
    fail_count = 0
    jobs = [(path, output_dir, idx, append_idm_extension, expert_actions) for idx, path in enumerate(bin_files)]
    print(f"Converting {len(jobs)} files with {workers} workers into {output_dir}")

    if workers == 1:
        results = (convert_one_for_batch(job) for job in jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        futures = [executor.submit(convert_one_for_batch, job) for job in jobs]
        results = (future.result() for future in as_completed(futures))

    try:
        for i, result in enumerate(results, 1):
            original_name, new_name, success, error = result
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"\n[ERROR] {original_name} -> {new_name}: {error}")
            if i % 100 == 0 or i == len(jobs):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0.0
                print(
                    f"Progress: {i}/{len(jobs)} ({i / len(jobs) * 100:.1f}%) "
                    f"| Success: {success_count} | Failed: {fail_count} | Rate: {rate:.1f} files/sec",
                    end="\r",
                )
    finally:
        if workers != 1:
            executor.shutdown()
    print(f"\nDone. Success: {success_count}, failed: {fail_count}")
    if fail_count:
        raise RuntimeError(f"Failed to convert {fail_count} files")


def print_summary(summary):
    print(f"\n== {summary['path']} ==")
    sample_keys = {"sample_objects", "sample_roads", "sample_agents"}
    for key, value in summary.items():
        if key not in sample_keys and key != "path":
            print(f"{key}: {value}")
    for key in ("sample_objects", "sample_agents", "sample_roads"):
        if key in summary:
            print(f"{key}:")
            for item in summary[key]:
                print(f"  {item}")


def inspect(paths, fmt):
    for path in paths:
        actual_fmt = detect_format(path) if fmt == "auto" else fmt
        try:
            if actual_fmt == "current":
                summary = parse_current(path)
            elif actual_fmt == "new":
                summary = parse_new(path)
            else:
                raise ValueError(f"could not detect format for {path}")
            print_summary(summary)
        except Exception as exc:
            print(f"\n== {path} ==\nformat: {actual_fmt}\nparse_error: {exc}")


def add_convert_args(parser):
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--unique-map-id", type=int, default=0)
    parser.add_argument("--sdc-track-index", type=int, default=None)
    parser.add_argument("--no-idm-extension", action="store_true")
    parser.add_argument("--expert-actions", choices=["missing", "zero"], default="missing")


def run_convert(args):
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


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] not in {"convert", "batch", "inspect", "-h", "--help"}:
        parser = argparse.ArgumentParser(description="Convert one new Drive binary to current Drive format.")
        add_convert_args(parser)
        return run_convert(parser.parse_args(argv))

    parser = argparse.ArgumentParser(description="Inspect and convert Drive binary formats.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert one new-format binary")
    add_convert_args(convert_parser)

    batch_parser = subparsers.add_parser("batch", help="Convert a directory to sequential map_XXX.bin files")
    batch_parser.add_argument("--input-dir", required=True, type=Path)
    batch_parser.add_argument("--output-dir", required=True, type=Path)
    batch_parser.add_argument("--workers", type=int, default=24)
    batch_parser.add_argument("--no-idm-extension", action="store_true")
    batch_parser.add_argument("--expert-actions", choices=["missing", "zero"], default="missing")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect current or new-format binaries")
    inspect_parser.add_argument("paths", nargs="+")
    inspect_parser.add_argument("--format", choices=["auto", "current", "new"], default="auto")

    args = parser.parse_args(argv)
    if args.command == "convert":
        return run_convert(args)
    if args.command == "batch":
        return batch_convert(
            args.input_dir,
            args.output_dir,
            args.workers,
            append_idm_extension=not args.no_idm_extension,
            expert_actions=args.expert_actions,
        )
    if args.command == "inspect":
        return inspect(args.paths, args.format)
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
