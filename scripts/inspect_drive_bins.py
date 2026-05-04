#!/usr/bin/env python3
import argparse
import math
import struct
from pathlib import Path


CURRENT_AGENT_TYPES = {1, 2, 3}
CURRENT_ROAD_TYPES = {4, 5, 6, 7, 8, 9, 10}


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


def clean_cstr(raw):
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def summarize_points(xs, ys):
    if not xs or not ys:
        return None
    return {
        "n": len(xs),
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
    }


def parse_current(path):
    r = Reader(path)
    scenario_id = clean_cstr(r.read_bytes(16))
    sdc_track_index = r.read("i")
    num_tracks_to_predict = r.read("i")
    tracks_to_predict = r.read_array("i", num_tracks_to_predict)
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
        zs = r.read_array("f", size)

        entity = {
            "idx": idx,
            "scenario_int": scenario_int,
            "type": entity_type,
            "id": entity_id,
            "size": size,
            "bbox": summarize_points(xs, ys),
        }

        if entity_type in CURRENT_AGENT_TYPES:
            r.read_array("f", size)  # vx
            r.read_array("f", size)  # vy
            r.read_array("f", size)  # vz
            headings = r.read_array("f", size)
            valid = r.read_array("i", size)
            r.read_array("f", size)  # expert_accel
            r.read_array("f", size)  # expert_steering
            r.read_array("f", size)  # expert_delta_x
            r.read_array("f", size)  # expert_delta_y
            r.read_array("f", size)  # expert_delta_yaw
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


def parse_new(path):
    r = Reader(path)
    num_agents = r.read("i")
    num_roads = r.read("i")
    num_traffic = r.read("i")
    num_objects = r.read("i")

    agents = []
    route_lengths = []
    for i in range(num_agents):
        agent_id = r.read("i")
        agent_type = r.read("i")
        tlen = r.read("i")
        xs = r.read_array("f", tlen)
        ys = r.read_array("f", tlen)
        r.read_array("f", tlen)  # z
        headings = r.read_array("f", tlen)
        r.read_array("f", tlen)  # vx
        r.read_array("f", tlen)  # vy
        lengths = r.read_array("f", tlen)
        widths = r.read_array("f", tlen)
        heights = r.read_array("f", tlen)
        valid = r.read_array("i", tlen)
        route_length = r.read("i")
        route = r.read_array("i", route_length)
        route_gt_len = r.read("i")
        goal = (r.read("f"), r.read("f"), r.read("f"))
        mark_as_expert = r.read("i")
        route_lengths.append(route_length)
        agents.append(
            {
                "id": agent_id,
                "type": agent_type,
                "tlen": tlen,
                "valid_count": sum(1 for v in valid if v),
                "route_length": route_length,
                "route_head": route[:8],
                "route_gt_len": route_gt_len,
                "goal": goal,
                "mark_as_expert": mark_as_expert,
                "bbox": summarize_points(xs, ys),
                "first_heading": next((h for h, v in zip(headings, valid) if v), None),
                "first_dims": first_valid_dims(lengths, widths, heights, valid),
            }
        )

    roads = []
    lane_count = 0
    speed_limits = []
    total_exits = 0
    for _ in range(num_roads):
        road_id = r.read("i")
        road_type = r.read("i")
        slen = r.read("i")
        xs = r.read_array("f", slen)
        ys = r.read_array("f", slen)
        r.read_array("f", slen)  # z
        headings = r.read_array("f", slen)
        road = {
            "id": road_id,
            "type": road_type,
            "slen": slen,
            "bbox": summarize_points(xs, ys),
            "first_heading": headings[0] if headings else None,
        }
        if 0 <= road_type <= 9:
            num_entries = r.read("i")
            entries = r.read_array("i", num_entries)
            num_exits = r.read("i")
            exits = r.read_array("i", num_exits)
            speed_limit = r.read("f")
            lane_count += 1
            total_exits += num_exits
            if speed_limit > 0 and math.isfinite(speed_limit):
                speed_limits.append(speed_limit)
            road.update(
                {
                    "num_entries": num_entries,
                    "entries": entries[:8],
                    "num_exits": num_exits,
                    "exits": exits[:8],
                    "speed_limit": speed_limit,
                }
            )
        roads.append(road)

    traffics = []
    for _ in range(num_traffic):
        traffic_id = r.read("i")
        traffic_type = r.read("i")
        stop_line = r.read_array("f", 6)
        heading = r.read("f")
        state_length = r.read("i")
        states = r.read_array("i", state_length)
        num_controlled_lanes = r.read("i")
        controlled_lanes = r.read_array("i", num_controlled_lanes)
        traffics.append(
            {
                "id": traffic_id,
                "type": traffic_type,
                "heading": heading,
                "state_length": state_length,
                "state_values": sorted(set(states)),
                "num_controlled_lanes": num_controlled_lanes,
                "controlled_lanes": controlled_lanes[:8],
                "stop_line": stop_line,
            }
        )

    skipped_objects = []
    for _ in range(num_objects):
        obj_id = r.read("i")
        obj_type = r.read("i")
        tlen = r.read("i")
        r.off += 9 * tlen * struct.calcsize("<f") + tlen * struct.calcsize("<i")
        skipped_objects.append({"id": obj_id, "type": obj_type, "tlen": tlen})

    n_lanes_graph = r.read("i")
    lane_ids = r.read_array("i", n_lanes_graph)
    lane_lengths = r.read_array("f", n_lanes_graph)
    distance_count = n_lanes_graph * n_lanes_graph
    finite_distance_count = 0
    if distance_count:
        distances = r.read_array("f", distance_count)
        finite_distance_count = sum(1 for d in distances if math.isfinite(d))

    scenario_id = clean_cstr(r.read_bytes(128))
    dataset_name = clean_cstr(r.read_bytes(32))
    log_length = r.read("i")
    log_dt = r.read("f")
    num_objects_of_interest = r.read("i")
    objects_of_interest = r.read_array("i", num_objects_of_interest)
    num_tracks_to_predict = r.read("i")
    tracks_to_predict = r.read_array("i", num_tracks_to_predict)

    return {
        "format": "new",
        "path": str(path),
        "size_bytes": len(r.data),
        "bytes_remaining": r.remaining(),
        "num_agents": num_agents,
        "num_roads": num_roads,
        "num_traffic": num_traffic,
        "num_objects": num_objects,
        "scenario_id": scenario_id,
        "dataset_name": dataset_name,
        "log_length": log_length,
        "log_dt": log_dt,
        "objects_of_interest": objects_of_interest,
        "tracks_to_predict": tracks_to_predict,
        "agent_type_counts": counts(a["type"] for a in agents),
        "road_type_counts": counts(r["type"] for r in roads),
        "lane_count": lane_count,
        "avg_lane_exits": total_exits / lane_count if lane_count else 0.0,
        "speed_limit_min_max": (min(speed_limits), max(speed_limits)) if speed_limits else None,
        "route_length_min_max": (min(route_lengths), max(route_lengths)) if route_lengths else None,
        "lane_graph_n": n_lanes_graph,
        "lane_graph_sample_ids": lane_ids[:8],
        "lane_graph_sample_lengths": lane_lengths[:8],
        "lane_graph_finite_distances": finite_distance_count,
        "sample_agents": agents[:3],
        "sample_roads": roads[:5],
        "sample_traffic": traffics[:3],
        "skipped_objects": skipped_objects[:3],
    }


def first_valid_dims(lengths, widths, heights, valid):
    for i, v in enumerate(valid):
        if v:
            return (lengths[i], widths[i], heights[i])
    return None


def counts(values):
    out = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def detect_format(path):
    raw = Path(path).read_bytes()
    if len(raw) < 16:
        return "unknown"
    first_four = struct.unpack_from("<4i", raw, 0)
    if all(0 <= x < 100000 for x in first_four) and first_four[0] > 0 and first_four[1] >= 0:
        return "new"
    return "current"


def print_summary(summary):
    print(f"\n== {summary['path']} ==")
    for key, value in summary.items():
        if key in {"path", "sample_objects", "sample_roads", "sample_agents", "sample_traffic", "skipped_objects"}:
            continue
        print(f"{key}: {value}")
    for key in ("sample_objects", "sample_agents", "sample_roads", "sample_traffic", "skipped_objects"):
        if key in summary:
            print(f"{key}:")
            for item in summary[key]:
                print(f"  {item}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--format", choices=["auto", "current", "new"], default="auto")
    args = parser.parse_args()

    for path in args.paths:
        fmt = detect_format(path) if args.format == "auto" else args.format
        try:
            if fmt == "current":
                summary = parse_current(path)
            elif fmt == "new":
                summary = parse_new(path)
            else:
                raise ValueError(f"Could not detect format for {path}")
        except Exception as exc:
            print(f"\n== {path} ==")
            print(f"format: {fmt}")
            print(f"parse_error: {exc}")
            continue
        print_summary(summary)


if __name__ == "__main__":
    main()
