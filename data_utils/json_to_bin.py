"""Convert old-format WOMD JSON scenario files to PufferDrive binary format.

Binary layout matches serialize.py in 123Drive/src/bin_factory/serialize.py.

Usage:
    python data_utils/json_to_bin.py \
        --input  data/womd_val_scenario_classes \
        --output pufferlib/resources/drive/binaries/womd_val_scenario_classes
"""

import argparse
import json
import struct
import os
import math
import numpy as np
from pathlib import Path


# ── road-type mapping (JSON map_element_id → puffer_types int) ────────────────
# LaneType:      0-9   (0=UNKNOWN,1=FREEWAY,2=SURFACE_STREET,3=BIKE_LANE,4=BUS)
# RoadLineType: 10-19  (10=UNKNOWN,11=BROKEN_WHITE,12=SOLID_WHITE,…)
# RoadEdgeType: 20-29  (20=UNKNOWN,21=BOUNDARY,22=MEDIAN)
# MiscRoadType: 30+    (30=UNKNOWN,31=CROSSWALK,32=SPEED_BUMP)

WOMD_MAP_ELEMENT_ID_TO_PUFFER = {
    # lanes
    0: 0,  # LANE_UNDEFINED  → LaneType.UNKNOWN
    1: 1,  # LANE_FREEWAY    → LaneType.FREEWAY
    2: 2,  # LANE_SURFACE_STREET → LaneType.SURFACE_STREET
    3: 3,  # LANE_BIKE_LANE  → LaneType.BIKE_LANE
    # road lines
    5: 10,  # ROAD_LINE_UNKNOWN              → RoadLineType.UNKNOWN
    6: 11,  # ROAD_LINE_BROKEN_SINGLE_WHITE  → RoadLineType.BROKEN_SINGLE_WHITE
    7: 12,  # ROAD_LINE_SOLID_SINGLE_WHITE   → RoadLineType.SOLID_SINGLE_WHITE
    8: 13,  # ROAD_LINE_SOLID_DOUBLE_WHITE   → RoadLineType.SOLID_DOUBLE_WHITE
    9: 14,  # ROAD_LINE_BROKEN_SINGLE_YELLOW → RoadLineType.BROKEN_SINGLE_YELLOW
    10: 15,  # ROAD_LINE_BROKEN_DOUBLE_YELLOW → RoadLineType.BROKEN_DOUBLE_YELLOW
    11: 16,  # ROAD_LINE_SOLID_SINGLE_YELLOW  → RoadLineType.SOLID_SINGLE_YELLOW
    12: 17,  # ROAD_LINE_SOLID_DOUBLE_YELLOW  → RoadLineType.SOLID_DOUBLE_YELLOW
    13: 18,  # ROAD_LINE_PASSING_DOUBLE_YELLOW → RoadLineType.PASSING_DOUBLE_YELLOW
    # road edges
    14: 20,  # ROAD_EDGE_UNKNOWN  → RoadEdgeType.UNKNOWN
    15: 21,  # ROAD_EDGE_BOUNDARY → RoadEdgeType.BOUNDARY
    16: 22,  # ROAD_EDGE_MEDIAN   → RoadEdgeType.MEDIAN
    # misc
    17: 30,  # STOP_SIGN  → MiscRoadType.UNKNOWN (stop signs handled as traffic controls)
    18: 31,  # CROSSWALK  → MiscRoadType.CROSSWALK
    19: 32,  # SPEED_BUMP → MiscRoadType.SPEED_BUMP
    20: 30,  # DRIVEWAY   → MiscRoadType.UNKNOWN
}

ROAD_TYPE_STR_TO_ID = {
    "lane": None,  # use map_element_id
    "road_line": None,  # use map_element_id
    "road_edge": None,  # use map_element_id
    "crosswalk": 18,
    "stop_sign": 17,
    "speed_bump": 19,
    "driveway": 20,
}

AGENT_TYPE_STR = {"vehicle": 1, "pedestrian": 2, "cyclist": 3}

WOMD_DT = 0.1  # 10 Hz
METADATA_ID_BYTES = 128
METADATA_DATASET_BYTES = 32


def _int_list(values):
    """Encode a length-prefixed int32 list."""
    values = list(values)
    buf = struct.pack("<i", len(values))
    if values:
        buf += struct.pack(f"<{len(values)}i", *values)
    return buf


def _polyline_heading(pts):
    """Compute per-point headings from a polyline (forward tangent, last point repeats)."""
    if len(pts) < 2:
        return np.zeros(len(pts), dtype=np.float32)
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    seg = np.arctan2(dy, dx).astype(np.float32)
    # Explicit float32 to prevent np.append upcasting to float64
    return np.append(seg, seg[-1]).astype(np.float32)


def _resolve_road_type(road):
    """Return the puffer int road type for a JSON road entry."""
    type_str = road.get("type", "")
    map_element_id = road.get("map_element_id", -1)

    if type_str in ("lane", "road_line", "road_edge"):
        return WOMD_MAP_ELEMENT_ID_TO_PUFFER.get(map_element_id, 30)
    else:
        override_id = ROAD_TYPE_STR_TO_ID.get(type_str)
        if override_id is not None:
            return WOMD_MAP_ELEMENT_ID_TO_PUFFER.get(override_id, 30)
        return WOMD_MAP_ELEMENT_ID_TO_PUFFER.get(map_element_id, 30)


def _compute_world_mean(objects, roads):
    """Compute mean x,y from all valid agent positions and road geometry for centering."""
    xs, ys = [], []
    for obj in objects:
        for p in obj.get("position", []):
            xs.append(p.get("x", 0.0))
            ys.append(p.get("y", 0.0))
    for road in roads:
        for g in road.get("geometry", []):
            xs.append(g.get("x", 0.0))
            ys.append(g.get("y", 0.0))
    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def json_to_bin(data: dict) -> bytes:
    buf = bytearray()

    objects = data.get("objects", [])
    roads = data.get("roads", [])
    metadata = data.get("metadata", {})
    scenario_id = data.get("scenario_id", data.get("name", "unknown"))
    dataset = "wod-motion"

    # All objects in this format are agents (vehicle/pedestrian/cyclist)
    agents = [o for o in objects if o.get("type") in AGENT_TYPE_STR]
    # No separate static objects in old WOMD JSON format
    static_objects = []

    # Center coordinates around world mean so the render camera (at 0,0) sees agents
    mean_x, mean_y = _compute_world_mean(agents, roads)

    # ── Header: num_agents, num_road_elements (placeholder), num_tcs, num_objects ──
    # We patch num_road_elements after filtering out empty roads
    buf.extend(struct.pack("<iiii", len(agents), 0, 0, len(static_objects)))

    # ── Agents ──────────────────────────────────────────────────────────────────
    T = 0
    for agent_idx, obj in enumerate(agents):
        agent_type = AGENT_TYPE_STR.get(obj.get("type", "vehicle"), 1)
        # Write sequential agent_idx (not the JSON id) — C reader checks id == idx
        buf.extend(struct.pack("<ii", agent_idx, agent_type))

        positions = obj.get("position", [])
        velocities = obj.get("velocity", [])
        headings_raw = obj.get("heading", [])
        valid_raw = obj.get("valid", [])
        T = len(positions)

        buf.extend(struct.pack("<i", T))

        # x, y, z, heading, vx, vy, length, width, height — each float32[T]
        xs = np.array([p.get("x", 0.0) - mean_x for p in positions], dtype=np.float32)
        ys = np.array([p.get("y", 0.0) - mean_y for p in positions], dtype=np.float32)
        zs = np.array([p.get("z", 0.0) for p in positions], dtype=np.float32)
        hs = np.array([float(h) for h in headings_raw], dtype=np.float32)
        vxs = np.array([v.get("x", 0.0) for v in velocities], dtype=np.float32)
        vys = np.array([v.get("y", 0.0) for v in velocities], dtype=np.float32)

        length = float(obj.get("length", 0.0))
        width = float(obj.get("width", 0.0))
        height = float(obj.get("height", 0.0))
        lengths = np.full(T, length, dtype=np.float32)
        widths = np.full(T, width, dtype=np.float32)
        heights = np.full(T, height, dtype=np.float32)

        valid = np.array([int(v) for v in valid_raw], dtype=np.int32)

        for col in [xs, ys, zs, hs, vxs, vys, lengths, widths, heights]:
            buf.extend(col.tobytes())
        buf.extend(valid.tobytes())

        # Route: not available in old JSON format.
        # CONTROL_SDC_ONLY requires agent 0 to have route_length != 0 to be
        # selected as the policy-controlled agent, so write a dummy route of [-1].
        route = [-1] if agent_idx == 0 else []
        buf.extend(_int_list(route))
        buf.extend(struct.pack("<i", 0))  # route_gt_len

        # Goal position: from goalPosition field or last valid frame (apply centering)
        goal = obj.get("goalPosition", {})
        if goal:
            gx = float(goal.get("x", 0.0)) - mean_x
            gy = float(goal.get("y", 0.0)) - mean_y
            gz = float(goal.get("z", 0.0))
        else:
            valid_idx = np.where(valid > 0)[0]
            if len(valid_idx) > 0:
                i = valid_idx[-1]
                gx, gy, gz = float(xs[i]), float(ys[i]), float(zs[i])
            else:
                gx, gy, gz = 0.0, 0.0, 0.0
        buf.extend(struct.pack("<fff", gx, gy, gz))

        mark_as_expert = 1 if obj.get("mark_as_expert", False) else 0
        buf.extend(struct.pack("<i", mark_as_expert))

    # ── Road map ────────────────────────────────────────────────────────────────
    road_count = 0
    for road in roads:
        geometry = road.get("geometry", [])
        if len(geometry) < 2:
            continue

        road_type = _resolve_road_type(road)
        is_lane = 0 <= road_type <= 9

        pts = np.array(
            [[g.get("x", 0.0) - mean_x, g.get("y", 0.0) - mean_y, g.get("z", 0.0)] for g in geometry], dtype=np.float32
        )
        heading = _polyline_heading(pts)

        # Write sequential road_count (not JSON road id) — C reader checks id == idx
        buf.extend(struct.pack("<ii", road_count, road_type))
        buf.extend(struct.pack("<i", len(pts)))
        for col in [pts[:, 0], pts[:, 1], pts[:, 2]]:
            buf.extend(col.tobytes())
        buf.extend(heading.tobytes())

        if is_lane:
            # No entry/exit lane info in old JSON format
            buf.extend(_int_list([]))  # entry_lanes
            buf.extend(_int_list([]))  # exit_lanes
            buf.extend(struct.pack("<f", -1.0))  # speed_limit unknown

        road_count += 1

    # Patch road count in header (offset 4)
    struct.pack_into("<i", buf, 4, road_count)

    # ── Traffic controls: empty (tl_states not in a form we can reliably convert) ──
    # (offset 8 is already 0 from header)

    # ── Objects (static): none in old format ────────────────────────────────────

    # ── Lane graph: skip ─────────────────────────────────────────────────────────
    buf.extend(struct.pack("<i", 0))

    # ── Metadata ────────────────────────────────────────────────────────────────
    buf.extend(str(scenario_id).encode("utf-8")[:METADATA_ID_BYTES].ljust(METADATA_ID_BYTES, b"\0"))
    buf.extend(dataset.encode("utf-8")[:METADATA_DATASET_BYTES].ljust(METADATA_DATASET_BYTES, b"\0"))
    buf.extend(struct.pack("<i", T if agents else 0))  # scenario_length
    buf.extend(struct.pack("<f", WOMD_DT))  # dt
    buf.extend(struct.pack("<i", 0))  # objects_of_interest (empty)
    buf.extend(struct.pack("<i", 0))  # tracks_to_predict (empty)

    return bytes(buf)


def convert_directory(input_dir: Path, output_dir: Path):
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"  No JSON files found in {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for json_path in json_files:
        out_path = output_dir / (json_path.stem + ".bin")
        try:
            with open(json_path) as f:
                data = json.load(f)
            binary = json_to_bin(data)
            with open(out_path, "wb") as f:
                f.write(binary)
            ok += 1
        except Exception as e:
            print(f"  ERROR {json_path.name}: {e}")

    print(f"  {ok}/{len(json_files)} converted → {output_dir}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Convert WOMD JSON scenarios to PufferDrive .bin format")
    parser.add_argument("--input", required=True, help="Root dir with per-class subdirectories of JSON files")
    parser.add_argument("--output", required=True, help="Root output dir; per-class subdirs will be created")
    parser.add_argument("--classes", nargs="+", default=None, help="Specific subdirs to process (default: all)")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    subdirs = sorted(d for d in input_root.iterdir() if d.is_dir())
    if args.classes:
        subdirs = [d for d in subdirs if d.name in args.classes]

    if not subdirs:
        print(f"No subdirectories found in {input_root}")
        return

    total = 0
    for subdir in subdirs:
        print(f"[{subdir.name}]")
        out = output_root / subdir.name
        total += convert_directory(subdir, out)

    print(f"\nDone. {total} total files converted.")


if __name__ == "__main__":
    main()
