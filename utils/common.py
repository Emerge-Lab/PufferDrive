"""Shared dataset helpers.

JSON scenarios live under data/<name>/. Binaries that this simulator
loads are written by pufferlib.ocean.drive.drive.save_map_binary:

  16-byte scenario_id, sdc_track_index, tracks_to_predict, object
  count, road count, then each entity. Trajectory length is 91.
  Object types are vehicle 1, pedestrian 2, cyclist 3. Type 4 is a
  road lane in drive.h, so a partner labeled truck is stored as a
  vehicle. A log longer than 91 states is truncated on write. A
  shorter one is zero-padded. The source dict is not modified.

JSON from the I24 converter may still use the labels below, including
truck, and a length up to DRIVE_LOG_MAX. That JSON is the converter's
record. The binary is the 91-state simulator record.

  data/<name>/json/<id>.json
  data/<name>/binaries/map_%03d.bin
  data/<name>/binaries/sdc.bin

Sorted json order is the map index. sdc.bin is int count, then one
int32 self-driving-car index per map.
"""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path
from typing import Any

# One step is 0.1 s. DRIVE_LOG_MAX is the longest log load_map_binary
# accepts, matching constants.h. A Waymo Open Motion segment is 91 states.
SAMPLE_HZ = 10
WAYMO_TRAJECTORY_STATES = 91
MIN_TRAJECTORY_STATES = 2
DRIVE_LOG_MAX = 90 * SAMPLE_HZ + 1
# Export cap. DRIVE_AGENT_MAX is 256. load_map_binary accepts 512.
MAX_OBJECTS = 256

# ocean/drive/constants.h entity types. Partners are 1..4. Roads follow,
# so a type below ROAD_LANE is a body and a truck is not a lane.
VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 3
TRUCK = 4
# Exported length. A vehicle-like body at least this long is a truck,
# including a tractor-trailer. Pedestrians and cyclists stay themselves.
TRUCK_MIN_LENGTH_M = 7.5
ROAD_LANE = 5
ROAD_LINE = 6
ROAD_EDGE = 7
STOP_SIGN_CLASS = 8
CROSSWALK_CLASS = 9
SPEED_BUMP_CLASS = 10
DRIVEWAY_CLASS = 11

# Source roadgraph ids, stored as roads[].map_element_id before
# drive_road_type. The numbers follow the Waymo Open Motion catalog:
# https://waymo.com/open/data/motion/tfexample/
LANE_UNKNOWN = 0
LANE_FREEWAY = 1
LANE_SURFACE_STREET = 2
LANE_BIKE = 3
ROAD_LINE_UNKNOWN = 5
ROAD_LINE_BROKEN_SINGLE_WHITE = 6
ROAD_LINE_SOLID_SINGLE_WHITE = 7
ROAD_LINE_SOLID_DOUBLE_WHITE = 8
ROAD_LINE_BROKEN_SINGLE_YELLOW = 9
ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10
ROAD_LINE_SOLID_SINGLE_YELLOW = 11
ROAD_LINE_SOLID_DOUBLE_YELLOW = 12
ROAD_LINE_PASSING_DOUBLE_YELLOW = 13
ROAD_EDGE_UNKNOWN = 14
ROAD_EDGE_BOUNDARY = 15
ROAD_EDGE_MEDIAN = 16
STOP_SIGN = 17
CROSSWALK = 18
SPEED_BUMP = 19
DRIVEWAY = 20

OBJECT_TYPES = {
    VEHICLE: "vehicle",
    PEDESTRIAN: "pedestrian",
    CYCLIST: "cyclist",
    TRUCK: "truck",
}
# Source labels that become vehicle or truck, chosen by length.
VEHICLE_LABELS = {
    "",
    "vehicle",
    "sedan",
    "midsize",
    "pickup",
    "van",
    "cs",
    "truck",
    "semi",
    "trailer",
}
# roads[].type strings in the scenario JSON.
ROAD_TYPES = (
    "lane",
    "road_line",
    "road_edge",
    "stop_sign",
    "crosswalk",
    "speed_bump",
    "driveway",
)


def drive_road_type(map_element_id: int) -> int:
    """Collapse a source roadgraph id to the PufferDrive road class."""
    if LANE_UNKNOWN <= map_element_id <= LANE_BIKE:
        return ROAD_LANE
    if ROAD_LINE_UNKNOWN <= map_element_id <= ROAD_LINE_PASSING_DOUBLE_YELLOW:
        return ROAD_LINE
    if ROAD_EDGE_UNKNOWN <= map_element_id <= ROAD_EDGE_MEDIAN:
        return ROAD_EDGE
    if map_element_id == STOP_SIGN:
        return STOP_SIGN_CLASS
    if map_element_id == CROSSWALK:
        return CROSSWALK_CLASS
    if map_element_id == SPEED_BUMP:
        return SPEED_BUMP_CLASS
    if map_element_id == DRIVEWAY:
        return DRIVEWAY_CLASS
    return map_element_id


def classify_partner(source_type: str, length_m: float) -> str:
    """Partner class shared by Waymo and I24.

    Pedestrians and cyclists stay those classes. Any other vehicle-like
    body of TRUCK_MIN_LENGTH_M or longer is a truck. A shorter one is a
    vehicle. Object 0 may be either a vehicle or a truck.
    """
    name = str(source_type or "").strip().lower()
    if name in (OBJECT_TYPES[PEDESTRIAN], OBJECT_TYPES[CYCLIST]):
        return name
    if name not in VEHICLE_LABELS:
        raise ValueError(f"Unknown partner class {source_type!r}")
    if float(length_m) >= TRUCK_MIN_LENGTH_M:
        return OBJECT_TYPES[TRUCK]
    return OBJECT_TYPES[VEHICLE]


def object_type_code(type_name: str) -> int:
    """Object name to the integer stored in a PufferDrive log."""
    for code, name in OBJECT_TYPES.items():
        if name == type_name:
            return code
    raise ValueError(f"Unknown PufferDrive object type {type_name!r}")


def validate_scenario(scenario: dict[str, Any], max_objects: int = MAX_OBJECTS) -> None:
    """Check the fields offline.h and load_map_binary require of a log."""
    required = {"name", "scenario_id", "objects", "roads", "metadata"}
    missing = required - scenario.keys()
    if missing:
        raise ValueError(f"Scenario is missing keys: {sorted(missing)}")
    if not 1 <= len(scenario["objects"]) <= max_objects:
        raise ValueError(f"Expected 1..{max_objects} objects")
    if scenario["metadata"].get("sdc_track_index") != 0:
        raise ValueError("The self-driving car must be object 0")
    expected_states = int(scenario["metadata"]["frame_count"])
    if expected_states < MIN_TRAJECTORY_STATES or expected_states > DRIVE_LOG_MAX:
        raise ValueError(
            f"Scenario length must be {MIN_TRAJECTORY_STATES}..{DRIVE_LOG_MAX} states"
        )
    for object_index, obj in enumerate(scenario["objects"]):
        if obj.get("type") not in OBJECT_TYPES.values():
            raise ValueError(f"Object {object_index} is not a PufferDrive object type")
        if object_index == 0 and obj.get("type") not in (
            OBJECT_TYPES[VEHICLE],
            OBJECT_TYPES[TRUCK],
        ):
            raise ValueError("The self-driving car must be a vehicle or a truck")
        for field in ("position", "velocity", "heading", "valid"):
            if len(obj.get(field, [])) != expected_states:
                raise ValueError(
                    f"Object {object_index} field {field} does not have {expected_states} states"
                )
        for key in ("width", "length", "goalPosition"):
            if key not in obj:
                raise ValueError(f"Object {object_index} is missing {key}")
        if any(
            not math.isfinite(point[axis])
            for point in obj["position"]
            for axis in ("x", "y", "z")
        ):
            raise ValueError(f"Object {object_index} contains non-finite positions")
    if not all(scenario["objects"][0]["valid"]):
        raise ValueError("The self-driving car must be valid on every state")
    if not scenario["roads"]:
        raise ValueError("Each scenario needs non-empty road geometry")
    for road in scenario["roads"]:
        if road.get("type") not in ROAD_TYPES:
            raise ValueError(f"Road type {road.get('type')!r} is not a PufferDrive road category")
        if len(road.get("geometry", [])) < 2:
            raise ValueError("Each scenario needs non-empty road geometry")
        collapsed = drive_road_type(int(road.get("map_element_id", -1)))
        if collapsed < ROAD_LANE or collapsed > DRIVEWAY_CLASS:
            raise ValueError(
                f"Road map_element_id {road.get('map_element_id')} is not a PufferDrive road category"
            )


_MAP_INDEX = re.compile(r"map_(\d+)\.bin$")


def _map_index(path: Path, unique_map_id: int | None) -> int:
    if unique_map_id is not None:
        return int(unique_map_id)
    match = _MAP_INDEX.search(path.name)
    if match:
        return int(match.group(1))
    return 0


def scenario_for_drive(scenario: dict[str, Any]) -> dict[str, Any]:
    """Copy a scenario into the types save_map_binary packs.

    truck and type 4 become vehicle. CARLA metadata that only has
    sdc_index is copied onto sdc_track_index. The caller's dict is
    left unchanged, including any on-disk JSON.
    """
    packed = dict(scenario)
    scenario_id = packed.get("scenario_id", packed.get("name", ""))
    packed["scenario_id"] = str(scenario_id)[:16]
    metadata = dict(scenario.get("metadata") or {})
    if "sdc_track_index" not in metadata and "sdc_index" in metadata:
        metadata["sdc_track_index"] = metadata["sdc_index"]
    packed["metadata"] = metadata
    objects = []
    for obj in scenario.get("objects") or []:
        copied = dict(obj)
        kind = copied.get("type")
        if kind == OBJECT_TYPES[TRUCK] or kind == TRUCK:
            copied["type"] = OBJECT_TYPES[VEHICLE]
        objects.append(copied)
    packed["objects"] = objects
    packed["roads"] = list(scenario.get("roads") or [])
    return packed


def write_log_binary(
    scenario: dict[str, Any],
    path: Path | str,
    unique_map_id: int | None = None,
) -> None:
    """Write the drive.py / drive.h record. Longer logs are truncated to 91 states."""
    path = Path(path)
    if not scenario.get("roads"):
        raise ValueError("A map binary needs roads")
    path.parent.mkdir(parents=True, exist_ok=True)
    from pufferlib.ocean.drive.drive import save_map_binary

    save_map_binary(scenario_for_drive(scenario), str(path), _map_index(path, unique_map_id))


def write_road_map(
    roads: list[dict[str, Any]],
    path: Path | str,
    unique_map_id: int | None = None,
) -> None:
    """Road-only file in the same record, with zero objects."""
    path = Path(path)
    write_log_binary(
        {
            "name": path.stem,
            "scenario_id": path.stem[:16],
            "objects": [],
            "roads": roads,
            "metadata": {},
        },
        path,
        unique_map_id=unique_map_id,
    )


def write_sdc_index(path: Path | str, indices: list[int]) -> None:
    """int count, then one int32 object index per map. Matches read_map_sdc."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        file.write(struct.pack("i", len(indices)))
        file.write(struct.pack(f"{len(indices)}i", *indices))
