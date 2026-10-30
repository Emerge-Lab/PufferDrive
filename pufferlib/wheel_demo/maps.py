"""CARLA map selection for the wheel demo.

The demo runs on CARLA towns only, and only on builds that carry the per-lane speed zones and
traffic-light phase groups the frozen policy trained against. An older build loads perfectly well
and then quietly degrades the policy: every lane falls back to one constant speed limit and the
road-edge geometry is far sparser, so offroad detection and the boundary observation no longer
match what the policy expects. Rejecting it loudly beats debugging odd traffic later.
"""

import os

CARLA_MAP_PREFIX = "opendrive__"
MAP_SUFFIX = ".bin"
SPEED_ZONE_SECTION_TAG = b"SPDZONE1"
TRAFFIC_PHASE_SECTION_TAG = b"TLPHASE1"

CARLA_TOWNS = ("Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD")

DEFAULT_MAP_DIR = os.path.join("pufferlib", "resources", "drive", "binaries", "carla")


def map_sections(map_path):
    with open(map_path, "rb") as handle:
        blob = handle.read()
    return {
        "speed_zones": SPEED_ZONE_SECTION_TAG in blob,
        "traffic_phases": TRAFFIC_PHASE_SECTION_TAG in blob,
    }


def resolve_carla_map_files(map_dir=DEFAULT_MAP_DIR, towns=CARLA_TOWNS, require_sections=True):
    """Absolute paths to the requested CARLA towns, in the given order.

    Raises rather than falling back: an unreadable or wrong-format map set is unrecoverable, and
    silently running on a subset would make the demo look broken for reasons nobody could see.
    """
    if not os.path.isdir(map_dir):
        raise FileNotFoundError(f"map_dir does not exist: {map_dir}")

    paths, missing, degraded = [], [], []
    for town in towns:
        path = os.path.join(map_dir, f"{CARLA_MAP_PREFIX}{town}{MAP_SUFFIX}")
        if not os.path.isfile(path):
            missing.append(town)
            continue
        if require_sections:
            sections = map_sections(path)
            if not (sections["speed_zones"] and sections["traffic_phases"]):
                degraded.append((town, sections))
        paths.append(os.path.abspath(path))

    if missing:
        raise FileNotFoundError(
            f"{map_dir} is missing {len(missing)} CARLA town(s): {missing}. "
            f"Expected files named {CARLA_MAP_PREFIX}<Town>{MAP_SUFFIX}."
        )
    if degraded:
        detail = ", ".join(
            f"{town} (speed_zones={s['speed_zones']}, traffic_phases={s['traffic_phases']})" for town, s in degraded
        )
        raise ValueError(
            f"{map_dir} holds an older CARLA build without the sections the policy trained on: {detail}. "
            f"Use a build carrying both {SPEED_ZONE_SECTION_TAG.decode()} and "
            f"{TRAFFIC_PHASE_SECTION_TAG.decode()}, or pass require_sections=False to accept the degradation."
        )
    return paths
