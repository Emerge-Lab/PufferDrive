"""Convert I24-3D ground-truth splines into PufferDrive logs.

A log is one self-driving car's track at 10 Hz, both endpoints
included, capped at the longest I24 scene (90 s, 901 states).
The JSON record is the one described in utils/common.py. The binary
is save_map_binary: 91 states, and a truck is stored as a vehicle.
JSON keeps the scene length, up to 901 states. A longer log is
truncated when the binary is written.

  data/<name>/json/<id>.json
  data/<name>/binaries/map_%03d.bin
  data/<name>/binaries/sdc.bin

Sorted json order is the map index. sdc.bin is the object index of
the self-driving car, one int32 per map. Appendix A.4 of arXiv
2606.19370 then infers actions and replays them in the simulator.
This script builds the logs. The raw scenes shipped with this repo
are under data/i24/raw/I24-3D and are not converted by default.

Source positions and sizes are feet. Exported distances are meters.
Building the JSON uses the Python standard library. Writing a binary
imports the simulator packer.

  python utils/i24.py --i24-root /path/to/I24-3D --output-dir data/i24
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.common import (  # noqa: E402
    DRIVE_LOG_MAX,
    LANE_FREEWAY,
    MAX_OBJECTS,
    ROAD_EDGE_BOUNDARY,
    ROAD_EDGE_MEDIAN,
    ROAD_LINE_BROKEN_SINGLE_WHITE,
    SAMPLE_HZ,
    classify_partner,
    drive_road_type,
    validate_scenario,
    write_log_binary,
    write_sdc_index,
)


FEET_TO_METERS = 0.3048
# Annotated boxes are treated as slightly larger than the body that
# drove the lane. Exported length, width, and height use this factor.
BOX_SCALE = 0.90
# The calibration edge is the outer lane line. No I24 vehicle is
# treated as truly overlapping that line, so each edge is moved 1.5 ft
# away from the lanes.
ROAD_EDGE_OUTWARD_M = 1.5 * FEET_TO_METERS
# The simulator stores one observation per segment midpoint, in 5 m
# grid cells, and only cells within about 50 m of the agent. A single
# chord across the whole highway has one midpoint, often far from the
# car, so the observation is empty. Sample every road at the cell size.
ROAD_SAMPLE_METERS = 5.0
# Longest I24 scene clock. Inclusive 10 Hz states: t = 0 through t = 90.
MAX_SCENARIO_SECONDS = 90
MAX_TRAJECTORY_STATES = MAX_SCENARIO_SECONDS * SAMPLE_HZ + 1
# Scene 1's raw file runs about 114 s. Samples after this clock are unused.
SCENE1_CLOCK_SECONDS = 90.0
assert MAX_TRAJECTORY_STATES == DRIVE_LOG_MAX
HEADING_WINDOW_HALF_SECONDS = 0.5
HEADING_MIN_RELIABLE_SPEED_MPS = 0.5
# A track that drives this far against its carriageway is removed.
# The carriageway direction is the majority of tracks there, and the
# lower carriageway is eastbound (+x) when that vote is tied.
WRONG_WAY_MIN_METERS = 10.0
# Intersection area above half the smaller box, on more than this
# many timesteps, drops the smaller track. Pairs are collected first.
DEEP_OVERLAP_FRACTION = 0.5
DEEP_OVERLAP_MIN_STEPS = 5
# A log that never leaves a 1 m box is not a moving vehicle. The next
# smallest track in the raw scenes travels 6 m, so this does not drop
# a car that only appears for a short moving glimpse.
STATIONARY_MAX_METERS = 1.0
# A 10 Hz step this far from the surrounding motion is a bad sample.
# 0.25 m in one frame is 2.5 m/s, above the accel these logs actually
# show. 1.25 m is a single-frame jump large enough to copy the
# previous point before the chord replacement.
SPIKE_STEP_M = 0.25
HUGE_SPIKE_STEP_M = 1.25
SPIKE_MEDIAN_RADIUS = 25
SPIKE_MERGE_GAP = 5
SPIKE_MAX_BURST = 30
# The far end of a burst has to land near the median continuation.
# 4 m keeps a 2 s stop (about 6 m of missed travel) and still covers a
# there-and-back of a few meters. A longer excursion is not invented.
SPIKE_CLOSE_M = 4.0
REVERSAL_STEP_M = 0.05
REVERSAL_MIN_M = 0.5
STRANGE_STEP_M = 2.0
MIN_REMAINING_STATES = 2 * SAMPLE_HZ + 1
REPAIR_PASSES = 2
MAJOR_QUALITY_FLAGS = {
    "speed_above_60_mps",
    "more_than_200m_outside_calibrated_x_range",
    "outside_calibrated_y_range",
}

# I24-3D body styles. Length then picks vehicle or truck, same as Waymo.
# Scene 3 labels one car-sized track (14.8 x 6.0 ft) as "cs".
I24_VEHICLE_CLASSES = {
    "sedan",
    "midsize",
    "pickup",
    "van",
    "semi",
    "truck",
    "cs",
}


def drive_object_type(source_class: str, length_m: float) -> str:
    """Map an I24 label and exported length onto a partner class."""
    name = source_class.strip().lower()
    if name in I24_VEHICLE_CLASSES or name in ("", "vehicle", "pedestrian", "cyclist"):
        return classify_partner(name, length_m)
    raise ValueError(f"I24 class {source_class!r} is not a PufferDrive object type")


def _point(x: float, y: float, z: float = 0.0) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "z": float(z)}


def _cluster(values: list[float], tolerance: float = 0.1) -> list[float]:
    """Merge nearly identical calibration coordinates from overlapping cameras."""
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or abs(value - sum(groups[-1]) / len(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [sum(group) / len(group) for group in groups]


def _edge_y_m(boundary_y_ft: float, lane_mid_ft: float) -> float:
    """Lane-line y, shifted 1.5 ft away from the carriageway center."""
    outward = -1.0 if boundary_y_ft < lane_mid_ft else 1.0
    return boundary_y_ft * FEET_TO_METERS + outward * ROAD_EDGE_OUTWARD_M


def _carriageway_split_y(boundary_y_m: list[float]) -> float:
    """Y of the median, the largest gap between calibration lines."""
    gaps = [right - left for left, right in zip(boundary_y_m, boundary_y_m[1:])]
    if not gaps:
        raise ValueError("The road grid has no carriageway split")
    median_index = max(range(len(gaps)), key=gaps.__getitem__)
    return 0.5 * (boundary_y_m[median_index] + boundary_y_m[median_index + 1])


def load_road_grid(homography_path: Path) -> dict[str, Any]:
    """Derive lane geometry from the 12-foot I24-3D calibration grid."""
    with homography_path.open() as file:
        homographies = json.load(file)

    points: list[tuple[float, float]] = []
    for direction in ("EB", "WB"):
        for camera in homographies[direction].values():
            points.extend((float(x), float(y)) for x, y in camera["space_pts"])

    x_values = [point[0] for point in points]
    y_values = _cluster([point[1] for point in points])
    if len(y_values) < 4:
        raise ValueError(f"Could not recover lane boundaries from {homography_path}")

    gaps = [right - left for left, right in zip(y_values, y_values[1:])]
    median_index = max(range(len(gaps)), key=gaps.__getitem__)
    carriageways = [y_values[: median_index + 1], y_values[median_index + 1 :]]
    if any(len(boundaries) < 2 for boundaries in carriageways):
        raise ValueError(f"Could not split the two carriageways in {homography_path}")

    x_min_ft, x_max_ft = min(x_values), max(x_values)
    roads: list[dict[str, Any]] = []
    road_id = 0
    for carriageway_index, boundaries_ft in enumerate(carriageways):
        # The lower-y carriageway is eastbound (+x). Westbound centerlines
        # run the other way so they follow traffic.
        direction = 1 if carriageway_index == 0 else -1
        start_x, end_x = (x_min_ft, x_max_ft) if direction == 1 else (x_max_ft, x_min_ft)

        for lower, upper in zip(boundaries_ft, boundaries_ft[1:]):
            center_y = 0.5 * (lower + upper)
            roads.append(
                {
                    "id": road_id,
                    "type": "lane",
                    "map_element_id": LANE_FREEWAY,
                    "geometry": [
                        _point(start_x * FEET_TO_METERS, center_y * FEET_TO_METERS),
                        _point(end_x * FEET_TO_METERS, center_y * FEET_TO_METERS),
                    ],
                }
            )
            road_id += 1

        for divider_y in boundaries_ft[1:-1]:
            roads.append(
                {
                    "id": road_id,
                    "type": "road_line",
                    "map_element_id": ROAD_LINE_BROKEN_SINGLE_WHITE,
                    "geometry": [
                        _point(x_min_ft * FEET_TO_METERS, divider_y * FEET_TO_METERS),
                        _point(x_max_ft * FEET_TO_METERS, divider_y * FEET_TO_METERS),
                    ],
                }
            )
            road_id += 1

        lane_mid_ft = 0.5 * (boundaries_ft[0] + boundaries_ft[-1])
        for boundary_index in (0, len(boundaries_ft) - 1):
            boundary_y = boundaries_ft[boundary_index]
            is_median = (
                carriageway_index == 0 and boundary_index == len(boundaries_ft) - 1
            ) or (carriageway_index == 1 and boundary_index == 0)
            roads.append(
                {
                    "id": road_id,
                    "type": "road_edge",
                    "map_element_id": ROAD_EDGE_MEDIAN if is_median else ROAD_EDGE_BOUNDARY,
                    "geometry": [
                        _point(x_min_ft * FEET_TO_METERS, _edge_y_m(boundary_y, lane_mid_ft)),
                        _point(x_max_ft * FEET_TO_METERS, _edge_y_m(boundary_y, lane_mid_ft)),
                    ],
                }
            )
            road_id += 1

    boundary_y_m = [value * FEET_TO_METERS for value in y_values]
    return {
        "source": homography_path.name,
        "boundary_y_m": boundary_y_m,
        "carriageway_split_y_m": _carriageway_split_y(boundary_y_m),
        "x_range_m": [x_min_ft * FEET_TO_METERS, x_max_ft * FEET_TO_METERS],
        "roads": roads,
    }


def _interpolate(
    timestamps: list[float], values: list[float], target: float, index: int
) -> tuple[float, int]:
    while index + 1 < len(timestamps) and timestamps[index + 1] < target:
        index += 1
    if index + 1 >= len(timestamps):
        return values[-1], len(timestamps) - 2
    left_t, right_t = timestamps[index], timestamps[index + 1]
    if right_t <= left_t:
        return values[index], index
    fraction = max(0.0, min(1.0, (target - left_t) / (right_t - left_t)))
    return values[index] + fraction * (values[index + 1] - values[index]), index


def _stabilized_headings(
    positions: list[list[float]], fallback_heading: float
) -> list[float]:
    """Infer heading from a local regression, holding it while motion is ambiguous."""
    radius = round(HEADING_WINDOW_HALF_SECONDS * SAMPLE_HZ)
    candidates: list[Optional[float]] = []
    for center in range(len(positions)):
        begin = max(0, center - radius)
        end = min(len(positions), center + radius + 1)
        mean_index = sum(range(begin, end)) / (end - begin)
        denominator = sum((index - mean_index) ** 2 for index in range(begin, end))
        if denominator == 0:
            candidates.append(None)
            continue
        vx = (
            sum((index - mean_index) * positions[index][0] for index in range(begin, end))
            / denominator
            * SAMPLE_HZ
        )
        vy = (
            sum((index - mean_index) * positions[index][1] for index in range(begin, end))
            / denominator
            * SAMPLE_HZ
        )
        if math.hypot(vx, vy) >= HEADING_MIN_RELIABLE_SPEED_MPS:
            candidates.append(math.atan2(vy, vx))
        else:
            candidates.append(None)

    first_reliable = next((value for value in candidates if value is not None), None)
    if first_reliable is None:
        return [fallback_heading] * len(positions)

    current = first_reliable
    headings = []
    for candidate in candidates:
        if candidate is not None:
            current = candidate
        headings.append(current)
    return headings


def _heading_estimation_metadata(stabilize_headings: bool) -> dict[str, Any]:
    if stabilize_headings:
        return {
            "stabilized": True,
            "method": "centered_linear_regression_on_position",
            "window_half_width_seconds": HEADING_WINDOW_HALF_SECONDS,
            "minimum_reliable_speed_mps": HEADING_MIN_RELIABLE_SPEED_MPS,
            "low_speed_policy": "hold_last_reliable_heading",
            "initial_low_speed_policy": "backfill_first_reliable_heading",
            "positions_modified": False,
            "velocities_modified": False,
        }
    return {
        "stabilized": False,
        "method": "centered_finite_difference_velocity",
        "minimum_speed_mps": 0.05,
        "low_speed_policy": "source_direction_fallback",
        "positions_modified": False,
        "velocities_modified": False,
    }


def _empty_position_repair() -> dict[str, Any]:
    return {
        "smoothed_frames": 0,
        "copied_huge_frames": 0,
        "dropped_prefix_frames": 0,
        "dropped_suffix_frames": 0,
        "interpolated_bursts": 0,
        "reversal_frames": 0,
        "reversal": "none",
        "dropped": None,
    }


def _marked_runs(marks: list[bool], merge_gap: int) -> list[tuple[int, int]]:
    """Inclusive runs of marked steps. A gap of `merge_gap` steps is one run."""
    runs: list[tuple[int, int]] = []
    index = 0
    count = len(marks)
    while index < count:
        if not marks[index]:
            index += 1
            continue
        end = index
        while True:
            cursor = end
            while cursor < count and marks[cursor]:
                cursor += 1
            gap = cursor
            while gap < count and not marks[gap] and gap - cursor <= merge_gap:
                gap += 1
            if gap < count and marks[gap] and gap > cursor:
                end = gap
                continue
            end = cursor - 1
            break
        runs.append((index, end))
        index = end + 1
    return runs


def _step_vectors(positions: list[list[float]]) -> list[tuple[float, float]]:
    return [
        (positions[index + 1][0] - positions[index][0], positions[index + 1][1] - positions[index][1])
        for index in range(len(positions) - 1)
    ]


def _local_step_medians(
    steps: list[tuple[float, float]],
) -> list[Optional[tuple[float, float]]]:
    """Median of the other steps in a ±2.5 s window. None when fewer than 5 exist."""
    medians: list[Optional[tuple[float, float]]] = []
    for index in range(len(steps)):
        begin = max(0, index - SPIKE_MEDIAN_RADIUS)
        end = min(len(steps), index + SPIKE_MEDIAN_RADIUS + 1)
        sample = [steps[other] for other in range(begin, end) if other != index]
        if len(sample) < 5:
            medians.append(None)
            continue
        medians.append((
            statistics.median(step[0] for step in sample),
            statistics.median(step[1] for step in sample),
        ))
    return medians


def _chord_pass(positions: list[list[float]]) -> tuple[list[list[float]], int, int, int]:
    """Put an interior sample back on the chord of its original neighbors.

    A huge miss is copied from the previous sample first. The chord
    replacement overwrites that copy, so the following step is not a
    zero-length catch-up. A huge last sample has no following anchor:
    it is replaced by the previous sample and the duplicate is dropped.
    """
    count = len(positions)
    copied = 0
    smoothed = 0
    dropped_suffix = 0
    usable = positions
    if count >= 3:
        previous = (
            positions[-2][0] - positions[-3][0],
            positions[-2][1] - positions[-3][1],
        )
        last = (
            positions[-1][0] - positions[-2][0],
            positions[-1][1] - positions[-2][1],
        )
        if math.hypot(last[0] - previous[0], last[1] - previous[1]) >= HUGE_SPIKE_STEP_M:
            usable = positions[:-1]
            copied += 1
            dropped_suffix = 1
    repaired = [point[:] for point in usable]
    for index in range(1, len(usable) - 1):
        mid_x = 0.5 * (usable[index - 1][0] + usable[index + 1][0])
        mid_y = 0.5 * (usable[index - 1][1] + usable[index + 1][1])
        deviation = math.hypot(usable[index][0] - mid_x, usable[index][1] - mid_y)
        if deviation >= HUGE_SPIKE_STEP_M:
            repaired[index] = usable[index - 1][:]
            copied += 1
            repaired[index] = [mid_x, mid_y]
            smoothed += 1
        elif deviation >= SPIKE_STEP_M:
            repaired[index] = [mid_x, mid_y]
            smoothed += 1
    return repaired, copied, smoothed, dropped_suffix


def _burst_pass(
    positions: list[list[float]],
) -> tuple[list[list[float]], int, int, int]:
    """Interpolate a short run that returns to the surrounding motion.

    The right anchor is two samples past the run, so a one-step run
    still has an interior sample to move. Prefix runs are dropped.
    Suffix runs are cut. A run that does not close, or is longer than
    3 s, is left for the reversal pass.
    """
    count = len(positions)
    steps = _step_vectors(positions)
    medians = _local_step_medians(steps)
    marks = [
        median is not None
        and math.hypot(step[0] - median[0], step[1] - median[1]) >= SPIKE_STEP_M
        for step, median in zip(steps, medians)
    ]
    repaired = [point[:] for point in positions]
    drop_prefix = 0
    drop_suffix: Optional[int] = None
    interpolated = 0
    for start, end in _marked_runs(marks, SPIKE_MERGE_GAP):
        if end - start + 1 > SPIKE_MAX_BURST:
            continue
        if start == 0:
            drop_prefix = max(drop_prefix, end + 1)
            continue
        if end >= count - 2:
            drop_suffix = start if drop_suffix is None else min(drop_suffix, start)
            continue
        right = end + 2
        if right >= count or medians[start] is None:
            continue
        median = medians[start]
        intervals = right - start
        predicted_x = positions[start][0] + median[0] * intervals
        predicted_y = positions[start][1] + median[1] * intervals
        closure = math.hypot(
            predicted_x - positions[right][0], predicted_y - positions[right][1]
        )
        if closure > SPIKE_CLOSE_M:
            continue
        origin_x, origin_y = positions[start]
        dest_x, dest_y = positions[right]
        for step in range(1, intervals):
            weight = step / intervals
            repaired[start + step] = [
                origin_x + (dest_x - origin_x) * weight,
                origin_y + (dest_y - origin_y) * weight,
            ]
        interpolated += 1
    if drop_prefix:
        repaired = repaired[drop_prefix:]
    dropped_suffix = 0
    if drop_suffix is not None:
        cut = max(0, drop_suffix - drop_prefix)
        dropped_suffix = max(0, len(repaired) - (cut + 1))
        repaired = repaired[: cut + 1]
    return repaired, drop_prefix, dropped_suffix, interpolated


def _interpolate_span(
    positions: list[list[float]], start_point: int, end_point: int
) -> int:
    """Replace samples strictly between two kept anchors. Returns the count."""
    span = end_point - start_point
    if span <= 1:
        return 0
    origin_x, origin_y = positions[start_point]
    dest_x, dest_y = positions[end_point]
    for step in range(1, span):
        weight = step / span
        positions[start_point + step] = [
            origin_x + (dest_x - origin_x) * weight,
            origin_y + (dest_y - origin_y) * weight,
        ]
    return span - 1


def _reversal_pass(
    positions: list[list[float]], repair: dict[str, Any]
) -> list[list[float]]:
    """Straighten a midway reversal, or cut a track that does not return.

    The freeway direction is the sign of the median Δx. A run moving
    against it by 0.5–10 m, and then resuming, is drawn between the
    last forward sample and the first sample back on that motion,
    including the snap-back when those steps are still off the median.
    A reversal above 10 m at either end is cut. A mid-track reversal
    above 10 m where both directions travel more than 10 m, and the
    path does not return, removes the track.
    """
    steps = _step_vectors(positions)
    if len(steps) < 5:
        return positions
    median_dx = statistics.median(step[0] for step in steps)
    median_dy = statistics.median(step[1] for step in steps)
    if abs(median_dx) < REVERSAL_STEP_M:
        east = sum(max(step[0], 0.0) for step in steps)
        west = sum(max(-step[0], 0.0) for step in steps)
        if east > WRONG_WAY_MIN_METERS and west > WRONG_WAY_MIN_METERS:
            repair["dropped"] = "unfixable_reversal"
            repair["reversal"] = "dropped"
        return [] if repair["dropped"] else positions

    sign = 1.0 if median_dx > 0.0 else -1.0
    opposing_marks = [step[0] * sign < -REVERSAL_STEP_M for step in steps]
    repaired = [point[:] for point in positions]
    drop_prefix = 0
    drop_suffix: Optional[int] = None
    interpolated = False
    cut = False
    consumed_until = -1

    for start, end in _marked_runs(opposing_marks, merge_gap=0):
        if start <= consumed_until:
            continue
        opposing = sum(-steps[index][0] * sign for index in range(start, end + 1))
        if opposing <= REVERSAL_MIN_M:
            continue
        expanded = end
        while expanded + 1 < len(steps) and expanded + 1 - start < SPIKE_MAX_BURST:
            nxt = steps[expanded + 1]
            deviation = math.hypot(nxt[0] - median_dx, nxt[1] - median_dy)
            if deviation < SPIKE_STEP_M:
                break
            expanded += 1
        right = expanded + 1
        intervals = right - start
        closure = math.hypot(
            positions[start][0] + median_dx * intervals - positions[right][0],
            positions[start][1] + median_dy * intervals - positions[right][1],
        )
        span = expanded - start + 1
        returns = closure <= SPIKE_CLOSE_M and span <= SPIKE_MAX_BURST
        has_both_ends = start > 0 and expanded < len(steps) - 1
        if has_both_ends and (returns or opposing <= WRONG_WAY_MIN_METERS):
            repair["reversal_frames"] += _interpolate_span(repaired, start, right)
            interpolated = True
            consumed_until = expanded
            continue
        if has_both_ends and opposing > WRONG_WAY_MIN_METERS:
            forward = sum(step[0] * sign for step in steps if step[0] * sign > 0.0)
            if forward > WRONG_WAY_MIN_METERS:
                repair["dropped"] = "unfixable_reversal"
                repair["reversal"] = "dropped"
                return []
            continue
        if opposing > WRONG_WAY_MIN_METERS and start == 0:
            drop_prefix = max(drop_prefix, expanded + 1)
            cut = True
        elif opposing > WRONG_WAY_MIN_METERS and expanded >= len(steps) - 1:
            drop_suffix = start if drop_suffix is None else min(drop_suffix, start)
            cut = True

    if repair["dropped"]:
        return []
    if drop_prefix:
        repaired = repaired[drop_prefix:]
        repair["dropped_prefix_frames"] += drop_prefix
    if drop_suffix is not None:
        cut_index = max(0, drop_suffix - drop_prefix)
        repair["dropped_suffix_frames"] += max(0, len(repaired) - (cut_index + 1))
        repaired = repaired[: cut_index + 1]
    if (drop_prefix or drop_suffix is not None) and len(repaired) < MIN_REMAINING_STATES:
        repair["dropped"] = "unfixable_reversal"
        repair["reversal"] = "dropped"
        return []
    if cut:
        repair["reversal"] = "cut"
    elif interpolated:
        repair["reversal"] = "interpolated"
    return repaired


def _strange_backstop(
    positions: list[list[float]], repair: dict[str, Any]
) -> list[list[float]]:
    """Drop or trim a track that still has a multi-meter step glitch."""
    steps = _step_vectors(positions)
    medians = _local_step_medians(steps)
    marks = [
        median is not None
        and math.hypot(step[0] - median[0], step[1] - median[1]) >= STRANGE_STEP_M
        for step, median in zip(steps, medians)
    ]
    drop_prefix = 0
    drop_suffix: Optional[int] = None
    for start, end in _marked_runs(marks, SPIKE_MERGE_GAP):
        if start > 0 and end < len(steps) - 1:
            repair["dropped"] = "too_strange"
            repair["reversal"] = "dropped"
            return []
        if start == 0:
            drop_prefix = max(drop_prefix, end + 1)
        if end >= len(steps) - 1:
            drop_suffix = start if drop_suffix is None else min(drop_suffix, start)
    repaired = positions
    if drop_prefix:
        repaired = repaired[drop_prefix:]
        repair["dropped_prefix_frames"] += drop_prefix
    if drop_suffix is not None:
        cut_index = max(0, drop_suffix - drop_prefix)
        repair["dropped_suffix_frames"] += max(0, len(repaired) - (cut_index + 1))
        repaired = repaired[: cut_index + 1]
    if len(repaired) < 2:
        repair["dropped"] = "too_strange"
        repair["reversal"] = "dropped"
        return []
    return repaired


def _repair_track_positions(
    positions: list[list[float]],
) -> tuple[list[list[float]], dict[str, Any]]:
    """Repair single bad samples and a midway reversal. Drop what remains wild."""
    repair = _empty_position_repair()
    repaired = [point[:] for point in positions]
    if len(repaired) < 2:
        repair["dropped"] = "shorter_than_two_states"
        repair["reversal"] = "dropped"
        return [], repair
    for _pass in range(REPAIR_PASSES):
        repaired, copied, smoothed, suffix = _chord_pass(repaired)
        repair["copied_huge_frames"] += copied
        repair["smoothed_frames"] += smoothed
        repair["dropped_suffix_frames"] += suffix
        repaired, prefix, suffix, bursts = _burst_pass(repaired)
        repair["dropped_prefix_frames"] += prefix
        repair["dropped_suffix_frames"] += suffix
        repair["interpolated_bursts"] += bursts
        if len(repaired) < 2:
            repair["dropped"] = "too_strange"
            repair["reversal"] = "dropped"
            return [], repair
    repaired = _reversal_pass(repaired, repair)
    if repair["dropped"]:
        return [], repair
    repaired = _strange_backstop(repaired, repair)
    if repair["dropped"]:
        return [], repair
    return repaired, repair


def _position_repair_summary(repairs: list[dict[str, Any]], removed: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _empty_position_repair()
    totals.pop("dropped")
    totals.pop("reversal")
    changed = 0
    reversal_interpolated = 0
    reversal_cut = 0
    for repair in repairs:
        for key in totals:
            totals[key] += int(repair.get(key, 0))
        if repair.get("reversal") == "interpolated":
            reversal_interpolated += 1
        elif repair.get("reversal") == "cut":
            reversal_cut += 1
        if any(int(repair.get(key, 0)) > 0 for key in totals) or repair.get("reversal") not in (None, "none"):
            changed += 1
    return {
        "spike_step_m": SPIKE_STEP_M,
        "huge_spike_step_m": HUGE_SPIKE_STEP_M,
        "median_radius_frames": SPIKE_MEDIAN_RADIUS,
        "merge_gap_steps": SPIKE_MERGE_GAP,
        "max_burst_steps": SPIKE_MAX_BURST,
        "close_m": SPIKE_CLOSE_M,
        "reversal_min_m": REVERSAL_MIN_M,
        "reversal_step_m": REVERSAL_STEP_M,
        "strange_step_m": STRANGE_STEP_M,
        "min_remaining_states": MIN_REMAINING_STATES,
        "passes": REPAIR_PASSES,
        "tracks_changed": changed,
        "reversal_interpolated_tracks": reversal_interpolated,
        "reversal_cut_tracks": reversal_cut,
        **totals,
        "removed_tracks": removed,
    }


def normalize_object(
    raw: dict[str, Any], road_grid: dict[str, Any], stabilize_headings: bool = True
) -> dict[str, Any]:
    """Resample one native spline at 10 Hz and repair bad samples.

    Velocity and heading are computed from the repaired positions.
    """
    timestamps = [float(value) for value in raw["timestamp"]]
    xs = [float(value) for value in raw["x_position"]]
    ys = [float(value) for value in raw["y_position"]]
    if not (len(timestamps) == len(xs) == len(ys)) or len(timestamps) < 2:
        raise ValueError(f"Trajectory {raw.get('id')} has inconsistent or insufficient samples")

    triples = sorted(zip(timestamps, xs, ys))
    deduped: list[tuple[float, float, float]] = []
    for sample in triples:
        if deduped and sample[0] <= deduped[-1][0]:
            if sample[0] == deduped[-1][0]:
                deduped[-1] = sample
                continue
            raise ValueError(f"Trajectory {raw.get('id')} has decreasing timestamps")
        deduped.append(sample)
    timestamps, xs, ys = map(list, zip(*deduped))

    first_tick = math.ceil(timestamps[0] * SAMPLE_HZ - 1e-5)
    last_tick = math.floor(timestamps[-1] * SAMPLE_HZ + 1e-5)
    if last_tick < first_tick:
        raise ValueError(f"Trajectory {raw.get('id')} is shorter than one 10 Hz state")

    positions: list[list[float]] = []
    source_index = 0
    for tick in range(first_tick, last_tick + 1):
        target = tick / SAMPLE_HZ
        x, source_index = _interpolate(timestamps, xs, target, source_index)
        y, source_index = _interpolate(timestamps, ys, target, source_index)
        positions.append([x * FEET_TO_METERS, y * FEET_TO_METERS])

    positions, position_repair = _repair_track_positions(positions)
    if position_repair["dropped"]:
        return {
            "id": int(raw["id"]),
            "dropped": position_repair["dropped"],
            "position_repair": position_repair,
        }
    first_tick += int(position_repair["dropped_prefix_frames"])
    last_tick -= int(position_repair["dropped_suffix_frames"])

    velocities: list[list[float]] = []
    for index in range(len(positions)):
        if len(positions) == 1:
            vx = vy = 0.0
        elif index == 0:
            vx = (positions[1][0] - positions[0][0]) * SAMPLE_HZ
            vy = (positions[1][1] - positions[0][1]) * SAMPLE_HZ
        elif index == len(positions) - 1:
            vx = (positions[-1][0] - positions[-2][0]) * SAMPLE_HZ
            vy = (positions[-1][1] - positions[-2][1]) * SAMPLE_HZ
        else:
            vx = (positions[index + 1][0] - positions[index - 1][0]) * SAMPLE_HZ / 2
            vy = (positions[index + 1][1] - positions[index - 1][1]) * SAMPLE_HZ / 2
        velocities.append([vx, vy])

    fallback_heading = 0.0 if int(raw.get("direction", 1)) >= 0 else math.pi
    if stabilize_headings:
        headings = _stabilized_headings(positions, fallback_heading)
    else:
        headings = [
            math.atan2(vy, vx) if math.hypot(vx, vy) > 0.05 else fallback_heading
            for vx, vy in velocities
        ]
    speeds = [math.hypot(vx, vy) for vx, vy in velocities]

    quality_flags: list[str] = []
    duration = (last_tick - first_tick) / SAMPLE_HZ
    # Descriptive only. Short tracks are still exported at their own length.
    if duration < 9.0:
        quality_flags.append("shorter_than_9_seconds")
    if max(speeds, default=0.0) > 60.0:
        quality_flags.append("speed_above_60_mps")
    map_x_min, map_x_max = road_grid["x_range_m"]
    map_y_min = road_grid["boundary_y_m"][0]
    map_y_max = road_grid["boundary_y_m"][-1]
    # Vehicles enter and leave beyond the overlapping camera grid. Only
    # flag extreme extrapolation, not that run-in and run-out.
    margin = 200.0
    if any(x < map_x_min - margin or x > map_x_max + margin for x, _ in positions):
        quality_flags.append("more_than_200m_outside_calibrated_x_range")
    if any(y < map_y_min - margin or y > map_y_max + margin for _, y in positions):
        quality_flags.append("outside_calibrated_y_range")

    length_m = float(raw.get("l", 0.0)) * FEET_TO_METERS
    width_m = float(raw.get("w", 0.0)) * FEET_TO_METERS
    height_m = float(raw.get("h", 0.0)) * FEET_TO_METERS
    if length_m < 0.5:
        length_m = 4.5
    if width_m < 0.2:
        width_m = 1.8
    if height_m <= 0.0:
        height_m = 1.5
    length_m *= BOX_SCALE
    width_m *= BOX_SCALE
    height_m *= BOX_SCALE

    source_class = str(raw.get("class", "vehicle"))
    normalized = {
        "id": int(raw["id"]),
        "source_class": source_class,
        "type": drive_object_type(source_class, length_m),
        "direction": int(raw.get("direction", 0)),
        "length_m": length_m,
        "width_m": width_m,
        "height_m": height_m,
        "start_tick": first_tick,
        "sample_hz": SAMPLE_HZ,
        "position_m": positions,
        "velocity_mps": velocities,
        "heading_rad": headings,
        "source_sample_count": len(raw["timestamp"]),
        "position_repair": position_repair,
    }
    if quality_flags:
        normalized["quality_flags"] = quality_flags
    return normalized


def clip_scene_clock(
    raw_objects: list[dict[str, Any]], seconds: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop samples later than `seconds` after the file's first timestamp.

    A track with fewer than two samples left is removed. Parallel sample
    arrays are sliced together. Scalar fields are copied through.
    """
    starts = [
        float(raw["timestamp"][0])
        for raw in raw_objects
        if raw.get("timestamp")
    ]
    if not starts:
        return raw_objects, {
            "clock_seconds": float(seconds),
            "origin_unix": None,
            "limit_unix": None,
            "dropped_tracks": 0,
            "clipped_tracks": 0,
            "kept_tracks": 0,
        }
    origin = min(starts)
    limit = origin + float(seconds)
    kept: list[dict[str, Any]] = []
    dropped = 0
    clipped = 0
    for raw in raw_objects:
        timestamps = [float(value) for value in raw["timestamp"]]
        chosen = [index for index, stamp in enumerate(timestamps) if stamp <= limit + 1e-9]
        if len(chosen) < 2:
            dropped += 1
            continue
        if chosen[-1] != len(timestamps) - 1 or chosen != list(range(len(chosen))):
            clipped += 1
            raw = dict(raw)
            raw["timestamp"] = [raw["timestamp"][index] for index in chosen]
            raw["x_position"] = [raw["x_position"][index] for index in chosen]
            raw["y_position"] = [raw["y_position"][index] for index in chosen]
        kept.append(raw)
    return kept, {
        "clock_seconds": float(seconds),
        "origin_unix": origin,
        "limit_unix": limit,
        "dropped_tracks": dropped,
        "clipped_tracks": clipped,
        "kept_tracks": len(kept),
    }


def normalize_scene(
    i24_root: Path, scene: int, stabilize_headings: bool = True
) -> dict[str, Any]:
    splines_path = i24_root / "data" / "spl_obj" / f"scene{scene}_splobj.json"
    homography_path = i24_root / "data" / "hg" / f"scene{scene}_hg.json"
    with splines_path.open() as file:
        raw_objects = json.load(file)
    source_count = len(raw_objects)
    clock_clip = None
    if scene == 1:
        raw_objects, clock_clip = clip_scene_clock(raw_objects, SCENE1_CLOCK_SECONDS)
    road_grid = load_road_grid(homography_path)
    objects = []
    removed = []
    repairs = []
    for raw in raw_objects:
        obj = normalize_object(raw, road_grid, stabilize_headings=stabilize_headings)
        if obj.get("dropped"):
            removed.append({"id": int(obj["id"]), "reason": obj["dropped"]})
            continue
        objects.append(obj)
        repairs.append(obj["position_repair"])
    return {
        "schema": "i24_3d_full_trajectory_v1",
        "scene": scene,
        "sample_hz": SAMPLE_HZ,
        "units": {
            "position": "meters",
            "velocity": "meters_per_second",
            "heading": "radians",
        },
        "source": str(splines_path),
        "source_object_count": source_count,
        "scene_clock_clip": clock_clip,
        "object_count": len(objects),
        "heading_estimation": _heading_estimation_metadata(stabilize_headings),
        "position_repair": _position_repair_summary(repairs, removed),
        "map": road_grid,
        "objects": objects,
    }


def _object_end_tick(obj: dict[str, Any]) -> int:
    return int(obj["start_tick"]) + len(obj["position_m"]) - 1


def _position_travel_m(points: list[tuple[float, float]]) -> float:
    """Diagonal of the axis-aligned box around the samples."""
    if len(points) < 2:
        return 0.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return math.hypot(max(xs) - min(xs), max(ys) - min(ys))


def _track_points(obj: dict[str, Any]) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in obj["position_m"]]


def _stationary_ids(
    objects: list[dict[str, Any]],
    start_tick: int,
    end_tick: int,
    sdc_id: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Tracks whose samples in the window stay inside a 1 m box.

    The focal track is reported and kept.
    """
    removed = []
    focal_stationary = False
    for obj in objects:
        points = _window_points(obj, start_tick, end_tick)
        if len(points) < 2:
            continue
        travel = _position_travel_m(points)
        if travel > STATIONARY_MAX_METERS:
            continue
        if int(obj["id"]) == int(sdc_id):
            focal_stationary = True
            continue
        removed.append({"id": int(obj["id"]), "travel_m": round(travel, 2)})
    removed.sort(key=lambda item: item["id"])
    return removed, focal_stationary


def choose_sdc(scene_data: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        obj
        for obj in scene_data["objects"]
        if not obj.get("dropped")
        and "position_m" in obj
        and len(obj["position_m"]) >= 2
        and _position_travel_m(_track_points(obj)) > STATIONARY_MAX_METERS
        and "speed_above_60_mps" not in obj.get("quality_flags", [])
        and "outside_calibrated_y_range" not in obj.get("quality_flags", [])
    ]
    if not candidates:
        raise ValueError("No trajectory has two or more states")
    candidates.sort(
        key=lambda obj: (len(obj.get("quality_flags", [])), -len(obj["position_m"]), obj["id"])
    )
    return candidates[0]


def _state_at_tick(obj: dict[str, Any], tick: int) -> tuple[list[float], list[float], float, int]:
    index = tick - int(obj["start_tick"])
    valid = 0 <= index < len(obj["position_m"])
    clamped = max(0, min(index, len(obj["position_m"]) - 1))
    position = obj["position_m"][clamped]
    velocity = obj["velocity_mps"][clamped] if valid else [0.0, 0.0]
    heading = obj["heading_rad"][clamped]
    return position, velocity, heading, int(valid)


def _sample_polyline(geometry: list[dict[str, float]], spacing: float) -> list[dict[str, float]]:
    """Keep both endpoints and insert points every `spacing` meters."""
    if len(geometry) < 2 or spacing <= 0:
        return geometry
    sampled = [geometry[0]]
    for start, end in zip(geometry, geometry[1:]):
        dx = end["x"] - start["x"]
        dy = end["y"] - start["y"]
        dz = end.get("z", 0.0) - start.get("z", 0.0)
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        steps = int(length // spacing)
        for step in range(1, steps + 1):
            t = (step * spacing) / length
            if t >= 1.0:
                break
            sampled.append(_point(
                start["x"] + t * dx,
                start["y"] + t * dy,
                start.get("z", 0.0) + t * dz,
            ))
        sampled.append(_point(end["x"], end["y"], end.get("z", 0.0)))
    return sampled


def _translated_roads(
    roads: list[dict[str, Any]],
    origin: list[float],
    x_bounds: Optional[tuple[float, float]] = None,
) -> list[dict[str, Any]]:
    translated = []
    for road in roads:
        geometry = [
            _point(point["x"] - origin[0], point["y"] - origin[1], point.get("z", 0.0))
            for point in road["geometry"]
        ]
        if x_bounds is not None and len(geometry) == 2:
            low = min(geometry[0]["x"], geometry[1]["x"], x_bounds[0])
            high = max(geometry[0]["x"], geometry[1]["x"], x_bounds[1])
            if geometry[0]["x"] <= geometry[1]["x"]:
                geometry[0]["x"], geometry[1]["x"] = low, high
            else:
                geometry[0]["x"], geometry[1]["x"] = high, low
        translated.append(
            {
                **{key: value for key, value in road.items() if key != "geometry"},
                "geometry": _sample_polyline(geometry, ROAD_SAMPLE_METERS),
            }
        )
    return translated


def _box_corners(
    x: float, y: float, heading: float, length: float, width: float
) -> list[tuple[float, float]]:
    """Clockwise corners. Length is along heading."""
    half_length = length * 0.5
    half_width = width * 0.5
    cosine = math.cos(heading)
    sine = math.sin(heading)
    corners = []
    for along, across in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        corners.append((
            x + along * cosine - across * sine,
            y + along * sine + across * cosine,
        ))
    return corners


def _clip_polygon(
    subject: list[tuple[float, float]],
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
) -> list[tuple[float, float]]:
    """Keep the part of subject on the inside of a clockwise edge."""
    if not subject:
        return []

    def cross(point: tuple[float, float]) -> float:
        return (
            (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
            - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
        )

    def intersection(
        start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[float, float]:
        start_cross = cross(start)
        end_cross = cross(end)
        denom = start_cross - end_cross
        if abs(denom) < 1e-12:
            return end
        weight = start_cross / denom
        return (
            start[0] + weight * (end[0] - start[0]),
            start[1] + weight * (end[1] - start[1]),
        )

    output: list[tuple[float, float]] = []
    previous = subject[-1]
    previous_cross = cross(previous)
    for current in subject:
        current_cross = cross(current)
        if current_cross <= 1e-9:
            if previous_cross > 1e-9:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_cross <= 1e-9:
            output.append(intersection(previous, current))
        previous = current
        previous_cross = current_cross
    return output


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(polygon):
        nxt = polygon[(index + 1) % len(polygon)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) * 0.5


def _intersection_area(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> float:
    """Area shared by two convex boxes."""
    polygon = first
    for index, start in enumerate(second):
        polygon = _clip_polygon(polygon, start, second[(index + 1) % len(second)])
        if len(polygon) < 3:
            return 0.0
    return _polygon_area(polygon)


def _valid_xy(obj: dict[str, Any]) -> list[tuple[float, float]]:
    return [
        (float(point["x"]), float(point["y"]))
        for point, valid in zip(obj["position"], obj["valid"])
        if valid
    ]


def _opposing_meters(points: list[tuple[float, float]], legal_sign: int) -> float:
    """Path length traveled against legal_sign (+1 east, -1 west)."""
    opposing = 0.0
    previous_x = None
    for x_pos, _y_pos in points:
        if previous_x is not None:
            step = x_pos - previous_x
            if step * legal_sign < 0.0:
                opposing += abs(step)
        previous_x = x_pos
    return opposing


def _legal_directions(
    tracks: list[tuple[float, list[tuple[float, float]]]],
) -> dict[str, int]:
    """Majority of net travel on each side of the median.

    tracks are (mean_y, valid xy). A tie uses eastbound below the
    median and westbound above it. The caller passes mean_y already
    compared against the split by the sign of the stored side key.
    """
    votes = {"lower": [0, 0], "upper": [0, 0]}
    for side, points in tracks:
        if len(points) < 2:
            continue
        net = points[-1][0] - points[0][0]
        if abs(net) < WRONG_WAY_MIN_METERS:
            continue
        votes[side][1 if net > 0.0 else 0] += 1
    legal: dict[str, int] = {}
    for side, fallback in (("lower", 1), ("upper", -1)):
        west_votes, east_votes = votes[side]
        if east_votes == west_votes:
            legal[side] = fallback
        else:
            legal[side] = 1 if east_votes > west_votes else -1
    return legal


def _window_points(obj: dict[str, Any], start_tick: int, end_tick: int) -> list[tuple[float, float]]:
    points = []
    for tick in range(start_tick, end_tick + 1):
        position, _velocity, _heading, valid = _state_at_tick(obj, tick)
        if valid:
            points.append((float(position[0]), float(position[1])))
    return points


def _wrong_way_ids(
    objects: list[dict[str, Any]],
    start_tick: int,
    end_tick: int,
    split_y: float,
    sdc_id: int,
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    """Tracks that drive WRONG_WAY_MIN_METERS against their carriageway.

    The focal car is reported but not listed for deletion.
    """
    samples = []
    for obj in objects:
        points = _window_points(obj, start_tick, end_tick)
        if len(points) < 2:
            continue
        mean_y = sum(point[1] for point in points) / len(points)
        side = "lower" if mean_y < split_y else "upper"
        samples.append((obj, points, side, mean_y))
    legal = _legal_directions([(side, points) for _obj, points, side, _mean in samples])
    removed = []
    focal_opposing = 0.0
    for obj, points, side, _mean_y in samples:
        opposing = _opposing_meters(points, legal[side])
        if int(obj["id"]) == int(sdc_id):
            focal_opposing = opposing
            continue
        if opposing > WRONG_WAY_MIN_METERS:
            removed.append(
                {
                    "id": int(obj["id"]),
                    "carriageway": "eastbound" if side == "lower" else "westbound",
                    "opposing_m": round(opposing, 2),
                }
            )
    removed.sort(key=lambda item: item["id"])
    return removed, legal, focal_opposing


def _deep_overlap_ids(
    objects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Smaller car of each pair that overlaps on more than 5 timesteps.

    Overlap means the intersection is more than half the smaller box.
    Pairs are marked together, then the smaller ids are dropped once.
    """
    count = len(objects)
    if count < 2:
        return [], []
    state_count = len(objects[0]["valid"])
    areas = [float(obj["length"]) * float(obj["width"]) for obj in objects]
    deep_steps = [[0 for _ in range(count)] for _ in range(count)]
    for tick in range(state_count):
        alive = [index for index in range(count) if objects[index]["valid"][tick]]
        boxes = {}
        centers = {}
        reach = {}
        for index in alive:
            obj = objects[index]
            point = obj["position"][tick]
            centers[index] = (float(point["x"]), float(point["y"]))
            reach[index] = 0.5 * math.hypot(float(obj["length"]), float(obj["width"]))
            boxes[index] = _box_corners(
                float(point["x"]),
                float(point["y"]),
                float(obj["heading"][tick]),
                float(obj["length"]),
                float(obj["width"]),
            )
        for left_pos, left in enumerate(alive):
            for right in alive[left_pos + 1 :]:
                dx = centers[left][0] - centers[right][0]
                dy = centers[left][1] - centers[right][1]
                limit = reach[left] + reach[right]
                if dx * dx + dy * dy > limit * limit:
                    continue
                shared = _intersection_area(boxes[left], boxes[right])
                smaller = min(areas[left], areas[right])
                if smaller > 0.0 and shared > DEEP_OVERLAP_FRACTION * smaller:
                    deep_steps[left][right] += 1
    pairs = []
    focal_smaller = []
    for left in range(count):
        for right in range(left + 1, count):
            steps = deep_steps[left][right]
            if steps <= DEEP_OVERLAP_MIN_STEPS:
                continue
            if areas[left] < areas[right] or (
                areas[left] == areas[right] and int(objects[left]["id"]) > int(objects[right]["id"])
            ):
                small, large = left, right
            else:
                small, large = right, left
            record = {
                "id": int(objects[small]["id"]),
                "other_id": int(objects[large]["id"]),
                "timesteps": int(steps),
                "area_m2": round(areas[small], 3),
                "other_area_m2": round(areas[large], 3),
            }
            if small == 0:
                focal_smaller.append(record)
                continue
            pairs.append(record)
    pairs.sort(key=lambda item: (item["id"], item["other_id"]))
    return pairs, focal_smaller


def _cleanup_tracks(
    scenario_objects: list[dict[str, Any]],
    scene_objects: list[dict[str, Any]],
    start_tick: int,
    end_tick: int,
    split_y: float,
    sdc_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop wrong-way and stationary tracks, then deep-overlap losers."""
    wrong_way, legal, focal_opposing = _wrong_way_ids(
        scene_objects, start_tick, end_tick, split_y, sdc_id
    )
    stationary, focal_stationary = _stationary_ids(
        scene_objects, start_tick, end_tick, sdc_id
    )
    drop_ids = {item["id"] for item in wrong_way} | {item["id"] for item in stationary}
    kept = [obj for obj in scenario_objects if int(obj["id"]) not in drop_ids]
    overlaps, focal_smaller = _deep_overlap_ids(kept)
    overlap_ids = {item["id"] for item in overlaps}
    cleaned = [obj for obj in kept if int(obj["id"]) not in overlap_ids]
    report = {
        "wrong_way_min_m": WRONG_WAY_MIN_METERS,
        "deep_overlap_fraction": DEEP_OVERLAP_FRACTION,
        "deep_overlap_min_steps": DEEP_OVERLAP_MIN_STEPS,
        "stationary_max_m": STATIONARY_MAX_METERS,
        "legal_direction": {
            "eastbound_lower": "east" if legal["lower"] > 0 else "west",
            "westbound_upper": "east" if legal["upper"] > 0 else "west",
        },
        "focal_opposing_m": round(focal_opposing, 2),
        "focal_kept_despite_wrong_way": focal_opposing > WRONG_WAY_MIN_METERS,
        "focal_kept_despite_stationary": focal_stationary,
        "removed_wrong_way": wrong_way,
        "removed_stationary": stationary,
        "removed_deep_overlap": overlaps,
        "focal_smaller_overlaps_kept": focal_smaller,
    }
    return cleaned, report


def build_scenario(
    scene_data: dict[str, Any],
    sdc_id: int,
    start_tick: int,
    window_index: int,
    max_objects: int,
    state_count: int,
    scenario_id: Optional[str] = None,
    full_range_context: bool = False,
) -> dict[str, Any]:
    """One PufferDrive scenario. The self-driving car is object 0."""
    scene_data = {
        **scene_data,
        "objects": [
            obj
            for obj in scene_data["objects"]
            if not obj.get("dropped") and "position_m" in obj
        ],
    }
    objects_by_id = {obj["id"]: obj for obj in scene_data["objects"]}
    if sdc_id not in objects_by_id:
        raise ValueError(f"No object {sdc_id} in scene {scene_data['scene']}")
    sdc = objects_by_id[sdc_id]
    end_tick = start_tick + state_count - 1
    if start_tick < sdc["start_tick"] or end_tick > _object_end_tick(sdc):
        raise ValueError(f"SDC {sdc_id} does not cover the requested {state_count}-state range")

    origin, _, _, _ = _state_at_tick(sdc, start_tick)
    road_grid = scene_data["map"]
    if "carriageway_split_y_m" in road_grid:
        split_y = float(road_grid["carriageway_split_y_m"])
    else:
        split_y = _carriageway_split_y(road_grid["boundary_y_m"])
    wrong_way, _legal, _focal_opposing = _wrong_way_ids(
        scene_data["objects"], start_tick, end_tick, split_y, sdc_id
    )
    stationary, _focal_stationary = _stationary_ids(
        scene_data["objects"], start_tick, end_tick, sdc_id
    )
    drop_ids = {item["id"] for item in wrong_way} | {item["id"] for item in stationary}
    candidates = []
    for obj in scene_data["objects"]:
        if obj["id"] == sdc_id or obj["id"] in drop_ids:
            continue
        if full_range_context:
            overlap_start = max(start_tick, int(obj["start_tick"]))
            overlap_end = min(end_tick, _object_end_tick(obj))
            if overlap_start > overlap_end:
                continue
            distance_sq = min(
                (obj["position_m"][tick - obj["start_tick"]][0]
                 - sdc["position_m"][tick - sdc["start_tick"]][0]) ** 2
                + (obj["position_m"][tick - obj["start_tick"]][1]
                   - sdc["position_m"][tick - sdc["start_tick"]][1]) ** 2
                for tick in range(overlap_start, overlap_end + 1)
            )
        else:
            position, _, _, valid = _state_at_tick(obj, start_tick)
            if not valid:
                continue
            distance_sq = (position[0] - origin[0]) ** 2 + (position[1] - origin[1]) ** 2
        candidates.append((distance_sq, obj["id"], obj))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected = [sdc] + [item[2] for item in candidates[: max_objects - 1]]

    scenario_objects = []
    for obj in selected:
        positions = []
        velocities = []
        headings = []
        valids = []
        for tick in range(start_tick, end_tick + 1):
            position, velocity, heading, valid = _state_at_tick(obj, tick)
            positions.append(_point(position[0] - origin[0], position[1] - origin[1]))
            velocities.append(_point(velocity[0], velocity[1]))
            headings.append(float(heading))
            valids.append(bool(valid))
        final_valid_index = max(index for index, valid in enumerate(valids) if valid)
        scenario_objects.append(
            {
                "id": int(obj["id"]),
                "type": obj["type"],
                "length": obj["length_m"],
                "width": obj["width_m"],
                "height": obj["height_m"],
                "position": positions,
                "velocity": velocities,
                "heading": headings,
                "valid": valids,
                "goalPosition": positions[final_valid_index],
                "mark_as_expert": 0,
            }
        )

    if scenario_id is None:
        scenario_id = f"i24s{scene_data['scene']}v{sdc_id}w{window_index}"
    all_x = [
        point["x"]
        for obj in scenario_objects
        for point, valid in zip(obj["position"], obj["valid"])
        if valid and math.isfinite(point["x"])
    ]
    all_y = [
        point["y"]
        for obj in scenario_objects
        for point, valid in zip(obj["position"], obj["valid"])
        if valid and math.isfinite(point["y"])
    ]
    # The calibration grid covers the camera overlap. These scenes are
    # straight, so extend the recovered lines over the exported window.
    roads = _translated_roads(
        scene_data["map"]["roads"], origin, (min(all_x) - 50.0, max(all_x) + 50.0)
    )
    # Roads keep the pre-cleanup extent. Wrong-way and stationary ids
    # were already left out of the nearest-object cut; this pass drops
    # deep overlaps and any of those tracks that still got in.
    scenario_objects, cleanup = _cleanup_tracks(
        scenario_objects,
        scene_data["objects"],
        start_tick,
        end_tick,
        split_y,
        sdc_id,
    )
    return {
        "name": scenario_id,
        "scenario_id": scenario_id,
        "objects": scenario_objects,
        "roads": roads,
        "tl_states": [],
        "metadata": {
            "sdc_track_index": 0,
            "frame_interval_micros": 100_000,
            "frame_count": state_count,
            "duration_seconds": (state_count - 1) / SAMPLE_HZ,
            "source_scene": scene_data["scene"],
            "source_sdc_id": sdc_id,
            "source_start_time_unix": start_tick / SAMPLE_HZ,
            "heading_estimation": scene_data.get(
                "heading_estimation", _heading_estimation_metadata(False)
            ),
            "coordinate_origin_m": {"x": origin[0], "y": origin[1]},
            "track_cleanup": cleanup,
            **(
                {"position_repair": scene_data["position_repair"]}
                if "position_repair" in scene_data
                else {}
            ),
        },
    }


def build_track_scenario(
    scene_data: dict[str, Any], sdc_id: int, max_objects: int
) -> dict[str, Any]:
    """One scenario for the self-driving car's whole track, capped at 90 s."""
    objects_by_id = {obj["id"]: obj for obj in scene_data["objects"]}
    if sdc_id not in objects_by_id:
        raise ValueError(f"No object {sdc_id} in scene {scene_data['scene']}")
    sdc = objects_by_id[sdc_id]
    state_count = len(sdc["position_m"])
    if state_count < 2:
        raise ValueError(f"SDC {sdc_id} has fewer than two states")
    if state_count > MAX_TRAJECTORY_STATES:
        state_count = MAX_TRAJECTORY_STATES
    return build_scenario(
        scene_data,
        sdc_id=sdc_id,
        start_tick=int(sdc["start_tick"]),
        window_index=0,
        max_objects=max_objects,
        state_count=state_count,
        scenario_id=f"i24s{scene_data['scene']}v{sdc_id}",
        full_range_context=True,
    )


def write_dataset(scenarios: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Write json/, binaries/map_%03d.bin, and binaries/sdc.bin.

    Map i is the i-th scenario. Filenames sort in that order. Each
    binary stores that scenario's own state count.
    """
    json_dir = output_dir / "json"
    binary_dir = output_dir / "binaries"
    json_dir.mkdir(parents=True, exist_ok=True)
    binary_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sdc_indices: list[int] = []
    for map_index, scenario in enumerate(scenarios):
        validate_scenario(scenario)
        name = f"i24_{map_index:05d}.json"
        path = json_dir / name
        scenario = dict(scenario)
        scenario["name"] = name
        scenario["scenario_id"] = Path(name).stem
        _write_json(path, scenario)
        write_log_binary(scenario, binary_dir / f"map_{map_index:03d}.bin")
        sdc_indices.append(int(scenario["metadata"]["sdc_track_index"]))
        written.append(path)
    if sdc_indices:
        write_sdc_index(binary_dir / "sdc.bin", sdc_indices)
    return written


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(value, file)
        file.write("\n")


def _selected_sdcs(scene_data: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.all_sdcs:
        return sorted(scene_data["objects"], key=lambda obj: obj["id"])
    if args.sdc_id is not None:
        matching = [obj for obj in scene_data["objects"] if obj["id"] == args.sdc_id]
        if not matching:
            raise ValueError(f"No object {args.sdc_id} in scene {scene_data['scene']}")
        return matching
    return [choose_sdc(scene_data)]


def convert(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes = (1, 2, 3) if args.all_scenes else (args.scene,)
    scenarios: list[dict[str, Any]] = []
    selected_sdcs: list[dict[str, int]] = []
    ineligible_sdcs: list[dict[str, Any]] = []
    excluded_sdcs: list[dict[str, Any]] = []
    flag_counts: Counter = Counter()
    scene_summaries = []
    source_object_count = 0

    for scene in scenes:
        scene_data = normalize_scene(
            args.i24_root.resolve(), scene, stabilize_headings=args.stabilize_headings
        )
        source_object_count += int(
            scene_data.get("source_object_count", scene_data["object_count"])
        )
        scene_flag_counts = Counter(
            flag for obj in scene_data["objects"] for flag in obj.get("quality_flags", [])
        )
        flag_counts.update(scene_flag_counts)
        if args.write_full:
            _write_json(
                args.output_dir / f"scene{scene}_full_10hz.json", scene_data
            )

        scene_eligible = 0
        scene_exported = 0
        scene_excluded = 0
        scene_scenarios_before = len(scenarios)
        for sdc in _selected_sdcs(scene_data, args):
            if len(sdc["position_m"]) < 2:
                if args.all_sdcs:
                    ineligible_sdcs.append(
                        {"scene": scene, "id": sdc["id"], "reason": "shorter_than_two_states"}
                    )
                    continue
                raise ValueError(f"SDC {sdc['id']} has fewer than two states")
            scene_eligible += 1

            major_flags = sorted(MAJOR_QUALITY_FLAGS & set(sdc.get("quality_flags", [])))
            if args.exclude_major_flags and major_flags:
                excluded_sdcs.append({"scene": scene, "id": sdc["id"], "flags": major_flags})
                scene_excluded += 1
                continue

            selected_sdcs.append({"scene": scene, "id": sdc["id"]})
            scene_exported += 1
            scenario = build_track_scenario(scene_data, sdc["id"], args.max_objects)
            validate_scenario(scenario, args.max_objects)
            scenarios.append(scenario)

        scene_summaries.append(
            {
                "scene": scene,
                "source_object_count": scene_data["object_count"],
                "eligible_sdc_count": scene_eligible,
                "exported_sdc_count": scene_exported,
                "ineligible_sdc_count": sum(
                    1 for item in ineligible_sdcs if item["scene"] == scene
                ),
                "excluded_major_quality_sdc_count": scene_excluded,
                "scenario_count": len(scenarios) - scene_scenarios_before,
                "quality_flag_counts": dict(sorted(scene_flag_counts.items())),
                "position_repair": scene_data.get("position_repair", {}),
                "scene_clock_clip": scene_data.get("scene_clock_clip"),
            }
        )

    written = write_dataset(scenarios, args.output_dir)
    frame_counts = [int(scenario["metadata"]["frame_count"]) for scenario in scenarios]
    manifest = {
        "schema": "i24_pufferdrive_v1",
        "layout": "pufferdrive",
        "scenes": list(scenes),
        "source_object_count": source_object_count,
        "sample_hz": SAMPLE_HZ,
        "max_scenario_seconds": MAX_SCENARIO_SECONDS,
        "max_trajectory_states": MAX_TRAJECTORY_STATES,
        "min_frame_count": min(frame_counts) if frame_counts else 0,
        "max_frame_count": max(frame_counts) if frame_counts else 0,
        "heading_estimation": _heading_estimation_metadata(args.stabilize_headings),
        "eligible_sdc_count": len(selected_sdcs) + len(excluded_sdcs),
        "exported_sdc_count": len(selected_sdcs),
        "ineligible_sdc_count": len(ineligible_sdcs),
        "excluded_major_quality_sdc_count": len(excluded_sdcs),
        "scenario_count": len(scenarios),
        "log_binary_count": len(scenarios),
        "selected_sdcs": selected_sdcs,
        "ineligible_sdcs": ineligible_sdcs,
        "excluded_sdcs": excluded_sdcs,
        "files": [str(path.relative_to(args.output_dir)) for path in written],
        "quality_flag_counts": dict(sorted(flag_counts.items())),
        "scene_summaries": scene_summaries,
        "notes": [
            "json/ and binaries/ are PufferDrive data.",
            "map_i.bin is the i-th json. Its header length is that scenario's state count.",
            "sdc.bin holds that map's self-driving-car object index.",
            "Each track keeps its own length, capped at 90 s (901 states).",
            "Goals are each track's last valid logged position.",
            "The road model is the scene's 12-foot calibration grid, extended along the roadway and sampled every 5 m. Road edges are 1.5 ft outward from the lane lines. Boxes are 0.90 of the annotated size.",
            "Action labels are the simulator inverse on these logs, as in arXiv 2606.19370 A.4.",
            "Position repair runs on the 10 Hz samples before velocity and heading. A sample at least 0.25 m off the chord of its neighbors is moved onto that chord. A sample at least 1.25 m off is copied from the previous sample first, then moved onto the chord. A run of steps at least 0.25 m off the local median is interpolated when it is at most 30 steps and the far end is within 4 m of the median continuation. A bad prefix is dropped and a bad suffix is cut.",
            "A reversal of 0.5 m to 10 m that resumes is straightened between the last forward sample and the resumed sample. A reversal above 10 m at either end is cut, and the track is removed when under 2 s remain. A mid-track reversal above 10 m that does not return, with both directions traveling more than 10 m, removes the track. An unfixable track is removed before the focal car is chosen.",
            "Heading stabilization does not modify positions. Velocity and heading are recomputed from the repaired positions.",
            "A track that travels more than 10 m against its carriageway is removed. The carriageway direction is the majority of tracks on that side of the median.",
            "A track whose positions stay inside a 1 m box is removed before the nearest-128 cut. The focal track is kept.",
            "When two boxes overlap by more than half the smaller box for more than 5 timesteps, the smaller track is removed. Pairs are marked before any of those tracks are deleted. The focal track is kept.",
        ],
    }
    manifest["track_cleanup"] = [
        {
            "scenario_id": scenario["scenario_id"],
            "source_scene": scenario["metadata"]["source_scene"],
            "source_sdc_id": scenario["metadata"]["source_sdc_id"],
            "object_count": len(scenario["objects"]),
            **scenario["metadata"].get("track_cleanup", {}),
        }
        for scenario in scenarios
    ]
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    default_i24_root = repo_root / "data" / "i24" / "raw" / "I24-3D"
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--i24-root", type=Path, default=default_i24_root)
    argument_parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "data" / "i24"
    )
    argument_parser.add_argument("--scene", type=int, choices=(1, 2, 3), default=2)
    argument_parser.add_argument(
        "--all-scenes", action="store_true", help="Process scenes 1, 2, and 3"
    )
    argument_parser.add_argument(
        "--sdc-id", type=int, default=None,
        help="Default: the longest clean trajectory",
    )
    argument_parser.add_argument(
        "--all-sdcs", action="store_true",
        help="Export every track that has at least two states",
    )
    argument_parser.add_argument(
        "--exclude-major-flags", action="store_true",
        help="Skip focal tracks with extreme speed or range flags",
    )
    argument_parser.add_argument(
        "--stabilize-headings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Estimate headings over a 1.0 s position window and hold the last "
            "reliable orientation below 0.5 m/s. This is the default. "
            "--no-stabilize-headings keeps the per-frame velocity heading."
        ),
    )
    argument_parser.add_argument("--max-objects", type=int, default=128)
    argument_parser.add_argument(
        "--write-full", action="store_true",
        help="Also write the resampled full-scene JSON",
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    if args.all_scenes and not args.all_sdcs:
        raise SystemExit("--all-scenes requires --all-sdcs")
    if args.all_sdcs and args.sdc_id is not None:
        raise SystemExit("--all-sdcs and --sdc-id are mutually exclusive")
    if not 1 <= args.max_objects <= MAX_OBJECTS:
        raise SystemExit(f"--max-objects must be between 1 and {MAX_OBJECTS}")
    if not args.i24_root.is_dir():
        raise SystemExit(f"I24 root does not exist: {args.i24_root}")
    manifest = convert(args)
    summary = {
        "scenes": manifest["scenes"],
        "source_object_count": manifest["source_object_count"],
        "scenario_count": manifest["scenario_count"],
        "log_binary_count": manifest["log_binary_count"],
        "exported_sdc_count": manifest["exported_sdc_count"],
        "manifest": str(args.output_dir / "manifest.json"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
