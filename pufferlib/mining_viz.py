import json
import math
import os
import pickle
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pufferlib.ocean.drive.drive import Drive


def load_compact_replay(path):
    with open(path, "rb") as f:
        replay_bundle = pickle.loads(zlib.decompress(f.read()))

    schema_version = int(replay_bundle.get("schema_version", 0) or 0)
    if schema_version not in (7, 8):
        raise ValueError(f"Unsupported compact replay schema_version={schema_version}. Expected 7 or 8.")

    required_top_level = ("metadata", "agent_arrays", "traffic_arrays", "episode_timesteps")
    missing_top_level = [key for key in required_top_level if key not in replay_bundle]
    if missing_top_level:
        raise ValueError(f"Compact replay is missing required fields: {', '.join(missing_top_level)}")

    return replay_bundle


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _resolve_map_path(map_path):
    path = Path(map_path)
    if path.exists():
        return path.resolve()

    repo_candidate = (_repo_root() / map_path).resolve()
    if repo_candidate.exists():
        return repo_candidate

    cwd_candidate = (Path.cwd() / map_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    raise FileNotFoundError(f"Could not resolve map path: {map_path}")


def _normalize_scenario(scenario):
    if isinstance(scenario, list):
        return scenario[0] if scenario else {}
    return scenario or {}


def _infer_map_simulation_mode(map_path):
    map_name = Path(map_path).name
    if map_name.startswith("opendrive__"):
        return "gigaflow"
    return "replay"


def _ensure_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _impact_zone_label(value):
    zone_code = int(float(value or 0))
    return {
        0: "none",
        1: "front",
        2: "rear",
        3: "left",
        4: "right",
    }.get(zone_code, f"unknown[{zone_code}]")


def _format_summary_for_render(summary):
    if not summary:
        return {}

    formatted = {key: _ensure_python_scalar(value) for key, value in summary.items()}
    if "target_collision_impact_zone" in formatted:
        formatted["target_collision_impact_zone_label"] = _impact_zone_label(
            formatted.get("target_collision_impact_zone", 0)
        )
    return formatted


def _materialize_agent_frames(replay_bundle):
    agent_arrays = replay_bundle["agent_arrays"]
    valid = agent_arrays["valid"]

    frames = []
    num_frames = int(valid.shape[0]) if hasattr(valid, "shape") and valid.ndim > 0 else 0
    for frame_idx in range(num_frames):
        frame = []
        frame_valid = valid[frame_idx]
        for slot_idx in np.flatnonzero(frame_valid):
            frame.append(
                {
                    "id": int(agent_arrays["id"][frame_idx, slot_idx]),
                    "type": int(agent_arrays["type"][frame_idx, slot_idx]),
                    "is_target": bool(agent_arrays["is_target"][frame_idx, slot_idx]),
                    "active": bool(agent_arrays["active"][frame_idx, slot_idx]),
                    "stopped": bool(agent_arrays["stopped"][frame_idx, slot_idx]),
                    "x": float(agent_arrays["x"][frame_idx, slot_idx]),
                    "y": float(agent_arrays["y"][frame_idx, slot_idx]),
                    "z": float(agent_arrays["z"][frame_idx, slot_idx]),
                    "heading": float(agent_arrays["heading"][frame_idx, slot_idx]),
                    "length": float(agent_arrays["length"][frame_idx, slot_idx]),
                    "width": float(agent_arrays["width"][frame_idx, slot_idx]),
                    "height": float(agent_arrays["height"][frame_idx, slot_idx]),
                    "vx": float(agent_arrays["vx"][frame_idx, slot_idx]),
                    "vy": float(agent_arrays["vy"][frame_idx, slot_idx]),
                }
            )
        frames.append(frame)
    return frames


def _materialize_traffic_frames(replay_bundle):
    traffic_arrays = replay_bundle["traffic_arrays"]
    valid = traffic_arrays["valid"]

    frames = []
    num_frames = int(valid.shape[0]) if hasattr(valid, "shape") and valid.ndim > 0 else 0
    for frame_idx in range(num_frames):
        frame = []
        frame_valid = valid[frame_idx]
        for slot_idx in np.flatnonzero(frame_valid):
            frame.append(
                {
                    "type": int(traffic_arrays["type"][frame_idx, slot_idx]),
                    "state": int(traffic_arrays["state"][frame_idx, slot_idx]),
                    "stop_line": [float(coord) for coord in traffic_arrays["stop_line"][frame_idx, slot_idx].tolist()],
                }
            )
        frames.append(frame)
    return frames


def _materialize_replay_bundle(replay_bundle):
    materialized = dict(replay_bundle)
    materialized["agent_frames"] = _materialize_agent_frames(replay_bundle)
    materialized["traffic_frames"] = _materialize_traffic_frames(replay_bundle)
    return materialized


@lru_cache(maxsize=16)
def load_map_static(map_path):
    resolved_map_path = _resolve_map_path(map_path)
    simulation_mode = _infer_map_simulation_mode(resolved_map_path)
    env = Drive(
        map_dir=str(resolved_map_path.parent),
        maps=resolved_map_path.name,
        num_maps=1,
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        simulation_mode=simulation_mode,
        eval_mode=1 if simulation_mode == "replay" else 0,
        num_eval_scenarios=1,
        scenario_length=91 if simulation_mode == "replay" else 1,
        resample_frequency=91 if simulation_mode == "replay" else 0,
        report_interval=10_000,
    )
    try:
        scenario = _normalize_scenario(env.get_state())
    finally:
        env.close()

    return {
        "map_name": scenario.get("map_name"),
        "map_corners": scenario.get("map_corners", []),
        "road_elements": scenario.get("road_elements", []),
        "traffic_elements": [
            {
                "type": element.get("type"),
                "stop_line": element.get("stop_line"),
                "heading": element.get("heading"),
                "controlled_lanes": element.get("controlled_lanes"),
            }
            for element in scenario.get("traffic_elements", [])
        ],
    }


COMPLIANCE_WINDOW_SECONDS = 5.0
COMPLIANCE_WRONG_WAY_DISTANCE_THRESHOLD = 2.0
COMPLIANCE_SPEED_LIMIT_RATIO_THRESHOLD = 1.05
COMPLIANCE_SOLID_LINE_TYPES = {12, 13, 16, 17}
COMPLIANCE_LANE_TYPES = {1, 2}


@lru_cache(maxsize=16)
def _compliance_map_geometry(map_path):
    map_static = load_map_static(map_path)
    road_elements = map_static.get("road_elements", [])
    lanes = []
    solid_segments = []
    for line_index, element in enumerate(road_elements):
        element_type = int(element.get("type", 0) or 0)
        xs = element.get("x") or []
        ys = element.get("y") or []
        zs = element.get("z") or []
        if element_type in COMPLIANCE_LANE_TYPES:
            lanes.append(
                {
                    "line_index": line_index,
                    "speed_limit": float(element.get("speed_limit", 0.0) or 0.0),
                    "x": [float(value) for value in xs],
                    "y": [float(value) for value in ys],
                    "z": [float(value) for value in zs],
                }
            )
        for segment_index in range(min(len(xs), len(ys)) - 1):
            z1 = float(zs[segment_index]) if segment_index < len(zs) else 0.0
            z2 = float(zs[segment_index + 1]) if segment_index + 1 < len(zs) else z1
            segment = (
                float(xs[segment_index]),
                float(ys[segment_index]),
                z1,
                float(xs[segment_index + 1]),
                float(ys[segment_index + 1]),
                z2,
            )
            if element_type in COMPLIANCE_SOLID_LINE_TYPES:
                solid_segments.append((line_index, segment_index, element_type, segment))
    return lanes, solid_segments, map_static.get("traffic_elements", [])


def _segment_intersection(a1, a2, b1, b2, epsilon=1e-9):
    ax = a2[0] - a1[0]
    ay = a2[1] - a1[1]
    bx = b2[0] - b1[0]
    by = b2[1] - b1[1]
    cross = ax * by - ay * bx
    if abs(cross) <= epsilon:
        return False
    dx = b1[0] - a1[0]
    dy = b1[1] - a1[1]
    ta = (dx * by - dy * bx) / cross
    tb = (dx * ay - dy * ax) / cross
    return -epsilon <= ta <= 1.0 + epsilon and -epsilon <= tb <= 1.0 + epsilon


def _angle_difference(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _closest_compliance_lane(sample, lanes, current_lane_index=-1):
    best = None
    best_score = float("inf")
    x = sample["x"]
    y = sample["y"]
    z = sample["z"]
    max_distance_sq = (3.0 * max(sample.get("width", 0.0), 0.1)) ** 2
    for lane in lanes:
        xs = lane["x"]
        ys = lane["y"]
        zs = lane["z"]
        closest_segment = -1
        closest_distance_sq = float("inf")
        for segment_index in range(min(len(xs), len(ys)) - 1):
            z1 = zs[segment_index] if segment_index < len(zs) else 0.0
            z2 = zs[segment_index + 1] if segment_index + 1 < len(zs) else z1
            if min(abs(z1 - z), abs(z2 - z)) > 4.0:
                continue
            x1, y1 = xs[segment_index], ys[segment_index]
            x2, y2 = xs[segment_index + 1], ys[segment_index + 1]
            dx = x2 - x1
            dy = y2 - y1
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-12:
                continue
            projection = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq))
            px = x1 + projection * dx
            py = y1 + projection * dy
            distance_sq = (x - px) ** 2 + (y - py) ** 2
            if distance_sq < closest_distance_sq:
                closest_distance_sq = distance_sq
                closest_segment = segment_index
        if closest_segment < 0 or closest_distance_sq > max_distance_sq:
            continue

        start = max(0, closest_segment - 1)
        end = min(len(xs) - 2, closest_segment + 1)
        headings = [math.atan2(ys[index + 1] - ys[index], xs[index + 1] - xs[index]) for index in range(start, end + 1)]
        weights = [2.0 if index == closest_segment else 1.0 for index in range(start, end + 1)]
        sin_sum = sum(weight * math.sin(heading) for weight, heading in zip(weights, headings))
        cos_sum = sum(weight * math.cos(heading) for weight, heading in zip(weights, headings))
        lane_heading = math.atan2(sin_sum, cos_sum)
        distance = math.sqrt(closest_distance_sq)
        heading_penalty = abs(_angle_difference(sample["heading"], lane_heading)) / math.pi
        score = 0.7 * distance / 4.0 + 0.3 * heading_penalty
        if current_lane_index >= 0 and lane["line_index"] != current_lane_index:
            score += 0.05
        if score < best_score:
            best_score = score
            best = {
                "line_index": lane["line_index"],
                "segment_index": closest_segment,
                "heading": lane_heading,
                "speed_limit": lane["speed_limit"],
            }
    return best


def _crossed_solid_line(previous, sample, solid_segments):
    movement_start = (previous["x"], previous["y"])
    movement_end = (sample["x"], sample["y"])
    for line_index, segment_index, line_type, segment in solid_segments:
        x1, y1, z1, x2, y2, z2 = segment
        if min(abs(z1 - sample["z"]), abs(z2 - sample["z"])) > 4.0:
            continue
        if _segment_intersection(movement_start, movement_end, (x1, y1), (x2, y2)):
            return line_index, segment_index, line_type
    return None


def _agent_corners(sample):
    half_length = 0.5 * sample["length"]
    half_width = 0.5 * sample["width"]
    cos_heading = math.cos(sample["heading"])
    sin_heading = math.sin(sample["heading"])
    return (
        (
            sample["x"] + half_length * cos_heading - half_width * sin_heading,
            sample["y"] + half_length * sin_heading + half_width * cos_heading,
        ),
        (
            sample["x"] + half_length * cos_heading + half_width * sin_heading,
            sample["y"] + half_length * sin_heading - half_width * cos_heading,
        ),
        (
            sample["x"] - half_length * cos_heading + half_width * sin_heading,
            sample["y"] - half_length * sin_heading - half_width * cos_heading,
        ),
        (
            sample["x"] - half_length * cos_heading - half_width * sin_heading,
            sample["y"] - half_length * sin_heading + half_width * cos_heading,
        ),
    )


def _crossed_red_light(sample, replay_bundle, lane_index, previous_lane_index, traffic_elements):
    frame_index = sample.get("frame_index")
    if frame_index is None:
        return None
    if lane_index < 0:
        return -1
    traffic = replay_bundle["traffic_arrays"]
    if frame_index >= traffic["valid"].shape[0]:
        return None
    corners = _agent_corners(sample)
    for control_index in np.flatnonzero(traffic["valid"][frame_index]):
        if int(traffic["type"][frame_index, control_index]) != 1:
            continue
        if int(traffic["state"][frame_index, control_index]) != 1:
            continue
        static_control = traffic_elements[int(control_index)]
        controlled_lanes = static_control.get("controlled_lanes") or []
        if lane_index not in controlled_lanes:
            continue
        stop_line = traffic["stop_line"][frame_index, control_index]
        if min(abs(float(stop_line[2]) - sample["z"]), abs(float(stop_line[5]) - sample["z"])) > 4.0:
            continue
        midpoint_x = 0.5 * (float(stop_line[0]) + float(stop_line[3]))
        midpoint_y = 0.5 * (float(stop_line[1]) + float(stop_line[4]))
        if (sample["x"] - midpoint_x) ** 2 + (sample["y"] - midpoint_y) ** 2 > 100.0:
            continue
        if previous_lane_index >= 0 and previous_lane_index != lane_index:
            return int(control_index)
        control_heading = static_control.get("heading")
        if control_heading is None or abs(_angle_difference(sample["heading"], float(control_heading))) > math.pi / 4:
            continue
        line_dx = float(stop_line[3]) - float(stop_line[0])
        line_dy = float(stop_line[4]) - float(stop_line[1])
        extended_start = (float(stop_line[0]) - 0.25 * line_dx, float(stop_line[1]) - 0.25 * line_dy)
        extended_end = (float(stop_line[3]) + 0.25 * line_dx, float(stop_line[4]) + 0.25 * line_dy)
        for edge_index in (0, 1, 3):
            if _segment_intersection(corners[edge_index], corners[(edge_index + 1) % 4], extended_start, extended_end):
                return int(control_index)
    return -1


def _historical_hitter_samples(replay_bundle, hitter_index, collision_timestep, collision_snapshot):
    arrays = replay_bundle["agent_arrays"]
    timesteps = replay_bundle["episode_timesteps"]
    samples = []
    for frame_index, timestep in enumerate(timesteps):
        if hitter_index >= arrays["valid"].shape[1] or not arrays["valid"][frame_index, hitter_index]:
            continue
        samples.append(
            {
                "frame_index": frame_index,
                "timestep": int(timestep),
                "x": float(arrays["x"][frame_index, hitter_index]),
                "y": float(arrays["y"][frame_index, hitter_index]),
                "z": float(arrays["z"][frame_index, hitter_index]),
                "heading": float(arrays["heading"][frame_index, hitter_index]),
                "length": float(arrays["length"][frame_index, hitter_index]),
                "vx": float(arrays["vx"][frame_index, hitter_index]),
                "vy": float(arrays["vy"][frame_index, hitter_index]),
                "width": float(arrays["width"][frame_index, hitter_index]),
            }
        )
    if collision_snapshot and collision_snapshot.get("valid"):
        samples.append(
            {
                "frame_index": None,
                "timestep": int(collision_timestep),
                "x": float(collision_snapshot["x"]),
                "y": float(collision_snapshot["y"]),
                "z": float(collision_snapshot.get("z", 0.0)),
                "heading": float(collision_snapshot.get("heading", 0.0)),
                "length": float(collision_snapshot.get("length", 0.0)),
                "vx": float(collision_snapshot.get("vx", 0.0)),
                "vy": float(collision_snapshot.get("vy", 0.0)),
                "width": float(collision_snapshot.get("width", 0.0)),
            }
        )
    samples.sort(key=lambda sample: sample["timestep"])
    return samples


def reconstruct_compliance_diagnostics(replay_bundle):
    avoidability = replay_bundle.get("avoidability_debug") or {}
    collision = avoidability.get("collision") or {}
    if not collision:
        return None

    hitter_index = int(collision.get("collision_adversary_index", -1))
    collision_timestep = int(collision.get("collision_timestep", -1))
    if hitter_index < 0 or collision_timestep < 0:
        return None
    dt = float((avoidability.get("constants") or {}).get("dt", 0.1) or 0.1)
    nominal_window_start = collision_timestep - int(math.ceil(COMPLIANCE_WINDOW_SECONDS / dt))
    samples = _historical_hitter_samples(replay_bundle, hitter_index, collision_timestep, collision.get("adversary"))
    samples = [sample for sample in samples if sample["timestep"] >= nominal_window_start]
    if not samples:
        return None

    map_path = replay_bundle.get("metadata", {}).get("map_path")
    lanes, solid_segments, traffic_elements = _compliance_map_geometry(str(_resolve_map_path(map_path)))
    diagnostics = {
        "valid": 1,
        "source": "reconstructed",
        "compliant": 1,
        "hitter_agent_index": hitter_index,
        "hitter_agent_id": int((collision.get("adversary") or {}).get("id", hitter_index)),
        "collision_timestep": collision_timestep,
        "dt": dt,
        "window_seconds": COMPLIANCE_WINDOW_SECONDS,
        "window_start_timestep": samples[0]["timestep"],
        "window_sample_count": len(samples),
        "lane_sample_count": 0,
        "lane_unavailable_sample_count": 0,
        "speed_limit_sample_count": 0,
        "speed_limit_unavailable_sample_count": 0,
        "red_light_sample_count": 0,
        "red_light_unavailable_sample_count": 0,
        "red_light_violation": 0,
        "wrong_way_violation": 0,
        "solid_line_violation": 0,
        "speed_limit_violation": 0,
        "first_red_light_timestep": -1,
        "first_wrong_way_timestep": -1,
        "first_solid_line_timestep": -1,
        "first_speed_limit_timestep": -1,
        "wrong_way_distance": 0.0,
        "max_speed_ratio": 0.0,
        "crossed_line_index": -1,
        "crossed_line_segment_index": -1,
        "crossed_line_type": -1,
        "crossing_segment_start_x": 0.0,
        "crossing_segment_start_y": 0.0,
        "crossing_segment_end_x": 0.0,
        "crossing_segment_end_y": 0.0,
        "wrong_way_distance_threshold": COMPLIANCE_WRONG_WAY_DISTANCE_THRESHOLD,
        "speed_limit_ratio_threshold": COMPLIANCE_SPEED_LIMIT_RATIO_THRESHOLD,
        "hitter_trajectory": [
            {"timestep": sample["timestep"], "x": sample["x"], "y": sample["y"]} for sample in samples
        ],
    }

    previous = None
    current_lane_index = -1
    for sample in samples:
        previous_lane_index = current_lane_index
        lane = _closest_compliance_lane(sample, lanes, current_lane_index)
        if lane is None:
            current_lane_index = -1
            diagnostics["lane_unavailable_sample_count"] += 1
            diagnostics["speed_limit_unavailable_sample_count"] += 1
        else:
            current_lane_index = lane["line_index"]
            diagnostics["lane_sample_count"] += 1
            speed_limit = lane["speed_limit"]
            if speed_limit > 0.0:
                diagnostics["speed_limit_sample_count"] += 1
                speed_ratio = math.hypot(sample["vx"], sample["vy"]) / speed_limit
                diagnostics["max_speed_ratio"] = max(diagnostics["max_speed_ratio"], speed_ratio)
                if speed_ratio > COMPLIANCE_SPEED_LIMIT_RATIO_THRESHOLD:
                    diagnostics["speed_limit_violation"] = 1
                    if diagnostics["first_speed_limit_timestep"] < 0:
                        diagnostics["first_speed_limit_timestep"] = sample["timestep"]
            else:
                diagnostics["speed_limit_unavailable_sample_count"] += 1

            if sample["timestep"] > nominal_window_start:
                lane_vx = math.cos(lane["heading"])
                lane_vy = math.sin(lane["heading"])
                wrong_speed = max(0.0, -(sample["vx"] * lane_vx + sample["vy"] * lane_vy))
                diagnostics["wrong_way_distance"] += wrong_speed * dt
                if (
                    diagnostics["wrong_way_distance"] > COMPLIANCE_WRONG_WAY_DISTANCE_THRESHOLD
                    and diagnostics["first_wrong_way_timestep"] < 0
                ):
                    diagnostics["wrong_way_violation"] = 1
                    diagnostics["first_wrong_way_timestep"] = sample["timestep"]

        red_control = _crossed_red_light(
            sample, replay_bundle, current_lane_index, previous_lane_index, traffic_elements
        )
        if red_control is None:
            diagnostics["red_light_unavailable_sample_count"] += 1
        else:
            diagnostics["red_light_sample_count"] += 1
            if red_control >= 0:
                diagnostics["red_light_violation"] = 1
                if diagnostics["first_red_light_timestep"] < 0:
                    diagnostics["first_red_light_timestep"] = sample["timestep"]

        if previous is not None and sample["timestep"] > nominal_window_start:
            crossing = _crossed_solid_line(previous, sample, solid_segments)
            if crossing is not None:
                diagnostics["solid_line_violation"] = 1
                if diagnostics["first_solid_line_timestep"] < 0:
                    diagnostics["first_solid_line_timestep"] = sample["timestep"]
                    diagnostics["crossed_line_index"] = crossing[0]
                    diagnostics["crossed_line_segment_index"] = crossing[1]
                    diagnostics["crossed_line_type"] = crossing[2]
                    diagnostics["crossing_segment_start_x"] = previous["x"]
                    diagnostics["crossing_segment_start_y"] = previous["y"]
                    diagnostics["crossing_segment_end_x"] = sample["x"]
                    diagnostics["crossing_segment_end_y"] = sample["y"]
        previous = sample

    diagnostics["wrong_way_violation"] = int(
        diagnostics["wrong_way_distance"] > COMPLIANCE_WRONG_WAY_DISTANCE_THRESHOLD
    )
    diagnostics["compliant"] = int(
        not any(
            diagnostics[key]
            for key in (
                "red_light_violation",
                "wrong_way_violation",
                "solid_line_violation",
                "speed_limit_violation",
            )
        )
    )
    return diagnostics


def _compute_bounds(map_static, replay_bundle):
    corners = map_static.get("map_corners") or []
    if len(corners) >= 4:
        min_x, min_y, max_x, max_y = [float(v) for v in corners[:4]]
        return [min_x, min_y, max_x, max_y]

    xs, ys = [], []
    for elem in map_static.get("road_elements", []):
        xs.extend(elem.get("x", []))
        ys.extend(elem.get("y", []))
    for frame in replay_bundle.get("agent_frames", []):
        for agent in frame:
            xs.append(agent["x"])
            ys.append(agent["y"])

    if not xs or not ys:
        return [-100.0, -100.0, 100.0, 100.0]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = max(10.0, 0.05 * max(max_x - min_x, max_y - min_y, 1.0))
    return [min_x - pad, min_y - pad, max_x + pad, max_y + pad]


def _build_render_payload(replay_bundle):
    materialized_bundle = _materialize_replay_bundle(replay_bundle)
    metadata = {key: _ensure_python_scalar(value) for key, value in materialized_bundle.get("metadata", {}).items()}
    map_static = load_map_static(metadata["map_path"])
    compliance = replay_bundle.get("compliance_diagnostics")
    if compliance:
        compliance = {key: _ensure_python_scalar(value) for key, value in compliance.items()}
        compliance.setdefault("source", "simulator")
        collision = (replay_bundle.get("avoidability_debug") or {}).get("collision") or {}
        samples = _historical_hitter_samples(
            replay_bundle,
            int(compliance.get("hitter_agent_index", -1)),
            int(compliance.get("collision_timestep", -1)),
            collision.get("adversary"),
        )
        window_start = int(compliance.get("window_start_timestep", -1))
        compliance["hitter_trajectory"] = [
            {"timestep": sample["timestep"], "x": sample["x"], "y": sample["y"]}
            for sample in samples
            if sample["timestep"] >= window_start
        ]
    else:
        compliance = reconstruct_compliance_diagnostics(replay_bundle)
    return {
        "metadata": metadata,
        "map": map_static,
        "bounds": _compute_bounds(map_static, materialized_bundle),
        "agent_frames": materialized_bundle.get("agent_frames", []),
        "traffic_frames": materialized_bundle.get("traffic_frames", []),
        "episode_timesteps": [int(value) for value in replay_bundle.get("episode_timesteps", [])],
        "avoidability_debug": replay_bundle.get("avoidability_debug"),
        "compliance_diagnostics": compliance,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #20262d;
      --panel: rgba(42,49,58,0.96);
      --text: #edf2f7;
      --muted: #aeb8c4;
      --border: rgba(255,255,255,0.12);
      --accent: #62b6ed;
      --target: #f05252;
      --stopped: #f59e0b;
      --active: #3b82f6;
      --inactive: #8995a3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #282f37 0%, #1d2228 100%);
    }
    .layout {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr) 300px;
      min-height: 100vh;
      gap: 18px;
      padding: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.22);
      backdrop-filter: blur(8px);
    }
    .sidebar { padding: 18px; }
    .title { margin: 0 0 6px; font-size: 22px; line-height: 1.15; }
    .subtitle { margin: 0 0 18px; color: var(--muted); font-size: 13px; }
    .meta {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin-top: 16px;
      margin-bottom: 18px;
    }
    .meta-item {
      padding: 10px 12px;
      border-radius: 12px;
      background: rgba(98,182,237,0.07);
      border: 1px solid rgba(98,182,237,0.10);
    }
    .meta-label { display: block; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
    .meta-value { display: block; margin-top: 4px; font-size: 16px; font-weight: 600; }
    .drive-weight-card {
      margin: 14px 0 16px;
      padding: 14px 16px;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(98,182,237,0.18), rgba(98,182,237,0.05));
      border: 1px solid rgba(98,182,237,0.22);
    }
    .drive-weight-card .meta-label { color: #a7d8f5; }
    .drive-weight-card .meta-value { font-size: 28px; color: #d8effd; letter-spacing: -0.03em; }
    .controls {
      display: grid;
      gap: 12px;
      margin-top: 0;
      margin-bottom: 12px;
    }
    .mode-toggle {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      padding: 4px;
      border-radius: 12px;
      background: rgba(255,255,255,0.06);
    }
    .mode-toggle button {
      border: 0;
      background: transparent;
      color: var(--muted);
      font-weight: 650;
    }
    .mode-toggle button.active {
      background: #39424d;
      color: var(--accent);
      box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }
    .mode-toggle button:disabled {
      cursor: not-allowed;
      opacity: 0.4;
    }
    .observed-controls, .avoidability-controls { display: grid; gap: 12px; }
    .avoidability-controls[hidden], .observed-controls[hidden] { display: none; }
    .detection-controls, .braking-controls { display: grid; gap: 12px; }
    .detection-controls[hidden], .braking-controls[hidden] { display: none; }
    .avoidability-result {
      padding: 11px 12px;
      border-radius: 12px;
      border: 1px solid rgba(39,174,96,0.18);
      background: rgba(39,174,96,0.08);
      color: #86efac;
      font-size: 13px;
      line-height: 1.35;
    }
    .avoidability-result.failed {
      border-color: rgba(214,69,69,0.18);
      background: rgba(214,69,69,0.08);
      color: #fca5a5;
    }
    .avoidability-result.reference {
      border-color: rgba(18,97,160,0.18);
      background: rgba(18,97,160,0.07);
      color: #93c5fd;
    }
    .avoidability-result strong { display: block; margin-bottom: 3px; }
    .compliance-card { margin: 14px 0; }
    .compliance-jumps { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .compliance-jumps button { padding: 5px 8px; font-size: 12px; }
    .rollout-markers {
      padding: 9px 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.05);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }
    .detector-toggles { display: flex; gap: 8px; }
    .detector-toggle {
      flex: 1;
      color: var(--muted);
      background: rgba(255,255,255,0.05);
    }
    .detector-toggle.active {
      color: var(--accent);
      border-color: rgba(18,97,160,0.35);
      background: rgba(18,97,160,0.09);
      font-weight: 700;
    }
    .detector-hud {
      position: absolute;
      z-index: 3;
      top: 72px;
      right: 30px;
      display: grid;
      gap: 7px;
      justify-items: end;
      pointer-events: none;
    }
    .detector-hud[hidden] { display: none; }
    .detector-badge {
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid rgba(22,163,74,0.3);
      background: rgba(20,83,45,0.94);
      color: #bbf7d0;
      font-size: 12px;
      font-weight: 750;
      box-shadow: 0 3px 12px rgba(31,41,51,0.1);
    }
    .detector-badge.danger {
      border-color: rgba(220,38,38,0.35);
      background: rgba(127,29,29,0.95);
      color: #fecaca;
    }
    .nav {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }
    .nav a, .nav button {
      text-align: center;
      text-decoration: none;
    }
    .controls-row {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: center;
    }
    button, select {
      border: 1px solid var(--border);
      background: #343c46;
      color: var(--text);
      border-radius: 10px;
      padding: 8px 12px;
      font-size: 14px;
      cursor: pointer;
    }
    input[type="range"] { width: 100%; }
    .legend {
      margin-top: 20px;
      display: grid;
      gap: 8px;
      font-size: 13px;
    }
    .legend-row { display: flex; align-items: center; gap: 8px; color: var(--muted); }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 4px;
      border: 1px solid rgba(0,0,0,0.08);
    }
    .viewer {
      position: relative;
      padding: 18px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 12px;
    }
    .episode-list {
      padding: 18px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 12px;
      min-height: 0;
    }
    .episode-list h3 {
      margin: 0;
      font-size: 16px;
    }
    .episode-scroll {
      overflow: auto;
      display: grid;
      gap: 8px;
      padding-right: 4px;
    }
    .episode-link {
      display: grid;
      gap: 3px;
      text-decoration: none;
      color: var(--text);
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.05);
    }
    .episode-link.active {
      border-color: rgba(18,97,160,0.35);
      background: rgba(18,97,160,0.09);
    }
    .episode-link small {
      color: var(--muted);
    }
    .badge {
      justify-self: start;
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 700;
      background: rgba(39,174,96,0.12);
      color: #86efac;
    }
    .badge.fail {
      background: rgba(214,69,69,0.12);
      color: #fca5a5;
    }
    .viewer-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }
    .viewer-title {
      margin: 0;
      font-size: 18px;
    }
    .viewer-subtitle {
      margin: 2px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .viewer-tools {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    canvas {
      width: 100%;
      height: calc(100vh - 96px);
      min-height: 540px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #303840;
      display: block;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 600;
      background: rgba(214,69,69,0.1);
      color: #fca5a5;
    }
    .pill.ok {
      background: rgba(39,174,96,0.12);
      color: #86efac;
    }
    @media (max-width: 1280px) {
      .layout {
        grid-template-columns: 320px minmax(0, 1fr);
      }
      .episode-list {
        grid-column: 1 / -1;
      }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel sidebar">
      <h1 class="title" id="title"></h1>
      <p class="subtitle" id="subtitle"></p>
      <div class="drive-weight-card" id="drive-weight-card">
        <span class="meta-label">Adv Drive Weight</span>
        <span class="meta-value" id="meta-adv-drive-weight">n/a</span>
      </div>
      <div class="controls">
        <div class="mode-toggle" aria-label="Replay mode">
          <button id="observed-mode" class="active" type="button">Observed</button>
          <button id="avoidability-mode" type="button">Avoidability</button>
        </div>
        <div id="observed-controls" class="observed-controls">
          <div class="controls-row">
            <button id="play-toggle" type="button">Play</button>
            <input id="frame-slider" type="range" min="0" max="0" value="0">
            <span id="frame-label">0 / 0</span>
          </div>
          <div class="controls-row">
            <span>Speed</span>
            <input id="speed-slider" type="range" min="1" max="12" value="6">
            <span id="speed-label">1.0x</span>
          </div>
        </div>
        <div id="avoidability-controls" class="avoidability-controls" hidden>
          <div class="mode-toggle" aria-label="Avoidability view">
            <button id="detection-phase" class="active" type="button">Detection window</button>
            <button id="braking-phase" type="button">Braking rollout</button>
          </div>
          <div id="detection-controls" class="detection-controls">
            <span class="meta-label">Recorded detection sample</span>
            <div class="controls-row">
              <button id="detection-play" type="button">Play</button>
              <input id="detection-slider" type="range" min="0" max="0" value="0" step="1">
              <span id="detection-label">n/a</span>
            </div>
            <div id="detection-result" class="avoidability-result reference">
              <strong>Detection window unavailable</strong>
            </div>
            <div id="detection-markers" class="rollout-markers"></div>
            <div class="detector-toggles">
              <button id="buffer-toggle" class="detector-toggle active" type="button" aria-pressed="true">Safety buffer</button>
              <button id="ttc-toggle" class="detector-toggle active" type="button" aria-pressed="true">TTC</button>
            </div>
          </div>
          <div id="braking-controls" class="braking-controls" hidden>
            <span class="meta-label">Brake start</span>
            <div class="controls-row">
              <span>0.0s</span>
              <input id="reaction-slider" type="range" min="0" max="0" value="0" step="1">
              <span id="reaction-label">collision</span>
            </div>
            <div id="avoidability-result" class="avoidability-result reference">
              <strong>Observed collision</strong>
            </div>
            <span class="meta-label">Counterfactual time</span>
            <div class="controls-row">
              <button id="rollout-play" type="button">Play</button>
              <input id="rollout-slider" type="range" min="0" max="0" value="0" step="1">
              <span id="rollout-label">t=0.0s</span>
            </div>
            <div id="rollout-markers" class="rollout-markers">Select a braking candidate.</div>
          </div>
        </div>
      </div>
      <div class="nav">
        <a id="back-link" href="index.html">Back To Table</a>
        <button id="focus-target" type="button">Focus Target</button>
        <a id="prev-link" href="#">Previous</a>
        <a id="next-link" href="#">Next</a>
      </div>
      <div id="compliance-card" class="avoidability-result compliance-card reference">
        <strong>Compliance unavailable</strong>
        <div id="compliance-detail"></div>
        <div id="compliance-jumps" class="compliance-jumps"></div>
      </div>
      <div class="meta">
        <div class="meta-item"><span class="meta-label">Focused Speed</span><span class="meta-value" id="focus-speed">n/a</span></div>
        <div class="meta-item"><span class="meta-label">Did Target Collide</span><span class="meta-value" id="meta-collide"></span></div>
        <div class="meta-item"><span class="meta-label">Did Target Offroad</span><span class="meta-value" id="meta-offroad"></span></div>
        <div class="meta-item"><span class="meta-label">Did Target Run Red Light</span><span class="meta-value" id="meta-run-light"></span></div>
        <div class="meta-item"><span class="meta-label">At-Fault Collision</span><span class="meta-value" id="meta-at-fault"></span></div>
        <div class="meta-item"><span class="meta-label">Collision Responsibility</span><span class="meta-value" id="meta-collision-responsibility"></span></div>
        <div class="meta-item"><span class="meta-label">last_t_brake</span><span class="meta-value" id="meta-last-t-brake"></span></div>
        <div class="meta-item"><span class="meta-label">Detection Window</span><span class="meta-value" id="meta-reaction-window"></span></div>
        <div class="meta-item"><span class="meta-label">Collision Outcome</span><span class="meta-value" id="meta-collision-outcome"></span></div>
        <div class="meta-item"><span class="meta-label">Collision Severity</span><span class="meta-value" id="meta-collision-severity"></span></div>
        <div class="meta-item"><span class="meta-label">Map</span><span class="meta-value" id="meta-map"></span></div>
        <div class="meta-item"><span class="meta-label">Episode</span><span class="meta-value" id="meta-episode"></span></div>
        <div class="meta-item"><span class="meta-label">Scenario</span><span class="meta-value" id="meta-scenario"></span></div>
        <div class="meta-item"><span class="meta-label">Episode Length</span><span class="meta-value" id="meta-length"></span></div>
        <div class="meta-item"><span class="meta-label">Impact Zone</span><span class="meta-value" id="meta-impact-zone"></span></div>
        <div class="meta-item"><span class="meta-label">Made Progress</span><span class="meta-value" id="meta-made-progress"></span></div>
        <div class="meta-item"><span class="meta-label">Goals Reached</span><span class="meta-value" id="meta-goals"></span></div>
        <div class="meta-item"><span class="meta-label">TTC Within Bound</span><span class="meta-value" id="meta-ttc"></span></div>
        <div class="meta-item"><span class="meta-label">Progress Ratio</span><span class="meta-value" id="meta-progress-ratio"></span></div>
        <div class="meta-item"><span class="meta-label">Puffer Score</span><span class="meta-value" id="meta-puffer-score"></span></div>
        <div class="meta-item"><span class="meta-label">Focused Agent</span><span class="meta-value" id="focus-agent">none</span></div>
        <div class="meta-item"><span class="meta-label">Focused Velocity</span><span class="meta-value" id="focus-velocity">n/a</span></div>
      </div>
      <div class="legend">
        <div class="legend-row"><span class="swatch" style="background: var(--target)"></span>Target</div>
        <div class="legend-row"><span class="swatch" style="background: var(--active)"></span>Active adversary</div>
        <div class="legend-row"><span class="swatch" style="background: var(--inactive)"></span>Inactive / static</div>
        <div class="legend-row"><span class="swatch" style="background: var(--stopped)"></span>Stopped / crashed</div>
        <div class="legend-row"><span class="swatch" style="background: #22d3ee"></span>Velocity vector</div>
        <div class="legend-row"><span class="swatch" style="background: #f59e0b"></span>Pre-braking response (legacy traces)</div>
        <div class="legend-row"><span class="swatch" style="background: #fb7185"></span>Target maximum braking</div>
        <div class="legend-row"><span class="swatch" style="background: #22d3ee"></span>Adversary projected trajectory</div>
        <div class="legend-row"><span class="swatch" style="background: #fb7185"></span>Target stopped / impact pose</div>
        <div class="legend-row"><span class="swatch" style="background: #22d3ee"></span>Adversary rollout / impact pose</div>
        <div class="legend-row"><span class="swatch" style="background: #c084fc"></span>Impact poses (secondary blocker)</div>
        <div class="legend-row"><span class="swatch" style="background: #fb7185"></span>Hitter compliance trajectory</div>
        <div class="legend-row"><span class="swatch" style="background: #ff2d55"></span>Crossed solid line / movement</div>
      </div>
    </aside>
    <main class="panel viewer">
      <div class="viewer-head">
        <div>
          <h2 class="viewer-title">Compact Replay</h2>
          <p class="viewer-subtitle">Lightweight mining viewer</p>
        </div>
        <div class="viewer-tools">
          <button id="velocity-toggle" type="button">Hide Vectors</button>
          <button id="reset-view" type="button">Reset View</button>
          <div id="status-pill" class="pill">Target failed</div>
        </div>
      </div>
      <div id="detector-hud" class="detector-hud" hidden>
        <div id="buffer-badge" class="detector-badge"></div>
        <div id="ttc-badge" class="detector-badge"></div>
      </div>
      <canvas id="scene"></canvas>
    </main>
    <aside class="panel episode-list">
      <h3>Replay Episodes</h3>
      <div class="episode-scroll" id="episode-list"></div>
    </aside>
  </div>
  <script>
    const DATA = __DATA__;
    const canvas = document.getElementById('scene');
    const ctx = canvas.getContext('2d');
    const slider = document.getElementById('frame-slider');
    const frameLabel = document.getElementById('frame-label');
    const playToggle = document.getElementById('play-toggle');
    const speedSlider = document.getElementById('speed-slider');
    const speedLabel = document.getElementById('speed-label');
    const statusPill = document.getElementById('status-pill');
    const backLink = document.getElementById('back-link');
    const prevLink = document.getElementById('prev-link');
    const nextLink = document.getElementById('next-link');
    const episodeList = document.getElementById('episode-list');
    const focusTargetButton = document.getElementById('focus-target');
    const resetViewButton = document.getElementById('reset-view');
    const velocityToggleButton = document.getElementById('velocity-toggle');
    const observedModeButton = document.getElementById('observed-mode');
    const avoidabilityModeButton = document.getElementById('avoidability-mode');
    const observedControls = document.getElementById('observed-controls');
    const avoidabilityControls = document.getElementById('avoidability-controls');
    const detectionPhaseButton = document.getElementById('detection-phase');
    const brakingPhaseButton = document.getElementById('braking-phase');
    const detectionControls = document.getElementById('detection-controls');
    const brakingControls = document.getElementById('braking-controls');
    const detectionPlay = document.getElementById('detection-play');
    const detectionSlider = document.getElementById('detection-slider');
    const detectionLabel = document.getElementById('detection-label');
    const detectionResult = document.getElementById('detection-result');
    const detectionMarkers = document.getElementById('detection-markers');
    const reactionSlider = document.getElementById('reaction-slider');
    const reactionLabel = document.getElementById('reaction-label');
    const avoidabilityResult = document.getElementById('avoidability-result');
    const rolloutPlay = document.getElementById('rollout-play');
    const rolloutSlider = document.getElementById('rollout-slider');
    const rolloutLabel = document.getElementById('rollout-label');
    const rolloutMarkers = document.getElementById('rollout-markers');
    const bufferToggle = document.getElementById('buffer-toggle');
    const ttcToggle = document.getElementById('ttc-toggle');
    const detectorHud = document.getElementById('detector-hud');
    const bufferBadge = document.getElementById('buffer-badge');
    const ttcBadge = document.getElementById('ttc-badge');
    const complianceCard = document.getElementById('compliance-card');
    const complianceDetail = document.getElementById('compliance-detail');
    const complianceJumps = document.getElementById('compliance-jumps');

    const metadata = DATA.metadata || {};
    const summary = DATA.summary || {};
    const navigation = DATA.navigation || {};
    const frames = DATA.agent_frames || [];
    const trafficFrames = DATA.traffic_frames || [];
    const episodeTimesteps = DATA.episode_timesteps || [];
    const avoidability = DATA.avoidability_debug || null;
    const compliance = DATA.compliance_diagnostics || null;
    const candidates = (avoidability && avoidability.candidate_arrays) || {};
    const candidateSteps = candidates.steps_back || [];
    const hasAvoidability = !!(avoidability && avoidability.collision && candidateSteps.length);
    const roadElements = (DATA.map && DATA.map.road_elements) || [];
    const bounds = DATA.bounds || [-100, -100, 100, 100];

    let frameIndex = 0;
    let playing = false;
    let lastTimestamp = 0;
    let speed = 1.0;
    let camera = null;
    let dragState = null;
    let hitAgents = [];
    let followTarget = false;
    let selectedAgentId = null;
    let showVelocityVectors = true;
    let replayMode = 'observed';
    let reactionSelection = 0;
    let observedFrameIndex = 0;
    let showLateralBuffer = true;
    let showTTC = true;
    let rolloutStep = 0;
    let rolloutPlaying = false;
    let rolloutLastTimestamp = 0;
    let avoidabilityPhase = 'detection';
    let detectionSelection = 0;
    let detectionPlaying = false;
    let detectionLastTimestamp = 0;
    let detectionSamplesCache = null;


    function summaryValue(key, fallback=null) {
      if (summary[key] != null) return summary[key];
      if (metadata[key] != null) return metadata[key];
      return fallback;
    }

    function metricEnabled(item, key) {
      return Number((item && item[key]) || 0) > 0;
    }

    function atFaultEnabled(item) {
      return metricEnabled(item, 'did_target_have_at_fault_collision') || metricEnabled(item, 'target_hit_at_fault_rate');
    }

    function compareNavigationValues(a, b, key, dir) {
      const av = a[key];
      const bv = b[key];
      if (av === bv) return 0;
      if (av == null || av === '') return 1;
      if (bv == null || bv === '') return -1;

      const an = Number(av);
      const bn = Number(bv);
      if (!Number.isNaN(an) && !Number.isNaN(bn)) {
        if (an === bn) return 0;
        return an > bn ? dir : -dir;
      }

      return String(av).toLowerCase().localeCompare(String(bv).toLowerCase()) * dir;
    }

    function navigationState() {
      const params = new URLSearchParams(window.location.search);
      return {
        sortKey: params.get('sort') || 'did_target_fail',
        sortDir: Number(params.get('dir') || '-1') >= 0 ? 1 : -1,
        replayOnly: params.get('replay') === '1',
        failuresOnly: params.get('failures') === '1',
        outcomeOnly: params.get('outcome') || '',
        offroadOnly: params.get('offroad') === '1',
        atFaultOnly: params.get('atfault') === '1',
        search: (params.get('q') || '').toLowerCase(),
      };
    }

    function stateQuery(state) {
      const params = new URLSearchParams();
      params.set('sort', state.sortKey);
      params.set('dir', String(state.sortDir));
      if (state.replayOnly) params.set('replay', '1');
      if (state.failuresOnly) params.set('failures', '1');
      if (state.outcomeOnly) params.set('outcome', state.outcomeOnly);
      if (state.offroadOnly) params.set('offroad', '1');
      if (state.atFaultOnly) params.set('atfault', '1');
      if (state.search) params.set('q', state.search);
      const query = params.toString();
      return query ? `?${query}` : '';
    }

    function hrefWithState(href, state) {
      if (!href || href === '#') return '#';
      return `${href}${stateQuery(state)}`;
    }

    function orderedNavigationEpisodes() {
      const state = navigationState();
      const items = (navigation.episodes || []).filter(item => {
        if (state.replayOnly && !item.href) return false;
        if (state.failuresOnly && !metricEnabled(item, 'did_target_fail')) return false;
        if (state.outcomeOnly === 'target_failure' &&
            !metricEnabled(item, 'target_collision_target_failure_rate')) return false;
        if (state.outcomeOnly === 'unavoidable' &&
            !metricEnabled(item, 'target_collision_unavoidable_rate')) return false;
        if (state.outcomeOnly === 'adversary_forced' &&
            !metricEnabled(item, 'target_collision_adversary_forced_rate')) return false;
        if (state.offroadOnly && !metricEnabled(item, 'did_target_offroad')) return false;
        if (state.atFaultOnly && !atFaultEnabled(item)) return false;
        if (state.search && !JSON.stringify(item).toLowerCase().includes(state.search)) return false;
        return true;
      });
      items.sort((a, b) => compareNavigationValues(a, b, state.sortKey, state.sortDir));
      return {items, state};
    }

    function formatMetric(value, digits=3) {
      const num = Number(value);
      if (!Number.isFinite(num)) return 'n/a';
      return num.toFixed(digits);
    }

    function formatSeconds(value) {
      const num = Number(value);
      return Number.isFinite(num) && num >= 0 ? `${num.toFixed(2)} s` : 'n/a';
    }

    function advDriveWeightValue() {
      return summaryValue('adv_reward_weight_drive', null);
    }

    function formatAdvDriveWeight(value) {
      const num = Number(value);
      return Number.isFinite(num) ? num.toFixed(3) : 'n/a';
    }

    function createDefaultCamera() {
      const minX = bounds[0], minY = bounds[1], maxX = bounds[2], maxY = bounds[3];
      return {
        x: (minX + maxX) / 2,
        y: (minY + maxY) / 2,
        zoom: 1.0,
      };
    }

    function setMeta() {
      document.getElementById('title').innerText = metadata.map_name || 'Replay';
      document.getElementById('subtitle').innerText = `dynamics=${metadata.dynamics_model || 'unknown'} | target=${metadata.target_type || 'unknown'}`;
      document.getElementById('meta-episode').innerText = metadata.episode_id ?? 'N/A';
      document.getElementById('meta-adv-drive-weight').innerText = formatAdvDriveWeight(advDriveWeightValue());
      document.getElementById('meta-map').innerText = metadata.map_name || 'N/A';
      document.getElementById('meta-scenario').innerText = metadata.scenario_id || 'N/A';
      document.getElementById('meta-length').innerText = metadata.episode_length ?? frames.length;
      document.getElementById('meta-collide').innerText = Number(summaryValue('did_target_collide', 0) || 0) > 0 ? 'yes' : 'no';
      document.getElementById('meta-offroad').innerText = Number(summaryValue('did_target_offroad', 0) || 0) > 0 ? 'yes' : 'no';
      document.getElementById('meta-run-light').innerText = Number(summaryValue('did_target_run_light', 0) || 0) > 0 ? 'yes' : 'no';
      document.getElementById('meta-impact-zone').innerText = summaryValue('target_collision_impact_zone_label', 'none');
      document.getElementById('meta-collision-severity').innerText = formatMetric(summaryValue('target_collision_severity', 0));
      document.getElementById('meta-collision-responsibility').innerText = formatMetric(summaryValue('target_collision_responsibility', 0));
      const classification = (avoidability && avoidability.classification) || {};
      const tBrake = Number(classification.t_brake);
      document.getElementById('meta-last-t-brake').innerText = formatSeconds(tBrake);
      const reactionCheck = computeReactionWindowDanger();
      document.getElementById('meta-reaction-window').innerText = reactionCheck.minimum == null
        ? 'n/a'
        : `${reactionCheck.maximumBeforeBraking.toFixed(2)} → ${reactionCheck.minimumBeforeBraking.toFixed(2)}s before brake · ` +
          `${reactionCheck.available ? (reactionCheck.dangerous ? 'danger' : 'safe') : 'not sampled'} · ` +
          `${reactionCheck.samples} sample(s)`;
      let collisionOutcome = 'not classified';
      if (Number(classification.unavoidable || 0) > 0) collisionOutcome = 'unavoidable';
      else if (Number(classification.genuine_target_failure || 0) > 0) collisionOutcome = 'genuine target failure';
      else if (Number(classification.adversary_forced || 0) > 0) collisionOutcome = 'adversary-forced';
      document.getElementById('meta-collision-outcome').innerText = collisionOutcome;
      document.getElementById('meta-made-progress').innerText = Number(summaryValue('did_target_make_progress', 0) || 0) > 0 ? 'yes' : 'no';
      document.getElementById('meta-at-fault').innerText = Number(summaryValue('did_target_have_at_fault_collision', 0) || 0) > 0 ? 'yes' : 'no';
      document.getElementById('meta-goals').innerText = String(summaryValue('target_num_goals_reached', 0));
      document.getElementById('meta-ttc').innerText = formatMetric(summaryValue('target_ttc_within_bound_rate', 0));
      document.getElementById('meta-progress-ratio').innerText = formatMetric(summaryValue('target_progress_ratio', 0));
      document.getElementById('meta-puffer-score').innerText = formatMetric(summaryValue('target_puffer_score', 0));
      const failed = Number(summaryValue('did_target_fail', 0) || 0) > 0;
      statusPill.className = failed ? 'pill' : 'pill ok';
      statusPill.innerText = failed ? 'Target failed' : 'Target survived';
      const nav = orderedNavigationEpisodes();
      const items = nav.items;
      const activeIdx = items.findIndex(item => item.episode_id === metadata.episode_id);
      const prevHref = activeIdx > 0 ? items[activeIdx - 1].href : null;
      const nextHref = activeIdx >= 0 && activeIdx + 1 < items.length ? items[activeIdx + 1].href : null;
      backLink.href = hrefWithState(navigation.index_html || 'index.html', nav.state);
      prevLink.href = hrefWithState(prevHref, nav.state);
      nextLink.href = hrefWithState(nextHref, nav.state);
      prevLink.style.pointerEvents = prevHref ? 'auto' : 'none';
      nextLink.style.pointerEvents = nextHref ? 'auto' : 'none';
      prevLink.style.opacity = prevHref ? '1' : '0.4';
      nextLink.style.opacity = nextHref ? '1' : '0.4';
      episodeList.innerHTML = items.map(item => {
        const active = item.episode_id === metadata.episode_id ? 'episode-link active' : 'episode-link';
        const badge = item.did_target_fail ? '<span class="badge fail">fail</span>' : '<span class="badge">ok</span>';
        const driveWeight = item.adv_reward_weight_drive;
        const driveWeightText = driveWeight == null ? '' : ` | drive=${formatAdvDriveWeight(driveWeight)}`;
        return `<a class="${active}" href="${hrefWithState(item.href, nav.state)}">${badge}<strong>Episode ${item.episode_id}${driveWeightText}</strong><small>${item.map_name || ''} | ${item.scenario_id || ''}</small></a>`;
      }).join('');
      setComplianceCard();
    }

    function complianceReasonLabel(key) {
      return {
        red_light: 'red light',
        wrong_way: 'wrong way',
        solid_line: 'solid line',
        speed_limit: 'speed limit',
      }[key];
    }

    function setComplianceCard() {
      if (!compliance || !Number(compliance.valid || 0)) return;
      const reasons = ['red_light', 'wrong_way', 'solid_line', 'speed_limit'].filter(
        reason => Number(compliance[`${reason}_violation`] || 0) > 0
      );
      const compliant = Number(compliance.compliant || 0) > 0;
      complianceCard.className = compliant
        ? 'avoidability-result compliance-card'
        : 'avoidability-result compliance-card failed';
      complianceCard.querySelector('strong').innerText = compliant ? 'Hitter compliant' : 'Hitter non-compliant';
      const laneMissing = Number(compliance.lane_unavailable_sample_count || 0);
      const speedMissing = Number(compliance.speed_limit_unavailable_sample_count || 0);
      const redMissing = Number(compliance.red_light_unavailable_sample_count || 0);
      const coverage = [
        laneMissing ? `lane unavailable ${laneMissing}` : null,
        speedMissing ? `limit unavailable ${speedMissing}` : null,
        redMissing ? `red unavailable ${redMissing}` : null,
      ].filter(Boolean).join(' · ') || 'coverage complete';
      const crossedLine = Number(compliance.solid_line_violation || 0) > 0
        ? ` · line ${Number(compliance.crossed_line_index)}:${Number(compliance.crossed_line_segment_index)}`
        : '';
      complianceDetail.innerText =
        `${reasons.length ? reasons.map(complianceReasonLabel).join(', ') : 'no violations'} · ` +
        `wrong-way ${Number(compliance.wrong_way_distance || 0).toFixed(2)}m · ` +
        `max speed ${(100 * Number(compliance.max_speed_ratio || 0)).toFixed(1)}% · ` +
        `${coverage}${crossedLine} · ${compliance.source || 'simulator'}`;
      complianceJumps.innerHTML = reasons.map(reason => {
        const timestep = Number(compliance[`first_${reason}_timestep`]);
        return timestep >= 0
          ? `<button type="button" data-compliance-timestep="${timestep}">${complianceReasonLabel(reason)} @ ${timestep}</button>`
          : '';
      }).join('');
      complianceJumps.querySelectorAll('button').forEach(button => {
        button.addEventListener('click', () => {
          if (replayMode !== 'observed') setReplayMode('observed');
          frameIndex = nearestFrameAtOrBefore(Number(button.dataset.complianceTimestep));
          observedFrameIndex = frameIndex;
          playing = false;
          playToggle.innerText = 'Play';
          draw();
        });
      });
    }

    function resizeCanvas() {
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }

    function worldToCanvas(x, y) {
      const minX = bounds[0], minY = bounds[1], maxX = bounds[2], maxY = bounds[3];
      const pad = 22;
      const usableW = canvas.clientWidth - pad * 2;
      const usableH = canvas.clientHeight - pad * 2;
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      const baseScale = Math.min(usableW / spanX, usableH / spanY);
      const scale = baseScale * (camera ? camera.zoom : 1.0);
      const centerX = canvas.clientWidth / 2;
      const centerY = canvas.clientHeight / 2;
      return {
        x: centerX + (x - (camera ? camera.x : (minX + maxX) / 2)) * scale,
        y: centerY - (y - (camera ? camera.y : (minY + maxY) / 2)) * scale,
        scale,
      };
    }

    function canvasToWorld(x, y) {
      const minX = bounds[0], minY = bounds[1], maxX = bounds[2], maxY = bounds[3];
      const pad = 22;
      const usableW = canvas.clientWidth - pad * 2;
      const usableH = canvas.clientHeight - pad * 2;
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      const baseScale = Math.min(usableW / spanX, usableH / spanY);
      const scale = baseScale * (camera ? camera.zoom : 1.0);
      const centerX = canvas.clientWidth / 2;
      const centerY = canvas.clientHeight / 2;
      return {
        x: (x - centerX) / scale + camera.x,
        y: (centerY - y) / scale + camera.y,
      };
    }

    function findTarget(frame) {
      return (frame || []).find(agent => agent.is_target);
    }

    function speedOfAgent(agent) {
      return Math.hypot(Number(agent.vx || 0), Number(agent.vy || 0));
    }

    function findAgentById(frame, id) {
      return (frame || []).find(agent => agent.id === id);
    }

    function nearestFrameAtOrBefore(timestep) {
      let bestIndex = 0;
      for (let i = 0; i < episodeTimesteps.length; i++) {
        if (Number(episodeTimesteps[i]) > Number(timestep)) break;
        bestIndex = i;
      }
      return Math.min(bestIndex, Math.max(frames.length - 1, 0));
    }

    function agentFromSnapshot(snapshot, isTarget) {
      if (!snapshot || !snapshot.valid) return null;
      return {
        id: Number(snapshot.id),
        type: Number(snapshot.type),
        is_target: !!isTarget,
        active: !!snapshot.active,
        stopped: !!snapshot.stopped,
        x: Number(snapshot.x),
        y: Number(snapshot.y),
        z: Number(snapshot.z),
        heading: Number(snapshot.heading),
        length: Number(snapshot.length),
        width: Number(snapshot.width),
        height: Number(snapshot.height),
        vx: Number(snapshot.vx),
        vy: Number(snapshot.vy),
      };
    }

    function collisionReferenceFrame() {
      const collision = avoidability.collision;
      const base = [...(frames[nearestFrameAtOrBefore(collision.collision_timestep)] || [])];
      const snapshots = [agentFromSnapshot(collision.target, true), agentFromSnapshot(collision.adversary, false)];
      for (const snapshot of snapshots) {
        if (!snapshot) continue;
        const existingIndex = base.findIndex(agent => Number(agent.id) === Number(snapshot.id));
        if (existingIndex >= 0) base[existingIndex] = snapshot;
        else base.push(snapshot);
      }
      return base;
    }

    function counterfactualDisplayFrame() {
      const trajectories = selectedCounterfactualTrajectories();
      if (!trajectories) return frames[frameIndex] || [];
      const step = Math.min(rolloutStep, trajectories.effectiveEndStep);
      const base = [...(frames[nearestFrameAtOrBefore(trajectories.startTimestep)] || [])];
      const replacements = [trajectories.targetSamples[step], trajectories.adversarySamples[step]];
      for (const agent of replacements) {
        if (!agent) continue;
        const existingIndex = base.findIndex(item => Number(item.id) === Number(agent.id));
        if (existingIndex >= 0) base[existingIndex] = agent;
        else base.push(agent);
      }
      if (trajectories.impact && step === trajectories.effectiveEndStep && !trajectories.impact.blockerIsOriginal) {
        const blocker = candidateValue('blocking_agent', trajectories.candidateIndex, null);
        const blockerAgent = agentFromSnapshot(blocker, false);
        if (blockerAgent) {
          blockerAgent.counterfactual = true;
          const existingIndex = base.findIndex(item => Number(item.id) === Number(blockerAgent.id));
          if (existingIndex >= 0) base[existingIndex] = blockerAgent;
          else base.push(blockerAgent);
        }
      }
      return base;
    }

    function currentDisplayFrame() {
      if (replayMode === 'avoidability' && avoidabilityPhase === 'detection') {
        const sample = selectedDetectionWindowSample();
        return sample ? frames[sample.frameIndex] || [] : collisionReferenceFrame();
      }
      if (replayMode === 'avoidability' && reactionSelection === 0) return collisionReferenceFrame();
      if (replayMode === 'avoidability') return counterfactualDisplayFrame();
      return frames[frameIndex] || [];
    }

    function observedAgentRail(agentId, startTimestep, collisionTimestep, collisionSnapshot) {
      const points = [];
      for (let i = 0; i < frames.length; i++) {
        const timestep = Number(episodeTimesteps[i]);
        if (timestep < startTimestep || timestep >= collisionTimestep) continue;
        const agent = findAgentById(frames[i], agentId);
        if (agent) points.push({
          x: Number(agent.x),
          y: Number(agent.y),
          z: Number(agent.z || 0),
          heading: Number(agent.heading || 0),
        });
      }
      if (collisionSnapshot && collisionSnapshot.valid) {
        points.push({
          x: Number(collisionSnapshot.x),
          y: Number(collisionSnapshot.y),
          z: Number(collisionSnapshot.z || 0),
          heading: Number(collisionSnapshot.heading || 0),
        });
      }
      return points;
    }

    function sampleRail(points, distance) {
      if (!points.length) return null;
      if (points.length === 1) {
        const heading = Number(points[0].heading || 0);
        return {
          x: points[0].x + distance * Math.cos(heading),
          y: points[0].y + distance * Math.sin(heading),
          z: Number(points[0].z || 0),
          heading,
        };
      }
      let traveled = 0;
      for (let i = 0; i < points.length - 1; i++) {
        const dx = points[i + 1].x - points[i].x;
        const dy = points[i + 1].y - points[i].y;
        const segmentLength = Math.hypot(dx, dy);
        if (traveled + segmentLength >= distance) {
          const ratio = segmentLength > 1e-6 ? Math.max(0, Math.min(1, (distance - traveled) / segmentLength)) : 0;
          return {
            x: points[i].x + ratio * dx,
            y: points[i].y + ratio * dy,
            z: Number(points[i].z || 0) + ratio * (Number(points[i + 1].z || 0) - Number(points[i].z || 0)),
            heading: segmentLength > 1e-6 ? Math.atan2(dy, dx) : Number(points[i].heading || 0),
          };
        }
        traveled += segmentLength;
      }
      const last = points.length - 1;
      const dx = points[last].x - points[last - 1].x;
      const dy = points[last].y - points[last - 1].y;
      const segmentLength = Math.hypot(dx, dy);
      const heading = segmentLength > 1e-6 ? Math.atan2(dy, dx) : Number(points[last].heading || 0);
      const overshoot = Math.max(0, distance - traveled);
      return {
        x: points[last].x + overshoot * Math.cos(heading),
        y: points[last].y + overshoot * Math.sin(heading),
        z: Number(points[last].z || 0),
        heading,
      };
    }

    function appendDistinct(points, point) {
      if (!point) return;
      const previous = points[points.length - 1];
      if (!previous || Math.hypot(point.x - previous.x, point.y - previous.y) > 1e-6) points.push(point);
    }

    function firstRolloutSampleAtOrAfter(time, dt) {
      return Math.ceil(Math.max(0, time) / dt - 1e-9) * dt;
    }

    function agentAtTimestep(agentId, timestep) {
      for (let i = 0; i < episodeTimesteps.length; i++) {
        if (Number(episodeTimesteps[i]) !== Number(timestep)) continue;
        return findAgentById(frames[i], agentId);
      }
      return null;
    }

    function signedSpeedOfAgent(agent) {
      const speed = speedOfAgent(agent);
      const heading = Number(agent.heading || 0);
      const alongHeading = Number(agent.vx || 0) * Math.cos(heading) + Number(agent.vy || 0) * Math.sin(heading);
      return alongHeading < 0 ? -speed : speed;
    }

    function counterfactualAgent(base, point, signedSpeed, isTarget) {
      const heading = Number(point.heading || 0);
      return {
        ...base,
        id: Number(base.id),
        is_target: isTarget,
        x: Number(point.x),
        y: Number(point.y),
        z: Number(point.z == null ? base.z || 0 : point.z),
        heading,
        vx: signedSpeed * Math.cos(heading),
        vy: signedSpeed * Math.sin(heading),
        stopped: Math.abs(signedSpeed) <= 1e-6,
        counterfactual: true,
      };
    }

    function selectedCounterfactualTrajectories() {
      if (replayMode !== 'avoidability' || avoidabilityPhase !== 'braking' ||
          reactionSelection === 0 || !hasAvoidability) return null;
      const candidateIndex = reactionSelection - 1;
      const stepsBack = Number(candidateValue('steps_back', candidateIndex, 0));
      const collision = avoidability.collision;
      const constants = avoidability.constants || {};
      const dt = Number(constants.dt || 0.1);
      const deceleration = Number(constants.braking_deceleration || 5.0);
      const reactionTime = Number(constants.reaction_time_seconds || 1.0);
      // New traces brake immediately and use reactionTime only for classification.
      // Traces captured before this field was added retain the delayed rollout.
      const brakingDelay = Object.prototype.hasOwnProperty.call(constants, 'braking_rollout_delay_seconds')
        ? Number(constants.braking_rollout_delay_seconds) : reactionTime;
      const maxRolloutSteps = Number(constants.max_rollout_steps || 91);
      const collisionTimestep = Number(collision.collision_timestep);
      const startTimestep = collisionTimestep - stepsBack;
      const targetId = Number(collision.target_agent_index);
      const adversaryId = Number(collision.collision_adversary_index);
      const startFrame = frames[nearestFrameAtOrBefore(startTimestep)] || [];
      const targetAtStart = findAgentById(startFrame, targetId);
      if (!targetAtStart) return null;

      const targetRail = observedAgentRail(targetId, startTimestep, collisionTimestep, collision.target);
      const initialSpeed = speedOfAgent(targetAtStart);
      const stopTime = initialSpeed <= 0 ? 0 : brakingDelay + initialSpeed / Math.max(deceleration, 1e-6);
      const targetStopSampleTime = firstRolloutSampleAtOrAfter(stopTime, dt);
      const targetStopStep = Math.round(targetStopSampleTime / dt);
      const observedCollisionTime = stepsBack * dt;
      const adversary = collision.adversary;
      const adversaryHeading = Number(adversary.heading || 0);
      const adversarySignedSpeed = signedSpeedOfAgent(adversary);
      const adversaryStopTime = Math.abs(adversarySignedSpeed) / Math.max(deceleration, 1e-6);
      const rolloutUntilAdversaryStop = Number(constants.rollout_until_adversary_stop || 0) > 0;
      const adversaryStopSampleTime = observedCollisionTime + firstRolloutSampleAtOrAfter(adversaryStopTime, dt);
      const adversaryStopStep = Math.round(adversaryStopSampleTime / dt);
      const fullEndStep = Math.min(
        maxRolloutSteps - 1,
        Math.max(targetStopStep, rolloutUntilAdversaryStop ? adversaryStopStep : stepsBack)
      );
      const blockingRolloutStep = Number(candidateValue('blocking_rollout_step', candidateIndex, -1));
      const effectiveEndStep = blockingRolloutStep >= 0 ? Math.min(blockingRolloutStep, fullEndStep) : fullEndStep;
      const rolloutEndTime = effectiveEndStep * dt;

      const targetPointAt = tau => {
        const motionTime = Math.min(Math.max(0, tau), stopTime);
        const brakingTime = Math.max(0, motionTime - brakingDelay);
        const distance = initialSpeed * motionTime - 0.5 * deceleration * brakingTime * brakingTime;
        return sampleRail(targetRail, Math.max(0, distance));
      };

      const targetSamples = [];
      const adversarySamples = [];
      for (let step = 0; step <= effectiveEndStep; step++) {
        const tau = step * dt;
        const targetPoint = targetPointAt(tau);
        const brakingTime = Math.max(0, Math.min(tau, stopTime) - brakingDelay);
        const targetSpeed = Math.max(0, initialSpeed - deceleration * brakingTime);
        targetSamples.push({
          ...counterfactualAgent(collision.target, targetPoint, targetSpeed, true),
          braking: tau >= brakingDelay - 1e-6 && targetSpeed > 1e-6,
        });

        if (step < stepsBack) {
          const recorded = agentAtTimestep(adversaryId, startTimestep + step);
          if (recorded) {
            adversarySamples.push({...recorded, counterfactual: true, braking: false});
            continue;
          }
        }
        const extensionTime = Math.min(Math.max(0, step - stepsBack) * dt, adversaryStopTime);
        const direction = adversarySignedSpeed < 0 ? -1 : 1;
        const distance = adversarySignedSpeed * extensionTime -
          direction * 0.5 * deceleration * extensionTime * extensionTime;
        const extensionSpeed = direction * Math.max(0, Math.abs(adversarySignedSpeed) - deceleration * extensionTime);
        adversarySamples.push({
          ...counterfactualAgent(adversary, {
            x: Number(adversary.x) + distance * Math.cos(adversaryHeading),
            y: Number(adversary.y) + distance * Math.sin(adversaryHeading),
            z: Number(adversary.z || 0),
            heading: adversaryHeading,
          }, extensionSpeed, false),
          braking: step >= stepsBack && Math.abs(extensionSpeed) > 1e-6,
        });
      }

      const blockingSnapshot = candidateValue('blocking_agent', candidateIndex, null);
      let impact = null;
      if (blockingRolloutStep >= 0 && blockingSnapshot && blockingSnapshot.valid) {
        const impactTime = blockingRolloutStep * dt;
        const targetImpact = targetSamples[targetSamples.length - 1];
        impact = {
          time: impactTime,
          target: targetImpact,
          blocker: {
            x: Number(blockingSnapshot.x),
            y: Number(blockingSnapshot.y),
            heading: Number(blockingSnapshot.heading || 0),
          },
          blockerDimensions: {
            length: Number(blockingSnapshot.length),
            width: Number(blockingSnapshot.width),
          },
          blockerId: Number(blockingSnapshot.index),
          blockerIsOriginal: Number(blockingSnapshot.index) === adversaryId,
        };
      }

      return {
        candidateIndex,
        stepsBack,
        dt,
        startTimestep,
        targetSamples,
        adversarySamples,
        targetStop: targetSamples[Math.min(targetStopStep, effectiveEndStep)],
        adversaryAtRolloutEnd: adversarySamples[adversarySamples.length - 1] || null,
        brakingStart: targetSamples[0],
        reactionTime: brakingDelay,
        targetStopSampleTime,
        targetStopStep,
        adversaryStopSampleTime,
        adversaryStopStep,
        observedCollisionTime,
        observedCollisionStep: stepsBack,
        fullEndStep,
        effectiveEndStep,
        rolloutEndTime,
        impact,
        exitedEarly: blockingRolloutStep >= 0,
        targetReachedStop: effectiveEndStep >= targetStopStep,
        adversaryReachedStop: effectiveEndStep >= adversaryStopStep,
        targetDimensions: {length: Number(collision.target.length), width: Number(collision.target.width)},
        adversaryDimensions: {length: Number(adversary.length), width: Number(adversary.width)},
      };
    }

    function candidateValue(key, index, fallback=null) {
      const values = candidates[key] || [];
      return values[index] == null ? fallback : values[index];
    }

    function detectorPair() {
      if (!hasAvoidability || replayMode !== 'avoidability') return null;
      return detectorPairForFrame(currentDisplayFrame());
    }

    function detectorPairForFrame(frame) {
      if (!avoidability || !avoidability.collision) return null;
      const collision = avoidability.collision;
      const target = findAgentById(frame, Number(collision.target_agent_index));
      const adversary = findAgentById(frame, Number(collision.collision_adversary_index));
      return target && adversary ? {target, adversary} : null;
    }

    function obbCorners(agent) {
      const heading = Number(agent.heading || 0);
      const forwardX = Math.cos(heading), forwardY = Math.sin(heading);
      const leftX = -forwardY, leftY = forwardX;
      const halfLength = Number(agent.length || 0) / 2;
      const halfWidth = Number(agent.width || 0) / 2;
      return [
        [agent.x + halfLength * forwardX + halfWidth * leftX, agent.y + halfLength * forwardY + halfWidth * leftY],
        [agent.x + halfLength * forwardX - halfWidth * leftX, agent.y + halfLength * forwardY - halfWidth * leftY],
        [agent.x - halfLength * forwardX - halfWidth * leftX, agent.y - halfLength * forwardY - halfWidth * leftY],
        [agent.x - halfLength * forwardX + halfWidth * leftX, agent.y - halfLength * forwardY + halfWidth * leftY],
      ];
    }

    function obbOverlap(agentA, agentB) {
      const cornersA = obbCorners(agentA);
      const cornersB = obbCorners(agentB);
      const headings = [Number(agentA.heading || 0), Number(agentB.heading || 0)];
      const axes = [];
      for (const heading of headings) {
        axes.push([Math.cos(heading), Math.sin(heading)]);
        axes.push([-Math.sin(heading), Math.cos(heading)]);
      }
      for (const [axisX, axisY] of axes) {
        const projectionsA = cornersA.map(([x, y]) => x * axisX + y * axisY);
        const projectionsB = cornersB.map(([x, y]) => x * axisX + y * axisY);
        const minA = Math.min(...projectionsA), maxA = Math.max(...projectionsA);
        const minB = Math.min(...projectionsB), maxB = Math.max(...projectionsB);
        if (maxA < minB || minA > maxB) return false;
      }
      return true;
    }

    function verticalOverlap(agentA, agentB) {
      const bottomA = Number(agentA.z || 0), topA = bottomA + Number(agentA.height || 0);
      const bottomB = Number(agentB.z || 0), topB = bottomB + Number(agentB.height || 0);
      return !(topA < bottomB || topB < bottomA);
    }

    function projectedPose(agent, time) {
      return {
        ...agent,
        x: Number(agent.x) + Number(agent.vx || 0) * time,
        y: Number(agent.y) + Number(agent.vy || 0) * time,
      };
    }

    function projectTargetToCapturedRoute(target) {
      const route = (avoidability && avoidability.target_route_lane_indices) || [];
      let best = null;
      for (let routeIndex = 0; routeIndex < route.length; routeIndex++) {
        const lane = roadElements[Number(route[routeIndex])];
        const xs = (lane && lane.x) || [], ys = (lane && lane.y) || [], zs = (lane && lane.z) || [];
        for (let segmentIndex = 0; segmentIndex + 1 < xs.length && segmentIndex + 1 < ys.length; segmentIndex++) {
          const x0 = Number(xs[segmentIndex]), y0 = Number(ys[segmentIndex]);
          const z0 = Number(zs[segmentIndex] || 0);
          const dx = Number(xs[segmentIndex + 1]) - x0;
          const dy = Number(ys[segmentIndex + 1]) - y0;
          const dz = Number(zs[segmentIndex + 1] || 0) - z0;
          const lengthSquared = dx * dx + dy * dy + dz * dz;
          if (lengthSquared < 1e-6) continue;
          const t = Math.max(0, Math.min(1,
            ((Number(target.x) - x0) * dx + (Number(target.y) - y0) * dy +
             (Number(target.z || 0) - z0) * dz) / lengthSquared));
          const ex = Number(target.x) - (x0 + t * dx);
          const ey = Number(target.y) - (y0 + t * dy);
          const ez = Number(target.z || 0) - (z0 + t * dz);
          const distanceSquared = ex * ex + ey * ey + ez * ez;
          if (!best || distanceSquared < best.distanceSquared) {
            best = {routeIndex, segmentIndex, t, distanceSquared};
          }
        }
      }
      return best;
    }

    function computeRouteTTC(pair, threshold, expandedTarget) {
      const route = (avoidability && avoidability.target_route_lane_indices) || [];
      const projection = route.length ? projectTargetToCapturedRoute(pair.target) : null;
      const constants = (avoidability && avoidability.constants) || {};
      const dt = Number(constants.dt || 0.1);
      const maxSteps = Number(constants.ttc_max_projection_steps || 92);
      const targetSpeed = speedOfAgent(pair.target);
      const routePath = [];
      const none = {
        ttc: null, routeGap: null, adversaryRouteSpeed: null, closingSpeed: null,
        routePath, targetPose: null, adversaryPose: pair.adversary,
      };
      if (!projection || targetSpeed <= 0) return none;

      let futureStep = 0;
      let traveled = 0;
      for (let routeIndex = projection.routeIndex; routeIndex < route.length; routeIndex++) {
        const lane = roadElements[Number(route[routeIndex])];
        const xs = (lane && lane.x) || [], ys = (lane && lane.y) || [], zs = (lane && lane.z) || [];
        if (xs.length < 2 || ys.length < 2) return none;
        const firstSegment = routeIndex === projection.routeIndex ? projection.segmentIndex : 0;
        for (let segmentIndex = firstSegment; segmentIndex + 1 < xs.length; segmentIndex++) {
          const x0 = Number(xs[segmentIndex]), y0 = Number(ys[segmentIndex]);
          const z0 = Number(zs[segmentIndex] || 0);
          const dx = Number(xs[segmentIndex + 1]) - x0;
          const dy = Number(ys[segmentIndex + 1]) - y0;
          const dz = Number(zs[segmentIndex + 1] || 0) - z0;
          const segmentLength = Math.hypot(dx, dy, dz);
          if (segmentLength <= 1e-6) continue;
          const segmentStartT = routeIndex === projection.routeIndex && segmentIndex === projection.segmentIndex
            ? projection.t : 0;
          const usableLength = (1 - segmentStartT) * segmentLength;
          while (futureStep < maxSteps) {
            const futureTime = futureStep * dt;
            if (futureTime >= threshold) return none;
            const targetDistance = targetSpeed * futureTime;
            if (targetDistance > traveled + usableLength + 1e-5) break;
            const localDistance = Math.max(0, targetDistance - traveled);
            const t = Math.max(segmentStartT, Math.min(1, segmentStartT + localDistance / segmentLength));
            const heading = Math.atan2(dy, dx);
            const targetPose = {
              ...expandedTarget,
              x: x0 + t * dx, y: y0 + t * dy, z: z0 + t * dz, heading,
            };
            const adversaryPose = projectedPose(pair.adversary, futureTime);
            routePath.push({x: targetPose.x, y: targetPose.y});
            if (verticalOverlap(targetPose, adversaryPose) && obbOverlap(targetPose, adversaryPose)) {
              const tangentX = dx / segmentLength, tangentY = dy / segmentLength;
              const adversaryRouteSpeed =
                Number(pair.adversary.vx || 0) * tangentX + Number(pair.adversary.vy || 0) * tangentY;
              const closingSpeed = targetSpeed - adversaryRouteSpeed;
              return {
                ttc: futureTime,
                routeGap: targetDistance, adversaryRouteSpeed, closingSpeed,
                routePath, targetPose, adversaryPose,
              };
            }
            futureStep += 1;
          }
          traveled += usableLength;
        }
      }
      return none;
    }

    function computeTTCOverlay(pair) {
      const constants = avoidability.constants || {};
      const dt = Number(constants.dt || 0.1);
      const deceleration = Number(constants.braking_deceleration || 5.0);
      const reactionTime = Number(constants.reaction_time_seconds || 1.0);
      const margin = Number(constants.ttc_margin_seconds || 0.1);
      const maxSteps = Number(constants.ttc_max_projection_steps || 92);
      const threshold = reactionTime + speedOfAgent(pair.target) / Math.max(deceleration, 1e-6) + margin;
      const expandedTarget = computeLateralBuffer(pair).expandedTarget;
      const targetPath = [];
      const adversaryPath = [];
      let targetPose = expandedTarget;
      let adversaryPose = pair.adversary;
      let contactTime = null;
      for (let futureStep = 0; futureStep < maxSteps; futureStep++) {
        const futureTime = futureStep * dt;
        if (futureTime >= threshold) break;
        targetPose = projectedPose(expandedTarget, futureTime);
        adversaryPose = projectedPose(pair.adversary, futureTime);
        targetPath.push({x: targetPose.x, y: targetPose.y});
        adversaryPath.push({x: adversaryPose.x, y: adversaryPose.y});
        if (verticalOverlap(targetPose, adversaryPose) && obbOverlap(targetPose, adversaryPose)) {
          contactTime = futureTime;
          break;
        }
      }
      const route = computeRouteTTC(pair, threshold, expandedTarget);
      const straightTTC = contactTime;
      const routeTTC = route.ttc;
      const finiteStraight = straightTTC == null ? Infinity : straightTTC;
      const finiteRoute = routeTTC == null ? Infinity : routeTTC;
      const effectiveTTC = Math.min(finiteStraight, finiteRoute);
      const winner = !Number.isFinite(effectiveTTC) ? 'none' : finiteRoute < finiteStraight ? 'route' : 'straight';
      return {
        dangerous: effectiveTTC < threshold,
        contactTime: Number.isFinite(effectiveTTC) ? effectiveTTC : null,
        straightTTC,
        routeTTC,
        effectiveTTC: Number.isFinite(effectiveTTC) ? effectiveTTC : null,
        winner,
        threshold,
        targetPath,
        adversaryPath,
        targetPose,
        adversaryPose,
        route,
      };
    }

    function computeLateralBuffer(pair) {
      const constants = avoidability.constants || {};
      const heading = Number(pair.target.heading || 0);
      const leftX = -Math.sin(heading), leftY = Math.cos(heading);
      const relX = Number(pair.adversary.x) - Number(pair.target.x);
      const relY = Number(pair.adversary.y) - Number(pair.target.y);
      const signedDistance = relX * leftX + relY * leftY;
      const relativeVX = Number(pair.adversary.vx || 0) - Number(pair.target.vx || 0);
      const relativeVY = Number(pair.adversary.vy || 0) - Number(pair.target.vy || 0);
      const relativeLateralVelocity = relativeVX * leftX + relativeVY * leftY;
      let intrusionSpeed = 0;
      if (Math.abs(signedDistance) > 1e-6) {
        intrusionSpeed = Math.max(0, relativeLateralVelocity * (-Math.sign(signedDistance)));
      }

      const base = Number(constants.lateral_buffer_base_distance || 0.2);
      const response = Number(constants.lateral_buffer_response_time_seconds || 0);
      const deceleration = Number(constants.lateral_buffer_deceleration || 0.8);
      const maximum = Number(constants.lateral_buffer_max_distance || 2.0);
      const buffer = Math.min(
        maximum,
        base + intrusionSpeed * response + intrusionSpeed * intrusionSpeed / (2 * Math.max(deceleration, 1e-6))
      );
      const expandedTarget = {
        ...pair.target,
        width: Number(pair.target.width || 0) + 2 * buffer,
      };
      const dangerous = verticalOverlap(expandedTarget, pair.adversary) && obbOverlap(expandedTarget, pair.adversary);
      return {
        dangerous,
        buffer,
        intrusionSpeed,
        expandedTarget,
      };
    }

    function detectionWindowSamples() {
      if (detectionSamplesCache !== null) return detectionSamplesCache;
      const classification = (avoidability && avoidability.classification) || {};
      const constants = (avoidability && avoidability.constants) || {};
      const collision = avoidability && avoidability.collision;
      const tBrake = Number(classification.t_brake);
      if (!collision || !(tBrake > 0)) {
        detectionSamplesCache = [];
        return detectionSamplesCache;
      }

      const dt = Number(constants.dt || 0.1);
      const reactionTime = Number(constants.reaction_time_seconds || 1.0);
      const halfWidth = Number(constants.reaction_window_half_width_seconds || 0.2);
      const minimum = tBrake + reactionTime - halfWidth;
      const maximum = tBrake + reactionTime + halfWidth;
      const collisionTimestep = Number(collision.collision_timestep);
      const samples = [];
      for (let i = 0; i < frames.length; i++) {
        const stepsBack = collisionTimestep - Number(episodeTimesteps[i]);
        if (stepsBack < 1) continue;
        const secondsBeforeCollision = stepsBack * dt;
        if (secondsBeforeCollision < minimum - 1e-5 || secondsBeforeCollision > maximum + 1e-5) continue;
        const pair = detectorPairForFrame(frames[i]);
        if (!pair) continue;
        const ttc = computeTTCOverlay(pair);
        const buffer = computeLateralBuffer(pair);
        samples.push({
          frameIndex: i,
          timestep: Number(episodeTimesteps[i]),
          stepsBack,
          secondsBeforeCollision,
          secondsBeforeBraking: secondsBeforeCollision - tBrake,
          pair,
          ttc,
          buffer,
          dangerous: ttc.dangerous || buffer.dangerous,
          ttcDangerous: ttc.dangerous,
          bufferDangerous: buffer.dangerous,
        });
      }
      detectionSamplesCache = samples;
      return detectionSamplesCache;
    }

    function selectedDetectionWindowSample() {
      const samples = detectionWindowSamples();
      if (!samples.length) return null;
      detectionSelection = Math.min(Math.max(0, detectionSelection), samples.length - 1);
      return samples[detectionSelection];
    }

    function computeReactionWindowDanger() {
      const classification = (avoidability && avoidability.classification) || {};
      const constants = (avoidability && avoidability.constants) || {};
      const tBrake = Number(classification.t_brake);
      if (!(tBrake > 0)) {
        return {
          available: false, dangerous: false, samples: 0, minimum: null, maximum: null,
          minimumBeforeBraking: null, maximumBeforeBraking: null,
        };
      }
      const reactionTime = Number(constants.reaction_time_seconds || 1.0);
      const halfWidth = Number(constants.reaction_window_half_width_seconds || 0.2);
      const samples = detectionWindowSamples();
      return {
        available: samples.length > 0,
        dangerous: samples.some(sample => sample.dangerous),
        samples: samples.length,
        minimum: tBrake + reactionTime - halfWidth,
        maximum: tBrake + reactionTime + halfWidth,
        minimumBeforeBraking: reactionTime - halfWidth,
        maximumBeforeBraking: reactionTime + halfWidth,
      };
    }

    function stopDetectionPlayback() {
      detectionPlaying = false;
      detectionPlay.innerText = 'Play';
      detectionLastTimestamp = 0;
    }

    function updateDetectionReadout() {
      const samples = detectionWindowSamples();
      if (!samples.length) {
        detectionSelection = 0;
        detectionSlider.max = 0;
        detectionSlider.value = 0;
        detectionSlider.disabled = true;
        detectionPlay.disabled = true;
        detectionLabel.innerText = 'n/a';
        detectionResult.className = 'avoidability-result reference';
        const reactionCheck = computeReactionWindowDanger();
        if (reactionCheck.minimum == null) {
          detectionResult.innerHTML = '<strong>No detection window</strong><code>t_brake</code> is unavailable for this collision.';
          detectionMarkers.innerText = 'C does not run the detection-window check when no avoidable braking time exists.';
        } else {
          detectionResult.innerHTML = '<strong>Detection window not sampled</strong>No recorded target/hitter state falls inside the required history window.';
          detectionMarkers.innerHTML =
            `Required history: <strong>${reactionCheck.maximum.toFixed(2)}s → ` +
            `${reactionCheck.minimum.toFixed(2)}s before collision</strong><br>` +
            `C checked <strong>0 samples</strong>, so danger=false.`;
        }
        return;
      }

      detectionSelection = Math.min(Math.max(0, detectionSelection), samples.length - 1);
      const sample = samples[detectionSelection];
      frameIndex = sample.frameIndex;
      detectionSlider.max = samples.length - 1;
      detectionSlider.value = detectionSelection;
      detectionSlider.disabled = false;
      detectionPlay.disabled = samples.length <= 1;
      detectionLabel.innerText = `${sample.secondsBeforeBraking.toFixed(1)}s before brake`;

      const sources = [];
      if (sample.ttcDangerous) sources.push(`TTC/${sample.ttc.winner}`);
      if (sample.bufferDangerous) sources.push('lateral buffer');
      const sourceText = sources.length ? sources.join(' + ') : 'none';
      const effectiveTTC = sample.ttc.effectiveTTC == null ? '∞' : `${sample.ttc.effectiveTTC.toFixed(2)}s`;
      detectionResult.className = sample.dangerous ? 'avoidability-result failed' : 'avoidability-result';
      detectionResult.innerHTML =
        `<strong>Sample ${detectionSelection + 1}/${samples.length} · ${sample.dangerous ? 'DANGER' : 'SAFE'} · source=${sourceText}</strong>` +
        `${sample.secondsBeforeCollision.toFixed(2)}s before collision · ` +
        `${sample.secondsBeforeBraking.toFixed(2)}s before braking · ` +
        `effective TTC=${effectiveTTC} / threshold=${sample.ttc.threshold.toFixed(2)}s · ` +
        `buffer=${sample.buffer.buffer.toFixed(2)}m/side`;

      const anyDanger = samples.some(item => item.dangerous);
      const cDanger = Number(((avoidability && avoidability.classification) || {}).genuine_target_failure || 0) > 0;
      const matches = anyDanger === cDanger;
      detectionMarkers.innerHTML =
        `Window samples: <strong>${samples[0].secondsBeforeBraking.toFixed(2)}s → ` +
        `${samples[samples.length - 1].secondsBeforeBraking.toFixed(2)}s before braking</strong><br>` +
        `C classification: <strong>${cDanger ? 'danger' : 'safe'}</strong> · ` +
        `renderer reconstruction: <strong>${anyDanger ? 'danger' : 'safe'}</strong> · ` +
        `<strong>${matches ? 'MATCH' : 'MISMATCH'}</strong>`;
    }

    function updateDetectionSelection() {
      stopDetectionPlayback();
      detectionSelection = Number(detectionSlider.value || 0);
      updateDetectionReadout();
      draw();
    }

    function setAvoidabilityPhase(phase) {
      stopDetectionPlayback();
      stopRolloutPlayback();
      avoidabilityPhase = phase;
      detectionPhaseButton.classList.toggle('active', phase === 'detection');
      brakingPhaseButton.classList.toggle('active', phase === 'braking');
      detectionControls.hidden = phase !== 'detection';
      brakingControls.hidden = phase !== 'braking';
      if (phase === 'detection') {
        detectionSelection = 0;
        updateDetectionReadout();
        draw();
      } else {
        reactionSlider.value = candidateSteps.length;
        updateAvoidabilitySelection();
      }
    }

    function stopRolloutPlayback() {
      rolloutPlaying = false;
      rolloutPlay.innerText = 'Play';
      rolloutLastTimestamp = 0;
    }

    function updateRolloutReadout(trajectories) {
      if (!trajectories) {
        rolloutStep = 0;
        rolloutSlider.min = 0;
        rolloutSlider.max = 0;
        rolloutSlider.value = 0;
        rolloutSlider.disabled = true;
        rolloutPlay.disabled = true;
        rolloutLabel.innerText = 't=0.0s';
        rolloutMarkers.innerText = 'Select a braking candidate.';
        return;
      }
      rolloutStep = Math.min(Math.max(0, rolloutStep), trajectories.effectiveEndStep);
      rolloutSlider.max = trajectories.effectiveEndStep;
      rolloutSlider.value = rolloutStep;
      rolloutSlider.disabled = false;
      rolloutPlay.disabled = trajectories.effectiveEndStep === 0;
      rolloutLabel.innerText = `t=${(rolloutStep * trajectories.dt).toFixed(1)} / ${trajectories.rolloutEndTime.toFixed(1)}s`;

      const targetStatus = trajectories.targetReachedStop ? 'reached' : 'not reached';
      const adversaryStatus = trajectories.adversaryReachedStop ? 'reached' : 'not reached';
      const endStatus = trajectories.exitedEarly
        ? `C exited at blocking collision · t=${trajectories.rolloutEndTime.toFixed(2)}s`
        : `C completed joint-stop horizon · t=${trajectories.rolloutEndTime.toFixed(2)}s`;
      rolloutMarkers.innerHTML =
        `Observed impact: <strong>${trajectories.observedCollisionTime.toFixed(2)}s</strong><br>` +
        `Target stop: <strong>${trajectories.targetStopSampleTime.toFixed(2)}s</strong> · ${targetStatus}<br>` +
        `Hitter stop: <strong>${trajectories.adversaryStopSampleTime.toFixed(2)}s</strong> · ${adversaryStatus}<br>` +
        endStatus;
    }

    function updateAvoidabilityResult(trajectories) {
      if (!trajectories) return;
      const candidateIndex = trajectories.candidateIndex;
      const leadSeconds = trajectories.stepsBack * trajectories.dt;
      const avoided = !!candidateValue('avoided', candidateIndex, false);
      const originalBlock = !!candidateValue('collision_with_original_adversary', candidateIndex, false);
      const secondaryBlock = !!candidateValue('at_fault_collision_with_other_adversary', candidateIndex, false);
      let title = 'Avoided';
      if (originalBlock) title = 'Rejected · original hitter';
      else if (secondaryBlock) title = 'Rejected · secondary blocker';
      else if (!avoided) title = 'Rejected';
      const detail = trajectories.exitedEarly
        ? `C terminated at rollout step ${trajectories.effectiveEndStep}, before the joint-stop horizon at step ${trajectories.fullEndStep}.`
        : `C checked all samples through step ${trajectories.fullEndStep}, when both vehicles have stopped.`;
      avoidabilityResult.className = avoided ? 'avoidability-result' : 'avoidability-result failed';
      avoidabilityResult.innerHTML =
        `<strong>${title} · braking ${leadSeconds.toFixed(1)}s before collision</strong>${detail}`;
    }

    function updateAvoidabilitySelection() {
      stopRolloutPlayback();
      reactionSelection = Number(reactionSlider.value || 0);
      rolloutStep = 0;
      if (reactionSelection === 0) {
        const collision = avoidability.collision;
        frameIndex = nearestFrameAtOrBefore(collision.collision_timestep);
        reactionLabel.innerText = 'collision';
        avoidabilityResult.className = 'avoidability-result reference';
        avoidabilityResult.innerHTML = '<strong>Observed collision · 0.0s</strong>';
        updateRolloutReadout(null);
        draw();
        return;
      }

      const candidateIndex = reactionSelection - 1;
      const stepsBack = Number(candidateValue('steps_back', candidateIndex, 0));
      const dt = Number((avoidability.constants || {}).dt || 0.1);
      const leadSeconds = stepsBack * dt;
      const collisionTimestep = Number(avoidability.collision.collision_timestep);
      frameIndex = nearestFrameAtOrBefore(collisionTimestep - stepsBack);
      reactionLabel.innerText = `${leadSeconds.toFixed(1)}s`;
      const trajectories = selectedCounterfactualTrajectories();
      updateAvoidabilityResult(trajectories);
      updateRolloutReadout(trajectories);
      draw();
    }

    function setReplayMode(mode) {
      if (mode === 'avoidability' && !hasAvoidability) return;
      if (mode === replayMode) return;
      if (mode === 'avoidability') {
        observedFrameIndex = frameIndex;
        playing = false;
        playToggle.innerText = 'Play';
      } else {
        stopDetectionPlayback();
        stopRolloutPlayback();
        frameIndex = observedFrameIndex;
      }
      replayMode = mode;
      observedModeButton.classList.toggle('active', mode === 'observed');
      avoidabilityModeButton.classList.toggle('active', mode === 'avoidability');
      observedControls.hidden = mode !== 'observed';
      avoidabilityControls.hidden = mode !== 'avoidability';
      if (mode === 'avoidability') {
        setAvoidabilityPhase('detection');
      } else {
        draw();
      }
    }

    function isAgentBraking(agent) {
      if (agent.counterfactual) return !!agent.braking || agent.stopped;
      if (frameIndex <= 0) return speedOfAgent(agent) < 0.1;
      const prev = findAgentById(frames[frameIndex - 1], agent.id);
      if (!prev) return speedOfAgent(agent) < 0.1;
      return speedOfAgent(agent) < speedOfAgent(prev) - 0.05 || speedOfAgent(agent) < 0.1;
    }

    function updateFocusedAgentTelemetry() {
      const agent = selectedAgentId == null ? null : findAgentById(currentDisplayFrame(), selectedAgentId);
      document.getElementById('focus-agent').innerText = agent ? String(agent.id) : 'none';
      if (!agent) {
        document.getElementById('focus-speed').innerText = 'n/a';
        document.getElementById('focus-velocity').innerText = 'n/a';
        return;
      }
      const speed = speedOfAgent(agent);
      document.getElementById('focus-speed').innerText = `${speed.toFixed(2)} m/s | ${(speed * 3.6).toFixed(1)} km/h`;
      document.getElementById('focus-velocity').innerText = `vx=${Number(agent.vx || 0).toFixed(2)}, vy=${Number(agent.vy || 0).toFixed(2)}`;
    }

    function focusOnAgent(agent, zoom = 2.5) {
      if (!agent) return;
      selectedAgentId = agent.id;
      camera.x = agent.x;
      camera.y = agent.y;
      camera.zoom = zoom;
      draw();
    }

    function setFollowTarget(enabled) {
      followTarget = !!enabled;
      focusTargetButton.innerText = followTarget ? 'Unlock Target' : 'Focus Target';
      if (followTarget) {
        const target = findTarget(currentDisplayFrame());
        if (target) {
          selectedAgentId = target.id;
          camera.x = target.x;
          camera.y = target.y;
          camera.zoom = Math.max(camera.zoom, 3.0);
        }
      }
      draw();
    }

    function drawRoads() {
      for (let roadIndex = 0; roadIndex < roadElements.length; roadIndex++) {
        const elem = roadElements[roadIndex];
        const xs = elem.x || [];
        const ys = elem.y || [];
        if (xs.length < 2 || ys.length < 2) continue;
        const type = Number(elem.type || 0);
        let style = null;
        if (type >= 1 && type <= 3) style = { color: '#606b77', width: 0.8, alpha: 0.75, dash: [] };
        else if (type === 11) style = { color: '#f3f4f6', width: 1.3, alpha: 0.95, dash: [8, 6] };
        else if (type === 12 || type === 13) style = { color: '#f3f4f6', width: 1.3, alpha: 0.95, dash: [] };
        else if (type === 14 || type === 15) style = { color: '#facc15', width: 1.4, alpha: 0.95, dash: [8, 6] };
        else if (type >= 16 && type <= 18) style = { color: '#facc15', width: 1.4, alpha: 0.95, dash: [] };
        else if (type >= 21 && type <= 23) style = { color: '#929ca7', width: 1.5, alpha: 0.8, dash: [] };
        if (!style) continue;
        ctx.setLineDash(style.dash);
        ctx.lineCap = style.dash.length ? 'butt' : 'round';
        ctx.beginPath();
        for (let i = 0; i < xs.length; i++) {
          const p = worldToCanvas(xs[i], ys[i]);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.strokeStyle = style.color;
        ctx.globalAlpha = style.alpha;
        ctx.lineWidth = style.width;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1.0;
      }
    }

    function drawComplianceOverlay() {
      if (!compliance || !Number(compliance.valid || 0)) return;
      const trajectory = compliance.hitter_trajectory || [];
      const currentTimestep = Number(episodeTimesteps[frameIndex] ?? compliance.collision_timestep);
      const visibleTrajectory = trajectory.filter(point => Number(point.timestep) <= currentTimestep);
      drawTrajectoryLine(trajectory, '#fbbf24', 2.0, [6, 5]);
      drawTrajectoryLine(visibleTrajectory, '#fb7185', 3.6);

      const lineIndex = Number(compliance.crossed_line_index ?? -1);
      const segmentIndex = Number(compliance.crossed_line_segment_index ?? -1);
      const line = roadElements[lineIndex];
      if (line && segmentIndex >= 0 && segmentIndex + 1 < (line.x || []).length) {
        const p1 = worldToCanvas(line.x[segmentIndex], line.y[segmentIndex]);
        const p2 = worldToCanvas(line.x[segmentIndex + 1], line.y[segmentIndex + 1]);
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = '#ff2d55';
        ctx.lineWidth = 6;
        ctx.globalAlpha = 0.95;
        ctx.stroke();
        ctx.restore();
      }

      if (Number(compliance.solid_line_violation || 0) > 0) {
        drawTrajectoryLine([
          {x: Number(compliance.crossing_segment_start_x), y: Number(compliance.crossing_segment_start_y)},
          {x: Number(compliance.crossing_segment_end_x), y: Number(compliance.crossing_segment_end_y)},
        ], '#ff2d55', 5.0);
      }
    }

    function drawTraffic(frame) {
      for (const control of frame || []) {
        const sl = control.stop_line || [];
        if (sl.length < 6) continue;
        const p1 = worldToCanvas(sl[0], sl[1]);
        const p2 = worldToCanvas(sl[3], sl[4]);
        let color = '#888';
        if (Number(control.type) === 1) {
          const state = Number(control.state || 0);
          if (state === 1) color = '#d64545';
          else if (state === 2) color = '#f1c40f';
          else if (state === 3) color = '#27ae60';
          else if (state === 4) color = '#94a3b8';
          else color = '#888';
        } else if (Number(control.type) === 2) {
          color = '#d64545';
        } else if (Number(control.type) === 3) {
          color = '#d4ac0d';
        }
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.9;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }
    }

    function drawDetectorPose(agent, color, dash=[]) {
      if (!agent) return;
      const center = worldToCanvas(agent.x, agent.y);
      const length = Math.max(Number(agent.length || 0) * center.scale, 6);
      const width = Math.max(Number(agent.width || 0) * center.scale, 4);
      ctx.save();
      ctx.translate(center.x, center.y);
      ctx.rotate(-Number(agent.heading || 0));
      ctx.fillStyle = `${color}18`;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash(dash);
      ctx.fillRect(-length / 2, -width / 2, length, width);
      ctx.strokeRect(-length / 2, -width / 2, length, width);
      ctx.restore();
    }

    function drawLateralBufferOverlay(bufferResult) {
      const color = bufferResult.dangerous ? '#fb7185' : '#4ade80';
      drawDetectorPose(bufferResult.expandedTarget, color, [7, 4]);
      bufferBadge.className = bufferResult.dangerous ? 'detector-badge danger' : 'detector-badge';
      bufferBadge.innerText = `BUFFER ${bufferResult.dangerous ? 'DANGER' : 'SAFE'} · b=${bufferResult.buffer.toFixed(2)}m/side · intrusion=${bufferResult.intrusionSpeed.toFixed(2)}m/s`;
    }

    function drawTTCOverlay(ttc) {
      const color = ttc.dangerous ? '#fb7185' : '#4ade80';
      drawTrajectoryLine(ttc.targetPath, color, 2.2, [3, 4]);
      drawTrajectoryLine(ttc.adversaryPath, color, 2.2, [8, 4]);
      drawDetectorPose(ttc.targetPose, color);
      drawDetectorPose(ttc.adversaryPose, color, [6, 4]);
      drawTrajectoryLine(ttc.route.routePath, '#c084fc', 3.0);
      if (ttc.route.targetPose) drawDetectorPose(ttc.route.targetPose, '#c084fc');
      if (ttc.route.targetPose) drawDetectorPose(ttc.route.adversaryPose, '#c084fc', [6, 4]);
      ttcBadge.className = ttc.dangerous ? 'detector-badge danger' : 'detector-badge';
      const seconds = value => value == null ? '∞' : `${value.toFixed(2)}s`;
      const meters = value => value == null ? 'n/a' : `${value.toFixed(2)}m`;
      const speed = value => value == null ? 'n/a' : `${value.toFixed(2)}m/s`;
      ttcBadge.innerText = `TTC ${ttc.dangerous ? 'DANGER' : 'SAFE'} · straight=${seconds(ttc.straightTTC)}` +
        ` · route=${seconds(ttc.routeTTC)} · effective=${seconds(ttc.effectiveTTC)}` +
        ` · threshold=${ttc.threshold.toFixed(2)}s · gap=${meters(ttc.route.routeGap)}` +
        ` · leader=${speed(ttc.route.adversaryRouteSpeed)} · closing=${speed(ttc.route.closingSpeed)}` +
        ` · source=${ttc.winner}`;
    }

    function drawDetectorOverlays() {
      if (replayMode !== 'avoidability' || avoidabilityPhase !== 'detection') {
        detectorHud.hidden = true;
        return;
      }
      const pair = detectorPair();
      const visible = !!pair && (showLateralBuffer || showTTC);
      detectorHud.hidden = !visible;
      bufferBadge.hidden = !showLateralBuffer;
      ttcBadge.hidden = !showTTC;
      if (!pair) return;
      if (showLateralBuffer) drawLateralBufferOverlay(computeLateralBuffer(pair));
      if (showTTC) drawTTCOverlay(computeTTCOverlay(pair));
    }

    function drawTrajectoryLine(points, color, width, dash=[]) {
      if (!points || points.length < 2) return;
      ctx.save();
      ctx.beginPath();
      for (let i = 0; i < points.length; i++) {
        const point = worldToCanvas(points[i].x, points[i].y);
        if (i === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.setLineDash(dash);
      ctx.globalAlpha = 0.92;
      ctx.stroke();
      ctx.restore();
    }

    function drawStopPose(point, dimensions, color, label, labelOffsetY=0) {
      if (!point) return;
      const canvasPoint = worldToCanvas(point.x, point.y);
      const length = Math.max(Number(dimensions.length || 0) * canvasPoint.scale, 6);
      const width = Math.max(Number(dimensions.width || 0) * canvasPoint.scale, 4);
      ctx.save();
      ctx.translate(canvasPoint.x, canvasPoint.y);
      ctx.rotate(-Number(point.heading || 0));
      ctx.fillStyle = `${color}22`;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.fillRect(-length / 2, -width / 2, length, width);
      ctx.strokeRect(-length / 2, -width / 2, length, width);
      ctx.restore();

      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,0.94)';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(canvasPoint.x, canvasPoint.y, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(canvasPoint.x - 3, canvasPoint.y);
      ctx.lineTo(canvasPoint.x + 3, canvasPoint.y);
      ctx.moveTo(canvasPoint.x, canvasPoint.y - 3);
      ctx.lineTo(canvasPoint.x, canvasPoint.y + 3);
      ctx.stroke();
      ctx.font = '600 11px ui-sans-serif, sans-serif';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(25,31,38,0.92)';
      ctx.fillRect(canvasPoint.x + 9, canvasPoint.y - 11 + labelOffsetY, textWidth + 8, 17);
      ctx.fillStyle = color;
      ctx.fillText(label, canvasPoint.x + 13, canvasPoint.y + 1 + labelOffsetY);
      ctx.restore();
    }

    function drawImpactPose(point, dimensions, color, label, labelOffsetY=0, dash=[]) {
      if (!point) return;
      const canvasPoint = worldToCanvas(point.x, point.y);
      const length = Math.max(Number(dimensions.length || 0) * canvasPoint.scale, 6);
      const width = Math.max(Number(dimensions.width || 0) * canvasPoint.scale, 4);
      ctx.save();
      ctx.translate(canvasPoint.x, canvasPoint.y);
      ctx.rotate(-Number(point.heading || 0));
      ctx.fillStyle = `${color}30`;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.setLineDash(dash);
      ctx.fillRect(-length / 2, -width / 2, length, width);
      ctx.strokeRect(-length / 2, -width / 2, length, width);
      ctx.restore();

      ctx.save();
      ctx.font = '700 11px ui-sans-serif, sans-serif';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(25,31,38,0.94)';
      ctx.fillRect(canvasPoint.x + 9, canvasPoint.y - 11 + labelOffsetY, textWidth + 8, 17);
      ctx.fillStyle = color;
      ctx.fillText(label, canvasPoint.x + 13, canvasPoint.y + 1 + labelOffsetY);
      ctx.restore();
    }

    function drawBrakingStart(point, reactionTime) {
      if (!point) return;
      const canvasPoint = worldToCanvas(point.x, point.y);
      const label = `braking starts · t=${reactionTime.toFixed(2)}s`;
      ctx.save();
      ctx.fillStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(canvasPoint.x, canvasPoint.y, 7, Math.PI / 2, Math.PI * 1.5);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = '#dc2626';
      ctx.beginPath();
      ctx.arc(canvasPoint.x, canvasPoint.y, 7, -Math.PI / 2, Math.PI / 2);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(canvasPoint.x, canvasPoint.y, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.font = '700 11px ui-sans-serif, sans-serif';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(25,31,38,0.94)';
      ctx.fillRect(canvasPoint.x + 10, canvasPoint.y - 25, textWidth + 8, 17);
      ctx.fillStyle = '#fca5a5';
      ctx.fillText(label, canvasPoint.x + 14, canvasPoint.y - 13);
      ctx.restore();
    }

    function drawIntermediatePose(agent, dimensions, color, alpha) {
      if (!agent) return;
      const center = worldToCanvas(agent.x, agent.y);
      const length = Math.max(Number(dimensions.length || 0) * center.scale, 6);
      const width = Math.max(Number(dimensions.width || 0) * center.scale, 4);
      ctx.save();
      ctx.translate(center.x, center.y);
      ctx.rotate(-Number(agent.heading || 0));
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.2;
      ctx.fillRect(-length / 2, -width / 2, length, width);
      ctx.strokeRect(-length / 2, -width / 2, length, width);
      ctx.restore();
    }

    function drawIntermediatePositions(samples, dimensions, color, currentStep) {
      let previous = null;
      for (let step = 0; step < samples.length; step++) {
        const sample = samples[step];
        const unchanged = previous && Math.hypot(sample.x - previous.x, sample.y - previous.y) <= 1e-6;
        previous = sample;
        if (step === currentStep || unchanged) continue;
        drawIntermediatePose(sample, dimensions, color, step < currentStep ? 0.22 : 0.09);
      }
    }

    function drawAvoidabilityTrajectories() {
      const trajectories = selectedCounterfactualTrajectories();
      if (!trajectories) return;
      const currentStep = Math.min(rolloutStep, trajectories.effectiveEndStep);
      drawTrajectoryLine(trajectories.adversarySamples, '#22d3ee', 2.6, [6, 5]);
      drawTrajectoryLine(trajectories.targetSamples, '#fb7185', 3.8);
      drawIntermediatePositions(
        trajectories.adversarySamples,
        trajectories.adversaryDimensions,
        '#22d3ee',
        currentStep
      );
      drawIntermediatePositions(
        trajectories.targetSamples,
        trajectories.targetDimensions,
        '#fb7185',
        currentStep
      );
      drawBrakingStart(trajectories.brakingStart, trajectories.reactionTime);
      if (trajectories.impact) {
        const blockerColor = trajectories.impact.blockerIsOriginal ? '#22d3ee' : '#c084fc';
        const blockerLabel = trajectories.impact.blockerIsOriginal
          ? `adversary at impact · t=${trajectories.impact.time.toFixed(2)}s`
          : `blocking agent ${trajectories.impact.blockerId} at impact · t=${trajectories.impact.time.toFixed(2)}s`;
        drawImpactPose(
          trajectories.impact.blocker,
          trajectories.impact.blockerDimensions,
          blockerColor,
          blockerLabel,
          29,
          [4, 3]
        );
        drawImpactPose(
          trajectories.impact.target,
          trajectories.targetDimensions,
          '#fb7185',
          `impact target · ${speedOfAgent(trajectories.impact.target).toFixed(2)} m/s`,
          -29
        );
      } else {
        if (trajectories.targetReachedStop) {
          drawStopPose(
            trajectories.targetStop,
            trajectories.targetDimensions,
            '#fb7185',
            `target stop · t=${trajectories.targetStopSampleTime.toFixed(2)}s`,
            -17
          );
        }
        drawImpactPose(
          trajectories.adversaryAtRolloutEnd,
          trajectories.adversaryDimensions,
          '#22d3ee',
          `hitter stop · t=${trajectories.adversaryStopSampleTime.toFixed(2)}s`,
          17,
          [7, 4]
        );
      }
    }

    function drawVelocityVector(agent, center) {
      if (!showVelocityVectors) return;
      const vx = Number(agent.vx || 0);
      const vy = Number(agent.vy || 0);
      const speed = Math.hypot(vx, vy);
      if (speed < 0.15) return;

      const rawEnd = worldToCanvas(agent.x + vx * 0.8, agent.y + vy * 0.8);
      let dx = rawEnd.x - center.x;
      let dy = rawEnd.y - center.y;
      const rawLen = Math.hypot(dx, dy);
      if (rawLen <= 0) return;

      const minLen = 10;
      const maxLen = agent.is_target ? 90 : 70;
      const targetLen = Math.min(maxLen, Math.max(minLen, rawLen));
      dx = dx / rawLen * targetLen;
      dy = dy / rawLen * targetLen;

      const endX = center.x + dx;
      const endY = center.y + dy;
      const angle = Math.atan2(dy, dx);
      const color = agent.is_target ? '#fb7185' : '#22d3ee';

      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.globalAlpha = agent.is_target ? 0.95 : 0.78;
      ctx.lineWidth = agent.is_target ? 2.4 : 1.8;
      ctx.beginPath();
      ctx.moveTo(center.x, center.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      const head = agent.is_target ? 8 : 6;
      ctx.beginPath();
      ctx.moveTo(endX, endY);
      ctx.lineTo(endX - head * Math.cos(angle - Math.PI / 6), endY - head * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(endX - head * Math.cos(angle + Math.PI / 6), endY - head * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function drawAgent(agent) {
      const center = worldToCanvas(agent.x, agent.y);
      const scale = center.scale;
      const length = Math.max(agent.length * scale, 6);
      const width = Math.max(agent.width * scale, 4);
      const heading = Number(agent.heading || 0);
      let fill = '#3b82f6';
      if (agent.is_target) fill = '#f05252';
      else if (!agent.active) fill = '#8995a3';
      if (agent.stopped) fill = '#f59e0b';

      ctx.save();
      ctx.translate(center.x, center.y);
      ctx.rotate(-heading);
      const ghosted = replayMode === 'avoidability' && avoidabilityPhase === 'braking' && !agent.counterfactual;
      ctx.globalAlpha = ghosted ? 0.18 : 1.0;
      ctx.fillStyle = fill;
      ctx.strokeStyle = 'rgba(255,255,255,0.48)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.rect(-length / 2, -width / 2, length, width);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(length / 2, 0);
      ctx.lineTo(length / 2 - Math.max(width * 0.9, 6), width * 0.32);
      ctx.lineTo(length / 2 - Math.max(width * 0.9, 6), -width * 0.32);
      ctx.closePath();
      ctx.fillStyle = ghosted ? 'rgba(255,255,255,0.28)' : 'rgba(255,255,255,0.75)';
      ctx.fill();
      if (isAgentBraking(agent)) {
        ctx.fillStyle = 'rgba(255,0,0,0.92)';
        ctx.shadowColor = 'rgba(255,0,0,0.85)';
        ctx.shadowBlur = 8;
        const lightW = Math.max(2, length * 0.05);
        const lightH = Math.max(2, width * 0.28);
        ctx.fillRect(-length / 2, -width / 2, lightW, lightH);
        ctx.fillRect(-length / 2, width / 2 - lightH, lightW, lightH);
        ctx.shadowBlur = 0;
      }
      ctx.restore();

      if (!ghosted) drawVelocityVector(agent, center);
      if (agent.id === selectedAgentId) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(center.x, center.y, Math.max(length, width) * 0.75, 0, Math.PI * 2);
        ctx.strokeStyle = agent.is_target ? '#fb7185' : '#22d3ee';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
      }

      hitAgents.push({
        agent,
        x: center.x,
        y: center.y,
        radius: Math.max(length, width) * 0.7,
      });
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
      hitAgents = [];
      if (followTarget) {
        const target = findTarget(currentDisplayFrame());
        if (target) {
          camera.x = target.x;
          camera.y = target.y;
        }
      }
      drawRoads();
      drawComplianceOverlay();
      drawTraffic(trafficFrames[frameIndex] || []);
      drawDetectorOverlays();
      drawAvoidabilityTrajectories();
      const frame = currentDisplayFrame();
      for (const agent of frame) drawAgent(agent);
      updateFocusedAgentTelemetry();
      if (replayMode === 'observed') {
        frameLabel.innerText = `${frameIndex + 1} / ${Math.max(frames.length, 1)}`;
        slider.value = frameIndex;
      }
    }

    function tick(ts) {
      if (!playing) return;
      if (!lastTimestamp) lastTimestamp = ts;
      const frameDuration = 1000 / (10 * speed);
      if (ts - lastTimestamp >= frameDuration) {
        frameIndex = (frameIndex + 1) % Math.max(frames.length, 1);
        observedFrameIndex = frameIndex;
        lastTimestamp = ts;
        draw();
      }
      requestAnimationFrame(tick);
    }

    function rolloutTick(ts) {
      if (!rolloutPlaying || replayMode !== 'avoidability') return;
      const trajectories = selectedCounterfactualTrajectories();
      if (!trajectories) {
        stopRolloutPlayback();
        return;
      }
      if (!rolloutLastTimestamp) rolloutLastTimestamp = ts;
      const frameDuration = 1000 / (10 * speed);
      if (ts - rolloutLastTimestamp >= frameDuration) {
        rolloutStep += 1;
        rolloutLastTimestamp = ts;
        if (rolloutStep >= trajectories.effectiveEndStep) {
          rolloutStep = trajectories.effectiveEndStep;
          stopRolloutPlayback();
        }
        updateRolloutReadout(trajectories);
        draw();
      }
      if (rolloutPlaying) requestAnimationFrame(rolloutTick);
    }

    function detectionTick(ts) {
      if (!detectionPlaying || replayMode !== 'avoidability' || avoidabilityPhase !== 'detection') return;
      const samples = detectionWindowSamples();
      if (!samples.length) {
        stopDetectionPlayback();
        return;
      }
      if (!detectionLastTimestamp) detectionLastTimestamp = ts;
      const frameDuration = 1000 / (10 * speed);
      if (ts - detectionLastTimestamp >= frameDuration) {
        detectionSelection += 1;
        detectionLastTimestamp = ts;
        if (detectionSelection >= samples.length - 1) {
          detectionSelection = samples.length - 1;
          stopDetectionPlayback();
        }
        updateDetectionReadout();
        draw();
      }
      if (detectionPlaying) requestAnimationFrame(detectionTick);
    }

    slider.max = Math.max(frames.length - 1, 0);
    slider.addEventListener('input', (e) => {
      frameIndex = Number(e.target.value || 0);
      observedFrameIndex = frameIndex;
      draw();
    });
    reactionSlider.addEventListener('input', updateAvoidabilitySelection);
    detectionPhaseButton.addEventListener('click', () => setAvoidabilityPhase('detection'));
    brakingPhaseButton.addEventListener('click', () => setAvoidabilityPhase('braking'));
    detectionSlider.addEventListener('input', updateDetectionSelection);
    detectionPlay.addEventListener('click', () => {
      const samples = detectionWindowSamples();
      if (samples.length <= 1) return;
      if (detectionPlaying) {
        stopDetectionPlayback();
        return;
      }
      if (detectionSelection >= samples.length - 1) detectionSelection = 0;
      detectionPlaying = true;
      detectionPlay.innerText = 'Pause';
      detectionLastTimestamp = 0;
      updateDetectionReadout();
      draw();
      requestAnimationFrame(detectionTick);
    });
    rolloutSlider.addEventListener('input', (event) => {
      stopRolloutPlayback();
      rolloutStep = Number(event.target.value || 0);
      const trajectories = selectedCounterfactualTrajectories();
      updateRolloutReadout(trajectories);
      draw();
    });
    rolloutPlay.addEventListener('click', () => {
      const trajectories = selectedCounterfactualTrajectories();
      if (!trajectories || trajectories.effectiveEndStep === 0) return;
      if (rolloutPlaying) {
        stopRolloutPlayback();
        return;
      }
      if (rolloutStep >= trajectories.effectiveEndStep) rolloutStep = 0;
      rolloutPlaying = true;
      rolloutPlay.innerText = 'Pause';
      rolloutLastTimestamp = 0;
      updateRolloutReadout(trajectories);
      draw();
      requestAnimationFrame(rolloutTick);
    });
    observedModeButton.addEventListener('click', () => setReplayMode('observed'));
    avoidabilityModeButton.addEventListener('click', () => setReplayMode('avoidability'));
    bufferToggle.addEventListener('click', () => {
      showLateralBuffer = !showLateralBuffer;
      bufferToggle.classList.toggle('active', showLateralBuffer);
      bufferToggle.setAttribute('aria-pressed', String(showLateralBuffer));
      draw();
    });
    ttcToggle.addEventListener('click', () => {
      showTTC = !showTTC;
      ttcToggle.classList.toggle('active', showTTC);
      ttcToggle.setAttribute('aria-pressed', String(showTTC));
      draw();
    });
    playToggle.addEventListener('click', () => {
      playing = !playing;
      playToggle.innerText = playing ? 'Pause' : 'Play';
      lastTimestamp = 0;
      if (playing) requestAnimationFrame(tick);
    });
    speedSlider.addEventListener('input', (e) => {
      speed = Number(e.target.value || 6) / 6;
      speedLabel.innerText = `${speed.toFixed(1)}x`;
    });
    focusTargetButton.addEventListener('click', () => setFollowTarget(!followTarget));
    resetViewButton.addEventListener('click', () => {
      followTarget = false;
      focusTargetButton.innerText = 'Focus Target';
      selectedAgentId = null;
      camera = createDefaultCamera();
      draw();
    });
    velocityToggleButton.addEventListener('click', () => {
      showVelocityVectors = !showVelocityVectors;
      velocityToggleButton.innerText = showVelocityVectors ? 'Hide Vectors' : 'Show Vectors';
      draw();
    });
    canvas.addEventListener('mousedown', (e) => {
      if (followTarget) setFollowTarget(false);
      dragState = { x: e.offsetX, y: e.offsetY, cameraX: camera.x, cameraY: camera.y };
    });
    canvas.addEventListener('mousemove', (e) => {
      if (!dragState) return;
      const before = canvasToWorld(dragState.x, dragState.y);
      const now = canvasToWorld(e.offsetX, e.offsetY);
      camera.x = dragState.cameraX + (before.x - now.x);
      camera.y = dragState.cameraY + (before.y - now.y);
      draw();
    });
    window.addEventListener('mouseup', () => { dragState = null; });
    canvas.addEventListener('mouseleave', () => { dragState = null; });
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (followTarget) setFollowTarget(false);
      const pointerBefore = canvasToWorld(e.offsetX, e.offsetY);
      const zoomFactor = e.deltaY < 0 ? 1.12 : 0.9;
      camera.zoom = Math.min(20, Math.max(0.4, camera.zoom * zoomFactor));
      const pointerAfter = canvasToWorld(e.offsetX, e.offsetY);
      camera.x += pointerBefore.x - pointerAfter.x;
      camera.y += pointerBefore.y - pointerAfter.y;
      draw();
    }, { passive: false });
    canvas.addEventListener('click', (e) => {
      const x = e.offsetX;
      const y = e.offsetY;
      for (const hit of hitAgents) {
        const dx = x - hit.x;
        const dy = y - hit.y;
        if (dx * dx + dy * dy <= hit.radius * hit.radius) {
          if (followTarget && !hit.agent.is_target) setFollowTarget(false);
          focusOnAgent(hit.agent, hit.agent.is_target ? 3.0 : 2.2);
          break;
        }
      }
    });

    setMeta();
    camera = createDefaultCamera();
    avoidabilityModeButton.disabled = !hasAvoidability;
    avoidabilityModeButton.title = hasAvoidability ? 'Inspect detection and C-recorded braking candidates' : 'No collision avoidability trace';
    reactionSlider.max = candidateSteps.length;
    updateRolloutReadout(null);
    speedLabel.innerText = `${speed.toFixed(1)}x`;
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""


def render_compact_replay_html(replay_path, output_path, render_context=None):
    replay_bundle = load_compact_replay(replay_path)
    payload = _build_render_payload(replay_bundle)
    render_context = render_context or {}
    payload["metadata"]["episode_id"] = payload["metadata"].get(
        "episode_id", render_context.get("episode_id", Path(output_path).stem)
    )
    payload["navigation"] = render_context.get("navigation", {})
    payload["summary"] = _format_summary_for_render(render_context.get("summary", {}))
    title = f"{payload['metadata'].get('map_name', 'Replay')} | {payload['metadata'].get('scenario_id', 'episode')}"
    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def _safe_value(value):
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _safe_first(row, *keys):
    for key in keys:
        value = _safe_value(row.get(key))
        if value is not None:
            return value
    return None


def generate_failure_index(episodes_df, render_lookup, output_path):
    rows = []
    preferred_columns = [
        "episode_id",
        "adv_reward_weight_drive",
        "map_name",
        "scenario_id",
        "did_target_fail",
        "did_target_collide",
        "did_target_offroad",
        "did_target_run_light",
        "did_target_make_progress",
        "did_target_have_at_fault_collision",
        "target_num_goals_reached",
        "target_ttc_within_bound_rate",
        "target_progress_ratio",
        "target_puffer_score",
        "target_hit_responsibility",
        "target_hit_low_responsibility_rate",
        "target_hit_at_fault_rate",
        "target_collision_impact_zone",
        "target_collision_responsibility",
        "target_collision_severity",
        "genuine_target_failure",
        "unavoidable",
        "adversary_forced",
        "target_collision_target_failure_rate",
        "target_collision_unavoidable_rate",
        "target_collision_adversary_forced_rate",
        "compliance_avoidability_outcome",
        "hitter_compliance_compliant",
        "hitter_compliance_reason_signature",
        "hitter_compliance_wrong_way_distance",
        "hitter_compliance_max_speed_ratio",
        "hitter_compliance_lane_unavailable_sample_count",
        "hitter_compliance_speed_limit_unavailable_sample_count",
        "t_brake",
        "target_episode_return",
        "target_episode_length",
        "has_replay",
    ]
    derived_columns = {"genuine_target_failure", "unavoidable", "adversary_forced", "t_brake"}
    existing_columns = [col for col in preferred_columns if col in episodes_df.columns or col in derived_columns]
    for row in episodes_df.to_dict(orient="records"):
        replay_html = render_lookup.get(row.get("episode_id"))
        out = {key: _safe_value(row.get(key)) for key in existing_columns}
        out["genuine_target_failure"] = _safe_first(
            row, "genuine_target_failure", "target_collision_target_failure_rate"
        )
        out["unavoidable"] = _safe_first(row, "unavoidable", "target_collision_unavoidable_rate")
        out["adversary_forced"] = _safe_first(row, "adversary_forced", "target_collision_adversary_forced_rate")
        out["t_brake"] = _safe_first(row, "t_brake", "target_mean_last_avoidable_braking_seconds_before_collision")
        if "target_collision_impact_zone" in out:
            out["target_collision_impact_zone"] = (
                f"{_impact_zone_label(out['target_collision_impact_zone'])}"
                f" [{int(float(out['target_collision_impact_zone'] or 0))}]"
            )
        out["rendered_html"] = replay_html
        rows.append(out)

    rows.sort(key=lambda item: (-(item.get("did_target_fail") or 0), item.get("target_episode_return") or 0))
    title = Path(output_path).parent.name
    drive_weight_columns = {"adv_reward_weight_drive"}
    header_cells = "".join(
        f"<th data-key='{col}' class='{'drive-weight-col' if col in drive_weight_columns else ''}'>{col}<span class='sort-indicator'></span></th>"
        for col in existing_columns
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Failure Index</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #20262d; color: #edf2f7; }}
    .wrap {{ padding: 20px; max-width: 1600px; margin: 0 auto; }}
    .head {{ margin-bottom: 18px; }}
    .head h1 {{ margin: 0 0 6px; font-size: 28px; }}
    .head p {{ margin: 0; color: #aeb8c4; }}
    .controls {{ margin: 14px 0 18px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    input {{ border: 1px solid rgba(255,255,255,0.14); border-radius: 10px; padding: 8px 12px; min-width: 280px; background: #303840; color: #edf2f7; }}
    .toggle {{
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 999px;
      padding: 8px 12px;
      background: #343c46;
      cursor: pointer;
      font-size: 13px;
      color: #d7dee7;
    }}
    .toggle.active {{
      background: #397faa;
      color: white;
      border-color: #62b6ed;
    }}
    table {{ width: 100%; border-collapse: collapse; background: #2a313a; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 28px rgba(0,0,0,0.24); }}
    thead {{ background: #343d47; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left; font-size: 14px; }}
    th {{ cursor: pointer; user-select: none; }}
    th.active {{ color: #62b6ed; }}
    th .sort-indicator {{ margin-left: 6px; color: #8995a3; font-size: 12px; }}
    th.active .sort-indicator {{ color: #62b6ed; }}
    thead th {{ position: sticky; top: 0; z-index: 1; background: #343d47; }}
    .drive-weight-col {{ background: rgba(98,182,237,0.08); color: #a7d8f5; font-weight: 700; }}
    thead th.drive-weight-col {{ background: #354957; }}
    tbody tr:hover {{ background: #333c46; }}
    a {{ color: #62b6ed; text-decoration: none; font-weight: 600; }}
    .muted {{ color: #8995a3; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <h1>Failure Index</h1>
      <p>{title}</p>
    </div>
    <div class="controls">
      <input id="search" type="search" placeholder="Filter rows">
      <button id="filter-replay" class="toggle" type="button">Replay Only</button>
      <button id="filter-failures" class="toggle" type="button">Any Failures</button>
      <button id="filter-genuine" class="toggle" type="button">Genuine Failures Only</button>
      <button id="filter-unavoidable" class="toggle" type="button">Unavoidable Only</button>
      <button id="filter-adversary-forced" class="toggle" type="button">Adversary-Forced Only</button>
      <button id="filter-offroad" class="toggle" type="button">Offroad Only</button>
      <button id="filter-at-fault" class="toggle" type="button">At-Fault Collision Only</button>
      <span class="muted" id="count"></span>
    </div>
    <table id="failure-table">
      <thead><tr><th data-key='rendered_html'>render<span class='sort-indicator'></span></th>{header_cells}</tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <script>
    const ROWS = {json.dumps(rows, separators=(",", ":"))};
    const COLS = {json.dumps(existing_columns)};
    const DRIVE_WEIGHT_COLS = new Set({json.dumps(sorted(drive_weight_columns))});
    const tbody = document.querySelector('#failure-table tbody');
    const count = document.getElementById('count');
    const search = document.getElementById('search');
    const replayFilter = document.getElementById('filter-replay');
    const failuresFilter = document.getElementById('filter-failures');
    const genuineFilter = document.getElementById('filter-genuine');
    const unavoidableFilter = document.getElementById('filter-unavoidable');
    const adversaryForcedFilter = document.getElementById('filter-adversary-forced');
    const offroadFilter = document.getElementById('filter-offroad');
    const atFaultFilter = document.getElementById('filter-at-fault');
    let sortKey = 'did_target_fail';
    let sortDir = -1;
    let replayOnly = false;
    let failuresOnly = false;
    let outcomeOnly = '';
    let offroadOnly = false;
    let atFaultOnly = false;

    function readStateFromUrl() {{
      const params = new URLSearchParams(window.location.search);
      sortKey = params.get('sort') || sortKey;
      sortDir = Number(params.get('dir') || sortDir) >= 0 ? 1 : -1;
      replayOnly = params.get('replay') === '1';
      failuresOnly = params.get('failures') === '1';
      outcomeOnly = params.get('outcome') || '';
      offroadOnly = params.get('offroad') === '1';
      atFaultOnly = params.get('atfault') === '1';
      search.value = params.get('q') || '';
    }}

    function stateParams() {{
      const params = new URLSearchParams();
      params.set('sort', sortKey);
      params.set('dir', String(sortDir));
      if (replayOnly) params.set('replay', '1');
      if (failuresOnly) params.set('failures', '1');
      if (outcomeOnly) params.set('outcome', outcomeOnly);
      if (offroadOnly) params.set('offroad', '1');
      if (atFaultOnly) params.set('atfault', '1');
      if (search.value) params.set('q', search.value);
      return params;
    }}

    function updateUrlState() {{
      const query = stateParams().toString();
      const nextUrl = query ? `${{window.location.pathname}}?${{query}}` : window.location.pathname;
      window.history.replaceState(null, '', nextUrl);
    }}

    function hrefWithState(href) {{
      if (!href) return href;
      const query = stateParams().toString();
      return query ? `${{href}}?${{query}}` : href;
    }}

    function metricEnabled(row, key) {{
      return Number((row && row[key]) || 0) > 0;
    }}

    function atFaultEnabled(row) {{
      return metricEnabled(row, 'did_target_have_at_fault_collision') || metricEnabled(row, 'target_hit_at_fault_rate');
    }}

    function compareValues(a, b, key, dir) {{
      if (key === 'rendered_html') {{
        const av = a.rendered_html ? 1 : 0;
        const bv = b.rendered_html ? 1 : 0;
        if (av === bv) return 0;
        return av > bv ? dir : -dir;
      }}

      const av = a[key];
      const bv = b[key];
      if (av === bv) return 0;
      if (av == null || av === '') return 1;
      if (bv == null || bv === '') return -1;

      const an = Number(av);
      const bn = Number(bv);
      const bothNumeric = !Number.isNaN(an) && !Number.isNaN(bn);
      if (bothNumeric) {{
        if (an === bn) return 0;
        return an > bn ? dir : -dir;
      }}

      const as = String(av).toLowerCase();
      const bs = String(bv).toLowerCase();
      const cmp = as.localeCompare(bs);
      return cmp === 0 ? 0 : cmp * dir;
    }}

    function renderTable() {{
      const term = (search.value || '').toLowerCase();
      const filtered = ROWS.filter(row => {{
        if (replayOnly && !row.rendered_html) return false;
        if (failuresOnly && !(Number(row.did_target_fail || 0) > 0)) return false;
        if (outcomeOnly === 'target_failure' &&
            !metricEnabled(row, 'target_collision_target_failure_rate')) return false;
        if (outcomeOnly === 'unavoidable' &&
            !metricEnabled(row, 'target_collision_unavoidable_rate')) return false;
        if (outcomeOnly === 'adversary_forced' &&
            !metricEnabled(row, 'target_collision_adversary_forced_rate')) return false;
        if (offroadOnly && !metricEnabled(row, 'did_target_offroad')) return false;
        if (atFaultOnly && !atFaultEnabled(row)) return false;
        return JSON.stringify(row).toLowerCase().includes(term);
      }});
      filtered.sort((a, b) => compareValues(a, b, sortKey, sortDir));
      tbody.innerHTML = filtered.map(row => {{
        const cells = COLS.map(col => `<td class="${{DRIVE_WEIGHT_COLS.has(col) ? 'drive-weight-col' : ''}}">${{row[col] == null ? '' : row[col]}}</td>`).join('');
        const link = row.rendered_html ? `<a href="${{hrefWithState(row.rendered_html)}}">open</a>` : '<span class="muted">n/a</span>';
        return `<tr><td>${{link}}</td>${{cells}}</tr>`;
      }}).join('');
      count.innerText = `${{filtered.length}} rows`;
      document.querySelectorAll('th[data-key]').forEach(th => {{
        const active = th.dataset.key === sortKey;
        th.classList.toggle('active', active);
        const indicator = th.querySelector('.sort-indicator');
        if (indicator) {{
          indicator.textContent = active ? (sortDir > 0 ? '▲' : '▼') : '';
        }}
      }});
      replayFilter.classList.toggle('active', replayOnly);
      failuresFilter.classList.toggle('active', failuresOnly);
      genuineFilter.classList.toggle('active', outcomeOnly === 'target_failure');
      unavoidableFilter.classList.toggle('active', outcomeOnly === 'unavoidable');
      adversaryForcedFilter.classList.toggle('active', outcomeOnly === 'adversary_forced');
      offroadFilter.classList.toggle('active', offroadOnly);
      atFaultFilter.classList.toggle('active', atFaultOnly);
      updateUrlState();
    }}

    document.querySelectorAll('th[data-key]').forEach(th => {{
      th.addEventListener('click', () => {{
        const key = th.dataset.key;
        if (sortKey === key) sortDir *= -1;
        else {{
          sortKey = key;
          sortDir = key === 'rendered_html' ? 1 : -1;
        }}
        renderTable();
      }});
    }});
    search.addEventListener('input', renderTable);
    replayFilter.addEventListener('click', () => {{
      replayOnly = !replayOnly;
      renderTable();
    }});
    failuresFilter.addEventListener('click', () => {{
      failuresOnly = !failuresOnly;
      renderTable();
    }});
    genuineFilter.addEventListener('click', () => {{
      outcomeOnly = outcomeOnly === 'target_failure' ? '' : 'target_failure';
      renderTable();
    }});
    unavoidableFilter.addEventListener('click', () => {{
      outcomeOnly = outcomeOnly === 'unavoidable' ? '' : 'unavoidable';
      renderTable();
    }});
    adversaryForcedFilter.addEventListener('click', () => {{
      outcomeOnly = outcomeOnly === 'adversary_forced' ? '' : 'adversary_forced';
      renderTable();
    }});
    offroadFilter.addEventListener('click', () => {{
      offroadOnly = !offroadOnly;
      renderTable();
    }});
    atFaultFilter.addEventListener('click', () => {{
      atFaultOnly = !atFaultOnly;
      renderTable();
    }});
    readStateFromUrl();
    renderTable();
  </script>
</body>
</html>"""
    with open(output_path, "w") as f:
        f.write(html)
    return output_path
