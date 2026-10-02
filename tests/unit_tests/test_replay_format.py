import dataclasses
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import pufferlib.viz
from pufferlib import pufferl, replay_format
from pufferlib.config_schema import normalize_puffer_drive_config
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = Path(__file__).resolve().parents[2]
CARLA_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/carla"
SEED = 7
CAPTURE_STEP_COUNT = 8
LIVE_SNAPSHOT_STEPS = (2, 6)
F16_UNIT_ROUNDOFF = 2.0**-11
FLOAT32_SLACK_M = 1e-3
EXACT_DECODE_TOLERANCE_M = 1e-9
PARTNER_HEADING_TOLERANCE_RAD = 2e-3
SIZE_SLACK_M = 1e-4
NEGATIVE_ZERO_SLOT = "negative_zero"
NON_DECODED_FILLER = 0.3125
REWARD_COEF_FILLER = -0.4375
LATTICE_MASK_FILLER = 1.0

HAND_NORM_M = {
    "xy_offset": 80.0,
    "road_seg_length": 4.0,
    "road_seg_width": 5.0,
    "veh_length": 12.0,
    "veh_width": 8.0,
    "goal_offset": 50.0,
}
HAND_EGO_LENGTH_M = 4.7
HAND_EGO_WIDTH_M = 2.1

SLOT_COUNTS = {
    "num_goals": 3,
    "obs_slots_partners_n": 3,
    "obs_slots_lane_n": 4,
    "obs_slots_boundary_n": 3,
    "obs_slots_traffic_controls_n": 2,
    "obs_dropout_lane": 0.0,
    "obs_dropout_boundary": 0.0,
}
DRIVE_VARIANTS = {
    "continuous": dict(SLOT_COUNTS, action_type="continuous", dynamics_model="jerk"),
    "discrete": dict(
        SLOT_COUNTS,
        action_type="discrete",
        dynamics_model="classic",
        num_goals=1,
        obs_slots_partners_n=2,
        obs_slots_lane_n=12,
        obs_dropout_lane=0.5,
        obs_slots_boundary_n=2,
        obs_slots_traffic_controls_n=1,
    ),
    "spline": dict(SLOT_COUNTS, action_type="spline", dynamics_model="spline"),
    "lattice_overtake": dict(
        SLOT_COUNTS, action_type="lattice", dynamics_model="spline_werling", lattice_oncoming_overtake=True
    ),
    "lattice_turnaround": dict(
        SLOT_COUNTS,
        action_type="lattice",
        dynamics_model="spline_werling",
        lattice_oncoming_overtake=True,
        lattice_turnaround=True,
    ),
    "reward_conditioning": dict(SLOT_COUNTS, action_type="continuous", dynamics_model="jerk", reward_conditioning=True),
}

# partner = (x_m, y_m, heading_rad, length_m, width_m); road = ((mid_x_m, mid_y_m), half_length_m, direction_rad)
DEFAULT_SCENE = {
    "goals": [None, (30.0, -10.0), (-5.0, 20.0)],
    "partners": [(12.0, 3.5, 0.4, 4.5, 2.0), None, (-20.0, -7.0, -2.5, 10.0, 3.0)],
    "lanes": [((10.0, 2.0), 2.5, 0.3), None, ((-3.0, -8.0), 1.0, 2.0), NEGATIVE_ZERO_SLOT],
    "boundaries": [((25.0, -6.0), 3.0, -1.2), ((0.5, 4.0), 0.75, 3.0), None],
    "traffic": [
        ((15.0, -2.0), (15.0, 2.0), binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT, binding.TRAFFIC_CONTROL_STATE_GREEN),
        None,
    ],
}
DISCRETE_SCENE = {
    "goals": [(8.0, 1.0)],
    "partners": [NEGATIVE_ZERO_SLOT, (5.0, -2.0, 1.0, 4.0, 1.8)],
    "lanes": [((1.0, 1.0), 2.0, 0.1), ((2.0, -1.0), 2.0, 0.2), None, None, ((40.0, 10.0), 5.0, -0.5), None],
    "boundaries": [None, ((3.0, 3.0), 1.0, 1.5)],
    "traffic": [
        ((-10.0, -1.0), (-10.0, 3.0), binding.TRAFFIC_CONTROL_TYPE_STOP_SIGN, binding.TRAFFIC_CONTROL_STATE_UNKNOWN)
    ],
}

CAPTURE_ENV_OVERRIDES = {
    "num_agents": 4,
    "min_agents_per_env": 4,
    "max_agents_per_env": 4,
    "num_maps": 1,
    "use_map_cache": False,
    "map_dir": str(CARLA_MAP_DIR),
    "action_type": "discrete",
    "dynamics_model": "classic",
    "scenario_length": 64,
    "resample_frequency": 64,
    "termination_mode": False,
    "collision_behavior": "ignore",
    "offroad_behavior": "ignore",
    "obs_slots_partners_n": 3,
    "obs_slots_lane_n": 12,
    "obs_slots_boundary_n": 12,
    "obs_slots_traffic_controls_n": 2,
    "obs_dropout_lane": 0.0,
    "obs_dropout_boundary": 0.0,
    "partner_blindness_prob": 0.0,
    "partner_blindness_trigger_prob": 0.0,
    "phantom_braking_prob": 0.0,
    "phantom_braking_trigger_prob": 0.0,
}
LIVE_ENV_OVERRIDES = dict(
    CAPTURE_ENV_OVERRIDES,
    num_agents=8,
    min_agents_per_env=8,
    max_agents_per_env=8,
    obs_slots_partners_n=4,
    obs_slots_lane_n=16,
    obs_slots_boundary_n=16,
)


def _normalized_env_config(overrides):
    with patch.object(sys, "argv", ["pufferl.py"]):
        args = pufferl.load_config("puffer_drive")
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None
    env_config = dict(normalize_puffer_drive_config(args, "test")["env"])
    env_config.update(overrides)
    return env_config


def _config_only_drive(variant_kwargs):
    return Drive(config_only=True, map_dir=str(CARLA_MAP_DIR), **variant_kwargs)


def _expected_counts(variant_kwargs):
    drive = _config_only_drive(variant_kwargs)
    action_type = variant_kwargs["action_type"]
    ego_features = binding.EGO_FEATURES
    lattice_mask_features = 0
    if action_type == "spline":
        ego_features += binding.SPLINE_INTENT_FEATURES
    if action_type == "lattice":
        ego_features += binding.LATTICE_PLAN_FEATURES + int(variant_kwargs.get("lattice_oncoming_overtake", False))
        ego_features += binding.LATTICE_TURN_PLAN_FEATURES * int(variant_kwargs.get("lattice_turnaround", False))
        lattice_mask_features = sum(drive.lattice_nvec)
    return {
        "ego_features": ego_features,
        "reward_coef_features": binding.NUM_REWARD_COEFS if variant_kwargs.get("reward_conditioning") else 0,
        "goal_count": variant_kwargs["num_goals"],
        "goal_features": binding.GOAL_FEATURES,
        "partner_count": variant_kwargs["obs_slots_partners_n"],
        "partner_features": binding.PARTNER_FEATURES,
        "lane_count": int(variant_kwargs["obs_slots_lane_n"] * (1.0 - variant_kwargs["obs_dropout_lane"])),
        "lane_features": binding.LANE_FEATURES,
        "boundary_count": int(variant_kwargs["obs_slots_boundary_n"] * (1.0 - variant_kwargs["obs_dropout_boundary"])),
        "boundary_features": binding.BOUNDARY_FEATURES,
        "traffic_count": variant_kwargs["obs_slots_traffic_controls_n"],
        "traffic_features": binding.TRAFFIC_CONTROL_FEATURES,
        "valid_count_features": binding.OBS_VALID_COUNT_FEATURES,
        "lattice_mask_features": lattice_mask_features,
    }


def _hand_header(counts):
    return {"obs_layout": replay_format.build_obs_layout(counts), "obs_norm_m": dict(HAND_NORM_M)}


def _slot_slice(start, slot_idx, feature_count):
    begin = start + slot_idx * feature_count
    return slice(begin, begin + feature_count)


def _write_slot(row, slot, features):
    if features is NEGATIVE_ZERO_SLOT:
        row[slot] = -0.0
        return 0
    if features is None:
        return 0
    row[slot] = features
    return 1


def _road_features(entry, norm, is_lane):
    if entry is None or entry is NEGATIVE_ZERO_SLOT:
        return entry
    (mid_x_m, mid_y_m), half_length_m, direction_rad = entry
    goal_distance_features = [0.625, -0.25] if is_lane else [0.0, 0.0]
    return [
        mid_x_m / norm["xy_offset"],
        mid_y_m / norm["xy_offset"],
        0.0625,
        half_length_m / norm["road_seg_length"],
        replay_format.REPLAY_VIEW_STYLE["lane_surface_width_m"] / norm["road_seg_width"],
        math.cos(direction_rad),
        math.sin(direction_rad),
        *goal_distance_features,
    ]


def _build_hand_row(layout, scene, norm):
    row = np.zeros(layout["obs_dim"], dtype=np.float64)
    row[: layout["ego_features"]] = NON_DECODED_FILLER
    row[1] = HAND_EGO_WIDTH_M / norm["veh_width"]
    row[2] = HAND_EGO_LENGTH_M / norm["veh_length"]
    coef_start = layout["ego_features"]
    row[coef_start : coef_start + layout["reward_coef_features"]] = REWARD_COEF_FILLER
    for goal_idx, goal in enumerate(scene["goals"][: layout["goal_count"]]):
        features = goal if goal is None else [goal[0] / norm["goal_offset"], goal[1] / norm["goal_offset"], 0.125]
        _write_slot(row, _slot_slice(layout["goal_start"], goal_idx, layout["goal_features"]), features)
    partner_written = 0
    for partner_idx, partner in enumerate(scene["partners"][: layout["partner_count"]]):
        features = partner
        if partner is not None and partner is not NEGATIVE_ZERO_SLOT:
            x_m, y_m, heading_rad, length_m, width_m = partner
            features = [
                x_m / norm["xy_offset"],
                y_m / norm["xy_offset"],
                0.03125,
                length_m / norm["veh_length"],
                width_m / norm["veh_width"],
                math.cos(heading_rad),
                math.sin(heading_rad),
                0.25,
                0.5,
            ]
        slot = _slot_slice(layout["partner_start"], partner_idx, layout["partner_features"])
        partner_written += _write_slot(row, slot, features)
    lane_written = 0
    for lane_idx, lane in enumerate(scene["lanes"][: layout["lane_count"]]):
        slot = _slot_slice(layout["lane_start"], lane_idx, layout["lane_features"])
        lane_written += _write_slot(row, slot, _road_features(lane, norm, is_lane=True))
    boundary_written = 0
    for boundary_idx, boundary in enumerate(scene["boundaries"][: layout["boundary_count"]]):
        slot = _slot_slice(layout["boundary_start"], boundary_idx, layout["boundary_features"])
        boundary_written += _write_slot(row, slot, _road_features(boundary, norm, is_lane=False))
    traffic_written = 0
    for traffic_idx, control in enumerate(scene["traffic"][: layout["traffic_count"]]):
        features = control
        if control is not None:
            (x1_m, y1_m), (x2_m, y2_m), control_type, control_state = control
            features = [
                x1_m / norm["xy_offset"],
                y1_m / norm["xy_offset"],
                x2_m / norm["xy_offset"],
                y2_m / norm["xy_offset"],
                0.015625,
                control_type,
                control_state,
            ]
        slot = _slot_slice(layout["traffic_start"], traffic_idx, layout["traffic_features"])
        traffic_written += _write_slot(row, slot, features)
    count_start = layout["valid_count_start"]
    row[count_start : count_start + 4] = [lane_written, boundary_written, partner_written, traffic_written]
    row[count_start + layout["valid_count_features"] :] = LATTICE_MASK_FILLER
    return row.astype(np.float16)


def _is_empty(features):
    return bool(np.all(features == 0.0))


def _reference_segments(row, start, count, feature_count, norm):
    segments = []
    for slot_idx in range(count):
        features = row[_slot_slice(start, slot_idx, feature_count)]
        if _is_empty(features):
            continue
        mid_m = features[0:2] * norm["xy_offset"]
        half_m = features[3] * norm["road_seg_length"]
        direction = features[5:7]
        segments.append([mid_m - direction * half_m, mid_m + direction * half_m])
    return np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)


def _reference_decode(row16, layout, norm):
    row = row16.astype(np.float64)
    goals = []
    for goal_idx in range(layout["goal_count"]):
        features = row[_slot_slice(layout["goal_start"], goal_idx, layout["goal_features"])]
        if not _is_empty(features):
            goals.append(features[0:2] * norm["goal_offset"])
    partner_slots, partner_xy, partner_heading, partner_length, partner_width = [], [], [], [], []
    for partner_idx in range(layout["partner_count"]):
        features = row[_slot_slice(layout["partner_start"], partner_idx, layout["partner_features"])]
        if _is_empty(features):
            continue
        partner_slots.append(partner_idx)
        partner_xy.append(features[0:2] * norm["xy_offset"])
        partner_heading.append(math.atan2(features[6], features[5]))
        partner_length.append(features[3] * norm["veh_length"])
        partner_width.append(features[4] * norm["veh_width"])
    stop_lines, stop_types, stop_states = [], [], []
    for traffic_idx in range(layout["traffic_count"]):
        features = row[_slot_slice(layout["traffic_start"], traffic_idx, layout["traffic_features"])]
        if _is_empty(features):
            continue
        stop_lines.append(np.array([[features[0], features[1]], [features[2], features[3]]]) * norm["xy_offset"])
        stop_types.append(int(round(features[5])))
        stop_states.append(int(round(features[6])))
    count_start = layout["valid_count_start"]
    return {
        "ego_length_m": row[2] * norm["veh_length"],
        "ego_width_m": row[1] * norm["veh_width"],
        "goal_xy_m": np.asarray(goals, dtype=np.float64).reshape(-1, 2),
        "partner_slot_idx": np.asarray(partner_slots, dtype=np.int64),
        "partner_xy_m": np.asarray(partner_xy, dtype=np.float64).reshape(-1, 2),
        "partner_heading_rad": np.asarray(partner_heading, dtype=np.float64),
        "partner_length_m": np.asarray(partner_length, dtype=np.float64),
        "partner_width_m": np.asarray(partner_width, dtype=np.float64),
        "lane_segment_xy_m": _reference_segments(
            row, layout["lane_start"], layout["lane_count"], layout["lane_features"], norm
        ),
        "boundary_segment_xy_m": _reference_segments(
            row, layout["boundary_start"], layout["boundary_count"], layout["boundary_features"], norm
        ),
        "stop_line_xy_m": np.asarray(stop_lines, dtype=np.float64).reshape(-1, 2, 2),
        "stop_line_type": np.asarray(stop_types, dtype=np.int64),
        "stop_line_state": np.asarray(stop_states, dtype=np.int64),
        "reported_counts": tuple(int(value) for value in row[count_start : count_start + 4]),
    }


def _scene_segments_m(entries):
    segments = []
    for entry in entries:
        if entry is None or entry is NEGATIVE_ZERO_SLOT:
            continue
        (mid_x_m, mid_y_m), half_length_m, direction_rad = entry
        direction = np.array([math.cos(direction_rad), math.sin(direction_rad)])
        mid_m = np.array([mid_x_m, mid_y_m])
        segments.append([mid_m - direction * half_length_m, mid_m + direction * half_length_m])
    return np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)


def _assert_float_array(actual, expected, tolerance):
    actual = np.asarray(actual)
    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=tolerance)


def _manual_payload(header, blob):
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = b"\0" * ((-(4 + len(header_bytes))) % 4)
    return zlib.compress(struct.pack("<I", len(header_bytes)) + header_bytes + padding + blob)


def _raw_header_and_blob(compressed):
    payload = zlib.decompress(compressed)
    header_length = struct.unpack_from("<I", payload)[0]
    header = json.loads(payload[4 : 4 + header_length])
    data_start = 4 + header_length + ((-(4 + header_length)) % 4)
    return header, payload[data_start:]


def test_format_constants_match_the_viewer_contract():
    assert replay_format.REPLAY_FORMAT_VERSION == 3
    assert (
        replay_format.AGENT_F32_X_IDX,
        replay_format.AGENT_F32_Y_IDX,
        replay_format.AGENT_F32_HEADING_IDX,
        replay_format.AGENT_F32_LENGTH_IDX,
        replay_format.AGENT_F32_WIDTH_IDX,
        replay_format.AGENT_F32_SPEED_IDX,
    ) == (0, 1, 3, 4, 5, 6)
    assert (
        replay_format.AGENT_I32_ID_IDX,
        replay_format.AGENT_I32_TYPE_IDX,
        replay_format.AGENT_I32_VALID_IDX,
        replay_format.AGENT_I32_ACTIVE_IDX,
        replay_format.AGENT_I32_REMOVED_IDX,
        replay_format.AGENT_I32_LANE_IDX,
        replay_format.AGENT_I32_SLOT_IDX,
        replay_format.AGENT_I32_BLIND_IDX,
        replay_format.AGENT_I32_BRAKING_IDX,
    ) == (0, 1, 2, 3, 5, 6, 7, 8, 9)
    assert replay_format.AGENT_I32_BRAKING_IDX < binding.AGENT_I32_FIELDS
    assert replay_format.AGENT_F32_SPEED_IDX < binding.AGENT_F32_FIELDS


def test_view_style_holds_the_camera_and_geometry_constants():
    style = replay_format.REPLAY_VIEW_STYLE
    expected_cameras = {
        "chase": {"back_m": 25, "eye_z_m": 15, "ahead_m": 40, "target_z_m": 1, "fovy_deg": 45, "draw_ego": True},
        "driver": {"back_m": 0, "eye_z_m": 1.4, "ahead_m": 30, "target_z_m": 1.0, "fovy_deg": 60, "draw_ego": False},
    }
    for camera_name, expected in expected_cameras.items():
        camera = style["cameras"][camera_name]
        for key, value in expected.items():
            assert camera[key] == value, f"{camera_name}.{key}"
    assert style["near_plane_m"] == 0.5
    assert style["cull_range_m"] == 160
    assert style["lane_surface_width_m"] == 3.7
    assert style["box_height_m"]["vehicle"] == 1.6
    assert style["box_height_m"]["pedestrian"] == 1.8
    assert style["box_height_m"]["cyclist"] == 1.7
    assert style["sort_chunk_length_m"] == 5
    assert style["goal_polygon_vertex_count"] == 32
    assert style["trail_seconds"] == 3
    assert style["logged_future_seconds"] == 5
    assert style["partner_match_tolerance_m"] == 1.0
    assert style["replay_frames_per_second"] == 10
    assert style["obs_panel_default_zoom"] == 2.2
    json.dumps(style)


def test_pack_then_decode_round_trips_exactly():
    header = {"frames": 3, "map_name": "Town01", "nested": {"values": [1, 2.5, None], "label": "ü"}, "flag": True}
    float32_values = (np.arange(30, dtype=np.float32).reshape(2, 3, 5) * np.float32(0.37)).copy()
    float32_values[0, 0, 0] = np.nan
    float32_values[1, 2, 4] = -0.0
    chunks = {
        "f32": float32_values,
        "i32": np.array([[-(2**31), 2**31 - 1, 0]], dtype=np.int32),
        "i16_odd": np.arange(-3, 4, dtype=np.int16),
        "u8_odd": np.arange(5, dtype=np.uint8),
        "f16": np.array([0.0, -0.0, 6e-8, 65504.0, -1.5], dtype=np.float16).reshape(5, 1),
        "i32_transposed": np.arange(6, dtype=np.int32).reshape(2, 3).T,
        "f32_single": np.array([3.25], dtype=np.float32),
    }
    compressed = pufferlib.viz._pack_replay_binary(header, chunks)
    raw_header, _ = _raw_header_and_blob(compressed)

    decoded_header, decoded_chunks = replay_format.decode_interactive_replay(compressed)

    assert decoded_header["chunks"] == raw_header["chunks"]
    assert {key: value for key, value in decoded_header.items() if key != "chunks"} == header
    assert list(decoded_chunks) == list(chunks)
    for name, original in chunks.items():
        expected = np.ascontiguousarray(original)
        decoded = decoded_chunks[name]
        assert decoded.dtype == expected.dtype, name
        assert decoded.shape == expected.shape, name
        assert decoded.tobytes() == expected.tobytes(), name
        assert not decoded.flags.writeable, name
        with pytest.raises(ValueError):
            decoded.reshape(-1)[0] = 0


def test_decode_accepts_a_payload_without_chunks():
    header, chunks = replay_format.decode_interactive_replay(pufferlib.viz._pack_replay_binary({"frames": 0}, {}))
    assert header == {"frames": 0, "chunks": {}}
    assert chunks == {}


def _valid_packed():
    return pufferlib.viz._pack_replay_binary(
        {"frames": 1}, {"u8": np.arange(5, dtype=np.uint8), "f32": np.ones(3, dtype=np.float32)}
    )


def _payload_with_suffix(suffix):
    return zlib.compress(zlib.decompress(_valid_packed()) + suffix)


MALFORMED_PAYLOADS = {
    "not_zlib": lambda: b"definitely not a zlib stream",
    "truncated_zlib": lambda: _valid_packed()[: len(_valid_packed()) // 2],
    "empty_payload": lambda: zlib.compress(b""),
    "header_length_overrun": lambda: zlib.compress(struct.pack("<I", 10_000) + b'{"chunks":{}}'),
    "unknown_dtype": lambda: _manual_payload(
        {"chunks": {"x": {"dtype": "float64", "shape": [2], "offset": 0, "nbytes": 16}}}, b"\0" * 16
    ),
    "nbytes_not_shape_times_itemsize": lambda: _manual_payload(
        {"chunks": {"x": {"dtype": "float32", "shape": [2, 3], "offset": 0, "nbytes": 20}}}, b"\0" * 24
    ),
    "chunk_past_end": lambda: _manual_payload(
        {"chunks": {"x": {"dtype": "float32", "shape": [4], "offset": 0, "nbytes": 16}}}, b"\0" * 8
    ),
    "chunk_negative_offset": lambda: _manual_payload(
        {"chunks": {"x": {"dtype": "float32", "shape": [1], "offset": -4, "nbytes": 4}}}, b"\0" * 8
    ),
    "trailing_zero_bytes": lambda: _payload_with_suffix(b"\0" * 4),
    "trailing_garbage": lambda: _payload_with_suffix(b"junkjunk"),
}


@pytest.mark.parametrize("malformed_name", sorted(MALFORMED_PAYLOADS))
def test_decode_rejects_malformed_payloads(malformed_name):
    with pytest.raises(ValueError):
        replay_format.decode_interactive_replay(MALFORMED_PAYLOADS[malformed_name]())


def test_build_obs_layout_places_blocks_in_observation_order():
    counts = {
        "ego_features": 16,
        "reward_coef_features": 19,
        "goal_count": 3,
        "goal_features": 3,
        "partner_count": 2,
        "partner_features": 9,
        "lane_count": 8,
        "lane_features": 9,
        "boundary_count": 5,
        "boundary_features": 9,
        "traffic_count": 1,
        "traffic_features": 7,
        "valid_count_features": 4,
        "lattice_mask_features": 11,
    }
    original = dict(counts)

    layout = replay_format.build_obs_layout(counts)

    assert counts == original
    assert {key: layout[key] for key in counts} == counts
    assert layout["goal_start"] == 35
    assert layout["partner_start"] == 44
    assert layout["lane_start"] == 62
    assert layout["boundary_start"] == 134
    assert layout["traffic_start"] == 179
    assert layout["valid_count_start"] == 186
    assert layout["obs_dim"] == 201


@pytest.mark.parametrize("variant_name", sorted(DRIVE_VARIANTS))
def test_drive_observation_layout_reports_the_env_counts(variant_name):
    variant_kwargs = DRIVE_VARIANTS[variant_name]
    drive = _config_only_drive(variant_kwargs)
    expected = _expected_counts(variant_kwargs)

    layout_counts = drive.observation_layout()

    assert {key: layout_counts[key] for key in expected} == expected
    assert replay_format.build_obs_layout(layout_counts)["obs_dim"] == drive.num_obs


def test_obs_norm_reads_required_env_keys():
    env_config = {
        "obs_norm_xy_offset_m": 200.0,
        "obs_norm_road_seg_length_m": 10.0,
        "obs_norm_road_seg_width_m": 5.0,
        "obs_norm_veh_length_m": 10.0,
        "obs_norm_veh_width_m": 4.0,
        "obs_norm_goal_offset_m": 150.0,
        "obs_norm_z_m": 10.0,
    }
    assert replay_format.obs_norm_from_env_config(env_config) == {
        "xy_offset": 200.0,
        "road_seg_length": 10.0,
        "road_seg_width": 5.0,
        "veh_length": 10.0,
        "veh_width": 4.0,
        "goal_offset": 150.0,
        "z": 10.0,
    }
    for missing_key in env_config:
        with pytest.raises(KeyError):
            replay_format.obs_norm_from_env_config({k: v for k, v in env_config.items() if k != missing_key})


@pytest.mark.parametrize("variant_name", sorted(DRIVE_VARIANTS))
def test_decode_observation_recovers_hand_built_geometry(variant_name):
    counts = _expected_counts(DRIVE_VARIANTS[variant_name])
    header = _hand_header(counts)
    layout = header["obs_layout"]
    scene = DISCRETE_SCENE if variant_name == "discrete" else DEFAULT_SCENE
    row16 = _build_hand_row(layout, scene, HAND_NORM_M)
    reference = _reference_decode(row16, layout, HAND_NORM_M)

    decoded = replay_format.decode_observation(header, row16)

    assert isinstance(decoded, replay_format.DecodedObservation)
    assert decoded.ego_length_m == pytest.approx(reference["ego_length_m"], abs=EXACT_DECODE_TOLERANCE_M)
    assert decoded.ego_width_m == pytest.approx(reference["ego_width_m"], abs=EXACT_DECODE_TOLERANCE_M)
    for field_name in (
        "goal_xy_m",
        "partner_xy_m",
        "partner_heading_rad",
        "partner_length_m",
        "partner_width_m",
        "lane_segment_xy_m",
        "boundary_segment_xy_m",
        "stop_line_xy_m",
    ):
        _assert_float_array(getattr(decoded, field_name), reference[field_name], EXACT_DECODE_TOLERANCE_M)
    for field_name in ("partner_slot_idx", "stop_line_type", "stop_line_state"):
        actual = np.asarray(getattr(decoded, field_name))
        assert np.issubdtype(actual.dtype, np.integer), field_name
        np.testing.assert_array_equal(actual, reference[field_name])
    assert tuple(decoded.reported_counts) == reference["reported_counts"]
    assert all(isinstance(count, int) for count in decoded.reported_counts)

    assert decoded.reported_counts == (
        len(decoded.lane_segment_xy_m),
        len(decoded.boundary_segment_xy_m),
        len(decoded.partner_xy_m),
        len(decoded.stop_line_xy_m),
    )
    float16_bound_m = math.sqrt(2.0) * F16_UNIT_ROUNDOFF * HAND_NORM_M["xy_offset"] + FLOAT32_SLACK_M
    expected_partners = [p for p in scene["partners"][: counts["partner_count"]] if isinstance(p, tuple)]
    np.testing.assert_allclose(
        decoded.partner_xy_m, np.array([p[:2] for p in expected_partners]).reshape(-1, 2), atol=float16_bound_m
    )
    np.testing.assert_allclose(
        decoded.partner_heading_rad, [p[2] for p in expected_partners], atol=PARTNER_HEADING_TOLERANCE_RAD
    )
    np.testing.assert_allclose(
        decoded.lane_segment_xy_m, _scene_segments_m(scene["lanes"][: counts["lane_count"]]), atol=float16_bound_m
    )
    np.testing.assert_allclose(
        decoded.boundary_segment_xy_m,
        _scene_segments_m(scene["boundaries"][: counts["boundary_count"]]),
        atol=float16_bound_m,
    )
    expected_goals = [goal for goal in scene["goals"][: counts["goal_count"]] if goal is not None]
    goal_bound_m = math.sqrt(2.0) * F16_UNIT_ROUNDOFF * HAND_NORM_M["goal_offset"] + FLOAT32_SLACK_M
    np.testing.assert_allclose(decoded.goal_xy_m, np.array(expected_goals).reshape(-1, 2), atol=goal_bound_m)
    assert decoded.ego_length_m == pytest.approx(HAND_EGO_LENGTH_M, abs=F16_UNIT_ROUNDOFF * HAND_EGO_LENGTH_M)
    assert decoded.ego_width_m == pytest.approx(HAND_EGO_WIDTH_M, abs=F16_UNIT_ROUNDOFF * HAND_EGO_WIDTH_M)


def test_decode_observation_skips_slots_that_are_negative_zero():
    counts = _expected_counts(DRIVE_VARIANTS["discrete"])
    header = _hand_header(counts)
    row16 = _build_hand_row(header["obs_layout"], DISCRETE_SCENE, HAND_NORM_M)
    partner_slot = _slot_slice(header["obs_layout"]["partner_start"], 0, counts["partner_features"])
    assert np.all(np.signbit(row16[partner_slot]))

    decoded = replay_format.decode_observation(header, row16)

    np.testing.assert_array_equal(decoded.partner_slot_idx, [1])


def test_decode_observation_of_an_empty_row_has_empty_shapes():
    counts = _expected_counts(DRIVE_VARIANTS["continuous"])
    header = _hand_header(counts)
    row16 = np.zeros(header["obs_layout"]["obs_dim"], dtype=np.float16)

    decoded = replay_format.decode_observation(header, row16)

    assert decoded.goal_xy_m.shape == (0, 2)
    assert decoded.partner_slot_idx.shape == (0,)
    assert decoded.partner_xy_m.shape == (0, 2)
    assert decoded.partner_heading_rad.shape == (0,)
    assert decoded.lane_segment_xy_m.shape == (0, 2, 2)
    assert decoded.boundary_segment_xy_m.shape == (0, 2, 2)
    assert decoded.stop_line_xy_m.shape == (0, 2, 2)
    assert decoded.stop_line_type.shape == (0,)
    assert tuple(decoded.reported_counts) == (0, 0, 0, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        decoded.ego_length_m = 1.0


def test_ego_frame_to_world_rotates_forward_and_left():
    forward_and_left = np.array([[1.0, 0.0], [0.0, 1.0]])

    east = replay_format.ego_frame_to_world(forward_and_left, 10.0, -5.0, 0.0)
    north = replay_format.ego_frame_to_world(forward_and_left, 10.0, -5.0, math.pi / 2)

    assert east.dtype == np.float64
    np.testing.assert_allclose(east, [[11.0, -5.0], [10.0, -4.0]], atol=1e-12)
    np.testing.assert_allclose(north, [[10.0, -4.0], [9.0, -5.0]], atol=1e-12)


def test_ego_frame_to_world_inverts_the_ego_projection_on_nested_arrays():
    rng = np.random.default_rng(SEED)
    world_points = rng.uniform(-300.0, 300.0, size=(2, 3, 2))
    ego_x_m, ego_y_m, ego_heading_rad = 123.4, -56.7, -2.3
    cos_heading, sin_heading = math.cos(ego_heading_rad), math.sin(ego_heading_rad)
    delta = world_points - np.array([ego_x_m, ego_y_m])
    ego_points = np.stack(
        [
            cos_heading * delta[..., 0] + sin_heading * delta[..., 1],
            -sin_heading * delta[..., 0] + cos_heading * delta[..., 1],
        ],
        axis=-1,
    )

    recovered = replay_format.ego_frame_to_world(ego_points, ego_x_m, ego_y_m, ego_heading_rad)

    assert recovered.shape == world_points.shape
    np.testing.assert_allclose(recovered, world_points, atol=1e-9)


def _capture_replay(env_overrides, step_count, seed):
    env_config = _normalized_env_config(env_overrides)
    env = Drive(**dict(env_config, capture_replay=True))
    rng = np.random.default_rng(seed)
    try:
        env.reset(seed=seed)
        observation_history = []
        for _ in range(step_count):
            observation_history.append(env.observations.copy())
            env.step(rng.integers(0, env.single_action_space.n, size=env.num_agents))
        capture = env._replay_captures[0]
        layout_counts = env.observation_layout()
    finally:
        env.close()
    frames = {key: np.stack(values, axis=0) for key, values in capture["frames"].items()}
    active_count = capture["metadata"]["active_agent_count"]
    active_offset = capture["metadata"]["active_agent_offset"]
    frame_count = frames["agent_f32"].shape[0]
    replay = {
        "env": env_config,
        **frames,
        "raw_action": np.zeros((frame_count, active_count), np.float32),
        "clipped_action": np.zeros((frame_count, active_count), np.float32),
        "value": np.zeros((frame_count, active_count), np.float32),
        "entropy": np.zeros((frame_count, active_count), np.float32),
        "obs": np.stack(observation_history)[:, active_offset : active_offset + active_count].astype(np.float16),
        "obs_layout": layout_counts,
    }
    return capture["scenario"], replay


@pytest.fixture(scope="module")
def captured_replay():
    return _capture_replay(CAPTURE_ENV_OVERRIDES, CAPTURE_STEP_COUNT, SEED)


def _finite_float16_patterns(shape, seed):
    bits = np.random.default_rng(seed).integers(0, 2**16, size=shape, dtype=np.uint32).astype(np.uint16)
    exponent_all_ones = (bits & 0x7C00) == 0x7C00
    bits[exponent_all_ones] &= 0x83FF
    flat_bits = bits.reshape(-1)
    flat_bits[:4] = [0x8000, 0x0001, 0x7BFF, 0x0000]
    return bits.view(np.float16)


def test_encode_stores_float16_observations_bit_exactly(captured_replay):
    scenario, replay = captured_replay
    replay = dict(replay, obs=_finite_float16_patterns(replay["obs"].shape, SEED))

    header, chunks = replay_format.decode_interactive_replay(pufferlib.viz.encode_interactive_replay(scenario, replay))

    assert header["chunks"]["obs"]["dtype"] == "float16"
    assert chunks["obs"].dtype == np.float16
    assert chunks["obs"].shape == replay["obs"].shape
    np.testing.assert_array_equal(chunks["obs"].view(np.uint16), replay["obs"].view(np.uint16))


def test_encode_writes_the_observation_header(captured_replay):
    scenario, replay = captured_replay
    env_config = replay["env"]
    full_layout = replay_format.build_obs_layout(replay["obs_layout"])

    header, chunks = replay_format.decode_interactive_replay(pufferlib.viz.encode_interactive_replay(scenario, replay))

    assert header["replay_format_version"] == replay_format.REPLAY_FORMAT_VERSION
    assert header["obs_layout"] == full_layout
    assert header["obs_dim"] == full_layout["obs_dim"] == replay["obs"].shape[-1]
    assert header["ego_dim"] == binding.EGO_FEATURES
    assert header["obs_norm_m"] == replay_format.obs_norm_from_env_config(env_config)
    assert header["obs_norm_m"]["xy_offset"] == env_config["obs_norm_xy_offset_m"]
    assert header["obs_range_m"] == {
        "road_front": env_config["obs_range_road_front_m"],
        "road_behind": env_config["obs_range_road_behind_m"],
        "road_side": env_config["obs_range_road_side_m"],
        "partner": env_config["obs_range_partner_m"],
        "traffic_control": env_config["obs_range_traffic_control_m"],
    }
    assert header["dt"] == env_config["dt"]
    assert header["frames"] == CAPTURE_STEP_COUNT
    np.testing.assert_array_equal(chunks["agent_f32"], replay["agent_f32"])
    frame_idx, slot_idx = CAPTURE_STEP_COUNT - 1, 1
    from_chunk = replay_format.decode_observation(header, chunks["obs"][frame_idx, slot_idx])
    from_capture = replay_format.decode_observation(header, replay["obs"][frame_idx, slot_idx])
    np.testing.assert_array_equal(from_chunk.lane_segment_xy_m, from_capture.lane_segment_xy_m)
    np.testing.assert_array_equal(from_chunk.partner_xy_m, from_capture.partner_xy_m)


def test_encode_without_observations_has_no_obs_chunk(captured_replay):
    scenario, replay = captured_replay
    replay = {key: value for key, value in replay.items() if key not in ("obs", "obs_layout")}

    header, chunks = replay_format.decode_interactive_replay(pufferlib.viz.encode_interactive_replay(scenario, replay))

    assert "obs" not in chunks
    assert header["obs_dim"] == 0
    assert header["replay_format_version"] == replay_format.REPLAY_FORMAT_VERSION


def test_encode_requires_the_obs_layout_with_observations(captured_replay):
    scenario, replay = captured_replay
    replay = {key: value for key, value in replay.items() if key != "obs_layout"}
    with pytest.raises((KeyError, ValueError)):
        pufferlib.viz.encode_interactive_replay(scenario, replay)


def test_encode_rejects_an_obs_layout_that_does_not_match_the_observations(captured_replay):
    scenario, replay = captured_replay
    wrong_layout = dict(replay["obs_layout"], partner_count=replay["obs_layout"]["partner_count"] + 1)
    with pytest.raises(ValueError):
        pufferlib.viz.encode_interactive_replay(scenario, dict(replay, obs_layout=wrong_layout))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "negative_inf"])
def test_encode_rejects_non_finite_observations(captured_replay, bad_value):
    scenario, replay = captured_replay
    observations = replay["obs"].copy()
    observations[CAPTURE_STEP_COUNT // 2, 1, 5] = bad_value
    with pytest.raises(ValueError):
        pufferlib.viz.encode_interactive_replay(scenario, dict(replay, obs=observations))


@pytest.mark.parametrize(
    "action_type, env_overrides, extra_ego_features",
    [
        ("spline", {"action_type": "spline", "dynamics_model": "spline"}, binding.SPLINE_INTENT_FEATURES),
        (
            "lattice",
            {"action_type": "lattice", "dynamics_model": "spline_werling", "lattice_oncoming_overtake": True},
            binding.LATTICE_PLAN_FEATURES + 1,
        ),
        (
            "lattice",
            {
                "action_type": "lattice",
                "dynamics_model": "spline_werling",
                "lattice_oncoming_overtake": True,
                "lattice_turnaround": True,
            },
            binding.LATTICE_PLAN_FEATURES + 1 + binding.LATTICE_TURN_PLAN_FEATURES,
        ),
    ],
)
def test_encode_header_ego_dim_counts_the_action_blocks(
    captured_replay, action_type, env_overrides, extra_ego_features
):
    scenario, replay = captured_replay
    ego_features = binding.EGO_FEATURES + extra_ego_features
    layout_counts = dict(replay["obs_layout"], ego_features=ego_features)
    if action_type == "lattice":
        layout_counts["lattice_mask_features"] = sum(
            _config_only_drive(dict(SLOT_COUNTS, **env_overrides)).lattice_nvec
        )
    obs_dim = replay_format.build_obs_layout(layout_counts)["obs_dim"]
    frame_count, active_count = replay["obs"].shape[:2]
    observations = np.full((frame_count, active_count, obs_dim), 0.25, dtype=np.float16)
    replay = dict(replay, env=dict(replay["env"], **env_overrides), obs=observations, obs_layout=layout_counts)

    header, _ = replay_format.decode_interactive_replay(pufferlib.viz.encode_interactive_replay(scenario, replay))

    assert header["ego_dim"] == ego_features
    assert header["obs_layout"]["ego_features"] == ego_features
    assert header["obs_dim"] == obs_dim


@pytest.fixture(scope="module")
def live_observation_snapshots():
    env_config = _normalized_env_config(LIVE_ENV_OVERRIDES)
    env = Drive(**env_config)
    rng = np.random.default_rng(SEED)
    snapshots = []
    try:
        env.reset(seed=SEED)
        header = {
            "obs_layout": replay_format.build_obs_layout(env.observation_layout()),
            "obs_norm_m": replay_format.obs_norm_from_env_config(env_config),
        }
        assert header["obs_layout"]["obs_dim"] == env.num_obs
        for step_idx in range(1, max(LIVE_SNAPSHOT_STEPS) + 1):
            env.step(rng.integers(0, env.single_action_space.n, size=env.num_agents))
            if step_idx not in LIVE_SNAPSHOT_STEPS:
                continue
            state = env.get_state()
            scenario = state[0] if isinstance(state, list) else state
            snapshots.append(
                {
                    "scenario": scenario,
                    "observations_f32": env.observations.copy(),
                    "observations_f16": env.observations.astype(np.float16),
                }
            )
    finally:
        env.close()
    return {"env_config": env_config, "header": header, "snapshots": snapshots}


def _agent_snapshot_rows(live):
    for snapshot in live["snapshots"]:
        scenario = snapshot["scenario"]
        for slot_idx, agent_idx in enumerate(scenario["active_agent_indices"]):
            ego = scenario["agents"][agent_idx]
            decoded = replay_format.decode_observation(live["header"], snapshot["observations_f16"][slot_idx])
            yield scenario, agent_idx, ego, snapshot["observations_f16"][slot_idx].astype(np.float64), decoded


def _to_world(ego, points_m):
    return replay_format.ego_frame_to_world(points_m, ego["sim_x"], ego["sim_y"], ego["sim_heading"])


def _wrap_angle(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _map_segments(scenario, type_low, type_high):
    segments = []
    for road in scenario["road_elements"]:
        if not type_low <= road["type"] <= type_high:
            continue
        xs = np.asarray(road["x"], dtype=np.float64)
        ys = np.asarray(road["y"], dtype=np.float64)
        points = np.stack([xs, ys], axis=-1)
        segments.append(np.stack([points[:-1], points[1:]], axis=1))
    return np.concatenate(segments, axis=0)


def test_live_partner_decode_matches_the_true_agents(live_observation_snapshots):
    live = live_observation_snapshots
    layout = live["header"]["obs_layout"]
    norm = live["header"]["obs_norm_m"]
    partner_range_m = live["env_config"]["obs_range_partner_m"]
    checked_partner_count = 0
    for scenario, agent_idx, ego, row, decoded in _agent_snapshot_rows(live):
        agents = scenario["agents"]
        others = [other_idx for other_idx in range(len(agents)) if other_idx != agent_idx]
        in_range = [
            other_idx
            for other_idx in others
            if math.dist(
                (agents[other_idx]["sim_x"], agents[other_idx]["sim_y"], agents[other_idx]["sim_z"]),
                (ego["sim_x"], ego["sim_y"], ego["sim_z"]),
            )
            <= partner_range_m
        ]
        assert len(decoded.partner_xy_m) == min(len(in_range), layout["partner_count"])
        assert decoded.reported_counts[2] == len(decoded.partner_xy_m)
        partner_world = _to_world(ego, decoded.partner_xy_m)
        for partner_idx, slot_idx in enumerate(decoded.partner_slot_idx):
            features = row[_slot_slice(layout["partner_start"], int(slot_idx), layout["partner_features"])]
            bound_m = math.sqrt(2.0) * F16_UNIT_ROUNDOFF * np.max(np.abs(features[0:2])) * norm["xy_offset"]
            distances = [
                math.hypot(
                    partner_world[partner_idx, 0] - agents[o]["sim_x"],
                    partner_world[partner_idx, 1] - agents[o]["sim_y"],
                )
                for o in others
            ]
            nearest = others[int(np.argmin(distances))]
            assert min(distances) <= bound_m + FLOAT32_SLACK_M
            world_heading = ego["sim_heading"] + decoded.partner_heading_rad[partner_idx]
            assert abs(_wrap_angle(world_heading - agents[nearest]["sim_heading"])) <= PARTNER_HEADING_TOLERANCE_RAD
            length_bound_m = F16_UNIT_ROUNDOFF * abs(features[3]) * norm["veh_length"] + SIZE_SLACK_M
            width_bound_m = F16_UNIT_ROUNDOFF * abs(features[4]) * norm["veh_width"] + SIZE_SLACK_M
            assert abs(decoded.partner_length_m[partner_idx] - agents[nearest]["sim_length"]) <= length_bound_m
            assert abs(decoded.partner_width_m[partner_idx] - agents[nearest]["sim_width"]) <= width_bound_m
            checked_partner_count += 1
    assert checked_partner_count > 0


@pytest.mark.parametrize(
    "segment_field, start_key, count_key, type_range, reported_idx",
    [
        ("lane_segment_xy_m", "lane_start", "lane_count", (0, 9), 0),
        ("boundary_segment_xy_m", "boundary_start", "boundary_count", (20, 29), 1),
    ],
    ids=["lanes", "boundaries"],
)
def test_live_road_segment_decode_matches_map_segments(
    live_observation_snapshots, segment_field, start_key, count_key, type_range, reported_idx
):
    live = live_observation_snapshots
    layout = live["header"]["obs_layout"]
    norm = live["header"]["obs_norm_m"]
    feature_count = layout["lane_features"] if count_key == "lane_count" else layout["boundary_features"]
    map_segments = _map_segments(live["snapshots"][0]["scenario"], *type_range)
    checked_segment_count = 0
    for _, _, ego, row, decoded in _agent_snapshot_rows(live):
        segments = getattr(decoded, segment_field)
        assert decoded.reported_counts[reported_idx] == len(segments)
        world_segments = _to_world(ego, segments)
        non_empty_slots = [
            slot_idx
            for slot_idx in range(layout[count_key])
            if not _is_empty(row[_slot_slice(layout[start_key], slot_idx, feature_count)])
        ]
        assert len(non_empty_slots) == len(segments)
        for segment_idx, slot_idx in enumerate(non_empty_slots):
            features = row[_slot_slice(layout[start_key], slot_idx, feature_count)]
            half_length_m = abs(features[3]) * norm["road_seg_length"]
            bound_m = (
                math.sqrt(2.0) * F16_UNIT_ROUNDOFF * np.max(np.abs(features[0:2])) * norm["xy_offset"]
                + 2.0 * F16_UNIT_ROUNDOFF * half_length_m
                + FLOAT32_SLACK_M
            )
            endpoint_error_m = np.max(np.linalg.norm(map_segments - world_segments[segment_idx][None], axis=-1), axis=1)
            assert endpoint_error_m.min() <= bound_m
            checked_segment_count += 1
    assert checked_segment_count > 0


def test_live_stop_line_decode_matches_traffic_elements(live_observation_snapshots):
    live = live_observation_snapshots
    layout = live["header"]["obs_layout"]
    norm = live["header"]["obs_norm_m"]
    checked_stop_line_count = 0
    for scenario, _, ego, row, decoded in _agent_snapshot_rows(live):
        assert decoded.reported_counts[3] == len(decoded.stop_line_xy_m)
        traffic = scenario["traffic_elements"]
        true_lines = np.array(
            [[[t["stop_line"][0], t["stop_line"][1]], [t["stop_line"][3], t["stop_line"][4]]] for t in traffic]
        )
        true_types = np.array([t["type"] for t in traffic])
        world_lines = _to_world(ego, decoded.stop_line_xy_m)
        non_empty_slots = [
            slot_idx
            for slot_idx in range(layout["traffic_count"])
            if not _is_empty(row[_slot_slice(layout["traffic_start"], slot_idx, layout["traffic_features"])])
        ]
        for line_idx, slot_idx in enumerate(non_empty_slots):
            features = row[_slot_slice(layout["traffic_start"], slot_idx, layout["traffic_features"])]
            bound_m = math.sqrt(2.0) * F16_UNIT_ROUNDOFF * np.max(np.abs(features[0:4])) * norm["xy_offset"]
            endpoint_error_m = np.max(np.linalg.norm(true_lines - world_lines[line_idx][None], axis=-1), axis=1)
            matched = int(np.argmin(endpoint_error_m))
            assert endpoint_error_m[matched] <= bound_m + FLOAT32_SLACK_M
            assert decoded.stop_line_type[line_idx] == true_types[matched]
            checked_stop_line_count += 1
    assert checked_stop_line_count > 0


def test_live_goal_and_ego_size_decode(live_observation_snapshots):
    live = live_observation_snapshots
    norm = live["header"]["obs_norm_m"]
    checked_goal_count = 0
    for _, _, ego, row, decoded in _agent_snapshot_rows(live):
        assert (
            abs(decoded.ego_length_m - ego["sim_length"])
            <= F16_UNIT_ROUNDOFF * abs(row[2]) * norm["veh_length"] + SIZE_SLACK_M
        )
        assert (
            abs(decoded.ego_width_m - ego["sim_width"])
            <= F16_UNIT_ROUNDOFF * abs(row[1]) * norm["veh_width"] + SIZE_SLACK_M
        )
        if len(decoded.goal_xy_m) == 0:
            continue
        goal_world = _to_world(ego, decoded.goal_xy_m[0])
        bound_m = math.sqrt(2.0) * F16_UNIT_ROUNDOFF * np.max(np.abs(decoded.goal_xy_m[0])) + FLOAT32_SLACK_M
        assert math.hypot(goal_world[0] - ego["current_goal_x"], goal_world[1] - ego["current_goal_y"]) <= bound_m
        checked_goal_count += 1
    assert checked_goal_count > 0


def test_lane_surface_width_matches_the_observed_lane_width_feature(live_observation_snapshots):
    live = live_observation_snapshots
    layout = live["header"]["obs_layout"]
    observations = live["snapshots"][0]["observations_f32"]
    lane_width_features = observations[:, layout["lane_start"] + 4]
    observed = lane_width_features[lane_width_features != 0.0]
    assert observed.size > 0
    road_seg_width_m = live["env_config"]["obs_norm_road_seg_width_m"]
    np.testing.assert_allclose(
        observed.astype(np.float64) * road_seg_width_m,
        replay_format.REPLAY_VIEW_STYLE["lane_surface_width_m"],
        rtol=1e-6,
    )
