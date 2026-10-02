"""Interactive replay payload format, observation decoding and the view style shared by the HTML viewer and videos."""

import json
import struct
import zlib
from dataclasses import dataclass

import numpy as np

REPLAY_FORMAT_VERSION = 3
REPLAY_HEADER_LENGTH_BYTES = 4
REPLAY_ALIGNMENT_BYTES = 4
REPLAY_DTYPES = {
    "float32": np.dtype(np.float32),
    "int32": np.dtype(np.int32),
    "int16": np.dtype(np.int16),
    "uint8": np.dtype(np.uint8),
    "float16": np.dtype(np.float16),
}
REPLAY_DTYPE_NAMES = {dtype: name for name, dtype in REPLAY_DTYPES.items()}

AGENT_F32_X_IDX = 0
AGENT_F32_Y_IDX = 1
AGENT_F32_Z_IDX = 2
AGENT_F32_HEADING_IDX = 3
AGENT_F32_LENGTH_IDX = 4
AGENT_F32_WIDTH_IDX = 5
AGENT_F32_SPEED_IDX = 6
AGENT_I32_ID_IDX = 0
AGENT_I32_TYPE_IDX = 1
AGENT_I32_VALID_IDX = 2
AGENT_I32_ACTIVE_IDX = 3
AGENT_I32_REMOVED_IDX = 5
AGENT_I32_LANE_IDX = 6
AGENT_I32_SLOT_IDX = 7
AGENT_I32_BLIND_IDX = 8
AGENT_I32_BRAKING_IDX = 9

AGENT_TYPE_VEHICLE = 1
AGENT_TYPE_PEDESTRIAN = 2
AGENT_TYPE_CYCLIST = 3
TRAFFIC_TYPE_LIGHT = 1
TRAFFIC_TYPE_STOP_SIGN = 2

OBS_LAYOUT_COUNT_KEYS = (
    "ego_features",
    "reward_coef_features",
    "goal_count",
    "goal_features",
    "partner_count",
    "partner_features",
    "lane_count",
    "lane_features",
    "boundary_count",
    "boundary_features",
    "traffic_count",
    "traffic_features",
    "valid_count_features",
    "lattice_mask_features",
)

REPLAY_VIEW_STYLE = {
    "vehicle_colors": [
        "#681D00",
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#9467BD",
        "#8C564B",
        "#D47CBA",
        "#BCBD22",
        "#17BECF",
        "#AEC7E8",
        "#FFBB78",
        "#98DF8A",
        "#FF9896",
        "#C5B0D5",
        "#C49C94",
        "#F7B6D2",
        "#DBDB8D",
        "#9EDAE5",
    ],
    "agent_colors": {"dynamic_expert": "#c4c8cf", "static": "#4a505a", "infraction": "#d92d20"},
    "infraction_metric_count": 4,
    "map_colors": {
        "light": {"background": "#e9ebee", "road": "#c6cad1", "line": "#959ca8", "edge": "#2a2e35"},
        "dark": {"background": "#0d0f12", "road": "#363b43", "line": "#5d6573", "edge": "#06070a"},
    },
    "agent_outline_color": "#111111",
    "follow_ring_color": "#0a66d0",
    "traffic_state_colors": {"1": "#ff0000", "2": "#ffff00", "3": "#00ff00"},
    "traffic_default_color": "#888888",
    "stop_sign_color": "#ff0000",
    "yield_sign_color": "#ffd700",
    "goal_color": "#38bdf8",
    "goal_fill_alpha": 0.22,
    "predicted_path_color": "#1976d2",
    "predicted_path_alpha": 0.55,
    "logged_future_color": "#ff0000",
    "ghost_fill_alpha": 0.22,
    "logged_future_alpha": 0.7,
    "trail_alpha": 0.6,
    "bev_colors": {
        "background": "#ffffff",
        "lane": "#bbbbbb",
        "boundary": "#333333",
        "partner_fill": "#888888",
        "partner_outline": "#333333",
        "goal": "#ff00ff",
        "ego": "#0066ff",
    },
    "agent_view_colors": {
        "sky": "#cfe0f0",
        "ground": "#dcdfd6",
        "road": "#9aa1ab",
        "line": "#f2f2f2",
        "edge": "#2a2e35",
        "box_outline": "#1a1d22",
    },
    "agent_view_line_width_px": {"road_line": 1.0, "edge": 1.5, "stop_line": 3.0, "trajectory": 2.5},
    "box_light_direction": [0.4, 0.6, 0.7],
    "box_ambient": 0.55,
    "box_diffuse": 0.45,
    "cameras": {
        "chase": {
            "back_m": 25.0,
            "eye_z_m": 15.0,
            "ahead_m": 40.0,
            "target_z_m": 1.0,
            "fovy_deg": 45.0,
            "draw_ego": True,
        },
        "driver": {
            "back_m": 0.0,
            "eye_z_m": 1.4,
            "ahead_m": 30.0,
            "target_z_m": 1.0,
            "fovy_deg": 60.0,
            "draw_ego": False,
        },
    },
    "near_plane_m": 0.5,
    "cull_range_m": 160.0,
    "lane_surface_width_m": 3.7,
    "box_height_m": {"vehicle": 1.6, "pedestrian": 1.8, "cyclist": 1.7},
    "sort_chunk_length_m": 5.0,
    "agent_view_piece_length_m": 2.0,
    "agent_view_sort_bias_m": {"surface": 14.0, "line": 10.5, "stop_line": 7.5, "goal": 7.0, "trajectory": 3.5},
    "overpass_clearance_m": 3.5,
    "overpass_search_radius_m": 3.0,
    "ground_grid_cell_m": 4.0,
    "ground_lookup_radius_m": 8.0,
    "ground_level_tolerance_m": 1.0,
    "ground_average_point_count": 4,
    "goal_polygon_vertex_count": 32,
    "trail_seconds": 3.0,
    "trail_break_distance_m": 20.0,
    "logged_future_seconds": 5.0,
    "partner_match_tolerance_m": 1.0,
    "agent_view_aspect": [16, 9],
    "replay_frames_per_second": 10,
    "obs_panel_default_zoom": 2.2,
}


@dataclass(frozen=True)
class DecodedObservation:
    ego_length_m: float
    ego_width_m: float
    goal_xy_m: np.ndarray
    partner_slot_idx: np.ndarray
    partner_xy_m: np.ndarray
    partner_heading_rad: np.ndarray
    partner_length_m: np.ndarray
    partner_width_m: np.ndarray
    lane_segment_xy_m: np.ndarray
    boundary_segment_xy_m: np.ndarray
    stop_line_xy_m: np.ndarray
    stop_line_type: np.ndarray
    stop_line_state: np.ndarray
    reported_counts: tuple


def decode_interactive_replay(compressed):
    """Inverse of viz._pack_replay_binary: returns (header, chunks) with read-only array views."""
    try:
        payload = zlib.decompress(compressed)
    except zlib.error as exc:
        raise ValueError(f"Replay payload is not valid zlib data: {exc}") from exc
    if len(payload) < REPLAY_HEADER_LENGTH_BYTES:
        raise ValueError("Replay payload is truncated before its header length")
    (header_length,) = struct.unpack_from("<I", payload, 0)
    header_end = REPLAY_HEADER_LENGTH_BYTES + header_length
    if header_end > len(payload):
        raise ValueError(f"Replay header length {header_length} overruns the {len(payload)}-byte payload")
    try:
        header = json.loads(payload[REPLAY_HEADER_LENGTH_BYTES:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Replay header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict) or not isinstance(header.get("chunks"), dict):
        raise ValueError("Replay header has no chunk table")

    data_start = header_end + (-header_end) % REPLAY_ALIGNMENT_BYTES
    data_end = data_start
    chunks = {}
    for name, chunk in header["chunks"].items():
        if chunk["dtype"] not in REPLAY_DTYPES:
            raise ValueError(f"Replay chunk {name!r} has unknown dtype {chunk['dtype']!r}")
        dtype = REPLAY_DTYPES[chunk["dtype"]]
        shape = tuple(int(dim) for dim in chunk["shape"])
        offset = int(chunk["offset"])
        byte_count = int(chunk["nbytes"])
        if any(dim < 0 for dim in shape) or byte_count != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
            raise ValueError(f"Replay chunk {name!r} has nbytes={byte_count} inconsistent with shape {shape}")
        chunk_start = data_start + offset
        if offset < 0 or offset % REPLAY_ALIGNMENT_BYTES or chunk_start + byte_count > len(payload):
            raise ValueError(f"Replay chunk {name!r} lies outside the payload")
        values = np.frombuffer(payload, dtype=dtype, count=byte_count // dtype.itemsize, offset=chunk_start)
        chunks[name] = values.reshape(shape)
        data_end = max(data_end, chunk_start + byte_count + (-byte_count) % REPLAY_ALIGNMENT_BYTES)
    if data_end != len(payload):
        raise ValueError(f"Replay payload has {len(payload) - data_end} trailing bytes after its last chunk")
    return header, chunks


def build_obs_layout(counts):
    """Add block start offsets and obs_dim to an observation layout count dict (C block order)."""
    missing_keys = [key for key in OBS_LAYOUT_COUNT_KEYS if key not in counts]
    if missing_keys:
        raise ValueError(f"Observation layout is missing {missing_keys}")
    layout = {key: int(counts[key]) for key in OBS_LAYOUT_COUNT_KEYS}
    negative_keys = [key for key, value in layout.items() if value < 0]
    if negative_keys:
        raise ValueError(f"Observation layout has negative entries {negative_keys}")
    layout["goal_start"] = layout["ego_features"] + layout["reward_coef_features"]
    layout["partner_start"] = layout["goal_start"] + layout["goal_count"] * layout["goal_features"]
    layout["lane_start"] = layout["partner_start"] + layout["partner_count"] * layout["partner_features"]
    layout["boundary_start"] = layout["lane_start"] + layout["lane_count"] * layout["lane_features"]
    layout["traffic_start"] = layout["boundary_start"] + layout["boundary_count"] * layout["boundary_features"]
    layout["valid_count_start"] = layout["traffic_start"] + layout["traffic_count"] * layout["traffic_features"]
    layout["obs_dim"] = layout["valid_count_start"] + layout["valid_count_features"] + layout["lattice_mask_features"]
    return layout


def obs_norm_from_env_config(env_cfg):
    return {
        "xy_offset": float(env_cfg["obs_norm_xy_offset_m"]),
        "road_seg_length": float(env_cfg["obs_norm_road_seg_length_m"]),
        "road_seg_width": float(env_cfg["obs_norm_road_seg_width_m"]),
        "veh_length": float(env_cfg["obs_norm_veh_length_m"]),
        "veh_width": float(env_cfg["obs_norm_veh_width_m"]),
        "goal_offset": float(env_cfg["obs_norm_goal_offset_m"]),
        "z": float(env_cfg["obs_norm_z_m"]),
    }


def obs_range_from_env_config(env_cfg):
    return {
        "road_front": float(env_cfg["obs_range_road_front_m"]),
        "road_behind": float(env_cfg["obs_range_road_behind_m"]),
        "road_side": float(env_cfg["obs_range_road_side_m"]),
        "partner": float(env_cfg["obs_range_partner_m"]),
        "traffic_control": float(env_cfg["obs_range_traffic_control_m"]),
    }


def _slot_block(row, layout, block_name):
    start = layout[f"{block_name}_start"]
    count = layout[f"{block_name}_count"]
    features = layout[f"{block_name}_features"]
    block = row[start : start + count * features].reshape(count, features)
    occupied = np.any(block != 0.0, axis=1)
    return block[occupied], np.flatnonzero(occupied)


def _segment_endpoints(block, obs_norm):
    mid_xy_m = block[:, 0:2] * obs_norm["xy_offset"]
    offset_xy_m = block[:, 5:7] * (block[:, 3] * obs_norm["road_seg_length"])[:, None]
    return np.stack([mid_xy_m - offset_xy_m, mid_xy_m + offset_xy_m], axis=1)


def decode_observation(header, observation_row):
    """Decode one agent's observation row into ego-frame metres (x forward, y left) and radians."""
    layout = header["obs_layout"]
    obs_norm = header["obs_norm_m"]
    row = np.asarray(observation_row).astype(np.float64)
    if row.shape != (layout["obs_dim"],):
        raise ValueError(f"Observation row has shape {row.shape}, expected ({layout['obs_dim']},)")

    goals, _ = _slot_block(row, layout, "goal")
    partners, partner_slot_idx = _slot_block(row, layout, "partner")
    lanes, _ = _slot_block(row, layout, "lane")
    boundaries, _ = _slot_block(row, layout, "boundary")
    traffic, _ = _slot_block(row, layout, "traffic")
    valid_count_start = layout["valid_count_start"]
    reported_counts = tuple(
        int(round(value)) for value in row[valid_count_start : valid_count_start + layout["valid_count_features"]]
    )
    return DecodedObservation(
        ego_length_m=float(row[2] * obs_norm["veh_length"]),
        ego_width_m=float(row[1] * obs_norm["veh_width"]),
        goal_xy_m=goals[:, 0:2] * obs_norm["goal_offset"],
        partner_slot_idx=partner_slot_idx,
        partner_xy_m=partners[:, 0:2] * obs_norm["xy_offset"],
        partner_heading_rad=np.arctan2(partners[:, 6], partners[:, 5]),
        partner_length_m=partners[:, 3] * obs_norm["veh_length"],
        partner_width_m=partners[:, 4] * obs_norm["veh_width"],
        lane_segment_xy_m=_segment_endpoints(lanes, obs_norm),
        boundary_segment_xy_m=_segment_endpoints(boundaries, obs_norm),
        stop_line_xy_m=np.stack([traffic[:, 0:2], traffic[:, 2:4]], axis=1) * obs_norm["xy_offset"],
        stop_line_type=np.rint(traffic[:, 5]).astype(np.int64),
        stop_line_state=np.rint(traffic[:, 6]).astype(np.int64),
        reported_counts=reported_counts,
    )


def ego_frame_to_world(points_xy_m, ego_x_m, ego_y_m, ego_heading_rad):
    points_xy_m = np.asarray(points_xy_m, dtype=np.float64)
    cos_heading = np.cos(float(ego_heading_rad))
    sin_heading = np.sin(float(ego_heading_rad))
    world_xy_m = np.empty_like(points_xy_m)
    world_xy_m[..., 0] = float(ego_x_m) + cos_heading * points_xy_m[..., 0] - sin_heading * points_xy_m[..., 1]
    world_xy_m[..., 1] = float(ego_y_m) + sin_heading * points_xy_m[..., 0] + cos_heading * points_xy_m[..., 1]
    return world_xy_m
