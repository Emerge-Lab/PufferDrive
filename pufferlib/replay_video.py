"""Rasterize interactive replays into world, BEV and agent-view mp4 videos (numpy + PIL + ffmpeg only)."""

import argparse
import glob
import json
import math
import operator
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw

from pufferlib import replay_format as rf
from pufferlib.replay_format import REPLAY_VIEW_STYLE

RENDER_VIEW_NAMES = ("world", "bev", "agent")
REPLAY_SUFFIX = ".replay.zlib"
MANIFEST_FILENAME = "manifest.json"
WORLD_VIDEO_SIZE_PX = 1024
BEV_VIDEO_SIZE_PX = 640
AGENT_VIDEO_WIDTH_PX = 960
AGENT_VIDEO_HEIGHT_PX = 540
SUPERSAMPLE_FACTOR = 2
MIN_LINE_WIDTH_PX = 2
WORLD_VIEW_PADDING_M = 40.0
WORLD_VIEW_MIN_SIDE_M = 150.0
WORLD_LANE_WIDTH_M = 0.5
WORLD_EDGE_WIDTH_M = 0.8
WORLD_STOP_LINE_WIDTH_M = 1.2
FOLLOW_RING_SCALE = 1.2
FOLLOW_RING_WIDTH_PX = 3
PEDESTRIAN_MIN_DIAMETER_M = 0.7
BEV_LANE_WIDTH_PX = 1.5
BEV_BOUNDARY_WIDTH_PX = 3.0
BEV_TRAFFIC_WIDTH_PX = 3.0
BEV_GOAL_RADIUS_PX = 5.0
BEV_TRAJECTORY_WIDTH_PX = 2.5
VIDEO_CODEC = "libx264"
VIDEO_PRESET = "veryfast"
VIDEO_CRF = 23
VIDEO_ENCODER_THREADS = 2
RENDER_PROCESS_NICENESS = 10
OPAQUE_ALPHA = 255


@dataclass(frozen=True)
class AgentViewCamera:
    eye_m: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    focal_length_px: float
    image_width_px: int
    image_height_px: int
    horizon_y_px: float


@dataclass(frozen=True)
class RenderedVideo:
    stem: str
    view: str
    path: str
    frame_count: int


def agent_view_camera(
    agent_x_m, agent_y_m, agent_heading_rad, camera_style, image_width_px, image_height_px, agent_z_m=0.0
):
    back_m = float(camera_style["back_m"])
    ahead_m = float(camera_style["ahead_m"])
    if back_m + ahead_m <= 0.0:
        raise ValueError("Agent view camera needs back_m + ahead_m > 0 so the view direction is defined")
    heading_x = math.cos(float(agent_heading_rad))
    heading_y = math.sin(float(agent_heading_rad))
    base_z_m = float(agent_z_m)
    eye = [
        float(agent_x_m) - back_m * heading_x,
        float(agent_y_m) - back_m * heading_y,
        base_z_m + float(camera_style["eye_z_m"]),
    ]
    target = [
        float(agent_x_m) + ahead_m * heading_x,
        float(agent_y_m) + ahead_m * heading_y,
        base_z_m + float(camera_style["target_z_m"]),
    ]
    delta = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]
    delta_norm = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2])
    forward = [delta[0] / delta_norm, delta[1] / delta_norm, delta[2] / delta_norm]
    right_norm = math.sqrt(forward[1] * forward[1] + forward[0] * forward[0])
    right = [forward[1] / right_norm, -forward[0] / right_norm, 0.0]
    up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    focal_length_px = (image_height_px / 2.0) / math.tan(float(camera_style["fovy_deg"]) * math.pi / 360.0)
    horizon_y_px = image_height_px / 2.0 - focal_length_px * (up[0] * heading_x + up[1] * heading_y) / (
        forward[0] * heading_x + forward[1] * heading_y
    )
    return AgentViewCamera(
        eye_m=np.array(eye),
        forward=np.array(forward),
        right=np.array(right),
        up=np.array(up),
        focal_length_px=focal_length_px,
        image_width_px=int(image_width_px),
        image_height_px=int(image_height_px),
        horizon_y_px=horizon_y_px,
    )


def project_points(points_world_m, camera):
    delta = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3) - camera.eye_m
    camera_x = delta[:, 0] * camera.right[0] + delta[:, 1] * camera.right[1] + delta[:, 2] * camera.right[2]
    camera_y = delta[:, 0] * camera.up[0] + delta[:, 1] * camera.up[1] + delta[:, 2] * camera.up[2]
    depth_m = delta[:, 0] * camera.forward[0] + delta[:, 1] * camera.forward[1] + delta[:, 2] * camera.forward[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = np.stack(
            [
                camera.image_width_px / 2.0 + camera.focal_length_px * camera_x / depth_m,
                camera.image_height_px / 2.0 - camera.focal_length_px * camera_y / depth_m,
            ],
            axis=1,
        )
    return pixels, depth_m


def _depth_m(points_world_m, camera):
    delta = np.asarray(points_world_m, dtype=np.float64) - camera.eye_m
    return delta[..., 0] * camera.forward[0] + delta[..., 1] * camera.forward[1] + delta[..., 2] * camera.forward[2]


def clip_polygon_near(points_world_m, camera, near_plane_m):
    points = np.asarray(points_world_m, dtype=np.float64).reshape(-1, 3)
    depth_m = _depth_m(points, camera)
    clipped = []
    point_count = len(points)
    for point_idx in range(point_count):
        next_idx = (point_idx + 1) % point_count
        start_inside = depth_m[point_idx] >= near_plane_m
        if start_inside:
            clipped.append(points[point_idx])
        if start_inside == (depth_m[next_idx] >= near_plane_m):
            continue
        fraction = (near_plane_m - depth_m[point_idx]) / (depth_m[next_idx] - depth_m[point_idx])
        clipped.append(points[point_idx] + (points[next_idx] - points[point_idx]) * fraction)
    if len(clipped) < 3:
        return np.zeros((0, 3))
    return np.array(clipped)


def _clip_segment_near(start_m, end_m, camera, near_plane_m):
    start_depth_m = _depth_m(start_m, camera)
    end_depth_m = _depth_m(end_m, camera)
    if start_depth_m < near_plane_m and end_depth_m < near_plane_m:
        return None
    if start_depth_m >= near_plane_m and end_depth_m >= near_plane_m:
        return start_m, end_m
    fraction = (near_plane_m - start_depth_m) / (end_depth_m - start_depth_m)
    crossing_m = start_m + (end_m - start_m) * fraction
    return (crossing_m, end_m) if start_depth_m < near_plane_m else (start_m, crossing_m)


def select_tracked_agent(agent_i32):
    slots = agent_i32[0, :, rf.AGENT_I32_SLOT_IDX]
    eligible = (
        (slots >= 0) & (agent_i32[0, :, rf.AGENT_I32_VALID_IDX] == 1) & (agent_i32[0, :, rf.AGENT_I32_REMOVED_IDX] == 0)
    )
    candidates = np.flatnonzero(eligible)
    if len(candidates) == 0:
        raise ValueError("Replay has no valid policy-controlled agent at frame 0 to track")
    agent_idx = int(candidates[np.argmin(slots[candidates])])
    alive = (agent_i32[:, agent_idx, rf.AGENT_I32_VALID_IDX] == 1) & (
        agent_i32[:, agent_idx, rf.AGENT_I32_REMOVED_IDX] == 0
    )
    dead_frames = np.flatnonzero(~alive)
    frame_count = int(dead_frames[0]) if len(dead_frames) else int(agent_i32.shape[0])
    return agent_idx, int(slots[agent_idx]), frame_count


def world_view_transform(agent_f32, agent_i32, road_points_m, road_lengths, image_size_px):
    alive = (agent_i32[..., rf.AGENT_I32_VALID_IDX] == 1) & (agent_i32[..., rf.AGENT_I32_REMOVED_IDX] == 0)
    agent_xy_m = np.stack(
        [agent_f32[..., rf.AGENT_F32_X_IDX][alive], agent_f32[..., rf.AGENT_F32_Y_IDX][alive]], axis=1
    ).astype(np.float64)
    road_point_count = int(np.sum(road_lengths))
    road_xy_m = np.asarray(road_points_m, dtype=np.float64)[:road_point_count]
    if len(agent_xy_m):
        low_m = agent_xy_m.min(axis=0) - WORLD_VIEW_PADDING_M
        high_m = agent_xy_m.max(axis=0) + WORLD_VIEW_PADDING_M
        if road_point_count:
            low_m = np.maximum(low_m, road_xy_m.min(axis=0))
            high_m = np.minimum(high_m, road_xy_m.max(axis=0))
    elif road_point_count:
        low_m, high_m = road_xy_m.min(axis=0), road_xy_m.max(axis=0)
    else:
        low_m, high_m = np.zeros(2), np.zeros(2)
    side_m = max(float(high_m[0] - low_m[0]), float(high_m[1] - low_m[1]), WORLD_VIEW_MIN_SIDE_M)
    center_m = (low_m + high_m) / 2.0
    return float(center_m[0] - side_m / 2.0), float(center_m[1] + side_m / 2.0), image_size_px / side_m


def bev_ego_pixel_transform(header, image_size_px):
    obs_range_m = header["obs_range_m"]
    pixels_per_meter = image_size_px / (obs_range_m["road_front"] + obs_range_m["road_behind"])
    return image_size_px / 2.0, pixels_per_meter * obs_range_m["road_front"], pixels_per_meter


def _check_frame(frame, width_px, height_px):
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (height_px, width_px, 3):
        raise ValueError(
            f"Video frames must be uint8 arrays of shape ({height_px}, {width_px}, 3), "
            f"got {getattr(frame, 'dtype', type(frame))} {getattr(frame, 'shape', None)}"
        )


def write_mp4(frames, path, width_px, height_px):
    if width_px <= 0 or height_px <= 0 or width_px % 2 or height_px % 2:
        raise ValueError(f"Video dimensions must be positive and even for yuv420p, got {width_px}x{height_px}")
    frame_iter = iter(frames)
    first_frame = next(frame_iter, None)
    if first_frame is None:
        raise ValueError("write_mp4 needs at least one frame")
    _check_frame(first_frame, width_px, height_px)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width_px}x{height_px}",
        "-r",
        str(REPLAY_VIEW_STYLE["replay_frames_per_second"]),
        "-i",
        "-",
        "-c:v",
        VIDEO_CODEC,
        "-preset",
        VIDEO_PRESET,
        "-crf",
        str(VIDEO_CRF),
        "-threads",
        str(VIDEO_ENCODER_THREADS),
        "-vf",
        "scale=out_color_matrix=bt709,format=yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-movflags",
        "+faststart",
        path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_count = 0
    try:
        process.stdin.write(np.ascontiguousarray(first_frame).tobytes())
        frame_count = 1
        for frame in frame_iter:
            _check_frame(frame, width_px, height_px)
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
            frame_count += 1
        process.stdin.close()
    except BrokenPipeError as exc:
        return_code = process.wait()
        _remove_partial_video(path)
        raise RuntimeError(f"ffmpeg exited with code {return_code} while writing {path}") from exc
    except BaseException:
        process.kill()
        process.wait()
        _remove_partial_video(path)
        raise
    return_code = process.wait()
    if return_code != 0:
        _remove_partial_video(path)
        raise RuntimeError(f"ffmpeg exited with code {return_code} while writing {path}")
    return frame_count


def _remove_partial_video(path):
    if os.path.exists(path):
        os.remove(path)


def _hex_rgb(color):
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _rgba(color, alpha):
    return (*_hex_rgb(color), int(round(OPAQUE_ALPHA * alpha)))


def _agent_color(header, chunks, frame_idx, agent_idx):
    agent_i32 = chunks["agent_i32"]
    is_active = agent_i32[frame_idx, agent_idx, rf.AGENT_I32_ACTIVE_IDX] == 1
    infraction_count = REPLAY_VIEW_STYLE["infraction_metric_count"]
    has_infraction = agent_i32[frame_idx, agent_idx, rf.AGENT_I32_TYPE_IDX] == rf.AGENT_TYPE_VEHICLE and np.any(
        chunks["metrics_f32"][frame_idx, agent_idx, :infraction_count] > 0
    )
    if has_infraction:
        return REPLAY_VIEW_STYLE["agent_colors"]["infraction"]
    if is_active:
        vehicle_colors = REPLAY_VIEW_STYLE["vehicle_colors"]
        return vehicle_colors[abs(int(agent_i32[frame_idx, agent_idx, rf.AGENT_I32_ID_IDX])) % len(vehicle_colors)]
    if agent_idx in header["expert_indices"]:
        return REPLAY_VIEW_STYLE["agent_colors"]["dynamic_expert"]
    return REPLAY_VIEW_STYLE["agent_colors"]["static"]


def _agent_alive(chunks, frame_idx, agent_idx):
    agent_i32 = chunks["agent_i32"]
    return (
        agent_i32[frame_idx, agent_idx, rf.AGENT_I32_VALID_IDX] == 1
        and agent_i32[frame_idx, agent_idx, rf.AGENT_I32_REMOVED_IDX] == 0
    )


def _agent_z_m(chunks, frame_idx, agent_idx):
    return float(chunks["agent_f32"][frame_idx, agent_idx, rf.AGENT_F32_Z_IDX])


def _agent_pose(chunks, frame_idx, agent_idx):
    fields = chunks["agent_f32"][frame_idx, agent_idx]
    return (
        float(fields[rf.AGENT_F32_X_IDX]),
        float(fields[rf.AGENT_F32_Y_IDX]),
        float(fields[rf.AGENT_F32_HEADING_IDX]),
        float(fields[rf.AGENT_F32_LENGTH_IDX]),
        float(fields[rf.AGENT_F32_WIDTH_IDX]),
    )


def _box_corners_xy_m(x_m, y_m, heading_rad, length_m, width_m):
    forward = np.array([math.cos(heading_rad), math.sin(heading_rad)])
    left = np.array([-math.sin(heading_rad), math.cos(heading_rad)])
    center = np.array([x_m, y_m])
    half_length_m = length_m / 2.0
    half_width_m = width_m / 2.0
    return np.array(
        [
            center - forward * half_length_m - left * half_width_m,
            center + forward * half_length_m - left * half_width_m,
            center + forward * half_length_m + left * half_width_m,
            center - forward * half_length_m + left * half_width_m,
        ]
    )


def _world_to_ego_xy_m(points_xy_m, ego_x_m, ego_y_m, ego_heading_rad):
    points_xy_m = np.asarray(points_xy_m, dtype=np.float64)
    cos_heading = math.cos(ego_heading_rad)
    sin_heading = math.sin(ego_heading_rad)
    delta_x = points_xy_m[..., 0] - ego_x_m
    delta_y = points_xy_m[..., 1] - ego_y_m
    return np.stack([cos_heading * delta_x + sin_heading * delta_y, -sin_heading * delta_x + cos_heading * delta_y], -1)


def _road_polylines(chunks):
    polylines = []
    point_start = 0
    for polyline_idx in range(len(chunks["road_lengths"])):
        point_count = int(chunks["road_lengths"][polyline_idx])
        if point_count <= 0:
            continue
        points = np.asarray(chunks["road_points"][point_start : point_start + point_count], dtype=np.float64)
        polylines.append((int(chunks["road_types"][polyline_idx]), points))
        point_start += point_count
    return polylines


def _predicted_path_xy_m(header, chunks, frame_idx, agent_idx):
    base = header["agent_path_field"]
    sample_count = header["agent_path_sample_count"]
    path = np.asarray(chunks["agent_f32"][frame_idx, agent_idx, base : base + 2 * sample_count], dtype=np.float64)
    if path[0] == 0.0 and path[1] == 0.0:
        return None
    return path.reshape(sample_count, 2)


def _goal_xy_m(header, chunks, frame_idx, agent_idx):
    if "goals_f32" not in chunks or chunks["agent_i32"][frame_idx, agent_idx, rf.AGENT_I32_SLOT_IDX] < 0:
        return np.zeros((0, 2))
    goals = np.asarray(chunks["goals_f32"][frame_idx, agent_idx], dtype=np.float64)
    goals = goals.reshape(header["num_goals"], -1)[:, 0:2]
    return goals[np.any(goals != 0.0, axis=1)]


def _goal_radius_m(header, chunks, frame_idx, agent_idx):
    field = header["agent_goal_radius_field"]
    if field < chunks["agent_f32"].shape[2]:
        return float(chunks["agent_f32"][frame_idx, agent_idx, field])
    return float(header["default_goal_radius_meters"])


def _trail_runs_m(chunks, frame_idx, agent_idx, trail_frame_count):
    break_distance_m = REPLAY_VIEW_STYLE["trail_break_distance_m"]
    runs = []
    current_run = []
    for trail_frame in range(max(0, frame_idx - trail_frame_count), frame_idx + 1):
        alive = _agent_alive(chunks, trail_frame, agent_idx)
        x_m, y_m, _, _, _ = _agent_pose(chunks, trail_frame, agent_idx)
        teleported = (
            bool(current_run) and math.hypot(x_m - current_run[-1][0], y_m - current_run[-1][1]) > break_distance_m
        )
        if not alive or teleported:
            if len(current_run) > 1:
                runs.append(np.array(current_run))
            current_run = []
        if alive:
            current_run.append((x_m, y_m, _agent_z_m(chunks, trail_frame, agent_idx)))
    if len(current_run) > 1:
        runs.append(np.array(current_run))
    return runs


def _logged_future_runs_m(chunks, frame_idx, slot, future_frame_count):
    if "ghost_f32" not in chunks or slot < 0 or slot >= chunks["ghost_f32"].shape[1]:
        return []
    ghost = chunks["ghost_f32"]
    ghost_z = chunks.get("ghost_z_f32")
    runs = []
    current_run = []
    for future_frame in range(frame_idx, min(ghost.shape[0], frame_idx + future_frame_count + 1)):
        if ghost[future_frame, slot, 4] <= 0.0:
            if len(current_run) > 1:
                runs.append(np.array(current_run))
            current_run = []
            continue
        z_m = float(ghost_z[future_frame, slot]) if ghost_z is not None else 0.0
        current_run.append((float(ghost[future_frame, slot, 0]), float(ghost[future_frame, slot, 1]), z_m))
    if len(current_run) > 1:
        runs.append(np.array(current_run))
    return runs


def _frame_count_for_seconds(header, seconds):
    return int(math.floor(seconds / header["dt"] + 0.5))


def _traffic_color(control_type, control_state):
    if control_type == rf.TRAFFIC_TYPE_LIGHT:
        return REPLAY_VIEW_STYLE["traffic_state_colors"].get(
            str(int(control_state)), REPLAY_VIEW_STYLE["traffic_default_color"]
        )
    if control_type == rf.TRAFFIC_TYPE_STOP_SIGN:
        return REPLAY_VIEW_STYLE["stop_sign_color"]
    return REPLAY_VIEW_STYLE["yield_sign_color"]


def _frame_traffic(chunks, frame_idx):
    controls = []
    for control_idx in range(chunks["traffic_i16"].shape[1]):
        present, packed_type, state = chunks["traffic_i16"][frame_idx, control_idx]
        if not present:
            continue
        static_type = int(chunks["traffic_types"][control_idx]) if control_idx < len(chunks["traffic_types"]) else 0
        stop_line = chunks["traffic_stop_lines"][control_idx]
        endpoints_m = np.array([[stop_line[0], stop_line[1]], [stop_line[3], stop_line[4]]], dtype=np.float64)
        controls.append((static_type or int(packed_type), int(state), endpoints_m))
    return controls


def _finish_frame(image):
    return np.asarray(image.reduce(SUPERSAMPLE_FACTOR).convert("RGB"))


def _composite_layer(image, layer):
    return Image.alpha_composite(image, layer)


def _world_frames(header, chunks):
    image_size_px = WORLD_VIDEO_SIZE_PX * SUPERSAMPLE_FACTOR
    x0_m, y1_m, pixels_per_meter = world_view_transform(
        chunks["agent_f32"], chunks["agent_i32"], chunks["road_points"], chunks["road_lengths"], WORLD_VIDEO_SIZE_PX
    )
    pixels_per_meter *= SUPERSAMPLE_FACTOR

    def to_pixels(points_xy_m):
        points_xy_m = np.asarray(points_xy_m, dtype=np.float64)
        return [((x - x0_m) * pixels_per_meter, (y1_m - y) * pixels_per_meter) for x, y in points_xy_m.reshape(-1, 2)]

    def width_px(width_m):
        return max(MIN_LINE_WIDTH_PX, int(round(width_m * pixels_per_meter)))

    map_colors = REPLAY_VIEW_STYLE["map_colors"]["light"]
    road_layer = Image.new("RGBA", (image_size_px, image_size_px), map_colors["background"])
    road_draw = ImageDraw.Draw(road_layer)
    polylines = _road_polylines(chunks)
    for road_type, color, road_width_m in (
        (0, "road", WORLD_LANE_WIDTH_M),
        (1, "line", WORLD_LANE_WIDTH_M),
        (2, "edge", WORLD_EDGE_WIDTH_M),
    ):
        for polyline_type, points in polylines:
            if polyline_type != road_type or len(points) < 2:
                continue
            road_draw.line(to_pixels(points), fill=map_colors[color], width=width_px(road_width_m), joint="curve")

    tracked_idx, _, _ = select_tracked_agent(chunks["agent_i32"])
    agent_count = chunks["agent_f32"].shape[1]
    lattice = header["action_type"] == "lattice"
    show_ghost = header["active_count"] == 1 and "ghost_f32" in chunks
    for frame_idx in range(header["frames"]):
        image = road_layer.copy()
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        if show_ghost:
            for slot in range(chunks["ghost_f32"].shape[1]):
                ghost_x, ghost_y, ghost_heading, ghost_length, ghost_width = chunks["ghost_f32"][frame_idx, slot]
                if ghost_width <= 0:
                    continue
                corners = _box_corners_xy_m(
                    float(ghost_x), float(ghost_y), float(ghost_heading), float(ghost_length), float(ghost_width)
                )
                overlay_draw.polygon(
                    to_pixels(corners),
                    fill=_rgba(REPLAY_VIEW_STYLE["logged_future_color"], REPLAY_VIEW_STYLE["ghost_fill_alpha"]),
                )
        for agent_idx in range(agent_count):
            if not lattice and agent_idx != 0:
                break
            path_xy_m = _predicted_path_xy_m(header, chunks, frame_idx, agent_idx)
            if path_xy_m is None or not _agent_alive(chunks, frame_idx, agent_idx):
                continue
            path_width_m = max(float(chunks["agent_f32"][frame_idx, agent_idx, rf.AGENT_F32_WIDTH_IDX]), 0.1)
            overlay_draw.line(
                to_pixels(path_xy_m),
                fill=_rgba(REPLAY_VIEW_STYLE["predicted_path_color"], REPLAY_VIEW_STYLE["predicted_path_alpha"]),
                width=width_px(path_width_m),
                joint="curve",
            )
        image = _composite_layer(image, overlay)
        draw = ImageDraw.Draw(image)
        for agent_idx in range(agent_count):
            if chunks["agent_i32"][frame_idx, agent_idx, rf.AGENT_I32_VALID_IDX] != 1:
                continue
            x_m, y_m, heading_rad, length_m, width_m = _agent_pose(chunks, frame_idx, agent_idx)
            color = _agent_color(header, chunks, frame_idx, agent_idx)
            if chunks["agent_i32"][frame_idx, agent_idx, rf.AGENT_I32_TYPE_IDX] == rf.AGENT_TYPE_PEDESTRIAN:
                radius_px = max(width_m, PEDESTRIAN_MIN_DIAMETER_M) / 2.0 * pixels_per_meter
                (center_px,) = to_pixels([[x_m, y_m]])
                draw.ellipse(
                    [
                        center_px[0] - radius_px,
                        center_px[1] - radius_px,
                        center_px[0] + radius_px,
                        center_px[1] + radius_px,
                    ],
                    fill=color,
                    outline=REPLAY_VIEW_STYLE["agent_outline_color"],
                )
                continue
            draw.polygon(
                to_pixels(_box_corners_xy_m(x_m, y_m, heading_rad, length_m, width_m)),
                fill=color,
                outline=REPLAY_VIEW_STYLE["agent_outline_color"],
            )
        for control_type, control_state, endpoints_m in _frame_traffic(chunks, frame_idx):
            draw.line(
                to_pixels(endpoints_m),
                fill=_traffic_color(control_type, control_state),
                width=width_px(WORLD_STOP_LINE_WIDTH_M),
            )
        if _agent_alive(chunks, frame_idx, tracked_idx):
            x_m, y_m, _, length_m, width_m = _agent_pose(chunks, frame_idx, tracked_idx)
            (center_px,) = to_pixels([[x_m, y_m]])
            ring_radius_px = max(length_m, width_m) * FOLLOW_RING_SCALE * pixels_per_meter
            draw.ellipse(
                [
                    center_px[0] - ring_radius_px,
                    center_px[1] - ring_radius_px,
                    center_px[0] + ring_radius_px,
                    center_px[1] + ring_radius_px,
                ],
                outline=REPLAY_VIEW_STYLE["follow_ring_color"],
                width=FOLLOW_RING_WIDTH_PX * SUPERSAMPLE_FACTOR,
            )
            goal_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            goal_draw = ImageDraw.Draw(goal_overlay)
            goal_radius_px = _goal_radius_m(header, chunks, frame_idx, tracked_idx) * pixels_per_meter
            for goal_x_m, goal_y_m in _goal_xy_m(header, chunks, frame_idx, tracked_idx):
                (goal_px,) = to_pixels([[goal_x_m, goal_y_m]])
                goal_draw.ellipse(
                    [
                        goal_px[0] - goal_radius_px,
                        goal_px[1] - goal_radius_px,
                        goal_px[0] + goal_radius_px,
                        goal_px[1] + goal_radius_px,
                    ],
                    fill=_rgba(REPLAY_VIEW_STYLE["goal_color"], REPLAY_VIEW_STYLE["goal_fill_alpha"]),
                    outline=_rgba(REPLAY_VIEW_STYLE["goal_color"], 1.0),
                    width=MIN_LINE_WIDTH_PX,
                )
            image = _composite_layer(image, goal_overlay)
        yield _finish_frame(image)


def _bev_frames(header, chunks):
    if "obs" not in chunks or not header.get("obs_layout"):
        raise ValueError("The bev view needs captured observations (eval.capture_observations=true)")
    image_size_px = BEV_VIDEO_SIZE_PX * SUPERSAMPLE_FACTOR
    ego_px_x, ego_px_y, pixels_per_meter = bev_ego_pixel_transform(header, BEV_VIDEO_SIZE_PX)
    ego_px_x *= SUPERSAMPLE_FACTOR
    ego_px_y *= SUPERSAMPLE_FACTOR
    pixels_per_meter *= SUPERSAMPLE_FACTOR

    def to_pixels(points_ego_xy_m):
        points_ego_xy_m = np.asarray(points_ego_xy_m, dtype=np.float64)
        return [
            (ego_px_x - pixels_per_meter * ly, ego_px_y - pixels_per_meter * lx)
            for lx, ly in points_ego_xy_m.reshape(-1, 2)
        ]

    bev_colors = REPLAY_VIEW_STYLE["bev_colors"]
    tracked_idx, slot, frame_count = select_tracked_agent(chunks["agent_i32"])
    trail_frame_count = _frame_count_for_seconds(header, REPLAY_VIEW_STYLE["trail_seconds"])
    future_frame_count = _frame_count_for_seconds(header, REPLAY_VIEW_STYLE["logged_future_seconds"])
    for frame_idx in range(frame_count):
        observation = rf.decode_observation(header, chunks["obs"][frame_idx, slot])
        ego_x_m, ego_y_m, ego_heading_rad, _, _ = _agent_pose(chunks, frame_idx, tracked_idx)
        image = Image.new("RGBA", (image_size_px, image_size_px), bev_colors["background"])
        draw = ImageDraw.Draw(image)
        for segment in observation.lane_segment_xy_m:
            draw.line(
                to_pixels(segment), fill=bev_colors["lane"], width=int(round(BEV_LANE_WIDTH_PX * SUPERSAMPLE_FACTOR))
            )
        for segment in observation.boundary_segment_xy_m:
            draw.line(
                to_pixels(segment),
                fill=bev_colors["boundary"],
                width=int(round(BEV_BOUNDARY_WIDTH_PX * SUPERSAMPLE_FACTOR)),
            )
        for segment, control_type, control_state in zip(
            observation.stop_line_xy_m, observation.stop_line_type, observation.stop_line_state
        ):
            draw.line(
                to_pixels(segment),
                fill=_traffic_color(int(control_type), int(control_state)),
                width=int(round(BEV_TRAFFIC_WIDTH_PX * SUPERSAMPLE_FACTOR)),
            )

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        trajectory_width_px = int(round(BEV_TRAJECTORY_WIDTH_PX * SUPERSAMPLE_FACTOR))
        agent_color = _agent_color(header, chunks, frame_idx, tracked_idx)
        for run_m in _trail_runs_m(chunks, frame_idx, tracked_idx, trail_frame_count):
            overlay_draw.line(
                to_pixels(_world_to_ego_xy_m(run_m[:, :2], ego_x_m, ego_y_m, ego_heading_rad)),
                fill=_rgba(agent_color, REPLAY_VIEW_STYLE["trail_alpha"]),
                width=trajectory_width_px,
                joint="curve",
            )
        for run_m in _logged_future_runs_m(chunks, frame_idx, slot, future_frame_count):
            overlay_draw.line(
                to_pixels(_world_to_ego_xy_m(run_m[:, :2], ego_x_m, ego_y_m, ego_heading_rad)),
                fill=_rgba(REPLAY_VIEW_STYLE["logged_future_color"], REPLAY_VIEW_STYLE["logged_future_alpha"]),
                width=trajectory_width_px,
                joint="curve",
            )
        path_xy_m = _predicted_path_xy_m(header, chunks, frame_idx, tracked_idx)
        if path_xy_m is not None:
            ego_width_m = float(chunks["agent_f32"][frame_idx, tracked_idx, rf.AGENT_F32_WIDTH_IDX])
            overlay_draw.line(
                to_pixels(_world_to_ego_xy_m(path_xy_m, ego_x_m, ego_y_m, ego_heading_rad)),
                fill=_rgba(REPLAY_VIEW_STYLE["predicted_path_color"], REPLAY_VIEW_STYLE["predicted_path_alpha"]),
                width=max(MIN_LINE_WIDTH_PX, int(round(ego_width_m * pixels_per_meter))),
                joint="curve",
            )
        image = _composite_layer(image, overlay)
        draw = ImageDraw.Draw(image)

        for partner_xy_m, heading_rad, length_m, width_m in zip(
            observation.partner_xy_m,
            observation.partner_heading_rad,
            observation.partner_length_m,
            observation.partner_width_m,
        ):
            corners = _box_corners_xy_m(
                float(partner_xy_m[0]), float(partner_xy_m[1]), float(heading_rad), float(length_m), float(width_m)
            )
            draw.polygon(to_pixels(corners), fill=bev_colors["partner_fill"], outline=bev_colors["partner_outline"])
        goal_radius_px = BEV_GOAL_RADIUS_PX * SUPERSAMPLE_FACTOR
        for goal_px in to_pixels(observation.goal_xy_m):
            draw.ellipse(
                [
                    goal_px[0] - goal_radius_px,
                    goal_px[1] - goal_radius_px,
                    goal_px[0] + goal_radius_px,
                    goal_px[1] + goal_radius_px,
                ],
                fill=bev_colors["goal"],
            )
        ego_corners = _box_corners_xy_m(0.0, 0.0, 0.0, observation.ego_length_m, observation.ego_width_m)
        draw.polygon(to_pixels(ego_corners), fill=bev_colors["ego"], outline=REPLAY_VIEW_STYLE["agent_outline_color"])
        yield _finish_frame(image)


PRIMITIVE_POLYGON = "polygon"
PRIMITIVE_LINE = "line"
PRIMITIVE_BOX = "box"


class GroundHeightLookup:
    """Road-surface height near a point, taken on the level closest to a reference height so bridges stay apart."""

    def __init__(self, road_points_xyz_m):
        self.points_m = np.asarray(road_points_xyz_m, dtype=np.float64).reshape(-1, 3)
        self.cell_m = REPLAY_VIEW_STYLE["ground_grid_cell_m"]
        self.radius_m = REPLAY_VIEW_STYLE["ground_lookup_radius_m"]
        self.tolerance_m = REPLAY_VIEW_STYLE["ground_level_tolerance_m"]
        reach = math.ceil(self.radius_m / self.cell_m)
        self.cell_offsets = [(dx, dy) for dx in range(-reach, reach + 1) for dy in range(-reach, reach + 1)]
        cells = {}
        for point_idx, (cell_x, cell_y) in enumerate(np.floor(self.points_m[:, :2] / self.cell_m).astype(np.int64)):
            cells.setdefault((int(cell_x), int(cell_y)), []).append(point_idx)
        self.cells = {cell: np.array(indices) for cell, indices in cells.items()}

    def height_m(self, x_m, y_m, reference_z_m):
        """Mean of the nearest lane points on the level closest to the reference height (the JS order and maths)."""
        squared_distances = []
        heights = []
        for dx, dy in self.cell_offsets:
            indices = self.cells.get((math.floor(x_m / self.cell_m) + dx, math.floor(y_m / self.cell_m) + dy))
            if indices is None:
                continue
            points_m = self.points_m[indices]
            offset_x = points_m[:, 0] - x_m
            offset_y = points_m[:, 1] - y_m
            d2 = offset_x * offset_x + offset_y * offset_y
            within = d2 < self.radius_m * self.radius_m
            squared_distances.append(d2[within])
            heights.append(points_m[within, 2])
        if not squared_distances or not sum(len(values) for values in squared_distances):
            return float(reference_z_m)
        squared_distances = np.concatenate(squared_distances)
        heights = np.concatenate(heights)
        level_z_m = float(reference_z_m)
        if not np.any(np.abs(heights - reference_z_m) <= self.tolerance_m):
            level_z_m = float(heights[int(np.argmin(np.abs(heights - reference_z_m)))])
        on_level = np.abs(heights - level_z_m) <= self.tolerance_m
        nearest = np.argsort(squared_distances[on_level], kind="stable")[
            : REPLAY_VIEW_STYLE["ground_average_point_count"]
        ]
        nearest_heights = heights[on_level][nearest].tolist()
        return sum(nearest_heights) / len(nearest_heights)

    def elevation_band(self, x_m, y_m, z_m):
        """1 when another road runs underneath (a bridge deck or anything on it), else 0."""
        radius_m = REPLAY_VIEW_STYLE["overpass_search_radius_m"]
        below_z_m = z_m - REPLAY_VIEW_STYLE["overpass_clearance_m"]
        reach = math.ceil(radius_m / self.cell_m)
        cell_x = math.floor(x_m / self.cell_m)
        cell_y = math.floor(y_m / self.cell_m)
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                indices = self.cells.get((cell_x + dx, cell_y + dy))
                if indices is None:
                    continue
                points_m = self.points_m[indices]
                offset_x = points_m[:, 0] - x_m
                offset_y = points_m[:, 1] - y_m
                below = (offset_x * offset_x + offset_y * offset_y <= radius_m * radius_m) & (
                    points_m[:, 2] <= below_z_m
                )
                if below.any():
                    return 1
        return 0


def _box_height_m(agent_type):
    box_heights_m = REPLAY_VIEW_STYLE["box_height_m"]
    if agent_type == rf.AGENT_TYPE_PEDESTRIAN:
        return box_heights_m["pedestrian"]
    if agent_type == rf.AGENT_TYPE_CYCLIST:
        return box_heights_m["cyclist"]
    return box_heights_m["vehicle"]


def _box_chunk_faces(x_m, y_m, heading_rad, length_m, width_m, height_m, base_z_m=0.0):
    """Faces as (corners (4,3), outward normal (3,), centre (3,)) for each sort chunk, rear to front."""
    forward = np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0])
    left = np.array([-math.sin(heading_rad), math.cos(heading_rad), 0.0])
    center = np.array([x_m, y_m, base_z_m])
    up = np.array([0.0, 0.0, 1.0])
    half_width_m = width_m / 2.0
    chunk_count = max(1, math.ceil(length_m / REPLAY_VIEW_STYLE["sort_chunk_length_m"]))
    chunk_length_m = length_m / chunk_count
    chunks = []
    for chunk_idx in range(chunk_count):
        rear_m = -length_m / 2.0 + chunk_idx * chunk_length_m
        front_m = rear_m + chunk_length_m
        rear_left = center + forward * rear_m + left * half_width_m
        rear_right = center + forward * rear_m - left * half_width_m
        front_left = center + forward * front_m + left * half_width_m
        front_right = center + forward * front_m - left * half_width_m
        lift = up * height_m
        middle = center + forward * ((rear_m + front_m) / 2.0)
        faces = [
            (np.array([rear_left, rear_right, front_right, front_left]) + lift, up, middle + lift),
            (
                np.array([rear_left, front_left, front_left + lift, rear_left + lift]),
                left,
                middle + left * half_width_m + lift / 2.0,
            ),
            (
                np.array([front_right, rear_right, rear_right + lift, front_right + lift]),
                -left,
                middle - left * half_width_m + lift / 2.0,
            ),
        ]
        if chunk_idx == 0:
            faces.append(
                (
                    np.array([rear_right, rear_left, rear_left + lift, rear_right + lift]),
                    -forward,
                    center + forward * rear_m + lift / 2.0,
                )
            )
        if chunk_idx == chunk_count - 1:
            faces.append(
                (
                    np.array([front_left, front_right, front_right + lift, front_left + lift]),
                    forward,
                    center + forward * front_m + lift / 2.0,
                )
            )
        chunks.append((middle + lift / 2.0, faces))
    return chunks


def _shaded_rgb(color, normal):
    light = np.asarray(REPLAY_VIEW_STYLE["box_light_direction"], dtype=np.float64)
    light = light / math.sqrt(float(light @ light))
    shade = REPLAY_VIEW_STYLE["box_ambient"] + REPLAY_VIEW_STYLE["box_diffuse"] * max(0.0, float(normal @ light))
    return tuple(min(255, int(round(channel * shade))) for channel in _hex_rgb(color))


def _lane_points_xyz_m(chunks):
    lane_points = []
    point_start = 0
    road_xyz_m = _road_points_xyz_m(chunks)
    for polyline_idx in range(len(chunks["road_lengths"])):
        point_count = int(chunks["road_lengths"][polyline_idx])
        if point_count <= 0:
            continue
        if int(chunks["road_types"][polyline_idx]) == 0:
            lane_points.append(road_xyz_m[point_start : point_start + point_count])
        point_start += point_count
    return np.concatenate(lane_points) if lane_points else np.zeros((0, 3))


def _road_points_xyz_m(chunks):
    point_count = int(np.sum(np.maximum(np.asarray(chunks["road_lengths"]), 0)))
    points_xy_m = np.asarray(chunks["road_points"][:point_count], dtype=np.float64)
    road_z = chunks.get("road_points_z")
    points_z_m = np.asarray(road_z[:point_count], dtype=np.float64) if road_z is not None else np.zeros(point_count)
    return np.concatenate([points_xy_m, points_z_m[:, None]], axis=1)


def _road_segment_pieces(chunks, road_type, ground):
    """Road segments of one draw type split into pieces no longer than the sort piece length, in JS order."""
    road_xyz_m = _road_points_xyz_m(chunks)
    segments = []
    point_start = 0
    for polyline_idx in range(len(chunks["road_lengths"])):
        point_count = int(chunks["road_lengths"][polyline_idx])
        if point_count <= 0:
            continue
        if int(chunks["road_types"][polyline_idx]) == road_type and point_count > 1:
            points_m = road_xyz_m[point_start : point_start + point_count]
            segments.append(np.stack([points_m[:-1], points_m[1:]], axis=1))
        point_start += point_count
    segments_m = np.concatenate(segments) if segments else np.zeros((0, 2, 3))
    delta_m = segments_m[:, 1] - segments_m[:, 0]
    length_m = np.hypot(delta_m[:, 0], delta_m[:, 1])
    piece_counts = np.maximum(1, np.ceil(length_m / REPLAY_VIEW_STYLE["agent_view_piece_length_m"]).astype(np.int64))
    segment_idx = np.repeat(np.arange(len(segments_m)), piece_counts)
    piece_idx = np.arange(len(segment_idx)) - np.repeat(np.cumsum(piece_counts) - piece_counts, piece_counts)
    start_fraction = piece_idx / piece_counts[segment_idx]
    end_fraction = (piece_idx + 1) / piece_counts[segment_idx]
    starts_m = segments_m[segment_idx, 0] + delta_m[segment_idx] * start_fraction[:, None]
    ends_m = segments_m[segment_idx, 0] + delta_m[segment_idx] * end_fraction[:, None]
    segment_mid_xy_m = (segments_m[segment_idx, 0, :2] + segments_m[segment_idx, 1, :2]) / 2.0
    segment_mid_m = (segments_m[:, 0] + segments_m[:, 1]) / 2.0
    segment_bands = np.array([ground.elevation_band(*mid_m) for mid_m in segment_mid_m], dtype=np.int64)
    return {
        "band": segment_bands[segment_idx],
        "starts_m": starts_m,
        "ends_m": ends_m,
        "segment_mid_xy_m": segment_mid_xy_m,
        "segment_length_m": length_m[segment_idx],
        "segment_delta_m": delta_m[segment_idx],
    }


def _lane_surface_quads(lane_pieces):
    keep = lane_pieces["segment_length_m"] > 0.0
    half_width_m = REPLAY_VIEW_STYLE["lane_surface_width_m"] / 2.0
    delta_m = lane_pieces["segment_delta_m"][keep]
    length_m = lane_pieces["segment_length_m"][keep]
    normal_x = -delta_m[:, 1] / length_m * half_width_m
    normal_y = delta_m[:, 0] / length_m * half_width_m
    starts_m = lane_pieces["starts_m"][keep]
    ends_m = lane_pieces["ends_m"][keep]
    quads_m = np.empty((len(starts_m), 4, 3))
    quads_m[:, 0] = np.stack([starts_m[:, 0] - normal_x, starts_m[:, 1] - normal_y, starts_m[:, 2]], axis=1)
    quads_m[:, 1] = np.stack([ends_m[:, 0] - normal_x, ends_m[:, 1] - normal_y, ends_m[:, 2]], axis=1)
    quads_m[:, 2] = np.stack([ends_m[:, 0] + normal_x, ends_m[:, 1] + normal_y, ends_m[:, 2]], axis=1)
    quads_m[:, 3] = np.stack([starts_m[:, 0] + normal_x, starts_m[:, 1] + normal_y, starts_m[:, 2]], axis=1)
    return {
        "band": lane_pieces["band"][keep],
        "quads_m": quads_m,
        "centroids_m": (starts_m + ends_m) / 2.0,
        "segment_mid_xy_m": lane_pieces["segment_mid_xy_m"][keep],
    }


def _resample_polyline_m(points_m, piece_length_m):
    resampled = [points_m[0]]
    for start_m, end_m in zip(points_m[:-1], points_m[1:]):
        pieces = max(1, math.ceil(math.hypot(end_m[0] - start_m[0], end_m[1] - start_m[1]) / piece_length_m))
        for piece in range(1, pieces + 1):
            resampled.append(start_m + (end_m - start_m) * (piece / pieces))
    return np.array(resampled)


def _ribbon_quads_m(points_m, half_width_m):
    point_count = len(points_m)
    normals = np.zeros((point_count, 2))
    for point_idx in range(point_count):
        previous_m = points_m[max(0, point_idx - 1)]
        next_m = points_m[min(point_count - 1, point_idx + 1)]
        dx, dy = next_m[0] - previous_m[0], next_m[1] - previous_m[1]
        length_m = math.hypot(dx, dy)
        if length_m > 0.0:
            normals[point_idx] = (-dy / length_m, dx / length_m)
    quads = []
    for point_idx in range(1, point_count):
        start_m, end_m = points_m[point_idx - 1], points_m[point_idx]
        if math.hypot(end_m[0] - start_m[0], end_m[1] - start_m[1]) <= 0.0:
            continue
        start_offset, end_offset = normals[point_idx - 1] * half_width_m, normals[point_idx] * half_width_m
        quads.append(
            np.array(
                [
                    [start_m[0] + start_offset[0], start_m[1] + start_offset[1], start_m[2]],
                    [end_m[0] + end_offset[0], end_m[1] + end_offset[1], end_m[2]],
                    [end_m[0] - end_offset[0], end_m[1] - end_offset[1], end_m[2]],
                    [start_m[0] - start_offset[0], start_m[1] - start_offset[1], start_m[2]],
                ]
            )
        )
    return quads


def _elevated_path_m(path_xy_m, start_z_m, ground):
    elevated = []
    previous_z_m = start_z_m
    for x_m, y_m in path_xy_m:
        previous_z_m = ground.height_m(float(x_m), float(y_m), previous_z_m)
        elevated.append((float(x_m), float(y_m), previous_z_m))
    return np.array(elevated)


def _bulk_projected(points_m, camera, near_plane_m):
    """Pixels for primitives fully in front of the near plane, plus a mask of those that cross it."""
    if not len(points_m):
        return np.zeros(points_m.shape[:2] + (2,)), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    depth_m = _depth_m(points_m, camera)
    in_front = np.all(depth_m >= near_plane_m, axis=1)
    crossing = np.any(depth_m >= near_plane_m, axis=1) & ~in_front
    pixels, _ = project_points(points_m.reshape(-1, 3), camera)
    return pixels.reshape(points_m.shape[:2] + (2,)), in_front, crossing


def _agent_view_primitives(header, chunks, frame_idx, tracked_idx, slot, camera, static, ground, draw_ego):
    """Painter's list per elevation band (bridge decks after the roads beneath), in the JS order and sort."""
    near_plane_m = REPLAY_VIEW_STYLE["near_plane_m"]
    cull_range_m = REPLAY_VIEW_STYLE["cull_range_m"]
    bias_m = REPLAY_VIEW_STYLE["agent_view_sort_bias_m"]
    colors = REPLAY_VIEW_STYLE["agent_view_colors"]
    line_widths_px = REPLAY_VIEW_STYLE["agent_view_line_width_px"]
    agent_x_m, agent_y_m, _, _, agent_width_m = _agent_pose(chunks, frame_idx, tracked_idx)
    agent_z_m = _agent_z_m(chunks, frame_idx, tracked_idx)
    agent_xy_m = np.array([agent_x_m, agent_y_m])
    primitives = []

    def in_range(mid_xy_m):
        return np.hypot(mid_xy_m[:, 0] - agent_xy_m[0], mid_xy_m[:, 1] - agent_xy_m[1]) <= cull_range_m

    def eye_distance_m(points_m):
        return np.linalg.norm(np.asarray(points_m, dtype=np.float64) - camera.eye_m, axis=-1)

    def add_line(start_m, end_m, color, width_px, sort_bias_m, band=None):
        mid_m = (np.asarray(start_m, dtype=np.float64) + np.asarray(end_m, dtype=np.float64)) / 2.0
        band = ground.elevation_band(*mid_m) if band is None else band
        key = float(eye_distance_m(mid_m)) + sort_bias_m
        primitives.append((band, -key, PRIMITIVE_LINE, (np.array([start_m, end_m], dtype=np.float64), color, width_px)))

    surface = static["surface"]
    surface_mask = in_range(surface["segment_mid_xy_m"])
    quads_m = surface["quads_m"][surface_mask]
    quad_pixels, quad_in_front, quad_crossing = _bulk_projected(quads_m, camera, near_plane_m)
    quad_keys = eye_distance_m(surface["centroids_m"][surface_mask]) + bias_m["surface"]
    quad_bands = surface["band"][surface_mask]
    for quad_idx in range(len(quads_m)):
        if quad_in_front[quad_idx]:
            pixels = [tuple(point) for point in quad_pixels[quad_idx]]
            primitives.append(
                (
                    int(quad_bands[quad_idx]),
                    -float(quad_keys[quad_idx]),
                    PRIMITIVE_POLYGON,
                    (None, pixels, colors["road"], None),
                )
            )
        elif quad_crossing[quad_idx]:
            primitives.append(
                (
                    int(quad_bands[quad_idx]),
                    -float(quad_keys[quad_idx]),
                    PRIMITIVE_POLYGON,
                    (quads_m[quad_idx], None, colors["road"], None),
                )
            )

    for road_type, color_name, width_name in ((1, "line", "road_line"), (2, "edge", "edge")):
        pieces = static["lines"][road_type]
        line_mask = in_range(pieces["segment_mid_xy_m"])
        width_px = max(MIN_LINE_WIDTH_PX, int(round(line_widths_px[width_name] * SUPERSAMPLE_FACTOR)))
        line_points_m = np.stack([pieces["starts_m"][line_mask], pieces["ends_m"][line_mask]], axis=1)
        keys = eye_distance_m((line_points_m[:, 0] + line_points_m[:, 1]) / 2.0) + bias_m["line"]
        bands = pieces["band"][line_mask]
        for line_idx in range(len(line_points_m)):
            primitives.append(
                (
                    int(bands[line_idx]),
                    -float(keys[line_idx]),
                    PRIMITIVE_LINE,
                    (line_points_m[line_idx], colors[color_name], width_px),
                )
            )

    stop_line_width_px = max(MIN_LINE_WIDTH_PX, int(round(line_widths_px["stop_line"] * SUPERSAMPLE_FACTOR)))
    for control_idx in range(chunks["traffic_i16"].shape[1]):
        present, packed_type, state = chunks["traffic_i16"][frame_idx, control_idx]
        if not present:
            continue
        static_type = int(chunks["traffic_types"][control_idx]) if control_idx < len(chunks["traffic_types"]) else 0
        stop_line = np.asarray(chunks["traffic_stop_lines"][control_idx], dtype=np.float64)
        start_m, end_m = stop_line[0:3], stop_line[3:6]
        if (
            math.hypot((start_m[0] + end_m[0]) / 2.0 - agent_x_m, (start_m[1] + end_m[1]) / 2.0 - agent_y_m)
            > cull_range_m
        ):
            continue
        add_line(
            start_m,
            end_m,
            _traffic_color(static_type or int(packed_type), int(state)),
            stop_line_width_px,
            bias_m["stop_line"],
        )

    goal_radius_m = _goal_radius_m(header, chunks, frame_idx, tracked_idx)
    vertex_count = REPLAY_VIEW_STYLE["goal_polygon_vertex_count"]
    for goal_x_m, goal_y_m in _goal_xy_m(header, chunks, frame_idx, tracked_idx):
        goal_z_m = ground.height_m(float(goal_x_m), float(goal_y_m), agent_z_m)
        angles = np.arange(vertex_count) * 2.0 * math.pi / vertex_count
        polygon_m = np.stack(
            [
                goal_x_m + goal_radius_m * np.cos(angles),
                goal_y_m + goal_radius_m * np.sin(angles),
                np.full(vertex_count, goal_z_m),
            ],
            axis=1,
        )
        key = float(eye_distance_m([goal_x_m, goal_y_m, goal_z_m])) + bias_m["goal"]
        goal_style = (
            _rgba(REPLAY_VIEW_STYLE["goal_color"], REPLAY_VIEW_STYLE["goal_fill_alpha"]),
            REPLAY_VIEW_STYLE["goal_color"],
        )
        band = ground.elevation_band(float(goal_x_m), float(goal_y_m), goal_z_m)
        primitives.append((band, -key, PRIMITIVE_POLYGON, (polygon_m, None) + goal_style))

    trajectory_width_px = max(MIN_LINE_WIDTH_PX, int(round(line_widths_px["trajectory"] * SUPERSAMPLE_FACTOR)))
    trail_color = _rgba(_agent_color(header, chunks, frame_idx, tracked_idx), REPLAY_VIEW_STYLE["trail_alpha"])
    trail_frame_count = _frame_count_for_seconds(header, REPLAY_VIEW_STYLE["trail_seconds"])
    for run_m in _trail_runs_m(chunks, frame_idx, tracked_idx, trail_frame_count):
        for start_m, end_m in zip(run_m[:-1], run_m[1:]):
            add_line(start_m, end_m, trail_color, trajectory_width_px, bias_m["trajectory"])
    future_color = _rgba(REPLAY_VIEW_STYLE["logged_future_color"], REPLAY_VIEW_STYLE["logged_future_alpha"])
    future_frame_count = _frame_count_for_seconds(header, REPLAY_VIEW_STYLE["logged_future_seconds"])
    for run_m in _logged_future_runs_m(chunks, frame_idx, slot, future_frame_count):
        for start_m, end_m in zip(run_m[:-1], run_m[1:]):
            add_line(start_m, end_m, future_color, trajectory_width_px, bias_m["trajectory"])
    path_xy_m = _predicted_path_xy_m(header, chunks, frame_idx, tracked_idx)
    if path_xy_m is not None:
        path_m = _resample_polyline_m(
            _elevated_path_m(path_xy_m, agent_z_m, ground), REPLAY_VIEW_STYLE["agent_view_piece_length_m"]
        )
        path_color = _rgba(REPLAY_VIEW_STYLE["predicted_path_color"], REPLAY_VIEW_STYLE["predicted_path_alpha"])
        for quad_m in _ribbon_quads_m(path_m, agent_width_m / 2.0):
            center_m = (quad_m[0] + quad_m[2]) / 2.0
            key = float(eye_distance_m(center_m)) + bias_m["trajectory"]
            band = ground.elevation_band(*center_m)
            primitives.append((band, -key, PRIMITIVE_POLYGON, (quad_m, None, path_color, None)))

    for agent_idx in range(chunks["agent_f32"].shape[1]):
        if chunks["agent_i32"][frame_idx, agent_idx, rf.AGENT_I32_VALID_IDX] != 1 or (
            agent_idx == tracked_idx and not draw_ego
        ):
            continue
        x_m, y_m, heading_rad, length_m, width_m = _agent_pose(chunks, frame_idx, agent_idx)
        if math.hypot(x_m - agent_x_m, y_m - agent_y_m) > cull_range_m:
            continue
        height_m = _box_height_m(int(chunks["agent_i32"][frame_idx, agent_idx, rf.AGENT_I32_TYPE_IDX]))
        color = _agent_color(header, chunks, frame_idx, agent_idx)
        base_z_m = _agent_z_m(chunks, frame_idx, agent_idx)
        band = ground.elevation_band(x_m, y_m, base_z_m)
        for centroid_m, faces in _box_chunk_faces(x_m, y_m, heading_rad, length_m, width_m, height_m, base_z_m):
            primitives.append((band, -float(eye_distance_m(centroid_m)), PRIMITIVE_BOX, (faces, color)))

    primitives.sort(key=operator.itemgetter(0, 1))
    return primitives


def _draw_agent_view_primitive(draw, camera, kind, payload):
    near_plane_m = REPLAY_VIEW_STYLE["near_plane_m"]
    if kind == PRIMITIVE_LINE:
        points_m, color, width_px = payload
        clipped = _clip_segment_near(points_m[0], points_m[1], camera, near_plane_m)
        if clipped is None:
            return
        segment_pixels, _ = project_points(np.array(clipped), camera)
        draw.line([tuple(point) for point in segment_pixels], fill=color, width=width_px)
        return
    if kind == PRIMITIVE_POLYGON:
        points_m, pixels, fill, outline = payload
        if pixels is None:
            clipped_m = clip_polygon_near(points_m, camera, near_plane_m)
            if not len(clipped_m):
                return
            clipped_pixels, _ = project_points(clipped_m, camera)
            pixels = [tuple(point) for point in clipped_pixels]
        draw.polygon(pixels, fill=fill, outline=outline, width=MIN_LINE_WIDTH_PX if outline else 1)
        return
    faces, color = payload
    for corners_m, normal, face_center_m in faces:
        if float((camera.eye_m - face_center_m) @ normal) <= 0.0:
            continue
        clipped_m = clip_polygon_near(corners_m, camera, near_plane_m)
        if not len(clipped_m):
            continue
        face_pixels, _ = project_points(clipped_m, camera)
        draw.polygon(
            [tuple(point) for point in face_pixels],
            fill=_shaded_rgb(color, normal),
            outline=REPLAY_VIEW_STYLE["agent_view_colors"]["box_outline"],
            width=SUPERSAMPLE_FACTOR,
        )


def _agent_frames(header, chunks):
    width_px = AGENT_VIDEO_WIDTH_PX * SUPERSAMPLE_FACTOR
    height_px = AGENT_VIDEO_HEIGHT_PX * SUPERSAMPLE_FACTOR
    camera_style = REPLAY_VIEW_STYLE["cameras"]["chase"]
    colors = REPLAY_VIEW_STYLE["agent_view_colors"]
    tracked_idx, slot, frame_count = select_tracked_agent(chunks["agent_i32"])
    ground = GroundHeightLookup(_lane_points_xyz_m(chunks))
    static = {
        "surface": _lane_surface_quads(_road_segment_pieces(chunks, 0, ground)),
        "lines": {road_type: _road_segment_pieces(chunks, road_type, ground) for road_type in (1, 2)},
    }
    for frame_idx in range(frame_count):
        agent_x_m, agent_y_m, agent_heading_rad, _, _ = _agent_pose(chunks, frame_idx, tracked_idx)
        agent_z_m = _agent_z_m(chunks, frame_idx, tracked_idx)
        camera = agent_view_camera(
            agent_x_m, agent_y_m, agent_heading_rad, camera_style, width_px, height_px, agent_z_m
        )
        image = Image.new("RGBA", (width_px, height_px), colors["ground"])
        draw = ImageDraw.Draw(image, "RGBA")
        horizon_y_px = min(max(camera.horizon_y_px, 0.0), float(height_px))
        if horizon_y_px > 0.0:
            draw.rectangle([0, 0, width_px, horizon_y_px], fill=colors["sky"])
        for _, _, kind, payload in _agent_view_primitives(
            header, chunks, frame_idx, tracked_idx, slot, camera, static, ground, camera_style["draw_ego"]
        ):
            _draw_agent_view_primitive(draw, camera, kind, payload)
        yield _finish_frame(image)


def render_replay_videos(replay_path, video_dir, views):
    replay_filename = os.path.basename(replay_path)
    if not replay_filename.endswith(REPLAY_SUFFIX):
        raise ValueError(f"Replay file {replay_path} does not end with {REPLAY_SUFFIX}")
    stem = replay_filename[: -len(REPLAY_SUFFIX)]
    with open(replay_path, "rb") as replay_file:
        header, chunks = rf.decode_interactive_replay(replay_file.read())
    os.makedirs(video_dir, exist_ok=True)
    rendered = []
    for view in views:
        video_path = os.path.abspath(os.path.join(video_dir, f"{stem}__{view}.mp4"))
        if view == "world":
            frame_count = write_mp4(_world_frames(header, chunks), video_path, WORLD_VIDEO_SIZE_PX, WORLD_VIDEO_SIZE_PX)
        elif view == "bev":
            frame_count = write_mp4(_bev_frames(header, chunks), video_path, BEV_VIDEO_SIZE_PX, BEV_VIDEO_SIZE_PX)
        elif view == "agent":
            frame_count = write_mp4(
                _agent_frames(header, chunks), video_path, AGENT_VIDEO_WIDTH_PX, AGENT_VIDEO_HEIGHT_PX
            )
        else:
            raise ValueError(f"Unknown render view {view!r}; expected one of {RENDER_VIEW_NAMES}")
        rendered.append(RenderedVideo(stem=stem, view=view, path=video_path, frame_count=frame_count))
    return rendered


def _write_manifest(video_dir, videos, failed_stems, git_commit):
    manifest = {"videos": [asdict(video) for video in videos], "failed": failed_stems, "git_commit": git_commit}
    manifest_path = os.path.join(video_dir, MANIFEST_FILENAME)
    with open(manifest_path + ".tmp", "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    os.replace(manifest_path + ".tmp", manifest_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render replay videos (world, bev, agent) from .replay.zlib files")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--views", required=True, help="Comma-separated subset of world,bev,agent")
    parser.add_argument("--delete-replays", action="store_true")
    parser.add_argument("--git-commit", default=None)
    args = parser.parse_args(argv)

    views = [view.strip() for view in args.views.split(",") if view.strip()]
    unknown_views = [view for view in views if view not in RENDER_VIEW_NAMES]
    if not views or unknown_views or len(set(views)) != len(views):
        parser.error(f"--views must be a non-empty, duplicate-free subset of {','.join(RENDER_VIEW_NAMES)}")
    replay_paths = sorted(glob.glob(os.path.join(args.replay_dir, f"*{REPLAY_SUFFIX}")))
    if not os.path.isdir(args.replay_dir) or not replay_paths:
        parser.error(f"--replay-dir {args.replay_dir} has no *{REPLAY_SUFFIX} files")

    os.nice(RENDER_PROCESS_NICENESS)
    os.makedirs(args.video_dir, exist_ok=True)
    videos = []
    failed_stems = []
    for replay_path in replay_paths:
        try:
            videos.extend(render_replay_videos(replay_path, args.video_dir, views))
        except Exception:
            traceback.print_exc()
            failed_stems.append(os.path.basename(replay_path)[: -len(REPLAY_SUFFIX)])
    _write_manifest(args.video_dir, videos, failed_stems, args.git_commit)
    print(f"Rendered {len(videos)} videos into {args.video_dir}; {len(failed_stems)} replays failed")
    if failed_stems:
        return 1
    if args.delete_replays:
        try:
            shutil.rmtree(args.replay_dir)
        except OSError as exc:
            print(f"WARNING: could not delete {args.replay_dir}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
