import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import pufferlib.viz
from pufferlib import pufferl, replay_format, replay_video
from pufferlib.config_schema import normalize_puffer_drive_config
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = Path(__file__).resolve().parents[2]
CARLA_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/carla"
SEED = 13
REPLAY_STEP_COUNT = 16
AGENT_VIEW_WIDTH_PX = 960
AGENT_VIEW_HEIGHT_PX = 540
WORLD_VIEW_SIZE_PX = 1024
BEV_VIEW_SIZE_PX = 640
WORLD_VIEW_MARGIN_M = 40.0
WORLD_VIEW_MIN_SIDE_M = 150.0
CLI_TIMEOUT_SECONDS = 300
REPLAY_SUFFIX = ".replay.zlib"
RENDER_VIEWS = ("world", "bev", "agent")
VIEW_SIZES_PX = {
    "world": (WORLD_VIEW_SIZE_PX, WORLD_VIEW_SIZE_PX),
    "bev": (BEV_VIEW_SIZE_PX, BEV_VIEW_SIZE_PX),
    "agent": (AGENT_VIEW_WIDTH_PX, AGENT_VIEW_HEIGHT_PX),
}
MIN_PIXEL_STD = 2.0
MIN_SKY_GROUND_CONTRAST = 8.0
HEADINGS_RAD = (0.0, 1.2, -2.8, math.pi)

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed"
)

REPLAY_ENV_OVERRIDES = {
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


def _camera_style(name):
    return replay_format.REPLAY_VIEW_STYLE["cameras"][name]


def _unit(vector):
    return vector / np.linalg.norm(vector)


def _expected_camera_vectors(x_m, y_m, heading_rad, style, height_px):
    heading_vector = np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0])
    up_axis = np.array([0.0, 0.0, 1.0])
    agent_point = np.array([x_m, y_m, 0.0])
    eye = agent_point - style["back_m"] * heading_vector + style["eye_z_m"] * up_axis
    target = agent_point + style["ahead_m"] * heading_vector + style["target_z_m"] * up_axis
    forward = _unit(target - eye)
    right = _unit(np.cross(forward, up_axis))
    up = np.cross(right, forward)
    focal_px = (height_px / 2.0) / math.tan(math.radians(style["fovy_deg"]) / 2.0)
    horizon_px = height_px / 2.0 - focal_px * np.dot(up, heading_vector) / np.dot(forward, heading_vector)
    return eye, forward, right, up, focal_px, horizon_px, target


def _agent_camera(x_m, y_m, heading_rad, style_name="chase"):
    return replay_video.agent_view_camera(
        x_m, y_m, heading_rad, _camera_style(style_name), AGENT_VIEW_WIDTH_PX, AGENT_VIEW_HEIGHT_PX
    )


def _camera_depth(camera, points):
    return (np.asarray(points, dtype=np.float64) - camera.eye_m) @ camera.forward


def _polygon_normal(points):
    normal = np.zeros(3)
    for point_idx in range(len(points)):
        current = points[point_idx]
        following = points[(point_idx + 1) % len(points)]
        normal += np.cross(current, following)
    return normal


def _points_in_camera_basis(camera, depths, rights, ups):
    return np.array(
        [
            camera.eye_m + depth * camera.forward + side * camera.right + height * camera.up
            for depth, side, height in zip(depths, rights, ups)
        ]
    )


@pytest.mark.parametrize("style_name", ["chase", "driver"])
@pytest.mark.parametrize("heading_rad", HEADINGS_RAD)
def test_agent_view_camera_follows_the_spec_model(style_name, heading_rad):
    style = _camera_style(style_name)
    eye, forward, right, up, focal_px, horizon_px, _ = _expected_camera_vectors(
        10.0, -5.0, heading_rad, style, AGENT_VIEW_HEIGHT_PX
    )

    camera = _agent_camera(10.0, -5.0, heading_rad, style_name)

    assert isinstance(camera, replay_video.AgentViewCamera)
    np.testing.assert_allclose(camera.eye_m, eye, atol=1e-12)
    np.testing.assert_allclose(camera.forward, forward, atol=1e-12)
    np.testing.assert_allclose(camera.right, right, atol=1e-12)
    np.testing.assert_allclose(camera.up, up, atol=1e-12)
    assert camera.focal_length_px == pytest.approx(focal_px, abs=1e-9)
    assert camera.horizon_y_px == pytest.approx(horizon_px, abs=1e-9)
    assert camera.image_width_px == AGENT_VIEW_WIDTH_PX
    assert camera.image_height_px == AGENT_VIEW_HEIGHT_PX
    assert camera.right[2] == pytest.approx(0.0, abs=1e-12)
    assert camera.up[2] > 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        camera.focal_length_px = 1.0


def test_driver_camera_sits_in_the_seat():
    camera = _agent_camera(3.0, 4.0, 0.0, "driver")
    np.testing.assert_allclose(camera.eye_m, [3.0, 4.0, 1.4], atol=1e-12)
    expected_focal_px = (AGENT_VIEW_HEIGHT_PX / 2.0) / math.tan(math.radians(30.0))
    assert camera.focal_length_px == pytest.approx(expected_focal_px, abs=1e-9)


@pytest.mark.parametrize("style_name", ["chase", "driver"])
@pytest.mark.parametrize("heading_rad", HEADINGS_RAD)
def test_camera_target_projects_to_the_image_centre(style_name, heading_rad):
    style = _camera_style(style_name)
    eye, _, _, _, _, _, target = _expected_camera_vectors(-40.0, 25.0, heading_rad, style, AGENT_VIEW_HEIGHT_PX)
    camera = _agent_camera(-40.0, 25.0, heading_rad, style_name)

    pixels, depth_m = replay_video.project_points(target[None], camera)

    np.testing.assert_allclose(pixels[0], [AGENT_VIEW_WIDTH_PX / 2.0, AGENT_VIEW_HEIGHT_PX / 2.0], atol=1e-9)
    assert depth_m[0] == pytest.approx(np.linalg.norm(target - eye), abs=1e-9)


@pytest.mark.parametrize("heading_rad", HEADINGS_RAD)
def test_points_to_the_agents_left_project_left_of_centre(heading_rad):
    style = _camera_style("chase")
    _, _, _, _, _, _, target = _expected_camera_vectors(0.0, 0.0, heading_rad, style, AGENT_VIEW_HEIGHT_PX)
    left = np.array([-math.sin(heading_rad), math.cos(heading_rad), 0.0])
    camera = _agent_camera(0.0, 0.0, heading_rad)
    points = np.stack([target + 5.0 * left, target - 5.0 * left, target - np.array([0.0, 0.0, 1.0])])

    pixels, depth_m = replay_video.project_points(points, camera)

    assert pixels[0, 0] < AGENT_VIEW_WIDTH_PX / 2.0
    assert pixels[1, 0] > AGENT_VIEW_WIDTH_PX / 2.0
    assert pixels[2, 1] > AGENT_VIEW_HEIGHT_PX / 2.0
    assert np.all(depth_m > 0.0)


def test_project_points_matches_the_pinhole_formula():
    camera = _agent_camera(12.0, -7.0, 0.7)
    points = np.random.default_rng(SEED).uniform([-60.0, -60.0, 0.0], [80.0, 60.0, 4.0], size=(64, 3))
    offsets = points - camera.eye_m
    camera_x = offsets @ camera.right
    camera_y = offsets @ camera.up
    camera_z = offsets @ camera.forward
    keep = np.abs(camera_z) > 0.5
    points, camera_x, camera_y, camera_z = points[keep], camera_x[keep], camera_y[keep], camera_z[keep]

    pixels, depth_m = replay_video.project_points(points, camera)

    assert pixels.shape == (len(points), 2)
    assert depth_m.shape == (len(points),)
    np.testing.assert_allclose(
        pixels[:, 0], AGENT_VIEW_WIDTH_PX / 2.0 + camera.focal_length_px * camera_x / camera_z, atol=1e-6
    )
    np.testing.assert_allclose(
        pixels[:, 1], AGENT_VIEW_HEIGHT_PX / 2.0 - camera.focal_length_px * camera_y / camera_z, atol=1e-6
    )
    np.testing.assert_allclose(depth_m, camera_z, atol=1e-9)


def test_points_behind_the_eye_have_negative_depth_and_are_not_culled():
    camera = _agent_camera(5.0, 5.0, -0.4)
    points = np.stack([camera.eye_m - 10.0 * camera.forward, camera.eye_m + 3.0 * camera.forward])

    pixels, depth_m = replay_video.project_points(points, camera)

    assert depth_m[0] == pytest.approx(-10.0, abs=1e-9)
    assert depth_m[1] == pytest.approx(3.0, abs=1e-9)
    assert pixels.shape == (2, 2)
    assert np.all(np.isfinite(pixels))


@pytest.mark.parametrize("style_name", ["chase", "driver"])
def test_horizon_is_where_far_ground_points_land(style_name):
    heading_rad = 0.9
    camera = _agent_camera(20.0, 30.0, heading_rad, style_name)
    far_ground = np.array([[20.0 + 1e7 * math.cos(heading_rad), 30.0 + 1e7 * math.sin(heading_rad), 0.0]])

    pixels, _ = replay_video.project_points(far_ground, camera)

    assert pixels[0, 1] == pytest.approx(camera.horizon_y_px, abs=1e-2)
    assert camera.horizon_y_px < AGENT_VIEW_HEIGHT_PX / 2.0


def test_clip_returns_a_polygon_fully_in_front_unchanged():
    camera = _agent_camera(0.0, 0.0, 0.3)
    polygon = _points_in_camera_basis(camera, [20.0, 22.0, 24.0, 21.0], [-2.0, -2.5, 1.0, 2.0], [-1.0, 0.0, 1.0, 0.5])

    clipped = replay_video.clip_polygon_near(polygon, camera, 0.5)

    np.testing.assert_array_equal(clipped, polygon)


def test_clip_drops_a_polygon_fully_behind_the_near_plane():
    camera = _agent_camera(0.0, 0.0, 0.3)
    polygon = _points_in_camera_basis(camera, [-5.0, 0.2, 0.4], [0.0, 1.0, -1.0], [0.0, 0.5, 0.5])

    clipped = replay_video.clip_polygon_near(polygon, camera, 0.5)

    assert clipped.shape == (0, 3)


def test_clip_one_vertex_behind_a_square_gives_a_pentagon():
    camera = _agent_camera(0.0, 0.0, -1.1)
    square = _points_in_camera_basis(camera, [-1.0, 2.0, 3.0, 2.0], [0.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0])

    clipped = replay_video.clip_polygon_near(square, camera, 0.5)

    assert clipped.shape == (5, 3)
    assert np.all(_camera_depth(camera, clipped) >= 0.5 - 1e-9)


def test_clip_one_vertex_in_front_of_a_triangle_gives_a_triangle():
    camera = _agent_camera(0.0, 0.0, 2.0)
    triangle = _points_in_camera_basis(camera, [5.0, -1.0, -2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0])

    clipped = replay_video.clip_polygon_near(triangle, camera, 0.5)

    assert clipped.shape == (3, 3)
    np.testing.assert_allclose(_camera_depth(camera, clipped).max(), 5.0, atol=1e-9)
    assert np.all(_camera_depth(camera, clipped) >= 0.5 - 1e-9)


def _random_convex_polygon(rng, camera, vertex_count):
    centre_depth = rng.uniform(-3.0, 4.0)
    radius_m = rng.uniform(1.0, 6.0)
    angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, size=vertex_count))
    tilt = _unit(rng.normal(size=3))
    axis_u = _unit(np.cross(tilt, [0.3, 0.5, 0.8]))
    axis_v = np.cross(tilt, axis_u)
    centre = camera.eye_m + centre_depth * camera.forward + rng.uniform(-3.0, 3.0) * camera.right
    return np.array([centre + radius_m * (math.cos(a) * axis_u + math.sin(a) * axis_v) for a in angles])


def _inside_convex_polygon(point, polygon, normal):
    for vertex_idx in range(len(polygon)):
        edge = polygon[(vertex_idx + 1) % len(polygon)] - polygon[vertex_idx]
        if np.dot(np.cross(edge, point - polygon[vertex_idx]), normal) < -1e-7:
            return False
    return True


def test_clip_polygon_near_properties_on_random_convex_polygons():
    rng = np.random.default_rng(SEED)
    near_plane_m = replay_format.REPLAY_VIEW_STYLE["near_plane_m"]
    camera = _agent_camera(7.0, -3.0, 0.6)
    straddling_count = 0
    for _ in range(300):
        vertex_count = int(rng.integers(3, 9))
        polygon = _random_convex_polygon(rng, camera, vertex_count)
        input_normal = _polygon_normal(polygon)

        clipped = replay_video.clip_polygon_near(polygon, camera, near_plane_m)

        assert clipped.ndim == 2 and clipped.shape[1] == 3
        assert clipped.shape[0] == 0 or 3 <= clipped.shape[0] <= vertex_count + 1
        depths = _camera_depth(camera, polygon)
        if np.all(depths >= near_plane_m):
            np.testing.assert_array_equal(clipped, polygon)
            continue
        if np.all(depths < near_plane_m):
            assert clipped.shape[0] == 0
            continue
        straddling_count += 1
        assert clipped.shape[0] >= 3
        assert np.all(_camera_depth(camera, clipped) >= near_plane_m - 1e-9)
        plane_offsets = (clipped - polygon[0]) @ _unit(input_normal)
        np.testing.assert_allclose(plane_offsets, 0.0, atol=1e-9)
        assert np.dot(_polygon_normal(clipped), input_normal) > 0.0
        for point in clipped:
            assert _inside_convex_polygon(point, polygon, input_normal)
        kept_inputs = polygon[depths >= near_plane_m]
        for vertex in kept_inputs:
            assert np.min(np.linalg.norm(clipped - vertex, axis=1)) < 1e-9
    assert straddling_count > 50


def _agent_tracks(frame_count, agent_count):
    agent_i32 = np.zeros((frame_count, agent_count, binding.AGENT_I32_FIELDS), dtype=np.int32)
    agent_i32[..., replay_format.AGENT_I32_VALID_IDX] = 1
    agent_i32[..., replay_format.AGENT_I32_SLOT_IDX] = -1
    agent_i32[..., replay_format.AGENT_I32_ID_IDX] = np.arange(agent_count) + 100
    return agent_i32


def test_select_tracked_agent_picks_the_lowest_active_slot():
    agent_i32 = _agent_tracks(6, 4)
    agent_i32[:, 1, replay_format.AGENT_I32_SLOT_IDX] = 1
    agent_i32[:, 2, replay_format.AGENT_I32_SLOT_IDX] = 0
    agent_i32[:, 3, replay_format.AGENT_I32_SLOT_IDX] = 2

    assert replay_video.select_tracked_agent(agent_i32) == (2, 0, 6)


@pytest.mark.parametrize("disqualifying_field", ["invalid", "removed"])
def test_select_tracked_agent_skips_agents_missing_at_frame_zero(disqualifying_field):
    agent_i32 = _agent_tracks(5, 3)
    agent_i32[:, 0, replay_format.AGENT_I32_SLOT_IDX] = 0
    agent_i32[:, 2, replay_format.AGENT_I32_SLOT_IDX] = 1
    if disqualifying_field == "invalid":
        agent_i32[0, 0, replay_format.AGENT_I32_VALID_IDX] = 0
    else:
        agent_i32[0, 0, replay_format.AGENT_I32_REMOVED_IDX] = 1

    assert replay_video.select_tracked_agent(agent_i32) == (2, 1, 5)


@pytest.mark.parametrize(
    "invalid_frame, removed_frame, expected_frame_count",
    [(4, None, 4), (None, 3, 3), (5, 2, 2), (1, None, 1)],
)
def test_select_tracked_agent_counts_frames_up_to_the_first_gap(invalid_frame, removed_frame, expected_frame_count):
    agent_i32 = _agent_tracks(8, 2)
    agent_i32[:, 1, replay_format.AGENT_I32_SLOT_IDX] = 0
    if invalid_frame is not None:
        agent_i32[invalid_frame, 1, replay_format.AGENT_I32_VALID_IDX] = 0
    if removed_frame is not None:
        agent_i32[removed_frame:, 1, replay_format.AGENT_I32_REMOVED_IDX] = 1

    assert replay_video.select_tracked_agent(agent_i32) == (1, 0, expected_frame_count)


def test_select_tracked_agent_raises_without_a_candidate():
    no_active = _agent_tracks(3, 3)
    with pytest.raises(ValueError):
        replay_video.select_tracked_agent(no_active)
    all_invalid = _agent_tracks(3, 3)
    all_invalid[:, :, replay_format.AGENT_I32_SLOT_IDX] = [0, 1, 2]
    all_invalid[0, :, replay_format.AGENT_I32_VALID_IDX] = 0
    with pytest.raises(ValueError):
        replay_video.select_tracked_agent(all_invalid)


def _world_view_agents():
    frame_count, agent_count = 3, 4
    agent_f32 = np.zeros((frame_count, agent_count, binding.AGENT_F32_FIELDS), dtype=np.float32)
    agent_i32 = _agent_tracks(frame_count, agent_count)
    agent_f32[:, 0, :2] = [[100.0, -50.0], [150.0, 0.0], [200.0, 20.0]]
    agent_f32[:, 1, :2] = [[300.0, 50.0], [290.0, 40.0], [280.0, 30.0]]
    agent_f32[:, 2, :2] = [5000.0, 5000.0]
    agent_i32[:, 2, replay_format.AGENT_I32_VALID_IDX] = 0
    agent_f32[:, 3, :2] = [-3000.0, 0.0]
    agent_i32[:, 3, replay_format.AGENT_I32_REMOVED_IDX] = 1
    return agent_f32, agent_i32


def test_world_view_frames_the_valid_agents_without_roads():
    agent_f32, agent_i32 = _world_view_agents()
    no_roads = np.zeros((1, 2), dtype=np.float32)

    x0_m, y1_m, pixels_per_meter = replay_video.world_view_transform(
        agent_f32, agent_i32, no_roads, np.array([0], dtype=np.int32), WORLD_VIEW_SIZE_PX
    )

    side_m = 280.0
    assert x0_m == pytest.approx(60.0)
    assert y1_m == pytest.approx(140.0)
    assert pixels_per_meter == pytest.approx(WORLD_VIEW_SIZE_PX / side_m)
    north_east_px = ((340.0 - x0_m) * pixels_per_meter, (y1_m - 90.0) * pixels_per_meter)
    assert north_east_px[0] > WORLD_VIEW_SIZE_PX / 2.0 and north_east_px[1] < WORLD_VIEW_SIZE_PX / 2.0
    south_west_px = ((60.0 - x0_m) * pixels_per_meter, (y1_m - (-90.0)) * pixels_per_meter)
    assert south_west_px[0] < WORLD_VIEW_SIZE_PX / 2.0 and south_west_px[1] > WORLD_VIEW_SIZE_PX / 2.0


def test_world_view_intersects_the_agent_box_with_the_road_extent():
    agent_f32, agent_i32 = _world_view_agents()
    road_points = np.array([[150.0, -500.0], [250.0, 500.0], [200.0, 0.0], [1e6, 1e6]], dtype=np.float32)
    road_lengths = np.array([2, 1], dtype=np.int32)

    x0_m, y1_m, pixels_per_meter = replay_video.world_view_transform(
        agent_f32, agent_i32, road_points, road_lengths, WORLD_VIEW_SIZE_PX
    )

    side_m = 180.0
    assert x0_m == pytest.approx(200.0 - side_m / 2.0)
    assert y1_m == pytest.approx(side_m / 2.0)
    assert pixels_per_meter == pytest.approx(WORLD_VIEW_SIZE_PX / side_m)


def test_world_view_keeps_a_minimum_side():
    agent_f32 = np.zeros((2, 1, binding.AGENT_F32_FIELDS), dtype=np.float32)
    agent_f32[:, 0, :2] = [10.0, -20.0]
    agent_i32 = _agent_tracks(2, 1)

    x0_m, y1_m, pixels_per_meter = replay_video.world_view_transform(
        agent_f32, agent_i32, np.zeros((1, 2), np.float32), np.array([0], np.int32), WORLD_VIEW_SIZE_PX
    )

    assert x0_m == pytest.approx(10.0 - WORLD_VIEW_MIN_SIDE_M / 2.0)
    assert y1_m == pytest.approx(-20.0 + WORLD_VIEW_MIN_SIDE_M / 2.0)
    assert pixels_per_meter == pytest.approx(WORLD_VIEW_SIZE_PX / WORLD_VIEW_MIN_SIDE_M)


def test_bev_transform_puts_the_ego_frame_on_the_image():
    road_front_m, road_behind_m = 120.0, 20.0
    header = {
        "obs_range_m": {
            "road_front": road_front_m,
            "road_behind": road_behind_m,
            "road_side": 30.0,
            "partner": 100.0,
            "traffic_control": 100.0,
        }
    }

    ego_px_x, ego_px_y, pixels_per_meter = replay_video.bev_ego_pixel_transform(header, BEV_VIEW_SIZE_PX)

    def to_pixels(forward_m, left_m):
        return ego_px_x - pixels_per_meter * left_m, ego_px_y - pixels_per_meter * forward_m

    assert pixels_per_meter == pytest.approx(BEV_VIEW_SIZE_PX / (road_front_m + road_behind_m))
    assert (ego_px_x, ego_px_y) == pytest.approx((BEV_VIEW_SIZE_PX / 2.0, pixels_per_meter * road_front_m))
    assert to_pixels(road_front_m, 0.0)[1] == pytest.approx(0.0, abs=1e-9)
    assert to_pixels(-road_behind_m, 0.0)[1] == pytest.approx(BEV_VIEW_SIZE_PX, abs=1e-9)
    assert to_pixels(0.0, (road_front_m + road_behind_m) / 2.0)[0] == pytest.approx(0.0, abs=1e-9)
    ahead_left = to_pixels(10.0, 5.0)
    assert ahead_left[0] < ego_px_x and ahead_left[1] < ego_px_y


def _ffprobe_frame_count(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def _ffprobe_stream(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,color_space,color_primaries,color_transfer",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def _decoded_frames(path, width_px, height_px):
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height_px, width_px, 3)


def _gradient_frames(frame_count, width_px, height_px):
    for frame_idx in range(frame_count):
        frame = np.zeros((height_px, width_px, 3), dtype=np.uint8)
        frame[..., 0] = (np.arange(width_px) * 4 + frame_idx * 9) % 256
        frame[..., 1] = (np.arange(height_px)[:, None] * 5) % 256
        frame[..., 2] = frame_idx * 20
        yield frame


@requires_ffmpeg
def test_write_mp4_streams_every_frame(tmp_path):
    video_path = tmp_path / "gradient.mp4"

    written = replay_video.write_mp4(_gradient_frames(7, 64, 48), str(video_path), 64, 48)

    assert written == 7
    assert _ffprobe_frame_count(video_path) == 7
    stream = _ffprobe_stream(video_path)
    assert stream["codec_name"] == "h264"
    assert (stream["width"], stream["height"]) == (64, 48)
    assert stream["pix_fmt"] == "yuv420p"
    assert stream["r_frame_rate"] == f"{replay_format.REPLAY_VIEW_STYLE['replay_frames_per_second']}/1"
    assert stream["color_space"] == "bt709"
    assert stream["color_primaries"] == "bt709"
    assert stream["color_transfer"] == "bt709"


@requires_ffmpeg
def test_write_mp4_accepts_a_single_frame_list(tmp_path):
    frames = list(_gradient_frames(1, 32, 32))
    assert replay_video.write_mp4(frames, str(tmp_path / "one.mp4"), 32, 32) == 1
    assert _ffprobe_frame_count(tmp_path / "one.mp4") == 1


BAD_FRAME_CASES = {
    "odd_width": (lambda: _gradient_frames(2, 65, 48), 65, 48),
    "odd_height": (lambda: _gradient_frames(2, 64, 47), 64, 47),
    "float_frame": (lambda: [np.zeros((48, 64, 3), dtype=np.float32)], 64, 48),
    "rgba_frame": (lambda: [np.zeros((48, 64, 4), dtype=np.uint8)], 64, 48),
    "transposed_frame": (lambda: [np.zeros((64, 48, 3), dtype=np.uint8)], 64, 48),
    "bad_frame_mid_stream": (lambda: [next(_gradient_frames(1, 64, 48)), np.zeros((48, 64), np.uint8)], 64, 48),
    "no_frames": (lambda: iter(()), 64, 48),
}


@requires_ffmpeg
@pytest.mark.parametrize("case_name", sorted(BAD_FRAME_CASES))
def test_write_mp4_rejects_bad_frames(tmp_path, case_name):
    make_frames, width_px, height_px = BAD_FRAME_CASES[case_name]
    with pytest.raises(ValueError):
        replay_video.write_mp4(make_frames(), str(tmp_path / f"{case_name}.mp4"), width_px, height_px)


@requires_ffmpeg
def test_write_mp4_raises_when_ffmpeg_fails(tmp_path):
    with pytest.raises(RuntimeError):
        replay_video.write_mp4(_gradient_frames(2, 32, 32), str(tmp_path / "video.not_a_container"), 32, 32)


def _normalized_env_config(overrides):
    with patch.object(sys, "argv", ["pufferl.py"]):
        args = pufferl.load_config("puffer_drive")
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None
    env_config = dict(normalize_puffer_drive_config(args, "test")["env"])
    env_config.update(overrides)
    return env_config


def _write_replay(path, include_observations):
    env_config = _normalized_env_config(REPLAY_ENV_OVERRIDES)
    env = Drive(**dict(env_config, capture_replay=True))
    rng = np.random.default_rng(SEED)
    try:
        env.reset(seed=SEED)
        observation_history = []
        for _ in range(REPLAY_STEP_COUNT):
            observation_history.append(env.observations.copy())
            env.step(rng.integers(0, env.single_action_space.n, size=env.num_agents))
        capture = env._replay_captures[0]
        layout_counts = env.observation_layout()
    finally:
        env.close()
    frames = {key: np.stack(values, axis=0) for key, values in capture["frames"].items()}
    active_count = capture["metadata"]["active_agent_count"]
    active_offset = capture["metadata"]["active_agent_offset"]
    action_count = len(binding.ACCELERATION_VALUES) * len(binding.STEERING_VALUES)
    replay = {
        "env": env_config,
        **frames,
        "raw_action": np.zeros((REPLAY_STEP_COUNT, active_count), np.float32),
        "clipped_action": np.zeros((REPLAY_STEP_COUNT, active_count), np.float32),
        "value": np.zeros((REPLAY_STEP_COUNT, active_count), np.float32),
        "entropy": np.zeros((REPLAY_STEP_COUNT, active_count), np.float32),
        "policy_probs": np.full((REPLAY_STEP_COUNT, active_count, action_count), 1.0 / action_count, np.float32),
    }
    if include_observations:
        replay["obs"] = np.stack(observation_history)[:, active_offset : active_offset + active_count].astype(
            np.float16
        )
        replay["obs_layout"] = layout_counts
    pufferlib.viz.save_interactive_replay_zlib(capture["scenario"], replay, str(path))
    return path


@pytest.fixture(scope="module")
def replay_files(tmp_path_factory):
    replay_dir = tmp_path_factory.mktemp("replay_sources")
    return {
        "with_obs": _write_replay(replay_dir / f"with_obs{REPLAY_SUFFIX}", include_observations=True),
        "without_obs": _write_replay(replay_dir / f"without_obs{REPLAY_SUFFIX}", include_observations=False),
    }


def _replay_chunks(replay_path):
    return replay_format.decode_interactive_replay(Path(replay_path).read_bytes())


def _expected_frame_counts(replay_path):
    header, chunks = _replay_chunks(replay_path)
    tracked_frame_count = replay_video.select_tracked_agent(chunks["agent_i32"])[2]
    return {"world": header["frames"], "bev": tracked_frame_count, "agent": tracked_frame_count}


@requires_ffmpeg
def test_render_replay_videos_writes_one_mp4_per_view(tmp_path, replay_files):
    replay_path = replay_files["with_obs"]
    expected_frames = _expected_frame_counts(replay_path)
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    rendered = replay_video.render_replay_videos(str(replay_path), str(video_dir), list(RENDER_VIEWS))

    assert sorted(video.view for video in rendered) == sorted(RENDER_VIEWS)
    assert expected_frames["world"] == REPLAY_STEP_COUNT
    for video in rendered:
        assert isinstance(video, replay_video.RenderedVideo)
        assert video.stem == "with_obs"
        assert Path(video.path) == video_dir / f"with_obs__{video.view}.mp4"
        assert Path(video.path).is_file()
        assert video.frame_count == expected_frames[video.view]
        assert _ffprobe_frame_count(video.path) == expected_frames[video.view]
        stream = _ffprobe_stream(video.path)
        assert (stream["width"], stream["height"]) == VIEW_SIZES_PX[video.view]
        with pytest.raises(dataclasses.FrozenInstanceError):
            video.frame_count = 0


@requires_ffmpeg
def test_rendered_views_draw_content(tmp_path, replay_files):
    replay_path = replay_files["with_obs"]
    rendered = {
        video.view: video
        for video in replay_video.render_replay_videos(str(replay_path), str(tmp_path), list(RENDER_VIEWS))
    }
    world_frames = _decoded_frames(rendered["world"].path, *VIEW_SIZES_PX["world"])
    assert world_frames[0].std() > MIN_PIXEL_STD
    assert np.abs(world_frames[0].astype(np.int16) - world_frames[-1].astype(np.int16)).mean() > 0.0
    bev_frames = _decoded_frames(rendered["bev"].path, *VIEW_SIZES_PX["bev"])
    assert bev_frames[0].std() > MIN_PIXEL_STD

    header, chunks = _replay_chunks(replay_path)
    agent_idx, _, _ = replay_video.select_tracked_agent(chunks["agent_i32"])
    agent_state = chunks["agent_f32"][0, agent_idx]
    camera = _agent_camera(
        float(agent_state[replay_format.AGENT_F32_X_IDX]),
        float(agent_state[replay_format.AGENT_F32_Y_IDX]),
        float(agent_state[replay_format.AGENT_F32_HEADING_IDX]),
    )
    agent_frame = _decoded_frames(rendered["agent"].path, *VIEW_SIZES_PX["agent"])[0].astype(np.float64)
    horizon_row = int(camera.horizon_y_px)
    sky_colour = agent_frame[: max(1, horizon_row - 10)].mean(axis=(0, 1))
    ground_colour = agent_frame[-40:].mean(axis=(0, 1))
    assert np.abs(sky_colour - ground_colour).max() > MIN_SKY_GROUND_CONTRAST


@requires_ffmpeg
def test_render_replay_videos_bev_needs_observations(tmp_path, replay_files):
    with pytest.raises(ValueError):
        replay_video.render_replay_videos(str(replay_files["without_obs"]), str(tmp_path), ["bev"])
    rendered = replay_video.render_replay_videos(str(replay_files["without_obs"]), str(tmp_path), ["world"])
    assert [video.view for video in rendered] == ["world"]
    assert _ffprobe_frame_count(rendered[0].path) == REPLAY_STEP_COUNT


def _run_cli(replay_dir, video_dir, views, delete_replays):
    command = [
        sys.executable,
        "-m",
        "pufferlib.replay_video",
        "--replay-dir",
        str(replay_dir),
        "--video-dir",
        str(video_dir),
        "--views",
        views,
    ]
    if delete_replays:
        command.append("--delete-replays")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run(
        command, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=CLI_TIMEOUT_SECONDS
    )


def _copy_replays(replay_dir, sources):
    replay_dir.mkdir()
    for stem, source in sources.items():
        if isinstance(source, bytes):
            (replay_dir / f"{stem}{REPLAY_SUFFIX}").write_bytes(source)
            continue
        shutil.copyfile(source, replay_dir / f"{stem}{REPLAY_SUFFIX}")


def _read_manifest(video_dir):
    return json.loads((video_dir / "manifest.json").read_text())


@requires_ffmpeg
def test_cli_writes_a_manifest_and_deletes_replays_on_success(tmp_path, replay_files):
    replay_dir, video_dir = tmp_path / "replays", tmp_path / "videos"
    _copy_replays(replay_dir, {"alpha": replay_files["with_obs"], "beta": replay_files["with_obs"]})
    expected_frames = _expected_frame_counts(replay_files["with_obs"])

    result = _run_cli(replay_dir, video_dir, "world,agent", delete_replays=True)

    assert result.returncode == 0, result.stdout + result.stderr
    manifest = _read_manifest(video_dir)
    assert set(manifest) == {"videos", "failed", "git_commit"}
    assert manifest["failed"] == []
    assert manifest["git_commit"] is None or isinstance(manifest["git_commit"], str)
    entries = {(video["stem"], video["view"]): video for video in manifest["videos"]}
    assert set(entries) == {(stem, view) for stem in ("alpha", "beta") for view in ("world", "agent")}
    for (stem, view), video in entries.items():
        assert set(video) == {"stem", "view", "path", "frame_count"}
        assert Path(video["path"]).resolve() == (video_dir / f"{stem}__{view}.mp4").resolve()
        assert video["frame_count"] == expected_frames[view]
        assert _ffprobe_frame_count(video["path"]) == expected_frames[view]
    assert not replay_dir.exists()


@requires_ffmpeg
def test_cli_keeps_replays_without_the_delete_flag(tmp_path, replay_files):
    replay_dir, video_dir = tmp_path / "replays", tmp_path / "videos"
    _copy_replays(replay_dir, {"alpha": replay_files["with_obs"]})

    result = _run_cli(replay_dir, video_dir, "world", delete_replays=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (replay_dir / f"alpha{REPLAY_SUFFIX}").is_file()
    assert [(video["stem"], video["view"]) for video in _read_manifest(video_dir)["videos"]] == [("alpha", "world")]


@requires_ffmpeg
def test_cli_partial_failure_keeps_replays_and_exits_one(tmp_path, replay_files):
    replay_dir, video_dir = tmp_path / "replays", tmp_path / "videos"
    _copy_replays(
        replay_dir,
        {"alpha": replay_files["with_obs"], "broken": b"not a replay payload", "noobs": replay_files["without_obs"]},
    )

    result = _run_cli(replay_dir, video_dir, "world,bev", delete_replays=True)

    assert result.returncode == 1, result.stdout + result.stderr
    manifest = _read_manifest(video_dir)
    assert sorted(manifest["failed"]) == ["broken", "noobs"]
    succeeded = {(video["stem"], video["view"]) for video in manifest["videos"]}
    assert {("alpha", "world"), ("alpha", "bev")} <= succeeded
    assert not any(stem == "broken" for stem, _ in succeeded)
    assert sorted(path.name for path in replay_dir.iterdir()) == sorted(
        f"{stem}{REPLAY_SUFFIX}" for stem in ("alpha", "broken", "noobs")
    )


def test_cli_rejects_an_unknown_view(tmp_path, replay_files):
    replay_dir, video_dir = tmp_path / "replays", tmp_path / "videos"
    _copy_replays(replay_dir, {"alpha": replay_files["with_obs"]})

    result = _run_cli(replay_dir, video_dir, "world,top", delete_replays=True)

    assert result.returncode != 0
    assert not (video_dir / "manifest.json").exists()
    assert (replay_dir / f"alpha{REPLAY_SUFFIX}").is_file()


def test_cli_rejects_a_missing_or_empty_replay_dir(tmp_path):
    missing = _run_cli(tmp_path / "missing", tmp_path / "videos_missing", "world", delete_replays=False)
    assert missing.returncode != 0
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty = _run_cli(empty_dir, tmp_path / "videos_empty", "world", delete_replays=True)
    assert empty.returncode != 0
    assert empty_dir.is_dir()
