"""The CARLA renderer's pure parts: camera rigs, steering sign, easing, blueprints, ground pitch.

Nothing here connects to a CARLA server. The one live test is opt-in via WHEEL_DEMO_CARLA_TESTS=1.
"""

import math
import os
import queue
import types
from collections import deque

import numpy as np
import pytest

from pufferlib.wheel_demo import carla_scene
from pufferlib.wheel_demo.carla_scene import (
    CAMERA_MODE_AGENT,
    CAMERA_MODE_CHASE,
    CAMERA_MODE_DRIVER,
    CarlaScene,
    ease_toward,
    front_wheel_steer_deg,
    main_camera_rig,
    nearest_blueprint,
    project_to_image,
    rear_camera_rig,
)

CHASE_BACK_M = 9.0
CHASE_EYE_HEIGHT_M = 3.6
CHASE_TARGET_AHEAD_M = 14.0
CHASE_TARGET_HEIGHT_M = 1.6
CHASE_FOV_DEG = 45.0
DRIVER_SEAT_FORWARD_M = 0.30
DRIVER_SEAT_LEFT_M = 0.38
DRIVER_EYE_HEIGHT_M = 1.26
DRIVER_FOV_DEG = 60.0
AGENT_VIEW_SPAN_M = 60.0
AGENT_VIEW_HEIGHT_M = 100.0
STRAIGHT_DOWN_PITCH_DEG = -90.0
REAR_YAW_DEG = 180.0
SIM_LEFT = +1
SIM_RIGHT = -1
GROUND_SNAP_DISTANCE_M = 4.0
FRAME_DT_S = 1.0 / 60.0
TIME_CONSTANT_S = 0.12


def test_chase_camera_sits_behind_and_above_looking_at_a_point_ahead():
    rig = main_camera_rig(CAMERA_MODE_CHASE)
    assert (rig.x, rig.y, rig.z) == pytest.approx((-CHASE_BACK_M, 0.0, CHASE_EYE_HEIGHT_M))
    assert rig.fov_deg == CHASE_FOV_DEG
    assert rig.yaw_deg == 0.0
    look_down_rad = math.atan2(CHASE_EYE_HEIGHT_M - CHASE_TARGET_HEIGHT_M, CHASE_BACK_M + CHASE_TARGET_AHEAD_M)
    assert rig.pitch_deg == pytest.approx(-math.degrees(look_down_rad))
    assert rig.pitch_deg < 0.0


def test_driver_camera_sits_in_the_left_hand_seat():
    rig = main_camera_rig(CAMERA_MODE_DRIVER)
    driver_left_m = -rig.y
    assert driver_left_m == pytest.approx(DRIVER_SEAT_LEFT_M)
    assert rig.x == pytest.approx(DRIVER_SEAT_FORWARD_M)
    assert rig.z == pytest.approx(DRIVER_EYE_HEIGHT_M)
    assert rig.fov_deg == DRIVER_FOV_DEG
    assert rig.yaw_deg == 0.0
    assert -10.0 < rig.pitch_deg < 0.0


def test_agent_camera_looks_straight_down_over_a_60_metre_span():
    rig = main_camera_rig(CAMERA_MODE_AGENT)
    assert (rig.x, rig.y, rig.z) == pytest.approx((0.0, 0.0, AGENT_VIEW_HEIGHT_M))
    assert rig.pitch_deg == STRAIGHT_DOWN_PITCH_DEG
    ground_span_m = 2.0 * rig.z * math.tan(math.radians(rig.fov_deg) / 2.0)
    assert ground_span_m == pytest.approx(AGENT_VIEW_SPAN_M)


@pytest.mark.parametrize("mode", [-1, 3, 99])
def test_unknown_camera_mode_raises(mode):
    with pytest.raises(ValueError):
        main_camera_rig(mode)


@pytest.mark.parametrize("half_length_m", [2.0, 2.45])
def test_rear_camera_looks_back_from_the_tail(half_length_m):
    rig = rear_camera_rig(half_length_m)
    assert rig.x == pytest.approx(-half_length_m)
    assert rig.y == 0.0
    assert rig.yaw_deg == REAR_YAW_DEG
    assert rig.z > 0.0


@pytest.mark.parametrize("magnitude_rad", [0.05, 0.3, 0.55])
def test_driver_turning_right_steers_carlas_wheels_right(magnitude_rad):
    carla_steer_deg = front_wheel_steer_deg(SIM_RIGHT * magnitude_rad)
    carla_steer_points_right = carla_steer_deg > 0.0
    assert carla_steer_points_right
    assert abs(carla_steer_deg) == pytest.approx(math.degrees(magnitude_rad))


@pytest.mark.parametrize("magnitude_rad", [0.05, 0.3, 0.55])
def test_driver_turning_left_steers_carlas_wheels_left(magnitude_rad):
    assert front_wheel_steer_deg(SIM_LEFT * magnitude_rad) == pytest.approx(-math.degrees(magnitude_rad))


def test_straight_wheels_stay_straight():
    assert front_wheel_steer_deg(0.0) == 0.0


@pytest.mark.parametrize("current,target", [(0.0, 10.0), (5.0, -3.0), (-2.0, -2.5)])
def test_ease_never_overshoots_and_moves_monotonically_towards_the_target(current, target):
    previous_gap = abs(target - current)
    value = current
    for _ in range(200):
        value = ease_toward(value, target, FRAME_DT_S, TIME_CONSTANT_S)
        gap = abs(target - value)
        assert gap <= previous_gap
        assert min(current, target) <= value <= max(current, target)
        previous_gap = gap
    assert value == pytest.approx(target, abs=1e-6)


@pytest.mark.parametrize("dt_s", [1e-4, FRAME_DT_S, TIME_CONSTANT_S, 1.0, 100.0])
def test_ease_is_exponential_in_elapsed_time(dt_s):
    eased = ease_toward(0.0, 1.0, dt_s, TIME_CONSTANT_S)
    assert eased == pytest.approx(1.0 - math.exp(-dt_s / TIME_CONSTANT_S))
    assert 0.0 < eased <= 1.0


def test_ease_over_zero_time_stays_put():
    assert ease_toward(3.0, 9.0, 0.0, TIME_CONSTANT_S) == 3.0


def test_two_half_steps_equal_one_full_step():
    one_step = ease_toward(0.0, 1.0, 2.0 * FRAME_DT_S, TIME_CONSTANT_S)
    two_steps = ease_toward(ease_toward(0.0, 1.0, FRAME_DT_S, TIME_CONSTANT_S), 1.0, FRAME_DT_S, TIME_CONSTANT_S)
    assert two_steps == pytest.approx(one_step)


BLUEPRINT_SIZES = [("mini", 3.6, 1.7), ("sedan_a", 4.7, 1.85), ("sedan_b", 4.8, 1.9), ("van", 5.4, 2.1)]


def test_nearest_blueprint_picks_the_closest_box():
    assert nearest_blueprint(BLUEPRINT_SIZES, 3.5, 1.7, 0) == "mini"
    assert nearest_blueprint(BLUEPRINT_SIZES, 5.5, 2.2, 0) == "van"


def test_nearest_blueprint_rotates_through_near_ties_only():
    picks = [nearest_blueprint(BLUEPRINT_SIZES, 4.75, 1.87, agent_idx) for agent_idx in range(6)]
    assert set(picks) == {"sedan_a", "sedan_b"}
    assert picks[0] != picks[1]


def test_nearest_blueprint_is_deterministic_per_agent():
    assert nearest_blueprint(BLUEPRINT_SIZES, 4.75, 1.87, 3) == nearest_blueprint(BLUEPRINT_SIZES, 4.75, 1.87, 3)


class FakeLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class FakeMap:
    """CARLA's map: one road surface whose pitch is reported in [0, 360)."""

    def __init__(self, surface_z, pitch_deg, lane_yaw_deg):
        self.surface_z = surface_z
        self.pitch_deg = pitch_deg
        self.lane_yaw_deg = lane_yaw_deg
        self.queries = []

    def get_waypoint(self, location):
        self.queries.append((location.x, location.y, location.z))
        transform = types.SimpleNamespace(
            location=FakeLocation(location.x, location.y, self.surface_z),
            rotation=types.SimpleNamespace(pitch=self.pitch_deg, yaw=self.lane_yaw_deg),
        )
        return types.SimpleNamespace(transform=transform)


def scene_over(fake_map):
    scene = CarlaScene.__new__(CarlaScene)
    scene.carla = types.SimpleNamespace(Location=FakeLocation)
    scene.map = fake_map
    scene._ground = {}
    scene._poses = {}
    return scene


@pytest.mark.parametrize("reported_pitch_deg,uphill_deg", [(359.0, -1.0), (355.5, -4.5), (3.0, 3.0), (0.0, 0.0)])
def test_ground_pitch_unwraps_carlas_0_to_360_range(reported_pitch_deg, uphill_deg):
    scene = scene_over(FakeMap(surface_z=12.0, pitch_deg=reported_pitch_deg, lane_yaw_deg=40.0))
    z, pitch = scene._ground_pose(7, 10.0, 20.0, 40.0, 11.5, FRAME_DT_S)
    assert z == 12.0
    assert pitch == pytest.approx(uphill_deg)


def test_ground_pitch_flips_for_a_car_driving_against_the_lane():
    scene = scene_over(FakeMap(surface_z=0.0, pitch_deg=357.0, lane_yaw_deg=10.0))
    _, pitch = scene._ground_pose(1, 0.0, 0.0, 190.0, 0.0, FRAME_DT_S)
    assert pitch == pytest.approx(3.0)


def test_ground_pitch_vanishes_across_the_slope():
    scene = scene_over(FakeMap(surface_z=0.0, pitch_deg=355.0, lane_yaw_deg=0.0))
    _, pitch = scene._ground_pose(1, 0.0, 0.0, 90.0, 0.0, FRAME_DT_S)
    assert pitch == pytest.approx(0.0, abs=1e-9)


def test_ground_pose_queries_at_the_sims_height():
    fake_map = FakeMap(surface_z=0.0, pitch_deg=0.0, lane_yaw_deg=0.0)
    scene_over(fake_map)._ground_pose(1, 4.0, 5.0, 0.0, 6.5, FRAME_DT_S)
    assert fake_map.queries == [(4.0, 5.0, 6.5)]


def test_ground_height_eases_on_small_moves_and_snaps_on_jumps():
    fake_map = FakeMap(surface_z=0.0, pitch_deg=0.0, lane_yaw_deg=0.0)
    scene = scene_over(fake_map)
    scene._ground_pose(1, 0.0, 0.0, 0.0, 0.0, FRAME_DT_S)
    fake_map.surface_z = 1.0
    eased_z, _ = scene._ground_pose(1, 0.5, 0.0, 0.0, 0.0, FRAME_DT_S)
    assert 0.0 < eased_z < 1.0
    snapped_z, _ = scene._ground_pose(1, 0.5 + GROUND_SNAP_DISTANCE_M + 0.1, 0.0, 0.0, 0.0, FRAME_DT_S)
    assert snapped_z == 1.0


def test_ground_easing_is_per_actor():
    fake_map = FakeMap(surface_z=0.0, pitch_deg=0.0, lane_yaw_deg=0.0)
    scene = scene_over(fake_map)
    scene._ground_pose(1, 0.0, 0.0, 0.0, 0.0, FRAME_DT_S)
    fake_map.surface_z = 2.0
    first_seen_z, _ = scene._ground_pose(2, 0.0, 0.0, 0.0, 0.0, FRAME_DT_S)
    assert first_seen_z == 2.0


def test_switching_the_human_car_resets_the_ego_easing():
    scene = scene_over(FakeMap(surface_z=0.0, pitch_deg=0.0, lane_yaw_deg=0.0))
    scene._ground[("ego",)] = (0.0, 0.0, 5.0, 0.0)
    scene._poses[("ego",)] = carla_scene.SmoothedPose(1.0, 2.0, 3.0)
    scene._poses[3] = carla_scene.SmoothedPose(4.0, 5.0, 6.0)
    scene.set_human_agent(3)
    assert scene.human_agent_idx == 3
    assert ("ego",) not in scene._ground
    assert ("ego",) not in scene._poses
    assert 3 in scene._poses


def test_importing_the_scene_does_not_import_carla():
    assert not hasattr(carla_scene, "carla")


@pytest.mark.skipif(os.environ.get("WHEEL_DEMO_CARLA_TESTS") != "1", reason="needs a CARLA server")
def test_live_scene_spawns_one_actor_per_agent_and_cleans_up():
    import carla

    from pufferlib.ocean.drive.drive import Drive

    map_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "pufferlib",
        "resources",
        "drive",
        "binaries",
        "carla",
    )
    env = Drive(num_agents=4, min_agents_per_env=1, max_agents_per_env=4, num_maps=1, map_dir=map_dir)
    env.reset()
    scene = CarlaScene(carla, "Town01", env, 0, FRAME_DT_S)
    try:
        assert len(scene._actor_ids) == len(env.get_global_agent_state()["x"])
    finally:
        scene.close()
        env.close()


class FakeCarlaWorld:
    def __init__(self):
        self.applied_settings = []

    def get_map(self):
        return types.SimpleNamespace(name="Carla/Maps/Town01")

    def get_settings(self):
        return types.SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None, no_rendering_mode=True)

    def apply_settings(self, settings):
        self.applied_settings.append(settings)


class FakeCarlaClient:
    def __init__(self, host, port):
        self.world = FakeCarlaWorld()

    def set_timeout(self, seconds):
        self.timeout_s = seconds

    def get_available_maps(self):
        return ["/Game/Carla/Maps/Town01"]

    def get_world(self):
        return self.world


def scene_built_offline(monkeypatch):
    """Runs the real __init__ against a fake client, recording what the spawn steps saw."""
    seen_night = []
    monkeypatch.setattr(carla_scene.carla_bridge, "bin_path_for_town", lambda town: "town.bin")
    monkeypatch.setattr(carla_scene.carla_bridge, "town_offset", lambda path: (0.0, 0.0))
    monkeypatch.setattr(carla_scene.carla_bridge, "driving_waypoint_samples", lambda carla_map: [])
    monkeypatch.setattr(
        carla_scene.carla_bridge, "calibrate_town_offset", lambda samples, transform, path: ((0.0, 0.0), 0.0, None, 0.0)
    )
    monkeypatch.setattr(carla_scene.carla_bridge, "CarlaTransform", lambda town, **kwargs: types.SimpleNamespace())
    for step_name in ("_spawn_traffic", "_spawn_ego", "_setup_traffic_lights"):
        monkeypatch.setattr(
            CarlaScene, step_name, lambda self, *args, **kwargs: seen_night.append(getattr(self, "night", "unset"))
        )
    wheel_locations = types.SimpleNamespace(FL_Wheel=0, FR_Wheel=1, BL_Wheel=2, BR_Wheel=3)
    fake_carla = types.SimpleNamespace(Client=FakeCarlaClient, VehicleWheelLocation=wheel_locations)
    scene = CarlaScene(fake_carla, "Town01", env=None, human_agent_idx=0, substep_dt_s=FRAME_DT_S)
    return scene, seen_night


def test_a_new_scene_starts_in_daylight_before_anything_spawns(monkeypatch):
    scene, seen_night = scene_built_offline(monkeypatch)
    assert scene.night is False
    assert seen_night == [False, False, False]
    assert scene.human_agent_idx == 0


# ---------------------------------------------------------------------------------------------
# Projection into a camera image: camera frame x forward, y right, z up; image y grows downward.
# ---------------------------------------------------------------------------------------------

IMAGE_WIDTH_PX = 1280
IMAGE_HEIGHT_PX = 720
IMAGE_FOV_DEG = 90.0
CONTRACT_GOAL_BEACON_HEIGHT_M = 3.0
IDENTITY_CAMERA = np.eye(4)


def focal_px(fov_deg, width_px=IMAGE_WIDTH_PX):
    return width_px / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def project(point_xyz, world_to_camera=IDENTITY_CAMERA, fov_deg=IMAGE_FOV_DEG):
    return project_to_image(world_to_camera, point_xyz, IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX, fov_deg)


def yawed_camera(position_xyz, yaw_deg):
    """World-to-camera for a level camera at position_xyz, turned yaw_deg (CARLA: positive turns x toward y)."""
    yaw = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    rotation = np.stack([forward, right, up])
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = -rotation @ np.asarray(position_xyz)
    return world_to_camera


@pytest.mark.parametrize("distance_m", [0.5, 10.0, 400.0])
def test_a_point_straight_ahead_lands_on_the_image_centre(distance_m):
    assert project((distance_m, 0.0, 0.0)) == pytest.approx((IMAGE_WIDTH_PX / 2.0, IMAGE_HEIGHT_PX / 2.0))


def test_right_is_right_and_up_is_up_on_screen():
    u_right, v_right = project((10.0, 2.0, 0.0))
    u_up, v_up = project((10.0, 0.0, 2.0))
    assert u_right > IMAGE_WIDTH_PX / 2.0 and v_right == pytest.approx(IMAGE_HEIGHT_PX / 2.0)
    assert v_up < IMAGE_HEIGHT_PX / 2.0 and u_up == pytest.approx(IMAGE_WIDTH_PX / 2.0)
    u_left, v_down = project((10.0, -2.0, -2.0))
    assert u_left < IMAGE_WIDTH_PX / 2.0 and v_down > IMAGE_HEIGHT_PX / 2.0


@pytest.mark.parametrize("point_xyz", [(-10.0, 0.0, 0.0), (0.0, 5.0, 1.0), (-0.01, -3.0, 2.0)])
def test_a_point_behind_or_beside_the_camera_has_no_pixel(point_xyz):
    assert project(point_xyz) is None


def test_ninety_degree_fov_puts_a_45_degree_point_on_the_image_edge():
    u, _ = project((10.0, 10.0, 0.0), fov_deg=90.0)
    assert u == pytest.approx(IMAGE_WIDTH_PX)


@pytest.mark.parametrize("fov_deg", [30.0, 60.0, 90.0, 110.0])
def test_offsets_scale_with_the_focal_length_of_the_fov(fov_deg):
    u, v = project((20.0, 3.0, 1.0), fov_deg=fov_deg)
    assert u - IMAGE_WIDTH_PX / 2.0 == pytest.approx(focal_px(fov_deg) * 3.0 / 20.0)
    assert IMAGE_HEIGHT_PX / 2.0 - v == pytest.approx(focal_px(fov_deg) * 1.0 / 20.0)


def test_a_narrower_fov_spreads_the_same_point_further_from_centre():
    wide_u, _ = project((20.0, 3.0, 0.0), fov_deg=90.0)
    narrow_u, _ = project((20.0, 3.0, 0.0), fov_deg=45.0)
    assert narrow_u > wide_u > IMAGE_WIDTH_PX / 2.0


def test_projection_uses_the_full_world_to_camera_transform():
    camera = yawed_camera((100.0, 50.0, 3.0), 90.0)
    assert project((100.0, 60.0, 3.0), camera) == pytest.approx((IMAGE_WIDTH_PX / 2.0, IMAGE_HEIGHT_PX / 2.0))
    u, _ = project((98.0, 60.0, 3.0), camera)
    assert u > IMAGE_WIDTH_PX / 2.0
    assert project((100.0, 40.0, 3.0), camera) is None


class FakeTownTransform:
    def __init__(self, offset_xy):
        self.offset_xy = offset_xy

    def bin_to_loc(self, bx, by):
        return bx + self.offset_xy[0], by + self.offset_xy[1]


def fake_image(world_to_camera, fov_deg=IMAGE_FOV_DEG):
    matrix = [list(row) for row in world_to_camera]
    return types.SimpleNamespace(
        transform=types.SimpleNamespace(get_inverse_matrix=lambda: matrix),
        width=IMAGE_WIDTH_PX,
        height=IMAGE_HEIGHT_PX,
        fov=fov_deg,
    )


def goal_scene(road_z_m, offset_xy=(0.0, 0.0)):
    fake_map = FakeMap(surface_z=road_z_m, pitch_deg=0.0, lane_yaw_deg=0.0)
    scene = scene_over(fake_map)
    scene.transform = FakeTownTransform(offset_xy)
    return scene, fake_map


def goal_human(goal_x, goal_y):
    return {"goal_x": goal_x, "goal_y": goal_y, "sim_x": 0.0, "sim_y": 0.0}


def test_goal_constant_is_the_contract():
    assert carla_scene.GOAL_BEACON_HEIGHT_M == CONTRACT_GOAL_BEACON_HEIGHT_M


def test_goal_marker_stands_a_beacon_on_the_road_at_the_goal():
    scene, fake_map = goal_scene(road_z_m=-1.5, offset_xy=(5.0, 0.0))
    base, top = scene.goal_marker(goal_human(15.0, 0.0), fake_image(IDENTITY_CAMERA))
    f = focal_px(IMAGE_FOV_DEG)
    assert base == pytest.approx((IMAGE_WIDTH_PX / 2.0, IMAGE_HEIGHT_PX / 2.0 + f * 1.5 / 20.0))
    assert top == pytest.approx((IMAGE_WIDTH_PX / 2.0, IMAGE_HEIGHT_PX / 2.0 - f * 1.5 / 20.0))
    assert fake_map.queries == [(20.0, 0.0, 0.0)]


def test_goal_marker_queries_the_road_near_the_eased_ego_height():
    scene, fake_map = goal_scene(road_z_m=30.0)
    scene._ground[("ego",)] = (0.0, 0.0, 29.5, 0.0)
    scene.goal_marker(goal_human(12.0, 3.0), fake_image(IDENTITY_CAMERA))
    assert fake_map.queries == [(12.0, 3.0, 29.5)]


def test_goal_marker_follows_the_images_own_camera_and_fov():
    scene, _ = goal_scene(road_z_m=0.0)
    camera = yawed_camera((0.0, 0.0, 1.0), 90.0)
    base, _ = scene.goal_marker(goal_human(-3.0, 20.0), fake_image(camera, fov_deg=60.0))
    assert base == pytest.approx(
        (IMAGE_WIDTH_PX / 2.0 + focal_px(60.0) * 3.0 / 20.0, IMAGE_HEIGHT_PX / 2.0 + focal_px(60.0) * 1.0 / 20.0)
    )


def test_goal_marker_is_none_for_a_goal_behind_the_camera():
    scene, _ = goal_scene(road_z_m=0.0)
    assert scene.goal_marker(goal_human(-25.0, 4.0), fake_image(IDENTITY_CAMERA)) is None



# ---------------------------------------------------------------------------------------------
# Render pose smoother, wheel spin and the synchronous batch before the tick.
# ---------------------------------------------------------------------------------------------

SMOOTH_FRAME_DT_S = 1.0 / 60.0
CRUISE_SPEED_MPS = 10.0
CRUISE_YAW_RATE_DEG_S = 20.0
SETTLE_FRAME_COUNT = 600


def test_first_pose_and_a_respawn_snap():
    first = carla_scene.smooth_pose(None, 1.0, 2.0, 30.0, SMOOTH_FRAME_DT_S, carla_scene.POSE_SMOOTHING_SECONDS)
    assert (first.x, first.y, first.yaw_deg, first.vx, first.vy, first.yaw_rate_deg) == (1.0, 2.0, 30.0, 0.0, 0.0, 0.0)
    jump_m = carla_scene.GROUND_SNAP_DISTANCE_M + 0.1
    snapped = carla_scene.smooth_pose(first, 1.0 + jump_m, 2.0, 80.0, SMOOTH_FRAME_DT_S, carla_scene.POSE_SMOOTHING_SECONDS)
    assert (snapped.x, snapped.y, snapped.yaw_deg) == (1.0 + jump_m, 2.0, 80.0)


def test_a_gap_the_loop_cannot_follow_snaps():
    previous = carla_scene.SmoothedPose(0.0, 0.0, 0.0, 3.0, 0.0, 0.0)
    too_long_s = carla_scene.POSE_SMOOTHING_SECONDS
    snapped = carla_scene.smooth_pose(previous, 1.0, 0.5, 10.0, too_long_s, carla_scene.POSE_SMOOTHING_SECONDS)
    assert (snapped.x, snapped.y, snapped.yaw_deg, snapped.vx) == (1.0, 0.5, 10.0, 0.0)


def test_the_smoother_has_no_lag_at_constant_speed_and_turn_rate():
    pose = None
    for frame in range(SETTLE_FRAME_COUNT):
        t_s = frame * SMOOTH_FRAME_DT_S
        sim_yaw_deg = CRUISE_YAW_RATE_DEG_S * t_s
        pose = carla_scene.smooth_pose(pose, CRUISE_SPEED_MPS * t_s, 0.0, sim_yaw_deg, SMOOTH_FRAME_DT_S,
                                       carla_scene.POSE_SMOOTHING_SECONDS)
    assert pose.x == pytest.approx(CRUISE_SPEED_MPS * t_s, abs=1e-6)
    assert pose.vx == pytest.approx(CRUISE_SPEED_MPS, abs=1e-6)
    assert pose.yaw_rate_deg == pytest.approx(CRUISE_YAW_RATE_DEG_S, abs=1e-6)


def test_the_smoother_eases_a_step_instead_of_jumping():
    previous = carla_scene.SmoothedPose(0.0, 0.0, 0.0)
    step_m = 1.0
    pose = carla_scene.smooth_pose(previous, step_m, 0.0, 0.0, SMOOTH_FRAME_DT_S, carla_scene.POSE_SMOOTHING_SECONDS)
    assert 0.0 < pose.x < step_m


def test_yaw_error_takes_the_short_way_round():
    previous = carla_scene.SmoothedPose(0.0, 0.0, 179.0)
    pose = carla_scene.smooth_pose(previous, 0.0, 0.0, -179.0, SMOOTH_FRAME_DT_S, carla_scene.POSE_SMOOTHING_SECONDS)
    assert 179.0 < pose.yaw_deg < 181.0


class FakeSpinVehicle:
    def __init__(self):
        self.pitches = {}

    def set_wheel_pitch_angle(self, wheel, angle_deg):
        self.pitches[wheel] = angle_deg


WHEEL_RADIUS_M = 0.35
WHEEL_LOCATIONS = ("FL", "FR", "BL", "BR")


def spin_scene(vx, vy, yaw_deg, spin_deg=0.0):
    scene = CarlaScene.__new__(CarlaScene)
    scene._poses = {7: carla_scene.SmoothedPose(0.0, 0.0, yaw_deg, vx, vy, 0.0)}
    scene._wheel_radius_m = {42: WHEEL_RADIUS_M}
    scene._wheel_spin_deg = {42: spin_deg}
    scene._vehicles = {42: FakeSpinVehicle()}
    scene._wheel_locations = WHEEL_LOCATIONS
    return scene


@pytest.mark.parametrize("yaw_deg", [0.0, 35.0, -120.0])
@pytest.mark.parametrize("forward_mps", [5.0, -2.0])
def test_wheels_roll_by_the_distance_along_the_heading(yaw_deg, forward_mps):
    yaw_rad = math.radians(yaw_deg)
    scene = spin_scene(forward_mps * math.cos(yaw_rad), forward_mps * math.sin(yaw_rad), yaw_deg)
    scene._spin_wheels(42, 7, SMOOTH_FRAME_DT_S)
    roll_deg = carla_scene.WHEEL_SPIN_FORWARD_SIGN * math.degrees(forward_mps * SMOOTH_FRAME_DT_S / WHEEL_RADIUS_M)
    expected_deg = roll_deg % 360.0
    assert scene._wheel_spin_deg[42] == pytest.approx(expected_deg)
    assert set(scene._vehicles[42].pitches) == set(WHEEL_LOCATIONS)
    assert all(angle == pytest.approx(expected_deg) for angle in scene._vehicles[42].pitches.values())


def test_sideways_motion_does_not_spin_the_wheels():
    scene = spin_scene(0.0, 4.0, 0.0, spin_deg=12.0)
    scene._spin_wheels(42, 7, SMOOTH_FRAME_DT_S)
    assert scene._wheel_spin_deg[42] == pytest.approx(12.0)


def test_wheel_spin_wraps_at_a_full_turn():
    scene = spin_scene(CRUISE_SPEED_MPS, 0.0, 0.0, spin_deg=1.0)
    for _ in range(SETTLE_FRAME_COUNT):
        scene._spin_wheels(42, 7, SMOOTH_FRAME_DT_S)
        assert 0.0 <= scene._wheel_spin_deg[42] < 360.0


def test_wheel_constants_are_the_contract():
    assert carla_scene.WHEEL_SPIN_FORWARD_SIGN == -1.0
    assert carla_scene.CM_PER_M == 100.0


class RecordingClient:
    def __init__(self, log):
        self.log = log

    def apply_batch_sync(self, commands, due_tick_cue):
        self.log.append(("apply_batch_sync", len(commands), due_tick_cue))

    def apply_batch(self, commands):
        self.log.append(("apply_batch", len(commands)))


class RecordingWorld:
    def __init__(self, log):
        self.log = log
        self.frame_id = 0

    def tick(self):
        self.log.append(("tick",))
        self.frame_id += 1
        return self.frame_id


def test_update_applies_the_poses_synchronously_before_the_tick():
    log = []
    scene = CarlaScene.__new__(CarlaScene)
    scene.carla = types.SimpleNamespace(
        command=types.SimpleNamespace(ApplyTransform=lambda actor_id, transform: ("transform", actor_id)),
        Transform=lambda location: location,
        Location=lambda **kwargs: kwargs,
        VehicleWheelLocation=types.SimpleNamespace(FL_Wheel=0, FR_Wheel=1),
    )
    scene.env = types.SimpleNamespace(get_global_agent_state=lambda: {
        name: np.zeros(3, dtype=np.float32) for name in ("x", "y", "z", "heading")})
    scene._actor_ids = [10, 11, 12]
    scene.human_agent_idx = 1
    scene.ego = types.SimpleNamespace(id=99, set_wheel_steer_direction=lambda wheel, deg: None)
    scene._box_centers = {10: (0.0, 0.0), 12: (0.0, 0.0), 99: (0.0, 0.0)}
    scene.client = RecordingClient(log)
    scene.world = RecordingWorld(log)
    scene._sent_steer_deg = math.inf
    scene._pending_frame_id = None
    scene._scene_time_s = 1.0
    laid = []
    scene._lay_skid_marks = lambda ego_transform: laid.append(ego_transform)
    spun = []
    scene._actor_transform = lambda key, *args: types.SimpleNamespace(location=None, key=key)
    scene._spin_wheels = lambda actor_id, key, dt_s: spun.append((actor_id, key))
    scene._traffic_car_light_command = lambda *args: []
    scene._ego_light_command = lambda *args: []
    scene._mirror_traffic_lights = lambda: None
    scene._collect_images = lambda frame_id: frame_id
    scene.update({"steering_angle_rad": 0.0}, None, SMOOTH_FRAME_DT_S)
    assert log == [("apply_batch_sync", 4, False), ("tick",)]
    assert spun == [(10, 0), (12, 2), (99, ("ego",))]
    assert scene._scene_time_s == pytest.approx(1.0 + SMOOTH_FRAME_DT_S)
    assert [transform.key for transform in laid] == [("ego",)]



# ---------------------------------------------------------------------------------------------
# Skid marks: laid behind each tyre while the rendered car slides, kept where the depth camera sees them.
# ---------------------------------------------------------------------------------------------

EGO_ID = 99
EGO_HALF_LENGTH_M = 2.3
EGO_HALF_WIDTH_M = 1.0
CAMERA_HEIGHT_M = 2.0
SLIDING_SIDE_MPS = carla_scene.SKID_MARK_SIDE_SPEED_MPS + 1.0
CRUISE_FORWARD_MPS = 10.0
DEPTH_SIZE_PX = (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX)
OCCLUDER_DEPTH_M = 5.0


class FakeTransform:
    """A CARLA transform with yaw only, which is all the tyre points need."""

    def __init__(self, x, y, z, yaw_deg):
        self.location = FakeLocation(x, y, z)
        self.yaw_rad = math.radians(yaw_deg)

    def transform(self, location):
        cos_yaw, sin_yaw = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return FakeLocation(self.location.x + cos_yaw * location.x - sin_yaw * location.y,
                            self.location.y + sin_yaw * location.x + cos_yaw * location.y,
                            self.location.z + location.z)


def skid_scene(forward_mps=CRUISE_FORWARD_MPS, side_mps=SLIDING_SIDE_MPS):
    scene = CarlaScene.__new__(CarlaScene)
    scene.carla = types.SimpleNamespace(Location=FakeLocation)
    scene.ego = types.SimpleNamespace(id=EGO_ID)
    scene.ego_half_length_m = EGO_HALF_LENGTH_M
    scene.ego_half_width_m = EGO_HALF_WIDTH_M
    scene._box_centers = {EGO_ID: (0.0, 0.0)}
    scene._ground = {}
    scene._skid_marks = deque(maxlen=carla_scene.SKID_MARK_MAX_SEGMENTS)
    scene._skid_mark_points = None
    scene._scene_time_s = 0.0
    scene._depth_camera = None
    scene._main_rig = None
    set_ego_pose(scene, 0.0, forward_mps, side_mps)
    return scene


def set_ego_pose(scene, yaw_deg, forward_mps, side_mps):
    yaw_rad = math.radians(yaw_deg)
    vx = forward_mps * math.cos(yaw_rad) - side_mps * math.sin(yaw_rad)
    vy = forward_mps * math.sin(yaw_rad) + side_mps * math.cos(yaw_rad)
    scene._poses = {("ego",): carla_scene.SmoothedPose(0.0, 0.0, yaw_deg, vx, vy, 0.0)}


def expected_tyre_points(transform):
    axle_m = carla_scene.SKID_MARK_AXLE_FRACTION * EGO_HALF_LENGTH_M
    track_m = carla_scene.SKID_MARK_TRACK_FRACTION * EGO_HALF_WIDTH_M
    height_m = carla_scene.SKID_MARK_ROAD_CLEARANCE_M - carla_scene.GROUND_CLEARANCE_M
    points = []
    for along_m in (axle_m, -axle_m):
        for across_m in (-track_m, track_m):
            world = transform.transform(FakeLocation(along_m, across_m, height_m))
            points.append((world.x, world.y, world.z))
    return points


def assert_points_equal(actual, expected):
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-9)


def test_skid_mark_constants_are_the_contract():
    assert carla_scene.SKID_MARK_SIDE_SPEED_MPS == 1.0
    assert carla_scene.SKID_MARK_MIN_SPEED_MPS == 2.0
    assert carla_scene.SKID_MARK_LIFETIME_S == 20.0
    assert carla_scene.DEPTH_CAMERA_SCALE == 256.0**3 - 1.0


@pytest.mark.parametrize(
    "forward_mps,side_mps",
    [(CRUISE_FORWARD_MPS, 0.5 * carla_scene.SKID_MARK_SIDE_SPEED_MPS), (1.0, 1.5)],
    ids=["gripping", "too_slow"],
)
def test_no_marks_without_a_slide(forward_mps, side_mps):
    scene = skid_scene(forward_mps, side_mps)
    for step in range(3):
        scene._lay_skid_marks(FakeTransform(step * 0.2, 0.0, 0.0, 0.0))
    assert scene._skid_mark_points is None
    assert len(scene._skid_marks) == 0


def test_a_slide_lays_one_segment_per_tyre_from_the_previous_points():
    scene = skid_scene()
    first = FakeTransform(0.0, 0.0, 0.0, 10.0)
    scene._lay_skid_marks(first)
    assert_points_equal(scene._skid_mark_points, expected_tyre_points(first))
    assert len(scene._skid_marks) == 0
    scene._scene_time_s = 3.0
    second = FakeTransform(0.3, 0.1, 0.0, 12.0)
    scene._lay_skid_marks(second)
    marks = list(scene._skid_marks)
    assert len(marks) == 4
    assert_points_equal([mark[0] for mark in marks], expected_tyre_points(first))
    assert_points_equal([mark[1] for mark in marks], expected_tyre_points(second))
    assert [mark[2] for mark in marks] == [3.0] * 4


def test_stopping_the_slide_leaves_a_gap():
    scene = skid_scene()
    scene._lay_skid_marks(FakeTransform(0.0, 0.0, 0.0, 0.0))
    set_ego_pose(scene, 0.0, CRUISE_FORWARD_MPS, 0.0)
    scene._lay_skid_marks(FakeTransform(0.3, 0.0, 0.0, 0.0))
    assert scene._skid_mark_points is None
    set_ego_pose(scene, 0.0, CRUISE_FORWARD_MPS, SLIDING_SIDE_MPS)
    scene._lay_skid_marks(FakeTransform(0.6, 0.0, 0.0, 0.0))
    assert len(scene._skid_marks) == 0
    scene._lay_skid_marks(FakeTransform(0.9, 0.0, 0.0, 0.0))
    assert len(scene._skid_marks) == 4


def test_a_jump_is_not_a_slide():
    scene = skid_scene()
    scene._lay_skid_marks(FakeTransform(0.0, 0.0, 0.0, 0.0))
    scene._lay_skid_marks(FakeTransform(carla_scene.GROUND_SNAP_DISTANCE_M + 1.0, 0.0, 0.0, 0.0))
    assert len(scene._skid_marks) == 0
    assert scene._skid_mark_points is not None


def test_switching_cars_breaks_the_strip_but_keeps_the_marks():
    scene = skid_scene()
    scene._lay_skid_marks(FakeTransform(0.0, 0.0, 0.0, 0.0))
    scene._lay_skid_marks(FakeTransform(0.3, 0.0, 0.0, 0.0))
    scene.set_human_agent(4)
    assert scene._skid_mark_points is None
    assert len(scene._skid_marks) == 4


def test_the_depth_camera_starts_with_the_first_mark_and_only_once():
    scene = skid_scene()
    spawned = []
    scene._spawn_depth_camera = lambda: spawned.append(True) or setattr(scene, "_depth_camera", object())
    scene._lay_skid_marks(FakeTransform(0.0, 0.0, 0.0, 0.0))
    scene._lay_skid_marks(FakeTransform(0.3, 0.0, 0.0, 0.0))
    assert spawned == []
    scene._main_rig = main_camera_rig(CAMERA_MODE_CHASE)
    scene._lay_skid_marks(FakeTransform(0.6, 0.0, 0.0, 0.0))
    scene._lay_skid_marks(FakeTransform(0.9, 0.0, 0.0, 0.0))
    assert spawned == [True]


@pytest.mark.parametrize("with_marks", [True, False])
def test_a_camera_change_respawns_the_depth_camera_only_when_there_are_marks(with_marks):
    scene = skid_scene()
    scene._camera_mode = CAMERA_MODE_CHASE
    scene._mirrors = {}
    spawned = []
    scene._destroy_cameras = lambda: None
    scene._spawn_camera = lambda rig, size_px, name, tick_s, blueprint="sensor.camera.rgb": spawned.append(name)
    scene._spawn_depth_camera = lambda: spawned.append("depth")
    scene.image_size_px = DEPTH_SIZE_PX
    if with_marks:
        scene._skid_marks.append(((1.0, 0.0, 0.0), (1.5, 0.0, 0.0), 0.0))
    scene.set_camera_mode(CAMERA_MODE_AGENT)
    assert spawned == (["main", "depth"] if with_marks else ["main"])
    assert scene._main_rig == main_camera_rig(CAMERA_MODE_AGENT)


def forward_camera():
    return yawed_camera((0.0, 0.0, CAMERA_HEIGHT_M), 0.0)


def depth_image(depth_m, near_columns=0, near_depth_m=OCCLUDER_DEPTH_M):
    """A CARLA depth frame: BGRA, depth / far packed into R + 256 G + 65536 B; the first columns can be nearer."""
    def packed(meters):
        counts = round(meters / carla_scene.DEPTH_CAMERA_FAR_M * carla_scene.DEPTH_CAMERA_SCALE)
        return [(counts >> 16) & 0xFF, (counts >> 8) & 0xFF, counts & 0xFF, 0xFF]

    width_px, height_px = DEPTH_SIZE_PX
    bgra = np.empty((height_px, width_px, 4), dtype=np.uint8)
    bgra[:, :] = packed(depth_m)
    bgra[:, :near_columns] = packed(near_depth_m)
    return types.SimpleNamespace(raw_data=bgra.tobytes(), width=width_px, height=height_px)


def mark_at(x_m, y_m, length_m=0.5, laid_s=0.0):
    return ((x_m, y_m, 0.0), (x_m + length_m, y_m, 0.0), laid_s)


def marks_in(scene, depth):
    return scene.skid_marks(fake_image(forward_camera()), depth)


def test_marks_project_like_project_to_image_with_a_depth_scaled_width():
    scene = skid_scene()
    scene._scene_time_s = 5.0
    start, end, laid_s = mark_at(10.0, 1.0, laid_s=1.0)
    scene._skid_marks.append((start, end, laid_s))
    marks = marks_in(scene, depth_image(carla_scene.DEPTH_CAMERA_FAR_M))
    assert len(marks) == 1
    start_px, end_px, width_px, age_fraction = marks[0]
    assert start_px == pytest.approx(project(start, forward_camera()))
    assert end_px == pytest.approx(project(end, forward_camera()))
    mean_depth_m = 0.5 * (start[0] + end[0])
    assert width_px == max(1, round(focal_px(IMAGE_FOV_DEG) * carla_scene.SKID_MARK_WIDTH_M / mean_depth_m))
    assert age_fraction == pytest.approx((5.0 - 1.0) / carla_scene.SKID_MARK_LIFETIME_S)


def test_no_depth_image_means_no_marks_yet():
    scene = skid_scene()
    scene._skid_marks.append(mark_at(10.0, 0.0))
    assert marks_in(scene, None) == []
    assert len(scene._skid_marks) == 1


def test_no_marks_means_an_empty_list():
    assert marks_in(skid_scene(), depth_image(carla_scene.DEPTH_CAMERA_FAR_M)) == []


def test_distant_marks_are_at_least_a_pixel_wide():
    scene = skid_scene()
    scene._skid_marks.append(mark_at(900.0, 0.0))
    assert marks_in(scene, depth_image(carla_scene.DEPTH_CAMERA_FAR_M))[0][2] == 1


def test_expired_marks_are_dropped_oldest_first():
    scene = skid_scene()
    scene._skid_marks.extend([mark_at(10.0, 0.0, laid_s=0.0), mark_at(11.0, 0.0, laid_s=5.0)])
    scene._scene_time_s = carla_scene.SKID_MARK_LIFETIME_S + 1.0
    marks = marks_in(scene, depth_image(carla_scene.DEPTH_CAMERA_FAR_M))
    assert len(marks) == 1
    assert [mark[2] for mark in scene._skid_marks] == [5.0]


def test_marks_behind_the_camera_or_off_screen_are_left_out():
    scene = skid_scene()
    scene._skid_marks.extend([mark_at(-10.0, 0.0), mark_at(-0.2, 0.0), mark_at(10.0, 0.0), mark_at(3.0, 40.0)])
    assert len(marks_in(scene, depth_image(carla_scene.DEPTH_CAMERA_FAR_M))) == 1


def mark_depth_m(x_m, length_m=0.5):
    return x_m + 0.5 * length_m


def tolerance_m(depth_m):
    return carla_scene.SKID_MARK_DEPTH_TOLERANCE_M + carla_scene.SKID_MARK_DEPTH_TOLERANCE_FRACTION * depth_m


@pytest.mark.parametrize("margin_fraction,visible", [(-0.5, True), (0.5, False), (5.0, False)])
def test_a_nearer_surface_hides_a_mark_beyond_the_tolerance(margin_fraction, visible):
    scene = skid_scene()
    x_m = 20.0
    scene._skid_marks.append(mark_at(x_m, 0.0))
    depth_m = mark_depth_m(x_m)
    scene_depth_m = depth_m - tolerance_m(depth_m) - margin_fraction * tolerance_m(depth_m)
    assert (len(marks_in(scene, depth_image(scene_depth_m))) == 1) == visible


def test_the_road_under_a_mark_does_not_hide_it():
    scene = skid_scene()
    scene._skid_marks.append(mark_at(20.0, 0.0))
    assert len(marks_in(scene, depth_image(mark_depth_m(20.0)))) == 1


def test_depth_is_read_at_each_marks_own_pixel():
    scene = skid_scene()
    scene._skid_marks.extend([mark_at(20.0, -10.0), mark_at(20.0, 10.0)])
    marks = marks_in(scene, depth_image(carla_scene.DEPTH_CAMERA_FAR_M, near_columns=IMAGE_WIDTH_PX // 2))
    assert len(marks) == 1
    assert marks[0][0][0] > IMAGE_WIDTH_PX / 2.0


def frame_queue(*frame_ids):
    frames = queue.Queue()
    for frame_id in frame_ids:
        frames.put(types.SimpleNamespace(frame=frame_id))
    return frames


def depth_scene(*depth_frame_ids):
    scene = CarlaScene.__new__(CarlaScene)
    scene._frames = {"main": frame_queue(), "depth": frame_queue(*depth_frame_ids)}
    scene._next_depth_image = None
    scene._depth_camera = object()
    scene._mirrors = {}
    return scene


def test_depth_for_frame_returns_exactly_that_frame_and_keeps_a_later_one():
    scene = depth_scene(3, 4, 6)
    assert scene._depth_for_frame(4).frame == 4
    assert scene._depth_for_frame(5) is None
    assert scene._next_depth_image.frame == 6
    assert scene._depth_for_frame(6).frame == 6


def test_collect_images_adds_depth_only_with_a_depth_camera():
    scene = depth_scene(7)
    scene._frames["main"] = frame_queue(7)
    images = scene._collect_images(7)
    assert (images["main"].frame, images["depth"].frame) == (7, 7)
    scene._depth_camera = None
    scene._frames["main"] = frame_queue(8)
    assert set(scene._collect_images(8)) == {"main"}


class FakeCameraActor:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def stop(self):
        self.log.append(("stop", self.name))

    def destroy(self):
        self.log.append(("destroy", self.name))


def test_destroying_the_cameras_includes_the_depth_camera():
    log = []
    scene = CarlaScene.__new__(CarlaScene)
    scene._camera = FakeCameraActor(log, "main")
    scene._depth_camera = FakeCameraActor(log, "depth")
    scene._mirrors = {"rear": FakeCameraActor(log, "rear")}
    scene._next_depth_image = types.SimpleNamespace(frame=1)
    scene._destroy_cameras()
    assert ("destroy", "depth") in log and ("destroy", "main") in log and ("destroy", "rear") in log
    assert scene._depth_camera is None
    assert scene._next_depth_image is None


class FakeBlueprint:
    def __init__(self, name):
        self.name = name
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


@pytest.mark.parametrize("night", [False, True])
def test_the_depth_camera_is_the_main_view_without_exposure_settings(night):
    scene = CarlaScene.__new__(CarlaScene)
    blueprints = []
    spawned = []

    def find(name):
        blueprints.append(FakeBlueprint(name))
        return blueprints[-1]

    scene.carla = types.SimpleNamespace(Transform=lambda location, rotation: (location, rotation),
                                        Location=lambda **kwargs: kwargs, Rotation=lambda **kwargs: kwargs)
    scene.world = types.SimpleNamespace(
        get_blueprint_library=lambda: types.SimpleNamespace(find=find),
        spawn_actor=lambda bp, transform, attach_to: spawned.append((bp, transform, attach_to))
        or types.SimpleNamespace(listen=lambda callback: None),
    )
    scene.ego = object()
    scene.night = night
    scene._frames = {}
    scene._main_rig = main_camera_rig(CAMERA_MODE_CHASE)
    scene.image_size_px = (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX)
    scene._spawn_depth_camera()
    blueprint, _transform, attach_to = spawned[0]
    assert blueprint.name == "sensor.camera.depth"
    assert attach_to is scene.ego
    assert blueprint.attributes["image_size_x"] == str(IMAGE_WIDTH_PX)
    assert blueprint.attributes["image_size_y"] == str(IMAGE_HEIGHT_PX)
    assert blueprint.attributes["fov"] == str(scene._main_rig.fov_deg)
    assert not any(key.startswith("exposure") for key in blueprint.attributes)
    assert "depth" in scene._frames
    assert scene._depth_camera is not None
