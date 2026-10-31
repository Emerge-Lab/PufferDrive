"""The CARLA renderer's pure parts: camera rigs, steering sign, easing, blueprints, ground pitch.

Nothing here connects to a CARLA server. The one live test is opt-in via WHEEL_DEMO_CARLA_TESTS=1.
"""

import math
import os
import types

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
    scene.set_human_agent(3)
    assert scene.human_agent_idx == 3
    assert ("ego",) not in scene._ground


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
    fake_carla = types.SimpleNamespace(Client=FakeCarlaClient)
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
