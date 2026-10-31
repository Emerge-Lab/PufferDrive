"""The window HUD's pure maths, asserted in DRIVER terms, plus a headless pygame draw."""

import math
import os
import re

import numpy as np
import pytest

from pufferlib.ocean.cosim import carla_bridge
from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo import hud
from pufferlib.wheel_demo.hud import (
    HudInfo,
    MinimapProjection,
    clockwise_point,
    gear_label,
    goal_bearing_driver_rad,
    pedal_fraction,
    ring_segment,
    speed_excess_mps,
    speed_limit_readout,
    wheel_display_deg,
)

TOWN = "Town01"
CAR_HEADINGS_RAD = (0.0, 0.7, math.pi / 2.0, math.pi, -2.4, 3.1)
CAR_POSITIONS = ((0.0, 0.0), (120.5, -40.25), (-3000.0, 2500.0))
GOAL_DISTANCE_M = 35.0
QUARTER_TURN_RAD = math.pi / 2.0
SIM_LEFT_OFFSET_RAD = QUARTER_TURN_RAD
SIM_RIGHT_OFFSET_RAD = -QUARTER_TURN_RAD
ANGLE_TOLERANCE_RAD = 1e-9
MPS_TO_KMH = 3.6
MINIMAP_SIZE_PX = 250
PIXEL_TOLERANCE = 1e-6


def goal_at(x, y, heading_rad, sim_offset_rad, distance_m=GOAL_DISTANCE_M):
    """A goal `sim_offset_rad` CCW from the car's nose (sim frame: positive = LEFT)."""
    angle = heading_rad + sim_offset_rad
    return x + distance_m * math.cos(angle), y + distance_m * math.sin(angle)


def all_poses():
    return [(x, y, heading) for x, y in CAR_POSITIONS for heading in CAR_HEADINGS_RAD]


# ---------------------------------------------------------------------------------------------
# goal_bearing_driver_rad
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("x,y,heading", all_poses())
def test_goal_on_the_drivers_right_is_positive_quarter_turn(x, y, heading):
    goal_x, goal_y = goal_at(x, y, heading, SIM_RIGHT_OFFSET_RAD)
    assert goal_bearing_driver_rad(x, y, heading, goal_x, goal_y) == pytest.approx(QUARTER_TURN_RAD, abs=1e-6)


@pytest.mark.parametrize("x,y,heading", all_poses())
def test_goal_on_the_drivers_left_is_negative_quarter_turn(x, y, heading):
    goal_x, goal_y = goal_at(x, y, heading, SIM_LEFT_OFFSET_RAD)
    assert goal_bearing_driver_rad(x, y, heading, goal_x, goal_y) == pytest.approx(-QUARTER_TURN_RAD, abs=1e-6)


@pytest.mark.parametrize("x,y,heading", all_poses())
def test_goal_dead_ahead_is_zero(x, y, heading):
    goal_x, goal_y = goal_at(x, y, heading, 0.0)
    assert goal_bearing_driver_rad(x, y, heading, goal_x, goal_y) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("x,y,heading", all_poses())
def test_goal_behind_is_half_a_turn(x, y, heading):
    goal_x, goal_y = goal_at(x, y, heading, math.pi)
    assert abs(goal_bearing_driver_rad(x, y, heading, goal_x, goal_y)) == pytest.approx(math.pi, abs=1e-6)


@pytest.mark.parametrize("x,y,heading", all_poses())
@pytest.mark.parametrize("sim_offset_deg", [-170, -120, -60, -10, 10, 60, 120, 170])
def test_bearing_sign_follows_the_cross_product_and_stays_wrapped(x, y, heading, sim_offset_deg):
    goal_x, goal_y = goal_at(x, y, heading, math.radians(sim_offset_deg))
    bearing = goal_bearing_driver_rad(x, y, heading, goal_x, goal_y)
    cross = math.cos(heading) * (goal_y - y) - math.sin(heading) * (goal_x - x)
    goal_is_on_the_right = cross < 0.0
    assert (bearing > 0.0) == goal_is_on_the_right
    assert -math.pi <= bearing <= math.pi
    assert bearing == pytest.approx(-math.radians(sim_offset_deg), abs=1e-6)


# ---------------------------------------------------------------------------------------------
# gear_label
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("gearbox_mode", [binding.GEARBOX_MODE_AUTOMATIC, binding.GEARBOX_MODE_MANUAL])
def test_reverse_and_neutral_read_the_same_in_both_gearboxes(gearbox_mode):
    assert gear_label(binding.GEAR_REVERSE, gearbox_mode) == "R"
    assert gear_label(binding.GEAR_NEUTRAL, gearbox_mode) == "N"


@pytest.mark.parametrize("gear", range(1, binding.GEARBOX_FORWARD_GEAR_COUNT + 1))
def test_forward_gears_read_drive_n_in_automatic_and_n_in_manual(gear):
    assert gear_label(gear, binding.GEARBOX_MODE_AUTOMATIC) == f"D{gear}"
    assert gear_label(gear, binding.GEARBOX_MODE_MANUAL) == str(gear)


# ---------------------------------------------------------------------------------------------
# speed_limit_readout
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "limit_mps,expected_kmh",
    [(8.333333, 30), (13.4112, 48), (13.8889, 50), (15.0, 54), (22.352, 80), (27.7778, 100)],
)
def test_sign_shows_the_rounded_limit_in_kmh(limit_mps, expected_kmh):
    readout = speed_limit_readout(0.0, limit_mps, 1.0)
    assert readout.limit_kmh == expected_kmh
    assert isinstance(readout.limit_kmh, int)


def sim_overspeed_flag(speed_mps, limit_mps, tolerance_mps):
    """drive.h's SPEED_LIMIT_IDX test, in the float32 arithmetic the sim does it in."""
    sim_speed = np.float32(abs(speed_mps))
    return bool(sim_speed > np.float32(limit_mps) + np.float32(tolerance_mps))


@pytest.mark.parametrize("tolerance_mps", [0.0, 1.0, 2.5])
def test_warning_fires_on_the_sims_test_including_the_tolerance(tolerance_mps):
    limit_mps = 12.0
    assert not speed_limit_readout(limit_mps + tolerance_mps - 0.01, limit_mps, tolerance_mps).over_limit
    assert not speed_limit_readout(limit_mps + tolerance_mps, limit_mps, tolerance_mps).over_limit
    assert speed_limit_readout(limit_mps + tolerance_mps + 0.01, limit_mps, tolerance_mps).over_limit
    assert speed_limit_readout(-(limit_mps + tolerance_mps + 0.01), limit_mps, tolerance_mps).over_limit


def test_speed_between_limit_and_tolerance_is_not_over():
    readout = speed_limit_readout(13.0, 12.0, 2.0)
    assert not readout.over_limit
    assert readout.limit_kmh == round(12.0 * MPS_TO_KMH)


def test_warning_agrees_with_the_sim_on_random_float32_states():
    rng = np.random.default_rng(1234)
    limits = rng.uniform(5.0, 35.0, 4000).astype(np.float32)
    tolerances = rng.uniform(0.0, 3.0, 4000).astype(np.float32)
    speeds = (limits + tolerances + rng.uniform(-2.0, 2.0, 4000)).astype(np.float32)
    for speed, limit, tolerance in zip(speeds, limits, tolerances):
        expected = sim_overspeed_flag(float(speed), float(limit), float(tolerance))
        assert speed_limit_readout(float(speed), float(limit), float(tolerance)).over_limit == expected


def test_warning_agrees_with_the_sim_at_the_float32_rounding_boundary():
    limit_mps, tolerance_mps = float(np.float32(15.0)), float(np.float32(0.1))
    boundary_speed = float(np.float32(limit_mps) + np.float32(tolerance_mps))
    assert boundary_speed > limit_mps + tolerance_mps
    expected = sim_overspeed_flag(boundary_speed, limit_mps, tolerance_mps)
    assert speed_limit_readout(boundary_speed, limit_mps, tolerance_mps).over_limit == expected


@pytest.mark.parametrize("speed", [13.0, -13.0, 20.0, -20.0, 0.0])
def test_speed_excess_is_magnitude_minus_the_float32_threshold(speed):
    limit_mps, tolerance_mps = 13.9, 1.5
    threshold = float(np.float32(limit_mps) + np.float32(tolerance_mps))
    assert speed_excess_mps(speed, limit_mps, tolerance_mps) == abs(speed) - threshold


def test_speed_excess_is_positive_only_above_the_threshold():
    limit_mps, tolerance_mps = 10.0, 2.0
    assert speed_excess_mps(12.0, limit_mps, tolerance_mps) == 0.0
    assert speed_excess_mps(12.5, limit_mps, tolerance_mps) > 0.0
    assert speed_excess_mps(-12.5, limit_mps, tolerance_mps) > 0.0
    assert speed_excess_mps(11.0, limit_mps, tolerance_mps) < 0.0


# ---------------------------------------------------------------------------------------------
# pedal_fraction
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [(-0.5, 0.0), (0.0, 0.0), (0.37, 0.37), (1.0, 1.0), (1.8, 1.0)])
def test_pedal_fraction_clamps_to_the_unit_interval(value, expected):
    assert pedal_fraction(value) == expected


def test_pedal_fraction_accepts_numpy_scalars():
    assert pedal_fraction(np.float32(0.5)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------
# MinimapProjection
# ---------------------------------------------------------------------------------------------

WIDE_TOWN_POINTS = np.array([[-400.0, -50.0], [600.0, 30.0], [100.0, 0.0]])


def test_minimap_is_north_up_and_east_right():
    projection = MinimapProjection(WIDE_TOWN_POINTS)
    centre_px = projection.to_px(100.0, -10.0)
    north_px = projection.to_px(100.0, 10.0)
    east_px = projection.to_px(120.0, -10.0)
    assert north_px[1] < centre_px[1]
    assert north_px[0] == pytest.approx(centre_px[0])
    assert east_px[0] > centre_px[0]
    assert east_px[1] == pytest.approx(centre_px[1])


def test_minimap_keeps_one_scale_on_both_axes():
    projection = MinimapProjection(WIDE_TOWN_POINTS)
    origin = projection.to_px(0.0, 0.0)
    east = projection.to_px(10.0, 0.0)
    north = projection.to_px(0.0, 10.0)
    assert east[0] - origin[0] == pytest.approx(origin[1] - north[1])


@pytest.mark.parametrize("size_px", [MINIMAP_SIZE_PX, 100, 512])
def test_every_point_fits_and_the_long_axis_spans_the_map(size_px):
    margin_m = 20.0
    projection = MinimapProjection(WIDE_TOWN_POINTS, size_px=size_px, margin_m=margin_m)
    assert projection.size_px == size_px
    for x, y in WIDE_TOWN_POINTS:
        px, py = projection.to_px(x, y)
        assert 0.0 <= px <= size_px and 0.0 <= py <= size_px
    west_px, _ = projection.to_px(-400.0 - margin_m, 0.0)
    east_px, _ = projection.to_px(600.0 + margin_m, 0.0)
    assert west_px == pytest.approx(0.0, abs=PIXEL_TOLERANCE)
    assert east_px == pytest.approx(size_px, abs=PIXEL_TOLERANCE)


def test_short_axis_is_centred():
    projection = MinimapProjection(WIDE_TOWN_POINTS)
    _, south_py = projection.to_px(0.0, -50.0)
    _, north_py = projection.to_px(0.0, 30.0)
    assert (south_py + north_py) / 2.0 == pytest.approx(projection.size_px / 2.0)


def test_minimap_rejects_an_empty_town():
    with pytest.raises(ValueError):
        MinimapProjection(np.zeros((0, 2)))


def test_the_whole_real_town_fits_on_the_minimap():
    points = np.vstack(hud.bin_lane_polylines(carla_bridge.bin_path_for_town(TOWN)))
    projection = MinimapProjection(points)
    px = projection.size_px / 2.0 + (points[:, 0] - projection.center[0]) * projection.scale_px_per_m
    py = projection.size_px / 2.0 - (points[:, 1] - projection.center[1]) * projection.scale_px_per_m
    assert px.min() >= 0.0 and px.max() <= projection.size_px
    assert py.min() >= 0.0 and py.max() <= projection.size_px


# ---------------------------------------------------------------------------------------------
# Hud drawing, headless.
# ---------------------------------------------------------------------------------------------

SCREEN_SIZE_PX = (1280, 720)
ARROW_CENTRE_PX = (hud.PANEL_MARGIN_PX + hud.GOAL_ARROW_RADIUS_PX, hud.PANEL_MARGIN_PX + hud.GOAL_ARROW_RADIUS_PX)


class WheelStateStub:
    def __init__(self, throttle_norm=0.4, brake_norm=0.1):
        self.throttle_norm = throttle_norm
        self.brake_norm = brake_norm


@pytest.fixture(scope="module")
def pygame_headless():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    pygame.init()
    yield pygame
    pygame.quit()


@pytest.fixture(scope="module")
def town_hud(pygame_headless):
    return hud.Hud(pygame_headless, carla_bridge.bin_path_for_town(TOWN))


def hud_human(heading_rad=0.0, sim_goal_offset_rad=0.0, **overrides):
    x, y = 150.0, 60.0
    goal_x, goal_y = goal_at(x, y, heading_rad, sim_goal_offset_rad)
    human = {
        "sim_x": x,
        "sim_y": y,
        "sim_heading": heading_rad,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "goal_idx": 0,
        "speed_mps": 9.0,
        "lane_speed_limit_mps": 13.9,
        "overspeed_tolerance_mps": 1.0,
        "gear": 2,
        "gearbox_mode": binding.GEARBOX_MODE_AUTOMATIC,
        "collision": 0.0,
        "offroad": 0.0,
        "engine_stalled": 0,
    }
    human.update(overrides)
    return human


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"speed_mps": 30.0},
        {"speed_mps": -4.0, "gear": binding.GEAR_REVERSE},
        {"collision": 1.0, "offroad": 1.0, "engine_stalled": 1},
        {"gearbox_mode": binding.GEARBOX_MODE_MANUAL, "gear": binding.GEAR_NEUTRAL},
    ],
)
def test_drawing_a_human_does_not_raise(pygame_headless, town_hud, overrides):
    surface = pygame_headless.Surface(SCREEN_SIZE_PX)
    for elapsed_s in (0.0, 0.3):
        town_hud.draw(surface, hud_human(**overrides), WheelStateStub(), HudInfo(help_text="status"), elapsed_s)


def test_drawing_with_mirrors_does_not_raise(pygame_headless, town_hud):
    surface = pygame_headless.Surface(SCREEN_SIZE_PX)
    mirror = pygame_headless.Surface((200, 100))
    rect = pygame_headless.Rect(10, 300, 300, 190)
    town_hud.draw(surface, hud_human(), WheelStateStub(), HudInfo(), 0.0, mirrors=[(mirror, rect)])


def arrow_pixel_centroid(pygame, town_hud, human):
    surface = pygame.Surface(SCREEN_SIZE_PX)
    surface.fill((0, 0, 0))
    town_hud._draw_goal_arrow(surface, human)
    pixels = pygame.surfarray.array3d(surface)
    radius = hud.GOAL_ARROW_RADIUS_PX
    cx, cy = ARROW_CENTRE_PX
    window = pixels[cx - radius : cx + radius, cy - radius : cy + radius]
    matches = np.argwhere(np.all(window == np.array(hud.COLOR_ARROW), axis=-1))
    assert len(matches) > 0
    return matches[:, 0].mean() + cx - radius, matches[:, 1].mean() + cy - radius


@pytest.mark.parametrize("heading", CAR_HEADINGS_RAD)
def test_arrow_points_right_for_a_goal_on_the_drivers_right(pygame_headless, town_hud, heading):
    x, _ = arrow_pixel_centroid(pygame_headless, town_hud, hud_human(heading, SIM_RIGHT_OFFSET_RAD))
    assert x > ARROW_CENTRE_PX[0] + 3


@pytest.mark.parametrize("heading", CAR_HEADINGS_RAD)
def test_arrow_points_left_for_a_goal_on_the_drivers_left(pygame_headless, town_hud, heading):
    x, _ = arrow_pixel_centroid(pygame_headless, town_hud, hud_human(heading, SIM_LEFT_OFFSET_RAD))
    assert x < ARROW_CENTRE_PX[0] - 3


@pytest.mark.parametrize("heading", CAR_HEADINGS_RAD)
def test_arrow_points_up_for_a_goal_ahead_and_down_for_one_behind(pygame_headless, town_hud, heading):
    _, ahead_y = arrow_pixel_centroid(pygame_headless, town_hud, hud_human(heading, 0.0))
    _, behind_y = arrow_pixel_centroid(pygame_headless, town_hud, hud_human(heading, math.pi))
    assert ahead_y < ARROW_CENTRE_PX[1] - 3
    assert behind_y > ARROW_CENTRE_PX[1] + 3


# ---------------------------------------------------------------------------------------------
# Steering wheel and assist dial, in DRIVER terms: clockwise = right.
# ---------------------------------------------------------------------------------------------

FULL_LOCK_WHEEL_DEG = binding.HUMAN_WHEEL_RANGE_DEG / 2.0
CAMERA_MODE_CHASE = 0
CAMERA_MODE_DRIVER = 1
# Small enough that the arrow stays in the dial's upper half, below the readout text.
SMALL_ROAD_ANGLE_RAD = 0.04
CONTRACT_DRIVER_WHEEL_CENTER_X_FRACTION = 0.36


def wheel_centre_x(camera_mode, width):
    if camera_mode == CAMERA_MODE_DRIVER:
        return int(CONTRACT_DRIVER_WHEEL_CENTER_X_FRACTION * width)
    return width // 2


@pytest.mark.parametrize("road_fraction", [0.0, 0.1, 0.5, 1.0])
def test_wheel_display_turns_clockwise_for_a_right_turn(road_fraction):
    sim_right_rad = -road_fraction * binding.HUMAN_STEERING_LIMIT_RAD
    degrees = wheel_display_deg(sim_right_rad, binding.HUMAN_STEERING_RATIO, binding.HUMAN_STEERING_LIMIT_RAD)
    assert degrees == pytest.approx(road_fraction * FULL_LOCK_WHEEL_DEG, rel=1e-5, abs=1e-9)
    assert wheel_display_deg(-sim_right_rad, binding.HUMAN_STEERING_RATIO, binding.HUMAN_STEERING_LIMIT_RAD) == (
        pytest.approx(-degrees, abs=1e-9)
    )


def test_wheel_display_clamps_at_lock():
    beyond_rad = 3.0 * binding.HUMAN_STEERING_LIMIT_RAD
    ratio, limit = binding.HUMAN_STEERING_RATIO, binding.HUMAN_STEERING_LIMIT_RAD
    assert wheel_display_deg(-beyond_rad, ratio, limit) == pytest.approx(FULL_LOCK_WHEEL_DEG, rel=1e-5)
    assert wheel_display_deg(beyond_rad, ratio, limit) == pytest.approx(-FULL_LOCK_WHEEL_DEG, rel=1e-5)


@pytest.mark.parametrize(
    "angle_deg,expected", [(0.0, (100.0, 90.0)), (90.0, (110.0, 100.0)), (180.0, (100.0, 110.0)), (-90.0, (90.0, 100.0))]
)
def test_clockwise_point_starts_at_twelve_and_turns_clockwise_on_screen(angle_deg, expected):
    x, y = clockwise_point((100.0, 100.0), 10.0, angle_deg)
    assert (x, y) == pytest.approx(expected, abs=1e-9)


def hud_assist(mode, reference_valid=1, reference_sim_rad=0.0, human_sim_rad=0.0):
    return {
        "mode": mode,
        "blend_target": 0.5,
        "blend_weight": 0.0,
        "reference_valid": reference_valid,
        "reference_steering_rad": reference_sim_rad,
        "reference_accel_mps2": 0.4,
        "human_steering_rad": human_sim_rad,
        "commanded_steering_rad": human_sim_rad,
    }


def dial_policy_columns(pygame, town_hud, assist, camera_mode=CAMERA_MODE_CHASE):
    """Columns of policy-coloured pixels inside the dial, below the readout text."""
    surface = pygame.Surface(SCREEN_SIZE_PX)
    surface.fill((0, 0, 0))
    human = hud_human(accel_long_mps2=0.0, steering_angle_rad=0.0)
    town_hud.draw(surface, human, WheelStateStub(), HudInfo(camera_mode=camera_mode), 0.0, assist=assist)
    pixels = pygame.surfarray.array3d(surface)
    width, height = SCREEN_SIZE_PX
    radius = hud.DRIVER_WHEEL_RADIUS_PX if camera_mode == CAMERA_MODE_DRIVER else hud.DIAL_WHEEL_RADIUS_PX
    head = hud.ASSIST_ARROW_HEAD_FRACTION * radius
    cx, cy = wheel_centre_x(camera_mode, width), height - hud.WHEEL_BOTTOM_GAP_PX - radius
    reach = int(radius + head + 2)
    window = pixels[cx - reach : cx + reach, cy - int(radius + head) : cy + reach]
    columns = np.argwhere(np.all(window == np.array(hud.COLOR_ASSIST_POLICY), axis=-1))[:, 0]
    return columns + cx - reach, cx


@pytest.mark.parametrize("camera_mode", [CAMERA_MODE_CHASE, CAMERA_MODE_DRIVER])
@pytest.mark.parametrize(
    "mode", [binding.ASSIST_MODE_DISPLAY, binding.ASSIST_MODE_BLEND, binding.ASSIST_MODE_NUDGE, binding.ASSIST_MODE_FULL]
)
def test_policy_arrow_points_right_when_the_policy_steers_right(pygame_headless, town_hud, mode, camera_mode):
    assist = hud_assist(mode, reference_sim_rad=-SMALL_ROAD_ANGLE_RAD)
    columns, cx = dial_policy_columns(pygame_headless, town_hud, assist, camera_mode)
    assert len(columns) > 0
    assert columns.mean() > cx + 3


def test_policy_arrow_points_left_when_the_policy_steers_left(pygame_headless, town_hud):
    assist = hud_assist(binding.ASSIST_MODE_DISPLAY, reference_sim_rad=SMALL_ROAD_ANGLE_RAD)
    columns, cx = dial_policy_columns(pygame_headless, town_hud, assist)
    assert len(columns) > 0
    assert columns.mean() < cx - 3


@pytest.mark.parametrize(
    "assist",
    [None, hud_assist(binding.ASSIST_MODE_OFF, reference_sim_rad=-0.04), hud_assist(binding.ASSIST_MODE_DISPLAY, 0, -0.04)],
    ids=["no_assist", "off", "invalid_reference"],
)
@pytest.mark.parametrize("camera_mode", [CAMERA_MODE_CHASE, CAMERA_MODE_DRIVER])
def test_no_policy_arrow_without_a_valid_reference_or_with_the_assist_off(pygame_headless, town_hud, assist, camera_mode):
    columns, _ = dial_policy_columns(pygame_headless, town_hud, assist, camera_mode)
    assert len(columns) == 0


@pytest.mark.parametrize("ffb_gain", [None, 0.0, 0.7, 1.0])
@pytest.mark.parametrize("level_name", ["calm", "normal", "aggressive", "unlisted"])
def test_drawing_every_info_combination_does_not_raise(pygame_headless, town_hud, ffb_gain, level_name):
    surface = pygame_headless.Surface(SCREEN_SIZE_PX)
    info = HudInfo(camera_mode=CAMERA_MODE_DRIVER, level_name=level_name, ffb_gain=ffb_gain, help_text="keys")
    for mode in (binding.ASSIST_MODE_OFF, binding.ASSIST_MODE_BLEND):
        town_hud.draw(surface, hud_human(accel_long_mps2=0.3, steering_angle_rad=0.1), WheelStateStub(), info, 0.0,
                      assist=hud_assist(mode, reference_sim_rad=-0.1))


def test_full_has_a_hud_name():
    assert hud.ASSIST_MODE_NAMES == {0: "off", 1: "display", 2: "blend", 3: "nudge", 4: "full"}


# ---------------------------------------------------------------------------------------------
# ring_segment: an annulus polygon between clockwise-from-twelve angles.
# ---------------------------------------------------------------------------------------------

RING_CENTRE_PX = (50.0, 80.0)


def test_ring_segment_is_the_outer_arc_then_the_inner_arc_back():
    points = ring_segment((0.0, 0.0), 10.0, 5.0, 0.0, 90.0, step_deg=45.0)
    half = math.sqrt(0.5)
    expected = [(0.0, -10.0), (10.0 * half, -10.0 * half), (10.0, 0.0), (5.0, 0.0), (5.0 * half, -5.0 * half), (0.0, -5.0)]
    assert len(points) == len(expected)
    for point, want in zip(points, expected):
        assert point == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("start_deg,stop_deg", [(-4.0, 4.0), (62.0, 118.0), (242.0, 298.0), (0.0, 360.0), (30.0, 10.0)])
@pytest.mark.parametrize("step_deg", [0.5, 2.0, 7.0])
def test_ring_segment_points_lie_on_their_radii_within_the_span(start_deg, stop_deg, step_deg):
    outer_px, inner_px = 40.0, 31.0
    points = ring_segment(RING_CENTRE_PX, outer_px, inner_px, start_deg, stop_deg, step_deg)
    count = len(points) // 2
    assert len(points) == 2 * count and count >= 2
    assert count == max(2, int(abs(stop_deg - start_deg) / step_deg) + 1)
    for idx, (x, y) in enumerate(points):
        radius = math.hypot(x - RING_CENTRE_PX[0], y - RING_CENTRE_PX[1])
        assert radius == pytest.approx(outer_px if idx < count else inner_px, abs=1e-9)
    assert points[0] == pytest.approx(clockwise_point(RING_CENTRE_PX, outer_px, start_deg), abs=1e-9)
    assert points[count - 1] == pytest.approx(clockwise_point(RING_CENTRE_PX, outer_px, stop_deg), abs=1e-9)
    assert points[count] == pytest.approx(clockwise_point(RING_CENTRE_PX, inner_px, stop_deg), abs=1e-9)
    assert points[-1] == pytest.approx(clockwise_point(RING_CENTRE_PX, inner_px, start_deg), abs=1e-9)


def test_ring_segment_narrower_than_one_step_still_has_both_ends():
    points = ring_segment(RING_CENTRE_PX, 20.0, 10.0, 10.0, 11.0, step_deg=5.0)
    assert len(points) == 4
    assert points[0] == pytest.approx(clockwise_point(RING_CENTRE_PX, 20.0, 10.0), abs=1e-9)
    assert points[1] == pytest.approx(clockwise_point(RING_CENTRE_PX, 20.0, 11.0), abs=1e-9)


def test_ring_segment_around_twelve_is_mirror_symmetric():
    points = ring_segment(RING_CENTRE_PX, 30.0, 25.0, -4.0, 4.0)
    xs = sorted(x - RING_CENTRE_PX[0] for x, _ in points)
    assert xs == pytest.approx(sorted(-x for x in xs), abs=1e-9)
    assert all(y < RING_CENTRE_PX[1] for _, y in points)


# ---------------------------------------------------------------------------------------------
# The drawn wheel: the HANDS angle, where it sits, and which way it turns.
# ---------------------------------------------------------------------------------------------

WHEEL_MARKER_COLOR = (3, 251, 7)
READOUT_PATTERN = re.compile(r"wheel\s+you ([+-]\d+) deg\s+policy ([+-]\d+) deg\s+\(([+-]\d+)\)")


@pytest.fixture
def wheel_spy(pygame_headless, town_hud, monkeypatch):
    """Records each rotozoom angle and draws the wheel as a solid marker, so its placement is measurable."""
    calls = []
    texts = []

    def rotozoom(sprite, angle_deg, scale):
        calls.append((sprite, angle_deg, scale))
        marker = pygame_headless.Surface(sprite.get_size())
        marker.fill(WHEEL_MARKER_COLOR)
        return marker

    real_text = town_hud._text

    def text(surface, font, message, position, color=hud.COLOR_TEXT, anchor="topleft"):
        texts.append(message)
        return real_text(surface, font, message, position, color, anchor)

    monkeypatch.setattr(pygame_headless.transform, "rotozoom", rotozoom)
    monkeypatch.setattr(town_hud, "_text", text)
    return calls, texts


def draw_wheel_frame(pygame, town_hud, human, camera_mode, assist):
    surface = pygame.Surface(SCREEN_SIZE_PX)
    surface.fill((0, 0, 0))
    town_hud.draw(surface, human, WheelStateStub(), HudInfo(camera_mode=camera_mode), 0.0, assist=assist)
    pixels = pygame.surfarray.array3d(surface)
    return np.argwhere(np.all(pixels == np.array(WHEEL_MARKER_COLOR), axis=-1))


def wheel_degrees(sim_rad):
    return wheel_display_deg(sim_rad, binding.HUMAN_STEERING_RATIO, binding.HUMAN_STEERING_LIMIT_RAD)


def readout_numbers(texts):
    matches = [READOUT_PATTERN.search(text) for text in texts]
    found = [tuple(int(group) for group in match.groups()) for match in matches if match]
    assert len(found) == 1
    return found[0]


@pytest.mark.parametrize("hands_sim_rad,road_sim_rad", [(-0.1, 0.05), (0.08, -0.12), (0.0, 0.2)])
@pytest.mark.parametrize("camera_mode", [CAMERA_MODE_CHASE, CAMERA_MODE_DRIVER])
def test_under_blend_the_wheel_and_you_show_the_hands_not_the_road_wheels(
    pygame_headless, town_hud, wheel_spy, hands_sim_rad, road_sim_rad, camera_mode
):
    calls, texts = wheel_spy
    assist = hud_assist(binding.ASSIST_MODE_BLEND, reference_sim_rad=-SMALL_ROAD_ANGLE_RAD, human_sim_rad=hands_sim_rad)
    assist["commanded_steering_rad"] = road_sim_rad
    human = hud_human(accel_long_mps2=0.0, steering_angle_rad=road_sim_rad)
    draw_wheel_frame(pygame_headless, town_hud, human, camera_mode, assist)
    assert len(calls) == 1
    assert calls[0][1] == pytest.approx(-wheel_degrees(hands_sim_rad), abs=1e-9)
    you_deg, policy_deg, delta_deg = readout_numbers(texts)
    assert you_deg == round(wheel_degrees(hands_sim_rad))
    assert policy_deg == round(wheel_degrees(-SMALL_ROAD_ANGLE_RAD))
    assert abs((policy_deg - you_deg) - delta_deg) <= 1


def test_driver_view_with_the_assist_off_still_draws_the_hands(pygame_headless, town_hud, wheel_spy):
    calls, texts = wheel_spy
    assist = hud_assist(binding.ASSIST_MODE_OFF, reference_valid=0, human_sim_rad=-0.15)
    human = hud_human(accel_long_mps2=0.0, steering_angle_rad=0.3)
    draw_wheel_frame(pygame_headless, town_hud, human, CAMERA_MODE_DRIVER, assist)
    assert [angle for _, angle, _ in calls] == [pytest.approx(-wheel_degrees(-0.15), abs=1e-9)]
    assert not [text for text in texts if READOUT_PATTERN.search(text)]


def test_without_an_assist_the_driver_view_draws_the_sims_steering(pygame_headless, town_hud, wheel_spy):
    calls, _ = wheel_spy
    draw_wheel_frame(pygame_headless, town_hud, hud_human(steering_angle_rad=-0.07), CAMERA_MODE_DRIVER, None)
    assert [angle for _, angle, _ in calls] == [pytest.approx(-wheel_degrees(-0.07), abs=1e-9)]


@pytest.mark.parametrize("turn", [SIM_RIGHT_OFFSET_RAD, SIM_LEFT_OFFSET_RAD], ids=["right", "left"])
def test_turning_right_rotates_the_sprite_clockwise(pygame_headless, town_hud, wheel_spy, turn):
    calls, _ = wheel_spy
    sim_rad = 0.05 * math.copysign(1.0, turn)
    draw_wheel_frame(pygame_headless, town_hud, hud_human(steering_angle_rad=sim_rad), CAMERA_MODE_DRIVER, None)
    pygame_ccw_deg = calls[0][1]
    driver_clockwise = sim_rad < 0.0
    assert (pygame_ccw_deg < 0.0) == driver_clockwise
    assert pygame_ccw_deg != 0.0


def marker_box_centre(columns_rows):
    xs, ys = columns_rows[:, 0], columns_rows[:, 1]
    return (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0


def test_driver_view_puts_the_wheel_left_of_centre(pygame_headless, town_hud, wheel_spy):
    calls, _ = wheel_spy
    marker = draw_wheel_frame(pygame_headless, town_hud, hud_human(steering_angle_rad=0.0), CAMERA_MODE_DRIVER, None)
    width, height = SCREEN_SIZE_PX
    centre_x, centre_y = marker_box_centre(marker)
    radius = hud.DRIVER_WHEEL_RADIUS_PX
    assert calls[0][0].get_width() >= 2 * radius
    assert centre_x == pytest.approx(int(CONTRACT_DRIVER_WHEEL_CENTER_X_FRACTION * width), abs=1.0)
    assert centre_y == pytest.approx(height - hud.WHEEL_BOTTOM_GAP_PX - radius, abs=1.0)


@pytest.mark.parametrize(
    "assist,draws",
    [
        (None, False),
        (hud_assist(binding.ASSIST_MODE_OFF, reference_sim_rad=-0.04), False),
        (hud_assist(binding.ASSIST_MODE_FULL, reference_valid=0), False),
        (hud_assist(binding.ASSIST_MODE_DISPLAY), True),
        (hud_assist(binding.ASSIST_MODE_FULL), True),
    ],
    ids=["none", "off", "invalid", "display", "full"],
)
def test_other_views_draw_the_small_dial_centred_only_with_an_active_assist(
    pygame_headless, town_hud, wheel_spy, assist, draws
):
    calls, _ = wheel_spy
    human = hud_human(accel_long_mps2=0.0, steering_angle_rad=0.0)
    marker = draw_wheel_frame(pygame_headless, town_hud, human, CAMERA_MODE_CHASE, assist)
    assert len(calls) == int(draws)
    if not draws:
        return
    width, height = SCREEN_SIZE_PX
    centre_x, centre_y = marker_box_centre(marker)
    assert centre_x == pytest.approx(width // 2, abs=1.0)
    assert centre_y == pytest.approx(height - hud.WHEEL_BOTTOM_GAP_PX - hud.DIAL_WHEEL_RADIUS_PX, abs=1.0)


def test_wheel_sprite_is_cached_per_radius(town_hud):
    driver = town_hud._wheel_sprite(hud.DRIVER_WHEEL_RADIUS_PX)
    dial = town_hud._wheel_sprite(hud.DIAL_WHEEL_RADIUS_PX)
    assert town_hud._wheel_sprite(hud.DRIVER_WHEEL_RADIUS_PX) is driver
    assert dial is not driver
    assert dial.get_width() < driver.get_width()


def test_full_mode_is_named_on_the_hud(pygame_headless, town_hud, wheel_spy):
    _, texts = wheel_spy
    draw_wheel_frame(pygame_headless, town_hud, hud_human(accel_long_mps2=0.0, steering_angle_rad=0.0),
                     CAMERA_MODE_CHASE, hud_assist(binding.ASSIST_MODE_FULL))
    assert "Assist: full" in texts


# ---------------------------------------------------------------------------------------------
# Goal beacon.
# ---------------------------------------------------------------------------------------------

BEACON_BASE_PX = (700, 420)
BEACON_TOP_PX = (700, 300)


def beacon_pixels(pygame, town_hud, goal_marker):
    surface = pygame.Surface(SCREEN_SIZE_PX)
    surface.fill((0, 0, 0))
    town_hud.draw(surface, hud_human(), WheelStateStub(), HudInfo(), 0.0, goal_marker=goal_marker)
    pixels = pygame.surfarray.array3d(surface)
    found = np.argwhere(np.all(pixels == np.array(hud.COLOR_GOAL_BEACON), axis=-1))
    return found[np.abs(found[:, 0] - BEACON_BASE_PX[0]) <= hud.GOAL_BEACON_WIDTH_PX]


def test_goal_beacon_is_drawn_from_base_to_top(pygame_headless, town_hud):
    on_pole = beacon_pixels(pygame_headless, town_hud, (BEACON_BASE_PX, BEACON_TOP_PX))
    assert on_pole[:, 1].max() >= BEACON_BASE_PX[1] - 1
    assert on_pole[:, 1].min() <= BEACON_TOP_PX[1] - hud.GOAL_DIAMOND_PX + 1


def test_no_goal_beacon_without_a_marker(pygame_headless, town_hud):
    assert len(beacon_pixels(pygame_headless, town_hud, None)) == 0


# ---------------------------------------------------------------------------------------------
# Policy pedal targets on the THR/BRK bars and the FULL speed flag.
# ---------------------------------------------------------------------------------------------

POLICY_SPEED_FLAG = "POLICY DRIVING SPEED"
FORWARD_GEAR = 2
# rounds to float32(0.05), so the sim (float32 compare) treats it as pressed
ROUNDS_TO_THRESHOLD_NORM = 0.0500000003
THROTTLE_BAR_IDX = 0
BRAKE_BAR_IDX = 1


@pytest.mark.parametrize(
    "accel_mps2,expected",
    [(0.0, (0.0, 0.0)), (2.0, (0.5, 0.0)), (4.0, (1.0, 0.0)), (10.0, (1.0, 0.0)), (-4.5, (0.0, 0.5)),
     (-9.0, (0.0, 1.0)), (-20.0, (0.0, 1.0)), (-0.9, (0.0, 0.1))],
)
def test_policy_pedal_targets_split_accel_onto_throttle_and_brake(accel_mps2, expected):
    targets = hud.policy_pedal_targets(accel_mps2, binding.HUMAN_ACCEL_MIN_MPS2, binding.HUMAN_ACCEL_MAX_MPS2)
    assert targets == pytest.approx(expected, abs=1e-12)
    assert min(targets) == 0.0


def hud_assist_accel(mode, accel_mps2, reference_valid=1):
    assist = hud_assist(mode, reference_valid=reference_valid)
    assist["reference_accel_mps2"] = accel_mps2
    return assist


@pytest.mark.parametrize(
    "assist,throttle,brake,expected",
    [
        (None, 0.0, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.0, 0.0, True),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.0499, 0.0499, True),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.05, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.0, 0.05, False),
        (hud_assist(binding.ASSIST_MODE_FULL), ROUNDS_TO_THRESHOLD_NORM, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.0, ROUNDS_TO_THRESHOLD_NORM, False),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.0501, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.0, 0.0501, False),
        (hud_assist(binding.ASSIST_MODE_FULL), 1.0, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_FULL, reference_valid=0), 0.0, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_BLEND), 0.0, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_DISPLAY), 0.0, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_NUDGE), 0.0, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_OFF), 0.0, 0.0, False),
    ],
)
def test_policy_drives_speed_only_under_full_with_both_feet_off(assist, throttle, brake, expected):
    wheel_state = WheelStateStub(throttle_norm=throttle, brake_norm=brake)
    drives = hud.policy_drives_speed(
        assist, wheel_state, FORWARD_GEAR, binding.ASSIST_MODE_FULL, binding.ASSIST_PEDAL_RELEASED_FRACTION
    )
    assert bool(drives) == expected


@pytest.mark.parametrize(
    "gear,expected",
    [(binding.GEAR_REVERSE, False), (binding.GEAR_NEUTRAL, False), (1, True), (binding.GEARBOX_FORWARD_GEAR_COUNT, True)],
)
def test_policy_drives_speed_only_in_a_forward_gear(gear, expected):
    drives = hud.policy_drives_speed(hud_assist(binding.ASSIST_MODE_FULL), WheelStateStub(0.0, 0.0), gear,
                                     binding.ASSIST_MODE_FULL, binding.ASSIST_PEDAL_RELEASED_FRACTION)
    assert bool(drives) == expected


@pytest.mark.parametrize("pedal_norm", [0.0, 0.03, 0.0499999, 0.05, ROUNDS_TO_THRESHOLD_NORM, 0.0500001, 0.2])
def test_policy_drives_speed_agrees_with_the_sims_float32_threshold(pedal_norm):
    sim_released = np.float32(pedal_norm) < np.float32(binding.ASSIST_PEDAL_RELEASED_FRACTION)
    for throttle, brake in ((pedal_norm, 0.0), (0.0, pedal_norm)):
        drives = hud.policy_drives_speed(hud_assist(binding.ASSIST_MODE_FULL), WheelStateStub(throttle, brake),
                                         FORWARD_GEAR, binding.ASSIST_MODE_FULL, binding.ASSIST_PEDAL_RELEASED_FRACTION)
        assert bool(drives) == bool(sim_released)


def panel_pixels(pygame, town_hud, assist, wheel_state):
    surface = pygame.Surface(SCREEN_SIZE_PX)
    surface.fill((0, 0, 0))
    town_hud.draw(surface, hud_human(accel_long_mps2=0.0, steering_angle_rad=0.0), wheel_state, HudInfo(), 0.0,
                  assist=assist)
    return pygame.surfarray.array3d(surface)


def bar_tracks(pygame, town_hud):
    """(left, right, top, bottom) of each pedal track, read off a frame with both pedals up."""
    pixels = panel_pixels(pygame, town_hud, None, WheelStateStub(0.0, 0.0))
    found = np.argwhere(np.all(pixels == np.array(hud.COLOR_BAR_TRACK), axis=-1))
    found = found[found[:, 1] > SCREEN_SIZE_PX[1] // 2]
    split_x = (found[:, 0].min() + found[:, 0].max()) / 2.0
    tracks = []
    for side in (found[found[:, 0] < split_x], found[found[:, 0] > split_x]):
        tracks.append((side[:, 0].min(), side[:, 0].max(), side[:, 1].min(), side[:, 1].max()))
    return tracks


def marker_rows(pixels, track):
    left, right, top, bottom = track
    reach = hud.PEDAL_MARKER_OVERHANG_PX + 2
    window = pixels[left - reach : right + reach + 1, top - reach : bottom + reach + 1]
    found = np.argwhere(np.all(window == np.array(hud.COLOR_ASSIST_POLICY), axis=-1))
    return found[:, 0] + left - reach, found[:, 1] + top - reach


def expected_marker_row(track, fraction):
    _, _, top, bottom = track
    track_height_px = bottom - top + 1
    return top + track_height_px - int(fraction * track_height_px)


@pytest.mark.parametrize(
    "mode", [binding.ASSIST_MODE_DISPLAY, binding.ASSIST_MODE_BLEND, binding.ASSIST_MODE_NUDGE, binding.ASSIST_MODE_FULL]
)
@pytest.mark.parametrize("accel_mps2", [2.0, 3.0, 10.0, -4.5, -2.25, -20.0, 0.0])
def test_policy_marker_sits_at_the_policys_pedal_target(pygame_headless, town_hud, mode, accel_mps2):
    tracks = bar_tracks(pygame_headless, town_hud)
    pixels = panel_pixels(pygame_headless, town_hud, hud_assist_accel(mode, accel_mps2), WheelStateStub(0.3, 0.6))
    targets = hud.policy_pedal_targets(accel_mps2, binding.HUMAN_ACCEL_MIN_MPS2, binding.HUMAN_ACCEL_MAX_MPS2)
    for bar_idx in (THROTTLE_BAR_IDX, BRAKE_BAR_IDX):
        columns, rows = marker_rows(pixels, tracks[bar_idx])
        assert len(rows) > 0
        assert np.median(rows) == pytest.approx(expected_marker_row(tracks[bar_idx], targets[bar_idx]), abs=1)
        assert rows.max() - rows.min() + 1 == hud.PEDAL_MARKER_THICKNESS_PX
        assert columns.min() == tracks[bar_idx][0] - hud.PEDAL_MARKER_OVERHANG_PX
        assert tracks[bar_idx][1] + hud.PEDAL_MARKER_OVERHANG_PX <= columns.max() <= tracks[bar_idx][1] + hud.PEDAL_MARKER_OVERHANG_PX + 1


def test_policy_marker_climbs_the_bar_as_the_policy_asks_for_more(pygame_headless, town_hud):
    tracks = bar_tracks(pygame_headless, town_hud)
    throttle_rows, brake_rows = [], []
    for accel_mps2 in (0.5, 1.5, 3.0):
        pixels = panel_pixels(pygame_headless, town_hud, hud_assist_accel(binding.ASSIST_MODE_DISPLAY, accel_mps2),
                              WheelStateStub(0.0, 0.0))
        throttle_rows.append(np.median(marker_rows(pixels, tracks[THROTTLE_BAR_IDX])[1]))
    for accel_mps2 in (-1.0, -4.0, -8.0):
        pixels = panel_pixels(pygame_headless, town_hud, hud_assist_accel(binding.ASSIST_MODE_DISPLAY, accel_mps2),
                              WheelStateStub(0.0, 0.0))
        brake_rows.append(np.median(marker_rows(pixels, tracks[BRAKE_BAR_IDX])[1]))
    assert throttle_rows[0] > throttle_rows[1] > throttle_rows[2]
    assert brake_rows[0] > brake_rows[1] > brake_rows[2]


def test_policy_marker_is_cyan():
    red, green, blue = hud.COLOR_ASSIST_POLICY
    assert blue > red and green > red
    assert hud.COLOR_ASSIST_POLICY not in (hud.COLOR_THROTTLE, hud.COLOR_BRAKE, hud.COLOR_BAR_TRACK)


@pytest.mark.parametrize(
    "assist",
    [None, hud_assist_accel(binding.ASSIST_MODE_OFF, 2.0), hud_assist_accel(binding.ASSIST_MODE_FULL, 2.0, 0),
     hud_assist_accel(binding.ASSIST_MODE_DISPLAY, -3.0, 0)],
    ids=["no_assist", "off", "full_invalid", "display_invalid"],
)
def test_no_policy_marker_without_an_active_valid_assist(pygame_headless, town_hud, assist):
    tracks = bar_tracks(pygame_headless, town_hud)
    pixels = panel_pixels(pygame_headless, town_hud, assist, WheelStateStub(0.3, 0.6))
    for track in tracks:
        assert len(marker_rows(pixels, track)[1]) == 0


@pytest.fixture
def panel_texts(pygame_headless, town_hud, monkeypatch):
    texts = []
    real_text = town_hud._text

    def text(surface, font, message, position, color=hud.COLOR_TEXT, anchor="topleft"):
        texts.append(message)
        return real_text(surface, font, message, position, color, anchor)

    monkeypatch.setattr(town_hud, "_text", text)
    return texts


@pytest.mark.parametrize(
    "assist,throttle,brake,expected",
    [
        (hud_assist(binding.ASSIST_MODE_FULL), 0.0, 0.0, True),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.04, 0.01, True),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.05, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.0501, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_FULL), 0.0, 0.3, False),
        (hud_assist(binding.ASSIST_MODE_FULL, reference_valid=0), 0.0, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_BLEND), 0.0, 0.0, False),
        (hud_assist(binding.ASSIST_MODE_NUDGE), 0.0, 0.0, False),
        (None, 0.0, 0.0, False),
    ],
)
def test_policy_speed_flag_is_shown_exactly_when_the_policy_drives_the_speed(
    pygame_headless, town_hud, panel_texts, assist, throttle, brake, expected
):
    surface = pygame_headless.Surface(SCREEN_SIZE_PX)
    town_hud.draw(surface, hud_human(accel_long_mps2=0.0, steering_angle_rad=0.0), WheelStateStub(throttle, brake),
                  HudInfo(), 0.0, assist=assist)
    assert any(POLICY_SPEED_FLAG in message for message in panel_texts) == expected


@pytest.mark.parametrize(
    "gear,gearbox_mode,expected",
    [(binding.GEAR_REVERSE, binding.GEARBOX_MODE_AUTOMATIC, False), (binding.GEAR_NEUTRAL, binding.GEARBOX_MODE_MANUAL, False),
     (1, binding.GEARBOX_MODE_MANUAL, True), (3, binding.GEARBOX_MODE_AUTOMATIC, True)],
)
def test_no_policy_speed_flag_in_reverse_or_neutral(pygame_headless, town_hud, panel_texts, gear, gearbox_mode, expected):
    surface = pygame_headless.Surface(SCREEN_SIZE_PX)
    human = hud_human(accel_long_mps2=0.0, steering_angle_rad=0.0, gear=gear, gearbox_mode=gearbox_mode)
    town_hud.draw(surface, human, WheelStateStub(0.0, 0.0), HudInfo(), 0.0, assist=hud_assist(binding.ASSIST_MODE_FULL))
    assert any(POLICY_SPEED_FLAG in message for message in panel_texts) == expected


def test_policy_speed_flag_joins_the_other_warnings(pygame_headless, town_hud, panel_texts):
    surface = pygame_headless.Surface(SCREEN_SIZE_PX)
    town_hud.draw(surface, hud_human(accel_long_mps2=0.0, steering_angle_rad=0.0, collision=1.0), WheelStateStub(0.0, 0.0),
                  HudInfo(), 0.0, assist=hud_assist(binding.ASSIST_MODE_FULL))
    flag_lines = [message for message in panel_texts if POLICY_SPEED_FLAG in message]
    assert len(flag_lines) == 1
    assert "COLLISION" in flag_lines[0]
