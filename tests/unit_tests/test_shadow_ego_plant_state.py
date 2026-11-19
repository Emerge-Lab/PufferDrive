import math
from types import SimpleNamespace

from opengigaflow.ocean.cosim.carla.controller import pedal_gains_for
from opengigaflow.ocean.cosim.carla.shadow_ego import DRIVE_WHEELBASE_PER_LENGTH, ShadowEgo


GAINS = pedal_gains_for("vehicle.lincoln.mkz2017")
EGO_HALF_LENGTH_M = 2.45
HEADING_RAD = 0.3
RESIDUAL_SPEED_MPS = 0.02


def _shadow(intent_accel_mps2=0.0, held=False, intent_steering_rad=0.0):
    return SimpleNamespace(
        controller=SimpleNamespace(gains=GAINS, observe=lambda speed_mps, yaw_rate_radps, tick_dt: None),
        tick_dt=0.05,
        ego_extent=[EGO_HALF_LENGTH_M, 1.0, 0.75],
        _intent_accel_mps2=intent_accel_mps2,
        _intent_steering_rad=intent_steering_rad,
        _held=held,
        _plant_speed_mps=0.0,
    )


def _plant_state(shadow, forward_speed_mps):
    vx, vy = forward_speed_mps * math.cos(HEADING_RAD), forward_speed_mps * math.sin(HEADING_RAD)
    return ShadowEgo._plant_state(shadow, (1.0, 2.0, 0.0, HEADING_RAD, vx, vy, 0.0, 0.0))


def test_a_moving_car_keeps_its_velocity_and_the_integrator_acceleration():
    shadow = _shadow(intent_accel_mps2=-2.0, held=True)
    _, _, _, _, vx, vy, _, accel_mps2 = _plant_state(shadow, 3.0)
    assert math.isclose(math.hypot(vx, vy), 3.0)
    assert accel_mps2 == -2.0
    assert not shadow._held


def test_braking_to_a_stop_holds_the_car_and_starts_the_acceleration_from_zero():
    shadow = _shadow(intent_accel_mps2=-2.0)
    _, _, _, _, vx, vy, _, accel_mps2 = _plant_state(shadow, RESIDUAL_SPEED_MPS)
    assert (vx, vy, accel_mps2) == (0.0, 0.0, 0.0)
    assert shadow._held


def test_a_launch_from_the_hold_keeps_its_acceleration():
    shadow = _shadow(intent_accel_mps2=0.4, held=True)
    accel_mps2 = _plant_state(shadow, 0.0)[7]
    assert accel_mps2 == 0.4
    assert shadow._held


def test_a_request_to_go_backwards_never_builds_up():
    shadow = _shadow(intent_accel_mps2=-0.4, held=True)
    accel_mps2 = _plant_state(shadow, 0.0)[7]
    assert accel_mps2 == 0.0
    assert shadow._held


def test_a_car_that_rolls_backwards_is_held_like_a_standing_one():
    shadow = _shadow(intent_accel_mps2=-1.0)
    _, _, _, _, vx, vy, _, accel_mps2 = _plant_state(shadow, -0.5)
    assert (vx, vy, accel_mps2) == (0.0, 0.0, 0.0)
    assert shadow._held


def test_a_standing_car_reads_stopped_even_when_it_creeps_against_the_launch():
    shadow = _shadow(intent_accel_mps2=0.4, held=True)
    _, _, _, _, vx, vy, _, accel_mps2 = _plant_state(shadow, -RESIDUAL_SPEED_MPS)
    assert (vx, vy) == (0.0, 0.0)
    assert accel_mps2 == 0.4


def test_a_car_that_creeps_in_the_direction_of_its_launch_keeps_its_velocity():
    shadow = _shadow(intent_accel_mps2=0.4, held=True)
    _, _, _, _, vx, vy, _, _ = _plant_state(shadow, RESIDUAL_SPEED_MPS)
    assert math.isclose(math.hypot(vx, vy), RESIDUAL_SPEED_MPS)


def test_a_car_that_is_still_accelerating_when_it_stalls_is_not_held():
    shadow = _shadow(intent_accel_mps2=1.0)
    accel_mps2 = _plant_state(shadow, RESIDUAL_SPEED_MPS)[7]
    assert accel_mps2 == 1.0
    assert not shadow._held


def test_the_reported_heading_is_the_direction_the_centre_travels_in():
    steering_rad = 0.3
    wheelbase_m = DRIVE_WHEELBASE_PER_LENGTH * 2.0 * EGO_HALF_LENGTH_M
    slip_rad = math.atan(GAINS.rear_axle_to_center_m * math.tan(steering_rad) / wheelbase_m)
    assert math.isclose(_plant_state(_shadow(intent_steering_rad=steering_rad), 3.0)[3], HEADING_RAD + slip_rad)
    assert math.isclose(_plant_state(_shadow(intent_steering_rad=-steering_rad), 3.0)[3], HEADING_RAD - slip_rad)
    assert _plant_state(_shadow(), 3.0)[3] == HEADING_RAD
