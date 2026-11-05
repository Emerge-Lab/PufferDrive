import math

import pytest

from pufferlib.ocean.cosim.carla.controller import PEDAL_GAINS_BY_BLUEPRINT, PedalMap, pedal_gains_for, pedals


GAINS = PEDAL_GAINS_BY_BLUEPRINT["vehicle.lincoln.mkz_2020"]
TICK_SECONDS = 0.05


def test_zero_acceleration_is_the_speed_holding_throttle():
    assert pedals(GAINS, 0.0, 0.0) == (GAINS.throttle_holding_speed_at_rest, 0.0)
    assert pedals(GAINS, 0.0, GAINS.cruise_speed_mps) == (GAINS.throttle_holding_cruise_speed, 0.0)
    assert pedals(GAINS, 0.0, 5.0 * GAINS.cruise_speed_mps) == (GAINS.throttle_holding_cruise_speed, 0.0)


def test_acceleration_adds_throttle_and_no_brake():
    throttle, brake = pedals(GAINS, 1.0, GAINS.cruise_speed_mps)
    assert brake == 0.0
    assert math.isclose(throttle, GAINS.throttle_holding_cruise_speed + GAINS.throttle_per_mps2)


def test_gentle_deceleration_only_lifts_the_throttle():
    throttle, brake = pedals(GAINS, -0.5 * GAINS.engine_braking_mps2, GAINS.cruise_speed_mps)
    assert brake == 0.0
    assert 0.0 < throttle < GAINS.throttle_holding_cruise_speed


def test_the_brake_supplies_what_engine_braking_does_not():
    throttle, brake = pedals(GAINS, -GAINS.engine_braking_mps2 - 1.0, GAINS.cruise_speed_mps)
    assert math.isclose(brake, GAINS.brake_per_mps2)
    assert throttle < GAINS.throttle_holding_cruise_speed


def test_pedals_stay_within_zero_and_one():
    assert pedals(GAINS, 100.0, 0.0) == (1.0, 0.0)
    assert pedals(GAINS, -100.0, 0.0) == (0.0, 1.0)


def test_steer_follows_the_curvature_sign_and_clips():
    pedal_map = PedalMap(GAINS)
    assert math.isclose(pedal_map.step(0.0, 0.1, 5.0, False)[0], 0.1 * GAINS.steer_per_curvature_m)
    assert pedal_map.step(0.0, -0.1, 5.0, False)[0] < 0.0
    assert pedal_map.step(0.0, 10.0, 5.0, False)[0] == 1.0


def test_standstill_hold_is_full_brake_and_is_not_scored():
    pedal_map = PedalMap(GAINS)
    assert pedal_map.step(0.0, 0.0, 0.0, True) == (0.0, 0.0, 1.0)
    pedal_map.observe(0.0, 0.0, TICK_SECONDS)
    assert pedal_map.stats() == {}


def test_stats_report_the_acceleration_bias_and_the_yaw_rate_gain():
    pedal_map = PedalMap(GAINS)
    speed_mps = 5.0
    hard_braking_mps2 = -GAINS.engine_braking_mps2 - 1.0
    for target_accel_mps2 in (1.0, 1.0, hard_braking_mps2, hard_braking_mps2):
        pedal_map.step(target_accel_mps2, 0.1, speed_mps, False)
        speed_mps += (target_accel_mps2 - 0.5) * TICK_SECONDS
        pedal_map.observe(speed_mps, 0.5 * 0.1 * speed_mps, TICK_SECONDS)
    stats = pedal_map.stats()
    assert stats["ticks"] == 4
    assert math.isclose(stats["accel_bias_throttle_mps2"], -0.5, abs_tol=1e-9)
    assert math.isclose(stats["accel_bias_brake_mps2"], -0.5, abs_tol=1e-9)
    assert math.isclose(stats["yaw_rate_gain"], 0.5, abs_tol=1e-9)


def test_both_leaderboard_cars_have_gains_and_an_unmeasured_car_is_refused():
    assert pedal_gains_for("vehicle.lincoln.mkz2017").throttle_per_mps2 > GAINS.throttle_per_mps2
    with pytest.raises(ValueError):
        pedal_gains_for("vehicle.tesla.model3")
    with pytest.raises(ValueError):
        pedal_gains_for(None)
