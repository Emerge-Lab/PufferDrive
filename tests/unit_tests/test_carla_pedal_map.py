import math

import numpy as np
import pytest

from pufferlib.ocean.cosim.carla.controller import PEDAL_GAINS_BY_BLUEPRINT, PedalMap, pedal_gains_for, pedals


GAINS = PEDAL_GAINS_BY_BLUEPRINT["vehicle.lincoln.mkz_2020"]
SECOND_GEAR = GAINS.gears[1]
TICK_SECONDS = 0.05
CRUISE_SPEED_MPS = 8.0
MAX_SPEED_MPS = 30.0


def _holding_throttle(gear_gains, speed_mps):
    return float(np.interp(speed_mps, gear_gains.speeds_mps, gear_gains.throttle_holding_speed))


def _accel_per_throttle(gear_gains, speed_mps):
    return float(np.interp(speed_mps, gear_gains.speeds_mps, gear_gains.accel_per_throttle_mps2))


def _engine_braking(gear_gains, speed_mps):
    steady = _holding_throttle(gear_gains, speed_mps) * _accel_per_throttle(gear_gains, speed_mps)
    return min(steady, GAINS.max_engine_braking_mps2)


def test_zero_acceleration_is_the_speed_holding_throttle_of_the_gear():
    throttle, brake = pedals(
        SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, 0.0, CRUISE_SPEED_MPS
    )
    assert brake == 0.0
    assert math.isclose(throttle, _holding_throttle(SECOND_GEAR, CRUISE_SPEED_MPS))


def test_an_extra_mps2_costs_more_throttle_in_a_higher_gear():
    def throttle_per_mps2(gear_gains):
        slow = pedals(gear_gains, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, 1.0, CRUISE_SPEED_MPS)[0]
        fast = pedals(gear_gains, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, 2.0, CRUISE_SPEED_MPS)[0]
        return fast - slow

    assert throttle_per_mps2(GAINS.gears[3]) > throttle_per_mps2(GAINS.gears[0])


def test_gentle_deceleration_only_lifts_the_throttle():
    engine_braking = _engine_braking(SECOND_GEAR, CRUISE_SPEED_MPS)
    throttle, brake = pedals(
        SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, -0.5 * engine_braking, CRUISE_SPEED_MPS
    )
    assert brake == 0.0
    assert 0.0 < throttle < _holding_throttle(SECOND_GEAR, CRUISE_SPEED_MPS)


def test_the_brake_supplies_what_engine_braking_does_not():
    engine_braking = _engine_braking(SECOND_GEAR, CRUISE_SPEED_MPS)
    _, brake = pedals(
        SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, -engine_braking - 1.0, CRUISE_SPEED_MPS
    )
    assert math.isclose(brake, GAINS.brake_per_mps2)


def test_while_the_gearbox_shifts_the_brake_supplies_the_whole_deceleration():
    _, brake = pedals(SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, True, -1.0, CRUISE_SPEED_MPS)
    assert math.isclose(brake, GAINS.brake_per_mps2)


def test_pedals_stay_within_zero_and_one():
    assert pedals(SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, 100.0, 0.0) == (1.0, 0.0)
    assert pedals(SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, -100.0, 0.0) == (0.0, 1.0)


@pytest.mark.parametrize("blueprint", sorted(PEDAL_GAINS_BY_BLUEPRINT))
def test_every_gear_has_usable_numbers_at_increasing_speeds(blueprint):
    gains = pedal_gains_for(blueprint)
    for gear_gains in gains.gears:
        assert all(low < high for low, high in zip(gear_gains.speeds_mps, gear_gains.speeds_mps[1:]))
        assert (
            len(gear_gains.speeds_mps)
            == len(gear_gains.throttle_holding_speed)
            == len(gear_gains.accel_per_throttle_mps2)
        )
        assert all(0.0 < throttle < 1.0 for throttle in gear_gains.throttle_holding_speed)
        assert all(accel > 1.0 for accel in gear_gains.accel_per_throttle_mps2)
    assert gains.gears[0].speeds_mps[0] == 0.0
    assert gains.curvature_per_steer_at_rest + gains.curvature_per_steer_per_mps * MAX_SPEED_MPS > 0.2


def test_outside_its_speed_span_a_gear_keeps_the_numbers_of_the_nearest_end():
    low_speed_mps, high_speed_mps = SECOND_GEAR.speeds_mps[0], SECOND_GEAR.speeds_mps[-1]
    below = pedals(SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, 1.0, low_speed_mps - 3.0)
    assert below == pedals(SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, 1.0, low_speed_mps)
    above = pedals(SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, 1.0, high_speed_mps + 3.0)
    assert above == pedals(SECOND_GEAR, GAINS.max_engine_braking_mps2, GAINS.brake_per_mps2, False, 1.0, high_speed_mps)


def test_steer_follows_the_curvature_sign_weakens_with_speed_and_clips():
    pedal_map = PedalMap(GAINS)
    slow = pedal_map.step(0.0, 0.1, 2.0, 1, False)[0]
    fast = pedal_map.step(0.0, 0.1, 20.0, 4, False)[0]
    assert math.isclose(slow, 0.1 / (GAINS.curvature_per_steer_at_rest + GAINS.curvature_per_steer_per_mps * 2.0))
    assert fast > slow
    assert pedal_map.step(0.0, -0.1, 5.0, 2, False)[0] < 0.0
    assert pedal_map.step(0.0, 10.0, 5.0, 2, False)[0] == 1.0


def test_a_shift_keeps_the_throttle_line_of_the_gear_last_engaged():
    pedal_map = PedalMap(GAINS)
    in_third = pedal_map.step(1.0, 0.0, CRUISE_SPEED_MPS, 3, False)
    shifting = pedal_map.step(1.0, 0.0, CRUISE_SPEED_MPS, 0, False)
    assert shifting == in_third


def test_standstill_hold_is_full_brake_and_is_not_scored():
    pedal_map = PedalMap(GAINS)
    assert pedal_map.step(0.0, 0.0, 0.0, 1, True) == (0.0, 0.0, 1.0)
    pedal_map.observe(0.0, 0.0, TICK_SECONDS)
    assert pedal_map.stats() == {}


def test_stats_report_the_acceleration_bias_and_the_yaw_rate_gain():
    pedal_map = PedalMap(GAINS)
    speed_mps = 5.0
    hard_braking_mps2 = -_engine_braking(SECOND_GEAR, speed_mps) - 2.0
    for target_accel_mps2 in (1.0, 1.0, hard_braking_mps2, hard_braking_mps2):
        pedal_map.step(target_accel_mps2, 0.1, speed_mps, 2, False)
        speed_mps += (target_accel_mps2 - 0.5) * TICK_SECONDS
        pedal_map.observe(speed_mps, 0.5 * 0.1 * speed_mps, TICK_SECONDS)
    stats = pedal_map.stats()
    assert stats["ticks"] == 4
    assert math.isclose(stats["accel_bias_throttle_mps2"], -0.5, abs_tol=1e-9)
    assert math.isclose(stats["accel_bias_brake_mps2"], -0.5, abs_tol=1e-9)
    assert math.isclose(stats["yaw_rate_gain"], 0.5, abs_tol=1e-9)


def test_an_unmeasured_car_is_refused():
    with pytest.raises(ValueError):
        pedal_gains_for("vehicle.tesla.model3")
    with pytest.raises(ValueError):
        pedal_gains_for(None)
