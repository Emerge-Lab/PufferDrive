"""Shadow-env accel/steering targets -> CaRL ActionTrajectory commands for nuPlan's KinematicBicycleModel."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.ocean.cosim.nuplan_bicycle import (
    BicycleCommandConfig,
    bicycle_commands,
    bicycle_response,
    center_accel_shift,
    low_pass_gain,
    low_pass_response,
)

DT_S = 0.1


def devkit_step(cfg, accel, steering, speed, accel_cmd, steering_rate_cmd):
    """KinematicBicycleModel._update_commands + forward Euler on the longitudinal state, written out."""
    accel_gain = low_pass_gain(DT_S, cfg.accel_time_constant_s)
    steering_gain = low_pass_gain(DT_S, cfg.steering_time_constant_s)
    accel_next = accel_gain * (accel_cmd - accel) + accel
    steering_next = steering_gain * (DT_S * steering_rate_cmd + steering - steering) + steering
    return accel_next, steering_next, speed + accel_next * DT_S


def test_undo_low_pass_lands_on_the_targets_after_one_devkit_step():
    cfg = BicycleCommandConfig(dt_s=DT_S, min_speed_mps=-2.0, undo_low_pass=True)
    accel_cmd, rate_cmd = bicycle_commands(cfg, 1.5, -0.2, 0.3, 0.1, 5.0)
    accel_next, steering_next, _ = devkit_step(cfg, 0.3, 0.1, 5.0, accel_cmd, rate_cmd)
    assert accel_next == pytest.approx(1.5)
    assert steering_next == pytest.approx(-0.2)


def test_without_undo_the_model_realises_the_lag_fraction_of_the_step():
    cfg = BicycleCommandConfig(dt_s=DT_S, min_speed_mps=-2.0, undo_low_pass=False)
    accel_cmd, rate_cmd = bicycle_commands(cfg, 1.5, -0.2, 0.3, 0.1, 5.0)
    assert accel_cmd == pytest.approx(1.5)
    assert rate_cmd == pytest.approx((-0.2 - 0.1) / DT_S)
    accel_next, steering_next, _ = devkit_step(cfg, 0.3, 0.1, 5.0, accel_cmd, rate_cmd)
    assert accel_next == pytest.approx(0.3 + (1.5 - 0.3) / 3.0)
    assert steering_next == pytest.approx(0.1 + (-0.2 - 0.1) * 2.0 / 3.0)


@pytest.mark.parametrize("undo_low_pass", [False, True])
@pytest.mark.parametrize("min_speed_mps, speed_mps", [(0.0, 0.05), (-2.0, -1.95)])
def test_speed_floor_binds_on_the_realised_acceleration(undo_low_pass, min_speed_mps, speed_mps):
    cfg = BicycleCommandConfig(dt_s=DT_S, min_speed_mps=min_speed_mps, undo_low_pass=undo_low_pass)
    accel_cmd, _ = bicycle_commands(cfg, -3.0, 0.0, -1.0, 0.0, speed_mps)
    _, _, speed_next = devkit_step(cfg, -1.0, 0.0, speed_mps, accel_cmd, 0.0)
    assert speed_next == pytest.approx(min_speed_mps)


def test_floor_zero_holds_a_standstill_against_a_braking_target():
    cfg = BicycleCommandConfig(dt_s=DT_S, min_speed_mps=0.0, undo_low_pass=False)
    accel_cmd, _ = bicycle_commands(cfg, -2.0, 0.0, -0.5, 0.0, 0.0)
    accel_next, _, speed_next = devkit_step(cfg, -0.5, 0.0, 0.0, accel_cmd, 0.0)
    assert accel_next == pytest.approx(0.0)
    assert speed_next == pytest.approx(0.0)


@pytest.mark.parametrize("undo_low_pass", [False, True])
@pytest.mark.parametrize("min_speed_mps", [0.0, -2.0])
def test_shadow_stop_snap_brings_the_model_to_exactly_zero_speed(undo_low_pass, min_speed_mps):
    cfg = BicycleCommandConfig(dt_s=DT_S, min_speed_mps=min_speed_mps, undo_low_pass=undo_low_pass)
    accel_cmd, _ = bicycle_commands(cfg, 0.0, 0.0, -1.8, 0.0, 0.15, stop_this_step=True)
    _, _, speed_next = devkit_step(cfg, -1.8, 0.0, 0.15, accel_cmd, 0.0)
    assert speed_next == pytest.approx(0.0, abs=1e-12)


def test_floor_leaves_a_non_crossing_command_alone():
    cfg = BicycleCommandConfig(dt_s=DT_S, min_speed_mps=0.0, undo_low_pass=False)
    accel_cmd, _ = bicycle_commands(cfg, -3.0, 0.0, -1.0, 0.0, 8.0)
    assert accel_cmd == pytest.approx(-3.0)


def test_response_matches_the_devkit_step_for_the_actual_elapsed_time():
    cfg = BicycleCommandConfig(dt_s=DT_S, min_speed_mps=-2.0, undo_low_pass=True)
    accel_cmd, _ = bicycle_commands(cfg, 1.0, 0.0, 0.0, 0.0, 3.0)
    accel_next, _, speed_next = devkit_step(cfg, 0.0, 0.0, 3.0, accel_cmd, 0.0)
    assert bicycle_response(cfg, 0.0, accel_cmd, 3.0, DT_S) == pytest.approx((accel_next, speed_next))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dt_s=0.0, min_speed_mps=0.0, undo_low_pass=False),
        dict(dt_s=math.nan, min_speed_mps=0.0, undo_low_pass=False),
        dict(dt_s=DT_S, min_speed_mps=0.5, undo_low_pass=False),
        dict(dt_s=DT_S, min_speed_mps=0.0, undo_low_pass=False, accel_time_constant_s=-0.1),
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        BicycleCommandConfig(**kwargs)


PACIFICA_WHEEL_BASE_M = 3.089
PACIFICA_REAR_AXLE_TO_CENTER_M = 1.461
ACCEL_CAP_MPS2 = 2.304


def _center_cap_cfg(undo_low_pass):
    return BicycleCommandConfig(
        dt_s=DT_S,
        min_speed_mps=0.0,
        undo_low_pass=undo_low_pass,
        center_accel_cap_mps2=ACCEL_CAP_MPS2,
        wheel_base_m=PACIFICA_WHEEL_BASE_M,
        rear_axle_to_center_m=PACIFICA_REAR_AXLE_TO_CENTER_M,
    )


@pytest.mark.parametrize("undo_low_pass", [False, True])
def test_center_accel_cap_binds_only_when_the_yaw_terms_push_the_center_over_the_cap(undo_low_pass):
    cfg = _center_cap_cfg(undo_low_pass)
    accel_gain = low_pass_gain(DT_S, cfg.accel_time_constant_s)
    speed, accel = 0.5, 2.0
    straight, _ = bicycle_commands(cfg, ACCEL_CAP_MPS2, 0.0, accel, 0.0, speed)
    unlimited = BicycleCommandConfig(dt_s=DT_S, min_speed_mps=0.0, undo_low_pass=undo_low_pass)
    assert straight == pytest.approx(bicycle_commands(unlimited, ACCEL_CAP_MPS2, 0.0, accel, 0.0, speed)[0])
    left_turn, rate = bicycle_commands(cfg, ACCEL_CAP_MPS2, 0.3, accel, 0.3, speed, yaw_rate_radps=0.0)
    realised = low_pass_response(accel, left_turn, accel_gain)
    assert realised < low_pass_response(accel, straight, accel_gain)
    steering_next = low_pass_response(0.3, 0.3 + rate * DT_S, low_pass_gain(DT_S, cfg.steering_time_constant_s))
    assert realised + center_accel_shift(cfg, realised, speed, steering_next, 0.0) <= ACCEL_CAP_MPS2 + 1e-9
    right_turn, _ = bicycle_commands(cfg, ACCEL_CAP_MPS2, -0.3, accel, -0.3, speed, yaw_rate_radps=0.0)
    assert right_turn == pytest.approx(straight)


def test_center_accel_cap_config_needs_the_vehicle_geometry():
    with pytest.raises(ValueError):
        BicycleCommandConfig(dt_s=DT_S, min_speed_mps=0.0, undo_low_pass=False, center_accel_cap_mps2=2.0)


def test_center_accel_cap_holds_in_the_devkit_kinematic_bicycle_model():
    pytest.importorskip("nuplan")
    from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel

    params = get_pacifica_parameters()
    cfg = BicycleCommandConfig(
        dt_s=DT_S,
        min_speed_mps=0.0,
        undo_low_pass=True,
        center_accel_cap_mps2=ACCEL_CAP_MPS2,
        wheel_base_m=params.wheel_base,
        rear_axle_to_center_m=params.rear_axle_to_center,
    )
    model = KinematicBicycleModel(params)
    speed, accel, steering = 0.5, 2.0, 0.3
    yaw_rate = speed * math.tan(steering) / params.wheel_base
    state = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(0.0, 0.0, 0.0),
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(accel, 0.0),
        tire_steering_angle=steering,
        time_point=TimePoint(0),
        vehicle_parameters=params,
        angular_vel=yaw_rate,
    )
    accel_cmd, rate_cmd = bicycle_commands(
        cfg, ACCEL_CAP_MPS2, steering, accel, steering, speed, yaw_rate_radps=yaw_rate
    )
    command = DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist=state.car_footprint.rear_axle_to_center_dist,
        rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
        rear_axle_acceleration_2d=StateVector2D(accel_cmd, 0.0),
        tire_steering_rate=rate_cmd,
    )
    propagated = model.propagate_state(state, command, TimePoint(int(DT_S * 1e6)))
    assert propagated.dynamic_car_state.rear_axle_acceleration_2d.x < ACCEL_CAP_MPS2
    center_accel = propagated.dynamic_car_state.center_acceleration_2d.x
    assert ACCEL_CAP_MPS2 - 0.1 < center_accel <= ACCEL_CAP_MPS2 + 1e-9  # one-sided clip: under the cap, close to it


def test_commands_drive_the_devkit_kinematic_bicycle_model_onto_the_targets():
    pytest.importorskip("nuplan")
    from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel

    params = get_pacifica_parameters()
    model = KinematicBicycleModel(params)
    accel, steering, speed = 0.3, 0.1, 5.0
    state = EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(10.0, 20.0, 0.5),
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(accel, 0.0),
        tire_steering_angle=steering,
        time_point=TimePoint(0),
        vehicle_parameters=params,
    )
    cfg = BicycleCommandConfig(dt_s=DT_S, min_speed_mps=-2.0, undo_low_pass=True)
    accel_cmd, rate_cmd = bicycle_commands(cfg, 1.5, -0.2, accel, steering, speed)
    command = DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist=state.car_footprint.rear_axle_to_center_dist,
        rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
        rear_axle_acceleration_2d=StateVector2D(accel_cmd, 0.0),
        tire_steering_rate=rate_cmd,
    )
    propagated = model.propagate_state(state, command, TimePoint(int(DT_S * 1e6)))
    assert propagated.dynamic_car_state.rear_axle_acceleration_2d.x == pytest.approx(1.5)
    assert propagated.tire_steering_angle == pytest.approx(-0.2)
    expected = bicycle_response(cfg, accel, accel_cmd, speed, DT_S)
    assert propagated.dynamic_car_state.rear_axle_velocity_2d.x == pytest.approx(expected[1])
