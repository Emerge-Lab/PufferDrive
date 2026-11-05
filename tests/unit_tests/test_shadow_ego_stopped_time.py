from types import SimpleNamespace

import numpy as np

from pufferlib.ocean.cosim.carla.shadow_ego import (
    PARTNER_ACTOR_ID,
    PARTNER_IS_WALKER,
    PARTNER_ROW_COLUMNS,
    STATE_VX,
    ShadowEgo,
)


STEP_SECONDS = 0.05
VEHICLE_ROW, WALKER_ROW = 0, 1


def _state_to_bin(x, y, z, yaw_deg, vx, vy, yaw_rate, accel_x, accel_y, box_z, extent_z):
    return x, y, z, 0.0, vx, vy, yaw_rate, accel_x


def _shadow(zero_partner_stopped_time):
    return SimpleNamespace(
        transform=SimpleNamespace(state_to_bin=_state_to_bin),
        step_count=0,
        tick_dt=STEP_SECONDS,
        dt=STEP_SECONDS,
        walker_speed_floor_mps=0.0,
        zero_partner_stopped_time=zero_partner_stopped_time,
        _partner_stopped_speed_threshold=0.2,
        _partner_stopped_s={},
    )


def _partners(walker_speed_mps=0.0):
    partners = np.zeros((2, PARTNER_ROW_COLUMNS), np.float32)
    partners[VEHICLE_ROW, PARTNER_ACTOR_ID] = 11
    partners[WALKER_ROW, PARTNER_ACTOR_ID] = 12
    partners[WALKER_ROW, PARTNER_IS_WALKER] = 1.0
    partners[WALKER_ROW, STATE_VX] = walker_speed_mps
    return partners


def _stopped_seconds(shadow, partners):
    return ShadowEgo._read_states(shadow, partners)[-1]


def test_zeroing_holds_vehicles_at_zero_and_keeps_the_walker_counter():
    shadow = _shadow(True)
    for _ in range(4):
        stopped_seconds = _stopped_seconds(shadow, _partners())
    assert stopped_seconds[VEHICLE_ROW] == 0.0
    assert abs(stopped_seconds[WALKER_ROW] - 4 * STEP_SECONDS) < 1e-6


def test_walker_counter_restarts_once_the_walker_moves():
    shadow = _shadow(True)
    for _ in range(4):
        _stopped_seconds(shadow, _partners())
    assert _stopped_seconds(shadow, _partners(walker_speed_mps=1.0))[WALKER_ROW] == 0.0
    assert abs(_stopped_seconds(shadow, _partners())[WALKER_ROW] - STEP_SECONDS) < 1e-6


def test_without_zeroing_every_partner_keeps_its_counter():
    shadow = _shadow(False)
    for _ in range(4):
        stopped_seconds = _stopped_seconds(shadow, _partners())
    assert abs(stopped_seconds[VEHICLE_ROW] - 4 * STEP_SECONDS) < 1e-6
    assert abs(stopped_seconds[WALKER_ROW] - 4 * STEP_SECONDS) < 1e-6
