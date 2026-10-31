"""M9 steering assist through the Python surface: binding, HUD readout and mode switching.

The sim is positive-LEFT; everything the driver reads (hud.assist_readout) is asserted in DRIVER
terms, positive = RIGHT. The C side is pinned in tests/drive/test_drive_steering_assist.c.
"""

import math
import os

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.wheel_demo.hud import assist_readout

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")
SCENARIO_LENGTH = 400
CONTRACT_RAMP_SECONDS = 0.5
POLICY_DT_SECONDS = 0.1
SUBSTEP_COUNT = 6
HUMAN_ACTION_ROW = 0
ASSIST_STATE_KEYS = {
    "mode",
    "blend_target",
    "blend_weight",
    "reference_valid",
    "reference_steering_rad",
    "reference_accel_mps2",
    "reference_speed_mps",
    "human_steering_rad",
    "commanded_steering_rad",
}
# The policy steering hard one way while the driver holds the other.
POLICY_STEER_ACTION = (0.3, 1.0)
DRIVER_STEERING_NORM = 0.5


def make_env():
    env = Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=1_000_000,
        termination_mode=False,
        goal_source="gt",
        report_interval=1,
        num_goals=3,
        goal_radius=2.0,
        reset_accel_on_stop=True,
        dynamics_model="jerk",
        action_type="continuous",
    )
    env.reset()
    env.select_human_agent(0)
    return env


@pytest.fixture
def env():
    drive_env = make_env()
    yield drive_env
    drive_env.close()


def policy_actions(env):
    actions = np.zeros_like(env.actions)
    actions[HUMAN_ACTION_ROW] = POLICY_STEER_ACTION
    return actions


def step(env, steering_norm=DRIVER_STEERING_NORM, accel_mps2=1.0):
    env.set_human_input(steering_norm, accel_mps2)
    env.step(policy_actions(env))
    return env.get_assist_state()


def test_assist_constants_are_exported():
    assert (
        binding.ASSIST_MODE_OFF,
        binding.ASSIST_MODE_DISPLAY,
        binding.ASSIST_MODE_BLEND,
        binding.ASSIST_MODE_NUDGE,
        binding.ASSIST_MODE_FULL,
    ) == (0, 1, 2, 3, 4)
    assert binding.ASSIST_RAMP_SECONDS == pytest.approx(CONTRACT_RAMP_SECONDS)


def test_assist_state_defaults_to_off_and_commands_the_driver(env):
    env.set_human_input(DRIVER_STEERING_NORM, 0.0)
    state = env.get_assist_state()
    assert set(state) == ASSIST_STATE_KEYS
    assert (state["mode"], state["blend_weight"], state["reference_valid"]) == (binding.ASSIST_MODE_OFF, 0.0, 0)
    expected_human = float(np.float32(-DRIVER_STEERING_NORM) * np.float32(binding.HUMAN_STEERING_LIMIT_RAD))
    assert state["human_steering_rad"] == expected_human
    assert state["commanded_steering_rad"] == state["human_steering_rad"]


@pytest.mark.parametrize(
    "mode,target",
    [(-1, 0.5), (5, 0.5), (binding.ASSIST_MODE_BLEND, math.nan), (binding.ASSIST_MODE_BLEND, math.inf),
     (binding.ASSIST_MODE_BLEND, -0.01), (binding.ASSIST_MODE_BLEND, 1.01)],
)
def test_set_assist_mode_rejects_and_keeps_the_previous_setting(env, mode, target):
    env.set_assist_mode(binding.ASSIST_MODE_DISPLAY, 0.25)
    with pytest.raises(ValueError):
        env.set_assist_mode(mode, target)
    state = env.get_assist_state()
    assert state["mode"] == binding.ASSIST_MODE_DISPLAY
    assert state["blend_target"] == pytest.approx(0.25)


@pytest.mark.parametrize("target", [0.0, 1.0])
def test_set_assist_mode_accepts_the_target_bounds(env, target):
    env.set_assist_mode(binding.ASSIST_MODE_BLEND, target)
    assert env.get_assist_state()["blend_target"] == target


def test_set_assist_mode_accepts_full(env):
    env.set_assist_mode(binding.ASSIST_MODE_FULL, 1.0)
    assert env.get_assist_state()["mode"] == binding.ASSIST_MODE_FULL


def test_full_steers_through_the_wheel_only_and_never_blends(env):
    env.set_assist_mode(binding.ASSIST_MODE_FULL, 1.0)
    for step_idx in range(15):
        state = step(env, steering_norm=math.sin(0.3 * step_idx))
        assert state["reference_valid"] == 1
        assert state["blend_weight"] == 0.0
        assert state["commanded_steering_rad"] == state["human_steering_rad"]
        assert env.get_human_state()["steering_angle_rad"] == state["human_steering_rad"]
    assert state["reference_steering_rad"] != state["human_steering_rad"]


def test_off_commands_exactly_the_driver_every_step(env):
    for step_idx in range(20):
        state = step(env, steering_norm=math.sin(0.3 * step_idx))
        assert state["commanded_steering_rad"] == state["human_steering_rad"]
        assert env.get_human_state()["steering_angle_rad"] == state["human_steering_rad"]


def test_display_reference_differs_from_the_driver_but_leaves_the_car_alone(env):
    env.set_assist_mode(binding.ASSIST_MODE_DISPLAY, 1.0)
    for _ in range(5):
        state = step(env)
        assert state["reference_valid"] == 1
        assert state["commanded_steering_rad"] == state["human_steering_rad"]
        assert env.get_human_state()["steering_angle_rad"] == state["human_steering_rad"]
    assert state["reference_steering_rad"] != state["human_steering_rad"]


def test_full_blend_drives_the_reference_once_the_ramp_is_done(env):
    env.set_assist_mode(binding.ASSIST_MODE_BLEND, 1.0)
    ramp_steps = math.ceil(CONTRACT_RAMP_SECONDS / POLICY_DT_SECONDS)
    for step_idx in range(ramp_steps + 10):
        state = step(env)
        if step_idx + 1 < ramp_steps:
            assert 0.0 < state["blend_weight"] < 1.0
            continue
        assert state["blend_weight"] == 1.0
        assert state["commanded_steering_rad"] == state["reference_steering_rad"]
        assert env.get_human_state()["steering_angle_rad"] == state["reference_steering_rad"]


def test_readout_delta_is_policy_minus_driver_in_driver_terms(env):
    env.set_assist_mode(binding.ASSIST_MODE_DISPLAY, 0.0)
    for steering_norm in (-0.8, -0.1, 0.0, 0.4, 0.9):
        assist = step(env, steering_norm=steering_norm)
        human = env.get_human_state()
        readout = assist_readout(assist, human["accel_long_mps2"], binding.HUMAN_STEERING_RATIO)
        policy_minus_you_sim_rad = assist["reference_steering_rad"] - assist["human_steering_rad"]
        assert readout.your_steering_rad == -assist["human_steering_rad"]
        assert readout.policy_steering_rad == -assist["reference_steering_rad"]
        assert readout.steer_delta_wheel_deg == pytest.approx(
            -math.degrees(policy_minus_you_sim_rad) * binding.HUMAN_STEERING_RATIO, abs=1e-9
        )
        assert readout.accel_delta_mps2 == pytest.approx(assist["reference_accel_mps2"] - human["accel_long_mps2"])


@pytest.mark.parametrize(
    "reference_sim_rad,human_sim_rad,expected_sign",
    [(-0.2, 0.0, +1), (0.2, 0.0, -1), (-0.1, -0.3, -1), (0.05, 0.05, 0)],
    ids=["policy_right", "policy_left", "driver_further_right", "agree"],
)
def test_readout_sign_is_positive_when_the_policy_wants_further_right(reference_sim_rad, human_sim_rad, expected_sign):
    assist = {"human_steering_rad": human_sim_rad, "reference_steering_rad": reference_sim_rad, "reference_accel_mps2": 1.5}
    readout = assist_readout(assist, 0.5, binding.HUMAN_STEERING_RATIO)
    assert np.sign(readout.steer_delta_wheel_deg) == expected_sign
    assert readout.accel_delta_mps2 == pytest.approx(1.0)


def test_readout_wheel_degrees_use_the_wheels_steering_ratio():
    road_delta_rad = math.radians(2.0)
    assist = {"human_steering_rad": 0.0, "reference_steering_rad": -road_delta_rad, "reference_accel_mps2": 0.0}
    readout = assist_readout(assist, 0.0, binding.HUMAN_STEERING_RATIO)
    full_lock_wheel_deg = binding.HUMAN_WHEEL_RANGE_DEG / 2.0
    assert readout.steer_delta_wheel_deg == pytest.approx(
        2.0 * full_lock_wheel_deg / math.degrees(binding.HUMAN_STEERING_LIMIT_RAD), rel=1e-5
    )


def ramp_bound(step_dt_s):
    return step_dt_s / CONTRACT_RAMP_SECONDS + 1e-6


@pytest.mark.parametrize(
    "sequence",
    [
        [(binding.ASSIST_MODE_BLEND, 1.0), (binding.ASSIST_MODE_OFF, 0.0)],
        [(binding.ASSIST_MODE_BLEND, 1.0), (binding.ASSIST_MODE_DISPLAY, 1.0), (binding.ASSIST_MODE_BLEND, 0.3)],
        [(binding.ASSIST_MODE_BLEND, 0.6), (binding.ASSIST_MODE_NUDGE, 1.0), (binding.ASSIST_MODE_BLEND, 0.0)],
    ],
)
def test_switching_modes_mid_window_never_jumps_the_weight(env, sequence):
    env.set_substepping(binding.SUBSTEP_MODE_ALL, SUBSTEP_COUNT)
    substep_dt_s = POLICY_DT_SECONDS / SUBSTEP_COUNT
    weights = [env.get_assist_state()["blend_weight"]]
    for mode, target in sequence:
        for policy_step in range(8):
            env.set_human_input(DRIVER_STEERING_NORM, 1.0)
            env.actions[:] = policy_actions(env)
            for substep_idx in range(SUBSTEP_COUNT):
                if policy_step == 0 and substep_idx == SUBSTEP_COUNT // 2:
                    env.set_assist_mode(mode, target)
                env.substep()
                weights.append(env.get_assist_state()["blend_weight"])
            env.step(env.actions.copy())
        expected = target if mode == binding.ASSIST_MODE_BLEND else 0.0
        assert weights[-1] == pytest.approx(expected, abs=1e-6)
    steps = np.abs(np.diff(weights))
    assert steps.max() <= ramp_bound(substep_dt_s)
    assert steps.max() > 0.0


def drive_off_window(env, steering_norm):
    env.set_human_input(steering_norm, 2.0)
    env.actions[:] = policy_actions(env)
    for _ in range(SUBSTEP_COUNT):
        env.substep()
    env.step(env.actions.copy())


@pytest.mark.parametrize("entered_mode", [binding.ASSIST_MODE_BLEND, binding.ASSIST_MODE_NUDGE, binding.ASSIST_MODE_FULL])
def test_entering_an_assist_mid_window_never_uses_a_stale_reference(env, entered_mode):
    env.set_substepping(binding.SUBSTEP_MODE_ALL, SUBSTEP_COUNT)
    env.set_assist_mode(binding.ASSIST_MODE_DISPLAY, 0.0)
    drive_off_window(env, DRIVER_STEERING_NORM)
    assert env.get_assist_state()["reference_valid"] == 1
    env.set_assist_mode(binding.ASSIST_MODE_OFF, 0.0)
    for _ in range(30):
        drive_off_window(env, -DRIVER_STEERING_NORM)
        assert env.get_assist_state()["reference_valid"] == 0
    env.set_human_input(-DRIVER_STEERING_NORM, 2.0)
    env.actions[:] = policy_actions(env)
    env.substep()
    env.set_assist_mode(entered_mode, 1.0)
    for _ in range(SUBSTEP_COUNT - 1):
        env.substep()
        state = env.get_assist_state()
        assert state["reference_valid"] == 0
        assert state["blend_weight"] == 0.0
        assert state["commanded_steering_rad"] == state["human_steering_rad"]
    env.step(env.actions.copy())
    env.substep()
    fresh = env.get_assist_state()
    assert fresh["reference_valid"] == 1
    expected_weight = (POLICY_DT_SECONDS / SUBSTEP_COUNT) / CONTRACT_RAMP_SECONDS
    if entered_mode == binding.ASSIST_MODE_BLEND:
        assert fresh["blend_weight"] == pytest.approx(expected_weight, rel=1e-5)
        assert fresh["commanded_steering_rad"] != fresh["human_steering_rad"]
        return
    assert fresh["blend_weight"] == 0.0
    assert fresh["commanded_steering_rad"] == fresh["human_steering_rad"]


def test_reset_clears_the_reference_and_the_weight(env):
    env.set_assist_mode(binding.ASSIST_MODE_BLEND, 1.0)
    for _ in range(3):
        step(env)
    state = env.get_assist_state()
    assert state["reference_valid"] == 1 and state["blend_weight"] > 0.0
    env.reset()
    state = env.get_assist_state()
    assert (state["reference_valid"], state["blend_weight"]) == (0, 0.0)


# ---------------------------------------------------------------------------------------------
# Longitudinal assist: BLEND shares the speed command, FULL drives it while both feet are off.
# ---------------------------------------------------------------------------------------------

CONTRACT_PEDAL_RELEASED_FRACTION = 0.05
CONTRACT_ACCEL_MIN_MPS2 = -9.0
CONTRACT_ACCEL_MAX_MPS2 = 4.0
JUST_RELEASED_NORM = 0.0499
CONTRACT_STOP_DECEL_MPS2 = 3.0
CREEP_THROTTLE_NORM = 0.06
CREEP_STEP_COUNT = 3
HOLD_STEP_COUNT = 20
HELD_THROTTLE_NORM = 0.5
LONG_STEP_COUNT = 15


def test_longitudinal_assist_constants_are_exported():
    assert binding.ASSIST_PEDAL_RELEASED_FRACTION == pytest.approx(CONTRACT_PEDAL_RELEASED_FRACTION)
    assert (binding.HUMAN_ACCEL_MIN_MPS2, binding.HUMAN_ACCEL_MAX_MPS2) == (CONTRACT_ACCEL_MIN_MPS2, CONTRACT_ACCEL_MAX_MPS2)
    assert binding.ASSIST_STOP_DECEL_MPS2 == CONTRACT_STOP_DECEL_MPS2


def clipped_reference(state):
    return float(np.float32(min(max(state["reference_accel_mps2"], CONTRACT_ACCEL_MIN_MPS2), CONTRACT_ACCEL_MAX_MPS2)))


def assisted_reference(state, speed_before_mps):
    reference = clipped_reference(state)
    if state["reference_speed_mps"] == 0.0 and speed_before_mps > 0.0:
        reference = min(reference, -CONTRACT_STOP_DECEL_MPS2)
    if speed_before_mps <= 0.0:
        reference = max(reference, 0.0)
    return reference


def accel_follows_the_reference(env, state):
    human = env.get_human_state()
    stopped_by_zero_crossing = human["speed_mps"] == 0.0 and human["accel_long_mps2"] == 0.0
    return human["accel_long_mps2"] == clipped_reference(state) or stopped_by_zero_crossing


def pedal_step(env, throttle_norm, brake_norm, steering_norm=DRIVER_STEERING_NORM):
    env.set_human_pedals(steering_norm, throttle_norm, brake_norm, 0.0)
    env.step(policy_actions(env))
    return env.get_assist_state()


def test_full_with_feet_off_drives_the_clipped_reference_accel(env):
    env.set_assist_mode(binding.ASSIST_MODE_FULL, 1.0)
    for step_idx in range(LONG_STEP_COUNT):
        state = pedal_step(env, JUST_RELEASED_NORM, JUST_RELEASED_NORM, steering_norm=math.sin(0.3 * step_idx))
        assert state["reference_valid"] == 1
        assert env.get_human_state()["gear"] >= 1
        assert accel_follows_the_reference(env, state)
        assert state["commanded_steering_rad"] == state["human_steering_rad"]


def human_trajectory(assist_mode, throttle_norm, brake_norm, blend_target=1.0, reverse=False):
    drive_env = make_env()
    try:
        if reverse:
            drive_env.set_human_pedals(0.0, 0.0, 1.0, 0.0, -1)
            drive_env.step(policy_actions(drive_env))
            assert drive_env.get_human_state()["gear"] == binding.GEAR_REVERSE
        drive_env.set_assist_mode(assist_mode, blend_target)
        trajectory = []
        for step_idx in range(LONG_STEP_COUNT):
            pedal_step(drive_env, throttle_norm, brake_norm, steering_norm=math.sin(0.3 * step_idx))
            human = drive_env.get_human_state()
            trajectory.append((human["sim_x"], human["sim_y"], human["speed_mps"], human["accel_long_mps2"], human["gear"]))
        return trajectory
    finally:
        drive_env.close()


@pytest.mark.parametrize(
    "throttle_norm,brake_norm",
    [(HELD_THROTTLE_NORM, 0.0), (CONTRACT_PEDAL_RELEASED_FRACTION, 0.0), (0.0, CONTRACT_PEDAL_RELEASED_FRACTION)],
    ids=["throttle_held", "throttle_at_threshold", "brake_at_threshold"],
)
def test_full_with_a_pedal_pressed_drives_exactly_like_off(throttle_norm, brake_norm):
    off = human_trajectory(binding.ASSIST_MODE_OFF, throttle_norm, brake_norm)
    full = human_trajectory(binding.ASSIST_MODE_FULL, throttle_norm, brake_norm)
    assert full == off


@pytest.mark.parametrize("mode", [binding.ASSIST_MODE_DISPLAY, binding.ASSIST_MODE_NUDGE])
def test_display_and_nudge_never_touch_the_speed_with_feet_off(mode):
    assert human_trajectory(mode, 0.0, 0.0) == human_trajectory(binding.ASSIST_MODE_OFF, 0.0, 0.0)


def test_full_with_feet_off_differs_from_off():
    assert human_trajectory(binding.ASSIST_MODE_FULL, 0.0, 0.0) != human_trajectory(binding.ASSIST_MODE_OFF, 0.0, 0.0)


def test_full_in_reverse_with_feet_off_is_the_driver():
    full = human_trajectory(binding.ASSIST_MODE_FULL, 0.0, 0.0, reverse=True)
    assert {gear for *_, gear in full} == {binding.GEAR_REVERSE}
    assert full == human_trajectory(binding.ASSIST_MODE_OFF, 0.0, 0.0, reverse=True)


def test_blend_at_one_drives_the_clipped_reference_accel_after_the_ramp(env):
    env.set_assist_mode(binding.ASSIST_MODE_BLEND, 1.0)
    ramp_steps = math.ceil(CONTRACT_RAMP_SECONDS / POLICY_DT_SECONDS)
    for step_idx in range(ramp_steps + 10):
        state = pedal_step(env, HELD_THROTTLE_NORM, 0.0)
        if step_idx + 1 < ramp_steps:
            continue
        assert state["blend_weight"] == 1.0
        assert accel_follows_the_reference(env, state)


POLICY_BRAKE_ACTION = (-1.0, 0.0)
POLICY_PULL_AWAY_ACTION = (1.0, 0.0)
STOP_STEP_COUNT = 30


def longitudinal_actions(env, action):
    actions = np.zeros_like(env.actions)
    actions[HUMAN_ACTION_ROW] = action
    return actions


def brake_to_rest(env):
    for _ in range(STOP_STEP_COUNT):
        env.set_human_pedals(0.0, 0.0, 1.0, 0.0)
        env.step(longitudinal_actions(env, POLICY_BRAKE_ACTION))
        if env.get_human_state()["speed_mps"] == 0.0:
            return
    pytest.fail("the brake pedal never brought the car to rest")


@pytest.mark.parametrize("mode", [binding.ASSIST_MODE_FULL, binding.ASSIST_MODE_BLEND])
def test_a_braking_policy_holds_a_car_at_rest_and_never_reverses_it(env, mode):
    brake_to_rest(env)
    env.set_assist_mode(mode, 1.0)
    braked_at_rest_count = 0
    for _ in range(LONG_STEP_COUNT):
        speed_before_mps = env.get_human_state()["speed_mps"]
        env.set_human_pedals(0.0, 0.0, 0.0, 0.0)
        env.step(longitudinal_actions(env, POLICY_BRAKE_ACTION))
        human = env.get_human_state()
        assert human["gear"] >= 1
        assert human["speed_mps"] >= 0.0
        state = env.get_assist_state()
        braked_at_rest_count += speed_before_mps == 0.0 and state["reference_accel_mps2"] < 0.0
    assert braked_at_rest_count > 0
    if mode == binding.ASSIST_MODE_FULL:
        assert env.get_human_state()["speed_mps"] == 0.0


@pytest.mark.parametrize("mode", [binding.ASSIST_MODE_FULL, binding.ASSIST_MODE_BLEND])
def test_a_pulling_away_policy_still_accelerates_from_rest(env, mode):
    brake_to_rest(env)
    env.set_assist_mode(mode, 1.0)
    for _ in range(LONG_STEP_COUNT):
        env.set_human_pedals(0.0, 0.0, 0.0, 0.0)
        env.step(longitudinal_actions(env, POLICY_PULL_AWAY_ACTION))
    assert env.get_human_state()["speed_mps"] > 1.0


@pytest.mark.parametrize("mode", [binding.ASSIST_MODE_FULL, binding.ASSIST_MODE_BLEND])
def test_a_creeping_car_whose_policy_chose_to_stop_comes_to_rest_and_holds(env, mode):
    env.set_assist_mode(mode, 1.0)
    speeds = []
    for step_idx in range(CREEP_STEP_COUNT + LONG_STEP_COUNT + HOLD_STEP_COUNT):
        throttle_norm = CREEP_THROTTLE_NORM if step_idx < CREEP_STEP_COUNT else 0.0
        env.set_human_pedals(0.0, throttle_norm, 0.0, 0.0)
        env.step(longitudinal_actions(env, POLICY_BRAKE_ACTION))
        speeds.append(env.get_human_state()["speed_mps"])
    assert min(speeds) >= 0.0
    assert speeds[-HOLD_STEP_COUNT:] == [0.0] * HOLD_STEP_COUNT


def test_reference_speed_is_reported_and_drives_the_stop_rule(env):
    env.set_assist_mode(binding.ASSIST_MODE_FULL, 1.0)
    saw_stop = False
    for step_idx in range(CREEP_STEP_COUNT + LONG_STEP_COUNT):
        speed_before_mps = env.get_human_state()["speed_mps"]
        throttle_norm = CREEP_THROTTLE_NORM if step_idx < CREEP_STEP_COUNT else 0.0
        env.set_human_pedals(0.0, throttle_norm, 0.0, 0.0)
        env.step(longitudinal_actions(env, POLICY_BRAKE_ACTION))
        state = env.get_assist_state()
        human = env.get_human_state()
        assert math.isfinite(state["reference_speed_mps"])
        if throttle_norm > 0.0 or human["speed_mps"] == 0.0:
            continue
        assert human["accel_long_mps2"] == pytest.approx(assisted_reference(state, speed_before_mps), abs=1e-6)
        saw_stop |= state["reference_speed_mps"] == 0.0
    assert saw_stop or env.get_human_state()["speed_mps"] == 0.0
