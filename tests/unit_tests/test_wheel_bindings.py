"""Python-facing surface of the human car: exported constants and the setter/getter bindings.

The dynamics themselves are covered in C (tests/drive/test_drive_human_control.c and
test_drive_human_powertrain.c). What can only be checked here is that the binding exports the
constants the Python side now reads instead of duplicating, that invalid input raises rather
than being silently clipped, and that get_human_state reports the powertrain.
"""

import math
import os
import re

import numpy as np

import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")

SCENARIO_LENGTH = 400
HUMAN_STATE_KEYS = {
    "agent_idx",
    "sim_x",
    "sim_y",
    "sim_heading",
    "speed_mps",
    "steering_angle_rad",
    "accel_long_mps2",
    "accel_lat_mps2",
    "yaw_rate_rps",
    "collision",
    "offroad",
    "grip_mode",
    "stopped",
    "removed",
    "gear",
    "gearbox_mode",
    "engine_rpm",
    "engine_stalled",
    "max_speed_mps",
    "lane_speed_limit_mps",
    "overspeed_tolerance_mps",
    "goal_x",
    "goal_y",
    "goal_idx",
}


def _make_env(reset_accel_on_stop):
    drive_env = Drive(
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
        reset_accel_on_stop=reset_accel_on_stop,
    )
    drive_env.reset()
    return drive_env


@pytest.fixture
def env():
    """Matches pufferlib/config/puffer_drive.yaml, which sets reset_accel_on_stop: true."""
    drive_env = _make_env(reset_accel_on_stop=True)
    yield drive_env
    drive_env.close()


@pytest.fixture
def default_flag_env():
    drive_env = _make_env(reset_accel_on_stop=False)
    yield drive_env
    drive_env.close()


def test_controller_and_mode_constants_are_exported():
    assert binding.CONTROLLER_HUMAN == 4
    assert binding.GRIP_MODE_KINEMATIC == 0
    assert binding.GRIP_MODE_REALISTIC == 1
    assert binding.GEARBOX_MODE_MANUAL == 0
    assert binding.GEARBOX_MODE_AUTOMATIC == 1
    assert binding.GEAR_REVERSE == -1
    assert binding.GEAR_NEUTRAL == 0
    assert binding.GEARBOX_FORWARD_GEAR_COUNT == 6


def test_steering_constants_are_self_consistent():
    half_lock_rad = math.radians(binding.HUMAN_WHEEL_RANGE_DEG / 2.0)
    assert half_lock_rad / binding.HUMAN_STEERING_RATIO == pytest.approx(binding.HUMAN_STEERING_LIMIT_RAD, abs=1e-6)
    assert 14.0 < binding.HUMAN_STEERING_RATIO < 18.0
    assert binding.HUMAN_TIRE_GRIP_MPS2 > 0.0


def test_engine_constants_are_ordered():
    assert binding.ENGINE_IDLE_RPM < binding.ENGINE_REDLINE_RPM
    assert 0.0 < binding.CLUTCH_ENGAGE_FRACTION < 1.0


def test_no_human_selected_reports_no_state(env):
    assert env.get_human_state() is None


def test_selecting_a_human_reports_the_full_powertrain_state(env):
    env.select_human_agent(0)
    state = env.get_human_state()
    assert state is not None
    assert set(state) == HUMAN_STATE_KEYS
    assert state["agent_idx"] == 0
    assert state["gear"] == binding.GEAR_NEUTRAL
    assert state["gearbox_mode"] == binding.GEARBOX_MODE_AUTOMATIC
    assert state["engine_rpm"] == pytest.approx(binding.ENGINE_IDLE_RPM)
    assert state["engine_stalled"] == 0
    assert state["max_speed_mps"] > 0.0

    env.select_human_agent(-1)
    assert env.get_human_state() is None


def test_gear_and_gearbox_setters_round_trip(env):
    env.select_human_agent(0)
    for gear in (binding.GEAR_REVERSE, binding.GEAR_NEUTRAL, 1, binding.GEARBOX_FORWARD_GEAR_COUNT):
        env.set_human_gear(gear)
        assert env.get_human_state()["gear"] == gear
    for mode in (binding.GEARBOX_MODE_MANUAL, binding.GEARBOX_MODE_AUTOMATIC):
        env.set_human_gearbox_mode(mode)
        assert env.get_human_state()["gearbox_mode"] == mode


@pytest.mark.parametrize("gear", [-2, 7, 100])
def test_out_of_range_gear_raises(env, gear):
    env.select_human_agent(0)
    with pytest.raises(ValueError):
        env.set_human_gear(gear)
    assert env.get_human_state()["gear"] == binding.GEAR_NEUTRAL


@pytest.mark.parametrize("mode", [-1, 2, 99])
def test_unknown_gearbox_mode_raises(env, mode):
    env.select_human_agent(0)
    with pytest.raises(ValueError):
        env.set_human_gearbox_mode(mode)


@pytest.mark.parametrize("mode", [-1, 2, 99])
def test_unknown_grip_mode_raises(env, mode):
    with pytest.raises(ValueError):
        env.set_human_grip_mode(mode)


@pytest.mark.parametrize(
    "pedals",
    [
        (float("nan"), 0.0, 0.0, 0.0),
        (0.0, float("nan"), 0.0, 0.0),
        (0.0, 0.0, float("inf"), 0.0),
        (0.0, 0.0, 0.0, float("-inf")),
    ],
)
def test_non_finite_pedals_raise(env, pedals):
    env.select_human_agent(0)
    with pytest.raises(ValueError):
        env.set_human_pedals(*pedals)


@pytest.mark.parametrize("shift_request", [-2, 2, 10])
def test_out_of_range_shift_request_raises(env, shift_request):
    env.select_human_agent(0)
    with pytest.raises(ValueError):
        env.set_human_pedals(0.0, 0.0, 0.0, 1.0, shift_request)


def test_non_finite_human_input_raises(env):
    env.select_human_agent(0)
    with pytest.raises(ValueError):
        env.set_human_input(float("nan"), 0.0)
    with pytest.raises(ValueError):
        env.set_human_input(0.0, float("inf"))


def test_out_of_range_agent_index_raises(env):
    with pytest.raises(ValueError):
        env.select_human_agent(10_000)
    assert env.get_human_state() is None


@pytest.mark.parametrize("rejected_speed_mps", [0.0, 1000.0, float("nan")])
def test_max_speed_outside_the_trained_band_raises(env, rejected_speed_mps):
    env.select_human_agent(0)
    band_top_mps = env.get_human_state()["max_speed_mps"]
    with pytest.raises(ValueError) as excinfo:
        env.set_human_max_speed(rejected_speed_mps)

    message = str(excinfo.value)
    assert f"{rejected_speed_mps:.3f}" in message
    band = re.search(r"\[(-?[\d.]+), (-?[\d.]+)\]", message)
    assert band is not None, message
    band_bottom_mps, reported_top_mps = float(band.group(1)), float(band.group(2))
    assert band_bottom_mps < reported_top_mps
    assert reported_top_mps == pytest.approx(band_top_mps, abs=1e-3)
    assert env.get_human_state()["max_speed_mps"] == pytest.approx(band_top_mps)


def test_max_speed_inside_the_trained_band_is_accepted(env):
    env.select_human_agent(0)
    band_top_mps = env.get_human_state()["max_speed_mps"]
    env.set_human_max_speed(band_top_mps)
    assert env.get_human_state()["max_speed_mps"] == pytest.approx(band_top_mps)


def test_pedals_drive_the_car_through_the_binding(env):
    env.select_human_agent(0)
    env.set_human_gearbox_mode(binding.GEARBOX_MODE_AUTOMATIC)
    env.set_human_pedals(0.0, 1.0, 0.0, 0.0)
    for _ in range(15):
        env.step(env.action_space.sample() * 0)
    state = env.get_human_state()
    assert state["speed_mps"] > 0.0
    assert state["gear"] >= 1
    assert state["engine_rpm"] >= binding.ENGINE_IDLE_RPM
    assert state["engine_stalled"] == 0


def test_brake_holds_the_car_through_the_binding(env):
    env.select_human_agent(0)
    env.set_human_gearbox_mode(binding.GEARBOX_MODE_AUTOMATIC)
    env.set_human_pedals(0.0, 0.0, 1.0, 0.0)
    for _ in range(30):
        env.step(env.action_space.sample() * 0)
    assert env.get_human_state()["speed_mps"] == pytest.approx(0.0, abs=1e-6)


# The human stop no longer depends on reset_accel_on_stop, whose Python default (False) still
# disagrees with puffer_drive.yaml (true).
def test_brake_settles_exactly_with_the_default_stop_accel_flag(default_flag_env):
    default_flag_env.select_human_agent(0)
    default_flag_env.set_human_gearbox_mode(binding.GEARBOX_MODE_AUTOMATIC)
    default_flag_env.set_human_pedals(0.0, 0.0, 1.0, 0.0)
    for _ in range(40):
        default_flag_env.step(default_flag_env.action_space.sample() * 0)
        assert default_flag_env.get_human_state()["speed_mps"] >= 0.0
    assert default_flag_env.get_human_state()["speed_mps"] == pytest.approx(0.0, abs=1e-6)


def test_engine_restart_through_the_binding(env):
    env.select_human_agent(0)
    env.set_human_gearbox_mode(binding.GEARBOX_MODE_MANUAL)
    env.set_human_gear(1)
    env.set_human_pedals(0.0, 1.0, 0.0, 0.0)
    env.step(env.action_space.sample() * 0)
    stalled_state = env.get_human_state()
    assert stalled_state["engine_stalled"] == 1
    assert stalled_state["engine_rpm"] == pytest.approx(0.0)

    env.restart_human_engine()
    restarted_state = env.get_human_state()
    assert restarted_state["engine_stalled"] == 0
    assert restarted_state["engine_rpm"] == pytest.approx(binding.ENGINE_IDLE_RPM)

    env.set_human_pedals(0.0, 0.8, 0.0, 0.3)
    for _ in range(20):
        env.step(env.action_space.sample() * 0)
    driving_state = env.get_human_state()
    assert driving_state["engine_stalled"] == 0
    assert driving_state["speed_mps"] > 0.0


def test_direct_input_takes_the_car_back_from_the_pedals(env):
    env.select_human_agent(0)
    env.set_human_gearbox_mode(binding.GEARBOX_MODE_AUTOMATIC)
    env.set_human_pedals(0.0, 1.0, 0.0, 0.0)
    for _ in range(10):
        env.step(env.action_space.sample() * 0)
    assert env.get_human_state()["speed_mps"] > 0.0

    env.set_human_input(0.0, -3.0)
    env.step(env.action_space.sample() * 0)
    assert env.get_human_state()["accel_long_mps2"] == pytest.approx(-3.0, abs=1e-5)


# Full lock and full throttle leave the road within this many policy steps on every test map.
OFFROAD_MAX_STEPS = 120
FULL_LOCK_STEERING_NORM = 1.0
FULL_THROTTLE = 1.0


def test_driving_off_the_road_reports_the_offroad_flag(env):
    """The force-feedback rumble has no other source for this, so the binding must export it."""
    env.select_human_agent(0)
    env.set_human_gearbox_mode(binding.GEARBOX_MODE_AUTOMATIC)
    assert env.get_human_state()["offroad"] == 0.0

    env.set_human_pedals(FULL_LOCK_STEERING_NORM, FULL_THROTTLE, 0.0, 0.0)
    offroad_state = None
    for _ in range(OFFROAD_MAX_STEPS):
        env.step(env.action_space.sample() * 0)
        state = env.get_human_state()
        if state["offroad"] > 0.0:
            offroad_state = state
            break

    assert offroad_state is not None, "full lock and full throttle never took the car off the road"
    assert offroad_state["offroad"] == 1.0
    # compute_metrics returns as soon as it flags offroad, so the two are mutually exclusive.
    assert offroad_state["collision"] == 0.0


def test_collision_and_offroad_are_per_step_flags_not_latched(env):
    """The force-feedback loop jolts on a RISING collision edge, so a flag that stayed set once
    the car was clear would mean no second jolt for the rest of the drive."""
    env.select_human_agent(0)
    env.set_human_gearbox_mode(binding.GEARBOX_MODE_AUTOMATIC)
    env.set_human_pedals(FULL_LOCK_STEERING_NORM, FULL_THROTTLE, 0.0, 0.0)
    collisions = []
    for _ in range(OFFROAD_MAX_STEPS):
        env.step(env.action_space.sample() * 0)
        collisions.append(env.get_human_state()["collision"] > 0.0)

    rising_edges = sum(1 for before, after in zip(collisions, collisions[1:]) if after and not before)
    assert any(collisions), "the car never hit anything"
    assert not all(collisions)
    assert rising_edges >= 2, "a per-step flag must be able to rise more than once"


# The sim holds every logged track, not just the one policy agent, so size generously.
RENDER_POSE_ARRAY_SIZE = 1024


def _render_pose_arrays(size):
    return (
        np.zeros(size, dtype=np.float32),
        np.zeros(size, dtype=np.float32),
        np.zeros(size, dtype=np.float32),
    )


def test_substep_mode_constants_are_exported():
    assert binding.SUBSTEP_MODE_ALL == 0
    assert binding.SUBSTEP_MODE_HUMAN_ONLY == 1


@pytest.mark.parametrize(
    "substep_mode,substeps",
    [(binding.SUBSTEP_MODE_ALL, 1), (binding.SUBSTEP_MODE_ALL, 18), (binding.SUBSTEP_MODE_HUMAN_ONLY, 18)],
)
def test_set_substepping_accepts_valid_settings(env, substep_mode, substeps):
    env.set_substepping(substep_mode, substeps)


@pytest.mark.parametrize("substep_mode,substeps", [(-1, 18), (2, 18), (99, 18), (0, 0), (0, -5)])
def test_set_substepping_rejects_invalid_settings(env, substep_mode, substeps):
    with pytest.raises(ValueError):
        env.set_substepping(substep_mode, substeps)


def test_substeps_then_step_drives_the_human(env):
    env.select_human_agent(0)
    env.set_substepping(binding.SUBSTEP_MODE_HUMAN_ONLY, 18)
    env.set_human_gearbox_mode(binding.GEARBOX_MODE_AUTOMATIC)
    env.set_human_pedals(0.0, 1.0, 0.0, 0.0)
    for _ in range(5):
        for _ in range(18):
            env.substep()
        env.step(env.action_space.sample() * 0)
    assert env.get_human_state()["speed_mps"] > 0.0


def test_render_poses_rejects_undersized_arrays(env):
    x, y, heading = _render_pose_arrays(1)
    with pytest.raises(ValueError):
        env.get_render_poses(0.0, x, y, heading)


def test_render_poses_rejects_non_arrays(env):
    x, y, heading = _render_pose_arrays(RENDER_POSE_ARRAY_SIZE)
    with pytest.raises(TypeError):
        env.get_render_poses(0.0, [0.0] * RENDER_POSE_ARRAY_SIZE, y, heading)


def test_render_poses_are_finite_and_leave_state_alone(env):
    env.select_human_agent(0)
    env.set_substepping(binding.SUBSTEP_MODE_HUMAN_ONLY, 18)
    env.set_human_gearbox_mode(binding.GEARBOX_MODE_AUTOMATIC)
    env.set_human_pedals(0.0, 1.0, 0.0, 0.0)
    for _ in range(5):
        for _ in range(18):
            env.substep()
        env.step(env.action_space.sample() * 0)

    x, y, heading = _render_pose_arrays(RENDER_POSE_ARRAY_SIZE)
    before = env.get_human_state()
    for elapsed_seconds in (0.0, 0.1, 0.3):
        env.get_render_poses(elapsed_seconds, x, y, heading)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))
        assert np.all(np.isfinite(heading))
    after = env.get_human_state()
    assert after == before
