"""The wheel demo loop's look-ahead, FULL steering presets and the assist wheel target, with no wheel attached.

Driver-facing angles are positive RIGHT; the sim is positive LEFT, so every sim angle is negated on the way in.
The pure parts run on a bare WheelDemoLoop over a fake env; the look-ahead itself runs on the real checkpoint.
"""

import math
import os
from collections import deque

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo import loop as loop_module
from pufferlib.wheel_demo.loop import (
    ASSIST_WHEEL_DEADBAND_RAD,
    ASSIST_WHEEL_HOLD_SPEED_MPS,
    ASSIST_WHEEL_SMOOTHING_SECONDS,
    FULL_STEERING_PRESETS,
    LOOKAHEAD_PHYSICS_HZ,
    DemoConfig,
    WheelDemoLoop,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "weights", "k_scaled_0050_1000", "final_model.pt")
CARLA_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "carla")

POLICY_DT_SECONDS = 0.3
SUBSTEPS_PER_POLICY_STEP = 18
LOOKAHEAD_DT_SECONDS = 1.0 / LOOKAHEAD_PHYSICS_HZ
PIECES_PER_POLICY_STEP = round(POLICY_DT_SECONDS / LOOKAHEAD_DT_SECONDS)
CUSTOM_WINDOW_SECONDS = 1.2
MOVING_SPEED_MPS = 8.0
POLICY_SIM_RAD = 0.1
WHEEL_GAP_RAD = 0.02
NEW_PLAN_RAD = 0.2
PREVIOUS_PLAN_RAD = -0.1
RECENT_CAR_RAD = 0.05
REAL_LOOP_AGENT_COUNT = 8
REAL_LOOP_STEP_COUNT = 6
SYNTHETIC_STEERING_NORM = 0.2

requires_checkpoint = pytest.mark.skipif(not os.path.isfile(CHECKPOINT_PATH), reason="demo checkpoint not present")


class FakeLoopEnv:
    """What the loop's wheel-target helpers read: assist state, human state, and the FULL dynamics setter."""

    def __init__(self):
        self.full_assist_dynamics_calls = []
        self.assist = {"mode": binding.ASSIST_MODE_FULL, "reference_valid": 1, "reference_steering_rad": POLICY_SIM_RAD}
        self.human = {"speed_mps": MOVING_SPEED_MPS, "steering_angle_rad": 0.0}

    def set_full_assist_dynamics(self, full_assist_dynamics):
        self.full_assist_dynamics_calls.append(full_assist_dynamics)

    def get_assist_state(self):
        return dict(self.assist)

    def get_human_state(self):
        return dict(self.human)


def bare_loop(full_assist_dynamics=binding.FULL_ASSIST_DYNAMICS_SET_STEERING, window_seconds=0.6, centred=False):
    loop = WheelDemoLoop.__new__(WheelDemoLoop)
    loop.config = DemoConfig(checkpoint_path="unused", lookahead_centred=centred)
    loop.env = FakeLoopEnv()
    loop.dt = POLICY_DT_SECONDS
    loop.substeps_per_policy_step = SUBSTEPS_PER_POLICY_STEP
    loop.substep_dt = POLICY_DT_SECONDS / SUBSTEPS_PER_POLICY_STEP
    loop.lookahead_dt = LOOKAHEAD_DT_SECONDS
    loop._assist_wheel_target_rad = None
    loop.set_full_steering(full_assist_dynamics, window_seconds)
    return loop


def test_presets_and_the_default_config_are_the_car_preset():
    assert FULL_STEERING_PRESETS == {
        "car": (binding.FULL_ASSIST_DYNAMICS_SET_STEERING, 0.6),
        "wheel": (binding.FULL_ASSIST_DYNAMICS_JERK_SPEED, 0.9),
    }
    config = DemoConfig(checkpoint_path="unused")
    assert (config.full_assist_dynamics, config.lookahead_window_seconds) == FULL_STEERING_PRESETS["car"]
    assert config.wheel_target == "lookahead"
    assert config.action_selection == "mode"
    assert loop_module.WHEEL_TARGETS == ("lookahead", "filtered", "car")


@pytest.mark.parametrize("centred", [False, True])
@pytest.mark.parametrize("preset", sorted(FULL_STEERING_PRESETS))
def test_set_full_steering_sizes_the_look_ahead(preset, centred):
    full_assist_dynamics, window_seconds = FULL_STEERING_PRESETS[preset]
    loop = bare_loop(full_assist_dynamics, window_seconds, centred)
    ahead_seconds = window_seconds * (0.5 if centred else 1.0)
    assert loop.env.full_assist_dynamics_calls == [full_assist_dynamics]
    assert loop.lookahead_steers_car == (full_assist_dynamics == binding.FULL_ASSIST_DYNAMICS_SET_STEERING)
    assert loop.lookahead_window_count == round(ahead_seconds / LOOKAHEAD_DT_SECONDS)
    assert loop.lookahead_policy_steps == math.ceil(round((2.0 * POLICY_DT_SECONDS + ahead_seconds) / POLICY_DT_SECONDS, 9))
    expected_past_count = round(0.5 * window_seconds / loop.substep_dt) if centred else 1
    assert loop._recent_car_steering_rad.maxlen == expected_past_count
    assert (loop.config.full_assist_dynamics, loop.config.lookahead_window_seconds) == (full_assist_dynamics, window_seconds)
    assert loop.full_steering == preset


def test_set_full_steering_drops_drives_of_the_old_length():
    loop = bare_loop()
    loop._lookahead_rad = np.zeros(3)
    loop._previous_lookahead_rad = np.zeros(3)
    loop._recent_car_steering_rad.append(RECENT_CAR_RAD)
    loop.set_full_steering(*FULL_STEERING_PRESETS["wheel"])
    assert loop._lookahead_rad is None
    assert loop._previous_lookahead_rad is None
    assert len(loop._recent_car_steering_rad) == 0


@pytest.mark.parametrize(
    "full_assist_dynamics,window_seconds",
    [(binding.FULL_ASSIST_DYNAMICS_JERK, 0.6), (binding.FULL_ASSIST_DYNAMICS_SET_STEERING, CUSTOM_WINDOW_SECONDS)],
)
def test_anything_but_a_preset_is_custom(full_assist_dynamics, window_seconds):
    assert bare_loop(full_assist_dynamics, window_seconds).full_steering == "custom"


def drive_samples(loop, seed=3):
    sample_count = loop.lookahead_policy_steps * PIECES_PER_POLICY_STEP
    return np.random.default_rng(seed).uniform(-0.3, 0.3, sample_count)


@pytest.mark.parametrize("start_sample_idx", [0, 1, 7, 29])
def test_window_mean_on_samples_is_the_plain_mean(start_sample_idx):
    loop = bare_loop()
    samples = drive_samples(loop)
    count = loop.lookahead_window_count
    mean_rad = loop._window_mean_rad(samples, start_sample_idx * LOOKAHEAD_DT_SECONDS)
    assert mean_rad == pytest.approx(np.mean(samples[start_sample_idx : start_sample_idx + count]), abs=1e-12)


@pytest.mark.parametrize("start_sample_idx", [0, 4, 20])
@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.8])
def test_window_mean_between_samples_weights_the_partial_ends(start_sample_idx, fraction):
    loop = bare_loop()
    samples = drive_samples(loop)
    count = loop.lookahead_window_count
    start_s = (start_sample_idx + fraction) * LOOKAHEAD_DT_SECONDS
    inner = samples[start_sample_idx + 1 : start_sample_idx + count]
    expected = ((1.0 - fraction) * samples[start_sample_idx] + inner.sum()
                + fraction * samples[start_sample_idx + count]) / count
    assert loop._window_mean_rad(samples, start_s) == pytest.approx(expected, abs=1e-12)


def test_window_mean_slides_without_ripple_on_a_steady_drive():
    loop = bare_loop()
    samples = np.full(loop.lookahead_policy_steps * PIECES_PER_POLICY_STEP, RECENT_CAR_RAD)
    for substep_idx in range(SUBSTEPS_PER_POLICY_STEP):
        start_s = (substep_idx + 1) * loop.substep_dt
        assert loop._window_mean_rad(samples, start_s) == pytest.approx(RECENT_CAR_RAD, abs=1e-12)


def test_window_mean_asserts_the_drive_covers_the_window():
    loop = bare_loop()
    short = np.zeros(loop.lookahead_window_count - 1)
    with pytest.raises(AssertionError):
        loop._window_mean_rad(short, 0.0)


def plan(loop, value_rad, step_count):
    return np.full(step_count * PIECES_PER_POLICY_STEP, value_rad)


def test_lookahead_mean_is_none_without_a_drive():
    loop = bare_loop()
    loop._lookahead_rad = None
    assert loop._lookahead_mean_rad(0) is None


@pytest.mark.parametrize("substep_idx", [0, 5, SUBSTEPS_PER_POLICY_STEP - 1])
def test_lookahead_mean_crossfades_from_the_previous_plan(substep_idx):
    loop = bare_loop()
    loop._lookahead_rad = plan(loop, NEW_PLAN_RAD, loop.lookahead_policy_steps)
    loop._previous_lookahead_rad = plan(loop, PREVIOUS_PLAN_RAD, loop.lookahead_policy_steps - 1)
    new_plan_weight = (substep_idx + 1) / SUBSTEPS_PER_POLICY_STEP
    expected = new_plan_weight * NEW_PLAN_RAD + (1.0 - new_plan_weight) * PREVIOUS_PLAN_RAD
    assert loop._lookahead_mean_rad(substep_idx) == pytest.approx(expected)
    loop._previous_lookahead_rad = None
    assert loop._lookahead_mean_rad(substep_idx) == pytest.approx(NEW_PLAN_RAD)


def test_centred_mean_averages_equally_with_the_cars_recent_steering():
    loop = bare_loop(centred=True)
    loop._lookahead_rad = plan(loop, NEW_PLAN_RAD, loop.lookahead_policy_steps)
    loop._previous_lookahead_rad = None
    assert loop._lookahead_mean_rad(0) == pytest.approx(NEW_PLAN_RAD)
    loop._recent_car_steering_rad.extend([RECENT_CAR_RAD, 3.0 * RECENT_CAR_RAD])
    assert loop._lookahead_mean_rad(0) == pytest.approx(0.5 * NEW_PLAN_RAD + 0.5 * 2.0 * RECENT_CAR_RAD)


def expected_filtered_step(target_rad, policy_rad, substep_dt):
    gap_rad = policy_rad - target_rad
    deadband_weight = gap_rad**2 / (gap_rad**2 + ASSIST_WHEEL_DEADBAND_RAD**2)
    smoothing = 1.0 - math.exp(-substep_dt / ASSIST_WHEEL_SMOOTHING_SECONDS)
    return target_rad + deadband_weight * smoothing * gap_rad


def test_filtered_target_starts_on_the_policy_in_driver_sign():
    loop = bare_loop()
    loop._update_assist_wheel_target()
    assert loop._assist_wheel_target_rad == -POLICY_SIM_RAD


@pytest.mark.parametrize("gap_multiple", [0.25, 1.0, 4.0, -2.0])
def test_filtered_target_follows_through_the_soft_deadband(gap_multiple):
    loop = bare_loop()
    loop._update_assist_wheel_target()
    start_rad = loop._assist_wheel_target_rad
    new_policy_sim_rad = POLICY_SIM_RAD - gap_multiple * ASSIST_WHEEL_DEADBAND_RAD
    loop.env.assist["reference_steering_rad"] = new_policy_sim_rad
    loop._update_assist_wheel_target()
    expected = expected_filtered_step(start_rad, -new_policy_sim_rad, loop.substep_dt)
    assert loop._assist_wheel_target_rad == pytest.approx(expected, abs=1e-12)


def test_a_gap_of_one_deadband_moves_at_half_the_smoothing_rate():
    loop = bare_loop()
    loop._update_assist_wheel_target()
    loop.env.assist["reference_steering_rad"] = POLICY_SIM_RAD - ASSIST_WHEEL_DEADBAND_RAD
    loop._update_assist_wheel_target()
    smoothing = 1.0 - math.exp(-loop.substep_dt / ASSIST_WHEEL_SMOOTHING_SECONDS)
    moved_rad = loop._assist_wheel_target_rad + POLICY_SIM_RAD
    assert moved_rad == pytest.approx(0.5 * smoothing * ASSIST_WHEEL_DEADBAND_RAD)


@pytest.mark.parametrize("speed_mps", [0.0, 0.5 * ASSIST_WHEEL_HOLD_SPEED_MPS, -0.5 * ASSIST_WHEEL_HOLD_SPEED_MPS])
def test_filtered_target_holds_while_nearly_stopped(speed_mps):
    loop = bare_loop()
    loop._update_assist_wheel_target()
    loop.env.human["speed_mps"] = speed_mps
    loop.env.assist["reference_steering_rad"] = POLICY_SIM_RAD + 0.3
    loop._update_assist_wheel_target()
    assert loop._assist_wheel_target_rad == -POLICY_SIM_RAD


NO_WHEEL_ASSIST = [
    {"mode": binding.ASSIST_MODE_OFF, "reference_valid": 1},
    {"mode": binding.ASSIST_MODE_DISPLAY, "reference_valid": 1},
    {"mode": binding.ASSIST_MODE_BLEND, "reference_valid": 1},
    {"mode": binding.ASSIST_MODE_FULL, "reference_valid": 0},
    {"mode": binding.ASSIST_MODE_NUDGE, "reference_valid": 0},
]


@pytest.mark.parametrize("assist", NO_WHEEL_ASSIST)
@pytest.mark.parametrize("update_name", ["_update_assist_wheel_target", "_update_car_wheel_target"])
def test_no_target_when_the_assist_does_not_steer_the_wheel(assist, update_name):
    loop = bare_loop()
    loop._assist_wheel_target_rad = 0.1
    loop.env.assist.update(assist)
    getattr(loop, update_name)()
    assert loop._assist_wheel_target_rad is None


@pytest.mark.parametrize("mode", [binding.ASSIST_MODE_NUDGE, binding.ASSIST_MODE_FULL])
def test_car_target_is_the_cars_own_angle_in_driver_sign(mode):
    loop = bare_loop()
    loop.env.assist["mode"] = mode
    loop.env.human["steering_angle_rad"] = 0.17
    loop._update_car_wheel_target()
    assert loop._assist_wheel_target_rad == -0.17


def test_published_motion_is_target_rate_and_time_together():
    loop = bare_loop()
    loop._assist_wheel_target_rad = 0.3
    loop._publish_wheel_target_motion(0.25)
    target_rad, rate_rad_s, published_s = loop._assist_wheel_target_motion
    assert target_rad == 0.3
    assert rate_rad_s == pytest.approx(0.05 / loop.substep_dt)
    assert published_s > 0.0
    loop._publish_wheel_target_motion(None)
    assert loop._assist_wheel_target_motion[:2] == (0.3, 0.0)
    loop._assist_wheel_target_rad = None
    loop._publish_wheel_target_motion(0.25)
    assert loop._assist_wheel_target_motion[:2] == (None, 0.0)


def real_loop(**config_overrides):
    from pufferlib.wheel_demo.input.trace import SyntheticWheelInput

    config = DemoConfig(
        checkpoint_path=CHECKPOINT_PATH,
        town="Town01",
        map_dir=CARLA_MAP_DIR,
        num_agents=REAL_LOOP_AGENT_COUNT,
        seed=0,
        **config_overrides,
    )
    return WheelDemoLoop(config, SyntheticWheelInput())


def env_snapshot(env):
    return dict(env.get_human_state()), env.observations.copy(), env.get_assist_state()


@requires_checkpoint
def test_look_ahead_leaves_the_env_exactly_as_it_was():
    loop = real_loop()
    try:
        loop.env.set_assist_mode(binding.ASSIST_MODE_FULL, 1.0)
        loop.step_once()
        human_before, observations_before, assist_before = env_snapshot(loop.env)
        actions = loop.traffic.act(loop.observations)
        loop._look_ahead(actions)
        human_after, observations_after, assist_after = env_snapshot(loop.env)
        assert human_after == human_before
        np.testing.assert_array_equal(observations_after, observations_before)
        assert assist_after == assist_before
        assert loop._lookahead_rad.shape == (loop.lookahead_policy_steps * PIECES_PER_POLICY_STEP,)
        assert np.all(np.isfinite(loop._lookahead_rad))
    finally:
        loop.close()


@requires_checkpoint
def test_look_ahead_keeps_the_previous_drive_one_policy_step_on():
    loop = real_loop()
    try:
        actions = loop.traffic.act(loop.observations)
        loop._look_ahead(actions)
        first = loop._lookahead_rad.copy()
        loop._look_ahead(actions)
        np.testing.assert_array_equal(loop._previous_lookahead_rad, first[PIECES_PER_POLICY_STEP:])
        np.testing.assert_array_equal(loop._lookahead_rad, first)
    finally:
        loop.close()


@requires_checkpoint
def test_look_ahead_is_skipped_without_a_human_car():
    loop = real_loop()
    try:
        loop.select_human_agent(-1)
        loop._look_ahead(loop.traffic.act(loop.observations))
        assert loop._lookahead_rad is None
        assert loop._previous_lookahead_rad is None
        assert loop._lookahead_mean_rad(0) is None
    finally:
        loop.close()


def human_trajectory(loop, step_count=REAL_LOOP_STEP_COUNT):
    loop.wheel.set_controls(steering_norm=SYNTHETIC_STEERING_NORM, throttle_norm=0.0)
    trajectory = []
    for _ in range(step_count):
        loop.step_once()
        trajectory.append(dict(loop.env.get_human_state()))
    return trajectory


@requires_checkpoint
def test_the_look_ahead_never_changes_the_drive():
    wheel = FULL_STEERING_PRESETS["wheel"]
    with_lookahead = real_loop(wheel_target="lookahead", full_assist_dynamics=wheel[0], lookahead_window_seconds=wheel[1])
    filtered = real_loop(wheel_target="filtered", full_assist_dynamics=wheel[0], lookahead_window_seconds=wheel[1])
    try:
        for loop in (with_lookahead, filtered):
            loop.env.set_assist_mode(binding.ASSIST_MODE_FULL, 1.0)
        assert human_trajectory(with_lookahead) == human_trajectory(filtered)
        assert with_lookahead._lookahead_rad is not None
    finally:
        with_lookahead.close()
        filtered.close()


@requires_checkpoint
def test_set_steering_sets_the_look_ahead_mean_before_every_substep():
    loop = real_loop()
    try:
        assert loop.lookahead_steers_car
        loop.env.set_assist_mode(binding.ASSIST_MODE_FULL, 1.0)
        events = []
        real_set_assist_steering = loop.env.set_assist_steering
        real_substep = loop.env.substep

        def set_assist_steering(steering_rad):
            events.append(("set", steering_rad))
            real_set_assist_steering(steering_rad)

        def substep():
            events.append(("substep",))
            real_substep()

        loop.env.set_assist_steering = set_assist_steering
        loop.env.substep = substep
        for _ in range(2):
            events.clear()
            loop.step_once()
            sets = [event for event in events if event[0] == "set"]
            assert [event[0] for event in events] == ["set", "substep"] * SUBSTEPS_PER_POLICY_STEP
            for substep_idx, (_, steering_rad) in enumerate(sets):
                assert steering_rad == pytest.approx(-loop._lookahead_mean_rad(substep_idx))
            assert loop._assist_wheel_target_rad == -loop.env.get_human_state()["steering_angle_rad"]
    finally:
        loop.close()


class FakeClock:
    def __init__(self):
        self.now_s = 100.0

    def perf_counter(self):
        return self.now_s


@requires_checkpoint
def test_sim_speed_ratio_is_policy_dt_over_the_wall_time_of_the_last_step(monkeypatch):
    loop = real_loop(wheel_target="filtered", full_assist_dynamics=binding.FULL_ASSIST_DYNAMICS_JERK_SPEED)
    try:
        clock = FakeClock()
        monkeypatch.setattr(loop_module, "time", clock)
        loop.step_once()
        assert loop._sim_speed_ratio == 1.0
        clock.now_s += 2.0 * loop.dt
        loop.step_once()
        assert loop._sim_speed_ratio == pytest.approx(0.5)
        clock.now_s += 0.5 * loop.dt
        loop.step_once()
        assert loop._sim_speed_ratio == 1.0
    finally:
        loop.close()


def test_recent_car_steering_is_a_bounded_window():
    loop = bare_loop(centred=True)
    assert isinstance(loop._recent_car_steering_rad, deque)
    for value in range(2 * loop._recent_car_steering_rad.maxlen):
        loop._recent_car_steering_rad.append(float(value))
    assert len(loop._recent_car_steering_rad) == loop._recent_car_steering_rad.maxlen


@requires_checkpoint
@pytest.mark.parametrize("action_selection", ["mode", "sample", "mean"])
def test_frozen_traffic_takes_each_action_selection(action_selection):
    from pufferlib.wheel_demo.policy_host import ACTION_SELECTIONS, FrozenTraffic

    assert ACTION_SELECTIONS == ("mode", "sample", "mean")
    loop = real_loop()
    try:
        traffic = FrozenTraffic(CHECKPOINT_PATH, loop.env, loop.checkpoint_config, "cpu", action_selection)
        assert traffic.action_selection == action_selection
        assert FrozenTraffic(CHECKPOINT_PATH, loop.env, loop.checkpoint_config, "cpu").action_selection == "mode"
        with pytest.raises(ValueError):
            FrozenTraffic(CHECKPOINT_PATH, loop.env, loop.checkpoint_config, "cpu", "argmax")
    finally:
        loop.close()


NUDGE_TARGET_SEPARATION_RAD = 1e-3


def wheel_targets_per_substep(loop, step_count=REAL_LOOP_STEP_COUNT):
    """(target, look-ahead mean, car angle) after every sub-step, all in driver sign."""
    seen = []

    def record(substep_idx, wheel_state):
        car_rad = -loop.env.get_human_state()["steering_angle_rad"]
        seen.append((loop._assist_wheel_target_rad, loop._lookahead_mean_rad(substep_idx), car_rad))

    loop.wheel.set_controls(steering_norm=SYNTHETIC_STEERING_NORM, throttle_norm=0.0)
    for _ in range(step_count):
        loop.step_once(on_substep=record)
    return seen


@requires_checkpoint
def test_nudge_with_the_default_preset_pulls_toward_the_look_ahead_not_the_car():
    loop = real_loop()
    try:
        assert loop.lookahead_steers_car
        loop.env.set_assist_mode(binding.ASSIST_MODE_NUDGE, 1.0)
        seen = wheel_targets_per_substep(loop)
        assert all(target_rad is not None for target_rad, _, _ in seen)
        assert all(target_rad == mean_rad for target_rad, mean_rad, _ in seen)
        assert max(abs(target_rad - car_rad) for target_rad, _, car_rad in seen) > NUDGE_TARGET_SEPARATION_RAD
    finally:
        loop.close()


@requires_checkpoint
def test_full_with_the_default_preset_shows_the_cars_own_angle():
    loop = real_loop()
    try:
        loop.env.set_assist_mode(binding.ASSIST_MODE_FULL, 1.0)
        seen = wheel_targets_per_substep(loop)
        assert all(target_rad == car_rad for target_rad, _, car_rad in seen)
    finally:
        loop.close()


HUMAN_ACTION_ROW_IDX = 0
RESTART_DRIVE_STEP_COUNT = 8


def start_snapshot(loop):
    state = loop.env.get_global_agent_state()
    human = loop.env.get_human_state()
    poses = {name: state[name].copy() for name in ("x", "y", "z", "heading")}
    return poses, (human["goal_x"], human["goal_y"], human["goal_idx"])


def human_conditioning(loop):
    start = loop.env.ego_features
    return loop.observations[HUMAN_ACTION_ROW_IDX, start : start + loop.env.num_reward_coefs].copy()


@requires_checkpoint
def test_restart_puts_every_car_and_goal_back_and_keeps_the_settings():
    from pufferlib.wheel_demo import levels

    loop = real_loop()
    try:
        loop.env.set_assist_mode(binding.ASSIST_MODE_FULL, 1.0)
        levels.apply_level(loop.env, levels.AGGRESSIVE)
        levels.apply_driving_style(loop.env, levels.CALM)
        loop.observations, _ = loop.env.reset(seed=loop.config.seed)
        loop.select_human_agent(loop.human_agent_idx)
        loop.env.set_human_gear(loop.config.start_gear)
        poses_before, goal_before = start_snapshot(loop)
        conditioning_before = human_conditioning(loop)
        assert conditioning_before.size == binding.NUM_REWARD_COEFS
        dynamics_before = loop.config.full_assist_dynamics

        loop.wheel.set_controls(steering_norm=SYNTHETIC_STEERING_NORM, throttle_norm=0.5)
        for _ in range(RESTART_DRIVE_STEP_COUNT):
            loop.step_once()
        assert loop.env.get_human_state()["speed_mps"] > 0.0
        loop._pending_shifts = 2

        loop.restart()
        poses_after, goal_after = start_snapshot(loop)
        for name in poses_before:
            np.testing.assert_array_equal(poses_after[name], poses_before[name])
        assert goal_after == goal_before
        human = loop.env.get_human_state()
        assert human["speed_mps"] == 0.0
        assert human["gear"] == loop.config.start_gear
        assert human["engine_stalled"] == 0
        assert loop.env.get_assist_state()["mode"] == binding.ASSIST_MODE_FULL
        assert loop.config.full_assist_dynamics == dynamics_before
        np.testing.assert_array_equal(human_conditioning(loop), conditioning_before)
        assert loop._pending_shifts == 0
        assert loop._previous_lookahead_rad is None
        assert loop._lookahead_rad is None
        assert loop._last_step_start_s is None
        assert loop._assist_wheel_target_rad is None
        assert len(loop._recent_car_steering_rad) == 0

        for _ in range(2):
            loop.step_once()
        assert loop.env.get_human_state()["speed_mps"] > 0.0
    finally:
        loop.close()


COMMANDED_SIM_RAD = 0.07


def test_blend_target_is_the_cars_commanded_angle_in_driver_sign():
    loop = bare_loop()
    loop.env.assist.update(mode=binding.ASSIST_MODE_BLEND, commanded_steering_rad=COMMANDED_SIM_RAD)
    loop._update_blend_wheel_target()
    assert loop._assist_wheel_target_rad == -COMMANDED_SIM_RAD


def test_blend_target_is_none_without_a_reference():
    loop = bare_loop()
    loop._assist_wheel_target_rad = 0.1
    loop.env.assist.update(mode=binding.ASSIST_MODE_BLEND, reference_valid=0, commanded_steering_rad=COMMANDED_SIM_RAD)
    loop._update_blend_wheel_target()
    assert loop._assist_wheel_target_rad is None


BLEND_MIN_PULL_GAP_RAD = 0.01


def blend_drive(blend_target, step_count=REAL_LOOP_STEP_COUNT):
    """Per sub-step under BLEND with the wheel held straight: (target, commanded angle, wheel angle, force)."""
    from pufferlib.wheel_demo.ffb import compute_force

    loop = real_loop()
    seen = []

    def record(substep_idx, wheel_state):
        assist = loop.env.get_assist_state()
        feel = loop.car_feel()
        seen.append((loop._assist_wheel_target_rad, -assist["commanded_steering_rad"], feel.steering_angle_rad,
                     compute_force(feel, binding.HUMAN_STEERING_LIMIT_RAD), feel.assist_full))

    try:
        loop.env.set_assist_mode(binding.ASSIST_MODE_BLEND, blend_target)
        loop.wheel.set_controls(steering_norm=0.0, throttle_norm=0.5)
        for _ in range(step_count):
            loop.step_once(on_substep=record)
    finally:
        loop.close()
    return seen


@requires_checkpoint
def test_blend_at_zero_weight_has_no_gap_to_pull_across():
    seen = blend_drive(0.0)
    assert all(target_rad is not None for target_rad, *_ in seen)
    assert all(target_rad == commanded_rad for target_rad, commanded_rad, *_ in seen)
    assert all(abs(target_rad - wheel_rad) < 1e-6 for target_rad, _, wheel_rad, _, _ in seen)
    assert not any(assist_full for *_, assist_full in seen)


@requires_checkpoint
def test_blend_pulls_the_wheel_toward_where_the_car_is_steered():
    seen = blend_drive(1.0, step_count=2 * REAL_LOOP_STEP_COUNT)
    assert all(target_rad == commanded_rad for target_rad, commanded_rad, *_ in seen)
    pulling = [(target_rad - wheel_rad, force) for target_rad, _, wheel_rad, force, _ in seen
               if abs(target_rad - wheel_rad) > BLEND_MIN_PULL_GAP_RAD]
    assert pulling, "the policy never steered away from the straight wheel"
    assert all(math.copysign(1.0, force) == math.copysign(1.0, gap_rad) for gap_rad, force in pulling)
