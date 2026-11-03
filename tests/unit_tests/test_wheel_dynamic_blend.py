"""Dynamic blend v2: Blend toward the policy's look-ahead with no help during a warm-up that learns the target from the
policy's own imagined drive, then a share that follows how the driver is doing. Fakes for the loop; one integration
test on the real checkpoint."""

import importlib.util
import math
import os

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo import dynamic_blend as dynamic_module
from pufferlib.wheel_demo.coadapt import CoAdapt, human_step_score
from pufferlib.wheel_demo.dynamic_blend import (
    DYNAMIC_BASELINE_FALLBACK,
    DYNAMIC_BASELINE_MIN_STEPS,
    DYNAMIC_DISAGREE_FULL_RAD,
    DYNAMIC_DISAGREE_SECONDS,
    DYNAMIC_DOWN_RATE_PER_S,
    DYNAMIC_GAP_SCALE,
    DYNAMIC_RECENT_MIN_SECONDS,
    DYNAMIC_RECENT_SECONDS,
    DYNAMIC_SHARE_MAX,
    DYNAMIC_SHARE_MIN,
    DYNAMIC_START_SHARE,
    DYNAMIC_UP_RATE_PER_S,
    DYNAMIC_USE_DISAGREEMENT,
    DYNAMIC_WARMUP_SECONDS,
    DynamicBlend,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "weights", "k_scaled_0050_1000", "final_model.pt")
CARLA_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "carla")
POLICY_DT_SECONDS = 0.3
HUMAN_AGENT_IDX = 1
POLICY_SCORE = 0.02
GOAL_REWARD = 1.0
POOR_SCORE = -0.5
SLOPE_REFERENCE_GAP = 0.24
SLOPE_REFERENCE_SHARE = 0.8
FULL_HELP_GAP = 0.30
SHARE_TOLERANCE = 1e-5
SDC_REPLAY_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")

requires_checkpoint = pytest.mark.skipif(not os.path.isfile(CHECKPOINT_PATH), reason="demo checkpoint not present")


class FakeEnv:
    def __init__(self):
        self.calls = []
        self.blend_steering_calls = []
        self.assist = {"mode": binding.ASSIST_MODE_OFF, "reference_valid": 1, "human_steering_rad": 0.0,
                       "blend_reference_steering_rad": 0.0, "reference_steering_rad": 0.0}

    def set_assist_mode(self, mode, blend_target):
        self.calls.append((mode, blend_target))
        self.assist["mode"] = mode

    def get_assist_state(self):
        return dict(self.assist)

    def set_blend_steering(self, steering_rad):
        self.blend_steering_calls.append(steering_rad)


class FakeLoop:
    """What DynamicBlend reads and drives: rewards, your car's goal reward, the assist state and the look-ahead."""

    def __init__(self, dt_s=POLICY_DT_SECONDS, agent_count=4):
        self.dt = dt_s
        self.env = FakeEnv()
        self.human_agent_idx = HUMAN_AGENT_IDX
        self.last_rewards = np.zeros(agent_count, dtype=np.float32)
        self.human = {"step_goal_reward": 0.0}
        self.policy_score = (POLICY_SCORE, True)
        self.blend_follows_lookahead = False
        self.follow_calls = []

    def human_state(self):
        return dict(self.human)

    def lookahead_policy_score(self):
        return self.policy_score

    def set_blend_follows_lookahead(self, follows):
        self.follow_calls.append(follows)
        self.blend_follows_lookahead = follows
        if not follows:
            self.env.set_blend_steering(None)

    def set_step(self, score, goal_reward=0.0, wheel_rad=0.0, policy_rad=0.0, reference_valid=1, policy_score=None):
        self.last_rewards[:] = -100.0
        self.last_rewards[self.human_agent_idx] = score + goal_reward
        self.human["step_goal_reward"] = goal_reward
        self.env.assist.update(human_steering_rad=wheel_rad, blend_reference_steering_rad=policy_rad,
                               reference_valid=reference_valid)
        if policy_score is not None:
            self.policy_score = policy_score


def drive(controller, loop, step_count, **step):
    for _ in range(step_count):
        loop.set_step(**step)
        controller.after_step()


def warmed_up(score=POLICY_SCORE, policy_score=(POLICY_SCORE, True)):
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    drive(controller, loop, controller.warmup_steps, score=score, policy_score=policy_score)
    assert controller.baseline is not None
    return controller, loop


def test_human_step_score_is_the_reward_without_the_goal():
    loop = FakeLoop()
    loop.set_step(POLICY_SCORE, goal_reward=GOAL_REWARD)
    assert human_step_score(loop) == pytest.approx(POLICY_SCORE, abs=1e-6)


def test_window_lengths_follow_the_seconds():
    controller = DynamicBlend(FakeLoop())
    assert controller.warmup_steps == round(DYNAMIC_WARMUP_SECONDS / POLICY_DT_SECONDS) == 100
    assert controller._recent_scores.maxlen == round(DYNAMIC_RECENT_SECONDS / POLICY_DT_SECONDS) == 100
    assert controller.recent_min_steps == round(DYNAMIC_RECENT_MIN_SECONDS / POLICY_DT_SECONDS) == 17
    assert controller._recent_disagreement_rad.maxlen == round(DYNAMIC_DISAGREE_SECONDS / POLICY_DT_SECONDS) == 10
    tiny = DynamicBlend(FakeLoop(dt_s=1e3))
    assert (tiny.warmup_steps, tiny.recent_min_steps) == (1, 1)


def test_constants_are_the_contract():
    assert DYNAMIC_DISAGREE_FULL_RAD == pytest.approx(90.0 / 450.0 * binding.HUMAN_STEERING_LIMIT_RAD)
    assert (DYNAMIC_SHARE_MIN, DYNAMIC_SHARE_MAX, DYNAMIC_START_SHARE) == (0.0, 1.0, 0.0)
    assert DYNAMIC_USE_DISAGREEMENT is False
    assert (DYNAMIC_UP_RATE_PER_S, DYNAMIC_DOWN_RATE_PER_S) == (0.10, 0.02)
    assert (DYNAMIC_GAP_SCALE, DYNAMIC_BASELINE_MIN_STEPS, DYNAMIC_BASELINE_FALLBACK) == (0.30, 50, -0.005)
    assert DYNAMIC_WARMUP_SECONDS == 30.0
    assert not hasattr(dynamic_module, "DYNAMIC_BASELINE_SECONDS")


def test_start_blends_at_the_start_share_toward_the_look_ahead_with_no_full_phase():
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    assert not controller.active and not controller.measuring_baseline
    controller.start()
    assert controller.active and controller.measuring_baseline
    assert loop.env.calls == [(binding.ASSIST_MODE_BLEND, DYNAMIC_START_SHARE)]
    assert loop.follow_calls == [True]
    assert (controller.share, controller.gap, controller.disagree) == (DYNAMIC_START_SHARE, 0.0, 0.0)
    assert controller.target is None
    assert controller.baseline_seconds_left == pytest.approx(controller.warmup_steps * POLICY_DT_SECONDS)


def test_stop_hands_the_blend_steering_back():
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    controller.stop()
    assert not controller.active
    assert loop.follow_calls == [True, False]
    assert loop.env.blend_steering_calls == [None]


def test_the_recent_window_fills_from_the_first_step():
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    drive(controller, loop, 3, score=POOR_SCORE, goal_reward=GOAL_REWARD)
    assert len(controller._recent_scores) == 3
    assert controller.recent_score == pytest.approx(POOR_SCORE, abs=1e-6)


def test_the_baseline_is_the_mean_clean_look_ahead_score():
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    clean_scores = []
    for step in range(controller.warmup_steps):
        imagined = step % 7 != 0
        clean = step % 3 != 0
        policy_score = POLICY_SCORE + 0.001 * step
        loop.set_step(POOR_SCORE)
        loop.policy_score = (policy_score, clean) if imagined else None
        if imagined and clean:
            clean_scores.append(policy_score)
        controller.after_step()
        assert controller.measuring_baseline == (step + 1 < controller.warmup_steps)
    assert len(clean_scores) >= DYNAMIC_BASELINE_MIN_STEPS
    assert controller.baseline == pytest.approx(np.mean(clean_scores))
    assert controller.target == controller.baseline


def test_too_few_clean_look_ahead_scores_fall_back():
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    drive(controller, loop, DYNAMIC_BASELINE_MIN_STEPS - 1, score=POLICY_SCORE, policy_score=(1.0, True))
    drive(controller, loop, controller.warmup_steps - DYNAMIC_BASELINE_MIN_STEPS + 1, score=POLICY_SCORE,
          policy_score=(1.0, False))
    assert controller.baseline == DYNAMIC_BASELINE_FALLBACK


@pytest.mark.parametrize("wheel_rad", [0.0, 0.05, DYNAMIC_DISAGREE_FULL_RAD])
def test_during_the_warm_up_the_share_stays_zero_whatever_the_disagreement(wheel_rad):
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    for _ in range(controller.warmup_steps - 1):
        drive(controller, loop, 1, score=POOR_SCORE, wheel_rad=wheel_rad)
        assert controller.measuring_baseline
        assert controller.gap == 0.0
        assert controller.disagree == pytest.approx(min(wheel_rad / DYNAMIC_DISAGREE_FULL_RAD, 1.0))
        assert controller.share == 0.0
    assert set(loop.env.calls) == {(binding.ASSIST_MODE_BLEND, 0.0)}


def expected_share(previous_share, gap, disagree, use_disagreement, dt_s=POLICY_DT_SECONDS):
    need = max(gap, disagree) if use_disagreement else gap
    desired = DYNAMIC_SHARE_MIN + (DYNAMIC_SHARE_MAX - DYNAMIC_SHARE_MIN) * need
    change = min(max(desired - previous_share, -DYNAMIC_DOWN_RATE_PER_S * dt_s), DYNAMIC_UP_RATE_PER_S * dt_s)
    return min(max(previous_share + change, DYNAMIC_SHARE_MIN), DYNAMIC_SHARE_MAX)


@pytest.mark.parametrize("use_disagreement", [False, True])
def test_the_controller_follows_the_contract_on_every_step_from_the_first(monkeypatch, use_disagreement):
    monkeypatch.setattr(dynamic_module, "DYNAMIC_USE_DISAGREEMENT", use_disagreement)
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    rng = np.random.default_rng(4)
    recent, disagreement = [], []
    share = controller.share
    for step in range(controller.warmup_steps + 2 * controller._recent_scores.maxlen):
        score = float(np.float32(POLICY_SCORE + rng.normal(0.0, 0.3)))
        wheel_rad, policy_rad = rng.normal(0.0, 0.1, 2)
        valid = int(step % 7 != 0)
        loop.set_step(score, goal_reward=GOAL_REWARD * (step % 11 == 0), wheel_rad=wheel_rad, policy_rad=policy_rad,
                      reference_valid=valid)
        controller.after_step()
        recent = (recent + [human_step_score(loop)])[-controller._recent_scores.maxlen:]
        if valid:
            disagreement = (disagreement + [abs(wheel_rad - policy_rad)])[-controller._recent_disagreement_rad.maxlen:]
        has_gap = controller.baseline is not None and len(recent) >= controller.recent_min_steps
        gap = min(max((controller.target - np.mean(recent)) / DYNAMIC_GAP_SCALE, 0.0), 1.0) if has_gap else 0.0
        disagree = min(max(np.mean(disagreement) / DYNAMIC_DISAGREE_FULL_RAD, 0.0), 1.0) if disagreement else 0.0
        share = expected_share(share, gap, disagree, use_disagreement)
        assert controller.gap == pytest.approx(gap, abs=1e-9)
        assert controller.disagree == pytest.approx(disagree, abs=1e-9)
        assert controller.share == pytest.approx(share, abs=1e-9)
        assert loop.env.calls[-1] == (binding.ASSIST_MODE_BLEND, controller.share)
    assert controller.baseline == pytest.approx(POLICY_SCORE)


def test_with_disagreement_on_a_disagreeing_good_drive_gets_help_again(monkeypatch):
    monkeypatch.setattr(dynamic_module, "DYNAMIC_USE_DISAGREEMENT", True)
    controller, loop = warmed_up()
    drive(controller, loop, controller._recent_scores.maxlen, score=POLICY_SCORE + 1.0,
          wheel_rad=DYNAMIC_DISAGREE_FULL_RAD)
    assert (controller.gap, controller.disagree) == (0.0, 1.0)
    assert controller.share == pytest.approx(DYNAMIC_SHARE_MAX)
    assert controller.hud_line().endswith(f"disagree {round(100 * controller.disagree)}%)")


def test_with_disagreement_off_a_large_disagreement_alone_never_raises_the_share():
    controller, loop = warmed_up()
    for _ in range(2 * controller._recent_scores.maxlen):
        drive(controller, loop, 1, score=POLICY_SCORE + 1.0, wheel_rad=DYNAMIC_DISAGREE_FULL_RAD)
        assert controller.gap == 0.0
        assert controller.share == DYNAMIC_SHARE_MIN
    assert controller.disagree == 1.0
    assert controller.hud_line().endswith("disagree 100% (off))")


def test_the_slope_is_unchanged_and_full_help_is_reached_at_the_full_gap():
    assert DYNAMIC_SHARE_MIN + (DYNAMIC_SHARE_MAX - DYNAMIC_SHARE_MIN) * SLOPE_REFERENCE_GAP / DYNAMIC_GAP_SCALE == (
        pytest.approx(SLOPE_REFERENCE_SHARE))
    assert DYNAMIC_GAP_SCALE == pytest.approx(FULL_HELP_GAP)


def test_a_drive_this_far_below_the_target_settles_at_80_percent():
    controller, loop = warmed_up()
    settle_step_count = 3 * controller._recent_scores.maxlen
    drive(controller, loop, settle_step_count, score=controller.target - SLOPE_REFERENCE_GAP)
    assert controller.gap == pytest.approx(SLOPE_REFERENCE_GAP / FULL_HELP_GAP, abs=SHARE_TOLERANCE)
    assert controller.share == pytest.approx(SLOPE_REFERENCE_SHARE, abs=SHARE_TOLERANCE)


@pytest.mark.parametrize("below_target", [FULL_HELP_GAP + 0.01, 2 * FULL_HELP_GAP, 10.0])
def test_a_gap_at_or_past_the_full_gap_drives_the_share_to_exactly_one(below_target):
    controller, loop = warmed_up()
    settle_step_count = 3 * controller._recent_scores.maxlen
    for _ in range(settle_step_count):
        drive(controller, loop, 1, score=controller.target - below_target)
        assert controller.share <= DYNAMIC_SHARE_MAX
        assert loop.env.calls[-1][1] <= DYNAMIC_SHARE_MAX
    assert controller.gap == 1.0
    assert controller.share == 1.0
    assert loop.env.calls[-1] == (binding.ASSIST_MODE_BLEND, 1.0)


def test_a_gap_of_exactly_the_full_gap_asks_for_about_full_help():
    controller, loop = warmed_up()
    drive(controller, loop, 3 * controller._recent_scores.maxlen, score=controller.target - FULL_HELP_GAP)
    assert controller.gap == pytest.approx(1.0, abs=SHARE_TOLERANCE)
    assert controller.share == pytest.approx(1.0, abs=SHARE_TOLERANCE)
    assert controller.share <= DYNAMIC_SHARE_MAX


class RealEnvCheckingEnv(FakeEnv):
    """Also hands every share to a real Drive env, which rejects a target outside its range."""

    def __init__(self, real_env):
        super().__init__()
        self.real_env = real_env

    def set_assist_mode(self, mode, blend_target):
        super().set_assist_mode(mode, blend_target)
        self.real_env.set_assist_mode(mode, blend_target)


def test_the_real_env_accepts_every_share_up_to_full_help():
    from pufferlib.ocean.drive.drive import Drive

    real_env = Drive(num_agents=1, min_agents_per_env=1, max_agents_per_env=1, num_maps=1, map_dir=SDC_REPLAY_MAP_DIR,
                     simulation_mode="replay", control_mode="control_sdc_only", sdc_controller="replay",
                     non_sdc_controller="replay", scenario_length=400, resample_frequency=1_000_000,
                     termination_mode=False, goal_source="gt", dynamics_model="jerk", action_type="continuous")
    try:
        real_env.reset()
        real_env.select_human_agent(0)
        loop = FakeLoop()
        loop.env = RealEnvCheckingEnv(real_env)
        controller = DynamicBlend(loop)
        controller.start()
        drive(controller, loop, controller.warmup_steps, score=POLICY_SCORE)
        drive(controller, loop, 3 * controller._recent_scores.maxlen, score=controller.target - 2 * FULL_HELP_GAP)
        assert controller.share == 1.0
        state = real_env.get_assist_state()
        assert state["mode"] == binding.ASSIST_MODE_BLEND
        assert state["blend_target"] == 1.0
    finally:
        real_env.close()


def test_after_the_warm_up_a_poor_drive_asks_for_help():
    controller, loop = warmed_up()
    assert controller.gap == pytest.approx(0.0, abs=1e-6)
    drive(controller, loop, controller._recent_scores.maxlen, score=POOR_SCORE)
    assert controller.gap == 1.0


def test_help_comes_quickly_and_goes_slowly_within_the_range():
    controller, loop = warmed_up(score=POOR_SCORE)
    up_step = DYNAMIC_UP_RATE_PER_S * POLICY_DT_SECONDS
    for _ in range(40):
        before = controller.share
        drive(controller, loop, 1, score=POOR_SCORE, wheel_rad=DYNAMIC_DISAGREE_FULL_RAD)
        assert controller.share - before <= up_step + 1e-12
    assert controller.share == pytest.approx(DYNAMIC_SHARE_MAX)
    down_step = DYNAMIC_DOWN_RATE_PER_S * POLICY_DT_SECONDS
    drive(controller, loop, controller._recent_scores.maxlen, score=POLICY_SCORE + 1.0)
    for _ in range(200):
        before = controller.share
        drive(controller, loop, 1, score=POLICY_SCORE + 1.0)
        assert before - controller.share <= down_step + 1e-12
        assert DYNAMIC_SHARE_MIN <= controller.share <= DYNAMIC_SHARE_MAX
    assert controller.share == pytest.approx(DYNAMIC_SHARE_MIN)


def test_inactive_after_step_does_nothing():
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    drive(controller, loop, 5, score=POLICY_SCORE)
    assert loop.env.calls == [] and controller.baseline is None
    assert controller.hud_line() == ""


def test_stopping_mid_warm_up_restarts_it_from_zero():
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    drive(controller, loop, controller.warmup_steps - 1, score=POOR_SCORE, policy_score=(POOR_SCORE, True))
    controller.stop()
    controller.start()
    assert controller.baseline_seconds_left == pytest.approx(controller.warmup_steps * POLICY_DT_SECONDS)
    drive(controller, loop, controller.warmup_steps, score=POLICY_SCORE, policy_score=(POLICY_SCORE, True))
    assert controller.baseline == pytest.approx(POLICY_SCORE)


def test_a_learned_target_survives_stop_start_and_restart():
    controller, loop = warmed_up()
    drive(controller, loop, controller.recent_min_steps, score=POOR_SCORE, wheel_rad=0.2)
    baseline = controller.baseline
    controller.stop()
    controller.start()
    assert controller.baseline == baseline
    assert loop.env.calls[-1] == (binding.ASSIST_MODE_BLEND, DYNAMIC_START_SHARE)
    assert (controller.gap, controller.disagree) == (0.0, 0.0)
    assert controller.recent_score is None and len(controller._recent_disagreement_rad) == 0
    drive(controller, loop, 3, score=POOR_SCORE, wheel_rad=0.2)
    controller.restart()
    assert controller.baseline == baseline and controller.share == DYNAMIC_START_SHARE
    assert (controller.gap, controller.disagree) == (0.0, 0.0)
    assert controller.recent_score is None and len(controller._recent_disagreement_rad) == 0
    assert loop.follow_calls[-1] is True


def test_restart_while_inactive_does_nothing():
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.restart()
    assert not controller.active and loop.env.calls == [] and loop.follow_calls == []


def test_hud_line_during_and_after_the_warm_up():
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    seconds = math.ceil(controller.warmup_steps * POLICY_DT_SECONDS)
    assert controller.hud_line() == f"0% policy (no help yet: learning the target, {seconds} s left)"
    drive(controller, loop, 1, score=POLICY_SCORE, wheel_rad=0.05)
    seconds = math.ceil((controller.warmup_steps - 1) * POLICY_DT_SECONDS)
    assert controller.hud_line() == f"0% policy (no help yet: learning the target, {seconds} s left)"
    drive(controller, loop, controller.warmup_steps, score=POOR_SCORE, wheel_rad=0.05)
    assert controller.hud_line() == (f"{round(100 * controller.share)}% policy (recent {controller.recent_score:+.3f} vs "
                                     f"target {controller.target:+.3f}, gap {round(100 * controller.gap)}%, disagree "
                                     f"{round(100 * controller.disagree)}% (off))")
    controller.stop()
    assert controller.hud_line() == ""


def load_script(name):
    path = os.path.join(REPO_ROOT, "scripts", "wheel_demo", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"wheel_demo_{name}_dynamic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_demo():
    return load_script("run_demo")


def test_assist_choices_end_with_co_adapt_and_dynamic(run_demo):
    assert run_demo.DYNAMIC == "dynamic"
    assert run_demo.ASSIST_CHOICES == tuple(run_demo.ASSIST_MODES_BY_NAME) + ("co-adapt", "dynamic")


def test_the_cycle_is_off_display_blend_nudge_dynamic_full(run_demo, capsys):
    from pufferlib.wheel_demo.hud import ASSIST_MODE_NAMES

    assert run_demo.ASSIST_CYCLE == ("off", "display", "blend", "nudge", "dynamic", "full")
    assert run_demo.ASSIST_NAMES_BY_MODE == {mode: name for name, mode in run_demo.ASSIST_MODES_BY_NAME.items()}
    loop = FakeLoop()
    coadapt = CoAdapt(POLICY_DT_SECONDS)
    dynamic = DynamicBlend(loop)
    printed = []
    for _ in range(len(run_demo.ASSIST_CYCLE)):
        run_demo.cycle_assist(loop, 0.5, coadapt, dynamic)
        printed.append(capsys.readouterr().out)
        assert not coadapt.active
    expected_modes = [binding.ASSIST_MODE_DISPLAY, binding.ASSIST_MODE_BLEND, binding.ASSIST_MODE_NUDGE,
                      binding.ASSIST_MODE_BLEND, binding.ASSIST_MODE_FULL, binding.ASSIST_MODE_OFF]
    assert [mode for mode, _ in loop.env.calls] == expected_modes
    assert loop.env.calls[3] == (binding.ASSIST_MODE_BLEND, DYNAMIC_START_SHARE)
    seconds = math.ceil(dynamic.warmup_steps * POLICY_DT_SECONDS)
    assert printed[3] == f"  assist -> dynamic blend (0% policy (no help yet: learning the target, {seconds} s left))\n"
    assert printed[4] == f"  assist -> {ASSIST_MODE_NAMES[binding.ASSIST_MODE_FULL]}\n"
    assert printed[5] == f"  assist -> {ASSIST_MODE_NAMES[binding.ASSIST_MODE_OFF]}\n"
    assert not dynamic.active
    assert loop.follow_calls == [True, False]


def test_nudge_leads_to_dynamic_and_dynamic_to_full(run_demo):
    loop = FakeLoop()
    loop.env.assist["mode"] = binding.ASSIST_MODE_NUDGE
    dynamic = DynamicBlend(loop)
    run_demo.cycle_assist(loop, 0.5, None, dynamic)
    assert dynamic.active
    run_demo.cycle_assist(loop, 0.5, None, dynamic)
    assert not dynamic.active and loop.env.calls[-1] == (binding.ASSIST_MODE_FULL, 0.5)


def test_the_warm_up_report_prints_once_on_the_transition(run_demo, capsys):
    loop = FakeLoop()
    controller = DynamicBlend(loop)
    controller.start()
    for _ in range(controller.warmup_steps + 3):
        was_measuring = controller.measuring_baseline
        loop.set_step(POLICY_SCORE)
        controller.after_step()
        run_demo.report_dynamic_baseline(controller, was_measuring)
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"  dynamic blend: policy's look-ahead baseline {controller.baseline:+.4f} per step; target "
                     f"{controller.target:+.4f}; now at {round(100 * controller.share)}% policy"]
    controller.stop()
    run_demo.report_dynamic_baseline(controller, True)
    assert capsys.readouterr().out == ""


class FakeWindow:
    def __init__(self, keys):
        self.keys = keys

    def poll(self):
        return dict(self.keys)

    def restart(self):
        pass


class RestartingLoop(FakeLoop):
    def restart(self):
        pass


def test_restart_key_restarts_dynamic_blend(run_demo):
    loop = RestartingLoop()
    controller = DynamicBlend(loop)
    controller.start()
    drive(controller, loop, 5, score=POOR_SCORE, wheel_rad=0.2)
    run_demo.apply_window_keys(FakeWindow({"restart": True}), loop, 0, None, None, 0.5, None, None, controller)
    assert controller.recent_score is None and controller.share == DYNAMIC_START_SHARE


def test_launcher_offers_dynamic_blend():
    pytest.importorskip("tkinter")
    launcher = load_script("launcher")
    command = launcher.demo_command(dict(launcher.DEFAULTS, assist="dynamic"))
    assert command[command.index("--assist") + 1] == "dynamic"


@requires_checkpoint
def test_a_real_drive_blends_from_the_start_and_learns_its_target_from_the_look_ahead():
    from pufferlib.wheel_demo.input.trace import SyntheticWheelInput
    from pufferlib.wheel_demo.loop import DemoConfig, WheelDemoLoop

    config = DemoConfig(checkpoint_path=CHECKPOINT_PATH, town="Town01", map_dir=CARLA_MAP_DIR, num_agents=8, seed=0)
    loop = WheelDemoLoop(config, SyntheticWheelInput())
    try:
        assert loop.dt == pytest.approx(POLICY_DT_SECONDS)
        controller = DynamicBlend(loop)
        controller.start()
        assert loop.blend_follows_lookahead
        assert controller.warmup_steps == 100
        for step in range(controller.warmup_steps):
            loop.step_once()
            controller.after_step()
            assist = loop.env.get_assist_state()
            assert assist["mode"] == binding.ASSIST_MODE_BLEND
            assert assist["blend_target"] == pytest.approx(controller.share, abs=1e-6)
            assert controller.measuring_baseline == (step + 1 < controller.warmup_steps)
        assert math.isfinite(controller.baseline)
        assert controller.baseline != DYNAMIC_BASELINE_FALLBACK
        controller.stop()
        assert not loop.blend_follows_lookahead
        loop.step_once()
        assist = loop.env.get_assist_state()
        assert assist["blend_reference_steering_rad"] == assist["reference_steering_rad"]
    finally:
        loop.close()
