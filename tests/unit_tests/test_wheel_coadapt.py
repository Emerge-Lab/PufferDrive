"""Co-adapt: a UCB1 bandit learning the Blend share per driver, and its wiring into run_demo, the HUD and the
launcher. No wheel, no CARLA; the real checkpoint only for the loop's rewards."""

import importlib.util
import math
import os

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo import coadapt as coadapt_module
from pufferlib.wheel_demo.coadapt import COADAPT_SHARES, COADAPT_TRIAL_SECONDS, CoAdapt, UcbBandit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "weights", "k_scaled_0050_1000", "final_model.pt")
CARLA_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "carla")
POLICY_DT_SECONDS = 0.3
CONTRACT_TRIAL_STEPS = round(COADAPT_TRIAL_SECONDS / POLICY_DT_SECONDS)
HUMAN_AGENT_IDX = 2
STEP_REWARD = 0.25
STEP_GOAL_REWARD = 1.0
UCB_SCORES = (1.0, 3.0, 2.0, 0.0)

requires_checkpoint = pytest.mark.skipif(not os.path.isfile(CHECKPOINT_PATH), reason="demo checkpoint not present")


def load_script(name):
    path = os.path.join(REPO_ROOT, "scripts", "wheel_demo", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"wheel_demo_{name}_coadapt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_demo():
    return load_script("run_demo")


class AssistEnv:
    def __init__(self, mode=binding.ASSIST_MODE_OFF):
        self.mode = mode
        self.calls = []

    def get_assist_state(self):
        return {"mode": self.mode}

    def set_assist_mode(self, mode, blend_target):
        self.calls.append((mode, blend_target))
        self.mode = mode


class RewardLoop:
    """What CoAdapt.after_step reads: the step's rewards, your car's index and its goal reward."""

    def __init__(self, agent_count=4, mode=binding.ASSIST_MODE_OFF):
        self.env = AssistEnv(mode)
        self.human_agent_idx = HUMAN_AGENT_IDX
        self.last_rewards = np.zeros(agent_count, dtype=np.float32)
        self.step_goal_reward = 0.0

    def human_state(self):
        return {"step_goal_reward": self.step_goal_reward}

    def set_step(self, reward, goal_reward=0.0):
        self.last_rewards[:] = -100.0
        self.last_rewards[self.human_agent_idx] = reward
        self.step_goal_reward = goal_reward


def run_trial(controller, loop, reward, goal_reward=0.0):
    finished = None
    for _ in range(controller.trial_steps):
        loop.set_step(reward, goal_reward)
        finished = controller.after_step(loop)
    return finished


def test_constants_are_the_contract():
    assert COADAPT_SHARES == (0.2, 0.4, 0.6, 0.8)
    assert COADAPT_TRIAL_SECONDS > 0.0
    assert CoAdapt(POLICY_DT_SECONDS).trial_steps == CONTRACT_TRIAL_STEPS
    assert CoAdapt(POLICY_DT_SECONDS, trial_seconds=0.01).trial_steps == 1


def test_bandit_tries_every_arm_in_order_first():
    bandit = UcbBandit(len(COADAPT_SHARES))
    order = []
    for score in UCB_SCORES:
        arm = bandit.select()
        order.append(arm)
        bandit.record(arm, score)
    assert order == [0, 1, 2, 3]
    assert [bandit.mean(arm) for arm in range(4)] == list(UCB_SCORES)


def expected_ucb_arm(counts, sums):
    total = sum(counts)
    low, high = min(sums[a] / counts[a] for a in range(len(counts))), max(sums[a] / counts[a] for a in range(len(counts)))
    bounds = [((s / c - low) / (high - low) if high > low else 0.5) + math.sqrt(2.0 * math.log(total) / c)
              for c, s in zip(counts, sums)]
    return max(range(len(bounds)), key=lambda arm: (bounds[arm], -arm))


def test_bandit_then_follows_the_scaled_upper_confidence_bound():
    bandit = UcbBandit(len(COADAPT_SHARES))
    for arm, score in enumerate(UCB_SCORES):
        bandit.record(arm, score)
    assert bandit.select() == expected_ucb_arm([1, 1, 1, 1], list(UCB_SCORES)) == 1
    bandit.record(1, 3.0)
    bandit.record(1, 3.0)
    counts, sums = [1, 3, 1, 1], [1.0, 9.0, 2.0, 0.0]
    assert bandit.select() == expected_ucb_arm(counts, sums)


def test_bandit_with_equal_scores_ties_to_the_lowest_arm():
    bandit = UcbBandit(3)
    for arm in range(3):
        bandit.record(arm, 5.0)
    assert bandit.select() == 0
    bandit.record(0, 5.0)
    assert bandit.select() == 1


def test_bandit_mean_is_none_until_tried():
    bandit = UcbBandit(2)
    assert bandit.mean(0) is None
    bandit.record(0, -2.0)
    bandit.record(0, 1.0)
    assert bandit.mean(0) == pytest.approx(-0.5)
    assert bandit.mean(1) is None


def test_start_begins_a_trial_at_the_first_share_under_blend():
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop()
    assert not controller.active and controller.share is None
    controller.start(loop.env)
    assert controller.active
    assert controller.share == COADAPT_SHARES[0]
    assert loop.env.calls == [(binding.ASSIST_MODE_BLEND, COADAPT_SHARES[0])]
    assert controller.seconds_left == pytest.approx(CONTRACT_TRIAL_STEPS * POLICY_DT_SECONDS)


def test_a_trial_scores_your_return_without_the_goal_reward():
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop()
    controller.start(loop.env)
    for step in range(controller.trial_steps - 1):
        loop.set_step(STEP_REWARD + STEP_GOAL_REWARD, STEP_GOAL_REWARD)
        assert controller.after_step(loop) is None
        assert controller.seconds_left == pytest.approx((controller.trial_steps - step - 1) * POLICY_DT_SECONDS)
    loop.set_step(STEP_REWARD + STEP_GOAL_REWARD, STEP_GOAL_REWARD)
    finished_arm, score = controller.after_step(loop)
    assert finished_arm == 0
    assert score == pytest.approx(controller.trial_steps * STEP_REWARD, rel=1e-6)
    assert controller.trial_count == 1
    assert controller.bandit.counts == [1, 0, 0, 0]
    assert controller.share == COADAPT_SHARES[1]
    assert loop.env.calls[-1] == (binding.ASSIST_MODE_BLEND, COADAPT_SHARES[1])
    assert controller.seconds_left == pytest.approx(CONTRACT_TRIAL_STEPS * POLICY_DT_SECONDS)


def test_trials_visit_every_share_in_order_then_follow_the_bandit():
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop()
    controller.start(loop.env)
    for score_per_step in UCB_SCORES:
        run_trial(controller, loop, score_per_step)
    applied = [share for mode, share in loop.env.calls]
    assert applied == list(COADAPT_SHARES) + [COADAPT_SHARES[controller.arm]]
    assert all(mode == binding.ASSIST_MODE_BLEND for mode, _ in loop.env.calls)
    assert controller.arm == controller.bandit.select() == 1


def test_inactive_after_step_does_nothing():
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop()
    loop.set_step(STEP_REWARD)
    assert controller.after_step(loop) is None
    assert controller.bandit.counts == [0, 0, 0, 0]


def test_stop_drops_the_trial_in_progress():
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop()
    controller.start(loop.env)
    for _ in range(controller.trial_steps - 1):
        loop.set_step(STEP_REWARD)
        controller.after_step(loop)
    controller.stop()
    assert not controller.active and controller.arm is None and controller.share is None
    assert controller.bandit.counts == [0, 0, 0, 0]
    loop.set_step(STEP_REWARD)
    assert controller.after_step(loop) is None


def test_restart_begins_a_fresh_trial_and_keeps_what_was_learned():
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop()
    controller.start(loop.env)
    run_trial(controller, loop, STEP_REWARD)
    for _ in range(5):
        loop.set_step(STEP_REWARD)
        controller.after_step(loop)
    controller.restart(loop.env)
    assert controller.bandit.counts == [1, 0, 0, 0]
    assert controller.share == COADAPT_SHARES[1]
    assert controller.seconds_left == pytest.approx(CONTRACT_TRIAL_STEPS * POLICY_DT_SECONDS)
    _, score = run_trial(controller, loop, -STEP_REWARD)
    assert score == pytest.approx(-controller.trial_steps * STEP_REWARD, rel=1e-6)


def test_restart_while_inactive_changes_nothing():
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop()
    controller.restart(loop.env)
    assert not controller.active
    assert loop.env.calls == []


def test_summary_shows_each_shares_mean_and_count():
    controller = CoAdapt(POLICY_DT_SECONDS, trial_seconds=POLICY_DT_SECONDS)
    loop = RewardLoop()
    controller.start(loop.env)
    run_trial(controller, loop, 1.5)
    run_trial(controller, loop, -0.5)
    assert controller.summary() == "20% +1.50 (1)  40% -0.50 (1)  60%   -   (0)  80%   -   (0)"


def test_the_module_needs_no_wheel_or_carla():
    assert not hasattr(coadapt_module, "carla")


def test_assist_choices_add_co_adapt(run_demo):
    assert run_demo.COADAPT == "co-adapt"
    assert run_demo.ASSIST_CHOICES[: len(run_demo.ASSIST_MODES_BY_NAME) + 1] == tuple(run_demo.ASSIST_MODES_BY_NAME) + (
        "co-adapt",)


@pytest.mark.parametrize("mode", [binding.ASSIST_MODE_OFF, binding.ASSIST_MODE_NUDGE, binding.ASSIST_MODE_FULL])
def test_the_cycle_never_starts_co_adapt(run_demo, mode):
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop(mode=mode)
    for _ in range(len(run_demo.ASSIST_CYCLE) + 1):
        run_demo.cycle_assist(loop, 0.5, controller)
        assert not controller.active


def test_a_running_co_adapt_stops_and_counts_as_blend(run_demo, capsys):
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop()
    controller.start(loop.env)
    capsys.readouterr()
    run_demo.cycle_assist(loop, 0.5, controller)
    assert not controller.active
    assert loop.env.calls[-1] == (binding.ASSIST_MODE_NUDGE, 0.5)
    from pufferlib.wheel_demo.hud import ASSIST_MODE_NAMES

    assert capsys.readouterr().out == f"  assist -> {ASSIST_MODE_NAMES[binding.ASSIST_MODE_NUDGE]}\n"


def test_cycle_without_controllers_skips_dynamic(run_demo):
    loop = RewardLoop(mode=binding.ASSIST_MODE_NUDGE)
    run_demo.cycle_assist(loop, 0.5)
    run_demo.cycle_assist(loop, 0.5)
    assert loop.env.calls == [(binding.ASSIST_MODE_FULL, 0.5), (binding.ASSIST_MODE_OFF, 0.5)]


def test_the_hud_line_is_empty_unless_co_adapt_runs(run_demo):
    controller = CoAdapt(POLICY_DT_SECONDS)
    assert run_demo.coadapt_hud_line(controller) == ""
    loop = RewardLoop()
    controller.start(loop.env)
    run_trial(controller, loop, STEP_REWARD)
    loop.set_step(STEP_REWARD)
    controller.after_step(loop)
    seconds_left = (CONTRACT_TRIAL_STEPS - 1) * POLICY_DT_SECONDS
    assert run_demo.coadapt_hud_line(controller) == f"40% policy, trial 2, {seconds_left:.0f} s left"


def test_a_finished_trial_prints_one_line(run_demo, capsys):
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RewardLoop()
    controller.start(loop.env)
    run_demo.report_coadapt_trial(controller, None)
    assert capsys.readouterr().out == ""
    finished = run_trial(controller, loop, STEP_REWARD)
    run_demo.report_coadapt_trial(controller, finished)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("  co-adapt trial 1: 20% scored +")
    assert lines[0].endswith("| next 40%")
    assert controller.summary() in lines[0]


class FakeWindow:
    def __init__(self, keys):
        self.keys = keys
        self.restarts = 0

    def poll(self):
        return dict(self.keys)

    def restart(self):
        self.restarts += 1


class RestartingLoop(RewardLoop):
    def restart(self):
        pass


def test_restart_starts_a_fresh_co_adapt_trial(run_demo):
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RestartingLoop()
    controller.start(loop.env)
    for _ in range(4):
        loop.set_step(STEP_REWARD)
        controller.after_step(loop)
    run_demo.apply_window_keys(FakeWindow({"restart": True}), loop, 0, None, None, 0.5, None, controller)
    assert controller.seconds_left == pytest.approx(CONTRACT_TRIAL_STEPS * POLICY_DT_SECONDS)
    assert len(loop.env.calls) == 2


def test_window_keys_pass_the_controller_to_the_assist_cycle(run_demo):
    controller = CoAdapt(POLICY_DT_SECONDS)
    loop = RestartingLoop()
    controller.start(loop.env)
    run_demo.apply_window_keys(FakeWindow({"cycle_assist": True}), loop, 0, None, None, 0.5, None, controller)
    assert not controller.active
    assert loop.env.calls[-1] == (binding.ASSIST_MODE_NUDGE, 0.5)


class MainBuilt(Exception):
    pass


class CoadaptDemoLoop:
    def __init__(self, config, wheel):
        self.env = AssistEnv()
        self.human_agent_idx = 0
        self.dt = POLICY_DT_SECONDS
        self.substep_dt = POLICY_DT_SECONDS / 18
        CoadaptDemoLoop.instance = self


@pytest.mark.parametrize(
    "assist,expected_calls",
    [("co-adapt", [(binding.ASSIST_MODE_BLEND, COADAPT_SHARES[0])]), ("full", [(binding.ASSIST_MODE_FULL, 0.5)])],
)
def test_main_starts_co_adapt_instead_of_setting_the_mode(run_demo, monkeypatch, tmp_path, assist, expected_calls):
    def refuse_window(*args, **kwargs):
        raise MainBuilt

    monkeypatch.setattr(run_demo, "build_wheel_input", lambda *args: (object(), None))
    monkeypatch.setattr(run_demo, "WheelDemoLoop", CoadaptDemoLoop)
    monkeypatch.setattr(run_demo, "choose_town", lambda town: town or "Town01")
    monkeypatch.setattr(run_demo, "CarlaDemoWindow", refuse_window)
    argv = ["run_demo.py", "--input", "synthetic", "--render", "--town", "Town01", "--assist", assist,
            "--calibration", str(tmp_path / "none.json"), "--buttons", str(tmp_path / "none_buttons.json")]
    monkeypatch.setattr(run_demo.sys, "argv", argv)
    with pytest.raises(MainBuilt):
        run_demo.main()
    assert CoadaptDemoLoop.instance.env.calls == expected_calls


def hud_texts(monkeypatch, coadapt_line, assist_mode):
    import os as os_module

    os_module.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    from pufferlib.ocean.cosim import carla_bridge
    from pufferlib.wheel_demo import hud

    pygame.init()
    town_hud = hud.Hud(pygame, carla_bridge.bin_path_for_town("Town01"))
    texts = []
    real_text = town_hud._text

    def text(surface, font, message, position, color=hud.COLOR_TEXT, anchor="topleft"):
        texts.append((message, position))
        return real_text(surface, font, message, position, color, anchor)

    monkeypatch.setattr(town_hud, "_text", text)
    surface = pygame.Surface((1280, 720))
    assist = {"mode": assist_mode, "reference_valid": 1, "reference_steering_rad": 0.0, "reference_accel_mps2": 0.0,
              "blend_target": 0.2, "blend_weight": 0.2, "human_steering_rad": 0.0, "commanded_steering_rad": 0.0}
    town_hud._draw_info(surface, hud.HudInfo(coadapt_line=coadapt_line), assist)
    return [message for message, _ in texts], hud


def test_hud_names_co_adapt_and_shows_its_line(monkeypatch):
    texts, hud = hud_texts(monkeypatch, "20% policy, trial 1, 10 s left", binding.ASSIST_MODE_BLEND)
    assert "Assist: co-adapt" in texts
    assert "Co-adapt: 20% policy, trial 1, 10 s left" in texts
    assert hud.HudInfo().coadapt_line == ""


def test_plain_blend_shows_no_co_adapt_line(monkeypatch):
    texts, hud = hud_texts(monkeypatch, "", binding.ASSIST_MODE_BLEND)
    assert f"Assist: {hud.ASSIST_MODE_NAMES[binding.ASSIST_MODE_BLEND]}" in texts
    assert not any(message.startswith("Co-adapt") for message in texts)


@pytest.fixture(scope="module")
def launcher():
    pytest.importorskip("tkinter")
    return load_script("launcher")


def test_launcher_hides_co_adapt_but_still_passes_it_through(launcher):
    source = open(os.path.join(REPO_ROOT, "scripts", "wheel_demo", "launcher.py")).read()
    assert '("co-adapt", "Co-adapt")' not in source
    settings = dict(launcher.DEFAULTS, assist="co-adapt")
    command = launcher.demo_command(settings)
    assert command[command.index("--assist") + 1] == "co-adapt"


@requires_checkpoint
def test_the_loop_keeps_the_steps_rewards_for_your_slot():
    from pufferlib.wheel_demo.input.trace import SyntheticWheelInput
    from pufferlib.wheel_demo.loop import DemoConfig, WheelDemoLoop

    config = DemoConfig(checkpoint_path=CHECKPOINT_PATH, town="Town01", map_dir=CARLA_MAP_DIR, num_agents=8, seed=0)
    loop = WheelDemoLoop(config, SyntheticWheelInput())
    try:
        assert loop.last_rewards is None
        loop.step_once()
        assert loop.last_rewards.shape[0] == loop.observations.shape[0]
        assert np.all(np.isfinite(loop.last_rewards))
        assert "step_goal_reward" in loop.human_state()
        assert loop.env.begin_preview() == loop.human_agent_idx
        loop.env.end_preview()
        controller = CoAdapt(loop.dt)
        controller.start(loop.env)
        assert loop.env.get_assist_state()["mode"] == binding.ASSIST_MODE_BLEND
        assert loop.env.get_assist_state()["blend_target"] == pytest.approx(COADAPT_SHARES[0])
        loop.step_once()
        controller.after_step(loop)
        expected = float(loop.last_rewards[loop.human_agent_idx]) - loop.human_state()["step_goal_reward"]
        assert controller._trial_score == pytest.approx(expected)
    finally:
        loop.close()


class ParserCaptured(Exception):
    pass


def test_assist_help_states_the_trial_length(run_demo, monkeypatch):
    import argparse

    captured = {}

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise ParserCaptured

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(ParserCaptured):
        run_demo.main()
    action = next(a for a in captured["parser"]._actions if a.dest == "assist")
    assert tuple(action.choices) == run_demo.ASSIST_CHOICES
    assert f"one {COADAPT_TRIAL_SECONDS:.0f} s trial at a time" in action.help


def oracle_belief(trial_scores_by_arm, arm_count):
    """Softmax of arm means over the population std of all trial scores; untried arms at the overall mean."""
    scores = np.array([score for scores in trial_scores_by_arm.values() for score in scores], dtype=np.float64)
    if scores.size == 0 or scores.std() == 0.0:
        return np.full(arm_count, 1.0 / arm_count)
    means = np.array([np.mean(trial_scores_by_arm[arm]) if trial_scores_by_arm.get(arm) else scores.mean()
                      for arm in range(arm_count)])
    logits = means / scores.std()
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def bandit_with(trial_scores_by_arm, arm_count=len(COADAPT_SHARES)):
    bandit = UcbBandit(arm_count)
    for arm, scores in trial_scores_by_arm.items():
        for score in scores:
            bandit.record(arm, score)
    return bandit


BELIEF_CASES = [
    {0: [-3.0], 1: [-1.0], 2: [-2.0], 3: [4.0]},
    {0: [1.0, 2.0], 3: [5.0, 3.0, 4.0]},
    {1: [-10.0, 10.0]},
    {0: [0.5], 1: [0.5, 0.7], 2: [0.6], 3: [0.55]},
]


@pytest.mark.parametrize("trial_scores_by_arm", BELIEF_CASES)
def test_belief_is_the_softmax_of_means_over_the_score_spread(trial_scores_by_arm):
    belief = bandit_with(trial_scores_by_arm).best_arm_belief()
    np.testing.assert_allclose(belief, oracle_belief(trial_scores_by_arm, len(COADAPT_SHARES)), rtol=1e-9)
    assert sum(belief) == pytest.approx(1.0)


def test_belief_is_uniform_without_trials_or_spread():
    uniform = [1.0 / len(COADAPT_SHARES)] * len(COADAPT_SHARES)
    assert UcbBandit(len(COADAPT_SHARES)).best_arm_belief() == uniform
    assert bandit_with({0: [3.0], 2: [3.0, 3.0]}).best_arm_belief() == uniform


def test_an_untried_arm_stands_at_the_overall_mean():
    belief = bandit_with({0: [0.0], 1: [2.0]}).best_arm_belief()
    assert belief[2] == pytest.approx(belief[3])
    assert belief[0] < belief[2] < belief[1]


@pytest.mark.parametrize("offset", [1e6, -1e6])
def test_belief_does_not_overflow_on_large_scores(offset):
    trial_scores_by_arm = {arm: [offset + 50.0 * arm] for arm in range(len(COADAPT_SHARES))}
    belief = bandit_with(trial_scores_by_arm).best_arm_belief()
    np.testing.assert_allclose(belief, oracle_belief(trial_scores_by_arm, len(COADAPT_SHARES)), rtol=1e-6)
    assert sum(belief) == pytest.approx(1.0)


def test_belief_stays_a_distribution_even_where_the_spread_cancels_out():
    trial_scores_by_arm = {arm: [1e12 + 50.0 * arm] for arm in range(len(COADAPT_SHARES))}
    belief = bandit_with(trial_scores_by_arm).best_arm_belief()
    assert all(math.isfinite(p) and p >= 0.0 for p in belief)
    assert sum(belief) == pytest.approx(1.0)


def test_the_best_mean_gets_the_most_belief():
    belief = bandit_with({0: [1.0], 1: [9.0], 2: [3.0], 3: [2.0]}).best_arm_belief()
    assert max(range(4), key=lambda arm: belief[arm]) == 1


def test_square_sum_accumulates_every_score():
    bandit = bandit_with({0: [1.0, -2.0], 3: [3.0]})
    assert bandit.score_square_sum == pytest.approx(1.0 + 4.0 + 9.0)


def test_hud_belief_is_empty_unless_co_adapt_runs(run_demo):
    controller = CoAdapt(POLICY_DT_SECONDS)
    assert run_demo.coadapt_hud_belief(controller) == ()
    loop = RewardLoop()
    controller.start(loop.env)
    run_trial(controller, loop, STEP_REWARD)
    rows = run_demo.coadapt_hud_belief(controller)
    belief = controller.bandit.best_arm_belief()
    assert rows == tuple((f"{round(100 * share)}%", belief[arm], arm == 1) for arm, share in enumerate(COADAPT_SHARES))
    assert [label for label, _, _ in rows] == ["20%", "40%", "60%", "80%"]
    controller.stop()
    assert run_demo.coadapt_hud_belief(controller) == ()


BELIEF_ROWS = (("20%", 0.1, False), ("40%", 0.6, True), ("60%", 0.25, False), ("80%", 0.05, False))


def belief_hud(monkeypatch, belief):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    from pufferlib.ocean.cosim import carla_bridge
    from pufferlib.wheel_demo import hud

    pygame.init()
    town_hud = hud.Hud(pygame, carla_bridge.bin_path_for_town("Town01"))
    texts = []
    real_text = town_hud._text

    def text(surface, font, message, position, color=hud.COLOR_TEXT, anchor="topleft"):
        texts.append((message, tuple(position), tuple(color)))
        return real_text(surface, font, message, position, color, anchor)

    monkeypatch.setattr(town_hud, "_text", text)
    surface = pygame.Surface((1280, 720))
    surface.fill((0, 0, 0))
    assist = {"mode": binding.ASSIST_MODE_BLEND, "reference_valid": 1, "reference_steering_rad": 0.0,
              "reference_accel_mps2": 0.0, "blend_target": 0.4, "blend_weight": 0.4, "human_steering_rad": 0.0,
              "commanded_steering_rad": 0.0}
    town_hud._draw_info(surface, hud.HudInfo(coadapt_line="40% policy, trial 2, 5 s left", coadapt_belief=belief),
                        assist)
    return texts, pygame.surfarray.array3d(surface), hud


def test_the_hud_draws_one_belief_row_per_share(monkeypatch):
    texts, pixels, hud = belief_hud(monkeypatch, BELIEF_ROWS)
    top_px = hud.INFO_TOP_PX + 4 * hud.INFO_LINE_PX
    assert ("Best share (belief):", (hud.PANEL_MARGIN_PX, top_px), tuple(hud.COLOR_TEXT)) in texts
    bar_x = hud.PANEL_MARGIN_PX + hud.BELIEF_LABEL_WIDTH_PX
    for row, (label, probability, on_trial) in enumerate(BELIEF_ROWS):
        y = top_px + (row + 1) * hud.BELIEF_ROW_PX
        color = tuple(hud.COLOR_BELIEF_ON_TRIAL if on_trial else hud.COLOR_TEXT)
        assert (label, (hud.PANEL_MARGIN_PX, y), color) in texts
        value_position = (bar_x + hud.BELIEF_BAR_WIDTH_PX + hud.BELIEF_VALUE_GAP_PX, y)
        assert (f"{round(100 * probability)}%", value_position, color) in texts
        bar_row = pixels[bar_x : bar_x + hud.BELIEF_BAR_WIDTH_PX, y + hud.BELIEF_BAR_INSET_PX + hud.BELIEF_BAR_HEIGHT_PX // 2]
        filled_px = int(np.all(bar_row == color, axis=1).sum())
        assert filled_px == round(probability * hud.BELIEF_BAR_WIDTH_PX)
        track_px = int(np.all(bar_row == tuple(hud.COLOR_BAR_TRACK), axis=1).sum())
        assert filled_px + track_px == hud.BELIEF_BAR_WIDTH_PX


def test_the_hud_draws_no_belief_without_one(monkeypatch):
    texts, _, _ = belief_hud(monkeypatch, ())
    assert not any(message.startswith("Best share") for message, _, _ in texts)
