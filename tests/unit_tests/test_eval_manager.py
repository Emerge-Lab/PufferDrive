"""Smoke tests for EvalManager config parsing + dispatch.

Doesn't load the full pufferl.py module (which pulls heavy training deps).
Verifies parser correctness, dispatch gating, info-flattening shape
handling, behavior-class symlink cleanup, and the train/section/macro
override resolution stack.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.benchmark.evaluators import EvalResult, Evaluator
from pufferlib.ocean.benchmark.evaluators.behavior_class import BehaviorClassEvaluator
from pufferlib.ocean.benchmark.manager import (
    CLEAN_EVAL_OVERRIDES,
    EvalManager,
    _build_section_config,
    _expand_dotted,
)


def test_dotted_expand():
    """`_expand_dotted` should turn `{"env.X": v}` flat keys into a nested
    `{"env": {"X": v}}` dict so configparser-style ini keys round-trip."""
    raw = {"env.simulation_mode": "replay", "interval": 25}
    out = _expand_dotted(raw)
    assert out == {"env": {"simulation_mode": "replay"}, "interval": 25}


def test_inheritance_chain():
    """Single-level inheritance: child should pull all parent fields it
    doesn't explicitly override, including nested env.*."""
    sections = {
        "behaviors_defaults": {
            "type": "behavior_class",
            "interval": 250,
            "env.simulation_mode": "replay",
            "env.scenario_length": 201,
        },
        "behaviors_hard_stop": {
            "inherits": "behaviors_defaults",
            "env.map_dir": "/tmp/hard_stop",
        },
    }
    cfg = _build_section_config("behaviors_hard_stop", sections["behaviors_hard_stop"], sections)
    assert cfg["type"] == "behavior_class"
    assert cfg["interval"] == 250
    assert cfg["env"]["simulation_mode"] == "replay"
    assert cfg["env"]["scenario_length"] == 201
    assert cfg["env"]["map_dir"] == "/tmp/hard_stop"


def test_inheritance_three_levels():
    """Three-level chain (C → B → A): nearest ancestor wins per field;
    grandparent fields survive when no descendant overrides them."""
    sections = {
        "A": {"interval": 100, "env.scenario_length": 91, "env.map_dir": "/A"},
        "B": {"inherits": "A", "interval": 200, "env.scenario_length": 201},
        "C": {"inherits": "B", "env.map_dir": "/C", "render": True},
    }
    cfg = _build_section_config("C", sections["C"], sections)
    assert cfg["interval"] == 200, "B should win over A on interval"
    assert cfg["env"]["scenario_length"] == 201, "B should win over A on scenario_length"
    assert cfg["env"]["map_dir"] == "/C", "C should win over A and B on map_dir"
    assert cfg["render"] is True, "C's own field"


def test_inheritance_self_cycle_detected():
    """A section that inherits from itself must raise rather than spin
    forever in the chain walk."""
    sections = {"a": {"inherits": "a"}}
    with pytest.raises(ValueError, match="Cyclic"):
        _build_section_config("a", sections["a"], sections)


def test_inheritance_child_wins():
    """When child and parent both set the same key (top-level scalar and
    nested env.*), the child's value should appear in the merged config."""
    sections = {
        "parent": {"interval": 250, "env.scenario_length": 201},
        "child": {"inherits": "parent", "interval": 100, "env.scenario_length": 91},
    }
    cfg = _build_section_config("child", sections["child"], sections)
    assert cfg["interval"] == 100
    assert cfg["env"]["scenario_length"] == 91


def test_inheritance_does_not_alias_parent_env():
    """Regression: _build_section_config used to merge by reference and mutate
    the shared parent's env dict, so processing N siblings that all override
    env.map_dir would leave every sibling reporting the LAST-processed sibling's
    map_dir (the parent's env was mutated in place)."""
    sections = {
        "defaults": {"env.scenario_length": 201, "interval": 5},
        "child_a": {"inherits": "defaults", "env.map_dir": "/path/a"},
        "child_b": {"inherits": "defaults", "env.map_dir": "/path/b"},
        "child_c": {"inherits": "defaults", "env.map_dir": "/path/c"},
    }
    cfg_a = _build_section_config("child_a", sections["child_a"], sections)
    cfg_b = _build_section_config("child_b", sections["child_b"], sections)
    cfg_c = _build_section_config("child_c", sections["child_c"], sections)
    assert cfg_a["env"]["map_dir"] == "/path/a"
    assert cfg_b["env"]["map_dir"] == "/path/b"
    assert cfg_c["env"]["map_dir"] == "/path/c"
    # The parent must remain untouched, even after building all children.
    assert "map_dir" not in sections["defaults"].get("env", {})


def test_inheritance_cycle_detected():
    """A two-section cycle (a→b→a) must raise rather than spin forever."""
    sections = {
        "a": {"inherits": "b"},
        "b": {"inherits": "a"},
    }
    with pytest.raises(ValueError, match="Cyclic"):
        _build_section_config("a", sections["a"], sections)


def test_inheritance_unknown_parent():
    """`inherits = "nonexistent"` should fail loudly rather than silently
    skip the missing parent."""
    sections = {
        "child": {"inherits": "nonexistent"},
    }
    with pytest.raises(ValueError, match="not a known section"):
        _build_section_config("child", sections["child"], sections)


def test_clean_macro_applied_by_default():
    """`clean = true` (the default) injects every key from CLEAN_EVAL_OVERRIDES
    into the merged env config, zeroing perturbations + enforcing red lights."""
    sections = {"foo": {"type": "multi_scenario"}}
    cfg = _build_section_config("foo", sections["foo"], sections)
    for k, v in CLEAN_EVAL_OVERRIDES.items():
        assert cfg["env"][k] == v


def test_clean_macro_disabled_when_clean_false():
    """`clean = false` opts out of the macro — none of the perturbation
    knobs get injected; they fall back to whatever the train config has."""
    sections = {"foo": {"type": "multi_scenario", "clean": False}}
    cfg = _build_section_config("foo", sections["foo"], sections)
    for k in CLEAN_EVAL_OVERRIDES:
        assert k not in cfg.get("env", {})


def test_clean_macro_loses_to_explicit_override():
    """An explicit env.* value in the section beats the macro default for
    that same key — useful when a particular eval wants to keep some
    perturbation on for a targeted test."""
    sections = {
        "foo": {
            "type": "multi_scenario",
            "env.obs_dropout_lane": 0.5,  # explicit > macro default of 0.0
            "env.obs_lane_stride": 3,
        }
    }
    cfg = _build_section_config("foo", sections["foo"], sections)
    assert cfg["env"]["obs_dropout_lane"] == 0.5
    assert cfg["env"]["obs_lane_stride"] == 3


def test_manager_from_config_skips_template_sections():
    """Sections without a `type` field are templates (parents only) — they
    should NOT be instantiated as Evaluators, only inherited from."""
    train_config = {
        "eval": {
            "behaviors_defaults": {"interval": 250, "env.scenario_length": 201},
            "behaviors_hard_stop": {
                "type": "behavior_class",
                "inherits": "behaviors_defaults",
                "env.map_dir": "/tmp/hard_stop",
            },
        },
    }
    mgr = EvalManager.from_config(train_config)
    names = [e.name for e in mgr.evaluators]
    assert "behaviors_hard_stop" in names
    assert "behaviors_defaults" not in names  # template, no `type` field


def test_render_num_scenarios_inheritable():
    """eval.* keys inherit from parent template just like env.* keys do —
    so a behaviors_defaults template's render budget is shared by every
    child class without each having to re-declare it."""
    sections = {
        "defaults": {
            "type": "behavior_class",
            "interval": 250,
            "eval.num_scenarios": 50,
            "eval.render_num_scenarios": 2,
        },
        "hard_stop": {
            "inherits": "defaults",
            "env.map_dir": "/tmp/hard_stop",
        },
    }
    cfg = _build_section_config("hard_stop", sections["hard_stop"], sections)
    assert cfg["eval"]["num_scenarios"] == 50
    assert cfg["eval"]["render_num_scenarios"] == 2


def test_manager_unknown_type_raises():
    """A section with `type = "<not in registry>"` must fail loudly at
    EvalManager construction rather than silently skipping the section."""
    train_config = {"eval": {"foo": {"type": "totally_made_up"}}}
    with pytest.raises(ValueError, match="not registered"):
        EvalManager.from_config(train_config)


def test_has_subprocess_evals_at():
    """has_subprocess_evals_at(epoch) should return True iff at least one
    enabled subprocess-mode evaluator's interval divides the epoch — the
    training loop uses this to decide whether to save_checkpoint() before
    firing evals (subprocesses load the policy from disk)."""
    train_config = {
        "eval": {
            "inline_one": {"type": "human_replay", "interval": 25, "mode": "inline"},
            "subprocess_one": {"type": "human_replay", "interval": 100, "mode": "subprocess"},
            "subprocess_disabled": {
                "type": "human_replay",
                "interval": 100,
                "mode": "subprocess",
                "enabled": False,
            },
        }
    }
    mgr = EvalManager.from_config(train_config)
    assert mgr.has_subprocess_evals_at(epoch=100) is True  # subprocess_one fires
    assert mgr.has_subprocess_evals_at(epoch=25) is False  # only inline at 25
    assert mgr.has_subprocess_evals_at(epoch=50) is False  # nothing at 50


def test_latest_checkpoint_finds_newest_pt(tmp_path):
    """latest_checkpoint should resolve to the most-recently-written .pt
    under data_dir/<env>_<run_id>/models/ — subprocess evals depend on
    this to load the freshest weights."""
    import time

    model_dir = tmp_path / "puffer_drive_run123" / "models"
    model_dir.mkdir(parents=True)
    p_old = model_dir / "model_puffer_drive_001.pt"
    p_old.write_text("a")
    time.sleep(0.05)
    p_new = model_dir / "model_puffer_drive_002.pt"
    p_new.write_text("b")

    train_config = {"data_dir": str(tmp_path), "eval": {}}
    mgr = EvalManager.from_config(train_config, run_id="run123")
    assert mgr.latest_checkpoint("puffer_drive") == str(p_new)


def test_latest_checkpoint_falls_back_to_load_model_path(tmp_path):
    """When no checkpoint dir exists yet (resume-from-elsewhere before
    first save), latest_checkpoint should return train_config['load_model_path']
    so the very first eval still has weights to evaluate."""
    train_config = {
        "data_dir": str(tmp_path),
        "load_model_path": "/some/resume/path.pt",
        "eval": {},
    }
    mgr = EvalManager.from_config(train_config, run_id="run123")
    # No models dir exists → falls back to load_model_path
    assert mgr.latest_checkpoint("puffer_drive") == "/some/resume/path.pt"


# -- Tier A: dispatch + invariants -----------------------------------------


def test_maybe_run_dispatches_by_interval_and_enabled(monkeypatch):
    """maybe_run should fire only enabled evaluators whose interval divides epoch."""
    train_config = {
        "eval": {
            "fires_at_25": {"type": "human_replay", "interval": 25},
            "fires_at_250": {"type": "human_replay", "interval": 250},
            "disabled": {"type": "human_replay", "interval": 25, "enabled": False},
            "zero_interval": {"type": "human_replay", "interval": 0},
        }
    }
    mgr = EvalManager.from_config(train_config)

    calls = []

    def fake_run(ev, *, policy, env_name, logger, global_step, epoch):
        calls.append(ev.name)
        return EvalResult(metrics={})

    monkeypatch.setattr(mgr, "_run_one", fake_run)

    mgr.maybe_run(epoch=25, policy=None, env_name="puffer_drive")
    assert calls == ["fires_at_25"], "only the 25-interval evaluator fires at epoch 25"
    calls.clear()

    mgr.maybe_run(epoch=250, policy=None, env_name="puffer_drive")
    assert sorted(calls) == ["fires_at_25", "fires_at_250"], "both fire at epoch 250"
    calls.clear()

    mgr.maybe_run(epoch=50, policy=None, env_name="puffer_drive")
    assert calls == ["fires_at_25"], "only fires_at_25 at epoch 50; nothing else"
    calls.clear()

    mgr.maybe_run(epoch=33, policy=None, env_name="puffer_drive")
    assert calls == [], "nothing fires when no interval divides the epoch"


def test_maybe_run_isolates_evaluator_failures(monkeypatch, capsys):
    """One failing evaluator must not skip the rest. The failure is printed
    visibly (traceback to stdout) and the loop continues with subsequent
    evaluators; the returned dict has entries only for those that succeeded."""
    train_config = {
        "eval": {
            "good_a": {"type": "human_replay", "interval": 5},
            "bad_middle": {"type": "human_replay", "interval": 5},
            "good_b": {"type": "human_replay", "interval": 5},
        }
    }
    mgr = EvalManager.from_config(train_config)

    def fake_run(ev, *, policy, env_name, logger, global_step, epoch):
        if ev.name == "bad_middle":
            raise RuntimeError("synthetic eval failure")
        return EvalResult(metrics={"ok": 1.0})

    monkeypatch.setattr(mgr, "_run_one", fake_run)
    results = mgr.maybe_run(epoch=5, policy=None, env_name="puffer_drive")

    # Successful evaluators land in results; the failed one is absent.
    assert set(results.keys()) == {"good_a", "good_b"}
    # The failure was loud — header in stdout, traceback in stderr.
    captured = capsys.readouterr()
    assert "bad_middle" in captured.out
    assert "synthetic eval failure" in captured.err


def test_flatten_infos_handles_shape_variations():
    """_flatten_infos must accept both list-of-list (multi-worker) and
    flat-list (PufferEnv) info shapes, plus None / empty entries."""

    class _Stub(Evaluator):
        type_name = "_stub_flatten"

        def _should_stop(self, *args, **kwargs):
            return True

    s = _Stub("test", {}, {})
    assert s._flatten_infos(None) == []
    assert s._flatten_infos([]) == []
    assert s._flatten_infos([None, None]) == []
    assert s._flatten_infos([[], []]) == []

    d1, d2, d3 = {"a": 1}, {"b": 2}, {"c": 3}
    # Multi-worker backend: list of per-worker info lists
    assert s._flatten_infos([[d1], [d2]]) == [d1, d2]
    assert s._flatten_infos([[d1, d2], [d3]]) == [d1, d2, d3]
    # PufferEnv backend: flat list of info dicts
    assert s._flatten_infos([d1, d2]) == [d1, d2]


def test_behavior_class_sets_num_eval_scenarios(tmp_path):
    """BehaviorClassEvaluator must set num_eval_scenarios alongside
    num_agents/num_maps. Without it, the C-side replay branch caps at
    drive.py's default of 16, so any category with >16 bins (or any
    eval.num_scenarios > 16 sampling target) silently truncates."""
    map_dir = tmp_path / "bins"
    map_dir.mkdir()
    for i in range(50):
        (map_dir / f"map_{i}.bin").write_text("x")

    # Sampling branch: num_scenarios < total bins.
    cfg_sampled = {
        "type": "behavior_class",
        "env": {"map_dir": str(map_dir)},
        "eval": {"num_scenarios": 50},
    }
    ev_s = BehaviorClassEvaluator("sampled", cfg_sampled, train_config={})
    env_s = ev_s.env_overrides()
    assert env_s["num_agents"] == 50
    assert env_s["num_maps"] == 50
    assert env_s["num_eval_scenarios"] == 50
    ev_s.cleanup()

    # All-bins branch: num_scenarios > total bins, no sampling.
    cfg_full = {
        "type": "behavior_class",
        "env": {"map_dir": str(map_dir)},
        "eval": {"num_scenarios": 999},
    }
    ev_f = BehaviorClassEvaluator("full", cfg_full, train_config={})
    env_f = ev_f.env_overrides()
    assert env_f["num_agents"] == 50
    assert env_f["num_maps"] == 50
    assert env_f["num_eval_scenarios"] == 50


def test_aggregate_infos_weighted_mean_by_emission_size():
    """When vec_log emissions have different agent counts (n), the per-key
    aggregate should be a weighted mean — not mean-of-means. Two emissions
    [{collision_rate: 0.1, n: 100}, {collision_rate: 0.5, n: 1}] should
    aggregate to ~0.104 (true per-agent global rate), NOT 0.30 (uniform
    mean-of-means)."""

    class _Stub(Evaluator):
        type_name = "_stub_agg"

        def _should_stop(self, *args, **kwargs):
            return True

    s = _Stub("test", {}, {})
    infos = [
        {"collision_rate": 0.1, "episode_return": 10.0, "n": 100.0},
        {"collision_rate": 0.5, "episode_return": 50.0, "n": 1.0},
    ]
    out = s._aggregate_infos(infos)

    # 0.1*100 + 0.5*1 = 10.5; /101 = ~0.1039
    assert abs(out["collision_rate"] - 10.5 / 101) < 1e-6
    assert abs(out["episode_return"] - (10.0 * 100 + 50.0 * 1) / 101) < 1e-6
    # Counts: 2 emissions, 101 agents behind the metric.
    assert out["num_log_cycles"] == 2
    assert out["num_agents_evaluated"] == 101
    # The "n" key is reported separately, not aggregated as a metric.
    assert "n" not in {k for k in out if k not in ("num_log_cycles", "num_agents_evaluated")}


def test_aggregate_infos_empty_returns_zero_counts():
    """No emissions (env produced no completed scenarios) should report
    both counts as zero rather than erroring."""

    class _Stub(Evaluator):
        type_name = "_stub_empty"

        def _should_stop(self, *args, **kwargs):
            return True

    out = _Stub("test", {}, {})._aggregate_infos([])
    assert out == {"num_log_cycles": 0, "num_agents_evaluated": 0.0}


def test_behavior_class_cleanup_removes_symlink_dir(tmp_path):
    """BehaviorClassEvaluator builds a tmp symlink dir when sampling.
    cleanup() must remove it; otherwise we accumulate leftovers."""
    map_dir = tmp_path / "bins"
    map_dir.mkdir()
    for i in range(5):
        (map_dir / f"map_{i}.bin").write_text("a")

    config = {
        "type": "behavior_class",
        "env": {"map_dir": str(map_dir)},
        "eval": {"num_scenarios": 2},
    }
    ev = BehaviorClassEvaluator("test_class", config, train_config={})

    overrides = ev.env_overrides()
    sampled = overrides["map_dir"]
    assert sampled != str(map_dir), "sampling should redirect to a tmp dir"
    assert os.path.isdir(sampled)
    assert len([f for f in os.listdir(sampled) if f.endswith(".bin")]) == 2

    ev.cleanup()
    assert not os.path.exists(sampled), "tmp dir should be gone after cleanup"
    assert ev._sampled_dir is None


def test_rollout_zeros_lstm_state_per_agent_on_done(monkeypatch):
    """Per-agent LSTM reset on terminations or truncations. Either signal
    means 'episode over, env reset' — the agent's next obs is from a fresh
    scenario and stale recurrent memory would bias the policy."""
    import gymnasium
    import numpy as np
    import torch

    import pufferlib.pytorch
    import pufferlib.spaces
    from pufferlib.ocean.benchmark.evaluators.base import Evaluator

    state = {"lstm_h": torch.ones(4, 8), "lstm_c": torch.ones(4, 8)}

    class _Ev(Evaluator):
        type_name = "_lstm_done"

        def _initial_reset(self, vecenv, args):
            return vecenv.reset_obs

        def _init_lstm_state(self, num_agents, policy, device, args):
            return state

        def _should_stop(self, args, infos_collected, steps):
            return steps >= 1

    class _Vec:
        observation_space = type("S", (), {"shape": (4, 6)})()
        # Discrete env, like drive "jerk": per-agent MultiDiscrete([12]) (not a
        # Box -> env_continuous is False), joint action_space is Box(shape=(4, 1)).
        single_action_space = gymnasium.spaces.MultiDiscrete([12])
        action_space = pufferlib.spaces.joint_space(single_action_space, 4)
        reset_obs = np.zeros((4, 6), dtype=np.float32)

        def step(self, action):
            # Agents 0,2 truncated; 1 terminated; 3 alive.
            return self.reset_obs, np.zeros(4), np.array([0, 1, 0, 0]), np.array([1, 0, 1, 0]), []

    class _Policy:
        is_continuous = False

        def forward_eval(self, ob, state):
            return torch.zeros(ob.shape[0], 1), None

    # Bypass sample_logits's distribution-shape gymnastics — return a
    # placeholder action; we only care about the post-step state masking.
    monkeypatch.setattr(
        pufferlib.pytorch,
        "sample_logits",
        lambda logits,
        action=None,
        action_selection=pufferlib.pytorch.ACTION_SELECT_MODE,
        env_continuous=None,
        policy=None: (
            torch.zeros(4, dtype=torch.long),
            None,
            None,
            None,
        ),
    )

    args = {"train": {"device": "cpu", "use_rnn": True}, "env": {}}
    _Ev("done_test", {}, args)._run_rollout_loop(_Vec(), _Policy(), args)

    # Done agents (0, 1, 2) zeroed; alive agent (3) untouched.
    assert state["lstm_h"][0].sum().item() == 0
    assert state["lstm_h"][1].sum().item() == 0
    assert state["lstm_h"][2].sum().item() == 0
    assert state["lstm_h"][3].sum().item() == 8
    assert state["lstm_c"][3].sum().item() == 8


def test_rollout_records_metric_render_and_total_seconds():
    """Rollout reports metric pass time, render pass time, and total
    separately. With render=False, render_seconds is ~0 and
    eval_seconds ≈ metric_seconds. With render=True, the two are
    summed into eval_seconds."""
    import time as _time

    class _Stub(Evaluator):
        type_name = "_stub_timing"

        def _run_rollout_loop(self, vecenv, policy, args):
            _time.sleep(0.02)
            return {"some_metric": 1.5}

        def _render_pass(self, vecenv, policy, args):
            _time.sleep(0.03)
            return []

    # render=false → no render time charged.
    s = _Stub("test", {}, {})
    result = s.rollout(vecenv=None, policy=None, args={})
    assert result.metrics["metric_seconds"] >= 0.02
    assert result.metrics["render_seconds"] < 0.01, "render skipped when render=false"
    assert result.metrics["eval_seconds"] >= 0.02
    assert abs(result.metrics["eval_seconds"] - result.metrics["metric_seconds"]) < 0.01

    # render=true → both timed, eval_seconds = metric + render.
    s2 = _Stub("test", {"render": True}, {})
    result2 = s2.rollout(vecenv=None, policy=None, args={})
    assert result2.metrics["metric_seconds"] >= 0.02
    assert result2.metrics["render_seconds"] >= 0.03
    expected_total = result2.metrics["metric_seconds"] + result2.metrics["render_seconds"]
    assert abs(result2.metrics["eval_seconds"] - expected_total) < 1e-6


def test_rollout_puts_policy_in_eval_mode_and_restores():
    """Policies are passed in their training-loop state (typically `.train()`).
    Eval rollout must flip the module to `.eval()` so dropout/batchnorm are
    deterministic, then restore the prior mode so subsequent training is
    unaffected. Verifies both: the rollout observes eval=True, and the
    policy ends back in train mode."""

    class _Stub(Evaluator):
        type_name = "_stub_eval_mode"

        def _run_rollout_loop(self, vecenv, policy, args):
            # Snapshot policy mode AT the moment of rollout so we can
            # assert it was flipped to eval.
            args["_observed_training"] = policy.training
            return {}

    class _Policy:
        training = True  # mimic nn.Module: starts in train mode

        def eval(self):
            self.training = False

        def train(self, mode: bool = True):
            self.training = mode

    policy = _Policy()
    args = {}
    _Stub("test", {}, {}).rollout(vecenv=None, policy=policy, args=args)
    assert args["_observed_training"] is False, "policy was not put in eval mode during rollout"
    assert policy.training is True, "policy was not restored to train mode after rollout"


def test_maybe_run_force_fires_every_enabled_evaluator(monkeypatch):
    """force=True is used at training shutdown — every enabled evaluator
    fires regardless of whether epoch % interval == 0. interval-disabled
    evaluators (interval=0 or enabled=false) are still skipped."""
    train_config = {
        "eval": {
            "every_5": {"type": "human_replay", "interval": 5},
            "every_250": {"type": "human_replay", "interval": 250},
            "disabled": {"type": "human_replay", "interval": 5, "enabled": False},
            "zero_interval": {"type": "human_replay", "interval": 0},
        }
    }
    mgr = EvalManager.from_config(train_config)
    calls = []

    def fake_run(ev, *, policy, env_name, logger, global_step, epoch):
        calls.append(ev.name)
        return EvalResult(metrics={})

    monkeypatch.setattr(mgr, "_run_one", fake_run)

    # epoch=37 lines up with no interval, but force=True should still fire
    # every enabled (and interval>0 OR force) evaluator.
    mgr.maybe_run(epoch=37, policy=None, env_name="puffer_drive", force=True)
    assert sorted(calls) == ["every_250", "every_5", "zero_interval"], (
        "force=True ignores interval check; disabled is still skipped"
    )


def test_eval_args_compose_train_section_and_clean_macro():
    """_build_eval_args must fold train_config['env'] (baseline) +
    section overrides + clean macro correctly. Section beats baseline,
    explicit beats clean macro, baseline survives when not overridden."""
    train_config = {
        "env": {
            "obs_dropout_lane": 0.5,  # training perturbation
            "scenario_length": 91,
            "num_agents": 1024,  # only present in train baseline
        },
        "train": {"seed": 42, "device": "cpu"},
        "eval": {
            "validation": {
                "type": "multi_scenario",
                "interval": 25,
                "env.scenario_length": 201,  # section overrides baseline
                # clean=true (default) → obs_dropout_lane zeroed by macro
                # num_agents not specified → falls through to train baseline
            },
        },
    }
    mgr = EvalManager.from_config(train_config)
    ev = mgr.evaluators[0]
    args = mgr._build_eval_args(ev, env_name="puffer_drive", global_step=0)

    assert args["env"]["scenario_length"] == 201, "section override wins"
    assert args["env"]["obs_dropout_lane"] == 0.0, "clean macro applied"
    assert args["env"]["num_agents"] == 1024, "train baseline preserved"


def test_run_subprocess_passes_step_epoch_and_reads_back_result(tmp_path, monkeypatch):
    """End-to-end contract of _run_subprocess:
      - cmd line includes --global-step, --epoch, --evaluator, --out,
        and --load-model-path when latest_checkpoint resolves.
      - After the fake subprocess writes the result JSON to --out,
        the parent reads it back into an EvalResult with the right
        metrics + frames.

    Mocks subprocess.run so the test doesn't actually launch python.
    """
    import json
    import subprocess

    train_config = {
        "data_dir": str(tmp_path),
        "load_model_path": "/fake/checkpoint.pt",
        "eval": {"my_sub": {"type": "human_replay", "interval": 5, "mode": "subprocess"}},
    }
    mgr = EvalManager.from_config(train_config, run_id="run123")
    ev = mgr.evaluators[0]

    seen = {}

    def fake_run(cmd, check):
        seen["cmd"] = list(cmd)
        # Locate --out and write the canonical subprocess payload there
        # so the parent can read it back.
        i = cmd.index("--out")
        out_path = cmd[i + 1]
        with open(out_path, "w") as f:
            json.dump(
                {
                    "name": "my_sub",
                    "metrics": {"collision_rate": 0.123},
                    "frames": ["/fake/Town01_epoch7_step1234.mp4"],
                },
                f,
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    res = mgr._run_subprocess(ev, env_name="puffer_drive", global_step=1234, epoch=7)

    # cmd carries the per-invocation knobs that used to be silently dropped.
    cmd = seen["cmd"]
    assert "--evaluator" in cmd and cmd[cmd.index("--evaluator") + 1] == "my_sub"
    assert "--global-step" in cmd and cmd[cmd.index("--global-step") + 1] == "1234"
    assert "--epoch" in cmd and cmd[cmd.index("--epoch") + 1] == "7"
    assert "--out" in cmd
    assert "--load-model-path" in cmd and cmd[cmd.index("--load-model-path") + 1] == "/fake/checkpoint.pt"
    # The result JSON written by the subprocess round-trips into EvalResult.
    assert res.metrics == {"collision_rate": 0.123}
    assert res.frames == ["/fake/Town01_epoch7_step1234.mp4"]


def test_render_pass_per_evaluator_subdir_and_step_glob(tmp_path, monkeypatch):
    """Render output isolation:
      - Each evaluator writes to its own `mp4/<ev_name>/` subdir.
      - Filenames carry `_epoch{E}_step{N}` from args.
      - Returned paths are filtered by the step suffix, so mp4s left over
        from a prior epoch (same dir, different step) don't bleed into
        this pass's result.frames.

    Mocks the env construction + render so we can control the filenames
    the fake env produces.
    """
    import gymnasium

    import pufferlib.spaces
    import pufferlib.vector
    from pufferlib.ocean.benchmark.evaluators.base import Evaluator

    suffixes_set = []

    class _FakeTarget:
        num_envs = 2

        def set_video_suffix(self, suffix, env_idx):
            suffixes_set.append((env_idx, suffix))

        def render(self, env_idx, view_mode):
            # Write a stub mp4 named like the real C-side would. The "bin
            # scenario_id" is faked as scene_{env_idx} for testability.
            from pathlib import Path

            view_part = "" if view_mode == 0 else "_bev"
            stem = f"scene_{env_idx}_epoch7_step1234{view_part}"
            (Path.cwd() / f"{stem}.mp4").write_bytes(b"\x00")

        def close_client(self, env_idx):
            pass

    class _FakeVec:
        observation_space = type("S", (), {"shape": (2, 4)})()
        # Discrete env (2 agents): per-agent MultiDiscrete([12]) (not a Box ->
        # env_continuous is False), joint action_space is Box(shape=(2, 1)).
        single_action_space = gymnasium.spaces.MultiDiscrete([12])
        action_space = pufferlib.spaces.joint_space(single_action_space, 2)

        def reset(self):
            import numpy as np

            return np.zeros((2, 4), dtype=np.float32), {}

        def get_state(self):
            return [0, 1]  # 2 envs

        def step(self, action):
            import numpy as np

            return np.zeros((2, 4)), np.zeros(2), np.zeros(2), np.ones(2), []

        def close(self):
            pass

    fake_vec = _FakeVec()

    # Whatever the real env_creator returns, we replace make() so the
    # render loop gets our fake; the kwargs/backend/num_envs are still
    # called but we don't care about them here.
    monkeypatch.setattr(pufferlib.vector, "make", lambda *a, **kw: fake_vec)

    # The render loop reads target via vec.envs[0] OR vec itself; force the
    # "no .envs" path by sticking _FakeTarget on the vec.
    fake_vec.__class__ = type("_F", (_FakeVec,), {})
    fake_vec_target = _FakeTarget()
    # Inject `_FakeTarget` as the target by patching getattr lookup.
    fake_vec.set_video_suffix = fake_vec_target.set_video_suffix
    fake_vec.render = fake_vec_target.render
    fake_vec.close_client = fake_vec_target.close_client
    fake_vec.num_envs = fake_vec_target.num_envs

    # Stub the policy + sample_logits so the rollout loop has no real model.
    import torch

    import pufferlib.pytorch

    class _Policy:
        is_continuous = False

        def forward_eval(self, ob, state):
            return torch.zeros(ob.shape[0], 1), None

    monkeypatch.setattr(
        pufferlib.pytorch,
        "sample_logits",
        lambda logits,
        action=None,
        action_selection=pufferlib.pytorch.ACTION_SELECT_MODE,
        env_continuous=None,
        policy=None: (
            torch.zeros(logits.shape[0], dtype=torch.long),
            None,
            None,
            None,
        ),
    )

    class _Ev(Evaluator):
        type_name = "_render_test"

        def _should_stop(self, args, infos_collected, steps):
            return True  # not used; render_pass doesn't run rollout loop

    cfg = {
        "render": True,
        "render_views": ["sim_state", "bev"],
        "eval": {"render_num_scenarios": 2, "render_max_steps": 1},
        "env": {},
    }
    ev = _Ev("my_eval", cfg, train_config={})
    args = {
        "env": {},
        "env_name": "puffer_drive",
        "train": {"device": "cpu"},
        "render_results_dir": str(tmp_path),
        "epoch": 7,
        "global_step": 1234,
        "eval": cfg["eval"],
    }

    # Drop a stale mp4 from a "previous epoch" in the SAME per-evaluator
    # subdir; the step-glob filter must NOT pick it up.
    out_dir = tmp_path / "mp4" / "my_eval"
    out_dir.mkdir(parents=True)
    (out_dir / "stale_epoch1_step100.mp4").write_bytes(b"\x00")

    paths = ev._render_pass(vecenv=None, policy=_Policy(), args=args)

    # 2 envs × 2 views = 4 mp4s, all named with this pass's epoch+step.
    names = sorted(p.name for p in paths)
    assert len(names) == 4, names
    for n in names:
        assert "_epoch7_step1234" in n
    # The stale mp4 from a prior epoch must NOT be returned.
    assert "stale_epoch1_step100.mp4" not in names
    # All mp4s land in this evaluator's per-eval subdir, isolated from
    # other evaluators' renders.
    for p in paths:
        assert p.parent == out_dir
