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
    raw = {"env.simulation_mode": "replay", "interval": 25}
    out = _expand_dotted(raw)
    assert out == {"env": {"simulation_mode": "replay"}, "interval": 25}


def test_inheritance_chain():
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
    # C inherits B inherits A. Each level overrides the one above.
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
    sections = {"a": {"inherits": "a"}}
    with pytest.raises(ValueError, match="Cyclic"):
        _build_section_config("a", sections["a"], sections)


def test_inheritance_child_wins():
    sections = {
        "parent": {"interval": 250, "env.scenario_length": 201},
        "child": {"inherits": "parent", "interval": 100, "env.scenario_length": 91},
    }
    cfg = _build_section_config("child", sections["child"], sections)
    assert cfg["interval"] == 100
    assert cfg["env"]["scenario_length"] == 91


def test_inheritance_cycle_detected():
    sections = {
        "a": {"inherits": "b"},
        "b": {"inherits": "a"},
    }
    with pytest.raises(ValueError, match="Cyclic"):
        _build_section_config("a", sections["a"], sections)


def test_inheritance_unknown_parent():
    sections = {
        "child": {"inherits": "nonexistent"},
    }
    with pytest.raises(ValueError, match="not a known section"):
        _build_section_config("child", sections["child"], sections)


def test_clean_macro_applied_by_default():
    sections = {"foo": {"type": "multi_scenario"}}
    cfg = _build_section_config("foo", sections["foo"], sections)
    for k, v in CLEAN_EVAL_OVERRIDES.items():
        assert cfg["env"][k] == v


def test_clean_macro_disabled_when_clean_false():
    sections = {"foo": {"type": "multi_scenario", "clean": False}}
    cfg = _build_section_config("foo", sections["foo"], sections)
    for k in CLEAN_EVAL_OVERRIDES:
        assert k not in cfg.get("env", {})


def test_clean_macro_loses_to_explicit_override():
    sections = {
        "foo": {
            "type": "multi_scenario",
            "env.lane_segment_dropout": 0.5,  # explicit > macro default of 0.0
        }
    }
    cfg = _build_section_config("foo", sections["foo"], sections)
    assert cfg["env"]["lane_segment_dropout"] == 0.5


def test_manager_from_config_skips_template_sections():
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
    # Behavior-style template specifies a small render budget; the per-class
    # section inherits it without re-declaring.
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
    train_config = {"eval": {"foo": {"type": "totally_made_up"}}}
    with pytest.raises(ValueError, match="not registered"):
        EvalManager.from_config(train_config)


def test_has_subprocess_evals_at():
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

    def fake_run(ev, *, policy, env_name, logger, global_step):
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


def test_rollout_records_eval_seconds():
    """Every rollout's metrics dict should include `eval_seconds` so wandb
    panels show wall-clock cost per evaluator."""
    import time as _time

    class _Stub(Evaluator):
        type_name = "_stub_timing"

        def _run_rollout_loop(self, vecenv, policy, args):
            _time.sleep(0.02)  # forced floor so the recorded time is > 0
            return {"some_metric": 1.5}

    s = _Stub("test", {}, {})
    result = s.rollout(vecenv=None, policy=None, args={})
    assert "eval_seconds" in result.metrics
    assert result.metrics["eval_seconds"] >= 0.02
    assert result.metrics["some_metric"] == 1.5


def test_eval_args_compose_train_section_and_clean_macro():
    """_build_eval_args must fold train_config['env'] (baseline) +
    section overrides + clean macro correctly. Section beats baseline,
    explicit beats clean macro, baseline survives when not overridden."""
    train_config = {
        "env": {
            "lane_segment_dropout": 0.5,  # training perturbation
            "scenario_length": 91,
            "num_agents": 1024,  # only present in train baseline
        },
        "train": {"seed": 42, "device": "cpu"},
        "eval": {
            "validation": {
                "type": "multi_scenario",
                "interval": 25,
                "env.scenario_length": 201,  # section overrides baseline
                # clean=true (default) → lane_segment_dropout zeroed by macro
                # num_agents not specified → falls through to train baseline
            },
        },
    }
    mgr = EvalManager.from_config(train_config)
    ev = mgr.evaluators[0]
    args = mgr._build_eval_args(ev, env_name="puffer_drive", global_step=0)

    assert args["env"]["scenario_length"] == 201, "section override wins"
    assert args["env"]["lane_segment_dropout"] == 0.0, "clean macro applied"
    assert args["env"]["num_agents"] == 1024, "train baseline preserved"
