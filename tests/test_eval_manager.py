"""Smoke tests for EvalManager config parsing.

Doesn't load the full pufferl.py module (which pulls heavy training deps).
Just verifies the inheritance + clean macro + dotted-key expansion logic
behaves as the design doc says.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
