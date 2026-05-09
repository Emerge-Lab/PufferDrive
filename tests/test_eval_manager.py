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


def test_manager_unknown_type_raises():
    train_config = {"eval": {"foo": {"type": "totally_made_up"}}}
    with pytest.raises(ValueError, match="not registered"):
        EvalManager.from_config(train_config)
