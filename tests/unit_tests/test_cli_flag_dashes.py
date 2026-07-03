"""Unit tests for CLI underscore/dash flag normalization in pufferl."""

import argparse

from pufferlib.pufferl import normalize_flag_dashes


def make_parser():
    parser = argparse.ArgumentParser()
    # Config-derived flags: registered dash-style only
    parser.add_argument("--env.num-agents", type=int)
    parser.add_argument("--train.total-timesteps", type=int)
    parser.add_argument("--train.device", type=str)
    # Hand-registered top-level flags: underscore spellings
    parser.add_argument("--num_scenarios", type=int)
    parser.add_argument("--load-model-path", type=str)
    return parser


def test_underscore_section_flags_are_rewritten():
    argv = ["--env.num_agents", "64", "--train.total_timesteps", "8192"]
    assert normalize_flag_dashes(argv, make_parser()) == [
        "--env.num-agents",
        "64",
        "--train.total-timesteps",
        "8192",
    ]


def test_registered_underscore_flags_are_untouched():
    argv = ["--num_scenarios", "3"]
    assert normalize_flag_dashes(argv, make_parser()) == ["--num_scenarios", "3"]


def test_equals_form_rewrites_only_the_flag():
    argv = ["--env.num_agents=64", "--load-model-path=models/model_a_b.pt"]
    assert normalize_flag_dashes(argv, make_parser()) == [
        "--env.num-agents=64",
        "--load-model-path=models/model_a_b.pt",
    ]


def test_values_and_unknown_flags_are_untouched():
    argv = ["train", "puffer_drive", "--train.device", "cpu", "--not_a_flag", "-1.0", "some_value"]
    assert normalize_flag_dashes(argv, make_parser()) == argv


def test_parser_accepts_both_spellings_end_to_end():
    parser = make_parser()
    argv = ["--env.num_agents", "64", "--num_scenarios", "3"]
    parsed = vars(parser.parse_args(normalize_flag_dashes(argv, parser)))
    assert parsed["env.num_agents"] == 64
    assert parsed["num_scenarios"] == 3
