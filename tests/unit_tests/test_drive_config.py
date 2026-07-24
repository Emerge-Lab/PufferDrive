#!/usr/bin/env python3
"""
Test script for PufferDrive configuration loading.

Details:
Running the test: python -m unittest tests/test_drive_config.py
"""

import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import load_config, pufferlib

VERBOSITY = 0


class TestDriveConfig(unittest.TestCase):
    @patch("sys.argv", ["pufferl.py"])
    def test_load_config(self):
        """
        Tests that load_config correctly loads the monolithic Hydra config
        for the env, without being affected by unittest's command-line
        arguments.
        """
        try:
            # The ENV_NAME 'puffer_drive' loads pufferlib/config/puffer_drive.yaml
            args = load_config("puffer_drive")

            # load_config should return a populated config dict without raising.
            self.assertIsInstance(args, dict)
            self.assertTrue(len(args) > 0)
            self.assertTrue(args["eval"]["training_enabled"])
            self.assertEqual(args["eval"]["training_interval"], 100)
            self.assertEqual(args["eval"]["training_datasets"], "carla_fast")
            self.assertEqual(args["eval"]["benchmark_sdc_num_envs"], 8)
            self.assertFalse(args["eval"]["render_scenarios"])
            self.assertIsNone(args["eval"]["render_failures_number"])
            self.assertIsNone(args["eval"]["replay_failures_csv"])
            self.assertTrue(os.path.isfile(args["eval"]["catalog"]))
            self.assertTrue(os.path.isfile(args["eval"]["evaluation_config"]))

        except Exception as err:
            self.fail(f"load_config failed with an unexpected exception: {err}")

    @patch("sys.argv", ["pufferl.py", "env.obs_lane_stride=3", "env.obs_boundary_stride=4"])
    def test_obs_stride_cli_override(self):
        args = load_config("puffer_drive")
        self.assertEqual(args["env"]["obs_lane_stride"], 3)
        self.assertEqual(args["env"]["obs_boundary_stride"], 4)

    def test_obs_stride_validation(self):
        with self.assertRaisesRegex(ValueError, "obs_lane_stride"):
            Drive(obs_lane_stride=0)
        with self.assertRaisesRegex(ValueError, "obs_boundary_stride"):
            Drive(obs_boundary_stride=0)

    @patch("sys.argv", ["pufferl.py", "train.learning_rate=0.5"])
    def test_cli_override(self):
        """Test that Hydra CLI overrides win over the config file values."""
        args = load_config("puffer_drive")
        self.assertEqual(args["train"]["learning_rate"], 0.5)

    @patch("sys.argv", ["pufferl.py"])
    def test_training_performance_defaults(self):
        args = load_config("puffer_drive")
        self.assertTrue(args["train"]["compile"])
        self.assertEqual(args["train"]["precision"], "bfloat16")

    @patch("sys.argv", ["pufferl.py", "eval.render_failures_number=10"])
    def test_render_failures_number_cli_override(self):
        args = load_config("puffer_drive")
        self.assertEqual(args["eval"]["render_failures_number"], 10)

    @patch("sys.argv", ["pufferl.py", "eval.render_scenarios=true"])
    def test_render_scenarios_cli_override(self):
        args = load_config("puffer_drive")
        self.assertTrue(args["eval"]["render_scenarios"])

    @patch("sys.argv", ["pufferl.py", "eval.benchmark_sdc_num_envs=4"])
    def test_benchmark_sdc_num_envs_cli_override(self):
        args = load_config("puffer_drive")
        self.assertEqual(args["eval"]["benchmark_sdc_num_envs"], 4)

    @patch("sys.argv", ["pufferl.py", "eval.replay_failures_csv=episode_metrics.csv"])
    def test_replay_failures_csv_cli_override(self):
        args = load_config("puffer_drive")
        self.assertEqual(args["eval"]["replay_failures_csv"], "episode_metrics.csv")

    @patch("sys.argv", ["pufferl.py", "--train.learning-rate=0.5"])
    def test_old_flag_syntax_rejected_with_hint(self):
        """Pre-Hydra dashed flags must fail loudly with a migration hint."""
        with self.assertRaisesRegex(pufferlib.APIUsageError, "train.learning_rate=<value>"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "train.learning_rat=0.5"])
    def test_unknown_override_key_rejected(self):
        """Typo'd override keys must fail at compose time, not train silently."""
        with self.assertRaises(Exception):
            load_config("puffer_drive")

    def test_custom_config_yaml(self):
        """A yaml dropped into the config dir loads by name; comments
        (full-line and inline) are ignored by the YAML parser."""
        config_dir = Path(pufferlib.__file__).parent / "config"
        temp_yaml_path = config_dir / "temp_comment_test.yaml"

        yaml_content = """\
env_name: temp_comment_test
rnn_name: null
train: {}

comments:
  real_key: I exist
  # commented_key: I do not
  inline_value: 12  # inline comment
"""

        try:
            with open(temp_yaml_path, "w") as f:
                f.write(yaml_content)

            with patch("sys.argv", ["pufferl.py"]):
                args = load_config("temp_comment_test")

            self.assertEqual(args["comments"]["real_key"], "I exist")
            self.assertNotIn("commented_key", args["comments"])
            self.assertEqual(args["comments"]["inline_value"], 12)
            self.assertIsInstance(args["comments"]["inline_value"], int)

        finally:
            if os.path.exists(temp_yaml_path):
                os.remove(temp_yaml_path)


if __name__ == "__main__":
    unittest.main(verbosity=VERBOSITY)
