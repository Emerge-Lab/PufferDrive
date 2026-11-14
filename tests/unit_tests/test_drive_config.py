#!/usr/bin/env python3
"""
Test script for PufferDrive configuration loading.

Details:
Running the test: python -m unittest tests/test_drive_config.py
"""

import io
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.config_schema import validate_puffer_drive_config
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import known_cli_overrides, load_config, pufferlib

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
        except Exception as err:
            self.fail(f"load_config failed with an unexpected exception: {err}")

    @patch("sys.argv", ["pufferl.py", "env.obs_lane_stride=3", "env.obs_boundary_stride=4"])
    def test_obs_stride_cli_override(self):
        args = load_config("puffer_drive")
        self.assertEqual(args["env"]["obs_lane_stride"], 3)
        self.assertEqual(args["env"]["obs_boundary_stride"], 4)

    def test_obs_stride_validation(self):
        with patch.object(sys, "argv", ["pufferl.py"]):
            args = load_config("puffer_drive")
        args["env"]["obs_lane_stride"] = 0
        with self.assertRaisesRegex(pufferlib.APIUsageError, "obs_lane_stride"):
            validate_puffer_drive_config(args, "test")
        args["env"]["obs_lane_stride"] = 1
        args["env"]["obs_boundary_stride"] = 0
        with self.assertRaisesRegex(pufferlib.APIUsageError, "obs_boundary_stride"):
            validate_puffer_drive_config(args, "test")

    def test_speed_limit_randomization_validation(self):
        with self.assertRaisesRegex(ValueError, "speed_limit_random_prob"):
            Drive(speed_limit_random_prob=1.5)
        with self.assertRaisesRegex(ValueError, "speed_limit_random_delta_mps"):
            Drive(speed_limit_random_delta_mps=-1.0)
        with self.assertRaisesRegex(ValueError, "min <= max"):
            Drive(speed_limit_random_min_mps=20.0, speed_limit_random_max_mps=10.0)

    def test_conditioning_speed_scale_validation(self):
        with self.assertRaisesRegex(ValueError, "conditioning_speed_scale"):
            Drive(conditioning_speed_scale=0.9)

    @patch("sys.argv", ["pufferl.py", "env.obs_traffic_light_yellow_remaining=false"])
    def test_yellow_remaining_cli_override(self):
        args = load_config("puffer_drive")
        self.assertFalse(args["env"]["obs_traffic_light_yellow_remaining"])

    def test_yellow_remaining_obs_widens_traffic_control_rows(self):
        plain = Drive(config_only=True)
        with_yellow = Drive(config_only=True, obs_traffic_light_yellow_remaining=True)
        self.assertEqual(
            with_yellow.traffic_control_features,
            plain.traffic_control_features + binding.TRAFFIC_CONTROL_YELLOW_REMAINING_FEATURES,
        )
        self.assertEqual(
            with_yellow.num_obs - plain.num_obs,
            plain.obs_slots_traffic_controls_n * binding.TRAFFIC_CONTROL_YELLOW_REMAINING_FEATURES,
        )

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

    @patch("sys.argv", ["pufferl.py", "--train.learning-rate=0.5"])
    def test_old_flag_syntax_rejected_with_hint(self):
        """Pre-Hydra dashed flags must fail loudly with a migration hint."""
        with self.assertRaisesRegex(pufferlib.APIUsageError, "train.learning_rate=<value>"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "train.learning_rat=0.5", "train.learning_rate=0.25"])
    def test_unknown_override_key_warns_and_is_ignored(self):
        """A job launched before a setting was removed must still run: its stale override only warns."""
        with patch("sys.stderr", new_callable=io.StringIO) as captured_stderr:
            args = load_config("puffer_drive")
        self.assertIn("ignoring override 'train.learning_rat=0.5'", captured_stderr.getvalue())
        self.assertNotIn("learning_rat", args["train"])
        self.assertEqual(args["train"]["learning_rate"], 0.25)

    @patch("sys.argv", ["pufferl.py", "env.removed_setting=0", "removed_section.setting=1", "removed=1", "wandb=true"])
    def test_known_cli_overrides_drop_every_unknown_setting(self):
        with patch("sys.stderr", new_callable=io.StringIO) as captured_stderr:
            overrides = known_cli_overrides("puffer_drive")
        self.assertEqual(overrides, ["wandb=true"])
        self.assertEqual(captured_stderr.getvalue().count("WARNING: ignoring override"), 3)


if __name__ == "__main__":
    unittest.main(verbosity=VERBOSITY)
