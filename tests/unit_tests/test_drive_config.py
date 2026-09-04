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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.config_schema import validate_puffer_drive_config
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

    @patch("sys.argv", ["pufferl.py", "train.learning_rate=0.5"])
    def test_cli_override(self):
        """Test that Hydra CLI overrides win over the config file values."""
        args = load_config("puffer_drive")
        self.assertEqual(args["train"]["learning_rate"], 0.5)

    @patch("sys.argv", ["pufferl.py"])
    def test_training_performance_defaults(self):
        args = load_config("puffer_drive")
        self.assertFalse(args["train"]["compile"])
        self.assertEqual(args["train"]["precision"], "bfloat16")

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


if __name__ == "__main__":
    unittest.main(verbosity=VERBOSITY)
