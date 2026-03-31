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

from pufferlib.pufferl import load_config, pufferlib

ASSERTION_LEVEL = 1
VERBOSITY = 0


class TestDriveConfig(unittest.TestCase):
    @patch("sys.argv", ["pufferl.py"])
    def test_load_config(self):
        """
        Tests that load_config correctly loads configurations
        from the default and environment-specific INI files, without
        being affected by unittest's command-line arguments.
        """
        try:
            # The ENV_NAME 'puffer_drive' should load config from:
            # 1. pufferlib/config/default.ini
            # 2. pufferlib/config/ocean/drive.ini (and override defaults)
            args = load_config("puffer_drive")

            # Test that args is a dictionary and not empty
            self.assertIsInstance(args, dict)
            self.assertTrue(len(args) > 0)

            # Test for a value from the base default.ini ([train] section)
            # This value is not in drive.ini, so it should come from default.ini
            self.assertEqual(args["train"]["torch_deterministic"], True)

            # Test for a value from the [base] section, which is at the top level.
            # This value is overridden in drive.ini.
            self.assertEqual(args["package"], "ocean")

            # Test for a value specific to drive.ini ([env] section)
            self.assertEqual(args["env"]["num_agents"], 1024)

            # Test for a value from the [policy] section in drive.ini
            self.assertEqual(args["policy"]["hidden_size"], 256)

        except Exception as err:
            self.fail(f"load_config failed with an unexpected exception: {err}")

    @patch("sys.argv", ["pufferl.py"])
    def test_drive_ini_config(self):
        """Test that the specific config from drive.ini is correctly loaded based on ASSERTION_LEVEL"""
        args = load_config("puffer_drive")

        # --- Stable parameters (tested at all strictness levels) ---
        # These define the environment and model structure
        self.assertEqual(args["package"], "ocean")
        self.assertEqual(args["env_name"], "puffer_drive")
        self.assertEqual(args["policy_name"], "Drive")
        self.assertEqual(args["rnn_name"], "Recurrent")
        self.assertEqual(args["env"]["num_agents"], 1024)
        self.assertEqual(args["env"]["action_type"], "discrete")
        self.assertEqual(args["policy"]["input_size"], 64)
        self.assertEqual(args["policy"]["hidden_size"], 256)
        self.assertEqual(args["rnn"]["input_size"], 256)
        self.assertEqual(args["rnn"]["hidden_size"], 256)
        self.assertEqual(args["vec"]["num_workers"], 16)
        self.assertEqual(args["vec"]["num_envs"], 16)

        # --- Tunable hyperparameters (tested at high strictness) ---
        if ASSERTION_LEVEL >= 3:
            self.assertEqual(args["train"]["total_timesteps"], 3_000_000_000)
            self.assertEqual(args["train"]["batch_size"], "auto")
            self.assertEqual(args["train"]["bptt_horizon"], 91)
            self.assertEqual(args["train"]["minibatch_size"], 11648)
            self.assertEqual(args["train"]["learning_rate"], 0.001)
            self.assertEqual(args["train"]["gamma"], 0.98)
            self.assertEqual(args["train"]["gae_lambda"], 0.95)
            self.assertEqual(args["train"]["ent_coef"], 0.001)
            self.assertEqual(args["env"]["reward_vehicle_collision"], -0.5)
            self.assertEqual(args["env"]["reward_offroad_collision"], -0.2)
            self.assertEqual(args["env"]["num_maps"], 1)

    @patch("sys.argv", ["pufferl.py", "--train.learning-rate=0.5"])
    def test_cli_override(self):
        """Test that command-line arguments override INI file values."""
        # learning_rate is 0.001 in drive.ini, but we override it to 0.5 here
        args = load_config("puffer_drive")
        self.assertEqual(args["train"]["learning_rate"], 0.5)

    def test_full_line_comment_handling(self):
        """Test that full-line comments in INI files are ignored."""
        config_dir = Path(pufferlib.__file__).parent / "config"
        temp_ini_path = config_dir / "temp_comment_test.ini"

        ini_content = """
        [base]
        env_name = temp_comment_test

        [comments]
        real_key = "I exist"
        # commented_key = "I do not"
        ; another_comment = "me neither"
        """

        try:
            with open(temp_ini_path, "w") as f:
                f.write(ini_content)

            with patch("sys.argv", ["pufferl.py"]):
                args = load_config("temp_comment_test")

            self.assertEqual(args["comments"]["real_key"], "I exist")
            self.assertNotIn("commented_key", args["comments"])
            self.assertNotIn("another_comment", args["comments"])

        finally:
            if os.path.exists(temp_ini_path):
                os.remove(temp_ini_path)

    @unittest.skip("Known limitation: The parser does not support inline comments.")
    def test_inline_comment_handling(self):
        """Test that inline comments are ignored (currently a known limitation)."""
        config_dir = Path(pufferlib.__file__).parent / "config"
        temp_ini_path = config_dir / "temp_inline_comment_test.ini"

        ini_content = """
        [base]
        env_name = temp_inline_comment_test

        [comments]
        inline_value = 12 ; inline comment
        some_element = true # inline comment as well
        """

        try:
            with open(temp_ini_path, "w") as f:
                f.write(ini_content)

            with patch("sys.argv", ["pufferl.py"]):
                args = load_config("temp_inline_comment_test")

            self.assertEqual(args["comments"]["inline_value"], 12)
            self.assertIsInstance(args["comments"]["inline_value"], int)
            self.assertEqual(args["comments"]["some_element"], True)
            self.assertIsInstance(args["comments"]["some_element"], bool)

        finally:
            if os.path.exists(temp_ini_path):
                os.remove(temp_ini_path)


if __name__ == "__main__":
    unittest.main(verbosity=VERBOSITY)
