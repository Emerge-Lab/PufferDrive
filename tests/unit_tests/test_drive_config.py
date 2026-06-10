#!/usr/bin/env python3
"""
Test script for PufferDrive configuration loading.

Details:
Running the test: python -m unittest tests/test_drive_config.py
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import _ARCH_ENV_KEYS, _merge_checkpoint_arch, load_config, pufferlib

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
            # 1. pufferlib/config/default.ini (base layer)
            # 2. pufferlib/config/ocean/drive/recipes/default.yaml (composed components)
            args = load_config("puffer_drive")

            # load_config should return a populated config dict without raising.
            self.assertIsInstance(args, dict)
            self.assertTrue(len(args) > 0)

        except Exception as err:
            self.fail(f"load_config failed with an unexpected exception: {err}")

    @patch("sys.argv", ["pufferl.py", "--env.obs-lane-stride=3", "--env.obs-boundary-stride=4"])
    def test_obs_stride_cli_override(self):
        args = load_config("puffer_drive")
        self.assertEqual(args["env"]["obs_lane_stride"], 3)
        self.assertEqual(args["env"]["obs_boundary_stride"], 4)

    def test_obs_stride_validation(self):
        with self.assertRaisesRegex(ValueError, "obs_lane_stride"):
            Drive(obs_lane_stride=0)
        with self.assertRaisesRegex(ValueError, "obs_boundary_stride"):
            Drive(obs_boundary_stride=0)

    def test_checkpoint_arch_merge_keeps_obs_stride(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir)
            models_dir = exp_dir / "models"
            models_dir.mkdir()
            model_path = models_dir / "model.pt"
            model_path.touch()
            with open(exp_dir / "config.yaml", "w") as f:
                f.write("env:\n  obs_lane_stride: 5\n  obs_boundary_stride: 6\n")

            args = {"env": {"obs_lane_stride": 1, "obs_boundary_stride": 1}, "train": {}}
            _merge_checkpoint_arch(args, str(model_path))

        self.assertIn("obs_lane_stride", _ARCH_ENV_KEYS)
        self.assertIn("obs_boundary_stride", _ARCH_ENV_KEYS)
        self.assertEqual(args["env"]["obs_lane_stride"], 5)
        self.assertEqual(args["env"]["obs_boundary_stride"], 6)

    @patch("sys.argv", ["pufferl.py", "--train.learning-rate=0.5"])
    def test_cli_override(self):
        """Test that command-line arguments override INI file values."""
        # learning_rate is set by the recipe's train component, but we override it to 0.5 here
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
