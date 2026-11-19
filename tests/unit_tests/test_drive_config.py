"""OpenGigaFlow config loading: Hydra CLI overrides and the env-side validation of loaded values."""

import io
import unittest
from unittest.mock import patch

import opengigaflow
from opengigaflow.config_schema import validate_opengigaflow_config
from opengigaflow.ocean.drive import binding
from opengigaflow.ocean.drive.drive import Drive
from opengigaflow.rl import known_cli_overrides, load_config


VERBOSITY = 0


class TestDriveConfig(unittest.TestCase):
    @patch(
        "sys.argv",
        [
            "rl.py",
            "env.obs_lane_stride=3",
            "env.obs_boundary_stride=4",
            "env.obs_traffic_light_yellow_remaining=false",
            "train.learning_rate=0.5",
        ],
    )
    def test_cli_overrides_win_over_the_config_file(self):
        args = load_config("opengigaflow")
        self.assertEqual(args["env"]["obs_lane_stride"], 3)
        self.assertEqual(args["env"]["obs_boundary_stride"], 4)
        self.assertFalse(args["env"]["obs_traffic_light_yellow_remaining"])
        self.assertEqual(args["train"]["learning_rate"], 0.5)

    def test_obs_stride_validation(self):
        args = load_config("opengigaflow", overrides=[])
        args["env"]["obs_lane_stride"] = 0
        with self.assertRaisesRegex(opengigaflow.APIUsageError, "obs_lane_stride"):
            validate_opengigaflow_config(args, "test")
        args["env"]["obs_lane_stride"] = 1
        args["env"]["obs_boundary_stride"] = 0
        with self.assertRaisesRegex(opengigaflow.APIUsageError, "obs_boundary_stride"):
            validate_opengigaflow_config(args, "test")

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

    def test_training_performance_defaults(self):
        args = load_config("opengigaflow", overrides=[])
        self.assertTrue(args["train"]["compile"])
        self.assertEqual(args["train"]["precision"], "bfloat16")

    @patch("sys.argv", ["rl.py", "--train.learning-rate=0.5"])
    def test_old_flag_syntax_rejected_with_hint(self):
        """Pre-Hydra dashed flags must fail loudly with a migration hint."""
        with self.assertRaisesRegex(opengigaflow.APIUsageError, "train.learning_rate=<value>"):
            load_config("opengigaflow")

    @patch("sys.argv", ["rl.py", "train.learning_rat=0.5", "train.learning_rate=0.25"])
    def test_unknown_override_key_warns_and_is_ignored(self):
        """A job launched before a setting was removed must still run: its stale override only warns."""
        with patch("sys.stderr", new_callable=io.StringIO) as captured_stderr:
            args = load_config("opengigaflow")
        self.assertIn("ignoring override 'train.learning_rat=0.5'", captured_stderr.getvalue())
        self.assertNotIn("learning_rat", args["train"])
        self.assertEqual(args["train"]["learning_rate"], 0.25)

    @patch("sys.argv", ["rl.py", "env.removed_setting=0", "removed_section.setting=1", "removed=1", "wandb=true"])
    def test_known_cli_overrides_drop_every_unknown_setting(self):
        with patch("sys.stderr", new_callable=io.StringIO) as captured_stderr:
            overrides = known_cli_overrides("opengigaflow")
        self.assertEqual(overrides, ["wandb=true"])
        self.assertEqual(captured_stderr.getvalue().count("WARNING: ignoring override"), 3)


if __name__ == "__main__":
    unittest.main(verbosity=VERBOSITY)
