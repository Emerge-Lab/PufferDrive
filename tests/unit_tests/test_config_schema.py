#!/usr/bin/env python3
"""Structured-config schema tests: load_config validates the env section of
puffer_drive against pufferlib.config_schema.DriveEnvConfig at load time.

Run: python -m unittest tests.unit_tests.test_config_schema
"""

import os
import sys
import unittest
from unittest.mock import patch

from omegaconf.errors import ConfigKeyError, ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pufferlib.config_schema import Controller, InfractionBehavior, TargetType
from pufferlib.ocean.drive import binding
from pufferlib.pufferl import load_config


class TestConfigSchema(unittest.TestCase):
    @patch("sys.argv", ["pufferl.py"])
    def test_valid_config_loads_with_plain_strings(self):
        """Schema validation must not change the plain-dict contract: enum
        fields come back as their string names, not Enum members."""
        args = load_config("puffer_drive")
        self.assertIsInstance(args["env"]["collision_behavior"], str)
        self.assertIn(args["env"]["collision_behavior"], ("ignore", "stop", "remove"))
        self.assertIsInstance(args["env"]["control_mode"], str)

    @patch("sys.argv", ["pufferl.py", "env.collision_behavior=sotp"])
    def test_enum_typo_fails_at_load(self):
        with self.assertRaisesRegex(ValidationError, "expected one of"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "env.num_agents=lots"])
    def test_wrong_type_fails_at_load(self):
        with self.assertRaisesRegex(ValidationError, "could not be converted"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "+env.collission_behavior=stop"])
    def test_unknown_env_key_fails_at_load(self):
        """Keys force-added with + that the schema doesn't declare are
        rejected (plain overrides of unknown keys already fail in compose)."""
        with self.assertRaisesRegex(ConfigKeyError, "collission_behavior"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "env.collision_behavior=1"])
    def test_enum_accepts_c_int_value(self):
        """OmegaConf coerces enum *values* as well as names. This is safe
        exactly because the schema ints mirror drive.h (see the sync test
        below) — the coerced member round-trips to the right name."""
        args = load_config("puffer_drive")
        self.assertEqual(args["env"]["collision_behavior"], "stop")

    @patch("sys.argv", ["pufferl.py"])
    def test_env_without_schema_loads_unvalidated(self):
        args = load_config("default")
        self.assertIsInstance(args, dict)

    def test_schema_enum_values_match_binding_constants(self):
        """drive.h #defines are the source of truth for the ints; the schema
        enums must never drift from them."""
        self.assertEqual(InfractionBehavior.ignore.value, binding.IGNORE_INFRACTION)
        self.assertEqual(InfractionBehavior.stop.value, binding.STOP_AGENT)
        self.assertEqual(InfractionBehavior.remove.value, binding.REMOVE_AGENT)
        self.assertEqual(Controller.static.value, binding.CONTROLLER_STATIC)
        self.assertEqual(Controller.policy.value, binding.CONTROLLER_POLICY)
        self.assertEqual(Controller.replay.value, binding.CONTROLLER_REPLAY)
        self.assertEqual(Controller.idm.value, binding.CONTROLLER_IDM)
        self.assertEqual(TargetType.static.value, binding.TARGET_STATIC)
        self.assertEqual(TargetType.dynamic.value, binding.TARGET_DYNAMIC)


if __name__ == "__main__":
    unittest.main()
