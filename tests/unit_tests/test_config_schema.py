#!/usr/bin/env python3
"""PufferDrive schema and final resolved-config checker tests.

Run: python -m unittest tests.unit_tests.test_config_schema
"""

import os
import re
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pufferlib
from pufferlib.config_schema import (
    ActionType,
    Controller,
    ControlMode,
    DynamicsModel,
    GoalRegen,
    GoalSource,
    InfractionBehavior,
    InitMode,
    NonVehicleController,
    SimulationMode,
    validate_puffer_drive_config,
)
from pufferlib.ocean.drive import binding
from pufferlib.pufferl import load_config


def _screaming_snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


# Enum classes checked against binding, and how to turn a member's Python
# name into the suffix of its C constant. Every class defaults to its member
# name verbatim; ControlMode's members stutter the group name in Python
# (control_vehicles, not vehicles) so the leading "control_" is stripped
# before joining, matching drive.h's CONTROL_MODE_VEHICLES etc.
_DRIFT_CHECKED_ENUMS = [
    (SimulationMode, lambda member: member.name),
    (ActionType, lambda member: member.name),
    (DynamicsModel, lambda member: member.name),
    (InfractionBehavior, lambda member: member.name),
    (ControlMode, lambda member: member.name.removeprefix("control_")),
    (Controller, lambda member: member.name),
    (InitMode, lambda member: member.name),
    (GoalRegen, lambda member: member.name),
    (GoalSource, lambda member: member.name),
]


class TestConfigSchema(unittest.TestCase):
    @patch("sys.argv", ["pufferl.py"])
    def test_valid_config_loads_with_plain_strings(self):
        """Final validation must not change the plain-dict contract: enum
        fields come back as their string names, not Enum members."""
        args = load_config("puffer_drive")
        self.assertIsNone(validate_puffer_drive_config(args, "test"))
        self.assertIsInstance(args["env"]["collision_behavior"], str)
        self.assertIn(args["env"]["collision_behavior"], ("ignore", "stop", "remove"))
        self.assertIsInstance(args["env"]["control_mode"], str)

    @patch("sys.argv", ["pufferl.py", "env.collision_behavior=sotp"])
    def test_enum_typo_fails_at_load(self):
        with self.assertRaisesRegex(pufferlib.APIUsageError, "collision_behavior"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "env.num_agents=lots"])
    def test_wrong_type_fails_at_load(self):
        with self.assertRaisesRegex(pufferlib.APIUsageError, "num_agents"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "+env.collission_behavior=stop"])
    def test_unknown_env_key_fails_at_load(self):
        """Keys force-added with + that the schema doesn't declare are
        rejected as soon as Hydra composition finishes."""
        with self.assertRaisesRegex(pufferlib.APIUsageError, "collission_behavior"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "env.collision_behavior=1"])
    def test_enum_accepts_c_int_value(self):
        args = load_config("puffer_drive")
        self.assertEqual(args["env"]["collision_behavior"], "stop")

    def test_schema_enums_match_binding_constants(self):
        """drive.h #defines are the source of truth for the ints. Naming
        convention (see config_schema.py docstring): every C constant is
        `<ENUM_CLASS_SCREAMING_SNAKE>_<MEMBER_UPPER>`. This walks every enum
        class instead of hand-picking members, so an enum added to
        config_schema.py without a matching binding constant — or with a
        mismatched value — fails here without needing a new assert line."""
        for enum_cls, member_suffix in _DRIFT_CHECKED_ENUMS:
            group = _screaming_snake(enum_cls.__name__)
            for member in enum_cls:
                const_name = f"{group}_{member_suffix(member).upper()}"
                self.assertTrue(
                    hasattr(binding, const_name),
                    f"binding.{const_name} not found for {enum_cls.__name__}.{member.name} "
                    "-- was the C #define renamed without updating config_schema.py, "
                    "or is it missing from env_binding.h's PyModule_AddIntConstant calls?",
                )
                self.assertEqual(
                    getattr(binding, const_name),
                    member.value,
                    f"binding.{const_name} != {enum_cls.__name__}.{member.name}.value",
                )

    def test_non_vehicle_controller_matches_controller_constants(self):
        """NonVehicleController reuses Controller's C constants; only its
        config-only 'auto' sentinel (-1) has no C counterpart (drive.py
        resolves it to a concrete Controller before it reaches C)."""
        for member in NonVehicleController:
            if member.name == "auto":
                continue
            const_name = f"CONTROLLER_{member.name.upper()}"
            self.assertEqual(getattr(binding, const_name), member.value)


if __name__ == "__main__":
    unittest.main()
