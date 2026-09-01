#!/usr/bin/env python3
"""PufferDrive schema and final resolved-config checker tests.

Run: python -m unittest tests.unit_tests.test_config_schema
"""

import copy
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
    normalize_puffer_drive_benchmarks,
    normalize_puffer_drive_config,
    validate_puffer_drive_config,
    validate_puffer_drive_resources,
)
from pufferlib.ocean.drive import binding
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark
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
    def test_schema_normalizes_enum_integer_value(self):
        args = load_config("puffer_drive")
        self.assertEqual(args["env"]["collision_behavior"], "stop")

    @patch("sys.argv", ["pufferl.py", "+policy.encoder_actvation=relu"])
    def test_unknown_non_environment_key_fails_at_load(self):
        with self.assertRaisesRegex(pufferlib.APIUsageError, "encoder_actvation"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "train.optimizer=sgd"])
    def test_config_only_enum_fails_at_load(self):
        with self.assertRaisesRegex(pufferlib.APIUsageError, "optimizer"):
            load_config("puffer_drive")

    @patch("sys.argv", ["pufferl.py", "train.optimizer=muon"])
    def test_supported_muon_optimizer_loads(self):
        args = load_config("puffer_drive")
        self.assertEqual(args["train"]["optimizer"], "muon")

    @patch("sys.argv", ["pufferl.py"])
    def test_non_recurrent_policy_allows_zero_layer_backbone(self):
        args = load_config("puffer_drive")
        args["policy"]["backbone_num_layers"] = 0
        validate_puffer_drive_config(args, "test")

        args["policy"]["backbone_num_layers"] = -1
        with self.assertRaisesRegex(pufferlib.APIUsageError, "backbone_num_layers"):
            validate_puffer_drive_config(args, "test")

    @patch("sys.argv", ["pufferl.py"])
    def test_recurrent_policy_requires_backbone_layer(self):
        args = load_config("puffer_drive")
        args["rnn_name"] = "Recurrent"
        args["policy"]["backbone_num_layers"] = 0
        with self.assertRaisesRegex(pufferlib.APIUsageError, "backbone_num_layers"):
            validate_puffer_drive_config(args, "test")

    @patch("sys.argv", ["pufferl.py"])
    def test_validation_is_idempotent_and_does_not_mutate_input(self):
        args = load_config("puffer_drive")
        original = copy.deepcopy(args)
        first = validate_puffer_drive_config(args, "test")
        second = validate_puffer_drive_config(args, "test")
        self.assertEqual(args, original)
        self.assertIsNone(first)
        self.assertIsNone(second)

    @patch("sys.argv", ["pufferl.py"])
    def test_boolean_is_rejected_for_integer_field(self):
        args = load_config("puffer_drive")
        args["env"]["num_agents"] = True
        with self.assertRaisesRegex(pufferlib.APIUsageError, "num_agents"):
            validate_puffer_drive_config(args, "test")

    @patch("sys.argv", ["pufferl.py"])
    def test_final_schema_rejects_unknown_keys_in_fixed_sections(self):
        args = load_config("puffer_drive")
        for section in (None, "vec", "env", "policy", "rnn", "train", "eval"):
            with self.subTest(section=section):
                invalid = copy.deepcopy(args)
                target = invalid if section is None else invalid[section]
                target["unexpected_key"] = 1
                with self.assertRaisesRegex(pufferlib.APIUsageError, "unexpected_key"):
                    normalize_puffer_drive_config(invalid, "test")

    @patch("sys.argv", ["pufferl.py"])
    def test_final_schema_rejects_missing_required_fields(self):
        args = load_config("puffer_drive")
        del args["train"]["learning_rate"]
        with self.assertRaisesRegex(pufferlib.APIUsageError, "train.learning_rate"):
            normalize_puffer_drive_config(args, "test")

    @patch("sys.argv", ["pufferl.py"])
    def test_field_constraints_apply_across_nested_sections(self):
        args = load_config("puffer_drive")
        invalid_values = (
            (("num_scenarios",), 0),
            (("vec", "num_envs"), 0),
            (("env", "dt"), 0.0),
            (("policy", "actor_num_layers"), -1),
            (("rnn", "input_size"), 0),
            (("train", "gamma"), 1.1),
            (("eval", "num_agents"), 0),
        )
        for path, invalid_value in invalid_values:
            with self.subTest(path=path):
                invalid = copy.deepcopy(args)
                target = invalid
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = invalid_value
                with self.assertRaisesRegex(pufferlib.APIUsageError, path[-1]):
                    validate_puffer_drive_config(invalid, "test")

    @patch("sys.argv", ["pufferl.py"])
    def test_observation_categories_allow_zero_and_reject_negative(self):
        args = load_config("puffer_drive")
        args["env"]["obs_slots_lane_n"] = 0
        args["env"]["obs_slots_boundary_n"] = 0
        args["env"]["obs_slots_partners_n"] = 0
        args["env"]["obs_slots_traffic_controls_n"] = 0
        validate_puffer_drive_config(args, "test")

        slot_fields = (
            "obs_slots_lane_n",
            "obs_slots_boundary_n",
            "obs_slots_partners_n",
            "obs_slots_traffic_controls_n",
        )
        for field_name in slot_fields:
            with self.subTest(field_name=field_name):
                invalid = copy.deepcopy(args)
                invalid["env"][field_name] = -1
                with self.assertRaisesRegex(pufferlib.APIUsageError, field_name):
                    validate_puffer_drive_config(invalid, "test")

    @patch("sys.argv", ["pufferl.py", "train.seed=null"])
    def test_training_seed_allows_none(self):
        args = load_config("puffer_drive")
        self.assertIsNone(args["train"]["seed"])
        validate_puffer_drive_config(args, "test")

    @patch("sys.argv", ["pufferl.py"])
    def test_goal_speed_can_exceed_physical_speed_cap(self):
        args = load_config("puffer_drive")
        args["env"]["base_max_speed_mps"] = 20.0
        args["env"]["goal_speed"] = 300.0
        validate_puffer_drive_config(args, "test")

        for invalid_goal_speed in (-1.0, float("inf"), float("nan")):
            with self.subTest(goal_speed=invalid_goal_speed):
                invalid = copy.deepcopy(args)
                invalid["env"]["goal_speed"] = invalid_goal_speed
                with self.assertRaisesRegex(pufferlib.APIUsageError, "goal_speed"):
                    validate_puffer_drive_config(invalid, "test")

    @patch("sys.argv", ["pufferl.py"])
    def test_optional_eval_is_allowed_only_when_training_evaluation_is_disabled(self):
        args = load_config("puffer_drive")
        args["eval"] = None
        validate_puffer_drive_config(args, "training")
        with self.assertRaisesRegex(pufferlib.APIUsageError, "complete eval section"):
            validate_puffer_drive_config(args, "evaluation")

    @patch("sys.argv", ["pufferl.py"])
    def test_load_config_does_not_apply_ddp_derivation(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "4"}):
            first = load_config("puffer_drive")
            second = load_config("puffer_drive")
        self.assertEqual(first["train"]["total_timesteps"], second["train"]["total_timesteps"])
        self.assertEqual(first["train"]["total_timesteps"], 500_000_000_000)

    @patch("sys.argv", ["pufferl.py"])
    def test_cli_values_are_validated_after_benchmark_merge(self):
        args = load_config("puffer_drive")
        benchmark = {
            "name": "merge_order",
            "seed": 42,
            "num_scenarios": 1,
            "env": {"simulation_mode": "gigaflow", "control_mode": "control_vehicles"},
        }
        environment_config = {"eval_mode": 1, "dt": -1.0}
        original_args = copy.deepcopy(args)
        original_benchmark = copy.deepcopy(benchmark)
        original_environment_config = copy.deepcopy(environment_config)

        with self.assertRaisesRegex(pufferlib.APIUsageError, "env.dt"):
            drive_benchmark.build_benchmark_args(args, benchmark, environment_config)

        checked = drive_benchmark.build_benchmark_args(
            args,
            benchmark,
            environment_config,
            ["env.dt=0.2", "env.eval_mode=0"],
        )
        self.assertEqual(checked["env"]["dt"], 0.2)
        self.assertEqual(checked["env"]["eval_mode"], 1)
        self.assertEqual(args, original_args)
        self.assertEqual(benchmark, original_benchmark)
        self.assertEqual(environment_config, original_environment_config)

    @patch("sys.argv", ["pufferl.py"])
    def test_benchmark_builder_validates_final_resources(self):
        args = load_config("puffer_drive")
        benchmark = {
            "name": "missing_maps",
            "seed": 42,
            "num_scenarios": 1,
            "env": {"simulation_mode": "gigaflow", "control_mode": "control_vehicles"},
        }
        missing_map_dir = os.path.join(os.path.dirname(__file__), "missing_benchmark_maps")
        self.assertFalse(os.path.exists(missing_map_dir))

        with self.assertRaisesRegex(pufferlib.APIUsageError, "env.map_dir"):
            drive_benchmark.build_benchmark_args(
                args,
                benchmark,
                {"eval_mode": 1, "map_dir": missing_map_dir},
            )

    @patch("sys.argv", ["pufferl.py"])
    def test_resource_validation_only_bounds_num_maps_by_available_files(self):
        args = load_config("puffer_drive")
        args["env"]["simulation_mode"] = "replay"
        args["env"]["map_dir"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "pufferlib",
            "resources",
            "drive",
            "binaries",
            "sdc_replay_test",
        )
        args["env"]["num_maps"] = 1
        args["num_scenarios"] = 3

        benchmarks = normalize_puffer_drive_benchmarks(
            {},
            [
                {
                    "name": "repeated_map",
                    "seed": 42,
                    "num_scenarios": 3,
                    "env": {
                        "simulation_mode": "replay",
                        "control_mode": "control_sdc_only",
                        "map_dir": args["env"]["map_dir"],
                        "num_maps": 1,
                    },
                }
            ],
            "evaluation",
        )
        self.assertEqual(benchmarks[0]["num_scenarios"], 3)
        validate_puffer_drive_resources(args, "evaluation.test")

        args["env"]["num_maps"] = 2
        with self.assertRaisesRegex(pufferlib.APIUsageError, "env.num_maps"):
            validate_puffer_drive_resources(args, "evaluation.test")

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
