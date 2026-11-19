"""OpenGigaFlow config schema and final resolved-config checks."""

import re
import unittest
from unittest.mock import patch

import opengigaflow
from opengigaflow.config_schema import (
    Controller,
    ControlMode,
    GoalRegen,
    GoalSource,
    InfractionBehavior,
    InitMode,
    NonVehicleController,
    SimulationMode,
    normalize_opengigaflow_config,
    validate_opengigaflow_config,
)
from opengigaflow.ocean.drive import binding
from opengigaflow.rl import load_config


def _screaming_snake(name):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).upper()


_DRIFT_CHECKED_ENUMS = (SimulationMode, InfractionBehavior, ControlMode, Controller, InitMode, GoalRegen, GoalSource)


class TestConfigSchema(unittest.TestCase):
    def test_valid_config_loads_with_plain_strings(self):
        """Final validation keeps the plain-dict contract: enum fields come back as strings, not Enum members."""
        args = load_config("opengigaflow", overrides=[])
        self.assertIsNone(validate_opengigaflow_config(args, "test"))
        self.assertIsInstance(args["env"]["collision_behavior"], str)
        self.assertIn(args["env"]["collision_behavior"], ("ignore", "stop"))
        self.assertIsInstance(args["env"]["control_mode"], str)
        for field_name in (
            "use_map_cache",
            "preload_map_cache",
            "use_neighbor_cache",
            "termination_mode",
        ):
            self.assertIsInstance(args["env"][field_name], bool)

    @patch("sys.argv", ["rl.py", "env.use_map_cache=false"])
    def test_boolean_cli_overrides_load(self):
        args = load_config("opengigaflow")

        self.assertIs(args["env"]["use_map_cache"], False)

    def test_legacy_zero_or_one_values_normalize_to_booleans(self):
        args = load_config("opengigaflow", overrides=[])
        args["env"].update(
            {
                "use_map_cache": 1,
                "preload_map_cache": 0,
                "use_neighbor_cache": 0,
                "termination_mode": 1,
            }
        )

        normalized = normalize_opengigaflow_config(args, "test")

        self.assertIs(normalized["env"]["use_map_cache"], True)
        self.assertIs(normalized["env"]["preload_map_cache"], False)
        self.assertIs(normalized["env"]["use_neighbor_cache"], False)
        self.assertIs(normalized["env"]["termination_mode"], True)

    @patch("sys.argv", ["rl.py", "env.collision_behavior=sotp"])
    def test_enum_typo_fails_at_load(self):
        with self.assertRaisesRegex(
            opengigaflow.APIUsageError,
            r"at env\.collision_behavior while loading: Invalid value 'sotp'",
        ):
            load_config("opengigaflow")

    @patch("sys.argv", ["rl.py", "env.num_agents=lots"])
    def test_wrong_type_fails_at_load(self):
        with self.assertRaisesRegex(
            opengigaflow.APIUsageError,
            r"at env\.num_agents while loading: Value 'lots'",
        ):
            load_config("opengigaflow")

    @patch("sys.argv", ["rl.py", "+env.collission_behavior=stop"])
    def test_unknown_env_key_fails_at_load(self):
        """Keys force-added with + that the schema lacks are rejected as soon as Hydra composition finishes."""
        with self.assertRaisesRegex(
            opengigaflow.APIUsageError,
            r"at env\.collission_behavior while loading: Key 'collission_behavior'",
        ):
            load_config("opengigaflow")

    @patch("sys.argv", ["rl.py", "env.collision_behavior=1"])
    def test_enum_accepts_c_int_value(self):
        args = load_config("opengigaflow")
        self.assertEqual(args["env"]["collision_behavior"], "stop")

    def test_schema_enums_match_binding_constants(self):
        """constants.h #defines own the ints; each is `<ENUM_CLASS_SCREAMING_SNAKE>_<MEMBER_UPPER>`."""
        for enum_cls in _DRIFT_CHECKED_ENUMS:
            group = _screaming_snake(enum_cls.__name__)
            for member in enum_cls:
                # ControlMode members repeat the group name: control_vehicles -> CONTROL_MODE_VEHICLES
                const_name = f"{group}_{member.name.removeprefix('control_').upper()}"
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
        """NonVehicleController reuses Controller's C constants; only its config-only 'auto' (-1) has none."""
        for member in NonVehicleController:
            if member.name == "auto":
                continue
            const_name = f"CONTROLLER_{member.name.upper()}"
            self.assertEqual(getattr(binding, const_name), member.value)

    def test_training_batch_relationships_are_checked(self):
        args = load_config("opengigaflow", overrides=[])

        invalid = {**args, "train": {**args["train"], "batch_size": "auto", "bptt_horizon": "auto"}}
        with self.assertRaisesRegex(opengigaflow.APIUsageError, "batch_size and bptt_horizon"):
            validate_opengigaflow_config(invalid, "test")

        invalid = {**args, "train": {**args["train"], "minibatch_size": 10, "max_minibatch_size": 6}}
        with self.assertRaisesRegex(opengigaflow.APIUsageError, "max_minibatch_size"):
            validate_opengigaflow_config(invalid, "test")

        invalid = {**args, "train": {**args["train"], "batch_size": 8, "minibatch_size": 16}}
        with self.assertRaisesRegex(opengigaflow.APIUsageError, "train.batch_size"):
            validate_opengigaflow_config(invalid, "test")

        invalid = {
            **args,
            "train": {
                **args["train"],
                "batch_size": 32,
                "minibatch_size": 10,
                "max_minibatch_size": 10,
                "bptt_horizon": 4,
            },
        }
        with self.assertRaisesRegex(opengigaflow.APIUsageError, "bptt_horizon"):
            validate_opengigaflow_config(invalid, "test")

        # Test derived batch_size < minibatch_size when batch_size is auto
        invalid = {
            **args,
            "vec": {**args["vec"], "num_envs": 1},
            "env": {**args["env"], "num_agents": 4},
            "train": {
                **args["train"],
                "batch_size": "auto",
                "bptt_horizon": 2,
                "minibatch_size": 1024,
                "max_minibatch_size": 1024,
            },
        }
        with self.assertRaisesRegex(opengigaflow.APIUsageError, "derived batch_size"):
            validate_opengigaflow_config(invalid, "test")

        # Test total_agents > segments when neither is auto
        invalid = {
            **args,
            "vec": {**args["vec"], "num_envs": 16},
            "env": {**args["env"], "num_agents": 4},
            "train": {
                **args["train"],
                "batch_size": 32,
                "bptt_horizon": 2,
                "minibatch_size": 16,
                "max_minibatch_size": 16,
            },
        }
        with self.assertRaisesRegex(opengigaflow.APIUsageError, "total agents .* must be <= segments"):
            validate_opengigaflow_config(invalid, "test")

    def test_semantic_error_separates_field_path_from_context(self):
        args = load_config("opengigaflow", overrides=[])
        args["env"]["obs_lane_stride"] = 0

        with self.assertRaisesRegex(
            opengigaflow.APIUsageError,
            r"at env\.obs_lane_stride during training: must be a positive integer",
        ):
            validate_opengigaflow_config(args, "training")

    def test_failure_replay_rejects_scenario_rendering(self):
        args = load_config("opengigaflow", overrides=[])
        args["eval"]["failure_replay_csv"] = "failures.csv"
        args["eval"]["render_filter"] = "collision"
        args["eval"]["render_scenarios"] = True

        with self.assertRaisesRegex(opengigaflow.APIUsageError, "scenario rendering"):
            validate_opengigaflow_config(args, "test")

    def test_preload_map_cache_requires_use_map_cache(self):
        args = load_config("opengigaflow", overrides=[])
        args["env"]["preload_map_cache"] = True
        args["env"]["use_map_cache"] = False

        with self.assertRaisesRegex(opengigaflow.APIUsageError, "preload_map_cache"):
            validate_opengigaflow_config(args, "test")

    def test_partner_blindness_prob_min_must_not_exceed_max(self):
        args = load_config("opengigaflow", overrides=[])
        args["env"]["partner_blindness_prob_min"] = 0.5
        args["env"]["partner_blindness_prob"] = 0.1

        with self.assertRaisesRegex(opengigaflow.APIUsageError, "partner_blindness_prob_min"):
            validate_opengigaflow_config(args, "test")


if __name__ == "__main__":
    unittest.main()
