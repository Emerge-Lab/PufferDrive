#!/usr/bin/env python3
"""Experiment configs in pufferlib/config/experiment compose onto puffer_drive,
pin only keys puffer_drive.yaml defines, and land their values.

Run: python -m unittest tests.unit_tests.test_experiment_configs
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pufferlib.pufferl import load_config  # noqa: E402

EXPERIMENT_DIR = REPO_ROOT / "pufferlib/config/experiment"
BASE_CONFIG_PATH = REPO_ROOT / "pufferlib/config/puffer_drive.yaml"
REFERENCE_EXPERIMENT = "nuplan_sdc_only_bootstrap_off_2026_09_08"
DISPLACEMENT_EXPERIMENT = "nuplan_sdc_only_displacement"


def flatten(config, prefix=""):
    flat = {}
    for key, value in config.items():
        if isinstance(value, dict) and value:
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def compose_experiment(name, *overrides):
    with patch.object(sys, "argv", ["pufferl.py", f"+experiment={name}", *overrides]):
        return flatten(dict(load_config("puffer_drive")))


def experiment_values(name):
    values = yaml.safe_load((EXPERIMENT_DIR / f"{name}.yaml").read_text())
    values.pop("defaults", None)
    return flatten(values)


class TestExperimentConfigs(unittest.TestCase):
    def test_every_experiment_pins_known_keys_and_applies_them(self):
        base_keys = set(flatten(yaml.safe_load(BASE_CONFIG_PATH.read_text())))
        experiment_names = sorted(path.stem for path in EXPERIMENT_DIR.glob("*.yaml"))
        self.assertIn(REFERENCE_EXPERIMENT, experiment_names)
        for name in experiment_names:
            with self.subTest(experiment=name):
                values = experiment_values(name)
                self.assertEqual(sorted(set(values) - base_keys), [], "keys missing from puffer_drive.yaml")
                composed = compose_experiment(name)
                for key, value in values.items():
                    self.assertEqual(composed[key], value, key)

    def test_displacement_experiment_only_adds_the_displacement_penalty(self):
        reference = compose_experiment(REFERENCE_EXPERIMENT)
        displacement = compose_experiment(DISPLACEMENT_EXPERIMENT)
        changed_keys = sorted(key for key in reference if reference[key] != displacement[key])
        self.assertEqual(changed_keys, ["env.reward_displacement"])
        self.assertEqual(reference["env.reward_displacement"], 0.0)
        self.assertGreater(displacement["env.reward_displacement"], 0.0)

    def test_cli_overrides_beat_the_experiment(self):
        composed = compose_experiment(DISPLACEMENT_EXPERIMENT, "env.reward_displacement=0.2", "train.seed=3")
        self.assertEqual(composed["env.reward_displacement"], 0.2)
        self.assertEqual(composed["train.seed"], 3)


if __name__ == "__main__":
    unittest.main()
