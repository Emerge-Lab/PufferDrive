#!/usr/bin/env python3
"""Smoke test: scripts/cluster_configs/single_agent_speed_run.yaml can be turned
into Hydra overrides that compose and final training validation both accept.

Run: python -m unittest tests/test_single_agent_yaml.py
"""

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pufferlib.config_schema import validate_puffer_drive_config  # noqa: E402
from pufferlib.pufferl import load_config  # noqa: E402

YAML_PATH = REPO_ROOT / "scripts/cluster_configs/single_agent_speed_run.yaml"


def yaml_to_argv(yaml_path: Path) -> list:
    """Convert a program_config yaml into Hydra overrides, mirroring the
    key=value formatting in submit_cluster.py."""
    cfg = yaml.safe_load(yaml_path.read_text())
    argv = ["pufferl.py"]
    for key, val in cfg.items():
        if isinstance(val, bool):
            val = str(val).lower()
        argv.append(f"{key}={val}")
    return argv


class TestSingleAgentYaml(unittest.TestCase):
    def test_yaml_passes_training_validation(self):
        """The launcher yaml must compose and satisfy final training semantics."""
        self.assertTrue(YAML_PATH.exists(), f"Missing launcher yaml: {YAML_PATH}")
        argv = yaml_to_argv(YAML_PATH)

        stderr_buf = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stderr(stderr_buf):
            try:
                args = load_config("puffer_drive")
                validate_puffer_drive_config(args, "training")
            except Exception as exc:
                self.fail(f"training validation rejected the launcher yaml: {exc}")

    def test_map_dir_points_at_existing_file_or_dir(self):
        """env.map_dir in the yaml must resolve to a real path under the repo
        so Drive() can find the .bin(s)."""
        cfg = yaml.safe_load(YAML_PATH.read_text())
        map_dir = cfg.get("env.map_dir")
        self.assertIsNotNone(map_dir, "yaml is missing env.map_dir")
        full = REPO_ROOT / map_dir
        self.assertTrue(
            full.exists(),
            f"env.map_dir does not exist on disk: {full}",
        )


if __name__ == "__main__":
    unittest.main()
