"""scripts/cluster_configs/single_agent_speed_run.yaml composes as Hydra overrides and passes training validation."""

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import yaml

from opengigaflow.config_schema import validate_opengigaflow_config
from opengigaflow.rl import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "scripts/cluster_configs/single_agent_speed_run.yaml"


def yaml_to_argv(yaml_path: Path) -> list:
    """A program_config yaml as Hydra overrides, in submit_cluster.py's key=value format."""
    cfg = yaml.safe_load(yaml_path.read_text())
    argv = ["rl.py"]
    for key, val in cfg.items():
        if isinstance(val, bool):
            val = str(val).lower()
        argv.append(f"{key}={val}")
    return argv


class TestSingleAgentYaml(unittest.TestCase):
    def test_yaml_parses_via_hydra_compose(self):
        """The launcher yaml must compose and satisfy final training semantics."""
        self.assertTrue(YAML_PATH.exists(), f"Missing launcher yaml: {YAML_PATH}")
        argv = yaml_to_argv(YAML_PATH)

        with patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()):
            validate_opengigaflow_config(load_config("opengigaflow"), "training")

    def test_map_dir_points_at_existing_file_or_dir(self):
        """env.map_dir in the yaml resolves to a real path under the repo, so Drive() finds the .bin(s)."""
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
