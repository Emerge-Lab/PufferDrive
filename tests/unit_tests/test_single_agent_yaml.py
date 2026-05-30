#!/usr/bin/env python3
"""Smoke test: scripts/cluster_configs/single_agent_speed_run.yaml can be turned
into pufferl CLI flags that argparse accepts without unrecognized-argument
errors, and the parsed config carries the yaml's values into the right places.

Run: python -m unittest tests/test_single_agent_yaml.py
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pufferlib.pufferl import load_config  # noqa: E402

# Boolean store_true flags in pufferl's argparser. Mirrors submit_cluster.py.
BOOLEAN_FLAGS = {"wandb", "neptune", "tb"}

YAML_PATH = REPO_ROOT / "scripts/cluster_configs/single_agent_speed_run.yaml"


def yaml_to_argv(yaml_path: Path) -> list:
    """Convert a program_config yaml into pufferl CLI flags, mirroring the
    underscore→dash conversion and boolean handling in submit_cluster.py."""
    cfg = yaml.safe_load(yaml_path.read_text())
    argv = ["pufferl.py"]
    for key, val in cfg.items():
        cli_key = key.replace("_", "-")
        if key in BOOLEAN_FLAGS:
            if val in (True, "True", "true", 1, "1"):
                argv.append(f"--{cli_key}")
            continue
        argv.extend([f"--{cli_key}", str(val)])
    return argv


class TestSingleAgentYaml(unittest.TestCase):
    def test_yaml_parses_via_pufferl_argparser(self):
        """Every key in the launcher yaml must correspond to a registered
        argparse flag. If any key is unrecognized, argparse calls
        parser.error() which sys.exits(2)."""
        self.assertTrue(YAML_PATH.exists(), f"Missing launcher yaml: {YAML_PATH}")
        argv = yaml_to_argv(YAML_PATH)

        stderr_buf = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stderr(stderr_buf):
            try:
                load_config("puffer_drive")
            except SystemExit as exc:
                self.fail(
                    f"load_config exited ({exc.code}) on the launcher yaml — argparse stderr:\n{stderr_buf.getvalue()}"
                )

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
