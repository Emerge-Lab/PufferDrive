"""Tests for the render asset root passed from Python to the C env.

Render assets (.glb models) are resolved through the resource_root env-init
kwarg, an absolute path derived from the installed pufferlib package. Two
invariants guard against past failure modes:

- resource_root is absolute and valid no matter the process CWD. The
  benchmark mp4 flow os.chdir's into its output directory before the render
  client is created, so any CWD-relative resolution breaks every model load.
- Every model filename render.h loads exists under resource_root, so an
  asset deleted from the repo (or a new model added without shipping its
  file) fails here instead of at render time.
"""

import os
import re

from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")
RENDER_HEADER = os.path.join(REPO_ROOT, "pufferlib", "ocean", "drive", "render.h")


def _make_replay_env():
    return Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=400,
        resample_frequency=1_000_000,
        termination_mode=0,
        report_interval=1,
    )


def _resource_root(env):
    return env._env_init_kwargs(env.map_files[0], 1)["resource_root"]


def test_resource_root_is_absolute_and_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = _make_replay_env()
    resource_root = _resource_root(env)
    assert os.path.isabs(resource_root)
    assert os.path.isdir(resource_root)
    env.reset()
    env.close()


def test_every_render_model_ships_with_the_package():
    with open(RENDER_HEADER) as f:
        render_source = f.read()
    model_filenames = set(re.findall(r'load_drive_model\(env, "([^"]+)"\)', render_source))
    assert len(model_filenames) >= 8, "expected the car/cyclist/pedestrian model loads in render.h"

    env = _make_replay_env()
    resource_root = _resource_root(env)
    missing = sorted(f for f in model_filenames if not os.path.isfile(os.path.join(resource_root, f)))
    assert not missing, f"render.h loads models not shipped under {resource_root}: {missing}"
    env.close()
