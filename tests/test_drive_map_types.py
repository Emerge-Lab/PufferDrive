"""Smoke test for Drive map ingestion.

For each supported map source (Carla / nuPlan / WOMD), confirm that
constructing the env, resetting, and stepping 100 times all complete
within a watchdog budget. Catches regressions where a new map format
parses but then hangs or crashes the engine."""

import os
import signal
from contextlib import contextmanager

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_ROOT = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries")

MAP_DIRS = [
    pytest.param(os.path.join(BIN_ROOT, "carla_py123d"), id="carla"),
    pytest.param(os.path.join(BIN_ROOT, "nuplan"), id="nuplan"),
    pytest.param(os.path.join(BIN_ROOT, "obstacles"), id="womd"),
]


@contextmanager
def _watchdog(seconds, what):
    # SIGALRM is POSIX-only — fine for the Linux/macOS CI matrix.
    def _handler(signum, frame):
        raise TimeoutError(f"{what} hung for >{seconds}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


@pytest.mark.parametrize("map_dir", MAP_DIRS)
def test_load_and_step_100(map_dir):
    assert os.path.isdir(map_dir), f"Test fixture missing: {map_dir}"

    with _watchdog(30, f"Drive() construction with map_dir={map_dir}"):
        env = Drive(
            num_agents=32,
            num_maps=1,
            scenario_length=200,
            resample_frequency=0,
            report_interval=1,
            map_dir=map_dir,
        )

    with _watchdog(30, f"env.reset() with map_dir={map_dir}"):
        env.reset(seed=0)

    with _watchdog(60, f"100 env.step() with map_dir={map_dir}"):
        for _ in range(100):
            actions = np.zeros_like(env.actions)
            env.step(actions)

    env.close()
