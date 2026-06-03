"""Map-source smoke + sanity tests for Drive.

For each (map_source, simulation_mode) pair we confirm that:
  - Construction, reset, and stepping a full episode complete inside a
    watchdog budget (no hangs).
  - Observations and rewards stay finite throughout.
  - The engine emits an episode log by `scenario_length`.

Carla bins ship only a road graph (no logged trajectories), so they're
exercised in gigaflow mode only. nuPlan and WOMD bins ship logged
trajectories and run in both gigaflow and replay so we cover the
full ingestion path on each format."""

import os
import signal
from contextlib import contextmanager

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN_ROOT = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries")

CASES = [
    pytest.param("carla", "gigaflow", id="carla-gigaflow"),
    pytest.param("nuplan", "gigaflow", id="nuplan-gigaflow"),
    pytest.param("nuplan", "replay", id="nuplan-replay"),
    # TODO: Rebuild obstacles bins with v0.3 123Drive
    # pytest.param("obstacles", "gigaflow", id="womd-gigaflow"),
    # pytest.param("obstacles", "replay", id="womd-replay"),
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


@pytest.mark.parametrize("map_subdir, sim_mode", CASES)
def test_load_step_log(map_subdir, sim_mode):
    map_dir = os.path.join(BIN_ROOT, map_subdir)
    assert os.path.isdir(map_dir), f"Test fixture missing: {map_dir}"

    # control_sdc_only is the natural pairing for replay (one policy-controlled
    # agent + log-replayed others). gigaflow tests the random-spawn path.
    if sim_mode == "replay":
        mode_kwargs = dict(
            num_agents=1,
            min_agents_per_env=1,
            max_agents_per_env=1,
            control_mode="control_sdc_only",
        )
    else:
        mode_kwargs = dict(num_agents=32)

    scenario_length = 91
    tag = f"{map_subdir}/{sim_mode}"

    with _watchdog(30, f"Drive() init [{tag}]"):
        env = Drive(
            num_maps=1,
            scenario_length=scenario_length,
            resample_frequency=0,
            report_interval=1,
            simulation_mode=sim_mode,
            map_dir=map_dir,
            **mode_kwargs,
        )

    with _watchdog(30, f"env.reset() [{tag}]"):
        obs, _ = env.reset(seed=0)
    assert np.isfinite(obs).all(), f"Non-finite obs after reset [{tag}]"

    logs = []
    with _watchdog(60, f"step loop [{tag}]"):
        for _ in range(scenario_length + 5):
            actions = np.zeros_like(env.actions)
            obs, reward, _, _, info = env.step(actions)
            assert np.isfinite(obs).all(), f"Non-finite obs during step [{tag}]"
            assert np.isfinite(reward).all(), f"Non-finite reward [{tag}]"
            if info:
                logs.extend(info)

    assert logs, f"No episode log emitted within {scenario_length + 5} steps [{tag}]"

    env.close()
