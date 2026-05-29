from pathlib import Path

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive

MAP_DIR = Path(__file__).resolve().parents[1] / "pufferlib/resources/drive/binaries/carla"


def test_drive_emits_log_after_custom_scenario_length():
    """Ensure the engine produces an episode log once the custom horizon elapses."""

    if not MAP_DIR.is_dir() or not any(MAP_DIR.glob("*.bin")):
        pytest.skip(f"Drive map binaries not available at {MAP_DIR}")

    env = Drive(
        num_agents=32,
        num_maps=1,
        scenario_length=5,
        resample_frequency=0,
        report_interval=1,
        map_dir=str(MAP_DIR),
    )

    env.reset(seed=0)
    logs = []
    for _ in range(6):
        actions = np.zeros_like(env.actions)
        _, _, _, _, info = env.step(actions)
        if info:
            logs.extend(info)

    env.close()

    assert logs, "Drive never flushed its log; scenario_length may be ignored"
    assert pytest.approx(logs[-1]["episode_length"], rel=0.0, abs=1e-6) == 5
