import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive


def test_drive_emits_log_after_custom_episode_length():
    """Ensure the engine produces an episode log once the custom horizon elapses."""

    try:
        env = Drive(
            num_agents=32,
            num_maps=1,
            episode_length=5,
            resample_frequency=0,
            report_interval=1,
        )
    except FileNotFoundError:
        pytest.skip("Drive map binaries are not available in this checkout")

    env.reset(seed=0)
    logs = []
    for _ in range(6):
        actions = np.zeros_like(env.actions)
        _, _, _, _, info = env.step(actions)
        if info:
            logs.extend(info)

    env.close()

    assert logs, "Drive never flushed its log; episode_length may be ignored"
    assert pytest.approx(logs[-1]["episode_length"], rel=0.0, abs=1e-6) == 5
