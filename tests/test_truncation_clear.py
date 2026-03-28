"""Test that truncations are properly cleared between steps.

Regression test for bug where resample_maps() set truncations[:] = 1
but step() never cleared them, causing bootstrap heuristic to fire
on every step after the first resample.
"""
import numpy as np
import pytest
from pufferlib.ocean.drive.drive import Drive


@pytest.fixture
def env():
    e = Drive(
        num_agents=2,
        num_maps=1,
        map_dir="pufferlib/resources/drive/binaries/carla_2D",
        dynamics_model="classic",
        min_agents_per_env=1,
        max_agents_per_env=2,
        resample_frequency=0,
        render_mode=None,
    )
    e.reset()
    yield e
    e.close()


def random_actions(env):
    return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)


def test_truncations_zero_during_normal_step(env):
    """Truncations should be 0 after a normal step (no resample)."""
    env.step(random_actions(env))
    assert (env.truncations == 0).all(), f"Expected all truncations=0, got {env.truncations}"


def test_truncations_one_on_resample(env):
    """Truncations should be 1 immediately after resample_maps."""
    env.resample_maps()
    assert (env.truncations == 1).all(), f"Expected all truncations=1, got {env.truncations}"


def test_truncations_cleared_after_resample(env):
    """Truncations should be cleared back to 0 on the step after resample."""
    env.resample_maps()
    assert (env.truncations == 1).all(), "Precondition: truncations should be 1 after resample"

    env.step(random_actions(env))
    assert (env.truncations == 0).all(), (
        f"Truncations should be 0 after step following resample, got {env.truncations}"
    )


def test_truncations_not_sticky_across_multiple_steps(env):
    """Truncations should only be 1 for one recv cycle, not persist."""
    env.resample_maps()
    env.step(random_actions(env))
    assert (env.truncations == 0).all()

    env.step(random_actions(env))
    assert (env.truncations == 0).all()

    env.step(random_actions(env))
    assert (env.truncations == 0).all()


def test_episode_end_sets_terminals():
    """When episodes end (e.g., episode_length reached), terminals should be set."""
    e = Drive(
        num_agents=2,
        num_maps=1,
        map_dir="pufferlib/resources/drive/binaries/carla_2D",
        dynamics_model="classic",
        min_agents_per_env=1,
        max_agents_per_env=2,
        episode_length=5,
        resample_frequency=0,
        render_mode=None,
    )
    e.reset()
    actions = np.zeros(e.action_space.shape, dtype=e.action_space.dtype)

    any_terminal = False
    for _ in range(10):
        e.step(actions)
        if e.terminals.any():
            any_terminal = True
            break

    assert any_terminal, "Expected at least one terminal=1 after episode_length steps"
    e.close()
