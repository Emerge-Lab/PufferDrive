"""Driver-side behaviour of the Multiprocessing vecenv.

Workers answer through shared-memory semaphores, so one that dies -- OOM-killed,
segfaulted, or crashed building its envs -- leaves the driver waiting on a
semaphore nobody will set. That used to surface as an eval that printed its
banner and then produced no output for hours. The driver also builds one env of
its own purely to read spaces off, which must not stay resident for the run.
"""

import os
import threading

import gymnasium
import numpy as np
import pytest

import pufferlib
import pufferlib.vector
from pufferlib import PufferEnv


# A regression hangs rather than fails, so bound the wait and assert it finished.
JOIN_TIMEOUT_SECONDS = 60
OBSERVATION_SIZE = 4
ACTION_COUNT = 2


class SuicidalEnv(PufferEnv):
    """Steps normally until `die_on_step`, then vanishes. os._exit skips
    interpreter cleanup, so the driver sees what an OOM kill looks like: no
    exception, no pipe traffic, just a process that stops answering."""

    def __init__(self, buf=None, seed=0, die_on_step=None):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(ACTION_COUNT)
        self.num_agents = 1
        self.die_on_step = die_on_step
        self.step_count = 0
        self.close_count = 0
        super().__init__(buf=buf)

    def reset(self, seed=None):
        self.observations[:] = 0
        return self.observations, []

    def step(self, actions):
        self.step_count += 1
        if self.die_on_step is not None and self.step_count >= self.die_on_step:
            os._exit(1)
        self.observations[:] = 0
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def close(self):
        self.close_count += 1


def _make_suicidal_env(die_on_step=None, **kwargs):
    return SuicidalEnv(die_on_step=die_on_step, **kwargs)


def _drive_until_raise(vecenv, result):
    try:
        vecenv.reset()
        for _ in range(1000):
            actions = np.zeros(vecenv.action_space.shape, dtype=np.int32)
            vecenv.step(actions)
    except BaseException as error:  # noqa: BLE001 - recorded and re-asserted below
        result.append(error)


def test_driver_raises_when_a_worker_dies():
    worker_count = 2
    # Worker 0 survives, so the driver is left waiting on worker 1 alone --
    # exactly the shape of the hang, rather than every worker vanishing at once.
    vecenv = pufferlib.vector.make(
        [_make_suicidal_env] * worker_count,
        env_args=[[]] * worker_count,
        env_kwargs=[{"die_on_step": None}, {"die_on_step": 3}],
        backend="Multiprocessing",
        num_envs=worker_count,
        num_workers=worker_count,
        batch_size=worker_count,
    )
    result = []
    try:
        driver = threading.Thread(target=_drive_until_raise, args=(vecenv, result), daemon=True)
        driver.start()
        driver.join(timeout=JOIN_TIMEOUT_SECONDS)
        assert not driver.is_alive(), "driver did not notice the dead worker and is still spinning"
    finally:
        vecenv.close()

    assert result, "driver returned without raising for a dead worker"
    error = result[0]
    assert isinstance(error, pufferlib.APIUsageError)
    assert "worker 1 died" in str(error)
    assert "exit code 1" in str(error)


def test_healthy_workers_are_not_reported_as_dead():
    worker_count = 2
    vecenv = pufferlib.vector.make(
        [_make_suicidal_env] * worker_count,
        env_args=[[]] * worker_count,
        env_kwargs=[{"die_on_step": None}] * worker_count,
        backend="Multiprocessing",
        num_envs=worker_count,
        num_workers=worker_count,
        batch_size=worker_count,
    )
    try:
        vecenv.reset()
        for _ in range(20):
            vecenv.step(np.zeros(vecenv.action_space.shape, dtype=np.int32))
        vecenv._check_workers_alive()
    finally:
        vecenv.close()


def test_liveness_poll_reports_the_first_dead_worker():
    """_check_workers_alive names the worker and the signal that killed it."""

    class DeadProcess:
        exitcode = -9

    class LiveProcess:
        exitcode = None

    vecenv = pufferlib.vector.Multiprocessing.__new__(pufferlib.vector.Multiprocessing)
    vecenv.processes = [LiveProcess(), DeadProcess()]
    with pytest.raises(pufferlib.APIUsageError, match=r"worker 1 died \(signal 9, .*OOM killer"):
        vecenv._check_workers_alive()


def test_driver_env_is_released_but_stays_readable():
    """The driver env exists only to report spaces and agent counts -- workers
    build their own -- so keeping it would cost a whole extra worker's
    environments for the run. It is closed once during construction, and stays
    bound because callers read config attributes off vecenv.driver_env."""
    worker_count = 2
    vecenv = pufferlib.vector.make(
        [_make_suicidal_env] * worker_count,
        env_args=[[]] * worker_count,
        env_kwargs=[{"die_on_step": None}] * worker_count,
        backend="Multiprocessing",
        num_envs=worker_count,
        num_workers=worker_count,
        batch_size=worker_count,
    )
    try:
        assert vecenv.driver_env.close_count == 1
        # Spaces and counts were harvested before the release.
        assert vecenv.single_observation_space.shape == (OBSERVATION_SIZE,)
        assert vecenv.agents_per_batch == worker_count
        obs, _ = vecenv.reset()
        assert obs.shape == (worker_count, OBSERVATION_SIZE)
        vecenv.step(np.zeros(vecenv.action_space.shape, dtype=np.int32))
    finally:
        vecenv.close()
    # close() must not close it a second time; on a native env that double frees.
    assert vecenv.driver_env.close_count == 1
