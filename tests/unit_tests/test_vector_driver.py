"""Driver-side behaviour of the Multiprocessing vecenv.

The driver builds one env of its own purely to read spaces off, which must not
stay resident for the run.
"""

import multiprocessing
import time

import gymnasium
import numpy as np
import pytest

import pufferlib.vector
from pufferlib import PufferEnv

OBSERVATION_SIZE = 4
ACTION_COUNT = 2
PRELOADED = False


class MinimalEnv(PufferEnv):
    def __init__(self, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(ACTION_COUNT)
        self.num_agents = 1
        self.close_count = 0
        super().__init__(buf=buf)

    def reset(self, seed=None):
        self.observations[:] = 0
        return self.observations, []

    def step(self, actions):
        self.observations[:] = 0
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def close(self):
        self.close_count += 1


class PreloadEnv(MinimalEnv):
    def __init__(self, buf=None, seed=0, config_only=False):
        self.preload_count = 0
        super().__init__(buf=buf, seed=seed)

    def preload_shared_resources(self):
        global PRELOADED
        PRELOADED = True
        self.preload_count += 1
        return 1

    def reset(self, seed=None):
        self.observations[:] = PRELOADED
        return self.observations, []


def _make_minimal_env(**kwargs):
    return MinimalEnv(**kwargs)


def _run_preload_from_forkserver(queue):
    vecenv = pufferlib.vector.make(
        PreloadEnv,
        backend="Multiprocessing",
        num_envs=1,
        num_workers=1,
        batch_size=1,
    )
    try:
        obs, _ = vecenv.reset()
        queue.put((multiprocessing.get_start_method(), vecenv.driver_env.preload_count, bool(obs[0, 0])))
    finally:
        vecenv.close()


def test_driver_env_is_released_but_stays_readable():
    """The driver env exists only to report spaces and agent counts -- workers
    build their own -- so keeping it would cost a whole extra worker's
    environments for the run. It is closed once during construction, and stays
    bound because callers read config attributes off vecenv.driver_env."""
    worker_count = 2
    vecenv = pufferlib.vector.make(
        [_make_minimal_env] * worker_count,
        env_args=[[]] * worker_count,
        env_kwargs=[{}] * worker_count,
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


def test_preloaded_resources_are_inherited_when_default_is_forkserver():
    context = multiprocessing.get_context("forkserver")
    queue = context.Queue()
    process = context.Process(target=_run_preload_from_forkserver, args=(queue,))
    process.start()
    process.join()

    assert process.exitcode == 0
    assert queue.get() == ("forkserver", 1, True)


def _make_multiprocessing(worker_count, batch_size):
    return pufferlib.vector.make(
        [_make_minimal_env] * worker_count,
        env_args=[[]] * worker_count,
        env_kwargs=[{}] * worker_count,
        backend="Multiprocessing",
        num_envs=worker_count,
        num_workers=worker_count,
        batch_size=batch_size,
    )


def test_reset_drains_pending_workers_and_steps_again():
    """A reset issued right after recv() finds half the workers idle and half
    mid-step; both halves must be reset without hanging and step afterwards."""
    worker_count = 4
    vecenv = _make_multiprocessing(worker_count, batch_size=2)
    try:
        obs, _ = vecenv.reset(seed=1)
        assert obs.shape == (2, OBSERVATION_SIZE)
        actions = np.zeros(vecenv.action_space.shape, dtype=np.int32)
        vecenv.send(actions)
        vecenv.recv()
        obs, _ = vecenv.reset(seed=2)
        assert obs.shape == (2, OBSERVATION_SIZE)
        assert vecenv.pending_result.sum() == 2
        for _ in range(worker_count):
            obs, *_ = vecenv.step(actions)
            assert obs.shape == (2, OBSERVATION_SIZE)
    finally:
        vecenv.close()


class SlowStepEnv(MinimalEnv):
    def step(self, actions):
        time.sleep(30)
        return super().step(actions)


def _make_slow_step_env(**kwargs):
    return SlowStepEnv(**kwargs)


def test_dead_worker_raises_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(pufferlib.vector, "WORKER_POLL_TIMEOUT_SECONDS", 0.2)
    vecenv = pufferlib.vector.make(
        [_make_minimal_env, _make_slow_step_env],
        env_args=[[]] * 2,
        env_kwargs=[{}] * 2,
        backend="Multiprocessing",
        num_envs=2,
        num_workers=2,
        batch_size=2,
    )
    try:
        vecenv.reset(seed=1)
        vecenv.send(np.zeros(vecenv.action_space.shape, dtype=np.int32))
        vecenv.processes[1].kill()
        vecenv.processes[1].join()
        with pytest.raises((RuntimeError, EOFError)):
            vecenv.recv()
    finally:
        vecenv.close()
