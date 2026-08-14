"""Driver-side behaviour of the Multiprocessing vecenv.

The driver builds one env of its own purely to read spaces off, which must not
stay resident for the run.
"""

import gymnasium
import numpy as np

import pufferlib.vector
from pufferlib import PufferEnv

OBSERVATION_SIZE = 4
ACTION_COUNT = 2


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


def _make_minimal_env(**kwargs):
    return MinimalEnv(**kwargs)


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
