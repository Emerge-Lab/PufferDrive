"""Driver-side behaviour of the Multiprocessing vecenv.

The driver builds one env of its own purely to read spaces off, which must not
stay resident for the run.
"""

import multiprocessing
import os
import time

import numpy as np
import pytest

import opengigaflow.vector
from opengigaflow import VecEnv

OBSERVATION_SIZE = 4
ACTION_SIZE = 2
FRACTIONAL_OBSERVATION = np.float32(1.0 / 3.0)
STEP_OBSERVATION_SCALE = np.float32(1.7)
SIBLING_CPU_OFFSET = 1000
CACHE_DOMAIN_CORE_COUNT = 8
PRELOADED = False


class MinimalEnv(VecEnv):
    def __init__(self, buf=None, seed=0):
        self.single_observation_space = opengigaflow.spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.single_action_space = opengigaflow.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
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
    vecenv = opengigaflow.vector.make(
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
    vecenv = opengigaflow.vector.make(
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
        vecenv.step(np.zeros(vecenv.action_space.shape, dtype=np.float32))
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
    return opengigaflow.vector.make(
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
        actions = np.zeros(vecenv.action_space.shape, dtype=np.float32)
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
    monkeypatch.setattr(opengigaflow.vector, "WORKER_POLL_TIMEOUT_SECONDS", 0.2)
    vecenv = opengigaflow.vector.make(
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
        vecenv.send(np.zeros(vecenv.action_space.shape, dtype=np.float32))
        vecenv.processes[1].kill()
        vecenv.processes[1].join()
        with pytest.raises((RuntimeError, EOFError)):
            vecenv.recv()
    finally:
        vecenv.close()


class FractionalObservationEnv(MinimalEnv):
    def reset(self, seed=None):
        self.observations[:] = FRACTIONAL_OBSERVATION
        return self.observations, []

    def step(self, actions):
        super().step(actions)
        self.observations[:] = FRACTIONAL_OBSERVATION * STEP_OBSERVATION_SCALE
        return self.observations, self.rewards, self.terminals, self.truncations, []


def test_float16_transport_delivers_the_rounded_env_observations():
    delivered = {}
    for transport_dtype in (None, "float16"):
        vecenv = opengigaflow.vector.make(
            FractionalObservationEnv,
            backend="Multiprocessing",
            num_envs=2,
            num_workers=2,
            batch_size=2,
            observation_transport_dtype=transport_dtype,
        )
        try:
            reset_observations, _ = vecenv.reset(seed=1)
            reset_observations = reset_observations.copy()
            step_observations, *_ = vecenv.step(np.zeros(vecenv.action_space.shape, dtype=np.float32))
            delivered[transport_dtype] = (reset_observations, step_observations.copy())
        finally:
            vecenv.close()
    for exact, rounded in zip(delivered[None], delivered["float16"]):
        assert exact.dtype == np.float32
        assert rounded.dtype == np.float16
        assert not np.array_equal(exact, exact.astype(np.float16).astype(np.float32))
        np.testing.assert_array_equal(rounded, exact.astype(np.float16))
    assert not np.array_equal(delivered["float16"][0], delivered["float16"][1])


def test_unsupported_transport_dtype_is_rejected():
    with pytest.raises(opengigaflow.APIUsageError, match="Cannot transport"):
        opengigaflow.vector.make(
            MinimalEnv,
            backend="Multiprocessing",
            num_envs=1,
            num_workers=1,
            batch_size=1,
            observation_transport_dtype="float64",
        )


def _synthetic_cores(core_count):
    """Cores as read_cpu_cores reports them: (cache domain, [first thread, sibling thread])."""
    return [
        (f"domain{core_idx // CACHE_DOMAIN_CORE_COUNT}", [core_idx, core_idx + SIBLING_CPU_OFFSET])
        for core_idx in range(core_count)
    ]


def _cache_domain_counts(worker_cpus):
    core_indices = [cpu % SIBLING_CPU_OFFSET for cpu in worker_cpus]
    return np.bincount([core_idx // CACHE_DOMAIN_CORE_COUNT for core_idx in core_indices]).tolist()


def test_pinning_gives_each_buffer_one_worker_per_core():
    worker_cpus = opengigaflow.vector.plan_worker_cpus(_synthetic_cores(32), 64)
    assert len(set(worker_cpus)) == 64
    first_buffer, second_buffer = worker_cpus[:32], worker_cpus[32:]
    assert sorted(first_buffer) == list(range(32))
    assert sorted(second_buffer) == [core_idx + SIBLING_CPU_OFFSET for core_idx in range(32)]


def test_pinning_spreads_small_buffers_evenly_over_cache_domains():
    worker_cpus = opengigaflow.vector.plan_worker_cpus(_synthetic_cores(32), 64)
    for buffer_start in range(0, 64, 16):
        assert _cache_domain_counts(worker_cpus[buffer_start : buffer_start + 16]) == [4, 4, 4, 4]
    assert not {cpu % SIBLING_CPU_OFFSET for cpu in worker_cpus[:16]} & {
        cpu % SIBLING_CPU_OFFSET for cpu in worker_cpus[16:32]
    }


def test_pinning_with_fewer_cores_than_workers_per_buffer_shares_threads_across_buffers():
    worker_cpus = opengigaflow.vector.plan_worker_cpus(_synthetic_cores(28), 64)
    assert len(set(worker_cpus)) == 56
    first_buffer, second_buffer = worker_cpus[:32], worker_cpus[32:]
    assert len(set(first_buffer)) == 32
    assert len(set(second_buffer)) == 32
    assert _cache_domain_counts(first_buffer[28:]) == [1, 1, 1, 1]
    assert len(set(first_buffer) & set(second_buffer)) == 8


def test_cache_domain_shares_partition_a_resource_in_proportion_to_workers():
    worker_cache_domains = ["a", "b", "a", "c", "b", "a", "a", "c"]
    shares = opengigaflow.vector.cache_domain_shares(worker_cache_domains)
    assert shares[0] == shares[2] == shares[5] == shares[6] == (0.0, 0.5)
    assert shares[1] == shares[4] == (0.5, 0.75)
    assert shares[3] == shares[7] == (0.75, 1.0)


class AffinityReportingEnv(MinimalEnv):
    def __init__(self, buf=None, seed=0, cache_domain_share=None):
        self.cache_domain_share = cache_domain_share
        super().__init__(buf=buf, seed=seed)

    def reset(self, seed=None):
        affinity = sorted(os.sched_getaffinity(0))
        self.observations[:, 0] = affinity[0]
        self.observations[:, 1] = len(affinity)
        self.observations[:, 2:4] = self.cache_domain_share
        return self.observations, []


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="CPU affinity is Linux-only")
def test_pinned_workers_run_on_their_planned_cpu():
    vecenv = opengigaflow.vector.make(
        AffinityReportingEnv, backend="Multiprocessing", num_envs=2, num_workers=2, batch_size=2, pin_workers=True
    )
    process_cpus = os.sched_getaffinity(0)
    try:
        observations, _ = vecenv.reset(seed=1)
        assert set(vecenv.worker_cpus) <= process_cpus
        assert observations[:, 0].tolist() == vecenv.worker_cpus
        assert observations[:, 1].tolist() == [1, 1]
        assert observations[:, 2:4].tolist() == [list(share) for share in vecenv.worker_cache_shares]
        assert min(share[0] for share in vecenv.worker_cache_shares) == 0.0
        assert max(share[1] for share in vecenv.worker_cache_shares) == 1.0
        vecenv.bind_driver_thread()
        assert os.sched_getaffinity(0) == (vecenv.driver_cpus or process_cpus)
        assert not set(vecenv.worker_cpus) & (vecenv.driver_cpus or set())
        vecenv.release_driver_thread()
        assert os.sched_getaffinity(0) == process_cpus
    finally:
        vecenv.close()


class SeedRecordingEnv(MinimalEnv):
    def __init__(self, buf=None, seed=0):
        self.constructor_seed = seed
        super().__init__(buf=buf, seed=seed)


def test_make_forwards_seed_to_backend():
    constructor_seeds = {}
    for vec_seed in (1, 2):
        vecenv = opengigaflow.vector.make(SeedRecordingEnv, backend="Serial", num_envs=2, seed=vec_seed)
        constructor_seeds[vec_seed] = [env.constructor_seed for env in vecenv.envs]
        vecenv.close()
    assert constructor_seeds[1] != constructor_seeds[2]
    assert len(set(constructor_seeds[1])) == 2
