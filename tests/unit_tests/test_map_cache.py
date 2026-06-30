"""Tests for the per-process map cache (use_map_cache env knob).

Covers: cache-on vs cache-off observation parity, refcount discipline across
close orderings, stride-specific cache keys, slot-reuse keeping cache size
bounded, and per-entry owner_pid correctness in a forked child.
"""

import os

import numpy as np
import pytest

from pufferlib.ocean.drive import binding as drive_binding
from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARLA_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "carla")


def _make_drive(use_map_cache: int, num_agents: int = 32, num_maps: int = 1, **kwargs):
    """Construct a Drive env on the bundled carla map dir.

    num_agents defaults to 32 to satisfy the default min_agents_per_env=32; the
    cache lifecycle is the same regardless of agent count.
    """
    return Drive(
        num_agents=num_agents,
        num_maps=num_maps,
        use_map_cache=use_map_cache,
        map_dir=CARLA_MAP_DIR,
        scenario_length=64,
        resample_frequency=0,
        report_interval=1,
        **kwargs,
    )


def _step_and_record(env, action_stream):
    """Step `env` through `action_stream` and return (obs, rewards, terminals, truncations) per step."""
    obs_log, rew_log, term_log, trunc_log = [], [], [], []
    for action in action_stream:
        np.copyto(env.actions, action)
        o, r, term, trunc, _ = env.step(env.actions)
        obs_log.append(o.copy())
        rew_log.append(r.copy())
        term_log.append(term.copy())
        trunc_log.append(trunc.copy())
    return obs_log, rew_log, term_log, trunc_log


def test_obs_reward_done_parity_cache_on_vs_off():
    """Cache-on must produce byte-identical obs/rewards/terminals/truncations to
    cache-off given the same map and seed. Any silent geometry mismatch in the
    hit path (wrong vision_range, swapped grid_map pointers, stale neighbor
    offsets) surfaces here on the first step. Zero actions are sufficient — the
    cache affects only which static geometry is read, so every trajectory
    exercises it.
    """
    seed = 12345
    num_steps = 50

    env_off = _make_drive(use_map_cache=0)
    env_off.reset(seed=seed)
    actions = [np.zeros_like(env_off.actions) for _ in range(num_steps)]
    off_obs, off_rew, off_term, off_trunc = _step_and_record(env_off, actions)
    env_off.close()

    env_on = _make_drive(use_map_cache=1)
    env_on.reset(seed=seed)
    on_obs, on_rew, on_term, on_trunc = _step_and_record(env_on, actions)
    env_on.close()

    for t in range(num_steps):
        np.testing.assert_array_equal(off_obs[t], on_obs[t], err_msg=f"observations diverged at step {t}")
        np.testing.assert_array_equal(off_rew[t], on_rew[t], err_msg=f"rewards diverged at step {t}")
        np.testing.assert_array_equal(off_term[t], on_term[t], err_msg=f"terminals diverged at step {t}")
        np.testing.assert_array_equal(off_trunc[t], on_trunc[t], err_msg=f"truncations diverged at step {t}")


def test_map_cache_key_includes_road_obs_stride():
    env_a = None
    env_b = None
    try:
        env_a = _make_drive(use_map_cache=1, obs_lane_stride=1, obs_boundary_stride=1)
        env_a.reset(seed=0)
        live_after_a = drive_binding.map_cache_live_count()

        env_b = _make_drive(use_map_cache=1, obs_lane_stride=2, obs_boundary_stride=3)
        env_b.reset(seed=0)
        live_after_b = drive_binding.map_cache_live_count()

        assert live_after_b == live_after_a + 1, (
            "Map cache reused a grid built with different obs_lane_stride/obs_boundary_stride."
        )
    finally:
        if env_b is not None:
            env_b.close()
        if env_a is not None:
            env_a.close()


@pytest.mark.parametrize(
    "close_order",
    [
        (0, 1, 2),  # forward
        (2, 1, 0),  # reverse
        (1, 2, 0),  # rotation
        (1, 0, 2),  # adjacent swap
    ],
)
def test_multi_env_close_orderings_do_not_crash(close_order):
    """Three Drive instances on the same map share one cached SharedMapData
    entry (refcount=3 once all are alive). Closing them in arbitrary order
    must succeed: every close decrements the refcount, only the last one frees
    the entry. A double-free or use-after-free surfaces as SIGSEGV / abort
    on the final close in that ordering."""
    envs = [_make_drive(use_map_cache=1) for _ in range(3)]
    for e in envs:
        e.reset(seed=0)
    # One step exercises the shared grid_map / neighbor_offsets reads.
    for e in envs:
        e.step(np.zeros_like(e.actions))
    for idx in close_order:
        envs[idx].close()


def test_forked_child_can_build_and_free_its_own_entry():
    """In a forked child with the parent's cache pre-populated: building an env
    with use_map_cache=1 raises the child's live count by 1, and closing it drops
    the live count back to 0 (i.e., the child frees its own entry). The parent's
    cache size is unchanged by the child."""
    if os.name != "posix":
        pytest.skip("fork start method is POSIX-only")
    import multiprocessing as mp

    # Pre-populate the parent's cache so the child inherits a non-empty cache.
    warm = _make_drive(use_map_cache=1)
    warm.reset(seed=0)
    warm.close()
    parent_size_before_fork = drive_binding.map_cache_size()

    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def child_target(out_q):
        try:
            env = _make_drive(use_map_cache=1)
            env.reset(seed=0)
            env.step(np.zeros_like(env.actions))
            live_after_build = drive_binding.map_cache_live_count()
            env.close()
            live_after_close = drive_binding.map_cache_live_count()
            out_q.put(("ok", live_after_build, live_after_close))
        except BaseException as e:
            out_q.put(("err", repr(e), None))

    p = ctx.Process(target=child_target, args=(q,))
    p.start()
    p.join(timeout=60)
    assert p.exitcode == 0, f"Child process exited with {p.exitcode}"
    payload = q.get(timeout=5)
    assert payload[0] == "ok", f"Child raised: {payload[1]}"
    _, live_after_build, live_after_close = payload
    assert live_after_build == 1, f"Expected 1 live cache entry after build in child; got {live_after_build}."
    assert live_after_close == 0, f"Expected 0 live cache entries after close in child; got {live_after_close}."

    assert drive_binding.map_cache_size() == parent_size_before_fork, (
        f"Parent's cache size changed across fork ({parent_size_before_fork} -> {drive_binding.map_cache_size()})."
    )


def test_cache_size_bounded_by_unique_maps():
    """map_cache_insert reuses NULL slots freed by free_shared_map_data, so the
    slot count stays bounded across alloc/free/alloc cycles. Without slot reuse,
    every realloc appends past the NULL holes and the count drifts upward.

    Doesn't assert exact slot count (other tests in the same process may have
    populated the cache already); asserts the invariant that the count never
    grows under repeated cycles on the same set of maps.
    """

    # Allocate envs, then drop and re-allocate twice. Each cycle exercises a
    # free followed by an insert. The slot count after each cycle must equal
    # the count after the first cycle.
    def alloc_three_then_close():
        envs = [_make_drive(use_map_cache=1) for _ in range(3)]
        for e in envs:
            e.reset(seed=0)
        for e in envs:
            e.close()

    alloc_three_then_close()
    size_after_cycle_1 = drive_binding.map_cache_size()

    alloc_three_then_close()
    size_after_cycle_2 = drive_binding.map_cache_size()

    alloc_three_then_close()
    size_after_cycle_3 = drive_binding.map_cache_size()

    assert size_after_cycle_2 == size_after_cycle_1, (
        f"Cache grew on the second alloc cycle ({size_after_cycle_1} -> {size_after_cycle_2}); "
        f"map_cache_insert is appending past NULL holes instead of reusing them."
    )
    assert size_after_cycle_3 == size_after_cycle_1, (
        f"Cache grew on the third alloc cycle ({size_after_cycle_1} -> {size_after_cycle_3}); "
        f"slot-reuse invariant should keep the count flat across repeated cycles."
    )
