"""Behavioral tests for the per-agent EMA logging path in Drive's vec_log.

Covered:
- T1 (gate): vec_log emits nothing while no agent has completed an episode;
  the first emission appears exactly when the scenario truncates.
- T2 (steady-state n): after enough steps for every agent to complete at
  least one episode, dict["n"] equals num_agents.
- T3 (state persistence): two consecutive emissions with no new completions
  between them produce identical metric values, proving prepare_log no
  longer resets the per-agent EMAs.
- T4 (unequal completion rates): with sub-envs truncating at different
  timesteps, every emission still weights one term per agent. This is the
  case T1-T3 cannot distinguish, since they run all agents in lockstep.
"""

from pathlib import Path

import numpy as np
import pytest

from pufferlib.ocean.drive.drive import Drive

MAP_DIR = Path(__file__).resolve().parents[2] / "pufferlib/resources/drive/binaries/carla"

NUM_AGENTS = 32
SCENARIO_LENGTH = 5


def _log_dicts(info):
    """Filter info for the vec_log aggregate dict (carries both n and episode_return)."""
    return [d for d in info if isinstance(d, dict) and "n" in d and "episode_return" in d]


def _make_env():
    if not MAP_DIR.is_dir() or not any(MAP_DIR.glob("*.bin")):
        pytest.skip(f"Drive map binaries not available at {MAP_DIR}")
    return Drive(
        num_agents=NUM_AGENTS,
        num_maps=1,
        min_agents_per_env=1,
        max_agents_per_env=8,
        scenario_length=SCENARIO_LENGTH,
        report_interval=1,
        map_dir=str(MAP_DIR),
        log_ema_alpha=0.95,
    )


def _make_staggered_env():
    """Sub-envs truncate on their own inactive-agent ratio, so they complete at
    different timesteps instead of in lockstep."""
    if not MAP_DIR.is_dir() or not any(MAP_DIR.glob("*.bin")):
        pytest.skip(f"Drive map binaries not available at {MAP_DIR}")
    return Drive(
        num_agents=NUM_AGENTS,
        num_maps=4,
        min_agents_per_env=1,
        max_agents_per_env=4,
        scenario_length=40,
        report_interval=1,
        map_dir=str(MAP_DIR),
        log_ema_alpha=0.95,
        termination_mode=1,
        offroad_behavior="remove",
        collision_behavior="remove",
        # Resampling rebuilds the envs and so discards the slots; keep it out of
        # the measured window so only completion weighting is under test.
        resample_frequency=100_000,
    )


def test_gate_skips_emissions_until_first_completion():
    """T1: vec_log returns no log dict until at least one agent has completed an
    episode. With synced agents under null actions, completions first happen
    when timestep reaches scenario_length."""
    env = _make_env()
    env.reset(seed=0)

    # Steps 1..(L-1): no agent has truncated yet, gate sees aggregate.n == 0.
    for step_idx in range(SCENARIO_LENGTH - 1):
        _, _, _, _, info = env.step(np.zeros_like(env.actions))
        assert not _log_dicts(info), (
            f"Unexpected emission at step {step_idx + 1} (timestep={step_idx + 1}); "
            "gate should skip before any agent has completed."
        )

    # Step L: all agents truncate at timestep == scenario_length.
    _, _, _, _, info = env.step(np.zeros_like(env.actions))
    logs = _log_dicts(info)
    assert len(logs) == 1, f"Expected exactly one log dict at scenario end, got {len(logs)}"
    assert logs[0]["n"] >= 1, f"Expected n>=1 at first emission, got n={logs[0]['n']}"

    env.close()


def test_steady_state_n_equals_num_agents():
    """T2: once every agent has completed at least one episode, the emitted
    n is the full population. With synced agents this is true from the first
    emission onward."""
    env = _make_env()
    env.reset(seed=0)

    # Run multiple scenarios so every agent has contributed.
    last_log = None
    for _ in range(4 * SCENARIO_LENGTH):
        _, _, _, _, info = env.step(np.zeros_like(env.actions))
        logs = _log_dicts(info)
        if logs:
            last_log = logs[-1]

    assert last_log is not None, "Expected at least one emission across 4 scenarios"
    assert last_log["n"] == NUM_AGENTS, (
        f"Expected steady-state n={NUM_AGENTS}, got n={last_log['n']} (some agents missing from the population mean)"
    )

    env.close()


def test_emissions_identical_when_no_new_completions():
    """T3: prepare_log preserves per-agent EMA state across emissions. Two
    consecutive emissions with no intervening completions must produce
    bit-for-bit identical metric values."""
    env = _make_env()
    env.reset(seed=0)

    # Reach the first completion at timestep == scenario_length.
    for _ in range(SCENARIO_LENGTH):
        env.step(np.zeros_like(env.actions))

    # Next two emissions sit between completions (no agent truncates between
    # timestep=1 and timestep=scenario_length-1 of the next scenario).
    _, _, _, _, info_a = env.step(np.zeros_like(env.actions))
    _, _, _, _, info_b = env.step(np.zeros_like(env.actions))

    log_a = _log_dicts(info_a)
    log_b = _log_dicts(info_b)
    assert log_a and log_b, "Both consecutive steps should emit"
    log_a, log_b = log_a[-1], log_b[-1]

    assert log_a["n"] == log_b["n"], f"n changed without new completions: {log_a['n']} vs {log_b['n']}"
    for key in ("episode_return", "episode_length", "collision_rate", "offroad_rate"):
        assert log_a[key] == pytest.approx(log_b[key], rel=0, abs=1e-6), (
            f"{key} changed without new completions: {log_a[key]} vs {log_b[key]} "
            "(prepare_log may be resetting per-agent state)"
        )

    env.close()


def test_population_size_is_agent_count_under_unequal_completion_rates():
    """T4: the discriminating case. Sub-envs truncate independently here, so
    completions arrive staggered rather than all at once. Completion-weighted
    aggregation would make the emitted n track how many agents completed in
    that interval; agent-weighted aggregation pins it to the number of agents
    that have ever completed, which saturates at num_agents and stays there."""
    env = _make_staggered_env()
    env.reset(seed=0)

    population_sizes = []
    for _ in range(240):
        _, _, _, _, info = env.step(np.zeros_like(env.actions))
        logs = _log_dicts(info)
        if logs:
            population_sizes.append(logs[-1]["n"])

    assert population_sizes, "Expected emissions once sub-envs began completing"

    # Guard against the test silently degenerating into the lockstep regime T1-T3
    # already cover: staggered completions make n climb through intermediate values.
    warmup = [size for size in population_sizes if size < NUM_AGENTS]
    assert len(set(warmup)) > 1, (
        f"Expected staggered completions to grow the population gradually, saw {sorted(set(warmup))}; "
        "sub-envs are completing in lockstep so this test is not exercising unequal completion rates"
    )

    assert max(population_sizes) == NUM_AGENTS, (
        f"Population never reached the full agent count: max n={max(population_sizes)} of {NUM_AGENTS}"
    )

    # Monotonic and saturating: once an agent has completed it contributes to
    # every later emission, whether or not it completed again.
    saturation_idx = population_sizes.index(NUM_AGENTS)
    after_saturation = population_sizes[saturation_idx:]
    assert set(after_saturation) == {NUM_AGENTS}, (
        f"n fell back below {NUM_AGENTS} after saturating: {sorted(set(after_saturation))}; "
        "the emitted mean is weighted by completions in the interval, not by agent"
    )

    env.close()
