"""Behavioral tests for the per-agent EMA logging path in Drive's vec_log.

Covered:
- T1 (gate): vec_log emits nothing while no agent has completed an episode;
  the first emission appears exactly when the scenario truncates.
- T2 (steady-state n): after enough steps for every agent to complete at
  least one episode, dict["n"] equals num_agents.
- T3 (state persistence): two consecutive emissions with no new completions
  between them produce identical metric values, proving prepare_log no
  longer resets the per-agent EMAs.
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
