"""Tests for the `terminate_on_goal` knob (early episode reset when the SDC
reaches its final goal).

Feature under test (drive.h, c_step): when terminate_on_goal=1 AND the env is in
replay + control_sdc_only mode, the episode truncates as soon as the single
active agent reaches its last goal (current_goal_idx ==
num_goals with REACHED_GOAL_IDX set), instead of running to
scenario_length.

Fixture map
-----------
Uses the single checked-in replay .bin under
pufferlib/resources/drive/binaries/sdc_replay_test/, whose logged SDC trajectory
reaches its final goal well before scenario_length ("all goals in position").

The env is built with termination_mode=0 so the ONLY source of early reset is
terminate_on_goal — that isolates the feature from the inactive-agent path.
"""

import os

import numpy as np

from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")

# Long enough that a goal-reaching replay truncates well before the length cap,
# so an early reset is unambiguously the goal firing and not the length boundary.
SCENARIO_LENGTH = 400


def _make_sdc_replay_env(terminate_on_goal: bool):
    """Single-agent replay/SDC env on the fixture map. sdc_controller='replay'
    makes the agent follow the logged trajectory deterministically, so it
    reaches its final goal every run regardless of policy/actions."""
    return Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=1_000_000,  # don't resample mid-episode
        termination_mode=0,  # isolate terminate_on_goal as the only early-reset source
        goal_source="gt",  # use the logged goal set from the replay
        terminate_on_goal=terminate_on_goal,
        report_interval=1,
        num_goals=3,
        goal_radius=2.0,
    )


def _first_truncation_step(env):
    """Step the (replay-driven) env with zero actions and return the 1-indexed
    step at which truncations[0] first fires, or None if it never does within
    scenario_length steps. On truncation c_step auto-resets, so we stop at the
    first fire to measure the initial episode length."""
    env.reset(seed=0)
    zero_action = np.zeros_like(env.actions)
    for step in range(1, SCENARIO_LENGTH + 1):
        _, _, _, truncations, _ = env.step(zero_action)
        if truncations[0]:
            return step
    return None


def test_terminate_on_goal_ends_episode_early():
    """With terminate_on_goal=1, the episode truncates strictly before the
    scenario_length cap (the SDC reaches its final goal mid-log)."""
    env = _make_sdc_replay_env(terminate_on_goal=True)
    try:
        step = _first_truncation_step(env)
    finally:
        env.close()

    assert step is not None, "Episode never truncated with terminate_on_goal=1."
    assert step < SCENARIO_LENGTH, (
        f"terminate_on_goal=1 should end the episode before scenario_length="
        f"{SCENARIO_LENGTH}, but first truncation was at step {step}."
    )


def test_no_terminate_on_goal_runs_to_scenario_length():
    """With terminate_on_goal=0, the same map runs the full scenario_length —
    reaching the goal must NOT end the episode early."""
    env = _make_sdc_replay_env(terminate_on_goal=False)
    try:
        step = _first_truncation_step(env)
    finally:
        env.close()

    assert step == SCENARIO_LENGTH, (
        f"terminate_on_goal=0 should only truncate at scenario_length="
        f"{SCENARIO_LENGTH}, but first truncation was at step {step}."
    )


def test_terminate_on_goal_is_earlier_than_length_cap():
    """Direct on-vs-off contrast on the same map/seed: the goal is reached at a
    fixed step, so enabling terminate_on_goal must truncate earlier than the
    length-capped baseline."""
    env_on = _make_sdc_replay_env(terminate_on_goal=True)
    try:
        step_on = _first_truncation_step(env_on)
    finally:
        env_on.close()

    env_off = _make_sdc_replay_env(terminate_on_goal=False)
    try:
        step_off = _first_truncation_step(env_off)
    finally:
        env_off.close()

    assert step_on is not None and step_off is not None
    assert step_on < step_off, (
        f"terminate_on_goal=1 (truncated at {step_on}) should end earlier than "
        f"terminate_on_goal=0 (truncated at {step_off})."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
