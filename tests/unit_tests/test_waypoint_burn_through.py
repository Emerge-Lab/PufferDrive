"""Tests that replay-mode waypoint progression burns exactly ONE waypoint per
goal-reach event, with the goal alias advancing contextually.

Feature under test (drive.h, compute_metrics): when the car spawns on top of a waypoint
the waypoint is consumed correctly and does not result in multiple ones being consumed
without updating the waypoint info.

Fixture map
-----------
The first 2 waypoints sit within goal_radius from step 0. This tests that we burn through
those 2 alone and we don't use the stale alias and burn through the third one as well.

Observability
-------------
get_state() exposes, per agent: the goal alias (current_goal_x/y), the
per-step reached-goal flag (metrics_array[REACHED_GOAL_METRIC_IDX], recomputed
every step), and the full logged trajectory. The expected waypoint list is
reconstructed with the same sampling formula c_reset uses in replay mode, so
each observed alias maps to a definite waypoint index k, and "one waypoint per
step" is asserted as: every fire moves k by exactly +1 (or saturates on the
final waypoint), and k never moves without a fire.

Note: env construction and env.reset() each run compute_metrics once at reset
time, and on this fixture each of those calls legitimately consumes one
waypoint. The tests account for that via the waypoint index of the post-reset
alias instead of assuming the episode starts at waypoint 0.
"""

import os

import numpy as np

from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")

SCENARIO_LENGTH = 400
NUM_GOALS = 3
GOAL_RADIUS = 2.0

# metrics_array layout (datatypes.h): REACHED_GOAL_IDX
REACHED_GOAL_METRIC_IDX = 4

# Waypoints on this fixture are distinct at the ~1e-5 m scale; the alias is a
# bit-exact copy of a trajectory sample, so a much tighter tolerance still
# identifies it unambiguously.
WAYPOINT_MATCH_TOLERANCE = 1e-9


def _make_sdc_replay_env():
    """Single-agent replay/SDC env on the fixture map (init_step=0)."""
    return Drive(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        goal_source="gt",
        control_mode="control_sdc_only",
        sdc_controller="replay",
        non_sdc_controller="replay",
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=1_000_000,
        termination_mode=False,
        report_interval=1,
        num_goals=NUM_GOALS,
        goal_radius=GOAL_RADIUS,
    )


def _sdc_state(env):
    """Per-agent state dict of the single active (SDC) agent."""
    env_state = env.get_state()[0]
    sdc_idx = env_state["active_agent_indices"][0]
    return env_state["agents"][sdc_idx]


def _expected_waypoints(agent_state):
    """Reconstruct the replay goal waypoints exactly as c_reset samples them
    from the logged trajectory (init_step=0): t = (g+1)*(size-1)/N."""
    trajectory_size = agent_state["trajectory_size"]
    remaining = max(trajectory_size - 1, 1)
    waypoints = []
    for g in range(NUM_GOALS):
        t = min((g + 1) * remaining // NUM_GOALS, trajectory_size - 1)
        waypoints.append((agent_state["log_trajectory_x"][t], agent_state["log_trajectory_y"][t]))
    return waypoints


def _alias_waypoint_index(agent_state, waypoints):
    """Map the current goal alias to its waypoint index; the alias must match
    exactly one expected waypoint."""
    alias = (agent_state["current_goal_x"], agent_state["current_goal_y"])
    matches = [
        k
        for k, (wp_x, wp_y) in enumerate(waypoints)
        if abs(alias[0] - wp_x) < WAYPOINT_MATCH_TOLERANCE and abs(alias[1] - wp_y) < WAYPOINT_MATCH_TOLERANCE
    ]
    assert len(matches) == 1, (
        f"Goal alias {alias} should match exactly one expected waypoint, matched {matches} of {waypoints}."
    )
    return matches[0]


def test_each_goal_fire_burns_exactly_one_waypoint():
    """Full episode: every reached-goal fire must
    advance the goal alias to the immediately-next waypoint (exactly one
    burned), the alias must never move without a fire, and once the final
    waypoint is consumed no further fires may occur (saturation)."""
    env = _make_sdc_replay_env()
    try:
        env.reset(seed=0)
        state = _sdc_state(env)
        waypoints = _expected_waypoints(state)
        current_wp_idx = _alias_waypoint_index(state, waypoints)
        # Waypoints consumed so far == index of the waypoint now aliased
        # (verified below: step 1 still fires, so the env is not saturated).
        consumed_count = current_wp_idx
        saturated = False

        zero_action = np.zeros_like(env.actions)
        truncated_at = None
        for step in range(1, SCENARIO_LENGTH + 1):
            _, _, _, truncations, _ = env.step(zero_action)
            if truncations[0]:
                truncated_at = step
                break  # env auto-reset: state now belongs to the next episode

            state = _sdc_state(env)
            fired = state["metrics_array"][REACHED_GOAL_METRIC_IDX] > 0.0
            new_wp_idx = _alias_waypoint_index(state, waypoints)

            if not fired:
                assert new_wp_idx == current_wp_idx, (
                    f"Step {step}: goal alias moved wp{current_wp_idx}->wp{new_wp_idx} without a reached-goal fire."
                )
                continue

            assert not saturated, (
                f"Step {step}: reached-goal fired again after the final waypoint "
                f"was already consumed (current_goal_idx should be saturated)."
            )
            consumed_count += 1
            if current_wp_idx < NUM_GOALS - 1:
                assert new_wp_idx == current_wp_idx + 1, (
                    f"Step {step}: fire on wp{current_wp_idx} must advance the alias "
                    f"by exactly one waypoint, but it moved to wp{new_wp_idx} "
                    f"(burn-through / stale alias)."
                )
                current_wp_idx = new_wp_idx
            else:
                # Final waypoint: alias stays put, index saturates.
                assert new_wp_idx == current_wp_idx, (
                    f"Step {step}: alias moved off the final waypoint wp{current_wp_idx} to wp{new_wp_idx}."
                )
                saturated = True
    finally:
        env.close()

    assert truncated_at == SCENARIO_LENGTH, (
        f"The episode must run to scenario_length={SCENARIO_LENGTH}, but truncated at {truncated_at}."
    )
    assert consumed_count == NUM_GOALS, (
        f"Episode should consume all {NUM_GOALS} waypoints exactly "
        f"once each, but accounting shows {consumed_count} consumptions."
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
