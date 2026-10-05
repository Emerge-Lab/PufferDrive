"""Tests for the `init_step_spread` knob (per-env randomized replay start step).

Feature under test (drive.py): when init_step_spread=1 AND the env is in replay
simulation_mode, each parallel env copy is seeded at a different, uniformly
sampled expert timestep in [0, scenario_length - init_step_min_horizon) instead
of all starting at the fixed `init_step`. This spreads a training batch over the
whole trajectory. The `init_step_min_horizon` guarantees every env keeps at
least that many steps of usable horizon.

Two things are checked here:
  1. Parallel envs are actually seeded at different steps (and within bounds).
  2. The setting is rejected outside replay mode (and when the horizon leaves no
     room to sample), since it only makes sense as a replay start-point spread.

Fixture map
-----------
Reuses the single checked-in nuPlan replay .bin under
pufferlib/resources/drive/binaries/sdc_replay_test/ (same fixture as
test_terminate_on_goal). Its logged trajectory is long enough that every sampled
start below SCENARIO_LENGTH stays in bounds.
"""

import os

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")

# Kept well within the fixture's logged trajectory so every sampled start is a
# valid, in-bounds expert timestep.
SCENARIO_LENGTH = 80
MIN_HORIZON = 10
# Enough parallel single-agent envs that a uniform draw over ~70 steps producing
# only one distinct value is astronomically unlikely (and deterministic per seed).
NUM_ENVS = 16


class _RecordingDrive(Drive):
    """Records every value `_sample_init_step` hands to a C env at construction,
    so a test can assert what start step each parallel env was actually seeded
    with. One entry is appended per env (see the env_init loop in Drive)."""

    def __init__(self, *args, **kwargs):
        self.recorded_init_steps = []
        super().__init__(*args, **kwargs)

    def _sample_init_step(self):
        step = super()._sample_init_step()
        self.recorded_init_steps.append(step)
        return step


def _make_spread_env(cls=Drive, *, init_step_spread=True, **overrides):
    """Multi-env single-agent replay env on the fixture map. One agent per env
    (max_agents_per_env=1) means NUM_ENVS parallel copies of the same scenario,
    each seeded independently by the spread sampler."""
    kwargs = dict(
        num_agents=NUM_ENVS,
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
        termination_mode=0,
        init_step_spread=init_step_spread,
        init_step_min_horizon=MIN_HORIZON,
        num_goals=3,
        goal_radius=2.0,
        seed=0,
    )
    kwargs.update(overrides)
    return cls(**kwargs)


def test_parallel_envs_seeded_at_different_steps():
    """With spread on, the NUM_ENVS parallel envs are seeded at more than one
    distinct expert timestep, and every sampled step is in
    [0, scenario_length - init_step_min_horizon)."""
    env = _make_spread_env(_RecordingDrive)
    try:
        steps = env.recorded_init_steps
    finally:
        env.close()

    upper = SCENARIO_LENGTH - MIN_HORIZON
    assert len(steps) == NUM_ENVS, f"Expected one sampled start per env, got {len(steps)}."
    assert all(0 <= s < upper for s in steps), f"Sampled starts out of [0,{upper}): {steps}"
    assert len(set(steps)) > 1, f"Spread produced identical starts for every env: {steps}"


def test_spread_off_starts_every_env_at_init_step():
    """Control: with spread off, every env is seeded at the fixed init_step, so
    the whole batch shares one start step (the pre-feature behavior)."""
    env = _make_spread_env(_RecordingDrive, init_step_spread=False, init_step=7)
    try:
        steps = env.recorded_init_steps
    finally:
        env.close()

    assert steps == [7] * NUM_ENVS, f"Spread off should seed every env at init_step=7, got {steps}."


def test_different_start_steps_yield_different_initial_observations():
    """End-to-end sanity that the sampled start actually reaches the C sim: two
    envs seeded at different steps produce different initial observations (the
    ego starts at a different point along the logged trajectory)."""
    env_early = _make_spread_env(init_step_spread=False, init_step=5)
    env_late = _make_spread_env(init_step_spread=False, init_step=60)
    try:
        env_early.reset(seed=0)
        env_late.reset(seed=0)
        obs_early = env_early.observations.copy()
        obs_late = env_late.observations.copy()
    finally:
        env_early.close()
        env_late.close()

    assert not np.allclose(obs_early, obs_late), (
        "Different init steps produced identical observations; start step may not be reaching the sim."
    )


def test_spread_rejected_in_non_replay_mode():
    """init_step_spread is replay-only: enabling it in gigaflow mode must raise,
    since there is no logged trajectory to spread over."""
    with pytest.raises(ValueError, match="replay"):
        _make_spread_env(simulation_mode="gigaflow")


@pytest.mark.parametrize(
    "min_horizon",
    [
        SCENARIO_LENGTH,  # exactly the episode length -> upper == 0
        SCENARIO_LENGTH + 1,  # one past the end
        SCENARIO_LENGTH + 100,  # well past the end
    ],
)
def test_spread_rejected_when_min_horizon_at_or_past_episode_length(min_horizon):
    """A min_horizon greater than (or equal to) the episode length leaves no
    valid start to sample (upper <= 0), so construction must fail fast instead of
    handing an empty/out-of-bounds range to the sampler."""
    with pytest.raises(ValueError, match="init_step_min_horizon"):
        _make_spread_env(init_step_min_horizon=min_horizon)


def test_spread_sizes_each_multi_agent_scene_at_its_own_start(capfd):
    """With control_vehicles the number of controlled vehicles depends on the start step, so shared() must size
    each scene's rows at the step that scene starts from; otherwise rows are left without an agent."""
    env = _make_spread_env(
        num_agents=128,
        max_agents_per_env=128,
        control_mode="control_vehicles",
        sdc_controller="policy",
        non_sdc_controller="policy",
        non_vehicle_controller="replay",
        goal_source="gt",
        learner_agents="sdc",
        log_sdc_only=True,
    )
    try:
        states = env.get_state()
        rows_per_scene = np.diff(env.agent_offsets)
        assert [state["active_agent_count"] for state in states] == rows_per_scene.tolist()
        expected_learner_rows = np.zeros(env.num_agents, dtype=bool)
        expected_learner_rows[env.agent_offsets[:-1]] = True
        np.testing.assert_array_equal(env.learner_masks, expected_learner_rows)
    finally:
        env.close()
    assert "Mismatch in active agent count" not in capfd.readouterr().out


def test_scene_given_more_rows_than_agents_is_rejected():
    """A replay scene whose buffer slice has more rows than controlled agents would leave rows no agent owns."""
    env = _make_spread_env(init_step_spread=False)
    rows = env.agent_offsets[1] - env.agent_offsets[0]
    try:
        with pytest.raises(ValueError, match="controls 1 agents but was given 3 rows"):
            binding.env_init(
                np.zeros((rows + 2, env.num_obs), dtype=np.float32),
                np.zeros((rows + 2, 1), dtype=np.float32),
                np.zeros(rows + 2, dtype=np.float32),
                np.zeros(rows + 2, dtype=bool),
                np.zeros(rows + 2, dtype=bool),
                np.ones(rows + 2, dtype=bool),
                0,
                learner_masks=np.ones(rows + 2, dtype=bool),
                **env._env_init_kwargs(env.map_files[0], rows + 2, env.init_step),
            )
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
