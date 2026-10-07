"""Failure branching: a main scene's first learner collision restarts its branch scenes shortly before the crash.

Scenes are packed in groups of 1 main + BRANCH_COUNT branches on one map. Branch k restarts at
k * OFFSET_STEPS before the main's crash, plays REPLAYS episodes from that state to scenario_length, then idles
masked until the main banks its next crash; crashes while branches still run are dropped.

The fixture is the checked-in nuPlan replay map with every vehicle policy-controlled and the SDC as the only
learner row, i.e. the frozen co-player layout. Branch rows are driven with the main's recorded actions, so an
exactly restored state must reproduce the main's observations, rewards and terminals bit for bit.
"""

import os

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")
SCENARIO_LENGTH = 80
BRANCH_COUNT = 3
OFFSET_STEPS = 4
REPLAYS = 2
GROUP_SIZE = BRANCH_COUNT + 1
MAX_STEPS = 3000


def _drive_kwargs(num_agents, **overrides):
    kwargs = dict(
        num_agents=num_agents,
        min_agents_per_env=1,
        max_agents_per_env=64,
        num_maps=1,
        map_dir=MAP_DIR,
        simulation_mode="replay",
        control_mode="control_vehicles",
        sdc_controller="policy",
        non_sdc_controller="policy",
        non_vehicle_controller="replay",
        learner_agents="sdc",
        log_sdc_only=True,
        goal_source="gt",
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=0,
        termination_mode=0,
        collision_behavior="stop",
        offroad_behavior="ignore",
        action_type="discrete",
        dynamics_model="jerk",
        failure_branching=True,
        failure_branch_count=BRANCH_COUNT,
        failure_branch_offset_steps=OFFSET_STEPS,
        failure_branch_replays=REPLAYS,
        seed=0,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(scope="module")
def rows_per_scene():
    probe = Drive(**_drive_kwargs(64, failure_branching=False))
    try:
        return int(np.diff(probe.agent_offsets)[0])
    finally:
        probe.close()


def _make_env(rows_per_scene, scene_count, **overrides):
    return Drive(**_drive_kwargs(rows_per_scene * scene_count, **overrides))


def _scene_rows(env):
    return [slice(env.agent_offsets[i], env.agent_offsets[i + 1]) for i in range(env.num_envs)]


def test_branches_replay_the_main_exactly_from_before_its_crash(rows_per_scene):
    env = _make_env(rows_per_scene, GROUP_SIZE)
    rows = _scene_rows(env)
    rng = np.random.default_rng(0)
    action_count = env.single_action_space.nvec[0]
    try:
        env.reset(seed=0)
        states = env.get_state()
        assert [state["failure_branch_role"] for state in states] == [binding.FAILURE_BRANCH_ROLE_MAIN] + [
            binding.FAILURE_BRANCH_ROLE_BRANCH
        ] * BRANCH_COUNT
        main_history = {}
        replayed_history = None
        replays_finished = [0] * GROUP_SIZE
        restore_steps = [None] * GROUP_SIZE
        compared_steps = 0
        for _ in range(MAX_STEPS):
            if all(count >= REPLAYS for count in replays_finished[1:]):
                break
            main_timestep = states[0]["timestep"]
            actions = rng.integers(0, action_count, size=(env.num_agents, 1))
            for branch_idx in range(1, GROUP_SIZE):
                if states[branch_idx]["failure_branch_running"]:
                    actions[rows[branch_idx]] = replayed_history[states[branch_idx]["timestep"]]["actions"]
            main_history[main_timestep] = dict(obs=env.observations[rows[0]].copy(), actions=actions[rows[0]].copy())
            previous_obs = env.observations.copy()

            _, rewards, terminals, truncations, _ = env.step(actions)
            new_states = env.get_state()
            main_history[main_timestep].update(rewards=rewards[rows[0]].copy(), terminals=terminals[rows[0]].copy())

            for branch_idx in range(1, GROUP_SIZE):
                before, after, branch_rows = states[branch_idx], new_states[branch_idx], rows[branch_idx]
                if not before["failure_branch_running"] and after["failure_branch_running"]:
                    crash_step = new_states[0]["timestep"]
                    replayed_history = main_history
                    restore_steps[branch_idx] = after["timestep"]
                    assert after["timestep"] == crash_step - branch_idx * OFFSET_STEPS
                    assert truncations[branch_rows].all(), "waking up must start a new episode on the branch rows"
                    np.testing.assert_array_equal(
                        env.observations[branch_rows], replayed_history[after["timestep"]]["obs"]
                    )
                    continue
                if not before["failure_branch_running"]:
                    assert not env.masks[branch_rows].any(), "dormant branch rows must stay masked"
                    assert not rewards[branch_rows].any() and not terminals[branch_rows].any()
                    np.testing.assert_array_equal(env.observations[branch_rows], previous_obs[branch_rows])
                    continue
                expected = replayed_history[before["timestep"]]
                np.testing.assert_array_equal(rewards[branch_rows], expected["rewards"])
                np.testing.assert_array_equal(terminals[branch_rows], expected["terminals"])
                compared_steps += 1
                if not truncations[branch_rows].any():
                    assert after["timestep"] == before["timestep"] + 1
                    np.testing.assert_array_equal(
                        env.observations[branch_rows], replayed_history[after["timestep"]]["obs"]
                    )
                    continue
                replays_finished[branch_idx] += 1
                if after["failure_branch_running"]:
                    assert after["timestep"] == restore_steps[branch_idx]
                    np.testing.assert_array_equal(
                        env.observations[branch_rows], replayed_history[after["timestep"]]["obs"]
                    )
                else:
                    assert replays_finished[branch_idx] % REPLAYS == 0
                    assert not env.masks[branch_rows].any()

            if new_states[0]["timestep"] < main_timestep:
                main_history = {}
            states = new_states
    finally:
        env.close()

    assert all(count >= REPLAYS for count in replays_finished[1:]), f"branches never finished: {replays_finished}"
    assert compared_steps > 0


def test_crashes_while_branches_run_are_dropped_and_branch_episodes_log_apart(rows_per_scene):
    env = _make_env(rows_per_scene, GROUP_SIZE)
    rows = _scene_rows(env)
    rng = np.random.default_rng(1)
    action_count = env.single_action_space.nvec[0]
    captured = discarded = 0
    branch_episode_count = 0.0
    try:
        assert env.log_agent_count == 1, "only the main scene's SDC feeds vec_log"
        env.reset(seed=1)
        for _ in range(MAX_STEPS):
            _, _, _, _, info = env.step(rng.integers(0, action_count, size=(env.num_agents, 1)))
            for log in info:
                captured += log["failure_branch"]["failures_captured"]
                discarded += log["failure_branch"]["failures_discarded"]
                branch_episode_count += log["failure_branch"].get("n", 0.0)
            if captured >= 2 and discarded >= 1 and branch_episode_count > 0:
                break
        assert captured >= 2 and discarded >= 1 and branch_episode_count > 0, (
            captured,
            discarded,
            branch_episode_count,
        )

        env.reset(seed=2)
        assert not any(state["failure_branch_running"] for state in env.get_state()), "reset must put branches to sleep"
        env.step(rng.integers(0, action_count, size=(env.num_agents, 1)))
        for branch_rows in rows[1:]:
            assert not env.masks[branch_rows].any()
    finally:
        env.close()


def test_scenes_that_do_not_fill_a_group_stay_plain(rows_per_scene):
    env = _make_env(rows_per_scene, GROUP_SIZE + 2)
    try:
        roles = [state["failure_branch_role"] for state in env.get_state()]
        assert env.failure_branch_group_count == 1
        assert roles[GROUP_SIZE:] == [binding.FAILURE_BRANCH_ROLE_NONE] * 2
        assert env.log_agent_count == 3, "the main and both plain scenes feed vec_log"
    finally:
        env.close()


@pytest.mark.parametrize(
    "scene_count, overrides, message",
    [
        (GROUP_SIZE - 1, {}, "needs 4 identical scenes"),
        (GROUP_SIZE, dict(failure_branch_offset_steps=SCENARIO_LENGTH), "must be < scenario_length"),
        (GROUP_SIZE, dict(failure_branch_count=0), "must be >= 1"),
        (GROUP_SIZE, dict(init_step_spread=True), "disable init_step_spread"),
    ],
)
def test_invalid_setups_are_rejected(rows_per_scene, scene_count, overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_env(rows_per_scene, scene_count, **overrides)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
