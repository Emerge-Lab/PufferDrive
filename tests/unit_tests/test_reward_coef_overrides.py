"""Per-agent reward-coef overrides and per-row learner masks on the Drive env."""

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import (
    REWARD_COEF_NAMES,
    Drive,
    resolve_reward_coef_overrides,
    reward_coef_from_normalized,
    reward_coef_to_normalized,
)

NUPLAN_MAP = "pufferlib/resources/drive/binaries/nuplan/nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin"

REPLAY_KWARGS = dict(
    map_dir=NUPLAN_MAP,
    num_maps=1,
    simulation_mode="replay",
    control_mode="control_vehicles",
    goal_source="gt",
    scenario_length=40,
    resample_frequency=0,
    num_agents=40,
    max_agents_per_env=64,
    reward_conditioning=True,
    reward_randomization=True,
)


def coef_column(env, name):
    return env.ego_features + REWARD_COEF_NAMES.index(name)


def rows_by_agent_idx(env):
    """(row, scenario agent index) for every active row."""
    rows = []
    for offset, state in zip(env.agent_offsets, env.get_state()):
        rows.extend((offset + row, agent_idx) for row, agent_idx in enumerate(state["active_agent_indices"]))
    return rows


def test_coef_names_follow_c_order():
    assert len(REWARD_COEF_NAMES) == binding.NUM_REWARD_COEFS
    assert REWARD_COEF_NAMES[:4] == ("goal_radius", "goal_speed", "collision", "offroad")
    assert REWARD_COEF_NAMES[-3:] == ("throttle", "steer", "acc")


@pytest.mark.parametrize("name", REWARD_COEF_NAMES)
def test_normalized_round_trip(name):
    for normalized in (-1.0, -0.4, 0.0, 0.6, 1.0):
        assert reward_coef_to_normalized(name, reward_coef_from_normalized(name, normalized)) == pytest.approx(
            normalized, abs=1e-5
        )
    with pytest.raises(ValueError, match="outside"):
        reward_coef_from_normalized(name, 1.5)


def test_raw_round_trip_inside_bounds():
    for raw in (-3.0, 0.0, 1.5, 6.0):
        assert reward_coef_from_normalized("collision", reward_coef_to_normalized("collision", raw)) == pytest.approx(
            raw, abs=1e-5
        )


def test_resolve_rejects_bad_specs():
    with pytest.raises(ValueError, match="Unknown reward coef"):
        resolve_reward_coef_overrides({"others": {"colision": 1.0}})
    with pytest.raises(ValueError, match="same agent"):
        resolve_reward_coef_overrides({"sdc": {"collision": 1.0}, 0: {"offroad": 1.0}})
    with pytest.raises(ValueError, match="must be 'sdc', 'others'"):
        resolve_reward_coef_overrides({-3: {"collision": 1.0}})
    with pytest.raises(ValueError, match="not finite"):
        resolve_reward_coef_overrides({"others": {"collision": float("nan")}})
    with pytest.raises(TypeError):
        resolve_reward_coef_overrides([("others", "collision", 1.0)])
    assert resolve_reward_coef_overrides({"7": {"offroad": 0.5}}) == [(7, REWARD_COEF_NAMES.index("offroad"), 0.5)]


def test_overrides_require_reward_conditioning():
    with pytest.raises(ValueError, match="require reward_conditioning"):
        Drive(
            **{**REPLAY_KWARGS, "reward_conditioning": False, "reward_randomization": False},
            reward_coef_overrides={"others": {"collision": 1.0}},
        )


def test_overrides_target_rows_and_stick_across_resets():
    env = Drive(
        **REPLAY_KWARGS,
        reward_coef_overrides={"sdc": {"collision": 5.0}, "others": {"collision": -2.0}, 6: {"offroad": 2.5}},
    )
    collision, offroad = coef_column(env, "collision"), coef_column(env, "offroad")
    lane_align = coef_column(env, "lane_align")
    first_lane_align = None
    for episode_seed in (1, 2):
        env.reset(seed=episode_seed)
        rows = rows_by_agent_idx(env)
        assert any(agent_idx == 6 for _, agent_idx in rows)
        for row, agent_idx in rows:
            expected = 5.0 if agent_idx == 0 else -2.0
            assert env.observations[row, collision] == pytest.approx(reward_coef_to_normalized("collision", expected))
            if agent_idx == 6:
                assert env.observations[row, offroad] == pytest.approx(reward_coef_to_normalized("offroad", 2.5))
        lane_align_now = env.observations[[row for row, _ in rows], lane_align].copy()
        if first_lane_align is not None:
            assert not np.array_equal(first_lane_align, lane_align_now), "non-overridden coefs stay randomized"
        first_lane_align = lane_align_now
    env.close()


def test_normalized_overrides_reach_obs_exactly():
    env = Drive(
        **REPLAY_KWARGS, reward_coef_overrides={"others": {"goal_speed": 0.25}}, reward_coef_overrides_normalized=True
    )
    env.reset(seed=3)
    column = coef_column(env, "goal_speed")
    for row, agent_idx in rows_by_agent_idx(env):
        if agent_idx != 0:
            assert env.observations[row, column] == pytest.approx(0.25, abs=1e-6)
    env.close()


def test_runtime_set_applies_at_next_reset_and_survives_resample():
    env = Drive(**{**REPLAY_KWARGS, "resample_frequency": 15})
    env.reset(seed=1)
    env.set_reward_coef_overrides({"others": {"collision": 0.0}}, normalized=True)
    column = coef_column(env, "collision")
    env.reset(seed=2)
    others = [row for row, agent_idx in rows_by_agent_idx(env) if agent_idx != 0]
    assert np.allclose(env.observations[others, column], 0.0, atol=1e-6)
    actions = np.zeros(env.actions.shape, dtype=env.actions.dtype)
    for _ in range(15):
        env.step(actions)
    others = [row for row, agent_idx in rows_by_agent_idx(env) if agent_idx != 0]
    assert np.allclose(env.observations[others, column], 0.0, atol=1e-6), "overrides survive a map resample"
    env.set_reward_coef_overrides(None)
    env.reset(seed=4)
    assert not np.allclose(env.observations[others, column], 0.0, atol=1e-6)
    env.close()


def test_learner_masks_mark_sdc_rows_only():
    with pytest.raises(ValueError, match="log_sdc_only"):
        Drive(**REPLAY_KWARGS, learner_agents="sdc")
    env = Drive(**{**REPLAY_KWARGS, "num_agents": 60}, learner_agents="sdc", log_sdc_only=True)
    expected = np.zeros(env.num_agents, dtype=bool)
    for offset, state in zip(env.agent_offsets, env.get_state()):
        expected[offset] = state["active_agent_indices"][0] == 0
    assert expected.sum() >= 2
    np.testing.assert_array_equal(env.learner_masks, expected)
    assert env.log_agent_count == expected.sum()
    env.close()

    default_env = Drive(**REPLAY_KWARGS)
    assert default_env.learner_masks[: default_env.agent_offsets[-1]].all()
    default_env.close()
