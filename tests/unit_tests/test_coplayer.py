"""Frozen co-player building blocks: obs-layout checks, checkpoint loading and recurrent state handling."""

import os

import numpy as np
import pytest
import torch
import yaml

import pufferlib
import pufferlib.ocean.torch as drive_torch
from pufferlib.ocean.drive.drive import Drive, RewardConditioningView
from pufferlib.pufferl import (
    COPLAYER_ENV_SETUP,
    FrozenCoplayer,
    load_coplayer_policy,
    reward_conditioning_obs_columns,
)

NUPLAN_MAP = "pufferlib/resources/drive/binaries/nuplan/nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin"
ENV_KWARGS = dict(
    map_dir=NUPLAN_MAP,
    num_maps=1,
    simulation_mode="replay",
    goal_source="gt",
    scenario_length=20,
    resample_frequency=0,
    num_agents=32,
    max_agents_per_env=64,
    obs_slots_lane_n=8,
    obs_slots_boundary_n=8,
    obs_slots_partners_n=4,
    dynamics_model="jerk",
    reward_conditioning=True,
    **COPLAYER_ENV_SETUP,
)
POLICY_KWARGS = dict(
    ego_input_size=8,
    partner_input_size=8,
    lane_input_size=8,
    boundary_input_size=8,
    traffic_control_input_size=8,
    context_input_size=8,
    backbone_hidden_size=16,
    backbone_num_layers=1,
    actor_hidden_size=8,
    actor_num_layers=1,
    critic_hidden_size=8,
    critic_num_layers=1,
    encoder_activation="relu",
    encoder_layer_norm=False,
    backbone_activation="relu",
    backbone_layer_norm=False,
    shared_network=True,
    mask_padded_features=False,
    action_type="discrete",
)


@pytest.fixture(scope="module")
def env():
    drive = Drive(**ENV_KWARGS)
    drive.reset(seed=0)
    yield drive
    drive.close()


def build_policy(layout_env, recurrent=False):
    policy = drive_torch.Drive(layout_env, **POLICY_KWARGS)
    if recurrent:
        policy = drive_torch.Recurrent(layout_env, policy, input_size=16, hidden_size=16)
    return policy


def save_run(run_dir, env_overrides, policy, policy_section=POLICY_KWARGS):
    os.makedirs(run_dir / "models")
    model_path = run_dir / "models" / "model.pt"
    torch.save(policy.state_dict(), model_path)
    config = {
        "policy_name": "Drive",
        "rnn_name": None,
        "env": {**ENV_KWARGS, "reward_conditioning": False, **env_overrides},
        "policy": dict(policy_section),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return str(model_path)


def test_obs_columns_strip_only_the_conditioning_slice(env):
    plain_policy = build_policy(RewardConditioningView(env, False))
    columns = reward_conditioning_obs_columns(env, plain_policy.io_layout, "test", "cpu")
    obs = torch.as_tensor(env.observations)
    stripped = obs.index_select(1, columns)
    ego = env.ego_features
    assert stripped.shape[1] == env.num_obs - env.num_reward_coefs
    torch.testing.assert_close(stripped[:, :ego], obs[:, :ego])
    torch.testing.assert_close(stripped[:, ego:], obs[:, ego + env.num_reward_coefs :])
    assert reward_conditioning_obs_columns(env, build_policy(env).io_layout, "test", "cpu") is None


def test_obs_layout_mismatch_lists_keys(env):
    other_env = Drive(**{**ENV_KWARGS, "obs_slots_partners_n": 6, "num_agents": 8})
    other_policy = build_policy(other_env)
    other_env.close()
    with pytest.raises(pufferlib.APIUsageError, match="obs_slots_partners_n"):
        reward_conditioning_obs_columns(env, other_policy.io_layout, "co-player", "cpu")

    plain_env = Drive(**{**ENV_KWARGS, "reward_conditioning": False, "num_agents": 8})
    with pytest.raises(pufferlib.APIUsageError, match="reward_conditioning=True"):
        reward_conditioning_obs_columns(plain_env, build_policy(env).io_layout, "co-player", "cpu")
    plain_env.close()


def test_load_coplayer_checks_config_and_falls_back_to_defaults(env, tmp_path):
    plain_policy = build_policy(RewardConditioningView(env, False))
    older_policy_section = {key: value for key, value in POLICY_KWARGS.items() if key != "action_type"}
    model_path = save_run(tmp_path / "ok", {}, plain_policy, policy_section=older_policy_section)
    loaded = load_coplayer_policy(model_path, env, "cpu")
    assert loaded.io_layout["num_reward_coefs"] == 0
    for key, value in plain_policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)

    mismatched_path = save_run(tmp_path / "mismatch", {"obs_slots_lane_n": 12, "num_goals": 2}, plain_policy)
    with pytest.raises(pufferlib.APIUsageError, match="num_goals.*obs_slots_lane_n"):
        load_coplayer_policy(mismatched_path, env, "cpu")

    with pytest.raises(pufferlib.APIUsageError, match="no config.yaml"):
        load_coplayer_policy(str(tmp_path / "missing" / "model.pt"), env, "cpu")


def test_frozen_coplayer_rejects_wrong_env_setup():
    vecenv = Drive(**{**ENV_KWARGS, "control_mode": "control_sdc_only", "num_agents": 4})
    try:
        with pytest.raises(pufferlib.APIUsageError, match="control_mode"):
            FrozenCoplayer(build_policy(vecenv), vecenv, "sample", "cpu")
        with pytest.raises(pufferlib.APIUsageError, match="coplayer_action_selection"):
            FrozenCoplayer(build_policy(vecenv), vecenv, "argmax", "cpu")
    finally:
        vecenv.close()


def test_recurrent_coplayer_resets_state_per_done_row(env):
    torch.manual_seed(0)
    policy = build_policy(env, recurrent=True)
    coplayer = FrozenCoplayer(policy, env, "mode", "cpu")
    assert not any(p.requires_grad for p in coplayer.policy.parameters())
    obs = torch.as_tensor(env.observations[:6])
    rows = torch.arange(10, 16)
    not_done = torch.zeros(6)
    coplayer.act(obs, not_done, rows)
    coplayer.act(obs, not_done, rows)
    carried_h = coplayer.lstm_h[rows].clone()
    assert carried_h.abs().sum() > 0
    assert coplayer.lstm_h[:10].abs().sum() == 0, "rows outside global_rows are untouched"

    done = torch.tensor([1.0, 0, 0, 1.0, 0, 0])
    coplayer.act(obs, done, rows)
    fresh_state = {"lstm_h": torch.zeros(6, 16), "lstm_c": torch.zeros(6, 16)}
    with torch.no_grad():
        policy.forward_eval(obs, fresh_state)
    done_rows = np.array([0, 3])
    torch.testing.assert_close(coplayer.lstm_h[rows][done_rows], fresh_state["lstm_h"][done_rows])
    assert not torch.allclose(coplayer.lstm_h[rows][1], fresh_state["lstm_h"][1])
