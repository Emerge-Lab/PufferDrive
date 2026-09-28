"""Tests for the human-imitation fine-tuning (HIFT-PPO) additions.

Covered here:
  - `pufferlib.pytorch.kl_divergence_to_reference`: exact forward KL for the
    categorical, multi-head categorical and Gaussian policy heads.
  - `env.reward_expert_similarity`: quadratic penalty on the ego-to-logged-expert
    position error in replay mode (zero when the ego replays the log).
  - `env.episode_max_steps`: fixed-length sub-episodes inside a long replay log.
  - `train.kl_ref_coef`: the frozen-reference KL penalty runs end to end on CPU
    and starts at zero when the reference is the untrained copy of the policy.
"""

import copy
import os
import sys

import numpy as np
import pytest
import torch

import pufferlib.pytorch as P
from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import PuffeRL, load_config, load_env, load_policy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPLAY_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "sdc_replay_test")
SCENARIO_LENGTH = 400
ZERO_REWARDS = dict(
    reward_goal=0.0,
    reward_collision=0.0,
    reward_offroad=0.0,
    reward_comfort=0.0,
    reward_lane_align=0.0,
    reward_vel_align=0.0,
    reward_lane_center=0.0,
    reward_center_bias=0.0,
    reward_velocity=0.0,
    reward_reverse=0.0,
    reward_stop_line=0.0,
    reward_timestep=0.0,
    reward_overspeed=0.0,
    reward_ade=0.0,
)


def test_categorical_kl_matches_torch_distributions():
    torch.manual_seed(0)
    logits = torch.randn(8, 12)
    reference_logits = torch.randn(8, 12)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.Categorical(logits=reference_logits),
    )
    torch.testing.assert_close(P.kl_divergence_to_reference(logits, reference_logits), expected)
    torch.testing.assert_close(P.kl_divergence_to_reference(logits, logits), torch.zeros(8))


def test_multi_head_kl_sums_over_heads():
    torch.manual_seed(0)
    heads = (torch.randn(5, 4), torch.randn(5, 3))
    reference_heads = (torch.randn(5, 4), torch.randn(5, 3))
    expected = sum(
        torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=head),
            torch.distributions.Categorical(logits=reference_head),
        )
        for head, reference_head in zip(heads, reference_heads)
    )
    torch.testing.assert_close(P.kl_divergence_to_reference(heads, reference_heads), expected)


def test_gaussian_kl_matches_torch_distributions():
    torch.manual_seed(0)
    policy = torch.distributions.Normal(torch.randn(6, 2), torch.rand(6, 2) + 0.1)
    reference = torch.distributions.Normal(torch.randn(6, 2), torch.rand(6, 2) + 0.1)
    expected = torch.distributions.kl_divergence(policy, reference).sum(1)
    torch.testing.assert_close(P.kl_divergence_to_reference(policy, reference), expected)


def _make_replay_env(sdc_controller, **overrides):
    kwargs = dict(
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=REPLAY_MAP_DIR,
        simulation_mode="replay",
        control_mode="control_sdc_only",
        sdc_controller=sdc_controller,
        non_sdc_controller="replay",
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=1_000_000,
        termination_mode=False,
        goal_source="gt",
        report_interval=1,
        num_goals=3,
        **ZERO_REWARDS,
    )
    kwargs.update(overrides)
    return Drive(**kwargs)


def _rollout_rewards(env, num_steps):
    env.reset(seed=0)
    zero_action = np.zeros_like(env.actions)
    rewards = []
    truncation_steps = []
    for step in range(1, num_steps + 1):
        _, reward, _, truncations, _ = env.step(zero_action)
        rewards.append(float(reward[0]))
        if truncations[0]:
            truncation_steps.append(step)
    return rewards, truncation_steps


def test_expert_similarity_reward_is_zero_on_log_and_negative_off_log():
    env = _make_replay_env("replay", reward_expert_similarity=1.0)
    try:
        on_log_rewards, _ = _rollout_rewards(env, 20)
    finally:
        env.close()
    assert np.allclose(on_log_rewards, 0.0, atol=1e-5)

    env = _make_replay_env("policy", reward_expert_similarity=1.0, dynamics_model="jerk")
    try:
        off_log_rewards, _ = _rollout_rewards(env, 20)
    finally:
        env.close()
    assert all(reward <= 0.0 for reward in off_log_rewards)
    assert sum(off_log_rewards) < 0.0


def test_episode_max_steps_truncates_every_sub_episode():
    episode_max_steps = 22
    env = _make_replay_env("replay", episode_max_steps=episode_max_steps)
    try:
        _, truncation_steps = _rollout_rewards(env, 3 * episode_max_steps)
    finally:
        env.close()
    assert truncation_steps == [episode_max_steps, 2 * episode_max_steps, 3 * episode_max_steps]


def _tiny_train_args():
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    args["vec"].update(backend="Serial", num_envs=2, seed=0)
    args["env"].update(
        num_agents=8,
        min_agents_per_env=8,
        max_agents_per_env=8,
        action_type="discrete",
        num_maps=1,
        use_map_cache=True,
        map_dir="pufferlib/resources/drive/binaries/carla",
        scenario_length=32,
    )
    args["policy"].update(
        ego_input_size=16,
        partner_input_size=16,
        lane_input_size=16,
        boundary_input_size=16,
        traffic_control_input_size=16,
        context_input_size=8,
        backbone_hidden_size=32,
        actor_hidden_size=32,
        critic_hidden_size=32,
    )
    args["train"].update(
        device="cpu",
        compile=False,
        seed=0,
        anneal_lr=False,
        update_epochs=1,
        bptt_horizon=32,
        minibatch_size=256,
        max_minibatch_size=256,
        total_timesteps=10_000_000,
        checkpoint_interval=10_000_000,
        kl_ref_coef=0.02,
    )
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None
    return args


class _NoLogger:
    run_id = "test"

    def log(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: None


def test_reference_kl_penalty_starts_at_zero_and_is_logged():
    args = _tiny_train_args()
    vecenv = load_env("puffer_drive", args, seed=0)
    policy = load_policy(args, vecenv, "puffer_drive")
    reference_policy = copy.deepcopy(policy)
    train_config = dict(**args["train"], env="puffer_drive", eval={}, run_name="test")
    pufferl = PuffeRL(train_config, vecenv, policy, logger=_NoLogger(), reference_policy=reference_policy)
    try:
        assert all(not param.requires_grad for param in pufferl.reference_policy.parameters())

        pufferl.evaluate()
        pufferl.last_log_time = 0.0
        pufferl.train()
        first_epoch_kl = pufferl.losses["reference_kl"]
        assert np.isfinite(first_epoch_kl)
        assert first_epoch_kl > -1e-6  # exact KL of identical policies is 0 up to float32 rounding
        assert first_epoch_kl < 1e-2

        pufferl.evaluate()
        pufferl.last_log_time = 0.0
        pufferl.train()
        assert np.isfinite(pufferl.losses["reference_kl"])
        assert pufferl.losses["reference_kl"] > -1e-6
    finally:
        pufferl.utilization.stop()
        vecenv.close()


def test_missing_reference_policy_is_rejected():
    args = _tiny_train_args()
    vecenv = load_env("puffer_drive", args, seed=0)
    try:
        policy = load_policy(args, vecenv, "puffer_drive")
        train_config = dict(**args["train"], env="puffer_drive", eval={}, run_name="test")
        with pytest.raises(Exception, match="reference policy"):
            PuffeRL(train_config, vecenv, policy, logger=_NoLogger())
    finally:
        vecenv.close()
