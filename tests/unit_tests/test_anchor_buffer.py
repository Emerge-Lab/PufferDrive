"""Anchor buffer for KL-to-base training (train.anchor_coef): the KL, collecting a buffer in the learner's layout,
and validating a buffer file before training starts."""

import os

import numpy as np
import pytest
import torch

import pufferlib
import pufferlib.ocean.torch as drive_torch
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive, RewardConditioningView
from pufferlib.pufferl import (
    COPLAYER_ENV_SETUP,
    LOGREPLAY_ENV_SETUP,
    action_kl,
    collect_anchor_obs,
    load_anchor_obs,
    policy_io_layout,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NUPLAN_MAP = os.path.join(
    REPO_ROOT, "pufferlib/resources/drive/binaries/nuplan/nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin"
)
# The co-player finetune env: reward-conditioned for the co-player, read without that slice by the learner.
ENV_KWARGS = dict(
    map_dir=NUPLAN_MAP,
    num_maps=1,
    simulation_mode="replay",
    goal_source="gt",
    scenario_length=40,
    resample_frequency=0,
    num_agents=8,
    max_agents_per_env=64,
    min_agents_per_env=1,
    obs_slots_lane_n=8,
    obs_slots_boundary_n=8,
    obs_slots_partners_n=4,
    dynamics_model="jerk",
    action_type="continuous",
    reward_conditioning=True,
    seed=0,
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
SAMPLE_COUNT = 100


@pytest.fixture(scope="module")
def learner():
    """An unconditioned discrete-head learner, as nuBenchmark's SDC: built without the conditioning slice."""
    env = Drive(**ENV_KWARGS)
    torch.manual_seed(0)
    policy = drive_torch.Drive(RewardConditioningView(env, False), **POLICY_KWARGS)
    env.close()
    return policy


@pytest.fixture(scope="module")
def anchor(learner):
    torch.manual_seed(0)
    return collect_anchor_obs(ENV_KWARGS, learner, SAMPLE_COUNT, "cpu")


def test_action_kl_matches_torch_and_is_zero_for_the_same_policy():
    generator = torch.Generator().manual_seed(0)
    base_heads = (torch.randn(5, 3, generator=generator), torch.randn(5, 4, generator=generator))
    heads = (torch.randn(5, 3, generator=generator), torch.randn(5, 4, generator=generator))
    expected = sum(
        torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=base_head), torch.distributions.Categorical(logits=head)
        )
        for base_head, head in zip(base_heads, heads)
    )
    torch.testing.assert_close(action_kl(base_heads, heads), expected)
    torch.testing.assert_close(action_kl(base_heads[0], heads[0]), expected - action_kl(base_heads[1:], heads[1:]))
    assert torch.equal(action_kl(base_heads, base_heads), torch.zeros(5))

    base_normal = torch.distributions.Normal(torch.randn(5, 2, generator=generator), torch.rand(5, 2) + 0.1)
    normal = torch.distributions.Normal(torch.randn(5, 2, generator=generator), torch.rand(5, 2) + 0.1)
    torch.testing.assert_close(
        action_kl(base_normal, normal), torch.distributions.kl_divergence(base_normal, normal).sum(1)
    )


def test_collected_obs_are_what_the_learner_reads(learner, anchor):
    env = Drive(**{**ENV_KWARGS, **LOGREPLAY_ENV_SETUP})
    observations, _ = env.reset(seed=0)
    sdc_obs = observations[env.learner_masks]
    env.close()
    conditioning_columns = np.arange(binding.EGO_FEATURES, binding.EGO_FEATURES + binding.NUM_REWARD_COEFS)
    learner_obs = np.delete(sdc_obs, conditioning_columns, axis=1)

    obs = anchor["obs"]
    assert obs.dtype == torch.float16 and obs.shape == (SAMPLE_COUNT, learner_obs.shape[1])
    np.testing.assert_array_equal(obs[: len(learner_obs)].numpy(), learner_obs.astype(np.float16))
    assert anchor["meta"]["io_layout"] == policy_io_layout(learner, "learner")


def test_load_anchor_obs_round_trip(tmp_path, learner, anchor):
    path = tmp_path / "anchor.pt"
    torch.save(anchor, path)
    obs = load_anchor_obs(str(path), tuple(anchor["obs"].shape[1:]), learner, "cpu")
    assert torch.equal(obs, anchor["obs"])


def corrupt(anchor, field, value):
    return {**anchor, field: value}


@pytest.mark.parametrize(
    "make_buffer, message",
    [
        (lambda anchor: anchor["obs"], "dict with an 'obs' tensor"),
        (lambda anchor: corrupt(anchor, "obs", anchor["obs"][:, :-1]), "learner's observation layout"),
        (lambda anchor: corrupt(anchor, "obs", anchor["obs"][:0]), "non-empty"),
        (lambda anchor: corrupt(anchor, "obs", anchor["obs"].int()), "float"),
        (
            lambda anchor: corrupt(anchor, "obs", anchor["obs"].index_fill(0, torch.tensor([3]), float("nan"))),
            "non-finite",
        ),
        (lambda anchor: corrupt(anchor, "meta", ["not", "a", "dict"]), "'meta' must be a dict"),
        (
            lambda anchor: corrupt(
                anchor, "meta", {"io_layout": {**anchor["meta"]["io_layout"], "num_reward_coefs": 3}}
            ),
            "collected for observation layout",
        ),
    ],
    ids=["not-a-dict", "wrong-width", "empty", "int", "nan", "meta-list", "other-layout"],
)
def test_load_anchor_obs_rejects(tmp_path, learner, anchor, make_buffer, message):
    path = tmp_path / "anchor.pt"
    torch.save(make_buffer(anchor), path)
    with pytest.raises(pufferlib.APIUsageError, match=message):
        load_anchor_obs(str(path), tuple(anchor["obs"].shape[1:]), learner, "cpu")


def test_load_anchor_obs_needs_a_path(learner):
    with pytest.raises(pufferlib.APIUsageError, match="anchor_obs_path"):
        load_anchor_obs(None, (8,), learner, "cpu")


def test_collect_anchor_obs_rejects_eval_and_recurrent_policies(learner):
    with pytest.raises(pufferlib.APIUsageError, match="eval_mode=0"):
        collect_anchor_obs({**ENV_KWARGS, "eval_mode": 1}, learner, SAMPLE_COUNT, "cpu")
    env = Drive(**ENV_KWARGS)
    recurrent = drive_torch.Recurrent(RewardConditioningView(env, False), learner, input_size=16, hidden_size=16)
    env.close()
    with pytest.raises(pufferlib.APIUsageError, match="feed-forward"):
        collect_anchor_obs(ENV_KWARGS, recurrent, SAMPLE_COUNT, "cpu")
