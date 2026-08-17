"""End-to-end tests for the distributional critic wired through the real
load_config -> load_env -> load_policy -> PuffeRL pipeline on CPU.

Covers the head geometry per mode, the state_dict compatibility claim, the
fail-fast config rejections, and one full evaluate()+train() epoch in each
binned mode.
"""

import contextlib
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pufferlib
from pufferlib.pufferl import PuffeRL, load_config, load_env, load_policy
from pufferlib.value_distribution import (
    VALUE_DISTRIBUTION_HL_GAUSS,
    VALUE_DISTRIBUTION_OFF,
    VALUE_DISTRIBUTION_TWO_HOT,
)

NUM_BINS = 101
SUPPORT_MIN = -10.0
SUPPORT_MAX = 10.0
BINNED_MODES = (VALUE_DISTRIBUTION_TWO_HOT, VALUE_DISTRIBUTION_HL_GAUSS)
SEED = 7
# Sized so one evaluate() fills the rollout buffer and completes episodes.
BPTT_HORIZON = 32
NUM_AGENTS = 16
MINIBATCH_SIZE = 64
FORWARD_BATCH = 5
# The loss keys PuffeRL logged before the distributional head existed.
BASELINE_LOSS_KEYS = frozenset(
    {
        "policy_loss",
        "value_loss",
        "entropy",
        "old_approx_kl",
        "approx_kl",
        "clipfrac",
        "explained_variance",
        "masked_fraction",
        "filter_threshold",
        "ema_max",
        "kept_fraction",
        "filtered_fraction",
    }
)


class _DummyLogger:
    run_id = "distributional-critic-test"

    def log(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: None


def build_args(mode=VALUE_DISTRIBUTION_OFF, **train_overrides):
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv

    args["vec"].update(backend="Serial", num_envs=1, seed=SEED)
    args["env"].update(
        num_agents=NUM_AGENTS,
        min_agents_per_env=NUM_AGENTS,
        max_agents_per_env=NUM_AGENTS,
        num_maps=1,
        use_map_cache=1,
        map_dir="pufferlib/resources/drive/binaries/carla",
        scenario_length=BPTT_HORIZON,
    )
    args["policy"].update(
        critic_distribution_mode=mode,
        critic_num_bins=NUM_BINS,
        critic_support_min=SUPPORT_MIN,
        critic_support_max=SUPPORT_MAX,
    )
    args["train"].update(
        device="cpu",
        precision="float32",
        compile=False,
        seed=SEED,
        torch_deterministic=True,
        anneal_lr=False,
        update_epochs=1,
        bptt_horizon=BPTT_HORIZON,
        minibatch_size=MINIBATCH_SIZE,
        max_minibatch_size=MINIBATCH_SIZE,
        total_timesteps=10_000_000,
        checkpoint_interval=10_000_000,
        render=False,
    )
    args["train"].update(train_overrides)
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = {}
    return args


def make_policy(args):
    vecenv = load_env("puffer_drive", args)
    return vecenv, load_policy(args, vecenv)


@contextlib.contextmanager
def make_pufferl(args, vecenv, policy):
    """PuffeRL starts a non-daemon Utilization thread that outlives the test and
    would hang the interpreter, so every construction is scoped."""
    train_config = dict(**args["train"], env="puffer_drive", eval=args["eval"])
    pufferl = PuffeRL(train_config, vecenv, policy, logger=_DummyLogger())
    try:
        yield pufferl
    finally:
        pufferl.utilization.stop()


def forward_value(policy, vecenv):
    observations = torch.zeros(FORWARD_BATCH, *vecenv.single_observation_space.shape)
    with torch.no_grad():
        _, value = policy(observations)
    return value


def test_off_mode_head_is_scalar():
    args = build_args(VALUE_DISTRIBUTION_OFF)
    vecenv, policy = make_policy(args)
    try:
        assert policy.value_distribution is None
        assert policy.critic_head[-1].out_features == 1
        assert forward_value(policy, vecenv).shape == (FORWARD_BATCH, 1)
    finally:
        vecenv.close()


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_binned_mode_head_is_binned(mode):
    args = build_args(mode)
    vecenv, policy = make_policy(args)
    try:
        assert policy.value_distribution.mode == mode
        assert policy.critic_head[-1].out_features == NUM_BINS
        value_logits = forward_value(policy, vecenv)
        assert value_logits.shape == (FORWARD_BATCH, NUM_BINS)
        expectation = policy.value_distribution.expectation(value_logits)
        assert expectation.shape == (FORWARD_BATCH,)
        assert expectation.ge(SUPPORT_MIN).all() and expectation.le(SUPPORT_MAX).all()
    finally:
        vecenv.close()


def test_state_dict_keys_identical_across_modes():
    state_dicts = {}
    for mode in (VALUE_DISTRIBUTION_OFF,) + BINNED_MODES:
        args = build_args(mode)
        vecenv, policy = make_policy(args)
        try:
            state_dicts[mode] = policy.state_dict()
        finally:
            vecenv.close()

    off_keys = sorted(state_dicts[VALUE_DISTRIBUTION_OFF].keys())
    for mode in BINNED_MODES:
        assert sorted(state_dicts[mode].keys()) == off_keys
        differing = [
            key for key in off_keys if state_dicts[mode][key].shape != state_dicts[VALUE_DISTRIBUTION_OFF][key].shape
        ]
        assert sorted(differing) == sorted(_final_critic_keys(off_keys))


def _final_critic_keys(keys):
    last = max(int(key.split(".")[1]) for key in keys if key.startswith("critic_head."))
    return [f"critic_head.{last}.weight", f"critic_head.{last}.bias"]


def test_invalid_mode_fails_at_policy_construction():
    args = build_args("twohot")
    vecenv = load_env("puffer_drive", args)
    try:
        with pytest.raises(ValueError, match="critic_distribution_mode"):
            load_policy(args, vecenv)
    finally:
        vecenv.close()


def test_pufferl_rejects_vf_clip_coef():
    args = build_args(VALUE_DISTRIBUTION_TWO_HOT, vf_clip_coef=0.2)
    vecenv, policy = make_policy(args)
    try:
        with pytest.raises(pufferlib.APIUsageError, match="vf_clip_coef"):
            with make_pufferl(args, vecenv, policy):
                pass
    finally:
        vecenv.close()


def test_pufferl_rejects_use_rnn():
    args = build_args(VALUE_DISTRIBUTION_TWO_HOT, use_rnn=True)
    vecenv, policy = make_policy(args)
    try:
        with pytest.raises(pufferlib.APIUsageError, match="rnn"):
            with make_pufferl(args, vecenv, policy):
                pass
    finally:
        vecenv.close()


def test_pufferl_accepts_vf_clip_coef_when_off():
    args = build_args(VALUE_DISTRIBUTION_OFF, vf_clip_coef=0.2)
    vecenv, policy = make_policy(args)
    try:
        with make_pufferl(args, vecenv, policy) as pufferl:
            assert pufferl.value_distribution is None
    finally:
        vecenv.close()


def run_one_epoch(args):
    torch.manual_seed(SEED)
    vecenv, policy = make_policy(args)
    try:
        with make_pufferl(args, vecenv, policy) as pufferl:
            pufferl.evaluate()
            pufferl.last_log_time = 0.0
            pufferl.train()
            return pufferl.losses, pufferl.values.detach().clone()
    finally:
        vecenv.close()


@pytest.mark.parametrize("mode", BINNED_MODES)
def test_one_epoch_train_binned(mode):
    losses, values = run_one_epoch(build_args(mode))

    assert losses["value_loss"] > 0.0
    assert math.isfinite(losses["value_loss"])
    assert "value_target_clip_fraction" in losses
    assert "value_expectation_mse" in losses
    assert 0.0 <= losses["value_target_clip_fraction"] <= 1.0
    assert math.isfinite(losses["value_expectation_mse"])
    assert math.isfinite(losses["explained_variance"])

    assert torch.isfinite(values).all()
    assert values.ge(SUPPORT_MIN).all() and values.le(SUPPORT_MAX).all()


def test_off_mode_loss_keys_unchanged():
    losses, _ = run_one_epoch(build_args(VALUE_DISTRIBUTION_OFF))
    assert "value_target_clip_fraction" not in losses
    assert "value_expectation_mse" not in losses
    assert set(losses) == BASELINE_LOSS_KEYS
