"""Filtered PPO batches keep epoch boundaries and sample weights when accumulated."""

import contextlib
import math
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from opengigaflow.rl import PPOTrainer


class ValuePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.value.weight, 0.25)

    def forward(self, observations, state=None):
        return observations.new_zeros((observations.shape[0], 2)), self.value(observations)


@pytest.mark.parametrize(
    "retained_count, microbatch_size, accumulation_count, discard_partial_minibatch, trained_count_per_epoch",
    [
        (6, 8, 4, False, 6),
        (6, 4, 2, False, 6),
        (10, 4, 2, False, 10),
        (8, 4, 2, False, 8),
        (10, 4, 1, False, 10),
        (0, 4, 2, False, 0),
        (10, 4, 1, True, 8),
        (10, 4, 2, True, 8),
        (20, 4, 2, True, 16),
        (8, 4, 2, True, 8),
        (6, 4, 2, True, 6),
        (3, 4, 1, True, 3),
        (0, 4, 2, True, 0),
    ],
)
def test_filtered_ppo_accumulation_matches_optimizer_batches(
    retained_count, microbatch_size, accumulation_count, discard_partial_minibatch, trained_count_per_epoch
):
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.policy = ValuePolicy()
    trainer.config = {
        "device": "cpu",
        "update_epochs": 3,
        "adv_filter_enabled": True,
        "adv_filter_ewma_beta": 0.25,
        "adv_filter_threshold_scale": 0.01,
        "adv_filter_leak_fraction": 0.0,
        "adv_filter_fixed_minibatch_count": None,
        "min_batch_size": None,
        "discard_partial_minibatch": discard_partial_minibatch,
        "clip_coef": 0.2,
        "vf_clip_coef": None,
        "vf_coef": 1.0,
        "ent_coef": 0.0,
        "ent_coef_anneal": False,
        "max_grad_norm": 1e6,
    }
    trainer.amp_context = contextlib.nullcontext()
    trainer.separate_grad_clip = False
    trainer.minibatch_size = microbatch_size
    trainer.accumulate_minibatches = accumulation_count
    trainer.ema_max = 1.0
    trainer.vecenv = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(1,)))

    # Four transitions fall below the filter and two are masked out of training.
    transition_count = retained_count + 6
    observations = torch.linspace(0.5, 1.5, transition_count).reshape(1, transition_count, 1)
    returns = 2.0 * observations.squeeze(-1)
    advantages = torch.zeros_like(returns)
    advantages[:, :retained_count] = 1.0
    advantages[:, -2:] = 1.0
    masks = torch.ones_like(returns, dtype=torch.bool)
    masks[:, -2:] = False
    trainer.observations = observations
    trainer.actions = torch.zeros_like(returns, dtype=torch.int32)
    trainer.logprobs = torch.full_like(returns, -math.log(2.0))
    trainer.values = torch.zeros_like(returns)
    trainer._compute_advantages = Mock(return_value=(advantages, returns, masks))
    trainer.optimizer = torch.optim.SGD(trainer.policy.parameters(), lr=0.1)
    gradients = []
    trainer.optimizer.register_step_pre_hook(
        lambda *_: gradients.append(trainer.policy.value.weight.grad.detach().clone())
    )

    # Compare against ordinary SGD on each optimizer batch, including its partial tail unless that is discarded.
    optimizer_batch_size = microbatch_size * accumulation_count
    reference_policy = ValuePolicy()
    reference_optimizer = torch.optim.SGD(reference_policy.parameters(), lr=0.1)
    reference_gradients = []
    random_seed = 7
    torch.manual_seed(random_seed)
    for _ in range(trainer.config["update_epochs"]):
        permutation = torch.randperm(retained_count)
        for start in range(0, trained_count_per_epoch, optimizer_batch_size):
            indices = permutation[start : start + optimizer_batch_size]
            predictions = reference_policy.value(observations[0, indices]).squeeze(-1)
            loss = 0.5 * (predictions - returns[0, indices]).square().mean()
            loss.backward()
            reference_gradients.append(reference_policy.value.weight.grad.detach().clone())
            reference_optimizer.step()
            reference_optimizer.zero_grad()

    torch.manual_seed(random_seed)
    trained_transition_count = trainer._train_ppo_transition(defaultdict(float), Mock(), epoch=1)

    assert trained_transition_count == trainer.config["update_epochs"] * trained_count_per_epoch
    expected_updates = trainer.config["update_epochs"] * math.ceil(trained_count_per_epoch / optimizer_batch_size)
    assert len(gradients) == expected_updates
    for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(gradient, reference_gradient)
    torch.testing.assert_close(trainer.policy.value.weight, reference_policy.value.weight)


@pytest.mark.parametrize(
    "valid_count, microbatch_size, accumulation_count, fixed_minibatch_count, discard_partial_minibatch, "
    "kept_count, trained_count_per_epoch",
    [
        (20, 4, 1, 3, False, 12, 12),
        (40, 4, 2, 3, False, 24, 24),
        (12, 4, 1, 3, False, 12, 12),
        (10, 4, 1, 3, False, 10, 10),
        (20, 4, 1, 3, True, 12, 12),
        (40, 4, 2, 3, True, 24, 24),
        (10, 4, 1, 3, True, 10, 8),
        (20, 4, 2, 3, True, 20, 16),
        (3, 4, 1, 3, True, 3, 3),
    ],
)
def test_fixed_minibatch_count_filter_keeps_top_advantages(
    valid_count,
    microbatch_size,
    accumulation_count,
    fixed_minibatch_count,
    discard_partial_minibatch,
    kept_count,
    trained_count_per_epoch,
):
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.policy = ValuePolicy()
    trainer.config = {
        "device": "cpu",
        "update_epochs": 3,
        "adv_filter_enabled": True,
        "adv_filter_ewma_beta": 0.25,
        "adv_filter_threshold_scale": 0.01,
        "adv_filter_leak_fraction": 0.0,
        "adv_filter_fixed_minibatch_count": fixed_minibatch_count,
        "min_batch_size": None,
        "discard_partial_minibatch": discard_partial_minibatch,
        "clip_coef": 0.2,
        "vf_clip_coef": None,
        "vf_coef": 1.0,
        "ent_coef": 0.0,
        "ent_coef_anneal": False,
        "max_grad_norm": 1e6,
    }
    trainer.amp_context = contextlib.nullcontext()
    trainer.separate_grad_clip = False
    trainer.minibatch_size = microbatch_size
    trainer.accumulate_minibatches = accumulation_count
    trainer.ema_max = 1.0
    trainer.vecenv = SimpleNamespace(single_observation_space=SimpleNamespace(shape=(1,)))

    # The two largest advantages are masked out of training, so they must not be selected.
    transition_count = valid_count + 2
    observations = torch.linspace(0.5, 1.5, transition_count).reshape(1, transition_count, 1)
    returns = 2.0 * observations.squeeze(-1)
    signs = torch.where(torch.arange(transition_count) % 2 == 0, 1.0, -1.0)
    advantages = (torch.arange(1, transition_count + 1, dtype=torch.float32) * signs).reshape(1, transition_count)
    masks = torch.ones_like(returns, dtype=torch.bool)
    masks[:, -2:] = False
    trainer.observations = observations
    trainer.actions = torch.zeros_like(returns, dtype=torch.int32)
    trainer.logprobs = torch.full_like(returns, -math.log(2.0))
    trainer.values = torch.zeros_like(returns)
    trainer._compute_advantages = Mock(return_value=(advantages, returns, masks))
    trainer.optimizer = torch.optim.SGD(trainer.policy.parameters(), lr=0.1)
    trained_observations = []
    original_ppo_loss = trainer._ppo_loss

    def record_ppo_loss(mb_obs, *loss_args):
        trained_observations.append(mb_obs.flatten())
        return original_ppo_loss(mb_obs, *loss_args)

    trainer._ppo_loss = record_ppo_loss
    optimizer_step_count = []
    trainer.optimizer.register_step_pre_hook(lambda *_: optimizer_step_count.append(1))

    losses = defaultdict(float)
    trained_transition_count = trainer._train_ppo_transition(losses, Mock(), epoch=1)

    update_epochs = trainer.config["update_epochs"]
    optimizer_batch_size = microbatch_size * accumulation_count
    assert trained_transition_count == update_epochs * trained_count_per_epoch
    assert len(optimizer_step_count) == update_epochs * math.ceil(trained_count_per_epoch / optimizer_batch_size)
    # A short batch keeps every valid row; a discarded partial minibatch then drops a random subset each PPO epoch.
    kept_observations = observations[0, valid_count - kept_count : valid_count, 0]
    epoch_observations = torch.cat(trained_observations).reshape(update_epochs, trained_count_per_epoch)
    assert torch.isin(epoch_observations, kept_observations).all()
    for ppo_epoch_observations in epoch_observations:
        assert ppo_epoch_observations.unique().numel() == trained_count_per_epoch
    assert losses["kept_fraction"] == kept_count / valid_count
    assert losses["filter_threshold"] == valid_count - kept_count + 1
    assert trainer.ema_max == 1.0
