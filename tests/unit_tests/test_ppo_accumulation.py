"""Filtered PPO batches keep epoch boundaries and sample weights when accumulated."""

import contextlib
import math
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from pufferlib.pufferl import PuffeRL


class ValuePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.constant_(self.value.weight, 0.25)

    def forward(self, observations, state=None):
        return observations.new_zeros((observations.shape[0], 2)), self.value(observations)


@pytest.mark.parametrize(
    "retained_count, microbatch_size, accumulation_count",
    [(6, 8, 4), (6, 4, 2), (10, 4, 2), (8, 4, 2), (10, 4, 1), (0, 4, 2)],
)
def test_filtered_ppo_accumulation_matches_optimizer_batches(retained_count, microbatch_size, accumulation_count):
    trainer = PuffeRL.__new__(PuffeRL)
    trainer.policy = ValuePolicy()
    trainer.config = {
        "device": "cpu",
        "cpu_offload": False,
        "update_epochs": 3,
        "adv_filter_enabled": True,
        "adv_filter_ewma_beta": 0.25,
        "adv_filter_threshold_scale": 0.01,
        "adv_filter_leak_fraction": 0.0,
        "min_batch_size": None,
        "clip_coef": 0.2,
        "vf_clip_coef": None,
        "vf_coef": 1.0,
        "ent_coef": 0.0,
        "ent_coef_anneal": False,
        "max_grad_norm": 1e6,
    }
    trainer.amp_context = contextlib.nullcontext()
    trainer.compress_observations = False
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

    # Compare against ordinary SGD on each full optimizer batch, including its partial tail.
    optimizer_batch_size = microbatch_size * accumulation_count
    reference_policy = ValuePolicy()
    reference_optimizer = torch.optim.SGD(reference_policy.parameters(), lr=0.1)
    reference_gradients = []
    random_seed = 7
    torch.manual_seed(random_seed)
    for _ in range(trainer.config["update_epochs"]):
        permutation = torch.randperm(retained_count)
        for start in range(0, retained_count, optimizer_batch_size):
            indices = permutation[start : start + optimizer_batch_size]
            predictions = reference_policy.value(observations[0, indices]).squeeze(-1)
            loss = 0.5 * (predictions - returns[0, indices]).square().mean()
            loss.backward()
            reference_gradients.append(reference_policy.value.weight.grad.detach().clone())
            reference_optimizer.step()
            reference_optimizer.zero_grad()

    torch.manual_seed(random_seed)
    trainer._train_ppo_transition(defaultdict(float), Mock(), epoch=1)

    expected_updates = trainer.config["update_epochs"] * math.ceil(retained_count / optimizer_batch_size)
    assert len(gradients) == expected_updates
    for gradient, reference_gradient in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(gradient, reference_gradient)
    torch.testing.assert_close(trainer.policy.value.weight, reference_policy.value.weight)
