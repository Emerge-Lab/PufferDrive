"""PuffeRL._ppo_loss stops on a non-finite loss instead of letting NaN reach the optimizer."""

import contextlib

import pytest
import torch

from pufferlib.pufferl import PuffeRL, minibatch_chunks


class _NanValuePolicy(torch.nn.Module):
    def forward(self, observations, state):
        batch = observations.shape[0]
        return torch.zeros(batch, 3), torch.full((batch, 1), float("nan"))


class _Trainer:
    compress_observations = False
    amp_context = contextlib.nullcontext()
    config = {"clip_coef": 0.2, "vf_clip_coef": None, "vf_coef": 0.5, "ent_coef": 0.01}
    policy = _NanValuePolicy()


def _batch(batch=8):
    return dict(
        mb_obs=torch.zeros(batch, 4),
        mb_actions=torch.zeros(batch, dtype=torch.long),
        mb_logprobs=torch.full((batch,), -1.0986),
        mb_values=torch.zeros(batch),
        mb_returns=torch.zeros(batch),
        mb_adv=torch.linspace(-1.0, 1.0, batch),
    )


def test_non_finite_loss_raises_with_its_source():
    with pytest.raises(RuntimeError, match="new values False"):
        PuffeRL._ppo_loss(_Trainer(), **_batch())


def test_finite_loss_passes():
    trainer = _Trainer()
    trainer.policy = lambda observations, state: (torch.zeros(observations.shape[0], 3), torch.zeros(observations.shape[0], 1))
    loss, _, _, _ = PuffeRL._ppo_loss(trainer, **_batch())
    assert torch.isfinite(loss)


@pytest.mark.parametrize("count", [1, 2, 65536, 65537, 5 * 65536 + 1, 6 * 65536 + 1, 1_500_000])
def test_minibatches_never_hold_a_single_sample(count):
    chunks = minibatch_chunks(torch.arange(count), 65536)
    sizes = [chunk.numel() for chunk in chunks]
    assert all(2 <= size <= 65536 for size in sizes)
    assert sum(sizes) == (count if count >= 2 else 0)
    assert len(sizes) == (-(-count // 65536) if count >= 2 else 0)
