"""Lattice factored action (spline_werling): masking and the conditional log-prob / entropy used by PPO.

A cell only acts when its gate says "new" (the exit slot with the lateral gate), so its log-prob and entropy count
only then: keep rows give those logits exactly zero gradient, and masked choices are never sampled.
"""

import pytest
import torch

import pufferlib.pytorch as P
from pufferlib.pufferl import logits_to_float

NVEC = (2, 20, 2, 60, 5)
MASKED = -1e9


def _random_lattice_logits(batch, generator, masks=None):
    factors = [torch.randn(batch, n, generator=generator, requires_grad=True) for n in NVEC]
    if masks is None:
        return P.LatticeLogits(*factors), factors
    masked = [f.masked_fill(m < 0.5, MASKED) for f, m in zip(factors, masks)]
    return P.LatticeLogits(*masked), factors


def test_masked_choices_are_never_sampled():
    generator = torch.Generator().manual_seed(0)
    masks = [torch.zeros(64, n) for n in NVEC]
    allowed = [0, 7, 1, 55, 2]
    for mask, index in zip(masks, allowed):
        mask[:, index] = 1.0
    logits, _ = _random_lattice_logits(64, generator, masks)
    action, logprob, entropy, cont = P.sample_logits(logits)
    assert cont is None
    assert action.shape == (64, 5)
    assert (action == torch.tensor(allowed)).all()
    assert torch.allclose(logprob, torch.zeros(64), atol=1e-6)
    assert torch.allclose(entropy, torch.zeros(64), atol=1e-6)


def test_conditional_log_prob_and_zero_cell_gradient_on_keep_rows():
    generator = torch.Generator().manual_seed(1)
    logits, raw = _random_lattice_logits(8, generator)
    action = torch.tensor([[0, 3, 0, 10, 1]] * 4 + [[1, 3, 1, 10, 1]] * 4)
    _, logprob, _, _ = P.sample_logits(logits, action=action)
    log_softmax = [torch.log_softmax(f, dim=-1) for f in raw]
    keep_expected = log_softmax[0][:4, 0] + log_softmax[2][:4, 0]
    new_expected = log_softmax[0][4:, 1] + log_softmax[1][4:, 3] + log_softmax[2][4:, 1] + log_softmax[3][4:, 10] + log_softmax[4][4:, 1]
    assert torch.allclose(logprob[:4], keep_expected, atol=1e-5)
    assert torch.allclose(logprob[4:], new_expected, atol=1e-5)
    logprob[:4].sum().backward()
    assert raw[1].grad[:4].abs().max().item() == 0.0
    assert raw[3].grad[:4].abs().max().item() == 0.0
    assert raw[4].grad[:4].abs().max().item() == 0.0
    assert raw[0].grad[:4].abs().max().item() > 0.0


def test_entropy_weights_cells_by_detached_gate_probability():
    generator = torch.Generator().manual_seed(2)
    logits, raw = _random_lattice_logits(5, generator)
    action = torch.zeros(5, 5, dtype=torch.long)
    _, _, entropy, _ = P.sample_logits(logits, action=action)
    probs = [torch.softmax(f, dim=-1) for f in raw]
    ent = [-(p * torch.log(p)).sum(-1) for p in probs]
    expected = ent[0] + probs[0][:, 1] * ent[1] + ent[2] + probs[2][:, 1] * ent[3] + probs[0][:, 1] * ent[4]
    assert torch.allclose(entropy, expected, atol=1e-5)


def test_mode_is_argmax_and_mean_is_rejected():
    generator = torch.Generator().manual_seed(3)
    logits, raw = _random_lattice_logits(6, generator)
    action, _, _, _ = P.sample_logits(logits, action_selection=P.ACTION_SELECT_MODE)
    expected = torch.stack([f.argmax(-1) for f in raw], dim=-1)
    assert (action == expected).all()
    with pytest.raises(ValueError):
        P.sample_logits(logits, action_selection=P.ACTION_SELECT_MEAN)


def test_logits_to_float_keeps_the_lattice_container():
    generator = torch.Generator().manual_seed(4)
    logits, _ = _random_lattice_logits(3, generator)
    half = P.LatticeLogits(*(f.detach().to(torch.bfloat16) for f in logits))
    converted = logits_to_float(half)
    assert isinstance(converted, P.LatticeLogits)
    assert all(f.dtype == torch.float32 for f in converted)
