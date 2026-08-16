"""Behavioral tests for the policy weight EMA."""

import copy

import pytest
import torch
from torch import nn

from pufferlib.ema import EMA


def _make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))


def test_shadow_starts_equal_and_lags_behind():
    model = _make_model()
    ema = EMA(model, decay=0.9)

    for shadow, live in zip(ema.ema_model.parameters(), model.parameters()):
        assert torch.equal(shadow, live), "shadow must start as a copy of the model"

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema.update()

    # One update at decay=0.9 moves the shadow a tenth of the way to the new weights.
    for shadow, live in zip(ema.ema_model.parameters(), model.parameters()):
        assert not torch.equal(shadow, live)
        assert torch.allclose(shadow, live - 0.9, atol=1e-6)


def test_shadow_converges_to_a_held_value():
    model = _make_model()
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(2.0)
    for _ in range(60):
        ema.update()
    for shadow in ema.ema_model.parameters():
        assert torch.allclose(shadow, torch.full_like(shadow, 2.0), atol=1e-6)


def test_update_does_not_touch_the_live_model():
    model = _make_model()
    ema = EMA(model, decay=0.9)
    before = copy.deepcopy([p.detach().clone() for p in model.parameters()])
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema.update()
    after = [p.detach().clone() for p in model.parameters()]
    for original, current in zip(before, after):
        assert torch.allclose(current, original + 1.0), "update must not write back into the live model"


def test_shadow_carries_no_grad():
    model = _make_model()
    ema = EMA(model, decay=0.9)
    for param in ema.ema_model.parameters():
        assert not param.requires_grad or param.grad is None
        assert param.is_leaf


def test_apply_shadow_overwrites_the_live_model():
    model = _make_model()
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema.update()
    ema.apply_shadow()
    for shadow, live in zip(ema.ema_model.parameters(), model.parameters()):
        assert torch.equal(shadow, live)


def test_round_trips_through_state_dict():
    model = _make_model()
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema.update()

    restored = EMA(_make_model(), decay=0.9)
    restored.load_state_dict(ema.state_dict())
    for a, b in zip(restored.ema_model.parameters(), ema.ema_model.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("decay", [-0.1, 1.0, 1.5])
def test_rejects_decay_outside_unit_interval(decay):
    with pytest.raises(ValueError):
        EMA(_make_model(), decay=decay)
