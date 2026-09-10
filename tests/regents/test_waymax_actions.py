import math

import pytest
import torch

from pufferlib.ocean.regents.dynamics import (
    STEERING_LIMIT_RADIANS,
    STEERING_RATE_LIMIT_RADIANS_PER_SECOND,
    TARGET_STEERING_SCALE_RADIANS,
    classic_step,
)
from pufferlib.ocean.regents.state import STATE_HEADING, STATE_SPEED, STATE_STEERING
from pufferlib.ocean.regents.waymax_actions import (
    NORMALIZED_CURVATURE_LIMIT,
    WAYMAX_MAXIMUM_CURVATURE_PER_METER,
    curvature_from_target_steering,
    target_steering_from_curvature,
)


DT_SECONDS = 0.1
TYPICAL_WHEELBASE_METERS = 3.0


def _wheelbase(shape, meters=TYPICAL_WHEELBASE_METERS):
    return torch.full(shape, meters, dtype=torch.float32)


def test_curvature_wheel_angle_conversion_round_trips_saturates_and_differentiates():
    """The closed-form inverse, its wheelbase scaling, its limit, and its gradients."""
    steering = torch.linspace(-STEERING_LIMIT_RADIANS, STEERING_LIMIT_RADIANS, 33, dtype=torch.float32)
    curvature = curvature_from_target_steering(steering, _wheelbase(steering.shape))
    recovered, saturated = target_steering_from_curvature(curvature, _wheelbase(steering.shape))
    assert not saturated.any()
    torch.testing.assert_close(recovered, steering, rtol=0.0, atol=1e-6)

    # Curvature is inversely proportional to wheelbase at a fixed wheel angle.
    wheelbase = torch.tensor([2.4, 2.8, 3.1, 5.5, 8.0], dtype=torch.float32)
    fixed_steering = torch.full_like(wheelbase, 0.4)
    scaled = curvature_from_target_steering(fixed_steering, wheelbase) * wheelbase
    torch.testing.assert_close(scaled, torch.full_like(wheelbase, float(scaled[0])))
    recovered, _ = target_steering_from_curvature(scaled / wheelbase, wheelbase)
    torch.testing.assert_close(recovered, fixed_steering, rtol=0.0, atol=1e-6)

    # The limit constant is the wheelbase-free product that keeps the inverse regular.
    unit_wheelbase = _wheelbase((1,), 1.0)
    assert curvature_from_target_steering(
        torch.tensor([STEERING_LIMIT_RADIANS], dtype=torch.float32), unit_wheelbase
    ).item() == pytest.approx(NORMALIZED_CURVATURE_LIMIT, abs=1e-6)
    assert NORMALIZED_CURVATURE_LIMIT == pytest.approx(
        math.cos(math.atan(0.5 * math.tan(STEERING_LIMIT_RADIANS))) * math.tan(STEERING_LIMIT_RADIANS)
    )

    # ReGentS allows a tighter turn than a typical car's wheel angle can produce.
    achievable = NORMALIZED_CURVATURE_LIMIT / TYPICAL_WHEELBASE_METERS
    assert achievable < WAYMAX_MAXIMUM_CURVATURE_PER_METER
    requested = torch.tensor([achievable * 0.5, achievable, WAYMAX_MAXIMUM_CURVATURE_PER_METER], dtype=torch.float32)
    steering, saturated = target_steering_from_curvature(requested, _wheelbase((3,)))
    assert saturated.tolist() == [False, False, True]
    assert steering[-1].item() == pytest.approx(STEERING_LIMIT_RADIANS, abs=1e-6)

    # Extreme input stays finite: clamping first keeps the radicand away from zero.
    extreme = torch.tensor([-1e3, -0.3, 0.3, 1e3], dtype=torch.float32)
    steering, saturated = target_steering_from_curvature(extreme, _wheelbase((4,), 8.0))
    assert saturated.all() and torch.isfinite(steering).all()
    torch.testing.assert_close(steering.abs(), torch.full_like(steering, STEERING_LIMIT_RADIANS), rtol=0.0, atol=1e-6)

    differentiable_steering = torch.tensor([0.25], dtype=torch.float32, requires_grad=True)
    curvature_from_target_steering(differentiable_steering, _wheelbase((1,))).sum().backward()
    assert torch.isfinite(differentiable_steering.grad).all()
    assert differentiable_steering.grad.abs().item() > 0.0
    differentiable_curvature = torch.tensor([0.1], dtype=torch.float32, requires_grad=True)
    target_steering_from_curvature(differentiable_curvature, _wheelbase((1,)))[0].sum().backward()
    assert torch.isfinite(differentiable_curvature.grad).all()
    assert differentiable_curvature.grad.abs().item() > 0.0


def test_converted_wheel_angle_realizes_the_requested_curvature_in_the_simulator():
    """A held wheel angle realizes the requested yaw-per-metre, up to the rate limit."""
    curvature = torch.tensor([[-0.20, -0.05, 0.0, 0.05, 0.20]], dtype=torch.float32)
    held_wheelbase = _wheelbase(curvature.shape)
    steering, saturated = target_steering_from_curvature(curvature, held_wheelbase)
    assert not saturated.any()
    speed_mps = 8.0
    state = torch.zeros((*curvature.shape, 5), dtype=torch.float32)
    state[..., STATE_SPEED] = speed_mps
    state[..., STATE_STEERING] = steering
    action = torch.stack((torch.zeros_like(steering), steering / TARGET_STEERING_SCALE_RADIANS), dim=-1)
    next_state = classic_step(state, action, held_wheelbase, _wheelbase(curvature.shape, 30.0), DT_SECONDS)
    realized = next_state[..., STATE_HEADING] / (speed_mps * DT_SECONDS)
    torch.testing.assert_close(realized, curvature, rtol=0.0, atol=1e-5)

    # But the converter cannot undo the simulator's steering rate limit.
    from_neutral = torch.zeros((1, 5), dtype=torch.float32)
    from_neutral[..., STATE_SPEED] = speed_mps
    target, _ = target_steering_from_curvature(torch.tensor([0.2], dtype=torch.float32), _wheelbase((1,)))
    assert target.item() > STEERING_RATE_LIMIT_RADIANS_PER_SECOND * DT_SECONDS
    stepped = classic_step(
        from_neutral,
        torch.stack((torch.zeros_like(target), target / TARGET_STEERING_SCALE_RADIANS), dim=-1),
        _wheelbase((1,)),
        _wheelbase((1,), 30.0),
        DT_SECONDS,
    )
    assert stepped[0, STATE_STEERING].item() == pytest.approx(
        STEERING_RATE_LIMIT_RADIANS_PER_SECOND * DT_SECONDS, abs=1e-6
    )
    assert stepped[0, STATE_STEERING].item() < target.item()
