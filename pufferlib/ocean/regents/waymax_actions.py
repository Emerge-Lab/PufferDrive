"""Convert Waymax `InvertibleBicycleModel` actions into PufferDrive classic actions.

The reference ReGentS optimizes Waymax actions: raw acceleration in `m/s^2` and
raw path curvature in `1/m`, bounded at `6.0` and `0.3`. PufferDrive optimizes a
normalized acceleration and a normalized *target wheel angle* that the simulator
rate-limits. The two steering channels are different physical quantities, so any
reference action replayed here has to pass through this module first.

Curvature is `yaw_rate / speed`, which the classic model makes speed-independent:

    kappa(delta) = cos(atan(rear_axle_ratio * tan(delta))) * tan(delta) / wheelbase

That is monotonic in `delta`, so it inverts in closed form. Both directions are
differentiable, so an optimizer may parameterize steering in curvature space and
convert at the simulator boundary.
"""

import math
from dataclasses import dataclass

import torch

from pufferlib.ocean.regents.dynamics import (
    ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED,
    ACTION_FEATURE_COUNT,
    REAR_AXLE_RATIO,
    STEERING_LIMIT_RADIANS,
    TARGET_STEERING_SCALE_RADIANS,
)


# Waymax `InvertibleBicycleModel` defaults, which ReGentS constructs unmodified.
WAYMAX_MAXIMUM_ACCELERATION_MPS2 = 6.0
WAYMAX_MAXIMUM_CURVATURE_PER_METER = 0.3
WAYMAX_ACCELERATION = 0
WAYMAX_CURVATURE = 1

# `kappa * wheelbase` at full lock is `cos(atan(r * tan(limit))) * tan(limit)`, which
# is wheelbase-free. It bounds `rear_axle_ratio * kappa * wheelbase` well below one,
# so the inverse never approaches its singularity at `1 / (rear_axle_ratio * wheelbase)`.
NORMALIZED_CURVATURE_LIMIT = math.cos(math.atan(REAR_AXLE_RATIO * math.tan(STEERING_LIMIT_RADIANS))) * math.tan(
    STEERING_LIMIT_RADIANS
)


@dataclass(frozen=True)
class WaymaxActionConversion:
    """Converted PufferDrive action plus the reference authority it could not express."""

    action: torch.Tensor
    acceleration_saturated: torch.Tensor
    curvature_saturated: torch.Tensor

    def __post_init__(self):
        if self.action.shape[-1] != ACTION_FEATURE_COUNT:
            raise ValueError("action must have shape [..., 2]")
        for name in ("acceleration_saturated", "curvature_saturated"):
            mask = getattr(self, name)
            if mask.dtype != torch.bool or mask.shape != self.action.shape[:-1]:
                raise ValueError(f"{name} must be bool with the action's leading shape")


def _validate_curvature_inputs(curvature_per_meter, wheelbase_meters):
    for name, tensor in (
        ("curvature_per_meter", curvature_per_meter),
        ("wheelbase_meters", wheelbase_meters),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a Torch tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be torch.float32")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if curvature_per_meter.shape != wheelbase_meters.shape:
        raise ValueError("curvature_per_meter and wheelbase_meters must share a shape")
    if curvature_per_meter.device != wheelbase_meters.device:
        raise ValueError("curvature_per_meter and wheelbase_meters must share a device")
    if torch.any(wheelbase_meters <= 0.0):
        raise ValueError("wheelbase_meters must be positive")


def curvature_from_target_steering(steering_radians, wheelbase_meters):
    """Return the classic model's path curvature in `1/m` for a held wheel angle."""
    _validate_curvature_inputs(steering_radians, wheelbase_meters)
    tangent_steering = torch.tan(steering_radians)
    slip_angle = torch.atan(REAR_AXLE_RATIO * tangent_steering)
    return torch.cos(slip_angle) * tangent_steering / wheelbase_meters


def target_steering_from_curvature(curvature_per_meter, wheelbase_meters):
    """Invert `curvature_from_target_steering`, saturating at the wheel-angle limit.

    Returns the wheel angle in radians and the mask of entries whose requested
    curvature exceeded what the wheel can produce for that wheelbase.
    """
    _validate_curvature_inputs(curvature_per_meter, wheelbase_meters)
    achievable_curvature = NORMALIZED_CURVATURE_LIMIT / wheelbase_meters
    saturated = curvature_per_meter.abs() > achievable_curvature
    clamped = torch.clamp(curvature_per_meter, -achievable_curvature, achievable_curvature)
    normalized_curvature = clamped * wheelbase_meters
    # Bounded by NORMALIZED_CURVATURE_LIMIT, so the radicand stays far from zero.
    tangent_steering = normalized_curvature / torch.sqrt(1.0 - (REAR_AXLE_RATIO * normalized_curvature) ** 2)
    return torch.atan(tangent_steering), saturated


def drive_action_from_waymax_action(waymax_action, wheelbase_meters):
    """Convert a Waymax `[..., 2]` action into a normalized PufferDrive action.

    The reference action carries raw `m/s^2` and `1/m`; both channels saturate
    because PufferDrive allows less acceleration and less curvature. The returned
    steering is a *target* wheel angle: the simulator still rate-limits how fast
    the wheel reaches it, so a converted action is not guaranteed to reproduce the
    reference yaw rate on the step it is first applied.
    """
    if not isinstance(waymax_action, torch.Tensor):
        raise TypeError("waymax_action must be a Torch tensor")
    if waymax_action.dtype != torch.float32:
        raise TypeError("waymax_action must be torch.float32")
    if waymax_action.ndim < 1 or waymax_action.shape[-1] != ACTION_FEATURE_COUNT:
        raise ValueError("waymax_action must have shape [..., 2]")
    if not torch.isfinite(waymax_action).all():
        raise ValueError("waymax_action contains NaN or Inf")
    if not isinstance(wheelbase_meters, torch.Tensor) or wheelbase_meters.shape != waymax_action.shape[:-1]:
        raise ValueError("wheelbase_meters must be a tensor with the action's leading shape")

    acceleration_mps2 = waymax_action[..., WAYMAX_ACCELERATION]
    if torch.any(acceleration_mps2.abs() > WAYMAX_MAXIMUM_ACCELERATION_MPS2):
        raise ValueError("waymax_action acceleration exceeds the reference bound")
    curvature_per_meter = waymax_action[..., WAYMAX_CURVATURE]
    if torch.any(curvature_per_meter.abs() > WAYMAX_MAXIMUM_CURVATURE_PER_METER):
        raise ValueError("waymax_action curvature exceeds the reference bound")

    acceleration_action = acceleration_mps2 / ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED
    acceleration_saturated = acceleration_action.abs() > 1.0
    steering_radians, curvature_saturated = target_steering_from_curvature(curvature_per_meter, wheelbase_meters)
    steering_action = steering_radians / TARGET_STEERING_SCALE_RADIANS
    action = torch.stack(
        (torch.clamp(acceleration_action, -1.0, 1.0), torch.clamp(steering_action, -1.0, 1.0)),
        dim=-1,
    )
    return WaymaxActionConversion(
        action=action,
        acceleration_saturated=acceleration_saturated,
        curvature_saturated=curvature_saturated,
    )
