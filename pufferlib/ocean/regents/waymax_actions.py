"""Convert between path curvature and the PufferDrive classic target wheel angle.

The reference ReGentS optimizes raw path curvature in `1/m`, bounded at `0.3`.
PufferDrive optimizes a normalized *target wheel angle* that the simulator
rate-limits. The two steering channels are different physical quantities, so the
optimizer parameterizes steering in curvature space and converts at the boundary.

Curvature is `yaw_rate / speed`, which the classic model makes speed-independent:

    kappa(delta) = cos(atan(rear_axle_ratio * tan(delta))) * tan(delta) / wheelbase

That is monotonic in `delta`, so it inverts in closed form. Both directions are
differentiable, so an optimizer may parameterize steering in curvature space and
convert at the simulator boundary.
"""

import math

import torch

from pufferlib.ocean.regents.dynamics import REAR_AXLE_RATIO, STEERING_LIMIT_RADIANS


# The Waymax `InvertibleBicycleModel` curvature bound, which ReGentS uses unmodified.
WAYMAX_MAXIMUM_CURVATURE_PER_METER = 0.3

# `kappa * wheelbase` at full lock is `cos(atan(r * tan(limit))) * tan(limit)`, which
# is wheelbase-free. It bounds `rear_axle_ratio * kappa * wheelbase` well below one,
# so the inverse never approaches its singularity at `1 / (rear_axle_ratio * wheelbase)`.
NORMALIZED_CURVATURE_LIMIT = math.cos(math.atan(REAR_AXLE_RATIO * math.tan(STEERING_LIMIT_RADIANS))) * math.tan(
    STEERING_LIMIT_RADIANS
)


def curvature_from_target_steering(steering_radians, wheelbase_meters):
    """Return the classic model's path curvature in `1/m` for a held wheel angle."""
    tangent_steering = torch.tan(steering_radians)
    slip_angle = torch.atan(REAR_AXLE_RATIO * tangent_steering)
    return torch.cos(slip_angle) * tangent_steering / wheelbase_meters


def target_steering_from_curvature(curvature_per_meter, wheelbase_meters):
    """Invert `curvature_from_target_steering`, saturating at the wheel-angle limit.

    Returns the wheel angle in radians and the mask of entries whose requested
    curvature exceeded what the wheel can produce for that wheelbase.
    """
    achievable_curvature = NORMALIZED_CURVATURE_LIMIT / wheelbase_meters
    saturated = curvature_per_meter.abs() > achievable_curvature
    clamped = torch.clamp(curvature_per_meter, -achievable_curvature, achievable_curvature)
    normalized_curvature = clamped * wheelbase_meters
    # Bounded by NORMALIZED_CURVATURE_LIMIT, so the radicand stays far from zero.
    tangent_steering = normalized_curvature / torch.sqrt(1.0 - (REAR_AXLE_RATIO * normalized_curvature) ** 2)
    return torch.atan(tangent_steering), saturated
