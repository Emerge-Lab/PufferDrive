"""Differentiable scenario tooling for the offline ReGentS workflow."""

from pufferlib.ocean.regents.adapter import export_drive_scenarios
from pufferlib.ocean.regents.dynamics import classic_rollout, classic_step
from pufferlib.ocean.regents.geometry import (
    SmoothedOutOfBoundsRaster,
    build_smoothed_out_of_bounds_raster,
    oriented_box_corners,
    sample_out_of_bounds_potential,
    signed_box_distance,
)
from pufferlib.ocean.regents.inverse_dynamics import InverseDynamicsResult, estimate_expert_actions
from pufferlib.ocean.regents.losses import (
    ReGentSCostConfig,
    ReGentSCosts,
    background_collision_avoidance_cost,
    combined_regents_cost,
    drivable_area_deviation_cost,
    ego_background_collision_cost,
    prepare_out_of_bounds_rasters,
)
from pufferlib.ocean.regents.state import DrivableAreaRaster, RasterTransform, ScenarioBatch


__all__ = [
    "DrivableAreaRaster",
    "InverseDynamicsResult",
    "RasterTransform",
    "ReGentSCostConfig",
    "ReGentSCosts",
    "ScenarioBatch",
    "SmoothedOutOfBoundsRaster",
    "background_collision_avoidance_cost",
    "build_smoothed_out_of_bounds_raster",
    "classic_rollout",
    "classic_step",
    "combined_regents_cost",
    "drivable_area_deviation_cost",
    "ego_background_collision_cost",
    "estimate_expert_actions",
    "export_drive_scenarios",
    "oriented_box_corners",
    "prepare_out_of_bounds_rasters",
    "sample_out_of_bounds_potential",
    "signed_box_distance",
]
