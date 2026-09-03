"""Differentiable scenario tooling for the offline ReGentS workflow."""

from pufferlib.ocean.regents.adapter import export_drive_scenarios
from pufferlib.ocean.regents.artifacts import (
    ARTIFACT_SCHEMA,
    load_generation_artifact,
    save_generation_artifact,
    source_configuration_hash,
)
from pufferlib.ocean.regents.dynamics import classic_rollout, classic_step
from pufferlib.ocean.regents.filters import (
    CandidateFilterReason,
    CandidateSelection,
    ReGentSFilterConfig,
    SceneFilterReason,
    front_divergence_mask,
    select_adversary_candidates,
)
from pufferlib.ocean.regents.generation import GenerationReport, generate_regents_scenarios, load_generation_config
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
from pufferlib.ocean.regents.optimizer import (
    CostSnapshot,
    FrozenEgoTrajectory,
    ReGentSOptimizationConfig,
    ReGentSOptimizationResult,
    capture_frozen_idm_trajectory,
    mask_front_divergence_gradients,
    optimize_frozen_ego_scenario,
)
from pufferlib.ocean.regents.rollout import (
    CReplayMetrics,
    CReplayResult,
    ReactiveGenerationResult,
    replay_optimized_scenario_in_c,
    run_reactive_idm_generation,
)
from pufferlib.ocean.regents.state import (
    DrivableAreaRaster,
    RasterTransform,
    ScenarioBatch,
    signed_speed_from_c_velocity,
)


__all__ = [
    "ARTIFACT_SCHEMA",
    "CReplayMetrics",
    "CReplayResult",
    "CandidateFilterReason",
    "CandidateSelection",
    "CostSnapshot",
    "DrivableAreaRaster",
    "FrozenEgoTrajectory",
    "GenerationReport",
    "InverseDynamicsResult",
    "RasterTransform",
    "ReGentSCostConfig",
    "ReGentSCosts",
    "ReGentSFilterConfig",
    "ReGentSOptimizationConfig",
    "ReGentSOptimizationResult",
    "ReactiveGenerationResult",
    "ScenarioBatch",
    "SceneFilterReason",
    "SmoothedOutOfBoundsRaster",
    "background_collision_avoidance_cost",
    "build_smoothed_out_of_bounds_raster",
    "capture_frozen_idm_trajectory",
    "classic_rollout",
    "classic_step",
    "combined_regents_cost",
    "drivable_area_deviation_cost",
    "ego_background_collision_cost",
    "estimate_expert_actions",
    "export_drive_scenarios",
    "front_divergence_mask",
    "generate_regents_scenarios",
    "load_generation_artifact",
    "load_generation_config",
    "mask_front_divergence_gradients",
    "optimize_frozen_ego_scenario",
    "oriented_box_corners",
    "prepare_out_of_bounds_rasters",
    "replay_optimized_scenario_in_c",
    "run_reactive_idm_generation",
    "sample_out_of_bounds_potential",
    "save_generation_artifact",
    "select_adversary_candidates",
    "signed_box_distance",
    "signed_speed_from_c_velocity",
    "source_configuration_hash",
]
