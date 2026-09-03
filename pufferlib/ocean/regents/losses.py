"""Independent KING/ReGentS costs over canonical Torch scenario states."""

import math
from dataclasses import dataclass

import torch

from pufferlib.ocean.regents.geometry import (
    DEFAULT_GAUSSIAN_SIGMA_METERS,
    DEFAULT_GAUSSIAN_TRUNCATE_SIGMA,
    SmoothedOutOfBoundsRaster,
    build_smoothed_out_of_bounds_raster,
    oriented_box_corners,
    sample_out_of_bounds_potential,
    signed_box_distance,
)
from pufferlib.ocean.regents.state import STATE_HEADING, STATE_X, STATE_Y, DrivableAreaRaster


DEFAULT_EGO_COLLISION_WEIGHT = 1.0
DEFAULT_BACKGROUND_COLLISION_WEIGHT = 5.0
DEFAULT_DRIVABLE_AREA_WEIGHT = 20.0
DEFAULT_BACKGROUND_DISTANCE_TRUNCATION_METERS = 1.25
SAFE_MASKED_BOX_SIZE_METERS = 1.0
PAIRWISE_DISTANCE_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class ReGentSCostConfig:
    """Named paper-cost weights and KING smoothing/truncation parameters."""

    ego_collision_weight: float = DEFAULT_EGO_COLLISION_WEIGHT
    background_collision_weight: float = DEFAULT_BACKGROUND_COLLISION_WEIGHT
    drivable_area_weight: float = DEFAULT_DRIVABLE_AREA_WEIGHT
    background_distance_truncation_meters: float = DEFAULT_BACKGROUND_DISTANCE_TRUNCATION_METERS
    gaussian_sigma_meters: float = DEFAULT_GAUSSIAN_SIGMA_METERS
    gaussian_truncate_sigma: float = DEFAULT_GAUSSIAN_TRUNCATE_SIGMA

    def __post_init__(self):
        for name in (
            "ego_collision_weight",
            "background_collision_weight",
            "drivable_area_weight",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "background_distance_truncation_meters",
            "gaussian_sigma_meters",
            "gaussian_truncate_sigma",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ReGentSCosts:
    ego_collision: torch.Tensor
    background_collision: torch.Tensor
    drivable_area: torch.Tensor
    total: torch.Tensor


def prepare_out_of_bounds_rasters(
    drivable_area_rasters,
    config=None,
    *,
    device=None,
    dtype=torch.float32,
):
    """Precompute all map-static Gaussian potentials before optimization."""
    if config is None:
        config = ReGentSCostConfig()
    if not isinstance(config, ReGentSCostConfig):
        raise TypeError("config must be a ReGentSCostConfig")
    if not isinstance(drivable_area_rasters, (tuple, list)) or not drivable_area_rasters:
        raise ValueError("drivable_area_rasters must be a non-empty sequence")
    if not all(isinstance(raster, DrivableAreaRaster) for raster in drivable_area_rasters):
        raise TypeError("Every drivable-area raster must be a DrivableAreaRaster")
    return tuple(
        build_smoothed_out_of_bounds_raster(
            raster,
            config.gaussian_sigma_meters,
            config.gaussian_truncate_sigma,
            device=device,
            dtype=dtype,
        )
        for raster in drivable_area_rasters
    )


def _validate_common_inputs(states, state_valid, length_meters, width_meters):
    if not isinstance(states, torch.Tensor) or not states.is_floating_point():
        raise TypeError("states must be a floating Torch tensor")
    if states.ndim != 4 or states.shape[-1] < 3:
        raise ValueError("states must have shape [batch, agent, time, feature>=3]")
    expected_state_shape = states.shape[:-1]
    if not isinstance(state_valid, torch.Tensor) or state_valid.dtype != torch.bool:
        raise TypeError("state_valid must be a bool Torch tensor")
    if tuple(state_valid.shape) != expected_state_shape or state_valid.device != states.device:
        raise ValueError("state_valid must match [batch, agent, time] on the states device")
    expected_agent_shape = states.shape[:2]
    for name, dimensions in (("length_meters", length_meters), ("width_meters", width_meters)):
        if not isinstance(dimensions, torch.Tensor) or dimensions.dtype != states.dtype:
            raise TypeError(f"{name} must share the states floating dtype")
        if dimensions.device != states.device or tuple(dimensions.shape) != expected_agent_shape:
            raise ValueError(f"{name} must have shape [batch, agent] on the states device")
    if not torch.isfinite(states).all():
        raise ValueError("states must be finite, including masked storage")


def _validate_agent_mask(mask, states, name):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise TypeError(f"{name} must be a bool Torch tensor")
    if tuple(mask.shape) != states.shape[:2] or mask.device != states.device:
        raise ValueError(f"{name} must have shape [batch, agent] on the states device")


def _masked_boxes(states, state_valid, length_meters, width_meters):
    expanded_length = length_meters[..., None].expand(states.shape[:-1])
    expanded_width = width_meters[..., None].expand(states.shape[:-1])
    if torch.any(expanded_length[state_valid] <= 0) or torch.any(expanded_width[state_valid] <= 0):
        raise ValueError("Valid states require positive agent length and width")
    boxes = torch.stack(
        (
            states[..., STATE_X],
            states[..., STATE_Y],
            expanded_length,
            expanded_width,
            states[..., STATE_HEADING],
        ),
        dim=-1,
    )
    safe_box = boxes.new_tensor((0.0, 0.0, SAFE_MASKED_BOX_SIZE_METERS, SAFE_MASKED_BOX_SIZE_METERS, 0.0))
    return torch.where(state_valid[..., None], boxes, safe_box)


def ego_background_collision_cost(
    states,
    state_valid,
    length_meters,
    width_meters,
    ego_mask,
    candidate_adversary_mask,
):
    """Return per-scenario minimum candidate time-averaged signed box distance."""
    _validate_common_inputs(states, state_valid, length_meters, width_meters)
    _validate_agent_mask(ego_mask, states, "ego_mask")
    _validate_agent_mask(candidate_adversary_mask, states, "candidate_adversary_mask")
    if not torch.all(ego_mask.sum(dim=-1) == 1):
        raise ValueError("Each scenario must contain exactly one ego agent")
    if torch.any(ego_mask & candidate_adversary_mask):
        raise ValueError("The ego agent cannot be a candidate adversary")

    boxes = _masked_boxes(states, state_valid, length_meters, width_meters)
    batch_indices = torch.arange(states.shape[0], device=states.device)
    ego_indices = torch.argmax(ego_mask.to(torch.int64), dim=-1)
    ego_boxes = boxes[batch_indices, ego_indices]
    ego_valid = state_valid[batch_indices, ego_indices]
    distances = signed_box_distance(boxes, ego_boxes[:, None])
    joint_valid = state_valid & ego_valid[:, None] & candidate_adversary_mask[..., None]
    valid_counts = joint_valid.sum(dim=-1)
    summed_distances = torch.where(joint_valid, distances, torch.zeros_like(distances)).sum(dim=-1)
    averaged_distances = summed_distances / valid_counts.clamp_min(1)
    averaged_distances = averaged_distances.masked_fill(valid_counts == 0, torch.inf)
    if torch.any(~torch.isfinite(averaged_distances.min(dim=-1).values)):
        raise ValueError("Every scenario needs a candidate with at least one jointly valid ego timestep")
    return averaged_distances.min(dim=-1).values


def background_collision_avoidance_cost(
    states,
    state_valid,
    length_meters,
    width_meters,
    background_mask,
    truncation_meters=DEFAULT_BACKGROUND_DISTANCE_TRUNCATION_METERS,
):
    """Return negative truncated minimum distance among distinct backgrounds."""
    _validate_common_inputs(states, state_valid, length_meters, width_meters)
    _validate_agent_mask(background_mask, states, "background_mask")
    if not isinstance(truncation_meters, (float, int)) or not math.isfinite(truncation_meters):
        raise ValueError("truncation_meters must be a finite positive scalar")
    if truncation_meters <= 0:
        raise ValueError("truncation_meters must be a finite positive scalar")

    boxes = _masked_boxes(states, state_valid, length_meters, width_meters)
    scenario_costs = []
    for scenario_idx in range(states.shape[0]):
        background_indices = torch.where(background_mask[scenario_idx])[0]
        if background_indices.numel() < 2:
            scenario_costs.append(states.new_zeros(()))
            continue
        local_pairs = torch.triu_indices(
            background_indices.numel(),
            background_indices.numel(),
            offset=1,
            device=states.device,
        )
        pair_indices = background_indices[local_pairs]
        chunk_minima = []
        for chunk_start in range(0, pair_indices.shape[1], PAIRWISE_DISTANCE_CHUNK_SIZE):
            chunk_pairs = pair_indices[:, chunk_start : chunk_start + PAIRWISE_DISTANCE_CHUNK_SIZE]
            first_indices, second_indices = chunk_pairs
            distances = signed_box_distance(
                boxes[scenario_idx, first_indices],
                boxes[scenario_idx, second_indices],
            )
            pair_valid = state_valid[scenario_idx, first_indices] & state_valid[scenario_idx, second_indices]
            truncated = torch.clamp_max(distances, float(truncation_meters))
            chunk_minima.append(torch.where(pair_valid, truncated, torch.full_like(truncated, torch.inf)).min())
        minimum = torch.stack(chunk_minima).min()
        scenario_costs.append(torch.where(torch.isfinite(minimum), -minimum, states.new_zeros(())))
    return torch.stack(scenario_costs)


def drivable_area_deviation_cost(
    states,
    state_valid,
    length_meters,
    width_meters,
    optimized_vehicle_mask,
    out_of_bounds_rasters,
):
    """Sum each optimized vehicle's four-corner potential averaged over valid steps."""
    _validate_common_inputs(states, state_valid, length_meters, width_meters)
    _validate_agent_mask(optimized_vehicle_mask, states, "optimized_vehicle_mask")
    if len(out_of_bounds_rasters) != states.shape[0]:
        raise ValueError("out_of_bounds_rasters must contain one raster per scenario")
    boxes = _masked_boxes(states, state_valid, length_meters, width_meters)
    corners = oriented_box_corners(boxes)

    scenario_costs = []
    for scenario_idx, raster in enumerate(out_of_bounds_rasters):
        if not isinstance(raster, SmoothedOutOfBoundsRaster):
            raise TypeError("Every out-of-bounds raster must be precomputed and smoothed")
        potential = raster.potential
        if potential.device != states.device or potential.dtype != states.dtype:
            raise ValueError("Out-of-bounds rasters and states must share device and dtype")
        corner_potential = sample_out_of_bounds_potential(corners[scenario_idx], raster)
        valid = state_valid[scenario_idx] & optimized_vehicle_mask[scenario_idx, :, None]
        valid_counts = valid.sum(dim=-1)
        potential_sum = torch.where(
            valid[..., None],
            corner_potential,
            torch.zeros_like(corner_potential),
        ).sum(dim=(-1, -2))
        per_vehicle_cost = potential_sum / valid_counts.clamp_min(1)
        scenario_costs.append(torch.where(valid_counts > 0, per_vehicle_cost, 0.0).sum())
    return torch.stack(scenario_costs)


def combined_regents_cost(
    states,
    state_valid,
    length_meters,
    width_meters,
    ego_mask,
    candidate_adversary_mask,
    background_mask,
    optimized_vehicle_mask,
    out_of_bounds_rasters,
    config=None,
):
    """Evaluate and combine the three paper-equivalent costs per scenario."""
    if config is None:
        config = ReGentSCostConfig()
    if not isinstance(config, ReGentSCostConfig):
        raise TypeError("config must be a ReGentSCostConfig")
    _validate_common_inputs(states, state_valid, length_meters, width_meters)
    for name, mask in (
        ("ego_mask", ego_mask),
        ("candidate_adversary_mask", candidate_adversary_mask),
        ("background_mask", background_mask),
        ("optimized_vehicle_mask", optimized_vehicle_mask),
    ):
        _validate_agent_mask(mask, states, name)
    if not isinstance(out_of_bounds_rasters, (tuple, list)):
        raise TypeError("out_of_bounds_rasters must be a sequence")
    for raster in out_of_bounds_rasters:
        if not isinstance(raster, SmoothedOutOfBoundsRaster):
            raise TypeError("Every out-of-bounds raster must be precomputed and smoothed")
        if not math.isclose(raster.gaussian_sigma_meters, config.gaussian_sigma_meters):
            raise ValueError("Out-of-bounds raster Gaussian sigma does not match the cost config")
        if not math.isclose(raster.gaussian_truncate_sigma, config.gaussian_truncate_sigma):
            raise ValueError("Out-of-bounds raster Gaussian truncation does not match the cost config")
    if torch.any(ego_mask & background_mask):
        raise ValueError("The ego agent cannot be included in background_mask")
    if torch.any(optimized_vehicle_mask & ~background_mask):
        raise ValueError("optimized_vehicle_mask must be a subset of background_mask")
    ego_collision = ego_background_collision_cost(
        states,
        state_valid,
        length_meters,
        width_meters,
        ego_mask,
        candidate_adversary_mask,
    )
    background_collision = background_collision_avoidance_cost(
        states,
        state_valid,
        length_meters,
        width_meters,
        background_mask,
        config.background_distance_truncation_meters,
    )
    drivable_area = drivable_area_deviation_cost(
        states,
        state_valid,
        length_meters,
        width_meters,
        optimized_vehicle_mask,
        out_of_bounds_rasters,
    )
    total = (
        config.ego_collision_weight * ego_collision
        + config.background_collision_weight * background_collision
        + config.drivable_area_weight * drivable_area
    )
    return ReGentSCosts(ego_collision, background_collision, drivable_area, total)
