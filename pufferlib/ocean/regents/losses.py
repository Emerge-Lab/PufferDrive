"""Independent KING/ReGentS costs over canonical Torch scenario states."""

import math
from dataclasses import dataclass

import torch

from pufferlib.ocean.regents.geometry import (
    DEFAULT_GAUSSIAN_SIGMA_METERS,
    DEFAULT_GAUSSIAN_TRUNCATE_SIGMA,
    SmoothedOutOfBoundsRaster,
    box_separation_lower_bound,
    build_smoothed_out_of_bounds_raster,
    oriented_box_corners,
    sample_out_of_bounds_potential,
    signed_box_distance,
)
from pufferlib.ocean.regents.state import STATE_HEADING, STATE_X, STATE_Y, DrivableAreaRaster


DEFAULT_EGO_COLLISION_WEIGHT = 1.0
# Released ReGentS `conf/config_scenario_opt.yaml`: adversary collision 5 and
# adversary deviation 20; the ego-collision term has an implicit coefficient 1.
DEFAULT_BACKGROUND_COLLISION_WEIGHT = 5.0
DEFAULT_DRIVABLE_AREA_WEIGHT = 20.0
DEFAULT_BACKGROUND_DISTANCE_TRUNCATION_METERS = 1.25
SAFE_MASKED_BOX_SIZE_METERS = 1.0
PAIRWISE_DISTANCE_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class ReGentSCostConfig:
    """Released ReGentS weights and Gaussian/truncation parameters."""

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
    background_collision_first_agent_idx: torch.Tensor
    background_collision_second_agent_idx: torch.Tensor
    background_collision_timestep_idx: torch.Tensor
    background_collision_signed_distance_meters: torch.Tensor
    background_collision_truncated: torch.Tensor


def prepare_out_of_bounds_raster(
    drivable_area_raster,
    config=None,
    *,
    device=None,
    dtype=torch.float32,
):
    """Precompute the map-static Gaussian potential before optimization.

    The kernel carries unit mass, so a fully out-of-bounds corner costs exactly one
    whatever the raster resolution. The released amplitude has neither discrete mass
    normalization nor a pixel-area factor, which makes its potential scale with the
    inverse square of the resolution; that scale is not reproducible here because the
    released deviation term never evaluates at the sampled position.
    """
    if config is None:
        config = ReGentSCostConfig()
    if not isinstance(config, ReGentSCostConfig):
        raise TypeError("config must be a ReGentSCostConfig")
    if not isinstance(drivable_area_raster, DrivableAreaRaster):
        raise TypeError("drivable_area_raster must be a DrivableAreaRaster")
    return build_smoothed_out_of_bounds_raster(
        drivable_area_raster,
        config.gaussian_sigma_meters,
        config.gaussian_truncate_sigma,
        device=device,
        dtype=dtype,
        normalize_kernel=True,
    )


def _masked_boxes(states, state_valid, length_meters, width_meters):
    expanded_length = length_meters[..., None].expand(states.shape[:-1])
    expanded_width = width_meters[..., None].expand(states.shape[:-1])
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


def mean_candidate_ego_distances(boxes, state_valid, ego_idx, candidate_indices):
    """Each candidate's signed box distance to the ego, averaged over jointly valid time.

    A candidate sharing no valid timestep with the ego averages to infinity, so it loses
    both the minimum this cost takes and the argmin adversary selection takes.
    """
    joint_valid = state_valid[candidate_indices] & state_valid[ego_idx, None]
    valid_counts = joint_valid.sum(dim=-1)
    distances = signed_box_distance(boxes[candidate_indices], boxes[ego_idx, None])
    summed_distances = torch.where(joint_valid, distances, torch.zeros_like(distances)).sum(dim=-1)
    averaged_distances = summed_distances / valid_counts.clamp_min(1)
    return averaged_distances.masked_fill(valid_counts == 0, torch.inf)


def ego_background_collision_cost(
    states,
    state_valid,
    length_meters,
    width_meters,
    ego_mask,
    candidate_adversary_mask,
    boxes=None,
):
    """Return the paper's minimum candidate mean signed box distance.

    ``boxes`` lets `combined_regents_cost` share one already-built box tensor
    across the three terms instead of rebuilding it per term.
    """
    if boxes is None:
        boxes = _masked_boxes(states, state_valid, length_meters, width_meters)
    ego_idx = int(torch.argmax(ego_mask.to(torch.int64)).item())
    candidate_indices = torch.where(candidate_adversary_mask)[0]
    if candidate_indices.numel() == 0:
        raise ValueError("The scenario needs a candidate with at least one jointly valid ego timestep")
    minimum = mean_candidate_ego_distances(boxes, state_valid, ego_idx, candidate_indices).min()
    if not bool(torch.isfinite(minimum)):
        raise ValueError("The scenario needs a candidate with at least one jointly valid ego timestep")
    return minimum


def _truncated_signed_box_distances(boxes_a, boxes_b, valid, truncation_meters):
    """Evaluate exact box clearance only where the truncated loss can have a gradient."""
    # Retain a zero-gradient graph for fully truncated batches so this remains a
    # differentiable loss even when no pair needs exact polygon geometry.
    distances = (boxes_a[..., 0] + boxes_b[..., 0]) * 0.0 + float(truncation_meters)
    potentially_active = valid & (box_separation_lower_bound(boxes_a, boxes_b) < truncation_meters)
    active_indices = torch.where(potentially_active)
    if active_indices[0].numel():
        exact_distances = signed_box_distance(boxes_a[active_indices], boxes_b[active_indices])
        distances = distances.index_put(
            active_indices,
            torch.clamp_max(exact_distances, float(truncation_meters)),
        )
    return torch.where(valid, distances, torch.full_like(distances, torch.inf))


def _background_collision_avoidance_cost_and_diagnostics(
    states,
    state_valid,
    length_meters,
    width_meters,
    candidate_adversary_mask,
    truncation_meters=DEFAULT_BACKGROUND_DISTANCE_TRUNCATION_METERS,
    boxes=None,
):
    """Return the paper's truncated signed-box cost and its winning pair."""
    if not isinstance(truncation_meters, (float, int)) or not math.isfinite(truncation_meters):
        raise ValueError("truncation_meters must be a finite positive scalar")
    if truncation_meters <= 0:
        raise ValueError("truncation_meters must be a finite positive scalar")
    if boxes is None:
        boxes = _masked_boxes(states, state_valid, length_meters, width_meters)
    timestep_count = states.shape[1]
    no_pair = (
        states.new_zeros(()),
        torch.tensor(-1, dtype=torch.int64, device=states.device),
        torch.tensor(-1, dtype=torch.int64, device=states.device),
        torch.tensor(-1, dtype=torch.int64, device=states.device),
        states.new_zeros(()),
        torch.tensor(False, dtype=torch.bool, device=states.device),
    )
    # Released ReGentS builds this term over the adversary trajectories alone, so both
    # endpoints of a pair are candidates. Admitting a pair with one untouched
    # background vehicle would penalize distances the method never shapes.
    candidate_indices = torch.where(candidate_adversary_mask)[0]
    local_pairs = torch.triu_indices(
        candidate_indices.numel(),
        candidate_indices.numel(),
        offset=1,
        device=states.device,
    )
    pair_indices = candidate_indices[local_pairs]
    pair_valid = state_valid[pair_indices[0]] & state_valid[pair_indices[1]]
    eligible_pair = torch.any(pair_valid, dim=-1)
    pair_indices = pair_indices[:, eligible_pair]
    pair_valid = pair_valid[eligible_pair]
    if pair_indices.shape[1] == 0:
        return no_pair

    chunk_minima = []
    chunk_raw_minima = []
    chunk_first_agent_indices = []
    chunk_second_agent_indices = []
    chunk_timestep_indices = []
    for chunk_start in range(0, pair_indices.shape[1], PAIRWISE_DISTANCE_CHUNK_SIZE):
        chunk_pairs = pair_indices[:, chunk_start : chunk_start + PAIRWISE_DISTANCE_CHUNK_SIZE]
        first_indices, second_indices = chunk_pairs
        chunk_valid = pair_valid[chunk_start : chunk_start + chunk_pairs.shape[1]]
        first_boxes = boxes[first_indices]
        second_boxes = boxes[second_indices]
        truncated_distances = _truncated_signed_box_distances(
            first_boxes,
            second_boxes,
            chunk_valid,
            float(truncation_meters),
        )
        chunk_minima.append(truncated_distances.min())
        flat_winner_idx = torch.argmin(truncated_distances.reshape(-1))
        pair_winner_idx = torch.div(flat_winner_idx, timestep_count, rounding_mode="floor")
        timestep_idx = flat_winner_idx % timestep_count
        chunk_raw_minima.append(
            signed_box_distance(
                first_boxes[pair_winner_idx, timestep_idx],
                second_boxes[pair_winner_idx, timestep_idx],
            ).detach()
        )
        chunk_first_agent_indices.append(first_indices[pair_winner_idx])
        chunk_second_agent_indices.append(second_indices[pair_winner_idx])
        chunk_timestep_indices.append(timestep_idx)
    winning_chunk_idx = int(torch.argmin(torch.stack(chunk_raw_minima)).item())
    # The chunk already evaluated the exact clearance at its winning entry, and the
    # winner indexes the same two boxes, so that value is the reported distance.
    winning_distance = chunk_raw_minima[winning_chunk_idx]
    return (
        -torch.stack(chunk_minima).min(),
        chunk_first_agent_indices[winning_chunk_idx],
        chunk_second_agent_indices[winning_chunk_idx],
        chunk_timestep_indices[winning_chunk_idx],
        winning_distance,
        winning_distance >= float(truncation_meters),
    )


def drivable_area_deviation_cost(
    states,
    state_valid,
    length_meters,
    width_meters,
    candidate_adversary_mask,
    out_of_bounds_raster,
    boxes=None,
):
    """Mean over time of summed valid-vehicle corner potential, as in the paper."""
    if boxes is None:
        boxes = _masked_boxes(states, state_valid, length_meters, width_meters)
    if not isinstance(out_of_bounds_raster, SmoothedOutOfBoundsRaster):
        raise TypeError("The out-of-bounds raster must be precomputed and smoothed")
    potential = out_of_bounds_raster.potential
    if potential.device != states.device or potential.dtype != states.dtype:
        raise ValueError("The out-of-bounds raster and states must share device and dtype")
    corners = oriented_box_corners(boxes)
    corner_potential = sample_out_of_bounds_potential(corners, out_of_bounds_raster)
    valid = state_valid & candidate_adversary_mask[:, None]
    potential_sum = torch.where(
        valid[..., None],
        corner_potential,
        torch.zeros_like(corner_potential),
    ).sum(dim=(-1, -2))
    return potential_sum.sum() / states.shape[1]


def combined_regents_cost(
    states,
    state_valid,
    length_meters,
    width_meters,
    ego_mask,
    candidate_adversary_mask,
    out_of_bounds_raster,
    config=None,
):
    """Combine paper box-distance collision costs and the grid-approximated road cost."""
    if config is None:
        config = ReGentSCostConfig()
    if not isinstance(config, ReGentSCostConfig):
        raise TypeError("config must be a ReGentSCostConfig")
    if not isinstance(out_of_bounds_raster, SmoothedOutOfBoundsRaster):
        raise TypeError("The out-of-bounds raster must be precomputed and smoothed")
    if not math.isclose(out_of_bounds_raster.gaussian_sigma_meters, config.gaussian_sigma_meters):
        raise ValueError("Out-of-bounds raster Gaussian sigma does not match the cost config")
    if not math.isclose(out_of_bounds_raster.gaussian_truncate_sigma, config.gaussian_truncate_sigma):
        raise ValueError("Out-of-bounds raster Gaussian truncation does not match the cost config")
    if int(ego_mask.sum()) != 1:
        raise ValueError("The scenario must contain exactly one ego agent")
    if torch.any(ego_mask & candidate_adversary_mask):
        raise ValueError("The ego agent cannot be a candidate adversary")
    # One box tensor serves all three terms; each would otherwise rebuild it.
    boxes = _masked_boxes(states, state_valid, length_meters, width_meters)
    ego_collision = ego_background_collision_cost(
        states,
        state_valid,
        length_meters,
        width_meters,
        ego_mask,
        candidate_adversary_mask,
        boxes,
    )
    (
        background_collision,
        background_collision_first_agent_idx,
        background_collision_second_agent_idx,
        background_collision_timestep_idx,
        background_collision_signed_distance_meters,
        background_collision_truncated,
    ) = _background_collision_avoidance_cost_and_diagnostics(
        states,
        state_valid,
        length_meters,
        width_meters,
        candidate_adversary_mask,
        config.background_distance_truncation_meters,
        boxes,
    )
    drivable_area = drivable_area_deviation_cost(
        states,
        state_valid,
        length_meters,
        width_meters,
        candidate_adversary_mask,
        out_of_bounds_raster,
        boxes,
    )
    total = (
        config.ego_collision_weight * ego_collision
        + config.background_collision_weight * background_collision
        + config.drivable_area_weight * drivable_area
    )
    return ReGentSCosts(
        ego_collision,
        background_collision,
        drivable_area,
        total,
        background_collision_first_agent_idx,
        background_collision_second_agent_idx,
        background_collision_timestep_idx,
        background_collision_signed_distance_meters,
        background_collision_truncated,
    )
