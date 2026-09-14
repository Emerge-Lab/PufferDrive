"""Deterministic ReGentS scene and adversary selection."""

import math
from dataclasses import dataclass
from enum import IntFlag

import torch

from pufferlib.ocean.regents.geometry import oriented_box_corners, signed_box_distance
from pufferlib.ocean.regents.losses import _masked_boxes
from pufferlib.ocean.regents.state import (
    STATE_FEATURE_COUNT,
    STATE_HEADING,
    STATE_SPEED,
    STATE_X,
    STATE_Y,
    wrapped_angle_difference,
)


DEFAULT_MINIMUM_VALID_STATE_FRACTION = 0.5
DEFAULT_STATIC_DISPLACEMENT_THRESHOLD_METERS = 0.2
DEFAULT_STATIC_SPEED_THRESHOLD_MPS = 0.2
DEFAULT_REAR_SECTOR_FRACTION = 0.8
DEFAULT_REAR_SECTOR_HALF_ANGLE_RADIANS = math.pi / 8.0
# Disabled by default: a finite threshold obliges the caller to measure the drift
# of the same baseline reconstruction the optimizer will start from.
DEFAULT_MAXIMUM_RECONSTRUCTION_DRIFT_METERS = math.inf
DEFAULT_FILTER_OFF_ROAD_START = True
DEFAULT_FRONT_DIVERGENCE_FRACTION = 0.5
PAPER_FRONT_APPLICABILITY_HALF_ANGLE_RADIANS = math.pi / 8.0
REFERENCE_FRONT_YAW_HALF_ANGLE_RADIANS = math.pi / 2.0


class CandidateFilterReason(IntFlag):
    NONE = 0
    EGO = 1 << 0
    NON_VEHICLE = 1 << 1
    INSUFFICIENT_VALID_STATES = 1 << 2
    STATIC = 1 << 3
    REAR_SECTOR = 1 << 4
    SCENE_UNSUITABLE = 1 << 5
    RECONSTRUCTION_FIDELITY = 1 << 7
    OFF_ROAD_START = 1 << 8


class SceneFilterReason(IntFlag):
    NONE = 0
    INVALID_EGO = 1 << 0
    NO_CANDIDATE = 1 << 3


@dataclass(frozen=True)
class ReGentSFilterConfig:
    minimum_valid_state_fraction: float = DEFAULT_MINIMUM_VALID_STATE_FRACTION
    static_displacement_threshold_meters: float = DEFAULT_STATIC_DISPLACEMENT_THRESHOLD_METERS
    static_speed_threshold_mps: float = DEFAULT_STATIC_SPEED_THRESHOLD_MPS
    rear_sector_fraction: float = DEFAULT_REAR_SECTOR_FRACTION
    rear_sector_half_angle_radians: float = DEFAULT_REAR_SECTOR_HALF_ANGLE_RADIANS
    maximum_reconstruction_drift_meters: float = DEFAULT_MAXIMUM_RECONSTRUCTION_DRIFT_METERS
    filter_off_road_start: bool = DEFAULT_FILTER_OFF_ROAD_START

    def __post_init__(self):
        if not isinstance(self.filter_off_road_start, bool):
            raise TypeError("filter_off_road_start must be a bool")
        for name in ("minimum_valid_state_fraction", "rear_sector_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if math.isnan(self.maximum_reconstruction_drift_meters) or self.maximum_reconstruction_drift_meters < 0.0:
            raise ValueError("maximum_reconstruction_drift_meters must be non-negative")
        for name in ("static_displacement_threshold_meters", "static_speed_threshold_mps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.rear_sector_half_angle_radians):
            raise ValueError("rear_sector_half_angle_radians must be finite")
        if not 0.0 < self.rear_sector_half_angle_radians < math.pi:
            raise ValueError("rear_sector_half_angle_radians must be in (0, pi)")


def _reason_names(reason_flags, reason_bits):
    """Name every raised flag; NONE is the absence of a reason, never one of them."""
    raised = int(reason_bits.item())
    return tuple(reason.name.lower() for reason in reason_flags if reason and raised & int(reason))


@dataclass(frozen=True)
class CandidateSelection:
    candidate_mask: torch.Tensor
    optimized_action_mask: torch.Tensor
    filter_reason_bits: torch.Tensor
    scene_eligible: torch.Tensor
    scene_reason_bits: torch.Tensor
    original_collision: torch.Tensor
    original_collision_timestep: torch.Tensor
    valid_transition_count: torch.Tensor
    valid_state_fraction: torch.Tensor
    displacement_meters: torch.Tensor
    rear_sector_fraction: torch.Tensor
    maximum_reconstruction_residual_meters: torch.Tensor
    model_consistent_transition_fraction: torch.Tensor
    reconstruction_drift_meters: torch.Tensor
    start_off_road: torch.Tensor

    def reasons_for(self, agent_idx):
        return _reason_names(CandidateFilterReason, self.filter_reason_bits[agent_idx])

    def scene_reasons(self):
        return _reason_names(SceneFilterReason, self.scene_reason_bits)


def _resolve_selection_horizon(
    scenario, config, horizon_transition_count, inverse_dynamics, reconstruction_drift_meters
):
    """Derive the selection horizon and check the drift gate has the input it needs."""
    maximum_transition_count = scenario.max_time_count - 1
    if horizon_transition_count is None:
        horizon_transition_count = maximum_transition_count
    if horizon_transition_count < 1 or horizon_transition_count > maximum_transition_count:
        raise ValueError(f"horizon_transition_count must be in [1, {maximum_transition_count}]")
    if inverse_dynamics is not None and inverse_dynamics.action_valid.shape[-1] < horizon_transition_count:
        raise ValueError("inverse_dynamics does not cover the candidate-selection horizon")
    if reconstruction_drift_meters is None and math.isfinite(config.maximum_reconstruction_drift_meters):
        raise ValueError("reconstruction_drift_meters is required when the drift gate is enabled")
    return horizon_transition_count


def _reconstruction_statistics(scenario, inverse_dynamics, horizon_transition_count):
    """Per-agent sequential reconstruction diagnostics; reported, never gated.

    The strict composite residual and consistency flag are useful for diagnosis,
    while the separately measured maximum position drift controls selection.
    """
    if inverse_dynamics is None:
        zeros = torch.zeros(
            scenario.ego_mask.shape, dtype=scenario.logged_state.dtype, device=scenario.logged_state.device
        )
        return zeros, zeros

    action_valid = inverse_dynamics.action_valid[..., :horizon_transition_count]
    consistent = inverse_dynamics.model_consistent[..., :horizon_transition_count] & action_valid
    valid_transition_count = action_valid.sum(dim=-1)
    consistent_fraction = consistent.sum(dim=-1).to(scenario.logged_state.dtype)
    consistent_fraction /= valid_transition_count.clamp_min(1)
    residual = inverse_dynamics.residual_meters[..., :horizon_transition_count]
    maximum_residual = residual.masked_fill(~action_valid, -torch.inf).max(dim=-1).values
    maximum_residual = torch.where(valid_transition_count > 0, maximum_residual, torch.zeros_like(maximum_residual))
    return maximum_residual, consistent_fraction


def _motion_statistics(scenario):
    """Displacement between an agent's first and last valid states, and its peak logged speed.

    Endpoints must be valid states: storage is zero filled outside an agent's logged
    frames, so differencing raw endpoints measures distance to the map origin for every
    agent that enters or leaves mid scene.
    """
    state = scenario.logged_state
    valid = scenario.state_valid
    has_valid_state = valid.any(dim=-1)
    time_count = state.shape[1]
    position = state[..., STATE_X : STATE_Y + 1]
    valid_steps = valid.to(torch.int8)
    first_valid_idx = valid_steps.argmax(dim=-1)
    last_valid_idx = time_count - 1 - valid_steps.flip(-1).argmax(dim=-1)
    gather_shape = (*first_valid_idx.shape, 1, 2)
    first_position = torch.gather(position, 1, first_valid_idx[..., None, None].expand(gather_shape)).squeeze(1)
    last_position = torch.gather(position, 1, last_valid_idx[..., None, None].expand(gather_shape)).squeeze(1)
    displacement = torch.linalg.vector_norm(last_position - first_position, dim=-1)
    masked_speed = state[..., STATE_SPEED].abs().masked_fill(~valid, -torch.inf)
    maximum_speed = masked_speed.max(dim=-1).values
    maximum_speed = torch.where(has_valid_state, maximum_speed, torch.zeros_like(maximum_speed))
    return displacement, maximum_speed


def _start_off_road_flags(scenario):
    """Flag agents whose whole footprint lies off the drivable raster at their logged start.

    An agent that starts fully off the drivable surface has no lane the optimizer could
    keep it on, so accelerating it into the ego would only manufacture an off-road
    adversary. The footprint, rather than the center alone, decides: lane corridors are
    rasterized to a nominal width, so a vehicle straddling a corridor edge is still on
    road. Pixels are sampled at the nearest center, as the off-road cost signature does.
    """
    state = scenario.logged_state
    valid = scenario.state_valid
    has_valid_state = valid.any(dim=-1)
    first_valid_idx = valid.to(torch.int8).argmax(dim=-1)
    gather_shape = (*first_valid_idx.shape, 1, STATE_FEATURE_COUNT)
    start_state = torch.gather(state, 1, first_valid_idx[..., None, None].expand(gather_shape))
    start_boxes = _masked_boxes(
        start_state, has_valid_state[..., None], scenario.length_meters, scenario.width_meters
    ).squeeze(1)
    sample_points = torch.cat((oriented_box_corners(start_boxes), start_boxes[..., None, :2]), dim=-2)
    off_road = scenario.drivable_area_raster.points_off_road(sample_points)
    # An agent with no valid state has no logged start to place on the map.
    return off_road.all(dim=-1) & has_valid_state


def _rear_sector_statistics(scenario, logged_time_count, half_angle_radians):
    state = scenario.logged_state
    ego_idx = int(torch.where(scenario.ego_mask)[0].item())
    displacement = state[:, :, STATE_X : STATE_Y + 1] - state[ego_idx, None, :, STATE_X : STATE_Y + 1]
    position_angle = torch.atan2(displacement[..., 1], displacement[..., 0])
    relative_bearing = wrapped_angle_difference(position_angle, state[ego_idx, None, :, STATE_HEADING])
    rear = relative_bearing.abs() > math.pi - half_angle_radians
    logged_time = torch.arange(state.shape[1], device=state.device) < logged_time_count
    return (rear & logged_time).sum(dim=-1).to(state.dtype) / logged_time_count


def _ego_overlap_timesteps(scenario, state, valid):
    """Return the first ego-overlap timestep per agent, mirroring the paper's overlap_with_ego.

    The ego, absent agents, and timesteps either party does not occupy never overlap.
    """
    timestep_count = state.shape[1]
    ego_idx = int(torch.where(scenario.ego_mask)[0].item())
    comparable = valid & valid[ego_idx, None]
    comparable &= (scenario.agent_present & scenario.agent_metadata_valid & ~scenario.ego_mask)[:, None]
    boxes = _masked_boxes(state, valid, scenario.length_meters, scenario.width_meters)
    overlapping = comparable & (signed_box_distance(boxes, boxes[ego_idx, None]) <= 0.0)
    timestep_idx = torch.arange(timestep_count, device=state.device).expand_as(overlapping)
    first_timestep = timestep_idx.masked_fill(~overlapping, timestep_count).min(dim=-1).values
    return first_timestep.masked_fill(first_timestep == timestep_count, -1)


def _original_collision_labels(scenario, horizon_transition_count):
    """Label per agent whether its logged trajectory already overlaps the ego."""
    state = scenario.logged_state[:, : horizon_transition_count + 1]
    valid = scenario.state_valid[:, : horizon_transition_count + 1]
    collision_timestep = _ego_overlap_timesteps(scenario, state, valid)
    return collision_timestep >= 0, collision_timestep


def select_adversary_candidates(
    scenario,
    config=None,
    *,
    horizon_transition_count=None,
    inverse_dynamics=None,
    reconstruction_drift_meters=None,
):
    """Filter candidates using logged trajectories and record every reason.

    Full-log trajectory filters are independent of rollout horizon. In addition to the
    released displacement test, peak logged speed rejects parked tracks whose position
    jitter exceeds the displacement threshold. The optional drift gate rejects an
    adversary whose baseline reconstruction leaves its own log, because a scene the
    optimizer cannot reproduce is not the scene the collision would be introduced into.
    """
    if config is None:
        config = ReGentSFilterConfig()
    horizon_transition_count = _resolve_selection_horizon(
        scenario, config, horizon_transition_count, inverse_dynamics, reconstruction_drift_meters
    )
    transition_valid = scenario.transition_valid[:, :horizon_transition_count]
    valid_transition_count = transition_valid.sum(dim=-1)
    logged_time_count = int(scenario.trajectory_length.max().item())
    if logged_time_count <= 0 or logged_time_count > scenario.max_time_count:
        raise ValueError("Candidate filtering requires a non-empty exported log within state storage")
    logged_time = torch.arange(scenario.max_time_count, device=scenario.device) < logged_time_count
    valid_state_fraction = (scenario.state_valid & logged_time).sum(dim=-1).to(scenario.logged_state.dtype)
    valid_state_fraction = valid_state_fraction / logged_time_count
    displacement, maximum_speed = _motion_statistics(scenario)
    rear_fraction = _rear_sector_statistics(scenario, logged_time_count, config.rear_sector_half_angle_radians)
    maximum_reconstruction_residual, model_consistent_fraction = _reconstruction_statistics(
        scenario, inverse_dynamics, horizon_transition_count
    )
    if reconstruction_drift_meters is None:
        reconstruction_drift_meters = torch.zeros_like(maximum_reconstruction_residual)
    original_collision, original_collision_timestep = _original_collision_labels(scenario, horizon_transition_count)

    start_off_road = _start_off_road_flags(scenario)
    static = displacement < config.static_displacement_threshold_meters
    static |= maximum_speed < config.static_speed_threshold_mps
    rejected_by = (
        (CandidateFilterReason.EGO, scenario.ego_mask),
        (CandidateFilterReason.NON_VEHICLE, ~scenario.vehicle_mask),
        (CandidateFilterReason.INSUFFICIENT_VALID_STATES, valid_state_fraction < config.minimum_valid_state_fraction),
        (CandidateFilterReason.STATIC, static),
        (CandidateFilterReason.REAR_SECTOR, rear_fraction > config.rear_sector_fraction),
        (
            CandidateFilterReason.RECONSTRUCTION_FIDELITY,
            reconstruction_drift_meters > config.maximum_reconstruction_drift_meters,
        ),
        (CandidateFilterReason.OFF_ROAD_START, start_off_road & config.filter_off_road_start),
    )
    reason_bits = torch.zeros(scenario.ego_mask.shape, dtype=torch.int64, device=scenario.logged_state.device)
    for reason, rejected in rejected_by:
        reason_bits |= rejected.to(torch.int64) * int(reason)

    invalid_ego = (int(scenario.ego_mask.sum().item()) != 1) or (
        int((transition_valid & scenario.ego_mask[..., None]).sum().item()) < 1
    )
    scene_reason_bits = torch.tensor(
        int(invalid_ego) * int(SceneFilterReason.INVALID_EGO), dtype=torch.int64, device=scenario.device
    )
    reason_bits |= int(invalid_ego) * int(CandidateFilterReason.SCENE_UNSUITABLE)

    candidate_mask = scenario.candidate_adversary_mask & (reason_bits == 0)
    if not bool(candidate_mask.any()):
        scene_reason_bits |= int(SceneFilterReason.NO_CANDIDATE)
    scene_eligible = scene_reason_bits == 0
    candidate_mask &= scene_eligible
    optimized_action_mask = candidate_mask[..., None] & transition_valid
    return CandidateSelection(
        candidate_mask=candidate_mask,
        optimized_action_mask=optimized_action_mask,
        filter_reason_bits=reason_bits,
        scene_eligible=scene_eligible,
        scene_reason_bits=scene_reason_bits,
        original_collision=original_collision,
        original_collision_timestep=original_collision_timestep,
        valid_transition_count=valid_transition_count,
        valid_state_fraction=valid_state_fraction,
        displacement_meters=displacement,
        rear_sector_fraction=rear_fraction,
        maximum_reconstruction_residual_meters=maximum_reconstruction_residual,
        model_consistent_transition_fraction=model_consistent_fraction,
        reconstruction_drift_meters=reconstruction_drift_meters,
        start_off_road=start_off_road,
    )


def front_divergence_mask(
    states,
    state_valid,
    ego_mask,
    candidate_mask,
    *,
    tau_front=DEFAULT_FRONT_DIVERGENCE_FRACTION,
    applicability_half_angle_radians=PAPER_FRONT_APPLICABILITY_HALF_ANGLE_RADIANS,
    yaw_half_angle_radians=REFERENCE_FRONT_YAW_HALF_ANGLE_RADIANS,
):
    """Return candidates whose current rollout occupies the paper's red zone.

    Bearing and yaw use separate windows, as in the reference implementation.
    """
    if not math.isfinite(tau_front) or tau_front < 0.0 or tau_front > 1.0:
        raise ValueError("tau_front must be finite and in [0, 1]")
    for name, value in (
        ("applicability_half_angle_radians", applicability_half_angle_radians),
        ("yaw_half_angle_radians", yaw_half_angle_radians),
    ):
        if not math.isfinite(value) or not 0.0 < value <= math.pi:
            raise ValueError(f"{name} must be finite and in (0, pi]")
    ego_positions = torch.where(ego_mask)[0]
    if ego_positions.numel() != 1:
        raise ValueError("front divergence requires exactly one ego per scenario")
    ego_idx = int(ego_positions.item())
    displacement = states[:, :, STATE_X : STATE_Y + 1] - states[ego_idx, None, :, STATE_X : STATE_Y + 1]
    position_angle = torch.atan2(displacement[..., 1], displacement[..., 0])
    ego_heading = states[ego_idx, None, :, STATE_HEADING]
    relative_bearing = wrapped_angle_difference(position_angle, ego_heading)
    relative_yaw = wrapped_angle_difference(states[:, :, STATE_HEADING], ego_heading)
    applicable = state_valid & state_valid[ego_idx, None, :]
    in_bounds = relative_bearing.abs() < applicability_half_angle_radians
    in_bounds &= relative_yaw.abs() < yaw_half_angle_radians
    same_side_inside_yaw = relative_bearing * relative_yaw > 0.0
    same_side_inside_yaw &= relative_bearing.abs() < relative_yaw.abs()
    red_zone = applicable & in_bounds & same_side_inside_yaw
    applicable_count = applicable.sum(dim=-1)
    fraction = red_zone.sum(dim=-1).to(states.dtype) / applicable_count.clamp_min(1).to(states.dtype)
    return candidate_mask & (applicable_count > 0) & (fraction > tau_front)
