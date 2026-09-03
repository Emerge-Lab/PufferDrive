"""Deterministic ReGentS scene and adversary selection."""

import math
from dataclasses import dataclass
from enum import IntFlag

import torch

from pufferlib.ocean.regents.geometry import signed_box_distance
from pufferlib.ocean.regents.state import STATE_HEADING, STATE_SPEED, STATE_X, STATE_Y, ScenarioBatch


DEFAULT_MINIMUM_VALID_TRANSITION_FRACTION = 0.5
DEFAULT_MINIMUM_VALID_TRANSITION_COUNT = 1
DEFAULT_STATIC_DISPLACEMENT_THRESHOLD_METERS = 0.2
DEFAULT_STATIC_SPEED_THRESHOLD_MPS = 0.2
DEFAULT_REAR_SECTOR_FRACTION = 0.8
DEFAULT_REAR_SECTOR_HALF_ANGLE_RADIANS = math.pi / 8.0
DEFAULT_FRONT_DIVERGENCE_FRACTION = 0.5
PAPER_FRONT_APPLICABILITY_HALF_ANGLE_RADIANS = math.pi / 8.0


class CandidateFilterReason(IntFlag):
    NONE = 0
    EGO = 1 << 0
    NON_VEHICLE = 1 << 1
    INSUFFICIENT_VALID_TRANSITIONS = 1 << 2
    STATIC = 1 << 3
    REAR_SECTOR = 1 << 4
    SCENE_UNSUITABLE = 1 << 5
    ORIGINAL_COLLISION = 1 << 6


class SceneFilterReason(IntFlag):
    NONE = 0
    INVALID_EGO = 1 << 0
    CALLER_UNSUITABLE = 1 << 1
    ORIGINAL_COLLISION = 1 << 2
    NO_CANDIDATE = 1 << 3


@dataclass(frozen=True)
class ReGentSFilterConfig:
    minimum_valid_transition_fraction: float = DEFAULT_MINIMUM_VALID_TRANSITION_FRACTION
    minimum_valid_transition_count: int = DEFAULT_MINIMUM_VALID_TRANSITION_COUNT
    static_displacement_threshold_meters: float = DEFAULT_STATIC_DISPLACEMENT_THRESHOLD_METERS
    static_speed_threshold_mps: float = DEFAULT_STATIC_SPEED_THRESHOLD_MPS
    rear_sector_fraction: float = DEFAULT_REAR_SECTOR_FRACTION
    rear_sector_half_angle_radians: float = DEFAULT_REAR_SECTOR_HALF_ANGLE_RADIANS

    def __post_init__(self):
        for name in ("minimum_valid_transition_fraction", "rear_sector_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not isinstance(self.minimum_valid_transition_count, int):
            raise TypeError("minimum_valid_transition_count must be an integer")
        if self.minimum_valid_transition_count < 1:
            raise ValueError("minimum_valid_transition_count must be positive")
        for name in ("static_displacement_threshold_meters", "static_speed_threshold_mps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.rear_sector_half_angle_radians):
            raise ValueError("rear_sector_half_angle_radians must be finite")
        if not 0.0 < self.rear_sector_half_angle_radians < math.pi:
            raise ValueError("rear_sector_half_angle_radians must be in (0, pi)")


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
    valid_transition_fraction: torch.Tensor
    displacement_meters: torch.Tensor
    maximum_absolute_speed_mps: torch.Tensor
    rear_sector_fraction: torch.Tensor
    horizon_transition_count: int

    def __post_init__(self):
        if self.candidate_mask.dtype != torch.bool or self.candidate_mask.ndim != 2:
            raise ValueError("candidate_mask must be bool [batch, agent]")
        batch_count, agent_count = self.candidate_mask.shape
        if self.optimized_action_mask.dtype != torch.bool:
            raise TypeError("optimized_action_mask must use torch.bool")
        if self.optimized_action_mask.shape != (batch_count, agent_count, self.horizon_transition_count):
            raise ValueError("optimized_action_mask has an incompatible shape")
        for name in (
            "filter_reason_bits",
            "valid_transition_count",
            "valid_transition_fraction",
            "displacement_meters",
            "maximum_absolute_speed_mps",
            "rear_sector_fraction",
        ):
            if getattr(self, name).shape != self.candidate_mask.shape:
                raise ValueError(f"{name} must have shape [batch, agent]")
        for name in ("scene_eligible", "scene_reason_bits"):
            if getattr(self, name).shape != (batch_count,):
                raise ValueError(f"{name} must have shape [batch]")
        for name in ("original_collision", "original_collision_timestep"):
            if getattr(self, name).shape != self.candidate_mask.shape:
                raise ValueError(f"{name} must have shape [batch, agent]")

    def reasons_for(self, batch_idx, agent_idx):
        reason_bits = int(self.filter_reason_bits[batch_idx, agent_idx].item())
        return tuple(
            reason.name.lower()
            for reason in CandidateFilterReason
            if reason is not CandidateFilterReason.NONE and reason_bits & int(reason)
        )

    def scene_reasons_for(self, batch_idx):
        reason_bits = int(self.scene_reason_bits[batch_idx].item())
        return tuple(
            reason.name.lower()
            for reason in SceneFilterReason
            if reason is not SceneFilterReason.NONE and reason_bits & int(reason)
        )


def wrapped_angle_difference(first, second):
    """Return ``first - second`` wrapped to ``[-pi, pi]``."""
    difference = first - second
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def _validate_selection_inputs(scenario, config, horizon_transition_count, scene_suitable):
    if not isinstance(scenario, ScenarioBatch):
        raise TypeError("scenario must be a ScenarioBatch")
    if not isinstance(config, ReGentSFilterConfig):
        raise TypeError("config must be a ReGentSFilterConfig")
    maximum_transition_count = scenario.max_time_count - 1
    if horizon_transition_count is None:
        horizon_transition_count = maximum_transition_count
    if not isinstance(horizon_transition_count, int):
        raise TypeError("horizon_transition_count must be an integer")
    if horizon_transition_count < 1 or horizon_transition_count > maximum_transition_count:
        raise ValueError(f"horizon_transition_count must be in [1, {maximum_transition_count}]")
    if scene_suitable is None:
        scene_suitable = torch.ones(scenario.batch_size, dtype=torch.bool, device=scenario.logged_state.device)
    if not isinstance(scene_suitable, torch.Tensor) or scene_suitable.dtype != torch.bool:
        raise TypeError("scene_suitable must be a bool Torch tensor")
    if scene_suitable.shape != (scenario.batch_size,) or scene_suitable.device != scenario.logged_state.device:
        raise ValueError("scene_suitable must have shape [batch] on the scenario device")
    return horizon_transition_count, scene_suitable


def _motion_statistics(scenario, horizon_transition_count):
    state = scenario.logged_state[:, :, : horizon_transition_count + 1]
    valid = scenario.state_valid[:, :, : horizon_transition_count + 1]
    time_count = state.shape[2]
    has_valid_state = valid.any(dim=-1)
    first_idx = torch.argmax(valid.to(torch.int64), dim=-1)
    last_idx = time_count - 1 - torch.argmax(valid.flip(dims=(-1,)).to(torch.int64), dim=-1)
    position = state[..., STATE_X : STATE_Y + 1]
    gather_shape = (*first_idx.shape, 1, 2)
    first_position = torch.gather(position, 2, first_idx[..., None, None].expand(gather_shape)).squeeze(2)
    last_position = torch.gather(position, 2, last_idx[..., None, None].expand(gather_shape)).squeeze(2)
    displacement = torch.linalg.vector_norm(last_position - first_position, dim=-1)
    displacement = torch.where(has_valid_state, displacement, torch.zeros_like(displacement))
    masked_speed = state[..., STATE_SPEED].abs().masked_fill(~valid, -torch.inf)
    maximum_speed = masked_speed.max(dim=-1).values
    maximum_speed = torch.where(has_valid_state, maximum_speed, torch.zeros_like(maximum_speed))
    return displacement, maximum_speed


def _rear_sector_statistics(scenario, horizon_transition_count, half_angle_radians):
    state = scenario.logged_state[:, :, : horizon_transition_count + 1]
    valid = scenario.state_valid[:, :, : horizon_transition_count + 1]
    output = torch.zeros(state.shape[:2], dtype=state.dtype, device=state.device)
    for batch_idx in range(scenario.batch_size):
        ego_idx = torch.where(scenario.ego_mask[batch_idx])[0]
        if ego_idx.numel() != 1:
            continue
        ego_idx = int(ego_idx.item())
        displacement = (
            state[batch_idx, :, :, STATE_X : STATE_Y + 1] - state[batch_idx, ego_idx, None, :, STATE_X : STATE_Y + 1]
        )
        position_angle = torch.atan2(displacement[..., 1], displacement[..., 0])
        relative_bearing = wrapped_angle_difference(
            position_angle,
            state[batch_idx, ego_idx, None, :, STATE_HEADING],
        )
        jointly_valid = valid[batch_idx] & valid[batch_idx, ego_idx, None, :]
        rear = relative_bearing.abs() > math.pi - half_angle_radians
        rear_count = (rear & jointly_valid).sum(dim=-1)
        applicable_count = jointly_valid.sum(dim=-1)
        output[batch_idx] = torch.where(
            applicable_count > 0,
            rear_count.to(state.dtype) / applicable_count.clamp_min(1).to(state.dtype),
            torch.zeros_like(output[batch_idx]),
        )
    return output


def _ego_overlap_timesteps(scenario, state, valid, batch_idx):
    """Return the first ego-overlap timestep per agent, mirroring the paper's overlap_with_ego."""
    agent_count = state.shape[1]
    first_timestep = torch.full((agent_count,), -1, dtype=torch.int64, device=state.device)
    ego_positions = torch.where(scenario.ego_mask[batch_idx])[0]
    if ego_positions.numel() != 1:
        return first_timestep
    ego_idx = int(ego_positions.item())
    present = scenario.agent_present[batch_idx] & scenario.agent_metadata_valid[batch_idx]
    for agent_idx in range(agent_count):
        if agent_idx == ego_idx or not bool(present[agent_idx]):
            continue
        jointly_valid = valid[batch_idx, ego_idx] & valid[batch_idx, agent_idx]
        if not jointly_valid.any():
            continue
        timestep_idx = torch.where(jointly_valid)[0]
        boxes = []
        for index in (ego_idx, agent_idx):
            agent_state = state[batch_idx, index, timestep_idx]
            boxes.append(
                torch.stack(
                    (
                        agent_state[:, STATE_X],
                        agent_state[:, STATE_Y],
                        torch.full_like(agent_state[:, STATE_X], scenario.length_meters[batch_idx, index]),
                        torch.full_like(agent_state[:, STATE_X], scenario.width_meters[batch_idx, index]),
                        agent_state[:, STATE_HEADING],
                    ),
                    dim=-1,
                )
            )
        overlap_idx = torch.where(signed_box_distance(boxes[0], boxes[1]) <= 0.0)[0]
        if overlap_idx.numel():
            first_timestep[agent_idx] = int(timestep_idx[overlap_idx[0]].item())
    return first_timestep


def _original_collision_labels(scenario, horizon_transition_count):
    """Label per agent whether its logged trajectory already overlaps the ego."""
    state = scenario.logged_state[:, :, : horizon_transition_count + 1]
    valid = scenario.state_valid[:, :, : horizon_transition_count + 1]
    agent_shape = (scenario.batch_size, state.shape[1])
    collision_timestep = torch.full(agent_shape, -1, dtype=torch.int64, device=state.device)
    for batch_idx in range(scenario.batch_size):
        collision_timestep[batch_idx] = _ego_overlap_timesteps(scenario, state, valid, batch_idx)
    return collision_timestep >= 0, collision_timestep


def select_adversary_candidates(
    scenario,
    config=None,
    *,
    horizon_transition_count=None,
    scene_suitable=None,
):
    """Filter candidates using logged trajectories and record every reason.

    A vehicle is static when either its first-to-last valid displacement or its
    maximum absolute logged speed is below the corresponding threshold. Rear
    occupancy uses jointly valid ego/agent states and strict angular/fraction
    boundaries, matching the ReGentS reference behavior.
    """
    if config is None:
        config = ReGentSFilterConfig()
    horizon_transition_count, scene_suitable = _validate_selection_inputs(
        scenario, config, horizon_transition_count, scene_suitable
    )
    transition_valid = scenario.transition_valid[:, :, :horizon_transition_count]
    valid_transition_count = transition_valid.sum(dim=-1)
    valid_transition_fraction = valid_transition_count.to(scenario.logged_state.dtype) / horizon_transition_count
    displacement, maximum_speed = _motion_statistics(scenario, horizon_transition_count)
    rear_fraction = _rear_sector_statistics(scenario, horizon_transition_count, config.rear_sector_half_angle_radians)
    original_collision, original_collision_timestep = _original_collision_labels(scenario, horizon_transition_count)

    agent_shape = scenario.ego_mask.shape
    reason_bits = torch.zeros(agent_shape, dtype=torch.int64, device=scenario.logged_state.device)
    reason_bits |= scenario.ego_mask.to(torch.int64) * int(CandidateFilterReason.EGO)
    reason_bits |= (~scenario.vehicle_mask).to(torch.int64) * int(CandidateFilterReason.NON_VEHICLE)
    insufficient = valid_transition_count < config.minimum_valid_transition_count
    insufficient |= valid_transition_fraction < config.minimum_valid_transition_fraction
    reason_bits |= insufficient.to(torch.int64) * int(CandidateFilterReason.INSUFFICIENT_VALID_TRANSITIONS)
    static = displacement < config.static_displacement_threshold_meters
    static |= maximum_speed < config.static_speed_threshold_mps
    reason_bits |= static.to(torch.int64) * int(CandidateFilterReason.STATIC)
    rear = rear_fraction > config.rear_sector_fraction
    reason_bits |= rear.to(torch.int64) * int(CandidateFilterReason.REAR_SECTOR)

    ego_count = scenario.ego_mask.sum(dim=-1)
    ego_transition_count = (transition_valid & scenario.ego_mask[..., None]).sum(dim=(-2, -1))
    invalid_ego = (ego_count != 1) | (ego_transition_count < config.minimum_valid_transition_count)
    scene_reason_bits = invalid_ego.to(torch.int64) * int(SceneFilterReason.INVALID_EGO)
    scene_reason_bits |= (~scene_suitable).to(torch.int64) * int(SceneFilterReason.CALLER_UNSUITABLE)
    all_candidates_collide = (original_collision | ~scenario.candidate_adversary_mask).all(dim=-1)
    scene_reason_bits |= all_candidates_collide.to(torch.int64) * int(SceneFilterReason.ORIGINAL_COLLISION)
    preliminarily_eligible = scene_reason_bits == 0
    reason_bits |= (~preliminarily_eligible[:, None]).to(torch.int64) * int(CandidateFilterReason.SCENE_UNSUITABLE)
    reason_bits |= original_collision.to(torch.int64) * int(CandidateFilterReason.ORIGINAL_COLLISION)

    candidate_mask = scenario.candidate_adversary_mask & (reason_bits == 0)
    no_candidate = ~candidate_mask.any(dim=-1)
    scene_reason_bits |= no_candidate.to(torch.int64) * int(SceneFilterReason.NO_CANDIDATE)
    scene_eligible = scene_reason_bits == 0
    candidate_mask &= scene_eligible[:, None]
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
        valid_transition_fraction=valid_transition_fraction,
        displacement_meters=displacement,
        maximum_absolute_speed_mps=maximum_speed,
        rear_sector_fraction=rear_fraction,
        horizon_transition_count=horizon_transition_count,
    )


def front_divergence_mask(
    states,
    state_valid,
    ego_mask,
    candidate_mask,
    *,
    tau_front=DEFAULT_FRONT_DIVERGENCE_FRACTION,
    applicability_half_angle_radians=PAPER_FRONT_APPLICABILITY_HALF_ANGLE_RADIANS,
):
    """Return candidates whose current rollout occupies the paper's red zone."""
    if not isinstance(states, torch.Tensor) or not states.is_floating_point():
        raise TypeError("states must be a floating Torch tensor")
    if states.ndim != 4 or states.shape[-1] < STATE_HEADING + 1:
        raise ValueError("states must have shape [batch, agent, time, feature>=3]")
    if state_valid.dtype != torch.bool or state_valid.shape != states.shape[:-1]:
        raise ValueError("state_valid must be bool [batch, agent, time]")
    if ego_mask.dtype != torch.bool or candidate_mask.dtype != torch.bool:
        raise TypeError("ego_mask and candidate_mask must use torch.bool")
    if ego_mask.shape != states.shape[:2] or candidate_mask.shape != states.shape[:2]:
        raise ValueError("ego_mask and candidate_mask must have shape [batch, agent]")
    if not math.isfinite(tau_front) or tau_front < 0.0 or tau_front > 1.0:
        raise ValueError("tau_front must be finite and in [0, 1]")
    if not math.isfinite(applicability_half_angle_radians):
        raise ValueError("applicability_half_angle_radians must be finite")
    if not 0.0 < applicability_half_angle_radians <= math.pi:
        raise ValueError("applicability_half_angle_radians must be in (0, pi]")
    if (
        states.device != state_valid.device
        or states.device != ego_mask.device
        or states.device != candidate_mask.device
    ):
        raise ValueError("all front-divergence inputs must share a device")

    output = torch.zeros(states.shape[:2], dtype=torch.bool, device=states.device)
    for batch_idx in range(states.shape[0]):
        ego_idx = torch.where(ego_mask[batch_idx])[0]
        if ego_idx.numel() != 1:
            raise ValueError("front divergence requires exactly one ego per scenario")
        ego_idx = int(ego_idx.item())
        displacement = (
            states[batch_idx, :, :, STATE_X : STATE_Y + 1] - states[batch_idx, ego_idx, None, :, STATE_X : STATE_Y + 1]
        )
        position_angle = torch.atan2(displacement[..., 1], displacement[..., 0])
        ego_heading = states[batch_idx, ego_idx, None, :, STATE_HEADING]
        relative_bearing = wrapped_angle_difference(position_angle, ego_heading)
        relative_yaw = wrapped_angle_difference(states[batch_idx, :, :, STATE_HEADING], ego_heading)
        jointly_valid = state_valid[batch_idx] & state_valid[batch_idx, ego_idx, None, :]
        applicable = jointly_valid
        in_bounds = relative_bearing.abs() < applicability_half_angle_radians
        in_bounds &= relative_yaw.abs() < applicability_half_angle_radians
        same_side_inside_yaw = relative_bearing * relative_yaw > 0.0
        same_side_inside_yaw &= relative_bearing.abs() < relative_yaw.abs()
        red_zone = applicable & in_bounds & same_side_inside_yaw
        applicable_count = applicable.sum(dim=-1)
        fraction = red_zone.sum(dim=-1).to(states.dtype) / applicable_count.clamp_min(1).to(states.dtype)
        output[batch_idx] = candidate_mask[batch_idx] & (applicable_count > 0) & (fraction > tau_front)
    return output
