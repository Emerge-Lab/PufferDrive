"""Frozen-ego ReGentS optimization for one offline scenario."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents.adapter import DEFAULT_RASTER_RESOLUTION_METERS, export_drive_scenarios
from pufferlib.ocean.regents.dynamics import (
    ACTION_ACCELERATION,
    ACTION_TARGET_STEERING,
    TARGET_STEERING_SCALE_RADIANS,
    classic_step,
)
from pufferlib.ocean.regents.filters import (
    DEFAULT_FRONT_DIVERGENCE_FRACTION,
    PAPER_FRONT_APPLICABILITY_HALF_ANGLE_RADIANS,
    REFERENCE_FRONT_YAW_HALF_ANGLE_RADIANS,
    CandidateSelection,
    ReGentSFilterConfig,
    front_divergence_mask,
    select_adversary_candidates,
)
from pufferlib.ocean.regents.geometry import (
    oriented_box_corners,
    sample_out_of_bounds_potential,
    signed_box_distance,
)
from pufferlib.ocean.regents.inverse_dynamics import InverseDynamicsResult, estimate_expert_actions
from pufferlib.ocean.regents.losses import (
    ReGentSCostConfig,
    _masked_boxes,
    combined_regents_cost,
    prepare_out_of_bounds_rasters,
)
from pufferlib.ocean.regents.state import (
    STATE_FEATURE_COUNT,
    ScenarioBatch,
    signed_speed_from_c_velocity,
)
from pufferlib.ocean.regents.waymax_actions import (
    NORMALIZED_CURVATURE_LIMIT,
    curvature_from_target_steering,
    target_steering_from_curvature,
)


DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_ITERATION_COUNT = 500
DEFAULT_ADAM_BETA1 = 0.9
DEFAULT_ADAM_BETA2 = 0.999
DEFAULT_ADAM_EPSILON = 1e-8
DEFAULT_ACTION_SATURATION_TOLERANCE = 1e-6
DEFAULT_COLLISION_DISTANCE_TOLERANCE_METERS = 0.0
DEFAULT_EARLY_STOP_MINIMUM_IMPROVEMENT = 0.0
DEFAULT_EARLY_STOP_PATIENCE_ITERATIONS = 0
DEFAULT_STEERING_UPDATE_SCALE = 4.0
MAXIMUM_STEERING_UPDATE_SCALE = 10.0
BACKGROUND_COLLISION_PAIR_CHUNK_SIZE = 4096

# Steering may be optimized as the simulator's normalized target wheel angle or as the
# reference's path curvature. The two carry different units, so each keeps its own
# update scale; `DEFAULT_CURVATURE_STEERING_UPDATE_SCALE` is the ReGentS value.
STEERING_PARAMETERIZATION_WHEEL_ANGLE = "wheel_angle"
STEERING_PARAMETERIZATION_CURVATURE = "curvature"
STEERING_PARAMETERIZATIONS = (STEERING_PARAMETERIZATION_WHEEL_ANGLE, STEERING_PARAMETERIZATION_CURVATURE)
DEFAULT_STEERING_PARAMETERIZATION = STEERING_PARAMETERIZATION_WHEEL_ANGLE
DEFAULT_CURVATURE_STEERING_UPDATE_SCALE = 0.5


@dataclass(frozen=True)
class FrozenEgoTrajectory:
    """Detached ego states captured from C IDM or a logged test fixture."""

    state: torch.Tensor
    valid: torch.Tensor
    scenario_ids: tuple[str, ...]
    source: str

    def __post_init__(self):
        if self.state.dtype != torch.float32:
            raise TypeError("Frozen ego state must use torch.float32")
        if self.state.ndim != 3 or self.state.shape[-1] != STATE_FEATURE_COUNT:
            raise ValueError("Frozen ego state must have shape [batch, time, 5]")
        if self.valid.dtype != torch.bool or self.valid.shape != self.state.shape[:-1]:
            raise ValueError("Frozen ego validity must be bool [batch, time]")
        if self.valid.device != self.state.device:
            raise ValueError("Frozen ego state and validity must share a device")
        if not isinstance(self.scenario_ids, tuple) or len(self.scenario_ids) != self.state.shape[0]:
            raise ValueError("Frozen ego scenario_ids must be one string per batch row")
        if not all(isinstance(item, str) and item for item in self.scenario_ids):
            raise ValueError("Frozen ego scenario_ids must be non-empty strings")
        if self.source not in ("c_idm", "logged_fixture", "c_replay"):
            raise ValueError("Frozen ego source must be 'c_idm', 'logged_fixture', or 'c_replay'")
        if not torch.isfinite(self.state[self.valid]).all():
            raise ValueError("Frozen valid ego states contain NaN or Inf")


@dataclass(frozen=True)
class ReGentSOptimizationConfig:
    filter: ReGentSFilterConfig = field(default_factory=ReGentSFilterConfig)
    costs: ReGentSCostConfig = field(default_factory=ReGentSCostConfig)
    learning_rate: float = DEFAULT_LEARNING_RATE
    iteration_count: int = DEFAULT_ITERATION_COUNT
    adam_beta1: float = DEFAULT_ADAM_BETA1
    adam_beta2: float = DEFAULT_ADAM_BETA2
    adam_epsilon: float = DEFAULT_ADAM_EPSILON
    tau_front: float = DEFAULT_FRONT_DIVERGENCE_FRACTION
    front_applicability_half_angle_radians: float = PAPER_FRONT_APPLICABILITY_HALF_ANGLE_RADIANS
    front_yaw_half_angle_radians: float = REFERENCE_FRONT_YAW_HALF_ANGLE_RADIANS
    steering_update_scale: float = DEFAULT_STEERING_UPDATE_SCALE
    steering_parameterization: str = DEFAULT_STEERING_PARAMETERIZATION
    curvature_steering_update_scale: float = DEFAULT_CURVATURE_STEERING_UPDATE_SCALE
    action_saturation_tolerance: float = DEFAULT_ACTION_SATURATION_TOLERANCE
    collision_distance_tolerance_meters: float = DEFAULT_COLLISION_DISTANCE_TOLERANCE_METERS
    early_stop_on_collision: bool = True
    early_stop_minimum_improvement: float = DEFAULT_EARLY_STOP_MINIMUM_IMPROVEMENT
    early_stop_patience_iterations: int = DEFAULT_EARLY_STOP_PATIENCE_ITERATIONS

    def __post_init__(self):
        if not isinstance(self.filter, ReGentSFilterConfig):
            raise TypeError("filter must be a ReGentSFilterConfig")
        if not isinstance(self.costs, ReGentSCostConfig):
            raise TypeError("costs must be a ReGentSCostConfig")
        for name in ("learning_rate", "adam_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.iteration_count, int) or self.iteration_count < 1:
            raise ValueError("iteration_count must be a positive integer")
        for name in ("adam_beta1", "adam_beta2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not math.isfinite(self.tau_front) or self.tau_front < 0.0 or self.tau_front > 1.0:
            raise ValueError("tau_front must be finite and in [0, 1]")
        for name in ("front_applicability_half_angle_radians", "front_yaw_half_angle_radians"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 < value <= math.pi:
                raise ValueError(f"{name} must be in (0, pi]")
        for name in ("steering_update_scale", "curvature_steering_update_scale"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= value <= MAXIMUM_STEERING_UPDATE_SCALE:
                raise ValueError(f"{name} must be in [0, {MAXIMUM_STEERING_UPDATE_SCALE}]")
        if self.steering_parameterization not in STEERING_PARAMETERIZATIONS:
            raise ValueError(f"steering_parameterization must be one of {STEERING_PARAMETERIZATIONS}")
        if not math.isfinite(self.action_saturation_tolerance):
            raise ValueError("action_saturation_tolerance must be finite")
        if not 0.0 <= self.action_saturation_tolerance < 1.0:
            raise ValueError("action_saturation_tolerance must be in [0, 1)")
        if not math.isfinite(self.collision_distance_tolerance_meters):
            raise ValueError("collision_distance_tolerance_meters must be finite")
        if self.collision_distance_tolerance_meters < 0.0:
            raise ValueError("collision_distance_tolerance_meters must be non-negative")
        if not isinstance(self.early_stop_on_collision, bool):
            raise TypeError("early_stop_on_collision must be a boolean")
        if not math.isfinite(self.early_stop_minimum_improvement):
            raise ValueError("early_stop_minimum_improvement must be finite")
        if self.early_stop_minimum_improvement < 0.0:
            raise ValueError("early_stop_minimum_improvement must be non-negative")
        if not isinstance(self.early_stop_patience_iterations, int):
            raise TypeError("early_stop_patience_iterations must be an integer")
        if self.early_stop_patience_iterations < 0:
            raise ValueError("early_stop_patience_iterations must be non-negative")


@dataclass(frozen=True)
class CostSnapshot:
    ego_collision: float
    background_collision: float
    drivable_area: float
    total: float
    background_collision_first_agent_idx: int
    background_collision_second_agent_idx: int
    background_collision_timestep_idx: int
    background_collision_signed_distance_meters: float
    background_collision_truncated: bool


@dataclass(frozen=True)
class ReGentSOptimizationResult:
    initial_actions: torch.Tensor
    optimized_actions: torch.Tensor
    optimized_action_mask: torch.Tensor
    optimized_states: torch.Tensor
    state_valid: torch.Tensor
    selection: CandidateSelection
    initial_costs: CostSnapshot | None
    final_costs: CostSnapshot | None
    selected_adversary_idx: int
    selected_adversary_id: int
    ego_collision_loss_adversary_idx: int
    ego_collision_loss_adversary_id: int
    gradient_norms: tuple[float, ...]
    acceleration_gradient_norms: tuple[float, ...]
    steering_gradient_norms: tuple[float, ...]
    front_divergence_iterations: tuple[int, ...]
    initial_action_saturation_fraction: float
    final_action_saturation_fraction: float
    steering_parameterization: str
    maximum_reconstruction_error_meters: float
    collision_timestep: int | None
    iteration_count: int
    best_iteration: int
    deterministic_seed: int
    success: bool
    background_collision: bool
    offroad: bool
    baseline_background_collision_pair_count: int
    background_collision_rejection_count: int
    offroad_rejection_count: int
    failure_reason: str | None
    frozen_ego_source: str
    cost_history: tuple[CostSnapshot, ...] = ()


def _single_scenario_payload(payload):
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return payload[0]
    raise ValueError("Frozen IDM capture requires exactly one Drive scenario")


def _ego_state_from_payload(payload, expected_controller):
    scenario = _single_scenario_payload(payload)
    agents = scenario.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Drive state contains no ego agent")
    ego = agents[0]
    if int(ego.get("id", -1)) != 0 or int(ego.get("controller", -1)) != expected_controller:
        raise ValueError(f"Stable agent zero must be controlled by {expected_controller}")
    values = np.asarray(
        (
            ego["sim_x"],
            ego["sim_y"],
            ego["sim_heading"],
            ego["sim_vx"],
            ego["sim_vy"],
            ego["sim_steering"],
        ),
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise ValueError("C SDC emitted a non-finite ego state")
    heading = np.float32(math.atan2(math.sin(float(values[2])), math.cos(float(values[2]))))
    signed_speed = np.float32(signed_speed_from_c_velocity(values[3], values[4], heading))
    state = np.asarray((values[0], values[1], heading, signed_speed, values[5]), dtype=np.float32)
    validity = int(ego["sim_valid"])
    if validity not in (0, 1):
        raise ValueError("C SDC emitted invalid ego validity")
    return scenario["scenario_id"], state, bool(validity)


def capture_frozen_idm_trajectory(
    drive,
    transition_count,
    *,
    seed=None,
    raster_resolution_meters=DEFAULT_RASTER_RESOLUTION_METERS,
):
    """Reset one Drive scenario and capture its native C IDM or replay ego rollout.

    Backgrounds remain under the Drive configuration's replay controller. The
    caller owns the Drive instance and remains responsible for closing it.
    """
    if not isinstance(transition_count, int) or transition_count < 1:
        raise ValueError("transition_count must be a positive integer")
    if drive.sdc_controller not in (binding.CONTROLLER_IDM, binding.CONTROLLER_REPLAY):
        raise ValueError("Frozen ego capture requires sdc_controller='idm' or 'replay'")
    if drive.non_sdc_controller != binding.CONTROLLER_REPLAY:
        raise ValueError("Frozen ego capture requires non_sdc_controller='replay'")
    if drive.simulation_mode != binding.SIMULATION_MODE_REPLAY:
        raise ValueError("Frozen ego capture requires simulation_mode='replay'")
    if drive.num_envs != 1:
        raise ValueError("Stage 5 frozen ego capture supports exactly one scenario")
    if drive.resample_frequency > 0 and transition_count >= drive.resample_frequency:
        raise ValueError("transition_count must end before Drive resamples the scenario")

    drive.reset(seed=seed)
    initial_payload = drive.get_state()
    scenario = export_drive_scenarios(
        drive,
        payload=initial_payload,
        raster_resolution_meters=raster_resolution_meters,
    )
    if transition_count > scenario.max_time_count - 1:
        raise ValueError("transition_count exceeds the exported scenario horizon")
    scenario_ids = []
    states = []
    validity = []
    scenario_id, state, valid = _ego_state_from_payload(initial_payload, drive.sdc_controller)
    scenario_ids.append(scenario_id)
    states.append(state)
    validity.append(valid)
    neutral_actions = np.zeros_like(drive.actions)
    for _ in range(transition_count):
        drive.step(neutral_actions)
        scenario_id, state, valid = _ego_state_from_payload(drive.get_state(), drive.sdc_controller)
        scenario_ids.append(scenario_id)
        states.append(state)
        validity.append(valid)
    if any(item != scenario.scenario_ids[0] for item in scenario_ids):
        raise RuntimeError("Drive changed scenario during frozen IDM capture")
    source = "c_idm" if drive.sdc_controller == binding.CONTROLLER_IDM else "c_replay"
    frozen_ego = FrozenEgoTrajectory(
        state=torch.from_numpy(np.ascontiguousarray(np.stack(states)[None, ...])),
        valid=torch.from_numpy(np.ascontiguousarray(np.asarray(validity, dtype=np.bool_)[None, ...])),
        scenario_ids=scenario.scenario_ids,
        source=source,
    )
    return scenario, frozen_ego


def mask_front_divergence_gradients(action_gradient, optimized_action_mask, divergent_agent_mask):
    """Mask all frozen actions and only steering for red-zone candidates."""
    if action_gradient.ndim != 4 or action_gradient.shape[-1] != 2:
        raise ValueError("action_gradient must have shape [batch, agent, time, 2]")
    if optimized_action_mask.dtype != torch.bool or optimized_action_mask.shape != action_gradient.shape[:-1]:
        raise ValueError("optimized_action_mask must be bool [batch, agent, time]")
    if divergent_agent_mask.dtype != torch.bool or divergent_agent_mask.shape != action_gradient.shape[:2]:
        raise ValueError("divergent_agent_mask must be bool [batch, agent]")
    masked = torch.where(optimized_action_mask[..., None], action_gradient, torch.zeros_like(action_gradient))
    steering_allowed = ~divergent_agent_mask[..., None]
    masked_steering = torch.where(
        steering_allowed,
        masked[..., ACTION_TARGET_STEERING],
        torch.zeros_like(masked[..., ACTION_TARGET_STEERING]),
    )
    return torch.stack((masked[..., ACTION_ACCELERATION], masked_steering), dim=-1)


def steering_conversion_metadata(scenario, optimized_action_mask, device):
    """Return per-transition wheelbase and the curvature a full wheel angle reaches."""
    wheelbase_meters = scenario.wheelbase_meters.to(device)
    optimized_agent = optimized_action_mask.any(dim=-1)
    usable = torch.isfinite(wheelbase_meters) & (wheelbase_meters > 0.0)
    if torch.any(optimized_agent & ~usable):
        raise ValueError("an optimized agent has a non-positive or non-finite wheelbase")
    # Frozen agents never reach the converter; the placeholder only keeps the math finite.
    safe_wheelbase = torch.where(optimized_agent, wheelbase_meters, torch.ones_like(wheelbase_meters))
    transition_count = optimized_action_mask.shape[-1]
    wheelbase_over_time = safe_wheelbase[..., None].expand(*safe_wheelbase.shape, transition_count).contiguous()
    return wheelbase_over_time, NORMALIZED_CURVATURE_LIMIT / safe_wheelbase[..., None]


def parameter_from_drive_actions(drive_actions, steering_parameterization, wheelbase_over_time):
    """Express normalized simulator actions in the optimizer's steering parameterization."""
    if steering_parameterization == STEERING_PARAMETERIZATION_WHEEL_ANGLE:
        return drive_actions
    curvature_per_meter = curvature_from_target_steering(
        drive_actions[..., ACTION_TARGET_STEERING] * TARGET_STEERING_SCALE_RADIANS,
        wheelbase_over_time,
    )
    return torch.stack((drive_actions[..., ACTION_ACCELERATION], curvature_per_meter), dim=-1)


def drive_actions_from_parameter(action_parameter, steering_parameterization, wheelbase_over_time):
    """Express optimizer parameters as the normalized actions the simulator consumes."""
    if steering_parameterization == STEERING_PARAMETERIZATION_WHEEL_ANGLE:
        return action_parameter
    steering_radians, _ = target_steering_from_curvature(
        action_parameter[..., ACTION_TARGET_STEERING], wheelbase_over_time
    )
    return torch.stack(
        (action_parameter[..., ACTION_ACCELERATION], steering_radians / TARGET_STEERING_SCALE_RADIANS),
        dim=-1,
    )


def _project_parameter(action_parameter, steering_parameter_limit):
    """Clamp acceleration to the action box and steering to its parameterization's limit."""
    action_parameter[..., ACTION_ACCELERATION] = torch.clamp(action_parameter[..., ACTION_ACCELERATION], -1.0, 1.0)
    action_parameter[..., ACTION_TARGET_STEERING] = torch.clamp(
        action_parameter[..., ACTION_TARGET_STEERING],
        -steering_parameter_limit,
        steering_parameter_limit,
    )


def _validate_optimization_inputs(scenario, frozen_ego, config, deterministic_seed, horizon_transition_count):
    if not isinstance(scenario, ScenarioBatch):
        raise TypeError("scenario must be a ScenarioBatch")
    if scenario.batch_size != 1:
        raise ValueError("Stage 5 optimizes exactly one scenario per job")
    if not isinstance(config, ReGentSOptimizationConfig):
        raise TypeError("config must be a ReGentSOptimizationConfig")
    if not isinstance(deterministic_seed, int) or deterministic_seed < 0 or deterministic_seed >= 2**63:
        raise ValueError("deterministic_seed must be an integer in [0, 2**63)")
    maximum_transition_count = scenario.max_time_count - 1
    if horizon_transition_count is None:
        horizon_transition_count = frozen_ego.state.shape[1] - 1 if frozen_ego is not None else maximum_transition_count
    if not isinstance(horizon_transition_count, int):
        raise TypeError("horizon_transition_count must be an integer")
    if horizon_transition_count < 1 or horizon_transition_count > maximum_transition_count:
        raise ValueError(f"horizon_transition_count must be in [1, {maximum_transition_count}]")
    if frozen_ego is not None:
        if not isinstance(frozen_ego, FrozenEgoTrajectory):
            raise TypeError("frozen_ego must be a FrozenEgoTrajectory")
        if frozen_ego.state.shape != (1, horizon_transition_count + 1, STATE_FEATURE_COUNT):
            raise ValueError("Frozen ego state does not match the optimization horizon")
        if frozen_ego.scenario_ids != scenario.scenario_ids:
            raise ValueError("Frozen ego scenario_ids do not match the ScenarioBatch")
        if frozen_ego.state.device != scenario.logged_state.device:
            raise ValueError("Frozen ego and scenario tensors must share a device")
    return horizon_transition_count


def _frozen_ego_fixture(scenario, inverse, horizon_transition_count):
    ego_idx = torch.where(scenario.ego_mask[0])[0]
    if ego_idx.numel() != 1:
        raise ValueError("Logged frozen ego fixture requires exactly one ego")
    ego_idx = int(ego_idx.item())
    return FrozenEgoTrajectory(
        state=inverse.state_with_estimated_steering[:, ego_idx, : horizon_transition_count + 1].detach().clone(),
        valid=scenario.state_valid[:, ego_idx, : horizon_transition_count + 1].detach().clone(),
        scenario_ids=scenario.scenario_ids,
        source="logged_fixture",
    )


def _compose_rollout(
    scenario,
    inverse,
    actions,
    frozen_ego,
    horizon_transition_count,
    optimized_agent_mask=None,
):
    reference_state = inverse.state_with_estimated_steering[:, :, : horizon_transition_count + 1].to(actions.device)
    transition_valid = inverse.action_valid[:, :, :horizon_transition_count].to(actions.device)
    if optimized_agent_mask is None:
        optimized_agent_mask = scenario.candidate_adversary_mask.to(actions.device)
    if optimized_agent_mask.dtype != torch.bool or optimized_agent_mask.shape != scenario.ego_mask.shape:
        raise ValueError("optimized_agent_mask must be bool [batch, agent]")
    optimized_transition_valid = transition_valid & optimized_agent_mask[..., None]
    ego_idx = int(torch.where(scenario.ego_mask[0])[0].item())
    reference_state = reference_state.clone()
    reference_state[:, ego_idx] = frozen_ego.state
    state_valid = scenario.state_valid[:, :, : horizon_transition_count + 1].to(actions.device).clone()
    state_valid[:, ego_idx] = frozen_ego.valid

    current_state = reference_state[:, :, 0]
    rollout = [current_state]
    for timestep in range(horizon_transition_count):
        if timestep > 0:
            run_start = optimized_transition_valid[:, :, timestep] & ~optimized_transition_valid[:, :, timestep - 1]
            current_state = torch.where(run_start[..., None], reference_state[:, :, timestep], current_state)
        next_state = reference_state[:, :, timestep + 1].clone()
        active_idx = torch.where(optimized_transition_valid[:, :, timestep])
        if active_idx[0].numel() > 0:
            proposed_state = classic_step(
                current_state[active_idx],
                actions[active_idx[0], active_idx[1], timestep],
                scenario.wheelbase_meters.to(actions.device)[active_idx],
                scenario.maximum_speed_mps.to(actions.device)[active_idx],
                scenario.dt_seconds,
            )
            next_state[active_idx] = proposed_state
        current_state = next_state
        current_state[:, ego_idx] = frozen_ego.state[:, timestep + 1]
        rollout.append(current_state)
    return torch.stack(rollout, dim=2), state_valid


def _ego_indices(ego_mask):
    """Per-scenario ego column; ScenarioBatch guarantees exactly one ego per row."""
    return torch.argmax(ego_mask.to(torch.int64), dim=-1)


def _candidate_ego_distances(boxes, state_valid, ego_indices, candidate_indices, scenario_idx):
    """Return per-candidate ego box distances and joint validity over the full horizon."""
    ego_idx = int(ego_indices[scenario_idx].item())
    scenario_boxes = boxes[scenario_idx]
    distances = signed_box_distance(scenario_boxes[candidate_indices], scenario_boxes[ego_idx][None])
    jointly_valid = state_valid[scenario_idx, candidate_indices] & state_valid[scenario_idx, ego_idx][None]
    return distances, jointly_valid


def _first_ego_collision(boxes, state_valid, ego_indices, candidate_mask, tolerance_meters, scenario_idx):
    candidate_indices = torch.where(candidate_mask[scenario_idx])[0]
    if candidate_indices.numel() == 0:
        return None, -1
    distances, jointly_valid = _candidate_ego_distances(
        boxes, state_valid, ego_indices, candidate_indices, scenario_idx
    )
    overlapping = jointly_valid & (distances <= tolerance_meters)
    # Timestep-major keys rank the earliest overlap first, ties going to the lowest
    # agent index because candidate_indices is ascending.
    candidate_count, timestep_count = overlapping.shape
    timestep_keys = torch.arange(timestep_count, device=overlapping.device) * candidate_count
    ranking_keys = timestep_keys[None] + torch.arange(candidate_count, device=overlapping.device)[:, None]
    no_overlap_key = candidate_count * timestep_count
    best_key = int(torch.where(overlapping, ranking_keys, no_overlap_key).min().item())
    if best_key == no_overlap_key:
        return None, -1
    return best_key // candidate_count, int(candidate_indices[best_key % candidate_count].item())


def _candidate_background_pair_indices(scenario, state_valid, candidate_mask):
    """Flat (scenario, left, right) columns over every candidate-involving background pair.

    Pair counts differ per scenario, so the batch dimension is carried as a row
    rather than padded; the layout is fixed for a run, which lets a signature be
    compared elementwise against its baseline.
    """
    columns = []
    for scenario_idx in range(scenario.batch_size):
        background_idx = torch.where(scenario.vehicle_mask[scenario_idx] & ~scenario.ego_mask[scenario_idx])[0]
        if background_idx.numel() < 2:
            continue
        local_pairs = torch.triu_indices(
            background_idx.numel(),
            background_idx.numel(),
            offset=1,
            device=scenario.logged_state.device,
        )
        pair_indices = background_idx[local_pairs]
        candidate_involved = (
            candidate_mask[scenario_idx, pair_indices[0]] | candidate_mask[scenario_idx, pair_indices[1]]
        )
        jointly_valid = state_valid[scenario_idx, pair_indices[0]] & state_valid[scenario_idx, pair_indices[1]]
        kept = pair_indices[:, candidate_involved & torch.any(jointly_valid, dim=-1)]
        scenario_row = torch.full((1, kept.shape[1]), scenario_idx, dtype=torch.int64, device=kept.device)
        columns.append(torch.cat((scenario_row, kept), dim=0))
    if not columns:
        return torch.empty((3, 0), dtype=torch.int64, device=scenario.logged_state.device)
    return torch.cat(columns, dim=1)


def _background_collision_signature(boxes, state_valid, tolerance_meters, pair_indices):
    pair_count = pair_indices.shape[1]
    signature = torch.zeros(pair_count, dtype=torch.bool, device=boxes.device)
    for chunk_start in range(0, pair_count, BACKGROUND_COLLISION_PAIR_CHUNK_SIZE):
        chunk_pairs = pair_indices[:, chunk_start : chunk_start + BACKGROUND_COLLISION_PAIR_CHUNK_SIZE]
        scenario_row, left_indices, right_indices = chunk_pairs
        distances = signed_box_distance(
            boxes[scenario_row, left_indices],
            boxes[scenario_row, right_indices],
        )
        jointly_valid = state_valid[scenario_row, left_indices] & state_valid[scenario_row, right_indices]
        signature[chunk_start : chunk_start + chunk_pairs.shape[1]] = torch.any(
            jointly_valid & (distances <= tolerance_meters),
            dim=-1,
        )
    return signature


def _candidate_offroad_signature(boxes, state_valid, scenario, candidate_mask):
    signature = torch.zeros((*boxes.shape[:3], 4), dtype=torch.bool, device=boxes.device)
    for scenario_idx, raster in enumerate(scenario.drivable_area_rasters):
        candidate_indices = torch.where(candidate_mask[scenario_idx])[0]
        if candidate_indices.numel() == 0:
            continue
        raster_mask = raster.mask.to(boxes.device)
        corners = oriented_box_corners(boxes[scenario_idx, candidate_indices])
        grid = raster.transform.world_to_grid(corners)
        column = torch.round(grid[..., 0]).to(torch.int64)
        row = torch.round(grid[..., 1]).to(torch.int64)
        outside = (column < 0) | (column >= raster.transform.width)
        outside |= (row < 0) | (row >= raster.transform.height)
        safe_column = column.clamp(0, raster.transform.width - 1)
        safe_row = row.clamp(0, raster.transform.height - 1)
        outside |= ~raster_mask[safe_row, safe_column]
        # Invalid timesteps carry a placeholder box, so they never enter the signature.
        signature[scenario_idx, candidate_indices] = outside & state_valid[scenario_idx, candidate_indices][..., None]
    return signature


def _baseline_corner_potential(states, state_valid, scenario, out_of_bounds_rasters):
    """Return the detached constant used to zero-center the absolute boundary loss."""
    boxes = _masked_boxes(states, state_valid, scenario.length_meters, scenario.width_meters)
    corners = oriented_box_corners(boxes)
    return torch.stack(
        [
            sample_out_of_bounds_potential(corners[scenario_idx], raster)
            for scenario_idx, raster in enumerate(out_of_bounds_rasters)
        ]
    ).detach()


def _cost_snapshot(costs, scenario_idx):
    return CostSnapshot(
        ego_collision=float(costs.ego_collision[scenario_idx].detach().item()),
        background_collision=float(costs.background_collision[scenario_idx].detach().item()),
        drivable_area=float(costs.drivable_area[scenario_idx].detach().item()),
        total=float(costs.total[scenario_idx].detach().item()),
        background_collision_first_agent_idx=int(costs.background_collision_first_agent_idx[scenario_idx].item()),
        background_collision_second_agent_idx=int(costs.background_collision_second_agent_idx[scenario_idx].item()),
        background_collision_timestep_idx=int(costs.background_collision_timestep_idx[scenario_idx].item()),
        background_collision_signed_distance_meters=float(
            costs.background_collision_signed_distance_meters[scenario_idx].item()
        ),
        background_collision_truncated=bool(costs.background_collision_truncated[scenario_idx].item()),
    )


def _saturation_fraction(actions, optimized_action_mask, tolerance, scenario_idx):
    expanded_mask = optimized_action_mask[scenario_idx][..., None].expand_as(actions[scenario_idx])
    if not expanded_mask.any():
        return 0.0
    saturated = actions[scenario_idx].detach().abs() >= 1.0 - tolerance
    return float(saturated[expanded_mask].to(torch.float32).mean().item())


def _selected_adversary(boxes, state_valid, ego_indices, candidate_mask, scenario_idx):
    candidate_indices = torch.where(candidate_mask[scenario_idx])[0]
    if candidate_indices.numel() == 0:
        return -1
    distances, jointly_valid = _candidate_ego_distances(
        boxes, state_valid, ego_indices, candidate_indices, scenario_idx
    )
    jointly_valid_counts = jointly_valid.sum(dim=-1)
    summed_distances = torch.where(jointly_valid, distances, torch.zeros_like(distances)).sum(dim=-1)
    mean_distances = (summed_distances / jointly_valid_counts.clamp_min(1)).detach()
    mean_distances = mean_distances.masked_fill(jointly_valid_counts == 0, math.inf)
    if not torch.isfinite(mean_distances).any():
        return -1
    return int(candidate_indices[int(torch.argmin(mean_distances).item())].item())


def _zero_adam_steering_momentum(optimizer, action_parameter, divergent_agent_mask):
    state = optimizer.state.get(action_parameter)
    if not state:
        return
    steering_mask = divergent_agent_mask[..., None]
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        moment = state.get(name)
        if moment is None:
            continue
        moment[..., ACTION_TARGET_STEERING] = torch.where(
            steering_mask,
            torch.zeros_like(moment[..., ACTION_TARGET_STEERING]),
            moment[..., ACTION_TARGET_STEERING],
        )


def optimize_frozen_ego_scenario(
    scenario,
    frozen_ego=None,
    config=None,
    *,
    deterministic_seed=0,
    horizon_transition_count=None,
    scene_suitable=None,
    show_progress=True,
    inverse_dynamics=None,
):
    """Optimize Stage 3 background actions against one detached ego rollout."""
    if config is None:
        config = ReGentSOptimizationConfig()
    horizon_transition_count = _validate_optimization_inputs(
        scenario, frozen_ego, config, deterministic_seed, horizon_transition_count
    )
    if inverse_dynamics is None:
        inverse = estimate_expert_actions(
            scenario,
            horizon_transition_count=horizon_transition_count,
        )
    else:
        if not isinstance(inverse_dynamics, InverseDynamicsResult):
            raise TypeError("inverse_dynamics must be an InverseDynamicsResult")
        expected_prefix = (scenario.batch_size, scenario.max_agent_count)
        if inverse_dynamics.action_valid.shape[:2] != expected_prefix:
            raise ValueError("inverse_dynamics does not match the scenario batch and agent dimensions")
        if inverse_dynamics.action_valid.shape[-1] < horizon_transition_count:
            raise ValueError("inverse_dynamics does not cover the optimization horizon")
        if inverse_dynamics.actions.device != scenario.logged_state.device:
            raise ValueError("inverse_dynamics and scenario tensors must share a device")
        inverse = inverse_dynamics
    if frozen_ego is None:
        frozen_ego = _frozen_ego_fixture(scenario, inverse, horizon_transition_count)
    baseline_actions = inverse.actions[:, :, :horizon_transition_count].detach().clone()
    selection = select_adversary_candidates(
        scenario,
        config.filter,
        horizon_transition_count=horizon_transition_count,
        scene_suitable=scene_suitable,
    )
    reference_states, state_valid = _compose_rollout(
        scenario,
        inverse,
        baseline_actions,
        frozen_ego,
        horizon_transition_count,
        selection.candidate_mask,
    )
    valid_reconstruction = inverse.action_valid[:, :, :horizon_transition_count]
    reconstruction_values = inverse.residual_meters[:, :, :horizon_transition_count][valid_reconstruction]
    maximum_reconstruction_error = float(reconstruction_values.max().item()) if reconstruction_values.numel() else 0.0
    ego_indices = _ego_indices(scenario.ego_mask)
    initial_saturation = _saturation_fraction(
        baseline_actions, selection.optimized_action_mask, config.action_saturation_tolerance, 0
    )
    reference_boxes = _masked_boxes(reference_states, state_valid, scenario.length_meters, scenario.width_meters)
    baseline_offroad_signature = _candidate_offroad_signature(
        reference_boxes, state_valid, scenario, selection.candidate_mask
    )
    background_pair_indices = _candidate_background_pair_indices(
        scenario,
        state_valid,
        selection.candidate_mask,
    )
    baseline_background_collision_signature = _background_collision_signature(
        reference_boxes,
        state_valid,
        config.collision_distance_tolerance_meters,
        background_pair_indices,
    )
    baseline_background_collision_pair_count = int(baseline_background_collision_signature.sum().item())

    if not bool(selection.scene_eligible[0]):
        return ReGentSOptimizationResult(
            initial_actions=baseline_actions,
            optimized_actions=baseline_actions.clone(),
            optimized_action_mask=selection.optimized_action_mask,
            optimized_states=reference_states,
            state_valid=state_valid,
            selection=selection,
            initial_costs=None,
            final_costs=None,
            selected_adversary_idx=-1,
            selected_adversary_id=-1,
            ego_collision_loss_adversary_idx=-1,
            ego_collision_loss_adversary_id=-1,
            gradient_norms=(),
            acceleration_gradient_norms=(),
            steering_gradient_norms=(),
            front_divergence_iterations=(),
            initial_action_saturation_fraction=initial_saturation,
            final_action_saturation_fraction=initial_saturation,
            steering_parameterization=config.steering_parameterization,
            maximum_reconstruction_error_meters=maximum_reconstruction_error,
            collision_timestep=None,
            iteration_count=0,
            best_iteration=0,
            deterministic_seed=deterministic_seed,
            success=False,
            background_collision=False,
            offroad=False,
            baseline_background_collision_pair_count=baseline_background_collision_pair_count,
            background_collision_rejection_count=0,
            offroad_rejection_count=0,
            failure_reason="scene_filtered:" + ",".join(selection.scene_reasons_for(0)),
            frozen_ego_source=frozen_ego.source,
            cost_history=(),
        )

    out_of_bounds_rasters = prepare_out_of_bounds_rasters(
        scenario.drivable_area_rasters,
        config.costs,
        device=scenario.logged_state.device,
        dtype=scenario.logged_state.dtype,
    )
    baseline_corner_potential = _baseline_corner_potential(
        reference_states, state_valid, scenario, out_of_bounds_rasters
    )
    wheelbase_over_time, achievable_curvature = steering_conversion_metadata(
        scenario, selection.optimized_action_mask, baseline_actions.device
    )
    in_curvature_space = config.steering_parameterization == STEERING_PARAMETERIZATION_CURVATURE
    steering_update_scale = (
        config.curvature_steering_update_scale if in_curvature_space else config.steering_update_scale
    )
    steering_parameter_limit = achievable_curvature if in_curvature_space else baseline_actions.new_ones(())
    baseline_parameter = parameter_from_drive_actions(
        baseline_actions, config.steering_parameterization, wheelbase_over_time
    )
    action_parameter = torch.nn.Parameter(baseline_parameter.clone())
    optimizer = torch.optim.Adam(
        (action_parameter,),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        foreach=False,
    )
    candidate_mask = selection.candidate_mask
    background_vehicle_mask = candidate_mask
    gradient_norms = []
    acceleration_gradient_norms = []
    steering_gradient_norms = []
    divergence_iterations = []
    best_actions = baseline_actions.clone()
    best_states = reference_states.detach().clone()
    best_costs = None
    best_total = math.inf
    best_iteration = 0
    initial_costs = None
    cost_history = []
    collision_timestep = None
    collision_agent_idx = -1
    success = False
    best_success_total = math.inf
    best_success_actions = None
    best_success_states = None
    best_success_costs = None
    best_success_iteration = 0
    failure_reason = None
    completed_update_count = 0
    no_improvement_count = 0
    background_collision_rejection_count = 0
    offroad_rejection_count = 0

    if show_progress:
        pbar = tqdm(range(config.iteration_count + 1), desc="Optimizing", leave=False)
        iterator = pbar
    else:
        iterator = range(config.iteration_count + 1)
        pbar = None

    for iteration in iterator:
        # Frozen entries take the baseline verbatim so a curvature round trip cannot
        # perturb an action the optimizer is not allowed to change.
        drive_actions = torch.where(
            selection.optimized_action_mask[..., None],
            drive_actions_from_parameter(action_parameter, config.steering_parameterization, wheelbase_over_time),
            baseline_actions,
        )
        states, state_valid = _compose_rollout(
            scenario,
            inverse,
            drive_actions,
            frozen_ego,
            horizon_transition_count,
            selection.candidate_mask,
        )
        costs = combined_regents_cost(
            states,
            state_valid,
            scenario.length_meters,
            scenario.width_meters,
            scenario.ego_mask,
            candidate_mask,
            background_vehicle_mask,
            candidate_mask,
            out_of_bounds_rasters,
            config.costs,
            baseline_corner_potential,
        )
        if not all(
            torch.isfinite(value).all()
            for value in (costs.ego_collision, costs.background_collision, costs.drivable_area, costs.total)
        ):
            failure_reason = "nonfinite_loss"
            break
        if iteration % 10 == 0 and pbar is not None:
            pbar.set_postfix(
                loss=f"{costs.total.item():.4f}", best=f"{best_total:.4f}" if best_total != math.inf else "inf"
            )
        snapshot = _cost_snapshot(costs, 0)
        cost_history.append(snapshot)
        if initial_costs is None:
            initial_costs = snapshot
        detached_boxes = _masked_boxes(states.detach(), state_valid, scenario.length_meters, scenario.width_meters)
        background_collision_signature = _background_collision_signature(
            detached_boxes,
            state_valid,
            config.collision_distance_tolerance_meters,
            background_pair_indices,
        )
        background_collision = bool(
            torch.any(background_collision_signature & ~baseline_background_collision_signature)
        )
        offroad_signature = _candidate_offroad_signature(detached_boxes, state_valid, scenario, candidate_mask)
        offroad = bool(torch.any(offroad_signature & ~baseline_offroad_signature))
        background_collision_rejection_count += int(background_collision)
        offroad_rejection_count += int(offroad)
        iteration_collision_timestep, iteration_collision_agent_idx = _first_ego_collision(
            detached_boxes,
            state_valid,
            ego_indices,
            candidate_mask,
            config.collision_distance_tolerance_meters,
            0,
        )
        feasible = not background_collision and not offroad
        improved = feasible and snapshot.total < best_total - config.early_stop_minimum_improvement
        if improved:
            best_total = snapshot.total
            best_actions = drive_actions.detach().clone()
            best_states = states.detach().clone()
            best_costs = snapshot
            best_iteration = iteration
            no_improvement_count = 0
        elif iteration > 0:
            no_improvement_count += 1

        initial_reconstruction_collision = iteration_collision_timestep is not None and feasible and iteration == 0
        if initial_reconstruction_collision:
            failure_reason = "initial_reconstruction_collision"
            break
        generated_collision = iteration_collision_timestep is not None and feasible and iteration > 0
        if generated_collision and snapshot.total < best_success_total:
            success = True
            best_success_total = snapshot.total
            best_success_actions = drive_actions.detach().clone()
            best_success_states = states.detach().clone()
            best_success_costs = snapshot
            best_success_iteration = iteration
            collision_timestep = iteration_collision_timestep
            collision_agent_idx = iteration_collision_agent_idx
        if generated_collision and config.early_stop_on_collision:
            break
        if iteration == config.iteration_count:
            break
        if config.early_stop_patience_iterations > 0 and no_improvement_count >= config.early_stop_patience_iterations:
            failure_reason = "stagnated"
            break

        optimizer.zero_grad(set_to_none=True)
        costs.total.backward()
        if action_parameter.grad is None or not torch.isfinite(action_parameter.grad).all():
            failure_reason = "nonfinite_gradient"
            break
        divergent = front_divergence_mask(
            states.detach(),
            state_valid,
            scenario.ego_mask,
            candidate_mask,
            tau_front=config.tau_front,
            applicability_half_angle_radians=config.front_applicability_half_angle_radians,
            yaw_half_angle_radians=config.front_yaw_half_angle_radians,
        )
        masked_gradient = mask_front_divergence_gradients(
            action_parameter.grad,
            selection.optimized_action_mask,
            divergent,
        )
        action_parameter.grad.copy_(masked_gradient)
        acceleration_gradient = masked_gradient[..., ACTION_ACCELERATION]
        steering_gradient = masked_gradient[..., ACTION_TARGET_STEERING]
        acceleration_gradient_norms.append(float(torch.linalg.vector_norm(acceleration_gradient).item()))
        steering_gradient_norms.append(float(torch.linalg.vector_norm(steering_gradient).item()))
        gradient_norms.append(float(torch.linalg.vector_norm(masked_gradient).item()))
        if divergent.any():
            divergence_iterations.append(iteration)

        previous_steering = action_parameter.detach()[..., ACTION_TARGET_STEERING].clone()
        optimizer.step()
        with torch.no_grad():
            _project_parameter(action_parameter, steering_parameter_limit)
            action_parameter.copy_(
                torch.where(
                    selection.optimized_action_mask[..., None],
                    action_parameter,
                    baseline_parameter,
                )
            )
            divergent_steering = divergent[..., None]
            damped_steering = previous_steering + steering_update_scale * (
                action_parameter[..., ACTION_TARGET_STEERING] - previous_steering
            )
            action_parameter[..., ACTION_TARGET_STEERING] = torch.where(
                divergent_steering,
                previous_steering,
                damped_steering,
            )
            # A scale above one extrapolates past the Adam step and can leave the box.
            _project_parameter(action_parameter, steering_parameter_limit)
        _zero_adam_steering_momentum(optimizer, action_parameter, divergent)
        completed_update_count += 1

    if success:
        best_actions = best_success_actions
        best_states = best_success_states
        final_costs = best_success_costs
        best_iteration = best_success_iteration
    elif best_costs is None:
        best_actions = baseline_actions.clone()
        best_states = reference_states.detach().clone()
        final_costs = initial_costs
        best_iteration = 0
        if failure_reason is None:
            failure_reason = "initial_state_infeasible"
    else:
        final_costs = best_costs
    if not success and failure_reason is None:
        failure_reason = "iteration_limit"

    frozen_storage_mask = ~selection.optimized_action_mask[..., None].expand_as(best_actions)
    if not torch.equal(best_actions[frozen_storage_mask], baseline_actions[frozen_storage_mask]):
        raise RuntimeError("Optimizer changed a non-candidate or invalid action")
    best_boxes = _masked_boxes(best_states, state_valid, scenario.length_meters, scenario.width_meters)
    ego_collision_loss_adversary_idx = _selected_adversary(best_boxes, state_valid, ego_indices, candidate_mask, 0)
    ego_collision_loss_adversary_id = int(scenario.agent_id[0, ego_collision_loss_adversary_idx].item())
    selected_idx = collision_agent_idx if success else ego_collision_loss_adversary_idx
    selected_id = int(scenario.agent_id[0, selected_idx].item()) if selected_idx >= 0 else -1
    final_background_collision_signature = _background_collision_signature(
        best_boxes,
        state_valid,
        config.collision_distance_tolerance_meters,
        background_pair_indices,
    )
    final_background_collision = bool(
        torch.any(final_background_collision_signature & ~baseline_background_collision_signature)
    )
    final_offroad_signature = _candidate_offroad_signature(best_boxes, state_valid, scenario, candidate_mask)
    final_offroad = bool(torch.any(final_offroad_signature & ~baseline_offroad_signature))
    return ReGentSOptimizationResult(
        initial_actions=baseline_actions,
        optimized_actions=best_actions,
        optimized_action_mask=selection.optimized_action_mask,
        optimized_states=best_states,
        state_valid=state_valid,
        selection=selection,
        initial_costs=initial_costs,
        final_costs=final_costs,
        selected_adversary_idx=selected_idx,
        selected_adversary_id=selected_id,
        ego_collision_loss_adversary_idx=ego_collision_loss_adversary_idx,
        ego_collision_loss_adversary_id=ego_collision_loss_adversary_id,
        gradient_norms=tuple(gradient_norms),
        acceleration_gradient_norms=tuple(acceleration_gradient_norms),
        steering_gradient_norms=tuple(steering_gradient_norms),
        front_divergence_iterations=tuple(divergence_iterations),
        initial_action_saturation_fraction=initial_saturation,
        final_action_saturation_fraction=_saturation_fraction(
            best_actions, selection.optimized_action_mask, config.action_saturation_tolerance, 0
        ),
        steering_parameterization=config.steering_parameterization,
        maximum_reconstruction_error_meters=maximum_reconstruction_error,
        collision_timestep=collision_timestep,
        iteration_count=completed_update_count,
        best_iteration=best_iteration,
        deterministic_seed=deterministic_seed,
        success=success,
        background_collision=final_background_collision,
        offroad=final_offroad,
        baseline_background_collision_pair_count=baseline_background_collision_pair_count,
        background_collision_rejection_count=background_collision_rejection_count,
        offroad_rejection_count=offroad_rejection_count,
        failure_reason=failure_reason,
        frozen_ego_source=frozen_ego.source,
        cost_history=tuple(cost_history),
    )
