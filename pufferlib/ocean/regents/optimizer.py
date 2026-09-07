"""Frozen-ego ReGentS optimization for one offline scenario."""

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch._inductor import config as torch_inductor_config
from tqdm import tqdm

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents.adapter import DEFAULT_RASTER_RESOLUTION_METERS, export_drive_scenarios
from pufferlib.ocean.regents.dynamics import (
    ACTION_ACCELERATION,
    ACTION_TARGET_STEERING,
    TARGET_STEERING_SCALE_RADIANS,
    _classic_step,
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
    BOX_FEATURE_COUNT,
    box_separation_lower_bound,
    oriented_box_corners,
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
DEFAULT_STEERING_UPDATE_SCALE = 0.5
MAXIMUM_STEERING_UPDATE_SCALE = 10.0
BACKGROUND_COLLISION_PAIR_CHUNK_SIZE = 4096
# Drive pins the ego to stable agent row zero, and `regents_get_states` preserves that order.
STABLE_EGO_AGENT_IDX = 0
REGENTS_EGO_ACTION_FEATURE_COUNT = 2

# Steering may be optimized as the simulator's normalized target wheel angle or as the
# reference's path curvature. The two carry different units, so each keeps its own
# update scale; `DEFAULT_CURVATURE_STEERING_UPDATE_SCALE` is the ReGentS value.
STEERING_PARAMETERIZATION_WHEEL_ANGLE = "wheel_angle"
STEERING_PARAMETERIZATION_CURVATURE = "curvature"
STEERING_PARAMETERIZATIONS = (STEERING_PARAMETERIZATION_WHEEL_ANGLE, STEERING_PARAMETERIZATION_CURVATURE)
DEFAULT_STEERING_PARAMETERIZATION = STEERING_PARAMETERIZATION_CURVATURE
DEFAULT_CURVATURE_STEERING_UPDATE_SCALE = 0.5

# A curvature sitting exactly at full lock round trips, in float32, to a wheel angle one
# ulp above the action box, which the C injector rejects. Project just inside the limit.
CURVATURE_PARAMETER_LIMIT_MARGIN = 1.0 - 1e-6


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
    # Trades reproducibility for rollout speed. The fused step's gradients differ from
    # eager, so a run with this on explores a different optimization path and generates
    # different adversaries. Off unless a caller accepts that.
    compile_dynamics: bool = False

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
        if not isinstance(self.compile_dynamics, bool):
            raise TypeError("compile_dynamics must be a boolean")
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
    """Current iterate and diagnostics.

    Legacy field names are retained for artifact compatibility: `best_iteration`
    identifies the returned iterate; rejection counts count infraction-bearing
    evaluations, which no longer prevent returning an iterate.
    """

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
    agent_count = int(_single_scenario_payload(initial_payload)["num_total_agents"])
    state_scratch = np.empty((agent_count, STATE_FEATURE_COUNT), dtype=np.float32)
    valid_scratch = np.empty(agent_count, dtype=np.bool_)
    ego_action_scratch = np.empty(REGENTS_EGO_ACTION_FEATURE_COUNT, dtype=np.float32)
    for _ in range(transition_count):
        drive.step(neutral_actions)
        binding.regents_get_states(drive.c_envs, state_scratch, valid_scratch, ego_action_scratch)
        if not np.isfinite(state_scratch[STABLE_EGO_AGENT_IDX]).all():
            raise ValueError("C SDC emitted a non-finite ego state")
        states.append(state_scratch[STABLE_EGO_AGENT_IDX].copy())
        validity.append(bool(valid_scratch[STABLE_EGO_AGENT_IDX]))
    # The buffer getter carries no scenario identity, so the horizon is bracketed by a
    # dict read at each end rather than one per step.
    scenario_ids.append(_ego_state_from_payload(drive.get_state(), drive.sdc_controller)[0])
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
    """Legacy gradient-mask utility; reference updates cancel divergence after Adam."""
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
    if scenario.batch_size < 1:
        raise ValueError("scenario must contain at least one scenario")
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
        expected_frozen_shape = (scenario.batch_size, horizon_transition_count + 1, STATE_FEATURE_COUNT)
        if frozen_ego.state.shape != expected_frozen_shape:
            raise ValueError("Frozen ego state does not match the optimization batch and horizon")
        if frozen_ego.scenario_ids != scenario.scenario_ids:
            raise ValueError("Frozen ego scenario_ids do not match the ScenarioBatch")
        if frozen_ego.state.device != scenario.logged_state.device:
            raise ValueError("Frozen ego and scenario tensors must share a device")
    return horizon_transition_count


def _frozen_ego_fixture(scenario, inverse, horizon_transition_count):
    if not torch.all(scenario.ego_mask.sum(dim=-1) == 1):
        raise ValueError("Logged frozen ego fixture requires exactly one ego per scenario")
    ego_indices = _ego_indices(scenario.ego_mask)
    batch_indices = torch.arange(scenario.batch_size, device=scenario.ego_mask.device)
    horizon_slice = slice(None, horizon_transition_count + 1)
    return FrozenEgoTrajectory(
        state=inverse.state_with_estimated_steering[batch_indices, ego_indices, horizon_slice].detach().clone(),
        valid=scenario.state_valid[batch_indices, ego_indices, horizon_slice].detach().clone(),
        scenario_ids=scenario.scenario_ids,
        source="logged_fixture",
    )


_compiled_step_cache = None


def _compiled_classic_step():
    """Fuse the rollout step once per process. Faster, but NOT bit-identical to eager.

    The step is ~30 elementwise ops on a few dozen rows, so eager mode is dominated by
    dispatch in both directions. Two settings are forced rather than left to defaults:
    dynamic shapes, because the candidate row count is constant within one optimization
    but differs between scenarios and per-shape recompiles would exhaust the dynamo cache
    and silently fall back to eager; and scalar math, because the vectorized approximations
    otherwise drift the rollout by more than C_REPLAY_TOLERANCE and every generation fails
    the C parity gate. Even so the gradients differ from eager, which moves the optimization
    onto a different path and changes which adversaries a run produces.
    """
    global _compiled_step_cache
    if _compiled_step_cache is None:
        torch_inductor_config.cpp.simdlen = 0
        _compiled_step_cache = torch.compile(_classic_step, dynamic=True)
    return _compiled_step_cache


def _compose_rollout(
    scenario,
    inverse,
    actions,
    frozen_ego,
    horizon_transition_count,
    optimized_agent_mask=None,
    compile_dynamics=False,
):
    reference_state = inverse.state_with_estimated_steering[:, :, : horizon_transition_count + 1].to(actions.device)
    transition_valid = inverse.action_valid[:, :, :horizon_transition_count].to(actions.device)
    if optimized_agent_mask is None:
        optimized_agent_mask = scenario.candidate_adversary_mask.to(actions.device)
    if optimized_agent_mask.dtype != torch.bool or optimized_agent_mask.shape != scenario.ego_mask.shape:
        raise ValueError("optimized_agent_mask must be bool [batch, agent]")
    optimized_transition_valid = transition_valid & optimized_agent_mask[..., None]
    ego_indices = _ego_indices(scenario.ego_mask).to(actions.device)
    batch_indices = torch.arange(scenario.batch_size, device=actions.device)
    reference_state = reference_state.clone()
    reference_state[batch_indices, ego_indices] = frozen_ego.state
    state_valid = scenario.state_valid[:, :, : horizon_transition_count + 1].to(actions.device).clone()
    state_valid[batch_indices, ego_indices] = frozen_ego.valid

    # Only candidate rows are ever integrated, so the sequential loop carries those rows
    # alone instead of every padded agent. Every other agent keeps its reference verbatim,
    # and the constant row count lets the fused step compile once per optimization.
    candidate_batch_rows, candidate_agent_rows = torch.where(optimized_agent_mask)
    if candidate_batch_rows.numel() == 0:
        return reference_state, state_valid
    candidate_rows = (candidate_batch_rows, candidate_agent_rows)
    candidate_active = optimized_transition_valid[candidate_rows]
    candidate_reference = reference_state[candidate_rows]
    candidate_actions = actions[candidate_rows]
    candidate_wheelbase = scenario.wheelbase_meters.to(actions.device)[candidate_rows]
    candidate_maximum_speed = scenario.maximum_speed_mps.to(actions.device)[candidate_rows]
    # A run start re-seeds integration from the reference; timestep zero already starts there.
    run_start = torch.zeros_like(candidate_active)
    run_start[:, 1:] = candidate_active[:, 1:] & ~candidate_active[:, :-1]

    step = _compiled_classic_step() if compile_dynamics else _classic_step
    current_state = candidate_reference[:, 0]
    rollout = [current_state]
    for timestep in range(horizon_transition_count):
        active = candidate_active[:, timestep, None]
        if timestep > 0:
            current_state = torch.where(run_start[:, timestep, None], candidate_reference[:, timestep], current_state)
        # Inactive rows still enter the step so its shape stays constant across timesteps.
        # They take a finite reference state, because a non-finite input would send NaN
        # back through their masked-out gradient and into the active rows' actions.
        step_state = torch.where(active, current_state, candidate_reference[:, timestep])
        proposed_state = step(
            step_state,
            candidate_actions[:, timestep],
            candidate_wheelbase,
            candidate_maximum_speed,
            scenario.dt_seconds,
        )
        current_state = torch.where(active, proposed_state, candidate_reference[:, timestep + 1])
        rollout.append(current_state)
    states = reference_state.clone()
    states[candidate_rows] = torch.stack(rollout, dim=1)
    return states, state_valid


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


def _box_contact_mask(boxes_a, boxes_b, jointly_valid, tolerance_meters):
    """Contact flags over a broadcast box grid, exact only where separation is unproven.

    The circumscribed-radius bound never rules out a true contact, so this matches an
    all-exact scan while running the Minkowski distance on a small fraction of entries.
    """
    grid_shape = jointly_valid.shape
    expanded_a = boxes_a.expand(*grid_shape, BOX_FEATURE_COUNT)
    expanded_b = boxes_b.expand(*grid_shape, BOX_FEATURE_COUNT)
    near = jointly_valid & (box_separation_lower_bound(expanded_a, expanded_b) <= tolerance_meters)
    contact = torch.zeros_like(jointly_valid)
    near_idx = torch.where(near)
    if near_idx[0].numel():
        contact[near_idx] = signed_box_distance(expanded_a[near_idx], expanded_b[near_idx]) <= tolerance_meters
    return contact


def _first_ego_collision(boxes, state_valid, ego_indices, candidate_mask, tolerance_meters, scenario_idx):
    candidate_indices = torch.where(candidate_mask[scenario_idx])[0]
    if candidate_indices.numel() == 0:
        return None, -1
    ego_idx = int(ego_indices[scenario_idx].item())
    scenario_boxes = boxes[scenario_idx]
    jointly_valid = state_valid[scenario_idx, candidate_indices] & state_valid[scenario_idx, ego_idx][None]
    overlapping = _box_contact_mask(
        scenario_boxes[candidate_indices],
        scenario_boxes[ego_idx][None],
        jointly_valid,
        tolerance_meters,
    )
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
    """Flag every background pair that touches at any timestep.

    Almost all pairs are far apart at almost every timestep, so a circumscribed-radius
    bound rejects them before the exact distance runs. The bound never rejects a true
    contact, which keeps the signature identical to the all-exact result.
    """
    pair_count = pair_indices.shape[1]
    signature = torch.zeros(pair_count, dtype=torch.bool, device=boxes.device)
    for chunk_start in range(0, pair_count, BACKGROUND_COLLISION_PAIR_CHUNK_SIZE):
        chunk_pairs = pair_indices[:, chunk_start : chunk_start + BACKGROUND_COLLISION_PAIR_CHUNK_SIZE]
        scenario_row, left_indices, right_indices = chunk_pairs
        jointly_valid = state_valid[scenario_row, left_indices] & state_valid[scenario_row, right_indices]
        contact = _box_contact_mask(
            boxes[scenario_row, left_indices],
            boxes[scenario_row, right_indices],
            jointly_valid,
            tolerance_meters,
        )
        signature[chunk_start : chunk_start + chunk_pairs.shape[1]] = torch.any(contact, dim=-1)
    return signature


def _candidate_offroad_signature(boxes, state_valid, drivable_area_rasters, candidate_mask):
    signature = torch.zeros((*boxes.shape[:3], 4), dtype=torch.bool, device=boxes.device)
    for scenario_idx, raster in enumerate(drivable_area_rasters):
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
    ego_idx = int(ego_indices[scenario_idx].item())
    distances = (
        (boxes[scenario_idx, candidate_indices, :, :2] - boxes[scenario_idx, ego_idx, None, :, :2]).square().sum(dim=-1)
    )
    jointly_valid = state_valid[scenario_idx, candidate_indices] & state_valid[scenario_idx, ego_idx, None]
    jointly_valid_counts = jointly_valid.sum(dim=-1)
    summed_distances = torch.where(jointly_valid, distances, torch.zeros_like(distances)).sum(dim=-1)
    mean_distances = (summed_distances / jointly_valid_counts.clamp_min(1)).detach()
    mean_distances = mean_distances.masked_fill(jointly_valid_counts == 0, math.inf)
    if not torch.isfinite(mean_distances).any():
        return -1
    return int(candidate_indices[int(torch.argmin(mean_distances).item())].item())


def _slice_selection(selection, scenario_idx):
    """One scenario's view of a batch selection; every tensor field is batch-major."""
    row = slice(scenario_idx, scenario_idx + 1)
    fields = {name: value[row] if isinstance(value, torch.Tensor) else value for name, value in vars(selection).items()}
    return CandidateSelection(**fields)


def _scenario_background_collision(pair_scenario_row, new_pair_signature, batch_size):
    """Reduce the flat per-pair signature to one flag per scenario."""
    hits = torch.zeros(batch_size, dtype=torch.int64, device=new_pair_signature.device)
    if pair_scenario_row.numel():
        hits.scatter_add_(0, pair_scenario_row, new_pair_signature.to(torch.int64))
    return hits > 0


def optimize_frozen_ego_scenarios(
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
    """Optimize Stage 3 background actions for a whole batch against detached ego rollouts.

    Every scenario shares one Adam loop, which amortizes the per-iteration Python and
    dispatch cost that dominates this workload. Scenarios are independent: a scenario's
    loss depends only on its own actions, so summing the batch loss hands each row
    exactly the gradient it would receive alone. A scenario that hits a stop condition
    is deactivated, contributes no gradient, and has its parameters restored after each
    step, so the survivors evolve exactly as they would on their own.
    """
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

    batch_size = scenario.batch_size
    baseline_actions = inverse.actions[:, :, :horizon_transition_count].detach().clone()
    device = baseline_actions.device
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
    ego_indices = _ego_indices(scenario.ego_mask)
    reference_boxes = _masked_boxes(reference_states, state_valid, scenario.length_meters, scenario.width_meters)
    baseline_offroad_signature = _candidate_offroad_signature(
        reference_boxes, state_valid, scenario.drivable_area_rasters, selection.candidate_mask
    )
    background_pair_indices = _candidate_background_pair_indices(
        scenario,
        state_valid,
        selection.candidate_mask,
    )
    pair_scenario_row = background_pair_indices[0]
    baseline_background_collision_signature = _background_collision_signature(
        reference_boxes,
        state_valid,
        config.collision_distance_tolerance_meters,
        background_pair_indices,
    )
    baseline_pair_hits = torch.zeros(batch_size, dtype=torch.int64, device=device)
    if pair_scenario_row.numel():
        baseline_pair_hits.scatter_add_(0, pair_scenario_row, baseline_background_collision_signature.to(torch.int64))
    baseline_background_collision_pair_count = baseline_pair_hits.tolist()

    maximum_reconstruction_error = []
    initial_saturation = []
    for scenario_idx in range(batch_size):
        residuals = inverse.residual_meters[scenario_idx, :, :horizon_transition_count][
            valid_reconstruction[scenario_idx]
        ]
        maximum_reconstruction_error.append(float(residuals.max().item()) if residuals.numel() else 0.0)
        initial_saturation.append(
            _saturation_fraction(
                baseline_actions, selection.optimized_action_mask, config.action_saturation_tolerance, scenario_idx
            )
        )

    out_of_bounds_rasters = prepare_out_of_bounds_rasters(
        scenario.drivable_area_rasters,
        config.costs,
        device=scenario.logged_state.device,
        dtype=scenario.logged_state.dtype,
    )
    wheelbase_over_time, achievable_curvature = steering_conversion_metadata(
        scenario, selection.optimized_action_mask, device
    )
    in_curvature_space = config.steering_parameterization == STEERING_PARAMETERIZATION_CURVATURE
    steering_update_scale = (
        config.curvature_steering_update_scale if in_curvature_space else config.steering_update_scale
    )
    steering_parameter_limit = (
        achievable_curvature * CURVATURE_PARAMETER_LIMIT_MARGIN if in_curvature_space else baseline_actions.new_ones(())
    )
    baseline_parameter = parameter_from_drive_actions(
        baseline_actions, config.steering_parameterization, wheelbase_over_time
    )
    action_parameter = torch.nn.Parameter(baseline_parameter.clone())
    # A saturated logged action reconstructs to the curvature limit, so projecting before
    # the first forward pass keeps iteration zero inside the box as well.
    with torch.no_grad():
        _project_parameter(action_parameter, steering_parameter_limit)
    optimizer = torch.optim.Adam(
        (action_parameter,),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        foreach=False,
    )
    candidate_mask = selection.candidate_mask
    background_vehicle_mask = candidate_mask

    # A filtered scene never enters the loop; it keeps its baseline actions verbatim.
    active = selection.scene_eligible.clone()
    failure_reason = [None] * batch_size
    for scenario_idx in torch.where(~active)[0].tolist():
        failure_reason[scenario_idx] = "scene_filtered:" + ",".join(selection.scene_reasons_for(scenario_idx))
    jointly_valid = state_valid & state_valid[torch.arange(batch_size, device=device), ego_indices][:, None]
    horizon_has_candidate = (jointly_valid & candidate_mask[..., None]).flatten(1).any(dim=-1)
    for scenario_idx in torch.where(active & ~horizon_has_candidate)[0].tolist():
        failure_reason[scenario_idx] = "no_candidate_in_optimization_horizon"
    active &= horizon_has_candidate
    # The ego-collision cost requires every scenario it sees to own a candidate, so a
    # filtered scene must be kept out of the shared loss rather than merely deactivated.
    eligible_rows = torch.where(active)[0]
    eligible_indices = eligible_rows.tolist()
    eligible_rasters = tuple(out_of_bounds_rasters[scenario_idx] for scenario_idx in eligible_indices)

    best_actions = baseline_actions.clone()
    best_states = reference_states.detach().clone()
    best_total = [math.inf] * batch_size
    best_costs = [None] * batch_size
    best_iteration = [0] * batch_size
    initial_costs = [None] * batch_size
    cost_history = [[] for _ in range(batch_size)]
    collision_timestep = [None] * batch_size
    collision_agent_idx = [-1] * batch_size
    success = [False] * batch_size
    completed_update_count = [0] * batch_size
    no_improvement_count = [0] * batch_size
    background_collision_rejection_count = [0] * batch_size
    offroad_rejection_count = [0] * batch_size
    gradient_norms = [[] for _ in range(batch_size)]
    acceleration_gradient_norms = [[] for _ in range(batch_size)]
    steering_gradient_norms = [[] for _ in range(batch_size)]
    divergence_iterations = [[] for _ in range(batch_size)]

    pbar = tqdm(range(config.iteration_count + 1), desc="Optimizing", leave=False) if show_progress else None
    iterator = pbar if pbar is not None else range(config.iteration_count + 1)

    for iteration in iterator:
        if not bool(active.any()):
            break
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
            compile_dynamics=config.compile_dynamics,
        )
        costs = combined_regents_cost(
            states[eligible_rows],
            state_valid[eligible_rows],
            scenario.length_meters[eligible_rows],
            scenario.width_meters[eligible_rows],
            scenario.ego_mask[eligible_rows],
            candidate_mask[eligible_rows],
            background_vehicle_mask[eligible_rows],
            candidate_mask[eligible_rows],
            eligible_rasters,
            config.costs,
        )
        eligible_finite = torch.isfinite(costs.total)
        for value in (costs.ego_collision, costs.background_collision, costs.drivable_area):
            eligible_finite = eligible_finite & torch.isfinite(value)
        finite_costs = torch.ones(batch_size, dtype=torch.bool, device=active.device)
        finite_costs[eligible_rows] = eligible_finite
        for scenario_idx in torch.where(active & ~finite_costs)[0].tolist():
            failure_reason[scenario_idx] = "nonfinite_loss"
        active = active & finite_costs
        if not bool(active.any()):
            break

        detached_boxes = _masked_boxes(states.detach(), state_valid, scenario.length_meters, scenario.width_meters)
        new_background_pairs = (
            _background_collision_signature(
                detached_boxes,
                state_valid,
                config.collision_distance_tolerance_meters,
                background_pair_indices,
            )
            & ~baseline_background_collision_signature
        )
        background_collision = _scenario_background_collision(
            pair_scenario_row, new_background_pairs, batch_size
        ).tolist()
        offroad_signature = _candidate_offroad_signature(
            detached_boxes, state_valid, scenario.drivable_area_rasters, candidate_mask
        )
        offroad = torch.any((offroad_signature & ~baseline_offroad_signature).flatten(1), dim=1).tolist()
        active_indices = torch.where(active)[0].tolist()

        if pbar is not None and iteration % 10 == 0:
            live_loss = costs.total[active[eligible_rows]]
            pbar.set_postfix(loss=f"{float(live_loss.mean().item()):.4f}", live=len(active_indices))

        stop_now = []
        for scenario_idx in active_indices:
            snapshot = _cost_snapshot(costs, eligible_indices.index(scenario_idx))
            cost_history[scenario_idx].append(snapshot)
            if initial_costs[scenario_idx] is None:
                initial_costs[scenario_idx] = snapshot
            background_collision_rejection_count[scenario_idx] += int(background_collision[scenario_idx])
            offroad_rejection_count[scenario_idx] += int(offroad[scenario_idx])
            iteration_collision_timestep, iteration_collision_agent_idx = _first_ego_collision(
                detached_boxes,
                state_valid,
                ego_indices,
                candidate_mask,
                config.collision_distance_tolerance_meters,
                scenario_idx,
            )
            # Retain the current iterate regardless of regularization violations,
            # matching the released ReGentS return policy. Events remain diagnostics.
            best_actions[scenario_idx] = drive_actions[scenario_idx].detach()
            best_states[scenario_idx] = states[scenario_idx].detach()
            best_costs[scenario_idx] = snapshot
            best_iteration[scenario_idx] = iteration
            improved = snapshot.total < best_total[scenario_idx] - config.early_stop_minimum_improvement
            if improved:
                best_total[scenario_idx] = snapshot.total
                no_improvement_count[scenario_idx] = 0
            elif iteration > 0:
                no_improvement_count[scenario_idx] += 1

            generated_collision = iteration_collision_timestep is not None
            success[scenario_idx] = generated_collision
            collision_timestep[scenario_idx] = iteration_collision_timestep
            collision_agent_idx[scenario_idx] = iteration_collision_agent_idx
            if generated_collision and config.early_stop_on_collision:
                stop_now.append(scenario_idx)
                continue
            if iteration == config.iteration_count:
                stop_now.append(scenario_idx)
                continue
            if (
                config.early_stop_patience_iterations > 0
                and no_improvement_count[scenario_idx] >= config.early_stop_patience_iterations
            ):
                failure_reason[scenario_idx] = "stagnated"
                stop_now.append(scenario_idx)

        if stop_now:
            active = active.clone()
            active[torch.tensor(stop_now, dtype=torch.int64, device=active.device)] = False
        if not bool(active.any()):
            break

        optimizer.zero_grad(set_to_none=True)
        eligible_active = active[eligible_rows]
        torch.where(eligible_active, costs.total, torch.zeros_like(costs.total)).sum().backward()
        if action_parameter.grad is None:
            for scenario_idx in torch.where(active)[0].tolist():
                failure_reason[scenario_idx] = "nonfinite_gradient"
            break
        finite_gradient = torch.isfinite(action_parameter.grad).flatten(1).all(dim=1)
        for scenario_idx in torch.where(active & ~finite_gradient)[0].tolist():
            failure_reason[scenario_idx] = "nonfinite_gradient"
        active = active & finite_gradient
        if not bool(active.any()):
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
        # Divergence cancels the post-Adam update, not the gradient or moments.
        masked_gradient = torch.where(
            selection.optimized_action_mask[..., None], action_parameter.grad, torch.zeros_like(action_parameter.grad)
        )
        # A deactivated row keeps whatever its last live gradient produced; zeroing it by
        # selection (never by multiplication) keeps a NaN from an abandoned scenario out.
        masked_gradient = torch.where(active[:, None, None, None], masked_gradient, torch.zeros_like(masked_gradient))
        action_parameter.grad.copy_(masked_gradient)
        for scenario_idx in torch.where(active)[0].tolist():
            scenario_gradient = masked_gradient[scenario_idx]
            acceleration_gradient_norms[scenario_idx].append(
                float(torch.linalg.vector_norm(scenario_gradient[..., ACTION_ACCELERATION]).item())
            )
            steering_gradient_norms[scenario_idx].append(
                float(torch.linalg.vector_norm(scenario_gradient[..., ACTION_TARGET_STEERING]).item())
            )
            gradient_norms[scenario_idx].append(float(torch.linalg.vector_norm(scenario_gradient).item()))
            if bool(divergent[scenario_idx].any()):
                divergence_iterations[scenario_idx].append(iteration)
            completed_update_count[scenario_idx] += 1

        previous_steering = action_parameter.detach()[..., ACTION_TARGET_STEERING].clone()
        parameter_before_step = action_parameter.detach().clone()
        optimizer.step()
        with torch.no_grad():
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
            # Adam still moves a deactivated row from its own momentum, so hold it here.
            action_parameter.copy_(torch.where(active[:, None, None, None], action_parameter, parameter_before_step))

    frozen_storage_mask = ~selection.optimized_action_mask[..., None].expand_as(best_actions)
    if not torch.equal(best_actions[frozen_storage_mask], baseline_actions[frozen_storage_mask]):
        raise RuntimeError("Optimizer changed a non-candidate or invalid action")

    results = []
    for scenario_idx in range(batch_size):
        row = slice(scenario_idx, scenario_idx + 1)
        if best_costs[scenario_idx] is None:
            scenario_actions = baseline_actions[row].clone()
            scenario_states = reference_states[row].detach().clone()
            final_costs = initial_costs[scenario_idx]
            scenario_best_iteration = 0
            if failure_reason[scenario_idx] is None:
                failure_reason[scenario_idx] = "initial_state_infeasible"
        else:
            scenario_actions = best_actions[row].clone()
            scenario_states = best_states[row].clone()
            final_costs = best_costs[scenario_idx]
            scenario_best_iteration = best_iteration[scenario_idx]
        if not success[scenario_idx] and failure_reason[scenario_idx] is None:
            failure_reason[scenario_idx] = "iteration_limit"

        scenario_boxes = _masked_boxes(
            scenario_states, state_valid[row], scenario.length_meters[row], scenario.width_meters[row]
        )
        loss_adversary_idx = _selected_adversary(
            scenario_boxes, state_valid[row], ego_indices[row], candidate_mask[row], 0
        )
        loss_adversary_id = (
            int(scenario.agent_id[scenario_idx, loss_adversary_idx].item()) if loss_adversary_idx >= 0 else -1
        )
        selected_idx = collision_agent_idx[scenario_idx] if success[scenario_idx] else loss_adversary_idx
        selected_id = int(scenario.agent_id[scenario_idx, selected_idx].item()) if selected_idx >= 0 else -1
        scenario_pairs = background_pair_indices[:, pair_scenario_row == scenario_idx]
        scenario_pairs = torch.stack((torch.zeros_like(scenario_pairs[0]), scenario_pairs[1], scenario_pairs[2]))
        final_pair_signature = _background_collision_signature(
            scenario_boxes,
            state_valid[row],
            config.collision_distance_tolerance_meters,
            scenario_pairs,
        )
        baseline_pair_signature = baseline_background_collision_signature[pair_scenario_row == scenario_idx]
        final_offroad_signature = _candidate_offroad_signature(
            scenario_boxes,
            state_valid[row],
            scenario.drivable_area_rasters[scenario_idx : scenario_idx + 1],
            candidate_mask[row],
        )
        results.append(
            ReGentSOptimizationResult(
                initial_actions=baseline_actions[row].clone(),
                optimized_actions=scenario_actions,
                optimized_action_mask=selection.optimized_action_mask[row],
                optimized_states=scenario_states,
                state_valid=state_valid[row],
                selection=_slice_selection(selection, scenario_idx),
                initial_costs=initial_costs[scenario_idx],
                final_costs=final_costs,
                selected_adversary_idx=selected_idx,
                selected_adversary_id=selected_id,
                ego_collision_loss_adversary_idx=loss_adversary_idx,
                ego_collision_loss_adversary_id=loss_adversary_id,
                gradient_norms=tuple(gradient_norms[scenario_idx]),
                acceleration_gradient_norms=tuple(acceleration_gradient_norms[scenario_idx]),
                steering_gradient_norms=tuple(steering_gradient_norms[scenario_idx]),
                front_divergence_iterations=tuple(divergence_iterations[scenario_idx]),
                initial_action_saturation_fraction=initial_saturation[scenario_idx],
                final_action_saturation_fraction=_saturation_fraction(
                    scenario_actions,
                    selection.optimized_action_mask[row],
                    config.action_saturation_tolerance,
                    0,
                ),
                steering_parameterization=config.steering_parameterization,
                maximum_reconstruction_error_meters=maximum_reconstruction_error[scenario_idx],
                collision_timestep=collision_timestep[scenario_idx],
                iteration_count=completed_update_count[scenario_idx],
                best_iteration=scenario_best_iteration,
                deterministic_seed=deterministic_seed,
                success=success[scenario_idx],
                background_collision=bool(torch.any(final_pair_signature & ~baseline_pair_signature)),
                offroad=bool(torch.any(final_offroad_signature & ~baseline_offroad_signature[row])),
                baseline_background_collision_pair_count=baseline_background_collision_pair_count[scenario_idx],
                background_collision_rejection_count=background_collision_rejection_count[scenario_idx],
                offroad_rejection_count=offroad_rejection_count[scenario_idx],
                failure_reason=failure_reason[scenario_idx],
                frozen_ego_source=frozen_ego.source,
                cost_history=tuple(cost_history[scenario_idx]),
            )
        )
    return tuple(results)


def optimize_frozen_ego_scenario(scenario, frozen_ego=None, config=None, **kwargs):
    """Optimize a single scenario. See `optimize_frozen_ego_scenarios` for the batch form."""
    if scenario.batch_size != 1:
        raise ValueError("optimize_frozen_ego_scenario takes one scenario; use optimize_frozen_ego_scenarios")
    return optimize_frozen_ego_scenarios(scenario, frozen_ego, config, **kwargs)[0]
