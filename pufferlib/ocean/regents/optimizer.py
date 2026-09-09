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
    _classic_step,
    injection_wheelbase_by_transition,
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
    STATE_X,
    STATE_Y,
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
DEFAULT_COLLISION_DISTANCE_TOLERANCE_METERS = 0.0
# Zero keeps the caller's captured ego for the whole run; a positive value re-rolls the ego
# against the current adversary plan after that many Adam updates.
DEFAULT_EGO_REFRESH_INTERVAL = 0
MAXIMUM_STEERING_UPDATE_SCALE = 10.0
BACKGROUND_COLLISION_PAIR_CHUNK_SIZE = 4096
# Drive pins the ego to stable agent row zero, and `regents_get_states` preserves that order.
STABLE_EGO_AGENT_IDX = 0
REGENTS_EGO_ACTION_FEATURE_COUNT = 2

# Steering is optimized in the reference's path-curvature space, converted to the
# simulator's normalized target wheel angle at the boundary. This is what keeps Adam
# exploring the same manifold ReGentS does; see `waymax_actions`.
STEERING_PARAMETERIZATION_CURVATURE = "curvature"
DEFAULT_STEERING_UPDATE_SCALE = 0.5

# A curvature sitting exactly at full lock round trips, in float32, to a wheel angle one
# ulp above the action box, which the C injector rejects. Project just inside the limit.
CURVATURE_PARAMETER_LIMIT_MARGIN = 1.0 - 1e-6


# Every ego controller ReGentS can freeze. C owns the ego whatever the controller is;
# 'logged_fixture' is the pinned-trajectory test source.
EGO_TRAJECTORY_SOURCES = ("c_idm", "c_corridor_idm", "c_pdm", "c_replay", "c_policy", "logged_fixture")

# The ego controllers ReGentS can freeze and refresh. Backgrounds must stay replay.
SUPPORTED_SDC_CONTROLLERS = (
    binding.CONTROLLER_IDM,
    binding.CONTROLLER_CORRIDOR_IDM,
    binding.CONTROLLER_PDM,
    binding.CONTROLLER_REPLAY,
    binding.CONTROLLER_POLICY,
)

# Maps a C controller constant onto the EGO_TRAJECTORY_SOURCES name recorded on an artifact.
EGO_TRAJECTORY_SOURCE_BY_CONTROLLER = {
    binding.CONTROLLER_IDM: "c_idm",
    binding.CONTROLLER_CORRIDOR_IDM: "c_corridor_idm",
    binding.CONTROLLER_PDM: "c_pdm",
    binding.CONTROLLER_REPLAY: "c_replay",
    binding.CONTROLLER_POLICY: "c_policy",
}

SUPPORTED_SDC_CONTROLLER_NAMES = "'idm', 'corridor_idm', 'pdm', 'replay', or 'policy'"


def ego_trajectory_source(sdc_controller):
    """Name the C ego controller a captured trajectory came from."""
    source = EGO_TRAJECTORY_SOURCE_BY_CONTROLLER.get(sdc_controller)
    if source is None:
        raise ValueError(f"ReGentS requires sdc_controller={SUPPORTED_SDC_CONTROLLER_NAMES}")
    return source


@dataclass(frozen=True)
class FrozenEgoTrajectory:
    """Detached ego states captured from a C ego controller or a logged test fixture."""

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
        if self.source not in EGO_TRAJECTORY_SOURCES:
            raise ValueError(f"Frozen ego source must be one of {EGO_TRAJECTORY_SOURCES}")
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
    collision_distance_tolerance_meters: float = DEFAULT_COLLISION_DISTANCE_TOLERANCE_METERS
    early_stop_on_collision: bool = True
    ego_refresh_interval: int = DEFAULT_EGO_REFRESH_INTERVAL

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
        if not math.isfinite(self.steering_update_scale):
            raise ValueError("steering_update_scale must be finite")
        if not 0.0 <= self.steering_update_scale <= MAXIMUM_STEERING_UPDATE_SCALE:
            raise ValueError(f"steering_update_scale must be in [0, {MAXIMUM_STEERING_UPDATE_SCALE}]")
        if not math.isfinite(self.collision_distance_tolerance_meters):
            raise ValueError("collision_distance_tolerance_meters must be finite")
        if self.collision_distance_tolerance_meters < 0.0:
            raise ValueError("collision_distance_tolerance_meters must be non-negative")
        if not isinstance(self.early_stop_on_collision, bool):
            raise TypeError("early_stop_on_collision must be a boolean")
        if not isinstance(self.ego_refresh_interval, int) or isinstance(self.ego_refresh_interval, bool):
            raise TypeError("ego_refresh_interval must be an integer")
        if self.ego_refresh_interval < 0:
            raise ValueError("ego_refresh_interval must be non-negative")


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
    evaluations, which no longer prevent returning an iterate. `iteration_count` counts
    Adam updates only, while `cost_history` also carries the extra evaluation passes a
    stale-ego collision recheck adds.
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
    ego_refresh_count: int = 0
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
    ego_action_fn=None,
):
    """Reset one Drive scenario and capture its C ego rollout under any ego controller.

    Backgrounds remain under the Drive configuration's replay controller. The
    caller owns the Drive instance and remains responsible for closing it.
    """
    if not isinstance(transition_count, int) or transition_count < 1:
        raise ValueError("transition_count must be a positive integer")
    if drive.sdc_controller not in SUPPORTED_SDC_CONTROLLERS:
        raise ValueError(f"Frozen ego capture requires sdc_controller={SUPPORTED_SDC_CONTROLLER_NAMES}")
    if (drive.sdc_controller == binding.CONTROLLER_POLICY) != (ego_action_fn is not None):
        raise ValueError("A policy ego requires an ego action provider, and no other controller accepts one")
    if drive.non_sdc_controller != binding.CONTROLLER_REPLAY:
        raise ValueError("Frozen ego capture requires non_sdc_controller='replay'")
    if drive.simulation_mode != binding.SIMULATION_MODE_REPLAY:
        raise ValueError("Frozen ego capture requires simulation_mode='replay'")
    if drive.num_envs != 1:
        raise ValueError("Stage 5 frozen ego capture supports exactly one scenario")
    if drive.resample_frequency > 0 and transition_count >= drive.resample_frequency:
        raise ValueError("transition_count must end before Drive resamples the scenario")

    observations, _ = drive.reset(seed=seed)
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
        step_actions = neutral_actions if ego_action_fn is None else ego_action_fn(observations)
        observations = drive.step(step_actions)[0]
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
    source = ego_trajectory_source(drive.sdc_controller)
    frozen_ego = FrozenEgoTrajectory(
        state=torch.from_numpy(np.ascontiguousarray(np.stack(states)[None, ...])),
        valid=torch.from_numpy(np.ascontiguousarray(np.asarray(validity, dtype=np.bool_)[None, ...])),
        scenario_ids=scenario.scenario_ids,
        source=source,
    )
    return scenario, frozen_ego


def steering_conversion_metadata(scenario, optimized_action_mask, device):
    """Return per-transition wheelbase and the curvature a full wheel angle reaches."""
    wheelbase_meters = scenario.wheelbase_meters.to(device)
    optimized_agent = optimized_action_mask.any(dim=-1)
    usable = torch.isfinite(wheelbase_meters) & (wheelbase_meters > 0.0)
    if torch.any(optimized_agent & ~usable):
        raise ValueError("an optimized agent has a non-positive or non-finite wheelbase")
    # Frozen agents never reach the converter; the placeholder only keeps the math finite.
    safe_wheelbase = torch.where(optimized_agent, wheelbase_meters, torch.ones_like(wheelbase_meters))
    wheelbase_over_time = injection_wheelbase_by_transition(
        scenario.logged_length_meters[..., : optimized_action_mask.shape[-1]].to(device),
        safe_wheelbase,
        optimized_action_mask.to(device),
    )
    return wheelbase_over_time, NORMALIZED_CURVATURE_LIMIT / wheelbase_over_time


def parameter_from_drive_actions(drive_actions, wheelbase_over_time):
    """Express normalized simulator actions in the optimizer's curvature space."""
    curvature_per_meter = curvature_from_target_steering(
        drive_actions[..., ACTION_TARGET_STEERING] * TARGET_STEERING_SCALE_RADIANS,
        wheelbase_over_time,
    )
    return torch.stack((drive_actions[..., ACTION_ACCELERATION], curvature_per_meter), dim=-1)


def drive_actions_from_parameter(action_parameter, wheelbase_over_time):
    """Express optimizer curvature parameters as the normalized actions the simulator consumes."""
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
    ego_indices = _ego_indices(scenario.ego_mask).to(actions.device)
    batch_indices = torch.arange(scenario.batch_size, device=actions.device)
    reference_state = reference_state.clone()
    reference_state[batch_indices, ego_indices] = frozen_ego.state
    state_valid = scenario.state_valid[:, :, : horizon_transition_count + 1].to(actions.device).clone()
    state_valid[batch_indices, ego_indices] = frozen_ego.valid

    # Only candidate rows are ever integrated, so the sequential loop carries those rows
    # alone instead of every agent in the scenario. Every other agent keeps its
    # reference verbatim.
    candidate_batch_rows, candidate_agent_rows = torch.where(optimized_agent_mask)
    if candidate_batch_rows.numel() == 0:
        return reference_state, state_valid
    candidate_rows = (candidate_batch_rows, candidate_agent_rows)
    candidate_active = optimized_transition_valid[candidate_rows]
    candidate_reference = reference_state[candidate_rows]
    candidate_actions = actions[candidate_rows]
    candidate_wheelbase = injection_wheelbase_by_transition(
        scenario.logged_length_meters[:, :, :horizon_transition_count].to(actions.device)[candidate_rows],
        scenario.wheelbase_meters.to(actions.device)[candidate_rows],
        candidate_active,
    )
    candidate_maximum_speed = scenario.maximum_speed_mps.to(actions.device)[candidate_rows]
    # A run start re-seeds integration from the reference; timestep zero already starts there.
    run_start = torch.zeros_like(candidate_active)
    run_start[:, 1:] = candidate_active[:, 1:] & ~candidate_active[:, :-1]

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
        proposed_state = _classic_step(
            step_state,
            candidate_actions[:, timestep],
            candidate_wheelbase[:, timestep],
            candidate_maximum_speed,
            scenario.dt_seconds,
        )
        current_state = torch.where(active, proposed_state, candidate_reference[:, timestep + 1])
        rollout.append(current_state)
    states = reference_state.clone()
    states[candidate_rows] = torch.stack(rollout, dim=1)
    return states, state_valid


def _reconstruction_drift_meters(scenario, inverse, baseline_actions, frozen_ego, horizon_transition_count):
    """Peak distance between an agent's baseline reconstruction and its own log.

    Every potential adversary is integrated, not just the survivors of the other
    filters, so the measurement does not depend on the selection it feeds.
    """
    reconstruction, _ = _compose_rollout(
        scenario,
        inverse,
        baseline_actions,
        frozen_ego,
        horizon_transition_count,
        scenario.candidate_adversary_mask.to(baseline_actions.device),
    )
    logged_position = scenario.logged_state[:, :, : horizon_transition_count + 1, STATE_X : STATE_Y + 1]
    valid = scenario.state_valid[:, :, : horizon_transition_count + 1]
    offset = reconstruction[..., STATE_X : STATE_Y + 1] - logged_position.to(reconstruction.device)
    return (
        torch.linalg.vector_norm(offset, dim=-1).masked_fill(~valid.to(reconstruction.device), 0.0).max(dim=-1).values
    )


def _ego_indices(ego_mask):
    """Per-scenario ego column; ScenarioBatch guarantees exactly one ego per row."""
    return torch.argmax(ego_mask.to(torch.int64), dim=-1)


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


def _first_ego_collision(boxes, state_valid, ego_idx, candidate_mask, tolerance_meters):
    candidate_indices = torch.where(candidate_mask[0])[0]
    if candidate_indices.numel() == 0:
        return None, -1
    scenario_boxes = boxes[0]
    jointly_valid = state_valid[0, candidate_indices] & state_valid[0, ego_idx][None]
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
    """Flat (left, right) columns over every candidate-involving background pair.

    The layout is fixed for a run, which lets a signature be compared elementwise
    against its baseline.
    """
    background_idx = torch.where(scenario.vehicle_mask[0] & ~scenario.ego_mask[0])[0]
    if background_idx.numel() < 2:
        return torch.empty((2, 0), dtype=torch.int64, device=scenario.logged_state.device)
    local_pairs = torch.triu_indices(
        background_idx.numel(), background_idx.numel(), offset=1, device=scenario.logged_state.device
    )
    pair_indices = background_idx[local_pairs]
    candidate_involved = candidate_mask[0, pair_indices[0]] | candidate_mask[0, pair_indices[1]]
    jointly_valid = state_valid[0, pair_indices[0]] & state_valid[0, pair_indices[1]]
    return pair_indices[:, candidate_involved & torch.any(jointly_valid, dim=-1)]


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
        left_indices, right_indices = chunk_pairs
        jointly_valid = state_valid[0, left_indices] & state_valid[0, right_indices]
        contact = _box_contact_mask(boxes[0, left_indices], boxes[0, right_indices], jointly_valid, tolerance_meters)
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


def _cost_snapshot(costs):
    return CostSnapshot(
        ego_collision=float(costs.ego_collision[0].detach().item()),
        background_collision=float(costs.background_collision[0].detach().item()),
        drivable_area=float(costs.drivable_area[0].detach().item()),
        total=float(costs.total[0].detach().item()),
        background_collision_first_agent_idx=int(costs.background_collision_first_agent_idx[0].item()),
        background_collision_second_agent_idx=int(costs.background_collision_second_agent_idx[0].item()),
        background_collision_timestep_idx=int(costs.background_collision_timestep_idx[0].item()),
        background_collision_signed_distance_meters=float(costs.background_collision_signed_distance_meters[0].item()),
        background_collision_truncated=bool(costs.background_collision_truncated[0].item()),
    )


def _finite_cost(costs):
    """Reject an iterate whose loss or any component has gone non-finite."""
    return all(
        bool(torch.isfinite(value).all())
        for value in (costs.total, costs.ego_collision, costs.background_collision, costs.drivable_area)
    )


def _selected_adversary(boxes, state_valid, ego_idx, candidate_mask):
    candidate_indices = torch.where(candidate_mask[0])[0]
    if candidate_indices.numel() == 0:
        return -1
    distances = (boxes[0, candidate_indices, :, :2] - boxes[0, ego_idx, None, :, :2]).square().sum(dim=-1)
    jointly_valid = state_valid[0, candidate_indices] & state_valid[0, ego_idx, None]
    jointly_valid_counts = jointly_valid.sum(dim=-1)
    summed_distances = torch.where(jointly_valid, distances, torch.zeros_like(distances)).sum(dim=-1)
    mean_distances = (summed_distances / jointly_valid_counts.clamp_min(1)).detach()
    mean_distances = mean_distances.masked_fill(jointly_valid_counts == 0, math.inf)
    if not torch.isfinite(mean_distances).any():
        return -1
    return int(candidate_indices[int(torch.argmin(mean_distances).item())].item())


def _refreshed_frozen_ego(ego_rollout_fn, drive_actions, optimized_action_mask, previous_frozen_ego):
    """Re-roll the reactive ego against the current adversary plan.

    The callback owns the simulator; this holds it to the contract the optimization
    horizon needs, because a wrong-shaped or foreign ego would silently mis-align the loss.
    """
    refreshed = ego_rollout_fn(drive_actions.detach(), optimized_action_mask)
    if not isinstance(refreshed, FrozenEgoTrajectory):
        raise TypeError("ego_rollout_fn must return a FrozenEgoTrajectory")
    if refreshed.state.shape != previous_frozen_ego.state.shape:
        raise ValueError("ego_rollout_fn returned a trajectory of the wrong shape")
    if refreshed.scenario_ids != previous_frozen_ego.scenario_ids:
        raise ValueError("ego_rollout_fn returned a different scenario")
    if refreshed.state.device != previous_frozen_ego.state.device:
        raise ValueError("ego_rollout_fn returned a trajectory on a different device")
    return refreshed


def optimize_frozen_ego_scenario(
    scenario,
    frozen_ego=None,
    config=None,
    *,
    deterministic_seed=0,
    horizon_transition_count=None,
    show_progress=True,
    inverse_dynamics=None,
    ego_rollout_fn=None,
):
    """Optimize Stage 3 background actions against a detached ego rollout.

    Adam moves the candidate adversaries' actions only; the ego replays a detached
    trajectory, so the loss depends on nothing the optimizer cannot change. With
    `config.ego_refresh_interval` set, `ego_rollout_fn` re-rolls that trajectory against
    the current adversary plan every that many updates, which is what makes the ego
    reactive without ever placing it in the gradient path. The loop stops on the first
    generated ego collision confirmed against a fresh ego, on a non-finite loss or
    gradient, or at the iteration limit, and returns the iterate it stopped on.
    """
    if scenario.batch_size != 1:
        raise ValueError("optimize_frozen_ego_scenario takes a single-scenario batch")
    if config is None:
        config = ReGentSOptimizationConfig()
    if config.ego_refresh_interval > 0 and ego_rollout_fn is None:
        raise ValueError("ego_refresh_interval requires an ego_rollout_fn")
    horizon_transition_count = _validate_optimization_inputs(
        scenario, frozen_ego, config, deterministic_seed, horizon_transition_count
    )
    if inverse_dynamics is None:
        inverse = estimate_expert_actions(scenario, horizon_transition_count=horizon_transition_count)
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
    reconstruction_drift = _reconstruction_drift_meters(
        scenario, inverse, baseline_actions, frozen_ego, horizon_transition_count
    )
    selection = select_adversary_candidates(
        scenario,
        config.filter,
        horizon_transition_count=horizon_transition_count,
        inverse_dynamics=inverse,
        reconstruction_drift_meters=reconstruction_drift,
    )
    candidate_mask = selection.candidate_mask
    reference_states, state_valid = _compose_rollout(
        scenario, inverse, baseline_actions, frozen_ego, horizon_transition_count, candidate_mask
    )
    ego_idx = int(_ego_indices(scenario.ego_mask)[0].item())
    reference_boxes = _masked_boxes(reference_states, state_valid, scenario.length_meters, scenario.width_meters)
    baseline_offroad_signature = _candidate_offroad_signature(
        reference_boxes, state_valid, scenario.drivable_area_rasters, candidate_mask
    )
    background_pair_indices = _candidate_background_pair_indices(scenario, state_valid, candidate_mask)
    baseline_background_pairs = _background_collision_signature(
        reference_boxes, state_valid, config.collision_distance_tolerance_meters, background_pair_indices
    )
    baseline_background_collision_pair_count = int(baseline_background_pairs.sum().item())

    out_of_bounds_rasters = prepare_out_of_bounds_rasters(
        scenario.drivable_area_rasters,
        config.costs,
        device=scenario.logged_state.device,
        dtype=scenario.logged_state.dtype,
    )
    wheelbase_over_time, achievable_curvature = steering_conversion_metadata(
        scenario, selection.optimized_action_mask, baseline_actions.device
    )
    steering_parameter_limit = achievable_curvature * CURVATURE_PARAMETER_LIMIT_MARGIN
    baseline_parameter = parameter_from_drive_actions(baseline_actions, wheelbase_over_time)
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

    # A filtered scene never enters the loop; it keeps its baseline actions verbatim.
    failure_reason = None
    if not bool(selection.scene_eligible[0]):
        failure_reason = "scene_filtered:" + ",".join(selection.scene_reasons_for(0))
    else:
        jointly_valid = state_valid & state_valid[:, ego_idx][:, None]
        if not bool((jointly_valid & candidate_mask[..., None]).any()):
            failure_reason = "no_candidate_in_optimization_horizon"

    current_actions = baseline_actions.clone()
    current_states = reference_states.detach().clone()
    current_costs = None
    initial_costs = None
    cost_history = []
    collision_timestep = None
    collision_agent_idx = -1
    success = False
    completed_update_count = 0
    last_iteration = 0
    background_collision_rejection_count = 0
    offroad_rejection_count = 0
    ego_refresh_count = 0
    updates_since_ego_refresh = 0
    ego_refresh_due = False
    # Without a refresh the caller's ego is the only ego there is, so it never goes stale.
    ego_is_fresh = True

    pbar = None
    if failure_reason is None and show_progress:
        pbar = tqdm(total=config.iteration_count, desc="Optimizing", leave=False)
    # A collision recheck costs one extra evaluation, so the pass budget is twice the
    # update budget plus the final evaluation.
    maximum_evaluation_count = 0 if failure_reason is not None else 2 * config.iteration_count + 1
    evaluation_idx = 0

    while evaluation_idx < maximum_evaluation_count:
        evaluation_idx += 1
        # Frozen entries take the baseline verbatim so a curvature round trip cannot
        # perturb an action the optimizer is not allowed to change.
        drive_actions = torch.where(
            selection.optimized_action_mask[..., None],
            drive_actions_from_parameter(action_parameter, wheelbase_over_time),
            baseline_actions,
        )
        if ego_refresh_due:
            frozen_ego = _refreshed_frozen_ego(
                ego_rollout_fn, drive_actions, selection.optimized_action_mask, frozen_ego
            )
            ego_refresh_count += 1
            updates_since_ego_refresh = 0
            ego_refresh_due = False
            ego_is_fresh = True
        states, state_valid = _compose_rollout(
            scenario,
            inverse,
            drive_actions,
            frozen_ego,
            horizon_transition_count,
            candidate_mask,
        )
        costs = combined_regents_cost(
            states,
            state_valid,
            scenario.length_meters,
            scenario.width_meters,
            scenario.ego_mask,
            candidate_mask,
            candidate_mask,
            candidate_mask,
            out_of_bounds_rasters,
            config.costs,
        )
        if not _finite_cost(costs):
            failure_reason = "nonfinite_loss"
            break

        detached_boxes = _masked_boxes(states.detach(), state_valid, scenario.length_meters, scenario.width_meters)
        new_background_pairs = (
            _background_collision_signature(
                detached_boxes, state_valid, config.collision_distance_tolerance_meters, background_pair_indices
            )
            & ~baseline_background_pairs
        )
        offroad_signature = _candidate_offroad_signature(
            detached_boxes, state_valid, scenario.drivable_area_rasters, candidate_mask
        )
        background_collision_rejection_count += int(bool(new_background_pairs.any()))
        offroad_rejection_count += int(bool((offroad_signature & ~baseline_offroad_signature).any()))

        snapshot = _cost_snapshot(costs)
        cost_history.append(snapshot)
        if initial_costs is None:
            initial_costs = snapshot
        if pbar is not None and completed_update_count % 10 == 0:
            pbar.set_postfix(loss=f"{snapshot.total:.4f}")

        collision_timestep, collision_agent_idx = _first_ego_collision(
            detached_boxes, state_valid, ego_idx, candidate_mask, config.collision_distance_tolerance_meters
        )
        # A stale ego has not answered the updates that produced this contact, so a
        # contact that would end the run is unproven: re-roll the ego and re-evaluate
        # this same iterate. Non-terminal contacts stay diagnostics and cost no rollout.
        terminal_collision = collision_timestep is not None and (
            config.early_stop_on_collision or completed_update_count == config.iteration_count
        )
        if terminal_collision and not ego_is_fresh:
            ego_refresh_due = True
            continue
        # Retain the current iterate regardless of regularization violations,
        # matching the released ReGentS return policy. Events remain diagnostics.
        current_actions = drive_actions.detach().clone()
        current_states = states.detach().clone()
        current_costs = snapshot
        last_iteration = completed_update_count
        success = collision_timestep is not None
        if success and config.early_stop_on_collision:
            break
        if completed_update_count == config.iteration_count:
            break

        optimizer.zero_grad(set_to_none=True)
        costs.total.sum().backward()
        if action_parameter.grad is None or not bool(torch.isfinite(action_parameter.grad).all()):
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
        # Divergence cancels the post-Adam update, not the gradient or moments.
        masked_gradient = torch.where(
            selection.optimized_action_mask[..., None], action_parameter.grad, torch.zeros_like(action_parameter.grad)
        )
        action_parameter.grad.copy_(masked_gradient)
        completed_update_count += 1

        previous_steering = action_parameter.detach()[..., ACTION_TARGET_STEERING].clone()
        optimizer.step()
        with torch.no_grad():
            action_parameter.copy_(
                torch.where(selection.optimized_action_mask[..., None], action_parameter, baseline_parameter)
            )
            damped_steering = previous_steering + config.steering_update_scale * (
                action_parameter[..., ACTION_TARGET_STEERING] - previous_steering
            )
            action_parameter[..., ACTION_TARGET_STEERING] = torch.where(
                divergent[..., None], previous_steering, damped_steering
            )
            # A scale above one extrapolates past the Adam step and can leave the box.
            _project_parameter(action_parameter, steering_parameter_limit)

        updates_since_ego_refresh += 1
        if config.ego_refresh_interval > 0:
            ego_is_fresh = False
            ego_refresh_due = updates_since_ego_refresh >= config.ego_refresh_interval
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    frozen_storage_mask = ~selection.optimized_action_mask[..., None].expand_as(current_actions)
    if not torch.equal(current_actions[frozen_storage_mask], baseline_actions[frozen_storage_mask]):
        raise RuntimeError("Optimizer changed a non-candidate or invalid action")

    if current_costs is None:
        current_actions = baseline_actions.clone()
        current_states = reference_states.detach().clone()
        last_iteration = 0
        if failure_reason is None:
            failure_reason = "initial_state_infeasible"
    if not success and failure_reason is None:
        failure_reason = "iteration_limit"

    final_boxes = _masked_boxes(current_states, state_valid, scenario.length_meters, scenario.width_meters)
    loss_adversary_idx = _selected_adversary(final_boxes, state_valid, ego_idx, candidate_mask)
    loss_adversary_id = int(scenario.agent_id[0, loss_adversary_idx].item()) if loss_adversary_idx >= 0 else -1
    selected_idx = collision_agent_idx if success else loss_adversary_idx
    selected_id = int(scenario.agent_id[0, selected_idx].item()) if selected_idx >= 0 else -1
    final_background_pairs = _background_collision_signature(
        final_boxes, state_valid, config.collision_distance_tolerance_meters, background_pair_indices
    )
    final_offroad_signature = _candidate_offroad_signature(
        final_boxes, state_valid, scenario.drivable_area_rasters, candidate_mask
    )
    return ReGentSOptimizationResult(
        initial_actions=baseline_actions.clone(),
        optimized_actions=current_actions,
        optimized_action_mask=selection.optimized_action_mask,
        optimized_states=current_states,
        state_valid=state_valid,
        selection=selection,
        initial_costs=initial_costs,
        final_costs=current_costs if current_costs is not None else initial_costs,
        selected_adversary_idx=selected_idx,
        selected_adversary_id=selected_id,
        ego_collision_loss_adversary_idx=loss_adversary_idx,
        ego_collision_loss_adversary_id=loss_adversary_id,
        collision_timestep=collision_timestep,
        iteration_count=completed_update_count,
        best_iteration=last_iteration,
        deterministic_seed=deterministic_seed,
        success=success,
        background_collision=bool((final_background_pairs & ~baseline_background_pairs).any()),
        offroad=bool((final_offroad_signature & ~baseline_offroad_signature).any()),
        baseline_background_collision_pair_count=baseline_background_collision_pair_count,
        background_collision_rejection_count=background_collision_rejection_count,
        offroad_rejection_count=offroad_rejection_count,
        failure_reason=failure_reason,
        frozen_ego_source=frozen_ego.source,
        ego_refresh_count=ego_refresh_count,
        cost_history=tuple(cost_history),
    )
