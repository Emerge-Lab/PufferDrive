"""Frozen-ego ReGentS optimization for one offline scenario."""

import math
from dataclasses import dataclass, field, fields

import torch
from tqdm import tqdm

from pufferlib.ocean.regents.dynamics import (
    ACTION_ACCELERATION,
    ACTION_TARGET_STEERING,
    TARGET_STEERING_SCALE_RADIANS,
    classic_step,
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
from pufferlib.ocean.regents.inverse_dynamics import estimate_expert_actions
from pufferlib.ocean.regents.losses import (
    ReGentSCostConfig,
    _masked_boxes,
    combined_regents_cost,
    mean_candidate_ego_distances,
    prepare_out_of_bounds_raster,
)
from pufferlib.ocean.regents.state import STATE_FEATURE_COUNT, STATE_X, STATE_Y
from pufferlib.ocean.regents.waymax_actions import (
    NORMALIZED_CURVATURE_LIMIT,
    WAYMAX_MAXIMUM_CURVATURE_PER_METER,
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
# Recheck after Adam has moved the rejected plan, with a fixed bound on C replay work.
MAX_C_VERIFICATION_ATTEMPTS = 3
C_VERIFICATION_RETRY_UPDATE_INTERVAL = 10
VERIFICATION_ACCEPT = "accept"
VERIFICATION_RETRY = "retry"
VERIFICATION_REJECT = "reject"

# Steering is optimized in the reference's path-curvature space, converted to the
# simulator's normalized target wheel angle at the boundary. This is what keeps Adam
# exploring the same manifold ReGentS does; see `waymax_actions`.
DEFAULT_STEERING_UPDATE_SCALE = 0.5

# A curvature sitting exactly at full lock round trips, in float32, to a wheel angle one
# ulp above the action box, which the C injector rejects. Project just inside the limit.
CURVATURE_PARAMETER_LIMIT_MARGIN = 1.0 - 1e-6

# The reference bounds curvature at a flat 0.3 1/m for every object, whatever its size. Our
# wheel angle caps a long vehicle below that, and a parameter past what the wheel can reach
# converts to a saturated angle with no gradient, so the box is the tighter of the two.
STEERING_PARAMETER_LIMIT_PER_METER = WAYMAX_MAXIMUM_CURVATURE_PER_METER


@dataclass(frozen=True)
class FrozenEgoTrajectory:
    """Detached ego states captured from a C ego controller or a logged test fixture."""

    state: torch.Tensor
    valid: torch.Tensor
    scenario_id: str
    source: str


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


def _resolve_optimization_horizon(scenario, frozen_ego, horizon_transition_count):
    """Derive the horizon and check the one contract the optimizer cannot assume.

    The frozen ego is captured from C independently of the scenario export, so its
    shape and identity are the only cross-object facts not established by construction.
    """
    maximum_transition_count = scenario.max_time_count - 1
    if horizon_transition_count is None:
        horizon_transition_count = frozen_ego.state.shape[0] - 1 if frozen_ego is not None else maximum_transition_count
    if horizon_transition_count < 1 or horizon_transition_count > maximum_transition_count:
        raise ValueError(f"horizon_transition_count must be in [1, {maximum_transition_count}]")
    if frozen_ego is not None:
        if frozen_ego.state.shape != (horizon_transition_count + 1, STATE_FEATURE_COUNT):
            raise ValueError("Frozen ego state does not match the optimization horizon")
        if frozen_ego.scenario_id != scenario.scenario_id:
            raise ValueError("Frozen ego scenario_id does not match the Scenario")
    return horizon_transition_count


def _compose_rollout(
    scenario,
    inverse,
    actions,
    frozen_ego,
    horizon_transition_count,
    optimized_agent_mask=None,
):
    reference_state = inverse.state_with_estimated_steering[:, : horizon_transition_count + 1].to(actions.device)
    transition_valid = inverse.action_valid[:, :horizon_transition_count].to(actions.device)
    if optimized_agent_mask is None:
        optimized_agent_mask = scenario.candidate_adversary_mask.to(actions.device)
    optimized_transition_valid = transition_valid & optimized_agent_mask[..., None]
    ego_idx = _ego_index(scenario.ego_mask)
    reference_state = reference_state.clone()
    reference_state[ego_idx] = frozen_ego.state
    state_valid = scenario.state_valid[:, : horizon_transition_count + 1].to(actions.device).clone()
    state_valid[ego_idx] = frozen_ego.valid

    # Only candidate rows are ever integrated, so the sequential loop carries those rows
    # alone instead of every agent in the scenario. Every other agent keeps its
    # reference verbatim.
    candidate_rows = torch.where(optimized_agent_mask)[0]
    if candidate_rows.numel() == 0:
        return reference_state, state_valid
    candidate_active = optimized_transition_valid[candidate_rows]
    candidate_reference = reference_state[candidate_rows]
    candidate_actions = actions[candidate_rows]
    candidate_wheelbase = injection_wheelbase_by_transition(
        scenario.logged_length_meters[:, :horizon_transition_count].to(actions.device)[candidate_rows],
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
        proposed_state = classic_step(
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
    logged_position = scenario.logged_state[:, : horizon_transition_count + 1, STATE_X : STATE_Y + 1]
    valid = scenario.state_valid[:, : horizon_transition_count + 1]
    offset = reconstruction[..., STATE_X : STATE_Y + 1] - logged_position.to(reconstruction.device)
    return (
        torch.linalg.vector_norm(offset, dim=-1).masked_fill(~valid.to(reconstruction.device), 0.0).max(dim=-1).values
    )


def _ego_index(ego_mask):
    """The scenario's ego column; the export guarantees exactly one."""
    return int(torch.argmax(ego_mask.to(torch.int64)).item())


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
    candidate_indices = torch.where(candidate_mask)[0]
    if candidate_indices.numel() == 0:
        return None, -1
    jointly_valid = state_valid[candidate_indices] & state_valid[ego_idx][None]
    overlapping = _box_contact_mask(
        boxes[candidate_indices],
        boxes[ego_idx][None],
        jointly_valid,
        tolerance_meters,
    )
    # The earliest overlapping timestep, then its lowest candidate row: argmax on a bool
    # returns the first True, and candidate_indices is ascending.
    overlapping_timestep = overlapping.any(dim=0)
    if not bool(overlapping_timestep.any()):
        return None, -1
    timestep = int(torch.argmax(overlapping_timestep.to(torch.uint8)).item())
    candidate_row = int(torch.argmax(overlapping[:, timestep].to(torch.uint8)).item())
    return timestep, int(candidate_indices[candidate_row].item())


def _candidate_background_pair_indices(scenario, state_valid, candidate_mask):
    """Flat (left, right) columns over every candidate-involving background pair.

    The layout is fixed for a run, which lets a signature be compared elementwise
    against its baseline.
    """
    background_idx = torch.where(scenario.vehicle_mask & ~scenario.ego_mask)[0]
    if background_idx.numel() < 2:
        return torch.empty((2, 0), dtype=torch.int64, device=scenario.logged_state.device)
    local_pairs = torch.triu_indices(
        background_idx.numel(), background_idx.numel(), offset=1, device=scenario.logged_state.device
    )
    pair_indices = background_idx[local_pairs]
    candidate_involved = candidate_mask[pair_indices[0]] | candidate_mask[pair_indices[1]]
    jointly_valid = state_valid[pair_indices[0]] & state_valid[pair_indices[1]]
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
        jointly_valid = state_valid[left_indices] & state_valid[right_indices]
        contact = _box_contact_mask(boxes[left_indices], boxes[right_indices], jointly_valid, tolerance_meters)
        signature[chunk_start : chunk_start + chunk_pairs.shape[1]] = torch.any(contact, dim=-1)
    return signature


def _candidate_offroad_signature(boxes, state_valid, drivable_area_raster, candidate_mask):
    signature = torch.zeros((*boxes.shape[:2], 4), dtype=torch.bool, device=boxes.device)
    candidate_indices = torch.where(candidate_mask)[0]
    if candidate_indices.numel() == 0:
        return signature
    outside = drivable_area_raster.points_off_road(oriented_box_corners(boxes[candidate_indices]))
    # Invalid timesteps carry a placeholder box, so they never enter the signature.
    signature[candidate_indices] = outside & state_valid[candidate_indices][..., None]
    return signature


def _infraction_signatures(boxes, state_valid, scenario, config, candidate_mask, background_pair_indices):
    """Flag the background contacts and off-road corners one set of boxes carries.

    The baseline, every iterate, and the returned iterate are all measured this way, so
    subtracting the baseline's signature from another's isolates what Adam introduced.
    """
    background_pairs = _background_collision_signature(
        boxes, state_valid, config.collision_distance_tolerance_meters, background_pair_indices
    )
    offroad = _candidate_offroad_signature(boxes, state_valid, scenario.drivable_area_raster, candidate_mask)
    return background_pairs, offroad


def _selected_adversary(boxes, state_valid, ego_idx, candidate_mask):
    """The candidate the ego-collision cost is shaped against: nearest on time average."""
    candidate_indices = torch.where(candidate_mask)[0]
    if candidate_indices.numel() == 0:
        return -1
    mean_distances = mean_candidate_ego_distances(boxes, state_valid, ego_idx, candidate_indices).detach()
    if not torch.isfinite(mean_distances).any():
        return -1
    return int(candidate_indices[int(torch.argmin(mean_distances).item())].item())


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
    verification_fn=None,
):
    """Optimize Stage 3 background actions against a detached ego rollout.

    Adam moves the candidate adversaries' actions only; the ego replays a detached
    trajectory, so the loss depends on nothing the optimizer cannot change. With
    `config.ego_refresh_interval` set, `ego_rollout_fn` re-rolls that trajectory against
    the current adversary plan every that many updates, which is what makes the ego
    reactive without ever placing it in the gradient path. The loop stops on the first
    generated ego collision confirmed against a fresh ego, on a non-finite loss or
    gradient, or at the iteration limit, and returns the iterate it stopped on. When
    supplied, `verification_fn` may reject a collision and keep the same Adam state
    moving through the remaining update budget.
    """
    if config is None:
        config = ReGentSOptimizationConfig()
    if config.ego_refresh_interval > 0 and ego_rollout_fn is None:
        raise ValueError("ego_refresh_interval requires an ego_rollout_fn")
    horizon_transition_count = _resolve_optimization_horizon(scenario, frozen_ego, horizon_transition_count)
    if inverse_dynamics is None:
        inverse = estimate_expert_actions(scenario, horizon_transition_count=horizon_transition_count)
    else:
        # A reused estimate may have been computed for a shorter horizon than this run.
        if inverse_dynamics.action_valid.shape[-1] < horizon_transition_count:
            raise ValueError("inverse_dynamics does not cover the optimization horizon")
        inverse = inverse_dynamics
    if frozen_ego is None:
        ego_idx = _ego_index(scenario.ego_mask)
        horizon_slice = slice(None, horizon_transition_count + 1)
        frozen_ego = FrozenEgoTrajectory(
            state=inverse.state_with_estimated_steering[ego_idx, horizon_slice].detach().clone(),
            valid=scenario.state_valid[ego_idx, horizon_slice].detach().clone(),
            scenario_id=scenario.scenario_id,
            source="logged_fixture",
        )

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
    ego_idx = _ego_index(scenario.ego_mask)
    reference_boxes = _masked_boxes(reference_states, state_valid, scenario.length_meters, scenario.width_meters)
    background_pair_indices = _candidate_background_pair_indices(scenario, state_valid, candidate_mask)
    baseline_background_pairs, baseline_offroad_signature = _infraction_signatures(
        reference_boxes, state_valid, scenario, config, candidate_mask, background_pair_indices
    )
    baseline_background_collision_pair_count = int(baseline_background_pairs.sum().item())

    out_of_bounds_raster = prepare_out_of_bounds_raster(
        scenario.drivable_area_raster,
        config.costs,
        device=scenario.logged_state.device,
        dtype=scenario.logged_state.dtype,
    )
    wheelbase_over_time, achievable_curvature = steering_conversion_metadata(
        scenario, selection.optimized_action_mask, baseline_actions.device
    )
    steering_parameter_limit = torch.clamp(
        achievable_curvature * CURVATURE_PARAMETER_LIMIT_MARGIN,
        max=STEERING_PARAMETER_LIMIT_PER_METER,
    )
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
    if not bool(selection.scene_eligible):
        failure_reason = "scene_filtered:" + ",".join(selection.scene_reasons())
    else:
        jointly_valid = state_valid & state_valid[ego_idx][None]
        if not bool((jointly_valid & candidate_mask[:, None]).any()):
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
    verification_count = 0
    last_verification_update = -C_VERIFICATION_RETRY_UPDATE_INTERVAL

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
            refreshed = ego_rollout_fn(drive_actions.detach(), selection.optimized_action_mask)
            # A wrong-shaped ego would silently mis-align the loss against the horizon.
            if refreshed.state.shape != frozen_ego.state.shape:
                raise ValueError("ego_rollout_fn returned a trajectory of the wrong shape")
            frozen_ego = refreshed
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
            out_of_bounds_raster,
            config.costs,
        )
        if not all(
            bool(torch.isfinite(value).all())
            for value in (costs.total, costs.ego_collision, costs.background_collision, costs.drivable_area)
        ):
            failure_reason = "nonfinite_loss"
            break

        detached_boxes = _masked_boxes(states.detach(), state_valid, scenario.length_meters, scenario.width_meters)
        background_pairs, offroad_signature = _infraction_signatures(
            detached_boxes, state_valid, scenario, config, candidate_mask, background_pair_indices
        )
        background_collision_rejection_count += int(bool((background_pairs & ~baseline_background_pairs).any()))
        offroad_rejection_count += int(bool((offroad_signature & ~baseline_offroad_signature).any()))

        snapshot = CostSnapshot(
            **{field.name: getattr(costs, field.name).detach().item() for field in fields(CostSnapshot)}
        )
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
        terminal_collision = success and config.early_stop_on_collision
        if terminal_collision and verification_fn is None:
            break
        verification_due = (
            terminal_collision
            and verification_fn is not None
            and (
                completed_update_count - last_verification_update >= C_VERIFICATION_RETRY_UPDATE_INTERVAL
                or completed_update_count == config.iteration_count
            )
        )
        decision = None
        if verification_due:
            decision = verification_fn(
                baseline_actions,
                current_actions,
                current_states,
                state_valid,
                selection.optimized_action_mask,
            )
            verification_count += 1
            last_verification_update = completed_update_count
        if verification_due and decision not in (VERIFICATION_ACCEPT, VERIFICATION_RETRY, VERIFICATION_REJECT):
            raise ValueError("verification_fn returned an invalid decision")
        if verification_due and (decision != VERIFICATION_RETRY or verification_count == MAX_C_VERIFICATION_ATTEMPTS):
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
    loss_adversary_id = int(scenario.agent_id[loss_adversary_idx].item()) if loss_adversary_idx >= 0 else -1
    selected_idx = collision_agent_idx if success else loss_adversary_idx
    selected_id = int(scenario.agent_id[selected_idx].item()) if selected_idx >= 0 else -1
    final_background_pairs, final_offroad_signature = _infraction_signatures(
        final_boxes, state_valid, scenario, config, candidate_mask, background_pair_indices
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
