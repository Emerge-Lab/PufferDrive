"""Bounded sequential expert-action estimation for PufferDrive trajectories.

ReGentS initializes background controls with Waymax's expert actor. At each
timestep that actor solves from the current simulated state toward the next
logged state, advances the simulator, and uses the result for the next solve.
This module preserves that feedback construction while inverting PufferDrive's
different forward model: signed speed, updated-speed Euler integration, slip
angle, wheelbase, and rate-limited target wheel angle.
"""

from dataclasses import dataclass

import torch

from pufferlib.ocean.regents.dynamics import (
    ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED,
    MAX_BACKWARD_SPEED_MPS,
    REAR_AXLE_RATIO,
    STEERING_LIMIT_RADIANS,
    STEERING_RATE_LIMIT_RADIANS_PER_SECOND,
    TARGET_STEERING_SCALE_RADIANS,
    classic_step,
    injection_wheelbase_by_transition,
)
from pufferlib.ocean.regents.state import (
    STATE_FEATURE_COUNT,
    STATE_HEADING,
    STATE_SPEED,
    STATE_STEERING,
    STATE_X,
    STATE_Y,
    wrapped_angle_difference,
)


# Match Waymax's noise guard while adapting its inverse to PufferDrive's model.
DEFAULT_LOW_SPEED_THRESHOLD_MPS = 0.6
MODEL_CONSISTENCY_POSITION_TOLERANCE_METERS = 1e-4
MODEL_CONSISTENCY_HEADING_TOLERANCE_RADIANS = 1e-4
MODEL_CONSISTENCY_SPEED_TOLERANCE_MPS = 1e-4
STEERING_SEARCH_GRID_COUNT = 65
STEERING_SEARCH_REFINEMENT_COUNT = 4
STEERING_SEARCH_REFINEMENT_GRID_COUNT = 17


@dataclass(frozen=True)
class InverseDynamicsResult:
    """Actions, inferred steering state, and one-step reconstruction diagnostics."""

    actions: torch.Tensor
    action_valid: torch.Tensor
    state_with_estimated_steering: torch.Tensor
    state_feature_valid: torch.Tensor
    predicted_next_state: torch.Tensor
    position_error_meters: torch.Tensor
    heading_error_radians: torch.Tensor
    speed_error_mps: torch.Tensor
    residual_meters: torch.Tensor
    low_speed_mask: torch.Tensor
    heading_residual_valid: torch.Tensor
    model_consistent: torch.Tensor


def _closest_reachable_acceleration_action(current_speed, target_speed, maximum_speed, dt_seconds):
    maximum_speed_delta = ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED * dt_seconds
    lowest_reachable_speed = torch.clamp(
        current_speed - maximum_speed_delta,
        min=MAX_BACKWARD_SPEED_MPS,
    )
    lowest_reachable_speed = torch.minimum(lowest_reachable_speed, maximum_speed)
    highest_reachable_speed = torch.clamp(
        current_speed + maximum_speed_delta,
        min=MAX_BACKWARD_SPEED_MPS,
    )
    highest_reachable_speed = torch.minimum(highest_reachable_speed, maximum_speed)
    desired_speed = torch.clamp(target_speed, min=lowest_reachable_speed, max=highest_reachable_speed)
    return torch.clamp(
        (desired_speed - current_speed) / maximum_speed_delta,
        min=-1.0,
        max=1.0,
    )


def _heading_inverse_steering(current_state, target_state, reconstructed_speed, wheelbase_meters, dt_seconds):
    heading_delta = wrapped_angle_difference(
        target_state[..., STATE_HEADING],
        current_state[..., STATE_HEADING],
    )
    safe_speed = torch.where(reconstructed_speed != 0, reconstructed_speed, torch.ones_like(reconstructed_speed))
    normalized_yaw_rate = heading_delta * wheelbase_meters / (dt_seconds * safe_speed)
    denominator_squared = 1.0 - (REAR_AXLE_RATIO * normalized_yaw_rate) ** 2
    tangent_steering = normalized_yaw_rate / torch.sqrt(torch.clamp(denominator_squared, min=1e-12))
    steering = torch.atan(tangent_steering)
    domain_saturated = denominator_squared <= 0
    return torch.where(
        domain_saturated,
        torch.copysign(torch.full_like(steering, STEERING_LIMIT_RADIANS), normalized_yaw_rate),
        steering,
    )


def _position_inverse_steering(current_state, target_state, reconstructed_speed):
    displacement_x = target_state[..., STATE_X] - current_state[..., STATE_X]
    displacement_y = target_state[..., STATE_Y] - current_state[..., STATE_Y]
    travel_sign = torch.where(reconstructed_speed < 0, -1.0, 1.0)
    motion_heading = torch.atan2(displacement_y * travel_sign, displacement_x * travel_sign)
    beta = wrapped_angle_difference(motion_heading, current_state[..., STATE_HEADING])
    return torch.atan(torch.tan(beta) / REAR_AXLE_RATIO)


def _steering_objective(predicted_state, target_state, wheelbase_meters, heading_residual_valid):
    position_error_squared = (predicted_state[..., STATE_X] - target_state[..., STATE_X]) ** 2
    position_error_squared += (predicted_state[..., STATE_Y] - target_state[..., STATE_Y]) ** 2
    heading_error = wrapped_angle_difference(
        predicted_state[..., STATE_HEADING],
        target_state[..., STATE_HEADING],
    )
    heading_error_squared_meters = (wheelbase_meters * heading_error) ** 2
    return position_error_squared + torch.where(
        heading_residual_valid,
        heading_error_squared_meters,
        torch.zeros_like(heading_error_squared_meters),
    )


def _select_bounded_steering(
    current_state,
    target_state,
    acceleration_action,
    wheelbase_meters,
    maximum_speed_mps,
    heading_residual_valid,
    observed_next_steering,
    observed_next_steering_valid,
    dt_seconds,
):
    previous_steering = current_state[..., STATE_STEERING]
    maximum_steering_delta = STEERING_RATE_LIMIT_RADIANS_PER_SECOND * dt_seconds
    lowest_steering = torch.clamp(
        previous_steering - maximum_steering_delta,
        min=-STEERING_LIMIT_RADIANS,
        max=STEERING_LIMIT_RADIANS,
    )
    highest_steering = torch.clamp(
        previous_steering + maximum_steering_delta,
        min=-STEERING_LIMIT_RADIANS,
        max=STEERING_LIMIT_RADIANS,
    )
    reconstructed_speed = current_state[..., STATE_SPEED] + (
        acceleration_action * ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED * dt_seconds
    )
    reconstructed_speed = torch.clamp(reconstructed_speed, min=MAX_BACKWARD_SPEED_MPS)
    reconstructed_speed = torch.minimum(reconstructed_speed, maximum_speed_mps)
    heading_steering = torch.clamp(
        _heading_inverse_steering(
            current_state,
            target_state,
            reconstructed_speed,
            wheelbase_meters,
            dt_seconds,
        ),
        min=lowest_steering,
        max=highest_steering,
    )
    position_steering = torch.clamp(
        _position_inverse_steering(current_state, target_state, reconstructed_speed),
        min=lowest_steering,
        max=highest_steering,
    )
    observed_steering = torch.clamp(observed_next_steering, min=lowest_steering, max=highest_steering)
    observed_steering = torch.where(observed_next_steering_valid, observed_steering, heading_steering)

    grid_fraction = torch.linspace(
        0.0,
        1.0,
        STEERING_SEARCH_GRID_COUNT,
        dtype=current_state.dtype,
        device=current_state.device,
    )
    grid_steering = lowest_steering[:, None] + (highest_steering - lowest_steering)[:, None] * grid_fraction
    candidate_steering = torch.cat(
        (
            grid_steering,
            heading_steering[:, None],
            position_steering[:, None],
            observed_steering[:, None],
            torch.clamp(previous_steering, min=lowest_steering, max=highest_steering)[:, None],
        ),
        dim=-1,
    )

    def evaluate(steering):
        candidate_count = steering.shape[-1]
        expanded_state = current_state[:, None, :].expand(-1, candidate_count, -1)
        actions = torch.stack(
            (
                acceleration_action[:, None].expand_as(steering),
                steering / TARGET_STEERING_SCALE_RADIANS,
            ),
            dim=-1,
        )
        predicted = classic_step(
            expanded_state,
            actions,
            wheelbase_meters[:, None].expand_as(steering),
            maximum_speed_mps[:, None].expand_as(steering),
            dt_seconds,
        )
        objective = _steering_objective(
            predicted,
            target_state[:, None, :],
            wheelbase_meters[:, None],
            heading_residual_valid[:, None],
        )
        best_idx = torch.argmin(objective, dim=-1)
        row_idx = torch.arange(steering.shape[0], device=steering.device)
        return steering[row_idx, best_idx]

    best_steering = evaluate(candidate_steering)
    search_half_width = (highest_steering - lowest_steering) / (STEERING_SEARCH_GRID_COUNT - 1)
    refinement_offsets = torch.linspace(
        -1.0,
        1.0,
        STEERING_SEARCH_REFINEMENT_GRID_COUNT,
        dtype=current_state.dtype,
        device=current_state.device,
    )
    for _ in range(STEERING_SEARCH_REFINEMENT_COUNT):
        refinement = best_steering[:, None] + search_half_width[:, None] * refinement_offsets
        refinement = torch.clamp(refinement, min=lowest_steering[:, None], max=highest_steering[:, None])
        best_steering = evaluate(refinement)
        search_half_width /= (STEERING_SEARCH_REFINEMENT_GRID_COUNT - 1) / 2
    return torch.where(heading_residual_valid, best_steering, previous_steering)


def estimate_expert_actions(
    scenario,
    low_speed_threshold_mps=DEFAULT_LOW_SPEED_THRESHOLD_MPS,
    *,
    horizon_transition_count=None,
):
    """Estimate bounded actions by sequentially tracking the next logged state.

    Acceleration first selects the closest reachable logged speed. Steering is
    then selected by a deterministic bounded search minimizing squared position
    error plus wheelbase-scaled squared heading error. Heading is omitted from
    that objective when the updated signed speed is near zero. Each subsequent
    solve starts from the preceding predicted state, as Waymax's expert actor
    does; only a new contiguous validity run is seeded from the log.
    """
    maximum_transition_count = scenario.max_time_count - 1
    if horizon_transition_count is None:
        horizon_transition_count = maximum_transition_count
    if not isinstance(horizon_transition_count, int) or isinstance(horizon_transition_count, bool):
        raise TypeError("horizon_transition_count must be an integer")
    if horizon_transition_count < 1 or horizon_transition_count > maximum_transition_count:
        raise ValueError(f"horizon_transition_count must be in [1, {maximum_transition_count}]")

    logged_state = scenario.logged_state[:, : horizon_transition_count + 1]
    track_count, time_count, _ = logged_state.shape
    transition_count = time_count - 1
    action_valid = scenario.transition_valid[:, :transition_count] & scenario.vehicle_mask[:, None]

    # One scenario, so an agent row is already a track row: no flatten is needed.
    flat_logged_state = logged_state
    flat_feature_valid = scenario.state_feature_valid[:, :time_count]
    flat_action_valid = action_valid
    flat_wheelbase = injection_wheelbase_by_transition(
        scenario.logged_length_meters[:, :transition_count],
        scenario.wheelbase_meters,
        action_valid,
    )
    flat_maximum_speed = scenario.maximum_speed_mps

    actions = torch.zeros((track_count, transition_count, 2), dtype=torch.float32, device=logged_state.device)
    estimated_state = flat_logged_state.clone()
    estimated_feature_valid = flat_feature_valid.clone()
    predicted_next_state = torch.zeros(
        (track_count, transition_count, STATE_FEATURE_COUNT), dtype=torch.float32, device=logged_state.device
    )
    position_error = torch.zeros((track_count, transition_count), dtype=torch.float32, device=logged_state.device)
    heading_error = torch.zeros_like(position_error)
    speed_error = torch.zeros_like(position_error)
    residual = torch.zeros_like(position_error)
    low_speed_mask = torch.zeros_like(flat_action_valid)
    heading_residual_valid = torch.zeros_like(flat_action_valid)
    model_consistent = torch.zeros_like(flat_action_valid)
    reconstructed_state = torch.zeros(
        (track_count, STATE_FEATURE_COUNT), dtype=torch.float32, device=logged_state.device
    )

    for timestep in range(transition_count):
        active_track_idx = torch.where(flat_action_valid[:, timestep])[0]
        if active_track_idx.numel() == 0:
            continue
        if timestep == 0:
            run_start = torch.ones_like(active_track_idx, dtype=torch.bool)
        else:
            run_start = ~flat_action_valid[active_track_idx, timestep - 1]
        logged_current_state = flat_logged_state[active_track_idx, timestep].clone()
        current_steering_observed = flat_feature_valid[active_track_idx, timestep, STATE_STEERING]
        logged_current_state[:, STATE_STEERING] = torch.where(
            current_steering_observed,
            logged_current_state[:, STATE_STEERING],
            torch.zeros_like(logged_current_state[:, STATE_STEERING]),
        )
        current_state = torch.where(
            run_start[:, None],
            logged_current_state,
            reconstructed_state[active_track_idx],
        )
        target_state = flat_logged_state[active_track_idx, timestep + 1]
        wheelbase = flat_wheelbase[active_track_idx, timestep]
        maximum_speed = flat_maximum_speed[active_track_idx]
        acceleration_action = _closest_reachable_acceleration_action(
            current_state[:, STATE_SPEED],
            target_state[:, STATE_SPEED],
            maximum_speed,
            scenario.dt_seconds,
        )
        transition_low_speed = current_state[:, STATE_SPEED].abs() <= low_speed_threshold_mps
        transition_low_speed |= target_state[:, STATE_SPEED].abs() <= low_speed_threshold_mps
        transition_heading_valid = ~transition_low_speed
        selected_steering = _select_bounded_steering(
            current_state,
            target_state,
            acceleration_action,
            wheelbase,
            maximum_speed,
            transition_heading_valid,
            flat_logged_state[active_track_idx, timestep + 1, STATE_STEERING],
            flat_feature_valid[active_track_idx, timestep + 1, STATE_STEERING],
            scenario.dt_seconds,
        )
        selected_action = torch.stack(
            (
                acceleration_action,
                selected_steering / TARGET_STEERING_SCALE_RADIANS,
            ),
            dim=-1,
        )
        predicted_state = classic_step(
            current_state,
            selected_action,
            wheelbase,
            maximum_speed,
            scenario.dt_seconds,
        )

        transition_position_error = torch.linalg.vector_norm(
            predicted_state[:, STATE_X : STATE_Y + 1] - target_state[:, STATE_X : STATE_Y + 1],
            dim=-1,
        )
        transition_heading_error = wrapped_angle_difference(
            predicted_state[:, STATE_HEADING],
            target_state[:, STATE_HEADING],
        ).abs()
        transition_speed_error = (predicted_state[:, STATE_SPEED] - target_state[:, STATE_SPEED]).abs()
        transition_residual = torch.sqrt(
            transition_position_error**2
            + torch.where(
                transition_heading_valid,
                (wheelbase * transition_heading_error) ** 2,
                torch.zeros_like(transition_heading_error),
            )
            + (scenario.dt_seconds * transition_speed_error) ** 2
        )
        transition_consistent = transition_position_error <= MODEL_CONSISTENCY_POSITION_TOLERANCE_METERS
        transition_consistent &= transition_speed_error <= MODEL_CONSISTENCY_SPEED_TOLERANCE_MPS
        transition_consistent &= ~transition_heading_valid | (
            transition_heading_error <= MODEL_CONSISTENCY_HEADING_TOLERANCE_RADIANS
        )

        actions[active_track_idx, timestep] = selected_action
        predicted_next_state[active_track_idx, timestep] = predicted_state
        position_error[active_track_idx, timestep] = transition_position_error
        heading_error[active_track_idx, timestep] = transition_heading_error
        speed_error[active_track_idx, timestep] = transition_speed_error
        residual[active_track_idx, timestep] = transition_residual
        low_speed_mask[active_track_idx, timestep] = transition_low_speed
        heading_residual_valid[active_track_idx, timestep] = transition_heading_valid
        model_consistent[active_track_idx, timestep] = transition_consistent
        estimated_state[active_track_idx, timestep, STATE_STEERING] = current_state[:, STATE_STEERING]
        estimated_state[active_track_idx, timestep + 1, STATE_STEERING] = predicted_state[:, STATE_STEERING]
        estimated_feature_valid[active_track_idx, timestep, STATE_STEERING] = True
        estimated_feature_valid[active_track_idx, timestep + 1, STATE_STEERING] = True
        reconstructed_state[active_track_idx] = predicted_state

    return InverseDynamicsResult(
        actions=actions,
        action_valid=action_valid,
        state_with_estimated_steering=estimated_state,
        state_feature_valid=estimated_feature_valid,
        predicted_next_state=predicted_next_state,
        position_error_meters=position_error,
        heading_error_radians=heading_error,
        speed_error_mps=speed_error,
        residual_meters=residual,
        low_speed_mask=low_speed_mask,
        heading_residual_valid=heading_residual_valid,
        model_consistent=model_consistent,
    )
