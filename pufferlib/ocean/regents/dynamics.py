"""Differentiable Torch implementation of PufferDrive classic dynamics."""

import math

import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents.state import (
    STATE_FEATURE_COUNT,
    STATE_HEADING,
    STATE_SPEED,
    STATE_STEERING,
    STATE_X,
    STATE_Y,
)


ACTION_FEATURE_COUNT = 2
ACTION_ACCELERATION = 0
ACTION_TARGET_STEERING = 1

BRAKING_ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED = float(
    binding.REGENTS_BRAKING_ACCELERATION_METERS_PER_SECOND_SQUARED
)
FORWARD_ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED = float(
    binding.REGENTS_FORWARD_ACCELERATION_METERS_PER_SECOND_SQUARED
)
TARGET_STEERING_SCALE_RADIANS = float(binding.STEERING_VALUES[-1])
STEERING_LIMIT_RADIAeNS = float(binding.STEERING_LIMIT_RADIANS)
STEERING_RATE_LIMIT_RADIANS_PER_SECOND = float(binding.REGENTS_STEERING_RATE_LIMIT_RADIANS_PER_SECOND)
MAX_BACKWARD_SPEED_MPS = float(binding.MAX_BACKWARD_SPEED_MPS)
REAR_AXLE_RATIO = float(binding.REAR_AXLE_RATIO)
WHEELBASE_LENGTH_RATIO = float(binding.WHEELBASE_LENGTH_RATIO)


def acceleration_from_normalized_action(normalized_acceleration):
    """Convert a ReGentS action to its asymmetric physical acceleration."""
    return torch.where(
        normalized_acceleration < 0,
        normalized_acceleration * BRAKING_ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED,
        normalized_acceleration * FORWARD_ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED,
    )


def injection_wheelbase_by_transition(logged_length_meters, fallback_wheelbase_meters, transition_active):
    """Match C's wheelbase refresh at the start of each injected validity run."""
    transition_count = transition_active.shape[-1]
    current_wheelbase = fallback_wheelbase_meters
    previous_active = torch.zeros_like(transition_active[..., 0])
    wheelbase_by_transition = []
    for timestep in range(transition_count):
        run_start = transition_active[..., timestep] & ~previous_active
        logged_wheelbase = WHEELBASE_LENGTH_RATIO * logged_length_meters[..., timestep]
        current_wheelbase = torch.where(run_start, logged_wheelbase, current_wheelbase)
        wheelbase_by_transition.append(current_wheelbase)
        previous_active = transition_active[..., timestep]
    return torch.stack(wheelbase_by_transition, dim=-1)


def _wrap_heading_like_c(heading):
    two_pi = heading.new_tensor(2.0 * math.pi)
    wrapped = torch.fmod(heading, two_pi)
    above_pi = wrapped.to(torch.float64) > math.pi
    below_negative_pi = wrapped.to(torch.float64) < -math.pi
    wrapped = torch.where(above_pi, wrapped - two_pi, wrapped)
    return torch.where(below_negative_pi, wrapped + two_pi, wrapped)


def classic_step(state, action, wheelbase_meters, maximum_speed_mps, dt_seconds):
    """Advance one classic-dynamics step without mutating any input."""
    acceleration = acceleration_from_normalized_action(action[..., ACTION_ACCELERATION])
    target_steering = action[..., ACTION_TARGET_STEERING] * TARGET_STEERING_SCALE_RADIANS

    previous_steering = state[..., STATE_STEERING]
    maximum_steering_delta = STEERING_RATE_LIMIT_RADIANS_PER_SECOND * dt_seconds
    steering_delta = torch.clamp(
        target_steering - previous_steering,
        min=-maximum_steering_delta,
        max=maximum_steering_delta,
    )
    steering = torch.clamp(
        previous_steering + steering_delta,
        min=-STEERING_LIMIT_RADIANS,
        max=STEERING_LIMIT_RADIANS,
    )

    speed = state[..., STATE_SPEED] + acceleration * dt_seconds
    speed = torch.clamp(speed, min=MAX_BACKWARD_SPEED_MPS)
    speed = torch.minimum(speed, maximum_speed_mps)
    beta = torch.atan(REAR_AXLE_RATIO * torch.tan(steering))
    yaw_rate = speed * torch.cos(beta) * torch.tan(steering) / wheelbase_meters

    previous_heading = state[..., STATE_HEADING]
    velocity_x = speed * torch.cos(previous_heading + beta)
    velocity_y = speed * torch.sin(previous_heading + beta)
    x_meters = state[..., STATE_X] + velocity_x * dt_seconds
    y_meters = state[..., STATE_Y] + velocity_y * dt_seconds
    heading = _wrap_heading_like_c(previous_heading + yaw_rate * dt_seconds)

    speed_squared = velocity_x * velocity_x + velocity_y * velocity_y
    safe_speed_squared = torch.clamp_min(speed_squared, torch.finfo(speed_squared.dtype).tiny)
    speed_magnitude = torch.sqrt(safe_speed_squared)
    speed_magnitude = torch.where(speed_squared > 0.0, speed_magnitude, torch.zeros_like(speed_magnitude))
    velocity_heading_projection = velocity_x * torch.cos(heading) + velocity_y * torch.sin(heading)
    c_signed_speed = torch.where(torch.signbit(velocity_heading_projection), -speed_magnitude, speed_magnitude)
    # C recovers signed speed as sqrt(vx^2 + vy^2) resigned onto the heading, which is the
    # identity on speed. Carrying the C value forward keeps parity; routing the derivative
    # through speed keeps backpropagation alive at an exactly stationary agent.
    signed_speed = speed + (c_signed_speed - speed).detach()
    return torch.stack((x_meters, y_meters, heading, signed_speed, steering), dim=-1)


def classic_rollout(initial_state, actions, transition_valid, wheelbase_meters, maximum_speed_mps, dt_seconds):
    """Roll out masked actions and return ``[..., time, 5]`` states.

    A false transition keeps the preceding state byte-for-byte. Callers must
    split trajectories at validity gaps rather than treating a later valid
    transition as a reconstruction of a newly appearing actor.
    """
    if not isinstance(actions, torch.Tensor) or actions.ndim < 2 or actions.shape[-1] != ACTION_FEATURE_COUNT:
        raise ValueError("actions must have shape [..., time - 1, 2]")
    if actions.shape[-2] < 1:
        raise ValueError("classic_rollout requires at least one transition")
    expected_state_shape = (*actions.shape[:-2], STATE_FEATURE_COUNT)
    if initial_state.shape != expected_state_shape:
        raise ValueError(f"initial_state must have shape {expected_state_shape}")
    expected_transition_shape = actions.shape[:-1]
    if not isinstance(transition_valid, torch.Tensor) or transition_valid.shape != expected_transition_shape:
        raise ValueError(f"transition_valid must have shape {expected_transition_shape}")
    if transition_valid.dtype != torch.bool or transition_valid.device != initial_state.device:
        raise ValueError("transition_valid must be bool on the state device")

    current_state = initial_state
    rollout_states = [current_state]
    for timestep in range(actions.shape[-2]):
        proposed_state = classic_step(
            current_state,
            actions[..., timestep, :],
            wheelbase_meters,
            maximum_speed_mps,
            dt_seconds,
        )
        current_state = torch.where(transition_valid[..., timestep, None], proposed_state, current_state)
        rollout_states.append(current_state)
    return torch.stack(rollout_states, dim=-2)
