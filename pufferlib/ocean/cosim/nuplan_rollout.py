"""Forecast rollout helpers for the nuPlan planner's ego_dynamics=nuplan_tracker.

Every planning step the shadow env is rolled out for the trajectory horizon: the policy drives the ego and the nearest
on-lane vehicles, every other object moves at its observed velocity and the lights hold their state. The ego's
rolled-out states are the trajectory nuPlan's two-stage controller tracks.
"""

import math

import numpy as np

from pufferlib.ocean.cosim.nuplan_bridge import LIGHT_STATE_RED, LIGHT_STATE_YELLOW


STEP_COUNT_TOLERANCE = 1e-9  # 8.1 s / 0.3 s is 27 steps, not 28, despite float rounding


def rollout_step_count(horizon_seconds: float, step_seconds: float) -> int:
    """Rollout steps whose end reaches the trajectory horizon."""
    if not (horizon_seconds > 0 and step_seconds > 0):
        raise ValueError(f"horizon and step must be > 0, got {horizon_seconds} and {step_seconds}")
    return max(1, math.ceil(horizon_seconds / step_seconds - STEP_COUNT_TOLERANCE))


def split_rollout_slots(distance_sq, policy_eligible, policy_slot_count: int):
    """-> (policy_rows, static_rows) of the partner arrays: the nearest policy-eligible partners take the policy slots,
    every other partner a static slot."""
    distance_sq = np.asarray(distance_sq, np.float64)
    eligible_rows = np.flatnonzero(np.asarray(policy_eligible, bool))
    nearest_first = eligible_rows[np.argsort(distance_sq[eligible_rows], kind="stable")]
    policy_rows = np.sort(nearest_first[:policy_slot_count])
    is_static = np.ones(len(distance_sq), bool)
    is_static[policy_rows] = False
    return policy_rows, np.flatnonzero(is_static)


def forecast_light_states(states, yellow_remaining_seconds, elapsed_seconds: float):
    """Light states elapsed_seconds into a forecast: a yellow whose remaining time has run out is red, every other
    light holds its state (nuPlan reports no future phases). -> (states, yellow_remaining_seconds)."""
    states = np.asarray(states, np.int32)
    remaining = np.maximum(np.asarray(yellow_remaining_seconds, np.float32) - np.float32(elapsed_seconds), 0.0)
    expired = (states == LIGHT_STATE_YELLOW) & (remaining <= 0.0)
    forecast_states = np.where(expired, LIGHT_STATE_RED, states).astype(np.int32)
    forecast_remaining = np.where(forecast_states == LIGHT_STATE_YELLOW, remaining, 0.0).astype(np.float32)
    return forecast_states, forecast_remaining
