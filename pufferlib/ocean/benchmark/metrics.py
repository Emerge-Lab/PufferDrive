"""Metrics computation for WOSAC realism evaluation. Adapted from
https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/metrics.py
"""

import numpy as np
from typing import Dict, Tuple


def compute_displacement_error(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_valid: np.ndarray,
) -> np.ndarray:
    """Compute average displacement error (ADE) between simulated and ground truth trajectories.

    Args:
        pred_x, pred_y: Simulated positions, shape (n_agents, n_rollouts, n_steps)
        ref_x, ref_y: Ground truth positions, shape (n_agents, 1, n_steps)
        ref_valid: Valid timesteps, shape (n_agents, 1, n_steps)

    Returns:
        Average displacement error per agent per rollout, shape (n_agents, n_rollouts)
    """

    ref_traj = np.stack([ref_x, ref_y], axis=-1)  # (n_agents, 1, n_steps, 2)
    pred_traj = np.stack([pred_x, pred_y], axis=-1)

    # Compute displacement error for each timestep and every agent and rollout
    displacement = np.linalg.norm(pred_traj - ref_traj, axis=-1)  # (n_agents, n_rollouts, n_steps)

    # Mask invalid timesteps
    displacement = np.where(ref_valid, displacement, 0.0)

    # Aggregate
    valid_count = np.sum(ref_valid, axis=2)  # (n_agents, 1)

    # Compute ADE
    ade_per_rollout = np.sum(displacement, axis=2) / np.maximum(valid_count, 1)  # (n_agents, n_rollouts)

    ade = ade_per_rollout.mean(axis=-1)  # (n_agents,)

    # The rollout with the minimum ADE for each agent
    min_ade = np.min(ade_per_rollout, axis=1)  # (n_agents,)

    return ade, min_ade


def central_diff(t: np.ndarray, pad_value: float) -> np.ndarray:
    """Computes the central difference along the last axis.

    This function is used to compute 1st order derivatives (speeds) when called
    once. Calling this function twice is used to compute 2nd order derivatives
    (accelerations) instead.
    This function returns the central difference as
    df(x)/dx = [f(x+h)-f(x-h)] / 2h.

    Args:
        t: A float array of shape [..., steps].
        pad_value: To maintain the original tensor shape, this value is prepended
            once and appended once to the difference.

    Returns:
        An array of shape [..., steps] containing the central differences,
        appropriately prepended and appended with `pad_value` to maintain the
        original shape.
    """
    # Prepare the array containing the value(s) to pad the result with.
    pad_shape = (*t.shape[:-1], 1)
    pad_array = np.full(pad_shape, pad_value)
    diff_t = (t[..., 2:] - t[..., :-2]) / 2
    return np.concatenate([pad_array, diff_t, pad_array], axis=-1)


def central_logical_and(t: np.ndarray, pad_value: bool) -> np.ndarray:
    """Computes the central `logical_and` along the last axis.

    This function is used to compute the validity tensor for 1st and 2nd order
    derivatives using central difference, where element [i] is valid only if
    both elements [i-1] and [i+1] are valid.

    Args:
        t: A bool array of shape [..., steps].
        pad_value: To maintain the original tensor shape, this value is prepended
            once and appended once to the difference.

    Returns:
        An array of shape [..., steps] containing the central `logical_and`,
        appropriately prepended and appended with `pad_value` to maintain the
        original shape.
    """
    # Prepare the array containing the value(s) to pad the result with.
    pad_shape = (*t.shape[:-1], 1)
    pad_array = np.full(pad_shape, pad_value)
    diff_t = np.logical_and(t[..., 2:], t[..., :-2])
    return np.concatenate([pad_array, diff_t, pad_array], axis=-1)


def compute_displacement_error_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray, ref_z: np.ndarray
) -> np.ndarray:
    """Computes displacement error (in x,y,z) w.r.t. a reference trajectory.

    Note: This operation doesn't put any constraint on the shape of the arrays,
    except that they are all consistent with each other, so this can be used
    with any arbitrary array shape.

    Args:
        x: The x-component of the predicted trajectories.
        y: The y-component of the predicted trajectories
        ref_x: The x-component of the reference trajectories.
        ref_y: The y-component of the reference trajectories.

    Returns:
        A float array with the same shape as all the arguments, containing
        the 3D distance between the predicted trajectories and the reference
        trajectories.
    """
    return np.linalg.norm(np.stack([x, y], axis=-1) - np.stack([ref_x, ref_y], axis=-1), ord=2, axis=-1)


def compute_kinematic_features(
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    seconds_per_step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes kinematic features (speeds and accelerations).

    Note: Everything is assumed to be valid, filtering must be done afterwards.
    To maintain the original tensor length, speeds are prepended and appended
    with 1 np.nan, while accelerations with 2 np.nan (since central difference
    invalidated the two extremes).

    Args:
        x: A float array of shape (..., num_steps) containing x coordinates.
        y: A float array of shape (..., num_steps) containing y coordinates.
        heading: A float array of shape (..., num_steps,) containing heading.
        seconds_per_step: The duration (in seconds) of one step. Defaults to 0.1s.

    Returns:
        A tuple containing the following 4 arrays:
            linear_speed: Magnitude of speed in (x, y, z). Shape (..., num_steps).
            linear_acceleration: Linear signed acceleration (changes in linear speed).
                Shape (..., num_steps).
            angular_speed: Angular speed (changes in heading). Shape (..., num_steps).
            angular_acceleration: Angular acceleration (changes in `angular_speed`).
                Shape (..., num_steps).
    """
    # Linear speed and acceleration.
    dpos = central_diff(np.stack([x, y], axis=0), pad_value=np.nan)
    linear_speed = np.linalg.norm(dpos, ord=2, axis=0) / seconds_per_step
    linear_accel = central_diff(linear_speed, pad_value=np.nan) / seconds_per_step
    # Angular speed and acceleration.
    dh_step = _wrap_angle(central_diff(heading, pad_value=np.nan) * 2) / 2
    dh = dh_step / seconds_per_step
    d2h_step = _wrap_angle(central_diff(dh_step, pad_value=np.nan) * 2) / 2
    d2h = d2h_step / (seconds_per_step**2)
    return linear_speed, linear_accel, dh, d2h


def compute_kinematic_validity(valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return validity arrays for speeds and accelerations.

    Since we compute speed and acceleration directly from x/y/z/heading as
    central differences, we need to make sure to properly transform the validity
    arrays to match the new fields. The requirement is for both the steps used to
    compute the difference to be valid in order for the result to be valid.
    This is applied once for speeds and twice for accelerations, following the
    same strategy used to compute the kinematic fields.

    Args:
        valid: A boolean array of shape (..., num_steps) containing whether a
            certain object is valid at that step.

    Returns:
        speed_validity: A validity array that applies to speeds fields, where
            `central_logical_and` is applied once.
        acceleration_validity: A validity array that applies to acceleration
            fields, where `central_logical_and` is applied twice.
    """
    speed_validity = central_logical_and(valid, pad_value=False)
    acceleration_validity = central_logical_and(speed_validity, pad_value=False)
    return speed_validity, acceleration_validity


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wraps angles in the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
