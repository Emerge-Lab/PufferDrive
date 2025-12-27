"""Metrics computation for WOSAC realism evaluation. Adapted from
https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/metrics.py
"""

import numpy as np
import torch
from typing import Tuple

from pufferlib.ocean.benchmark import kinematic_features, interaction_features, map_metric_features


def _to_tensor(value, dtype, device=None):
    """Utility to convert numpy inputs to tensors on the requested device."""
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value, dtype=dtype)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device)
    return tensor


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


def _reduce_average_with_validity(tensor: np.ndarray, validity: np.ndarray, axis: int = None) -> np.ndarray:
    """Returns the tensor's average, only selecting valid items.

    Args:
        tensor: A float array of any shape.
        validity: A boolean array of the same shape as `tensor`.
        axis: The axis or axes along which to average. If None, averages over all axes.

    Returns:
        A float or array containing the average of the valid elements of `tensor`.
    """
    if tensor.shape != validity.shape:
        raise ValueError(
            f"Shapes of `tensor` and `validity` must be the same. (Actual: {tensor.shape}, {validity.shape})."
        )
    cond_sum = np.sum(np.where(validity, tensor, np.zeros_like(tensor)), axis=axis, keepdims=False)
    valid_sum = np.sum(validity.astype(np.float32), axis=axis, keepdims=False)

    # Safe division:
    safe_valid_sum = np.where(valid_sum == 0, 1, valid_sum)

    return np.where(valid_sum == 0, np.nan, cond_sum / safe_valid_sum)


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
    dpos = kinematic_features.central_diff(np.stack([x, y], axis=0), pad_value=np.nan)
    linear_speed = np.linalg.norm(dpos, ord=2, axis=0) / seconds_per_step
    linear_accel = kinematic_features.central_diff(linear_speed, pad_value=np.nan) / seconds_per_step

    dh_step = kinematic_features._wrap_angle(kinematic_features.central_diff(heading, pad_value=np.nan) * 2) / 2
    dh = dh_step / seconds_per_step
    d2h_step = kinematic_features._wrap_angle(kinematic_features.central_diff(dh_step, pad_value=np.nan) * 2) / 2
    d2h = d2h_step / (seconds_per_step**2)

    return linear_speed, linear_accel, dh, d2h


def compute_kinematic_validity(valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return validity tensors for speeds and accelerations.

    Computes validity using central logical_and: element [i] is valid only if
    both elements [i-1] and [i+1] are valid. Applied once for speeds and twice
    for accelerations.

    Args:
        valid: A boolean array of shape (..., num_steps) containing whether a
            certain object is valid at that step.

    Returns:
        speed_validity: A validity array for speed fields (central_and applied once).
        acceleration_validity: A validity array for acceleration fields (central_and applied twice).
    """
    pad_shape = (*valid.shape[:-1], 1)
    pad_tensor = np.full(pad_shape, False)
    speed_validity = np.concatenate([pad_tensor, np.logical_and(valid[..., 2:], valid[..., :-2]), pad_tensor], axis=-1)

    pad_tensor = np.full(pad_shape, False)
    acceleration_validity = np.concatenate(
        [pad_tensor, np.logical_and(speed_validity[..., 2:], speed_validity[..., :-2]), pad_tensor], axis=-1
    )

    return speed_validity, acceleration_validity


def compute_interaction_features(
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    scenario_ids: np.ndarray,
    agent_length: np.ndarray,
    agent_width: np.ndarray,
    eval_mask: np.ndarray,
    device: torch.device,
    valid: np.ndarray | None = None,
    corner_rounding_factor: float = 0.7,
    seconds_per_step: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes distance to nearest object for each agent, grouped by scenario.

    Args:
        x: Shape (num_agents, num_rollouts, num_steps)
        y: Shape (num_agents, num_rollouts, num_steps)
        heading: Shape (num_agents, num_rollouts, num_steps)
        scenario_ids: Shape (num_agents, 1)
        agent_length: Shape (num_agents,)
        agent_width: Shape (num_agents,)
        eval_mask: Shape (num_agents,) - boolean mask for evaluated agents
        valid: Shape (num_agents, num_rollouts, num_steps), optional

    Returns:
        Tuple of:
            - Distance to nearest object, shape (num_eval_agents, num_rollouts, num_steps)
            - Collision indicator per step, shape (num_eval_agents, num_rollouts, num_steps)
            - Time to collision, shape (num_eval_agents, num_rollouts, num_steps)
    """
    x_t = _to_tensor(x, torch.float32, device=device)
    y_t = _to_tensor(y, torch.float32, device=device)
    heading_t = _to_tensor(heading, torch.float32, device=device)
    agent_length_t = _to_tensor(agent_length, torch.float32, device=device)
    agent_width_t = _to_tensor(agent_width, torch.float32, device=device)

    num_agents = x_t.shape[0]
    num_eval_agents = int(np.sum(eval_mask))
    num_rollouts = x_t.shape[1]
    num_steps = x_t.shape[2]

    if valid is None:
        valid_t = torch.ones((num_agents, num_rollouts, num_steps), dtype=torch.bool, device=x_t.device)
    else:
        valid_t = _to_tensor(valid, torch.bool, device=x_t.device)

    length_broadcast = agent_length_t.unsqueeze(-1).expand(num_agents, num_rollouts)
    width_broadcast = agent_width_t.unsqueeze(-1).expand(num_agents, num_rollouts)

    result_distances = np.full(
        (num_eval_agents, num_rollouts, num_steps), interaction_features.EXTREMELY_LARGE_DISTANCE, dtype=np.float32
    )
    result_collisions = np.full((num_eval_agents, num_rollouts, num_steps), False, dtype=bool)
    result_ttc = np.full(
        (num_eval_agents, num_rollouts, num_steps), interaction_features.MAXIMUM_TIME_TO_COLLISION, dtype=np.float32
    )

    unique_scenarios = np.unique(scenario_ids)

    eval_indices = np.where(eval_mask)[0]
    eval_to_result = {idx: i for i, idx in enumerate(eval_indices)}

    for scenario_id in unique_scenarios:
        scenario_mask_np = scenario_ids[:, 0] == scenario_id
        agent_indices = np.where(scenario_mask_np)[0]
        if agent_indices.size == 0:
            continue

        scenario_mask = torch.as_tensor(scenario_mask_np, dtype=torch.bool, device=x_t.device)
        scenario_x = x_t[scenario_mask]
        scenario_y = y_t[scenario_mask]
        episode_length = length_broadcast[scenario_mask]
        scenario_width = width_broadcast[scenario_mask]
        scenario_heading = heading_t[scenario_mask]
        scenario_valid = valid_t[scenario_mask]

        scenario_eval_mask_np = eval_mask[scenario_mask_np]
        scenario_eval_mask = torch.as_tensor(scenario_eval_mask_np, dtype=torch.bool, device=x_t.device)

        distances_to_objects = interaction_features.compute_distance_to_nearest_object(
            center_x=scenario_x,
            center_y=scenario_y,
            length=episode_length,
            width=scenario_width,
            heading=scenario_heading,
            valid=scenario_valid,
            corner_rounding_factor=corner_rounding_factor,
            evaluated_object_mask=scenario_eval_mask,
        )

        is_colliding_per_step = distances_to_objects < interaction_features.COLLISION_DISTANCE_THRESHOLD

        times_to_collision = interaction_features.compute_time_to_collision(
            center_x=scenario_x,
            center_y=scenario_y,
            length=episode_length,
            width=scenario_width,
            heading=scenario_heading,
            valid=scenario_valid,
            seconds_per_step=seconds_per_step,
            evaluated_object_mask=scenario_eval_mask,
        )

        eval_agents_in_scenario = agent_indices[scenario_eval_mask_np]
        result_indices = [eval_to_result[idx] for idx in eval_agents_in_scenario]

        distances_np = distances_to_objects.cpu().numpy()
        collisions_np = is_colliding_per_step.cpu().numpy()
        ttc_np = times_to_collision.cpu().numpy()

        result_distances[result_indices] = distances_np
        result_collisions[result_indices] = collisions_np
        result_ttc[result_indices] = ttc_np

    return result_distances, result_collisions, result_ttc


def compute_map_features(
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    scenario_ids: np.ndarray,
    agent_length: np.ndarray,
    agent_width: np.ndarray,
    road_edge_polylines: dict,
    device: torch.device,
    valid: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes distance to road edge and offroad indication for each agent.

    Args:
        x: Shape (num_agents, num_rollouts, num_steps)
        y: Shape (num_agents, num_rollouts, num_steps)
        heading: Shape (num_agents, num_rollouts, num_steps)
        scenario_ids: Shape (num_agents, 1)
        agent_length: Shape (num_agents,)
        agent_width: Shape (num_agents,)
        road_edge_polylines: Dictionary with polyline data
        valid: Shape (num_agents, num_rollouts, num_steps), optional

    Returns:
        Tuple of:
            - Distance to road edge, shape (num_agents, num_rollouts, num_steps)
            - Offroad indication per step, shape (num_agents, num_rollouts, num_steps)
    """
    x_t = _to_tensor(x, torch.float32, device=device)
    y_t = _to_tensor(y, torch.float32, device=device)
    heading_t = _to_tensor(heading, torch.float32, device=device)
    agent_length_t = _to_tensor(agent_length, torch.float32, device=device)
    agent_width_t = _to_tensor(agent_width, torch.float32, device=device)
    num_agents = x_t.shape[0]
    num_rollouts = x_t.shape[1]
    num_steps = x_t.shape[2]

    if valid is None:
        valid_t = torch.ones((num_agents, num_rollouts, num_steps), dtype=torch.bool, device=device)
    else:
        valid_t = _to_tensor(valid, torch.bool, device=device)

    result_distances = np.zeros((num_agents, num_rollouts, num_steps), dtype=np.float32)
    result_offroad = np.zeros((num_agents, num_rollouts, num_steps), dtype=bool)

    unique_scenarios = np.unique(scenario_ids)

    polyline_boundaries = np.cumsum(np.concatenate([[0], road_edge_polylines["lengths"]]))

    for scenario_id in unique_scenarios:
        agent_mask_np = scenario_ids[:, 0] == scenario_id
        agent_indices = np.where(agent_mask_np)[0]

        if len(agent_indices) == 0:
            continue

        polyline_mask = road_edge_polylines["scenario_id"] == scenario_id
        polyline_indices = np.where(polyline_mask)[0]

        scenario_lengths = road_edge_polylines["lengths"][polyline_mask]

        scenario_x_list = []
        scenario_y_list = []
        for idx in polyline_indices:
            start = polyline_boundaries[idx]
            end = polyline_boundaries[idx + 1]
            scenario_x_list.append(road_edge_polylines["x"][start:end])
            scenario_y_list.append(road_edge_polylines["y"][start:end])

        scenario_polyline_x = torch.as_tensor(np.concatenate(scenario_x_list), dtype=torch.float32, device=x_t.device)
        scenario_polyline_y = torch.as_tensor(np.concatenate(scenario_y_list), dtype=torch.float32, device=x_t.device)
        scenario_lengths_t = torch.as_tensor(scenario_lengths, dtype=torch.int64, device=x_t.device)

        agent_mask = torch.as_tensor(agent_mask_np, dtype=torch.bool, device=x_t.device)
        scenario_x = x_t[agent_mask]
        scenario_y = y_t[agent_mask]
        scenario_heading = heading_t[agent_mask]
        scenario_valid = valid_t[agent_mask]
        scenario_length = agent_length_t[agent_mask]
        scenario_width = agent_width_t[agent_mask]

        for rollout_idx in range(num_rollouts):
            distances = map_metric_features.compute_distance_to_road_edge(
                center_x=scenario_x[:, rollout_idx, :],
                center_y=scenario_y[:, rollout_idx, :],
                length=scenario_length,
                width=scenario_width,
                heading=scenario_heading[:, rollout_idx, :],
                valid=scenario_valid[:, rollout_idx, :],
                polyline_x=scenario_polyline_x,
                polyline_y=scenario_polyline_y,
                polyline_lengths=scenario_lengths_t,
            )

            distances_np = distances.cpu().numpy()
            result_distances[agent_mask_np, rollout_idx, :] = distances_np
            result_offroad[agent_mask_np, rollout_idx, :] = (
                distances_np > map_metric_features.OFFROAD_DISTANCE_THRESHOLD
            )

    return result_distances, result_offroad
