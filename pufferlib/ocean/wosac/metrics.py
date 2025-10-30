"""Metrics computation for WOSAC evaluation."""

import numpy as np
from typing import Dict, Tuple


def compute_displacement_error(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    gt_valid: np.ndarray,
) -> np.ndarray:
    """Compute average displacement error (ADE) between simulated and ground truth trajectories.

    Args:
        pred_x, pred_y: Simulated positions, shape (n_agents, n_rollouts, n_steps)
        gt_x, gt_y: Ground truth positions, shape (n_agents, 1, n_steps)
        gt_valid: Valid timesteps, shape (n_agents, 1, n_steps)

    Returns:
        Average displacement error per agent per rollout, shape (n_agents, n_rollouts)
    """

    gt_traj = np.stack([gt_x, gt_y], axis=-1)  # (n_agents, 1, n_steps, 2)
    pred_traj = np.stack([pred_x, pred_y], axis=-1)

    # Compute displacement error for each timestep and every agent and rollout
    displacement = np.linalg.norm(pred_traj - gt_traj, axis=-1)  # (n_agents, n_rollouts, n_steps)

    # Mask invalid timesteps
    displacement = np.where(gt_valid, displacement, 0.0)

    # Aggregate
    valid_count = np.sum(gt_valid, axis=2)  # (n_agents, 1)

    # Compute ADE
    ade_per_rollout = np.sum(displacement, axis=2) / np.maximum(valid_count, 1)  # (n_agents, n_rollouts)

    ade = ade_per_rollout.mean(axis=-1)  # (n_agents,)

    # The rollout with the minimum ADE for each agent
    min_ade = np.min(ade_per_rollout, axis=1)  # (n_agents,)

    return ade, min_ade
