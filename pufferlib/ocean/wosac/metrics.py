"""Metrics computation for WOSAC evaluation."""

import numpy as np
from typing import Dict, Tuple


def compute_displacement_error(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    pred_z: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    gt_z: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Compute displacement error between predicted and ground truth trajectories.

    Args:
        pred_x, pred_y, pred_z: Predicted positions, shape (n_agents, n_rollouts, n_steps)
        gt_x, gt_y, gt_z: Ground truth positions, shape (n_agents, 1, n_steps)
        valid: Valid timesteps, shape (n_agents, 1, n_steps)

    Returns:
        Average displacement error per agent per rollout, shape (n_agents, n_rollouts)
    """
    # Compute Euclidean distance
    dx = pred_x - gt_x
    dy = pred_y - gt_y
    dz = pred_z - gt_z
    displacement = np.sqrt(dx**2 + dy**2 + dz**2)

    # Mask invalid timesteps
    displacement = np.where(valid, displacement, 0.0)

    # Compute average over time for each agent and rollout
    valid_count = np.sum(valid, axis=2)  # (n_agents, n_rollouts)
    ade = np.sum(displacement, axis=2) / np.maximum(valid_count, 1)  # (n_agents, n_rollouts)

    return ade  # (n_agents, n_rollouts)


def compute_collision_rate(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    pred_z: np.ndarray,
    length: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    collision_distance_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute collision rate for each agent.

    Args:
        pred_x, pred_y, pred_z: Predicted positions, shape (n_agents, n_rollouts, n_steps)
        length, width, height: Agent dimensions, shape (n_agents,) or (n_agents, 1, 1)
        heading: Agent heading, shape (n_agents, n_rollouts, n_steps)
        valid: Valid timesteps, shape (n_agents, 1, n_steps)
        collision_distance_threshold: Distance threshold for collision detection

    Returns:
        Tuple of:
        - collision_per_step: Boolean mask of collisions per step, shape (n_agents, n_rollouts, n_steps)
        - collision_rate: Collision rate per agent, shape (n_agents,)
    """
    n_agents = pred_x.shape[0]
    n_rollouts = pred_x.shape[1]
    n_steps = pred_x.shape[2]

    collision_per_step = np.zeros((n_agents, n_rollouts, n_steps), dtype=bool)

    # Compute pairwise distances between all agents at each timestep
    for t in range(n_steps):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Compute distance between agent i and j at timestep t
                dx = pred_x[i, :, t] - pred_x[j, :, t]
                dy = pred_y[i, :, t] - pred_y[j, :, t]
                dz = pred_z[i, :, t] - pred_z[j, :, t]

                distance = np.sqrt(dx**2 + dy**2 + dz**2)

                # Simple collision check: distance < threshold
                # For more accuracy, could use OBB or similar
                threshold = collision_distance_threshold
                collided = distance < threshold

                collision_per_step[i, :, t] |= collided
                collision_per_step[j, :, t] |= collided

    # Mask by validity
    collision_per_step = collision_per_step & valid.astype(bool)

    # Compute collision rate per agent (fraction of steps with collision)
    collision_rate = np.sum(collision_per_step, axis=(1, 2)) / (n_rollouts * np.sum(valid, axis=(1, 2)))

    return collision_per_step, collision_rate


def compute_offroad_rate(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    pred_z: np.ndarray,
    valid: np.ndarray,
    offroad_distance_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute offroad rate for each agent.

    For now, this is a placeholder that returns zeros. In practice, you would
    compare against road edge polylines.

    Args:
        pred_x, pred_y, pred_z: Predicted positions, shape (n_agents, n_rollouts, n_steps)
        valid: Valid timesteps, shape (n_agents, 1, n_steps)
        offroad_distance_threshold: Distance threshold for offroad detection

    Returns:
        Tuple of:
        - offroad_per_step: Boolean mask of offroad per step, shape (n_agents, n_rollouts, n_steps)
        - offroad_rate: Offroad rate per agent, shape (n_agents,)
    """
    n_agents = pred_x.shape[0]
    n_rollouts = pred_x.shape[1]
    n_steps = pred_x.shape[2]

    # Placeholder: no agents are offroad (would need map data to compute properly)
    offroad_per_step = np.zeros((n_agents, n_rollouts, n_steps), dtype=bool)
    offroad_rate = np.zeros(n_agents, dtype=np.float32)

    return offroad_per_step, offroad_rate


def compute_metrics(
    simulated_trajectories: Dict,
    ground_truth_trajectories: Dict,
) -> Dict:
    """Compute evaluation metrics comparing simulated and ground truth trajectories.

    Args:
        simulated_trajectories: Dict with keys ['x', 'y', 'z', 'heading', 'id', 'scenario_id', 'valid']
                               Each trajectory has shape (n_agents, n_rollouts, n_steps)
        ground_truth_trajectories: Dict with same keys, shape (n_agents, 1, n_steps)

    Returns:
        Dictionary with scores per scenario_id containing:
        - 'ade': Average displacement error
        - 'collision_rate': Collision rate
        - 'offroad_rate': Offroad rate (currently zeros as placeholder)
    """
    results = {}

    scenario_ids = simulated_trajectories["scenario_id"]

    # Match agents by ID between simulated and ground truth
    sim_ids = simulated_trajectories["id"][:, 0, 0]  # (n_agents,)
    gt_ids = ground_truth_trajectories["id"][:, 0, 0]  # (n_agents_gt,)

    # Find indices where IDs match
    matched_indices = []
    for sim_idx, sim_id in enumerate(sim_ids):
        gt_idx = np.where(gt_ids == sim_id)[0]
        if len(gt_idx) > 0:
            matched_indices.append((sim_idx, gt_idx[0]))

    if not matched_indices:
        # No matched agents, return empty results
        for scenario_id in scenario_ids:
            results[scenario_id] = {
                "ade": None,
                "collision_rate": None,
                "offroad_rate": None,
            }
        return results

    sim_matched_idx, gt_matched_idx = zip(*matched_indices)
    sim_matched_idx = np.array(sim_matched_idx)
    gt_matched_idx = np.array(gt_matched_idx)

    # Extract matched trajectories
    pred_x = simulated_trajectories["x"][sim_matched_idx]
    pred_y = simulated_trajectories["y"][sim_matched_idx]
    pred_z = simulated_trajectories["z"][sim_matched_idx]
    pred_heading = simulated_trajectories["heading"][sim_matched_idx]
    pred_valid = simulated_trajectories["valid"][sim_matched_idx]

    gt_x = ground_truth_trajectories["x"][gt_matched_idx]
    gt_y = ground_truth_trajectories["y"][gt_matched_idx]
    gt_z = ground_truth_trajectories["z"][gt_matched_idx]
    gt_valid = ground_truth_trajectories["valid"][gt_matched_idx]

    # Use ground truth validity
    valid = gt_valid

    # Compute metrics
    ade = compute_displacement_error(pred_x, pred_y, pred_z, gt_x, gt_y, gt_z, valid)
    collision_per_step, collision_rate = compute_collision_rate(
        pred_x,
        pred_y,
        pred_z,
        simulated_trajectories.get("length", np.ones(pred_x.shape[0])),
        simulated_trajectories.get("width", np.ones(pred_x.shape[0])),
        simulated_trajectories.get("height", np.ones(pred_x.shape[0])),
        pred_heading,
        valid,
    )
    offroad_per_step, offroad_rate = compute_offroad_rate(pred_x, pred_y, pred_z, valid)

    # Aggregate per scenario (all matched agents belong to same scenario for now)
    scenario_id = scenario_ids[0]
    results[scenario_id] = {
        "ade": np.mean(ade),
        "ade_per_agent": ade,
        "collision_rate": np.mean(collision_rate),
        "collision_rate_per_agent": collision_rate,
        "offroad_rate": np.mean(offroad_rate),
        "offroad_rate_per_agent": offroad_rate,
    }

    return results
