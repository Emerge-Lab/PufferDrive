"""Error metrics for trajectory approximation evaluation."""

import numpy as np
from .trajectory_utils import wrap_angle


def compute_position_error(predicted_traj, ground_truth_traj, valid_mask):
    """
    Compute position errors (ADE, FDE, percentiles).

    Args:
        predicted_traj: Dict with {x, y, ...}
        ground_truth_traj: Dict with {x, y, ...}
        valid_mask: Boolean array indicating valid timesteps
    Returns:
        dict with {'ade', 'fde', 'p50', 'p90', 'p95', 'max'}
    """
    dx = predicted_traj['x'] - ground_truth_traj['x']
    dy = predicted_traj['y'] - ground_truth_traj['y']
    errors = np.sqrt(dx**2 + dy**2)[valid_mask]

    if len(errors) == 0:
        return {
            'ade': 0.0,
            'fde': 0.0,
            'p50': 0.0,
            'p90': 0.0,
            'p95': 0.0,
            'max': 0.0,
        }

    return {
        'ade': float(np.mean(errors)),  # Average Displacement Error
        'fde': float(errors[-1]),        # Final Displacement Error
        'p50': float(np.percentile(errors, 50)),
        'p90': float(np.percentile(errors, 90)),
        'p95': float(np.percentile(errors, 95)),
        'max': float(np.max(errors)),
    }


def compute_velocity_error(predicted_traj, ground_truth_traj, valid_mask):
    """
    Compute velocity magnitude and direction errors.

    Args:
        predicted_traj: Dict with {vx, vy, ...}
        ground_truth_traj: Dict with {vx, vy, ...}
        valid_mask: Boolean array indicating valid timesteps
    Returns:
        dict with speed and direction error statistics
    """
    # Speed (magnitude) error
    pred_speed = np.sqrt(predicted_traj['vx']**2 + predicted_traj['vy']**2)
    gt_speed = np.sqrt(ground_truth_traj['vx']**2 + ground_truth_traj['vy']**2)
    speed_error = np.abs(pred_speed - gt_speed)[valid_mask]

    # Velocity direction error (angle between velocity vectors)
    pred_vel_angle = np.arctan2(predicted_traj['vy'], predicted_traj['vx'])
    gt_vel_angle = np.arctan2(ground_truth_traj['vy'], ground_truth_traj['vx'])
    direction_error = np.abs(wrap_angle(pred_vel_angle - gt_vel_angle))[valid_mask]

    if len(speed_error) == 0:
        return {
            'speed_mae': 0.0,
            'speed_max': 0.0,
            'speed_p90': 0.0,
            'direction_mae': 0.0,
            'direction_max': 0.0,
            'direction_p90': 0.0,
        }

    return {
        'speed_mae': float(np.mean(speed_error)),
        'speed_max': float(np.max(speed_error)),
        'speed_p90': float(np.percentile(speed_error, 90)),
        'direction_mae': float(np.mean(direction_error) * 180 / np.pi),  # degrees
        'direction_max': float(np.max(direction_error) * 180 / np.pi),
        'direction_p90': float(np.percentile(direction_error, 90) * 180 / np.pi),
    }


def compute_heading_error(predicted_traj, ground_truth_traj, valid_mask):
    """
    Compute heading errors in degrees.

    Args:
        predicted_traj: Dict with {heading, ...}
        ground_truth_traj: Dict with {heading, ...}
        valid_mask: Boolean array indicating valid timesteps
    Returns:
        dict with heading error statistics
    """
    heading_diff = wrap_angle(
        predicted_traj['heading'] - ground_truth_traj['heading']
    )
    errors = np.abs(heading_diff)[valid_mask] * 180 / np.pi  # Convert to degrees

    if len(errors) == 0:
        return {
            'mae': 0.0,
            'p50': 0.0,
            'p90': 0.0,
            'max': 0.0,
        }

    return {
        'mae': float(np.mean(errors)),
        'p50': float(np.percentile(errors, 50)),
        'p90': float(np.percentile(errors, 90)),
        'max': float(np.max(errors)),
    }


def compute_all_metrics(predicted_traj, ground_truth_traj, valid_mask):
    """
    Compute all metrics and return unified dict.

    Args:
        predicted_traj: Dict with {x, y, heading, vx, vy, ...}
        ground_truth_traj: Dict with {x, y, heading, vx, vy, valid, ...}
        valid_mask: Boolean array indicating valid timesteps
    Returns:
        dict with {'position': {...}, 'velocity': {...}, 'heading': {...}}
    """
    return {
        'position': compute_position_error(predicted_traj, ground_truth_traj, valid_mask),
        'velocity': compute_velocity_error(predicted_traj, ground_truth_traj, valid_mask),
        'heading': compute_heading_error(predicted_traj, ground_truth_traj, valid_mask),
    }


def aggregate_metrics(all_metrics, model_name=''):
    """
    Aggregate metrics across multiple trajectory segments.

    Args:
        all_metrics: List of metric dicts from compute_all_metrics
        model_name: Optional name for the model (for display)
    Returns:
        dict with aggregated statistics
    """
    if len(all_metrics) == 0:
        return {}

    # Extract position metrics
    position_ade = [m['position']['ade'] for m in all_metrics]
    position_fde = [m['position']['fde'] for m in all_metrics]
    position_max = [m['position']['max'] for m in all_metrics]

    # Extract velocity metrics
    speed_mae = [m['velocity']['speed_mae'] for m in all_metrics]
    speed_max = [m['velocity']['speed_max'] for m in all_metrics]

    # Extract heading metrics
    heading_mae = [m['heading']['mae'] for m in all_metrics]
    heading_max = [m['heading']['max'] for m in all_metrics]

    aggregated = {
        'model': model_name,
        'num_trajectories': len(all_metrics),
        'position': {
            'ade_mean': float(np.mean(position_ade)),
            'ade_std': float(np.std(position_ade)),
            'ade_median': float(np.median(position_ade)),
            'ade_p90': float(np.percentile(position_ade, 90)),
            'fde_mean': float(np.mean(position_fde)),
            'fde_std': float(np.std(position_fde)),
            'max_mean': float(np.mean(position_max)),
            'max_p90': float(np.percentile(position_max, 90)),
        },
        'velocity': {
            'speed_mae_mean': float(np.mean(speed_mae)),
            'speed_mae_std': float(np.std(speed_mae)),
            'speed_mae_median': float(np.median(speed_mae)),
            'speed_max_mean': float(np.mean(speed_max)),
            'speed_max_p90': float(np.percentile(speed_max, 90)),
        },
        'heading': {
            'mae_mean': float(np.mean(heading_mae)),
            'mae_std': float(np.std(heading_mae)),
            'mae_median': float(np.median(heading_mae)),
            'max_mean': float(np.mean(heading_max)),
            'max_p90': float(np.percentile(heading_max, 90)),
        },
    }

    return aggregated


def format_metrics_summary(aggregated_metrics):
    """
    Format aggregated metrics as a readable string.

    Args:
        aggregated_metrics: Dict from aggregate_metrics
    Returns:
        Formatted string
    """
    if not aggregated_metrics:
        return "No metrics available"

    lines = []
    model_name = aggregated_metrics.get('model', 'Unknown')
    num_traj = aggregated_metrics.get('num_trajectories', 0)

    lines.append(f"\n{'='*60}")
    lines.append(f"Model: {model_name} ({num_traj} trajectory segments)")
    lines.append(f"{'='*60}")

    # Position errors
    pos = aggregated_metrics['position']
    lines.append(f"\nPosition Errors (meters):")
    lines.append(f"  ADE: {pos['ade_mean']:.3f} ± {pos['ade_std']:.3f} (median: {pos['ade_median']:.3f}, p90: {pos['ade_p90']:.3f})")
    lines.append(f"  FDE: {pos['fde_mean']:.3f} ± {pos['fde_std']:.3f}")
    lines.append(f"  Max: {pos['max_mean']:.3f} (p90: {pos['max_p90']:.3f})")

    # Velocity errors
    vel = aggregated_metrics['velocity']
    lines.append(f"\nVelocity Errors:")
    lines.append(f"  Speed MAE: {vel['speed_mae_mean']:.3f} ± {vel['speed_mae_std']:.3f} m/s (median: {vel['speed_mae_median']:.3f})")
    lines.append(f"  Speed Max: {vel['speed_max_mean']:.3f} m/s (p90: {vel['speed_max_p90']:.3f})")

    # Heading errors
    hdg = aggregated_metrics['heading']
    lines.append(f"\nHeading Errors (degrees):")
    lines.append(f"  MAE: {hdg['mae_mean']:.2f} ± {hdg['mae_std']:.2f}° (median: {hdg['mae_median']:.2f}°)")
    lines.append(f"  Max: {hdg['max_mean']:.2f}° (p90: {hdg['max_p90']:.2f}°)")

    return '\n'.join(lines)
