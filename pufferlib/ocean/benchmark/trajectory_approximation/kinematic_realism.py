"""Population-level kinematic realism metrics for trajectory approximation.

This module computes kinematic distribution similarity between ground truth trajectories
and inverse dynamics approximations, adapted from WOSAC methodology for deterministic systems.
"""

import numpy as np
import configparser
import os
from typing import Dict, List, Tuple


def compute_kinematic_features(
    x: np.ndarray,
    y: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    dt: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Compute kinematic features (speeds and accelerations) from trajectory.

    Uses central difference for derivatives, matching WOSAC implementation.

    Args:
        x: X positions, shape (n_steps,)
        y: Y positions, shape (n_steps,)
        heading: Heading angles, shape (n_steps,)
        valid: Valid mask, shape (n_steps,)
        dt: Time step in seconds (default 0.1s)

    Returns:
        Dict with:
            - linear_speed: Shape (n_steps,)
            - linear_acceleration: Shape (n_steps,)
            - angular_speed: Shape (n_steps,)
            - angular_acceleration: Shape (n_steps,)
            - speed_valid: Validity mask for speeds
            - accel_valid: Validity mask for accelerations
    """
    n_steps = len(x)

    # Compute position changes using central difference
    # Central diff: f'(x) = [f(x+h) - f(x-h)] / 2h
    dx = np.full(n_steps, np.nan)
    dy = np.full(n_steps, np.nan)
    dx[1:-1] = (x[2:] - x[:-2]) / 2
    dy[1:-1] = (y[2:] - y[:-2]) / 2

    # Linear speed
    linear_speed = np.sqrt(dx**2 + dy**2) / dt

    # Linear acceleration (central diff of speed)
    linear_accel = np.full(n_steps, np.nan)
    linear_accel[1:-1] = (linear_speed[2:] - linear_speed[:-2]) / (2 * dt)

    # Angular speed (central diff of heading, with wrapping)
    dheading = np.full(n_steps, np.nan)
    dheading[1:-1] = _wrap_angle((heading[2:] - heading[:-2]) / 2) * 2  # Wrap half-differences
    angular_speed = dheading / dt

    # Angular acceleration (central diff of angular speed)
    angular_accel = np.full(n_steps, np.nan)
    dheading_step = dheading.copy()
    angular_accel[1:-1] = _wrap_angle((dheading_step[2:] - dheading_step[:-2]) / 2) * 2 / (dt**2)

    # Compute validity masks using central logical_and
    # Speed is valid if neighbors are valid
    speed_valid = np.zeros(n_steps, dtype=bool)
    speed_valid[1:-1] = valid[2:] & valid[:-2]

    # Acceleration is valid if speed neighbors are valid
    accel_valid = np.zeros(n_steps, dtype=bool)
    accel_valid[1:-1] = speed_valid[2:] & speed_valid[:-2]

    return {
        'linear_speed': linear_speed,
        'linear_acceleration': linear_accel,
        'angular_speed': angular_speed,
        'angular_acceleration': angular_accel,
        'speed_valid': speed_valid,
        'accel_valid': accel_valid,
    }


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi] range."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def aggregate_features_across_segments(
    all_segment_features: List[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """Flatten and concatenate features from multiple segments into population arrays.

    Args:
        all_segment_features: List of feature dicts from compute_kinematic_features()

    Returns:
        Dict with flattened population-level features:
            - linear_speed: (N_total_valid_steps,)
            - linear_acceleration: (N_total_valid_steps,)
            - angular_speed: (N_total_valid_steps,)
            - angular_acceleration: (N_total_valid_steps,)
    """
    # Collect all valid samples across segments
    linear_speeds = []
    linear_accels = []
    angular_speeds = []
    angular_accels = []

    for features in all_segment_features:
        # Filter by validity and exclude NaNs
        speed_mask = features['speed_valid'] & ~np.isnan(features['linear_speed']) & ~np.isnan(features['angular_speed'])
        accel_mask = features['accel_valid'] & ~np.isnan(features['linear_acceleration']) & ~np.isnan(features['angular_acceleration'])

        linear_speeds.append(features['linear_speed'][speed_mask])
        angular_speeds.append(features['angular_speed'][speed_mask])
        linear_accels.append(features['linear_acceleration'][accel_mask])
        angular_accels.append(features['angular_acceleration'][accel_mask])

    return {
        'linear_speed': np.concatenate(linear_speeds) if linear_speeds else np.array([]),
        'linear_acceleration': np.concatenate(linear_accels) if linear_accels else np.array([]),
        'angular_speed': np.concatenate(angular_speeds) if angular_speeds else np.array([]),
        'angular_acceleration': np.concatenate(angular_accels) if angular_accels else np.array([]),
    }


def compute_histogram_log_likelihood(
    log_samples: np.ndarray,
    sim_samples: np.ndarray,
    min_val: float,
    max_val: float,
    num_bins: int,
    additive_smoothing: float = 0.1,
) -> float:
    """Compute mean log-likelihood of log samples under sim distribution.

    Adapted from WOSAC's histogram_estimate function for population-level evaluation.

    Args:
        log_samples: Samples to evaluate (e.g., ground truth features)
        sim_samples: Samples to build distribution from (e.g., simulated features)
        min_val: Minimum value for histogram bins
        max_val: Maximum value for histogram bins
        num_bins: Number of histogram bins
        additive_smoothing: Laplace smoothing pseudocount

    Returns:
        Mean log-likelihood (scalar)
    """
    if len(log_samples) == 0 or len(sim_samples) == 0:
        return np.nan

    # Clip samples to valid range
    log_samples_clipped = np.clip(log_samples, min_val, max_val)
    sim_samples_clipped = np.clip(sim_samples, min_val, max_val)

    # Create bin edges
    edges = np.linspace(min_val, max_val, num_bins + 1)

    # Build histogram from sim samples
    sim_counts, _ = np.histogram(sim_samples_clipped, bins=edges)

    # Apply smoothing and normalize to probabilities
    sim_counts = sim_counts.astype(float) + additive_smoothing
    sim_probs = sim_counts / sim_counts.sum()

    # Find which bin each log sample belongs to
    log_bins = np.digitize(log_samples_clipped, edges, right=False) - 1
    log_bins = np.clip(log_bins, 0, num_bins - 1)

    # Get log probabilities for each sample
    log_probs = np.log(sim_probs[log_bins])

    # Return mean log-likelihood
    return float(np.mean(log_probs))


def load_wosac_config() -> configparser.ConfigParser:
    """Load WOSAC metric configuration."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'wosac.ini'
    )
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def compute_kinematic_realism_score(
    gt_features: Dict[str, np.ndarray],
    sim_features: Dict[str, np.ndarray],
    config: configparser.ConfigParser = None,
) -> Dict[str, float]:
    """Compute population-level kinematic realism scores.

    Evaluates how likely ground truth features are under simulated distribution,
    using WOSAC histogram-based log-likelihood estimation.

    Args:
        gt_features: Ground truth population features from aggregate_features_across_segments()
        sim_features: Simulated population features from aggregate_features_across_segments()
        config: WOSAC config (if None, loads default wosac.ini)

    Returns:
        Dict with:
            - linear_speed_likelihood: exp(mean_log_likelihood)
            - linear_acceleration_likelihood: exp(mean_log_likelihood)
            - angular_speed_likelihood: exp(mean_log_likelihood)
            - angular_acceleration_likelihood: exp(mean_log_likelihood)
            - kinematic_realism_score: Weighted average of above
    """
    if config is None:
        config = load_wosac_config()

    # Compute log-likelihoods for each metric
    metrics = {}

    # Linear speed
    min_val = config.getfloat('linear_speed', 'histogram.min_val')
    max_val = config.getfloat('linear_speed', 'histogram.max_val')
    num_bins = config.getint('linear_speed', 'histogram.num_bins')
    smoothing = config.getfloat('linear_speed', 'histogram.additive_smoothing_pseudocount')

    log_likelihood = compute_histogram_log_likelihood(
        gt_features['linear_speed'],
        sim_features['linear_speed'],
        min_val, max_val, num_bins, smoothing
    )
    metrics['linear_speed_log_likelihood'] = log_likelihood
    metrics['linear_speed_likelihood'] = np.exp(log_likelihood)

    # Linear acceleration
    min_val = config.getfloat('linear_acceleration', 'histogram.min_val')
    max_val = config.getfloat('linear_acceleration', 'histogram.max_val')
    num_bins = config.getint('linear_acceleration', 'histogram.num_bins')
    smoothing = config.getfloat('linear_acceleration', 'histogram.additive_smoothing_pseudocount')

    log_likelihood = compute_histogram_log_likelihood(
        gt_features['linear_acceleration'],
        sim_features['linear_acceleration'],
        min_val, max_val, num_bins, smoothing
    )
    metrics['linear_acceleration_log_likelihood'] = log_likelihood
    metrics['linear_acceleration_likelihood'] = np.exp(log_likelihood)

    # Angular speed
    min_val = config.getfloat('angular_speed', 'histogram.min_val')
    max_val = config.getfloat('angular_speed', 'histogram.max_val')
    num_bins = config.getint('angular_speed', 'histogram.num_bins')
    smoothing = config.getfloat('angular_speed', 'histogram.additive_smoothing_pseudocount')

    log_likelihood = compute_histogram_log_likelihood(
        gt_features['angular_speed'],
        sim_features['angular_speed'],
        min_val, max_val, num_bins, smoothing
    )
    metrics['angular_speed_log_likelihood'] = log_likelihood
    metrics['angular_speed_likelihood'] = np.exp(log_likelihood)

    # Angular acceleration
    min_val = config.getfloat('angular_acceleration', 'histogram.min_val')
    max_val = config.getfloat('angular_acceleration', 'histogram.max_val')
    num_bins = config.getint('angular_acceleration', 'histogram.num_bins')
    smoothing = config.getfloat('angular_acceleration', 'histogram.additive_smoothing_pseudocount')

    log_likelihood = compute_histogram_log_likelihood(
        gt_features['angular_acceleration'],
        sim_features['angular_acceleration'],
        min_val, max_val, num_bins, smoothing
    )
    metrics['angular_acceleration_log_likelihood'] = log_likelihood
    metrics['angular_acceleration_likelihood'] = np.exp(log_likelihood)

    # Compute weighted kinematic realism score (metametric)
    weights = {
        'linear_speed': config.getfloat('linear_speed', 'metametric_weight'),
        'linear_acceleration': config.getfloat('linear_acceleration', 'metametric_weight'),
        'angular_speed': config.getfloat('angular_speed', 'metametric_weight'),
        'angular_acceleration': config.getfloat('angular_acceleration', 'metametric_weight'),
    }

    kinematic_score = (
        weights['linear_speed'] * metrics['linear_speed_likelihood'] +
        weights['linear_acceleration'] * metrics['linear_acceleration_likelihood'] +
        weights['angular_speed'] * metrics['angular_speed_likelihood'] +
        weights['angular_acceleration'] * metrics['angular_acceleration_likelihood']
    ) / sum(weights.values())

    metrics['kinematic_realism_score'] = kinematic_score

    return metrics


def compute_leave_one_out_baseline(
    all_segment_trajectories: List[Dict[str, np.ndarray]],
    dt: float = 0.1,
) -> Dict[str, float]:
    """Compute leave-one-out kinematic realism baseline for ground truth.

    Each segment is evaluated against the distribution built from all other segments.

    Args:
        all_segment_trajectories: List of trajectory dicts with {x, y, heading, valid, ...}
        dt: Time step in seconds

    Returns:
        Dict with baseline kinematic realism metrics
    """
    config = load_wosac_config()
    n_segments = len(all_segment_trajectories)

    # Compute features for all segments
    all_features = []
    for traj in all_segment_trajectories:
        features = compute_kinematic_features(
            traj['x'], traj['y'], traj['heading'], traj['valid'], dt
        )
        all_features.append(features)

    # Leave-one-out evaluation
    segment_scores = []

    for i in range(n_segments):
        # Build distribution from all segments except i
        other_features = [all_features[j] for j in range(n_segments) if j != i]
        population_features = aggregate_features_across_segments(other_features)

        # Evaluate segment i against population
        test_features = aggregate_features_across_segments([all_features[i]])

        score = compute_kinematic_realism_score(
            test_features,
            population_features,
            config
        )
        segment_scores.append(score)

    # Average across all segments
    avg_metrics = {}
    for key in segment_scores[0].keys():
        values = [s[key] for s in segment_scores if not np.isnan(s[key])]
        avg_metrics[key] = float(np.mean(values)) if values else np.nan

    return avg_metrics


def format_kinematic_summary(metrics: Dict[str, float], model_name: str = '') -> str:
    """Format kinematic realism metrics as readable string.

    Args:
        metrics: Dict from compute_kinematic_realism_score()
        model_name: Optional model name for display

    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Kinematic Realism Metrics: {model_name}")
    lines.append(f"{'='*60}")
    lines.append(f"  Linear speed:         {metrics['linear_speed_likelihood']:.4f}")
    lines.append(f"  Linear acceleration:  {metrics['linear_acceleration_likelihood']:.4f}")
    lines.append(f"  Angular speed:        {metrics['angular_speed_likelihood']:.4f}")
    lines.append(f"  Angular acceleration: {metrics['angular_acceleration_likelihood']:.4f}")
    lines.append(f"{'─'*60}")
    lines.append(f"  Kinematic Realism Score: {metrics['kinematic_realism_score']:.4f}")
    lines.append(f"{'='*60}")

    return '\n'.join(lines)
