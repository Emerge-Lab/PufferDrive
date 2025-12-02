"""Utility functions for trajectory processing and state initialization."""

import numpy as np


def wrap_angle(angle):
    """
    Wrap angle to [-pi, pi] range.

    Args:
        angle: Angle or array of angles in radians
    Returns:
        Wrapped angle in [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def find_valid_segments(valid_mask, min_length=10):
    """
    Extract continuous valid segments from trajectory.

    Args:
        valid_mask: Boolean array indicating valid timesteps
        min_length: Minimum segment length to consider
    Returns:
        List of (start_idx, end_idx) tuples
    """
    segments = []
    in_segment = False
    start = 0

    for i, is_valid in enumerate(valid_mask):
        if is_valid and not in_segment:
            start = i
            in_segment = True
        elif not is_valid and in_segment:
            if i - start >= min_length:
                segments.append((start, i))
            in_segment = False

    # Handle segment at end
    if in_segment and len(valid_mask) - start >= min_length:
        segments.append((start, len(valid_mask)))

    return segments


def extract_segment(trajectory, start, end):
    """
    Extract a segment from a trajectory dict.

    Args:
        trajectory: Dict with arrays {x, y, z, heading, vx, vy, valid, ...}
        start: Start index (inclusive)
        end: End index (exclusive)
    Returns:
        Trajectory dict with sliced arrays
    """
    segment = {}
    for key in ['x', 'y', 'z', 'heading', 'vx', 'vy', 'valid']:
        if key in trajectory:
            segment[key] = trajectory[key][start:end].copy()

    # Copy scalar values
    for key in ['width', 'length', 'height', 'object_id']:
        if key in trajectory:
            segment[key] = trajectory[key]

    return segment


def initialize_classic_state(trajectory):
    """
    Initialize CLASSIC dynamics state from trajectory.

    Args:
        trajectory: Dict with {x, y, heading, vx, vy, length, ...}
    Returns:
        Initial state dict for CLASSIC model
    """
    return {
        'x': trajectory['x'][0],
        'y': trajectory['y'][0],
        'heading': trajectory['heading'][0],
        'vx': trajectory['vx'][0],
        'vy': trajectory['vy'][0],
        'length': trajectory['length'],
    }


def initialize_jerk_state(trajectory, dt=0.1):
    """
    Initialize JERK dynamics state from trajectory.
    Estimates initial acceleration and steering angle from first few timesteps.

    Args:
        trajectory: Dict with {x, y, heading, vx, vy, length, ...}
        dt: Time delta between steps (default 0.1s)
    Returns:
        Initial state dict for JERK model
    """
    # Compute initial speed
    v0 = np.sqrt(trajectory['vx'][0]**2 + trajectory['vy'][0]**2)

    # Estimate longitudinal acceleration from speed change
    if len(trajectory['vx']) > 1:
        v1 = np.sqrt(trajectory['vx'][1]**2 + trajectory['vy'][1]**2)
        a_long_init = (v1 - v0) / dt
        a_long_init = np.clip(a_long_init, -5.0, 2.5)
    else:
        a_long_init = 0.0

    # Estimate curvature from heading change
    if len(trajectory['heading']) > 1:
        delta_heading = wrap_angle(trajectory['heading'][1] - trajectory['heading'][0])
        distance = np.sqrt(
            (trajectory['x'][1] - trajectory['x'][0])**2 +
            (trajectory['y'][1] - trajectory['y'][0])**2
        )
        # Avoid division by zero
        curvature = delta_heading / max(distance, 1e-5) if distance > 1e-5 else 0.0
    else:
        curvature = 0.0

    # Estimate steering angle from curvature
    wheelbase = trajectory['length']  # Approximate wheelbase as vehicle length
    steering_angle_init = np.arctan(curvature * wheelbase)
    steering_angle_init = np.clip(steering_angle_init, -0.55, 0.55)

    # Estimate lateral acceleration from curvature
    # a_lat = v^2 * curvature
    a_lat_init = v0**2 * curvature
    a_lat_init = np.clip(a_lat_init, -4.0, 4.0)

    return {
        'x': trajectory['x'][0],
        'y': trajectory['y'][0],
        'heading': trajectory['heading'][0],
        'vx': trajectory['vx'][0],
        'vy': trajectory['vy'][0],
        'a_long': a_long_init,
        'a_lat': a_lat_init,
        'steering_angle': steering_angle_init,
        'wheelbase': wheelbase,
    }
