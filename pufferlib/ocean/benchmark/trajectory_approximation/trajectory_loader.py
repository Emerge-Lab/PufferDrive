"""Load Waymo trajectory data from JSON files."""

import json
import numpy as np
import glob
import os


def load_trajectory_from_json(json_path):
    """
    Load Waymo trajectory data from JSON file.

    Args:
        json_path: Path to JSON file
    Returns:
        List of trajectory dicts, one per vehicle object
    """
    with open(json_path, 'r') as f:
        scenario = json.load(f)

    trajectories = []
    scenario_name = os.path.basename(json_path)

    for obj in scenario['objects']:
        # Only process vehicles (skip pedestrians)
        if obj['type'] != 'vehicle':
            continue

        # Extract trajectory data
        traj = {
            'x': np.array([p['x'] for p in obj['position']], dtype=np.float32),
            'y': np.array([p['y'] for p in obj['position']], dtype=np.float32),
            'z': np.array([p['z'] for p in obj['position']], dtype=np.float32),
            'heading': np.array(obj['heading'], dtype=np.float32),
            'vx': np.array([v['x'] for v in obj['velocity']], dtype=np.float32),
            'vy': np.array([v['y'] for v in obj['velocity']], dtype=np.float32),
            'valid': np.array(obj['valid'], dtype=bool),
            'width': obj['width'],
            'length': obj['length'],
            'height': obj['height'],
            'object_id': obj['id'],
            'scenario': scenario_name,
        }

        # Filter out invalid velocity sentinels (-10000 indicates missing data)
        invalid_mask = (traj['vx'] == -10000) | (traj['vy'] == -10000)
        traj['valid'] = traj['valid'] & ~invalid_mask

        trajectories.append(traj)

    return trajectories


def load_trajectories_from_directory(data_dir, num_scenarios=None, pattern='*.json'):
    """
    Load trajectories from all JSON files in a directory.

    Args:
        data_dir: Directory containing JSON files
        num_scenarios: Maximum number of scenarios to load (None = all)
        pattern: Glob pattern for files (default '*.json')
    Returns:
        List of all trajectory dicts from all scenarios
    """
    json_files = sorted(glob.glob(os.path.join(data_dir, pattern)))

    if num_scenarios is not None:
        json_files = json_files[:num_scenarios]

    all_trajectories = []
    for json_path in json_files:
        try:
            trajectories = load_trajectory_from_json(json_path)
            all_trajectories.extend(trajectories)
        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}")
            continue

    return all_trajectories


def get_trajectory_statistics(trajectories):
    """
    Compute statistics across a list of trajectories.

    Args:
        trajectories: List of trajectory dicts
    Returns:
        Dict with statistics
    """
    num_trajectories = len(trajectories)

    # Count valid timesteps
    total_valid_timesteps = sum(np.sum(traj['valid']) for traj in trajectories)
    avg_valid_timesteps = total_valid_timesteps / num_trajectories if num_trajectories > 0 else 0

    # Speed statistics
    speeds = []
    for traj in trajectories:
        speed = np.sqrt(traj['vx'][traj['valid']]**2 + traj['vy'][traj['valid']]**2)
        speeds.extend(speed.tolist())
    speeds = np.array(speeds)

    # Length statistics
    lengths = np.array([traj['length'] for traj in trajectories])
    widths = np.array([traj['width'] for traj in trajectories])

    return {
        'num_trajectories': num_trajectories,
        'avg_valid_timesteps': avg_valid_timesteps,
        'total_valid_timesteps': total_valid_timesteps,
        'speed_mean': np.mean(speeds) if len(speeds) > 0 else 0,
        'speed_std': np.std(speeds) if len(speeds) > 0 else 0,
        'speed_max': np.max(speeds) if len(speeds) > 0 else 0,
        'length_mean': np.mean(lengths),
        'length_std': np.std(lengths),
        'width_mean': np.mean(widths),
        'width_std': np.std(widths),
    }
