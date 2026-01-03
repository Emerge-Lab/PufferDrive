#!/usr/bin/env python3
"""
Ablation study for JERK dynamics discretization.

Tests how trajectory approximation quality improves with finer action discretization.
Compares baseline (12 actions) vs expanded (40 actions) configurations.
"""

import argparse
import os
import json
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from pufferlib.ocean.benchmark.trajectory_approximation import (
    ClassicDynamics,
    JerkDynamics,
    load_trajectories_from_directory,
    find_valid_segments,
    extract_segment,
    initialize_classic_state,
    initialize_jerk_state,
    approximate_trajectory,
    compute_all_metrics,
    aggregate_metrics,
    format_metrics_summary,
    compute_kinematic_features,
    aggregate_features_across_segments,
    compute_kinematic_realism_score,
    format_kinematic_summary,
    plot_jerk_ablation_trajectory,
)


# Define JERK configurations to test
JERK_CONFIGS = {
    'baseline': {
        'jerk_long': [-15.0, -4.0, 0.0, 4.0],
        'jerk_lat': [-4.0, 0.0, 4.0],
        'description': 'Baseline (4x3=12 actions)',
    },
    'expanded': {
        'jerk_long': [-15.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0],
        'jerk_lat': [-4.0, -2.0, 0.0, 2.0, 4.0],
        'description': 'Expanded (8x5=40 actions)',
    },
}


def parse_weights(weight_string):
    """Parse comma-separated weights string."""
    try:
        pos, vel, heading = map(float, weight_string.split(','))
        return {'pos': pos, 'vel': vel, 'heading': heading}
    except:
        raise ValueError(f"Invalid weights format: {weight_string}. Expected: pos,vel,heading")


def run_configuration(config_name, config, trajectories, args, weights, store_trajectories=False):
    """
    Run trajectory approximation for a single JERK configuration.

    Args:
        store_trajectories: If True, store full trajectory data for visualization

    Returns:
        dict with metrics_list, segments_list, kinematic scores, and optionally trajectory data
    """
    print(f"\n{'='*60}")
    print(f"Running: {config['description']}")
    print(f"  Longitudinal jerk: {config['jerk_long']}")
    print(f"  Lateral jerk: {config['jerk_lat']}")
    print(f"{'='*60}")

    # Initialize dynamics model
    jerk_dynamics = JerkDynamics(
        dt=args.dt,
        jerk_long=config['jerk_long'],
        jerk_lat=config['jerk_lat']
    )
    print(f"  Total actions: {jerk_dynamics.num_actions}")

    # Process trajectories
    metrics_list = []
    segments_list = []
    trajectory_data = [] if store_trajectories else None
    skipped_stationary = 0
    processed_segments = 0

    for traj_idx, traj in enumerate(tqdm(trajectories, desc=f"{config_name}")):
        # Find valid segments
        segments = find_valid_segments(traj['valid'], min_length=args.min_segment_length)

        if len(segments) == 0:
            continue

        for seg_idx, (start, end) in enumerate(segments):
            # Extract segment
            segment = extract_segment(traj, start, end)

            # Filter stationary vehicles
            position_change = np.sqrt(
                (segment['x'][-1] - segment['x'][0])**2 +
                (segment['y'][-1] - segment['y'][0])**2
            )
            speed = np.sqrt(segment['vx']**2 + segment['vy']**2)
            max_speed = np.max(speed)
            mean_speed = np.mean(speed)

            if position_change < 0.5 and max_speed < 0.5:
                skipped_stationary += 1
                continue

            if mean_speed < args.min_mean_speed:
                skipped_stationary += 1
                continue

            processed_segments += 1

            # Initialize state
            jerk_state = initialize_jerk_state(segment, dt=args.dt)

            # Approximate trajectory
            jerk_traj, jerk_actions, jerk_errors = approximate_trajectory(
                jerk_dynamics, segment, jerk_state, weights=weights
            )
            jerk_traj['valid'] = segment['valid'].copy()

            # Compute metrics
            metrics = compute_all_metrics(jerk_traj, segment, segment['valid'])
            metrics_list.append(metrics)

            # Store for kinematic realism
            segments_list.append({
                'ground_truth': segment,
                'predicted': jerk_traj,
            })

            # Store for trajectory visualization
            if store_trajectories:
                trajectory_data.append({
                    'scenario': traj['scenario'],
                    'object_id': traj['object_id'],
                    'traj_idx': traj_idx,
                    'seg_idx': seg_idx,
                    'segment_range': (start, end),
                    'ground_truth': segment,
                    'predicted': jerk_traj,
                })

    print(f"\n  Processed: {processed_segments} segments")
    print(f"  Skipped: {skipped_stationary} stationary/slow segments")

    if len(metrics_list) == 0:
        return None

    # Aggregate metrics
    aggregated = aggregate_metrics(metrics_list, model_name=config['description'])

    # Compute kinematic realism
    gt_features_list = []
    pred_features_list = []

    for seg in segments_list:
        gt_feat = compute_kinematic_features(
            seg['ground_truth']['x'],
            seg['ground_truth']['y'],
            seg['ground_truth']['heading'],
            seg['ground_truth']['valid'],
            dt=args.dt
        )
        gt_features_list.append(gt_feat)

        pred_feat = compute_kinematic_features(
            seg['predicted']['x'],
            seg['predicted']['y'],
            seg['predicted']['heading'],
            seg['predicted']['valid'],
            dt=args.dt
        )
        pred_features_list.append(pred_feat)

    gt_population = aggregate_features_across_segments(gt_features_list)
    pred_population = aggregate_features_across_segments(pred_features_list)
    kinematic_score = compute_kinematic_realism_score(gt_population, pred_population)

    # Print summary
    print(format_metrics_summary(aggregated))
    print(format_kinematic_summary(kinematic_score, config['description']))

    result = {
        'metrics_list': metrics_list,
        'aggregated': aggregated,
        'kinematic_score': kinematic_score,
        'num_segments': len(metrics_list),
    }

    if store_trajectories:
        result['trajectory_data'] = trajectory_data

    return result


def plot_ablation_comparison(results, output_path):
    """
    Create histogram comparison for ablation study.
    Shows baseline vs expanded error distributions for position, speed, and heading.

    Args:
        results: dict mapping config_name -> result dict
        output_path: path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extract error distributions from metrics_list
    baseline_metrics = results['baseline']['metrics_list']
    expanded_metrics = results['expanded']['metrics_list']

    # Extract position errors (ADE)
    baseline_ade = [m['position']['ade'] for m in baseline_metrics]
    expanded_ade = [m['position']['ade'] for m in expanded_metrics]

    # Extract speed errors (MAE)
    baseline_speed = [m['velocity']['speed_mae'] for m in baseline_metrics]
    expanded_speed = [m['velocity']['speed_mae'] for m in expanded_metrics]

    # Extract heading errors (MAE)
    baseline_heading = [m['heading']['mae'] for m in baseline_metrics]
    expanded_heading = [m['heading']['mae'] for m in expanded_metrics]

    # Panel 1: Position error histogram
    axes[0].hist(baseline_ade, bins=30, alpha=0.5, color='red', label='Baseline (12 actions)')
    axes[0].hist(expanded_ade, bins=30, alpha=0.5, color='blue', label='Expanded (40 actions)')
    axes[0].set_xlabel('ADE (m)', fontweight='bold')
    axes[0].set_ylabel('Frequency', fontweight='bold')
    axes[0].set_title('Position Error Distribution', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Panel 2: Speed error histogram
    axes[1].hist(baseline_speed, bins=30, alpha=0.5, color='red', label='Baseline (12 actions)')
    axes[1].hist(expanded_speed, bins=30, alpha=0.5, color='blue', label='Expanded (40 actions)')
    axes[1].set_xlabel('Speed MAE (m/s)', fontweight='bold')
    axes[1].set_ylabel('Frequency', fontweight='bold')
    axes[1].set_title('Speed Error Distribution', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Panel 3: Heading error histogram
    axes[2].hist(baseline_heading, bins=30, alpha=0.5, color='red', label='Baseline (12 actions)')
    axes[2].hist(expanded_heading, bins=30, alpha=0.5, color='blue', label='Expanded (40 actions)')
    axes[2].set_xlabel('Heading MAE (deg)', fontweight='bold')
    axes[2].set_ylabel('Frequency', fontweight='bold')
    axes[2].set_title('Heading Error Distribution', fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nAblation comparison plot saved to: {output_path}")


def save_ablation_summary(results, output_path):
    """Save ablation study summary to JSON."""
    summary = {}

    for config_name, result in results.items():
        summary[config_name] = {
            'description': JERK_CONFIGS[config_name]['description'],
            'jerk_long': JERK_CONFIGS[config_name]['jerk_long'],
            'jerk_lat': JERK_CONFIGS[config_name]['jerk_lat'],
            'num_actions': len(JERK_CONFIGS[config_name]['jerk_long']) *
                          len(JERK_CONFIGS[config_name]['jerk_lat']),
            'num_segments': result['num_segments'],
            'aggregated_metrics': result['aggregated'],
            'kinematic_realism': result['kinematic_score'],
        }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Ablation summary saved to: {output_path}")


def print_comparison_table(results):
    """Print comparison table of key metrics."""
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPARISON")
    print(f"{'='*80}")

    # Header
    print(f"\n{'Metric':<30} {'Baseline (12)':<20} {'Expanded (40)':<20} {'Improvement':<15}")
    print(f"{'-'*30} {'-'*20} {'-'*20} {'-'*15}")

    # Get values
    def get_val(config, path):
        val = results[config]['aggregated']
        for key in path.split('.'):
            val = val[key]
        return val

    metrics = [
        ('Position ADE (m)', 'position.ade_mean'),
        ('Position FDE (m)', 'position.fde_mean'),
        ('Position P90 (m)', 'position.ade_p90'),
        ('Position Max (m)', 'position.max_mean'),
        ('Speed MAE (m/s)', 'velocity.speed_mae_mean'),
        ('Speed P90 (m/s)', 'velocity.speed_max_p90'),
        ('Heading MAE (deg)', 'heading.mae_mean'),
        ('Heading P90 (deg)', 'heading.max_p90'),
    ]

    for label, path in metrics:
        baseline = get_val('baseline', path)
        expanded = get_val('expanded', path)
        improvement = ((baseline - expanded) / baseline) * 100

        print(f"{label:<30} {baseline:<20.4f} {expanded:<20.4f} {improvement:>+14.2f}%")

    # Kinematic realism (higher is better)
    baseline_kr = results['baseline']['kinematic_score']['kinematic_realism_score']
    expanded_kr = results['expanded']['kinematic_score']['kinematic_realism_score']
    kr_improvement = ((expanded_kr - baseline_kr) / baseline_kr) * 100

    print(f"\n{'Kinematic Realism Score':<30} {baseline_kr:<20.4f} {expanded_kr:<20.4f} {kr_improvement:>+14.2f}%")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Ablation study for JERK dynamics discretization'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/training',
        help='Directory containing JSON trajectory files'
    )
    parser.add_argument(
        '--num_scenarios',
        type=int,
        default=100,
        help='Number of scenarios to process (default: 100)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/jerk_ablation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--min_segment_length',
        type=int,
        default=10,
        help='Minimum valid segment length (default: 10 timesteps)'
    )
    parser.add_argument(
        '--error_weights',
        type=str,
        default='1.0,1.0,10.0',
        help='Weights for pos,vel,heading errors (default: 1.0,1.0,10.0)'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.1,
        help='Time step in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--min_mean_speed',
        type=float,
        default=0.5,
        help='Minimum mean speed (m/s) to process segment (default: 0.5)'
    )
    parser.add_argument(
        '--configs',
        type=str,
        nargs='+',
        default=['baseline', 'expanded'],
        choices=['baseline', 'expanded'],
        help='Which configurations to run (default: both)'
    )
    parser.add_argument(
        '--max_visualizations',
        type=int,
        default=10,
        help='Maximum number of individual trajectory visualizations (default: 10)'
    )

    args = parser.parse_args()

    # Parse weights
    weights = parse_weights(args.error_weights)
    print(f"Using error weights: pos={weights['pos']}, vel={weights['vel']}, heading={weights['heading']}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Load trajectories (shared across all configs)
    print(f"\nLoading trajectories from {args.data_dir}...")
    trajectories = load_trajectories_from_directory(
        args.data_dir,
        num_scenarios=args.num_scenarios
    )
    print(f"Loaded {len(trajectories)} vehicle trajectories")

    # Run each configuration
    # Store trajectories only if we need visualizations and we have both configs
    store_trajs = args.max_visualizations > 0 and len(args.configs) >= 2
    results = {}
    for config_name in args.configs:
        config = JERK_CONFIGS[config_name]
        result = run_configuration(config_name, config, trajectories, args, weights,
                                   store_trajectories=store_trajs)

        if result is None:
            print(f"WARNING: No valid segments for {config_name}")
            continue

        results[config_name] = result

    if len(results) == 0:
        print("ERROR: No configurations produced valid results")
        return

    # Generate individual trajectory visualizations
    if store_trajs and 'baseline' in results and 'expanded' in results:
        print("\nGenerating individual trajectory visualizations...")
        baseline_trajs = results['baseline']['trajectory_data']
        expanded_trajs = results['expanded']['trajectory_data']

        # Match trajectories from both configs (they should be in same order)
        num_viz = min(args.max_visualizations, len(baseline_trajs))
        for i in range(num_viz):
            baseline_data = baseline_trajs[i]
            expanded_data = expanded_trajs[i]

            # Verify they're the same segment
            assert baseline_data['scenario'] == expanded_data['scenario']
            assert baseline_data['object_id'] == expanded_data['object_id']
            assert baseline_data['segment_range'] == expanded_data['segment_range']

            # Create visualization
            viz_path = os.path.join(
                viz_dir,
                f'trajectory_{baseline_data["traj_idx"]:04d}_seg{baseline_data["seg_idx"]:02d}.png'
            )

            plot_jerk_ablation_trajectory(
                baseline_data['ground_truth'],
                baseline_data['predicted'],
                expanded_data['predicted'],
                baseline_data['segment_range'],
                viz_path,
                config_names=('Baseline (12 actions)', 'Expanded (40 actions)')
            )

        print(f"  Generated {num_viz} trajectory visualizations in {viz_dir}")

    # Generate comparison visualizations
    print("\nGenerating aggregate comparison plots...")
    comparison_plot_path = os.path.join(args.output_dir, 'ablation_comparison.png')
    plot_ablation_comparison(results, comparison_plot_path)

    # Save summary
    summary_path = os.path.join(args.output_dir, 'ablation_summary.json')
    save_ablation_summary(results, summary_path)

    # Print comparison table
    if len(results) >= 2:
        print_comparison_table(results)

    print(f"\nAblation study complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
