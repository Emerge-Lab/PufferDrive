#!/usr/bin/env python3
"""
Command-line interface for trajectory approximation using CLASSIC and JERK dynamics models.

This script loads Waymo trajectories from JSON files and approximates them using
greedy search over discrete action spaces for both dynamics models.
"""

import argparse
import os
import json
import csv
from tqdm import tqdm
import numpy as np

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
    plot_trajectory_comparison,
    plot_aggregate_metrics,
    plot_error_histogram,
    compute_kinematic_features,
    aggregate_features_across_segments,
    compute_kinematic_realism_score,
    compute_leave_one_out_baseline,
    format_kinematic_summary,
)


def parse_weights(weight_string):
    """Parse comma-separated weights string."""
    try:
        pos, vel, heading = map(float, weight_string.split(','))
        return {'pos': pos, 'vel': vel, 'heading': heading}
    except:
        raise ValueError(f"Invalid weights format: {weight_string}. Expected: pos,vel,heading")


def save_metrics_to_csv(all_results, output_path):
    """Save detailed metrics to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'scenario', 'object_id', 'segment_start', 'segment_end', 'segment_length',
            'model',
            'ade', 'fde', 'pos_p90', 'pos_max',
            'speed_mae', 'speed_max', 'speed_p90',
            'heading_mae', 'heading_max', 'heading_p90'
        ])

        # Write rows
        for result in all_results:
            scenario = result['scenario']
            object_id = result['object_id']
            start, end = result['segment']
            segment_length = end - start

            for model_name in ['classic', 'jerk']:
                metrics = result[f'{model_name}_metrics']

                writer.writerow([
                    scenario, object_id, start, end, segment_length,
                    model_name.upper(),
                    metrics['position']['ade'],
                    metrics['position']['fde'],
                    metrics['position']['p90'],
                    metrics['position']['max'],
                    metrics['velocity']['speed_mae'],
                    metrics['velocity']['speed_max'],
                    metrics['velocity']['speed_p90'],
                    metrics['heading']['mae'],
                    metrics['heading']['max'],
                    metrics['heading']['p90'],
                ])


def save_summary_to_json(aggregated_classic, aggregated_jerk, kinematic_scores, output_path):
    """Save aggregated metrics to JSON file."""
    summary = {
        'classic': aggregated_classic,
        'jerk': aggregated_jerk,
        'kinematic_realism': kinematic_scores,
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Approximate Waymo trajectories using CLASSIC and JERK dynamics models'
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
        default='output/trajectory_approximation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--min_segment_length',
        type=int,
        default=10,
        help='Minimum valid segment length to process (default: 10 timesteps)'
    )
    parser.add_argument(
        '--error_weights',
        type=str,
        default='1.0,1.0,10.0',
        help='Weights for pos,vel,heading errors (default: 1.0,1.0,10.0)'
    )
    parser.add_argument(
        '--max_visualizations',
        type=int,
        default=10,
        help='Maximum number of individual trajectory visualizations (default: 10)'
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
        help='Minimum mean speed (m/s) to process segment (default: 0.5, filters stationary vehicles)'
    )

    args = parser.parse_args()

    # Parse weights
    weights = parse_weights(args.error_weights)
    print(f"Using error weights: pos={weights['pos']}, vel={weights['vel']}, heading={weights['heading']}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # Initialize dynamics models
    print(f"\nInitializing dynamics models (dt={args.dt}s)...")
    classic_dynamics = ClassicDynamics(dt=args.dt)
    jerk_dynamics = JerkDynamics(dt=args.dt)
    print(f"  CLASSIC: {classic_dynamics.num_actions} actions")
    print(f"  JERK: {jerk_dynamics.num_actions} actions")

    # Load trajectories
    print(f"\nLoading trajectories from {args.data_dir}...")
    trajectories = load_trajectories_from_directory(
        args.data_dir,
        num_scenarios=args.num_scenarios
    )
    print(f"Loaded {len(trajectories)} vehicle trajectories")

    # Process each trajectory
    print(f"\nProcessing trajectories (min segment length: {args.min_segment_length}, min mean speed: {args.min_mean_speed} m/s)...")
    all_results = []
    all_gt_segments = []  # For kinematic realism
    all_classic_segments = []  # For kinematic realism
    all_jerk_segments = []  # For kinematic realism
    viz_count = 0
    skipped_stationary = 0
    processed_segments = 0

    for traj_idx, traj in enumerate(tqdm(trajectories, desc="Trajectories")):
        # Find valid segments
        segments = find_valid_segments(traj['valid'], min_length=args.min_segment_length)

        if len(segments) == 0:
            continue

        for seg_idx, (start, end) in enumerate(segments):
            # Extract segment
            segment = extract_segment(traj, start, end)

            # Skip stationary vehicles (parked or stopped in traffic)
            # Calculate position change and speed statistics
            position_change = np.sqrt(
                (segment['x'][-1] - segment['x'][0])**2 +
                (segment['y'][-1] - segment['y'][0])**2
            )
            speed = np.sqrt(segment['vx']**2 + segment['vy']**2)
            max_speed = np.max(speed)
            mean_speed = np.mean(speed)

            # Filter out stationary vehicles (moving less than 0.5m total and max speed < 0.5 m/s)
            if position_change < 0.5 and max_speed < 0.5:
                skipped_stationary += 1
                continue  # Skip this segment - stationary vehicle

            # Also skip very slow moving vehicles that won't test the dynamics well
            if mean_speed < args.min_mean_speed:
                skipped_stationary += 1
                continue

            processed_segments += 1

            # Initialize states
            classic_state = initialize_classic_state(segment)
            jerk_state = initialize_jerk_state(segment, dt=args.dt)

            # Approximate with CLASSIC model
            classic_traj, classic_actions, classic_errors = approximate_trajectory(
                classic_dynamics, segment, classic_state, weights=weights
            )
            # Add valid mask from ground truth
            classic_traj['valid'] = segment['valid'].copy()

            # Approximate with JERK model
            jerk_traj, jerk_actions, jerk_errors = approximate_trajectory(
                jerk_dynamics, segment, jerk_state, weights=weights
            )
            # Add valid mask from ground truth
            jerk_traj['valid'] = segment['valid'].copy()

            # Compute metrics
            classic_metrics = compute_all_metrics(classic_traj, segment, segment['valid'])
            jerk_metrics = compute_all_metrics(jerk_traj, segment, segment['valid'])

            # Store results
            result = {
                'scenario': traj['scenario'],
                'object_id': traj['object_id'],
                'segment': (start, end),
                'classic_metrics': classic_metrics,
                'jerk_metrics': jerk_metrics,
            }
            all_results.append(result)

            # Store trajectories for kinematic realism computation
            all_gt_segments.append(segment)
            all_classic_segments.append(classic_traj)
            all_jerk_segments.append(jerk_traj)

            # Create visualization for first few segments
            if viz_count < args.max_visualizations:
                viz_path = os.path.join(
                    viz_dir,
                    f'comparison_{traj_idx:04d}_seg{seg_idx:02d}.png'
                )
                plot_trajectory_comparison(
                    segment, classic_traj, jerk_traj,
                    (start, end), viz_path
                )
                viz_count += 1

    print(f"\nProcessing complete:")
    print(f"  Processed: {processed_segments} moving trajectory segments")
    print(f"  Skipped: {skipped_stationary} stationary/slow segments")
    print(f"  Total results: {len(all_results)} segments")

    if len(all_results) == 0:
        print("No valid trajectory segments found. Exiting.")
        return

    # Aggregate metrics
    print("\nAggregating metrics...")
    classic_metrics_list = [r['classic_metrics'] for r in all_results]
    jerk_metrics_list = [r['jerk_metrics'] for r in all_results]

    aggregated_classic = aggregate_metrics(classic_metrics_list, model_name='CLASSIC')
    aggregated_jerk = aggregate_metrics(jerk_metrics_list, model_name='JERK')

    # Print summary
    print(format_metrics_summary(aggregated_classic))
    print(format_metrics_summary(aggregated_jerk))

    # Compute kinematic realism scores
    print("\nComputing kinematic realism scores...")

    # Compute features for all segments
    print("  Computing kinematic features...")
    gt_features_list = []
    classic_features_list = []
    jerk_features_list = []

    for i in range(len(all_gt_segments)):
        gt_feat = compute_kinematic_features(
            all_gt_segments[i]['x'],
            all_gt_segments[i]['y'],
            all_gt_segments[i]['heading'],
            all_gt_segments[i]['valid'],
            dt=args.dt
        )
        gt_features_list.append(gt_feat)

        classic_feat = compute_kinematic_features(
            all_classic_segments[i]['x'],
            all_classic_segments[i]['y'],
            all_classic_segments[i]['heading'],
            all_classic_segments[i]['valid'],
            dt=args.dt
        )
        classic_features_list.append(classic_feat)

        jerk_feat = compute_kinematic_features(
            all_jerk_segments[i]['x'],
            all_jerk_segments[i]['y'],
            all_jerk_segments[i]['heading'],
            all_jerk_segments[i]['valid'],
            dt=args.dt
        )
        jerk_features_list.append(jerk_feat)

    # Aggregate features across population
    print("  Aggregating population-level features...")
    gt_population = aggregate_features_across_segments(gt_features_list)
    classic_population = aggregate_features_across_segments(classic_features_list)
    jerk_population = aggregate_features_across_segments(jerk_features_list)

    # Compute leave-one-out baseline for ground truth
    print("  Computing ground truth baseline (leave-one-out)...")
    gt_baseline_metrics = compute_leave_one_out_baseline(all_gt_segments, dt=args.dt)

    # Compute kinematic realism scores for inverse dynamics
    print("  Evaluating CLASSIC kinematic realism...")
    classic_kinematic = compute_kinematic_realism_score(gt_population, classic_population)

    print("  Evaluating JERK kinematic realism...")
    jerk_kinematic = compute_kinematic_realism_score(gt_population, jerk_population)

    # Print kinematic realism summaries
    print(format_kinematic_summary(gt_baseline_metrics, 'Ground Truth (Leave-One-Out)'))
    print(format_kinematic_summary(classic_kinematic, 'CLASSIC'))
    print(format_kinematic_summary(jerk_kinematic, 'JERK'))

    # Compute realism loss
    print(f"\n{'='*60}")
    print("Kinematic Realism Loss (GT Baseline - Inverse Dynamics)")
    print(f"{'='*60}")
    classic_loss = gt_baseline_metrics['kinematic_realism_score'] - classic_kinematic['kinematic_realism_score']
    jerk_loss = gt_baseline_metrics['kinematic_realism_score'] - jerk_kinematic['kinematic_realism_score']
    print(f"  CLASSIC loss: {classic_loss:.4f}")
    print(f"  JERK loss:    {jerk_loss:.4f}")
    print(f"{'='*60}")

    # Save results
    print("\nSaving results...")

    # Save detailed metrics to CSV
    csv_path = os.path.join(args.output_dir, 'metrics_detailed.csv')
    save_metrics_to_csv(all_results, csv_path)
    print(f"  Detailed metrics saved to: {csv_path}")

    # Save aggregated metrics to JSON
    json_path = os.path.join(args.output_dir, 'metrics_summary.json')
    kinematic_scores = {
        'ground_truth_baseline': gt_baseline_metrics,
        'classic': classic_kinematic,
        'jerk': jerk_kinematic,
        'classic_realism_loss': classic_loss,
        'jerk_realism_loss': jerk_loss,
    }
    save_summary_to_json(aggregated_classic, aggregated_jerk, kinematic_scores, json_path)
    print(f"  Summary metrics saved to: {json_path}")

    # Generate aggregate visualizations
    print("\nGenerating visualizations...")

    # Box plots
    boxplot_path = os.path.join(args.output_dir, 'comparison_boxplots.png')
    plot_aggregate_metrics(classic_metrics_list, jerk_metrics_list, boxplot_path)
    print(f"  Box plots saved to: {boxplot_path}")

    # Histograms
    histogram_path = os.path.join(args.output_dir, 'error_histograms.png')
    plot_error_histogram(classic_metrics_list, jerk_metrics_list, histogram_path)
    print(f"  Histograms saved to: {histogram_path}")

    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
