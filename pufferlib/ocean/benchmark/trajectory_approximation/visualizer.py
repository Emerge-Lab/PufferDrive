"""Visualization functions for trajectory approximation results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def plot_trajectory_comparison(ground_truth, classic_approx, jerk_approx,
                               segment_range, output_path):
    """
    Create 6-panel comparison plot.

    Panels:
    - Row 1: XY trajectory, Speed over time, Heading over time
    - Row 2: Position error, Velocity error, Heading error

    Args:
        ground_truth: Ground truth trajectory dict
        classic_approx: CLASSIC model approximation dict
        jerk_approx: JERK model approximation dict
        segment_range: Tuple (start, end) for segment indices
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    start, end = segment_range

    # Time axis
    t = np.arange(end - start) * 0.1  # 0.1s timestep

    # Panel 1: XY trajectory
    axes[0, 0].plot(ground_truth['x'], ground_truth['y'],
                    'k-', label='Ground Truth', linewidth=2)
    axes[0, 0].plot(classic_approx['x'], classic_approx['y'],
                    'r--', label='CLASSIC', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(jerk_approx['x'], jerk_approx['y'],
                    'b--', label='JERK', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].legend()
    axes[0, 0].set_title('Trajectory Comparison')
    axes[0, 0].axis('equal')
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Speed over time
    gt_speed = np.sqrt(ground_truth['vx']**2 + ground_truth['vy']**2)
    classic_speed = np.sqrt(classic_approx['vx']**2 + classic_approx['vy']**2)
    jerk_speed = np.sqrt(jerk_approx['vx']**2 + jerk_approx['vy']**2)

    axes[0, 1].plot(t, gt_speed, 'k-', label='Ground Truth', linewidth=2)
    axes[0, 1].plot(t, classic_speed, 'r--', label='CLASSIC', alpha=0.7, linewidth=1.5)
    axes[0, 1].plot(t, jerk_speed, 'b--', label='JERK', alpha=0.7, linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Speed (m/s)')
    axes[0, 1].legend()
    axes[0, 1].set_title('Speed Profile')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Heading over time
    axes[0, 2].plot(t, ground_truth['heading'] * 180/np.pi,
                    'k-', label='Ground Truth', linewidth=2)
    axes[0, 2].plot(t, classic_approx['heading'] * 180/np.pi,
                    'r--', label='CLASSIC', alpha=0.7, linewidth=1.5)
    axes[0, 2].plot(t, jerk_approx['heading'] * 180/np.pi,
                    'b--', label='JERK', alpha=0.7, linewidth=1.5)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Heading (deg)')
    axes[0, 2].legend()
    axes[0, 2].set_title('Heading Profile')
    axes[0, 2].grid(True, alpha=0.3)

    # Panel 4: Position error over time
    classic_pos_error = np.sqrt(
        (classic_approx['x'] - ground_truth['x'])**2 +
        (classic_approx['y'] - ground_truth['y'])**2
    )
    jerk_pos_error = np.sqrt(
        (jerk_approx['x'] - ground_truth['x'])**2 +
        (jerk_approx['y'] - ground_truth['y'])**2
    )

    axes[1, 0].plot(t, classic_pos_error, 'r-', label='CLASSIC', linewidth=1.5)
    axes[1, 0].plot(t, jerk_pos_error, 'b-', label='JERK', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Position Error (m)')
    axes[1, 0].legend()
    axes[1, 0].set_title('Position Error Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 5: Speed error over time
    classic_speed_error = np.abs(classic_speed - gt_speed)
    jerk_speed_error = np.abs(jerk_speed - gt_speed)

    axes[1, 1].plot(t, classic_speed_error, 'r-', label='CLASSIC', linewidth=1.5)
    axes[1, 1].plot(t, jerk_speed_error, 'b-', label='JERK', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Speed Error (m/s)')
    axes[1, 1].legend()
    axes[1, 1].set_title('Speed Error Over Time')
    axes[1, 1].grid(True, alpha=0.3)

    # Panel 6: Heading error over time
    from .trajectory_utils import wrap_angle
    classic_heading_error = np.abs(wrap_angle(
        classic_approx['heading'] - ground_truth['heading']
    )) * 180/np.pi
    jerk_heading_error = np.abs(wrap_angle(
        jerk_approx['heading'] - ground_truth['heading']
    )) * 180/np.pi

    axes[1, 2].plot(t, classic_heading_error, 'r-', label='CLASSIC', linewidth=1.5)
    axes[1, 2].plot(t, jerk_heading_error, 'b-', label='JERK', linewidth=1.5)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Heading Error (deg)')
    axes[1, 2].legend()
    axes[1, 2].set_title('Heading Error Over Time')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_jerk_ablation_trajectory(ground_truth, baseline_approx, expanded_approx,
                                   segment_range, output_path, config_names=None):
    """
    Create 6-panel comparison plot for JERK ablation study.

    Panels:
    - Row 1: XY trajectory, Speed over time, Heading over time
    - Row 2: Position error, Velocity error, Heading error

    Args:
        ground_truth: Ground truth trajectory dict
        baseline_approx: Baseline JERK approximation dict
        expanded_approx: Expanded JERK approximation dict
        segment_range: Tuple (start, end) for segment indices
        output_path: Path to save the plot
        config_names: Optional tuple of (baseline_name, expanded_name)
    """
    if config_names is None:
        config_names = ('JERK Baseline', 'JERK Expanded')

    baseline_name, expanded_name = config_names

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    start, end = segment_range

    # Time axis
    t = np.arange(end - start) * 0.1  # 0.1s timestep

    # Panel 1: XY trajectory
    axes[0, 0].plot(ground_truth['x'], ground_truth['y'],
                    'k-', label='Ground Truth', linewidth=2)
    axes[0, 0].plot(baseline_approx['x'], baseline_approx['y'],
                    'r--', label=baseline_name, alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(expanded_approx['x'], expanded_approx['y'],
                    'b--', label=expanded_name, alpha=0.7, linewidth=1.5)
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].legend()
    axes[0, 0].set_title('Trajectory Comparison')
    axes[0, 0].axis('equal')
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Speed over time
    gt_speed = np.sqrt(ground_truth['vx']**2 + ground_truth['vy']**2)
    baseline_speed = np.sqrt(baseline_approx['vx']**2 + baseline_approx['vy']**2)
    expanded_speed = np.sqrt(expanded_approx['vx']**2 + expanded_approx['vy']**2)

    axes[0, 1].plot(t, gt_speed, 'k-', label='Ground Truth', linewidth=2)
    axes[0, 1].plot(t, baseline_speed, 'r--', label=baseline_name, alpha=0.7, linewidth=1.5)
    axes[0, 1].plot(t, expanded_speed, 'b--', label=expanded_name, alpha=0.7, linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Speed (m/s)')
    axes[0, 1].legend()
    axes[0, 1].set_title('Speed Profile')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Heading over time
    axes[0, 2].plot(t, ground_truth['heading'] * 180/np.pi,
                    'k-', label='Ground Truth', linewidth=2)
    axes[0, 2].plot(t, baseline_approx['heading'] * 180/np.pi,
                    'r--', label=baseline_name, alpha=0.7, linewidth=1.5)
    axes[0, 2].plot(t, expanded_approx['heading'] * 180/np.pi,
                    'b--', label=expanded_name, alpha=0.7, linewidth=1.5)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Heading (deg)')
    axes[0, 2].legend()
    axes[0, 2].set_title('Heading Profile')
    axes[0, 2].grid(True, alpha=0.3)

    # Panel 4: Position error over time
    baseline_pos_error = np.sqrt(
        (baseline_approx['x'] - ground_truth['x'])**2 +
        (baseline_approx['y'] - ground_truth['y'])**2
    )
    expanded_pos_error = np.sqrt(
        (expanded_approx['x'] - ground_truth['x'])**2 +
        (expanded_approx['y'] - ground_truth['y'])**2
    )

    axes[1, 0].plot(t, baseline_pos_error, 'r-', label=baseline_name, linewidth=1.5)
    axes[1, 0].plot(t, expanded_pos_error, 'b-', label=expanded_name, linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Position Error (m)')
    axes[1, 0].legend()
    axes[1, 0].set_title('Position Error Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 5: Speed error over time
    baseline_speed_error = np.abs(baseline_speed - gt_speed)
    expanded_speed_error = np.abs(expanded_speed - gt_speed)

    axes[1, 1].plot(t, baseline_speed_error, 'r-', label=baseline_name, linewidth=1.5)
    axes[1, 1].plot(t, expanded_speed_error, 'b-', label=expanded_name, linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Speed Error (m/s)')
    axes[1, 1].legend()
    axes[1, 1].set_title('Speed Error Over Time')
    axes[1, 1].grid(True, alpha=0.3)

    # Panel 6: Heading error over time
    from .trajectory_utils import wrap_angle
    baseline_heading_error = np.abs(wrap_angle(
        baseline_approx['heading'] - ground_truth['heading']
    )) * 180/np.pi
    expanded_heading_error = np.abs(wrap_angle(
        expanded_approx['heading'] - ground_truth['heading']
    )) * 180/np.pi

    axes[1, 2].plot(t, baseline_heading_error, 'r-', label=baseline_name, linewidth=1.5)
    axes[1, 2].plot(t, expanded_heading_error, 'b-', label=expanded_name, linewidth=1.5)
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Heading Error (deg)')
    axes[1, 2].legend()
    axes[1, 2].set_title('Heading Error Over Time')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_metrics(all_metrics_classic, all_metrics_jerk, output_path):
    """
    Create box plots comparing CLASSIC vs JERK across all trajectories.

    Args:
        all_metrics_classic: List of metric dicts for CLASSIC model
        all_metrics_jerk: List of metric dicts for JERK model
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Extract metrics for CLASSIC
    classic_ade = [m['position']['ade'] for m in all_metrics_classic]
    classic_speed = [m['velocity']['speed_mae'] for m in all_metrics_classic]
    classic_heading = [m['heading']['mae'] for m in all_metrics_classic]
    classic_fde = [m['position']['fde'] for m in all_metrics_classic]
    classic_speed_max = [m['velocity']['speed_max'] for m in all_metrics_classic]
    classic_heading_max = [m['heading']['max'] for m in all_metrics_classic]

    # Extract metrics for JERK
    jerk_ade = [m['position']['ade'] for m in all_metrics_jerk]
    jerk_speed = [m['velocity']['speed_mae'] for m in all_metrics_jerk]
    jerk_heading = [m['heading']['mae'] for m in all_metrics_jerk]
    jerk_fde = [m['position']['fde'] for m in all_metrics_jerk]
    jerk_speed_max = [m['velocity']['speed_max'] for m in all_metrics_jerk]
    jerk_heading_max = [m['heading']['max'] for m in all_metrics_jerk]

    # Panel 1: Average Displacement Error
    bp1 = axes[0, 0].boxplot([classic_ade, jerk_ade],
                              labels=['CLASSIC', 'JERK'],
                              patch_artist=True)
    bp1['boxes'][0].set_facecolor('red')
    bp1['boxes'][1].set_facecolor('blue')
    axes[0, 0].set_ylabel('ADE (m)')
    axes[0, 0].set_title('Average Displacement Error')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Panel 2: Speed MAE
    bp2 = axes[0, 1].boxplot([classic_speed, jerk_speed],
                              labels=['CLASSIC', 'JERK'],
                              patch_artist=True)
    bp2['boxes'][0].set_facecolor('red')
    bp2['boxes'][1].set_facecolor('blue')
    axes[0, 1].set_ylabel('Speed MAE (m/s)')
    axes[0, 1].set_title('Speed Error')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Panel 3: Heading MAE
    bp3 = axes[0, 2].boxplot([classic_heading, jerk_heading],
                              labels=['CLASSIC', 'JERK'],
                              patch_artist=True)
    bp3['boxes'][0].set_facecolor('red')
    bp3['boxes'][1].set_facecolor('blue')
    axes[0, 2].set_ylabel('Heading MAE (deg)')
    axes[0, 2].set_title('Heading Error')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # Panel 4: Final Displacement Error
    bp4 = axes[1, 0].boxplot([classic_fde, jerk_fde],
                              labels=['CLASSIC', 'JERK'],
                              patch_artist=True)
    bp4['boxes'][0].set_facecolor('red')
    bp4['boxes'][1].set_facecolor('blue')
    axes[1, 0].set_ylabel('FDE (m)')
    axes[1, 0].set_title('Final Displacement Error')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Panel 5: Max Speed Error
    bp5 = axes[1, 1].boxplot([classic_speed_max, jerk_speed_max],
                              labels=['CLASSIC', 'JERK'],
                              patch_artist=True)
    bp5['boxes'][0].set_facecolor('red')
    bp5['boxes'][1].set_facecolor('blue')
    axes[1, 1].set_ylabel('Max Speed Error (m/s)')
    axes[1, 1].set_title('Maximum Speed Error')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Panel 6: Max Heading Error
    bp6 = axes[1, 2].boxplot([classic_heading_max, jerk_heading_max],
                              labels=['CLASSIC', 'JERK'],
                              patch_artist=True)
    bp6['boxes'][0].set_facecolor('red')
    bp6['boxes'][1].set_facecolor('blue')
    axes[1, 2].set_ylabel('Max Heading Error (deg)')
    axes[1, 2].set_title('Maximum Heading Error')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    # Add overall title
    fig.suptitle(f'Trajectory Approximation Comparison ({len(all_metrics_classic)} segments)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_histogram(all_metrics_classic, all_metrics_jerk, output_path):
    """
    Create histograms of error distributions.

    Args:
        all_metrics_classic: List of metric dicts for CLASSIC model
        all_metrics_jerk: List of metric dicts for JERK model
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extract metrics
    classic_ade = [m['position']['ade'] for m in all_metrics_classic]
    jerk_ade = [m['position']['ade'] for m in all_metrics_jerk]

    classic_speed = [m['velocity']['speed_mae'] for m in all_metrics_classic]
    jerk_speed = [m['velocity']['speed_mae'] for m in all_metrics_jerk]

    classic_heading = [m['heading']['mae'] for m in all_metrics_classic]
    jerk_heading = [m['heading']['mae'] for m in all_metrics_jerk]

    # Panel 1: Position error histogram
    axes[0].hist(classic_ade, bins=30, alpha=0.5, color='red', label='CLASSIC')
    axes[0].hist(jerk_ade, bins=30, alpha=0.5, color='blue', label='JERK')
    axes[0].set_xlabel('ADE (m)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Position Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Panel 2: Speed error histogram
    axes[1].hist(classic_speed, bins=30, alpha=0.5, color='red', label='CLASSIC')
    axes[1].hist(jerk_speed, bins=30, alpha=0.5, color='blue', label='JERK')
    axes[1].set_xlabel('Speed MAE (m/s)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Speed Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Panel 3: Heading error histogram
    axes[2].hist(classic_heading, bins=30, alpha=0.5, color='red', label='CLASSIC')
    axes[2].hist(jerk_heading, bins=30, alpha=0.5, color='blue', label='JERK')
    axes[2].set_xlabel('Heading MAE (deg)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Heading Error Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
