"""Trajectory approximation using CLASSIC and JERK dynamics models."""

from .dynamics import ClassicDynamics, JerkDynamics
from .trajectory_loader import (
    load_trajectory_from_json,
    load_trajectories_from_directory,
    get_trajectory_statistics,
)
from .trajectory_utils import (
    wrap_angle,
    find_valid_segments,
    extract_segment,
    initialize_classic_state,
    initialize_jerk_state,
)
from .greedy_search import (
    GreedyActionSelector,
    approximate_trajectory,
    decode_action,
)
from .error_metrics import (
    compute_position_error,
    compute_velocity_error,
    compute_heading_error,
    compute_all_metrics,
    aggregate_metrics,
    format_metrics_summary,
)
from .visualizer import (
    plot_trajectory_comparison,
    plot_jerk_ablation_trajectory,
    plot_aggregate_metrics,
    plot_error_histogram,
)
from .kinematic_realism import (
    compute_kinematic_features,
    aggregate_features_across_segments,
    compute_kinematic_realism_score,
    compute_leave_one_out_baseline,
    format_kinematic_summary,
)

__all__ = [
    # Dynamics models
    'ClassicDynamics',
    'JerkDynamics',
    # Trajectory loading
    'load_trajectory_from_json',
    'load_trajectories_from_directory',
    'get_trajectory_statistics',
    # Utilities
    'wrap_angle',
    'find_valid_segments',
    'extract_segment',
    'initialize_classic_state',
    'initialize_jerk_state',
    # Greedy search
    'GreedyActionSelector',
    'approximate_trajectory',
    'decode_action',
    # Metrics
    'compute_position_error',
    'compute_velocity_error',
    'compute_heading_error',
    'compute_all_metrics',
    'aggregate_metrics',
    'format_metrics_summary',
    # Visualization
    'plot_trajectory_comparison',
    'plot_jerk_ablation_trajectory',
    'plot_aggregate_metrics',
    'plot_error_histogram',
    # Kinematic realism
    'compute_kinematic_features',
    'aggregate_features_across_segments',
    'compute_kinematic_realism_score',
    'compute_leave_one_out_baseline',
    'format_kinematic_summary',
]
