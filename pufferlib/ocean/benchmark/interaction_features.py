"""Interaction features for the computation of the WOSAC score.
Adapted from: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/interaction_features.py
"""

import math
import numpy as np
from pufferlib.ocean.benchmark import geometry_utils

EXTREMELY_LARGE_DISTANCE = 1e10
COLLISION_DISTANCE_THRESHOLD = 0.0
CORNER_ROUNDING_FACTOR = 0.7
MAX_HEADING_DIFF = math.radians(75.0)
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = math.radians(10.0)
SMALL_OVERLAP_THRESHOLD = 0.5
MAXIMUM_TIME_TO_COLLISION = 5.0


def compute_signed_distances(
    center_x: np.ndarray,
    center_y: np.ndarray,
    length: np.ndarray,
    width: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> np.ndarray:
    """Computes pairwise signed distances for all agents.

    Objects are represented by 2D rectangles with rounded corners.
    Rollout dimension is preserved - agents from different rollouts do not interact.

    Args:
        center_x: Shape (num_agents, num_rollouts, num_steps)
        center_y: Shape (num_agents, num_rollouts, num_steps)
        length: Shape (num_agents, num_rollouts) - constant per timestep
        width: Shape (num_agents, num_rollouts) - constant per timestep
        heading: Shape (num_agents, num_rollouts, num_steps)
        valid: Shape (num_agents, num_rollouts, num_steps)
        corner_rounding_factor: Rounding factor for box corners, between 0 (sharp) and 1 (capsule)

    Returns:
        Tuple of:
        - min_distances: Distance to nearest object, shape (num_agents, num_rollouts, num_steps)
        - all_distances: All pairwise distances, shape (num_agents, num_agents, num_rollouts, num_steps)
    """
    num_agents = center_x.shape[0]
    num_rollouts = center_x.shape[1]
    num_steps = center_x.shape[2]

    length = np.broadcast_to(length[..., None], (num_agents, num_rollouts, num_steps))
    width = np.broadcast_to(width[..., None], (num_agents, num_rollouts, num_steps))

    boxes = np.stack([center_x, center_y, length, width, heading], axis=-1)

    shrinking_distance = np.minimum(boxes[:, :, :, 2], boxes[:, :, :, 3]) * corner_rounding_factor / 2.0

    boxes = np.concatenate(
        [
            boxes[:, :, :, :2],
            boxes[:, :, :, 2:3] - 2.0 * shrinking_distance[..., np.newaxis],
            boxes[:, :, :, 3:4] - 2.0 * shrinking_distance[..., np.newaxis],
            boxes[:, :, :, 4:],
        ],
        axis=3,
    )

    boxes_flat = boxes.reshape(num_agents * num_rollouts * num_steps, 5)
    box_corners = geometry_utils.get_2d_box_corners(boxes_flat)
    box_corners = box_corners.reshape(num_agents, num_rollouts, num_steps, 4, 2)

    corners_1 = box_corners[:, np.newaxis, :, :, :, :]
    corners_2 = box_corners[np.newaxis, :, :, :, :, :]

    corners_broadcast_1 = np.broadcast_to(corners_1, (num_agents, num_agents, num_rollouts, num_steps, 4, 2))
    corners_broadcast_2 = np.broadcast_to(corners_2, (num_agents, num_agents, num_rollouts, num_steps, 4, 2))

    batch_size = num_agents * num_agents * num_rollouts * num_steps
    corners_flat_1 = corners_broadcast_1.reshape(batch_size, 4, 2)
    corners_flat_2 = corners_broadcast_2.reshape(batch_size, 4, 2)

    neg_corners_2 = -1.0 * corners_flat_2
    minkowski_sum = geometry_utils.minkowski_sum_of_box_and_box_points(corners_flat_1, neg_corners_2)

    signed_distances_flat = geometry_utils.signed_distance_from_point_to_convex_polygon(
        query_points=np.zeros((batch_size, 2), dtype=np.float32), polygon_points=minkowski_sum
    )

    signed_distances = signed_distances_flat.reshape(num_agents, num_agents, num_rollouts, num_steps)

    signed_distances -= shrinking_distance[:, np.newaxis, :, :]
    signed_distances -= shrinking_distance[np.newaxis, :, :, :]

    self_mask = np.eye(num_agents, dtype=np.float32)[:, :, np.newaxis, np.newaxis]
    signed_distances = signed_distances + self_mask * EXTREMELY_LARGE_DISTANCE

    valid_1 = valid[:, np.newaxis, :, :]
    valid_2 = valid[np.newaxis, :, :, :]
    valid_mask = np.logical_and(valid_1, valid_2)
    signed_distances = np.where(valid_mask, signed_distances, EXTREMELY_LARGE_DISTANCE)

    return signed_distances


def compute_distance_to_nearest_object(
    center_x: np.ndarray,
    center_y: np.ndarray,
    length: np.ndarray,
    width: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    signed_distances = compute_signed_distances(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        corner_rounding_factor=corner_rounding_factor,
    )

    min_distances = np.min(signed_distances, axis=1)
    return min_distances


def compute_time_to_collision(
    center_x: np.ndarray,
    center_y: np.ndarray,
    length: np.ndarray,
    width: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    seconds_per_step: float,
) -> np.ndarray:
    """Computes time-to-collision of the evaluated objects.

    The time-to-collision measures, in seconds, the time until an object collides
    with the object it is following, assuming constant speeds.

    Args:
        center_x: Shape (num_agents, num_rollouts, num_steps)
        center_y: Shape (num_agents, num_rollouts, num_steps)
        length: Shape (num_agents, num_rollouts) - constant per timestep
        width: Shape (num_agents, num_rollouts) - constant per timestep
        heading: Shape (num_agents, num_rollouts, num_steps)
        valid: Shape (num_agents, num_rollouts, num_steps)
        seconds_per_step: Duration of one step in seconds

    Returns:
        Time-to-collision, shape (num_agents, num_rollouts, num_steps)
    """
    from pufferlib.ocean.benchmark import metrics

    num_agents = center_x.shape[0]
    num_rollouts = center_x.shape[1]
    num_steps = center_x.shape[2]

    speed = metrics.compute_kinematic_features(
        x=center_x, y=center_y, heading=heading, seconds_per_step=seconds_per_step
    )[0]

    length_broadcast = np.broadcast_to(length[..., None], (num_agents, num_rollouts, num_steps))
    width_broadcast = np.broadcast_to(width[..., None], (num_agents, num_rollouts, num_steps))

    boxes = np.stack([center_x, center_y, length_broadcast, width_broadcast, heading, speed], axis=-1)
    # (num_agents, num_rollouts, num_steps, 6) -> (num_steps, num_agents, num_rollouts, 6)
    boxes = np.transpose(boxes, (2, 0, 1, 3))

    # (num_agents, num_rollouts, num_steps) -> (num_steps, num_agents, num_rollouts)
    valid_transposed = np.transpose(valid, (2, 0, 1))

    # Split box features: xy (2), sizes (2), yaw (1), speed (1)
    # Each has shape (num_steps, num_agents, num_rollouts, feature_dim)
    ego_xy, ego_sizes, ego_yaw, ego_speed = np.split(boxes, [2, 4, 5], axis=-1)
    other_xy, other_sizes, other_yaw, _ = np.split(boxes, [2, 4, 5], axis=-1)

    # Compute pairwise yaw differences
    # (num_steps, 1, num_agents, num_rollouts, 1) - (num_steps, num_agents, 1, num_rollouts, 1)
    # -> (num_steps, num_agents, num_agents, num_rollouts, 1)
    yaw_diff = np.abs(other_yaw[:, np.newaxis] - ego_yaw[:, :, np.newaxis])

    yaw_diff_cos = np.cos(yaw_diff)
    yaw_diff_sin = np.sin(yaw_diff)

    # Longitudinal and lateral offsets from other box center to closest corner
    # (num_steps, 1, num_agents, num_rollouts, 2) -> (num_steps, num_agents, num_agents, num_rollouts)
    other_long_offset = geometry_utils.dot_product_2d(
        other_sizes[:, np.newaxis] / 2.0,
        np.abs(np.concatenate([yaw_diff_cos, yaw_diff_sin], axis=-1)),
    )
    other_lat_offset = geometry_utils.dot_product_2d(
        other_sizes[:, np.newaxis] / 2.0,
        np.abs(np.concatenate([yaw_diff_sin, yaw_diff_cos], axis=-1)),
    )

    # Transform other boxes to ego-relative coordinates
    # (num_steps, num_agents, num_agents, num_rollouts, 2)
    other_relative_xy = geometry_utils.rotate_2d_points(
        (other_xy[:, np.newaxis] - ego_xy[:, :, np.newaxis]),
        -ego_yaw[:, :, np.newaxis, :, 0],
    )

    # Longitudinal distance from ego front to other box back
    # (num_steps, num_agents, num_agents, num_rollouts)
    long_distance = other_relative_xy[..., 0] - ego_sizes[:, :, np.newaxis, :, 0] / 2.0 - other_long_offset

    # Lateral overlap (negative means overlap exists)
    # (num_steps, num_agents, num_agents, num_rollouts)
    lat_overlap = np.abs(other_relative_xy[..., 1]) - ego_sizes[:, :, np.newaxis, :, 1] / 2.0 - other_lat_offset

    # Check following criteria
    # (num_steps, num_agents, num_agents, num_rollouts) -> (num_agents, num_agents, num_rollouts, num_steps)
    following_mask = _get_object_following_mask(
        long_distance.transpose(1, 2, 0, 3),
        lat_overlap.transpose(1, 2, 0, 3),
        yaw_diff[..., 0].transpose(1, 2, 0, 3),
    )

    # (num_steps, 1, num_agents, num_agents, num_rollouts)
    valid_mask = np.logical_and(valid_transposed[:, np.newaxis], following_mask.transpose(2, 0, 1, 3))
    # Mask out invalid or non-following objects with large distance
    # (num_steps, num_agents, num_agents, num_rollouts)
    masked_long_distance = long_distance + (1.0 - valid_mask.astype(np.float32)) * EXTREMELY_LARGE_DISTANCE

    # Find nearest object ahead for each ego agent
    # (num_steps, num_agents, num_rollouts)
    box_ahead_index = np.argmin(masked_long_distance, axis=-2)
    distance_to_box_ahead = np.take_along_axis(masked_long_distance, box_ahead_index[:, :, np.newaxis, :], axis=-2)[
        ..., 0, :
    ]

    # Get speed of the object ahead
    # speed: (num_agents, num_rollouts, num_steps) -> (num_steps, 1, num_agents, num_rollouts)
    # broadcasts to (num_steps, num_agents, num_agents, num_rollouts)
    box_ahead_speed = np.take_along_axis(
        np.broadcast_to(
            np.transpose(speed, (2, 0, 1))[:, np.newaxis, :, :],
            masked_long_distance.shape,
        ),
        box_ahead_index[:, :, np.newaxis, :],
        axis=-2,
    )[..., 0, :]

    # Compute TTC = distance / relative_speed
    # (num_steps, num_agents, num_rollouts)
    rel_speed = ego_speed[..., 0] - box_ahead_speed

    # Trick to avoid division by zero
    rel_speed_safe = np.where(rel_speed > 0.0, rel_speed, 1.0)
    time_to_collision = np.where(
        rel_speed > 0.0,
        np.minimum(distance_to_box_ahead / rel_speed_safe, MAXIMUM_TIME_TO_COLLISION),
        MAXIMUM_TIME_TO_COLLISION,
    )
    # (num_steps, num_agents, num_rollouts) -> (num_agents, num_rollouts, num_steps)
    return np.transpose(time_to_collision, (1, 2, 0))


def _get_object_following_mask(
    longitudinal_distance: np.ndarray,
    lateral_overlap: np.ndarray,
    yaw_diff: np.ndarray,
) -> np.ndarray:
    """Checks whether objects satisfy criteria for following another object.

    Args:
        longitudinal_distance: Shape (num_agents, num_agents, num_rollouts, num_steps)
            Longitudinal distances from back side of each ego box to other boxes.
        lateral_overlap: Shape (num_agents, num_agents, num_rollouts, num_steps)
            Lateral overlaps of other boxes over trails of ego boxes.
        yaw_diff: Shape (num_agents, num_agents, num_rollouts, num_steps)
            Absolute yaw differences between egos and other boxes.

    Returns:
        Boolean array indicating for each ego box if it is following the other boxes.
        Shape (num_agents, num_agents, num_rollouts, num_steps)
    """
    valid_mask = longitudinal_distance > 0.0
    valid_mask = np.logical_and(valid_mask, yaw_diff <= MAX_HEADING_DIFF)
    valid_mask = np.logical_and(valid_mask, lateral_overlap < 0.0)
    return np.logical_and(
        valid_mask,
        np.logical_or(
            lateral_overlap < -SMALL_OVERLAP_THRESHOLD,
            yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP,
        ),
    )
