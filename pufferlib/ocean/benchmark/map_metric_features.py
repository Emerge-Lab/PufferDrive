"""Map-based metric features for WOSAC evaluation.

Adapted from Waymo Open Dataset:
https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/map_metric_features.py
"""

import numpy as np
from pufferlib.ocean.benchmark.geometry_utils import (
    get_2d_box_corners,
    cross_product_2d,
    dot_product_2d,
)

EXTREMELY_LARGE_DISTANCE = 1e10
OFFROAD_DISTANCE_THRESHOLD = 0.0


def compute_distance_to_road_edge(
    center_x: np.ndarray,
    center_y: np.ndarray,
    length: np.ndarray,
    width: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
    polyline_x: np.ndarray,
    polyline_y: np.ndarray,
    polyline_lengths: np.ndarray,
) -> np.ndarray:
    """Computes signed distance to road edge for each agent at each timestep.

    Args:
        center_x: Shape (num_agents, num_steps)
        center_y: Shape (num_agents, num_steps)
        length: Shape (num_agents,) or (num_agents, num_steps)
        width: Shape (num_agents,) or (num_agents, num_steps)
        heading: Shape (num_agents, num_steps)
        valid: Shape (num_agents, num_steps) boolean
        polyline_x: Flattened x coordinates of all polyline points
        polyline_y: Flattened y coordinates of all polyline points
        polyline_lengths: Length of each polyline

    Returns:
        Signed distances, shape (num_agents, num_steps).
        Negative = on-road, positive = off-road.
    """
    num_agents, num_steps = center_x.shape

    if length.ndim == 1:
        length = np.broadcast_to(length[:, np.newaxis], (num_agents, num_steps))
    if width.ndim == 1:
        width = np.broadcast_to(width[:, np.newaxis], (num_agents, num_steps))

    boxes = np.stack([center_x, center_y, length, width, heading], axis=-1)
    boxes_flat = boxes.reshape(-1, 5)

    corners = get_2d_box_corners(boxes_flat)
    corners = corners.reshape(num_agents, num_steps, 4, 2)

    flat_corners = corners.reshape(-1, 2)

    polylines_padded, polylines_valid = _pad_polylines(
        polyline_x, polyline_y, polyline_lengths
    )

    corner_distances = _compute_signed_distance_to_polylines(
        flat_corners, polylines_padded, polylines_valid
    )

    corner_distances = corner_distances.reshape(num_agents, num_steps, 4)
    signed_distances = np.max(corner_distances, axis=-1)

    signed_distances = np.where(valid, signed_distances, -EXTREMELY_LARGE_DISTANCE)

    return signed_distances


def _pad_polylines(
    polyline_x: np.ndarray,
    polyline_y: np.ndarray,
    polyline_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert flattened polylines to padded tensor format.

    Returns:
        polylines: Shape (num_polylines, max_length, 2)
        valid: Shape (num_polylines, max_length)
    """
    num_polylines = len(polyline_lengths)
    max_length = polyline_lengths.max()

    polylines = np.zeros((num_polylines, max_length, 2), dtype=np.float32)
    valid = np.zeros((num_polylines, max_length), dtype=bool)

    boundaries = np.cumsum(np.concatenate([[0], polyline_lengths]))

    for i in range(num_polylines):
        start, end = boundaries[i], boundaries[i + 1]
        length = polyline_lengths[i]
        polylines[i, :length, 0] = polyline_x[start:end]
        polylines[i, :length, 1] = polyline_y[start:end]
        valid[i, :length] = True

    return polylines, valid


def _compute_signed_distance_to_polylines(
    xys: np.ndarray,
    polylines: np.ndarray,
    polylines_valid: np.ndarray,
) -> np.ndarray:
    """Computes signed distance from points to polylines (2D).

    Args:
        xys: Shape (num_points, 2)
        polylines: Shape (num_polylines, max_length, 2)
        polylines_valid: Shape (num_polylines, max_length)

    Returns:
        Signed distances, shape (num_points,).
        Negative = on-road (port side), positive = off-road (starboard).
    """
    num_points = xys.shape[0]
    num_polylines, max_length = polylines.shape[:2]
    num_segments = max_length - 1

    is_segment_valid = polylines_valid[:, :-1] & polylines_valid[:, 1:]

    xy_starts = polylines[:, :-1, :]
    xy_ends = polylines[:, 1:, :]
    start_to_end = xy_ends - xy_starts

    start_to_point = xys[np.newaxis, np.newaxis, :, :] - xy_starts[:, :, np.newaxis, :]

    dot_se_se = dot_product_2d(start_to_end, start_to_end)
    dot_sp_se = dot_product_2d(
        start_to_point,
        start_to_end[:, :, np.newaxis, :]
    )
    rel_t = np.divide(dot_sp_se, dot_se_se[:, :, np.newaxis],
                      out=np.zeros_like(dot_sp_se),
                      where=dot_se_se[:, :, np.newaxis] != 0)

    n = np.sign(cross_product_2d(
        start_to_point,
        start_to_end[:, :, np.newaxis, :]
    ))

    segment_to_point = start_to_point - (
        start_to_end[:, :, np.newaxis, :] * np.clip(rel_t, 0.0, 1.0)[:, :, :, np.newaxis]
    )
    distance_to_segment_2d = np.linalg.norm(segment_to_point, axis=-1)

    start_to_end_padded = np.concatenate([
        start_to_end[:, -1:, :],
        start_to_end,
        start_to_end[:, :1, :],
    ], axis=1)

    is_locally_convex = cross_product_2d(
        start_to_end_padded[:, :-1, np.newaxis, :],
        start_to_end_padded[:, 1:, np.newaxis, :]
    ) > 0.0

    n_prior = np.concatenate([n[:, :1, :], n[:, :-1, :]], axis=1)
    n_next = np.concatenate([n[:, 1:, :], n[:, -1:, :]], axis=1)

    is_prior_valid = np.concatenate([
        is_segment_valid[:, :1],
        is_segment_valid[:, :-1]
    ], axis=1)
    is_next_valid = np.concatenate([
        is_segment_valid[:, 1:],
        is_segment_valid[:, -1:]
    ], axis=1)

    sign_if_before = np.where(
        is_locally_convex[:, :-1, :],
        np.maximum(n, n_prior),
        np.minimum(n, n_prior),
    )
    sign_if_after = np.where(
        is_locally_convex[:, 1:, :],
        np.maximum(n, n_next),
        np.minimum(n, n_next),
    )

    sign_to_segment = np.where(
        (rel_t < 0.0) & is_prior_valid[:, :, np.newaxis],
        sign_if_before,
        np.where(
            (rel_t > 1.0) & is_next_valid[:, :, np.newaxis],
            sign_if_after,
            n
        )
    )

    distance_to_segment_2d = distance_to_segment_2d.reshape(
        num_polylines * num_segments, num_points
    ).T
    sign_to_segment = sign_to_segment.reshape(
        num_polylines * num_segments, num_points
    ).T

    is_segment_valid_flat = is_segment_valid.reshape(num_polylines * num_segments)
    distance_to_segment_2d = np.where(
        is_segment_valid_flat[np.newaxis, :],
        distance_to_segment_2d,
        EXTREMELY_LARGE_DISTANCE,
    )

    closest_idx = np.argmin(distance_to_segment_2d, axis=1)
    distance_2d = distance_to_segment_2d[np.arange(num_points), closest_idx]
    distance_sign = sign_to_segment[np.arange(num_points), closest_idx]

    return distance_sign * distance_2d
