"""Geometry utilities for distance computation between 2D boxes.

Adapted from:
- https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/box_utils.py
- https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/geometry_utils.py
"""

import numpy as np
from typing import Tuple

NUM_VERTICES_IN_BOX = 4


def get_yaw_rotation_2d(heading: np.ndarray) -> np.ndarray:
    """Gets 2D rotation matrices from heading angles.

    Args:
        heading: Rotation angles in radians, any shape

    Returns:
        Rotation matrices, shape [..., 2, 2]
    """
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)

    return np.stack([
        np.stack([cos_heading, -sin_heading], axis=-1),
        np.stack([sin_heading, cos_heading], axis=-1)
    ], axis=-2)


def cross_product_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes signed magnitude of cross product of 2D vectors.

    Args:
        a: Array with shape (..., 2)
        b: Array with same shape as a

    Returns:
        Cross product a[0]*b[1] - a[1]*b[0], shape (...)
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _get_downmost_edge_in_box(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the downmost (lowest y-coordinate) edge in the box.

    Assumes box edges are given in counter-clockwise order.

    Args:
        box: Array of shape (num_boxes, num_points_per_box, 2) with x-y coordinates

    Returns:
        Tuple of:
            - downmost_vertex_idx: Index of downmost vertex, shape (num_boxes, 1)
            - downmost_edge_direction: Tangent unit vector of downmost edge, shape (num_boxes, 1, 2)
    """
    downmost_vertex_idx = np.argmin(box[..., 1], axis=-1)[:, np.newaxis]

    edge_start_vertex = np.take_along_axis(box, downmost_vertex_idx[:, :, np.newaxis], axis=1)
    edge_end_idx = np.mod(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
    edge_end_vertex = np.take_along_axis(box, edge_end_idx[:, :, np.newaxis], axis=1)

    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = np.linalg.norm(downmost_edge, axis=-1)
    downmost_edge_direction = downmost_edge / downmost_edge_length[:, :, np.newaxis]

    return downmost_vertex_idx, downmost_edge_direction


def _get_edge_info(polygon_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes properties about the edges of a polygon.

    Args:
        polygon_points: Vertices of each polygon, shape (num_polygons, num_points_per_polygon, 2)

    Returns:
        Tuple of:
            - tangent_unit_vectors: Shape (num_polygons, num_points_per_polygon, 2)
            - normal_unit_vectors: Shape (num_polygons, num_points_per_polygon, 2)
            - edge_lengths: Shape (num_polygons, num_points_per_polygon)
    """
    first_point_in_polygon = polygon_points[:, 0:1, :]
    shifted_polygon_points = np.concatenate(
        [polygon_points[:, 1:, :], first_point_in_polygon], axis=-2
    )
    edge_vectors = shifted_polygon_points - polygon_points

    edge_lengths = np.linalg.norm(edge_vectors, axis=-1)
    tangent_unit_vectors = edge_vectors / edge_lengths[:, :, np.newaxis]
    normal_unit_vectors = np.stack(
        [-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], axis=-1
    )

    return tangent_unit_vectors, normal_unit_vectors, edge_lengths


def get_2d_box_corners(boxes: np.ndarray) -> np.ndarray:
    """Given a set of 2D boxes, return its 4 corners.

    Args:
        boxes: Array of shape [..., 5] with [center_x, center_y, length, width, heading]

    Returns:
        Corners array of shape [..., 4, 2] in counter-clockwise order
    """
    center_x, center_y, length, width, heading = np.split(boxes, 5, axis=-1)
    center_x = center_x[..., 0]
    center_y = center_y[..., 0]
    length = length[..., 0]
    width = width[..., 0]
    heading = heading[..., 0]

    rotation = get_yaw_rotation_2d(heading)
    translation = np.stack([center_x, center_y], axis=-1)

    l2 = length * 0.5
    w2 = width * 0.5

    corners = np.reshape(
        np.stack([
            l2, w2, -l2, w2, -l2, -w2, l2, -w2
        ], axis=-1),
        boxes.shape[:-1] + (4, 2)
    )

    corners = np.einsum('...ij,...kj->...ki', rotation, corners) + np.expand_dims(translation, axis=-2)

    return corners


def minkowski_sum_of_box_and_box_points(box1_points: np.ndarray, box2_points: np.ndarray) -> np.ndarray:
    """Batched Minkowski sum of two boxes (counter-clockwise corners in xy).

    Args:
        box1_points: Vertices for box 1, shape (num_boxes, 4, 2)
        box2_points: Vertices for box 2, shape (num_boxes, 4, 2)

    Returns:
        Minkowski sum of the two boxes, shape (num_boxes, 8, 2), in counter-clockwise order
    """
    point_order_1 = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    point_order_2 = np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int64)

    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(box1_points)
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(box2_points)

    condition = (
        cross_product_2d(downmost_box1_edge_direction, downmost_box2_edge_direction) >= 0.0
    )
    condition = np.tile(condition, [1, 8])

    box1_point_order = np.where(condition, point_order_2, point_order_1)
    box1_point_order = np.mod(box1_point_order + box1_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box1_points = np.take_along_axis(
        box1_points, box1_point_order[:, :, np.newaxis], axis=1
    )

    box2_point_order = np.where(condition, point_order_1, point_order_2)
    box2_point_order = np.mod(box2_point_order + box2_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box2_points = np.take_along_axis(
        box2_points, box2_point_order[:, :, np.newaxis], axis=1
    )

    minkowski_sum = ordered_box1_points + ordered_box2_points

    return minkowski_sum


def dot_product_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Computes the dot product of 2D vectors.

    Args:
        a: Array with shape (..., 2)
        b: Array with same shape as a

    Returns:
        Dot product a[0]*b[0] + a[1]*b[1], shape (...)
    """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def rotate_2d_points(xys: np.ndarray, rotation_yaws: np.ndarray) -> np.ndarray:
    """Rotates xys counter-clockwise using rotation_yaws.

    Rotates about the origin counter-clockwise in the x-y plane.

    Args:
        xys: Array with shape (..., 2) containing xy coordinates
        rotation_yaws: Array with shape (..., 1) containing angles in radians

    Returns:
        Rotated xys, shape (..., 2)
    """
    rel_cos_yaws = np.cos(rotation_yaws)
    rel_sin_yaws = np.sin(rotation_yaws)
    xs_out = rel_cos_yaws * xys[..., 0] - rel_sin_yaws * xys[..., 1]
    ys_out = rel_sin_yaws * xys[..., 0] + rel_cos_yaws * xys[..., 1]
    return np.stack([xs_out, ys_out], axis=-1)


def signed_distance_from_point_to_convex_polygon(
    query_points: np.ndarray, polygon_points: np.ndarray
) -> np.ndarray:
    """Finds signed distances from query points to convex polygons.

    Vertices must be ordered counter-clockwise.

    Args:
        query_points: Shape (batch_size, 2) with x-y coordinates
        polygon_points: Shape (batch_size, num_points_per_polygon, 2) with x-y coordinates

    Returns:
        Signed distances, shape (batch_size,). Negative if point is inside polygon.
    """
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = _get_edge_info(polygon_points)

    query_points = np.expand_dims(query_points, axis=1)
    vertices_to_query_vectors = query_points - polygon_points
    vertices_distances = np.linalg.norm(vertices_to_query_vectors, axis=-1)

    edge_signed_perp_distances = np.sum(
        -normal_unit_vectors * vertices_to_query_vectors, axis=-1
    )

    is_inside = np.all(edge_signed_perp_distances <= 0, axis=-1)

    projection_along_tangent = np.sum(
        tangent_unit_vectors * vertices_to_query_vectors, axis=-1
    )
    projection_along_tangent_proportion = np.divide(
        projection_along_tangent, edge_lengths
    )

    is_projection_on_edge = np.logical_and(
        projection_along_tangent_proportion >= 0.0,
        projection_along_tangent_proportion <= 1.0
    )

    edge_perp_distances = np.abs(edge_signed_perp_distances)
    edge_distances = np.where(is_projection_on_edge, edge_perp_distances, np.inf)

    edge_and_vertex_distance = np.concatenate(
        [edge_distances, vertices_distances], axis=-1
    )

    min_distance = np.min(edge_and_vertex_distance, axis=-1)
    signed_distances = np.where(is_inside, -min_distance, min_distance)

    return signed_distances
