"""Geometry utilities for distance computation between 2D boxes.

Adapted from:
- https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/box_utils.py
- https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/utils/geometry_utils.py
"""

import torch
from typing import Tuple

NUM_VERTICES_IN_BOX = 4


def get_yaw_rotation_2d(heading: torch.Tensor) -> torch.Tensor:
    """Gets 2D rotation matrices from heading angles.

    Args:
        heading: Rotation angles in radians, any shape

    Returns:
        Rotation matrices, shape [..., 2, 2]
    """
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)

    return torch.stack(
        [torch.stack([cos_heading, -sin_heading], dim=-1), torch.stack([sin_heading, cos_heading], dim=-1)], dim=-2
    )


def cross_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes signed magnitude of cross product of 2D vectors.

    Args:
        a: Tensor with shape (..., 2)
        b: Tensor with same shape as a

    Returns:
        Cross product a[0]*b[1] - a[1]*b[0], shape (...)
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _get_downmost_edge_in_box(box: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds the downmost (lowest y-coordinate) edge in the box.

    Assumes box edges are given in counter-clockwise order.

    Args:
        box: Tensor of shape (num_boxes, num_points_per_box, 2) with x-y coordinates

    Returns:
        Tuple of:
            - downmost_vertex_idx: Index of downmost vertex, shape (num_boxes, 1)
            - downmost_edge_direction: Tangent unit vector of downmost edge, shape (num_boxes, 1, 2)
    """
    downmost_vertex_idx = torch.argmin(box[..., 1], dim=-1).unsqueeze(-1)

    edge_start_vertex = torch.gather(box, 1, downmost_vertex_idx.unsqueeze(-1).expand(-1, -1, 2))
    edge_end_idx = torch.remainder(downmost_vertex_idx + 1, NUM_VERTICES_IN_BOX)
    edge_end_vertex = torch.gather(box, 1, edge_end_idx.unsqueeze(-1).expand(-1, -1, 2))

    downmost_edge = edge_end_vertex - edge_start_vertex
    downmost_edge_length = torch.linalg.norm(downmost_edge, dim=-1)
    downmost_edge_direction = downmost_edge / downmost_edge_length.unsqueeze(-1)

    return downmost_vertex_idx, downmost_edge_direction


def _get_edge_info(polygon_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    shifted_polygon_points = torch.cat([polygon_points[:, 1:, :], first_point_in_polygon], dim=-2)
    edge_vectors = shifted_polygon_points - polygon_points

    edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)
    tangent_unit_vectors = edge_vectors / edge_lengths.unsqueeze(-1)
    normal_unit_vectors = torch.stack([-tangent_unit_vectors[..., 1], tangent_unit_vectors[..., 0]], dim=-1)

    return tangent_unit_vectors, normal_unit_vectors, edge_lengths


def get_2d_box_corners(boxes: torch.Tensor) -> torch.Tensor:
    """Given a set of 2D boxes, return its 4 corners.

    Args:
        boxes: Tensor of shape [..., 5] with [center_x, center_y, length, width, heading]

    Returns:
        Corners tensor of shape [..., 4, 2] in counter-clockwise order
    """
    center_x = boxes[..., 0]
    center_y = boxes[..., 1]
    length = boxes[..., 2]
    width = boxes[..., 3]
    heading = boxes[..., 4]

    rotation = get_yaw_rotation_2d(heading)
    translation = torch.stack([center_x, center_y], dim=-1)

    l2 = length * 0.5
    w2 = width * 0.5

    corners = torch.stack([l2, w2, -l2, w2, -l2, -w2, l2, -w2], dim=-1).reshape(boxes.shape[:-1] + (4, 2))

    corners = torch.einsum("...ij,...kj->...ki", rotation, corners) + translation.unsqueeze(-2)

    return corners


def minkowski_sum_of_box_and_box_points(box1_points: torch.Tensor, box2_points: torch.Tensor) -> torch.Tensor:
    """Batched Minkowski sum of two boxes (counter-clockwise corners in xy).

    Args:
        box1_points: Vertices for box 1, shape (num_boxes, 4, 2)
        box2_points: Vertices for box 2, shape (num_boxes, 4, 2)

    Returns:
        Minkowski sum of the two boxes, shape (num_boxes, 8, 2), in counter-clockwise order
    """
    device = box1_points.device
    point_order_1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64, device=device)
    point_order_2 = torch.tensor([0, 1, 1, 2, 2, 3, 3, 0], dtype=torch.int64, device=device)

    box1_start_idx, downmost_box1_edge_direction = _get_downmost_edge_in_box(box1_points)
    box2_start_idx, downmost_box2_edge_direction = _get_downmost_edge_in_box(box2_points)

    condition = cross_product_2d(downmost_box1_edge_direction, downmost_box2_edge_direction) >= 0.0
    condition = condition.repeat(1, 8)

    box1_point_order = torch.where(condition, point_order_2, point_order_1)
    box1_point_order = torch.remainder(box1_point_order + box1_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box1_points = torch.gather(box1_points, 1, box1_point_order.unsqueeze(-1).expand(-1, -1, 2))

    box2_point_order = torch.where(condition, point_order_1, point_order_2)
    box2_point_order = torch.remainder(box2_point_order + box2_start_idx, NUM_VERTICES_IN_BOX)
    ordered_box2_points = torch.gather(box2_points, 1, box2_point_order.unsqueeze(-1).expand(-1, -1, 2))

    minkowski_sum = ordered_box1_points + ordered_box2_points

    return minkowski_sum


def dot_product_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Computes the dot product of 2D vectors.

    Args:
        a: Tensor with shape (..., 2)
        b: Tensor with same shape as a

    Returns:
        Dot product a[0]*b[0] + a[1]*b[1], shape (...)
    """
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]


def rotate_2d_points(xys: torch.Tensor, rotation_yaws: torch.Tensor) -> torch.Tensor:
    """Rotates xys counter-clockwise using rotation_yaws.

    Rotates about the origin counter-clockwise in the x-y plane.

    Args:
        xys: Tensor with shape (..., 2) containing xy coordinates
        rotation_yaws: Tensor with shape (...) containing angles in radians

    Returns:
        Rotated xys, shape (..., 2)
    """
    rel_cos_yaws = torch.cos(rotation_yaws)
    rel_sin_yaws = torch.sin(rotation_yaws)
    xs_out = rel_cos_yaws * xys[..., 0] - rel_sin_yaws * xys[..., 1]
    ys_out = rel_sin_yaws * xys[..., 0] + rel_cos_yaws * xys[..., 1]
    return torch.stack([xs_out, ys_out], dim=-1)


def signed_distance_from_point_to_convex_polygon(
    query_points: torch.Tensor, polygon_points: torch.Tensor
) -> torch.Tensor:
    """Finds signed distances from query points to convex polygons.

    Vertices must be ordered counter-clockwise.

    Args:
        query_points: Shape (batch_size, 2) with x-y coordinates
        polygon_points: Shape (batch_size, num_points_per_polygon, 2) with x-y coordinates

    Returns:
        Signed distances, shape (batch_size,). Negative if point is inside polygon.
    """
    tangent_unit_vectors, normal_unit_vectors, edge_lengths = _get_edge_info(polygon_points)

    query_points = query_points.unsqueeze(1)
    vertices_to_query_vectors = query_points - polygon_points
    vertices_distances = torch.linalg.norm(vertices_to_query_vectors, dim=-1)

    edge_signed_perp_distances = torch.sum(-normal_unit_vectors * vertices_to_query_vectors, dim=-1)

    is_inside = torch.all(edge_signed_perp_distances <= 0, dim=-1)

    projection_along_tangent = torch.sum(tangent_unit_vectors * vertices_to_query_vectors, dim=-1)
    projection_along_tangent_proportion = projection_along_tangent / edge_lengths

    is_projection_on_edge = (projection_along_tangent_proportion >= 0.0) & (projection_along_tangent_proportion <= 1.0)

    edge_perp_distances = torch.abs(edge_signed_perp_distances)
    edge_distances = torch.where(
        is_projection_on_edge, edge_perp_distances, torch.tensor(float("inf"), device=query_points.device)
    )

    edge_and_vertex_distance = torch.cat([edge_distances, vertices_distances], dim=-1)

    min_distance = torch.min(edge_and_vertex_distance, dim=-1).values
    signed_distances = torch.where(is_inside, -min_distance, min_distance)

    return signed_distances
