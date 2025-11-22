"""Map-based metric features for WOSAC evaluation.

Adapted from Waymo Open Dataset:
https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/map_metric_features.py
"""

import torch

from pufferlib.ocean.benchmark.geometry_utils import (
    get_2d_box_corners,
    cross_product_2d,
    dot_product_2d,
)

EXTREMELY_LARGE_DISTANCE = 1e10
OFFROAD_DISTANCE_THRESHOLD = 0.0


def compute_distance_to_road_edge(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor,
    polyline_x: torch.Tensor,
    polyline_y: torch.Tensor,
    polyline_lengths: torch.Tensor,
) -> torch.Tensor:
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
        length = length.unsqueeze(-1).expand(-1, num_steps)
    if width.ndim == 1:
        width = width.unsqueeze(-1).expand(-1, num_steps)

    boxes = torch.stack([center_x, center_y, length, width, heading], dim=-1)
    boxes_flat = boxes.reshape(-1, 5)

    corners = get_2d_box_corners(boxes_flat)
    corners = corners.reshape(num_agents, num_steps, 4, 2)

    flat_corners = corners.reshape(-1, 2)

    polylines_padded, polylines_valid = _pad_polylines(polyline_x, polyline_y, polyline_lengths)

    corner_distances = _compute_signed_distance_to_polylines(flat_corners, polylines_padded, polylines_valid)

    corner_distances = corner_distances.reshape(num_agents, num_steps, 4)
    signed_distances = torch.max(corner_distances, dim=-1).values

    offroad_fill = signed_distances.new_full((), -EXTREMELY_LARGE_DISTANCE)
    signed_distances = torch.where(valid, signed_distances, offroad_fill)

    return signed_distances


def _pad_polylines(
    polyline_x: torch.Tensor,
    polyline_y: torch.Tensor,
    polyline_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert flattened polylines to padded tensor format.

    Returns:
        polylines: Shape (num_polylines, max_length, 2)
        valid: Shape (num_polylines, max_length)
    """
    device = polyline_x.device
    num_polylines = polyline_lengths.shape[0]
    max_length = int(polyline_lengths.max().item())

    polylines = torch.zeros((num_polylines, max_length, 2), dtype=torch.float32, device=device)
    valid = torch.zeros((num_polylines, max_length), dtype=torch.bool, device=device)

    lengths_long = polyline_lengths.to(torch.long)
    boundaries = torch.cumsum(torch.cat([lengths_long.new_zeros(1), lengths_long]), dim=0)

    for i in range(num_polylines):
        start = int(boundaries[i].item())
        end = int(boundaries[i + 1].item())
        length_i = int(lengths_long[i].item())
        polylines[i, :length_i, 0] = polyline_x[start:end]
        polylines[i, :length_i, 1] = polyline_y[start:end]
        valid[i, :length_i] = True

    return polylines, valid


def _check_polyline_cycles(
    polylines: torch.Tensor,
    polylines_valid: torch.Tensor,
    tolerance: float = 1e-3,
) -> torch.Tensor:
    """Check if polylines are cyclic (first point == last point).

    Args:
        polylines: Shape (num_polylines, max_length, 2)
        polylines_valid: Shape (num_polylines, max_length)
        tolerance: Distance threshold for considering points equal

    Returns:
        Boolean array of shape (num_polylines,)
    """
    device = polylines.device
    max_length = polylines.shape[1]
    valid_counts = polylines_valid.sum(dim=-1)
    has_enough_points = valid_counts >= 2

    indices = torch.arange(max_length, device=device)
    last_idx = torch.argmax(polylines_valid.int() * indices, dim=-1)

    first_pts = polylines[:, 0]
    gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, 2)
    last_pts = torch.gather(polylines, 1, gather_idx).squeeze(1)
    dist = torch.linalg.norm(first_pts - last_pts, dim=-1)

    return (dist < tolerance) & has_enough_points


def _compute_signed_distance_to_polylines(
    xys: torch.Tensor,
    polylines: torch.Tensor,
    polylines_valid: torch.Tensor,
) -> torch.Tensor:
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
    is_polyline_cyclic = _check_polyline_cycles(polylines, polylines_valid)

    xy_starts = polylines[:, :-1, :]
    xy_ends = polylines[:, 1:, :]
    start_to_end = xy_ends - xy_starts

    start_to_point = xys.unsqueeze(0).unsqueeze(0) - xy_starts[:, :, None, :]

    dot_se_se = dot_product_2d(start_to_end, start_to_end)
    dot_sp_se = dot_product_2d(start_to_point, start_to_end[:, :, None, :])

    denom = dot_se_se[:, :, None]
    rel_t = torch.where(
        denom != 0,
        dot_sp_se / denom,
        torch.zeros_like(dot_sp_se),
    )

    n = torch.sign(cross_product_2d(start_to_point, start_to_end[:, :, None, :]))

    segment_to_point = start_to_point - (start_to_end[:, :, None, :] * torch.clamp(rel_t, 0.0, 1.0)[:, :, :, None])
    distance_to_segment_2d = torch.linalg.norm(segment_to_point, dim=-1)

    start_to_end_padded = torch.cat(
        [
            start_to_end[:, -1:, :],
            start_to_end,
            start_to_end[:, :1, :],
        ],
        dim=1,
    )

    is_locally_convex = (
        cross_product_2d(start_to_end_padded[:, :-1, None, :], start_to_end_padded[:, 1:, None, :]) > 0.0
    )

    n_prior = torch.cat(
        [
            torch.where(
                is_polyline_cyclic[:, None, None],
                n[:, -1:, :],
                n[:, :1, :],
            ),
            n[:, :-1, :],
        ],
        dim=1,
    )
    n_next = torch.cat(
        [
            n[:, 1:, :],
            torch.where(
                is_polyline_cyclic[:, None, None],
                n[:, :1, :],
                n[:, -1:, :],
            ),
        ],
        dim=1,
    )

    is_prior_valid = torch.cat(
        [
            torch.where(
                is_polyline_cyclic[:, None],
                is_segment_valid[:, -1:],
                is_segment_valid[:, :1],
            ),
            is_segment_valid[:, :-1],
        ],
        dim=1,
    )
    is_next_valid = torch.cat(
        [
            is_segment_valid[:, 1:],
            torch.where(
                is_polyline_cyclic[:, None],
                is_segment_valid[:, :1],
                is_segment_valid[:, -1:],
            ),
        ],
        dim=1,
    )

    sign_if_before = torch.where(
        is_locally_convex[:, :-1, :],
        torch.maximum(n, n_prior),
        torch.minimum(n, n_prior),
    )
    sign_if_after = torch.where(
        is_locally_convex[:, 1:, :],
        torch.maximum(n, n_next),
        torch.minimum(n, n_next),
    )

    sign_to_segment = torch.where(
        (rel_t < 0.0) & is_prior_valid[:, :, None],
        sign_if_before,
        torch.where((rel_t > 1.0) & is_next_valid[:, :, None], sign_if_after, n),
    )

    distance_to_segment_2d = distance_to_segment_2d.reshape(num_polylines * num_segments, num_points).T
    sign_to_segment = sign_to_segment.reshape(num_polylines * num_segments, num_points).T

    is_segment_valid_flat = is_segment_valid.reshape(num_polylines * num_segments)
    valid_mask = is_segment_valid_flat.unsqueeze(0).expand(num_points, -1)
    distance_to_segment_2d = distance_to_segment_2d.masked_fill(
        ~valid_mask,
        EXTREMELY_LARGE_DISTANCE,
    )

    closest_idx = torch.argmin(distance_to_segment_2d, dim=1)
    point_indices = torch.arange(num_points, device=xys.device)
    distance_2d = distance_to_segment_2d[point_indices, closest_idx]
    distance_sign = sign_to_segment[point_indices, closest_idx]

    return distance_sign * distance_2d
