"""Interaction features for the computation of the WOSAC score.
Adapted from: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/interaction_features.py
"""

import math
import torch

from pufferlib.ocean.benchmark import geometry_utils

EXTREMELY_LARGE_DISTANCE = 1e10
COLLISION_DISTANCE_THRESHOLD = 0.0
CORNER_ROUNDING_FACTOR = 0.7
MAX_HEADING_DIFF = math.radians(75.0)
MAX_HEADING_DIFF_FOR_SMALL_OVERLAP = math.radians(10.0)
SMALL_OVERLAP_THRESHOLD = 0.5
MAXIMUM_TIME_TO_COLLISION = 5.0


def compute_signed_distances(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor,
    evaluated_object_mask: torch.Tensor,
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> torch.Tensor:
    """Computes pairwise signed distances between evaluated objects and all other objects.

    Objects are represented by 2D rectangles with rounded corners.

    Args:
        center_x: Shape (num_agents, num_rollouts, num_steps)
        center_y: Shape (num_agents, num_rollouts, num_steps)
        length: Shape (num_agents, num_rollouts) - constant per timestep
        width: Shape (num_agents, num_rollouts) - constant per timestep
        heading: Shape (num_agents, num_rollouts, num_steps)
        valid: Shape (num_agents, num_rollouts, num_steps)
        corner_rounding_factor: Rounding factor for box corners, between 0 (sharp) and 1 (capsule)

    Returns:
        signed_distances: shape (num_eval, num_agents, num_rollouts, num_steps)
    """

    num_agents = center_x.shape[0]
    num_rollouts = center_x.shape[1]
    num_steps = center_x.shape[2]

    eval_indices = torch.nonzero(evaluated_object_mask, as_tuple=False).squeeze(-1)
    num_eval = eval_indices.numel()

    if length.dim() == 2:
        length = length.unsqueeze(-1)
    if width.dim() == 2:
        width = width.unsqueeze(-1)
    length = length.expand(num_agents, num_rollouts, num_steps)
    width = width.expand(num_agents, num_rollouts, num_steps)

    boxes = torch.stack([center_x, center_y, length, width, heading], dim=-1)

    shrinking_distance = torch.minimum(boxes[..., 2], boxes[..., 3]) * corner_rounding_factor / 2.0

    shrunk_len = boxes[..., 2:3] - 2.0 * shrinking_distance.unsqueeze(-1)
    shrunk_wid = boxes[..., 3:4] - 2.0 * shrinking_distance.unsqueeze(-1)

    boxes = torch.cat(
        [
            boxes[..., :2],
            shrunk_len,
            shrunk_wid,
            boxes[..., 4:],
        ],
        dim=-1,
    )

    boxes_flat = boxes.reshape(num_agents * num_rollouts * num_steps, 5)
    box_corners = geometry_utils.get_2d_box_corners(boxes_flat)
    box_corners = box_corners.reshape(num_agents, num_rollouts, num_steps, 4, 2)

    eval_corners = box_corners[eval_indices]

    batch_size = num_eval * num_agents * num_rollouts * num_steps

    corners_flat_1 = (
        eval_corners.unsqueeze(1).expand(num_eval, num_agents, num_rollouts, num_steps, 4, 2).reshape(batch_size, 4, 2)
    )

    corners_flat_2 = (
        box_corners.unsqueeze(0).expand(num_eval, num_agents, num_rollouts, num_steps, 4, 2).reshape(batch_size, 4, 2)
    )

    corners_flat_2.neg_()

    minkowski_sum = geometry_utils.minkowski_sum_of_box_and_box_points(corners_flat_1, corners_flat_2)

    del corners_flat_1, corners_flat_2

    query_points = torch.zeros((batch_size, 2), dtype=center_x.dtype, device=center_x.device)

    signed_distances_flat = geometry_utils.signed_distance_from_point_to_convex_polygon(
        query_points=query_points, polygon_points=minkowski_sum
    )

    del minkowski_sum, query_points

    signed_distances = signed_distances_flat.reshape(num_eval, num_agents, num_rollouts, num_steps)

    eval_shrinking = shrinking_distance[eval_indices]

    signed_distances.sub_(eval_shrinking[:, None, :, :])
    signed_distances.sub_(shrinking_distance[None, :, :, :])

    agent_indices = torch.arange(num_agents, device=center_x.device)
    self_mask = eval_indices[:, None] == agent_indices[None, :]

    self_mask = self_mask.unsqueeze(-1).unsqueeze(-1)

    signed_distances.masked_fill_(self_mask, EXTREMELY_LARGE_DISTANCE)

    eval_valid = valid[eval_indices]

    valid_mask = torch.logical_and(eval_valid[:, None, :, :], valid[None, :, :, :])

    signed_distances.masked_fill_(~valid_mask, EXTREMELY_LARGE_DISTANCE)

    return signed_distances


def compute_distance_to_nearest_object(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor,
    evaluated_object_mask: torch.Tensor,
    corner_rounding_factor: float = CORNER_ROUNDING_FACTOR,
) -> torch.Tensor:
    signed_distances = compute_signed_distances(
        center_x=center_x,
        center_y=center_y,
        length=length,
        width=width,
        heading=heading,
        valid=valid,
        evaluated_object_mask=evaluated_object_mask,
        corner_rounding_factor=corner_rounding_factor,
    )

    min_distances = torch.min(signed_distances, dim=1).values
    return min_distances


def compute_time_to_collision(
    center_x: torch.Tensor,
    center_y: torch.Tensor,
    length: torch.Tensor,
    width: torch.Tensor,
    heading: torch.Tensor,
    valid: torch.Tensor,
    evaluated_object_mask: torch.Tensor,
    seconds_per_step: float,
) -> torch.Tensor:
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
        evaluated_object_mask: Shape (num_agents,) - boolean mask for evaluated agents
        seconds_per_step: Duration of one step in seconds

    Returns:
        Time-to-collision, shape (num_eval_agents, num_rollouts, num_steps)
    """
    from pufferlib.ocean.benchmark import metrics

    valid = valid.to(dtype=torch.bool, device=center_x.device)
    evaluated_object_mask = evaluated_object_mask.to(dtype=torch.bool, device=center_x.device)

    num_agents = center_x.shape[0]
    num_rollouts = center_x.shape[1]
    num_steps = center_x.shape[2]

    eval_indices = torch.nonzero(evaluated_object_mask, as_tuple=False).squeeze(-1)
    num_eval = eval_indices.numel()

    # TODO: Convert to torch
    speed = metrics.compute_kinematic_features(
        x=center_x.cpu().numpy(),
        y=center_y.cpu().numpy(),
        heading=heading.cpu().numpy(),
        seconds_per_step=seconds_per_step,
    )[0]
    if not isinstance(speed, torch.Tensor):
        speed = torch.as_tensor(speed, device=center_x.device, dtype=center_x.dtype)

    if length.dim() == 2:
        length = length.unsqueeze(-1)
    if width.dim() == 2:
        width = width.unsqueeze(-1)

    length = length.expand(num_agents, num_rollouts, num_steps).permute(2, 0, 1)
    width = width.expand(num_agents, num_rollouts, num_steps).permute(2, 0, 1)

    center_x = center_x.permute(2, 0, 1)
    center_y = center_y.permute(2, 0, 1)
    heading = heading.permute(2, 0, 1)
    speed = speed.permute(2, 0, 1)
    valid = valid.permute(2, 0, 1)

    ego_x = center_x[:, eval_indices]
    ego_y = center_y[:, eval_indices]
    ego_len = length[:, eval_indices]
    ego_wid = width[:, eval_indices]
    ego_heading = heading[:, eval_indices]
    ego_speed = speed[:, eval_indices]

    yaw_diff = torch.abs(heading.unsqueeze(1) - ego_heading.unsqueeze(2))

    yaw_diff_cos = torch.cos(yaw_diff)
    yaw_diff_sin = torch.sin(yaw_diff)

    all_sizes_half = torch.stack([length, width], dim=-1).unsqueeze(1) / 2.0

    other_long_offset = geometry_utils.dot_product_2d(
        all_sizes_half,
        torch.abs(torch.stack([yaw_diff_cos, yaw_diff_sin], dim=-1)),
    )
    other_lat_offset = geometry_utils.dot_product_2d(
        all_sizes_half,
        torch.abs(torch.stack([yaw_diff_sin, yaw_diff_cos], dim=-1)),
    )

    del all_sizes_half

    relative_x = center_x.unsqueeze(1) - ego_x.unsqueeze(2)
    relative_y = center_y.unsqueeze(1) - ego_y.unsqueeze(2)
    relative_xy = torch.stack([relative_x, relative_y], dim=-1)

    del relative_x, relative_y

    rotation = -ego_heading.unsqueeze(2).expand(-1, -1, num_agents, -1)

    other_relative_xy = geometry_utils.rotate_2d_points(relative_xy, rotation)

    del relative_xy, rotation

    long_distance = other_relative_xy[..., 0] - ego_len.unsqueeze(2) / 2.0 - other_long_offset
    lat_overlap = torch.abs(other_relative_xy[..., 1]) - ego_wid.unsqueeze(2) / 2.0 - other_lat_offset

    del other_relative_xy, other_long_offset, other_lat_offset

    following_mask = _get_object_following_mask(
        long_distance.permute(1, 2, 0, 3),
        lat_overlap.permute(1, 2, 0, 3),
        yaw_diff.permute(1, 2, 0, 3),
    )

    del lat_overlap, yaw_diff

    valid_mask = torch.logical_and(valid.unsqueeze(1), following_mask.permute(2, 0, 1, 3))

    del following_mask

    long_distance.masked_fill_(~valid_mask, EXTREMELY_LARGE_DISTANCE)

    box_ahead_index = torch.argmin(long_distance, dim=2, keepdim=True)
    distance_to_box_ahead = torch.gather(long_distance, 2, box_ahead_index).squeeze(2)

    del long_distance

    speed_expanded = speed.unsqueeze(1).expand(-1, num_eval, -1, -1)
    box_ahead_speed = torch.gather(speed_expanded, 2, box_ahead_index).squeeze(2)

    rel_speed = ego_speed - box_ahead_speed

    rel_speed_safe = torch.where(rel_speed > 0.0, rel_speed, torch.ones_like(rel_speed))

    max_ttc = torch.full_like(rel_speed, MAXIMUM_TIME_TO_COLLISION)

    time_to_collision = torch.where(
        rel_speed > 0.0,
        torch.minimum(distance_to_box_ahead / rel_speed_safe, max_ttc),
        max_ttc,
    )

    return time_to_collision.permute(1, 2, 0)


def _get_object_following_mask(
    longitudinal_distance,
    lateral_overlap,
    yaw_diff,
):
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
    valid_mask = torch.logical_and(valid_mask, yaw_diff <= MAX_HEADING_DIFF)
    valid_mask = torch.logical_and(valid_mask, lateral_overlap < 0.0)
    return torch.logical_and(
        valid_mask,
        torch.logical_or(
            lateral_overlap < -SMALL_OVERLAP_THRESHOLD,
            yaw_diff <= MAX_HEADING_DIFF_FOR_SMALL_OVERLAP,
        ),
    )
