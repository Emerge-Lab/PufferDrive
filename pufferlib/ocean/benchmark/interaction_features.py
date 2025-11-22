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

    eval_indices = torch.nonzero(evaluated_object_mask, as_tuple=False).squeeze(-1)
    other_indices = torch.nonzero(~evaluated_object_mask, as_tuple=False).squeeze(-1)
    num_eval = eval_indices.numel()

    if length.dim() == 2:
        length = length.unsqueeze(-1)
    if width.dim() == 2:
        width = width.unsqueeze(-1)
    length = length.expand(num_agents, num_rollouts, num_steps)
    width = width.expand(num_agents, num_rollouts, num_steps)

    boxes = torch.stack([center_x, center_y, length, width, heading], dim=-1)

    shrinking_distance = torch.minimum(boxes[..., 2], boxes[..., 3]) * corner_rounding_factor / 2.0

    boxes = torch.cat(
        [
            boxes[..., :2],
            boxes[..., 2:3] - 2.0 * shrinking_distance.unsqueeze(-1),
            boxes[..., 3:4] - 2.0 * shrinking_distance.unsqueeze(-1),
            boxes[..., 4:],
        ],
        dim=-1,
    )

    boxes_flat = boxes.reshape(num_agents * num_rollouts * num_steps, 5)
    box_corners = geometry_utils.get_2d_box_corners(boxes_flat)
    box_corners = box_corners.reshape(num_agents, num_rollouts, num_steps, 4, 2)

    eval_corners = box_corners[eval_indices]
    other_corners = box_corners[other_indices]
    all_corners = torch.cat([eval_corners, other_corners], dim=0)

    corners_1 = eval_corners.unsqueeze(1)
    corners_2 = all_corners.unsqueeze(0)

    corners_broadcast_1 = corners_1.expand(num_eval, num_agents, num_rollouts, num_steps, 4, 2)
    corners_broadcast_2 = corners_2.expand(num_eval, num_agents, num_rollouts, num_steps, 4, 2)

    batch_size = num_eval * num_agents * num_rollouts * num_steps
    corners_flat_1 = corners_broadcast_1.reshape(batch_size, 4, 2)
    corners_flat_2 = corners_broadcast_2.reshape(batch_size, 4, 2)

    neg_corners_2 = -1.0 * corners_flat_2
    minkowski_sum = geometry_utils.minkowski_sum_of_box_and_box_points(corners_flat_1, neg_corners_2)

    query_points = torch.zeros((batch_size, 2), dtype=torch.float32, device=center_x.device)
    signed_distances_flat = geometry_utils.signed_distance_from_point_to_convex_polygon(
        query_points=query_points, polygon_points=minkowski_sum
    )

    signed_distances = signed_distances_flat.reshape(num_eval, num_agents, num_rollouts, num_steps)

    eval_shrinking = shrinking_distance[eval_indices]
    other_shrinking = shrinking_distance[other_indices]
    all_shrinking = torch.cat([eval_shrinking, other_shrinking], dim=0)

    signed_distances = signed_distances - eval_shrinking[:, None, :, :]
    signed_distances = signed_distances - all_shrinking[None, :, :, :]

    self_mask = torch.eye(num_eval, num_agents, dtype=torch.float32, device=center_x.device).unsqueeze(-1).unsqueeze(-1)
    signed_distances = signed_distances + self_mask * EXTREMELY_LARGE_DISTANCE

    eval_valid = valid[eval_indices]
    other_valid = valid[other_indices]
    all_valid = torch.cat([eval_valid, other_valid], dim=0)

    valid_mask = torch.logical_and(eval_valid[:, None, :, :], all_valid[None, :, :, :])
    signed_distances = torch.where(
        valid_mask,
        signed_distances,
        torch.full_like(signed_distances, EXTREMELY_LARGE_DISTANCE),
    )

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
    other_indices = torch.nonzero(~evaluated_object_mask, as_tuple=False).squeeze(-1)
    num_eval = eval_indices.numel()

    # NOTE: I know this is ugly, I will fix it another day
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
    length_broadcast = length.expand(num_agents, num_rollouts, num_steps)
    width_broadcast = width.expand(num_agents, num_rollouts, num_steps)

    boxes = torch.stack([center_x, center_y, length_broadcast, width_broadcast, heading, speed], dim=-1)
    boxes = boxes.permute(2, 0, 1, 3)

    valid_transposed = valid.permute(2, 0, 1)

    eval_boxes = boxes[:, eval_indices]
    other_boxes = boxes[:, other_indices]
    all_boxes = torch.cat([eval_boxes, other_boxes], dim=1)

    eval_valid = valid_transposed[:, eval_indices]
    other_valid = valid_transposed[:, other_indices]
    all_valid = torch.cat([eval_valid, other_valid], dim=1)

    ego_xy, ego_sizes, ego_yaw, ego_speed = torch.split(eval_boxes, [2, 2, 1, 1], dim=-1)
    other_xy, other_sizes, other_yaw, _ = torch.split(all_boxes, [2, 2, 1, 1], dim=-1)

    yaw_diff = torch.abs(other_yaw.unsqueeze(1) - ego_yaw.unsqueeze(2))

    yaw_diff_cos = torch.cos(yaw_diff)
    yaw_diff_sin = torch.sin(yaw_diff)

    other_long_offset = geometry_utils.dot_product_2d(
        other_sizes.unsqueeze(1) / 2.0,
        torch.abs(torch.cat([yaw_diff_cos, yaw_diff_sin], dim=-1)),
    )
    other_lat_offset = geometry_utils.dot_product_2d(
        other_sizes.unsqueeze(1) / 2.0,
        torch.abs(torch.cat([yaw_diff_sin, yaw_diff_cos], dim=-1)),
    )

    relative_xy = other_xy.unsqueeze(1) - ego_xy.unsqueeze(2)
    rotation = -ego_yaw[..., 0].unsqueeze(2).expand(-1, -1, relative_xy.shape[2], -1)
    other_relative_xy = geometry_utils.rotate_2d_points(relative_xy, rotation)

    long_distance = other_relative_xy[..., 0] - ego_sizes[..., 0].unsqueeze(2) / 2.0 - other_long_offset

    lat_overlap = torch.abs(other_relative_xy[..., 1]) - ego_sizes[..., 1].unsqueeze(2) / 2.0 - other_lat_offset

    following_mask = _get_object_following_mask(
        long_distance.permute(1, 2, 0, 3),
        lat_overlap.permute(1, 2, 0, 3),
        yaw_diff[..., 0].permute(1, 2, 0, 3),
    )

    valid_mask = torch.logical_and(all_valid.unsqueeze(1), following_mask.permute(2, 0, 1, 3))
    masked_long_distance = long_distance + (1.0 - valid_mask.to(torch.float32)) * EXTREMELY_LARGE_DISTANCE

    box_ahead_index = torch.argmin(masked_long_distance, dim=-2)
    gather_index = box_ahead_index.unsqueeze(-2)
    distance_to_box_ahead = torch.gather(masked_long_distance, -2, gather_index).squeeze(-2)

    speed_transposed = speed.permute(2, 0, 1)
    speed_eval = speed_transposed[:, eval_indices]
    speed_other = speed_transposed[:, other_indices]
    ordered_speed = torch.cat([speed_eval, speed_other], dim=1)
    speed_broadcast = ordered_speed.unsqueeze(1).expand(-1, num_eval, -1, -1)
    box_ahead_speed = torch.gather(speed_broadcast, -2, gather_index).squeeze(-2)

    rel_speed = ego_speed[..., 0] - box_ahead_speed

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
