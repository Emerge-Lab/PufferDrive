import torch
import torch.nn as nn
import numpy as np


def rollout_state_trajectory_ego(
    actions_N2: torch.Tensor,
    observations: torch.Tensor,
    dt: float = 0.1,
) -> torch.Tensor:
    """reproduction of the classic dynamics in pytorch in ego position (x=0, y=0, heading=0)
    this is a temporary thing to make sure we can rollout a trajecotry of actions into a trajectory of coordinates

    input shapes (K is the sequence length)
    actions_N2 : B x BPTT x K x 2
    initial_position : B x BPTT x 3 (x, y, heading)
    initial_speed_vector : B x BPTT x 2 (vx, vy)
    length : B x BPTT x 1
    """
    MAX_SPEED = 100.0
    MAX_LENGTH = 30.0
    ACCELERATION_VALUES = torch.tensor(
        [-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0], device=actions_N2.device, dtype=torch.float32
    )
    STEERING_VALUES = torch.tensor(
        [-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
        device=actions_N2.device,
        dtype=torch.float32,
    )
    _B = actions_N2.shape[0]
    _BPTT = actions_N2.shape[1]
    tensorshapes = (_B, _BPTT)

    # convert the action into nice stuff
    accel_values = ACCELERATION_VALUES[actions_N2[..., 0].long()]  # convert into long to use it as an index
    steering_values = STEERING_VALUES[actions_N2[..., 1].long()]

    # note: the multiplication create a new tensor so no need to clone to protect the original objects
    signed_speed = observations[..., 3] * MAX_SPEED  # already in obs, de-normalize
    length = observations[..., 5] * MAX_LENGTH

    x = torch.zeros(tensorshapes, device=actions_N2.device, dtype=torch.float32)
    y = torch.zeros(tensorshapes, device=actions_N2.device, dtype=torch.float32)
    heading = torch.zeros(tensorshapes, device=actions_N2.device, dtype=torch.float32)

    vx = signed_speed
    vy = torch.zeros(tensorshapes, device=actions_N2.device, dtype=torch.float32)

    positions = []

    for k in range(actions_N2.shape[2]):
        speed_magnitude = torch.sqrt(vx**2 + vy**2)
        v_dot_heading = vx * torch.cos(heading) + vy * torch.sin(heading)
        signed_speed = torch.copysign(speed_magnitude, v_dot_heading)

        signed_speed = signed_speed + accel_values[..., k] * dt
        signed_speed = torch.where(signed_speed > MAX_SPEED, MAX_SPEED, signed_speed)

        # yaw rate i.e. how much you can steeer

        beta = torch.tanh(0.5 * torch.tan(steering_values[..., k]))
        yaw_rate = (signed_speed * torch.cos(beta) * torch.tan(steering_values[..., k])) / length

        new_vx = signed_speed * torch.cos(heading + beta)
        new_vy = signed_speed * torch.sin(heading + beta)

        x = x + new_vx * dt
        y = y + new_vy * dt
        heading = heading + yaw_rate * dt

        positions.append(torch.stack([x, y], dim=-1))
        if k == 0:
            heading_t0 = heading

    return torch.stack(positions, dim=2), heading_t0  # B x BPTT x K x 2


def compute_l2_loss_ego_action_traj(
    traj: torch.Tensor,
    traj_tm1: torch.Tensor,
    heading: torch.Tensor,
    terminals: torch.Tensor,
    trajectory_loss_norm: float = 1000.0,
    trajectory_loss_clamp_min: float = -0.05,
) -> torch.Tensor:
    """
    this function computes the difference in implictly planned occupied states between two action sequences
    since the framing is always ego, we have to zero in and rotate the second trajectory so that it matches
    the frame of the first one

    expects:

    traj : B x BPTT x K x 2
    traj_tm1 : B x K x 2
    heading : B x BPTT x 1
    terminals : B x BPTT (1 = auto-reset happened, obs is post-reset)
    """

    if traj.shape[2] == 1:
        return torch.zeros_like(heading)  # if we have no trajectory (i.e. only one action we can return 0)

    newtraj = torch.cat(
        [traj_tm1.unsqueeze(1), traj], dim=1
    )  # concatenate the previous trajectory to compute loss at first step

    # we take up to the last element and renormalize the position to the second position of the trajectory.
    # since the second position is the position occupied after the action, it is the starting point (the zero) of the second trajectory
    zeroed_tm1 = newtraj[:, :-1, 1:, :] - newtraj[:, :-1, 1, :].unsqueeze(2)

    # now we rotate everything to bring the tm1 trajectory in the same ego reference as the current time trajectory
    cos_heading = torch.cos(heading)[..., None]
    sin_heading = torch.sin(heading)[..., None]

    rotated_x = zeroed_tm1[..., 0] * cos_heading + zeroed_tm1[..., 1] * sin_heading
    rotated_y = -zeroed_tm1[..., 0] * sin_heading + zeroed_tm1[..., 1] * cos_heading

    rotated = torch.stack([rotated_x, rotated_y], dim=-1)  # this should be B X BPTT X K-1 X 2

    # we now compute the loss i.e. the difference of positions occupied at future timesteps
    per_point_l2 = torch.sqrt(((newtraj[:, 1:, :-1, :] - rotated[:, :, :, :].detach()) ** 2).sum(-1)).sum(
        -1
    )  # this should be B X BPTT
    per_point_l2 = per_point_l2 * (
        1.0 - terminals
    )  # if a trajecotry is after a terminal point, we don't consider it for the loss (the first plan is free)

    # compute the normalization factors based on the length of the trajectories
    diffs_t = newtraj[:, 1:, 1:-1, :] - newtraj[:, 1:, 0:-2, :]
    arc_lengths_t = torch.sqrt((diffs_t**2).sum(-1)).sum(-1)

    diffs_tm1 = (
        newtraj[:, :-1, 2:, :] - newtraj[:, :-1, 1:-1, :]
    )  # distance is invariant to rotation and shifting so we can use newtraj
    arc_lengths_tm1 = torch.sqrt((diffs_tm1**2).sum(-1)).sum(-1)

    loss = per_point_l2 / (torch.maximum(arc_lengths_t, arc_lengths_tm1) + 1.0)
    return torch.clamp(-loss / trajectory_loss_norm, trajectory_loss_clamp_min, 0.0)


def rollout_state_trajectory(
    actions_N2: torch.Tensor,
    initial_position: torch.Tensor,
    initial_speed_vector: torch.Tensor,
    length: torch.Tensor,
    dt: float = 0.1,
) -> torch.Tensor:
    raise NotImplementedError()
    """reproduction of the classic dynamics in pytorch
    this is a temporary thing to make sure we can rollout a trajecotry of actions into a trajectory of coordinates

    input shapes (K is the sequence length)
    actions_N2 : B X K X 12
    initial_position : B X 3 (x, y, heading)
    initial_speed_vector : B X 2 (vx, vy)
    length : B x 1
    """
    MAX_SPEED = 100
    ACCELERATION_VALUES = torch.tensor(
        [-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0], device=actions_N2.device, dtype=torch.float32
    )
    STEERING_VALUES = torch.tensor(
        [-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
        device=actions_N2.device,
        dtype=torch.float32,
    )

    # convert the action into nice stuff
    accel_values = ACCELERATION_VALUES[actions_N2[..., 0].long()]  # convert into long to use it as an index
    steering_values = STEERING_VALUES[actions_N2[..., 1].long()]

    x = initial_position[..., 0]
    y = initial_position[..., 1]
    heading = initial_position[..., 2]

    vx = initial_speed_vector[..., 0]
    vy = initial_speed_vector[..., 1]

    positions = []

    for k in range(actions_N2.shape[1]):
        speed_magnitude = torch.sqrt(vx**2 + vy**2)
        v_dot_heading = vx * torch.cos(heading) + vy * torch.sin(heading)
        signed_speed = torch.copysign(speed_magnitude, v_dot_heading)

        signed_speed = signed_speed + accel_values * dt
        signed_speed = torch.where(signed_speed > MAX_SPEED, MAX_SPEED, signed_speed)

        # yaw rate i.e. how much you can steeer

        beta = torch.tanh(0.5 * torch.tan(steering_values))
        yaw_rate = (signed_speed * torch.cos(beta) * torch.tan(steering_values)) / length

        new_vx = signed_speed * torch.cos(heading + beta)
        new_vy = signed_speed * torch.sin(heading + beta)

        x = x + new_vx * dt
        y = y + new_vy * dt
        heading = heading + yaw_rate * dt

        positions.append(torch.stack([x, y], dim=-1))

    return torch.stack(positions, dim=1)  # B x K x 2
