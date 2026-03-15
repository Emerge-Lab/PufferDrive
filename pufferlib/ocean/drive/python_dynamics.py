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
    traj: torch.Tensor, traj_tm1: torch.Tensor, heading: torch.Tensor, prev_shifted_terminals: torch.Tensor
) -> torch.Tensor:
    """
    this function computes the difference in implictly planned occupied states between two action sequences
    since the framing is always ego, we have to zero in and rotate the second trajectory so that it matches
    the frame of the first one

    expects:

    traj : B x BPTT x K x 2
    traj_tm1 : B x K x 2
    heading : B x BPTT x 1
    """

    newtraj = torch.cat([traj_tm1.unsqueeze(1), traj], dim=1)

    if newtraj.shape[2] == 1:
        return torch.zeros_like(heading)
    zeroed = newtraj[:, :-1, 1:, :] - newtraj[:, :-1, 1, :].unsqueeze(2)
    cos_heading = torch.cos(heading)[..., None]
    sin_heading = torch.sin(heading)[..., None]
    rotated_x = zeroed[..., 0] * cos_heading - zeroed[..., 1] * sin_heading
    rotated_y = zeroed[..., 0] * sin_heading + zeroed[..., 1] * cos_heading

    rotated = torch.stack([rotated_x, rotated_y], dim=-1)  # this should be B X K-1 X 2

    per_point_l2 = torch.sqrt(((newtraj[:, 1:, :-1, :] - rotated[:, :, :, :].detach()) ** 2).sum(-1))
    per_point_l2 = per_point_l2 * torch.abs(prev_shifted_terminals - 1).unsqueeze(-1)
    diffs = newtraj[:, 1:, :1, :] - newtraj[:, 1:, :-1, :]
    arc_length = torch.sqrt((diffs**2).sum(-1)).sum(-1)
    loss = per_point_l2.sum(-1) / (arc_length + 1e-6)
    return -loss / 500.0


def ego_trajectories_to_world(ego_traj, agent_x, agent_y, agent_heading):
    """Transform ego-frame trajectories to world coordinates per agent.

    Args:
        ego_traj: [B, K, 2] ego-frame positions (from rollout_state_trajectory_ego)
        agent_x: [B] world x positions (numpy, from get_global_agent_state)
        agent_y: [B] world y positions (numpy, from get_global_agent_state)
        agent_heading: [B] world heading (numpy, from get_global_agent_state)

    Returns:
        world_x: numpy float32 [B * K] flattened world x coords (ready for set_predicted_trajectories)
        world_y: numpy float32 [B * K] flattened world y coords
    """
    cos_h = np.cos(agent_heading)[:, None]  # [B, 1]
    sin_h = np.sin(agent_heading)[:, None]

    # ego_traj is torch, move to numpy
    ex = ego_traj[..., 0].detach().cpu().numpy()  # [B, K]
    ey = ego_traj[..., 1].detach().cpu().numpy()

    # Rotate ego frame into world frame and translate
    world_x = ex * cos_h - ey * sin_h + agent_x[:, None]
    world_y = ex * sin_h + ey * cos_h + agent_y[:, None]

    return world_x.ravel().astype(np.float32), world_y.ravel().astype(np.float32)


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
