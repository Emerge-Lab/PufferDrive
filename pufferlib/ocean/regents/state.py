"""Canonical Torch-side scenario representation for ReGentS.

All poses use the scenario-local centered Cartesian frame emitted by
``Drive.get_state()``. Tensor axes are always ``[batch, agent, time, feature]``.
The feature order is x, y, wrapped heading, signed longitudinal speed, and
actual steering angle.
"""

import math
from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from pufferlib.ocean.drive import binding


STATE_X = 0
STATE_Y = 1
STATE_HEADING = 2
STATE_SPEED = 3
STATE_STEERING = 4
STATE_FEATURE_COUNT = 5


def moved_to_device(instance, device):
    """Return a frozen dataclass copy with every tensor field on ``device``."""
    moved = {}
    for field in fields(instance):
        value = getattr(instance, field.name)
        if isinstance(value, torch.Tensor):
            moved[field.name] = value.to(device)
    return replace(instance, **moved)


def wrapped_angle_difference(first, second):
    """Return ``first - second`` wrapped to ``[-pi, pi]``."""
    difference = first - second
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def single_scenario_payload(payload):
    """Unwrap the one-scenario Drive state payload, which arrives dict or list-of-one."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return payload[0]
    raise ValueError("ReGentS requires exactly one Drive scenario")


def signed_speed_from_c_velocity(velocity_x, velocity_y, wrapped_heading):
    """Recover the C simulator's ``sim_speed_signed``: magnitude signed by heading."""
    magnitude = np.hypot(velocity_x, velocity_y)
    heading_projection = velocity_x * np.cos(wrapped_heading) + velocity_y * np.sin(wrapped_heading)
    return np.copysign(magnitude, heading_projection)


REGENTS_EGO_ACTION_FEATURE_COUNT = 2


def agent_state_rows(payload, expected_agent_count):
    """Parse one ``Drive.get_state()`` payload into stable-indexed state rows.

    Agents are serialized in the simulator's own array order, so a row index is a
    stable agent index. Returns the scenario dict, the state rows, per-agent validity,
    and the ego's last normalized action.
    """
    scenario = single_scenario_payload(payload)
    agents = scenario.get("agents")
    if not isinstance(agents, list) or len(agents) != expected_agent_count:
        raise ValueError("Drive state does not carry the expected agent count")
    states = np.zeros((expected_agent_count, STATE_FEATURE_COUNT), dtype=np.float32)
    valid = np.zeros(expected_agent_count, dtype=np.bool_)
    for stable_agent_idx, agent in enumerate(agents):
        if int(agent.get("id", -1)) != stable_agent_idx:
            raise ValueError("Drive state changed stable agent identity")
        values = np.asarray(
            (
                agent["sim_x"],
                agent["sim_y"],
                agent["sim_heading"],
                agent["sim_vx"],
                agent["sim_vy"],
                agent["sim_steering"],
            ),
            dtype=np.float32,
        )
        serialized_valid = int(agent["sim_valid"])
        if serialized_valid not in (0, 1):
            raise ValueError("Drive state emitted invalid sim_valid")
        if serialized_valid and not np.isfinite(values).all():
            raise ValueError("Drive state emitted NaN or Inf")
        heading = np.float32(math.atan2(math.sin(float(values[2])), math.cos(float(values[2]))))
        states[stable_agent_idx] = (
            values[0],
            values[1],
            heading,
            signed_speed_from_c_velocity(values[3], values[4], heading),
            values[5],
        )
        valid[stable_agent_idx] = bool(serialized_valid)
    ego_action = np.asarray(
        (
            agents[0]["accel_long"] / float(binding.ACCELERATION_VALUES[6]),
            agents[0]["sim_steering"] / float(binding.STEERING_VALUES[8]),
        ),
        dtype=np.float32,
    )
    return scenario, states, valid, ego_action


@dataclass(frozen=True)
class RasterTransform:
    """Cartesian world-to-raster transform.

    Column zero is ``origin_x_m`` and columns increase with world x. Row zero
    is ``origin_y_m`` and rows increase with world y. Pixel centers, rather
    than pixel edges, lie on these coordinates.
    """

    origin_x_m: float
    origin_y_m: float
    resolution_meters_per_pixel: float
    height: int
    width: int

    def __post_init__(self):
        if self.resolution_meters_per_pixel <= 0:
            raise ValueError("Raster resolution must be positive")
        if self.height < 1 or self.width < 1:
            raise ValueError("Raster dimensions must be positive")

    def world_to_grid(self, xy_meters):
        """Return fractional ``[column, row]`` coordinates."""
        origin = xy_meters.new_tensor((self.origin_x_m, self.origin_y_m))
        return (xy_meters - origin) / self.resolution_meters_per_pixel

    def world_to_normalized_grid(self, xy_meters):
        """Return grid_sample coordinates for ``align_corners=True``."""
        grid = self.world_to_grid(xy_meters)
        denominators = grid.new_tensor((max(self.width - 1, 1), max(self.height - 1, 1)))
        return 2.0 * grid / denominators - 1.0


@dataclass(frozen=True)
class DrivableAreaRaster:
    """Boolean raster where ``True`` denotes a drivable lane surface."""

    mask: torch.Tensor
    transform: RasterTransform

    def __post_init__(self):
        expected_shape = (self.transform.height, self.transform.width)
        if self.mask.dtype != torch.bool:
            raise TypeError(f"Drivable raster must be bool, got {self.mask.dtype}")
        if tuple(self.mask.shape) != expected_shape:
            raise ValueError(f"Drivable raster shape must be {expected_shape}, got {tuple(self.mask.shape)}")

    def to(self, device):
        return moved_to_device(self, device)

    def points_off_road(self, xy_meters):
        """Flag world points lying outside the drivable surface, sampled at the nearest pixel.

        A point beyond the raster is off road: the raster covers the scenario's own map
        extent, so there is no drivable surface to fall back on outside it.
        """
        grid = self.transform.world_to_grid(xy_meters)
        column = torch.round(grid[..., 0]).to(torch.int64)
        row = torch.round(grid[..., 1]).to(torch.int64)
        outside = (column < 0) | (column >= self.transform.width)
        outside |= (row < 0) | (row >= self.transform.height)
        nearest_drivable = self.mask.to(xy_meters.device)[
            row.clamp(0, self.transform.height - 1), column.clamp(0, self.transform.width - 1)
        ]
        return outside | ~nearest_drivable


@dataclass(frozen=True)
class Scenario:
    """One scenario exported from a single-env Drive instance.

    Tensor axes are ``[agent, time, feature]``; there is no batch axis, because
    ReGentS optimizes one scenario at a time.

    ``logged_state`` has five features. The logged files do not contain
    steering, so that channel is zero and ``state_feature_valid[..., 4]`` is
    false until inverse dynamics fills it in during Stage 3. No downstream
    consumer may treat a feature as observed without consulting that mask.
    """

    logged_state: torch.Tensor
    state_valid: torch.Tensor
    state_feature_valid: torch.Tensor
    transition_valid: torch.Tensor
    current_state: torch.Tensor
    current_valid: torch.Tensor
    agent_present: torch.Tensor
    agent_metadata_valid: torch.Tensor
    active_agent_mask: torch.Tensor
    agent_id: torch.Tensor
    agent_type: torch.Tensor
    controller: torch.Tensor
    trajectory_length: torch.Tensor
    ego_mask: torch.Tensor
    vehicle_mask: torch.Tensor
    candidate_adversary_mask: torch.Tensor
    logged_length_meters: torch.Tensor
    logged_width_meters: torch.Tensor
    length_meters: torch.Tensor
    width_meters: torch.Tensor
    wheelbase_meters: torch.Tensor
    maximum_speed_mps: torch.Tensor
    scenario_id: str
    dataset_name: str
    log_dt_seconds: float
    dt_seconds: float
    init_step: int
    scenario_length: int
    drivable_area_raster: DrivableAreaRaster
    coordinate_frame: str = "scenario_centered_cartesian"

    def to(self, device):
        """Return this scenario with every tensor and its raster on ``device``."""
        moved = moved_to_device(self, device)
        return replace(moved, drivable_area_raster=self.drivable_area_raster.to(device))

    @property
    def device(self):
        return self.logged_state.device

    @property
    def max_agent_count(self):
        return self.logged_state.shape[0]

    @property
    def max_time_count(self):
        return self.logged_state.shape[1]
