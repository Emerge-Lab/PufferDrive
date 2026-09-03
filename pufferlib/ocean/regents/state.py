"""Canonical Torch-side scenario representation for ReGentS.

All poses use the scenario-local centered Cartesian frame emitted by
``Drive.get_state()``. Tensor axes are always ``[batch, agent, time, feature]``.
The feature order is x, y, wrapped heading, signed longitudinal speed, and
actual steering angle.
"""

from dataclasses import dataclass

import numpy as np
import torch


STATE_X = 0
STATE_Y = 1
STATE_HEADING = 2
STATE_SPEED = 3
STATE_STEERING = 4
STATE_FEATURE_COUNT = 5


def signed_speed_from_c_velocity(velocity_x, velocity_y, wrapped_heading):
    """Recover the C simulator's ``sim_speed_signed``: magnitude signed by heading."""
    magnitude = np.hypot(velocity_x, velocity_y)
    heading_projection = velocity_x * np.cos(wrapped_heading) + velocity_y * np.sin(wrapped_heading)
    return np.copysign(magnitude, heading_projection)


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


@dataclass(frozen=True)
class ScenarioBatch:
    """Validated, padded batch exported from one vectorized Drive instance.

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
    scenario_ids: tuple[str, ...]
    dataset_names: tuple[str, ...]
    log_dt_seconds: torch.Tensor
    dt_seconds: float
    init_step: int
    scenario_length: int
    drivable_area_rasters: tuple[DrivableAreaRaster, ...]
    coordinate_frame: str = "scenario_centered_cartesian"

    def __post_init__(self):
        if self.logged_state.dtype != torch.float32:
            raise TypeError("logged_state must use torch.float32")
        if self.logged_state.ndim != 4 or self.logged_state.shape[-1] != STATE_FEATURE_COUNT:
            raise ValueError("logged_state must have shape [batch, agent, time, 5]")
        batch_count, agent_count, time_count, _ = self.logged_state.shape
        state_shape = (batch_count, agent_count, time_count)
        agent_shape = (batch_count, agent_count)
        if tuple(self.state_valid.shape) != state_shape or self.state_valid.dtype != torch.bool:
            raise ValueError("state_valid must be bool [batch, agent, time]")
        if tuple(self.state_feature_valid.shape) != (*state_shape, STATE_FEATURE_COUNT):
            raise ValueError("state_feature_valid must have shape [batch, agent, time, 5]")
        if self.state_feature_valid.dtype != torch.bool:
            raise TypeError("state_feature_valid must use torch.bool")
        if tuple(self.transition_valid.shape) != (batch_count, agent_count, max(time_count - 1, 0)):
            raise ValueError("transition_valid must be bool [batch, agent, time - 1]")
        if self.transition_valid.dtype != torch.bool:
            raise TypeError("transition_valid must use torch.bool")
        for name in ("logged_length_meters", "logged_width_meters"):
            tensor = getattr(self, name)
            if tuple(tensor.shape) != state_shape:
                raise ValueError(f"{name} must have shape [batch, agent, time]")
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} must use torch.float32")
        if tuple(self.current_state.shape) != (*agent_shape, STATE_FEATURE_COUNT):
            raise ValueError("current_state must have shape [batch, agent, 5]")
        if self.current_state.dtype != torch.float32:
            raise TypeError("current_state must use torch.float32")
        for name in (
            "current_valid",
            "agent_present",
            "agent_metadata_valid",
            "active_agent_mask",
            "ego_mask",
            "vehicle_mask",
            "candidate_adversary_mask",
        ):
            tensor = getattr(self, name)
            if tuple(tensor.shape) != agent_shape or tensor.dtype != torch.bool:
                raise ValueError(f"{name} must be bool [batch, agent]")
        for name in (
            "agent_id",
            "agent_type",
            "controller",
            "trajectory_length",
        ):
            tensor = getattr(self, name)
            if tuple(tensor.shape) != agent_shape or tensor.dtype != torch.int64:
                raise ValueError(f"{name} must be int64 [batch, agent]")
        for name in (
            "length_meters",
            "width_meters",
            "wheelbase_meters",
            "maximum_speed_mps",
        ):
            tensor = getattr(self, name)
            if tuple(tensor.shape) != agent_shape or tensor.dtype != torch.float32:
                raise ValueError(f"{name} must be float32 [batch, agent]")
        if len(self.scenario_ids) != batch_count or len(self.dataset_names) != batch_count:
            raise ValueError("Scenario metadata must have one entry per batch item")
        if tuple(self.log_dt_seconds.shape) != (batch_count,):
            raise ValueError("log_dt_seconds must have shape [batch]")
        if self.log_dt_seconds.dtype != torch.float32:
            raise TypeError("log_dt_seconds must use torch.float32")
        if len(self.drivable_area_rasters) != batch_count:
            raise ValueError("drivable_area_rasters must have one entry per batch item")

    @property
    def batch_size(self):
        return self.logged_state.shape[0]

    @property
    def max_agent_count(self):
        return self.logged_state.shape[1]

    @property
    def max_time_count(self):
        return self.logged_state.shape[2]
