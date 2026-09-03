"""Differentiable oriented-box and drivable-area geometry for ReGentS."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as torch_functional

from pufferlib.ocean.evaluation_utils.wosac import geometry_utils as wosac_geometry
from pufferlib.ocean.regents.state import DrivableAreaRaster, RasterTransform


DEFAULT_GAUSSIAN_SIGMA_METERS = 0.5
DEFAULT_GAUSSIAN_TRUNCATE_SIGMA = 3.0
BOX_FEATURE_COUNT = 5


@dataclass(frozen=True)
class SmoothedOutOfBoundsRaster:
    """Differentiable out-of-bounds potential sampled in centered world space."""

    potential: torch.Tensor
    transform: RasterTransform
    gaussian_sigma_meters: float
    gaussian_truncate_sigma: float

    def __post_init__(self):
        expected_shape = (self.transform.height, self.transform.width)
        if not self.potential.is_floating_point():
            raise TypeError("Out-of-bounds potential must use a floating dtype")
        if tuple(self.potential.shape) != expected_shape:
            raise ValueError(f"Out-of-bounds potential must have shape {expected_shape}")
        if not torch.isfinite(self.potential).all():
            raise ValueError("Out-of-bounds potential must be finite")
        if self.gaussian_sigma_meters <= 0 or not math.isfinite(self.gaussian_sigma_meters):
            raise ValueError("Gaussian sigma must be finite and positive")
        if self.gaussian_truncate_sigma <= 0 or not math.isfinite(self.gaussian_truncate_sigma):
            raise ValueError("Gaussian truncation must be finite and positive")

    def to(self, device=None, dtype=None):
        """Return a raster transferred once to an optimization device/dtype."""
        potential = self.potential.to(device=device, dtype=dtype)
        return SmoothedOutOfBoundsRaster(
            potential=potential,
            transform=self.transform,
            gaussian_sigma_meters=self.gaussian_sigma_meters,
            gaussian_truncate_sigma=self.gaussian_truncate_sigma,
        )


def _validate_boxes(boxes, name):
    if not isinstance(boxes, torch.Tensor) or not boxes.is_floating_point():
        raise TypeError(f"{name} must be a floating Torch tensor")
    if boxes.ndim < 1 or boxes.shape[-1] != BOX_FEATURE_COUNT:
        raise ValueError(f"{name} must have shape [..., 5]")
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} must be finite")
    if torch.any(boxes[..., 2:4] <= 0):
        raise ValueError(f"{name} length and width must be positive")


def oriented_box_corners(boxes):
    """Return counter-clockwise corners for ``[..., x, y, length, width, heading]``."""
    _validate_boxes(boxes, "boxes")
    return wosac_geometry.get_2d_box_corners(boxes)


def signed_box_distance(boxes_a, boxes_b):
    """Return exact signed Euclidean distance between broadcastable oriented boxes.

    Positive values denote separation, zero denotes boundary contact, and negative
    values denote penetration. The magnitude inside overlap is the minimum
    translation needed to separate the rectangles.
    """
    _validate_boxes(boxes_a, "boxes_a")
    _validate_boxes(boxes_b, "boxes_b")
    if boxes_a.device != boxes_b.device or boxes_a.dtype != boxes_b.dtype:
        raise ValueError("boxes_a and boxes_b must share device and dtype")
    try:
        output_shape = torch.broadcast_shapes(boxes_a.shape[:-1], boxes_b.shape[:-1])
    except RuntimeError as error:
        raise ValueError("boxes_a and boxes_b prefixes must be broadcastable") from error

    expanded_a = boxes_a.expand(*output_shape, BOX_FEATURE_COUNT)
    expanded_b = boxes_b.expand(*output_shape, BOX_FEATURE_COUNT)
    corners_a = wosac_geometry.get_2d_box_corners(expanded_a).reshape(-1, 4, 2)
    corners_b = wosac_geometry.get_2d_box_corners(expanded_b).reshape(-1, 4, 2)
    minkowski_polygon = wosac_geometry.minkowski_sum_of_box_and_box_points(corners_a, -corners_b)
    query_points = torch.zeros(
        (minkowski_polygon.shape[0], 2),
        dtype=minkowski_polygon.dtype,
        device=minkowski_polygon.device,
    )
    distances = wosac_geometry.signed_distance_from_point_to_convex_polygon(
        query_points,
        minkowski_polygon,
    )
    return distances.reshape(output_shape)


def _gaussian_kernel_2d(sigma_pixels, truncate_sigma, dtype, device):
    kernel_radius = max(1, math.ceil(sigma_pixels * truncate_sigma))
    coordinates = torch.arange(-kernel_radius, kernel_radius + 1, dtype=dtype, device=device)
    kernel_1d = torch.exp(-0.5 * (coordinates / sigma_pixels).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d[:, None] * kernel_1d[None, :]


def build_smoothed_out_of_bounds_raster(
    drivable_area,
    gaussian_sigma_meters=DEFAULT_GAUSSIAN_SIGMA_METERS,
    gaussian_truncate_sigma=DEFAULT_GAUSSIAN_TRUNCATE_SIGMA,
    *,
    device=None,
    dtype=torch.float32,
):
    """Build the map-static Gaussian out-of-bounds potential once per map.

    A one-pixel out-of-bounds frame makes sampling beyond map coverage return
    one with ``padding_mode='border'`` instead of incorrectly becoming drivable.
    """
    if not isinstance(drivable_area, DrivableAreaRaster):
        raise TypeError("drivable_area must be a DrivableAreaRaster")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating Torch dtype")
    for name, value in (
        ("gaussian_sigma_meters", gaussian_sigma_meters),
        ("gaussian_truncate_sigma", gaussian_truncate_sigma),
    ):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    resolution = drivable_area.transform.resolution_meters_per_pixel
    sigma_pixels = float(gaussian_sigma_meters) / resolution
    out_of_bounds = (~drivable_area.mask).to(device=device, dtype=dtype)[None, None]
    kernel = _gaussian_kernel_2d(
        sigma_pixels,
        float(gaussian_truncate_sigma),
        dtype,
        out_of_bounds.device,
    )
    kernel_radius = kernel.shape[-1] // 2
    padded = torch_functional.pad(out_of_bounds, (kernel_radius,) * 4, value=1.0)
    smoothed = torch_functional.conv2d(padded, kernel[None, None])[0, 0]
    potential = torch_functional.pad(smoothed, (1, 1, 1, 1), value=1.0)
    transform = RasterTransform(
        origin_x_m=drivable_area.transform.origin_x_m - resolution,
        origin_y_m=drivable_area.transform.origin_y_m - resolution,
        resolution_meters_per_pixel=resolution,
        height=drivable_area.transform.height + 2,
        width=drivable_area.transform.width + 2,
    )
    return SmoothedOutOfBoundsRaster(
        potential=potential,
        transform=transform,
        gaussian_sigma_meters=float(gaussian_sigma_meters),
        gaussian_truncate_sigma=float(gaussian_truncate_sigma),
    )


def sample_out_of_bounds_potential(xy_meters, out_of_bounds_raster):
    """Bilinearly sample the out-of-bounds potential at ``[..., x, y]`` points."""
    if not isinstance(xy_meters, torch.Tensor) or not xy_meters.is_floating_point():
        raise TypeError("xy_meters must be a floating Torch tensor")
    if xy_meters.ndim < 1 or xy_meters.shape[-1] != 2:
        raise ValueError("xy_meters must have shape [..., 2]")
    if not torch.isfinite(xy_meters).all():
        raise ValueError("xy_meters must be finite")
    if not isinstance(out_of_bounds_raster, SmoothedOutOfBoundsRaster):
        raise TypeError("out_of_bounds_raster must be a SmoothedOutOfBoundsRaster")
    potential = out_of_bounds_raster.potential
    if xy_meters.device != potential.device or xy_meters.dtype != potential.dtype:
        raise ValueError("Sample points and out-of-bounds raster must share device and dtype")

    normalized = out_of_bounds_raster.transform.world_to_normalized_grid(xy_meters)
    flattened_grid = normalized.reshape(1, -1, 1, 2)
    sampled = torch_functional.grid_sample(
        potential[None, None],
        flattened_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(xy_meters.shape[:-1])
