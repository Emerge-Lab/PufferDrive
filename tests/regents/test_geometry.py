import math

import pytest
import torch

from pufferlib.ocean.regents.geometry import (
    SmoothedOutOfBoundsRaster,
    build_smoothed_out_of_bounds_raster,
    oriented_box_corners,
    sample_out_of_bounds_potential,
    signed_box_distance,
)
from pufferlib.ocean.regents.state import DrivableAreaRaster, RasterTransform


def test_axis_aligned_box_corners_and_signed_distances():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 4.0, 2.0, 0.0],
            [7.0, 0.0, 4.0, 2.0, 0.0],
            [3.0, 0.0, 4.0, 2.0, 0.0],
            [4.0, 0.0, 4.0, 2.0, 0.0],
        ],
        dtype=torch.float64,
    )
    expected_corners = torch.tensor(
        [[2.0, 1.0], [-2.0, 1.0], [-2.0, -1.0], [2.0, -1.0]],
        dtype=torch.float64,
    )

    torch.testing.assert_close(oriented_box_corners(boxes[:1])[0], expected_corners)
    distances = signed_box_distance(boxes[:1], boxes[1:])
    torch.testing.assert_close(distances, torch.tensor([3.0, -1.0, 0.0], dtype=torch.float64), atol=1e-12, rtol=0)


def test_rotated_overlap_sign_agrees_with_sat_away_from_boundary():
    reference = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.0]], dtype=torch.float64)
    rotated = torch.tensor(
        [
            [1.0, 0.0, 4.0, 2.0, math.pi / 4.0],
            [8.0, 0.0, 4.0, 2.0, math.pi / 4.0],
        ],
        dtype=torch.float64,
    )
    distances = signed_box_distance(reference, rotated)
    assert distances[0] < 0
    assert distances[1] > 0


def test_signed_distance_collision_sign_matches_c_sat_algorithm():
    generator = torch.Generator().manual_seed(19402)
    boxes_a = torch.empty((128, 5), dtype=torch.float64)
    boxes_b = torch.empty((128, 5), dtype=torch.float64)
    for boxes in (boxes_a, boxes_b):
        boxes[:, :2].uniform_(-8.0, 8.0, generator=generator)
        boxes[:, 2:4].uniform_(1.0, 5.0, generator=generator)
        boxes[:, 4].uniform_(-math.pi, math.pi, generator=generator)

    corners_a = oriented_box_corners(boxes_a)
    corners_b = oriented_box_corners(boxes_b)
    axes = torch.stack(
        (
            torch.stack((torch.cos(boxes_a[:, 4]), torch.sin(boxes_a[:, 4])), dim=-1),
            torch.stack((-torch.sin(boxes_a[:, 4]), torch.cos(boxes_a[:, 4])), dim=-1),
            torch.stack((torch.cos(boxes_b[:, 4]), torch.sin(boxes_b[:, 4])), dim=-1),
            torch.stack((-torch.sin(boxes_b[:, 4]), torch.cos(boxes_b[:, 4])), dim=-1),
        ),
        dim=1,
    )
    projections_a = torch.einsum("ncd,nkd->nkc", corners_a, axes)
    projections_b = torch.einsum("ncd,nkd->nkc", corners_b, axes)
    overlaps = (projections_a.max(dim=-1).values >= projections_b.min(dim=-1).values) & (
        projections_a.min(dim=-1).values <= projections_b.max(dim=-1).values
    )
    c_sat_collision = overlaps.all(dim=-1)
    distances = signed_box_distance(boxes_a, boxes_b)
    away_from_boundary = distances.abs() > 1e-8
    assert torch.equal(distances[away_from_boundary] < 0, c_sat_collision[away_from_boundary])


def test_signed_box_distance_gradcheck_and_finite_difference():
    moving_box = torch.tensor([[0.2, 0.4, 4.0, 1.8, 0.15]], dtype=torch.float64, requires_grad=True)
    fixed_box = torch.tensor([[6.0, 1.1, 3.8, 2.0, -0.2]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda value: signed_box_distance(value, fixed_box), (moving_box,))

    distance = signed_box_distance(moving_box, fixed_box).sum()
    gradient_x = torch.autograd.grad(distance, moving_box)[0][0, 0]
    epsilon = 1e-5
    plus = moving_box.detach().clone()
    minus = moving_box.detach().clone()
    plus[0, 0] += epsilon
    minus[0, 0] -= epsilon
    finite_difference = (signed_box_distance(plus, fixed_box) - signed_box_distance(minus, fixed_box)) / (2 * epsilon)
    torch.testing.assert_close(gradient_x, finite_difference[0], atol=1e-6, rtol=1e-5)


def test_world_to_normalized_transform_samples_pixel_centers_and_outside_is_oob():
    transform = RasterTransform(origin_x_m=-1.0, origin_y_m=2.0, resolution_meters_per_pixel=0.5, height=3, width=3)
    potential = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 0.25, 1.0], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    raster = SmoothedOutOfBoundsRaster(potential, transform, 0.5, 3.0)
    points = torch.tensor([[-0.5, 2.5], [-1.0, 2.0], [-10.0, 2.5], [10.0, 2.5]], dtype=torch.float64)
    sampled = sample_out_of_bounds_potential(points, raster)
    torch.testing.assert_close(sampled, torch.tensor([0.25, 1.0, 1.0, 1.0], dtype=torch.float64))

    differentiable_point = torch.tensor([[-0.4, 2.4]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: sample_out_of_bounds_potential(value, raster),
        (differentiable_point,),
    )


def test_gaussian_raster_is_smooth_bounded_and_has_spatial_gradient():
    drivable_mask = torch.zeros((9, 9), dtype=torch.bool)
    drivable_mask[2:7, 2:7] = True
    drivable = DrivableAreaRaster(drivable_mask, RasterTransform(-4.0, -4.0, 1.0, 9, 9))
    raster = build_smoothed_out_of_bounds_raster(drivable, gaussian_sigma_meters=1.0)
    assert raster.potential.shape == (11, 11)
    assert torch.all((raster.potential >= 0) & (raster.potential <= 1))
    assert raster.potential[5, 5] < raster.potential[3, 5] < raster.potential[1, 5]

    point = torch.tensor([[1.5, 0.0]], dtype=torch.float32, requires_grad=True)
    sample = sample_out_of_bounds_potential(point, raster)
    gradient = torch.autograd.grad(sample.sum(), point)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 0] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_geometry_cpu_gpu_consistency():
    boxes_a = torch.tensor([[0.2, 0.4, 4.0, 1.8, 0.15]], dtype=torch.float32)
    boxes_b = torch.tensor([[6.0, 1.1, 3.8, 2.0, -0.2]], dtype=torch.float32)
    cpu_distance = signed_box_distance(boxes_a, boxes_b)
    gpu_distance = signed_box_distance(boxes_a.cuda(), boxes_b.cuda()).cpu()
    torch.testing.assert_close(gpu_distance, cpu_distance, atol=1e-5, rtol=1e-5)
