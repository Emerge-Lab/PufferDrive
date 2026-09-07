import math

import torch

from pufferlib.ocean.regents.geometry import (
    SmoothedOutOfBoundsRaster,
    box_separation_lower_bound,
    build_smoothed_out_of_bounds_raster,
    oriented_box_corners,
    sample_out_of_bounds_potential,
    signed_box_distance,
)
from pufferlib.ocean.regents.state import DrivableAreaRaster, RasterTransform


def test_oriented_box_distance_matches_the_c_sat_algorithm_and_is_differentiable():
    """Corner layout, separation and penetration signs, SAT agreement, and gradients."""
    boxes = torch.tensor(
        [
            [0.0, 0.0, 4.0, 2.0, 0.0],
            [7.0, 0.0, 4.0, 2.0, 0.0],
            [3.0, 0.0, 4.0, 2.0, 0.0],
            [4.0, 0.0, 4.0, 2.0, 0.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        oriented_box_corners(boxes[:1])[0],
        torch.tensor([[2.0, 1.0], [-2.0, 1.0], [-2.0, -1.0], [2.0, -1.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        signed_box_distance(boxes[:1], boxes[1:]),
        torch.tensor([3.0, -1.0, 0.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )

    rotated = torch.tensor(
        [[1.0, 0.0, 4.0, 2.0, math.pi / 4.0], [8.0, 0.0, 4.0, 2.0, math.pi / 4.0]], dtype=torch.float64
    )
    rotated_distances = signed_box_distance(boxes[:1], rotated)
    assert rotated_distances[0] < 0 and rotated_distances[1] > 0

    # The distance sign must agree with the separating-axis test the C oracle uses.
    generator = torch.Generator().manual_seed(19402)
    boxes_a = torch.empty((128, 5), dtype=torch.float64)
    boxes_b = torch.empty((128, 5), dtype=torch.float64)
    for random_boxes in (boxes_a, boxes_b):
        random_boxes[:, :2].uniform_(-8.0, 8.0, generator=generator)
        random_boxes[:, 2:4].uniform_(1.0, 5.0, generator=generator)
        random_boxes[:, 4].uniform_(-math.pi, math.pi, generator=generator)
    axes = torch.stack(
        (
            torch.stack((torch.cos(boxes_a[:, 4]), torch.sin(boxes_a[:, 4])), dim=-1),
            torch.stack((-torch.sin(boxes_a[:, 4]), torch.cos(boxes_a[:, 4])), dim=-1),
            torch.stack((torch.cos(boxes_b[:, 4]), torch.sin(boxes_b[:, 4])), dim=-1),
            torch.stack((-torch.sin(boxes_b[:, 4]), torch.cos(boxes_b[:, 4])), dim=-1),
        ),
        dim=1,
    )
    projections_a = torch.einsum("ncd,nkd->nkc", oriented_box_corners(boxes_a), axes)
    projections_b = torch.einsum("ncd,nkd->nkc", oriented_box_corners(boxes_b), axes)
    c_sat_collision = (
        (projections_a.max(dim=-1).values >= projections_b.min(dim=-1).values)
        & (projections_a.min(dim=-1).values <= projections_b.max(dim=-1).values)
    ).all(dim=-1)
    distances = signed_box_distance(boxes_a, boxes_b)
    away_from_boundary = distances.abs() > 1e-8
    assert torch.equal(distances[away_from_boundary] < 0, c_sat_collision[away_from_boundary])

    moving_box = torch.tensor([[0.2, 0.4, 4.0, 1.8, 0.15]], dtype=torch.float64, requires_grad=True)
    fixed_box = torch.tensor([[6.0, 1.1, 3.8, 2.0, -0.2]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda value: signed_box_distance(value, fixed_box), (moving_box,))
    gradient_x = torch.autograd.grad(signed_box_distance(moving_box, fixed_box).sum(), moving_box)[0][0, 0]
    epsilon = 1e-5
    plus = moving_box.detach().clone()
    minus = moving_box.detach().clone()
    plus[0, 0] += epsilon
    minus[0, 0] -= epsilon
    finite_difference = (signed_box_distance(plus, fixed_box) - signed_box_distance(minus, fixed_box)) / (2 * epsilon)
    torch.testing.assert_close(gradient_x, finite_difference[0], atol=1e-6, rtol=1e-5)


def test_out_of_bounds_raster_samples_pixel_centers_and_stays_smooth_and_bounded():
    """Transform convention, out-of-raster handling, Gaussian shape, and spatial gradient."""
    transform = RasterTransform(origin_x_m=-1.0, origin_y_m=2.0, resolution_meters_per_pixel=0.5, height=3, width=3)
    potential = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.25, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    raster = SmoothedOutOfBoundsRaster(potential, transform, 0.5, 3.0)
    points = torch.tensor([[-0.5, 2.5], [-1.0, 2.0], [-10.0, 2.5], [10.0, 2.5]], dtype=torch.float64)
    torch.testing.assert_close(
        sample_out_of_bounds_potential(points, raster),
        torch.tensor([0.25, 1.0, 1.0, 1.0], dtype=torch.float64),
    )
    differentiable_point = torch.tensor([[-0.4, 2.4]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: sample_out_of_bounds_potential(value, raster), (differentiable_point,)
    )

    drivable_mask = torch.zeros((9, 9), dtype=torch.bool)
    drivable_mask[2:7, 2:7] = True
    smoothed = build_smoothed_out_of_bounds_raster(
        DrivableAreaRaster(drivable_mask, RasterTransform(-4.0, -4.0, 1.0, 9, 9)), gaussian_sigma_meters=1.0
    )
    assert smoothed.potential.shape == (11, 11)
    assert torch.all((smoothed.potential >= 0) & (smoothed.potential <= 1))
    assert smoothed.potential[5, 5] < smoothed.potential[3, 5] < smoothed.potential[1, 5]

    point = torch.tensor([[1.5, 0.0]], dtype=torch.float32, requires_grad=True)
    gradient = torch.autograd.grad(sample_out_of_bounds_potential(point, smoothed).sum(), point)[0]
    assert torch.isfinite(gradient).all() and gradient[0, 0] > 0


def test_circumscribed_radius_bound_never_rules_out_a_true_contact():
    """The broad phase must be conservative, or a collision gate would miss contacts."""
    generator = torch.Generator().manual_seed(60915)
    boxes_a = torch.empty((4096, 5), dtype=torch.float64)
    boxes_b = torch.empty((4096, 5), dtype=torch.float64)
    for random_boxes in (boxes_a, boxes_b):
        random_boxes[:, :2].uniform_(-8.0, 8.0, generator=generator)
        random_boxes[:, 2:4].uniform_(1.0, 5.0, generator=generator)
        random_boxes[:, 4].uniform_(-math.pi, math.pi, generator=generator)

    bound = box_separation_lower_bound(boxes_a, boxes_b)
    exact = signed_box_distance(boxes_a, boxes_b)
    assert torch.all(bound <= exact + 1e-9)

    # A pair the bound rejects must be separated at that tolerance, at every tolerance
    # the gate is ever configured with.
    for tolerance_meters in (0.0, 0.5, 2.0, 5.0):
        rejected = bound > tolerance_meters
        assert not torch.any(exact[rejected] <= tolerance_meters)
        assert torch.any(~rejected)
