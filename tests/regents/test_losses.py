import math

import torch

import pufferlib.ocean.regents.losses as regents_losses
from pufferlib.ocean.regents import classic_rollout
from pufferlib.ocean.regents.geometry import (
    SmoothedOutOfBoundsRaster,
    build_smoothed_out_of_bounds_raster,
    sample_out_of_bounds_potential,
)
from pufferlib.ocean.regents.losses import (
    ReGentSCostConfig,
    _background_collision_avoidance_cost_and_diagnostics,
    combined_regents_cost,
    drivable_area_deviation_cost,
    ego_background_collision_cost,
    prepare_out_of_bounds_rasters,
)
from pufferlib.ocean.regents.state import DrivableAreaRaster, RasterTransform


def _background_cost(*args, **kwargs):
    """The optimizer reads the diagnostics too; these tests only assert the cost."""
    return _background_collision_avoidance_cost_and_diagnostics(*args, **kwargs)[0]


def _states(agent_x_positions, time_count=3, dtype=torch.float64):
    state = torch.zeros((1, len(agent_x_positions), time_count, 5), dtype=dtype)
    state[0, :, :, 0] = torch.tensor(agent_x_positions, dtype=dtype)[:, None]
    return state


def _dimensions(agent_count, dtype=torch.float64):
    length = torch.full((1, agent_count), 4.0, dtype=dtype)
    width = torch.full((1, agent_count), 2.0, dtype=dtype)
    return length, width


def _constant_raster(value, dtype=torch.float64):
    potential = torch.full((3, 3), value, dtype=dtype)
    return SmoothedOutOfBoundsRaster(
        potential,
        RasterTransform(-100.0, -100.0, 100.0, 3, 3),
        gaussian_sigma_meters=0.5,
        gaussian_truncate_sigma=3.0,
    )


def _drivable_raster():
    drivable_mask = torch.zeros((9, 9), dtype=torch.bool)
    drivable_mask[2:7, 2:7] = True
    return build_smoothed_out_of_bounds_raster(
        DrivableAreaRaster(drivable_mask, RasterTransform(-4.0, -4.0, 1.0, 9, 9)),
        gaussian_sigma_meters=1.0,
    )


def test_cost_terms_use_the_documented_reduction_semantics(monkeypatch):
    """Reference squared-center reductions, pair masking, and road sum over time."""
    default_config = ReGentSCostConfig()
    assert default_config.ego_collision_weight == 1.0
    assert default_config.background_collision_weight == 5.0
    assert default_config.drivable_area_weight == 20.0

    states = _states([0.0, 7.0, 9.0, 0.0])
    states[0, 1, 2, 0] = 0.0
    valid = torch.ones((1, 4, 3), dtype=torch.bool)
    valid[0, 1, 2] = False
    valid[0, 3] = False
    length, width = _dimensions(4)
    ego_cost = ego_background_collision_cost(
        states,
        valid,
        length,
        width,
        torch.tensor([[True, False, False, False]]),
        torch.tensor([[False, True, True, False]]),
    )
    torch.testing.assert_close(ego_cost, torch.tensor([49.0], dtype=torch.float64))

    two_step_valid = torch.ones((1, 3, 2), dtype=torch.bool)
    three_length, three_width = _dimensions(3)
    background_mask = torch.tensor([[False, True, True]])
    torch.testing.assert_close(
        _background_cost(
            _states([0.0, 7.0, 14.0], time_count=2),
            two_step_valid,
            three_length,
            three_width,
            background_mask,
            background_mask,
            truncation_meters=1.25,
        ),
        torch.tensor([-1.5625], dtype=torch.float64),
    )
    torch.testing.assert_close(
        _background_cost(
            _states([0.0, 7.0, 8.0], time_count=2),
            two_step_valid,
            three_length,
            three_width,
            background_mask,
            background_mask,
            truncation_meters=1.25,
        ),
        torch.tensor([-1.0], dtype=torch.float64),
    )
    # A single remaining valid actor has no pair, so the term is neutral.
    one_actor_valid = torch.ones((1, 3, 2), dtype=torch.bool)
    one_actor_valid[0, 2] = False
    torch.testing.assert_close(
        _background_cost(
            _states([0.0, 3.0, 6.0], time_count=2),
            one_actor_valid,
            three_length,
            three_width,
            background_mask,
            background_mask,
        ),
        torch.zeros(1, dtype=torch.float64),
    )

    # Released ReGentS scores this term over the adversary trajectories alone, so a pair
    # counts only when both endpoints are optimized. Agents one and two overlap, but they
    # are frozen traffic here, and the method never shapes a distance it cannot change.
    mixed_states = _states([30.0, 10.0, 10.0])
    mixed_valid = torch.ones((1, 3, 3), dtype=torch.bool)
    mixed_length, mixed_width = _dimensions(3)
    all_background = torch.tensor([[True, True, True]])
    single_optimized = torch.tensor([[True, False, False]])
    torch.testing.assert_close(
        _background_cost(
            mixed_states,
            mixed_valid,
            mixed_length,
            mixed_width,
            all_background,
            single_optimized,
        ),
        torch.zeros(1, dtype=torch.float64),
    )
    # Two optimized adversaries do pair, and a separation past the threshold truncates.
    separated_optimized = torch.tensor([[True, True, False]])
    torch.testing.assert_close(
        _background_cost(
            mixed_states,
            mixed_valid,
            mixed_length,
            mixed_width,
            all_background,
            separated_optimized,
        ),
        torch.tensor([-1.5625], dtype=torch.float64),
    )
    # The same overlapping pair drives the term to its floor once both are optimized.
    overlapping_optimized = torch.tensor([[False, True, True]])
    torch.testing.assert_close(
        _background_cost(
            mixed_states,
            mixed_valid,
            mixed_length,
            mixed_width,
            all_background,
            overlapping_optimized,
        ),
        torch.zeros(1, dtype=torch.float64),
    )
    touching_states = _states([0.0, 0.0])
    torch.testing.assert_close(
        _background_cost(
            touching_states,
            torch.ones((1, 2, 3), dtype=torch.bool),
            *_dimensions(2),
            torch.ones((1, 2), dtype=torch.bool),
            torch.tensor([[True, False]]),
        ),
        torch.zeros(1, dtype=torch.float64),
    )
    torch.testing.assert_close(
        _background_cost(
            _states([0.0, 5.25]),
            torch.ones((1, 2, 3), dtype=torch.bool),
            *_dimensions(2),
            torch.ones((1, 2), dtype=torch.bool),
            torch.ones((1, 2), dtype=torch.bool),
        ),
        torch.tensor([-1.5625], dtype=torch.float64),
    )
    torch.testing.assert_close(
        _background_cost(
            _states([0.0, 10.0]),
            torch.ones((1, 2, 3), dtype=torch.bool),
            *_dimensions(2),
            torch.tensor([[True, False]]),
            torch.tensor([[True, False]]),
        ),
        torch.zeros(1, dtype=torch.float64),
    )

    diagnostic_states = _states([100.0, 0.0, 4.2, 8.8])
    diagnostic_costs = combined_regents_cost(
        diagnostic_states,
        torch.ones((1, 4, 3), dtype=torch.bool),
        *_dimensions(4),
        torch.tensor([[True, False, False, False]]),
        torch.tensor([[False, True, False, False]]),
        torch.tensor([[False, True, True, True]]),
        # Both endpoints of the winning pair are optimized, as the released term requires.
        torch.tensor([[False, True, True, False]]),
        (_constant_raster(0.0),),
    )
    torch.testing.assert_close(
        diagnostic_costs.background_collision,
        torch.tensor([-1.5625], dtype=torch.float64),
    )
    assert diagnostic_costs.background_collision_first_agent_idx.item() == 1
    assert diagnostic_costs.background_collision_second_agent_idx.item() == 2
    assert diagnostic_costs.background_collision_timestep_idx.item() == 0
    assert diagnostic_costs.background_collision_truncated.item()
    torch.testing.assert_close(
        diagnostic_costs.background_collision_signed_distance_meters,
        torch.tensor([0.2], dtype=torch.float64),
    )
    unchunked = _background_cost(
        diagnostic_states,
        torch.ones((1, 4, 3), dtype=torch.bool),
        *_dimensions(4),
        torch.tensor([[False, True, True, True]]),
        torch.tensor([[False, True, True, True]]),
    )
    monkeypatch.setattr(regents_losses, "PAIRWISE_DISTANCE_CHUNK_SIZE", 1)
    torch.testing.assert_close(
        _background_cost(
            diagnostic_states,
            torch.ones((1, 4, 3), dtype=torch.bool),
            *_dimensions(4),
            torch.tensor([[False, True, True, True]]),
            torch.tensor([[False, True, True, True]]),
        ),
        unchunked,
    )

    drivable_valid = torch.ones((1, 3, 3), dtype=torch.bool)
    drivable_valid[0, 2, 2] = False
    torch.testing.assert_close(
        drivable_area_deviation_cost(
            _states([0.0, 2.0, 4.0]),
            drivable_valid,
            three_length,
            three_width,
            torch.tensor([[False, True, True]]),
            (_constant_raster(0.25),),
        ),
        torch.tensor([5.0], dtype=torch.float64),
    )

    # The road cost is differentiable at an out-of-bounds pose.
    raster = _drivable_raster()
    baseline_states = _states([1.5], time_count=2, dtype=torch.float32).requires_grad_()
    unit_length = torch.ones((1, 1), dtype=torch.float32)
    unit_width = torch.ones((1, 1), dtype=torch.float32)
    optimized_mask = torch.ones((1, 1), dtype=torch.bool)
    absolute_cost = drivable_area_deviation_cost(
        baseline_states, torch.ones((1, 1, 2), dtype=torch.bool), unit_length, unit_width, optimized_mask, (raster,)
    ).sum()
    absolute_gradient = torch.autograd.grad(absolute_cost, baseline_states)[0]
    assert absolute_gradient.abs().sum() > 0

    # At interior grid nodes, convolution must match the radial density carrying unit
    # mass, so a fully out-of-bounds sample costs one at any raster resolution.
    mask = torch.ones((15, 15), dtype=torch.bool)
    mask[7, 7] = False
    density_raster = prepare_out_of_bounds_rasters(
        (DrivableAreaRaster(mask, RasterTransform(-3.5, -3.5, 0.5, 15, 15)),),
        dtype=torch.float64,
    )[0]
    locations = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float64)
    kernel_1d = torch.exp(-0.5 * torch.arange(-3, 4, dtype=torch.float64).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    assert math.isclose(float(kernel_1d.sum()), 1.0)
    expected_density = kernel_1d[3] * torch.stack((kernel_1d[3], kernel_1d[2]))
    torch.testing.assert_close(sample_out_of_bounds_potential(locations, density_raster), expected_density)
    between_nodes = torch.tensor([[0.2, 0.0]], dtype=torch.float64, requires_grad=True)
    density_gradient = torch.autograd.grad(
        sample_out_of_bounds_potential(between_nodes, density_raster).sum(), between_nodes
    )[0]
    torch.testing.assert_close(density_gradient[0, 0], (expected_density[1] - expected_density[0]) / 0.5)


def test_cost_gradients_reach_actions_and_the_combined_loss_decreases():
    """Repulsion, drivable attraction, action-space gradients, and end-to-end descent."""
    repelled_states = _states([0.0, 0.0, 0.5], time_count=2).requires_grad_()
    repulsion = _background_cost(
        repelled_states,
        torch.ones((1, 3, 2), dtype=torch.bool),
        *_dimensions(3),
        torch.tensor([[False, True, True]]),
        torch.tensor([[False, True, True]]),
    ).sum()
    repulsion_gradient = torch.autograd.grad(repulsion, repelled_states)[0]
    assert torch.isfinite(repulsion_gradient).all()
    assert repulsion_gradient[0, 2, :, 0].abs().sum() > 0
    torch.testing.assert_close(repulsion_gradient[0, 2, :, 0], torch.full((2,), -0.5, dtype=torch.float64))

    multi_candidate_states = _states([0.0, 0.2, 0.8, 1.5], time_count=2).requires_grad_()
    multi_candidate_cost = _background_cost(
        multi_candidate_states,
        torch.ones((1, 4, 2), dtype=torch.bool),
        *_dimensions(4),
        torch.ones((1, 4), dtype=torch.bool),
        torch.ones((1, 4), dtype=torch.bool),
    ).sum()
    multi_candidate_gradient = torch.autograd.grad(multi_candidate_cost, multi_candidate_states)[0]
    assert multi_candidate_gradient[0, :2, :, 0].abs().sum() > 0
    torch.testing.assert_close(
        multi_candidate_gradient[0, 2:, :, 0],
        torch.zeros((2, 2), dtype=torch.float64),
    )

    raster = _drivable_raster()
    offroad_states = _states([0.0, 1.5], time_count=2, dtype=torch.float32).requires_grad_()
    drivable = drivable_area_deviation_cost(
        offroad_states,
        torch.ones((1, 2, 2), dtype=torch.bool),
        torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        torch.tensor([[False, True]]),
        (raster,),
    ).sum()
    drivable_gradient = torch.autograd.grad(drivable, offroad_states)[0]
    assert torch.isfinite(drivable_gradient).all()
    assert drivable_gradient[0, 1, :, 0].sum() > 0

    # The ego-collision term must push through the dynamics into both action channels.
    transition_count = 5
    actions = torch.zeros((1, 1, transition_count, 2), dtype=torch.float32, requires_grad=True)
    adversary = classic_rollout(
        torch.tensor([[[0.0, 0.0, 0.0, 4.0, 0.0]]], dtype=torch.float32),
        actions,
        torch.ones((1, 1, transition_count), dtype=torch.bool),
        torch.tensor([[2.7]], dtype=torch.float32),
        torch.tensor([[20.0]], dtype=torch.float32),
        0.2,
    )
    time_count = transition_count + 1
    ego = torch.zeros((1, 1, time_count, 5), dtype=torch.float32)
    ego[..., 0] = 5.0
    ego[..., 1] = 1.0
    action_cost = ego_background_collision_cost(
        torch.cat((ego, adversary), dim=1),
        torch.ones((1, 2, time_count), dtype=torch.bool),
        torch.tensor([[4.0, 4.0]], dtype=torch.float32),
        torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        torch.tensor([[True, False]]),
        torch.tensor([[False, True]]),
    ).sum()
    action_gradient = torch.autograd.grad(action_cost, actions)[0]
    assert torch.isfinite(action_gradient).all()
    assert torch.any(action_gradient[..., 0] != 0)
    assert torch.any(action_gradient[..., 1] != 0)

    candidate_x = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([candidate_x], lr=0.1)
    combined_valid = torch.ones((1, 3, 3), dtype=torch.bool)
    combined_length, combined_width = _dimensions(3)
    ego_mask = torch.tensor([[True, False, False]])
    candidate_mask = torch.tensor([[False, True, False]])
    losses = []
    for _ in range(20):
        all_x = torch.cat(
            (
                candidate_x.new_zeros((1, 1, 3)),
                candidate_x.expand(1, 1, 3),
                candidate_x.new_full((1, 1, 3), 30.0),
            ),
            dim=1,
        )
        states = torch.zeros((1, 3, 3, 5), dtype=torch.float64)
        states[..., 0] = all_x
        loss = combined_regents_cost(
            states,
            combined_valid,
            combined_length,
            combined_width,
            ego_mask,
            candidate_mask,
            torch.tensor([[False, True, True]]),
            candidate_mask,
            (_constant_raster(0.0),),
            ReGentSCostConfig(),
        ).total.sum()
        losses.append(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.isfinite(torch.stack(losses)).all()
    assert losses[-1] < losses[0]
