import torch

from pufferlib.ocean.regents import classic_rollout
from pufferlib.ocean.regents.geometry import (
    SmoothedOutOfBoundsRaster,
    build_smoothed_out_of_bounds_raster,
    oriented_box_corners,
    sample_out_of_bounds_potential,
)
from pufferlib.ocean.regents.losses import (
    ReGentSCostConfig,
    background_collision_avoidance_cost,
    combined_regents_cost,
    drivable_area_deviation_cost,
    ego_background_collision_cost,
)
from pufferlib.ocean.regents.state import DrivableAreaRaster, RasterTransform


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


def test_cost_terms_use_the_documented_reduction_semantics():
    """Ego mean-then-min, background truncation and masking, drivable sum-then-average."""
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
    torch.testing.assert_close(ego_cost, torch.tensor([3.0], dtype=torch.float64))

    two_step_valid = torch.ones((1, 3, 2), dtype=torch.bool)
    three_length, three_width = _dimensions(3)
    background_mask = torch.tensor([[False, True, True]])
    torch.testing.assert_close(
        background_collision_avoidance_cost(
            _states([0.0, 7.0, 14.0], time_count=2),
            two_step_valid,
            three_length,
            three_width,
            background_mask,
            truncation_meters=1.25,
        ),
        torch.tensor([-1.25], dtype=torch.float64),
    )
    torch.testing.assert_close(
        background_collision_avoidance_cost(
            _states([0.0, 7.0, 10.0], time_count=2),
            two_step_valid,
            three_length,
            three_width,
            background_mask,
            truncation_meters=1.25,
        ),
        torch.tensor([1.0], dtype=torch.float64),
    )
    # A single remaining valid actor has no pair, so the term is neutral.
    one_actor_valid = torch.ones((1, 3, 2), dtype=torch.bool)
    one_actor_valid[0, 2] = False
    torch.testing.assert_close(
        background_collision_avoidance_cost(
            _states([0.0, 3.0, 6.0], time_count=2),
            one_actor_valid,
            three_length,
            three_width,
            background_mask,
        ),
        torch.zeros(1, dtype=torch.float64),
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
        torch.tensor([2.0], dtype=torch.float64),
    )

    # Charging only the increase over a baseline shifts the value to zero at the
    # baseline pose while leaving the gradient identical to the absolute cost.
    raster = _drivable_raster()
    baseline_states = _states([1.5], time_count=2, dtype=torch.float32).requires_grad_()
    unit_length = torch.ones((1, 1), dtype=torch.float32)
    unit_width = torch.ones((1, 1), dtype=torch.float32)
    optimized_mask = torch.ones((1, 1), dtype=torch.bool)
    boxes = torch.stack(
        (
            baseline_states[..., 0],
            baseline_states[..., 1],
            unit_length[..., None].expand_as(baseline_states[..., 0]),
            unit_width[..., None].expand_as(baseline_states[..., 0]),
            baseline_states[..., 2],
        ),
        dim=-1,
    )
    baseline = sample_out_of_bounds_potential(oriented_box_corners(boxes), raster).detach()
    absolute_cost = drivable_area_deviation_cost(
        baseline_states, torch.ones((1, 1, 2), dtype=torch.bool), unit_length, unit_width, optimized_mask, (raster,)
    ).sum()
    shifted_cost = drivable_area_deviation_cost(
        baseline_states,
        torch.ones((1, 1, 2), dtype=torch.bool),
        unit_length,
        unit_width,
        optimized_mask,
        (raster,),
        baseline,
    ).sum()
    absolute_gradient = torch.autograd.grad(absolute_cost, baseline_states, retain_graph=True)[0]
    shifted_gradient = torch.autograd.grad(shifted_cost, baseline_states)[0]
    torch.testing.assert_close(shifted_cost, torch.zeros_like(shifted_cost))
    assert absolute_gradient.abs().sum() > 0
    torch.testing.assert_close(shifted_gradient, absolute_gradient)


def test_cost_gradients_reach_actions_and_the_combined_loss_decreases():
    """Repulsion, drivable attraction, action-space gradients, and end-to-end descent."""
    repelled_states = _states([0.0, 0.0, 4.5], time_count=2).requires_grad_()
    repulsion = background_collision_avoidance_cost(
        repelled_states,
        torch.ones((1, 3, 2), dtype=torch.bool),
        *_dimensions(3),
        torch.tensor([[False, True, True]]),
    ).sum()
    repulsion_gradient = torch.autograd.grad(repulsion, repelled_states)[0]
    assert torch.isfinite(repulsion_gradient).all()
    assert repulsion_gradient[0, 2, :, 0].abs().sum() > 0

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
