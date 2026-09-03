import pytest
import torch

from pufferlib.ocean.regents import classic_rollout
from pufferlib.ocean.regents.geometry import SmoothedOutOfBoundsRaster, build_smoothed_out_of_bounds_raster
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


def test_ego_collision_cost_uses_joint_valid_mean_then_candidate_minimum():
    states = _states([0.0, 7.0, 9.0, 0.0])
    states[0, 1, 2, 0] = 0.0
    valid = torch.ones((1, 4, 3), dtype=torch.bool)
    valid[0, 1, 2] = False
    valid[0, 3] = False
    length, width = _dimensions(4)
    ego_mask = torch.tensor([[True, False, False, False]])
    candidate_mask = torch.tensor([[False, True, True, False]])

    cost = ego_background_collision_cost(states, valid, length, width, ego_mask, candidate_mask)
    torch.testing.assert_close(cost, torch.tensor([3.0], dtype=torch.float64))


def test_background_collision_cost_truncates_separation_and_penalizes_overlap():
    valid = torch.ones((1, 3, 2), dtype=torch.bool)
    length, width = _dimensions(3)
    background_mask = torch.tensor([[False, True, True]])

    separated = background_collision_avoidance_cost(
        _states([0.0, 7.0, 14.0], time_count=2),
        valid,
        length,
        width,
        background_mask,
        truncation_meters=1.25,
    )
    overlapping = background_collision_avoidance_cost(
        _states([0.0, 7.0, 10.0], time_count=2),
        valid,
        length,
        width,
        background_mask,
        truncation_meters=1.25,
    )
    torch.testing.assert_close(separated, torch.tensor([-1.25], dtype=torch.float64))
    torch.testing.assert_close(overlapping, torch.tensor([1.0], dtype=torch.float64))


def test_background_collision_masks_invalid_pairs_and_one_actor_is_neutral():
    states = _states([0.0, 3.0, 6.0], time_count=2)
    valid = torch.ones((1, 3, 2), dtype=torch.bool)
    valid[0, 2] = False
    length, width = _dimensions(3)
    background_mask = torch.tensor([[False, True, True]])
    cost = background_collision_avoidance_cost(states, valid, length, width, background_mask)
    torch.testing.assert_close(cost, torch.zeros(1, dtype=torch.float64))


def test_background_collision_cost_has_finite_repulsion_gradient():
    states = _states([0.0, 0.0, 4.5], time_count=2).requires_grad_()
    valid = torch.ones((1, 3, 2), dtype=torch.bool)
    length, width = _dimensions(3)
    cost = background_collision_avoidance_cost(
        states,
        valid,
        length,
        width,
        torch.tensor([[False, True, True]]),
    ).sum()
    gradient = torch.autograd.grad(cost, states)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 2, :, 0].abs().sum() > 0


def test_drivable_cost_sums_corner_potential_per_vehicle_and_averages_valid_steps():
    states = _states([0.0, 2.0, 4.0])
    valid = torch.ones((1, 3, 3), dtype=torch.bool)
    valid[0, 2, 2] = False
    length, width = _dimensions(3)
    optimized_mask = torch.tensor([[False, True, True]])
    cost = drivable_area_deviation_cost(
        states,
        valid,
        length,
        width,
        optimized_mask,
        (_constant_raster(0.25),),
    )
    torch.testing.assert_close(cost, torch.tensor([2.0], dtype=torch.float64))


def test_drivable_cost_has_finite_gradient_toward_drivable_area():
    drivable_mask = torch.zeros((9, 9), dtype=torch.bool)
    drivable_mask[2:7, 2:7] = True
    raster = build_smoothed_out_of_bounds_raster(
        DrivableAreaRaster(drivable_mask, RasterTransform(-4.0, -4.0, 1.0, 9, 9)),
        gaussian_sigma_meters=1.0,
    )
    states = _states([0.0, 1.5], time_count=2, dtype=torch.float32).requires_grad_()
    valid = torch.ones((1, 2, 2), dtype=torch.bool)
    length = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    width = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    cost = drivable_area_deviation_cost(
        states,
        valid,
        length,
        width,
        torch.tensor([[False, True]]),
        (raster,),
    ).sum()
    gradient = torch.autograd.grad(cost, states)[0]
    assert torch.isfinite(gradient).all()
    assert gradient[0, 1, :, 0].sum() > 0


def test_collision_cost_gradients_reach_acceleration_and_steering_actions():
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
    states = torch.cat((ego, adversary), dim=1)
    cost = ego_background_collision_cost(
        states,
        torch.ones((1, 2, time_count), dtype=torch.bool),
        torch.tensor([[4.0, 4.0]], dtype=torch.float32),
        torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        torch.tensor([[True, False]]),
        torch.tensor([[False, True]]),
    ).sum()
    gradient = torch.autograd.grad(cost, actions)[0]
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient[..., 0] != 0)
    assert torch.any(gradient[..., 1] != 0)


def test_combined_loss_decreases_in_small_synthetic_optimization():
    candidate_x = torch.tensor(9.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([candidate_x], lr=0.1)
    valid = torch.ones((1, 3, 3), dtype=torch.bool)
    length, width = _dimensions(3)
    ego_mask = torch.tensor([[True, False, False]])
    candidate_mask = torch.tensor([[False, True, False]])
    background_mask = torch.tensor([[False, True, True]])
    raster = _constant_raster(0.0)
    config = ReGentSCostConfig()

    losses = []
    for _ in range(20):
        ego_x = candidate_x.new_zeros((1, 1, 3))
        optimized_x = candidate_x.expand(1, 1, 3)
        other_x = candidate_x.new_full((1, 1, 3), 30.0)
        all_x = torch.cat((ego_x, optimized_x, other_x), dim=1)
        states = torch.zeros((1, 3, 3, 5), dtype=torch.float64)
        states = states.clone()
        states[..., 0] = all_x
        costs = combined_regents_cost(
            states,
            valid,
            length,
            width,
            ego_mask,
            candidate_mask,
            background_mask,
            candidate_mask,
            (raster,),
            config,
        )
        loss = costs.total.sum()
        losses.append(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert torch.isfinite(torch.stack(losses)).all()
    assert losses[-1] < losses[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_combined_cost_cpu_gpu_consistency():
    states = _states([0.0, 7.0, 14.0], dtype=torch.float32)
    valid = torch.ones((1, 3, 3), dtype=torch.bool)
    length, width = _dimensions(3, dtype=torch.float32)
    ego_mask = torch.tensor([[True, False, False]])
    candidate_mask = torch.tensor([[False, True, False]])
    background_mask = torch.tensor([[False, True, True]])
    raster = _constant_raster(0.25, dtype=torch.float32)
    cpu = combined_regents_cost(
        states,
        valid,
        length,
        width,
        ego_mask,
        candidate_mask,
        background_mask,
        candidate_mask,
        (raster,),
    )
    gpu = combined_regents_cost(
        states.cuda(),
        valid.cuda(),
        length.cuda(),
        width.cuda(),
        ego_mask.cuda(),
        candidate_mask.cuda(),
        background_mask.cuda(),
        candidate_mask.cuda(),
        (raster.to(device="cuda"),),
    )
    for cpu_value, gpu_value in zip(
        (cpu.ego_collision, cpu.background_collision, cpu.drivable_area, cpu.total),
        (gpu.ego_collision, gpu.background_collision, gpu.drivable_area, gpu.total),
    ):
        torch.testing.assert_close(gpu_value.cpu(), cpu_value, atol=1e-5, rtol=1e-5)
