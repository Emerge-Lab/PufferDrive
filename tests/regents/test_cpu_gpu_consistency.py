"""CUDA parity for every differentiable stage. Skipped wholesale without a GPU."""

import math

import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents import classic_rollout, estimate_expert_actions
from pufferlib.ocean.regents.geometry import signed_box_distance
from pufferlib.ocean.regents.losses import combined_regents_cost
from pufferlib.ocean.regents.optimizer import ReGentSOptimizationConfig, optimize_frozen_ego_scenario
from tests.regents.test_losses import _constant_raster, _dimensions, _states
from tests.regents.test_optimizer import _optimization_config, _scenario, _straight_track


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")

CONSISTENCY_ATOL = 1e-5
OPTIMIZER_ATOL = 1e-4


def test_every_differentiable_stage_agrees_between_cpu_and_gpu(synthetic_scenario_batch):
    """Geometry, costs, inverse dynamics, and both optimizer parameterizations."""
    boxes_a = torch.tensor([[0.2, 0.4, 4.0, 1.8, 0.15]], dtype=torch.float32)
    boxes_b = torch.tensor([[6.0, 1.1, 3.8, 2.0, -0.2]], dtype=torch.float32)
    torch.testing.assert_close(
        signed_box_distance(boxes_a.cuda(), boxes_b.cuda()).cpu(),
        signed_box_distance(boxes_a, boxes_b),
        atol=CONSISTENCY_ATOL,
        rtol=CONSISTENCY_ATOL,
    )

    states = _states([0.0, 7.0, 14.0], dtype=torch.float32)
    valid = torch.ones((1, 3, 3), dtype=torch.bool)
    length, width = _dimensions(3, dtype=torch.float32)
    ego_mask = torch.tensor([[True, False, False]])
    candidate_mask = torch.tensor([[False, True, False]])
    background_mask = torch.tensor([[False, True, True]])
    raster = _constant_raster(0.25, dtype=torch.float32)
    cpu = combined_regents_cost(
        states, valid, length, width, ego_mask, candidate_mask, background_mask, candidate_mask, (raster,)
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
        torch.testing.assert_close(gpu_value.cpu(), cpu_value, atol=CONSISTENCY_ATOL, rtol=CONSISTENCY_ATOL)
    for name in (
        "background_collision_first_agent_idx",
        "background_collision_second_agent_idx",
        "background_collision_timestep_idx",
        "background_collision_signed_distance_meters",
        "background_collision_truncated",
    ):
        torch.testing.assert_close(getattr(gpu, name).cpu(), getattr(cpu, name))

    steering_targets = torch.tensor([0.02, 0.04, 0.06, 0.08, 0.06, 0.04], dtype=torch.float32)
    actions = torch.stack(
        (
            torch.tensor([0.1, -0.2, 0.3, -0.1, 0.0, 0.2]),
            steering_targets / float(binding.STEERING_LIMIT_RADIANS),
        ),
        dim=-1,
    ).reshape(1, 1, -1, 2)
    states = classic_rollout(
        torch.tensor([[[2.0, -3.0, 3.13, 7.0, 0.0]]], dtype=torch.float32),
        actions,
        torch.ones(actions.shape[:-1], dtype=torch.bool),
        torch.tensor([[2.7]], dtype=torch.float32),
        torch.tensor([[20.0]], dtype=torch.float32),
        0.1,
    )
    scenario_cpu = synthetic_scenario_batch(states, steering_observed=False)
    result_cpu = estimate_expert_actions(scenario_cpu)
    result_gpu = estimate_expert_actions(scenario_cpu.to("cuda"))

    for name in (
        "actions",
        "predicted_next_state",
        "state_with_estimated_steering",
        "position_error_meters",
        "heading_error_radians",
        "speed_error_mps",
        "residual_meters",
    ):
        torch.testing.assert_close(
            getattr(result_gpu, name).cpu(),
            getattr(result_cpu, name),
            atol=CONSISTENCY_ATOL,
            rtol=CONSISTENCY_ATOL,
        )
    for name in (
        "action_valid",
        "state_feature_valid",
        "low_speed_mask",
        "heading_residual_valid",
        "model_consistent",
    ):
        assert (getattr(result_gpu, name).cpu() == getattr(result_cpu, name)).all()

    for steering_parameterization in ("wheel_angle", "curvature"):
        scenario_cpu = _scenario(torch.stack((_straight_track(0.0, 0.0, 5.0, 13), _straight_track(12.0, 0.0, 3.0, 13))))
        config = ReGentSOptimizationConfig(
            **{
                **{
                    field: getattr(_optimization_config(), field)
                    for field in ("filter", "costs", "learning_rate", "iteration_count")
                },
                "steering_parameterization": steering_parameterization,
            }
        )
        result_cpu = optimize_frozen_ego_scenario(scenario_cpu, config=config, deterministic_seed=17)
        result_gpu = optimize_frozen_ego_scenario(scenario_cpu.to("cuda"), config=config, deterministic_seed=17)

        torch.testing.assert_close(
            result_gpu.optimized_actions.cpu(), result_cpu.optimized_actions, atol=OPTIMIZER_ATOL, rtol=OPTIMIZER_ATOL
        )
        torch.testing.assert_close(
            result_gpu.optimized_states.cpu(), result_cpu.optimized_states, atol=OPTIMIZER_ATOL, rtol=OPTIMIZER_ATOL
        )
        torch.testing.assert_close(result_gpu.state_valid.cpu(), result_cpu.state_valid)
        assert (result_gpu.optimized_action_mask.cpu() == result_cpu.optimized_action_mask).all()
        if result_cpu.initial_costs is not None:
            assert math.isclose(result_gpu.initial_costs.total, result_cpu.initial_costs.total, abs_tol=OPTIMIZER_ATOL)
        if result_cpu.final_costs is not None:
            assert math.isclose(result_gpu.final_costs.total, result_cpu.final_costs.total, abs_tol=OPTIMIZER_ATOL)
        assert result_gpu.success == result_cpu.success
