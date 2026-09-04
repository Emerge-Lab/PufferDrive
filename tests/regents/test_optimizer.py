import dataclasses
import math
from pathlib import Path

import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.filters import ReGentSFilterConfig
from pufferlib.ocean.regents.inverse_dynamics import estimate_expert_actions
from pufferlib.ocean.regents.losses import ReGentSCostConfig
from pufferlib.ocean.regents.optimizer import (
    STEERING_PARAMETERIZATION_CURVATURE,
    STEERING_PARAMETERIZATION_WHEEL_ANGLE,
    ReGentSOptimizationConfig,
    capture_frozen_idm_trajectory,
    drive_actions_from_parameter,
    mask_front_divergence_gradients,
    optimize_frozen_ego_scenario,
    parameter_from_drive_actions,
    steering_conversion_metadata,
)
from pufferlib.ocean.regents.state import DrivableAreaRaster, RasterTransform, ScenarioBatch


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MAP = min((REPO_ROOT / "pufferlib/resources/drive/binaries/sdc_replay_test").glob("*.bin"))
NUPLAN_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/nuplan"


def _scenario(states, drivable_mask=None, resolution_meters=1.0, origin_xy=(-50.0, -50.0)):
    states = states[None].to(torch.float32)
    _, agent_count, time_count, _ = states.shape
    valid = torch.ones((1, agent_count, time_count), dtype=torch.bool)
    present = torch.ones((1, agent_count), dtype=torch.bool)
    ego_mask = torch.zeros_like(present)
    ego_mask[0, 0] = True
    if drivable_mask is None:
        drivable_mask = torch.ones((101, 101), dtype=torch.bool)
    raster = DrivableAreaRaster(
        drivable_mask,
        RasterTransform(
            origin_xy[0],
            origin_xy[1],
            resolution_meters,
            drivable_mask.shape[0],
            drivable_mask.shape[1],
        ),
    )
    length = torch.full((1, agent_count), 4.0, dtype=torch.float32)
    width = torch.full((1, agent_count), 2.0, dtype=torch.float32)
    return ScenarioBatch(
        logged_state=states,
        state_valid=valid,
        state_feature_valid=valid[..., None].expand_as(states).clone(),
        transition_valid=valid[:, :, :-1] & valid[:, :, 1:],
        current_state=states[:, :, 0].clone(),
        current_valid=valid[:, :, 0].clone(),
        agent_present=present,
        agent_metadata_valid=present.clone(),
        active_agent_mask=present.clone(),
        agent_id=torch.arange(agent_count, dtype=torch.int64)[None],
        agent_type=torch.full((1, agent_count), binding.AGENT_TYPE_VEHICLE, dtype=torch.int64),
        controller=torch.full((1, agent_count), binding.CONTROLLER_REPLAY, dtype=torch.int64),
        trajectory_length=torch.full((1, agent_count), time_count, dtype=torch.int64),
        ego_mask=ego_mask,
        vehicle_mask=present.clone(),
        candidate_adversary_mask=present & ~ego_mask,
        logged_length_meters=length[..., None].expand(1, agent_count, time_count).clone(),
        logged_width_meters=width[..., None].expand(1, agent_count, time_count).clone(),
        length_meters=length,
        width_meters=width,
        wheelbase_meters=binding.WHEELBASE_LENGTH_RATIO * length,
        maximum_speed_mps=torch.full((1, agent_count), 20.0, dtype=torch.float32),
        scenario_ids=("synthetic",),
        dataset_names=("test",),
        log_dt_seconds=torch.tensor([0.2], dtype=torch.float32),
        dt_seconds=0.2,
        init_step=0,
        scenario_length=time_count,
        drivable_area_rasters=(raster,),
    )


def _straight_track(x_start, y_start, speed_mps, time_count, dt_seconds=0.2, heading=0.0):
    state = torch.zeros((time_count, 5), dtype=torch.float32)
    timesteps = torch.arange(time_count, dtype=torch.float32)
    state[:, 0] = x_start + timesteps * dt_seconds * speed_mps * math.cos(heading)
    state[:, 1] = y_start + timesteps * dt_seconds * speed_mps * math.sin(heading)
    state[:, 2] = heading
    state[:, 3] = speed_mps
    return state


def _optimization_config(**overrides):
    values = {
        "filter": ReGentSFilterConfig(
            static_displacement_threshold_meters=0.0,
            static_speed_threshold_mps=0.0,
        ),
        "costs": ReGentSCostConfig(),
        "learning_rate": 0.1,
        "iteration_count": 120,
    }
    values.update(overrides)
    return ReGentSOptimizationConfig(**values)


def test_optimizer_configuration_and_gradient_masking_contracts(real_scenarios):
    """Parameterization defaults, range checks, gradient masking, and conversion metadata."""
    default_config = ReGentSOptimizationConfig()
    assert default_config.steering_parameterization == STEERING_PARAMETERIZATION_WHEEL_ANGLE
    assert default_config.steering_update_scale == 4.0
    assert default_config.curvature_steering_update_scale == 0.5
    with pytest.raises(ValueError, match="steering_parameterization"):
        ReGentSOptimizationConfig(steering_parameterization="curvature_rate")
    with pytest.raises(ValueError, match="curvature_steering_update_scale"):
        ReGentSOptimizationConfig(curvature_steering_update_scale=-1.0)

    # Front divergence cancels steering only, and frozen entries lose both channels.
    masked = mask_front_divergence_gradients(
        torch.tensor([[[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]]]),
        torch.tensor([[[True, True], [True, False]]]),
        torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(masked[0, 0, :, 0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(masked[0, 0, :, 1], torch.zeros(2))
    torch.testing.assert_close(masked[0, 1, :, 0], torch.tensor([6.0, 0.0]))
    torch.testing.assert_close(masked[0, 1, :, 1], torch.tensor([7.0, 0.0]))

    # Curvature conversion round trips a real action array inside the per-agent box.
    scenario = real_scenarios(8)
    inverse = estimate_expert_actions(scenario)
    drive_actions = inverse.actions[:, :, :16].detach().clone()
    optimized_action_mask = inverse.action_valid[:, :, :16] & scenario.vehicle_mask[..., None]
    wheelbase_over_time, achievable_curvature = steering_conversion_metadata(
        scenario, optimized_action_mask, drive_actions.device
    )
    parameter = parameter_from_drive_actions(drive_actions, STEERING_PARAMETERIZATION_CURVATURE, wheelbase_over_time)
    assert torch.all(parameter[..., 1].abs() <= achievable_curvature + 1e-6)
    torch.testing.assert_close(
        drive_actions_from_parameter(parameter, STEERING_PARAMETERIZATION_CURVATURE, wheelbase_over_time),
        drive_actions,
        rtol=0.0,
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="wheelbase"):
        steering_conversion_metadata(
            dataclasses.replace(scenario, wheelbase_meters=torch.zeros_like(scenario.wheelbase_meters)),
            torch.ones((*scenario.ego_mask.shape, 4), dtype=torch.bool),
            scenario.ego_mask.device,
        )


def test_synthetic_scenes_optimize_to_collision_and_preserve_frozen_actions():
    """A braking scene and a merging scene, determinism, and post-Adam steering damping."""
    braking = _scenario(torch.stack((_straight_track(0.0, 0.0, 5.0, 13), _straight_track(12.0, 0.0, 3.0, 13))))
    first = optimize_frozen_ego_scenario(braking, config=_optimization_config(), deterministic_seed=17)
    repeated = optimize_frozen_ego_scenario(braking, config=_optimization_config(), deterministic_seed=17)
    assert first.success
    assert first.collision_timestep is not None
    assert first.selected_adversary_idx == 1
    assert first.ego_collision_loss_adversary_idx == 1
    assert first.ego_collision_loss_adversary_id == 1
    assert first.final_costs.ego_collision < first.initial_costs.ego_collision
    assert first.failure_reason is None
    assert torch.equal(first.optimized_actions, repeated.optimized_actions)
    assert first.gradient_norms == repeated.gradient_norms
    frozen = ~first.optimized_action_mask[..., None].expand_as(first.optimized_actions)
    assert torch.equal(first.optimized_actions[frozen], first.initial_actions[frozen])
    assert torch.isfinite(first.optimized_actions).all()
    assert torch.isfinite(first.optimized_states).all()

    merging_states = torch.stack((_straight_track(0.0, 0.0, 4.0, 16), _straight_track(8.0, 4.0, 3.0, 16)))
    merging = _scenario(merging_states)
    merged = optimize_frozen_ego_scenario(
        merging, config=_optimization_config(iteration_count=180), deterministic_seed=23
    )
    assert merged.success
    assert max(merged.steering_gradient_norms) > 0.0
    assert torch.any(merged.optimized_actions[..., 1] != merged.initial_actions[..., 1])
    assert not merged.background_collision
    assert not merged.offroad

    # The steering scale multiplies the post-Adam update exactly, acceleration untouched.
    full = optimize_frozen_ego_scenario(
        merging,
        config=_optimization_config(iteration_count=1, steering_update_scale=1.0),
        deterministic_seed=23,
        show_progress=False,
    )
    damped = optimize_frozen_ego_scenario(
        merging,
        config=_optimization_config(iteration_count=1, steering_update_scale=0.5),
        deterministic_seed=23,
        show_progress=False,
    )
    assert full.best_iteration == 1 and damped.best_iteration == 1
    baseline_steering = full.initial_actions[..., 1]
    full_step = full.optimized_actions[..., 1] - baseline_steering
    assert full_step.abs().sum() > 0
    torch.testing.assert_close(damped.optimized_actions[..., 1] - baseline_steering, 0.5 * full_step)
    torch.testing.assert_close(full.optimized_actions[..., 0], damped.optimized_actions[..., 0])


def test_infeasible_iterates_are_rejected_against_a_logged_baseline():
    """New background collisions and off-road excursions are rejected; logged ones are not."""
    rejection_config = ReGentSOptimizationConfig(
        filter=ReGentSFilterConfig(rear_sector_fraction=1.0),
        costs=ReGentSCostConfig(background_collision_weight=0.0, drivable_area_weight=0.0),
        learning_rate=0.1,
        iteration_count=60,
    )
    background = optimize_frozen_ego_scenario(
        _scenario(
            torch.stack(
                (
                    _straight_track(10.0, 0.0, 0.0, 16),
                    _straight_track(0.0, 0.5, 0.3, 16),
                    _straight_track(5.0, 0.0, 0.0, 16),
                )
            )
        ),
        config=rejection_config,
        deterministic_seed=29,
    )
    assert not background.success
    assert background.background_collision_rejection_count > 0
    assert not background.background_collision
    assert background.best_iteration == 0
    assert torch.equal(background.optimized_actions, background.initial_actions)

    # An overlap already present in the logged data is a baseline, not a rejection.
    preexisting = optimize_frozen_ego_scenario(
        _scenario(
            torch.stack(
                (
                    _straight_track(100.0, 0.0, 0.0, 16),
                    _straight_track(0.0, 0.0, 1.0, 16),
                    _straight_track(0.0, 0.0, 0.0, 16),
                )
            )
        ),
        config=ReGentSOptimizationConfig(
            filter=ReGentSFilterConfig(rear_sector_fraction=1.0),
            costs=ReGentSCostConfig(background_collision_weight=0.0, drivable_area_weight=0.0),
            learning_rate=0.01,
            iteration_count=1,
        ),
        deterministic_seed=30,
    )
    assert preexisting.failure_reason == "iteration_limit"
    assert preexisting.background_collision_rejection_count == 0
    assert not preexisting.background_collision
    assert preexisting.best_iteration > 0

    drivable_mask = torch.zeros((7, 31), dtype=torch.bool)
    drivable_mask[2:5] = True
    offroad = optimize_frozen_ego_scenario(
        _scenario(
            torch.stack((_straight_track(10.0, 5.0, 0.0, 16), _straight_track(0.0, 0.0, 0.3, 16))),
            drivable_mask=drivable_mask,
            resolution_meters=1.0,
            origin_xy=(-10.0, -3.0),
        ),
        config=rejection_config,
        deterministic_seed=31,
    )
    assert not offroad.success
    assert offroad.offroad_rejection_count > 0
    assert not offroad.offroad
    assert offroad.final_costs.total <= offroad.initial_costs.total


def test_real_scenario_optimization_is_deterministic_in_both_parameterizations(real_scenarios):
    """Native frozen-ego capture, cost reduction, and curvature parity on real NuPlan data."""
    kwargs = {
        "map_dir": str(FIXTURE_MAP),
        "num_maps": 1,
        "num_agents": 1,
        "min_agents_per_env": 1,
        "max_agents_per_env": 1,
        "num_eval_scenarios": 1,
        "max_scenarios_per_batch": 1,
        "starting_map": 0,
        "eval_map_indices": [0],
        "eval_scenario_seeds": [42],
        "seed": 42,
        "simulation_mode": "replay",
        "eval_mode": True,
        "control_mode": "control_sdc_only",
        "sdc_controller": "idm",
        "non_sdc_controller": "replay",
        "non_vehicle_controller": "replay",
        "action_type": "continuous",
        "dynamics_model": "classic",
        "dt": 0.1,
        "scenario_length": 8,
        "resample_frequency": 8,
        "init_step": 0,
        "init_step_spread": False,
        "reward_conditioning": False,
        "reward_randomization": False,
        "terminate_on_goal": False,
        "goal_source": "route",
        "use_neighbor_cache": 0,
    }
    first_drive = Drive(**kwargs)
    second_drive = Drive(**kwargs)
    try:
        first_scenario, first = capture_frozen_idm_trajectory(first_drive, 4, seed=42, raster_resolution_meters=2.0)
        second_scenario, second = capture_frozen_idm_trajectory(second_drive, 4, seed=42, raster_resolution_meters=2.0)
    finally:
        first_drive.close()
        second_drive.close()
    assert first.source == "c_idm"
    assert first.state.shape == (1, 5, 5)
    assert first.scenario_id == first_scenario.scenario_ids[0]
    assert first_scenario.drivable_area_rasters[0].transform.resolution_meters_per_pixel == 2.0
    assert second.scenario_id == second_scenario.scenario_ids[0]
    assert torch.equal(first.state, second.state)
    assert torch.equal(first.valid, second.valid)
    captured = optimize_frozen_ego_scenario(
        first_scenario, first, ReGentSOptimizationConfig(iteration_count=1), deterministic_seed=42
    )
    assert captured.frozen_ego_source == "c_idm"
    assert captured.optimized_states.shape[2] == first.state.shape[1]

    scenario = real_scenarios(8)
    wheel_angle_config = ReGentSOptimizationConfig(iteration_count=5, learning_rate=1e-3)
    wheel_angle = optimize_frozen_ego_scenario(
        scenario, config=wheel_angle_config, deterministic_seed=50, horizon_transition_count=16
    )
    repeated = optimize_frozen_ego_scenario(
        scenario, config=wheel_angle_config, deterministic_seed=50, horizon_transition_count=16
    )
    assert wheel_angle.initial_costs is not None
    assert wheel_angle.final_costs.ego_collision < wheel_angle.initial_costs.ego_collision
    assert wheel_angle.final_costs.total < wheel_angle.initial_costs.total
    assert all(math.isfinite(value) for value in wheel_angle.gradient_norms)
    assert torch.isfinite(wheel_angle.optimized_actions).all()
    assert torch.isfinite(wheel_angle.optimized_states).all()
    assert torch.equal(wheel_angle.optimized_actions, repeated.optimized_actions)
    assert wheel_angle.final_costs == repeated.final_costs

    curvature_config = ReGentSOptimizationConfig(
        iteration_count=5, learning_rate=1e-3, steering_parameterization=STEERING_PARAMETERIZATION_CURVATURE
    )
    curvature = optimize_frozen_ego_scenario(
        scenario, config=curvature_config, deterministic_seed=50, horizon_transition_count=16
    )
    curvature_repeated = optimize_frozen_ego_scenario(
        scenario, config=curvature_config, deterministic_seed=50, horizon_transition_count=16
    )
    assert curvature.steering_parameterization == STEERING_PARAMETERIZATION_CURVATURE
    assert torch.isfinite(curvature.optimized_actions).all()
    # The optimizer works in curvature but must still emit in-contract simulator actions.
    assert torch.all(curvature.optimized_actions.abs() <= 1.0)
    assert curvature.final_costs.total <= curvature.initial_costs.total
    assert torch.equal(curvature.optimized_actions, curvature_repeated.optimized_actions)
    assert curvature.final_costs == curvature_repeated.final_costs
    curvature_frozen = ~curvature.optimized_action_mask[..., None].expand_as(curvature.optimized_actions)
    assert torch.equal(curvature.optimized_actions[curvature_frozen], curvature.initial_actions[curvature_frozen])
    # Same scenario and seed, different steering variable, so the iterates must differ.
    assert torch.equal(wheel_angle.initial_actions, curvature.initial_actions)
    assert not torch.equal(wheel_angle.optimized_actions, curvature.optimized_actions)
