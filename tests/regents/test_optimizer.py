import math
from pathlib import Path

import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.adapter import export_drive_scenarios
from pufferlib.ocean.regents.filters import ReGentSFilterConfig
from pufferlib.ocean.regents.losses import ReGentSCostConfig
from pufferlib.ocean.regents.optimizer import (
    ReGentSOptimizationConfig,
    capture_frozen_idm_trajectory,
    mask_front_divergence_gradients,
    optimize_frozen_ego_scenario,
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


def test_front_divergence_masks_steering_but_preserves_acceleration_gradient():
    gradient = torch.tensor([[[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]]])
    optimized = torch.tensor([[[True, True], [True, False]]])
    divergent = torch.tensor([[True, False]])
    masked = mask_front_divergence_gradients(gradient, optimized, divergent)
    torch.testing.assert_close(masked[0, 0, :, 0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(masked[0, 0, :, 1], torch.zeros(2))
    torch.testing.assert_close(masked[0, 1, :, 0], torch.tensor([6.0, 0.0]))
    torch.testing.assert_close(masked[0, 1, :, 1], torch.tensor([7.0, 0.0]))


def test_braking_scene_optimizes_to_collision_and_preserves_frozen_actions():
    time_count = 13
    scenario = _scenario(
        torch.stack(
            (
                _straight_track(0.0, 0.0, 5.0, time_count),
                _straight_track(12.0, 0.0, 3.0, time_count),
            )
        )
    )
    first = optimize_frozen_ego_scenario(scenario, config=_optimization_config(), deterministic_seed=17)
    repeated = optimize_frozen_ego_scenario(scenario, config=_optimization_config(), deterministic_seed=17)

    assert first.success
    assert first.collision_timestep is not None
    assert first.selected_adversary_idx == 1
    assert first.final_costs.ego_collision < first.initial_costs.ego_collision
    assert first.failure_reason is None
    assert torch.equal(first.optimized_actions, repeated.optimized_actions)
    assert first.gradient_norms == repeated.gradient_norms
    frozen = ~first.optimized_action_mask[..., None].expand_as(first.optimized_actions)
    assert torch.equal(first.optimized_actions[frozen], first.initial_actions[frozen])
    assert torch.isfinite(first.optimized_actions).all()
    assert torch.isfinite(first.optimized_states).all()


def test_merging_scene_uses_steering_and_reaches_ego_without_constraint_violation():
    time_count = 16
    scenario = _scenario(
        torch.stack(
            (
                _straight_track(0.0, 0.0, 4.0, time_count),
                _straight_track(8.0, 4.0, 3.0, time_count),
            )
        )
    )
    result = optimize_frozen_ego_scenario(
        scenario,
        config=_optimization_config(iteration_count=180),
        deterministic_seed=23,
    )

    assert result.success
    assert max(result.steering_gradient_norms) > 0.0
    assert torch.any(result.optimized_actions[..., 1] != result.initial_actions[..., 1])
    assert not result.background_collision
    assert not result.offroad


def test_background_collision_iterates_are_rejected_and_best_feasible_actions_retained():
    time_count = 16
    scenario = _scenario(
        torch.stack(
            (
                _straight_track(10.0, 0.0, 0.0, time_count),
                _straight_track(0.0, 0.5, 0.3, time_count),
                _straight_track(5.0, 0.0, 0.0, time_count),
            )
        )
    )
    config = ReGentSOptimizationConfig(
        filter=ReGentSFilterConfig(rear_sector_fraction=1.0),
        costs=ReGentSCostConfig(background_collision_weight=0.0, drivable_area_weight=0.0),
        learning_rate=0.1,
        iteration_count=60,
    )
    result = optimize_frozen_ego_scenario(scenario, config=config, deterministic_seed=29)

    assert not result.success
    assert result.background_collision_rejection_count > 0
    assert not result.background_collision
    assert result.best_iteration == 0
    assert torch.equal(result.optimized_actions, result.initial_actions)


def test_new_offroad_iterates_are_rejected_and_logged_raster_mismatch_is_tolerated():
    time_count = 16
    drivable_mask = torch.zeros((7, 31), dtype=torch.bool)
    drivable_mask[2:5] = True
    scenario = _scenario(
        torch.stack(
            (
                _straight_track(10.0, 5.0, 0.0, time_count),
                _straight_track(0.0, 0.0, 0.3, time_count),
            )
        ),
        drivable_mask=drivable_mask,
        resolution_meters=1.0,
        origin_xy=(-10.0, -3.0),
    )
    config = ReGentSOptimizationConfig(
        filter=ReGentSFilterConfig(rear_sector_fraction=1.0),
        costs=ReGentSCostConfig(background_collision_weight=0.0, drivable_area_weight=0.0),
        learning_rate=0.1,
        iteration_count=60,
    )
    result = optimize_frozen_ego_scenario(scenario, config=config, deterministic_seed=31)

    assert not result.success
    assert result.offroad_rejection_count > 0
    assert not result.offroad
    assert result.final_costs.total <= result.initial_costs.total


def test_capture_frozen_idm_trajectory_uses_native_controller_deterministically():
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
        first_scenario, first = capture_frozen_idm_trajectory(first_drive, 4, seed=42)
        second_scenario, second = capture_frozen_idm_trajectory(second_drive, 4, seed=42)
    finally:
        first_drive.close()
        second_drive.close()

    assert first.source == "c_idm"
    assert first.state.shape == (1, 5, 5)
    assert first.scenario_id == first_scenario.scenario_ids[0]
    assert second.scenario_id == second_scenario.scenario_ids[0]
    assert torch.equal(first.state, second.state)
    assert torch.equal(first.valid, second.valid)
    result = optimize_frozen_ego_scenario(
        first_scenario,
        first,
        ReGentSOptimizationConfig(iteration_count=1),
        deterministic_seed=42,
    )
    assert result.frozen_ego_source == "c_idm"
    assert result.optimized_states.shape[2] == first.state.shape[1]


def _fixed_real_scenario(map_idx):
    drive = Drive(
        map_dir=str(NUPLAN_MAP_DIR),
        num_maps=map_idx + 1,
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_eval_scenarios=1,
        max_scenarios_per_batch=1,
        eval_map_indices=[map_idx],
        eval_scenario_seeds=[42 + map_idx],
        seed=42,
        simulation_mode="replay",
        eval_mode=True,
        control_mode="control_sdc_only",
        sdc_controller="idm",
        non_sdc_controller="replay",
        non_vehicle_controller="replay",
        action_type="continuous",
        dynamics_model="classic",
        dt=0.1,
        scenario_length=200,
        resample_frequency=200,
        init_step=0,
        init_step_spread=False,
        reward_conditioning=False,
        reward_randomization=False,
        use_neighbor_cache=0,
    )
    try:
        drive.reset()
        return export_drive_scenarios(drive, raster_resolution_meters=5.0)
    finally:
        drive.close()


@pytest.fixture(scope="module")
def cached_real_scenarios():
    return {
        8: _fixed_real_scenario(8),
    }


@pytest.mark.parametrize(("map_idx", "seed"), [(8, 50)])
def test_fixed_real_scenario_optimization_is_finite_deterministic_and_reduces_collision_cost(
    map_idx, seed, cached_real_scenarios
):
    scenario = cached_real_scenarios[map_idx]
    config = ReGentSOptimizationConfig(iteration_count=5, learning_rate=1e-3)
    first = optimize_frozen_ego_scenario(
        scenario,
        config=config,
        deterministic_seed=seed,
        horizon_transition_count=16,
    )
    repeated = optimize_frozen_ego_scenario(
        scenario,
        config=config,
        deterministic_seed=seed,
        horizon_transition_count=16,
    )

    assert first.initial_costs is not None
    assert first.final_costs.ego_collision < first.initial_costs.ego_collision
    assert first.final_costs.total < first.initial_costs.total
    assert all(math.isfinite(value) for value in first.gradient_norms)
    assert torch.isfinite(first.optimized_actions).all()
    assert torch.isfinite(first.optimized_states).all()
    assert torch.equal(first.optimized_actions, repeated.optimized_actions)
    assert first.final_costs == repeated.final_costs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_optimizer_cpu_gpu_consistency():
    time_count = 13
    scenario_cpu = _scenario(
        torch.stack(
            (
                _straight_track(0.0, 0.0, 5.0, time_count),
                _straight_track(12.0, 0.0, 3.0, time_count),
            )
        )
    )
    config = _optimization_config()
    result_cpu = optimize_frozen_ego_scenario(scenario_cpu, config=config, deterministic_seed=17)

    scenario_gpu = scenario_cpu.to("cuda")
    result_gpu = optimize_frozen_ego_scenario(scenario_gpu, config=config, deterministic_seed=17)

    torch.testing.assert_close(result_gpu.optimized_actions.cpu(), result_cpu.optimized_actions, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result_gpu.optimized_states.cpu(), result_cpu.optimized_states, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(result_gpu.state_valid.cpu(), result_cpu.state_valid)
    assert (result_gpu.optimized_action_mask.cpu() == result_cpu.optimized_action_mask).all()

    if result_cpu.initial_costs is not None:
        assert math.isclose(result_gpu.initial_costs.total, result_cpu.initial_costs.total, abs_tol=1e-4)
    if result_cpu.final_costs is not None:
        assert math.isclose(result_gpu.final_costs.total, result_cpu.final_costs.total, abs_tol=1e-4)

    assert result_gpu.success == result_cpu.success
