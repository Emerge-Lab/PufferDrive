import dataclasses
import math
from pathlib import Path

import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents import classic_step
from pufferlib.ocean.regents.filters import ReGentSFilterConfig, select_adversary_candidates
from pufferlib.ocean.regents.inverse_dynamics import estimate_expert_actions
from pufferlib.ocean.regents.geometry import signed_box_distance
from pufferlib.ocean.regents.losses import ReGentSCostConfig, _masked_boxes
from pufferlib.ocean.regents.optimizer import (
    FrozenEgoTrajectory,
    ReGentSOptimizationConfig,
    capture_frozen_idm_trajectory,
    drive_actions_from_parameter,
    optimize_frozen_ego_scenario,
    parameter_from_drive_actions,
    steering_conversion_metadata,
)
from pufferlib.ocean.regents.optimizer import (
    _background_collision_signature,
    _candidate_background_pair_indices,
    _compose_rollout,
)
from pufferlib.ocean.regents.state import (
    DrivableAreaRaster,
    RasterTransform,
    ScenarioBatch,
)


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
    assert default_config.steering_update_scale == 0.5
    with pytest.raises(ValueError, match="steering_update_scale"):
        ReGentSOptimizationConfig(steering_update_scale=-1.0)

    # Curvature conversion round trips a real action array inside the per-agent box.
    scenario = real_scenarios(8)
    inverse = estimate_expert_actions(scenario)
    drive_actions = inverse.actions[:, :, :16].detach().clone()
    optimized_action_mask = inverse.action_valid[:, :, :16] & scenario.vehicle_mask[..., None]
    wheelbase_over_time, achievable_curvature = steering_conversion_metadata(
        scenario, optimized_action_mask, drive_actions.device
    )
    parameter = parameter_from_drive_actions(drive_actions, wheelbase_over_time)
    assert torch.all(parameter[..., 1].abs() <= achievable_curvature + 1e-6)
    torch.testing.assert_close(
        drive_actions_from_parameter(parameter, wheelbase_over_time),
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


def test_synthetic_scenes_optimize_to_collision_and_preserve_frozen_actions(monkeypatch):
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
    frozen = ~first.optimized_action_mask[..., None].expand_as(first.optimized_actions)
    assert torch.equal(first.optimized_actions[frozen], first.initial_actions[frozen])
    assert torch.isfinite(first.optimized_actions).all()
    assert torch.isfinite(first.optimized_states).all()

    # The gate reads drift the optimizer measures itself, so an exclusive 0 m threshold
    # rejects the one agent whose reconstruction leaves its log at all.
    drift_filtered = optimize_frozen_ego_scenario(
        braking,
        config=_optimization_config(
            filter=ReGentSFilterConfig(
                static_displacement_threshold_meters=0.0,
                maximum_reconstruction_drift_meters=0.0,
            )
        ),
        deterministic_seed=17,
        show_progress=False,
    )
    assert not drift_filtered.selection.candidate_mask.any()
    assert "reconstruction_fidelity" in drift_filtered.selection.reasons_for(0, 1)
    assert drift_filtered.failure_reason == "scene_filtered:no_candidate"
    assert drift_filtered.selection.reconstruction_drift_meters[0, 1] > 0.0
    # The default threshold is infinite, so the same scene keeps its candidate.
    assert "reconstruction_fidelity" not in first.selection.reasons_for(0, 1)
    assert first.selection.candidate_mask[0, 1]

    merging_states = torch.stack((_straight_track(0.0, 0.0, 4.0, 16), _straight_track(8.0, 4.0, 3.0, 16)))
    merging = _scenario(merging_states)
    merged = optimize_frozen_ego_scenario(
        merging, config=_optimization_config(iteration_count=180), deterministic_seed=23
    )
    assert merged.success
    assert torch.any(merged.optimized_actions[..., 1] != merged.initial_actions[..., 1])
    assert not merged.background_collision
    assert not merged.offroad

    # The steering scale multiplies the post-Adam update exactly, acceleration untouched.
    full = optimize_frozen_ego_scenario(
        merging,
        config=_optimization_config(
            iteration_count=1,
            steering_update_scale=1.0,
        ),
        deterministic_seed=23,
        show_progress=False,
    )
    damped = optimize_frozen_ego_scenario(
        merging,
        config=_optimization_config(
            iteration_count=1,
            steering_update_scale=0.5,
        ),
        deterministic_seed=23,
        show_progress=False,
    )
    assert full.best_iteration == 1 and damped.best_iteration == 1
    # The scale multiplies the post-Adam update in curvature space; the wheel-angle
    # conversion is non-linear, so the exact ratio only holds on the parameter itself.
    wheelbase_over_time, _ = steering_conversion_metadata(
        merging, full.optimized_action_mask, full.optimized_actions.device
    )
    baseline_curvature = parameter_from_drive_actions(full.initial_actions, wheelbase_over_time)[..., 1]
    full_step = parameter_from_drive_actions(full.optimized_actions, wheelbase_over_time)[..., 1] - baseline_curvature
    damped_step = (
        parameter_from_drive_actions(damped.optimized_actions, wheelbase_over_time)[..., 1] - baseline_curvature
    )
    assert full_step.abs().sum() > 0
    torch.testing.assert_close(damped_step, 0.5 * full_step)
    torch.testing.assert_close(full.optimized_actions[..., 0], damped.optimized_actions[..., 0])

    # Divergence holds the applied steering, but Adam must accumulate its moments.
    original_step = torch.optim.Adam.step
    steering_moments = []

    def capture_adam_step(optimizer, *args, **kwargs):
        parameter = optimizer.param_groups[0]["params"][0]
        previous_moment = optimizer.state.get(parameter, {}).get("exp_avg", torch.zeros_like(parameter)).clone()
        result = original_step(optimizer, *args, **kwargs)
        current_moment = optimizer.state[parameter]["exp_avg"].clone()
        beta1 = optimizer.param_groups[0]["betas"][0]
        torch.testing.assert_close(current_moment, beta1 * previous_moment + (1 - beta1) * parameter.grad)
        steering_moments.append((previous_moment[..., 1], current_moment[..., 1]))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.Adam, "step", capture_adam_step)
        patch.setattr(
            "pufferlib.ocean.regents.optimizer.front_divergence_mask",
            lambda states, valid, ego, candidates, **kwargs: candidates,
        )
        divergent = optimize_frozen_ego_scenario(
            merging,
            config=_optimization_config(iteration_count=2, early_stop_on_collision=False),
            deterministic_seed=23,
            show_progress=False,
        )
    assert len(steering_moments) == 2
    assert steering_moments[0][1].abs().sum() > 0
    torch.testing.assert_close(steering_moments[1][0], steering_moments[0][1])
    torch.testing.assert_close(divergent.optimized_actions[..., 1], divergent.initial_actions[..., 1])


def test_current_iterate_is_returned_with_baseline_relative_infraction_diagnostics():
    """The current iterate and baseline-relative infraction diagnostics are returned."""
    mixed_control = optimize_frozen_ego_scenario(
        _scenario(
            torch.stack(
                (
                    _straight_track(100.0, 0.0, 0.0, 8),
                    _straight_track(0.0, 0.0, 2.0, 8),
                    _straight_track(30.0, 0.0, 0.0, 8),
                    _straight_track(6.0, 0.0, 1.0, 8),
                )
            )
        ),
        config=ReGentSOptimizationConfig(
            filter=ReGentSFilterConfig(rear_sector_fraction=1.0),
            costs=ReGentSCostConfig(ego_collision_weight=0.0, drivable_area_weight=0.0),
            learning_rate=0.1,
            iteration_count=20,
            early_stop_on_collision=False,
        ),
        deterministic_seed=28,
        show_progress=False,
    )
    assert mixed_control.selection.candidate_mask.tolist() == [[False, True, False, True]]
    assert mixed_control.initial_costs.background_collision_first_agent_idx == 1
    assert mixed_control.initial_costs.background_collision_second_agent_idx == 3
    assert mixed_control.initial_costs.background_collision_timestep_idx == 7
    assert math.isclose(mixed_control.initial_costs.background_collision, -0.6, abs_tol=2e-6)
    assert mixed_control.final_costs.background_collision == -1.25
    assert mixed_control.final_costs.background_collision_truncated
    assert not torch.equal(mixed_control.optimized_actions[0, 1], mixed_control.initial_actions[0, 1])
    assert torch.equal(mixed_control.optimized_actions[0, 2], mixed_control.initial_actions[0, 2])
    assert not torch.equal(mixed_control.optimized_actions[0, 3], mixed_control.initial_actions[0, 3])
    assert not mixed_control.background_collision
    assert mixed_control.iteration_count == 20
    assert mixed_control.best_iteration == 20
    assert mixed_control.final_costs == mixed_control.cost_history[-1]

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
    assert background.success
    assert background.background_collision_rejection_count > 0
    assert background.background_collision
    assert 0 < background.iteration_count < rejection_config.iteration_count
    assert background.best_iteration == background.iteration_count
    assert background.final_costs == background.cost_history[-1]
    assert not torch.equal(background.optimized_actions, background.initial_actions)

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
    assert offroad.offroad_rejection_count > 0
    assert offroad.offroad
    assert offroad.best_iteration == offroad.iteration_count
    assert offroad.final_costs == offroad.cost_history[-1]
    assert offroad.final_costs.total <= offroad.initial_costs.total


def test_real_scenario_optimization_is_deterministic_in_curvature_space(real_scenarios):
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
    assert first.scenario_ids == first_scenario.scenario_ids
    assert first_scenario.drivable_area_rasters[0].transform.resolution_meters_per_pixel == 2.0
    assert second.scenario_ids == second_scenario.scenario_ids
    assert torch.equal(first.state, second.state)
    assert torch.equal(first.valid, second.valid)
    captured = optimize_frozen_ego_scenario(
        first_scenario, first, ReGentSOptimizationConfig(iteration_count=1), deterministic_seed=42
    )
    assert captured.frozen_ego_source == "c_idm"
    assert captured.optimized_states.shape[2] == first.state.shape[1]

    # Map 8's full-log candidates enter after this horizon; map 3 exercises live updates.
    scenario = real_scenarios(3)
    curvature_config = ReGentSOptimizationConfig(iteration_count=5, learning_rate=1e-3)
    curvature = optimize_frozen_ego_scenario(
        scenario, config=curvature_config, deterministic_seed=50, horizon_transition_count=16
    )
    repeated = optimize_frozen_ego_scenario(
        scenario, config=curvature_config, deterministic_seed=50, horizon_transition_count=16
    )
    assert curvature.initial_costs is not None
    # Road regularization can trade a small ego-distance increase for lower total cost.
    assert curvature.iteration_count == curvature_config.iteration_count
    assert curvature.final_costs.total < curvature.initial_costs.total
    assert torch.isfinite(curvature.optimized_actions).all()
    assert torch.isfinite(curvature.optimized_states).all()
    # The optimizer works in curvature but must still emit in-contract simulator actions.
    assert torch.all(curvature.optimized_actions.abs() <= 1.0)
    assert torch.equal(curvature.optimized_actions, repeated.optimized_actions)
    assert curvature.final_costs == repeated.final_costs
    curvature_frozen = ~curvature.optimized_action_mask[..., None].expand_as(curvature.optimized_actions)
    assert torch.equal(curvature.optimized_actions[curvature_frozen], curvature.initial_actions[curvature_frozen])


def test_background_collision_gate_matches_an_all_exact_scan(real_scenarios):
    """The prefiltered gate is an optimization, so it must agree pair for pair."""
    scenario = real_scenarios(1)
    horizon_transition_count = 40
    state_valid = scenario.state_valid[:, :, : horizon_transition_count + 1]
    boxes = _masked_boxes(
        scenario.logged_state[:, :, : horizon_transition_count + 1],
        state_valid,
        scenario.length_meters,
        scenario.width_meters,
    )
    selection = select_adversary_candidates(scenario, ReGentSFilterConfig())
    pair_indices = _candidate_background_pair_indices(scenario, state_valid, selection.candidate_mask)
    assert pair_indices.shape[1] > 0

    left_indices, right_indices = pair_indices
    jointly_valid = state_valid[0, left_indices] & state_valid[0, right_indices]
    exact_distances = signed_box_distance(boxes[0, left_indices], boxes[0, right_indices])
    # A real log rarely touches, so widened tolerances are what actually exercise a hit.
    for tolerance_meters in (0.0, 2.0, 5.0):
        expected = torch.any(jointly_valid & (exact_distances <= tolerance_meters), dim=-1)
        actual = _background_collision_signature(boxes, state_valid, tolerance_meters, pair_indices)
        assert torch.equal(actual, expected)
    assert bool(torch.any(_background_collision_signature(boxes, state_valid, 5.0, pair_indices))), (
        "widened tolerance must produce hits or the comparison is vacuous"
    )


def test_compacted_rollout_matches_a_full_width_reference(real_scenarios):
    """Compaction preserves frozen rows and refreshes wheelbase after validity gaps."""
    scenario = real_scenarios(1)
    horizon_transition_count = 30
    frozen_ego = FrozenEgoTrajectory(
        state=scenario.logged_state[:, 0, : horizon_transition_count + 1].clone(),
        valid=scenario.state_valid[:, 0, : horizon_transition_count + 1].clone(),
        scenario_ids=scenario.scenario_ids,
        source="logged_fixture",
    )
    inverse = estimate_expert_actions(scenario, horizon_transition_count=horizon_transition_count)
    selection = select_adversary_candidates(scenario, ReGentSFilterConfig())
    generator = torch.Generator().manual_seed(1234)
    actions = torch.rand(
        (scenario.batch_size, scenario.max_agent_count, horizon_transition_count, 2), generator=generator
    )
    actions = (actions * 2 - 1).requires_grad_(True)

    states, state_valid = _compose_rollout(
        scenario, inverse, actions, frozen_ego, horizon_transition_count, selection.candidate_mask
    )
    states.square().sum().backward()

    # Non-candidate agents are never integrated, so they must equal the reference exactly.
    reference = inverse.state_with_estimated_steering[:, :, : horizon_transition_count + 1].clone()
    reference[:, 0] = frozen_ego.state
    untouched = ~selection.candidate_mask
    assert torch.equal(states.detach()[untouched], reference[untouched])
    # Gradients reach candidate actions and nothing else.
    action_gradient_rows = (actions.grad != 0).any(dim=-1).any(dim=-1)
    assert not bool((action_gradient_rows & untouched).any())
    assert bool((action_gradient_rows & selection.candidate_mask).any())
    assert torch.isfinite(actions.grad).all()
    assert torch.isfinite(states).all()
    assert state_valid.shape == states.shape[:-1]

    gap_scenario = _scenario(torch.stack((_straight_track(0.0, 0.0, 2.0, 5), _straight_track(5.0, 2.0, 3.0, 5))))
    gap_valid = gap_scenario.state_valid.clone()
    gap_valid[0, 1, 2] = False
    logged_length_meters = gap_scenario.logged_length_meters.clone()
    logged_length_meters[0, 1, 3] = 6.0
    gap_scenario = dataclasses.replace(
        gap_scenario,
        state_valid=gap_valid,
        state_feature_valid=gap_valid[..., None].expand_as(gap_scenario.logged_state).clone(),
        transition_valid=gap_valid[..., :-1] & gap_valid[..., 1:],
        logged_length_meters=logged_length_meters,
    )
    gap_inverse = estimate_expert_actions(gap_scenario)
    gap_candidate_mask = torch.tensor([[False, True]])
    gap_action_mask = gap_inverse.action_valid & gap_candidate_mask[..., None]
    gap_actions = torch.zeros((1, 2, 4, 2), dtype=torch.float32)
    gap_actions[0, 1, :, 1] = 0.5
    gap_frozen_ego = FrozenEgoTrajectory(
        state=gap_scenario.logged_state[:, 0].clone(),
        valid=gap_scenario.state_valid[:, 0].clone(),
        scenario_ids=gap_scenario.scenario_ids,
        source="logged_fixture",
    )
    gap_states, _ = _compose_rollout(
        gap_scenario,
        gap_inverse,
        gap_actions,
        gap_frozen_ego,
        4,
        gap_candidate_mask,
    )
    reference = gap_inverse.state_with_estimated_steering
    expected_initial_run = classic_step(
        reference[0, 1, 0:1],
        gap_actions[0, 1, 0:1],
        torch.tensor([2.4]),
        torch.tensor([20.0]),
        gap_scenario.dt_seconds,
    )[0]
    expected_resumed_run = classic_step(
        reference[0, 1, 3:4],
        gap_actions[0, 1, 3:4],
        torch.tensor([3.6]),
        torch.tensor([20.0]),
        gap_scenario.dt_seconds,
    )[0]
    torch.testing.assert_close(gap_states[0, 1, 1], expected_initial_run)
    torch.testing.assert_close(gap_states[0, 1, 3], reference[0, 1, 3])
    torch.testing.assert_close(gap_states[0, 1, 4], expected_resumed_run)

    gap_wheelbase, _ = steering_conversion_metadata(gap_scenario, gap_action_mask, gap_actions.device)
    torch.testing.assert_close(gap_wheelbase[0, 1], torch.tensor([2.4, 2.4, 2.4, 3.6]))


def _frozen_ego_from_log(scenario, horizon_transition_count):
    return FrozenEgoTrajectory(
        state=scenario.logged_state[:, 0, : horizon_transition_count + 1].clone(),
        valid=scenario.state_valid[:, 0, : horizon_transition_count + 1].clone(),
        scenario_ids=scenario.scenario_ids,
        source="logged_fixture",
    )


def test_ego_refresh_interleaves_reactive_rollouts_with_gradient_steps():
    """A periodic ego re-roll runs on schedule, keeps Adam state, and gates collisions."""
    braking = _scenario(torch.stack((_straight_track(0.0, 0.0, 5.0, 13), _straight_track(12.0, 0.0, 3.0, 13))))
    horizon = 12
    frozen_ego = _frozen_ego_from_log(braking, horizon)

    with pytest.raises(ValueError, match="ego_refresh_interval requires an ego_rollout_fn"):
        optimize_frozen_ego_scenario(
            braking, frozen_ego, _optimization_config(ego_refresh_interval=5), deterministic_seed=17
        )
    with pytest.raises(ValueError, match="ego_refresh_interval must be non-negative"):
        ReGentSOptimizationConfig(ego_refresh_interval=-1)

    # An identity refresh must leave the optimization bit-identical to a frozen run,
    # which is what proves Adam's moments survive the swap.
    frozen_config = _optimization_config(iteration_count=20, early_stop_on_collision=False)
    refreshed_config = dataclasses.replace(frozen_config, ego_refresh_interval=5)
    frozen_result = optimize_frozen_ego_scenario(braking, frozen_ego, frozen_config, deterministic_seed=17)

    refresh_calls = []

    def identity_rollout(drive_actions, action_mask):
        refresh_calls.append((drive_actions.detach().clone(), action_mask.clone()))
        return _frozen_ego_from_log(braking, horizon)

    refreshed_result = optimize_frozen_ego_scenario(
        braking, frozen_ego, refreshed_config, deterministic_seed=17, ego_rollout_fn=identity_rollout
    )
    assert frozen_result.ego_refresh_count == 0
    assert refreshed_result.ego_refresh_count == 4
    assert len(refresh_calls) == 4
    assert refresh_calls[0][1].shape == (1, 2, horizon)
    assert torch.equal(refreshed_result.optimized_actions, frozen_result.optimized_actions)
    assert refreshed_result.iteration_count == frozen_result.iteration_count

    # An ego that leaves the scene cannot be hit, so no iterate may be reported as a
    # success even though the frozen-ego run collides.
    def dodging_rollout(drive_actions, action_mask):
        dodged = braking.logged_state[:, 0, : horizon + 1].clone()
        dodged[..., 1] += 500.0
        return FrozenEgoTrajectory(
            state=dodged,
            valid=braking.state_valid[:, 0, : horizon + 1].clone(),
            scenario_ids=braking.scenario_ids,
            source="logged_fixture",
        )

    collided = optimize_frozen_ego_scenario(braking, frozen_ego, _optimization_config(), deterministic_seed=17)
    dodged_result = optimize_frozen_ego_scenario(
        braking,
        frozen_ego,
        _optimization_config(ego_refresh_interval=5),
        deterministic_seed=17,
        ego_rollout_fn=dodging_rollout,
    )
    assert collided.success
    assert not dodged_result.success
    assert dodged_result.failure_reason == "iteration_limit"

    def wrong_shape_rollout(drive_actions, action_mask):
        return _frozen_ego_from_log(braking, horizon - 1)

    with pytest.raises(ValueError, match="wrong shape"):
        optimize_frozen_ego_scenario(
            braking,
            frozen_ego,
            _optimization_config(ego_refresh_interval=1),
            deterministic_seed=17,
            ego_rollout_fn=wrong_shape_rollout,
        )
