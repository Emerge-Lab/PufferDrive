from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents import classic_rollout, estimate_expert_actions, export_drive_scenarios
from pufferlib.ocean.regents.inverse_dynamics import DEFAULT_LOW_SPEED_THRESHOLD_MPS
from pufferlib.ocean.regents.state import (
    STATE_FEATURE_COUNT,
    STATE_HEADING,
    STATE_STEERING,
    DrivableAreaRaster,
    RasterTransform,
    ScenarioBatch,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MAP = min((REPO_ROOT / "pufferlib/resources/drive/binaries/sdc_replay_test").glob("*.bin"))
NUPLAN_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/nuplan"
EXACT_RECONSTRUCTION_ATOL = 1e-4
AUDIT_SCENARIO_COUNT = 16
NORMAL_SPEED_POSITION_P95_LIMIT_METERS = 0.25
LOW_SPEED_POSITION_P95_LIMIT_METERS = 0.08
NORMAL_SPEED_HEADING_P95_LIMIT_RADIANS = 0.012
SPEED_P95_LIMIT_MPS = 1e-5
TIMESTEP_MEAN_POSITION_LIMIT_METERS = 0.06
TIMESTEP_MEAN_HEADING_LIMIT_RADIANS = 0.007
TIMESTEP_MEAN_SPEED_LIMIT_MPS = 0.04


def _scenario_batch(states, state_valid=None, steering_observed=True, wheelbase_meters=2.7, maximum_speed_mps=20.0):
    if states.ndim != 4 or states.shape[-1] != STATE_FEATURE_COUNT:
        raise ValueError("test states must have shape [batch, agent, time, 5]")
    batch_count, agent_count, time_count, _ = states.shape
    if state_valid is None:
        state_valid = torch.ones((batch_count, agent_count, time_count), dtype=torch.bool)
    feature_valid = state_valid[..., None].expand_as(states).clone()
    feature_valid[..., STATE_STEERING] = state_valid & steering_observed
    transition_valid = state_valid[..., :-1] & state_valid[..., 1:]
    agent_shape = (batch_count, agent_count)
    state_shape = (batch_count, agent_count, time_count)
    present = torch.ones(agent_shape, dtype=torch.bool)
    vehicle = torch.ones(agent_shape, dtype=torch.bool)
    metadata_valid = torch.ones(agent_shape, dtype=torch.bool)
    length = torch.full(agent_shape, wheelbase_meters / float(binding.WHEELBASE_LENGTH_RATIO))
    raster = DrivableAreaRaster(
        torch.ones((2, 2), dtype=torch.bool),
        RasterTransform(0.0, 0.0, 1.0, 2, 2),
    )
    return ScenarioBatch(
        logged_state=states,
        state_valid=state_valid,
        state_feature_valid=feature_valid,
        transition_valid=transition_valid,
        current_state=states[..., 0, :].clone(),
        current_valid=present.clone(),
        agent_present=present,
        agent_metadata_valid=metadata_valid,
        active_agent_mask=present.clone(),
        agent_id=torch.arange(agent_count, dtype=torch.int64).expand(batch_count, -1).clone(),
        agent_type=torch.full(agent_shape, binding.AGENT_TYPE_VEHICLE, dtype=torch.int64),
        controller=torch.full(agent_shape, binding.CONTROLLER_REPLAY, dtype=torch.int64),
        trajectory_length=torch.full(agent_shape, time_count, dtype=torch.int64),
        ego_mask=torch.zeros(agent_shape, dtype=torch.bool),
        vehicle_mask=vehicle,
        candidate_adversary_mask=vehicle.clone(),
        logged_length_meters=length[..., None].expand(state_shape).clone(),
        logged_width_meters=torch.full(state_shape, 2.0),
        length_meters=length,
        width_meters=torch.full(agent_shape, 2.0),
        wheelbase_meters=torch.full(agent_shape, wheelbase_meters),
        maximum_speed_mps=torch.full(agent_shape, maximum_speed_mps),
        scenario_ids=tuple(f"synthetic-{batch_idx}" for batch_idx in range(batch_count)),
        dataset_names=tuple("synthetic" for _ in range(batch_count)),
        log_dt_seconds=torch.full((batch_count,), 0.1),
        dt_seconds=0.1,
        init_step=0,
        scenario_length=time_count,
        drivable_area_rasters=tuple(raster for _ in range(batch_count)),
    )


def _c_rollout(initial_state, actions, wheelbase_meters, maximum_speed_mps, dt_seconds):
    current_state = np.ascontiguousarray(initial_state.reshape(-1, STATE_FEATURE_COUNT), dtype=np.float32)
    flat_actions = np.ascontiguousarray(actions.reshape(current_state.shape[0], -1, 2), dtype=np.float32)
    wheelbase = np.full(current_state.shape[0], wheelbase_meters, dtype=np.float32)
    maximum_speed = np.full(current_state.shape[0], maximum_speed_mps, dtype=np.float32)
    states = [current_state.copy()]
    for timestep in range(flat_actions.shape[-2]):
        current_state = binding.classic_step_diagnostic(
            current_state,
            flat_actions[:, timestep],
            wheelbase,
            maximum_speed,
            dt_seconds,
        )
        states.append(current_state.copy())
    return torch.from_numpy(np.stack(states, axis=-2)).reshape(1, 1, -1, STATE_FEATURE_COUNT)


def test_inverse_exactly_reconstructs_torch_trajectory_with_heading_wrap():
    initial_state = torch.tensor([[[2.0, -3.0, 3.13, 7.0, 0.0]]], dtype=torch.float32)
    steering_targets = torch.tensor([0.02, 0.04, 0.06, 0.08, 0.06, 0.04], dtype=torch.float32)
    actions = torch.stack(
        (
            torch.tensor([0.1, -0.2, 0.3, -0.1, 0.0, 0.2]),
            steering_targets / float(binding.STEERING_LIMIT_RADIANS),
        ),
        dim=-1,
    ).reshape(1, 1, -1, 2)
    states = classic_rollout(
        initial_state,
        actions,
        torch.ones(actions.shape[:-1], dtype=torch.bool),
        torch.tensor([[2.7]], dtype=torch.float32),
        torch.tensor([[20.0]], dtype=torch.float32),
        0.1,
    )

    result = estimate_expert_actions(_scenario_batch(states, steering_observed=False))

    torch.testing.assert_close(
        result.predicted_next_state, states[..., 1:, :], rtol=0.0, atol=EXACT_RECONSTRUCTION_ATOL
    )
    torch.testing.assert_close(result.actions, actions, rtol=0.0, atol=2e-5)
    assert result.model_consistent.all()
    assert torch.any(states[..., STATE_HEADING] < -3.0)


def test_inverse_reconstructs_c_limit_and_reverse_trajectory():
    initial_state = np.asarray([[[[0.0, 0.0, -3.0, -1.7, 0.62]]]], dtype=np.float32).reshape(1, 1, 5)
    actions = np.asarray(
        [[[[-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]]],
        dtype=np.float32,
    )
    states = _c_rollout(initial_state, actions, 2.7, 1.0, 0.1)

    result = estimate_expert_actions(_scenario_batch(states, steering_observed=True, maximum_speed_mps=1.0))

    torch.testing.assert_close(
        result.predicted_next_state, states[..., 1:, :], rtol=0.0, atol=EXACT_RECONSTRUCTION_ATOL
    )
    assert result.model_consistent.all()
    assert torch.all(result.actions.abs() <= 1.0)


def test_invalid_gaps_are_not_inferred_and_reset_unobserved_steering():
    states = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0, 2.0, 0.4],
                    [0.2, 0.0, 0.0, 2.0, 0.4],
                    [8.0, 8.0, 1.0, 3.0, 0.4],
                    [1.0, 1.0, 0.0, 2.0, 0.4],
                    [1.2, 1.0, 0.0, 2.0, 0.4],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    state_valid = torch.tensor([[[True, True, False, True, True]]])

    result = estimate_expert_actions(_scenario_batch(states, state_valid, steering_observed=False))

    assert torch.equal(result.action_valid, torch.tensor([[[True, False, False, True]]]))
    assert torch.equal(result.actions[~result.action_valid], torch.zeros((2, 2)))
    assert result.state_with_estimated_steering[0, 0, 0, STATE_STEERING] == 0
    assert result.state_with_estimated_steering[0, 0, 3, STATE_STEERING] == 0


def test_near_zero_motion_keeps_steering_and_excludes_heading_residual():
    states = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.0, 0.2]]]],
        dtype=torch.float32,
    )
    result = estimate_expert_actions(_scenario_batch(states, steering_observed=True))

    assert result.low_speed_mask.item()
    assert not result.heading_residual_valid.item()
    assert result.heading_error_radians.item() == pytest.approx(1.0)
    assert result.residual_meters.item() == pytest.approx(0.0)
    assert result.actions[0, 0, 0, 1] == pytest.approx(0.2 / float(binding.STEERING_LIMIT_RADIANS))
    assert result.model_consistent.item()
    assert DEFAULT_LOW_SPEED_THRESHOLD_MPS == pytest.approx(0.6)


def test_inconsistent_transition_reports_residual_and_bounded_actions():
    states = torch.tensor(
        [[[[0.0, 0.0, 0.0, 5.0, 0.0], [0.0, 1.0, 0.8, 9.0, 0.0]]]],
        dtype=torch.float32,
    )
    result = estimate_expert_actions(_scenario_batch(states, steering_observed=False))

    assert not result.model_consistent.item()
    assert result.residual_meters.item() > 0
    assert result.position_error_meters.item() > 0
    assert torch.all(result.actions.abs() <= 1.0)
    assert torch.isfinite(result.predicted_next_state).all()


def _real_replay_drive():
    return Drive(
        map_dir=str(NUPLAN_MAP_DIR),
        num_maps=AUDIT_SCENARIO_COUNT,
        num_agents=AUDIT_SCENARIO_COUNT,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_eval_scenarios=AUDIT_SCENARIO_COUNT,
        max_scenarios_per_batch=AUDIT_SCENARIO_COUNT,
        eval_map_indices=list(range(AUDIT_SCENARIO_COUNT)),
        eval_scenario_seeds=[42 + scenario_idx for scenario_idx in range(AUDIT_SCENARIO_COUNT)],
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


def test_real_replay_reconstruction_metrics_are_finite_and_split_by_speed():
    drive = _real_replay_drive()
    try:
        drive.reset()
        scenario = export_drive_scenarios(drive, raster_resolution_meters=5.0)
    finally:
        drive.close()

    result = estimate_expert_actions(scenario)
    valid = result.action_valid
    normal_speed = valid & result.heading_residual_valid
    low_speed = valid & result.low_speed_mask

    assert valid.any()
    assert normal_speed.any()
    assert low_speed.any()
    assert torch.isfinite(result.position_error_meters[valid]).all()
    assert torch.isfinite(result.heading_error_radians[normal_speed]).all()
    assert torch.isfinite(result.speed_error_mps[valid]).all()
    assert valid.sum() == 67_111
    assert torch.quantile(result.position_error_meters[normal_speed], 0.95) <= NORMAL_SPEED_POSITION_P95_LIMIT_METERS
    assert torch.quantile(result.position_error_meters[low_speed], 0.95) <= LOW_SPEED_POSITION_P95_LIMIT_METERS
    assert torch.quantile(result.heading_error_radians[normal_speed], 0.95) <= NORMAL_SPEED_HEADING_P95_LIMIT_RADIANS
    assert torch.quantile(result.speed_error_mps[normal_speed], 0.95) <= SPEED_P95_LIMIT_MPS
    assert torch.quantile(result.speed_error_mps[low_speed], 0.95) <= SPEED_P95_LIMIT_MPS

    timestep_mean_position = []
    timestep_mean_heading = []
    timestep_mean_speed = []
    for timestep in range(valid.shape[-1]):
        timestep_valid = valid[..., timestep]
        if not timestep_valid.any():
            continue
        timestep_mean_position.append(result.position_error_meters[..., timestep][timestep_valid].mean())
        timestep_mean_speed.append(result.speed_error_mps[..., timestep][timestep_valid].mean())
        timestep_heading_valid = normal_speed[..., timestep]
        if timestep_heading_valid.any():
            timestep_mean_heading.append(result.heading_error_radians[..., timestep][timestep_heading_valid].mean())
    assert torch.stack(timestep_mean_position).max() <= TIMESTEP_MEAN_POSITION_LIMIT_METERS
    assert torch.stack(timestep_mean_heading).max() <= TIMESTEP_MEAN_HEADING_LIMIT_RADIANS
    assert torch.stack(timestep_mean_speed).max() <= TIMESTEP_MEAN_SPEED_LIMIT_MPS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_inverse_dynamics_cpu_gpu_consistency():
    initial_state = torch.tensor([[[2.0, -3.0, 3.13, 7.0, 0.0]]], dtype=torch.float32)
    steering_targets = torch.tensor([0.02, 0.04, 0.06, 0.08, 0.06, 0.04], dtype=torch.float32)
    actions = torch.stack(
        (
            torch.tensor([0.1, -0.2, 0.3, -0.1, 0.0, 0.2]),
            steering_targets / float(binding.STEERING_LIMIT_RADIANS),
        ),
        dim=-1,
    ).reshape(1, 1, -1, 2)
    states = classic_rollout(
        initial_state,
        actions,
        torch.ones(actions.shape[:-1], dtype=torch.bool),
        torch.tensor([[2.7]], dtype=torch.float32),
        torch.tensor([[20.0]], dtype=torch.float32),
        0.1,
    )
    scenario_cpu = _scenario_batch(states, steering_observed=False)
    scenario_gpu = scenario_cpu.to("cuda")

    result_cpu = estimate_expert_actions(scenario_cpu)
    result_gpu = estimate_expert_actions(scenario_gpu)

    torch.testing.assert_close(result_gpu.actions.cpu(), result_cpu.actions, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        result_gpu.predicted_next_state.cpu(), result_cpu.predicted_next_state, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        result_gpu.state_with_estimated_steering.cpu(), result_cpu.state_with_estimated_steering, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        result_gpu.position_error_meters.cpu(), result_cpu.position_error_meters, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        result_gpu.heading_error_radians.cpu(), result_cpu.heading_error_radians, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(result_gpu.speed_error_mps.cpu(), result_cpu.speed_error_mps, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(result_gpu.residual_meters.cpu(), result_cpu.residual_meters, atol=1e-5, rtol=1e-5)

    assert (result_gpu.action_valid.cpu() == result_cpu.action_valid).all()
    assert (result_gpu.state_feature_valid.cpu() == result_cpu.state_feature_valid).all()
    assert (result_gpu.low_speed_mask.cpu() == result_cpu.low_speed_mask).all()
    assert (result_gpu.heading_residual_valid.cpu() == result_cpu.heading_residual_valid).all()
    assert (result_gpu.model_consistent.cpu() == result_cpu.model_consistent).all()
