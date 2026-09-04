from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents import classic_rollout, estimate_expert_actions, export_drive_scenarios
from pufferlib.ocean.regents.inverse_dynamics import DEFAULT_LOW_SPEED_THRESHOLD_MPS
from pufferlib.ocean.regents.state import STATE_FEATURE_COUNT, STATE_HEADING, STATE_STEERING


REPO_ROOT = Path(__file__).resolve().parents[2]
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


def test_inverse_exactly_reconstructs_torch_c_limit_and_reverse_trajectories(synthetic_scenario_batch):
    """Exact reconstruction through a heading wrap, at the C limits in reverse, and on a prefix."""
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
    scenario = synthetic_scenario_batch(states, steering_observed=False)
    result = estimate_expert_actions(scenario)
    torch.testing.assert_close(
        result.predicted_next_state, states[..., 1:, :], rtol=0.0, atol=EXACT_RECONSTRUCTION_ATOL
    )
    torch.testing.assert_close(result.actions, actions, rtol=0.0, atol=2e-5)
    assert result.model_consistent.all()
    assert torch.any(states[..., STATE_HEADING] < -3.0)

    # A bounded horizon must be the exact prefix of the full reconstruction.
    prefix = estimate_expert_actions(scenario, horizon_transition_count=2)
    assert prefix.actions.shape == (1, 1, 2, 2)
    assert prefix.state_with_estimated_steering.shape == (1, 1, 3, STATE_FEATURE_COUNT)
    torch.testing.assert_close(prefix.actions, result.actions[:, :, :2])
    torch.testing.assert_close(prefix.state_with_estimated_steering, result.state_with_estimated_steering[:, :, :3])
    torch.testing.assert_close(prefix.residual_meters, result.residual_meters[:, :, :2])

    # Saturated actions and reverse motion are only expressible against the C model.
    c_states = _c_rollout(
        np.asarray([[[[0.0, 0.0, -3.0, -1.7, 0.62]]]], dtype=np.float32).reshape(1, 1, 5),
        np.asarray([[[[-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]]], dtype=np.float32),
        2.7,
        1.0,
        0.1,
    )
    c_result = estimate_expert_actions(
        synthetic_scenario_batch(c_states, steering_observed=True, maximum_speed_mps=1.0)
    )
    torch.testing.assert_close(
        c_result.predicted_next_state, c_states[..., 1:, :], rtol=0.0, atol=EXACT_RECONSTRUCTION_ATOL
    )
    assert c_result.model_consistent.all()
    assert torch.all(c_result.actions.abs() <= 1.0)


def test_inverse_handles_gaps_low_speed_and_inconsistency(synthetic_scenario_batch):
    """Validity gaps are never bridged, low speed keeps steering, and residuals are reported."""
    gapped_states = torch.tensor(
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
    gapped = estimate_expert_actions(
        synthetic_scenario_batch(
            gapped_states, torch.tensor([[[True, True, False, True, True]]]), steering_observed=False
        )
    )
    assert torch.equal(gapped.action_valid, torch.tensor([[[True, False, False, True]]]))
    assert torch.equal(gapped.actions[~gapped.action_valid], torch.zeros((2, 2)))
    assert gapped.state_with_estimated_steering[0, 0, 0, STATE_STEERING] == 0
    assert gapped.state_with_estimated_steering[0, 0, 3, STATE_STEERING] == 0

    stationary = estimate_expert_actions(
        synthetic_scenario_batch(
            torch.tensor([[[[0.0, 0.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.0, 0.2]]]], dtype=torch.float32),
            steering_observed=True,
        )
    )
    assert stationary.low_speed_mask.item()
    assert not stationary.heading_residual_valid.item()
    assert stationary.heading_error_radians.item() == pytest.approx(1.0)
    assert stationary.residual_meters.item() == pytest.approx(0.0)
    assert stationary.actions[0, 0, 0, 1] == pytest.approx(0.2 / float(binding.STEERING_LIMIT_RADIANS))
    assert stationary.model_consistent.item()
    assert DEFAULT_LOW_SPEED_THRESHOLD_MPS == pytest.approx(0.6)

    unreachable = estimate_expert_actions(
        synthetic_scenario_batch(
            torch.tensor([[[[0.0, 0.0, 0.0, 5.0, 0.0], [0.0, 1.0, 0.8, 9.0, 0.0]]]], dtype=torch.float32),
            steering_observed=False,
        )
    )
    assert not unreachable.model_consistent.item()
    assert unreachable.residual_meters.item() > 0
    assert unreachable.position_error_meters.item() > 0
    assert torch.all(unreachable.actions.abs() <= 1.0)
    assert torch.isfinite(unreachable.predicted_next_state).all()


def test_real_replay_reconstruction_metrics_are_finite_and_meet_the_p95_gates():
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

    assert valid.any() and normal_speed.any() and low_speed.any()
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
