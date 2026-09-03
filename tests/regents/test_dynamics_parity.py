from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents import classic_rollout, classic_step, export_drive_scenarios
from pufferlib.ocean.regents.dynamics import (
    MAX_BACKWARD_SPEED_MPS,
    STEERING_LIMIT_RADIANS,
    STEERING_RATE_LIMIT_RADIANS_PER_SECOND,
)
from pufferlib.ocean.regents.state import STATE_HEADING, STATE_SPEED, STATE_STEERING


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MAP = min((REPO_ROOT / "pufferlib/resources/drive/binaries/sdc_replay_test").glob("*.bin"))
PARITY_ATOL = 1e-4
PARITY_ROLLOUT_TRANSITION_COUNT = 64
STRICT_ONE_STEP_ATOL = 1e-6


def _c_classic_step(state, action, wheelbase_meters, maximum_speed_mps, dt_seconds):
    original_shape = state.shape
    flat_state = np.ascontiguousarray(state.reshape(-1, 5), dtype=np.float32)
    flat_action = np.ascontiguousarray(action.reshape(-1, 2), dtype=np.float32)
    flat_wheelbase = np.ascontiguousarray(wheelbase_meters.reshape(-1), dtype=np.float32)
    flat_maximum_speed = np.ascontiguousarray(maximum_speed_mps.reshape(-1), dtype=np.float32)
    output = binding.classic_step_diagnostic(
        flat_state,
        flat_action,
        flat_wheelbase,
        flat_maximum_speed,
        dt_seconds,
    )
    return output.reshape(original_shape)


def _torch_classic_step(state, action, wheelbase_meters, maximum_speed_mps, dt_seconds):
    return classic_step(
        torch.from_numpy(np.ascontiguousarray(state)),
        torch.from_numpy(np.ascontiguousarray(action)),
        torch.from_numpy(np.ascontiguousarray(wheelbase_meters)),
        torch.from_numpy(np.ascontiguousarray(maximum_speed_mps)),
        dt_seconds,
    ).numpy()


ONE_STEP_CASES = [
    pytest.param([1.0, 2.0, 0.2, 5.0, 0.0], [0.0, 0.0], 0.1, id="neutral"),
    pytest.param([1.0, 2.0, 0.2, 5.0, 0.0], [-1.0, 0.0], 0.1, id="braking"),
    pytest.param([1.0, 2.0, 0.2, 5.0, 0.0], [1.0, 0.0], 0.1, id="acceleration"),
    pytest.param([1.0, 2.0, 0.2, -1.0, 0.0], [-0.5, 0.0], 0.1, id="reversing"),
    pytest.param([1.0, 2.0, 0.2, 5.0, 0.0], [0.0, 1.0], 0.1, id="steering_rate_saturation"),
    pytest.param([1.0, 2.0, 0.2, 5.0, 0.8], [0.0, 1.0], 0.1, id="steering_clipping"),
    pytest.param([1.0, 2.0, 0.2, 19.9, 0.0], [1.0, 0.0], 0.1, id="forward_speed_clipping"),
    pytest.param([1.0, 2.0, 0.2, -1.9, 0.0], [-1.0, 0.0], 0.1, id="reverse_speed_clipping"),
    pytest.param([1.0, 2.0, 3.0, 10.0, 0.0], [0.0, 1.0], 1.0, id="positive_heading_wrap"),
    pytest.param([1.0, 2.0, -3.0, 10.0, 0.0], [0.0, -1.0], 1.0, id="negative_heading_wrap"),
]


@pytest.mark.parametrize(("state_values", "action_values", "dt_seconds"), ONE_STEP_CASES)
def test_classic_one_step_matches_c(state_values, action_values, dt_seconds):
    state = np.asarray([state_values], dtype=np.float32)
    action = np.asarray([action_values], dtype=np.float32)
    wheelbase = np.asarray([2.7], dtype=np.float32)
    maximum_speed = np.asarray([20.0], dtype=np.float32)

    expected = _c_classic_step(state, action, wheelbase, maximum_speed, dt_seconds)
    actual = _torch_classic_step(state, action, wheelbase, maximum_speed, dt_seconds)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=STRICT_ONE_STEP_ATOL)


def test_classic_limits_and_heading_interval_are_observable():
    states = np.asarray(
        [
            [0.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.8],
            [0.0, 0.0, 0.0, 19.9, 0.0],
            [0.0, 0.0, 0.0, -1.9, 0.0],
            [0.0, 0.0, 3.0, 10.0, 0.0],
            [0.0, 0.0, -3.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    actions = np.asarray([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    actions = np.ascontiguousarray(actions, dtype=np.float32)
    wheelbase = np.full(6, 2.7, dtype=np.float32)
    maximum_speed = np.full(6, 20.0, dtype=np.float32)
    output = _torch_classic_step(states, actions, wheelbase, maximum_speed, 1.0)

    assert output[0, STATE_STEERING] == pytest.approx(STEERING_RATE_LIMIT_RADIANS_PER_SECOND)
    assert output[1, STATE_STEERING] == pytest.approx(STEERING_LIMIT_RADIANS)
    assert output[2, STATE_SPEED] == pytest.approx(20.0)
    assert output[3, STATE_SPEED] == pytest.approx(MAX_BACKWARD_SPEED_MPS)
    assert -np.pi <= output[4, STATE_HEADING] <= np.pi
    assert -np.pi <= output[5, STATE_HEADING] <= np.pi


def _c_rollout(initial_state, actions, wheelbase_meters, maximum_speed_mps, dt_seconds):
    current_state = np.ascontiguousarray(initial_state, dtype=np.float32)
    states = [current_state.copy()]
    for timestep in range(actions.shape[-2]):
        current_state = _c_classic_step(
            current_state,
            actions[..., timestep, :],
            wheelbase_meters,
            maximum_speed_mps,
            dt_seconds,
        )
        states.append(current_state.copy())
    return np.stack(states, axis=-2)


def test_deterministic_random_full_rollout_matches_c_at_every_step():
    rng = np.random.default_rng(98431)
    batch_count = 2
    agent_count = 3
    transition_count = PARITY_ROLLOUT_TRANSITION_COUNT
    initial_state = np.empty((batch_count, agent_count, 5), dtype=np.float32)
    initial_state[..., 0:2] = rng.uniform(-100.0, 100.0, size=(batch_count, agent_count, 2))
    initial_state[..., STATE_HEADING] = rng.uniform(-np.pi, np.pi, size=(batch_count, agent_count))
    initial_state[..., STATE_SPEED] = rng.uniform(-1.5, 18.0, size=(batch_count, agent_count))
    initial_state[..., STATE_STEERING] = rng.uniform(-0.5, 0.5, size=(batch_count, agent_count))
    actions = rng.uniform(-1.0, 1.0, size=(batch_count, agent_count, transition_count, 2)).astype(np.float32)
    wheelbase = rng.uniform(2.2, 3.5, size=(batch_count, agent_count)).astype(np.float32)
    maximum_speed = rng.uniform(15.0, 25.0, size=(batch_count, agent_count)).astype(np.float32)
    expected = _c_rollout(initial_state, actions, wheelbase, maximum_speed, 0.1)

    actual = classic_rollout(
        torch.from_numpy(initial_state),
        torch.from_numpy(actions),
        torch.ones(actions.shape[:-1], dtype=torch.bool),
        torch.from_numpy(wheelbase),
        torch.from_numpy(maximum_speed),
        0.1,
    ).numpy()

    maximum_error = float(np.max(np.abs(actual - expected)))
    assert maximum_error <= PARITY_ATOL, f"maximum state error was {maximum_error}"


def test_masked_rollout_preserves_state_without_advancing_invalid_transitions():
    initial_state = torch.tensor([[[0.0, 0.0, 0.0, 5.0, 0.0]]], dtype=torch.float32)
    actions = torch.tensor([[[[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]]]], dtype=torch.float32)
    transition_valid = torch.tensor([[[True, False, True]]])
    wheelbase = torch.tensor([[2.7]], dtype=torch.float32)
    maximum_speed = torch.tensor([[20.0]], dtype=torch.float32)
    states = classic_rollout(initial_state, actions, transition_valid, wheelbase, maximum_speed, 0.1)

    assert torch.equal(states[..., 2, :], states[..., 1, :])
    expected_final = classic_step(states[..., 2, :], actions[..., 2, :], wheelbase, maximum_speed, 0.1)
    assert torch.equal(states[..., 3, :], expected_final)


def _real_replay_drive():
    return Drive(
        map_dir=str(FIXTURE_MAP),
        num_maps=1,
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_eval_scenarios=1,
        max_scenarios_per_batch=1,
        eval_map_indices=[0],
        eval_scenario_seeds=[42],
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
        scenario_length=64,
        resample_frequency=64,
        init_step=0,
        init_step_spread=False,
        reward_conditioning=False,
        reward_randomization=False,
        use_neighbor_cache=0,
    )


def _longest_valid_transition_run(transition_valid):
    best_start = 0
    best_length = 0
    run_start = 0
    run_length = 0
    for timestep, is_valid in enumerate(transition_valid.tolist()):
        if is_valid:
            if run_length == 0:
                run_start = timestep
            run_length += 1
            if run_length > best_length:
                best_start = run_start
                best_length = run_length
            continue
        run_length = 0
    return best_start, best_length


def test_actions_approximately_derived_from_real_trajectory_match_c_rollout():
    drive = _real_replay_drive()
    try:
        drive.reset()
        scenario = export_drive_scenarios(drive, raster_resolution_meters=5.0)
    finally:
        drive.close()

    selected_agent_idx = None
    selected_start = None
    for agent_idx in torch.where(scenario.vehicle_mask[0])[0].tolist():
        run_start, run_length = _longest_valid_transition_run(scenario.transition_valid[0, agent_idx])
        if run_length >= 32:
            selected_agent_idx = agent_idx
            selected_start = run_start
            break
    assert selected_agent_idx is not None

    transition_count = 32
    logged = scenario.logged_state[0, selected_agent_idx, selected_start : selected_start + transition_count + 1]
    wrapped_heading_delta = torch.atan2(
        torch.sin(logged[1:, STATE_HEADING] - logged[:-1, STATE_HEADING]),
        torch.cos(logged[1:, STATE_HEADING] - logged[:-1, STATE_HEADING]),
    )
    acceleration_action = (logged[1:, STATE_SPEED] - logged[:-1, STATE_SPEED]) / scenario.dt_seconds
    acceleration_action /= float(binding.ACCELERATION_VALUES[-1])
    next_speed = logged[1:, STATE_SPEED]
    safe_speed = torch.where(next_speed.abs() > 0.5, next_speed, torch.full_like(next_speed, 0.5))
    yaw_rate = wrapped_heading_delta / scenario.dt_seconds
    target_steering = torch.atan(yaw_rate * scenario.wheelbase_meters[0, selected_agent_idx] / safe_speed)
    steering_action = target_steering / float(binding.STEERING_VALUES[-1])
    actions = torch.stack((acceleration_action, steering_action), dim=-1).clamp(-1.0, 1.0)
    initial_state = logged[0].clone()
    initial_state[STATE_STEERING] = 0.0
    wheelbase = scenario.wheelbase_meters[0, selected_agent_idx].reshape(1)
    maximum_speed = scenario.maximum_speed_mps[0, selected_agent_idx].reshape(1)

    expected = _c_rollout(
        initial_state.numpy().reshape(1, 5),
        actions.numpy().reshape(1, transition_count, 2),
        wheelbase.numpy(),
        maximum_speed.numpy(),
        scenario.dt_seconds,
    )
    actual = classic_rollout(
        initial_state.reshape(1, 5),
        actions.reshape(1, transition_count, 2),
        torch.ones((1, transition_count), dtype=torch.bool),
        wheelbase,
        maximum_speed,
        scenario.dt_seconds,
    ).numpy()

    maximum_error = float(np.max(np.abs(actual - expected)))
    assert maximum_error <= PARITY_ATOL, f"maximum real-action state error was {maximum_error}"


def test_action_gradients_reach_acceleration_and_steering():
    initial_state = torch.tensor([[0.0, 0.0, 0.2, 8.0, 0.0]], dtype=torch.float32)
    actions = torch.full((1, 12, 2), 0.1, dtype=torch.float32, requires_grad=True)
    states = classic_rollout(
        initial_state,
        actions,
        torch.ones((1, 12), dtype=torch.bool),
        torch.tensor([2.8], dtype=torch.float32),
        torch.tensor([20.0], dtype=torch.float32),
        0.1,
    )
    loss = states[0, -1, 0] + 0.5 * states[0, -1, 1]
    loss.backward()

    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert torch.any(actions.grad[..., 0] != 0)
    assert torch.any(actions.grad[..., 1] != 0)


def test_c_diagnostic_rejects_out_of_contract_actions():
    state = np.zeros((1, 5), dtype=np.float32)
    action = np.asarray([[1.01, 0.0]], dtype=np.float32)
    metadata = np.ones(1, dtype=np.float32)

    with pytest.raises(ValueError, match="within"):
        binding.classic_step_diagnostic(state, action, metadata, metadata, 0.1)
