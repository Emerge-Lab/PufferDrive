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
    ("neutral", [1.0, 2.0, 0.2, 5.0, 0.0], [0.0, 0.0], 0.1),
    ("braking", [1.0, 2.0, 0.2, 5.0, 0.0], [-1.0, 0.0], 0.1),
    ("acceleration", [1.0, 2.0, 0.2, 5.0, 0.0], [1.0, 0.0], 0.1),
    ("reversing", [1.0, 2.0, 0.2, -1.0, 0.0], [-0.5, 0.0], 0.1),
    ("steering_rate_saturation", [1.0, 2.0, 0.2, 5.0, 0.0], [0.0, 1.0], 0.1),
    ("steering_clipping", [1.0, 2.0, 0.2, 5.0, 0.8], [0.0, 1.0], 0.1),
    ("forward_speed_clipping", [1.0, 2.0, 0.2, 19.9, 0.0], [1.0, 0.0], 0.1),
    ("reverse_speed_clipping", [1.0, 2.0, 0.2, -1.9, 0.0], [-1.0, 0.0], 0.1),
    ("positive_heading_wrap", [1.0, 2.0, 3.0, 10.0, 0.0], [0.0, 1.0], 1.0),
    ("negative_heading_wrap", [1.0, 2.0, -3.0, 10.0, 0.0], [0.0, -1.0], 1.0),
]


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


def test_classic_step_and_rollout_match_c_including_limits_and_masking():
    """One-step parity at every limit, a 64-transition rollout, and validity masking."""
    wheelbase = np.asarray([2.7], dtype=np.float32)
    maximum_speed = np.asarray([20.0], dtype=np.float32)
    for label, state_values, action_values, dt_seconds in ONE_STEP_CASES:
        state = np.asarray([state_values], dtype=np.float32)
        action = np.asarray([action_values], dtype=np.float32)
        expected = _c_classic_step(state, action, wheelbase, maximum_speed, dt_seconds)
        actual = _torch_classic_step(state, action, wheelbase, maximum_speed, dt_seconds)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=STRICT_ONE_STEP_ATOL, err_msg=label)

    # Each saturation the C model applies must be observable in the Torch output.
    limit_states = np.asarray(
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
    limit_actions = np.ascontiguousarray(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float32
    )
    limits = _torch_classic_step(
        limit_states, limit_actions, np.full(6, 2.7, dtype=np.float32), np.full(6, 20.0, dtype=np.float32), 1.0
    )
    assert limits[0, STATE_STEERING] == pytest.approx(STEERING_RATE_LIMIT_RADIANS_PER_SECOND)
    assert limits[1, STATE_STEERING] == pytest.approx(STEERING_LIMIT_RADIANS)
    assert limits[2, STATE_SPEED] == pytest.approx(20.0)
    assert limits[3, STATE_SPEED] == pytest.approx(MAX_BACKWARD_SPEED_MPS)
    assert -np.pi <= limits[4, STATE_HEADING] <= np.pi
    assert -np.pi <= limits[5, STATE_HEADING] <= np.pi

    rng = np.random.default_rng(98431)
    batch_count, agent_count = 2, 3
    transition_count = PARITY_ROLLOUT_TRANSITION_COUNT
    initial_state = np.empty((batch_count, agent_count, 5), dtype=np.float32)
    initial_state[..., 0:2] = rng.uniform(-100.0, 100.0, size=(batch_count, agent_count, 2))
    initial_state[..., STATE_HEADING] = rng.uniform(-np.pi, np.pi, size=(batch_count, agent_count))
    initial_state[..., STATE_SPEED] = rng.uniform(-1.5, 18.0, size=(batch_count, agent_count))
    initial_state[..., STATE_STEERING] = rng.uniform(-0.5, 0.5, size=(batch_count, agent_count))
    random_actions = rng.uniform(-1.0, 1.0, size=(batch_count, agent_count, transition_count, 2)).astype(np.float32)
    random_wheelbase = rng.uniform(2.2, 3.5, size=(batch_count, agent_count)).astype(np.float32)
    random_maximum_speed = rng.uniform(15.0, 25.0, size=(batch_count, agent_count)).astype(np.float32)
    expected_rollout = _c_rollout(initial_state, random_actions, random_wheelbase, random_maximum_speed, 0.1)
    actual_rollout = classic_rollout(
        torch.from_numpy(initial_state),
        torch.from_numpy(random_actions),
        torch.ones(random_actions.shape[:-1], dtype=torch.bool),
        torch.from_numpy(random_wheelbase),
        torch.from_numpy(random_maximum_speed),
        0.1,
    ).numpy()
    maximum_error = float(np.max(np.abs(actual_rollout - expected_rollout)))
    assert maximum_error <= PARITY_ATOL, f"maximum state error was {maximum_error}"

    # An invalid transition holds the previous state byte-for-byte.
    masked_initial = torch.tensor([[[0.0, 0.0, 0.0, 5.0, 0.0]]], dtype=torch.float32)
    masked_actions = torch.tensor([[[[1.0, 0.5], [1.0, 0.5], [1.0, 0.5]]]], dtype=torch.float32)
    masked_wheelbase = torch.tensor([[2.7]], dtype=torch.float32)
    masked_maximum_speed = torch.tensor([[20.0]], dtype=torch.float32)
    masked = classic_rollout(
        masked_initial,
        masked_actions,
        torch.tensor([[[True, False, True]]]),
        masked_wheelbase,
        masked_maximum_speed,
        0.1,
    )
    assert torch.equal(masked[..., 2, :], masked[..., 1, :])
    assert torch.equal(
        masked[..., 3, :],
        classic_step(masked[..., 2, :], masked_actions[..., 2, :], masked_wheelbase, masked_maximum_speed, 0.1),
    )

    with pytest.raises(ValueError, match="within"):
        binding.classic_step_diagnostic(
            np.zeros((1, 5), dtype=np.float32),
            np.asarray([[1.01, 0.0]], dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.ones(1, dtype=np.float32),
            0.1,
        )


def test_actions_derived_from_a_real_trajectory_match_the_c_rollout():
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
    actions = torch.stack((acceleration_action, target_steering / float(binding.STEERING_VALUES[-1])), dim=-1).clamp(
        -1.0, 1.0
    )
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


def test_action_gradients_reach_both_channels_and_survive_a_stationary_agent():
    """A parked adversary must still receive the full multi-step acceleration gradient.

    C recovers signed speed as sqrt(vx^2 + vy^2), which is flat at an exactly
    stationary agent. Carrying that derivative would truncate the rollout
    gradient to its single-step direct effect.
    """
    actions = torch.full((1, 12, 2), 0.1, dtype=torch.float32, requires_grad=True)
    states = classic_rollout(
        torch.tensor([[0.0, 0.0, 0.2, 8.0, 0.0]], dtype=torch.float32),
        actions,
        torch.ones((1, 12), dtype=torch.bool),
        torch.tensor([2.8], dtype=torch.float32),
        torch.tensor([20.0], dtype=torch.float32),
        0.1,
    )
    (states[0, -1, 0] + 0.5 * states[0, -1, 1]).backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert torch.any(actions.grad[..., 0] != 0)
    assert torch.any(actions.grad[..., 1] != 0)

    transition_count = 20
    wheelbase = torch.tensor([3.0], dtype=torch.float32)
    maximum_speed = torch.tensor([30.0], dtype=torch.float32)
    transition_valid = torch.ones((1, transition_count), dtype=torch.bool)

    def final_x(action_values, initial_speed_mps):
        initial_state = torch.tensor([[0.0, 0.0, 0.0, initial_speed_mps, 0.0]], dtype=torch.float32)
        rollout = classic_rollout(initial_state, action_values, transition_valid, wheelbase, maximum_speed, 0.1)
        return rollout[0, -1, 0]

    for initial_speed_mps in (0.0, 1e-6, 0.5):
        stationary_actions = torch.zeros((1, transition_count, 2), dtype=torch.float32, requires_grad=True)
        final_x(stationary_actions, initial_speed_mps).backward()
        analytic = float(stationary_actions.grad[0, 0, 0])
        epsilon = 1e-3
        forward = torch.zeros((1, transition_count, 2), dtype=torch.float32)
        forward[0, 0, 0] = epsilon
        backward = torch.zeros((1, transition_count, 2), dtype=torch.float32)
        backward[0, 0, 0] = -epsilon
        finite_difference = (
            float(final_x(forward, initial_speed_mps)) - float(final_x(backward, initial_speed_mps))
        ) / (2 * epsilon)
        assert finite_difference == pytest.approx(analytic, rel=1e-3)
