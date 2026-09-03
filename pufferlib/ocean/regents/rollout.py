"""Authoritative C replay and reactive-controller iteration for ReGentS."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents.optimizer import (
    FrozenEgoTrajectory,
    ReGentSOptimizationConfig,
    ReGentSOptimizationResult,
    capture_frozen_idm_trajectory,
    optimize_frozen_ego_scenario,
)
from pufferlib.ocean.regents.state import (
    STATE_FEATURE_COUNT,
    STATE_HEADING,
    STATE_SPEED,
    STATE_STEERING,
    STATE_X,
    STATE_Y,
    ScenarioBatch,
    signed_speed_from_c_velocity,
)


C_REPLAY_TOLERANCE = 1e-4
INITIAL_STATE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class CReplayMetrics:
    maximum_trajectory_error: float
    maximum_ego_reference_error: float
    first_collision_timestep: int | None
    first_collision_pair: tuple[int, int] | None
    ego_collision: bool
    actionable_collision: bool
    background_collision: bool
    offroad: bool
    baseline_collision_pair_count: int
    compared_state_count: int


@dataclass(frozen=True)
class CReplayResult:
    states: torch.Tensor
    state_valid: torch.Tensor
    ego_actions: torch.Tensor
    baseline_states: torch.Tensor
    metrics: CReplayMetrics
    success: bool
    failure_reason: str | None
    scenario_payload: dict | None = None
    baseline_frames: dict | None = None
    adversarial_frames: dict | None = None


@dataclass(frozen=True)
class ReactiveGenerationResult:
    scenario: ScenarioBatch
    optimization: ReGentSOptimizationResult
    replay: CReplayResult
    outer_iteration_count: int
    deterministic_seed: int


@dataclass(frozen=True)
class _CRollout:
    states: torch.Tensor
    state_valid: torch.Tensor
    ego_actions: torch.Tensor
    collision_pairs: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    offroad: np.ndarray
    scenario_payload: dict | None
    html_frames: dict | None


def _single_payload(payload):
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return payload[0]
    raise ValueError("Stage 6 C replay requires exactly one Drive scenario")


def _current_states(payload, expected_agent_count):
    scenario = _single_payload(payload)
    agents = scenario.get("agents")
    if not isinstance(agents, list) or len(agents) != expected_agent_count:
        raise RuntimeError("C replay agent count changed")
    states = np.zeros((expected_agent_count, STATE_FEATURE_COUNT), dtype=np.float32)
    valid = np.zeros(expected_agent_count, dtype=np.bool_)
    ego_action = np.zeros(2, dtype=np.float32)
    for stable_agent_idx, agent in enumerate(agents):
        if int(agent.get("id", -1)) != stable_agent_idx:
            raise RuntimeError("C replay changed stable agent identity")
        values = np.asarray(
            (
                agent["sim_x"],
                agent["sim_y"],
                agent["sim_heading"],
                agent["sim_vx"],
                agent["sim_vy"],
                agent["sim_steering"],
            ),
            dtype=np.float32,
        )
        serialized_valid = int(agent["sim_valid"])
        if serialized_valid not in (0, 1):
            raise RuntimeError("C replay emitted invalid sim_valid")
        if serialized_valid and not np.isfinite(values).all():
            raise RuntimeError("C replay emitted NaN or Inf")
        heading = np.float32(math.atan2(math.sin(float(values[2])), math.cos(float(values[2]))))
        states[stable_agent_idx, STATE_X] = values[0]
        states[stable_agent_idx, STATE_Y] = values[1]
        states[stable_agent_idx, STATE_HEADING] = heading
        states[stable_agent_idx, STATE_SPEED] = signed_speed_from_c_velocity(values[3], values[4], heading)
        states[stable_agent_idx, STATE_STEERING] = values[5]
        valid[stable_agent_idx] = bool(serialized_valid)
        if stable_agent_idx == 0:
            ego_action[0] = np.float32(agent["accel_long"] / float(binding.ACCELERATION_VALUES[6]))
            ego_action[1] = np.float32(agent["sim_steering"] / float(binding.STEERING_VALUES[8]))
    return scenario, states, valid, ego_action


def _html_frame_arrays(agent_count, traffic_count):
    return {
        "agent_f32": np.empty((1, agent_count, binding.AGENT_F32_FIELDS), dtype=np.float32),
        "agent_i32": np.empty((1, agent_count, binding.AGENT_I32_FIELDS), dtype=np.int32),
        "metrics_f32": np.empty((1, agent_count, binding.METRICS_F32_FIELDS), dtype=np.float32),
        "puffer_f32": np.empty((1, agent_count, binding.SCORE_F32_FIELDS), dtype=np.float32),
        "traffic_i16": np.empty((1, traffic_count, binding.TRAFFIC_I16_FIELDS), dtype=np.int16),
    }


def _capture_c_rollout(drive, transition_count, expected_scenario_id, agent_count, seed, capture_html_frames=False):
    """Reset and step one installed action plan, recording states and C events."""
    drive.reset(seed=seed)
    payload_scenario, initial_state, initial_valid, _ = _current_states(drive.get_state(), agent_count)
    if payload_scenario.get("scenario_id") != expected_scenario_id:
        raise RuntimeError("C replay reset to a different scenario")
    initial_payload_scenario = payload_scenario
    states = [initial_state]
    validity = [initial_valid]
    ego_actions = []
    collision_pairs = []
    offroad = np.zeros(agent_count, dtype=np.bool_)
    neutral_actions = np.zeros_like(drive.actions)
    scratch = None
    html_frames = None
    if capture_html_frames:
        traffic_count = max(len(payload_scenario.get("traffic_elements") or []), 1)
        scratch = _html_frame_arrays(agent_count, traffic_count)
        html_frames = {key: [] for key in scratch}
        html_frames["obs"] = []
        drive.get_obs_html_frame(*(scratch[key] for key in scratch))
        for key, array in scratch.items():
            html_frames[key].append(array[0].copy())
        html_frames["obs"].append(np.asarray(drive.observations, dtype=np.float32).copy())
    for transition_idx in range(transition_count):
        drive.step(neutral_actions)
        payload_scenario, state, valid, ego_action = _current_states(drive.get_state(), agent_count)
        if payload_scenario.get("scenario_id") != expected_scenario_id:
            raise RuntimeError("Drive changed scenario during C replay")
        states.append(state)
        validity.append(valid)
        ego_actions.append(ego_action)
        if scratch is not None:
            drive.get_obs_html_frame(*(scratch[key] for key in scratch))
            for key, array in scratch.items():
                html_frames[key].append(array[0].copy())
            html_frames["obs"].append(np.asarray(drive.observations, dtype=np.float32).copy())
        events = binding.regents_get_events(drive.c_envs)
        pairs = tuple(sorted(tuple(sorted(int(index) for index in pair)) for pair in events["collision_pairs"]))
        if pairs:
            collision_pairs.append((transition_idx + 1, pairs))
        offroad |= np.asarray(events["offroad"], dtype=np.bool_)
    stacked_states = torch.from_numpy(np.ascontiguousarray(np.stack(states)[None, ...])).transpose(1, 2).contiguous()
    stacked_valid = torch.from_numpy(np.ascontiguousarray(np.stack(validity)[None, ...])).transpose(1, 2).contiguous()
    if ego_actions:
        stacked_ego_actions = torch.from_numpy(np.ascontiguousarray(np.stack(ego_actions)[None, ...]))
    else:
        stacked_ego_actions = torch.empty((1, 0, 2), dtype=torch.float32)
    return _CRollout(
        states=stacked_states,
        state_valid=stacked_valid,
        ego_actions=stacked_ego_actions,
        collision_pairs=tuple(collision_pairs),
        offroad=offroad,
        scenario_payload=initial_payload_scenario if capture_html_frames else None,
        html_frames=None
        if html_frames is None
        else {key: np.stack(frames, axis=0) for key, frames in html_frames.items()},
    )


def _validate_replay_inputs(drive, scenario, optimization, tolerance):
    if not isinstance(scenario, ScenarioBatch) or scenario.batch_size != 1:
        raise ValueError("Stage 6 C replay supports one ScenarioBatch item")
    if not isinstance(optimization, ReGentSOptimizationResult):
        raise TypeError("optimization must be a ReGentSOptimizationResult")
    if drive.num_envs != 1 or drive.simulation_mode != binding.SIMULATION_MODE_REPLAY:
        raise ValueError("Stage 6 requires one Drive environment in replay mode")
    if drive._action_type_flag != binding.ACTION_TYPE_CONTINUOUS:
        raise ValueError("Stage 6 requires continuous actions")
    if drive.dynamics_model_flag != binding.DYNAMICS_MODEL_CLASSIC:
        raise ValueError("Stage 6 requires classic dynamics")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    expected_action_shape = (1, scenario.max_agent_count, optimization.optimized_states.shape[2] - 1, 2)
    if tuple(optimization.optimized_actions.shape) != expected_action_shape:
        raise ValueError("Optimized actions do not match the scenario and rollout horizon")
    if tuple(optimization.initial_actions.shape) != expected_action_shape:
        raise ValueError("Initial actions do not match the scenario and rollout horizon")
    if optimization.optimized_action_mask.shape != optimization.optimized_actions.shape[:-1]:
        raise ValueError("Optimized action mask shape does not match actions")


def replay_optimized_scenario_in_c(
    drive,
    scenario,
    optimization,
    *,
    seed=None,
    tolerance=C_REPLAY_TOLERANCE,
    capture_html_frames=False,
):
    """Replay optimized background controls in C and use C as the success oracle."""
    _validate_replay_inputs(drive, scenario, optimization, tolerance)
    transition_count = optimization.optimized_actions.shape[2]
    if drive.resample_frequency > 0 and transition_count >= drive.resample_frequency:
        raise ValueError("C replay horizon must end before Drive resamples")

    scenario_id = scenario.scenario_ids[0]
    agent_count = scenario.max_agent_count
    action_mask = np.ascontiguousarray(optimization.optimized_action_mask[0].detach().cpu().numpy(), dtype=np.bool_)
    baseline_plan = np.ascontiguousarray(optimization.initial_actions[0].detach().cpu().numpy(), dtype=np.float32)
    optimized_plan = np.ascontiguousarray(optimization.optimized_actions[0].detach().cpu().numpy(), dtype=np.float32)

    # The baseline drives the same actors from their reconstructed logged actions,
    # so only events the optimization introduced are attributed to the adversary.
    try:
        binding.regents_set_action_plan(drive.c_envs, baseline_plan, action_mask)
        baseline = _capture_c_rollout(drive, transition_count, scenario_id, agent_count, seed, capture_html_frames)
        binding.regents_set_action_plan(drive.c_envs, optimized_plan, action_mask)
        adversarial = _capture_c_rollout(drive, transition_count, scenario_id, agent_count, seed, capture_html_frames)
    finally:
        binding.regents_set_action_plan(drive.c_envs, baseline_plan, np.zeros_like(action_mask))

    torch_states = optimization.optimized_states.detach().cpu()
    injected_agent_mask = optimization.optimized_action_mask.any(dim=2).detach().cpu()
    ego_mask = scenario.ego_mask.detach().cpu()
    joint_valid = optimization.state_valid.detach().cpu() & adversarial.state_valid
    initial_difference = torch.abs(
        adversarial.states[:, :, 0, : STATE_HEADING + 1] - torch_states[:, :, 0, : STATE_HEADING + 1]
    )
    initial_difference = initial_difference[joint_valid[:, :, 0]]
    if initial_difference.numel() and float(initial_difference.max().item()) > INITIAL_STATE_TOLERANCE:
        raise RuntimeError("C reset to a different initial pose than the Torch scenario")

    # A state carries C speed and wheel steering only once injection has integrated it.
    injected_state_mask = torch.zeros_like(joint_valid)
    injected_state_mask[:, :, 1:] = optimization.optimized_action_mask.detach().cpu()
    feature_mask = joint_valid[..., None].expand_as(torch_states).clone()
    feature_mask[..., STATE_SPEED] &= injected_state_mask
    feature_mask[..., STATE_STEERING] &= injected_state_mask
    # The initial state is a shared input, and C stores adversary speed only once injection starts.
    timestep_index = torch.arange(torch_states.shape[2])[None, None, :, None]
    feature_mask &= timestep_index > 0
    first_any_collision_timestep = adversarial.collision_pairs[0][0] if adversarial.collision_pairs else None
    if first_any_collision_timestep is not None:
        feature_mask &= timestep_index <= first_any_collision_timestep
    differences = torch.abs(adversarial.states - torch_states)
    ego_feature_mask = feature_mask & ego_mask[..., None, None]
    maximum_ego_reference_error = float(differences[ego_feature_mask].max().item()) if ego_feature_mask.any() else 0.0
    # A reactive ego answers the new adversary, so it is reported instead of gated.
    if drive.sdc_controller != binding.CONTROLLER_REPLAY:
        feature_mask &= ~ego_mask[..., None, None]
    maximum_error = float(differences[feature_mask].max().item()) if feature_mask.any() else 0.0
    compared_state_count = int(feature_mask.sum().item())

    baseline_pairs = {pair for _, pairs in baseline.collision_pairs for pair in pairs}
    selected_adversary_idx = optimization.selected_adversary_idx
    ego_collision = False
    actionable_collision = False
    background_collision = False
    first_collision_timestep = None
    first_collision_pair = None
    for state_timestep, pairs in adversarial.collision_pairs:
        introduced = [pair for pair in pairs if pair not in baseline_pairs]
        if not introduced:
            continue
        if first_collision_timestep is None:
            first_collision_timestep = state_timestep
            first_collision_pair = min(introduced)
        for left_idx, right_idx in introduced:
            if left_idx != 0 and right_idx != 0:
                background_collision = True
                continue
            ego_collision = True
            other_idx = right_idx if left_idx == 0 else left_idx
            actionable_collision |= other_idx == selected_adversary_idx
    introduced_offroad = adversarial.offroad & ~baseline.offroad
    offroad = bool(introduced_offroad[injected_agent_mask[0].numpy()].any())

    failure_reason = None
    if maximum_error > tolerance:
        failure_reason = "c_torch_trajectory_mismatch"
    elif not optimization.success:
        failure_reason = optimization.failure_reason or "torch_optimization_failed"
    elif not actionable_collision:
        failure_reason = "c_did_not_confirm_actionable_ego_collision"
    elif background_collision:
        failure_reason = "c_background_collision"
    elif offroad:
        failure_reason = "c_offroad"
    metrics = CReplayMetrics(
        maximum_trajectory_error=maximum_error,
        maximum_ego_reference_error=maximum_ego_reference_error,
        first_collision_timestep=first_collision_timestep,
        first_collision_pair=first_collision_pair,
        ego_collision=ego_collision,
        actionable_collision=actionable_collision,
        background_collision=background_collision,
        offroad=offroad,
        baseline_collision_pair_count=len(baseline_pairs),
        compared_state_count=compared_state_count,
    )
    return CReplayResult(
        states=adversarial.states,
        state_valid=adversarial.state_valid,
        ego_actions=adversarial.ego_actions,
        baseline_states=baseline.states,
        metrics=metrics,
        success=failure_reason is None,
        failure_reason=failure_reason,
        scenario_payload=adversarial.scenario_payload,
        baseline_frames=baseline.html_frames,
        adversarial_frames=adversarial.html_frames,
    )


def run_reactive_idm_generation(
    drive,
    optimization_config=None,
    *,
    deterministic_seed,
    horizon_transition_count,
    maximum_outer_iterations=3,
    tolerance=C_REPLAY_TOLERANCE,
    capture_html_frames=False,
):
    """Alternate detached native-IDM C rollouts and Torch adversary blocks."""
    if drive.sdc_controller != binding.CONTROLLER_IDM:
        raise ValueError("Reactive IDM generation requires sdc_controller='idm'")
    if not isinstance(maximum_outer_iterations, int) or maximum_outer_iterations < 1:
        raise ValueError("maximum_outer_iterations must be a positive integer")
    if optimization_config is None:
        optimization_config = ReGentSOptimizationConfig()

    scenario, frozen_ego = capture_frozen_idm_trajectory(drive, horizon_transition_count, seed=deterministic_seed)
    previous_actions = None
    for outer_iteration_idx in range(maximum_outer_iterations):
        optimization = optimize_frozen_ego_scenario(
            scenario,
            frozen_ego,
            optimization_config,
            deterministic_seed=deterministic_seed,
            horizon_transition_count=horizon_transition_count,
        )
        replay = replay_optimized_scenario_in_c(
            drive,
            scenario,
            optimization,
            seed=deterministic_seed,
            tolerance=tolerance,
            capture_html_frames=capture_html_frames,
        )
        if replay.success:
            break
        if previous_actions is not None and torch.equal(previous_actions, optimization.optimized_actions):
            break
        previous_actions = optimization.optimized_actions.detach().clone()
        ego_idx = int(torch.where(scenario.ego_mask[0])[0].item())
        frozen_ego = FrozenEgoTrajectory(
            state=replay.states[:, ego_idx].detach().clone(),
            valid=replay.state_valid[:, ego_idx].detach().clone(),
            scenario_id=scenario.scenario_ids[0],
            source="c_idm",
        )
    return ReactiveGenerationResult(
        scenario=scenario,
        optimization=optimization,
        replay=replay,
        outer_iteration_count=outer_iteration_idx + 1,
        deterministic_seed=deterministic_seed,
    )
