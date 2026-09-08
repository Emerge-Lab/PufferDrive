"""Authoritative C replay and reactive-controller iteration for ReGentS."""

import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents.adapter import DEFAULT_RASTER_RESOLUTION_METERS
from pufferlib.ocean.regents.inverse_dynamics import estimate_expert_actions
from pufferlib.ocean.regents.optimizer import (
    SUPPORTED_SDC_CONTROLLERS,
    FrozenEgoTrajectory,
    ReGentSOptimizationConfig,
    ReGentSOptimizationResult,
    capture_frozen_idm_trajectory,
    ego_trajectory_source,
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


# Ninety float32 integration steps differ by up to 3.586e-4 between Torch and C
# on the WOD audit cohort; keep the parity gate sub-millimetric without rejecting
# C-confirmed collisions for expected transcendental rounding accumulation.
C_REPLAY_TOLERANCE = 5e-4
INITIAL_STATE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class CReplayMetrics:
    maximum_trajectory_error: float
    maximum_ego_reference_error: float
    first_collision_timestep: int | None
    first_collision_pair: tuple[int, int] | None
    first_ego_collision_timestep: int | None
    ego_collision: bool
    actionable_collision: bool
    background_collision: bool
    offroad: bool
    baseline_ego_collision: bool
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
    ego_refresh_count: int
    deterministic_seed: int
    ego_source: str


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


def _capture_c_rollout(
    drive,
    transition_count,
    expected_scenario_id,
    agent_count,
    seed,
    capture_html_frames=False,
    ego_action_fn=None,
    capture_observations=False,
):
    """Reset and step one installed action plan, recording states and C events.

    `ego_action_fn` supplies the ego action for a policy SDC. A native controller
    ignores the action buffer, so leaving it None steps neutral actions as before.
    Observations ride alongside the HTML frames, one per captured frame.
    """
    if capture_observations and not capture_html_frames:
        raise ValueError("capture_observations requires capture_html_frames")
    observations, _ = drive.reset(seed=seed)
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
        drive.get_obs_html_frame(*(scratch[key] for key in scratch))
        for key, array in scratch.items():
            html_frames[key].append(array[0].copy())
        if capture_observations:
            html_frames["obs"] = [np.asarray(observations, dtype=np.float32).copy()]
    state_scratch = np.empty((agent_count, STATE_FEATURE_COUNT), dtype=np.float32)
    valid_scratch = np.empty(agent_count, dtype=np.bool_)
    ego_action_scratch = np.empty(2, dtype=np.float32)
    for transition_idx in range(transition_count):
        step_actions = neutral_actions if ego_action_fn is None else ego_action_fn(observations)
        observations = drive.step(step_actions)[0]
        binding.regents_get_states(drive.c_envs, state_scratch, valid_scratch, ego_action_scratch)
        if not np.isfinite(state_scratch[valid_scratch]).all():
            raise RuntimeError("C replay emitted NaN or Inf")
        states.append(state_scratch.copy())
        validity.append(valid_scratch.copy())
        ego_actions.append(ego_action_scratch.copy())
        if scratch is not None:
            drive.get_obs_html_frame(*(scratch[key] for key in scratch))
            for key, array in scratch.items():
                html_frames[key].append(array[0].copy())
            if capture_observations:
                html_frames["obs"].append(np.asarray(observations, dtype=np.float32).copy())
        events = binding.regents_get_events(drive.c_envs)
        pairs = tuple(sorted(tuple(sorted(int(index) for index in pair)) for pair in events["collision_pairs"]))
        if pairs:
            collision_pairs.append((transition_idx + 1, pairs))
        offroad |= np.asarray(events["offroad"], dtype=np.bool_)
    # The buffer getter carries no scenario identity, so the horizon is bracketed by a
    # dict read at each end rather than one per step.
    if _single_payload(drive.get_state()).get("scenario_id") != expected_scenario_id:
        raise RuntimeError("Drive changed scenario during C replay")
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


def _unconfirmed_collision_reason(ego_collision, baseline_ego_collision):
    """Name why C attributed no ego/selected-adversary collision to the optimization.

    Torch found a collision and parity held, so the three outcomes below are the only
    ways the authoritative replay can still decline to credit the adversary. They are
    reported separately because they call for different work: a pre-existing baseline
    contact is an ego-quality problem, an ego collision with another agent is an
    adversary-selection problem, and neither is the optimization failing to converge.
    """
    if ego_collision:
        return "ego_collision_with_other_agent"
    if baseline_ego_collision:
        return "ego_collision_present_in_baseline"
    return "no_ego_collision_in_replay"


def baseline_relative_events(baseline, adversarial, selected_adversary_idx, injected_agent_mask):
    """Attribute C events to the optimization by subtracting the baseline rollout's.

    This is the single definition of ego / actionable / background collision and
    introduced off-road; generation replay and artifact-set evaluation share it so the
    two can never disagree about what an event means.
    """
    baseline_pairs = {pair for _, pairs in baseline.collision_pairs for pair in pairs}
    baseline_ego_collision = any(0 in pair for pair in baseline_pairs)
    ego_collision = False
    actionable_collision = False
    background_collision = False
    first_collision_timestep = None
    first_collision_pair = None
    first_ego_collision_timestep = None
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
            if first_ego_collision_timestep is None:
                first_ego_collision_timestep = state_timestep
            other_idx = right_idx if left_idx == 0 else left_idx
            actionable_collision |= other_idx == selected_adversary_idx
    introduced_offroad = adversarial.offroad & ~baseline.offroad
    offroad = bool(introduced_offroad[injected_agent_mask].any())
    return (
        ego_collision,
        actionable_collision,
        background_collision,
        offroad,
        first_collision_timestep,
        first_collision_pair,
        first_ego_collision_timestep,
        baseline_ego_collision,
        baseline_pairs,
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
    # Injected adversaries integrate classic dynamics whatever the ego's model is.
    if drive.dynamics_model_flag not in (binding.DYNAMICS_MODEL_CLASSIC, binding.DYNAMICS_MODEL_JERK):
        raise ValueError("Stage 6 requires dynamics_model='classic' or 'jerk'")
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
    ego_action_fn=None,
    capture_observations=False,
):
    """Replay optimized background controls in C and use C as the success oracle."""
    _validate_replay_inputs(drive, scenario, optimization, tolerance)
    if not isinstance(capture_html_frames, bool):
        raise TypeError("capture_html_frames must be a boolean")
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
        baseline = _capture_c_rollout(
            drive,
            transition_count,
            scenario_id,
            agent_count,
            seed,
            capture_html_frames,
            ego_action_fn,
            capture_observations,
        )
        binding.regents_set_action_plan(drive.c_envs, optimized_plan, action_mask)
        adversarial = _capture_c_rollout(
            drive,
            transition_count,
            scenario_id,
            agent_count,
            seed,
            capture_html_frames,
            ego_action_fn,
            capture_observations,
        )
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

    events = baseline_relative_events(
        baseline,
        adversarial,
        optimization.selected_adversary_idx,
        injected_agent_mask[0].numpy(),
    )
    (
        ego_collision,
        actionable_collision,
        background_collision,
        offroad,
        first_collision_timestep,
        first_collision_pair,
        first_ego_collision_timestep,
        baseline_ego_collision,
        baseline_pairs,
    ) = events

    failure_reason = None
    if maximum_error > tolerance:
        failure_reason = "c_torch_trajectory_mismatch"
    elif not optimization.success:
        failure_reason = optimization.failure_reason or "torch_optimization_failed"
    elif not actionable_collision:
        failure_reason = _unconfirmed_collision_reason(ego_collision, baseline_ego_collision)
    metrics = CReplayMetrics(
        maximum_trajectory_error=maximum_error,
        maximum_ego_reference_error=maximum_ego_reference_error,
        first_collision_timestep=first_collision_timestep,
        first_collision_pair=first_collision_pair,
        first_ego_collision_timestep=first_ego_collision_timestep,
        ego_collision=ego_collision,
        actionable_collision=actionable_collision,
        background_collision=background_collision,
        offroad=offroad,
        baseline_ego_collision=baseline_ego_collision,
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


def make_c_ego_rollout(drive, scenario, horizon_transition_count, seed, ego_action_fn=None):
    """Build the callback that re-rolls the C ego against a candidate action plan.

    Drive is the authority on how the ego answers an adversary, so a refresh is a real C
    rollout with the plan installed. The plan is cleared afterwards so the verification
    replay starts from the same clean env the optimizer's caller handed over.
    """
    scenario_id = scenario.scenario_ids[0]
    agent_count = scenario.max_agent_count
    ego_idx = int(torch.where(scenario.ego_mask[0])[0].item())
    source = ego_trajectory_source(drive.sdc_controller)

    def ego_rollout(drive_actions, action_mask):
        plan = np.ascontiguousarray(drive_actions[0].detach().cpu().numpy(), dtype=np.float32)
        mask = np.ascontiguousarray(action_mask[0].detach().cpu().numpy(), dtype=np.bool_)
        try:
            binding.regents_set_action_plan(drive.c_envs, plan, mask)
            rollout = _capture_c_rollout(
                drive,
                horizon_transition_count,
                scenario_id,
                agent_count,
                seed,
                ego_action_fn=ego_action_fn,
            )
        finally:
            binding.regents_set_action_plan(drive.c_envs, plan, np.zeros_like(mask))
        return FrozenEgoTrajectory(
            state=rollout.states[:, ego_idx].detach().clone(),
            valid=rollout.state_valid[:, ego_idx].detach().clone(),
            scenario_ids=scenario.scenario_ids,
            source=source,
        )

    return ego_rollout


def run_reactive_generation(
    drive,
    optimization_config=None,
    *,
    deterministic_seed,
    horizon_transition_count,
    tolerance=C_REPLAY_TOLERANCE,
    capture_html_frames=False,
    show_progress=True,
    raster_resolution_meters=DEFAULT_RASTER_RESOLUTION_METERS,
    ego_action_fn=None,
    capture_observations=False,
):
    """Optimize one scenario against a periodically re-rolled C ego, then verify in C.

    The ego reacts inside the optimization loop, on `ego_refresh_interval`; C stays the
    success oracle through a single verification replay at the end. `ego_action_fn`
    supplies actions for a policy ego and must be absent for a native C controller.
    """
    if drive.sdc_controller not in SUPPORTED_SDC_CONTROLLERS:
        raise ValueError("Reactive generation requires sdc_controller='idm', 'replay', or 'policy'")
    if optimization_config is None:
        optimization_config = ReGentSOptimizationConfig()

    scenario, frozen_ego = capture_frozen_idm_trajectory(
        drive,
        horizon_transition_count,
        seed=deterministic_seed,
        raster_resolution_meters=raster_resolution_meters,
        ego_action_fn=ego_action_fn,
    )
    inverse_dynamics = estimate_expert_actions(
        scenario,
        horizon_transition_count=horizon_transition_count,
    )
    # A log-replayed ego ignores the adversary, so a refresh would spend C steps
    # reproducing the trajectory already captured.
    ego_rollout_fn = None
    if drive.sdc_controller != binding.CONTROLLER_REPLAY and optimization_config.ego_refresh_interval > 0:
        ego_rollout_fn = make_c_ego_rollout(
            drive, scenario, horizon_transition_count, deterministic_seed, ego_action_fn
        )
    else:
        optimization_config = replace(optimization_config, ego_refresh_interval=0)

    optimization = optimize_frozen_ego_scenario(
        scenario,
        frozen_ego,
        optimization_config,
        deterministic_seed=deterministic_seed,
        horizon_transition_count=horizon_transition_count,
        show_progress=show_progress,
        inverse_dynamics=inverse_dynamics,
        ego_rollout_fn=ego_rollout_fn,
    )
    replay = replay_optimized_scenario_in_c(
        drive,
        scenario,
        optimization,
        seed=deterministic_seed,
        tolerance=tolerance,
        capture_html_frames=capture_html_frames,
        ego_action_fn=ego_action_fn,
        capture_observations=capture_observations,
    )
    return ReactiveGenerationResult(
        scenario=scenario,
        optimization=optimization,
        replay=replay,
        ego_refresh_count=optimization.ego_refresh_count,
        deterministic_seed=deterministic_seed,
        ego_source=frozen_ego.source,
    )
