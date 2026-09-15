"""Authoritative C replay and reactive-controller iteration for ReGentS."""

import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents.adapter import DEFAULT_RASTER_RESOLUTION_METERS, export_drive_scenarios
from pufferlib.ocean.regents.inverse_dynamics import estimate_expert_actions
from pufferlib.ocean.regents.optimizer import (
    FrozenEgoTrajectory,
    ReGentSOptimizationConfig,
    ReGentSOptimizationResult,
    VERIFICATION_ACCEPT,
    VERIFICATION_REJECT,
    VERIFICATION_RETRY,
    optimize_frozen_ego_scenario,
)
from pufferlib.ocean.regents.state import (
    REGENTS_EGO_ACTION_FEATURE_COUNT,
    STATE_FEATURE_COUNT,
    STATE_HEADING,
    STATE_SPEED,
    STATE_STEERING,
    agent_state_rows,
    single_scenario_payload,
    Scenario,
)


# Ninety float32 integration steps differ by up to 3.586e-4 between Torch and C
# on the WOD audit cohort; keep the parity gate sub-millimetric without rejecting
# C-confirmed collisions for expected transcendental rounding accumulation.
C_REPLAY_TOLERANCE = 5e-4
INITIAL_STATE_TOLERANCE = 1e-4
# Drive pins the ego to stable agent row zero, and `regents_get_states` preserves that order.
STABLE_EGO_AGENT_IDX = 0

# The ego controllers ReGentS can freeze and refresh; backgrounds must stay replay. The
# mapped name is the ego trajectory source recorded on an artifact.
EGO_TRAJECTORY_SOURCE_BY_CONTROLLER = {
    binding.CONTROLLER_IDM: "c_idm",
    binding.CONTROLLER_CORRIDOR_IDM: "c_corridor_idm",
    binding.CONTROLLER_PDM: "c_pdm",
    binding.CONTROLLER_REPLAY: "c_replay",
    binding.CONTROLLER_POLICY: "c_policy",
}
SUPPORTED_SDC_CONTROLLERS = tuple(EGO_TRAJECTORY_SOURCE_BY_CONTROLLER)
SUPPORTED_SDC_CONTROLLER_NAMES = "'idm', 'corridor_idm', 'pdm', 'replay', or 'policy'"


def ego_trajectory_source(sdc_controller):
    """Name the C ego controller a captured trajectory came from."""
    source = EGO_TRAJECTORY_SOURCE_BY_CONTROLLER.get(sdc_controller)
    if source is None:
        raise ValueError(f"ReGentS requires sdc_controller={SUPPORTED_SDC_CONTROLLER_NAMES}")
    return source


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
    episode_log: dict | None = None
    avoidability_debug: dict | None = None
    baseline_avoidability_debug: dict | None = None


@dataclass(frozen=True)
class CVerificationCandidate:
    initial_actions: torch.Tensor
    optimized_actions: torch.Tensor
    optimized_states: torch.Tensor
    state_valid: torch.Tensor
    optimized_action_mask: torch.Tensor
    success: bool = True
    failure_reason: str | None = None


@dataclass(frozen=True)
class ReactiveGenerationResult:
    scenario: Scenario
    optimization: ReGentSOptimizationResult
    replay: CReplayResult
    ego_refresh_count: int
    deterministic_seed: int
    ego_source: str


@dataclass(frozen=True)
class BaselineRelativeEvents:
    ego_collision: bool
    actionable_collision: bool
    background_collision: bool
    offroad: bool
    first_collision_timestep: int | None
    first_collision_pair: tuple[int, int] | None
    first_ego_collision_timestep: int | None
    baseline_ego_collision: bool
    baseline_collision_pairs: frozenset


@dataclass(frozen=True)
class _CRollout:
    states: torch.Tensor
    state_valid: torch.Tensor
    ego_actions: torch.Tensor
    collision_pairs: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    offroad: np.ndarray
    scenario_payload: dict
    html_frames: dict | None
    episode_log: dict | None
    avoidability_debug: dict | None


def _capture_c_rollout(
    drive,
    transition_count,
    *,
    expected_scenario_id=None,
    agent_count=None,
    seed=None,
    capture_html_frames=False,
    ego_action_fn=None,
    capture_observations=False,
):
    """Reset and step one installed action plan, recording states and C events.

    `ego_action_fn` supplies the ego action for a policy SDC. A native controller
    ignores the action buffer, so leaving it None steps neutral actions as before.
    Observations ride alongside the HTML frames, one per captured frame. A caller
    that already knows the scenario identity passes it to be checked; the capture
    that establishes it adopts what C reset to and still brackets the horizon with it.
    """
    if capture_observations and not capture_html_frames:
        raise ValueError("capture_observations requires capture_html_frames")
    observations, _ = drive.reset(seed=seed)
    payload = single_scenario_payload(drive.get_state())
    if agent_count is None:
        agent_count = len(payload.get("agents") or ())
    payload_scenario, initial_state, initial_valid, _ = agent_state_rows(payload, agent_count)
    if expected_scenario_id is None:
        expected_scenario_id = payload_scenario.get("scenario_id")
    elif payload_scenario.get("scenario_id") != expected_scenario_id:
        raise RuntimeError("C replay reset to a different scenario")
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
        scratch = {
            "agent_f32": np.empty((1, agent_count, binding.AGENT_F32_FIELDS), dtype=np.float32),
            "agent_i32": np.empty((1, agent_count, binding.AGENT_I32_FIELDS), dtype=np.int32),
            "metrics_f32": np.empty((1, agent_count, binding.METRICS_F32_FIELDS), dtype=np.float32),
            "puffer_f32": np.empty((1, agent_count, binding.SCORE_F32_FIELDS), dtype=np.float32),
            "traffic_i16": np.empty((1, traffic_count, binding.TRAFFIC_I16_FIELDS), dtype=np.int16),
        }
        html_frames = {key: [] for key in scratch}
        drive.get_obs_html_frame(*(scratch[key] for key in scratch))
        for key, array in scratch.items():
            html_frames[key].append(array[0].copy())
        if capture_observations:
            html_frames["obs"] = [np.asarray(observations, dtype=np.float32).copy()]
    state_scratch = np.empty((agent_count, STATE_FEATURE_COUNT), dtype=np.float32)
    valid_scratch = np.empty(agent_count, dtype=np.bool_)
    ego_action_scratch = np.empty(REGENTS_EGO_ACTION_FEATURE_COUNT, dtype=np.float32)
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
    if single_scenario_payload(drive.get_state()).get("scenario_id") != expected_scenario_id:
        raise RuntimeError("Drive changed scenario during C replay")
    episode_log = binding.regents_episode_log(drive.c_envs)
    # Split out of the log so the metrics CSV stays numeric; the replay HTML wants the trace.
    avoidability_debug = None if episode_log is None else episode_log.pop("avoidability_debug", None)
    stacked_states = torch.from_numpy(np.ascontiguousarray(np.stack(states))).transpose(0, 1).contiguous()
    stacked_valid = torch.from_numpy(np.ascontiguousarray(np.stack(validity))).transpose(0, 1).contiguous()
    if ego_actions:
        stacked_ego_actions = torch.from_numpy(np.ascontiguousarray(np.stack(ego_actions)))
    else:
        stacked_ego_actions = torch.empty((0, REGENTS_EGO_ACTION_FEATURE_COUNT), dtype=torch.float32)
    return _CRollout(
        states=stacked_states,
        state_valid=stacked_valid,
        ego_actions=stacked_ego_actions,
        collision_pairs=tuple(collision_pairs),
        offroad=offroad,
        scenario_payload=payload_scenario,
        html_frames=None
        if html_frames is None
        else {key: np.stack(frames, axis=0) for key, frames in html_frames.items()},
        episode_log=episode_log,
        avoidability_debug=avoidability_debug,
    )


def capture_plan_pair(drive, baseline_plan, optimized_plan, action_mask, **capture_options):
    """Capture the baseline and adversarial C rollouts of one action plan pair.

    The baseline drives the same actors from their reconstructed logged actions, so
    only events the optimization introduced are attributed to the adversary. The plan
    is always cleared, leaving the env as clean as it was handed over.
    """
    rollouts = []
    try:
        for plan in (baseline_plan, optimized_plan):
            binding.regents_set_action_plan(drive.c_envs, plan, action_mask)
            rollouts.append(_capture_c_rollout(drive, **capture_options))
    finally:
        binding.regents_set_action_plan(drive.c_envs, baseline_plan, np.zeros_like(action_mask))
    return rollouts


def capture_frozen_ego_trajectory(
    drive,
    transition_count,
    *,
    seed=None,
    raster_resolution_meters=DEFAULT_RASTER_RESOLUTION_METERS,
    ego_action_fn=None,
):
    """Reset one Drive scenario and capture its C ego rollout under any ego controller.

    Backgrounds remain under the Drive configuration's replay controller. The
    caller owns the Drive instance and remains responsible for closing it.
    """
    if not isinstance(transition_count, int) or transition_count < 1:
        raise ValueError("transition_count must be a positive integer")
    if drive.sdc_controller not in SUPPORTED_SDC_CONTROLLERS:
        raise ValueError(f"Frozen ego capture requires sdc_controller={SUPPORTED_SDC_CONTROLLER_NAMES}")
    if (drive.sdc_controller == binding.CONTROLLER_POLICY) != (ego_action_fn is not None):
        raise ValueError("A policy ego requires an ego action provider, and no other controller accepts one")
    if drive.non_sdc_controller != binding.CONTROLLER_REPLAY:
        raise ValueError("Frozen ego capture requires non_sdc_controller='replay'")
    if drive.simulation_mode != binding.SIMULATION_MODE_REPLAY:
        raise ValueError("Frozen ego capture requires simulation_mode='replay'")
    if drive.num_envs != 1:
        raise ValueError("Stage 5 frozen ego capture supports exactly one scenario")
    if drive.resample_frequency > 0 and transition_count >= drive.resample_frequency:
        raise ValueError("transition_count must end before Drive resamples the scenario")

    rollout = _capture_c_rollout(drive, transition_count, seed=seed, ego_action_fn=ego_action_fn)
    scenario = export_drive_scenarios(
        drive,
        payload=rollout.scenario_payload,
        raster_resolution_meters=raster_resolution_meters,
    )
    # A backstop: the logged horizon is only known once the export has read the payload,
    # so the callers that can bound the horizon up front already do.
    if transition_count > scenario.max_time_count - 1:
        raise ValueError("transition_count exceeds the exported scenario horizon")
    ego = rollout.scenario_payload["agents"][STABLE_EGO_AGENT_IDX]
    if int(ego.get("controller", -1)) != drive.sdc_controller:
        raise ValueError(f"Stable agent zero must be controlled by {drive.sdc_controller}")
    frozen_ego = FrozenEgoTrajectory(
        state=rollout.states[STABLE_EGO_AGENT_IDX].clone(),
        valid=rollout.state_valid[STABLE_EGO_AGENT_IDX].clone(),
        scenario_id=scenario.scenario_id,
        source=ego_trajectory_source(drive.sdc_controller),
    )
    return scenario, frozen_ego


def _unconfirmed_collision_reason(ego_collision, baseline_ego_collision):
    """Name why C attributed no ego/adversary collision to the optimization.

    Torch found a collision and parity held, so the three outcomes below are the only
    ways the authoritative replay can still decline to credit an injected adversary.
    They are reported separately because they call for different work: a pre-existing
    baseline contact is an ego-quality problem, an ego collision with an unperturbed
    actor is an attribution problem, and neither is the optimization failing to converge.
    """
    if ego_collision:
        return "ego_collision_with_other_agent"
    if baseline_ego_collision:
        return "ego_collision_present_in_baseline"
    return "no_ego_collision_in_replay"


def baseline_relative_events(baseline, adversarial, injected_agent_mask):
    """Attribute C events to the optimization by subtracting the baseline rollout's.

    This is the single definition of ego / actionable / background collision and
    introduced off-road; generation replay and artifact-set evaluation share it so the
    two can never disagree about what an event means. Any injected actor is an
    adversary, so a new ego collision with any such actor is actionable.
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
            actionable_collision |= bool(injected_agent_mask[other_idx])
    introduced_offroad = adversarial.offroad & ~baseline.offroad
    return BaselineRelativeEvents(
        ego_collision=ego_collision,
        actionable_collision=actionable_collision,
        background_collision=background_collision,
        offroad=bool(introduced_offroad[injected_agent_mask].any()),
        first_collision_timestep=first_collision_timestep,
        first_collision_pair=first_collision_pair,
        first_ego_collision_timestep=first_ego_collision_timestep,
        baseline_ego_collision=baseline_ego_collision,
        baseline_collision_pairs=frozenset(baseline_pairs),
    )


def _parity_feature_mask(optimization, adversarial, joint_valid, timestep_count):
    """Flag the state features C and Torch are required to agree on.

    The initial state is a shared input rather than a result, C stores an adversary's
    speed and wheel steering only once injection has integrated it, and once C reports
    a contact the two integrators are no longer describing the same scene.
    """
    injected_state_mask = torch.zeros_like(joint_valid)
    injected_state_mask[:, 1:] = optimization.optimized_action_mask.detach().cpu()
    feature_mask = joint_valid[..., None].expand(*joint_valid.shape, STATE_FEATURE_COUNT).clone()
    feature_mask[..., STATE_SPEED] &= injected_state_mask
    feature_mask[..., STATE_STEERING] &= injected_state_mask
    timestep_index = torch.arange(timestep_count)[None, :, None]
    feature_mask &= timestep_index > 0
    if adversarial.collision_pairs:
        feature_mask &= timestep_index <= adversarial.collision_pairs[0][0]
    return feature_mask


def _validate_replay_inputs(drive, tolerance):
    """Check the Drive instance is configured for the replay; its flags come from config."""
    if drive.num_envs != 1 or drive.simulation_mode != binding.SIMULATION_MODE_REPLAY:
        raise ValueError("Stage 6 requires one Drive environment in replay mode")
    if drive._action_type_flag != binding.ACTION_TYPE_CONTINUOUS:
        raise ValueError("Stage 6 requires continuous actions")
    # Injected adversaries integrate classic dynamics whatever the ego's model is.
    if drive.dynamics_model_flag not in (binding.DYNAMICS_MODEL_CLASSIC, binding.DYNAMICS_MODEL_JERK):
        raise ValueError("Stage 6 requires dynamics_model='classic' or 'jerk'")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")


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
    verified_parity_metrics=None,
):
    """Replay optimized controls in C; optionally reuse a full-horizon parity check."""
    _validate_replay_inputs(drive, tolerance)
    if not isinstance(capture_html_frames, bool):
        raise TypeError("capture_html_frames must be a boolean")
    if verified_parity_metrics is not None and not isinstance(verified_parity_metrics, CReplayMetrics):
        raise TypeError("verified_parity_metrics must be CReplayMetrics")
    transition_count = optimization.optimized_actions.shape[1]
    if drive.resample_frequency > 0 and transition_count >= drive.resample_frequency:
        raise ValueError("C replay horizon must end before Drive resamples")

    scenario_id = scenario.scenario_id
    agent_count = scenario.max_agent_count
    action_mask = np.ascontiguousarray(optimization.optimized_action_mask.detach().cpu().numpy(), dtype=np.bool_)
    baseline_plan = np.ascontiguousarray(optimization.initial_actions.detach().cpu().numpy(), dtype=np.float32)
    optimized_plan = np.ascontiguousarray(optimization.optimized_actions.detach().cpu().numpy(), dtype=np.float32)

    baseline, adversarial = capture_plan_pair(
        drive,
        baseline_plan,
        optimized_plan,
        action_mask,
        transition_count=transition_count,
        expected_scenario_id=scenario_id,
        agent_count=agent_count,
        seed=seed,
        capture_html_frames=capture_html_frames,
        ego_action_fn=ego_action_fn,
        capture_observations=capture_observations,
    )

    torch_states = optimization.optimized_states.detach().cpu()
    injected_agent_mask = optimization.optimized_action_mask.any(dim=1).detach().cpu()
    ego_mask = scenario.ego_mask.detach().cpu()
    joint_valid = optimization.state_valid.detach().cpu() & adversarial.state_valid
    initial_difference = torch.abs(
        adversarial.states[:, 0, : STATE_HEADING + 1] - torch_states[:, 0, : STATE_HEADING + 1]
    )
    initial_difference = initial_difference[joint_valid[:, 0]]
    if initial_difference.numel() and float(initial_difference.max().item()) > INITIAL_STATE_TOLERANCE:
        raise RuntimeError("C reset to a different initial pose than the Torch scenario")

    if verified_parity_metrics is None:
        feature_mask = _parity_feature_mask(optimization, adversarial, joint_valid, torch_states.shape[1])
        differences = torch.abs(adversarial.states - torch_states)
        ego_feature_mask = feature_mask & ego_mask[:, None, None]
        maximum_ego_reference_error = (
            float(differences[ego_feature_mask].max().item()) if ego_feature_mask.any() else 0.0
        )
        # A reactive ego answers the new adversary, so it is reported instead of gated.
        if drive.sdc_controller != binding.CONTROLLER_REPLAY:
            feature_mask &= ~ego_mask[:, None, None]
        maximum_error = float(differences[feature_mask].max().item()) if feature_mask.any() else 0.0
        compared_state_count = int(feature_mask.sum().item())
    else:
        # Stop mode deliberately diverges after contact; the same plan was already
        # checked against Torch with infractions ignored before this final C replay.
        maximum_error = verified_parity_metrics.maximum_trajectory_error
        maximum_ego_reference_error = verified_parity_metrics.maximum_ego_reference_error
        compared_state_count = verified_parity_metrics.compared_state_count

    events = baseline_relative_events(
        baseline,
        adversarial,
        injected_agent_mask.numpy(),
    )

    failure_reason = None
    if maximum_error > tolerance:
        failure_reason = "c_torch_trajectory_mismatch"
    elif not optimization.success:
        failure_reason = optimization.failure_reason or "torch_optimization_failed"
    elif not events.actionable_collision:
        failure_reason = _unconfirmed_collision_reason(events.ego_collision, events.baseline_ego_collision)
    metrics = CReplayMetrics(
        maximum_trajectory_error=maximum_error,
        maximum_ego_reference_error=maximum_ego_reference_error,
        first_collision_timestep=events.first_collision_timestep,
        first_collision_pair=events.first_collision_pair,
        first_ego_collision_timestep=events.first_ego_collision_timestep,
        ego_collision=events.ego_collision,
        actionable_collision=events.actionable_collision,
        background_collision=events.background_collision,
        offroad=events.offroad,
        baseline_ego_collision=events.baseline_ego_collision,
        baseline_collision_pair_count=len(events.baseline_collision_pairs),
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
        # The payload rides every capture, but only the frames it accompanies make it
        # worth retaining, so the default path still hands back nothing to hold.
        scenario_payload=adversarial.scenario_payload if capture_html_frames else None,
        baseline_frames=baseline.html_frames,
        adversarial_frames=adversarial.html_frames,
        episode_log=adversarial.episode_log,
        avoidability_debug=adversarial.avoidability_debug,
        baseline_avoidability_debug=baseline.avoidability_debug,
    )


def make_c_ego_rollout(drive, scenario, horizon_transition_count, seed, ego_action_fn=None):
    """Build the callback that re-rolls the C ego against a candidate action plan.

    Drive is the authority on how the ego answers an adversary, so a refresh is a real C
    rollout with the plan installed. The plan is cleared afterwards so the verification
    replay starts from the same clean env the optimizer's caller handed over.
    """
    scenario_id = scenario.scenario_id
    agent_count = scenario.max_agent_count
    ego_idx = int(torch.where(scenario.ego_mask[0])[0].item())
    source = ego_trajectory_source(drive.sdc_controller)

    def ego_rollout(drive_actions, action_mask):
        plan = np.ascontiguousarray(drive_actions.detach().cpu().numpy(), dtype=np.float32)
        mask = np.ascontiguousarray(action_mask.detach().cpu().numpy(), dtype=np.bool_)
        try:
            binding.regents_set_action_plan(drive.c_envs, plan, mask)
            rollout = _capture_c_rollout(
                drive,
                horizon_transition_count,
                expected_scenario_id=scenario_id,
                agent_count=agent_count,
                seed=seed,
                ego_action_fn=ego_action_fn,
            )
        finally:
            binding.regents_set_action_plan(drive.c_envs, plan, np.zeros_like(mask))
        return FrozenEgoTrajectory(
            state=rollout.states[ego_idx].detach().clone(),
            valid=rollout.state_valid[ego_idx].detach().clone(),
            scenario_id=scenario.scenario_id,
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
    verification_drive=None,
    verification_ego_action_fn=None,
):
    """Optimize one scenario against a periodically re-rolled C ego, then verify in C.

    The ego reacts inside the optimization loop, on `ego_refresh_interval`; C stays the
    success oracle. With a stopped `verification_drive`, C checks promising iterates
    inside the Adam loop and allows bounded retries after an unconfirmed collision.
    `ego_action_fn` supplies actions for a policy ego and must be absent for a native
    C controller.
    """
    if drive.sdc_controller not in SUPPORTED_SDC_CONTROLLERS:
        raise ValueError(f"Reactive generation requires sdc_controller={SUPPORTED_SDC_CONTROLLER_NAMES}")
    if verification_drive is not None and (
        verification_drive.collision_behavior != binding.INFRACTION_BEHAVIOR_STOP
        or verification_drive.offroad_behavior != binding.INFRACTION_BEHAVIOR_STOP
    ):
        raise ValueError("verification_drive must stop on collisions and off-road events")
    if verification_drive is not None and verification_drive.sdc_controller != drive.sdc_controller:
        raise ValueError("verification_drive must use the same ego controller")
    if (
        verification_drive is not None
        and drive.sdc_controller == binding.CONTROLLER_POLICY
        and verification_ego_action_fn is None
    ):
        raise ValueError("policy verification requires a verification ego action provider")
    if optimization_config is None:
        optimization_config = ReGentSOptimizationConfig()

    scenario, frozen_ego = capture_frozen_ego_trajectory(
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

    verification_fn = None
    checked_actions = None
    checked_parity = None
    checked_stopped = None
    if verification_drive is not None:

        def verify_candidate(initial_actions, optimized_actions, optimized_states, state_valid, optimized_action_mask):
            nonlocal checked_actions, checked_parity, checked_stopped
            candidate = CVerificationCandidate(
                initial_actions=initial_actions,
                optimized_actions=optimized_actions,
                optimized_states=optimized_states,
                state_valid=state_valid,
                optimized_action_mask=optimized_action_mask,
            )
            parity_replay = replay_optimized_scenario_in_c(
                drive,
                scenario,
                candidate,
                seed=deterministic_seed,
                tolerance=tolerance,
                ego_action_fn=ego_action_fn,
            )
            checked_actions = optimized_actions
            checked_parity = parity_replay
            checked_stopped = None
            if parity_replay.failure_reason == "c_torch_trajectory_mismatch":
                return VERIFICATION_REJECT
            stopped_replay = replay_optimized_scenario_in_c(
                verification_drive,
                scenario,
                candidate,
                seed=deterministic_seed,
                tolerance=tolerance,
                ego_action_fn=verification_ego_action_fn,
                verified_parity_metrics=parity_replay.metrics,
            )
            checked_stopped = stopped_replay
            if stopped_replay.success:
                return VERIFICATION_ACCEPT
            if stopped_replay.failure_reason in ("no_ego_collision_in_replay", "ego_collision_with_other_agent"):
                return VERIFICATION_RETRY
            return VERIFICATION_REJECT

        verification_fn = verify_candidate

    optimization = optimize_frozen_ego_scenario(
        scenario,
        frozen_ego,
        optimization_config,
        deterministic_seed=deterministic_seed,
        horizon_transition_count=horizon_transition_count,
        show_progress=show_progress,
        inverse_dynamics=inverse_dynamics,
        ego_rollout_fn=ego_rollout_fn,
        verification_fn=verification_fn,
    )
    # The checked iterate is the returned iterate when Adam stopped at that C decision.
    checked_final_actions = checked_actions is optimization.optimized_actions
    replay = (
        checked_parity
        if checked_final_actions
        else replay_optimized_scenario_in_c(
            drive,
            scenario,
            optimization,
            seed=deterministic_seed,
            tolerance=tolerance,
            capture_html_frames=capture_html_frames and verification_drive is None,
            ego_action_fn=ego_action_fn,
            capture_observations=capture_observations and verification_drive is None,
        )
    )
    if verification_drive is not None:
        if (
            checked_final_actions
            and checked_stopped is not None
            and not capture_html_frames
            and not capture_observations
        ):
            replay = checked_stopped
        else:
            replay = replay_optimized_scenario_in_c(
                verification_drive,
                scenario,
                optimization,
                seed=deterministic_seed,
                tolerance=tolerance,
                capture_html_frames=capture_html_frames,
                ego_action_fn=verification_ego_action_fn,
                capture_observations=capture_observations,
                verified_parity_metrics=replay.metrics,
            )
    return ReactiveGenerationResult(
        scenario=scenario,
        optimization=optimization,
        replay=replay,
        ego_refresh_count=optimization.ego_refresh_count,
        deterministic_seed=deterministic_seed,
        ego_source=frozen_ego.source,
    )
