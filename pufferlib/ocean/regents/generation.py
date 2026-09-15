"""Offline ReGentS generation entry point over a fixed scenario range."""

import csv
import inspect
import math
import multiprocessing
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from tqdm import tqdm

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark
from pufferlib.ocean.regents.adapter import DEFAULT_RASTER_RESOLUTION_METERS
from pufferlib.ocean.regents.dynamics import ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED
from pufferlib.ocean.regents.artifacts import cost_row, save_generation_artifact
from pufferlib.ocean.regents.filters import ReGentSFilterConfig
from pufferlib.ocean.regents.losses import ReGentSCostConfig
from pufferlib.ocean.regents.optimizer import ReGentSOptimizationConfig
from pufferlib.ocean.regents.policy_ego import PolicyEgoActor, ReGentSPolicyEgoConfig, checkpoint_digest
from pufferlib.ocean.regents.rollout import run_reactive_generation


METRICS_FILE_NAME = "generation_metrics.csv"
RENDER_DIR_NAME = "rendered_replays"
REGENTS_ACTIVE_AGENT_COUNT = 1
STANDARD_EVAL_INFRACTION_BEHAVIOR = "stop"
# `puffer eval` reports one Log row per episode; the ReGentS horizon closes the same row and
# carries it here under a prefix, so these never collide with the columns above.
EVAL_METRIC_FIELD_PREFIX = "eval_"
# Each closed episode carries exactly one of these as 1.0 when its ego collision was analyzed.
TARGET_COLLISION_CLASS_FIELDS = (
    ("genuine_failure", "sdc_target_collision_genuine_failure_rate"),
    ("adversary_forced", "sdc_target_collision_adversary_forced_rate"),
    ("unavoidable", "sdc_target_collision_unavoidable_rate"),
)


@dataclass(frozen=True)
class GenerationReport:
    output_dir: Path
    replay_index: Path | None
    scenario_count: int
    candidate_scenario_count: int
    filtered_no_candidate_count: int
    generation_success_count: int
    torch_collision_count: int
    c_confirmed_actionable_collision_count: int
    c_unconfirmed_actionable_collision_count: int
    generation_success_rate: float
    candidate_success_rate: float
    torch_collision_rate: float
    c_collision_confirmation_rate: float
    ego_collision_rate: float
    actionable_collision_rate: float
    background_collision_rate: float
    offroad_rate: float
    maximum_c_torch_trajectory_error: float
    total_optimization_seconds: float
    wall_clock_seconds: float
    rejection_reasons: dict
    target_collision_class_counts: dict
    successful_target_collision_class_counts: dict


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_positive_int(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_experiment_name(experiment_name):
    if experiment_name is None:
        return None
    if not isinstance(experiment_name, str):
        raise TypeError("ReGentS experiment name must be a string")
    experiment_name = experiment_name.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", experiment_name):
        raise ValueError(
            "ReGentS experiment name must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'"
        )
    return experiment_name


def _apply_generation_overrides(generation, experiment_name, drivable_area_weight, scenario_count=None):
    """Apply validated CLI overrides to the configuration persisted in artifacts."""
    resolved = dict(generation)
    resolved["experiment_name"] = _validate_experiment_name(experiment_name)
    if scenario_count is not None:
        resolved["scenario_count"] = _require_positive_int(scenario_count, "scenario_count")
        resolved["env"] = {**resolved["env"], "num_maps": scenario_count}
    if drivable_area_weight is None:
        return resolved
    if isinstance(drivable_area_weight, bool) or not isinstance(drivable_area_weight, (int, float)):
        raise TypeError("ReGentS drivable-area weight must be a number")
    drivable_area_weight = float(drivable_area_weight)
    if not math.isfinite(drivable_area_weight) or drivable_area_weight < 0.0:
        raise ValueError("ReGentS drivable-area weight must be finite and non-negative")
    optimizer = dict(resolved["optimizer"])
    costs = dict(_require_mapping(optimizer.get("costs", {}), "optimizer costs"))
    costs["drivable_area_weight"] = drivable_area_weight
    optimizer["costs"] = costs
    resolved["optimizer"] = optimizer
    return resolved


def _resolve_ego_policy(ego_policy, environment, generation_name):
    """Validate the ego policy block against the generation's configured ego controller."""
    requires_policy = environment.get("sdc_controller") == "policy"
    if ego_policy is None:
        if requires_policy:
            raise ValueError(f"Generation {generation_name} uses sdc_controller='policy' but declares no ego_policy")
        return None
    if not requires_policy:
        raise ValueError(f"Generation {generation_name} declares an ego_policy without sdc_controller='policy'")
    settings = _require_mapping(ego_policy, f"Generation {generation_name} ego_policy")
    unknown_keys = set(settings) - set(inspect.signature(ReGentSPolicyEgoConfig).parameters)
    if unknown_keys:
        raise ValueError(f"ReGentS ego_policy has unsupported keys: {', '.join(sorted(unknown_keys))}")
    # Resolved as a plain JSON-safe mapping: it is hashed into the artifact source
    # configuration, and a checkpoint path is not an identity - its bytes are.
    resolved = asdict(ReGentSPolicyEgoConfig(**settings))
    resolved["checkpoint_sha256"] = checkpoint_digest(resolved["checkpoint_path"])
    return resolved


def load_generation_config(config_path, generation_name):
    """Validate the untrusted generation config and resolve one named entry."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config = _require_mapping(config, "ReGentS generation config")
    shared_env = _require_mapping(config.get("env"), "ReGentS generation config env")
    shared_optimizer = _require_mapping(config.get("optimizer", {}), "ReGentS generation config optimizer")
    shared_filter = _require_mapping(shared_optimizer.get("filter", {}), "ReGentS generation config optimizer filter")
    # Filter has explicit key-by-key override semantics. Other shared nested optimizer
    # blocks are rejected because silently replacing them would lose standard values.
    nested_shared_keys = [key for key, value in shared_optimizer.items() if isinstance(value, dict) and key != "filter"]
    if nested_shared_keys:
        raise ValueError(f"ReGentS generation config optimizer must not nest: {', '.join(sorted(nested_shared_keys))}")
    shared_costs = _require_mapping(config.get("costs", {}), "ReGentS generation config costs")
    generations = config.get("generations")
    if not isinstance(generations, list) or not generations:
        raise ValueError("ReGentS generation config must contain a non-empty generations list")
    if not isinstance(generation_name, str) or not generation_name.strip():
        raise ValueError("Generation name must be a non-empty string")
    generation_name = generation_name.strip()

    selected = None
    seen_names = set()
    for entry_idx, entry in enumerate(generations):
        entry = _require_mapping(entry, f"generations[{entry_idx}]")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"generations[{entry_idx}].name must be a non-empty string")
        if name.strip() in seen_names:
            raise ValueError(f"ReGentS generation config contains duplicate name: {name.strip()}")
        seen_names.add(name.strip())
        if name.strip() == generation_name:
            selected = entry
    if selected is None:
        raise ValueError(f"Unknown ReGentS generation: {generation_name}")

    environment = dict(shared_env)
    environment.update(_require_mapping(selected.get("env", {}), f"Generation {generation_name} env"))
    unknown_keys = set(environment) - (set(inspect.signature(Drive.__init__).parameters) - {"self"})
    if unknown_keys:
        raise ValueError(f"ReGentS generation config has unsupported env keys: {', '.join(sorted(unknown_keys))}")

    selected_optimizer = _require_mapping(selected.get("optimizer", {}), f"Generation {generation_name} optimizer")
    optimizer = dict(shared_optimizer)
    optimizer.update(selected_optimizer)
    filter_settings = dict(shared_filter)
    filter_settings.update(
        _require_mapping(selected_optimizer.get("filter", {}), f"Generation {generation_name} optimizer filter")
    )
    if filter_settings:
        optimizer["filter"] = filter_settings
    # Shared cost geometry, overridden key by key by a generation that declares its own.
    costs = dict(shared_costs)
    costs.update(_require_mapping(optimizer.get("costs", {}), f"Generation {generation_name} optimizer costs"))
    if costs:
        optimizer["costs"] = costs
    ego_policy = _resolve_ego_policy(selected.get("ego_policy"), environment, generation_name)
    num_workers = selected.get("num_workers", 1)
    if num_workers != "auto":
        _require_positive_int(num_workers, "num_workers")

    resolved = {
        "name": generation_name,
        "seed": _require_positive_int(selected.get("seed"), "seed"),
        "scenario_count": _require_positive_int(selected.get("scenario_count"), "scenario_count"),
        "num_workers": num_workers,
        "horizon_transition_count": _require_positive_int(
            selected.get("horizon_transition_count"), "horizon_transition_count"
        ),
        "raster_resolution_meters": float(
            selected.get(
                "raster_resolution_meters",
                config.get("raster_resolution_meters", DEFAULT_RASTER_RESOLUTION_METERS),
            )
        ),
        "output_dir": str(selected.get("output_dir", "experiments/regents")),
        "render_replays": bool(selected.get("render_replays", False)),
        "capture_observations": bool(selected.get("capture_observations", False)),
        "optimizer": optimizer,
        "ego_policy": ego_policy,
        "env": environment,
    }
    # Observations ride along with the HTML frames, so there is nowhere to put them
    # when replays are not being rendered.
    if resolved["capture_observations"] and not resolved["render_replays"]:
        raise ValueError(f"Generation {generation_name} sets capture_observations without render_replays")
    raster_resolution_meters = resolved["raster_resolution_meters"]
    if not math.isfinite(raster_resolution_meters) or raster_resolution_meters <= 0.0:
        raise ValueError("raster_resolution_meters must be finite and positive")
    horizon = resolved["horizon_transition_count"]
    for name in ("resample_frequency", "scenario_length"):
        limit = environment.get(name)
        if limit is not None and int(limit) > 0 and horizon >= int(limit):
            raise ValueError(
                f"Generation {generation_name}: horizon_transition_count={horizon} must be below "
                f"env.{name}={int(limit)}. Set env.{name} to at least {horizon + 1} for this generation."
            )
    # Generation indexes one map per scenario, so the map budget is the scenario count.
    # Deriving it here keeps the two from drifting apart in the config.
    environment["num_maps"] = resolved["scenario_count"]
    return resolved


def _full_env_config(environment):
    """Complete the Drive keyword set so viz.py can read every field it expects."""
    signature = inspect.signature(Drive.__init__)
    config = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }
    config.update(environment)
    return config


def _replay_bundle(env_config, frames, ego_actions):
    frames = dict(frames)
    observations = frames.pop("obs", None)
    frame_count = frames["agent_f32"].shape[0]
    expected_ego_action_shape = (frame_count - 1, 2)
    if tuple(ego_actions.shape) != expected_ego_action_shape:
        raise ValueError(f"Captured ReGentS ego actions must have shape {expected_ego_action_shape}")
    actions = np.zeros((frame_count, REGENTS_ACTIVE_AGENT_COUNT, 2), dtype=np.float32)
    actions[1:, 0] = ego_actions.detach().cpu().numpy()
    bundle = {"env": env_config, **frames, "raw_action": actions, "clipped_action": actions}
    if observations is not None:
        if observations.ndim != 3 or observations.shape[:2] != (frame_count, REGENTS_ACTIVE_AGENT_COUNT):
            raise ValueError("Captured ReGentS observations must have shape [frame, one active ego, feature]")
        bundle["obs"] = observations
    return bundle


def save_loss_history_csv(destination, scenario_idx, result):
    """Write the optimization loss history to a CSV file per scenario/map.

    The columns are the cost snapshot's own fields, so the loss history and the
    artifact metadata always describe an iterate the same way.
    """
    cost_history = result.optimization.cost_history
    if not cost_history:
        return
    losses_dir = Path(destination) / "losses"
    losses_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"iteration": iteration, **cost_row(snapshot, result.scenario)}
        for iteration, snapshot in enumerate(cost_history)
    ]
    with (losses_dir / f"scenario_{scenario_idx:05d}.losses.csv").open("w", newline="", encoding="utf-8") as loss_file:
        writer = csv.DictWriter(loss_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_plan(actions, candidate_rows, transition_count):
    """Return the candidate rows of an action plan, cut to the rendered transitions."""
    return actions[candidate_rows][:, :transition_count].detach().cpu().numpy()


def render_scenario_replays(destination, scenario_idx, result, env_config):
    """Write logged and adversarial replays as interactive HTML with the shared viewer."""
    import pufferlib.viz

    replay = result.replay
    if replay.scenario_payload is None or replay.adversarial_frames is None:
        raise ValueError("Replay rendering requires capture_html_frames")
    render_dir = Path(destination) / RENDER_DIR_NAME
    render_dir.mkdir(parents=True, exist_ok=True)
    replays_dir = Path(destination) / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)
    rendered = {}
    sources = (
        ("logged", replay.baseline_frames, replay.baseline_avoidability_debug),
        ("adversarial", replay.adversarial_frames, replay.avoidability_debug),
    )
    candidate_rows = result.optimization.selection.candidate_mask
    candidate_adversary_ids = result.scenario.agent_id[candidate_rows].tolist()
    for label, frames, avoidability_debug in sources:
        stem = f"scenario_{scenario_idx:05d}.{label}"
        optimization = result.optimization
        transition_count = replay.ego_actions.shape[0]
        bundle = {
            **_replay_bundle(env_config, frames, replay.ego_actions),
            # Each page carries the counterfactual recorded for its own rollout.
            "avoidability_debug": avoidability_debug,
            "candidate_adversary_ids": candidate_adversary_ids,
            # The optimizer's plan for every candidate, so the viewer can show the acceleration
            # the log started from next to the one Adam ended on, frame-aligned with the replay.
            "adversary_plan_ids": candidate_adversary_ids,
            "adversary_plan_initial": candidate_plan(optimization.initial_actions, candidate_rows, transition_count),
            "adversary_plan_optimized": candidate_plan(
                optimization.optimized_actions, candidate_rows, transition_count
            ),
            "adversary_plan_acceleration_scale": ACCELERATION_SCALE_METERS_PER_SECOND_SQUARED,
            "selected_adversary_idx": optimization.selected_adversary_idx,
            "selected_adversary_id": optimization.selected_adversary_id,
            "ego_collision_loss_adversary_idx": optimization.ego_collision_loss_adversary_idx,
            "ego_collision_loss_adversary_id": optimization.ego_collision_loss_adversary_id,
            "optimization_update_count": optimization.iteration_count,
            "optimization_ego_collision": optimization.success,
            "ego_refresh_count": optimization.ego_refresh_count,
        }
        binary_path = replays_dir / f"{stem}.replay.zlib"
        html_path = render_dir / f"{stem}.html"
        pufferlib.viz.save_interactive_replay_zlib(replay.scenario_payload, bundle, str(binary_path))
        pufferlib.viz.render_interactive_replay_zlib(str(binary_path), str(html_path))
        rendered[html_path.name] = {
            "scenario_index": scenario_idx,
            "no_candidate": float(not candidate_adversary_ids),
            "iteration_limit": float(replay.failure_reason == "iteration_limit"),
            "failure_reason": replay.failure_reason or "",
            "collision": float(replay.metrics.ego_collision and label == "adversarial"),
            "offroad": float(replay.metrics.offroad and label == "adversarial"),
            **{
                name: float(label == "adversarial" and (replay.episode_log or {}).get(field, 0.0) > 0.0)
                for name, field in TARGET_COLLISION_CLASS_FIELDS
            },
            "generation_success": float(replay.success),
            "idm_reconstruction_collision": float(
                result.optimization.frozen_ego_source == "c_idm" and replay.metrics.baseline_ego_collision
            ),
        }
    return rendered


def _optimization_config(optimizer_config):
    settings = dict(optimizer_config)
    costs = _require_mapping(settings.pop("costs", {}), "optimizer costs")
    filters = _require_mapping(settings.pop("filter", {}), "optimizer filter")
    unknown_keys = set(settings) - set(inspect.signature(ReGentSOptimizationConfig).parameters)
    if unknown_keys:
        raise ValueError(f"ReGentS generation config has unsupported optimizer keys: {', '.join(sorted(unknown_keys))}")
    return ReGentSOptimizationConfig(
        costs=ReGentSCostConfig(**costs),
        filter=ReGentSFilterConfig(**filters),
        **settings,
    )


def _metric_row(scenario_idx, seed, result, elapsed_seconds, artifact_path, ego_controller):
    """One generation_metrics.csv row. Generation indexes one map per scenario."""
    metrics = result.replay.metrics
    return {
        "scenario_index": scenario_idx,
        "scenario_id": result.scenario.scenario_id,
        "ego_controller": ego_controller,
        "map_index": scenario_idx,
        "seed": seed,
        "candidate_count": int(result.optimization.selection.candidate_mask.sum().item()),
        "torch_collision": int(result.optimization.success),
        "generation_success": int(result.replay.success),
        "ego_collision": int(metrics.ego_collision),
        "actionable_collision": int(metrics.actionable_collision),
        "background_collision": int(metrics.background_collision),
        "baseline_ego_collision": int(metrics.baseline_ego_collision),
        "baseline_background_collision_pair_count": result.optimization.baseline_background_collision_pair_count,
        "background_collision_rejection_count": result.optimization.background_collision_rejection_count,
        "offroad": int(metrics.offroad),
        "c_torch_trajectory_error": metrics.maximum_trajectory_error,
        "ego_reference_error": metrics.maximum_ego_reference_error,
        "torch_collision_timestep": result.optimization.collision_timestep,
        "collision_timestep": metrics.first_collision_timestep,
        "selected_adversary_idx": result.optimization.selected_adversary_idx,
        "selected_adversary_id": result.optimization.selected_adversary_id,
        "ego_refresh_count": result.ego_refresh_count,
        "optimization_seconds": elapsed_seconds,
        "failure_reason": result.replay.failure_reason,
        "artifact_path": str(artifact_path),
        **{f"{EVAL_METRIC_FIELD_PREFIX}{name}": value for name, value in (result.replay.episode_log or {}).items()},
    }


def _generate_scenario(task):
    """Generate, C-verify, and persist one scenario. Runs in-process or spawned."""
    import traceback
    import torch

    scenario_idx = task.scenario_idx
    generation = task.generation
    destination = task.destination
    if task.worker_torch_thread_count is not None:
        torch.set_num_threads(task.worker_torch_thread_count)
    seed = generation["seed"] + scenario_idx
    try:
        drive = Drive(
            **generation["env"],
            eval_map_indices=[scenario_idx],
            eval_scenario_seeds=[seed],
            seed=seed,
        )
        started_at = time.perf_counter()
        verification_drive = None
        try:
            verification_drive = Drive(
                **{
                    **generation["env"],
                    "collision_behavior": STANDARD_EVAL_INFRACTION_BEHAVIOR,
                    "offroad_behavior": STANDARD_EVAL_INFRACTION_BEHAVIOR,
                },
                eval_map_indices=[scenario_idx],
                eval_scenario_seeds=[seed],
                seed=seed,
            )
            ego_policy = None
            if generation["ego_policy"] is not None:
                policy_settings = {
                    key: value for key, value in generation["ego_policy"].items() if key != "checkpoint_sha256"
                }
                ego_policy = ReGentSPolicyEgoConfig(**policy_settings)
            result = run_reactive_generation(
                drive,
                task.optimization_config,
                deterministic_seed=seed,
                horizon_transition_count=generation["horizon_transition_count"],
                show_progress=task.show_progress,
                raster_resolution_meters=generation["raster_resolution_meters"],
                ego_action_fn=None if ego_policy is None else PolicyEgoActor(ego_policy, drive),
                verification_drive=verification_drive,
                verification_ego_action_fn=(
                    None if ego_policy is None else PolicyEgoActor(ego_policy, verification_drive)
                ),
                capture_html_frames=task.render_replays,
                capture_observations=generation["capture_observations"],
            )
        finally:
            drive.close()
            if verification_drive is not None:
                verification_drive.close()
        elapsed_seconds = time.perf_counter() - started_at
        npz_dir = Path(destination) / "npz"
        npz_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = npz_dir / f"scenario_{scenario_idx:05d}.npz"
        save_generation_artifact(artifact_path, result, generation, str(task.map_path))
        save_loss_history_csv(destination, scenario_idx, result)
        row = _metric_row(
            scenario_idx,
            seed,
            result,
            elapsed_seconds,
            artifact_path,
            generation["env"]["sdc_controller"],
        )
        rendered_files = (
            render_scenario_replays(destination, scenario_idx, result, task.env_config) if task.render_replays else {}
        )
        return scenario_idx, row, rendered_files
    except Exception:
        # A spawned worker only ships the exception repr back, so log the real traceback here.
        print(f"\n[ERROR] Generation failed for scenario {scenario_idx}:")
        traceback.print_exc()
        raise


@dataclass(frozen=True)
class _ScenarioTask:
    """One scenario's whole generation input, shipped whole to a spawned worker."""

    scenario_idx: int
    generation: dict
    optimization_config: ReGentSOptimizationConfig
    destination: Path
    map_path: Path
    env_config: dict
    render_replays: bool
    show_progress: bool
    # A spawned worker shares the machine, so it takes one Torch thread; in process, None
    # leaves the caller's own thread count alone.
    worker_torch_thread_count: int | None


def _generation_report(rows, destination, replay_index, wall_clock_seconds):
    """Aggregate the per-scenario rows into the run-level report.

    Rates are reported over two different denominators on purpose: eligibility and the
    infraction rates are per scenario, while success and Torch collision are per
    candidate-bearing scenario, because a scene with no candidate was never optimized.
    """
    rejection_reasons = {}
    for row in rows:
        if row["failure_reason"] is None:
            continue
        rejection_reasons[row["failure_reason"]] = rejection_reasons.get(row["failure_reason"], 0) + 1
    scenario_count = len(rows)
    candidate_scenario_count = sum(row["candidate_count"] > 0 for row in rows)
    generation_success_count = sum(row["generation_success"] for row in rows)
    torch_collision_count = sum(row["torch_collision"] for row in rows)
    c_confirmed_actionable_collision_count = sum(row["torch_collision"] and row["actionable_collision"] for row in rows)

    def rate(count, denominator):
        return count / denominator if denominator else 0.0

    return GenerationReport(
        output_dir=destination,
        replay_index=replay_index,
        scenario_count=scenario_count,
        candidate_scenario_count=candidate_scenario_count,
        filtered_no_candidate_count=scenario_count - candidate_scenario_count,
        generation_success_count=generation_success_count,
        torch_collision_count=torch_collision_count,
        c_confirmed_actionable_collision_count=c_confirmed_actionable_collision_count,
        c_unconfirmed_actionable_collision_count=torch_collision_count - c_confirmed_actionable_collision_count,
        generation_success_rate=rate(generation_success_count, scenario_count),
        candidate_success_rate=rate(generation_success_count, candidate_scenario_count),
        torch_collision_rate=rate(torch_collision_count, candidate_scenario_count),
        c_collision_confirmation_rate=rate(c_confirmed_actionable_collision_count, torch_collision_count),
        ego_collision_rate=rate(sum(row["ego_collision"] for row in rows), scenario_count),
        actionable_collision_rate=rate(sum(row["actionable_collision"] for row in rows), scenario_count),
        background_collision_rate=rate(sum(row["background_collision"] for row in rows), scenario_count),
        offroad_rate=rate(sum(row["offroad"] for row in rows), scenario_count),
        maximum_c_torch_trajectory_error=max(row["c_torch_trajectory_error"] for row in rows),
        total_optimization_seconds=sum(row["optimization_seconds"] for row in rows),
        wall_clock_seconds=wall_clock_seconds,
        rejection_reasons=rejection_reasons,
        target_collision_class_counts={
            name: sum(row.get(f"{EVAL_METRIC_FIELD_PREFIX}{field}", 0.0) > 0.0 for row in rows)
            for name, field in TARGET_COLLISION_CLASS_FIELDS
        },
        successful_target_collision_class_counts={
            name: sum(
                row["generation_success"] and row.get(f"{EVAL_METRIC_FIELD_PREFIX}{field}", 0.0) > 0.0 for row in rows
            )
            for name, field in TARGET_COLLISION_CLASS_FIELDS
        },
    )


def generate_regents_scenarios(
    config_path,
    generation_name,
    output_dir=None,
    *,
    experiment_name=None,
    drivable_area_weight=None,
    scenario_count=None,
):
    """Generate, C-verify, and save one artifact per scenario in the configured range."""
    overall_start = time.perf_counter()
    generation = load_generation_config(config_path, generation_name)
    generation = _apply_generation_overrides(generation, experiment_name, drivable_area_weight, scenario_count)
    destination = Path(output_dir) if output_dir is not None else Path(generation["output_dir"]) / generation_name
    if generation["experiment_name"] is not None:
        destination /= generation["experiment_name"]
    destination.mkdir(parents=True, exist_ok=True)
    optimization_config = _optimization_config(generation["optimizer"])
    map_directory = Path(generation["env"]["map_dir"])
    map_paths = sorted(map_directory.glob("*.bin"))
    if len(map_paths) < generation["scenario_count"]:
        raise ValueError("Configured scenario_count exceeds the available map binaries")

    render_replays = generation["render_replays"]
    env_config = _full_env_config(
        {
            **generation["env"],
            "collision_behavior": STANDARD_EVAL_INFRACTION_BEHAVIOR,
            "offroad_behavior": STANDARD_EVAL_INFRACTION_BEHAVIOR,
        }
    )
    rows = [None] * generation["scenario_count"]
    rendered_files = {}

    num_workers = generation["num_workers"]
    if num_workers == "auto":
        num_workers = os.cpu_count() or 1
    num_workers = min(num_workers, generation["scenario_count"])
    tasks = [
        _ScenarioTask(
            scenario_idx=scenario_idx,
            generation=generation,
            optimization_config=optimization_config,
            destination=destination,
            map_path=map_paths[scenario_idx],
            env_config=env_config,
            render_replays=render_replays,
            show_progress=num_workers <= 1,
            worker_torch_thread_count=1 if num_workers > 1 else None,
        )
        for scenario_idx in range(generation["scenario_count"])
    ]

    if num_workers <= 1:
        pool = None
        completions = map(_generate_scenario, tasks)
        description = "Generating scenarios"
    else:
        pool = multiprocessing.get_context("spawn").Pool(processes=num_workers)
        completions = pool.imap_unordered(_generate_scenario, tasks)
        description = "Generating scenarios (parallel)"

    success_count = 0
    candidate_scenario_count = 0
    completed_count = 0
    try:
        pbar = tqdm(completions, total=len(tasks), desc=description)
        for scenario_idx, row, scenario_rendered in pbar:
            rendered_files.update(scenario_rendered)
            rows[scenario_idx] = row
            completed_count += 1
            success_count += int(row["generation_success"])
            candidate_scenario_count += int(row["candidate_count"] > 0)
            candidate_success_rate = success_count / candidate_scenario_count if candidate_scenario_count else 0.0
            pbar.set_postfix(
                eligible=f"{candidate_scenario_count}/{completed_count}",
                success=f"{candidate_success_rate:.1%}",
            )
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    if rendered_files:
        import pufferlib.viz

        pufferlib.viz.build_gallery_index(str(destination / RENDER_DIR_NAME), file_metrics=rendered_files)

    metrics_path = destination / METRICS_FILE_NAME
    with metrics_path.open("w", encoding="utf-8", newline="") as metrics_file:
        # _metric_row is the single source of the schema: its own key order gives the
        # fixed columns, and the eval columns are the union over rows, since a scenario
        # whose horizon closed no episode carries none.
        eval_field_names = sorted({name for row in rows for name in row if name.startswith(EVAL_METRIC_FIELD_PREFIX)})
        fixed_field_names = [name for name in rows[0] if not name.startswith(EVAL_METRIC_FIELD_PREFIX)]
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[*fixed_field_names, *eval_field_names],
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)

    # Same aggregate `puffer eval` writes: episode_metrics.csv plus evaluation_summary.json
    # of metric means, built from the per-scenario episode logs by the same reducer.
    episode_summaries = [
        {
            name[len(EVAL_METRIC_FIELD_PREFIX) :]: value
            for name, value in row.items()
            if name.startswith(EVAL_METRIC_FIELD_PREFIX)
        }
        for row in rows
    ]
    episode_summaries = [summary for summary in episode_summaries if summary]
    if episode_summaries:
        drive_benchmark._write_eval_reports(episode_summaries, str(destination), len(rows))

    return _generation_report(
        rows,
        destination,
        (destination / RENDER_DIR_NAME / "index.html") if rendered_files else None,
        time.perf_counter() - overall_start,
    )
