"""Offline ReGentS generation entry point over a fixed scenario range."""

import csv
import inspect
import math
import multiprocessing
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from tqdm import tqdm

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.adapter import DEFAULT_RASTER_RESOLUTION_METERS
from pufferlib.ocean.regents.artifacts import save_generation_artifact
from pufferlib.ocean.regents.filters import ReGentSFilterConfig
from pufferlib.ocean.regents.losses import ReGentSCostConfig
from pufferlib.ocean.regents.optimizer import ReGentSOptimizationConfig
from pufferlib.ocean.regents.rollout import run_reactive_idm_generation


METRICS_FILE_NAME = "generation_metrics.csv"
RENDER_DIR_NAME = "rendered_replays"
REGENTS_ACTIVE_AGENT_COUNT = 1
METRIC_FIELD_NAMES = (
    "scenario_index",
    "scenario_id",
    "map_index",
    "seed",
    "candidate_count",
    "torch_collision",
    "generation_success",
    "ego_collision",
    "actionable_collision",
    "background_collision",
    "baseline_background_collision_pair_count",
    "background_collision_rejection_count",
    "offroad",
    "c_torch_trajectory_error",
    "ego_reference_error",
    "torch_collision_timestep",
    "collision_timestep",
    "selected_adversary_idx",
    "selected_adversary_id",
    "outer_iteration_count",
    "optimization_seconds",
    "failure_reason",
    "artifact_path",
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


def _apply_generation_overrides(generation, experiment_name, drivable_area_weight):
    """Apply validated CLI overrides to the configuration persisted in artifacts."""
    resolved = dict(generation)
    resolved["experiment_name"] = _validate_experiment_name(experiment_name)
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


def load_generation_config(config_path, generation_name):
    """Validate the untrusted generation config and resolve one named entry."""
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config = _require_mapping(config, "ReGentS generation config")
    shared_env = _require_mapping(config.get("env"), "ReGentS generation config env")
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

    optimizer = _require_mapping(selected.get("optimizer", {}), f"Generation {generation_name} optimizer")
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
        "maximum_outer_iterations": _require_positive_int(
            selected.get("maximum_outer_iterations"), "maximum_outer_iterations"
        ),
        "raster_resolution_meters": float(selected.get("raster_resolution_meters", DEFAULT_RASTER_RESOLUTION_METERS)),
        "output_dir": str(selected.get("output_dir", "experiments/regents")),
        "render_replays": bool(selected.get("render_replays", False)),
        "optimizer": optimizer,
        "env": environment,
    }
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
    expected_ego_action_shape = (1, frame_count - 1, 2)
    if tuple(ego_actions.shape) != expected_ego_action_shape:
        raise ValueError(f"Captured ReGentS ego actions must have shape {expected_ego_action_shape}")
    actions = np.zeros((frame_count, REGENTS_ACTIVE_AGENT_COUNT, 2), dtype=np.float32)
    actions[1:, 0] = ego_actions[0].detach().cpu().numpy()
    bundle = {"env": env_config, **frames, "raw_action": actions, "clipped_action": actions}
    if observations is not None:
        if observations.ndim != 3 or observations.shape[:2] != (frame_count, REGENTS_ACTIVE_AGENT_COUNT):
            raise ValueError("Captured ReGentS observations must have shape [frame, one active ego, feature]")
        bundle["obs"] = observations
    return bundle


def save_loss_history_csv(destination, scenario_idx, result):
    """Write the optimization loss history to a CSV file per scenario/map."""
    optimization = result.optimization
    if not optimization.cost_history:
        return
    losses_dir = Path(destination) / "losses"
    losses_dir.mkdir(parents=True, exist_ok=True)
    csv_path = losses_dir / f"scenario_{scenario_idx:05d}.losses.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "total_loss",
                "ego_collision_cost",
                "background_collision_cost",
                "drivable_area_cost",
                "background_collision_first_agent_idx",
                "background_collision_first_agent_id",
                "background_collision_second_agent_idx",
                "background_collision_second_agent_id",
                "background_collision_timestep_idx",
                "background_collision_signed_distance_meters",
                "background_collision_truncated",
            ]
        )
        for idx, snap in enumerate(optimization.cost_history):
            first_agent_id = (
                int(result.scenario.agent_id[0, snap.background_collision_first_agent_idx].item())
                if snap.background_collision_first_agent_idx >= 0
                else -1
            )
            second_agent_id = (
                int(result.scenario.agent_id[0, snap.background_collision_second_agent_idx].item())
                if snap.background_collision_second_agent_idx >= 0
                else -1
            )
            writer.writerow(
                [
                    idx,
                    snap.total,
                    snap.ego_collision,
                    snap.background_collision,
                    snap.drivable_area,
                    snap.background_collision_first_agent_idx,
                    first_agent_id,
                    snap.background_collision_second_agent_idx,
                    second_agent_id,
                    snap.background_collision_timestep_idx,
                    snap.background_collision_signed_distance_meters,
                    int(snap.background_collision_truncated),
                ]
            )


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
    sources = (("logged", replay.baseline_frames), ("adversarial", replay.adversarial_frames))
    candidate_adversary_ids = result.scenario.agent_id[0][result.optimization.selection.candidate_mask[0]].tolist()
    for label, frames in sources:
        stem = f"scenario_{scenario_idx:05d}.{label}"
        bundle = _replay_bundle(env_config, frames, replay.ego_actions)
        bundle["candidate_adversary_ids"] = candidate_adversary_ids
        bundle["selected_adversary_idx"] = result.optimization.selected_adversary_idx
        bundle["selected_adversary_id"] = result.optimization.selected_adversary_id
        bundle["ego_collision_loss_adversary_idx"] = result.optimization.ego_collision_loss_adversary_idx
        bundle["ego_collision_loss_adversary_id"] = result.optimization.ego_collision_loss_adversary_id
        binary_path = replays_dir / f"{stem}.replay.zlib"
        html_path = render_dir / f"{stem}.html"
        pufferlib.viz.save_interactive_replay_zlib(replay.scenario_payload, bundle, str(binary_path))
        pufferlib.viz.render_interactive_replay_zlib(str(binary_path), str(html_path))
        rendered[html_path.name] = {
            "scenario_index": scenario_idx,
            "collision": float(replay.metrics.ego_collision and label == "adversarial"),
            "offroad": float(replay.metrics.offroad and label == "adversarial"),
            "generation_success": float(replay.success),
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


def _build_drive(environment, map_idx, seed):
    return Drive(**environment, eval_map_indices=[map_idx], eval_scenario_seeds=[seed], seed=seed)


def _metric_row(scenario_idx, map_idx, seed, result, elapsed_seconds, artifact_path):
    metrics = result.replay.metrics
    return {
        "scenario_index": scenario_idx,
        "scenario_id": result.scenario.scenario_ids[0],
        "map_index": map_idx,
        "seed": seed,
        "candidate_count": int(result.optimization.selection.candidate_mask[0].sum().item()),
        "torch_collision": int(result.optimization.success),
        "generation_success": int(result.replay.success),
        "ego_collision": int(metrics.ego_collision),
        "actionable_collision": int(metrics.actionable_collision),
        "background_collision": int(metrics.background_collision),
        "baseline_background_collision_pair_count": result.optimization.baseline_background_collision_pair_count,
        "background_collision_rejection_count": result.optimization.background_collision_rejection_count,
        "offroad": int(metrics.offroad),
        "c_torch_trajectory_error": metrics.maximum_trajectory_error,
        "ego_reference_error": metrics.maximum_ego_reference_error,
        "torch_collision_timestep": result.optimization.collision_timestep,
        "collision_timestep": metrics.first_collision_timestep,
        "selected_adversary_idx": result.optimization.selected_adversary_idx,
        "selected_adversary_id": result.optimization.selected_adversary_id,
        "outer_iteration_count": result.outer_iteration_count,
        "optimization_seconds": elapsed_seconds,
        "failure_reason": result.replay.failure_reason,
        "artifact_path": str(artifact_path),
    }


def _generate_scenario(task):
    """Generate, C-verify, and persist one scenario. Runs in-process or spawned."""
    import traceback
    import torch

    (
        scenario_idx,
        generation,
        optimization_config,
        destination,
        map_path,
        env_config,
        render_replays,
        show_progress,
        worker_torch_thread_count,
    ) = task
    if worker_torch_thread_count is not None:
        torch.set_num_threads(worker_torch_thread_count)
    seed = generation["seed"] + scenario_idx
    try:
        drive = _build_drive(generation["env"], scenario_idx, seed)
        started_at = time.perf_counter()
        try:
            result = run_reactive_idm_generation(
                drive,
                optimization_config,
                deterministic_seed=seed,
                horizon_transition_count=generation["horizon_transition_count"],
                maximum_outer_iterations=generation["maximum_outer_iterations"],
                capture_html_frames=render_replays,
                show_progress=show_progress,
                raster_resolution_meters=generation["raster_resolution_meters"],
            )
        finally:
            drive.close()
        elapsed_seconds = time.perf_counter() - started_at
        npz_dir = Path(destination) / "npz"
        npz_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = npz_dir / f"scenario_{scenario_idx:05d}.npz"
        save_generation_artifact(artifact_path, result, generation, str(map_path))
        save_loss_history_csv(destination, scenario_idx, result)
        row = _metric_row(scenario_idx, scenario_idx, seed, result, elapsed_seconds, artifact_path)
        rendered_files = (
            render_scenario_replays(destination, scenario_idx, result, env_config) if render_replays else {}
        )
        return scenario_idx, row, rendered_files
    except Exception:
        # A spawned worker only ships the exception repr back, so log the real traceback here.
        print(f"\n[ERROR] Generation failed for scenario {scenario_idx}:")
        traceback.print_exc()
        raise


def generate_regents_scenarios(
    config_path,
    generation_name,
    output_dir=None,
    *,
    experiment_name=None,
    drivable_area_weight=None,
):
    """Generate, C-verify, and save one artifact per scenario in the configured range."""
    overall_start = time.perf_counter()
    generation = load_generation_config(config_path, generation_name)
    generation = _apply_generation_overrides(generation, experiment_name, drivable_area_weight)
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
    env_config = _full_env_config(generation["env"])
    rows = [None] * generation["scenario_count"]
    rendered_files = {}

    num_workers = generation["num_workers"]
    if num_workers == "auto":
        num_workers = os.cpu_count() or 1
    num_workers = min(num_workers, generation["scenario_count"])
    tasks = [
        (
            scenario_idx,
            generation,
            optimization_config,
            destination,
            map_paths[scenario_idx],
            env_config,
            render_replays,
            num_workers <= 1,
            1 if num_workers > 1 else None,
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
        writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)

    rejection_reasons = {}
    for row in rows:
        if row["failure_reason"] is None:
            continue
        rejection_reasons[row["failure_reason"]] = rejection_reasons.get(row["failure_reason"], 0) + 1
    scenario_count = len(rows)
    candidate_scenario_count = sum(row["candidate_count"] > 0 for row in rows)
    filtered_no_candidate_count = scenario_count - candidate_scenario_count
    generation_success_count = sum(row["generation_success"] for row in rows)
    torch_collision_count = sum(row["torch_collision"] for row in rows)
    c_confirmed_actionable_collision_count = sum(row["torch_collision"] and row["actionable_collision"] for row in rows)
    c_unconfirmed_actionable_collision_count = torch_collision_count - c_confirmed_actionable_collision_count
    wall_clock_seconds = time.perf_counter() - overall_start
    return GenerationReport(
        output_dir=destination,
        replay_index=(destination / RENDER_DIR_NAME / "index.html") if rendered_files else None,
        scenario_count=scenario_count,
        candidate_scenario_count=candidate_scenario_count,
        filtered_no_candidate_count=filtered_no_candidate_count,
        generation_success_count=generation_success_count,
        torch_collision_count=torch_collision_count,
        c_confirmed_actionable_collision_count=c_confirmed_actionable_collision_count,
        c_unconfirmed_actionable_collision_count=c_unconfirmed_actionable_collision_count,
        generation_success_rate=generation_success_count / scenario_count,
        candidate_success_rate=(
            generation_success_count / candidate_scenario_count if candidate_scenario_count else 0.0
        ),
        torch_collision_rate=(torch_collision_count / candidate_scenario_count if candidate_scenario_count else 0.0),
        c_collision_confirmation_rate=(
            c_confirmed_actionable_collision_count / torch_collision_count if torch_collision_count else 0.0
        ),
        ego_collision_rate=sum(row["ego_collision"] for row in rows) / scenario_count,
        actionable_collision_rate=sum(row["actionable_collision"] for row in rows) / scenario_count,
        background_collision_rate=sum(row["background_collision"] for row in rows) / scenario_count,
        offroad_rate=sum(row["offroad"] for row in rows) / scenario_count,
        maximum_c_torch_trajectory_error=max(row["c_torch_trajectory_error"] for row in rows),
        total_optimization_seconds=sum(row["optimization_seconds"] for row in rows),
        wall_clock_seconds=wall_clock_seconds,
        rejection_reasons=rejection_reasons,
    )
