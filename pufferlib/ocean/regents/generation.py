"""Offline ReGentS generation entry point over a fixed scenario range."""

import csv
import inspect
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from tqdm import tqdm

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.artifacts import save_generation_artifact
from pufferlib.ocean.regents.losses import ReGentSCostConfig
from pufferlib.ocean.regents.optimizer import ReGentSOptimizationConfig
from pufferlib.ocean.regents.rollout import run_reactive_idm_generation


METRICS_FILE_NAME = "generation_metrics.csv"
RENDER_DIR_NAME = "rendered_replays"
METRIC_FIELD_NAMES = (
    "scenario_index",
    "scenario_id",
    "map_index",
    "seed",
    "generation_success",
    "ego_collision",
    "actionable_collision",
    "background_collision",
    "offroad",
    "c_torch_trajectory_error",
    "ego_reference_error",
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
    generation_success_rate: float
    ego_collision_rate: float
    actionable_collision_rate: float
    background_collision_rate: float
    offroad_rate: float
    maximum_c_torch_trajectory_error: float
    total_optimization_seconds: float
    rejection_reasons: dict


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return value


def _require_positive_int(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


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
        "raster_resolution_meters": float(selected.get("raster_resolution_meters", 5.0)),
        "output_dir": str(selected.get("output_dir", "experiments/regents")),
        "render_replays": bool(selected.get("render_replays", False)),
        "optimizer": optimizer,
        "env": environment,
    }
    if resolved["raster_resolution_meters"] <= 0.0:
        raise ValueError("raster_resolution_meters must be positive")
    horizon = resolved["horizon_transition_count"]
    for name in ("resample_frequency", "scenario_length"):
        limit = environment.get(name)
        if limit is not None and int(limit) > 0 and horizon >= int(limit):
            raise ValueError(
                f"Generation {generation_name}: horizon_transition_count={horizon} must be below "
                f"env.{name}={int(limit)}. Set env.{name} to at least {horizon + 1} for this generation."
            )
    if resolved["scenario_count"] > int(environment.get("num_maps", 0)):
        raise ValueError("scenario_count exceeds the configured num_maps")
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
    actions = np.zeros((frame_count, max(ego_actions.shape[1], 1), 2), dtype=np.float32)
    actions[1:, : ego_actions.shape[1]] = ego_actions[0].detach().cpu().numpy()[: frame_count - 1, None, :]
    bundle = {"env": env_config, **frames, "raw_action": actions, "clipped_action": actions}
    if observations is not None:
        bundle["obs"] = observations
    return bundle


def render_scenario_replays(destination, scenario_idx, result, env_config):
    """Write logged and adversarial replays as interactive HTML with the shared viewer."""
    import pufferlib.viz

    replay = result.replay
    if replay.scenario_payload is None or replay.adversarial_frames is None:
        raise ValueError("Replay rendering requires capture_html_frames")
    render_dir = Path(destination) / RENDER_DIR_NAME
    render_dir.mkdir(parents=True, exist_ok=True)
    rendered = {}
    sources = (("logged", replay.baseline_frames), ("adversarial", replay.adversarial_frames))
    for label, frames in sources:
        stem = f"scenario_{scenario_idx:05d}.{label}"
        bundle = _replay_bundle(env_config, frames, replay.ego_actions)
        bundle["selected_adversary_idx"] = result.optimization.selected_adversary_idx
        bundle["selected_adversary_id"] = result.optimization.selected_adversary_id
        binary_path = render_dir / f"{stem}.replay.zlib"
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
    return ReGentSOptimizationConfig(
        costs=ReGentSCostConfig(**_require_mapping(optimizer_config.get("costs", {}), "optimizer costs")),
        learning_rate=float(optimizer_config.get("learning_rate", 1e-3)),
        iteration_count=_require_positive_int(optimizer_config.get("iteration_count", 500), "iteration_count"),
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
        "generation_success": int(result.replay.success),
        "ego_collision": int(metrics.ego_collision),
        "actionable_collision": int(metrics.actionable_collision),
        "background_collision": int(metrics.background_collision),
        "offroad": int(metrics.offroad),
        "c_torch_trajectory_error": metrics.maximum_trajectory_error,
        "ego_reference_error": metrics.maximum_ego_reference_error,
        "collision_timestep": metrics.first_collision_timestep,
        "selected_adversary_idx": result.optimization.selected_adversary_idx,
        "selected_adversary_id": result.optimization.selected_adversary_id,
        "outer_iteration_count": result.outer_iteration_count,
        "optimization_seconds": elapsed_seconds,
        "failure_reason": result.replay.failure_reason,
        "artifact_path": str(artifact_path),
    }


def _generate_single_scenario_worker(
    scenario_idx,
    generation,
    optimization_config,
    destination,
    map_path,
    env_config,
    render_replays,
):
    """Worker function to generate a single ReGentS scenario in a separate process."""
    import traceback
    import torch

    torch.set_num_threads(1)
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
                show_progress=False,
            )
        finally:
            drive.close()
        elapsed_seconds = time.perf_counter() - started_at
        artifact_path = destination / f"scenario_{scenario_idx:05d}.npz"
        save_generation_artifact(artifact_path, result, generation, str(map_path))
        row = _metric_row(scenario_idx, scenario_idx, seed, result, elapsed_seconds, artifact_path)
        rendered_files = {}
        if render_replays:
            rendered_files = render_scenario_replays(destination, scenario_idx, result, env_config)
        return scenario_idx, row, rendered_files
    except Exception as e:
        print(f"\n[ERROR] Worker failed for scenario {scenario_idx}:")
        traceback.print_exc()
        raise e


def _generate_single_scenario_worker_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing Pool."""
    return _generate_single_scenario_worker(*args)


def generate_regents_scenarios(config_path, generation_name, output_dir=None):
    """Generate, C-verify, and save one artifact per scenario in the configured range."""
    generation = load_generation_config(config_path, generation_name)
    destination = Path(output_dir) if output_dir is not None else Path(generation["output_dir"]) / generation_name
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

    num_workers = generation.get("num_workers", 1)
    if num_workers == "auto":
        import os

        num_workers = os.cpu_count() or 1
    num_workers = min(num_workers, generation["scenario_count"])

    if num_workers <= 1:
        pbar = tqdm(range(generation["scenario_count"]), desc="Generating scenarios")
        for scenario_idx in pbar:
            seed = generation["seed"] + scenario_idx
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
                    show_progress=False,
                )
            finally:
                drive.close()
            elapsed_seconds = time.perf_counter() - started_at
            artifact_path = destination / f"scenario_{scenario_idx:05d}.npz"
            save_generation_artifact(artifact_path, result, generation, str(map_paths[scenario_idx]))
            rows[scenario_idx] = _metric_row(scenario_idx, scenario_idx, seed, result, elapsed_seconds, artifact_path)
            if render_replays:
                rendered_files.update(render_scenario_replays(destination, scenario_idx, result, env_config))
            success_so_far = sum(row["generation_success"] for row in rows if row is not None)
            completed_so_far = sum(1 for row in rows if row is not None)
            success_rate = success_so_far / completed_so_far if completed_so_far else 0.0
            pbar.set_postfix(success=f"{success_rate:.1%}")
    else:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")

        tasks = [
            (
                scenario_idx,
                generation,
                optimization_config,
                destination,
                map_paths[scenario_idx],
                env_config,
                render_replays,
            )
            for scenario_idx in range(generation["scenario_count"])
        ]

        success_count = 0
        completed_count = 0

        with ctx.Pool(processes=num_workers) as pool:
            iterator = pool.imap_unordered(_generate_single_scenario_worker_wrapper, tasks)
            pbar = tqdm(iterator, total=len(tasks), desc="Generating scenarios (parallel)")
            for scenario_idx, row, worker_rendered in pbar:
                rows[scenario_idx] = row
                rendered_files.update(worker_rendered)

                completed_count += 1
                if row["generation_success"]:
                    success_count += 1

                success_rate = success_count / completed_count
                pbar.set_postfix(success=f"{success_rate:.1%}")

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
    return GenerationReport(
        output_dir=destination,
        replay_index=(destination / RENDER_DIR_NAME / "index.html") if rendered_files else None,
        scenario_count=scenario_count,
        generation_success_rate=sum(row["generation_success"] for row in rows) / scenario_count,
        ego_collision_rate=sum(row["ego_collision"] for row in rows) / scenario_count,
        actionable_collision_rate=sum(row["actionable_collision"] for row in rows) / scenario_count,
        background_collision_rate=sum(row["background_collision"] for row in rows) / scenario_count,
        offroad_rate=sum(row["offroad"] for row in rows) / scenario_count,
        maximum_c_torch_trajectory_error=max(row["c_torch_trajectory_error"] for row in rows),
        total_optimization_seconds=sum(row["optimization_seconds"] for row in rows),
        rejection_reasons=rejection_reasons,
    )
