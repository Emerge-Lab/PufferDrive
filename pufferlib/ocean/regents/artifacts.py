"""Portable, validation-first artifacts for offline ReGentS generation."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pufferlib.ocean.regents.rollout import ReactiveGenerationResult


ARTIFACT_SCHEMA = "pufferdrive_regents_generation_v2"
SUPPORTED_ARTIFACT_SCHEMAS = ("pufferdrive_regents_generation_v1", ARTIFACT_SCHEMA)
BACKGROUND_COLLISION_LOSS_SCOPE = "candidate_pairs"
MAX_ARTIFACT_ARRAY_ELEMENTS = 100_000_000


def source_configuration_hash(configuration, map_path):
    """Hash canonical configuration and exact source scenario bytes."""
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(canonical)
    path = Path(map_path)
    with path.open("rb") as source_file:
        for block in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cost_dict(costs, scenario):
    if costs is None:
        return None
    values = asdict(costs)
    for endpoint in ("first", "second"):
        index_key = f"background_collision_{endpoint}_agent_idx"
        agent_idx = values[index_key]
        values[f"background_collision_{endpoint}_agent_id"] = (
            int(scenario.agent_id[0, agent_idx].item()) if agent_idx >= 0 else -1
        )
    return values


def save_generation_artifact(path, result, source_configuration, map_path):
    """Atomically save actions, masks, trajectories, replay metadata, and metrics."""
    if not isinstance(result, ReactiveGenerationResult):
        raise TypeError("result must be a ReactiveGenerationResult")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scenario = result.scenario
    optimization = result.optimization
    replay = result.replay
    metadata = {
        "schema": ARTIFACT_SCHEMA,
        "scenario_id": scenario.scenario_ids[0],
        "dataset_name": scenario.dataset_names[0],
        "source_map": str(Path(map_path)),
        "source_configuration_hash": source_configuration_hash(source_configuration, map_path),
        "source_configuration": source_configuration,
        "deterministic_seed": result.deterministic_seed,
        "dt_seconds": scenario.dt_seconds,
        "init_step": scenario.init_step,
        "outer_iteration_count": result.outer_iteration_count,
        "optimization": {
            "background_collision_loss_scope": BACKGROUND_COLLISION_LOSS_SCOPE,
            "success": optimization.success,
            "failure_reason": optimization.failure_reason,
            "selected_adversary_idx": optimization.selected_adversary_idx,
            "selected_adversary_id": optimization.selected_adversary_id,
            "ego_collision_loss_adversary_idx": optimization.ego_collision_loss_adversary_idx,
            "ego_collision_loss_adversary_id": optimization.ego_collision_loss_adversary_id,
            "collision_timestep": optimization.collision_timestep,
            "iteration_count": optimization.iteration_count,
            "best_iteration": optimization.best_iteration,
            "initial_costs": _cost_dict(optimization.initial_costs, scenario),
            "final_costs": _cost_dict(optimization.final_costs, scenario),
            "background_collision": optimization.background_collision,
            "offroad": optimization.offroad,
            "baseline_background_collision_pair_count": optimization.baseline_background_collision_pair_count,
            "background_collision_rejection_count": optimization.background_collision_rejection_count,
            "offroad_rejection_count": optimization.offroad_rejection_count,
            "failure_reasons_by_agent": [
                optimization.selection.reasons_for(0, agent_idx) for agent_idx in range(scenario.max_agent_count)
            ],
        },
        "c_replay": {**asdict(replay.metrics), "success": replay.success, "failure_reason": replay.failure_reason},
        "replay_metadata": {
            "original": {"controller": "logged", "state_array": "original_states"},
            "adversarial": {"controller": "mixed_c", "state_array": "c_states"},
        },
    }
    serialized_metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    metadata_bytes = np.frombuffer(serialized_metadata.encode("utf-8"), dtype=np.uint8)
    arrays = {
        "metadata_json_utf8": metadata_bytes,
        "agent_id": scenario.agent_id.cpu().numpy(),
        "state_valid": scenario.state_valid.cpu().numpy(),
        "transition_valid": scenario.transition_valid.cpu().numpy(),
        "candidate_mask": optimization.selection.candidate_mask.cpu().numpy(),
        "optimized_action_mask": optimization.optimized_action_mask.cpu().numpy(),
        "initial_actions": optimization.initial_actions.detach().cpu().numpy(),
        "optimized_actions": optimization.optimized_actions.detach().cpu().numpy(),
        "torch_states": optimization.optimized_states.detach().cpu().numpy(),
        "original_states": scenario.logged_state.cpu().numpy(),
        "c_states": replay.states.cpu().numpy(),
        "c_state_valid": replay.state_valid.cpu().numpy(),
        "c_ego_actions": replay.ego_actions.cpu().numpy(),
    }
    temporary_path = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(temporary_path, **arrays)
    temporary_path.replace(destination)
    return json.loads(serialized_metadata)


def load_generation_artifact(path):
    """Load an artifact without pickle and reject malformed or oversized arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "metadata_json_utf8",
            "agent_id",
            "state_valid",
            "transition_valid",
            "candidate_mask",
            "optimized_action_mask",
            "initial_actions",
            "optimized_actions",
            "torch_states",
            "original_states",
            "c_states",
            "c_state_valid",
            "c_ego_actions",
        }
        if set(archive.files) != required:
            raise ValueError("Artifact fields do not match the ReGentS schema")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in required}
    for name, array in arrays.items():
        if array.size > MAX_ARTIFACT_ARRAY_ELEMENTS:
            raise ValueError(f"Artifact array {name} exceeds the element limit")
    metadata_array = arrays.pop("metadata_json_utf8")
    if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
        raise ValueError("Artifact metadata must be a one-dimensional uint8 array")
    metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
    if metadata.get("schema") not in SUPPORTED_ARTIFACT_SCHEMAS:
        raise ValueError("Unknown ReGentS artifact schema")
    return metadata, arrays
