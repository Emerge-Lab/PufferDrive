from __future__ import annotations

import copy
import json
import os
import shutil

import torch

from .types import AgentState, ExperimentState


def save_experiment_checkpoint(experiment_state: ExperimentState, path: str) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = f"{path}.tmp"
    torch.save(experiment_state, tmp_path)
    os.replace(tmp_path, path)
    return path


def load_experiment_checkpoint(path: str) -> ExperimentState:
    return torch.load(path, weights_only=False)


def save_archive_checkpoint(experiment_state: ExperimentState, archive_dir: str) -> str:
    os.makedirs(archive_dir, exist_ok=True)
    trainer_state_dir = os.path.join(archive_dir, "trainer_states")
    os.makedirs(trainer_state_dir, exist_ok=True)

    archived_state = copy.deepcopy(experiment_state)
    for agent in archived_state.agents:
        source_path = agent.trainer_state.checkpoint_path
        if source_path is None:
            continue

        destination_path = os.path.join(trainer_state_dir, os.path.basename(source_path))
        tmp_destination_path = f"{destination_path}.tmp"
        shutil.copy2(source_path, tmp_destination_path)
        os.replace(tmp_destination_path, destination_path)
        agent.trainer_state.checkpoint_path = destination_path

    checkpoint_path = os.path.join(archive_dir, "checkpoint.pt")
    save_experiment_checkpoint(archived_state, checkpoint_path)
    return checkpoint_path


def save_round_best_model(agent: AgentState, round_index: int, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    source_path = agent.trainer_state.checkpoint_path
    if source_path is None:
        raise ValueError("Round-best model cannot be saved because checkpoint_path is missing")

    agent_id = agent.metadata.global_id
    model_path = os.path.join(output_dir, f"round_{round_index:06d}_agent_id_{agent_id}.pt")
    metadata_path = os.path.join(output_dir, f"round_{round_index:06d}_agent_id_{agent_id}.json")

    tmp_model_path = f"{model_path}.tmp"
    shutil.copy2(source_path, tmp_model_path)
    os.replace(tmp_model_path, model_path)

    metadata = {
        "round_index": round_index,
        "agent_id": agent_id,
        "global_id": agent.metadata.global_id,
        "local_id": agent.metadata.local_id,
        "population_id": agent.metadata.population_id,
        "selection_score": agent.selection_score,
        "env_steps": agent.env_steps,
        "parent_network": agent.metadata.parent_network,
        "parent_hps": agent.metadata.parent_hps,
        "hyperparameters": agent.hyperparameters,
        "source_checkpoint_path": source_path,
        "saved_model_path": model_path,
    }
    tmp_metadata_path = f"{metadata_path}.tmp"
    with open(tmp_metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_metadata_path, metadata_path)

    return model_path
