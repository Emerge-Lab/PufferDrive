from __future__ import annotations

import copy
import os
import shutil

import torch

from .types import ExperimentState


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
