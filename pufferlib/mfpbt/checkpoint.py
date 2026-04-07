from __future__ import annotations

import os

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
