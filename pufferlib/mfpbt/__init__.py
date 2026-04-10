from __future__ import annotations

from importlib import import_module

__all__ = [
    "AgentMetadata",
    "AgentState",
    "ExperimentState",
    "MFPBTController",
    "MFPBTConfig",
    "MFPBT_Logger",
    "MFPBTProgressDisplay",
    "PufferLTrainerBackend",
    "TrainerBackend",
    "TrainerState",
    "WorkerEvent",
    "WorkerPoolScheduler",
    "WorkerResult",
    "WorkerTask",
    "apply_mf_pbt_genetics",
    "load_experiment_checkpoint",
    "load_mfpbt_config",
    "mf_pbt_genetics",
    "perturbation",
    "run_mfpbt",
    "save_archive_checkpoint",
    "save_experiment_checkpoint",
]

_EXPORTS = {
    "TrainerBackend": (".backend", "TrainerBackend"),
    "PufferLTrainerBackend": (".backend_pufferl", "PufferLTrainerBackend"),
    "load_experiment_checkpoint": (".checkpoint", "load_experiment_checkpoint"),
    "save_archive_checkpoint": (".checkpoint", "save_archive_checkpoint"),
    "save_experiment_checkpoint": (".checkpoint", "save_experiment_checkpoint"),
    "MFPBTConfig": (".config", "MFPBTConfig"),
    "load_mfpbt_config": (".config", "load_mfpbt_config"),
    "MFPBTController": (".controller", "MFPBTController"),
    "run_mfpbt": (".controller", "run_mfpbt"),
    "MFPBTProgressDisplay": (".display", "MFPBTProgressDisplay"),
    "apply_mf_pbt_genetics": (".genetics", "apply_mf_pbt_genetics"),
    "mf_pbt_genetics": (".genetics", "mf_pbt_genetics"),
    "perturbation": (".genetics", "perturbation"),
    "MFPBT_Logger": (".logging", "MFPBT_Logger"),
    "WorkerPoolScheduler": (".scheduler", "WorkerPoolScheduler"),
    "AgentMetadata": (".types", "AgentMetadata"),
    "AgentState": (".types", "AgentState"),
    "ExperimentState": (".types", "ExperimentState"),
    "TrainerState": (".types", "TrainerState"),
    "WorkerEvent": (".types", "WorkerEvent"),
    "WorkerResult": (".types", "WorkerResult"),
    "WorkerTask": (".types", "WorkerTask"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
