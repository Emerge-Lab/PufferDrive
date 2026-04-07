from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MFPBTConfig:
    num_agents: int
    frequencies: list[int]
    num_devices: int
    num_agents_per_device: int
    num_rounds: int
    round_train_env_steps: int
    selection_metric: str = "puffer_score"
    eval_simulation_mode: str = "gigaflow"
    eval_map_dir: str | None = None
    eval_num_scenarios: int | None = None
    eval_num_agents: int | None = None
    eval_num_carla_maps: int = 8
    checkpoint_path: str | None = None
    log_dir: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    tune_hyperparameters: list[str] = field(default_factory=list)
    perturb_factors: list[float] = field(default_factory=lambda: [0.8, 1.25])
    start_method: str = "spawn"

    @property
    def num_populations(self) -> int:
        return len(self.frequencies)

    @property
    def num_agents_per_population(self) -> int:
        return self.num_agents // self.num_populations

    @property
    def max_concurrent_agents(self) -> int:
        return self.num_devices * self.num_agents_per_device

    def validate(self) -> None:
        if self.num_devices <= 0:
            raise ValueError("num_devices must be positive")
        if self.num_agents_per_device <= 0:
            raise ValueError("num_agents_per_device must be positive")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if not self.frequencies:
            raise ValueError("frequencies must be non-empty")
        if self.num_agents % self.num_populations != 0:
            raise ValueError("num_agents must be divisible by num_populations")
        if self.num_agents_per_population % 4 != 0:
            raise ValueError("num_agents_per_population must be divisible by 4")
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if self.round_train_env_steps <= 0:
            raise ValueError("round_train_env_steps must be positive")
        if self.eval_simulation_mode not in ("gigaflow", "replay"):
            raise ValueError("eval_simulation_mode must be 'gigaflow' or 'replay'")
        if self.eval_num_carla_maps <= 0:
            raise ValueError("eval_num_carla_maps must be positive")
        for hp_name in self.tune_hyperparameters:
            if hp_name not in self.hyperparameters:
                raise ValueError(f"Missing initial value for tuned hyperparameter: {hp_name}")


def load_mfpbt_config(path: str | Path) -> MFPBTConfig:
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}

    config = MFPBTConfig(**data)
    config.validate()
    return config
