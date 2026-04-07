from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainerState:
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None = None
    normalizer_state: Any = None
    extra_state: dict[str, Any] | None = None
    checkpoint_path: str | None = None


@dataclass
class AgentMetadata:
    global_id: int
    local_id: int
    population_id: int
    parent_hps: int
    parent_network: int


@dataclass
class AgentState:
    metadata: AgentMetadata
    hyperparameters: dict[str, Any]
    trainer_state: TrainerState
    selection_score: float = float("-inf")
    env_steps: int = 0


@dataclass
class ExperimentState:
    round_index: int
    frequencies: list[int]
    agents: list[AgentState] = field(default_factory=list)
    checkpoint_stage: str = "post_evolution"


@dataclass
class WorkerTask:
    agent: AgentState | None
    round_budget: int
    seed: int | None = None
    stop: bool = False


@dataclass
class WorkerResult:
    agent: AgentState
    worker_id: int
    device_id: int


@dataclass
class WorkerEvent:
    event_type: str
    global_id: int
    worker_id: int
    device_id: int
    agent: AgentState | None = None
