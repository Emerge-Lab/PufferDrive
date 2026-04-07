from __future__ import annotations

from abc import ABC, abstractmethod

from .types import AgentState


class TrainerBackend(ABC):
    def __init__(self, device_id: int, **kwargs):
        self.device_id = device_id

    @abstractmethod
    def run_round(self, agent: AgentState, round_budget: int, seed: int | None = None) -> AgentState:
        pass

    def close(self) -> None:
        pass
