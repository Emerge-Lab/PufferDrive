from __future__ import annotations

import math
from dataclasses import dataclass

from .types import AgentState, WorkerEvent


@dataclass
class AgentDisplayState:
    status: str
    device_id: int | None = None
    selection_score: float | None = None


class MFPBTProgressDisplay:
    def __init__(self, num_rounds: int, round_train_env_steps: int, frequencies: list[int]):
        self.num_rounds = num_rounds
        self.round_train_env_steps = round_train_env_steps
        self.frequencies = frequencies
        self.agent_states: dict[int, AgentDisplayState] = {}

    def initialize_agents(self, agents: list[AgentState]) -> None:
        for agent in agents:
            score = agent.selection_score if math.isfinite(agent.selection_score) else None
            self.agent_states[agent.metadata.global_id] = AgentDisplayState(status="waiting", selection_score=score)

    def begin_round(self, agents: list[AgentState]) -> None:
        for agent in agents:
            state = self.agent_states[agent.metadata.global_id]
            state.status = "waiting"

    def handle_event(self, event: WorkerEvent) -> None:
        state = self.agent_states[event.global_id]
        state.device_id = event.device_id
        if event.event_type == "started":
            state.status = "training"
        elif event.event_type == "completed":
            state.status = "complete"
            if event.agent is not None and math.isfinite(event.agent.selection_score):
                state.selection_score = event.agent.selection_score

    def render(self, round_index: int, agents: list[AgentState]) -> None:
        current_round = round_index + 1
        total_steps = current_round * self.round_train_env_steps
        best_score = max(
            (
                state.selection_score
                for state in self.agent_states.values()
                if state.selection_score is not None and math.isfinite(state.selection_score)
            ),
            default=float("-inf"),
        )

        lines = [
            f"Current round: {current_round}/{self.num_rounds}",
            f"Steps: {total_steps}",
            f"Best score: {best_score if math.isfinite(best_score) else 'n/a'}",
            "",
            "Agents:",
            "gid pop freq gpu status score",
        ]

        for agent in sorted(agents, key=lambda item: item.metadata.global_id):
            state = self.agent_states[agent.metadata.global_id]
            gpu = state.device_id if state.device_id is not None else "-"
            score = state.selection_score if state.selection_score is not None else "n/a"
            lines.append(
                f"{agent.metadata.global_id:>3} "
                f"{agent.metadata.population_id:>3} "
                f"{self.frequencies[agent.metadata.population_id]:>4} "
                f"{str(gpu):>3} "
                f"{state.status:>8} "
                f"{score}"
            )

        print("\033[2J\033[H" + "\n".join(lines), flush=True)
