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
        self.best_ever_score: float | None = None

    def initialize_agents(self, agents: list[AgentState]) -> None:
        for agent in agents:
            score = agent.selection_score if math.isfinite(agent.selection_score) else None
            self.agent_states[agent.metadata.global_id] = AgentDisplayState(status="waiting", selection_score=score)
            if score is not None:
                self._update_best_ever(score)

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
                self._update_best_ever(event.agent.selection_score)

    def _update_best_ever(self, score: float) -> None:
        if self.best_ever_score is None or score > self.best_ever_score:
            self.best_ever_score = score

    def render(
        self,
        round_index: int,
        agents: list[AgentState],
        avg_round_duration: float | None = None,
        eta_seconds: float | None = None,
    ) -> None:
        current_round = round_index + 1
        total_steps = current_round * self.round_train_env_steps
        avg_round_str = _format_duration(avg_round_duration)
        eta_str = _format_duration(eta_seconds)

        lines = [
            f"Current round: {current_round}/{self.num_rounds}",
            f"Steps: {total_steps}",
            f"Best score: {self.best_ever_score if self.best_ever_score is not None else 'n/a'}",
            f"Avg round time: {avg_round_str}",
            f"ETA: {eta_str}",
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


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(int(seconds), 0)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
