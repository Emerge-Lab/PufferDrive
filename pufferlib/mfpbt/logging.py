from __future__ import annotations

import csv
import os

from .types import AgentState


class MFPBT_Logger:
    def __init__(self, directory: str, tune_hyperparameters: list[str] | None = None):
        self.directory = directory
        self.tune_hyperparameters = tune_hyperparameters or []
        os.makedirs(directory, exist_ok=True)
        self.agent_history_path = os.path.join(directory, "agent_history.csv")
        self.round_summary_path = os.path.join(directory, "round_summary.csv")

    def log_round(
        self,
        round_index: int,
        agents: list[AgentState],
        frequencies: list[int],
        need_explore: list[bool],
        round_duration_sec: float | None = None,
        avg_round_duration_sec: float | None = None,
        eta_seconds: float | None = None,
    ) -> None:
        self._append_agent_rows(round_index, agents, frequencies, need_explore)
        self._append_round_summary(round_index, agents, round_duration_sec, avg_round_duration_sec, eta_seconds)

    def _append_agent_rows(
        self,
        round_index: int,
        agents: list[AgentState],
        frequencies: list[int],
        need_explore: list[bool],
    ) -> None:
        fieldnames = [
            "round_index",
            "global_id",
            "local_id",
            "population_id",
            "population_frequency",
            "selection_score",
            "env_steps",
            "parent_network",
            "parent_hps",
            "need_explore",
        ]

        hyperparameter_keys = sorted({key for agent in agents for key in agent.hyperparameters})
        fieldnames.extend(hyperparameter_keys)

        rows = []
        for index, agent in enumerate(agents):
            row = {
                "round_index": round_index,
                "global_id": agent.metadata.global_id,
                "local_id": agent.metadata.local_id,
                "population_id": agent.metadata.population_id,
                "population_frequency": frequencies[agent.metadata.population_id],
                "selection_score": agent.selection_score,
                "env_steps": agent.env_steps,
                "parent_network": agent.metadata.parent_network,
                "parent_hps": agent.metadata.parent_hps,
                "need_explore": need_explore[index],
            }
            for hp_name in hyperparameter_keys:
                row[hp_name] = agent.hyperparameters.get(hp_name)
            rows.append(row)

        self._append_rows(self.agent_history_path, fieldnames, rows)

    def _append_round_summary(
        self,
        round_index: int,
        agents: list[AgentState],
        round_duration_sec: float | None,
        avg_round_duration_sec: float | None,
        eta_seconds: float | None,
    ) -> None:
        fieldnames = [
            "round_index",
            "best_global_id",
            "best_selection_score",
            "mean_selection_score",
            "min_selection_score",
            "max_env_steps",
            "mean_env_steps",
            "round_duration_sec",
            "avg_round_duration_sec",
            "eta_seconds",
        ]
        fieldnames.extend([f"mean_{hp_name}" for hp_name in self.tune_hyperparameters])

        scores = [agent.selection_score for agent in agents]
        env_steps = [agent.env_steps for agent in agents]
        best_agent = max(agents, key=lambda agent: agent.selection_score)
        row = {
            "round_index": round_index,
            "best_global_id": best_agent.metadata.global_id,
            "best_selection_score": best_agent.selection_score,
            "mean_selection_score": sum(scores) / len(scores),
            "min_selection_score": min(scores),
            "max_env_steps": max(env_steps),
            "mean_env_steps": sum(env_steps) / len(env_steps),
            "round_duration_sec": round_duration_sec,
            "avg_round_duration_sec": avg_round_duration_sec,
            "eta_seconds": eta_seconds,
        }
        for hp_name in self.tune_hyperparameters:
            values = [
                agent.hyperparameters.get(hp_name) for agent in agents if agent.hyperparameters.get(hp_name) is not None
            ]
            row[f"mean_{hp_name}"] = sum(values) / len(values) if values else None

        self._append_rows(self.round_summary_path, fieldnames, [row])

    def _append_rows(self, path: str, fieldnames: list[str], rows: list[dict]) -> None:
        file_exists = os.path.exists(path)
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
