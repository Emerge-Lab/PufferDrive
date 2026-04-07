from __future__ import annotations

import copy
import os
import time
from collections import defaultdict

from .checkpoint import load_experiment_checkpoint, save_experiment_checkpoint
from .config import MFPBTConfig
from .genetics import apply_mf_pbt_genetics, mf_pbt_genetics, perturbation
from .logging import MFPBT_Logger
from .scheduler import WorkerPoolScheduler
from .types import AgentMetadata, AgentState, ExperimentState, TrainerState


class MFPBTController:
    def __init__(
        self,
        config: MFPBTConfig,
        scheduler: WorkerPoolScheduler,
        checkpoint_path: str | None = None,
    ):
        self.config = config
        self.scheduler = scheduler
        self.checkpoint_path = checkpoint_path or config.checkpoint_path
        self.explore_fns = {hp_name: perturbation(config.perturb_factors) for hp_name in config.tune_hyperparameters}
        self.csv_logger = self._build_csv_logger()

    def _build_csv_logger(self):
        if self.config.log_dir is not None:
            return MFPBT_Logger(self.config.log_dir)

        if self.checkpoint_path:
            checkpoint_dir = os.path.dirname(self.checkpoint_path) or "."
            return MFPBT_Logger(os.path.join(checkpoint_dir, "logs"))

        return None

    def initialize_experiment(self) -> ExperimentState:
        if self.checkpoint_path:
            try:
                return load_experiment_checkpoint(self.checkpoint_path)
            except FileNotFoundError:
                pass

        agents = []
        per_population = self.config.num_agents_per_population
        for global_id in range(self.config.num_agents):
            population_id = global_id // per_population
            local_id = global_id % per_population
            agents.append(
                AgentState(
                    metadata=AgentMetadata(
                        global_id=global_id,
                        local_id=local_id,
                        population_id=population_id,
                        parent_hps=global_id,
                        parent_network=global_id,
                    ),
                    hyperparameters=copy.deepcopy(self.config.hyperparameters),
                    trainer_state=TrainerState(model_state={}, optimizer_state={}),
                )
            )

        return ExperimentState(round_index=0, frequencies=list(self.config.frequencies), agents=agents)

    def _global_ranking(self, agents: list[AgentState]) -> list[int]:
        ranked = sorted(agents, key=lambda agent: (agent.selection_score, -agent.metadata.global_id), reverse=True)
        return [agent.metadata.global_id for agent in ranked]

    def _local_rankings(self, agents: list[AgentState]) -> list[list[int]]:
        grouped = defaultdict(list)
        for agent in agents:
            grouped[agent.metadata.population_id].append(agent)

        local_rankings = []
        for population_id in range(self.config.num_populations):
            ranked = sorted(
                grouped[population_id],
                key=lambda agent: (agent.selection_score, -agent.metadata.local_id),
                reverse=True,
            )
            local_rankings.append([agent.metadata.local_id for agent in ranked])
        return local_rankings

    def run_round(self, experiment_state: ExperimentState, seeds: list[int] | None = None) -> ExperimentState:
        updated_agents = self.scheduler.run_round(
            experiment_state.agents,
            round_budget=self.config.round_train_env_steps,
            seeds=seeds,
        )

        global_ranking = self._global_ranking(updated_agents)
        local_rankings = self._local_rankings(updated_agents)
        parents_hps, parents_network, need_explore = mf_pbt_genetics(
            global_ranking=global_ranking,
            local_rankings=local_rankings,
            round_index=experiment_state.round_index,
            frequencies=self.config.frequencies,
        )

        apply_mf_pbt_genetics(
            updated_agents,
            parents_hps=parents_hps,
            parents_network=parents_network,
            need_explore=need_explore,
            explore_fns=self.explore_fns,
        )

        next_state = ExperimentState(
            round_index=experiment_state.round_index + 1,
            frequencies=list(experiment_state.frequencies),
            agents=updated_agents,
            checkpoint_stage="post_evolution",
        )

        if self.checkpoint_path:
            save_experiment_checkpoint(next_state, self.checkpoint_path)

        if self.csv_logger is not None:
            self.csv_logger.log_round(
                round_index=experiment_state.round_index,
                agents=updated_agents,
                frequencies=self.config.frequencies,
                need_explore=need_explore,
            )

        return next_state

    def close(self) -> None:
        self.scheduler.close()


def run_mfpbt(
    config: MFPBTConfig,
    scheduler: WorkerPoolScheduler,
    num_rounds: int,
    checkpoint_path: str | None = None,
) -> ExperimentState:
    controller = MFPBTController(config, scheduler, checkpoint_path=checkpoint_path)
    state = controller.initialize_experiment()

    try:
        for _ in range(num_rounds):
            seeds = [time.time_ns() & 0xFFFFFFFF for _ in range(config.num_agents)]
            state = controller.run_round(state, seeds=seeds)
    finally:
        controller.close()

    return state
