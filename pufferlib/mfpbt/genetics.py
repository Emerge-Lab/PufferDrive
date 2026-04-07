from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

from .types import AgentState


def perturbation(factors: list[float] | None = None) -> Callable[[float], float]:
    factors = factors or [0.8, 1.25]

    def perturbation_func(hp_value: float) -> float:
        factor = np.random.choice(factors)
        return hp_value * factor

    return perturbation_func


def mf_pbt_genetics(
    global_ranking: list[int],
    local_rankings: list[list[int]],
    round_index: int,
    frequencies: list[int],
) -> tuple[list[int], list[int], list[bool]]:
    num_populations = len(frequencies)
    num_agents = len(global_ranking)
    num_agents_per_population = len(local_rankings[0])

    concerned = [((round_index + 1) % frequency == 0) for frequency in frequencies]

    parents_hps = list(range(num_agents))
    parents_network = list(range(num_agents))
    need_explore = [False] * num_agents

    inverse_rank = [0] * num_agents
    for pos, global_id in enumerate(global_ranking):
        inverse_rank[global_id] = num_agents - pos

    def local_to_global(population_id: int, local_id: int) -> int:
        return population_id * num_agents_per_population + local_id

    def get_population(global_id: int) -> int:
        return global_id // num_agents_per_population

    for population_id in range(num_populations):
        if not concerned[population_id]:
            continue

        local_rank = local_rankings[population_id]
        share = num_agents_per_population // 4

        best = local_rank[:share]
        worst = local_rank[-share:]

        for loser_local, winner_local in zip(worst, best):
            loser_global = local_to_global(population_id, loser_local)
            winner_global = local_to_global(population_id, winner_local)

            parents_hps[loser_global] = winner_global
            parents_network[loser_global] = winner_global
            need_explore[loser_global] = True

    for population_id in range(num_populations):
        if not concerned[population_id]:
            continue

        local_rank = local_rankings[population_id]
        share = num_agents_per_population // 4
        migration_local = local_rank[2 * share : 3 * share]
        migration_global = [local_to_global(population_id, local_id) for local_id in migration_local]

        external_ranking = [global_id for global_id in global_ranking if get_population(global_id) != population_id]

        for global_id in migration_global:
            if not external_ranking:
                break

            migrant = external_ranking[0]
            if inverse_rank[global_id] < inverse_rank[migrant]:
                parents_network[global_id] = migrant

                destination_population = population_id
                source_population = get_population(migrant)
                if frequencies[destination_population] < frequencies[source_population]:
                    parents_hps[global_id] = migrant
                elif frequencies[destination_population] > frequencies[source_population]:
                    best_internal_local = local_rank[0]
                    parents_hps[global_id] = local_to_global(destination_population, best_internal_local)

                external_ranking.pop(0)

    return parents_hps, parents_network, need_explore


def apply_mf_pbt_genetics(
    agents: list[AgentState],
    parents_hps: list[int],
    parents_network: list[int],
    need_explore: list[bool],
    explore_fns: dict[str, Callable[[Any], Any]],
) -> None:
    old_agents = copy.deepcopy(agents)

    for index, agent in enumerate(agents):
        agent.metadata.parent_hps = parents_hps[index]
        agent.metadata.parent_network = parents_network[index]

        if parents_network[index] != index:
            agent.trainer_state = copy.deepcopy(old_agents[parents_network[index]].trainer_state)

        if parents_hps[index] != index:
            parent_agent = old_agents[parents_hps[index]]
            new_hps = copy.deepcopy(parent_agent.hyperparameters)

            if need_explore[index]:
                for hp_name, hp_value in new_hps.items():
                    if hp_name not in explore_fns:
                        continue
                    new_hps[hp_name] = explore_fns[hp_name](hp_value)

            agent.hyperparameters = new_hps
