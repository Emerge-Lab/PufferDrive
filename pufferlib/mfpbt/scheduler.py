from __future__ import annotations

import multiprocessing as mp
import os
import queue
from collections.abc import Callable

from .backend import TrainerBackend
from .types import AgentState, WorkerResult, WorkerTask


def _worker_main(
    worker_id: int,
    device_id: int,
    backend_factory: Callable[..., TrainerBackend],
    task_queue,
    result_queue,
    backend_kwargs: dict,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    backend = backend_factory(device_id=device_id, **backend_kwargs)

    try:
        while True:
            task = task_queue.get()
            if task.stop:
                break

            updated_agent = backend.run_round(task.agent, task.round_budget, seed=task.seed)
            result_queue.put(WorkerResult(agent=updated_agent, worker_id=worker_id, device_id=device_id))
    finally:
        backend.close()


class WorkerPoolScheduler:
    def __init__(
        self,
        backend_factory: Callable[..., TrainerBackend],
        num_devices: int,
        num_agents_per_device: int,
        start_method: str = "spawn",
        **backend_kwargs,
    ):
        self.backend_factory = backend_factory
        self.num_devices = num_devices
        self.num_agents_per_device = num_agents_per_device
        self.backend_kwargs = backend_kwargs

        ctx = mp.get_context(start_method)
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.workers = []

        worker_count = num_devices * num_agents_per_device
        for worker_id in range(worker_count):
            device_id = worker_id % num_devices
            worker = ctx.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    device_id,
                    backend_factory,
                    self.task_queue,
                    self.result_queue,
                    backend_kwargs,
                ),
            )
            worker.start()
            self.workers.append(worker)

    def run_round(
        self,
        agents: list[AgentState],
        round_budget: int,
        seeds: list[int] | None = None,
    ) -> list[AgentState]:
        seeds = seeds or [None] * len(agents)
        if len(seeds) != len(agents):
            raise ValueError("seeds length must match agents length")

        for agent, seed in zip(agents, seeds):
            self.task_queue.put(WorkerTask(agent=agent, round_budget=round_budget, seed=seed))

        results_by_id = {}
        while len(results_by_id) < len(agents):
            try:
                result = self.result_queue.get(timeout=1.0)
            except queue.Empty:
                if any(not worker.is_alive() for worker in self.workers):
                    raise RuntimeError("MF-PBT worker died during round execution")
                continue

            results_by_id[result.agent.metadata.global_id] = result.agent

        return [results_by_id[agent.metadata.global_id] for agent in agents]

    def close(self) -> None:
        for _ in self.workers:
            self.task_queue.put(WorkerTask(agent=None, round_budget=0, stop=True))

        for worker in self.workers:
            worker.join()
