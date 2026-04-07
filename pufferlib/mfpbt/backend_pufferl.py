from __future__ import annotations

import copy
import os
import tempfile
import time

import torch

from pufferlib.pufferl import PuffeRL, build_eval_overrides, eval_multi_scenarios, load_env, load_policy

from .backend import TrainerBackend
from .types import AgentState


class PufferLTrainerBackend(TrainerBackend):
    def __init__(
        self,
        device_id: int,
        env_name: str,
        base_args: dict,
        selection_metric: str,
        eval_simulation_mode: str = "gigaflow",
        eval_map_dir: str | None = None,
        eval_num_scenarios: int | None = None,
        eval_num_agents: int | None = None,
        eval_num_carla_maps: int = 8,
    ):
        super().__init__(device_id=device_id)
        self.env_name = env_name
        self.base_args = base_args
        self.selection_metric = selection_metric
        self.eval_simulation_mode = eval_simulation_mode
        self.eval_map_dir = eval_map_dir
        self.eval_num_scenarios = eval_num_scenarios
        self.eval_num_agents = eval_num_agents
        self.eval_num_carla_maps = eval_num_carla_maps

    def _make_train_args(self, hyperparameters: dict, round_budget: int) -> dict:
        args = copy.deepcopy(self.base_args)
        train_args = args["train"]

        if "learning_rate" in hyperparameters:
            train_args["learning_rate"] = hyperparameters["learning_rate"]

        train_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        train_args["render"] = False
        train_args["checkpoint_interval"] = 10**18
        train_args["save_best_model"] = False
        train_args["quiet"] = True
        train_args["total_timesteps"] = round_budget

        eval_args = args.get("eval", {})
        eval_args["multi_scenario_eval"] = False
        eval_args["wosac_realism_eval"] = False
        eval_args["human_replay_eval"] = False
        args["eval"] = eval_args
        return args

    def _make_eval_args(self, hyperparameters: dict, global_step: int, seed: int | None, global_id: int) -> dict:
        eval_cfg = self.base_args["eval"]
        simulation_mode = self.eval_simulation_mode
        num_agents = self.eval_num_agents or eval_cfg["num_agents"]
        num_scenarios = self.eval_num_scenarios or eval_cfg["multi_scenario_num_scenarios"]
        if self.eval_map_dir is not None:
            map_dir = self.eval_map_dir
        elif simulation_mode == "gigaflow":
            map_dir = "pufferlib/resources/drive/binaries/carla"
        else:
            map_dir = eval_cfg["map_dir"]

        eval_overrides = build_eval_overrides(
            simulation_mode=simulation_mode,
            num_agents=num_agents,
            num_scenarios=num_scenarios,
            map_dir=map_dir,
            num_carla_maps=self.eval_num_carla_maps,
        )

        args = copy.deepcopy(self.base_args)
        for section, values in eval_overrides.items():
            if isinstance(values, dict):
                args[section].update(values)
            else:
                args[section] = values

        train_args = args["train"]
        if "learning_rate" in hyperparameters:
            train_args["learning_rate"] = hyperparameters["learning_rate"]

        train_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        if seed is not None:
            train_args["seed"] = seed

        args["global_step"] = global_step
        args["num_scenarios"] = num_scenarios
        args["eval_simulation"] = simulation_mode
        args["inline_eval"] = True
        args["eval_results_dir"] = tempfile.mkdtemp(prefix=f"mfpbt_eval_agent_{global_id}_")
        return args

    def run_round(self, agent: AgentState, round_budget: int, seed: int | None = None) -> AgentState:
        round_train_args = self._make_train_args(agent.hyperparameters, agent.env_steps + round_budget)
        if seed is not None:
            round_train_args["train"]["seed"] = seed

        vecenv = load_env(self.env_name, round_train_args)
        policy = load_policy(round_train_args, vecenv, self.env_name)
        train_config = dict(**round_train_args["train"], env=self.env_name, eval=round_train_args.get("eval", {}))
        pufferl = PuffeRL(train_config, vecenv, policy, logger=None)

        try:
            if agent.trainer_state.model_state:
                pufferl.import_trainer_state(agent.trainer_state)
            pufferl.set_hyperparameters(agent.hyperparameters)

            target_steps = agent.env_steps + round_budget
            while pufferl.global_step < target_steps:
                pufferl.evaluate()
                pufferl.train()

            eval_args = self._make_eval_args(agent.hyperparameters, pufferl.global_step, seed, agent.metadata.global_id)
            eval_metrics = eval_multi_scenarios(
                self.env_name,
                args=eval_args,
                vecenv=None,
                policy=pufferl.uncompiled_policy,
                logger=None,
                metric_prefix="validation",
                quiet=True,
            )

            updated_agent = copy.deepcopy(agent)
            updated_agent.trainer_state = pufferl.export_trainer_state()
            updated_agent.selection_score = float(eval_metrics[self.selection_metric])
            updated_agent.env_steps = int(pufferl.global_step)
            return updated_agent
        finally:
            pufferl.vecenv.close()
            pufferl.utilization.stop()
            time.sleep(0.05)
