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
        selection_source: str = "eval",
        eval_simulation_mode: str = "gigaflow",
        eval_map_dir: str | None = None,
        eval_num_scenarios: int | None = None,
        eval_num_agents: int | None = None,
        eval_num_carla_maps: int = 8,
        trainer_state_dir: str | None = None,
    ):
        super().__init__(device_id=device_id)
        self.env_name = env_name
        self.base_args = base_args
        self.selection_metric = selection_metric
        self.selection_source = selection_source
        self.eval_simulation_mode = eval_simulation_mode
        self.eval_map_dir = eval_map_dir
        self.eval_num_scenarios = eval_num_scenarios
        self.eval_num_agents = eval_num_agents
        self.eval_num_carla_maps = eval_num_carla_maps
        self.trainer_state_dir = trainer_state_dir
        if self.trainer_state_dir is not None:
            os.makedirs(self.trainer_state_dir, exist_ok=True)

    def _load_agent_trainer_state(self, agent: AgentState):
        if agent.trainer_state.checkpoint_path:
            return torch.load(agent.trainer_state.checkpoint_path, weights_only=False)
        return agent.trainer_state

    def _save_agent_trainer_state(self, global_id: int, trainer_state):
        if self.trainer_state_dir is None:
            return trainer_state

        path = os.path.join(self.trainer_state_dir, f"agent_{global_id}.pt")
        tmp_path = f"{path}.tmp"
        torch.save(trainer_state, tmp_path)
        os.replace(tmp_path, path)
        return agent_state_placeholder(path)

    def _make_train_args(self, hyperparameters: dict, round_budget: int) -> dict:
        args = copy.deepcopy(self.base_args)
        train_args = args["train"]

        for key, value in hyperparameters.items():
            train_args[key] = value

        train_args["device"] = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        train_args["render"] = False
        train_args["checkpoint_interval"] = 10**18
        train_args["enable_checkpointing"] = False
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
        for key, value in hyperparameters.items():
            train_args[key] = value

        train_args["device"] = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        if seed is not None:
            train_args["seed"] = seed
            args["vec"]["seed"] = seed

        args["global_step"] = global_step
        args["num_scenarios"] = num_scenarios
        args["eval_simulation"] = simulation_mode
        args["inline_eval"] = True
        args["eval_results_dir"] = tempfile.mkdtemp(prefix=f"mfpbt_eval_agent_{global_id}_")
        return args

    def _selection_score_from_eval(self, pufferl, hyperparameters: dict, seed: int | None, global_id: int) -> float:
        eval_args = self._make_eval_args(hyperparameters, pufferl.global_step, seed, global_id)
        eval_metrics = eval_multi_scenarios(
            self.env_name,
            args=eval_args,
            vecenv=None,
            policy=pufferl.uncompiled_policy,
            logger=None,
            metric_prefix="validation",
            quiet=True,
        )
        return float(eval_metrics[self.selection_metric])

    def _selection_score_from_train(self, pufferl, last_logs):
        metric_keys = [f"environment/{self.selection_metric}", self.selection_metric]
        if last_logs is not None:
            for key in metric_keys:
                if key in last_logs:
                    return float(last_logs[key])

        for key in (self.selection_metric,):
            if key in pufferl.stats:
                value = pufferl.stats[key]
                if isinstance(value, (int, float)):
                    return float(value)
            if key in pufferl.last_stats:
                value = pufferl.last_stats[key]
                if isinstance(value, (int, float)):
                    return float(value)

        raise KeyError(f"Training logs did not contain selection metric '{self.selection_metric}'")

    def run_round(self, agent: AgentState, round_budget: int, seed: int | None = None) -> AgentState:
        round_train_args = self._make_train_args(agent.hyperparameters, agent.env_steps + round_budget)
        if seed is not None:
            round_train_args["train"]["seed"] = seed
            round_train_args["vec"]["seed"] = seed

        vecenv = load_env(self.env_name, round_train_args)
        policy = load_policy(round_train_args, vecenv, self.env_name)
        train_config = dict(**round_train_args["train"], env=self.env_name, eval=round_train_args.get("eval", {}))
        pufferl = PuffeRL(train_config, vecenv, policy, logger=None)

        try:
            loaded_trainer_state = self._load_agent_trainer_state(agent)
            if loaded_trainer_state.model_state or loaded_trainer_state.checkpoint_path is not None:
                pufferl.import_trainer_state(loaded_trainer_state)
            pufferl.set_hyperparameters(agent.hyperparameters)

            target_steps = agent.env_steps + round_budget
            last_logs = None
            while pufferl.global_step < target_steps:
                pufferl.evaluate()
                logs = pufferl.train()
                if logs is not None:
                    last_logs = logs
            if self.selection_source == "eval":
                selection_score = self._selection_score_from_eval(
                    pufferl, agent.hyperparameters, seed, agent.metadata.global_id
                )
            else:
                try:
                    selection_score = self._selection_score_from_train(pufferl, last_logs)
                except KeyError:
                    selection_score = self._selection_score_from_eval(
                        pufferl, agent.hyperparameters, seed, agent.metadata.global_id
                    )

            updated_agent = copy.deepcopy(agent)
            exported_trainer_state = pufferl.export_trainer_state()
            updated_agent.trainer_state = self._save_agent_trainer_state(
                agent.metadata.global_id, exported_trainer_state
            )
            updated_agent.selection_score = selection_score
            updated_agent.env_steps = int(pufferl.global_step)
            return updated_agent
        finally:
            pufferl.vecenv.close()
            pufferl.utilization.stop()
            time.sleep(0.05)


def agent_state_placeholder(path: str):
    from .types import TrainerState

    return TrainerState(model_state={}, optimizer_state={}, checkpoint_path=path)
