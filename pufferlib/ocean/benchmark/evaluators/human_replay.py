"""HumanReplayEvaluator — replay mode + control_sdc_only, one rollout per
bin in the map_dir, mean of per-episode info dicts."""

import os
from typing import ClassVar

import numpy as np
import torch

import pufferlib
from pufferlib.ocean.benchmark.evaluators.base import EvalResult, Evaluator


class HumanReplayEvaluator(Evaluator):
    type_name: ClassVar[str] = "human_replay"

    def env_overrides(self) -> dict:
        env = {
            "simulation_mode": "replay",
            "control_mode": "control_sdc_only",
            "init_mode": "create_all_valid",
            "eval_mode": 1,
            "termination_mode": 0,
            "reward_randomization": False,
        }
        env.update(self.config.get("env", {}))
        # num_agents = number of bins so each gets one episode slot
        if "num_agents" not in env:
            map_dir = env.get("map_dir", "")
            if map_dir and os.path.isdir(map_dir):
                env["num_agents"] = len([f for f in os.listdir(map_dir) if f.endswith(".bin")])
                env["num_maps"] = env["num_agents"]
        return env

    def rollout(self, vecenv, policy, args) -> EvalResult:
        device = args["train"]["device"]
        scenario_length = int(args["env"]["scenario_length"])
        init_steps = int(args["env"].get("init_steps", 0))
        num_maps = int(args["env"]["num_maps"])
        num_agents = vecenv.observation_space.shape[0]

        # +1 step margin: env emits done on the step after scenario_length.
        total_steps = (scenario_length - init_steps + 1) * num_maps

        obs, _ = vecenv.reset()
        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        all_infos = []
        for _ in range(total_steps):
            with torch.no_grad():
                ob_t = torch.as_tensor(obs).to(device)
                logits, _ = policy.forward_eval(ob_t, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                action_np = action.cpu().numpy().reshape(vecenv.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, vecenv.action_space.low, vecenv.action_space.high)
            obs, _, _, _, info_list = vecenv.step(action_np)
            if info_list:
                all_infos.extend(info_list)
            # Stop once every bin has yielded one info to avoid double-counting
            # on the second cycle through the dir.
            if len(all_infos) >= num_maps:
                break

        if not all_infos:
            return EvalResult(metrics={"num_scenarios_completed": 0})

        metrics = {"num_scenarios_completed": float(len(all_infos))}
        keys = set().union(*(d.keys() for d in all_infos))
        for k in keys:
            vals = [d[k] for d in all_infos if isinstance(d.get(k), (int, float))]
            if vals:
                metrics[k] = float(np.mean(vals))
        return EvalResult(metrics=metrics, frames=[])
