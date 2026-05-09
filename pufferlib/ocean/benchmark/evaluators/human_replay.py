"""HumanReplayEvaluator — replay mode + control_sdc_only, one rollout per
bin in the map_dir, mean of per-episode info dicts.

Inherits the default rollout loop from `Evaluator`; only overrides
`_should_stop` to terminate once every bin has produced one info."""

import os
from typing import ClassVar

from pufferlib.ocean.benchmark.evaluators.base import Evaluator


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

    def _should_stop(self, args, infos_collected, steps) -> bool:
        # Stop once every bin has yielded one info, OR after a step budget
        # generous enough to give every bin a chance (env auto-resamples).
        scenario_length = int(args["env"]["scenario_length"])
        init_steps = int(args["env"].get("init_steps", 0))
        num_maps = int(args["env"]["num_maps"])
        max_steps = (scenario_length - init_steps + 1) * num_maps
        return len(infos_collected) >= num_maps or steps >= max_steps
