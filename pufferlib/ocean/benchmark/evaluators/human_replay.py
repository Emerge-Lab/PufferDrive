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
        # 1 SDC per bin → num_agents == num_maps == num_eval_scenarios.
        # Override the train-config defaults (num_agents=1024 etc.) since
        # they don't apply in replay+control_sdc_only mode.
        # Without setting num_eval_scenarios, the C-side replay branch caps
        # env_count at drive.py's default of 16 — only 16 of N bins would
        # ever instantiate as envs.
        map_dir = env.get("map_dir", "")
        if map_dir and os.path.isdir(map_dir):
            n_bins = len([f for f in os.listdir(map_dir) if f.endswith(".bin")])
            env["num_agents"] = n_bins
            env["num_maps"] = n_bins
            env["num_eval_scenarios"] = n_bins
        return env

    def _should_stop(self, args, infos_collected, steps) -> bool:
        # All bins run as parallel envs in one batched rollout (because
        # num_eval_scenarios = num_maps). Every env truncates on the same
        # tick at scenario_length, vec_log emits one aggregate dict at
        # that boundary, and we have the full per-bin average. Stop right
        # after that — waiting longer just deterministically re-runs the
        # same bins (resample_frequency is effectively never in replay).
        scenario_length = int(args["env"]["scenario_length"])
        return steps >= scenario_length + 1
