"""BehaviorClassEvaluator — one nuPlan behavior category at a time.

Runs a HumanReplayEvaluator-style rollout against a single map_dir, with
optional fresh random sampling each pass when `num_scenarios` < total bins.
"""

import os
import random
import shutil
import tempfile
from typing import ClassVar

from pufferlib.ocean.benchmark.evaluators.base import EvalResult
from pufferlib.ocean.benchmark.evaluators.human_replay import HumanReplayEvaluator


class BehaviorClassEvaluator(HumanReplayEvaluator):
    type_name: ClassVar[str] = "behavior_class"

    def __init__(self, name, config, train_config):
        super().__init__(name, config, train_config)
        self._sampled_dir = None  # tmp symlink dir created per pass

    def env_overrides(self) -> dict:
        # Reuse HumanReplay's defaults, then handle the random-sampling
        # cap. If num_scenarios is smaller than total bins, build a tmp
        # symlink dir with a fresh sample each pass and point map_dir there.
        env = super().env_overrides()
        map_dir = env.get("map_dir", "")
        if not map_dir or not os.path.isdir(map_dir):
            return env

        num_scenarios = int(self.config.get("eval", {}).get("num_scenarios", 0))
        all_bins = [f for f in os.listdir(map_dir) if f.endswith(".bin")]
        if num_scenarios > 0 and num_scenarios < len(all_bins):
            sampled = random.sample(all_bins, num_scenarios)
            self._sampled_dir = tempfile.mkdtemp(prefix=f"{self.name}_")
            for fname in sampled:
                os.symlink(os.path.join(map_dir, fname), os.path.join(self._sampled_dir, fname))
            env["map_dir"] = self._sampled_dir
            env["num_agents"] = num_scenarios
            env["num_maps"] = num_scenarios
        else:
            env["num_agents"] = len(all_bins)
            env["num_maps"] = len(all_bins)
        return env

    def rollout(self, vecenv, policy, args) -> EvalResult:
        result = super().rollout(vecenv, policy, args)
        # Manager owns the cleanup window — defer rmtree until after vecenv.close
        # so any open file descriptors on the symlinks are released first.
        return result

    def cleanup(self):
        if self._sampled_dir and os.path.isdir(self._sampled_dir):
            shutil.rmtree(self._sampled_dir, ignore_errors=True)
            self._sampled_dir = None
