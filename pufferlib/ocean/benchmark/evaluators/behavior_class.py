"""BehaviorClassEvaluator — one nuPlan behavior category at a time.

Inherits HumanReplayEvaluator's loop. Adds optional fresh random sampling
each pass when `num_scenarios` < total bins (via a tmp symlink dir)."""

import os
import random
import shutil
import tempfile
from typing import ClassVar

from pufferlib.ocean.benchmark.evaluators.human_replay import HumanReplayEvaluator


class BehaviorClassEvaluator(HumanReplayEvaluator):
    type_name: ClassVar[str] = "behavior_class"

    def __init__(self, name, config, train_config):
        super().__init__(name, config, train_config)
        self._sampled_dir = None  # tmp symlink dir created per pass

    def env_overrides(self) -> dict:
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
            n = num_scenarios
        else:
            n = len(all_bins)
        env["num_agents"] = n
        env["num_maps"] = n
        env["num_eval_scenarios"] = n
        return env

    def cleanup(self):
        if self._sampled_dir and os.path.isdir(self._sampled_dir):
            shutil.rmtree(self._sampled_dir, ignore_errors=True)
            self._sampled_dir = None
