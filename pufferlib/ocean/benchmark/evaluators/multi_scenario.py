"""MultiScenarioEvaluator — gigaflow validation eval. C-side eval_mode
cycles maps sequentially in one batched rollout, so the base loop +
PufferEnv defaults handle parallelism without multi-process workers."""

from typing import ClassVar

from pufferlib.ocean.benchmark.evaluators.base import Evaluator


class MultiScenarioEvaluator(Evaluator):
    type_name: ClassVar[str] = "multi_scenario"

    def env_overrides(self) -> dict:
        env = {
            "eval_mode": 1,
            "termination_mode": 0,
            "reward_randomization": False,
        }
        env.update(self.config.get("env", {}))
        # Replay scenarios are unique: make C eval_mode sweep exactly
        # num_scenarios distinct maps (its default, 16, would re-evaluate only
        # the first 16). Gigaflow maps cycle, so leave its default alone.
        # Skip when the section pins num_eval_scenarios explicitly.
        if env.get("simulation_mode") == "replay" and "num_eval_scenarios" not in self.config.get("env", {}):
            env["num_eval_scenarios"] = int(self.config.get("eval", {}).get("num_scenarios", env.get("num_maps", 1)))
        return env

    def _should_stop(self, args, infos_collected, steps) -> bool:
        target = int(self.config.get("eval", {}).get("num_scenarios", 1))
        eval_cfg = self.config.get("eval", {})
        # When per-episode summaries are being collected (CSV / coverage), count
        # actual completed episodes — they map 1:1 to scenarios. Otherwise count
        # my_log emissions (the legacy behaviour for evaluators without them).
        if eval_cfg.get("export_episode_csv") or eval_cfg.get("verify_coverage"):
            return len(getattr(self, "_episode_rows", [])) >= target
        return len(infos_collected) >= target

    def _render_env_overrides(self, args) -> dict:
        # Random starting_map per render epoch — every epoch shows a
        # different bin from the dir rather than the same alphabetical
        # first-N. Pin by setting env.starting_map explicitly in the
        # [eval.<name>] section.
        import random

        out = super()._render_env_overrides(args)
        if "starting_map" not in self.config.get("env", {}):
            num_maps = int(out.get("num_maps", 1))
            if num_maps > 1:
                out["starting_map"] = random.randint(0, num_maps - 1)
        return out
