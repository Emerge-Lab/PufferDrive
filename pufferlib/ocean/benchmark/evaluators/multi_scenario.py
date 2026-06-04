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
        # terminate_on_goal makes episodes variable-length: a raw episode count
        # lets fast (e.g. parked-SDC) scenarios reset and cycle, crowding out
        # slow ones before they finish even once. Stop instead once every
        # distinct scenario has produced its first episode — each map gets its
        # own env in eval replay, and scenario_length truncation guarantees one
        # episode per scenario within ~scenario_length steps.
        if args["env"].get("terminate_on_goal"):
            num_maps = int(args["env"].get("num_maps") or 0)
            goal = min(target, num_maps) if num_maps else target
            seen = {r.get("map_name") for r in getattr(self, "_episode_rows", []) if r.get("map_name")}
            return len(seen) >= goal
        # When per-episode summaries are being collected (CSV / coverage), count
        # actual completed episodes — they map 1:1 to scenarios. Otherwise count
        # my_log emissions (the legacy behaviour for evaluators without them).
        if eval_cfg.get("export_episode_csv") or eval_cfg.get("verify_coverage"):
            return len(getattr(self, "_episode_rows", [])) >= target
        return len(infos_collected) >= target

    def _finalize_metrics(self, args, infos_collected) -> dict:
        # Under terminate_on_goal the vec_log stream can't aggregate
        # asynchronously-terminating episodes (its harvest is gated on env[0]),
        # so aggregate the first completed episode of each distinct scenario.
        if not args["env"].get("terminate_on_goal"):
            return super()._finalize_metrics(args, infos_collected)
        first = {}
        for row in getattr(self, "_episode_rows", []):
            key = row.get("map_name") or row.get("scenario_id")
            if key is not None and key not in first:
                first[key] = row
        return self._aggregate_episode_rows(list(first.values()))

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
