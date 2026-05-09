"""WOSACEvaluator — Waymo Open Sim Agents Challenge realism eval.

Wraps the existing WOSACEvaluator class in benchmark/evaluator.py — that
file owns the realism math (per-feature likelihood under learned
estimators) and the per-scene multi-rollout structure. This adapter
fits it into the unified Evaluator interface.
"""

from typing import ClassVar

from pufferlib.ocean.benchmark.evaluators.base import EvalResult, Evaluator


class WOSACEvaluator(Evaluator):
    type_name: ClassVar[str] = "wosac"

    def env_overrides(self) -> dict:
        env = {
            "control_mode": "control_wosac",
            "init_mode": "create_all_valid",
            "eval_mode": 1,
            "termination_mode": 0,
            "reward_randomization": False,
        }
        env.update(self.config.get("env", {}))
        return env

    def rollout(self, vecenv, policy, args) -> EvalResult:
        # Inner class pulls pandas/matplotlib — keep the import inside the
        # rollout so the wrapper class can be imported in environments
        # that don't have those (e.g. unit-test smoke envs).
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator as _WOSACInner

        inner = _WOSACInner(args)
        df = inner.evaluate(args, vecenv, policy)
        # df has one row per scene; aggregate to a single dict.
        results = df.mean(numeric_only=True).to_dict()
        results["total_num_agents"] = float(df["num_agents_per_scene"].sum())
        results["total_unique_scenarios"] = float(df.index.unique().shape[0])
        results["realism_meta_score_std"] = float(df["realism_meta_score"].std())
        results = {k: (float(v) if hasattr(v, "item") else v) for k, v in results.items()}
        return EvalResult(metrics=results, frames=[])
