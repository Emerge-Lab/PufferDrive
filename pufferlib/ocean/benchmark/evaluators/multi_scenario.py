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
        return env

    def _maybe_reset_lstm(self, state, steps, args):
        # Reset between scenarios — gigaflow's auto-resample fires at the
        # end of scenario_length, so steps % scenario_length == 0 is the
        # natural boundary. No-op when LSTM is unused.
        if not state or steps == 0:
            return
        scenario_length = int(args["env"].get("scenario_length", 0))
        if scenario_length > 0 and steps % scenario_length == 0:
            state["lstm_h"].zero_()
            state["lstm_c"].zero_()

    def _should_stop(self, args, infos_collected, steps) -> bool:
        target = int(self.config.get("eval", {}).get("num_scenarios", 1))
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
