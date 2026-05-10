"""MultiScenarioEvaluator — distribute scenarios across workers, one rollout
per scenario, mean per-scenario metrics. Drives both the gigaflow validation
path and replay-style multi-scenario evals.

Inherits the default loop from `Evaluator`; overrides `_should_stop` (cap by
scenario count), `_initial_reset` (async reset for multi-worker throughput),
`_maybe_reset_lstm` (per-scenario LSTM reset), and `_render_pass` (the C-side
EGL → ffmpeg mp4 dump)."""

import os
from pathlib import Path
from typing import ClassVar

from pufferlib.ocean.benchmark.evaluators.base import Evaluator


class MultiScenarioEvaluator(Evaluator):
    type_name: ClassVar[str] = "multi_scenario"

    def vec_overrides(self) -> dict:
        # Multi-worker by default for throughput. Override via [eval.<name>.vec].
        backend = self.train_config.get("vec", {}).get("backend", "PufferEnv")
        num_envs = int(self.config.get("vec", {}).get("num_envs", 1))
        return {"backend": backend, "num_envs": num_envs}

    def env_overrides(self) -> dict:
        env = {
            "eval_mode": 1,
            "termination_mode": 0,
            "reward_randomization": False,
        }
        env.update(self.config.get("env", {}))
        return env

    # -- Loop hooks --

    def _initial_reset(self, vecenv, args):
        # Multi-worker async reset gives us the parallel-throughput path.
        vecenv.async_reset(args.get("seed", 42))
        ob, _, _, _, _, _, _ = vecenv.recv()
        return ob

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

    # -- Render --

    def _render_pass(self, vecenv, policy, args) -> list:
        """One rollout per view, all writing mp4s to a single dir.

        Builds a fresh single-worker env per view (C-side ffmpeg-per-env
        wiring assumes one bin at a time per process). Render budget and
        starting position are independent of the metric pass:

          eval.render_num_scenarios — how many scenarios to render. Defaults
              to min(eval.num_scenarios, 3). Always respected over
              num_scenarios so renders stay cheap.
          starting_map — randomized per render epoch so successive epochs
              show different scenarios from the dir, not the same first-N
              alphabetically. Set explicitly in env.* to pin.
        """
        import importlib
        import random

        import pufferlib

        backend = args.get("render_backend", "egl")
        if backend != "egl":
            return []

        env_name = args["env_name"]
        out_dir = Path(args.get("render_results_dir") or args.get("eval_results_dir") or ".") / "mp4"
        out_dir.mkdir(parents=True, exist_ok=True)

        package = args.get("package", "ocean")
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(env_name)

        render_env_kwargs = dict(args["env"])
        render_env_kwargs["render_mode"] = "headless"

        # Random starting map per render epoch — every epoch shows a
        # different bin from the directory rather than the first N
        # alphabetically. The user can pin by setting env.starting_map
        # explicitly in the [eval.<name>] section.
        if "starting_map" not in self.config.get("env", {}):
            num_maps = int(render_env_kwargs.get("num_maps", 1))
            if num_maps > 1:
                render_env_kwargs["starting_map"] = random.randint(0, num_maps - 1)

        all_paths = []
        for view in self.render_views:
            view_idx = _VIEW_NAME_TO_IDX.get(view, 0)
            view_suffix = "" if view == "sim_state" else f"_{view}"

            vec = pufferlib.vector.make(
                [make_env],
                env_args=[[]],
                env_kwargs=[render_env_kwargs],
                backend="PufferEnv",
                num_envs=1,
                num_workers=1,
                batch_size=1,
            )
            target = vec if not hasattr(vec, "envs") else vec.envs[0]
            internal = getattr(target, "num_envs", 1)
            for e in range(internal):
                target.set_video_suffix(view_suffix, env_idx=e)

            paths = self._render_view(vec, target, policy, args, view_idx, out_dir)
            vec.close()
            all_paths.extend(paths)
        return all_paths

    def _render_view(self, vecenv, target_env, policy, args, view_idx: int, out_dir: Path) -> list:
        import numpy as np
        import torch

        import pufferlib

        device = args["train"]["device"]
        num_agents = vecenv.observation_space.shape[0]
        # Render budget defaults to min(num_scenarios, 3) if not set explicitly.
        # Renders are expensive (mp4 encode + wandb upload) so we don't want
        # them at metric-pass scale.
        eval_cfg = self.config.get("eval", {})
        metric_count = int(eval_cfg.get("num_scenarios", 1))
        num_scenarios = int(eval_cfg.get("render_num_scenarios", min(metric_count, 3)))
        max_steps = args.get("render_max_steps") or int(args["env"].get("scenario_length", 91))

        saved_cwd = os.getcwd()
        os.chdir(out_dir)
        try:
            state = self._init_lstm_state(num_agents, policy, device, args)
            scenarios_processed = 0
            while scenarios_processed < num_scenarios:
                ob, _ = vecenv.reset()
                scenarios = vecenv.get_state()
                num_in_batch = len(scenarios)
                remaining = num_scenarios - scenarios_processed - num_in_batch
                target_env.batch_size_eval = max(1, remaining)
                if state:
                    state["lstm_h"].zero_()
                    state["lstm_c"].zero_()
                for _ in range(max_steps):
                    with torch.no_grad():
                        ob_t = torch.as_tensor(ob).to(device)
                        logits, _ = policy.forward_eval(ob_t, state)
                        action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                        action = action.cpu().numpy().reshape(vecenv.action_space.shape)
                    if isinstance(logits, torch.distributions.Normal):
                        action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)
                    ob, _, _, _, _ = vecenv.step(action)
                    for e in range(num_in_batch):
                        target_env.render(env_idx=e, view_mode=view_idx)
                for e in range(num_in_batch):
                    target_env.close_client(env_idx=e)
                scenarios_processed += num_in_batch
        finally:
            os.chdir(saved_cwd)

        return sorted(p for p in out_dir.glob("*.mp4"))


_VIEW_NAME_TO_IDX = {
    "sim_state": 0,
    "bev": 1,
    "topdown_sim": 2,
    "bev_all": 3,
}
