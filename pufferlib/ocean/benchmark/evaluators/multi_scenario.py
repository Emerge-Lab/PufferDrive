"""MultiScenarioEvaluator — distribute scenarios across workers, one rollout
per scenario, mean per-scenario metrics."""

import contextlib
import os
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

import pufferlib
from pufferlib.ocean.benchmark.evaluators.base import EvalResult, Evaluator


class MultiScenarioEvaluator(Evaluator):
    type_name = "multi_scenario"

    def vec_overrides(self) -> dict:
        # Multi-worker by default for throughput. Override via [eval.<name>.vec].
        backend = self.train_config.get("vec", {}).get("backend", "PufferEnv")
        num_envs = int(self.config.get("vec", {}).get("num_envs", 1))
        return {"backend": backend, "num_envs": num_envs}

    def env_overrides(self) -> dict:
        # Sensible defaults for the gigaflow path; replay configs are expected
        # to set the relevant knobs in [eval.<name>.env.*].
        env = {
            "eval_mode": 1,
            "termination_mode": 0,
            "reward_randomization": False,
        }
        env.update(self.config.get("env", {}))
        return env

    def rollout(self, vecenv, policy, args) -> EvalResult:
        t0 = time.time()
        num_scenarios = int(self.config.get("eval", {}).get("num_scenarios", 1))
        scenario_length = int(args["env"].get("scenario_length", 91))
        device = args["train"]["device"]
        num_agents = vecenv.observation_space.shape[0]

        global_infos = {}

        # LSTM hidden state shared across the rollout; reset each scenario batch.
        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        vecenv.async_reset(args.get("seed", 42))
        ob, _, _, _, infos, _, _ = vecenv.recv()
        scenarios_processed = 0
        with tqdm.tqdm(total=num_scenarios, desc=f"[{self.name}] scenarios", disable=args.get("quiet", False)) as pbar:
            while scenarios_processed < num_scenarios:
                if args["train"]["use_rnn"]:
                    state["lstm_h"].zero_()
                    state["lstm_c"].zero_()

                for _ in range(scenario_length):
                    with torch.no_grad():
                        ob_t = torch.as_tensor(ob).to(device)
                        logits, _ = policy.forward_eval(ob_t, state)
                        action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                        action = action.cpu().numpy().reshape(vecenv.action_space.shape)
                    if isinstance(logits, torch.distributions.Normal):
                        action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

                    ob, _, _, _, infos = vecenv.step(action)

                    if infos and infos[0]:
                        for sub_env in infos:
                            for env_idx, summary in enumerate(sub_env):
                                map_name = summary["map_name"].split("/")[-1].split(".")[0]
                                summary["episode_id"] = env_idx
                                summary["map_name"] = map_name
                                scenarios_processed += 1
                                pbar.update(1)
                                for k, v in summary.items():
                                    global_infos.setdefault(k, []).append(v)

        metrics = self._average(global_infos)
        if not args.get("quiet", False):
            print(f"[{self.name}] {scenarios_processed} scenarios in {time.time() - t0:.1f}s")

        frames = []
        if self.render:
            frames = self._render_pass(vecenv, policy, args)

        return EvalResult(metrics=metrics, frames=frames)

    def _average(self, global_infos: dict) -> dict:
        out = {}
        import numbers

        for k, vs in global_infos.items():
            if k == "num_scenarios":
                out[k] = float(np.sum(vs))
            elif vs and isinstance(vs[0], numbers.Number):
                out[k] = float(np.mean(vs))
        return out

    def _render_pass(self, vecenv, policy, args) -> list:
        """One rollout per view, all writing mp4s to a single dir.

        Re-uses the same vecenv if it's a single-worker setup; otherwise
        delegates to a serial render env built fresh per view.
        """
        import importlib

        env_name = args["env_name"]
        backend = args.get("render_backend", "egl")
        if backend != "egl":
            return []

        out_dir = Path(args.get("render_results_dir") or args.get("eval_results_dir") or ".") / "mp4"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Render with a fresh single-worker env so frame capture is sequential
        # and starting_map_counter starts at 0. Multi-worker render doesn't
        # match the C-side ffmpeg-per-env wiring cleanly.
        package = args.get("package", "ocean")
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(env_name)

        render_env_kwargs = dict(args["env"])
        render_env_kwargs["render_mode"] = "headless"

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
        device = args["train"]["device"]
        num_agents = vecenv.observation_space.shape[0]
        num_scenarios = int(self.config.get("eval", {}).get("num_scenarios", 1))
        max_steps = args.get("render_max_steps") or int(args["env"].get("scenario_length", 91))

        saved_cwd = os.getcwd()
        os.chdir(out_dir)
        try:
            state = {}
            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                )
            scenarios_processed = 0
            while scenarios_processed < num_scenarios:
                ob, _ = vecenv.reset()
                scenarios = vecenv.get_state()
                num_in_batch = len(scenarios)
                remaining = num_scenarios - scenarios_processed - num_in_batch
                target_env.batch_size_eval = max(1, remaining)
                if args["train"]["use_rnn"]:
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
