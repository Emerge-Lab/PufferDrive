"""Evaluator base class + default rollout loop + EvalResult dataclass."""

import time
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class EvalResult:
    metrics: dict
    frames: list = field(default_factory=list)


class Evaluator:
    """Base class for all evaluators.

    Subclasses typically override only `_should_stop` (the loop termination
    condition) and `env_overrides`. The default `rollout` runs a step loop
    suitable for "stream of episode infos until target count reached" evals.

    To diverge from the default loop entirely, override `rollout` directly.
    """

    type_name: ClassVar[str] = ""

    def __init__(self, name: str, config: dict, train_config: dict):
        # `name` = the [eval.<name>] section name. Used as the wandb prefix.
        self.name = name
        # `config` = merged per-evaluator config (after inheritance + clean
        # macro expansion). Has nested `env`, `vec`, plus flat scalar knobs.
        self.config = config
        # `train_config` = the full training config from drive.ini, used as
        # the base layer that `config` overrides on top of.
        self.train_config = train_config

        # Common scalars pulled out for ergonomics.
        self.enabled: bool = bool(config.get("enabled", True))
        self.interval: int = int(config.get("interval", 0))
        self.mode: str = config.get("mode", "inline")
        self.render: bool = bool(config.get("render", False))
        self.render_views: list = list(config.get("render_views", ["sim_state"]))
        self.clean: bool = bool(config.get("clean", True))

    # -- Config hooks ---------------------------------------------------

    def env_overrides(self) -> dict:
        """Per-evaluator [env] overrides. Defaults to whatever the section
        wrote under `env.*`. Subclasses can override to add baseline knobs."""
        return dict(self.config.get("env", {}))

    def vec_overrides(self) -> dict:
        """Per-evaluator [vec] overrides. Default: serial single-worker —
        the safe default for replay-style evals where each worker is a
        single bin replay. Subclasses that want parallel throughput
        (gigaflow validation) override this."""
        base = {"backend": "PufferEnv", "num_envs": 1}
        base.update(self.config.get("vec", {}))
        return base

    # -- Rollout (default) ----------------------------------------------

    def rollout(self, vecenv, policy, args) -> EvalResult:
        """Default rollout: reset → step → collect infos → aggregate.

        Times the inner work and adds `eval_seconds` to metrics so wandb
        panels show wall-clock cost per evaluator. Subclasses tune
        behavior by overriding `_run_rollout_loop` (and optionally
        `_render_pass`); only override this method if the loop shape
        itself needs to differ.
        """
        t0 = time.time()
        metrics = self._run_rollout_loop(vecenv, policy, args)
        frames = self._render_pass(vecenv, policy, args) if self.render else []
        metrics["eval_seconds"] = float(time.time() - t0)
        return EvalResult(metrics=metrics, frames=frames)

    def _run_rollout_loop(self, vecenv, policy, args) -> dict:
        import numpy as np
        import torch

        import pufferlib

        device = args["train"]["device"]
        num_agents = vecenv.observation_space.shape[0]
        state = self._init_lstm_state(num_agents, policy, device, args)

        obs = self._initial_reset(vecenv, args)

        infos_collected: list = []
        steps = 0
        while not self._should_stop(args, infos_collected, steps):
            self._maybe_reset_lstm(state, steps, args)

            with torch.no_grad():
                ob_t = torch.as_tensor(obs).to(device)
                logits, _ = policy.forward_eval(ob_t, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                action = action.cpu().numpy().reshape(vecenv.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

            obs, _, _, _, infos = vecenv.step(action)
            infos_collected.extend(self._flatten_infos(infos))
            steps += 1

        return self._aggregate_infos(infos_collected)

    # -- Loop hooks (subclass-overridable) ------------------------------

    def _initial_reset(self, vecenv, args):
        """Return the initial observation. Default: synchronous reset."""
        obs, _ = vecenv.reset()
        return obs

    def _init_lstm_state(self, num_agents, policy, device, args) -> dict:
        if not args["train"].get("use_rnn"):
            return {}
        import torch

        return dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    def _maybe_reset_lstm(self, state, steps, args):
        """Hook for resetting LSTM state mid-rollout. Default: no-op."""
        pass

    def _should_stop(self, args, infos_collected, steps) -> bool:
        """Loop termination. Subclasses must override."""
        raise NotImplementedError

    def _flatten_infos(self, infos) -> list:
        """Pufferlib backends return either a list-of-list (multi-worker) or
        a single list (PufferEnv backend). Flatten to a list of dicts."""
        out = []
        if not infos:
            return out
        for sub in infos:
            if not sub:
                continue
            if isinstance(sub, list):
                out.extend(sub)
            else:
                out.append(sub)
        return out

    def _aggregate_infos(self, infos: list) -> dict:
        """Default: numeric mean per key, plus a num_scenarios_completed count."""
        if not infos:
            return {"num_scenarios_completed": 0}
        import numpy as np

        out = {"num_scenarios_completed": float(len(infos))}
        keys = set().union(*(d.keys() for d in infos))
        for k in keys:
            vals = [d[k] for d in infos if isinstance(d.get(k), (int, float))]
            if vals:
                out[k] = float(np.mean(vals))
        return out

    # -- Render (default EGL → ffmpeg mp4 pipeline) ----------------------

    def _render_pass(self, vecenv, policy, args) -> list:
        """Build a fresh PufferEnv with `render_mode=headless`, render one
        clip per (scenario, view), return mp4 paths. Returns [] for non-egl
        backends. Subclasses customize the render env via `_render_env_overrides`.
        """
        backend = args.get("render_backend", "egl")
        if backend != "egl":
            return []

        import importlib
        from pathlib import Path

        import pufferlib

        # Per-evaluator subdir so each evaluator's mp4s don't get re-globbed
        # by the next evaluator's _render_view (every evaluator runs at the
        # same global_step, so a shared dir + step glob would collect every
        # earlier evaluator's mp4s into this one's result.frames).
        out_dir = Path(args.get("render_results_dir") or args.get("eval_results_dir") or ".") / "mp4" / self.name
        out_dir.mkdir(parents=True, exist_ok=True)

        package = args.get("package", "ocean")
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(args["env_name"])

        render_env_kwargs = self._render_env_overrides(args)
        # Stamp the training step into the filename so successive epochs
        # produce distinct mp4s and wandb's render carousel shows policy
        # evolution. global_step falls back to 0 for ad-hoc CLI runs.
        step_suffix = f"_step{int(args.get('global_step') or 0)}"

        all_paths = []
        for view in self.render_views:
            view_idx = _VIEW_NAME_TO_IDX.get(view, 0)
            view_suffix = step_suffix + ("" if view == "sim_state" else f"_{view}")

            vec = pufferlib.vector.make(
                make_env,
                env_args=[],
                env_kwargs=render_env_kwargs,
                backend="PufferEnv",
                num_envs=1,
            )
            target = vec if not hasattr(vec, "envs") else vec.envs[0]
            internal = getattr(target, "num_envs", 1)
            for e in range(internal):
                target.set_video_suffix(view_suffix, env_idx=e)

            paths = self._render_view(vec, target, policy, args, view_idx, out_dir, step_suffix)
            vec.close()
            all_paths.extend(paths)
        return all_paths

    def _render_env_overrides(self, args) -> dict:
        """Build env kwargs for the render env. Default: same as the
        metric-pass env plus `render_mode=headless`. Subclasses override
        to inject things like a random starting_map (gigaflow validation)
        or a shrunken bin set (behavior class)."""
        out = dict(args["env"])
        out["render_mode"] = "headless"
        return out

    def _render_view(self, vecenv, target_env, policy, args, view_idx, out_dir, step_suffix) -> list:
        """One rollout per render-env, writes one mp4 per active env per view.
        Caps how many internal envs actually feed ffmpeg pipes via
        `eval.render_num_scenarios` so render cost stays bounded."""
        import os

        import numpy as np
        import torch

        import pufferlib

        device = args["train"]["device"]
        num_agents = vecenv.observation_space.shape[0]

        eval_cfg = self.config.get("eval", {})
        metric_count = int(eval_cfg.get("num_scenarios", 1))
        num_scenarios = int(eval_cfg.get("render_num_scenarios", min(metric_count, 3)))
        # Render-clip length: independent of scenario_length (which is the
        # metric-pass length). At 30 fps, 300 steps = ~10s mp4. Per-step EGL
        # render is the bottleneck (~3 fps wall-clock at 1080p), so keeping
        # this small directly bounds the render-pass runtime.
        max_steps = int(eval_cfg.get("render_max_steps", 300))

        saved_cwd = os.getcwd()
        os.chdir(out_dir)
        # Glob for files written this pass: every mp4 has the step suffix,
        # so a step_suffix-prefixed glob filters out accumulated mp4s from
        # prior epochs (the dir is shared across runs).
        step_glob = f"*{step_suffix}*.mp4"
        try:
            state = self._init_lstm_state(num_agents, policy, device, args)
            scenarios_processed = 0
            while scenarios_processed < num_scenarios:
                ob, _ = vecenv.reset()
                scenarios = vecenv.get_state()
                num_in_batch = len(scenarios)
                # Cap how many envs render this iteration: the C kernel
                # steps the full batch regardless, but only the first
                # `to_render` envs feed ffmpeg pipes.
                to_render = min(num_in_batch, num_scenarios - scenarios_processed)
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
                    for e in range(to_render):
                        target_env.render(env_idx=e, view_mode=view_idx)
                for e in range(to_render):
                    target_env.close_client(env_idx=e)
                scenarios_processed += to_render
        finally:
            os.chdir(saved_cwd)

        return sorted(out_dir.glob(step_glob))


_VIEW_NAME_TO_IDX = {
    "sim_state": 0,
    "bev": 1,
    "topdown_sim": 2,
    "bev_all": 3,
}
