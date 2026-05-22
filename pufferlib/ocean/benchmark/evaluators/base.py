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

        Switches the policy to `.eval()` for the duration so dropout /
        batchnorm / etc. behave deterministically; restores the prior
        mode afterward so subsequent training is unaffected.

        Times the metric pass and the render pass separately and reports
        both alongside the total. Render is EGL+ffmpeg-bound and varies
        wildly with `render_max_steps`/`render_num_scenarios`; lumping
        it with the policy-driven metric pass hides where time is going.

          metric_seconds  — _run_rollout_loop wall time (policy + env step)
          render_seconds  — _render_pass wall time (0.0 if render=false)
          eval_seconds    — total = metric + render
        """
        prev_training = getattr(policy, "training", None)
        if prev_training is not None:
            policy.eval()
        t0 = time.time()
        try:
            metrics = self._run_rollout_loop(vecenv, policy, args)
            t_metric = time.time()
            frames = self._render_pass(vecenv, policy, args) if self.render else []
            t_render = time.time()
        finally:
            if prev_training:
                policy.train()
        metrics["metric_seconds"] = float(t_metric - t0)
        metrics["render_seconds"] = float(t_render - t_metric)
        metrics["eval_seconds"] = float(t_render - t0)
        # Opt-in per-episode CSV + coverage check (writes files, folds
        # coverage_* scalars into metrics). No-op unless the evaluator set
        # eval.export_episode_csv / eval.verify_coverage.
        self._maybe_export_episodes(args, metrics)
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
        # Per-episode `completed_episode` summaries, collected only when an
        # evaluator opts into emit_completed_episodes (via the manager) for the
        # CSV / coverage features. Kept out of `infos_collected` so the default
        # my_log weighted-mean aggregation is unaffected. Stored on self so
        # `_should_stop` can count episodes as they complete.
        self._episode_rows = []
        steps = 0
        # Stall backstop: if the env stops producing episodes/emissions (e.g.
        # eval_mode exhausted its map sweep) the stop target may never be met,
        # so bail after a few scenario-lengths of no progress.
        scenario_length = int(args["env"].get("scenario_length", 0) or 0)
        stall_limit = 3 * scenario_length if scenario_length else 0
        last_progress = 0
        stall_steps = 0
        while not self._should_stop(args, infos_collected, steps):
            with torch.no_grad():
                ob_t = torch.as_tensor(obs).to(device)
                logits, _ = policy.forward_eval(ob_t, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                action = action.cpu().numpy().reshape(vecenv.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

            obs, _, terminals, truncations, infos = vecenv.step(action)
            for d in self._flatten_infos(infos):
                if isinstance(d, dict) and d.get("summary_type") == "completed_episode":
                    self._episode_rows.append(d)
                else:
                    infos_collected.append(d)
            # Mask LSTM state per-agent for envs that just terminated or
            # truncated — those agents' next obs is from a fresh scenario
            # and the recurrent memory of the previous one would bias
            # the policy. Either signal alone means "episode over, env
            # reset," so OR them.
            if state:
                done = np.asarray(terminals).astype(bool) | np.asarray(truncations).astype(bool)
                mask = torch.as_tensor(~done, device=device, dtype=state["lstm_h"].dtype).reshape(-1, 1)
                state["lstm_h"] *= mask
                state["lstm_c"] *= mask

            progress = len(infos_collected) + len(self._episode_rows)
            if progress > last_progress:
                last_progress = progress
                stall_steps = 0
            elif stall_limit:
                stall_steps += 1
                if stall_steps >= stall_limit:
                    print(
                        f"[eval.{self.name}] no new episodes for {stall_limit} steps; "
                        f"stopping at {len(self._episode_rows)} episode(s)."
                    )
                    break
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
        """Weighted per-agent mean across vec_log emissions.

        Each emission from vec_log is already a per-agent mean over the
        envs that finished in that batch, plus an `n` field carrying the
        total agent count behind that mean. So `sum(d[k] * d["n"]) / sum(d["n"])`
        recovers the true per-agent global mean — regardless of whether
        emissions had identical batch sizes (uniform truncation) or
        varying sizes (mixed scenario_lengths / early_reset / mid-run
        agent removals).

        Reported counts:
          - `num_log_cycles`     — vec_log emissions seen (1 cycle each).
          - `num_agents_evaluated` — total agent-trajectories behind the
            metric (sum of n across emissions). This is what readers
            usually want when they ask "how many scenes did we eval?";
            for behaviors it's the bin count, for gigaflow it's
            cycles × envs × agents-per-env.

        Caveat: vec_log already divides every numeric field by n. For
        ratio fields where the numerator and denominator are themselves
        per-agent sums (e.g. avg_distance_per_infraction in my_log),
        weighted mean across emissions only approximates the true ratio
        — that would require the parent to see numerator/denominator
        separately. Same approximation as the prior mean-of-means; just
        making the math correct for the majority of per-agent fields.
        """
        if not infos:
            return {"num_log_cycles": 0, "num_agents_evaluated": 0.0}

        total_n = sum(float(d.get("n", 1)) for d in infos)
        out = {
            "num_log_cycles": float(len(infos)),
            "num_agents_evaluated": float(total_n),
        }
        keys = set().union(*(d.keys() for d in infos))
        keys.discard("n")  # reported separately; not a metric
        for k in keys:
            num = 0.0
            den = 0.0
            for d in infos:
                v = d.get(k)
                if isinstance(v, (int, float)):
                    w = float(d.get("n", 1))
                    num += float(v) * w
                    den += w
            if den > 0:
                out[k] = num / den
        return out

    # -- Per-episode CSV + scenario coverage (opt-in) -------------------

    def _maybe_export_episodes(self, args, metrics) -> None:
        """Write a per-episode metrics CSV and/or a scenario-coverage report.

        Both are off by default and enabled per-evaluator via the
        [eval.<name>] section:

            eval.export_episode_csv = true
            eval.verify_coverage    = true

        Either flag makes the manager turn on `emit_completed_episodes` for
        this evaluator's env, so the rollout collects one `completed_episode`
        summary per finished episode into `self._last_episode_rows`. The
        default weighted-mean metric path is untouched.
        """
        from pathlib import Path

        eval_cfg = self.config.get("eval", {})
        want_csv = bool(eval_cfg.get("export_episode_csv", False))
        want_coverage = bool(eval_cfg.get("verify_coverage", False))
        if not (want_csv or want_coverage):
            return

        rows = list(getattr(self, "_episode_rows", []) or [])
        out_dir = Path(args.get("eval_results_dir") or args.get("render_results_dir") or ".") / "episode_metrics"
        epoch = int(args.get("epoch") or 0)
        global_step = int(args.get("global_step") or 0)
        suffix = f"_epoch{epoch}_step{global_step}"

        if want_csv:
            self._write_episode_csv(rows, out_dir, suffix)

        if want_coverage:
            cov = self._coverage_report(rows, eval_cfg)
            metrics["coverage_expected"] = float(cov["expected"])
            metrics["coverage_found"] = float(cov["found"])
            metrics["coverage_unique_maps"] = float(cov["unique"])
            metrics["coverage_complete"] = float(cov["complete"])
            if not cov["complete"]:
                print(
                    f"[eval.{self.name}] coverage: evaluated {cov['found']} episode(s) "
                    f"across {cov['unique']} unique map(s), expected {cov['expected']}."
                )
            if cov["duplicates"]:
                top = sorted(cov["duplicates"].items(), key=lambda kv: -kv[1])[:5]
                print(f"[eval.{self.name}] {len(cov['duplicates'])} map(s) evaluated more than once; top: {top}")

    def _write_episode_csv(self, rows, out_dir, suffix) -> None:
        """One row per finished episode, all summary fields. Identity columns
        (episode_index / scenario_id / map_name) lead when present."""
        if not rows:
            print(f"[eval.{self.name}] export_episode_csv set but no per-episode summaries were collected.")
            return
        import pandas as pd

        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        lead = [c for c in ("episode_index", "scenario_id", "map_name") if c in df.columns]
        df = df[lead + [c for c in df.columns if c not in lead]]
        path = out_dir / f"{self.name}{suffix}.csv"
        df.to_csv(path, index=False)
        print(f"[eval.{self.name}] wrote {len(df)} per-episode rows to {path}")

    def _coverage_report(self, rows, eval_cfg) -> dict:
        """Coverage of the scenario set: episode count, unique maps, duplicates.

        `expected` is the evaluator's `num_scenarios` target when set, else
        `env.num_maps`, else however many episodes were collected. `found` is
        the number of `completed_episode` summaries seen and `unique` the
        number of distinct maps among them (by basename). `duplicates` maps
        each repeated scenario to its count — meaningful for unique-scenario
        sweeps (replay), expected-and-harmless when maps cycle (gigaflow).
        """
        import os
        from collections import Counter

        found = len(rows)
        expected = eval_cfg.get("num_scenarios")
        if expected is None:
            expected = self.config.get("env", {}).get("num_maps", found)
        expected = int(expected or found)

        names = []
        for r in rows:
            ident = r.get("map_name") or r.get("scenario_id")
            if ident:
                names.append(os.path.basename(str(ident)).split(".")[0])
        counts = Counter(names)
        duplicates = {n: c for n, c in counts.items() if c > 1}
        unique = len(counts)
        # For a unique-scenario sweep (the target fits within the loaded map
        # set, e.g. replay) completeness means we covered that many *distinct*
        # maps; when maps cycle (expected > num_maps, e.g. gigaflow) it means
        # we ran that many episodes.
        num_maps = int(self.config.get("env", {}).get("num_maps", 0) or 0)
        measure = unique if (num_maps and expected <= num_maps) else found
        return {
            "expected": expected,
            "found": found,
            "unique": unique,
            "complete": measure >= expected,
            "duplicates": duplicates,
        }

    # -- Render (default EGL → ffmpeg mp4 pipeline) ----------------------

    def _render_pass(self, vecenv, policy, args) -> list:
        """Build a fresh PufferEnv with `render_mode=headless`, render one
        clip per (scenario, view), return mp4 paths (egl) or html paths (html).
        Subclasses customize the render env via `_render_env_overrides`.
        """
        # Observation render (opt-in): an interactive HTML per scenario that
        # also shows each agent's unpacked observation. CPU-only, so it takes
        # precedence over the EGL/html backends when eval.render_obs is set.
        if self.config.get("eval", {}).get("render_obs", False):
            return self._render_pass_obs(vecenv, policy, args)

        backend = args.get("render_backend", "egl")
        if backend == "html":
            return self._render_pass_html(vecenv, policy, args)
        if backend not in ["egl", "html"]:
            raise ValueError(f"Evaluator: _render_pass only accepts egl, html backends but got {backend} instead.")

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
        # Stamp epoch + training step into the filename so successive
        # epochs produce distinct mp4s and wandb's render carousel shows
        # policy evolution. Epoch is the human-readable index ("which
        # checkpoint did this come from"); global_step is the precise
        # env-step count. Both fall back to 0 for ad-hoc CLI runs.
        epoch = int(args.get("epoch") or 0)
        global_step = int(args.get("global_step") or 0)
        step_suffix = f"_epoch{epoch}_step{global_step}"

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

            paths = self._render_view(vec, target, policy, args, view_idx, out_dir, view_suffix)
            vec.close()
            all_paths.extend(paths)
        return all_paths

    def _render_pass_html(self, vecenv, policy, args) -> list:
        """CPU-only HTML render path. Creates a fresh env with
        capture_compact_replay=True, runs one episode per requested scenario,
        and writes one .html file per episode via mining_viz."""
        import importlib
        import pickle
        import tempfile
        import zlib
        from pathlib import Path

        import numpy as np
        import torch

        import pufferlib
        from pufferlib import mining_viz

        eval_cfg = self.config.get("eval", {})
        for required in ("render_num_scenarios", "render_max_steps"):
            if required not in eval_cfg:
                raise KeyError(f"[eval.{self.name}] has render=true but eval.{required} is not set.")
        num_scenarios = int(eval_cfg["render_num_scenarios"])
        max_steps = int(eval_cfg["render_max_steps"])

        out_dir = Path(args.get("render_results_dir") or args.get("eval_results_dir") or ".") / "gif" / self.name
        out_dir.mkdir(parents=True, exist_ok=True)

        epoch = int(args.get("epoch") or 0)
        global_step = int(args.get("global_step") or 0)
        step_suffix = f"_epoch{epoch}_step{global_step}"

        package = args.get("package", "ocean")
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(args["env_name"])

        render_env_kwargs = self._render_env_overrides(args)
        render_env_kwargs["capture_compact_replay"] = True
        render_env_kwargs["emit_completed_episodes"] = True

        device = args["train"]["device"]
        html_paths = []
        scenarios_done = 0

        vec = pufferlib.vector.make(
            make_env,
            env_args=[],
            env_kwargs=render_env_kwargs,
            backend="PufferEnv",
            num_envs=1,
        )
        try:
            state = self._init_lstm_state(vec.observation_space.shape[0], policy, device, args)
            ob, _ = vec.reset()
            if state:
                state["lstm_h"].zero_()
                state["lstm_c"].zero_()

            for _ in range(max_steps * num_scenarios):
                with torch.no_grad():
                    ob_t = torch.as_tensor(ob).to(device)
                    logits, _ = policy.forward_eval(ob_t, state)
                    action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                    action = action.cpu().numpy().reshape(vec.action_space.shape)
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, vec.action_space.low, vec.action_space.high)

                ob, _, terminals, truncations, infos = vec.step(action)

                if state:
                    done = np.asarray(terminals).astype(bool) | np.asarray(truncations).astype(bool)
                    mask = torch.as_tensor(~done, device=device, dtype=state["lstm_h"].dtype).reshape(-1, 1)
                    state["lstm_h"] *= mask
                    state["lstm_c"] *= mask

                for info in infos or []:
                    bundle_bytes = info.get("compact_replay_bundle")
                    if bundle_bytes is None:
                        continue
                    scenario_id = info.get("scenario_id") or f"{scenarios_done:04d}"
                    map_name = info.get("map_name") or "map"
                    stem = f"{map_name}_{scenario_id}{step_suffix}"
                    tmp_path = out_dir / f"{stem}.pkl.zlib"
                    html_path = out_dir / f"{stem}.html"
                    tmp_path.write_bytes(bundle_bytes)
                    mining_viz.render_compact_replay_html(str(tmp_path), str(html_path))
                    tmp_path.unlink(missing_ok=True)
                    html_paths.append(html_path)
                    scenarios_done += 1
                    if scenarios_done >= num_scenarios:
                        break

                if scenarios_done >= num_scenarios:
                    break
        finally:
            vec.close()

        return html_paths

    def _render_pass_obs(self, vecenv, policy, args) -> list:
        """CPU-only observation render. Rolls out `render_num_scenarios`
        episodes and writes one interactive HTML per scenario via
        pufferlib.viz — including each agent's unpacked observation. Reads
        env state + the obs array, so it needs no EGL/ffmpeg."""
        import importlib
        import os
        from pathlib import Path

        import numpy as np
        import torch

        import pufferlib
        from pufferlib import viz

        eval_cfg = self.config.get("eval", {})
        for required in ("render_num_scenarios", "render_max_steps"):
            if required not in eval_cfg:
                raise KeyError(f"[eval.{self.name}] has render_obs but eval.{required} is not set.")
        num_scenarios = int(eval_cfg["render_num_scenarios"])
        max_steps = int(eval_cfg["render_max_steps"])

        out_dir = Path(args.get("render_results_dir") or args.get("eval_results_dir") or ".") / "obs" / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(args.get("epoch") or 0)
        global_step = int(args.get("global_step") or 0)
        step_suffix = f"_epoch{epoch}_step{global_step}"

        package = args.get("package", "ocean")
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(args["env_name"])

        render_env_kwargs = self._render_env_overrides(args)
        render_env_kwargs.pop("render_mode", None)  # obs viz reads state, no EGL

        device = args["train"]["device"]
        use_traj = "trajectory" in str(args["env"].get("action_type", ""))
        html_paths = []
        scenarios_done = 0

        vec = pufferlib.vector.make(make_env, env_args=[], env_kwargs=render_env_kwargs, backend="PufferEnv", num_envs=1)
        try:
            state = self._init_lstm_state(vec.observation_space.shape[0], policy, device, args)
            while scenarios_done < num_scenarios:
                ob, _ = vec.reset()
                scenarios = vec.get_state()
                n_in_batch = len(scenarios)
                to_render = min(n_in_batch, num_scenarios - scenarios_done)
                if state:
                    state["lstm_h"].zero_()
                    state["lstm_c"].zero_()
                agent_hist = [[] for _ in range(n_in_batch)]
                traffic_hist = [[] for _ in range(n_in_batch)]
                traj_hist = [[] for _ in range(n_in_batch)]
                obs_hist = [[] for _ in range(n_in_batch)]
                for t in range(max_steps):
                    cur = vec.get_state()
                    start_obs_index = 0
                    for e in range(n_in_batch):
                        sc = cur[e]
                        agent_hist[e].append(viz.fill_agents_state(sc, use_trajectory=use_traj))
                        traffic_hist[e].append(viz.fill_traffics_state(sc, t))
                        if use_traj:
                            traj_hist[e].append(viz.fill_trajectories(sc, t))
                        if e > 0:
                            start_obs_index += cur[e - 1]["active_agent_count"]
                        step_obs = {}
                        for a in range(sc["active_agent_count"]):
                            aid = int(sc["active_agent_indices"][a])
                            step_obs[aid] = viz.extract_obs_frame(
                                ob, sc, args, timestep=t, obs_index=start_obs_index + a, agent_idx=a, head_north=True
                            )
                        obs_hist[e].append(step_obs)
                    with torch.no_grad():
                        ob_t = torch.as_tensor(ob).to(device)
                        logits, _ = policy.forward_eval(ob_t, state)
                        action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                        action = action.cpu().numpy().reshape(vec.action_space.shape)
                    if isinstance(logits, torch.distributions.Normal):
                        action = np.clip(action, vec.action_space.low, vec.action_space.high)
                    ob, _, _, _, _ = vec.step(action)
                for e in range(to_render):
                    map_name = os.path.basename(str(scenarios[e].get("map_name") or "map")).split(".")[0]
                    # Numeric index last so build_gallery_index's `*_<N>.html`
                    # pattern matches.
                    path = out_dir / f"{map_name}{step_suffix}_{scenarios_done:03d}.html"
                    viz.generate_interactive_replay(
                        scenarios[e], agent_hist[e], traffic_hist[e], traj_hist[e], obs_hist[e], str(path), head_north=True
                    )
                    html_paths.append(path)
                    scenarios_done += 1
                    if scenarios_done >= num_scenarios:
                        break
        finally:
            vec.close()

        if html_paths:
            viz.build_gallery_index(str(out_dir))
        return html_paths

    def _render_env_overrides(self, args) -> dict:
        """Build env kwargs for the render env. Default: same as the
        metric-pass env plus `render_mode=headless`. Subclasses override
        to inject things like a random starting_map (gigaflow validation)
        or a shrunken bin set (behavior class)."""
        out = dict(args["env"])
        out["render_mode"] = "headless"
        return out

    def _render_view(self, vecenv, target_env, policy, args, view_idx, out_dir, view_suffix) -> list:
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
        for required in ("render_num_scenarios", "render_max_steps"):
            if required not in eval_cfg:
                raise KeyError(
                    f"[eval.{self.name}] has render=true but eval.{required} is not set. "
                    "Render is expensive — set it explicitly per evaluator."
                )
        num_scenarios = int(eval_cfg["render_num_scenarios"])
        max_steps = int(eval_cfg["render_max_steps"])

        saved_cwd = os.getcwd()
        os.chdir(out_dir)
        # Glob by full view_suffix (= step_suffix + view marker) so we get
        # only this-view's mp4s — not files written by a prior view in the
        # same render pass, which would otherwise duplicate in all_paths.
        # The trailing `.mp4` is exact, so e.g. `*_epoch7_step12_bev.mp4`
        # matches bev files but not the bare sim_state ones.
        view_glob = f"*{view_suffix}.mp4"
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

        return sorted(out_dir.glob(view_glob))


_VIEW_NAME_TO_IDX = {
    "sim_state": 0,
    "bev": 1,
    "topdown_sim": 2,
    "bev_all": 3,
}
