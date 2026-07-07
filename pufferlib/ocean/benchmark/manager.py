"""EvalManager — discovers `[eval.<name>]` sections, instantiates Evaluators,
dispatches them inline or as subprocesses, logs results.

Config schema (see docs/eval_unification.md):

    [eval.<name>]
    type = "<registered_type>"
    enabled = true|false
    interval = <epochs>
    mode = "inline" | "subprocess"
    inherits = "<other_eval_name>"      # optional, recursive merge
    clean = true|false
    render = true|false
    render_views = ["sim_state", ...]
    env.<key> = <value>                 # any [env] override
    eval.<key> = <value>                # evaluator-specific knobs
    vec.<key> = <value>                 # any [vec] override

Sections without a `type` field are templates (only usable via `inherits`).
"""

import copy
import glob
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pufferlib

from pufferlib.ocean.benchmark.evaluators import EVALUATOR_REGISTRY, EvalResult, Evaluator
from pufferlib.ocean.drive.drive import compute_effective_road_obs_count
from pufferlib.ocean.torch import DriveBackbone

# clean_eval macro — env knobs to zero/enforce. Per-section explicit values
# win over the macro (see _build_section_config).
CLEAN_EVAL_OVERRIDES = {
    "obs_dropout_lane": 0.0,
    "obs_dropout_boundary": 0.0,
    "partner_blindness_prob": 0.0,
    "phantom_braking_prob": 0.0,
    "phantom_braking_trigger_prob": 0.0,
    "traffic_light_behavior": "stop",
}


class EvalManager:
    def __init__(self, evaluators: list, train_config: dict, run_id: str = None):
        self.evaluators = evaluators
        self.train_config = train_config
        # `run_id` is needed to resolve the latest checkpoint for subprocess
        # evals. None is fine if no evaluator is mode=subprocess.
        self.run_id = run_id

    @classmethod
    def from_config(cls, train_config: dict, run_id: str = None) -> "EvalManager":
        sections = _discover_eval_sections(train_config)
        evaluators = []
        for name, raw in sections.items():
            cfg = _build_section_config(name, raw, sections)
            type_name = cfg.get("type")
            if type_name is None:
                # Template section — referenced via inherits but not instantiated.
                continue
            cls_for_type = EVALUATOR_REGISTRY.get(type_name)
            if cls_for_type is None:
                raise ValueError(
                    f"[eval.{name}] type='{type_name}' is not registered. "
                    f"Known types: {sorted(EVALUATOR_REGISTRY.keys())}"
                )
            evaluators.append(cls_for_type(name=name, config=cfg, train_config=train_config))
        return cls(evaluators=evaluators, train_config=train_config, run_id=run_id)

    def has_subprocess_evals_at(self, epoch: int) -> bool:
        """True if any enabled subprocess evaluator would fire at this epoch.
        Training loop uses this to decide whether to save_checkpoint() before
        calling maybe_run() — subprocesses load the checkpoint from disk."""
        for ev in self.evaluators:
            if not ev.enabled or ev.mode != "subprocess" or ev.interval <= 0:
                continue
            if epoch % ev.interval == 0:
                return True
        return False

    def latest_checkpoint(self, env_name: str) -> str:
        """Return the path to the most recent model_*.pt under the experiment
        dir. Falls back to train_config['load_model_path'] if no checkpoints
        have been written yet (e.g. resume-from path before first save).
        Returns None if neither resolves."""
        if self.run_id and self.train_config.get("data_dir"):
            model_dir = os.path.join(self.train_config["data_dir"], f"{env_name}_{self.run_id}", "models")
            if os.path.isdir(model_dir):
                files = glob.glob(os.path.join(model_dir, "model_*.pt"))
                if files:
                    return max(files, key=os.path.getctime)
        return self.train_config.get("load_model_path")

    def maybe_run(self, epoch: int, policy, env_name: str, logger=None, global_step=None, force: bool = False) -> dict:
        """Called from the training loop. Runs every enabled evaluator
        whose `interval` divides `epoch` (or all enabled evaluators if
        `force=True` — used at training shutdown to capture final metrics
        regardless of where the epoch lands). Returns a dict mapping
        `eval_name → EvalResult` (each result has `.metrics` and
        `.frames`).

        One evaluator's failure (e.g., a missing bin dir for a single
        behavior class) doesn't skip the rest — the error is printed in
        full and the loop moves on. Errors stay loud (traceback + clear
        prefix) but the eval batch isn't all-or-nothing.
        """
        import traceback

        results = {}
        for ev in self.evaluators:
            if not ev.enabled:
                continue
            if ev.interval <= 0 and not force:
                continue
            if not force and epoch % ev.interval != 0:
                continue
            try:
                res = self._run_one(
                    ev, policy=policy, env_name=env_name, logger=logger, global_step=global_step, epoch=epoch
                )
                results[ev.name] = res
            except Exception:
                print(f"\n[EvalManager] Evaluator '{ev.name}' raised at epoch {epoch}; continuing with the rest:")
                traceback.print_exc()
        return results

    def run_one_by_name(
        self, name: str, policy, env_name: str, logger=None, global_step=None, epoch=None
    ) -> EvalResult:
        """Run a single named evaluator regardless of interval. Used for
        the subprocess CLI entry and for standalone `puffer eval --evaluator <name>`."""
        for ev in self.evaluators:
            if ev.name == name:
                return self._run_one(
                    ev, policy=policy, env_name=env_name, logger=logger, global_step=global_step, epoch=epoch
                )
        raise KeyError(f"No evaluator named '{name}'. Known: {[e.name for e in self.evaluators]}")

    def _run_one(self, ev: Evaluator, policy, env_name: str, logger, global_step, epoch=None) -> EvalResult:
        if ev.mode == "subprocess":
            res = self._run_subprocess(ev, env_name=env_name, global_step=global_step, epoch=epoch)
        else:
            res = self._run_inline(ev, policy=policy, env_name=env_name, global_step=global_step, epoch=epoch)
        if logger is not None:
            self._log(ev, res, logger=logger, global_step=global_step)
        if hasattr(ev, "cleanup"):
            ev.cleanup()
        return res

    def _run_inline(self, ev: Evaluator, policy, env_name: str, global_step, epoch=None) -> EvalResult:
        args = self._build_eval_args(ev, env_name=env_name, global_step=global_step, epoch=epoch)

        package = args.get("package", "ocean")
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(env_name)

        vec_kwargs = ev.vec_overrides()
        backend = vec_kwargs.get("backend", "PufferEnv")
        num_envs = int(vec_kwargs.get("num_envs", 1))

        # PufferEnv is the default: Drive's C kernel batches all internal
        # envs in one call so we get per-map parallelism without paying
        # fork/IPC cost, and render shares the single ffmpeg pipeline.
        # Multiprocessing is opt-in via [eval.<name>.vec] backend = ...
        # for evals that genuinely need it (memory-split for big replay
        # sweeps, hetero scenarios, async overlap on long rollouts).
        # The two backends have incompatible call shapes; branch here.
        if backend == "PufferEnv":
            vecenv = pufferlib.vector.make(
                make_env, env_args=[], env_kwargs=args["env"], backend=backend, num_envs=num_envs
            )
        else:
            vec_call_kwargs = dict(vec_kwargs)
            vec_call_kwargs.setdefault("num_workers", num_envs)
            vec_call_kwargs.setdefault("batch_size", num_envs)
            vecenv = pufferlib.vector.make(
                [make_env] * num_envs,
                env_args=[[]] * num_envs,
                env_kwargs=[args["env"] for _ in range(num_envs)],
                **vec_call_kwargs,
            )
        # TODO: temporary fix for the DriveBackbone slot-count mismatch bug
        # Swap backbones at the eval layout and restore the training layout after
        saved_road_slots = _swap_policy_road_slots(policy, args["env"])
        try:
            res = ev.rollout(vecenv, policy, args)
        finally:
            vecenv.close()
            _restore_policy_road_slots(saved_road_slots)
        return res

    def _run_subprocess(self, ev: Evaluator, env_name: str, global_step, epoch=None) -> EvalResult:
        out_path = Path(self.train_config.get("data_dir", ".")) / "eval_subprocess_out" / f"{ev.name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "pufferlib.pufferl",
            "eval",
            env_name,
            "--evaluator",
            ev.name,
            "--out",
            str(out_path),
        ]
        # global_step and epoch flow through the CLI so the subprocess's
        # render path can stamp them into mp4 filenames (otherwise every
        # epoch's renders collide as `_epoch0_step0.mp4`).
        if global_step is not None:
            cmd += ["--global-step", str(int(global_step))]
        if epoch is not None:
            cmd += ["--epoch", str(int(epoch))]
        # Subprocess loads the freshest checkpoint on disk. Caller (training
        # loop) is responsible for save_checkpoint() before this fires —
        # see has_subprocess_evals_at.
        ckpt = self.latest_checkpoint(env_name)
        if ckpt:
            cmd += ["--load-model-path", ckpt]
        subprocess.run(cmd, check=True)
        with open(out_path) as f:
            payload = json.load(f)
        return EvalResult(metrics=payload.get("metrics", {}), frames=payload.get("frames", []))

    def _build_eval_args(self, ev: Evaluator, env_name: str, global_step, epoch=None) -> dict:
        args = copy.deepcopy(self.train_config)
        args["env"].update(ev.env_overrides())
        args.setdefault("vec", {})
        args["vec"].update(ev.vec_overrides())
        args["env_name"] = env_name
        args["global_step"] = global_step
        args["epoch"] = epoch
        args["seed"] = int(self.train_config.get("train", {}).get("seed", 42)) or 42
        # Pass through evaluator-private fields that subclasses look up on args.
        ev_eval = ev.config.get("eval", {})
        if ev_eval:
            args.setdefault("eval", {})
            args["eval"].update(ev_eval)
            # The per-episode CSV + coverage report consume `completed_episode`
            # summaries, which the env only emits when this is on. Enabling it
            # is additive — it doesn't change the my_log metric stream.
            if ev_eval.get("export_episode_csv") or ev_eval.get("verify_coverage"):
                args["env"]["emit_completed_episodes"] = True
        # Forward top-level render config so _render_pass can read it.
        for key in ("render_backend", "render_results_dir", "eval_results_dir"):
            if key in ev.config:
                args[key] = ev.config[key]
        return args

    def _log(self, ev: Evaluator, result: EvalResult, logger, global_step):
        if not result.metrics and not result.frames:
            return
        log_dict = {f"{ev.name}/{k}": float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
        if hasattr(logger, "local_writer") and logger.local_writer and global_step is not None:
            for k, v in log_dict.items():
                logger.local_writer.add_scalar(k, v, global_step)
        if hasattr(logger, "log") and log_dict:
            if global_step is not None:
                logger.log(log_dict, global_step)
            else:
                logger.log(log_dict)
        if result.frames and hasattr(logger, "log"):
            try:
                import wandb

                videos = [
                    wandb.Video(str(p), fps=30, format="mp4", caption=Path(p).stem)
                    for p in result.frames
                    if str(p).endswith(".mp4")
                ]
                if videos:
                    payload = {f"{ev.name}/render": videos if len(videos) > 1 else videos[0]}
                    if global_step is not None:
                        logger.log(payload, global_step)
                    else:
                        logger.log(payload)
            except ImportError:
                pass


def _swap_policy_road_slots(policy, env_args: dict) -> list:
    """
    Point every DriveBackbone in `policy` at the eval env's road-obs layout
    DriveBackbone slices the flat obs buffer with the training env's dropout-reduced kept slot counts
    Eval env emits obs_slots_*_n based on its own config
    Returns the saved per-module counts for _restore_policy_road_slots
    """
    if "obs_slots_lane_n" not in env_args:
        return []
    eval_lane_kept = compute_effective_road_obs_count(
        int(env_args["obs_slots_lane_n"]), float(env_args.get("obs_dropout_lane", 0.0))
    )
    eval_boundary_kept = compute_effective_road_obs_count(
        int(env_args["obs_slots_boundary_n"]), float(env_args.get("obs_dropout_boundary", 0.0))
    )
    saved = []
    for module in policy.modules():
        if not isinstance(module, DriveBackbone):
            continue
        saved.append((module, module.obs_slots_lane_kept, module.obs_slots_boundary_kept))
        module.obs_slots_lane_kept = eval_lane_kept
        module.obs_slots_boundary_kept = eval_boundary_kept
    return saved


def _restore_policy_road_slots(saved: list) -> None:
    for module, lane_kept, boundary_kept in saved:
        module.obs_slots_lane_kept = lane_kept
        module.obs_slots_boundary_kept = boundary_kept


def _discover_eval_sections(args: dict) -> dict:
    """Pull `[eval.<name>]` sections out of the parsed config.

    `load_config` flattens dotted section names into a nested dict. So
    `[eval.foo]` becomes `args["eval"]["foo"]`. We collect every direct
    child of `args["eval"]` that's itself a dict and treat it as a section."""
    eval_root = args.get("eval", {})
    if not isinstance(eval_root, dict):
        return {}
    sections = {}
    for name, body in eval_root.items():
        if isinstance(body, dict):
            sections[name] = body
    return sections


def _build_section_config(name: str, raw: dict, all_sections: dict) -> dict:
    """Resolve `inherits` chain + `clean` macro + dotted-key flattening."""
    chain = []
    current_name = name
    current_raw = raw
    visited = set()
    while True:
        if current_name in visited:
            raise ValueError(f"Cyclic 'inherits' chain involving [eval.{current_name}]")
        visited.add(current_name)
        chain.append(current_raw)
        parent_name = current_raw.get("inherits")
        if parent_name is None:
            break
        if parent_name not in all_sections:
            raise ValueError(f"[eval.{current_name}].inherits='{parent_name}' is not a known section")
        current_name = parent_name
        current_raw = all_sections[parent_name]

    merged = {}
    for level in reversed(chain):
        # Deepcopy each level before merging — _deep_merge recurses into
        # nested dicts in `merged` and mutates them in place. Without the
        # copy, `merged["env"]` would alias the parent section's env dict;
        # the child merge would mutate the parent in place, and every
        # subsequent evaluator that inherits from the same parent would
        # see the last-processed child's overrides instead of its own.
        _deep_merge(merged, _expand_dotted(copy.deepcopy(level)))

    if merged.get("clean", True):
        env_section = merged.setdefault("env", {})
        for k, v in CLEAN_EVAL_OVERRIDES.items():
            env_section.setdefault(k, v)

    return merged


def _expand_dotted(raw: dict) -> dict:
    """`{"env.simulation_mode": "replay"}` → `{"env": {"simulation_mode": "replay"}}`."""
    out = {}
    for k, v in raw.items():
        if "." in k:
            head, _, tail = k.partition(".")
            sub = out.setdefault(head, {})
            sub[tail] = v
        else:
            out[k] = v
    return out


def _deep_merge(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
