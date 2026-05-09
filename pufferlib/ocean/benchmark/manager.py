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
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pufferlib

from pufferlib.ocean.benchmark.evaluators import EVALUATOR_REGISTRY, EvalResult, Evaluator

# clean_eval macro — env knobs to zero/enforce. Per-section explicit values
# win over the macro (see _build_section_config).
CLEAN_EVAL_OVERRIDES = {
    "lane_segment_dropout": 0.0,
    "boundary_segment_dropout": 0.0,
    "partner_blindness_prob": 0.0,
    "phantom_braking_prob": 0.0,
    "phantom_braking_trigger_prob": 0.0,
    "traffic_light_behavior": 1,
}


class EvalManager:
    def __init__(self, evaluators: list, train_config: dict):
        self.evaluators = evaluators
        self.train_config = train_config

    @classmethod
    def from_config(cls, train_config: dict) -> "EvalManager":
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
        return cls(evaluators=evaluators, train_config=train_config)

    def maybe_run(self, epoch: int, policy, env_name: str, logger=None, global_step=None) -> dict:
        """Called from the training loop. Runs every enabled evaluator
        whose `interval` divides `epoch`. Returns a dict of {eval_name → metrics}."""
        results = {}
        for ev in self.evaluators:
            if not ev.enabled:
                continue
            if ev.interval <= 0:
                continue
            if epoch % ev.interval != 0:
                continue
            res = self._run_one(ev, policy=policy, env_name=env_name, logger=logger, global_step=global_step)
            results[ev.name] = res
        return results

    def run_one_by_name(self, name: str, policy, env_name: str, logger=None, global_step=None) -> EvalResult:
        """Run a single named evaluator regardless of interval. Used for
        the subprocess CLI entry and for standalone `puffer eval --evaluator <name>`."""
        for ev in self.evaluators:
            if ev.name == name:
                return self._run_one(ev, policy=policy, env_name=env_name, logger=logger, global_step=global_step)
        raise KeyError(f"No evaluator named '{name}'. Known: {[e.name for e in self.evaluators]}")

    def _run_one(self, ev: Evaluator, policy, env_name: str, logger, global_step) -> EvalResult:
        if ev.mode == "subprocess":
            res = self._run_subprocess(ev, env_name=env_name, global_step=global_step)
        else:
            res = self._run_inline(ev, policy=policy, env_name=env_name, global_step=global_step)
        if logger is not None:
            self._log(ev, res, logger=logger, global_step=global_step)
        if hasattr(ev, "cleanup"):
            ev.cleanup()
        return res

    def _run_inline(self, ev: Evaluator, policy, env_name: str, global_step) -> EvalResult:
        args = self._build_eval_args(ev, env_name=env_name, global_step=global_step)

        package = args.get("package", "ocean")
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(env_name)

        vec_kwargs = ev.vec_overrides()
        num_envs = int(vec_kwargs.get("num_envs", 1))
        env_kwargs_list = [args["env"] for _ in range(num_envs)]
        env_creators = [make_env] * num_envs
        env_args_list = [[]] * num_envs

        vec_call_kwargs = dict(vec_kwargs)
        vec_call_kwargs.setdefault("num_workers", num_envs)
        vec_call_kwargs.setdefault("batch_size", num_envs)

        vecenv = pufferlib.vector.make(
            env_creators, env_args=env_args_list, env_kwargs=env_kwargs_list, **vec_call_kwargs
        )
        try:
            res = ev.rollout(vecenv, policy, args)
        finally:
            vecenv.close()
        return res

    def _run_subprocess(self, ev: Evaluator, env_name: str, global_step) -> EvalResult:
        out_path = Path(self.train_config.get("data_dir", ".")) / "eval_subprocess_out" / f"{ev.name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path = out_path.with_suffix(".cfg.json")
        with open(cfg_path, "w") as f:
            json.dump({"name": ev.name, "global_step": global_step}, f)

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
        # Subprocess inherits the same checkpoint via train_config.load_model_path.
        if self.train_config.get("load_model_path"):
            cmd += ["--load-model-path", self.train_config["load_model_path"]]
        subprocess.run(cmd, check=True)
        with open(out_path) as f:
            payload = json.load(f)
        return EvalResult(metrics=payload.get("metrics", {}), frames=payload.get("frames", []))

    def _build_eval_args(self, ev: Evaluator, env_name: str, global_step) -> dict:
        args = copy.deepcopy(self.train_config)
        args["env"].update(ev.env_overrides())
        args.setdefault("vec", {})
        args["vec"].update(ev.vec_overrides())
        args["env_name"] = env_name
        args["global_step"] = global_step
        args["seed"] = int(self.train_config.get("train", {}).get("seed", 42)) or 42
        # Pass through evaluator-private fields that subclasses look up on args.
        ev_eval = ev.config.get("eval", {})
        if ev_eval:
            args.setdefault("eval", {})
            args["eval"].update(ev_eval)
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
        _deep_merge(merged, _expand_dotted(level))

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
