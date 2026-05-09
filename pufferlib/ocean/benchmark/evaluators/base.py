"""Evaluator base class + EvalResult dataclass."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class EvalResult:
    metrics: dict
    frames: list = field(default_factory=list)


class Evaluator:
    """Base class for all evaluators.

    Subclasses set `type_name` (the value used in `[eval.<name>].type`) and
    implement `rollout()`. Optionally override `env_overrides()`,
    `vec_overrides()`, and `aggregate()`.
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

    def rollout(self, vecenv, policy, args) -> EvalResult:
        raise NotImplementedError

    def aggregate(self, per_rollout: list) -> dict:
        """Reduce a list of per-rollout dicts to a single metrics dict.

        Default: numeric mean over keys present in any sub-dict. WOSAC
        overrides for likelihood-style aggregation."""
        import numpy as np

        if not per_rollout:
            return {}
        keys = set()
        for r in per_rollout:
            keys.update(r.keys())
        out = {}
        for k in keys:
            vals = [r[k] for r in per_rollout if k in r and isinstance(r[k], (int, float))]
            if vals:
                out[k] = float(np.mean(vals))
        return out
