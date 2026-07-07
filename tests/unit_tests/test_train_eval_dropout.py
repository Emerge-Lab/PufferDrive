#!/usr/bin/env python3
"""
End-to-end train -> eval consistency test for observation dropout.

Trains a tiny policy on a CPU env with nonzero road-obs dropout, then runs
the real EvalManager inline-eval path against eval envs whose dropout rate
differs from training. The policy slices the flat obs buffer by the env's
dropout-reduced road-slot layout, so any train/eval layout mismatch surfaces
as a shape/view error inside the rollout forward pass.

Deliberately not tied to any specific fix: it exercises the public pipeline
(load_config -> load_env -> load_policy -> PuffeRL -> EvalManager) and only
asserts that nothing crashes, the rollouts actually evaluate agents, and
training still works after the evals.
"""

import os
import random
import signal
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.benchmark.manager import EvalManager
from pufferlib.pufferl import PuffeRL, load_config, load_env, load_policy

SEED = 42
BPTT_HORIZON = 32  # == scenario_length so episodes complete each epoch
TRAIN_DROPOUT_LANE = 0.5
TRAIN_DROPOUT_BOUNDARY = 0.25
WATCHDOG_SECONDS = 900

# One eval section per layout relation to the training env:
#   same_dropout    — inherits training dropout (identity layout)
#   zero_dropout    — clean macro zeroes dropout (wider obs than training)
#   heavier_dropout — explicit higher dropout (narrower obs than training)
EVAL_SECTIONS = {
    "same_dropout": {
        "type": "multi_scenario",
        "interval": 1,
        "mode": "inline",
        "clean": False,
    },
    "zero_dropout": {
        "type": "multi_scenario",
        "interval": 1,
        "mode": "inline",
        "clean": True,
    },
    "heavier_dropout": {
        "type": "multi_scenario",
        "interval": 1,
        "mode": "inline",
        "clean": False,
        "env.obs_dropout_lane": 0.75,
        "env.obs_dropout_boundary": 0.75,
    },
}


class _DummyLogger:
    """PuffeRL calls self.logger.log() inside mean_and_log(); no-op it."""

    run_id = "eval_dropout_test"

    def log(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: None


class _Watchdog:
    """Hard cap on wall time so a hung rollout fails fast instead of
    eating the whole CI job."""

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._fire)
        signal.alarm(WATCHDOG_SECONDS)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        return False

    @staticmethod
    def _fire(signum, frame):
        raise TimeoutError(f"train/eval dropout test exceeded {WATCHDOG_SECONDS}s watchdog")


def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)


def _set_existing(section, updates):
    """Only override keys that already exist, so we never inject an unknown
    kwarg that a constructor would reject."""
    for key, value in updates.items():
        if key in section:
            section[key] = value


def _build_config():
    # load_config() calls argparse.parse_args(), which would otherwise choke
    # on pytest's argv. Hide it for the duration of the call.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv

    _set_existing(
        args["vec"],
        {
            "backend": "Serial",
            "num_envs": 2,
            "seed": SEED,
        },
    )

    _set_existing(
        args["env"],
        {
            "num_agents": 16,
            "min_agents_per_env": 16,
            "max_agents_per_env": 16,
            "action_type": "discrete",
            "num_maps": 2,
            "use_map_cache": 1,
            "map_dir": "pufferlib/resources/drive/binaries/carla",
            "scenario_length": BPTT_HORIZON,
            "obs_dropout_lane": TRAIN_DROPOUT_LANE,
            "obs_dropout_boundary": TRAIN_DROPOUT_BOUNDARY,
            "seed": SEED,
        },
    )

    # Shrink the net: this test checks layout plumbing, not learning.
    _set_existing(
        args["policy"],
        {
            "ego_input_size": 32,
            "partner_input_size": 128,
            "lane_input_size": 64,
            "boundary_input_size": 64,
            "traffic_control_input_size": 16,
            "context_input_size": 8,
            "backbone_hidden_size": 256,
            "actor_hidden_size": 128,
            "critic_hidden_size": 64,
        },
    )

    args["wandb"] = False
    args["neptune"] = False
    # Replace the ini's eval suites with the three dropout-relation sections.
    args["eval"] = {name: dict(section) for name, section in EVAL_SECTIONS.items()}

    return args


def _finalize_train_config(args, total_agents):
    minibatch_size = total_agents * BPTT_HORIZON // 4
    _set_existing(
        args["train"],
        {
            "device": "cpu",
            "compile": False,
            "seed": SEED,
            "torch_deterministic": True,
            "anneal_lr": False,
            "learning_rate": 0.001,
            "update_epochs": 1,
            "bptt_horizon": BPTT_HORIZON,
            "minibatch_size": minibatch_size,
            "max_minibatch_size": minibatch_size,
            "total_timesteps": 10_000_000,  # large -> never "done" during the test
            "checkpoint_interval": 10_000_000,
            "render": False,
        },
    )
    # The training loop itself must not fire evaluators (we dispatch them
    # explicitly below so their exceptions propagate).
    return dict(**args["train"], env="puffer_drive", eval={})


def _run_train_epoch(pufferl):
    pufferl.evaluate()
    pufferl.train()


def test_train_then_eval_across_dropout_rates():
    _seed_everything()
    args = _build_config()

    vecenv = None
    pufferl = None
    with _Watchdog():
        try:
            vecenv = load_env("puffer_drive", args)
            train_config = _finalize_train_config(args, vecenv.num_agents)
            policy = load_policy(args, vecenv, "puffer_drive")
            pufferl = PuffeRL(train_config, vecenv, policy, logger=_DummyLogger())

            _run_train_epoch(pufferl)

            manager = EvalManager.from_config(args)
            for eval_name in EVAL_SECTIONS:
                result = manager.run_one_by_name(
                    eval_name,
                    policy=pufferl.uncompiled_policy,
                    env_name="puffer_drive",
                    global_step=0,
                    epoch=1,
                )
                assert result.metrics.get("num_agents_evaluated", 0) > 0, (
                    f"[eval.{eval_name}] rollout completed but evaluated no agents"
                )

            # Eval must leave the policy usable on the training layout: a
            # policy left pointed at an eval layout would crash right here.
            _run_train_epoch(pufferl)
        finally:
            if pufferl is not None and hasattr(pufferl, "utilization"):
                try:
                    pufferl.utilization.stop()
                except Exception:
                    assert False, "PuffeRL.utilization.stop() failed; check for dangling threads"
            if vecenv is not None:
                try:
                    vecenv.close()
                except Exception:
                    assert False, "vecenv.close() failed; check for dangling threads"


if __name__ == "__main__":
    test_train_then_eval_across_dropout_rates()
    print("Train/eval dropout test passed!")
