"""Deterministic CPU smoke test: two PPO epochs of the real training pipeline against a committed golden."""

import contextlib
import json
import random
import signal

import numpy as np
import torch
from smoke_golden import BPTT_HORIZON, SEED, assert_infractions_occur, build_config, is_number, record_or_compare_golden

from opengigaflow.rl import PPOTrainer, load_env, load_policy
from opengigaflow.utils import finalize_environment_metrics


EPOCHS = 2  # CI emulates the CPU under QEMU, where torch is ~30x slower
WATCHDOG_SECONDS = 1500


class _DummyLogger:
    run_id = "smoke"

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def _finalize_train_config(args, total_agents):
    minibatch_size = total_agents * BPTT_HORIZON // 8
    args["train"].update(
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
            "total_timesteps": 10_000_000,  # never reached within EPOCHS
            "checkpoint_interval": 10_000_000,
        }
    )
    return dict(**args["train"], env="opengigaflow")


def _capture_metrics(trainer):
    env_acc = {}
    for _ in range(EPOCHS):
        trainer.evaluate()
        # train()'s mean_and_log() resets the env sums
        for k, (value_sum, weight_sum) in trainer.env_metric_sums.items():
            entry = env_acc.setdefault(k, [0.0, 0.0])
            entry[0] += value_sum
            entry[1] += weight_sum
        # log every epoch: trainer.losses otherwise updates on a wall-clock interval
        trainer.last_log_time = 0.0
        trainer.train()

    losses = {k: float(v) for k, v in trainer.losses.items() if is_number(v)}
    return losses, finalize_environment_metrics(env_acc)


class _Watchdog:
    def __enter__(self):
        signal.signal(signal.SIGALRM, self._fire)
        signal.alarm(WATCHDOG_SECONDS)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        return False

    @staticmethod
    def _fire(signum, frame):
        raise TimeoutError(f"smoke test exceeded {WATCHDOG_SECONDS}s watchdog")


def test_drive_smoke():
    _seed_everything()
    args = build_config(num_envs=4)
    args["policy"].update(
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
            "fused_slot_encoder": False,  # CUDA-only kernels that need wider encoders than this run's
        }
    )

    vecenv = None
    trainer = None
    with _Watchdog():
        try:
            vecenv = load_env(args)
            total_agents = vecenv.num_agents
            train_config = _finalize_train_config(args, total_agents)

            _seed_everything()  # re-seed right before weight init
            policy = load_policy(args, vecenv)

            trainer = PPOTrainer(train_config, vecenv, policy, logger=_DummyLogger())
            losses, env_means = _capture_metrics(trainer)
        finally:
            if trainer is not None:
                trainer.utilization.stop()
            if vecenv is not None:
                with contextlib.suppress(Exception):
                    vecenv.close()

    print("\n[smoke] total_agents:", total_agents)
    print("[smoke] losses:", json.dumps(losses, indent=2, sort_keys=True))
    print("[smoke] env_means:", json.dumps(env_means, indent=2, sort_keys=True))

    assert_infractions_occur(env_means)
    record_or_compare_golden("drive_smoke_golden.json", total_agents, EPOCHS, {"losses": losses, "env": env_means})


if __name__ == "__main__":
    test_drive_smoke()
    print("Smoke test passed!")
