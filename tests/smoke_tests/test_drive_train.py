#!/usr/bin/env python3
"""
Deterministic CPU smoke test for the PufferDrive training pipeline.

Runs the real pipeline (load_config -> load_env -> load_policy -> PuffeRL) for
2 epochs on CPU (synchronous Serial backend), then compares the captured training
metrics (PPO losses) and environment metrics (collision/offroad/goal/return)
against a committed golden file. Kept deliberately small because CI runs it under
QEMU (emulated fixed CPU) where torch is ~30x slower.

The run is intentionally light but sized (short scenarios + enough agents) so
that collisions, offroad events and episode completions actually occur, giving
the metrics non-trivial values.

Golden values
-------------
The expected metrics live in tests/smoke_tests/data/drive_smoke_golden.json. The
golden is bit-reproducible only inside the pinned image (tests/smoke_tests/
Dockerfile), which freezes the compiler + torch/numpy wheels AND runs under QEMU
emulating a fixed CPU (-cpu Haswell) so glibc libm dispatch and torch kernels are
identical on every host. A golden generated outside that image will NOT match CI.
To regenerate it after an intentional pipeline change (the image ENTRYPOINT
already wraps qemu + python -m pytest, so pass only test args):

    docker build -f tests/smoke_tests/Dockerfile -t pufferdrive-smoke .
    docker run --rm -e SMOKE_UPDATE_GOLDEN=1 \
        -v "$PWD/tests/smoke_tests/data:/app/tests/smoke_tests/data" \
        pufferdrive-smoke tests/smoke_tests/test_drive_train.py

Then commit the updated json. Subsequent runs assert against it within tolerance
(np.isclose; tighten/loosen via SMOKE_RTOL / SMOKE_ATOL).
"""

import json
import os
import random
import signal
import sys

# Stabilize CPU math for cross-machine reproducibility (must be set before
# torch/numpy load their BLAS). Single-threaded + a fixed, CPU-agnostic kernel
# so floating-point rounding does not depend on the host CPU's SIMD features.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # numpy's BLAS (OpenBLAS)
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")  # pin one OpenBLAS kernel
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")  # torch's BLAS (MKL): reproducible across CPUs
# torch is built with oneDNN (MKL-DNN), which JIT-dispatches AVX2/AVX-512 kernels
# by host CPU and is NOT covered by MKL_CBWR. Cap it to a fixed ISA floor so an
# AVX-512 host and an AVX2 host run the identical kernel.
os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")
os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX2")  # older oneDNN env-var name

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.pufferl import PuffeRL, load_config, load_env, load_policy

SEED = 42
EPOCHS = 2  # small: CI emulates the CPU under QEMU where torch is ~30x slower
BPTT_HORIZON = 64  # env steps per evaluate(); == scenario length so episodes complete each epoch
WATCHDOG_SECONDS = 1500  # generous: QEMU-emulated torch is slow; still catches true hangs

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data", "drive_smoke_golden.json")
# Two-tier tolerance. Under QEMU (-cpu Haswell) almost every metric is bit-stable
# across hosts (<1% drift); only float-accumulation metrics (loss sums, EMAs,
# episode_return) carry ~5% host residue, because qemu-user runs FP on the host
# CPU and glibc libm dispatches on the host's AT_HWCAP. So check the stable
# metrics tightly and only those few loosely.
RTOL = float(os.environ.get("SMOKE_RTOL", "2e-2"))
LOOSE_RTOL = float(os.environ.get("SMOKE_LOOSE_RTOL", "8e-2"))
ATOL = float(os.environ.get("SMOKE_ATOL", "1e-3"))
# Metrics that are sums/EMAs of many floats -> host-FP-sensitive; checked loosely.
LOOSE_KEYS = frozenset({"value_loss", "ema_max", "episode_return"})

# Env metrics we expect a random policy to exercise within the run.
SANITY_KEYS = ("collision_rate", "offroad_rate")


class _DummyLogger:
    """PuffeRL calls self.logger.log() inside mean_and_log(); no-op it."""

    run_id = "smoke"

    def log(self, *args, **kwargs):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def _set_existing(section, updates):
    """Only override keys that already exist, so we never inject an unknown
    kwarg that a constructor would reject."""
    for k, v in updates.items():
        if k in section:
            section[k] = v


def _build_config():
    # load_config() calls argparse.parse_args(), which would otherwise choke on
    # pytest's argv. Hide it for the duration of the call.
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv

    _set_existing(
        args["vec"],
        {
            # Serial (synchronous) on purpose: async Multiprocessing interleaves
            # env recv by host scheduling/core-count, which reorders the batch and
            # flips sampled actions -> metrics diverge across machines. Serial is
            # deterministic everywhere, which is what the golden needs.
            "backend": "Serial",
            "num_envs": 4,  # fewer envs -> less per-epoch compute under QEMU
            "seed": SEED,
        },
    )

    _set_existing(
        args["env"],
        {
            "num_agents": 16,  # per-env active count
            "min_agents_per_env": 16,  # fixed active count -> deterministic, packed
            "max_agents_per_env": 16,
            "action_type": "discrete",
            "num_maps": 2,  # few maps -> fast load
            "use_map_cache": 1,  # share map geometry across envs in a worker
            "map_dir": "pufferlib/resources/drive/binaries/carla",
            "scenario_length": BPTT_HORIZON,  # short -> episodes complete each epoch
            "seed": SEED,
        },
    )

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
    _set_existing(args["rnn"], {"input_size": 256, "hidden_size": 256})

    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = {}  # disable all evaluators during the smoke run

    return args


def _finalize_train_config(args, total_agents):
    minibatch_size = total_agents * BPTT_HORIZON // 8
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
            "total_timesteps": 10_000_000,  # large -> never "done" during 5 epochs
            "checkpoint_interval": 10_000_000,
            "render": False,
        },
    )
    return dict(**args["train"], env="puffer_drive", eval=args.get("eval", {}))


def _capture_metrics(pufferl):
    from collections import defaultdict

    env_acc = defaultdict(list)
    for _ in range(EPOCHS):
        pufferl.evaluate()
        # Snapshot env stats BEFORE train()'s mean_and_log() resets them.
        for k, v in pufferl.stats.items():
            env_acc[k].extend(v if isinstance(v, list) else [v])
        # Force logging this epoch so self.losses is set deterministically
        # (otherwise it only updates on a wall-clock interval -> flaky).
        pufferl.last_log_time = 0.0
        pufferl.train()

    losses = {k: float(v) for k, v in pufferl.losses.items() if _is_number(v)}

    env_means = {}
    for k, vals in env_acc.items():
        nums = [x for x in vals if _is_number(x)]
        if nums and len(nums) == len(vals):
            env_means[k] = float(np.mean(nums))

    return losses, env_means


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _nan_eq(a, b, rtol):
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isclose(a, b, rtol=rtol, atol=ATOL))


def _compare(label, actual, expected):
    mismatches = []
    for key, exp in expected.items():
        rtol = LOOSE_RTOL if key in LOOSE_KEYS else RTOL
        if key not in actual:
            mismatches.append(f"  {label}/{key}: MISSING (expected {exp})")
        elif not _nan_eq(actual[key], exp, rtol):
            mismatches.append(f"  {label}/{key}: {actual[key]!r} != expected {exp!r}")
    return mismatches


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
    args = _build_config()

    vecenv = None
    pufferl = None
    with _Watchdog():
        try:
            vecenv = load_env("puffer_drive", args)
            total_agents = vecenv.num_agents
            train_config = _finalize_train_config(args, total_agents)

            _seed_everything()  # re-seed right before weight init
            policy = load_policy(args, vecenv, "puffer_drive")

            pufferl = PuffeRL(train_config, vecenv, policy, logger=_DummyLogger())
            losses, env_means = _capture_metrics(pufferl)
        finally:
            if pufferl is not None and hasattr(pufferl, "utilization"):
                try:
                    pufferl.utilization.stop()
                except Exception:
                    pass
            if vecenv is not None:
                try:
                    vecenv.close()
                except Exception:
                    pass

    print("\n[smoke] total_agents:", total_agents)
    print("[smoke] losses:", json.dumps(losses, indent=2, sort_keys=True))
    print("[smoke] env_means:", json.dumps(env_means, indent=2, sort_keys=True))

    # The run must actually exercise interesting cases.
    present = [k for k in SANITY_KEYS if k in env_means]
    assert present, f"no sanity env metrics captured; got keys: {sorted(env_means)}"
    assert any(env_means[k] > 0 for k in present), (
        f"expected nonzero {present} (collisions/offroad), got { {k: env_means[k] for k in present} }"
    )

    record = os.environ.get("SMOKE_UPDATE_GOLDEN") == "1" or not os.path.exists(GOLDEN_PATH)
    if record:
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as f:
            json.dump(
                {
                    "meta": {
                        "total_agents": int(total_agents),
                        "bptt_horizon": BPTT_HORIZON,
                        "epochs": EPOCHS,
                        "seed": SEED,
                    },
                    "losses": losses,
                    "env": env_means,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[smoke] wrote golden -> {GOLDEN_PATH}")
        return

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    mismatches = _compare("losses", losses, golden["losses"]) + _compare("env", env_means, golden["env"])
    assert not mismatches, "smoke metrics drifted from golden:\n" + "\n".join(mismatches)


if __name__ == "__main__":
    test_drive_smoke()
    print("Smoke test passed!")
