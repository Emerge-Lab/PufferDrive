#!/usr/bin/env python3
"""Deterministic CPU smoke test for the PufferDrive simulator under random actions.

The training-free twin of test_drive_train.py: same env configuration and the
same 5 x BPTT_HORIZON env steps, but with NO policy / PPO update in the loop.
Every step feeds seeded random actions, so the trajectory depends only on the C
simulator + a platform-independent numpy RNG (no torch, no BLAS). That makes the
golden comparison robust across machines without the chaos of a learning loop.

Golden values
-------------
Expected metrics live in tests/smoke_tests/data/drive_rollout_golden.json. The
golden is generated inside the pinned image, which runs under QEMU emulating a
fixed CPU so the result is identical on every host. To regenerate after an
intentional simulator change (the image ENTRYPOINT already wraps qemu + python -m
pytest, so pass only test args):

    docker build -f tests/smoke_tests/Dockerfile -t pufferdrive-smoke .
    docker run --rm -e SMOKE_UPDATE_GOLDEN=1 \
        -v "$PWD/tests/smoke_tests/data:/app/tests/smoke_tests/data" \
        pufferdrive-smoke tests/smoke_tests/test_drive_rollout.py

Then commit the updated json. Subsequent runs assert against it within tolerance
(np.isclose; tune via SMOKE_RTOL / SMOKE_ATOL).
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pufferlib
from pufferlib.pufferl import load_config, load_env

SEED = 42
EPOCHS = 5
BPTT_HORIZON = 64  # env steps per epoch; == scenario length so episodes complete

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "data", "drive_rollout_golden.json")
# Two-tier tolerance: rates/counts are bit-stable across hosts (<1%), but
# episode_return is a float sum that carries ~5% host residue (qemu-user runs FP
# on the host CPU and glibc libm dispatches on host AT_HWCAP). Check the stable
# metrics tightly and only episode_return loosely.
RTOL = float(os.environ.get("SMOKE_RTOL", "2e-2"))
LOOSE_RTOL = float(os.environ.get("SMOKE_LOOSE_RTOL", "8e-2"))
ATOL = float(os.environ.get("SMOKE_ATOL", "1e-3"))
LOOSE_KEYS = frozenset({"episode_return"})

SANITY_KEYS = ("collision_rate", "offroad_rate")


def _set_existing(section, updates):
    for k, v in updates.items():
        if k in section:
            section[k] = v


def _build_config():
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv

    _set_existing(args["vec"], {"backend": "Serial", "num_envs": 8, "seed": SEED})
    _set_existing(
        args["env"],
        {
            "num_agents": 16,  # per-env -> 8 envs * 16 = 128 total agents
            "min_agents_per_env": 16,
            "max_agents_per_env": 16,
            "action_type": "discrete",
            "num_maps": 2,
            "use_map_cache": True,
            "map_dir": "pufferlib/resources/drive/binaries/carla",
            "scenario_length": BPTT_HORIZON,
            "seed": SEED,
        },
    )
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None
    return args


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _random_actions(vecenv, rng):
    return rng.integers(0, vecenv.single_action_space.n, size=vecenv.num_agents).astype(np.int32)


def _capture_env_means(vecenv, rng):
    env_acc = defaultdict(list)
    vecenv.reset(seed=SEED)
    for _ in range(EPOCHS):
        for _ in range(BPTT_HORIZON):
            _, _, _, _, infos = vecenv.step(_random_actions(vecenv, rng))
            for info in infos:
                for k, v in pufferlib.unroll_nested_dict(info):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    if isinstance(v, (list, tuple)):
                        env_acc[k].extend(v)
                    else:
                        env_acc[k].append(v)

    env_means = {}
    for k, vals in env_acc.items():
        nums = [x for x in vals if _is_number(x)]
        if nums and len(nums) == len(vals):
            env_means[k] = float(np.mean(nums))
    return env_means


def _compare(actual, expected):
    mismatches = []
    for key, exp in expected.items():
        rtol = LOOSE_RTOL if key in LOOSE_KEYS else RTOL
        if key not in actual:
            mismatches.append(f"  env/{key}: MISSING (expected {exp})")
        elif not bool(np.isclose(actual[key], exp, rtol=rtol, atol=ATOL)):
            mismatches.append(f"  env/{key}: {actual[key]!r} != expected {exp!r}")
    return mismatches


def test_drive_rollout():
    args = _build_config()
    rng = np.random.default_rng(SEED)

    vecenv = load_env("puffer_drive", args)
    try:
        env_means = _capture_env_means(vecenv, rng)
    finally:
        vecenv.close()

    print("[rollout] env_means:", json.dumps(env_means, indent=2, sort_keys=True))

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
                        "total_agents": int(vecenv.num_agents),
                        "bptt_horizon": BPTT_HORIZON,
                        "epochs": EPOCHS,
                        "seed": SEED,
                    },
                    "env": env_means,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[rollout] wrote golden -> {GOLDEN_PATH}")
        return

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    mismatches = _compare(env_means, golden["env"])
    assert not mismatches, "rollout metrics drifted from golden:\n" + "\n".join(mismatches)


if __name__ == "__main__":
    test_drive_rollout()
    print("Rollout smoke test passed!")
