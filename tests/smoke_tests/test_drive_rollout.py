"""Deterministic CPU smoke test: the simulator alone under seeded random actions against a committed golden."""

import json
from collections import defaultdict

import numpy as np
from smoke_golden import (
    ATOL,
    BPTT_HORIZON,
    SEED,
    assert_infractions_occur,
    build_config,
    is_number,
    record_or_compare_golden,
)

import opengigaflow
from opengigaflow.ocean.drive import binding
from opengigaflow.rl import load_env


EPOCHS = 5


def _jerk_action_table():
    """Continuous action of every class of the policy's jerk grid, class = long_idx * num_lat + lat_idx."""
    jerk_long = np.asarray(binding.JERK_LONG, dtype=np.float32)
    jerk_lat = np.asarray(binding.JERK_LAT, dtype=np.float32)
    long_norm = np.where(jerk_long < 0.0, jerk_long / -jerk_long[0], jerk_long / jerk_long[-1])
    lat_norm = jerk_lat / jerk_lat[-1]
    class_idx = np.arange(len(jerk_long) * len(jerk_lat))
    return np.stack([long_norm[class_idx // len(jerk_lat)], lat_norm[class_idx % len(jerk_lat)]], axis=-1)


def _capture_env_means(vecenv, rng):
    env_acc = defaultdict(list)
    action_table = _jerk_action_table()
    vecenv.reset(seed=SEED)
    for _ in range(EPOCHS * BPTT_HORIZON):
        actions = action_table[rng.integers(0, len(action_table), size=vecenv.num_agents)]
        _, _, _, _, infos = vecenv.step(actions)
        for info in infos:
            for k, v in opengigaflow.unroll_nested_dict(info):
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                if isinstance(v, (list, tuple)):
                    env_acc[k].extend(v)
                else:
                    env_acc[k].append(v)

    env_means = {}
    for k, vals in env_acc.items():
        nums = [x for x in vals if is_number(x)]
        if nums and len(nums) == len(vals):
            env_means[k] = float(np.mean(nums))
    return env_means


def test_drive_rollout():
    args = build_config(num_envs=8)
    rng = np.random.default_rng(SEED)

    vecenv = load_env(args)
    try:
        env_means = _capture_env_means(vecenv, rng)
    finally:
        vecenv.close()

    print("[rollout] env_means:", json.dumps(env_means, indent=2, sort_keys=True))

    assert_infractions_occur(env_means)
    # the rollout golden keeps its stricter tolerance: plain atol for episode_return, NaN never matches
    record_or_compare_golden(
        "drive_rollout_golden.json", vecenv.num_agents, EPOCHS, {"env": env_means}, loose_atol=ATOL, equal_nan=False
    )


if __name__ == "__main__":
    test_drive_rollout()
    print("Rollout smoke test passed!")
