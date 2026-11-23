"""Starter eval for the spiced self-play checkpoints on a small WOMD slice.

Compares the regularized policy (30 minutes of human anchor data, lambda 0.075)
with the unregularized 50k-map self-play baseline. Reports self-play score and
collision rate, and human-replay at-fault collision rate.

Run from the repo root:

    python reproduce/eval.py

The scenario slice is the first ``--num-maps`` binaries in
``resources/drive/binaries/validation`` (held-out Waymo Open Motion validation).
The full 10k set is https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_val
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator
from pufferlib.pufferl import load_config, load_env, load_policy


ENV_NAME = "puffer_drive"
MAP_DIR = ROOT / "resources" / "drive" / "binaries" / "validation"
CHECKPOINTS = (
    ("regularized", ROOT / "reproduce" / "policies" / "reg_delta_50k_maps_anchor_200_maps.pt"),
    ("unregularized", ROOT / "reproduce" / "policies" / "unreg_delta_50k_maps.pt"),
)
MODES = (
    ("self_play", "control_vehicles"),
    ("human_replay", "control_sdc_only"),
)
EPISODE_LENGTH = 200
DATASET_URL = "https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_val"


def load_base_config():
    """load_config reads sys.argv, so hide this script's flags while it runs."""
    saved = sys.argv[:]
    try:
        sys.argv = [saved[0]]
        return load_config(ENV_NAME)
    finally:
        sys.argv = saved


def checkpoint_config(path, fallback):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "full_args" in checkpoint:
        config = copy.deepcopy(checkpoint["full_args"])
    else:
        config = copy.deepcopy(fallback)
    config["load_model_path"] = str(path)
    if torch.cuda.is_available():
        config["train"]["device"] = "cuda"
    return config


def make_eval_config(base, control_mode, num_maps, num_agents, map_dir=None, seed=1):
    """Same overrides as examples/analyze.py, on a small map slice.

    lambda_value is left at the value stored in the checkpoint: 0.075 for the
    regularized policy and 0.0 for the unregularized one.
    """
    config = copy.deepcopy(base)
    env = config["env"]
    env["map_dir"] = str(MAP_DIR if map_dir is None else map_dir)
    env["control_mode"] = control_mode
    env["num_maps"] = num_maps
    env["num_agents"] = num_agents
    env["episode_length"] = EPISODE_LENGTH
    env["seed"] = seed
    env["async_resets"] = False
    env["goal_behavior"] = 0
    env["render_mode"] = 1
    env["termination_mode"] = 1
    env["fix_lambdas"] = True
    env["fix_rewards"] = True
    env["obs_partner_noise_speed"] = 0.0
    env["obs_partner_noise_pos"] = 0.0
    env["resample_frequency"] = 0
    config["vec"] = dict(backend="PufferEnv", num_envs=1)
    return config


def populated_logs(info_list):
    return [log for log in info_list if log and log.get("n", 0) > 0]


def mean_metric(logs, key):
    if not logs:
        return float("nan")
    return float(np.mean([float(log.get(key, 0.0)) for log in logs]))


def load_policy_once(config, num_maps, num_agents):
    probe = make_eval_config(config, "control_sdc_only", num_maps, num_agents)
    env = load_env(ENV_NAME, probe)
    try:
        policy = load_policy(probe, env, ENV_NAME)
        policy.eval()
    finally:
        env.close()
    return policy


def evaluate_mode(policy, config, control_mode, num_maps, num_agents, seed=1):
    eval_config = make_eval_config(config, control_mode, num_maps, num_agents, seed=seed)
    env = load_env(ENV_NAME, eval_config)
    try:
        evaluator = CheckpointEvaluator(eval_config)
        info = evaluator.rollout(env=env, policy=policy, deterministic=False)
        return populated_logs(info)
    finally:
        env.close()


def print_row(name, mode, logs):
    print(
        f"{name:<16} {mode:<14} {len(logs):6d} "
        f"{mean_metric(logs, 'score'):8.3f} "
        f"{100 * mean_metric(logs, 'collision_rate'):8.2f} "
        f"{100 * mean_metric(logs, 'at_fault_collision_rate'):8.2f}"
    )


def rollout_until(policy, config, control_mode, num_maps, num_agents, target_scenes):
    """One rollout, or several seeds, until target_scenes are collected."""
    logs = []
    seed = 1
    while True:
        print(f"  rollout seed {seed}", flush=True)
        batch = evaluate_mode(policy, config, control_mode, num_maps, num_agents, seed=seed)
        logs.extend(batch)
        if target_scenes <= 0 or len(logs) >= target_scenes or not batch:
            break
        seed += 1
        if seed > max(40, target_scenes):
            break
    if target_scenes > 0:
        logs = logs[:target_scenes]
    return logs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-maps", type=int, default=128, help="First N validation maps to sample from")
    parser.add_argument("--num-agents", type=int, default=1024, help="Controlled agents per rollout")
    parser.add_argument(
        "--target-scenes",
        type=int,
        default=0,
        help="Keep sampling new seeds until each mode has this many scenes",
    )
    args = parser.parse_args()

    if not (MAP_DIR / "map_000.bin").exists():
        raise SystemExit(
            f"Missing {MAP_DIR / 'map_000.bin'}.\n"
            f"Download the held-out WOMD validation scenarios from {DATASET_URL}\n"
            "and convert them into resources/drive/binaries/validation."
        )
    for _, path in CHECKPOINTS:
        if not path.exists():
            raise SystemExit(f"Missing checkpoint {path}")

    print(f"Scenarios: {MAP_DIR}/map_000.bin .. map_{args.num_maps - 1:03d}.bin")
    print(f"Full validation set: {DATASET_URL}")
    print(f"{'policy':<16} {'mode':<14} {'scenes':>6} {'score':>8} {'coll%':>8} {'at_fault%':>8}")

    fallback = load_base_config()
    for name, path in CHECKPOINTS:
        print(f"Loading {name}: {path.name}", flush=True)
        config = checkpoint_config(path, fallback)
        policy = load_policy_once(config, num_maps=1, num_agents=1)
        for mode_name, control_mode in MODES:
            print(f"  rollout {mode_name}", flush=True)
            logs = rollout_until(policy, config, control_mode, args.num_maps, args.num_agents, args.target_scenes)
            print_row(name, mode_name, logs)


if __name__ == "__main__":
    main()
