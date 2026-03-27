"""Evaluate RL checkpoints across multiple eval modes.

Modes:
  1. Self-play on training maps
  2. Self-play on validation maps
  3. Human-replay on training maps
  4. Human-replay on validation maps

Output: One row per scene per mode, with checkpoint name and metrics.

Usage:
    python evaluate_checkpoints.py
"""

import copy
import numpy as np
import pandas as pd

from pufferlib.pufferl import load_env, load_policy, load_config
from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
CHECKPOINTS = [
    "models/rl/pure_self_play_50k.pt",
    "models/rl/reg_self_play_50k.pt",
]

TRAIN_MAP_DIR = "resources/drive/binaries/training_50k"
VAL_MAP_DIR = "resources/drive/binaries/validation"
NUM_EVAL_AGENTS = 1024
ENV_NAME = "puffer_drive"
DATASET = "womd"
OUTPUT_CSV = "checkpoint_eval_results.csv"
# ────────────────────────────────────────────────────────────────────────────────

METRICS = [
    "n",
    "score",
    "collision_rate",
    "collisions_per_agent",
    "offroad_rate",
    "offroad_per_agent",
    "completion_rate",
    "episode_length",
    "episode_return",
    "perc_controlled",
]


def make_eval_config(base_config, map_dir, control_mode, goal_behavior=0):
    """Build an eval-ready config from the base config."""
    config = copy.deepcopy(base_config)
    config["env"]["map_dir"] = map_dir
    config["env"]["num_agents"] = NUM_EVAL_AGENTS
    config["env"]["episode_length"] = 150
    config["env"]["termination_mode"] = 1
    config["env"]["control_mode"] = control_mode
    config["env"]["goal_behavior"] = goal_behavior
    config["env"]["fix_lambdas"] = True
    config["env"]["fix_rewards"] = True
    config["env"]["obs_partner_noise_speed"] = 0.0
    config["env"]["obs_partner_noise_pos"] = 0.0
    config["vec"] = dict(backend="PufferEnv", num_envs=1)
    return config


def collect_scene_rows(info_list, checkpoint, mode, dataset=DATASET):
    """Return one dict per scene (populated env log) with checkpoint/mode metadata."""
    rows = []
    for scene_idx, log in enumerate(info_list):
        if not log or log.get("n", 0) <= 0:
            continue
        row = {"checkpoint": checkpoint, "dataset": dataset, "mode": mode, "scene_idx": scene_idx}
        for key in METRICS:
            row[key] = float(log.get(key, 0.0))
        rows.append(row)
    return rows


def run_mode(evaluator, policy, base_config, map_dir, control_mode, checkpoint, mode_name, goal_behavior=0):
    """Create env, rollout, collect per-scene rows, close env."""
    config = make_eval_config(base_config, map_dir, control_mode, goal_behavior)
    env = load_env(ENV_NAME, config)
    rows = []
    try:
        info_list = evaluator.rollout(policy, env)
        rows = collect_scene_rows(info_list, checkpoint, mode_name)
        n_scenes = len(rows)
        if n_scenes > 0:
            mean_score = np.mean([r["score"] for r in rows])
            mean_coll = np.mean([r["collision_rate"] for r in rows])
            print(f"  {mode_name}: {n_scenes} scenes, score={mean_score:.3f}, collision_rate={mean_coll:.3f}")
        else:
            print(f"  {mode_name}: no populated scenes")
    except Exception as e:
        print(f"  {mode_name} failed (non-fatal): {e}")
    env.close()
    return rows


def evaluate_checkpoint(checkpoint_path, base_config):
    """Run all eval modes for a single checkpoint. Returns list of per-scene dicts."""

    # Create first env before loading policy (load_policy needs vecenv.driver_env)
    sp_train_config = make_eval_config(
        base_config,
        TRAIN_MAP_DIR,
        control_mode="control_agents",
        goal_behavior=0,
    )
    env = load_env(ENV_NAME, sp_train_config)

    # load_policy reads load_model_path for checkpoint weights
    base_config["load_model_path"] = checkpoint_path
    policy = load_policy(base_config, env, ENV_NAME)
    policy.eval()

    evaluator = CheckpointEvaluator(base_config)
    all_rows = []

    # ── 1. Self-play on training maps (reuse the env we already created) ─────
    try:
        info_list = evaluator.rollout(policy, env)
        all_rows.extend(collect_scene_rows(info_list, checkpoint_path, "sp_train"))
        n = len([r for r in all_rows if r["mode"] == "sp_train"])
        if n > 0:
            mean_score = np.mean([r["score"] for r in all_rows if r["mode"] == "sp_train"])
            mean_coll = np.mean([r["collision_rate"] for r in all_rows if r["mode"] == "sp_train"])
            print(f"  sp_train: {n} scenes, score={mean_score:.3f}, collision_rate={mean_coll:.3f}")
    except Exception as e:
        print(f"  sp_train failed (non-fatal): {e}")
    env.close()

    # ── 2. Self-play on validation maps ──────────────────────────────────────
    all_rows.extend(
        run_mode(
            evaluator,
            policy,
            base_config,
            VAL_MAP_DIR,
            "control_agents",
            checkpoint_path,
            "sp_val",
        )
    )

    # ── 3. Human-replay on training maps ─────────────────────────────────────
    all_rows.extend(
        run_mode(
            evaluator,
            policy,
            base_config,
            TRAIN_MAP_DIR,
            "control_sdc_only",
            checkpoint_path,
            "hr_train",
        )
    )

    # ── 4. Human-replay on validation maps ───────────────────────────────────
    all_rows.extend(
        run_mode(
            evaluator,
            policy,
            base_config,
            VAL_MAP_DIR,
            "control_sdc_only",
            checkpoint_path,
            "hr_val",
        )
    )

    return all_rows


def main():
    base_config = load_config(ENV_NAME)

    all_rows = []
    for cpt_path in CHECKPOINTS:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {cpt_path}")
        print(f"{'=' * 60}")
        rows = evaluate_checkpoint(cpt_path, base_config)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV} ({len(df)} rows)")

    # Print summary per checkpoint and mode
    if not df.empty:
        summary = df.groupby(["checkpoint", "mode"])[["score", "collision_rate", "offroad_rate"]].mean()
        print(f"\n{summary}")

    return df


if __name__ == "__main__":
    main()
