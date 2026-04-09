"""Evaluate a single RL checkpoint in self-play and human-replay modes with video rendering.

Usage:
    python examples/analyze_single_cpt.py
"""

import copy
import glob
import os
import shutil

import numpy as np
import pandas as pd
import torch

from pufferlib.pufferl import load_env, load_policy, load_config
from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator

# ─── CONFIG ──────────────────────────────────────────────────────────────────
CPT_PATH = "models/scaling_cpts/unreg_classic_test.pt"

ENV_NAME = "puffer_drive"
TRAIN_MAP_DIR = "resources/drive/binaries/training_50k"
VAL_MAP_DIR = "resources/drive/binaries/validation"  # 10k maps
NUM_AGENTS_PER_VECENV = 1024
DETERMINISTIC = True
OUTPUT_CSV = "single_checkpoint_eval.csv"

# Rendering
RENDER_OUTPUT_DIR = "eval_videos"
NUM_ENVS_TO_RENDER = 1
RENDER_MODE = "worst_collision"  # "first", "random", or "worst_collision"

METRICS = [
    "n",
    "score",
    "collision_rate",
    "at_fault_collision_rate",
    "rear_collision_rate",
    "collisions_per_agent",
    "offroad_rate",
    "offroad_per_agent",
    "completion_rate",
    "route_progress",
    "lateral_error_avg",
    "episode_length",
    "episode_return",
    "perc_controlled",
]
# ─────────────────────────────────────────────────────────────────────────────


def load_checkpoint_config(checkpoint_path, fallback_config):
    """Load full_args from checkpoint if available, else use fallback ini config."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "full_args" in checkpoint:
        print(f"  Using env config stored in checkpoint.")
        return copy.deepcopy(checkpoint["full_args"]), True
    print(f"  No config found in checkpoint, falling back to ini config.")
    return copy.deepcopy(fallback_config), False


def make_eval_config(base_config, map_dir, episode_len, control_mode, goal_behavior=0, num_maps=50000):
    """Build an eval-ready config, overriding only eval-specific settings."""
    config = copy.deepcopy(base_config)
    config["env"]["map_dir"] = map_dir
    config["env"]["num_maps"] = num_maps
    config["env"]["num_agents"] = NUM_AGENTS_PER_VECENV
    config["env"]["episode_length"] = episode_len
    config["env"]["termination_mode"] = 1
    config["env"]["obs_partner_noise_speed"] = 0.0
    config["env"]["obs_partner_noise_pos"] = 0.0
    config["env"]["async_resets"] = False
    config["env"]["resample_frequency"] = 0
    config["env"]["fix_lambdas"] = True
    config["env"]["fix_rewards"] = True
    config["env"]["control_mode"] = control_mode
    config["env"]["goal_behavior"] = goal_behavior
    config["env"]["render_mode"] = 1
    config["vec"] = dict(backend="PufferEnv", num_envs=1)
    return config


def select_render_envs(evaluator, policy, env, num_to_render):
    """Run a non-rendering stats rollout and pick envs to render."""
    info_list = evaluator.rollout(policy, env, deterministic=DETERMINISTIC)
    populated = [(i, log) for i, log in enumerate(info_list) if log and log.get("n", 0) > 0]

    if not populated:
        return list(range(min(num_to_render, env.driver_env.num_envs)))

    if RENDER_MODE == "worst_collision":
        populated.sort(key=lambda x: x[1].get("collision_rate", 0.0), reverse=True)
    elif RENDER_MODE == "random":
        import random

        random.shuffle(populated)

    selected = populated[:num_to_render]
    for idx, log in selected:
        print(f"    env {idx}: collision_rate={log.get('collision_rate', 0.0):.3f}, score={log.get('score', 0.0):.3f}")
    return [idx for idx, _ in selected]


def render_envs(evaluator, policy, env, env_indices, video_dir):
    """Render each selected env and move mp4s into video_dir."""
    os.makedirs(video_dir, exist_ok=True)

    for i, env_idx in enumerate(env_indices):
        print(f"    Rendering env {env_idx} ({i + 1}/{len(env_indices)})...")
        evaluator.rollout(policy, env, render_env_idx=env_idx, deterministic=DETERMINISTIC)
        env.driver_env.stop_recorder(env_idx)

    for mp4_path in glob.glob("*.mp4"):
        dest = os.path.join(video_dir, os.path.basename(mp4_path))
        shutil.move(mp4_path, dest)
        print(f"    Saved: {dest}")


def process_rollout_data(info_list, checkpoint, mode):
    """Return one dict per populated env log."""
    rows = []
    for i, log in enumerate(info_list):
        if not log or log.get("n", 0) == 0:
            continue
        row = {"checkpoint": checkpoint, "mode": mode, "scene_idx": i}
        for key in METRICS:
            row[key] = float(log.get(key, 0.0))
        rows.append(row)
    return rows


def run_eval_and_render(checkpoint_path, base_config):
    cpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    cpt_config, _ = load_checkpoint_config(checkpoint_path, base_config)
    cpt_config["load_model_path"] = checkpoint_path

    all_rows = []

    for mode_name, map_dir, control_mode, num_maps in [
        ("sp_train", TRAIN_MAP_DIR, "control_vehicles", 50_000),
        ("sp_val", VAL_MAP_DIR, "control_vehicles", 10_000),
        ("hr_train", TRAIN_MAP_DIR, "control_sdc_only", 50_000),
    ]:
        print(f"\n{'─' * 60}")
        print(f"Mode: {mode_name}")
        print(f"{'─' * 60}")

        config = make_eval_config(
            cpt_config,
            map_dir,
            episode_len=91,
            control_mode=control_mode,
            num_maps=num_maps,
        )
        env = load_env(ENV_NAME, config)
        policy = load_policy(cpt_config, env, ENV_NAME)
        policy.eval()
        evaluator = CheckpointEvaluator(cpt_config)

        print(f"  Running stats rollout...")
        if RENDER_MODE in ("worst_collision", "random"):
            print(f"  Selecting envs to render ({RENDER_MODE})...")
            env_indices = select_render_envs(evaluator, policy, env, NUM_ENVS_TO_RENDER)
            info_list = evaluator.rollout(policy, env, deterministic=DETERMINISTIC)
        else:
            info_list = evaluator.rollout(policy, env, deterministic=DETERMINISTIC)
            env_indices = list(range(min(NUM_ENVS_TO_RENDER, env.driver_env.num_envs)))

        rows = process_rollout_data(info_list, checkpoint_path, mode_name)
        all_rows.extend(rows)

        if rows:
            print(
                f"  {len(rows)} scenes | "
                f"score={np.mean([r['score'] for r in rows]):.3f} | "
                f"collision_rate={np.mean([r['collision_rate'] for r in rows]):.3f} | "
                f"offroad_rate={np.mean([r['offroad_rate'] for r in rows]):.3f}"
            )

        video_dir = os.path.join(RENDER_OUTPUT_DIR, cpt_name, mode_name)
        print(f"  Rendering {len(env_indices)} scenarios -> {video_dir}")
        render_envs(evaluator, policy, env, env_indices, video_dir)

        env.close()

    return all_rows


def main():
    base_config = load_config(ENV_NAME)

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {CPT_PATH}")
    print(f"{'=' * 60}")

    all_rows = run_eval_and_render(CPT_PATH, base_config)

    df = pd.DataFrame(all_rows)
    # df.to_csv(OUTPUT_CSV, index=False)
    # print(f"\nResults saved to {OUTPUT_CSV} ({len(df)} rows)")

    if not df.empty:
        summary = df.groupby("mode").agg(
            scenes=("score", "count"),
            score=("score", "mean"),
            collision_rate=("collision_rate", "mean"),
            at_fault_collision_rate=("at_fault_collision_rate", "mean"),
            rear_collision_rate=("rear_collision_rate", "mean"),
            offroad_rate=("offroad_rate", "mean"),
            route_progress=("route_progress", "mean"),
            lateral_error_avg=("lateral_error_avg", "mean"),
        )
        print(f"\n{summary}")

    return df


if __name__ == "__main__":
    main()
