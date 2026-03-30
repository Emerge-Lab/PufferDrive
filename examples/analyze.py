"""Evaluate RL checkpoints across multiple eval modes.

Modes:
  1. Self-play on training maps
  2. Self-play on validation maps
  3. Human-replay on training maps
  4. Human-replay on validation maps
  5. Human-replay on interactive scenes (1k scenes selected for SDC interactivity)
  6. Scaling analysis: all checkpoints in SCALING_CHECKPOINTS_PATH on validation
     set in both self-play and human-replay modes

When NUM_TOTAL_EVAL_AGENTS > NUM_AGENTS_PER_VECENV, we keep the buffer at
NUM_AGENTS_PER_VECENV and loop resample_maps() to cover more scenes.

Output: One row per scene per mode, with checkpoint name and metrics.

Usage:
    python evaluate_checkpoints.py
"""

import copy
import os
import re

import numpy as np
import pandas as pd

from pufferlib.pufferl import load_env, load_policy, load_config
from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
CHECKPOINTS = [
    # "models/rl/pure_self_play_50k.pt",
    # "models/rl/reg_self_play_50k.pt",
]

SCALING_CHECKPOINTS_PATH = "models/scaling_metadata"

TRAIN_MAP_DIR = "resources/drive/binaries/training_50k"
VAL_MAP_DIR = "resources/drive/binaries/validation"  # 10k maps
INTERACTIVE_MAP_DIR = "resources/drive/binaries/interactive_data_training"
NUM_TOTAL_EVAL_AGENTS = 1024 * 4
NUM_AGENTS_PER_VECENV = 1024
ENV_NAME = "puffer_drive"
DATASET = "womd"
OUTPUT_CSV = "checkpoint_eval_results.csv"
MAKE_FIGURES = True
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


def make_eval_config(base_config, map_dir, control_mode, goal_behavior=0, num_maps=50000):
    """Build an eval-ready config from the base config."""
    config = copy.deepcopy(base_config)
    config["env"]["map_dir"] = map_dir
    config["env"]["num_maps"] = num_maps
    config["env"]["num_agents"] = NUM_AGENTS_PER_VECENV
    config["env"]["goal_radius"] = 2.5
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


def collect_scene_rows(info_list, checkpoint, mode, scene_offset=0, dataset=DATASET):
    """Return one dict per scene (populated env log) with checkpoint/mode metadata."""
    rows = []
    for scene_idx, log in enumerate(info_list):
        if not log or log.get("n", 0) <= 0:
            continue
        row = {
            "checkpoint": checkpoint,
            "dataset": dataset,
            "mode": mode,
            "scene_idx": scene_offset + scene_idx,
        }
        for key in METRICS:
            row[key] = float(log.get(key, 0.0))
        rows.append(row)
    return rows


def num_resample_rounds():
    """How many rollout rounds needed to cover NUM_TOTAL_EVAL_AGENTS."""
    if NUM_TOTAL_EVAL_AGENTS <= NUM_AGENTS_PER_VECENV:
        return 1
    return (NUM_TOTAL_EVAL_AGENTS + NUM_AGENTS_PER_VECENV - 1) // NUM_AGENTS_PER_VECENV


def run_mode(
    evaluator, policy, base_config, map_dir, control_mode, checkpoint, mode_name, goal_behavior=0, num_maps=50000
):
    """Create env, rollout (with resampling if needed), collect per-scene rows, close env."""
    config = make_eval_config(base_config, map_dir, control_mode, goal_behavior, num_maps)
    env = load_env(ENV_NAME, config)
    rows = []
    n_rounds = num_resample_rounds()

    try:
        for round_idx in range(n_rounds):
            if round_idx > 0:
                env.driver_env.resample_maps()

            info_list = evaluator.rollout(policy, env)
            scene_offset = round_idx * env.driver_env.num_envs
            rows.extend(collect_scene_rows(info_list, checkpoint, mode_name, scene_offset))

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
    n_rounds = num_resample_rounds()
    try:
        for round_idx in range(n_rounds):
            if round_idx > 0:
                env.driver_env.resample_maps()

            info_list = evaluator.rollout(policy, env)
            scene_offset = round_idx * env.driver_env.num_envs
            all_rows.extend(collect_scene_rows(info_list, checkpoint_path, "sp_train", scene_offset))

        sp_rows = [r for r in all_rows if r["mode"] == "sp_train"]
        if sp_rows:
            mean_score = np.mean([r["score"] for r in sp_rows])
            mean_coll = np.mean([r["collision_rate"] for r in sp_rows])
            print(f"  sp_train: {len(sp_rows)} scenes, score={mean_score:.3f}, collision_rate={mean_coll:.3f}")
    except Exception as e:
        print(f"  sp_train failed (non-fatal): {e}")
    env.close()

    # ── 2. Self-play on validation maps ──────────────────────────────────────
    all_rows.extend(
        run_mode(
            evaluator, policy, base_config, VAL_MAP_DIR, "control_agents", checkpoint_path, "sp_val", num_maps=10_000
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
            num_maps=50_000,
        )
    )

    # ── 4. Human-replay on validation maps ───────────────────────────────────
    all_rows.extend(
        run_mode(
            evaluator, policy, base_config, VAL_MAP_DIR, "control_sdc_only", checkpoint_path, "hr_val", num_maps=10_000
        )
    )

    # ── 5. Human-replay on interactive scenes ────────────────────────────────
    all_rows.extend(
        run_mode(
            evaluator,
            policy,
            base_config,
            INTERACTIVE_MAP_DIR,
            "control_sdc_only",
            checkpoint_path,
            "hr_interactive",
            num_maps=1000,
        )
    )

    return all_rows


def parse_scaling_checkpoint_name(filename):
    """Parse a scaling checkpoint filename to extract num_maps and regularization.

    Expected format: {num}k_maps_{reg|unreg}.pt
    Examples:
        1k_maps_reg.pt   -> (1000, True)
        10k_maps_unreg.pt -> (10000, False)
        500_maps_reg.pt  -> (500, True)

    Returns:
        (num_maps: int, is_regularized: bool) or None if parsing fails.
    """
    stem = filename.replace(".pt", "")
    match = re.match(r"^(\d+)(k?)_maps_(reg|unreg)$", stem)
    if not match:
        return None
    num = int(match.group(1))
    if match.group(2) == "k":
        num *= 1000
    is_reg = match.group(3) == "reg"
    return num, is_reg


def evaluate_scaling_checkpoints(base_config):
    """Evaluate all scaling checkpoints on validation set in sp and hr modes.

    For each checkpoint in SCALING_CHECKPOINTS_PATH:
      - Self-play on validation maps  (mode = "scaling_sp_val")
      - Human-replay on validation maps (mode = "scaling_hr_val")

    Extra columns added to each row: num_train_maps, is_regularized.

    Returns:
        List of per-scene row dicts.
    """
    scaling_entries = []
    for fname in sorted(os.listdir(SCALING_CHECKPOINTS_PATH)):
        if not fname.endswith(".pt"):
            continue
        parsed = parse_scaling_checkpoint_name(fname)
        if parsed is None:
            print(f"  Warning: could not parse scaling checkpoint name '{fname}', skipping")
            continue
        num_maps, is_reg = parsed
        scaling_entries.append((os.path.join(SCALING_CHECKPOINTS_PATH, fname), num_maps, is_reg))

    scaling_entries.sort(key=lambda x: (x[1], x[2]))
    if not scaling_entries:
        print("No scaling checkpoints found — skipping scaling eval.")
        return []

    print(f"\nFound {len(scaling_entries)} scaling checkpoints:")
    for cpt_path, n_maps, is_reg in scaling_entries:
        tag = "reg" if is_reg else "unreg"
        print(f"  {os.path.basename(cpt_path)}  ->  {n_maps} maps, {tag}")

    all_rows = []
    for cpt_path, n_maps, is_reg in scaling_entries:
        print(f"\n{'─' * 60}")
        print(f"Scaling eval: {os.path.basename(cpt_path)}")
        print(f"{'─' * 60}")

        # Bootstrap an env so load_policy has a vecenv to inspect
        init_config = make_eval_config(base_config, VAL_MAP_DIR, control_mode="control_agents", num_maps=10_000)
        env = load_env(ENV_NAME, init_config)
        base_config["load_model_path"] = cpt_path
        policy = load_policy(base_config, env, ENV_NAME)
        policy.eval()
        env.close()

        evaluator = CheckpointEvaluator(base_config)

        # ── Self-play on training ───────────────────────────────────────
        sp_train_rows = run_mode(
            evaluator,
            policy,
            base_config,
            TRAIN_MAP_DIR,
            "control_agents",
            cpt_path,
            "scaling_sp_train",
            num_maps=50_000,
        )

        # ── Self-play on validation ──────────────────────────────────────
        sp_rows = run_mode(
            evaluator,
            policy,
            base_config,
            VAL_MAP_DIR,
            "control_agents",
            cpt_path,
            "scaling_sp_val",
            num_maps=10_000,
        )

        # ── Human-replay on validation ──────────────────────────────────
        hr_rows = run_mode(
            evaluator,
            policy,
            base_config,
            VAL_MAP_DIR,
            "control_sdc_only",
            cpt_path,
            "scaling_hr_val",
            num_maps=10_000,
        )

        # Attach scaling metadata to every row
        for row in sp_train_rows + sp_rows + hr_rows:
            row["num_train_maps"] = n_maps
            row["is_regularized"] = is_reg

        all_rows.extend(sp_train_rows)
        all_rows.extend(sp_rows)
        all_rows.extend(hr_rows)

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

    # ── 6. Scaling analysis ──────────────────────────────────────────────────
    scaling_rows = evaluate_scaling_checkpoints(base_config)
    all_rows.extend(scaling_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV} ({len(df)} rows)")

    # Print summary per checkpoint and mode
    if not df.empty:
        summary = df.groupby(["checkpoint", "mode"])[["score", "collision_rate", "offroad_rate"]].mean()
        print(f"\n{summary}")

    if MAKE_FIGURES and not df.empty:
        from pufferlib.ocean.benchmark.plot import make_all_figures

        make_all_figures(df)

    return df


if __name__ == "__main__":
    main()
