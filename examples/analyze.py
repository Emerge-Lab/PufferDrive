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

Checkpoint naming convention (scaling):
  [reg|unreg]_[dynamics]_[N]_maps[_anchor_[M]_maps].pt

  Examples:
    reg_delta_100_maps_anchor_50k_maps.pt
      -> regularized, delta dynamics, 100 self-play maps, 50k anchor maps
    unreg_delta_10_maps.pt
      -> unregularized, delta dynamics, 10 self-play maps, no anchor

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
from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
CHECKPOINTS = [
    # "models/rl/pure_self_play_50k.pt",
    # "models/rl/reg_self_play_50k.pt",
]

SCALING_CHECKPOINTS_PATH = "models/cpts_scaling"

TRAIN_MAP_DIR = "resources/drive/binaries/training_50k"
VAL_MAP_DIR = "resources/drive/binaries/validation"  # 10k maps
INTERACTIVE_MAP_DIR = "resources/drive/binaries/interactive_data_training"
NUM_TOTAL_EVAL_AGENTS = 100  # * 4
NUM_AGENTS_PER_VECENV = 100  # 1024
ENV_NAME = "puffer_drive"
DATASET = "womd"
OUTPUT_CSV = "checkpoint_eval_results.csv"
WOSAC_OUTPUT_CSV = "checkpoint_wosac_results.csv"
MAKE_FIGURES = True
RUN_WOSAC = False
WOSAC_ONLY = False
RUN_RENDER = True

# ─── VIDEO RENDERING CONFIG ─────────────────────────────────────────────────
CHECKPOINTS_TO_RENDER = [
    "models/cpts_scaling/unreg_delta_10_maps.pt",
    "models/cpts_scaling/reg_delta_1k_maps_anchor_100_maps.pt",
]
NUM_ENVS_TO_RENDER = 15
RENDER_MAP_DIR = INTERACTIVE_MAP_DIR  # Which maps to render on
RENDER_NUM_MAPS = 1000
RENDER_OUTPUT_DIR = "eval_videos"

# WOSAC evaluation settings (aligned with run_wosac_eval_in_subprocess defaults)
WOSAC_NUM_ROLLOUTS = 6
WOSAC_TARGET_SCENARIOS = 512
WOSAC_MAX_BATCHES = 1
WOSAC_INIT_STEPS = 0
WOSAC_INIT_MODE = "create_all_valid"
WOSAC_CONTROL_MODE = "control_wosac"
WOSAC_GOAL_BEHAVIOR = 2
WOSAC_GOAL_RADIUS = 2.5
WOSAC_SCENARIO_POOL_SIZE = 10_000
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


def _parse_num(s):
    """Parse a number string like '10', '1k', '50k' into an integer."""
    m = re.match(r"(\d+)(k?)", s)
    if not m:
        return None
    n = int(m.group(1))
    if m.group(2) == "k":
        n *= 1000
    return n


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
        control_mode="control_vehicles",
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
            evaluator, policy, base_config, VAL_MAP_DIR, "control_vehicles", checkpoint_path, "sp_val", num_maps=10_000
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
    """Parse a scaling checkpoint filename.

    Format: [reg|unreg]_[dynamics]_[N]_maps[_anchor_[M]_maps].pt

    Examples:
        reg_delta_100_maps_anchor_50k_maps.pt  -> (100, True, "delta", 50000)
        unreg_delta_10_maps.pt                 -> (10, False, "delta", None)

    Returns:
        (sp_maps: int, is_regularized: bool, dynamics: str, anchor_maps: int|None)
        or None if parsing fails.
    """
    stem = filename.replace(".pt", "")

    # Try regularized with anchor: reg_delta_100_maps_anchor_50k_maps
    match = re.match(r"^(reg|unreg)_(\w+?)_(\d+k?)_maps_anchor_(\d+k?)_maps$", stem)
    if match:
        is_reg = match.group(1) == "reg"
        dynamics = match.group(2)
        sp_maps = _parse_num(match.group(3))
        anchor_maps = _parse_num(match.group(4))
        return sp_maps, is_reg, dynamics, anchor_maps

    # Try without anchor: unreg_delta_10_maps
    match = re.match(r"^(reg|unreg)_(\w+?)_(\d+k?)_maps$", stem)
    if match:
        is_reg = match.group(1) == "reg"
        dynamics = match.group(2)
        sp_maps = _parse_num(match.group(3))
        return sp_maps, is_reg, dynamics, None

    return None


def evaluate_scaling_checkpoints(base_config):
    """Evaluate all scaling checkpoints on validation set in sp and hr modes.

    For each checkpoint in SCALING_CHECKPOINTS_PATH:
      - Self-play on validation maps  (mode = "scaling_sp_val")
      - Human-replay on validation maps (mode = "scaling_hr_val")

    Extra columns added to each row:
      - sp_maps: number of maps used for self-play training
      - is_regularized: whether regularization was used
      - dynamics: dynamics model name (e.g. "delta")
      - anchor_maps: number of maps used to train the anchor (None for unreg)

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
        sp_maps, is_reg, dynamics, anchor_maps = parsed
        scaling_entries.append(
            (
                os.path.join(SCALING_CHECKPOINTS_PATH, fname),
                sp_maps,
                is_reg,
                dynamics,
                anchor_maps,
            )
        )

    scaling_entries.sort(key=lambda x: (x[1], x[2], x[4] or 0))
    if not scaling_entries:
        print("No scaling checkpoints found — skipping scaling eval.")
        return []

    print(f"\nFound {len(scaling_entries)} scaling checkpoints:")
    for cpt_path, sp_maps, is_reg, dynamics, anchor_maps in scaling_entries:
        tag = "reg" if is_reg else "unreg"
        anchor_str = f"anchor={anchor_maps}" if anchor_maps is not None else "no anchor"
        print(f"  {os.path.basename(cpt_path)}  ->  sp={sp_maps}, {tag}, {dynamics}, {anchor_str}")

    all_rows = []
    for cpt_path, sp_maps, is_reg, dynamics, anchor_maps in scaling_entries:
        print(f"\n{'─' * 60}")
        print(f"Scaling eval: {os.path.basename(cpt_path)}")
        print(f"{'─' * 60}")

        # Bootstrap an env so load_policy has a vecenv to inspect
        init_config = make_eval_config(base_config, VAL_MAP_DIR, control_mode="control_vehicles", num_maps=10_000)
        env = load_env(ENV_NAME, init_config)
        base_config["load_model_path"] = cpt_path
        policy = load_policy(base_config, env, ENV_NAME)
        policy.eval()
        env.close()

        evaluator = CheckpointEvaluator(base_config)

        # ── Self-play on validation ──────────────────────────────────────
        sp_rows = run_mode(
            evaluator,
            policy,
            base_config,
            VAL_MAP_DIR,
            "control_vehicles",
            cpt_path,
            "scaling_sp_val",
            num_maps=10_000,
        )

        # ── Human-replay on interactive scenes ───────────────────────
        hr_rows = run_mode(
            evaluator,
            policy,
            base_config,
            INTERACTIVE_MAP_DIR,
            "control_sdc_only",
            cpt_path,
            "scaling_hr_interactive",
            num_maps=1_000,
        )

        # Attach scaling metadata to every row
        for row in sp_rows + hr_rows:
            row["sp_maps"] = sp_maps
            row["is_regularized"] = is_reg
            row["dynamics"] = dynamics
            row["anchor_maps"] = anchor_maps  # None for unreg

        all_rows.extend(sp_rows)
        all_rows.extend(hr_rows)

    return all_rows


def make_wosac_config(base_config, map_dir, num_maps=None):
    """Build a config suitable for WOSACEvaluator from the base config.

    Settings aligned with run_wosac_eval_in_subprocess defaults.
    """
    num_maps = num_maps or WOSAC_SCENARIO_POOL_SIZE
    config = copy.deepcopy(base_config)
    config["env"]["map_dir"] = map_dir
    config["env"]["num_maps"] = num_maps
    config["env"]["num_agents"] = NUM_AGENTS_PER_VECENV
    config["env"]["control_mode"] = WOSAC_CONTROL_MODE
    config["env"]["goal_behavior"] = WOSAC_GOAL_BEHAVIOR
    config["env"]["goal_radius"] = WOSAC_GOAL_RADIUS
    config["env"]["init_mode"] = WOSAC_INIT_MODE
    config["env"]["episode_length"] = 91
    config["env"]["termination_mode"] = 1
    config["env"]["fix_lambdas"] = True
    config["env"]["fix_rewards"] = True
    config["env"]["obs_partner_noise_speed"] = 0.0
    config["env"]["obs_partner_noise_pos"] = 0.0
    config["vec"] = dict(backend="PufferEnv", num_envs=1)
    config.setdefault("eval", {})
    config["eval"]["wosac_num_rollouts"] = WOSAC_NUM_ROLLOUTS
    config["eval"]["wosac_target_scenarios"] = WOSAC_TARGET_SCENARIOS
    config["eval"]["wosac_max_batches"] = WOSAC_MAX_BATCHES
    config["eval"]["wosac_init_steps"] = WOSAC_INIT_STEPS
    config["eval"]["wosac_filter_out_post_done"] = True
    config["eval"]["wosac_sanity_check"] = False
    return config


WOSAC_METRICS = [
    "realism_meta_score",
    "kinematic_metrics",
    "interactive_metrics",
    "map_based_metrics",
]


def evaluate_scaling_wosac(base_config):
    """Run WOSAC evaluation for all scaling checkpoints on the validation set.

    Returns:
        pd.DataFrame with one row per scenario per checkpoint, containing
        WOSAC metrics and scaling metadata columns.
    """
    scaling_entries = []
    for fname in sorted(os.listdir(SCALING_CHECKPOINTS_PATH)):
        if not fname.endswith(".pt"):
            continue
        parsed = parse_scaling_checkpoint_name(fname)
        if parsed is None:
            continue
        sp_maps, is_reg, dynamics, anchor_maps = parsed
        scaling_entries.append(
            (
                os.path.join(SCALING_CHECKPOINTS_PATH, fname),
                sp_maps,
                is_reg,
                dynamics,
                anchor_maps,
            )
        )

    scaling_entries.sort(key=lambda x: (x[1], x[2], x[4] or 0))
    if not scaling_entries:
        print("No scaling checkpoints found — skipping WOSAC eval.")
        return pd.DataFrame()

    print(f"\nWOSAC evaluation for {len(scaling_entries)} scaling checkpoints:")

    all_dfs = []
    for cpt_path, sp_maps, is_reg, dynamics, anchor_maps in scaling_entries:
        print(f"\n{'─' * 60}")
        print(f"WOSAC eval: {os.path.basename(cpt_path)}")
        print(f"{'─' * 60}")

        wosac_config = make_wosac_config(base_config, VAL_MAP_DIR, num_maps=10_000)
        env = load_env(ENV_NAME, wosac_config)
        base_config["load_model_path"] = cpt_path
        policy = load_policy(base_config, env, ENV_NAME)
        policy.eval()

        wosac_eval = WOSACEvaluator(wosac_config)
        try:
            df_scenes = wosac_eval.evaluate(wosac_config, env, policy, drop_scene_duplicates=True)

            # Keep only the metrics we need, reset index to get scenario_id as column
            df_scenes = df_scenes[WOSAC_METRICS].copy()
            df_scenes = df_scenes.reset_index()
            df_scenes["checkpoint"] = cpt_path
            df_scenes["sp_maps"] = sp_maps
            df_scenes["is_regularized"] = is_reg
            df_scenes["dynamics"] = dynamics
            df_scenes["anchor_maps"] = anchor_maps

            n = len(df_scenes)
            meta = df_scenes["realism_meta_score"].mean()
            print(f"  WOSAC: {n} scenarios, realism_meta_score={meta:.4f}")
            all_dfs.append(df_scenes)
        except Exception as e:
            print(f"  WOSAC failed (non-fatal): {e}")

        env.close()

    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def make_render_config(base_config, map_dir, num_maps=1000):
    """Build a config for human-replay rendering with headless ffmpeg output."""
    config = make_eval_config(
        base_config,
        map_dir,
        control_mode="control_sdc_only",
        goal_behavior=0,
        num_maps=num_maps,
    )
    config["env"]["render_mode"] = 1  # Headless ffmpeg
    return config


def render_checkpoint_videos(base_config):
    """Render human-replay videos for each checkpoint in CHECKPOINTS_TO_RENDER.

    Uses driver.reset_recorder() between env indices to flush the current mp4
    and start a new one, keeping the raylib window alive across all renders.
    """
    import glob
    import shutil

    env = None

    for cpt_path in CHECKPOINTS_TO_RENDER:
        cpt_name = os.path.splitext(os.path.basename(cpt_path))[0]
        cpt_video_dir = os.path.join(RENDER_OUTPUT_DIR, cpt_name)
        os.makedirs(cpt_video_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Rendering videos: {cpt_name}")
        print(f"{'=' * 60}")

        # Create env once (or reuse across checkpoints — same map config)
        if env is None:
            config = make_render_config(base_config, RENDER_MAP_DIR, num_maps=RENDER_NUM_MAPS)
            env = load_env(ENV_NAME, config)

        base_config["load_model_path"] = cpt_path
        policy = load_policy(base_config, env, ENV_NAME)
        policy.eval()

        evaluator = CheckpointEvaluator(base_config)

        for env_idx in range(NUM_ENVS_TO_RENDER):
            print(f"  Rendering env {env_idx + 1}/{NUM_ENVS_TO_RENDER}...")
            evaluator.rollout(policy, env, render_env_idx=env_idx)
            # Flush this scenario's mp4 and stop the recorder cleanly.
            # The next c_render call will auto-start a new recorder.
            env.driver_env.stop_recorder(env_idx)

        # Move mp4s produced by ffmpeg into the checkpoint subdirectory
        for mp4_path in glob.glob("*.mp4"):
            dest = os.path.join(cpt_video_dir, os.path.basename(mp4_path))
            shutil.move(mp4_path, dest)
            print(f"  Saved: {dest}")

    if env is not None:
        env.close()

    print(f"\nAll videos saved to {RENDER_OUTPUT_DIR}/")


def main():
    base_config = load_config(ENV_NAME)

    df = pd.DataFrame()
    wosac_df = pd.DataFrame()

    if not WOSAC_ONLY:
        all_rows = []
        for cpt_path in CHECKPOINTS:
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {cpt_path}")
            print(f"{'=' * 60}")
            rows = evaluate_checkpoint(cpt_path, base_config)
            all_rows.extend(rows)

        # ── Scaling analysis ─────────────────────────────────────────────────
        scaling_rows = evaluate_scaling_checkpoints(base_config)
        all_rows.extend(scaling_rows)

        df = pd.DataFrame(all_rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nResults saved to {OUTPUT_CSV} ({len(df)} rows)")

        if not df.empty:
            summary = df.groupby(["checkpoint", "mode"])[["score", "collision_rate", "offroad_rate"]].mean()
            print(f"\n{summary}")

    # ── WOSAC scaling analysis ───────────────────────────────────────────────
    if RUN_WOSAC:
        wosac_df = evaluate_scaling_wosac(base_config)
        if not wosac_df.empty:
            wosac_df.to_csv(WOSAC_OUTPUT_CSV, index=False)
            print(f"\nWOSAC results saved to {WOSAC_OUTPUT_CSV} ({len(wosac_df)} rows)")

    # ── Figures ──────────────────────────────────────────────────────────────
    if MAKE_FIGURES:
        from pufferlib.ocean.benchmark.plot import make_all_figures

        make_all_figures(
            df if not df.empty else None,
            wosac_df if not wosac_df.empty else None,
        )

    # ── Video rendering (last: env.close() may segfault in raylib cleanup) ──
    if RUN_RENDER and CHECKPOINTS_TO_RENDER:
        render_checkpoint_videos(base_config)

    return df


if __name__ == "__main__":
    main()
