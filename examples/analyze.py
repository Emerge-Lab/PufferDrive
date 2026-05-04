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
import torch

from pufferlib.pufferl import load_env, load_policy, load_config
from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
SCALING_CHECKPOINTS_PATH = "models/scaling_cpts"
DETERMINISTIC = True

TRAIN_MAP_DIR = "resources/drive/binaries/training"  # 50k maps
VAL_MAP_DIR = "resources/drive/binaries/validation"  # 10k maps
INTERACTIVE_MAP_DIR = "resources/drive/binaries/interactive_data_validation"  # 200 maps selected for SDC interactivity
INTERACTIVE_MAP_DIR_MAPS = 50
NUM_TOTAL_EVAL_AGENTS = 1024 * 5
NUM_AGENTS_PER_VECENV = 1024
ENV_NAME = "puffer_drive"
DATASET = "womd"
OUTPUT_CSV = "results/checkpoint_eval_results.csv"
MAKE_FIGURES = True
RUN_RENDER = False

# ─── VIDEO RENDERING CONFIG ─────────────────────────────────────────────────
CHECKPOINTS_TO_RENDER = ["models/scaling_cpts/unreg_classic_50k_maps.pt"]
NUM_ENVS_TO_RENDER = 0
RENDER_MAP_DIR = INTERACTIVE_MAP_DIR  # Which maps to render on
RENDER_NUM_MAPS = 200
RENDER_OUTPUT_DIR = "eval_videos"
RENDER_MODE = "worst_collision"  # "random" or "worst_collision"
# ────────────────────────────────────────────────────────────────────────────────

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
    "longitudinal_error_avg",
]


def load_checkpoint_config(checkpoint_path, fallback_config):
    """Load full_args from checkpoint if available, else use fallback ini config."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "full_args" in checkpoint:
        print(f"  Using env config stored in checkpoint.")
        return copy.deepcopy(checkpoint["full_args"]), True
    print(f"  No config found in checkpoint, falling back to ini config.")
    return copy.deepcopy(fallback_config), False


def _parse_num(s):
    """Parse a number string like '10', '1k', '50k' into an integer."""
    m = re.match(r"(\d+)(k?)", s)
    if not m:
        return None
    n = int(m.group(1))
    if m.group(2) == "k":
        n *= 1000
    return n


def make_eval_config(cpt_config, map_dir, control_mode, num_maps, episode_length=150):
    """Build an eval-ready config from the checkpoint config.

    Takes everything from the checkpoint and only overwrites eval-specific fields:
    map_dir, control_mode, num_maps, and optionally episode_length.
    """
    config = copy.deepcopy(cpt_config)
    config["env"]["map_dir"] = map_dir
    config["env"]["control_mode"] = control_mode
    config["env"]["num_maps"] = num_maps
    config["env"]["num_agents"] = NUM_AGENTS_PER_VECENV

    # Fixed: Important for getting valid stats
    config["env"]["async_resets"] = False

    config["env"]["goal_behavior"] = 0
    config["env"]["render_mode"] = 1
    config["env"]["termination_mode"] = 1
    config["env"]["fix_lambdas"] = True
    config["env"]["fix_rewards"] = True
    config["env"]["obs_partner_noise_speed"] = 0.0
    config["env"]["obs_partner_noise_pos"] = 0.0
    config["env"]["termination_mode"] = 1
    if episode_length is not None:
        config["env"]["episode_length"] = episode_length
    config["vec"] = dict(backend="PufferEnv", num_envs=1)
    return config


def process_rollout_data(info_list, checkpoint, mode, scene_offset=0, dataset=DATASET):
    """Return one dict per rollout (populated env log) with checkpoint/mode metadata."""

    populated = [log for log in info_list if log and log.get("n", 0) > 0]
    if not populated:
        return []
    rows = []
    for i, log in enumerate(populated):
        row = {
            "checkpoint": checkpoint,
            "dataset": dataset,
            "mode": mode,
            "scene_idx": scene_offset + i,  # TODO: Fix
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


def run_mode(evaluator, policy, cpt_config, map_dir, control_mode, checkpoint, mode_name, num_maps):
    """Create env, rollout (with resampling if needed), collect per-scene rows, close env."""
    config = make_eval_config(cpt_config, map_dir, control_mode, num_maps)
    env = load_env(ENV_NAME, config)
    rows = []
    n_rounds = num_resample_rounds()

    try:
        for round_idx in range(n_rounds):
            if round_idx > 0:
                env.driver_env.resample_maps()

            rollout_stats = evaluator.rollout(env=env, policy=policy, deterministic=DETERMINISTIC)
            scene_offset = round_idx * env.driver_env.num_envs
            rows.extend(process_rollout_data(rollout_stats, checkpoint, mode_name, scene_offset))

        n_scenes = len(rows)
        if n_scenes > 0:
            mean_score = np.mean([r["score"] for r in rows])
            mean_coll = np.mean([r["collision_rate"] for r in rows])
            mean_coll_fault = np.mean([r["at_fault_collision_rate"] for r in rows])
            print(
                f"  {mode_name}: {n_scenes} scenes, score={mean_score:.3f}, collision_rate={mean_coll:.3f}, at_fault_collision_rate={mean_coll_fault:.3f}"
            )
        else:
            print(f"  {mode_name}: no populated scenes")
    except Exception as e:
        print(f"  {mode_name} failed (non-fatal): {e}")

    env.close()
    return rows


def evaluate_checkpoint(checkpoint_path, base_config):
    """Run all eval modes for a single checkpoint. Returns list of per-scene dicts."""

    cpt_config, _ = load_checkpoint_config(checkpoint_path, base_config)
    cpt_config["load_model_path"] = checkpoint_path

    # Create first env before loading policy (load_policy needs vecenv.driver_env)
    sp_train_config = make_eval_config(cpt_config, TRAIN_MAP_DIR, control_mode="control_vehicles", num_maps=50_000)
    env = load_env(ENV_NAME, sp_train_config)

    policy = load_policy(cpt_config, env, ENV_NAME)
    policy.eval()

    evaluator = CheckpointEvaluator(cpt_config)
    all_rows = []

    # ── 1. Self-play on training maps (reuse the env we already created) ─────
    n_rounds = num_resample_rounds()
    try:
        for round_idx in range(n_rounds):
            if round_idx > 0:
                env.driver_env.resample_maps()

            info_list = evaluator.rollout(env=env, policy=policy, deterministic=DETERMINISTIC)
            scene_offset = round_idx * env.driver_env.num_envs
            all_rows.extend(process_rollout_data(info_list, checkpoint_path, "sp_train", scene_offset))

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
            evaluator, policy, cpt_config, VAL_MAP_DIR, "control_vehicles", checkpoint_path, "sp_val", num_maps=10_000
        )
    )

    # ── 3. Human-replay on training maps ─────────────────────────────────────
    all_rows.extend(
        run_mode(
            evaluator,
            policy,
            cpt_config,
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
            evaluator, policy, cpt_config, VAL_MAP_DIR, "control_sdc_only", checkpoint_path, "hr_val", num_maps=10_000
        )
    )

    # ── 5. Human-replay on interactive scenes ────────────────────────────────
    all_rows.extend(
        run_mode(
            evaluator,
            policy,
            cpt_config,
            INTERACTIVE_MAP_DIR,
            "control_sdc_only",
            checkpoint_path,
            "hr_interactive",
            num_maps=INTERACTIVE_MAP_DIR_MAPS,
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
    scaling_entries = []

    for fname in sorted(os.listdir(SCALING_CHECKPOINTS_PATH)):
        if not fname.endswith(".pt"):
            continue
        parsed = parse_scaling_checkpoint_name(fname)
        if parsed is None:
            print(f"  Warning: could not parse scaling checkpoint name '{fname}', skipping")
            continue

        metadata_maps, is_reg, dynamics, anchor_maps = parsed
        scaling_entries.append(
            (
                os.path.join(SCALING_CHECKPOINTS_PATH, fname),
                metadata_maps,
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
    for cpt_path, metadata_maps, is_reg, dynamics, anchor_maps in scaling_entries:
        tag = "reg" if is_reg else "unreg"
        anchor_str = f"anchor={anchor_maps}" if anchor_maps is not None else "no anchor"
        print(f"  {os.path.basename(cpt_path)}  ->  sp={metadata_maps}, {tag}, {dynamics}, {anchor_str}")

    DYNAMICS_CONFIG_MAP = {
        "delta": "delta_local",
        "classic": "classic",
        "jerk": "jerk",
    }

    all_rows = []

    for cpt_path, sp_maps, is_reg, dynamics, anchor_maps in scaling_entries:
        print(f"\n{'─' * 60}")
        print(f"Scaling eval: {os.path.basename(cpt_path)}")
        print(f"{'─' * 60}")

        dyn_config_name = DYNAMICS_CONFIG_MAP.get(dynamics, dynamics)

        cpt_config, _ = load_checkpoint_config(cpt_path, base_config)
        cpt_config["load_model_path"] = cpt_path
        cpt_config["env"]["dynamics_model"] = dyn_config_name

        # load policy once using sp_train env
        sp_train_config = make_eval_config(
            cpt_config,
            TRAIN_MAP_DIR,
            control_mode="control_vehicles",
            num_maps=50_000,
        )
        sp_train_env = load_env(ENV_NAME, sp_train_config)
        policy = load_policy(cpt_config, sp_train_env, ENV_NAME)
        policy.eval()
        sp_train_env.close()

        evaluator = CheckpointEvaluator(cpt_config)

        # ── Self-play on training ────────────────────────────────────────
        sp_train_rows = run_mode(
            evaluator,
            policy,
            cpt_config,
            TRAIN_MAP_DIR,
            "control_vehicles",
            cpt_path,
            "scaling_sp_train",
            num_maps=50_000,
        )

        # ── Self-play on validation ──────────────────────────────────────
        sp_val_rows = run_mode(
            evaluator,
            policy,
            cpt_config,
            VAL_MAP_DIR,
            "control_vehicles",
            cpt_path,
            "scaling_sp_val",
            num_maps=10_000,
        )

        # ── Human-replay on randomly sampled validation scenes ───────────
        hr_val_rows = run_mode(
            evaluator,
            policy,
            cpt_config,
            VAL_MAP_DIR,
            "control_sdc_only",
            cpt_path,
            "scaling_hr_val",
            num_maps=10_000,
        )

        # ── Human-replay on interactive scenes ───────────────────────────
        hr_interactive_rows = run_mode(
            evaluator,
            policy,
            cpt_config,
            INTERACTIVE_MAP_DIR,
            "control_sdc_only",
            cpt_path,
            "scaling_hr_interactive",
            num_maps=INTERACTIVE_MAP_DIR_MAPS,
        )

        # Attach scaling metadata to every row
        for row in sp_train_rows + sp_val_rows + hr_val_rows + hr_interactive_rows:
            row["sp_maps"] = sp_maps
            row["is_regularized"] = is_reg
            row["dynamics"] = dynamics
            row["anchor_maps"] = anchor_maps

        all_rows.extend(sp_train_rows)
        all_rows.extend(sp_val_rows)
        all_rows.extend(hr_val_rows)
        all_rows.extend(hr_interactive_rows)

    return all_rows


def make_render_config(cpt_config, map_dir, num_maps=1000):
    """Build a config for human-replay rendering with headless ffmpeg output."""
    return make_eval_config(cpt_config, map_dir, control_mode="control_sdc_only", num_maps=num_maps)


def select_render_envs(evaluator, policy, env, num_to_render):
    """Run a non-rendering rollout and return env indices with the worst collision rates.

    Returns:
        List of (env_idx, collision_rate) tuples, sorted by collision rate descending,
        truncated to num_to_render.
    """
    info_list = evaluator.rollout(env=env, policy=policy, deterministic=DETERMINISTIC)
    populated = [log for log in info_list if log and log.get("n", 0) > 0]
    did_collide = np.array([log["collision_rate"] for log in populated])

    print(f"average collision rate: {did_collide.mean()}, n = {did_collide.shape[0]}")

    scored = []
    for env_idx, log in enumerate(info_list):
        if not log or log.get("n", 0) <= 0:
            continue
        scored.append((env_idx, log.get("collision_rate", 0.0)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:num_to_render]


def render_checkpoint_videos(base_config):
    """Render human-replay videos for each checkpoint in CHECKPOINTS_TO_RENDER.

    Supports two modes (RENDER_MODE):
      - "random": render the first NUM_ENVS_TO_RENDER env indices
      - "worst_collision": run a stats-only rollout first, then render
        the NUM_ENVS_TO_RENDER envs with the highest collision rates
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

        cpt_config, _ = load_checkpoint_config(cpt_path, base_config)
        cpt_config["load_model_path"] = cpt_path

        # Create env once (or reuse across checkpoints — same map config)
        if env is None:
            config = make_render_config(cpt_config, RENDER_MAP_DIR, num_maps=RENDER_NUM_MAPS)
            env = load_env(ENV_NAME, config)

        policy = load_policy(cpt_config, env, ENV_NAME)
        policy.eval()

        evaluator = CheckpointEvaluator(cpt_config)

        # Select which envs to render
        collision_rates = {}  # scenario_id -> collision_rate for filename tagging
        if RENDER_MODE == "worst_collision":
            print(f"  Running stats rollout to find worst collisions...")
            selected = select_render_envs(evaluator, policy, env, NUM_ENVS_TO_RENDER)
            env_indices = [idx for idx, _ in selected]
            collision_rates = {idx: rate for idx, rate in selected}
            for idx, coll_rate in selected:
                print(f"    env {idx}: collision_rate={coll_rate:.3f}")
        else:
            env_indices = list(range(NUM_ENVS_TO_RENDER))

        # Build env_idx -> scenario_id mapping for filename tagging
        scenario_ids = env.driver_env.scenario_ids
        idx_to_scenario = {idx: scenario_ids[idx].rstrip("\x00") for idx in env_indices}

        # Run a stats rollout for "random" mode to get collision rates
        if RENDER_MODE == "random" and not collision_rates:
            info_list = evaluator.rollout(env=env, policy=policy, deterministic=DETERMINISTIC)
            for idx in env_indices:
                if idx < len(info_list) and info_list[idx]:
                    collision_rates[idx] = info_list[idx].get("collision_rate", 0.0)

        # Map scenario_id -> collision_rate
        scenario_collision = {}
        for idx in env_indices:
            sid = idx_to_scenario.get(idx, "")
            scenario_collision[sid] = collision_rates.get(idx, 0.0)

        # Render selected envs
        for i, env_idx in enumerate(env_indices):
            print(f"  Rendering env {env_idx} ({i + 1}/{len(env_indices)})...")
            evaluator.rollout(env=env, policy=policy, deterministic=DETERMINISTIC)
            env.driver_env.stop_recorder(env_idx)

        # Move mp4s into the checkpoint subdirectory, tagging with collision rate
        for mp4_path in glob.glob("*.mp4"):
            scenario_id = os.path.splitext(os.path.basename(mp4_path))[0]
            coll_rate = scenario_collision.get(scenario_id, 0.0)
            dest = os.path.join(cpt_video_dir, f"coll{coll_rate:.2f}_{scenario_id}.mp4")
            shutil.move(mp4_path, dest)
            print(f"  Saved: {dest}")

    if env is not None:
        env.close()

    print(f"\nAll videos saved to {RENDER_OUTPUT_DIR}/")


def main():
    base_config = load_config(ENV_NAME)

    all_rows = []
    # ── Scaling analysis ─────────────────────────────────────────────────
    all_rows.extend(evaluate_scaling_checkpoints(base_config))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV} ({len(df)} rows)")

    if not df.empty:
        summary = df.groupby(["checkpoint", "mode"]).agg(
            scenes=("score", "count"),
            score=("score", "mean"),
            collision_rate=("collision_rate", "mean"),
            at_fault_collision_rate=("at_fault_collision_rate", "mean"),
            rear_collision_rate=("rear_collision_rate", "mean"),
            offroad_rate=("offroad_rate", "mean"),
        )
        print(f"\n{summary}")

    # ── Figures ──────────────────────────────────────────────────────────────
    if MAKE_FIGURES:
        from pufferlib.ocean.benchmark.plot_and_format import make_all_figures

        make_all_figures(df if not df.empty else None)

    # ── Video rendering (last: env.close() may segfault in raylib cleanup) ──
    if RUN_RENDER and CHECKPOINTS_TO_RENDER:
        render_checkpoint_videos(base_config)

    return df


if __name__ == "__main__":
    main()
