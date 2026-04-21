"""Run WOSAC realism evaluation for all scaling checkpoints.

Mirrors the structure of evaluate_checkpoints.py but runs only the WOSAC
evaluator (WOSACEvaluator) instead of the standard rollout modes.

Checkpoint naming convention (identical to evaluate_checkpoints.py):
  [reg|unreg]_[dynamics]_[N]_maps[_anchor_[M]_maps].pt

Output CSV columns:
  checkpoint, sp_maps, is_regularized, dynamics, anchor_maps,
  realism_meta_score, kinematic_metrics, interactive_metrics,
  map_based_metrics, num_agents_per_scene, total_unique_scenarios

Each row is one scenario evaluated by the WOSAC evaluator (before averaging),
so that plot_scaling_wosac can compute mean ± SEM across scenarios.

Usage:
    python evaluate_wosac.py
"""

import copy
import os
import re

import numpy as np
import pandas as pd
import torch

from pufferlib.pufferl import load_env, load_policy, load_config

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
SCALING_CHECKPOINTS_PATH = "models/scaling_cpts"
ENV_NAME = "puffer_drive"
OUTPUT_CSV = "results/checkpoint_wosac_results.csv"

# WOSAC eval settings (mirrors [eval] section defaults in the .ini)
WOSAC_MAP_DIR = "resources/drive/binaries/validation"
WOSAC_BATCH_SIZE = 64  # num_scenes_per_batch
WOSAC_SCENARIO_POOL_SIZE = 10000
WOSAC_MAX_BATCHES = 2
WOSAC_NUM_ROLLOUTS = 32
WOSAC_INIT_STEPS = 10
WOSAC_CONTROL_MODE = "control_wosac"
WOSAC_INIT_MODE = "create_all_valid"
WOSAC_GOAL_BEHAVIOR = 2
WOSAC_GOAL_RADIUS = 2.5

# ────────────────────────────────────────────────────────────────────────────────

WOSAC_METRICS = [
    # Group scores
    "realism_meta_score",
    "kinematic_metrics",
    "interactive_metrics",
    "map_based_metrics",
    "num_agents_per_scene",
    # Kinematic sub-metrics
    "likelihood_linear_speed",
    "likelihood_linear_acceleration",
    "likelihood_angular_speed",
    "likelihood_angular_acceleration",
    # Interactive sub-metrics
    "likelihood_collision_indication",
    "likelihood_distance_to_nearest_object",
    "likelihood_time_to_collision",
    # Map sub-metrics
    "likelihood_distance_to_road_edge",
    "likelihood_offroad_indication",
]

DYNAMICS_CONFIG_MAP = {
    "delta": "delta_local",
    "classic": "classic",
    "jerk": "jerk",
}


# ─── Helpers shared with evaluate_checkpoints.py ────────────────────────────────


def _parse_num(s):
    """Parse '10', '1k', '50k' -> int."""
    m = re.match(r"(\d+)(k?)", s)
    if not m:
        return None
    n = int(m.group(1))
    if m.group(2) == "k":
        n *= 1000
    return n


def parse_scaling_checkpoint_name(filename):
    """Parse a scaling checkpoint filename into (sp_maps, is_reg, dynamics, anchor_maps).

    Returns None if the filename doesn't match the expected pattern.
    """
    stem = filename.replace(".pt", "")

    # reg/unreg with anchor
    match = re.match(r"^(reg|unreg)_(\w+?)_(\d+k?)_maps_anchor_(\d+k?)_maps$", stem)
    if match:
        return (
            _parse_num(match.group(3)),
            match.group(1) == "reg",
            match.group(2),
            _parse_num(match.group(4)),
        )

    # reg/unreg without anchor
    match = re.match(r"^(reg|unreg)_(\w+?)_(\d+k?)_maps$", stem)
    if match:
        return (
            _parse_num(match.group(3)),
            match.group(1) == "reg",
            match.group(2),
            None,
        )

    return None


def load_checkpoint_config(checkpoint_path, fallback_config):
    """Load full_args from checkpoint if available, else use fallback ini config."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "full_args" in checkpoint:
        print(f"  Using env config stored in checkpoint.")
        return copy.deepcopy(checkpoint["full_args"]), True
    print(f"  No config found in checkpoint, falling back to ini config.")
    return copy.deepcopy(fallback_config), False


# ─── WOSAC config builder ────────────────────────────────────────────────────────


def make_wosac_config(cpt_config):
    """Return a config dict wired up for WOSAC evaluation."""
    config = copy.deepcopy(cpt_config)

    config["env"]["map_dir"] = WOSAC_MAP_DIR
    config["env"]["num_maps"] = WOSAC_SCENARIO_POOL_SIZE
    config["env"]["num_agents"] = WOSAC_BATCH_SIZE * 10
    config["env"]["termination_mode"] = 0
    config["env"]["async_resets"] = False
    config["env"]["init_mode"] = WOSAC_INIT_MODE
    config["env"]["control_mode"] = WOSAC_CONTROL_MODE
    config["env"]["init_steps"] = WOSAC_INIT_STEPS
    config["env"]["goal_behavior"] = WOSAC_GOAL_BEHAVIOR
    config["env"]["goal_radius"] = WOSAC_GOAL_RADIUS
    config["env"]["episode_length"] = 91

    config["vec"] = dict(backend="PufferEnv", num_envs=1)

    # Mirror the [eval] keys expected by WOSACEvaluator
    config.setdefault("eval", {})
    config["eval"]["wosac_realism_eval"] = True
    config["eval"]["wosac_batch_size"] = WOSAC_BATCH_SIZE
    config["eval"]["wosac_scenario_pool_size"] = WOSAC_SCENARIO_POOL_SIZE
    config["eval"]["wosac_max_batches"] = WOSAC_MAX_BATCHES
    config["eval"]["wosac_num_rollouts"] = WOSAC_NUM_ROLLOUTS
    config["eval"]["wosac_init_steps"] = WOSAC_INIT_STEPS
    config["eval"]["wosac_control_mode"] = WOSAC_CONTROL_MODE
    config["eval"]["wosac_init_mode"] = WOSAC_INIT_MODE
    config["eval"]["wosac_goal_behavior"] = WOSAC_GOAL_BEHAVIOR
    config["eval"]["wosac_goal_radius"] = WOSAC_GOAL_RADIUS
    config["eval"]["wosac_aggregate_results"] = False  # keep per-scenario rows
    config["eval"]["wosac_filter_out_post_done"] = False
    config["eval"]["wosac_sanity_check"] = False
    config["eval"]["wosac_eval_mode"] = "policy"
    config["eval"]["backend"] = "PufferEnv"
    config["eval"]["map_dir"] = WOSAC_MAP_DIR

    return config


# ─── Per-checkpoint evaluation ───────────────────────────────────────────────────


def evaluate_wosac_checkpoint(cpt_path, sp_maps, is_reg, dynamics, anchor_maps, base_config):
    """Run WOSAC evaluation for a single checkpoint.

    Returns a list of per-scenario dicts (one per row from WOSACEvaluator).
    """
    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

    cpt_config, _ = load_checkpoint_config(cpt_path, base_config)
    cpt_config["load_model_path"] = cpt_path
    dyn_config_name = DYNAMICS_CONFIG_MAP.get(dynamics, dynamics)
    cpt_config["env"]["dynamics_model"] = dyn_config_name

    wosac_config = make_wosac_config(cpt_config)

    vecenv = load_env(ENV_NAME, wosac_config)
    policy = load_policy(wosac_config, vecenv, ENV_NAME)
    policy.eval()

    evaluator = WOSACEvaluator(wosac_config)

    try:
        df_results = evaluator.evaluate(wosac_config, vecenv, policy)
    except Exception as e:
        print(f"  WOSAC eval failed (non-fatal): {e}")
        vecenv.close()
        return []

    vecenv.close()

    rows = []
    for scenario_id, row in df_results.iterrows():
        entry = {
            "checkpoint": cpt_path,
            "scenario_id": scenario_id,
            "sp_maps": sp_maps,
            "is_regularized": is_reg,
            "dynamics": dynamics,
            "anchor_maps": anchor_maps if anchor_maps is not None else 0,
        }
        for metric in WOSAC_METRICS:
            entry[metric] = float(row.get(metric, float("nan")))
        rows.append(entry)

    if rows:
        mean_realism = float(df_results["realism_meta_score"].mean())
        print(f"  WOSAC: {len(rows)} scenarios, realism_meta_score={mean_realism:.4f}")
    else:
        print("  WOSAC: no scenario results returned.")

    return rows


# ─── Main ────────────────────────────────────────────────────────────────────────


def main():
    base_config = load_config(ENV_NAME)

    # Discover and parse scaling checkpoints
    scaling_entries = []
    for fname in sorted(os.listdir(SCALING_CHECKPOINTS_PATH)):
        if not fname.endswith(".pt"):
            continue
        parsed = parse_scaling_checkpoint_name(fname)
        if parsed is None:
            print(f"  Warning: could not parse '{fname}', skipping.")
            continue
        sp_maps, is_reg, dynamics, anchor_maps = parsed
        scaling_entries.append((os.path.join(SCALING_CHECKPOINTS_PATH, fname), sp_maps, is_reg, dynamics, anchor_maps))

    scaling_entries.sort(key=lambda x: (x[1], x[2], x[4] or 0))

    if not scaling_entries:
        print("No scaling checkpoints found — nothing to evaluate.")
        return

    print(f"\nFound {len(scaling_entries)} scaling checkpoints:")
    for cpt_path, sp_maps, is_reg, dynamics, anchor_maps in scaling_entries:
        tag = "reg" if is_reg else "unreg"
        anchor_str = f"anchor={anchor_maps}" if anchor_maps is not None else "no anchor"
        print(f"  {os.path.basename(cpt_path)}  ->  sp={sp_maps}, {tag}, {dynamics}, {anchor_str}")

    all_rows = []
    for cpt_path, sp_maps, is_reg, dynamics, anchor_maps in scaling_entries:
        print(f"\n{'─' * 60}")
        print(f"WOSAC eval: {os.path.basename(cpt_path)}")
        print(f"{'─' * 60}")
        rows = evaluate_wosac_checkpoint(cpt_path, sp_maps, is_reg, dynamics, anchor_maps, base_config)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo results collected — CSV not written.")
        return

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV} ({len(df)} rows, {df['checkpoint'].nunique()} checkpoints)")

    # Summary table
    summary = (
        df.groupby(["checkpoint", "sp_maps", "anchor_maps"])["realism_meta_score"]
        .agg(["mean", "sem", "count"])
        .rename(columns={"mean": "realism_mean", "sem": "realism_sem", "count": "n_scenarios"})
    )
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
