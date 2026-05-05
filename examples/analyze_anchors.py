"""
Evaluate BC anchor checkpoints from models/anchors/.

For each checkpoint this script collects:
  - Open-loop metrics stored inside the .pt file (train/val loss & accuracy).
  - Closed-loop self-play metrics: rollout the BC policy on the validation
    dataset with all vehicles controlled (control_vehicles).
  - Closed-loop human-replay metrics: rollout the BC policy as SDC only,
    with all other agents following expert trajectories (control_sdc_only).

Results are written to a pandas DataFrame and saved as a CSV, with one row
per scene per mode (matching evaluate_checkpoints.py conventions).

Checkpoint naming convention:
  bc_<dynamics>_<N>maps_<run_id>.pt
  e.g.  bc_delta_local_100maps_k0p5m0v8.pt

Usage:
    python eval_bc_anchors.py
    python eval_bc_anchors.py --anchor-dir models/anchors --out results/anchor_eval.csv
    python eval_bc_anchors.py --val-maps 1000
"""

import argparse
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from examples.train_bc_policy import BCPolicy, load_bc_policy, get_output_sizes
from pufferlib.pufferl import load_config, load_env
from pufferlib.ocean.drive import binding

# ─── CONFIG ─────────────────────────────────────────────────────────────────
ANCHOR_DIR = "models/anchors"
VAL_MAP_DIR = "resources/drive/binaries/interactive_data_validation"
VAL_NUM_MAPS = 50
OUTPUT_CSV = "results/anchor_eval.csv"
DETERMINISTIC = True

METRICS = [
    "n",
    "score",
    "collision_rate",
    "at_fault_collision_rate",
    "offroad_rate",
    "route_progress",
    "episode_length",
    "episode_return",
]
# ────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def parse_num_maps(stem: str) -> int:
    """Extract number of training maps from checkpoint stem.

    bc_delta_local_100maps_k0p5m0v8  ->  100
    """
    match = re.search(r"_(\d+)maps", stem)
    return int(match.group(1)) if match else -1


def parse_dynamics(stem: str) -> str:
    """Extract dynamics model name from checkpoint stem."""
    if "delta_local" in stem:
        return "delta_local"
    if "classic" in stem:
        return "classic"
    return "unknown"


# ---------------------------------------------------------------------------
# Environment construction  (mirrors make_eval_config in evaluate_checkpoints)
# ---------------------------------------------------------------------------


def make_bc_eval_env(dynamics_model: str, control_mode: str, num_maps: int):
    """Build a single-vecenv for BC anchor evaluation.

    Settings mirror make_eval_config() in evaluate_checkpoints.py:
    async_resets=False, termination_mode=1, fix_lambdas/fix_rewards=True.
    """
    args = load_config("puffer_drive")
    args["vec"] = dict(backend="PufferEnv", num_envs=1)
    args["env"]["num_agents"] = 2048
    args["env"]["map_dir"] = VAL_MAP_DIR
    args["env"]["num_maps"] = num_maps
    args["env"]["dynamics_model"] = dynamics_model
    args["env"]["control_mode"] = control_mode
    args["env"]["reg_mode"] = "None"
    args["env"]["fix_lambdas"] = True
    args["env"]["fix_rewards"] = True
    args["env"]["lambda_value"] = 0.0
    args["env"]["async_resets"] = False
    args["env"]["termination_mode"] = 1
    args["env"]["goal_behavior"] = 0
    args["env"]["obs_partner_noise_speed"] = 0.0
    args["env"]["obs_partner_noise_pos"] = 0.0
    args["base"]["rnn_name"] = "none"
    return load_env("puffer_drive", args)


# ---------------------------------------------------------------------------
# Per-scene result extraction  (mirrors process_rollout_data)
# ---------------------------------------------------------------------------


def process_rollout_data(info_list: list, checkpoint: str, mode: str) -> list[dict]:
    """Return one dict per populated env log, with checkpoint/mode metadata.

    Mirrors process_rollout_data() in evaluate_checkpoints.py — filters to
    logs where n > 0 and extracts the standard METRICS keys.
    """
    populated = [log for log in info_list if log and log.get("n", 0) > 0]
    rows = []
    for i, log in enumerate(populated):
        row = {
            "checkpoint": checkpoint,
            "mode": mode,
            "scene_idx": i,
        }
        for key in METRICS:
            row[key] = float(log.get(key, 0.0))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# BC rollout  (BCPolicy has no forward_eval / lstm state)
# ---------------------------------------------------------------------------


def rollout_bc_policy(
    policy: BCPolicy,
    env,
    device: torch.device,
    deterministic: bool = True,
) -> list[dict]:
    """Run one full episode with a BCPolicy and return per-env info logs.

    BCPolicy does not implement the PufferLib policy interface (no
    forward_eval, no lstm state), so this replaces CheckpointEvaluator.rollout
    for BC anchors specifically.

    Returns:
        Raw info_list from env.step(per_env_logs=True), one entry per env.
    """
    obs, _ = env.reset()
    episode_length = env.driver_env.episode_length or 91

    info_list = []
    for _ in range(episode_length):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions = policy(obs_tensor, deterministic=deterministic)

        action_np = actions.cpu().numpy()
        # Ensure shape is (num_agents, action_dim)
        if action_np.ndim == 1:
            action_np = action_np[:, None]

        obs, _rewards, _terminals, truncated, info_list = env.step(action_np, per_env_logs=True)

        if truncated.all():
            break

    return info_list


# ---------------------------------------------------------------------------
# Single-mode runner  (mirrors run_mode in evaluate_checkpoints)
# ---------------------------------------------------------------------------


def run_mode(
    policy: BCPolicy,
    dynamics: str,
    control_mode: str,
    mode_name: str,
    num_maps: int,
    checkpoint_stem: str,
    device: torch.device,
) -> list[dict]:
    """Build env, rollout, collect per-scene rows, close env."""
    env = make_bc_eval_env(dynamics, control_mode=control_mode, num_maps=num_maps)
    rows = []
    try:
        info_list = rollout_bc_policy(policy, env, device=device, deterministic=DETERMINISTIC)
        rows = process_rollout_data(info_list, checkpoint_stem, mode_name)

        if rows:
            mean_score = np.mean([r["score"] for r in rows])
            mean_coll = np.mean([r["collision_rate"] for r in rows])
            print(f"   {mode_name}: {len(rows)} scenes  score={mean_score:.3f}  collision_rate={mean_coll:.3f}")
        else:
            print(f"   {mode_name}: no populated scenes")
    except Exception as e:
        print(f"   {mode_name} failed (non-fatal): {e}")
    finally:
        env.close()

    return rows


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate_anchors(anchor_dir: str, out_path: str, val_maps: int = VAL_NUM_MAPS):
    anchor_dir = Path(anchor_dir)
    checkpoints = sorted(anchor_dir.glob("*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No .pt files found in {anchor_dir}")

    print(f"Found {len(checkpoints)} checkpoint(s) in {anchor_dir}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute obs_dim from binding constants (same formula as Drive.__init__)
    obs_dim = (
        binding.EGO_FEATURES_DELTA_LOCAL
        + (binding.MAX_AGENTS - 1) * binding.PARTNER_FEATURES
        + binding.MAX_ROAD_SEGMENT_OBSERVATIONS * binding.ROAD_FEATURES
    )

    all_rows = []

    for cpt_path in checkpoints:
        stem = cpt_path.stem
        dynamics = parse_dynamics(stem)
        num_maps_trained = parse_num_maps(stem)
        output_sizes = get_output_sizes(dynamics)

        print(f"{'=' * 60}")
        print(f"Checkpoint : {stem}")
        print(f"dynamics   : {dynamics}   trained_on : {num_maps_trained} maps")
        print(f"{'=' * 60}")

        # ------------------------------------------------------------------
        # Load policy + open-loop metrics from checkpoint
        # ------------------------------------------------------------------
        policy, ckpt_metrics = load_bc_policy(
            checkpoint_path=str(cpt_path),
            obs_dim=obs_dim,
            input_size=128,
            max_partner_objects=binding.MAX_AGENTS - 1,
            partner_features=binding.PARTNER_FEATURES,
            max_road_objects=binding.MAX_ROAD_SEGMENT_OBSERVATIONS,
            road_features=binding.ROAD_FEATURES,
            ego_dim=binding.EGO_FEATURES_DELTA_LOCAL,
            hidden_size=512,
            output_sizes=output_sizes,
            device=str(device),
        )

        ol_meta = {
            "checkpoint": stem,
            "dynamics": dynamics,
            "num_maps_trained": num_maps_trained,
            "ol_train_loss": ckpt_metrics.get("train_loss", float("nan")),
            "ol_train_accuracy": ckpt_metrics.get("train_accuracy", float("nan")),
            "ol_val_loss": ckpt_metrics.get("val_loss", float("nan")),
            "ol_val_accuracy": ckpt_metrics.get("val_accuracy", float("nan")),
        }

        # ------------------------------------------------------------------
        # Closed-loop: self-play (all vehicles controlled by BC policy)
        # ------------------------------------------------------------------
        print("   Running self-play rollout …")
        sp_rows = run_mode(
            policy,
            dynamics,
            control_mode="control_vehicles",
            mode_name="cl_selfplay",
            num_maps=val_maps,
            checkpoint_stem=stem,
            device=device,
        )
        for row in sp_rows:
            row.update(ol_meta)

        # ------------------------------------------------------------------
        # Closed-loop: human-replay (BC policy controls SDC only)
        # ------------------------------------------------------------------
        print("   Running human-replay rollout …")
        hr_rows = run_mode(
            policy,
            dynamics,
            control_mode="control_sdc_only",
            mode_name="cl_humanreplay",
            num_maps=val_maps,
            checkpoint_stem=stem,
            device=device,
        )
        for row in hr_rows:
            row.update(ol_meta)

        all_rows.extend(sp_rows)
        all_rows.extend(hr_rows)
        print()

    df = pd.DataFrame(all_rows)

    # Column order: identity → open-loop → closed-loop metrics
    id_cols = ["checkpoint", "dynamics", "num_maps_trained", "mode", "scene_idx"]
    ol_cols = sorted(c for c in df.columns if c.startswith("ol_"))
    cl_cols = [m for m in METRICS if m in df.columns]
    df = df[id_cols + ol_cols + cl_cols]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

    if not df.empty:
        summary = df.groupby(["checkpoint", "mode"]).agg(
            scenes=("score", "count"),
            score=("score", "mean"),
            collision_rate=("collision_rate", "mean"),
            offroad_rate=("offroad_rate", "mean"),
            route_progress=("route_progress", "mean"),
        )
        print(f"\n{summary}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC anchor checkpoints")
    parser.add_argument("--anchor-dir", type=str, default=ANCHOR_DIR)
    parser.add_argument("--out", type=str, default=OUTPUT_CSV)
    parser.add_argument(
        "--val-maps", type=int, default=VAL_NUM_MAPS, help="Number of validation maps for closed-loop rollouts"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_anchors(
        anchor_dir=args.anchor_dir,
        out_path=args.out,
        val_maps=args.val_maps,
    )
