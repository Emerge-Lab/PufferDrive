"""Evaluate noise-ablation RL checkpoints.

Evaluates every checkpoint in CHECKPOINTS_PATH matching
    [reg|unreg]_[dynamics]_dynamics_noise_[VALUE]_sensor_noise_[VALUE].pt
in three modes:
  1. Self-play on validation maps
  2. Human-replay on validation maps
  3. Human-replay on interactive scenes

Then plots collision-rate metrics (in %) as a function of the two training
noise levels (dynamics noise, sensor noise) using a grid of heatmaps:
  rows    = regularized / unregularized (whichever are present)
  columns = the same four metric/mode combos as the gamma-scan script
            - self-play (val)              collision rate
            - human-replay (val)           collision rate
            - human-replay (interactive)   collision rate
            - human-replay (val)           at-fault collision rate

Eval convention:
    Observation noise is forced to 0.0 at eval time, just as in the gamma-scan
    script. The noise values in the filename describe *training* conditions,
    not eval conditions, so we measure how training noise affects clean-env
    performance. If you want to also evaluate under sensor noise, set
    obs_partner_noise_{pos,speed} from sensor_noise inside make_eval_config.

Unit convention:
    All rate metrics are stored internally and in the CSV as ratios in [0, 1].
    They are converted to percentages only at display time (run logs, the
    summary printout, and the plot).

Examples of accepted filenames:
    reg_delta_dynamics_noise_0.0_sensor_noise_0.0.pt
    unreg_delta_dynamics_noise_0.05_sensor_noise_0.1.pt

Usage:
    python evaluate_noise_checkpoints.py
"""

import copy
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pufferlib.pufferl import load_env, load_policy, load_config
from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator

# ─── USER CONFIG ────────────────────────────────────────────────────────────────
CHECKPOINTS_PATH = "models/noise_ablation_cpts"
DETERMINISTIC = True

VAL_MAP_DIR = "resources/drive/binaries/validation"  # 10k maps
INTERACTIVE_MAP_DIR = "resources/drive/binaries/interactive_data_validation"  # 200 maps
NUM_TOTAL_EVAL_AGENTS = 1024 * 5
NUM_AGENTS_PER_VECENV = 1024
ENV_NAME = "puffer_drive"
DATASET = "womd"
OUTPUT_CSV = "results/noise_ablation_eval_results.csv"
PLOT_PATH = "results/noise_vs_collision_rate.png"

DYNAMICS_CONFIG_MAP = {
    "delta": "delta_local",
    "classic": "classic",
    "jerk": "jerk",
}

# Rate columns that should be displayed as percentages (data stays as ratios).
RATE_COLS_FOR_DISPLAY = [
    "collision_rate",
    "at_fault_collision_rate",
    "rear_collision_rate",
    "offroad_rate",
    "completion_rate",
]

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
]


def load_checkpoint_config(checkpoint_path, fallback_config):
    """Load full_args from checkpoint if available, else use fallback ini config."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "full_args" in checkpoint:
        print(f"  Using env config stored in checkpoint.")
        return copy.deepcopy(checkpoint["full_args"]), True
    print(f"  No config found in checkpoint, falling back to ini config.")
    return copy.deepcopy(fallback_config), False


def make_eval_config(cpt_config, map_dir, control_mode, num_maps, episode_length=150):
    """Build an eval-ready config from the checkpoint config.

    Note: obs_partner_noise_{pos,speed} are forced to 0.0 here so we always
    evaluate on clean observations regardless of what the policy was trained
    with. Change this if you want noisy eval.
    """
    config = copy.deepcopy(cpt_config)
    config["env"]["map_dir"] = map_dir
    config["env"]["control_mode"] = control_mode
    config["env"]["num_maps"] = num_maps
    config["env"]["num_agents"] = NUM_AGENTS_PER_VECENV
    config["env"]["async_resets"] = False
    config["env"]["goal_behavior"] = 0
    config["env"]["render_mode"] = 1
    config["env"]["termination_mode"] = 1
    config["env"]["fix_lambdas"] = True
    config["env"]["fix_rewards"] = True
    config["env"]["obs_partner_noise_speed"] = 0.0
    config["env"]["obs_partner_noise_pos"] = 0.0
    if episode_length is not None:
        config["env"]["episode_length"] = episode_length
    config["vec"] = dict(backend="PufferEnv", num_envs=1)

    print(config["env"]["lambda_value"])
    return config


def process_rollout_data(info_list, checkpoint, mode, scene_offset=0, dataset=DATASET):
    """Return one dict per rollout (populated env log) with checkpoint/mode metadata.

    All metrics are stored exactly as the env reports them (ratios in [0, 1]
    for rate-style metrics). Conversion to percentages happens later, at the
    display layer.
    """
    populated = [log for log in info_list if log and log.get("n", 0) > 0]
    if not populated:
        return []
    rows = []
    for i, log in enumerate(populated):
        row = {
            "checkpoint": checkpoint,
            "dataset": dataset,
            "mode": mode,
            "scene_idx": scene_offset + i,
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

            rollout_stats = evaluator.rollout(policy, env, deterministic=DETERMINISTIC)
            scene_offset = round_idx * env.driver_env.num_envs
            rows.extend(process_rollout_data(rollout_stats, checkpoint, mode_name, scene_offset))

        n_scenes = len(rows)
        if n_scenes > 0:
            mean_score = np.mean([r["score"] for r in rows])
            # rates are ratios in the data — multiply by 100 only for display
            mean_coll_pct = np.mean([r["collision_rate"] for r in rows]) * 100.0
            mean_coll_fault_pct = np.mean([r["at_fault_collision_rate"] for r in rows]) * 100.0
            print(
                f"  {mode_name}: {n_scenes} scenes, score={mean_score:.3f}, "
                f"collision_rate={mean_coll_pct:.2f}%, at_fault_collision_rate={mean_coll_fault_pct:.2f}%"
            )
        else:
            print(f"  {mode_name}: no populated scenes")
    except Exception as e:
        print(f"  {mode_name} failed (non-fatal): {e}")

    env.close()
    return rows


def parse_noise_checkpoint_name(filename):
    """Parse a noise-ablation checkpoint filename.

    Format: [reg|unreg]_[dynamics]_dynamics_noise_[VALUE]_sensor_noise_[VALUE].pt

    Examples:
        reg_delta_dynamics_noise_0.0_sensor_noise_0.0.pt
            -> (True,  "delta", 0.0,  0.0)
        unreg_delta_dynamics_noise_0.05_sensor_noise_0.1.pt
            -> (False, "delta", 0.05, 0.1)

    Returns:
        (is_regularized: bool, dynamics: str,
         dynamics_noise: float, sensor_noise: float)
        or None if the filename does not match.
    """
    stem = filename.replace(".pt", "")
    # `\w+?` is non-greedy so multi-token dynamics names (e.g. delta_local)
    # also work — the engine backtracks until `_dynamics_noise_` matches.
    match = re.match(
        r"^(reg|unreg)_(\w+?)_dynamics_noise_(\d+\.\d+)_sensor_noise_(\d+\.\d+)$",
        stem,
    )
    if not match:
        return None
    is_reg = match.group(1) == "reg"
    dynamics = match.group(2)
    dynamics_noise = float(match.group(3))
    sensor_noise = float(match.group(4))
    return is_reg, dynamics, dynamics_noise, sensor_noise


def evaluate_noise_checkpoints(base_config):
    """Run sp_val, hr_val, and hr_interactive for each noise-ablation checkpoint."""
    entries = []
    for fname in sorted(os.listdir(CHECKPOINTS_PATH)):
        if not fname.endswith(".pt"):
            continue
        parsed = parse_noise_checkpoint_name(fname)
        if parsed is None:
            print(f"  Warning: could not parse '{fname}', skipping")
            continue
        is_reg, dynamics, dyn_noise, sensor_noise = parsed
        entries.append(
            (
                os.path.join(CHECKPOINTS_PATH, fname),
                is_reg,
                dynamics,
                dyn_noise,
                sensor_noise,
            )
        )

    # Sort by (reg flag, dynamics_noise, sensor_noise) so the log is easy to read
    entries.sort(key=lambda x: (x[1], x[3], x[4]))

    if not entries:
        print(f"No checkpoints found in {CHECKPOINTS_PATH}")
        return []

    print(f"\nFound {len(entries)} checkpoints:")
    for cpt_path, is_reg, dynamics, dyn_noise, sensor_noise in entries:
        tag = "reg" if is_reg else "unreg"
        print(
            f"  {os.path.basename(cpt_path)}  ->  {tag}, {dynamics}, dyn_noise={dyn_noise}, sensor_noise={sensor_noise}"
        )

    all_rows = []

    for cpt_path, is_reg, dynamics, dyn_noise, sensor_noise in entries:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {os.path.basename(cpt_path)}")
        print(f"{'=' * 60}")

        dyn_config_name = DYNAMICS_CONFIG_MAP.get(dynamics, dynamics)

        cpt_config, _ = load_checkpoint_config(cpt_path, base_config)
        cpt_config["load_model_path"] = cpt_path
        cpt_config["env"]["dynamics_model"] = dyn_config_name

        # Build a starter env to load the policy against, then close it
        sp_val_config = make_eval_config(
            cpt_config,
            VAL_MAP_DIR,
            control_mode="control_vehicles",
            num_maps=10_000,
        )
        sp_val_env = load_env(ENV_NAME, sp_val_config)
        policy = load_policy(cpt_config, sp_val_env, ENV_NAME)
        policy.eval()
        sp_val_env.close()

        evaluator = CheckpointEvaluator(cpt_config)

        # 1. Self-play on validation
        sp_val_rows = run_mode(
            evaluator,
            policy,
            cpt_config,
            VAL_MAP_DIR,
            "control_vehicles",
            cpt_path,
            "sp_val",
            num_maps=10_000,
        )

        # 2. Human-replay on validation
        hr_val_rows = run_mode(
            evaluator,
            policy,
            cpt_config,
            VAL_MAP_DIR,
            "control_sdc_only",
            cpt_path,
            "hr_val",
            num_maps=10_000,
        )

        # 3. Human-replay on interactive scenes
        hr_interactive_rows = run_mode(
            evaluator,
            policy,
            cpt_config,
            INTERACTIVE_MAP_DIR,
            "control_sdc_only",
            cpt_path,
            "hr_interactive",
            num_maps=200,
        )

        # Tag every row with the noise-ablation metadata
        for row in sp_val_rows + hr_val_rows + hr_interactive_rows:
            row["is_regularized"] = is_reg
            row["dynamics"] = dynamics
            row["dynamics_noise"] = dyn_noise
            row["sensor_noise"] = sensor_noise

        all_rows.extend(sp_val_rows)
        all_rows.extend(hr_val_rows)
        all_rows.extend(hr_interactive_rows)

    return all_rows


def _annotate_heatmap(ax, values_pct, vmin, vmax):
    """Write each cell's value (in %) on top of the heatmap."""
    span = vmax - vmin if vmax > vmin else 1.0
    for i in range(values_pct.shape[0]):
        for j in range(values_pct.shape[1]):
            val = values_pct[i, j]
            if np.isnan(val):
                continue
            # white text on dark cells, black on light cells (viridis)
            norm = (val - vmin) / span
            color = "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)


def plot_noise_vs_collision(df, output_path=PLOT_PATH):
    """Plot collision-rate metrics (in %) vs (dynamics_noise, sensor_noise).

    Layout: rows = {regularized, unregularized} (whichever exist),
            cols = the same four metric/mode panels as the gamma-scan script.

    Each cell is a heatmap with sensor_noise on the x-axis and dynamics_noise
    on the y-axis. Values shown are percentages (ratios * 100).
    """
    if df.empty:
        print("No data to plot.")
        return

    panels = [
        ("sp_val", "collision_rate", "Self-play (val)\ncollision rate"),
        ("hr_val", "collision_rate", "Human-replay (val)\ncollision rate"),
        ("hr_val", "at_fault_collision_rate", "Human-replay (val)\nat-fault collision rate"),
        ("hr_interactive", "collision_rate", "Human-replay (interactive)\ncollision rate"),
    ]

    # Show reg and unreg as separate rows (only those actually present).
    reg_groups = sorted(df["is_regularized"].unique())  # e.g. [False, True]
    nrows = len(reg_groups)
    ncols = len(panels)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for r, is_reg in enumerate(reg_groups):
        df_reg = df[df["is_regularized"] == is_reg]
        reg_label = "regularized" if is_reg else "unregularized"

        for c, (mode, metric, title) in enumerate(panels):
            ax = axes[r, c]
            sub = df_reg[df_reg["mode"] == mode]
            if sub.empty:
                ax.set_title(f"{reg_label}\n{title}\n(no data)")
                ax.set_xlabel("sensor noise")
                ax.set_ylabel("dynamics noise")
                continue

            # Mean over scenes per (dynamics_noise, sensor_noise) cell.
            agg = sub.groupby(["dynamics_noise", "sensor_noise"])[metric].mean().reset_index()
            pivot = (
                agg.pivot(
                    index="dynamics_noise",
                    columns="sensor_noise",
                    values=metric,
                )
                .sort_index()
                .sort_index(axis=1)
            )

            # ratios -> percentages, only at the display step
            pivot_pct = pivot * 100.0
            arr = pivot_pct.values

            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                ax.set_title(f"{reg_label}\n{title}\n(no data)")
                ax.set_xlabel("sensor noise")
                ax.set_ylabel("dynamics noise")
                continue
            vmin = float(finite.min())
            vmax = float(finite.max())

            im = ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                cmap="Reds",
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xticks(range(len(pivot_pct.columns)))
            ax.set_xticklabels([f"{v:g}" for v in pivot_pct.columns])
            ax.set_yticks(range(len(pivot_pct.index)))
            ax.set_yticklabels([f"{v:g}" for v in pivot_pct.index])

            _annotate_heatmap(ax, arr, vmin, vmax)

            ax.set_xlabel("sensor noise")
            ax.set_ylabel("dynamics noise")
            ax.set_title(f"{reg_label}\n{title}")
            fig.colorbar(im, ax=ax, label="rate (%)")

    fig.suptitle("Collision rate vs training noise (dynamics × sensor)", y=1.00)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


def main():
    base_config = load_config(ENV_NAME)
    all_rows = evaluate_noise_checkpoints(base_config)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)  # CSV stays in ratios
    print(f"\nResults saved to {OUTPUT_CSV} ({len(df)} rows)")

    if not df.empty:
        summary = df.groupby(["checkpoint", "mode", "is_regularized", "dynamics_noise", "sensor_noise"]).agg(
            scenes=("score", "count"),
            score=("score", "mean"),
            collision_rate=("collision_rate", "mean"),
            at_fault_collision_rate=("at_fault_collision_rate", "mean"),
            rear_collision_rate=("rear_collision_rate", "mean"),
            offroad_rate=("offroad_rate", "mean"),
        )
        # Convert rate columns to % for the printed summary only
        summary_display = summary.copy()
        for col in ["collision_rate", "at_fault_collision_rate", "rear_collision_rate", "offroad_rate"]:
            if col in summary_display.columns:
                summary_display[col] = summary_display[col] * 100.0
        summary_display = summary_display.rename(
            columns={
                "collision_rate": "collision_rate_pct",
                "at_fault_collision_rate": "at_fault_collision_rate_pct",
                "rear_collision_rate": "rear_collision_rate_pct",
                "offroad_rate": "offroad_rate_pct",
            }
        )
        print(f"\n{summary_display}")

    plot_noise_vs_collision(df)
    return df


if __name__ == "__main__":
    main()
