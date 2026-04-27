"""Evaluate gamma-scan RL checkpoints.

Evaluates every checkpoint in CHECKPOINTS_PATH matching
    [reg|unreg]_[dynamics]_gamma_[VALUE].pt
in three modes:
  1. Self-play on validation maps
  2. Human-replay on validation maps
  3. Human-replay on interactive scenes

Then plots metrics (mean ± SEM across rollout rows) vs gamma in a 4 x 4 grid
where columns share a consistent mode (col 0 = sp_val, col 1 = hr_val,
col 2 = hr_interactive, col 3 = hr_val with at-fault metric):

  Row 1: collision rate            (sp_val, hr_val, hr_interactive, hr_val/at_fault)
  Row 2: rear collision rate       (        hr_val, hr_interactive)
  Row 3: route progress            (sp_val, hr_val, hr_interactive)
  Row 4: score                     (sp_val, hr_val, hr_interactive)

Why SEM and not std:
    Each row in `df` is one populated env log (effectively per-agent for
    rate metrics), so `collision_rate` per row is close to binary. The std
    of a binary variable with mean p is sqrt(p(1-p)), which for p=5% is
    ~22pp — mathematically correct but visually useless and not the
    quantity you want to plot here. With N ~ 5000 rows per checkpoint,
    SEM = std/sqrt(N) is the meaningful "uncertainty in the mean estimate"
    and gives readable error bars. (You can't get a std across checkpoints
    because the gamma scan has one checkpoint per (is_reg, gamma) cell.)

Unit convention:
    All rate metrics (in RATE_COLS_FOR_DISPLAY) are stored internally and in
    the CSV as ratios in [0, 1] and converted to percentages only at display
    time. Non-rate metrics like score and route_progress are plotted as-is.

Examples of accepted filenames:
    unreg_classic_gamma_0.995.pt
    reg_delta_gamma_0.997.pt

Usage:
    python evaluate_gamma_checkpoints.py
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
CHECKPOINTS_PATH = "models/gamma_ablation_cpts"
DETERMINISTIC = True

VAL_MAP_DIR = "resources/drive/binaries/validation"  # 10k maps
INTERACTIVE_MAP_DIR = "resources/drive/binaries/interactive_data_validation"  # 200 maps
NUM_TOTAL_EVAL_AGENTS = 1024 * 3
NUM_AGENTS_PER_VECENV = 1024
ENV_NAME = "puffer_drive"
DATASET = "womd"
OUTPUT_CSV = "results/gamma_ablation_eval_results.csv"
PLOT_PATH = "results/gamma_vs_collision_rate.png"

DYNAMICS_CONFIG_MAP = {
    "delta": "delta_local",
    "classic": "classic",
    "jerk": "jerk",
}

# Rate columns that should be displayed as percentages (data stays as ratios).
# Metrics not in this list are plotted as-is, with their raw values.
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
    """Build an eval-ready config from the checkpoint config."""
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
            mean_coll = np.mean([r["collision_rate"] for r in rows])
            mean_coll_fault = np.mean([r["at_fault_collision_rate"] for r in rows])
            print(
                f"  {mode_name}: {n_scenes} scenes, score={mean_score:.3f}, "
                f"collision_rate={mean_coll:.3f}, at_fault_collision_rate={mean_coll_fault:.3f}"
            )
        else:
            print(f"  {mode_name}: no populated scenes")
    except Exception as e:
        print(f"  {mode_name} failed (non-fatal): {e}")

    env.close()
    return rows


def parse_gamma_checkpoint_name(filename):
    """Parse a gamma-scan checkpoint filename.

    Format: [reg|unreg]_[dynamics]_gamma_[VALUE].pt

    Examples:
        unreg_classic_gamma_0.995.pt -> (False, "classic", 0.995)
        reg_delta_gamma_0.997.pt     -> (True,  "delta",   0.997)

    Returns:
        (is_regularized: bool, dynamics: str, gamma: float) or None.
    """
    stem = filename.replace(".pt", "")
    # `\w+?` is non-greedy so multi-token dynamics names (e.g. delta_local)
    # also work — the engine backtracks until `_gamma_` matches.
    match = re.match(r"^(reg|unreg)_(\w+?)_gamma_(\d+\.\d+)$", stem)
    if not match:
        return None
    is_reg = match.group(1) == "reg"
    dynamics = match.group(2)
    gamma = float(match.group(3))
    return is_reg, dynamics, gamma


def evaluate_gamma_checkpoints(base_config):
    """Run sp_val, hr_val, and hr_interactive for each gamma-scan checkpoint."""
    entries = []
    for fname in sorted(os.listdir(CHECKPOINTS_PATH)):
        if not fname.endswith(".pt"):
            continue
        parsed = parse_gamma_checkpoint_name(fname)
        if parsed is None:
            print(f"  Warning: could not parse '{fname}', skipping")
            continue
        is_reg, dynamics, gamma = parsed
        entries.append((os.path.join(CHECKPOINTS_PATH, fname), is_reg, dynamics, gamma))

    # Sort by (reg flag, gamma) so the log is easy to read
    entries.sort(key=lambda x: (x[1], x[3]))

    if not entries:
        print(f"No checkpoints found in {CHECKPOINTS_PATH}")
        return []

    print(f"\nFound {len(entries)} checkpoints:")
    for cpt_path, is_reg, dynamics, gamma in entries:
        tag = "reg" if is_reg else "unreg"
        print(f"  {os.path.basename(cpt_path)}  ->  {tag}, {dynamics}, gamma={gamma}")

    all_rows = []

    for cpt_path, is_reg, dynamics, gamma in entries:
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

        # Tag every row with the gamma-scan metadata
        for row in sp_val_rows + hr_val_rows + hr_interactive_rows:
            row["is_regularized"] = is_reg
            row["dynamics"] = dynamics
            row["gamma"] = gamma

        all_rows.extend(sp_val_rows)
        all_rows.extend(hr_val_rows)
        all_rows.extend(hr_interactive_rows)

    return all_rows


def _plot_panel(ax, df, mode, metric, title):
    """Draw one panel: mean ± SEM across rollout rows vs gamma, split by reg flag.

    Uses SEM (standard error of the mean) — not std — because each row is one
    rollout log (effectively per-agent for rate metrics), so std is dominated
    by per-agent binary noise (sqrt(p(1-p))) rather than checkpoint-level
    variation. SEM = std/sqrt(N) gives a readable "uncertainty in the mean"
    appropriate for ~5000 rollouts per checkpoint.

    Rate-style metrics (those in RATE_COLS_FOR_DISPLAY) are converted to
    percentages at display time. Non-rate metrics (score, route_progress, etc.)
    are plotted as-is in their native units.

    Returns True if any data was drawn, False if the panel was empty.
    """
    is_rate = metric in RATE_COLS_FOR_DISPLAY
    scale = 100.0 if is_rate else 1.0
    ylabel = "rate (%)" if is_rate else metric.replace("_", " ")

    sub = df[df["mode"] == mode]
    if sub.empty:
        ax.set_title(f"{title}\n(no data)")
        ax.set_xlabel("gamma")
        ax.set_ylabel(ylabel)
        return False

    agg = sub.groupby(["is_regularized", "gamma"])[metric].agg(["mean", "sem", "count"]).reset_index()
    # SEM is NaN when count == 1 (sample std is undefined for n=1); treat as
    # zero so a single-row group still renders as a marker without an error bar.
    agg["sem"] = agg["sem"].fillna(0.0)

    drew_any = False
    for is_reg, group in agg.groupby("is_regularized"):
        label = "regularized" if is_reg else "unregularized"
        color = "tab:purple" if is_reg else "tab:blue"
        group = group.sort_values("gamma")
        # ratios -> percentages only for rate-style metrics
        ax.errorbar(
            group["gamma"],
            group["mean"] * scale,
            yerr=group["sem"] * scale,
            marker="o",
            capsize=3,
            linewidth=2,
            markersize=6,
            label=label,
            color=color,
        )
        drew_any = True

    ax.set_xlabel("gamma")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if drew_any:
        ax.legend()
    return drew_any


def plot_gamma_vs_collision(df, output_path=PLOT_PATH):
    """Plot metrics (mean ± SEM across rollout rows) vs gamma in a 4 x 4 grid.

    Layout — columns share a consistent mode (col 0 = sp_val, col 1 = hr_val,
    col 2 = hr_interactive, col 3 = hr_val with the at-fault metric):

      Row 1: collision rate         sp_val, hr_val, hr_interactive, hr_val/at_fault
      Row 2: rear collision rate           hr_val, hr_interactive
      Row 3: route progress         sp_val, hr_val, hr_interactive
      Row 4: score                  sp_val, hr_val, hr_interactive

    Error bars are ±1 SEM. Rate metrics (collision_rate etc.) are scaled to
    percentages at display time; route_progress and score are plotted in
    their native units.
    """
    if df.empty:
        print("No data to plot.")
        return

    # (row, col, mode, metric, title)
    panels = [
        # Row 0: collision rate
        (0, 0, "sp_val", "collision_rate", "Self-play (val)\ncollision rate"),
        (0, 1, "hr_val", "collision_rate", "Human-replay (val)\ncollision rate"),
        (0, 2, "hr_interactive", "collision_rate", "Human-replay (interactive)\ncollision rate"),
        (0, 3, "hr_val", "at_fault_collision_rate", "Human-replay (val)\nat-fault collision rate"),
        # Row 1: rear collision rate
        (1, 1, "hr_val", "rear_collision_rate", "Human-replay (val)\nrear collision rate"),
        (1, 2, "hr_interactive", "rear_collision_rate", "Human-replay (interactive)\nrear collision rate"),
        # Row 2: route progress
        (2, 0, "sp_val", "route_progress", "Self-play (val)\nroute progress"),
        (2, 1, "hr_val", "route_progress", "Human-replay (val)\nroute progress"),
        (2, 2, "hr_interactive", "route_progress", "Human-replay (interactive)\nroute progress"),
        # Row 3: score
        (3, 0, "sp_val", "score", "Self-play (val)\nscore"),
        (3, 1, "hr_val", "score", "Human-replay (val)\nscore"),
        (3, 2, "hr_interactive", "score", "Human-replay (interactive)\nscore"),
    ]

    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20), sharey=False)

    used = set()
    for r, c, mode, metric, title in panels:
        _plot_panel(axes[r, c], df, mode, metric, title)
        used.add((r, c))

    # Hide unused cells
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) not in used:
                axes[r, c].axis("off")

    fig.suptitle("Metrics vs discount factor (mean ± SEM across rollout rows)", y=1.00)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


def main():
    base_config = load_config(ENV_NAME)
    all_rows = evaluate_gamma_checkpoints(base_config)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)  # CSV stays in ratios
    print(f"\nResults saved to {OUTPUT_CSV} ({len(df)} rows)")

    if not df.empty:
        summary = df.groupby(["checkpoint", "mode"]).agg(
            scenes=("score", "count"),
            score=("score", "mean"),
            route_progress=("route_progress", "mean"),
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

    plot_gamma_vs_collision(df)
    return df


if __name__ == "__main__":
    main()
