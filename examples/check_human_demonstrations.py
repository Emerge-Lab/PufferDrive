"""Evaluate inferred-expert-action quality on the delta-local dynamics model.

Compares discrete (bin-quantized) vs continuous (direct float) expert actions.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pufferlib.pufferl import load_config, load_env
from pufferlib.ocean.benchmark.evaluator_minimal import CheckpointEvaluator


ENV_NAME = "puffer_drive"
NUM_ITERATIONS = 20
EPISODE_LENGTH = 91
NUM_VIDEOS = 0
ACTION_TYPES = ["discrete", "continuous"]
VIDEO_DIR = "videos/inferred_expert"
PLOT_DIR = "plots/inferred_expert"
DPI = 120

PALETTE = {"discrete": "tab:blue", "continuous": "tab:green"}

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def run_eval(action_type):
    args = load_config(ENV_NAME)
    args["env"]["control_mode"] = "inferred_expert_actions"
    args["env"]["action_type"] = action_type
    args["env"]["episode_length"] = EPISODE_LENGTH
    args["env"]["termination_mode"] = 1
    args["env"]["num_agents"] = 256
    args["env"]["async_resets"] = False
    args["env"]["resample_frequency"] = 0
    args["vec"] = dict(backend="PufferEnv", num_envs=1)

    env = load_env(ENV_NAME, args)
    evaluator = CheckpointEvaluator(args)

    num_envs = env.driver_env.num_envs
    total_scenarios = NUM_ITERATIONS * num_envs
    print(f"\n[{action_type}] Total scenarios: {total_scenarios} ({NUM_ITERATIONS} iterations x {num_envs} envs)")

    rows = []
    for it in range(NUM_ITERATIONS):
        env_logs = evaluator.rollout(env=env, policy=None, render_env_idx=0)
        populated = [log for log in env_logs if log and log.get("n", 0) > 0]

        for log in populated:
            rows.append(
                {
                    "action_type": action_type,
                    "iteration": it,
                    "route_progress": log["route_progress"],
                    "lateral_error": log["lateral_error_avg"],
                    "longitudinal_error": log["longitudinal_error_avg"],
                    "collision_rate": log["collision_rate"],
                    "offroad_rate": log["offroad_rate"],
                    "ADE": log["displacement_error_avg"],
                }
            )

    # Render videos
    os.makedirs(f"{VIDEO_DIR}/{action_type}", exist_ok=True)
    n_videos = min(NUM_VIDEOS, num_envs)
    print(f"[{action_type}] Recording {n_videos} videos to {VIDEO_DIR}/{action_type}/")
    for env_idx in range(n_videos):
        evaluator.rollout(env=env, policy=None, render_env_idx=env_idx)
        env.driver_env.stop_recorder(env_idx)
        scenario_id = env.driver_env.scenario_ids[env_idx]
        src = f"{scenario_id}.mp4"
        if os.path.exists(src):
            dst = os.path.join(f"{VIDEO_DIR}/{action_type}", src)
            shutil.move(src, dst)
            print(f"  env {env_idx}: {dst}")
        else:
            print(f"  env {env_idx}: missing {src}")

    env.close()
    return pd.DataFrame(rows)


def plot_summary(df, save_path):
    df = df.copy()
    df["collision_rate_pct"] = df["collision_rate"] * 100
    df["offroad_rate_pct"] = df["offroad_rate"] * 100
    df["route_progress_pct"] = df["route_progress"] * 100

    palette = [PALETTE[at] for at in ACTION_TYPES]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Subplot 0: grouped bar of rates ---
    ax = axes[0]
    rate_cols = ["collision_rate_pct", "offroad_rate_pct", "route_progress_pct"]
    rate_labels = ["collision_rate", "offroad_rate", "route_progress"]
    long_df = df.melt(
        id_vars=["action_type"],
        value_vars=rate_cols,
        var_name="metric",
        value_name="value",
    )
    metric_to_label = dict(zip(rate_cols, rate_labels))
    long_df["metric"] = long_df["metric"].map(metric_to_label)

    sns.barplot(
        data=long_df,
        x="metric",
        y="value",
        hue="action_type",
        hue_order=ACTION_TYPES,
        order=rate_labels,
        palette=palette,
        errorbar="se",
        ax=ax,
        alpha=0.8,
    )
    ax.set_xlabel("")
    ax.set_ylabel("SDC rates (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, title="")
    sns.despine(ax=ax)

    # Value labels above each bar (positioned above error bars when present)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=6, fontsize=10)

    # --- Subplots 1 & 2: overlaid histograms ---
    for ax, col, xlabel, ylabel in [
        (axes[1], "lateral_error", "lateral error (m)", "count"),
        (axes[2], "longitudinal_error", "longitudinal error (m)", "count"),
    ]:
        for at in ACTION_TYPES:
            sub = df[df["action_type"] == at]
            ax.hist(
                sub[col],
                bins=40,
                alpha=0.55,
                color=PALETTE[at],
                edgecolor="black",
                linewidth=0.5,
                label=at,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(frameon=False)
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig


def generate_latex_table(df, save_path):
    """LaTeX table comparing discrete vs continuous inferred expert actions.

    Reports mean ± SE aggregated across iterations.

    Required LaTeX packages:
      \\usepackage{booktabs}
    """
    # (mean_col, header, higher_is_better, as_pct, decimals)
    specs = [
        ("route_progress", r"Route prog. (\%) $\uparrow$", True, True, 1),
        ("collision_rate", r"Coll. (\%) $\downarrow$", False, True, 1),
        ("offroad_rate", r"Off-road (\%) $\downarrow$", False, True, 1),
        ("ADE", r"ADE (m) $\downarrow$", False, False, 3),
        ("lateral_error", r"Lateral L2 (m) $\downarrow$", False, False, 3),
        ("longitudinal_error", r"Longitudinal L2 (m) $\downarrow$", False, False, 3),
    ]
    specs = [s for s in specs if s[0] in df.columns]

    # Aggregate mean and SEM per action_type
    agg = df.groupby("action_type")[[s[0] for s in specs]].agg(["mean", "sem"])

    def _fmt_cell(at, mean_col, as_pct, decimals):
        mean = agg.loc[at, (mean_col, "mean")]
        sem = agg.loc[at, (mean_col, "sem")]
        if pd.isna(mean):
            return "---"
        m_val = mean * 100 if as_pct else mean
        s_val = sem * 100 if (as_pct and pd.notna(sem)) else sem
        fmt = f".{decimals}f"
        if pd.notna(s_val) and s_val != 0:
            return f"${m_val:{fmt}} \\pm {s_val:{fmt}}$"
        return f"{m_val:{fmt}}"

    col_spec = "l" + "r" * len(specs)
    headers = [s[1] for s in specs]

    lines = []
    lines.append(r"% Requires: \usepackage{booktabs}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Inferred-expert-action quality on the delta-local dynamics model. "
        r"Comparison of discrete (bin-quantized) vs continuous (direct float) expert actions. "
        f"Aggregated over {df['iteration'].nunique()} iterations "
        f"({len(df)} pooled samples). "
        r"Values are mean $\pm$ SE.}"
    )
    lines.append(r"\label{tab:inferred_expert_actions}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(r"Action type & " + " & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for at in ACTION_TYPES:
        if at not in agg.index:
            continue
        cells = [at]
        for mean_col, _h, _hib, as_pct, decimals in specs:
            cells.append(_fmt_cell(at, mean_col, as_pct, decimals))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    _ensure_dir(save_path)
    with open(save_path, "w") as f:
        f.write(latex_str)
    print(f"  LaTeX table written to {save_path}")
    return latex_str


def main():
    df = pd.concat([run_eval(at) for at in ACTION_TYPES], ignore_index=True)

    print(f"\nTotal pooled samples: {len(df)}")
    print("\nMean ± std by action_type:")
    rate_metrics = {"collision_rate", "offroad_rate", "route_progress"}
    metric_keys = ["collision_rate", "offroad_rate", "route_progress", "ADE", "lateral_error", "longitudinal_error"]
    for at in ACTION_TYPES:
        sub = df[df["action_type"] == at]
        print(f"\n  [{at}] n={len(sub)}")
        for k in metric_keys:
            if k not in sub.columns:
                continue
            if k in rate_metrics:
                print(f"    {k:<20s} {sub[k].mean() * 100:.2f}% ± {sub[k].std() * 100:.2f}%")
            else:
                print(f"    {k:<20s} {sub[k].mean():.4f} ± {sub[k].std():.4f}")

    save_path = f"{PLOT_DIR}/summary.pdf"
    plot_summary(df, save_path)
    print(f"\nPlot saved to {save_path}")

    table_path = f"{PLOT_DIR}/summary.tex"
    generate_latex_table(df, table_path)


if __name__ == "__main__":
    main()
