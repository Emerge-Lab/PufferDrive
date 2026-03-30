"""Plotting functions for checkpoint evaluation results."""

import re
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

DPI = 600


def _set_style(n_policies):
    sns.set("notebook", font_scale=1.05, rc={"figure.figsize": (16, 5)})
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")
    mpl.rcParams["lines.markersize"] = 8
    return sns.color_palette("colorblind", n_colors=n_policies)


def _short_name(checkpoint_path):
    """Extract a short label from a checkpoint path."""
    return checkpoint_path.split("/")[-1].replace(".pt", "")


def _draw_upper_bound(ax, value, label="upper bound"):
    """Draw a green dashed line indicating an upper bound (e.g. perfect score)."""
    ax.axhline(value, color="green", linestyle="--", linewidth=1.5, alpha=0.7, label=label)


def _draw_lower_bound(ax, value, label="lower bound"):
    """Draw a purple dashed line indicating a lower bound (e.g. zero collisions)."""
    ax.axhline(value, color="purple", linestyle="--", linewidth=1.5, alpha=0.7, label=label)


def _format_percent(ax):
    """Format y-axis as percentages and clip at 0."""
    ax.set_ylim(bottom=0)


def plot_scores(df, save_path="eval_scores.pdf"):
    """Figure 1: Self-play and human-replay scores on validation sets.

    Three columns:
      1) Self-play score (sp_val)
      2) Human-replay score on full validation (hr_val)
      3) Human-replay score on interactive scenes (hr_interactive)
    """
    df = df.copy()
    df["policy"] = df["checkpoint"].apply(_short_name)
    df["score_pct"] = df["score"] * 100
    palette = _set_style(df["policy"].nunique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    modes = ["sp_val", "hr_val", "hr_interactive"]
    titles = [
        "Self-play score (validation)",
        "Human-replay score (validation)",
        "Human-replay score (interactive)",
    ]

    for ax, mode, title in zip(axes, modes, titles):
        subset = df[df["mode"] == mode]
        sns.barplot(data=subset, x="policy", y="score_pct", errorbar="se", palette=palette, ax=ax, alpha=0.8)
        _draw_upper_bound(ax, 100, label="perfect score")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, loc="lower right")
        _format_percent(ax)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_collision_rates(df, save_path="eval_collision_rates.pdf"):
    """Figure 2: Self-play and human-replay collision rates on validation sets.

    Three columns:
      1) Self-play collision rate (sp_val)
      2) Human-replay collision rate on full validation (hr_val)
      3) Human-replay collision rate on interactive scenes (hr_interactive)
    """
    df = df.copy()
    df["policy"] = df["checkpoint"].apply(_short_name)
    df["collision_rate_pct"] = df["collision_rate"] * 100
    palette = _set_style(df["policy"].nunique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    modes = ["sp_val", "hr_val", "hr_interactive"]
    titles = [
        "Self-play collision rate (validation)",
        "Human-replay collision rate (validation)",
        "Human-replay collision rate (interactive)",
    ]

    for ax, mode, title in zip(axes, modes, titles):
        subset = df[df["mode"] == mode]
        sns.barplot(data=subset, x="policy", y="collision_rate_pct", palette=palette, ax=ax, alpha=0.8)
        _draw_lower_bound(ax, 0.0, label="zero collisions")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylabel("Collision rate (%)")
        _format_percent(ax)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def _prepare_scaling_metadata(df):
    """Shared preprocessing for scaling plots. Returns None if data is missing."""
    scaling_modes = ["scaling_sp_train", "scaling_sp_val", "scaling_hr_val"]
    scaling_df = df[df["mode"].isin(scaling_modes)].copy()
    if scaling_df.empty:
        return None
    if "num_train_maps" not in scaling_df.columns or "is_regularized" not in scaling_df.columns:
        return None

    scaling_df["reg_label"] = scaling_df["is_regularized"].map({True: "regularized", False: "unregularized"})
    scaling_df["score_pct"] = scaling_df["score"] * 100
    scaling_df["collision_rate_pct"] = scaling_df["collision_rate"] * 100
    return scaling_df


def _agg(src, metric):
    g = src.groupby(["num_train_maps", "reg_label"])[metric]
    agg = g.agg(["mean", "sem"]).reset_index()
    agg.columns = ["num_train_maps", "reg_label", "mean", "sem"]
    agg = agg.sort_values("num_train_maps")
    return agg


def _plot_scaling_metadata_axes(plot_specs, color_map):
    """Shared plotting loop for scaling subplots."""
    for ax, agg_df, title, ylabel in plot_specs:
        for label, grp in agg_df.groupby("reg_label"):
            ax.errorbar(
                grp["num_train_maps"],
                grp["mean"],
                yerr=grp["sem"],
                marker="o",
                capsize=3,
                label=label,
                color=color_map[label],
                linewidth=2,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else str(int(x)))
        )
        ax.set_xlabel("Number of maps in training dataset")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)
        sns.despine(ax=ax)


def plot_scaling_metadata_scores(df, save_path="eval_scaling_metadata_scores.pdf"):
    """Figure 3: Scaling scores — train, val, and generalization gap.

    Three columns (first two share y-axis):
      0) Self-play score on training set
      1) Self-play score on validation set
      2) Score generalization gap (train - val)
    """
    scaling_df = _prepare_scaling_metadata(df)
    if scaling_df is None:
        print("  No scaling data found — skipping plot_scaling_metadata_scores.")
        return None

    sp_train_score = _agg(scaling_df[scaling_df["mode"] == "scaling_sp_train"], "score_pct")
    sp_val_score = _agg(scaling_df[scaling_df["mode"] == "scaling_sp_val"], "score_pct")

    # Compute delta (train - val) per (num_train_maps, reg_label)
    delta = sp_train_score.merge(sp_val_score, on=["num_train_maps", "reg_label"], suffixes=("_train", "_val"))
    delta["mean"] = delta["mean_train"] - delta["mean_val"]
    delta["sem"] = np.sqrt(delta["sem_train"] ** 2 + delta["sem_val"] ** 2)

    _set_style(2)
    color_map = {"regularized": "tab:orange", "unregularized": "tab:blue"}

    fig = plt.figure(figsize=(16, 5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2, sharey=ax0)
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.8)

    plot_specs = [
        (ax0, sp_train_score, "Score (%) — self-play (train)", "Score (%)"),
        (ax1, sp_val_score, "Score (%) — self-play (val)", "Score (%)"),
        (ax2, delta, "Score generalization gap (train − val)", "Delta score (%)"),
    ]

    _plot_scaling_metadata_axes(plot_specs, color_map)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_scaling_metadata_collisions(df, save_path="eval_scaling_metadata_collisions.pdf"):
    """Figure 4: Scaling collision rates — self-play and ZSC gap.

    Two columns:
      0) Self-play collision rate on validation set
      1) ZSC gap: human-replay collision rate on validation set
    """
    scaling_df = _prepare_scaling_metadata(df)
    if scaling_df is None:
        print("  No scaling data found — skipping plot_scaling_metadata_collisions.")
        return None

    sp_coll = _agg(scaling_df[scaling_df["mode"] == "scaling_sp_val"], "collision_rate_pct")
    hr_coll = _agg(scaling_df[scaling_df["mode"] == "scaling_hr_val"], "collision_rate_pct")

    _set_style(2)
    color_map = {"regularized": "tab:orange", "unregularized": "tab:blue"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    plot_specs = [
        (axes[0], sp_coll, "Collision rate (%) — self-play (val)", "Collision rate (%)"),
        (axes[1], hr_coll, "ZSC gap: Human-replay collision rate (val)", "Collision rate (%)"),
    ]

    _plot_scaling_metadata_axes(plot_specs, color_map)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def make_all_figures(df):
    """Generate all evaluation figures."""
    print("\nGenerating figures...")
    plot_scores(df)
    print("  Saved eval_scores.pdf")
    plot_collision_rates(df)
    print("  Saved eval_collision_rates.pdf")
    plot_scaling_metadata_scores(df)
    print("  Saved eval_scaling_metadata_scores.pdf")
    plot_scaling_metadata_collisions(df)
    print("  Saved eval_scaling_metadata_collisions.pdf")
