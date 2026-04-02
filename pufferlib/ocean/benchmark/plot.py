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


def _fmt_maps(n):
    """Format a map count as a human-readable label: 10 -> '10', 1000 -> '1k', 50000 -> '50k'."""
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(int(n))


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
    scaling_modes = ["scaling_sp_val", "scaling_hr_interactive"]
    scaling_df = df[df["mode"].isin(scaling_modes)].copy()
    if scaling_df.empty:
        return None
    if "sp_maps" not in scaling_df.columns:
        return None

    scaling_df["collision_rate_pct"] = scaling_df["collision_rate"] * 100
    scaling_df["score_pct"] = scaling_df["score"] * 100
    return scaling_df


def _scaling_scatter_common(
    agg, subplot_specs, anchor_vals, color_map, marker_map, figsize, save_path, clip_y_zero=True, ref_lines=None
):
    """Shared logic for scaling scatter plots.

    Args:
        ref_lines: Optional list (one per subplot) of lists of (value, color, linestyle, label) tuples
                   to draw as horizontal reference lines.
    """
    _set_style(len(anchor_vals))
    fig, axes = plt.subplots(1, len(subplot_specs), figsize=figsize)
    if len(subplot_specs) == 1:
        axes = [axes]

    for ax, (mode, y_col, yerr_col, ylabel, title) in zip(axes, subplot_specs):
        mode_agg = agg[agg["mode"] == mode] if "mode" in agg.columns else agg
        for anchor_val in anchor_vals:
            grp = mode_agg[mode_agg["anchor_maps"] == anchor_val].sort_values("sp_maps")
            if grp.empty:
                continue
            label = grp["anchor_label"].iloc[0]
            ax.errorbar(
                grp["sp_maps"],
                grp[y_col],
                yerr=grp[yerr_col],
                marker=marker_map[anchor_val],
                color=color_map[anchor_val],
                capsize=3,
                linewidth=1.5,
                markersize=8,
                label=label,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else str(int(x)))
        )
        ax.set_xlabel("Self-play training maps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if clip_y_zero:
            ax.set_ylim(bottom=0)
        if ref_lines is not None:
            subplot_idx = list(axes).index(ax)
            for value, color, linestyle, label in ref_lines[subplot_idx]:
                ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.7, label=label)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, title="Anchor data")
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def _build_anchor_style_maps(anchor_vals):
    """Build consistent color and marker maps for anchor values."""
    palette = sns.color_palette("colorblind", n_colors=len(anchor_vals))
    color_map = {v: palette[i] for i, v in enumerate(anchor_vals)}
    markers = ["X", "o", "s", "D", "^", "v", "P", "*"]
    marker_map = {v: markers[i % len(markers)] for i, v in enumerate(anchor_vals)}
    return color_map, marker_map


def plot_scaling_scatter(df, save_path="eval_scaling_scatter.pdf"):
    """Summary scaling figure: 3-column scatter plot.

    x-axis: self-play training maps (sp_maps), log-scaled
    color:  anchor maps (human data used to train anchor); unreg → 0
    shape:  anchor maps

    Subplot 0: Self-play score (scaling_sp_val)
    Subplot 1: Self-play collision rate (scaling_sp_val)
    Subplot 2: Human-replay collision rate / ZSC (scaling_hr_interactive)
    """
    scaling_df = _prepare_scaling_metadata(df)
    if scaling_df is None:
        print("  No scaling data found — skipping plot_scaling_scatter.")
        return None

    scaling_df["anchor_maps"] = scaling_df["anchor_maps"].fillna(0).astype(int)
    scaling_df["at_fault_collision_rate_pct"] = scaling_df["at_fault_collision_rate"] * 100

    agg = (
        scaling_df.groupby(["sp_maps", "anchor_maps", "mode"])[
            ["collision_rate_pct", "score_pct", "at_fault_collision_rate_pct"]
        ]
        .agg(["mean", "sem"])
        .reset_index()
    )
    agg.columns = [
        "sp_maps",
        "anchor_maps",
        "mode",
        "coll_mean",
        "coll_sem",
        "score_mean",
        "score_sem",
        "at_fault_coll_mean",
        "at_fault_coll_sem",
    ]
    agg["anchor_label"] = agg["anchor_maps"].apply(
        lambda v: f"anchor = {_fmt_maps(v)}" if v > 0 else "no anchor (unreg)"
    )

    anchor_vals = sorted(agg["anchor_maps"].unique())
    color_map, marker_map = _build_anchor_style_maps(anchor_vals)

    subplot_specs = [
        ("scaling_sp_val", "score_mean", "score_sem", "Score (%)", "Self-play score (%) — validation"),
        ("scaling_sp_val", "coll_mean", "coll_sem", "Collision rate (%)", "Self-play collision rate (%) — validation"),
        (
            "scaling_hr_interactive",
            "coll_mean",
            "coll_sem",
            "Collision rate (%)",
            "Human-replay collision rate (%) — interactive",
        ),
        (
            "scaling_hr_interactive",
            "at_fault_coll_mean",
            "at_fault_coll_sem",
            "At-fault collision rate (%)",
            "Human-replay at-fault collision rate (%) — interactive",
        ),
    ]

    return _scaling_scatter_common(
        agg,
        subplot_specs,
        anchor_vals,
        color_map,
        marker_map,
        figsize=(24, 5),
        save_path=save_path,
    )


def plot_scaling_wosac(wosac_df, save_path="eval_scaling_wosac.pdf"):
    """WOSAC scaling figure: 4-column scatter plot.

    x-axis: self-play training maps (sp_maps), log-scaled
    color:  anchor maps; unreg → 0
    shape:  anchor maps

    Subplots: realism_meta_score, kinematic_metrics, interactive_metrics, map_based_metrics
    """
    if wosac_df is None or wosac_df.empty:
        print("  No WOSAC data found — skipping plot_scaling_wosac.")
        return None

    wdf = wosac_df.copy()
    wdf["anchor_maps"] = wdf["anchor_maps"].fillna(0).astype(int)

    wosac_metrics = [
        "realism_meta_score",
        "kinematic_metrics",
        "interactive_metrics",
        "map_based_metrics",
    ]
    agg = wdf.groupby(["sp_maps", "anchor_maps"])[wosac_metrics].agg(["mean", "sem"]).reset_index()
    # Flatten multi-level columns
    flat_cols = ["sp_maps", "anchor_maps"]
    for m in wosac_metrics:
        flat_cols.extend([f"{m}_mean", f"{m}_sem"])
    agg.columns = flat_cols

    agg["anchor_label"] = agg["anchor_maps"].apply(
        lambda v: f"anchor = {_fmt_maps(v)}" if v > 0 else "no anchor (unreg)"
    )

    anchor_vals = sorted(agg["anchor_maps"].unique())
    color_map, marker_map = _build_anchor_style_maps(anchor_vals)

    titles = [
        "Realism meta-score",
        "Kinematic metrics",
        "Interactive metrics",
        "Map-based metrics",
    ]
    subplot_specs = [(None, f"{m}_mean", f"{m}_sem", t, t) for m, t in zip(wosac_metrics, titles)]

    # Reference baselines per subplot (realism, kinematic, interactive, map)
    smart_scores = [0.7818, 0.5200, 0.8914, 0.8378]
    random_scores = [0.4459, 0.0506, 0.34, 0.4704]
    ref_lines = [
        [
            (smart, "green", "--", "SMART"),
            (rand, "grey", "--", "Random"),
        ]
        for smart, rand in zip(smart_scores, random_scores)
    ]

    return _scaling_scatter_common(
        agg,
        subplot_specs,
        anchor_vals,
        color_map,
        marker_map,
        figsize=(22, 5),
        save_path=save_path,
        clip_y_zero=False,
        ref_lines=ref_lines,
    )


def make_all_figures(df=None, wosac_df=None):
    """Generate all evaluation figures."""
    print("\nGenerating figures...")
    if df is not None and not df.empty:
        plot_scores(df)
        print("  Saved eval_scores.pdf")
        plot_collision_rates(df)
        print("  Saved eval_collision_rates.pdf")
        plot_scaling_scatter(df)
        print("  Saved eval_scaling_scatter.pdf")
    plot_scaling_wosac(wosac_df)
    print("  Saved eval_scaling_wosac.pdf")


if __name__ == "__main__":
    import os
    import pandas as pd

    EVAL_CSV = "checkpoint_eval_results.csv"
    WOSAC_CSV = "checkpoint_wosac_results.csv"

    df = None
    wosac_df = None

    if os.path.exists(EVAL_CSV):
        df = pd.read_csv(EVAL_CSV)
        print(f"Loaded {EVAL_CSV} ({len(df)} rows)")
    else:
        print(f"{EVAL_CSV} not found — skipping standard eval figures.")

    if os.path.exists(WOSAC_CSV):
        wosac_df = pd.read_csv(WOSAC_CSV)
        print(f"Loaded {WOSAC_CSV} ({len(wosac_df)} rows)")
    else:
        print(f"{WOSAC_CSV} not found — skipping WOSAC figures.")

    make_all_figures(df, wosac_df)
