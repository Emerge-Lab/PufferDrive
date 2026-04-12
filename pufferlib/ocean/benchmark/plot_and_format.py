"""Plotting functions for checkpoint evaluation results."""

import re
import warnings

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import pandas as pd

DPI = 600


def _ensure_dir(path):
    """Create parent directories for *path* if they don't already exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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
    """Draw a lightgreen striped line indicating an upper bound (e.g. perfect score)."""
    ax.axhline(value, color="lightgreen", linestyle=(0, (5, 2, 1, 2)), linewidth=1.5, alpha=0.9, label=label)


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


def _maps_to_human_time(maps):
    """Convert map count to human-readable time: minutes = (maps * 9 * 5) / 60."""
    minutes = (maps * 9 * 5) / 60
    if minutes >= 60:
        hours = minutes / 60
        if hours == int(hours):
            return f"{int(hours)} hour{'s' if int(hours) != 1 else ''}"
        return f"{hours:.1f} hours"
    if minutes == int(minutes):
        return f"{int(minutes)} minutes"
    return f"{minutes:.1f} minutes"


def _maps_to_human_hours(maps: int) -> float:
    """Convert map count to hours of human driving data.

    Each map is a 9-second scenario with ~5 controlled agents on average:
        hours = (maps × 9s × 5) / 3600
    """
    return (maps * 9 * 5) / 3600


# ---------------------------------------------------------------------------
# Shared colour convention
# ---------------------------------------------------------------------------

UNREG_COLOR = "k"
REG_COLORS = ["#ff7f0e", "#d62728", "#e377c2", "#9467bd", "#bcbd22", "#a8174a"]


def _reg_unreg_colors(anchor_vals):
    """Return a color dict mapping anchor_val -> color.

    anchor_val == 0  ->  black (unregularized)
    anchor_val  > 0  ->  successive entries from REG_COLORS
    """
    color_map = {}
    reg_idx = 0
    for v in sorted(anchor_vals):
        if v == 0:
            color_map[v] = UNREG_COLOR
        else:
            color_map[v] = REG_COLORS[reg_idx % len(REG_COLORS)]
            reg_idx += 1
    return color_map


def plot_scores(df, save_path="results/figures/eval_scores.pdf"):
    """Figure 1: Self-play and human-replay scores on validation sets.

    Three columns:
      1) Self-play score (sp_val)
      2) Human-replay score on full validation (hr_val)
      3) Human-replay score on interactive scenes (hr_interactive)
    """
    df = df.copy()
    df["policy"] = df["checkpoint"].apply(_short_name)
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
        sns.barplot(data=subset, x="policy", y="score", errorbar="se", palette=palette, ax=ax, alpha=0.8)
        _draw_upper_bound(ax, 1.0, label="perfect score")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8, loc="lower right")
        _format_percent(ax)
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
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
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def _build_anchor_style_maps(anchor_vals):
    """Build consistent color and marker maps for anchor values."""
    color_map = _reg_unreg_colors(anchor_vals)
    markers = ["X", "o", "s", "D", "^", "v", "P", "*"]
    marker_map = {v: markers[i % len(markers)] for i, v in enumerate(sorted(anchor_vals))}
    return color_map, marker_map


def plot_scaling_scatter(df, save_path="results/figures/eval_scaling_scatter.pdf"):
    """Scaling scatter plot: 5-column layout.

    x-axis: self-play training maps (sp_maps), log-scaled
    color/shape: anchor maps

    Subplots:
      0) Self-play score (validation)
      1) Self-play collision rate (validation)
      2) Self-play offroad rate (validation)
      3) Human-replay collision rate (interactive)
      4) Human-replay at-fault collision rate (interactive)
    """
    scaling_df = _prepare_scaling_metadata(df)
    if scaling_df is None:
        print("  No scaling data found — skipping plot_scaling_scatter.")
        return None

    scaling_df["anchor_maps"] = scaling_df["anchor_maps"].fillna(0).astype(int)
    scaling_df["at_fault_collision_rate_pct"] = scaling_df["at_fault_collision_rate"] * 100
    scaling_df["offroad_rate_pct"] = scaling_df["offroad_rate"] * 100

    if "dynamics" not in scaling_df.columns:
        scaling_df["dynamics"] = "delta"

    agg = (
        scaling_df.groupby(["sp_maps", "anchor_maps", "dynamics", "mode"])[
            ["collision_rate_pct", "score", "at_fault_collision_rate_pct", "offroad_rate_pct"]
        ]
        .agg(["mean", "sem"])
        .reset_index()
    )
    agg.columns = [
        "sp_maps",
        "anchor_maps",
        "dynamics",
        "mode",
        "coll_mean",
        "coll_sem",
        "score_mean",
        "score_sem",
        "at_fault_coll_mean",
        "at_fault_coll_sem",
        "offroad_mean",
        "offroad_sem",
    ]

    agg["series_key"] = agg.apply(lambda r: f"{r['dynamics']}_anchor{r['anchor_maps']}", axis=1)
    agg["anchor_label"] = agg.apply(
        lambda r: (
            f"regularized, anchor with {_maps_to_human_time(r['anchor_maps'])} of human data"
            if r["anchor_maps"] > 0
            else "unregularized"
        ),
        axis=1,
    )

    series_keys = sorted(agg["series_key"].unique())

    # Build color map: series_key -> color, using shared reg/unreg convention
    # Extract anchor_maps from each series_key to look up color
    anchor_vals = sorted(agg["anchor_maps"].unique())
    base_color_map = _reg_unreg_colors(anchor_vals)
    color_map = {}
    for sk in series_keys:
        anchor_val = int(sk.split("anchor")[1])
        color_map[sk] = base_color_map[anchor_val]

    markers = ["X", "o", "s", "D", "^", "v", "P", "*"]
    marker_map = {k: markers[i % len(markers)] for i, k in enumerate(series_keys)}

    # (mode, y_col, yerr_col, ylabel, title)
    subplot_specs = [
        ("scaling_sp_val", "score_mean", "score_sem", "Score", "Self-play score — validation"),
        ("scaling_sp_val", "coll_mean", "coll_sem", "Collision rate (%)", "Self-play collision rate (%) — validation"),
        (
            "scaling_sp_val",
            "offroad_mean",
            "offroad_sem",
            "Offroad rate (%)",
            "Self-play offroad rate (%) — validation",
        ),
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

    _set_style(len(series_keys))
    fig, axes = plt.subplots(1, len(subplot_specs), figsize=(30, 5))

    for ax, (mode, y_col, yerr_col, ylabel, title) in zip(axes, subplot_specs):
        mode_agg = agg[agg["mode"] == mode]
        for sk in series_keys:
            grp = mode_agg[mode_agg["series_key"] == sk].sort_values("sp_maps")
            if grp.empty:
                continue
            label = grp["anchor_label"].iloc[0]
            ax.errorbar(
                grp["sp_maps"],
                grp[y_col],
                yerr=grp[yerr_col],
                marker=marker_map[sk],
                color=color_map[sk],
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
        ax.set_ylim(bottom=0)
        if y_col == "score_mean":
            _draw_upper_bound(ax, 1.0, label="perfect score")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, title="Policy")
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_scaling_barplot(df, save_path="results/figures/eval_scaling_barplot.pdf"):
    """Bar plots for 50k self-play checkpoints on training set: score, collision rate, offroad rate."""
    scaling_df = df[df["mode"] == "scaling_sp_train"].copy()
    if scaling_df.empty:
        print("  No scaling_sp_train data found — skipping plot_scaling_barplot.")
        return None

    scaling_df["anchor_maps"] = scaling_df["anchor_maps"].fillna(0).astype(int)
    if "dynamics" not in scaling_df.columns:
        scaling_df["dynamics"] = "delta"

    # Filter to only 50k self-play checkpoints
    scaling_df = scaling_df[scaling_df["sp_maps"] == 50000]
    if scaling_df.empty:
        print("  No 50k sp_maps data found — skipping plot_scaling_barplot.")
        return None

    scaling_df["collision_rate_pct"] = scaling_df["collision_rate"] * 100
    scaling_df["offroad_rate_pct"] = scaling_df["offroad_rate"] * 100

    scaling_df["policy"] = scaling_df.apply(
        lambda r: (
            f"regularized ({_maps_to_human_time(r['anchor_maps'])})" if r["anchor_maps"] > 0 else "unregularized"
        ),
        axis=1,
    )

    # Build per-row palette using shared reg/unreg color convention
    anchor_vals = sorted(scaling_df["anchor_maps"].unique())
    base_color_map = _reg_unreg_colors(anchor_vals)
    # One color per unique policy label, preserving order
    policy_order = scaling_df.drop_duplicates("anchor_maps").sort_values("anchor_maps")
    palette = [base_color_map[a] for a in policy_order["anchor_maps"]]

    _set_style(len(anchor_vals))

    subplot_specs = [
        ("score", "Score", "Self-play score — training, 50k maps"),
        ("collision_rate_pct", "Collision rate (%)", "Self-play collision rate (%) — training, 50k maps"),
        ("offroad_rate_pct", "Offroad rate (%)", "Self-play offroad rate (%) — training, 50k maps"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (y_col, ylabel, title) in zip(axes, subplot_specs):
        sns.barplot(
            data=scaling_df,
            x="policy",
            y=y_col,
            errorbar="se",
            palette=palette,
            order=policy_order["policy"].tolist(),
            ax=ax,
            alpha=0.8,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        if y_col == "score":
            _draw_upper_bound(ax, 1.0, label="perfect score")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.tick_params(axis="x", rotation=30)
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_scaling_wosac(wosac_df, save_path="results/figures/eval_scaling_wosac.pdf"):
    """WOSAC scaling figure: 4-column scatter plot.

    x-axis: self-play training maps (sp_maps), log-scaled
    color:  anchor maps; unreg -> 0
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


def generate_scaling_latex_table(df, save_path="results/figures/eval_scaling_table.tex"):
    """Generate a LaTeX-formatted table with heatmap coloring.

    Score columns use a Greens colormap (darker = higher = better).
    Collision rate columns use a Reds colormap (lighter = lower = better).
    Colors are normalized per column.
    Unregularized rows come first and have a light-blue background in the anchor column.

    Required LaTeX packages:
      \\usepackage{booktabs}
      \\usepackage[table]{xcolor}
      \\usepackage{graphicx}   % for \\resizebox
      \\usepackage{makecell}
      \\usepackage{bm}
    """
    scaling_modes = ["scaling_sp_val", "scaling_hr_interactive"]
    scaling_df = df[df["mode"].isin(scaling_modes)].copy()
    if scaling_df.empty:
        print("  No scaling data found — skipping generate_scaling_latex_table.")
        return None

    scaling_df["anchor_maps"] = scaling_df["anchor_maps"].fillna(0).astype(int)

    sp_metrics = ["score", "collision_rate", "offroad_rate"]
    hr_metrics = [
        "score",
        "collision_rate",
        "at_fault_collision_rate",
        "rear_collision_rate",
        "route_progress",
        "lateral_error_avg",
    ]
    all_metrics = list(set(sp_metrics + hr_metrics))
    available_metrics = [m for m in all_metrics if m in scaling_df.columns]

    agg = scaling_df.groupby(["sp_maps", "anchor_maps", "mode"])[available_metrics].agg(["mean", "sem"]).reset_index()
    flat_cols = ["sp_maps", "anchor_maps", "mode"]
    for m in available_metrics:
        flat_cols.extend([f"{m}_mean", f"{m}_sem"])
    agg.columns = flat_cols

    sp = agg[agg["mode"] == "scaling_sp_val"].drop(columns=["mode"]).copy()
    hr = agg[agg["mode"] == "scaling_hr_interactive"].drop(columns=["mode"]).copy()

    sp_rename = {c: f"sp_{c}" for c in sp.columns if c not in ("sp_maps", "anchor_maps")}
    hr_rename = {c: f"hr_{c}" for c in hr.columns if c not in ("sp_maps", "anchor_maps")}
    sp = sp.rename(columns=sp_rename)
    hr = hr.rename(columns=hr_rename)

    merged = sp.merge(hr, on=["sp_maps", "anchor_maps"], how="outer")

    # unreg rows first, then reg rows, each sorted by sp_maps
    unreg = merged[merged["anchor_maps"] == 0].sort_values("sp_maps")
    reg = merged[merged["anchor_maps"] != 0].sort_values(["sp_maps", "anchor_maps"])
    merged = pd.concat([unreg, reg]).reset_index(drop=True)

    has_offroad = "offroad_rate" in available_metrics
    has_at_fault = "at_fault_collision_rate" in available_metrics
    has_rear = "rear_collision_rate" in available_metrics
    has_route_prog = "route_progress" in available_metrics
    has_lateral = "lateral_error_avg" in available_metrics

    # higher is better -> green (max = best, darker = higher)
    # lower is better  -> red   (min = best, lighter = lower)
    score_mean_cols = ["sp_score_mean", "hr_score_mean"]
    if has_route_prog:
        score_mean_cols.append("hr_route_progress_mean")

    coll_mean_cols = ["sp_collision_rate_mean", "hr_collision_rate_mean"]
    if has_offroad:
        coll_mean_cols.append("sp_offroad_rate_mean")
    if has_at_fault:
        coll_mean_cols.append("hr_at_fault_collision_rate_mean")
    if has_rear:
        coll_mean_cols.append("hr_rear_collision_rate_mean")
    if has_lateral:
        coll_mean_cols.append("hr_lateral_error_avg_mean")

    existing_score_cols = [c for c in score_mean_cols if c in merged.columns]
    existing_coll_cols = [c for c in coll_mean_cols if c in merged.columns]

    # Per-column min/max for normalization
    col_min = {}
    col_max = {}
    for c in existing_score_cols + existing_coll_cols:
        vals = merged[c].dropna()
        col_min[c] = vals.min() if not vals.empty else 0
        col_max[c] = vals.max() if not vals.empty else 1

    def _intensity(val, col, higher_is_better):
        if np.isnan(val):
            return 0
        vmin, vmax = col_min.get(col, 0), col_max.get(col, 1)
        if vmax == vmin:
            return 25
        t = (val - vmin) / (vmax - vmin)
        # green: darker = higher = better -> t as-is
        # red:   lighter = lower = better -> t as-is (low val -> light = low intensity)
        return int(5 + t * 45)

    def _fmt_score(mean, sem, col, is_best=False):
        if np.isnan(mean):
            return "---"
        intensity = _intensity(mean, col, higher_is_better=True)
        if not (np.isnan(sem) or sem == 0):
            text = f"$\\bm{{{mean:.3f} \\pm {sem:.3f}}}$" if is_best else f"${mean:.3f} \\pm {sem:.3f}$"
        else:
            text = f"\\textbf{{{mean:.3f}}}" if is_best else f"{mean:.3f}"
        return f"\\cellcolor{{green!{intensity}}} {text}"

    def _fmt_coll(mean, sem, col, is_best=False, as_pct=True, decimals=1):
        if np.isnan(mean):
            return "---"
        intensity = _intensity(mean, col, higher_is_better=False)
        m_val = mean * 100 if as_pct else mean
        s_val = sem * 100 if as_pct else sem
        fmt = f".{decimals}f"
        if not (np.isnan(s_val) or s_val == 0):
            text = f"$\\bm{{{m_val:{fmt}} \\pm {s_val:{fmt}}}}$" if is_best else f"${m_val:{fmt}} \\pm {s_val:{fmt}}$"
        else:
            text = f"\\textbf{{{m_val:{fmt}}}}" if is_best else f"{m_val:{fmt}}"
        return f"\\cellcolor{{red!{intensity}}} {text}"

    def _anchor_label(anchor_maps):
        return "0 (unreg.)" if anchor_maps == 0 else _maps_to_human_time(anchor_maps)

    # best: max for score cols, min for coll cols
    best = {}
    for col in existing_score_cols:
        best[col] = merged[col].max()
    for col in existing_coll_cols:
        best[col] = merged[col].min()

    def _is_best(col, val):
        if col not in best or np.isnan(val):
            return False
        return np.isclose(val, best[col])

    n_sp_metric_cols = 2 + int(has_offroad)
    n_hr_metric_cols = 2 + int(has_at_fault) + int(has_rear) + int(has_route_prog) + int(has_lateral)
    col_spec = "rr" + "|" + "r" * n_sp_metric_cols + "|" + "r" * n_hr_metric_cols

    lines = []
    lines.append(
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor}, \usepackage{graphicx}, \usepackage{makecell}, \usepackage{bm}"
    )
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Scaling evaluation on held-out Waymo scenarios. Self-play scores are reported on 10k randomly sampled validation scenarios; human-replay metrics are reported on 200 interactive validation scenarios.}"
    )
    lines.append(r"\label{tab:scaling_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    sp_header = r"\multicolumn{" + str(n_sp_metric_cols) + r"}{c|}{Self-play (test)}"
    hr_header = r"\multicolumn{" + str(n_hr_metric_cols) + r"}{c}{Human-replay (test)}"
    lines.append(r" & & " + sp_header + " & " + hr_header + r" \\")

    sp_metric_headers = ["Score $\\uparrow$", "Coll. (\\%) $\\downarrow$"]
    if has_offroad:
        sp_metric_headers.append("Offroad (\\%) $\\downarrow$")
    hr_metric_headers = ["Score $\\uparrow$", "Coll. (\\%) $\\downarrow$"]
    if has_at_fault:
        hr_metric_headers.append("At-fault (\\%) $\\downarrow$")
    if has_rear:
        hr_metric_headers.append("Rear coll. (\\%) $\\downarrow$")
    if has_route_prog:
        hr_metric_headers.append("Route prog. (\\%) $\\uparrow$")
    if has_lateral:
        hr_metric_headers.append("Lateral L2 $\\downarrow$")

    header2 = (
        "\\makecell{Self-play maps \\\\ (metadata)} & \\makecell{Anchor data \\\\ (human demonstrations)} & "
        + " & ".join(sp_metric_headers)
        + " & "
        + " & ".join(hr_metric_headers)
        + r" \\"
    )
    lines.append(header2)
    lines.append(r"\midrule")

    prev_was_unreg = None
    for _, row in merged.iterrows():
        is_unreg = int(row["anchor_maps"]) == 0

        # separator line between unreg and reg blocks
        if prev_was_unreg is True and not is_unreg:
            lines.append(r"\midrule")
        prev_was_unreg = is_unreg

        anchor_cell = (
            f"\\cellcolor{{blue!10}} {_anchor_label(int(row['anchor_maps']))}"
            if is_unreg
            else f"\\cellcolor[HTML]{{FDCFF1}} {_anchor_label(int(row['anchor_maps']))}"
        )
        cells = [_fmt_maps(int(row["sp_maps"])), anchor_cell]

        # SP: score, collision rate, offroad
        cells.append(
            _fmt_score(
                row.get("sp_score_mean", np.nan),
                row.get("sp_score_sem", np.nan),
                col="sp_score_mean",
                is_best=_is_best("sp_score_mean", row.get("sp_score_mean", np.nan)),
            )
        )
        cells.append(
            _fmt_coll(
                row.get("sp_collision_rate_mean", np.nan),
                row.get("sp_collision_rate_sem", np.nan),
                col="sp_collision_rate_mean",
                is_best=_is_best("sp_collision_rate_mean", row.get("sp_collision_rate_mean", np.nan)),
            )
        )
        if has_offroad:
            cells.append(
                _fmt_coll(
                    row.get("sp_offroad_rate_mean", np.nan),
                    row.get("sp_offroad_rate_sem", np.nan),
                    col="sp_offroad_rate_mean",
                    is_best=_is_best("sp_offroad_rate_mean", row.get("sp_offroad_rate_mean", np.nan)),
                )
            )

        # HR: score, collision, at-fault, rear, route progress, lateral L2
        cells.append(
            _fmt_score(
                row.get("hr_score_mean", np.nan),
                row.get("hr_score_sem", np.nan),
                col="hr_score_mean",
                is_best=_is_best("hr_score_mean", row.get("hr_score_mean", np.nan)),
            )
        )
        cells.append(
            _fmt_coll(
                row.get("hr_collision_rate_mean", np.nan),
                row.get("hr_collision_rate_sem", np.nan),
                col="hr_collision_rate_mean",
                is_best=_is_best("hr_collision_rate_mean", row.get("hr_collision_rate_mean", np.nan)),
            )
        )
        if has_at_fault:
            cells.append(
                _fmt_coll(
                    row.get("hr_at_fault_collision_rate_mean", np.nan),
                    row.get("hr_at_fault_collision_rate_sem", np.nan),
                    col="hr_at_fault_collision_rate_mean",
                    is_best=_is_best(
                        "hr_at_fault_collision_rate_mean", row.get("hr_at_fault_collision_rate_mean", np.nan)
                    ),
                )
            )
        if has_rear:
            cells.append(
                _fmt_coll(
                    row.get("hr_rear_collision_rate_mean", np.nan),
                    row.get("hr_rear_collision_rate_sem", np.nan),
                    col="hr_rear_collision_rate_mean",
                    is_best=_is_best("hr_rear_collision_rate_mean", row.get("hr_rear_collision_rate_mean", np.nan)),
                )
            )
        if has_route_prog:
            cells.append(
                _fmt_score(
                    row.get("hr_route_progress_mean", np.nan),
                    row.get("hr_route_progress_sem", np.nan),
                    col="hr_route_progress_mean",
                    is_best=_is_best("hr_route_progress_mean", row.get("hr_route_progress_mean", np.nan)),
                )
            )
        if has_lateral:
            cells.append(
                _fmt_coll(
                    row.get("hr_lateral_error_avg_mean", np.nan),
                    row.get("hr_lateral_error_avg_sem", np.nan),
                    col="hr_lateral_error_avg_mean",
                    is_best=_is_best("hr_lateral_error_avg_mean", row.get("hr_lateral_error_avg_mean", np.nan)),
                    as_pct=False,
                    decimals=2,
                )
            )

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


# ---------------------------------------------------------------------------
# Anchor evaluation plot
# ---------------------------------------------------------------------------


def plot_anchor_eval(anchor_df, save_path="results/figures/eval_anchor.pdf"):
    """Three-subplot summary figure for BC anchor evaluation.

    All subplots share the same x-axis: hours of human driving data used to
    train the anchor (num_maps_trained via _maps_to_human_hours).

    Subplot 0 — Open-loop: validation accuracy (one point per checkpoint).
    Subplot 1 — Closed-loop: route progress, self-play vs human-replay.
    Subplot 2 — Closed-loop: score, self-play vs human-replay.
    """
    if anchor_df is None or anchor_df.empty:
        print("  No anchor eval data — skipping plot_anchor_eval.")
        return None

    MODE_STYLES = {
        "cl_selfplay": dict(color="black", marker="o", label="Self-play"),
        "cl_humanreplay": dict(color="green", marker="s", label="Human-replay (control SDC only)"),
    }

    # Open-loop: one point per checkpoint
    ol_df = (
        anchor_df[["checkpoint", "num_maps_trained", "ol_val_accuracy", "ol_val_loss"]]
        .drop_duplicates(subset="checkpoint")
        .copy()
    )
    ol_df["human_hours"] = ol_df["num_maps_trained"].apply(_maps_to_human_hours)
    ol_df = ol_df.sort_values("human_hours")

    # Closed-loop: mean ± SEM per (checkpoint, mode)
    cl_df = (
        anchor_df[anchor_df["mode"].isin(["cl_selfplay", "cl_humanreplay"])]
        .groupby(["checkpoint", "num_maps_trained", "mode"])[["route_progress", "score"]]
        .agg(["mean", "sem"])
        .reset_index()
    )
    cl_df.columns = [
        "checkpoint",
        "num_maps_trained",
        "mode",
        "route_progress_mean",
        "route_progress_sem",
        "score_mean",
        "score_sem",
    ]
    cl_df["human_hours"] = cl_df["num_maps_trained"].apply(_maps_to_human_hours)
    cl_df = cl_df.sort_values("human_hours")

    _set_style(len(MODE_STYLES))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    # ── Subplot 0: open-loop validation accuracy (%) ─────────────────────────
    ax = axes[0]
    ax.plot(
        ol_df["human_hours"],
        ol_df["ol_val_accuracy"] * 100,
        color="#1f77b4",
        marker="D",
        linewidth=1.5,
        markersize=8,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Human driving demonstrations (hours)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Open-loop: validation accuracy")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    # ── Subplot 1: open-loop validation loss ──────────────────────────────────
    ax = axes[1]
    ax.plot(
        ol_df["human_hours"],
        ol_df["ol_val_loss"],
        color="#d62728",
        marker="o",
        linewidth=1.5,
        markersize=8,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Human driving demonstrations (hours)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Open-loop: validation loss")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    # ── Subplots 2 & 3: closed-loop route progress and score ──────────────────
    cl_specs = [
        ("route_progress_mean", "route_progress_sem", "Route progress", "Closed-loop: route progress"),
        ("score_mean", "score_sem", "Score", "Closed-loop: score"),
    ]
    for ax, (y_mean, y_sem, ylabel, title) in zip(axes[2:], cl_specs):
        for mode, style in MODE_STYLES.items():
            grp = cl_df[cl_df["mode"] == mode].sort_values("human_hours")
            if grp.empty:
                continue
            ax.errorbar(
                grp["human_hours"],
                grp[y_mean],
                yerr=grp[y_sem],
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=1.5,
                capsize=3,
                markersize=8,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Human driving demonstrations (hours)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def generate_anchor_latex_table(anchor_df, save_path="results/figures/anchor_eval_table.tex"):
    """LaTeX table for BC anchor evaluation results.

    Rows:    one per checkpoint (num_maps_trained), sorted ascending.
    Columns: Human data (hours) | OL val accuracy (%) | OL val loss
             | CL self-play route progress | CL self-play score
             | CL human-replay route progress | CL human-replay score

    Uses the same green/red cellcolor heatmap convention as
    generate_scaling_latex_table: green = higher is better (accuracy, progress,
    score), red = lower is better (loss).

    Required LaTeX packages:
      \\usepackage{booktabs}
      \\usepackage[table]{xcolor}
      \\usepackage{graphicx}
      \\usepackage{bm}
    """
    if anchor_df is None or anchor_df.empty:
        print("  No anchor eval data -- skipping generate_anchor_latex_table.")
        return None

    # Open-loop: one row per checkpoint
    ol = (
        anchor_df[["checkpoint", "num_maps_trained", "ol_val_accuracy", "ol_val_loss"]]
        .drop_duplicates(subset="checkpoint")
        .copy()
    )
    ol["human_hours"] = ol["num_maps_trained"].apply(_maps_to_human_hours)

    # Closed-loop: mean +/- SEM per (checkpoint, mode)
    cl = (
        anchor_df[anchor_df["mode"].isin(["cl_selfplay", "cl_humanreplay"])]
        .groupby(["checkpoint", "num_maps_trained", "mode"])[["route_progress", "score"]]
        .agg(["mean", "sem"])
        .reset_index()
    )
    cl.columns = [
        "checkpoint",
        "num_maps_trained",
        "mode",
        "route_progress_mean",
        "route_progress_sem",
        "score_mean",
        "score_sem",
    ]

    sp = cl[cl["mode"] == "cl_selfplay"].drop(columns="mode").copy()
    sp = sp.rename(columns={c: f"sp_{c}" for c in sp.columns if c not in ("checkpoint", "num_maps_trained")})
    hr = cl[cl["mode"] == "cl_humanreplay"].drop(columns="mode").copy()
    hr = hr.rename(columns={c: f"hr_{c}" for c in hr.columns if c not in ("checkpoint", "num_maps_trained")})

    merged = ol.merge(sp, on=["checkpoint", "num_maps_trained"], how="left")
    merged = merged.merge(hr, on=["checkpoint", "num_maps_trained"], how="left")
    merged = merged.sort_values("num_maps_trained").reset_index(drop=True)

    # Colour intensity helpers (mirrors generate_scaling_latex_table)
    def _intensity(val, col_vals, higher_is_better=True):
        """Map value to xcolor intensity 5-50.

        For green (higher_is_better=True):  high value -> high intensity (dark green).
        For red   (higher_is_better=False): low value  -> low intensity  (light red),
                                            high value -> high intensity (dark red).
        """
        finite = col_vals.dropna()
        if finite.empty or np.isnan(val):
            return 0
        vmin, vmax = finite.min(), finite.max()
        if vmax == vmin:
            return 25
        t = (val - vmin) / (vmax - vmin)
        # t is already 0 for min, 1 for max.
        # Green: darker = higher = better  -> use t as-is.
        # Red:   lighter = lower  = better -> use t as-is too (low val -> light red).
        return int(5 + t * 45)

    def _fmt_green(val, col, sem=None, is_best=False, scale=1.0, decimals=3):
        if np.isnan(val):
            return "---"
        intensity = _intensity(val, merged[col], higher_is_better=True)
        v = val * scale
        s = sem * scale if (sem is not None and not np.isnan(sem)) else None
        fmt = f".{decimals}f"
        if s:
            text = f"$\\bm{{{v:{fmt}} \\pm {s:{fmt}}}}$" if is_best else f"${v:{fmt}} \\pm {s:{fmt}}$"
        else:
            text = f"\\textbf{{{v:{fmt}}}}" if is_best else f"{v:{fmt}}"
        return f"\\cellcolor{{green!{intensity}}} {text}"

    def _fmt_red(val, col, sem=None, is_best=False, decimals=3):
        if np.isnan(val):
            return "---"
        intensity = _intensity(val, merged[col], higher_is_better=False)
        fmt = f".{decimals}f"
        if sem is not None and not np.isnan(sem):
            text = f"$\\bm{{{val:{fmt}} \\pm {sem:{fmt}}}}$" if is_best else f"${val:{fmt}} \\pm {sem:{fmt}}$"
        else:
            text = f"\\textbf{{{val:{fmt}}}}" if is_best else f"{val:{fmt}}"
        return f"\\cellcolor{{red!{intensity}}} {text}"

    best = {
        "ol_val_accuracy": merged["ol_val_accuracy"].max(),
        "ol_val_loss": merged["ol_val_loss"].min(),
        "sp_route_progress_mean": merged["sp_route_progress_mean"].max(),
        "sp_score_mean": merged["sp_score_mean"].max(),
        "hr_route_progress_mean": merged["hr_route_progress_mean"].max(),
        "hr_score_mean": merged["hr_score_mean"].max(),
    }

    def _is_best(col, val):
        return not np.isnan(val) and np.isclose(val, best[col])

    # Build LaTeX
    col_spec = "r|rr|rr|rr"
    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor}, \usepackage{graphicx}, \usepackage{bm}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{BC anchor evaluation. Open-loop metrics on the held-out validation set; closed-loop metrics averaged over validation scenes.}",
        r"\label{tab:anchor_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r" & \multicolumn{2}{c|}{Open-loop} & \multicolumn{2}{c|}{Closed-loop self-play} & \multicolumn{2}{c}{Closed-loop human-replay (SDC only)} \\",
        r"Human data (h) & Acc. (\%) & Loss & Route prog. & Score & Route prog. & Score \\",
        r"\midrule",
    ]

    for _, row in merged.iterrows():
        cells = [f"{row['human_hours']:.1f}"]
        cells.append(
            _fmt_green(
                row["ol_val_accuracy"],
                "ol_val_accuracy",
                scale=100,
                decimals=1,
                is_best=_is_best("ol_val_accuracy", row["ol_val_accuracy"]),
            )
        )
        cells.append(
            _fmt_red(
                row["ol_val_loss"],
                "ol_val_loss",
                decimals=3,
                is_best=_is_best("ol_val_loss", row["ol_val_loss"]),
            )
        )
        for col_mean, col_sem, key in [
            ("sp_route_progress_mean", "sp_route_progress_sem", "sp_route_progress_mean"),
            ("sp_score_mean", "sp_score_sem", "sp_score_mean"),
            ("hr_route_progress_mean", "hr_route_progress_sem", "hr_route_progress_mean"),
            ("hr_score_mean", "hr_score_sem", "hr_score_mean"),
        ]:
            cells.append(
                _fmt_green(
                    row.get(col_mean, np.nan),
                    key,
                    sem=row.get(col_sem, np.nan),
                    decimals=3,
                    is_best=_is_best(key, row.get(col_mean, np.nan)),
                )
            )
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    latex_str = "\n".join(lines)

    _ensure_dir(save_path)
    with open(save_path, "w") as f:
        f.write(latex_str)
    print(f"  LaTeX table written to {save_path}")
    return latex_str


def plot_data_requirements(df, save_path="results/figures/eval_data_requirements.pdf"):
    """Master summary figure: how much data is needed for human-compatible policies?

    3 subplots, all using scaling modes.
    x-axis: self-play training maps (sp_maps), log scale
    Color and shape both encode anchor maps (unreg = black).
    All y-axes reported in percentage [%].

    Subplot 0: SP collision rate [%]
    Subplot 1: HR at-fault collision rate [%]
    Subplot 2: Delta at-fault = SP - HR [%] (positive = SP better = ZSC gap exists)
               Green striped line at 0 = "no ZSC gap"
    """
    scaling_modes = ["scaling_sp_val", "scaling_hr_interactive"]
    df = df[df["mode"].isin(scaling_modes)].copy()
    if df.empty:
        print("  No scaling data — skipping plot_data_requirements.")
        return None

    df["anchor_maps"] = df["anchor_maps"].fillna(0).astype(int)
    if "dynamics" not in df.columns:
        df["dynamics"] = "delta"

    hr = df[df["mode"] == "scaling_hr_interactive"]
    sp = df[df["mode"] == "scaling_sp_val"]

    hr_agg = (
        hr.groupby(["sp_maps", "anchor_maps"])[["at_fault_collision_rate", "collision_rate"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "at_fault_collision_rate": "hr_atfault",
                "collision_rate": "hr_collision_rate",
            }
        )
    )
    sp_agg = (
        sp.groupby(["sp_maps", "anchor_maps"])[["at_fault_collision_rate", "collision_rate"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "at_fault_collision_rate": "sp_atfault",
                "collision_rate": "sp_collision_rate",
            }
        )
    )
    agg = hr_agg.merge(sp_agg, on=["sp_maps", "anchor_maps"], how="left")

    # Convert all metrics to percentage
    agg["hr_atfault"] = agg["hr_atfault"] * 100
    agg["sp_atfault"] = agg["sp_atfault"] * 100
    agg["hr_collision_rate"] = agg["hr_collision_rate"] * 100
    agg["sp_collision_rate"] = agg["sp_collision_rate"] * 100
    # Recompute delta after scaling to percentages
    agg["delta_atfault"] = agg["sp_atfault"] - agg["hr_atfault"]

    anchor_vals = sorted(agg["anchor_maps"].unique())
    color_map = _reg_unreg_colors(anchor_vals)

    markers = ["^", "s", "o", "D", "P", "X", "v", "*"]
    marker_map = {a: markers[i % len(markers)] for i, a in enumerate(anchor_vals)}

    def _anchor_label(a):
        return "no anchor (unreg)" if a == 0 else f"{_maps_to_human_time(a)} anchor"

    _set_style(len(anchor_vals))

    subplot_specs = [
        ("sp_collision_rate", "Self-play collision rate [%]"),
        ("hr_collision_rate", "Human-replay collision rate [%]"),
        ("hr_atfault", "Human-replay at-fault collision rate [%]"),
        ("delta_atfault", "\u0394 at-fault (SP \u2212 HR) [%]"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, (y_col, ylabel) in zip(axes, subplot_specs):
        for anchor in anchor_vals:
            subset = agg[agg["anchor_maps"] == anchor].sort_values("sp_maps")
            if subset.empty:
                continue
            ax.plot(
                subset["sp_maps"],
                subset[y_col],
                color=color_map[anchor],
                marker=marker_map[anchor],
                markersize=8,
                linewidth=1.2,
                linestyle="-",
                zorder=3,
                label=_anchor_label(anchor),
            )

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else str(int(x)))
        )
        ax.set_xlabel("Self-play training maps")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        if y_col == "delta_atfault":
            ax.axhline(0, color="#2e7cf8", linestyle="--", linewidth=2.0, alpha=0.9, zorder=2)
            ax.text(
                min(agg["sp_maps"]) * 1.1,
                0.2,
                "No ZSC gap",
                fontsize=14,
                color="#2e7cf8",
                ha="left",
                va="bottom",
            )
        else:
            ax.set_ylim(bottom=0)

        ax.legend(fontsize=8, loc="best", framealpha=1.0, facecolor="white", edgecolor="lightgray")
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_compatibility_tradeoff_bar(df, save_path="results/figures/eval_compatibility_tradeoff_bar.pdf"):
    """Bar chart comparing two checkpoints across four HR metrics.

    4 subplots, one per metric. Two bars per subplot: unreg vs reg.
    Raw values, no normalization.
    """
    CHECKPOINTS_OF_INTEREST = {
        "models/scaling_cpts/unreg_classic_50k_maps.pt": "unregularized",
        "models/scaling_cpts/reg_delta_10k_maps_anchor_10k_maps.pt": "regularized",
    }

    df = df[df["mode"] == "scaling_hr_interactive"].copy()
    df = df[df["checkpoint"].isin(CHECKPOINTS_OF_INTEREST)].copy()
    if df.empty:
        print("  No data for checkpoints of interest — skipping plot_compatibility_tradeoff_bar.")
        return None

    required_cols = [
        "collision_rate",
        "at_fault_collision_rate",
        "rear_collision_rate",
        "route_progress",
        "lateral_error_avg",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  Missing columns {missing} — skipping plot_compatibility_tradeoff_bar.")
        return None

    agg = df.groupby("checkpoint")[required_cols].agg(["mean", "sem"]).reset_index()
    agg.columns = ["checkpoint"] + [f"{m}_{s}" for m in required_cols for s in ["mean", "sem"]]
    agg["label"] = agg["checkpoint"].map(CHECKPOINTS_OF_INTEREST)

    # unreg first (black), reg second (pink)
    agg["is_reg"] = ~agg["checkpoint"].str.contains("unreg")
    agg = agg.sort_values("is_reg").reset_index(drop=True)
    colors = ["black" if not r else "#d62728" for r in agg["is_reg"]]

    subplot_specs = [
        ("collision_rate", "HR collision rate [%]", True),
        ("at_fault_collision_rate", "HR at-fault collision rate [%]", True),
        ("rear_collision_rate", "HR rear collision rate [%]", True),
        ("route_progress", "HR route progress [%]", True),
        ("lateral_error_avg", "HR lateral L2 distance", False),
    ]

    _set_style(2)
    fig = plt.figure(figsize=(20, 4))
    gs = fig.add_gridspec(1, 5)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4])
    axes = [ax0, ax1, ax2, ax3, ax4]

    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)

    for ax, (col, ylabel, as_pct) in zip(axes, subplot_specs):
        means = agg[f"{col}_mean"].values * (100 if as_pct else 1)
        sems = agg[f"{col}_sem"].values * (100 if as_pct else 1)
        labels = agg["label"].values
        x = np.arange(len(labels))

        for i, (mean, sem, color) in enumerate(zip(means, sems, colors)):
            ax.bar(
                x[i],
                mean,
                yerr=sem,
                color=color,
                alpha=0.8,
                width=0.5,
                capsize=4,
                error_kw=dict(lw=1.2),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        sns.despine(ax=ax)

    if ax in (axes[0], axes[1], axes[2]):
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.tick_params(axis="y", which="minor", length=3, color="gray")

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Human-replay val vs interactive comparison table
# ---------------------------------------------------------------------------


def generate_hr_comparison_latex_table(df, save_path="results/figures/eval_hr_comparison_table.tex"):
    """LaTeX table comparing human-replay performance on the full validation set
    vs. the interactive validation subset, for all scaling checkpoints.

    Rows: one per (sp_maps, anchor_maps) combination, unreg rows first.
    Column groups:
      - HR validation (scaling_hr_val):       score, collision, at-fault, rear, route progress, lateral L2
      - HR interactive (scaling_hr_interactive): same metrics

    Formatting identical to generate_scaling_latex_table: green cellcolor for
    higher-is-better metrics, red for lower-is-better, intensity normalised
    per column, best value in each column bold.

    Required LaTeX packages:
      \\usepackage{booktabs}
      \\usepackage[table]{xcolor}
      \\usepackage{graphicx}
      \\usepackage{makecell}
      \\usepackage{bm}
    """
    hr_modes = ["scaling_hr_val", "scaling_hr_interactive"]
    scaling_df = df[df["mode"].isin(hr_modes)].copy()
    if scaling_df.empty:
        print("  No scaling HR data found — skipping generate_hr_comparison_latex_table.")
        return None

    scaling_df["anchor_maps"] = scaling_df["anchor_maps"].fillna(0).astype(int)

    hr_metrics = [
        "score",
        "collision_rate",
        "at_fault_collision_rate",
        "rear_collision_rate",
        "route_progress",
        "lateral_error_avg",
    ]
    available_metrics = [m for m in hr_metrics if m in scaling_df.columns]

    agg = scaling_df.groupby(["sp_maps", "anchor_maps", "mode"])[available_metrics].agg(["mean", "sem"]).reset_index()
    flat_cols = ["sp_maps", "anchor_maps", "mode"]
    for m in available_metrics:
        flat_cols.extend([f"{m}_mean", f"{m}_sem"])
    agg.columns = flat_cols

    val_df = agg[agg["mode"] == "scaling_hr_val"].drop(columns=["mode"]).copy()
    int_df = agg[agg["mode"] == "scaling_hr_interactive"].drop(columns=["mode"]).copy()

    val_df = val_df.rename(columns={c: f"val_{c}" for c in val_df.columns if c not in ("sp_maps", "anchor_maps")})
    int_df = int_df.rename(columns={c: f"int_{c}" for c in int_df.columns if c not in ("sp_maps", "anchor_maps")})

    merged = val_df.merge(int_df, on=["sp_maps", "anchor_maps"], how="outer")

    # unreg rows first, then reg, each sorted by sp_maps
    unreg = merged[merged["anchor_maps"] == 0].sort_values("sp_maps")
    reg = merged[merged["anchor_maps"] != 0].sort_values(["sp_maps", "anchor_maps"])
    merged = pd.concat([unreg, reg]).reset_index(drop=True)

    # ── Colour helpers (identical logic to generate_scaling_latex_table) ────
    has_at_fault = "at_fault_collision_rate" in available_metrics
    has_rear = "rear_collision_rate" in available_metrics
    has_route_prog = "route_progress" in available_metrics
    has_lateral = "lateral_error_avg" in available_metrics

    # higher-is-better -> green; lower-is-better -> red
    green_cols = []
    red_cols = []
    for prefix in ("val", "int"):
        green_cols.append(f"{prefix}_score_mean")
        if has_route_prog:
            green_cols.append(f"{prefix}_route_progress_mean")
        red_cols.append(f"{prefix}_collision_rate_mean")
        if has_at_fault:
            red_cols.append(f"{prefix}_at_fault_collision_rate_mean")
        if has_rear:
            red_cols.append(f"{prefix}_rear_collision_rate_mean")
        if has_lateral:
            red_cols.append(f"{prefix}_lateral_error_avg_mean")

    existing_green = [c for c in green_cols if c in merged.columns]
    existing_red = [c for c in red_cols if c in merged.columns]

    col_min, col_max = {}, {}
    for c in existing_green + existing_red:
        vals = merged[c].dropna()
        col_min[c] = vals.min() if not vals.empty else 0
        col_max[c] = vals.max() if not vals.empty else 1

    def _intensity(val, col):
        if np.isnan(val):
            return 0
        vmin, vmax = col_min.get(col, 0), col_max.get(col, 1)
        if vmax == vmin:
            return 25
        return int(5 + (val - vmin) / (vmax - vmin) * 45)

    def _fmt_green(mean, sem, col, is_best=False):
        if np.isnan(mean):
            return "---"
        intensity = _intensity(mean, col)
        if not (np.isnan(sem) or sem == 0):
            text = f"$\\bm{{{mean:.3f} \\pm {sem:.3f}}}$" if is_best else f"${mean:.3f} \\pm {sem:.3f}$"
        else:
            text = f"\\textbf{{{mean:.3f}}}" if is_best else f"{mean:.3f}"
        return f"\\cellcolor{{green!{intensity}}} {text}"

    def _fmt_red(mean, sem, col, is_best=False, as_pct=True, decimals=1):
        if np.isnan(mean):
            return "---"
        intensity = _intensity(mean, col)
        m_val = mean * 100 if as_pct else mean
        s_val = sem * 100 if as_pct else sem
        fmt = f".{decimals}f"
        if not (np.isnan(s_val) or s_val == 0):
            text = f"$\\bm{{{m_val:{fmt}} \\pm {s_val:{fmt}}}}$" if is_best else f"${m_val:{fmt}} \\pm {s_val:{fmt}}$"
        else:
            text = f"\\textbf{{{m_val:{fmt}}}}" if is_best else f"{m_val:{fmt}}"
        return f"\\cellcolor{{red!{intensity}}} {text}"

    # best: max for green cols, min for red cols
    best = {}
    for c in existing_green:
        best[c] = merged[c].max()
    for c in existing_red:
        best[c] = merged[c].min()

    def _is_best(col, val):
        return col in best and not np.isnan(val) and np.isclose(val, best[col])

    def _anchor_label(anchor_maps):
        return "0 (unreg.)" if anchor_maps == 0 else _maps_to_human_time(anchor_maps)

    # ── Column spec ─────────────────────────────────────────────────────────
    n_hr_cols = (
        2  # score, collision
        + int(has_at_fault)
        + int(has_rear)
        + int(has_route_prog)
        + int(has_lateral)
    )
    col_spec = "rr" + "|" + "r" * n_hr_cols + "|" + "r" * n_hr_cols

    # ── Build LaTeX ─────────────────────────────────────────────────────────
    lines = []
    lines.append(
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor}, "
        r"\usepackage{graphicx}, \usepackage{makecell}, \usepackage{bm}"
    )
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Human-replay performance on the randomly sampled validation set "
        r"vs.\ the interactive validation subset for all scaling checkpoints. "
        r"Metrics are averaged over all scenes in each split.}"
    )
    lines.append(r"\label{tab:hr_comparison_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    val_header = r"\multicolumn{" + str(n_hr_cols) + r"}{c|}{HR — validation (random)}"
    int_header = r"\multicolumn{" + str(n_hr_cols) + r"}{c}{HR — interactive}"
    lines.append(r" & & " + val_header + " & " + int_header + r" \\")

    def _metric_headers():
        hdrs = ["Score $\\uparrow$", "Coll. (\\%) $\\downarrow$"]
        if has_at_fault:
            hdrs.append("At-fault (\\%) $\\downarrow$")
        if has_rear:
            hdrs.append("Rear coll. (\\%) $\\downarrow$")
        if has_route_prog:
            hdrs.append("Route prog. $\\uparrow$")
        if has_lateral:
            hdrs.append("Lateral L2 $\\downarrow$")
        return hdrs

    metric_headers = _metric_headers()
    header2 = (
        "\\makecell{Self-play maps \\\\ (metadata)} & "
        "\\makecell{Anchor data \\\\ (human demonstrations)} & "
        + " & ".join(metric_headers)
        + " & "
        + " & ".join(metric_headers)
        + r" \\"
    )
    lines.append(header2)
    lines.append(r"\midrule")

    prev_was_unreg = None
    for _, row in merged.iterrows():
        is_unreg = int(row["anchor_maps"]) == 0
        if prev_was_unreg is True and not is_unreg:
            lines.append(r"\midrule")
        prev_was_unreg = is_unreg

        anchor_cell = (
            f"\\cellcolor{{blue!10}} {_anchor_label(int(row['anchor_maps']))}"
            if is_unreg
            else f"\\cellcolor[HTML]{{FDCFF1}} {_anchor_label(int(row['anchor_maps']))}"
        )
        cells = [_fmt_maps(int(row["sp_maps"])), anchor_cell]

        for prefix in ("val", "int"):
            # score (green)
            s_col = f"{prefix}_score_mean"
            cells.append(
                _fmt_green(
                    row.get(s_col, np.nan),
                    row.get(f"{prefix}_score_sem", np.nan),
                    col=s_col,
                    is_best=_is_best(s_col, row.get(s_col, np.nan)),
                )
            )
            # collision rate (red, %)
            c_col = f"{prefix}_collision_rate_mean"
            cells.append(
                _fmt_red(
                    row.get(c_col, np.nan),
                    row.get(f"{prefix}_collision_rate_sem", np.nan),
                    col=c_col,
                    is_best=_is_best(c_col, row.get(c_col, np.nan)),
                )
            )
            # at-fault (red, %)
            if has_at_fault:
                af_col = f"{prefix}_at_fault_collision_rate_mean"
                cells.append(
                    _fmt_red(
                        row.get(af_col, np.nan),
                        row.get(f"{prefix}_at_fault_collision_rate_sem", np.nan),
                        col=af_col,
                        is_best=_is_best(af_col, row.get(af_col, np.nan)),
                    )
                )
            # rear collision (red, %)
            if has_rear:
                r_col = f"{prefix}_rear_collision_rate_mean"
                cells.append(
                    _fmt_red(
                        row.get(r_col, np.nan),
                        row.get(f"{prefix}_rear_collision_rate_sem", np.nan),
                        col=r_col,
                        is_best=_is_best(r_col, row.get(r_col, np.nan)),
                    )
                )
            # route progress (green)
            if has_route_prog:
                rp_col = f"{prefix}_route_progress_mean"
                cells.append(
                    _fmt_green(
                        row.get(rp_col, np.nan),
                        row.get(f"{prefix}_route_progress_sem", np.nan),
                        col=rp_col,
                        is_best=_is_best(rp_col, row.get(rp_col, np.nan)),
                    )
                )
            # lateral L2 (red, raw value, 2 decimals)
            if has_lateral:
                l_col = f"{prefix}_lateral_error_avg_mean"
                cells.append(
                    _fmt_red(
                        row.get(l_col, np.nan),
                        row.get(f"{prefix}_lateral_error_avg_sem", np.nan),
                        col=l_col,
                        is_best=_is_best(l_col, row.get(l_col, np.nan)),
                        as_pct=False,
                        decimals=2,
                    )
                )

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


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------


def make_all_figures(df=None, wosac_df=None, anchor_df=None):
    """Generate all evaluation figures."""
    print("\nGenerating figures...")
    if df is not None and not df.empty:
        # plot_scores(df)
        # print("  Saved eval_scores.pdf")
        # plot_scaling_barplot(df)
        # print("  Saved eval_scaling_barplot.pdf")
        # plot_scaling_scatter(df)
        # print("  Saved eval_scaling_scatter.pdf")
        plot_data_requirements(df)
        print("  Saved eval_data_requirements.pdf")
        generate_scaling_latex_table(df)
        generate_hr_comparison_latex_table(df)
        plot_compatibility_tradeoff_bar(df)
        print("  Saved eval_compatibility_tradeoff_bar.pdf")
    plot_scaling_wosac(wosac_df)
    print("  Saved eval_scaling_wosac.pdf")
    if anchor_df is not None and not anchor_df.empty:
        plot_anchor_eval(anchor_df)
        print("  Saved eval_anchor.pdf")
        generate_anchor_latex_table(anchor_df)
        print("  Saved anchor_eval_table.tex")


if __name__ == "__main__":
    import os

    EVAL_CSV = "results/checkpoint_eval_results.csv"
    WOSAC_CSV = "results/checkpoint_wosac_results.csv"
    ANCHOR_CSV = "results/anchor_eval.csv"

    df = None
    wosac_df = None
    anchor_df = None

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

    if os.path.exists(ANCHOR_CSV):
        anchor_df = pd.read_csv(ANCHOR_CSV)
        print(f"Loaded {ANCHOR_CSV} ({len(anchor_df)} rows)")
    else:
        print(f"{ANCHOR_CSV} not found — skipping anchor eval figure.")

    make_all_figures(df, wosac_df, anchor_df)
