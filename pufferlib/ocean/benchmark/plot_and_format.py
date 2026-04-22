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

# ── Colours ─────────────────────────────────────────────────────────────
COLOR_SMART = "#FFA8CC"
COLOR_SMART_EDGE = "#C14B8A"
COLOR_OURS = "#CCCCFF"
COLOR_OURS_EDGE = "#6B3FA0"
COLOR_SELFPLAY = "#4A7FD4"


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
    """Convert map count to human-readable time: minutes = (maps * 9 seconds) / 60."""
    minutes = (maps * 9) / 60
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

    Each map is a 9-second scenario with 1 controlled agent on average:
        hours = (maps × 9s × 1) / 3600
    """
    return (maps * 9) / 3600


# ---------------------------------------------------------------------------
# Shared colour convention
# ---------------------------------------------------------------------------

UNREG_COLOR = "k"
REG_COLORS = ["#ff7f0e", "#d62728", "#e377c2", "#9467bd", "#a8174a"]


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
    # ax.set_xscale("log")
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
        # ax.set_xscale("log")
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


# ---------------------------------------------------------------------------
# SMART baseline loader (used by plot_human_data_requirements)
# ---------------------------------------------------------------------------

# Checkpoints that do NOT follow the smart_mini_vehicle_only_<N> naming and
# need an explicit num_maps. smart_tiny_clsft_e9 was trained on the full
# Waymo dataset (~500k maps → "52 days" on the plot's x-axis).
_SMART_CHECKPOINT_NUM_MAPS = {
    "smart_tiny_clsft_e9": 500_000,
}

# Pre-BC epoch-31 snapshot: explicitly excluded from the scaling study.
_SMART_EXCLUDED_CHECKPOINTS = {"smart_tiny_pre_bc_e31"}


def _smart_ckpt_to_num_maps(ckpt: str) -> float:
    """Map a SMART checkpoint name to its number of training maps."""
    if ckpt in _SMART_CHECKPOINT_NUM_MAPS:
        return _SMART_CHECKPOINT_NUM_MAPS[ckpt]
    m = re.match(r"smart_mini_vehicle_only_(\d+)$", ckpt)
    if m:
        return int(m.group(1))
    return np.nan


def _load_smart_baseline(csv_path: str = "results/smart_baseline_res.csv") -> pd.DataFrame:
    """Load the SMART baseline CSV and pivot into per-num_maps rows.

    Returns a DataFrame with columns:
        num_maps, minutes,
        hr_atfault_pct, hr_coll_pct, hr_progress_pct, sp_coll_pct

    Missing metrics are left as NaN so downstream `dropna(subset=[col])` skips
    them cleanly. Returns an empty (but correctly-typed) DataFrame if the CSV
    isn't present, which keeps the plotting code behaving as if no SMART data
    were available.
    """
    cols = [
        "num_maps",
        "minutes",
        "hr_atfault_pct",
        "hr_coll_pct",
        "hr_progress_pct",
        "sp_coll_pct",
        "sp_progress_pct",
    ]
    if not os.path.exists(csv_path):
        print(f"  {csv_path} not found — SMART baseline will be omitted from the plot.")
        return pd.DataFrame(columns=cols)

    raw = pd.read_csv(csv_path)

    # Drop explicitly-excluded checkpoints (e.g. the pre-BC snapshot).
    raw = raw[~raw["checkpoint"].isin(_SMART_EXCLUDED_CHECKPOINTS)]

    # Map each remaining checkpoint to a num_maps value; drop anything we
    # don't recognise rather than silently plotting it at NaN.
    raw["num_maps"] = raw["checkpoint"].apply(_smart_ckpt_to_num_maps)
    unknown = raw[raw["num_maps"].isna()]["checkpoint"].unique()
    if len(unknown) > 0:
        print(f"  Warning: SMART checkpoints with no num_maps mapping, skipping: {list(unknown)}")
    raw = raw.dropna(subset=["num_maps"]).copy()
    raw["num_maps"] = raw["num_maps"].astype(int)

    # Pivot: one row per num_maps with HR + SP metrics side by side.
    hr = (
        raw[raw["mode"] == "scaling_hr_val"]
        .set_index("num_maps")[["at_fault_collision_rate", "collision_rate", "route_progress"]]
        .rename(
            columns={
                "at_fault_collision_rate": "hr_atfault_pct",
                "collision_rate": "hr_coll_pct",
                "route_progress": "hr_progress_pct",
            }
        )
    )
    sp = (
        raw[raw["mode"] == "scaling_sp_val"]
        .set_index("num_maps")[["collision_rate", "route_progress"]]
        .rename(
            columns={
                "collision_rate": "sp_coll_pct",
                "route_progress": "sp_progress_pct",
            }
        )
    )

    # Fractions → percentages (matches the *_mean_pct convention used
    # elsewhere in the plotting code).
    out = hr.join(sp, how="outer") * 100
    out = out.reset_index().sort_values("num_maps").reset_index(drop=True)
    out["minutes"] = out["num_maps"] * 9 / 60
    return out[cols]


def plot_human_data_requirements(
    df,
    save_path="results/figures/eval_human_data_requirements.pdf",
    save_path_gains="results/figures/eval_human_data_gains.pdf",
    smart_csv="results/smart_baseline_res.csv",
):
    """Human-data sweep at fixed 50k metadata maps.

    Saves two PDFs:
      - save_path:        1×4 line plots (HR at-fault / HR coll / SP coll / HR progress).
                          HR at-fault is linear; others log; route progress clipped to [50, 110].
      - save_path_gains:  1×4 categorical bar plots of reg-self-play's relative improvement
                          vs SMART at each matched human-data amount, expressed as a ratio.
                          Collision metrics: SMART / reg (lower is better). Route progress:
                          reg / SMART. Reference line at y=1 marks parity; bars above 1 mean
                          reg self-play wins. "52 days" is omitted from the x-axis.

    Returns (fig_lines, fig_gains).
    """
    # ── SMART baseline data ─────────────────────────────────────────────────
    SMART_DATA = _load_smart_baseline(smart_csv)

    # ── Filter to scaling modes and 50k metadata maps only ──────────────────
    scaling_modes = ["scaling_sp_val", "scaling_hr_val"]
    df = df[df["mode"].isin(scaling_modes)].copy()
    df = df[df["sp_maps"] == 50000]
    if df.empty:
        print("  No 50k sp_maps data — skipping plot_human_data_requirements.")
        return None

    df["anchor_maps"] = df["anchor_maps"].fillna(0).astype(int)

    # ── Aggregate per anchor_maps ───────────────────────────────────────────
    hr = df[df["mode"] == "scaling_hr_val"]
    hr_agg = (
        hr.groupby("anchor_maps")[["at_fault_collision_rate", "collision_rate", "route_progress"]]
        .agg(["mean", "sem"])
        .reset_index()
    )
    hr_agg.columns = [
        "anchor_maps",
        "hr_atfault_mean",
        "hr_atfault_sem",
        "hr_coll_mean",
        "hr_coll_sem",
        "hr_progress_mean",
        "hr_progress_sem",
    ]

    sp = df[df["mode"] == "scaling_sp_val"]
    sp_agg = sp.groupby("anchor_maps")[["collision_rate"]].agg(["mean", "sem"]).reset_index()
    sp_agg.columns = ["anchor_maps", "sp_coll_mean", "sp_coll_sem"]

    agg = hr_agg.merge(sp_agg, on="anchor_maps", how="outer")
    agg["anchor_minutes"] = agg["anchor_maps"] * 9 / 60

    for col in ("hr_atfault", "hr_coll", "hr_progress", "sp_coll"):
        agg[f"{col}_mean_pct"] = agg[f"{col}_mean"] * 100
        agg[f"{col}_sem_pct"] = agg[f"{col}_sem"] * 100

    unreg = agg[agg["anchor_maps"] == 0]
    reg = agg[agg["anchor_maps"] > 0].sort_values("anchor_minutes")

    # (y_mean, y_sem, ylabel, smart_col, metric_label, lower_is_better, top_yscale)
    subplot_specs = [
        (
            "hr_atfault_mean_pct",
            "hr_atfault_sem_pct",
            "Human-replay at-fault collision rate [%]",
            "hr_atfault_pct",
            "HR at-fault coll.",
            True,
            "linear",
        ),
        (
            "hr_coll_mean_pct",
            "hr_coll_sem_pct",
            "Human-replay collision rate [%]",
            "hr_coll_pct",
            "HR collision",
            True,
            "linear",
        ),
        (
            "sp_coll_mean_pct",
            "sp_coll_sem_pct",
            "Self-play collision rate [%]",
            "sp_coll_pct",
            "SP collision",
            True,
            "linear",
        ),
        (
            "hr_progress_mean_pct",
            "hr_progress_sem_pct",
            "Route progress [%]",
            "hr_progress_pct",
            "HR route progress",
            False,
            "linear",
        ),
    ]

    tick_positions = [10, 30, 180, 1800, 75000]
    tick_labels = ["10 min", "30 min", "3 hours", "30 hours", "52 days"]

    # Labels excluded from the gains figure's x-axis (line plots still show them).
    GAINS_EXCLUDED_LABELS = {"52 days"}

    def _minutes_to_label(m):
        for target, label in zip(tick_positions, tick_labels):
            if abs(m - target) / target < 0.02:
                return label
        if m < 60:
            return f"{m:.0f} min"
        if m < 1440:
            return f"{m / 60:.0f} hours"
        return f"{m / 1440:.0f} days"

    # Shared category order for the gains figure, driven by SMART's anchors.
    if not SMART_DATA.empty:
        category_order = [
            _minutes_to_label(m)
            for m in SMART_DATA.sort_values("minutes")["minutes"]
            if _minutes_to_label(m) not in GAINS_EXCLUDED_LABELS
        ]
    else:
        category_order = []

    green_palette = sns.color_palette("Greens_d", n_colors=max(len(category_order), 1))

    # ── FIGURE 1: line plots ────────────────────────────────────────────────
    _set_style(3)
    fig_lines, line_axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, (y_mean, y_sem, ylabel, smart_col, _, _, top_yscale) in zip(line_axes, subplot_specs):
        if not reg.empty:
            ax.errorbar(
                reg["anchor_minutes"],
                reg[y_mean],
                yerr=reg[y_sem],
                color=COLOR_OURS,
                marker="o",
                markersize=9,
                linewidth=2.0,
                capsize=3,
                markeredgecolor=COLOR_OURS_EDGE,
                markerfacecolor=COLOR_OURS,
                label="regularized self-play (ours)",
                zorder=4,
            )
        if not unreg.empty:
            ax.axhline(
                unreg[y_mean].iloc[0],
                color=COLOR_SELFPLAY,
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                label="best unregularized self-play",
                zorder=2,
            )
        smart_valid = SMART_DATA.dropna(subset=[smart_col]) if smart_col in SMART_DATA.columns else SMART_DATA.iloc[0:0]
        if not smart_valid.empty:
            ax.plot(
                smart_valid["minutes"],
                smart_valid[smart_col],
                color=COLOR_SMART,
                marker="o",
                markersize=9,
                linewidth=2.0,
                linestyle="-",
                markeredgecolor=COLOR_SMART_EDGE,
                markerfacecolor=COLOR_SMART,
                label="SMART-tiny-CLSFT",
                zorder=3,
            )

        ax.set_xscale("symlog", linthresh=60, linscale=1.2)
        ax.set_xticks(tick_positions, labels=tick_labels, rotation=35, ha="right")
        ax.minorticks_off()
        ax.set_yscale(top_yscale)
        if top_yscale == "linear":
            ax.set_ylim(bottom=0)
        if y_mean == "hr_progress_mean_pct":
            ax.set_ylim(50, 110)
        ax.set_xlabel("Human demonstration data")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="best", framealpha=1.0, facecolor="white", edgecolor="lightgray")
        sns.despine(ax=ax)

    fig_lines.tight_layout()
    _ensure_dir(save_path)
    fig_lines.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")

    # ── FIGURE 2: relative-improvement bars ─────────────────────────────────
    fig_gains, gain_axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

    for bax, (y_mean, _, _, smart_col, _, lower_better, _) in zip(gain_axes, subplot_specs):
        # Parity line: ratio = 1 means reg and SMART are equal.
        bax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", zorder=1)

        records = []
        if not SMART_DATA.empty and smart_col in SMART_DATA.columns:
            for _, s in SMART_DATA.sort_values("minutes").iterrows():
                label = _minutes_to_label(s["minutes"])
                if label in GAINS_EXCLUDED_LABELS:
                    continue
                matches = reg[np.isclose(reg["anchor_minutes"], s["minutes"], rtol=0.02)]
                if matches.empty:
                    continue
                s_val = s[smart_col]
                r_val = matches.iloc[0][y_mean]
                if pd.isna(s_val) or pd.isna(r_val) or s_val == 0 or r_val == 0:
                    continue
                # Orient so "higher = reg self-play is better" in every metric.
                ratio = s_val / r_val if lower_better else r_val / s_val
                records.append({"human_data": label, "ratio": ratio})

        if records:
            gain_df = pd.DataFrame(records)
            sns.barplot(
                data=gain_df,
                x="human_data",
                y="ratio",
                order=category_order,
                color="g",
                ax=bax,
                zorder=2,
            )
            for container in bax.containers:
                bax.bar_label(container, fmt="%.2fx", fontsize=8, padding=2)
        else:
            bax.text(
                0.5,
                0.5,
                "no overlapping data",
                ha="center",
                va="center",
                transform=bax.transAxes,
                fontsize=10,
                color="gray",
            )

        bax.set_xlabel("Human demonstration data")
        bax.set_ylabel("Relative improvement \n of reg. self-play RL", fontsize=14)
        bax.tick_params(axis="x", rotation=35)
        for tick in bax.get_xticklabels():
            tick.set_ha("right")
        bax.grid(axis="y", alpha=0.3, linestyle="--")
        sns.despine(ax=bax)

    fig_gains.tight_layout()
    _ensure_dir(save_path_gains)
    fig_gains.savefig(save_path_gains, dpi=DPI, bbox_inches="tight", facecolor="white")

    plt.show()
    return fig_lines, fig_gains


def generate_human_data_latex_table(
    df,
    save_path="results/figures/eval_human_data_table.tex",
    smart_csv="results/smart_baseline_res.csv",
):
    """Companion table to plot_human_data_requirements.

    Top-3 values per metric column are highlighted with a three-tier pastel
    colormap sampled from the Depth Pro paper screenshot:
      - best   -> soft pastel green   (#6FCF6A)
      - 2nd    -> soft chartreuse     (#DFF04B)
      - 3rd    -> pale cream-yellow   (#FBF4D0)
    Best value in each column is additionally bolded. Ties share a tier.

    Required LaTeX packages:
      \\usepackage{booktabs}
      \\usepackage[table]{xcolor}
      \\usepackage{graphicx}
      \\usepackage{makecell}
      \\usepackage{bm}
    """

    SMART_DATA = _load_smart_baseline(smart_csv)

    scaling_modes = ["scaling_sp_val", "scaling_hr_val"]
    df = df[df["mode"].isin(scaling_modes)].copy()
    df = df[df["sp_maps"] == 50000]
    if df.empty:
        print("  No 50k sp_maps data — skipping generate_human_data_latex_table.")
        return None

    df["anchor_maps"] = df["anchor_maps"].fillna(0).astype(int)

    # ── Aggregate regularized results ───────────────────────────────────────
    hr = df[df["mode"] == "scaling_hr_val"]
    hr_agg = (
        hr.groupby("anchor_maps")[["at_fault_collision_rate", "collision_rate", "route_progress"]]
        .agg(["mean", "sem"])
        .reset_index()
    )
    hr_agg.columns = [
        "anchor_maps",
        "hr_atfault_mean",
        "hr_atfault_sem",
        "hr_coll_mean",
        "hr_coll_sem",
        "hr_progress_mean",
        "hr_progress_sem",
    ]

    sp = df[df["mode"] == "scaling_sp_val"]
    sp_agg = sp.groupby("anchor_maps")[["collision_rate", "route_progress"]].agg(["mean", "sem"]).reset_index()
    sp_agg.columns = [
        "anchor_maps",
        "sp_coll_mean",
        "sp_coll_sem",
        "sp_progress_mean",
        "sp_progress_sem",
    ]

    reg_agg = hr_agg.merge(sp_agg, on="anchor_maps", how="outer")
    reg_agg = reg_agg[reg_agg["anchor_maps"] > 0].copy()
    reg_agg["minutes"] = reg_agg["anchor_maps"] * 9 / 60

    for col in ("hr_atfault", "hr_coll", "hr_progress", "sp_coll", "sp_progress"):
        reg_agg[f"{col}_mean_pct"] = reg_agg[f"{col}_mean"] * 100
        reg_agg[f"{col}_sem_pct"] = reg_agg[f"{col}_sem"] * 100

    # ── Unified row structure ───────────────────────────────────────────────
    rows = []
    smart_sorted = SMART_DATA.sort_values("minutes").reset_index(drop=True)
    for _, r in smart_sorted.iterrows():
        rows.append(
            {
                "method": "SMART",
                "minutes": r["minutes"],
                "sp_coll_mean": r["sp_coll_pct"],
                "sp_coll_sem": np.nan,
                "sp_progress_mean": r["sp_progress_pct"],
                "sp_progress_sem": np.nan,
                "hr_coll_mean": r["hr_coll_pct"],
                "hr_coll_sem": np.nan,
                "hr_atfault_mean": r["hr_atfault_pct"],
                "hr_atfault_sem": np.nan,
                "hr_progress_mean": r["hr_progress_pct"],
                "hr_progress_sem": np.nan,
            }
        )
    reg_sorted = reg_agg.sort_values("minutes").reset_index(drop=True)
    for _, r in reg_sorted.iterrows():
        rows.append(
            {
                "method": "regularized self-play",
                "minutes": r["minutes"],
                "sp_coll_mean": r["sp_coll_mean_pct"],
                "sp_coll_sem": r["sp_coll_sem_pct"],
                "sp_progress_mean": r["sp_progress_mean_pct"],
                "sp_progress_sem": r["sp_progress_sem_pct"],
                "hr_coll_mean": r["hr_coll_mean_pct"],
                "hr_coll_sem": r["hr_coll_sem_pct"],
                "hr_atfault_mean": r["hr_atfault_mean_pct"],
                "hr_atfault_sem": r["hr_atfault_sem_pct"],
                "hr_progress_mean": r["hr_progress_mean_pct"],
                "hr_progress_sem": r["hr_progress_sem_pct"],
            }
        )

    table = pd.DataFrame(rows)

    def _fmt_minutes(minutes):
        if minutes < 60:
            return f"{int(round(minutes))} min"
        hours = minutes / 60
        if hours < 24:
            if hours == int(hours):
                return f"{int(hours)} hours"
            return f"{hours:.1f} hours"
        days = hours / 24
        if days == int(days):
            return f"{int(days)} days"
        return f"{days:.1f} days"

    table["human_data_label"] = table["minutes"].apply(_fmt_minutes)

    # ── Metric metadata ──────────────────────────────────────────────────────
    metrics = [
        ("sp_coll_mean", "sp_coll_sem", "Coll. (\\%) $\\downarrow$", False),
        ("sp_progress_mean", "sp_progress_sem", "Route prog. (\\%) $\\uparrow$", True),
        ("hr_coll_mean", "hr_coll_sem", "Coll. (\\%) $\\downarrow$", False),
        ("hr_atfault_mean", "hr_atfault_sem", "At-fault (\\%) $\\downarrow$", False),
        ("hr_progress_mean", "hr_progress_sem", "Route prog. (\\%) $\\uparrow$", True),
    ]

    # ── Top-3 ranking per column ────────────────────────────────────────────
    # Exact colors sampled from the Depth Pro screenshot.
    TIER_COLORS = {
        1: "tierbest",  # #6FCF6A — soft pastel green
        2: "tiersecond",  # #DFF04B — soft chartreuse
        3: "tierthird",  # #FBF4D0 — pale cream-yellow
    }

    rank_lookup = {}
    for mean_col, _, _, higher_is_better in metrics:
        vals = table[mean_col]
        finite = vals.dropna()
        if finite.empty:
            for i in range(len(table)):
                rank_lookup[(mean_col, i)] = None
            continue

        distinct_sorted = sorted(finite.unique(), reverse=higher_is_better)
        top3 = distinct_sorted[:3]
        val_to_tier = {v: i + 1 for i, v in enumerate(top3)}

        for i, v in enumerate(vals):
            if pd.isna(v):
                rank_lookup[(mean_col, i)] = None
                continue
            matched_tier = None
            for tv, tier in val_to_tier.items():
                if np.isclose(v, tv):
                    matched_tier = tier
                    break
            rank_lookup[(mean_col, i)] = matched_tier

    def _fmt_cell(mean, sem, mean_col, row_idx):
        if pd.isna(mean):
            return "---"
        tier = rank_lookup.get((mean_col, row_idx))
        is_best = tier == 1

        if pd.notna(sem) and sem != 0:
            body = f"{mean:.1f} \\pm {sem:.1f}"
            text = f"$\\bm{{{body}}}$" if is_best else f"${body}$"
        else:
            body = f"{mean:.1f}"
            text = f"\\textbf{{{body}}}" if is_best else body

        if tier is None:
            return text
        return f"\\cellcolor{{{TIER_COLORS[tier]}}} {text}"

    # ── Build LaTeX ──────────────────────────────────────────────────────────
    n_sp_cols = 2
    n_hr_cols = 3
    col_spec = "ll" + "|" + "r" * n_sp_cols + "|" + "r" * n_hr_cols

    lines = []
    lines.append(
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor}, "
        r"\usepackage{graphicx}, \usepackage{makecell}, \usepackage{bm}"
    )
    lines.append(r"\definecolor{tierbest}{HTML}{6FCF6A}")
    lines.append(r"\definecolor{tiersecond}{HTML}{DFF04B}")
    lines.append(r"\definecolor{tierthird}{HTML}{FBF4D0}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Performance vs.\ amount of human demonstration data at fixed 50k "
        r"self-play training maps. Top-3 values per column are highlighted "
        r"(\colorbox{tierbest}{best}, \colorbox{tiersecond}{2nd}, "
        r"\colorbox{tierthird}{3rd}); best value additionally in bold. "
        r"SMART self-play metrics are not reported because SMART is a behaviour "
        r"model rather than an RL agent.}"
    )
    lines.append(r"\label{tab:human_data_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    sp_header = r"\multicolumn{" + str(n_sp_cols) + r"}{c|}{Self-play (test)}"
    hr_header = r"\multicolumn{" + str(n_hr_cols) + r"}{c}{Human-replay (test)}"
    lines.append(r" & & " + sp_header + " & " + hr_header + r" \\")

    header2 = (
        r"\makecell{Human demos \\ used} & Method & "
        + " & ".join(m[2] for m in metrics[:n_sp_cols])
        + " & "
        + " & ".join(m[2] for m in metrics[n_sp_cols:])
        + r" \\"
    )
    lines.append(header2)
    lines.append(r"\midrule")

    prev_method = None
    for i, row in table.iterrows():
        if prev_method is not None and row["method"] != prev_method:
            lines.append(r"\midrule")
        prev_method = row["method"]

        data_cell = row["human_data_label"]
        method_cell = "SMART" if row["method"] == "SMART" else "reg. self-play"

        cells = [data_cell, method_cell]
        for mean_col, sem_col, _, _ in metrics:
            cells.append(_fmt_cell(row[mean_col], row[sem_col], mean_col, i))
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
# WOSAC bar chart: one bar per checkpoint, four subplots
# ---------------------------------------------------------------------------


def plot_wosac_lineplot(wosac_df, save_path="results/figures/eval_wosac_lineplot.pdf"):
    """Line plot of WOSAC realism metrics vs self-play training maps.

    Four subplots — realism meta-score, kinematic, interactive, map-based.
    x-axis: sp_maps (log scale); color/marker: anchor_maps (reg/unreg convention).
    SMART and Random reference lines drawn on each subplot.
    """
    if wosac_df is None or wosac_df.empty:
        print("  No WOSAC data found — skipping plot_wosac_lineplot.")
        return None

    wdf = wosac_df.copy()
    wdf["anchor_maps"] = wdf["anchor_maps"].fillna(0).astype(int)

    wosac_metrics = [
        "realism_meta_score",
        "kinematic_metrics",
        "interactive_metrics",
        "map_based_metrics",
    ]
    available = [m for m in wosac_metrics if m in wdf.columns]
    if not available:
        print("  WOSAC metric columns missing — skipping plot_wosac_lineplot.")
        return None

    agg = wdf.groupby(["sp_maps", "anchor_maps"])[available].agg(["mean", "sem"]).reset_index()
    flat_cols = ["sp_maps", "anchor_maps"]
    for m in available:
        flat_cols.extend([f"{m}_mean", f"{m}_sem"])
    agg.columns = flat_cols

    anchor_vals = sorted(agg["anchor_maps"].unique())
    color_map = _reg_unreg_colors(anchor_vals)
    markers = ["^", "s", "o", "D", "P", "X", "v", "*"]
    marker_map = {a: markers[i % len(markers)] for i, a in enumerate(anchor_vals)}

    def _anchor_label(a):
        return "no anchor (unreg)" if a == 0 else f"{_maps_to_human_time(a)} anchor"

    # Reference baselines per metric
    smart_scores = [0.7818, 0.5200, 0.8914, 0.8378]
    random_scores = [0.4459, 0.0506, 0.34, 0.4704]

    subplot_specs = [
        ("realism_meta_score", "Realism meta-score", "WOSAC realism meta-score"),
        ("kinematic_metrics", "Kinematic metrics", "WOSAC kinematic metrics"),
        ("interactive_metrics", "Interactive metrics", "WOSAC interactive metrics"),
        ("map_based_metrics", "Map-based metrics", "WOSAC map-based metrics"),
    ]
    subplot_specs = [(m, yl, t) for m, yl, t in subplot_specs if m in available]

    _set_style(len(anchor_vals))
    fig, axes = plt.subplots(1, len(subplot_specs), figsize=(5 * len(subplot_specs), 4.5))
    if len(subplot_specs) == 1:
        axes = [axes]

    for ax, (metric, ylabel, title), smart, rand in zip(axes, subplot_specs, smart_scores, random_scores):
        mean_col = f"{metric}_mean"
        sem_col = f"{metric}_sem"
        if mean_col not in agg.columns:
            continue

        for anchor in anchor_vals:
            subset = agg[agg["anchor_maps"] == anchor].sort_values("sp_maps")
            if subset.empty:
                continue
            ax.plot(
                subset["sp_maps"],
                subset[mean_col],
                color=color_map[anchor],
                marker=marker_map[anchor],
                markersize=8,
                linewidth=1.2,
                linestyle="-",
                zorder=3,
                label=_anchor_label(anchor),
            )

        ax.axhline(smart, color=COLOR_SMART, linestyle="--", linewidth=1.5, alpha=0.8, label="SMART")
        ax.axhline(rand, color="grey", linestyle="--", linewidth=1.5, alpha=0.7, label="Random")

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x >= 1000 else str(int(x)))
        )
        ax.set_xlabel("Self-play training maps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="best", framealpha=1.0, facecolor="white", edgecolor="lightgray")
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_wosac_submetrics(wosac_df, save_path="results/figures/eval_wosac_submetrics.pdf"):
    """3×3 line plot of every individual WOSAC sub-metric vs self-play training maps.

    Rows / columns:
      Kinematic (4):   likelihood_linear_speed, likelihood_linear_acceleration,
                       likelihood_angular_speed, likelihood_angular_acceleration
      Interactive (3): likelihood_collision_indication,
                       likelihood_distance_to_nearest_object,
                       likelihood_time_to_collision
      Map (2):         likelihood_distance_to_road_edge, likelihood_offroad_indication

    9 subplots total laid out in a 3×3 grid. Each panel has the same style as
    plot_wosac_lineplot: log x-axis, reg/unreg color+marker convention, no
    reference lines (sub-metric baselines are not publicly available).
    """
    if wosac_df is None or wosac_df.empty:
        print("  No WOSAC data found — skipping plot_wosac_submetrics.")
        return None

    # All 9 individual sub-metrics grouped by category
    subplot_specs = [
        # (column_name, ylabel, title)
        # ── Kinematic ──────────────────────────────────────────────────────
        ("likelihood_linear_speed", "Likelihood", "Linear speed"),
        ("likelihood_linear_acceleration", "Likelihood", "Linear acceleration"),
        ("likelihood_angular_speed", "Likelihood", "Angular speed"),
        ("likelihood_angular_acceleration", "Likelihood", "Angular acceleration"),
        # ── Interactive ────────────────────────────────────────────────────
        ("likelihood_collision_indication", "Likelihood", "Collision indication"),
        ("likelihood_distance_to_nearest_object", "Likelihood", "Dist. to nearest object"),
        ("likelihood_time_to_collision", "Likelihood", "Time to collision"),
        # ── Map ────────────────────────────────────────────────────────────
        ("likelihood_distance_to_road_edge", "Likelihood", "Dist. to road edge"),
        ("likelihood_offroad_indication", "Likelihood", "Offroad indication"),
    ]

    wdf = wosac_df.copy()
    wdf["anchor_maps"] = wdf["anchor_maps"].fillna(0).astype(int)

    available = [col for col, _, _ in subplot_specs if col in wdf.columns]
    if not available:
        print("  No WOSAC sub-metric columns found — skipping plot_wosac_submetrics.")
        return None

    subplot_specs = [(col, yl, t) for col, yl, t in subplot_specs if col in wdf.columns]

    agg = wdf.groupby(["sp_maps", "anchor_maps"])[available].agg(["mean", "sem"]).reset_index()
    flat_cols = ["sp_maps", "anchor_maps"]
    for m in available:
        flat_cols.extend([f"{m}_mean", f"{m}_sem"])
    agg.columns = flat_cols

    anchor_vals = sorted(agg["anchor_maps"].unique())
    color_map = _reg_unreg_colors(anchor_vals)
    markers = ["^", "s", "o", "D", "P", "X", "v", "*"]
    marker_map = {a: markers[i % len(markers)] for i, a in enumerate(anchor_vals)}

    def _anchor_label(a):
        return "no anchor (unreg)" if a == 0 else f"{_maps_to_human_time(a)} anchor"

    _set_style(len(anchor_vals))
    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes_flat = axes.flatten()

    # Group labels for row titles
    group_labels = {
        0: "Kinematic",
        1: "Kinematic",
        2: "Kinematic",
        3: "Kinematic",
        4: "Interactive",
        5: "Interactive",
        6: "Interactive",
        7: "Map",
        8: "Map",
    }

    for idx, (ax, (metric, ylabel, title)) in enumerate(zip(axes_flat, subplot_specs)):
        mean_col = f"{metric}_mean"
        if mean_col not in agg.columns:
            ax.set_visible(False)
            continue

        for anchor in anchor_vals:
            subset = agg[agg["anchor_maps"] == anchor].sort_values("sp_maps")
            if subset.empty:
                continue
            ax.plot(
                subset["sp_maps"],
                subset[mean_col],
                color=color_map[anchor],
                marker=marker_map[anchor],
                markersize=7,
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
        ax.set_title(f"[{group_labels[idx]}] {title}")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=7, loc="best", framealpha=1.0, facecolor="white", edgecolor="lightgray")
        sns.despine(ax=ax)

    # Hide any unused axes (in case fewer than 9 metrics are available)
    for ax in axes_flat[len(subplot_specs) :]:
        ax.set_visible(False)

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


def _load_smart_wosac_baseline(csv_path: str = "results/smart_baseline_res.csv") -> pd.DataFrame:
    """Load WOSAC scores from the SMART baseline CSV.

    Returns a DataFrame with columns:
        num_maps, minutes,
        wosac_score, wosac_kinematic_metrics,
        wosac_interactive_metrics, wosac_map_based_metrics

    WOSAC metrics in the SMART CSV are only populated in the scaling_sp_val
    mode (the former `all_agents` rows). Missing values stay NaN; returns an
    empty (correctly-typed) DataFrame if the CSV is absent.
    """
    cols = [
        "num_maps",
        "minutes",
        "wosac_score",
        "wosac_kinematic_metrics",
        "wosac_interactive_metrics",
        "wosac_map_based_metrics",
    ]
    if not os.path.exists(csv_path):
        print(f"  {csv_path} not found — SMART WOSAC baseline will be omitted.")
        return pd.DataFrame(columns=cols)

    raw = pd.read_csv(csv_path)
    raw = raw[~raw["checkpoint"].isin(_SMART_EXCLUDED_CHECKPOINTS)]
    # WOSAC metrics only live in the scaling_sp_val rows.
    raw = raw[raw["mode"] == "scaling_sp_val"].copy()

    raw["num_maps"] = raw["checkpoint"].apply(_smart_ckpt_to_num_maps)
    raw = raw.dropna(subset=["num_maps"]).copy()
    raw["num_maps"] = raw["num_maps"].astype(int)

    metric_cols = [c for c in cols if c.startswith("wosac_") and c in raw.columns]
    out = raw[["num_maps"] + metric_cols].sort_values("num_maps").reset_index(drop=True)
    out["minutes"] = out["num_maps"] * 9 / 60
    # Re-index to the full expected column list so missing WOSAC columns become NaN.
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def plot_human_data_requirements_wosac(
    wosac_df,
    save_path="results/figures/eval_human_data_requirements_wosac.pdf",
    smart_csv="results/smart_baseline_res.csv",
):
    """WOSAC-score version of plot_human_data_requirements.

    4 line-plot subplots, all higher-is-better WOSAC scores in [0, 1]:
        0) WOSAC realism meta-score   (main: realism_meta_score,   SMART: wosac_score)
        1) WOSAC kinematic metrics    (main: kinematic_metrics,    SMART: wosac_kinematic_metrics)
        2) WOSAC interactive metrics  (main: interactive_metrics,  SMART: wosac_interactive_metrics)
        3) WOSAC map-based metrics    (main: map_based_metrics,    SMART: wosac_map_based_metrics)

    Series per panel:
        - regularized self-play (ours):       line across anchor points (pale purple)
        - best unregularized self-play:       horizontal dashed line (blue)
        - SMART-tiny-CLSFT:                   line across SMART checkpoints (pink)
        - Ground-truth (UB):                  dashed reference line (upper bound)
        - Random:                             dashed reference line (lower bound)

    Reference baselines for Ground-truth and Random are fixed per metric,
    taken from the PufferDrive WOSAC baseline table (229 clean held-out
    validation scenes).
    """
    # ── Colours ─────────────────────────────────────────────────────────────
    COLOR_OURS = "#CCCCFF"
    COLOR_OURS_EDGE = "#6B3FA0"
    COLOR_SELFPLAY = "#4A7FD4"
    COLOR_SMART = "#E8609A"
    COLOR_SMART_EDGE = "#B4437A"

    # ── Data sources ────────────────────────────────────────────────────────
    if wosac_df is None or wosac_df.empty:
        print("  No WOSAC data — skipping plot_human_data_requirements_wosac.")
        return None

    SMART_DATA = _load_smart_wosac_baseline(smart_csv)

    wdf = wosac_df.copy()
    wdf["anchor_maps"] = wdf["anchor_maps"].fillna(0).astype(int)
    wdf = wdf[wdf["sp_maps"] == 50000]
    if wdf.empty:
        print("  No 50k sp_maps WOSAC data — skipping plot_human_data_requirements_wosac.")
        return None

    # (main_col, smart_col, ylabel, ub_score, random_score)
    # UB and Random from PufferDrive WOSAC baselines on 229 clean held-out scenes.
    subplot_specs = [
        ("realism_meta_score", "wosac_score", "WOSAC realism meta-score", 0.8179, 0.4459),
        ("kinematic_metrics", "wosac_kinematic_metrics", "WOSAC kinematic metrics", 0.6070, 0.0506),
        ("interactive_metrics", "wosac_interactive_metrics", "WOSAC interactive metrics", 0.9590, 0.7843),
        ("map_based_metrics", "wosac_map_based_metrics", "WOSAC map-based metrics", 0.8722, 0.4704),
    ]

    available_main = [m for m, _, _, _, _ in subplot_specs if m in wdf.columns]
    if not available_main:
        print("  WOSAC metric columns missing — skipping plot_human_data_requirements_wosac.")
        return None

    # ── Aggregate per anchor_maps ───────────────────────────────────────────
    agg = wdf.groupby("anchor_maps")[available_main].agg(["mean", "sem"]).reset_index()
    flat_cols = ["anchor_maps"]
    for m in available_main:
        flat_cols.extend([f"{m}_mean", f"{m}_sem"])
    agg.columns = flat_cols
    agg["anchor_minutes"] = agg["anchor_maps"] * 9 / 60

    unreg = agg[agg["anchor_maps"] == 0]
    reg = agg[agg["anchor_maps"] > 0].sort_values("anchor_minutes")

    tick_positions = [10, 30, 180, 1800, 75000]
    tick_labels = ["10 min", "30 min", "3 hours", "30 hours", "52 days"]

    # ── Plot ────────────────────────────────────────────────────────────────
    _set_style(3)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, (main_col, smart_col, ylabel, ub, rand) in zip(axes, subplot_specs):
        mean_col = f"{main_col}_mean"
        sem_col = f"{main_col}_sem"

        # Upper bound and random floor (drawn first so series lines sit on top).
        _draw_upper_bound(ax, ub, label="Ground-truth (UB)")
        # _draw_lower_bound(ax, rand, label="Random")

        if mean_col in reg.columns and not reg.empty:
            ax.errorbar(
                reg["anchor_minutes"],
                reg[mean_col],
                yerr=reg[sem_col],
                color=COLOR_OURS,
                marker="o",
                markersize=9,
                linewidth=2.0,
                capsize=3,
                markeredgecolor=COLOR_OURS_EDGE,
                markerfacecolor=COLOR_OURS,
                label="regularized self-play (ours)",
                zorder=4,
            )
        if mean_col in unreg.columns and not unreg.empty:
            ax.axhline(
                unreg[mean_col].iloc[0],
                color=COLOR_SELFPLAY,
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                label="best unregularized self-play",
                zorder=2,
            )
        smart_valid = SMART_DATA.dropna(subset=[smart_col]) if smart_col in SMART_DATA.columns else SMART_DATA.iloc[0:0]
        if not smart_valid.empty:
            ax.plot(
                smart_valid["minutes"],
                smart_valid[smart_col],
                color=COLOR_SMART,
                marker="o",
                markersize=9,
                linewidth=2.0,
                linestyle="-",
                markeredgecolor=COLOR_SMART_EDGE,
                markerfacecolor=COLOR_SMART,
                label="SMART-tiny-CLSFT",
                zorder=3,
            )

        ax.set_xscale("symlog", linthresh=60, linscale=1.2)
        ax.set_xticks(tick_positions, labels=tick_labels, rotation=35, ha="right")
        ax.minorticks_off()
        # ax.set_ylim(0, 1.02)  # WOSAC scores are in [0, 1]; leave a sliver above UB.
        ax.set_xlabel("Human demonstration data")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=7, loc="best", framealpha=1.0, facecolor="white", edgecolor="lightgray")
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------


def make_all_figures(df=None, wosac_df=None, anchor_df=None):
    """Generate all evaluation figures."""
    print("\nGenerating figures...")
    if df is not None and not df.empty:
        plot_scores(df)
        print("  Saved eval_scores.pdf")
        plot_scaling_barplot(df)
        print("  Saved eval_scaling_barplot.pdf")
        plot_scaling_scatter(df)
        print("  Saved eval_scaling_scatter.pdf")
        plot_data_requirements(df)
        print("  Saved eval_data_requirements.pdf")
        plot_human_data_requirements(df)
        print("  Saved eval_human_data_requirements.pdf")
        generate_scaling_latex_table(df)
        generate_hr_comparison_latex_table(df)
        plot_compatibility_tradeoff_bar(df)
        generate_human_data_latex_table(df)
        print("  Saved eval_compatibility_tradeoff_bar.pdf")
    plot_wosac_lineplot(wosac_df)
    print("  Saved eval_wosac_lineplot.pdf")
    plot_wosac_submetrics(wosac_df)
    print("  Saved eval_wosac_submetrics.pdf")
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
        plot_human_data_requirements_wosac(wosac_df)
        print(f"Loaded {WOSAC_CSV} ({len(wosac_df)} rows)")
    else:
        print(f"{WOSAC_CSV} not found — skipping WOSAC figures.")

    if os.path.exists(ANCHOR_CSV):
        anchor_df = pd.read_csv(ANCHOR_CSV)
        print(f"Loaded {ANCHOR_CSV} ({len(anchor_df)} rows)")
    else:
        print(f"{ANCHOR_CSV} not found — skipping anchor eval figure.")

    make_all_figures(df, wosac_df, anchor_df)
