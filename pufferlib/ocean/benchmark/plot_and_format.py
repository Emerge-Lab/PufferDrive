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
    palette = sns.color_palette("colorblind", n_colors=len(anchor_vals))
    color_map = {v: palette[i] for i, v in enumerate(anchor_vals)}
    markers = ["X", "o", "s", "D", "^", "v", "P", "*"]
    marker_map = {v: markers[i % len(markers)] for i, v in enumerate(anchor_vals)}
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

    # Assign colors: orange shades for regularized, blue/green for unregularized
    unreg_colors = ["k", "#2ca02c", "#17becf"]
    reg_colors = ["#ff7f0e", "#d62728", "#e377c2", "#9467bd", "#bcbd22"]

    unreg_keys = [k for k in series_keys if "_anchor0" in k]
    reg_keys = [k for k in series_keys if "_anchor0" not in k]

    color_map = {}
    for i, k in enumerate(unreg_keys):
        color_map[k] = unreg_colors[i % len(unreg_colors)]
    for i, k in enumerate(reg_keys):
        color_map[k] = reg_colors[i % len(reg_colors)]

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
            f"regularized, anchor with {_maps_to_human_time(r['anchor_maps'])} of human data"
            if r["anchor_maps"] > 0
            else "unregularized"
        ),
        axis=1,
    )

    palette = _set_style(scaling_df["policy"].nunique())

    subplot_specs = [
        ("score", "Score", "Self-play score — training, 50k maps"),
        ("collision_rate_pct", "Collision rate (%)", "Self-play collision rate (%) — training, 50k maps"),
        ("offroad_rate_pct", "Offroad rate (%)", "Self-play offroad rate (%) — training, 50k maps"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (y_col, ylabel, title) in zip(axes, subplot_specs):
        sns.barplot(data=scaling_df, x="policy", y=y_col, errorbar="se", palette=palette, ax=ax, alpha=0.8)
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

    Required LaTeX packages:
      \\usepackage{booktabs}
      \\usepackage[table]{xcolor}
      \\usepackage{graphicx}   % for \\resizebox

    Rows:   one per (sp_maps, anchor_maps) configuration
    Columns:
      - Self-play maps (metadata), Anchor data
      - Self-play (test): score, collision rate, offroad rate
      - Human-replay (test): score, collision rate, at-fault collision rate
    """
    scaling_modes = ["scaling_sp_val", "scaling_hr_interactive"]
    scaling_df = df[df["mode"].isin(scaling_modes)].copy()
    if scaling_df.empty:
        print("  No scaling data found — skipping generate_scaling_latex_table.")
        return None

    scaling_df["anchor_maps"] = scaling_df["anchor_maps"].fillna(0).astype(int)

    # Self-play metrics: score, collision_rate, offroad_rate
    # Human-replay metrics: score, collision_rate, at_fault_collision_rate
    sp_metrics = ["score", "collision_rate", "offroad_rate"]
    hr_metrics = ["score", "collision_rate", "at_fault_collision_rate"]
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
    merged = merged.sort_values(["sp_maps", "anchor_maps"]).reset_index(drop=True)

    has_offroad = "offroad_rate" in available_metrics
    has_at_fault = "at_fault_collision_rate" in available_metrics

    # --- Collect all values for normalization ---
    score_cols = ["sp_score_mean", "hr_score_mean"]
    coll_cols = ["sp_collision_rate_mean", "hr_collision_rate_mean"]
    if has_offroad:
        coll_cols.append("sp_offroad_rate_mean")
    if has_at_fault:
        coll_cols.append("hr_at_fault_collision_rate_mean")

    existing_score_cols = [c for c in score_cols if c in merged.columns]
    existing_coll_cols = [c for c in coll_cols if c in merged.columns]

    all_scores = merged[existing_score_cols].values.flatten()
    all_scores = all_scores[~np.isnan(all_scores)]
    all_colls = merged[existing_coll_cols].values.flatten()
    all_colls = all_colls[~np.isnan(all_colls)]

    score_min, score_max = (all_scores.min(), all_scores.max()) if len(all_scores) > 0 else (0, 1)
    coll_min, coll_max = (all_colls.min(), all_colls.max()) if len(all_colls) > 0 else (0, 1)

    def _score_intensity(val):
        """Map score to green intensity 5-50 (darker = better = higher score)."""
        if np.isnan(val):
            return 0
        if score_max == score_min:
            return 25
        t = (val - score_min) / (score_max - score_min)
        return int(5 + t * 45)

    def _coll_intensity(val):
        """Map collision rate to red intensity 5-50 (lighter = better = lower rate)."""
        if np.isnan(val):
            return 0
        if coll_max == coll_min:
            return 25
        t = (val - coll_min) / (coll_max - coll_min)
        return int(5 + t * 45)

    def _fmt_score(mean, sem, is_best=False):
        """Format score with green cellcolor. Bold if best in column."""
        if np.isnan(mean):
            return "---"
        intensity = _score_intensity(mean)
        if not (np.isnan(sem) or sem == 0):
            if is_best:
                text = f"$\\bm{{{mean:.3f} \\pm {sem:.3f}}}$"
            else:
                text = f"${mean:.3f} \\pm {sem:.3f}$"
        else:
            text = f"\\textbf{{{mean:.3f}}}" if is_best else f"{mean:.3f}"
        return f"\\cellcolor{{green!{intensity}}} {text}"

    def _fmt_coll(mean, sem, is_best=False):
        """Format collision/offroad rate (as %) with red cellcolor. Bold if best in column."""
        if np.isnan(mean):
            return "---"
        intensity = _coll_intensity(mean)
        m_pct, s_pct = mean * 100, sem * 100
        if not (np.isnan(s_pct) or s_pct == 0):
            if is_best:
                text = f"$\\bm{{{m_pct:.1f} \\pm {s_pct:.1f}}}$"
            else:
                text = f"${m_pct:.1f} \\pm {s_pct:.1f}$"
        else:
            text = f"\\textbf{{{m_pct:.1f}}}" if is_best else f"{m_pct:.1f}"
        return f"\\cellcolor{{red!{intensity}}} {text}"

    def _anchor_label(anchor_maps):
        if anchor_maps == 0:
            return "0 (unreg.)"
        return _maps_to_human_time(anchor_maps)

    # --- Build LaTeX ---
    n_sp_metric_cols = 2 + int(has_offroad)  # score, coll, offroad
    n_hr_metric_cols = 2 + int(has_at_fault)  # score, coll, at-fault
    col_spec = "rr" + "|" + "r" * n_sp_metric_cols + "|" + "r" * n_hr_metric_cols

    lines = []
    lines.append(
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor}, \usepackage{graphicx}, \usepackage{makecell}, \usepackage{bm}"
    )
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Scaling evaluation results on held-out test maps.}")
    lines.append(r"\label{tab:scaling_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    sp_header = r"\multicolumn{" + str(n_sp_metric_cols) + r"}{c|}{Self-play (test)}"
    hr_header = r"\multicolumn{" + str(n_hr_metric_cols) + r"}{c}{Human-replay (test)}"
    lines.append(r" & & " + sp_header + " & " + hr_header + r" \\")

    sp_metric_headers = ["Score", "Coll. (\\%)"]
    if has_offroad:
        sp_metric_headers.append("Offroad (\\%)")
    hr_metric_headers = ["Score", "Coll. (\\%)"]
    if has_at_fault:
        hr_metric_headers.append("At-fault (\\%)")
    header2 = (
        "\\makecell{Self-play maps \\\\ (metadata)} & Anchor data & "
        + " & ".join(sp_metric_headers)
        + " & "
        + " & ".join(hr_metric_headers)
        + r" \\"
    )
    lines.append(header2)
    lines.append(r"\midrule")

    # --- Compute best value per column (highest score, lowest collision/offroad/at-fault rate) ---
    best = {}
    for col in existing_score_cols:
        best[col] = merged[col].max()
    for col in existing_coll_cols:
        best[col] = merged[col].min()

    def _is_best(col, val):
        if col not in best or np.isnan(val):
            return False
        return np.isclose(val, best[col])

    for _, row in merged.iterrows():
        sp_maps_str = _fmt_maps(int(row["sp_maps"]))
        anchor_str = _anchor_label(int(row["anchor_maps"]))
        cells = [sp_maps_str, anchor_str]

        # Self-play: score, collision rate, offroad rate
        cells.append(
            _fmt_score(
                row.get("sp_score_mean", np.nan),
                row.get("sp_score_sem", np.nan),
                is_best=_is_best("sp_score_mean", row.get("sp_score_mean", np.nan)),
            )
        )
        cells.append(
            _fmt_coll(
                row.get("sp_collision_rate_mean", np.nan),
                row.get("sp_collision_rate_sem", np.nan),
                is_best=_is_best("sp_collision_rate_mean", row.get("sp_collision_rate_mean", np.nan)),
            )
        )
        if has_offroad:
            cells.append(
                _fmt_coll(
                    row.get("sp_offroad_rate_mean", np.nan),
                    row.get("sp_offroad_rate_sem", np.nan),
                    is_best=_is_best("sp_offroad_rate_mean", row.get("sp_offroad_rate_mean", np.nan)),
                )
            )

        # Human-replay: score, collision rate, at-fault collision rate
        cells.append(
            _fmt_score(
                row.get("hr_score_mean", np.nan),
                row.get("hr_score_sem", np.nan),
                is_best=_is_best("hr_score_mean", row.get("hr_score_mean", np.nan)),
            )
        )
        cells.append(
            _fmt_coll(
                row.get("hr_collision_rate_mean", np.nan),
                row.get("hr_collision_rate_sem", np.nan),
                is_best=_is_best("hr_collision_rate_mean", row.get("hr_collision_rate_mean", np.nan)),
            )
        )
        if has_at_fault:
            cells.append(
                _fmt_coll(
                    row.get("hr_at_fault_collision_rate_mean", np.nan),
                    row.get("hr_at_fault_collision_rate_sem", np.nan),
                    is_best=_is_best(
                        "hr_at_fault_collision_rate_mean", row.get("hr_at_fault_collision_rate_mean", np.nan)
                    ),
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

    _cb = sns.color_palette("colorblind")
    REG_COLORS = ["#f4a6c0", "#e8729a", "#d63b73", "#a8174a", "#6b0f2e"]
    color_map = {0: "black"}
    color_map.update({a: REG_COLORS[i] for i, a in enumerate(a for a in anchor_vals if a != 0)})
    markers = ["^", "s", "o", "D", "P", "X", "v", "*"]
    marker_map = {a: markers[i % len(markers)] for i, a in enumerate(anchor_vals)}

    def _anchor_label(a):
        return "no anchor (unreg)" if a == 0 else f"{_maps_to_human_time(a)} anchor"

    _set_style(len(anchor_vals))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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
                # fontweight="bold",
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
    colors = ["black" if not r else "#d63b73" for r in agg["is_reg"]]

    subplot_specs = [
        ("collision_rate", "HR collision rate [%]", True),
        ("at_fault_collision_rate", "HR at-fault collision rate [%]", True),
        ("rear_collision_rate", "HR rear collision rate [%]", True),
        ("route_progress", "HR route progress [%]", True),
        ("lateral_error_avg", "HR lateral L2 distance", False),
    ]

    _set_style(2)
    fig = plt.figure(figsize=(14, 3.5))
    gs = fig.add_gridspec(1, 4)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)
    ax3 = fig.add_subplot(gs[3])
    axes = [ax0, ax1, ax2, ax3]

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
        generate_scaling_latex_table(df)
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
    import pandas as pd

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
