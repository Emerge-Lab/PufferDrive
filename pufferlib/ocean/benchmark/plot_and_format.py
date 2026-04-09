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


def plot_scores(df, save_path="eval_scores.pdf"):
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
    unreg_colors = ["#1f77b4", "#2ca02c", "#17becf", "#9467bd"]
    reg_colors = ["#ff7f0e", "#d62728", "#e377c2", "#bcbd22"]

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
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_scaling_barplot(df, save_path="eval_scaling_barplot.pdf"):
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
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def plot_scaling_wosac(wosac_df, save_path="eval_scaling_wosac.pdf"):
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


def generate_scaling_latex_table(df, save_path="eval_scaling_table.tex"):
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

    with open(save_path, "w") as f:
        f.write(latex_str)

    print(f"  LaTeX table written to {save_path}")
    return latex_str


def make_all_figures(df=None, wosac_df=None):
    """Generate all evaluation figures."""
    print("\nGenerating figures...")
    if df is not None and not df.empty:
        plot_scores(df)
        print("  Saved eval_scores.pdf")
        plot_scaling_barplot(df)
        print("  Saved eval_scaling_barplot.pdf")
        plot_scaling_scatter(df)
        print("  Saved eval_scaling_scatter.pdf")
        generate_scaling_latex_table(df)
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
