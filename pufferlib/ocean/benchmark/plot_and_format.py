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

# ── Shared colour palette ───────────────────────────────────────────────────
# Anything plot- or table-facing that references a method or reg/unreg policy
# should read from here so the figures and tables stay in sync.
PALETTE = {
    # Methods (used across line plots and as LaTeX cell colours).
    "smart": "#d62728",  # tab:red
    "smart_edge": "#8B1A1B",  # darker red for marker edges
    "ours": "#6BAED6",  # medium blue (regularized self-play)
    "ours_edge": "#08519C",  # dark blue for marker edges
    # "selfplay" is the unregularized baseline colour everywhere — lines,
    # markers, dashed reference lines, and unreg reg/unreg map entries.
    "selfplay": "#000000",  # black
    # Regularized anchor runs: light → dark = less → more human data.
    # Sequence of blues so multiple reg lines stay distinguishable.
    "reg_sequence": [
        "#C6DBEF",
        "#9ECAE1",
        "#6BAED6",
        "#3182BD",
        "#08519C",
    ],
    # LaTeX tier highlighting
    "tier_best": "#6FCF6A",  # soft pastel green
    "tier_second": "#DFF04B",  # soft chartreuse
    "tier_third": "#FBF4D0",  # pale cream-yellow
    # Best-per-column among unregularized rows. Overlays tier color when both apply.
    # Soft gray tint, matching the new black `selfplay` colour.
    "tier_unreg_best": "#D9D9D9",
}

# Back-compat aliases so existing references keep working without edits.
# New code should prefer PALETTE[...] directly.
COLOR_SMART = PALETTE["smart"]
COLOR_SMART_EDGE = PALETTE["smart_edge"]
COLOR_OURS = PALETTE["ours"]
COLOR_OURS_EDGE = PALETTE["ours_edge"]
COLOR_SELFPLAY = PALETTE["selfplay"]


def _ensure_dir(path):
    """Create parent directories for *path* if they don't already exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ── Tier-colouring helpers for LaTeX tables ─────────────────────────────────

_TIER_NAMES = {1: "tierbest", 2: "tiersecond", 3: "tierthird"}


def _tier_latex_preamble():
    """Return the \\definecolor lines shared by every tier-highlighted table."""
    return [
        rf"\definecolor{{tierbest}}{{HTML}}{{{PALETTE['tier_best'].lstrip('#')}}}",
        rf"\definecolor{{tiersecond}}{{HTML}}{{{PALETTE['tier_second'].lstrip('#')}}}",
        rf"\definecolor{{tierthird}}{{HTML}}{{{PALETTE['tier_third'].lstrip('#')}}}",
        rf"\definecolor{{tierunregbest}}{{HTML}}{{{PALETTE['tier_unreg_best'].lstrip('#')}}}",
    ]


def _build_tier_rank_lookup(table, metrics):
    """Compute per-column top-3 tier assignments.

    `metrics` is a list of tuples whose first element is the mean-column name
    and whose fourth element is `higher_is_better` (bool). Extra tuple
    elements are allowed and ignored.

    Returns a dict {(mean_col, row_idx): tier} where tier ∈ {1, 2, 3, None}.
    Ties at a rank share that rank's tier.
    """
    rank_lookup = {}
    for spec in metrics:
        mean_col = spec[0]
        higher_is_better = spec[3]
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
            matched = None
            for tv, tier in val_to_tier.items():
                if np.isclose(v, tv):
                    matched = tier
                    break
            rank_lookup[(mean_col, i)] = matched
    return rank_lookup


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
    return f"{minutes:.0f} minutes"


def _maps_to_human_hours(maps: int) -> float:
    """Convert map count to hours of human driving data.

    Each map is a 9-second scenario with 1 controlled agent on average:
        hours = (maps × 9s × 1) / 3600
    """
    return (maps * 9) / 3600


def _reg_unreg_colors(anchor_vals):
    """Map each anchor value to a colour.

    anchor == 0  -> black  (PALETTE['selfplay']; unregularized baseline)
    anchor  > 0  -> blues from PALETTE['reg_sequence'], assigned in
                    ascending order of anchor size so darker = more human data.
    """
    reg_seq = PALETTE["reg_sequence"]
    color_map = {}
    reg_idx = 0
    for v in sorted(anchor_vals):
        if v == 0:
            color_map[v] = PALETTE["selfplay"]
        else:
            color_map[v] = reg_seq[reg_idx % len(reg_seq)]
            reg_idx += 1
    return color_map


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
    """Scaling evaluation as a LaTeX table with Depth-Pro top-3 tier coloring.

    Top-3 values per metric column are highlighted:
      - best   -> soft pastel green   (#6FCF6A)
      - 2nd    -> soft chartreuse     (#DFF04B)
      - 3rd    -> pale cream-yellow   (#FBF4D0)
    Best value in each column is additionally bolded. Ties share a tier.

    The best unregularized value per column is highlighted via the
    `tier_unreg_best` palette entry (a soft gray, matching the new black
    `selfplay` colour). This wins over the tier color when a cell qualifies
    for both, so the unreg signal stays visible even when unreg is competitive
    with the overall top-3. Bold (for overall-best) is preserved regardless.

    Row layout: unregularized rows first, then regularized, each sorted by sp_maps.

    Required LaTeX packages:
      \\usepackage{booktabs}, \\usepackage[table]{xcolor},
      \\usepackage{graphicx}, \\usepackage{makecell}, \\usepackage{bm}
    """
    scaling_modes = ["scaling_sp_val", "scaling_hr_interactive"]
    scaling_df = df[df["mode"].isin(scaling_modes)].copy()
    if scaling_df.empty:
        print("  No scaling data found — skipping generate_scaling_latex_table.")
        return None

    scaling_df["anchor_maps"] = scaling_df["anchor_maps"].fillna(0).astype(int)

    sp_cols = ["score", "collision_rate", "offroad_rate"]
    hr_cols = [
        "score",
        "collision_rate",
        "at_fault_collision_rate",
        "rear_collision_rate",
        "route_progress",
        "lateral_error_avg",
    ]
    available = [m for m in set(sp_cols + hr_cols) if m in scaling_df.columns]

    agg = scaling_df.groupby(["sp_maps", "anchor_maps", "mode"])[available].agg(["mean", "sem"]).reset_index()
    flat_cols = ["sp_maps", "anchor_maps", "mode"]
    for m in available:
        flat_cols.extend([f"{m}_mean", f"{m}_sem"])
    agg.columns = flat_cols

    sp = agg[agg["mode"] == "scaling_sp_val"].drop(columns=["mode"]).copy()
    hr = agg[agg["mode"] == "scaling_hr_interactive"].drop(columns=["mode"]).copy()
    sp = sp.rename(columns={c: f"sp_{c}" for c in sp.columns if c not in ("sp_maps", "anchor_maps")})
    hr = hr.rename(columns={c: f"hr_{c}" for c in hr.columns if c not in ("sp_maps", "anchor_maps")})
    merged = sp.merge(hr, on=["sp_maps", "anchor_maps"], how="outer")

    # unreg rows first, then reg rows, each sorted by sp_maps
    unreg = merged[merged["anchor_maps"] == 0].sort_values("sp_maps")
    reg = merged[merged["anchor_maps"] != 0].sort_values(["sp_maps", "anchor_maps"])
    merged = pd.concat([unreg, reg]).reset_index(drop=True)

    # Metric specs: (mean_col, sem_col, header, higher_is_better, as_pct, decimals)
    sp_specs = [
        ("sp_score_mean", "sp_score_sem", r"Score $\uparrow$", True, False, 3),
        ("sp_collision_rate_mean", "sp_collision_rate_sem", r"Coll. (\%) $\downarrow$", False, True, 1),
    ]
    if "offroad_rate" in available:
        sp_specs.append(("sp_offroad_rate_mean", "sp_offroad_rate_sem", r"Off-road (\%) $\downarrow$", False, True, 1))

    hr_specs = [
        ("hr_score_mean", "hr_score_sem", r"Score $\uparrow$", True, False, 3),
        ("hr_collision_rate_mean", "hr_collision_rate_sem", r"Coll. (\%) $\downarrow$", False, True, 1),
    ]
    if "at_fault_collision_rate" in available:
        hr_specs.append(
            (
                "hr_at_fault_collision_rate_mean",
                "hr_at_fault_collision_rate_sem",
                r"At-fault (\%) $\downarrow$",
                False,
                True,
                1,
            )
        )
    if "rear_collision_rate" in available:
        hr_specs.append(
            (
                "hr_rear_collision_rate_mean",
                "hr_rear_collision_rate_sem",
                r"Rear coll. (\%) $\downarrow$",
                False,
                True,
                1,
            )
        )
    if "route_progress" in available:
        hr_specs.append(("hr_route_progress_mean", "hr_route_progress_sem", r"Route prog. $\uparrow$", True, False, 3))
    if "lateral_error_avg" in available:
        hr_specs.append(
            ("hr_lateral_error_avg_mean", "hr_lateral_error_avg_sem", r"Lateral L2 $\downarrow$", False, False, 2)
        )

    all_specs = [s for s in sp_specs + hr_specs if s[0] in merged.columns]
    n_sp = sum(1 for s in all_specs if s[0].startswith("sp_"))
    n_hr = sum(1 for s in all_specs if s[0].startswith("hr_"))

    rank_lookup = _build_tier_rank_lookup(merged, all_specs)

    # Per-column best among unregularized rows (anchor_maps == 0).
    # Ties share the highlight. Stored as a set of (mean_col, row_idx).
    unreg_mask = merged["anchor_maps"] == 0
    unreg_best_cells = set()
    for spec in all_specs:
        mean_col = spec[0]
        higher_is_better = spec[3]
        unreg_vals = merged.loc[unreg_mask, mean_col].dropna()
        if unreg_vals.empty:
            continue
        target = unreg_vals.max() if higher_is_better else unreg_vals.min()
        for i, v in merged[mean_col].items():
            if unreg_mask.iloc[i] and pd.notna(v) and np.isclose(v, target):
                unreg_best_cells.add((mean_col, i))

    def _fmt_cell(mean, sem, mean_col, row_idx, as_pct, decimals):
        if pd.isna(mean):
            return "---"
        tier = rank_lookup.get((mean_col, row_idx))
        is_best = tier == 1
        is_unreg_best = (mean_col, row_idx) in unreg_best_cells
        m_val = mean * 100 if as_pct else mean
        s_val = sem * 100 if (as_pct and pd.notna(sem)) else sem
        fmt = f".{decimals}f"
        if pd.notna(s_val) and s_val != 0:
            body = f"{m_val:{fmt}} \\pm {s_val:{fmt}}"
            text = f"$\\bm{{{body}}}$" if is_best else f"${body}$"
        else:
            body = f"{m_val:{fmt}}"
            text = f"\\textbf{{{body}}}" if is_best else body
        # Unreg-best gray wins over tier color — otherwise the new signal
        # would be invisible exactly when the unreg row is also top-3.
        if is_unreg_best:
            return f"\\cellcolor{{tierunregbest}} {text}"
        if tier is None:
            return text
        return f"\\cellcolor{{{_TIER_NAMES[tier]}}} {text}"

    def _anchor_label(anchor_maps):
        return "0 (unreg.)" if anchor_maps == 0 else _maps_to_human_time(anchor_maps)

    col_spec = "rr" + "|" + "r" * n_sp + "|" + "r" * n_hr

    lines = []
    lines.append(
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor}, "
        r"\usepackage{graphicx}, \usepackage{makecell}, \usepackage{bm}"
    )
    lines.extend(_tier_latex_preamble())
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Scaling evaluation on held-out Waymo scenarios. "
        r"Self-play metrics are reported on 10k randomly sampled validation "
        r"scenarios; human-replay metrics on 200 interactive validation "
        r"scenarios. Top-3 values per column are highlighted "
        r"(\colorbox{tierbest}{best}, \colorbox{tiersecond}{2nd}, "
        r"\colorbox{tierthird}{3rd}); best value additionally in bold. "
        r"\colorbox{tierunregbest}{Gray} marks the best unregularized "
        r"value per column.}"
    )
    lines.append(r"\label{tab:scaling_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(
        r" & & "
        + r"\multicolumn{"
        + str(n_sp)
        + r"}{c|}{Self-play (test)} & "
        + r"\multicolumn{"
        + str(n_hr)
        + r"}{c}{Human-replay (test)}"
        + r" \\"
    )

    headers = [s[2] for s in all_specs]
    header_row = (
        r"\makecell{Self-play maps \\ (metadata)} & "
        r"\makecell{Anchor data \\ (human demonstrations)} & "
        + " & ".join(headers[:n_sp])
        + " & "
        + " & ".join(headers[n_sp:])
        + r" \\"
    )
    lines.append(header_row)
    lines.append(r"\midrule")

    prev_was_unreg = None
    for i, row in merged.iterrows():
        is_unreg = int(row["anchor_maps"]) == 0
        if prev_was_unreg is True and not is_unreg:
            lines.append(r"\midrule")
        prev_was_unreg = is_unreg

        cells = [_fmt_maps(int(row["sp_maps"])), _anchor_label(int(row["anchor_maps"]))]
        for mean_col, sem_col, _h, _hib, as_pct, decimals in all_specs:
            cells.append(
                _fmt_cell(
                    row.get(mean_col, np.nan),
                    row.get(sem_col, np.nan),
                    mean_col,
                    i,
                    as_pct,
                    decimals,
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


# ---------------------------------------------------------------------------
# Helper: final-step within-5-bin accuracy from wandb training-curve CSVs
# ---------------------------------------------------------------------------


def _load_final_within5_accuracy(
    dx_csv="results/anchor_train_dx.csv",
    dy_csv="results/anchor_train_dy.csv",
    dyaw_csv="results/anchor_train_dyaw.csv",
):
    """Return a DataFrame [num_maps_trained, within5_avg_pct] from the wandb CSVs.

    For each run (one column per axis CSV), takes the last non-NaN value of the
    mean column, then averages across the three axes per num_maps. Drops runs
    that don't have all three axes available so the "average" means the same
    thing on every row. Returns an empty (correctly-typed) frame if none of
    the CSVs are present.
    """
    axis_files = [("dx", dx_csv), ("dy", dy_csv), ("dyaw", dyaw_csv)]
    rows = []  # (num_maps, axis, final_acc)

    for axis, path in axis_files:
        if not os.path.exists(path):
            print(f"  {path} not found — skipping {axis} for within-5-bin overlay.")
            continue
        df = pd.read_csv(path)
        metric = f"val/acc_within_5bins_{axis}"
        for col in df.columns:
            # Match the mean column only; skip wandb's __MIN/__MAX siblings.
            if not col.endswith(f" - {metric}") or col.endswith("__MIN") or col.endswith("__MAX"):
                continue
            run = col.split(" - ")[0]
            m = re.search(r"_(\d+)maps", run)
            if not m:
                continue
            num_maps = int(m.group(1))
            series = df[col].dropna()
            if series.empty:
                continue
            rows.append({"num_maps_trained": num_maps, "axis": axis, "final_acc": series.iloc[-1]})

    if not rows:
        return pd.DataFrame(columns=["num_maps_trained", "within5_avg_pct"])

    long_df = pd.DataFrame(rows)
    counts = long_df.groupby("num_maps_trained")["axis"].nunique()
    complete = counts[counts == 3].index
    incomplete = sorted(set(long_df["num_maps_trained"]) - set(complete))
    if incomplete:
        print(f"  Skipping within-5-bin overlay for runs missing axes: {incomplete}")
    long_df = long_df[long_df["num_maps_trained"].isin(complete)]

    out = (
        long_df.groupby("num_maps_trained")["final_acc"]
        .mean()
        .reset_index()
        .rename(columns={"final_acc": "within5_avg_pct"})
    )
    out["within5_avg_pct"] = out["within5_avg_pct"] * 100  # to %
    return out.sort_values("num_maps_trained").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Updated: BC anchor figure (subplot 0 now overlays within-5-bin accuracy)
# ---------------------------------------------------------------------------


def plot_anchor_eval(
    anchor_df,
    save_path="results/figures/eval_anchor.pdf",
    within5_csvs=("results/anchor_train_dx.csv", "results/anchor_train_dy.csv", "results/anchor_train_dyaw.csv"),
):
    """Four-subplot summary figure for BC anchor evaluation.

    All subplots share the same x-axis: hours of human driving data used to
    train the anchor (num_maps_trained via _maps_to_human_hours).

    Subplot 0 — Open-loop: argmax accuracy + final within-5-bin accuracy
                (averaged across dx/dy/dyaw, taken at last training step).
    Subplot 1 — Open-loop: validation loss.
    Subplot 2 — Closed-loop: route progress, self-play vs human-replay.
    Subplot 3 — Closed-loop: score, self-play vs human-replay.
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

    # Within-5-bin: average final-step accuracy across dx/dy/dyaw, joined by num_maps.
    w5_df = _load_final_within5_accuracy(*within5_csvs) if within5_csvs else pd.DataFrame()
    if not w5_df.empty:
        w5_df["human_hours"] = w5_df["num_maps_trained"].apply(_maps_to_human_hours)
        w5_df = w5_df.sort_values("human_hours")

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

    # ── Subplot 0: open-loop accuracy + within-5-bin avg ─────────────────────
    ax = axes[0]
    ax.plot(
        ol_df["human_hours"],
        ol_df["ol_val_accuracy"] * 100,
        color="#1f77b4",
        marker="D",
        linewidth=1.5,
        markersize=8,
        label="Argmax accuracy",
    )
    if not w5_df.empty:
        ax.plot(
            w5_df["human_hours"],
            w5_df["within5_avg_pct"],
            color="#9467bd",
            marker="s",
            linewidth=1.5,
            markersize=8,
            label=r"Within-5-bin accuracy (avg over $\Delta x, \Delta y, \Delta\mathrm{yaw}$)",
        )
    ax.set_xlabel("Human driving demonstrations (hours)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation accuracy")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="best")
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
    ax.set_xlabel("Human driving demonstrations (hours)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss")
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


def plot_selfplay_behavior_analysis(df, save_path="results/figures/eval_selfplay_behavior_analysis.pdf"):
    """Bar chart comparing two 50k checkpoints across four self-play behavior metrics.

    4 subplots, one per metric. Two bars per subplot: unregularized vs regularized,
    using the shared reg/unreg color convention (black = unreg, blue = reg).
    Mode: scaling_sp_val (self-play). Mean ± SEM, raw values, no normalization.

    Metrics:
        - collisions_per_agent
        - lateral_error_avg
        - longitudinal_error_avg
        - displacement_error_avg
    """
    CHECKPOINTS_OF_INTEREST = {
        "models/scaling_cpts/unreg_delta_50k_maps.pt": "unregularized",
        "models/scaling_cpts/reg_delta_50k_maps_anchor_200_maps.pt": "regularized",
    }

    df = df[df["mode"] == "scaling_sp_val"].copy()
    df = df[df["checkpoint"].isin(CHECKPOINTS_OF_INTEREST)].copy()
    if df.empty:
        print("  No self-play data for checkpoints of interest — skipping plot_selfplay_behavior_analysis.")
        return None

    required_cols = [
        "collision_rate",
        "lateral_error_avg",
        "longitudinal_error_avg",
        "displacement_error_avg",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  Missing columns {missing} — skipping plot_selfplay_behavior_analysis.")
        return None

    agg = df.groupby("checkpoint")[required_cols].agg(["mean", "sem"]).reset_index()
    agg.columns = ["checkpoint"] + [f"{m}_{s}" for m in required_cols for s in ["mean", "sem"]]
    agg["label"] = agg["checkpoint"].map(CHECKPOINTS_OF_INTEREST)

    # unreg first (black), reg second (blue) — matches the rest of the file.
    agg["is_reg"] = ~agg["checkpoint"].str.contains("unreg")
    agg = agg.sort_values("is_reg").reset_index(drop=True)
    colors = [PALETTE["selfplay"] if not r else PALETTE["ours"] for r in agg["is_reg"]]

    # (column, ylabel). None of these are percentages — they're counts/distances.
    subplot_specs = [
        ("collision_rate", "SP collision rate"),
        ("lateral_error_avg", "SP lateral L2 distance"),
        ("longitudinal_error_avg", "SP longitudinal L2 distance"),
        ("displacement_error_avg", "SP avg. displacement error"),
    ]

    _set_style(2)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, (col, ylabel) in zip(axes, subplot_specs):
        means = agg[f"{col}_mean"].values
        sems = agg[f"{col}_sem"].values
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
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.tick_params(axis="y", which="minor", length=3, color="gray")
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Updated: BC anchor LaTeX table (now includes within-5-bin accuracy column)
# ---------------------------------------------------------------------------


def generate_anchor_latex_table(
    anchor_df,
    save_path="results/figures/anchor_eval_table.tex",
    within5_csvs=("results/anchor_train_dx.csv", "results/anchor_train_dy.csv", "results/anchor_train_dyaw.csv"),
):
    """LaTeX table for BC anchor evaluation results.

    Rows:    one per checkpoint (num_maps_trained), sorted ascending.
    Columns: Human data (hours)
             | OL argmax accuracy (%) | OL within-5-bin accuracy (%) | OL val loss
             | CL self-play route progress | CL self-play score
             | CL human-replay route progress | CL human-replay score

    Within-5-bin accuracy is the final-step value averaged across dx/dy/dyaw
    from the wandb training-curve CSVs. Pass `within5_csvs=None` to skip the
    column entirely (e.g. if the CSVs aren't available on a given run).

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

    # Within-5-bin column. Stored as a fraction (0-1) here so it shares the
    # same `scale=100` formatting path as ol_val_accuracy below.
    if within5_csvs:
        w5 = _load_final_within5_accuracy(*within5_csvs)
        if not w5.empty:
            w5 = w5.rename(columns={"within5_avg_pct": "ol_within5_acc"}).copy()
            w5["ol_within5_acc"] = w5["ol_within5_acc"] / 100.0
            ol = ol.merge(w5[["num_maps_trained", "ol_within5_acc"]], on="num_maps_trained", how="left")
        else:
            ol["ol_within5_acc"] = np.nan
    else:
        ol["ol_within5_acc"] = np.nan

    has_within5 = ol["ol_within5_acc"].notna().any()

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

    def _intensity(val, col_vals, higher_is_better=True):
        finite = col_vals.dropna()
        if finite.empty or np.isnan(val):
            return 0
        vmin, vmax = finite.min(), finite.max()
        if vmax == vmin:
            return 25
        t = (val - vmin) / (vmax - vmin)
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
    if has_within5:
        best["ol_within5_acc"] = merged["ol_within5_acc"].max()

    def _is_best(col, val):
        if col not in best:
            return False
        return not np.isnan(val) and np.isclose(val, best[col])

    n_ol = 3 if has_within5 else 2
    col_spec = "r|" + "r" * n_ol + "|rr|rr"

    ol_header_cells = [r"Acc. (\%)"]
    if has_within5:
        ol_header_cells.append(r"Acc. $\pm 5$ bins (\%)")
    ol_header_cells.append("Loss")

    lines = [
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor}, \usepackage{graphicx}, \usepackage{bm}",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{BC anchor evaluation. Open-loop metrics on the held-out validation set; "
        r"closed-loop metrics averaged over validation scenes. "
        r"Within-5-bin accuracy is the average of $\Delta x$, $\Delta y$, "
        r"$\Delta\mathrm{yaw}$ accuracies at the final training step.}",
        r"\label{tab:anchor_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r" & \multicolumn{" + str(n_ol) + r"}{c|}{Open-loop} & "
        r"\multicolumn{2}{c|}{Closed-loop self-play} & "
        r"\multicolumn{2}{c}{Closed-loop human-replay (SDC only)} \\",
        r"Human data (h) & " + " & ".join(ol_header_cells) + r" & Route prog. & Score & Route prog. & Score \\",
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
        if has_within5:
            cells.append(
                _fmt_green(
                    row["ol_within5_acc"],
                    "ol_within5_acc",
                    scale=100,
                    decimals=1,
                    is_best=_is_best("ol_within5_acc", row["ol_within5_acc"]),
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
    """How much data is needed for human-compatible policies?

    4 subplots, all using scaling modes; x-axis is self-play training maps
    on a log scale; y-axes in percent.

    Colour convention (from module-level PALETTE via _reg_unreg_colors):
      - unregularized runs: black
      - regularized runs:   shades of blue (darker = more anchor data)
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

    # Convert to percentages (must come BEFORE computing delta so both sides
    # of the subtraction are in the same units).
    agg["hr_atfault"] = agg["hr_atfault"] * 100
    agg["sp_atfault"] = agg["sp_atfault"] * 100
    agg["hr_collision_rate"] = agg["hr_collision_rate"] * 100
    agg["sp_collision_rate"] = agg["sp_collision_rate"] * 100
    agg["delta_atfault"] = agg["sp_atfault"] - agg["hr_atfault"]

    anchor_vals = sorted(agg["anchor_maps"].unique())
    color_map = _reg_unreg_colors(anchor_vals)  # black for 0, blues for >0

    markers = ["s", "^", "o", "D", "P", "X", "v", "*"]
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

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

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
                linewidth=1.5,
                linestyle="-",
                # Thin black edge on light blues so the lightest shades stay
                # visible against white; no edge needed for the black unreg.
                markeredgecolor="black" if anchor > 0 else color_map[anchor],
                markeredgewidth=0.4 if anchor > 0 else 0,
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
            ax.axhline(0, color=PALETTE["tier_best"], linestyle="--", linewidth=2.0, alpha=0.9, zorder=2)
            ax.text(
                min(agg["sp_maps"]) * 1.1,
                0.2,
                "No ZSC gap",
                fontsize=14,
                color=PALETTE["tier_best"],
                ha="left",
                va="bottom",
            )
        else:
            ax.set_ylim(bottom=0)

        ax.legend(fontsize=9, loc="best", framealpha=1.0, facecolor="white", edgecolor="lightgray")
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


def _load_smart_baseline(csv_path: str = "results/smart_baseline_32_rollouts.csv") -> pd.DataFrame:
    """Load the SMART baseline CSV and pivot into per-num_maps rows.

    Returns a DataFrame with columns:
        num_maps, minutes,
        hr_atfault_pct, hr_coll_pct, hr_offroad_pct, hr_progress_pct,
        sp_coll_pct, sp_offroad_pct, sp_progress_pct

    Missing metrics are left as NaN so downstream `dropna(subset=[col])` skips
    them cleanly. Returns an empty (but correctly-typed) DataFrame if the CSV
    isn't present.
    """
    cols = [
        "num_maps",
        "minutes",
        "hr_atfault_pct",
        "hr_coll_pct",
        "hr_offroad_pct",
        "hr_progress_pct",
        "sp_coll_pct",
        "sp_offroad_pct",
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

    # Build HR and SP pivots from the columns that are actually present in the
    # CSV. `offroad_rate` is optional and will only show up if recorded.
    hr_wanted = {
        "at_fault_collision_rate": "hr_atfault_pct",
        "collision_rate": "hr_coll_pct",
        "offroad_rate": "hr_offroad_pct",
        "route_progress": "hr_progress_pct",
    }
    sp_wanted = {
        "collision_rate": "sp_coll_pct",
        "offroad_rate": "sp_offroad_pct",
        "route_progress": "sp_progress_pct",
    }

    hr_raw = raw[raw["mode"] == "scaling_hr_val"]
    hr_present = [c for c in hr_wanted if c in hr_raw.columns]
    hr = hr_raw.set_index("num_maps")[hr_present].rename(columns={c: hr_wanted[c] for c in hr_present})

    sp_raw = raw[raw["mode"] == "scaling_sp_val"]
    sp_present = [c for c in sp_wanted if c in sp_raw.columns]
    sp = sp_raw.set_index("num_maps")[sp_present].rename(columns={c: sp_wanted[c] for c in sp_present})

    # Fractions → percentages.
    out = hr.join(sp, how="outer") * 100
    out = out.reset_index().sort_values("num_maps").reset_index(drop=True)
    out["minutes"] = out["num_maps"] * 9 / 60

    # Re-index to the full expected column list so any metric the CSV is
    # missing surfaces as NaN rather than raising a KeyError.
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
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

    tick_positions = [10, 30, 180, 75000]
    tick_labels = ["10 min", "30 min", "3 hours", "52 days"]

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
    fig_lines, line_axes = plt.subplots(1, 3, figsize=(15, 4.5))

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
            ax.set_ylim(50, 102)
        ax.set_xlabel("Human demonstration data")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, loc="best", framealpha=1.0, facecolor="white", edgecolor="lightgray")
        sns.despine(ax=ax)

    fig_lines.tight_layout()
    _ensure_dir(save_path)
    fig_lines.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")

    # ── FIGURE 2: relative-improvement bars ─────────────────────────────────
    fig_gains, gain_axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

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

    Row layout:
      1) SMART rows, sorted by increasing human data.
      2) A single unregularized self-play row (the 50k unreg checkpoint —
         same checkpoint the plot's dashed baseline uses).
      3) Regularized self-play rows, sorted by increasing anchor data.

    Columns: self-play block (Coll., Off-road, Route prog.) and human-replay
    block (Coll., At-fault, Off-road, Route prog.). Off-road columns are
    emitted only if at least one row has non-NaN data for them.

    Top-3 values per metric column are highlighted via the shared tier palette;
    best value additionally bolded. Ties share a tier.
    """
    SMART_DATA = _load_smart_baseline(smart_csv)

    scaling_modes = ["scaling_sp_val", "scaling_hr_val"]
    df = df[df["mode"].isin(scaling_modes)].copy()
    df = df[df["sp_maps"] == 50000]
    if df.empty:
        print("  No 50k sp_maps data — skipping generate_human_data_latex_table.")
        return None

    df["anchor_maps"] = df["anchor_maps"].fillna(0).astype(int)

    # Aggregate both modes, keeping anchor=0 this time so the unreg row survives.
    hr = df[df["mode"] == "scaling_hr_val"]
    hr_metrics = ["at_fault_collision_rate", "collision_rate", "route_progress"]
    if "offroad_rate" in hr.columns:
        hr_metrics.append("offroad_rate")
    hr_agg = hr.groupby("anchor_maps")[hr_metrics].agg(["mean", "sem"]).reset_index()
    hr_short = {
        "at_fault_collision_rate": "hr_atfault",
        "collision_rate": "hr_coll",
        "route_progress": "hr_progress",
        "offroad_rate": "hr_offroad",
    }
    hr_col_names = ["anchor_maps"]
    for m in hr_metrics:
        short = hr_short[m]
        hr_col_names.extend([f"{short}_mean", f"{short}_sem"])
    hr_agg.columns = hr_col_names

    sp = df[df["mode"] == "scaling_sp_val"]
    sp_metrics = ["collision_rate", "route_progress"]
    if "offroad_rate" in sp.columns:
        sp_metrics.append("offroad_rate")
    sp_agg = sp.groupby("anchor_maps")[sp_metrics].agg(["mean", "sem"]).reset_index()
    sp_short = {
        "collision_rate": "sp_coll",
        "route_progress": "sp_progress",
        "offroad_rate": "sp_offroad",
    }
    sp_col_names = ["anchor_maps"]
    for m in sp_metrics:
        short = sp_short[m]
        sp_col_names.extend([f"{short}_mean", f"{short}_sem"])
    sp_agg.columns = sp_col_names

    full_agg = hr_agg.merge(sp_agg, on="anchor_maps", how="outer")
    has_sp_offroad = "sp_offroad_mean" in full_agg.columns
    has_hr_offroad = "hr_offroad_mean" in full_agg.columns

    pct_cols = ["hr_atfault", "hr_coll", "hr_progress", "sp_coll", "sp_progress"]
    if has_sp_offroad:
        pct_cols.append("sp_offroad")
    if has_hr_offroad:
        pct_cols.append("hr_offroad")
    for col in pct_cols:
        full_agg[f"{col}_mean_pct"] = full_agg[f"{col}_mean"] * 100
        full_agg[f"{col}_sem_pct"] = full_agg[f"{col}_sem"] * 100

    unreg_agg = full_agg[full_agg["anchor_maps"] == 0].copy()
    reg_agg = full_agg[full_agg["anchor_maps"] > 0].copy()
    reg_agg["minutes"] = reg_agg["anchor_maps"] * 9 / 60

    # ── Build row structure ─────────────────────────────────────────────────
    rows = []

    # (1) SMART rows
    for _, r in SMART_DATA.sort_values("minutes").reset_index(drop=True).iterrows():
        if abs(r["minutes"] - 1800) / 1800 < 0.02:  # skip 30h
            continue
        rows.append(
            {
                "method": "SMART",
                "minutes": r["minutes"],
                "sp_coll_mean": r["sp_coll_pct"],
                "sp_coll_sem": np.nan,
                "sp_progress_mean": r["sp_progress_pct"],
                "sp_progress_sem": np.nan,
                "sp_offroad_mean": r["sp_offroad_pct"] if "sp_offroad_pct" in r else np.nan,
                "sp_offroad_sem": np.nan,
                "hr_coll_mean": r["hr_coll_pct"],
                "hr_coll_sem": np.nan,
                "hr_atfault_mean": r["hr_atfault_pct"],
                "hr_atfault_sem": np.nan,
                "hr_progress_mean": r["hr_progress_pct"],
                "hr_progress_sem": np.nan,
                "hr_offroad_mean": r["hr_offroad_pct"] if "hr_offroad_pct" in r else np.nan,
                "hr_offroad_sem": np.nan,
            }
        )

    # (2) Single unregularized self-play row (the 50k unreg checkpoint).
    if not unreg_agg.empty:
        u = unreg_agg.iloc[0]
        rows.append(
            {
                "method": "unreg. self-play",
                "minutes": np.nan,  # no human data used
                "sp_coll_mean": u["sp_coll_mean_pct"],
                "sp_coll_sem": u["sp_coll_sem_pct"],
                "sp_progress_mean": u["sp_progress_mean_pct"],
                "sp_progress_sem": u["sp_progress_sem_pct"],
                "sp_offroad_mean": u["sp_offroad_mean_pct"] if has_sp_offroad else np.nan,
                "sp_offroad_sem": u["sp_offroad_sem_pct"] if has_sp_offroad else np.nan,
                "hr_coll_mean": u["hr_coll_mean_pct"],
                "hr_coll_sem": u["hr_coll_sem_pct"],
                "hr_atfault_mean": u["hr_atfault_mean_pct"],
                "hr_atfault_sem": u["hr_atfault_sem_pct"],
                "hr_progress_mean": u["hr_progress_mean_pct"],
                "hr_progress_sem": u["hr_progress_sem_pct"],
                "hr_offroad_mean": u["hr_offroad_mean_pct"] if has_hr_offroad else np.nan,
                "hr_offroad_sem": u["hr_offroad_sem_pct"] if has_hr_offroad else np.nan,
            }
        )

    # (3) Regularized rows, sorted by minutes
    for _, r in reg_agg.sort_values("minutes").reset_index(drop=True).iterrows():
        rows.append(
            {
                "method": "regularized self-play",
                "minutes": r["minutes"],
                "sp_coll_mean": r["sp_coll_mean_pct"],
                "sp_coll_sem": r["sp_coll_sem_pct"],
                "sp_progress_mean": r["sp_progress_mean_pct"],
                "sp_progress_sem": r["sp_progress_sem_pct"],
                "sp_offroad_mean": r["sp_offroad_mean_pct"] if has_sp_offroad else np.nan,
                "sp_offroad_sem": r["sp_offroad_sem_pct"] if has_sp_offroad else np.nan,
                "hr_coll_mean": r["hr_coll_mean_pct"],
                "hr_coll_sem": r["hr_coll_sem_pct"],
                "hr_atfault_mean": r["hr_atfault_mean_pct"],
                "hr_atfault_sem": r["hr_atfault_sem_pct"],
                "hr_progress_mean": r["hr_progress_mean_pct"],
                "hr_progress_sem": r["hr_progress_sem_pct"],
                "hr_offroad_mean": r["hr_offroad_mean_pct"] if has_hr_offroad else np.nan,
                "hr_offroad_sem": r["hr_offroad_sem_pct"] if has_hr_offroad else np.nan,
            }
        )

    table = pd.DataFrame(rows)

    def _fmt_minutes(minutes):
        if pd.isna(minutes):
            return "---"  # unreg row: no human demonstrations
        if minutes < 60:
            return f"{int(round(minutes))} min"
        hours = minutes / 60
        # Keep "hours" format up through ~2 days so 30h reads as "30 hours",
        # not "1.2 days". Only genuinely multi-day amounts use the days format.
        if hours < 48:
            if hours == int(hours):
                return f"{int(hours)} hours"
            return f"{hours:.1f} hours"
        days = hours / 24
        if days == int(days):
            return f"{int(days)} days"
        return f"{days:.1f} days"

    table["human_data_label"] = table["minutes"].apply(_fmt_minutes)

    # ── Metric metadata ──────────────────────────────────────────────────────
    # Off-road columns are included only if at least one row has non-NaN data.
    any_sp_offroad = "sp_offroad_mean" in table.columns and table["sp_offroad_mean"].notna().any()
    any_hr_offroad = "hr_offroad_mean" in table.columns and table["hr_offroad_mean"].notna().any()

    # (mean, sem, header, higher_is_better, as_pct, decimals)
    sp_specs = [("sp_coll_mean", "sp_coll_sem", r"Coll. (\%) $\downarrow$", False, False, 1)]
    if any_sp_offroad:
        sp_specs.append(("sp_offroad_mean", "sp_offroad_sem", r"Off-road (\%) $\downarrow$", False, False, 1))
    sp_specs.append(("sp_progress_mean", "sp_progress_sem", r"Route prog. (\%) $\uparrow$", True, False, 1))

    hr_specs = [
        ("hr_coll_mean", "hr_coll_sem", r"Coll. (\%) $\downarrow$", False, False, 1),
        ("hr_atfault_mean", "hr_atfault_sem", r"At-fault (\%) $\downarrow$", False, False, 1),
    ]
    if any_hr_offroad:
        hr_specs.append(("hr_offroad_mean", "hr_offroad_sem", r"Off-road (\%) $\downarrow$", False, False, 1))
    hr_specs.append(("hr_progress_mean", "hr_progress_sem", r"Route prog. (\%) $\uparrow$", True, False, 1))

    all_specs = sp_specs + hr_specs
    n_sp_cols = len(sp_specs)
    n_hr_cols = len(hr_specs)

    rank_lookup = _build_tier_rank_lookup(table, all_specs)

    def _fmt_cell(mean, sem, mean_col, row_idx, as_pct, decimals):
        if pd.isna(mean):
            return "---"
        tier = rank_lookup.get((mean_col, row_idx))
        is_best = tier == 1
        m_val = mean * 100 if as_pct else mean
        s_val = sem * 100 if (as_pct and pd.notna(sem)) else sem
        fmt = f".{decimals}f"
        if pd.notna(s_val) and s_val != 0:
            body = f"{m_val:{fmt}} \\pm {s_val:{fmt}}"
            text = f"$\\bm{{{body}}}$" if is_best else f"${body}$"
        else:
            body = f"{m_val:{fmt}}"
            text = f"\\textbf{{{body}}}" if is_best else body
        if tier is None:
            return text
        return f"\\cellcolor{{{_TIER_NAMES[tier]}}} {text}"

    # ── Build LaTeX ──────────────────────────────────────────────────────────
    col_spec = "ll" + "|" + "r" * n_sp_cols + "|" + "r" * n_hr_cols

    lines = []
    lines.append(
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor}, "
        r"\usepackage{graphicx}, \usepackage{makecell}, \usepackage{bm}"
    )
    lines.extend(_tier_latex_preamble())
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Performance vs.\ amount of human demonstration data at fixed 50k "
        r"self-play training maps. Top-3 values per column are highlighted "
        r"(\colorbox{tierbest}{best}, \colorbox{tiersecond}{2nd}, "
        r"\colorbox{tierthird}{3rd}); best value additionally in bold. "
        r"The unregularized self-play row uses no human demonstrations.}"
    )
    lines.append(r"\label{tab:human_data_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(
        r" & & "
        + r"\multicolumn{"
        + str(n_sp_cols)
        + r"}{c|}{Self-play (test)} & "
        + r"\multicolumn{"
        + str(n_hr_cols)
        + r"}{c}{Human-replay (test)}"
        + r" \\"
    )

    headers = [s[2] for s in all_specs]
    header_row = (
        r"\makecell{Human demos \\ used} & Method & "
        + " & ".join(headers[:n_sp_cols])
        + " & "
        + " & ".join(headers[n_sp_cols:])
        + r" \\"
    )
    lines.append(header_row)
    lines.append(r"\midrule")

    prev_method = None
    for i, row in table.iterrows():
        if prev_method is not None and row["method"] != prev_method:
            lines.append(r"\midrule")
        prev_method = row["method"]

        method_label = {
            "SMART": "SMART",
            "unreg. self-play": "unreg. self-play",
        }.get(row["method"], "reg. self-play (ours)")

        cells = [row["human_data_label"], method_label]
        for mean_col, sem_col, _h, _hib, as_pct, decimals in all_specs:
            cells.append(_fmt_cell(row[mean_col], row[sem_col], mean_col, i, as_pct, decimals))
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

    # unreg first (black), reg second (blue, matching the shared palette)
    agg["is_reg"] = ~agg["checkpoint"].str.contains("unreg")
    agg = agg.sort_values("is_reg").reset_index(drop=True)
    colors = [PALETTE["selfplay"] if not r else PALETTE["ours"] for r in agg["is_reg"]]

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
    """Human-replay: random-validation vs interactive-validation comparison table.

    Row layout: unreg first, then reg, each sorted by sp_maps. The two column
    groups (val / interactive) each carry the same metrics and each metric
    column is tier-ranked independently with the Depth-Pro top-3 palette.

    Required LaTeX packages:
      \\usepackage{booktabs}, \\usepackage[table]{xcolor},
      \\usepackage{graphicx}, \\usepackage{makecell}, \\usepackage{bm}
    """
    hr_modes = ["scaling_hr_val", "scaling_hr_interactive"]
    scaling_df = df[df["mode"].isin(hr_modes)].copy()
    if scaling_df.empty:
        print("  No scaling HR data found — skipping generate_hr_comparison_latex_table.")
        return None

    scaling_df["anchor_maps"] = scaling_df["anchor_maps"].fillna(0).astype(int)

    hr_metric_cols = [
        "score",
        "collision_rate",
        "at_fault_collision_rate",
        "rear_collision_rate",
        "route_progress",
        "lateral_error_avg",
    ]
    available = [m for m in hr_metric_cols if m in scaling_df.columns]

    agg = scaling_df.groupby(["sp_maps", "anchor_maps", "mode"])[available].agg(["mean", "sem"]).reset_index()
    flat_cols = ["sp_maps", "anchor_maps", "mode"]
    for m in available:
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

    has_at_fault = "at_fault_collision_rate" in available
    has_rear = "rear_collision_rate" in available
    has_route = "route_progress" in available
    has_lateral = "lateral_error_avg" in available

    def _block_specs(prefix):
        """(mean_col, sem_col, header, higher_is_better, as_pct, decimals) for one block."""
        specs = [
            (f"{prefix}_score_mean", f"{prefix}_score_sem", r"Score $\uparrow$", True, False, 3),
            (
                f"{prefix}_collision_rate_mean",
                f"{prefix}_collision_rate_sem",
                r"Coll. (\%) $\downarrow$",
                False,
                True,
                1,
            ),
        ]
        if has_at_fault:
            specs.append(
                (
                    f"{prefix}_at_fault_collision_rate_mean",
                    f"{prefix}_at_fault_collision_rate_sem",
                    r"At-fault (\%) $\downarrow$",
                    False,
                    True,
                    1,
                )
            )
        if has_rear:
            specs.append(
                (
                    f"{prefix}_rear_collision_rate_mean",
                    f"{prefix}_rear_collision_rate_sem",
                    r"Rear coll. (\%) $\downarrow$",
                    False,
                    True,
                    1,
                )
            )
        if has_route:
            specs.append(
                (
                    f"{prefix}_route_progress_mean",
                    f"{prefix}_route_progress_sem",
                    r"Route prog. $\uparrow$",
                    True,
                    False,
                    3,
                )
            )
        if has_lateral:
            specs.append(
                (
                    f"{prefix}_lateral_error_avg_mean",
                    f"{prefix}_lateral_error_avg_sem",
                    r"Lateral L2 $\downarrow$",
                    False,
                    False,
                    2,
                )
            )
        return specs

    val_specs = [s for s in _block_specs("val") if s[0] in merged.columns]
    int_specs = [s for s in _block_specs("int") if s[0] in merged.columns]
    all_specs = val_specs + int_specs
    n_val, n_int = len(val_specs), len(int_specs)

    rank_lookup = _build_tier_rank_lookup(merged, all_specs)

    def _fmt_cell(mean, sem, mean_col, row_idx, as_pct, decimals):
        if pd.isna(mean):
            return "---"
        tier = rank_lookup.get((mean_col, row_idx))
        is_best = tier == 1
        m_val = mean * 100 if as_pct else mean
        s_val = sem * 100 if (as_pct and pd.notna(sem)) else sem
        fmt = f".{decimals}f"
        if pd.notna(s_val) and s_val != 0:
            body = f"{m_val:{fmt}} \\pm {s_val:{fmt}}"
            text = f"$\\bm{{{body}}}$" if is_best else f"${body}$"
        else:
            body = f"{m_val:{fmt}}"
            text = f"\\textbf{{{body}}}" if is_best else body
        if tier is None:
            return text
        return f"\\cellcolor{{{_TIER_NAMES[tier]}}} {text}"

    def _anchor_label(anchor_maps):
        return "0 (unreg.)" if anchor_maps == 0 else _maps_to_human_time(anchor_maps)

    col_spec = "rr" + "|" + "r" * n_val + "|" + "r" * n_int

    lines = []
    lines.append(
        r"% Requires: \usepackage{booktabs}, \usepackage[table]{xcolor}, "
        r"\usepackage{graphicx}, \usepackage{makecell}, \usepackage{bm}"
    )
    lines.extend(_tier_latex_preamble())
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Human-replay performance on the randomly sampled validation "
        r"set vs.\ the interactive validation subset for all scaling "
        r"checkpoints. Top-3 values per column are highlighted "
        r"(\colorbox{tierbest}{best}, \colorbox{tiersecond}{2nd}, "
        r"\colorbox{tierthird}{3rd}); best value additionally in bold.}"
    )
    lines.append(r"\label{tab:hr_comparison_results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append(
        r" & & "
        + r"\multicolumn{"
        + str(n_val)
        + r"}{c|}{HR --- validation (random)} & "
        + r"\multicolumn{"
        + str(n_int)
        + r"}{c}{HR --- interactive}"
        + r" \\"
    )

    headers = [s[2] for s in all_specs]
    header_row = (
        r"\makecell{Self-play maps \\ (metadata)} & "
        r"\makecell{Anchor data \\ (human demonstrations)} & "
        + " & ".join(headers[:n_val])
        + " & "
        + " & ".join(headers[n_val:])
        + r" \\"
    )
    lines.append(header_row)
    lines.append(r"\midrule")

    prev_was_unreg = None
    for i, row in merged.iterrows():
        is_unreg = int(row["anchor_maps"]) == 0
        if prev_was_unreg is True and not is_unreg:
            lines.append(r"\midrule")
        prev_was_unreg = is_unreg

        cells = [_fmt_maps(int(row["sp_maps"])), _anchor_label(int(row["anchor_maps"]))]
        for mean_col, sem_col, _h, _hib, as_pct, decimals in all_specs:
            cells.append(
                _fmt_cell(
                    row.get(mean_col, np.nan),
                    row.get(sem_col, np.nan),
                    mean_col,
                    i,
                    as_pct,
                    decimals,
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
        - regularized self-play (ours):       line across anchor points (medium blue)
        - best unregularized self-play:       horizontal dashed line (black)
        - SMART-tiny-CLSFT:                   line across SMART checkpoints (red)
        - Ground-truth (UB):                  dashed reference line (upper bound)
        - Random:                             dashed reference line (lower bound)

    Reference baselines for Ground-truth and Random are fixed per metric,
    taken from the PufferDrive WOSAC baseline table (229 clean held-out
    validation scenes).
    """
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


def plot_collision_severity(
    df,
    save_path="results/figures/eval_collision_severity.pdf",
    modes=("hr_interactive", "scaling_hr_interactive"),
    sp_maps_filter=50000,
):
    """Compare collision severity (Delta-V) between regularized and unregularized models.

    Three panels:
      0) ECDF of per-event Delta-V — distributional comparison.
         Single-collision agents contribute their exact dv (= delta_v_max).
         Multi-collision agents contribute mean = delta_v_sum / delta_v_count
         and max = delta_v_max as separate proxy points (count is typically
         small in practice; a header note prints the multi-collision share).
         A dashed vertical line marks the Waymo 1 mph (0.447 m/s) threshold.
      1) Bar of mean Delta-V per collision event (conditional on collision).
      2) KDE (density-normalized) of per-event Delta-V, overlaid by
         group. X-axis clipped at the 99th percentile across both groups
         so the long tail doesn't compress the body of the distribution.

    Pulls only rows from `modes` and (optionally) restricts to a single
    metadata-map count via `sp_maps_filter` to match the human-data figure.
    Pass sp_maps_filter=None to use all rows (e.g. main eval CSV without
    scaling metadata).

    Returns the matplotlib Figure, or None if there's not enough data.
    """
    CHECKPOINTS_OF_INTEREST = {
        "models/scaling_cpts/unreg_delta_50k_maps.pt": "unregularized",
        "models/scaling_cpts/reg_delta_50k_maps_anchor_1200_maps.pt": "regularized",
    }

    # ── Filter ──────────────────────────────────────────────────────────────
    sub = df[df["mode"].isin(modes)].copy()
    if sub.empty:
        print(f"  No rows in modes={modes} — skipping plot_collision_severity.")
        return None

    sub = sub[sub["checkpoint"].isin(CHECKPOINTS_OF_INTEREST)].copy()
    if sub.empty:
        print("  No rows for checkpoints of interest — skipping plot_collision_severity.")
        return None

    if sp_maps_filter is not None and "sp_maps" in sub.columns:
        sub = sub[sub["sp_maps"] == sp_maps_filter]
        if sub.empty:
            print(f"  No rows with sp_maps={sp_maps_filter} — skipping plot_collision_severity.")
            return None

    sub["group"] = sub["checkpoint"].map(CHECKPOINTS_OF_INTEREST)

    coll = sub[(sub["delta_v_count"] > 0) & (sub["at_fault_collision_rate"] > 0)].copy()

    if coll.empty:
        print("  No collision events in filtered data — skipping plot_collision_severity.")
        return None

    multi_frac = (coll["delta_v_count"] > 1).mean()
    print(f"  Collision events: {len(coll)} agent-episodes with at least one collision.")
    print(f"  Multi-collision agent-episodes: {multi_frac:.1%}")
    if multi_frac > 0.10:
        print("  Note: multi-collision share > 10%; CDF/histogram are approximate.")

    # ── Build per-event proxy frame ─────────────────────────────────────────
    # Single-collision rows: dv == delta_v_max exactly.
    single = coll[coll["delta_v_count"] == 1][["group", "delta_v_max"]].copy()
    single = single.rename(columns={"delta_v_max": "dv"})

    # Multi-collision rows: emit max and mean as separate proxy events.
    multi = coll[coll["delta_v_count"] > 1].copy()
    multi_max = multi[["group", "delta_v_max"]].rename(columns={"delta_v_max": "dv"})
    multi_mean = multi[["group"]].copy()
    multi_mean["dv"] = multi["delta_v_sum"] / multi["delta_v_count"]

    events = pd.concat([single, multi_max, multi_mean], ignore_index=True)
    events = events[events["dv"] > 0]  # drop any zeros from edge cases

    # ── Headline aggregates (used in panels 1 and 2) ────────────────────────
    def _agg(group_df):
        total_sum = group_df["delta_v_sum"].sum()
        total_count = group_df["delta_v_count"].sum()
        if total_count == 0:
            return pd.Series({"mean_dv": np.nan, "frac_under_1mph": np.nan, "n_events": 0})
        # frac_under_1mph: per-agent indicator (1 if agent's worst collision
        # was under 1 mph). This is the cleanest agent-level summary, even
        # though a strictly per-event version would require a C-side log.
        return pd.Series(
            {
                "mean_dv": total_sum / total_count,
                "frac_under_1mph": group_df["delta_v_under_1mph"].mean(),
                "n_events": int(total_count),
            }
        )

    headline = coll.groupby("group").apply(_agg).reset_index()

    # Stable ordering: unreg first, reg second (matches rest of file)
    group_order = ["unregularized", "regularized"]
    headline = headline.set_index("group").reindex(group_order).reset_index()
    colors = [PALETTE["selfplay"], PALETTE["ours"]]

    # ── Plot ────────────────────────────────────────────────────────────────
    _set_style(2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    one_mph_mps = 0.447  # 1 mph = 0.447 m/s, Waymo's threshold

    # Panel 0: ECDF
    ax = axes[0]
    for group, color in zip(group_order, colors):
        dv = np.sort(events.loc[events["group"] == group, "dv"].values)
        if dv.size == 0:
            continue
        ecdf = np.arange(1, dv.size + 1) / dv.size
        ax.plot(
            dv,
            ecdf,
            color=color,
            linewidth=2.0,
            label=f"{group} (n={dv.size})",
            zorder=3,
        )
    ax.axvline(
        one_mph_mps,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        zorder=2,
        label="1 mph (0.45 m/s)",
    )
    ax.set_xlabel(r"Per-event $\Delta v$ (m/s)")
    ax.set_ylabel("Cumulative fraction of events")
    ax.set_title("Distribution of collision severity")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="lower right", framealpha=1.0, facecolor="white", edgecolor="lightgray")
    sns.despine(ax=ax)

    # Panel 1: mean Delta-V per event
    ax = axes[1]
    x = np.arange(len(group_order))
    means = headline["mean_dv"].values
    ax.bar(x, means, color=colors, alpha=0.85, width=0.55)
    for xi, m in zip(x, means):
        if pd.notna(m):
            ax.text(xi, m, f"{m:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=15, ha="right")
    ax.set_ylabel(r"Mean $\Delta v$ per collision (m/s)")
    ax.set_title(r"Mean severity $\downarrow$")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    # Panel 2: KDE of per-event Delta-V, overlaid by group.
    # KDE leaks a small amount of mass past the dv >= 0 boundary (kernel
    # smoothing artifact); we clip the x-axis at 0 to hide it. Bandwidth
    # uses seaborn's default (Scott's rule), which is fine for n in 100s+.
    ax = axes[2]
    all_dvs = events["dv"].values
    if all_dvs.size > 0:
        x_hi = np.percentile(all_dvs, 99)
    else:
        x_hi = 1.0
    x_hi = float(np.ceil(x_hi * 2) / 2)  # nearest 0.5

    for group, color in zip(group_order, colors):
        dv = events.loc[events["group"] == group, "dv"].values
        if dv.size < 2:  # KDE needs at least 2 points
            continue
        sns.kdeplot(
            x=dv,
            ax=ax,
            color=color,
            fill=True,
            alpha=0.35,
            linewidth=2.0,
            label=f"{group} (n={dv.size})",
            clip=(0, None),  # don't extrapolate to negative Delta-V
            zorder=3,
        )
    ax.axvline(
        one_mph_mps,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        zorder=2,
        label="1 mph",
    )
    ax.set_xlabel(r"Per-event $\Delta v$ (m/s)")
    ax.set_ylabel("Density")
    ax.set_title(rf"Severity density (KDE, view clipped at {x_hi:.1f} m/s)")
    ax.set_xlim(0, x_hi)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9, loc="upper right", framealpha=1.0, facecolor="white", edgecolor="lightgray")
    sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def generate_collision_severity_latex_table(
    df,
    save_path="results/figures/eval_collision_severity_table.tex",
    modes=("hr_interactive", "scaling_hr_interactive"),
    sp_maps_filter=50000,
):
    """LaTeX table for the collision severity tail breakdown at 50k metadata maps.

    Companion to plot_collision_severity. For each group (regularized vs
    unregularized) reports:
      - Total collision events
      - Mean and max per-event Delta-V (m/s)
      - % of events above injury-risk thresholds (1, 5, 10, 15, 20 mph)

    Best value per metric column is bolded. For severity columns, lower
    is better.

    Required LaTeX packages:
      \\usepackage{booktabs}, \\usepackage{graphicx},
      \\usepackage{makecell}, \\usepackage{bm}
    """
    CHECKPOINTS_OF_INTEREST = {
        "models/scaling_cpts/unreg_delta_50k_maps.pt": "unregularized",
        "models/scaling_cpts/reg_delta_50k_maps_anchor_1200_maps.pt": "regularized",
    }

    # ── Filter ──────────────────────────────────────────────────────────────
    sub = df[df["mode"].isin(modes)].copy()
    if sub.empty:
        print(f"  No rows in modes={modes} — skipping plot_collision_severity.")
        return None

    sub = sub[sub["checkpoint"].isin(CHECKPOINTS_OF_INTEREST)].copy()
    if sub.empty:
        print("  No rows for checkpoints of interest — skipping plot_collision_severity.")
        return None

    if sp_maps_filter is not None and "sp_maps" in sub.columns:
        sub = sub[sub["sp_maps"] == sp_maps_filter]
        if sub.empty:
            print(f"  No rows with sp_maps={sp_maps_filter} — skipping plot_collision_severity.")
            return None

    sub["group"] = sub["checkpoint"].map(CHECKPOINTS_OF_INTEREST)

    coll = sub[(sub["delta_v_count"] > 0) & (sub["at_fault_collision_rate"] > 0)].copy()
    if coll.empty:
        print("  No collision events — skipping severity table.")
        return None

    # Per-event proxy frame (same construction as plot_collision_severity).
    single = coll[coll["delta_v_count"] == 1][["group", "delta_v_max"]].copy()
    single = single.rename(columns={"delta_v_max": "dv"})
    multi = coll[coll["delta_v_count"] > 1].copy()
    multi_max = multi[["group", "delta_v_max"]].rename(columns={"delta_v_max": "dv"})
    multi_mean = multi[["group"]].copy()
    multi_mean["dv"] = multi["delta_v_sum"] / multi["delta_v_count"]
    events = pd.concat([single, multi_max, multi_mean], ignore_index=True)
    events = events[events["dv"] > 0]

    # Severity thresholds (matches stdout printout in plot_collision_severity).
    # 1 mph: cosmetic / Waymo "minor" floor.
    # 5 mph: typical airbag-deployment threshold.
    # 10 mph: MAIS2+ injury risk begins climbing in V2V crashes.
    # 15 mph: moderate; MAIS2+ ~15-25%.
    # 20 mph: elevated serious-injury risk.
    thresholds = [
        (r"$> 1$ mph", 0.447),
        (r"$> 5$ mph", 2.235),
        (r"$> 10$ mph", 4.470),
        (r"$> 15$ mph", 6.706),
        (r"$> 20$ mph", 8.941),
    ]

    # Aggregate per group.
    rows = []
    for group in ["unregularized", "regularized"]:  # stable order
        dv = events.loc[events["group"] == group, "dv"].values
        if dv.size == 0:
            continue
        row = {
            "group": group,
            "n_events": int(dv.size),
            "mean_dv": float(dv.mean()),
            "max_dv": float(dv.max()),
        }
        for _label, thresh in thresholds:
            row[f"pct_{thresh:.3f}"] = float((dv > thresh).mean() * 100.0)
        rows.append(row)

    if not rows:
        print("  No populated severity rows — skipping table.")
        return None

    table = pd.DataFrame(rows)

    # Column specs. The 4th tuple element (higher_is_better) drives bolding
    # direction for each column — False means "lower wins", True "higher wins".
    # (mean_col, sem_col, header, higher_is_better, as_pct, decimals)
    metric_specs = [
        ("mean_dv", None, r"Mean $\Delta v$ (m/s) $\downarrow$", False, False, 2),
        ("max_dv", None, r"Max $\Delta v$ (m/s) $\downarrow$", False, False, 2),
    ]
    for label, thresh in thresholds:
        col = f"pct_{thresh:.3f}"
        metric_specs.append((col, None, label + r" (\%) $\downarrow$", False, False, 1))

    # Best-per-column for bolding. Ties share "best".
    best_per_col = {}
    for spec in metric_specs:
        col = spec[0]
        higher_is_better = spec[3]
        vals = table[col].dropna()
        if vals.empty:
            best_per_col[col] = None
            continue
        best_per_col[col] = vals.max() if higher_is_better else vals.min()

    def _fmt_cell(val, mean_col, row_idx, decimals):
        if pd.isna(val):
            return "---"
        target = best_per_col.get(mean_col)
        is_best = target is not None and np.isclose(val, target)
        fmt = f".{decimals}f"
        body = f"{val:{fmt}}"
        return f"\\textbf{{{body}}}" if is_best else body

    n_metric_cols = len(metric_specs)
    col_spec = "l|r|" + "r" * n_metric_cols

    lines = []
    lines.append(
        r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, "
        r"\usepackage{makecell}, \usepackage{bm}"
    )
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Collision severity tail breakdown at 50k metadata maps "
        r"(human-replay evaluation). For each group we report the mean and "
        r"max per-event $\Delta v$, and the percentage of collision events "
        r"exceeding common injury-risk thresholds ($1$ mph: cosmetic; "
        r"$5$ mph: typical airbag-deployment floor; $10$+ mph: MAIS2+ "
        r"injury risk begins climbing). Best value per column in bold. "
        r"All severity metrics: lower is better.}"
    )
    lines.append(r"\label{tab:collision_severity}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    headers = ["Method", "Events"] + [s[2] for s in metric_specs]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for i, row in table.iterrows():
        cells = [row["group"], f"{int(row['n_events'])}"]
        for spec in metric_specs:
            mean_col = spec[0]
            decimals = spec[5]
            cells.append(_fmt_cell(row[mean_col], mean_col, i, decimals))
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


def plot_three_metric_comparison(df, save_path="results/figures/eval_three_metric_comparison.pdf"):
    """Bar chart comparing unregularized vs regularized across three key collision metrics.

    3 subplots (1 row):
      0) Self-play collision rate         (mode: scaling_sp_val)
      1) IDM collision rate               (mode: scaling_idm_interactive)
      2) Human-replay at-fault collision  (mode: scaling_hr_interactive)

    Two bars per subplot: unregularized (black) vs regularized (blue),
    following the shared PALETTE reg/unreg colour convention.
    Error bars show SEM. Values plotted as percentages.

    Note: scaling_idm_interactive rows do not have sp_maps/anchor_maps
    attached in the evaluator (bug in evaluate_checkpoints.py — the metadata
    loop omits idm_interactive_rows). Filtering is therefore done on
    checkpoint path only, not on sp_maps.
    """
    CHECKPOINTS_OF_INTEREST = {
        "models/scaling_cpts/unreg_delta_50k_maps.pt": "unregularized",
        "models/scaling_cpts/reg_delta_50k_maps_anchor_200_maps.pt": "regularized",
    }

    subplot_specs = [
        {
            "mode": "scaling_sp_val",
            "col": "collision_rate",
            "ylabel": "Collision rate [%]",
            "title": "Self-play collision rate",
            "filter_sp_maps": True,  # sp_maps is reliably set for this mode
        },
        {
            "mode": "scaling_idm_interactive",
            "col": "collision_rate",
            "ylabel": "Collision rate [%]",
            "title": "IDM collision rate",
            "filter_sp_maps": False,  # sp_maps NOT attached to IDM rows (evaluator bug)
        },
        {
            "mode": "scaling_hr_interactive",
            "col": "at_fault_collision_rate",
            "ylabel": "At-fault collision rate [%]",
            "title": "HR at-fault collision rate",
            "filter_sp_maps": True,
        },
    ]

    required_cols = {"collision_rate", "at_fault_collision_rate"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  Missing columns {missing} — skipping plot_three_metric_comparison.")
        return None

    _set_style(2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    bar_labels = ["unregularized", "regularized"]
    colors = [PALETTE["selfplay"], PALETTE["ours"]]
    x = np.arange(len(bar_labels))

    for ax, spec in zip(axes, subplot_specs):
        sub = df[df["mode"] == spec["mode"]].copy()
        sub = sub[sub["checkpoint"].isin(CHECKPOINTS_OF_INTEREST)].copy()

        # For modes where sp_maps is reliably set, restrict to the 50k checkpoint
        # to avoid accidentally pulling in rows from other sp_maps values that
        # happen to share the same checkpoint path (shouldn't occur given the
        # naming convention, but is an explicit safeguard).
        if spec["filter_sp_maps"] and "sp_maps" in sub.columns:
            sub = sub[sub["sp_maps"] == 50_000]

        if sub.empty:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=11,
                color="gray",
            )
            ax.set_title(spec["title"])
            ax.set_ylabel(spec["ylabel"])
            sns.despine(ax=ax)
            continue

        agg = sub.groupby("checkpoint")[spec["col"]].agg(mean="mean", sem="sem").reset_index()
        agg["label"] = agg["checkpoint"].map(CHECKPOINTS_OF_INTEREST)
        agg["is_reg"] = ~agg["checkpoint"].str.contains("unreg")
        agg = agg.sort_values("is_reg").reset_index(drop=True)

        means = agg["mean"].values * 100
        sems = agg["sem"].values * 100

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
        ax.set_xticklabels(bar_labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(spec["ylabel"])
        ax.set_title(spec["title"])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.tick_params(axis="y", which="minor", length=3, color="gray")
        sns.despine(ax=ax)

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    return fig


def generate_main_comparison_latex_table(
    df,
    save_path="results/figures/eval_main_comparison_table.tex",
):
    """LaTeX table comparing two checkpoints across self-play, HR, and IDM metrics.

    Rows:    unregularized (first), regularized (second).
    Columns: SP score | HR score | IDM score | HR at-fault coll. |
             HR longitudinal L2 | HR lateral L2 | HR displacement error

    Best value per column is bolded. No colour coding.

    Note: scaling_idm_interactive rows do not have sp_maps attached (evaluator
    bug — see evaluate_checkpoints.py). IDM filtering is by checkpoint only.

    Required LaTeX packages:
      \\usepackage{booktabs}, \\usepackage{graphicx}, \\usepackage{bm}
    """
    CHECKPOINTS_OF_INTEREST = {
        "models/scaling_cpts/unreg_delta_50k_maps.pt": "Unregularized",
        "models/scaling_cpts/reg_delta_50k_maps_anchor_200_maps.pt": "Regularized (ours)",
    }
    ROW_ORDER = [
        "models/scaling_cpts/unreg_delta_50k_maps.pt",
        "models/scaling_cpts/reg_delta_50k_maps_anchor_200_maps.pt",
    ]

    # ── Per-mode aggregation ────────────────────────────────────────────────
    # (mode, col, output_key, filter_sp_maps)
    metric_sources = [
        ("scaling_sp_val", "score", "sp_score", True),
        ("scaling_hr_interactive", "score", "hr_score", True),
        ("scaling_idm_interactive", "score", "idm_score", False),
        ("scaling_hr_interactive", "at_fault_collision_rate", "hr_atfault", True),
        ("scaling_hr_interactive", "longitudinal_error_avg", "hr_long_err", True),
        ("scaling_hr_interactive", "lateral_error_avg", "hr_lat_err", True),
        ("scaling_hr_interactive", "displacement_error_avg", "hr_disp_err", True),
    ]

    # Collect mean ± sem per checkpoint for each metric
    records = {cpt: {} for cpt in CHECKPOINTS_OF_INTEREST}

    for mode, col, key, filter_sp_maps in metric_sources:
        if col not in df.columns:
            print(f"  Warning: column '{col}' not found — '{key}' will show as ---.")
            for cpt in CHECKPOINTS_OF_INTEREST:
                records[cpt][f"{key}_mean"] = float("nan")
                records[cpt][f"{key}_sem"] = float("nan")
            continue

        sub = df[(df["mode"] == mode) & (df["checkpoint"].isin(CHECKPOINTS_OF_INTEREST))].copy()
        if filter_sp_maps and "sp_maps" in sub.columns:
            sub = sub[sub["sp_maps"] == 50_000]

        for cpt in CHECKPOINTS_OF_INTEREST:
            grp = sub.loc[sub["checkpoint"] == cpt, col].dropna()
            records[cpt][f"{key}_mean"] = grp.mean() if not grp.empty else float("nan")
            records[cpt][f"{key}_sem"] = grp.sem() if len(grp) > 1 else float("nan")

    # ── Column specs ────────────────────────────────────────────────────────
    # (key, header, higher_is_better, as_pct, decimals)
    col_specs = [
        ("sp_score", r"Score $\uparrow$", True, False, 3),
        ("hr_score", r"Score $\uparrow$", True, False, 3),
        ("idm_score", r"Score $\uparrow$", True, False, 3),
        ("hr_atfault", r"At-fault (\%) $\downarrow$", False, True, 1),
        ("hr_long_err", r"Long. L2 $\downarrow$", False, False, 3),
        ("hr_lat_err", r"Lat. L2 $\downarrow$", False, False, 3),
        ("hr_disp_err", r"Disp. err. $\downarrow$", False, False, 3),
    ]

    # ── Best-per-column for bolding ─────────────────────────────────────────
    best = {}
    for key, _, higher_is_better, _, _ in col_specs:
        vals = [records[cpt][f"{key}_mean"] for cpt in ROW_ORDER]
        finite = [v for v in vals if not np.isnan(v)]
        if not finite:
            best[key] = None
        elif higher_is_better:
            best[key] = max(finite)
        else:
            best[key] = min(finite)

    # ── Cell formatter ──────────────────────────────────────────────────────
    def _fmt_cell(key, cpt, as_pct, decimals):
        mean = records[cpt][f"{key}_mean"]
        sem = records[cpt][f"{key}_sem"]
        if np.isnan(mean):
            return "---"
        is_best = best[key] is not None and np.isclose(mean, best[key])
        m_val = mean * 100 if as_pct else mean
        s_val = sem * 100 if (as_pct and not np.isnan(sem)) else sem
        fmt = f".{decimals}f"
        if not np.isnan(s_val):
            body = f"{m_val:{fmt}} \\pm {s_val:{fmt}}"
            text = f"$\\bm{{{body}}}$" if is_best else f"${body}$"
        else:
            body = f"{m_val:{fmt}}"
            text = f"\\textbf{{{body}}}" if is_best else body
        return text

    # ── Build LaTeX ─────────────────────────────────────────────────────────
    n_metric_cols = len(col_specs)
    col_spec = "l" + "|" + "r" * 3 + "|" + "r" * 4  # method | SP HR IDM | 4×HR

    lines = []
    lines.append(r"% Requires: \usepackage{booktabs}, \usepackage{graphicx}, \usepackage{bm}")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Main results comparing unregularized and regularized self-play "
        r"at 50k training maps. "
        r"Self-play score on 10k validation scenes; human-replay and IDM-replay "
        r"metrics on 200 interactive validation scenes. "
        r"Best value per column in \textbf{bold}.}"
    )
    lines.append(r"\label{tab:main_comparison}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: block labels
    lines.append(r" & \multicolumn{3}{c|}{Score} & \multicolumn{4}{c}{Human-replay (interactive)} \\")

    # Header row 2: per-column labels
    col_headers = " & ".join(s[1] for s in col_specs)
    lines.append(
        r"\makecell{Method} & "
        r"\makecell{Self-play} & \makecell{HR} & \makecell{IDM} & " + " & ".join(s[1] for s in col_specs[3:]) + r" \\"
    )
    lines.append(r"\midrule")

    # Data rows
    for cpt in ROW_ORDER:
        label = CHECKPOINTS_OF_INTEREST[cpt]
        cells = [label] + [_fmt_cell(key, cpt, as_pct, decimals) for key, _, _, as_pct, decimals in col_specs]
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
        plot_scaling_barplot(df)
        print("  Saved eval_scaling_barplot.pdf")
        plot_data_requirements(df)
        print("  Saved eval_data_requirements.pdf")
        plot_human_data_requirements(df)
        print("  Saved eval_human_data_requirements.pdf")
        generate_scaling_latex_table(df)
        generate_hr_comparison_latex_table(df)
        plot_compatibility_tradeoff_bar(df)
        generate_human_data_latex_table(df)
        print("  Saved eval_compatibility_tradeoff_bar.pdf")
        plot_selfplay_behavior_analysis(df)
        print("  Saved eval_self_play behavior.pdf")
    plot_wosac_lineplot(wosac_df)
    print("  Saved eval_wosac_lineplot.pdf")
    plot_wosac_submetrics(wosac_df)
    print("  Saved eval_wosac_submetrics.pdf")
    plot_collision_severity(
        df,
        modes=("hr_interactive", "scaling_hr_interactive"),
        sp_maps_filter=50000,
    )
    generate_collision_severity_latex_table(df)
    plot_three_metric_comparison(df)
    generate_main_comparison_latex_table(df)
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
