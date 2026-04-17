"""Analyse and visualise training experience efficiency.

Compares our regularized self-play RL approach against SMART across five
dimensions:
  - Total experience (transitions)
  - Human demonstrations (transitions)
  - Self-play collision rate (%)
  - Human-replay collision rate (%)
  - Total training time (hours)

Output: results/figures/experience_comparison.pdf
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

DPI = 600
SAVE_PATH = "results/figures/experience_comparison.pdf"

# ─── Font sizes ──────────────────────────────────────────────────────────────
FONT_TITLE = 12  # panel titles
FONT_AXIS_LABEL = 11  # x and y axis labels
FONT_TICK = 11  # x-tick method names
FONT_BAR_LABEL = 11  # values printed above bars
FONT_SUMM_TITLE = 11  # summary panel heading
FONT_SUMM_LABEL = 9  # summary row category label
FONT_SUMM_VALUE = 11  # summary row bold value
FONT_SUMM_DETAIL = 6  # summary row italic detail
FONT_LEGEND = 8  # colour legend text

# ─── Colors ──────────────────────────────────────────────────────────────────
COLOR_OURS = "#CCCCFF"  # reg self-play RL (ours)
COLOR_SELFPLAY = "#4A7FD4"  # self-play RL baseline (tab:blue)
COLOR_SMART = "#E8609A"  # SMART baseline

# ─── Data ────────────────────────────────────────────────────────────────────

df = pd.DataFrame(
    [
        {
            "Method": "Self-play \n RL",
            "Total experience learned from": 5_000_000_000,
            "Human demonstrations used": 0,
            "Self-play\ncollision rate": 0.5,
            "Human-replay\ncollision rate": 5.0,
            "Cumulative training time": 7,
        },
        {
            "Method": "Reg self-play \n RL (ours)",
            "Total experience learned from": 5_000_000_000 + 45_000,
            "Human demonstrations used": 45_000,
            "Self-play\ncollision rate": 0.1,
            "Human-replay\ncollision rate": 0.4,
            "Cumulative training time": 7.3,
        },
        {
            "Method": "SMART \n (IL-based)",
            "Total experience learned from": 225_000_000,
            "Human demonstrations used": 225_000_000,
            "Self-play\ncollision rate": 0.5,
            "Human-replay\ncollision rate": 4.0,
            "Cumulative training time": 168,
        },
    ]
)

# ─── Plot ────────────────────────────────────────────────────────────────────


def _abbreviate(x):
    """Format large numbers as e.g. 5B, 225M, 45K."""
    if x >= 1e9:
        return f"{x / 1e9:.4g}B"
    if x >= 1e6:
        return f"{x / 1e6:.4g}M"
    if x >= 1e3:
        return f"{x / 1e3:.4g}K"
    return str(x)


def make_figure(df: pd.DataFrame, save_path: str = SAVE_PATH) -> plt.Figure:
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")

    methods = df["Method"].tolist()
    colors = [COLOR_SELFPLAY, COLOR_OURS, COLOR_SMART]
    n_methods = len(methods)
    x = np.arange(n_methods)
    bar_w = 0.45

    # ── Layout: 2 rows × 3 cols, last cell used for a legend/summary box ────
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.38)

    ax_exp = fig.add_subplot(gs[0, 0])  # total experience
    ax_demo = fig.add_subplot(gs[0, 1], sharey=ax_exp)  # human demos — shared y
    ax_time = fig.add_subplot(gs[0, 2])  # training time
    ax_sp = fig.add_subplot(gs[1, 0])  # self-play collision
    ax_hr = fig.add_subplot(gs[1, 1])  # human-replay collision
    ax_summ = fig.add_subplot(gs[1, 2])  # summary / ratio panel

    # ── Helper: draw one grouped bar panel ──────────────────────────────────
    def _bar(ax, col, ylabel, log=False, lower_is_better=True):
        vals = df[col].values
        bars = ax.bar(x, vals, width=bar_w, color=colors, alpha=0.85, edgecolor="white", linewidth=0.6)

        # Value labels above each bar — fixed 6pt offset avoids floating on log scale
        for bar, val in zip(bars, vals):
            label = _abbreviate(val) if val >= 1000 else f"{val:g}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FONT_BAR_LABEL,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=FONT_TICK)
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS_LABEL)

        # Arrow in title indicates direction of improvement
        arrow = " (↓)" if lower_is_better else " (↑)"
        ax.set_title(col + arrow, fontsize=FONT_TITLE, fontweight="normal", pad=6, y=1.1)

        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if log:
            ax.set_yscale("log")
        ax.set_ylim(bottom=0 if not log else None)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _abbreviate(v) if v >= 1000 else f"{v:g}"))

        sns.despine(ax=ax)

    _bar(ax_exp, "Total experience learned from", "Transitions", log=False, lower_is_better=False)
    _bar(ax_demo, "Human demonstrations used", "Transitions", log=False, lower_is_better=True)
    ax_demo.set_ylabel("")
    ax_demo.tick_params(labelleft=False)
    _bar(ax_time, "Cumulative training time", "Hours", log=False, lower_is_better=True)
    _bar(ax_sp, "Self-play\ncollision rate", "Collision rate (%)", log=False, lower_is_better=True)
    _bar(ax_hr, "Human-replay\ncollision rate", "Collision rate (%)", log=False, lower_is_better=True)

    # ── Summary / ratio panel ────────────────────────────────────────────────
    ax_summ.axis("off")

    ours = df[df["Method"].str.startswith("Reg")].iloc[0]
    smart = df[df["Method"] == "SMART \n (IL-based)"].iloc[0]

    # Pre-compute summary statistics
    ours_speed = ours["Total experience learned from"] / ours["Cumulative training time"]
    smart_speed = smart["Total experience learned from"] / smart["Cumulative training time"]
    speed_ratio = ours_speed / smart_speed

    demo_ours = ours["Human demonstrations used"]
    demo_smart = smart["Human demonstrations used"]
    demo_ratio = demo_smart / demo_ours

    time_ours = ours["Cumulative training time"]
    time_smart = smart["Cumulative training time"]
    time_reduction = time_smart / time_ours

    sp_ours = ours["Self-play\ncollision rate"]
    sp_smart = smart["Self-play\ncollision rate"]
    hr_ours = ours["Human-replay\ncollision rate"]
    hr_smart = smart["Human-replay\ncollision rate"]

    GREEN = "#5cca61"
    # (label, value_str, detail_str, color)
    rows = [
        (
            "Training speed",
            f"{speed_ratio:.0f}×  faster",
            f"({_abbreviate(int(ours_speed))} vs. {_abbreviate(int(smart_speed))} transitions/hr)",
            GREEN,
        ),
        (
            "Human data required",
            f"{demo_ratio:,.0f}× less",
            f"({_abbreviate(int(demo_ours))} vs. {_abbreviate(int(demo_smart))} transitions)",
            GREEN,
        ),
        ("Training time", f"{time_reduction:.0f}× shorter", f"({time_ours:.0f} hrs vs. {time_smart:.0f} hrs)", GREEN),
        ("SP collision rate", f"{sp_smart / sp_ours:.0f}× lower", f"({sp_ours:.1f}% vs. {sp_smart:.1f}%)", GREEN),
        ("HR collision rate", f"{hr_smart / hr_ours:.0f}× lower", f"({hr_ours:.1f}% vs. {hr_smart:.1f}%)", GREEN),
    ]

    # Title
    title_y = 1.23
    ax_summ.text(
        -0.1,
        title_y,
        "Ours (RL-based) vs. SMART (IL-based)",
        transform=ax_summ.transAxes,
        fontsize=FONT_SUMM_TITLE,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax_summ.text(
        -0.1,
        title_y - 0.10,
        "Advantage of regularized self-play RL",
        transform=ax_summ.transAxes,
        fontsize=FONT_SUMM_LABEL,
        color="grey",
        ha="left",
        va="top",
        style="italic",
    )
    # Rows — fixed spacing, label above value within each row
    row_start = title_y - 0.30
    row_step = 0.20

    for i, (label, value_str, detail_str, row_color) in enumerate(rows):
        yc = row_start - i * row_step
        ax_summ.text(
            0.0,
            yc,
            label,
            transform=ax_summ.transAxes,
            fontsize=FONT_SUMM_LABEL,
            va="center",
            ha="left",
            color="dimgrey",
        )
        ax_summ.text(
            0.0,
            yc - 0.08,
            value_str,
            transform=ax_summ.transAxes,
            fontsize=FONT_SUMM_VALUE,
            fontweight="bold",
            va="center",
            ha="left",
            color=row_color,
        )
        ax_summ.text(
            0.97,
            yc - 0.08,
            detail_str,
            transform=ax_summ.transAxes,
            fontsize=FONT_SUMM_DETAIL,
            va="center",
            ha="right",
            color="grey",
            style="italic",
        )

    # ── Colour legend (bottom of summary panel) ───────────────────────────────
    legend_items = list(zip(methods, colors))
    legend_y = 0.00
    patch_w, patch_h = 0.10, 0.07
    col_positions = [0.04, 0.38, 0.68]  # left edges of each legend entry

    # for j, ((method, color), lx) in enumerate(zip(legend_items, col_positions)):
    #     ax_summ.add_patch(
    #         mpl.patches.FancyBboxPatch(
    #             (lx, legend_y - patch_h / 2), patch_w, patch_h,
    #             transform=ax_summ.transAxes, color=color, alpha=0.85,
    #             boxstyle="round,pad=0.01",
    #         )
    #     )
    #     ax_summ.text(lx + patch_w + 0.03, legend_y, method.replace("\n", " "),
    #                  transform=ax_summ.transAxes, fontsize=FONT_LEGEND,
    #                  va="top", ha="left")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"Saved {save_path}")
    return fig


if __name__ == "__main__":
    print("Experience comparison dataframe:")
    print(df.to_string(index=False))
    print()
    make_figure(df)
