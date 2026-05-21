"""Analyse and visualise training experience efficiency.

Two-panel figure summarising the paper's headline efficiency claim:

  Panel 1 — Safe task completion (task completion − at-fault collision rate
            with human-replay agents) vs. human-demonstration data.
            Regularized self-play RL (ours) uses a tiny amount of human data
            and beats methods that use orders of magnitude more.

  Panel 2 — Total training transitions per method, with the data type
            (self-play / synthetic vs. human) called out per bar.
            Ours and pure self-play RL train on 20B transitions
            (~63 years of driving at 10 Hz); SMART trains on 45M
            transitions of human data (the full Waymo dataset of
            ~500,000 scenarios × 9 s × 10 Hz = 45M transitions).

Safe task completion = task_completion − at_fault_collision_rate,
  where task_completion ∈ [0, 1] and at_fault_collision_rate is expressed
  as a fraction (e.g. 2.1% → 0.021).

Time conversion convention (single rule, used everywhere):
    Waymo scenarios are discretised at 10 Hz, so 1 transition = 0.1 s.
    => 20B transitions  ≈ 63.38 years
    => 45M transitions  ≈ 52 days  (full Waymo training set)
    =>  18,000 transitions ≈ 30 minutes (the regularizer's anchor data)

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

# ─── Time conversion ─────────────────────────────────────────────────────────
SECONDS_PER_TRANSITION = 0.1

# ─── Font sizes ──────────────────────────────────────────────────────────────
FONT_AXIS_LABEL = 12
FONT_TICK = 12
FONT_BAR_LABEL = 12
FONT_LEGEND = 12
FONT_ANNOT = 12
FONT_METHOD_LABEL = 12

# ─── Colors ──────────────────────────────────────────────────────────────────
COLOR_OURS = "#08519C"  # dark blue — regularized self-play RL
COLOR_SELFPLAY = "#000000"  # black    — self-play RL baseline
COLOR_SMART = "#d62728"  # tab:red  — SMART baseline

# ─── Data ────────────────────────────────────────────────────────────────────
# safe_task_completion = task_completion − at_fault_collision_rate (fraction)
#   Self-play RL : 1.000 − 0.021 = 0.979
#   Ours         : 1.000 − 0.006 = 0.994
#   SMART        : 0.846 − 0.016 = 0.830

df = pd.DataFrame(
    [
        {
            "Method": "Self-play RL",
            "selfplay_transitions": 20_000_000_000,
            "human_transitions": 0,
            "hr_score": 1.000 - 0.021,  # 0.979
        },
        {
            "Method": "Reg self-play RL (ours)",
            "selfplay_transitions": 20_000_000_000,
            "human_transitions": 18_000,  # ~30 min at 10 Hz
            "hr_score": 1.000 - 0.006,  # 0.994
        },
        {
            "Method": "SMART (IL)",
            "selfplay_transitions": 0,
            "human_transitions": 45_000_000,  # 500k scenarios × 9 s × 10 Hz
            "hr_score": 0.846 - 0.016,  # 0.830
        },
    ]
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _abbreviate_count(x: float) -> str:
    if x == 0:
        return "0"
    if x >= 1e9:
        return f"{x / 1e9:.4g}B"
    if x >= 1e6:
        return f"{x / 1e6:.4g}M"
    if x >= 1e3:
        return f"{x / 1e3:.4g}K"
    return f"{x:g}"


def _human_time_label(transitions: float) -> str:
    """Convert a transition count to a wall-clock label using the 10 Hz rule."""
    if transitions == 0:
        return "no data"
    seconds = transitions * SECONDS_PER_TRANSITION
    if seconds < 60:
        return f"~{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"~{minutes:.0f} min"
    hours = minutes / 60
    if hours < 24:
        return f"~{hours:.0f} h"
    days = hours / 24
    if days < 365:
        return f"~{days:.0f} days"
    years = days / 365.25
    return f"~{years:.2f} years"


def _color_for(method: str) -> str:
    if method.startswith("Reg"):
        return COLOR_OURS
    if method.startswith("Self-play"):
        return COLOR_SELFPLAY
    return COLOR_SMART


# ─── Plot ────────────────────────────────────────────────────────────────────


def make_figure(df: pd.DataFrame, save_path: str = SAVE_PATH) -> plt.Figure:
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    plt.subplots_adjust(wspace=0.32)

    # ── Panel 1: Safe task completion vs. human-demonstration data ───────────
    ax = ax1

    marker_for = {
        "Reg self-play RL (ours)": "*",
        "Self-play RL": "s",
        "SMART (IL)": "o",
    }
    size_for = {
        "Reg self-play RL (ours)": 700,
        "Self-play RL": 240,
        "SMART (IL)": 240,
    }

    for _, row in df.iterrows():
        method = row["Method"]
        x_h = row["human_transitions"]
        y = row["hr_score"]
        c = _color_for(method)

        ax.scatter(
            x_h,
            y,
            marker=marker_for[method],
            s=size_for[method],
            color=c,
            linewidth=1.5,
            zorder=4,
        )

    # Dashed arrow: SMART → regularized self-play
    ax.annotate(
        "",
        xy=(18_000, 1.000 - 0.006),
        xycoords="data",
        xytext=(45_000_000, 0.846 - 0.016),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="->",
            color="#FFEE8C",
            linewidth=2.0,
            shrinkA=10,
            shrinkB=18,
            linestyle="dashed",
            connectionstyle="arc3,rad=0.25",
        ),
        zorder=2,
    )

    # Dashed arrow: unregularized self-play → regularized self-play
    ax.annotate(
        "",
        xy=(18_000, 1.000 - 0.006),
        xycoords="data",
        xytext=(0, 1.000 - 0.021),
        textcoords="data",
        arrowprops=dict(
            arrowstyle="->",
            color="#FFEE8C",
            linewidth=2.0,
            shrinkA=10,
            shrinkB=18,
            linestyle="dashed",
            connectionstyle="arc3,rad=-0.25",
        ),
        zorder=2,
    )

    # Method labels
    ax.annotate(
        "Reg self-play RL (ours)",
        xy=(18_000, 1.000 - 0.006),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=FONT_METHOD_LABEL,
        color=COLOR_OURS,
    )
    ax.annotate(
        "Self-play RL",
        xy=(0, 1.000 - 0.021),
        xytext=(14, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=FONT_METHOD_LABEL,
        color=COLOR_SELFPLAY,
    )
    ax.annotate(
        "SMART (IL)",
        xy=(45_000_000, 0.846 - 0.016),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=FONT_METHOD_LABEL,
        color=COLOR_SMART,
    )

    # Human-data quantity callouts
    ours = df[df["Method"].str.startswith("Reg")].iloc[0]
    smart = df[df["Method"] == "SMART (IL)"].iloc[0]

    ax.annotate(
        _human_time_label(ours["human_transitions"]) + " of\nhuman data",
        xy=(ours["human_transitions"], ours["hr_score"]),
        xytext=(0, -30),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=FONT_ANNOT,
        color=COLOR_OURS,
        style="italic",
    )
    ax.annotate(
        _human_time_label(smart["human_transitions"]) + " of\nhuman data",
        xy=(smart["human_transitions"], smart["hr_score"]),
        xytext=(0, -28),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=FONT_ANNOT,
        color=COLOR_SMART,
        style="italic",
    )

    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-2, 1.5e9)
    ax.set_ylim(0.75, 1.02)

    xticks = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    ax.set_xticks(xticks)
    ax.set_xticklabels([_abbreviate_count(t) for t in xticks])
    ax.minorticks_off()

    ax.set_xlabel("Human demonstration data (transitions)", fontsize=14)
    ax.set_ylabel(
        "Safe task completion  ↑\n"
        r"(task completion $-$ at-fault collision rate)",
        fontsize=FONT_AXIS_LABEL,
    )
    ax.grid(axis="both", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    # ── Panel 2: total training data, per data source ───────────────────────
    ax = ax2

    panel2 = pd.DataFrame(
        [
            {"Method": "SMART (IL)", "selfplay": 0, "human": 45_000_000, "dashed": False},
            {"Method": "SMART (IL, full Waymo est.)", "selfplay": 0, "human": 225_000_000, "dashed": True},
            {"Method": "Self-play RL", "selfplay": 20_000_000_000, "human": 0, "dashed": False},
            {"Method": "Reg self-play RL (ours)", "selfplay": 20_000_000_000, "human": 18_000, "dashed": False},
        ]
    )
    panel2["total"] = panel2["selfplay"] + panel2["human"]

    methods = panel2["Method"].tolist()
    vals = panel2["total"].values
    bar_colors = [_color_for(m) for m in methods]
    x_pos = np.arange(len(methods))

    bars = []
    for xi, val, color, dashed in zip(x_pos, vals, bar_colors, panel2["dashed"]):
        b = ax.bar(
            xi,
            val,
            width=0.55,
            color=color,
            edgecolor=color,
            linewidth=1.2,
            alpha=0.9,
        )
        bars.append(b[0])

    def _compact_time(transitions: float) -> str:
        seconds = transitions * SECONDS_PER_TRANSITION
        if seconds < 60:
            return f"{seconds:.0f} s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.0f} min"
        hours = minutes / 60
        if hours < 24:
            return f"~{hours:.0f} h"
        days = hours / 24
        if days < 365:
            return f"{days:.0f} days"
        years = days / 365.25
        return f"{years:.0f} yrs"

    def _bar_label(sp: float, hu: float) -> str:
        components = []
        if sp > 0:
            components.append(("self-play", sp))
        if hu > 0:
            components.append(("human", hu))

        if len(components) == 1:
            tag, count = components[0]
            return f"{_abbreviate_count(count)} {tag}\n({_compact_time(count)})"

        lines = []
        for i, (tag, count) in enumerate(components):
            prefix = "" if i == 0 else "+ "
            lines.append(f"{prefix}{_abbreviate_count(count)} {tag}")
        times = " + ".join(_compact_time(c) for _, c in components)
        lines.append(f"({times})")
        return "\n".join(lines)

    for bar, sp, hu in zip(bars, panel2["selfplay"], panel2["human"]):
        label = _bar_label(sp, hu)
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_BAR_LABEL,
        )

    method_short = {
        "SMART (IL)": "SMART\n(IL)",
        "SMART (IL, full Waymo est.)": "SMART (IL)\nfull Waymo\n(est.)",
        "Self-play RL": "Self-play\nRL",
        "Reg self-play RL (ours)": "Reg self-play\nRL (ours)",
    }
    ax.set_xticks(x_pos)
    ax.set_xticklabels([method_short[m] for m in methods], fontsize=FONT_TICK - 1)

    ax.set_ylabel("Total training transitions", fontsize=14)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: _abbreviate_count(v)))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

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
    print("Time-equivalents under the 10 Hz rule (0.1 s per transition):")
    for _, r in df.iterrows():
        print(
            f"  {r['Method']:30s}  human={_human_time_label(r['human_transitions']):>15s}  "
            f"self-play={_human_time_label(r['selfplay_transitions']):>15s}"
        )
    print()
    make_figure(df)
