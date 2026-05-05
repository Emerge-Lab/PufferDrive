"""Analyse and visualise training experience efficiency.

Two-panel figure summarising the paper's headline efficiency claim:

  Panel 1 — Performance (HR score, proxy for human-compatibility) vs.
            human-demonstration data. Regularized self-play RL (ours)
            uses a tiny amount of human data and beats methods that
            use orders of magnitude more.

  Panel 2 — Total training transitions per method, with the data type
            (self-play / synthetic vs. human) called out per bar.
            Ours and pure self-play RL train on 10B transitions
            (~31.68 years of driving at 10 Hz); SMART trains on 45M
            transitions of human data (the full Waymo dataset of
            ~500,000 scenarios × 9 s × 10 Hz = 45M transitions).

Time conversion convention (single rule, used everywhere):
    Waymo scenarios are discretised at 10 Hz, so 1 transition = 0.1 s.
    => 10B transitions  ≈ 31.68 years
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
# 10 Hz Waymo scenarios → 0.1 s per transition. Single source of truth so
# every "≈ X years" label in the figure stays consistent.
SECONDS_PER_TRANSITION = 0.1

# ─── Font sizes ──────────────────────────────────────────────────────────────
FONT_AXIS_LABEL = 12
FONT_TICK = 12
FONT_BAR_LABEL = 12
FONT_LEGEND = 12
FONT_ANNOT = 12
FONT_METHOD_LABEL = 12

# ─── Colors ──────────────────────────────────────────────────────────────────
# Same scheme as plotting.py:
#   regularized self-play RL = blue, self-play RL = black, SMART = tab:red.
COLOR_OURS = "#6BAED6"  # reg self-play RL (ours): medium blue
COLOR_OURS_EDGE = "#08519C"  # dark blue
COLOR_SELFPLAY = "#000000"  # self-play RL baseline: black
COLOR_SMART = "#d62728"  # SMART baseline: tab:red
COLOR_SMART_EDGE = "#8B1A1B"  # darker red

# ─── Data ────────────────────────────────────────────────────────────────────
# The `hr_score` column is a proxy for human-compatibility on the
# human-replay eval — replace with the exact numbers from the eval CSV.

df = pd.DataFrame(
    [
        {
            "Method": "Self-play RL",
            "selfplay_transitions": 10_000_000_000,
            "human_transitions": 0,
            "hr_score": 0.55,  # proxy — replace with real
        },
        {
            "Method": "Reg self-play RL (ours)",
            "selfplay_transitions": 10_000_000_000,
            "human_transitions": 18_000,  # ~30 min at 10 Hz
            "hr_score": 0.85,  # proxy — replace with real
        },
        {
            "Method": "SMART (IL)",
            "selfplay_transitions": 0,
            "human_transitions": 45_000_000,  # 500k scenarios × 9s × 10 Hz
            "hr_score": 0.70,  # proxy — replace with real
        },
    ]
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _abbreviate_count(x: float) -> str:
    """Format large counts as e.g. 10B, 225M, 1.8K, 0."""
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
    """Convert a transition count to a wall-clock label using the 10 Hz rule.

    Picks the unit that reads most naturally (min / hours / days / years).
    Used everywhere in the figure so labels stay consistent with the
    convention stated in the paper's appendix.
    """
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
    # 365.25 days/year (Julian year) — gives 10B transitions ≈ 31.69 years,
    # essentially matching the paper appendix's stated 31.68 years (the
    # appendix appears to use 365 days/year and quote a slightly different
    # rounding; the discrepancy is < 0.05 years).
    years = days / 365.25
    return f"~{years:.2f} years"


def _color_for(method: str) -> str:
    if method.startswith("Reg"):
        return COLOR_OURS
    if method.startswith("Self-play"):
        return COLOR_SELFPLAY
    return COLOR_SMART


def _edge_for(method: str) -> str:
    if method.startswith("Reg"):
        return COLOR_OURS_EDGE
    if method.startswith("Self-play"):
        return "#000000"
    return COLOR_SMART_EDGE


# ─── Plot ────────────────────────────────────────────────────────────────────


def make_figure(df: pd.DataFrame, save_path: str = SAVE_PATH) -> plt.Figure:
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks", rc={"figure.facecolor": "none", "axes.facecolor": "none"})
    warnings.filterwarnings("ignore")
    plt.set_loglevel("WARNING")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    plt.subplots_adjust(wspace=0.32)

    # ── Panel 1: HR score vs. human-demonstration data ──────────────────────
    # x = human transitions (symlog so the "0 demonstrations" point sits
    # naturally at the left edge); y = HR score (higher = better, proxy for
    # how well the agent coordinates with human drivers).

    ax = ax1

    # Ours = star (highlights the headline); SMART = circle; Self-play RL = square.
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
        ec = _edge_for(method)

        ax.scatter(
            x_h,
            y,
            marker=marker_for[method],
            s=size_for[method],
            color=c,
            edgecolor=ec,
            linewidth=1.5,
            zorder=4,
        )

    # Subtle arrow from the unregularized self-play square up to the
    # regularized self-play star — visualises "what 30 min of human data
    # buys you." Drawn behind the markers (zorder<4) so the arrow tail
    # tucks under the marker edges instead of covering them. Soft gray
    # plus a thin curved line keep it as a visual cue, not a focal point.
    ax.annotate(
        "",
        xy=(18_000, 0.85),
        xycoords="data",  # arrowhead at the star
        xytext=(0, 0.55),
        textcoords="data",  # tail at the square
        arrowprops=dict(
            arrowstyle="->",
            color="#888888",
            linewidth=1.2,
            shrinkA=10,
            shrinkB=18,  # don't draw inside the markers
            connectionstyle="arc3,rad=-0.25",
        ),
        zorder=2,
    )

    # Method labels — placed by hand so they don't collide with markers,
    # the y-axis, or each other.
    ax.annotate(
        "Reg self-play RL (ours)",
        xy=(18_000, 0.85),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=FONT_METHOD_LABEL,
        color=COLOR_OURS_EDGE,
    )
    # "Self-play RL" sits at x=0 (left edge of symlog). Anchor its label to
    # the right of the marker so it doesn't get clipped.
    ax.annotate(
        "Self-play RL",
        xy=(0, 0.55),
        xytext=(14, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=FONT_METHOD_LABEL,
        color=COLOR_SELFPLAY,
    )
    ax.annotate(
        "SMART (IL)",
        xy=(45_000_000, 0.70),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=FONT_METHOD_LABEL,
        color=COLOR_SMART_EDGE,
    )

    # Human-data quantity callouts under the two markers that have it.
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
        color=COLOR_OURS_EDGE,
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
        color=COLOR_SMART_EDGE,
        style="italic",
    )

    # x-axis: symlog so 0 fits cleanly. linthresh=10 keeps 18,000 in the
    # log region and gives the 0 point a small linear band on the left.
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(-2, 1.5e9)
    ax.set_ylim(0.4, 1.0)

    xticks = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    ax.set_xticks(xticks)
    ax.set_xticklabels([_abbreviate_count(t) for t in xticks])
    ax.minorticks_off()

    ax.set_xlabel("Human demonstration data (transitions)", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Score with proxy for human players  ↑", fontsize=FONT_AXIS_LABEL)
    ax.grid(axis="both", alpha=0.3, linestyle="--")
    sns.despine(ax=ax)

    # ── Panel 2: total training data, per data source ───────────────────────
    # The story: ours and self-play RL each consume ~10B transitions, almost
    # all of it cheap synthetic self-play. SMART trains on 45M transitions
    # (the full Waymo dataset) of expensive human data. Each bar carries a
    # "(self-play)" or "(human)" tag so the reader sees not just the *amount*
    # but also the *source*. Ours is a hybrid — its label calls out *both*
    # the 10B self-play and the 18K human anchor, even though 18K is
    # vanishingly small on this axis.

    ax = ax2

    # Each method's training data, split by source. Bar height = sum.
    # Ours adds 18,000 human transitions on top of 10B self-play; visually
    # negligible (1.8 ppm of the bar) but it's still real training data, so
    # we surface it explicitly in the bar's label.
    # Order left-to-right: SMART (smallest), Self-play RL, Reg self-play RL
    # (ours, rightmost). Putting ours last keeps the reading order
    # consistent with panel 1's "scan up to the star" trajectory.
    panel2 = pd.DataFrame(
        [
            {"Method": "SMART (IL)", "selfplay": 0, "human": 45_000_000},
            {"Method": "Self-play RL", "selfplay": 10_000_000_000, "human": 0},
            {"Method": "Reg self-play RL (ours)", "selfplay": 10_000_000_000, "human": 18_000},
        ]
    )
    panel2["total"] = panel2["selfplay"] + panel2["human"]

    methods = panel2["Method"].tolist()
    vals = panel2["total"].values
    bar_colors = [_color_for(m) for m in methods]
    bar_edges = [_edge_for(m) for m in methods]
    x_pos = np.arange(len(methods))

    # Linear y-axis on purpose. The story is that 10B utterly dwarfs 45M;
    # a log axis would visually equalise them and bury the point.
    bars = ax.bar(
        x_pos,
        vals,
        width=0.55,
        color=bar_colors,
        edgecolor=bar_edges,
        linewidth=1.2,
        alpha=0.9,
    )

    def _compact_time(transitions: float) -> str:
        """Compact form of `_human_time_label` for tight bar annotations."""
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
        return f"~{years:.0f} yrs"

    def _bar_label(sp: float, hu: float) -> str:
        """Annotation for a panel-2 bar, surfacing each non-zero source.

        Single-source bars get a compact 2-line label:
          (10B, 0)  -> "10B self-play\n(~32 yrs)"
          (0, 45M)  -> "45M human\n(~52 days)"

        Hybrid bars (only "ours" today) split the components onto their
        own lines so the per-line text stays narrow enough not to collide
        with neighbouring bars' labels:
          (10B, 18K) -> "10B self-play\n+ 18K human\n(~32 yrs + ~30 min)"
        """
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

    # Wrap long method names so the x-tick labels don't collide.
    method_short = {
        "SMART (IL)": "SMART\n(IL)",
        "Self-play RL": "Self-play\nRL",
        "Reg self-play RL (ours)": "Reg self-play\nRL (ours)",
    }
    ax.set_xticks(x_pos)
    ax.set_xticklabels([method_short[m] for m in methods], fontsize=FONT_TICK)
    ax.set_ylabel("Total training transitions", fontsize=FONT_AXIS_LABEL)

    # Pad enough headroom that the multi-line annotation above the tallest
    # bar isn't clipped at the top of the axes. Ours has a 3-line hybrid
    # label, hence the extra room.
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
