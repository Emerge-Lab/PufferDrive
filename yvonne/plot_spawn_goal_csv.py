"""
Plot spawn→goal lines from episode_metrics CSVs produced by the validation/mine pipeline.

Usage
-----
    python yvonne/plot_spawn_goal_csv.py <csv_or_dir> [<csv_or_dir> ...] [--out out.png]

Examples
    # single CSV
    python yvonne/plot_spawn_goal_csv.py runs/.../episode_metrics/validation_*.csv

    # whole run dir (finds all CSVs under it)
    python yvonne/plot_spawn_goal_csv.py /scratch/yw4142/PufferDrive_exp/.../

    # explicit output path
    python yvonne/plot_spawn_goal_csv.py data.csv --out my_plot.png

The CSV must have columns: agent_spawn_x, agent_spawn_y,
agent_final_goal_x, agent_final_goal_y, agent_outcome.
Values are either plain floats or string-encoded Python lists like "[34.5]".
"""

import argparse
import ast
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


OUTCOME_INFRACTION = 0
OUTCOME_SUCCESS = 1
OUTCOME_DNF = 2

_CMAPS = {
    OUTCOME_SUCCESS: "Greens",
    OUTCOME_DNF: "Oranges",
    OUTCOME_INFRACTION: "Reds",
}
_OUTCOME_LABELS = {
    OUTCOME_SUCCESS: "Success",
    OUTCOME_DNF: "DNF",
    OUTCOME_INFRACTION: "Infraction",
}


def _parse_val(v):
    """Return a float from either a plain number or a string like '[34.5]'."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s.startswith("["):
        lst = ast.literal_eval(s)
        return float(lst[0]) if lst else float("nan")
    return float(s)


def load_csvs(paths):
    frames = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.endswith(".csv"):
                        frames.append(pd.read_csv(os.path.join(root, f)))
        else:
            frames.append(pd.read_csv(p))
    if not frames:
        sys.exit("No CSV files found.")
    df = pd.concat(frames, ignore_index=True)
    needed = {"agent_spawn_x", "agent_spawn_y", "agent_final_goal_x", "agent_final_goal_y", "agent_outcome"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"CSV missing columns: {missing}")
    for col in needed:
        df[col] = df[col].map(_parse_val)
    return df


def plot(df, out_path):
    outcomes = (OUTCOME_SUCCESS, OUTCOME_DNF, OUTCOME_INFRACTION)
    total = len(df)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Spawn & goal density  (n={total:,} episodes)", fontsize=13)

    def _extent(vals, pad=20.0):
        return vals.min() - pad, vals.max() + pad

    all_sx = df["agent_spawn_x"]; all_sy = df["agent_spawn_y"]
    all_gx = df["agent_final_goal_x"]; all_gy = df["agent_final_goal_y"]
    sx_lo, sx_hi = _extent(all_sx); sy_lo, sy_hi = _extent(all_sy)
    gx_lo, gx_hi = _extent(all_gx); gy_lo, gy_hi = _extent(all_gy)

    BINS = 60
    row_labels = ["Spawn (init)", "Goal (final)"]

    for col, outcome in enumerate(outcomes):
        sub = df[df["agent_outcome"] == outcome]
        label = _OUTCOME_LABELS[outcome]
        cmap = _CMAPS[outcome]

        for row, (xcol, ycol, xlim, ylim) in enumerate([
            ("agent_spawn_x",      "agent_spawn_y",      (sx_lo, sx_hi), (sy_lo, sy_hi)),
            ("agent_final_goal_x", "agent_final_goal_y", (gx_lo, gx_hi), (gy_lo, gy_hi)),
        ]):
            ax = axes[row, col]
            ax.set_title(f"{label} — {row_labels[row]}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)", fontsize=7)
            ax.set_ylabel("y (m)", fontsize=7)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            if len(sub) >= 2:
                xs = sub[xcol].to_numpy()
                ys = sub[ycol].to_numpy()
                h, xe, ye, img = ax.hist2d(
                    xs, ys,
                    bins=BINS,
                    range=[[xlim[0], xlim[1]], [ylim[0], ylim[1]]],
                    cmap=cmap,
                    norm=LogNorm(vmin=1),
                )
                fig.colorbar(img, ax=ax, shrink=0.7, label="count")
                ax.text(0.02, 0.98, f"n={len(sub):,}", transform=ax.transAxes,
                        fontsize=7, va="top", ha="left", color="black",
                        bbox=dict(fc="white", alpha=0.6, pad=1, ec="none"))
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="gray", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="CSV files or run directories to load")
    parser.add_argument("--out", default="spawn_goal.png", help="Output PNG path (default: spawn_goal.png)")
    args = parser.parse_args()

    df = load_csvs(args.inputs)
    print(f"Loaded {len(df):,} episodes — outcome counts: {df['agent_outcome'].value_counts().to_dict()}")
    plot(df, args.out)


if __name__ == "__main__":
    main()
