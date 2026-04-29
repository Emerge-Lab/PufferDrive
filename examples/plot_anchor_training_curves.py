"""Plot within-5-bin validation accuracy curves for BC anchor training runs.

Reads the three wandb-exported CSVs (dx / dy / dyaw), one subplot per axis,
one line per training-data size. Run names are expected to follow the
pattern `..._<N>maps`; that integer is parsed for the legend and color
ordering (light -> dark = less -> more human data, matching the convention
in plot_results.PALETTE).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DPI = 600


def _maps_to_human_time(maps: int) -> str:
    """Convert map count to a human-readable label (9s per map, 1 agent)."""
    minutes = (maps * 9) / 60
    if minutes >= 60:
        hours = minutes / 60
        if hours == int(hours):
            return f"{int(hours)} hours"
        return f"{hours:.1f} hours"
    if minutes == int(minutes):
        return f"{int(minutes)} min"
    return f"{minutes:.0f} min"


def _parse_runs(df: pd.DataFrame, axis: str):
    """Find (run_name, num_maps, mean_col, min_col, max_col) tuples for one axis CSV.

    Only mean columns are matched directly; MIN/MAX are looked up by suffix.
    Runs without a parseable `_<N>maps` token are skipped with a warning.
    """
    metric = f"val/acc_within_5bins_{axis}"
    runs = []
    for col in df.columns:
        if not col.endswith(f" - {metric}"):
            continue
        run_name = col.split(" - ")[0]
        m = re.search(r"_(\d+)maps", run_name)
        if not m:
            print(f"  Skipping run with no _Nmaps token: {run_name}")
            continue
        num_maps = int(m.group(1))
        min_col = f"{col}__MIN"
        max_col = f"{col}__MAX"
        runs.append(
            {
                "run": run_name,
                "num_maps": num_maps,
                "mean_col": col,
                "min_col": min_col if min_col in df.columns else None,
                "max_col": max_col if max_col in df.columns else None,
            }
        )
    runs.sort(key=lambda r: r["num_maps"])
    return runs


def plot_anchor_training_curves(
    dx_csv: str = "results/anchor_train_dx.csv",
    dy_csv: str = "results/anchor_train_dy.csv",
    dyaw_csv: str = "results/anchor_train_dyaw.csv",
    save_path: str = "results/anchor_training_curves.pdf",
):
    sns.set("notebook", font_scale=1.05)
    sns.set_style("ticks")

    axes_specs = [
        ("dx", dx_csv, r"$\Delta x$ (longitudinal)"),
        ("dy", dy_csv, r"$\Delta y$ (lateral)"),
        ("dyaw", dyaw_csv, r"$\Delta \mathrm{yaw}$ (heading)"),
    ]

    fig, axarr = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)

    for ax, (axis, csv_path, title) in zip(axarr, axes_specs):
        df = pd.read_csv(csv_path)
        runs = _parse_runs(df, axis)

        # Use matplotlib's default color cycle, assigned in run order.
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, r in enumerate(runs):
            r["color"] = default_cycle[i % len(default_cycle)]

        for r in runs:
            sub = df[["Step", r["mean_col"]]].copy()
            sub = sub.dropna(subset=[r["mean_col"]])
            if sub.empty:
                continue
            steps = sub["Step"].to_numpy()
            mean = sub[r["mean_col"]].to_numpy() * 100  # to %
            label = f"{r['num_maps']:,} maps ({_maps_to_human_time(r['num_maps'])})"

            ax.plot(steps, mean, color=r["color"], linewidth=1.8, label=label, zorder=3)

            # Shade MIN/MAX band if it carries any spread (wandb often dumps
            # mean==min==max when only one seed exists; skip in that case so
            # the figure isn't visually noisy with zero-width bands).
            if r["min_col"] and r["max_col"]:
                band = df[["Step", r["min_col"], r["max_col"]]].dropna()
                if not band.empty:
                    lo = band[r["min_col"]].to_numpy() * 100
                    hi = band[r["max_col"]].to_numpy() * 100
                    if np.any(hi - lo > 1e-9):
                        ax.fill_between(
                            band["Step"].to_numpy(),
                            lo,
                            hi,
                            color=r["color"],
                            alpha=0.18,
                            linewidth=0,
                            zorder=2,
                        )

        ax.set_xlabel("Training step")
        ax.set_ylabel("Within-5-bin val accuracy (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=9, loc="lower right", framealpha=1.0, facecolor="white", edgecolor="lightgray")
        sns.despine(ax=ax)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved {save_path}")
    return fig


if __name__ == "__main__":
    plot_anchor_training_curves()
