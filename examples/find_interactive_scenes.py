"""
Find and filter driving scenarios by interactivity.

Interactivity proxy: number of times the SDC trajectory intersects
with any other agent's trajectory (line-segment intersections).

Pipeline:
  1. Score every scenario JSON in --data_folder
  2. Save a CSV/parquet dataframe with per-scenario intersection counts
  3. Copy the top-K most interactive JSONs into --output_folder
  4. (Optional) --visualize: render plots of the top-K scenarios into --plot_folder

Usage:
    # Basic: score all, copy top 500, save dataframe
    python examples/find_interactive_scenes.py \
        --data_folder data/processed/training \
        --top_k 500

    # With visualization of top 50
    python examples/find_interactive_scenes.py \
        --data_folder data/processed/training \
        --top_k 100 \
        --visualize \
        --vis_top_k 50
        --prioritize_intersections

    # Cap file count for quick testing
    python examples/find_interactive_scenes.py \
        --data_folder data/processed/training \
        --top_k 100 \
        --max_files 1000 \
        --visualize
"""

import json
import shutil
import argparse
import math
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.collections as mc


# ── geometry helpers ─────────────────────────────────────────────────────────


def ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)


def segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """Return True if segment AB intersects segment CD."""
    if ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and ccw(ax, ay, bx, by, cx, cy) != ccw(
        ax, ay, bx, by, dx, dy
    ):
        return True
    return False


def segment_intersection_point(ax, ay, bx, by, cx, cy, dx, dy):
    """Return (x, y) of intersection between AB and CD, or None."""
    denom = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if abs(denom) < 1e-12:
        return None
    t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / denom
    ix = ax + t * (bx - ax)
    iy = ay + t * (by - ay)
    return (ix, iy)


def extract_trajectory(obj):
    """Return list of (x, y) for valid timesteps, None for gaps."""
    positions = obj.get("position", [])
    valids = obj.get("valid", [])
    pts = []
    for i, pos in enumerate(positions):
        if i < len(valids) and valids[i]:
            pts.append((pos.get("x", 0.0), pos.get("y", 0.0)))
        else:
            pts.append(None)
    return pts


def traj_to_segments(traj):
    """Convert trajectory to list of ((x0,y0),(x1,y1)) segments, skipping gaps."""
    segs = []
    for i in range(len(traj) - 1):
        if traj[i] is not None and traj[i + 1] is not None:
            segs.append((traj[i], traj[i + 1]))
    return segs


def segment_angle(ax, ay, bx, by, cx, cy, dx, dy):
    """Return the acute angle (radians) between segments AB and CD."""
    # Direction vectors
    ux, uy = bx - ax, by - ay
    vx, vy = dx - cx, dy - cy
    dot = ux * vx + uy * vy
    mag_u = math.sqrt(ux * ux + uy * uy)
    mag_v = math.sqrt(vx * vx + vy * vy)
    if mag_u < 1e-9 or mag_v < 1e-9:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (mag_u * mag_v)))
    angle = math.acos(abs(cos_theta))  # acute angle in [0, pi/2]
    return angle


def score_intersections(traj_a, traj_b, angle_threshold_rad=0.0):
    """Score intersections between two trajectories.

    Returns:
        raw_count:      total segment-segment intersections
        angled_count:   intersections where the acute angle >= angle_threshold_rad
    """
    segs_a = traj_to_segments(traj_a)
    segs_b = traj_to_segments(traj_b)
    raw_count = 0
    angled_count = 0
    for (ax, ay), (bx, by) in segs_a:
        for (cx, cy), (dx, dy) in segs_b:
            if segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                raw_count += 1
                angle = segment_angle(ax, ay, bx, by, cx, cy, dx, dy)
                if angle >= angle_threshold_rad:
                    angled_count += 1
    return raw_count, angled_count


# ── scoring worker ───────────────────────────────────────────────────────────

# Default angle threshold: 15 degrees. Intersections below this are
# considered near-parallel (e.g. highway lane changes) and won't count
# as "angled" intersections.
ANGLE_THRESHOLD_DEG = 15
ANGLE_THRESHOLD_RAD = math.radians(ANGLE_THRESHOLD_DEG)


def process_scenario(filepath):
    """Score a single scenario file. Returns tuple for DataFrame construction."""
    filepath = Path(filepath)
    try:
        with open(filepath) as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        sdc_idx = metadata.get("sdc_track_index", -1)
        objects = data.get("objects", [])

        if sdc_idx < 0 or sdc_idx >= len(objects):
            return (str(filepath.name), 0, 0, len(objects), sdc_idx, None)

        sdc_traj = extract_trajectory(objects[sdc_idx])

        total_raw = 0
        total_angled = 0
        for i, obj in enumerate(objects):
            if i == sdc_idx:
                continue
            raw, angled = score_intersections(sdc_traj, extract_trajectory(obj), ANGLE_THRESHOLD_RAD)
            total_raw += raw
            total_angled += angled

        return (str(filepath.name), total_raw, total_angled, len(objects), sdc_idx, None)

    except Exception as e:
        return (str(filepath.name), -1, 0, 0, -1, str(e))


# ── visualization ────────────────────────────────────────────────────────────

ROAD_COLORS = {
    "lane": "#777777",
    "road_line": "#AAAAAA",
    "road_edge": "#999999",
    "stop_sign": "#FF4444",
    "crosswalk": "#DDAA33",
    "speed_bump": "#BB8833",
}
DEFAULT_ROAD_COLOR = "#888888"


def draw_roads(ax, roads):
    for road in roads:
        geometry = road.get("geometry", [])
        if len(geometry) < 2:
            continue
        xs = [p.get("x", 0) for p in geometry]
        ys = [p.get("y", 0) for p in geometry]
        road_type = road.get("type", "")
        color = ROAD_COLORS.get(road_type, DEFAULT_ROAD_COLOR)
        lw = 0.5 if road_type == "lane" else 0.8
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.6, zorder=1)


def visualize_scenario(data, title="", ax=None):
    """Plot road graph + trajectories. SDC green, others blue, intersections red."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))

    metadata = data.get("metadata", {})
    sdc_idx = metadata.get("sdc_track_index", -1)
    objects = data.get("objects", [])
    roads = data.get("roads", [])

    # Roads
    draw_roads(ax, roads)

    # Trajectories
    sdc_traj = None
    other_trajs = []

    for i, obj in enumerate(objects):
        traj = extract_trajectory(obj)
        segs = traj_to_segments(traj)
        if not segs:
            continue

        valid_pts = [p for p in traj if p is not None]
        mid_pt = valid_pts[len(valid_pts) // 2] if valid_pts else None

        if i == sdc_idx:
            sdc_traj = traj
            lines = mc.LineCollection(segs, colors="#66FF66", linewidths=2.5, zorder=4, label="SDC")
            ax.add_collection(lines)
            ax.plot(segs[0][0][0], segs[0][0][1], "o", color="#44DD44", markersize=7, zorder=5)
            ax.plot(segs[-1][1][0], segs[-1][1][1], "s", color="#33BB33", markersize=7, zorder=5)
            if mid_pt:
                ax.text(
                    mid_pt[0],
                    mid_pt[1],
                    str(i),
                    color="#66FF66",
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000", edgecolor="none", alpha=0.6),
                )
        else:
            other_trajs.append(traj)
            lines = mc.LineCollection(segs, colors="#6699FF", linewidths=1.2, alpha=0.7, zorder=3)
            ax.add_collection(lines)
            if mid_pt:
                ax.text(
                    mid_pt[0],
                    mid_pt[1],
                    str(i),
                    color="white",
                    fontsize=6,
                    ha="center",
                    va="center",
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#000000", edgecolor="none", alpha=0.5),
                )

    # Intersection points
    intersection_count = 0
    if sdc_traj is not None:
        sdc_segs = traj_to_segments(sdc_traj)
        ix_pts = []
        for other_traj in other_trajs:
            other_segs = traj_to_segments(other_traj)
            for (ax1, ay1), (bx1, by1) in sdc_segs:
                for (cx1, cy1), (dx1, dy1) in other_segs:
                    if segments_intersect(ax1, ay1, bx1, by1, cx1, cy1, dx1, dy1):
                        pt = segment_intersection_point(ax1, ay1, bx1, by1, cx1, cy1, dx1, dy1)
                        if pt:
                            ix_pts.append(pt)
                            intersection_count += 1
        if ix_pts:
            ax.scatter(
                [p[0] for p in ix_pts],
                [p[1] for p in ix_pts],
                color="#FF6666",
                s=30,
                zorder=6,
                edgecolors="white",
                linewidths=0.5,
                label=f"Intersections ({len(ix_pts)})",
            )

    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    if own_fig:
        ax.figure.set_facecolor("#111111")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333")

    label = title or "Scenario"
    ax.set_title(
        f"{label}  |  agents={len(objects)}  intersections={intersection_count}", color="white", fontsize=11, pad=10
    )
    ax.legend(loc="upper right", fontsize=8, facecolor="#222", edgecolor="#444", labelcolor="white")
    ax.autoscale_view()
    return ax


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Find, rank, and export the most interactive driving scenarios")
    parser.add_argument(
        "--data_folder", type=str, default="data/processed/training", help="Source folder with scenario JSONs"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/processed/interactive_data_training",
        help="Where to copy the top-K interactive scenario JSONs",
    )
    parser.add_argument(
        "--dataframe_path",
        type=str,
        default="data/meta_info/interactive_ranking.csv",
        help="Path to save the full ranking dataframe (csv or parquet)",
    )
    parser.add_argument("--top_k", type=int, default=500, help="Number of most interactive scenarios to copy")
    parser.add_argument("--max_files", type=int, default=None, help="Cap on number of source files to process")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--prioritize_intersections",
        action="store_true",
        help="Rank scenarios with angled trajectory crossings above all others",
    )
    parser.add_argument(
        "--angle_threshold",
        type=float,
        default=5.0,
        help="Min acute angle (degrees) to count as an angled intersection (default: 15)",
    )

    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Generate plots for the top scenarios")
    parser.add_argument(
        "--plot_folder", type=str, default="data/meta_info/interactive_plots", help="Folder to store visualization PNGs"
    )
    parser.add_argument(
        "--vis_top_k", type=int, default=None, help="How many top scenarios to visualize (defaults to --top_k)"
    )
    parser.add_argument("--dpi", type=int, default=150)

    args = parser.parse_args()
    num_workers = args.num_workers or cpu_count()

    # Override the module-level angle threshold with CLI arg
    global ANGLE_THRESHOLD_RAD
    ANGLE_THRESHOLD_RAD = math.radians(args.angle_threshold)

    # ── 1. Score all scenarios ───────────────────────────────────────────────
    data_dir = Path(args.data_folder)
    json_files = sorted(data_dir.glob("*.json"))
    if args.max_files:
        json_files = json_files[: args.max_files]

    print(f"Found {len(json_files)} scenario files in {data_dir}")
    print(f"Using {num_workers} workers")
    print(f"Angle threshold: {args.angle_threshold}°")
    if args.prioritize_intersections:
        print("Prioritizing scenarios with angled intersections\n")
    else:
        print()

    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_scenario, [str(f) for f in json_files], chunksize=64),
                total=len(json_files),
                desc="Scoring scenarios",
                unit="file",
            )
        )

    # Separate successes / failures (error is last element)
    successes = [r for r in results if r[5] is None]
    failures = [r for r in results if r[5] is not None]

    if failures:
        print(f"\n{len(failures)} files failed:")
        for entry in failures[:20]:
            print(f"  {entry[0]}: {entry[5]}")

    # ── 2. Build & save dataframe ────────────────────────────────────────────
    df = pd.DataFrame(
        [(name, raw, angled, n_agents, sdc_idx) for name, raw, angled, n_agents, sdc_idx, _ in successes],
        columns=["filename", "sdc_intersections", "angled_intersections", "num_agents", "sdc_track_index"],
    )

    # Sort: if prioritize_intersections, put all scenes with angled crossings
    # first (sorted by raw count within that group), then the rest by raw count.
    # Otherwise just sort by raw count.
    if args.prioritize_intersections:
        df["has_angled"] = (df["angled_intersections"] > 0).astype(int)
        df = df.sort_values(["has_angled", "sdc_intersections"], ascending=[False, False]).reset_index(drop=True)
        df = df.drop(columns=["has_angled"])
    else:
        df = df.sort_values("sdc_intersections", ascending=False).reset_index(drop=True)

    df_path = Path(args.dataframe_path)
    df_path.parent.mkdir(parents=True, exist_ok=True)

    if df_path.suffix == ".parquet":
        df.to_parquet(df_path, index=False)
    else:
        df.to_csv(df_path, index=False)
    print(f"\nSaved ranking dataframe ({len(df)} rows) to {df_path}")

    # Summary stats
    counts = df["sdc_intersections"]
    angled = df["angled_intersections"]
    print(f"  Raw intersections    — max: {counts.max()},  median: {counts.median():.0f},  mean: {counts.mean():.1f}")
    print(f"  Angled intersections — max: {angled.max()},  median: {angled.median():.0f},  mean: {angled.mean():.1f}")
    print(f"  Zero-intersection:     {(counts == 0).sum()}")
    print(f"  Has angled (>0):       {(angled > 0).sum()}")

    # ── 2b. Plot interactivity distribution ─────────────────────────────────
    plot_dir = Path(args.plot_folder)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.set_facecolor("white")
    fig.suptitle("SDC Trajectory Intersection Distribution", color="black", fontsize=14, y=1.02)

    # (0) Histogram — raw intersection count
    ax = axes[0]
    ax.hist(counts, bins=60, color="#4A86C8", edgecolor="#2A5680", alpha=0.85)
    ax.axvline(counts.median(), color="#E04040", linestyle="--", linewidth=1.5, label=f"Median ({counts.median():.0f})")
    ax.axvline(counts.mean(), color="#2EA82E", linestyle="--", linewidth=1.5, label=f"Mean ({counts.mean():.1f})")
    ax.set_xlabel("Raw intersection count")
    ax.set_ylabel("Number of scenarios")
    ax.set_title("Raw intersections (all)")
    ax.legend(fontsize=8)

    # (1) Histogram — angled intersection count
    ax = axes[1]
    angled_nonzero = angled[angled > 0]
    if len(angled_nonzero) > 0:
        ax.hist(angled_nonzero, bins=60, color="#D4952A", edgecolor="#8B6420", alpha=0.85)
    ax.set_xlabel(f"Angled intersection count (>={args.angle_threshold}°)")
    ax.set_ylabel("Number of scenarios")
    ax.set_title(f"Angled intersections — non-zero ({len(angled_nonzero)}/{len(angled)})")

    # (2) Scatter — raw count vs angled count
    ax = axes[2]
    ax.scatter(counts, angled, s=4, alpha=0.4, color="#2EA82E", edgecolors="none")
    ax.plot(
        [0, counts.max()],
        [0, counts.max()],
        color="#E04040",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="y=x (all angled)",
    )
    ax.set_xlabel("Raw intersection count")
    ax.set_ylabel("Angled intersection count")
    ax.set_title("Raw vs Angled")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")

    plt.tight_layout()
    dist_path = plot_dir / "interactivity_distribution.png"
    fig.savefig(dist_path, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved distribution plot to {dist_path}")

    # ── 3. Copy top-K to output folder ───────────────────────────────────────
    top_df = df.head(args.top_k)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for filename in tqdm(top_df["filename"], desc="Copying top-K files", unit="file"):
        src = data_dir / filename
        dst = out_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1

    print(f"\nCopied {copied} interactive scenarios to {out_dir}")

    # Print top 10
    print("\nTop 10 most interactive scenarios:")
    for _, row in top_df.head(10).iterrows():
        print(
            f"  {row['filename']:>55s}  raw={row['sdc_intersections']:4d}  "
            f"angled={row['angled_intersections']:4d}  agents={row['num_agents']}"
        )

    # ── 4. Visualize (optional) ──────────────────────────────────────────────
    if args.visualize:
        vis_k = args.vis_top_k or args.top_k
        vis_df = df.head(vis_k)

        plot_dir = Path(args.plot_folder)
        plot_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRendering {len(vis_df)} scenario plots to {plot_dir}")
        for idx, row in tqdm(vis_df.iterrows(), total=len(vis_df), desc="Plotting", unit="plot"):
            fpath = data_dir / row["filename"]
            if not fpath.exists():
                continue

            with open(fpath) as f:
                data = json.load(f)

            fig, ax = plt.subplots(1, 1, figsize=(14, 14))
            fig.set_facecolor("#111111")
            visualize_scenario(data, title=row["filename"], ax=ax)

            out_name = fpath.stem + ".png"
            fig.savefig(plot_dir / out_name, dpi=args.dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {len(vis_df)} plots to {plot_dir}")


if __name__ == "__main__":
    main()
