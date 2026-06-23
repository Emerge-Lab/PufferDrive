#!/usr/bin/env python3
"""plot_obs_stride.py — Visualize the effect of obs stride on road geometry.

Replicates the C `valid_for_obs` selection from init_grid_map (drive.h): for
lanes (obs_lane_stride) and road edges (obs_boundary_stride) it keeps one point
every `stride` points, plus any point whose heading deviates more than
OBS_STRIDE_HEADING_THRESHOLD from the last kept point (densifies curves).

Outputs one self-contained zoomable HTML per CARLA map: kept points highlighted,
dropped points faint, full polylines underneath as context. Pan/zoom in browser.

Usage:
    python data_utils/plot_obs_stride.py \\
        --input pufferlib/resources/drive/binaries/carla \\
        --output obs_stride_plots \\
        --lane-stride 4 --boundary-stride 4
"""

import argparse
import math
from pathlib import Path

import plotly.graph_objects as go

from mirror_map_bin import read_bin


def is_road_lane(t):
    return 0 <= t <= 9


def is_road_edge(t):
    return 20 <= t <= 29


def is_road_line(t):
    return 10 <= t <= 19


def normalize_heading(h):
    h = math.fmod(h, 2.0 * math.pi)
    if h > math.pi:
        h -= 2.0 * math.pi
    elif h < -math.pi:
        h += 2.0 * math.pi
    return h


def valid_for_obs_mask(headings, stride, heading_threshold):
    """Same selection as drive.h init_grid_map: per segment j in [0, S-1)."""
    n_seg = len(headings) - 1
    mask = [True] * max(n_seg, 0)
    if stride <= 1:
        return mask
    last_kept = 0
    for j in range(n_seg):
        keep = True
        if j > 0:
            dev = abs(normalize_heading(headings[j] - headings[last_kept]))
            keep = (j - last_kept) >= stride or dev > heading_threshold
        if keep:
            last_kept = j
        mask[j] = keep
    return mask


def _poly_trace(roads, predicate, color, name):
    """One faint Scatter with NaN separators for every matching polyline."""
    xs, ys = [], []
    for r in roads:
        if not predicate(r["type"]) or r["S"] < 2:
            continue
        xs.extend(r["x"])
        xs.append(None)
        ys.extend(r["y"])
        ys.append(None)
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="lines",
        name=name,
        line=dict(color=color, width=1),
        opacity=0.35,
        hoverinfo="skip",
    )


def _point_trace(xs, ys, color, name, size, opacity):
    return go.Scattergl(
        x=xs,
        y=ys,
        mode="markers",
        name=name,
        marker=dict(color=color, size=size),
        opacity=opacity,
        hovertemplate="%{x:.1f}, %{y:.1f}<extra>" + name + "</extra>",
    )


def build_figure(data, map_name, lane_stride, boundary_stride, heading_threshold):
    roads = data["roads"]
    kept = {"lane": ([], []), "edge": ([], [])}
    dropped = {"lane": ([], []), "edge": ([], [])}
    counts = {"lane": [0, 0], "edge": [0, 0]}  # [kept, total]

    for r in roads:
        if is_road_lane(r["type"]):
            cat, stride = "lane", lane_stride
        elif is_road_edge(r["type"]):
            cat, stride = "edge", boundary_stride
        else:
            continue
        if r["S"] < 2:
            continue
        mask = valid_for_obs_mask(r["headings"], stride, heading_threshold)
        for j, keep in enumerate(mask):
            counts[cat][1] += 1
            bucket = kept[cat] if keep else dropped[cat]
            bucket[0].append(r["x"][j])
            bucket[1].append(r["y"][j])
            if keep:
                counts[cat][0] += 1

    fig = go.Figure()
    # Context polylines underneath.
    fig.add_trace(_poly_trace(roads, is_road_lane, "#9bb7e0", "lane polyline"))
    fig.add_trace(_poly_trace(roads, is_road_edge, "#222831", "edge polyline"))
    fig.add_trace(_poly_trace(roads, is_road_line, "#b0b0b0", "marking polyline"))
    # Dropped points (faint), then kept (bright) on top.
    fig.add_trace(_point_trace(*dropped["lane"], "#d98c8c", "lane dropped", 3, 0.45))
    fig.add_trace(_point_trace(*dropped["edge"], "#8c8cd9", "edge dropped", 3, 0.45))
    fig.add_trace(_point_trace(*kept["lane"], "#1f77b4", "lane kept", 5, 0.9))
    fig.add_trace(_point_trace(*kept["edge"], "#000000", "edge kept", 5, 0.9))

    lk, lt = counts["lane"]
    ek, et = counts["edge"]
    lpct = 100 * lk / lt if lt else 0
    epct = 100 * ek / et if et else 0
    fig.update_layout(
        title=(
            f"{map_name} — lane stride {lane_stride} (kept {lk}/{lt}, {lpct:.0f}%) | "
            f"edge stride {boundary_stride} (kept {ek}/{et}, {epct:.0f}%) | "
            f"heading keep {math.degrees(heading_threshold):.0f}°"
        ),
        showlegend=True,
        dragmode="pan",
        plot_bgcolor="white",
        xaxis=dict(title="x (m)", showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="y (m)", showgrid=True, gridcolor="#eee", scaleanchor="x", scaleratio=1),
    )
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input", default="pufferlib/resources/drive/binaries/carla", help="Directory of *.bin maps or a single .bin"
    )
    p.add_argument("--output", default="obs_stride_plots", help="Output directory for HTML")
    p.add_argument("--lane-stride", type=int, default=2)
    p.add_argument("--boundary-stride", type=int, default=1)
    p.add_argument(
        "--heading-threshold-deg",
        type=float,
        default=15.0,
        help="Force-keep a point when heading drifts this many degrees from the last kept point",
    )
    p.add_argument("--glob", default="*.bin", help="Glob for selecting maps in a directory")
    args = p.parse_args()
    heading_threshold = math.radians(args.heading_threshold_deg)

    src = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bins = [src] if src.is_file() else sorted(src.glob(args.glob))
    if not bins:
        raise SystemExit(f"No maps matched {src}/{args.glob}")

    for b in bins:
        data = read_bin(b)
        fig = build_figure(data, b.stem, args.lane_stride, args.boundary_stride, heading_threshold)
        html = (
            out / f"{b.stem}_lane{args.lane_stride}_edge{args.boundary_stride}_hdg{args.heading_threshold_deg:g}.html"
        )
        fig.write_html(html, include_plotlyjs="cdn", config={"scrollZoom": True, "displaylogo": False})
        print(f"  {b.name} -> {html}")

    print(f"\nWrote {len(bins)} plot(s) to {out}/")


if __name__ == "__main__":
    main()
