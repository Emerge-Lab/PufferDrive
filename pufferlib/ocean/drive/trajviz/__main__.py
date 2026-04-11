"""CLI entry point for the trajviz Vulkan renderer.

Usage:
    python -m pufferlib.ocean.drive.trajviz INPUT [INPUT...] \\
        --maps-dir DIR --out DIR \\
        [--width 1280] [--height 720] [--fps 30] \\
        [--views topdown,bev]

INPUT can be a saved trajectories_*.npz file or a directory to glob
recursively. One Vulkan context is created up front and reused for every
episode in every input file — pay the GPU init cost once for an entire
batch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pufferlib.ocean.drive.trajviz import Renderer, render_npz


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m pufferlib.ocean.drive.trajviz",
        description="Vulkan offline renderer for saved Drive trajectories.",
    )
    p.add_argument("inputs", nargs="+", type=Path, help="trajectories_*.npz files (or directories to glob).")
    p.add_argument(
        "--maps-dir", type=Path, required=True, help="Directory containing the .bin map files referenced in the npz."
    )
    p.add_argument("--out", type=Path, required=True, help="Output directory for MP4 files.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--views", default="topdown,bev", help="Comma-separated subset of {topdown, bev}.")
    args = p.parse_args()

    npz_files: list[Path] = []
    for inp in args.inputs:
        if inp.is_dir():
            npz_files.extend(sorted(inp.rglob("trajectories_*.npz")))
        else:
            npz_files.append(inp)

    if not npz_files:
        raise SystemExit("No trajectories_*.npz files found.")

    views = tuple(v.strip() for v in args.views.split(",") if v.strip())

    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    with Renderer(width=args.width, height=args.height) as renderer:
        for npz in npz_files:
            print(f"[{npz}]")
            out_paths = render_npz(
                npz,
                args.maps_dir,
                args.out,
                width=args.width,
                height=args.height,
                fps=args.fps,
                views=views,
                renderer=renderer,
            )
            total += len(out_paths)

    print(f"Wrote {total} MP4 files to {args.out}")


if __name__ == "__main__":
    main()
