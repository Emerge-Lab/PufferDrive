#!/usr/bin/env python3
"""
Convert a tree of GPUDrive JSON maps into engine-ready binary maps.

Usage:
  scripts/convert_gpudrive_bins.py --src ../temp/GPUDrive/training \
      [--out resources/drive/binaries] [--start-index auto] [--limit N]

Notes:
  - Recursively scans for *.json under --src
  - Appends new binaries as map_<index>.bin, continuing after the max existing index
  - Safe to re-run; it skips existing outputs by index
"""

import argparse
import os
import re
from pathlib import Path

from pufferlib.ocean.drive.drive import load_map


def find_next_index(out_dir: Path) -> int:
    pat = re.compile(r"map_(\d+)\.bin$")
    max_idx = -1
    for p in out_dir.glob("map_*.bin"):
        m = pat.search(p.name)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def iter_json(src: Path):
    # Recursively yield JSON files
    for p in src.rglob("*.json"):
        if p.is_file():
            yield p


def main():
    ap = argparse.ArgumentParser(description="Convert GPUDrive JSON maps into binary maps for PufferDrive.")
    ap.add_argument("--src", required=True, help="Path to training/ root (contains group_* or flat JSONs)")
    ap.add_argument("--out", default="resources/drive/binaries", help="Output binaries dir (default: resources/drive/binaries)")
    ap.add_argument("--start-index", type=int, default=None, help="Starting map index (default: auto, after max existing)")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of JSON files to convert (0 = all)")
    ap.add_argument("--log-every", type=int, default=50, help="Progress interval (default: 50)")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise SystemExit(f"[fatal] src not found: {src}")

    start_idx = args.start_index if args.start_index is not None else find_next_index(out_dir)
    done = 0
    failed = 0
    idx = start_idx

    for jpath in iter_json(src):
        ofile = out_dir / f"map_{idx:03d}.bin"
        if ofile.exists() and ofile.stat().st_size > 0:
            idx += 1
            continue
        try:
            load_map(str(jpath), str(ofile))
            done += 1
        except Exception as e:
            failed += 1
            # remove possibly partial file
            try:
                if ofile.exists():
                    ofile.unlink()
            except Exception:
                pass
        idx += 1
        if args.log_every and (done + failed) % args.log_every == 0:
            print(f"[progress] converted={done} failed={failed} last_index={idx-1}")
        if args.limit and done >= args.limit:
            break

    total_bins = sum(1 for _ in out_dir.glob("map_*.bin"))
    print(f"[done] converted={done} failed={failed} start_index={start_idx} next_index={idx} total_bins={total_bins}")


if __name__ == "__main__":
    main()

