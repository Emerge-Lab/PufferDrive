#!/usr/bin/env python3
"""scan_map_bins.py — Scan PufferDrive map .bin files for non-finite values.

Checks every road field (x/y/z/headings/speed_limit/length/cum_lengths) and
every agent trajectory column for NaN/Inf. A single poisoned map silently
corrupts observations and crashes eval with an opaque action-space error.

Usage:
    python data_utils/scan_map_bins.py <map_dir_or_file> [...]
"""

import math
import sys
from pathlib import Path

from mirror_map_bin import read_bin


def scan_file(path):
    data = read_bin(path)
    findings = []
    for road in data["roads"]:
        for key in ("x", "y", "z", "headings", "cum_lengths"):
            values = road.get(key) or ()
            bad_count = sum(1 for v in values if not math.isfinite(v))
            if bad_count:
                findings.append(f"road {road['id']} type {road['type']} {key}: {bad_count} non-finite")
        for key in ("speed_limit", "length"):
            if key in road and not math.isfinite(road[key]):
                findings.append(f"road {road['id']} type {road['type']} {key}={road[key]}")
    for agent in data["agents"]:
        for key, values in agent["cols"].items():
            bad_count = sum(1 for v in values if not math.isfinite(v))
            if bad_count:
                findings.append(f"agent {agent['id']} {key}: {bad_count} non-finite")
        if any(not math.isfinite(v) for v in agent["goal"]):
            findings.append(f"agent {agent['id']} goal: {agent['goal']}")
    return findings


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bin_paths = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        bin_paths += sorted(path.glob("*.bin")) if path.is_dir() else [path]
    if not bin_paths:
        sys.exit("no .bin files found")

    bad_file_count = 0
    for i, path in enumerate(bin_paths):
        try:
            findings = scan_file(path)
        except Exception as parse_error:
            bad_file_count += 1
            print(f"[{i + 1}/{len(bin_paths)}] {path.name}: PARSE ERROR: {parse_error}")
            continue
        if findings:
            bad_file_count += 1
            print(f"[{i + 1}/{len(bin_paths)}] {path.name}:")
            for finding in findings:
                print(f"    {finding}")
        elif (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(bin_paths)}] ...clean so far")
    print(f"\n{len(bin_paths)} files scanned, {bad_file_count} with problems")
    sys.exit(1 if bad_file_count else 0)


if __name__ == "__main__":
    main()
