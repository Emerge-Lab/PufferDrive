"""Summarize [DRIVE DIAG] short-early-reset lines from training stderr logs.

Usage: python scripts/kesai/summarize_reset_diag.py LOG.err [LOG2.err ...]
"""
import re
import sys
from collections import defaultdict

LINE_RE = re.compile(r"\[DRIVE DIAG\] short early reset: (.*)")
FIELD_RE = re.compile(r"(\w+)=(\S+)")
SUM_FIELDS = [
    "active",
    "removed",
    "stopped",
    "spawn_failed",
    "stopped_at_reset",
    "collision",
    "offroad",
    "stop_line",
    "empty_cell",
    "goal_failed",
]


def main(paths):
    per_map = defaultdict(lambda: {"lines": 0, **{k: 0.0 for k in SUM_FIELDS}})
    seeds = defaultdict(list)
    for path in paths:
        with open(path, errors="replace") as handle:
            for line in handle:
                match = LINE_RE.search(line)
                if not match:
                    continue
                fields = dict(FIELD_RE.findall(match.group(1)))
                entry = per_map[fields["map"].rsplit("/", 1)[-1]]
                entry["lines"] += 1
                for key in SUM_FIELDS:
                    entry[key] += float(fields[key])
                seeds[fields["map"].rsplit("/", 1)[-1]].append(fields["episode_seed"])
    if not per_map:
        print("no [DRIVE DIAG] lines found")
        return
    print(f"{'map':40s} {'lines':>5s} {'active':>7s} {'removed':>8s} {'stopped':>8s} {'spawnfail':>9s} "
          f"{'stop@rst':>8s} {'coll':>6s} {'offrd':>6s} {'stopl':>6s} {'empty':>6s} {'goal':>5s}")
    for map_name, entry in sorted(per_map.items(), key=lambda kv: -kv[1]["lines"]):
        n = entry["lines"]
        print(f"{map_name:40s} {n:5d} " + " ".join(
            f"{entry[k] / n:{w}.1f}" for k, w in zip(SUM_FIELDS, (7, 8, 8, 9, 8, 6, 6, 6, 6, 5))))
    print("\nper-line means are per sub-env at the moment of the short reset; rejects are summed over all agents "
          "of that reset (max 30 per agent).")
    print("example episode seeds per map (rerun locally with use_exact_episode_seed):")
    for map_name, values in sorted(seeds.items()):
        print(f"  {map_name}: {values[:3]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
