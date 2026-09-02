"""Render the interactive observation replays (obs_html/<token>.replay.zlib) of a nuPlan run into HTML,
only for the scenarios worth looking at.

usage: python scripts/eval/render_obs_html.py <group_dir> [--max-score 0.9] [--metric no_ego_at_fault_collisions ...]
       [--tokens tok1 tok2 ...] [--prune]

Selection: aggregated score < --max-score OR any --metric < 1 OR explicitly listed --tokens.
--prune deletes the .replay.zlib of every scenario that was not selected.
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from pufferlib.ocean.cosim.obs_replay import render_replay_html  # noqa: E402


def selected_tokens(group_dir, max_score, metrics):
    tokens = set()
    for csv in glob.glob(f"{group_dir}/simulation/*/20*/aggregator_metric/*_weighted_average_metrics_*.csv"):
        agg = pd.read_csv(csv)
        rows = agg[(agg["scenario_type"] != "final_score") & (agg["scenario"] != agg["scenario_type"])]
        mask = rows["score"] < max_score
        for metric in metrics:
            mask |= rows[metric] < 1.0
        tokens |= set(rows[mask]["scenario"].astype(str))
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group_dir")
    ap.add_argument("--max-score", type=float, default=1.01)
    ap.add_argument("--metric", action="append", default=[])
    ap.add_argument("--tokens", nargs="*", default=[])
    ap.add_argument("--prune", action="store_true")
    args = ap.parse_args()
    replays = {Path(p).name[: -len(".replay.zlib")]: p for p in glob.glob(f"{args.group_dir}/obs_html/*.replay.zlib")}
    if not replays:
        raise SystemExit(f"no obs_html/*.replay.zlib under {args.group_dir}")
    wanted = selected_tokens(args.group_dir, args.max_score, args.metric) | set(args.tokens)
    rendered, missing = 0, 0
    for token in sorted(wanted):
        if token not in replays:
            missing += 1
            continue
        render_replay_html(replays[token])
        rendered += 1
    pruned = 0
    if args.prune:
        for token, path in replays.items():
            if token not in wanted:
                Path(path).unlink()
                pruned += 1
    print(
        f"[render_obs_html] {len(replays)} replays, {len(wanted)} scenarios selected, {rendered} pages rendered"
        f"{f', {missing} selected without a replay' if missing else ''}{f', {pruned} replays pruned' if pruned else ''}"
        f" -> {args.group_dir}/obs_html"
    )


if __name__ == "__main__":
    main()
