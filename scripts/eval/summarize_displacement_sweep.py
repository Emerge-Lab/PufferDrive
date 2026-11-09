#!/usr/bin/env python3
"""Tabulate a displacement-coef eval sweep next to the nuplan_policies val_standard baselines.

Usage:
    python scripts/eval/summarize_displacement_sweep.py <eval_dir> <out_csv> --policy <label> [--benchmark-prefix <prefix>]

<eval_dir> is the checkpoint's eval root: it holds eval/<prefix>_<label>/<timestamp>/evaluation_summary.json,
as written by pufferl eval with nuplan_policies/eval_configs/val_displacement_sweep*.yaml.
"""

import argparse
import glob
import json
import os

import pandas as pd
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP_LABELS = ("min", "p25", "mid", "p75", "max")
METRIC_COLUMNS = [
    "score",
    "puffer_score",
    "nuplan/score",
    "collision_rate",
    "at_fault_collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
    "dnf_rate",
    "num_goals_reached",
    "progress_ratio",
    "displacement_error",
    "avg_speed_per_agent",
    "episode_length",
    "comfort_violation_count",
    "reward_components/displacement",
]


def summary_row(summary_path, policy, displacement_coef):
    with open(summary_path) as summary_file:
        summary = json.load(summary_file)
    metrics = summary["metrics_mean"]
    row = {"policy": policy, "displacement_coef": displacement_coef, "maps": summary["num_episodes"]}
    row.update({column: metrics.get(column) for column in METRIC_COLUMNS})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir")
    parser.add_argument("out_csv")
    parser.add_argument("--policy", required=True, help="label for the swept checkpoint's rows")
    parser.add_argument("--benchmark-prefix", default="val_displacement", help="sweep benchmark names minus _<label>")
    args = parser.parse_args()

    rows = []
    for label in SWEEP_LABELS:
        benchmark = f"{args.benchmark_prefix}_{label}"
        pattern = os.path.join(args.eval_dir, "eval", benchmark, "*", "evaluation_summary.json")
        summary_paths = sorted(glob.glob(pattern))
        if not summary_paths:
            raise FileNotFoundError(f"no eval output matches {pattern}")
        with open(os.path.join(os.path.dirname(summary_paths[-1]), "resolved_benchmark.yaml")) as resolved_file:
            displacement_coef = yaml.safe_load(resolved_file)["args"]["env"]["reward_displacement"]
        rows.append(summary_row(summary_paths[-1], args.policy, displacement_coef))
    baseline_pattern = os.path.join(
        REPO_ROOT, "nuplan_policies", "*", "eval", "val_standard", "*", "evaluation_summary.json"
    )
    for summary_path in sorted(glob.glob(baseline_pattern)):
        rows.append(summary_row(summary_path, summary_path.split(os.sep)[-5], None))

    table = pd.DataFrame(rows)
    table.to_csv(args.out_csv, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    short = table.rename(columns=lambda c: c.split("/")[-1].replace("_rate", "").replace("violation_", "")[:11])
    short["policy"] = short["policy"].str.replace("sdc-only-", "sdc:").str[:34]
    print(short.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
