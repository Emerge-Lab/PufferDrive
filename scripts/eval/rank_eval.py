import argparse
import glob
import os
from collections import defaultdict

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--key", default="avg_distance_per_infraction")
parser.add_argument("--runs_dir", default="runs")
parser.add_argument("--ascending", action="store_true", help="sort ascending (default descending)")
args = parser.parse_args()

pattern = os.path.join(args.runs_dir, "*", "final_evaluation", "*", "*", "evaluation_summary.csv")
display_keys = [
    "offroad_rate",
    "collision_rate",
    "score",
    "avg_speed_per_agent",
    "total_infractions",
]
display_names = {
    "avg_distance_per_infraction": "distance_per_infraction",
    "offroad_rate": "offroad",
    "collision_rate": "collision",
    "avg_speed_per_agent": "avg_speed",
    "total_infractions": "infractions",
}

per_dataset = defaultdict(list)
for csv_path in glob.glob(pattern):
    parts = csv_path.split(os.sep)
    dataset = parts[-2]
    model = parts[-3]
    run = parts[-5]
    metrics = pd.read_csv(csv_path).set_index("Metric")["Average"]
    if args.key not in metrics.index:
        continue
    display_metrics = {key: float(metrics[key]) for key in display_keys if key in metrics.index}
    per_dataset[dataset].append((run, model, float(metrics[args.key]), display_metrics))

if not per_dataset:
    raise SystemExit(f"No evaluation_summary.csv with key '{args.key}' under {args.runs_dir}")

for dataset in sorted(per_dataset):
    rows = sorted(per_dataset[dataset], key=lambda r: r[2], reverse=not args.ascending)
    print(f"\n=== {dataset} (sorted by {args.key}, {'asc' if args.ascending else 'desc'}) ===")
    width = max(len(r[0]) for r in rows)
    shown_keys = []
    for key in [args.key] + display_keys:
        if key in shown_keys:
            continue
        if key == args.key or any(key in row[3] for row in rows):
            shown_keys.append(key)
    column_widths = {key: max(len(display_names.get(key, key)), 8) for key in shown_keys}
    header = " | ".join(f"{display_names.get(key, key):>{column_widths[key]}}" for key in shown_keys)
    print(f"{'':>4} {'run':<{width}} | {header} | model")
    print(f"{'-' * 4} {'-' * width} | {' | '.join('-' * column_widths[key] for key in shown_keys)} | {'-' * 20}")
    for rank, (run, model, value, display_metrics) in enumerate(rows, 1):
        values = []
        for key in shown_keys:
            metric_value = value if key == args.key else display_metrics.get(key)
            if metric_value is None:
                values.append(f"{'-':>{column_widths[key]}}")
                continue
            if key in ("avg_distance_per_infraction", "total_infractions"):
                metric_text = f"{metric_value:,.0f}"
            else:
                metric_text = f"{metric_value:.4f}"
            values.append(f"{metric_text:>{column_widths[key]}}")
        print(f"{rank:>3}. {run:<{width}} | {' | '.join(values)} | [{model}]")
