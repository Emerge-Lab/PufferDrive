import pandas as pd
from pathlib import Path

BENCHMARK_DIR = Path("benchmark")
SUMMARY_FILE = "evaluation_summary.csv"

DROP_METRICS = {"episode_id", "dnf_rate", "episode_return"}
META_COLS = [
    "model_name",
    "xp",
    "eval_mode",
    "score",
    "collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
    # "stop_sign_violation_rate",
    "num_goals_reached",
    "avg_speed_per_agent",
    "avg_distance_per_infraction",
]

rows = []

for model_dir in BENCHMARK_DIR.iterdir():
    if not model_dir.is_dir():
        continue

    for summary_path in model_dir.rglob(SUMMARY_FILE):
        # model/xp/eval_mode/evaluation_summary.csv
        try:
            xp = summary_path.parents[1].name
            eval_mode = summary_path.parent.name
        except IndexError:
            continue

        df = pd.read_csv(summary_path)  # Metric | Average

        metrics = {metric: value for metric, value in zip(df["Metric"], df["Average"]) if metric not in DROP_METRICS}

        row = {
            "model_name": model_dir.name,
            "xp": xp,
            "eval_mode": eval_mode,
            **metrics,
        }
        rows.append(row)

# Build dataframe
out_df = pd.DataFrame(rows)
out_df = out_df.round(3)

# Enforce column order
metric_cols = [c for c in out_df.columns if c not in META_COLS]
out_df = out_df[META_COLS + metric_cols]

out_df.to_csv("benchmark/benchmark_summary.csv", index=False)

print("✅ Saved benchmark_all_summaries.csv")
