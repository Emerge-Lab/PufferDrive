"""Paired comparison of two nuPlan closed-loop eval runs of the PufferDrive planner, overall and per city.

usage: python scripts/eval/compare_nuplan_evals.py <baseline_group_dir> <candidate_group_dir> [--worst N]

Each group dir is a run_simulation.py `group` directory (scripts/kesai/9_nuPlan.sh writes them under
$RUN_DIR/eval); its aggregator_metric/scenarios.csv is compared on the scenarios both runs contain. The city of
a scenario is read from the nuPlan log database under $NUPLAN_DATA_ROOT/nuplan-v1.1/splits/val when that exists.
"""

import argparse
import functools
import glob
import os
import sqlite3

import numpy as np
import pandas as pd

METRICS = (
    "score",
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
    "time_to_collision_within_bound",
    "ego_progress_along_expert_route",
    "speed_limit_compliance",
    "ego_is_comfortable",
)
BOOTSTRAP_SAMPLES = 5000
SCORE_EPSILON = 1e-9


@functools.lru_cache(maxsize=None)
def city_of_log(log_name):
    db_dir = os.path.join(os.environ.get("NUPLAN_DATA_ROOT", ""), "nuplan-v1.1", "splits", "val")
    db_path = os.path.join(db_dir, f"{log_name}.db")
    if not os.path.isfile(db_path):
        return "unknown"
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    location = connection.execute("select location from log").fetchone()[0]
    connection.close()
    return location


def load_scenarios(group_dir):
    paths = glob.glob(os.path.join(group_dir, "simulation", "*", "*", "aggregator_metric", "scenarios.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one aggregator_metric/scenarios.csv under {group_dir}, found {paths}")
    scenarios = pd.read_csv(paths[0])
    scenarios["city"] = scenarios["log_name"].map(city_of_log)
    return scenarios.set_index("scenario")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline_group_dir")
    parser.add_argument("candidate_group_dir")
    parser.add_argument("--worst", type=int, default=10, help="scenarios listed at both ends of the delta ranking")
    args = parser.parse_args()

    baseline = load_scenarios(args.baseline_group_dir)
    candidate = load_scenarios(args.candidate_group_dir)
    common = baseline.index.intersection(candidate.index)
    print(f"baseline  {os.path.basename(os.path.normpath(args.baseline_group_dir))}: {len(baseline)} scenarios")
    print(f"candidate {os.path.basename(os.path.normpath(args.candidate_group_dir))}: {len(candidate)} scenarios")
    print(f"compared on {len(common)} common scenarios\n")
    baseline, candidate = baseline.loc[common], candidate.loc[common]

    rows = []
    for metric in METRICS:
        delta = candidate[metric] - baseline[metric]
        rows.append(
            {
                "metric": metric,
                "baseline": baseline[metric].mean(),
                "candidate": candidate[metric].mean(),
                "delta": delta.mean(),
                "improved": int((delta > SCORE_EPSILON).sum()),
                "worsened": int((delta < -SCORE_EPSILON).sum()),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    per_city = pd.DataFrame(
        {
            "n": baseline.groupby("city").size(),
            "baseline": baseline.groupby("city")["score"].mean(),
            "candidate": candidate.groupby("city")["score"].mean(),
        }
    )
    per_city["delta"] = per_city["candidate"] - per_city["baseline"]
    print("\nscore per city:")
    print(per_city.to_string(float_format=lambda value: f"{value:.4f}"))

    score_delta = candidate["score"] - baseline["score"]
    rng = np.random.default_rng(0)
    bootstrap_means = [
        rng.choice(score_delta.values, len(score_delta), replace=True).mean() for _ in range(BOOTSTRAP_SAMPLES)
    ]
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    print(f"\nscore delta {score_delta.mean():+.4f} (paired bootstrap 95% CI [{low:+.4f}, {high:+.4f}])")

    changed = score_delta[score_delta.abs() > SCORE_EPSILON].sort_values()
    listing = pd.concat([changed.head(args.worst), changed.tail(args.worst)]).to_frame("delta")
    listing = listing.join(baseline[["city", "score"]].rename(columns={"score": "baseline"}))
    listing = listing.join(candidate[["score"]].rename(columns={"score": "candidate"}))
    print(f"\n{len(changed)} scenarios changed; largest regressions and gains:")
    print(listing.to_string(float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
