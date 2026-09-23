"""Summarize ReGentS target-collision outcomes for every downloaded generation run.

Reproduces the columns of experiments/regents/seed_29_collision_summary.csv from each
run's generation_metrics.csv, across all <generation>/<cohort> pairs found under the
results root. Hitter-compliance columns are added separately by the replay script.
"""

import argparse
import csv
import re
from pathlib import Path

SCENARIOS_PER_GENERATION = 10000
GENERATION_ORDER = ("regents_idm", "regents_corridor_idm", "regents_pdm", "regents_rl_nocond", "regents_rl_cond")
MAP_COHORT_PATTERN = re.compile(r"carla_generated_policy_10k_100_(seed_\d+)")

NATIVE_VALID = "eval_hitter_compliance_valid"
NATIVE_COMPLIANT = "eval_hitter_compliance_compliant"

SUMMARY_FIELDS = (
    "cohort",
    "generation",
    "map_cohort",
    "hitter_source",
    "genuine_failure_pct_of_scenarios",
    "unavoidable_pct_of_scenarios",
    "adversary_forced_pct_of_scenarios",
    "genuine_failure_count",
    "unavoidable_count",
    "adversary_forced_count",
    "scenario_count",
    "target_collision_count",
    "generation_success_count",
    "actionable_collision_count",
    "successful_genuine_failure_count",
    "successful_unavoidable_count",
    "successful_adversary_forced_count",
    "hitter_compliance_valid_count",
    "hitter_compliant_count",
    "hitter_compliance_pct_of_target_collisions",
    "hitter_compliance_pct_of_valid_collisions",
    "hitter_compliant_pct_of_scenarios",
    "collision_mismatch_count",
)


def summarize_run(metrics_path):
    """Count target-collision outcomes for one generation run."""
    with metrics_path.open(newline="") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    if len(rows) != SCENARIOS_PER_GENERATION:
        raise ValueError(f"{metrics_path} has {len(rows)} scenarios, expected {SCENARIOS_PER_GENERATION}")

    map_cohorts = {
        match.group(1)
        for match in (MAP_COHORT_PATTERN.search(row["eval_map_name"]) for row in rows)
        if match is not None
    }
    if len(map_cohorts) != 1:
        raise ValueError(f"{metrics_path} mixes map cohorts: {sorted(map_cohorts)}")

    counts = dict.fromkeys(
        (
            "target_collision_count",
            "generation_success_count",
            "actionable_collision_count",
            "genuine_failure_count",
            "unavoidable_count",
            "adversary_forced_count",
            "successful_genuine_failure_count",
            "successful_unavoidable_count",
            "successful_adversary_forced_count",
        ),
        0,
    )
    outcome_columns = {
        "genuine_failure": "eval_sdc_target_collision_genuine_failure_rate",
        "unavoidable": "eval_sdc_target_collision_unavoidable_rate",
        "adversary_forced": "eval_sdc_target_collision_adversary_forced_rate",
    }
    for row in rows:
        generation_success = int(float(row["generation_success"]))
        counts["target_collision_count"] += int(float(row["eval_sdc_target_collision_rate"]))
        counts["generation_success_count"] += generation_success
        counts["actionable_collision_count"] += int(float(row["actionable_collision"]))
        for outcome, column in outcome_columns.items():
            outcome_value = int(float(row[column]))
            counts[f"{outcome}_count"] += outcome_value
            if generation_success:
                counts[f"successful_{outcome}_count"] += outcome_value

    outcome_total = counts["genuine_failure_count"] + counts["unavoidable_count"] + counts["adversary_forced_count"]
    if outcome_total != counts["target_collision_count"]:
        raise ValueError(f"{metrics_path} outcomes {outcome_total} do not partition {counts['target_collision_count']}")

    native = {}
    if NATIVE_VALID in rows[0]:
        valid_count = sum(int(float(row[NATIVE_VALID])) for row in rows)
        compliant_count = sum(int(float(row[NATIVE_COMPLIANT])) for row in rows)
        target_count = counts["target_collision_count"]
        native = {
            "hitter_source": "native",
            "hitter_compliance_valid_count": valid_count,
            "hitter_compliant_count": compliant_count,
            "hitter_compliance_pct_of_target_collisions": (
                f"{100 * compliant_count / target_count:.2f}%" if target_count else ""
            ),
            "hitter_compliance_pct_of_valid_collisions": (
                f"{100 * compliant_count / valid_count:.2f}%" if valid_count else ""
            ),
            "hitter_compliant_pct_of_scenarios": f"{100 * compliant_count / SCENARIOS_PER_GENERATION:.2f}%",
            "collision_mismatch_count": 0,
        }

    percent = lambda count: f"{100 * count / SCENARIOS_PER_GENERATION:.2f}%"
    return {
        **native,
        "map_cohort": map_cohorts.pop(),
        "scenario_count": SCENARIOS_PER_GENERATION,
        "genuine_failure_pct_of_scenarios": percent(counts["genuine_failure_count"]),
        "unavoidable_pct_of_scenarios": percent(counts["unavoidable_count"]),
        "adversary_forced_pct_of_scenarios": percent(counts["adversary_forced_count"]),
        **counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/regents_res")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        parser.error(f"{results_root} is not a directory")
    output_path = Path(args.output) if args.output else results_root / "collision_summary.csv"

    summary_rows = []
    for metrics_path in sorted(results_root.glob("*/*/generation_metrics.csv")):
        generation = metrics_path.parents[1].name
        cohort = metrics_path.parent.name
        summary_rows.append({"cohort": cohort, "generation": generation, **summarize_run(metrics_path)})
    if not summary_rows:
        parser.error(f"no generation_metrics.csv found under {results_root}")

    generation_rank = {name: index for index, name in enumerate(GENERATION_ORDER)}
    summary_rows.sort(key=lambda row: (row["cohort"], generation_rank.get(row["generation"], len(GENERATION_ORDER))))

    with output_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {output_path} with {len(summary_rows)} runs")


if __name__ == "__main__":
    main()
