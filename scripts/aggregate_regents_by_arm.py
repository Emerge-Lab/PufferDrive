"""Aggregate the ReGentS collision summary across map seeds, one row per mode and arm."""

import argparse
import csv
import statistics
from pathlib import Path

SCENARIOS_PER_GENERATION = 10000
ARM_ORDER = ("regents_idm", "regents_corridor_idm", "regents_pdm", "regents_rl_nocond", "regents_rl_cond")
COHORT_MODE_SEED = {
    "seed_29": ("regents", 29),
    "seed_4_n2": ("regents", 4),
    "seed_99": ("regents", 99),
    "seed_29_king": ("king", 29),
    "seed_4_king_n1": ("king", 4),
    "seed_99_king": ("king", 99),
    "seed_29_regents_rlcond_v2": ("regents", 29),
    "seed_4_regents_rlcond_v2": ("regents", 4),
    "seed_99_regents_rlcond_v2": ("regents", 99),
    "seed_29_king_rlcond_v2": ("king", 29),
    "seed_4_king_rlcond_v2": ("king", 4),
    "seed_99_king_rlcond_v2": ("king", 99),
}
METRICS = (
    ("collisions_pct", lambda row: 100 * int(row["target_collision_count"]) / SCENARIOS_PER_GENERATION),
    ("genuine_pct", lambda row: 100 * int(row["genuine_failure_count"]) / SCENARIOS_PER_GENERATION),
    ("unavoidable_pct", lambda row: 100 * int(row["unavoidable_count"]) / SCENARIOS_PER_GENERATION),
    ("adv_forced_pct", lambda row: 100 * int(row["adversary_forced_count"]) / SCENARIOS_PER_GENERATION),
    ("gen_success_pct", lambda row: 100 * int(row["generation_success_count"]) / SCENARIOS_PER_GENERATION),
    ("genuine_of_coll_pct", lambda row: 100 * int(row["genuine_failure_count"]) / int(row["target_collision_count"])),
    ("hitter_of_coll_pct", lambda row: 100 * int(row["hitter_compliant_count"]) / int(row["target_collision_count"])),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/regents_res")
    parser.add_argument("--output", default="")
    parser.add_argument("--exclude-arms", default="", help="Comma-separated generation names to leave out")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    summary_path = results_root / "collision_summary.csv"
    with summary_path.open(newline="") as summary_file:
        rows = list(csv.DictReader(summary_file))

    excluded = {name.strip() for name in args.exclude_arms.split(",") if name.strip()}
    grouped = {}
    for row in rows:
        if row["generation"] in excluded:
            continue
        mode, seed = COHORT_MODE_SEED[row["cohort"]]
        grouped.setdefault((mode, row["generation"]), {})[seed] = row

    output_path = Path(args.output) if args.output else results_root / "collision_summary_by_arm.csv"
    header = ["arm", "mode", "seed_count", "hitter_seeds"] + [name for name, _ in METRICS]
    table = []
    for arm in ARM_ORDER:
        for mode in ("regents", "king"):
            group = grouped.get((mode, arm))
            if not group:
                continue
            label = arm.replace("regents_", "") if mode == "regents" else "king_" + arm.replace("regents_", "")
            record = {"arm": label, "mode": mode, "seed_count": len(group)}
            for name, extract in METRICS:
                usable = [
                    group[seed]
                    for seed in sorted(group)
                    if name != "hitter_of_coll_pct" or group[seed].get("hitter_compliant_count")
                ]
                record["hitter_seeds"] = sum(1 for r in group.values() if r.get("hitter_compliant_count"))
                if not usable:
                    record[name] = ""
                    continue
                values = [extract(row) for row in usable]
                mean_value = statistics.mean(values)
                std_value = statistics.stdev(values) if len(values) > 1 else 0.0
                record[name] = f"{mean_value:.2f} ({std_value:.2f})"
            table.append(record)

    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)

    widths = {"arm": 20}
    print(f"{'arm':20}" + "".join(f"{name:>22}" for name, _ in METRICS))
    print("-" * (20 + 22 * len(METRICS)))
    for record in table:
        cells = "".join((record[name] or "pending").rjust(22) for name, _ in METRICS)
        print(f"{record['arm']:20}{cells}")
    print(f"\nmean (sample std) over {table[0]['seed_count']} map seeds: 4, 29, 99")
    partial = [r["arm"] for r in table if r["hitter_seeds"] < r["seed_count"]]
    if partial:
        print(f"hitter column incomplete for: {', '.join(partial)} (awaiting replay of pre-metric runs)")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
