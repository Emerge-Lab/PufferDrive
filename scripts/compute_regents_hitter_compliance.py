"""Replay saved ReGentS scenarios and add hitter compliance to the collision summary.

Generalizes scripts/reprocess_seed_29_hitter_compliance.py to every <generation>/<cohort>
run under the results root. Runs already present in hitter_compliance_scenarios.csv are
reused instead of replayed unless --recompute-all is given.
"""

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCENARIOS_PER_GENERATION = 10000
METRIC_FIELDS = (
    "hitter_compliance_valid",
    "hitter_compliance_compliant",
    "hitter_compliance_red_light_violation",
    "hitter_compliance_wrong_way_violation",
    "hitter_compliance_solid_line_violation",
    "hitter_compliance_speed_limit_violation",
)
SCENARIO_FIELDS = ("cohort", "generation", "scenario_index", "scenario_id", "target_collision", *METRIC_FIELDS)
MISMATCH_FIELDS = ("cohort", "generation", "scenario_index", "scenario_id", "metrics_collision", "replay_collision")
ADDED_SUMMARY_FIELDS = (
    "hitter_compliance_valid_count",
    "hitter_compliant_count",
    "hitter_compliance_pct_of_target_collisions",
    "hitter_compliance_pct_of_valid_collisions",
    "hitter_compliant_pct_of_scenarios",
    "collision_mismatch_count",
)


def replay_collision(task):
    cohort, generation, scenario_index, scenario_id, artifact_path, expected_collision, allow_mismatch = task
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.regents.artifacts import load_generation_artifact
    from pufferlib.ocean.regents.rollout import capture_plan_pair

    metadata, arrays = load_generation_artifact(artifact_path)
    if metadata["scenario_id"] != scenario_id:
        raise ValueError(f"{cohort}/{generation} scenario {scenario_index} artifact ID does not match metrics")
    source = metadata["source_configuration"]
    env_config = dict(source["env"])
    repository_root = Path(__file__).resolve().parents[1]
    env_config["map_dir"] = str(repository_root / env_config["map_dir"])
    env_config["collision_behavior"] = "stop"
    env_config["offroad_behavior"] = "stop"
    scenario_seed = source["seed"] + scenario_index
    drive = Drive(
        **env_config,
        eval_map_indices=[scenario_index],
        eval_scenario_seeds=[scenario_seed],
        seed=scenario_seed,
    )
    try:
        _, adversarial = capture_plan_pair(
            drive,
            arrays["initial_actions"],
            arrays["optimized_actions"],
            arrays["optimized_action_mask"],
            transition_count=arrays["optimized_actions"].shape[1],
            expected_scenario_id=scenario_id,
            agent_count=arrays["agent_id"].size,
            seed=scenario_seed,
        )
    finally:
        drive.close()
    episode_log = adversarial.episode_log
    if episode_log is None:
        raise ValueError(f"{cohort}/{generation} scenario {scenario_index} replay has no episode log")
    actual_collision = float(episode_log["sdc_target_collision_rate"])
    if actual_collision != expected_collision:
        if not allow_mismatch:
            raise ValueError(
                f"{cohort}/{generation} scenario {scenario_index} target collision changed: "
                f"{expected_collision} to {actual_collision}"
            )
        return {
            "cohort": cohort,
            "generation": generation,
            "scenario_index": scenario_index,
            "scenario_id": scenario_id,
            "metrics_collision": expected_collision,
            "replay_collision": actual_collision,
            "collision_mismatch": True,
        }
    if int(episode_log["hitter_compliance_valid"]) != int(expected_collision):
        raise ValueError(
            f"{cohort}/{generation} scenario {scenario_index} hitter compliance validity disagrees with collision"
        )
    return {
        "cohort": cohort,
        "generation": generation,
        "scenario_index": scenario_index,
        "scenario_id": scenario_id,
        "target_collision": int(expected_collision),
        "collision_mismatch": False,
        **{field: int(episode_log[field]) for field in METRIC_FIELDS},
    }


def read_existing(path, fields):
    if not path.exists():
        return {}
    grouped = {}
    with path.open(newline="") as existing_file:
        for row in csv.DictReader(existing_file):
            grouped.setdefault((row["cohort"], row["generation"]), []).append({f: row[f] for f in fields})
    return grouped


def scan_runs(results_root, only_cohort, only_generation):
    runs = {}
    for metrics_path in sorted(results_root.glob("*/*/generation_metrics.csv")):
        generation = metrics_path.parents[1].name
        cohort = metrics_path.parent.name
        if only_cohort and cohort != only_cohort:
            continue
        if only_generation and generation != only_generation:
            continue
        with metrics_path.open(newline="") as metrics_file:
            rows = list(csv.DictReader(metrics_file))
        if len(rows) != SCENARIOS_PER_GENERATION:
            raise ValueError(f"{metrics_path} has {len(rows)} scenarios, expected {SCENARIOS_PER_GENERATION}")
        target_collision_count = 0
        scenarios = []
        for expected_index, row in enumerate(rows):
            scenario_index = int(row["scenario_index"])
            if scenario_index != expected_index:
                raise ValueError(f"{metrics_path} scenario indices are not contiguous")
            target_collision = float(row["eval_sdc_target_collision_rate"])
            if target_collision not in (0.0, 1.0):
                raise ValueError(f"{metrics_path} scenario {scenario_index} has an invalid target collision value")
            target_collision_count += int(target_collision)
            scenarios.append((scenario_index, row["scenario_id"], target_collision))
        runs[(cohort, generation)] = {
            "run_dir": metrics_path.parent,
            "target_collision_count": target_collision_count,
            "scenarios": scenarios,
        }
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="experiments/regents_res")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Replay only this many scenarios for a smoke check")
    parser.add_argument("--cohort", default="", help="Restrict to one cohort directory name")
    parser.add_argument("--generation", default="", help="Restrict to one generation directory name")
    parser.add_argument(
        "--allow-collision-mismatch",
        action="store_true",
        help="Record scenarios whose replay collision disagrees with the generation metrics and exclude them",
    )
    parser.add_argument("--recompute-all", action="store_true", help="Replay runs even if results already exist")
    args = parser.parse_args()
    if args.workers <= 0 or args.limit < 0:
        parser.error("workers must be positive and limit must be nonnegative")

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        parser.error(f"{results_root} is not a directory")
    runs = scan_runs(results_root, args.cohort, args.generation)
    if not runs:
        parser.error(f"no runs found under {results_root}")

    results_path = results_root / "hitter_compliance_scenarios.csv"
    mismatch_path = results_root / "collision_mismatches.csv"
    kept_scenarios = {} if args.recompute_all else read_existing(results_path, SCENARIO_FIELDS)
    kept_mismatches = {} if args.recompute_all else read_existing(mismatch_path, MISMATCH_FIELDS)

    replay_keys = []
    for key in runs:
        covered = len(kept_scenarios.get(key, [])) + len(kept_mismatches.get(key, []))
        if covered == SCENARIOS_PER_GENERATION:
            continue
        if covered:
            kept_scenarios.pop(key, None)
            kept_mismatches.pop(key, None)
        replay_keys.append(key)
    reused_keys = [key for key in runs if key not in replay_keys]
    for key in list(kept_scenarios) + list(kept_mismatches):
        if key not in runs:
            kept_scenarios.pop(key, None)
            kept_mismatches.pop(key, None)

    tasks = []
    for key in replay_keys:
        cohort, generation = key
        for scenario_index, scenario_id, target_collision in runs[key]["scenarios"]:
            artifact_path = runs[key]["run_dir"] / f"npz/scenario_{scenario_index:05d}.npz"
            if not artifact_path.exists():
                raise FileNotFoundError(f"missing artifact {artifact_path}")
            tasks.append(
                (
                    cohort,
                    generation,
                    scenario_index,
                    scenario_id,
                    artifact_path,
                    target_collision,
                    args.allow_collision_mismatch,
                )
            )
    for key in sorted(reused_keys):
        print(f"Reusing existing results for {key[0]}/{key[1]}")
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Replaying {len(tasks)} scenarios across {len(replay_keys)} runs", flush=True)

    new_scenarios = {}
    new_mismatches = {}
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result_index, result in enumerate(executor.map(replay_collision, tasks, chunksize=8), start=1):
                key = (result["cohort"], result["generation"])
                if result["collision_mismatch"]:
                    new_mismatches.setdefault(key, []).append({f: result[f] for f in MISMATCH_FIELDS})
                else:
                    new_scenarios.setdefault(key, []).append({f: result[f] for f in SCENARIO_FIELDS})
                if result_index % 1000 == 0:
                    print(f"Replayed {result_index}/{len(tasks)} scenarios", flush=True)
    if args.limit:
        print(f"Smoke check passed for {len(tasks)} scenarios; output files were not changed")
        return

    scenarios_by_run = {**kept_scenarios, **new_scenarios}
    mismatches_by_run = {**kept_mismatches, **new_mismatches}
    for key in runs:
        covered = len(scenarios_by_run.get(key, [])) + len(mismatches_by_run.get(key, []))
        if covered != SCENARIOS_PER_GENERATION:
            raise ValueError(f"{key[0]}/{key[1]} has {covered} results, expected {SCENARIOS_PER_GENERATION}")

    counts = {}
    for key, rows in scenarios_by_run.items():
        counts[key] = Counter()
        for row in rows:
            counts[key]["valid"] += int(row["hitter_compliance_valid"])
            counts[key]["compliant"] += int(row["hitter_compliance_compliant"])

    ordered_keys = sorted(scenarios_by_run, key=lambda key: (key[0], key[1]))
    with results_path.with_suffix(".csv.tmp").open("w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=SCENARIO_FIELDS, lineterminator="\n")
        writer.writeheader()
        for key in ordered_keys:
            writer.writerows(sorted(scenarios_by_run[key], key=lambda row: int(row["scenario_index"])))
    mismatch_total = sum(len(rows) for rows in mismatches_by_run.values())
    if mismatch_total:
        with mismatch_path.with_suffix(".csv.tmp").open("w", newline="") as mismatch_file:
            writer = csv.DictWriter(mismatch_file, fieldnames=MISMATCH_FIELDS, lineterminator="\n")
            writer.writeheader()
            for key in sorted(mismatches_by_run):
                writer.writerows(sorted(mismatches_by_run[key], key=lambda row: int(row["scenario_index"])))
        print(f"WARNING: excluded {mismatch_total} scenarios whose replay collision disagrees with the metrics")
        for key in sorted(mismatches_by_run):
            print(f"  {key[0]}/{key[1]}: {len(mismatches_by_run[key])}")

    summary_path = results_root / "collision_summary.csv"
    with summary_path.open(newline="") as summary_file:
        summary_reader = csv.DictReader(summary_file)
        fieldnames = list(summary_reader.fieldnames)
        summary_rows = list(summary_reader)
    for row in summary_rows:
        key = (row["cohort"], row["generation"])
        if key not in counts:
            continue
        if int(row["target_collision_count"]) != runs[key]["target_collision_count"]:
            raise ValueError(f"{key} target collision count changed")
        valid_count = counts[key]["valid"]
        compliant_count = counts[key]["compliant"]
        target_count = runs[key]["target_collision_count"]
        row["hitter_compliance_valid_count"] = valid_count
        row["hitter_compliant_count"] = compliant_count
        row["hitter_compliance_pct_of_target_collisions"] = (
            f"{100 * compliant_count / target_count:.2f}%" if target_count else ""
        )
        row["hitter_compliance_pct_of_valid_collisions"] = (
            f"{100 * compliant_count / valid_count:.2f}%" if valid_count else ""
        )
        row["hitter_compliant_pct_of_scenarios"] = f"{100 * compliant_count / SCENARIOS_PER_GENERATION:.2f}%"
        row["collision_mismatch_count"] = len(mismatches_by_run.get(key, []))
    fieldnames.extend(field for field in ADDED_SUMMARY_FIELDS if field not in fieldnames)
    with summary_path.with_suffix(".csv.tmp").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames, lineterminator="\n", restval="")
        writer.writeheader()
        writer.writerows(summary_rows)

    results_path.with_suffix(".csv.tmp").replace(results_path)
    if mismatch_total:
        mismatch_path.with_suffix(".csv.tmp").replace(mismatch_path)
    summary_path.with_suffix(".csv.tmp").replace(summary_path)
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()
