"""Replay saved seed 29 collisions and add hitter compliance to their summary."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


GENERATIONS = ("regents_corridor_idm", "regents_idm", "regents_pdm")
SCENARIOS_PER_GENERATION = 10000
METRIC_FIELDS = (
    "hitter_compliance_valid",
    "hitter_compliance_compliant",
    "hitter_compliance_red_light_violation",
    "hitter_compliance_wrong_way_violation",
    "hitter_compliance_solid_line_violation",
    "hitter_compliance_speed_limit_violation",
)
SCENARIO_FIELDS = ("generation", "scenario_index", "scenario_id", *METRIC_FIELDS)


def replay_collision(task):
    generation, scenario_index, scenario_id, artifact_path, expected_collision = task
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.regents.artifacts import load_generation_artifact
    from pufferlib.ocean.regents.rollout import capture_plan_pair

    metadata, arrays = load_generation_artifact(artifact_path)
    if metadata["scenario_id"] != scenario_id:
        raise ValueError(f"Scenario {scenario_index} artifact ID does not match generation metrics")
    source = metadata["source_configuration"]
    env_config = dict(source["env"])
    repository_root = Path(__file__).resolve().parents[1]
    env_config["map_dir"] = str(repository_root / env_config["map_dir"])
    # Generation metrics came from the verification replay, which stops on infractions.
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
        raise ValueError(f"Scenario {scenario_index} replay has no episode log")
    actual_collision = float(episode_log["sdc_target_collision_rate"])
    if actual_collision != expected_collision:
        raise ValueError(
            f"Scenario {scenario_index} target collision changed: {expected_collision} to {actual_collision}"
        )
    return {
        "generation": generation,
        "scenario_index": scenario_index,
        "scenario_id": scenario_id,
        **{field: int(episode_log[field]) for field in METRIC_FIELDS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Replay only this many collisions for a smoke check")
    args = parser.parse_args()
    if args.workers <= 0 or args.limit < 0:
        parser.error("workers must be positive and limit must be nonnegative")

    experiment_root = Path(__file__).resolve().parents[1] / "experiments/regents"
    tasks = []
    target_collision_counts = {}
    for generation in GENERATIONS:
        run_dir = experiment_root / generation / "seed_29"
        with (run_dir / "generation_metrics.csv").open(newline="") as metrics_file:
            rows = list(csv.DictReader(metrics_file))
        if len(rows) != SCENARIOS_PER_GENERATION:
            raise ValueError(f"{generation} has {len(rows)} scenarios, expected {SCENARIOS_PER_GENERATION}")
        target_collision_counts[generation] = 0
        for expected_index, row in enumerate(rows):
            scenario_index = int(row["scenario_index"])
            if scenario_index != expected_index:
                raise ValueError(f"{generation} scenario indices are not contiguous")
            target_collision = float(row["eval_sdc_target_collision_rate"])
            if target_collision not in (0.0, 1.0):
                raise ValueError(f"{generation} scenario {scenario_index} has an invalid target collision value")
            if target_collision == 0.0:
                continue
            target_collision_counts[generation] += 1
            tasks.append(
                (
                    generation,
                    scenario_index,
                    row["scenario_id"],
                    run_dir / f"npz/scenario_{scenario_index:05d}.npz",
                    target_collision,
                )
            )
    if args.limit:
        tasks = tasks[: args.limit]

    results_path = experiment_root / "seed_29_hitter_compliance_scenarios.csv"
    results_temp_path = results_path.with_suffix(".csv.tmp")
    counts = {generation: {"valid": 0, "compliant": 0} for generation in GENERATIONS}
    with results_temp_path.open("w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=SCENARIO_FIELDS, lineterminator="\n")
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result_index, result in enumerate(executor.map(replay_collision, tasks), start=1):
                writer.writerow(result)
                counts[result["generation"]]["valid"] += result["hitter_compliance_valid"]
                counts[result["generation"]]["compliant"] += result["hitter_compliance_compliant"]
                if result_index % 1000 == 0:
                    results_file.flush()
                    print(f"Replayed {result_index}/{len(tasks)} collisions", flush=True)
    if args.limit:
        print(f"Smoke check passed for {len(tasks)} collisions; summary was not changed")
        results_temp_path.unlink()
        return

    summary_path = experiment_root / "seed_29_collision_summary.csv"
    with summary_path.open(newline="") as summary_file:
        summary_reader = csv.DictReader(summary_file)
        fieldnames = list(summary_reader.fieldnames)
        summary_rows = list(summary_reader)
    if [row["generation"] for row in summary_rows] != list(GENERATIONS):
        raise ValueError("Summary generations do not match the replayed runs")
    for row in summary_rows:
        generation = row["generation"]
        if int(row["target_collision_count"]) != target_collision_counts[generation]:
            raise ValueError(f"{generation} target collision count changed")
        valid_count = counts[generation]["valid"]
        compliant_count = counts[generation]["compliant"]
        row["hitter_compliance_valid_count"] = valid_count
        row["hitter_compliant_count"] = compliant_count
        row["hitter_compliance_pct_of_target_collisions"] = (
            f"{100 * compliant_count / target_collision_counts[generation]:.2f}%"
        )
        row["hitter_compliance_pct_of_valid_collisions"] = (
            f"{100 * compliant_count / valid_count:.2f}%" if valid_count else ""
        )
    added_fieldnames = (
        "hitter_compliance_valid_count",
        "hitter_compliant_count",
        "hitter_compliance_pct_of_target_collisions",
        "hitter_compliance_pct_of_valid_collisions",
    )
    fieldnames.extend(field for field in added_fieldnames if field not in fieldnames)
    summary_temp_path = summary_path.with_suffix(".csv.tmp")
    with summary_temp_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    results_temp_path.replace(results_path)
    summary_temp_path.replace(summary_path)
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()
