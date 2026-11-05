"""Write one finished evaluation's headline metric into the summary of the wandb run that trained the checkpoint."""

import argparse
import csv
import fcntl
import glob
import json
import math
import numbers
import os
import re
import time

import yaml


EVAL_TYPE_LONGEST6_V1 = "longest6_v1"
EVAL_TYPE_LONGEST6_V2 = "longest6_v2"
EVAL_TYPE_NUPLAN_REACTIVE = "nuplan_reactive"
EVAL_TYPE_NUPLAN_NONREACTIVE = "nuplan_nonreactive"
EVAL_TYPE_SELFPLAY = "selfplay"
EVAL_TYPE_ALPASIM = "alpasim"

HEADLINE_METRIC_KEYS = {
    EVAL_TYPE_LONGEST6_V1: "final_eval/longest6_v1_DS",
    EVAL_TYPE_LONGEST6_V2: "final_eval/longest6_v2_DS",
    EVAL_TYPE_NUPLAN_REACTIVE: "final_eval/nuplan_reactive_CLS",
    EVAL_TYPE_NUPLAN_NONREACTIVE: "final_eval/nuplan_nonreactive_CLS",
    EVAL_TYPE_SELFPLAY: "final_eval/selfplay_km_per_infraction",
    EVAL_TYPE_ALPASIM: "final_eval/alpasim_score",
}
SOURCE_KEY_SUFFIX = "_source"

SELFPLAY_BENCHMARK_NAME = "carla"
SELFPLAY_SUMMARY_FILE_NAME = "evaluation_summary.json"
METERS_PER_KM = 1000.0

LONGEST6_DRIVING_SCORE_ROW = "Avg. driving score"
LONGEST6_DRIVING_SCORE_TOLERANCE = 0.01
NUPLAN_SCORE_COLUMN = "CLS"
NUPLAN_SCENARIOS_FILE_NAME = "scenarios.csv"
ALPASIM_SCORE_LINE_PATTERN = re.compile(r"official_score (\S+)\s+scenes (\d+)\s+failed (\d+)")

RUN_CONFIG_FILE_NAME = "config.yaml"
RUN_IDENTITY_KEYS = ("run_name", "wandb_project")
REPORT_LOCK_FILE_NAME = ".eval_wandb_report.lock"
WANDB_RUNNING_STATE = "running"
SUMMARY_FLOAT_REL_TOLERANCE = 1e-9
SUMMARY_VERIFY_ATTEMPT_COUNT = 3
SUMMARY_VERIFY_BACKOFF_SECONDS = 2.0


class EvalReportError(Exception):
    pass


def read_longest6_driving_score(results_csv_path, expected_route_run_count):
    with open(results_csv_path, newline="") as results_file:
        driving_scores = [
            row[1] for row in csv.reader(results_file) if len(row) >= 2 and row[0] == LONGEST6_DRIVING_SCORE_ROW
        ]
    if len(driving_scores) != 1:
        raise EvalReportError(f"{results_csv_path} must hold exactly one '{LONGEST6_DRIVING_SCORE_ROW}' row")
    driving_score = float(driving_scores[0])

    route_scores = []
    for record_path in glob.glob(os.path.join(os.path.dirname(results_csv_path), "**", "*.json"), recursive=True):
        with open(record_path) as record_file:
            route_records = json.load(record_file)["_checkpoint"]["records"]
        route_scores.extend(float(record["scores"]["score_composed"]) for record in route_records)
    if len(route_scores) != expected_route_run_count:
        raise EvalReportError(
            f"incomplete longest6 eval: {len(route_scores)} of {expected_route_run_count} route runs have a record "
            f"next to {results_csv_path}"
        )
    mean_route_score = sum(route_scores) / len(route_scores)
    if abs(driving_score - mean_route_score) > LONGEST6_DRIVING_SCORE_TOLERANCE:
        raise EvalReportError(
            f"{results_csv_path} is stale: its driving score {driving_score} is not the mean {mean_route_score} "
            "of the route records next to it"
        )
    return driving_score


def read_nuplan_closed_loop_score(overview_csv_path, expected_scenario_count):
    with open(overview_csv_path, newline="") as overview_file:
        overview_rows = list(csv.DictReader(overview_file))
    if len(overview_rows) != 1 or NUPLAN_SCORE_COLUMN not in overview_rows[0]:
        raise EvalReportError(f"{overview_csv_path} must hold exactly one row with a {NUPLAN_SCORE_COLUMN} column")

    scenarios_csv_path = os.path.join(os.path.dirname(overview_csv_path), NUPLAN_SCENARIOS_FILE_NAME)
    with open(scenarios_csv_path, newline="") as scenarios_file:
        scenario_count = sum(1 for _ in csv.DictReader(scenarios_file))
    if scenario_count != expected_scenario_count:
        raise EvalReportError(
            f"incomplete nuPlan eval: {scenario_count} of {expected_scenario_count} scenarios scored in "
            f"{scenarios_csv_path}"
        )
    return float(overview_rows[0][NUPLAN_SCORE_COLUMN])


def read_selfplay_km_per_infraction(summary_json_path):
    with open(summary_json_path) as summary_file:
        summary = json.load(summary_file)
    if summary["num_episodes"] != summary["num_scenarios"]:
        raise EvalReportError(
            f"incomplete self-play eval: {summary['num_episodes']} of {summary['num_scenarios']} scenarios finished "
            f"in {summary_json_path}"
        )
    return float(summary["metrics_mean"]["avg_distance_per_infraction"]) / METERS_PER_KM


def read_alpasim_score(results_txt_path, expected_scene_count):
    with open(results_txt_path) as results_file:
        score_line = results_file.readline().strip()
    score_match = ALPASIM_SCORE_LINE_PATTERN.fullmatch(score_line)
    if score_match is None:
        raise EvalReportError(f"{results_txt_path} holds no official score: '{score_line}'")
    scored_scene_count = int(score_match.group(2))
    failed_scene_count = int(score_match.group(3))
    if scored_scene_count != expected_scene_count or failed_scene_count != 0:
        raise EvalReportError(
            f"incomplete AlpaSim eval: {scored_scene_count} of {expected_scene_count} scenes scored, "
            f"{failed_scene_count} failed in {results_txt_path}"
        )
    return float(score_match.group(1))


def read_headline_value(eval_type, result_path, expected_episode_count):
    """Read an eval type's headline number from its result file; refuses incomplete evaluations."""
    if not os.path.isfile(result_path):
        raise EvalReportError(f"missing result file {result_path}")
    if eval_type != EVAL_TYPE_SELFPLAY and expected_episode_count is None:
        raise EvalReportError(f"{eval_type} needs the benchmark's episode count to tell a complete eval from a subset")

    if eval_type == EVAL_TYPE_SELFPLAY:
        headline_value = read_selfplay_km_per_infraction(result_path)
    elif eval_type == EVAL_TYPE_ALPASIM:
        headline_value = read_alpasim_score(result_path, expected_episode_count)
    elif eval_type in (EVAL_TYPE_NUPLAN_REACTIVE, EVAL_TYPE_NUPLAN_NONREACTIVE):
        headline_value = read_nuplan_closed_loop_score(result_path, expected_episode_count)
    elif eval_type in (EVAL_TYPE_LONGEST6_V1, EVAL_TYPE_LONGEST6_V2):
        headline_value = read_longest6_driving_score(result_path, expected_episode_count)
    else:
        raise EvalReportError(f"unknown eval type '{eval_type}'; expected one of {sorted(HEADLINE_METRIC_KEYS)}")

    if not math.isfinite(headline_value):
        raise EvalReportError(f"{result_path} holds a non-finite {eval_type} headline value: {headline_value}")
    return headline_value


def load_run_identity(run_config_path):
    """Resolve the wandb run a checkpoint belongs to from its training config."""
    if not os.path.isfile(run_config_path):
        raise EvalReportError(f"missing run config {run_config_path}")
    with open(run_config_path) as run_config_file:
        run_config = yaml.safe_load(run_config_file)
    if not isinstance(run_config, dict):
        raise EvalReportError(f"run config {run_config_path} must be a mapping")

    run_identity = {}
    for key in RUN_IDENTITY_KEYS:
        value = run_config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise EvalReportError(
                f"run config {run_config_path} is missing a usable {key}; cannot attach eval results to the "
                "training run"
            )
        run_identity[key] = value
    return run_identity


def find_summary_mismatches(expected_summary, stored_summary):
    """Keys whose stored value is not the expected one; numbers may differ by the server's float formatting."""
    mismatched_keys = []
    for key in sorted(expected_summary.keys() | stored_summary.keys()):
        expected_value = expected_summary.get(key)
        stored_value = stored_summary.get(key)
        values_are_numbers = isinstance(expected_value, numbers.Real) and isinstance(stored_value, numbers.Real)
        if values_are_numbers and math.isclose(expected_value, stored_value, rel_tol=SUMMARY_FLOAT_REL_TOLERANCE):
            continue
        if json.dumps(expected_value, sort_keys=True) != json.dumps(stored_value, sort_keys=True):
            mismatched_keys.append(key)
    return mismatched_keys


def write_run_summary(run_identity, reported_values):
    """Set summary values of a finished wandb run and verify that nothing else in its summary changed."""
    import wandb

    run_path = f"{run_identity['wandb_project']}/{run_identity['run_name']}"
    run = wandb.Api().run(run_path)
    if run.state == WANDB_RUNNING_STATE:
        raise EvalReportError(
            f"wandb run {run_path} is still running and its own logger would overwrite the summary; report again "
            "once training has finished"
        )

    expected_summary = {**run.summary_metrics, **reported_values}
    run.summary.update(reported_values)

    mismatched_keys = []
    for attempt_idx in range(SUMMARY_VERIFY_ATTEMPT_COUNT):
        time.sleep(attempt_idx * SUMMARY_VERIFY_BACKOFF_SECONDS)
        mismatched_keys = find_summary_mismatches(expected_summary, wandb.Api().run(run_path).summary_metrics)
        if not mismatched_keys:
            return
    raise EvalReportError(
        f"wandb run {run_path}: summary keys differ from the intended update, check them: {mismatched_keys}"
    )


def report_headline_metric(eval_type, result_path, run_dir, expected_episode_count=None):
    """Report one evaluation's headline metric; a repeated report replaces the run's value for that eval type."""
    headline_value = read_headline_value(eval_type, result_path, expected_episode_count)
    run_identity = load_run_identity(os.path.join(run_dir, RUN_CONFIG_FILE_NAME))
    metric_key = HEADLINE_METRIC_KEYS[eval_type]
    result_source = os.path.relpath(os.path.abspath(result_path), os.path.abspath(run_dir))
    reported_values = {metric_key: headline_value, metric_key + SOURCE_KEY_SUFFIX: result_source}

    # the wandb API replaces the whole summary: evals of one run finishing together must not interleave
    with open(os.path.join(run_dir, REPORT_LOCK_FILE_NAME), "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        write_run_summary(run_identity, reported_values)

    print(
        f"wandb run {run_identity['wandb_project']}/{run_identity['run_name']}: {metric_key} = {headline_value} "
        f"({result_source})"
    )
    return reported_values


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_type", choices=sorted(HEADLINE_METRIC_KEYS))
    parser.add_argument(
        "result_path",
        help="longest6: aggregate/results.csv; nuPlan: overview.csv; selfplay: evaluation_summary.json; "
        "alpasim: results.txt",
    )
    parser.add_argument("--run-dir", required=True, help="Training run folder holding config.yaml")
    parser.add_argument(
        "--expected-episode-count",
        type=int,
        help="Route runs, scenarios or scenes of the complete benchmark; selfplay reads it from its summary",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    try:
        report_headline_metric(
            arguments.eval_type, arguments.result_path, arguments.run_dir, arguments.expected_episode_count
        )
    except EvalReportError as error:
        raise SystemExit(f"eval_wandb_report: {error}")


if __name__ == "__main__":
    main()
