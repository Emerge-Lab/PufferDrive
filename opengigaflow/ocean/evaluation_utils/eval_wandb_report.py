"""Log one finished evaluation's headline metric to the wandb run that trained the checkpoint."""

import argparse
import csv
import fcntl
import glob
import json
import math
import numbers
import os
import re
import tempfile
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
ALPASIM_AT_FAULT_METRIC_KEY = "final_eval/alpasim_km_per_at_fault_incident"
SOURCE_KEY_SUFFIX = "_source"

SELFPLAY_BENCHMARK_NAME = "carla"
SELFPLAY_SUMMARY_FILE_NAME = "evaluation_summary.json"
METERS_PER_KM = 1000.0

LONGEST6_DRIVING_SCORE_ROW = "Avg. driving score"
LONGEST6_DRIVING_SCORE_TOLERANCE = 0.01
NUPLAN_SCORE_COLUMN = "CLS"
NUPLAN_SCENARIOS_FILE_NAME = "scenarios.csv"
LONGEST6_ROUTES_DIR_NAME = "routes"
LONGEST6_RECORD_SUFFIX = ".result.json"
ALPASIM_SCORE_LINE_PATTERN = re.compile(r"official_score (\S+)\s+scenes (\d+)\s+failed (\d+)")
ALPASIM_AT_FAULT_ROW_NAME = "avg_dist_between_incidents_at_fault"
ALPASIM_AT_FAULT_LINE_PATTERN = re.compile(ALPASIM_AT_FAULT_ROW_NAME + r"\s+(?:(\S+) km|n/a)\s.*")

RUN_CONFIG_FILE_NAME = "config.yaml"
RUN_IDENTITY_KEYS = ("run_name", "wandb_project")
REPORT_LOCK_FILE_NAME = ".eval_wandb_report.lock"
WANDB_RUNNING_STATE = "running"
WANDB_RESUME_EXISTING_RUN = "must"
WANDB_BOOKKEEPING_KEY_PREFIX = "_"
STORED_FLOAT_REL_TOLERANCE = 1e-9
REPORT_VERIFY_ATTEMPT_COUNT = 3
REPORT_VERIFY_BACKOFF_SECONDS = 2.0


class EvalReportError(Exception):
    pass


def longest6_route_record_paths(results_csv_path):
    """The route records results.csv summarizes: routes/*.result.json of a run tidy_carla_run.py flattened (results.csv
    at its top), else every json next to a fresh run's aggregate/results.csv (the parser's copies of the records)."""
    results_dir = os.path.dirname(results_csv_path)
    flat_records = glob.glob(os.path.join(results_dir, LONGEST6_ROUTES_DIR_NAME, f"*{LONGEST6_RECORD_SUFFIX}"))
    return flat_records or glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)


def read_longest6_driving_score(results_csv_path, expected_route_run_count):
    with open(results_csv_path, newline="") as results_file:
        driving_scores = [
            row[1] for row in csv.reader(results_file) if len(row) >= 2 and row[0] == LONGEST6_DRIVING_SCORE_ROW
        ]
    if len(driving_scores) != 1:
        raise EvalReportError(f"{results_csv_path} must hold exactly one '{LONGEST6_DRIVING_SCORE_ROW}' row")
    driving_score = float(driving_scores[0])

    route_scores = []
    for record_path in longest6_route_record_paths(results_csv_path):
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


def read_alpasim_at_fault_metric(results_txt_path):
    """AlpaSim's km per at-fault incident (offroad, front or lateral collision); a run without one has no distance."""
    with open(results_txt_path) as results_file:
        line_matches = [ALPASIM_AT_FAULT_LINE_PATTERN.fullmatch(line.strip()) for line in results_file]
    at_fault_matches = [line_match for line_match in line_matches if line_match is not None]
    if len(at_fault_matches) != 1:
        raise EvalReportError(f"{results_txt_path} must hold exactly one '{ALPASIM_AT_FAULT_ROW_NAME}' row")
    km_text = at_fault_matches[0].group(1)
    if km_text is None:
        print(f"{results_txt_path}: no at-fault incident, {ALPASIM_AT_FAULT_METRIC_KEY} is not reported")
        return {}
    km_per_at_fault_incident = float(km_text)
    if not math.isfinite(km_per_at_fault_incident) or km_per_at_fault_incident <= 0.0:
        raise EvalReportError(f"{results_txt_path} holds an unusable '{ALPASIM_AT_FAULT_ROW_NAME}' value: {km_text}")
    return {ALPASIM_AT_FAULT_METRIC_KEY: km_per_at_fault_incident}


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


def stored_value_matches(expected_value, stored_value):
    """Whether wandb holds the expected value; numbers may differ by the server's float formatting."""
    values_are_numbers = isinstance(expected_value, numbers.Real) and isinstance(stored_value, numbers.Real)
    if values_are_numbers and math.isclose(expected_value, stored_value, rel_tol=STORED_FLOAT_REL_TOLERANCE):
        return True
    return json.dumps(expected_value, sort_keys=True) == json.dumps(stored_value, sort_keys=True)


def find_summary_mismatches(expected_summary, stored_summary):
    """Keys whose stored value is not the expected one; wandb's own bookkeeping keys advance with every session."""
    mismatched_keys = []
    for key in sorted(expected_summary.keys() | stored_summary.keys()):
        if key.startswith(WANDB_BOOKKEEPING_KEY_PREFIX):
            continue
        if not stored_value_matches(expected_summary.get(key), stored_summary.get(key)):
            mismatched_keys.append(key)
    return mismatched_keys


def log_run_metrics(run_identity, metric_values, result_source):
    """Append the metrics to a finished wandb run as one history row (charts plot history, not the summary); verify."""
    import wandb

    run_path = f"{run_identity['wandb_project']}/{run_identity['run_name']}"
    api = wandb.Api()
    trained_run = api.run(run_path)
    if trained_run.state == WANDB_RUNNING_STATE:
        raise EvalReportError(
            f"wandb run {run_path} is still running and a second session would collide with its logger; report "
            "again once training has finished"
        )
    source_values = {metric_key + SOURCE_KEY_SUFFIX: result_source for metric_key in metric_values}
    expected_summary = {**trained_run.summary_metrics, **metric_values, **source_values}
    last_step_before_report = trained_run.history_keys["lastStep"]

    with tempfile.TemporaryDirectory() as session_dir:
        session = wandb.init(
            entity=trained_run.entity,
            project=run_identity["wandb_project"],
            id=run_identity["run_name"],
            resume=WANDB_RESUME_EXISTING_RUN,
            dir=session_dir,
            settings=wandb.Settings(console="off", x_disable_meta=True, x_disable_stats=True),
        )
        try:
            session.log(metric_values)
            session.summary.update(source_values)
        finally:
            session.finish()

    mismatched_keys = []
    for attempt_idx in range(REPORT_VERIFY_ATTEMPT_COUNT):
        time.sleep(attempt_idx * REPORT_VERIFY_BACKOFF_SECONDS)
        # the api object caches runs; a second Api() would repeat the service handshake, which can time out
        api.flush()
        stored_run = api.run(run_path)
        mismatched_keys = find_summary_mismatches(expected_summary, stored_run.summary_metrics)
        # the run's key index, not a history query: sampled history trails a finished session by seconds
        stored_history_keys = stored_run.history_keys
        history_row_stored = stored_history_keys["lastStep"] > last_step_before_report and all(
            metric_key in stored_history_keys["keys"] for metric_key in metric_values
        )
        if not mismatched_keys and history_row_stored:
            return
    if mismatched_keys:
        raise EvalReportError(
            f"wandb run {run_path}: summary keys differ from the intended report, check them: {mismatched_keys}"
        )
    raise EvalReportError(
        f"wandb run {run_path}: {sorted(metric_values)} did not reach the run's history; the charts will not show it"
    )


def report_headline_metric(eval_type, result_path, run_dir, expected_episode_count=None):
    """Report one evaluation's headline metric (AlpaSim: and its km per at-fault incident); a repeated report
    becomes the run's value for that eval type."""
    headline_value = read_headline_value(eval_type, result_path, expected_episode_count)
    metric_values = {HEADLINE_METRIC_KEYS[eval_type]: headline_value}
    if eval_type == EVAL_TYPE_ALPASIM:
        metric_values.update(read_alpasim_at_fault_metric(result_path))
    run_identity = load_run_identity(os.path.join(run_dir, RUN_CONFIG_FILE_NAME))
    result_source = os.path.relpath(os.path.abspath(result_path), os.path.abspath(run_dir))

    # a session appends at the history offset and summary it read when it opened: evals of one run must not overlap
    with open(os.path.join(run_dir, REPORT_LOCK_FILE_NAME), "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        log_run_metrics(run_identity, metric_values, result_source)

    reported_text = ", ".join(f"{metric_key} = {metric_value}" for metric_key, metric_value in metric_values.items())
    print(f"wandb run {run_identity['wandb_project']}/{run_identity['run_name']}: {reported_text} ({result_source})")
    return {**metric_values, **{metric_key + SOURCE_KEY_SUFFIX: result_source for metric_key in metric_values}}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_type", choices=sorted(HEADLINE_METRIC_KEYS))
    parser.add_argument(
        "result_path",
        help="longest6: results.csv (aggregate/results.csv before the tidy); nuPlan: overview.csv; selfplay: evaluation_summary.json; "
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
