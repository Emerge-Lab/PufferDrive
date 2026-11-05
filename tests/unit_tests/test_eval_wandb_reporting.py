"""Unit tests for writing an evaluation's headline metric into the summary of the training wandb run."""

import csv
import json
import os
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from pufferlib import pufferl
from pufferlib.ocean.evaluation_utils import eval_wandb_report
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark


RUN_IDENTITY = {"run_name": "k_exp_0002_1000", "wandb_project": "nightly-multi-long"}
RUN_PATH = "nightly-multi-long/k_exp_0002_1000"
TRAINING_SUMMARY = {"environment/collision_rate": 0.0123, "_step": 1000190509056, "early_stop": False}
ROUTE_COUNT = 3
SEED_COUNT = 2
SCENARIO_COUNT = 5
SCENE_COUNT = 4
STUB_WANDB_MODULE = """
import json
import os
import sys


class _Summary:
    def update(self, values):
        with open(os.environ["STUB_WANDB_SUMMARY_PATH"]) as summary_file:
            stored_summary = json.load(summary_file)
        stored_summary.update(values)
        with open(os.environ["STUB_WANDB_SUMMARY_PATH"], "w") as summary_file:
            json.dump(stored_summary, summary_file)


class _Run:
    state = "finished"
    summary = _Summary()

    @property
    def summary_metrics(self):
        with open(os.environ["STUB_WANDB_SUMMARY_PATH"]) as summary_file:
            return json.load(summary_file)


class Api:
    def run(self, run_path):
        assert "pufferlib" not in sys.modules
        return _Run()
"""


class _StubSummary:
    def __init__(self, stub):
        self._stub = stub

    def update(self, values):
        self._stub.updates.append(dict(values))
        self._stub.store_update(values)


class _StubRun:
    def __init__(self, stub):
        self.state = stub.run_state
        self.summary_metrics = dict(stub.stored_summary)
        self.summary = _StubSummary(stub)


class _StubApi:
    def __init__(self, stub):
        self._stub = stub

    def run(self, run_path):
        self._stub.requested_run_paths.append(run_path)
        return _StubRun(self._stub)


@pytest.fixture
def stub_wandb(monkeypatch):
    """A wandb module that only has the public API: an attempt to open a session or log history fails."""
    stub = types.ModuleType("wandb")
    stub.run_state = "finished"
    stub.stored_summary = dict(TRAINING_SUMMARY)
    stub.updates = []
    stub.requested_run_paths = []
    stub.store_update = stub.stored_summary.update
    stub.Api = lambda: _StubApi(stub)
    monkeypatch.setitem(sys.modules, "wandb", stub)
    monkeypatch.setattr(eval_wandb_report, "SUMMARY_VERIFY_BACKOFF_SECONDS", 0.0)
    return stub


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({**RUN_IDENTITY, "wandb_group": "emerge_"}))
    return tmp_path


def _write_longest6_results(run_dir, route_scores, driving_score):
    """results.csv plus one route record per score, in the layout of the longest6 scripts' aggregate folder."""
    aggregate_dir = run_dir / "eval" / "carla_longest6_v1_seeds0-1_20261105_085352_2176" / "aggregate"
    for score_idx, route_score in enumerate(route_scores):
        seed_dir = aggregate_dir / f"s{score_idx % SEED_COUNT}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "route_id": f"RouteScenario_{score_idx}",
            "status": "Completed",
            "scores": {"score_composed": route_score},
        }
        (seed_dir / f"longest_weathers_{score_idx}.json").write_text(json.dumps({"_checkpoint": {"records": [record]}}))
    with open(aggregate_dir / "results.csv", "w", newline="") as results_file:
        writer = csv.writer(results_file)
        writer.writerows([["sep=,"], ["Avg. driving score", driving_score], ["Avg. route completion", 93.3]])
    return aggregate_dir / "results.csv"


def _write_nuplan_results(run_dir, closed_loop_score, scenario_count):
    group_dir = run_dir / "eval" / "nuplan_val14_roadblock_20261105_085444_2173"
    group_dir.mkdir(parents=True)
    (group_dir / "overview.csv").write_text(f"CLS,NAC,DAC\n{closed_loop_score},0.98,0.99\n")
    scenario_rows = "".join(f"token{scenario_idx},log,1.0\n" for scenario_idx in range(scenario_count))
    (group_dir / "scenarios.csv").write_text("scenario,log_name,score\n" + scenario_rows)
    return group_dir / "overview.csv"


def _write_selfplay_summary(run_dir, distance_per_infraction_meters, num_episodes):
    aggregate_dir = run_dir / "eval" / "carla_k_exp_0002_1000" / "20261105-081916"
    aggregate_dir.mkdir(parents=True)
    summary = {
        "num_scenarios": SCENARIO_COUNT,
        "num_episodes": num_episodes,
        "metrics_mean": {"avg_distance_per_infraction": distance_per_infraction_meters, "collision_rate": 2.7e-05},
    }
    (aggregate_dir / "evaluation_summary.json").write_text(json.dumps(summary))
    return aggregate_dir / "evaluation_summary.json"


def _write_alpasim_results(run_dir, score_line):
    out_dir = run_dir / "eval" / "alpasim_gt_time5_r2_final_model_20261102_013615_909"
    out_dir.mkdir(parents=True)
    (out_dir / "results.txt").write_text(score_line + "\ncollision_any   mean    0.0925  n 400\n")
    return out_dir / "results.txt"


def _complete_result(run_dir, eval_type):
    """A complete result of the eval type -> (result path, expected episode count, expected headline value)."""
    if eval_type in (eval_wandb_report.EVAL_TYPE_LONGEST6_V1, eval_wandb_report.EVAL_TYPE_LONGEST6_V2):
        route_scores = [100.0, 60.0, 80.0, 90.0, 70.0, 80.0]
        return _write_longest6_results(run_dir, route_scores, 80.00001), ROUTE_COUNT * SEED_COUNT, 80.00001
    if eval_type in (eval_wandb_report.EVAL_TYPE_NUPLAN_REACTIVE, eval_wandb_report.EVAL_TYPE_NUPLAN_NONREACTIVE):
        return _write_nuplan_results(run_dir, 0.9028013951163436, SCENARIO_COUNT), SCENARIO_COUNT, 0.9028013951163436
    if eval_type == eval_wandb_report.EVAL_TYPE_ALPASIM:
        score_line = f"official_score 0.7832  scenes {SCENE_COUNT}  failed 0"
        return _write_alpasim_results(run_dir, score_line), SCENE_COUNT, 0.7832
    return _write_selfplay_summary(run_dir, 244218962.33697918, SCENARIO_COUNT), None, 244218.96233697918


@pytest.mark.parametrize("eval_type", sorted(eval_wandb_report.HEADLINE_METRIC_KEYS))
def test_each_eval_type_writes_its_headline_value_and_source(stub_wandb, run_dir, eval_type):
    result_path, expected_episode_count, expected_value = _complete_result(run_dir, eval_type)

    eval_wandb_report.report_headline_metric(eval_type, str(result_path), str(run_dir), expected_episode_count)

    metric_key = eval_wandb_report.HEADLINE_METRIC_KEYS[eval_type]
    reported_values = {metric_key: expected_value, f"{metric_key}_source": str(result_path.relative_to(run_dir))}
    assert stub_wandb.requested_run_paths[0] == RUN_PATH
    assert stub_wandb.updates == [reported_values]
    # Training's own summary values stay exactly as they were.
    assert stub_wandb.stored_summary == {**TRAINING_SUMMARY, **reported_values}


def test_headline_keys_are_one_per_eval_type():
    assert eval_wandb_report.HEADLINE_METRIC_KEYS == {
        "longest6_v1": "final_eval/longest6_v1_DS",
        "longest6_v2": "final_eval/longest6_v2_DS",
        "nuplan_reactive": "final_eval/nuplan_reactive_CLS",
        "nuplan_nonreactive": "final_eval/nuplan_nonreactive_CLS",
        "selfplay": "final_eval/selfplay_km_per_infraction",
        "alpasim": "final_eval/alpasim_score",
    }


def test_repeated_report_replaces_the_value_instead_of_adding_one(stub_wandb, run_dir):
    first_result = _write_selfplay_summary(run_dir, 691229300.0, SCENARIO_COUNT)
    eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(first_result), str(run_dir))

    second_result = run_dir / "eval" / "carla_k_exp_0002_1000_redlight" / "evaluation_summary.json"
    second_result.parent.mkdir(parents=True)
    second_result.write_text(first_result.read_text().replace("691229300.0", "200820400.0"))
    eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(second_result), str(run_dir))

    # One value per run and eval type, and the source names the eval the value came from.
    assert stub_wandb.stored_summary == {
        **TRAINING_SUMMARY,
        "final_eval/selfplay_km_per_infraction": 200820.4,
        "final_eval/selfplay_km_per_infraction_source": "eval/carla_k_exp_0002_1000_redlight/evaluation_summary.json",
    }


def _incomplete_longest6(run_dir):
    return _write_longest6_results(run_dir, [100.0, 60.0, 80.0, 90.0, 70.0], 80.0), ROUTE_COUNT * SEED_COUNT


def _incomplete_nuplan(run_dir):
    return _write_nuplan_results(run_dir, 0.9028, SCENARIO_COUNT - 1), SCENARIO_COUNT


def _incomplete_selfplay(run_dir):
    return _write_selfplay_summary(run_dir, 244218962.3, SCENARIO_COUNT - 1), None


def _alpasim_with_failed_scenes(run_dir):
    return _write_alpasim_results(run_dir, f"official_score 0.7832  scenes {SCENE_COUNT}  failed 1"), SCENE_COUNT


def _alpasim_subset(run_dir):
    return _write_alpasim_results(run_dir, "official_score 0.7832  scenes 2  failed 0"), SCENE_COUNT


def _alpasim_without_score(run_dir):
    score_line = "official_score n/a  (incomplete: 398 scenes finished, 2 failed)"
    return _write_alpasim_results(run_dir, score_line), SCENE_COUNT


@pytest.mark.parametrize(
    ("eval_type", "write_result", "error_match"),
    [
        ("longest6_v1", _incomplete_longest6, "5 of 6 route runs"),
        ("nuplan_reactive", _incomplete_nuplan, "4 of 5 scenarios"),
        ("selfplay", _incomplete_selfplay, "4 of 5 scenarios"),
        ("alpasim", _alpasim_with_failed_scenes, "1 failed"),
        ("alpasim", _alpasim_subset, "2 of 4 scenes"),
        ("alpasim", _alpasim_without_score, "no official score"),
    ],
)
def test_incomplete_evals_are_refused_before_wandb_is_touched(
    stub_wandb, run_dir, eval_type, write_result, error_match
):
    result_path, expected_episode_count = write_result(run_dir)

    with pytest.raises(eval_wandb_report.EvalReportError, match=error_match):
        eval_wandb_report.report_headline_metric(eval_type, str(result_path), str(run_dir), expected_episode_count)

    assert stub_wandb.requested_run_paths == []
    assert stub_wandb.stored_summary == TRAINING_SUMMARY


def test_stale_longest6_results_csv_is_refused(stub_wandb, run_dir):
    """results.csv left over from before a completion no longer matches the route records next to it."""
    result_path = _write_longest6_results(run_dir, [100.0, 60.0, 80.0, 90.0, 70.0, 80.0], 85.6)

    with pytest.raises(eval_wandb_report.EvalReportError, match="stale"):
        eval_wandb_report.report_headline_metric(
            "longest6_v1", str(result_path), str(run_dir), ROUTE_COUNT * SEED_COUNT
        )

    assert stub_wandb.updates == []


def test_missing_episode_count_is_refused(stub_wandb, run_dir):
    result_path, _, _ = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_NUPLAN_REACTIVE)

    with pytest.raises(eval_wandb_report.EvalReportError, match="episode count"):
        eval_wandb_report.report_headline_metric("nuplan_reactive", str(result_path), str(run_dir))

    assert stub_wandb.updates == []


def test_missing_result_file_is_refused(stub_wandb, run_dir):
    with pytest.raises(eval_wandb_report.EvalReportError, match="missing result file"):
        eval_wandb_report.report_headline_metric("alpasim", str(run_dir / "results.txt"), str(run_dir), SCENE_COUNT)


def test_running_training_run_is_refused_without_a_write(stub_wandb, run_dir):
    stub_wandb.run_state = "running"
    result_path, _, _ = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_SELFPLAY)

    with pytest.raises(eval_wandb_report.EvalReportError, match="still running"):
        eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(result_path), str(run_dir))

    assert stub_wandb.updates == []


def test_summary_damaged_by_the_update_is_reported(stub_wandb, run_dir):
    def store_update_and_drop_a_training_value(values):
        stub_wandb.stored_summary.update(values)
        del stub_wandb.stored_summary["environment/collision_rate"]

    stub_wandb.store_update = store_update_and_drop_a_training_value
    result_path, _, _ = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_SELFPLAY)

    with pytest.raises(eval_wandb_report.EvalReportError, match="environment/collision_rate"):
        eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(result_path), str(run_dir))


def test_server_float_formatting_is_not_a_mismatch(stub_wandb, run_dir):
    def store_update_with_16_digits(values):
        for key, value in values.items():
            stub_wandb.stored_summary[key] = float(f"{value:.16g}") if isinstance(value, float) else value

    stub_wandb.store_update = store_update_with_16_digits
    result_path, _, expected_value = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_SELFPLAY)

    eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(result_path), str(run_dir))

    stored_value = stub_wandb.stored_summary["final_eval/selfplay_km_per_infraction"]
    assert stored_value != expected_value
    assert stored_value == pytest.approx(expected_value, rel=1e-12)


@pytest.mark.parametrize("missing_key", sorted(eval_wandb_report.RUN_IDENTITY_KEYS))
def test_incomplete_run_identity_is_rejected(tmp_path, missing_key):
    config = {key: value for key, value in RUN_IDENTITY.items() if key != missing_key}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    with pytest.raises(eval_wandb_report.EvalReportError, match=missing_key):
        eval_wandb_report.load_run_identity(str(tmp_path / "config.yaml"))


def test_blank_run_identity_is_rejected(tmp_path):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({**RUN_IDENTITY, "run_name": "   "}))

    with pytest.raises(eval_wandb_report.EvalReportError, match="run_name"):
        eval_wandb_report.load_run_identity(str(tmp_path / "config.yaml"))


def test_command_line_reports_without_importing_pufferlib(tmp_path, run_dir):
    """The eval jobs run the file by path in interpreters that must not pay for, or break on, the training stack."""
    stub_package_dir = tmp_path / "stub_site"
    stub_package_dir.mkdir()
    (stub_package_dir / "wandb.py").write_text(STUB_WANDB_MODULE)
    stored_summary_path = tmp_path / "stored_summary.json"
    stored_summary_path.write_text(json.dumps(TRAINING_SUMMARY))
    result_path, expected_episode_count, expected_value = _complete_result(run_dir, "nuplan_nonreactive")

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(eval_wandb_report.__file__)),
            "nuplan_nonreactive",
            str(result_path),
            "--run-dir",
            str(run_dir),
            "--expected-episode-count",
            str(expected_episode_count),
        ],
        env={**os.environ, "PYTHONPATH": str(stub_package_dir), "STUB_WANDB_SUMMARY_PATH": str(stored_summary_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(stored_summary_path.read_text()) == {
        **TRAINING_SUMMARY,
        "final_eval/nuplan_nonreactive_CLS": expected_value,
        "final_eval/nuplan_nonreactive_CLS_source": str(result_path.relative_to(run_dir)),
    }


def test_command_line_exits_with_the_reason_for_a_refused_report(run_dir):
    result_path, expected_episode_count = _incomplete_nuplan(run_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(eval_wandb_report.__file__)),
            "nuplan_reactive",
            str(result_path),
            "--run-dir",
            str(run_dir),
            "--expected-episode-count",
            str(expected_episode_count),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "eval_wandb_report: incomplete nuPlan eval: 4 of 5 scenarios" in completed.stderr


def test_training_logger_opens_the_run_with_its_config(monkeypatch):
    """Args shaped like load_config's return: optional top-level keys (no_model_upload) materialize on access."""
    stub = types.ModuleType("wandb")
    stub.run = types.SimpleNamespace(id="stub_run_id")
    stub.init_kwargs = None
    stub.init = lambda **kwargs: setattr(stub, "init_kwargs", kwargs)
    stub.Settings = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "wandb", stub)
    args = defaultdict(dict, {**RUN_IDENTITY, "wandb_group": "emerge_", "tag": None})

    logger = pufferl.WandbLogger(args)

    assert logger.run_id == "stub_run_id"
    assert stub.init_kwargs["id"] == RUN_IDENTITY["run_name"]
    assert stub.init_kwargs["name"] == RUN_IDENTITY["run_name"]
    assert stub.init_kwargs["project"] == RUN_IDENTITY["wandb_project"]
    assert stub.init_kwargs["group"] == "emerge_"
    assert stub.init_kwargs["resume"] == "allow"
    assert stub.init_kwargs["config"] is args


def test_summarize_benchmark_metrics_skips_benchmarks_without_a_summary():
    summary = {"num_scenarios": 1000, "num_episodes": 998, "metrics_mean": {"collision_rate": 0.11}}
    results = {
        "carla": {"episodes": [], "summary": summary},
        "nuplan_single": {"episodes": [], "summary": None},
    }

    metrics = drive_benchmark.summarize_benchmark_metrics(results, "eval_")

    assert set(metrics) == {"eval_carla/num_scenarios", "eval_carla/num_episodes", "eval_carla/collision_rate"}
    assert drive_benchmark.summarize_benchmark_metrics({}, "eval_") == {}
