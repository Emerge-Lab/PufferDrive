"""Unit tests for logging an evaluation's headline metric to the history and summary of the training wandb run."""

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


RUN_IDENTITY = {"run_name": "k_exp_0002_1000", "wandb_project": "nightly-multi-long"}
RUN_PATH = "nightly-multi-long/k_exp_0002_1000"
RUN_ENTITY = "emerge_"
TRAINING_LAST_STEP = 1000190509056
TRAINING_SUMMARY = {"environment/collision_rate": 0.0123, "_step": TRAINING_LAST_STEP, "early_stop": False}
TRAINING_HISTORY_ROW = {"environment/collision_rate": 0.0123, "_step": TRAINING_LAST_STEP}
ROUTE_COUNT = 3
SEED_COUNT = 2
SCENARIO_COUNT = 5
SCENE_COUNT = 4
STUB_WANDB_MODULE = """
import json
import os
import sys


def _load_run():
    with open(os.environ["STUB_WANDB_RUN_PATH"]) as run_file:
        return json.load(run_file)


def _store_run(stored_run):
    with open(os.environ["STUB_WANDB_RUN_PATH"], "w") as run_file:
        json.dump(stored_run, run_file)


class _SessionSummary:
    def __setitem__(self, key, value):
        stored_run = _load_run()
        stored_run["summary"][key] = value
        _store_run(stored_run)


class _Session:
    summary = _SessionSummary()

    def log(self, values):
        stored_run = _load_run()
        step = stored_run["history"][-1]["_step"] + 1
        stored_run["history"].append({**values, "_step": step})
        stored_run["summary"].update({**values, "_step": step})
        _store_run(stored_run)

    def finish(self):
        pass


class _Run:
    state = "finished"
    entity = "emerge_"

    @property
    def summary_metrics(self):
        return _load_run()["summary"]

    @property
    def history_keys(self):
        history = _load_run()["history"]
        keys = {key: {} for row in history for key in row if not key.startswith("_")}
        return {"keys": keys, "lastStep": history[-1]["_step"]}


class Api:
    def flush(self):
        pass

    def run(self, run_path):
        assert "pufferlib" not in sys.modules
        return _Run()


def Settings(**kwargs):
    return kwargs


def init(**kwargs):
    return _Session()
"""


class _StubSession:
    """The run wandb.init resumes: its summary is the stored one, log() goes through the stub's store_row."""

    def __init__(self, stub):
        self._stub = stub
        self.summary = stub.stored_summary

    def log(self, values):
        self._stub.store_row(values)

    def finish(self):
        self._stub.finished_session_count += 1


class _StubRun:
    def __init__(self, stub):
        self.state = stub.run_state
        self.entity = RUN_ENTITY
        self.summary_metrics = dict(stub.stored_summary)
        logged_keys = {key for row in stub.stored_history for key in row if not key.startswith("_")}
        self.history_keys = {"keys": {key: {} for key in logged_keys}, "lastStep": stub.stored_history[-1]["_step"]}


class _StubApi:
    """Caches the run like wandb.Api does: read again without flush(), it is the object from before the session."""

    def __init__(self, stub):
        self._stub = stub
        self._cached_run = None

    def flush(self):
        self._cached_run = None

    def run(self, run_path):
        self._stub.requested_run_paths.append(run_path)
        if self._cached_run is None:
            self._cached_run = _StubRun(self._stub)
        return self._cached_run


def _append_history_row(stub, values):
    """What wandb does with a logged row: one step after the last one, and its values become the summary's."""
    step = stub.stored_history[-1]["_step"] + 1
    stub.stored_history.append({**values, "_step": step})
    stub.stored_summary.update({**values, "_step": step})


@pytest.fixture
def stub_wandb(monkeypatch):
    """A wandb module backed by one stored run: the public API reads it, a resumed session logs to it."""
    stub = types.ModuleType("wandb")
    stub.run_state = "finished"
    stub.stored_summary = dict(TRAINING_SUMMARY)
    stub.stored_history = [dict(TRAINING_HISTORY_ROW)]
    stub.requested_run_paths = []
    stub.init_kwargs = None
    stub.finished_session_count = 0
    stub.store_row = lambda values: _append_history_row(stub, values)
    stub.api_count = 0

    def api():
        stub.api_count += 1
        return _StubApi(stub)

    stub.Api = api
    stub.Settings = lambda **kwargs: kwargs

    def init(**kwargs):
        stub.init_kwargs = kwargs
        return _StubSession(stub)

    stub.init = init
    monkeypatch.setitem(sys.modules, "wandb", stub)
    monkeypatch.setattr(eval_wandb_report, "REPORT_VERIFY_BACKOFF_SECONDS", 0.0)
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
def test_each_eval_type_logs_its_headline_value_and_source(stub_wandb, run_dir, eval_type):
    result_path, expected_episode_count, expected_value = _complete_result(run_dir, eval_type)

    eval_wandb_report.report_headline_metric(eval_type, str(result_path), str(run_dir), expected_episode_count)

    metric_key = eval_wandb_report.HEADLINE_METRIC_KEYS[eval_type]
    reported_values = {metric_key: expected_value, f"{metric_key}_source": str(result_path.relative_to(run_dir))}
    reported_step = TRAINING_LAST_STEP + 1
    assert stub_wandb.requested_run_paths[0] == RUN_PATH
    # The value is a history row, which is what the charts plot; its source string stays out of the history.
    assert stub_wandb.stored_history == [TRAINING_HISTORY_ROW, {metric_key: expected_value, "_step": reported_step}]
    # Training's own summary values stay exactly as they were; only wandb's step bookkeeping advances.
    assert stub_wandb.stored_summary == {**TRAINING_SUMMARY, **reported_values, "_step": reported_step}
    assert stub_wandb.finished_session_count == 1
    # One api object for the whole report: each further one repeats a handshake that can time out.
    assert stub_wandb.api_count == 1


def test_session_resumes_the_training_run_and_sends_nothing_else_of_it(stub_wandb, run_dir):
    result_path, _, _ = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_SELFPLAY)

    eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(result_path), str(run_dir))

    # Name, group, tags, notes and config are not sent again, so wandb keeps the training run's.
    assert set(stub_wandb.init_kwargs) == {"entity", "project", "id", "resume", "dir", "settings"}
    assert stub_wandb.init_kwargs["entity"] == RUN_ENTITY
    assert stub_wandb.init_kwargs["project"] == RUN_IDENTITY["wandb_project"]
    assert stub_wandb.init_kwargs["id"] == RUN_IDENTITY["run_name"]
    # A missing run must fail loudly rather than open a stray eval-only run.
    assert stub_wandb.init_kwargs["resume"] == "must"
    # No console log, machine metadata or system metrics of the reporting process on the training run.
    assert stub_wandb.init_kwargs["settings"] == {"console": "off", "x_disable_meta": True, "x_disable_stats": True}
    # The session's local files are temporary, not a second run folder in the checkout.
    assert not os.path.exists(stub_wandb.init_kwargs["dir"])


def test_headline_keys_are_one_per_eval_type():
    assert eval_wandb_report.HEADLINE_METRIC_KEYS == {
        "longest6_v1": "final_eval/longest6_v1_DS",
        "longest6_v2": "final_eval/longest6_v2_DS",
        "nuplan_reactive": "final_eval/nuplan_reactive_CLS",
        "nuplan_nonreactive": "final_eval/nuplan_nonreactive_CLS",
        "selfplay": "final_eval/selfplay_km_per_infraction",
        "alpasim": "final_eval/alpasim_score",
    }


def test_repeated_report_becomes_the_runs_value(stub_wandb, run_dir):
    first_result = _write_selfplay_summary(run_dir, 691229300.0, SCENARIO_COUNT)
    eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(first_result), str(run_dir))

    second_result = run_dir / "eval" / "carla_k_exp_0002_1000_redlight" / "evaluation_summary.json"
    second_result.parent.mkdir(parents=True)
    second_result.write_text(first_result.read_text().replace("691229300.0", "200820400.0"))
    eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(second_result), str(run_dir))

    # One summary value per run and eval type, and the source names the eval the value came from.
    assert stub_wandb.stored_summary == {
        **TRAINING_SUMMARY,
        "final_eval/selfplay_km_per_infraction": 200820.4,
        "final_eval/selfplay_km_per_infraction_source": "eval/carla_k_exp_0002_1000_redlight/evaluation_summary.json",
        "_step": TRAINING_LAST_STEP + 2,
    }
    # The history keeps both reports, the later one on the later step.
    reported_rows = stub_wandb.stored_history[1:]
    assert [row["final_eval/selfplay_km_per_infraction"] for row in reported_rows] == [691229.3, 200820.4]


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
    assert stub_wandb.init_kwargs is None
    assert stub_wandb.stored_summary == TRAINING_SUMMARY


def test_stale_longest6_results_csv_is_refused(stub_wandb, run_dir):
    """results.csv left over from before a completion no longer matches the route records next to it."""
    result_path = _write_longest6_results(run_dir, [100.0, 60.0, 80.0, 90.0, 70.0, 80.0], 85.6)

    with pytest.raises(eval_wandb_report.EvalReportError, match="stale"):
        eval_wandb_report.report_headline_metric(
            "longest6_v1", str(result_path), str(run_dir), ROUTE_COUNT * SEED_COUNT
        )

    assert stub_wandb.init_kwargs is None


def test_missing_episode_count_is_refused(stub_wandb, run_dir):
    result_path, _, _ = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_NUPLAN_REACTIVE)

    with pytest.raises(eval_wandb_report.EvalReportError, match="episode count"):
        eval_wandb_report.report_headline_metric("nuplan_reactive", str(result_path), str(run_dir))

    assert stub_wandb.init_kwargs is None


def test_missing_result_file_is_refused(stub_wandb, run_dir):
    with pytest.raises(eval_wandb_report.EvalReportError, match="missing result file"):
        eval_wandb_report.report_headline_metric("alpasim", str(run_dir / "results.txt"), str(run_dir), SCENE_COUNT)


def test_running_training_run_is_refused_without_a_session(stub_wandb, run_dir):
    stub_wandb.run_state = "running"
    result_path, _, _ = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_SELFPLAY)

    with pytest.raises(eval_wandb_report.EvalReportError, match="still running"):
        eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(result_path), str(run_dir))

    assert stub_wandb.init_kwargs is None


def test_summary_damaged_by_the_session_is_reported(stub_wandb, run_dir):
    def append_row_and_drop_a_training_value(values):
        _append_history_row(stub_wandb, values)
        del stub_wandb.stored_summary["environment/collision_rate"]

    stub_wandb.store_row = append_row_and_drop_a_training_value
    result_path, _, _ = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_SELFPLAY)

    with pytest.raises(eval_wandb_report.EvalReportError, match="environment/collision_rate"):
        eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(result_path), str(run_dir))


def test_value_that_reaches_only_the_summary_is_reported(stub_wandb, run_dir):
    """A summary value without a history row is in no chart: the state every run from 0072 to 0077 was left in."""
    stub_wandb.store_row = stub_wandb.stored_summary.update
    result_path, _, _ = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_SELFPLAY)

    with pytest.raises(eval_wandb_report.EvalReportError, match="did not reach the run's history"):
        eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(result_path), str(run_dir))


def test_failed_log_still_closes_the_session(stub_wandb, run_dir):
    """An open session keeps the training run in the running state, which refuses every later report."""

    def fail_to_store(values):
        raise ConnectionError("file stream unreachable")

    stub_wandb.store_row = fail_to_store
    result_path, _, _ = _complete_result(run_dir, eval_wandb_report.EVAL_TYPE_SELFPLAY)

    with pytest.raises(ConnectionError):
        eval_wandb_report.report_headline_metric(eval_wandb_report.EVAL_TYPE_SELFPLAY, str(result_path), str(run_dir))

    assert stub_wandb.finished_session_count == 1


def test_server_float_formatting_is_not_a_mismatch(stub_wandb, run_dir):
    def append_row_with_16_digits(values):
        _append_history_row(stub_wandb, {key: float(f"{value:.16g}") for key, value in values.items()})

    stub_wandb.store_row = append_row_with_16_digits
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
    stored_run_path = tmp_path / "stored_run.json"
    stored_run_path.write_text(json.dumps({"summary": TRAINING_SUMMARY, "history": [TRAINING_HISTORY_ROW]}))
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
        env={**os.environ, "PYTHONPATH": str(stub_package_dir), "STUB_WANDB_RUN_PATH": str(stored_run_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    reported_step = TRAINING_LAST_STEP + 1
    assert json.loads(stored_run_path.read_text()) == {
        "summary": {
            **TRAINING_SUMMARY,
            "final_eval/nuplan_nonreactive_CLS": expected_value,
            "final_eval/nuplan_nonreactive_CLS_source": str(result_path.relative_to(run_dir)),
            "_step": reported_step,
        },
        "history": [
            TRAINING_HISTORY_ROW,
            {"final_eval/nuplan_nonreactive_CLS": expected_value, "_step": reported_step},
        ],
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


def test_tidied_longest6_run_reads_its_flat_route_records(tmp_path):
    """After tidy_carla_run.py results.csv sits at the run's top and the records are routes/*.result.json; other
    json files under the run are not records."""
    run = tmp_path / "carla_longest6_v2_seeds0-1_20261109_120000_7"
    (run / "routes").mkdir(parents=True)
    (run / "details").mkdir()
    (run / "details" / "not_a_record.json").write_text("{}")
    scores = [100.0, 60.0, 80.0]
    for idx, score in enumerate(scores):
        record = {"route_id": f"RouteScenario_{idx}", "status": "Completed", "scores": {"score_composed": score}}
        record_path = run / "routes" / f"route_{idx:02d}_s0.result.json"
        record_path.write_text(json.dumps({"_checkpoint": {"records": [record]}}))
    with open(run / "results.csv", "w", newline="") as results_file:
        csv.writer(results_file).writerows([["sep=,"], ["Avg. driving score", 80.0]])

    assert eval_wandb_report.read_longest6_driving_score(str(run / "results.csv"), len(scores)) == 80.0
    with pytest.raises(eval_wandb_report.EvalReportError, match="incomplete"):
        eval_wandb_report.read_longest6_driving_score(str(run / "results.csv"), len(scores) + 1)
