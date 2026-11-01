"""tidy_nuplan_run.py flattens a nuPlan run to overview.csv, scenarios.csv, obs_html/ and details/ and keeps the logs
and videos of the selected scenarios only; nuplan_run_layout resolves the fresh and the tidied layout."""

import os
import sys

import pandas as pd

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "eval")
)

import nuplan_run_layout
import tidy_nuplan_run as tidy

METRICS = (
    "no_ego_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_making_progress",
    "time_to_collision_within_bound",
    "ego_progress_along_expert_route",
    "speed_limit_compliance",
    "ego_is_comfortable",
)
# token -> (score, at-fault collision metric): an infraction, a failure without infraction, a perfect scenario
COLLIDING, SLOW, CLEAN = "aaaa000000000001", "aaaa000000000002", "aaaa000000000003"
SCENARIOS = {COLLIDING: (0.0, 0.0), SLOW: (0.5, 1.0), CLEAN: (1.0, 1.0)}
CHALLENGE = "closed_loop_reactive_agents"
TIMESTAMP = "2026.01.01.00.00.00"


def _aggregator_table():
    rows = []
    for token, (score, collision) in SCENARIOS.items():
        row = {"scenario": token, "log_name": "log", "scenario_type": "turn", "score": score, "num_scenarios": None}
        row.update({metric: 1.0 for metric in METRICS})
        row["no_ego_at_fault_collisions"] = collision
        rows.append(row)
    final = {"scenario": "final_score", "log_name": None, "scenario_type": "final_score", "score": 0.5}
    rows.append({**final, "num_scenarios": len(SCENARIOS), **{metric: 1.0 for metric in METRICS}})
    return pd.DataFrame(rows)


def _write_run(group):
    sim = group / "simulation" / f"{CHALLENGE}_pufferdrive" / TIMESTAMP
    aggregator = sim / "aggregator_metric"
    aggregator.mkdir(parents=True)
    table = _aggregator_table()
    table.to_csv(aggregator / f"{CHALLENGE}_weighted_average_metrics_{TIMESTAMP}.csv", index=False)
    table.to_parquet(aggregator / f"{CHALLENGE}_weighted_average_metrics_{TIMESTAMP}.parquet")
    (aggregator / "overview.csv").write_text("CLS\n0.5\n")
    (aggregator / "scenarios.csv").write_text(
        "scenario,score\n" + "".join(f"{t},{s}\n" for t, (s, _) in SCENARIOS.items())
    )
    (sim / "metrics").mkdir()
    comfort = pd.DataFrame({"scenario_name": list(SCENARIOS), "metric_score": [0.0, 1.0, 1.0]})
    comfort.to_parquet(sim / "metrics" / "ego_is_comfortable.parquet")
    (sim / "visualization").mkdir()
    for token in SCENARIOS:
        log_dir = sim / "simulation_log" / "PufferDrivePlanner" / "turn" / "log" / token
        log_dir.mkdir(parents=True)
        (log_dir / f"{token}.msgpack.xz").write_bytes(b"log " + token.encode())
        (sim / "visualization" / f"{token}.avi").write_bytes(b"avi " + token.encode())
    (sim / "code" / "hydra").mkdir(parents=True)
    for name in ("config.yaml", "overrides.yaml", "hydra.yaml"):
        (sim / "code" / "hydra" / name).write_text(name)
    (sim / "runner_report.parquet").write_bytes(b"report")
    (sim / "log.txt").write_text("log")
    (sim / "nuboard_1.nuboard").write_bytes(b"nuboard")
    (sim / "logs").mkdir()
    obs = group / "obs_html"
    obs.mkdir()
    (obs / f"s0.00_COL_turn_{COLLIDING}.html").write_text("page")
    (obs / f"{COLLIDING}.replay.zlib").write_bytes(b"replay")
    (obs / f"{SLOW}.replay.zlib").write_bytes(b"replay")
    (obs / "index.html").write_text("index")
    return sim


def _copy_instead_of_transcode(avi_path, mp4_path, ffmpeg):
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    mp4_path.write_bytes(avi_path.read_bytes())


def _tree(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _run(group, *args, monkeypatch):
    monkeypatch.setattr(tidy, "transcode_video", _copy_instead_of_transcode)
    monkeypatch.setattr(sys, "argv", ["tidy_nuplan_run", str(group), *args])
    tidy.main()


def test_layout_resolves_the_fresh_run(tmp_path):
    sim = _write_run(tmp_path)
    assert nuplan_run_layout.details_dir(tmp_path) == sim
    assert nuplan_run_layout.metrics_dir(tmp_path) == sim / "metrics"
    assert nuplan_run_layout.aggregated_metrics_csv(tmp_path).parent == sim / "aggregator_metric"
    assert nuplan_run_layout.scenarios_csv(tmp_path) == sim / "aggregator_metric" / "scenarios.csv"
    assert set(nuplan_run_layout.simulation_logs(tmp_path)) == set(SCENARIOS)


def test_tidy_keeps_infractions_only(tmp_path, monkeypatch):
    _write_run(tmp_path)
    _run(tmp_path, monkeypatch=monkeypatch)
    details = tmp_path / "details"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["details", "obs_html", "overview.csv", "scenarios.csv"]
    assert sorted(p.name for p in details.iterdir()) == [
        "aggregated_metrics.csv",
        "config.yaml",
        "log.txt",
        "metrics",
        "overrides.yaml",
        "runner_report.parquet",
        "simulation_log",
        "videos",
    ]
    assert [p.name for p in (details / "simulation_log").iterdir()] == [f"{COLLIDING}.msgpack.xz"]
    assert [p.name for p in (details / "videos").iterdir()] == [f"{COLLIDING}.mp4"]
    assert (details / "metrics" / "ego_is_comfortable.parquet").is_file()
    assert (tmp_path / "overview.csv").read_text() == "CLS\n0.5\n"
    # the replay of the rendered page goes, the replay without a page stays
    assert sorted(p.name for p in (tmp_path / "obs_html").iterdir()) == [
        f"{SLOW}.replay.zlib",
        "index.html",
        f"s0.00_COL_turn_{COLLIDING}.html",
    ]
    assert nuplan_run_layout.details_dir(tmp_path) == details
    assert nuplan_run_layout.metrics_dir(tmp_path) == details / "metrics"
    assert nuplan_run_layout.aggregated_metrics_csv(tmp_path) == details / "aggregated_metrics.csv"
    assert nuplan_run_layout.scenarios_csv(tmp_path) == tmp_path / "scenarios.csv"
    assert list(nuplan_run_layout.simulation_logs(tmp_path)) == [COLLIDING]


def test_tidy_failures_mode_keeps_low_scores(tmp_path, monkeypatch):
    _write_run(tmp_path)
    _run(tmp_path, "--keep", "failures", monkeypatch=monkeypatch)
    assert set(nuplan_run_layout.simulation_logs(tmp_path)) == {COLLIDING, SLOW}
    assert sorted(p.stem for p in (tmp_path / "details" / "videos").iterdir()) == sorted([COLLIDING, SLOW])


def test_tidy_leaves_unknown_files_and_reports_them(tmp_path, monkeypatch, capsys):
    sim = _write_run(tmp_path)
    (sim / "extra.bin").write_bytes(b"?")
    _run(tmp_path, monkeypatch=monkeypatch)
    assert _tree(tmp_path / "simulation") == [
        f"{CHALLENGE}_pufferdrive",
        f"{CHALLENGE}_pufferdrive/{TIMESTAMP}",
        f"{CHALLENGE}_pufferdrive/{TIMESTAMP}/extra.bin",
    ]
    assert "left in place, unknown" in capsys.readouterr().out


def test_tidy_is_a_no_op_the_second_time_and_on_an_unscored_run(tmp_path, monkeypatch, capsys):
    _write_run(tmp_path)
    _run(tmp_path, monkeypatch=monkeypatch)
    before = _tree(tmp_path)
    _run(tmp_path, monkeypatch=monkeypatch)
    assert _tree(tmp_path) == before
    assert "already tidied" in capsys.readouterr().out
    unscored = tmp_path / "unscored"
    sim = _write_run(unscored)
    for path in (sim / "aggregator_metric").iterdir():
        path.unlink()
    before = _tree(unscored)
    _run(unscored, monkeypatch=monkeypatch)
    assert _tree(unscored) == before
    assert "never scored" in capsys.readouterr().out
