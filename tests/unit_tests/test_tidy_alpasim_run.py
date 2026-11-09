"""tidy_alpasim_run.py flattens an AlpaSim run to results.txt, run_info.txt, obs_html/ and details/, migrates the
2026-10-12 layout and leaves an unaggregated run's wizard tree; render_alpasim_obs_html.py pages the imperfect
scenes by default and re-renders a pruned run from its pages."""

import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

from pufferlib import viz


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "eval")
)

import render_alpasim_obs_html as render
import tidy_alpasim_run as tidy


FRAMES, AGENT_CAP = 20, 2
FAIL, SLOW, CLEAN = (
    "clipgt-aaaaaaaa-0000-0000-0000-000000000001",
    "clipgt-aaaaaaaa-0000-0000-0000-000000000002",
    "clipgt-aaaaaaaa-0000-0000-0000-000000000003",
)
# scene -> (score, status, at-fault collision, progress score)
SCENES = {FAIL: (0.0, "fail", 1.0, 1.0), SLOW: (0.5, "pass", 0.0, 0.5), CLEAN: (1.0, "pass", 0.0, 1.0)}
SESSION = {FAIL: "a" * 32, SLOW: "b" * 32, CLEAN: "c" * 32}
SERIES_SECONDS = np.arange(FRAMES) * 0.1
INFO_LINE = b"runtime-0-1  | 2026-11-08 08:30:00 INFO: sim event\n"
HOST_LINE = b"[2026-11-08 08:29:11][alpasim_wizard][WARNING] - host line\n"


def _replay():
    replay = {
        "env": {
            "num_goals": 2,
            "goal_radius": 2.0,
            "reward_conditioning": False,
            "obs_slots_partners_n": 4,
            "obs_slots_lane_n": 4,
            "obs_slots_boundary_n": 4,
            "obs_slots_traffic_controls_n": 1,
        },
        "agent_f32": np.zeros((FRAMES, AGENT_CAP, 13), np.float32),
        "agent_i32": np.zeros((FRAMES, AGENT_CAP, 10), np.int32),
        "metrics_f32": np.zeros((FRAMES, AGENT_CAP, len(viz.METRIC_LABELS)), np.float32),
        "puffer_f32": np.zeros((FRAMES, AGENT_CAP, 4), np.float32),
        "traffic_i16": np.zeros((FRAMES, 1, 3), np.int16),
        "raw_action": np.ones((FRAMES, 1, 2), np.float32),
        "clipped_action": np.ones((FRAMES, 1, 2), np.float32),
        "value": np.zeros((FRAMES, 1), np.float32),
        "entropy": np.zeros((FRAMES, 1), np.float32),
        "obs": None,
    }
    replay["agent_f32"][:, 0, 0] = np.arange(FRAMES)
    return replay


def _row(clipgt):
    score, status, collision, progress = SCENES[clipgt]
    metrics = {
        "collision_any": collision,
        "collision_front": collision,
        "collision_lateral": 0.0,
        "collision_rear": 0.0,
        "wrong_lane": 0.0,
        "dist_traveled_m": 100.0 * progress,
        "gt_dist_traveled_m": 100.0,
        "dist_to_gt_trajectory": 1.0,
        "lateral_dist_to_gt_trajectory": 0.5,
    }
    score_metrics = {
        "collision_at_fault": collision,
        "offroad": 0.0,
        "left_corridor_laterally": 0.0,
        "progress_score": progress,
        "progress_clipped_rel": progress * 0.8,
    }
    return {
        "clipgt_id": clipgt,
        "rollout_id": SESSION[clipgt],
        "score": score,
        "status": status,
        "failure_reason": "collision_at_fault" if status == "fail" else "",
        "metrics": metrics,
        "score_metrics": score_metrics,
    }


def _summary():
    return {
        "rollouts": [_row(c) for c in SCENES],
        "score_criteria": {"collision_at_fault": "== 0"},
        "metrics_results": [],
    }


def _metrics_frame():
    rows = []
    for clipgt in SCENES:
        for name in render.PER_STEP_SERIES:
            for step, t in enumerate(SERIES_SECONDS):
                value = 0.0
                if name == "eval_relevant" and step >= 2:
                    value = 1.0
                if name in ("collision_any", "collision_front") and clipgt == FAIL and step >= 10:
                    value = 1.0
                if name == "dist_to_gt_trajectory":
                    value = 1.0
                rows.append((name, t * 1e6, value, clipgt, SESSION[clipgt]))
    return pd.DataFrame(rows, columns=["name", "timestamps_us", "values", "clipgt_id", "rollout_id"])


def _write_fresh_run(root, aggregated=True, source_exists=True):
    """A finished 12_alpasim.sh run before the tidy: the host's alpagym/ tree with the wizard underneath."""
    run = root / "alpasim_gt_time5_r2_final_model_20261109_120000_7"
    alpagym, wizard = run / "alpagym", run / "alpagym" / "alpasim" / "wizard_0"
    (alpagym / "logs").mkdir(parents=True)
    (alpagym / "config.json").write_text("{}")
    (alpagym / "evaluation.yaml").write_text("scenes: []\n")
    (alpagym / "logs" / "evaluation.log").write_text("evaluation\n")
    obs = alpagym / "obs_replays"
    obs.mkdir()
    for clipgt in SCENES:
        viz.save_interactive_replay_zlib({}, _replay(), str(obs / f"{clipgt}.replay.zlib"))
        (obs / f"{clipgt}.html").write_text("page the run rendered")
    for clipgt in SCENES:
        rollout = wizard / "rollouts" / clipgt / SESSION[clipgt]
        rollout.mkdir(parents=True)
        (rollout / "metrics.parquet").write_bytes(b"per-rollout metrics")
        (rollout / "_complete").write_text("")
        (rollout / "rollout.asl").write_bytes(b"rollout log " + clipgt.encode())
    (wizard / "controller").mkdir()
    (wizard / "controller" / "alpasim_controller_x.csv").write_text("t,steer\n")
    (wizard / "txt-logs").mkdir()
    (wizard / "txt-logs" / "out-7-attempt-0-physics-0-log.txt").write_bytes(b"[INFO] tick\nphysics warning\n")
    (wizard / "txt-logs" / "runtime_worker_0.log").write_bytes(b"worker copy\n" * 10)
    (wizard / "prometheus").mkdir()
    (wizard / "prometheus" / "dcgm-counters.csv").write_text("x")
    for name in ("eval-config.yaml", "wizard-config.yaml", "docker-compose.yaml", "generated-user-config-0.yaml"):
        (wizard / name).write_text(yaml.safe_dump({"aggregation_modifiers": {"max_dist_to_gt_trajectory": 4.0}}))
    (wizard / "failed_rollouts.json").write_text("[]")
    if aggregated:
        (wizard / "aggregate").mkdir()
        (wizard / "aggregate" / "results-summary.json").write_text(json.dumps(_summary()))
        _metrics_frame().to_parquet(wizard / "aggregate" / "metrics_unprocessed.parquet")
        (wizard / "aggregate" / "metrics_results.png").write_bytes(b"png")
    source = root / "k_scaled_0099_1000" / "final_model.pt"
    source.parent.mkdir(exist_ok=True)
    if source_exists:
        source.write_bytes(b"weights")
    (run / "final_model.pt").write_bytes(b"weights")
    (run / "config.yaml").write_text("train: {}\n")
    (run / "run_info.txt").write_text(f"checkpoint {source} (evaluated copy: {run / 'final_model.pt'})\nscenes 3\n")
    (run / "results.txt").write_text("official_score 0.5000  scenes 3  failed 0\n")
    (run / "nre_rc.local").write_text('exec "$@"\n')
    (run / "evaluate.log").write_bytes(HOST_LINE + INFO_LINE * 5 + HOST_LINE)
    return run


def _tree(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _tidy(run, *args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tidy_alpasim_run", str(run), *args])
    tidy.main()


def _render(run, *args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["render_alpasim_obs_html", str(run), *args])
    render.main()


def test_tidy_flattens_a_fresh_run(tmp_path, monkeypatch):
    run = _write_fresh_run(tmp_path)
    _tidy(run, monkeypatch=monkeypatch)
    assert sorted(path.name for path in run.iterdir()) == ["details", "obs_html", "results.txt", "run_info.txt"]
    assert sorted(path.name for path in (run / "details").iterdir()) == [
        "config.json",
        "eval-config.yaml",
        "evaluate.log",
        "evaluation.log",
        "evaluation.yaml",
        "metrics_unprocessed.parquet",
        "results-summary.json",
        "services.log",
        "wizard-config.yaml",
    ]
    assert (run / "details" / "evaluate.log").read_bytes() == HOST_LINE + HOST_LINE
    services = (run / "details" / "services.log").read_text()
    assert "==== out-7-attempt-0-physics-0-log.txt ====" in services and "physics warning" in services
    assert "[INFO]" not in services and "worker copy" not in services
    assert sorted(path.name for path in (run / "obs_html").iterdir()) == sorted(
        [f"{c}.replay.zlib" for c in SCENES] + [f"{c}.html" for c in SCENES]
    )
    before = _tree(run)
    _tidy(run, monkeypatch=monkeypatch)
    assert _tree(run) == before


def test_tidy_keeps_the_checkpoint_copy_when_its_source_is_gone(tmp_path, monkeypatch):
    run = _write_fresh_run(tmp_path, source_exists=False)
    _tidy(run, monkeypatch=monkeypatch)
    assert (run / "final_model.pt").is_file() and (run / "config.yaml").is_file()


def test_tidy_keeps_the_selected_rollout_logs_where_the_offline_tools_look(tmp_path, monkeypatch):
    run = _write_fresh_run(tmp_path)
    _tidy(run, "--keep_rollout_logs", "imperfect", monkeypatch=monkeypatch)
    kept = sorted(str(path.relative_to(run)) for path in run.rglob("rollout.asl"))
    assert kept == [f"alpasim/wizard_0/rollouts/{c}/{SESSION[c]}/rollout.asl" for c in (FAIL, SLOW)]


def test_tidy_keeps_the_wizard_tree_of_an_unaggregated_run(tmp_path, monkeypatch, capsys):
    run = _write_fresh_run(tmp_path, aggregated=False)
    _tidy(run, monkeypatch=monkeypatch)
    assert "never aggregated" in capsys.readouterr().out
    wizard = run / "alpasim" / "wizard_0"
    assert len(list(wizard.rglob("metrics.parquet"))) == 3 and (wizard / "eval-config.yaml").is_file()
    assert not (wizard / "controller").exists() and not (wizard / "txt-logs").exists()
    assert (run / "details" / "services.log").is_file() and not (run / "details" / "results-summary.json").exists()


def test_tidy_migrates_the_2026_10_layout(tmp_path, monkeypatch):
    run = tmp_path / "alpasim_gt_time5_r2_final_model_20261012_120000_1"
    (run / "obs_replays").mkdir(parents=True)
    (run / "obs_replays" / f"s0.00_COL_{FAIL}.html").write_text("page")
    (run / "logs" / "txt-logs").mkdir(parents=True)
    for name in ("evaluate.log", "config.json", "evaluation.yaml", "evaluation.log", "run_info.txt"):
        (run / "logs" / name).write_text(name)
    (run / "logs" / "txt-logs" / "runtime_worker_3.log").write_text("copy")
    (run / "logs" / "txt-logs" / "out-1-attempt-0-renderer-0-log.txt").write_text("renderer line\n")
    (run / "alpasim").mkdir()
    for name in ("results-summary.json", "metrics_unprocessed.parquet", "eval-config.yaml", "wizard-config.yaml"):
        (run / "alpasim" / name).write_text(name)
    (run / "results.txt").write_text("official_score 0.7\n")
    _tidy(run, monkeypatch=monkeypatch)
    assert sorted(path.name for path in run.iterdir()) == ["details", "obs_html", "results.txt", "run_info.txt"]
    assert sorted(path.name for path in (run / "details").iterdir()) == [
        "config.json",
        "eval-config.yaml",
        "evaluate.log",
        "evaluation.log",
        "evaluation.yaml",
        "metrics_unprocessed.parquet",
        "results-summary.json",
        "services.log",
        "wizard-config.yaml",
    ]
    assert "renderer line" in (run / "details" / "services.log").read_text()


def test_gallery_pages_the_imperfect_scenes_and_prunes_the_replays(tmp_path, monkeypatch):
    run = _write_fresh_run(tmp_path)
    _tidy(run, monkeypatch=monkeypatch)
    _render(run, "--prune", monkeypatch=monkeypatch)
    assert sorted(path.name for path in (run / "obs_html").iterdir()) == [
        "index.html",
        f"s0.00_COL_{FAIL}.html",
        f"s0.50_PROG_{SLOW}.html",
    ]
    page = (run / "obs_html" / f"s0.00_COL_{FAIL}.html").read_text()
    assert "AlpaSim score 0.00" in page and "page the run rendered" not in page
    # a pruned run renders again from its pages
    _render(run, monkeypatch=monkeypatch)
    assert sorted(path.name for path in (run / "obs_html").iterdir()) == [
        "index.html",
        f"s0.00_COL_{FAIL}.html",
        f"s0.50_PROG_{SLOW}.html",
    ]


def test_gallery_select_failed_and_all(tmp_path, monkeypatch):
    run = _write_fresh_run(tmp_path)
    _tidy(run, monkeypatch=monkeypatch)
    _render(run, "--select", "failed", monkeypatch=monkeypatch)
    assert sorted(path.name for path in (run / "obs_html").glob("s*.html")) == [f"s0.00_COL_{FAIL}.html"]
    assert not list((run / "obs_html").glob("clipgt-*.html"))
    other = _write_fresh_run(tmp_path / "other")
    _tidy(other, monkeypatch=monkeypatch)
    _render(other, "--select", "all", monkeypatch=monkeypatch)
    assert len(list((other / "obs_html").glob("s*.html"))) == 3
