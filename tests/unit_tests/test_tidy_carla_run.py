"""tidy_carla_run.py flattens a longest6 run to results.csv, run_info.txt, obs_html/, routes/ (flat) and details/,
keeping the telemetry and world logs of the failed route runs only; render_carla_obs_html.py pages the failed route
runs by default and renders a tidied run from its pages; carla_run_layout resolves both layouts."""

import json
import os
import sys

import numpy as np

from pufferlib import viz
from pufferlib.ocean.cosim.carla.shadow_ego import TELEMETRY_HEADER


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "eval")
)

import carla_run_layout as layout
import render_carla_obs_html as render
import tidy_carla_run as tidy


FRAMES, AGENT_CAP = 40, 2
# route run -> (status, infractions): a collision, a perfect route, a min-speed-only route, a crashed route
ROUTE_RUNS = {
    "route_03_s0": (
        "Completed",
        {
            "collisions_vehicle": [
                "Agent collided against object with type=vehicle.audi.tt and id=42 at (x=1.0, y=2.0, z=0.0)"
            ]
        },
    ),
    "route_03_s1": ("Completed", {}),
    "route_07_s0": ("Completed", {"min_speed_infractions": ["Agent drove too slow"] * 3}),
    "route_07_s1": ("Failed - Simulation crashed", {}),
}
SCORES = {"route_03_s0": 50.0, "route_03_s1": 100.0, "route_07_s0": 80.5, "route_07_s1": 0.0}
TOWNS = {3: "Town01", 7: "Town02"}


def _record(route_run):
    status, infractions = ROUTE_RUNS[route_run]
    route_id = int(route_run.split("_")[1])
    return {
        "route_id": f"RouteScenario_{route_id}",
        "status": status,
        "infractions": infractions,
        "scores": {"score_composed": SCORES[route_run], "score_route": 100.0, "score_penalty": SCORES[route_run] / 100},
        "meta": {"route_length": 1000.0, "duration_game": 300.0, "duration_system": 150.0},
    }


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


def _write_attempt(route_dir, stamp, with_replay=True):
    """The files one evaluator attempt leaves: telemetry, world log, replay (unless it crashed first) and logs."""
    for sub in ("telemetry", "world_log", "obs_html"):
        (route_dir / sub).mkdir(exist_ok=True)
    (route_dir / "telemetry" / f"route_{stamp}.csv").write_text(TELEMETRY_HEADER + "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n")
    meta = json.dumps({"town": "Town01", "tick_dt": 0.05, "offset": [0.0, 0.0]})
    np.savez_compressed(
        route_dir / "world_log" / f"route_{stamp}.npz", ego=np.zeros((FRAMES, 7), np.float32), meta=meta
    )
    if with_replay:
        viz.save_interactive_replay_zlib({}, _replay(), str(route_dir / "obs_html" / f"route_{stamp}.replay.zlib"))


def _write_run(root):
    """A longest6 run as the scripts leave it before the gallery and the tidy: two routes x two seeds."""
    run = root / "carla_longest6_v2_seeds0-1_20261109_120000_7"
    for route_run in ROUTE_RUNS:
        route_dir = run / "routes" / route_run
        route_dir.mkdir(parents=True)
        _write_attempt(route_dir, "20261109_120100_000001")
        (route_dir / "evaluator_attempt1.log").write_text(f"[puffer_agent] route done {route_run}\n")
        (route_dir / "carla_server_attempt1.log").write_text("4.24.3 UE4\n")
        (route_dir / "result.json").write_text(json.dumps({"_checkpoint": {"records": [_record(route_run)]}}))
    # the crashed route was retried: a first attempt without a replay, server and evaluator logs of both
    retried = run / "routes" / "route_07_s1"
    _write_attempt(retried, "20261109_120000_000000", with_replay=False)
    (retried / "evaluator_attempt2.log").write_text("second attempt\n")
    (retried / "carla_server_attempt2.log").write_text("4.24.3 UE4\n")
    aggregate = run / "aggregate"
    for route_run in ROUTE_RUNS:
        seed_dir = aggregate / f"s{route_run[-1]}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / f"{route_run[:-3]}.json").write_text((run / "routes" / route_run / "result.json").read_text())
    (aggregate / "results.csv").write_text('"sep=,"\nAvg. driving score,57.625\n')
    (run / "run_info.txt").write_text("checkpoint /x/final_model.pt\nseeds 0 1\n")
    (run / "timing.csv").write_text(
        "route,worker,gpu,attempt,start,wall_s,game_s,pace,status,driving_score,seed\n3,0,0,1,12:01:00,100,300.0,3.0,Completed,50.00,0\n"
    )
    (run / "server_pids").write_text("1234\n")
    for worker in (0, 1, 10):
        (run / f"worker_w{worker}.log").write_text(f"[w{worker}] queue empty\n")
    (run / "compat_lib").mkdir()
    (run / "compat_lib" / "libtiff.so.5").symlink_to("/usr/lib/libtiff.so.6")
    return run


def _tree(root):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _render(run, *args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["render_carla_obs_html", str(run), *args])
    render.main()


def _tidy(run, *args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tidy_carla_run", str(run), *args])
    tidy.main()


def test_route_failed_flags_infractions_and_failed_statuses_only():
    assert layout.route_failed(_record("route_03_s0"))
    assert layout.route_failed(_record("route_07_s1"))
    assert not layout.route_failed(_record("route_03_s1"))
    assert not layout.route_failed(_record("route_07_s0"))  # min-speed infractions alone


def test_flat_stems_name_repetitions_before_the_seed():
    assert layout.flat_stem("route_08_s1", 0) == "route_08_s1"
    assert layout.flat_stem("route_08_s1", 2) == "route_08_rep2_s1"
    assert layout.flat_stem("route_08", 1) == "route_08_rep1"
    assert layout.seed_of("route_08_s1") == "1" and layout.seed_of("route_08") is None


def test_layout_pairs_the_newest_attempt_files_with_the_records(tmp_path):
    run = _write_run(tmp_path)
    runs = dict(layout.route_runs(run))
    assert sorted(runs) == sorted(ROUTE_RUNS)
    retried = runs["route_07_s1"]
    assert layout.record_stems(retried, 1) == ["route_20261109_120100_000001"]
    assert layout.record_file(retried, "obs_html", ".replay.zlib", "route_20261109_120100_000001").is_file()
    assert layout.record_stems(retried, 3) == [None, "route_20261109_120000_000000", "route_20261109_120100_000001"]


def test_gallery_pages_the_failed_route_runs_only(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    _render(run, monkeypatch=monkeypatch)
    pages = sorted(path.name for path in (run / "obs_html").iterdir())
    assert pages == ["ds000.0_Town01_route07_s1.html", "ds050.0_COL_Town01_route03_s0.html", "index.html"]
    _render(run, "--select", "all", monkeypatch=monkeypatch)
    assert len(list((run / "obs_html").glob("ds*.html"))) == 4


def test_tidy_flattens_the_run_and_keeps_the_failed_routes_logs(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    _render(run, monkeypatch=monkeypatch)
    _tidy(run, monkeypatch=monkeypatch)
    assert sorted(path.name for path in run.iterdir()) == [
        "details",
        "obs_html",
        "results.csv",
        "routes",
        "run_info.txt",
    ]
    assert sorted(path.name for path in (run / "routes").iterdir()) == [
        "route_03_s0.result.json",
        "route_03_s0.telemetry.csv",
        "route_03_s0.world_log.npz",
        "route_03_s1.result.json",
        "route_07_s0.result.json",
        "route_07_s1.result.json",
        "route_07_s1.telemetry.csv",
        "route_07_s1.world_log.npz",
    ]
    assert sorted(path.name for path in (run / "details").iterdir()) == ["evaluators.log", "timing.csv", "workers.log"]
    evaluators = (run / "details" / "evaluators.log").read_text()
    assert "==== route_07_s1 evaluator_attempt1.log ====" in evaluators and "second attempt" in evaluators
    assert (run / "details" / "workers.log").read_text().index("[w1]") < (
        run / "details" / "workers.log"
    ).read_text().index("[w10]")
    assert (run / "results.csv").read_text().startswith('"sep=,"')
    assert len(list((run / "obs_html").glob("ds*.html"))) == 2
    # the layout resolves the flat run; the kept world log pairs with its record
    runs = dict(layout.route_runs(run))
    assert sorted(runs) == sorted(ROUTE_RUNS)
    assert layout.record_stems(runs["route_03_s0"], 1) == ["route_03_s0"]
    assert (
        layout.record_file(runs["route_03_s0"], "world_log", ".npz", "route_03_s0").name == "route_03_s0.world_log.npz"
    )
    assert layout.record_file(runs["route_03_s1"], "world_log", ".npz", "route_03_s1") is None
    before = _tree(run)
    _tidy(run, monkeypatch=monkeypatch)
    assert _tree(run) == before


def test_tidy_keep_all_and_none(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    _tidy(run, "--keep", "all", monkeypatch=monkeypatch)
    assert len(list((run / "routes").glob("*.world_log.npz"))) == 4
    other = _write_run(tmp_path / "other")
    _tidy(other, "--keep", "none", monkeypatch=monkeypatch)
    assert sorted(path.name for path in (other / "routes").iterdir()) == [
        f"{r}.result.json" for r in sorted(ROUTE_RUNS)
    ]


def test_tidied_run_renders_again_from_its_pages(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    _render(run, monkeypatch=monkeypatch)
    _tidy(run, monkeypatch=monkeypatch)
    page = run / "obs_html" / "ds050.0_COL_Town01_route03_s0.html"
    page.write_text(page.read_text().replace("<body>", "<body><!-- stale panel -->", 1))
    _render(run, monkeypatch=monkeypatch)
    assert "<!-- stale panel -->" not in page.read_text()
    assert sorted(path.name for path in (run / "obs_html").iterdir()) == [
        "ds000.0_Town01_route07_s1.html",
        "ds050.0_COL_Town01_route03_s0.html",
        "index.html",
    ]


def test_a_route_driven_again_replaces_its_old_page(tmp_path, monkeypatch):
    run = _write_run(tmp_path)
    _render(run, monkeypatch=monkeypatch)
    record = _record("route_03_s0")
    record["infractions"] = {}
    record["status"] = "Failed - Agent got blocked"
    (run / "routes" / "route_03_s0" / "result.json").write_text(json.dumps({"_checkpoint": {"records": [record]}}))
    _render(run, monkeypatch=monkeypatch)
    pages = sorted(path.name for path in (run / "obs_html").iterdir())
    assert pages == ["ds000.0_Town01_route07_s1.html", "ds050.0_Town01_route03_s0.html", "index.html"]
