"""tidy_selfplay_eval.py folds the shard folders of a parallel self-play eval into evaluation_summary.json,
obs_html/ and details/; EvalReplayCapture keeps the observations of the infracting agents only when asked."""

import json
import os
import pickle
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

import pufferlib
from pufferlib import viz
from pufferlib.ocean.evaluation_utils import eval_replay


sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "eval")
)

import tidy_selfplay_eval as tidy


STAMP = "20261109-120000"
# shard -> failing (map, seed) pairs it re-simulated and paged
SHARD_FAILURES = {
    "shard00": [("opendrive__Town02", 11)],
    "shard01": [("opendrive__Town07", 21), ("opendrive__Town07", 22)],
}
METRIC_COLUMNS = ("collision_rate", "offroad_rate", "episode_length")


def _episode_rows(pairs, shard_dir=None):
    rows = []
    for episode_id, (map_name, seed) in enumerate(pairs):
        row = {
            "map_name": map_name,
            "scenario_id": map_name.split("__")[1],
            "seed": seed,
            "collision_rate": 0.02,
            "offroad_rate": 0.0,
            "episode_length": 9000.0,
            "agents_per_batch": 50,
        }
        if shard_dir is not None:
            stem = f"{map_name}__seed_{seed}__episode_{episode_id:06d}"
            row["has_replay"] = 1
            row["replay_path"] = f"/cephfs/{shard_dir}/failures/replays_zlib/{stem}.replay.zlib"
        rows.append(row)
    return pd.DataFrame(rows)


def _write_parallel_eval(root):
    """The aggregate dir and two shard folders as parallel_eval.py leaves them."""
    benchmark_dir = root / "eval" / "carla_k_scaled_0099_1000"
    run_dir = benchmark_dir / STAMP
    run_dir.mkdir(parents=True)
    (run_dir / "evaluation_summary.json").write_text(json.dumps({"num_scenarios": 20, "num_episodes": 20}))
    all_rows = pd.concat(
        [
            _episode_rows([("opendrive__Town02", s) for s in range(10)]),
            _episode_rows([("opendrive__Town07", s) for s in range(10)]),
        ]
    )
    all_rows.to_csv(run_dir / "episode_metrics.csv", index=False)
    for shard, pairs in SHARD_FAILURES.items():
        (run_dir / f"{shard}.log").write_text(f"{shard} log line\n")
        shard_dir = benchmark_dir / f"{STAMP}_{shard}"
        failures = shard_dir / "failures"
        (failures / "rendered_replays").mkdir(parents=True)
        (shard_dir / "resolved_benchmark.yaml").write_text(f"shard: {shard}\n")
        _episode_rows([("opendrive__Town02", s) for s in range(10)]).to_csv(
            shard_dir / "episode_metrics.csv", index=False
        )
        (shard_dir / "evaluation_summary.json").write_text("{}")
        (failures / "evaluation_summary.json").write_text("{}")
        _episode_rows(pairs).to_csv(failures / "selected_failures.csv", index=False)
        _episode_rows(pairs, shard_dir=shard).to_csv(failures / "episode_metrics.csv", index=False)
        (failures / "rendered_replays" / "index.html").write_text("shard index")
        for episode_id, (map_name, seed) in enumerate(pairs):
            (failures / "rendered_replays" / f"{map_name}__seed_{seed}__episode_{episode_id:06d}.html").write_text(
                "page"
            )
    return run_dir


def _tidy(run_dir, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tidy_selfplay_eval", str(run_dir)])
    tidy.main()


def test_tidy_folds_the_shards_into_the_aggregate_dir(tmp_path, monkeypatch):
    run_dir = _write_parallel_eval(tmp_path)
    _tidy(run_dir, monkeypatch)
    assert sorted(path.name for path in run_dir.parent.iterdir()) == [STAMP]
    assert sorted(path.name for path in run_dir.iterdir()) == ["details", "evaluation_summary.json", "obs_html"]
    assert sorted(path.name for path in (run_dir / "details").iterdir()) == [
        "episode_metrics.csv",
        "failure_episodes.csv",
        "resolved_benchmark.yaml",
        "selected_failures.csv",
        "shards.log",
    ]
    assert sorted(path.name for path in (run_dir / "obs_html").iterdir()) == [
        "index.html",
        "opendrive__Town02__seed_11__episode_000000.html",
        "opendrive__Town07__seed_21__episode_000000.html",
        "opendrive__Town07__seed_22__episode_000001.html",
    ]
    assert len(pd.read_csv(run_dir / "details" / "episode_metrics.csv")) == 20
    failures = pd.read_csv(run_dir / "details" / "failure_episodes.csv")
    assert failures["shard"].tolist() == ["shard00", "shard01", "shard01"]
    assert (run_dir / "details" / "resolved_benchmark.yaml").read_text() == "shard: shard00\n"
    shards_log = (run_dir / "details" / "shards.log").read_text()
    assert "==== shard00.log ====" in shards_log and "shard01 log line" in shards_log
    index = (run_dir / "obs_html" / "index.html").read_text()
    assert "opendrive__Town07__seed_22__episode_000001" in index and "shard index" not in index


def test_tidy_is_a_no_op_the_second_time(tmp_path, monkeypatch, capsys):
    run_dir = _write_parallel_eval(tmp_path)
    _tidy(run_dir, monkeypatch)
    before = sorted(str(path) for path in run_dir.rglob("*"))
    _tidy(run_dir, monkeypatch)
    assert sorted(str(path) for path in run_dir.rglob("*")) == before
    assert "already tidied" in capsys.readouterr().out


def test_tidy_handles_a_single_process_eval_dir(tmp_path, monkeypatch):
    run_dir = tmp_path / "eval" / "carla" / STAMP
    failures = run_dir / "failures" / "rendered_replays"
    failures.mkdir(parents=True)
    (run_dir / "evaluation_summary.json").write_text("{}")
    (run_dir / "resolved_benchmark.yaml").write_text("single\n")
    _episode_rows([("opendrive__Town02", s) for s in range(3)]).to_csv(run_dir / "episode_metrics.csv", index=False)
    _episode_rows([("opendrive__Town02", 1)], shard_dir="x").to_csv(
        run_dir / "failures" / "episode_metrics.csv", index=False
    )
    (failures / "opendrive__Town02__seed_1__episode_000000.html").write_text("page")
    (failures / "index.html").write_text("old index")
    _tidy(run_dir, monkeypatch)
    assert sorted(path.name for path in run_dir.iterdir()) == ["details", "evaluation_summary.json", "obs_html"]
    assert (run_dir / "details" / "episode_metrics.csv").is_file() and (
        run_dir / "details" / "resolved_benchmark.yaml"
    ).is_file()
    assert sorted(path.name for path in (run_dir / "obs_html").iterdir()) == [
        "index.html",
        "opendrive__Town02__seed_1__episode_000000.html",
    ]


def _capture(tmp_path, observation_agents):
    args = {
        "eval": {"observation_replay_writer_count": 1, "capture_observations_agents": observation_agents},
        "env": {},
    }
    return eval_replay.EvalReplayCapture(args, None, str(tmp_path / "replays"), True, 1, 2, 4, 0)


def _bundle_summary(colliding_agent):
    metrics = np.zeros((4, 2, len(viz.METRIC_LABELS)), np.float32)
    metrics[2, colliding_agent, viz.METRIC_LABELS.index("collision")] = 1.0
    bundle = {
        "schema": "interactive_replay_environment_v1",
        "metadata": {"episode_length": 4, "worker_idx": 0, "active_agent_offset": 0, "active_agent_count": 2},
        "scenario": {},
        "frames": {"metrics_f32": metrics},
    }
    return {
        "replay_environment_bundle": zlib.compress(pickle.dumps(bundle)),
        "map_name": "x/opendrive__Town01.bin",
        "seed": 7,
    }


def test_infracting_mode_keeps_only_the_infracting_agents_observations(tmp_path):
    capture = _capture(tmp_path, "infracting")
    capture.policy_history = {
        "obs": np.ones((4, 2, 3), np.float16),
        "pool_lane": np.ones((4, 2, 5), np.int16),
        "value": np.ones((4, 2), np.float32),
    }
    capture.policy_history_frame_count = 4
    capture.queue_replay(_bundle_summary(colliding_agent=1), 0)
    replay = capture.pending_replays[0][1]
    assert replay["obs_agents"] == [1]
    assert not replay["obs"][:, 0].any() and replay["obs"][:, 1].all()
    assert not replay["pool_lane"][:, 0].any() and replay["pool_lane"][:, 1].all()
    assert replay["value"].all()  # the small per-agent chunks stay


def test_all_mode_keeps_every_agents_observations(tmp_path):
    capture = _capture(tmp_path, "all")
    capture.policy_history = {"obs": np.ones((4, 2, 3), np.float16)}
    capture.policy_history_frame_count = 4
    capture.queue_replay(_bundle_summary(colliding_agent=1), 0)
    replay = capture.pending_replays[0][1]
    assert replay["obs"].all() and "obs_agents" not in replay


def test_unknown_observation_agents_mode_is_refused(tmp_path):
    with pytest.raises(pufferlib.APIUsageError, match="capture_observations_agents"):
        _capture(tmp_path, "nobody")
