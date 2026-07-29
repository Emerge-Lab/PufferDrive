from datetime import datetime
import os
import pickle
from types import SimpleNamespace
import zlib

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import pufferlib
from pufferlib import pufferl
from pufferlib.ocean.drive import benchmark as drive_benchmark
from pufferlib.ocean.drive.eval_replay import EvalReplayCapture


def test_benchmark_rejects_agent_capacity_below_benchmark_maximum():
    args = {
        "train": {},
        "vec": {},
        "env": {},
        "eval": {"num_agents": 16},
    }
    benchmark = {
        "name": "carla_test",
        "seed": 42,
        "simulation_mode": "gigaflow",
        "map_dir": "maps",
        "num_maps": 1,
        "num_scenarios": 1,
        "scenario_length": 100,
        "max_agents_per_env": 50,
        "control_mode": "control_vehicles",
    }

    with pytest.raises(pufferlib.APIUsageError, match="eval.num_agents.*max_agents_per_env"):
        drive_benchmark.build_benchmark_args(args, benchmark, {})


def test_benchmark_allows_metadata_and_deduplicates_selection(tmp_path):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    (map_dir / "map.bin").write_bytes(b"")
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "description": "Optional benchmark metadata",
                "env": {},
                "benchmarks": [
                    {
                        "name": "carla_test",
                        "seed": 42,
                        "simulation_mode": "gigaflow",
                        "num_scenarios": 1,
                        "num_maps": 1,
                        "max_agents_per_env": 1,
                        "scenario_length": 100,
                        "control_mode": "control_vehicles",
                        "map_dir": str(map_dir),
                        "notes": "Optional benchmark metadata",
                    }
                ],
            }
        )
    )

    _, benchmarks = drive_benchmark.load_benchmark_config(config_path, ["carla_test", "carla_test"])

    assert [benchmark["name"] for benchmark in benchmarks] == ["carla_test"]


def test_render_filter_selection_deduplicates_names():
    render_filter = drive_benchmark.parse_render_filter_columns("collision_rate,collision_rate")

    assert render_filter == ("collision_rate",)


def test_training_evaluation_accepts_recurrent_policy(monkeypatch):
    args = {
        "train": {
            "use_rnn": True,
            "evaluation_interval_epochs": 1,
            "evaluation_benchmarks": "carla_fast",
        },
        "eval": {"benchmark_config": "benchmark.yaml"},
    }
    monkeypatch.setattr(drive_benchmark, "load_benchmark_config", lambda *_: ({}, []))

    assert drive_benchmark.validate_training_evaluation_config(args) is True


class _ZeroPolicy:
    def eval(self):
        return self

    def forward_eval(self, observations):
        return (torch.zeros((observations.shape[0], 1)),), torch.zeros((observations.shape[0], 1))


class _RecordingPolicy(_ZeroPolicy):
    def forward_eval(self, observations):
        self.observations = observations.clone()
        return super().forward_eval(observations)


class _RecordingRecurrentPolicy:
    hidden_size = 2

    def __init__(self):
        self.hidden_inputs = []

    def eval(self):
        return self

    def forward_eval(self, observations, state):
        self.hidden_inputs.append(state["lstm_h"].clone())
        state["lstm_h"] = state["lstm_h"] + 1
        state["lstm_c"] = state["lstm_c"] + 2
        return (torch.zeros((observations.shape[0], 1)),), torch.zeros((observations.shape[0], 1))


class _PoolRecordingPolicy(_ZeroPolicy):
    def __init__(self):
        self.pool_call_count = 0

    def pool_slot_counts(self, observations):
        self.pool_call_count += 1
        return {"pool_lane": torch.zeros((observations.shape[0], 1), dtype=torch.int64)}


class _EvaluationReplayVec:
    agents_per_batch = 1
    action_space = SimpleNamespace(shape=(1, 1))

    def reset(self, seed):
        return np.ones((1, 1), dtype=np.float32), {}

    def step(self, action):
        self.last_action = action
        summary = {
            "summary_type": "evaluation_episode",
            "map_name": "test_map.bin",
            "seed": 42,
            "replay_environment_bundle": zlib.compress(
                pickle.dumps(
                    {
                        "schema": "interactive_replay_environment_v1",
                        "metadata": {
                            "episode_length": 1,
                            "worker_idx": 0,
                            "active_agent_offset": 0,
                            "active_agent_count": 1,
                        },
                        "scenario": {},
                        "frames": {},
                    }
                )
            ),
        }
        empty = np.zeros(1, dtype=np.float32)
        return np.zeros((1, 1), dtype=np.float32), empty, empty, empty, [[summary]]

    def close(self):
        pass


class _RecurrentEvaluationVec:
    agents_per_batch = 2
    action_space = SimpleNamespace(shape=(2, 1))

    def __init__(self):
        self.step_count = 0

    def reset(self, seed):
        return np.ones((2, 1), dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        rewards = np.zeros(2, dtype=np.float32)
        terminals = np.array([self.step_count == 1, False])
        truncations = np.array([False, self.step_count == 2])
        infos = []
        if self.step_count == 2:
            infos = [[{"summary_type": "evaluation_episode"}]]
        return np.ones((2, 1), dtype=np.float32), rewards, terminals, truncations, infos

    def close(self):
        pass


def test_training_evaluation_keeps_training_observation_dropout(monkeypatch, tmp_path):
    benchmark = {
        "name": "carla_test",
        "seed": 42,
        "simulation_mode": "gigaflow",
        "map_dir": "maps",
        "num_maps": 1,
        "num_scenarios": 1,
        "scenario_length": 100,
        "max_agents_per_env": 16,
        "control_mode": "control_vehicles",
    }
    args = {
        "package": "ocean",
        "train": {"seed": 1, "use_rnn": False},
        "vec": {"seed": 1, "num_envs": 1},
        "env": {
            "obs_dropout_lane": 0.5,
            "obs_dropout_boundary": 0.75,
        },
        "eval": {
            "benchmark_config": "benchmark.yaml",
            "benchmarks": "carla_test",
            "output_name": None,
            "num_agents": 16,
            "max_sdc_replay_workers": 8,
            "render_scenarios": False,
            "render_filter": None,
            "max_rendered_failures": None,
            "failure_replay_csv": None,
            "capture_observations": False,
        },
    }
    captured_args = {}

    monkeypatch.setattr(
        drive_benchmark,
        "load_benchmark_config",
        lambda *_: ({"obs_dropout_lane": 0.0, "obs_dropout_boundary": 0.0}, [benchmark]),
    )
    monkeypatch.setattr(pufferl, "_plan_benchmark_eval_workers", lambda *_, **__: ([{}], 1))

    def capture_rollout(run_args, *_args, **_kwargs):
        captured_args.update(run_args)
        return []

    monkeypatch.setattr(pufferl, "_run_eval_rollout", capture_rollout)
    monkeypatch.setattr(pufferl, "_write_eval_reports", lambda *_: None)

    pufferl.eval(
        env_name="puffer_drive",
        args=args,
        policy=_ZeroPolicy(),
        eval_output_dir=str(tmp_path),
        use_training_config=True,
    )

    assert captured_args["env"]["obs_dropout_lane"] == 0.5
    assert captured_args["env"]["obs_dropout_boundary"] == 0.75


def test_eval_caps_only_sdc_replay_workers(monkeypatch, tmp_path):
    benchmarks = [
        {
            "name": "womd_single",
            "seed": 42,
            "simulation_mode": "replay",
            "map_dir": "maps",
            "num_maps": 100,
            "num_scenarios": 100,
            "scenario_length": 91,
            "max_agents_per_env": 64,
            "control_mode": "control_sdc_only",
        },
        {
            "name": "womd_multi",
            "seed": 42,
            "simulation_mode": "replay",
            "map_dir": "maps",
            "num_maps": 100,
            "num_scenarios": 100,
            "scenario_length": 91,
            "max_agents_per_env": 64,
            "control_mode": "control_vehicles",
        },
    ]
    args = {
        "package": "ocean",
        "train": {"seed": 42, "use_rnn": False},
        "vec": {"seed": 42, "num_envs": 16},
        "env": {
            "obs_dropout_lane": 0.0,
            "obs_dropout_boundary": 0.0,
        },
        "eval": {
            "benchmark_config": "benchmark.yaml",
            "benchmarks": ["womd_single", "womd_multi"],
            "output_name": None,
            "num_agents": 64,
            "max_sdc_replay_workers": 8,
            "render_scenarios": False,
            "render_filter": None,
            "max_rendered_failures": None,
            "failure_replay_csv": None,
            "capture_observations": False,
        },
    }
    planned_worker_counts = []

    monkeypatch.setattr(drive_benchmark, "load_benchmark_config", lambda *_: ({}, benchmarks))
    monkeypatch.setattr(drive_benchmark, "write_resolved_benchmark_config", lambda *_: None)

    def capture_worker_plan(_run_args, _num_scenarios, num_workers, _scenario_length, capture_replay):
        assert not capture_replay
        planned_worker_counts.append(num_workers)
        return [{} for _ in range(num_workers)], 1

    monkeypatch.setattr(pufferl, "_plan_benchmark_eval_workers", capture_worker_plan)
    monkeypatch.setattr(pufferl, "_run_eval_rollout", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pufferl, "_write_eval_reports", lambda *_: None)

    pufferl.eval(
        env_name="puffer_drive",
        args=args,
        policy=_ZeroPolicy(),
        eval_output_dir=str(tmp_path),
        use_training_config=True,
    )

    assert planned_worker_counts == [8, 16]


def test_eval_captures_and_renders_all_scenarios_during_standard_rollout(monkeypatch, tmp_path):
    evaluation_datetime = datetime(2026, 7, 28, 12, 34, 56)
    monkeypatch.setattr(pufferl, "datetime", SimpleNamespace(now=lambda: evaluation_datetime))
    benchmark = {
        "name": "carla_test",
        "seed": 42,
        "simulation_mode": "gigaflow",
        "map_dir": "maps",
        "num_maps": 1,
        "num_scenarios": 2,
        "scenario_length": 100,
        "max_agents_per_env": 16,
        "control_mode": "control_vehicles",
    }
    args = {
        "package": "ocean",
        "train": {"seed": 1, "use_rnn": False},
        "vec": {"seed": 1, "num_envs": 2},
        "env": {
            "obs_dropout_lane": 0.0,
            "obs_dropout_boundary": 0.0,
        },
        "eval": {
            "benchmark_config": "benchmark.yaml",
            "benchmarks": "carla_test",
            "output_name": None,
            "num_agents": 16,
            "render_scenarios": True,
            "render_filter": "collision_rate",
            "max_rendered_failures": None,
            "failure_replay_csv": None,
            "capture_observations": True,
            "max_sdc_replay_workers": 8,
        },
    }
    captured = {}
    summaries = [{"map_name": "map_0"}, {"map_name": "map_1"}]
    summary = {"num_scenarios": 2, "num_episodes": 2, "metrics_mean": {}}

    monkeypatch.setattr(drive_benchmark, "load_benchmark_config", lambda *_: ({}, [benchmark]))
    monkeypatch.setattr(drive_benchmark, "write_resolved_benchmark_config", lambda *_: None)

    def capture_worker_plan(_run_args, num_scenarios, num_workers, scenario_length, capture_replay):
        captured["worker_plan"] = (num_scenarios, num_workers, scenario_length, capture_replay)
        return [{} for _ in range(num_workers)], 1

    def capture_rollout(*_args, **kwargs):
        captured["rollout"] = kwargs
        return summaries

    def capture_render(render_summaries, output_dir):
        captured["render"] = (render_summaries, output_dir)

    monkeypatch.setattr(pufferl, "_plan_benchmark_eval_workers", capture_worker_plan)
    monkeypatch.setattr(pufferl, "_run_eval_rollout", capture_rollout)
    monkeypatch.setattr(pufferl, "_write_eval_reports", lambda *_: summary)
    monkeypatch.setattr(pufferl, "_render_eval_replays", capture_render)
    monkeypatch.setattr(
        pufferl,
        "_render_eval_failures",
        lambda *_args, **_kwargs: pytest.fail("all-scenario rendering must skip failure replay"),
    )

    result = pufferl.eval(
        env_name="puffer_drive",
        args=args,
        policy=_ZeroPolicy(),
        eval_output_dir=str(tmp_path),
        use_training_config=True,
    )

    benchmark_output_dir = tmp_path / "carla_test" / evaluation_datetime.strftime("%Y%m%d-%H%M%S")
    assert captured["worker_plan"] == (2, 2, 100, True)
    assert captured["rollout"]["replay_output_dir"] == str(benchmark_output_dir / "replays")
    assert captured["rollout"]["capture_observations"] is True
    assert captured["render"] == (summaries, str(benchmark_output_dir))
    assert result == {
        "carla_test": {
            "episodes": summaries,
            "summary": summary,
        }
    }


def test_eval_rejects_render_scenarios_with_failure_csv():
    args = {
        "eval": {
            "benchmark_config": "benchmark.yaml",
            "benchmarks": "carla_test",
            "output_name": None,
            "num_agents": 16,
            "max_sdc_replay_workers": 8,
            "render_scenarios": True,
            "render_filter": None,
            "max_rendered_failures": None,
            "failure_replay_csv": "episode_metrics.csv",
            "capture_observations": False,
        }
    }

    with pytest.raises(pufferlib.APIUsageError, match="render_scenarios.*failure_replay_csv"):
        pufferl.eval("puffer_drive", args=args)


def test_eval_replays_failure_csv_without_standard_rollout(monkeypatch, tmp_path):
    failure_csv_path = tmp_path / "existing_episode_metrics.csv"
    benchmark = {
        "name": "carla",
        "seed": 42,
        "simulation_mode": "gigaflow",
        "map_dir": "maps",
        "num_maps": 1,
        "num_scenarios": 10,
        "scenario_length": 100,
        "max_agents_per_env": 16,
        "control_mode": "control_vehicles",
    }
    args = {
        "load_model_path": "model.pt",
        "train": {"seed": 42, "use_rnn": False},
        "env": {
            "simulation_mode": "gigaflow",
            "control_mode": "control_vehicles",
        },
        "eval": {
            "benchmark_config": "benchmark.yaml",
            "benchmarks": "carla",
            "output_name": None,
            "num_agents": 16,
            "max_sdc_replay_workers": 8,
            "render_scenarios": False,
            "render_filter": "collision_rate",
            "max_rendered_failures": 2,
            "failure_replay_csv": str(failure_csv_path),
            "capture_observations": False,
        },
    }
    replay_call = {}

    monkeypatch.setattr(drive_benchmark, "load_benchmark_config", lambda *_: ({}, [benchmark]))
    monkeypatch.setattr(
        drive_benchmark,
        "load_checkpoint_architecture",
        lambda loaded_args: (loaded_args, "checkpoint_config.yaml"),
    )
    monkeypatch.setattr(
        drive_benchmark,
        "build_benchmark_args",
        lambda loaded_args, _benchmark, _environment_config: loaded_args,
    )
    monkeypatch.setattr(drive_benchmark, "write_resolved_benchmark_config", lambda *_: None)
    monkeypatch.setattr(
        pufferl,
        "_plan_benchmark_eval_workers",
        lambda *_args: pytest.fail("standard benchmark worker setup should be skipped"),
    )
    monkeypatch.setattr(
        pufferl,
        "_run_eval_rollout",
        lambda *_args, **_kwargs: pytest.fail("standard benchmark rollout should be skipped"),
    )

    def capture_failure_replay(
        env_name,
        _run_args,
        selected_benchmark,
        metrics_path,
        _benchmark_output_dir,
        _policy,
        capture_observations,
        max_rendered_failures,
        evaluation_policy_cache,
    ):
        assert evaluation_policy_cache == {"policy": None}
        replay_call.update(
            {
                "env_name": env_name,
                "benchmark": selected_benchmark["name"],
                "metrics_path": metrics_path,
                "capture_observations": capture_observations,
                "max_rendered_failures": max_rendered_failures,
            }
        )
        return {
            "episodes": ["failure replay"],
            "summary": {"num_scenarios": 1, "num_episodes": 1, "metrics_mean": {}},
        }

    monkeypatch.setattr(pufferl, "_render_eval_failures", capture_failure_replay)

    summaries = pufferl.eval("puffer_drive", args=args, eval_output_dir=str(tmp_path / "output"))

    assert summaries == {
        "carla": {
            "episodes": ["failure replay"],
            "summary": {"num_scenarios": 1, "num_episodes": 1, "metrics_mean": {}},
        }
    }
    assert replay_call == {
        "env_name": "puffer_drive",
        "benchmark": "carla",
        "metrics_path": str(failure_csv_path),
        "capture_observations": False,
        "max_rendered_failures": 2,
    }


def test_training_evaluation_disables_scenario_rendering(monkeypatch, tmp_path):
    captured_args = {}
    logged = {}
    args = {
        "train": {
            "evaluation_benchmarks": "carla_fast",
        },
        "eval": {
            "benchmark_config": "benchmark.yaml",
            "render_scenarios": True,
            "render_filter": "collision_rate",
            "failure_replay_csv": "episode_metrics.csv",
        },
    }

    def capture_eval(**kwargs):
        captured_args.update(kwargs["args"])
        return {
            "carla_fast": {
                "episodes": [{}],
                "summary": {
                    "num_scenarios": 1,
                    "num_episodes": 1,
                    "metrics_mean": {"offroad_rate": 0.5},
                },
            }
        }

    monkeypatch.setattr(pufferl, "eval", capture_eval)
    monkeypatch.setattr(
        drive_benchmark,
        "load_benchmark_config",
        lambda *_: pytest.fail("training evaluation must reuse the report returned by eval"),
    )

    pufferl.run_training_evaluation(
        env_name="puffer_drive",
        args=args,
        policy=_ZeroPolicy(),
        logger=SimpleNamespace(log=lambda metrics, step: logged.update({"metrics": metrics, "step": step})),
        epoch=1,
        global_step=100,
        run_dir=str(tmp_path),
    )

    assert captured_args["eval"]["render_scenarios"] is False
    assert captured_args["eval"]["render_filter"] is None
    assert captured_args["eval"]["failure_replay_csv"] is None
    assert logged == {
        "metrics": {
            "eval_carla_fast/num_scenarios": 1,
            "eval_carla_fast/num_episodes": 1,
            "eval_carla_fast/offroad_rate": 0.5,
        },
        "step": 100,
    }


def test_eval_rollout_writes_replay_bundle_and_keeps_bytes_out_of_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(pufferlib.vector, "make", lambda *args, **kwargs: _EvaluationReplayVec())

    def fake_save(scenario, replay, path):
        assert replay["obs"].shape == (1, 1, 1)
        assert replay["raw_action"].shape == (1, 1, 1)
        assert replay["clipped_action"].shape == (1, 1, 1)
        assert replay["value"].shape == (1, 1)
        assert replay["entropy"].shape == (1, 1)
        assert replay["policy_probs"].shape == (1, 1, 1)
        with open(path, "wb") as replay_file:
            replay_file.write(b"standard replay")

    monkeypatch.setattr(pufferlib.viz, "save_interactive_replay_zlib", fake_save)
    args = {
        "package": "ocean",
        "env": {},
        "eval": {"observation_replay_writer_count": 1},
        "train": {
            "seed": 42,
            "device": "cpu",
            "compile": False,
            "compile_mode": "default",
            "compile_fullgraph": False,
            "amp": True,
            "precision": "float32",
        },
        "vec": {"seed": 42},
    }

    summaries = pufferl._run_eval_rollout(
        args=args,
        env_name="puffer_drive",
        worker_env_kwargs=[{"resample_frequency": 1}],
        total_steps=1,
        desc="test replay",
        expected_episodes=1,
        policy=_ZeroPolicy(),
        replay_output_dir=tmp_path,
        capture_observations=True,
    )

    replay_path = tmp_path / "test_map__seed_42__episode_000000.replay.zlib"
    assert replay_path.read_bytes() == b"standard replay"
    assert summaries[0]["has_replay"] == 1
    assert summaries[0]["replay_path"] == str(replay_path.resolve())
    assert "replay_environment_bundle" not in summaries[0]


def test_eval_replay_capture_skips_pooling_without_observations(tmp_path):
    policy = _PoolRecordingPolicy()
    replay_capture = EvalReplayCapture(
        args={"env": {}},
        policy=policy,
        replay_output_dir=tmp_path,
        capture_observations=False,
        num_workers=1,
        agents_per_batch=1,
        capture_batch_steps=1,
        episode_id_offset=0,
    )

    replay_capture.capture_frame(
        obs=np.zeros((1, 1), dtype=np.float32),
        policy_obs_tensor=torch.zeros((1, 1)),
        raw_action=np.zeros((1, 1), dtype=np.float32),
        action=np.zeros((1, 1), dtype=np.float32),
        logits=(torch.zeros((1, 1)),),
        value=torch.zeros((1, 1)),
        logprob=torch.zeros(1),
        entropy=torch.zeros(1),
    )

    assert policy.pool_call_count == 0
    assert "pool_lane" not in replay_capture.policy_history


def test_eval_rollout_pads_policy_batch_and_slices_environment_actions(monkeypatch):
    vecenv = _EvaluationReplayVec()
    policy = _RecordingPolicy()
    monkeypatch.setattr(pufferlib.vector, "make", lambda *args, **kwargs: vecenv)
    args = {
        "package": "ocean",
        "train": {
            "seed": 42,
            "device": "cpu",
            "compile": False,
            "compile_mode": "default",
            "compile_fullgraph": False,
            "amp": True,
            "precision": "float32",
        },
        "vec": {"seed": 42},
    }

    summaries = pufferl._run_eval_rollout(
        args=args,
        env_name="puffer_drive",
        worker_env_kwargs=[{"resample_frequency": 1}],
        total_steps=1,
        desc="test padded replay",
        expected_episodes=1,
        policy=policy,
        recorded_agents_per_batch=4,
    )

    assert policy.observations.shape == (4, 1)
    assert policy.observations[0].item() == 1
    assert torch.count_nonzero(policy.observations[1:]).item() == 0
    assert vecenv.last_action.shape == (1, 1)
    assert summaries[0]["agents_per_batch"] == 4


def test_eval_rollout_carries_and_resets_recurrent_state_with_padding(monkeypatch):
    vecenv = _RecurrentEvaluationVec()
    policy = _RecordingRecurrentPolicy()
    monkeypatch.setattr(pufferlib.vector, "make", lambda *args, **kwargs: vecenv)
    args = {
        "package": "ocean",
        "train": {
            "seed": 42,
            "device": "cpu",
            "use_rnn": True,
            "compile": False,
            "compile_mode": "default",
            "compile_fullgraph": False,
            "amp": True,
            "precision": "float32",
        },
        "vec": {"seed": 42},
    }

    summaries = pufferl._run_eval_rollout(
        args=args,
        env_name="puffer_drive",
        worker_env_kwargs=[{"resample_frequency": 1}],
        total_steps=2,
        desc="test recurrent replay",
        expected_episodes=1,
        policy=policy,
        recorded_agents_per_batch=4,
    )

    assert len(policy.hidden_inputs) == 2
    assert policy.hidden_inputs[0].shape == (4, policy.hidden_size)
    assert torch.count_nonzero(policy.hidden_inputs[0]).item() == 0
    assert torch.equal(
        policy.hidden_inputs[1],
        torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )
    assert summaries[0]["agents_per_batch"] == 4


def test_failure_replay_worker_plan_balances_without_fillers():
    args = {
        "env": {"num_agents": 1024},
        "save_zlib": False,
    }
    pairs = [(map_idx, 100 + map_idx) for map_idx in range(5)]

    worker_kwargs, total_steps = pufferl._plan_failure_replay_workers(
        args=args,
        map_seed_pairs=pairs,
        num_workers=4,
        scenario_length=128,
    )

    assert total_steps == 256
    assert [kwargs["num_eval_scenarios"] for kwargs in worker_kwargs] == [2, 1, 1, 1]
    assert [map_idx for kwargs in worker_kwargs for map_idx in kwargs["eval_map_indices"]] == list(range(5))
    assert all(kwargs["capture_replay"] for kwargs in worker_kwargs)


def test_render_eval_replays_writes_pages_with_navigation_and_index(monkeypatch, tmp_path):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_paths = [replay_dir / f"episode_{episode_id:06d}.replay.zlib" for episode_id in range(2)]
    for replay_path in replay_paths:
        replay_path.write_bytes(b"replay")

    render_calls = []

    def fake_render(replay_path, output_path):
        render_calls.append((replay_path, output_path))
        with open(output_path, "w") as html_file:
            html_file.write("rendered")

    def fake_index(output_path, file_metrics):
        assert sorted(file_metrics) == ["episode_000000.html", "episode_000001.html"]
        with open(os.path.join(output_path, "index.html"), "w") as html_file:
            html_file.write("index")

    monkeypatch.setattr(pufferlib.viz, "render_interactive_replay_zlib", fake_render)
    monkeypatch.setattr(pufferlib.viz, "build_gallery_index", fake_index)
    summaries = [
        {
            "map_name": f"map_{episode_id}",
            "scenario_id": f"scenario_{episode_id}",
            "has_replay": 1,
            "replay_path": str(replay_path),
        }
        for episode_id, replay_path in enumerate(replay_paths)
    ]

    render_dir = pufferl._render_eval_replays(summaries, str(tmp_path))

    assert render_dir == str(tmp_path / "rendered_replays")
    assert (tmp_path / "rendered_replays" / "episode_000000.html").read_text() == "rendered"
    assert (tmp_path / "rendered_replays" / "episode_000001.html").read_text() == "rendered"
    assert (tmp_path / "rendered_replays" / "index.html").read_text() == "index"
    assert render_calls[0][0] == str(replay_paths[0])
    assert render_calls[1][0] == str(replay_paths[1])


def test_render_eval_failures_limits_selection_to_first_rows(monkeypatch, tmp_path):
    selected_rows = pd.DataFrame(
        {
            "map_name": ["map_0", "map_1", "map_2"],
            "seed": [100, 101, 102],
            "agents_per_batch": [16, 16, 16],
        }
    )
    replay_pairs = []

    monkeypatch.setattr(drive_benchmark, "select_render_rows", lambda *_: selected_rows)
    monkeypatch.setattr(pufferl, "_resolve_map_indices", lambda _map_dir, map_names: list(range(len(map_names))))

    def capture_replay_pairs(_args, map_seed_pairs, _num_workers, _scenario_length):
        replay_pairs.extend(map_seed_pairs)
        return [{} for _ in map_seed_pairs], 1

    monkeypatch.setattr(pufferl, "_plan_failure_replay_workers", capture_replay_pairs)

    def fake_rollout(
        _run_args,
        _env_name,
        _worker_env_kwargs,
        _total_steps,
        _description,
        expected_episodes,
        **_kwargs,
    ):
        return [{} for _ in range(expected_episodes)]

    monkeypatch.setattr(pufferl, "_run_eval_rollout", fake_rollout)
    monkeypatch.setattr(pufferl, "_write_eval_reports", lambda *_: None)
    monkeypatch.setattr(pufferl, "_render_eval_replays", lambda *_: None)
    run_args = {
        "eval": {"render_filter": "collision_rate"},
        "vec": {"num_envs": 4},
        "env": {
            "map_dir": "maps",
            "scenario_length": 100,
        },
    }

    result = pufferl._render_eval_failures(
        "puffer_drive",
        run_args,
        {"name": "carla"},
        str(tmp_path / "episode_metrics.csv"),
        str(tmp_path),
        _ZeroPolicy(),
        capture_observations=False,
        max_rendered_failures=2,
    )

    written_rows = pd.read_csv(tmp_path / "failures" / "selected_failures.csv")
    assert written_rows["map_name"].tolist() == ["map_0", "map_1"]
    assert replay_pairs == [(0, 100), (1, 101)]
    assert len(result["episodes"]) == 2
    assert result["summary"] is None
