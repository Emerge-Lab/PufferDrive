"""Boundary and data-contract tests for benchmark evaluation.

These tests deliberately stop short of a real simulator rollout. They document
the inputs accepted by the evaluator and the reports, worker plans, and replay
metadata it produces.
"""

import json
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import pufferlib
from pufferlib import pufferl
from pufferlib.ocean.drive import benchmark as drive_benchmark
from pufferlib.ocean.drive.eval_replay import EvalReplayCapture


def _benchmark(default_map_dir, **overrides):
    benchmark = {
        "name": "carla_tiny",
        "seed": 42,
        "simulation_mode": "gigaflow",
        "map_dir": str(default_map_dir),
        "num_maps": 1,
        "num_scenarios": 1,
        "scenario_length": 32,
        "max_agents_per_env": 8,
        "control_mode": "control_vehicles",
    }
    benchmark.update(overrides)
    return benchmark


def _write_benchmark_config(tmp_path, benchmark, environment_config=None):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "env": {} if environment_config is None else environment_config,
                "benchmarks": [benchmark],
            }
        )
    )
    return config_path


def _make_map_dir(tmp_path, map_count=2):
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    for map_idx in range(map_count):
        (map_dir / f"map_{map_idx}.bin").write_bytes(b"map")
    return map_dir


@pytest.mark.parametrize(
    ("config_contents", "error_match"),
    [
        ("env: [", "invalid YAML"),
        ("- not\n- a mapping\n", "must be a mapping"),
    ],
)
def test_benchmark_config_rejects_invalid_yaml_or_root_type(tmp_path, config_contents, error_match):
    """Malformed YAML and non-mapping roots fail before environment creation."""
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(config_contents)

    with pytest.raises(pufferlib.APIUsageError, match=error_match):
        drive_benchmark.load_benchmark_config(config_path, "carla_tiny")


def test_benchmark_config_rejects_non_mapping_environment(tmp_path):
    """The shared environment section must be a key-value mapping."""
    map_dir = _make_map_dir(tmp_path)
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump({"env": [], "benchmarks": [_benchmark(map_dir)]}))

    with pytest.raises(pufferlib.APIUsageError, match="config env must be a mapping"):
        drive_benchmark.load_benchmark_config(config_path, "carla_tiny")


def test_benchmark_config_rejects_unknown_environment_key(tmp_path):
    """Typos in benchmark environment overrides must not be silently ignored."""
    map_dir = _make_map_dir(tmp_path)
    config_path = _write_benchmark_config(
        tmp_path,
        _benchmark(map_dir),
        environment_config={"unknown_drive_setting": 1},
    )

    with pytest.raises(pufferlib.APIUsageError, match="unsupported environment keys.*unknown_drive_setting"):
        drive_benchmark.load_benchmark_config(config_path, "carla_tiny")


def test_benchmark_config_rejects_duplicate_and_unknown_names(tmp_path):
    """Benchmark names are unique, and every requested name must be configured."""
    map_dir = _make_map_dir(tmp_path)
    config_path = tmp_path / "benchmark.yaml"
    benchmark = _benchmark(map_dir)
    config_path.write_text(yaml.safe_dump({"env": {}, "benchmarks": [benchmark, benchmark]}))

    with pytest.raises(pufferlib.APIUsageError, match="duplicate benchmark name"):
        drive_benchmark.load_benchmark_config(config_path, "carla_tiny")

    config_path = _write_benchmark_config(tmp_path, benchmark)
    with pytest.raises(pufferlib.APIUsageError, match="Unknown benchmarks: missing"):
        drive_benchmark.load_benchmark_config(config_path, "missing")


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_match"),
    [
        ("simulation_mode", "invalid", "simulation_mode"),
        ("seed", True, "seed"),
        ("seed", -1, "seed"),
        ("seed", 2**31, "seed"),
        ("num_scenarios", 0, "num_scenarios"),
        ("scenario_length", 0, "scenario_length"),
        ("max_agents_per_env", 0, "max_agents_per_env"),
        ("num_maps", 0, "num_maps"),
        ("control_mode", "", "control_mode"),
    ],
)
def test_benchmark_config_rejects_invalid_fields(tmp_path, field_name, invalid_value, error_match):
    """Invalid benchmark scalars fail at the external configuration boundary."""
    map_dir = _make_map_dir(tmp_path)
    config_path = _write_benchmark_config(
        tmp_path,
        _benchmark(map_dir, **{field_name: invalid_value}),
    )

    with pytest.raises(pufferlib.APIUsageError, match=error_match):
        drive_benchmark.load_benchmark_config(config_path, "carla_tiny")


def test_benchmark_config_rejects_invalid_map_paths_and_counts(tmp_path):
    """Map paths and requested scenario counts are validated before allocation."""
    map_dir = _make_map_dir(tmp_path)
    non_map_path = tmp_path / "not_a_map.txt"
    non_map_path.write_text("not a binary map")

    for invalid_path in (tmp_path / "missing", non_map_path):
        config_path = _write_benchmark_config(
            tmp_path,
            _benchmark(map_dir, map_dir=str(invalid_path)),
        )
        with pytest.raises(pufferlib.APIUsageError, match="map path does not exist"):
            drive_benchmark.load_benchmark_config(config_path, "carla_tiny")

    config_path = _write_benchmark_config(tmp_path, _benchmark(map_dir, num_maps=3))
    with pytest.raises(pufferlib.APIUsageError, match="requests 3 maps"):
        drive_benchmark.load_benchmark_config(config_path, "carla_tiny")

    config_path = _write_benchmark_config(
        tmp_path,
        _benchmark(
            map_dir,
            simulation_mode="replay",
            num_maps=2,
            num_scenarios=3,
        ),
    )
    with pytest.raises(pufferlib.APIUsageError, match="requests 3 scenarios"):
        drive_benchmark.load_benchmark_config(config_path, "carla_tiny")


def _checkpoint_fixture(tmp_path):
    run_dir = tmp_path / "run"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.pt"
    model_path.write_bytes(b"checkpoint")
    config_path = run_dir / "config.yaml"
    checkpoint_config = {
        "policy_name": "Drive",
        "rnn_name": None,
        "policy": {"backbone_hidden_size": 32},
        "rnn": {"hidden_size": 16},
        "env": {
            "num_goals": 3,
            "unknown_checkpoint_env_key": "ignored",
        },
    }
    return model_path, config_path, checkpoint_config


def test_checkpoint_architecture_merges_supported_configuration(tmp_path):
    """Evaluation rebuilds the checkpoint architecture without mutating CLI args."""
    model_path, config_path, checkpoint_config = _checkpoint_fixture(tmp_path)
    config_path.write_text(yaml.safe_dump(checkpoint_config))
    args = {
        "load_model_path": str(model_path),
        "policy_name": "RuntimePolicy",
        "rnn_name": "RuntimeRnn",
        "policy": {"actor_hidden_size": 8},
        "rnn": {"input_size": 8},
        "env": {"num_agents": 8},
        "train": {"use_rnn": True},
    }

    merged_args, resolved_config_path = drive_benchmark.load_checkpoint_architecture(args)

    assert resolved_config_path == str(config_path)
    assert merged_args["policy"] == {
        "actor_hidden_size": 8,
        "backbone_hidden_size": 32,
    }
    assert merged_args["rnn"] == {"input_size": 8, "hidden_size": 16}
    assert merged_args["env"] == {"num_agents": 8, "num_goals": 3}
    assert merged_args["policy_name"] == "Drive"
    assert merged_args["rnn_name"] is None
    assert merged_args["train"]["use_rnn"] is False
    assert args["policy"] == {"actor_hidden_size": 8}


@pytest.mark.parametrize("model_name", ["missing.pt", "model.bin"])
def test_checkpoint_architecture_rejects_invalid_model_path(tmp_path, model_name):
    """A benchmark evaluation requires an existing PyTorch checkpoint."""
    args = {"load_model_path": str(tmp_path / model_name)}

    with pytest.raises(pufferlib.APIUsageError, match="valid load_model_path"):
        drive_benchmark.load_checkpoint_architecture(args)


@pytest.mark.parametrize(
    ("config_contents", "error_match"),
    [
        (None, "Checkpoint config not found"),
        ("policy: [", "invalid YAML"),
        ("- not\n- a mapping\n", "must be a mapping"),
    ],
)
def test_checkpoint_architecture_rejects_invalid_config_file(tmp_path, config_contents, error_match):
    """Missing, malformed, or non-mapping checkpoint configs fail explicitly."""
    model_path, config_path, _ = _checkpoint_fixture(tmp_path)
    if config_contents is not None:
        config_path.write_text(config_contents)
    args = {"load_model_path": str(model_path)}

    with pytest.raises(pufferlib.APIUsageError, match=error_match):
        drive_benchmark.load_checkpoint_architecture(args)


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_match"),
    [
        ("policy", [], "config policy must be a mapping"),
        ("rnn", [], "config rnn must be a mapping"),
        ("env", [], "config env must be a mapping"),
    ],
)
def test_checkpoint_architecture_rejects_non_mapping_sections(
    tmp_path,
    field_name,
    invalid_value,
    error_match,
):
    """Architecture sections must be mappings before they can be merged."""
    model_path, config_path, checkpoint_config = _checkpoint_fixture(tmp_path)
    checkpoint_config[field_name] = invalid_value
    config_path.write_text(yaml.safe_dump(checkpoint_config))
    args = {
        "load_model_path": str(model_path),
        "policy": {},
        "rnn": {},
        "env": {},
        "train": {},
    }

    with pytest.raises(pufferlib.APIUsageError, match=error_match):
        drive_benchmark.load_checkpoint_architecture(args)


@pytest.mark.parametrize("missing_field", ["policy_name", "rnn_name"])
def test_checkpoint_architecture_requires_policy_and_rnn_names(tmp_path, missing_field):
    """Checkpoint metadata must identify both policy and recurrent wrappers."""
    model_path, config_path, checkpoint_config = _checkpoint_fixture(tmp_path)
    checkpoint_config.pop(missing_field)
    config_path.write_text(yaml.safe_dump(checkpoint_config))
    args = {
        "load_model_path": str(model_path),
        "policy": {},
        "rnn": {},
        "env": {},
        "train": {},
    }

    with pytest.raises(pufferlib.APIUsageError, match=f"missing {missing_field}"):
        drive_benchmark.load_checkpoint_architecture(args)


def test_eval_report_normalizes_rows_and_writes_csv_and_json(tmp_path):
    """Episode rows and aggregate means remain consistent across both reports."""
    episode_summaries = [
        {
            "summary_type": "evaluation_episode",
            "env_slot": 0,
            "map_name": "/maps/map_a.bin",
            "scenario_id": "scenario_a",
            "seed": 10,
            "collision_rate": 0.0,
            "score": 2.0,
        },
        {
            "summary_type": "evaluation_episode",
            "env_slot": 1,
            "map_name": "map_b.bin",
            "scenario_id": "scenario_b",
            "seed": 20,
            "collision_rate": 1.0,
            "score": 4.0,
        },
    ]

    report_frame, report_summary = pufferl._build_eval_report(episode_summaries, num_scenarios=3)

    assert report_frame.columns.tolist() == [
        "map_name",
        "scenario_id",
        "seed",
        "collision_rate",
        "score",
    ]
    assert report_frame["map_name"].tolist() == ["map_a", "map_b"]
    assert report_summary == {
        "num_scenarios": 3,
        "num_episodes": 2,
        "metrics_mean": {
            "collision_rate": 0.5,
            "score": 3.0,
        },
    }

    written_summary = pufferl._write_eval_reports(episode_summaries, tmp_path, num_scenarios=3)

    assert written_summary == report_summary
    written_frame = pd.read_csv(tmp_path / "episode_metrics.csv")
    pd.testing.assert_frame_equal(written_frame, report_frame)
    assert json.loads((tmp_path / "evaluation_summary.json").read_text()) == report_summary


def test_eval_report_skips_empty_episode_list(tmp_path):
    """An empty rollout produces no misleading empty report directory."""
    output_dir = tmp_path / "reports"

    assert pufferl._write_eval_reports([], output_dir, num_scenarios=1) is None
    assert not output_dir.exists()


def test_resolved_benchmark_config_records_sources_and_values(tmp_path):
    """Each evaluation records enough resolved configuration to be reproducible."""
    benchmark_config_path = tmp_path / "benchmark.yaml"
    checkpoint_config_path = tmp_path / "checkpoint.yaml"
    output_path = tmp_path / "resolved.yaml"
    args = {"train": {"seed": 7}, "env": {"num_agents": 8}}
    benchmark = {"name": "carla_tiny", "num_scenarios": 1}

    drive_benchmark.write_resolved_benchmark_config(
        args,
        benchmark,
        benchmark_config_path,
        checkpoint_config_path,
        output_path,
    )

    resolved = yaml.safe_load(output_path.read_text())
    assert resolved == {
        "benchmark_config": str(benchmark_config_path.resolve()),
        "checkpoint_config": str(checkpoint_config_path.resolve()),
        "benchmark": benchmark,
        "args": args,
    }


def test_benchmark_worker_plan_distributes_each_scenario_once():
    """Workers cover every scenario exactly once, including an uneven remainder."""
    args = {"env": {"num_agents": 8, "marker": []}}

    worker_args, total_steps = pufferl._plan_benchmark_eval_workers(
        args,
        num_scenarios=5,
        num_workers=4,
        scenario_length=32,
        capture_replay=True,
    )

    assert total_steps == 64
    assert [worker["num_eval_scenarios"] for worker in worker_args] == [2, 1, 1, 1]
    assert [worker["starting_map"] for worker in worker_args] == [0, 2, 3, 4]
    assert [worker["replay_worker_idx"] for worker in worker_args] == [0, 1, 2, 3]
    assert all(worker["capture_replay"] for worker in worker_args)
    assert all(worker["resample_frequency"] == 32 for worker in worker_args)
    assert all(worker["eval_mode"] == 1 for worker in worker_args)


def test_resolve_map_indices_uses_sorted_map_stems(tmp_path):
    """Failure CSV map names resolve deterministically against sorted map files."""
    map_dir = _make_map_dir(tmp_path)

    assert pufferl._resolve_map_indices(map_dir, ["/logged/map_1.bin", "map_0"]) == [1, 0]
    with pytest.raises(pufferlib.APIUsageError, match="Replay map 'missing' not found"):
        pufferl._resolve_map_indices(map_dir, ["missing.bin"])


class _PoolingPolicy:
    def pool_slot_counts(self, observations):
        assert observations.shape == (2, 3)
        return {
            "pool_partner": torch.tensor([[1], [2]]),
            "pool_lane": torch.tensor([[3], [4]]),
            "pool_boundary": torch.tensor([[5], [6]]),
            "pool_traffic": torch.tensor([[7], [8]]),
        }


def test_eval_replay_capture_records_continuous_policy_and_pooling(tmp_path):
    """Continuous replays retain policy statistics and observation-pooling counts."""
    replay_capture = EvalReplayCapture(
        args={
            "env": {"action_type": "continuous"},
            "eval": {"observation_replay_writer_count": 2},
        },
        policy=_PoolingPolicy(),
        replay_output_dir=tmp_path,
        capture_observations=True,
        num_workers=2,
        agents_per_batch=2,
        capture_batch_steps=1,
        episode_id_offset=0,
    )
    observations = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    policy_observations = torch.as_tensor(observations)
    policy_distribution = torch.distributions.Normal(
        loc=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        scale=torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
    )

    replay_capture.capture_frame(
        obs=observations,
        policy_obs_tensor=policy_observations,
        raw_action=np.asarray([[0.2, -0.3], [0.4, -0.5]], dtype=np.float32),
        action=np.asarray([[0.2, -0.25], [0.25, -0.25]], dtype=np.float32),
        logits=policy_distribution,
        value=torch.tensor([[1.0], [2.0]]),
        logprob=torch.tensor([-0.5, -0.75]),
        entropy=torch.tensor([0.25, 0.5]),
    )

    history = replay_capture.policy_history
    assert "policy_probs" not in history
    np.testing.assert_array_equal(history["obs"][0], observations.astype(np.float16))
    np.testing.assert_array_equal(history["policy_mean"][0], policy_distribution.loc.numpy())
    np.testing.assert_array_equal(history["policy_std"][0], policy_distribution.scale.numpy())
    np.testing.assert_array_equal(history["policy_log_prob"][0], [-0.5, -0.75])
    for pool_name, expected_values in {
        "pool_partner": [[1], [2]],
        "pool_lane": [[3], [4]],
        "pool_boundary": [[5], [6]],
        "pool_traffic": [[7], [8]],
    }.items():
        assert history[pool_name].dtype == np.int16
        np.testing.assert_array_equal(history[pool_name][0], expected_values)


@pytest.mark.parametrize("invalid_writer_count", [0, -1, True, 1.5])
def test_eval_replay_capture_rejects_invalid_observation_writer_count(tmp_path, invalid_writer_count):
    """Observation replay writer counts must be positive integers."""
    with pytest.raises(pufferlib.APIUsageError, match="observation_replay_writer_count"):
        EvalReplayCapture(
            args={
                "env": {},
                "eval": {"observation_replay_writer_count": invalid_writer_count},
            },
            policy=SimpleNamespace(),
            replay_output_dir=tmp_path,
            capture_observations=True,
            num_workers=1,
            agents_per_batch=1,
            capture_batch_steps=1,
            episode_id_offset=0,
        )


def _seed_rngs():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)


def _draw_rng_values():
    return random.random(), np.random.random(), torch.rand(3)


@pytest.mark.parametrize("initial_training_mode", [False, True])
@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_training_evaluation_isolated_from_training_state(
    monkeypatch,
    tmp_path,
    initial_training_mode,
    evaluation_fails,
):
    """Evaluation restores RNG and policy mode after both success and failure."""
    policy = torch.nn.Linear(1, 1)
    policy.train(initial_training_mode)
    logged_calls = []
    benchmark_results = {
        "carla_tiny": {
            "episodes": [{}],
            "summary": {
                "num_scenarios": 1,
                "num_episodes": 1,
                "metrics_mean": {"score": 0.75},
            },
        }
    }

    def fake_eval(**kwargs):
        kwargs["policy"].eval()
        _draw_rng_values()
        if evaluation_fails:
            raise RuntimeError("evaluation failed")
        return benchmark_results

    monkeypatch.setattr(pufferl, "eval", fake_eval)
    args = {
        "train": {"evaluation_benchmarks": "carla_tiny"},
        "eval": {
            "render_scenarios": True,
            "render_filter": "collision_rate",
            "failure_replay_csv": "failures.csv",
        },
    }
    logger = SimpleNamespace(log=lambda metrics, step: logged_calls.append((metrics, step)))

    _seed_rngs()
    expected_python, expected_numpy, expected_torch = _draw_rng_values()
    _seed_rngs()

    result = pufferl.run_training_evaluation(
        env_name="puffer_drive",
        args=args,
        policy=policy,
        logger=logger,
        epoch=2,
        global_step=64,
        run_dir=str(tmp_path),
    )
    actual_python, actual_numpy, actual_torch = _draw_rng_values()

    assert policy.training is initial_training_mode
    assert actual_python == expected_python
    assert actual_numpy == expected_numpy
    assert torch.equal(actual_torch, expected_torch)
    if evaluation_fails:
        assert result == {}
        assert logged_calls == []
    else:
        assert result == benchmark_results
        assert logged_calls == [
            (
                {
                    "eval_carla_tiny/num_scenarios": 1,
                    "eval_carla_tiny/num_episodes": 1,
                    "eval_carla_tiny/score": 0.75,
                },
                64,
            )
        ]
