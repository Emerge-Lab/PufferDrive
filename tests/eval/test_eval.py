import json
import random
import struct
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import pufferlib
from pufferlib import pufferl
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark
from pufferlib.ocean.evaluation_utils import eval_replay as drive_eval_replay


SEED = 42
CARLA_SCENARIO_COUNT = 7
CARLA_WORKER_COUNT = 2
CARLA_MAP_COUNT = 4
CARLA_SCENARIO_LENGTH = 32
TRAIN_EPOCH_COUNT = 7
TRAIN_EVAL_INTERVAL = 3
TRAIN_HORIZON = 16
TRAIN_ENV_COUNT = 2
TRAIN_AGENTS_PER_ENV = 8
REPO_ROOT = Path(__file__).resolve().parents[2]
CARLA_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/carla"
REPLAY_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/sdc_replay_test"


class ZeroPolicy(torch.nn.Module):
    def __init__(self, action_count):
        super().__init__()
        self.action_count = action_count

    def forward_eval(self, observations):
        batch_size = observations.shape[0]
        logits = torch.zeros((batch_size, self.action_count), device=observations.device)
        values = torch.zeros((batch_size, 1), device=observations.device)
        return (logits,), values


class RecordingLogger:
    def __init__(self, run_id):
        self.run_id = run_id
        self.calls = []

    def log(self, metrics, step):
        self.calls.append((metrics, step))

    def close(self, model_path, early_stop):
        self.model_path = model_path
        self.early_stop = early_stop


def _load_config():
    with patch.object(sys, "argv", ["pufferl.py"]):
        return pufferl.load_config("puffer_drive")


def _set_small_observation_config(args):
    args["env"].update(
        {
            "obs_slots_partners_n": 2,
            "obs_slots_lane_n": 8,
            "obs_slots_boundary_n": 8,
            "obs_slots_traffic_controls_n": 1,
            "obs_dropout_lane": 0.0,
            "obs_dropout_boundary": 0.0,
        }
    )


def _write_benchmark_config(
    output_path,
    name,
    map_dir,
    simulation_mode,
    num_maps,
    num_scenarios,
    scenario_length,
    max_agents_per_env,
    control_mode,
    use_neighbor_cache,
    max_scenarios_per_batch=None,
):
    output_path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "eval_mode": 1,
                    "compute_eval_metrics": True,
                    "termination_mode": 0,
                    "obs_dropout_lane": 0.0,
                    "obs_dropout_boundary": 0.0,
                },
                "benchmarks": [
                    {
                        "name": name,
                        "seed": SEED,
                        "num_scenarios": num_scenarios,
                        "env": {
                            "simulation_mode": simulation_mode,
                            "map_dir": str(map_dir),
                            "num_maps": num_maps,
                            "scenario_length": scenario_length,
                            "max_agents_per_env": max_agents_per_env,
                            "control_mode": control_mode,
                            "use_neighbor_cache": use_neighbor_cache,
                            "max_scenarios_per_batch": max_scenarios_per_batch,
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )
    return output_path


def _standalone_eval_args(benchmark_config_path):
    args = _load_config()
    args["train"].update(
        {
            "seed": SEED,
            "device": "cpu",
            "compile": False,
            "amp": False,
            "torch_deterministic": True,
        }
    )
    args["vec"].update(
        {
            "seed": SEED,
            "num_envs": CARLA_WORKER_COUNT,
            "num_workers": CARLA_WORKER_COUNT,
        }
    )
    args["env"].update(
        {
            "num_agents": 8,
            "min_agents_per_env": 8,
            "max_agents_per_env": 8,
            "num_maps": CARLA_MAP_COUNT,
            "map_dir": str(CARLA_MAP_DIR),
            "use_map_cache": 1,
            "scenario_length": CARLA_SCENARIO_LENGTH,
            "resample_frequency": CARLA_SCENARIO_LENGTH,
            "action_type": "discrete",
            "dynamics_model": "classic",
        }
    )
    _set_small_observation_config(args)
    args["eval"].update(
        {
            "benchmark_config": str(benchmark_config_path),
            "benchmarks": "carla_test",
            "num_agents": 8,
            "render_scenarios": False,
            "render_filter": None,
            "failure_replay_csv": None,
            "capture_observations": False,
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["tb"] = False
    return args


@pytest.fixture(scope="module")
def carla_evaluation(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("carla_evaluation")
    benchmark_config_path = _write_benchmark_config(
        output_root / "benchmark.yaml",
        name="carla_test",
        map_dir=CARLA_MAP_DIR,
        simulation_mode="gigaflow",
        num_maps=CARLA_MAP_COUNT,
        num_scenarios=CARLA_SCENARIO_COUNT,
        scenario_length=CARLA_SCENARIO_LENGTH,
        max_agents_per_env=8,
        control_mode="control_vehicles",
        use_neighbor_cache=1,
    )
    args = _standalone_eval_args(benchmark_config_path)
    multiprocessing_calls = []
    original_vector_make = pufferlib.vector.make

    def record_vector_make(*make_args, **make_kwargs):
        if make_kwargs.get("backend") == "Multiprocessing":
            multiprocessing_calls.append(dict(make_kwargs))
        return original_vector_make(*make_args, **make_kwargs)

    pufferlib.vector.make = record_vector_make
    try:
        benchmark_results = pufferl.eval(
            env_name="puffer_drive",
            args=args,
            policy=ZeroPolicy(action_count=63),
            eval_output_dir=str(output_root),
            eval_output_subdir="standard",
            use_training_config=True,
            benchmark_names="carla_test",
        )
    finally:
        pufferlib.vector.make = original_vector_make

    benchmark_output_dir = output_root / "carla_test" / "standard"
    return {
        "args": args,
        "benchmark_config_path": benchmark_config_path,
        "benchmark_results": benchmark_results,
        "benchmark_output_dir": benchmark_output_dir,
        "multiprocessing_calls": multiprocessing_calls,
        "output_root": output_root,
    }


def test_multiprocess_eval_dispatches_all_scenarios_and_maps(carla_evaluation):
    result = carla_evaluation["benchmark_results"]["carla_test"]
    metrics = pd.read_csv(carla_evaluation["benchmark_output_dir"] / "episode_metrics.csv")
    summary = json.loads((carla_evaluation["benchmark_output_dir"] / "evaluation_summary.json").read_text())

    assert len(carla_evaluation["multiprocessing_calls"]) == 1
    multiprocessing_call = carla_evaluation["multiprocessing_calls"][0]
    assert multiprocessing_call["num_envs"] == CARLA_WORKER_COUNT
    assert multiprocessing_call["num_workers"] == CARLA_WORKER_COUNT
    assert multiprocessing_call["batch_size"] == CARLA_WORKER_COUNT

    assert len(result["episodes"]) == CARLA_SCENARIO_COUNT
    assert len(metrics) == CARLA_SCENARIO_COUNT
    assert summary["num_scenarios"] == CARLA_SCENARIO_COUNT
    assert summary["num_episodes"] == CARLA_SCENARIO_COUNT
    assert metrics["seed"].nunique() == CARLA_SCENARIO_COUNT
    assert set(metrics["map_name"]) == {
        "opendrive__Town01",
        "opendrive__Town02",
        "opendrive__Town03",
        "opendrive__Town04",
    }
    assert {
        "avg_distance_per_infraction",
        "total_distance_travelled_sum",
        "total_infraction_count",
    }.issubset(metrics.columns)
    expected_episode_average = metrics["total_distance_travelled_sum"] / metrics["total_infraction_count"].clip(
        lower=1.0
    )
    np.testing.assert_allclose(metrics["avg_distance_per_infraction"], expected_episode_average)

    metric_summary = summary["metrics_mean"]
    expected_global_average = metrics["total_distance_travelled_sum"].sum() / max(
        metrics["total_infraction_count"].sum(), 1.0
    )
    assert metric_summary["total_distance_travelled"] == pytest.approx(metrics["total_distance_travelled_sum"].sum())
    assert metric_summary["total_infractions"] == pytest.approx(metrics["total_infraction_count"].sum())
    assert metric_summary["avg_distance_per_infraction"] == pytest.approx(expected_global_average)


def test_eval_report_reduces_distance_per_infraction_from_raw_totals():
    episode_summaries = [
        {
            "map_name": "town01.bin",
            "seed": 1,
            "collision_rate": 0.0,
            "avg_distance_per_infraction": 100.0,
            "total_distance_travelled_sum": 100.0,
            "total_infraction_count": 1.0,
        },
        {
            "map_name": "town02.bin",
            "seed": 2,
            "collision_rate": 1.0,
            "avg_distance_per_infraction": 10.0,
            "total_distance_travelled_sum": 100.0,
            "total_infraction_count": 10.0,
        },
    ]

    metrics, summary = drive_benchmark._build_eval_report(episode_summaries, num_scenarios=2)

    assert metrics["avg_distance_per_infraction"].tolist() == [100.0, 10.0]
    assert summary["metrics_mean"]["collision_rate"] == 0.5
    assert summary["metrics_mean"]["total_distance_travelled"] == 200.0
    assert summary["metrics_mean"]["total_infractions"] == 11.0
    assert summary["metrics_mean"]["avg_distance_per_infraction"] == pytest.approx(200.0 / 11.0)


def test_seed_replay_writes_exactly_identical_metrics(carla_evaluation):
    args = carla_evaluation["args"]
    benchmark_config_path = carla_evaluation["benchmark_config_path"]
    standard_metrics = pd.read_csv(carla_evaluation["benchmark_output_dir"] / "episode_metrics.csv")
    environment_config, benchmarks = drive_benchmark.load_benchmark_config(
        benchmark_config_path,
        "carla_test",
    )
    environment_config["obs_dropout_lane"] = args["env"]["obs_dropout_lane"]
    environment_config["obs_dropout_boundary"] = args["env"]["obs_dropout_boundary"]
    replay_args = drive_benchmark.build_benchmark_args(args, benchmarks[0], environment_config)
    replay_args["env"]["num_agents"] = replay_args["eval"]["num_agents"]

    map_indices = drive_benchmark._resolve_map_indices(
        replay_args["env"]["map_dir"],
        standard_metrics["map_name"].tolist(),
    )
    map_seed_pairs = [(map_idx, int(seed)) for map_idx, seed in zip(map_indices, standard_metrics["seed"].tolist())]
    worker_env_kwargs, total_steps = drive_benchmark._plan_failure_replay_workers(
        replay_args,
        map_seed_pairs,
        CARLA_WORKER_COUNT,
        CARLA_SCENARIO_LENGTH,
    )
    for worker_kwargs in worker_env_kwargs:
        worker_kwargs["capture_replay"] = False

    replay_summaries = pufferl._run_eval_rollout(
        replay_args,
        "puffer_drive",
        worker_env_kwargs,
        total_steps,
        "Seed replay",
        CARLA_SCENARIO_COUNT,
        policy=ZeroPolicy(action_count=63),
    )
    replay_output_dir = carla_evaluation["output_root"] / "seed_replay"
    replay_summary = drive_benchmark._write_eval_reports(
        replay_summaries,
        replay_output_dir,
        CARLA_SCENARIO_COUNT,
    )
    replay_metrics = pd.read_csv(replay_output_dir / "episode_metrics.csv")

    assert replay_summary["num_episodes"] == CARLA_SCENARIO_COUNT
    sort_columns = ["map_name", "seed"]
    standard_metrics = standard_metrics.sort_values(sort_columns).reset_index(drop=True)
    replay_metrics = replay_metrics.sort_values(sort_columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        replay_metrics,
        standard_metrics,
        check_exact=True,
        check_dtype=True,
    )


def _replay_render_args():
    args = _load_config()
    args["train"].update(
        {
            "seed": SEED,
            "device": "cpu",
            "compile": False,
            "amp": False,
        }
    )
    args["vec"].update({"seed": SEED, "num_envs": 2, "num_workers": 2})
    args["env"].update(
        {
            "num_agents": 1,
            "min_agents_per_env": 1,
            "max_agents_per_env": 1,
            "num_maps": 1,
            "map_dir": str(REPLAY_MAP_DIR),
            "action_type": "discrete",
            "dynamics_model": "jerk",
            "simulation_mode": "replay",
            "control_mode": "control_sdc_only",
            "sdc_controller": "replay",
            "non_sdc_controller": "replay",
            "scenario_length": 64,
            "resample_frequency": 64,
            "termination_mode": 0,
            "terminate_on_goal": False,
            "goal_source": "gt",
            "num_goals": 3,
            "compute_eval_metrics": True,
        }
    )
    _set_small_observation_config(args)
    args["eval"]["observation_replay_writer_count"] = 1
    return args


def _read_replay_header(replay_path):
    payload = zlib.decompress(replay_path.read_bytes())
    header_length = struct.unpack_from("<I", payload)[0]
    return json.loads(payload[4 : 4 + header_length])


def test_multiprocess_replay_capture_renders_zlib_to_html(tmp_path, monkeypatch):
    args = _replay_render_args()
    map_seed_pairs = [(0, 1234), (0, 5678)]
    worker_env_kwargs, total_steps = drive_benchmark._plan_failure_replay_workers(
        args,
        map_seed_pairs,
        num_workers=2,
        scenario_length=64,
    )
    multiprocessing_calls = []
    original_vector_make = pufferlib.vector.make

    def record_vector_make(*make_args, **make_kwargs):
        multiprocessing_calls.append(dict(make_kwargs))
        return original_vector_make(*make_args, **make_kwargs)

    monkeypatch.setattr(pufferlib.vector, "make", record_vector_make)
    replay_output_dir = tmp_path / "replays"
    summaries = pufferl._run_eval_rollout(
        args,
        "puffer_drive",
        worker_env_kwargs,
        total_steps,
        "Replay capture",
        expected_episodes=2,
        policy=ZeroPolicy(action_count=12),
        replay_output_dir=replay_output_dir,
        capture_observations=True,
    )

    assert len(multiprocessing_calls) == 1
    assert multiprocessing_calls[0]["backend"] == "Multiprocessing"
    assert multiprocessing_calls[0]["num_workers"] == 2
    assert len(summaries) == 2
    replay_paths = [Path(summary["replay_path"]) for summary in summaries]
    assert all(replay_path.is_file() for replay_path in replay_paths)

    required_chunks = {
        "agent_f32",
        "agent_i32",
        "metrics_f32",
        "traffic_i16",
        "obs",
        "raw_action",
        "policy_probs",
    }
    for replay_path in replay_paths:
        header = _read_replay_header(replay_path)
        assert header["frames"] == 64
        assert header["active_count"] == 1
        assert header["obs_dim"] > 0
        assert required_chunks <= set(header["chunks"])

    render_dir = Path(drive_eval_replay._render_eval_replays(summaries, str(tmp_path)))
    rendered_pages = sorted(path for path in render_dir.glob("*.html") if path.name != "index.html")
    assert len(rendered_pages) == 2
    assert (render_dir / "index.html").is_file()
    assert all('class="payload-chunk"' in page.read_text() for page in rendered_pages)


def _write_training_benchmark(tmp_path):
    return _write_benchmark_config(
        tmp_path / "training_benchmark.yaml",
        name="training_eval",
        map_dir=CARLA_MAP_DIR,
        simulation_mode="gigaflow",
        num_maps=2,
        num_scenarios=2,
        scenario_length=TRAIN_HORIZON,
        max_agents_per_env=TRAIN_AGENTS_PER_ENV,
        control_mode="control_vehicles",
        use_neighbor_cache=1,
    )


def _training_args(tmp_path, benchmark_config_path, evaluation_enabled):
    args = _load_config()
    total_agents = TRAIN_ENV_COUNT * TRAIN_AGENTS_PER_ENV
    batch_size = total_agents * TRAIN_HORIZON
    args["vec"].update(
        {
            "backend": "Serial",
            "num_envs": TRAIN_ENV_COUNT,
            "num_workers": TRAIN_ENV_COUNT,
            "batch_size": TRAIN_ENV_COUNT,
            "seed": SEED,
        }
    )
    args["env"].update(
        {
            "num_agents": TRAIN_AGENTS_PER_ENV,
            "min_agents_per_env": TRAIN_AGENTS_PER_ENV,
            "max_agents_per_env": TRAIN_AGENTS_PER_ENV,
            "num_maps": 2,
            "map_dir": str(CARLA_MAP_DIR),
            "use_map_cache": 1,
            "scenario_length": TRAIN_HORIZON,
            "resample_frequency": TRAIN_HORIZON,
        }
    )
    _set_small_observation_config(args)
    args["policy"].update(
        {
            "ego_input_size": 8,
            "partner_input_size": 8,
            "lane_input_size": 8,
            "boundary_input_size": 8,
            "traffic_control_input_size": 8,
            "context_input_size": 8,
            "backbone_hidden_size": 32,
            "backbone_num_layers": 1,
            "actor_hidden_size": 16,
            "actor_num_layers": 0,
            "critic_hidden_size": 16,
            "critic_num_layers": 0,
        }
    )
    args["train"].update(
        {
            "seed": SEED,
            "device": "cpu",
            "compile": False,
            "amp": False,
            "precision": "float32",
            "torch_deterministic": True,
            "anneal_lr": False,
            "update_epochs": 1,
            "batch_size": batch_size,
            "bptt_horizon": TRAIN_HORIZON,
            "minibatch_size": 64,
            "max_minibatch_size": 64,
            "total_timesteps": TRAIN_EPOCH_COUNT * batch_size,
            "checkpoint_interval": 10_000,
            "evaluation_interval_epochs": TRAIN_EVAL_INTERVAL if evaluation_enabled else None,
            "evaluation_benchmarks": "training_eval",
            "data_dir": str(tmp_path),
            "render": False,
        }
    )
    args["eval"].update(
        {
            "benchmark_config": str(benchmark_config_path),
            "benchmarks": "training_eval",
            "num_agents": TRAIN_AGENTS_PER_ENV,
            "render_scenarios": False,
            "render_filter": None,
            "failure_replay_csv": None,
            "capture_observations": False,
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["tb"] = False
    return args


def _seed_training():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _assert_nested_exact(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual, expected)
        return
    if isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_exact(actual[key], expected[key])
        return
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_exact(actual_item, expected_item)
        return
    assert actual == expected


def _run_training(tmp_path, benchmark_config_path, run_id, evaluation_enabled):
    _seed_training()
    logger = RecordingLogger(run_id)
    args = _training_args(tmp_path, benchmark_config_path, evaluation_enabled)
    # The run directory is train.data_dir, so each run needs its own; sharing one
    # would make the second run resume the first instead of training from scratch.
    run_dir = tmp_path / run_id
    args["train"]["data_dir"] = str(run_dir)
    args["run_name"] = run_id
    pufferl.train("puffer_drive", args=args, logger=logger)
    trainer_state = torch.load(run_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    return run_dir, trainer_state, logger


def test_mid_training_evaluation_preserves_exact_training_state(tmp_path):
    benchmark_config_path = _write_training_benchmark(tmp_path)
    previous_thread_count = torch.get_num_threads()
    previous_deterministic_setting = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        baseline_dir, baseline_state, _ = _run_training(
            tmp_path,
            benchmark_config_path,
            run_id="without_eval",
            evaluation_enabled=False,
        )
        evaluated_dir, evaluated_state, evaluated_logger = _run_training(
            tmp_path,
            benchmark_config_path,
            run_id="with_eval",
            evaluation_enabled=True,
        )
    finally:
        torch.set_num_threads(previous_thread_count)
        torch.use_deterministic_algorithms(previous_deterministic_setting, warn_only=True)

    assert baseline_dir.is_dir()
    for state_key in (
        "policy_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "global_step",
        "agent_step",
        "epoch",
        "update",
        "rng_state",
    ):
        _assert_nested_exact(evaluated_state[state_key], baseline_state[state_key])
    assert evaluated_state["epoch"] == TRAIN_EPOCH_COUNT

    steps_per_epoch = TRAIN_ENV_COUNT * TRAIN_AGENTS_PER_ENV * TRAIN_HORIZON
    expected_eval_directories = [f"epoch_{epoch:06d}_step_{epoch * steps_per_epoch}" for epoch in (3, 6, 7)]
    eval_root = evaluated_dir / "eval/training/training_eval"
    actual_eval_directories = sorted(path.name for path in eval_root.iterdir() if path.is_dir())
    assert actual_eval_directories == expected_eval_directories
    for eval_directory_name in expected_eval_directories:
        eval_directory = eval_root / eval_directory_name
        assert (eval_directory / "resolved_benchmark.yaml").is_file()
        metrics = pd.read_csv(eval_directory / "episode_metrics.csv")
        summary = json.loads((eval_directory / "evaluation_summary.json").read_text())
        assert len(metrics) == 2
        assert summary["num_scenarios"] == 2
        assert summary["num_episodes"] == 2

    eval_log_calls = [
        (metrics, step) for metrics, step in evaluated_logger.calls if "eval_training_eval/num_episodes" in metrics
    ]
    assert [step for _, step in eval_log_calls] == [
        3 * steps_per_epoch,
        6 * steps_per_epoch,
        7 * steps_per_epoch,
    ]
    assert all(metrics["eval_training_eval/num_episodes"] == 2 for metrics, _ in eval_log_calls)


SDC_SCENARIO_COUNT = 8
SDC_WORKER_COUNT = 2
SDC_SCENARIO_LENGTH = 32
SDC_MAX_AGENTS_PER_ENV = 64
SDC_BATCH_CAP = 2


@pytest.fixture(scope="module")
def sdc_replay_map_dir(tmp_path_factory):
    """A replay map dir with several distinctly named maps. The checked-in
    fixture ships one, and a batch cap only bites when maps outnumber it, so copy
    it under distinct names -- the name is what the map cache keys on."""
    map_dir = tmp_path_factory.mktemp("sdc_replay_maps")
    source_map = sorted(REPLAY_MAP_DIR.glob("*.bin"))[0]
    map_bytes = source_map.read_bytes()
    for scenario_idx in range(SDC_SCENARIO_COUNT):
        (map_dir / f"nuplan__batch{scenario_idx:02d}.bin").write_bytes(map_bytes)
    return map_dir


def _sdc_eval_args(benchmark_config_path, benchmark_name, map_dir):
    args = _load_config()
    args["train"].update(
        {
            "seed": SEED,
            "device": "cpu",
            "compile": False,
            "amp": False,
            "torch_deterministic": True,
        }
    )
    args["vec"].update(
        {
            "seed": SEED,
            "num_envs": SDC_WORKER_COUNT,
            "num_workers": SDC_WORKER_COUNT,
        }
    )
    args["env"].update(
        {
            "num_agents": SDC_SCENARIO_COUNT,
            "min_agents_per_env": 1,
            "max_agents_per_env": SDC_MAX_AGENTS_PER_ENV,
            "num_maps": SDC_SCENARIO_COUNT,
            "map_dir": str(map_dir),
            "use_map_cache": 1,
            "simulation_mode": "replay",
            "control_mode": "control_sdc_only",
            "scenario_length": SDC_SCENARIO_LENGTH,
            "resample_frequency": SDC_SCENARIO_LENGTH,
            "action_type": "discrete",
            "dynamics_model": "classic",
        }
    )
    _set_small_observation_config(args)
    args["eval"].update(
        {
            "benchmark_config": str(benchmark_config_path),
            "benchmarks": benchmark_name,
            "num_agents": SDC_SCENARIO_COUNT,
            "max_sdc_replay_workers": SDC_WORKER_COUNT,
            "render_scenarios": False,
            "render_filter": None,
            "failure_replay_csv": None,
            "capture_observations": False,
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["tb"] = False
    return args


def _run_sdc_eval(output_root, map_dir, benchmark_name, max_scenarios_per_batch):
    benchmark_config_path = _write_benchmark_config(
        output_root / f"{benchmark_name}.yaml",
        name=benchmark_name,
        map_dir=map_dir,
        simulation_mode="replay",
        num_maps=SDC_SCENARIO_COUNT,
        num_scenarios=SDC_SCENARIO_COUNT,
        scenario_length=SDC_SCENARIO_LENGTH,
        max_agents_per_env=None,
        control_mode="control_sdc_only",
        use_neighbor_cache=1,
        max_scenarios_per_batch=max_scenarios_per_batch,
    )
    args = _sdc_eval_args(benchmark_config_path, benchmark_name, map_dir)
    pufferl.eval(
        env_name="puffer_drive",
        args=args,
        policy=ZeroPolicy(action_count=63),
        eval_output_dir=str(output_root),
        eval_output_subdir=benchmark_name,
        use_training_config=True,
        benchmark_names=benchmark_name,
    )
    return pd.read_csv(output_root / benchmark_name / benchmark_name / "episode_metrics.csv")


def test_batch_cap_bounds_resident_envs(sdc_replay_map_dir):
    """The cap, not the num_agents budget, decides how many scenarios are live."""

    def build(max_scenarios_per_batch):
        return Drive(
            num_agents=SDC_SCENARIO_COUNT,
            num_maps=SDC_SCENARIO_COUNT,
            map_dir=str(sdc_replay_map_dir),
            simulation_mode="replay",
            control_mode="control_sdc_only",
            eval_mode=1,
            num_eval_scenarios=SDC_SCENARIO_COUNT,
            starting_map=0,
            scenario_length=SDC_SCENARIO_LENGTH,
            resample_frequency=SDC_SCENARIO_LENGTH,
            min_agents_per_env=1,
            max_agents_per_env=SDC_MAX_AGENTS_PER_ENV,
            max_scenarios_per_batch=max_scenarios_per_batch,
        )

    uncapped = build(None)
    try:
        assert uncapped.num_envs == SDC_SCENARIO_COUNT
    finally:
        uncapped.close()

    capped = build(SDC_BATCH_CAP)
    try:
        assert capped.num_envs == SDC_BATCH_CAP
    finally:
        capped.close()


def test_batch_cap_rejects_non_positive_values(sdc_replay_map_dir):
    with pytest.raises(ValueError, match="max_scenarios_per_batch"):
        Drive(
            num_agents=SDC_SCENARIO_COUNT,
            num_maps=SDC_SCENARIO_COUNT,
            map_dir=str(sdc_replay_map_dir),
            simulation_mode="replay",
            control_mode="control_sdc_only",
            eval_mode=1,
            num_eval_scenarios=SDC_SCENARIO_COUNT,
            min_agents_per_env=1,
            max_agents_per_env=SDC_MAX_AGENTS_PER_ENV,
            max_scenarios_per_batch=0,
        )


def test_batch_cap_preserves_scenario_coverage(tmp_path, sdc_replay_map_dir):
    """Smaller batches must still cover the window exactly once: nothing dropped
    or replayed at a batch boundary.

    Metrics are not compared against the uncapped run: binding.shared draws its
    seed from the same stream as the per-env seeds, once per batch, so more
    batches shift each scenario's seed. That predates the cap -- eval.num_agents
    or the worker count already move seeds -- so runs are reproducible for a
    fixed batch layout, not across layouts.
    """
    uncapped_metrics = _run_sdc_eval(tmp_path, sdc_replay_map_dir, "sdc_uncapped", None)
    capped_metrics = _run_sdc_eval(tmp_path, sdc_replay_map_dir, "sdc_capped", SDC_BATCH_CAP)

    assert len(uncapped_metrics) == SDC_SCENARIO_COUNT
    assert len(capped_metrics) == SDC_SCENARIO_COUNT
    assert sorted(capped_metrics["map_name"]) == sorted(uncapped_metrics["map_name"])
    assert capped_metrics["map_name"].nunique() == SDC_SCENARIO_COUNT
    assert capped_metrics["seed"].nunique() == SDC_SCENARIO_COUNT


def test_batch_cap_evaluation_is_reproducible(tmp_path, sdc_replay_map_dir):
    """A capped layout re-run at the same settings reproduces itself exactly."""
    first_metrics = _run_sdc_eval(tmp_path, sdc_replay_map_dir, "sdc_capped_first", SDC_BATCH_CAP)
    second_metrics = _run_sdc_eval(tmp_path, sdc_replay_map_dir, "sdc_capped_second", SDC_BATCH_CAP)
    pd.testing.assert_frame_equal(first_metrics, second_metrics, check_exact=True, check_dtype=True)


def test_policy_builds_from_the_released_driver_env(sdc_replay_map_dir):
    """load_policy constructs the policy from vecenv.driver_env, which the
    Multiprocessing backend releases once it has read the spaces off it. The
    policy only reads Python config attributes, which outlive close() -- but
    eval runs this path on every benchmark, so pin it."""
    with patch.object(sys, "argv", ["pufferl.py"]):
        args = pufferl.load_config("puffer_drive")
    args["train"]["device"] = "cpu"
    env_kwargs = {
        "num_agents": SDC_SCENARIO_COUNT,
        "num_maps": SDC_SCENARIO_COUNT,
        "map_dir": str(sdc_replay_map_dir),
        "simulation_mode": "replay",
        "control_mode": "control_sdc_only",
        "eval_mode": 1,
        "num_eval_scenarios": SDC_SCENARIO_COUNT // SDC_WORKER_COUNT,
        "scenario_length": SDC_SCENARIO_LENGTH,
        "resample_frequency": SDC_SCENARIO_LENGTH,
        "min_agents_per_env": 1,
        "max_agents_per_env": SDC_MAX_AGENTS_PER_ENV,
        "max_scenarios_per_batch": SDC_BATCH_CAP,
    }
    args["env"].update(env_kwargs)
    vecenv = pufferlib.vector.make(
        [Drive] * SDC_WORKER_COUNT,
        env_args=[[]] * SDC_WORKER_COUNT,
        env_kwargs=[env_kwargs] * SDC_WORKER_COUNT,
        backend="Multiprocessing",
        num_envs=SDC_WORKER_COUNT,
        num_workers=SDC_WORKER_COUNT,
        batch_size=SDC_WORKER_COUNT,
    )
    try:
        assert pufferl.load_policy(args, vecenv, "puffer_drive") is not None
        obs, _ = vecenv.reset(SEED)
        assert obs.shape[0] == vecenv.agents_per_batch
    finally:
        vecenv.close()
