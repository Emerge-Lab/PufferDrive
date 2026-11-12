import json
import struct
import sys
import types
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
from pufferlib.config_schema import validate_puffer_drive_config
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark
from pufferlib.ocean.evaluation_utils import eval_replay as drive_eval_replay


SEED = 42
CARLA_SCENARIO_COUNT = 7
CARLA_WORKER_COUNT = 2
CARLA_MAP_COUNT = 4
CARLA_SCENARIO_LENGTH = 32
BENCHMARK_SCENARIO_LENGTH = 48
WANDB_RUN_NAME = "k_exp_0002_1000"
WANDB_PROJECT = "nightly-multi-long"
WANDB_TRAINING_SUMMARY = {"environment/collision_rate": 0.0123}
WANDB_TRAINING_LAST_STEP = 1000190509056
REPO_ROOT = Path(__file__).resolve().parents[2]
CARLA_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/carla"
REPLAY_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/sdc_replay_test"


class ZeroPolicy(torch.nn.Module):
    """Discrete jerk-grid policy with uniform logits: mode selection always takes class 0."""

    is_continuous = False

    def __init__(self):
        super().__init__()
        jerk_long = torch.tensor(binding.JERK_LONG)
        jerk_lat = torch.tensor(binding.JERK_LAT)
        long_norm = torch.where(jerk_long < 0.0, jerk_long / -jerk_long[0], jerk_long / jerk_long[-1])
        lat_norm = jerk_lat / jerk_lat[-1]
        class_idx = torch.arange(len(jerk_long) * len(jerk_lat))
        self.action_table = torch.stack(
            [long_norm[class_idx // len(jerk_lat)], lat_norm[class_idx % len(jerk_lat)]], -1
        )

    def forward_eval(self, observations):
        batch_size = observations.shape[0]
        logits = torch.zeros((batch_size, len(self.action_table)), device=observations.device)
        values = torch.zeros((batch_size, 1), device=observations.device)
        return logits, values

    def discrete_actions_to_continuous(self, actions):
        return self.action_table[actions.long()]


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
                    "termination_mode": False,
                    "obs_dropout_lane": 0.0,
                    "obs_dropout_boundary": 0.0,
                    "speed_limit_random_prob": 0.0,
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


def _write_checkpoint_run_dir(run_dir, args):
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(args))))
    (run_dir / "final_model.pt").write_bytes(b"")
    args["load_model_path"] = str(run_dir / "final_model.pt")


def _standalone_eval_args(benchmark_config_path):
    args = _load_config()
    args["train"].update(
        {
            "seed": SEED,
            "device": "cpu",
            "compile": False,
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
            "use_map_cache": True,
            "scenario_length": CARLA_SCENARIO_LENGTH,
            "resample_frequency": CARLA_SCENARIO_LENGTH,
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
            "action_selection": "mode",
        }
    )
    args["wandb"] = False
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
        use_neighbor_cache=True,
    )
    args = _standalone_eval_args(benchmark_config_path)
    _write_checkpoint_run_dir(output_root / "checkpoint", args)
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
            policy=ZeroPolicy(),
            eval_output_dir=str(output_root),
            eval_output_subdir="standard",
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


def _run_standalone_eval_with_wandb(tmp_path, monkeypatch, benchmark_name):
    """Evaluate a checkpoint folder with wandb=True against a stub wandb
    -> (run dir, stored summary, stored history rows, requested run paths)."""
    run_dir = tmp_path / WANDB_RUN_NAME
    benchmark_config_path = _write_benchmark_config(
        tmp_path / "benchmark.yaml",
        name=benchmark_name,
        map_dir=CARLA_MAP_DIR,
        simulation_mode="gigaflow",
        num_maps=CARLA_MAP_COUNT,
        num_scenarios=CARLA_SCENARIO_COUNT,
        scenario_length=CARLA_SCENARIO_LENGTH,
        max_agents_per_env=8,
        control_mode="control_vehicles",
        use_neighbor_cache=True,
    )
    args = _standalone_eval_args(benchmark_config_path)
    args["eval"]["benchmarks"] = benchmark_name
    args["run_name"] = WANDB_RUN_NAME
    args["wandb_project"] = WANDB_PROJECT
    _write_checkpoint_run_dir(run_dir, args)
    args["wandb"] = True

    stored_summary = dict(WANDB_TRAINING_SUMMARY)
    stored_history = [{**WANDB_TRAINING_SUMMARY, "_step": WANDB_TRAINING_LAST_STEP}]
    requested_run_paths = []

    def fetch_run(run_path):
        requested_run_paths.append(run_path)
        logged_keys = {key for row in stored_history for key in row if not key.startswith("_")}
        return types.SimpleNamespace(
            state="finished",
            entity="emerge_",
            summary_metrics=dict(stored_summary),
            history_keys={"keys": dict.fromkeys(logged_keys, {}), "lastStep": stored_history[-1]["_step"]},
        )

    def log_history_row(values):
        stored_history.append({**values, "_step": stored_history[-1]["_step"] + 1})
        stored_summary.update(values)

    def open_session(**init_kwargs):
        return types.SimpleNamespace(log=log_history_row, summary=stored_summary, finish=lambda: None)

    stub_wandb = types.ModuleType("wandb")
    stub_wandb.Api = lambda: types.SimpleNamespace(run=fetch_run, flush=lambda: None)
    stub_wandb.Settings = lambda **settings: settings
    stub_wandb.init = open_session
    monkeypatch.setitem(sys.modules, "wandb", stub_wandb)

    pufferl.eval(env_name="puffer_drive", args=args, policy=ZeroPolicy())
    return run_dir, stored_summary, stored_history, requested_run_paths


def test_standalone_selfplay_eval_reports_its_km_per_infraction_to_wandb(tmp_path, monkeypatch):
    run_dir, stored_summary, stored_history, requested_run_paths = _run_standalone_eval_with_wandb(
        tmp_path, monkeypatch, "carla"
    )

    summary_path = next(run_dir.glob("eval/carla/*/evaluation_summary.json"))
    distance_per_infraction_meters = json.loads(summary_path.read_text())["metrics_mean"]["avg_distance_per_infraction"]
    km_per_infraction = distance_per_infraction_meters / 1000.0
    assert set(requested_run_paths) == {f"{WANDB_PROJECT}/{WANDB_RUN_NAME}"}
    assert stored_summary == {
        **WANDB_TRAINING_SUMMARY,
        "final_eval/selfplay_km_per_infraction": km_per_infraction,
        "final_eval/selfplay_km_per_infraction_source": str(summary_path.relative_to(run_dir)),
    }
    # The charts plot history rows: the value is one, a step after training's last.
    assert stored_history[1:] == [
        {"final_eval/selfplay_km_per_infraction": km_per_infraction, "_step": WANDB_TRAINING_LAST_STEP + 1}
    ]


def test_standalone_eval_of_another_benchmark_reports_nothing_to_wandb(tmp_path, monkeypatch):
    _, stored_summary, stored_history, requested_run_paths = _run_standalone_eval_with_wandb(
        tmp_path, monkeypatch, "carla_fast"
    )

    assert requested_run_paths == []
    assert stored_summary == WANDB_TRAINING_SUMMARY
    assert len(stored_history) == 1


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
        policy=ZeroPolicy(),
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
            "simulation_mode": "replay",
            "control_mode": "control_sdc_only",
            "sdc_controller": "replay",
            "non_sdc_controller": "replay",
            "scenario_length": 64,
            "resample_frequency": 64,
            "termination_mode": False,
            "goal_source": "gt",
            "num_goals": 3,
            "compute_eval_metrics": True,
            "speed_limit_random_prob": 0.0,
        }
    )
    _set_small_observation_config(args)
    args["eval"]["observation_replay_writer_count"] = 1
    args["eval"]["action_selection"] = "mode"
    return args


def _read_replay_header(replay_path):
    payload = zlib.decompress(replay_path.read_bytes())
    header_length = struct.unpack_from("<I", payload)[0]
    return json.loads(payload[4 : 4 + header_length])


def _read_replay_float32_chunk(replay_path, chunk_name):
    payload = zlib.decompress(replay_path.read_bytes())
    header_length = struct.unpack_from("<I", payload)[0]
    header = json.loads(payload[4 : 4 + header_length])
    chunk = header["chunks"][chunk_name]
    assert chunk["dtype"] == "float32"
    data_start = 4 + header_length + (-(4 + header_length) % 4)
    values = np.frombuffer(
        payload,
        dtype=np.float32,
        count=chunk["nbytes"] // np.dtype(np.float32).itemsize,
        offset=data_start + chunk["offset"],
    )
    return values.reshape(chunk["shape"])


@pytest.mark.parametrize("capture_observations", [False, True], ids=["without_observations", "with_observations"])
def test_multiprocess_replay_capture_renders_zlib_to_html(tmp_path, monkeypatch, capture_observations):
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
    replay_output_dir = tmp_path / drive_eval_replay.ZLIB_REPLAY_DIR_NAME
    summaries = pufferl._run_eval_rollout(
        args,
        "puffer_drive",
        worker_env_kwargs,
        total_steps,
        "Replay capture",
        expected_episodes=2,
        policy=ZeroPolicy(),
        replay_output_dir=replay_output_dir,
        capture_observations=capture_observations,
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
        "goals_f32",
        "rewards_f32",
        "coefs_f32",
        "raw_action",
        "policy_probs",
    }
    if capture_observations:
        required_chunks.add("obs")
    for replay_path in replay_paths:
        header = _read_replay_header(replay_path)
        assert header["frames"] == 64
        assert header["active_count"] == 1
        assert (header["obs_dim"] > 0) is capture_observations
        assert required_chunks <= set(header["chunks"])
        assert header["agent_goal_radius_field"] == 12
        assert header["chunks"]["agent_f32"]["shape"][2] == 13
        assert header["chunks"]["rewards_f32"]["shape"][2] == binding.REWARD_F32_FIELDS
        agent_frames = _read_replay_float32_chunk(replay_path, "agent_f32")
        assert np.any(agent_frames[..., header["agent_goal_radius_field"]] > 0.0)
        rewards = _read_replay_float32_chunk(replay_path, "rewards_f32")
        assert np.any(rewards[..., 0] != 0.0)
        np.testing.assert_allclose(rewards[..., 0], rewards[..., 1:].sum(axis=-1), atol=1e-4)

        coefs = _read_replay_float32_chunk(replay_path, "coefs_f32")
        assert header["chunks"]["coefs_f32"]["shape"][2] == binding.NUM_REWARD_COEFS
        goal_radius_coefs = coefs[..., 0]
        agent_goal_radii = agent_frames[..., header["agent_goal_radius_field"]]
        np.testing.assert_allclose(goal_radius_coefs, agent_goal_radii, atol=1e-6)

    render_dir = Path(drive_eval_replay._render_eval_replays(summaries, str(tmp_path), keep_zlib_replays=True))
    rendered_pages = sorted(path for path in render_dir.glob("*.html") if path.name != "index.html")
    assert len(rendered_pages) == 2
    assert (render_dir / "index.html").is_file()
    rendered_html = [page.read_text() for page in rendered_pages]
    assert all('class="payload-chunk"' in html for html in rendered_html)
    assert all('id="reward-grid"' in html and '"return (cum)"' in html for html in rendered_html)
    assert all("ctx.arc(g.x,g.y,g.radius" in html for html in rendered_html)
    assert all(replay_path.is_file() for replay_path in replay_paths)

    drive_eval_replay._render_eval_replays(summaries, str(tmp_path), keep_zlib_replays=False)
    assert not replay_output_dir.exists()
    assert (render_dir / "index.html").is_file()
    assert len(sorted(path for path in render_dir.glob("*.html") if path.name != "index.html")) == 2


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
            "use_map_cache": True,
            "simulation_mode": "replay",
            "control_mode": "control_sdc_only",
            "scenario_length": SDC_SCENARIO_LENGTH,
            "resample_frequency": SDC_SCENARIO_LENGTH,
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
            "action_selection": "mode",
        }
    )
    args["wandb"] = False
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
        use_neighbor_cache=True,
        max_scenarios_per_batch=max_scenarios_per_batch,
    )
    args = _sdc_eval_args(benchmark_config_path, benchmark_name, map_dir)
    _write_checkpoint_run_dir(output_root / f"{benchmark_name}_checkpoint", args)
    pufferl.eval(
        env_name="puffer_drive",
        args=args,
        policy=ZeroPolicy(),
        eval_output_dir=str(output_root),
        eval_output_subdir=benchmark_name,
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


def test_batch_cap_rejects_non_positive_values():
    args = _load_config()
    args["env"]["max_scenarios_per_batch"] = 0
    with pytest.raises(pufferlib.APIUsageError, match="max_scenarios_per_batch"):
        validate_puffer_drive_config(args, "test")


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
