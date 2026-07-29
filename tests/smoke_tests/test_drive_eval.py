#!/usr/bin/env python3
"""Deterministic CPU smoke test for checkpoint-backed PufferDrive evaluation.

The test creates a seeded, randomly initialized compact Drive checkpoint, runs
the public evaluation API on one short CARLA benchmark scenario, and checks both
the report artifacts and the complete metric summary against a committed golden.

The golden is reproducible only inside the pinned smoke image. Regenerate it
after an intentional evaluation change with:

    docker build -f tests/smoke_tests/Dockerfile -t pufferdrive-smoke .
    docker run --rm -e SMOKE_UPDATE_GOLDEN=1 \
        -v "$PWD/tests/smoke_tests/data:/app/tests/smoke_tests/data" \
        pufferdrive-smoke tests/smoke_tests/test_drive_eval.py
"""

import json
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

# Keep policy initialization and inference reproducible across CPU hosts. These
# must be set before numpy and torch load their numerical backends.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("ONEDNN_MAX_CPU_ISA", "AVX2")
os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX2")

import numpy as np
import pandas as pd
import torch
import yaml

import pufferlib
from pufferlib import pufferl


SEED = 42
AGENT_COUNT = 8
SCENARIO_COUNT = 1
SCENARIO_LENGTH = 32
BENCHMARK_NAME = "carla_smoke"
MAP_DIR = Path(pufferlib.__path__[0]) / "resources" / "drive" / "binaries" / "carla"
SOURCE_BENCHMARK_CONFIG_PATH = Path(pufferlib.__path__[0]) / "config" / "evaluation" / "benchmark.yaml"
GOLDEN_PATH = Path(__file__).parent / "data" / "drive_eval_golden.json"

RTOL = float(os.environ.get("SMOKE_RTOL", "2e-2"))
LOOSE_RTOL = float(os.environ.get("SMOKE_LOOSE_RTOL", "8e-2"))
ATOL = float(os.environ.get("SMOKE_ATOL", "1e-3"))
LOOSE_KEYS = frozenset({"episode_return"})


def _load_test_config(benchmark_config_path):
    with patch.object(sys, "argv", ["pufferl.py"]):
        args = pufferl.load_config("puffer_drive")

    args["vec"].update(
        {
            "backend": "Serial",
            "num_envs": 1,
            "num_workers": 1,
            "seed": SEED,
        }
    )
    args["env"].update(
        {
            "num_agents": AGENT_COUNT,
            "min_agents_per_env": AGENT_COUNT,
            "max_agents_per_env": AGENT_COUNT,
            "num_maps": 1,
            "map_dir": str(MAP_DIR),
            "use_map_cache": 1,
            "scenario_length": SCENARIO_LENGTH,
            "resample_frequency": SCENARIO_LENGTH,
            "action_type": "discrete",
            "termination_mode": 0,
            "obs_slots_partners_n": 2,
            "obs_slots_lane_n": 8,
            "obs_slots_boundary_n": 8,
            "obs_slots_traffic_controls_n": 1,
            "obs_dropout_lane": 0.0,
            "obs_dropout_boundary": 0.0,
        }
    )
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
            "torch_deterministic": True,
        }
    )
    args["eval"].update(
        {
            "benchmark_config": str(benchmark_config_path),
            "benchmarks": BENCHMARK_NAME,
            "num_agents": AGENT_COUNT,
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


def _write_tiny_benchmark(tmp_path):
    benchmark_config = yaml.safe_load(SOURCE_BENCHMARK_CONFIG_PATH.read_text())
    benchmark_config["benchmarks"] = [
        {
            "name": BENCHMARK_NAME,
            "seed": SEED,
            "simulation_mode": "gigaflow",
            "map_dir": str(MAP_DIR),
            "num_maps": 1,
            "num_scenarios": SCENARIO_COUNT,
            "scenario_length": SCENARIO_LENGTH,
            "max_agents_per_env": AGENT_COUNT,
            "control_mode": "control_vehicles",
        }
    ]
    benchmark_config_path = tmp_path / "benchmark.yaml"
    benchmark_config_path.write_text(yaml.safe_dump(benchmark_config, sort_keys=False))
    return benchmark_config_path


def _write_random_checkpoint(tmp_path, args):
    run_dir = tmp_path / "random_policy_run"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model_puffer_drive_random.pt"

    policy_env = pufferl.load_env("puffer_drive", args)
    try:
        policy = pufferl.load_policy(args, policy_env, "puffer_drive")
        torch.save(policy.state_dict(), model_path)
    finally:
        policy_env.close()

    checkpoint_config = {
        "policy_name": args["policy_name"],
        "rnn_name": args["rnn_name"],
        "policy": dict(args["policy"]),
        "rnn": dict(args["rnn"]),
        "env": dict(args["env"]),
    }
    checkpoint_config_path = run_dir / "config.yaml"
    checkpoint_config_path.write_text(yaml.safe_dump(checkpoint_config, sort_keys=False))
    return model_path, checkpoint_config_path


def _golden_metadata():
    return {
        "benchmark": BENCHMARK_NAME,
        "seed": SEED,
        "agent_count": AGENT_COUNT,
        "scenario_count": SCENARIO_COUNT,
        "scenario_length": SCENARIO_LENGTH,
    }


def _assert_matches_golden(summary):
    golden_record = {
        "meta": _golden_metadata(),
        "summary": summary,
    }
    if os.environ.get("SMOKE_UPDATE_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(golden_record, indent=2, sort_keys=True) + "\n")
        print(f"[eval] wrote golden -> {GOLDEN_PATH}")
        return

    assert GOLDEN_PATH.is_file(), (
        f"evaluation golden is missing: {GOLDEN_PATH}; regenerate it in the pinned smoke image"
    )
    expected_record = json.loads(GOLDEN_PATH.read_text())
    assert expected_record["meta"] == golden_record["meta"]

    expected_summary = expected_record["summary"]
    assert summary["num_scenarios"] == expected_summary["num_scenarios"]
    assert summary["num_episodes"] == expected_summary["num_episodes"]

    actual_metrics = summary["metrics_mean"]
    expected_metrics = expected_summary["metrics_mean"]
    assert set(actual_metrics) == set(expected_metrics), (
        "evaluation metric keys drifted:\n"
        f"  added: {sorted(set(actual_metrics) - set(expected_metrics))}\n"
        f"  removed: {sorted(set(expected_metrics) - set(actual_metrics))}"
    )

    mismatches = []
    for metric_name, expected_value in expected_metrics.items():
        actual_value = actual_metrics[metric_name]
        if np.isnan(actual_value) and np.isnan(expected_value):
            continue
        relative_tolerance = LOOSE_RTOL if metric_name in LOOSE_KEYS else RTOL
        if not np.isclose(actual_value, expected_value, rtol=relative_tolerance, atol=ATOL):
            mismatches.append(
                f"  {metric_name}: {actual_value!r} != expected {expected_value!r} "
                f"(rtol={relative_tolerance}, atol={ATOL})"
            )
    assert not mismatches, "evaluation metrics drifted from golden:\n" + "\n".join(mismatches)


def test_drive_eval(tmp_path):
    benchmark_config_path = _write_tiny_benchmark(tmp_path)
    args = _load_test_config(benchmark_config_path)
    previous_thread_count = torch.get_num_threads()
    previous_deterministic_setting = torch.are_deterministic_algorithms_enabled()
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True, warn_only=True)

        model_path, checkpoint_config_path = _write_random_checkpoint(tmp_path, args)
        args["load_model_path"] = str(model_path)
        eval_output_dir = tmp_path / "eval"
        eval_output_subdir = "smoke"
        benchmark_results = pufferl.eval(
            env_name="puffer_drive",
            args=args,
            eval_output_dir=str(eval_output_dir),
            eval_output_subdir=eval_output_subdir,
            use_training_config=False,
            benchmark_names=BENCHMARK_NAME,
        )
    finally:
        torch.set_num_threads(previous_thread_count)
        torch.use_deterministic_algorithms(previous_deterministic_setting, warn_only=True)

    assert set(benchmark_results) == {BENCHMARK_NAME}
    benchmark_result = benchmark_results[BENCHMARK_NAME]
    assert len(benchmark_result["episodes"]) == SCENARIO_COUNT
    summary = benchmark_result["summary"]
    assert summary["num_scenarios"] == SCENARIO_COUNT
    assert summary["num_episodes"] == SCENARIO_COUNT

    benchmark_output_dir = eval_output_dir / BENCHMARK_NAME / eval_output_subdir
    report_path = benchmark_output_dir / "evaluation_summary.json"
    metrics_path = benchmark_output_dir / "episode_metrics.csv"
    resolved_config_path = benchmark_output_dir / "resolved_benchmark.yaml"
    assert json.loads(report_path.read_text()) == summary

    metrics_rows = pd.read_csv(metrics_path)
    assert len(metrics_rows) == SCENARIO_COUNT
    assert {"map_name", "scenario_id", "seed", "agents_per_batch"} <= set(metrics_rows.columns)

    resolved_config = yaml.safe_load(resolved_config_path.read_text())
    assert resolved_config["benchmark_config"] == str(benchmark_config_path.resolve())
    assert resolved_config["checkpoint_config"] == str(checkpoint_config_path.resolve())
    assert resolved_config["benchmark"]["name"] == BENCHMARK_NAME
    assert resolved_config["args"]["load_model_path"] == str(model_path)
    assert resolved_config["args"]["env"]["num_agents"] == AGENT_COUNT
    assert resolved_config["args"]["env"]["scenario_length"] == SCENARIO_LENGTH
    assert resolved_config["args"]["env"]["compute_eval_metrics"] is True

    print("[eval] summary:", json.dumps(summary, indent=2, sort_keys=True))
    _assert_matches_golden(summary)
