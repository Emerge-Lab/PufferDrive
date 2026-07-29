"""Integration coverage for evaluation performed between training epochs."""

import random
import sys
from unittest.mock import patch

import numpy as np
import torch
import yaml

import pufferlib
from pufferlib import pufferl


SEED = 42
AGENT_COUNT = 8
BPTT_HORIZON = 32
TRAIN_DROPOUT_LANE = 0.5
TRAIN_DROPOUT_BOUNDARY = 0.25
MAP_DIR = pufferlib.__path__[0] + "/resources/drive/binaries/carla"


class _RecordingLogger:
    run_id = "training_evaluation_test"

    def __init__(self):
        self.calls = []

    def log(self, metrics, step):
        self.calls.append((metrics, step))

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


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
            "map_dir": MAP_DIR,
            "use_map_cache": 1,
            "scenario_length": BPTT_HORIZON,
            "resample_frequency": BPTT_HORIZON,
            "obs_slots_partners_n": 2,
            "obs_slots_lane_n": 8,
            "obs_slots_boundary_n": 8,
            "obs_slots_traffic_controls_n": 1,
            "obs_dropout_lane": TRAIN_DROPOUT_LANE,
            "obs_dropout_boundary": TRAIN_DROPOUT_BOUNDARY,
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
            "anneal_lr": False,
            "update_epochs": 1,
            "batch_size": AGENT_COUNT * BPTT_HORIZON,
            "bptt_horizon": BPTT_HORIZON,
            "minibatch_size": 64,
            "max_minibatch_size": 64,
            "total_timesteps": 10_000_000,
            "checkpoint_interval": 10_000_000,
            "evaluation_benchmarks": "carla_tiny",
            "render": False,
        }
    )
    args["eval"].update(
        {
            "benchmark_config": str(benchmark_config_path),
            "benchmarks": "carla_tiny",
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
    benchmark_config_path = tmp_path / "benchmark.yaml"
    benchmark_config_path.write_text(
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
                        "name": "carla_tiny",
                        "seed": SEED,
                        "simulation_mode": "gigaflow",
                        "map_dir": MAP_DIR,
                        "num_maps": 1,
                        "num_scenarios": 1,
                        "scenario_length": BPTT_HORIZON,
                        "max_agents_per_env": AGENT_COUNT,
                        "control_mode": "control_vehicles",
                    }
                ],
            }
        )
    )
    return benchmark_config_path


def _run_training_epoch(trainer):
    trainer.evaluate()
    trainer.train()


def test_training_evaluation_writes_step_scoped_report_without_disrupting_training(tmp_path):
    """A real eval writes under its training step and leaves PPO ready to continue."""
    benchmark_config_path = _write_tiny_benchmark(tmp_path)
    args = _load_test_config(benchmark_config_path)
    logger = _RecordingLogger()
    vecenv = None
    trainer = None
    previous_thread_count = torch.get_num_threads()
    previous_deterministic_setting = torch.are_deterministic_algorithms_enabled()
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.set_num_threads(1)

        vecenv = pufferl.load_env("puffer_drive", args)
        policy = pufferl.load_policy(args, vecenv, "puffer_drive")
        train_config = dict(**args["train"], env="puffer_drive", eval={})
        trainer = pufferl.PuffeRL(train_config, vecenv, policy, logger=logger)

        _run_training_epoch(trainer)
        first_epoch = trainer.epoch
        first_global_step = trainer.global_step
        assert trainer.uncompiled_policy.training

        benchmark_results = pufferl.run_training_evaluation(
            env_name="puffer_drive",
            args=args,
            policy=trainer.uncompiled_policy,
            logger=logger,
            epoch=first_epoch,
            global_step=first_global_step,
            run_dir=str(tmp_path),
        )

        benchmark_result = benchmark_results["carla_tiny"]
        assert len(benchmark_result["episodes"]) == 1
        assert benchmark_result["summary"]["num_scenarios"] == 1
        assert benchmark_result["summary"]["num_episodes"] == 1
        assert trainer.uncompiled_policy.training
        assert any("eval_carla_tiny/num_episodes" in metrics for metrics, _ in logger.calls)

        resolved_config_path = (
            tmp_path
            / "eval"
            / "training"
            / "carla_tiny"
            / f"epoch_{first_epoch:06d}_step_{first_global_step}"
            / "resolved_benchmark.yaml"
        )
        resolved_config = yaml.safe_load(resolved_config_path.read_text())
        assert resolved_config["args"]["env"]["obs_dropout_lane"] == TRAIN_DROPOUT_LANE
        assert resolved_config["args"]["env"]["obs_dropout_boundary"] == TRAIN_DROPOUT_BOUNDARY

        _run_training_epoch(trainer)
        assert trainer.epoch == first_epoch + 1
        assert trainer.global_step > first_global_step
    finally:
        if trainer is not None:
            trainer.utilization.stop()
        if vecenv is not None:
            vecenv.close()
        torch.set_num_threads(previous_thread_count)
        torch.use_deterministic_algorithms(previous_deterministic_setting, warn_only=True)
