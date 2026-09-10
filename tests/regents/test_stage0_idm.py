import json
import struct
import sys
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

import pufferlib
from pufferlib import pufferl
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MAP = min((REPO_ROOT / "pufferlib/resources/drive/binaries/sdc_replay_test").glob("*.bin"))

SNAPSHOT_INT_FIELDS = ("id", "type", "sim_valid", "stopped", "active_agent", "controller")
SNAPSHOT_FLOAT_FIELDS = ("sim_x", "sim_y", "sim_z", "sim_heading", "sim_vx", "sim_vy", "sim_speed")
BACKGROUND_LOG_FIELDS = (
    ("sim_x", "log_trajectory_x"),
    ("sim_y", "log_trajectory_y"),
    ("sim_z", "log_trajectory_z"),
    ("sim_heading", "log_heading"),
    ("sim_vx", "log_velocity_x"),
    ("sim_vy", "log_velocity_y"),
)


def _drive_kwargs(scenario_length=16):
    return {
        "map_dir": str(FIXTURE_MAP),
        "num_maps": 1,
        "num_agents": 1,
        "min_agents_per_env": 1,
        "max_agents_per_env": 1,
        "num_eval_scenarios": 1,
        "max_scenarios_per_batch": 1,
        "starting_map": 0,
        "eval_map_indices": [0],
        "eval_scenario_seeds": [42],
        "seed": 42,
        "simulation_mode": "replay",
        "eval_mode": True,
        "compute_eval_metrics": True,
        "control_mode": "control_sdc_only",
        "sdc_controller": "idm",
        "non_sdc_controller": "replay",
        "non_vehicle_controller": "replay",
        "action_type": "continuous",
        "dynamics_model": "classic",
        "dt": 0.1,
        "scenario_length": scenario_length,
        "resample_frequency": scenario_length,
        "init_step": 0,
        "init_step_spread": False,
        "reward_conditioning": False,
        "reward_randomization": False,
        "collision_behavior": "stop",
        "offroad_behavior": "stop",
        "traffic_light_behavior": "stop",
        "termination_mode": 0,
        "terminate_on_goal": False,
        "goal_source": "route",
        "use_neighbor_cache": 0,
    }


def _single_scenario(state):
    if isinstance(state, dict):
        return state
    assert len(state) == 1
    return state[0]


def _snapshot_bytes(scenario):
    snapshot = {
        "scenario_id": scenario["scenario_id"],
        "active_agent_indices": scenario["active_agent_indices"],
        "agents": [
            {
                **{name: int(agent[name]) for name in SNAPSHOT_INT_FIELDS},
                **{name: float(agent[name]).hex() for name in SNAPSHOT_FLOAT_FIELDS},
            }
            for agent in scenario["agents"]
        ],
    }
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()


def _rollout_prefix(action_value):
    env = Drive(**_drive_kwargs())
    try:
        env.reset()
        scenarios = [_single_scenario(env.get_state())]
        actions = np.full_like(env.actions, action_value)
        for _ in range(8):
            env.step(actions)
            scenarios.append(_single_scenario(env.get_state()))
        return scenarios, tuple(_snapshot_bytes(scenario) for scenario in scenarios)
    finally:
        env.close()


def _assert_controller_routing(scenario):
    assert scenario["active_agent_count"] == 1
    assert scenario["active_agent_indices"] == [0]
    ego = scenario["agents"][0]
    assert ego["id"] == 0
    assert ego["active_agent"] == 1
    assert ego["controller"] == binding.CONTROLLER_IDM
    assert all(agent["active_agent"] == 0 for agent in scenario["agents"][1:])
    assert all(agent["controller"] == binding.CONTROLLER_REPLAY for agent in scenario["agents"][1:])


POLICY_REQUIREMENT_CASES = [
    ("control_sdc_only", "idm", "policy", "policy", False),
    ("control_sdc_only", "policy", "replay", "replay", True),
    ("control_vehicles", "idm", "replay", "policy", False),
    ("control_vehicles", "idm", "policy", "replay", True),
    ("control_agents", "idm", "replay", "replay", False),
    ("control_agents", "idm", "replay", "policy", True),
    ("control_wosac", "idm", "replay", "policy", True),
]


def test_real_replay_routes_idm_replays_background_and_scopes_the_policy_requirement():
    """Native controller routing, action-independent IDM determinism, and policy gating."""
    first_scenarios, first_snapshots = _rollout_prefix(action_value=0.0)
    repeated_scenarios, repeated_snapshots = _rollout_prefix(action_value=0.0)
    _, ignored_action_snapshots = _rollout_prefix(action_value=1.0)

    assert first_snapshots == repeated_snapshots
    assert first_snapshots == ignored_action_snapshots
    assert first_snapshots[0] != first_snapshots[1]
    for scenario in first_scenarios + repeated_scenarios:
        _assert_controller_routing(scenario)

    checked_background_count = 0
    for step, scenario in enumerate(first_scenarios[1:], start=1):
        for agent in scenario["agents"][1:]:
            if agent["log_valid"][step] != 1:
                continue
            assert agent["sim_valid"] == 1
            for sim_field, log_field in BACKGROUND_LOG_FIELDS:
                assert agent[sim_field] == agent[log_field][step]
            checked_background_count += 1
    assert checked_background_count > 0

    # A policy is required only when an actively controlled category routes to it.
    for control_mode, sdc, non_sdc, non_vehicle, expected in POLICY_REQUIREMENT_CASES:
        environment_config = {
            "control_mode": control_mode,
            "sdc_controller": sdc,
            "non_sdc_controller": non_sdc,
            "non_vehicle_controller": non_vehicle,
        }
        assert drive_benchmark.environment_requires_policy(environment_config) is expected, environment_config


def _write_idm_benchmark(config_path):
    config_path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "eval_mode": True,
                    "compute_eval_metrics": True,
                    "collision_behavior": "stop",
                    "offroad_behavior": "stop",
                    "traffic_light_behavior": "stop",
                    "termination_mode": 0,
                },
                "benchmarks": [
                    {
                        "name": "regents_idm_test",
                        "seed": 42,
                        "num_scenarios": 1,
                        "env": {
                            "simulation_mode": "replay",
                            "num_maps": 1,
                            "scenario_length": 8,
                            "max_scenarios_per_batch": 1,
                            "control_mode": "control_sdc_only",
                            "min_agents_per_env": 1,
                            "sdc_controller": "idm",
                            "non_sdc_controller": "replay",
                            "non_vehicle_controller": "replay",
                            "action_type": "continuous",
                            "dynamics_model": "classic",
                            "dt": 0.1,
                            "init_step": 0,
                            "init_step_spread": False,
                            "reward_conditioning": False,
                            "reward_randomization": False,
                            "terminate_on_goal": False,
                            "goal_source": "route",
                            "map_dir": str(FIXTURE_MAP),
                            "use_neighbor_cache": 0,
                        },
                    }
                ],
            },
            sort_keys=False,
        )
    )


def _checkpoint_free_eval_args(tmp_path, benchmark_config_path):
    with patch.object(sys, "argv", ["pufferl.py"]):
        args = pufferl.load_config("puffer_drive")
    args["load_model_path"] = None
    args["train"]["data_dir"] = str(tmp_path / "default_output")
    args["vec"].update({"num_envs": 1, "num_workers": 1, "seed": 42})
    args["eval"].update(
        {
            "benchmark_config": str(benchmark_config_path),
            "benchmarks": "regents_idm_test",
            "num_agents": 1,
            "max_sdc_replay_workers": 1,
            "render_scenarios": True,
            "render_filter": None,
            "failure_replay_csv": None,
            "capture_observations": True,
            "observation_replay_writer_count": 1,
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["tb"] = False
    return args


def _read_replay_header(replay_path):
    payload = zlib.decompress(replay_path.read_bytes())
    header_length = struct.unpack_from("<I", payload)[0]
    return json.loads(payload[4 : 4 + header_length])


def test_puffer_eval_runs_idm_without_a_checkpoint_but_still_demands_one_for_policy(tmp_path, monkeypatch):
    """Stage 0 runs through the standard eval command with no policy loaded anywhere."""
    calls = []
    monkeypatch.setattr(pufferl, "eval", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["puffer", "eval", "puffer_drive", "regents_idm"])
    pufferl.main()
    assert calls == [{"env_name": "puffer_drive", "benchmark_names": "regents_idm"}]
    monkeypatch.undo()

    benchmark_config_path = tmp_path / "benchmark.yaml"
    _write_idm_benchmark(benchmark_config_path)
    args = _checkpoint_free_eval_args(tmp_path, benchmark_config_path)

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("Controller-only evaluation attempted to load or run a policy")

    monkeypatch.setattr(drive_benchmark, "load_checkpoint_architecture", unexpected_call)
    monkeypatch.setattr(pufferl, "load_policy", unexpected_call)
    result = pufferl.eval(
        env_name="puffer_drive",
        args=args,
        eval_output_subdir="run",
        benchmark_names="regents_idm_test",
    )

    benchmark_output = tmp_path / "default_output/eval/regents_idm_test/run"
    assert result["regents_idm_test"]["summary"]["num_episodes"] == 1
    episode = result["regents_idm_test"]["episodes"][0]
    assert {
        "collision_rate",
        "at_fault_collision_rate",
        "offroad_rate",
        "red_light_violation_rate",
        "episode_length",
    } <= episode.keys()
    assert (benchmark_output / "episode_metrics.csv").is_file()
    assert (benchmark_output / "evaluation_summary.json").is_file()
    resolved = yaml.safe_load((benchmark_output / "resolved_benchmark.yaml").read_text())
    assert resolved["checkpoint_config"] is None
    assert resolved["args"]["env"]["sdc_controller"] == "idm"

    replay_header = _read_replay_header(next((benchmark_output / "replays").glob("*.replay.zlib")))
    assert {"obs", "raw_action", "clipped_action"} <= set(replay_header["chunks"])
    assert {"value", "entropy", "policy_probs", "policy_mean"}.isdisjoint(replay_header["chunks"])
    rendered_pages = [
        path for path in (benchmark_output / "rendered_replays").glob("*.html") if path.name != "index.html"
    ]
    assert len(rendered_pages) == 1
    assert (benchmark_output / "rendered_replays/index.html").is_file()

    # Switching the ego to a policy must fail loudly rather than silently skipping it,
    # so the guards that assert no policy is loaded have to come off first.
    monkeypatch.undo()
    policy_config = yaml.safe_load(benchmark_config_path.read_text())
    policy_config["benchmarks"][0]["env"]["sdc_controller"] = "policy"
    benchmark_config_path.write_text(yaml.safe_dump(policy_config, sort_keys=False))
    policy_args = _checkpoint_free_eval_args(tmp_path, benchmark_config_path)
    policy_args["eval"]["render_scenarios"] = False
    policy_args["eval"]["capture_observations"] = False
    with pytest.raises(pufferlib.APIUsageError, match="Evaluation with policy controllers requires load_model_path"):
        pufferl.eval(
            env_name="puffer_drive",
            args=policy_args,
            eval_output_subdir="run",
            benchmark_names="regents_idm_test",
        )


def _read_replay_header(replay_path):
    payload = zlib.decompress(replay_path.read_bytes())
    header_length = struct.unpack_from("<I", payload)[0]
    return json.loads(payload[4 : 4 + header_length])
