import pickle
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from data_utils.generate_carla_policy import (
    DATASET_NAME,
    PolicyScenarioWriter,
    _append_terminal_frame,
    _episode_infraction_reasons,
    _select_map_cycled_entries,
    generate_policy_scenarios,
    resolve_generation_args,
)
from data_utils.mirror_map_bin import read_bin


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = REPO_ROOT / "experiments/3_0_no_conditionning_target.pt"
CHECKPOINT_CONFIG = REPO_ROOT / "experiments/3_0_no_conditionning_target_config.yaml"
BENCHMARK_CONFIG = REPO_ROOT / "pufferlib/config/evaluation/benchmark.yaml"


def test_terminal_frame_completes_full_horizon_capture():
    replay_environment = {
        "metadata": {"initial_timestep": 0, "episode_timestep": 2},
        "frames": {
            key: np.zeros((2, 1, 1), dtype=np.float32)
            for key in ("agent_f32", "agent_i32", "metrics_f32", "puffer_f32", "traffic_i16")
        },
        "terminal_frame": {
            key: np.ones((1, 1), dtype=np.float32)
            for key in ("agent_f32", "agent_i32", "metrics_f32", "puffer_f32", "traffic_i16")
        },
    }

    frames = _append_terminal_frame(replay_environment)

    assert frames["agent_f32"].shape[0] == 3
    assert frames["agent_f32"][-1, 0, 0] == 1.0


@pytest.mark.parametrize(
    ("offroad_rate", "collision_rate", "expected_reasons"),
    [
        (0.0, 0.0, ()),
        (0.25, 0.0, ("offroad",)),
        (0.0, 0.5, ("collision",)),
        (0.25, 0.5, ("offroad", "collision")),
    ],
)
def test_episode_infraction_reasons_rejects_any_agent_infraction(
    offroad_rate,
    collision_rate,
    expected_reasons,
):
    summary = {"offroad_rate": offroad_rate, "collision_rate": collision_rate}

    assert _episode_infraction_reasons(summary) == expected_reasons


@pytest.mark.parametrize("metric_value", [-1.0, float("nan"), float("inf")])
def test_episode_infraction_reasons_rejects_invalid_metrics(metric_value):
    with pytest.raises(RuntimeError, match="invalid offroad_rate"):
        _episode_infraction_reasons({"offroad_rate": metric_value, "collision_rate": 0.0})


def test_policy_writer_rejects_failed_spawn_without_aborting_generation(tmp_path):
    replay_environment = {
        "schema": "interactive_replay_environment_v1",
        "scenario": {
            "agents": [{"route": None}],
            "active_agent_indices": [0],
        },
    }
    summary = {
        "seed": 29,
        "map_name": "opendrive__Town01.bin",
        "episode_timestep": 100,
        "offroad_rate": 0.0,
        "collision_rate": 0.0,
        "replay_environment_bundle": zlib.compress(pickle.dumps(replay_environment)),
    }
    writer = PolicyScenarioWriter(tmp_path, dt_seconds=0.1, reject_infractions=True)

    writer(summary, episode_idx=0)

    assert writer.entries == []
    assert writer.rejections == [
        {
            "map": "opendrive__Town01.bin",
            "seed": 29,
            "termination_timestep": 100,
            "offroad_rate": 0.0,
            "collision_rate": 0.0,
            "reasons": ("spawn_failure",),
        }
    ]


def test_select_map_cycled_entries_interleaves_maps_and_sorts_each_map_by_seed():
    entries = [
        {"map": "map_2.bin", "seed": 22, "episode_idx": 0},
        {"map": "map_1.bin", "seed": 13, "episode_idx": 1},
        {"map": "map_3.bin", "seed": 31, "episode_idx": 2},
        {"map": "map_1.bin", "seed": 11, "episode_idx": 3},
        {"map": "map_3.bin", "seed": 33, "episode_idx": 4},
        {"map": "map_2.bin", "seed": 21, "episode_idx": 5},
    ]

    selected = _select_map_cycled_entries(entries, 6, ["map_1.bin", "map_2.bin", "map_3.bin"])

    assert [entry["map"] for entry in selected] == [
        "map_1.bin",
        "map_2.bin",
        "map_3.bin",
        "map_1.bin",
        "map_2.bin",
        "map_3.bin",
    ]
    assert [entry["seed"] for entry in selected] == [11, 21, 31, 13, 22, 33]
    assert [entry["episode_idx"] for entry in selected] == list(range(6))
    assert [entry["map_cycle_idx"] for entry in selected] == [0, 1, 2, 0, 1, 2]


def test_select_map_cycled_entries_requires_each_map_quota():
    entries = [
        {"map": "map_1.bin", "seed": 11, "episode_idx": 0},
        {"map": "map_2.bin", "seed": 21, "episode_idx": 1},
    ]

    with pytest.raises(RuntimeError, match="Map-cycled output needs scenario 2 from map_1.bin"):
        _select_map_cycled_entries(entries, 3, ["map_1.bin", "map_2.bin"])


@pytest.mark.skipif(
    not (CHECKPOINT.is_file() and CHECKPOINT_CONFIG.is_file()),
    reason="The policy checkpoint is a local artifact, not a repository fixture",
)
def test_policy_generation_uses_one_unconditioned_policy_and_writes_deterministic_standard_bins(tmp_path):
    run_args, _ = resolve_generation_args(
        CHECKPOINT,
        CHECKPOINT_CONFIG,
        BENCHMARK_CONFIG,
        "adversarial_carla",
        scenario_count=1,
        scenario_length=4,
        device="cpu",
    )
    assert run_args["policy_name"] == "TargetDrive"
    assert run_args["eval"]["action_selection"] == "mean"
    assert run_args["env"]["control_mode"] == "control_vehicles"
    assert run_args["env"]["sdc_controller"] == "policy"
    assert run_args["env"]["non_sdc_controller"] == "policy"

    output_directories = [tmp_path / "first", tmp_path / "second"]
    entries = []
    for output_directory in output_directories:
        generated_entries = generate_policy_scenarios(
            checkpoint_path=CHECKPOINT,
            checkpoint_config_path=CHECKPOINT_CONFIG,
            benchmark_config_path=BENCHMARK_CONFIG,
            output_directory=output_directory,
            scenario_count=1,
            scenario_length=4,
            num_workers=1,
            device="cpu",
        )
        entries.append(generated_entries[0])

    first_path = output_directories[0] / entries[0]["file"]
    second_path = output_directories[1] / entries[1]["file"]
    assert first_path.read_bytes() == second_path.read_bytes()
    assert entries[0]["sha256"] == entries[1]["sha256"]
    assert entries[0]["file"].startswith("000000_opendrive__Town01_")

    generated = read_bin(first_path)
    manifest = yaml.safe_load((output_directories[0] / "generation_manifest.yaml").read_text())
    assert manifest["policy_name"] == "TargetDrive"
    assert manifest["map_cycle"] == [
        "opendrive__Town01.bin",
        "opendrive__Town02.bin",
        "opendrive__Town03.bin",
        "opendrive__Town04.bin",
        "opendrive__Town05.bin",
        "opendrive__Town06.bin",
        "opendrive__Town07.bin",
        "opendrive__Town10HD.bin",
    ]
    dataset_name = generated["dataset_name"].split(b"\0", 1)[0].decode("utf-8")
    assert dataset_name == DATASET_NAME
    assert 2 <= len(generated["agents"]) <= 8
    assert generated["log_length"] == 5
    assert generated["objects_of_interest"] == ()
    assert generated["tracks_to_predict"] == ()
    assert all(agent["mark_as_expert"] == 0 for agent in generated["agents"])
    assert all(agent["T"] == generated["log_length"] for agent in generated["agents"])

    clean_output_directory = tmp_path / "clean"
    clean_entries = generate_policy_scenarios(
        checkpoint_path=CHECKPOINT,
        checkpoint_config_path=CHECKPOINT_CONFIG,
        benchmark_config_path=BENCHMARK_CONFIG,
        output_directory=clean_output_directory,
        scenario_count=1,
        scenario_length=4,
        num_workers=1,
        device="cpu",
        reject_infractions=True,
        max_candidate_scenarios=1,
    )
    clean_manifest = yaml.safe_load((clean_output_directory / "generation_manifest.yaml").read_text())
    assert len(clean_entries) == 1
    assert clean_manifest["reject_infractions"] is True
    assert clean_manifest["candidate_scenario_count"] == 1
    assert clean_manifest["clean_candidate_count"] == 1
    assert clean_manifest["rejections"] == []
