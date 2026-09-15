import math
from pathlib import Path

import numpy as np
import pytest

from data_utils.generate_carla_idm import DATASET_NAME, generate_scenario
from data_utils.mirror_map_bin import read_bin
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.generation import load_generation_config


REPO_ROOT = Path(__file__).resolve().parents[2]
TOWN01_MAP = REPO_ROOT / "pufferlib/resources/drive/binaries/carla/opendrive__Town01.bin"
REGENTS_CONFIG = REPO_ROOT / "pufferlib/config/evaluation/regents.yaml"
TEST_AGENT_COUNT = 8
TEST_TRANSITION_COUNT = 4
TEST_SEED = 42


def _decoded(value):
    return value.split(b"\0", 1)[0].decode("utf-8")


def _replay_drive(binary_path):
    return Drive(
        map_dir=str(binary_path),
        num_maps=1,
        num_agents=TEST_AGENT_COUNT,
        min_agents_per_env=TEST_AGENT_COUNT,
        max_agents_per_env=TEST_AGENT_COUNT,
        num_eval_scenarios=1,
        max_scenarios_per_batch=1,
        eval_map_indices=[0],
        eval_scenario_seeds=[TEST_SEED],
        seed=TEST_SEED,
        simulation_mode="replay",
        eval_mode=True,
        compute_eval_metrics=False,
        control_mode="control_agents",
        sdc_controller="replay",
        non_sdc_controller="replay",
        non_vehicle_controller="replay",
        action_type="continuous",
        dynamics_model="classic",
        dt=0.1,
        scenario_length=TEST_TRANSITION_COUNT + 1,
        resample_frequency=0,
        init_step=0,
        init_step_spread=False,
        init_mode="create_all_valid",
        reward_conditioning=False,
        reward_randomization=False,
        collision_behavior="ignore",
        offroad_behavior="ignore",
        traffic_light_behavior="ignore",
        target_infraction_behavior="normal",
        termination_mode=False,
        adversarial_termination_mode="disabled",
        terminate_on_goal=False,
        goal_source="route",
        partner_blindness_prob=0.0,
        phantom_braking_prob=0.0,
    )


def _single_scenario(state):
    if isinstance(state, dict):
        return state
    assert len(state) == 1
    return state[0]


def _assert_replay_matches_binary(binary_path, generated_data):
    replay = _replay_drive(binary_path)
    try:
        replay.reset()
        neutral_actions = np.zeros_like(replay.actions)
        for sample_idx in range(TEST_TRANSITION_COUNT + 1):
            if sample_idx > 0:
                _, _, terminals, truncations, _ = replay.step(neutral_actions)
                assert not np.any(terminals)
                assert not np.any(truncations)
            scenario = _single_scenario(replay.get_state())
            assert scenario["timestep"] == sample_idx
            assert scenario["active_agent_count"] == TEST_AGENT_COUNT
            for agent_idx, agent in enumerate(scenario["agents"]):
                captured = generated_data["agents"][agent_idx]["cols"]
                assert agent["sim_x"] == captured["x"][sample_idx]
                assert agent["sim_y"] == captured["y"][sample_idx]
                assert agent["sim_z"] == captured["z"][sample_idx]
                assert agent["sim_heading"] == captured["h"][sample_idx]
                assert agent["sim_vx"] == captured["vx"][sample_idx]
                assert agent["sim_vy"] == captured["vy"][sample_idx]
                assert agent["sim_length"] == captured["len"][sample_idx]
                assert agent["sim_width"] == captured["wid"][sample_idx]
                assert agent["sim_height"] == captured["hgt"][sample_idx]
                assert agent["sim_valid"] == captured["valid"][sample_idx]
    finally:
        replay.close()


def test_regents_profile_uses_all_generated_carla_logs_with_a_pdm_ego():
    generation = load_generation_config(REGENTS_CONFIG, "regents_carla_generated_idm_pdm")

    assert generation["seed"] == 42
    assert generation["scenario_count"] == 8
    assert generation["num_workers"] == 24
    assert generation["horizon_transition_count"] == 150
    assert generation["raster_resolution_meters"] == 1.0
    assert generation["render_replays"] is True
    assert generation["optimizer"]["iteration_count"] == 500
    assert generation["optimizer"]["ego_refresh_interval"] == 1
    assert generation["optimizer"]["filter"] == {
        "static_speed_threshold_mps": 0.2,
        "maximum_reconstruction_drift_meters": 1.0,
    }
    assert generation["env"]["map_dir"] == "pufferlib/resources/drive/binaries/carla_generated_idm"
    assert generation["env"]["num_maps"] == 8
    assert generation["env"]["scenario_length"] == 151
    assert generation["env"]["resample_frequency"] == 151
    assert generation["env"]["sdc_controller"] == "pdm"
    assert generation["env"]["non_sdc_controller"] == "replay"


def test_regents_generation_overrides_standard_optimizer_and_filter_values(tmp_path):
    config_path = tmp_path / "regents.yaml"
    config_path.write_text(
        """\
env:
  scenario_length: 11
  resample_frequency: 11
raster_resolution_meters: 1.0
optimizer:
  learning_rate: 0.001
  iteration_count: 500
  ego_refresh_interval: 1
  filter:
    static_speed_threshold_mps: 0.2
    maximum_reconstruction_drift_meters: 1.0
generations:
  - name: overridden
    seed: 1
    scenario_count: 1
    horizon_transition_count: 10
    raster_resolution_meters: 0.25
    optimizer:
      iteration_count: 3
      filter:
        static_speed_threshold_mps: 0.4
""",
        encoding="utf-8",
    )

    generation = load_generation_config(config_path, "overridden")

    assert generation["raster_resolution_meters"] == 0.25
    assert generation["optimizer"]["learning_rate"] == 0.001
    assert generation["optimizer"]["iteration_count"] == 3
    assert generation["optimizer"]["ego_refresh_interval"] == 1
    assert generation["optimizer"]["filter"] == {
        "static_speed_threshold_mps": 0.4,
        "maximum_reconstruction_drift_meters": 1.0,
    }


def test_town01_idm_generation_is_complete_deterministic_atomic_and_replayable(tmp_path):
    first_path = tmp_path / "first" / TOWN01_MAP.name
    second_path = tmp_path / "second" / TOWN01_MAP.name
    summary = generate_scenario(
        TOWN01_MAP,
        first_path,
        agent_count=TEST_AGENT_COUNT,
        transition_count=TEST_TRANSITION_COUNT,
        seed=TEST_SEED,
    )
    source_data = read_bin(TOWN01_MAP)
    generated_data = read_bin(first_path)

    assert summary.agent_count == TEST_AGENT_COUNT
    assert summary.sample_count == TEST_TRANSITION_COUNT + 1
    assert summary.scenario_id == "Town01_seed_42"
    assert len(summary.sha256) == 64
    assert _decoded(generated_data["scenario_id"]) == summary.scenario_id
    assert _decoded(generated_data["dataset_name"]) == DATASET_NAME
    assert generated_data["log_length"] == TEST_TRANSITION_COUNT + 1
    assert generated_data["log_dt"] == pytest.approx(0.1)
    assert generated_data["roads"] == source_data["roads"]
    assert generated_data["objects"] == source_data["objects"]
    assert generated_data["lane_graph"] == source_data["lane_graph"]
    assert generated_data["has_phase_section"] == source_data["has_phase_section"]
    assert generated_data["has_width_section"] == source_data["has_width_section"]

    assert len(generated_data["agents"]) == TEST_AGENT_COUNT
    for agent_idx, agent in enumerate(generated_data["agents"]):
        assert agent["id"] == agent_idx
        assert agent["T"] == TEST_TRANSITION_COUNT + 1
        assert agent["route"]
        assert agent["route_gt_len"] == len(agent["route"])
        assert agent["goal"] == tuple(agent["cols"][name][-1] for name in ("x", "y", "z"))
        assert tuple(agent["cols"]["valid"]) == (1,) * (TEST_TRANSITION_COUNT + 1)
        assert agent["cols"]["vx"][0] == 0.0
        assert agent["cols"]["vy"][0] == 0.0
        for column_name in ("x", "y", "z", "h", "vx", "vy", "len", "wid", "hgt"):
            values = agent["cols"][column_name]
            assert len(values) == TEST_TRANSITION_COUNT + 1
            assert all(math.isfinite(value) for value in values)

    assert len(generated_data["traffic"]) == len(source_data["traffic"])
    for generated_traffic, source_traffic in zip(generated_data["traffic"], source_data["traffic"]):
        assert generated_traffic["id"] == source_traffic["id"]
        assert generated_traffic["type"] == source_traffic["type"]
        assert generated_traffic["stop_line"] == source_traffic["stop_line"]
        assert generated_traffic["controlled_lanes"] == source_traffic["controlled_lanes"]
        assert generated_traffic["junction_id"] == source_traffic["junction_id"]
        assert generated_traffic["phase_idx"] == source_traffic["phase_idx"]
        assert len(generated_traffic["states"]) == TEST_TRANSITION_COUNT + 1

    generate_scenario(
        TOWN01_MAP,
        second_path,
        agent_count=TEST_AGENT_COUNT,
        transition_count=TEST_TRANSITION_COUNT,
        seed=TEST_SEED,
    )
    expected_bytes = second_path.read_bytes()
    assert first_path.read_bytes() == expected_bytes

    first_path.write_bytes(b"different existing output")
    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_scenario(
            TOWN01_MAP,
            first_path,
            agent_count=TEST_AGENT_COUNT,
            transition_count=TEST_TRANSITION_COUNT,
            seed=TEST_SEED,
        )
    assert first_path.read_bytes() == b"different existing output"
    generate_scenario(
        TOWN01_MAP,
        first_path,
        agent_count=TEST_AGENT_COUNT,
        transition_count=TEST_TRANSITION_COUNT,
        seed=TEST_SEED,
        overwrite=True,
    )
    assert first_path.read_bytes() == expected_bytes

    _assert_replay_matches_binary(first_path, generated_data)
