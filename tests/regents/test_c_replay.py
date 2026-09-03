import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.adapter import export_drive_scenarios
from pufferlib.ocean.regents.artifacts import load_generation_artifact, save_generation_artifact
from pufferlib.ocean.regents.generation import (
    RENDER_DIR_NAME,
    _full_env_config,
    generate_regents_scenarios,
    load_generation_config,
    render_scenario_replays,
)
from pufferlib.ocean.regents.optimizer import ReGentSOptimizationConfig, optimize_frozen_ego_scenario
from pufferlib.ocean.regents.rollout import (
    C_REPLAY_TOLERANCE,
    replay_optimized_scenario_in_c,
    run_reactive_idm_generation,
)
from pufferlib.ocean.regents.state import STATE_HEADING, STATE_X


REPO_ROOT = Path(__file__).resolve().parents[2]
NUPLAN_MAP_DIR = REPO_ROOT / "pufferlib/resources/drive/binaries/nuplan"
GENERATION_CONFIG = REPO_ROOT / "pufferlib/config/evaluation/regents.yaml"
HORIZON_TRANSITION_COUNT = 16
REPLAY_FIXTURES = ((8, 50),)


def _drive(map_idx, seed, sdc_controller):
    return Drive(
        map_dir=str(NUPLAN_MAP_DIR),
        num_maps=map_idx + 1,
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_eval_scenarios=1,
        max_scenarios_per_batch=1,
        eval_map_indices=[map_idx],
        eval_scenario_seeds=[seed],
        seed=42,
        simulation_mode="replay",
        eval_mode=True,
        control_mode="control_sdc_only",
        sdc_controller=sdc_controller,
        non_sdc_controller="replay",
        non_vehicle_controller="replay",
        action_type="continuous",
        dynamics_model="classic",
        dt=0.1,
        scenario_length=200,
        resample_frequency=200,
        init_step=0,
        init_step_spread=False,
        reward_conditioning=False,
        reward_randomization=False,
        use_neighbor_cache=0,
    )


def _open_loop_replay(map_idx, seed, iteration_count=5, learning_rate=1e-3):
    drive = _drive(map_idx, seed, "replay")
    try:
        drive.reset(seed=seed)
        scenario = export_drive_scenarios(drive, raster_resolution_meters=5.0)
        optimization = optimize_frozen_ego_scenario(
            scenario,
            config=ReGentSOptimizationConfig(iteration_count=iteration_count, learning_rate=learning_rate),
            deterministic_seed=seed,
            horizon_transition_count=HORIZON_TRANSITION_COUNT,
        )
        replay = replay_optimized_scenario_in_c(drive, scenario, optimization, seed=seed)
    finally:
        drive.close()
    return scenario, optimization, replay


@pytest.fixture(scope="module", params=REPLAY_FIXTURES)
def cached_replay(request):
    map_idx, seed = request.param
    scenario, optimization, replay = _open_loop_replay(map_idx, seed)
    return map_idx, seed, scenario, optimization, replay


def test_open_loop_replay_reproduces_torch_within_stage_two_tolerance(cached_replay):
    map_idx, seed, scenario, optimization, replay = cached_replay

    assert replay.metrics.maximum_trajectory_error <= C_REPLAY_TOLERANCE
    assert replay.metrics.compared_state_count > 0
    assert replay.states.shape == optimization.optimized_states.shape
    assert replay.baseline_states.shape == replay.states.shape
    assert replay.ego_actions.shape == (1, HORIZON_TRANSITION_COUNT, 2)
    assert torch.isfinite(replay.states[replay.state_valid]).all()
    assert scenario.scenario_ids[0]


def test_c_replay_starts_from_the_torch_initial_pose(cached_replay):
    map_idx, seed, scenario, optimization, replay = cached_replay

    joint_valid = optimization.state_valid & replay.state_valid
    pose = slice(STATE_X, STATE_HEADING + 1)
    difference = torch.abs(replay.states[:, :, 0, pose] - optimization.optimized_states[:, :, 0, pose])
    assert float(difference[joint_valid[:, :, 0]].max()) <= C_REPLAY_TOLERANCE


def test_c_replay_is_deterministic(cached_replay):
    map_idx, seed, scenario, optimization, first = cached_replay
    drive = _drive(map_idx, seed, "replay")
    try:
        drive.reset(seed=seed)
        repeated = replay_optimized_scenario_in_c(drive, scenario, optimization, seed=seed)
    finally:
        drive.close()

    assert torch.equal(first.states, repeated.states)
    assert torch.equal(first.state_valid, repeated.state_valid)
    assert first.metrics == repeated.metrics
    assert first.failure_reason == repeated.failure_reason


def test_non_injected_replay_actors_follow_their_logged_trajectory(cached_replay):
    map_idx, seed, scenario, optimization, replay = cached_replay

    injected = optimization.optimized_action_mask.any(dim=2)
    logged = scenario.logged_state[:, :, : replay.states.shape[2]]
    joint_valid = optimization.state_valid & replay.state_valid & ~injected[..., None]
    pose = slice(STATE_X, STATE_HEADING + 1)
    difference = torch.abs(replay.states[..., pose] - logged[..., pose])
    assert float(difference[joint_valid].max()) <= C_REPLAY_TOLERANCE


def test_baseline_replay_absorbs_pre_existing_logged_collisions(cached_replay):
    map_idx, seed, scenario, optimization, replay = cached_replay

    # Map 8 logs overlapping actors, so C must not blame the adversary for them.
    assert replay.metrics.baseline_collision_pair_count > 0
    assert not replay.metrics.background_collision
    assert replay.failure_reason != "c_background_collision"


def test_reactive_idm_generation_confirms_an_actionable_collision_in_c():
    drive = _drive(8, 50, "idm")
    try:
        result = run_reactive_idm_generation(
            drive,
            ReGentSOptimizationConfig(iteration_count=30, learning_rate=1e-2),
            deterministic_seed=50,
            horizon_transition_count=HORIZON_TRANSITION_COUNT,
            maximum_outer_iterations=3,
        )
    finally:
        drive.close()

    assert result.replay.success
    assert result.replay.failure_reason is None
    assert result.replay.metrics.actionable_collision
    assert result.replay.metrics.ego_collision
    assert not result.replay.metrics.background_collision
    assert not result.replay.metrics.offroad
    assert result.replay.metrics.first_collision_pair == (0, result.optimization.selected_adversary_idx)
    assert result.replay.metrics.maximum_trajectory_error <= C_REPLAY_TOLERANCE
    assert 1 <= result.outer_iteration_count <= 3
    assert result.optimization.frozen_ego_source == "c_idm"


def test_reactive_idm_generation_requires_an_idm_ego():
    drive = _drive(8, 50, "policy")
    try:
        with pytest.raises(ValueError, match="sdc_controller"):
            run_reactive_idm_generation(
                drive,
                deterministic_seed=50,
                horizon_transition_count=HORIZON_TRANSITION_COUNT,
            )
    finally:
        drive.close()


def test_injection_binding_rejects_the_ego_and_out_of_range_actions():
    drive = _drive(8, 50, "replay")
    try:
        drive.reset(seed=50)
        agent_count = len(drive.get_state()[0]["agents"])
        actions = np.zeros((agent_count, HORIZON_TRANSITION_COUNT, 2), dtype=np.float32)
        mask = np.zeros((agent_count, HORIZON_TRANSITION_COUNT), dtype=np.bool_)
        mask[0] = True
        with pytest.raises(ValueError, match="replay-controlled non-ego vehicles"):
            binding.regents_set_action_plan(drive.c_envs, actions, mask)
        mask[:] = False
        actions[1, 0, 0] = 1.5
        with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
            binding.regents_set_action_plan(drive.c_envs, actions, mask)
        actions[1, 0, 0] = math.nan
        with pytest.raises(ValueError, match="finite"):
            binding.regents_set_action_plan(drive.c_envs, actions, mask)
    finally:
        drive.close()


def test_generation_artifact_round_trips(tmp_path):
    drive = _drive(8, 50, "idm")
    try:
        result = run_reactive_idm_generation(
            drive,
            ReGentSOptimizationConfig(iteration_count=5, learning_rate=1e-2),
            deterministic_seed=50,
            horizon_transition_count=HORIZON_TRANSITION_COUNT,
            maximum_outer_iterations=1,
        )
    finally:
        drive.close()
    map_path = sorted(NUPLAN_MAP_DIR.glob("*.bin"))[8]
    artifact_path = tmp_path / "scenario.npz"
    saved = save_generation_artifact(artifact_path, result, {"fixture": "test"}, str(map_path))
    metadata, arrays = load_generation_artifact(artifact_path)

    assert metadata == saved
    assert metadata["scenario_id"] == result.scenario.scenario_ids[0]
    assert metadata["deterministic_seed"] == 50
    assert len(metadata["source_configuration_hash"]) == 64
    assert np.array_equal(arrays["optimized_actions"], result.optimization.optimized_actions.detach().numpy())
    assert np.array_equal(arrays["c_states"], result.replay.states.numpy())
    assert arrays["original_states"].shape[1] == result.scenario.max_agent_count


def test_replay_frames_render_through_the_shared_viewer(tmp_path):
    drive = _drive(8, 50, "idm")
    try:
        result = run_reactive_idm_generation(
            drive,
            ReGentSOptimizationConfig(iteration_count=5, learning_rate=1e-2),
            deterministic_seed=50,
            horizon_transition_count=HORIZON_TRANSITION_COUNT,
            maximum_outer_iterations=1,
            capture_html_frames=True,
        )
    finally:
        drive.close()

    frames = result.replay.adversarial_frames
    assert frames is not None
    assert frames["agent_f32"].shape[0] == HORIZON_TRANSITION_COUNT + 1
    assert frames["obs"].shape[0] == HORIZON_TRANSITION_COUNT + 1
    assert result.replay.baseline_frames["agent_f32"].shape == frames["agent_f32"].shape
    assert result.replay.scenario_payload["scenario_id"] == result.scenario.scenario_ids[0]

    environment = _full_env_config({"map_dir": str(NUPLAN_MAP_DIR), "dt": 0.1})
    rendered = render_scenario_replays(tmp_path, 8, result, environment)
    assert sorted(rendered) == ["scenario_00008.adversarial.html", "scenario_00008.logged.html"]
    for name in rendered:
        page = (tmp_path / RENDER_DIR_NAME / name).read_text(encoding="utf-8")
        assert "<title>PufferDrive Replay</title>" in page
        assert "http://" not in page and "https://" not in page


def test_replay_frames_are_absent_unless_capture_is_requested():
    _, _, replay = _open_loop_replay(8, 50, iteration_count=1)

    assert replay.adversarial_frames is None
    assert replay.baseline_frames is None
    assert replay.scenario_payload is None


def test_generation_config_rejects_unknown_names_and_env_keys(tmp_path):
    with pytest.raises(ValueError, match="Unknown ReGentS generation"):
        load_generation_config(GENERATION_CONFIG, "missing_generation")
    broken = tmp_path / "regents.yaml"
    broken.write_text(
        "env:\n  not_a_drive_argument: 1\ngenerations:\n"
        "  - name: broken\n    seed: 1\n    scenario_count: 1\n"
        "    horizon_transition_count: 1\n    maximum_outer_iterations: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported env keys"):
        load_generation_config(broken, "broken")


def test_generation_config_rejects_a_horizon_the_drive_guards_would_reject(tmp_path):
    config = tmp_path / "regents.yaml"
    config.write_text(
        "env:\n  scenario_length: 200\n  resample_frequency: 200\n"
        "generations:\n  - name: too_long\n    seed: 1\n    scenario_count: 1\n"
        "    horizon_transition_count: 200\n    maximum_outer_iterations: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be below env.resample_frequency=200"):
        load_generation_config(config, "too_long")

    config.write_text(
        "env:\n  scenario_length: 200\n  resample_frequency: 201\n"
        "generations:\n  - name: too_long\n    seed: 1\n    scenario_count: 1\n"
        "    horizon_transition_count: 200\n    maximum_outer_iterations: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be below env.scenario_length=200"):
        load_generation_config(config, "too_long")


def test_offline_generation_entry_point_writes_artifacts_and_metrics(tmp_path):
    config = load_generation_config(GENERATION_CONFIG, "regents_nuplan_smoke")
    expected_count = config["scenario_count"]
    report = generate_regents_scenarios(GENERATION_CONFIG, "regents_nuplan_smoke", output_dir=tmp_path)

    assert report.scenario_count == expected_count
    expected_files = [f"scenario_{i:05d}.npz" for i in range(expected_count)]
    assert sorted(path.name for path in (tmp_path / "npz").glob("*.npz")) == expected_files
    assert (tmp_path / "generation_metrics.csv").is_file()
    assert report.maximum_c_torch_trajectory_error <= C_REPLAY_TOLERANCE
    assert 0.0 <= report.generation_success_rate <= 1.0
    assert report.total_optimization_seconds > 0.0
    assert sum(report.rejection_reasons.values()) == report.scenario_count - int(
        report.generation_success_rate * report.scenario_count
    )
