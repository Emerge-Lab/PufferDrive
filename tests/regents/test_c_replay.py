import json
import math
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

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
    save_loss_history_csv,
)
from pufferlib.ocean.regents.filters import ReGentSFilterConfig
from pufferlib.ocean.regents.policy_ego import PolicyEgoActor, ReGentSPolicyEgoConfig
from pufferlib.ocean.regents.optimizer import ReGentSOptimizationConfig, optimize_frozen_ego_scenario
from pufferlib.ocean.regents.rollout import (
    C_REPLAY_TOLERANCE,
    baseline_relative_events,
    replay_optimized_scenario_in_c,
    run_reactive_generation,
)
from pufferlib.ocean.regents.state import (
    STATE_FEATURE_COUNT,
    STATE_HEADING,
    STATE_X,
    agent_state_rows,
    single_scenario_payload,
)
from tests.regents.real_fixtures import NUPLAN_MAP_DIR, REGENTS_AUDIT_SCENARIO_IDS, resolve_nuplan_scenarios


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATION_CONFIG = REPO_ROOT / "pufferlib/config/evaluation/regents.yaml"
HORIZON_TRANSITION_COUNT = 16
# Full-log filtering keeps candidates that a short window never activates, so the C
# oracle needs a fixture that actually injects. Map 7 also logs an overlapping pair
# inside this window, which the adversary must not be blamed for.
OPEN_LOOP_HORIZON_TRANSITION_COUNT = 50
REPLAY_STOPPED_FIELD_IDX = 4
REPLAY_FIXTURES = ((7, 50),)


def test_baseline_relative_events_accepts_any_injected_adversary_collision():
    baseline = SimpleNamespace(collision_pairs=(), offroad=np.zeros(4, dtype=np.bool_))
    adversarial = SimpleNamespace(
        collision_pairs=((3, ((0, 2), (0, 3))),),
        offroad=np.zeros(4, dtype=np.bool_),
    )
    injected_agent_mask = np.array((False, False, True, False), dtype=np.bool_)

    events = baseline_relative_events(baseline, adversarial, injected_agent_mask)

    assert events.ego_collision
    assert events.actionable_collision
    assert events.first_ego_collision_timestep == 3


def test_baseline_relative_events_rejects_unperturbed_collision():
    baseline = SimpleNamespace(collision_pairs=(), offroad=np.zeros(3, dtype=np.bool_))
    adversarial = SimpleNamespace(collision_pairs=((4, ((0, 2),)),), offroad=np.zeros(3, dtype=np.bool_))
    injected_agent_mask = np.array((False, True, False), dtype=np.bool_)

    events = baseline_relative_events(baseline, adversarial, injected_agent_mask)

    assert events.ego_collision
    assert not events.actionable_collision


def _drive_kwargs(map_idx, sdc_controller, dynamics_model="classic", zero_erratic=False):
    """The Drive keywords a pinned fixture uses, minus its per-scenario map and seed."""
    if map_idx < 0 or map_idx >= len(REGENTS_AUDIT_SCENARIO_IDS):
        raise ValueError(f"Pinned ReGentS fixture index must be in [0, {len(REGENTS_AUDIT_SCENARIO_IDS)})")
    map_paths, _, _ = resolve_nuplan_scenarios()
    return dict(
        map_dir=str(NUPLAN_MAP_DIR),
        num_maps=len(map_paths),
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_eval_scenarios=1,
        max_scenarios_per_batch=1,
        seed=42,
        simulation_mode="replay",
        eval_mode=True,
        control_mode="control_sdc_only",
        sdc_controller=sdc_controller,
        non_sdc_controller="replay",
        non_vehicle_controller="replay",
        action_type="continuous",
        dynamics_model=dynamics_model,
        dt=0.1,
        **(
            {
                "obs_dropout_lane": 0.0,
                "obs_dropout_boundary": 0.0,
                "partner_blindness_prob": 0.0,
                "partner_blindness_trigger_prob": 0.0,
                "phantom_braking_prob": 0.0,
                "phantom_braking_trigger_prob": 0.0,
            }
            if zero_erratic
            else {}
        ),
        scenario_length=200,
        resample_frequency=200,
        init_step=0,
        init_step_spread=False,
        reward_conditioning=False,
        reward_randomization=False,
        use_neighbor_cache=0,
    )


def _drive(map_idx, seed, sdc_controller, dynamics_model="classic", zero_erratic=False):
    _, map_indices, _ = resolve_nuplan_scenarios()
    return Drive(
        **_drive_kwargs(map_idx, sdc_controller, dynamics_model, zero_erratic),
        eval_map_indices=[map_indices[map_idx]],
        eval_scenario_seeds=[seed],
    )


def _open_loop_replay(map_idx, seed, iteration_count=5, learning_rate=1e-3):
    drive = _drive(map_idx, seed, "replay")
    try:
        drive.reset(seed=seed)
        scenario = export_drive_scenarios(drive, raster_resolution_meters=5.0)
        optimization = optimize_frozen_ego_scenario(
            scenario,
            # This fixture specifically exercises baseline-relative collision accounting.
            # Its logged overlapping actor is stationary, so opt out of the production
            # speed gate without weakening that gate for normal generation.
            config=ReGentSOptimizationConfig(
                filter=ReGentSFilterConfig(static_speed_threshold_mps=0.0),
                iteration_count=iteration_count,
                learning_rate=learning_rate,
            ),
            deterministic_seed=seed,
            horizon_transition_count=OPEN_LOOP_HORIZON_TRANSITION_COUNT,
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


def test_open_loop_c_replay_reproduces_torch_and_is_deterministic(cached_replay):
    """C is authoritative: parity, shared initial pose, untouched actors, and repeatability."""
    assert C_REPLAY_TOLERANCE == 1e-3
    map_idx, seed, scenario, optimization, replay = cached_replay

    assert replay.metrics.maximum_trajectory_error <= C_REPLAY_TOLERANCE
    assert replay.metrics.compared_state_count > 0
    assert replay.states.shape == optimization.optimized_states.shape
    assert replay.baseline_states.shape == replay.states.shape
    assert replay.ego_actions.shape == (OPEN_LOOP_HORIZON_TRANSITION_COUNT, 2)
    assert torch.isfinite(replay.states[replay.state_valid]).all()
    assert scenario.scenario_id == REGENTS_AUDIT_SCENARIO_IDS[map_idx]

    pose = slice(STATE_X, STATE_HEADING + 1)
    joint_valid = optimization.state_valid & replay.state_valid
    initial_difference = torch.abs(replay.states[:, 0, pose] - optimization.optimized_states[:, 0, pose])
    assert float(initial_difference[joint_valid[:, 0]].max()) <= C_REPLAY_TOLERANCE

    # Actors without an injected plan must still follow their logged trajectory.
    injected = optimization.optimized_action_mask.any(dim=1)
    assert int(injected.sum()) > 0
    logged = scenario.logged_state[:, : replay.states.shape[1]]
    logged_difference = torch.abs(replay.states[..., pose] - logged[..., pose])
    assert float(logged_difference[joint_valid & ~injected[..., None]].max()) <= C_REPLAY_TOLERANCE

    # Map 7 logs overlapping actors, so C must not blame the adversary for them.
    assert replay.metrics.baseline_collision_pair_count > 0
    assert not replay.metrics.baseline_ego_collision
    assert not replay.metrics.background_collision

    # Frames are opt-in so the default path stays cheap.
    assert replay.adversarial_frames is None
    assert replay.baseline_frames is None
    assert replay.scenario_payload is None

    drive = _drive(map_idx, seed, "replay")
    try:
        drive.reset(seed=seed)
        repeated = replay_optimized_scenario_in_c(drive, scenario, optimization, seed=seed)
    finally:
        drive.close()
    assert torch.equal(replay.states, repeated.states)
    assert torch.equal(replay.state_valid, repeated.state_valid)
    assert replay.metrics == repeated.metrics
    assert replay.failure_reason == repeated.failure_reason

    # The injection binding refuses the ego and any out-of-contract action plan.
    injection_drive = _drive(map_idx, seed, "replay")
    try:
        injection_drive.reset(seed=seed)
        agent_count = len(injection_drive.get_state()[0]["agents"])
        actions = np.zeros((agent_count, OPEN_LOOP_HORIZON_TRANSITION_COUNT, 2), dtype=np.float32)
        mask = np.zeros((agent_count, OPEN_LOOP_HORIZON_TRANSITION_COUNT), dtype=np.bool_)
        mask[0] = True
        with pytest.raises(ValueError, match="replay-controlled non-ego vehicles"):
            binding.regents_set_action_plan(injection_drive.c_envs, actions, mask)
        mask[:] = False
        actions[1, 0, 0] = 1.5
        with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
            binding.regents_set_action_plan(injection_drive.c_envs, actions, mask)
        actions[1, 0, 0] = math.nan
        with pytest.raises(ValueError, match="finite"):
            binding.regents_set_action_plan(injection_drive.c_envs, actions, mask)
    finally:
        injection_drive.close()


def test_final_stop_replay_freezes_colliding_agents_and_reuses_checked_parity(cached_replay):
    map_idx, seed, scenario, optimization, full_horizon_replay = cached_replay
    _, map_indices, _ = resolve_nuplan_scenarios()
    drive = Drive(
        **_drive_kwargs(map_idx, "replay"),
        collision_behavior="stop",
        offroad_behavior="stop",
        eval_map_indices=[map_indices[map_idx]],
        eval_scenario_seeds=[seed],
    )
    try:
        stopped_replay = replay_optimized_scenario_in_c(
            drive,
            scenario,
            optimization,
            seed=seed,
            capture_html_frames=True,
            verified_parity_metrics=full_horizon_replay.metrics,
        )
    finally:
        drive.close()

    stopped = stopped_replay.baseline_frames["agent_i32"][:, :, REPLAY_STOPPED_FIELD_IDX]
    assert np.any((stopped[1:] == 1) & (stopped[:-1] == 0))
    assert stopped_replay.adversarial_frames["agent_f32"].shape[0] == OPEN_LOOP_HORIZON_TRANSITION_COUNT + 1
    assert stopped_replay.metrics.maximum_trajectory_error == full_horizon_replay.metrics.maximum_trajectory_error
    assert stopped_replay.metrics.compared_state_count == full_horizon_replay.metrics.compared_state_count


def test_reactive_idm_generation_reports_absent_horizon_candidates_and_round_trips_its_artifact(tmp_path):
    """Full-log candidates outside a short horizon remain reportable and replayable."""
    drive = _drive(8, 50, "idm")
    try:
        result = run_reactive_generation(
            drive,
            ReGentSOptimizationConfig(iteration_count=30, learning_rate=1e-2),
            deterministic_seed=50,
            horizon_transition_count=HORIZON_TRANSITION_COUNT,
            capture_html_frames=True,
        )
    finally:
        drive.close()

    assert not result.replay.success
    assert result.replay.failure_reason == "no_candidate_in_optimization_horizon"
    assert not result.replay.metrics.actionable_collision
    assert not result.replay.metrics.ego_collision
    assert not result.replay.metrics.background_collision
    assert not result.replay.metrics.offroad
    assert result.replay.metrics.first_collision_pair is None
    assert result.optimization.ego_collision_loss_adversary_idx == -1
    assert result.optimization.ego_collision_loss_adversary_id == -1
    assert result.optimization.selection.candidate_mask.sum() == 1
    assert result.optimization.selection.reasons_for(20) == ("static",)
    assert result.optimization.selection.reasons_for(22) == ("static",)
    assert torch.equal(result.optimization.initial_actions, result.optimization.optimized_actions)
    assert result.replay.metrics.maximum_trajectory_error <= C_REPLAY_TOLERANCE
    assert result.ego_refresh_count == 0
    assert result.optimization.frozen_ego_source == "c_idm"

    # A jerk env governs the ego alone: injected adversaries still integrate the classic
    # bicycle model, so C/Torch parity is unchanged and the plan is accepted.
    jerk_drive = _drive(8, 50, "idm", dynamics_model="jerk")
    try:
        jerk_result = run_reactive_generation(
            jerk_drive,
            ReGentSOptimizationConfig(iteration_count=30, learning_rate=1e-2),
            deterministic_seed=50,
            horizon_transition_count=HORIZON_TRANSITION_COUNT,
        )
    finally:
        jerk_drive.close()
    assert jerk_result.replay.metrics.maximum_trajectory_error <= C_REPLAY_TOLERANCE
    assert torch.equal(jerk_result.optimization.optimized_actions, result.optimization.optimized_actions)

    _, _, fixture_paths = resolve_nuplan_scenarios()
    map_path = fixture_paths[8]
    artifact_path = tmp_path / "scenario.npz"
    saved = save_generation_artifact(artifact_path, result, {"fixture": "test"}, str(map_path))
    metadata, arrays = load_generation_artifact(artifact_path)
    assert metadata == saved
    assert metadata["scenario_id"] == result.scenario.scenario_id
    assert metadata["c_replay"]["baseline_ego_collision"] == result.replay.metrics.baseline_ego_collision
    assert metadata["deterministic_seed"] == 50
    assert len(metadata["source_configuration_hash"]) == 64
    assert metadata["optimization"]["background_collision_loss_scope"] == "candidate_pairs"
    assert metadata["optimization"]["result_selection_policy"] == "current_iterate"
    assert metadata["optimization"]["infractions_are_acceptance_gates"] is False
    assert (
        metadata["optimization"]["ego_collision_loss_adversary_id"]
        == result.optimization.ego_collision_loss_adversary_id
    )
    initial_costs = metadata["optimization"]["initial_costs"]
    assert initial_costs is None
    assert metadata["optimization"]["baseline_background_collision_pair_count"] >= 0
    assert metadata["optimization"]["background_collision_rejection_count"] >= 0
    assert np.array_equal(arrays["optimized_actions"], result.optimization.optimized_actions.detach().numpy())
    assert np.array_equal(arrays["c_states"], result.replay.states.numpy())
    assert arrays["original_states"].shape[0] == result.scenario.max_agent_count

    frames = result.replay.adversarial_frames
    assert frames is not None
    assert frames["agent_f32"].shape[0] == HORIZON_TRANSITION_COUNT + 1
    assert "obs" not in frames
    assert "obs" not in result.replay.baseline_frames
    assert result.replay.baseline_frames["agent_f32"].shape == frames["agent_f32"].shape
    assert result.replay.scenario_payload["scenario_id"] == result.scenario.scenario_id

    # No candidate ever entered the window, so there is no cost history to write.
    assert result.optimization.cost_history == ()
    save_loss_history_csv(tmp_path, 8, result)
    assert not (tmp_path / "losses").exists()

    observation_drive = _drive(8, 50, "idm")
    try:
        observation_replay = replay_optimized_scenario_in_c(
            observation_drive,
            result.scenario,
            result.optimization,
            seed=50,
            capture_html_frames=True,
        )
    finally:
        observation_drive.close()
    assert observation_replay.adversarial_frames["agent_f32"].shape[0] == HORIZON_TRANSITION_COUNT + 1

    rendered = render_scenario_replays(
        tmp_path, 8, result, _full_env_config({"map_dir": str(NUPLAN_MAP_DIR), "dt": 0.1})
    )
    assert sorted(rendered) == ["scenario_00008.adversarial.html", "scenario_00008.logged.html"]
    for name in rendered:
        page = (tmp_path / RENDER_DIR_NAME / name).read_text(encoding="utf-8")
        assert "<title>PufferDrive Replay</title>" in page
        assert 'const ADVERSARY_COLOR = "#a16207";' in page
        assert 'const LOSS_ADVERSARY_COLOR = "#c026d3";' in page
        assert "isLossAdversary?LOSS_ADVERSARY_COLOR" in page
        assert "a.id+' [LOSS ADV]'" in page
        assert "Ego collision loss adversary" in page
        assert "http://" not in page and "https://" not in page
        replay_path = tmp_path / "replays" / name.replace(".html", ".replay.zlib")
        replay_payload = zlib.decompress(replay_path.read_bytes())
        header_length = struct.unpack_from("<I", replay_payload)[0]
        header = json.loads(replay_payload[4 : 4 + header_length])
        candidate_indices = torch.where(result.optimization.selection.candidate_mask)[0]
        expected_candidate_ids = result.scenario.agent_id[candidate_indices].tolist()
        assert header["candidate_adversary_ids"] == expected_candidate_ids
        assert header["active_count"] == 1
        assert header["obs_dim"] == 0
        assert "obs" not in header["chunks"]
        assert header["chunks"]["raw_action"]["shape"] == [HORIZON_TRANSITION_COUNT + 1, 1, 2]
        assert header["ego_collision_loss_adversary_idx"] == result.optimization.ego_collision_loss_adversary_idx
        assert header["ego_collision_loss_adversary_id"] == result.optimization.ego_collision_loss_adversary_id

    import pufferlib.viz

    pufferlib.viz.build_gallery_index(str(tmp_path / RENDER_DIR_NAME), file_metrics=rendered)
    gallery = (tmp_path / RENDER_DIR_NAME / "index.html").read_text(encoding="utf-8")
    assert "No candidate" in gallery
    assert 'data-nocandidate="false"' in gallery
    assert "Iteration limit" in gallery
    assert 'data-iterationlimit="false"' in gallery
    assert 'data-failure-reason="no_candidate_in_optimization_horizon"' in gallery
    assert "IDM reconstruction collisions" in gallery
    assert "IDM reconstruction collision" in gallery
    assert 'data-idmcollision="false"' in gallery
    for class_key in ("genuine_failure", "adversary_forced", "unavoidable"):
        assert rendered["scenario_00008.logged.html"][class_key] == 0.0
        assert class_key in rendered["scenario_00008.adversarial.html"]
    for filter_key, filter_label in (
        ("genuinefailure", "Genuine failure"),
        ("adversaryforced", "Adversary-forced"),
        ("unavoidable", "Unavoidable"),
    ):
        assert f'data-{filter_key}="false"' in gallery
        assert filter_label in gallery

    # A policy ego is supported, but only with an action provider to drive it.
    policy_drive = _drive(8, 50, "policy")
    try:
        with pytest.raises(ValueError, match="ego action provider"):
            run_reactive_generation(
                policy_drive, deterministic_seed=50, horizon_transition_count=HORIZON_TRANSITION_COUNT
            )
    finally:
        policy_drive.close()


def test_generation_config_is_validated_and_the_offline_entry_point_writes_artifacts(tmp_path):
    """Every config guard, the raster default, and one end-to-end smoke generation."""
    with pytest.raises(ValueError, match="Unknown ReGentS generation"):
        load_generation_config(GENERATION_CONFIG, "missing_generation")
    with pytest.raises(ValueError, match="experiment name"):
        generate_regents_scenarios(
            GENERATION_CONFIG,
            "regents_nuplan_smoke",
            output_dir=tmp_path,
            experiment_name="../escape",
        )
    with pytest.raises(ValueError, match="drivable-area weight"):
        generate_regents_scenarios(
            GENERATION_CONFIG,
            "regents_nuplan_smoke",
            output_dir=tmp_path,
            drivable_area_weight=math.nan,
        )

    wod_motion = load_generation_config(GENERATION_CONFIG, "regents_wod_motion_val")
    assert wod_motion["scenario_count"] == 250
    assert wod_motion["horizon_transition_count"] == 90
    assert wod_motion["raster_resolution_meters"] == 1.0
    assert wod_motion["optimizer"]["learning_rate"] == 0.005
    assert wod_motion["optimizer"]["filter"]["maximum_reconstruction_drift_meters"] == 1.0
    assert wod_motion["env"]["map_dir"] == "pufferlib/resources/drive/binaries/wod-motion_val"
    assert wod_motion["env"]["scenario_length"] == 91
    assert wod_motion["env"]["resample_frequency"] == 91
    assert wod_motion["env"]["sdc_controller"] == "idm"
    assert wod_motion["optimizer"]["ego_refresh_interval"] == 1
    assert wod_motion["env"]["base_max_speed_mps"] == 40.0

    config = tmp_path / "regents.yaml"
    config.write_text(
        "env:\n  not_a_drive_argument: 1\ngenerations:\n"
        "  - name: broken\n    seed: 1\n    scenario_count: 1\n"
        "    horizon_transition_count: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported env keys"):
        load_generation_config(config, "broken")

    config.write_text(
        "env:\n  scenario_length: 200\n  resample_frequency: 200\n"
        "generations:\n  - name: too_long\n    seed: 1\n    scenario_count: 1\n"
        "    horizon_transition_count: 200\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be below env.resample_frequency=200"):
        load_generation_config(config, "too_long")

    config.write_text(
        "env:\n  scenario_length: 200\n  resample_frequency: 201\n"
        "generations:\n  - name: too_long\n    seed: 1\n    scenario_count: 1\n"
        "    horizon_transition_count: 200\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be below env.scenario_length=200"):
        load_generation_config(config, "too_long")

    config.write_text(
        "env:\n  num_maps: 1\ngenerations:\n  - name: broken\n    seed: 1\n"
        "    scenario_count: 1\n    horizon_transition_count: 1\n"
        "    raster_resolution_meters: .nan\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite and positive"):
        load_generation_config(config, "broken")

    config.write_text(
        "env:\n  num_maps: 1\ngenerations:\n  - name: defaulted\n    seed: 1\n"
        "    scenario_count: 1\n    horizon_transition_count: 1\n",
        encoding="utf-8",
    )
    defaulted = load_generation_config(config, "defaulted")
    assert defaulted["raster_resolution_meters"] == 0.5

    smoke = load_generation_config(GENERATION_CONFIG, "regents_nuplan_smoke")
    expected_count = smoke["scenario_count"]
    report = generate_regents_scenarios(GENERATION_CONFIG, "regents_nuplan_smoke", output_dir=tmp_path)
    assert report.scenario_count == expected_count
    assert sorted(path.name for path in (tmp_path / "npz").glob("*.npz")) == [
        f"scenario_{i:05d}.npz" for i in range(expected_count)
    ]
    assert (tmp_path / "generation_metrics.csv").is_file()
    metrics_header = (tmp_path / "generation_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "candidate_count" in metrics_header
    assert "torch_collision" in metrics_header
    assert "torch_collision_timestep" in metrics_header
    assert "baseline_ego_collision" in metrics_header
    loss_header = (tmp_path / "losses/scenario_00000.losses.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "background_collision_first_agent_id" in loss_header
    assert "background_collision_signed_distance_meters" in loss_header
    assert report.maximum_c_torch_trajectory_error <= C_REPLAY_TOLERANCE
    assert 0.0 <= report.generation_success_rate <= 1.0
    assert 0.0 <= report.candidate_success_rate <= 1.0
    assert 0.0 <= report.torch_collision_rate <= 1.0
    assert 0.0 <= report.c_collision_confirmation_rate <= 1.0
    assert report.candidate_scenario_count + report.filtered_no_candidate_count == report.scenario_count
    assert report.c_confirmed_actionable_collision_count + report.c_unconfirmed_actionable_collision_count == (
        report.torch_collision_count
    )
    assert report.total_optimization_seconds > 0.0
    assert report.wall_clock_seconds > 0.0
    assert sum(report.rejection_reasons.values()) == report.scenario_count - int(
        report.generation_success_rate * report.scenario_count
    )


def test_pufferl_regents_prints_success(tmp_path, capsys, monkeypatch):
    from pufferlib import pufferl

    report = pufferl.regents(
        "regents_nuplan_smoke",
        config_path=GENERATION_CONFIG,
        output_dir=tmp_path,
        experiment_name="roadless",
        drivable_area_weight=0.0,
    )
    captured = capsys.readouterr()
    assert "[REGENTS] eligible" in captured.out
    assert "[REGENTS] C-verified success" in captured.out
    assert "[REGENTS] collision funnel" in captured.out
    assert report.output_dir == tmp_path / "roadless"
    metadata, _ = load_generation_artifact(report.output_dir / "npz/scenario_00000.npz")
    source_configuration = metadata["source_configuration"]
    assert source_configuration["experiment_name"] == "roadless"
    assert source_configuration["optimizer"]["costs"]["drivable_area_weight"] == 0.0
    assert metadata["schema"] == "pufferdrive_regents_generation"
    assert metadata["optimization"]["collision_loss_metric"] == "signed_oriented_box_distance_meters"
    assert metadata["optimization"]["road_loss_reduction"] == "mean_timesteps_sum_valid_agents_and_corners"
    assert metadata["optimization"]["road_loss_kernel"] == "unit_mass_truncated_grid_convolution"

    calls = []
    monkeypatch.setattr(pufferl, "regents", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "puffer",
            "regents",
            "puffer_drive",
            "regents_nuplan",
            "--exp-name",
            "road_weight_zero",
            "--road-weight",
            "0",
        ],
    )
    pufferl.main()
    assert calls == [
        {
            "generation_name": "regents_nuplan",
            "experiment_name": "road_weight_zero",
            "drivable_area_weight": 0.0,
        }
    ]


def test_buffer_state_getter_reproduces_the_dict_getter_exactly():
    """The replay path reads states from buffers, so both getters must agree bit for bit."""
    drive = _drive(1, 43, "replay")
    try:
        drive.reset(seed=43)
        agent_count = int(single_scenario_payload(drive.get_state())["num_total_agents"])
        states = np.empty((agent_count, STATE_FEATURE_COUNT), dtype=np.float32)
        valid = np.empty(agent_count, dtype=np.bool_)
        ego_action = np.empty(2, dtype=np.float32)
        neutral_actions = np.zeros_like(drive.actions)
        for _ in range(30):
            drive.step(neutral_actions)
            _, dict_states, dict_valid, dict_ego_action = agent_state_rows(drive.get_state(), agent_count)
            binding.regents_get_states(drive.c_envs, states, valid, ego_action)
            assert np.array_equal(states, dict_states)
            assert np.array_equal(valid, dict_valid)
            assert np.array_equal(ego_action, dict_ego_action)
    finally:
        drive.close()


def test_buffer_state_getter_rejects_malformed_output_buffers():
    """Buffers cross the trust boundary from Python, so shape and dtype are checked."""
    drive = _drive(1, 43, "replay")
    try:
        drive.reset(seed=43)
        agent_count = int(single_scenario_payload(drive.get_state())["num_total_agents"])
        states = np.empty((agent_count, STATE_FEATURE_COUNT), dtype=np.float32)
        valid = np.empty(agent_count, dtype=np.bool_)
        ego_action = np.empty(2, dtype=np.float32)
        with pytest.raises(ValueError):
            binding.regents_get_states(drive.c_envs, states[:-1], valid, ego_action)
        with pytest.raises(ValueError):
            binding.regents_get_states(drive.c_envs, states, valid, ego_action[:1])
        with pytest.raises(TypeError):
            binding.regents_get_states(drive.c_envs, states.astype(np.float64), valid, ego_action)
        with pytest.raises(TypeError):
            binding.regents_get_states(drive.c_envs, states, valid.astype(np.int32), ego_action)
    finally:
        drive.close()


POLICY_CHECKPOINT = REPO_ROOT / "experiments/3_0_no_conditionning_target.pt"
POLICY_CHECKPOINT_CONFIG = REPO_ROOT / "experiments/3_0_no_conditionning_target_config.yaml"


@pytest.mark.skipif(
    not (POLICY_CHECKPOINT.is_file() and POLICY_CHECKPOINT_CONFIG.is_file()),
    reason="The policy-ego checkpoint is a local artifact, not a repository fixture",
)
def test_policy_ego_generation_is_deterministic_and_its_artifact_set_replays_under_every_controller(tmp_path):
    """A learned ego drives generation, and the saved set replays under any ego."""
    policy_config = ReGentSPolicyEgoConfig(
        checkpoint_path=str(POLICY_CHECKPOINT),
        config_path=str(POLICY_CHECKPOINT_CONFIG),
    )
    # The head is 4 jerk-long x 3 jerk-lat classes, so the env must be jerk for the
    # checkpoint to load at all; injection stays classic.
    optimization_config = ReGentSOptimizationConfig(iteration_count=20, learning_rate=1e-2, ego_refresh_interval=5)
    results = []
    for _ in range(2):
        drive = _drive(1, 43, "policy", dynamics_model="jerk", zero_erratic=True)
        try:
            results.append(
                run_reactive_generation(
                    drive,
                    optimization_config,
                    deterministic_seed=43,
                    horizon_transition_count=OPEN_LOOP_HORIZON_TRANSITION_COUNT,
                    show_progress=False,
                    ego_action_fn=PolicyEgoActor(policy_config, drive),
                )
            )
        finally:
            drive.close()
    first, second = results
    assert first.ego_source == "c_policy"
    assert first.optimization.frozen_ego_source == "c_policy"
    assert first.ego_refresh_count > 0
    assert first.replay.metrics.maximum_trajectory_error <= C_REPLAY_TOLERANCE
    # Inference must not depend on RNG or worker state, or an ego refresh is not reproducible.
    assert torch.equal(first.optimization.optimized_actions, second.optimization.optimized_actions)
    assert torch.equal(first.replay.states, second.replay.states)

    # A policy ego without an action provider, and an action provider without a policy
    # ego, are both rejected before any work happens.
    idm_drive = _drive(1, 43, "idm")
    try:
        with pytest.raises(ValueError, match="ego action provider"):
            run_reactive_generation(
                idm_drive,
                optimization_config,
                deterministic_seed=43,
                horizon_transition_count=OPEN_LOOP_HORIZON_TRANSITION_COUNT,
                ego_action_fn=lambda observations: None,
            )
    finally:
        idm_drive.close()
