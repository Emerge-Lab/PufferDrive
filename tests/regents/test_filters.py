import math
from dataclasses import replace

import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents.filters import (
    ReGentSFilterConfig,
    front_divergence_mask,
    select_adversary_candidates,
)
from pufferlib.ocean.regents.inverse_dynamics import estimate_expert_actions
from pufferlib.ocean.regents.state import DrivableAreaRaster, RasterTransform, Scenario


def make_scenario(states, valid=None, agent_types=None, drivable_mask=None):
    states = states.to(torch.float32)
    if states.ndim == 4:
        assert states.shape[0] == 1
        states = states[0]
        if valid is not None and valid.ndim == 3:
            valid = valid[0]
        if agent_types is not None and agent_types.ndim == 2:
            agent_types = agent_types[0]
    agent_count, time_count, _ = states.shape
    if valid is None:
        valid = torch.ones((agent_count, time_count), dtype=torch.bool)
    if agent_types is None:
        agent_types = torch.full((agent_count,), binding.AGENT_TYPE_VEHICLE, dtype=torch.int64)
    present = torch.ones((agent_count,), dtype=torch.bool)
    ego_mask = torch.zeros_like(present)
    ego_mask[0] = True
    vehicle_mask = present & (agent_types == binding.AGENT_TYPE_VEHICLE)
    dimensions = torch.full((agent_count,), 2.0, dtype=torch.float32)
    lengths = torch.full((agent_count,), 4.0, dtype=torch.float32)
    if drivable_mask is None:
        drivable_mask = torch.ones((101, 101), dtype=torch.bool)
    raster = DrivableAreaRaster(drivable_mask, RasterTransform(-50.0, -50.0, 1.0, *drivable_mask.shape))
    feature_valid = valid[..., None].expand_as(states).clone()
    return Scenario(
        logged_state=states,
        state_valid=valid,
        state_feature_valid=feature_valid,
        transition_valid=valid[:, :-1] & valid[:, 1:],
        current_state=states[:, 0].clone(),
        agent_present=present,
        agent_metadata_valid=present.clone(),
        agent_id=torch.arange(agent_count, dtype=torch.int64),
        trajectory_length=torch.full((agent_count,), time_count, dtype=torch.int64),
        ego_mask=ego_mask,
        vehicle_mask=vehicle_mask,
        candidate_adversary_mask=vehicle_mask & ~ego_mask,
        logged_length_meters=lengths[:, None].expand(agent_count, time_count).clone(),
        logged_width_meters=dimensions[:, None].expand(agent_count, time_count).clone(),
        length_meters=lengths,
        width_meters=dimensions,
        wheelbase_meters=0.6 * lengths,
        maximum_speed_mps=torch.full((agent_count,), 20.0, dtype=torch.float32),
        scenario_id="synthetic",
        dataset_name="test",
        log_dt_seconds=0.1,
        dt_seconds=0.1,
        init_step=0,
        scenario_length=time_count,
        drivable_area_raster=raster,
    )


def _linear_track(x_start, y, speed, time_count=6, heading=0.0, dt=0.1):
    state = torch.zeros((time_count, 5), dtype=torch.float32)
    state[:, 0] = x_start + torch.arange(time_count) * speed * dt
    state[:, 1] = y
    state[:, 2] = heading
    state[:, 3] = speed
    return state


def test_rear_exclusion_switch_keeps_other_candidate_filters():
    scenario = make_scenario(
        torch.stack(
            (
                _linear_track(0.0, 0.0, 2.0),
                _linear_track(-10.0, 0.0, 2.0),
                _linear_track(20.0, 0.0, 0.0),
            )
        )
    )
    regents = select_adversary_candidates(scenario, use_regents=True)
    king_comparison = select_adversary_candidates(scenario, use_regents=False)

    assert regents.candidate_mask.tolist() == [False, False, False]
    assert king_comparison.candidate_mask.tolist() == [False, True, False]
    assert "rear_sector" in regents.reasons_for(1)
    assert "rear_sector" not in king_comparison.reasons_for(1)
    assert "static" in regents.reasons_for(2)
    assert "static" in king_comparison.reasons_for(2)
    assert regents.rear_sector_fraction[1] == king_comparison.rear_sector_fraction[1]
    with pytest.raises(TypeError, match="use_regents"):
        select_adversary_candidates(scenario, use_regents=1)


def test_candidate_selection_records_every_reason_and_its_boundaries():
    """Per-agent reason bits, ego-only original collisions, and the strict rear sector."""
    assert ReGentSFilterConfig().static_speed_threshold_mps == 0.2
    assert ReGentSFilterConfig().maximum_reconstruction_drift_meters == math.inf
    with pytest.raises(ValueError, match="static_speed_threshold_mps"):
        ReGentSFilterConfig(static_speed_threshold_mps=-0.1)
    with pytest.raises(ValueError, match="maximum_reconstruction_drift_meters"):
        ReGentSFilterConfig(maximum_reconstruction_drift_meters=-1.0)

    states = torch.stack(
        (
            _linear_track(0.0, 0.0, 2.0),
            _linear_track(8.0, 4.0, 2.0),
            _linear_track(20.0, 0.0, 0.0),
            _linear_track(-10.0, 0.0, 2.0),
            _linear_track(0.0, 20.0, 2.0),
            _linear_track(0.0, 30.0, 2.0),
        )
    )[None]
    valid = torch.ones((1, 6, 6), dtype=torch.bool)
    valid[0, 5, 2:] = False
    agent_types = torch.full((1, 6), binding.AGENT_TYPE_VEHICLE, dtype=torch.int64)
    agent_types[0, 4] = binding.AGENT_TYPE_PEDESTRIAN
    selection = select_adversary_candidates(
        make_scenario(states, valid, agent_types),
        ReGentSFilterConfig(),
    )
    assert selection.scene_eligible.item() is True
    assert selection.candidate_mask.tolist() == [False, True, False, False, False, False]
    assert "ego" in selection.reasons_for(0)
    assert "static" in selection.reasons_for(2)
    assert "rear_sector" in selection.reasons_for(3)
    assert "non_vehicle" in selection.reasons_for(4)
    assert "insufficient_valid_states" in selection.reasons_for(5)

    # Logged overlap is reported but does not exclude a reference candidate.
    only_collider = select_adversary_candidates(
        make_scenario(torch.stack((_linear_track(0.0, 0.0, 2.0), _linear_track(3.0, 0.0, 2.0)))[None])
    )
    assert only_collider.original_collision.tolist() == [False, True]
    assert only_collider.original_collision_timestep.tolist() == [-1, 0]
    assert only_collider.scene_reasons() == ()
    assert only_collider.candidate_mask.tolist() == [False, True]

    with_survivor = select_adversary_candidates(
        make_scenario(
            torch.stack((_linear_track(0.0, 0.0, 2.0), _linear_track(3.0, 0.0, 2.0), _linear_track(8.0, 4.0, 2.0)))[
                None
            ]
        )
    )
    assert with_survivor.scene_eligible.item() is True
    assert with_survivor.candidate_mask.tolist() == [False, True, True]
    assert "original_collision" not in with_survivor.reasons_for(1)
    assert "original_collision" not in with_survivor.reasons_for(2)

    # Two logged backgrounds overlapping each other is not something the C simulator
    # ever scores, so it must not disqualify the scene or an uninvolved candidate.
    background_overlap = select_adversary_candidates(
        make_scenario(
            torch.stack((_linear_track(0.0, 0.0, 2.0), _linear_track(40.0, 30.0, 2.0), _linear_track(43.0, 30.0, 2.0)))[
                None
            ]
        )
    )
    assert background_overlap.original_collision.tolist() == [False, False, False]
    assert background_overlap.scene_eligible.item() is True
    assert background_overlap.scene_reasons() == ()
    assert background_overlap.candidate_mask.tolist() == [False, True, True]

    # The rear sector boundary is strict: exactly pi/8 from directly behind survives.
    exact_angle = 7.0 * math.pi / 8.0
    rear = select_adversary_candidates(
        make_scenario(
            torch.stack(
                (
                    _linear_track(0.0, 0.0, 1.0, 6),
                    _linear_track(10.0 * math.cos(exact_angle), 10.0 * math.sin(exact_angle), 1.0, 6),
                    _linear_track(10.0 * math.cos(exact_angle + 1e-4), 10.0 * math.sin(exact_angle + 1e-4), 1.0, 6),
                )
            )[None]
        ),
        ReGentSFilterConfig(
            static_displacement_threshold_meters=0.0,
            rear_sector_fraction=0.8,
        ),
    )
    assert "rear_sector" not in rear.reasons_for(1)
    assert "rear_sector" in rear.reasons_for(2)

    # Reference validity counts states, even if none form usable transitions.
    # Displacement spans valid endpoints while rear bearings include invalid samples;
    # horizon does not change selection. Zero reported speed marks a jittering track
    # static, while the exact speed and displacement thresholds remain inclusive.
    reference_states = torch.stack(
        (
            _linear_track(0.0, 0.0, 2.0),
            _linear_track(10.0, 10.0, 2.0),
            _linear_track(-10.0, 0.0, 2.0),
            _linear_track(0.0, 20.0, 0.0),
        )
    )
    reference_states[1, :, 3] = 0.0
    reference_states[2, -1, 0] = 10.0
    reference_states[3, -1, 0] = 0.2
    reference_states[3, :, 3] = 0.2
    reference_valid = torch.ones((4, 6), dtype=torch.bool)
    reference_valid[1, 1::2] = False
    reference_valid[2, :3] = False
    scenario = make_scenario(reference_states, reference_valid)
    full = select_adversary_candidates(scenario)
    short = select_adversary_candidates(scenario, horizon_transition_count=1)
    assert full.candidate_mask.tolist() == [False, False, False, True]
    assert torch.equal(full.candidate_mask, short.candidate_mask)
    assert full.valid_state_fraction[1] == 0.5
    assert full.valid_transition_count[1] == 0
    assert not full.optimized_action_mask[1].any()
    assert math.isclose(float(full.displacement_meters[1]), 0.8, abs_tol=1e-5)
    assert full.rear_sector_fraction[2] > 0.8

    # Independent transcription of the released selection expression, with the one
    # documented deviation: displacement spans valid endpoints, not raw storage.
    position = reference_states[..., :2]
    displacement = position - position[:1]
    angle = torch.atan2(displacement[..., 1], displacement[..., 0]) - reference_states[:1, :, 2]
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
    excluded = reference_valid.float().mean(dim=-1) < 0.5
    first_valid = reference_valid.to(torch.int8).argmax(dim=-1)
    last_valid = reference_valid.shape[-1] - 1 - reference_valid.to(torch.int8).flip(-1).argmax(dim=-1)
    endpoints = torch.stack([position[agent, [first_valid[agent], last_valid[agent]]] for agent in range(4)])
    excluded |= torch.linalg.vector_norm(endpoints[:, 1] - endpoints[:, 0], dim=-1) < 0.2
    maximum_speed = reference_states[..., 3].abs().masked_fill(~reference_valid, -torch.inf).max(dim=-1).values
    excluded |= maximum_speed < 0.2
    excluded |= ((angle > 7 * math.pi / 8) | (angle < -7 * math.pi / 8)).float().mean(dim=-1) > 0.8
    excluded |= scenario.ego_mask | ~scenario.vehicle_mask
    assert torch.equal(full.candidate_mask, ~excluded)

    reconstruction_scenario = make_scenario(torch.stack((_linear_track(0.0, 0.0, 2.0), _linear_track(8.0, 4.0, 2.0))))
    reconstruction_inverse = estimate_expert_actions(reconstruction_scenario)
    inconsistent_mask = reconstruction_inverse.model_consistent.clone()
    inconsistent_mask[1, 2] = False
    reconstruction_residual = reconstruction_inverse.residual_meters.clone()
    reconstruction_residual[1, 2] = 0.25
    reconstruction_inverse = replace(
        reconstruction_inverse,
        model_consistent=inconsistent_mask,
        residual_meters=reconstruction_residual,
    )
    drift_config = ReGentSFilterConfig(maximum_reconstruction_drift_meters=1.5)
    with pytest.raises(ValueError, match="reconstruction_drift_meters is required"):
        select_adversary_candidates(reconstruction_scenario, drift_config)
    measured_drift = torch.tensor([0.5, 1.75])
    drift_filtered = select_adversary_candidates(
        reconstruction_scenario,
        drift_config,
        inverse_dynamics=reconstruction_inverse,
        reconstruction_drift_meters=measured_drift,
    )
    # The threshold is exclusive: 1.5 m of drift is retained, 1.75 m is not.
    assert drift_filtered.candidate_mask.tolist() == [False, False]
    assert "reconstruction_fidelity" in drift_filtered.reasons_for(1)
    assert "reconstruction_fidelity" not in drift_filtered.reasons_for(0)
    assert torch.equal(drift_filtered.reconstruction_drift_meters, measured_drift)
    retained = select_adversary_candidates(
        reconstruction_scenario,
        ReGentSFilterConfig(maximum_reconstruction_drift_meters=1.75),
        inverse_dynamics=reconstruction_inverse,
        reconstruction_drift_meters=measured_drift,
    )
    assert "reconstruction_fidelity" not in retained.reasons_for(1)
    # Strict composite statistics stay reported even though the gate ignores them.
    assert drift_filtered.maximum_reconstruction_residual_meters[1] == 0.25
    assert drift_filtered.model_consistent_transition_fraction[1] == 0.8


def test_static_filter_ignores_zero_filled_frames_before_an_agent_enters():
    """A parked late entrant is static; storage padding must not read as displacement."""
    states = torch.zeros((3, 6, 5), dtype=torch.float32)
    states[0] = _linear_track(0.0, 0.0, 2.0)
    states[1, 2:, 0] = 30.0
    states[1, 2:, 1] = 5.0
    states[2, 2:, 0] = 20.0 + torch.arange(4) * 0.2
    states[2, 2:, 1] = -5.0
    states[2, 2:, 3] = 2.0
    valid = torch.ones((3, 6), dtype=torch.bool)
    valid[1:, :2] = False

    selection = select_adversary_candidates(make_scenario(states, valid))
    assert "static" in selection.reasons_for(1)
    assert "static" not in selection.reasons_for(2)
    assert selection.candidate_mask.tolist() == [False, False, True]
    assert math.isclose(float(selection.displacement_meters[1]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(selection.displacement_meters[2]), 0.6, abs_tol=1e-5)


def _front_states(position_angle, yaw, time_count=5):
    states = torch.zeros((2, time_count, 5), dtype=torch.float32)
    states[1, :, 0] = 10.0 * math.cos(position_angle)
    states[1, :, 1] = 10.0 * math.sin(position_angle)
    states[1, :, 2] = yaw
    return states


def test_front_divergence_uses_separate_bearing_and_yaw_windows_and_a_strict_fraction():
    """Both sides, the pi/8 bearing and pi/2 yaw boundaries, and the tau_front threshold."""
    valid = torch.ones((2, 5), dtype=torch.bool)
    ego_mask = torch.tensor([True, False])
    candidate_mask = torch.tensor([False, True])
    epsilon = 1e-4
    for sign in (-1.0, 1.0):
        bearing = sign * math.pi / 16.0
        inside = front_divergence_mask(
            _front_states(bearing, sign * (math.pi / 8.0 - epsilon)), valid, ego_mask, candidate_mask
        )
        wide_yaw = front_divergence_mask(
            _front_states(bearing, sign * (math.pi / 2.0 - epsilon)), valid, ego_mask, candidate_mask
        )
        yaw_boundary = front_divergence_mask(
            _front_states(bearing, sign * math.pi / 2.0), valid, ego_mask, candidate_mask
        )
        position_boundary = front_divergence_mask(
            _front_states(sign * math.pi / 8.0, sign * (math.pi / 8.0 - epsilon)),
            valid,
            ego_mask,
            candidate_mask,
        )
        assert inside.tolist() == [False, True]
        assert wide_yaw.tolist() == [False, True]
        assert yaw_boundary.tolist() == [False, False]
        assert position_boundary.tolist() == [False, False]

    half_diverging = _front_states(math.pi / 16.0, math.pi / 8.0 - epsilon, time_count=4)
    half_diverging[1, 2:, 1] *= -1.0
    four_step_valid = torch.ones((2, 4), dtype=torch.bool)
    assert not front_divergence_mask(half_diverging, four_step_valid, ego_mask, candidate_mask, tau_front=0.5)[1]
    assert front_divergence_mask(half_diverging, four_step_valid, ego_mask, candidate_mask, tau_front=0.49)[1]


def test_off_road_start_filter_keeps_agents_whose_footprint_touches_the_drivable_area():
    """A logged start fully off the raster is excluded; straddling its edge is not."""
    drivable_mask = torch.zeros((101, 101), dtype=torch.bool)
    drivable_mask[47:54] = True
    states = torch.stack(
        (
            _linear_track(0.0, 0.0, 2.0),
            _linear_track(8.0, 4.0, 2.0),
            _linear_track(8.0, 20.0, 2.0),
        )
    )[None]
    scenario = make_scenario(states, drivable_mask=drivable_mask)

    selection = select_adversary_candidates(scenario)
    assert selection.start_off_road.tolist() == [False, False, True]
    assert selection.candidate_mask.tolist() == [False, True, False]
    assert "off_road_start" in selection.reasons_for(2)
    assert "off_road_start" not in selection.reasons_for(1)

    unfiltered = select_adversary_candidates(scenario, ReGentSFilterConfig(filter_off_road_start=False))
    assert unfiltered.start_off_road.tolist() == [False, False, True]
    assert unfiltered.candidate_mask.tolist() == [False, True, True]
    assert unfiltered.reasons_for(2) == ()

    # An agent that enters mid log is placed by its own first valid state, not by
    # the zero-filled storage that precedes it.
    late_states = states.clone()
    late_states[0, 2, :2] = 0.0
    late_states[0, 2, 2:, 1] = 2.0
    late_valid = torch.ones((1, 3, 6), dtype=torch.bool)
    late_valid[0, 2, :2] = False
    late = select_adversary_candidates(make_scenario(late_states, late_valid, drivable_mask=drivable_mask))
    assert late.start_off_road.tolist() == [False, False, False]
