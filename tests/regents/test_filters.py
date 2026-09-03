import math

import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents.filters import (
    CandidateFilterReason,
    ReGentSFilterConfig,
    SceneFilterReason,
    front_divergence_mask,
    select_adversary_candidates,
)
from pufferlib.ocean.regents.state import DrivableAreaRaster, RasterTransform, ScenarioBatch


def make_scenario(states, valid=None, agent_types=None, drivable_mask=None):
    states = states.to(torch.float32)
    batch_count, agent_count, time_count, _ = states.shape
    assert batch_count == 1
    if valid is None:
        valid = torch.ones((1, agent_count, time_count), dtype=torch.bool)
    if agent_types is None:
        agent_types = torch.full((1, agent_count), binding.AGENT_TYPE_VEHICLE, dtype=torch.int64)
    present = torch.ones((1, agent_count), dtype=torch.bool)
    ego_mask = torch.zeros_like(present)
    ego_mask[:, 0] = True
    vehicle_mask = present & (agent_types == binding.AGENT_TYPE_VEHICLE)
    dimensions = torch.full((1, agent_count), 2.0, dtype=torch.float32)
    lengths = torch.full((1, agent_count), 4.0, dtype=torch.float32)
    if drivable_mask is None:
        drivable_mask = torch.ones((101, 101), dtype=torch.bool)
    raster = DrivableAreaRaster(drivable_mask, RasterTransform(-50.0, -50.0, 1.0, *drivable_mask.shape))
    feature_valid = valid[..., None].expand_as(states).clone()
    return ScenarioBatch(
        logged_state=states,
        state_valid=valid,
        state_feature_valid=feature_valid,
        transition_valid=valid[:, :, :-1] & valid[:, :, 1:],
        current_state=states[:, :, 0].clone(),
        current_valid=valid[:, :, 0].clone(),
        agent_present=present,
        agent_metadata_valid=present.clone(),
        active_agent_mask=present.clone(),
        agent_id=torch.arange(agent_count, dtype=torch.int64)[None],
        agent_type=agent_types,
        controller=torch.full((1, agent_count), binding.CONTROLLER_REPLAY, dtype=torch.int64),
        trajectory_length=torch.full((1, agent_count), time_count, dtype=torch.int64),
        ego_mask=ego_mask,
        vehicle_mask=vehicle_mask,
        candidate_adversary_mask=vehicle_mask & ~ego_mask,
        logged_length_meters=lengths[..., None].expand(1, agent_count, time_count).clone(),
        logged_width_meters=dimensions[..., None].expand(1, agent_count, time_count).clone(),
        length_meters=lengths,
        width_meters=dimensions,
        wheelbase_meters=0.6 * lengths,
        maximum_speed_mps=torch.full((1, agent_count), 20.0, dtype=torch.float32),
        scenario_ids=("synthetic",),
        dataset_names=("test",),
        log_dt_seconds=torch.tensor([0.1], dtype=torch.float32),
        dt_seconds=0.1,
        init_step=0,
        scenario_length=time_count,
        drivable_area_rasters=(raster,),
    )


def _linear_track(x_start, y, speed, time_count=6, heading=0.0, dt=0.1):
    state = torch.zeros((time_count, 5), dtype=torch.float32)
    state[:, 0] = x_start + torch.arange(time_count) * speed * dt
    state[:, 1] = y
    state[:, 2] = heading
    state[:, 3] = speed
    return state


def test_candidate_filter_records_every_agent_reason():
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
        ReGentSFilterConfig(minimum_valid_transition_count=3),
    )

    assert selection.scene_eligible.tolist() == [True]
    assert selection.candidate_mask.tolist() == [[False, True, False, False, False, False]]
    assert "ego" in selection.reasons_for(0, 0)
    assert "static" in selection.reasons_for(0, 2)
    assert "rear_sector" in selection.reasons_for(0, 3)
    assert "non_vehicle" in selection.reasons_for(0, 4)
    assert "insufficient_valid_transitions" in selection.reasons_for(0, 5)


def test_original_collision_is_labeled_per_agent_against_the_ego():
    states = torch.stack((_linear_track(0.0, 0.0, 2.0), _linear_track(3.0, 0.0, 2.0)))[None]
    selection = select_adversary_candidates(make_scenario(states))
    assert selection.original_collision.tolist() == [[False, True]]
    assert selection.original_collision_timestep.tolist() == [[-1, 0]]
    assert selection.scene_reasons_for(0) == ("original_collision", "no_candidate")
    assert int(selection.filter_reason_bits[0, 1]) & int(CandidateFilterReason.ORIGINAL_COLLISION)


def test_background_overlap_away_from_the_ego_keeps_the_scene_and_its_candidates():
    # Two logged backgrounds overlapping each other is not something the C simulator
    # ever scores, so it must not disqualify the scene or an uninvolved candidate.
    states = torch.stack(
        (
            _linear_track(0.0, 0.0, 2.0),
            _linear_track(40.0, 30.0, 2.0),
            _linear_track(43.0, 30.0, 2.0),
        )
    )[None]
    selection = select_adversary_candidates(make_scenario(states))

    assert selection.original_collision.tolist() == [[False, False, False]]
    assert selection.scene_eligible.tolist() == [True]
    assert selection.scene_reasons_for(0) == ()
    assert selection.candidate_mask.tolist() == [[False, True, True]]


def test_only_the_ego_colliding_candidate_is_excluded():
    states = torch.stack(
        (
            _linear_track(0.0, 0.0, 2.0),
            _linear_track(3.0, 0.0, 2.0),
            _linear_track(8.0, 4.0, 2.0),
        )
    )[None]
    selection = select_adversary_candidates(make_scenario(states))

    assert selection.original_collision.tolist() == [[False, True, False]]
    assert selection.scene_eligible.tolist() == [True]
    assert selection.candidate_mask.tolist() == [[False, False, True]]
    assert "original_collision" in selection.reasons_for(0, 1)
    assert "original_collision" not in selection.reasons_for(0, 2)


def test_caller_unsuitable_scene_is_recorded_without_mutating_inputs():
    states = torch.stack((_linear_track(0.0, 0.0, 2.0), _linear_track(8.0, 4.0, 2.0)))[None]
    scenario = make_scenario(states)
    selection = select_adversary_candidates(
        scenario,
        scene_suitable=torch.tensor([False]),
    )
    assert not selection.scene_eligible[0]
    assert int(selection.scene_reason_bits[0]) & int(SceneFilterReason.CALLER_UNSUITABLE)
    assert "scene_unsuitable" in selection.reasons_for(0, 1)


def test_rear_sector_uses_strict_angular_and_fraction_boundaries():
    time_count = 6
    ego = _linear_track(0.0, 0.0, 1.0, time_count)
    exact_angle = 7.0 * math.pi / 8.0
    exact = _linear_track(10.0 * math.cos(exact_angle), 10.0 * math.sin(exact_angle), 1.0, time_count)
    outside = _linear_track(
        10.0 * math.cos(exact_angle + 1e-4),
        10.0 * math.sin(exact_angle + 1e-4),
        1.0,
        time_count,
    )
    states = torch.stack((ego, exact, outside))[None]
    selection = select_adversary_candidates(
        make_scenario(states),
        ReGentSFilterConfig(
            static_displacement_threshold_meters=0.0,
            static_speed_threshold_mps=0.0,
            rear_sector_fraction=0.8,
        ),
    )
    assert "rear_sector" not in selection.reasons_for(0, 1)
    assert "rear_sector" in selection.reasons_for(0, 2)


def _front_states(position_angle, yaw, time_count=5):
    states = torch.zeros((1, 2, time_count, 5), dtype=torch.float32)
    states[0, 1, :, 0] = 10.0 * math.cos(position_angle)
    states[0, 1, :, 1] = 10.0 * math.sin(position_angle)
    states[0, 1, :, 2] = yaw
    return states


def test_front_divergence_covers_both_sides_and_strict_pi_over_eight_boundaries():
    valid = torch.ones((1, 2, 5), dtype=torch.bool)
    ego_mask = torch.tensor([[True, False]])
    candidate_mask = torch.tensor([[False, True]])
    epsilon = 1e-4
    for sign in (-1.0, 1.0):
        inside = front_divergence_mask(
            _front_states(sign * math.pi / 16.0, sign * (math.pi / 8.0 - epsilon)),
            valid,
            ego_mask,
            candidate_mask,
        )
        yaw_boundary = front_divergence_mask(
            _front_states(sign * math.pi / 16.0, sign * math.pi / 8.0),
            valid,
            ego_mask,
            candidate_mask,
        )
        position_boundary = front_divergence_mask(
            _front_states(sign * math.pi / 8.0, sign * (math.pi / 8.0 - epsilon)),
            valid,
            ego_mask,
            candidate_mask,
        )
        assert inside.tolist() == [[False, True]]
        assert yaw_boundary.tolist() == [[False, False]]
        assert position_boundary.tolist() == [[False, False]]


def test_front_divergence_fraction_threshold_is_strict():
    states = _front_states(math.pi / 16.0, math.pi / 8.0 - 1e-4, time_count=4)
    states[0, 1, 2:, 1] *= -1.0
    valid = torch.ones((1, 2, 4), dtype=torch.bool)
    masks = (torch.tensor([[True, False]]), torch.tensor([[False, True]]))
    assert not front_divergence_mask(states, valid, *masks, tau_front=0.5)[0, 1]
    assert front_divergence_mask(states, valid, *masks, tau_front=0.49)[0, 1]
