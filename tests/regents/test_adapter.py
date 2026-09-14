import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.adapter import (
    TIMESTEP_TOLERANCE_SECONDS,
    _rasterize_drivable_area,
    export_drive_scenarios,
)
from pufferlib.ocean.regents.state import STATE_HEADING, STATE_SPEED, STATE_STEERING, STATE_X, STATE_Y


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_MAP = min((REPO_ROOT / "pufferlib/resources/drive/binaries/sdc_replay_test").glob("*.bin"))


def _drive_kwargs():
    return {
        "map_dir": str(FIXTURE_MAP),
        "num_maps": 1,
        "num_agents": 1,
        "min_agents_per_env": 1,
        "max_agents_per_env": 1,
        "num_eval_scenarios": 1,
        "max_scenarios_per_batch": 1,
        "eval_map_indices": [0],
        "eval_scenario_seeds": [42],
        "seed": 42,
        "simulation_mode": "replay",
        "eval_mode": True,
        "control_mode": "control_sdc_only",
        "sdc_controller": "idm",
        "non_sdc_controller": "replay",
        "non_vehicle_controller": "replay",
        "action_type": "continuous",
        "dynamics_model": "classic",
        "dt": 0.1,
        "scenario_length": 16,
        "resample_frequency": 16,
        "init_step": 0,
        "init_step_spread": False,
        "reward_conditioning": False,
        "reward_randomization": False,
        "use_neighbor_cache": 0,
    }


@pytest.fixture
def drive_and_payload():
    drive = Drive(**_drive_kwargs())
    drive.reset()
    state = drive.get_state()
    payload = state[0] if isinstance(state, list) else state
    try:
        yield drive, payload
    finally:
        drive.close()


def _tensor_bytes(batch):
    tensor_names = (
        "logged_state",
        "state_valid",
        "state_feature_valid",
        "transition_valid",
        "current_state",
        "agent_present",
        "agent_metadata_valid",
        "agent_id",
        "trajectory_length",
        "ego_mask",
        "vehicle_mask",
        "candidate_adversary_mask",
        "logged_length_meters",
        "logged_width_meters",
        "length_meters",
        "width_meters",
        "wheelbase_meters",
        "maximum_speed_mps",
    )
    tensor_payload = tuple(getattr(batch, name).numpy().tobytes() for name in tensor_names)
    return tensor_payload + (batch.drivable_area_raster.mask.numpy().tobytes(), repr(batch.log_dt_seconds))


def test_real_export_schema_identity_coordinates_and_determinism(drive_and_payload):
    """Shapes, dtypes, stable identity, centered frame, raster agreement, and byte determinism."""
    drive, payload = drive_and_payload
    batch = export_drive_scenarios(drive, payload=payload, raster_resolution_meters=1.0)

    assert batch.logged_state.shape == (payload["num_total_agents"], payload["length"], 5)
    assert batch.logged_state.dtype == torch.float32
    assert batch.state_valid.dtype == torch.bool
    assert torch.equal(batch.agent_id, torch.arange(payload["num_total_agents"]))
    assert batch.ego_mask[0]
    assert batch.ego_mask.sum() == 1
    assert batch.vehicle_mask[0]
    assert not batch.candidate_adversary_mask[0]
    assert batch.logged_length_meters.shape == batch.state_valid.shape
    assert batch.logged_width_meters.shape == batch.state_valid.shape
    assert batch.logged_length_meters[0, 0] == pytest.approx(payload["agents"][0]["log_length"][0])
    assert batch.coordinate_frame == "scenario_centered_cartesian"
    assert abs(batch.logged_state[0, 0, STATE_X]) < 1e-3
    assert abs(batch.logged_state[0, 0, STATE_Y]) < 1e-3
    assert torch.all(batch.logged_state[..., STATE_HEADING].abs() <= torch.pi)
    assert not batch.state_feature_valid[..., STATE_STEERING].any()
    assert torch.equal(batch.transition_valid, batch.state_valid[:, :-1] & batch.state_valid[:, 1:])

    # Speed is longitudinal velocity projected onto heading, logged and current alike.
    first_agent = payload["agents"][0]
    expected_logged_speed = first_agent["log_velocity_x"][0] * np.cos(first_agent["log_heading"][0])
    expected_logged_speed += first_agent["log_velocity_y"][0] * np.sin(first_agent["log_heading"][0])
    assert batch.logged_state[0, 0, STATE_SPEED] == pytest.approx(expected_logged_speed)
    expected_current_speed = first_agent["sim_vx"] * np.cos(first_agent["sim_heading"])
    expected_current_speed += first_agent["sim_vy"] * np.sin(first_agent["sim_heading"])
    assert batch.current_state[0, STATE_SPEED] == pytest.approx(expected_current_speed)
    assert batch.current_state[0, STATE_STEERING] == pytest.approx(first_agent["sim_steering"])

    # Fully invalid tracks keep their slot but never become candidates.
    invalid_track_indices = torch.where(~batch.state_valid.any(dim=-1))[0]
    assert invalid_track_indices.numel() > 0
    assert batch.agent_present[invalid_track_indices].all()
    assert not batch.agent_metadata_valid[invalid_track_indices].any()
    assert not batch.candidate_adversary_mask[invalid_track_indices].any()
    assert torch.equal(batch.agent_id[invalid_track_indices], invalid_track_indices)

    # A non-off-road ego must land on drivable raster cells.
    fine_batch = export_drive_scenarios(drive, payload=payload, raster_resolution_meters=0.5)
    raster = fine_batch.drivable_area_raster
    column, row = (
        raster.transform.world_to_grid(fine_batch.current_state[0, [STATE_X, STATE_Y]]).round().to(torch.int64)
    )
    assert payload["agents"][0]["metrics_array"][1] == 0.0
    assert raster.mask[row, column]

    exports = []
    for _ in range(2):
        repeat_drive = Drive(**_drive_kwargs())
        try:
            repeat_drive.reset()
            exports.append(export_drive_scenarios(repeat_drive, raster_resolution_meters=2.0))
        finally:
            repeat_drive.close()
    assert exports[0].scenario_id == exports[1].scenario_id
    assert _tensor_bytes(exports[0]) == _tensor_bytes(exports[1])


def test_export_rasterizes_lanes_and_rejects_unsupported_modes(drive_and_payload):
    """One scenario per export, the raster transform convention, and every mode guard."""
    drive, payload = drive_and_payload
    # ReGentS optimizes one scenario at a time, so a multi-scenario payload is refused
    # rather than padded into a batch.
    second = copy.deepcopy(payload)
    second["scenario_id"] = f"{payload['scenario_id']}-copy"
    with pytest.raises(ValueError, match="exactly one Drive scenario"):
        export_drive_scenarios(drive, payload=[payload, second], raster_resolution_meters=2.0)

    # One lane between two road edges. The edges are wound in opposite directions, as
    # exported maps are, so the drivable side is found by orienting them on the lane.
    lane_between_edges = {
        "map_corners": [-2.0, -2.0, 2.0, 2.0],
        "num_road_elements": 3,
        "road_elements": [
            {
                "id": 0,
                "type": binding.ROAD_TYPE_LANE_SURFACE_STREET,
                "segment_size": 2,
                "x": [-2.0, 2.0],
                "y": [0.0, 0.0],
            },
            {
                "id": 1,
                "type": binding.ROAD_TYPE_ROAD_EDGE_BOUNDARY,
                "segment_size": 2,
                "x": [-2.0, 2.0],
                "y": [1.0, 1.0],
            },
            {
                "id": 2,
                "type": binding.ROAD_TYPE_ROAD_EDGE_BOUNDARY,
                "segment_size": 2,
                "x": [-2.0, 2.0],
                "y": [-1.0, -1.0],
            },
        ],
    }
    raster = _rasterize_drivable_area(lane_between_edges, resolution_meters=1.0)
    assert raster.mask.shape == (5, 5)
    assert raster.mask[2, 2] and raster.mask[1, 2] and not raster.mask[0, 2]
    assert raster.mask[1:4, 2].all() and not raster.mask[4, 2]
    xy = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
    assert torch.equal(raster.transform.world_to_grid(xy), torch.tensor([[0.0, 0.0], [4.0, 4.0]]))
    assert torch.equal(raster.transform.world_to_normalized_grid(xy), torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))

    # The lane corridor still covers what the edge sign test mislabels, and a map
    # without road edges cannot be bounded at all.
    edgeless = copy.deepcopy(lane_between_edges)
    edgeless["road_elements"] = edgeless["road_elements"][:1]
    edgeless["num_road_elements"] = 1
    with pytest.raises(ValueError, match="no road-edge polylines"):
        _rasterize_drivable_area(edgeless, resolution_meters=1.0)

    mismatched = copy.deepcopy(payload)
    mismatched["log_dt"] = 0.2
    with pytest.raises(ValueError, match="does not match simulation dt"):
        export_drive_scenarios(drive, payload=mismatched, raster_resolution_meters=2.0)

    wod_rounded_timestep = copy.deepcopy(payload)
    wod_rounded_timestep["log_dt"] = 0.099
    rounded = export_drive_scenarios(drive, payload=wod_rounded_timestep, raster_resolution_meters=2.0)
    assert rounded.log_dt_seconds == pytest.approx(0.099)
    assert TIMESTEP_TOLERANCE_SECONDS == pytest.approx(1.01e-3)

    unsupported_modes = (
        ("simulation_mode", binding.SIMULATION_MODE_GIGAFLOW, "simulation_mode='replay'"),
        ("_action_type_flag", binding.ACTION_TYPE_DISCRETE, "action_type='continuous'"),
        ("dynamics_model_flag", -1, "dynamics_model='classic' or 'jerk'"),
        ("init_step_spread", True, "fixed init_step"),
        ("reward_conditioning", True, "conditioning and randomization"),
    )
    for attribute, value, message in unsupported_modes:
        original = getattr(drive, attribute)
        setattr(drive, attribute, value)
        try:
            with pytest.raises(ValueError, match=message):
                export_drive_scenarios(drive, payload=payload, raster_resolution_meters=2.0)
        finally:
            setattr(drive, attribute, original)

    # A jerk env is admissible: it governs the ego controller only, while injected
    # adversaries always integrate the classic bicycle model.
    original_dynamics = drive.dynamics_model_flag
    drive.dynamics_model_flag = binding.DYNAMICS_MODEL_JERK
    jerk_payload = copy.deepcopy(payload)
    jerk_payload["dynamics_model"] = binding.DYNAMICS_MODEL_JERK
    try:
        jerk_scenario = export_drive_scenarios(drive, payload=jerk_payload, raster_resolution_meters=2.0)
        assert jerk_scenario.max_agent_count == payload["num_total_agents"]
    finally:
        drive.dynamics_model_flag = original_dynamics
