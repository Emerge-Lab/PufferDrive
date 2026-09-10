"""Shared scenario builders for the ReGentS tests."""

import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.regents.adapter import export_drive_scenarios
from pufferlib.ocean.regents.state import (
    STATE_FEATURE_COUNT,
    STATE_STEERING,
    DrivableAreaRaster,
    RasterTransform,
    Scenario,
)
from tests.regents.real_fixtures import NUPLAN_MAP_DIR, REGENTS_AUDIT_SCENARIO_IDS, resolve_nuplan_scenarios


def _scenario_batch(states, state_valid=None, steering_observed=True, wheelbase_meters=2.7, maximum_speed_mps=20.0):
    """Build a Scenario around `[agent, time, 5]` states.

    A leading singleton batch axis is accepted and squeezed, so the many call sites
    written against the old batched contract keep working unchanged.
    """
    if states.ndim == 4:
        if states.shape[0] != 1:
            raise ValueError("test states carry at most one scenario")
        states = states[0]
        if state_valid is not None and state_valid.ndim == 3:
            state_valid = state_valid[0]
    if states.ndim != 3 or states.shape[-1] != STATE_FEATURE_COUNT:
        raise ValueError("test states must have shape [agent, time, 5]")
    agent_count, time_count, _ = states.shape
    if state_valid is None:
        state_valid = torch.ones((agent_count, time_count), dtype=torch.bool)
    feature_valid = state_valid[..., None].expand_as(states).clone()
    feature_valid[..., STATE_STEERING] = state_valid & steering_observed
    transition_valid = state_valid[..., :-1] & state_valid[..., 1:]
    agent_shape = (agent_count,)
    state_shape = (agent_count, time_count)
    present = torch.ones(agent_shape, dtype=torch.bool)
    vehicle = torch.ones(agent_shape, dtype=torch.bool)
    metadata_valid = torch.ones(agent_shape, dtype=torch.bool)
    length = torch.full(agent_shape, wheelbase_meters / float(binding.WHEELBASE_LENGTH_RATIO))
    raster = DrivableAreaRaster(
        torch.ones((2, 2), dtype=torch.bool),
        RasterTransform(0.0, 0.0, 1.0, 2, 2),
    )
    return Scenario(
        logged_state=states,
        state_valid=state_valid,
        state_feature_valid=feature_valid,
        transition_valid=transition_valid,
        current_state=states[:, 0, :].clone(),
        current_valid=present.clone(),
        agent_present=present,
        agent_metadata_valid=metadata_valid,
        active_agent_mask=present.clone(),
        agent_id=torch.arange(agent_count, dtype=torch.int64),
        agent_type=torch.full(agent_shape, binding.AGENT_TYPE_VEHICLE, dtype=torch.int64),
        controller=torch.full(agent_shape, binding.CONTROLLER_REPLAY, dtype=torch.int64),
        trajectory_length=torch.full(agent_shape, time_count, dtype=torch.int64),
        ego_mask=torch.zeros(agent_shape, dtype=torch.bool),
        vehicle_mask=vehicle,
        candidate_adversary_mask=vehicle.clone(),
        logged_length_meters=length[:, None].expand(state_shape).clone(),
        logged_width_meters=torch.full(state_shape, 2.0),
        length_meters=length,
        width_meters=torch.full(agent_shape, 2.0),
        wheelbase_meters=torch.full(agent_shape, wheelbase_meters),
        maximum_speed_mps=torch.full(agent_shape, maximum_speed_mps),
        scenario_id="synthetic-0",
        dataset_name="synthetic",
        log_dt_seconds=0.1,
        dt_seconds=0.1,
        init_step=0,
        scenario_length=time_count,
        drivable_area_raster=raster,
    )


def _real_scenario(map_idx):
    if map_idx < 0 or map_idx >= len(REGENTS_AUDIT_SCENARIO_IDS):
        raise ValueError(f"Pinned ReGentS fixture index must be in [0, {len(REGENTS_AUDIT_SCENARIO_IDS)})")
    map_paths, map_indices, _ = resolve_nuplan_scenarios()
    drive = Drive(
        map_dir=str(NUPLAN_MAP_DIR),
        num_maps=len(map_paths),
        num_agents=1,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_eval_scenarios=1,
        max_scenarios_per_batch=1,
        eval_map_indices=[map_indices[map_idx]],
        eval_scenario_seeds=[42 + map_idx],
        seed=42,
        simulation_mode="replay",
        eval_mode=True,
        control_mode="control_sdc_only",
        sdc_controller="idm",
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
    try:
        drive.reset()
        return export_drive_scenarios(drive, raster_resolution_meters=5.0)
    finally:
        drive.close()


@pytest.fixture(scope="session")
def synthetic_scenario_batch():
    """Build a single-vehicle Scenario around explicit `[agent, time, 5]` states."""
    return _scenario_batch


@pytest.fixture(scope="session")
def real_scenarios():
    """Exported NuPlan scenarios keyed by map index, built once per session."""
    cache = {}

    def load(map_idx):
        if map_idx not in cache:
            cache[map_idx] = _real_scenario(map_idx)
        return cache[map_idx]

    return load
