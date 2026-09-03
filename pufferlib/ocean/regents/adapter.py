"""Validated conversion from ``Drive.get_state()`` to Torch tensors."""

import math

import numpy as np
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.regents.state import (
    STATE_FEATURE_COUNT,
    STATE_HEADING,
    STATE_SPEED,
    STATE_STEERING,
    STATE_X,
    STATE_Y,
    DrivableAreaRaster,
    RasterTransform,
    ScenarioBatch,
)


DEFAULT_RASTER_RESOLUTION_METERS = 0.5
TIMESTEP_TOLERANCE_SECONDS = 1e-6
INVALID_AGENT_ID = -1
MAX_SCENARIO_AGENT_COUNT = 10_000
MAX_TRAJECTORY_TIMESTEP_COUNT = 9_999
MAX_BATCH_STATE_COUNT = 25_000_000
MAX_RASTER_PIXEL_COUNT = 25_000_000
MAX_ROAD_POINT_COUNT = 99_999


def _as_scenario_list(payload):
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and payload and all(isinstance(item, dict) for item in payload):
        return payload
    raise ValueError("Drive.get_state() must return a scenario dict or a non-empty list of scenario dicts")


def _torch_from_contiguous(array):
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise ValueError("Internal adapter error: Torch conversion requires a contiguous NumPy array")
    return torch.from_numpy(array)


def _validate_drive_contract(drive):
    if drive.simulation_mode != binding.SIMULATION_MODE_REPLAY:
        raise ValueError("ReGentS Stage 1 requires simulation_mode='replay'")
    if drive._action_type_flag != binding.ACTION_TYPE_CONTINUOUS:
        raise ValueError("ReGentS Stage 1 requires action_type='continuous'")
    if drive.dynamics_model_flag != binding.DYNAMICS_MODEL_CLASSIC:
        raise ValueError("ReGentS Stage 1 requires dynamics_model='classic'")
    if drive.init_step_spread:
        raise ValueError("ReGentS Stage 1 requires a fixed init_step (init_step_spread=False)")
    if drive.reward_conditioning or drive.reward_randomization:
        raise ValueError("ReGentS Stage 1 requires reward conditioning and randomization to be disabled")
    if not math.isfinite(drive.dt) or drive.dt <= 0:
        raise ValueError(f"Drive dt must be finite and positive, got {drive.dt}")
    if not math.isfinite(drive.base_max_speed_mps) or drive.base_max_speed_mps <= 0:
        raise ValueError("Drive base_max_speed_mps must be finite and positive")
    if not isinstance(drive.init_step, int) or drive.init_step < 0:
        raise ValueError("Drive init_step must be a non-negative integer")
    if not isinstance(drive.scenario_length, int) or drive.scenario_length <= 0:
        raise ValueError("Drive scenario_length must be a positive integer")


def _float_array(values, expected_length, field_name):
    if values is None:
        raise ValueError(f"get_state() is missing required trajectory field {field_name}")
    array = np.ascontiguousarray(values, dtype=np.float32)
    if array.shape != (expected_length,):
        raise ValueError(f"{field_name} must have length {expected_length}, got shape {array.shape}")
    return array


def _valid_array(values, expected_length):
    if values is None:
        raise ValueError("get_state() is missing required trajectory field log_valid")
    raw = np.ascontiguousarray(values, dtype=np.int32)
    if raw.shape != (expected_length,):
        raise ValueError(f"log_valid must have length {expected_length}, got shape {raw.shape}")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("log_valid may contain only 0 or 1")
    return np.ascontiguousarray(raw.astype(np.bool_, copy=False))


def _rasterize_drivable_area(scenario, resolution_meters):
    if not math.isfinite(resolution_meters) or resolution_meters <= 0:
        raise ValueError("Raster resolution must be finite and positive")
    map_corners = np.ascontiguousarray(scenario.get("map_corners"), dtype=np.float32)
    if map_corners.shape != (4,) or not np.isfinite(map_corners).all():
        raise ValueError("map_corners must contain finite [min_x, min_y, max_x, max_y]")
    min_x_meters, min_y_meters, max_x_meters, max_y_meters = map(float, map_corners)
    if min_x_meters >= max_x_meters or min_y_meters >= max_y_meters:
        raise ValueError(f"Invalid map_corners ordering: {map_corners.tolist()}")

    width = math.ceil((max_x_meters - min_x_meters) / resolution_meters) + 1
    height = math.ceil((max_y_meters - min_y_meters) / resolution_meters) + 1
    if width * height > MAX_RASTER_PIXEL_COUNT:
        raise ValueError(f"Raster dimensions {height}x{width} exceed the {MAX_RASTER_PIXEL_COUNT} pixel limit")
    transform = RasterTransform(min_x_meters, min_y_meters, resolution_meters, height, width)
    drivable = np.zeros((height, width), dtype=np.bool_)
    half_lane_width_meters = 0.5 * float(binding.LANE_WIDTH_METERS)
    drivable_types = {binding.ROAD_TYPE_LANE_FREEWAY, binding.ROAD_TYPE_LANE_SURFACE_STREET}

    road_elements = scenario.get("road_elements")
    if not isinstance(road_elements, list) or len(road_elements) != int(scenario["num_road_elements"]):
        raise ValueError("road_elements must match num_road_elements")
    for road_idx, road in enumerate(road_elements):
        if int(road.get("id", INVALID_AGENT_ID)) != road_idx:
            raise ValueError(f"Road element id must equal its stable array index {road_idx}")
        if int(road["type"]) not in drivable_types:
            continue
        point_count = int(road["segment_size"])
        if point_count < 0 or point_count > MAX_ROAD_POINT_COUNT:
            raise ValueError(f"Road element {road_idx} has invalid segment_size={point_count}")
        if point_count < 2:
            continue
        x_meters = _float_array(road.get("x"), point_count, f"road_elements[{road_idx}].x")
        y_meters = _float_array(road.get("y"), point_count, f"road_elements[{road_idx}].y")
        if not np.isfinite(x_meters).all() or not np.isfinite(y_meters).all():
            raise ValueError(f"Road element {road_idx} contains non-finite geometry")

        for segment_idx in range(point_count - 1):
            start_x = float(x_meters[segment_idx])
            start_y = float(y_meters[segment_idx])
            delta_x = float(x_meters[segment_idx + 1]) - start_x
            delta_y = float(y_meters[segment_idx + 1]) - start_y
            segment_length_squared = delta_x * delta_x + delta_y * delta_y
            segment_min_x = min(start_x, start_x + delta_x) - half_lane_width_meters
            segment_max_x = max(start_x, start_x + delta_x) + half_lane_width_meters
            segment_min_y = min(start_y, start_y + delta_y) - half_lane_width_meters
            segment_max_y = max(start_y, start_y + delta_y) + half_lane_width_meters
            min_column = max(0, math.floor((segment_min_x - min_x_meters) / resolution_meters))
            max_column = min(width - 1, math.ceil((segment_max_x - min_x_meters) / resolution_meters))
            min_row = max(0, math.floor((segment_min_y - min_y_meters) / resolution_meters))
            max_row = min(height - 1, math.ceil((segment_max_y - min_y_meters) / resolution_meters))
            if min_column > max_column or min_row > max_row:
                continue

            columns = np.arange(min_column, max_column + 1, dtype=np.float32)
            rows = np.arange(min_row, max_row + 1, dtype=np.float32)
            pixel_x = min_x_meters + columns[None, :] * resolution_meters
            pixel_y = min_y_meters + rows[:, None] * resolution_meters
            if segment_length_squared == 0.0:
                distance_squared = (pixel_x - start_x) ** 2 + (pixel_y - start_y) ** 2
            else:
                projection = (pixel_x - start_x) * delta_x + (pixel_y - start_y) * delta_y
                projection = np.clip(projection / segment_length_squared, 0.0, 1.0)
                distance_squared = (pixel_x - (start_x + projection * delta_x)) ** 2
                distance_squared += (pixel_y - (start_y + projection * delta_y)) ** 2
            drivable[min_row : max_row + 1, min_column : max_column + 1] |= (
                distance_squared <= half_lane_width_meters * half_lane_width_meters
            )

    if not drivable.any():
        raise ValueError("Scenario contains no rasterizable drivable freeway or surface-street lanes")
    return DrivableAreaRaster(_torch_from_contiguous(np.ascontiguousarray(drivable)), transform)


def export_drive_scenarios(drive, payload=None, raster_resolution_meters=DEFAULT_RASTER_RESOLUTION_METERS):
    """Export complete logged scenarios from a configured Drive instance.

    The payload defaults to one call to ``drive.get_state()``. Python lists are
    first materialized as contiguous NumPy arrays; each returned Torch tensor
    is then created through the explicit NumPy-to-Torch boundary.
    """
    _validate_drive_contract(drive)
    scenarios = _as_scenario_list(drive.get_state() if payload is None else payload)
    batch_count = len(scenarios)
    agent_counts = []
    time_counts = []
    for scenario_idx, scenario in enumerate(scenarios):
        agents = scenario.get("agents")
        if not isinstance(agents, list) or not agents:
            raise ValueError(f"Scenario {scenario_idx} must contain a non-empty agents list")
        agent_count = int(scenario.get("num_total_agents", -1))
        if agent_count != len(agents):
            raise ValueError(f"Scenario {scenario_idx} num_total_agents does not match agents")
        if agent_count > MAX_SCENARIO_AGENT_COUNT:
            raise ValueError(f"Scenario {scenario_idx} exceeds the {MAX_SCENARIO_AGENT_COUNT} agent limit")
        log_length = int(scenario.get("length", 0))
        if log_length <= 0 or log_length > MAX_TRAJECTORY_TIMESTEP_COUNT:
            raise ValueError(f"Scenario {scenario_idx} has invalid length={log_length}")
        agent_counts.append(agent_count)
        scenario_time_counts = [int(agent.get("trajectory_size", 0)) for agent in agents]
        if max(scenario_time_counts) > log_length:
            raise ValueError(f"Scenario {scenario_idx} agent trajectory exceeds scenario length={log_length}")
        time_counts.extend(scenario_time_counts)
    if not time_counts or min(time_counts) <= 0:
        raise ValueError("Every serialized agent must have a positive trajectory_size")

    max_agent_count = max(agent_counts)
    max_time_count = max(time_counts)
    if batch_count * max_agent_count * max_time_count > MAX_BATCH_STATE_COUNT:
        raise ValueError(f"Padded batch exceeds the {MAX_BATCH_STATE_COUNT} state limit")
    logged_state = np.zeros((batch_count, max_agent_count, max_time_count, STATE_FEATURE_COUNT), dtype=np.float32)
    state_valid = np.zeros((batch_count, max_agent_count, max_time_count), dtype=np.bool_)
    state_feature_valid = np.zeros_like(logged_state, dtype=np.bool_)
    logged_length_meters = np.zeros((batch_count, max_agent_count, max_time_count), dtype=np.float32)
    logged_width_meters = np.zeros((batch_count, max_agent_count, max_time_count), dtype=np.float32)
    current_state = np.zeros((batch_count, max_agent_count, STATE_FEATURE_COUNT), dtype=np.float32)
    current_valid = np.zeros((batch_count, max_agent_count), dtype=np.bool_)
    agent_present = np.zeros((batch_count, max_agent_count), dtype=np.bool_)
    agent_metadata_valid = np.zeros((batch_count, max_agent_count), dtype=np.bool_)
    active_agent_mask = np.zeros((batch_count, max_agent_count), dtype=np.bool_)
    agent_id = np.full((batch_count, max_agent_count), INVALID_AGENT_ID, dtype=np.int64)
    agent_type = np.zeros((batch_count, max_agent_count), dtype=np.int64)
    controller = np.zeros((batch_count, max_agent_count), dtype=np.int64)
    trajectory_length = np.zeros((batch_count, max_agent_count), dtype=np.int64)
    length_meters = np.zeros((batch_count, max_agent_count), dtype=np.float32)
    width_meters = np.zeros((batch_count, max_agent_count), dtype=np.float32)
    wheelbase_meters = np.zeros((batch_count, max_agent_count), dtype=np.float32)
    maximum_speed_mps = np.zeros((batch_count, max_agent_count), dtype=np.float32)
    scenario_ids = []
    dataset_names = []
    log_dt_seconds = np.empty(batch_count, dtype=np.float32)
    drivable_area_rasters = []

    for scenario_idx, scenario in enumerate(scenarios):
        log_dt = float(scenario.get("log_dt", math.nan))
        if not math.isfinite(log_dt) or log_dt <= 0:
            raise ValueError(f"Scenario {scenario_idx} log_dt must be finite and positive")
        if not math.isclose(log_dt, drive.dt, rel_tol=0.0, abs_tol=TIMESTEP_TOLERANCE_SECONDS):
            raise ValueError(f"Scenario {scenario_idx} log_dt={log_dt} does not match simulation dt={drive.dt}")
        if int(scenario.get("dynamics_model", -1)) != binding.DYNAMICS_MODEL_CLASSIC:
            raise ValueError(f"Scenario {scenario_idx} was not exported with classic dynamics")
        scenario_id = scenario.get("scenario_id")
        dataset_name = scenario.get("dataset_name")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"Scenario {scenario_idx} has no stable scenario_id")
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError(f"Scenario {scenario_idx} has no dataset_name")
        scenario_ids.append(scenario_id)
        dataset_names.append(dataset_name)
        log_dt_seconds[scenario_idx] = log_dt
        drivable_area_rasters.append(_rasterize_drivable_area(scenario, raster_resolution_meters))

        active_indices = scenario.get("active_agent_indices") or []
        active_index_set = set(map(int, active_indices))
        if len(active_index_set) != len(active_indices):
            raise ValueError(f"Scenario {scenario_idx} active_agent_indices contains duplicates")
        if any(agent_idx < 0 or agent_idx >= len(scenario["agents"]) for agent_idx in active_index_set):
            raise ValueError(f"Scenario {scenario_idx} active_agent_indices contains an out-of-range index")
        if int(scenario["active_agent_count"]) != len(active_indices):
            raise ValueError(f"Scenario {scenario_idx} active_agent_count does not match active_agent_indices")
        for stable_agent_idx, agent in enumerate(scenario["agents"]):
            if int(agent.get("id", INVALID_AGENT_ID)) != stable_agent_idx:
                raise ValueError(f"Scenario {scenario_idx} agent id must equal stable C array index {stable_agent_idx}")
            time_count = int(agent["trajectory_size"])
            valid = _valid_array(agent.get("log_valid"), time_count)
            x = _float_array(agent.get("log_trajectory_x"), time_count, "log_trajectory_x")
            y = _float_array(agent.get("log_trajectory_y"), time_count, "log_trajectory_y")
            heading = _float_array(agent.get("log_heading"), time_count, "log_heading")
            velocity_x = _float_array(agent.get("log_velocity_x"), time_count, "log_velocity_x")
            velocity_y = _float_array(agent.get("log_velocity_y"), time_count, "log_velocity_y")
            logged_length = _float_array(agent.get("log_length"), time_count, "log_length")
            logged_width = _float_array(agent.get("log_width"), time_count, "log_width")
            all_float_values = np.stack((x, y, heading, velocity_x, velocity_y, logged_length, logged_width), axis=-1)
            if not np.isfinite(all_float_values).all():
                raise ValueError(f"Scenario {scenario_idx} agent {stable_agent_idx} has non-finite logged data")
            if np.any(logged_length[valid] <= 0) or np.any(logged_width[valid] <= 0):
                raise ValueError(f"Scenario {scenario_idx} agent {stable_agent_idx} has non-positive valid dimensions")

            wrapped_heading = np.arctan2(np.sin(heading), np.cos(heading)).astype(np.float32)
            signed_speed = velocity_x * np.cos(wrapped_heading) + velocity_y * np.sin(wrapped_heading)
            logged_state[scenario_idx, stable_agent_idx, :time_count, STATE_X] = x
            logged_state[scenario_idx, stable_agent_idx, :time_count, STATE_Y] = y
            logged_state[scenario_idx, stable_agent_idx, :time_count, STATE_HEADING] = wrapped_heading
            logged_state[scenario_idx, stable_agent_idx, :time_count, STATE_SPEED] = signed_speed
            state_valid[scenario_idx, stable_agent_idx, :time_count] = valid
            state_feature_valid[scenario_idx, stable_agent_idx, :time_count, :STATE_STEERING] = valid[:, None]
            logged_length_meters[scenario_idx, stable_agent_idx, :time_count] = logged_length
            logged_width_meters[scenario_idx, stable_agent_idx, :time_count] = logged_width

            sim_values = np.asarray(
                [
                    agent["sim_x"],
                    agent["sim_y"],
                    agent["sim_heading"],
                    agent["sim_vx"],
                    agent["sim_vy"],
                    agent["sim_steering"],
                ],
                dtype=np.float32,
            )
            if not np.isfinite(sim_values).all():
                raise ValueError(f"Scenario {scenario_idx} agent {stable_agent_idx} has non-finite current state")
            sim_heading = np.float32(math.atan2(math.sin(float(sim_values[2])), math.cos(float(sim_values[2]))))
            current_state[scenario_idx, stable_agent_idx, STATE_X] = sim_values[0]
            current_state[scenario_idx, stable_agent_idx, STATE_Y] = sim_values[1]
            current_state[scenario_idx, stable_agent_idx, STATE_HEADING] = sim_heading
            current_state[scenario_idx, stable_agent_idx, STATE_SPEED] = sim_values[3] * math.cos(
                float(sim_heading)
            ) + sim_values[4] * math.sin(float(sim_heading))
            current_state[scenario_idx, stable_agent_idx, STATE_STEERING] = sim_values[5]
            serialized_current_valid = int(agent["sim_valid"])
            if serialized_current_valid not in (0, 1):
                raise ValueError(f"Scenario {scenario_idx} agent {stable_agent_idx} has invalid sim_valid")
            current_valid[scenario_idx, stable_agent_idx] = bool(serialized_current_valid)
            agent_present[scenario_idx, stable_agent_idx] = True
            active_agent_mask[scenario_idx, stable_agent_idx] = stable_agent_idx in active_index_set
            agent_id[scenario_idx, stable_agent_idx] = stable_agent_idx
            serialized_agent_type = int(agent["type"])
            if serialized_agent_type not in (
                binding.AGENT_TYPE_UNKNOWN,
                binding.AGENT_TYPE_VEHICLE,
                binding.AGENT_TYPE_PEDESTRIAN,
                binding.AGENT_TYPE_CYCLIST,
            ):
                raise ValueError(f"Scenario {scenario_idx} agent {stable_agent_idx} has invalid type")
            serialized_controller = int(agent["controller"])
            if serialized_controller not in (
                binding.CONTROLLER_STATIC,
                binding.CONTROLLER_POLICY,
                binding.CONTROLLER_REPLAY,
                binding.CONTROLLER_IDM,
            ):
                raise ValueError(f"Scenario {scenario_idx} agent {stable_agent_idx} has invalid controller")
            agent_type[scenario_idx, stable_agent_idx] = serialized_agent_type
            controller[scenario_idx, stable_agent_idx] = serialized_controller
            trajectory_length[scenario_idx, stable_agent_idx] = time_count
            if valid.any():
                reference_timestep = int(np.flatnonzero(valid)[0])
                agent_metadata_valid[scenario_idx, stable_agent_idx] = True
                length_meters[scenario_idx, stable_agent_idx] = logged_length[reference_timestep]
                width_meters[scenario_idx, stable_agent_idx] = logged_width[reference_timestep]
                wheelbase_meters[scenario_idx, stable_agent_idx] = (
                    binding.WHEELBASE_LENGTH_RATIO * logged_length[reference_timestep]
                )
            maximum_speed_mps[scenario_idx, stable_agent_idx] = drive.base_max_speed_mps

    transition_valid = np.ascontiguousarray(state_valid[:, :, :-1] & state_valid[:, :, 1:])
    ego_mask = np.zeros_like(agent_present)
    ego_mask[:, 0] = agent_present[:, 0]
    vehicle_mask = np.ascontiguousarray(agent_present & (agent_type == binding.AGENT_TYPE_VEHICLE))
    candidate_adversary_mask = np.ascontiguousarray(vehicle_mask & agent_metadata_valid & ~ego_mask)

    return ScenarioBatch(
        logged_state=_torch_from_contiguous(np.ascontiguousarray(logged_state)),
        state_valid=_torch_from_contiguous(np.ascontiguousarray(state_valid)),
        state_feature_valid=_torch_from_contiguous(np.ascontiguousarray(state_feature_valid)),
        transition_valid=_torch_from_contiguous(transition_valid),
        current_state=_torch_from_contiguous(np.ascontiguousarray(current_state)),
        current_valid=_torch_from_contiguous(np.ascontiguousarray(current_valid)),
        agent_present=_torch_from_contiguous(np.ascontiguousarray(agent_present)),
        agent_metadata_valid=_torch_from_contiguous(np.ascontiguousarray(agent_metadata_valid)),
        active_agent_mask=_torch_from_contiguous(np.ascontiguousarray(active_agent_mask)),
        agent_id=_torch_from_contiguous(np.ascontiguousarray(agent_id)),
        agent_type=_torch_from_contiguous(np.ascontiguousarray(agent_type)),
        controller=_torch_from_contiguous(np.ascontiguousarray(controller)),
        trajectory_length=_torch_from_contiguous(np.ascontiguousarray(trajectory_length)),
        ego_mask=_torch_from_contiguous(np.ascontiguousarray(ego_mask)),
        vehicle_mask=_torch_from_contiguous(vehicle_mask),
        candidate_adversary_mask=_torch_from_contiguous(candidate_adversary_mask),
        logged_length_meters=_torch_from_contiguous(np.ascontiguousarray(logged_length_meters)),
        logged_width_meters=_torch_from_contiguous(np.ascontiguousarray(logged_width_meters)),
        length_meters=_torch_from_contiguous(np.ascontiguousarray(length_meters)),
        width_meters=_torch_from_contiguous(np.ascontiguousarray(width_meters)),
        wheelbase_meters=_torch_from_contiguous(np.ascontiguousarray(wheelbase_meters)),
        maximum_speed_mps=_torch_from_contiguous(np.ascontiguousarray(maximum_speed_mps)),
        scenario_ids=tuple(scenario_ids),
        dataset_names=tuple(dataset_names),
        log_dt_seconds=_torch_from_contiguous(np.ascontiguousarray(log_dt_seconds)),
        dt_seconds=float(drive.dt),
        init_step=drive.init_step,
        scenario_length=drive.scenario_length,
        drivable_area_rasters=tuple(drivable_area_rasters),
    )
