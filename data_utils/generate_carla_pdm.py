#!/usr/bin/env python3
"""Generate deterministic, segmented PDM replays from CARLA map binaries.

Each map is instantiated once and simulated continuously. The captured states
are split into replay binaries with an overlapping state at segment boundaries.
"""

import argparse
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive

try:
    from data_utils.mirror_map_bin import read_bin, write_bin
except ModuleNotFoundError:  # Direct execution adds data_utils/, not the repository root.
    from mirror_map_bin import read_bin, write_bin


DEFAULT_INPUT = Path("pufferlib/resources/drive/binaries/carla")
DEFAULT_OUTPUT = Path("pufferlib/resources/drive/binaries/carla_generated_pdm")
DEFAULT_AGENT_COUNT = 150
DEFAULT_TRANSITION_COUNT = 1000
DEFAULT_SEGMENT_TRANSITION_COUNT = 100
DEFAULT_BASE_SEED = 42
LOG_DT_SECONDS = 0.1
INITIAL_SPEED_MPS = 0.0
PDM_HORIZON_SECONDS = 4.0
PDM_PLANNING_DT_SECONDS = 0.1
DATASET_NAME = "carla_generated_pdm"
SCENARIO_ID_BYTES = 128
DATASET_NAME_BYTES = 32

AGENT_X_IDX = 0
AGENT_Y_IDX = 1
AGENT_Z_IDX = 2
AGENT_HEADING_IDX = 3
AGENT_LENGTH_IDX = 4
AGENT_WIDTH_IDX = 5
AGENT_SPEED_IDX = 6
AGENT_ID_IDX = 0
AGENT_TYPE_IDX = 1
AGENT_VALID_IDX = 2
AGENT_ACTIVE_IDX = 3
AGENT_REMOVED_IDX = 5
TRAFFIC_VALID_IDX = 0
TRAFFIC_TYPE_IDX = 1
TRAFFIC_STATE_IDX = 2


@dataclass(frozen=True)
class GenerationSummary:
    source_path: Path
    output_path: Path
    scenario_id: str
    seed: int
    agent_count: int
    sample_count: int
    sha256: str


def _fixed_string(value: str, byte_count: int) -> bytes:
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) >= byte_count:
        raise ValueError(f"{value!r} must encode to between 1 and {byte_count - 1} bytes")
    return encoded.ljust(byte_count, b"\0")


def _decoded_fixed_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files_equal(first_path: Path, second_path: Path) -> bool:
    if first_path.stat().st_size != second_path.stat().st_size:
        return False
    with first_path.open("rb") as first_file, second_path.open("rb") as second_file:
        while True:
            first_block = first_file.read(1024 * 1024)
            second_block = second_file.read(1024 * 1024)
            if first_block != second_block:
                return False
            if not first_block:
                return True


def _single_scenario(state):
    if isinstance(state, dict):
        return state
    if isinstance(state, list) and len(state) == 1:
        return state[0]
    raise RuntimeError("CARLA PDM generation requires exactly one Drive environment")


def _create_generation_drive(source_path: Path, agent_count: int, sample_count: int, seed: int) -> Drive:
    return Drive(
        map_dir=str(source_path),
        num_maps=1,
        num_agents=agent_count,
        min_agents_per_env=agent_count,
        max_agents_per_env=agent_count,
        num_eval_scenarios=1,
        max_scenarios_per_batch=1,
        eval_map_indices=[0],
        eval_scenario_seeds=[seed],
        eval_agent_counts=[agent_count],
        eval_agent_count_mode="fixed",
        seed=seed,
        simulation_mode="gigaflow",
        eval_mode=True,
        compute_eval_metrics=False,
        control_mode="control_agents",
        sdc_controller="pdm",
        non_sdc_controller="pdm",
        non_vehicle_controller="auto",
        action_type="continuous",
        dynamics_model="classic",
        dt=LOG_DT_SECONDS,
        pdm_horizon=PDM_HORIZON_SECONDS,
        pdm_planning_dt=PDM_PLANNING_DT_SECONDS,
        spawn_initial_speed=INITIAL_SPEED_MPS,
        spawn_speed_mode="fixed",
        gigaflow_spawn_mode="uniform",
        scenario_length=sample_count,
        resample_frequency=0,
        init_step=0,
        init_step_spread=False,
        init_mode="create_all_valid",
        reward_conditioning=False,
        reward_randomization=False,
        reward_log_sampling=False,
        collision_behavior="ignore",
        offroad_behavior="ignore",
        traffic_light_behavior="ignore",
        traffic_light_junction_phases=True,
        target_infraction_behavior="normal",
        termination_mode=False,
        adversarial_termination_mode="disabled",
        terminate_on_goal=False,
        goal_source="route",
        goal_regen_mode="finite",
        obs_dropout_lane=0.0,
        obs_dropout_boundary=0.0,
        partner_blindness_prob=0.0,
        partner_blindness_trigger_prob=0.0,
        phantom_braking_prob=0.0,
        phantom_braking_trigger_prob=0.0,
    )


def _validate_boundary_scenario(
    scenario: dict, source_path: Path, agent_count: int, sample_idx: int
) -> tuple[dict, ...]:
    if scenario["timestep"] != sample_idx:
        raise RuntimeError(
            f"{source_path.name} reset or skipped a state at sample {sample_idx}: "
            f"reported timestep {scenario['timestep']}"
        )
    if scenario["num_total_agents"] != agent_count:
        raise RuntimeError(
            f"{source_path.name} created {scenario['num_total_agents']} agent slots, expected {agent_count}"
        )
    if scenario["active_agent_count"] != agent_count:
        raise RuntimeError(
            f"{source_path.name} has {scenario['active_agent_count']} active agents at sample {sample_idx}, "
            f"expected {agent_count}"
        )
    if scenario["active_agent_indices"] != list(range(agent_count)):
        raise RuntimeError(f"{source_path.name} did not preserve every stable agent index at sample {sample_idx}")

    agents = scenario["agents"]
    if len(agents) != agent_count:
        raise RuntimeError(f"{source_path.name} returned incomplete agent metadata at sample {sample_idx}")
    for agent_idx, agent in enumerate(agents):
        if agent["id"] != agent_idx:
            raise RuntimeError(f"{source_path.name} returned unstable agent metadata at sample {sample_idx}")
        if agent["type"] != binding.AGENT_TYPE_VEHICLE:
            raise RuntimeError(f"{source_path.name} generated a non-vehicle agent at sample {sample_idx}")
        if agent["controller"] != binding.CONTROLLER_PDM:
            raise RuntimeError(f"{source_path.name} did not keep every generated vehicle under PDM control")
        if not agent["route"]:
            raise RuntimeError(f"{source_path.name} agent {agent_idx} has no route at sample {sample_idx}")

    # get_state() allocates these dictionaries, so retaining the boundary payload
    # preserves the route and dimensions even if Gigaflow later replaces a route.
    return tuple(agents)


def _capture_rollout(
    source_path: Path,
    agent_count: int,
    transition_count: int,
    segment_transition_count: int,
    seed: int,
):
    sample_count = transition_count + 1
    drive = _create_generation_drive(source_path, agent_count, sample_count, seed)
    try:
        drive.reset()
        initial_scenario = _single_scenario(drive.get_state())
        initial_agents = _validate_boundary_scenario(initial_scenario, source_path, agent_count, 0)
        traffic_elements = initial_scenario["traffic_elements"] or []
        traffic_count = len(traffic_elements)

        agent_frames = np.empty((sample_count, agent_count, binding.AGENT_F32_FIELDS), dtype=np.float32)
        agent_integer_frames = np.empty((sample_count, agent_count, binding.AGENT_I32_FIELDS), dtype=np.int32)
        traffic_frames = np.empty(
            (sample_count, max(traffic_count, 1), binding.TRAFFIC_I16_FIELDS),
            dtype=np.int16,
        )
        metric_frame = np.empty((1, agent_count, binding.METRICS_F32_FIELDS), dtype=np.float32)
        score_frame = np.empty((1, agent_count, binding.SCORE_F32_FIELDS), dtype=np.float32)
        live_agent_frame = np.empty((1, agent_count, binding.AGENT_F32_FIELDS), dtype=np.float32)
        live_agent_integer_frame = np.empty((1, agent_count, binding.AGENT_I32_FIELDS), dtype=np.int32)
        live_traffic_frame = np.empty(
            (1, max(traffic_count, 1), binding.TRAFFIC_I16_FIELDS),
            dtype=np.int16,
        )
        neutral_actions = np.zeros_like(drive.actions)
        boundary_agents = [initial_agents]

        for sample_idx in range(sample_count):
            if sample_idx > 0:
                _, _, terminals, truncations, _ = drive.step(neutral_actions)
                if np.any(terminals) or np.any(truncations):
                    raise RuntimeError(
                        f"{source_path.name} ended after {sample_idx} transitions, expected {transition_count}"
                    )
            drive.get_obs_html_frame(
                live_agent_frame,
                live_agent_integer_frame,
                metric_frame,
                score_frame,
                live_traffic_frame,
            )
            agent_frames[sample_idx] = live_agent_frame[0]
            agent_integer_frames[sample_idx] = live_agent_integer_frame[0]
            traffic_frames[sample_idx] = live_traffic_frame[0]

            if sample_idx > 0 and sample_idx % segment_transition_count == 0:
                scenario = _single_scenario(drive.get_state())
                boundary_agents.append(_validate_boundary_scenario(scenario, source_path, agent_count, sample_idx))

        expected_boundary_count = transition_count // segment_transition_count + 1
        if len(boundary_agents) != expected_boundary_count:
            raise RuntimeError(f"{source_path.name} did not capture every segment boundary")

        expected_ids = np.arange(agent_count, dtype=np.int32)
        if not np.all(agent_integer_frames[:, :, AGENT_ID_IDX] == expected_ids):
            raise RuntimeError(f"{source_path.name} telemetry returned unstable agent IDs")
        if not np.all(agent_integer_frames[:, :, AGENT_TYPE_IDX] == binding.AGENT_TYPE_VEHICLE):
            raise RuntimeError(f"{source_path.name} generated a non-vehicle agent")
        if not np.all(agent_integer_frames[:, :, AGENT_VALID_IDX] == 1):
            raise RuntimeError(f"{source_path.name} contains an invalid agent sample")
        if not np.all(agent_integer_frames[:, :, AGENT_ACTIVE_IDX] == 1):
            raise RuntimeError(f"{source_path.name} contains an inactive agent sample")
        if np.any(agent_integer_frames[:, :, AGENT_REMOVED_IDX]):
            raise RuntimeError(f"{source_path.name} removed an agent during generation")
        if not np.isfinite(agent_frames).all():
            raise RuntimeError(f"{source_path.name} contains non-finite agent telemetry")
        if np.any(agent_frames[:, :, AGENT_SPEED_IDX] < 0.0):
            raise RuntimeError(f"{source_path.name} contains a negative PDM speed")
        if traffic_count:
            if not np.all(traffic_frames[:, :traffic_count, TRAFFIC_VALID_IDX] == 1):
                raise RuntimeError(f"{source_path.name} contains invalid traffic-control telemetry")
            expected_traffic_types = np.asarray(
                [traffic_element["type"] for traffic_element in traffic_elements],
                dtype=np.int16,
            )
            if not np.all(traffic_frames[:, :traffic_count, TRAFFIC_TYPE_IDX] == expected_traffic_types):
                raise RuntimeError(f"{source_path.name} traffic-control types changed during generation")

        return boundary_agents, agent_frames, agent_integer_frames, traffic_frames[:, :traffic_count]
    finally:
        drive.close()


def _build_segment_data(
    source_data: dict,
    town_name: str,
    seed: int,
    segment_idx: int,
    boundary_agents: tuple[dict, ...],
    agent_frames: np.ndarray,
    agent_integer_frames: np.ndarray,
    traffic_frames: np.ndarray,
) -> tuple[dict, str]:
    sample_count, agent_count, _ = agent_frames.shape
    scenario_id = f"{town_name}_seed_{seed}_segment_{segment_idx:02d}"
    generated_agents = []
    for agent_idx in range(agent_count):
        boundary_agent = boundary_agents[agent_idx]
        route = tuple(int(lane_idx) for lane_idx in boundary_agent["route"])
        if not route:
            raise RuntimeError(f"Agent {agent_idx} has no generated route in segment {segment_idx}")
        headings = agent_frames[:, agent_idx, AGENT_HEADING_IDX]
        speeds = agent_frames[:, agent_idx, AGENT_SPEED_IDX]
        velocity_x = speeds * np.cos(headings)
        velocity_y = speeds * np.sin(headings)
        height = np.full(sample_count, boundary_agent["sim_height"], dtype=np.float32)
        generated_agents.append(
            {
                "id": agent_idx,
                "type": int(boundary_agent["type"]),
                "T": sample_count,
                "cols": {
                    "x": tuple(agent_frames[:, agent_idx, AGENT_X_IDX]),
                    "y": tuple(agent_frames[:, agent_idx, AGENT_Y_IDX]),
                    "z": tuple(agent_frames[:, agent_idx, AGENT_Z_IDX]),
                    "h": tuple(headings),
                    "vx": tuple(velocity_x),
                    "vy": tuple(velocity_y),
                    "len": tuple(agent_frames[:, agent_idx, AGENT_LENGTH_IDX]),
                    "wid": tuple(agent_frames[:, agent_idx, AGENT_WIDTH_IDX]),
                    "hgt": tuple(height),
                    "valid": tuple(int(value) for value in agent_integer_frames[:, agent_idx, AGENT_VALID_IDX]),
                },
                "route": route,
                "route_gt_len": len(route),
                "goal": (
                    float(agent_frames[-1, agent_idx, AGENT_X_IDX]),
                    float(agent_frames[-1, agent_idx, AGENT_Y_IDX]),
                    float(agent_frames[-1, agent_idx, AGENT_Z_IDX]),
                ),
                "mark_as_expert": int(boundary_agent["mark_as_expert"]),
            }
        )

    generated_traffic = []
    for traffic_idx, source_traffic_element in enumerate(source_data["traffic"]):
        generated_traffic_element = dict(source_traffic_element)
        if source_traffic_element["type"] == binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT:
            generated_traffic_element["states"] = tuple(
                int(value) for value in traffic_frames[:, traffic_idx, TRAFFIC_STATE_IDX]
            )
        generated_traffic.append(generated_traffic_element)

    generated_data = dict(source_data)
    generated_data.update(
        {
            "agents": generated_agents,
            "traffic": generated_traffic,
            "scenario_id": _fixed_string(scenario_id, SCENARIO_ID_BYTES),
            "dataset_name": _fixed_string(DATASET_NAME, DATASET_NAME_BYTES),
            "log_length": sample_count,
            "log_dt": LOG_DT_SECONDS,
        }
    )
    return generated_data, scenario_id


def _validate_generated_data(
    generated_data: dict,
    source_data: dict,
    expected_agent_count: int,
    expected_sample_count: int,
    expected_scenario_id: str,
) -> None:
    if len(generated_data["agents"]) != expected_agent_count:
        raise RuntimeError("Validated binary has an unexpected agent count")
    if generated_data["log_length"] != expected_sample_count:
        raise RuntimeError("Validated binary has an unexpected log length")
    if generated_data["log_dt"] != float(np.float32(LOG_DT_SECONDS)):
        raise RuntimeError("Validated binary has an unexpected timestep")
    if _decoded_fixed_string(generated_data["scenario_id"]) != expected_scenario_id:
        raise RuntimeError("Validated binary has an unexpected scenario ID")
    if _decoded_fixed_string(generated_data["dataset_name"]) != DATASET_NAME:
        raise RuntimeError("Validated binary has an unexpected dataset name")

    float_columns = ("x", "y", "z", "h", "vx", "vy", "len", "wid", "hgt")
    for agent_idx, agent in enumerate(generated_data["agents"]):
        if agent["id"] != agent_idx or agent["type"] != binding.AGENT_TYPE_VEHICLE:
            raise RuntimeError(f"Validated binary has invalid metadata for agent {agent_idx}")
        if agent["T"] != expected_sample_count:
            raise RuntimeError(f"Validated binary has an unexpected sample count for agent {agent_idx}")
        if agent["route_gt_len"] != len(agent["route"]) or not agent["route"]:
            raise RuntimeError(f"Validated binary has an invalid route for agent {agent_idx}")
        if any(lane_idx < 0 or lane_idx >= len(generated_data["roads"]) for lane_idx in agent["route"]):
            raise RuntimeError(f"Validated binary route {agent_idx} references an invalid lane")
        for column_name in float_columns:
            column = agent["cols"][column_name]
            if len(column) != expected_sample_count or not all(math.isfinite(value) for value in column):
                raise RuntimeError(f"Validated binary has invalid {column_name} values for agent {agent_idx}")
        if tuple(agent["cols"]["valid"]) != (1,) * expected_sample_count:
            raise RuntimeError(f"Validated binary has invalid samples for agent {agent_idx}")
        expected_goal = tuple(agent["cols"][name][-1] for name in ("x", "y", "z"))
        if agent["goal"] != expected_goal:
            raise RuntimeError(f"Validated binary has an invalid ground-truth goal for agent {agent_idx}")

    if generated_data["roads"] != source_data["roads"]:
        raise RuntimeError("Validated binary changed source road geometry")
    if generated_data["objects"] != source_data["objects"]:
        raise RuntimeError("Validated binary changed source objects")
    if generated_data["lane_graph"] != source_data["lane_graph"]:
        raise RuntimeError("Validated binary changed the source lane graph")
    if generated_data["has_phase_section"] != source_data["has_phase_section"]:
        raise RuntimeError("Validated binary changed traffic phase metadata")
    if generated_data["has_width_section"] != source_data["has_width_section"]:
        raise RuntimeError("Validated binary changed lane-width metadata")
    if generated_data["objects_of_interest"] != source_data["objects_of_interest"]:
        raise RuntimeError("Validated binary changed objects-of-interest metadata")
    if generated_data["tracks_to_predict"] != source_data["tracks_to_predict"]:
        raise RuntimeError("Validated binary changed tracks-to-predict metadata")
    if len(generated_data["traffic"]) != len(source_data["traffic"]):
        raise RuntimeError("Validated binary changed the traffic-control count")
    for generated_traffic_element, source_traffic_element in zip(generated_data["traffic"], source_data["traffic"]):
        for field_name in (
            "id",
            "type",
            "stop_line",
            "heading",
            "controlled_lanes",
            "junction_id",
            "phase_idx",
        ):
            if generated_traffic_element[field_name] != source_traffic_element[field_name]:
                raise RuntimeError(f"Validated binary changed traffic-control field {field_name}")
        state_count = len(generated_traffic_element["states"])
        if generated_traffic_element["type"] == binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT:
            if state_count != expected_sample_count:
                raise RuntimeError("Validated binary has an incomplete traffic-light state horizon")
        elif generated_traffic_element["states"] != source_traffic_element["states"]:
            raise RuntimeError("Validated binary changed a static traffic-control state")


def _validate_generation_arguments(
    source_path: Path,
    output_directory: Path,
    agent_count: int,
    transition_count: int,
    segment_transition_count: int,
    seed: int,
) -> None:
    if not source_path.is_file() or source_path.suffix != ".bin":
        raise ValueError(f"Input is not a .bin file: {source_path}")
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(f"Output path is not a directory: {output_directory}")
    if not isinstance(agent_count, int) or isinstance(agent_count, bool) or agent_count <= 0:
        raise ValueError("agent_count must be positive")
    if not isinstance(transition_count, int) or isinstance(transition_count, bool) or transition_count <= 0:
        raise ValueError("transition_count must be positive")
    if (
        not isinstance(segment_transition_count, int)
        or isinstance(segment_transition_count, bool)
        or segment_transition_count <= 0
    ):
        raise ValueError("segment_transition_count must be positive")
    if transition_count % segment_transition_count != 0:
        raise ValueError("transition_count must be divisible by segment_transition_count")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63)")


def generate_map_segments(
    source_path: Path,
    output_directory: Path,
    agent_count: int = DEFAULT_AGENT_COUNT,
    transition_count: int = DEFAULT_TRANSITION_COUNT,
    segment_transition_count: int = DEFAULT_SEGMENT_TRANSITION_COUNT,
    seed: int = DEFAULT_BASE_SEED,
    overwrite: bool = False,
) -> list[GenerationSummary]:
    source_path = Path(source_path)
    output_directory = Path(output_directory)
    _validate_generation_arguments(
        source_path,
        output_directory,
        agent_count,
        transition_count,
        segment_transition_count,
        seed,
    )

    source_data = read_bin(source_path)
    if source_data["agents"]:
        raise ValueError(f"Source must be a map-only binary without agent tracks: {source_path}")
    town_name = _decoded_fixed_string(source_data["scenario_id"])
    if not town_name:
        raise ValueError(f"Source binary has no scenario ID: {source_path}")

    boundary_agents, agent_frames, agent_integer_frames, traffic_frames = _capture_rollout(
        source_path,
        agent_count,
        transition_count,
        segment_transition_count,
        seed,
    )
    if traffic_frames.shape[1] != len(source_data["traffic"]):
        raise RuntimeError(f"{source_path.name} changed its traffic-control count during initialization")

    output_directory.mkdir(parents=True, exist_ok=True)
    segment_count = transition_count // segment_transition_count
    summaries = []
    temporary_paths = []
    final_paths = []
    with tempfile.TemporaryDirectory(dir=output_directory, prefix=f".{source_path.stem}.") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        for segment_idx in range(segment_count):
            first_sample_idx = segment_idx * segment_transition_count
            last_sample_idx = first_sample_idx + segment_transition_count + 1
            generated_data, scenario_id = _build_segment_data(
                source_data,
                town_name,
                seed,
                segment_idx,
                boundary_agents[segment_idx],
                agent_frames[first_sample_idx:last_sample_idx],
                agent_integer_frames[first_sample_idx:last_sample_idx],
                traffic_frames[first_sample_idx:last_sample_idx],
            )
            output_name = f"{source_path.stem}__segment_{segment_idx:02d}.bin"
            temporary_path = temporary_directory / output_name
            final_path = output_directory / output_name
            if source_path.resolve() == final_path.resolve():
                raise ValueError("Input and output binary paths must differ")
            write_bin(generated_data, temporary_path)
            validated_data = read_bin(temporary_path)
            _validate_generated_data(
                validated_data,
                source_data,
                agent_count,
                segment_transition_count + 1,
                scenario_id,
            )
            digest = _sha256(temporary_path)
            temporary_paths.append(temporary_path)
            final_paths.append(final_path)
            summaries.append(
                GenerationSummary(
                    source_path=source_path,
                    output_path=final_path,
                    scenario_id=scenario_id,
                    seed=seed,
                    agent_count=agent_count,
                    sample_count=segment_transition_count + 1,
                    sha256=digest,
                )
            )

        # Resolve every collision before installing any file from this map.
        should_install_segments = []
        for temporary_path, final_path in zip(temporary_paths, final_paths):
            if final_path.exists() and not final_path.is_file():
                raise ValueError(f"Output path is not a file: {final_path}")
            if final_path.exists() and _files_equal(temporary_path, final_path):
                should_install_segments.append(False)
                continue
            if final_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Output differs from the deterministic generation: {final_path}; pass --overwrite to replace it"
                )
            should_install_segments.append(True)

        for should_install, temporary_path, final_path in zip(should_install_segments, temporary_paths, final_paths):
            if should_install:
                os.replace(temporary_path, final_path)

    return summaries


def generate_scenario(*args, **kwargs) -> list[GenerationSummary]:
    """Compatibility name for generating every segment belonging to one map."""
    return generate_map_segments(*args, **kwargs)


def generate_all(
    input_path: Path = DEFAULT_INPUT,
    output_directory: Path = DEFAULT_OUTPUT,
    agent_count: int = DEFAULT_AGENT_COUNT,
    transition_count: int = DEFAULT_TRANSITION_COUNT,
    segment_transition_count: int = DEFAULT_SEGMENT_TRANSITION_COUNT,
    base_seed: int = DEFAULT_BASE_SEED,
    overwrite: bool = False,
) -> list[GenerationSummary]:
    input_path = Path(input_path)
    output_directory = Path(output_directory)
    if input_path.is_file():
        source_paths = [input_path]
    elif input_path.is_dir():
        source_paths = sorted(input_path.glob("*.bin"))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    if not source_paths:
        raise ValueError(f"Input contains no .bin files: {input_path}")
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(f"Output path is not a directory: {output_directory}")
    if (
        not isinstance(base_seed, int)
        or isinstance(base_seed, bool)
        or base_seed < 0
        or base_seed + len(source_paths) - 1 >= 2**63
    ):
        raise ValueError("Assigned seeds must be in [0, 2**63)")

    summaries = []
    for map_idx, source_path in enumerate(source_paths):
        map_summaries = generate_map_segments(
            source_path,
            output_directory,
            agent_count=agent_count,
            transition_count=transition_count,
            segment_transition_count=segment_transition_count,
            seed=base_seed + map_idx,
            overwrite=overwrite,
        )
        summaries.extend(map_summaries)
        for summary in map_summaries:
            print(
                f"{source_path.name}: seed={summary.seed} agents={summary.agent_count} "
                f"samples={summary.sample_count} output={summary.output_path} sha256={summary.sha256}",
                flush=True,
            )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CARLA binary directory or one .bin file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENT_COUNT, help="PDM vehicles per map")
    parser.add_argument("--steps", type=int, default=DEFAULT_TRANSITION_COUNT, help="Transitions to record per map")
    parser.add_argument(
        "--segment-steps",
        type=int,
        default=DEFAULT_SEGMENT_TRANSITION_COUNT,
        help="Transitions stored in each output segment",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED, help="Seed assigned to the first sorted map")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs that differ")
    args = parser.parse_args()
    try:
        generate_all(
            input_path=args.input,
            output_directory=args.output,
            agent_count=args.agents,
            transition_count=args.steps,
            segment_transition_count=args.segment_steps,
            base_seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
