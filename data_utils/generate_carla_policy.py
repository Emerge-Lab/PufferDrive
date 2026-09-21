#!/usr/bin/env python3
"""Generate standard replay binaries from policy-driven CARLA Gigaflow episodes."""

import argparse
import copy
import hashlib
import math
import os
import pickle
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    from data_utils.generate_carla_pdm import _files_equal, _fixed_string
    from data_utils.mirror_map_bin import read_bin, write_bin
except ModuleNotFoundError:  # Direct execution adds data_utils/, not the repository root.
    from generate_carla_pdm import _files_equal, _fixed_string
    from mirror_map_bin import read_bin, write_bin
from pufferlib.ocean.drive import binding
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark
from pufferlib.pufferl import _run_eval_rollout, load_config


DEFAULT_CHECKPOINT = Path("experiments/3_0_no_conditionning_target.pt")
DEFAULT_CHECKPOINT_CONFIG = Path("experiments/3_0_no_conditionning_target_config.yaml")
DEFAULT_BENCHMARK_CONFIG = Path("pufferlib/config/evaluation/benchmark.yaml")
DEFAULT_BENCHMARK_NAME = "adversarial_carla"
DEFAULT_OUTPUT = Path("pufferlib/resources/drive/binaries/carla_generated_policy")
DEFAULT_WORKER_COUNT = 16
DEFAULT_MAX_CANDIDATE_MULTIPLIER = 2
DATASET_NAME = "carla_generated_policy"
MANIFEST_FILE_NAME = "generation_manifest.yaml"
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
TRAFFIC_VALID_IDX = 0
TRAFFIC_TYPE_IDX = 1
TRAFFIC_STATE_IDX = 2


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for block in iter(lambda: source_file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode_infraction_reasons(summary):
    reasons = []
    for metric_name, reason in (("offroad_rate", "offroad"), ("collision_rate", "collision")):
        metric_value = float(summary[metric_name])
        if not math.isfinite(metric_value) or metric_value < 0.0:
            raise RuntimeError(f"Evaluation summary has invalid {metric_name}={metric_value}")
        if metric_value > 0.0:
            reasons.append(reason)
    return tuple(reasons)


def _select_map_cycled_entries(entries, scenario_count, map_names):
    if not map_names:
        raise RuntimeError("Scenario generation has no configured maps")
    entries_by_map = {map_name: [] for map_name in map_names}
    for entry in entries:
        map_name = entry["map"]
        if map_name not in entries_by_map:
            raise RuntimeError(f"Generated scenario references an unexpected map: {map_name}")
        entries_by_map[map_name].append(entry)
    for map_entries in entries_by_map.values():
        map_entries.sort(key=lambda entry: (entry["seed"], entry["episode_idx"]))

    selected_entries = []
    for scenario_idx in range(scenario_count):
        map_cycle_idx = scenario_idx % len(map_names)
        map_name = map_names[map_cycle_idx]
        map_scenario_idx = scenario_idx // len(map_names)
        if map_scenario_idx >= len(entries_by_map[map_name]):
            raise RuntimeError(
                f"Map-cycled output needs scenario {map_scenario_idx + 1} from {map_name}, "
                f"but only {len(entries_by_map[map_name])} clean candidates are available. "
                "Increase max_candidate_scenarios."
            )
        selected_entries.append(
            {
                **entries_by_map[map_name][map_scenario_idx],
                "episode_idx": scenario_idx,
                "map_cycle_idx": map_cycle_idx,
            }
        )
    return selected_entries


def _add_ordered_file_names(staging_directory, entries):
    staging_directory = Path(staging_directory)
    ordered_entries = []
    for scenario_idx, entry in enumerate(entries):
        source_file = entry["file"]
        ordered_file = f"{scenario_idx:06d}_{source_file}"
        os.replace(staging_directory / source_file, staging_directory / ordered_file)
        ordered_entries.append({**entry, "file": ordered_file})
    return ordered_entries


def _load_mapping(path, label):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    with path.open(encoding="utf-8") as source_file:
        value = yaml.safe_load(source_file)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping: {path}")
    return value


def _resolve_policy_name(checkpoint_path, configured_policy_name):
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    partner_weight = next(
        (value for key, value in state_dict.items() if key.endswith("actor_backbone.partner_encoder.0.weight")),
        None,
    )
    if partner_weight is None or partner_weight.ndim != 2:
        raise ValueError("Checkpoint has no recognizable actor partner-encoder input layer")
    checkpoint_partner_feature_count = int(partner_weight.shape[1])
    if checkpoint_partner_feature_count == binding.PARTNER_FEATURES:
        return configured_policy_name
    if checkpoint_partner_feature_count == binding.PARTNER_FEATURES - 1:
        return "TargetDrive"
    raise ValueError(
        f"Checkpoint expects {checkpoint_partner_feature_count} partner features, but the simulator exposes "
        f"{binding.PARTNER_FEATURES}"
    )


def resolve_generation_args(
    checkpoint_path,
    checkpoint_config_path,
    benchmark_config_path,
    benchmark_name,
    scenario_count=None,
    scenario_length=None,
    seed=None,
    device=None,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file() or checkpoint_path.suffix != ".pt":
        raise ValueError(f"Checkpoint must be an existing .pt file: {checkpoint_path}")
    checkpoint_config = _load_mapping(checkpoint_config_path, "Checkpoint config")
    environment_config, benchmarks = drive_benchmark.load_benchmark_config(
        str(benchmark_config_path),
        [benchmark_name],
    )
    benchmark = copy.deepcopy(benchmarks[0])
    if scenario_count is not None:
        if isinstance(scenario_count, bool) or not isinstance(scenario_count, int) or scenario_count <= 0:
            raise ValueError("scenario_count must be a positive integer")
        benchmark["num_scenarios"] = scenario_count
    if scenario_length is not None:
        if isinstance(scenario_length, bool) or not isinstance(scenario_length, int) or scenario_length <= 0:
            raise ValueError("scenario_length must be a positive integer")
        benchmark["env"]["scenario_length"] = scenario_length
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        benchmark["seed"] = seed

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        base_args = load_config("puffer_drive")
    finally:
        sys.argv = original_argv
    for section in ("policy", "rnn"):
        base_args[section].update(checkpoint_config[section])
    base_args["env"].update(checkpoint_config["env"])
    resolved_policy_name = _resolve_policy_name(checkpoint_path, checkpoint_config["policy_name"])
    base_args["policy_name"] = checkpoint_config["policy_name"]
    base_args["rnn_name"] = checkpoint_config["rnn_name"]
    base_args["train"]["use_rnn"] = base_args["rnn_name"] is not None
    base_args["load_model_path"] = str(checkpoint_path)
    base_args["wandb"] = False
    if device is not None:
        base_args["train"]["device"] = device
    run_args = drive_benchmark.build_benchmark_args(
        base_args,
        benchmark,
        environment_config,
    )
    run_args["policy_name"] = resolved_policy_name
    environment = run_args["env"]
    required_values = {
        "simulation_mode": "gigaflow",
        "control_mode": "control_vehicles",
        "sdc_controller": "policy",
        "non_sdc_controller": "policy",
    }
    for field_name, expected_value in required_values.items():
        if environment[field_name] != expected_value:
            raise ValueError(
                f"Generation requires env.{field_name}={expected_value!r}, got {environment[field_name]!r}"
            )
    if run_args["eval"]["action_selection"] != "mean":
        raise ValueError("Standard policy scenario generation requires eval.action_selection='mean'")
    return run_args, benchmark


def _append_terminal_frame(replay_environment):
    frames = {key: np.asarray(value) for key, value in replay_environment["frames"].items()}
    terminal_frame = replay_environment.get("terminal_frame")
    if terminal_frame is not None:
        frames = {
            key: np.concatenate((values, np.asarray(terminal_frame[key])[None]), axis=0)
            for key, values in frames.items()
        }
    metadata = replay_environment["metadata"]
    expected_sample_count = int(metadata["episode_timestep"]) - int(metadata["initial_timestep"]) + 1
    actual_sample_count = frames["agent_f32"].shape[0]
    if actual_sample_count != expected_sample_count:
        raise RuntimeError(
            f"Captured {actual_sample_count} state samples, expected {expected_sample_count} "
            f"through terminal timestep {metadata['episode_timestep']}"
        )
    return frames


def _build_scenario_data(source_data, replay_environment, summary, episode_idx, dt_seconds):
    scenario = replay_environment["scenario"]
    metadata = replay_environment["metadata"]
    frames = _append_terminal_frame(replay_environment)
    # Gigaflow sizes replay rows by requested vehicles, so failed spawns leave inactive rows.
    active_agent_indices = scenario["active_agent_indices"]
    if not active_agent_indices:
        raise RuntimeError("Policy rollout produced a scenario with no spawned vehicles")
    active_agent_indices = np.asarray(active_agent_indices, dtype=np.intp)
    if active_agent_indices.size != int(metadata["active_agent_count"]):
        raise RuntimeError("Replay agent capacity does not match its active policy-agent count")
    agent_frames = frames["agent_f32"][:, active_agent_indices]
    agent_integer_frames = frames["agent_i32"][:, active_agent_indices]
    traffic_frames = frames["traffic_i16"]
    sample_count, agent_count, _ = agent_frames.shape
    replay_agents = scenario.get("agents") or []
    if len(replay_agents) != frames["agent_f32"].shape[1]:
        raise RuntimeError("Initial scenario metadata does not cover every replay agent row")
    boundary_agents = [replay_agents[agent_idx] for agent_idx in active_agent_indices]

    expected_ids = np.asarray([agent["id"] for agent in boundary_agents], dtype=np.int32)
    if not np.all(agent_integer_frames[:, :, AGENT_ID_IDX] == expected_ids):
        raise RuntimeError("Policy rollout returned unstable agent IDs")
    if not np.all(agent_integer_frames[:, :, AGENT_TYPE_IDX] == binding.AGENT_TYPE_VEHICLE):
        raise RuntimeError("Policy rollout contains a non-vehicle active agent")
    if not np.isfinite(agent_frames).all():
        raise RuntimeError("Policy rollout contains non-finite agent telemetry")

    generated_agents = []
    for agent_idx, boundary_agent in enumerate(boundary_agents):
        route = tuple(int(lane_idx) for lane_idx in boundary_agent["route"])
        if not route:
            raise RuntimeError(f"Policy-controlled agent {agent_idx} has no route")
        headings = agent_frames[:, agent_idx, AGENT_HEADING_IDX]
        speeds = agent_frames[:, agent_idx, AGENT_SPEED_IDX]
        valid = agent_integer_frames[:, agent_idx, AGENT_VALID_IDX].astype(np.int32, copy=False)
        if valid[0] != 1 or np.any((valid != 0) & (valid != 1)):
            raise RuntimeError(f"Policy-controlled agent {agent_idx} has an invalid validity mask")
        last_valid_sample_idx = int(np.flatnonzero(valid)[-1])
        generated_agents.append(
            {
                "id": int(boundary_agent["id"]),
                "type": binding.AGENT_TYPE_VEHICLE,
                "T": sample_count,
                "cols": {
                    "x": tuple(agent_frames[:, agent_idx, AGENT_X_IDX]),
                    "y": tuple(agent_frames[:, agent_idx, AGENT_Y_IDX]),
                    "z": tuple(agent_frames[:, agent_idx, AGENT_Z_IDX]),
                    "h": tuple(headings),
                    "vx": tuple(speeds * np.cos(headings)),
                    "vy": tuple(speeds * np.sin(headings)),
                    "len": tuple(agent_frames[:, agent_idx, AGENT_LENGTH_IDX]),
                    "wid": tuple(agent_frames[:, agent_idx, AGENT_WIDTH_IDX]),
                    "hgt": (float(boundary_agent["sim_height"]),) * sample_count,
                    "valid": tuple(int(value) for value in valid),
                },
                "route": route,
                "route_gt_len": len(route),
                "goal": tuple(
                    float(agent_frames[last_valid_sample_idx, agent_idx, coordinate_idx])
                    for coordinate_idx in (AGENT_X_IDX, AGENT_Y_IDX, AGENT_Z_IDX)
                ),
                "mark_as_expert": 0,
            }
        )

    source_traffic = source_data["traffic"]
    if traffic_frames.shape[1] < len(source_traffic):
        raise RuntimeError("Captured traffic telemetry is incomplete")
    generated_traffic = []
    for traffic_idx, source_element in enumerate(source_traffic):
        generated_element = dict(source_element)
        if source_element["type"] == binding.TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT:
            if not np.all(traffic_frames[:, traffic_idx, TRAFFIC_VALID_IDX] == 1):
                raise RuntimeError(f"Traffic light {traffic_idx} became invalid during generation")
            if not np.all(traffic_frames[:, traffic_idx, TRAFFIC_TYPE_IDX] == source_element["type"]):
                raise RuntimeError(f"Traffic light {traffic_idx} changed type during generation")
            generated_element["states"] = tuple(
                int(value) for value in traffic_frames[:, traffic_idx, TRAFFIC_STATE_IDX]
            )
        generated_traffic.append(generated_element)

    map_stem = Path(metadata["map_path"]).stem
    scenario_id = f"{map_stem}_seed_{summary['seed']}"
    if episode_idx is not None:
        scenario_id += f"_episode_{episode_idx:06d}"
    generated_data = dict(source_data)
    generated_data.update(
        {
            "agents": generated_agents,
            "traffic": generated_traffic,
            "scenario_id": _fixed_string(scenario_id, SCENARIO_ID_BYTES),
            "dataset_name": _fixed_string(DATASET_NAME, DATASET_NAME_BYTES),
            "log_length": sample_count,
            "log_dt": float(dt_seconds),
            "objects_of_interest": (),
            "tracks_to_predict": (),
        }
    )
    return generated_data, scenario_id


def _validate_written_scenario(path, source_data, scenario_id, expected_agent_count, expected_sample_count):
    data = read_bin(path)
    decoded_scenario_id = data["scenario_id"].split(b"\0", 1)[0].decode("utf-8")
    decoded_dataset_name = data["dataset_name"].split(b"\0", 1)[0].decode("utf-8")
    if decoded_scenario_id != scenario_id or decoded_dataset_name != DATASET_NAME:
        raise RuntimeError("Generated binary has invalid scenario identity metadata")
    if len(data["agents"]) != expected_agent_count or data["log_length"] != expected_sample_count:
        raise RuntimeError("Generated binary has an unexpected agent or sample count")
    if data["objects_of_interest"] or data["tracks_to_predict"]:
        raise RuntimeError("Generated standard scenario contains adversarial role annotations")
    if data["roads"] != source_data["roads"] or data["lane_graph"] != source_data["lane_graph"]:
        raise RuntimeError("Generated binary changed source map geometry")
    for agent in data["agents"]:
        if agent["type"] != binding.AGENT_TYPE_VEHICLE or agent["mark_as_expert"] != 0:
            raise RuntimeError("Generated binary has invalid vehicle metadata")
        if agent["T"] != expected_sample_count or not agent["route"]:
            raise RuntimeError("Generated binary has an invalid vehicle trajectory")
        valid = np.asarray(agent["cols"]["valid"])
        if valid[0] != 1 or np.any((valid != 0) & (valid != 1)):
            raise RuntimeError("Generated binary has an invalid trajectory validity mask")


class PolicyScenarioWriter:
    def __init__(
        self,
        staging_directory,
        dt_seconds,
        reject_infractions=False,
        verbose=False,
    ):
        self.staging_directory = Path(staging_directory)
        self.dt_seconds = float(dt_seconds)
        self.reject_infractions = bool(reject_infractions)
        self.verbose = bool(verbose)
        self.source_cache = {}
        self.entries = []
        self.rejections = []

    def _record_rejection(self, summary, seed, rejection_reasons):
        rejection = {
            "map": Path(summary["map_name"]).name,
            "seed": seed,
            "termination_timestep": int(summary["episode_timestep"]),
            "offroad_rate": float(summary["offroad_rate"]),
            "collision_rate": float(summary["collision_rate"]),
            "reasons": rejection_reasons,
        }
        self.rejections.append(rejection)
        if self.verbose:
            # The progress bar owns the terminal line, so step around it.
            tqdm.write(f"REJECT {rejection['map']} seed={seed} reasons={','.join(rejection_reasons)}")

    def __call__(self, summary, episode_idx):
        seed = int(summary["seed"])
        rejection_reasons = _episode_infraction_reasons(summary) if self.reject_infractions else ()
        if rejection_reasons:
            self._record_rejection(summary, seed, rejection_reasons)
            return
        replay_bytes = summary.get("replay_environment_bundle")
        if not isinstance(replay_bytes, bytes):
            raise RuntimeError("Completed evaluation episode is missing its environment replay bundle")
        replay_environment = pickle.loads(zlib.decompress(replay_bytes))
        if replay_environment.get("schema") != "interactive_replay_environment_v1":
            raise RuntimeError("Environment replay bundle has an unsupported schema")
        scenario = replay_environment.get("scenario")
        if not isinstance(scenario, dict):
            raise RuntimeError("Environment replay bundle is missing its initial scenario")
        replay_agents = scenario.get("agents")
        active_agent_indices = scenario.get("active_agent_indices")
        if not isinstance(replay_agents, list) or not isinstance(active_agent_indices, list):
            raise RuntimeError("Initial scenario has invalid agent metadata")
        for active_agent_idx in active_agent_indices:
            if (
                isinstance(active_agent_idx, bool)
                or not isinstance(active_agent_idx, int)
                or active_agent_idx < 0
                or active_agent_idx >= len(replay_agents)
            ):
                raise RuntimeError(f"Initial scenario has invalid active agent index: {active_agent_idx!r}")
        if any(not replay_agents[active_agent_idx].get("route") for active_agent_idx in active_agent_indices):
            self._record_rejection(summary, seed, ("spawn_failure",))
            return
        source_path = Path(replay_environment["metadata"]["map_path"])
        if source_path not in self.source_cache:
            source_data = read_bin(source_path)
            if source_data["agents"]:
                raise RuntimeError(f"CARLA generation source must be map-only: {source_path}")
            self.source_cache[source_path] = source_data
        source_data = self.source_cache[source_path]
        generated_data, scenario_id = _build_scenario_data(
            source_data,
            replay_environment,
            summary,
            None if self.reject_infractions else episode_idx,
            self.dt_seconds,
        )
        output_name = f"{scenario_id}.bin"
        output_path = self.staging_directory / output_name
        write_bin(generated_data, output_path)
        _validate_written_scenario(
            output_path,
            source_data,
            scenario_id,
            len(generated_data["agents"]),
            generated_data["log_length"],
        )
        entry = {
            "episode_idx": episode_idx,
            "scenario_id": scenario_id,
            "file": output_name,
            "map": source_path.name,
            "seed": int(summary["seed"]),
            "agent_count": len(generated_data["agents"]),
            "sample_count": generated_data["log_length"],
            "termination_timestep": int(summary["episode_timestep"]),
            "sha256": _sha256(output_path),
        }
        self.entries.append(entry)


def _install_outputs(staging_directory, output_directory, entries, overwrite):
    output_directory.mkdir(parents=True, exist_ok=True)
    install_plan = []
    for entry in entries:
        staged_path = Path(staging_directory) / entry["file"]
        final_path = output_directory / entry["file"]
        if final_path.exists() and not final_path.is_file():
            raise ValueError(f"Output path is not a file: {final_path}")
        if final_path.exists() and _files_equal(staged_path, final_path):
            install_plan.append((False, staged_path, final_path))
            continue
        if final_path.exists() and not overwrite:
            raise FileExistsError(f"Output differs from deterministic generation: {final_path}")
        install_plan.append((True, staged_path, final_path))
    for should_install, staged_path, final_path in install_plan:
        if should_install:
            os.replace(staged_path, final_path)


def generate_policy_scenarios(
    checkpoint_path=DEFAULT_CHECKPOINT,
    checkpoint_config_path=DEFAULT_CHECKPOINT_CONFIG,
    benchmark_config_path=DEFAULT_BENCHMARK_CONFIG,
    benchmark_name=DEFAULT_BENCHMARK_NAME,
    output_directory=DEFAULT_OUTPUT,
    scenario_count=None,
    scenario_length=None,
    seed=None,
    num_workers=None,
    device=None,
    overwrite=False,
    reject_infractions=False,
    max_candidate_scenarios=None,
    verbose=False,
):
    output_directory = Path(output_directory)
    if output_directory.exists() and not output_directory.is_dir():
        raise ValueError(f"Output path is not a directory: {output_directory}")
    requested_scenario_count = scenario_count
    if requested_scenario_count is None:
        benchmark_config = _load_mapping(benchmark_config_path, "Benchmark config")
        selected_benchmark = next(
            (item for item in benchmark_config["benchmarks"] if item.get("name") == benchmark_name),
            None,
        )
        if selected_benchmark is None:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        requested_scenario_count = int(selected_benchmark["num_scenarios"])
    if reject_infractions:
        if max_candidate_scenarios is None:
            max_candidate_scenarios = requested_scenario_count * DEFAULT_MAX_CANDIDATE_MULTIPLIER
        if (
            isinstance(max_candidate_scenarios, bool)
            or not isinstance(max_candidate_scenarios, int)
            or max_candidate_scenarios < requested_scenario_count
        ):
            raise ValueError("max_candidate_scenarios must be an integer at least as large as scenario_count")
    elif max_candidate_scenarios is not None:
        raise ValueError("max_candidate_scenarios requires reject_infractions=True")
    candidate_scenario_count = max_candidate_scenarios or requested_scenario_count

    run_args, benchmark = resolve_generation_args(
        checkpoint_path,
        checkpoint_config_path,
        benchmark_config_path,
        benchmark_name,
        scenario_count=candidate_scenario_count,
        scenario_length=scenario_length,
        seed=seed,
        device=device,
    )
    candidate_scenario_count = int(run_args["num_scenarios"])
    configured_worker_count = int(run_args["vec"]["num_envs"])
    if num_workers is None:
        num_workers = min(configured_worker_count, DEFAULT_WORKER_COUNT)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers <= 0:
        raise ValueError("num_workers must be a positive integer")
    num_workers = min(num_workers, candidate_scenario_count)
    worker_env_kwargs, total_steps = drive_benchmark._plan_benchmark_eval_workers(
        run_args,
        candidate_scenario_count,
        num_workers,
        run_args["env"]["scenario_length"],
        capture_replay=True,
    )
    for worker_environment in worker_env_kwargs:
        maximum_needed_agent_count = worker_environment["num_eval_scenarios"] * worker_environment["max_agents_per_env"]
        worker_environment["num_agents"] = min(worker_environment["num_agents"], maximum_needed_agent_count)

    output_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_directory, prefix=".policy_generation.") as staging_directory:
        writer = PolicyScenarioWriter(
            staging_directory,
            run_args["env"]["dt"],
            reject_infractions=reject_infractions,
            verbose=verbose,
        )
        summaries = _run_eval_rollout(
            run_args,
            "puffer_drive",
            worker_env_kwargs,
            total_steps,
            f"Generating {benchmark_name}",
            candidate_scenario_count,
            episode_callback=writer,
        )
        if len(summaries) != candidate_scenario_count:
            raise RuntimeError(f"Generation completed {len(summaries)} summaries, expected {candidate_scenario_count}")
        candidate_seeds = [int(summary["seed"]) for summary in summaries]
        if len(set(candidate_seeds)) != len(candidate_seeds):
            raise RuntimeError("Candidate scenario seeds must be unique")
        if len(writer.entries) < requested_scenario_count:
            raise RuntimeError(
                f"Only {len(writer.entries)} of {candidate_scenario_count} candidates were clean; "
                f"need {requested_scenario_count}. Increase max_candidate_scenarios."
            )
        map_directory = Path(run_args["env"]["map_dir"])
        configured_map_names = [
            path.name for path in sorted(map_directory.glob("*.bin"))[: int(run_args["env"]["num_maps"])]
        ]
        selected_entries = _select_map_cycled_entries(
            writer.entries,
            requested_scenario_count,
            configured_map_names,
        )
        selected_entries = _add_ordered_file_names(staging_directory, selected_entries)
        _install_outputs(staging_directory, output_directory, selected_entries, overwrite)
        benchmark["num_scenarios"] = requested_scenario_count
        manifest = {
            "schema": "puffer_drive_policy_generation_v2",
            "benchmark": benchmark,
            "checkpoint": str(Path(checkpoint_path)),
            "checkpoint_config": str(Path(checkpoint_config_path)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "policy_name": run_args["policy_name"],
            "resolved_env": run_args["env"],
            "action_selection": run_args["eval"]["action_selection"],
            "scenario_count": requested_scenario_count,
            "candidate_scenario_count": candidate_scenario_count,
            "clean_candidate_count": len(writer.entries),
            "reject_infractions": bool(reject_infractions),
            "map_cycle": configured_map_names,
            "rejections": sorted(writer.rejections, key=lambda rejection: rejection["seed"]),
            "num_workers": num_workers,
            "files": selected_entries,
        }
        staged_manifest = Path(staging_directory) / MANIFEST_FILE_NAME
        with staged_manifest.open("w", encoding="utf-8") as manifest_file:
            yaml.safe_dump(manifest, manifest_file, sort_keys=False)
        os.replace(staged_manifest, output_directory / MANIFEST_FILE_NAME)
    print(
        f"Generated {len(selected_entries)} scenarios from {candidate_scenario_count} candidates "
        f"({len(writer.rejections)} rejected) -> {output_directory / MANIFEST_FILE_NAME}",
        flush=True,
    )
    return selected_entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--checkpoint-config", type=Path, default=DEFAULT_CHECKPOINT_CONFIG)
    parser.add_argument("--benchmark-config", type=Path, default=DEFAULT_BENCHMARK_CONFIG)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_NAME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-scenarios", type=int, default=None)
    parser.add_argument("--scenario-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the benchmark evaluation seed")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reject-infractions", action="store_true")
    parser.add_argument("--max-candidate-scenarios", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Log every rejected candidate scenario")
    cli = parser.parse_args()
    try:
        generate_policy_scenarios(
            checkpoint_path=cli.checkpoint,
            checkpoint_config_path=cli.checkpoint_config,
            benchmark_config_path=cli.benchmark_config,
            benchmark_name=cli.benchmark,
            output_directory=cli.output,
            scenario_count=cli.num_scenarios,
            scenario_length=cli.scenario_length,
            seed=cli.seed,
            num_workers=cli.num_workers,
            device=cli.device,
            overwrite=cli.overwrite,
            reject_infractions=cli.reject_infractions,
            max_candidate_scenarios=cli.max_candidate_scenarios,
            verbose=cli.verbose,
        )
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
