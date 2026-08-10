import copy
import inspect
import os

import numpy as np
import pandas as pd
import yaml

import pufferlib


MAX_C_SEED = 2**31 - 1
ALL_INFRACTIONS_RENDER_FILTER = "all_infractions"
FAILURE_RENDER_FILTER_COLUMNS = (
    "collision_rate",
    "at_fault_collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
)


def _drive_env_keys():
    from pufferlib.ocean.drive.drive import Drive

    return set(inspect.signature(Drive.__init__).parameters) - {"self"}


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise pufferlib.APIUsageError(f"{label} must be a mapping")
    return value


def _load_yaml_mapping(path, label):
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(path):
        raise pufferlib.APIUsageError(f"{label.capitalize()} not found: {path}")
    try:
        with open(path, "r") as yaml_file:
            value = yaml.safe_load(yaml_file)
    except yaml.YAMLError as exc:
        raise pufferlib.APIUsageError(f"{label.capitalize()} is invalid YAML: {path}") from exc
    return _require_mapping(value, label)


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise pufferlib.APIUsageError(f"{label} must be a positive integer")
    return value


def _resolve_map_indices(map_dir, map_names):
    """Map each logged map name back to its index in the sorted .bin map set."""
    if os.path.isfile(map_dir) and str(map_dir).endswith(".bin"):
        map_files = [map_dir]
    else:
        map_files = sorted(f for f in os.listdir(map_dir) if f.endswith(".bin"))
    name_to_idx = {os.path.basename(f).split(".")[0]: i for i, f in enumerate(map_files)}
    indices = []
    for name in map_names:
        key = os.path.basename(str(name)).split(".")[0]
        if key not in name_to_idx:
            raise pufferlib.APIUsageError(f"Replay map '{key}' not found in {map_dir}")
        indices.append(name_to_idx[key])
    return indices


def validate_training_evaluation_config(args):
    eval_config = args["eval"]
    train_config = args["train"]
    evaluation_interval_epochs = train_config["evaluation_interval_epochs"]
    if evaluation_interval_epochs is None:
        return False

    if (
        isinstance(evaluation_interval_epochs, bool)
        or not isinstance(evaluation_interval_epochs, int)
        or evaluation_interval_epochs <= 0
    ):
        raise pufferlib.APIUsageError("train.evaluation_interval_epochs must be a positive integer or null")

    evaluation_benchmarks = train_config["evaluation_benchmarks"]
    if not isinstance(evaluation_benchmarks, str) or not evaluation_benchmarks.strip():
        raise pufferlib.APIUsageError("train.evaluation_benchmarks must select at least one benchmark")

    load_benchmark_config(eval_config["benchmark_config"], evaluation_benchmarks)
    return True


def load_benchmark_config(config_path, selected_names):
    """Load the shared environment and selected benchmarks."""
    config = _load_yaml_mapping(config_path, "benchmark config")
    environment_config = _require_mapping(config.get("env"), "benchmark config env")
    unknown_env_keys = set(environment_config) - _drive_env_keys()
    if unknown_env_keys:
        raise pufferlib.APIUsageError(
            f"Benchmark config has unsupported environment keys: {', '.join(sorted(unknown_env_keys))}"
        )
    benchmarks = config.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise pufferlib.APIUsageError("Benchmark config must contain a non-empty benchmarks list")

    if isinstance(selected_names, str):
        selected_names = [name.strip() for name in selected_names.split(",") if name.strip()]
    if not isinstance(selected_names, list) or not selected_names:
        raise pufferlib.APIUsageError("At least one benchmark must be selected")
    if any(not isinstance(name, str) or not name for name in selected_names):
        raise pufferlib.APIUsageError("Benchmark names must be non-empty strings")
    selected_names = list(dict.fromkeys(selected_names))

    configured_benchmarks = {}
    for benchmark_idx, benchmark in enumerate(benchmarks):
        benchmark = _require_mapping(benchmark, f"benchmarks[{benchmark_idx}]")
        name = benchmark.get("name")
        if not isinstance(name, str) or not name.strip():
            raise pufferlib.APIUsageError(f"benchmarks[{benchmark_idx}].name must be a non-empty string")
        name = name.strip()
        if name in configured_benchmarks:
            raise pufferlib.APIUsageError(f"Benchmark config contains duplicate benchmark name: {name}")
        configured_benchmarks[name] = benchmark

    missing_names = [name for name in selected_names if name not in configured_benchmarks]
    if missing_names:
        raise pufferlib.APIUsageError(f"Unknown benchmarks: {', '.join(missing_names)}")

    resolved_benchmarks = []
    for name in selected_names:
        benchmark = configured_benchmarks[name]
        benchmark_environment_config = _require_mapping(benchmark.get("env"), f"Benchmark {name} env")
        unknown_env_keys = set(benchmark_environment_config) - _drive_env_keys()
        if unknown_env_keys:
            raise pufferlib.APIUsageError(
                f"Benchmark {name} has unsupported environment keys: {', '.join(sorted(unknown_env_keys))}"
            )
        simulation_mode = benchmark_environment_config.get("simulation_mode")
        if simulation_mode not in ("gigaflow", "replay"):
            raise pufferlib.APIUsageError(f"Benchmark {name} env simulation_mode must be 'gigaflow' or 'replay'")
        seed = benchmark.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_C_SEED:
            raise pufferlib.APIUsageError(f"Benchmark {name} seed must be an integer in [0, {MAX_C_SEED}]")
        num_scenarios = _positive_int(benchmark.get("num_scenarios"), f"Benchmark {name} num_scenarios")
        num_maps = _positive_int(benchmark_environment_config.get("num_maps"), f"Benchmark {name} env num_maps")
        control_mode = benchmark_environment_config.get("control_mode")
        if not isinstance(control_mode, str) or not control_mode:
            raise pufferlib.APIUsageError(f"Benchmark {name} env control_mode must be a non-empty string")
        max_agents_per_env = benchmark_environment_config.get("max_agents_per_env")
        is_single_agent_replay = simulation_mode == "replay" and control_mode == "control_sdc_only"
        if max_agents_per_env is not None or not is_single_agent_replay:
            _positive_int(max_agents_per_env, f"Benchmark {name} env max_agents_per_env")

        map_dir = benchmark_environment_config.get("map_dir")
        if not isinstance(map_dir, str) or not map_dir:
            raise pufferlib.APIUsageError(f"Benchmark {name} env map_dir must be a non-empty path")
        map_dir = os.path.abspath(map_dir)
        if not os.path.isdir(map_dir) and not (os.path.isfile(map_dir) and map_dir.endswith(".bin")):
            raise pufferlib.APIUsageError(f"Benchmark {name} map path does not exist: {map_dir}")

        available_map_count = (
            1
            if os.path.isfile(map_dir)
            else len([filename for filename in os.listdir(map_dir) if filename.endswith(".bin")])
        )
        if num_maps > available_map_count:
            raise pufferlib.APIUsageError(
                f"Benchmark {name} requests {num_maps} maps, but {map_dir} contains {available_map_count}"
            )
        if simulation_mode == "replay" and num_scenarios > available_map_count:
            raise pufferlib.APIUsageError(
                f"Replay benchmark {name} requests {num_scenarios} scenarios, but {map_dir} contains "
                f"{available_map_count} maps"
            )

        resolved_environment_config = copy.deepcopy(benchmark_environment_config)
        resolved_environment_config["map_dir"] = map_dir
        if max_agents_per_env is None:
            resolved_environment_config.pop("max_agents_per_env", None)
        resolved_benchmarks.append(
            {
                "name": name,
                "seed": seed,
                "num_scenarios": num_scenarios,
                "env": resolved_environment_config,
            }
        )
    return environment_config, resolved_benchmarks


def load_checkpoint_architecture(args):
    """Load a 3.0 checkpoint's policy and observation architecture."""
    model_path = args["load_model_path"]
    if not isinstance(model_path, str) or not model_path.endswith(".pt") or not os.path.isfile(model_path):
        raise pufferlib.APIUsageError("Benchmark requires a valid load_model_path checkpoint")
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(model_path))), "config.yaml")
    checkpoint_config = _load_yaml_mapping(config_path, "checkpoint config")

    merged = copy.deepcopy(args)
    for section in ("policy", "rnn"):
        values = _require_mapping(checkpoint_config.get(section), f"checkpoint config {section}")
        merged[section].update(values)
    checkpoint_env = _require_mapping(checkpoint_config.get("env"), "checkpoint config env")
    accepted_env_keys = _drive_env_keys()
    merged["env"].update({key: value for key, value in checkpoint_env.items() if key in accepted_env_keys})
    for key in ("policy_name", "rnn_name"):
        if key not in checkpoint_config:
            raise pufferlib.APIUsageError(f"Checkpoint config is missing {key}")
        merged[key] = checkpoint_config[key]
    merged["train"]["use_rnn"] = merged["rnn_name"] is not None
    return merged, config_path


def build_benchmark_args(base_args, benchmark, environment_config):
    """Apply the benchmark evaluation overrides."""
    args = copy.deepcopy(base_args)
    eval_agent_count = _positive_int(args["eval"]["num_agents"], "eval.num_agents")
    benchmark_environment_config = benchmark["env"]
    max_agents_per_env = benchmark_environment_config.get("max_agents_per_env")
    if max_agents_per_env is not None and eval_agent_count < max_agents_per_env:
        raise pufferlib.APIUsageError(
            f"eval.num_agents ({eval_agent_count}) must be at least benchmark {benchmark['name']} "
            f"max_agents_per_env ({max_agents_per_env})"
        )
    seed = benchmark["seed"]
    args["train"]["seed"] = seed
    args["vec"]["seed"] = seed
    args["env"].update(copy.deepcopy(environment_config))
    args["env"].update(copy.deepcopy(benchmark_environment_config))
    args["env"]["num_agents"] = eval_agent_count
    args["env"]["resample_frequency"] = benchmark_environment_config["scenario_length"]
    args["num_scenarios"] = benchmark["num_scenarios"]
    return args


def _plan_benchmark_eval_workers(args, num_scenarios, num_workers, scenario_length, capture_replay=False):
    """One disjoint contiguous map window per worker; together they cover the set once."""
    scenarios_per_worker, remainder = divmod(num_scenarios, num_workers)
    worker_env_kwargs = []
    next_map_idx = 0
    for worker_idx in range(num_workers):
        worker_num_scenarios = scenarios_per_worker + (1 if worker_idx < remainder else 0)
        env_kwargs = copy.deepcopy(args["env"])
        env_kwargs["eval_mode"] = 1
        env_kwargs["starting_map"] = next_map_idx
        env_kwargs["num_eval_scenarios"] = worker_num_scenarios
        env_kwargs["resample_frequency"] = scenario_length
        env_kwargs["capture_replay"] = capture_replay
        env_kwargs["replay_worker_idx"] = worker_idx
        worker_env_kwargs.append(env_kwargs)
        next_map_idx += worker_num_scenarios
    max_scenarios_per_worker = scenarios_per_worker + (1 if remainder else 0)
    return worker_env_kwargs, max_scenarios_per_worker * scenario_length


def _plan_failure_replay_workers(args, map_seed_pairs, num_workers, scenario_length):
    """Split the (map, seed) pairs across workers; each worker cycles through its
    pairs in fit-aware batches (num_agents from config bounds a batch)."""
    pairs_per_worker, remainder = divmod(len(map_seed_pairs), num_workers)
    worker_env_kwargs = []
    pair_start = 0
    for worker_idx in range(num_workers):
        worker_pair_count = pairs_per_worker + (1 if worker_idx < remainder else 0)
        worker_pairs = map_seed_pairs[pair_start : pair_start + worker_pair_count]
        pair_start += worker_pair_count
        env_kwargs = copy.deepcopy(args["env"])
        env_kwargs["eval_mode"] = 1
        env_kwargs["resample_frequency"] = scenario_length
        env_kwargs["starting_map"] = 0
        env_kwargs["num_eval_scenarios"] = worker_pair_count
        env_kwargs["eval_map_indices"] = [map_idx for map_idx, _ in worker_pairs]
        env_kwargs["eval_scenario_seeds"] = [seed for _, seed in worker_pairs]
        env_kwargs["capture_replay"] = True
        env_kwargs["replay_worker_idx"] = worker_idx
        worker_env_kwargs.append(env_kwargs)
    max_pairs_per_worker = pairs_per_worker + (1 if remainder else 0)
    return worker_env_kwargs, max_pairs_per_worker * scenario_length


def write_resolved_benchmark_config(args, benchmark, benchmark_config_path, checkpoint_config_path, output_path):
    resolved = {
        "benchmark_config": os.path.abspath(benchmark_config_path),
        "checkpoint_config": os.path.abspath(checkpoint_config_path) if checkpoint_config_path is not None else None,
        "benchmark": benchmark,
        "args": args,
    }
    with open(output_path, "w") as output_file:
        yaml.safe_dump(resolved, output_file, sort_keys=False)


def _build_eval_report(episode_summaries, num_scenarios):
    if not episode_summaries:
        return None

    df = pd.DataFrame(episode_summaries)
    df = df.drop(columns=[col for col in ("summary_type", "env_slot") if col in df.columns])
    if "map_name" in df.columns:
        df["map_name"] = df["map_name"].map(lambda name: os.path.basename(str(name)).split(".")[0])
    lead_cols = [col for col in ("map_name", "scenario_id") if col in df.columns]
    df = df[lead_cols + [col for col in df.columns if col not in lead_cols]]

    metric_means = df.drop(columns=["seed"], errors="ignore").select_dtypes(include=[np.number]).mean().to_dict()
    summary = {
        "num_scenarios": num_scenarios,
        "num_episodes": int(len(df)),
        "metrics_mean": {key: float(value) for key, value in metric_means.items()},
    }
    return df, summary


def _write_eval_reports(episode_summaries, out_dir, num_scenarios):
    """Write a per-episode metrics CSV and a JSON of metric averages to out_dir."""
    import json

    report = _build_eval_report(episode_summaries, num_scenarios)
    if report is None:
        print("No evaluation episodes were recorded; skipping report.")
        return None

    df, summary = report
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "episode_metrics.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(out_dir, "evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(df)} per-episode rows to {csv_path}")
    print(f"Wrote metric averages to {json_path}")
    return summary


def parse_render_filter_columns(configured_render_filter):
    """Resolve and validate the metric columns used to select rendered episodes."""
    if isinstance(configured_render_filter, str):
        render_filter_columns = [column.strip() for column in configured_render_filter.split(",") if column.strip()]
    elif isinstance(configured_render_filter, (list, tuple)):
        render_filter_columns = list(configured_render_filter)
    else:
        raise pufferlib.APIUsageError("eval.render_filter must be a comma-separated string or list")

    if not render_filter_columns or any(not isinstance(column, str) or not column for column in render_filter_columns):
        raise pufferlib.APIUsageError("eval.render_filter must contain non-empty metric names")

    resolved_render_filter_columns = []
    for column in render_filter_columns:
        if column == ALL_INFRACTIONS_RENDER_FILTER:
            resolved_render_filter_columns.extend(FAILURE_RENDER_FILTER_COLUMNS)
            continue
        resolved_render_filter_columns.append(column)
    return tuple(dict.fromkeys(resolved_render_filter_columns))


def select_render_rows(metrics_path, configured_render_filter):
    """Select rows where any configured render metric is greater than zero."""
    if not os.path.isfile(metrics_path):
        raise pufferlib.APIUsageError(f"Benchmark metrics CSV not found: {metrics_path}")
    rows = pd.read_csv(metrics_path)
    render_filter_columns = parse_render_filter_columns(configured_render_filter)
    missing_columns = [column for column in render_filter_columns if column not in rows.columns]
    if missing_columns:
        raise pufferlib.APIUsageError(
            f"Benchmark metrics CSV is missing configured render filter columns: {', '.join(missing_columns)}"
        )
    selected = pd.Series(False, index=rows.index)
    for column in render_filter_columns:
        selected |= rows[column] > 0
    return rows[selected].copy()
