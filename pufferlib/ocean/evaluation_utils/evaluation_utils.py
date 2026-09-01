import copy
import os

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

import pufferlib
import pufferlib.pytorch
import pufferlib.utils
from pufferlib.config_schema import (
    normalize_puffer_drive_benchmarks,
    normalize_puffer_drive_config,
    puffer_drive_constructor_keys,
    validate_puffer_drive_config,
    validate_puffer_drive_resources,
)


ALL_INFRACTIONS_RENDER_FILTER = "all_infractions"
FAILURE_RENDER_FILTER_COLUMNS = (
    "collision_rate",
    "at_fault_collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
)


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise pufferlib.APIUsageError(f"{label} must be a mapping")
    return value


def resolve_run_dir(model_path):
    """Run directory holding a checkpoint's config.yaml.

    Checkpoints sit either at the run root (train.final_model_name) or one level
    down in models/ and best_models/, so accept both instead of assuming a depth.
    """
    checkpoint_dir = os.path.dirname(os.path.abspath(model_path))
    if os.path.isfile(os.path.join(checkpoint_dir, "config.yaml")):
        return checkpoint_dir
    return os.path.dirname(checkpoint_dir)


def _load_yaml_mapping(path, label):
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(path):
        raise pufferlib.APIUsageError(f"{label.capitalize()} not found: {path}")
    try:
        with open(path, "r") as yaml_file:
            value = yaml.safe_load(yaml_file)
    except yaml.YAMLError as exc:
        raise pufferlib.APIUsageError(f"{label.capitalize()} is invalid YAML: {path}") from exc
    return _require_mapping(value, label)


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
    """Load benchmark sources and resolve selection without validating final values."""
    config = _load_yaml_mapping(config_path, "benchmark config")
    environment_config = _require_mapping(config.get("env"), "benchmark config env")
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

    selected_benchmark_configs = [configured_benchmarks[name] for name in selected_names]
    resolved_benchmarks = normalize_puffer_drive_benchmarks(
        environment_config,
        selected_benchmark_configs,
        "benchmark_config",
    )
    return environment_config, resolved_benchmarks


def load_checkpoint_architecture(args):
    """Load a 3.0 checkpoint's policy and observation architecture."""
    model_path = args["load_model_path"]
    if not isinstance(model_path, str) or not model_path.endswith(".pt") or not os.path.isfile(model_path):
        raise pufferlib.APIUsageError("Benchmark requires a valid load_model_path checkpoint")
    config_path = os.path.join(resolve_run_dir(model_path), "config.yaml")
    checkpoint_config = _load_yaml_mapping(config_path, "checkpoint config")

    merged = copy.deepcopy(args)
    for section in ("policy", "rnn"):
        values = _require_mapping(checkpoint_config.get(section), f"checkpoint config {section}")
        merged[section].update(values)
    checkpoint_env = _require_mapping(checkpoint_config.get("env"), "checkpoint config env")
    accepted_env_keys = puffer_drive_constructor_keys()
    merged["env"].update({key: value for key, value in checkpoint_env.items() if key in accepted_env_keys})
    for key in ("policy_name", "rnn_name"):
        if key not in checkpoint_config:
            raise pufferlib.APIUsageError(f"Checkpoint config is missing {key}")
        merged[key] = checkpoint_config[key]
    merged["train"]["use_rnn"] = merged["rnn_name"] is not None
    return merged, config_path


CHECKPOINT_RUN_IDENTITY_KEYS = ("run_name", "wandb_project", "wandb_group")


def load_checkpoint_run_identity(config_path):
    """Resolve the tracker run a checkpoint belongs to from its training config."""
    checkpoint_config = _load_yaml_mapping(config_path, "checkpoint config")
    identity = {}
    for key in CHECKPOINT_RUN_IDENTITY_KEYS:
        value = checkpoint_config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise pufferlib.APIUsageError(
                f"Checkpoint config {config_path} is missing a usable {key}; cannot attach eval "
                "results to the training run"
            )
        identity[key] = value
    return identity


def summarize_benchmark_metrics(benchmark_results, key_prefix):
    """Flatten benchmark results into one scalar metric dict for the experiment logger."""
    metrics = {}
    for benchmark_name, benchmark_result in benchmark_results.items():
        summary = benchmark_result["summary"]
        if summary is None:
            continue
        prefix = f"{key_prefix}{benchmark_name}"
        metrics[f"{prefix}/num_scenarios"] = summary["num_scenarios"]
        metrics[f"{prefix}/num_episodes"] = summary["num_episodes"]
        metrics.update({f"{prefix}/{key}": value for key, value in summary["metrics_mean"].items()})
    return metrics


def _build_benchmark_args(base_args, benchmark, environment_config):
    """Apply benchmark evaluation overrides without validating the result."""
    args = copy.deepcopy(base_args)
    eval_training_render = args["env"]["eval_training_render"]
    eval_agent_count = args["eval"]["num_agents"]
    benchmark_environment_config = benchmark["env"]
    seed = benchmark["seed"]
    args["train"]["seed"] = seed
    args["vec"]["seed"] = seed
    if eval_training_render:
        args["eval"]["action_selection"] = pufferlib.pytorch.ACTION_SELECT_SAMPLE
    else:
        args["env"].update(copy.deepcopy(environment_config))
        args["env"].update(copy.deepcopy(benchmark_environment_config))
    args["env"]["num_agents"] = eval_agent_count
    args["env"]["resample_frequency"] = args["env"]["scenario_length"]
    args["num_scenarios"] = benchmark["num_scenarios"]
    return args


def _finalize_benchmark_args(args, cli_overrides, eval_training_render, validation_context):
    cli_override_config = OmegaConf.from_dotlist(list(cli_overrides))
    args = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.create(dict(args)), cli_override_config),
        resolve=True,
    )
    args["env"]["eval_training_render"] = eval_training_render
    args["env"]["eval_mode"] = 1
    args["env"]["num_agents"] = args["eval"]["num_agents"]
    if eval_training_render:
        args["env"]["compute_eval_metrics"] = True
        args["env"]["resample_frequency"] = args["env"]["scenario_length"]

    args = normalize_puffer_drive_config(args, validation_context)
    single_agent_replay = (
        args["env"]["simulation_mode"] == "replay" and args["env"]["control_mode"] == "control_sdc_only"
    )
    if single_agent_replay:
        args["vec"]["num_envs"] = min(
            args["vec"]["num_envs"],
            args["eval"]["max_sdc_replay_workers"],
        )
    validate_puffer_drive_config(args, validation_context)
    validate_puffer_drive_resources(args, validation_context)
    return args


def build_benchmark_args(base_args, benchmark, environment_config, cli_overrides=()):
    """Compose and validate final arguments for one benchmark evaluation."""
    eval_training_render = base_args["env"]["eval_training_render"]
    validation_context = f"evaluation.{benchmark['name']}"
    if eval_training_render:
        benchmark_validation_base_args = copy.deepcopy(base_args)
        benchmark_validation_base_args["env"]["eval_training_render"] = False
        benchmark_validation_args = _build_benchmark_args(
            benchmark_validation_base_args,
            benchmark,
            environment_config,
        )
        _finalize_benchmark_args(
            benchmark_validation_args,
            cli_overrides,
            False,
            validation_context,
        )

    args = _build_benchmark_args(base_args, benchmark, environment_config)
    return _finalize_benchmark_args(
        args,
        cli_overrides,
        eval_training_render,
        validation_context,
    )


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

    numeric_metrics = df.drop(columns=["seed"], errors="ignore").select_dtypes(include=[np.number])
    metric_lists = {key: numeric_metrics[key].dropna().tolist() for key in numeric_metrics}
    metric_means = pufferlib.utils.reduce_environment_metrics(metric_lists)
    summary = {
        "num_scenarios": num_scenarios,
        "num_episodes": int(len(df)),
        "metrics_mean": metric_means,
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
