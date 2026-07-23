import copy
import inspect
import os

import pandas as pd
import yaml

import pufferlib


MAX_C_SEED = 2**31 - 1
FAILURE_METRIC_COLUMNS = (
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


def _seed(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_C_SEED:
        raise pufferlib.APIUsageError(f"{label} must be an integer in [0, {MAX_C_SEED}]")
    return value


def validate_training_eval_config(args):
    eval_config = args.get("eval", {})
    training_enabled = eval_config.get("training_enabled", False)
    if not isinstance(training_enabled, bool):
        raise pufferlib.APIUsageError("eval.training_enabled must be true or false")
    if not training_enabled:
        return False

    training_interval = eval_config.get("training_interval")
    if isinstance(training_interval, bool) or not isinstance(training_interval, int) or training_interval <= 0:
        raise pufferlib.APIUsageError("eval.training_interval must be a positive integer")

    training_datasets = eval_config.get("training_datasets")
    if not isinstance(training_datasets, str) or not training_datasets.strip():
        raise pufferlib.APIUsageError("eval.training_datasets must select at least one benchmark dataset")
    if args["train"].get("use_rnn"):
        raise pufferlib.APIUsageError("Multiprocessed training evaluation does not support RNN policies yet")

    load_catalog(eval_config.get("catalog"), training_datasets)
    load_evaluation_config(eval_config.get("evaluation_config"))
    return True


def load_catalog(catalog_path, selected_names):
    """Load benchmark suites and apply one deterministic catalog seed."""
    catalog = _load_yaml_mapping(catalog_path, "benchmark catalog")
    catalog_seed = _seed(catalog.get("seed", 42), "benchmark catalog seed")
    suites = catalog.get("benchmarks")
    if not isinstance(suites, list) or not suites:
        raise pufferlib.APIUsageError("Benchmark catalog must contain a non-empty benchmarks list")

    if isinstance(selected_names, str):
        selected_names = [name.strip() for name in selected_names.split(",") if name.strip()]
    if not isinstance(selected_names, list) or not selected_names:
        raise pufferlib.APIUsageError("At least one benchmark dataset must be selected")
    if any(not isinstance(name, str) or not name for name in selected_names):
        raise pufferlib.APIUsageError("Benchmark dataset names must be non-empty strings")
    selected_names = list(dict.fromkeys(selected_names))

    catalog_suites = {}
    for suite_idx, suite in enumerate(suites):
        suite = _require_mapping(suite, f"benchmarks[{suite_idx}]")
        name = suite.get("name")
        if not isinstance(name, str) or not name.strip():
            raise pufferlib.APIUsageError(f"benchmarks[{suite_idx}].name must be a non-empty string")
        name = name.strip()
        if name in catalog_suites:
            raise pufferlib.APIUsageError(f"Benchmark catalog contains duplicate suite name: {name}")
        catalog_suites[name] = suite

    missing_names = [name for name in selected_names if name not in catalog_suites]
    if missing_names:
        raise pufferlib.APIUsageError(f"Unknown benchmark datasets: {', '.join(missing_names)}")

    resolved_suites = []
    for name in selected_names:
        suite = catalog_suites[name]
        mode = suite.get("mode")
        if mode not in ("gigaflow", "replay"):
            raise pufferlib.APIUsageError(f"Benchmark {name} mode must be 'gigaflow' or 'replay'")
        num_scenarios = _positive_int(suite.get("num_scenarios"), f"Benchmark {name} num_scenarios")
        scenario_length = _positive_int(suite.get("scenario_length"), f"Benchmark {name} scenario_length")
        max_agents_per_env = _positive_int(suite.get("max_agents_per_env", 64), f"Benchmark {name} max_agents_per_env")
        num_maps = suite.get("num_maps", num_scenarios if mode == "replay" else None)
        num_maps = _positive_int(num_maps, f"Benchmark {name} num_maps")
        control_mode = suite.get("control_mode")
        if not isinstance(control_mode, str) or not control_mode:
            raise pufferlib.APIUsageError(f"Benchmark {name} control_mode must be a non-empty string")

        paths = _require_mapping(suite.get("paths"), f"Benchmark {name} paths")
        map_dir = paths.get("local")
        if not isinstance(map_dir, str) or not map_dir:
            raise pufferlib.APIUsageError(f"Benchmark {name} paths.local must be a non-empty path")
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
        if mode == "replay" and num_scenarios > available_map_count:
            raise pufferlib.APIUsageError(
                f"Replay benchmark {name} requests {num_scenarios} scenarios, but {map_dir} contains "
                f"{available_map_count} maps"
            )

        resolved_suites.append(
            {
                "name": name,
                "mode": mode,
                "seed": _seed(suite.get("seed", catalog_seed), f"Benchmark {name} seed"),
                "num_scenarios": num_scenarios,
                "num_maps": num_maps,
                "max_agents_per_env": max_agents_per_env,
                "scenario_length": scenario_length,
                "control_mode": control_mode,
                "map_dir": map_dir,
            }
        )
    return resolved_suites


def load_evaluation_config(config_path):
    """Load the environment overrides shared by every benchmark suite."""
    config = _load_yaml_mapping(config_path, "benchmark evaluation config")
    env_config = _require_mapping(config.get("env"), "benchmark evaluation config env")
    unknown_env_keys = set(env_config) - _drive_env_keys()
    if unknown_env_keys:
        raise pufferlib.APIUsageError(
            f"Benchmark evaluation config has unsupported environment keys: {', '.join(sorted(unknown_env_keys))}"
        )
    return copy.deepcopy(env_config)


def load_checkpoint_architecture(args):
    """Load a 3.0 checkpoint's policy and observation architecture."""
    model_path = args.get("load_model_path")
    if not isinstance(model_path, str) or not model_path.endswith(".pt") or not os.path.isfile(model_path):
        raise pufferlib.APIUsageError("Benchmark requires a valid load_model_path checkpoint")
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(model_path))), "config.yaml")
    checkpoint_config = _load_yaml_mapping(config_path, "checkpoint config")

    merged = copy.deepcopy(args)
    for section in ("policy", "rnn"):
        values = _require_mapping(checkpoint_config.get(section), f"checkpoint config {section}")
        merged[section].update(copy.deepcopy(values))
    checkpoint_env = _require_mapping(checkpoint_config.get("env"), "checkpoint config env")
    accepted_env_keys = _drive_env_keys()
    merged["env"].update(
        {key: copy.deepcopy(value) for key, value in checkpoint_env.items() if key in accepted_env_keys}
    )
    for key in ("policy_name", "rnn_name"):
        if key not in checkpoint_config:
            raise pufferlib.APIUsageError(f"Checkpoint config is missing {key}")
        merged[key] = checkpoint_config[key]
    merged["train"]["use_rnn"] = merged["rnn_name"] is not None
    return merged, config_path


def build_suite_args(base_args, suite, evaluation_env_config):
    """Apply the fixed benchmark evaluation overrides."""
    args = copy.deepcopy(base_args)
    eval_agent_count = _positive_int(args["eval"].get("num_agents", 101), "eval.num_agents")
    if eval_agent_count < suite["max_agents_per_env"]:
        raise pufferlib.APIUsageError(
            f"eval.num_agents ({eval_agent_count}) must be at least benchmark {suite['name']} "
            f"max_agents_per_env ({suite['max_agents_per_env']})"
        )
    seed = suite["seed"]
    args["train"]["seed"] = seed
    args["vec"]["seed"] = seed
    args["env"].update(copy.deepcopy(evaluation_env_config))
    args["env"].update(
        {
            "num_agents": eval_agent_count,
            "simulation_mode": suite["mode"],
            "map_dir": suite["map_dir"],
            "num_maps": suite["num_maps"],
            "scenario_length": suite["scenario_length"],
            "resample_frequency": suite["scenario_length"],
            "max_agents_per_env": suite["max_agents_per_env"],
            "control_mode": suite["control_mode"],
        }
    )
    args["num_scenarios"] = suite["num_scenarios"]
    return args


def write_resolved_benchmark_config(
    args, suite, catalog_path, evaluation_config_path, checkpoint_config_path, output_path
):
    import json

    resolved = {
        "benchmark_catalog": os.path.abspath(catalog_path),
        "benchmark_evaluation_config": os.path.abspath(evaluation_config_path),
        "checkpoint_config": os.path.abspath(checkpoint_config_path) if checkpoint_config_path is not None else None,
        "suite": suite,
        "args": json.loads(json.dumps(args)),
    }
    with open(output_path, "w") as output_file:
        yaml.safe_dump(resolved, output_file, sort_keys=False)


def parse_failure_metric_columns(configured_failure_metrics):
    """Resolve and validate the metric columns that define a failed episode."""
    if configured_failure_metrics is None:
        return FAILURE_METRIC_COLUMNS
    if isinstance(configured_failure_metrics, str):
        failure_metric_columns = [column.strip() for column in configured_failure_metrics.split(",") if column.strip()]
    elif isinstance(configured_failure_metrics, (list, tuple)):
        failure_metric_columns = list(configured_failure_metrics)
    else:
        raise pufferlib.APIUsageError("eval.failure_metrics must be a comma-separated string or list")

    if not failure_metric_columns or any(
        not isinstance(column, str) or not column for column in failure_metric_columns
    ):
        raise pufferlib.APIUsageError("eval.failure_metrics must contain non-empty metric names")
    failure_metric_columns = tuple(dict.fromkeys(failure_metric_columns))
    unknown_columns = set(failure_metric_columns) - set(FAILURE_METRIC_COLUMNS)
    if unknown_columns:
        raise pufferlib.APIUsageError(
            "eval.failure_metrics contains unsupported metrics: "
            f"{', '.join(sorted(unknown_columns))}. Supported metrics: {', '.join(FAILURE_METRIC_COLUMNS)}"
        )
    return failure_metric_columns


def select_failure_rows(metrics_path, configured_failure_metrics=None):
    """Select failures using the infraction columns emitted by PufferDrive."""
    if not os.path.isfile(metrics_path):
        raise pufferlib.APIUsageError(f"Benchmark metrics CSV not found: {metrics_path}")
    rows = pd.read_csv(metrics_path)
    failure_metric_columns = parse_failure_metric_columns(configured_failure_metrics)
    present_columns = [column for column in failure_metric_columns if column in rows.columns]
    if not present_columns:
        raise pufferlib.APIUsageError(
            f"Benchmark metrics CSV has none of the configured failure columns: {', '.join(failure_metric_columns)}"
        )
    failure = pd.Series(False, index=rows.index)
    for column in present_columns:
        failure |= pd.to_numeric(rows[column], errors="coerce").fillna(0) > 0
    return rows[failure].copy()
