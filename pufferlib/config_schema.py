"""PufferDrive structured schemas and final configuration validation.

The dataclasses in this module define the admissible shape, types, and enum
values for configuration input. Required fields use ``MISSING`` so absent YAML
or overlay values fail explicitly. Range, cross-field, and filesystem rules
that cannot be expressed by OmegaConf's structured configs live alongside the
schemas below.

Enum int values mirror the #defines in pufferlib/ocean/drive/drive.h. We enforce
the match via tests via tests/unit_tests/test_config_schema.py. If a python enum
class does not have corresponding int-class entries to the respective C enum
the test fails.

Enums classes are matched to the respective C counterparts via name matching. The
naming convention is CamelCase for python classes, LowerPascalCase for python values
and PascalCase for C class names and values: every C #define is
`<ENUM_CLASS_NAME_IN_SCREAMING_SNAKE>_<MEMBER_NAME_UPPER>`, e.g.
`InfractionBehavior.stop` <-> `INFRACTION_BEHAVIOR_STOP`.
"""

import copy
import inspect
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, get_args

from omegaconf import MISSING, OmegaConf
from omegaconf.errors import OmegaConfBaseException

import pufferlib


POSITIVE_INT_CONSTRAINT = 1
NONNEGATIVE_INT_CONSTRAINT = 2
ZERO_OR_ONE_INT_CONSTRAINT = 3
POSITIVE_NUMBER_CONSTRAINT = 4
NONNEGATIVE_NUMBER_CONSTRAINT = 5
PROBABILITY_CONSTRAINT = 6
NONEMPTY_STRING_CONSTRAINT = 7
FINITE_NUMBER_CONSTRAINT = 8


def _raise_config_error(context, path, message):
    """Raise a config error whose path is qualified by the caller's context."""
    location = f"{context}.{path}" if context else path
    raise pufferlib.APIUsageError(f"Invalid PufferDrive configuration at {location}: {message}")


def _constrained_field(constraint_mode, default=MISSING):
    """Declare a dataclass field carrying a runtime value constraint."""
    return field(default=default, metadata={"constraint_mode": constraint_mode})


def _validate_value_constraint(value, constraint_mode, context, path):
    """Validate one scalar value against its configured constraint."""
    if value is None and constraint_mode in (
        POSITIVE_INT_CONSTRAINT,
        NONNEGATIVE_INT_CONSTRAINT,
        POSITIVE_NUMBER_CONSTRAINT,
        NONNEGATIVE_NUMBER_CONSTRAINT,
    ):
        return

    if constraint_mode == POSITIVE_INT_CONSTRAINT:
        valid = not isinstance(value, bool) and isinstance(value, int) and value > 0
        message = "must be a positive integer"
    elif constraint_mode == NONNEGATIVE_INT_CONSTRAINT:
        valid = not isinstance(value, bool) and isinstance(value, int) and value >= 0
        message = "must be a non-negative integer"
    elif constraint_mode == ZERO_OR_ONE_INT_CONSTRAINT:
        valid = not isinstance(value, bool) and value in (0, 1)
        message = "must be 0 or 1"
    elif constraint_mode == POSITIVE_NUMBER_CONSTRAINT:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        message = "must be positive"
    elif constraint_mode == NONNEGATIVE_NUMBER_CONSTRAINT:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        message = "must be non-negative"
    elif constraint_mode == PROBABILITY_CONSTRAINT:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and 0.0 <= value <= 1.0
        )
        message = "must be in [0, 1]"
    elif constraint_mode == NONEMPTY_STRING_CONSTRAINT:
        valid = isinstance(value, str) and bool(value.strip())
        message = "must be a non-empty string"
    elif constraint_mode == FINITE_NUMBER_CONSTRAINT:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        message = "must be finite"
    else:
        raise RuntimeError(f"Unknown config constraint mode: {constraint_mode}")
    if not valid:
        _raise_config_error(context, path, message)


class SimulationMode(Enum):
    gigaflow = 0
    replay = 1


class ActionType(Enum):
    discrete = 0
    continuous = 1


class DynamicsModel(Enum):
    classic = 0
    jerk = 1


class InfractionBehavior(Enum):
    ignore = 0
    stop = 1
    remove = 2


class ControlMode(Enum):
    control_vehicles = 0
    control_agents = 1
    control_wosac = 2
    control_sdc_only = 3


class Controller(Enum):
    static = 0
    policy = 1
    replay = 2
    idm = 3


class NonVehicleController(Enum):
    # "auto" is config-side only: drive.py resolves it to a Controller
    # (replay when non_sdc_controller is idm, else non_sdc_controller).
    auto = -1
    static = 0
    policy = 1
    replay = 2
    idm = 3


class InitMode(Enum):
    create_all_valid = 0
    create_only_controlled = 1


class GoalRegen(Enum):
    finite = 0
    rolling = 1


class GoalSource(Enum):
    route = 0
    map = 1
    gt = 2


class PackageName(Enum):
    ocean = 0


class EnvironmentName(Enum):
    puffer_drive = 0


class PolicyName(Enum):
    Drive = 0


class RNNName(Enum):
    Recurrent = 0


class Activation(Enum):
    relu = 0
    tanh = 1
    gelu = 2


class Optimizer(Enum):
    adam = 0
    adamw = 1
    muon = 2


class Precision(Enum):
    float32 = 0
    bfloat16 = 1


class RolloutDtype(Enum):
    float32 = 0
    float16 = 1


class VectorBackend(Enum):
    PufferEnv = 0
    Serial = 1
    Multiprocessing = 2
    Ray = 3


class ActionSelection(Enum):
    sample = 0
    mode = 1
    mean = 2


@dataclass
class VectorConfig:
    backend: VectorBackend = MISSING
    num_envs: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    num_workers: int | str = MISSING
    batch_size: int | str | None = MISSING
    zero_copy: bool = MISSING
    seed: int | None = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)


@dataclass
class DriveEnvConfig:
    simulation_mode: SimulationMode = MISSING
    num_agents: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    min_agents_per_env: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    max_agents_per_env: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    action_type: ActionType = MISSING
    dynamics_model: DynamicsModel = MISSING
    reset_accel_on_stop: bool = MISSING
    dt: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    base_max_speed_mps: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    spawn_initial_speed: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    collision_behavior: InfractionBehavior = MISSING
    offroad_behavior: InfractionBehavior = MISSING
    traffic_light_behavior: InfractionBehavior = MISSING
    use_map_cache: int = _constrained_field(ZERO_OR_ONE_INT_CONSTRAINT)
    use_neighbor_cache: int = _constrained_field(ZERO_OR_ONE_INT_CONSTRAINT)
    scenario_length: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    resample_frequency: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    termination_mode: int = _constrained_field(ZERO_OR_ONE_INT_CONSTRAINT)
    inactive_agent_threshold: float = _constrained_field(PROBABILITY_CONSTRAINT)
    terminate_on_goal: int = _constrained_field(ZERO_OR_ONE_INT_CONSTRAINT)
    init_step: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    init_step_spread: bool = MISSING
    init_step_min_horizon: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    control_mode: ControlMode = MISSING
    sdc_controller: Controller = MISSING
    non_sdc_controller: Controller = MISSING
    non_vehicle_controller: NonVehicleController = MISSING
    init_mode: InitMode = MISSING
    compute_eval_metrics: bool = MISSING
    eval_training_render: bool = MISSING
    goal_regen_mode: GoalRegen = MISSING
    goal_source: GoalSource = MISSING
    obs_goal_lane_distance: bool = MISSING
    goal_radius: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    goal_speed: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    num_goals: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    min_goal_spacing: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    max_goal_spacing: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    reward_conditioning: bool = MISSING
    reward_randomization: bool = MISSING
    reward_log_sampling: bool = MISSING
    reward_goal: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_collision: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_offroad: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_stop_line: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_comfort: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_lane_align: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_vel_align: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_lane_center: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_center_bias: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_velocity: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_reverse: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_timestep: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_overspeed: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    reward_ade: float = _constrained_field(FINITE_NUMBER_CONSTRAINT)
    map_dir: str = MISSING
    num_maps: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    obs_slots_lane_n: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    obs_slots_boundary_n: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    obs_slots_partners_n: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    obs_slots_traffic_controls_n: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    obs_dropout_lane: float = _constrained_field(PROBABILITY_CONSTRAINT)
    obs_dropout_boundary: float = _constrained_field(PROBABILITY_CONSTRAINT)
    obs_lane_stride: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    obs_boundary_stride: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    obs_norm_speed_mps: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_norm_goal_offset_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_norm_xy_offset_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_norm_veh_length_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_norm_veh_width_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_norm_road_seg_length_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_norm_road_seg_width_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_norm_z_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    eval_perceived_size_margin_m: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    obs_range_road_front_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_range_road_behind_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_range_road_side_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_range_partner_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    obs_range_traffic_control_m: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    partner_blindness_prob: float = _constrained_field(PROBABILITY_CONSTRAINT)
    partner_blindness_trigger_prob: float = _constrained_field(PROBABILITY_CONSTRAINT)
    partner_blindness_duration_seconds: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    phantom_braking_prob: float = _constrained_field(PROBABILITY_CONSTRAINT)
    phantom_braking_trigger_prob: float = _constrained_field(PROBABILITY_CONSTRAINT)
    phantom_braking_duration_seconds: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)

    # Added by benchmark overlays after the base Hydra config is composed.
    eval_mode: bool | int = 0
    max_scenarios_per_batch: int | None = _constrained_field(POSITIVE_INT_CONSTRAINT, default=None)


@dataclass
class DrivePolicyConfig:
    ego_input_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    partner_input_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    lane_input_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    boundary_input_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    traffic_control_input_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    context_input_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    encoder_activation: Activation = MISSING
    encoder_layer_norm: bool = MISSING
    mask_padded_features: bool = MISSING
    backbone_hidden_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    backbone_num_layers: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    backbone_activation: Activation = MISSING
    backbone_layer_norm: bool = MISSING
    actor_hidden_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    actor_num_layers: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    actor_head_layer_norm: bool = MISSING
    critic_hidden_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    critic_num_layers: int = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    critic_head_layer_norm: bool = MISSING
    shared_network: bool = MISSING
    action_type: ActionType = MISSING


@dataclass
class RecurrentConfig:
    input_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    hidden_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)


@dataclass
class TrainingConfig:
    name: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    project: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    seed: int | None = MISSING
    final_model_name: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    evaluation_interval_epochs: int | None = _constrained_field(POSITIVE_INT_CONSTRAINT)
    # OmegaConf does not support unions between scalar and container types.
    # These selection fields are checked explicitly after structured merging.
    evaluation_benchmarks: Any = MISSING
    torch_deterministic: bool = MISSING
    cpu_offload: bool = MISSING
    device: str | int = MISSING
    optimizer: Optimizer = MISSING
    anneal_lr: bool = MISSING
    precision: Precision = MISSING
    rollout_dtype: RolloutDtype = MISSING
    total_timesteps: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    learning_rate: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    gamma: float = _constrained_field(PROBABILITY_CONSTRAINT)
    gae_lambda: float = _constrained_field(PROBABILITY_CONSTRAINT)
    update_epochs: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    clip_coef: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    vf_coef: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    vf_clip_coef: float | None = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    max_grad_norm: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    normalize_rewards: bool = MISSING
    ent_coef: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    use_value_bootstrapping: bool = MISSING
    adam_beta1: float = _constrained_field(PROBABILITY_CONSTRAINT)
    adam_beta2: float = _constrained_field(PROBABILITY_CONSTRAINT)
    adam_eps: float = _constrained_field(POSITIVE_NUMBER_CONSTRAINT)
    adam_weight_decay: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    data_dir: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    resume_state_path: str | None = MISSING
    checkpoint_interval: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    batch_size: int | str = MISSING
    min_batch_size: int | None = _constrained_field(POSITIVE_INT_CONSTRAINT)
    minibatch_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    max_minibatch_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    bptt_horizon: int | str = MISSING
    compile: bool = MISSING
    compile_mode: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    compile_fullgraph: bool = MISSING
    vtrace_rho_clip: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    vtrace_c_clip: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    adv_sampling_prio_alpha: float = _constrained_field(PROBABILITY_CONSTRAINT)
    adv_sampling_prio_beta0: float = _constrained_field(PROBABILITY_CONSTRAINT)
    adv_filter_enabled: bool = MISSING
    adv_filter_ewma_beta: float = _constrained_field(PROBABILITY_CONSTRAINT)
    adv_filter_threshold_scale: float = _constrained_field(NONNEGATIVE_NUMBER_CONSTRAINT)
    render: bool = MISSING
    render_interval: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    obs_only: bool = MISSING
    show_grid: bool = MISSING
    show_lasers: bool = MISSING
    show_human_logs: bool = MISSING
    render_map: Any = MISSING

    # Derived by load_config from rnn_name and intentionally absent from YAML.
    use_rnn: bool = MISSING


@dataclass
class EvaluationConfig:
    num_agents: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    max_sdc_replay_workers: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    benchmark_config: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    benchmarks: Any = MISSING
    output_name: str | None = MISSING
    output_dir_name: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    render_scenarios: bool = MISSING
    render_filter: Any = MISSING
    max_rendered_failures: int | None = _constrained_field(POSITIVE_INT_CONSTRAINT)
    failure_replay_csv: str | None = MISSING
    capture_observations: bool = MISSING
    observation_replay_wave_size: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    observation_replay_writer_count: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    action_selection: ActionSelection = MISSING


@dataclass
class PufferDriveConfig:
    load_model_path: str | None = MISSING
    load_id: str | None = MISSING
    render_mode: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    video_path: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    num_scenarios: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    render: int = _constrained_field(ZERO_OR_ONE_INT_CONSTRAINT)
    agent_index: int | None = _constrained_field(NONNEGATIVE_INT_CONSTRAINT)
    save_frames: int = _constrained_field(ZERO_OR_ONE_INT_CONSTRAINT)
    gif_path: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    fps: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    max_runs: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    wandb: bool = MISSING
    wandb_project: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    wandb_group: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    run_name: str = _constrained_field(NONEMPTY_STRING_CONSTRAINT)
    neptune: bool = MISSING
    neptune_name: str = MISSING
    neptune_project: str = MISSING
    tb: bool = MISSING
    local_rank: int = MISSING
    tag: str | None = MISSING
    eval_simulation: bool | None = MISSING
    package: PackageName = MISSING
    env_name: EnvironmentName = MISSING
    policy_name: PolicyName = MISSING
    rnn_name: RNNName | None = MISSING
    max_suggestion_cost: int = _constrained_field(POSITIVE_INT_CONSTRAINT)
    vec: VectorConfig = MISSING
    env: DriveEnvConfig = MISSING
    policy: DrivePolicyConfig = MISSING
    rnn: RecurrentConfig = MISSING
    train: TrainingConfig = MISSING
    sweep: dict = MISSING
    eval: EvaluationConfig | None = None
    controlled_exp: dict = MISSING

    # Known runtime metadata/options are optional and are not materialized in
    # the caller's plain dictionary when absent.
    no_model_upload: bool = False
    git: dict | None = None


MAX_C_SEED = 2**31 - 1


def normalize_puffer_drive_config(config, context="load"):
    """Validate config structure and types, then return a normalized plain dictionary."""
    if not isinstance(config, Mapping):
        _raise_config_error(context, "root", "must be a mapping")
    try:
        validated = OmegaConf.merge(OmegaConf.structured(PufferDriveConfig), dict(config))
        missing_keys = sorted(OmegaConf.missing_keys(validated))
        if missing_keys:
            _raise_config_error(context, missing_keys[0], "is required")
        container = OmegaConf.to_container(
            validated,
            resolve=True,
            enum_to_str=True,
            throw_on_missing=True,
        )
    except (OmegaConfBaseException, TypeError, ValueError) as exc:
        _raise_config_error(context, "root", str(exc))
    return container


def _validate_field_constraints(config, schema, context):
    """Apply dataclass field constraints recursively across a normalized config."""
    pending_schemas = [(config, schema, "")]
    for current_config, current_schema, schema_path in pending_schemas:
        if not current_config:
            continue
        for schema_field in fields(current_schema):
            field_name = schema_field.name
            field_path = f"{schema_path}.{field_name}" if schema_path else field_name
            field_value = current_config[field_name]
            constraint_mode = schema_field.metadata.get("constraint_mode")
            if constraint_mode is not None:
                _validate_value_constraint(field_value, constraint_mode, context, field_path)
            nested_schema = schema_field.type
            if not is_dataclass(nested_schema):
                nested_schema = next(
                    (candidate for candidate in get_args(nested_schema) if is_dataclass(candidate)),
                    None,
                )
            if nested_schema is not None and field_value is not None:
                pending_schemas.append((field_value, nested_schema, field_path))


def _validate_string_selection(value, context, path, *, allow_none=True):
    """Validate a string or non-empty list of strings used to select inputs."""
    if value is None and allow_none:
        return
    if isinstance(value, str):
        if not value.strip():
            _raise_config_error(context, path, "must not be empty")
        return
    valid_list = isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value)
    if not valid_list:
        _raise_config_error(context, path, "must be a non-empty string or list of non-empty strings")


def _validate_cross_field_constraints(config, context):
    """Validate relationships and context-dependent rules spanning config fields."""
    env = config["env"]
    if env["min_agents_per_env"] > env["max_agents_per_env"]:
        _raise_config_error(context, "env.min_agents_per_env", "must not exceed env.max_agents_per_env")
    if env["num_agents"] < env["min_agents_per_env"]:
        _raise_config_error(context, "env.num_agents", "must be at least env.min_agents_per_env")
    if env["init_step"] >= env["scenario_length"]:
        _raise_config_error(context, "env.init_step", "must be smaller than env.scenario_length")
    if env["spawn_initial_speed"] > env["base_max_speed_mps"]:
        _raise_config_error(context, "env.spawn_initial_speed", "must not exceed env.base_max_speed_mps")
    if env["min_goal_spacing"] > env["max_goal_spacing"]:
        _raise_config_error(context, "env.min_goal_spacing", "must not exceed env.max_goal_spacing")

    if env["reward_randomization"] and not env["reward_conditioning"]:
        _raise_config_error(context, "env.reward_randomization", "requires env.reward_conditioning")
    if env["init_step_spread"] and env["simulation_mode"] != "replay":
        _raise_config_error(context, "env.init_step_spread", "is only supported in replay mode")
    if env["init_step_spread"] and env["init_step_min_horizon"] >= env["scenario_length"]:
        _raise_config_error(context, "env.init_step_min_horizon", "must be smaller than env.scenario_length")
    if env["goal_source"] == "gt" and env["simulation_mode"] != "replay":
        _raise_config_error(context, "env.goal_source", "'gt' is only supported in replay mode")
    if env["terminate_on_goal"] and (env["simulation_mode"] != "replay" or env["control_mode"] != "control_sdc_only"):
        _raise_config_error(context, "env.terminate_on_goal", "requires replay mode with control_sdc_only")
    if env.get("eval_mode") is not None and (
        not isinstance(env["eval_mode"], (bool, int)) or env["eval_mode"] not in (0, 1)
    ):
        _raise_config_error(context, "env.eval_mode", "must be a boolean or 0/1")
    if env["eval_training_render"] and not env.get("eval_mode"):
        _raise_config_error(context, "env.eval_training_render", "requires env.eval_mode")
    if env["eval_training_render"] and env["simulation_mode"] != "gigaflow":
        _raise_config_error(context, "env.eval_training_render", "is only supported in gigaflow mode")
    single_agent_replay = env["simulation_mode"] == "replay" and env["control_mode"] == "control_sdc_only"
    if env.get("eval_mode") and not single_agent_replay and env["num_agents"] < env["max_agents_per_env"]:
        _raise_config_error(context, "env.num_agents", "must be at least env.max_agents_per_env during evaluation")

    from pufferlib.ocean.drive import binding

    if env["num_goals"] > binding.MAX_GOALS:
        _raise_config_error(context, "env.num_goals", f"must not exceed {binding.MAX_GOALS}")

    policy = config["policy"]
    if config["rnn_name"] is not None:
        if policy["backbone_num_layers"] == 0:
            _raise_config_error(
                context,
                "policy.backbone_num_layers",
                "must be positive when rnn_name is enabled",
            )
        backbone_size = policy["backbone_hidden_size"]
        if config["rnn"]["input_size"] != backbone_size or config["rnn"]["hidden_size"] != backbone_size:
            _raise_config_error(context, "rnn", "input_size and hidden_size must match policy.backbone_hidden_size")

    train = config["train"]
    _validate_string_selection(
        train["evaluation_benchmarks"],
        context,
        "train.evaluation_benchmarks",
        allow_none=train["evaluation_interval_epochs"] is None,
    )
    for field_name in ("batch_size", "bptt_horizon"):
        if train[field_name] != "auto":
            _validate_value_constraint(train[field_name], POSITIVE_INT_CONSTRAINT, context, f"train.{field_name}")
    if train["batch_size"] == "auto" and train["bptt_horizon"] == "auto":
        _raise_config_error(context, "train", "batch_size and bptt_horizon cannot both be 'auto'")
    _validate_string_selection(train["render_map"], context, "train.render_map")

    eval_config = config["eval"]
    evaluation_required = context.startswith("evaluation") or config["train"]["evaluation_interval_epochs"] is not None
    if not eval_config:
        if evaluation_required:
            _raise_config_error(context, "eval", "a complete eval section is required")
    else:
        _validate_string_selection(eval_config["benchmarks"], context, "eval.benchmarks")
        _validate_string_selection(eval_config["render_filter"], context, "eval.render_filter")
        if eval_config["failure_replay_csv"] is not None and eval_config["render_filter"] is None:
            _raise_config_error(context, "eval.failure_replay_csv", "requires eval.render_filter")

    vector_config = config["vec"]
    if vector_config["num_workers"] != "auto":
        _validate_value_constraint(vector_config["num_workers"], POSITIVE_INT_CONSTRAINT, context, "vec.num_workers")
    if vector_config["batch_size"] not in ("auto", None):
        _validate_value_constraint(vector_config["batch_size"], POSITIVE_INT_CONSTRAINT, context, "vec.batch_size")
    if vector_config["seed"] is not None and vector_config["seed"] > MAX_C_SEED:
        _raise_config_error(context, "vec.seed", f"must not exceed the C RNG maximum {MAX_C_SEED}")


def validate_puffer_drive_config(config, context) -> None:
    """Validate field and cross-field semantics on a normalized PufferDrive config."""
    _validate_field_constraints(config, PufferDriveConfig, context)
    _validate_cross_field_constraints(config, context)


def _validate_map_resources(env, context):
    """Validate an environment's map path and count, returning the available map count."""
    map_dir = env["map_dir"]
    if not isinstance(map_dir, str) or not map_dir:
        _raise_config_error(context, "env.map_dir", "must be a non-empty path")
    if os.path.isfile(map_dir) and map_dir.endswith(".bin"):
        available_maps = 1
    elif os.path.isdir(map_dir):
        available_maps = sum(name.endswith(".bin") for name in os.listdir(map_dir))
    else:
        message = f"path does not exist: {map_dir}"
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        manifest_path = os.path.join(repo_root, "data_utils", "datasets.yaml")
        dataset_name = os.path.basename(os.path.normpath(str(map_dir)))
        if os.path.isfile(manifest_path):
            with open(manifest_path) as manifest_file:
                is_registered_dataset = any(line.startswith(f"{dataset_name}:") for line in manifest_file)
            if is_registered_dataset:
                message += f"; fetch it with: python data_utils/fetch_data.py {dataset_name}"
        _raise_config_error(context, "env.map_dir", message)
    if available_maps == 0:
        _raise_config_error(context, "env.map_dir", "contains no .bin maps")
    if env["num_maps"] > available_maps:
        _raise_config_error(
            context, "env.num_maps", f"requests {env['num_maps']} maps but only {available_maps} are available"
        )
    return available_maps


def puffer_drive_constructor_keys():
    """Return keyword arguments accepted by the PufferDrive constructor."""
    from pufferlib.ocean.drive.drive import Drive

    return set(inspect.signature(Drive.__init__).parameters) - {"self"}


def normalize_puffer_drive_benchmarks(environment_config, benchmarks, context):
    """Validate benchmark definitions and return normalized copies for merging."""
    if not isinstance(environment_config, Mapping):
        _raise_config_error(context, "env", "must be a mapping")
    accepted_environment_fields = puffer_drive_constructor_keys()
    unknown_environment_fields = set(environment_config) - accepted_environment_fields
    if unknown_environment_fields:
        _raise_config_error(
            context,
            "env",
            f"has unsupported fields: {', '.join(sorted(unknown_environment_fields))}",
        )

    normalized_benchmarks = []
    for benchmark_idx, benchmark in enumerate(benchmarks):
        benchmark_path = f"benchmarks[{benchmark_idx}]"
        if not isinstance(benchmark, Mapping):
            _raise_config_error(context, benchmark_path, "must be a mapping")
        benchmark_name = benchmark.get("name")
        if not isinstance(benchmark_name, str) or not benchmark_name.strip():
            _raise_config_error(context, f"{benchmark_path}.name", "must be a non-empty string")
        benchmark_name = benchmark_name.strip()
        benchmark_path = f"benchmark.{benchmark_name}"

        unknown_benchmark_fields = set(benchmark) - {"name", "seed", "num_scenarios", "env"}
        if unknown_benchmark_fields:
            _raise_config_error(
                context,
                benchmark_path,
                f"has unsupported fields: {', '.join(sorted(unknown_benchmark_fields))}",
            )

        benchmark_environment = benchmark.get("env")
        if not isinstance(benchmark_environment, Mapping):
            _raise_config_error(context, f"{benchmark_path}.env", "must be a mapping")
        unknown_environment_fields = set(benchmark_environment) - accepted_environment_fields
        if unknown_environment_fields:
            _raise_config_error(
                context,
                f"{benchmark_path}.env",
                f"has unsupported fields: {', '.join(sorted(unknown_environment_fields))}",
            )

        simulation_mode = benchmark_environment.get("simulation_mode")
        if simulation_mode not in ("gigaflow", "replay"):
            _raise_config_error(context, f"{benchmark_path}.env.simulation_mode", "must be 'gigaflow' or 'replay'")
        control_mode = benchmark_environment.get("control_mode")
        if not isinstance(control_mode, str) or not control_mode:
            _raise_config_error(context, f"{benchmark_path}.env.control_mode", "must be a non-empty string")

        seed = benchmark.get("seed")
        if seed is None:
            _raise_config_error(context, f"{benchmark_path}.seed", "must be a non-negative integer")
        _validate_value_constraint(seed, NONNEGATIVE_INT_CONSTRAINT, context, f"{benchmark_path}.seed")
        if seed > MAX_C_SEED:
            _raise_config_error(context, f"{benchmark_path}.seed", f"must not exceed the C RNG maximum {MAX_C_SEED}")
        num_scenarios = benchmark.get("num_scenarios")
        if num_scenarios is None:
            _raise_config_error(context, f"{benchmark_path}.num_scenarios", "must be a positive integer")
        _validate_value_constraint(
            num_scenarios,
            POSITIVE_INT_CONSTRAINT,
            context,
            f"{benchmark_path}.num_scenarios",
        )
        num_maps = benchmark_environment.get("num_maps")
        if num_maps is None:
            _raise_config_error(context, f"{benchmark_path}.env.num_maps", "must be a positive integer")
        _validate_value_constraint(num_maps, POSITIVE_INT_CONSTRAINT, context, f"{benchmark_path}.env.num_maps")

        max_agents_per_env = benchmark_environment.get("max_agents_per_env")
        single_agent_replay = simulation_mode == "replay" and control_mode == "control_sdc_only"
        if max_agents_per_env is None and not single_agent_replay:
            _raise_config_error(context, f"{benchmark_path}.env.max_agents_per_env", "must be a positive integer")
        if max_agents_per_env is not None:
            _validate_value_constraint(
                max_agents_per_env,
                POSITIVE_INT_CONSTRAINT,
                context,
                f"{benchmark_path}.env.max_agents_per_env",
            )
        max_scenarios_per_batch = benchmark_environment.get("max_scenarios_per_batch")
        if max_scenarios_per_batch is not None:
            _validate_value_constraint(
                max_scenarios_per_batch,
                POSITIVE_INT_CONSTRAINT,
                context,
                f"{benchmark_path}.env.max_scenarios_per_batch",
            )

        normalized_environment = copy.deepcopy(dict(benchmark_environment))
        map_dir = normalized_environment.get("map_dir")
        if not isinstance(map_dir, str) or not map_dir:
            _raise_config_error(context, f"{benchmark_path}.env.map_dir", "must be a non-empty path")
        normalized_environment["map_dir"] = os.path.abspath(map_dir)
        _validate_map_resources(normalized_environment, f"{context}.{benchmark_path}")
        if max_agents_per_env is None:
            normalized_environment.pop("max_agents_per_env", None)

        normalized_benchmarks.append(
            {
                "name": benchmark_name,
                "seed": seed,
                "num_scenarios": num_scenarios,
                "env": normalized_environment,
            }
        )
    return normalized_benchmarks


def validate_puffer_drive_resources(config, context):
    """Check final filesystem inputs separately from config semantics."""
    _validate_map_resources(config["env"], context)

    load_model_path = config.get("load_model_path")
    if load_model_path not in (None, "latest") and not os.path.isfile(load_model_path):
        _raise_config_error(context, "load_model_path", f"checkpoint does not exist: {load_model_path}")

    eval_config = config.get("eval")
    if eval_config and eval_config.get("failure_replay_csv"):
        failure_csv = eval_config["failure_replay_csv"]
        if not os.path.isfile(failure_csv):
            _raise_config_error(context, "eval.failure_replay_csv", f"file does not exist: {failure_csv}")
