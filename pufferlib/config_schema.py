"""Structured-config schemas for load_config — Hydra migration phase 2.

Each entry in ENV_SCHEMAS is an OmegaConf structured schema merged over the
composed monolithic YAML in `pufferlib.pufferl.load_config`, so type errors,
unknown keys, and invalid enum names fail at config load instead of deep in
env construction.

Every field is MISSING on purpose: the YAML is the single source of values,
the schema only contributes types. A key present here but absent from the
YAML surfaces as a missing-value error at load time.

Enum int values mirror the #defines in pufferlib/ocean/drive/drive.h (the
source of truth, exported to Python via binding). Only the *names* gate
config validation — drive.py still maps names to binding constants itself.

Naming convention (Google enum style): every C #define is
`<ENUM_CLASS_NAME_IN_SCREAMING_SNAKE>_<MEMBER_NAME_UPPER>`, e.g.
`InfractionBehavior.stop` <-> `INFRACTION_BEHAVIOR_STOP`. This makes the C
name mechanically derivable from the Python one (and vice versa) instead of
needing a lookup table. NonVehicleController is the one exception: it
reuses Controller's C constants (`CONTROLLER_*`) since it's the same
underlying controller, plus a Python-only `auto` sentinel that drive.py
resolves before it ever reaches C. tests/unit_tests/test_config_schema.py
walks every class here and asserts the derived name exists in binding with
the matching value, so a new enum is checked automatically without a new
hand-written assert.
"""

from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING


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


class TargetType(Enum):
    static = 0
    dynamic = 1


@dataclass
class DriveEnvConfig:
    simulation_mode: SimulationMode = MISSING
    num_agents: int = MISSING
    min_agents_per_env: int = MISSING
    max_agents_per_env: int = MISSING
    action_type: ActionType = MISSING
    dynamics_model: DynamicsModel = MISSING
    dt: float = MISSING
    spawn_initial_speed: float = MISSING
    collision_behavior: InfractionBehavior = MISSING
    offroad_behavior: InfractionBehavior = MISSING
    traffic_light_behavior: InfractionBehavior = MISSING
    use_map_cache: int = MISSING
    scenario_length: int = MISSING
    resample_frequency: int = MISSING
    termination_mode: int = MISSING
    inactive_agent_threshold: float = MISSING
    terminate_on_goal: int = MISSING
    init_step: int = MISSING
    init_step_spread: bool = MISSING
    init_step_min_horizon: int = MISSING
    control_mode: ControlMode = MISSING
    sdc_controller: Controller = MISSING
    non_sdc_controller: Controller = MISSING
    non_vehicle_controller: NonVehicleController = MISSING
    init_mode: InitMode = MISSING
    compute_eval_metrics: bool = MISSING
    target_type: TargetType = MISSING
    goal_on_lane: bool = MISSING
    goal_radius: float = MISSING
    goal_speed: float = MISSING
    num_goals: int = MISSING
    min_goal_spacing: float = MISSING
    max_goal_spacing: float = MISSING
    reward_conditioning: bool = MISSING
    reward_randomization: bool = MISSING
    reward_goal: float = MISSING
    reward_collision: float = MISSING
    reward_offroad: float = MISSING
    reward_stop_line: float = MISSING
    reward_comfort: float = MISSING
    reward_lane_align: float = MISSING
    reward_vel_align: float = MISSING
    reward_lane_center: float = MISSING
    reward_center_bias: float = MISSING
    reward_velocity: float = MISSING
    reward_reverse: float = MISSING
    reward_timestep: float = MISSING
    reward_overspeed: float = MISSING
    reward_ade: float = MISSING
    map_dir: str = MISSING
    num_maps: int = MISSING
    obs_slots_lane_n: int = MISSING
    obs_slots_boundary_n: int = MISSING
    obs_slots_partners_n: int = MISSING
    obs_slots_traffic_controls_n: int = MISSING
    obs_dropout_lane: float = MISSING
    obs_dropout_boundary: float = MISSING
    obs_lane_stride: int = MISSING
    obs_boundary_stride: int = MISSING
    obs_norm_goal_offset_m: float = MISSING
    obs_norm_xy_offset_m: float = MISSING
    obs_norm_veh_length_m: float = MISSING
    obs_norm_veh_width_m: float = MISSING
    obs_norm_road_seg_length_m: float = MISSING
    obs_norm_road_seg_width_m: float = MISSING
    obs_range_road_front_m: float = MISSING
    obs_range_road_behind_m: float = MISSING
    obs_range_road_side_m: float = MISSING
    obs_range_partner_m: float = MISSING
    obs_range_traffic_control_m: float = MISSING
    partner_blindness_prob: float = MISSING
    partner_blindness_trigger_prob: float = MISSING
    phantom_braking_prob: float = MISSING
    phantom_braking_trigger_prob: float = MISSING
    phantom_braking_duration: int = MISSING


# env_name -> structured schema for the `env` section. Envs without an entry
# load unvalidated (phase-1 behavior).
ENV_SCHEMAS = {
    "puffer_drive": DriveEnvConfig,
}
