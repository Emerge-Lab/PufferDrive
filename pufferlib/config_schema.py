"""Structured-config schemas for load_config

This files defines input admissible values for the config, and allows to raise
errors whenever an input value does not respect the schema. Every field is
MISSING by design: missing values in the YAML raise errors.

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

from dataclasses import dataclass
from typing import Optional
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
    create_controllable_types = 2


class GoalRegen(Enum):
    finite = 0
    rolling = 1


class GoalSource(Enum):
    route = 0
    map = 1
    gt = 2


@dataclass
class DriveEnvConfig:
    simulation_mode: SimulationMode = MISSING
    num_agents: int = MISSING
    min_agents_per_env: int = MISSING
    max_agents_per_env: int = MISSING
    action_type: ActionType = MISSING
    dynamics_model: DynamicsModel = MISSING
    reset_accel_on_stop: bool = MISSING
    dt: float = MISSING
    base_max_speed_mps: float = MISSING
    max_speed_mps: Optional[float] = None
    spawn_initial_speed: float = MISSING
    spawn_lateral_offset_max_frac: float = MISSING
    spawn_heading_max_deg: float = MISSING
    collision_behavior: InfractionBehavior = MISSING
    offroad_behavior: InfractionBehavior = MISSING
    traffic_light_behavior: InfractionBehavior = MISSING
    disable_red_light_infractions: int = MISSING
    traffic_light_junction_phases: int = MISSING
    use_map_cache: int = MISSING
    use_neighbor_cache: int = MISSING
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
    replay_expert_agents: int = MISSING
    init_mode: InitMode = MISSING
    compute_eval_metrics: bool = MISSING
    eval_training_render: bool = MISSING
    goal_regen_mode: GoalRegen = MISSING
    goal_source: GoalSource = MISSING
    obs_goal_lane_distance: bool = MISSING
    goal_radius: float = MISSING
    goal_speed: float = MISSING
    goal_speed_randomization: bool = MISSING
    goal_reach_requires_speed: bool = MISSING
    num_goals: int = MISSING
    min_goal_spacing: float = MISSING
    max_goal_spacing: float = MISSING
    goal_heading_max_deg: float = MISSING
    reward_conditioning: bool = MISSING
    reward_randomization: bool = MISSING
    reward_log_sampling: bool = MISSING
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
    allow_map_subset: bool = MISSING
    obs_slots_lane_n: int = MISSING
    obs_slots_boundary_n: int = MISSING
    obs_slots_partners_n: int = MISSING
    obs_partner_relative_velocity: bool = MISSING
    obs_slots_traffic_controls_n: int = MISSING
    obs_dropout_lane: float = MISSING
    obs_dropout_boundary: float = MISSING
    obs_lane_stride: int = MISSING
    obs_boundary_stride: int = MISSING
    obs_norm_speed_mps: float = MISSING
    obs_norm_goal_offset_m: float = MISSING
    obs_norm_xy_offset_m: float = MISSING
    obs_norm_veh_length_m: float = MISSING
    obs_norm_veh_width_m: float = MISSING
    obs_norm_road_seg_length_m: float = MISSING
    obs_norm_road_seg_width_m: float = MISSING
    obs_norm_z_m: float = MISSING
    eval_perceived_size_margin_m: float = MISSING
    eval_standstill_jerk_deadband_mps3: float = MISSING
    obs_range_road_front_m: float = MISSING
    obs_range_road_behind_m: float = MISSING
    obs_range_road_side_m: float = MISSING
    obs_range_partner_m: float = MISSING
    obs_range_traffic_control_m: float = MISSING
    partner_blindness_prob: float = MISSING
    partner_blindness_trigger_prob: float = MISSING
    partner_blindness_duration_seconds: float = MISSING
    phantom_braking_prob: float = MISSING
    phantom_braking_trigger_prob: float = MISSING
    phantom_braking_duration_seconds: float = MISSING
    phantom_braking_freeze_steering: bool = MISSING


# env_name -> structured schema for the `env` section. Envs without an entry
# load unvalidated (phase-1 behavior).
ENV_SCHEMAS = {
    "puffer_drive": DriveEnvConfig,
}
