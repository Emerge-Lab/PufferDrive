"""Shadow-env Drive kwargs and conventions shared by the CARLA and nuPlan co-sims."""

import inspect
from pathlib import Path

import yaml

from opengigaflow.ocean.drive.drive import Drive


FAR_AWAY = 1.0e6  # park unused shadow-env agent slots out of observation range
SCENARIO_LENGTH_MARGIN_STEPS = 2  # the shadow env must never hit its own truncation while the simulator runs on
# write_ego_obs column order: speed, width, length, steering, accel_long, ...
EGO_OBS_SPEED_IDX = 0
EGO_OBS_STEERING_IDX = 3
EGO_OBS_ACCEL_LONG_IDX = 4

# the benchmark's clean-eval profile, duplicated so the co-sim venvs never import the training stack
CLEAN_EVAL_OVERRIDES = {
    "obs_dropout_lane": 0.0,
    "obs_dropout_boundary": 0.0,
    "partner_blindness_prob": 0.0,
    "partner_blindness_prob_min": 0.0,
    "partner_blindness_trigger_prob": 0.0,
    "phantom_braking_prob": 0.0,
    "phantom_braking_trigger_prob": 0.0,
    "traffic_light_behavior": "stop",
    "reward_randomization": False,
    "goal_speed": 3.0,
    "goal_radius": 10.0,
    "reward_collision": 3.0,
    "reward_offroad": 3.0,
    "reward_stop_line": 1.0,
    "reward_goal": 1.0,
    "reward_overspeed": 0.1,  # training-range max (REWARD_BOUNDS)
    "reward_comfort": 0.05,
    "reward_velocity": 0.0025,
    "reward_lane_align": 0.025,
    "reward_lane_center": 0.0075,  # training-range max (REWARD_BOUNDS), not the paper's 0.0038
    "reward_timestep": 0.000025,
    "reward_reverse": 0.005,
    "spawn_lateral_offset_max_frac": 0.0,
    "spawn_heading_max_deg": 0.0,
    "pose_noise_xy_m": 0.0,
    "pose_noise_yaw_deg": 0.0,
    "speed_limit_random_prob": 0.0,
    "stagger_first_episode": False,
    "scripted_pedestrian_ratio": 0.0,
}


def checkpoint_config_path(checkpoint):
    """The config.yaml saved with a checkpoint: next to it, or one level up for models/model_*.pt."""
    checkpoint = Path(checkpoint).resolve()
    for candidate in (checkpoint.parent / "config.yaml", checkpoint.parents[1] / "config.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no config.yaml next to or one level above {checkpoint}")


# Settings of removed features: a config saved before the removal may carry them, but only with this value.
RETIRED_SETTINGS = {
    ("package",): "ocean",
    ("rnn_name",): None,
    ("env", "action_type"): "continuous",
    ("env", "dynamics_model"): "jerk",
    ("env", "reset_accel_on_stop"): True,
    ("env", "yield_signs_enabled"): False,
    ("env", "terminate_on_goal"): False,
    ("env", "init_step_spread"): False,
    ("env", "reward_ade"): 0.0,
    ("env", "overspeed_tolerance_mps"): 0.0,
    ("env", "spawn_initial_speed"): 0.0,
    ("env", "obs_partner_relative_velocity"): True,
    ("env", "obs_lane_heading_signed"): True,
    ("env", "obs_lane_speed_limit"): True,
    ("env", "obs_goal_lane_distance"): True,
    ("policy", "shared_network"): False,
    ("policy", "mask_padded_features"): True,
}
# Removed settings that only shaped training: evals switch those off, so a saved config may carry any value.
RETIRED_TRAINING_SETTINGS = (
    ("env", "reward_log_sampling"),
    ("env", "goal_speed_randomization"),
    ("env", "partner_blindness_corridor"),
    ("env", "partner_blindness_time_fraction_max"),
    ("env", "phantom_braking_freeze_steering"),
    ("env", "red_light_train_rear_margin_m"),
    ("env", "traffic_light_junction_phases"),
)
RETIRED_SECTIONS = ("rnn",)


def drop_retired_settings(config):
    """Strip the settings of removed features from a saved run config, in place; raises when one selects a removed
    feature (the checkpoint cannot be run by this code)."""
    for key_path in (*RETIRED_SETTINGS, *RETIRED_TRAINING_SETTINGS):
        section = config
        for section_name in key_path[:-1]:
            section = section.get(section_name) or {}
        setting_name = key_path[-1]
        if setting_name not in section:
            continue
        if key_path in RETIRED_SETTINGS and section[setting_name] != RETIRED_SETTINGS[key_path]:
            raise ValueError(
                f"saved config sets {'.'.join(key_path)}={section[setting_name]!r}; that feature was removed, "
                f"only {RETIRED_SETTINGS[key_path]!r} is supported"
            )
        del section[setting_name]
    for section_name in RETIRED_SECTIONS:
        config.pop(section_name, None)
    return config


def load_checkpoint_config(config_path):
    """A checkpoint's saved run config without the settings of removed features."""
    with open(config_path) as config_file:
        return drop_retired_settings(yaml.safe_load(config_file))


# eval_mode's semantics, which the co-sim cannot enter: its scenario batching needs a full agent pool
COSIM_EVAL_OVERRIDES = {
    "cosim_eval_semantics": True,
    "eval_perceived_size_margin_m": 0.2,
    "eval_standstill_jerk_deadband_mps3": 1.5,
}


def shadow_env_kwargs(cfg, overrides=None):
    """Co-sim Drive kwargs; precedence: the checkpoint's Drive-accepted env keys < CLEAN_EVAL_OVERRIDES <
    COSIM_EVAL_OVERRIDES < overrides (the co-sim's structural keys)."""
    accepted = set(inspect.signature(Drive.__init__).parameters)
    adopted = {k: v for k, v in ((cfg or {}).get("env") or {}).items() if k in accepted}
    return {**adopted, **CLEAN_EVAL_OVERRIDES, **COSIM_EVAL_OVERRIDES, **(overrides or {})}
