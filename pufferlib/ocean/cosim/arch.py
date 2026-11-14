"""Shadow-env Drive kwargs from a checkpoint config (CARLA + nuPlan co-sim).

The shadow env adopts EVERY Drive-accepted key from the checkpoint's
config.yaml env section -- full parity with the training env, no allowlist to
fall out of date -- then applies CLEAN_EVAL_OVERRIDES (the repo's clean-eval
profile), then the caller's structural co-sim keys (map_dir, num_agents,
scenario_length, ...) win.
"""

import inspect
from pathlib import Path

import yaml

from pufferlib.ocean.drive.drive import Drive

# Mirror of the noise/light/conditioning keys in
# pufferlib/config/evaluation/benchmark.yaml, duplicated so the co-sim venvs
# (CARLA cp310 / nuPlan) never import the training stack;
# tests/unit_tests/test_cosim_config_contract.py pins the two equal.
# The reward_*/goal_* keys pin the conditioning obs to the GigaFlow paper's
# eval values (Table A2) instead of adopting the training config's
# reward_randomization=true, which would sample a random command profile.
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


# Some saved checkpoint configs store the infraction behaviors as resolved
# binding enums (0/1) instead of the strings Drive.__init__ accepts
# (collision_behavior: 1); normalize either spelling.
_INFRACTION_BEHAVIOR_NAMES = {0: "ignore", 1: "stop"}
_INFRACTION_BEHAVIOR_KEYS = ("collision_behavior", "offroad_behavior", "traffic_light_behavior")


def checkpoint_config_path(checkpoint):
    """config.yaml saved with a checkpoint: next to a run-dir `final_model.pt`, or one level up for
    `models/model_*.pt`. Raises when neither exists (a shadow env built from Drive defaults would be
    silently wrong)."""
    checkpoint = Path(checkpoint).resolve()
    for candidate in (checkpoint.parent / "config.yaml", checkpoint.parents[1] / "config.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no config.yaml next to or one level above {checkpoint}")


# Settings of removed features: a config saved before the removal may carry them, but only with this value.
RETIRED_SETTINGS = {
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
# Removed settings that only shaped training (blind and braking agents, reward sampling, training margins):
# evals switch those off, so a saved config may carry any value.
RETIRED_TRAINING_SETTINGS = (
    ("env", "reward_log_sampling"),
    ("env", "goal_speed_randomization"),
    ("env", "partner_blindness_corridor"),
    ("env", "partner_blindness_time_fraction_max"),
    ("env", "phantom_braking_freeze_steering"),
    ("env", "red_light_train_rear_margin_m"),
)
RETIRED_SECTIONS = ("rnn",)


def drop_retired_settings(config):
    """Strip the settings of removed features from a saved run config, in place. Raises when one of them
    selects a feature that no longer exists (the checkpoint cannot be run by this code)."""
    for key_path, remaining_value in RETIRED_SETTINGS.items():
        section = config
        for section_name in key_path[:-1]:
            section = section.get(section_name) or {}
        setting_name = key_path[-1]
        if setting_name not in section:
            continue
        if section[setting_name] != remaining_value:
            raise ValueError(
                f"saved config sets {'.'.join(key_path)}={section[setting_name]!r}; that feature was removed, "
                f"only {remaining_value!r} is supported"
            )
        del section[setting_name]
    for key_path in RETIRED_TRAINING_SETTINGS:
        section = config
        for section_name in key_path[:-1]:
            section = section.get(section_name) or {}
        section.pop(key_path[-1], None)
    for section_name in RETIRED_SECTIONS:
        config.pop(section_name, None)
    return config


def load_checkpoint_config(config_path):
    """A checkpoint's saved run config without the settings of removed features."""
    with open(config_path) as config_file:
        return drop_retired_settings(yaml.safe_load(config_file))


# Applied to every co-sim shadow env after CLEAN_EVAL_OVERRIDES: the eval-only semantics the native
# benchmark gets from eval_mode (scripts/kesai/3_train_64GPU_multinode_slurm.sh passes the 0.2 m margin and the
# 1.5 m/s^3 standstill deadband), which the co-sim env cannot enter because eval_mode's scenario
# batching needs a full agent pool.
COSIM_EVAL_OVERRIDES = {
    "cosim_eval_semantics": True,
    "eval_perceived_size_margin_m": 0.2,
    "eval_standstill_jerk_deadband_mps3": 1.5,
}


def shadow_env_kwargs(cfg, overrides=None):
    """Drive kwargs for a co-sim shadow env.

    Precedence: checkpoint env config (every Drive-accepted key) < CLEAN_EVAL_OVERRIDES
    < COSIM_EVAL_OVERRIDES < `overrides` (the co-sim's structural keys)."""
    accepted = set(inspect.signature(Drive.__init__).parameters)
    adopted = {k: v for k, v in ((cfg or {}).get("env") or {}).items() if k in accepted}
    for key in _INFRACTION_BEHAVIOR_KEYS:
        if key in adopted and isinstance(adopted[key], int):
            adopted[key] = _INFRACTION_BEHAVIOR_NAMES[adopted[key]]
    return {**adopted, **CLEAN_EVAL_OVERRIDES, **COSIM_EVAL_OVERRIDES, **(overrides or {})}
