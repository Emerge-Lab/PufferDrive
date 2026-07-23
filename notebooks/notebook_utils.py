import os
import sys
from pathlib import Path

import numpy as np
import yaml

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding
from pufferlib.ocean.torch import Drive as DrivePolicy
from pufferlib.pufferl import load_config

# pufferl escalates RuntimeWarning->error process-wide (NaN/Inf guard for training).
# Notebooks do exploratory numpy/plotting that legitimately warns (empty-slice means,
# etc.); undo the escalation so benign warnings don't abort a notebook.
import warnings

warnings.filterwarnings("default", category=RuntimeWarning)


ROOT = Path(__file__).resolve().parents[1]
MAP_DIR = str(ROOT / "pufferlib/resources/drive/binaries/carla")

COEF_NAMES = [
    "goal_radius",
    "goal_speed",
    "collision",
    "offroad",
    "comfort",
    "lane_align",
    "vel_align",
    "lane_center",
    "center_bias",
    "velocity",
    "reverse",
    "stop_line",
    "timestep",
    "overspeed",
    "throttle",
    "steer",
    "acc",
]

EGO_LABELS = [
    "speed",
    "width",
    "length",
    "steering",
    "accel_long",
    "accel_lat",
    "lane_center",
    "lane_align",
    "speed_limit",
    "stopped",
]

DEFAULT_ENV_KWARGS = {
    "num_agents": 64,
    "num_maps": 1,
    "min_agents_per_env": 64,
    "max_agents_per_env": 64,
    "simulation_mode": "gigaflow",
    "dynamics_model": "jerk",
    "action_type": "discrete",
    "dt": 0.1,
    "scenario_length": 512,
    "resample_frequency": 0,
    "reward_conditioning": True,
    "reward_randomization": False,
    "goal_regen_mode": "finite",
    "map_dir": MAP_DIR,
    "collision_behavior": 1,
    "offroad_behavior": 1,
    "obs_slots_lane_n": 80,
    "obs_slots_boundary_n": 80,
    "obs_lane_stride": 1,
    "obs_boundary_stride": 1,
    "obs_slots_partners_n": 16,
    "obs_slots_traffic_controls_n": 4,
    "obs_dropout_lane": 0.0,
    "obs_dropout_boundary": 0.0,
    "obs_norm_goal_offset_m": 120.0,
    "obs_norm_xy_offset_m": 120.0,
    "obs_norm_veh_length_m": 15.0,
    "obs_norm_veh_width_m": 10.0,
    "obs_norm_road_seg_length_m": 10.0,
    "obs_norm_road_seg_width_m": 5.0,
    "obs_range_road_front_m": 120.0,
    "obs_range_road_behind_m": 20.0,
    "obs_range_road_side_m": 30.0,
    "obs_range_partner_m": 100.0,
    "obs_range_traffic_control_m": 100.0,
    "seed": 42,
}

DEFAULT_POLICY_KWARGS = {
    "ego_input_size": 64,
    "partner_input_size": 64,
    "lane_input_size": 64,
    "boundary_input_size": 64,
    "traffic_control_input_size": 64,
    "context_input_size": 64,
    "backbone_hidden_size": 128,
    "backbone_num_layers": 1,
    "actor_hidden_size": 128,
    "actor_num_layers": 0,
    "critic_hidden_size": 128,
    "critic_num_layers": 0,
    "encoder_activation": "tanh",
    "encoder_layer_norm": True,
    "backbone_activation": "gelu",
    "backbone_layer_norm": False,
    "shared_network": True,
    "mask_padded_features": False,
    "action_type": "discrete",
}


def drive_kwargs(**overrides):
    return {**DEFAULT_ENV_KWARGS, **overrides}


def make_drive_env(**overrides):
    kwargs = drive_kwargs(**overrides)
    env = Drive(**kwargs)
    obs, info = env.reset(seed=kwargs["seed"])
    return env, obs, info


def action_shape(env):
    if hasattr(env.single_action_space, "nvec"):
        return (env.num_agents, len(env.single_action_space.nvec))
    return (env.num_agents, env.single_action_space.shape[0])


def zero_actions(env):
    dtype = np.int64 if hasattr(env.single_action_space, "nvec") else np.float32
    return np.zeros(action_shape(env), dtype=dtype)


def random_actions(env):
    if hasattr(env.single_action_space, "nvec"):
        return np.stack([np.random.randint(0, n, size=env.num_agents) for n in env.single_action_space.nvec], axis=1)
    return np.random.uniform(-1.0, 1.0, size=action_shape(env)).astype(np.float32)


def make_drive_policy(env, device, **overrides):
    return DrivePolicy(env, **{**DEFAULT_POLICY_KWARGS, **overrides}).to(device)


def load_notebook_config(checkpoint_path=None, env_name="puffer_drive"):
    argv = sys.argv
    sys.argv = [argv[0]]
    config = load_config(env_name)
    sys.argv = argv

    if checkpoint_path:
        cfg_yaml = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "config.yaml")
        with open(cfg_yaml) as f:
            ycfg = yaml.safe_load(f)
        for section in ["env", "train", "policy", "rnn"]:
            if section in ycfg and isinstance(ycfg[section], dict):
                config[section].update(ycfg[section])

    return config
