import os
import sys
from pathlib import Path

import numpy as np
import yaml

import torch

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding
from pufferlib.ocean.torch import Drive as DrivePolicy
from pufferlib.pufferl import load_config


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
    "a_long",
    "a_lat",
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
    "target_type": "static",
    "map_dir": MAP_DIR,
    "collision_behavior": 1,
    "offroad_behavior": 1,
    "obs_slots_lane": 32,
    "obs_slots_boundary": 32,
    "obs_slots_partners": 16,
    "obs_slots_traffic_controls": 10,
    "seed": 42,
}

DEFAULT_POLICY_KWARGS = {
    "input_size": 64,
    "backbone_hidden_size": 128,
    "backbone_num_layers": 1,
    "actor_hidden_size": 128,
    "actor_num_layers": 0,
    "critic_hidden_size": 128,
    "critic_num_layers": 0,
    "encoder_gigaflow": True,
    "dropout": 0.0,
    "split_network": False,
}


def drive_kwargs(**overrides):
    return {**DEFAULT_ENV_KWARGS, **overrides}


def make_drive_env(**overrides):
    kwargs = drive_kwargs(**overrides)
    env = Drive(**kwargs)
    obs, info = env.reset(seed=kwargs["seed"])
    return env, obs, info


def notebook_dims(env):
    return {
        "EGO_DIM": env.ego_features,
        "NUM_COEFS": binding.NUM_REWARD_COEFS,
        "PARTNER_F": env.partner_features,
        "ROAD_F": env.road_features,
        "TRAFFIC_CONTROL_F": env.traffic_control_features,
        "NUM_TRAFFIC_CONTROL_TYPES": binding.NUM_TRAFFIC_CONTROL_TYPES,
        "MAX_PARTNERS": env.obs_slots_partners,
        "MAX_LANES": env.obs_slots_lane_kept,
        "MAX_BOUNDS": env.obs_slots_boundary_kept,
        "MAX_TRAFFIC": env.obs_slots_traffic_controls,
        "MAX_TARGET": env.num_target_waypoints,
        "TARGET_F": env.target_features,
        "TARGET_DIM": env.target_dim,
        "N_ACTIONS": int(env.single_action_space.nvec[0]) if hasattr(env.single_action_space, "nvec") else 1,
        "N": env.num_agents,
        "ACT_SHAPE": action_shape(env),
    }


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

    config["train"]["use_rnn"] = config.get("rnn_name") is not None
    return config


def make_rnn_state(policy, n, device):
    return {
        "lstm_h": torch.zeros(n, policy.hidden_size, device=device),
        "lstm_c": torch.zeros(n, policy.hidden_size, device=device),
    }
