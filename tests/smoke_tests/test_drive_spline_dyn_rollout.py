#!/usr/bin/env python3
"""CPU smoke test for dynamics_model: spline under random [-1, 1]^6 actions.

Spline dynamics enforces no physical limits, so this only proves the path is safe to train
against: finite observations and rewards, an unchanged observation layout, and a policy that
builds and runs. It also pins the config rule that spline dynamics needs the spline action type.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.config_schema import normalize_puffer_drive_config
from pufferlib.pufferl import load_config, load_env

SEED = 7
STEPS = 200


def _build_spline_dyn_config():
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv

    args["trajectory_training"] = True
    args["vec"].update({"backend": "Serial", "num_envs": 4, "seed": SEED})
    args["env"].update(
        {
            "dynamics_model": "spline",
            "num_agents": 8,
            "min_agents_per_env": 8,
            "max_agents_per_env": 8,
            "num_maps": 2,
            "use_map_cache": True,
            "map_dir": "pufferlib/resources/drive/binaries/carla",
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None

    normalized = normalize_puffer_drive_config(args, "test")
    assert normalized["env"]["action_type"] == "spline"
    assert normalized["env"]["dynamics_model"] == "spline"
    return normalized


def _random_actions(vecenv, rng):
    shape = (vecenv.num_agents,) + vecenv.single_action_space.shape
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def test_drive_spline_dyn_rollout():
    args = _build_spline_dyn_config()
    rng = np.random.default_rng(SEED)

    vecenv = load_env("puffer_drive", args)
    try:
        obs, _ = vecenv.reset(seed=SEED)
        assert obs.shape[1] == vecenv.driver_env.num_obs
        for step in range(STEPS):
            obs, rewards, _, _, _ = vecenv.step(_random_actions(vecenv, rng))
            assert np.isfinite(obs).all(), f"non-finite observation at step {step}"
            assert np.isfinite(rewards).all(), f"non-finite reward at step {step}"
    finally:
        vecenv.close()


def test_drive_spline_dyn_policy_runs():
    from pufferlib.ocean.drive import binding
    from pufferlib.pufferl import load_policy

    args = _build_spline_dyn_config()
    args["device"] = "cpu"
    args["train"]["device"] = "cpu"

    vecenv = load_env("puffer_drive", args)
    try:
        obs, _ = vecenv.reset(seed=SEED)
        policy = load_policy(args, vecenv, "puffer_drive")
        assert policy.ego_dim == binding.EGO_FEATURES + binding.SPLINE_INTENT_FEATURES
        with torch.no_grad():
            policy(torch.as_tensor(obs, dtype=torch.float32))
    finally:
        vecenv.close()


def test_spline_dynamics_rejects_non_spline_action_type():
    from pufferlib.ocean.drive.drive import Drive

    try:
        Drive(dynamics_model="spline", action_type="continuous")
    except ValueError as error:
        assert "requires action_type 'spline'" in str(error)
    else:
        raise AssertionError("spline dynamics accepted a continuous action type")


if __name__ == "__main__":
    test_drive_spline_dyn_rollout()
    test_drive_spline_dyn_policy_runs()
    test_spline_dynamics_rejects_non_spline_action_type()
    print("Spline dynamics smoke tests passed!")
