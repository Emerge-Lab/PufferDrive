"""Tests for the SHIFT experiment matrix additions.

Covered here:
  - `train.kl_ref_direction=reference_to_policy`: the HR-PPO direction D_KL(pi_ref || pi).
  - `env.sdc_controller=expert_tracking`: the SDC follows the log through the sim
    dynamics and writes the chosen discrete action back into the actions buffer.
  - `puffer bc`: fits the actor on expert_tracking labels and saves a checkpoint a
    fresh policy can load.
  - Config schema guards for the new knobs.
"""

import copy
import os
import sys

import numpy as np
import pytest
import torch

import pufferlib
import pufferlib.pytorch as P
from pufferlib.bc import bc, collect_expert_dataset
from pufferlib.config_schema import normalize_puffer_drive_config, validate_puffer_drive_config
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import load_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPLAY_MAP_DIR = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "nuplan")
SCENARIO_LENGTH = 200
TRACKING_ENV = dict(
    simulation_mode="replay",
    control_mode="control_sdc_only",
    sdc_controller="expert_tracking",
    non_sdc_controller="replay",
    non_vehicle_controller="replay",
    goal_source="gt",
    action_type="discrete",
    dynamics_model="jerk",
    dt=0.1,
    collision_behavior="ignore",
    offroad_behavior="ignore",
    traffic_light_behavior="ignore",
    stop_sign_behavior="ignore",
    termination_mode=False,
)


def test_reverse_kl_direction_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(8, 12)
    reference_logits = torch.randn(8, 12)
    expected = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=reference_logits),
        torch.distributions.Categorical(logits=logits),
    )
    torch.testing.assert_close(P.kl_divergence_to_reference(reference_logits, logits), expected)


def _tracking_env(num_agents=2, **overrides):
    kwargs = dict(
        num_agents=num_agents,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=2,
        map_dir=REPLAY_MAP_DIR,
        scenario_length=SCENARIO_LENGTH,
        resample_frequency=1_000_000,
        report_interval=1,
        num_goals=3,
        **TRACKING_ENV,
    )
    kwargs.update(overrides)
    return Drive(**kwargs)


def test_expert_tracking_follows_log_and_writes_labels():
    env = _tracking_env()
    try:
        env.reset(seed=0)
        placeholder = np.zeros_like(env.actions)
        labels = set()
        stats = {}
        for _ in range(SCENARIO_LENGTH):
            _, _, _, _, info = env.step(placeholder)
            labels.update(env.actions.reshape(-1).tolist())
            assert env.masks.all()
            for log in info:
                stats.update({k: log[k] for k in ("avg_displacement_error", "final_displacement_error") if k in log})
    finally:
        env.close()
    num_actions = len(binding.JERK_LONG) * len(binding.JERK_LAT)
    assert labels and all(0 <= label < num_actions for label in labels)
    assert len(labels) > 1
    assert stats["avg_displacement_error"] < 0.1
    assert stats["final_displacement_error"] < 0.5


def test_expert_tracking_rejects_stop_behaviors():
    with pytest.raises(ValueError, match="ignore"):
        _tracking_env(collision_behavior="stop")


def test_collect_expert_dataset_shapes():
    env = _tracking_env()
    try:
        observations, expert_actions = collect_expert_dataset(env, 20)
    finally:
        env.close()
    assert observations.shape == (40, env.single_observation_space.shape[0])
    assert expert_actions.shape == (40,)
    assert observations.dtype == np.float32


def _bc_args(tmp_path):
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    args["env"].update(
        num_agents=4,
        min_agents_per_env=1,
        max_agents_per_env=1,
        num_maps=1,
        map_dir=REPLAY_MAP_DIR,
        scenario_length=SCENARIO_LENGTH,
        **TRACKING_ENV,
    )
    args["policy"].update(
        ego_input_size=16,
        partner_input_size=16,
        lane_input_size=16,
        boundary_input_size=16,
        traffic_control_input_size=16,
        context_input_size=8,
        backbone_hidden_size=32,
        actor_hidden_size=32,
        critic_hidden_size=32,
        action_type="discrete",
    )
    args["train"].update(device="cpu", seed=0, data_dir=str(tmp_path))
    args["bc"].update(num_steps=150, batch_size=64, max_epochs=3, patience=2, val_fraction=0.1)
    args["wandb"] = False
    args["neptune"] = False
    return args


def test_bc_trains_and_saves_loadable_checkpoint(tmp_path):
    args = _bc_args(tmp_path)
    model_path = bc("puffer_drive", copy.deepcopy(args))
    assert os.path.exists(model_path)
    assert os.path.exists(os.path.join(tmp_path, "config.yaml"))
    assert os.path.exists(os.path.join(tmp_path, "bc_metrics.json"))

    env = Drive(**{**args["env"], "num_agents": 1})
    try:
        from pufferlib.ocean.torch import Drive as DrivePolicy

        policy = DrivePolicy(env, **args["policy"])
        policy.load_state_dict(torch.load(model_path, map_location="cpu"))
        env.reset(seed=0)
        logits, _ = policy(torch.from_numpy(np.array(env.observations)))
        assert logits.shape == (1, 12)
    finally:
        env.close()


def _validated(args):
    validate_puffer_drive_config(normalize_puffer_drive_config(copy.deepcopy(args), "training"), "training")


def test_schema_guards_for_experiment_knobs():
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    _validated(args)

    replay = copy.deepcopy(args)
    replay["env"].update(TRACKING_ENV, map_dir=REPLAY_MAP_DIR, num_maps=1)
    _validated(replay)

    bad = copy.deepcopy(replay)
    bad["env"]["collision_behavior"] = "stop"
    with pytest.raises(pufferlib.APIUsageError, match="expert_tracking"):
        _validated(bad)

    bad = copy.deepcopy(replay)
    bad["env"]["sdc_controller"] = "policy"
    bad["env"]["expert_similarity_only"] = True
    with pytest.raises(pufferlib.APIUsageError, match="expert_similarity_only"):
        _validated(bad)
    bad["env"]["reward_expert_similarity"] = 0.01
    _validated(bad)

    anchored = copy.deepcopy(args)
    anchored["train"]["kl_ref_coef"] = 0.075
    anchored["train"]["kl_ref_direction"] = "reference_to_policy"
    with pytest.raises(pufferlib.APIUsageError, match="kl_ref_coef"):
        _validated(anchored)
    anchored["train"]["kl_ref_model_path"] = "some/anchor.pt"
    _validated(anchored)
