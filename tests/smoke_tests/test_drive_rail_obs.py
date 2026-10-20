#!/usr/bin/env python3
"""Rail-frame observations (env.obs_rail_frame / env.obs_car_frame), the longer rail trail and goal-exit refresh:
sizes agree between drive.py, C and the replay layout; the policy routes every rail column to its object's encoder;
the run configurations roll out with finite observations and a working PPO loss."""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pufferlib
import pufferlib.pytorch
from pufferlib import replay_format
from pufferlib.config_schema import normalize_puffer_drive_config, validate_puffer_drive_config
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import load_config, load_env, load_policy

HYBRID_CONFIG = "puffer_drive_spline_werling_hybrid"
SEED = 7
STEPS = 40
RUN_VARIANTS = {
    "railcar_goalexit": {"obs_rail_frame": True, "lattice_exit_mode": "goal", "lattice_exit_goal_refresh": True},
    "railcar_policyexit": {"obs_rail_frame": True},
    "railonly_goalexit": {
        "obs_rail_frame": True,
        "obs_car_frame": False,
        "lattice_exit_mode": "goal",
        "lattice_exit_goal_refresh": True,
    },
    "railonly_policyexit": {"obs_rail_frame": True, "obs_car_frame": False},
    "carframe_goalexit": {"lattice_exit_mode": "goal", "lattice_exit_goal_refresh": True},
}


def _load_args(config_name):
    with patch.object(sys, "argv", ["pufferl.py"]):
        return load_config(config_name)


def _build_config(variant):
    args = _load_args(HYBRID_CONFIG)
    args["vec"].update({"backend": "Serial", "num_envs": 2, "seed": SEED})
    args["env"].update(
        {
            "num_agents": 16,
            "min_agents_per_env": 8,
            "max_agents_per_env": 8,
            "num_maps": 2,
            "use_map_cache": True,
            "map_dir": "pufferlib/resources/drive/binaries/carla",
            "goal_source": "map",
            "lattice_oncoming_overtake": True,
            "reward_oncoming_penalty_frac": 2.5e-4,
            "lattice_turnaround": True,
            "lattice_trail_keep_m": 60.0 if "rail" in variant else 30.0,
            **RUN_VARIANTS[variant],
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None
    args["train"].update({"device": "cpu", "minibatch_size": 4096, "max_minibatch_size": 4096})
    normalized = normalize_puffer_drive_config(args, "test")
    validate_puffer_drive_config(normalized, "test")
    return normalized


def _random_valid_actions(obs, nvec, rng):
    masks = obs[:, -int(sum(nvec)) :]
    actions = np.zeros((obs.shape[0], len(nvec)), dtype=np.int32)
    offset = 0
    for factor, count in enumerate(nvec):
        for row in range(obs.shape[0]):
            actions[row, factor] = rng.choice(np.flatnonzero(masks[row, offset : offset + count] > 0.5))
        offset += count
    return actions


@pytest.mark.parametrize("variant", list(RUN_VARIANTS))
def test_run_variants_roll_out_and_train(variant):
    args = _build_config(variant)
    rng = np.random.default_rng(SEED)
    vecenv = load_env("puffer_drive", args)
    try:
        driver = vecenv.driver_env
        rail_on = RUN_VARIANTS[variant].get("obs_rail_frame", False)
        expected_rail = (
            driver.obs_slots_partners_n * binding.RAIL_PARTNER_FEATURES
            + (driver.obs_slots_lane_kept + driver.obs_slots_boundary_kept) * binding.RAIL_ROAD_FEATURES
            + (driver.obs_slots_traffic_controls_n + driver.num_goals) * binding.RAIL_POINT_FEATURES
            + binding.LATTICE_PREVIEW_POINTS * binding.RAIL_PREVIEW_FEATURES
        )
        assert driver.rail_obs_features == (expected_rail if rail_on else 0)
        obs, _ = vecenv.reset(seed=SEED)
        assert obs.shape[1] == driver.num_obs
        layout = replay_format.build_obs_layout(driver.observation_layout())
        assert layout["obs_dim"] == driver.num_obs
        rail = obs[:, layout["rail_start"] : layout["rail_start"] + layout["rail_features"]]
        for _ in range(STEPS):
            obs, rewards, _, _, _ = vecenv.step(_random_valid_actions(obs, driver.lattice_nvec, rng))
            assert np.isfinite(obs).all() and np.isfinite(rewards).all()
            rail = obs[:, layout["rail_start"] : layout["rail_start"] + layout["rail_features"]]
        if rail_on:
            partner_valid = rail[
                :, : driver.obs_slots_partners_n * binding.RAIL_PARTNER_FEATURES : binding.RAIL_PARTNER_FEATURES
            ]
            assert set(np.unique(partner_valid)) <= {0.0, 1.0}
            assert rail.any()

        policy = load_policy(args, vecenv, "puffer_drive")
        obs_tensor = torch.as_tensor(obs)
        logits, value = policy(obs_tensor)
        action, logprob, entropy, _ = pufferlib.pytorch.sample_logits(logits)
        loss = -(logprob.mean() + 0.01 * entropy.mean()) + value.float().pow(2).mean()
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        pools = policy.pool_slot_counts(obs_tensor)
        assert pools["pool_partner"].shape == (obs.shape[0], driver.obs_slots_partners_n)
        assert pools["pool_lane"].sum(dim=1).eq(policy.actor_backbone.lane_input_size).all()
    finally:
        vecenv.close()


def test_backbone_routes_rail_columns_to_their_objects():
    args = _build_config("railcar_policyexit")
    vecenv = load_env("puffer_drive", args)
    try:
        driver = vecenv.driver_env
        backbone = load_policy(args, vecenv, "puffer_drive").actor_backbone
        layout = replay_format.build_obs_layout(driver.observation_layout())
        ego_dim = layout["ego_features"]
        columns = torch.arange(driver.num_obs, dtype=torch.float32).unsqueeze(0)
        for slot in range(driver.obs_slots_traffic_controls_n):
            type_column = (
                layout["traffic_start"] + slot * binding.TRAFFIC_CONTROL_FEATURES + binding.TRAFFIC_CONTROL_FEATURES - 2
            )
            columns[0, type_column : type_column + 2] = 0.0
        ego, context, objects, _ = backbone._observation_blocks(columns, ego_dim)
        rail = layout["rail_start"]
        partner_rail = rail
        lane_rail = partner_rail + driver.obs_slots_partners_n * binding.RAIL_PARTNER_FEATURES
        boundary_rail = lane_rail + driver.obs_slots_lane_kept * binding.RAIL_ROAD_FEATURES
        traffic_rail = boundary_rail + driver.obs_slots_boundary_kept * binding.RAIL_ROAD_FEATURES
        goal_rail = traffic_rail + driver.obs_slots_traffic_controls_n * binding.RAIL_POINT_FEATURES
        preview_rail = goal_rail + driver.num_goals * binding.RAIL_POINT_FEATURES
        assert (
            preview_rail + binding.LATTICE_PREVIEW_POINTS * binding.RAIL_PREVIEW_FEATURES
            == layout["rail_start"] + layout["rail_features"]
        )
        assert ego[0, :ego_dim].tolist() == list(range(ego_dim))
        assert ego[0, ego_dim:].tolist() == list(range(preview_rail, preview_rail + 12))
        assert context[0, -driver.num_goals * binding.RAIL_POINT_FEATURES :].tolist() == list(
            range(goal_rail, preview_rail)
        )
        slot = 3
        partner_row = objects["partner"][0, slot].tolist()
        assert partner_row[: binding.PARTNER_FEATURES] == list(
            range(
                layout["partner_start"] + slot * binding.PARTNER_FEATURES,
                layout["partner_start"] + (slot + 1) * binding.PARTNER_FEATURES,
            )
        )
        assert partner_row[binding.PARTNER_FEATURES :] == list(
            range(
                partner_rail + slot * binding.RAIL_PARTNER_FEATURES,
                partner_rail + (slot + 1) * binding.RAIL_PARTNER_FEATURES,
            )
        )
        lane_row = objects["lane"][0, slot].tolist()
        assert lane_row[binding.LANE_FEATURES :] == list(
            range(lane_rail + slot * binding.RAIL_ROAD_FEATURES, lane_rail + (slot + 1) * binding.RAIL_ROAD_FEATURES)
        )
        boundary_row = objects["boundary"][0, slot].tolist()
        assert boundary_row[binding.BOUNDARY_FEATURES :] == list(
            range(
                boundary_rail + slot * binding.RAIL_ROAD_FEATURES,
                boundary_rail + (slot + 1) * binding.RAIL_ROAD_FEATURES,
            )
        )
        traffic_row = objects["traffic_control"][0, 1].tolist()
        continuous = binding.TRAFFIC_CONTROL_FEATURES - 2
        assert traffic_row[continuous : continuous + binding.RAIL_POINT_FEATURES] == list(
            range(traffic_rail + binding.RAIL_POINT_FEATURES, traffic_rail + 2 * binding.RAIL_POINT_FEATURES)
        )
    finally:
        vecenv.close()


def test_python_c_rail_size_mismatch_is_rejected(monkeypatch):
    original = Drive._env_init_kwargs

    def off_by_one(self, *args, **kwargs):
        env_kwargs = original(self, *args, **kwargs)
        env_kwargs["rail_obs_feature_count"] += 1
        return env_kwargs

    monkeypatch.setattr(Drive, "_env_init_kwargs", off_by_one)
    env_kwargs = dict(_build_config("railcar_policyexit")["env"])
    env_kwargs.update({"num_maps": 1, "use_map_cache": False})
    with pytest.raises(ValueError, match="rail obs features"):
        Drive(**env_kwargs)


def test_replay_layout_without_rail_entry_still_builds():
    args = _build_config("carframe_goalexit")
    env_kwargs = dict(args["env"])
    env_kwargs.update({"num_maps": 1, "use_map_cache": False})
    env = Drive(**env_kwargs)
    try:
        counts = env.observation_layout()
        assert counts["rail_features"] == 0
        del counts["rail_features"]
        assert replay_format.build_obs_layout(counts)["obs_dim"] == env.num_obs
    finally:
        env.close()


@pytest.mark.parametrize(
    "config_name,overrides",
    [
        ("puffer_drive", {"obs_rail_frame": True}),
        (HYBRID_CONFIG, {"obs_car_frame": False}),
        (HYBRID_CONFIG, {"lattice_trail_keep_m": 60.25}),
        (HYBRID_CONFIG, {"lattice_trail_keep_m": 29.5}),
        (HYBRID_CONFIG, {"lattice_trail_keep_m": 94.5}),
        (HYBRID_CONFIG, {"lattice_exit_goal_refresh": True}),
    ],
)
def test_invalid_rail_settings_are_rejected(config_name, overrides):
    args = _load_args(config_name)
    args["env"].update(overrides)
    with pytest.raises(pufferlib.APIUsageError):
        validate_puffer_drive_config(normalize_puffer_drive_config(args, "test"), "test")


def test_defaults_leave_the_rail_block_off():
    args = _load_args(HYBRID_CONFIG)
    env = args["env"]
    assert env["obs_rail_frame"] is False and env["obs_car_frame"] is True
    assert env["lattice_trail_keep_m"] == 30.0 and env["lattice_exit_goal_refresh"] is False
