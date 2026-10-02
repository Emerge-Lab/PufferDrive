#!/usr/bin/env python3
"""CPU smoke test for dynamics_model spline_werling (Werling lattice) with the lattice policy.

Random valid actions drawn from the observed masks: finite observations and rewards, the masks at the end of each
row with one valid gate and exit per row, and a policy whose masked logits, PPO loss and LSTM path all run.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import struct
import tempfile
import zlib

import pufferlib.pytorch
import pufferlib.viz
from pufferlib.config_schema import normalize_puffer_drive_config
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import load_config, load_env, load_policy

PATH_START_TOLERANCE_M = 1.0
EXPORT_STEPS = 30

SEED = 5
STEPS = 150


def _build_config(rnn=False, oncoming_overtake=False, turnaround=False):
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive_spline_werling")
    finally:
        sys.argv = saved_argv
    args["vec"].update({"backend": "Serial", "num_envs": 2, "seed": SEED})
    args["env"].update(
        {
            "num_agents": 16,
            "min_agents_per_env": 8,
            "max_agents_per_env": 8,
            "num_maps": 2,
            "use_map_cache": True,
            "map_dir": "pufferlib/resources/drive/binaries/carla",
            "lattice_oncoming_overtake": oncoming_overtake,
            "reward_oncoming_penalty_frac": 5e-4 if oncoming_overtake else 0.0,
            "lattice_turnaround": turnaround,
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None
    args["train"]["device"] = "cpu"
    if rnn:
        args["rnn_name"] = "Recurrent"
        args["rnn"] = {"input_size": args["policy"]["backbone_hidden_size"], "hidden_size": args["policy"]["backbone_hidden_size"]}
        args["policy"]["shared_network"] = True
    return normalize_puffer_drive_config(args, "test")


def _random_valid_actions(obs, nvec, rng):
    mask_count = int(sum(nvec))
    masks = obs[:, -mask_count:]
    actions = np.zeros((obs.shape[0], len(nvec)), dtype=np.int32)
    offset = 0
    for factor, count in enumerate(nvec):
        factor_masks = masks[:, offset : offset + count]
        for row in range(obs.shape[0]):
            valid = np.flatnonzero(factor_masks[row] > 0.5)
            assert len(valid) > 0, f"factor {factor} row {row} has no valid choice"
            actions[row, factor] = rng.choice(valid)
        offset += count
    return actions


def test_spline_werling_rollout():
    args = _build_config()
    rng = np.random.default_rng(SEED)
    vecenv = load_env("puffer_drive", args)
    try:
        driver = vecenv.driver_env
        nvec = driver.lattice_nvec
        assert list(vecenv.single_action_space.nvec) == nvec == [2, 20, 2, 60, 5]
        obs, _ = vecenv.reset(seed=SEED)
        assert obs.shape[1] == driver.num_obs
        for step in range(STEPS):
            obs, rewards, _, _, _ = vecenv.step(_random_valid_actions(obs, nvec, rng))
            assert np.isfinite(obs).all(), f"non-finite observation at step {step}"
            assert np.isfinite(rewards).all(), f"non-finite reward at step {step}"
    finally:
        vecenv.close()


def test_spline_werling_oncoming_overtake_rollout():
    args = _build_config(oncoming_overtake=True)
    rng = np.random.default_rng(SEED)
    vecenv = load_env("puffer_drive", args)
    try:
        driver = vecenv.driver_env
        nvec = driver.lattice_nvec
        assert list(vecenv.single_action_space.nvec) == nvec == [2, 24, 2, 60, 5]
        assert driver.lattice_plan_features == binding.LATTICE_PLAN_FEATURES + 1
        obs, _ = vecenv.reset(seed=SEED)
        assert obs.shape[1] == driver.num_obs
        for step in range(STEPS):
            obs, rewards, _, _, _ = vecenv.step(_random_valid_actions(obs, nvec, rng))
            assert np.isfinite(obs).all(), f"non-finite observation at step {step}"
            assert np.isfinite(rewards).all(), f"non-finite reward at step {step}"
        policy = load_policy(args, vecenv, "puffer_drive")
        logits, _ = policy(torch.as_tensor(obs))
        assert logits[1].shape[1] == 24
    finally:
        vecenv.close()


def test_spline_werling_turnaround_rollout():
    args = _build_config(oncoming_overtake=True, turnaround=True)
    rng = np.random.default_rng(SEED)
    vecenv = load_env("puffer_drive", args)
    try:
        driver = vecenv.driver_env
        nvec = driver.lattice_nvec
        assert list(vecenv.single_action_space.nvec) == nvec == [2, 24, 2, 61, 5]
        assert driver.lattice_plan_features == binding.LATTICE_PLAN_FEATURES + 1 + binding.LATTICE_TURN_PLAN_FEATURES
        obs, _ = vecenv.reset(seed=SEED)
        assert obs.shape[1] == driver.num_obs
        for step in range(STEPS):
            obs, rewards, _, _, _ = vecenv.step(_random_valid_actions(obs, nvec, rng))
            assert np.isfinite(obs).all(), f"non-finite observation at step {step}"
            assert np.isfinite(rewards).all(), f"non-finite reward at step {step}"
        policy = load_policy(args, vecenv, "puffer_drive")
        logits, _ = policy(torch.as_tensor(obs))
        assert logits[3].shape[1] == 61
    finally:
        vecenv.close()


def test_spline_werling_low_speed_penalty_rollout():
    args = _build_config(oncoming_overtake=True, turnaround=True)
    args["env"].update(
        {
            "reward_wait_penalty_frac": 7e-4,
            "reward_wait_full_speed_mps": 5.0,
            "reward_oncoming_penalty_frac": 2.5e-4,
            "reward_trajectory_consistency": 2e-3,
        }
    )
    args = normalize_puffer_drive_config(args, "test")
    rng = np.random.default_rng(SEED)
    vecenv = load_env("puffer_drive", args)
    try:
        driver = vecenv.driver_env
        assert driver.reward_wait_full_speed_mps == 5.0
        obs, _ = vecenv.reset(seed=SEED)
        total = 0.0
        for step in range(STEPS):
            obs, rewards, _, _, _ = vecenv.step(_random_valid_actions(obs, driver.lattice_nvec, rng))
            assert np.isfinite(rewards).all(), f"non-finite reward at step {step}"
            total += float(rewards.sum())
        assert np.isfinite(total)
    finally:
        vecenv.close()
    bad = _build_config()
    bad["env"]["reward_wait_full_speed_mps"] = 0.0
    try:
        load_env("puffer_drive", bad).close()
    except ValueError as error:
        assert "reward_wait_full_speed_mps" in str(error)
    else:
        raise AssertionError("a zero full speed must be rejected by the binding")


def _policy_losses(rnn):
    args = _build_config(rnn=rnn)
    vecenv = load_env("puffer_drive", args)
    try:
        policy = load_policy(args, vecenv, "puffer_drive")
        obs, _ = vecenv.reset(seed=SEED)
        obs_tensor = torch.as_tensor(obs)
        state = {"lstm_h": None, "lstm_c": None}
        if rnn:
            logits, value = policy(obs_tensor.unsqueeze(1), state)
        else:
            logits, value = policy(obs_tensor)
        assert isinstance(logits, pufferlib.pytorch.LatticeLogits)
        nvec = vecenv.driver_env.lattice_nvec
        masks = torch.split(obs_tensor[:, -sum(nvec) :], nvec, dim=1)
        for factor_logits, factor_mask in zip(logits, masks):
            assert (factor_logits[factor_mask < 0.5] < -1e8).all()
        action, logprob, entropy, _ = pufferlib.pytorch.sample_logits(logits)
        for factor, factor_mask in enumerate(masks):
            assert (factor_mask.gather(1, action[:, factor : factor + 1]) > 0.5).all()
        _, newlogprob, newentropy, _ = pufferlib.pytorch.sample_logits(logits, action=action)
        assert torch.allclose(newlogprob, logprob)
        loss = -(newlogprob.mean() + 0.01 * newentropy.mean()) + value.float().pow(2).mean()
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        vecenv.step(action.numpy().astype(np.int32))
    finally:
        vecenv.close()


def test_spline_werling_policy_and_ppo_loss():
    _policy_losses(rnn=False)


def test_spline_werling_recurrent_policy():
    _policy_losses(rnn=True)


def _capture_lattice_replay():
    env_kwargs = dict(_build_config()["env"])
    env_kwargs.update({"num_maps": 1, "use_map_cache": False, "capture_replay": True})
    env = Drive(**env_kwargs)
    rng = np.random.default_rng(SEED)
    try:
        obs, _ = env.reset(seed=SEED)
        for _ in range(EXPORT_STEPS):
            obs, _, _, _, _ = env.step(_random_valid_actions(obs, env.lattice_nvec, rng))
        return env._replay_captures[0]
    finally:
        env.close()


def test_spline_werling_exports_every_agents_committed_plan():
    capture = _capture_lattice_replay()
    agent_f32 = capture["frames"]["agent_f32"]
    agent_i32 = capture["frames"]["agent_i32"]
    path_start = binding.AGENT_F32_PATH_BASE_IDX
    path_end = path_start + 2 * binding.AGENT_F32_PATH_SAMPLES
    checked = 0
    for frame_idx in range(1, len(agent_f32)):
        for agent_idx in range(agent_f32[frame_idx].shape[0]):
            ints = agent_i32[frame_idx][agent_idx]
            if not ints[2] or not ints[3] or ints[4] or ints[5]:
                continue
            row = agent_f32[frame_idx][agent_idx]
            path = row[path_start:path_end].reshape(binding.AGENT_F32_PATH_SAMPLES, 2)
            assert np.isfinite(path).all()
            assert np.abs(path).sum() > 0.0, f"frame {frame_idx} agent {agent_idx}: no exported plan"
            assert np.linalg.norm(path[0] - row[0:2]) < PATH_START_TOLERANCE_M, (
                f"frame {frame_idx} agent {agent_idx}: plan starts at {path[0]}, car at {row[0:2]}"
            )
            checked += 1
    assert checked > EXPORT_STEPS


def test_viewer_draws_lattice_plans_for_every_agent():
    capture = _capture_lattice_replay()
    frames = {key: np.stack(values, axis=0) for key, values in capture["frames"].items()}
    frame_count = frames["agent_f32"].shape[0]
    active_count = capture["metadata"]["active_agent_count"]
    replay = {
        "env": _build_config()["env"],
        **frames,
        "raw_action": np.zeros((frame_count, active_count, 5), np.float32),
        "clipped_action": np.zeros((frame_count, active_count, 5), np.float32),
        "value": np.zeros((frame_count, active_count), np.float32),
        "entropy": np.zeros((frame_count, active_count), np.float32),
    }
    compressed = pufferlib.viz.encode_interactive_replay(capture["scenario"], replay)
    payload = zlib.decompress(compressed)
    header_length = struct.unpack("<I", payload[:4])[0]
    header = json.loads(payload[4 : 4 + header_length])
    assert header["action_type"] == "lattice"
    assert header["dynamics_model"] == "spline_werling"
    assert header["agent_path_sample_count"] == binding.AGENT_F32_PATH_SAMPLES
    with tempfile.TemporaryDirectory() as output_dir:
        html_path = os.path.join(output_dir, "replay.html")
        pufferlib.viz._render_interactive_replay_payload(compressed, html_path)
        with open(html_path) as html_file:
            html = html_file.read()
    assert 'const lattice = H.action_type === "lattice"' in html


if __name__ == "__main__":
    test_spline_werling_rollout()
    test_spline_werling_oncoming_overtake_rollout()
    test_spline_werling_turnaround_rollout()
    test_spline_werling_low_speed_penalty_rollout()
    test_spline_werling_policy_and_ppo_loss()
    test_spline_werling_recurrent_policy()
    print("OK")
