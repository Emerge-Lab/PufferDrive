#!/usr/bin/env python3
"""CPU smoke test for env.trajectory_baseline, the view-only quintic fitted to every jerk step.

The flag must not change the simulation, so a rollout with it on is compared bit for bit against one
with it off. The fitted curve reaches the viewer through the replay export, whose first path sample
(t = dt) must sit on the car. The wrapper and C init must both reject the flag outside jerk + continuous.
"""

import json
import os
import re
import struct
import sys
import tempfile
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pufferlib.viz
from pufferlib.config_schema import normalize_puffer_drive_config
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import load_config, load_env

SEED = 11
STEPS = 200
RESAMPLE_FREQUENCY = 64
EXPORT_STEPS = 60
VIEWER_STEPS = 10
CARLA_MAP_DIR = "pufferlib/resources/drive/binaries/carla"
EGO_IDX = 0  # matches constants.h
PATH_START_TOLERANCE_M = 1e-3


def _build_config(trajectory_baseline):
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv

    args["vec"].update({"backend": "Serial", "num_envs": 2, "seed": SEED})
    args["env"].update(
        {
            "trajectory_baseline": trajectory_baseline,
            "num_agents": 8,
            "min_agents_per_env": 8,
            "max_agents_per_env": 8,
            "num_maps": 2,
            "use_map_cache": True,
            "map_dir": CARLA_MAP_DIR,
            # episodes end and maps resample inside the rollout, so the C re-init path sees the flag too
            "scenario_length": 91,
            "resample_frequency": RESAMPLE_FREQUENCY,
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None

    normalized = normalize_puffer_drive_config(args, "test")
    assert normalized["env"]["dynamics_model"] == "jerk"
    assert normalized["env"]["action_type"] == "continuous"
    assert normalized["env"]["trajectory_baseline"] is trajectory_baseline
    return normalized


def _record_rollout(trajectory_baseline):
    vecenv = load_env("puffer_drive", _build_config(trajectory_baseline))
    rng = np.random.default_rng(SEED)
    records = []
    try:
        obs, _ = vecenv.reset(seed=SEED)
        records.append(obs.copy())
        shape = (vecenv.num_agents,) + vecenv.single_action_space.shape
        for _ in range(STEPS):
            actions = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
            obs, rewards, terminals, truncations, _ = vecenv.step(actions)
            records.extend([obs.copy(), rewards.copy(), terminals.copy(), truncations.copy()])
    finally:
        vecenv.close()
    return records


def test_trajectory_baseline_leaves_the_rollout_unchanged():
    plain = _record_rollout(False)
    baseline = _record_rollout(True)
    assert len(plain) == len(baseline)
    for record_idx, (plain_record, baseline_record) in enumerate(zip(plain, baseline)):
        assert np.array_equal(plain_record, baseline_record), f"rollouts diverged at record {record_idx}"


def _make_drive(**overrides):
    env_kwargs = dict(_build_config(False)["env"])
    env_kwargs.update({"num_maps": 1, "use_map_cache": False})
    env_kwargs.update(overrides)
    return Drive(**env_kwargs)


def _capture_replay(step_count):
    env = _make_drive(trajectory_baseline=True, capture_replay=True)
    rng = np.random.default_rng(SEED)
    try:
        env.reset(seed=SEED)
        for _ in range(step_count):
            env.step(rng.uniform(-1.0, 1.0, size=(env.num_agents, 2)).astype(np.float32))
        return env._replay_captures[0]
    finally:
        env.close()


def test_trajectory_baseline_exports_a_path_that_starts_at_the_car():
    frames = _capture_replay(EXPORT_STEPS)["frames"]["agent_f32"]
    assert len(frames) == EXPORT_STEPS
    path_start = binding.AGENT_F32_PATH_BASE_IDX
    path_end = path_start + 2 * binding.AGENT_F32_PATH_SAMPLES
    # frame k is captured before step k, so frame 0 and post-reset frames show a curve collapsed onto the car
    for frame_idx, frame in enumerate(frames):
        ego = frame[EGO_IDX]
        path = ego[path_start:path_end].reshape(binding.AGENT_F32_PATH_SAMPLES, 2)
        assert np.isfinite(path).all(), f"non-finite exported path at frame {frame_idx}"
        assert np.allclose(path[0], ego[0:2], rtol=0.0, atol=PATH_START_TOLERANCE_M), (
            f"frame {frame_idx}: path starts at {path[0]}, car is at {ego[0:2]}"
        )


def test_viewer_is_told_to_draw_the_fitted_path():
    capture = _capture_replay(VIEWER_STEPS)
    frames = {key: np.stack(values, axis=0) for key, values in capture["frames"].items()}
    frame_count = frames["agent_f32"].shape[0]
    active_count = capture["metadata"]["active_agent_count"]
    replay = {
        "env": _build_config(True)["env"],
        **frames,
        "raw_action": np.zeros((frame_count, active_count, 2), np.float32),
        "clipped_action": np.zeros((frame_count, active_count, 2), np.float32),
        "value": np.zeros((frame_count, active_count), np.float32),
        "entropy": np.zeros((frame_count, active_count), np.float32),
    }
    compressed = pufferlib.viz.encode_interactive_replay(capture["scenario"], replay)
    payload = zlib.decompress(compressed)
    header_length = struct.unpack("<I", payload[:4])[0]
    header = json.loads(payload[4 : 4 + header_length])
    assert header["trajectory_baseline"] is True
    assert header["action_type"] == "continuous"

    with tempfile.TemporaryDirectory() as output_dir:
        html_path = os.path.join(output_dir, "replay.html")
        pufferlib.viz._render_interactive_replay_payload(compressed, html_path)
        with open(html_path) as html_file:
            html = html_file.read()
    # the path drawing and its P-key toggle both gate on the flag
    assert html.count("H.trajectory_baseline") == 2


def _expect_error(error_type, message, build):
    try:
        result = build()
    except error_type as error:
        assert re.search(message, str(error)), f"{error_type.__name__} message {str(error)!r} lacks {message!r}"
        return
    raise AssertionError(f"expected {error_type.__name__} matching {message!r}, got {result!r}")


def test_wrapper_rejects_trajectory_baseline_outside_jerk_continuous():
    cases = [
        (dict(dynamics_model="classic"), "requires dynamics_model 'jerk' and action_type 'continuous'"),
        (dict(action_type="discrete"), "requires dynamics_model 'jerk' and action_type 'continuous'"),
        (dict(dynamics_model="spline", action_type="spline"), "requires dynamics_model 'jerk'"),
        (dict(spline_horizon_seconds=0.3), r"requires spline_horizon_seconds > dt"),
    ]
    for overrides, message in cases:
        _expect_error(ValueError, message, lambda: _make_drive(trajectory_baseline=True, **overrides))


def _c_env_init(env, kwarg_overrides):
    agent_count = env.agent_offsets[1] - env.agent_offsets[0]
    kwargs = env._env_init_kwargs(env.map_files[env.map_ids[0]], agent_count)
    kwargs.update(kwarg_overrides)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return binding.env_init(
        env.observations[:agent_count],
        env.actions[:agent_count],
        env.rewards[:agent_count],
        env.terminals[:agent_count],
        env.truncations[:agent_count],
        env.masks[:agent_count],
        SEED,
        **kwargs,
    )


def test_c_init_rejects_bad_trajectory_baseline():
    """Python params are untrusted at the C boundary, so env_init must refuse them without the wrapper."""
    wrong_mode = "requires dynamics_model jerk and action_type continuous"
    cases = [
        (dict(trajectory_baseline=2), ValueError, "trajectory_baseline must be 0 or 1"),
        (dict(trajectory_baseline=True, action_type=binding.ACTION_TYPE_DISCRETE), ValueError, wrong_mode),
        (dict(trajectory_baseline=True, dynamics_model=binding.DYNAMICS_MODEL_CLASSIC), ValueError, wrong_mode),
        (dict(trajectory_baseline=True, spline_horizon_seconds=0.3), ValueError, r"spline_horizon_seconds > dt"),
        (dict(trajectory_baseline=None), TypeError, "Missing required keyword argument 'trajectory_baseline'"),
    ]
    env = _make_drive()
    try:
        for kwarg_overrides, error_type, message in cases:
            _expect_error(error_type, message, lambda: _c_env_init(env, kwarg_overrides))
    finally:
        env.close()


if __name__ == "__main__":
    test_trajectory_baseline_leaves_the_rollout_unchanged()
    test_trajectory_baseline_exports_a_path_that_starts_at_the_car()
    test_viewer_is_told_to_draw_the_fitted_path()
    test_wrapper_rejects_trajectory_baseline_outside_jerk_continuous()
    test_c_init_rejects_bad_trajectory_baseline()
    print("Trajectory baseline smoke tests passed!")
