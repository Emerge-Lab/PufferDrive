"""Speed bonus counted from a start speed: same rollout with and without it; rewards differ by the formula."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.config_schema import normalize_puffer_drive_config
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive
from pufferlib.pufferl import load_config

SEED = 5
STEPS = 80
SPEED_BONUS = 4e-3
BONUS_FROM_MPS = 1.5  # random actions stay below 5 m/s, so a low start speed exercises the counted-from path
CARLA_MAP_DIR = "pufferlib/resources/drive/binaries/carla"
X_FIELD, Y_FIELD, HEADING_FIELD, SPEED_FIELD = 0, 1, 3, 6  # AGENT_F32_*_IDX; the speed is unsigned
VALID_FIELD, ACTIVE_FIELD, STOPPED_FIELD, REMOVED_FIELD, SLOT_FIELD = 2, 3, 4, 5, 7  # agent_i32 columns
LANE_ANGLE_METRIC = 6  # LANE_ANGLE_IDX in constants.h


def _env_kwargs():
    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        args = load_config("puffer_drive")
    finally:
        sys.argv = saved_argv
    args["vec"].update({"backend": "Serial", "num_envs": 1, "seed": SEED})
    args["env"].update(
        {
            "num_agents": 16,
            "min_agents_per_env": 16,
            "max_agents_per_env": 16,
            "num_maps": 8,
            "use_map_cache": False,
            "map_dir": CARLA_MAP_DIR,
            # episodes end inside the rollout so the per-episode log reports the bonus
            "scenario_length": 60,
            "resample_frequency": 1000,
        }
    )
    args["wandb"] = False
    args["neptune"] = False
    args["eval"] = None
    env_kwargs = dict(normalize_puffer_drive_config(args, "test")["env"])
    assert env_kwargs["reward_speed_bonus"] == 0.0 and env_kwargs["reward_speed_bonus_from_mps"] == 0.0
    return env_kwargs


def _rollout(speed_bonus):
    overrides = {
        "reward_speed_bonus": speed_bonus,
        "reward_speed_bonus_from_mps": BONUS_FROM_MPS,
        "capture_replay": True,
    }
    env = Drive(**{**_env_kwargs(), **overrides})
    rng = np.random.default_rng(SEED)
    rewards, logs = [], []
    try:
        env.reset(seed=SEED)
        for _ in range(STEPS):
            _, step_rewards, _, _, info = env.step(rng.uniform(-1.0, 1.0, size=(env.num_agents, 2)).astype(np.float32))
            rewards.append(np.array(step_rewards, copy=True))
            logs += info
        logs.append(binding.vec_log(env.c_envs, 1))
        frames = env._replay_captures[0]["frames"]
        captured = (np.stack(frames["agent_f32"]), np.stack(frames["agent_i32"]), np.stack(frames["metrics_f32"]))
        bonus_logged = [log["reward_components/speed_bonus"] for log in logs if "reward_components/speed_bonus" in log]
        return captured, np.stack(rewards), env.dt, env.base_max_speed_mps, bonus_logged
    finally:
        env.close()


def test_speed_bonus_adds_exactly_its_formula_to_each_step():
    (agents_off, flags_off, metrics_off), rewards_off, dt, top_speed, logged_off = _rollout(0.0)
    (agents_on, flags_on, metrics_on), rewards_on, _, _, logged_on = _rollout(SPEED_BONUS)
    assert logged_off and all(value == 0.0 for value in logged_off)
    assert logged_on and max(logged_on) > 0.0
    # rewards never feed back into the simulation, so both rollouts drive identically
    np.testing.assert_array_equal(agents_off, agents_on)
    np.testing.assert_array_equal(flags_off, flags_on)
    np.testing.assert_array_equal(metrics_off, metrics_on)

    # frame k is captured before step k, so step k's reward reads the state in frame k + 1
    before, after = agents_on[:-1], agents_on[1:]
    slot = flags_on[1:, :, SLOT_FIELD]
    live = slot >= 0
    for flags in (flags_on[:-1], flags_on[1:]):
        live &= (flags[..., VALID_FIELD] == 1) & (flags[..., ACTIVE_FIELD] == 1)
        live &= (flags[..., STOPPED_FIELD] == 0) & (flags[..., REMOVED_FIELD] == 0)
    step_x = after[..., X_FIELD] - before[..., X_FIELD]
    step_y = after[..., Y_FIELD] - before[..., Y_FIELD]
    # a car respawned at the end of a step was paid for its speed before the jump
    live &= np.hypot(step_x, step_y) < top_speed * dt + 1.0
    forward = step_x * np.cos(after[..., HEADING_FIELD]) + step_y * np.sin(after[..., HEADING_FIELD]) > 0.0
    alignment = np.maximum(metrics_on[1:, :, LANE_ANGLE_METRIC], 0.0)
    above = np.maximum(after[..., SPEED_FIELD] - BONUS_FROM_MPS, 0.0)
    share = np.minimum(above / (top_speed - BONUS_FROM_MPS), 1.0) * forward
    expected = SPEED_BONUS * dt * alignment * share
    added_by_slot = (rewards_on - rewards_off)[:-1]
    added = np.take_along_axis(added_by_slot, np.clip(slot, 0, None), axis=1)

    assert live.sum() > 0.5 * live.size
    assert (live & (expected > 1e-5)).sum() > 30
    np.testing.assert_allclose(added[live], expected[live], rtol=0.0, atol=2e-6)
    assert np.all(added_by_slot >= -1e-6)
