"""M7 difficulty levels: sampling windows, validation, the human car, and their effect on traffic."""

import math
import os

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo import levels, policy_host

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "weights", "k_scaled_0050_1000", "final_model.pt")
MAP_FILE = os.path.join(
    REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "carla", "opendrive__Town01.bin"
)

AGENT_COUNT = 32
SEED = 7
HUMAN_AGENT_IDX = 0
FRAME_AGENT_CAPACITY = 256
FRAME_TRAFFIC_CAPACITY = 1
AGENT_I32_ACTIVE_FIELD = 3
AGENT_I32_SLOT_FIELD = 7
AGENT_F32_SPEED_FIELD = 6
INACTIVE_SLOT = -1
RESAMPLE_ROUNDS = 40
NORMALIZED_SPAN = 2.0
ROUND_TRIP_RELATIVE_TOLERANCE = 1e-4
MIN_CONDITIONING_GAP = 0.5
SPEED_COEF_PIVOT = 1.0
TRAINING_SPEED_SHARE_BELOW_PIVOT = 0.5
SPEED_SHARE_TOLERANCE = 0.05
FULL_WINDOW_LOW_FRACTION = 0.1
FULL_WINDOW_HIGH_FRACTION = 0.9
POLICY_STEP_COUNT = 40
MIN_MEAN_SPEED_GAP_MPS = 0.5
ALTERNATE_SPEED_SCALE = 1.5
# REWARD_COEF_ACC is not exported by the binding; it sits just before REWARD_COEF_SPEED
FIXED_PROFILE_COEF_IDX = levels.COEF_SPEED - 1
NARROW_WINDOW = (0.6, 0.8)
FULL_WINDOW = (0.0, 1.0)

WINDOWED_COEFS = sorted({coef_idx for level in levels.LEVELS for coef_idx in level.windows})

requires_checkpoint = pytest.mark.skipif(
    not (os.path.isfile(CHECKPOINT_PATH) and os.path.isfile(MAP_FILE)),
    reason="demo checkpoint or Town01 map not present",
)


def make_env(overrides=None, agent_count=AGENT_COUNT):
    config, _ = policy_host.load_checkpoint_config(CHECKPOINT_PATH)
    env = policy_host.build_demo_env(config, MAP_FILE, agent_count, overrides)
    env.reset(seed=SEED)
    return env


@pytest.fixture
def env():
    drive_env = make_env()
    yield drive_env
    drive_env.close()


def agent_frame(env):
    """(coefs, active_slot, speed_mps) indexed by agent idx; active_slot is -1 for inactive agents."""
    agent_f32 = np.empty((1, FRAME_AGENT_CAPACITY, binding.AGENT_F32_FIELDS), dtype=np.float32)
    agent_i32 = np.empty((1, FRAME_AGENT_CAPACITY, binding.AGENT_I32_FIELDS), dtype=np.int32)
    metrics_f32 = np.empty((1, FRAME_AGENT_CAPACITY, binding.METRICS_F32_FIELDS), dtype=np.float32)
    puffer_f32 = np.empty((1, FRAME_AGENT_CAPACITY, binding.SCORE_F32_FIELDS), dtype=np.float32)
    traffic_i16 = np.empty((1, FRAME_TRAFFIC_CAPACITY, binding.TRAFFIC_I16_FIELDS), dtype=np.int16)
    goals_f32 = np.empty((1, FRAME_AGENT_CAPACITY, env.num_goals * binding.GOAL_XY_FIELDS), dtype=np.float32)
    rewards_f32 = np.empty((1, FRAME_AGENT_CAPACITY, binding.REWARD_F32_FIELDS), dtype=np.float32)
    coefs_f32 = np.empty((1, FRAME_AGENT_CAPACITY, binding.NUM_REWARD_COEFS), dtype=np.float32)
    env.get_obs_html_frame(
        agent_f32, agent_i32, metrics_f32, puffer_f32, traffic_i16, goals_f32, rewards_f32, coefs_f32
    )
    active = agent_i32[0, :, AGENT_I32_ACTIVE_FIELD] == 1
    active_slot = np.where(active, agent_i32[0, :, AGENT_I32_SLOT_FIELD], INACTIVE_SLOT)
    return coefs_f32[0], active_slot, agent_f32[0, :, AGENT_F32_SPEED_FIELD]


def policy_car_coefs(env, human_agent_idx=None):
    coefs, active_slot, _speeds = agent_frame(env)
    is_policy_car = active_slot != INACTIVE_SLOT
    if human_agent_idx is not None:
        is_policy_car[human_agent_idx] = False
    return coefs[is_policy_car]


def reward_target_slice(env, observations):
    return observations[:, env.ego_features : env.ego_features + env.num_reward_coefs]


def sampled_speed_coefs(env):
    samples = []
    for _round in range(RESAMPLE_ROUNDS):
        env.resample_reward_coefs()
        samples.append(policy_car_coefs(env)[:, levels.COEF_SPEED])
    return np.concatenate(samples)


@requires_checkpoint
def test_level_shifts_the_window_without_pinning_a_value(env):
    low, high = levels.level_bounds(env, levels.AGGRESSIVE)
    levels.apply_level(env, levels.AGGRESSIVE)
    coefs = policy_car_coefs(env)
    assert len(coefs) == env.num_agents
    for coef_idx in levels.AGGRESSIVE.windows:
        values = coefs[:, coef_idx]
        assert np.all(values >= low[coef_idx]) and np.all(values <= high[coef_idx]), coef_idx
        assert len(np.unique(values)) > 1, f"coefficient {coef_idx} is pinned, not windowed"


@requires_checkpoint
@pytest.mark.parametrize(
    "window",
    [(-0.1, 0.5), (0.2, 1.1), (0.6, 0.4), (math.nan, 0.5), (0.0, math.inf)],
)
def test_fractions_outside_unit_interval_raise(env, window):
    bad_level = levels.Level(name="bad", description="invalid window", windows={levels.COEF_COLLISION: window})
    with pytest.raises(ValueError):
        levels.level_bounds(env, bad_level)
    with pytest.raises(ValueError):
        levels.apply_level(env, bad_level)


@requires_checkpoint
@pytest.mark.parametrize("speed_scale", [None, ALTERNATE_SPEED_SCALE])
def test_speed_window_uses_conditioning_speed_scale(speed_scale):
    overrides = None if speed_scale is None else {"conditioning_speed_scale": speed_scale}
    drive_env = make_env(overrides)
    try:
        scale = drive_env.conditioning_speed_scale
        trained_min, trained_max = drive_env.get_reward_coef_limits()
        assert trained_min[levels.COEF_SPEED] == pytest.approx(1.0 / scale)
        assert trained_max[levels.COEF_SPEED] == pytest.approx(scale)
        low_fraction, high_fraction = levels.CALM.windows[levels.COEF_SPEED]
        low, high = levels.level_bounds(drive_env, levels.CALM)
        span = scale - 1.0 / scale
        assert low[levels.COEF_SPEED] == pytest.approx(1.0 / scale + low_fraction * span)
        assert high[levels.COEF_SPEED] == pytest.approx(1.0 / scale + high_fraction * span)
    finally:
        drive_env.close()


@requires_checkpoint
@pytest.mark.parametrize("log_sampling", [False, True])
def test_every_preset_lies_inside_the_trained_window(log_sampling):
    drive_env = make_env({"reward_log_sampling": log_sampling})
    try:
        trained_min, trained_max = drive_env.get_reward_coef_limits()
        for level in levels.LEVELS:
            low, high = levels.level_bounds(drive_env, level)
            assert np.all(low >= trained_min) and np.all(high <= trained_max), level.name
            assert np.all(low <= high), level.name
            assert levels.apply_level(drive_env, level) == drive_env.num_agents
    finally:
        drive_env.close()


@requires_checkpoint
def test_human_car_coefficients_never_touched(env):
    env.select_human_agent(HUMAN_AGENT_IDX)
    coefs_before, active_slot, _speeds = agent_frame(env)
    assert active_slot[HUMAN_AGENT_IDX] != INACTIVE_SLOT
    human_before = coefs_before[HUMAN_AGENT_IDX].copy()
    for level in levels.LEVELS:
        cars_changed = levels.apply_level(env, level)
        assert cars_changed == env.num_agents - 1
        coefs_after, _slot, _speeds = agent_frame(env)
        np.testing.assert_array_equal(coefs_after[HUMAN_AGENT_IDX], human_before)


@requires_checkpoint
def test_apply_level_returns_the_number_of_cars_changed(env):
    coefs_before, active_slot, _speeds = agent_frame(env)
    cars_changed = levels.apply_level(env, levels.CALM)
    coefs_after, _slot, _speeds = agent_frame(env)
    actually_changed = np.any(coefs_after != coefs_before, axis=1) & (active_slot != INACTIVE_SLOT)
    assert cars_changed == int(actually_changed.sum()) == env.num_agents


@requires_checkpoint
def test_cycling_levels_returns_to_the_training_window(env):
    trained_min, trained_max = env.get_reward_coef_limits()
    for level in (levels.CALM, levels.AGGRESSIVE, levels.NORMAL):
        levels.apply_level(env, level)
    low, high = levels.level_bounds(env, levels.NORMAL)
    np.testing.assert_array_equal(low, trained_min)
    np.testing.assert_array_equal(high, trained_max)
    samples = []
    for _round in range(RESAMPLE_ROUNDS):
        env.resample_reward_coefs()
        samples.append(policy_car_coefs(env))
    samples = np.concatenate(samples)
    span = trained_max - trained_min
    for coef_idx in WINDOWED_COEFS:
        fractions = (samples[:, coef_idx] - trained_min[coef_idx]) / span[coef_idx]
        assert fractions.min() < FULL_WINDOW_LOW_FRACTION, coef_idx
        assert fractions.max() > FULL_WINDOW_HIGH_FRACTION, coef_idx


@requires_checkpoint
def test_normal_after_a_level_restores_the_training_speed_distribution(env):
    levels.apply_level(env, levels.CALM)
    levels.apply_level(env, levels.NORMAL)
    share_below_pivot = float(np.mean(sampled_speed_coefs(env) < SPEED_COEF_PIVOT))
    assert share_below_pivot == pytest.approx(TRAINING_SPEED_SHARE_BELOW_PIVOT, abs=SPEED_SHARE_TOLERANCE)


@requires_checkpoint
def test_untouched_env_samples_the_training_speed_distribution(env):
    share_below_pivot = float(np.mean(sampled_speed_coefs(env) < SPEED_COEF_PIVOT))
    assert share_below_pivot == pytest.approx(TRAINING_SPEED_SHARE_BELOW_PIVOT, abs=SPEED_SHARE_TOLERANCE)


def stepped_reward_targets(level):
    drive_env = make_env()
    try:
        levels.apply_level(drive_env, level)
        actions = np.zeros_like(drive_env.actions)
        observations, *_rest = drive_env.step(actions)
        coefs, active_slot, _speeds = agent_frame(drive_env)
        coefs_by_slot = np.empty((drive_env.num_agents, binding.NUM_REWARD_COEFS), dtype=np.float32)
        is_active = active_slot != INACTIVE_SLOT
        coefs_by_slot[active_slot[is_active]] = coefs[is_active]
        trained_min, trained_max = drive_env.get_reward_coef_limits()
        return reward_target_slice(drive_env, observations).copy(), coefs_by_slot, trained_min, trained_max
    finally:
        drive_env.close()


@requires_checkpoint
def test_level_moves_the_conditioning_slice_and_round_trips():
    calm_targets, calm_coefs, trained_min, trained_max = stepped_reward_targets(levels.CALM)
    aggressive_targets, aggressive_coefs, _min, _max = stepped_reward_targets(levels.AGGRESSIVE)
    for coef_idx in WINDOWED_COEFS:
        calm_mean = calm_targets[:, coef_idx].mean()
        aggressive_mean = aggressive_targets[:, coef_idx].mean()
        assert abs(calm_mean - aggressive_mean) > MIN_CONDITIONING_GAP, coef_idx
    assert calm_targets[:, levels.COEF_COLLISION].mean() > aggressive_targets[:, levels.COEF_COLLISION].mean()
    assert calm_targets[:, levels.COEF_SPEED].mean() < aggressive_targets[:, levels.COEF_SPEED].mean()
    span = trained_max - trained_min
    for targets, coefs in ((calm_targets, calm_coefs), (aggressive_targets, aggressive_coefs)):
        decoded = trained_min + (targets + 1.0) / NORMALIZED_SPAN * span
        for coef_idx in WINDOWED_COEFS:
            np.testing.assert_allclose(
                decoded[:, coef_idx],
                coefs[:, coef_idx],
                rtol=ROUND_TRIP_RELATIVE_TOLERANCE,
                atol=ROUND_TRIP_RELATIVE_TOLERANCE * span[coef_idx],
            )


def mean_traffic_speed_mps(level):
    config, _ = policy_host.load_checkpoint_config(CHECKPOINT_PATH)
    drive_env = make_env()
    try:
        traffic = policy_host.FrozenTraffic(CHECKPOINT_PATH, drive_env, config)
        levels.apply_level(drive_env, level)
        observations = drive_env.observations
        speeds = []
        for _step in range(POLICY_STEP_COUNT):
            observations, *_rest = drive_env.step(traffic.act(observations))
            _coefs, active_slot, speed_mps = agent_frame(drive_env)
            speeds.append(speed_mps[active_slot != INACTIVE_SLOT].mean())
        return float(np.mean(speeds))
    finally:
        drive_env.close()


@requires_checkpoint
def test_aggressive_traffic_drives_faster_than_calm_traffic():
    pytest.importorskip("torch")
    calm_speed_mps = mean_traffic_speed_mps(levels.CALM)
    aggressive_speed_mps = mean_traffic_speed_mps(levels.AGGRESSIVE)
    assert aggressive_speed_mps - calm_speed_mps > MIN_MEAN_SPEED_GAP_MPS


@requires_checkpoint
def test_explicit_full_window_keeps_the_training_speed_distribution(env):
    full_level = levels.Level(
        name="full",
        description="every coefficient at its whole trained range",
        windows={coef_idx: FULL_WINDOW for coef_idx in range(binding.NUM_REWARD_COEFS)},
    )
    levels.apply_level(env, levels.AGGRESSIVE)
    levels.apply_level(env, full_level)
    share_below_pivot = float(np.mean(sampled_speed_coefs(env) < SPEED_COEF_PIVOT))
    assert share_below_pivot == pytest.approx(TRAINING_SPEED_SHARE_BELOW_PIVOT, abs=SPEED_SHARE_TOLERANCE)


@requires_checkpoint
def test_level_narrowing_a_fixed_profile_coefficient_is_rejected(env):
    levels.apply_level(env, levels.CALM)
    calm_low, calm_high = levels.level_bounds(env, levels.CALM)
    punchy = levels.Level(
        name="punchy", description="narrows a coefficient the sampler never reads",
        windows={FIXED_PROFILE_COEF_IDX: NARROW_WINDOW},
    )
    with pytest.raises(ValueError):
        levels.apply_level(env, punchy)
    env.resample_reward_coefs()
    coefs = policy_car_coefs(env)
    for coef_idx in levels.CALM.windows:
        assert np.all(coefs[:, coef_idx] >= calm_low[coef_idx]), coef_idx
        assert np.all(coefs[:, coef_idx] <= calm_high[coef_idx]), coef_idx


@requires_checkpoint
def test_apply_level_without_randomization_mutates_nothing():
    drive_env = make_env({"reward_randomization": False})
    try:
        coefs_before, _slot, _speeds = agent_frame(drive_env)
        with pytest.raises(ValueError):
            levels.apply_level(drive_env, levels.AGGRESSIVE)
        coefs_after, _slot, _speeds = agent_frame(drive_env)
        np.testing.assert_array_equal(coefs_after, coefs_before)
    finally:
        drive_env.close()


def strided_view(values):
    backing = np.zeros(2 * binding.NUM_REWARD_COEFS, dtype=np.float32)
    backing[::2] = values
    return backing[::2]


def read_only_copy(values):
    frozen = np.array(values, dtype=np.float32)
    frozen.flags.writeable = False
    return frozen


@requires_checkpoint
@pytest.mark.parametrize("make_bad_array", [strided_view, read_only_copy, lambda values: values[::-1]])
def test_coef_bindings_reject_unsafe_arrays(env, make_bad_array):
    trained_min, trained_max = env.get_reward_coef_limits()
    with pytest.raises(ValueError):
        binding.vec_set_reward_coef_bounds(env.c_envs, make_bad_array(trained_min), trained_max)
    with pytest.raises(ValueError):
        binding.vec_set_reward_coef_bounds(env.c_envs, trained_min, make_bad_array(trained_max))
    with pytest.raises(ValueError):
        binding.vec_get_reward_coef_limits(env.c_envs, make_bad_array(trained_min), trained_max.copy())
    with pytest.raises(ValueError):
        binding.vec_get_reward_coef_limits(env.c_envs, trained_min.copy(), make_bad_array(trained_max))
