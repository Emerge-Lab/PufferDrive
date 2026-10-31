"""What the sim exports for the wheel demo's HUD and CARLA lights: get_human_state's lane limit,
tolerance and goal, and get_traffic_light_states with its input validation."""

import os
from pathlib import Path

import numpy as np
import pytest

from pufferlib.ocean.drive import binding
from pufferlib.wheel_demo import policy_host

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_PATH = os.path.join(REPO_ROOT, "weights", "k_scaled_0050_1000", "final_model.pt")
MAP_FILE = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "carla", "opendrive__Town01.bin")

AGENT_COUNT = 8
SEED = 3
TOLERANCE_MPS = 1.25
BIN_LANE_TYPE_MAX = 9
GOAL_ON_LANE_TOLERANCE_M = 1.0
STEP_COUNT = 30
UNKNOWN_LANE_SPEED_LIMIT_MPS = 15.0
LIGHT_STATES = (
    binding.TRAFFIC_CONTROL_STATE_RED,
    binding.TRAFFIC_CONTROL_STATE_YELLOW,
    binding.TRAFFIC_CONTROL_STATE_GREEN,
    binding.TRAFFIC_CONTROL_STATE_OFF,
)

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(CHECKPOINT_PATH) and os.path.isfile(MAP_FILE)),
    reason="demo checkpoint or Town01 map not present",
)


def read_town():
    from data_utils import mirror_map_bin

    return mirror_map_bin.read_bin(Path(MAP_FILE))


@pytest.fixture(scope="module")
def town():
    return read_town()


@pytest.fixture
def env():
    config, _ = policy_host.load_checkpoint_config(CHECKPOINT_PATH)
    drive_env = policy_host.build_demo_env(config, MAP_FILE, AGENT_COUNT, {"overspeed_tolerance_mps": TOLERANCE_MPS})
    drive_env.reset(seed=SEED)
    yield drive_env
    drive_env.close()


def lane_segments(town):
    starts, ends = [], []
    for road in town["roads"]:
        if road["type"] > BIN_LANE_TYPE_MAX or len(road["x"]) < 2:
            continue
        points = np.column_stack([road["x"], road["y"]]).astype(np.float64)
        starts.append(points[:-1])
        ends.append(points[1:])
    return np.vstack(starts), np.vstack(ends)


def distance_to_lanes_m(segments, x, y):
    starts, ends = segments
    direction = ends - starts
    length_sq = np.maximum(np.sum(direction**2, axis=1), 1e-12)
    along = np.clip(((x - starts[:, 0]) * direction[:, 0] + (y - starts[:, 1]) * direction[:, 1]) / length_sq, 0.0, 1.0)
    nearest = starts + along[:, None] * direction
    return float(np.min(np.hypot(nearest[:, 0] - x, nearest[:, 1] - y)))


def lane_speed_limits(town):
    return {float(np.float32(road["speed_limit"])) for road in town["roads"] if road["type"] <= BIN_LANE_TYPE_MAX}


def element_count(town):
    return len(town["traffic"])


def test_human_state_types_for_the_new_fields(env):
    env.select_human_agent(0)
    human = env.get_human_state()
    assert isinstance(human["goal_idx"], int)
    for key in ("lane_speed_limit_mps", "overspeed_tolerance_mps", "goal_x", "goal_y"):
        assert isinstance(human[key], float)
        assert np.isfinite(human[key])


def test_tolerance_is_the_configured_one(env):
    env.select_human_agent(0)
    assert env.get_human_state()["overspeed_tolerance_mps"] == TOLERANCE_MPS


def test_lane_limit_is_one_the_map_posts_or_the_unknown_fallback(env, town):
    allowed = lane_speed_limits(town) | {UNKNOWN_LANE_SPEED_LIMIT_MPS}
    no_action = env.action_space.sample() * 0
    for _ in range(STEP_COUNT):
        env.step(no_action)
        for agent_idx in range(AGENT_COUNT):
            env.select_human_agent(agent_idx)
            assert env.get_human_state()["lane_speed_limit_mps"] in allowed


def test_goal_is_on_a_lane_in_the_same_frame_as_the_car(env, town):
    segments = lane_segments(town)
    state = env.get_global_agent_state()
    for agent_idx in range(AGENT_COUNT):
        env.select_human_agent(agent_idx)
        human = env.get_human_state()
        assert human["sim_x"] == pytest.approx(float(state["x"][agent_idx]), abs=1e-3)
        assert human["sim_y"] == pytest.approx(float(state["y"][agent_idx]), abs=1e-3)
        assert distance_to_lanes_m(segments, human["goal_x"], human["goal_y"]) < GOAL_ON_LANE_TOLERANCE_M
        assert human["goal_idx"] >= 0


def test_the_human_state_follows_the_selected_car(env):
    env.select_human_agent(1)
    first = env.get_human_state()
    env.select_human_agent(2)
    second = env.get_human_state()
    assert (first["goal_x"], first["goal_y"]) != (second["goal_x"], second["goal_y"])


def test_light_states_fill_the_given_buffer_and_return_it(env, town):
    out = np.full(element_count(town), -1, dtype=np.int32)
    returned = env.get_traffic_light_states(out)
    assert returned is out
    assert set(out.tolist()) <= set(LIGHT_STATES)


def test_light_states_read_back_what_was_set(env, town):
    count = element_count(town)
    wanted = np.array([LIGHT_STATES[i % len(LIGHT_STATES)] for i in range(count)], dtype=np.int32)
    env.set_traffic_light_states(wanted)
    out = np.zeros(count, dtype=np.int32)
    env.get_traffic_light_states(out)
    np.testing.assert_array_equal(out, wanted)


@pytest.mark.parametrize("size_delta", [-1, 1])
def test_light_states_reject_the_wrong_size(env, town, size_delta):
    with pytest.raises(ValueError):
        env.get_traffic_light_states(np.zeros(element_count(town) + size_delta, dtype=np.int32))


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.int16, np.uint32])
def test_light_states_reject_other_dtypes(env, town, dtype):
    with pytest.raises(TypeError):
        env.get_traffic_light_states(np.zeros(element_count(town), dtype=dtype))


def test_light_states_reject_a_non_contiguous_view(env, town):
    strided = np.zeros(2 * element_count(town), dtype=np.int32)[::2]
    with pytest.raises(TypeError):
        env.get_traffic_light_states(strided)


def test_light_states_reject_a_list(env, town):
    with pytest.raises(TypeError):
        env.get_traffic_light_states([0] * element_count(town))


def test_light_states_reject_a_read_only_buffer(env, town):
    frozen = np.zeros(element_count(town), dtype=np.int32)
    frozen.setflags(write=False)
    with pytest.raises((TypeError, ValueError)):
        env.get_traffic_light_states(frozen)
    assert not frozen.any()
