"""nuplan_tracker: the planner's forecast rollout. Slot assignment, what the policy drives, what moves at its observed
velocity, the no-reverse hold and the trajectory handed to nuPlan's tracker."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

pytest.importorskip("nuplan")
pytest.importorskip("carl_nuplan")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from pufferlib.ocean.cosim.nuplan.planner import (
    DUMMY_ROLLOUT_EGO_ACTION,
    EGO_DYNAMICS_NUPLAN_TRACKER,
    EGO_OBS_ACCEL_LONG_IDX,
    EGO_OBS_STEERING_IDX,
    PufferDrivePlanner,
    startup_jerk_action_bounds,
)
from pufferlib.ocean.cosim.nuplan_bridge import LIGHT_STATE_GREEN, LIGHT_STATE_RED, LIGHT_STATE_YELLOW
from pufferlib.ocean.drive import binding


POLICY_VEHICLES = 2
OBS_DIM = 12
OBS_NORM_SPEED_MPS = 50.0
STEP_SECONDS = 0.3
ROLLOUT_STEPS = 4
EGO_STEP_M = 1.5  # the scripted env moves the ego this far along x every step


class _FakeEnv:
    """Records the co-sim calls; step() moves the ego along x and writes its speed, accel and steering obs."""

    def __init__(self, agent_count=1 + POLICY_VEHICLES):
        self.num_agents = agent_count
        self.single_action_space = SimpleNamespace(shape=(2,))
        self.obs_norm_speed_mps = OBS_NORM_SPEED_MPS
        self.obs_traffic_light_yellow_remaining = False
        self.observations = np.zeros((agent_count, OBS_DIM), np.float32)
        self.x = np.zeros(agent_count, np.float32)
        self.lane_goal_counts = {}
        self.state_calls = []
        self.light_calls = []
        self.step_actions = []
        self.recompute_count = 0
        self.step_speed_mps = 5.0

    def set_agent_states(
        self, idx, x, y, z, heading, vx, vy, yaw_rate, accel_long, seconds_stopped=None, steering=None
    ):
        call = {"idx": idx, "x": x, "y": y, "heading": heading, "vx": vx, "vy": vy}
        call = {name: np.array(values) for name, values in call.items()}
        call["steering"] = None if steering is None else np.array(steering)
        self.state_calls.append(call)

    def set_agent_sizes(self, idx, length, width):
        pass

    def set_agent_lane_goals(self, idx, spacing_m):
        return np.array([self.lane_goal_counts.get(int(agent_idx), 3) for agent_idx in idx], np.int32)

    def set_traffic_light_states(self, states, yellow_remaining_s=None):
        self.light_calls.append(
            (np.array(states), None if yellow_remaining_s is None else np.array(yellow_remaining_s))
        )

    def recompute_observations(self):
        self.recompute_count += 1
        return self.observations

    def step(self, actions):
        self.step_actions.append(np.array(actions))
        self.x[0] += EGO_STEP_M
        self.observations[0, 0] = self.step_speed_mps / OBS_NORM_SPEED_MPS
        self.observations[0, EGO_OBS_ACCEL_LONG_IDX] = 1.0 / binding.ACCEL_LONG_NORM
        self.observations[0, EGO_OBS_STEERING_IDX] = 0.1 / binding.STEERING_LIMIT

    def get_global_agent_state(self):
        zeros = np.zeros(self.num_agents, np.float32)
        return {"x": self.x.copy(), "y": zeros.copy(), "z": zeros.copy(), "heading": zeros.copy()}


class _FakeGoalWindow:
    def __init__(self, push_on_sync=()):
        self.sync_poses = []
        self.push_on_sync = set(push_on_sync)
        self.restored = None

    def sync(self, x, y, heading):
        self.sync_poses.append(x)
        return len(self.sync_poses) in self.push_on_sync

    def snapshot(self):
        return ("snapshot after sync", len(self.sync_poses))

    def restore(self, snapshot):
        self.restored = snapshot


def _planner(env, **state):
    planner = PufferDrivePlanner.__new__(PufferDrivePlanner)
    planner.__dict__.update(
        {
            "_env": env,
            "_policy": None,
            "_obs_replay": None,
            "_num_agents": 6,
            "_goal_spacing": 20.0,
            "_rollout_policy_vehicles": POLICY_VEHICLES,
            "_rollout_step_seconds": STEP_SECONDS,
            "_rollout_steps": ROLLOUT_STEPS,
            "_rollout_allow_reverse": False,
            "_rollout_policy_rows": np.zeros(1, np.int64),
            "_forecast_objects": None,
            "_planning_step_s": 0.1,
            "_startup_action_bounds": None,
            "_startup_jerk_cap_seconds": 1.5,
            "_num_traffic": 0,
            "_last_light_states": None,
            "_last_yellow_remaining_s": None,
            "_transform": SimpleNamespace(bin_to_loc=lambda x, y: (x + 1000.0, y + 2000.0)),
            "_goal_window": _FakeGoalWindow(),
        }
    )
    planner.__dict__.update(state)
    return planner


def _ego_state(time_us=5_000_000):
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(1000.0, 2000.0, 0.0),
        rear_axle_velocity_2d=StateVector2D(4.0, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.5, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(time_us),
        vehicle_parameters=get_pacifica_parameters(),
    )


def _track(type_name, x, speed_mps):
    return SimpleNamespace(
        tracked_object_type=SimpleNamespace(name=type_name),
        track_token=type_name,
        center=SimpleNamespace(x=x, y=2000.0, heading=0.0),
        velocity=SimpleNamespace(x=speed_mps, y=0.0),
        box=SimpleNamespace(length=2.0, width=1.0),
    )


def test_sync_lets_the_policy_drive_vehicles_and_bicycles_but_not_pedestrians():
    env = _FakeEnv()
    planner = _planner(
        env,
        _ego_dynamics=EGO_DYNAMICS_NUPLAN_TRACKER,
        _ego_stopped_s=0.0,
        _partner_stopped_s={},
        _initialization=SimpleNamespace(map_api=None),
        _static_on_lane={},
        _vru_size_pad_m=0.0,
        _partner_split_max_length_m=0.0,
        _transform=SimpleNamespace(loc_to_bin=lambda x, y: (x - 1000.0, y - 2000.0)),
    )
    tracks = [_track("PEDESTRIAN", 1003.0, 1.5), _track("BICYCLE", 1010.0, 4.0), _track("VEHICLE", 1020.0, 8.0)]
    planner._sync(_ego_state(), SimpleNamespace(tracked_objects=tracks), None)
    partners = env.state_calls[1]
    slot_at_x = dict(zip(partners["x"].tolist(), partners["idx"].tolist()))
    assert slot_at_x == {3.0: 3, 10.0: 1, 20.0: 2}  # the pedestrian is nearest, yet it takes the first static slot
    assert planner._rollout_policy_rows.tolist() == [0, 1, 2]
    assert planner._forecast_objects[0].tolist() == [3]  # the pedestrian walks on at its observed velocity


def test_nearest_vehicles_get_the_policy_slots_and_the_rest_static_slots():
    planner = _planner(_FakeEnv())
    x = np.array([30.0, 2.0, 10.0, 5.0, 60.0], np.float32)
    eligible = np.array([True, False, True, True, True])  # row 1: a pedestrian next to the ego
    idx = planner._rollout_partner_slots(x, np.zeros(5, np.float32), eligible, 0.0, 0.0)
    assert idx.tolist() == [3, 4, 1, 2, 5]  # vehicles at 10 m and 5 m take slots 1 and 2, statics start at 3


def test_policy_drives_only_vehicles_with_a_lane_and_statics_keep_their_velocity():
    env = _FakeEnv()
    env.lane_goal_counts = {2: 0}  # the vehicle in slot 2 has no lane to follow
    planner = _planner(env)
    idx = np.array([1, 3, 2, 4], np.int32)
    x = np.array([10.0, 20.0, 30.0, 40.0], np.float32)
    zeros = np.zeros(4, np.float32)
    vx = np.array([5.0, 1.5, 0.0, 0.0], np.float32)  # slot 3: a walking pedestrian, slot 4: a cone
    planner._arm_rollout_partners(idx, x, zeros, zeros, zeros, vx, zeros)
    assert planner._rollout_policy_rows.tolist() == [0, 1]
    forecast_idx, forecast_x, _, _, _, forecast_vx, _ = planner._forecast_objects
    assert forecast_idx.tolist() == [3] and forecast_x.tolist() == [20.0] and forecast_vx.tolist() == [1.5]


def test_forecast_moves_statics_at_their_velocity_and_lets_a_yellow_run_out():
    env = _FakeEnv()
    # one pedestrian in static slot 4 at (20, 1), walking at (1.5, -2) m/s
    pedestrian = [np.array([value], np.float32) for value in (20.0, 1.0, 0.0, 0.5, 1.5, -2.0)]
    planner = _planner(
        env,
        _forecast_objects=(np.array([4], np.int32), *pedestrian),
        _num_traffic=2,
        _last_light_states=np.array([LIGHT_STATE_GREEN, LIGHT_STATE_YELLOW], np.int32),
        _last_yellow_remaining_s=np.array([0.0, 0.5], np.float32),
    )
    planner._stamp_forecast(0.6)
    stamped = env.state_calls[-1]
    assert stamped["idx"].tolist() == [4]
    np.testing.assert_allclose(stamped["x"], [20.9], atol=1e-5)
    np.testing.assert_allclose(stamped["y"], [-0.2], atol=1e-5)
    np.testing.assert_allclose(stamped["vx"], [1.5])
    states, remaining = env.light_calls[-1]
    assert states.tolist() == [LIGHT_STATE_GREEN, LIGHT_STATE_RED]
    assert remaining is None  # the checkpoint has no yellow countdown column

    env.obs_traffic_light_yellow_remaining = True
    planner._stamp_forecast(0.3)
    states, remaining = env.light_calls[-1]
    assert states.tolist() == [LIGHT_STATE_GREEN, LIGHT_STATE_YELLOW]
    np.testing.assert_allclose(remaining, [0.0, 0.2], atol=1e-6)


def test_a_policy_agent_that_would_back_up_is_held_where_it_stood():
    env = _FakeEnv(agent_count=4)
    planner = _planner(env)
    obs = np.zeros((4, OBS_DIM), np.float32)
    obs[:, 0] = [0.02, -0.01, -0.03, 0.1]  # agent 1: policy-driven, backing up; agent 2: coasting backwards
    obs[1, EGO_OBS_STEERING_IDX] = 0.3 / binding.STEERING_LIMIT
    before = {key: np.array([1.0, 2.0, 3.0, 4.0], np.float32) for key in ("x", "y", "z", "heading")}
    after = {key: values + 0.5 for key, values in before.items()}
    assert planner._hold_reversing_agents(np.array([0, 1, 3]), obs, before, after) is True
    held = env.state_calls[-1]
    assert held["idx"].tolist() == [1]
    assert held["x"].tolist() == [2.0] and held["heading"].tolist() == [2.0]
    assert held["vx"].tolist() == [0.0] and held["vy"].tolist() == [0.0]
    np.testing.assert_allclose(held["steering"], [0.3], atol=1e-6)
    assert after["x"].tolist() == [1.5, 2.0, 3.5, 4.5]  # only the held agent is back at its old pose

    calls = len(env.state_calls)
    obs[1, 0] = 0.0
    assert planner._hold_reversing_agents(np.array([0, 1, 3]), obs, before, after) is False
    assert len(env.state_calls) == calls


def test_rollout_returns_the_ego_states_as_a_trajectory_and_undoes_its_goal_progress():
    env = _FakeEnv()
    goal_window = _FakeGoalWindow(push_on_sync={3})
    planner = _planner(env, _goal_window=goal_window, _rollout_policy_rows=np.array([0, 2]))
    ego_state = _ego_state()

    trajectory = planner._rollout_trajectory(ego_state, iteration_index=0)

    states = trajectory.get_sampled_trajectory()
    assert len(states) == 1 + ROLLOUT_STEPS
    assert states[0] is ego_state
    step_us = round(STEP_SECONDS * 1e6)
    assert [s.time_point.time_us for s in states] == [5_000_000 + k * step_us for k in range(1 + ROLLOUT_STEPS)]
    for step_idx, state in enumerate(states[1:], start=1):
        assert state.center.x == pytest.approx(1000.0 + EGO_STEP_M * step_idx)  # bin frame -> nuPlan frame
        assert state.center.y == pytest.approx(2000.0)
        assert state.dynamic_car_state.center_velocity_2d.x == pytest.approx(5.0, abs=1e-5)
        assert state.dynamic_car_state.center_acceleration_2d.x == pytest.approx(1.0, abs=1e-5)
        assert state.tire_steering_angle == pytest.approx(0.1, abs=1e-6)
    assert trajectory.end_time.time_us - trajectory.start_time.time_us == ROLLOUT_STEPS * step_us

    assert len(env.step_actions) == ROLLOUT_STEPS
    for actions in env.step_actions:  # no checkpoint: the ego gets the wiring-test action, everything else coasts
        assert actions[0].tolist() == list(DUMMY_ROLLOUT_EGO_ACTION)
        assert not actions[1:].any()
    assert goal_window.sync_poses == pytest.approx([EGO_STEP_M * k for k in range(1, 1 + ROLLOUT_STEPS)])
    assert goal_window.restored == ("snapshot after sync", 1)  # the first step's progress is the one that is kept
    assert env.recompute_count == 2  # once before the rollout, once after the goal window pushed on step 3


def test_rollout_caps_the_ego_jerk_only_inside_the_startup_window():
    env = _FakeEnv()
    bounds = startup_jerk_action_bounds(2.0, 2.0)
    planner = _planner(env, _startup_action_bounds=bounds, _startup_jerk_cap_seconds=0.5)
    planner._rollout_trajectory(_ego_state(), iteration_index=0)
    ego_jerk_actions = [float(actions[0, 0]) for actions in env.step_actions]
    assert ego_jerk_actions == pytest.approx([bounds[1], bounds[1], 1.0, 1.0])  # steps at 0.0 and 0.3 s are capped

    env.step_actions.clear()
    planner._rollout_trajectory(_ego_state(), iteration_index=4)  # 0.4 s into the scenario: only the first step
    ego_jerk_actions = [float(actions[0, 0]) for actions in env.step_actions]
    assert ego_jerk_actions == pytest.approx([bounds[1], 1.0, 1.0, 1.0])


class _FakeObsReplay:
    def __init__(self):
        self.captures = []

    def policy_outputs(self, obs_tensor, logits, value, entropy):
        return {"obs_rows": obs_tensor.shape[0], "logit_rows": logits.shape[0], "value": value, "entropy": entropy}

    def capture(self, obs, actions, aux, action_index=None):
        # like ObsReplayCapture: the action array is kept by reference until the replay is written
        self.captures.append((np.array(obs), np.asarray(actions, dtype=np.float32), aux, action_index))


@pytest.mark.parametrize("policy_rows", [[0], [0, 2]])
def test_obs_replay_gets_the_ego_row_of_the_first_rollout_step(policy_rows):
    torch = pytest.importorskip("torch")
    env = _FakeEnv()
    replay = _FakeObsReplay()
    planner = _planner(
        env,
        _policy=SimpleNamespace(is_continuous=False),
        _obs_replay=replay,
        _rollout_policy_rows=np.array(policy_rows),
    )
    row_count = len(policy_rows)

    calls = []

    def policy_forward(obs):
        # shapes as sample_logits returns them: a one-row batch has its entropy squeezed to 0-dim
        entropy = torch.full((row_count,), 0.25).squeeze(0) if row_count == 1 else torch.full((row_count,), 0.25)
        first_call = not calls
        calls.append(len(obs))
        act = np.tile(np.array([[0.5, -0.5]] if first_call else [[-1.0, 1.0]], np.float32), (row_count, 1))
        action = torch.arange(7, 7 + row_count)
        return act, torch.as_tensor(obs), torch.zeros(row_count, 12), torch.ones(row_count, 1), entropy, action

    planner._policy_forward = policy_forward
    planner._rollout_trajectory(_ego_state(), iteration_index=0)

    assert len(replay.captures) == 1  # only the step taken from the real state is a replay frame
    obs, actions, aux, action_index = replay.captures[0]
    assert obs.shape == (1, OBS_DIM)
    assert actions.tolist() == [[0.5, -0.5]]  # the first step's action, not what later steps wrote into the buffer
    assert calls == [row_count] * ROLLOUT_STEPS
    assert aux["obs_rows"] == 1 and aux["logit_rows"] == 1
    assert tuple(aux["value"].shape) == (1, 1) and tuple(aux["entropy"].shape) == (1,)
    assert action_index.tolist() == [7]


def test_tracker_mode_rejects_more_policy_vehicles_than_partners():
    with pytest.raises(ValueError, match="rollout_policy_vehicles"):
        PufferDrivePlanner("dummy", "/tmp", num_agents=8, ego_dynamics="nuplan_tracker", rollout_policy_vehicles=8)
    with pytest.raises(ValueError, match="rollout_step_seconds"):
        PufferDrivePlanner("dummy", "/tmp", ego_dynamics="nuplan_tracker", rollout_step_seconds=0.0)
    planner = PufferDrivePlanner(
        "dummy", "/tmp", num_agents=8, ego_dynamics="nuplan_tracker", rollout_policy_vehicles=7
    )
    assert planner._rollout_policy_vehicles == 7
