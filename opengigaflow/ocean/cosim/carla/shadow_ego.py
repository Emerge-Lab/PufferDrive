"""Shadow OpenGigaFlow env + policy driving one CARLA leaderboard ego from ground truth (CARLA frame, plain data).

Environment variables:
  COSIM_DEVICE=cpu                    torch device of the policy
  COSIM_PEDAL_YELLOW_MARGIN=1         the policy sees the remaining yellow less the gear shift time
  COSIM_MAX_SPEED_MPS                 ego speed cap (C_vel = cap / base_max_speed_mps) inside the trained C_vel range
  COSIM_ZERO_PARTNER_STOPPED_TIME=0   1: vehicle partners' seconds_stopped held at 0 (walkers keep theirs)
  COSIM_VRU_SIZE_PAD_M=0.0            added to the length and width of every walker and two-wheeler partner box
  COSIM_OBS_SLOTS_PARTNERS_N          partner obs slots (default: the checkpoint's; max-pooled, so any count transfers)
  COSIM_TELEMETRY=/dir                per-route CSV of the shadow env's and the simulator's infraction flags
  COSIM_OBS_HTML=/dir                 per-route opengigaflow.viz replay of the exact obs and policy outputs
  COSIM_OBS_HTML_RENDER=1             0: save only the .replay.zlib (scripts/eval/render_carla_obs_html.py)
  COSIM_OBS_HTML_MAX_STEPS=12000      frames kept per route replay
  COSIM_WORLD_LOG=/dir                per-route bin-frame world state (scripts/eval/analyze_carla_cosim.py)
"""

import json
import math
import os
from pathlib import Path

import numpy as np

import data_utils.mirror_map_bin as mbin
from opengigaflow.ocean.cosim import carla_bridge as cb
from opengigaflow.ocean.cosim.arch import (
    EGO_OBS_ACCEL_LONG_IDX,
    EGO_OBS_SPEED_IDX,
    EGO_OBS_STEERING_IDX,
    FAR_AWAY,
    SCENARIO_LENGTH_MARGIN_STEPS,
    checkpoint_config_path,
    load_checkpoint_config,
    shadow_env_kwargs,
)
from opengigaflow.ocean.cosim.carla.controller import PedalMap, pedal_gains_for
from opengigaflow.ocean.cosim.goals import RouteGoalWindow, route_goals_from_target_points
from opengigaflow.ocean.cosim.light_timing import YellowCountdown
from opengigaflow.ocean.drive import binding
from opengigaflow.ocean.drive.drive import Drive


# CARLA-frame actor rows (yaw in degrees, angular velocity in deg/s); a partner row appends box, flags and actor id
ACTOR_STATE_COLUMNS = 10
PARTNER_ROW_COLUMNS = 17
(
    STATE_X,
    STATE_Y,
    STATE_Z,
    STATE_YAW_DEG,
    STATE_VX,
    STATE_VY,
    STATE_VZ,
    STATE_ANGULAR_VELOCITY_Z,
    STATE_ACCEL_X,
    STATE_ACCEL_Y,
) = range(ACTOR_STATE_COLUMNS)
(
    PARTNER_BOX_Z,
    PARTNER_EXTENT_X,
    PARTNER_EXTENT_Y,
    PARTNER_EXTENT_Z,
    PARTNER_IS_WALKER,
    PARTNER_IS_TWO_WHEELER,
    PARTNER_ACTOR_ID,
) = range(ACTOR_STATE_COLUMNS, PARTNER_ROW_COLUMNS)

PARTNER_MAX_ABS_DZ_M = 20.0  # CARLA actors farther above/below the ego are hidden scenario props, not traffic
DEFAULT_BASE_MAX_SPEED_MPS = 20.0  # Drive() defaults, for checkpoints whose config predates the keys
DEFAULT_CONDITIONING_SPEED_SCALE = 1.5
MAX_POLICY_STEPS = 100_000  # shadow-env episode cap (~1.4 h at the 0.05 s CARLA tick); sizes the light state buffers
MIN_PARTNER_SIZE_M = 0.1
EGO_INFRACTION_METRICS = {"collision": 0, "offroad": 1, "red_light": 2, "stop_sign": 3}  # metrics_array indices
DRIVE_WHEELBASE_PER_LENGTH = 0.6  # must equal the jerk model's wheelbase / sim_length
STANDSTILL_SPEED_MPS = 0.1  # slower than this, rolling backwards included, the ego counts as standing
TELEMETRY_HEADER = (
    "step,current_speed,target_speed,ego_action,goal_cursor,goal_dist_m,"
    "near_light_dist_m,near_light_state,"
    "pd_infr_collision,pd_infr_offroad,pd_infr_red,pd_infr_stop,"
    "carla_infr_collision,carla_infr_offroad,carla_infr_red,carla_infr_stop\n"
)


def cosim_max_speed_mps(env_cfg):
    """COSIM_MAX_SPEED_MPS as a float inside the checkpoint's trained C_vel range, or None when unset."""
    raw = os.environ.get("COSIM_MAX_SPEED_MPS")
    if raw is None:
        return None
    max_speed_mps = float(raw)
    base_max_speed_mps = float(env_cfg.get("base_max_speed_mps") or DEFAULT_BASE_MAX_SPEED_MPS)
    scale = float(env_cfg.get("conditioning_speed_scale") or DEFAULT_CONDITIONING_SPEED_SCALE)
    low_mps, high_mps = base_max_speed_mps / scale, base_max_speed_mps * scale
    if not (low_mps <= max_speed_mps <= high_mps):
        raise ValueError(
            f"COSIM_MAX_SPEED_MPS={max_speed_mps} is outside the trained C_vel range "
            f"[{low_mps:.1f}, {high_mps:.1f}] m/s (base_max_speed_mps={base_max_speed_mps}, "
            f"conditioning_speed_scale={scale})"
        )
    return max_speed_mps


def cosim_obs_slots_partners_n():
    """COSIM_OBS_SLOTS_PARTNERS_N as a positive int, or None when unset/empty (the checkpoint's training value)."""
    raw = os.environ.get("COSIM_OBS_SLOTS_PARTNERS_N")
    if not raw:
        return None
    obs_slots_partners_n = int(raw)
    if obs_slots_partners_n < 1:
        raise ValueError(f"COSIM_OBS_SLOTS_PARTNERS_N must be >= 1, got {raw!r}")
    return obs_slots_partners_n


def env_flag(name, default="0"):
    raw = os.environ.get(name, default)
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1', got {raw!r}")
    return raw == "1"


def ego_speed_from_state(state):
    """CARLA's Vector3D.length() of the ego velocity (3-D, as the leaderboard criteria read it)."""
    return math.sqrt(state[STATE_VX] ** 2 + state[STATE_VY] ** 2 + state[STATE_VZ] ** 2)


class StepResult:
    """One policy step: `actions` (1, act_dim) float32, the ego command `control` = (steer, throttle, brake) and the
    shadow env's infraction flags `pd_flags` (None unless telemetry is written)."""

    def __init__(self, actions, control, pd_flags=None):
        self.actions = actions
        self.control = control
        self.pd_flags = pd_flags


class ShadowEgo:
    def __init__(self, path_to_conf_file, route_tag):
        self.route_tag = route_tag
        checkpoint = Path(path_to_conf_file).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"no checkpoint file at {checkpoint}")
        self.checkpoint = str(checkpoint)
        self.cfg = load_checkpoint_config(checkpoint_config_path(checkpoint))
        self.device = os.environ.get("COSIM_DEVICE", "cpu")
        env_cfg = self.cfg["env"]
        self.max_speed_mps = cosim_max_speed_mps(env_cfg)
        self.obs_slots_partners_n = cosim_obs_slots_partners_n()
        self.zero_partner_stopped_time = env_flag("COSIM_ZERO_PARTNER_STOPPED_TIME")
        self.pedal_yellow_margin = env_flag("COSIM_PEDAL_YELLOW_MARGIN", "1")
        self.vru_size_pad_m = float(os.environ.get("COSIM_VRU_SIZE_PAD_M", "0.0"))
        if not (self.vru_size_pad_m >= 0.0):
            raise ValueError(f"COSIM_VRU_SIZE_PAD_M must be >= 0, got {self.vru_size_pad_m}")
        # Shadow agent pool == the training per-env cap
        self.num_agents = int(env_cfg["max_agents_per_env"])

        self.telemetry_dir = os.environ.get("COSIM_TELEMETRY", None)
        self.telemetry_file = None
        self.obs_html_dir = os.environ.get("COSIM_OBS_HTML", None)
        self.world_log_dir = os.environ.get("COSIM_WORLD_LOG", None)
        self._world_log = {"ego": [], "partners": [], "lights": [], "light_yellow_s": []}
        self._last_partners = None
        self._partner_stopped_s = {}  # CARLA actor id -> seconds at standstill
        self._partner_stopped_speed_threshold = float(binding.AGENT_STOPPED_SPEED_THRESHOLD)
        self._steering_limit_rad = float(binding.STEERING_LIMIT)
        self._intent_accel_mps2 = 0.0  # the jerk integrator's own state, never re-read from CARLA
        self._intent_steering_rad = 0.0
        self._plant_speed_mps = 0.0  # along the car's forward axis, negative while it rolls backwards
        self._plant_gear = 0
        self._held = False  # braked to a stop and kept on the brake until the policy asks to move
        self._yellow_margin_s = 0.0  # the policy sees this much less yellow than CARLA has left
        self.obs_html_max_steps = int(os.environ.get("COSIM_OBS_HTML_MAX_STEPS", "12000"))
        self.obs_html_render = env_flag("COSIM_OBS_HTML_RENDER", "1")
        self._obs_html = None

        self.step_count = 0
        self.initialized = False
        self.policy = None
        self.env = None
        self.controller = None
        self.goal_window = None
        self.target_speed_mps = 0.0  # the shadow ego's post-step speed, held between policy steps
        self._ego_speed_carla = 0.0  # CARLA's own ego speed at the last sync (carla dynamics telemetry)
        self._stop_sign_lines = np.zeros((0, 6), np.float32)  # CARLA trigger volumes as bin-frame stop lines

    @property
    def partner_slots(self):
        return self.num_agents - 1

    def _load_policy_and_env(self, town_bin):
        arch = shadow_env_kwargs(
            self.cfg,
            overrides=dict(
                map_dir=town_bin,
                num_maps=1,
                # One policy agent (the ego); every other slot is a static partner streamed from CARLA.
                num_agents=1,
                min_agents_per_env=1,
                cosim_partner_slots=self.partner_slots,
                goal_source="external",
                dt=self.dt,
                scenario_length=MAX_POLICY_STEPS + SCENARIO_LENGTH_MARGIN_STEPS,
                resample_frequency=0,
                termination_mode=0,
                # enforcement off (flags still fire): in one endless episode a "stop" latch would freeze the ego
                collision_behavior="ignore",
                offroad_behavior="ignore",
                traffic_light_behavior="ignore",
                stop_sign_behavior="ignore",
                **({} if self.max_speed_mps is None else {"max_speed_mps": self.max_speed_mps}),
                **({} if self.obs_slots_partners_n is None else {"obs_slots_partners_n": self.obs_slots_partners_n}),
            ),
        )
        self.env = Drive(**arch)
        self.env.reset()

        import torch

        import opengigaflow.ocean.torch as drive_torch
        from opengigaflow.pytorch import load_policy_weights

        checkpoint = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        policy_cls = getattr(drive_torch, self.cfg.get("policy_name", "Drive"))
        policy = policy_cls(self.env, **self.cfg["policy"]).to(self.device)
        self.policy = load_policy_weights(policy, checkpoint)
        self.policy.eval()
        print(
            f"[opengigaflow_agent] loaded policy from {self.checkpoint} "
            f"(obs partner slots {self.env.obs_slots_partners_n})"
        )

    def init_route(self, route):
        """Build the shadow env for the route's town from route = {town, tick_dt, ego: {extent, box_z, blueprint,
        state: ACTOR_STATE}, sparse_plan, dense_plan, driving_waypoints, lights, stop_signs} (carla_bridge layouts)."""
        town = route["town"]
        self.tick_dt = float(route["tick_dt"])
        self.dt = self.tick_dt  # the policy runs every CARLA tick, like the leaderboard's sensor agents
        ego = route["ego"]
        self.ego_extent = [float(v) for v in ego["extent"]]
        self.ego_box_z = float(ego["box_z"])

        self.town_bin = cb.bin_path_for_town(town)
        self._load_policy_and_env(self.town_bin)

        self.transform = cb.CarlaTransform(town)
        offset, z_offset, residual_before, residual_after = cb.calibrate_town_offset(
            route["driving_waypoints"], self.transform, self.town_bin
        )
        print(
            f"[opengigaflow_agent] bin offset calibrated against CARLA lanes: shift "
            f"({offset[0] - self.transform.tx:+.2f}, {offset[1] - self.transform.ty:+.2f}) m, z {z_offset:+.2f} m, "
            f"median lane residual {residual_before:.2f} -> {residual_after:.2f} m"
        )
        self.transform = cb.CarlaTransform(town, offset=offset, z_offset=z_offset)
        ego_state = [float(v) for v in ego["state"]]
        target_points = np.asarray(route["sparse_plan"], dtype=np.float64).reshape(-1, 3).copy()
        dense_points = np.asarray(route["dense_plan"], dtype=np.float64).reshape(-1, 3).copy()
        self.dense_plan_xy = dense_points[:, :2].copy()  # CARLA frame, for the world log
        target_points[:, 2] = self.transform.z_to_bin(target_points[:, 2])
        dense_points[:, 2] = self.transform.z_to_bin(dense_points[:, 2])
        self.route_goals = route_goals_from_target_points(
            target_points,
            dense_points,
            self.transform.loc_to_bin,
            self.transform.loc_to_bin(ego_state[STATE_X], ego_state[STATE_Y]),
            self.env.goal_radius,
        )
        # the leaderboard's target points, next num_goals of them at all times (fewer only at the route end)
        self.goal_window = RouteGoalWindow(self.env, self.route_goals, sliding=True)

        bin_data = mbin.read_bin(Path(self.town_bin))
        bin_traffic = bin_data["traffic"]
        self.stop_line_centers = np.array(
            [
                [0.5 * (t["stop_line"][0] + t["stop_line"][3]), 0.5 * (t["stop_line"][1] + t["stop_line"][4])]
                for t in bin_traffic
            ]
        ).reshape(-1, 2)

        lights = route["lights"]
        light_map, bin_num_traffic = cb.map_lights_to_bin(lights, self.transform, self.town_bin)
        dense_route_bin_xy = np.array(
            [self.transform.loc_to_bin(x, y) for x, y in self.dense_plan_xy], np.float64
        ).reshape(-1, 2)
        light_line_indices, light_lines = cb.light_stop_line_overrides(
            lights, light_map, self.transform, self.town_bin, route_xy=dense_route_bin_xy
        )
        if len(light_line_indices):
            moved_centers = 0.5 * (light_lines[:, 0:2] + light_lines[:, 3:5])
            max_shift_m = float(np.hypot(*(moved_centers - self.stop_line_centers[light_line_indices]).T).max())
            self.env.set_traffic_light_lines(light_line_indices, light_lines)
            self.stop_line_centers[light_line_indices] = moved_centers
            print(
                f"[opengigaflow_agent] light stop lines: {len(light_line_indices)} on the route moved onto the "
                f"leaderboard's red-light lines (max {max_shift_m:.1f} m)"
            )
        # the shadow env runs on CARLA's trigger volumes: the bin's exported stop lines sit up to 9 m off and miss a few
        self._stop_sign_lines, stop_sign_headings = cb.stop_sign_lines(route["stop_signs"], self.transform)
        self.num_traffic = self.env.set_stop_signs(self._stop_sign_lines, stop_sign_headings)
        print(
            f"[opengigaflow_agent] stop signs: {len(stop_sign_headings)} CARLA trigger volumes replace the bin's "
            f"(traffic elements {bin_num_traffic} -> {self.num_traffic})"
        )
        self.light_bin_indices_by_id = {int(light["id"]): light_map[i] for i, light in enumerate(lights)}
        self.last_light_states = np.zeros(self.num_traffic, np.int32)
        self.last_yellow_remaining_s = np.zeros(self.num_traffic, np.float32)
        self._yellow_countdown = YellowCountdown(self.env.traffic_light_yellow_min_s, self.dt)

        self.controller = PedalMap(pedal_gains_for(ego.get("blueprint")))
        if self.pedal_yellow_margin:
            # a sprint through a yellow can lose one gear shift without drive, which the policy does not know about
            self._yellow_margin_s = self.controller.gains.gear_shift_seconds

        if self.obs_html_dir:
            from opengigaflow.ocean.cosim.obs_replay import ObsReplayCapture

            Path(self.obs_html_dir).mkdir(parents=True, exist_ok=True)
            self._obs_html = ObsReplayCapture(
                self.env, self.policy, Path(self.obs_html_dir) / self.route_tag, max_steps=self.obs_html_max_steps
            )

        if self.telemetry_dir:
            Path(self.telemetry_dir).mkdir(parents=True, exist_ok=True)
            self.telemetry_file = open(Path(self.telemetry_dir) / f"{self.route_tag}.csv", "w")
            self.telemetry_file.write(TELEMETRY_HEADER)

        print(
            f"[opengigaflow_agent] town={town} tick_dt={self.tick_dt} dt={self.dt} route_goals={len(self.route_goals)} "
            f"max_speed_mps={self.env.max_speed_mps:.1f} (C_vel={self.env.max_speed_mps / self.env.base_max_speed_mps:.2f}) "
            f"zero_partner_stopped_time={int(self.zero_partner_stopped_time)} "
            f"vru_size_pad_m={self.vru_size_pad_m:g}"
        )
        self.initialized = True

    def _read_states(self, partners):
        idx, x, y, z, h, vx, vy, yaw_rate, accel_long, stopped_s = [], [], [], [], [], [], [], [], [], []
        stopped_by_id = {}
        for j, row in enumerate(partners):
            idx.append(1 + j)  # agent 0 = ego; others fill 1..M
            bx, by, bz, bh, bvx, bvy, byr, bal = self.transform.state_to_bin(
                row[STATE_X],
                row[STATE_Y],
                row[STATE_Z],
                row[STATE_YAW_DEG],
                row[STATE_VX],
                row[STATE_VY],
                row[STATE_ANGULAR_VELOCITY_Z],
                row[STATE_ACCEL_X],
                row[STATE_ACCEL_Y],
                row[PARTNER_BOX_Z],
                row[PARTNER_EXTENT_Z],
            )
            x.append(bx)
            y.append(by)
            z.append(bz)
            h.append(bh)
            vx.append(bvx)
            vy.append(bvy)
            yaw_rate.append(byr)
            accel_long.append(bal)
            # stopped time follows the CARLA actor, not the shadow slot (slots reshuffle by distance every step)
            actor_id = int(row[PARTNER_ACTOR_ID])
            is_stopped = math.hypot(bvx, bvy) < self._partner_stopped_speed_threshold
            stopped_by_id[actor_id] = self._partner_stopped_s.get(actor_id, 0.0) + self.dt if is_stopped else 0.0
            stopped_s.append(stopped_by_id[actor_id])
        self._partner_stopped_s = stopped_by_id
        if self.zero_partner_stopped_time:
            # walkers keep their counter: held at 0, a walker that stays standing reads as about to cross forever
            stopped_s = [s if row[PARTNER_IS_WALKER] > 0.0 else 0.0 for s, row in zip(stopped_s, partners)]
        return (
            np.array(idx, np.int32),
            np.array(x, np.float32),
            np.array(y, np.float32),
            np.array(z, np.float32),
            np.array(h, np.float32),
            np.array(vx, np.float32),
            np.array(vy, np.float32),
            np.array(yaw_rate, np.float32),
            np.array(accel_long, np.float32),
            np.array(stopped_s, np.float32),
        )

    def _read_sizes(self, partners):
        idx, length, width = [], [], []
        for j, row in enumerate(partners):
            idx.append(1 + j)
            is_vru = row[PARTNER_IS_WALKER] > 0.0 or row[PARTNER_IS_TWO_WHEELER] > 0.0
            vru_pad_m = self.vru_size_pad_m if is_vru else 0.0
            length.append(max(2.0 * row[PARTNER_EXTENT_X], MIN_PARTNER_SIZE_M) + vru_pad_m)
            width.append(max(2.0 * row[PARTNER_EXTENT_Y], MIN_PARTNER_SIZE_M) + vru_pad_m)
        return (np.array(idx, np.int32), np.array(length, np.float32), np.array(width, np.float32))

    def _read_light_states(self, snapshot):
        """(state, seconds it stays yellow) of every bin traffic element from the snapshot's lights."""
        # unmapped light elements read OFF (a training state), never UNKNOWN
        states = np.full(self.num_traffic, cb.TRAFFIC_LIGHT_STATE_OFF, np.int32)
        yellow_remaining_s = np.zeros(self.num_traffic, np.float32)
        state_by_id = {int(light_id): cb.LIGHT_STATE_BY_NAME[name] for light_id, name in snapshot["light_states"]}
        remaining_by_id = self._yellow_remaining_by_id(snapshot, state_by_id)
        ego_light_id = int(snapshot["ego_light_id"])
        # the ego's own light is written last so it wins over any other light mapped onto the same element
        ordered_ids = [light_id for light_id in state_by_id if light_id != ego_light_id]
        if ego_light_id in self.light_bin_indices_by_id:
            ordered_ids.append(ego_light_id)
        for light_id in ordered_ids:
            for j in self.light_bin_indices_by_id[light_id]:
                states[j] = state_by_id[light_id]
                yellow_remaining_s[j] = max(remaining_by_id.get(light_id, 0.0) - self._yellow_margin_s, 0.0)
        return states, yellow_remaining_s

    def _yellow_remaining_by_id(self, snapshot, state_by_id):
        """{light_id: seconds it stays yellow} for the yellow lights: CARLA's own phase timer, or the
        shortest-trained-yellow countdown from the onset where a scenario froze that timer."""
        yellow_ids = {light_id for light_id, state in state_by_id.items() if state == cb.TRAFFIC_LIGHT_STATE_YELLOW}
        countdown_s = self._yellow_countdown.tick(yellow_ids, self.step_count * self.tick_dt)
        remaining_by_id = {}
        for light_id, elapsed_s, yellow_time_s in snapshot["light_timers"]:
            light_id = int(light_id)
            if light_id not in yellow_ids:
                continue
            timed_s = cb.carla_yellow_remaining_s(elapsed_s, yellow_time_s)
            remaining_by_id[light_id] = countdown_s[light_id] if timed_s is None else timed_s
        return remaining_by_id

    def _sync_ego(self, ego_state):
        """Overwrite agent 0 (ego) from the simulator's ground-truth pose/size."""
        bin_state = self.transform.state_to_bin(
            ego_state[STATE_X],
            ego_state[STATE_Y],
            ego_state[STATE_Z],
            ego_state[STATE_YAW_DEG],
            ego_state[STATE_VX],
            ego_state[STATE_VY],
            ego_state[STATE_ANGULAR_VELOCITY_Z],
            ego_state[STATE_ACCEL_X],
            ego_state[STATE_ACCEL_Y],
            self.ego_box_z,
            self.ego_extent[2],
        )
        bin_state = self._plant_state(bin_state)
        self.env.set_agent_states([0], *[[value] for value in bin_state], steering=[self._intent_steering_rad])
        self.env.set_agent_sizes([0], [2.0 * self.ego_extent[0]], [2.0 * self.ego_extent[1]])

    def _plant_state(self, bin_state):
        """Pose, velocity and yaw rate from CARLA's physics, acceleration from the jerk integrator."""
        x, y, z, heading, vx, vy, yaw_rate, _ = bin_state
        self._plant_speed_mps = vx * math.cos(heading) + vy * math.sin(heading)
        self.controller.observe(self._plant_speed_mps, -yaw_rate, self.tick_dt)
        if self._plant_speed_mps < STANDSTILL_SPEED_MPS:
            braked_to_a_stop = not self._held and self._intent_accel_mps2 <= 0.0
            if braked_to_a_stop or self._intent_accel_mps2 < 0.0:
                # like the jerk model's, the acceleration starts again from zero; the ego never reverses
                self._held, self._intent_accel_mps2 = True, 0.0
            if self._plant_speed_mps * self._intent_accel_mps2 <= 0.0:
                # held, or creeping against the launch: the jerk model would read that as a stop and cancel the launch
                vx, vy = 0.0, 0.0
        else:
            self._held = False
        # the jerk model moves the box centre along its heading; CARLA's centre swings about the rear axle instead
        wheelbase_m = DRIVE_WHEELBASE_PER_LENGTH * 2.0 * self.ego_extent[0]
        curvature_per_m = math.tan(self._intent_steering_rad) / wheelbase_m
        travel_heading = heading + math.atan(self.controller.gains.rear_axle_to_center_m * curvature_per_m)
        return (x, y, z, travel_heading, vx, vy, yaw_rate, self._intent_accel_mps2)

    def _sync(self, snapshot):
        """Overwrite the shadow env's ego, partners and lights from the simulator's ground truth -> the recomputed
        observations, row 0 = ego."""
        ego_state = [float(v) for v in snapshot["ego"]]
        self._ego_speed_carla = ego_speed_from_state(ego_state)
        self._plant_gear = int(snapshot["ego_gear"])
        self._sync_ego(ego_state)
        ego = self.env.get_global_agent_state()
        ebx, eby = float(ego["x"][0]), float(ego["y"][0])

        partners = np.asarray(snapshot["partners"], dtype=np.float64).reshape(-1, PARTNER_ROW_COLUMNS)
        if len(partners):
            states = self._read_states(partners)
            sizes = self._read_sizes(partners)
            self.env.set_agent_states(*states)
            self.env.set_agent_sizes(*sizes)
            is_walker = (partners[:, PARTNER_IS_WALKER] > 0.0).astype(np.float32)
            self._last_partners = (states[1], states[2], states[4], states[5], states[6], sizes[1], sizes[2], is_walker)
        else:
            self._last_partners = None
        surplus = np.arange(1 + len(partners), self.num_agents, dtype=np.int32)
        if len(surplus):
            far = np.full(len(surplus), FAR_AWAY, np.float32)
            zero = np.zeros_like(far)
            self.env.set_agent_states(surplus, far, far, far, zero, zero, zero, zero, zero)

        self.last_light_states, self.last_yellow_remaining_s = self._read_light_states(snapshot)
        self.env.set_traffic_light_states(
            self.last_light_states,
            self.last_yellow_remaining_s if self.env.obs_traffic_light_yellow_remaining else None,
        )

        self.goal_window.sync(ebx, eby, float(ego["heading"][0]))
        return np.asarray(self.env.recompute_observations())

    def _integrate(self, actions):
        """Step the shadow env once and turn the jerk integrator's acceleration and steering into pedals and steer
        for CARLA's physics."""
        if self.step_count >= MAX_POLICY_STEPS:
            raise RuntimeError(f"route exceeded MAX_POLICY_STEPS={MAX_POLICY_STEPS} policy steps")
        self.env.step(actions)
        self.target_speed_mps = self._ego_speed()

        ego_obs = np.asarray(self.env.observations)[0]
        self._intent_accel_mps2 = float(ego_obs[EGO_OBS_ACCEL_LONG_IDX]) * binding.ACCEL_LONG_NORM
        self._intent_steering_rad = float(ego_obs[EGO_OBS_STEERING_IDX]) * self._steering_limit_rad
        wheelbase_m = DRIVE_WHEELBASE_PER_LENGTH * 2.0 * self.ego_extent[0]
        # the bin frame is y-mirrored: its left-positive curvature is a negative CARLA steer
        curvature_per_m = -math.tan(self._intent_steering_rad) / wheelbase_m
        control = self.controller.step(
            self._intent_accel_mps2,
            curvature_per_m,
            self._plant_speed_mps,
            self._plant_gear,
            self._held and self._intent_accel_mps2 <= 0.0,
        )
        return StepResult(actions, control)

    def step(self, snapshot):
        """One policy step from snapshot = {ego: ACTOR_STATE, ego_gear, partners: PARTNER_ROW list (nearest first),
        light_states: [[id, state name]], ego_light_id (-1: none), light_timers: [[id, elapsed_s, yellow_time_s]]}."""
        self.step_count += 1
        obs = self._sync(snapshot)
        actions, aux = self._policy_actions(obs)
        if self.world_log_dir:
            self._record_world_state(obs)
        if self._obs_html is not None:
            self._obs_html.capture(obs, actions, aux, aux.get("action_index"))
        result = self._integrate(actions)
        if self.telemetry_file is not None:
            metrics = self.env.get_state()[0]["agents"][0]["metrics_array"]
            result.pd_flags = {name: float(metrics[idx]) for name, idx in EGO_INFRACTION_METRICS.items()}
        return result

    def _ego_speed(self):
        return float(np.asarray(self.env.observations)[0][EGO_OBS_SPEED_IDX]) * self.env.obs_norm_speed_mps

    def _policy_actions(self, obs):
        """-> (actions (1, act_dim) float32, aux dict of policy outputs for the obs_html capture)."""
        import torch

        import opengigaflow.pytorch

        action_selection = (
            opengigaflow.pytorch.ACTION_SELECT_MODE
            if self.policy.is_continuous
            else opengigaflow.pytorch.ACTION_SELECT_MEAN
        )
        obs_tensor = torch.as_tensor(obs).to(self.device)
        with torch.no_grad():
            logits, value = self.policy.forward_eval(obs_tensor)
            action, _, entropy, cont_action = opengigaflow.pytorch.sample_logits(
                logits, action_selection=action_selection, env_continuous=True, policy=self.policy
            )
        actions = (cont_action if cont_action is not None else action).cpu().numpy().reshape(1, -1)
        aux = {}
        if self._obs_html is not None:
            aux = self._obs_html.policy_outputs(obs_tensor, logits, value, entropy)
            aux["action_index"] = action.cpu().numpy().reshape(-1) if not self.policy.is_continuous else None
        return actions, aux

    def _record_world_state(self, obs):
        """Bin-frame snapshot at sync time (pre-step ego, partners as streamed, light states)."""
        ego = self.env.get_global_agent_state()
        accel_long = float(obs[0][EGO_OBS_ACCEL_LONG_IDX]) * binding.ACCEL_LONG_NORM
        step_idx = len(self._world_log["ego"])
        self._world_log["ego"].append(
            (
                self.step_count,
                float(ego["x"][0]),
                float(ego["y"][0]),
                float(ego["heading"][0]),
                self._ego_speed(),
                accel_long,
                float(self.goal_window.current_index),
            )
        )
        if self._last_partners is not None:
            x, y, h, vx, vy, length, width, is_walker = self._last_partners
            speed = np.hypot(vx, vy)
            rows = np.column_stack([np.full(len(x), step_idx, np.float32), x, y, h, length, width, speed, is_walker])
            self._world_log["partners"].append(rows.astype(np.float32))
        self._world_log["lights"].append(self.last_light_states.astype(np.int8))
        self._world_log["light_yellow_s"].append(self.last_yellow_remaining_s.astype(np.float32))

    def _write_world_log(self):
        Path(self.world_log_dir).mkdir(parents=True, exist_ok=True)
        dense = np.array([self.transform.loc_to_bin(x, y) for x, y in self.dense_plan_xy], np.float32).reshape(-1, 2)
        partners = self._world_log["partners"]
        np.savez_compressed(
            Path(self.world_log_dir) / f"{self.route_tag}.npz",
            ego=np.array(self._world_log["ego"], np.float32),
            partners=np.concatenate(partners) if partners else np.zeros((0, 8), np.float32),
            lights=np.stack(self._world_log["lights"]) if self._world_log["lights"] else np.zeros((0, 0), np.int8),
            light_yellow_s=np.stack(self._world_log["light_yellow_s"])
            if self._world_log["light_yellow_s"]
            else np.zeros((0, 0), np.float32),
            route_goals=self.route_goals,
            stop_signs=self._stop_sign_lines,
            dense_route=dense,
            meta=json.dumps(
                {
                    "town": Path(self.town_bin).stem.split("__")[-1],
                    "town_bin": self.town_bin,
                    "dt": self.dt,
                    "tick_dt": self.tick_dt,
                    "offset": [self.transform.tx, self.transform.ty],
                    "z_offset": self.transform.tz,
                }
            ),
        )

    def telemetry(self, result, carla_flags):
        """One CSV row per policy step (COSIM_TELEMETRY): commanded vs achieved speed, the nearest mapped light and
        both infraction sources (pd_* = the shadow env, carla_* = the simulator side, None -> all 0)."""
        if self.telemetry_file is None:
            return
        pd_flags = result.pd_flags
        carla_flags = carla_flags or {name: 0.0 for name in EGO_INFRACTION_METRICS}
        ego = self.env.get_global_agent_state()
        ex, ey = float(ego["x"][0]), float(ego["y"][0])
        cur = min(self.goal_window.current_index, len(self.route_goals) - 1)
        goal_dist = float(np.hypot(self.route_goals[cur, 0] - ex, self.route_goals[cur, 1] - ey))
        near_dist, near_state = -1.0, -1
        if len(self.stop_line_centers):
            d2 = (self.stop_line_centers[:, 0] - ex) ** 2 + (self.stop_line_centers[:, 1] - ey) ** 2
            j = int(d2.argmin())
            near_dist, near_state = float(np.sqrt(d2[j])), int(self.last_light_states[j])
        self.telemetry_file.write(
            f"{self.step_count},{self._ego_speed_carla:.3f},{self.target_speed_mps:.3f},"
            f"{float(result.actions[0, 0])},{self.goal_window.current_index},{goal_dist:.1f},{near_dist:.1f},{near_state},"
            f"{pd_flags['collision']:.0f},{pd_flags['offroad']:.0f},{pd_flags['red_light']:.0f},"
            f"{pd_flags['stop_sign']:.0f},"
            f"{carla_flags['collision']:.0f},{carla_flags['offroad']:.0f},{carla_flags['red_light']:.0f},"
            f"{carla_flags['stop_sign']:.0f}\n"
        )

    def finish(self):
        """Route done: flush every per-route output and release the shadow env."""
        if not self.initialized:
            return
        print(
            f"[opengigaflow_agent] route done: goals {self.goal_window.current_index + 1}/{len(self.route_goals)}, "
            f"tracking {self.controller.stats()}"
        )
        if self.telemetry_file is not None:
            self.telemetry_file.close()
        if self._obs_html is not None:
            written = self._obs_html.write(render_html=self.obs_html_render)
            print(f"[opengigaflow_agent] wrote obs replay ({len(self._obs_html)} frames) -> {written}")
        if self.world_log_dir and self._world_log["ego"]:
            self._write_world_log()
        self.env.close()
        self.initialized = False
