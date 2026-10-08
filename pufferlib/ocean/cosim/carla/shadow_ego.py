"""PufferDrive shadow env + policy driving one ego from another simulator's ground truth.

Shared by the CARLA leaderboard integrations: CaRL's Leaderboard 2.0 agent (leaderboard_agent.py,
CARLA 0.9.15, in-process) and the Leaderboard 1.0 client/server pair (lb1/, CARLA 0.9.10, whose
Python 3.7 cannot import PufferDrive). The simulator side only reads CARLA and moves the ego;
everything the policy needs happens here and never imports carla. Inputs and outputs are in the
CARLA frame (metres, degrees, left-handed y) as plain lists/dicts, so they also travel as JSON.

Everything about the shadow env (obs layout, dynamics, dt, goal count, agent pool size, ...) comes
from the checkpoint's sibling config.yaml with the clean-eval profile applied on top (cosim/arch.py
CLEAN_EVAL_OVERRIDES); only the structural co-sim keys (map, pool wiring) are set here.

Actor state rows (CARLA frame, physics quantities from CARLA's own physics):
  ACTOR_STATE  [x, y, z, yaw_deg, vx, vy, vz, angular_velocity_z_deg_s, accel_x, accel_y]
  PARTNER_ROW  ACTOR_STATE + [box_z, extent_x, extent_y, extent_z, is_walker, actor_id]

init_route(route) once the ego exists, route = {
  town, tick_dt,
  ego: {extent: [x, y, z], box_z, wheels: controller.wheel_rows, state: ACTOR_STATE},
  sparse_plan: (N, 3) leaderboard target points, dense_plan: (M, 3) the dense route,
  driving_waypoints: carla_bridge.driving_waypoint_samples rows,
  lights: carla_bridge.light_geometry_from_carla dicts, stop_signs: carla_bridge.stop_sign_geometry_from_carla dicts,
  infraction_flags: bool, also compute the shadow env's own infraction flags when no telemetry is written }
step(snapshot) every simulator tick, snapshot = {
  ego: ACTOR_STATE, partners: PARTNER_ROW list (nearest first, at most partner_slots, |dz| <= PARTNER_MAX_ABS_DZ_M),
  light_states: [[light_id, carla.TrafficLightState name], ...], ego_light_id: id reported for the ego or -1 }

Environment variables:
  COSIM_DEVICE=cpu             torch device for the policy
  COSIM_DYNAMICS_SOURCE=pufferdrive
                               "pufferdrive" (default): PufferDrive's own dynamics (the ones the policy
                               trained on) move the ego; StepResult.motion is the pose the simulator
                               teleports the ego to every step.
                               "carla": TrackingController converts the shadow env's target (speed, yaw)
                               into StepResult.control = (steer, throttle, brake) and CARLA's own vehicle
                               physics moves the ego (subject to dynamics-mismatch tracking lag).
  COSIM_MAX_SPEED_MPS=20       ego speed cap; sets the shadow env's C_vel = cap / base_max_speed_mps.
                               Default: the checkpoint's base_max_speed_mps (C_vel = 1). Must lie in the
                               trained range base * [1/conditioning_speed_scale, scale]
                               (0036: 13.3-30 m/s, 0038+: 10-40 m/s).
  COSIM_ZERO_PARTNER_STOPPED_TIME=0
                               1: every partner's seconds_stopped observation feature is held at 0 (the
                               ego's own counter is untouched). Ablates the stopped-time cue behind
                               parked-car / red-light-queue avoidance.
  COSIM_PEDESTRIAN_MIN_SIZE_M=0.0
                               walker partner boxes below this on either axis grow to it (nuPlan's
                               pedestrian_min_size_m; the training spawn floor is 0.8 m, CARLA walkers
                               are 0.4-0.5 m)
  COSIM_TELEMETRY=/dir         write a per-policy-step CSV per route, with the shadow env's and the
                               simulator's infraction flags as columns
  COSIM_OBS_HTML=/dir          write an interactive pufferlib.viz replay per route (the exact obs +
                               policy outputs the ego saw)
  COSIM_OBS_HTML_RENDER=1      0: save only the compact .replay.zlib per route;
                               scripts/eval/render_carla_obs_html.py renders the pages of a whole run
                               into one gallery afterwards
  COSIM_OBS_HTML_MAX_STEPS=12000  frames kept per route replay
  COSIM_DUMP_OBS=/dir          write the raw ego obs + action per policy step per route (.npz)
  COSIM_WORLD_LOG=/dir         write the bin-frame world state per policy step per route (ego,
                               partners, light states, route); input of scripts/eval/analyze_carla_cosim.py
  COSIM_DEBUG_GOAL_LANE=/dir   per-step CSV of the current goal vs the lane find_goal_lane snapped it to
"""

import json
import math
import os
from pathlib import Path

import numpy as np

from pufferlib.ocean.cosim import carla_bridge as cb
from pufferlib.ocean.cosim.arch import checkpoint_config_path, shadow_env_kwargs
from pufferlib.ocean.cosim.carla.controller import TrackingController, vehicle_geometry_from_wheels
from pufferlib.ocean.cosim.goals import RouteGoalWindow, route_goals_from_target_points
from pufferlib.ocean.drive.drive import Drive


ACTOR_STATE_COLUMNS = 10
PARTNER_ROW_COLUMNS = 16
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
    PARTNER_ACTOR_ID,
) = range(ACTOR_STATE_COLUMNS, PARTNER_ROW_COLUMNS)

FAR_AWAY = 1.0e6  # park unused shadow-env agent slots out of observation range
PARTNER_MAX_ABS_DZ_M = 20.0  # CARLA actors farther above/below the ego are hidden scenario props, not traffic
DEFAULT_BASE_MAX_SPEED_MPS = 20.0  # Drive() defaults, for checkpoints whose config predates the keys
DEFAULT_CONDITIONING_SPEED_SCALE = 1.5
MAX_POLICY_STEPS = 100_000  # shadow-env episode cap (~1.4 h at the 0.05 s CARLA tick); sizes the light state buffers
SCENARIO_LENGTH_MARGIN_STEPS = 2
MIN_PARTNER_SIZE_M = 0.1
# Shadow-env metrics_array indices (datatypes.h): collision/offroad/red-light/stop-sign flags.
EGO_INFRACTION_METRICS = {"collision": 0, "offroad": 1, "red_light": 2, "stop_sign": 3}
DYNAMICS_SOURCES = ("carla", "pufferdrive")
TELEMETRY_HEADER = (
    "step,current_speed,target_speed,ego_action,goal_cursor,goal_dist_m,"
    "near_light_dist_m,near_light_state,"
    "pd_infr_collision,pd_infr_offroad,pd_infr_red,pd_infr_stop,"
    "carla_infr_collision,carla_infr_offroad,carla_infr_red,carla_infr_stop\n"
)
GOAL_LANE_DEBUG_HEADER = (
    "step,ego_x,ego_y,goal_x,goal_y,gdir_x,gdir_y,goal_lane_idx,lane_dir_dot_route,lane_dist_to_goal_m\n"
)


def clean_policy_state_dict(state_dict):
    """Strip torch.compile / DDP prefixes (wandb, neptune, the _C kernel, ...)
    from the leaderboard's evaluation environment."""

    def clean(key):
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        return key

    return {clean(k): v for k, v in state_dict.items()}


def resolve_checkpoint(path_to_conf_file):
    """(checkpoint_path, config_dict) from a .pt file or an experiment dir."""
    import yaml

    p = Path(path_to_conf_file).resolve()
    if p.is_file():
        ckpt, cfg_path = p, checkpoint_config_path(p)
    else:
        models = sorted((p / "models").glob("*.pt")) or sorted(p.glob("*.pt"))
        if not models:
            raise FileNotFoundError(f"no .pt checkpoint under {p}")
        ckpt, cfg_path = models[-1], p / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return str(ckpt), cfg


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


def env_flag(name, default="0"):
    raw = os.environ.get(name, default)
    if raw not in ("0", "1"):
        raise ValueError(f"{name} must be '0' or '1', got {raw!r}")
    return raw == "1"


def ego_speed_from_state(state):
    """CARLA's Vector3D.length() of the ego velocity (3-D, as the leaderboard criteria read it)."""
    return math.sqrt(state[STATE_VX] ** 2 + state[STATE_VY] ** 2 + state[STATE_VZ] ** 2)


class StepResult:
    """One policy step: `actions` (num_agents, act_dim), the shadow env's infraction flags (None unless
    requested), and the ego command -- `motion` = [x0, y0, yaw0_deg, speed0, x1, y1, yaw1_deg, speed1, z1]
    (CARLA frame, pre- and post-step pose to teleport along) in pufferdrive mode, `control` = (steer,
    throttle, brake) in carla mode; `target` = (speed_mps, yaw_deg) the policy intends one dt ahead."""

    def __init__(self, actions, target, motion=None, control=None, pd_flags=None):
        self.actions = actions
        self.target = target
        self.motion = motion
        self.control = control
        self.pd_flags = pd_flags


class ShadowEgo:
    def __init__(self, path_to_conf_file, route_tag):
        self.route_tag = route_tag
        self.checkpoint, self.cfg = resolve_checkpoint(path_to_conf_file)
        self.device = os.environ.get("COSIM_DEVICE", "cpu")
        self.dynamics_source = os.environ.get("COSIM_DYNAMICS_SOURCE", "pufferdrive")
        if self.dynamics_source not in DYNAMICS_SOURCES:
            raise ValueError(f"COSIM_DYNAMICS_SOURCE must be one of {DYNAMICS_SOURCES}, got {self.dynamics_source!r}")
        env_cfg = self.cfg["env"]
        self.max_speed_mps = cosim_max_speed_mps(env_cfg)
        self.zero_partner_stopped_time = env_flag("COSIM_ZERO_PARTNER_STOPPED_TIME")
        self.pedestrian_min_size_m = float(os.environ.get("COSIM_PEDESTRIAN_MIN_SIZE_M", "0.0"))
        if not (self.pedestrian_min_size_m >= 0.0):
            raise ValueError(f"COSIM_PEDESTRIAN_MIN_SIZE_M must be >= 0, got {self.pedestrian_min_size_m}")
        # Shadow agent pool == the training per-env cap
        self.num_agents = int(env_cfg["max_agents_per_env"])

        self.telemetry_dir = os.environ.get("COSIM_TELEMETRY", None)
        self.telemetry_file = None
        self.obs_html_dir = os.environ.get("COSIM_OBS_HTML", None)
        self.obs_dump_dir = os.environ.get("COSIM_DUMP_OBS", None)
        self._obs_dump = []
        self.world_log_dir = os.environ.get("COSIM_WORLD_LOG", None)
        self._world_log = {"ego": [], "partners": [], "lights": []}
        self._last_partners = None
        self._partner_stopped_s = {}  # CARLA actor id -> seconds at standstill
        from pufferlib.ocean.drive import binding

        self._partner_stopped_speed_threshold = float(binding.AGENT_STOPPED_SPEED_THRESHOLD)
        self.obs_html_max_steps = int(os.environ.get("COSIM_OBS_HTML_MAX_STEPS", "12000"))
        self.obs_html_render = env_flag("COSIM_OBS_HTML_RENDER", "1")
        self._obs_html = None
        self.debug_goal_lane_dir = os.environ.get("COSIM_DEBUG_GOAL_LANE", None)
        self._goal_lane_debug_file = None
        self._road_data = None

        self.step_count = 0
        self.initialized = False
        self.infraction_flags = False
        self.policy = None
        self.env = None
        self.goal_window = None
        self.target = (0.0, 0.0)  # (target_speed, target_yaw_deg), held between policy steps
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
                # Enforcement off (detection flags still fire): in one endless
                # episode a "stop" latch is permanent and fires only AFTER
                # CARLA scored the infraction (route 5 froze 320 s -> DNF).
                collision_behavior="ignore",
                offroad_behavior="ignore",
                traffic_light_behavior="ignore",
                **({} if self.max_speed_mps is None else {"max_speed_mps": self.max_speed_mps}),
            ),
        )
        self.dynamics_model = arch.get("dynamics_model", "classic")
        # jerk dynamics build accel across steps while sync() re-seeds accel
        self.speed_intent_extension_s = 2.0 * self.dt if self.dynamics_model == "jerk" else 0.0

        self.env = Drive(**arch)
        self.env.reset()

        import torch

        import pufferlib.ocean.torch as drive_torch

        policy_state_dict = clean_policy_state_dict(
            torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        )
        env_for_policy = self.env
        self._boundary_feat_ckpt = None
        bw = policy_state_dict.get("actor_backbone.boundary_encoder.0.weight")
        if bw is not None and bw.shape[1] < env_for_policy.boundary_features:
            # Older checkpoints emit fewer boundary features than this drive.h
            # (e.g. 7 vs 9 -- two zero-padded GPS columns were added later).
            import copy

            self._boundary_feat_ckpt = int(bw.shape[1])
            env_for_policy = copy.copy(self.env)
            env_for_policy.boundary_features = self._boundary_feat_ckpt
            print(
                f"[puffer_agent] checkpoint uses {self._boundary_feat_ckpt} boundary features "
                f"(env emits {self.env.boundary_features}); stripping zero-padded columns"
            )

        policy_cls = getattr(drive_torch, self.cfg.get("policy_name", "Drive"))
        # pre-3.0 checkpoints keep action_type only in the env section
        self.cfg["policy"].setdefault("action_type", self.cfg["env"]["action_type"])
        self.policy = policy_cls(env_for_policy, **self.cfg["policy"]).to(self.device)
        self.policy.load_state_dict(policy_state_dict)
        self.policy.eval()
        print(f"[puffer_agent] loaded policy from {self.checkpoint}")

    def init_route(self, route):
        """Build the shadow env for this route's town and seed it from the simulator's ground truth (see
        the module docstring for the `route` dict). Read-only with respect to the simulator."""
        town = route["town"]
        self.tick_dt = float(route["tick_dt"])
        self.dt = self.tick_dt  # the policy runs every CARLA tick, like the leaderboard's sensor agents
        self.infraction_flags = bool(route.get("infraction_flags", False))
        ego = route["ego"]
        self.ego_extent = [float(v) for v in ego["extent"]]
        self.ego_box_z = float(ego["box_z"])

        self.town_bin = cb.bin_path_for_town(town)
        self._load_policy_and_env(self.town_bin)

        self.transform = cb.CarlaTransform(town, offset=cb.town_offset(self.town_bin))
        offset, z_offset, residual_before, residual_after = cb.calibrate_town_offset(
            route["driving_waypoints"], self.transform, self.town_bin
        )
        print(
            f"[puffer_agent] bin offset calibrated against CARLA lanes: shift "
            f"({offset[0] - self.transform.tx:+.2f}, {offset[1] - self.transform.ty:+.2f}) m, z {z_offset:+.2f} m, "
            f"median lane residual {residual_before:.2f} -> {residual_after:.2f} m"
        )
        self.transform = cb.CarlaTransform(town, offset=offset, z_offset=z_offset)
        ego_state = [float(v) for v in ego["state"]]
        if self.dynamics_source == "pufferdrive":
            self._sync_ego(ego_state, zero_velocity=True)
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

        import data_utils.mirror_map_bin as mbin

        bin_data = mbin.read_bin(Path(self.town_bin))
        bin_traffic = bin_data["traffic"]
        self.stop_line_centers = np.array(
            [
                [0.5 * (t["stop_line"][0] + t["stop_line"][3]), 0.5 * (t["stop_line"][1] + t["stop_line"][4])]
                for t in bin_traffic
            ]
        ).reshape(-1, 2)

        if self.debug_goal_lane_dir:
            # goal_lane_idx comes from road_elements[] in the C file; bin_data["roads"]
            # is the same bin read in the same order, so it's usable as-is.
            self._road_data = bin_data["roads"]
            Path(self.debug_goal_lane_dir).mkdir(parents=True, exist_ok=True)
            self._goal_lane_debug_file = open(Path(self.debug_goal_lane_dir) / f"{self.route_tag}.csv", "w")
            self._goal_lane_debug_file.write(GOAL_LANE_DEBUG_HEADER)

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
                f"[puffer_agent] light stop lines: {len(light_line_indices)} on the route moved onto the leaderboard's "
                f"red-light lines (max {max_shift_m:.1f} m)"
            )
        # the shadow env runs on CARLA's trigger volumes: the bin's exported stop lines sit up to 9 m off and miss a few
        self._stop_sign_lines, stop_sign_headings = cb.stop_sign_lines(route["stop_signs"], self.transform)
        self.num_traffic = self.env.set_stop_signs(self._stop_sign_lines, stop_sign_headings)
        print(
            f"[puffer_agent] stop signs: {len(stop_sign_headings)} CARLA trigger volumes replace the bin's "
            f"(traffic elements {bin_num_traffic} -> {self.num_traffic})"
        )
        # One id -> bin-indices table, reused by both passes in
        # _read_light_states, so they can't disagree on which bin element a
        # given light maps to (a prior independent geometric lookup for the
        # ego-governance pass picked the wrong element ~5% of the time).
        self.light_bin_indices_by_id = {int(light["id"]): light_map[i] for i, light in enumerate(lights)}
        self.last_light_states = np.zeros(self.num_traffic, np.int32)

        wheelbase, max_steer = vehicle_geometry_from_wheels(ego["wheels"])
        self.controller = TrackingController(wheelbase_m=wheelbase, max_steer_rad=max_steer, horizon_s=self.dt)

        if self.obs_html_dir:
            from pufferlib.ocean.cosim.obs_replay import ObsReplayCapture

            Path(self.obs_html_dir).mkdir(parents=True, exist_ok=True)
            self._obs_html = ObsReplayCapture(
                self.env, self.policy, Path(self.obs_html_dir) / self.route_tag, max_steps=self.obs_html_max_steps
            )

        if self.telemetry_dir:
            Path(self.telemetry_dir).mkdir(parents=True, exist_ok=True)
            self.telemetry_file = open(Path(self.telemetry_dir) / f"{self.route_tag}.csv", "w")
            self.telemetry_file.write(TELEMETRY_HEADER)

        print(
            f"[puffer_agent] town={town} tick_dt={self.tick_dt} dt={self.dt} route_goals={len(self.route_goals)} "
            f"max_speed_mps={self.env.max_speed_mps:.1f} (C_vel={self.env.max_speed_mps / self.env.base_max_speed_mps:.2f}) "
            f"zero_partner_stopped_time={int(self.zero_partner_stopped_time)} "
            f"pedestrian_min_size_m={self.pedestrian_min_size_m:g}"
        )
        self.initialized = True

    # --- shadow-env sync (read-only w.r.t. the simulator) -------------------

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
            is_stopped = math.hypot(bvx, bvy) <= self._partner_stopped_speed_threshold
            stopped_by_id[actor_id] = self._partner_stopped_s.get(actor_id, 0.0) + self.dt if is_stopped else 0.0
            stopped_s.append(stopped_by_id[actor_id])
        self._partner_stopped_s = stopped_by_id
        if self.zero_partner_stopped_time:
            stopped_s = [0.0] * len(partners)
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
            walker_floor_m = self.pedestrian_min_size_m if row[PARTNER_IS_WALKER] > 0.0 else 0.0
            length.append(max(2.0 * row[PARTNER_EXTENT_X], MIN_PARTNER_SIZE_M, walker_floor_m))
            width.append(max(2.0 * row[PARTNER_EXTENT_Y], MIN_PARTNER_SIZE_M, walker_floor_m))
        return (np.array(idx, np.int32), np.array(length, np.float32), np.array(width, np.float32))

    def _read_light_states(self, light_states, ego_light_id):
        """Ground truth for every mapped bin traffic element, via the
        precomputed light_bin_indices_by_id table (see init_route)."""
        # unmapped light elements read OFF (a training state), never UNKNOWN
        states = np.full(self.num_traffic, cb.TRAFFIC_LIGHT_STATE_OFF, np.int32)
        state_by_id = {}
        for light_id, state_name in light_states:
            state = cb.light_state_from_name(state_name)
            state_by_id[int(light_id)] = state
            for j in self.light_bin_indices_by_id[int(light_id)]:
                if 0 <= j < self.num_traffic:
                    states[j] = state
        ego_light_id = int(ego_light_id)
        if ego_light_id in self.light_bin_indices_by_id:
            state = state_by_id[ego_light_id]
            for j in self.light_bin_indices_by_id[ego_light_id]:
                if 0 <= j < self.num_traffic:
                    states[j] = state
        return states

    def _sync_ego(self, ego_state, zero_velocity=False):
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
        if zero_velocity:
            x, y, z, heading, *_ = bin_state
            bin_state = (x, y, z, heading, 0.0, 0.0, 0.0, 0.0)
        self.env.set_agent_states(np.array([0], np.int32), *[np.array([v], np.float32) for v in bin_state])
        self.env.set_agent_sizes(
            np.array([0], np.int32),
            np.array([2.0 * self.ego_extent[0]], np.float32),
            np.array([2.0 * self.ego_extent[1]], np.float32),
        )

    def _sync(self, snapshot):
        """Overwrite the shadow env's background agents/lights (+ ego, only in
        dynamics_source='carla' mode) from the simulator's ground truth and return the
        recomputed observation array (num_agents, obs_dim). Row 0 = ego."""
        ego_state = [float(v) for v in snapshot["ego"]]
        self._ego_speed_carla = ego_speed_from_state(ego_state)
        if self.dynamics_source == "carla":
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
        n_used = 1 + len(partners)
        if n_used < self.num_agents:
            sp = np.arange(n_used, self.num_agents, dtype=np.int32)
            zf = np.full(len(sp), FAR_AWAY, np.float32)
            zz = np.zeros_like(zf)
            self.env.set_agent_states(sp, zf, zf, zf, zz, zz, zz, zz, zz)

        self.last_light_states = self._read_light_states(snapshot["light_states"], snapshot["ego_light_id"])
        self.env.set_traffic_light_states(self.last_light_states)

        self.goal_window.sync(ebx, eby, float(ego["heading"][0]))
        if self._goal_lane_debug_file is not None:
            self._write_goal_lane_debug_row(ebx, eby, self.route_goals[self.goal_window.current_index])
        return np.asarray(self.env.recompute_observations())

    def _write_goal_lane_debug_row(self, ebx, eby, current_goal):
        """One CSV row: the CURRENT goal (window slot 0, current_goal_idx after
        the reset in c_set_agent_goals) vs the lane find_goal_lane snapped it
        to -- that lane's direction (nearest segment to the goal, mirroring
        find_goal_lane's own point-to-segment search) dotted against the route
        direction used at snap time, and the snap distance. dot < 0 means the
        snapped lane runs opposite the route (oncoming lane); a large snap
        distance means no nearby lane passed find_goal_lane's gates at all."""
        gx, gy, _, gdir_x, gdir_y = current_goal
        state = self.env.get_state()
        scenario = state[0] if isinstance(state, list) else state
        ego_agent = (scenario.get("agents") or [{}])[0]
        lane_idx = int(ego_agent.get("goal_lane_idx", -1))
        dot, dist = float("nan"), float("nan")
        if lane_idx >= 0 and self._road_data is not None and lane_idx < len(self._road_data):
            road = self._road_data[lane_idx]
            xs, ys = np.asarray(road["x"], dtype=np.float64), np.asarray(road["y"], dtype=np.float64)
            if len(xs) >= 2:
                seg_dx, seg_dy = xs[1:] - xs[:-1], ys[1:] - ys[:-1]
                seg_len = np.hypot(seg_dx, seg_dy)
                seg_len_safe = np.where(seg_len > 1e-6, seg_len, 1.0)
                t = np.clip(((gx - xs[:-1]) * seg_dx + (gy - ys[:-1]) * seg_dy) / (seg_len_safe**2), 0.0, 1.0)
                px, py = xs[:-1] + t * seg_dx, ys[:-1] + t * seg_dy
                d = np.hypot(gx - px, gy - py)
                k = int(np.argmin(d))
                dist = float(d[k])
                route_norm = math.hypot(gdir_x, gdir_y)
                if seg_len[k] > 1e-6 and route_norm > 1e-6:
                    dot = float((gdir_x * seg_dx[k] + gdir_y * seg_dy[k]) / (seg_len[k] * route_norm))
        self._goal_lane_debug_file.write(
            f"{self.step_count},{ebx:.2f},{eby:.2f},{gx:.2f},{gy:.2f},{gdir_x:.3f},{gdir_y:.3f},"
            f"{lane_idx},{dot:.3f},{dist:.2f}\n"
        )
        self._goal_lane_debug_file.flush()

    def _integrate(self, actions, ego_state):
        """Step the shadow env ONCE (one policy step == one shadow tick).

        dynamics_source='carla': the tracking controller chases the ego's target kinematic state
        with CARLA's physics -> StepResult.control.

        dynamics_source='pufferdrive': PufferDrive's own jerk/classic dynamics (the ones the policy
        trained on) ARE the ego's motion -- no target to chase. StepResult.motion is the pre/post-step
        pose the simulator teleports the ego along, where the next tick's observation reads it.
        """
        if self.step_count >= MAX_POLICY_STEPS:
            raise RuntimeError(f"route exceeded MAX_POLICY_STEPS={MAX_POLICY_STEPS} policy steps")
        before = self.env.get_global_agent_state()
        speed_before = self._ego_speed()
        self.env.step(actions)
        after = self.env.get_global_agent_state()
        target_yaw_deg = self.transform.bin_heading_to_yaw(float(after["heading"][0]))
        speed_after = self._ego_speed()
        accel_after = float(np.asarray(self.env.observations)[0][4]) * self._accel_long_norm()

        if self.dynamics_source == "pufferdrive":
            x0, y0 = self.transform.bin_to_loc(float(before["x"][0]), float(before["y"][0]))
            x1, y1 = self.transform.bin_to_loc(float(after["x"][0]), float(after["y"][0]))
            yaw0_deg = self.transform.bin_heading_to_yaw(float(before["heading"][0]))
            z1 = self.transform.z_to_carla(float(after["z"][0]))
            self.target = (speed_after, target_yaw_deg)  # logged, not chased
            return StepResult(
                actions, self.target, motion=[x0, y0, yaw0_deg, speed_before, x1, y1, target_yaw_deg, speed_after, z1]
            )

        target_speed = speed_after + max(accel_after, 0.0) * self.speed_intent_extension_s
        self.target = (target_speed, target_yaw_deg)
        # Controller runs every tick against the latest CARLA state, chasing the target of this policy step.
        control = self.controller.step(
            self._ego_speed_carla, ego_state[STATE_YAW_DEG], target_speed, target_yaw_deg, self.tick_dt
        )
        return StepResult(actions, self.target, control=control)

    def step(self, snapshot):
        """Shadow env <- simulator ground truth, policy, one dt of dynamics -> StepResult."""
        self.step_count += 1
        obs = self._sync(snapshot)
        actions, aux = self._policy_actions(obs)
        if self.obs_dump_dir:
            self._obs_dump.append((self.step_count, obs[0].astype(np.float32), actions[0].astype(np.float32)))
        if self.world_log_dir:
            self._record_world_state(obs)
        if self._obs_html is not None:
            self._obs_html.capture(obs, actions, aux, aux.get("action_index"))
        result = self._integrate(actions, [float(v) for v in snapshot["ego"]])
        if self.telemetry_file is not None or self.infraction_flags:
            result.pd_flags = self._ego_infractions()
        return result

    def _ego_speed(self):
        return float(np.asarray(self.env.observations)[0][0]) * self._max_speed()

    def _max_speed(self):
        return self.env.obs_norm_speed_mps

    def _accel_long_norm(self):
        from pufferlib.ocean.drive import binding

        return binding.ACCEL_LONG_NORM

    def _ego_infractions(self):
        """{'collision': f, 'offroad': f, 'red_light': f, 'stop_sign': f} from the shadow
        pufferdrive env's own model of the ego (compute_metrics, refreshed by
        the last integrate()) -- what the policy would have caused according
        to PufferDrive's own dynamics, not necessarily what the real CARLA
        vehicle did (the simulator side's own detectors tell that)."""
        state = self.env.get_state()
        scenario = state[0] if isinstance(state, list) else state
        agents = scenario.get("agents") or []
        if not agents:
            return {name: 0.0 for name in EGO_INFRACTION_METRICS}
        metrics = agents[0].get("metrics_array") or []
        return {
            name: float(metrics[idx]) if idx < len(metrics) else 0.0 for name, idx in EGO_INFRACTION_METRICS.items()
        }

    # --- policy + capture ---------------------------------------------------

    def _adapt_obs_for_policy(self, obs):
        """Strip the trailing zero-padded boundary columns when the checkpoint
        predates them (see _load_policy_and_env). No-op otherwise."""
        if self._boundary_feat_ckpt is None:
            return obs
        env = self.env
        n_slots = env.obs_slots_boundary_kept
        feat_env, feat_ckpt = env.boundary_features, self._boundary_feat_ckpt
        b0 = (
            env.ego_features
            + env.num_reward_coefs
            + env.goal_dim
            + env.obs_slots_partners_n * env.partner_features
            + env.obs_slots_lane_kept * env.lane_features
        )
        b1 = b0 + n_slots * feat_env
        obs = np.asarray(obs)
        boundary = obs[:, b0:b1].reshape(obs.shape[0], n_slots, feat_env)[:, :, :feat_ckpt]
        return np.concatenate([obs[:, :b0], boundary.reshape(obs.shape[0], -1), obs[:, b1:]], axis=1)

    def _policy_actions(self, obs):
        """-> (actions (num_agents, act_dim) int32, aux dict). aux carries
        per-agent value/entropy/action-probs/pool for the obs_html capture"""
        import torch

        import pufferlib.pytorch
        import pufferlib.spaces

        obs = self._adapt_obs_for_policy(obs)
        # A discrete policy head on a continuous env needs the bin->continuous
        # mapping (pufferl.py feeds cont_action to the env, never the bin indices).
        env_continuous = isinstance(self.env.single_action_space, pufferlib.spaces.Box)
        action_selection = (
            pufferlib.pytorch.ACTION_SELECT_MEAN
            if env_continuous and not self.policy.is_continuous
            else pufferlib.pytorch.ACTION_SELECT_MODE
        )
        with torch.no_grad():
            logits, value = self.policy.forward_eval(torch.as_tensor(obs).to(self.device))
            action, _, entropy, cont_action = pufferlib.pytorch.sample_logits(
                logits, action_selection=action_selection, env_continuous=env_continuous, policy=self.policy
            )
        env_action = cont_action if env_continuous and cont_action is not None else action
        actions = env_action.cpu().numpy().reshape(1, -1)
        if not env_continuous:
            actions = actions.astype(np.int32)
        aux = {}
        if self._obs_html is not None:
            aux = self._obs_html.policy_outputs(torch.as_tensor(obs).to(self.device), logits, value, entropy)
            aux["action_index"] = action.cpu().numpy().reshape(-1) if not self.policy.is_continuous else None
        return actions, aux

    def _record_world_state(self, obs):
        """Bin-frame snapshot at sync time (pre-step ego, partners as streamed, light states)."""
        ego = self.env.get_global_agent_state()
        accel_long = float(obs[0][4]) * self._accel_long_norm()
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

    def _write_world_log(self):
        Path(self.world_log_dir).mkdir(parents=True, exist_ok=True)
        dense = np.array([self.transform.loc_to_bin(x, y) for x, y in self.dense_plan_xy], np.float32).reshape(-1, 2)
        partners = self._world_log["partners"]
        np.savez_compressed(
            Path(self.world_log_dir) / f"{self.route_tag}.npz",
            ego=np.array(self._world_log["ego"], np.float32),
            partners=np.concatenate(partners) if partners else np.zeros((0, 8), np.float32),
            lights=np.stack(self._world_log["lights"]) if self._world_log["lights"] else np.zeros((0, 0), np.int8),
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
        """One CSV row per policy step (when COSIM_TELEMETRY is set): what the loop commanded vs
        achieved, the nearest mapped light's state and both infraction sources (pd_* = shadow pufferdrive
        env, carla_* = the simulator side's own ground truth, None -> all 0)."""
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
        current_speed = (
            float(np.asarray(self.env.observations)[0, 0]) * self._max_speed()
            if self.dynamics_source == "pufferdrive"
            else self._ego_speed_carla
        )
        self.telemetry_file.write(
            f"{self.step_count},{current_speed:.3f},{self.target[0]:.3f},"
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
            f"[puffer_agent] route done: goals {self.goal_window.current_index + 1}/{len(self.route_goals)}, "
            f"tracking {self.controller.stats()}"
        )
        if self.telemetry_file is not None:
            self.telemetry_file.close()
        if self._goal_lane_debug_file is not None:
            self._goal_lane_debug_file.close()
        if self._obs_html is not None:
            written = self._obs_html.write(render_html=self.obs_html_render)
            print(f"[puffer_agent] wrote obs replay ({len(self._obs_html)} frames) -> {written}")
        if self.world_log_dir and self._world_log["ego"]:
            self._write_world_log()
        if self.obs_dump_dir and self._obs_dump:
            Path(self.obs_dump_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                Path(self.obs_dump_dir) / f"{self.route_tag}.npz",
                step=np.array([s for s, _, _ in self._obs_dump], np.int32),
                obs=np.stack([o for _, o, _ in self._obs_dump]),
                action=np.stack([a for _, _, a in self._obs_dump]),
            )
        self.env.close()
        self.initialized = False
