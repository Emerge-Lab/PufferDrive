"""PufferDrive policy as a CARLA leaderboard agent (CaRL original_leaderboard).

Usage (CaRL env vars/paths as in CaRL/CARLA/README.md, plus PufferDrive repo
root on PYTHONPATH for `pufferlib` and `data_utils`):

  python ${CARL_WORK_DIR}/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py \
      --routes ${CARL_WORK_DIR}/custom_leaderboard/leaderboard/data/longest6_split/longest6_00.xml \
      --agent /path/to/pufferlib/ocean/cosim/carla/leaderboard_agent.py \
      --agent-config /path/to/experiments/puffer_drive_xxx/models/model_xxx.pt \
      --checkpoint /path/to/results/result.json --track MAP

Everything about the shadow env (obs layout, dynamics, dt, goal count, agent
pool size, ...) comes from the checkpoint's sibling config.yaml, with the
clean-eval profile applied on top (cosim/arch.py CLEAN_EVAL_OVERRIDES); only
the structural co-sim keys (map, pool wiring) are set here.

Environment variables:
  SCENARIO_RUNNER_ROOT         REQUIRED: path to the scenario_runner checkout
                               (route scenarios silently fail to load without it)
  CARL_WORK_DIR                REQUIRED if COSIM_TELEMETRY or
                               COSIM_RECORD_INFRACTIONS is set: path to the
                               CaRL/CARLA checkout, to import its
                               reward.criteria collision/offroad detectors
  COSIM_DEVICE=cpu             torch device for the policy
  COSIM_DYNAMICS_SOURCE=pufferdrive
                               "pufferdrive" (default): PufferDrive's own
                               dynamics (the ones the policy trained on) move
                               the ego; the CARLA actor is teleported to match
                               every policy step.
                               "carla": TrackingController converts the shadow
                               env's target (speed, yaw) into throttle/brake/
                               steer and CARLA's own vehicle physics moves the
                               ego (subject to dynamics-mismatch tracking lag).
  COSIM_DEBUG_CARLA_VIEW=/dir  write a CARLA chase-camera mp4 per route (native
                               tick rate, streamed to disk frame-by-frame)
  COSIM_RECORD_INFRACTIONS=/dir  write a short chase-cam clip (last ~5 s) per
                               ego infraction (collision/offroad/red-light),
                               from either the shadow pufferdrive env's own
                               model (_ego_infractions) or real CARLA ground
                               truth (_carla_infractions)
  COSIM_TELEMETRY=/dir         write a per-policy-step CSV per route, with
                               both infraction sources above as columns
  COSIM_OBS_HTML=/dir          write an interactive pufferlib.viz replay per
                               route (the exact obs + policy outputs the ego saw)
  COSIM_DUMP_OBS=/dir          write the raw ego obs + action per policy step
                               per route (.npz: step, obs, action)
  COSIM_WORLD_LOG=/dir         write the bin-frame world state per policy step
                               per route (ego, partners, light states, route);
                               input of scripts/eval/analyze_carla_cosim.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import carla

from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.cosim.goals import RouteGoalWindow, route_goals_from_target_points
from pufferlib.ocean.cosim import carla_bridge as cb
from pufferlib.ocean.cosim.arch import checkpoint_config_path, shadow_env_kwargs
from pufferlib.ocean.cosim.carla.controller import TrackingController, read_vehicle_geometry

# Rolling chase-cam window kept for COSIM_RECORD_INFRACTIONS clips, and the
# minimum ego travel between two logged infractions (suppresses re-triggering
INFRACTION_CLIP_SECONDS = 5.0
INFRACTION_MIN_SEPARATION_M = 10.0
RED_LIGHT_CHECK_DISTANCE_M = 30.0  # matches CaRL RunRedLight's own distance_light default

FAR_AWAY = 1.0e6  # park unused shadow-env agent slots out of observation range
PARTNER_MAX_ABS_DZ_M = 20.0  # CARLA actors farther above/below the ego are hidden scenario props, not traffic
ROAD_RAY_HALF_SPAN_M = 5.0  # vertical ray around the waypoint z; the ego roof and underpass roads are filtered by label/nearest
ROAD_RAY_LABELS = (carla.CityObjectLabel.Roads, carla.CityObjectLabel.RoadLines, carla.CityObjectLabel.Bridge)
MAX_POLICY_STEPS = 100_000  # shadow-env episode cap (~1.4 h at the 0.05 s CARLA tick); sizes the light state buffers
SCENARIO_LENGTH_MARGIN_STEPS = 2
# Shadow-env metrics_array indices (datatypes.h): collision/offroad/red-light flags.
EGO_INFRACTION_METRICS = {"collision": 0, "offroad": 1, "red_light": 2}

CARLA_VIEW_SENSOR_ID = "puffer_chase_cam"
CARLA_VIEW_WIDTH, CARLA_VIEW_HEIGHT, CARLA_VIEW_FOV = 960, 540, 90
# Behind + above the ego, looking forward and down. Closer than
# carla_cosim.py's standalone chase cam (x=-6.5, z=3.2): the leaderboard
# enforces sqrt(x^2+y^2+z^2) <= agent_wrapper.MAX_ALLOWED_RADIUS_SENSOR (3.0 m)
CARLA_VIEW_TRANSFORM = dict(x=-2.0, y=0.0, z=2.1, roll=0.0, pitch=-15.0, yaw=0.0)


def get_entry_point():
    return "PufferAgent"


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


def road_mesh_z(world, x, y, reference_z):
    """Height of the road mesh at (x, y) from a vertical ray, the road hit nearest reference_z; None when none is hit."""
    top = carla.Location(x=x, y=y, z=reference_z + ROAD_RAY_HALF_SPAN_M)
    bottom = carla.Location(x=x, y=y, z=reference_z - ROAD_RAY_HALF_SPAN_M)
    road_z = [hit.location.z for hit in world.cast_ray(top, bottom) if hit.label in ROAD_RAY_LABELS]
    if not road_z:
        return None
    return min(road_z, key=lambda hit_z: abs(hit_z - reference_z))


def road_aligned_attitude(road_up, yaw_deg):
    """(pitch_deg, roll_deg, unit forward) of a body heading yaw_deg that lies flat on the road plane with normal road_up."""
    normal = np.array([road_up.x, road_up.y, road_up.z], dtype=np.float64)
    yaw_rad = math.radians(yaw_deg)
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
    forward -= normal * float(forward @ normal)
    forward /= np.linalg.norm(forward)
    right = np.cross(normal, forward)  # CARLA is left-handed: right = up x forward
    pitch_deg = math.degrees(math.asin(forward[2]))
    roll_deg = math.degrees(math.asin(-right[2]))  # positive roll = right side down
    return pitch_deg, roll_deg, forward


def plan_xyz(plan):
    """[(carla.Transform, RoadOption)] -> (N, 3) CARLA-frame positions."""
    return np.array([[t.location.x, t.location.y, t.location.z] for t, _ in plan], dtype=np.float64).reshape(-1, 3)


class PufferAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        if "SCENARIO_RUNNER_ROOT" not in os.environ:
            raise RuntimeError(
                "SCENARIO_RUNNER_ROOT is not set. Export it to the scenario_runner checkout "
                "(see README.md / run_leaderboard.sh); without it route scenarios silently fail to load."
            )
        self.track = autonomous_agent.Track.MAP
        self.route_index = re.sub(r"[^\w.-]", "_", str(route_index)) if route_index else "route"
        # CaRL's `route_index` is route_date_string = Path(ROUTES).stem, fixed
        # once per evaluator process unless --collect-dataset is passed.
        self.video_tag = f"{self.route_index}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.checkpoint, self.cfg = resolve_checkpoint(path_to_conf_file)

        self.device = os.environ.get("COSIM_DEVICE", "cpu")
        self.dynamics_source = os.environ.get("COSIM_DYNAMICS_SOURCE", "pufferdrive")
        if self.dynamics_source not in ("carla", "pufferdrive"):
            raise ValueError(f"COSIM_DYNAMICS_SOURCE must be 'carla' or 'pufferdrive', got {self.dynamics_source!r}")
        env_cfg = self.cfg["env"]
        # Shadow agent pool == the training per-env cap
        self.num_agents = int(env_cfg["max_agents_per_env"])

        self.debug_carla_view_dir = os.environ.get("COSIM_DEBUG_CARLA_VIEW", None)
        self.record_infractions_dir = os.environ.get("COSIM_RECORD_INFRACTIONS", None)
        self.telemetry_dir = os.environ.get("COSIM_TELEMETRY", None)
        self.telemetry_file = None
        self.obs_html_dir = os.environ.get("COSIM_OBS_HTML", None)
        self.obs_dump_dir = os.environ.get("COSIM_DUMP_OBS", None)  # raw ego obs per policy step (.npz)
        self._obs_dump = []
        self.world_log_dir = os.environ.get("COSIM_WORLD_LOG", None)  # bin-frame world state per policy step (.npz)
        self._world_log = {"ego": [], "partners": [], "lights": []}
        self._last_partners = None
        self._partner_stopped_s = {}  # CARLA actor id -> seconds at standstill
        from pufferlib.ocean.drive import binding

        self._partner_stopped_speed_threshold = float(binding.AGENT_STOPPED_SPEED_THRESHOLD)
        self.obs_html_max_steps = int(os.environ.get("COSIM_OBS_HTML_MAX_STEPS", "12000"))
        self._obs_html = None
        # Per-policy-step CSV of the ego's current goal vs the lane
        # find_goal_lane snapped it to (goal_lane_idx from env.get_state()):
        # lane direction, lane-to-goal distance, dot vs the route direction
        # used at snap time -- diagnoses whether the goal feature is guiding
        # toward a real, correctly-directed lane. Off unless set.
        self.debug_goal_lane_dir = os.environ.get("COSIM_DEBUG_GOAL_LANE", None)
        self._goal_lane_debug_file = None
        self._road_data = None

        self.step = -1
        self.initialized = False
        self.policy = None
        self.carla_view_writer = None
        self.goal_window = None
        self.target = (0.0, 0.0)  # (target_speed, target_yaw_deg), held between policy steps
        # (x, y, yaw_deg, speed) of the shadow ego before and after the current policy step, CARLA frame
        self._motion = None
        self._carla_collision = None  # set by _init_carla_infraction_detectors when needed

    def sensors(self):
        if not self.debug_carla_view_dir and not self.record_infractions_dir:
            return []
        return [
            {
                "type": "sensor.camera.rgb",
                "id": CARLA_VIEW_SENSOR_ID,
                "width": CARLA_VIEW_WIDTH,
                "height": CARLA_VIEW_HEIGHT,
                "fov": CARLA_VIEW_FOV,
                **CARLA_VIEW_TRANSFORM,
            }
        ]

    def _load_policy_and_env(self, town_bin):
        arch = shadow_env_kwargs(
            self.cfg,
            overrides=dict(
                map_dir=town_bin,
                num_maps=1,
                # One policy agent (the ego); every other slot is a static partner streamed from CARLA.
                num_agents=1,
                min_agents_per_env=1,
                cosim_partner_slots=self.num_agents - 1,
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

    def _init_on_first_step(self):
        """Deferred init (the ego and world only exist once the route runs) —
        same pattern as CaRL's eval_agent.agent_init. Everything here is
        read-only with respect to CARLA."""
        self.vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.vehicle.get_world()
        self.cmap = self.world.get_map()
        town = CarlaDataProvider.get_map().name.split("/")[-1]
        self.tick_dt = float(self.world.get_settings().fixed_delta_seconds)  # 0.05 @ 20 Hz
        self.dt = self.tick_dt  # the policy runs every CARLA tick, like the leaderboard's sensor agents

        self.town_bin = cb.bin_path_for_town(town)
        self._load_policy_and_env(self.town_bin)

        self.transform = cb.CarlaTransform(town, offset=cb.town_offset(self.town_bin))
        offset, residual_before, residual_after = cb.calibrate_town_offset(self.cmap, self.transform, self.town_bin)
        print(
            f"[puffer_agent] bin offset calibrated against CARLA lanes: shift "
            f"({offset[0] - self.transform.tx:+.2f}, {offset[1] - self.transform.ty:+.2f}) m, "
            f"median lane residual {residual_before:.2f} -> {residual_after:.2f} m"
        )
        self.transform = cb.CarlaTransform(town, offset=offset)
        if self.dynamics_source == "pufferdrive":
            self._sync_ego_from_carla(zero_velocity=True)
        ego_loc = self.vehicle.get_location()
        self.route_goals = route_goals_from_target_points(
            plan_xyz(self._global_plan_world_coord),
            plan_xyz(self.dense_global_plan_world_coord),
            self.transform.loc_to_bin,
            self.transform.loc_to_bin(ego_loc.x, ego_loc.y),
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
            self._goal_lane_debug_file = open(Path(self.debug_goal_lane_dir) / f"{self.video_tag}.csv", "w")
            self._goal_lane_debug_file.write(
                "step,ego_x,ego_y,goal_x,goal_y,gdir_x,gdir_y,goal_lane_idx,lane_dir_dot_route,lane_dist_to_goal_m\n"
            )

        self.lights = list(self.world.get_actors().filter("traffic.traffic_light"))
        light_map, self.num_traffic = cb.map_lights_to_bin(self.lights, self.transform, self.town_bin)
        # One id -> bin-indices table, reused by both passes in
        # _read_light_states, so they can't disagree on which bin element a
        # given light maps to (a prior independent geometric lookup for the
        # ego-governance pass picked the wrong element ~5% of the time).
        self.light_bin_indices_by_id = {lt.id: light_map[li] for li, lt in enumerate(self.lights)}
        self.last_light_states = np.zeros(self.num_traffic, np.int32)

        wheelbase, max_steer = read_vehicle_geometry(self.vehicle)
        self.wheelbase_m = wheelbase
        self.controller = TrackingController(wheelbase_m=wheelbase, max_steer_rad=max_steer, horizon_s=self.dt)

        if self.obs_html_dir:
            from pufferlib.ocean.cosim.obs_replay import ObsReplayCapture

            Path(self.obs_html_dir).mkdir(parents=True, exist_ok=True)
            self._obs_html = ObsReplayCapture(
                self.env, self.policy, Path(self.obs_html_dir) / self.video_tag, max_steps=self.obs_html_max_steps
            )

        if self.debug_carla_view_dir:
            from pufferlib.ocean.cosim.carla_cosim import Mp4Writer

            Path(self.debug_carla_view_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(self.debug_carla_view_dir) / f"{self.video_tag}.mp4")
            self.carla_view_writer = Mp4Writer(out, fps=round(1.0 / self.tick_dt))

        if self.telemetry_dir:
            Path(self.telemetry_dir).mkdir(parents=True, exist_ok=True)
            self.telemetry_file = open(Path(self.telemetry_dir) / f"{self.video_tag}.csv", "w")
            self.telemetry_file.write(
                "step,current_speed,target_speed,ego_action,goal_cursor,goal_dist_m,"
                "near_light_dist_m,near_light_state,"
                "pd_infr_collision,pd_infr_offroad,pd_infr_red,"
                "carla_infr_collision,carla_infr_offroad,carla_infr_red\n"
            )

        if self.record_infractions_dir:
            from collections import deque

            Path(self.record_infractions_dir).mkdir(parents=True, exist_ok=True)
            # Rolling window of the last INFRACTION_CLIP_SECONDS of chase-cam
            self.infraction_buffer = deque(maxlen=int(INFRACTION_CLIP_SECONDS / self.tick_dt))
            self.infraction_counter = 0
            self.last_infraction_location = self.vehicle.get_location()

        if self.telemetry_dir or self.record_infractions_dir:
            self._init_carla_infraction_detectors()

        print(f"[puffer_agent] town={town} tick_dt={self.tick_dt} dt={self.dt} route_goals={len(self.route_goals)}")
        self.initialized = True

    # --- shadow-env sync (read-only w.r.t. CARLA) --------------------------

    def _nearby_actors(self):
        ego_loc = self.vehicle.get_location()
        candidates = []
        for a in self.world.get_actors():
            if a.id == self.vehicle.id or not ("vehicle" in a.type_id or "walker.pedestrian" in a.type_id):
                continue
            loc = a.get_location()
            # scenario_runner parks pending scenario actors 50-500 m underground at the ego's own waypoint
            if abs(loc.z - ego_loc.z) > PARTNER_MAX_ABS_DZ_M:
                continue
            candidates.append((loc.distance(ego_loc), a))
        candidates.sort(key=lambda item: item[0])
        return [a for _, a in candidates[: self.num_agents - 1]]

    def _read_states(self, actors):
        idx, x, y, z, h, vx, vy, yaw_rate, accel_long, stopped_s = [], [], [], [], [], [], [], [], [], []
        stopped_by_id = {}
        for j, a in enumerate(actors):
            idx.append(1 + j)  # agent 0 = ego; others fill 1..M
            bx, by, bz, bh, bvx, bvy, byr, bal = self.transform.actor_state_to_bin(a)
            x.append(bx)
            y.append(by)
            z.append(bz)
            h.append(bh)
            vx.append(bvx)
            vy.append(bvy)
            yaw_rate.append(byr)
            accel_long.append(bal)
            # stopped time follows the CARLA actor, not the shadow slot (slots reshuffle by distance every step)
            is_stopped = math.hypot(bvx, bvy) <= self._partner_stopped_speed_threshold
            stopped_by_id[a.id] = self._partner_stopped_s.get(a.id, 0.0) + self.dt if is_stopped else 0.0
            stopped_s.append(stopped_by_id[a.id])
        self._partner_stopped_s = stopped_by_id
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

    def _read_sizes(self, actors):
        idx, length, width = [], [], []
        for j, a in enumerate(actors):
            idx.append(1 + j)
            ext = a.bounding_box.extent
            length.append(max(2.0 * ext.x, 0.1))
            width.append(max(2.0 * ext.y, 0.1))
        return (np.array(idx, np.int32), np.array(length, np.float32), np.array(width, np.float32))

    def _read_light_states(self):
        """Ground truth for every mapped bin traffic element, via the
        precomputed light_bin_indices_by_id table (see _init_on_first_step)."""
        # unmapped light elements read OFF (a training state), never UNKNOWN
        states = np.full(self.num_traffic, cb.TRAFFIC_LIGHT_STATE_OFF, np.int32)
        for lt in self.lights:
            state = cb.carla_light_to_puffer(lt.get_state())
            for j in self.light_bin_indices_by_id[lt.id]:
                if 0 <= j < self.num_traffic:
                    states[j] = state
        ego_light = self.vehicle.get_traffic_light()
        if ego_light is not None and ego_light.id in self.light_bin_indices_by_id:
            state = cb.carla_light_to_puffer(ego_light.get_state())
            for j in self.light_bin_indices_by_id[ego_light.id]:
                if 0 <= j < self.num_traffic:
                    states[j] = state
        return states

    def _sync_ego_from_carla(self, zero_velocity=False):
        """Overwrite agent 0 (ego) from CARLA's ground-truth pose/size."""
        ego_state = self.transform.actor_state_to_bin(self.vehicle)
        if zero_velocity:
            x, y, z, heading, *_ = ego_state
            ego_state = (x, y, z, heading, 0.0, 0.0, 0.0, 0.0)
        self.env.set_agent_states(np.array([0], np.int32), *[np.array([v], np.float32) for v in ego_state])
        ego_ext = self.vehicle.bounding_box.extent
        self.env.set_agent_sizes(
            np.array([0], np.int32), np.array([2.0 * ego_ext.x], np.float32), np.array([2.0 * ego_ext.y], np.float32)
        )

    def _sync_carla(self):
        """Overwrite the shadow env's background agents/lights (+ ego, only in
        dynamics_source='carla' mode) from CARLA ground truth and return the
        recomputed observation array (num_agents, obs_dim). Row 0 = ego."""
        if self.dynamics_source == "carla":
            self._sync_ego_from_carla()
        ego = self.env.get_global_agent_state()
        ebx, eby = float(ego["x"][0]), float(ego["y"][0])

        actors = self._nearby_actors()
        if actors:
            states = self._read_states(actors)
            sizes = self._read_sizes(actors)
            self.env.set_agent_states(*states)
            self.env.set_agent_sizes(*sizes)
            is_walker = np.array(["walker" in a.type_id for a in actors], np.float32)
            self._last_partners = (states[1], states[2], states[4], states[5], states[6], sizes[1], sizes[2], is_walker)
        else:
            self._last_partners = None
        n_used = 1 + len(actors)
        if n_used < self.num_agents:
            sp = np.arange(n_used, self.num_agents, dtype=np.int32)
            zf = np.full(len(sp), FAR_AWAY, np.float32)
            zz = np.zeros_like(zf)
            self.env.set_agent_states(sp, zf, zf, zf, zz, zz, zz, zz, zz)

        self.last_light_states = self._read_light_states()
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
            f"{self.step},{ebx:.2f},{eby:.2f},{gx:.2f},{gy:.2f},{gdir_x:.3f},{gdir_y:.3f},"
            f"{lane_idx},{dot:.3f},{dist:.2f}\n"
        )
        self._goal_lane_debug_file.flush()

    def _carla_integrate(self, actions):
        """Step the shadow env ONCE (one policy step == one shadow tick).

        dynamics_source='carla': return the ego's target kinematic state for
        the tracking controller

        dynamics_source='pufferdrive': PufferDrive's own jerk/classic dynamics
        (the ones the policy trained on) ARE the ego's motion -- no target to
        chase. The CARLA ego is teleported to the post-step pose (see
        _teleport_carla_ego), where the next tick's observation reads it.
        """
        if self.step >= MAX_POLICY_STEPS:
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
            self._motion = (x0, y0, yaw0_deg, speed_before, x1, y1, target_yaw_deg, speed_after, float(after["z"][0]))
            self._teleport_carla_ego(1.0)
            return speed_after, target_yaw_deg  # logged, not chased (see run_step)

        target_speed = speed_after + max(accel_after, 0.0) * self.speed_intent_extension_s
        return target_speed, target_yaw_deg

    def _teleport_carla_ego(self, fraction):
        """Place the CARLA ego at `fraction` (0 = pre-step, 1 = post-step) of the current policy step's motion."""
        x0, y0, yaw0_deg, speed0, x1, y1, yaw1_deg, speed1, sim_z = self._motion
        yaw_delta_deg = cb.wrap_deg_180(yaw1_deg - yaw0_deg)
        x = x0 + fraction * (x1 - x0)
        y = y0 + fraction * (y1 - y0)
        yaw_deg = yaw0_deg + fraction * yaw_delta_deg
        speed = speed0 + fraction * (speed1 - speed0)
        # CARLA's live road-mesh height, not the shadow env's lane-averaged sim_z: on graded
        # multi-level roads the average can land the body mid-structure (measured: 20-100+
        # road collisions per Town03/04 route). sim_z only off the drivable network.
        wp = self.cmap.get_waypoint(carla.Location(x=x, y=y))
        z = wp.transform.location.z if wp is not None else sim_z
        road_up = wp.transform.rotation.get_up_vector() if wp is not None else carla.Vector3D(x=0.0, y=0.0, z=1.0)
        pitch_deg, roll_deg, forward = road_aligned_attitude(road_up, yaw_deg)
        # Waypoint z is quantised in ~0.5 m steps on steep grades (measured 0.32 m low at a Town03
        # descent); the mesh raycast is exact and the axle chord follows crests the waypoint pitch misses.
        surface = self._road_surface(x, y, z, yaw_deg)
        if surface is not None:
            z, pitch_deg = surface
            pitch_rad = math.radians(pitch_deg)
            yaw_rad = math.radians(yaw_deg)
            forward = (math.cos(yaw_rad) * math.cos(pitch_rad), math.sin(yaw_rad) * math.cos(pitch_rad), math.sin(pitch_rad))
        # Zero momentum before the teleport: CARLA's collision resolver reacts violently to a
        # physics body carrying velocity into a new pose (carla issue #8076).
        zero = carla.Vector3D(x=0.0, y=0.0, z=0.0)
        self.vehicle.set_target_velocity(zero)
        self.vehicle.set_target_angular_velocity(zero)
        self.vehicle.set_transform(
            carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch_deg, yaw=yaw_deg, roll=roll_deg))
        )
        self.vehicle.set_target_velocity(
            carla.Vector3D(x=speed * forward[0], y=speed * forward[1], z=speed * forward[2])
        )
        self.vehicle.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=yaw_delta_deg / self.dt))

    def _road_surface(self, x, y, reference_z, yaw_deg):
        """(z, pitch_deg) of the road mesh under the ego: centre height plus the front-to-rear axle chord, None off the mesh."""
        yaw_rad = math.radians(yaw_deg)
        half_wheelbase = 0.5 * self.wheelbase_m
        dx, dy = half_wheelbase * math.cos(yaw_rad), half_wheelbase * math.sin(yaw_rad)
        z_centre = road_mesh_z(self.world, x, y, reference_z)
        z_front = road_mesh_z(self.world, x + dx, y + dy, reference_z)
        z_rear = road_mesh_z(self.world, x - dx, y - dy, reference_z)
        if z_centre is None or z_front is None or z_rear is None:
            return None
        return z_centre, math.degrees(math.atan2(z_front - z_rear, self.wheelbase_m))

    def _ego_speed(self):
        return float(np.asarray(self.env.observations)[0][0]) * self._max_speed()

    def _max_speed(self):
        return self.env.obs_norm_speed_mps

    def _accel_long_norm(self):
        from pufferlib.ocean.drive import binding

        return binding.ACCEL_LONG_NORM

    def _ego_infractions(self):
        """{'collision': f, 'offroad': f, 'red_light': f} from the shadow
        pufferdrive env's own model of the ego (compute_metrics, refreshed by
        the last integrate()) -- what the policy would have caused according
        to PufferDrive's own dynamics, not necessarily what the real CARLA
        vehicle did (see _carla_infractions)."""
        state = self.env.get_state()
        scenario = state[0] if isinstance(state, list) else state
        agents = scenario.get("agents") or []
        if not agents:
            return {name: 0.0 for name in EGO_INFRACTION_METRICS}
        metrics = agents[0].get("metrics_array") or []
        return {
            name: float(metrics[idx]) if idx < len(metrics) else 0.0 for name, idx in EGO_INFRACTION_METRICS.items()
        }

    def _init_carla_infraction_detectors(self):
        """Real CARLA ground truth for collision/offroad/red-light, reusing
        CaRL's own standalone criteria (CARL_WORK_DIR/team_code/reward).
        RunRedLight uses _get_traffic_light_waypoints and do
        RunRedLight's rear-bumper-crossing check in _poll_carla_red_light."""
        if "CARL_WORK_DIR" not in os.environ:
            raise RuntimeError(
                "CARL_WORK_DIR is not set; required to import CaRL's reward.criteria "
                "collision/offroad detectors for COSIM_TELEMETRY/COSIM_RECORD_INFRACTIONS."
            )
        team_code_dir = str(Path(os.environ["CARL_WORK_DIR"]) / "team_code")
        if team_code_dir not in sys.path:
            sys.path.insert(0, team_code_dir)
        from reward.criteria.collision import Collision
        from reward.criteria.outside_route_lanes import OutsideRouteLanesTest
        from birds_eye_view.traffic_light import _get_traffic_light_waypoints

        self._carla_collision = Collision(self.vehicle, self.world)
        self._carla_collision_pending = False
        self._carla_offroad_test = OutsideRouteLanesTest(self.vehicle, self.cmap)

        self._red_light_geometry = {}
        for lt in self.lights:
            tv_loc, stopline_wps, _stopline_vertices, long_line, _junction_paths = _get_traffic_light_waypoints(
                lt, self.cmap
            )
            self._red_light_geometry[lt.id] = (tv_loc, list(zip(stopline_wps, long_line)))
        self._last_red_light_id = None
        self._ran_red_light_pending = False

    def _poll_carla_collision(self):
        elapsed = self.world.get_snapshot().timestamp.elapsed_seconds
        if self._carla_collision.tick(self.vehicle, elapsed) is not None:
            self._carla_collision_pending = True

    def _poll_carla_red_light(self):
        """Real CARLA ground truth for 'ran a red light'"""
        import shapely.geometry

        ev_tra = self.vehicle.get_transform()
        ev_dir = ev_tra.get_forward_vector()
        ev_extent = self.vehicle.bounding_box.extent.x
        tail_close = ev_tra.transform(carla.Location(x=-0.8 * ev_extent))
        tail_far = ev_tra.transform(carla.Location(x=-ev_extent - 1.0))
        tail_line = shapely.geometry.LineString([(tail_close.x, tail_close.y), (tail_far.x, tail_far.y)])
        speed = self.vehicle.get_velocity().length()
        if speed <= 0.001:
            return
        for lt in self.lights:
            if lt.get_state() != carla.TrafficLightState.Red or lt.id == self._last_red_light_id:
                continue
            tv_loc, crossings = self._red_light_geometry[lt.id]
            if tv_loc.distance(ev_tra.location) > RED_LIGHT_CHECK_DISTANCE_M:
                continue
            for wp, (stop_left, stop_right) in crossings:
                wp_dir = wp.transform.get_forward_vector()
                if ev_dir.x * wp_dir.x + ev_dir.y * wp_dir.y + ev_dir.z * wp_dir.z <= 0:
                    continue  # this light's stop line doesn't face our lane
                stop_line = shapely.geometry.LineString([(stop_left.x, stop_left.y), (stop_right.x, stop_right.y)])
                if not tail_line.intersection(stop_line).is_empty:
                    self._last_red_light_id = lt.id
                    self._ran_red_light_pending = True
                    return

    def _carla_infractions(self):
        """{'collision': f, 'offroad': f, 'red_light': f} from real CARLA
        ground truth (_init_carla_infraction_detectors) -- contrast with
        _ego_infractions, which is the shadow env's model of the ego."""
        collision, self._carla_collision_pending = self._carla_collision_pending, False
        red_light, self._ran_red_light_pending = self._ran_red_light_pending, False
        self._carla_offroad_test.update()
        offroad = self._carla_offroad_test.outside_lane_active or self._carla_offroad_test.wrong_lane_active
        return {"collision": float(collision), "offroad": float(offroad), "red_light": float(red_light)}

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
                self.step,
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
        dense = np.array(
            [self.transform.loc_to_bin(t.location.x, t.location.y) for t, _ in self.dense_global_plan_world_coord],
            np.float32,
        )
        partners = self._world_log["partners"]
        np.savez_compressed(
            Path(self.world_log_dir) / f"{self.video_tag}.npz",
            ego=np.array(self._world_log["ego"], np.float32),
            partners=np.concatenate(partners) if partners else np.zeros((0, 8), np.float32),
            lights=np.stack(self._world_log["lights"]) if self._world_log["lights"] else np.zeros((0, 0), np.int8),
            route_goals=self.route_goals,
            dense_route=dense,
            meta=json.dumps(
                {
                    "town": Path(self.town_bin).stem.split("__")[-1],
                    "town_bin": self.town_bin,
                    "dt": self.dt,
                    "tick_dt": self.tick_dt,
                    "offset": [self.transform.tx, self.transform.ty],
                }
            ),
        )

    def _write_telemetry_row(self, ego_action, pd_flags, carla_flags):
        """One CSV row per policy step: what the loop commanded vs achieved,
        plus the nearest mapped light's state and both infraction sources
        (pd_* = shadow pufferdrive env, carla_* = real CARLA ground truth)."""
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
            else self.vehicle.get_velocity().length()
        )
        self.telemetry_file.write(
            f"{self.step},{current_speed:.3f},{self.target[0]:.3f},"
            f"{ego_action},{self.goal_window.current_index},{goal_dist:.1f},{near_dist:.1f},{near_state},"
            f"{pd_flags['collision']:.0f},{pd_flags['offroad']:.0f},{pd_flags['red_light']:.0f},"
            f"{carla_flags['collision']:.0f},{carla_flags['offroad']:.0f},{carla_flags['red_light']:.0f}\n"
        )

    def _maybe_save_infraction_clip(self, pd_flags, carla_flags):
        """Dump the rolling chase-cam buffer as one mp4 when either infraction
        source flags an ego infraction (collision/offroad/red-light), at most
        once per INFRACTION_MIN_SEPARATION_M of ego travel (CaRL eval_agent-
        style)."""
        fired = [f"pd_{name}" for name, value in pd_flags.items() if value > 0.0]
        fired += [f"carla_{name}" for name, value in carla_flags.items() if value > 0.0]
        if not fired or not self.infraction_buffer:
            return
        location = self.vehicle.get_location()
        if location.distance(self.last_infraction_location) <= INFRACTION_MIN_SEPARATION_M:
            return
        from pufferlib.ocean.cosim.carla_cosim import write_mp4

        out = str(
            Path(self.record_infractions_dir) / f"{self.video_tag}_{'_'.join(fired)}_{self.infraction_counter:02d}.mp4"
        )
        write_mp4(out, list(self.infraction_buffer), fps=round(1.0 / self.tick_dt))
        print(f"[puffer_agent] infraction {fired} -> {out}")
        self.infraction_counter += 1
        self.last_infraction_location = location

    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1
        if not self.initialized:
            self._init_on_first_step()
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

        if CARLA_VIEW_SENSOR_ID in input_data:
            _, bgra = input_data[CARLA_VIEW_SENSOR_ID]  # (H, W, 4) uint8, leaderboard's CallBack format
            rgb = bgra[:, :, [2, 1, 0]]  # BGRA -> RGB
            if self.carla_view_writer is not None:
                self.carla_view_writer.append_data(rgb)  # streamed to disk
            if self.record_infractions_dir:
                self.infraction_buffer.append(rgb.copy())

        if self._carla_collision is not None:
            self._poll_carla_collision()
            self._poll_carla_red_light()  # filters single-tick get_traffic_light() noise

        obs = self._sync_carla()  # shadow env <- CARLA ground truth
        actions, aux = self._policy_actions(obs)
        if self.obs_dump_dir:
            self._obs_dump.append((self.step, obs[0].astype(np.float32), actions[0].astype(np.float32)))
        if self.world_log_dir:
            self._record_world_state(obs)
        if self._obs_html is not None:
            self._obs_html.capture(obs, actions, aux, aux.get("action_index"))
        self.target = self._carla_integrate(actions)  # policy intent, one dt ahead
        if self.telemetry_file is not None or self.record_infractions_dir:
            # _init_carla_infraction_detectors ran for either dir being set
            pd_flags, carla_flags = self._ego_infractions(), self._carla_infractions()
            if self.telemetry_file is not None:
                self._write_telemetry_row(float(actions[0, 0]), pd_flags, carla_flags)
            if self.record_infractions_dir:
                self._maybe_save_infraction_clip(pd_flags, carla_flags)

        if self.dynamics_source == "pufferdrive":
            return carla.VehicleControl()

        # Controller runs every tick against the latest CARLA state, chasing the
        # target held from the last policy step.
        current_speed = self.vehicle.get_velocity().length()
        current_yaw = self.vehicle.get_transform().rotation.yaw
        target_speed, target_yaw = self.target
        return self.controller.step(current_speed, current_yaw, target_speed, target_yaw, self.tick_dt)

    def destroy(self, results=None):
        if not self.initialized:
            return
        print(
            f"[puffer_agent] route done: goals {self.goal_window.current_index + 1}/{len(self.route_goals)}, "
            f"tracking {self.controller.stats()}"
        )
        if self.carla_view_writer is not None:
            self.carla_view_writer.close()
            print(f"[puffer_agent] wrote CARLA chase-cam video for route {self.video_tag}")
        if self.telemetry_file is not None:
            self.telemetry_file.close()
        if self._carla_collision is not None:
            self._carla_collision.clean()  # stop+destroy the spawned collision sensor actor
        if self._goal_lane_debug_file is not None:
            self._goal_lane_debug_file.close()
        if self._obs_html is not None:
            html = self._obs_html.write()
            print(f"[puffer_agent] wrote obs_html viewer ({len(self._obs_html)} frames) -> {html}")
        if self.world_log_dir and self._world_log["ego"]:
            self._write_world_log()
        if self.obs_dump_dir and self._obs_dump:
            Path(self.obs_dump_dir).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                Path(self.obs_dump_dir) / f"{self.video_tag}.npz",
                step=np.array([s for s, _, _ in self._obs_dump], np.int32),
                obs=np.stack([o for _, o, _ in self._obs_dump]),
                action=np.stack([a for _, _, a in self._obs_dump]),
            )
        self.env.close()
        self.initialized = False
