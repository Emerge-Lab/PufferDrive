"""PufferDrive policy as a CARLA leaderboard agent (CaRL original_leaderboard).

Usage (CaRL env vars/paths as in CaRL/CARLA/README.md, plus PufferDrive repo
root on PYTHONPATH for `pufferlib` and `data_utils`):

  python ${CARL_WORK_DIR}/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py \
      --routes ${CARL_WORK_DIR}/custom_leaderboard/leaderboard/data/longest6_split/longest6_00.xml \
      --agent /path/to/pufferlib/ocean/cosim/carla/leaderboard_agent.py \
      --agent-config /path/to/experiments/puffer_drive_xxx/models/model_xxx.pt \
      --checkpoint /path/to/results/result.json --track MAP

Everything about the shadow env (obs layout, dynamics, dt, goal spacing, agent
pool size, ...) comes from the checkpoint's sibling config.yaml, with the
clean-eval profile applied on top (cosim/arch.py CLEAN_EVAL_OVERRIDES); only
the structural co-sim keys (map, pool wiring) are set here.

Environment variables:
  SCENARIO_RUNNER_ROOT         REQUIRED: path to the scenario_runner checkout
                               (route scenarios silently fail to load without it)
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
  COSIM_DEBUG_BEV=/dir         write a top-down BEV mp4 of the shadow env per route
  COSIM_DEBUG_CARLA_VIEW=/dir  write a CARLA chase-camera mp4 per route (native
                               tick rate, streamed to disk frame-by-frame)
  COSIM_RECORD_INFRACTIONS=/dir  write a short chase-cam clip (last ~5 s) per
                               shadow-env-detected ego infraction
                               (collision/offroad/red-light), CaRL
                               eval_agent-style; off when unset
  COSIM_TELEMETRY=/dir         write a per-policy-step CSV per route
  COSIM_OBS_HTML=/dir          write an interactive pufferlib.viz replay per
                               route (the exact obs + policy outputs the ego saw)
"""

import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import carla

from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.cosim import carla_bridge as cb
from pufferlib.ocean.cosim.arch import shadow_env_kwargs
from pufferlib.ocean.cosim.carla.controller import TrackingController, read_vehicle_geometry


def _wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0

# Rolling chase-cam window kept for COSIM_RECORD_INFRACTIONS clips, and the
# minimum ego travel between two logged infractions (suppresses re-triggering
INFRACTION_CLIP_SECONDS = 5.0
INFRACTION_MIN_SEPARATION_M = 10.0

FAR_AWAY = 1.0e6  # park unused shadow-env agent slots out of observation range
EGO_LIGHT_OVERRIDE_RANGE_M = 45.0  # ground-truth ego-light override reach (see _read_light_states)
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
        ckpt, cfg_path = p, p.parents[1] / "config.yaml"
    else:
        models = sorted((p / "models").glob("*.pt")) or sorted(p.glob("*.pt"))
        if not models:
            raise FileNotFoundError(f"no .pt checkpoint under {p}")
        ckpt, cfg_path = models[-1], p / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    return str(ckpt), cfg


def route_goals_from_plan(dense_route, transform, min_goal_spacing, max_goal_spacing):
    """Fixed (N, 5) sequence of route goals in the bin frame -- columns
    (x, y, z, dir_x, dir_y) -- from the leaderboard's dense global plan
    ([(carla.Transform, RoadOption)], ~1 m spacing, already lane-centered).
    Successive goals are spaced by arc length drawn uniformly from
    [min_goal_spacing, max_goal_spacing], matching the training env's gigaflow
    goal placement (drive.h samples random_uniform(min_goal_spacing,
    max_goal_spacing) per goal)"""
    pts = np.array([[t.location.x, t.location.y, t.location.z] for t, _ in dense_route], dtype=np.float64)
    rng = np.random.default_rng(0)  # fixed seed: deterministic goals per route

    def goal_at(i):
        bx, by = transform.loc_to_bin(pts[i, 0], pts[i, 1])
        k = max(i - 1, 0)
        dx = pts[i, 0] - pts[k, 0]
        dy = pts[i, 1] - pts[k, 1]  # CARLA frame; y flips into the bin frame
        return (bx, by, pts[i, 2], dx, -dy)

    goals, cum = [], 0.0
    next_at = rng.uniform(min_goal_spacing, max_goal_spacing)
    for i in range(1, len(pts)):
        cum += float(np.hypot(pts[i, 0] - pts[i - 1, 0], pts[i, 1] - pts[i - 1, 1]))
        if cum >= next_at:
            goals.append(goal_at(i))
            next_at += rng.uniform(min_goal_spacing, max_goal_spacing)
    goals.append(goal_at(len(pts) - 1))
    return np.array(goals, np.float32)


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
        self.dt = float(env_cfg["dt"])
        # Shadow agent pool == the training per-env cap
        self.num_agents = int(env_cfg["max_agents_per_env"])
        self.min_goal_spacing = float(env_cfg["min_goal_spacing"])
        self.max_goal_spacing = float(env_cfg["max_goal_spacing"])
        self.goal_radius = float(env_cfg["goal_radius"])

        self.debug_bev_dir = os.environ.get("COSIM_DEBUG_BEV", None)
        self.debug_carla_view_dir = os.environ.get("COSIM_DEBUG_CARLA_VIEW", None)
        self.record_infractions_dir = os.environ.get("COSIM_RECORD_INFRACTIONS", None)
        self.telemetry_dir = os.environ.get("COSIM_TELEMETRY", None)
        self.telemetry_file = None
        self.obs_html_dir = os.environ.get("COSIM_OBS_HTML", None)
        self.obs_html_max_steps = int(os.environ.get("COSIM_OBS_HTML_MAX_STEPS", "800"))
        self._obs_html = None

        self.step = -1
        self.initialized = False
        self.policy = None
        self.bev = None
        self.carla_view_writer = None
        self.goal_cursor = 0
        self.target = (0.0, 0.0)  # (target_speed, target_yaw_deg), held between policy steps
        # [x, y, yaw_deg, speed, yaw_rate_deg_s, accel_long] in the CARLA frame:
        # the ego's post-step state, advanced tick-by-tick between policy steps
        # to smooth the CARLA-side motion (display only -- see run_step).
        self._display = None

    def sensors(self):
        if not self.debug_carla_view_dir and not self.record_infractions_dir:
            return []
        return [{"type": "sensor.camera.rgb", "id": CARLA_VIEW_SENSOR_ID,
                 "width": CARLA_VIEW_WIDTH, "height": CARLA_VIEW_HEIGHT, "fov": CARLA_VIEW_FOV,
                 **CARLA_VIEW_TRANSFORM}]

    def _load_policy_and_env(self, town_bin):
        arch = shadow_env_kwargs(
            self.cfg,
            overrides=dict(
                map_dir=town_bin, num_maps=1, num_agents=self.num_agents,
                scenario_length=10_000_000, resample_frequency=0,
                termination_mode=0,
                # Enforcement off (detection flags still fire): in one endless
                # episode a "stop" latch is permanent and fires only AFTER
                # CARLA scored the infraction (route 5 froze 320 s -> DNF).
                collision_behavior="ignore", offroad_behavior="ignore",
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

        sd = clean_policy_state_dict(torch.load(self.checkpoint, map_location=self.device, weights_only=False))
        env_for_policy = self.env
        self._boundary_feat_ckpt = None
        bw = sd.get("actor_backbone.boundary_encoder.0.weight")
        if bw is not None and bw.shape[1] < env_for_policy.boundary_features:
            # Older checkpoints emit fewer boundary features than this drive.h
            # (e.g. 7 vs 9 -- two zero-padded GPS columns were added later).
            import copy

            self._boundary_feat_ckpt = int(bw.shape[1])
            env_for_policy = copy.copy(self.env)
            env_for_policy.boundary_features = self._boundary_feat_ckpt
            print(f"[puffer_agent] checkpoint uses {self._boundary_feat_ckpt} boundary features "
                  f"(env emits {self.env.boundary_features}); stripping zero-padded columns")

        policy_cls = getattr(drive_torch, self.cfg.get("policy_name", "Drive"))
        self.policy = policy_cls(env_for_policy, **self.cfg["policy"]).to(self.device)
        self.policy.load_state_dict(sd)
        self.policy.eval()
        print(f"[puffer_agent] loaded policy from {self.checkpoint}")

    def _init_on_first_step(self):
        """Deferred init (the ego and world only exist once the route runs) —
        same pattern as CaRL's eval_agent.agent_init. Everything here is
        read-only with respect to CARLA."""
        self.vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.vehicle.get_world()
        town = CarlaDataProvider.get_map().name.split("/")[-1]
        self.tick_dt = float(self.world.get_settings().fixed_delta_seconds)  # 0.05 @ 20 Hz
        self.action_repeat = max(1, round(self.dt / self.tick_dt))

        self.town_bin = cb.bin_path_for_town(town)
        self._load_policy_and_env(self.town_bin)

        self.transform = cb.CarlaTransform(town, offset=cb.town_offset(self.town_bin))
        if self.dynamics_source == "pufferdrive":
            self._sync_ego_from_carla(zero_velocity=True)
            self._last_target_yaw_deg = self.vehicle.get_transform().rotation.yaw
        self.route_goals = route_goals_from_plan(
            self.dense_global_plan_world_coord, self.transform,
            min_goal_spacing=self.min_goal_spacing, max_goal_spacing=self.max_goal_spacing)

        import data_utils.mirror_map_bin as mbin

        bin_traffic = mbin.read_bin(Path(self.town_bin))["traffic"]
        self.stop_line_centers = np.array(
            [[0.5 * (t["stop_line"][0] + t["stop_line"][3]),
              0.5 * (t["stop_line"][1] + t["stop_line"][4])] for t in bin_traffic]
        ).reshape(-1, 2)

        self.lights = list(self.world.get_actors().filter("traffic.traffic_light"))
        self.light_map, self.num_traffic = cb.map_lights_to_bin(self.lights, self.transform, self.town_bin)
        self.last_light_states = np.zeros(self.num_traffic, np.int32)

        wheelbase, max_steer = read_vehicle_geometry(self.vehicle)
        self.controller = TrackingController(wheelbase_m=wheelbase, max_steer_rad=max_steer, horizon_s=self.dt)

        if self.debug_bev_dir:
            from pufferlib.ocean.cosim.carla_cosim import BEVRenderer

            Path(self.debug_bev_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(self.debug_bev_dir) / f"{self.video_tag}.mp4")
            self.bev = BEVRenderer(self.town_bin, out)

        if self.obs_html_dir:
            from pufferlib.ocean.drive import binding

            Path(self.obs_html_dir).mkdir(parents=True, exist_ok=True)
            state = self.env.get_state()
            scenario = state[0] if isinstance(state, list) else state
            self._obs_html = {
                "scenario": scenario,
                "agent_cap": int(scenario["num_total_agents"]),
                "traffic_cap": max(int(scenario["num_traffic_elements"]), 1),
                "frames": {key: [] for key in (
                    "agent_f32", "agent_i32", "metrics_f32", "puffer_f32",
                    "traffic_i16", "obs", "raw_action", "value", "entropy", "policy_probs")},
            }

        if self.debug_carla_view_dir:
            import imageio

            Path(self.debug_carla_view_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(self.debug_carla_view_dir) / f"{self.video_tag}.mp4")
            self.carla_view_writer = imageio.get_writer(
                out, fps=round(1.0 / self.tick_dt), codec="libx264", macro_block_size=1)

        if self.telemetry_dir:
            Path(self.telemetry_dir).mkdir(parents=True, exist_ok=True)
            self.telemetry_file = open(Path(self.telemetry_dir) / f"{self.video_tag}.csv", "w")
            self.telemetry_file.write(
                "step,current_speed,target_speed,ego_action,goal_cursor,goal_dist_m,"
                "near_light_dist_m,near_light_state,infr_collision,infr_offroad,infr_red\n")

        if self.record_infractions_dir:
            from collections import deque

            Path(self.record_infractions_dir).mkdir(parents=True, exist_ok=True)
            # Rolling window of the last INFRACTION_CLIP_SECONDS of chase-cam
            self.infraction_buffer = deque(maxlen=int(INFRACTION_CLIP_SECONDS / self.tick_dt))
            self.infraction_counter = 0
            self.last_infraction_location = self.vehicle.get_location()

        print(f"[puffer_agent] town={town} tick_dt={self.tick_dt} dt={self.dt} "
              f"action_repeat={self.action_repeat} route_goals={len(self.route_goals)}")
        self.initialized = True

    # --- shadow-env sync (read-only w.r.t. CARLA) --------------------------

    def _nearby_actors(self):
        ego_loc = self.vehicle.get_location()
        actors = [
            a for a in self.world.get_actors()
            if a.id != self.vehicle.id and ("vehicle" in a.type_id or "walker.pedestrian" in a.type_id)
        ]
        actors.sort(key=lambda a: a.get_location().distance(ego_loc))
        return actors[: self.num_agents - 1]

    def _read_states(self, actors):
        idx, x, y, z, h, vx, vy, yaw_rate, accel_long = [], [], [], [], [], [], [], [], []
        for j, a in enumerate(actors):
            idx.append(1 + j)  # agent 0 = ego; others fill 1..M
            bx, by, bz, bh, bvx, bvy, byr, bal = self.transform.actor_state_to_bin(a)
            x.append(bx); y.append(by); z.append(bz); h.append(bh); vx.append(bvx); vy.append(bvy)
            yaw_rate.append(byr); accel_long.append(bal)
        return (np.array(idx, np.int32), np.array(x, np.float32), np.array(y, np.float32),
                np.array(z, np.float32), np.array(h, np.float32),
                np.array(vx, np.float32), np.array(vy, np.float32),
                np.array(yaw_rate, np.float32), np.array(accel_long, np.float32))

    def _read_sizes(self, actors):
        idx, length, width = [], [], []
        for j, a in enumerate(actors):
            idx.append(1 + j)
            ext = a.bounding_box.extent
            length.append(max(2.0 * ext.x, 0.1)); width.append(max(2.0 * ext.y, 0.1))
        return (np.array(idx, np.int32), np.array(length, np.float32), np.array(width, np.float32))

    def _read_light_states(self):
        states = np.zeros(self.num_traffic, np.int32)
        for li, lt in enumerate(self.lights):
            state = cb.carla_light_to_puffer(lt.get_state())
            for j in self.light_map[li]:
                if 0 <= j < self.num_traffic:
                    states[j] = state
        ego_light = self.vehicle.get_traffic_light()
        if ego_light is not None and len(self.stop_line_centers):
            ex, ey, _, heading, *_ = self.transform.actor_state_to_bin(self.vehicle)
            dx = self.stop_line_centers[:, 0] - ex
            dy = self.stop_line_centers[:, 1] - ey
            dist = np.hypot(dx, dy)
            ahead = (dx * np.cos(heading) + dy * np.sin(heading)) > 0.0
            candidates = np.nonzero(ahead & (dist <= EGO_LIGHT_OVERRIDE_RANGE_M))[0]
            if len(candidates):
                j = int(candidates[dist[candidates].argmin()])
                states[j] = cb.carla_light_to_puffer(ego_light.get_state())
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
            np.array([0], np.int32),
            np.array([2.0 * ego_ext.x], np.float32), np.array([2.0 * ego_ext.y], np.float32))

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
            self.env.set_agent_states(*self._read_states(actors))
            self.env.set_agent_sizes(*self._read_sizes(actors))
        n_used = 1 + len(actors)
        if n_used < self.num_agents:
            sp = np.arange(n_used, self.num_agents, dtype=np.int32)
            zf = np.full(len(sp), FAR_AWAY, np.float32)
            zz = np.zeros_like(zf)
            self.env.set_agent_states(sp, zf, zf, zf, zz, zz, zz, zz, zz)

        self.last_light_states = self._read_light_states()
        self.env.set_traffic_light_states(self.last_light_states)

        # Route goals: advance the cursor only once the ego actually reaches
        # the current goal, then feed the next few goals.
        while (self.goal_cursor < len(self.route_goals) - 1
               and np.hypot(self.route_goals[self.goal_cursor, 0] - ebx,
                            self.route_goals[self.goal_cursor, 1] - eby) < self.goal_radius):
            self.goal_cursor += 1
        sel = self.route_goals[
            [min(self.goal_cursor + k, len(self.route_goals) - 1) for k in range(self.env.num_goals)]
        ]
        self.env.set_agent_goals(0, sel[:, 0].copy(), sel[:, 1].copy(), sel[:, 2].copy(),
                                 sel[:, 3].copy(), sel[:, 4].copy())
        return np.asarray(self.env.recompute_observations())

    def _carla_integrate(self, actions):
        """Step the shadow env ONCE (one policy step == one shadow tick).

        dynamics_source='carla': return the ego's target kinematic state for
        the tracking controller

        dynamics_source='pufferdrive': PufferDrive's own jerk/classic dynamics
        (the ones the policy trained on) ARE the ego's motion -- no target to
        chase. Teleport the CARLA ego to match the shadow env's post-step pose
        """
        self.env.step(np.asarray(actions, dtype=np.int32))
        after = self.env.get_global_agent_state()
        target_yaw_deg = self.transform.bin_heading_to_yaw(float(after["heading"][0]))
        ego_obs = np.asarray(self.env.observations)[0]
        speed_after = float(ego_obs[0]) * self._max_speed()
        accel_after = float(ego_obs[4]) * self._accel_long_norm()

        if self.dynamics_source == "pufferdrive":
            ex, ey = self.transform.bin_to_loc(float(after["x"][0]), float(after["y"][0]))
            # Zero the physics body's velocity/angular velocity BEFORE the
            # teleport: a physics-active actor carries whatever momentum it
            # had into the new position, and CARLA's collision resolver reacts
            # violently to the resulting interpenetration/discontinuity
            # (github.com/carla-simulator/carla/issues/8076 ).
            zero = carla.Vector3D(x=0.0, y=0.0, z=0.0)
            self.vehicle.set_target_velocity(zero)
            self.vehicle.set_target_angular_velocity(zero)
            self.vehicle.set_transform(carla.Transform(
                carla.Location(x=ex, y=ey, z=0.3), carla.Rotation(yaw=target_yaw_deg)))
            yaw_rad = math.radians(target_yaw_deg)
            self.vehicle.set_target_velocity(
                carla.Vector3D(x=speed_after * math.cos(yaw_rad), y=speed_after * math.sin(yaw_rad), z=0.0))
            yaw_rate_deg_s = _wrap_deg(target_yaw_deg - self._last_target_yaw_deg) / self.dt
            self.vehicle.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=yaw_rate_deg_s))
            self._last_target_yaw_deg = target_yaw_deg
            self._display = [ex, ey, target_yaw_deg, speed_after, yaw_rate_deg_s, accel_after]
            return speed_after, target_yaw_deg  # logged, not chased (see run_step)

        target_speed = speed_after + max(accel_after, 0.0) * self.speed_intent_extension_s
        return target_speed, target_yaw_deg

    def _max_speed(self):
        from pufferlib.ocean.drive import binding

        return binding.MAX_SPEED

    def _accel_long_norm(self):
        from pufferlib.ocean.drive import binding

        return binding.ACCEL_LONG_NORM

    def _ego_infractions(self):
        """{'collision': f, 'offroad': f, 'red_light': f} from the shadow env's
        own infraction detectors (compute_metrics, refreshed by the last
        integrate())."""
        state = self.env.get_state()
        scenario = state[0] if isinstance(state, list) else state
        agents = scenario.get("agents") or []
        if not agents:
            return {name: 0.0 for name in EGO_INFRACTION_METRICS}
        metrics = agents[0].get("metrics_array") or []
        return {name: float(metrics[idx]) if idx < len(metrics) else 0.0
                for name, idx in EGO_INFRACTION_METRICS.items()}

    # --- policy + capture ---------------------------------------------------

    def _adapt_obs_for_policy(self, obs):
        """Strip the trailing zero-padded boundary columns when the checkpoint
        predates them (see _load_policy_and_env). No-op otherwise."""
        if self._boundary_feat_ckpt is None:
            return obs
        env = self.env
        n_slots = env.obs_slots_boundary_kept
        feat_env, feat_ckpt = env.boundary_features, self._boundary_feat_ckpt
        b0 = (env.ego_features + env.num_reward_coefs + env.goal_dim
              + env.obs_slots_partners_n * env.partner_features
              + env.obs_slots_lane_kept * env.lane_features)
        b1 = b0 + n_slots * feat_env
        obs = np.asarray(obs)
        boundary = obs[:, b0:b1].reshape(obs.shape[0], n_slots, feat_env)[:, :, :feat_ckpt]
        return np.concatenate([obs[:, :b0], boundary.reshape(obs.shape[0], -1), obs[:, b1:]], axis=1)

    def _policy_actions(self, obs):
        """-> (actions (num_agents, act_dim) int32, aux dict). aux carries
        per-agent value/entropy/action-probs/pool for the obs_html capture"""
        import torch
        import pufferlib.pytorch

        obs = self._adapt_obs_for_policy(obs)
        with torch.no_grad():
            logits, value = self.policy.forward_eval(torch.as_tensor(obs).to(self.device))
            action, _, entropy = pufferlib.pytorch.sample_logits(logits, deterministic=True)
        actions = action.cpu().numpy().reshape(self.num_agents, -1).astype(np.int32)
        aux = {}
        if self._obs_html is not None:
            probs = torch.softmax(logits if isinstance(logits, torch.Tensor) else logits[0], dim=-1)
            aux = {
                "value": value.cpu().numpy().reshape(-1).astype(np.float32),
                "entropy": entropy.cpu().numpy().reshape(-1).astype(np.float32),
                "policy_probs": probs.cpu().numpy().astype(np.float32),
            }
            # Max-pool win counts per obs slot: the viewer shades the
            # ego-centric observation by them, showing which observed
            # elements the encoders actually attended to.
            pool_method = getattr(self.policy, "pool_slot_counts", None)
            if pool_method is not None:
                with torch.no_grad():
                    pool = pool_method(torch.as_tensor(obs).to(self.device))
                aux["pool"] = {k: v.cpu().numpy().astype(np.int16) for k, v in pool.items()}
        return actions, aux

    def _capture_obs_html_frame(self, obs, actions, aux):
        """Mirrors benchmark/evaluators/base.py's
        _render_pass_obs capture loop for a single env."""
        from pufferlib.ocean.drive import binding

        oh = self._obs_html
        cap, tcap = oh["agent_cap"], oh["traffic_cap"]
        agent_f32 = np.zeros((1, cap, binding.AGENT_F32_FIELDS), dtype=np.float32)
        agent_i32 = np.zeros((1, cap, binding.AGENT_I32_FIELDS), dtype=np.int32)
        metrics_f32 = np.zeros((1, cap, binding.METRICS_F32_FIELDS), dtype=np.float32)
        puffer_f32 = np.zeros((1, cap, binding.SCORE_F32_FIELDS), dtype=np.float32)
        traffic_i16 = np.zeros((1, tcap, binding.TRAFFIC_I16_FIELDS), dtype=np.int16)
        self.env.get_obs_html_frame(agent_f32, agent_i32, metrics_f32, puffer_f32, traffic_i16)
        frames = oh["frames"]
        frames["agent_f32"].append(agent_f32[0])
        frames["agent_i32"].append(agent_i32[0])
        frames["metrics_f32"].append(metrics_f32[0])
        frames["puffer_f32"].append(puffer_f32[0])
        frames["traffic_i16"].append(traffic_i16[0])
        # Clip to the legitimate obs range before storage: the viewer quantizes
        # obs to int16 using the GLOBAL max |value|, and the parked FAR_AWAY
        # agents' rows carry ~5e3-magnitude goal offsets that blow up the scale
        # and quantize the ego's real observations (<= ~70) to zero.
        frames["obs"].append(np.clip(np.asarray(obs, dtype=np.float32), -100.0, 100.0))
        frames["raw_action"].append(actions.astype(np.float32))
        frames["value"].append(aux.get("value", np.zeros(self.num_agents, np.float32)))
        frames["entropy"].append(aux.get("entropy", np.zeros(self.num_agents, np.float32)))
        if "policy_probs" in aux:
            frames["policy_probs"].append(aux["policy_probs"])
        for pool_name, counts in aux.get("pool", {}).items():
            frames.setdefault(pool_name, []).append(counts)

    def _write_obs_html(self):
        """Write the interactive obs viewer HTML for this route (pufferlib.viz)."""
        from pufferlib import viz

        oh = self._obs_html
        frames = oh["frames"]
        if not frames["obs"]:
            return
        env = self.env
        env_cfg = {
            "init_step": 0,
            "goal_regen_mode": "finite",
            "action_type": "discrete",
            "dynamics_model": env.dynamics_model,
            "num_goals": int(env.num_goals),
            "reward_conditioning": bool(env.num_reward_coefs),
            "obs_slots_partners_n": int(env.obs_slots_partners_n),
            "obs_slots_lane_n": int(env.obs_slots_lane_n),
            "obs_slots_boundary_n": int(env.obs_slots_boundary_n),
            "obs_lane_stride": int(env.obs_lane_stride),
            "obs_boundary_stride": int(env.obs_boundary_stride),
            "obs_slots_traffic_controls_n": int(env.obs_slots_traffic_controls_n),
            "obs_dropout_lane": float(env.obs_dropout_lane),
            "obs_dropout_boundary": float(env.obs_dropout_boundary),
            "obs_norm_goal_offset_m": float(env.obs_norm_goal_offset_m),
            "obs_norm_xy_offset_m": float(env.obs_norm_xy_offset_m),
            "obs_norm_veh_width_m": float(env.obs_norm_veh_width_m),
            "obs_norm_veh_length_m": float(env.obs_norm_veh_length_m),
            "obs_norm_road_seg_length_m": float(env.obs_norm_road_seg_length_m),
            "obs_norm_road_seg_width_m": float(env.obs_norm_road_seg_width_m),
        }
        replay = {
            "schema": "obs_html_compact_v1",
            "env": env_cfg,
            "agent_f32": np.stack(frames["agent_f32"]),
            "agent_i32": np.stack(frames["agent_i32"]),
            "metrics_f32": np.stack(frames["metrics_f32"]),
            "puffer_f32": np.stack(frames["puffer_f32"]),
            "traffic_i16": np.stack(frames["traffic_i16"]),
            "obs": np.stack(frames["obs"]),
            "raw_action": np.stack(frames["raw_action"]),
            "clipped_action": np.stack(frames["raw_action"]),
            "value": np.stack(frames["value"]),
            "entropy": np.stack(frames["entropy"]),
            "policy_probs": np.stack(frames["policy_probs"]) if frames["policy_probs"] else None,
            "policy_mean": None,
            "policy_std": None,
            "policy_log_prob": None,
        }
        for pool_name in ("pool_partner", "pool_lane", "pool_boundary", "pool_traffic"):
            if frames.get(pool_name):
                replay[pool_name] = np.stack(frames[pool_name])
        out = str(Path(self.obs_html_dir) / f"{self.video_tag}.html")
        viz.generate_interactive_replay(oh["scenario"], replay, filename=out)
        print(f"[puffer_agent] wrote obs_html viewer ({len(frames['obs'])} frames) -> {out}")

    def _write_telemetry_row(self, ego_action):
        """One CSV row per policy step: what the loop commanded vs achieved,
        plus the nearest mapped light's state and the shadow infraction flags."""
        ego = self.env.get_global_agent_state()
        ex, ey = float(ego["x"][0]), float(ego["y"][0])
        cur = min(self.goal_cursor, len(self.route_goals) - 1)
        goal_dist = float(np.hypot(self.route_goals[cur, 0] - ex, self.route_goals[cur, 1] - ey))
        near_dist, near_state = -1.0, -1
        if len(self.stop_line_centers):
            d2 = (self.stop_line_centers[:, 0] - ex) ** 2 + (self.stop_line_centers[:, 1] - ey) ** 2
            j = int(d2.argmin())
            near_dist, near_state = float(np.sqrt(d2[j])), int(self.last_light_states[j])
        flags = self._ego_infractions()
        current_speed = (float(np.asarray(self.env.observations)[0, 0]) * self._max_speed()
                         if self.dynamics_source == "pufferdrive" else self.vehicle.get_velocity().length())
        self.telemetry_file.write(
            f"{self.step},{current_speed:.3f},{self.target[0]:.3f},"
            f"{ego_action},{self.goal_cursor},{goal_dist:.1f},{near_dist:.1f},{near_state},"
            f"{flags['collision']:.0f},{flags['offroad']:.0f},{flags['red_light']:.0f}\n")

    def _maybe_save_infraction_clip(self):
        """Dump the rolling chase-cam buffer as one mp4 when the shadow env
        flags an ego infraction (collision/offroad/red-light), at most once per
        INFRACTION_MIN_SEPARATION_M of ego travel (CaRL eval_agent-style)."""
        flags = self._ego_infractions()
        fired = [name for name, value in flags.items() if value > 0.0]
        if not fired or not self.infraction_buffer:
            return
        location = self.vehicle.get_location()
        if location.distance(self.last_infraction_location) <= INFRACTION_MIN_SEPARATION_M:
            return
        import imageio

        out = str(Path(self.record_infractions_dir)
                  / f"{self.video_tag}_{'_'.join(fired)}_{self.infraction_counter:02d}.mp4")
        imageio.mimwrite(out, list(self.infraction_buffer), fps=round(1.0 / self.tick_dt),
                         codec="libx264", macro_block_size=1)
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

        if self.step % self.action_repeat == 0:
            obs = self._sync_carla()  # shadow env <- CARLA ground truth
            if self.bev is not None:
                # capture the SYNCED state (CARLA truth), before integrate()
                # extrapolates every agent one dt past it
                cur = self.goal_cursor
                goals = self.route_goals[cur:cur + 3]
                self.bev.capture(self.env.get_global_agent_state(), ego_idx=0,
                                 goals=(goals[:, 0], goals[:, 1]),
                                 light_states=self.last_light_states)
            actions, aux = self._policy_actions(obs)
            if self._obs_html is not None and len(self._obs_html["frames"]["obs"]) < self.obs_html_max_steps:
                self._capture_obs_html_frame(obs, actions, aux)
            self.target = self._carla_integrate(actions)  # policy intent, one dt ahead
            if self.telemetry_file is not None:
                self._write_telemetry_row(int(actions[0, 0]))
            if self.record_infractions_dir:
                self._maybe_save_infraction_clip()

        if self.dynamics_source == "pufferdrive":
            # Policy ticks teleport the ego to the shadow env's post-step pose
            # in _carla_integrate. On the intermediate ticks, advance that pose
            # with the shadow env's own accel/yaw-rate so the CARLA ego moves
            # smoothly at tick rate instead of jumping once per policy step.
            if self.step % self.action_repeat and self._display is not None:
                x, y, yaw_deg, speed, yaw_rate_deg_s, accel_long = self._display
                new_speed = speed + accel_long * self.tick_dt
                # zero-crossing snap, mirroring the shadow dynamics (drive.h):
                # a held braking accel must not creep the display backward.
                speed = 0.0 if speed * new_speed < 0.0 else new_speed
                yaw_deg += yaw_rate_deg_s * self.tick_dt
                yaw_rad = math.radians(yaw_deg)
                x += speed * math.cos(yaw_rad) * self.tick_dt
                y += speed * math.sin(yaw_rad) * self.tick_dt
                self._display = [x, y, yaw_deg, speed, yaw_rate_deg_s, accel_long]
                self.vehicle.set_transform(carla.Transform(
                    carla.Location(x=x, y=y, z=0.3), carla.Rotation(yaw=yaw_deg)))
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
        print(f"[puffer_agent] route done: goals {self.goal_cursor + 1}/{len(self.route_goals)}, "
              f"tracking {self.controller.stats()}")
        if self.bev is not None and self.bev.frames:
            self.bev.save()
        if self.carla_view_writer is not None:
            self.carla_view_writer.close()
            print(f"[puffer_agent] wrote CARLA chase-cam video for route {self.video_tag}")
        if self.telemetry_file is not None:
            self.telemetry_file.close()
        if self._obs_html is not None:
            self._write_obs_html()
        self.env.close()
        self.initialized = False
