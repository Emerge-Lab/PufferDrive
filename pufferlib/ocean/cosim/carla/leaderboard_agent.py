"""PufferDrive policy as a CARLA leaderboard agent (CaRL original_leaderboard).

Usage (CaRL env vars/paths as in CaRL/CARLA/README.md, plus PufferDrive repo
root on PYTHONPATH for `pufferlib` and `data_utils`):

  python ${CARL_WORK_DIR}/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py \
      --routes ${CARL_WORK_DIR}/custom_leaderboard/leaderboard/data/longest6_split/longest6_00.xml \
      --agent /path/to/pufferlib/ocean/cosim/carla/leaderboard_agent.py \
      --agent-config /path/to/experiments/puffer_drive_xxx/models/model_xxx.pt \
      --checkpoint /path/to/results/result.json --track MAP

Environment variables:
  COSIM_DEVICE=cpu             torch device for the policy
  COSIM_DT=0.1                 policy dt in seconds (shadow-env integration horizon)
  COSIM_NUM_AGENTS=64          shadow-env agent slots (1 ego + N-1 background)
  COSIM_DEBUG_BEV=/dir         write a top-down BEV mp4 of the shadow env per route
  COSIM_DEBUG_CARLA_VIEW=/dir  write a CARLA chase-camera mp4 per route (native
                               tick rate, streamed to disk frame-by-frame — safe
                               for long routes, unlike COSIM_DEBUG_BEV's
                               in-memory frame list)
"""

import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import carla

import srunner
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# route_scenario.py's get_all_scenario_classes() globs
# f"{os.getenv('SCENARIO_RUNNER_ROOT', './')}/srunner/scenarios/*.py" to find
# every scenario class (DynamicObjectCrossing, SignalizedJunctionLeftTurn,
# ...).
os.environ.setdefault("SCENARIO_RUNNER_ROOT", str(Path(srunner.__file__).resolve().parents[1]))

from pufferlib.ocean.cosim.carla.world_sync import WorldSync
from pufferlib.ocean.cosim.carla.controller import TrackingController, read_vehicle_geometry

# jerk action: idx = j_long_idx*3 + j_lat_idx; (3,1) = +long jerk, straight
DUMMY_FORWARD_JERK = 3 * 3 + 1

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
    if not path_to_conf_file or path_to_conf_file == "dummy":
        return None, None
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


class PufferAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        self.track = autonomous_agent.Track.MAP
        self.route_index = re.sub(r"[^\w.-]", "_", str(route_index)) if route_index else "route"
        # CaRL's `route_index` is route_date_string = Path(ROUTES).stem,
        # fixed once per evaluator process unless --collect-dataset
        # is passed. A fresh timestamp per agent instance is the
        # unique tag for filenames.
        self.video_tag = f"{self.route_index}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.checkpoint, self.cfg = resolve_checkpoint(path_to_conf_file)

        self.device = os.environ.get("COSIM_DEVICE", "cpu")
        # Policy dt defaults to the checkpoint's training dt (shadow-env
        # integration horizon must match the dynamics the policy learned).
        cfg_dt = ((self.cfg or {}).get("env", {}) or {}).get("dt", 0.1)
        self.dt = float(os.environ.get("COSIM_DT", cfg_dt))
        self.num_agents = int(os.environ.get("COSIM_NUM_AGENTS", 64)) # TODO: should get from cfg?
        self.debug_bev_dir = os.environ.get("COSIM_DEBUG_BEV", None) # PufferDrive BEV mp4 output dir (per route) for better visulization
        self.debug_carla_view_dir = os.environ.get("COSIM_DEBUG_CARLA_VIEW", None)  # CARLA chase-cam mp4 dir (per route)

        self.step = -1
        self.initialized = False
        self.sync = None
        self.policy = None
        self.bev = None
        self.carla_view_writer = None
        self.target = (0.0, 0.0)  # (target_speed, target_yaw_deg), held between policy steps

    def sensors(self):
        if not self.debug_carla_view_dir:
            return []
        return [{"type": "sensor.camera.rgb", "id": CARLA_VIEW_SENSOR_ID,
                 "width": CARLA_VIEW_WIDTH, "height": CARLA_VIEW_HEIGHT, "fov": CARLA_VIEW_FOV,
                 **CARLA_VIEW_TRANSFORM}]

    def _init_on_first_step(self): #TODO: what is CaRL's goal mode
        """Deferred init (the ego and world only exist once the route runs) —
        same pattern as CaRL's eval_agent.agent_init. Everything here is
        read-only with respect to CARLA."""
        self.vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.vehicle.get_world()
        town = CarlaDataProvider.get_map().name.split("/")[-1]
        self.tick_dt = float(self.world.get_settings().fixed_delta_seconds)  # 0.05 @ 20 Hz
        self.action_repeat = max(1, round(self.dt / self.tick_dt))

        self.sync = WorldSync(
            self.world, self.vehicle, town, self.dense_global_plan_world_coord,
            num_agents=self.num_agents, dt=self.dt, cfg=self.cfg,
        )

        wheelbase, max_steer = read_vehicle_geometry(self.vehicle)
        self.controller = TrackingController(
            wheelbase_m=wheelbase, max_steer_rad=max_steer, horizon_s=self.dt
        )

        if self.checkpoint is not None:
            import torch
            import pufferlib.ocean.torch as drive_torch

            policy_cls = getattr(drive_torch, self.cfg.get("policy_name", "Drive"))
            self.policy = policy_cls(self.sync.env, **self.cfg["policy"]).to(self.device)
            sd = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(clean_policy_state_dict(sd))
            self.policy.eval()
            print(f"[puffer_agent] loaded policy from {self.checkpoint}")
        else:
            print("[puffer_agent] no checkpoint: dummy forward-jerk action (wiring test)")

        if self.debug_bev_dir:
            from pufferlib.ocean.cosim.carla_cosim import BEVRenderer

            Path(self.debug_bev_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(self.debug_bev_dir) / f"{self.video_tag}.mp4")
            self.bev = BEVRenderer(self.sync.town_bin, out)

        if self.debug_carla_view_dir:
            import imageio

            Path(self.debug_carla_view_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(self.debug_carla_view_dir) / f"{self.video_tag}.mp4")
            self.carla_view_writer = imageio.get_writer(
                out, fps=round(1.0 / self.tick_dt), codec="libx264", macro_block_size=1)

        print(f"[puffer_agent] town={town} tick_dt={self.tick_dt} dt={self.dt} "
              f"action_repeat={self.action_repeat} route_goals={len(self.sync.route_goals)}")
        self.initialized = True

    def _policy_actions(self, obs):
        if self.policy is None:
            return np.full((self.num_agents, 1), DUMMY_FORWARD_JERK, dtype=np.int32)
        import torch
        import pufferlib.pytorch

        with torch.no_grad():
            logits, _ = self.policy.forward_eval(torch.as_tensor(obs).to(self.device))
            action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
        return action.cpu().numpy().reshape(self.num_agents, -1).astype(np.int32)

    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1
        if not self.initialized:
            self._init_on_first_step()
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

        if self.carla_view_writer is not None and CARLA_VIEW_SENSOR_ID in input_data:
            _, bgra = input_data[CARLA_VIEW_SENSOR_ID]  # (H, W, 4) uint8, leaderboard's CallBack format
            self.carla_view_writer.append_data(bgra[:, :, [2, 1, 0]])  # BGRA -> RGB, streamed to disk

        if self.step % self.action_repeat == 0:
            obs = self.sync.sync()  # shadow env <- CARLA ground truth
            actions = self._policy_actions(obs)
            self.target = self.sync.integrate(actions)  # policy intent, one dt ahead
            if self.bev is not None:
                cur = self.sync.goal_cursor
                goals = self.sync.route_goals[cur:cur + 3]
                self.bev.capture(self.sync.ego_bin_state(), ego_idx=0,
                                 goals=(goals[:, 0], goals[:, 1]),
                                 light_states=self.sync.last_light_states)

        # Controller runs every tick against the latest CARLA state, chasing the
        # target held from the last policy step.
        v = self.vehicle.get_velocity()
        current_speed = math.hypot(v.x, v.y)
        current_yaw = self.vehicle.get_transform().rotation.yaw
        target_speed, target_yaw = self.target
        return self.controller.step(current_speed, current_yaw, target_speed, target_yaw,
                                    self.tick_dt)

    def destroy(self, results=None):
        if not self.initialized:
            return
        cursor, total = self.sync.route_progress()
        print(f"[puffer_agent] route done: goals {cursor + 1}/{total}, "
              f"tracking {self.controller.stats()}")
        if self.bev is not None and self.bev.frames:
            self.bev.save()
        if self.carla_view_writer is not None:
            self.carla_view_writer.close()
            print(f"[puffer_agent] wrote CARLA chase-cam video for route {self.video_tag}")
        if hasattr(self.sync.env, "close"):
            self.sync.env.close()
        self.initialized = False
