"""OpenGigaFlow policy as a carla_garage Leaderboard 1.0 agent (CARLA 0.9.10.1, Python 3.7, no opengigaflow imports):
the CARLA side, talking lb1_protocol to lb1/policy_server.py, which runs ShadowEgo in the OpenGigaFlow venv per route.

  python $GARAGE_WORK_DIR/leaderboard/leaderboard/leaderboard_evaluator_local.py --routes <routes.xml> \
      --scenarios $GARAGE_WORK_DIR/leaderboard/data/scenarios/eval_scenarios.json \
      --agent opengigaflow/ocean/cosim/carla/lb1/leaderboard_agent.py --agent-config experiments/<run>/final_model.pt \
      --checkpoint <result.json> --track MAP --timeout 600      (with BENCHMARK=longest6 and ROUTES=<routes.xml>)

Environment variables (policy_server.py reads the shadow env's COSIM_* from the inherited environment):
  SCENARIO_RUNNER_ROOT         required: carla_garage's scenario_runner checkout
  COSIM_SERVER_PYTHON          python of the OpenGigaFlow venv (default: <repo>/.venv/bin/python)
  COSIM_DEBUG_CARLA_VIEW=/dir  CARLA chase-camera mp4 per route
"""

import math
import os
import re
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import carla
import lb1_protocol
import numpy as np
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


REPO_ROOT = Path(__file__).resolve().parents[5]
POLICY_SERVER = Path(__file__).resolve().with_name("policy_server.py")
DEFAULT_SERVER_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
SERVER_ENV_DROPPED = ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")  # the garage env's libs/paths break the venv
PROTOCOL_TIMEOUT_S = 600.0  # one reply per tick; the evaluator's own agent watchdog is --timeout (600 s)
SERVER_EXIT_TIMEOUT_S = 60.0

CARLA_VIEW_SENSOR_ID = "opengigaflow_chase_cam"
CARLA_VIEW_WIDTH, CARLA_VIEW_HEIGHT, CARLA_VIEW_FOV = 960, 540, 90
CARLA_VIEW_TRANSFORM = dict(x=-2.0, y=0.0, z=2.1, roll=0.0, pitch=-15.0, yaw=0.0)  # within the 3 m sensor radius
MP4_FOURCC = "mp4v"

OFFSET_CALIBRATION_SPACING_M = 10.0  # carla_bridge.driving_waypoint_samples
TRIGGER_SAMPLE_SPAN_FRACTION = 0.9  # RunningRedLightTest.get_traffic_light_waypoints: avoids adjacent lanes
TRIGGER_SAMPLE_STEP_M = 1.0
JUNCTION_ADVANCE_STEP_M = 0.5
MAX_JUNCTION_ADVANCE_STEPS = 400  # 200 m: a stop waypoint farther from its junction is a map error
NO_LIGHT_ID = -1
NO_PARKING_LANE = -1.0


def get_entry_point():
    return "OpenGigaFlowAgentLB1"


def actor_state_row(actor):
    """shadow_ego ACTOR_STATE of a live CARLA actor (CARLA frame)."""
    tf = actor.get_transform()
    v = actor.get_velocity()
    av = actor.get_angular_velocity()
    acc = actor.get_acceleration()
    return [tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw, v.x, v.y, v.z, av.z, acc.x, acc.y]


def partner_row(actor):
    """shadow_ego PARTNER_ROW of a live CARLA vehicle or walker."""
    box = actor.bounding_box
    is_walker = 1.0 if "walker" in actor.type_id else 0.0
    is_two_wheeler = 1.0 if actor.attributes.get("number_of_wheels") == "2" else 0.0
    return actor_state_row(actor) + [
        box.location.z,
        box.extent.x,
        box.extent.y,
        box.extent.z,
        is_walker,
        is_two_wheeler,
        actor.id,
    ]


def plan_xyz(plan):
    """[(carla.Transform, RoadOption)] -> [[x, y, z], ...] CARLA-frame positions."""
    return [[t.location.x, t.location.y, t.location.z] for t, _ in plan]


def driving_waypoint_samples(carla_map):
    """[[x, y, z, right_x, right_y], ...] of the non-junction driving waypoints (carla_bridge.driving_waypoint_samples)."""
    rows = []
    for wp in carla_map.generate_waypoints(OFFSET_CALIBRATION_SPACING_M):
        if wp.is_junction or str(wp.lane_type) != "Driving":
            continue
        loc, right = wp.transform.location, wp.transform.get_right_vector()
        rows.append([loc.x, loc.y, loc.z, right.x, right.y])
    return rows


def light_stop_waypoints(light, carla_map):
    """One waypoint per lane the light's trigger volume spans, advanced to the junction entry: the py3.7 copy of
    carla_bridge.light_stop_waypoints (RunningRedLightTest's line)."""
    base_transform = light.get_transform()
    trigger = light.trigger_volume
    lane_waypoints = []
    for x in np.arange(
        -TRIGGER_SAMPLE_SPAN_FRACTION * trigger.extent.x,
        TRIGGER_SAMPLE_SPAN_FRACTION * trigger.extent.x,
        TRIGGER_SAMPLE_STEP_M,
    ):
        point = base_transform.transform(trigger.location + carla.Location(x=float(x)))
        wp = carla_map.get_waypoint(point)
        if wp is None:
            continue
        if not lane_waypoints or lane_waypoints[-1].road_id != wp.road_id or lane_waypoints[-1].lane_id != wp.lane_id:
            lane_waypoints.append(wp)
    stop_waypoints = []
    for wp in lane_waypoints:
        for _ in range(MAX_JUNCTION_ADVANCE_STEPS):
            if wp.is_junction:
                break
            ahead = wp.next(JUNCTION_ADVANCE_STEP_M)
            if not ahead or ahead[0].is_junction:
                break
            wp = ahead[0]
        stop_waypoints.append(wp)
    return stop_waypoints


def light_geometry(lights, carla_map, probe_steps_m):
    """carla_bridge.light_geometry_from_carla layout, built with the 0.9.10 API (probe steps from the server's config)."""
    geometry = []
    for light in lights:
        location = light.get_location()
        trigger = light.get_transform().transform(light.trigger_volume.location)
        stop_waypoints = []
        for wp in light_stop_waypoints(light, carla_map):
            backward, forward = [], []
            for step_m in probe_steps_m:
                probes = [wp] if step_m == 0.0 else wp.previous(step_m)
                backward.append(
                    None if not probes else [probes[0].transform.location.x, probes[0].transform.location.y]
                )
            for step_m in probe_steps_m[1:]:
                probes = wp.next(step_m) or []
                forward.append(
                    None
                    if not probes
                    else [
                        probes[0].transform.location.x,
                        probes[0].transform.location.y,
                        probes[0].transform.rotation.yaw,
                    ]
                )
            stop_waypoints.append(
                {
                    "x": wp.transform.location.x,
                    "y": wp.transform.location.y,
                    "yaw_deg": wp.transform.rotation.yaw,
                    "lane_width": wp.lane_width,
                    "backward": backward,
                    "forward": forward,
                }
            )
        geometry.append(
            {
                "id": light.id,
                "x": location.x,
                "y": location.y,
                "trigger_x": trigger.x,
                "trigger_y": trigger.y,
                "stop_waypoints": stop_waypoints,
            }
        )
    return geometry


def stop_sign_geometry(world, carla_map):
    """carla_bridge.stop_sign_geometry_from_carla layout, built with the 0.9.10 API."""
    geometry = []
    for actor in world.get_actors().filter("traffic.stop"):
        actor_transform = actor.get_transform()
        trigger = actor.trigger_volume
        center = actor_transform.transform(trigger.location)
        waypoint = carla_map.get_waypoint(center, project_to_road=True, lane_type=carla.LaneType.Driving)
        if waypoint is None:
            continue
        forward = waypoint.transform.get_forward_vector()
        lane_location = waypoint.transform.location
        geometry.append(
            {
                "id": actor.id,
                "center": [center.x, center.y, center.z],
                "extent": [trigger.extent.x, trigger.extent.y, trigger.extent.z],
                "yaw_deg": actor_transform.rotation.yaw,
                "lane": [
                    lane_location.x,
                    lane_location.y,
                    lane_location.z,
                    math.degrees(math.atan2(forward.y, forward.x)),
                ],
            }
        )
    return geometry


class Mp4Writer:
    """Streams RGB uint8 frames to an mp4 via OpenCV (cosim.mp4.Mp4Writer, importable here)."""

    def __init__(self, out_path, fps):
        self.out_path = str(out_path)
        self.fps = float(fps)
        self.writer = None

    def append_data(self, frame):
        import cv2

        frame = np.ascontiguousarray(frame)
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*MP4_FOURCC), self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Mp4Writer({self.out_path}): OpenCV could not open the video writer")
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class PolicyServerLink:
    """policy_server.py as a child process in the OpenGigaFlow venv, one request/reply per message."""

    def __init__(self, checkpoint, route_tag):
        python = os.environ.get("COSIM_SERVER_PYTHON", str(DEFAULT_SERVER_PYTHON))
        if not Path(python).is_file():
            raise FileNotFoundError(f"COSIM_SERVER_PYTHON={python} does not exist (the OpenGigaFlow venv python)")
        self.sock, child_sock = socket.socketpair()
        self.sock.settimeout(PROTOCOL_TIMEOUT_S)
        env = {k: v for k, v in os.environ.items() if k not in SERVER_ENV_DROPPED}
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHONUNBUFFERED"] = "1"  # the server shares the evaluator's log file
        command = [
            python,
            str(POLICY_SERVER),
            "--checkpoint",
            checkpoint,
            "--socket-fd",
            str(child_sock.fileno()),
            "--route-tag",
            route_tag,
        ]
        self.process = subprocess.Popen(command, cwd=str(REPO_ROOT), env=env, pass_fds=(child_sock.fileno(),))
        child_sock.close()
        self.config = self._receive()
        if self.config.get("type") != "config":
            raise RuntimeError(f"policy_server.py handshake failed: {self.config!r}")

    def _receive(self):
        try:
            message = lb1_protocol.recv_message(self.sock)
        except (ConnectionError, socket.timeout) as error:  # noqa: UP041  (alias of TimeoutError only from 3.10)
            raise RuntimeError(f"policy_server.py connection lost ({error}, exit code {self.process.poll()!r})")
        if message is None:
            raise RuntimeError(f"policy_server.py exited (code {self.process.poll()!r}) without replying")
        return message

    def request(self, message):
        lb1_protocol.send_message(self.sock, message)
        return self._receive()

    def close(self):
        self.sock.close()
        try:
            self.process.wait(timeout=SERVER_EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            print("[opengigaflow_agent_lb1] policy_server.py did not exit; killed", flush=True)


class OpenGigaFlowAgentLB1(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        if "SCENARIO_RUNNER_ROOT" not in os.environ:
            raise RuntimeError("SCENARIO_RUNNER_ROOT is not set; export carla_garage's scenario_runner checkout")
        self.track = autonomous_agent.Track.MAP
        route_index = re.sub(r"[^\w.-]", "_", str(route_index)) if route_index else "route"
        self.route_tag = f"{route_index}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.debug_carla_view_dir = os.environ.get("COSIM_DEBUG_CARLA_VIEW", None)
        self.ground_truth = bool(os.environ.get("COSIM_TELEMETRY"))
        self.server = PolicyServerLink(str(Path(path_to_conf_file).resolve()), self.route_tag)
        self.partner_slots = int(self.server.config["partner_slots"])
        self.partner_max_abs_dz_m = float(self.server.config["partner_max_abs_dz_m"])
        self.light_probe_steps_m = [float(v) for v in self.server.config["light_probe_steps_m"]]
        self.dense_global_plan_world_coord = None
        self.initialized = False
        self.carla_view_writer = None
        self._collision_sensor = None
        self._collision_events = []

    def sensors(self):
        if not self.debug_carla_view_dir:
            return []
        sensor = {
            "type": "sensor.camera.rgb",
            "id": CARLA_VIEW_SENSOR_ID,
            "width": CARLA_VIEW_WIDTH,
            "height": CARLA_VIEW_HEIGHT,
            "fov": CARLA_VIEW_FOV,
        }
        sensor.update(CARLA_VIEW_TRANSFORM)
        return [sensor]

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """The base class keeps only the 50 m target points; the dense 1 m route gives the goals their direction."""
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self.dense_global_plan_world_coord = global_plan_world_coord

    def _init_on_first_step(self):
        """Deferred init (the ego and world only exist once the route runs)."""
        self.vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.vehicle.get_world()
        if not self.sensors():  # no camera anywhere: skip the server's spectator rendering
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)
        self.cmap = self.world.get_map()
        town = CarlaDataProvider.get_map().name.split("/")[-1]
        self.tick_dt = float(self.world.get_settings().fixed_delta_seconds)
        self.lights = list(self.world.get_actors().filter("traffic.traffic_light"))
        body = self.vehicle.bounding_box
        route = {
            "town": town,
            "tick_dt": self.tick_dt,
            "ego": {
                "extent": [body.extent.x, body.extent.y, body.extent.z],
                "box_z": body.location.z,
                "blueprint": self.vehicle.type_id,
                "state": actor_state_row(self.vehicle),
            },
            "sparse_plan": plan_xyz(self._global_plan_world_coord),
            "dense_plan": plan_xyz(self.dense_global_plan_world_coord),
            "driving_waypoints": driving_waypoint_samples(self.cmap),
            "lights": light_geometry(self.lights, self.cmap, self.light_probe_steps_m),
            "stop_signs": stop_sign_geometry(self.world, self.cmap),
            "ground_truth": self.ground_truth,
        }
        reply = self.server.request({"type": "init", "route": route})
        if reply.get("type") != "ready":
            raise RuntimeError(f"policy_server.py did not become ready: {reply!r}")

        if self.debug_carla_view_dir:
            Path(self.debug_carla_view_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(self.debug_carla_view_dir) / f"{self.route_tag}.mp4")
            self.carla_view_writer = Mp4Writer(out, fps=round(1.0 / self.tick_dt))
        if self.ground_truth:
            blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
            self._collision_sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self.vehicle)
            self._collision_sensor.listen(self._on_collision)
        print(
            f"[opengigaflow_agent_lb1] town={town} tick_dt={self.tick_dt:.3f} lights={len(self.lights)} "
            f"stop_signs={len(route['stop_signs'])} target_points={len(route['sparse_plan'])} "
            f"dense={len(route['dense_plan'])}",
            flush=True,
        )
        self.initialized = True

    def _on_collision(self, event):
        self._collision_events.append(event.other_actor.type_id)

    def _nearby_actors(self):
        ego_loc = self.vehicle.get_location()
        candidates = []
        for a in self.world.get_actors():
            if a.id == self.vehicle.id or not ("vehicle" in a.type_id or "walker.pedestrian" in a.type_id):
                continue
            loc = a.get_location()
            # scenario_runner parks pending scenario actors hundreds of metres underground at the ego's waypoint
            if abs(loc.z - ego_loc.z) > self.partner_max_abs_dz_m:
                continue
            candidates.append((loc.distance(ego_loc), a))
        candidates.sort(key=lambda item: item[0])
        return [a for _, a in candidates[: self.partner_slots]]

    def _snapshot(self):
        ego_light = self.vehicle.get_traffic_light()
        return {
            "ego": actor_state_row(self.vehicle),
            "ego_gear": self.vehicle.get_control().gear,
            "partners": [partner_row(a) for a in self._nearby_actors()],
            "light_states": [[lt.id, lt.get_state().name] for lt in self.lights],
            "light_timers": [[lt.id, lt.get_elapsed_time(), lt.get_yellow_time()] for lt in self.lights],
            "ego_light_id": NO_LIGHT_ID if ego_light is None else ego_light.id,
        }

    def _ground_truth_observation(self):
        """lb1/ground_truth.py's per-tick input: collisions since the last tick and the ego's lane situation."""
        collisions, self._collision_events = self._collision_events, []
        location = self.vehicle.get_location()
        driving = self.cmap.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        parking = self.cmap.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Parking)
        parking_distance = NO_PARKING_LANE if parking is None else location.distance(parking.transform.location)
        return {
            "collision": collisions,
            "lane": [
                location.distance(driving.transform.location),
                driving.lane_width,
                parking_distance,
                0.0 if parking is None else parking.lane_width,
                driving.transform.rotation.yaw,
                1.0 if driving.is_junction else 0.0,
            ],
        }

    def run_step(self, input_data, timestamp, sensors=None):
        if not self.initialized:
            self._init_on_first_step()
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

        if CARLA_VIEW_SENSOR_ID in input_data:
            _, bgra = input_data[CARLA_VIEW_SENSOR_ID]  # (H, W, 4) uint8, leaderboard's CallBack format
            self.carla_view_writer.append_data(bgra[:, :, [2, 1, 0]])

        message = {"type": "tick", "snapshot": self._snapshot(), "ground_truth": None}
        if self.ground_truth:
            message["ground_truth"] = self._ground_truth_observation()
        steer, throttle, brake = self.server.request(message)["control"]
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

    def destroy(self, results=None):
        if not hasattr(self, "server"):
            return
        if self.initialized:
            reply = self.server.request({"type": "finish"})
            if reply.get("type") != "done":
                print(f"[opengigaflow_agent_lb1] unexpected finish reply {reply!r}", flush=True)
        self.server.close()
        if self.carla_view_writer is not None:
            self.carla_view_writer.close()
            print(f"[opengigaflow_agent_lb1] wrote CARLA chase-cam video for route {self.route_tag}", flush=True)
        if self._collision_sensor is not None:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None
        self.initialized = False
