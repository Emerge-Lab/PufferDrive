"""PufferDrive policy as a CARLA Leaderboard 1.0 agent (carla_garage's leaderboard_evaluator_local.py,
CARLA 0.9.10.1, longest6 v1).

The leaderboard runs in carla_garage's `garage` conda env: Python 3.7, the only Python the CARLA 0.9.10
egg supports, and one that cannot import PufferDrive (Python >= 3.9). This file is therefore the CARLA
side only -- the counterpart of ../leaderboard_agent.py's CARLA code: it reads ground truth (ego, nearby
actors, lights, stop signs, the route), ships it over a socket to lb1/policy_server.py (started here in
the PufferDrive venv, one process per route, shadow_ego.ShadowEgo inside) and moves the ego the way the
reply says. Nothing from `pufferlib` is imported here; lb1_protocol.py is a sibling module.

Usage (BENCHMARK=longest6 and ROUTES=<xml> exported, see run_leaderboard.sh / 14_carla_longest6_v1.sh):

  python $GARAGE_WORK_DIR/leaderboard/leaderboard/leaderboard_evaluator_local.py \
      --routes $GARAGE_WORK_DIR/leaderboard/data/longest6_split/longest_weathers_0.xml \
      --scenarios $GARAGE_WORK_DIR/leaderboard/data/scenarios/eval_scenarios.json \
      --agent /path/to/pufferlib/ocean/cosim/carla/lb1/leaderboard_agent.py \
      --agent-config /path/to/experiments/<run>/final_model.pt \
      --checkpoint /path/to/results/result.json --track MAP --timeout 600

Environment variables (this process; the shadow env's COSIM_DEVICE ... COSIM_WORLD_LOG are read by
policy_server.py from the inherited environment, see shadow_ego.py):
  SCENARIO_RUNNER_ROOT         REQUIRED: carla_garage's scenario_runner checkout
  COSIM_SERVER_PYTHON          python of the PufferDrive venv that runs policy_server.py
                               (default: <repo>/.venv/bin/python)
  COSIM_DEBUG_CARLA_VIEW=/dir  write a CARLA chase-camera mp4 per route (native tick rate)
  COSIM_RECORD_INFRACTIONS=/dir  write a short chase-cam clip (last ~5 s) per ego infraction flagged by
                               the shadow env (pd_*) or by the Leaderboard 1.0 criteria recomputed in
                               the server (carla_*, lb1/ground_truth.py)

CARLA 0.9.10 differences handled here: no World.cast_ray (the ego's body chord rests on the lane's OpenDRIVE
height profile, lifted by the road mesh's offset from it as probed by a downward semantic lidar; the egg's
Rotation.get_up_vector() points down, so the pitch comes from the profile too), no
TrafficLight.get_stop_waypoints (stop waypoints come from the trigger volume, RunningRedLightTest's own recipe).
"""

import math
import os
import re
import socket
import subprocess
from collections import deque
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

INFRACTION_CLIP_SECONDS = 5.0
INFRACTION_MIN_SEPARATION_M = 10.0
CARLA_VIEW_SENSOR_ID = "puffer_chase_cam"
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

WHEEL_POSITION_CM_PER_M = 100.0
ROAD_PROBE_MOUNT_Z_M = 0.5  # inside the hull, above any plausible penetration
ROAD_PROBE_RANGE_M = 20.0
ROAD_PROBE_POINTS_PER_SECOND = 2000
ROAD_SURFACE_TAGS = (6, 7, 14, 15, 16, 22)  # CityObjectLabel RoadLines, Roads, Ground, Bridge, RailTrack, Terrain
REST_ORIGIN_ABOVE_ROAD_M = 0.033  # mkz2017 origin height at rest on a flat road (measured, 0.9.10)
MESH_LIFT_MARGIN_M = 0.05
MESH_OFFSET_DEADBAND_M = 0.02
MESH_OFFSET_MAX_STEP_M = 0.3


def get_entry_point():
    return "PufferAgentLB1"


def wrap_deg_180(d):
    return (d + 180.0) % 360.0 - 180.0


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
    return actor_state_row(actor) + [box.location.z, box.extent.x, box.extent.y, box.extent.z, is_walker, actor.id]


def plan_xyz(plan):
    """[(carla.Transform, RoadOption)] -> [[x, y, z], ...] CARLA-frame positions."""
    return [[t.location.x, t.location.y, t.location.z] for t, _ in plan]


def lane_z_at(wp, offset_m):
    """Lane height offset_m along the lane from wp (negative = behind); None past the lane network."""
    if offset_m == 0.0:
        return wp.transform.location.z
    found = wp.next(offset_m) if offset_m > 0.0 else wp.previous(-offset_m)
    return found[0].transform.location.z if found else None


def lane_chord_pose(wp, sample_offsets_m):
    """(origin z, pitch_deg) of a rigid body chord resting on the lane's height profile around wp."""
    sample_z = [lane_z_at(wp, s) for s in sample_offsets_m]
    if any(z is None for z in sample_z):
        return wp.transform.location.z, wrap_deg_180(wp.transform.rotation.pitch)
    slope = (sample_z[-1] - sample_z[0]) / (sample_offsets_m[-1] - sample_offsets_m[0])
    z = max(z_i - slope * s_i for z_i, s_i in zip(sample_z, sample_offsets_m))
    return z, math.degrees(math.atan(slope))


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
    """Stop waypoints of a traffic light from its trigger volume: one waypoint per lane the volume spans,
    advanced to the junction entry (RunningRedLightTest.get_traffic_light_waypoints; CARLA >= 0.9.11
    exposes the same list as TrafficLight.get_stop_waypoints)."""
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
    """Streams RGB uint8 frames to an mp4 via OpenCV (carla_cosim.Mp4Writer, importable here)."""

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


def write_mp4(out_path, frames, fps):
    writer = Mp4Writer(out_path, fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


class RoadProbe:
    """Semantic lidar firing straight down from inside the hull: 0.9.10's substitute for World.cast_ray."""

    def __init__(self, world, vehicle, tick_dt):
        blueprint = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        for name, value in (
            ("channels", "1"),
            ("range", str(ROAD_PROBE_RANGE_M)),
            ("points_per_second", str(ROAD_PROBE_POINTS_PER_SECOND)),
            ("rotation_frequency", str(1.0 / tick_dt)),
            ("upper_fov", "-90"),
            ("lower_fov", "-90"),
        ):
            blueprint.set_attribute(name, value)
        mount = carla.Transform(carla.Location(x=0.0, y=0.0, z=ROAD_PROBE_MOUNT_Z_M))
        self.sensor = world.spawn_actor(blueprint, mount, attach_to=vehicle)
        self.sensor.listen(self._on_data)
        self._frame = None
        self._distance_m = None
        self._consumed_frame = None

    def _on_data(self, data):
        distances = sorted(-d.point.z for d in data if d.object_tag in ROAD_SURFACE_TAGS)
        if distances:
            self._frame, self._distance_m = data.frame, distances[len(distances) // 2]

    def fresh_distance_m(self):
        """Median road distance along the body's down axis from the newest unread scan; None when nothing new."""
        if self._frame is None or self._frame == self._consumed_frame:
            return None
        self._consumed_frame = self._frame
        return self._distance_m

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()


class PolicyServerLink:
    """policy_server.py as a child process in the PufferDrive venv, one request/reply per message."""

    def __init__(self, checkpoint, route_tag):
        python = os.environ.get("COSIM_SERVER_PYTHON", str(DEFAULT_SERVER_PYTHON))
        if not Path(python).is_file():
            raise FileNotFoundError(f"COSIM_SERVER_PYTHON={python} does not exist (the PufferDrive venv python)")
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
            print("[puffer_agent_lb1] policy_server.py did not exit; killed", flush=True)


class PufferAgentLB1(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        if "SCENARIO_RUNNER_ROOT" not in os.environ:
            raise RuntimeError("SCENARIO_RUNNER_ROOT is not set; export carla_garage's scenario_runner checkout")
        self.track = autonomous_agent.Track.MAP
        route_index = re.sub(r"[^\w.-]", "_", str(route_index)) if route_index else "route"
        self.route_tag = f"{route_index}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.debug_carla_view_dir = os.environ.get("COSIM_DEBUG_CARLA_VIEW", None)
        self.record_infractions_dir = os.environ.get("COSIM_RECORD_INFRACTIONS", None)
        self.ground_truth = bool(self.record_infractions_dir or os.environ.get("COSIM_TELEMETRY"))
        self.server = PolicyServerLink(str(Path(path_to_conf_file).resolve()), self.route_tag)
        self.partner_slots = int(self.server.config["partner_slots"])
        self.partner_max_abs_dz_m = float(self.server.config["partner_max_abs_dz_m"])
        self.light_probe_steps_m = [float(v) for v in self.server.config["light_probe_steps_m"]]
        self.dense_global_plan_world_coord = None
        self.step = -1
        self.initialized = False
        self.carla_view_writer = None
        self._collision_sensor = None
        self._collision_events = []
        self.road_probe = None
        self.body_chord_offsets_m = None
        self.mesh_offset_m = 0.0
        self._previous_body_z = None
        self._previous_lane_center_z = None
        self._previous_pitch_rad = 0.0

    def sensors(self):
        if not self.debug_carla_view_dir and not self.record_infractions_dir:
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
        self.cmap = self.world.get_map()
        town = CarlaDataProvider.get_map().name.split("/")[-1]
        self.tick_dt = float(self.world.get_settings().fixed_delta_seconds)
        self.lights = list(self.world.get_actors().filter("traffic.traffic_light"))
        body = self.vehicle.bounding_box
        wheels = [
            [w.max_steer_angle, w.position.x, w.position.y, w.position.z]
            for w in self.vehicle.get_physics_control().wheels
        ]
        front_left, rear_left = wheels[0][1:], wheels[2][1:]
        half_wheelbase_m = 0.5 * math.sqrt(sum((a - b) ** 2 for a, b in zip(front_left, rear_left))) / WHEEL_POSITION_CM_PER_M
        body_rear_m, body_front_m = body.location.x - body.extent.x, body.location.x + body.extent.x
        assert body_rear_m < -half_wheelbase_m < half_wheelbase_m < body_front_m
        self.body_chord_offsets_m = (body_rear_m, -half_wheelbase_m, 0.0, half_wheelbase_m, body_front_m)
        self.road_probe = RoadProbe(self.world, self.vehicle, self.tick_dt)
        route = {
            "town": town,
            "tick_dt": self.tick_dt,
            "ego": {
                "extent": [body.extent.x, body.extent.y, body.extent.z],
                "box_z": body.location.z,
                "wheels": wheels,
                "state": actor_state_row(self.vehicle),
            },
            "sparse_plan": plan_xyz(self._global_plan_world_coord),
            "dense_plan": plan_xyz(self.dense_global_plan_world_coord),
            "driving_waypoints": driving_waypoint_samples(self.cmap),
            "lights": light_geometry(self.lights, self.cmap, self.light_probe_steps_m),
            "stop_signs": stop_sign_geometry(self.world, self.cmap),
            "infraction_flags": bool(self.record_infractions_dir),
            "ground_truth": self.ground_truth,
        }
        reply = self.server.request({"type": "init", "route": route})
        if reply.get("type") != "ready":
            raise RuntimeError(f"policy_server.py did not become ready: {reply!r}")

        if self.debug_carla_view_dir:
            Path(self.debug_carla_view_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(self.debug_carla_view_dir) / f"{self.route_tag}.mp4")
            self.carla_view_writer = Mp4Writer(out, fps=round(1.0 / self.tick_dt))
        if self.record_infractions_dir:
            Path(self.record_infractions_dir).mkdir(parents=True, exist_ok=True)
            self.infraction_buffer = deque(maxlen=int(INFRACTION_CLIP_SECONDS / self.tick_dt))
            self.infraction_counter = 0
            self.last_infraction_location = self.vehicle.get_location()
        if self.ground_truth:
            blueprint = self.world.get_blueprint_library().find("sensor.other.collision")
            self._collision_sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self.vehicle)
            self._collision_sensor.listen(self._on_collision)
        print(
            f"[puffer_agent_lb1] town={town} tick_dt={self.tick_dt:.3f} lights={len(self.lights)} "
            f"stop_signs={len(route['stop_signs'])} target_points={len(route['sparse_plan'])} "
            f"dense={len(route['dense_plan'])}",
            flush=True,
        )
        self.initialized = True

    def _on_collision(self, event):
        self._collision_events.append(event.other_actor.type_id)

    # --- CARLA ground truth for the shadow env (read-only w.r.t. CARLA) ----

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
            "partners": [partner_row(a) for a in self._nearby_actors()],
            "light_states": [[lt.id, lt.get_state().name] for lt in self.lights],
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

    def _teleport_carla_ego(self, motion):
        """Place the CARLA ego at the shadow env's post-step pose (shadow_ego.StepResult.motion): the body chord on
        the lane's height profile, lifted by the probed road-mesh offset (0.9.10 has no road-mesh raycast)."""
        _, _, yaw0_deg, _, x, y, yaw_deg, speed, carla_z = motion
        yaw_delta_deg = wrap_deg_180(yaw_deg - yaw0_deg)
        wp = self.cmap.get_waypoint(carla.Location(x=x, y=y, z=carla_z))
        chord_z, pitch_deg = (carla_z, 0.0) if wp is None else lane_chord_pose(wp, self.body_chord_offsets_m)
        self._update_mesh_offset()
        z = chord_z + self.mesh_offset_m + REST_ORIGIN_ABOVE_ROAD_M + MESH_LIFT_MARGIN_M
        pitch_rad, yaw_rad = math.radians(pitch_deg), math.radians(yaw_deg)
        self._previous_body_z = z
        self._previous_lane_center_z = carla_z if wp is None else wp.transform.location.z
        self._previous_pitch_rad = pitch_rad
        forward = (math.cos(yaw_rad) * math.cos(pitch_rad), math.sin(yaw_rad) * math.cos(pitch_rad), math.sin(pitch_rad))
        # Zero momentum before the teleport: CARLA's collision resolver reacts violently to a
        # physics body carrying velocity into a new pose (carla issue #8076).
        zero = carla.Vector3D(x=0.0, y=0.0, z=0.0)
        self.vehicle.set_target_velocity(zero)
        self.vehicle.set_target_angular_velocity(zero)
        self.vehicle.set_transform(
            carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch_deg, yaw=yaw_deg, roll=0.0))
        )
        self.vehicle.set_target_velocity(carla.Vector3D(x=speed * forward[0], y=speed * forward[1], z=speed * forward[2]))
        self.vehicle.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=yaw_delta_deg / self.tick_dt))

    def _update_mesh_offset(self):
        """Road mesh height above the lane's height profile under the body, from the probe's newest scan at the previous pose."""
        distance_m = self.road_probe.fresh_distance_m()
        if distance_m is None or self._previous_body_z is None:
            return
        # the ray follows the pitched body's down axis; the mount offset is along that axis too
        vertical_clearance_m = (distance_m - ROAD_PROBE_MOUNT_Z_M) * math.cos(self._previous_pitch_rad)
        measured_m = self._previous_body_z - vertical_clearance_m - self._previous_lane_center_z
        step_m = measured_m - self.mesh_offset_m
        if abs(step_m) > MESH_OFFSET_DEADBAND_M:
            self.mesh_offset_m += max(-MESH_OFFSET_MAX_STEP_M, min(MESH_OFFSET_MAX_STEP_M, step_m))

    def _maybe_save_infraction_clip(self, pd_flags, carla_flags):
        """One mp4 of the rolling chase-cam buffer per flagged ego infraction, at most once per
        INFRACTION_MIN_SEPARATION_M of ego travel (as ../leaderboard_agent.py)."""
        fired = [f"pd_{name}" for name, value in (pd_flags or {}).items() if value > 0.0]
        fired += [f"carla_{name}" for name, value in (carla_flags or {}).items() if value > 0.0]
        if not fired or not self.infraction_buffer:
            return
        location = self.vehicle.get_location()
        if location.distance(self.last_infraction_location) <= INFRACTION_MIN_SEPARATION_M:
            return
        out = str(
            Path(self.record_infractions_dir) / f"{self.route_tag}_{'_'.join(fired)}_{self.infraction_counter:02d}.mp4"
        )
        write_mp4(out, list(self.infraction_buffer), fps=round(1.0 / self.tick_dt))
        print(f"[puffer_agent_lb1] infraction {fired} -> {out}", flush=True)
        self.infraction_counter += 1
        self.last_infraction_location = location

    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1
        if not self.initialized:
            self._init_on_first_step()
            return carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

        if CARLA_VIEW_SENSOR_ID in input_data:
            _, bgra = input_data[CARLA_VIEW_SENSOR_ID]  # (H, W, 4) uint8, leaderboard's CallBack format
            rgb = bgra[:, :, [2, 1, 0]]
            if self.carla_view_writer is not None:
                self.carla_view_writer.append_data(rgb)
            if self.record_infractions_dir:
                self.infraction_buffer.append(rgb.copy())

        message = {"type": "tick", "snapshot": self._snapshot(), "ground_truth": None}
        if self.ground_truth:
            message["ground_truth"] = self._ground_truth_observation()
        reply = self.server.request(message)
        if reply.get("motion") is not None:
            self._teleport_carla_ego(reply["motion"])
        if self.record_infractions_dir:
            self._maybe_save_infraction_clip(reply.get("pd_flags"), reply.get("carla_flags"))
        if reply.get("control") is None:
            return carla.VehicleControl()
        steer, throttle, brake = reply["control"]
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

    def destroy(self, results=None):
        if not hasattr(self, "server"):
            return
        if self.initialized:
            reply = self.server.request({"type": "finish"})
            if reply.get("type") != "done":
                print(f"[puffer_agent_lb1] unexpected finish reply {reply!r}", flush=True)
        self.server.close()
        if self.carla_view_writer is not None:
            self.carla_view_writer.close()
            print(f"[puffer_agent_lb1] wrote CARLA chase-cam video for route {self.route_tag}", flush=True)
        if self.road_probe is not None:
            self.road_probe.destroy()
            self.road_probe = None
        if self._collision_sensor is not None:
            self._collision_sensor.stop()
            self._collision_sensor.destroy()
            self._collision_sensor = None
        self.initialized = False
