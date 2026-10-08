"""PufferDrive policy as a CARLA leaderboard agent (CaRL original_leaderboard, CARLA 0.9.15).

Usage (CaRL env vars/paths as in CaRL/CARLA/README.md, plus PufferDrive repo
root on PYTHONPATH for `pufferlib` and `data_utils`):

  python ${CARL_WORK_DIR}/original_leaderboard/leaderboard/leaderboard/leaderboard_evaluator.py \
      --routes ${CARL_WORK_DIR}/custom_leaderboard/leaderboard/data/longest6_split/longest6_00.xml \
      --agent /path/to/pufferlib/ocean/cosim/carla/leaderboard_agent.py \
      --agent-config /path/to/experiments/puffer_drive_xxx/models/model_xxx.pt \
      --checkpoint /path/to/results/result.json --track MAP

This file is the CARLA side only: it reads ground truth (ego, nearby actors, lights, stop signs, the
route), hands it to shadow_ego.ShadowEgo (the shadow Drive env + policy, built from the checkpoint's
config.yaml) and moves the ego the way the result says. The Leaderboard 1.0 integration (lb1/) runs
the same core behind a socket.

Environment variables (the shadow env's own, COSIM_DEVICE ... COSIM_WORLD_LOG, are listed in
shadow_ego.py):
  SCENARIO_RUNNER_ROOT         REQUIRED: path to the scenario_runner checkout
                               (route scenarios silently fail to load without it)
  CARL_WORK_DIR                REQUIRED if COSIM_TELEMETRY or
                               COSIM_RECORD_INFRACTIONS is set: path to the
                               CaRL/CARLA checkout, to import its
                               reward.criteria collision/offroad detectors
  COSIM_DEBUG_CARLA_VIEW=/dir  write a CARLA chase-camera mp4 per route (native
                               tick rate, streamed to disk frame-by-frame)
  COSIM_RECORD_INFRACTIONS=/dir  write a short chase-cam clip (last ~5 s) per
                               ego infraction (collision/offroad/red-light/stop-sign),
                               from either the shadow pufferdrive env's own
                               model (pd_*) or real CARLA ground truth (carla_*)
"""

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

from pufferlib.ocean.cosim import carla_bridge as cb
from pufferlib.ocean.cosim.carla.controller import vehicle_geometry_from_wheels, wheel_rows
from pufferlib.ocean.cosim.carla.shadow_ego import PARTNER_MAX_ABS_DZ_M, ShadowEgo

# Rolling chase-cam window kept for COSIM_RECORD_INFRACTIONS clips, and the
# minimum ego travel between two logged infractions (suppresses re-triggering
INFRACTION_CLIP_SECONDS = 5.0
INFRACTION_MIN_SEPARATION_M = 10.0
RED_LIGHT_CHECK_DISTANCE_M = 30.0  # matches CaRL RunRedLight's own distance_light default

ROAD_RAY_HALF_SPAN_M = (
    5.0  # vertical ray around the waypoint z; the ego roof and underpass roads are filtered by label/nearest
)
ROAD_RAY_LABELS = (carla.CityObjectLabel.Roads, carla.CityObjectLabel.RoadLines, carla.CityObjectLabel.Bridge)

CARLA_VIEW_SENSOR_ID = "puffer_chase_cam"
CARLA_VIEW_WIDTH, CARLA_VIEW_HEIGHT, CARLA_VIEW_FOV = 960, 540, 90
# Behind + above the ego, looking forward and down. Closer than
# carla_cosim.py's standalone chase cam (x=-6.5, z=3.2): the leaderboard
# enforces sqrt(x^2+y^2+z^2) <= agent_wrapper.MAX_ALLOWED_RADIUS_SENSOR (3.0 m)
CARLA_VIEW_TRANSFORM = dict(x=-2.0, y=0.0, z=2.1, roll=0.0, pitch=-15.0, yaw=0.0)


def get_entry_point():
    return "PufferAgent"


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


def actor_state_row(actor):
    """shadow_ego ACTOR_STATE of a live CARLA actor (CARLA frame)."""
    tf = actor.get_transform()
    v = actor.get_velocity()
    av = actor.get_angular_velocity()  # deg/s, world frame (CARLA's rotation convention)
    acc = actor.get_acceleration()  # m/s^2, world frame
    return [tf.location.x, tf.location.y, tf.location.z, tf.rotation.yaw, v.x, v.y, v.z, av.z, acc.x, acc.y]


def partner_row(actor):
    """shadow_ego PARTNER_ROW of a live CARLA vehicle or walker."""
    box = actor.bounding_box
    is_walker = 1.0 if "walker" in actor.type_id else 0.0
    return actor_state_row(actor) + [box.location.z, box.extent.x, box.extent.y, box.extent.z, is_walker, actor.id]


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
        self.shadow = ShadowEgo(path_to_conf_file, self.video_tag)

        self.debug_carla_view_dir = os.environ.get("COSIM_DEBUG_CARLA_VIEW", None)
        self.record_infractions_dir = os.environ.get("COSIM_RECORD_INFRACTIONS", None)

        self.step = -1
        self.initialized = False
        self.carla_view_writer = None
        self._carla_collision = None  # set by _init_carla_infraction_detectors when needed
        self._carla_stop_sign = None

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

    def _init_on_first_step(self):
        """Deferred init (the ego and world only exist once the route runs) —
        same pattern as CaRL's eval_agent.agent_init. Everything here is
        read-only with respect to CARLA."""
        self.vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.vehicle.get_world()
        self.cmap = self.world.get_map()
        town = CarlaDataProvider.get_map().name.split("/")[-1]
        self.tick_dt = float(self.world.get_settings().fixed_delta_seconds)  # 0.05 @ 20 Hz
        self.lights = list(self.world.get_actors().filter("traffic.traffic_light"))

        body = self.vehicle.bounding_box
        self.body_front_m = body.location.x + body.extent.x
        self.body_rear_m = body.location.x - body.extent.x
        wheels = wheel_rows(self.vehicle)
        self.wheelbase_m, _ = vehicle_geometry_from_wheels(wheels)
        self.shadow.init_route(
            {
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
                "driving_waypoints": cb.driving_waypoint_samples(self.cmap),
                "lights": cb.light_geometry_from_carla(self.lights, self.cmap),
                "stop_signs": cb.stop_sign_geometry_from_carla(self.world, self.cmap),
                "infraction_flags": bool(self.record_infractions_dir),
            }
        )

        if self.debug_carla_view_dir:
            from pufferlib.ocean.cosim.carla_cosim import Mp4Writer

            Path(self.debug_carla_view_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(self.debug_carla_view_dir) / f"{self.video_tag}.mp4")
            self.carla_view_writer = Mp4Writer(out, fps=round(1.0 / self.tick_dt))

        if self.record_infractions_dir:
            from collections import deque

            Path(self.record_infractions_dir).mkdir(parents=True, exist_ok=True)
            # Rolling window of the last INFRACTION_CLIP_SECONDS of chase-cam
            self.infraction_buffer = deque(maxlen=int(INFRACTION_CLIP_SECONDS / self.tick_dt))
            self.infraction_counter = 0
            self.last_infraction_location = self.vehicle.get_location()

        if self.shadow.telemetry_dir or self.record_infractions_dir:
            self._init_carla_infraction_detectors()
        self.initialized = True

    # --- CARLA ground truth for the shadow env (read-only w.r.t. CARLA) ----

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
        return [a for _, a in candidates[: self.shadow.partner_slots]]

    def _snapshot(self):
        ego_light = self.vehicle.get_traffic_light()
        return {
            "ego": actor_state_row(self.vehicle),
            "partners": [partner_row(a) for a in self._nearby_actors()],
            "light_states": [[lt.id, lt.get_state().name] for lt in self.lights],
            "ego_light_id": -1 if ego_light is None else ego_light.id,
        }

    def _teleport_carla_ego(self, motion):
        """Place the CARLA ego at the shadow env's post-step pose (StepResult.motion)."""
        _, _, yaw0_deg, _, x, y, yaw_deg, speed, carla_z = motion
        yaw_delta_deg = cb.wrap_deg_180(yaw_deg - yaw0_deg)
        # CARLA's live road-mesh height, not the shadow env's lane-averaged sim_z: on graded
        # multi-level roads the average can land the body mid-structure (measured: 20-100+
        # road collisions per Town03/04 route). sim_z only off the drivable network.
        # get_waypoint is nearest-in-3D: without sim_z the z=0 default snaps an overpass ego to the road beneath
        wp = self.cmap.get_waypoint(carla.Location(x=x, y=y, z=carla_z))
        z = wp.transform.location.z if wp is not None else carla_z
        road_up = wp.transform.rotation.get_up_vector() if wp is not None else carla.Vector3D(x=0.0, y=0.0, z=1.0)
        pitch_deg, roll_deg, forward = road_aligned_attitude(road_up, yaw_deg)
        # Waypoint z is quantised in ~0.5 m steps on steep grades (measured 0.32 m low at a Town03
        # descent); the mesh raycast is exact and resting the body on its bumper chord keeps the overhangs out of sags.
        surface = self._road_surface(x, y, z, yaw_deg)
        if surface is not None:
            z, pitch_deg = surface
            pitch_rad = math.radians(pitch_deg)
            yaw_rad = math.radians(yaw_deg)
            forward = (
                math.cos(yaw_rad) * math.cos(pitch_rad),
                math.sin(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
            )
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
        self.vehicle.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=yaw_delta_deg / self.shadow.dt))

    def _road_surface(self, x, y, reference_z, yaw_deg):
        """(z, pitch_deg) resting the body on the road mesh: bumper-to-bumper chord, lifted until no sample is below the mesh; None off the mesh."""
        yaw_rad = math.radians(yaw_deg)
        cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
        half_wheelbase = 0.5 * self.wheelbase_m
        sample_offsets_m = (self.body_rear_m, -half_wheelbase, 0.0, half_wheelbase, self.body_front_m)
        sample_z = [road_mesh_z(self.world, x + s * cos_yaw, y + s * sin_yaw, reference_z) for s in sample_offsets_m]
        if any(z is None for z in sample_z):
            return None
        slope = (sample_z[-1] - sample_z[0]) / (self.body_front_m - self.body_rear_m)
        z = max(z_i - slope * s_i for z_i, s_i in zip(sample_z, sample_offsets_m))
        return z, math.degrees(math.atan(slope))

    # --- CARLA ground-truth infraction detectors (CaRL's criteria) -----------

    def _init_carla_infraction_detectors(self):
        """Real CARLA ground truth for collision/offroad/red-light/stop-sign, reusing
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
        from reward.criteria.run_stop_sign2 import RunStopSign2
        from birds_eye_view.traffic_light import _get_traffic_light_waypoints

        self._carla_collision = Collision(self.vehicle, self.world)
        self._carla_collision_pending = False
        self._carla_offroad_test = OutsideRouteLanesTest(self.vehicle, self.cmap)
        self._carla_stop_sign = RunStopSign2(self.world, self.cmap)  # the training-side definition, on CARLA's actors
        self._ran_stop_sign_pending = False

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

    def _poll_carla_stop_sign(self):
        """Real CARLA ground truth for 'ran a stop sign' (RunStopSign2: the centre crossed the trigger
        volume's line without a standstill inside the volume first)."""
        info = self._carla_stop_sign.tick(self.vehicle)
        if info is not None and info["event"] == "run":
            self._ran_stop_sign_pending = True

    def _carla_infractions(self):
        """{'collision': f, 'offroad': f, 'red_light': f, 'stop_sign': f} from real CARLA
        ground truth (_init_carla_infraction_detectors) -- contrast with
        StepResult.pd_flags, which is the shadow env's model of the ego."""
        collision, self._carla_collision_pending = self._carla_collision_pending, False
        red_light, self._ran_red_light_pending = self._ran_red_light_pending, False
        stop_sign, self._ran_stop_sign_pending = self._ran_stop_sign_pending, False
        self._carla_offroad_test.update()
        offroad = self._carla_offroad_test.outside_lane_active or self._carla_offroad_test.wrong_lane_active
        return {
            "collision": float(collision),
            "offroad": float(offroad),
            "red_light": float(red_light),
            "stop_sign": float(stop_sign),
        }

    def _maybe_save_infraction_clip(self, pd_flags, carla_flags):
        """Dump the rolling chase-cam buffer as one mp4 when either infraction
        source flags an ego infraction (collision/offroad/red-light/stop-sign), at most
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
            self._poll_carla_stop_sign()

        result = self.shadow.step(self._snapshot())  # shadow env <- CARLA ground truth, policy, one dt
        if result.motion is not None:
            self._teleport_carla_ego(result.motion)
        if self._carla_collision is not None:
            carla_flags = self._carla_infractions()
            self.shadow.telemetry(result, carla_flags)
            if self.record_infractions_dir:
                self._maybe_save_infraction_clip(result.pd_flags, carla_flags)

        if result.control is None:
            return carla.VehicleControl()
        steer, throttle, brake = result.control
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)

    def destroy(self, results=None):
        if not self.initialized:
            return
        self.shadow.finish()
        if self.carla_view_writer is not None:
            self.carla_view_writer.close()
            print(f"[puffer_agent] wrote CARLA chase-cam video for route {self.video_tag}")
        if self._carla_collision is not None:
            self._carla_collision.clean()  # stop+destroy the spawned collision sensor actor
        self.initialized = False
