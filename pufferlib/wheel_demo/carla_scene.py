"""CARLA as the renderer: every sim car becomes a CARLA actor posed from the sim each frame.

The sim stays the only source of truth. Nothing here writes sim state; CARLA physics, traffic manager
and traffic-light timers are all off, and CARLA's lights are frozen to whatever the sim says.

Heights come from CARLA's own road surface (its OpenDRIVE, parsed client-side), not from the sim's z,
which steps at lane boundaries. The sim's z only picks WHICH surface (bridge or road below).
"""

import math
import queue
from dataclasses import dataclass

import numpy as np

from pufferlib.ocean.cosim import carla_bridge


CAMERA_MODE_CHASE = 0
CAMERA_MODE_DRIVER = 1
CAMERA_MODE_AGENT = 2

# Solid roof and a full interior; the Lincoln's glass roof reads as roofless from the chase camera.
EGO_BLUEPRINT = "vehicle.dodge.charger_2020"
PARKED_HEIGHT_M = -200.0
SPAWN_HEIGHT_M = 500.0
SPAWN_SPACING_M = 6.0
CARLA_TIMEOUT_S = 120.0
FRAME_TIMEOUT_S = 10.0

# Heights and pitch ease toward the road surface over this time, so lane-to-lane steps never show.
GROUND_SMOOTHING_SECONDS = 0.12
# A jump this large is a respawn or takeover, so the render snaps instead of easing.
GROUND_SNAP_DISTANCE_M = 4.0
GROUND_CLEARANCE_M = 0.03
# Below this the front wheels are not re-sent: each steer update is a blocking server call.
WHEEL_STEER_RESEND_DEG = 0.2

BRAKE_LIGHT_DECEL_MPS2 = 1.0
BRAKE_LIGHT_PEDAL = 0.05

CHASE_BACK_M = 9.0
CHASE_EYE_HEIGHT_M = 3.6
CHASE_AHEAD_M = 14.0
CHASE_TARGET_HEIGHT_M = 1.6
CHASE_FOV_DEG = 45.0
DRIVER_EYE_HEIGHT_M = 1.26
DRIVER_SEAT_FORWARD_M = 0.30
DRIVER_SEAT_LEFT_M = 0.38
DRIVER_AHEAD_M = 30.0
DRIVER_TARGET_HEIGHT_M = 0.35
DRIVER_FOV_DEG = 60.0
AGENT_VIEW_SPAN_M = 60.0
AGENT_VIEW_HEIGHT_M = 100.0

# One wide rear camera feeds all three mirrors: every extra camera costs a full scene render.
REAR_CAMERA_SIZE_PX = (1800, 320)
REAR_CAMERA_FOV_DEG = 120.0
REAR_CAMERA_HEIGHT_M = 1.35
MIRROR_FRAME_STRIDE = 2
# CARLA's auto-exposure defaults to EV 10-12, a daylight range: at night it cannot open up and stays black.
NIGHT_EXPOSURE_MIN_EV = 8.5
NIGHT_EXPOSURE_MAX_EV = 11.0
# The rear camera sees open sky by day and blows out without this.
DAY_MIRROR_EXPOSURE_COMPENSATION_EV = -1.0

GOAL_BEACON_HEIGHT_M = 3.0

TRAFFIC_LIGHT_STATE_RED = 1
TRAFFIC_LIGHT_STATE_YELLOW = 2
TRAFFIC_LIGHT_STATE_GREEN = 3


@dataclass(frozen=True)
class CameraRig:
    """A camera in the CARLA vehicle frame: x forward, y RIGHT, z up (CARLA is left-handed)."""

    x: float
    y: float
    z: float
    pitch_deg: float
    yaw_deg: float
    fov_deg: float


def main_camera_rig(camera_mode):
    if camera_mode == CAMERA_MODE_CHASE:
        pitch = math.degrees(math.atan2(CHASE_TARGET_HEIGHT_M - CHASE_EYE_HEIGHT_M, CHASE_BACK_M + CHASE_AHEAD_M))
        return CameraRig(-CHASE_BACK_M, 0.0, CHASE_EYE_HEIGHT_M, pitch, 0.0, CHASE_FOV_DEG)
    if camera_mode == CAMERA_MODE_DRIVER:
        pitch = math.degrees(
            math.atan2(DRIVER_TARGET_HEIGHT_M - DRIVER_EYE_HEIGHT_M, DRIVER_AHEAD_M - DRIVER_SEAT_FORWARD_M)
        )
        return CameraRig(DRIVER_SEAT_FORWARD_M, -DRIVER_SEAT_LEFT_M, DRIVER_EYE_HEIGHT_M, pitch, 0.0, DRIVER_FOV_DEG)
    if camera_mode == CAMERA_MODE_AGENT:
        fov = math.degrees(2.0 * math.atan2(AGENT_VIEW_SPAN_M / 2.0, AGENT_VIEW_HEIGHT_M))
        return CameraRig(0.0, 0.0, AGENT_VIEW_HEIGHT_M, -90.0, 0.0, fov)
    raise ValueError(f"unknown camera mode {camera_mode}")


def rear_camera_rig(ego_half_length_m):
    """Looks straight back from the tail; the window flips it left-right, as a mirror is."""
    return CameraRig(-ego_half_length_m, 0.0, REAR_CAMERA_HEIGHT_M, 0.0, 180.0, REAR_CAMERA_FOV_DEG)


def project_to_image(world_to_camera, point_xyz, width_px, height_px, fov_deg):
    """Pixel of a CARLA-world point in a camera image, or None behind the camera. The camera frame is
    x forward, y right, z up; image y grows downward."""
    forward, right, up, _ = world_to_camera @ np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0])
    if forward <= 0.0:
        return None
    focal_px = width_px / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return (width_px / 2.0 + focal_px * right / forward, height_px / 2.0 - focal_px * up / forward)


def front_wheel_steer_deg(sim_steering_angle_rad):
    """CARLA steers positive to the RIGHT; the sim's steering_angle is positive-LEFT."""
    return -math.degrees(sim_steering_angle_rad)


def ease_toward(current, target, dt_s, time_constant_s):
    blend = 1.0 - math.exp(-dt_s / time_constant_s)
    return current + blend * (target - current)


def nearest_blueprint(blueprint_sizes, length_m, width_m, agent_idx):
    """Closest box among the blueprints, rotating through near-ties so traffic is not all one model."""
    errors = [abs(length - length_m) + abs(width - width_m) for _, length, width in blueprint_sizes]
    best = min(errors)
    candidates = [idx for idx, error in enumerate(errors) if error <= best + 0.35]
    return blueprint_sizes[candidates[agent_idx % len(candidates)]][0]


class CarlaScene:
    """Owns the CARLA connection, one actor per sim agent, the cameras and the light mirroring."""

    def __init__(
        self, carla, town, env, human_agent_idx, substep_dt_s, host="localhost", port=2000, image_size_px=(1280, 720)
    ):
        self.carla = carla
        self.town = town
        self.env = env
        self.substep_dt_s = substep_dt_s
        self.image_size_px = image_size_px
        self.client = carla.Client(host, port)
        self.client.set_timeout(CARLA_TIMEOUT_S)
        available = [name.rsplit("/", 1)[-1] for name in self.client.get_available_maps()]
        if town not in available:
            raise RuntimeError(f"CARLA server has no {town}; it has {sorted(available)}")
        self.world = self.client.get_world()
        if self.world.get_map().name.rsplit("/", 1)[-1] != town:
            self.world = self.client.load_world(town)
        self._original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = substep_dt_s
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        self.map = self.world.get_map()

        town_bin = carla_bridge.bin_path_for_town(town)
        transform = carla_bridge.CarlaTransform(town, offset=carla_bridge.town_offset(town_bin))
        offset, z_offset, _, residual_m = carla_bridge.calibrate_town_offset(
            carla_bridge.driving_waypoint_samples(self.map), transform, town_bin
        )
        self.transform = carla_bridge.CarlaTransform(town, offset=offset, z_offset=z_offset)
        self.alignment_residual_m = residual_m

        self._actor_ids = []
        self._camera = None
        self._mirrors = {}
        self._frames = {}
        self.night = False
        self._camera_mode = None
        self._pending_frame_id = None
        self._latest_mirror = {}
        self._light_states = {}
        self._vehicle_light_states = {}
        self._ground = {}
        self._last_speed = {}
        self._box_centers = {}
        self._sent_steer_deg = math.inf
        try:
            self._spawn_traffic()
            self._spawn_ego()
            self._setup_traffic_lights(town_bin)
            self.human_agent_idx = human_agent_idx
        except Exception:
            self.close()
            raise

    # ----- setup -----

    def _vehicle_blueprints(self):
        library = self.world.get_blueprint_library()
        blueprints = [
            bp
            for bp in library.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels")) == 4 and bp.get_attribute("base_type") == "car"
        ]
        if not blueprints:
            raise RuntimeError("CARLA server exposes no 4-wheel car blueprints")
        return blueprints

    def _measure_blueprints(self, blueprints):
        carla = self.carla
        batch = [
            carla.command.SpawnActor(bp, carla.Transform(carla.Location(x=SPAWN_SPACING_M * i, z=SPAWN_HEIGHT_M))).then(
                carla.command.SetSimulatePhysics(carla.command.FutureActor, False)
            )
            for i, bp in enumerate(blueprints)
        ]
        results = self.client.apply_batch_sync(batch, True)
        sizes = []
        probe_ids = [r.actor_id for r in results if not r.error]
        actors = {a.id: a for a in self.world.get_actors(probe_ids)}
        for bp, result in zip(blueprints, results):
            if result.error:
                continue
            extent = actors[result.actor_id].bounding_box.extent
            sizes.append((bp, 2.0 * extent.x, 2.0 * extent.y))
        self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in probe_ids], True)
        if not sizes:
            raise RuntimeError("no CARLA car blueprint could be spawned")
        return sizes

    def _spawn_traffic(self):
        carla = self.carla
        state = self.env.get_global_agent_state()
        blueprint_sizes = self._measure_blueprints(self._vehicle_blueprints())
        batch = []
        for agent_idx in range(len(state["x"])):
            bp = nearest_blueprint(
                blueprint_sizes, float(state["length"][agent_idx]), float(state["width"][agent_idx]), agent_idx
            )
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "traffic")
            spawn = carla.Transform(carla.Location(x=SPAWN_SPACING_M * agent_idx, y=SPAWN_SPACING_M, z=SPAWN_HEIGHT_M))
            batch.append(
                carla.command.SpawnActor(bp, spawn).then(
                    carla.command.SetSimulatePhysics(carla.command.FutureActor, False)
                )
            )
        results = self.client.apply_batch_sync(batch, True)
        failures = [r.error for r in results if r.error]
        if failures:
            raise RuntimeError(f"could not spawn {len(failures)} traffic actors: {failures[0]}")
        self._actor_ids = [r.actor_id for r in results]
        for actor in self.world.get_actors(self._actor_ids):
            self._box_centers[actor.id] = (actor.bounding_box.location.x, actor.bounding_box.location.y)

    def _spawn_ego(self):
        carla = self.carla
        bp = self.world.get_blueprint_library().find(EGO_BLUEPRINT)
        bp.set_attribute("role_name", "hero")
        self.ego = self.world.spawn_actor(bp, carla.Transform(carla.Location(z=SPAWN_HEIGHT_M + 50.0)))
        self.ego.set_simulate_physics(False)
        box = self.ego.bounding_box
        self.ego_half_length_m = box.extent.x
        self.ego_half_width_m = box.extent.y
        self._box_centers[self.ego.id] = (box.location.x, box.location.y)

    def _setup_traffic_lights(self, town_bin):
        lights = sorted(self.world.get_actors().filter("traffic.traffic_light"), key=lambda light: light.id)
        geometry = carla_bridge.light_geometry_from_carla(lights, self.map)
        mapping, element_count = carla_bridge.map_lights_to_bin(geometry, self.transform, town_bin)
        self._lights = lights
        self._light_elements = mapping
        self._light_state_buffer = np.zeros(element_count, dtype=np.int32)
        self.world.freeze_all_traffic_lights(True)

    # ----- per frame -----

    def set_human_agent(self, agent_idx):
        self.human_agent_idx = agent_idx
        self._ground.pop(("ego",), None)

    def set_camera_mode(self, camera_mode):
        if camera_mode == self._camera_mode:
            return
        self._destroy_cameras()
        rig = main_camera_rig(camera_mode)
        self._camera = self._spawn_camera(rig, self.image_size_px, "main", 0.0)
        if camera_mode == CAMERA_MODE_DRIVER:
            self._mirrors["rear"] = self._spawn_camera(
                rear_camera_rig(self.ego_half_length_m),
                REAR_CAMERA_SIZE_PX,
                "rear",
                MIRROR_FRAME_STRIDE * self.substep_dt_s,
            )
        self._camera_mode = camera_mode
        self._pending_frame_id = None

    def _spawn_camera(self, rig, size_px, name, sensor_tick_s):
        carla = self.carla
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(size_px[0]))
        bp.set_attribute("image_size_y", str(size_px[1]))
        bp.set_attribute("fov", str(rig.fov_deg))
        bp.set_attribute("sensor_tick", str(sensor_tick_s))
        if self.night:
            bp.set_attribute("exposure_min_bright", str(NIGHT_EXPOSURE_MIN_EV))
            bp.set_attribute("exposure_max_bright", str(NIGHT_EXPOSURE_MAX_EV))
        elif name == "rear":
            bp.set_attribute("exposure_compensation", str(DAY_MIRROR_EXPOSURE_COMPENSATION_EV))
        transform = carla.Transform(
            carla.Location(x=rig.x, y=rig.y, z=rig.z), carla.Rotation(pitch=rig.pitch_deg, yaw=rig.yaw_deg)
        )
        camera = self.world.spawn_actor(bp, transform, attach_to=self.ego)
        frames = queue.Queue()
        camera.listen(frames.put)
        self._frames[name] = frames
        return camera

    def _destroy_cameras(self):
        for camera in [self._camera, *self._mirrors.values()]:
            if camera is not None:
                camera.stop()
                camera.destroy()
        self._camera = None
        self._mirrors = {}
        self._frames = {}
        self._latest_mirror = {}

    def set_night(self, night):
        carla = self.carla
        self.world.set_weather(carla.WeatherParameters.ClearNight if night else carla.WeatherParameters.ClearNoon)
        light_manager = self.world.get_lightmanager()
        for group in (carla.LightGroup.Street, carla.LightGroup.Building):
            lights = light_manager.get_all_lights(group)
            if night:
                light_manager.turn_on(lights)
            else:
                light_manager.turn_off(lights)
        self.night = night
        self._vehicle_light_states = {}
        if self._camera_mode is not None:
            camera_mode, self._camera_mode = self._camera_mode, None
            self.set_camera_mode(camera_mode)

    def _ground_pose(self, key, x, y, yaw_deg, sim_z_carla, dt_s):
        """Render-only height and pitch from CARLA's road surface under the car, eased over time."""
        surface = self.map.get_waypoint(self.carla.Location(x=x, y=y, z=sim_z_carla)).transform
        target_z = surface.location.z
        # the lane's grade, seen along the car's own heading
        lane_pitch = (surface.rotation.pitch + 180.0) % 360.0 - 180.0  # CARLA reports pitch in [0, 360)
        target_pitch = lane_pitch * math.cos(math.radians(yaw_deg - surface.rotation.yaw))
        previous = self._ground.get(key)
        if previous is None or math.hypot(x - previous[0], y - previous[1]) > GROUND_SNAP_DISTANCE_M:
            z, pitch = target_z, target_pitch
        else:
            z = ease_toward(previous[2], target_z, dt_s, GROUND_SMOOTHING_SECONDS)
            pitch = ease_toward(previous[3], target_pitch, dt_s, GROUND_SMOOTHING_SECONDS)
        self._ground[key] = (x, y, z, pitch)
        return z, pitch

    def _actor_transform(self, key, box_center_xy, bx, by, bz, heading, dt_s):
        carla = self.carla
        x, y = self.transform.bin_to_loc(bx, by)
        yaw = self.transform.bin_heading_to_yaw(heading)
        z, pitch = self._ground_pose(key, x, y, yaw, self.transform.z_to_carla(bz), dt_s)
        cos_yaw, sin_yaw = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        origin_x = x - (box_center_xy[0] * cos_yaw - box_center_xy[1] * sin_yaw)
        origin_y = y - (box_center_xy[0] * sin_yaw + box_center_xy[1] * cos_yaw)
        return carla.Transform(
            carla.Location(x=origin_x, y=origin_y, z=z + GROUND_CLEARANCE_M), carla.Rotation(yaw=yaw, pitch=pitch)
        )

    def update(self, human, wheel_state, dt_s):
        """Pose every actor from the sim, mirror the lights, tick, and return the camera images."""
        carla = self.carla
        state = self.env.get_global_agent_state()
        commands = []
        light_commands = []
        xs, ys, zs, headings = (state[name].tolist() for name in ("x", "y", "z", "heading"))
        for agent_idx, actor_id in enumerate(self._actor_ids):
            if agent_idx == self.human_agent_idx:
                commands.append(
                    carla.command.ApplyTransform(
                        actor_id, carla.Transform(carla.Location(x=SPAWN_SPACING_M * agent_idx, z=PARKED_HEIGHT_M))
                    )
                )
                continue
            transform = self._actor_transform(
                agent_idx,
                self._box_centers[actor_id],
                xs[agent_idx],
                ys[agent_idx],
                zs[agent_idx],
                headings[agent_idx],
                dt_s,
            )
            commands.append(carla.command.ApplyTransform(actor_id, transform))
            light_commands.extend(self._traffic_car_light_command(agent_idx, actor_id, transform.location, dt_s))
        ego_idx = self.human_agent_idx
        ego_transform = self._actor_transform(
            ("ego",), self._box_centers[self.ego.id], xs[ego_idx], ys[ego_idx], zs[ego_idx], headings[ego_idx], dt_s
        )
        commands.append(carla.command.ApplyTransform(self.ego.id, ego_transform))
        light_commands.extend(self._ego_light_command(human, wheel_state))
        self.client.apply_batch(commands + light_commands)
        steer_deg = front_wheel_steer_deg(human["steering_angle_rad"])
        if abs(steer_deg - self._sent_steer_deg) >= WHEEL_STEER_RESEND_DEG:
            self.ego.set_wheel_steer_direction(carla.VehicleWheelLocation.FL_Wheel, steer_deg)
            self.ego.set_wheel_steer_direction(carla.VehicleWheelLocation.FR_Wheel, steer_deg)
            self._sent_steer_deg = steer_deg
        self._mirror_traffic_lights()
        frame_id = self.world.tick()
        # show the previous frame while the GPU renders this one, so Python and rendering overlap
        shown_frame_id = frame_id if self._pending_frame_id is None else self._pending_frame_id
        self._pending_frame_id = frame_id
        return self._collect_images(shown_frame_id)

    def _vehicle_light_state(self, braking, reversing, high_beam=False):
        carla = self.carla
        flags = carla.VehicleLightState.NONE
        if self.night:
            flags |= carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
            if high_beam:
                flags |= carla.VehicleLightState.HighBeam
        if braking:
            flags |= carla.VehicleLightState.Brake
        if reversing:
            flags |= carla.VehicleLightState.Reverse
        return flags

    def _traffic_car_light_command(self, agent_idx, actor_id, location, dt_s):
        previous = self._last_speed.get(agent_idx)
        self._last_speed[agent_idx] = (location.x, location.y, previous[2] if previous else 0.0)
        if previous is None:
            return []
        speed = math.hypot(location.x - previous[0], location.y - previous[1]) / dt_s
        braking = (previous[2] - speed) / dt_s > BRAKE_LIGHT_DECEL_MPS2
        self._last_speed[agent_idx] = (location.x, location.y, speed)
        return self._light_command(actor_id, self._vehicle_light_state(braking, False))

    def _ego_light_command(self, human, wheel_state):
        braking = wheel_state.brake_norm > BRAKE_LIGHT_PEDAL
        # CARLA's low beam barely reaches the road, so your own car drives on high beam
        return self._light_command(self.ego.id, self._vehicle_light_state(braking, human["gear"] < 0, high_beam=True))

    def _light_command(self, actor_id, flags):
        if self._vehicle_light_states.get(actor_id) == flags:
            return []
        self._vehicle_light_states[actor_id] = flags
        return [self.carla.command.SetVehicleLightState(actor_id, flags)]

    def _mirror_traffic_lights(self):
        carla = self.carla
        states = self.env.get_traffic_light_states(self._light_state_buffer)
        to_carla = {
            TRAFFIC_LIGHT_STATE_RED: carla.TrafficLightState.Red,
            TRAFFIC_LIGHT_STATE_YELLOW: carla.TrafficLightState.Yellow,
            TRAFFIC_LIGHT_STATE_GREEN: carla.TrafficLightState.Green,
        }
        for light, elements in zip(self._lights, self._light_elements):
            if not elements:
                continue
            target = to_carla.get(int(states[elements[0]]), carla.TrafficLightState.Off)
            if self._light_states.get(light.id) != target:
                light.set_state(target)
                self._light_states[light.id] = target

    def goal_marker(self, human, image):
        """Base and top pixel of a beacon standing on the road at the current goal, in this image's own
        camera pose (the image carries the transform it was rendered from), or None when off-camera."""
        x, y = self.transform.bin_to_loc(human["goal_x"], human["goal_y"])
        ground = self._ground.get(("ego",))
        hint_z = ground[2] if ground is not None else 0.0
        z = self.map.get_waypoint(self.carla.Location(x=x, y=y, z=hint_z)).transform.location.z
        world_to_camera = np.array(image.transform.get_inverse_matrix())
        base = project_to_image(world_to_camera, (x, y, z), image.width, image.height, image.fov)
        top = project_to_image(world_to_camera, (x, y, z + GOAL_BEACON_HEIGHT_M), image.width, image.height, image.fov)
        if base is None or top is None:
            return None
        return base, top

    def _collect_images(self, frame_id):
        images = {"main": self._wait_for_frame(self._frames["main"], frame_id)}
        for name in self._mirrors:
            frames = self._frames[name]
            latest = self._latest_mirror.get(name)
            while latest is None or not frames.empty():
                latest = frames.get(timeout=FRAME_TIMEOUT_S)
            self._latest_mirror[name] = latest
            images[name] = latest
        return images

    @staticmethod
    def _wait_for_frame(frames, frame_id):
        while True:
            image = frames.get(timeout=FRAME_TIMEOUT_S)
            if image.frame >= frame_id:
                return image

    def close(self):
        carla = self.carla
        self._destroy_cameras()
        ids = list(self._actor_ids)
        if getattr(self, "ego", None) is not None:
            ids.append(self.ego.id)
            self.ego = None
        if ids:
            self.client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], True)
        self._actor_ids = []
        self.world.apply_settings(self._original_settings)
