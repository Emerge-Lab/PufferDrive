"""longest6v2 scenario hazards reimplemented for the teleport co-sim loop.

scenario_runner drives these via py_trees + the full leaderboard (which owns the
ego). We keep our PufferDrive-teleport ego and instead, when the ego nears a
scenario's XML trigger point, spawn the hazard actor(s) and run a simple scripted
behavior; the actor then syncs into the ego's observation like any background
vehicle (carla_cosim feeds `ScenarioManager.alive_actors()` into set_agent_states).

Implemented: DynamicObjectCrossing (pedestrian) + a parameterized JunctionVehicle
for SignalizedJunctionLeftTurn / SignalizedJunctionRightTurn /
OppositeVehicleRunningRedLight / VehicleTurningRoute. ControlLoss is intentionally
skipped (it perturbs the ego's own control, which a teleported ego can't model).
"""

import math
import xml.etree.ElementTree as ET

import carla

TRIGGER_RADIUS_M = 35.0


def parse_scenarios(xml_path, route_id):
    root = ET.parse(xml_path).getroot()
    routes = [x for x in root.findall("route") if x.get("id") == str(route_id)]
    if not routes:
        return []
    out = []
    sc = routes[0].find("scenarios")
    for s in sc if sc is not None else []:
        tp = s.find("trigger_point")
        if tp is None:
            continue
        out.append(dict(type=s.get("type"), name=s.get("name"), x=float(tp.get("x")),
                        y=float(tp.get("y")), z=float(tp.get("z")), yaw=float(tp.get("yaw"))))
    return out


def _fwd(yaw_deg):
    r = math.radians(yaw_deg)
    return math.cos(r), math.sin(r)


class Scenario:
    def __init__(self, spec, world, tm, car_bps, walker_bps):
        self.spec, self.world, self.tm = spec, world, tm
        self.car_bps, self.walker_bps = car_bps, walker_bps
        self.triggered = self.done = False
        self.actors = []
        self._t = 0

    def trigger_loc(self):
        return carla.Location(x=self.spec["x"], y=self.spec["y"], z=self.spec["z"])

    def maybe_trigger(self, ego_loc):
        if self.triggered or self.done:
            return
        if ego_loc.distance(self.trigger_loc()) < TRIGGER_RADIUS_M:
            self.triggered = True
            try:
                self._spawn()
                print(f"[scenario] triggered {self.spec['type']} ({self.spec['name']})")
            except Exception as e:  # never let a scenario break the co-sim loop
                print(f"[scenario] {self.spec['type']} spawn failed: {e}")
                self.done = True

    def step(self):
        if self.triggered and not self.done:
            self._t += 1
            try:
                self._behave()
            except Exception:
                self.done = True

    def cleanup(self):
        for a in self.actors:
            try:
                if a is not None and a.is_alive:
                    a.destroy()
            except Exception:
                pass
        self.actors = []

    def alive(self):
        return [a for a in self.actors if a is not None and a.is_alive]

    def _spawn(self):
        raise NotImplementedError

    def _behave(self):
        pass


class DynamicObjectCrossing(Scenario):
    """A pedestrian crosses the road in front of the ego at the trigger."""

    def _spawn(self):
        fx, fy = _fwd(self.spec["yaw"])      # ego forward at the trigger
        lx, ly = -fy, fx                     # left of forward
        loc = self.trigger_loc()
        ahead, side = 6.0, 4.5               # a bit ahead, offset to one side
        sx, sy = loc.x + fx * ahead + lx * side, loc.y + fy * ahead + ly * side
        w = self.world.try_spawn_actor(self.walker_bps[0],
                                       carla.Transform(carla.Location(x=sx, y=sy, z=loc.z + 1.0)))
        if w is None:
            self.done = True
            return
        self.actors = [w]
        self._dir = carla.Vector3D(-lx, -ly, 0.0)  # walk across to the far side
        w.apply_control(carla.WalkerControl(direction=self._dir, speed=2.0))

    def _behave(self):
        if self._t > 60:                     # ~6 s crossing, then stop + finish
            if self.alive():
                self.actors[0].apply_control(carla.WalkerControl(speed=0.0))
            self.done = True


class JunctionVehicle(Scenario):
    """A vehicle drives through the junction near the trigger (cross / oncoming
    traffic). The red-light variant ignores traffic lights."""

    ignore_lights = False

    def _spawn(self):
        cmap = self.world.get_map()
        wp = cmap.get_waypoint(self.trigger_loc())
        jwp = wp
        for _ in range(25):                  # walk forward to the next junction
            nxt = jwp.next(2.0)
            if not nxt:
                break
            jwp = nxt[0]
            if jwp.is_junction:
                break
        if not jwp.is_junction:
            self.done = True
            return
        fx, fy = _fwd(self.spec["yaw"])
        spawn_wp = None
        for entry, _exit in jwp.get_junction().get_waypoints(carla.LaneType.Driving):
            d = entry.transform.get_forward_vector()
            if d.x * fx + d.y * fy < 0.3:    # cross or oncoming (not the ego's direction)
                back = entry.previous(12.0)
                spawn_wp = back[0] if back else entry
                break
        if spawn_wp is None:
            self.done = True
            return
        tf = spawn_wp.transform
        tf.location.z += 0.3
        v = self.world.try_spawn_actor(self.car_bps[0], tf)
        if v is None:
            self.done = True
            return
        v.set_autopilot(True, self.tm.get_port())
        self.tm.vehicle_percentage_speed_difference(v, -25)
        if self.ignore_lights:
            self.tm.ignore_lights_percentage(v, 100.0)
        self.actors = [v]

    def _behave(self):
        if self._t > 120:                    # cleared the junction; leave it as ambient TM traffic
            self.done = True


def _make(spec, world, tm, car_bps, walker_bps):
    t = spec["type"]
    if t == "DynamicObjectCrossing":
        return DynamicObjectCrossing(spec, world, tm, car_bps, walker_bps)
    if t == "OppositeVehicleRunningRedLight":
        s = JunctionVehicle(spec, world, tm, car_bps, walker_bps)
        s.ignore_lights = True
        return s
    if t in ("SignalizedJunctionLeftTurn", "SignalizedJunctionRightTurn", "VehicleTurningRoute"):
        return JunctionVehicle(spec, world, tm, car_bps, walker_bps)
    return None  # ControlLoss and anything else: skipped


class ScenarioManager:
    def __init__(self, specs, world, tm, car_bps, walker_bps):
        self.scenarios = [s for s in (_make(sp, world, tm, car_bps, walker_bps) for sp in specs) if s]

    def tick(self, ego_loc):
        for s in self.scenarios:
            s.maybe_trigger(ego_loc)
            s.step()

    def alive_actors(self):
        out = []
        for s in self.scenarios:
            out.extend(s.alive())
        return out

    def active_count(self):
        return sum(1 for s in self.scenarios if s.triggered and not s.done)

    def cleanup(self):
        for s in self.scenarios:
            s.cleanup()
