"""CARLA <-> PufferDrive co-simulation driver.

PufferDrive simulates only the EGO (its own kinematics) and teleports it into
CARLA; CARLA's Traffic Manager drives all background vehicles and the traffic
lights. Each tick we read CARLA's background poses + light states and overwrite
the matching PufferDrive agents/lights, so the ego's observation reflects the
CARLA world. See pufferlib/ocean/cosim/carla_bridge.py for the transform.

Loop:
  obs  = ego observation (PufferDrive, synced to CARLA)
  act  = policy(obs)                       # or a constant dummy action
  env.step(act)                            # PufferDrive integrates the ego
  teleport CARLA ego to the ego's new pose
  world.tick() x N                         # CARLA advances background + lights
  overwrite PufferDrive background + lights from CARLA
  obs  = env.recompute_observations()

Usage (dummy action, no checkpoint):
  python -m pufferlib.ocean.cosim.carla_cosim --route-id 0 --steps 60
With a policy:
  python -m pufferlib.ocean.cosim.carla_cosim --route-id 0 \
      --checkpoint experiments/puffer_drive_zg6rezam/models/model_puffer_drive_004769.pt
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# `carla` is imported lazily inside the functions that need it (build_carla,
# _route_goal_xy, attach_chase_camera, main), NOT at module scope: BEVRenderer
# and the other pure-python helpers here are reused by
# pufferlib/ocean/cosim/nuplan/planner.py, which runs in the pufferdrive venv
# (cp312) where the carla package (CARLA 0.9.15, cp310/cp37 wheels only) isn't
# installed at all — same principle as carla_bridge.py's lazy carla import.

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.cosim import carla_bridge as cb
from pufferlib.ocean.cosim.arch import shadow_env_kwargs
from pufferlib.ocean.cosim.goals import RouteGoalWindow
# carla_scenarios imports carla at module scope (it's pure CARLA-actor
# manipulation, unlike this module's carla_bridge deps) and is only used
# inside main() below, so it's imported there too.

DEFAULT_ROUTES = "/scratch/yw4142/CaRL/PlanT/data/longest6.xml"
FAR_AWAY = 1.0e6  # park surplus PufferDrive agents out of observation range
SCENARIO_LENGTH_MARGIN_STEPS = 2  # shadow env must never hit its own truncation while the co-sim runs
DEFAULT_VEHICLE_LENGTH_M = 4.5  # fallback size for a parked/dead actor slot (out of obs range anyway)
DEFAULT_VEHICLE_WIDTH_M = 2.0
EGO_BLUEPRINT = "vehicle.lincoln.mkz_2017"  # the longest6/CARLA-leaderboard hero vehicle


def clean_policy_state_dict(state_dict):
    """Strip torch.compile / DDP prefixes. Inlined from pufferlib.pufferl to
    avoid importing the training stack (wandb, neptune, the _C kernel, ...) in
    the co-sim environment — same as cosim/carla/leaderboard_agent.py."""

    def clean(key):
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        return key

    return {clean(k): v for k, v in state_dict.items()}


def load_checkpoint_config(checkpoint):
    """The checkpoint's sibling config.yaml — the single source of truth for the
    obs/action layout the policy expects (and the policy arch). None if no ckpt."""
    if not checkpoint:
        return None
    import yaml

    return yaml.safe_load(open(Path(checkpoint).resolve().parents[1] / "config.yaml"))


DEFAULT_DT = 0.1  # co-sim runs lockstep with CARLA at 0.1s (sub_ticks=1)


def parse_route(xml_path, route_id):
    root = ET.parse(xml_path).getroot()
    routes = [r for r in root.findall("route") if r.get("id") == str(route_id)]
    if not routes:
        raise ValueError(f"route id {route_id} not found in {xml_path}")
    r = routes[0]
    town = r.get("town")
    wps = np.array(
        [[float(p.get("x")), float(p.get("y")), float(p.get("z"))] for p in r.find("waypoints").findall("position")],
        dtype=np.float64,
    )
    return town, wps


def build_carla(client, town, route_wps, num_background, dt_sub):
    import carla

    client.load_world(town)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt_sub
    world.apply_settings(settings)
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    bplib = world.get_blueprint_library()
    car_bps = [b for b in bplib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    cmap = world.get_map()
    spawn_points = cmap.get_spawn_points()

    # Ego: spawn at the CARLA spawn point nearest the route start; physics OFF
    # (we teleport it from PufferDrive each tick).
    start = carla.Location(x=route_wps[0, 0], y=route_wps[0, 1], z=route_wps[0, 2])
    ego_sp = min(spawn_points, key=lambda sp: sp.location.distance(start))
    ego = world.spawn_actor(bplib.find(EGO_BLUEPRINT), ego_sp)
    ego.set_simulate_physics(False)

    # Background: spawn at the spawn points NEAREST the ego (TM autopilot) so the
    # traffic clusters around the ego — visible in the camera and within the ego's
    # observation range, rather than scattered across the whole town.
    bg = []
    for sp in sorted(spawn_points, key=lambda s: s.location.distance(ego_sp.location)):
        if len(bg) >= num_background:
            break
        if sp.location.distance(ego_sp.location) < 8.0:
            continue
        actor = world.try_spawn_actor(car_bps[len(bg) % len(car_bps)], sp)
        if actor is not None:
            actor.set_autopilot(True, tm.get_port())
            bg.append(actor)

    world.tick()
    lights = list(world.get_actors().filter("traffic.traffic_light"))
    return world, tm, ego, bg, lights


def read_background(bg_actors, transform):
    idx, x, y, z, h, vx, vy, yaw_rate, accel_long = [], [], [], [], [], [], [], [], []
    for j, a in enumerate(bg_actors):
        idx.append(1 + j)  # agent 0 = ego; background fills 1..M
        if a is not None and a.is_alive:
            bx, by, bz, bh, bvx, bvy, byr, bal = transform.actor_state_to_bin(a)
        else:  # TM may remove a vehicle; park its PufferDrive slot out of range
            bx = by = bz = FAR_AWAY
            bh = bvx = bvy = byr = bal = 0.0
        x.append(bx)
        y.append(by)
        z.append(bz)
        h.append(bh)
        vx.append(bvx)
        vy.append(bvy)
        yaw_rate.append(byr)
        accel_long.append(bal)
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
    )


def read_actor_sizes(actors):
    """Bounding-box (length, width) in meters for each CARLA actor, so the ego
    observes a truck as a truck. CARLA extent is half-size along the actor's
    local axes (x=forward, y=lateral)."""
    idx, length, width = [], [], []
    for j, a in enumerate(actors):
        idx.append(1 + j)
        if a is not None and a.is_alive:
            ext = a.bounding_box.extent
            length.append(2.0 * ext.x)
            width.append(2.0 * ext.y)
        else:
            length.append(DEFAULT_VEHICLE_LENGTH_M)
            width.append(DEFAULT_VEHICLE_WIDTH_M)
    return (np.array(idx, np.int32), np.array(length, np.float32), np.array(width, np.float32))


def densify_route(route_wps, step=2.0):
    """Interpolate the sparse longest6v2 route waypoints to ~step-m spacing."""
    pts = [route_wps[0, :2].astype(float)]
    for i in range(len(route_wps) - 1):
        a, b = route_wps[i, :2].astype(float), route_wps[i + 1, :2].astype(float)
        seg = float(np.hypot(*(b - a)))
        n = max(1, int(seg / step))
        for k in range(1, n + 1):
            pts.append(a + (b - a) * (k / n))
    return np.asarray(pts)


def _route_goal_xy(cmap, cx, cy, route_yaw_deg):
    """CARLA-frame (x, y, z) for a route goal: snap to the nearest DRIVING lane
    whose travel direction matches the route (within 90 deg) so a goal never lands
    in the oncoming lane. `cmap.get_waypoint` returns the nearest lane regardless of
    direction, so at the center of a two-way road it can pick the oncoming one; we
    then search lateral neighbors for a matching-direction lane. If none is reachable
    (e.g. across a solid center line), keep the raw route point — it is at least in
    the correct travel direction even if not perfectly lane-centered."""
    import carla

    def matches(w):
        return abs(((route_yaw_deg - w.transform.rotation.yaw) + 180.0) % 360.0 - 180.0) <= 90.0

    wp = cmap.get_waypoint(carla.Location(x=float(cx), y=float(cy)))
    if wp is None:
        return (cx, cy, 0.0)
    if matches(wp):
        loc = wp.transform.location
        return (loc.x, loc.y, loc.z)
    for go_left in (True, False):
        w = wp
        for _ in range(4):
            w = w.get_left_lane() if go_left else w.get_right_lane()
            if w is None or w.road_id != wp.road_id:
                break
            if str(w.lane_type) == "Driving" and matches(w):
                loc = w.transform.location
                return (loc.x, loc.y, loc.z)
    return (cx, cy, 0.0)


def build_route_goals(dense_route, transform, cmap, spacing=20.0):
    """Fixed sequence of lane-centered goals along the WHOLE route, one every
    `spacing` m, each snapped to a DIRECTION-MATCHED driving lane, in the bin frame.
    Returns an (N, 5) array (x, y, z, dir_x, dir_y) where dir_* is the local route
    travel direction at the goal (bin frame), consumed by set_agent_goals' lane
    snapping. The ego marches through these via a cursor that only advances on
    arrival (GOAL_RADIUS_M) — goals do not float with the ego."""

    def goal_at(i):
        d = dense_route[i] - dense_route[i - 1]
        route_yaw = np.degrees(np.arctan2(d[1], d[0]))  # route travel direction (CARLA frame)
        gx, gy, gz = _route_goal_xy(cmap, dense_route[i][0], dense_route[i][1], route_yaw)
        return (*transform.loc_to_bin(gx, gy), gz, d[0], -d[1])  # y flips into the bin frame

    goals, next_at, cum = [], spacing, 0.0
    for i in range(1, len(dense_route)):
        cum += float(np.hypot(*(dense_route[i] - dense_route[i - 1])))
        if cum >= next_at:
            goals.append(goal_at(i))
            next_at += spacing
    goals.append(goal_at(len(dense_route) - 1))  # always finish on the route's end
    return np.array(goals, np.float32)


def attach_chase_camera(world, ego, w=960, h=540):
    """RGB chase camera attached to the ego (behind + above, looking forward).
    Returns (camera_actor, image_queue). Requires CARLA running WITH rendering
    (-RenderOffScreen), not -nullrhi."""
    import queue

    import carla

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", "90")
    cam = world.spawn_actor(
        bp, carla.Transform(carla.Location(x=-6.5, z=3.2), carla.Rotation(pitch=-12.0)), attach_to=ego
    )
    q = queue.Queue()
    cam.listen(q.put)
    return cam, q


def carla_image_to_rgb(img):
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return arr[:, :, [2, 1, 0]].copy()  # BGRA -> RGB


class BEVRenderer:
    """Top-down render of the PufferDrive scene (ego red, CARLA-synced background
    blue, ego goals gold), centered on the ego. Frames -> mp4."""

    def __init__(self, town_bin, out_path, span=70.0):
        import data_utils.mirror_map_bin as mbin

        data = mbin.read_bin(Path(town_bin))
        self.roads = [(np.asarray(r["x"]), np.asarray(r["y"]), r["type"]) for r in data["roads"]]
        # traffic-control stop lines (2 endpoints in xy), drawn colored by live state
        self.traffic = [np.asarray(t["stop_line"], dtype=float).reshape(2, 3)[:, :2] for t in data["traffic"]]
        self.out_path = out_path
        self.span = span
        self.frames = []

    # UNKNOWN=0 RED=1 YELLOW=2 GREEN=3 OFF=4 (datatypes.h)
    _LIGHT_COLOR = {1: "red", 2: "orange", 3: "limegreen", 4: "0.5", 0: "0.7"}

    def capture(self, agents, ego_idx=0, goals=None, light_states=None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtransforms
        from matplotlib.patches import Rectangle

        ex, ey = float(agents["x"][ego_idx]), float(agents["y"][ego_idx])
        fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
        for rx, ry, rt in self.roads:
            col = "0.6" if 0 <= rt <= 9 else ("0.25" if 20 <= rt <= 29 else "0.8")
            ax.plot(rx, ry, color=col, lw=0.6, zorder=1)
        for i in range(len(agents["x"])):
            x, y = float(agents["x"][i]), float(agents["y"][i])
            if abs(x - ex) > self.span or abs(y - ey) > self.span:
                continue  # surplus agents parked far away
            h, L, W = float(agents["heading"][i]), float(agents["length"][i]), float(agents["width"][i])
            rect = Rectangle((-L / 2, -W / 2), L, W, color="red" if i == ego_idx else "tab:blue", alpha=0.95, zorder=3)
            rect.set_transform(mtransforms.Affine2D().rotate(h).translate(x, y) + ax.transData)
            ax.add_patch(rect)
        if goals is not None:
            ax.scatter(goals[0], goals[1], c="gold", marker="*", s=70, zorder=4, edgecolors="k", linewidths=0.4)
        if light_states is not None:
            for j, sl in enumerate(self.traffic):
                if j >= len(light_states):
                    break
                mx, my = sl[:, 0].mean(), sl[:, 1].mean()
                if abs(mx - ex) > self.span or abs(my - ey) > self.span:
                    continue
                c = self._LIGHT_COLOR.get(int(light_states[j]), "0.7")
                ax.plot(sl[:, 0], sl[:, 1], color=c, lw=3, zorder=2)  # stop line, colored by state
                ax.scatter([mx], [my], c=c, s=45, marker="s", zorder=4, edgecolors="k", linewidths=0.4)
        ax.set_xlim(ex - self.span, ex + self.span)
        ax.set_ylim(ey - self.span, ey + self.span)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.canvas.draw()
        self.frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    def save(self, fps=10):
        write_mp4(self.out_path, self.frames, fps=fps)
        print(f"[cosim] wrote {self.out_path} ({len(self.frames)} frames)")


class Mp4Writer:
    """Streams RGB uint8 frames to an mp4 via OpenCV."""

    def __init__(self, out_path, fps):
        self.out_path = str(out_path)
        self.fps = float(fps)
        self.writer = None
        self.frame_count = 0

    def append_data(self, frame):
        import cv2

        frame = np.ascontiguousarray(frame)
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.out_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Mp4Writer({self.out_path}): OpenCV could not open the video writer")
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def write_mp4(out_path, frames, fps=10):
    if not frames:
        raise ValueError(f"write_mp4({out_path}): no frames")
    writer = Mp4Writer(out_path, fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def main():
    import carla

    from pufferlib.ocean.cosim import carla_scenarios

    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=DEFAULT_ROUTES)
    ap.add_argument("--route-id", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--num-background", type=int, default=30)
    ap.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help="shadow agent pool (default: the checkpoint's max_agents_per_env, else 64)",
    )
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--carla-host", default="localhost")
    ap.add_argument("--carla-port", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sub-ticks", type=int, default=None)  # default: round(dt / 0.1)
    ap.add_argument(
        "--dt", type=float, default=None, help="ego dynamics dt (default: the checkpoint's training dt, else 0.1)"
    )
    ap.add_argument("--render", default=None, help="output mp4 path for a top-down BEV render")
    ap.add_argument("--bev-span", type=float, default=70.0, help="BEV half-window in meters")
    ap.add_argument(
        "--carla-view",
        default=None,
        help="output mp4 path for a CARLA chase-camera view (needs CARLA -RenderOffScreen)",
    )
    ap.add_argument(
        "--town-bin",
        default=None,
        help="override the town .bin (e.g. a shoulder-retyped variant); default: repo carla bin",
    )
    args = ap.parse_args()

    town, route_wps = parse_route(args.routes, args.route_id)
    town_bin = args.town_bin or cb.bin_path_for_town(town)
    print(f"[cosim] route {args.route_id} town={town} waypoints={len(route_wps)} bin={town_bin}")

    cfg = load_checkpoint_config(args.checkpoint)  # for the env arch + policy arch (below)
    # Policy dt defaults to the checkpoint's training dt (matching dynamics).
    dt = args.dt if args.dt is not None else ((cfg or {}).get("env", {}).get("dt", DEFAULT_DT))
    sub_ticks = args.sub_ticks or max(1, round(dt / 0.1))  # CARLA at 0.1s, PufferDrive at dt

    # Shadow agent pool: the checkpoint's per-env training cap by default, so
    # binding.shared builds exactly one gigaflow C env (the co-sim setters only
    # address env 0).
    num_agents = args.num_agents or int(((cfg or {}).get("env") or {}).get("max_agents_per_env", 64))

    # PufferDrive env: one policy agent (the ego) plus static partner slots
    # streamed from CARLA. Env kwargs come from the checkpoint's config.yaml with
    # the clean-eval profile applied on top (see cosim/arch.py).
    env = Drive(
        **shadow_env_kwargs(
            cfg,
            overrides=dict(
                map_dir=town_bin,
                num_maps=1,
                num_agents=1,
                min_agents_per_env=1,
                cosim_partner_slots=num_agents - 1,
                goal_source="external",
                scenario_length=args.steps + SCENARIO_LENGTH_MARGIN_STEPS,
                resample_frequency=0,
                dt=dt,
                # External sim owns the episode: a training-config
                # termination_mode=1 would c_reset() the pool (ego included) once
                # the parked FAR_AWAY slots latch under "stop" infraction behaviors.
                termination_mode=0,
                # Enforcement off (flags still fire): in one endless episode a
                # "stop" latch is permanent and would freeze the ego for good.
                collision_behavior="ignore",
                offroad_behavior="ignore",
                traffic_light_behavior="ignore",
            ),
        )
    )
    obs, _ = env.reset()
    obs = np.asarray(obs)
    n_slots = 1 + env.cosim_partner_slots
    print(f"[cosim] env reset: obs {obs.shape}, action {env.single_action_space}, dt={dt}, sub_ticks={sub_ticks}")

    # Policy (optional) — falls back to a constant forward jerk action.
    policy = None
    if cfg is not None:
        import torch
        import pufferlib.ocean.torch as drive_torch

        # pre-3.0 checkpoints keep action_type only in the env section
        cfg["policy"].setdefault("action_type", cfg["env"]["action_type"])
        policy = getattr(drive_torch, cfg.get("policy_name", "Drive"))(env, **cfg["policy"]).to(args.device)
        sd = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        policy.load_state_dict(clean_policy_state_dict(sd))
        policy.eval()
        print(f"[cosim] loaded policy from {args.checkpoint}")
    # jerk action: idx = j_long_idx*3 + j_lat_idx; (3,1) = +long jerk, straight (accelerate)
    dummy_action = np.full((1, 1), 3 * 3 + 1, dtype=np.int32)

    # CARLA
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(60.0)
    world, tm, ego, bg, lights = build_carla(client, town, route_wps, args.num_background, 0.1)
    cmap = world.get_map()
    transform = cb.CarlaTransform(town, offset=cb.town_offset(town_bin))
    _bin_lanes = cb._bin_lane_points(town_bin)  # bin lane points (== global frame) for diagnostics
    dense_route = densify_route(route_wps)  # fine-sampled route for lane-centered goal placement
    route_goals = build_route_goals(dense_route, transform, cmap)  # fixed 20-m lane-centered goal sequence
    goal_window = RouteGoalWindow(env, route_goals)
    light_map, num_traffic = cb.map_lights_to_bin(lights, transform, town_bin)
    print(
        f"[cosim] carla: ego + {len(bg)} background + {len(lights)} lights; offset={transform.tx:.1f},{transform.ty:.1f}"
    )

    car_bps = [
        b for b in world.get_blueprint_library().filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4
    ]
    walker_bps = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    scenario_mgr = carla_scenarios.ScenarioManager(
        carla_scenarios.parse_scenarios(args.routes, args.route_id), world, tm, car_bps, walker_bps
    )
    print(f"[cosim] {len(scenario_mgr.scenarios)} scenarios loaded (ControlLoss skipped)")

    # --- init sync: ego pose + size + goals; background; park surplus agents ---
    ego_state = transform.actor_state_to_bin(ego)
    env.set_agent_states(np.array([0], np.int32), *[np.array([v], np.float32) for v in ego_state])
    ego_ext = ego.bounding_box.extent  # match the ego to its CARLA blueprint box (static)
    env.set_agent_sizes(
        np.array([0], np.int32), np.array([2.0 * ego_ext.x], np.float32), np.array([2.0 * ego_ext.y], np.float32)
    )
    goal_window.sync(ego_state[0], ego_state[1], ego_state[3])
    bg_idx, *_ = read_background(bg, transform)
    surplus = np.arange(1 + len(bg), n_slots, dtype=np.int32)
    if len(surplus):
        z6 = np.full(len(surplus), FAR_AWAY, np.float32)
        zero = np.zeros_like(z6)
        env.set_agent_states(surplus, z6, z6, z6, zero, zero, zero, zero, zero)
    obs = np.asarray(env.recompute_observations())

    bev = BEVRenderer(town_bin, args.render, span=args.bev_span) if args.render else None
    cam, cam_q = attach_chase_camera(world, ego) if args.carla_view else (None, None)
    carla_frames = []

    # --- co-sim loop ---
    for step in range(args.steps):
        if policy is not None:
            import torch
            import pufferlib.pytorch

            import pufferlib.spaces

            # A discrete policy head on a continuous env needs the bin->continuous
            # mapping (pufferl.py feeds cont_action to the env, never the bin indices).
            env_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
            action_selection = (
                pufferlib.pytorch.ACTION_SELECT_MEAN
                if env_continuous and not policy.is_continuous
                else pufferlib.pytorch.ACTION_SELECT_MODE
            )
            with torch.no_grad():
                logits, _ = policy.forward_eval(torch.as_tensor(obs).to(args.device))
                action, _, _, cont_action = pufferlib.pytorch.sample_logits(
                    logits,
                    action_selection=action_selection,
                    env_continuous=env_continuous,
                    policy=policy,
                )
                env_action = cont_action if env_continuous and cont_action is not None else action
                act = env_action.cpu().numpy().reshape(1, -1)
                if not env_continuous:
                    act = act.astype(np.int32)
        else:
            act = dummy_action

        env.step(act)  # PufferDrive integrates the ego (and others; overwritten below)
        ego_bin = env.get_global_agent_state()
        # teleport CARLA ego to the ego's new pose
        ex, ey = transform.bin_to_loc(float(ego_bin["x"][0]), float(ego_bin["y"][0]))
        eyaw = transform.bin_heading_to_yaw(float(ego_bin["heading"][0]))
        ego.set_transform(carla.Transform(carla.Location(x=ex, y=ey, z=0.3), carla.Rotation(yaw=eyaw)))
        scenario_mgr.tick(ego.get_location())  # trigger/spawn hazards near the ego
        cam_img = None
        for _ in range(sub_ticks):
            world.tick()
            if cam_q is not None:
                cam_img = cam_q.get(timeout=20.0)
        if cam_img is not None:
            carla_frames.append(carla_image_to_rgb(cam_img))
        # overwrite background + scenario actors + lights from CARLA
        actors = [a for a in bg if a.is_alive] + scenario_mgr.alive_actors()
        env.set_agent_states(*read_background(actors, transform))
        env.set_agent_sizes(*read_actor_sizes(actors))  # true CARLA bounding-box sizes
        n_used = 1 + len(actors)  # ego + actors; park the rest (count varies)
        if n_used < n_slots:
            sp = np.arange(n_used, n_slots, dtype=np.int32)
            zf = np.full(len(sp), FAR_AWAY, np.float32)
            zz = np.zeros_like(zf)
            env.set_agent_states(sp, zf, zf, zf, zz, zz, zz, zz, zz)
        states = np.zeros(num_traffic, np.int32)
        for li, lt in enumerate(lights):
            state = cb.carla_light_to_puffer(lt.get_state())
            for j in light_map[li]:
                if 0 <= j < num_traffic:
                    states[j] = state
        env.set_traffic_light_states(states)
        ebx, eby = float(ego_bin["x"][0]), float(ego_bin["y"][0])
        goal_window.sync(ebx, eby, float(ego_bin["heading"][0]))
        gx, gy = goal_window.window[:, 0], goal_window.window[:, 1]
        obs = np.asarray(env.recompute_observations())

        if bev is not None:
            bev.capture(env.get_global_agent_state(include_static=True), ego_idx=0, goals=(gx, gy), light_states=states)

        if step % 10 == 0:
            el = ego.get_location()
            nl = min(lights, key=lambda lt: lt.get_location().distance(el)) if lights else None
            ls = f"{nl.get_state()}@{nl.get_location().distance(el):.0f}m" if nl else "n/a"
            goal_cursor = goal_window.current_index
            gd = float(np.hypot(route_goals[goal_cursor, 0] - ebx, route_goals[goal_cursor, 1] - eby))
            near = sum(1 for a in bg if a.is_alive and a.get_location().distance(el) < 40.0)
            wp = cmap.get_waypoint(ego.get_location())  # nearest drivable lane center
            lat = ego.get_location().distance(wp.transform.location) if wp else -1.0
            onroad = cmap.get_waypoint(ego.get_location(), project_to_road=False) is not None
            bin_off = (
                float(np.min(np.hypot(_bin_lanes[:, 0] - ebx, _bin_lanes[:, 1] - eby))) if len(_bin_lanes) else -1.0
            )
            print(
                f"[cosim] step {step}: ego carla=({ex:.1f},{ey:.1f})  carla_off={lat:.1f}m onroad={onroad}  bin_off={bin_off:.1f}m  goal0={gd:.0f}m[{goal_cursor}/{len(route_goals)}] light={ls}  scn={scenario_mgr.active_count()}"
            )

    # cleanup
    if bev is not None:
        bev.save()
    if carla_frames:
        write_mp4(args.carla_view, carla_frames, fps=10)
        print(f"[cosim] wrote {args.carla_view} ({len(carla_frames)} frames)")
    # Teardown while STILL synchronous: stop sensors, detach the TM from its
    # vehicles, destroy actors, tick to apply — and only then release the world
    # to async. Going async first races the TM thread against the destruction
    # batch and aborts the process with "trying to operate on a destroyed
    # actor" (std::terminate in the TM worker thread).
    if cam is not None and cam.is_alive:
        cam.stop()
    tm_vehicles = [
        a for a in [*bg, *scenario_mgr.alive_actors()] if a is not None and a.is_alive and "vehicle" in a.type_id
    ]
    client.apply_batch_sync([carla.command.SetAutopilot(a.id, False, tm.get_port()) for a in tm_vehicles], True)
    world.tick()  # let the TM observe the detach before anything is destroyed
    scenario_mgr.cleanup()
    client.apply_batch_sync(
        [carla.command.DestroyActor(a) for a in [cam, ego, *bg] if a is not None and a.is_alive], True
    )
    world.tick()
    s = world.get_settings()
    s.synchronous_mode = False
    world.apply_settings(s)
    tm.set_synchronous_mode(False)
    print("[cosim] done")


if __name__ == "__main__":
    main()
