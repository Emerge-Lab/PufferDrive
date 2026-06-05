"""CARLA <-> PufferDrive co-simulation driver.

PufferDrive simulates only the EGO (its own kinematics) and teleports it into
CARLA; CARLA's Traffic Manager drives all background vehicles and the traffic
lights. Each tick we read CARLA's background poses + light states and overwrite
the matching PufferDrive agents/lights, so the ego's observation reflects the
CARLA world. See pufferlib/ocean/drive/carla_bridge.py for the transform.

Loop:
  obs  = ego observation (PufferDrive, synced to CARLA)
  act  = policy(obs)                       # or a constant dummy action
  env.step(act)                            # PufferDrive integrates the ego
  teleport CARLA ego to the ego's new pose
  world.tick() x N                         # CARLA advances background + lights
  overwrite PufferDrive background + lights from CARLA
  obs  = env.recompute_observations()

Usage (dummy action, no checkpoint):
  python -m pufferlib.ocean.drive.carla_cosim --route-id 0 --steps 60
With a policy:
  python -m pufferlib.ocean.drive.carla_cosim --route-id 0 \
      --checkpoint experiments/puffer_drive_zg6rezam/models/model_puffer_drive_004769.pt
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import carla

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import carla_bridge as cb
from pufferlib.ocean.drive import carla_scenarios

DEFAULT_ROUTES = "/workspace/CaRL/PlanT/data/longest6.xml"
FAR_AWAY = 1.0e6  # park surplus PufferDrive agents out of observation range
GOAL_RADIUS_M = 6.0  # advance the route-goal cursor only when the ego arrives within this
DEFAULT_VEHICLE_LENGTH_M = 4.5  # fallback size for a parked/dead actor slot (out of obs range anyway)
DEFAULT_VEHICLE_WIDTH_M = 2.0
EGO_BLUEPRINT = "vehicle.lincoln.mkz_2017"  # the longest6/CARLA-leaderboard hero vehicle

def load_checkpoint_config(checkpoint):
    """The checkpoint's sibling config.yaml — the single source of truth for the
    obs/action layout the policy expects (and the policy arch). None if no ckpt."""
    if not checkpoint:
        return None
    import yaml

    return yaml.safe_load(open(Path(checkpoint).resolve().parents[1] / "config.yaml"))


# Eval-time env config for the carla_combined gigaflow policy (obs dropout 0,
# unlike the 0.5/0.4 used at train time — matches how the policy is evaluated).
CARLA_ARCH = dict(
    num_target_waypoints=3,
    obs_slots_lane_n=80,
    obs_slots_boundary_n=40,
    obs_slots_partners_n=16,
    obs_slots_traffic_controls_n=4,
    obs_range_partner_m=200.0,
    obs_range_road_front_m=200.0,
    obs_range_road_behind_m=40.0,
    obs_range_road_side_m=50.0,
    obs_range_traffic_control_m=100.0,
    obs_norm_xy_offset_m=200.0,
    obs_norm_goal_offset_m=200.0,
    obs_norm_road_seg_length_m=10.0,
    obs_norm_road_seg_width_m=5.0,
    obs_norm_veh_length_m=15.0,
    obs_norm_veh_width_m=10.0,
    reward_conditioning=True,
    target_type="static",
    dynamics_model="jerk",
    dt=0.1,  # co-sim runs lockstep with CARLA at 0.1s (sub_ticks=1)
)


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


def map_lights_to_bin(lights, transform, town_bin, num_traffic):
    """Map each CARLA traffic light to a bin traffic-element index by nearest
    stop-line / location (transformed into the bin frame)."""
    import data_utils.mirror_map_bin as mbin

    data = mbin.read_bin(Path(town_bin))
    tl_pos = []  # bin-frame position per traffic element
    for t in data["traffic"]:
        sx = 0.5 * (t["stop_line"][0] + t["stop_line"][3])
        sy = 0.5 * (t["stop_line"][1] + t["stop_line"][4])
        tl_pos.append((sx, sy))
    tl_pos = np.array(tl_pos) if tl_pos else np.zeros((0, 2))
    mapping = []  # mapping[i] = bin element idx for lights[i]
    for lt in lights:
        loc = lt.get_transform().location
        bx, by = transform.loc_to_bin(loc.x, loc.y)
        if len(tl_pos):
            j = int(((tl_pos[:, 0] - bx) ** 2 + (tl_pos[:, 1] - by) ** 2).argmin())
        else:
            j = -1
        mapping.append(j)
    return mapping, num_traffic


def read_background(bg_actors, transform):
    idx, x, y, z, h, vx, vy = [], [], [], [], [], [], []
    for j, a in enumerate(bg_actors):
        idx.append(1 + j)  # agent 0 = ego; background fills 1..M
        if a is not None and a.is_alive:
            bx, by, bz, bh, bvx, bvy = transform.actor_state_to_bin(a)
        else:  # TM may remove a vehicle; park its PufferDrive slot out of range
            bx = by = bz = FAR_AWAY
            bh = bvx = bvy = 0.0
        x.append(bx); y.append(by); z.append(bz); h.append(bh); vx.append(bvx); vy.append(bvy)
    return (np.array(idx, np.int32), np.array(x, np.float32), np.array(y, np.float32),
            np.array(z, np.float32), np.array(h, np.float32), np.array(vx, np.float32), np.array(vy, np.float32))


def read_actor_sizes(actors):
    """Bounding-box (length, width) in meters for each CARLA actor, so the ego
    observes a truck as a truck. CARLA extent is half-size along the actor's
    local axes (x=forward, y=lateral)."""
    idx, length, width = [], [], []
    for j, a in enumerate(actors):
        idx.append(1 + j)
        if a is not None and a.is_alive:
            ext = a.bounding_box.extent
            length.append(2.0 * ext.x); width.append(2.0 * ext.y)
        else:
            length.append(DEFAULT_VEHICLE_LENGTH_M); width.append(DEFAULT_VEHICLE_WIDTH_M)
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


def build_route_goals(dense_route, transform, cmap, spacing=20.0):
    """Fixed sequence of lane-centered goals along the WHOLE route, one every
    `spacing` m, each snapped to the CARLA lane center, in the bin frame. Returns
    an (N, 3) array (x, y, z). The ego marches through these via a cursor that
    only advances on arrival (see GOAL_RADIUS_M) — goals do not float with the ego."""
    def lane_goal(cx, cy):
        wp = cmap.get_waypoint(carla.Location(x=float(cx), y=float(cy)))  # snap to lane center
        lc = wp.transform.location if wp is not None else carla.Location(x=float(cx), y=float(cy), z=0.0)
        return (*transform.loc_to_bin(lc.x, lc.y), lc.z)

    goals, next_at, cum = [], spacing, 0.0
    for i in range(1, len(dense_route)):
        cum += float(np.hypot(*(dense_route[i] - dense_route[i - 1])))
        if cum >= next_at:
            goals.append(lane_goal(*dense_route[i]))
            next_at += spacing
    goals.append(lane_goal(*dense_route[-1]))  # always finish on the route's end
    return np.array(goals, np.float32)


def select_goals(route_goals, cursor, num=3):
    """The next `num` goals from the cursor (clamped at the route's end), as
    (gx, gy, gz) arrays for set_agent_goals."""
    idx = [min(cursor + k, len(route_goals) - 1) for k in range(num)]
    sel = route_goals[idx]
    return sel[:, 0].copy(), sel[:, 1].copy(), sel[:, 2].copy()


def attach_chase_camera(world, ego, w=960, h=540):
    """RGB chase camera attached to the ego (behind + above, looking forward).
    Returns (camera_actor, image_queue). Requires CARLA running WITH rendering
    (-RenderOffScreen), not -nullrhi."""
    import queue

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
        self.out_path = out_path
        self.span = span
        self.frames = []

    def capture(self, agents, ego_idx=0, goals=None):
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
            rect = Rectangle((-L / 2, -W / 2), L, W, color="red" if i == ego_idx else "tab:blue",
                             alpha=0.95, zorder=3)
            rect.set_transform(mtransforms.Affine2D().rotate(h).translate(x, y) + ax.transData)
            ax.add_patch(rect)
        if goals is not None:
            ax.scatter(goals[0], goals[1], c="gold", marker="*", s=70, zorder=4, edgecolors="k", linewidths=0.4)
        ax.set_xlim(ex - self.span, ex + self.span)
        ax.set_ylim(ey - self.span, ey + self.span)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.canvas.draw()
        self.frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    def save(self, fps=10):
        import imageio

        imageio.mimwrite(self.out_path, self.frames, fps=fps, codec="libx264")
        print(f"[cosim] wrote {self.out_path} ({len(self.frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default=DEFAULT_ROUTES)
    ap.add_argument("--route-id", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--num-background", type=int, default=30)
    ap.add_argument("--num-agents", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--carla-host", default="localhost")
    ap.add_argument("--carla-port", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sub-ticks", type=int, default=None)  # default: round(dt / 0.1)
    ap.add_argument("--dt", type=float, default=None, help="ego dynamics dt (default CARLA_ARCH=0.1; policy trained at 0.3)")
    ap.add_argument("--render", default=None, help="output mp4 path for a top-down BEV render")
    ap.add_argument("--bev-span", type=float, default=70.0, help="BEV half-window in meters")
    ap.add_argument("--carla-view", default=None,
                    help="output mp4 path for a CARLA chase-camera view (needs CARLA -RenderOffScreen)")
    ap.add_argument("--town-bin", default=None,
                    help="override the town .bin (e.g. a shoulder-retyped variant); default: repo carla bin")
    args = ap.parse_args()

    town, route_wps = parse_route(args.routes, args.route_id)
    town_bin = args.town_bin or cb.bin_path_for_town(town)
    print(f"[cosim] route {args.route_id} town={town} waypoints={len(route_wps)} bin={town_bin}")

    cfg = load_checkpoint_config(args.checkpoint)  # for the policy arch (below)
    dt = args.dt if args.dt is not None else CARLA_ARCH["dt"]
    sub_ticks = args.sub_ticks or max(1, round(dt / 0.1))  # CARLA at 0.1s, PufferDrive at dt

    # PufferDrive env (gigaflow spawns an agent pool; agent 0 = ego, rest = background).
    env = Drive(
        map_dir=town_bin, num_maps=1, num_agents=args.num_agents,
        simulation_mode="gigaflow", control_mode="control_vehicles",
        scenario_length=1_000_000, resample_frequency=0,
        # The co-sim sets the ego's goals from the route and overwrites all
        # background agents each tick, so gigaflow's reset-time goals are
        # throwaway. goal_on_lane=False places them at random drivable points
        # (no lane-graph routing), which keeps reset fast even when a patched
        # bin has had non-driving lanes retyped out of the routable network.
        goal_on_lane=False,
        **{**CARLA_ARCH, "dt": dt},
    )
    obs, _ = env.reset()
    obs = np.asarray(obs)
    n_active = int(env.num_agents)
    print(f"[cosim] env reset: obs {obs.shape}, action {env.single_action_space}, dt={dt}, sub_ticks={sub_ticks}")

    # Policy (optional) — falls back to a constant forward jerk action.
    policy = None
    if cfg is not None:
        import torch
        import pufferlib.ocean.torch as drive_torch
        from pufferlib.pufferl import clean_policy_state_dict

        policy = getattr(drive_torch, cfg.get("policy_name", "Drive"))(env, **cfg["policy"]).to(args.device)
        sd = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        policy.load_state_dict(clean_policy_state_dict(sd))
        policy.eval()
        print(f"[cosim] loaded policy from {args.checkpoint}")
    # jerk action: idx = j_long_idx*3 + j_lat_idx; (3,1) = +long jerk, straight (accelerate)
    dummy_action = np.full((n_active, 1), 3 * 3 + 1, dtype=np.int32)

    # CARLA
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(60.0)
    world, tm, ego, bg, lights = build_carla(client, town, route_wps, args.num_background, 0.1)
    cmap = world.get_map()
    transform = cb.CarlaTransform(town, offset=cb.town_offset(cmap, town_bin))
    _bin_lanes = cb._bin_lane_points(town_bin)  # bin lane points (== global frame) for diagnostics
    dense_route = densify_route(route_wps)  # fine-sampled route for lane-centered goal placement
    route_goals = build_route_goals(dense_route, transform, cmap)  # fixed 20-m lane-centered goal sequence
    goal_cursor = 0  # index of the current (not-yet-reached) route goal
    light_map, num_traffic = map_lights_to_bin(lights, transform, town_bin, len(lights))
    print(f"[cosim] carla: ego + {len(bg)} background + {len(lights)} lights; offset={transform.tx:.1f},{transform.ty:.1f}")

    car_bps = [b for b in world.get_blueprint_library().filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
    walker_bps = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    scenario_mgr = carla_scenarios.ScenarioManager(
        carla_scenarios.parse_scenarios(args.routes, args.route_id), world, tm, car_bps, walker_bps)
    print(f"[cosim] {len(scenario_mgr.scenarios)} scenarios loaded (ControlLoss skipped)")

    # --- init sync: ego pose + size + goals; background; park surplus agents ---
    ego_state = transform.actor_state_to_bin(ego)
    env.set_agent_states(np.array([0], np.int32), *[np.array([v], np.float32) for v in ego_state])
    ego_ext = ego.bounding_box.extent  # match the ego to its CARLA blueprint box (static)
    env.set_agent_sizes(np.array([0], np.int32),
                        np.array([2.0 * ego_ext.x], np.float32), np.array([2.0 * ego_ext.y], np.float32))
    gx, gy, gz = select_goals(route_goals, goal_cursor)
    env.set_agent_goals(0, gx, gy, gz)
    bg_idx, *_ = (read_background(bg, transform))
    surplus = np.arange(1 + len(bg), n_active, dtype=np.int32)
    if len(surplus):
        z6 = np.full(len(surplus), FAR_AWAY, np.float32)
        env.set_agent_states(surplus, z6, z6, z6, np.zeros_like(z6), np.zeros_like(z6), np.zeros_like(z6))
    obs = np.asarray(env.recompute_observations())

    bev = BEVRenderer(town_bin, args.render, span=args.bev_span) if args.render else None
    cam, cam_q = attach_chase_camera(world, ego) if args.carla_view else (None, None)
    carla_frames = []

    # --- co-sim loop ---
    for step in range(args.steps):
        if policy is not None:
            import torch
            import pufferlib.pytorch

            with torch.no_grad():
                logits, _ = policy.forward_eval(torch.as_tensor(obs).to(args.device))
                action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                act = action.cpu().numpy().reshape(n_active, -1).astype(np.int32)
        else:
            act = dummy_action

        env.step(act)  # PufferDrive integrates the ego (and others; overwritten below)
        ego_bin = env.get_global_agent_state()
        # teleport CARLA ego to the ego's new pose
        ex, ey = transform.bin_to_loc(float(ego_bin["x"][0]), float(ego_bin["y"][0]))
        eyaw = transform.bin_heading_to_yaw(float(ego_bin["heading"][0]))
        ego.set_transform(carla.Transform(carla.Location(x=ex, y=ey, z=0.3),
                                           carla.Rotation(yaw=eyaw)))
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
        if n_used < n_active:
            sp = np.arange(n_used, n_active, dtype=np.int32)
            zf = np.full(len(sp), FAR_AWAY, np.float32)
            zz = np.zeros_like(zf)
            env.set_agent_states(sp, zf, zf, zf, zz, zz, zz)
        states = np.zeros(num_traffic, np.int32)
        for li, lt in enumerate(lights):
            j = light_map[li]
            if 0 <= j < num_traffic:
                states[j] = cb.carla_light_to_puffer(lt.get_state())
        env.set_traffic_light_states(states)
        # advance the goal cursor only once the ego actually reaches the current goal
        ebx, eby = float(ego_bin["x"][0]), float(ego_bin["y"][0])
        while (goal_cursor < len(route_goals) - 1
               and np.hypot(route_goals[goal_cursor, 0] - ebx, route_goals[goal_cursor, 1] - eby) < GOAL_RADIUS_M):
            goal_cursor += 1
        gx, gy, gz = select_goals(route_goals, goal_cursor)
        env.set_agent_goals(0, gx, gy, gz)
        obs = np.asarray(env.recompute_observations())

        if bev is not None:
            bev.capture(env.get_global_agent_state(), ego_idx=0, goals=(gx, gy))

        if step % 10 == 0:
            el = ego.get_location()
            nl = min(lights, key=lambda lt: lt.get_location().distance(el)) if lights else None
            ls = f"{nl.get_state()}@{nl.get_location().distance(el):.0f}m" if nl else "n/a"
            gd = float(np.hypot(gx[0] - float(ego_bin["x"][0]), gy[0] - float(ego_bin["y"][0])))
            near = sum(1 for a in bg if a.is_alive and a.get_location().distance(el) < 40.0)
            wp = cmap.get_waypoint(ego.get_location())  # nearest drivable lane center
            lat = ego.get_location().distance(wp.transform.location) if wp else -1.0
            onroad = cmap.get_waypoint(ego.get_location(), project_to_road=False) is not None
            bin_off = float(np.min(np.hypot(_bin_lanes[:, 0] - ebx, _bin_lanes[:, 1] - eby))) if len(_bin_lanes) else -1.0
            print(f"[cosim] step {step}: ego carla=({ex:.1f},{ey:.1f})  carla_off={lat:.1f}m onroad={onroad}  bin_off={bin_off:.1f}m  goal0={gd:.0f}m[{goal_cursor}/{len(route_goals)}] light={ls}  scn={scenario_mgr.active_count()}")

    # cleanup
    if bev is not None:
        bev.save()
    if carla_frames:
        import imageio

        imageio.mimwrite(args.carla_view, carla_frames, fps=10, codec="libx264")
        print(f"[cosim] wrote {args.carla_view} ({len(carla_frames)} frames)")
    if cam is not None and cam.is_alive:
        cam.stop()
        cam.destroy()
    scenario_mgr.cleanup()
    s = world.get_settings(); s.synchronous_mode = False; world.apply_settings(s)
    tm.set_synchronous_mode(False)
    client.apply_batch([carla.command.DestroyActor(a) for a in [ego, *bg] if a is not None and a.is_alive])
    print("[cosim] done")


if __name__ == "__main__":
    main()
