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

DEFAULT_ROUTES = "/workspace/CaRL/PlanT/data/longest6.xml"
FAR_AWAY = 1.0e6  # park surplus PufferDrive agents out of observation range

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
    ego = world.spawn_actor(car_bps[0], ego_sp)
    ego.set_simulate_physics(False)

    # Background: random spawn points (excluding the ego's), TM autopilot.
    bg = []
    used = {ego_sp}
    for sp in spawn_points:
        if len(bg) >= num_background:
            break
        if sp in used or sp.location.distance(ego_sp.location) < 8.0:
            continue
        actor = world.try_spawn_actor(car_bps[len(bg) % len(car_bps)], sp)
        if actor is not None:
            actor.set_autopilot(True, tm.get_port())
            bg.append(actor)
            used.add(sp)

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


def ego_goals(route_wps, ego_xy_carla, transform):
    """3 route waypoints ahead of the ego (bin frame), ~[15,35,60] m spacing."""
    d = np.hypot(route_wps[:, 0] - ego_xy_carla[0], route_wps[:, 1] - ego_xy_carla[1])
    i0 = int(d.argmin())
    targets = []
    for want in (15.0, 35.0, 60.0):
        j = i0
        acc = 0.0
        while j + 1 < len(route_wps) and acc < want:
            acc += np.hypot(*(route_wps[j + 1, :2] - route_wps[j, :2]))
            j += 1
        targets.append(route_wps[min(j, len(route_wps) - 1)])
    gx, gy, gz = [], [], []
    for w in targets:
        bx, by = transform.loc_to_bin(w[0], w[1])
        gx.append(bx); gy.append(by); gz.append(w[2])
    return np.array(gx, np.float32), np.array(gy, np.float32), np.array(gz, np.float32)


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
    args = ap.parse_args()

    town, route_wps = parse_route(args.routes, args.route_id)
    town_bin = cb.bin_path_for_town(town)
    print(f"[cosim] route {args.route_id} town={town} waypoints={len(route_wps)} bin={town_bin}")

    cfg = load_checkpoint_config(args.checkpoint)  # for the policy arch (below)
    dt = CARLA_ARCH["dt"]
    sub_ticks = args.sub_ticks or max(1, round(dt / 0.1))  # CARLA at 0.1s, PufferDrive at dt

    # PufferDrive env (gigaflow spawns an agent pool; agent 0 = ego, rest = background).
    env = Drive(
        map_dir=town_bin, num_maps=1, num_agents=args.num_agents,
        simulation_mode="gigaflow", control_mode="control_vehicles",
        scenario_length=1_000_000, resample_frequency=0, **CARLA_ARCH,
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
    transform = cb.CarlaTransform(town, offset=cb.compute_town_offset(world.get_map(), town_bin))
    light_map, num_traffic = map_lights_to_bin(lights, transform, town_bin, len(lights))
    print(f"[cosim] carla: ego + {len(bg)} background + {len(lights)} lights; offset={transform.tx:.1f},{transform.ty:.1f}")

    # --- init sync: ego pose + goals; background; park surplus agents ---
    ego_state = transform.actor_state_to_bin(ego)
    env.set_agent_states(np.array([0], np.int32), *[np.array([v], np.float32) for v in ego_state])
    gx, gy, gz = ego_goals(route_wps, (ego.get_transform().location.x, ego.get_transform().location.y), transform)
    env.set_agent_goals(0, gx, gy, gz)
    bg_idx, *_ = (read_background(bg, transform))
    surplus = np.arange(1 + len(bg), n_active, dtype=np.int32)
    if len(surplus):
        z6 = np.full(len(surplus), FAR_AWAY, np.float32)
        env.set_agent_states(surplus, z6, z6, z6, np.zeros_like(z6), np.zeros_like(z6), np.zeros_like(z6))
    obs = np.asarray(env.recompute_observations())

    # --- co-sim loop ---
    import data_utils.mirror_map_bin as mbin  # noqa: F401 (kept warm)
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
        for _ in range(sub_ticks):
            world.tick()
        # overwrite background + lights from CARLA
        bg_state = read_background(bg, transform)
        env.set_agent_states(*bg_state)
        states = np.zeros(num_traffic, np.int32)
        for li, lt in enumerate(lights):
            j = light_map[li]
            if 0 <= j < num_traffic:
                states[j] = cb.carla_light_to_puffer(lt.get_state())
        env.set_traffic_light_states(states)
        # advance ego goals along the route
        gx, gy, gz = ego_goals(route_wps, (ex, ey), transform)
        env.set_agent_goals(0, gx, gy, gz)
        obs = np.asarray(env.recompute_observations())

        if step % 10 == 0:
            spd = float(np.hypot(ego.get_velocity().x, ego.get_velocity().y)) if False else float(ego_bin.get("heading")[0])
            print(f"[cosim] step {step}: ego carla=({ex:.1f},{ey:.1f}) heading={eyaw:.0f}deg  bg={len(bg)}")

    # cleanup
    s = world.get_settings(); s.synchronous_mode = False; world.apply_settings(s)
    tm.set_synchronous_mode(False)
    client.apply_batch([carla.command.DestroyActor(a) for a in [ego, *bg] if a is not None and a.is_alive])
    print("[cosim] done")


if __name__ == "__main__":
    main()
