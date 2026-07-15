"""Read-only CARLA world -> shadow PufferDrive env sync for leaderboard evaluation.

This module reads actor transforms/velocities/bounding boxes and traffic-light states

The shadow Drive env is a private observation encoder + kinematic integrator
"""

from pathlib import Path

import numpy as np

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import carla_bridge as cb

FAR_AWAY = 1.0e6  # park unused PufferDrive agent slots out of observation range
GOAL_RADIUS_M = 6.0  # advance the route-goal cursor only when the ego arrives within this
GOAL_SPACING_M = 20.0  # one route goal every N meters of route arc length

# No-checkpoint (dummy) wiring test: Drive's own defaults, except jerk dynamics
# so the constant DUMMY_FORWARD_JERK action indexes a forward jerk.
DUMMY_ARCH = dict(dynamics_model="jerk")


def resolve_arch(cfg, dt=None, env_overrides=None):
    """Shadow-env Drive kwargs. The checkpoint's training env section decides
    every observation-layout / dynamics key (encoding parity with how the
    policy was trained); train-time observation dropout is disabled at eval.
    Without a checkpoint config (dummy wiring test) Drive's own defaults apply
    (DUMMY_ARCH). `env_overrides` wins over adopted keys; an explicit `dt`
    wins over everything. Mirrors cosim/nuplan/planner._resolve_arch."""
    import inspect

    accepted = set(inspect.signature(Drive.__init__).parameters)
    adopted = {
        k: v for k, v in (cfg or {}).get("env", {}).items()
        if k in accepted and not k.startswith("obs_dropout")
        and (k.startswith("obs_") or k in (
            "num_goals", "dynamics_model", "target_type",
            "reward_conditioning", "goal_radius", "goal_speed", "dt"))
    }
    # Train-time route-goal modes this checkout lacks (e.g. "dijkstra") don't
    # matter here: the co-sim feeds explicit route goals every step.
    if adopted.get("target_type") not in ("static", "dynamic"):
        adopted.pop("target_type", None)
    arch = {**(DUMMY_ARCH if cfg is None else {}), **adopted, **(env_overrides or {})}
    if dt is not None:
        arch["dt"] = dt
    return arch


def route_goals_from_plan(dense_route, transform, spacing=GOAL_SPACING_M): # TODO: hack
    """Fixed (N, 3) sequence of route goals in the bin frame, one every `spacing`
    meters of arc length, from the leaderboard's dense global plan
    ([(carla.Transform, RoadOption)], ~1 m spacing, already lane-centered).
    Always ends on the route's final point."""
    pts = np.array(
        [[t.location.x, t.location.y, t.location.z] for t, _ in dense_route], dtype=np.float64
    )
    goals, next_at, cum = [], spacing, 0.0
    for i in range(1, len(pts)):
        cum += float(np.hypot(pts[i, 0] - pts[i - 1, 0], pts[i, 1] - pts[i - 1, 1]))
        if cum >= next_at:
            bx, by = transform.loc_to_bin(pts[i, 0], pts[i, 1])
            goals.append((bx, by, pts[i, 2]))
            next_at += spacing
    bx, by = transform.loc_to_bin(pts[-1, 0], pts[-1, 1])
    goals.append((bx, by, pts[-1, 2]))
    return np.array(goals, np.float32)


def map_lights_to_bin(lights, transform, town_bin):
    """mapping[i] = bin traffic-element index for lights[i] (nearest stop line in
    the bin frame). Returns (mapping, num_traffic_elements). The state array
    passed to set_traffic_light_states is sized to the bin's element count."""
    import data_utils.mirror_map_bin as mbin

    data = mbin.read_bin(Path(town_bin))
    tl_pos = []
    for t in data["traffic"]:
        sx = 0.5 * (t["stop_line"][0] + t["stop_line"][3])
        sy = 0.5 * (t["stop_line"][1] + t["stop_line"][4])
        tl_pos.append((sx, sy))
    tl_pos = np.array(tl_pos) if tl_pos else np.zeros((0, 2))
    mapping = []
    for lt in lights:
        loc = lt.get_transform().location
        bx, by = transform.loc_to_bin(loc.x, loc.y)
        if len(tl_pos):
            j = int(((tl_pos[:, 0] - bx) ** 2 + (tl_pos[:, 1] - by) ** 2).argmin())
        else:
            j = -1
        mapping.append(j)
    return mapping, len(tl_pos)


class WorldSync:
    """One route's CARLA -> shadow-env bridge. Construct once per route (the
    leaderboard builds a fresh agent per route), call sync() before every policy
    step and integrate() after it."""

    def __init__(self, world, ego, town, dense_route, num_agents=64, dt=0.1, town_bin=None,
                 cfg=None, env_overrides=None, offset=None):
        self.world = world
        self.ego = ego
        self.dt = dt
        town_bin = town_bin or cb.bin_path_for_town(town)
        self.town_bin = town_bin

        arch = resolve_arch(cfg, dt=dt, env_overrides=env_overrides)
        self.env = Drive(
            map_dir=town_bin, num_maps=1, num_agents=num_agents,
            simulation_mode="gigaflow", control_mode="control_vehicles",
            scenario_length=1_000_000, resample_frequency=0,
            # Reset-time goals are throwaway (we set the ego's from the route and
            # overwrite all other agents every sync). goal_on_lane=False keeps
            # reset fast and independent of the routable lane network.
            goal_on_lane=False,
            **arch,
        )
        self.env.reset()
        self.num_agents = int(self.env.num_agents)
        self.num_goals = int(self.env.num_goals)

        if offset is None:
            offset = cb.town_offset(world.get_map(), town_bin)
        self.transform = cb.CarlaTransform(town, offset=offset)
        self.route_goals = route_goals_from_plan(dense_route, self.transform)
        self.goal_cursor = 0

        self.lights = list(world.get_actors().filter("traffic.traffic_light"))
        self.light_map, self.num_traffic = map_lights_to_bin(self.lights, self.transform, town_bin)
        self.last_light_states = np.zeros(self.num_traffic, np.int32)

    # --- CARLA readers (all read-only) -----------------------------------

    def _nearby_actors(self):
        """Non-ego vehicles + walkers nearest the ego, at most num_agents - 1.
        Enumerated fresh each call: the leaderboard spawns/destroys scenario
        actors dynamically, so a fixed list would go stale."""
        ego_loc = self.ego.get_location()
        actors = [
            a for a in self.world.get_actors()
            if a.id != self.ego.id
            and ("vehicle" in a.type_id or "walker.pedestrian" in a.type_id)
        ]
        actors.sort(key=lambda a: a.get_location().distance(ego_loc))
        return actors[: self.num_agents - 1]

    def _read_states(self, actors):
        idx, x, y, z, h, vx, vy = [], [], [], [], [], [], []
        for j, a in enumerate(actors):
            idx.append(1 + j)  # agent 0 = ego; others fill 1..M
            bx, by, bz, bh, bvx, bvy = self.transform.actor_state_to_bin(a)
            x.append(bx); y.append(by); z.append(bz); h.append(bh); vx.append(bvx); vy.append(bvy)
        return (np.array(idx, np.int32), np.array(x, np.float32), np.array(y, np.float32),
                np.array(z, np.float32), np.array(h, np.float32),
                np.array(vx, np.float32), np.array(vy, np.float32))

    def _read_sizes(self, actors):
        """Bounding-box (length, width) in meters so the ego observes a truck as
        a truck. CARLA extent is half-size along the actor's local axes."""
        idx, length, width = [], [], []
        for j, a in enumerate(actors):
            idx.append(1 + j)
            ext = a.bounding_box.extent
            length.append(max(2.0 * ext.x, 0.1))
            width.append(max(2.0 * ext.y, 0.1))
        return (np.array(idx, np.int32), np.array(length, np.float32), np.array(width, np.float32))

    def _read_light_states(self):
        states = np.zeros(self.num_traffic, np.int32)
        for li, lt in enumerate(self.lights):
            j = self.light_map[li]
            if 0 <= j < self.num_traffic:
                states[j] = cb.carla_light_to_puffer(lt.get_state())
        return states

    # --- sync + integrate --------------------------------------------------

    def sync(self):
        """Overwrite the whole shadow env from CARLA ground truth and return the
        recomputed observation array (num_agents, obs_dim). Row 0 is the ego."""
        # Ego (slot 0): always ground truth from CARLA — integration error from
        # the previous integrate() is discarded here every policy step.
        ego_state = self.transform.actor_state_to_bin(self.ego)
        self.env.set_agent_states(
            np.array([0], np.int32), *[np.array([v], np.float32) for v in ego_state]
        )
        ego_ext = self.ego.bounding_box.extent
        self.env.set_agent_sizes(
            np.array([0], np.int32),
            np.array([2.0 * ego_ext.x], np.float32), np.array([2.0 * ego_ext.y], np.float32),
        )

        # Background (slots 1..M): nearest vehicles/walkers; park the rest.
        actors = self._nearby_actors()
        if actors:
            self.env.set_agent_states(*self._read_states(actors))
            self.env.set_agent_sizes(*self._read_sizes(actors))
        n_used = 1 + len(actors)
        if n_used < self.num_agents:
            sp = np.arange(n_used, self.num_agents, dtype=np.int32)
            zf = np.full(len(sp), FAR_AWAY, np.float32)
            zz = np.zeros_like(zf)
            self.env.set_agent_states(sp, zf, zf, zf, zz, zz, zz)

        self.last_light_states = self._read_light_states()
        self.env.set_traffic_light_states(self.last_light_states)

        # Route goals: advance the cursor only once the ego actually reaches the
        # current goal, then feed the next few goals.
        ebx, eby = ego_state[0], ego_state[1]
        while (self.goal_cursor < len(self.route_goals) - 1
               and np.hypot(self.route_goals[self.goal_cursor, 0] - ebx,
                            self.route_goals[self.goal_cursor, 1] - eby) < GOAL_RADIUS_M):
            self.goal_cursor += 1
        sel = self.route_goals[
            [min(self.goal_cursor + k, len(self.route_goals) - 1)
             for k in range(self.num_goals)]
        ]
        self.env.set_agent_goals(0, sel[:, 0].copy(), sel[:, 1].copy(), sel[:, 2].copy())

        return np.asarray(self.env.recompute_observations())

    def integrate(self, actions):
        """Step the shadow env with the policy actions and return the ego's
        target kinematic state for the tracking controller:
        (target_speed m/s, target_yaw_deg CARLA frame). Speed comes from the
        integrated displacement over dt (the env exposes pose, not velocity)."""
        before = self.env.get_global_agent_state()
        bx0, by0 = float(before["x"][0]), float(before["y"][0])
        self.env.step(np.asarray(actions, dtype=np.int32))
        after = self.env.get_global_agent_state()
        bx1, by1 = float(after["x"][0]), float(after["y"][0])

        target_speed = float(np.hypot(bx1 - bx0, by1 - by0)) / self.dt
        target_yaw_deg = self.transform.bin_heading_to_yaw(float(after["heading"][0]))
        return target_speed, target_yaw_deg

    def ego_bin_state(self):
        """Shadow-env ego pose (for debug/BEV rendering)."""
        return self.env.get_global_agent_state()

    def route_progress(self):
        return self.goal_cursor, len(self.route_goals)

    def close(self):
        self.env.close()
