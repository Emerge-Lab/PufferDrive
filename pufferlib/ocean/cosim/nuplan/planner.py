import json
import math
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from pufferlib.ocean.cosim import nuplan_bridge as nb
from pufferlib.ocean.drive.drive import Drive

from pufferlib.ocean.cosim.nuplan_bridge import DEFAULT_ARCH, FAR_AWAY


def clean_policy_state_dict(state_dict):
    """Strip torch.compile / DDP prefixes. Inlined from pufferlib.pufferl to
    avoid importing the training stack (wandb, neptune, the _C kernel, ...)
    inside the evaluation environment."""
    def clean(key):
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        return key

    return {clean(k): v for k, v in state_dict.items()}


def _angle_diff(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _steering_from(yaw_rate: float, speed: float, wheel_base: float) -> float:
    if abs(speed) < 0.1:
        return 0.0
    return float(np.clip(math.atan(yaw_rate * wheel_base / speed), -0.83, 0.83))


class PufferDrivePlanner(AbstractPlanner):
    """nuPlan planner that delegates observation + ego dynamics to PufferDrive."""

    requires_scenario: bool = True

    def __init__(
        self,
        checkpoint_path: str,
        bin_cache_dir: str,
        city_bin_dir: Optional[str] = None,
        scenario: Optional[AbstractScenario] = None,
        device: str = "cpu",
        map_radius: float = 400.0,
        goal_spacing: float = 20.0,
        num_agents: int = 64,
        horizon_seconds: float = 8.0,
        deterministic: bool = True,
        env_overrides: Optional[Dict] = None,
        debug_bev_dir: Optional[str] = None,
    ):
        """
        :param checkpoint_path: PufferDrive policy checkpoint (.pt); config.yaml
            two levels up defines the policy architecture.
        :param bin_cache_dir: fallback cache dir for a city bin's `<bin>.origin.json`
            sidecar when the bin's own directory isn't writable.
        :param city_bin_dir: directory of whole-city map-only bins named
            `nuplan__<map_name>.bin` (the only map source; see module
            docstring). A missing bin is a hard error, no fallback.
        :param scenario: injected by the devkit (requires_scenario).
        :param map_radius: map extraction / light-matching radius [m].
        :param goal_spacing: route-goal spacing fed to the policy [m].
        :param num_agents: PufferDrive agent-pool size (ego + streamed background).
        :param horizon_seconds: returned trajectory horizon (constant-velocity
            extrapolation past the integrated first step) [s].
        :param env_overrides: Drive env kwargs overriding DEFAULT_ARCH.
        :param debug_bev_dir: write a top-down BEV mp4 of the shadow PufferDrive
            env per scenario (roads/traffic from the resolved bin, agents from
            the shadow env) — the nuPlan analog of the CARLA co-sim's
            COSIM_DEBUG_BEV. This is the shadow env's OWN view (what the policy
            actually sees), not nuPlan's ground truth; for that, enable CaRL's
            own carl_visualization_callback in the simulation `callback` list
            instead (see module docstring "How to run").
        """
        self._checkpoint_path = checkpoint_path
        self._bin_cache_dir = Path(bin_cache_dir)
        self._city_bin_dir = Path(city_bin_dir) if city_bin_dir else None
        self._scenario = scenario
        self._device = device
        self._map_radius = float(map_radius)
        self._goal_spacing = float(goal_spacing)
        self._num_agents = int(num_agents)
        self._horizon_seconds = float(horizon_seconds)
        self._deterministic = deterministic
        self._env_overrides = env_overrides or {}
        self._debug_bev_dir = Path(debug_bev_dir) if debug_bev_dir else None
        self._arch: Optional[Dict] = None  # resolved in _build from the checkpoint config

        # lazy state (built on the first compute call, which knows the ego pose)
        self._initialization: Optional[PlannerInitialization] = None
        self._env: Optional[Drive] = None
        self._policy = None
        self._transform: Optional[nb.NuPlanTransform] = None
        self._connector_map: Dict[str, int] = {}
        self._num_traffic = 0
        self._route_goals: Optional[np.ndarray] = None
        self._goal_cursor = 0
        self._bev = None  # BEVRenderer, built in _build if debug_bev_dir is set
        self._last_goal_xy = None  # (gx, gy) selected in the most recent _sync, for BEV capture
        self._last_light_states = None  # traffic-light state array from the most recent _sync

    # --- AbstractPlanner interface -------------------------------------------
    def initialize(self, initialization: PlannerInitialization) -> None:
        self._initialization = initialization

    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks  # type: ignore

    # --- whole-city map-only bins -------------------------------------------
    def _find_city_bin(self) -> Path:
        """City bin for this scenario's map (the map id the devkit read from
        the log db's `map_version`). Assumed to exist -- city_bin_dir covers
        every map, no fallback."""
        map_name = str(self._initialization.map_api.map_name).replace(".gpkg", "")
        return self._city_bin_dir / f"nuplan__{map_name}.bin"

    def _gpkg_layer(self, layer_name: str):
        """Whole-city vector layer (GeoDataFrame, UTM) from the scenario's map."""
        map_api = self._initialization.map_api
        try:
            return map_api._load_vector_map_layer(layer_name)
        except AttributeError:
            return map_api._maps_db.load_vector_layer(map_api.map_name, layer_name)

    def _city_bin_origin(self, bin_path: Path, stop_line_centers: np.ndarray) -> nb.NuPlanTransform:
        """Recover a city bin's frame origin (it stores no centroid): global
        translation vote on stop-line landmarks, then translation-only ICP of
        the bin's lane centerlines onto the GPKG baseline paths. Cached in a
        sidecar json — one registration per city, ever."""
        sidecar = bin_path.with_suffix(".origin.json")
        cache = sidecar if sidecar.exists() else self._bin_cache_dir / sidecar.name
        if cache.exists():
            origin = json.loads(cache.read_text())["origin"]
            return nb.NuPlanTransform(origin[0], origin[1])

        if not len(stop_line_centers):
            raise RuntimeError(f"{bin_path}: no traffic elements to register the origin with")
        import shapely

        stops = self._gpkg_layer("stop_polygons").geometry.centroid
        init = nb.coarse_translation_vote(
            stop_line_centers, np.stack([stops.x.to_numpy(), stops.y.to_numpy()], axis=1)
        )
        lane_utm = shapely.get_coordinates(self._gpkg_layer("baseline_paths").geometry.values)
        t, resid = nb.fit_translation(nb.read_bin_lane_points(bin_path), lane_utm, init=init)
        if resid > 0.5:
            raise RuntimeError(
                f"{bin_path}: lane-geometry registration residual {resid:.2f} m — "
                "bin does not match this city's GPKG map, refusing to guess the origin."
            )
        payload = json.dumps({"origin": [float(t[0]), float(t[1])], "residual_m": resid})
        for target in (sidecar, self._bin_cache_dir / sidecar.name):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(payload)
                break
            except OSError:
                continue
        print(f"[pufferdrive_planner] registered {bin_path.name}: "
              f"origin=({t[0]:.2f}, {t[1]:.2f}) resid={resid * 100:.1f} cm")
        return nb.NuPlanTransform(t[0], t[1])

    def _resolve_map_bin(self):
        """-> (bin_path, NuPlanTransform, stop_line_centers, num_traffic)."""
        city_bin = self._find_city_bin()
        geo = nb.read_bin_geometry(city_bin)
        tf = (nb.NuPlanTransform(*geo["origin"]) if geo["origin"] is not None
              else self._city_bin_origin(city_bin, geo["stop_line_centers"]))
        return city_bin, tf, geo["stop_line_centers"], geo["num_traffic"]

    def _match_traffic_lights(self, stop_line_centers: np.ndarray, ex: float, ey: float) -> Dict[str, int]:
        """Signalized lane connectors within map_radius -> bin traffic elements,
        matched by entry-point proximity (no sidecar file needed)."""
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        objs = self._initialization.map_api.get_proximal_map_objects(
            Point2D(ex, ey), self._map_radius, [SemanticMapLayer.LANE_CONNECTOR]
        )
        entries = {}
        for e in objs[SemanticMapLayer.LANE_CONNECTOR]:
            if e.has_traffic_lights():
                p0 = e.baseline_path.discrete_path[0]
                entries[str(e.id)] = (float(p0.x), float(p0.y))
        return nb.match_connectors_to_stop_lines(entries, self._transform, stop_line_centers)

    def _resolve_arch(self, cfg: Dict, bin_path: Path) -> Dict:
        """Shadow-env Drive kwargs: the checkpoint's training env section
        decides every Drive-accepted key, the clean-eval profile is applied on
        top (see cosim/arch.py), DEFAULT_ARCH fills the no-checkpoint gaps,
        `env_overrides` wins -- except for the structural co-sim keys (map/
        pool wiring), which are not overridable."""
        from pufferlib.ocean.cosim.arch import shadow_env_kwargs

        return shadow_env_kwargs(
            cfg,
            defaults=DEFAULT_ARCH,
            overrides={
                **self._env_overrides,
                "map_dir": str(bin_path), "num_maps": 1, "num_agents": self._num_agents,
                "scenario_length": 1_000_000, "resample_frequency": 0,
                # External sim owns the episode: a training-config
                # termination_mode=1 would c_reset() the pool (ego included)
                # once parked FAR_AWAY slots latch under "stop" behaviors.
                "termination_mode": 0,
                # Enforcement off (flags still fire): in one endless episode a
                # "stop" latch is permanent and would freeze the ego for good.
                "collision_behavior": "ignore", "offroad_behavior": "ignore",
                "traffic_light_behavior": "ignore",
            },
        )

    @property
    def _dummy(self) -> bool:
        """checkpoint_path in ('', 'dummy'): no policy — constant forward-jerk
        actions, the no-checkpoint wiring test (same as the CARLA agent)."""
        return str(self._checkpoint_path or "").lower() in ("", "dummy", "none")

    # --- lazy construction ----------------------------------------------------
    def _build(self, ego_state: EgoState) -> None:
        import torch
        import yaml

        import pufferlib.ocean.torch as drive_torch

        init = self._initialization
        ex, ey = float(ego_state.center.x), float(ego_state.center.y)

        cfg = ({} if self._dummy else
               yaml.safe_load(open(Path(self._checkpoint_path).resolve().parents[1] / "config.yaml")))

        bin_path, self._transform, stop_centers, self._num_traffic = self._resolve_map_bin()
        self._connector_map = self._match_traffic_lights(stop_centers, ex, ey)

        self._arch = self._resolve_arch(cfg, bin_path)
        self._env = Drive(**self._arch)
        self._env.reset()

        if self._dummy:
            self._policy = None
        else:
            policy = getattr(drive_torch, cfg.get("policy_name", "Drive"))(self._env, **cfg["policy"])
            sd = torch.load(self._checkpoint_path, map_location=self._device, weights_only=False)
            policy.load_state_dict(clean_policy_state_dict(sd))
            self._policy = policy.to(self._device).eval()

        # fixed route goals (bin frame): prefer the scenario's logged expert
        # trajectory (the actual driven path, unambiguous about lane/fork
        # choice) over the roadblock lane-graph walk, which only fixes route at
        # the roadblock level and can hop onto a parallel or turning lane at
        # multi-lane blocks / forks. See nb.expert_route_xy docstring.
        centerline = nb.expert_route_xy(self._scenario)
        if len(centerline) < 2:  # no usable logged trajectory: fall back to the lane graph
            centerline = nb.route_centerline(init.map_api, init.route_roadblock_ids, ex, ey)
        goals = nb.goals_along(centerline, self._goal_spacing)
        if len(goals) == 0:  # degenerate route: fall back to the mission goal
            goals = np.array([[init.mission_goal.x, init.mission_goal.y]])
        goals -= (self._transform.ox, self._transform.oy)
        self._route_goals = goals
        self._goal_cursor = 0

        # ego bounding box from nuPlan vehicle parameters (static)
        fp = ego_state.car_footprint
        self._env.set_agent_sizes(np.array([0], np.int32),
                                  np.array([fp.length], np.float32),
                                  np.array([fp.width], np.float32))

        if self._debug_bev_dir:
            from datetime import datetime

            from pufferlib.ocean.cosim.carla_cosim import BEVRenderer

            # scenario.token is a genuine per-scenario unique id (unlike the
            # CARLA leaderboard's route_index, which turned out to just be the
            # routes-file stem) — still add a timestamp as cheap defense
            # against re-running the same scenario into the same output dir.
            token = self._scenario.token if self._scenario else "scenario"
            tag = f"{token}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            self._debug_bev_dir.mkdir(parents=True, exist_ok=True)
            self._bev = BEVRenderer(str(bin_path), str(self._debug_bev_dir / f"{tag}.mp4"))

        print(f"[pufferdrive_planner] bin={bin_path.name} lights={len(self._connector_map)}/"
              f"{self._num_traffic} goals={len(goals)}")

    # --- world sync -----------------------------------------------------------
    def _sync(self, ego_state: EgoState, detections: DetectionsTracks, traffic_light_data) -> None:
        tf = self._transform
        env = self._env

        # ego (slot 0): center pose + global-frame velocity. yaw_rate/accel_long come straight from
        # nuPlan's own EgoState telemetry (already local-frame, rad/s and m/s^2, no reflection needed
        # -- see the module docstring), not finite-differenced: PufferDrive's own integrate() may have
        # already overwritten the ego's previous state by the time this setter would otherwise read it.
        h = float(ego_state.center.heading)
        dcs = ego_state.dynamic_car_state
        v_local = dcs.center_velocity_2d
        vx_g = v_local.x * math.cos(h) - v_local.y * math.sin(h)
        vy_g = v_local.x * math.sin(h) + v_local.y * math.cos(h)
        ex, ey = tf.loc_to_bin(ego_state.center.x, ego_state.center.y)
        env.set_agent_states(np.array([0], np.int32),
                             np.array([ex], np.float32), np.array([ey], np.float32),
                             np.array([0.0], np.float32), np.array([h], np.float32),
                             np.array([vx_g], np.float32), np.array([vy_g], np.float32),
                             np.array([float(dcs.angular_velocity)], np.float32),
                             np.array([float(dcs.center_acceleration_2d.x)], np.float32))

        # background (slots 1..): streamed from nuPlan, never simulated here. nuPlan's TrackedObject
        # carries no acceleration/angular-velocity (perception detections, not ego telemetry), but
        # write_partner_obs never reads those fields for non-ego agents, so zero is exact, not a stopgap.
        objs = list(detections.tracked_objects)[: self._num_agents - 1]
        if objs:
            idx, x, y, z, hh, vx, vy, _tp, ln, wd = nb.tracked_objects_to_arrays(objs, tf)
            env.set_agent_states(idx, x, y, z, hh, vx, vy, np.zeros_like(vx), np.zeros_like(vx))
            env.set_agent_sizes(idx, ln, wd)
        surplus = np.arange(1 + len(objs), self._num_agents, dtype=np.int32)
        if len(surplus):
            far = np.full(len(surplus), FAR_AWAY, np.float32)
            zero = np.zeros_like(far)
            env.set_agent_states(surplus, far, far, far, zero, zero, zero, zero, zero)

        # traffic lights: state array sized to the bin's element count
        if self._num_traffic:
            self._last_light_states = nb.traffic_light_states(
                traffic_light_data, self._connector_map, self._num_traffic
            )
            env.set_traffic_light_states(self._last_light_states)

        # route goals: cursor advances only on arrival (like the CARLA side)
        goals = self._route_goals
        while (self._goal_cursor < len(goals) - 1
               and np.hypot(goals[self._goal_cursor, 0] - ex, goals[self._goal_cursor, 1] - ey)
               < self._arch["goal_radius"]):
            self._goal_cursor += 1
        k = self._arch["num_goals"]
        indices = [min(self._goal_cursor + i, len(goals) - 1) for i in range(k)]
        sel = goals[indices]
        self._last_goal_xy = (sel[:, 0].copy(), sel[:, 1].copy())
        # Local route direction per goal (previous goal -> goal; ~goal_spacing
        # apart, adequate at nuPlan's 20 m spacing) for route-aligned lane snapping.
        dir_sel = np.array([goals[i] - goals[max(i - 1, 0)] for i in indices], np.float32)
        env.set_agent_goals(0, sel[:, 0].astype(np.float32).copy(),
                            sel[:, 1].astype(np.float32).copy(),
                            np.zeros(k, np.float32),
                            dir_sel[:, 0].copy(), dir_sel[:, 1].copy())

    # --- planning -------------------------------------------------------------
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        import torch

        import pufferlib.pytorch

        ego_state, detections = current_input.history.current_state
        if self._env is None:
            self._build(ego_state)

        self._sync(ego_state, detections, current_input.traffic_light_data)
        obs = np.asarray(self._env.recompute_observations())

        if self._policy is None:  # dummy wiring test: constant forward jerk
            act = np.full((self._env.num_agents, 1), 3 * 3 + 1, dtype=np.int32)
        else:
            with torch.no_grad():
                logits, _ = self._policy.forward_eval(torch.as_tensor(obs).to(self._device))
                action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=self._deterministic)
            act = action.cpu().numpy().reshape(self._env.num_agents, -1).astype(np.int32)

        self._env.step(act)  # integrates the ego one dt; background is re-synced next call

        if self._bev is not None:
            try:
                self._bev.capture(self._env.get_global_agent_state(), ego_idx=0,
                                  goals=self._last_goal_xy, light_states=self._last_light_states)
                # AbstractPlanner has no close()/destroy() hook, so save at the
                # scenario's last planning iteration instead (BEVRenderer
                # buffers frames in memory until .save(), like the CARLA-side
                # BEV). simulations_runner.py's loop is `while
                # is_simulation_running()`, checked BEFORE each call, and
                # reached_end() (-> not running) fires once the iteration
                # index hits get_number_of_iterations()-1 -- so the LAST index
                # the planner is ever actually invoked with is
                # get_number_of_iterations()-2, not -1 (confirmed empirically:
                # -1 never saved anything, debug_bev_dir stayed empty).
                last_iter = self._scenario.get_number_of_iterations() - 2
                if current_input.iteration.index >= last_iter:
                    self._bev.save()
            except Exception as e:
                print(f"[pufferdrive_planner] debug_bev_dir capture/save failed (non-fatal): {e}")
                self._bev = None

        # integrated ego -> nuPlan EgoState at t + dt
        dt = float(self._arch["dt"])
        state = self._env.get_global_agent_state()
        nx_, ny_ = self._transform.bin_to_loc(float(state["x"][0]), float(state["y"][0]))
        nh = float(state["heading"][0])

        # speed = the shadow ego's POST-STEP velocity state (ego obs column 0),
        # not displacement/dt, which halves the intent during acceleration and
        # makes the closed loop crawl (see WorldSync.integrate).
        from pufferlib.ocean.drive import binding as drive_binding

        speed = float(np.asarray(self._env.observations)[0, 0]) * drive_binding.MAX_SPEED
        v0 = ego_state.dynamic_car_state.center_velocity_2d.x
        yaw_rate = _angle_diff(nh, float(ego_state.center.heading)) / dt

        params = ego_state.car_footprint.vehicle_parameters
        t0 = ego_state.time_point

        def _state_at(i: int) -> EgoState:
            """i-th 10 Hz pose: i=0 current, i=1 the PufferDrive-integrated step,
            then constant velocity + yaw-rate extrapolation for the horizon."""
            if i == 0:
                return ego_state
            t = (i - 1) * dt
            heading = nh + yaw_rate * t
            x = nx_ + speed * math.cos(heading) * t
            y = ny_ + speed * math.sin(heading) * t
            return EgoState.build_from_center(
                center=StateSE2(x, y, heading),
                center_velocity_2d=StateVector2D(speed, 0.0),
                center_acceleration_2d=StateVector2D((speed - v0) / dt if i == 1 else 0.0, 0.0),
                tire_steering_angle=_steering_from(yaw_rate, speed, params.wheel_base),
                time_point=TimePoint(t0.time_us + int(i * dt * 1e6)),
                vehicle_parameters=params,
                angular_vel=yaw_rate,
            )

        n_poses = int(round(self._horizon_seconds / dt))
        return InterpolatedTrajectory([_state_at(i) for i in range(n_poses + 1)])
