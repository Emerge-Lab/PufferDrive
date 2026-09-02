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
from pufferlib.ocean.cosim.arch import checkpoint_config_path
from pufferlib.ocean.cosim.goals import RouteGoalWindow, route_goals_from_xy
from pufferlib.ocean.drive.drive import Drive

from pufferlib.ocean.cosim.nuplan_bridge import DEFAULT_ARCH, FAR_AWAY
from pufferlib.ocean.drive import binding

EGO_OBS_ACCEL_LONG_IDX = 4  # write_ego_obs column order: speed, width, length, steering, accel_long, ...
EGO_TRACKING_TOL_M = 0.05  # perfect_tracking_controller hands back our integrated pose up to log-timestamp jitter
EGO_TRACKING_TOL_RAD = 0.005
SCENARIO_LENGTH_MARGIN_STEPS = 2  # shadow env must never hit its own truncation while nuPlan still plans
NUPLAN_COMFORT_MAX_LON_ACCEL_MPS2 = 2.40  # ego_lon_acceleration_statistics.yaml max_lon_accel
TRAJECTORY_TAIL_SPACING_S = 0.5  # constant-velocity tail past the integrated step; the controller only reads t+dt
COMFORT_ACCEL_MARGIN = 0.96  # nuPlan Savitzky-Golay-smooths the reported accel and overshoots a hard cap by ~0.5%


def clean_policy_state_dict(state_dict):
    """Strip torch.compile / DDP prefixes. Inlined from pufferlib.pufferl to
    avoid importing the training stack (wandb, neptune, the _C kernel, ...)
    inside the evaluation environment."""

    def clean(key):
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        return key

    return {clean(k): v for k, v in state_dict.items()}


# per-process caches: one planner per scenario, same checkpoint/bins each time
_state_dict_cache: Dict[str, dict] = {}
_bin_geometry_cache: Dict[str, dict] = {}


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
        goal_source: str = "route",
        num_agents: int = 64,
        horizon_seconds: float = 8.0,
        deterministic: bool = True,
        env_overrides: Optional[Dict] = None,
        obs_html_dir: Optional[str] = None,
        obs_html_max_steps: int = 800,
        obs_html_render: bool = True,
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
        :param goal_spacing: goal spacing along the route / logged path fed to the policy [m].
        :param goal_source: "route" (lane-graph route through the route roadblocks) or "gt_map"
            (logged ego path, every sample snapped to the nearest co-directional lane center).
        :param num_agents: PufferDrive agent-pool size (ego + streamed background).
        :param horizon_seconds: returned trajectory horizon (constant-velocity
            extrapolation past the integrated first step) [s].
        :param env_overrides: Drive env kwargs overriding DEFAULT_ARCH.
        :param obs_html_dir: write the interactive pufferlib.viz observation replay per
            scenario (the exact obs the policy received, its outputs and encoder pool winners).
        :param obs_html_render: render the html right away; False saves only the compact
            .replay.zlib so pages can be rendered later for selected scenarios
            (scripts/eval/render_obs_html.py, e.g. only those scoring below a threshold).
        """
        self._checkpoint_path = checkpoint_path
        self._bin_cache_dir = Path(bin_cache_dir)
        self._city_bin_dir = Path(city_bin_dir) if city_bin_dir else None
        self._scenario = scenario
        self._device = device
        self._map_radius = float(map_radius)
        self._goal_spacing = float(goal_spacing)
        if goal_source not in ("route", "gt_map"):
            raise ValueError(f"goal_source must be 'route' or 'gt_map', got {goal_source!r}")
        self._goal_source = goal_source
        self._num_agents = int(num_agents)
        self._horizon_seconds = float(horizon_seconds)
        self._deterministic = deterministic
        self._env_overrides = env_overrides or {}
        self._obs_html_dir = Path(obs_html_dir) if obs_html_dir else None
        self._obs_html_max_steps = int(obs_html_max_steps)
        self._obs_html_render = bool(obs_html_render)
        self._obs_replay = None  # ObsReplayCapture, built in _build if obs_html_dir is set
        self._arch: Optional[Dict] = None  # resolved in _build from the checkpoint config

        # lazy state (built on the first compute call, which knows the ego pose)
        self._initialization: Optional[PlannerInitialization] = None
        self._env: Optional[Drive] = None
        self._policy = None
        self._transform: Optional[nb.NuPlanTransform] = None
        self._connector_map: Dict[str, int] = {}
        self._route_connector_ids = ()  # lane ids along the Dijkstra route: they decide shared stop-line elements
        self._num_traffic = 0
        self._goal_window: Optional[RouteGoalWindow] = None
        self._last_integrated = None  # (bin_x, bin_y, heading) the shadow ego was left at by the last step
        self._last_light_states = None  # traffic-light state array from the most recent _sync

    # --- AbstractPlanner interface -------------------------------------------
    def initialize(self, initialization: PlannerInitialization) -> None:
        self._initialization = initialization

    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks  # type: ignore

    def generate_planner_report(self, clear_stats: bool = True):
        # runners stay referenced until the whole process exits; free per-scenario state
        report = super().generate_planner_report(clear_stats)
        if self._env is not None:
            self._env.close()
        self._env = None
        self._policy = None
        self._obs_replay = None
        return report

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
        init = nb.coarse_translation_vote(stop_line_centers, np.stack([stops.x.to_numpy(), stops.y.to_numpy()], axis=1))
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
        print(
            f"[pufferdrive_planner] registered {bin_path.name}: "
            f"origin=({t[0]:.2f}, {t[1]:.2f}) resid={resid * 100:.1f} cm"
        )
        return nb.NuPlanTransform(t[0], t[1])

    def _resolve_map_bin(self):
        """-> (bin_path, NuPlanTransform, stop_line_centers, num_traffic)."""
        city_bin = self._find_city_bin()
        geo = _bin_geometry_cache.get(str(city_bin))
        if geo is None:
            geo = nb.read_bin_geometry(city_bin)
            _bin_geometry_cache[str(city_bin)] = geo
        tf = (
            nb.NuPlanTransform(*geo["origin"])
            if geo["origin"] is not None
            else self._city_bin_origin(city_bin, geo["stop_line_centers"])
        )
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

        if self._scenario is None:
            raise RuntimeError("PufferDrivePlanner needs the scenario injected (requires_scenario)")
        return shadow_env_kwargs(
            cfg,
            defaults=DEFAULT_ARCH,
            overrides={
                **self._env_overrides,
                "map_dir": str(bin_path),
                "num_maps": 1,
                # One policy agent (the ego); every other slot is a static partner streamed from nuPlan.
                "num_agents": 1,
                "min_agents_per_env": 1,
                "cosim_partner_slots": self._num_agents - 1,
                "goal_source": "external",
                # Mirrors the nuplan_* benchmarks in config/evaluation/benchmark.yaml: a nuPlan route
                # never ends inside the scenario, so no window-final goal is speed gated.
                "goal_reach_requires_speed": False,
                # C_acc conditioning: cap the jerk model's positive accel at nuPlan's comfort bound instead of 2.5
                "conditioning_accel_scale": COMFORT_ACCEL_MARGIN * NUPLAN_COMFORT_MAX_LON_ACCEL_MPS2 / binding.ACCEL_LONG_MAX,
                # lockstep with nuPlan's planning iterations, not the training dt (mimolette: 0.3)
                "dt": float(self._scenario.database_interval),
                "scenario_length": int(self._scenario.get_number_of_iterations()) + SCENARIO_LENGTH_MARGIN_STEPS,
                "resample_frequency": 0,
                # External sim owns the episode: a training-config
                # termination_mode=1 would c_reset() the pool (ego included)
                # once parked FAR_AWAY slots latch under "stop" behaviors.
                "termination_mode": 0,
                # native-eval scenario batching builds zero envs on a map-only bin
                "eval_mode": False,
                # Enforcement off (flags still fire): a "stop" latch would freeze the ego for good.
                "collision_behavior": "ignore",
                "offroad_behavior": "ignore",
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

        cfg = (
            {}
            if self._dummy
            else yaml.safe_load(open(checkpoint_config_path(self._checkpoint_path)))
        )

        bin_path, self._transform, stop_centers, self._num_traffic = self._resolve_map_bin()
        self._connector_map = self._match_traffic_lights(stop_centers, ex, ey)

        self._arch = self._resolve_arch(cfg, bin_path)
        self._env = Drive(**self._arch)
        self._env.reset()
        torch.set_num_threads(1)  # one scenario per worker process; BLAS threads only fight the other workers

        # Round-trip probe: slot 0 must be the ego we set. It is silently wrong when
        # gigaflow spawning failed (e.g. a city bin without lane connectivity).
        probe_x, probe_y = self._transform.loc_to_bin(ex, ey)
        zero = np.zeros(1, np.float32)
        self._env.set_agent_states(
            np.array([0], np.int32),
            np.array([probe_x], np.float32),
            np.array([probe_y], np.float32),
            zero,
            np.array([float(ego_state.center.heading)], np.float32),
            zero,
            zero,
            zero,
            zero,
        )
        readback = self._env.get_global_agent_state()
        if abs(float(readback["x"][0]) - probe_x) > 0.5 or abs(float(readback["y"][0]) - probe_y) > 0.5:
            raise RuntimeError(
                f"{bin_path.name}: shadow env agent slot 0 is not the ego "
                f"(set ({probe_x:.1f}, {probe_y:.1f}), read ({readback['x'][0]:.1f}, {readback['y'][0]:.1f})). "
                "Gigaflow spawning likely failed on this bin -- check for '[GIGAFLOW WARNING]'/'[ERROR]' in the "
                "worker stdout; a map-only city bin without lane connectivity (exit_lanes) is the known cause."
            )

        if self._dummy:
            self._policy = None
        else:
            # pre-3.0 checkpoints keep action_type only in the env section
            cfg["policy"].setdefault("action_type", cfg["env"]["action_type"])
            policy = getattr(drive_torch, cfg.get("policy_name", "Drive"))(self._env, **cfg["policy"])
            sd = _state_dict_cache.get(str(self._checkpoint_path))
            if sd is None:
                sd = clean_policy_state_dict(
                    torch.load(self._checkpoint_path, map_location=self._device, weights_only=False)
                )
                _state_dict_cache[str(self._checkpoint_path)] = sd
            policy.load_state_dict(sd)
            self._policy = policy.to(self._device).eval()

        # The lane-graph route through the CaRL-corrected route roadblocks always decides the shared
        # stop-line elements; it also seeds the goals unless goal_source is gt_map (logged ego path,
        # lane-snapped so the expert's raw positions never leak).
        from carl_nuplan.planning.simulation.planner.pdm_planner.utils.route_utils import (
            route_roadblock_correction_v2,
        )

        route_ids = route_roadblock_correction_v2(ego_state, init.map_api, list(init.route_roadblock_ids))
        centerline, self._route_connector_ids = nb.route_centerline(
            init.map_api, route_ids, ex, ey, float(ego_state.center.heading)
        )
        snapped_count = 0
        if self._goal_source == "gt_map":
            goals, goal_headings, snapped_count = nb.logged_ego_goals(self._scenario, init.map_api, self._goal_spacing)
            goals -= (self._transform.ox, self._transform.oy)
            route_goals = np.column_stack(
                [goals, np.zeros(len(goals)), np.cos(goal_headings), np.sin(goal_headings)]
            ).astype(np.float32)
        else:
            goals = nb.goals_along(centerline, self._goal_spacing)
            if len(goals) == 0:  # degenerate route: fall back to the mission goal
                goals = np.array([[init.mission_goal.x, init.mission_goal.y]])
            goals -= (self._transform.ox, self._transform.oy)
            route_goals = route_goals_from_xy(goals)
        self._goal_window = RouteGoalWindow(self._env, route_goals)

        # ego bounding box from nuPlan vehicle parameters (static)
        fp = ego_state.car_footprint
        self._env.set_agent_sizes(
            np.array([0], np.int32), np.array([fp.length], np.float32), np.array([fp.width], np.float32)
        )

        if self._obs_html_dir and self._policy is not None:
            from pufferlib.ocean.cosim.obs_replay import ObsReplayCapture

            token = self._scenario.token if self._scenario else "scenario"
            self._obs_replay = ObsReplayCapture(
                self._env, self._policy, self._obs_html_dir / token, max_steps=self._obs_html_max_steps
            )

        print(
            f"[pufferdrive_planner] bin={bin_path.name} lights={len(self._connector_map)}/"
            f"{self._num_traffic} goals={len(goals)} goal_source={self._goal_source} snapped={snapped_count}"
        )

    # --- world sync -----------------------------------------------------------
    def _sync(self, ego_state: EgoState, detections: DetectionsTracks, traffic_light_data) -> None:
        tf = self._transform
        env = self._env

        # ego (slot 0): synced from nuPlan telemetry once, at the scenario's initial state. Afterwards the
        # shadow env owns the ego: nuPlan's perfect_tracking_controller hands back exactly the pose we
        # integrated, and re-deriving accel_long/steering from telemetry every tick re-injects
        # finite-difference accel after every stop and drives the ego into reverse.
        h = float(ego_state.center.heading)
        dcs = ego_state.dynamic_car_state
        v_local = dcs.center_velocity_2d
        vx_g = v_local.x * math.cos(h) - v_local.y * math.sin(h)
        vy_g = v_local.x * math.sin(h) + v_local.y * math.cos(h)
        ex, ey = tf.loc_to_bin(ego_state.center.x, ego_state.center.y)
        if self._last_integrated is not None:
            self._check_ego_tracking(ex, ey, h)
        else:
            env.set_agent_states(
                np.array([0], np.int32),
                np.array([ex], np.float32),
                np.array([ey], np.float32),
                np.array([0.0], np.float32),
                np.array([h], np.float32),
                np.array([vx_g], np.float32),
                np.array([vy_g], np.float32),
                np.array([float(dcs.angular_velocity)], np.float32),
                np.array([float(dcs.center_acceleration_2d.x)], np.float32),
            )

        # background (slots 1..): streamed from nuPlan, never simulated here. nuPlan's TrackedObject
        # carries no acceleration/angular-velocity (perception detections, not ego telemetry), but
        # write_partner_obs never reads those fields for non-ego agents, so zero is exact, not a stopgap.
        objs = list(detections.tracked_objects)
        if len(objs) > self._num_agents - 1:
            # keep the nearest: devkit ordering is by type/token, so blind
            # truncation could drop a close vehicle while keeping far ones
            ox, oy = ego_state.center.x, ego_state.center.y
            objs.sort(key=lambda o: (o.center.x - ox) ** 2 + (o.center.y - oy) ** 2)
            objs = objs[: self._num_agents - 1]
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
                traffic_light_data, self._connector_map, self._num_traffic, self._route_connector_ids
            )
            env.set_traffic_light_states(self._last_light_states)

        self._goal_window.sync(ex, ey, h)

    def _check_ego_tracking(self, bin_x: float, bin_y: float, heading: float) -> None:
        lx, ly, lh = self._last_integrated
        if (
            abs(bin_x - lx) > EGO_TRACKING_TOL_M
            or abs(bin_y - ly) > EGO_TRACKING_TOL_M
            or abs(_angle_diff(heading, lh)) > EGO_TRACKING_TOL_RAD
        ):
            raise RuntimeError(
                f"nuPlan ego pose ({bin_x:.3f}, {bin_y:.3f}, {heading:.4f}) diverged from the integrated "
                f"shadow pose ({lx:.3f}, {ly:.3f}, {lh:.4f}); the planner assumes perfect_tracking_controller"
            )

    # --- planning -------------------------------------------------------------
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        import torch

        import pufferlib.pytorch

        ego_state, detections = current_input.history.current_state
        if self._env is None:
            self._build(ego_state)

        self._sync(ego_state, detections, current_input.traffic_light_data)
        if self._env.tick >= self._arch["scenario_length"] - 1:
            raise RuntimeError(
                f"shadow env tick {self._env.tick} reached scenario_length {self._arch['scenario_length']}; "
                "nuPlan planned more iterations than the scenario declared"
            )
        obs = np.asarray(self._env.recompute_observations())

        if self._policy is None:  # dummy wiring test: constant forward jerk
            act = np.full((self._env.num_agents, 1), 3 * 3 + 1, dtype=np.int32)
        else:
            import pufferlib.spaces

            # A discrete policy head on a continuous env needs the bin->continuous
            # mapping (pufferl.py feeds cont_action to the env, never the bin indices).
            env_continuous = isinstance(self._env.single_action_space, pufferlib.spaces.Box)
            if self._deterministic:
                action_selection = (
                    pufferlib.pytorch.ACTION_SELECT_MEAN
                    if env_continuous and not self._policy.is_continuous
                    else pufferlib.pytorch.ACTION_SELECT_MODE
                )
            else:
                action_selection = pufferlib.pytorch.ACTION_SELECT_SAMPLE
            obs_tensor = torch.as_tensor(obs).to(self._device)
            with torch.no_grad():
                logits, value = self._policy.forward_eval(obs_tensor)
                action, _, entropy, cont_action = pufferlib.pytorch.sample_logits(
                    logits,
                    action_selection=action_selection,
                    env_continuous=env_continuous,
                    policy=self._policy,
                )
            env_action = cont_action if env_continuous and cont_action is not None else action
            act = env_action.cpu().numpy().reshape(self._env.num_agents, -1)
            if not env_continuous:
                act = act.astype(np.int32)
            if self._obs_replay is not None:
                # sample_logits' discrete `action` is the class taken (mode/sample) or the argmax under mean selection
                action_index = action.cpu().numpy().reshape(-1) if not self._policy.is_continuous else None
                self._obs_replay.capture(
                    obs, act, self._obs_replay.policy_outputs(obs_tensor, logits, value, entropy), action_index
                )

        self._env.step(act)  # integrates the ego one dt; background is re-synced next call

        if self._obs_replay is not None and current_input.iteration.index >= self._scenario.get_number_of_iterations() - 2:
            out = self._obs_replay.write(render_html=self._obs_html_render)
            print(f"[pufferdrive_planner] wrote obs replay ({len(self._obs_replay)} steps) -> {out}")
            self._obs_replay = None

        # integrated ego -> nuPlan EgoState at t + dt
        dt = float(self._arch["dt"])
        state = self._env.get_global_agent_state()
        nx_, ny_ = self._transform.bin_to_loc(float(state["x"][0]), float(state["y"][0]))
        nh = float(state["heading"][0])
        self._last_integrated = (float(state["x"][0]), float(state["y"][0]), nh)

        # speed = the shadow ego's POST-STEP velocity state (ego obs column 0),
        # not displacement/dt, which halves the intent during acceleration and
        # makes the closed loop crawl (see WorldSync.integrate).
        ego_obs = np.asarray(self._env.observations)[0]
        speed = float(ego_obs[0]) * self._env.obs_norm_speed_mps
        accel_long = float(ego_obs[EGO_OBS_ACCEL_LONG_IDX]) * binding.ACCEL_LONG_NORM
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
                center_acceleration_2d=StateVector2D(accel_long if i == 1 else 0.0, 0.0),
                tire_steering_angle=_steering_from(yaw_rate, speed, params.wheel_base),
                time_point=TimePoint(t0.time_us + int(i * dt * 1e6)),
                vehicle_parameters=params,
                angular_vel=yaw_rate,
            )

        n_poses = int(round(self._horizon_seconds / dt))
        tail_stride = max(1, int(round(TRAJECTORY_TAIL_SPACING_S / dt)))
        pose_indices = [0, 1] + list(range(1 + tail_stride, n_poses + 1, tail_stride))
        if pose_indices[-1] != n_poses:
            pose_indices.append(n_poses)
        return InterpolatedTrajectory([_state_at(i) for i in pose_indices])
