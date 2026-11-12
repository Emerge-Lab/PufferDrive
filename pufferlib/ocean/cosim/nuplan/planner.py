import json
import math
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Type

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from carl_nuplan.planning.simulation.trajectory.action_trajectory import ActionTrajectory

from pufferlib.ocean.cosim import nuplan_bridge as nb
from pufferlib.ocean.cosim.arch import checkpoint_config_path, load_checkpoint_config
from pufferlib.ocean.cosim.goals import RouteGoalWindow, route_goals_from_xy
from pufferlib.ocean.cosim.light_timing import YellowCountdown
from pufferlib.ocean.cosim.nuplan_bicycle import (
    DEVKIT_ACCEL_TIME_CONSTANT_S,
    DEVKIT_STEERING_TIME_CONSTANT_S,
    BicycleCommandConfig,
    bicycle_commands,
    bicycle_response,
    floor_clamped_intent_accel,
)
from pufferlib.ocean.drive.drive import Drive

from pufferlib.ocean.cosim.nuplan_bridge import FAR_AWAY
from pufferlib.ocean.drive import binding

EGO_OBS_ACCEL_LONG_IDX = 4  # write_ego_obs column order: speed, width, length, steering, accel_long, ...
EGO_OBS_STEERING_IDX = 3
BICYCLE_RESPONSE_TOL = 1e-4  # m/s^2 and m/s: nuPlan's ego state must be the bicycle model's response to our command
SCENARIO_LENGTH_MARGIN_STEPS = 2  # shadow env must never hit its own truncation while nuPlan still plans
NUPLAN_COMFORT_MAX_LON_ACCEL_MPS2 = 2.40  # ego_lon_acceleration_statistics.yaml max_lon_accel
COMFORT_ACCEL_MARGIN = 0.96  # nuPlan Savitzky-Golay-smooths the reported accel and overshoots a hard cap by ~0.5%
EGO_ACCEL_CAP_MPS2 = COMFORT_ACCEL_MARGIN * NUPLAN_COMFORT_MAX_LON_ACCEL_MPS2  # sets the policy's c_acc conditioning
ROADBLOCK_GOAL_MARGIN_M = 5.0  # first roadblock goal sits at least this far beyond the goal radius ahead of the ego
LOG_EXCLUDED_STATE = ("_env", "_policy", "_goal_window", "_obs_replay")  # per-scenario runtime, never pickled


def startup_jerk_action_bounds(accel_cap_mps3: float, brake_cap_mps3: float):
    """[lo, hi] of the continuous longitudinal jerk action for the caps (0 = uncapped): the env maps
    action < 0 onto [JERK_LONG[0], 0] and action > 0 onto [0, JERK_LONG[-1]] m/s³."""
    lo = -1.0 if brake_cap_mps3 <= 0 else max(-1.0, -brake_cap_mps3 / abs(float(binding.JERK_LONG[0])))
    hi = 1.0 if accel_cap_mps3 <= 0 else min(1.0, accel_cap_mps3 / float(binding.JERK_LONG[-1]))
    return lo, hi


# per-process caches: one planner per scenario, same checkpoint/bins each time
_checkpoint_cache: Dict[str, dict] = {}
_bin_geometry_cache: Dict[str, dict] = {}


class BicycleStep(NamedTuple):
    """One planning call: the plant state it saw, the command it sent and the jerk integrator's post-step state
    (the accel/steering the policy observes and the next call restamps)."""

    time_us: int
    plant_accel_mps2: float
    plant_speed_mps: float
    accel_cmd_mps2: float
    intent_accel_mps2: float
    intent_steering_rad: float


class PufferDrivePlanner(AbstractPlanner):
    """nuPlan planner driven by a PufferDrive policy. nuPlan's KinematicBicycleModel is the ego plant, commanded
    through CaRL's one_stage_controller like CaRL: every step the shadow ego takes nuPlan's pose, speed and yaw
    rate, keeps its own jerk-integrated accel/steering (the state the policy was trained to observe; the plant
    lags behind it), and its post-step accel/steering become the model's commands."""

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
        sliding_goal_window: bool = False,
        startup_jerk_cap_seconds: float = 1.5,
        startup_accel_jerk_cap_mps3: float = 0.0,
        startup_brake_jerk_cap_mps3: float = 0.0,
        vru_size_pad_m: float = 0.0,
        bicycle_min_speed_mps: float = 0.0,
        bicycle_accel_time_constant_s: float = DEVKIT_ACCEL_TIME_CONSTANT_S,
        bicycle_steering_time_constant_s: float = DEVKIT_STEERING_TIME_CONSTANT_S,
        bicycle_center_accel_cap: bool = False,
        bicycle_center_accel_cap_margin: float = 1.0,
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
        :param goal_spacing: minimum distance between consecutive route roadblock goals fed to the policy [m].
        :param num_agents: PufferDrive agent-pool size (ego + streamed background).
        :param horizon_seconds: horizon of the returned action trajectory's sampling [s].
        :param sliding_goal_window: refill the goal window to `num_goals` goals ahead after every
            consumed goal (default: replace it only once exhausted, as in training), so the policy
            sees a window-final speed-gated goal only at the true route end.
        :param startup_jerk_cap_seconds: length of the scenario-start window the jerk caps apply to [s].
        :param startup_accel_jerk_cap_mps3: cap on the ego's positive (accelerating) longitudinal jerk
            during the start window [m/s³]; 0 = uncapped. nuPlan's Savitzky-Golay jerk extrapolates at
            the trajectory edges, so a ramp that starts at t=0 reads ~2x its raw jerk.
        :param startup_brake_jerk_cap_mps3: same cap for braking jerk; 0 = uncapped.
        :param vru_size_pad_m: eval hack: pedestrian and bicycle partner boxes grow by this much in length
            and width [m] before the policy sees them, so they read strictly larger than the smallest
            training vehicle (0.8 x 0.8 m; nuPlan pedestrians are 0.4-0.8 m, bicycles ~0.5 m wide); nuPlan
            keeps scoring the true boxes. 0 = off.
        :param bicycle_min_speed_mps: speed floor of the plant [m/s], enforced on the command (the model has
            none): 0 = CaRL's no-reverse driving, -2 = Drive's MAX_BACKWARD_SPEED.
        :param bicycle_accel_time_constant_s: the motion model's acceleration low-pass time constant [s]
            (devkit default 0.2); a mismatch fails the per-step response check.
        :param bicycle_steering_time_constant_s: same for steering [s] (devkit default 0.05).
        :param bicycle_center_accel_cap: CaRL's clip_angular_adjustment. nuPlan scores the
            acceleration at the vehicle center, which the devkit computes as rear-axle accel + d*yaw_rate^2 +
            d*yaw_accel (d = rear axle to center), so the plant's acceleration is reduced while turning to keep
            that reading under EGO_ACCEL_CAP_MPS2 x bicycle_center_accel_cap_margin. The policy keeps its own
            intent and observations.
        :param bicycle_center_accel_cap_margin: factor on EGO_ACCEL_CAP_MPS2 for the center cap (nuPlan smooths
            the reported acceleration with a Savitzky-Golay window and overshoots a peaky capped profile).
        :param env_overrides: Drive env kwargs applied on top of the checkpoint's env config.
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
        self._num_agents = int(num_agents)
        self._horizon_seconds = float(horizon_seconds)
        self._deterministic = deterministic
        self._sliding_goal_window = bool(sliding_goal_window)
        self._startup_jerk_cap_seconds = float(startup_jerk_cap_seconds)
        self._startup_accel_jerk_cap_mps3 = float(startup_accel_jerk_cap_mps3)
        self._startup_brake_jerk_cap_mps3 = float(startup_brake_jerk_cap_mps3)
        if (
            min(self._startup_jerk_cap_seconds, self._startup_accel_jerk_cap_mps3, self._startup_brake_jerk_cap_mps3)
            < 0
        ):
            raise ValueError("startup_jerk_cap_seconds and the startup jerk caps must be >= 0")
        self._vru_size_pad_m = float(vru_size_pad_m)
        if self._vru_size_pad_m < 0:
            raise ValueError("vru_size_pad_m must be >= 0")
        self._bicycle_min_speed_mps = float(bicycle_min_speed_mps)
        if not binding.MAX_BACKWARD_SPEED <= self._bicycle_min_speed_mps <= 0.0:
            raise ValueError(
                f"bicycle_min_speed_mps must be within [{binding.MAX_BACKWARD_SPEED}, 0], got {bicycle_min_speed_mps}"
            )
        self._bicycle_accel_time_constant_s = float(bicycle_accel_time_constant_s)
        self._bicycle_steering_time_constant_s = float(bicycle_steering_time_constant_s)
        if not (self._bicycle_accel_time_constant_s >= 0 and self._bicycle_steering_time_constant_s >= 0):
            raise ValueError("the bicycle model time constants must be >= 0")
        self._bicycle_center_accel_cap = bool(bicycle_center_accel_cap)
        self._bicycle_center_accel_cap_margin = float(bicycle_center_accel_cap_margin)
        if not 0 < self._bicycle_center_accel_cap_margin <= 1:
            raise ValueError(
                f"bicycle_center_accel_cap_margin must be in (0, 1], got {bicycle_center_accel_cap_margin}"
            )
        self._bicycle_cfg: Optional[BicycleCommandConfig] = None  # resolved in _build from the scenario dt
        self._last_bicycle_step: Optional[BicycleStep] = None
        self._startup_jerk_cap_steps = 0  # resolved in _build from the scenario dt
        self._startup_action_bounds = None  # (lo, hi) on the continuous jerk action while the cap is live
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
        self._last_light_states = None  # traffic-light state array from the most recent _sync
        self._yellow_countdown = None  # YellowCountdown, built with the env
        self._static_on_lane: Dict[str, bool] = {}  # static object track_token -> its box overlaps a lane polygon
        self._partner_stopped_s: Dict[str, float] = {}  # track_token -> seconds below the stopped speed threshold
        self._planning_step_s: Optional[float] = None  # nuPlan's planning interval, resolved in _build

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

    def __getstate__(self):
        # nuPlan pickles the planner into every simulation log: 30 MB of policy weights per scenario without this
        state = dict(self.__dict__)
        for name in LOG_EXCLUDED_STATE:
            state[name] = None
        return state

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

    @staticmethod
    def _bin_fingerprint(bin_path: Path) -> str:
        """Size and mtime of a bin; files derived from it carry this, so a regenerated bin never reuses them."""
        stat = bin_path.stat()
        return f"{stat.st_size:x}-{stat.st_mtime_ns:x}"

    def _city_bin_origin(self, bin_path: Path, stop_line_centers: np.ndarray) -> nb.NuPlanTransform:
        """Recover a city bin's frame origin (it stores no centroid): global
        translation vote on stop-line landmarks, then translation-only ICP of
        the bin's lane centerlines onto the GPKG baseline paths. Cached in a
        sidecar json — one registration per city, ever."""
        sidecar = bin_path.with_suffix(".origin.json")
        fingerprint = self._bin_fingerprint(bin_path)
        cache = sidecar if sidecar.exists() else self._bin_cache_dir / sidecar.name
        cached = json.loads(cache.read_text()) if cache.exists() else None
        # sidecars written before the fingerprint existed carry no "source" and stay trusted
        if cached is not None and cached.get("source", fingerprint) == fingerprint:
            return nb.NuPlanTransform(cached["origin"][0], cached["origin"][1])

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
        payload = json.dumps({"origin": [float(t[0]), float(t[1])], "residual_m": resid, "source": fingerprint})
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
        """-> (bin_path, NuPlanTransform, bin geometry dict from read_bin_geometry)."""
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
        return city_bin, tf, geo

    def _match_traffic_lights(self, geo: dict, ex: float, ey: float) -> Dict[str, int]:
        """Signalized lane connectors within map_radius -> bin traffic elements,
        matched by entry pose to co-directional stop lines (no sidecar file needed)."""
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer

        objs = self._initialization.map_api.get_proximal_map_objects(
            Point2D(ex, ey), self._map_radius, [SemanticMapLayer.LANE_CONNECTOR]
        )
        entries = {}
        for e in objs[SemanticMapLayer.LANE_CONNECTOR]:
            if e.has_traffic_lights():
                p0 = e.baseline_path.discrete_path[0]
                entries[str(e.id)] = (float(p0.x), float(p0.y), float(p0.heading))
        return nb.match_connectors_to_stop_lines(
            entries, self._transform, geo["stop_lines"], geo["stop_line_headings"], geo["traffic_types"]
        )

    def _resolve_arch(self, cfg: Dict, bin_path: Path) -> Dict:
        """Shadow-env Drive kwargs: the checkpoint's training env section
        decides every Drive-accepted key, the clean-eval profile is applied on
        top (see cosim/arch.py), `env_overrides` wins -- except for the
        structural co-sim keys (map/pool wiring), which are not overridable."""
        from pufferlib.ocean.cosim.arch import shadow_env_kwargs

        if self._scenario is None:
            raise RuntimeError("PufferDrivePlanner needs the scenario injected (requires_scenario)")
        return shadow_env_kwargs(
            cfg,
            overrides={
                **self._env_overrides,
                "map_dir": str(bin_path),
                "num_maps": 1,
                # One policy agent (the ego); every other slot is a static partner streamed from nuPlan.
                "num_agents": 1,
                "min_agents_per_env": 1,
                "cosim_partner_slots": self._num_agents - 1,
                "goal_source": "external",
                # Training semantics: a window's final goal clears only below goal_speed (3 m/s)
                "goal_reach_requires_speed": True,
                # C_acc conditioning: cap the jerk model's positive accel at EGO_ACCEL_CAP_MPS2 instead of 2.5
                "conditioning_accel_scale": EGO_ACCEL_CAP_MPS2 / binding.ACCEL_LONG_MAX,
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

    # --- lazy construction ----------------------------------------------------
    def _build(self, ego_state: EgoState) -> None:
        import torch

        import pufferlib.ensemble
        import pufferlib.ocean.torch as drive_torch

        init = self._initialization
        ex, ey = float(ego_state.center.x), float(ego_state.center.y)

        cfg = load_checkpoint_config(checkpoint_config_path(self._checkpoint_path))

        bin_path, self._transform, geo = self._resolve_map_bin()
        self._num_traffic = geo["num_traffic"]
        self._connector_map = self._match_traffic_lights(geo, ex, ey)

        self._arch = self._resolve_arch(cfg, bin_path)
        self._env = Drive(**self._arch)
        self._env.reset()
        self._planning_step_s = float(self._scenario.database_interval)
        # nuPlan reports light status only: a yellow counts down from the shortest trained yellow
        self._yellow_countdown = YellowCountdown(self._env.traffic_light_yellow_min_s, self._env.dt)
        torch.set_num_threads(1)  # one scenario per worker process; BLAS threads only fight the other workers
        if max(self._startup_accel_jerk_cap_mps3, self._startup_brake_jerk_cap_mps3) > 0:
            self._startup_jerk_cap_steps = int(round(self._startup_jerk_cap_seconds / float(self._arch["dt"])))
            self._startup_action_bounds = startup_jerk_action_bounds(
                self._startup_accel_jerk_cap_mps3, self._startup_brake_jerk_cap_mps3
            )

        vehicle = ego_state.car_footprint.vehicle_parameters
        self._bicycle_cfg = BicycleCommandConfig(
            dt_s=float(self._arch["dt"]),
            min_speed_mps=self._bicycle_min_speed_mps,
            accel_time_constant_s=self._bicycle_accel_time_constant_s,
            steering_time_constant_s=self._bicycle_steering_time_constant_s,
            center_accel_cap_mps2=EGO_ACCEL_CAP_MPS2 * self._bicycle_center_accel_cap_margin
            if self._bicycle_center_accel_cap
            else 0.0,
            wheel_base_m=float(vehicle.wheel_base),
            rear_axle_to_center_m=float(vehicle.rear_axle_to_center),
        )

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
        policy_kwargs = dict(cfg["policy"])
        if torch.device(self._device).type == "cpu":
            policy_kwargs["fused_slot_encoder"] = False  # CUDA-only triton kernel; CPU uses the reference path
        policy = getattr(drive_torch, cfg.get("policy_name", "Drive"))(self._env, **policy_kwargs)
        checkpoint = _checkpoint_cache.get(str(self._checkpoint_path))
        if checkpoint is None:
            checkpoint = torch.load(self._checkpoint_path, map_location=self._device, weights_only=False)
            _checkpoint_cache[str(self._checkpoint_path)] = checkpoint
        self._policy = pufferlib.ensemble.load_policy_weights(policy, checkpoint).to(self._device).eval()

        # the lane-graph route through the CaRL-corrected route roadblocks decides the shared stop-line elements
        from carl_nuplan.planning.simulation.planner.pdm_planner.utils.route_utils import (
            route_roadblock_correction_v2,
        )

        route_ids = route_roadblock_correction_v2(ego_state, init.map_api, list(init.route_roadblock_ids))
        self._route_connector_ids = nb.route_lane_ids(init.map_api, route_ids, ex, ey, float(ego_state.center.heading))
        goals = nb.roadblock_centroid_goals(
            init.map_api,
            nb.extend_route_past_loop_cut(route_ids, init.route_roadblock_ids),
            ex,
            ey,
            float(ego_state.center.heading),
            self._goal_spacing,
            float(self._env.goal_radius) + ROADBLOCK_GOAL_MARGIN_M,
        )
        if len(goals) == 0:  # degenerate route: fall back to the mission goal
            goals = np.array([[init.mission_goal.x, init.mission_goal.y]])
        goals -= (self._transform.ox, self._transform.oy)
        route_goals = route_goals_from_xy(goals, origin_xy=self._transform.loc_to_bin(ex, ey))
        self._goal_window = RouteGoalWindow(self._env, route_goals, sliding=self._sliding_goal_window)

        # ego bounding box from nuPlan vehicle parameters (static)
        fp = ego_state.car_footprint
        self._env.set_agent_sizes(
            np.array([0], np.int32), np.array([fp.length], np.float32), np.array([fp.width], np.float32)
        )

        if self._obs_html_dir:
            from pufferlib.ocean.cosim.obs_replay import ObsReplayCapture

            token = self._scenario.token if self._scenario else "scenario"
            self._obs_replay = ObsReplayCapture(
                self._env,
                self._policy,
                self._obs_html_dir / token,
                max_steps=self._obs_html_max_steps,
                ghost_trajectory=nb.logged_ego_boxes(self._scenario, self._transform),
            )

        print(
            f"[pufferdrive_planner] bin={bin_path.name} lights={len(self._connector_map)}/"
            f"{self._num_traffic} goals={len(goals)}"
        )

    # --- world sync -----------------------------------------------------------
    def _advance_partner_stopped_s(self, tracked_objects) -> None:
        """One observation step of the stopped time per nuPlan track; a track missing from the step starts over."""
        step_s = self._planning_step_s
        stopped_s = {}
        for obj in tracked_objects:
            velocity = getattr(obj, "velocity", None)
            speed_mps = math.hypot(velocity.x, velocity.y) if velocity is not None else 0.0
            is_stopped = speed_mps < binding.AGENT_STOPPED_SPEED_THRESHOLD
            previous_s = self._partner_stopped_s.get(obj.track_token, 0.0)
            stopped_s[obj.track_token] = previous_s + step_s if is_stopped else 0.0
        self._partner_stopped_s = stopped_s

    def _sync(self, ego_state: EgoState, detections: DetectionsTracks, traffic_light_data) -> None:
        tf = self._transform
        env = self._env

        # ego (slot 0): the plant's pose, speed and yaw rate; its lagged accel/steering would make the snaps stick
        h = float(ego_state.center.heading)
        dcs = ego_state.dynamic_car_state
        ex, ey = tf.loc_to_bin(ego_state.center.x, ego_state.center.y)
        self._check_bicycle_response(ego_state)
        speed = float(dcs.rear_axle_velocity_2d.x)
        last = self._last_bicycle_step
        intent_accel = float(dcs.rear_axle_acceleration_2d.x) if last is None else last.intent_accel_mps2
        self._stamp_ego(
            ex,
            ey,
            h,
            speed * math.cos(h),
            speed * math.sin(h),
            float(dcs.angular_velocity),
            floor_clamped_intent_accel(self._bicycle_cfg, speed, intent_accel),
            float(ego_state.tire_steering_angle) if last is None else last.intent_steering_rad,
        )

        # background (slots 1..): streamed from nuPlan, never simulated here. nuPlan's TrackedObject
        # carries no acceleration/angular-velocity (perception detections, not ego telemetry), but
        # write_partner_obs never reads those fields for non-ego agents, so zero is exact, not a stopgap.
        # stopped time follows the nuPlan track, not the shadow slot (slots shift whenever a track appears or ends)
        self._advance_partner_stopped_s(detections.tracked_objects)
        objs = nb.partner_tracked_objects(
            detections.tracked_objects, self._initialization.map_api, self._static_on_lane
        )
        partner_count = 0
        if objs:
            _, x, y, z, hh, vx, vy, tp, ln, wd = nb.tracked_objects_to_arrays(objs, tf)
            stopped_s = np.array([self._partner_stopped_s[obj.track_token] for obj in objs], np.float32)
            if self._vru_size_pad_m > 0:
                ln, wd = nb.pad_vru_partner_sizes(tp, ln, wd, self._vru_size_pad_m)
            # nearest first only when the slots run out: devkit ordering is by type/token, so blind truncation
            # could drop a close vehicle while keeping far ones
            keep = nb.nearest_partner_slots(x, y, ex, ey, self._num_agents - 1)
            partner_count = len(keep)
            idx = np.arange(1, 1 + partner_count, dtype=np.int32)
            zeros = np.zeros(partner_count, np.float32)
            env.set_agent_states(
                idx, x[keep], y[keep], z[keep], hh[keep], vx[keep], vy[keep], zeros, zeros, stopped_s[keep]
            )
            env.set_agent_sizes(idx, ln[keep], wd[keep])
        surplus = np.arange(1 + partner_count, self._num_agents, dtype=np.int32)
        if len(surplus):
            far = np.full(len(surplus), FAR_AWAY, np.float32)
            zero = np.zeros_like(far)
            env.set_agent_states(surplus, far, far, far, zero, zero, zero, zero, zero)

        # traffic lights: state array sized to the bin's element count
        if self._num_traffic:
            self._last_light_states = nb.traffic_light_states(
                traffic_light_data, self._connector_map, self._num_traffic, self._route_connector_ids
            )
            yellow_remaining_s = None
            if env.obs_traffic_light_yellow_remaining:
                yellow_elements = [int(j) for j in np.flatnonzero(self._last_light_states == nb.LIGHT_STATE_YELLOW)]
                countdown_s = self._yellow_countdown.tick(yellow_elements, ego_state.time_point.time_s)
                yellow_remaining_s = np.zeros(self._num_traffic, np.float32)
                for j, remaining_s in countdown_s.items():
                    yellow_remaining_s[j] = remaining_s
            env.set_traffic_light_states(self._last_light_states, yellow_remaining_s)

        self._goal_window.sync(ex, ey, h)

    def _stamp_ego(
        self,
        bin_x: float,
        bin_y: float,
        heading: float,
        vx: float,
        vy: float,
        yaw_rate: float,
        accel_long: float,
        steering: float,
    ) -> None:
        self._env.set_agent_states(
            np.array([0], np.int32),
            np.array([bin_x], np.float32),
            np.array([bin_y], np.float32),
            np.array([0.0], np.float32),
            np.array([heading], np.float32),
            np.array([vx], np.float32),
            np.array([vy], np.float32),
            np.array([yaw_rate], np.float32),
            np.array([accel_long], np.float32),
            steering=np.array([steering], np.float32),
        )

    def _check_bicycle_response(self, ego_state: EgoState) -> None:
        last = self._last_bicycle_step
        if last is None:
            return
        elapsed_s = (ego_state.time_point.time_us - last.time_us) * 1e-6
        if elapsed_s <= 0:
            raise RuntimeError(f"nuPlan ego time did not advance since the last command ({elapsed_s:.6f} s)")
        expected_accel, expected_speed = bicycle_response(
            self._bicycle_cfg, last.plant_accel_mps2, last.accel_cmd_mps2, last.plant_speed_mps, elapsed_s
        )
        dcs = ego_state.dynamic_car_state
        accel = float(dcs.rear_axle_acceleration_2d.x)
        speed = float(dcs.rear_axle_velocity_2d.x)
        if abs(accel - expected_accel) > BICYCLE_RESPONSE_TOL or abs(speed - expected_speed) > BICYCLE_RESPONSE_TOL:
            raise RuntimeError(
                f"nuPlan ego accel/speed ({accel:.4f}, {speed:.4f}) are not the kinematic bicycle model's response "
                f"({expected_accel:.4f}, {expected_speed:.4f}) to the last command; the planner needs "
                "ego_controller=one_stage_controller and its bicycle time constants must match that model"
            )

    # --- planning -------------------------------------------------------------
    def _policy_forward(self, obs: np.ndarray):
        """One policy call on observation rows -> (env actions [row, action dim], obs tensor, logits, value,
        entropy, discrete action classes)."""
        import torch

        import pufferlib.pytorch
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
        act = env_action.cpu().numpy().reshape(len(obs), -1)
        if not env_continuous:
            act = act.astype(np.int32)
        return act, obs_tensor, logits, value, entropy, action

    def _capture_obs_replay(self, obs, act, obs_tensor, logits, value, entropy, action) -> None:
        # sample_logits' discrete `action` is the class taken (mode/sample) or the argmax under mean selection
        action_index = action.cpu().numpy().reshape(-1) if not self._policy.is_continuous else None
        self._obs_replay.capture(
            obs, act, self._obs_replay.policy_outputs(obs_tensor, logits, value, entropy), action_index
        )

    def _write_obs_replay_at_scenario_end(self, iteration_index: int) -> None:
        if self._obs_replay is None or iteration_index < self._scenario.get_number_of_iterations() - 2:
            return
        out = self._obs_replay.write(render_html=self._obs_html_render)
        print(f"[pufferdrive_planner] wrote obs replay ({len(self._obs_replay)} steps) -> {out}")
        self._obs_replay = None

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        ego_state, detections = current_input.history.current_state
        if self._env is None:
            self._build(ego_state)
            # nuPlan's history buffer: its last entry is the current observation, which _sync counts
            for past_observation in current_input.history.observations[:-1]:
                self._advance_partner_stopped_s(past_observation.tracked_objects)

        self._sync(ego_state, detections, current_input.traffic_light_data)
        if self._env.tick + 1 > self._arch["scenario_length"] - 1:
            raise RuntimeError(
                f"shadow env tick {self._env.tick} reached scenario_length {self._arch['scenario_length']}; "
                "nuPlan planned more iterations than the scenario declared"
            )
        obs = np.asarray(self._env.recompute_observations())

        act, obs_tensor, logits, value, entropy, action = self._policy_forward(obs)
        if self._startup_action_bounds is not None and self._env.tick < self._startup_jerk_cap_steps:
            np.clip(act[:, 0], *self._startup_action_bounds, out=act[:, 0])
        if self._obs_replay is not None:
            self._capture_obs_replay(obs, act, obs_tensor, logits, value, entropy, action)

        self._env.step(act)  # integrates the ego one dt; background is re-synced next call

        self._write_obs_replay_at_scenario_end(current_input.iteration.index)

        return self._bicycle_action_trajectory(ego_state, act[0])

    def _bicycle_action_trajectory(self, ego_state: EgoState, action: np.ndarray) -> ActionTrajectory:
        """Shadow-env post-step accel/steering targets -> commands for nuPlan's bicycle model over the next dt."""
        ego_obs = np.asarray(self._env.observations)[0]
        intent_accel = float(ego_obs[EGO_OBS_ACCEL_LONG_IDX]) * binding.ACCEL_LONG_NORM
        intent_steering = float(ego_obs[EGO_OBS_STEERING_IDX]) * binding.STEERING_LIMIT
        shadow_speed = float(ego_obs[0]) * self._env.obs_norm_speed_mps
        dcs = ego_state.dynamic_car_state
        accel = float(dcs.rear_axle_acceleration_2d.x)
        speed = float(dcs.rear_axle_velocity_2d.x)
        accel_cmd, steering_rate_cmd = bicycle_commands(
            self._bicycle_cfg,
            intent_accel,
            intent_steering,
            accel,
            float(ego_state.tire_steering_angle),
            speed,
            yaw_rate_radps=float(dcs.angular_velocity),
            stop_this_step=speed != 0.0 and shadow_speed == 0.0,
        )
        self._last_bicycle_step = BicycleStep(
            ego_state.time_point.time_us, accel, speed, accel_cmd, intent_accel, intent_steering
        )
        sampling = TrajectorySampling(
            num_poses=int(round(self._horizon_seconds / self._bicycle_cfg.dt_s)), interval_length=self._bicycle_cfg.dt_s
        )
        return ActionTrajectory(
            accel_cmd,
            steering_rate_cmd,
            ego_state,
            raw_action=[float(a) for a in action],
            trajectory_sampling=sampling,
        )
