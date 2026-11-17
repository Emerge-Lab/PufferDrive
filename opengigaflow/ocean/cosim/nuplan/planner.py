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

from opengigaflow.ocean.cosim import nuplan_bridge as nb
from opengigaflow.ocean.cosim.arch import (
    EGO_OBS_ACCEL_LONG_IDX,
    EGO_OBS_SPEED_IDX,
    EGO_OBS_STEERING_IDX,
    FAR_AWAY,
    SCENARIO_LENGTH_MARGIN_STEPS,
    checkpoint_config_path,
    load_checkpoint_config,
    shadow_env_kwargs,
)
from opengigaflow.ocean.cosim.goals import RouteGoalWindow, route_goals_from_xy
from opengigaflow.ocean.cosim.light_timing import YellowCountdown
from opengigaflow.ocean.cosim.nuplan_bicycle import (
    BicycleCommandConfig,
    bicycle_commands,
    bicycle_response,
    floor_clamped_intent_accel,
)
from opengigaflow.ocean.drive import binding
from opengigaflow.ocean.drive.drive import Drive

BICYCLE_RESPONSE_TOL = 1e-4  # m/s^2 and m/s: nuPlan's ego state must be the bicycle model's response to our command
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
    """One planning call: the plant state it saw, the command it sent and the jerk integrator's post-step state."""

    time_us: int
    plant_accel_mps2: float
    plant_speed_mps: float
    accel_cmd_mps2: float
    intent_accel_mps2: float
    intent_steering_rad: float


class OpenGigaFlowPlanner(AbstractPlanner):
    """nuPlan planner driven by an OpenGigaFlow policy; the ego plant is nuPlan's KinematicBicycleModel behind CaRL's
    one_stage_controller. The shadow ego takes the plant's pose, speed and yaw rate but keeps its own jerk-integrated
    accel/steering (the state training showed; the plant lags), whose post-step values become the commands."""

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
        sliding_goal_window: bool = False,
        startup_jerk_cap_seconds: float = 1.5,
        startup_accel_jerk_cap_mps3: float = 0.0,
        startup_brake_jerk_cap_mps3: float = 0.0,
        vru_size_pad_m: float = 0.0,
        bicycle_min_speed_mps: float = 0.0,
        bicycle_center_accel_cap: bool = False,
        bicycle_center_accel_cap_margin: float = 1.0,
        env_overrides: Optional[Dict] = None,
        obs_html_dir: Optional[str] = None,
        obs_html_max_steps: int = 800,
        obs_html_render: bool = True,
    ):
        self._checkpoint_path = checkpoint_path
        self._bin_cache_dir = Path(bin_cache_dir)
        self._city_bin_dir = Path(city_bin_dir) if city_bin_dir else None
        self._scenario = scenario
        self._device = device
        self._map_radius = float(map_radius)
        self._goal_spacing = float(goal_spacing)
        self._num_agents = int(num_agents)
        self._horizon_seconds = float(horizon_seconds)
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

    def _find_city_bin(self) -> Path:
        """City bin of the scenario's map (the log db's map_version); city_bin_dir covers every map."""
        map_name = str(self._initialization.map_api.map_name).replace(".gpkg", "")
        return self._city_bin_dir / f"nuplan__{map_name}.bin"

    @staticmethod
    def _bin_fingerprint(bin_path: Path) -> str:
        """Size and mtime of a bin; files derived from it carry this, so a regenerated bin never reuses them."""
        stat = bin_path.stat()
        return f"{stat.st_size:x}-{stat.st_mtime_ns:x}"

    def _city_bin_origin(self, bin_path: Path, stop_line_centers: np.ndarray) -> nb.NuPlanTransform:
        """A city bin's frame origin (it stores none): stop-line translation vote, then translation-only ICP of the
        bin's lane centerlines onto the GPKG baseline paths; cached in a sidecar json, one registration per city."""
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

        map_api = self._initialization.map_api
        stops = map_api._load_vector_map_layer("stop_polygons").geometry.centroid
        init = nb.coarse_translation_vote(stop_line_centers, np.stack([stops.x.to_numpy(), stops.y.to_numpy()], axis=1))
        lane_utm = shapely.get_coordinates(map_api._load_vector_map_layer("baseline_paths").geometry.values)
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
            f"[opengigaflow_planner] registered {bin_path.name}: "
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
        return city_bin, self._city_bin_origin(city_bin, geo["stop_line_centers"]), geo

    def _match_traffic_lights(self, geo: dict, ex: float, ey: float) -> Dict[str, int]:
        """Signalized lane connectors within map_radius -> bin traffic elements, by entry pose to co-directional
        stop lines."""
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
        """Shadow-env Drive kwargs: shadow_env_kwargs with env_overrides on top, the structural co-sim keys last."""
        if self._scenario is None:
            raise RuntimeError("OpenGigaFlowPlanner needs the scenario injected (requires_scenario)")
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
                # the external sim owns the episode: termination_mode=1 would c_reset() the pool, ego included
                "termination_mode": 0,
                # native-eval scenario batching builds zero envs on a map-only bin
                "eval_mode": False,
                # Enforcement off (flags still fire): a "stop" latch would freeze the ego for good.
                "collision_behavior": "ignore",
                "offroad_behavior": "ignore",
                "traffic_light_behavior": "ignore",
            },
        )

    def _build(self, ego_state: EgoState) -> None:
        import torch

        import opengigaflow.ocean.torch as drive_torch
        import opengigaflow.pytorch

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
            center_accel_cap_mps2=EGO_ACCEL_CAP_MPS2 * self._bicycle_center_accel_cap_margin
            if self._bicycle_center_accel_cap
            else 0.0,
            wheel_base_m=float(vehicle.wheel_base),
            rear_axle_to_center_m=float(vehicle.rear_axle_to_center),
        )

        # round-trip probe: slot 0 is silently not the ego when gigaflow spawning failed on the bin
        probe_x, probe_y = self._transform.loc_to_bin(ex, ey)
        self._env.set_agent_states(
            [0], [probe_x], [probe_y], [0.0], [float(ego_state.center.heading)], [0.0], [0.0], [0.0], [0.0]
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
            policy_kwargs["fused_slot_encoder"] = False  # CUDA-only kernels; CPU uses the reference path
        policy = getattr(drive_torch, cfg.get("policy_name", "Drive"))(self._env, **policy_kwargs)
        checkpoint = _checkpoint_cache.get(str(self._checkpoint_path))
        if checkpoint is None:
            checkpoint = torch.load(self._checkpoint_path, map_location=self._device, weights_only=False)
            _checkpoint_cache[str(self._checkpoint_path)] = checkpoint
        self._policy = opengigaflow.pytorch.load_policy_weights(policy, checkpoint).to(self._device).eval()

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

        fp = ego_state.car_footprint
        self._env.set_agent_sizes([0], [fp.length], [fp.width])

        if self._obs_html_dir:
            from opengigaflow.ocean.cosim.obs_replay import ObsReplayCapture

            token = self._scenario.token if self._scenario else "scenario"
            self._obs_replay = ObsReplayCapture(
                self._env,
                self._policy,
                self._obs_html_dir / token,
                max_steps=self._obs_html_max_steps,
                ghost_trajectory=nb.logged_ego_boxes(self._scenario, self._transform),
            )

        print(
            f"[opengigaflow_planner] bin={bin_path.name} lights={len(self._connector_map)}/"
            f"{self._num_traffic} goals={len(goals)}"
        )

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
        env.set_agent_states(
            [0],
            [ex],
            [ey],
            [0.0],
            [h],
            [speed * math.cos(h)],
            [speed * math.sin(h)],
            [float(dcs.angular_velocity)],
            [floor_clamped_intent_accel(self._bicycle_cfg, speed, intent_accel)],
            steering=[float(ego_state.tire_steering_angle) if last is None else last.intent_steering_rad],
        )

        # stopped time follows the nuPlan track, not the shadow slot (slots shift whenever a track appears or ends)
        self._advance_partner_stopped_s(detections.tracked_objects)
        objs = nb.partner_tracked_objects(
            detections.tracked_objects, self._initialization.map_api, self._static_on_lane
        )
        partner_count = 0
        if objs:
            x, y, hh, vx, vy, tp, ln, wd = nb.tracked_objects_to_arrays(objs, tf)
            stopped_s = np.array([self._partner_stopped_s[obj.track_token] for obj in objs], np.float32)
            if self._vru_size_pad_m > 0:
                ln, wd = nb.pad_vru_partner_sizes(tp, ln, wd, self._vru_size_pad_m)
            # nearest first only when the slots run out: the devkit orders by type/token
            keep = nb.nearest_partner_slots(x, y, ex, ey, self._num_agents - 1)
            partner_count = len(keep)
            idx = np.arange(1, 1 + partner_count, dtype=np.int32)
            # tracks carry no z, yaw rate or acceleration; partner obs never read the latter two, so zero is exact
            zeros = np.zeros(partner_count, np.float32)
            env.set_agent_states(
                idx, x[keep], y[keep], zeros, hh[keep], vx[keep], vy[keep], zeros, zeros, stopped_s[keep]
            )
            env.set_agent_sizes(idx, ln[keep], wd[keep])
        surplus = np.arange(1 + partner_count, self._num_agents, dtype=np.int32)
        if len(surplus):
            far = np.full(len(surplus), FAR_AWAY, np.float32)
            zero = np.zeros_like(far)
            env.set_agent_states(surplus, far, far, far, zero, zero, zero, zero, zero)

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

    def _policy_forward(self, obs: np.ndarray):
        """One policy call -> (env actions [row, action dim], obs tensor, logits, value, entropy, discrete classes)."""
        import torch

        import opengigaflow.pytorch

        action_selection = (
            opengigaflow.pytorch.ACTION_SELECT_MODE
            if self._policy.is_continuous
            else opengigaflow.pytorch.ACTION_SELECT_MEAN
        )
        obs_tensor = torch.as_tensor(obs).to(self._device)
        with torch.no_grad():
            logits, value = self._policy.forward_eval(obs_tensor)
            action, _, entropy, cont_action = opengigaflow.pytorch.sample_logits(
                logits,
                action_selection=action_selection,
                env_continuous=True,
                policy=self._policy,
            )
        env_action = cont_action if cont_action is not None else action
        act = env_action.cpu().numpy().reshape(len(obs), -1)
        return act, obs_tensor, logits, value, entropy, action

    def _capture_obs_replay(self, obs, act, obs_tensor, logits, value, entropy, action) -> None:
        # sample_logits' discrete `action` is the mode, or the argmax under mean selection
        action_index = action.cpu().numpy().reshape(-1) if not self._policy.is_continuous else None
        self._obs_replay.capture(
            obs, act, self._obs_replay.policy_outputs(obs_tensor, logits, value, entropy), action_index
        )

    def _write_obs_replay_at_scenario_end(self, iteration_index: int) -> None:
        if self._obs_replay is None or iteration_index < self._scenario.get_number_of_iterations() - 2:
            return
        out = self._obs_replay.write(render_html=self._obs_html_render)
        print(f"[opengigaflow_planner] wrote obs replay ({len(self._obs_replay)} steps) -> {out}")
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
        shadow_speed = float(ego_obs[EGO_OBS_SPEED_IDX]) * self._env.obs_norm_speed_mps
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
