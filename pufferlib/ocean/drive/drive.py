import argparse
import pickle
import zlib
from pathlib import Path
import numpy as np
import gymnasium
import json
import struct
import os
import pufferlib
from pufferlib.ocean.drive import binding


def compute_effective_road_obs_count(max_count, dropout):
    if max_count <= 0:
        return 0
    clipped_dropout = min(max(float(dropout), 0.0), 1.0)
    return int(max_count * (1.0 - clipped_dropout))


class Drive(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode=None,
        report_interval=1,
        width=1280,
        height=1024,
        human_agent_idx=0,
        reward_goal=1.0,
        reward_collision=3.0,
        reward_offroad=3.0,
        reward_comfort=0.05,
        reward_lane_align=0.025,
        reward_vel_align=1.0,
        reward_lane_center=0.0038,
        reward_center_bias=0.0,
        reward_velocity=0.0025,
        reward_reverse=0.005,
        reward_stop_line=1.0,
        reward_timestep=0.000025,
        reward_overspeed=0.05,
        reward_ade=0.0,
        min_waypoint_spacing=20.0,
        max_waypoint_spacing=60.0,
        num_target_waypoints=3,
        goal_radius=2.0,
        collision_behavior=0,
        offroad_behavior=0,
        traffic_light_behavior=0,
        use_map_cache=0,
        # emit_completed_episodes=True: env emits one summary dict per
        # completed episode via info (drained from a per-env C-side queue).
        # capture_compact_replay=True additionally records per-step agent and
        # traffic state and attaches a pickled+zlib'd schema_version=2
        # `compact_replay_bundle` to each summary.
        capture_compact_replay=False,
        emit_completed_episodes=False,
        dt=0.1,
        spawn_initial_speed=0.0,
        goal_speed=3.0,
        scenario_length=None,
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        min_agents_per_env=32,
        max_agents_per_env=64,
        action_type="discrete",
        dynamics_model="classic",
        simulation_mode="gigaflow",
        termination_mode=0,
        terminate_on_goal=0,
        inactive_agent_threshold=0.4,
        buf=None,
        seed=1,
        init_step=0,
        eval_mode=0,
        num_eval_scenarios=16,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        sdc_controller="policy",
        non_sdc_controller="policy",
        non_vehicle_controller="auto",
        map_dir=None,
        target_type="static",
        goal_on_lane=True,
        reward_conditioning=False,
        reward_randomization=False,
        compute_eval_metrics=True,
        split_network=False,
        obs_slots_lane_n=32,
        obs_slots_boundary_n=32,
        obs_slots_partners_n=16,
        obs_slots_traffic_controls_n=4,
        traffic_control_scope=0,
        starting_map=0,
        obs_norm_goal_offset_m=100.0,
        obs_norm_xy_offset_m=100.0,
        obs_norm_veh_length_m=15.0,
        obs_norm_veh_width_m=10.0,
        obs_norm_road_seg_length_m=5.0,
        obs_norm_road_seg_width_m=5.0,
        obs_range_traffic_control_m=100.0,
        obs_range_partner_m=100.0,
        obs_range_road_front_m=120.0,
        obs_range_road_behind_m=20.0,
        obs_range_road_side_m=30.0,
        obs_dropout_lane=0.0,
        obs_dropout_boundary=0.0,
        partner_blindness_prob=0.0,
        partner_blindness_trigger_prob=0.1,
        phantom_braking_prob=0.0,
        phantom_braking_trigger_prob=0.0,
        phantom_braking_duration=10,
    ):
        self.dt = dt
        self.spawn_initial_speed = float(spawn_initial_speed)
        self.goal_speed = float(goal_speed)
        self.reward_conditioning = reward_conditioning
        self.reward_randomization = reward_randomization
        self.compute_eval_metrics = compute_eval_metrics
        self.split_network = split_network
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_goal = reward_goal
        self.reward_collision = reward_collision
        self.reward_offroad = reward_offroad
        self.reward_comfort = reward_comfort
        self.reward_lane_align = reward_lane_align
        self.reward_vel_align = reward_vel_align
        self.reward_lane_center = reward_lane_center
        self.reward_center_bias = reward_center_bias
        self.reward_velocity = reward_velocity
        self.reward_reverse = reward_reverse
        self.reward_stop_line = reward_stop_line
        self.reward_timestep = reward_timestep
        self.reward_overspeed = reward_overspeed
        self.reward_ade = reward_ade
        self.goal_radius = goal_radius
        self.min_waypoint_spacing = min_waypoint_spacing
        self.max_waypoint_spacing = max_waypoint_spacing
        if num_target_waypoints > binding.MAX_TARGET_WAYPOINTS:
            num_target_waypoints = binding.MAX_TARGET_WAYPOINTS
        self.num_target_waypoints = num_target_waypoints
        self.target_type_str = target_type
        if target_type == "static":
            self.target_type = binding.TARGET_STATIC
        elif target_type == "dynamic":
            self.target_type = binding.TARGET_DYNAMIC
        else:
            raise ValueError(f"target_type must be 'static' or 'dynamic'. Got: {target_type}")
        self.goal_on_lane = int(bool(goal_on_lane))
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.traffic_light_behavior = traffic_light_behavior
        if use_map_cache not in (0, 1):
            raise ValueError(f"use_map_cache must be 0 (off) or 1 (on). Got: {use_map_cache}")
        self.use_map_cache = use_map_cache
        self.capture_compact_replay = bool(capture_compact_replay)
        # capture_compact_replay implies emit_completed_episodes, since the
        # bundle rides on the per-episode summary.
        self.emit_completed_episodes = bool(emit_completed_episodes) or self.capture_compact_replay
        self._compact_replay_buffers = []
        self.human_agent_idx = human_agent_idx
        self.scenario_length = scenario_length
        self.resample_frequency = resample_frequency
        self.dynamics_model = dynamics_model
        if dynamics_model == "classic":
            self.dynamics_model_flag = 0
        elif dynamics_model == "jerk":
            self.dynamics_model_flag = 1
        else:
            raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        self.eval_mode = eval_mode
        self.num_eval_scenarios = num_eval_scenarios
        self.termination_mode = termination_mode
        self.terminate_on_goal = terminate_on_goal
        self.inactive_agent_threshold = inactive_agent_threshold
        self.rng = np.random.default_rng(seed)
        self.min_agents_per_env = min_agents_per_env
        self.max_agents_per_env = max_agents_per_env

        self.ego_features = binding.EGO_FEATURES

        # Extract observation shapes from constants
        self.obs_slots_lane_n = obs_slots_lane_n
        self.obs_slots_boundary_n = obs_slots_boundary_n
        self.obs_slots_partners_n = obs_slots_partners_n
        self.traffic_control_scope = traffic_control_scope
        self.obs_slots_traffic_controls_n = obs_slots_traffic_controls_n
        self.obs_norm_goal_offset_m = float(obs_norm_goal_offset_m)
        self.obs_norm_xy_offset_m = float(obs_norm_xy_offset_m)
        self.obs_norm_veh_length_m = float(obs_norm_veh_length_m)
        self.obs_norm_veh_width_m = float(obs_norm_veh_width_m)
        self.obs_norm_road_seg_length_m = float(obs_norm_road_seg_length_m)
        self.obs_norm_road_seg_width_m = float(obs_norm_road_seg_width_m)
        self.obs_range_traffic_control_m = float(obs_range_traffic_control_m)
        self.obs_range_partner_m = float(obs_range_partner_m)
        self.obs_range_road_front_m = float(obs_range_road_front_m)
        self.obs_range_road_behind_m = float(obs_range_road_behind_m)
        self.obs_range_road_side_m = float(obs_range_road_side_m)
        self.obs_dropout_lane = float(obs_dropout_lane)
        self.obs_dropout_boundary = float(obs_dropout_boundary)
        self.obs_slots_lane_kept = compute_effective_road_obs_count(
            self.obs_slots_lane_n,
            self.obs_dropout_lane,
        )
        self.obs_slots_boundary_kept = compute_effective_road_obs_count(
            self.obs_slots_boundary_n,
            self.obs_dropout_boundary,
        )
        self.partner_blindness_prob = float(partner_blindness_prob)
        self.partner_blindness_trigger_prob = float(partner_blindness_trigger_prob)
        self.phantom_braking_prob = float(phantom_braking_prob)
        self.phantom_braking_trigger_prob = float(phantom_braking_trigger_prob)
        self.phantom_braking_duration = int(phantom_braking_duration)
        self.partner_features = binding.PARTNER_FEATURES
        self.road_features = binding.ROAD_FEATURES
        self.traffic_control_features = binding.TRAFFIC_CONTROL_FEATURES
        self.num_reward_coefs = binding.NUM_REWARD_COEFS if reward_conditioning else 0

        # Target features based on target_type
        if target_type == "static":
            self.target_features = binding.STATIC_TARGET_FEATURES
        else:
            self.target_features = binding.DYNAMIC_TARGET_FEATURES
        self.target_dim = self.num_target_waypoints * self.target_features

        self.num_obs = (
            self.ego_features
            + self.num_reward_coefs
            + self.target_dim
            + self.obs_slots_partners_n * self.partner_features
            + self.obs_slots_lane_kept * self.road_features
            + self.obs_slots_boundary_kept * self.road_features
            + self.obs_slots_traffic_controls_n * self.traffic_control_features
        )

        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)

        self.init_step = init_step
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        self.sdc_controller_str = sdc_controller
        self.non_sdc_controller_str = non_sdc_controller
        self.non_vehicle_controller_str = non_vehicle_controller
        self.simulation_mode_str = simulation_mode
        self.map_dir = map_dir
        # map_dir may point either at a directory containing .bin files or at
        # a single .bin file (to pin training/eval to one specific map).
        if isinstance(map_dir, str) and os.path.isfile(map_dir) and map_dir.endswith(".bin"):
            self.map_files = [map_dir]
        else:
            self.map_files = sorted(os.path.join(map_dir, f) for f in os.listdir(map_dir) if f.endswith(".bin"))

        if self.simulation_mode_str == "gigaflow":
            self.simulation_mode = 0
        elif self.simulation_mode_str == "replay":
            self.simulation_mode = 1
        else:
            raise ValueError(f"simulation_mode must be one of 'gigaflow' or 'replay'. Got: {self.simulation_mode_str}")

        if self.control_mode_str == "control_vehicles":
            self.control_mode = 0
        elif self.control_mode_str == "control_agents":
            self.control_mode = 1
        elif self.control_mode_str == "control_wosac":
            self.control_mode = 2
        elif self.control_mode_str == "control_sdc_only":
            self.control_mode = 3
        else:
            raise ValueError(
                "control_mode must be one of 'control_vehicles', 'control_agents', 'control_wosac', or "
                f"'control_sdc_only'. Got: {self.control_mode_str}"
            )

        controller_values = {
            "static": binding.CONTROLLER_STATIC,
            "policy": binding.CONTROLLER_POLICY,
            "replay": binding.CONTROLLER_REPLAY,
            "idm": binding.CONTROLLER_IDM,
        }
        controller_options = "'static', 'policy', 'replay', or 'idm'"
        if self.sdc_controller_str not in controller_values:
            raise ValueError(f"sdc_controller must be one of {controller_options}. Got: {self.sdc_controller_str}")
        if self.non_sdc_controller_str not in controller_values:
            raise ValueError(
                f"non_sdc_controller must be one of {controller_options}. Got: {self.non_sdc_controller_str}"
            )
        if self.non_vehicle_controller_str == "auto":
            if self.non_sdc_controller_str == "idm":
                self.non_vehicle_controller_str = "replay"
            else:
                self.non_vehicle_controller_str = self.non_sdc_controller_str
        elif self.non_vehicle_controller_str not in controller_values:
            raise ValueError(
                f"non_vehicle_controller must be 'auto' or one of {controller_options}. "
                f"Got: {self.non_vehicle_controller_str}"
            )
        self.sdc_controller = controller_values[self.sdc_controller_str]
        self.non_sdc_controller = controller_values[self.non_sdc_controller_str]
        self.non_vehicle_controller = controller_values[self.non_vehicle_controller_str]

        if self.init_mode_str == "create_all_valid":
            self.init_mode = 0
        elif self.init_mode_str == "create_only_controlled":
            self.init_mode = 1
        else:
            raise ValueError(
                f"init_mode must be one of 'create_all_valid' or 'create_only_controlled'. Got: {self.init_mode_str}"
            )

        if action_type == "discrete":
            self._action_type_flag = 0
            if dynamics_model == "classic":
                # Joint action space (assume dependence)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 9])
                # Multi discrete (assume independence)
                # self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 9])
            elif dynamics_model == "jerk":
                # Joint action space (assume dependence) - 4 longitudinal × 3 lateral = 12
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
            else:
                raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self._action_type_flag = 1
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"action_space must be 'discrete' or 'continuous'. Got: {action_type}")

        # Check if resources directory exists
        if not self.map_files:
            raise FileNotFoundError(
                f"No .bin files found in {map_dir}. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )

        # Check maps availability
        available_maps = len(self.map_files)
        if num_maps > available_maps:
            raise ValueError(f"num_maps ({num_maps}) exceeds available maps in {map_dir} ({available_maps}).")
        self.starting_map_counter = starting_map
        self.starting_map_counter_init = starting_map

        # Calculate dynamic batch size for Eval + Replay mode
        self.current_num_eval_scenarios = self.num_eval_scenarios
        if self.eval_mode:
            self.current_num_eval_scenarios = min(
                self.num_eval_scenarios,
                self.num_eval_scenarios + self.starting_map_counter_init - self.starting_map_counter,
            )

        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs = binding.shared(
            map_files=self.map_files,
            num_agents=num_agents,
            num_maps=num_maps,
            starting_map_counter=self.starting_map_counter,
            eval_mode=self.eval_mode,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            sdc_controller=self.sdc_controller,
            non_sdc_controller=self.non_sdc_controller,
            non_vehicle_controller=self.non_vehicle_controller,
            simulation_mode=self.simulation_mode,
            init_step=self.init_step,
            seed=self.random_seed,
            min_agents_per_env=self.min_agents_per_env,
            max_agents_per_env=self.max_agents_per_env,
            num_eval_scenarios=self.current_num_eval_scenarios,
            goal_radius=self.goal_radius,
        )
        # In eval mode, don't wrap counter - allows termination condition to work correctly
        self.starting_map_counter = self.starting_map_counter + num_envs

        self.num_agents = num_agents
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        super().__init__(buf=buf)
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            env_id = binding.env_init(
                self.observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                self.masks[cur:nxt],
                self.random_seed,
                **self._env_init_kwargs(self.map_files[map_ids[i]], nxt - cur),
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)
        binding.vec_reset(self.c_envs, self.random_seed)

    def _env_init_kwargs(self, map_file, max_agents):
        # render_mode_flag: 0 = live viewer (RENDER_WINDOW), 1 = headless batch
        # recorder (RENDER_HEADLESS). The C side only distinguishes these two;
        # Python's render_mode = "rgb_array" / "human" / None map to the viewer
        # path, and only "headless" / "record" flip to RENDER_HEADLESS.
        if self.render_mode in ("headless", "record", "rgb_array_headless"):
            render_mode_flag = 1
        else:
            render_mode_flag = 0
        return {
            "render_mode": render_mode_flag,
            "action_type": self._action_type_flag,
            "dynamics_model": self.dynamics_model_flag,
            "human_agent_idx": self.human_agent_idx,
            "reward_goal": self.reward_goal,
            "reward_collision": self.reward_collision,
            "reward_offroad": self.reward_offroad,
            "reward_comfort": self.reward_comfort,
            "reward_lane_align": self.reward_lane_align,
            "reward_vel_align": self.reward_vel_align,
            "reward_lane_center": self.reward_lane_center,
            "reward_center_bias": self.reward_center_bias,
            "reward_velocity": self.reward_velocity,
            "reward_reverse": self.reward_reverse,
            "reward_stop_line": self.reward_stop_line,
            "reward_timestep": self.reward_timestep,
            "reward_overspeed": self.reward_overspeed,
            "reward_ade": self.reward_ade,
            "collision_behavior": self.collision_behavior,
            "offroad_behavior": self.offroad_behavior,
            "traffic_light_behavior": self.traffic_light_behavior,
            "use_map_cache": self.use_map_cache,
            "emit_completed_episodes": int(self.emit_completed_episodes),
            "goal_radius": self.goal_radius,
            "min_waypoint_spacing": self.min_waypoint_spacing,
            "max_waypoint_spacing": self.max_waypoint_spacing,
            "num_target_waypoints": self.num_target_waypoints,
            "target_type": self.target_type,
            "goal_on_lane": self.goal_on_lane,
            "obs_slots_lane_n": self.obs_slots_lane_n,
            "obs_slots_boundary_n": self.obs_slots_boundary_n,
            "obs_slots_partners_n": self.obs_slots_partners_n,
            "obs_slots_traffic_controls_n": self.obs_slots_traffic_controls_n,
            "traffic_control_scope": self.traffic_control_scope,
            "dt": self.dt,
            "spawn_initial_speed": self.spawn_initial_speed,
            "goal_speed": self.goal_speed,
            "scenario_length": int(self.scenario_length) if self.scenario_length is not None else None,
            "termination_mode": int(self.termination_mode),
            "terminate_on_goal": int(self.terminate_on_goal),
            "inactive_agent_threshold": float(self.inactive_agent_threshold),
            "map_file": map_file,
            "max_agents": max_agents,
            "max_agents_per_env": self.max_agents_per_env,
            "init_step": self.init_step,
            "init_mode": self.init_mode,
            "control_mode": self.control_mode,
            "sdc_controller": self.sdc_controller,
            "non_sdc_controller": self.non_sdc_controller,
            "non_vehicle_controller": self.non_vehicle_controller,
            "simulation_mode": self.simulation_mode,
            "reward_conditioning": self.reward_conditioning,
            "reward_randomization": self.reward_randomization,
            "compute_eval_metrics": self.compute_eval_metrics,
            "eval_mode": self.eval_mode,
            "obs_norm_goal_offset_m": self.obs_norm_goal_offset_m,
            "obs_norm_xy_offset_m": self.obs_norm_xy_offset_m,
            "obs_norm_veh_length_m": self.obs_norm_veh_length_m,
            "obs_norm_veh_width_m": self.obs_norm_veh_width_m,
            "obs_norm_road_seg_length_m": self.obs_norm_road_seg_length_m,
            "obs_norm_road_seg_width_m": self.obs_norm_road_seg_width_m,
            "obs_range_traffic_control_m": self.obs_range_traffic_control_m,
            "obs_range_partner_m": self.obs_range_partner_m,
            "obs_range_road_front_m": self.obs_range_road_front_m,
            "obs_range_road_behind_m": self.obs_range_road_behind_m,
            "obs_range_road_side_m": self.obs_range_road_side_m,
            "obs_slots_lane_kept": self.obs_slots_lane_kept,
            "obs_slots_boundary_kept": self.obs_slots_boundary_kept,
            "partner_blindness_prob": self.partner_blindness_prob,
            "partner_blindness_trigger_prob": self.partner_blindness_trigger_prob,
            "phantom_braking_prob": self.phantom_braking_prob,
            "phantom_braking_trigger_prob": self.phantom_braking_trigger_prob,
            "phantom_braking_duration": self.phantom_braking_duration,
        }

    @property
    def random_seed(self):
        return int(self.rng.integers(0, 2**24))

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        self.truncations[:] = 0
        if self.capture_compact_replay:
            self._initialize_compact_replay_buffers()
        return self.observations, []

    def step(self, actions):
        if self.capture_compact_replay:
            self._capture_compact_replay_step()
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.emit_completed_episodes:
            completed = binding.vec_pop_completed_episodes(self.c_envs)
            if completed:
                scenarios_after = None
                if self.capture_compact_replay:
                    scenarios_after = self._normalize_scenarios(self.get_state())
                for summary in completed:
                    if not isinstance(summary, dict):
                        continue
                    tagged = dict(summary)
                    tagged["summary_type"] = "completed_episode"
                    env_slot = int(tagged.get("env_slot", 0))
                    if self.capture_compact_replay:
                        bundle = self._build_compact_replay_bundle(env_slot, tagged)
                        if bundle is not None:
                            tagged["compact_replay_bundle"] = bundle
                        if scenarios_after is not None and env_slot < len(scenarios_after):
                            self._reset_compact_replay_buffer(env_slot, scenarios_after[env_slot])
                    info.append(tagged)
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                info.append(log)
                # print(log)
        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            will_resample = 1
            if will_resample:
                # Calculate dynamic batch size for Eval + Replay mode
                self.current_num_eval_scenarios = self.num_eval_scenarios
                if self.eval_mode:
                    self.current_num_eval_scenarios = min(
                        self.num_eval_scenarios,
                        self.num_eval_scenarios + self.starting_map_counter_init - self.starting_map_counter,
                    )
                if self.current_num_eval_scenarios == 0:
                    return (self.observations, self.rewards, self.terminals, self.truncations, info)
                binding.vec_close(self.c_envs)
                agent_offsets, map_ids, num_envs = binding.shared(
                    num_agents=self.num_agents,
                    num_maps=self.num_maps,
                    starting_map_counter=self.starting_map_counter,
                    eval_mode=self.eval_mode,
                    init_mode=self.init_mode,
                    control_mode=self.control_mode,
                    sdc_controller=self.sdc_controller,
                    non_sdc_controller=self.non_sdc_controller,
                    non_vehicle_controller=self.non_vehicle_controller,
                    simulation_mode=self.simulation_mode,
                    init_step=self.init_step,
                    map_files=self.map_files,
                    seed=self.random_seed,
                    min_agents_per_env=self.min_agents_per_env,
                    max_agents_per_env=self.max_agents_per_env,
                    num_eval_scenarios=self.current_num_eval_scenarios,  # Use the dynamic size here
                    goal_radius=self.goal_radius,
                )

                # In eval mode, don't wrap counter - allows termination condition to work correctly
                self.starting_map_counter = self.starting_map_counter + num_envs
                env_ids = []
                for i in range(num_envs):
                    cur = agent_offsets[i]
                    nxt = agent_offsets[i + 1]
                    env_id = binding.env_init(
                        self.observations[cur:nxt],
                        self.actions[cur:nxt],
                        self.rewards[cur:nxt],
                        self.terminals[cur:nxt],
                        self.truncations[cur:nxt],
                        self.masks[cur:nxt],
                        self.random_seed,
                        **self._env_init_kwargs(self.map_files[map_ids[i]], nxt - cur),
                    )
                    env_ids.append(env_id)
                self.c_envs = binding.vectorize(*env_ids)

                binding.vec_reset(self.c_envs, self.random_seed)
                # Map resampling is an external reset boundary (dataset/map switch). Treat as truncation.
                self.truncations[:] = 1
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def get_global_agent_state(self):
        """Get current global state of all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'id', 'length', 'width' containing numpy arrays
            of shape (num_active_agents,)
        """
        num_agents = self.num_agents

        states = {
            "x": np.zeros(num_agents, dtype=np.float32),
            "y": np.zeros(num_agents, dtype=np.float32),
            "z": np.zeros(num_agents, dtype=np.float32),
            "heading": np.zeros(num_agents, dtype=np.float32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "length": np.zeros(num_agents, dtype=np.float32),
            "width": np.zeros(num_agents, dtype=np.float32),
        }

        binding.vec_get_global_agent_state(
            self.c_envs,
            states["x"],
            states["y"],
            states["z"],
            states["heading"],
            states["id"],
            states["length"],
            states["width"],
        )

        return states

    def get_ground_truth_trajectories(self):
        """Get ground truth trajectories for all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'valid', 'id', 'scenario_id' containing numpy arrays.
        """
        num_agents = self.num_agents

        trajectories = {
            "x": np.zeros((num_agents, self.scenario_length - self.init_step), dtype=np.float32),
            "y": np.zeros((num_agents, self.scenario_length - self.init_step), dtype=np.float32),
            "z": np.zeros((num_agents, self.scenario_length - self.init_step), dtype=np.float32),
            "heading": np.zeros((num_agents, self.scenario_length - self.init_step), dtype=np.float32),
            "valid": np.zeros((num_agents, self.scenario_length - self.init_step), dtype=np.int32),
            "id": np.zeros(num_agents, dtype=np.int32),
            "scenario_id": np.zeros(num_agents, dtype=np.int32),
        }

        binding.vec_get_global_ground_truth_trajectories(
            self.c_envs,
            trajectories["x"],
            trajectories["y"],
            trajectories["z"],
            trajectories["heading"],
            trajectories["valid"],
            trajectories["id"],
            trajectories["scenario_id"],
        )

        for key in trajectories:
            trajectories[key] = trajectories[key][:, None]

        return trajectories

    def get_road_edge_polylines(self):
        """Get road edge polylines for all scenarios.

        Returns:
            dict with keys 'x', 'y', 'lengths', 'scenario_id' containing numpy arrays.
            x, y are flattened point coordinates; lengths indicates points per polyline.
        """
        num_polylines, total_points = binding.vec_get_road_edge_counts(self.c_envs)

        polylines = {
            "x": np.zeros(total_points, dtype=np.float32),
            "y": np.zeros(total_points, dtype=np.float32),
            "lengths": np.zeros(num_polylines, dtype=np.int32),
            "scenario_id": np.zeros(num_polylines, dtype=np.int32),
        }

        binding.vec_get_road_edge_polylines(
            self.c_envs,
            polylines["x"],
            polylines["y"],
            polylines["lengths"],
            polylines["scenario_id"],
        )

        return polylines

    def render(self, env_idx=0, view_mode=0):
        # view_mode: 0=default fixed perspective, 1=BEV ego-centered ortho.
        # See VIEW_MODE_* defines in pufferlib/ocean/drive/render.h.
        binding.vec_render(self.c_envs, view_mode, env_idx)

    def set_video_suffix(self, suffix, env_idx=0):
        # Append `suffix` to the next mp4 filename for the given env.
        # Must be called BEFORE the first render of a rollout because
        # make_client reads env->video_suffix when forking ffmpeg.
        binding.vec_set_video_suffix(self.c_envs, suffix, env_idx)

    def close_client(self, env_idx=0):
        # Tear down the render Client for one env without destroying the env.
        # Flushes ffmpeg + PBOs on the headless path so the mp4 is fully written.
        binding.vec_close_client(self.c_envs, env_idx)

    # ====== Compact-replay capture (active when capture_compact_replay=True) ======

    def _normalize_scenarios(self, state):
        if isinstance(state, list):
            return state
        if isinstance(state, dict):
            return [state]
        return []

    def _build_compact_replay_metadata(self, env_idx, scenario):
        map_idx = self.map_ids[env_idx] if env_idx < len(self.map_ids) else 0
        map_path = self.map_files[map_idx] if map_idx < len(self.map_files) else None
        raw_map_name = scenario.get("map_name") or map_path
        if isinstance(raw_map_name, str):
            map_name = os.path.basename(raw_map_name).split(".")[0]
        else:
            map_name = raw_map_name
        return {
            "map_name": map_name,
            "map_path": map_path,
            "scenario_id": scenario.get("scenario_id"),
            "dynamics_model": self.dynamics_model,
        }

    def _create_compact_replay_buffer(self, env_idx, scenario):
        agents = scenario.get("agents", []) or []
        traffic_elements = scenario.get("traffic_elements", []) or []
        return {
            "metadata": self._build_compact_replay_metadata(env_idx, scenario),
            "agent_capacity": len(agents),
            "traffic_capacity": len(traffic_elements),
            "agent_frames": {
                k: []
                for k in (
                    "valid",
                    "id",
                    "type",
                    "active",
                    "stopped",
                    "x",
                    "y",
                    "z",
                    "heading",
                    "length",
                    "width",
                    "goal_x",
                    "goal_y",
                )
            },
            "traffic_frames": {k: [] for k in ("valid", "type", "state", "stop_line")},
        }

    def _initialize_compact_replay_buffers(self):
        scenarios = self._normalize_scenarios(self.get_state())
        self._compact_replay_buffers = [self._create_compact_replay_buffer(i, s) for i, s in enumerate(scenarios)]

    def _extract_compact_agents_frame(self, scenario, capacity):
        valid = np.zeros(capacity, dtype=np.bool_)
        agent_id = np.full(capacity, -1, dtype=np.int32)
        agent_type = np.zeros(capacity, dtype=np.int16)
        active = np.zeros(capacity, dtype=np.bool_)
        stopped = np.zeros(capacity, dtype=np.bool_)
        x = np.zeros(capacity, dtype=np.float32)
        y = np.zeros(capacity, dtype=np.float32)
        z = np.zeros(capacity, dtype=np.float32)
        heading = np.zeros(capacity, dtype=np.float32)
        length = np.zeros(capacity, dtype=np.float32)
        width = np.zeros(capacity, dtype=np.float32)
        goal_x = np.zeros(capacity, dtype=np.float32)
        goal_y = np.zeros(capacity, dtype=np.float32)
        active_indices = set(scenario.get("active_agent_indices") or [])
        for idx, agent in enumerate(scenario.get("agents") or []):
            if idx >= capacity:
                break
            if not agent.get("sim_valid"):
                continue
            valid[idx] = True
            agent_id[idx] = int(agent.get("id", idx))
            agent_type[idx] = int(agent.get("type", 1))
            active[idx] = idx in active_indices
            stopped[idx] = bool(agent.get("stopped", False))
            x[idx] = np.float32(agent.get("sim_x", 0.0))
            y[idx] = np.float32(agent.get("sim_y", 0.0))
            z[idx] = np.float32(agent.get("sim_z", 0.0))
            heading[idx] = np.float32(agent.get("sim_heading", 0.0))
            length[idx] = np.float32(agent.get("sim_length", 0.0))
            width[idx] = np.float32(agent.get("sim_width", 0.0))
            goal_x[idx] = np.float32(agent.get("goal_position_x", 0.0))
            goal_y[idx] = np.float32(agent.get("goal_position_y", 0.0))
        return {
            "valid": valid,
            "id": agent_id,
            "type": agent_type,
            "active": active,
            "stopped": stopped,
            "x": x,
            "y": y,
            "z": z,
            "heading": heading,
            "length": length,
            "width": width,
            "goal_x": goal_x,
            "goal_y": goal_y,
        }

    def _extract_compact_traffic_frame(self, scenario, timestep, capacity):
        valid = np.zeros(capacity, dtype=np.bool_)
        control_type = np.zeros(capacity, dtype=np.int16)
        state = np.zeros(capacity, dtype=np.int16)
        stop_line = np.zeros((capacity, 6), dtype=np.float32)
        for idx, elem in enumerate(scenario.get("traffic_elements") or []):
            if idx >= capacity:
                break
            if not isinstance(elem, dict):
                continue
            raw_stop_line = elem.get("stop_line")
            if raw_stop_line is None or len(raw_stop_line) < 6:
                continue
            valid[idx] = True
            control_type[idx] = int(elem.get("type", 0))
            stop_line[idx, :] = np.asarray(raw_stop_line[:6], dtype=np.float32)
            states = elem.get("states") or []
            if states and len(states) > timestep:
                state[idx] = int(states[timestep])
        return {"valid": valid, "type": control_type, "state": state, "stop_line": stop_line}

    def _capture_compact_replay_step(self):
        scenarios = self._normalize_scenarios(self.get_state())
        if len(self._compact_replay_buffers) != len(scenarios):
            self._initialize_compact_replay_buffers()
        for env_idx, scenario in enumerate(scenarios):
            buffer = self._compact_replay_buffers[env_idx]
            episode_timestep = int(scenario.get("episode_timestep", self.tick) or 0)
            agent_frame = self._extract_compact_agents_frame(scenario, buffer["agent_capacity"])
            traffic_frame = self._extract_compact_traffic_frame(scenario, episode_timestep, buffer["traffic_capacity"])
            for k, v in agent_frame.items():
                buffer["agent_frames"][k].append(v)
            for k, v in traffic_frame.items():
                buffer["traffic_frames"][k].append(v)

    def _stack_compact_replay_frames(self, frames_dict):
        stacked = {}
        for k, frames in frames_dict.items():
            if frames:
                stacked[k] = np.stack(frames, axis=0)
        return stacked

    def _build_compact_replay_bundle(self, env_slot, summary):
        if env_slot < 0 or env_slot >= len(self._compact_replay_buffers):
            return None
        buffer = self._compact_replay_buffers[env_slot]
        if not buffer["agent_frames"]["valid"]:
            return None
        metadata = dict(buffer["metadata"])
        metadata.update(
            {
                "episode_index": int(summary.get("episode_index", 0) or 0),
                "episode_length": int(summary.get("episode_length", len(buffer["agent_frames"]["valid"]))),
                "episode_return": float(summary.get("episode_return", 0.0) or 0.0),
                "collision_rate": float(summary.get("collision_rate", 0.0) or 0.0),
                "offroad_rate": float(summary.get("offroad_rate", 0.0) or 0.0),
                "red_light_violation_rate": float(summary.get("red_light_violation_rate", 0.0) or 0.0),
                "num_goals_reached": float(summary.get("num_goals_reached", 0.0) or 0.0),
            }
        )
        bundle = {
            "schema_version": 2,
            "metadata": metadata,
            "agent_arrays": self._stack_compact_replay_frames(buffer["agent_frames"]),
            "traffic_arrays": self._stack_compact_replay_frames(buffer["traffic_frames"]),
        }
        return zlib.compress(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL), level=3)

    def _reset_compact_replay_buffer(self, env_idx, scenario):
        if env_idx < 0 or env_idx >= len(self._compact_replay_buffers):
            return
        self._compact_replay_buffers[env_idx] = self._create_compact_replay_buffer(env_idx, scenario)

    def close(self):
        binding.vec_close(self.c_envs)

    def get_state(self):
        try:
            return binding.vec_get(self.c_envs)
        except Exception:
            return binding.env_get(self.c_envs)

    def get_obs_html_frame(self, agent_f32, agent_i32, metrics_f32, puffer_f32, traffic_i16):
        binding.vec_get_obs_html_frame(
            self.c_envs,
            agent_f32,
            agent_i32,
            metrics_f32,
            puffer_f32,
            traffic_i16,
        )


def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def simplify_polyline(geometry, polyline_reduction_threshold):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)

            if area < polyline_reduction_threshold:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def save_map_binary(map_data, output_file):
    trajectory_length = 91
    """Saves map data in a binary format readable by C"""
    with open(output_file, "wb") as f:
        # Count total entities
        print(len(map_data.get("objects", [])))
        print(len(map_data.get("roads", [])))
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get("objects", []):
            # Write base entity data
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))  # type
            # f.write(struct.pack("i", obj.get("id", 0)))  # id
            f.write(struct.pack("i", trajectory_length))  # array_size
            # Write position arrays
            positions = obj.get("position", [])
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("x", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("y", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("z", 0.0))))

            # Write velocity arrays
            velocities = obj.get("velocity", [])
            for arr, key in [(velocities, "x"), (velocities, "y"), (velocities, "z")]:
                for i in range(trajectory_length):
                    vel = arr[i] if i < len(arr) else {"x": 0.0, "y": 0.0, "z": 0.0}
                    f.write(struct.pack("f", float(vel.get(key, 0.0))))

            # Write heading and valid arrays
            headings = obj.get("heading", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}f",
                    *[float(headings[i]) if i < len(headings) else 0.0 for i in range(trajectory_length)],
                )
            )

            valids = obj.get("valid", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}i",
                    *[int(valids[i]) if i < len(valids) else 0 for i in range(trajectory_length)],
                )
            )

            # Write scalar fields
            f.write(struct.pack("f", float(obj.get("width", 0.0))))
            f.write(struct.pack("f", float(obj.get("length", 0.0))))
            f.write(struct.pack("f", float(obj.get("height", 0.0))))
            goal_pos = obj.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", obj.get("mark_as_expert", 0)))

        # Write roads
        for idx, road in enumerate(map_data.get("roads", [])):
            geometry = road.get("geometry", [])
            road_type = road.get("map_element_id", 0)
            road_type_word = road.get("type", 0)
            if road_type_word == "lane":
                road_type = 2
            elif road_type_word == "road_edge":
                road_type = 15
            # breakpoint()
            if len(geometry) > 10 and road_type <= 16:
                geometry = simplify_polyline(geometry, 0.1)
            size = len(geometry)
            # breakpoint()
            if road_type >= 0 and road_type <= 3:
                road_type = 4
            elif road_type >= 5 and road_type <= 13:
                road_type = 5
            elif road_type >= 14 and road_type <= 16:
                road_type = 6
            elif road_type == 17:
                road_type = 7
            elif road_type == 18:
                road_type = 8
            elif road_type == 19:
                road_type = 9
            elif road_type == 20:
                road_type = 10
            # Write base entity data
            f.write(struct.pack("i", road_type))  # type
            # f.write(struct.pack("i", road.get("id", 0)))  # id
            f.write(struct.pack("i", size))  # array_size

            # Write position arrays
            for coord in ["x", "y", "z"]:
                for point in geometry:
                    f.write(struct.pack("f", float(point.get(coord, 0.0))))
            # Write scalar fields
            f.write(struct.pack("f", float(road.get("width", 0.0))))
            f.write(struct.pack("f", float(road.get("length", 0.0))))
            f.write(struct.pack("f", float(road.get("height", 0.0))))
            goal_pos = road.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", road.get("mark_as_expert", 0)))


def load_map(map_name, binary_output=None):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, "r") as f:
        map_data = json.load(f)

    if binary_output:
        save_map_binary(map_data, binary_output)


def process_all_maps(dataset_path: str, max_file_to_process: int = 1000):
    """Process all maps from a local path (or GCS) and save them as binaries."""
    # Create the binaries directory if it doesn't exist
    binary_dir = Path("pufferlib/resources/drive/binaries")
    binary_dir.mkdir(parents=True, exist_ok=True)

    # --- GCS FUSE ---
    if dataset_path.startswith("gs://") and os.path.exists("/gcs/"):
        print("Vertex AI GCS FUSE mount detected. Translating GCS URI to local path.")
        dataset_path = dataset_path.replace("gs://", "/gcs/")
        print(f"Using mounted dataset path: {dataset_path}")

    file_iterator = None
    fs = None  # Will hold the gcsfs filesystem object if needed

    path = Path(dataset_path)
    print(f"Searching for JSON map files in local path: {path.resolve()}")
    # Use rglob for recursive globbing to match the GCS '**' behavior
    file_iterator = sorted(path.rglob("*.json"))
    print(f"Found {len(file_iterator)} JSON files locally.")

    file_count = 0
    # Process each JSON file from the appropriate source
    for i, item in enumerate(file_iterator):
        if i >= max_file_to_process:
            print(f"Reached file limit of {max_file_to_process}.")
            break

        map_path_str = ""
        try:
            # if is_gcs_stream:
            #     # item is a path string from gcsfs.glob, e.g., "my-bucket/path/file.json"
            #     map_path_str = f"gs://{item}"
            #     # Use 'with' to ensure the stream is automatically closed
            #     with fs.open(item, "rt", encoding="utf-8") as stream:
            #         map_data = json.load(stream)
            # else:
            # item is a Path object from Path.rglob
            map_path_str = str(item)
            # Use 'with' for local files too (good practice)
            with open(map_path_str, "r") as f:
                map_data = json.load(f)

            map_name = Path(map_path_str).name
            binary_file = f"map_{i:03d}.bin"
            binary_path = binary_dir / binary_file

            print(f"Processing {map_name} -> {binary_file}")
            save_map_binary(map_data, str(binary_path))
            file_count += 1

        except Exception as e:
            print(f"Error processing {map_path_str}: {e}")
            continue

    print(f"Found and processed {file_count} JSON files.")


def test_performance(timeout=10, atn_cache=1024, num_agents=1024):
    import time

    env = Drive(num_agents=num_agents)
    env.reset()
    tick = 0
    num_agents = 1024
    actions = np.stack(
        [np.random.randint(0, space.n + 1, (atn_cache, num_agents)) for space in env.single_action_space], axis=-1
    )

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {num_agents * tick / (time.time() - start)}")
    env.close()


if __name__ == "__main__":
    # test_performance()
    parser = argparse.ArgumentParser(description="Process maps for PufferDrive.")
    parser.add_argument(
        "--data_dir", type=str, default="data/train", help="Path to the directory containing JSON map files."
    )
    args = parser.parse_args()
    process_all_maps(args.data_dir)
