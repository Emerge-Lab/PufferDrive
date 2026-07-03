import argparse
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
        inactive_agent_threshold=0.4,
        buf=None,
        seed=1,
        scenario_seed=None,
        eval_map_indices=None,
        eval_scenario_seeds=None,
        init_step=0,
        eval_mode=0,
        num_eval_scenarios=16,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        replay_expert_actions=False,
        map_dir=None,
        target_type="static",
        reward_conditioning=False,
        reward_randomization=False,
        compute_eval_metrics=True,
        shared_network=True,
        obs_slots_lane_n=32,
        obs_slots_boundary_n=32,
        obs_lane_stride=1,
        obs_boundary_stride=1,
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
        self.tick = 0
        self.spawn_initial_speed = float(spawn_initial_speed)
        self.goal_speed = float(goal_speed)
        self.reward_conditioning = reward_conditioning
        self.reward_randomization = reward_randomization
        self.compute_eval_metrics = compute_eval_metrics
        self.shared_network = shared_network
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
        if num_target_waypoints > binding.MAX_TARGET_POINTS:
            num_target_waypoints = binding.MAX_TARGET_POINTS
        self.num_target_waypoints = num_target_waypoints
        self.target_type_str = target_type
        if target_type == "static":
            self.target_type = binding.TARGET_STATIC
        elif target_type == "dynamic":
            self.target_type = binding.TARGET_DYNAMIC
        else:
            raise ValueError(f"target_type must be 'static' or 'dynamic'. Got: {target_type}")
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.traffic_light_behavior = traffic_light_behavior
        self.human_agent_idx = human_agent_idx
        self.scenario_length = scenario_length
        self.resample_frequency = resample_frequency
        if use_map_cache not in (0, 1):
            raise ValueError(f"use_map_cache must be 0 (off) or 1 (on). Got: {use_map_cache}")
        self.use_map_cache = use_map_cache
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
        self.inactive_agent_threshold = inactive_agent_threshold
        self.rng = np.random.default_rng(seed)
        self.scenario_seed = None if scenario_seed in (None, "None") else int(scenario_seed)
        self.eval_map_indices = self._int_list_or_none(eval_map_indices)
        self.eval_scenario_seeds = self._int_list_or_none(eval_scenario_seeds)
        if self.eval_map_indices is not None:
            expected = len(self.eval_map_indices)
            if self.eval_scenario_seeds is None or len(self.eval_scenario_seeds) != expected:
                raise ValueError("eval_scenario_seeds must have one seed per eval_map_indices entry")
        self.min_agents_per_env = min_agents_per_env
        self.max_agents_per_env = max_agents_per_env

        self.ego_features = binding.EGO_FEATURES

        # Extract observation shapes from constants
        obs_lane_stride = int(obs_lane_stride)
        obs_boundary_stride = int(obs_boundary_stride)
        if obs_lane_stride < 1:
            raise ValueError(f"obs_lane_stride must be >= 1. Got: {obs_lane_stride}")
        if obs_boundary_stride < 1:
            raise ValueError(f"obs_boundary_stride must be >= 1. Got: {obs_boundary_stride}")
        self.obs_slots_lane_n = obs_slots_lane_n
        self.obs_slots_boundary_n = obs_slots_boundary_n
        self.obs_lane_stride = obs_lane_stride
        self.obs_boundary_stride = obs_boundary_stride
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
        self.obs_valid_count_features = binding.OBS_VALID_COUNT_FEATURES
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
            + self.obs_valid_count_features
        )

        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)

        self.init_step = init_step
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        self.replay_expert_actions = bool(replay_expert_actions)
        self.simulation_mode_str = simulation_mode
        self.map_dir = map_dir
        self.map_files = sorted(os.path.join(map_dir, f) for f in os.listdir(map_dir) if f.endswith(".bin"))

        if self.simulation_mode_str == "gigaflow":
            self.simulation_mode = 0
        elif self.simulation_mode_str == "replay":
            self.simulation_mode = 1
        else:
            raise ValueError(f"simulation_mode must be one of 'gigaflow' or 'replay'. Got: {self.simulation_mode_str}")
        if self.replay_expert_actions and self.simulation_mode_str != "replay":
            raise ValueError("replay_expert_actions requires simulation_mode='replay'")

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
                f"control_mode must be one of 'control_vehicles', 'control_wosac', or 'control_agents'. Got: {self.control_mode_str}"
            )
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
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 9])
            elif dynamics_model == "jerk":
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
        self.num_agents = num_agents
        base_seed = self._allocate_env_batch()
        super().__init__(buf=buf)
        self._build_env_batch(base_seed)

    def _env_init_kwargs(self, map_file, max_agents):
        return {
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
            "goal_radius": self.goal_radius,
            "min_waypoint_spacing": self.min_waypoint_spacing,
            "max_waypoint_spacing": self.max_waypoint_spacing,
            "num_target_waypoints": self.num_target_waypoints,
            "target_type": self.target_type,
            "obs_slots_lane_n": self.obs_slots_lane_n,
            "obs_slots_boundary_n": self.obs_slots_boundary_n,
            "obs_lane_stride": self.obs_lane_stride,
            "obs_boundary_stride": self.obs_boundary_stride,
            "obs_slots_partners_n": self.obs_slots_partners_n,
            "obs_slots_traffic_controls_n": self.obs_slots_traffic_controls_n,
            "traffic_control_scope": self.traffic_control_scope,
            "dt": self.dt,
            "spawn_initial_speed": self.spawn_initial_speed,
            "goal_speed": self.goal_speed,
            "scenario_length": int(self.scenario_length) if self.scenario_length is not None else None,
            "termination_mode": int(self.termination_mode),
            "inactive_agent_threshold": float(self.inactive_agent_threshold),
            "map_file": map_file,
            "max_agents": max_agents,
            "max_agents_per_env": self.max_agents_per_env,
            "init_step": self.init_step,
            "init_mode": self.init_mode,
            "control_mode": self.control_mode,
            "simulation_mode": self.simulation_mode,
            "replay_expert_actions": self.replay_expert_actions,
            "reward_conditioning": self.reward_conditioning,
            "reward_randomization": self.reward_randomization,
            "use_exact_episode_seed": (
                bool(self.eval_mode) and (self.scenario_seed is not None or self.eval_scenario_seeds is not None)
            ),
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

    @staticmethod
    def _int_list_or_none(values):
        if values is None or values == "None":
            return None
        return [int(value) for value in values]

    def _episode_seed_args(self, base_seed):
        if self.eval_scenario_seeds is not None:
            seeds = self.eval_scenario_seeds[: self.num_envs]
            return seeds, seeds
        if self.eval_mode or self.scenario_seed is not None:
            return [base_seed + i for i in range(self.num_envs)], base_seed
        return [self.random_seed for _ in range(self.num_envs)], self.random_seed

    def _allocate_env_batch(self):
        base_seed = self.scenario_seed if self.scenario_seed is not None else self.random_seed
        batch_start = self.starting_map_counter
        agent_offsets, map_ids, num_envs = binding.shared(
            map_files=self.map_files,
            num_agents=self.num_agents,
            num_maps=self.num_maps,
            starting_map_counter=self.starting_map_counter,
            eval_mode=self.eval_mode,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            simulation_mode=self.simulation_mode,
            init_step=self.init_step,
            seed=base_seed,
            min_agents_per_env=self.min_agents_per_env,
            max_agents_per_env=self.max_agents_per_env,
            num_eval_scenarios=self.current_num_eval_scenarios,
            eval_map_indices=self.eval_map_indices,
        )
        # In eval mode, don't wrap counter - allows termination condition to work correctly
        self.starting_map_counter = self.starting_map_counter + num_envs
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.scenario_indices = [batch_start + i for i in range(num_envs)]
        self.num_envs = num_envs
        return base_seed

    def _build_env_batch(self, base_seed):
        env_seeds, reset_seed = self._episode_seed_args(base_seed)
        env_ids = []
        for i in range(self.num_envs):
            cur = self.agent_offsets[i]
            nxt = self.agent_offsets[i + 1]
            env_id = binding.env_init(
                self.observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                self.masks[cur:nxt],
                env_seeds[i],
                **self._env_init_kwargs(self.map_files[self.map_ids[i]], nxt - cur),
            )
            env_ids.append(env_id)
        self.c_envs = binding.vectorize(*env_ids)
        self._reset_envs(reset_seed)

    def _reset_envs(self, reset_seed):
        binding.vec_reset(self.c_envs, reset_seed)
        self.last_reset_seed = reset_seed

    def reset(self, seed=0):
        if self.eval_scenario_seeds is not None or self.scenario_seed is not None:
            _, reset_seed = self._episode_seed_args(self.scenario_seed)
        elif self.tick == 0 and self.last_reset_seed is not None:
            # First reset after construction: reuse the base that built & logged this batch.
            reset_seed = self.last_reset_seed
        else:
            reset_seed = seed

        self._reset_envs(reset_seed)
        self.tick = 0
        self.truncations[:] = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                if isinstance(log, list):
                    for env_idx, summary in enumerate(log):
                        # "Seed" is provided by the C env (my_log) as the exact per-episode RNG
                        # seed that produced these metrics; do not overwrite it here.
                        summary["map_index"] = int(self.map_ids[env_idx])
                        summary["scenario_index"] = int(self.scenario_indices[env_idx])
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
                base_seed = self._allocate_env_batch()
                self._build_env_batch(base_seed)
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

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

    def get_state(self):
        try:
            return binding.vec_get(self.c_envs)
        except Exception:
            return binding.env_get(self.c_envs)


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
