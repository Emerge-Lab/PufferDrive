import pickle
import zlib
import numpy as np
import gymnasium
import os
from importlib.resources import files as package_files
import pufferlib
from pufferlib.ocean.drive import binding


def map_dir_missing_message(map_dir):
    """Error text for a nonexistent map_dir. When its basename is a dataset
    registered in data_utils/datasets.yaml, the message names the exact fetch
    command instead of leaving the user with a bare missing-path error."""
    message = f"map_dir '{map_dir}' does not exist."
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    manifest_path = os.path.join(repo_root, "data_utils", "datasets.yaml")
    dataset_name = os.path.basename(os.path.normpath(str(map_dir)))
    if not os.path.isfile(manifest_path):
        return message
    with open(manifest_path) as f:
        is_registered_dataset = any(line.startswith(f"{dataset_name}:") for line in f)
    if is_registered_dataset:
        message += (
            f" It is a fetchable dataset:\n"
            f"    python data_utils/fetch_data.py {dataset_name}\n"
            f"Run from the repo root, or point map_dir at wherever you fetched it"
            f" — see docs/data_storage.md."
        )
    return message


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
        min_goal_spacing=20.0,
        max_goal_spacing=60.0,
        goal_heading_max_deg=0.0,
        num_goals=3,
        goal_radius=2.0,
        collision_behavior="ignore",
        offroad_behavior="ignore",
        traffic_light_behavior="ignore",
        disable_red_light_infractions=0,
        traffic_light_junction_phases=0,
        use_map_cache=0,
        use_neighbor_cache=1,
        capture_replay=False,
        replay_worker_idx=0,
        dt=0.1,
        base_max_speed_mps=20.0,
        max_speed_mps=None,
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
        reset_accel_on_stop=False,
        simulation_mode="gigaflow",
        termination_mode=0,
        inactive_agent_threshold=0.4,
        terminate_on_goal=False,
        buf=None,
        seed=1,
        init_step=0,
        init_step_spread=False,
        init_step_min_horizon=20,
        eval_mode=0,
        num_eval_scenarios=16,
        max_scenarios_per_batch=None,
        eval_map_indices=None,
        eval_scenario_seeds=None,
        eval_training_render=False,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        sdc_controller="policy",
        non_sdc_controller="policy",
        non_vehicle_controller="auto",
        replay_expert_agents=1,
        map_dir=None,
        goal_regen_mode="finite",
        goal_source="route",
        obs_goal_lane_distance=False,
        reward_conditioning=False,
        reward_randomization=False,
        reward_log_sampling=False,
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
        obs_norm_speed_mps=60.0,
        obs_norm_goal_offset_m=100.0,
        obs_norm_xy_offset_m=100.0,
        obs_norm_veh_length_m=15.0,
        obs_norm_veh_width_m=10.0,
        obs_norm_road_seg_length_m=5.0,
        obs_norm_road_seg_width_m=5.0,
        obs_norm_z_m=10.0,
        eval_perceived_size_margin_m=0.1,
        eval_standstill_jerk_deadband_mps3=0.0,
        obs_range_traffic_control_m=100.0,
        obs_range_partner_m=100.0,
        obs_range_road_front_m=120.0,
        obs_range_road_behind_m=20.0,
        obs_range_road_side_m=30.0,
        obs_dropout_lane=0.0,
        obs_dropout_boundary=0.0,
        partner_blindness_prob=0.0,
        partner_blindness_trigger_prob=0.1,
        partner_blindness_duration_seconds=1.0,
        phantom_braking_prob=0.0,
        phantom_braking_trigger_prob=0.0,
        phantom_braking_duration_seconds=1.0,
        phantom_braking_freeze_steering=True,
    ):
        self.dt = dt
        self.base_max_speed_mps = float(base_max_speed_mps)
        self.max_speed_mps = self.base_max_speed_mps if max_speed_mps is None else float(max_speed_mps)
        self.spawn_initial_speed = float(spawn_initial_speed)
        self.goal_speed = float(goal_speed)
        if reward_randomization and not reward_conditioning:
            raise ValueError("reward_randomization requires reward_conditioning")
        self.reward_conditioning = reward_conditioning
        self.reward_randomization = reward_randomization
        self.reward_log_sampling = reward_log_sampling
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
        self.min_goal_spacing = min_goal_spacing
        self.max_goal_spacing = max_goal_spacing
        self.goal_heading_max_deg = goal_heading_max_deg
        if not 1 <= num_goals <= binding.MAX_GOALS:
            raise ValueError(f"num_goals must be in [1, {binding.MAX_GOALS}]. Got: {num_goals}")
        self.num_goals = num_goals
        if goal_regen_mode == "finite":
            self.goal_regen_mode = binding.GOAL_REGEN_FINITE
        elif goal_regen_mode == "rolling":
            self.goal_regen_mode = binding.GOAL_REGEN_ROLLING
        else:
            raise ValueError(f"goal_regen_mode must be 'finite' or 'rolling'. Got: {goal_regen_mode}")
        if goal_source == "route":
            self.goal_source = binding.GOAL_SOURCE_ROUTE
        elif goal_source == "map":
            self.goal_source = binding.GOAL_SOURCE_MAP
        elif goal_source == "gt":
            self.goal_source = binding.GOAL_SOURCE_GT
        else:
            raise ValueError(f"goal_source must be 'route', 'map', or 'gt'. Got: {goal_source}")
        self.obs_goal_lane_distance = int(bool(obs_goal_lane_distance))
        infraction_behavior_values = {
            "ignore": binding.INFRACTION_BEHAVIOR_IGNORE,
            "stop": binding.INFRACTION_BEHAVIOR_STOP,
            "remove": binding.INFRACTION_BEHAVIOR_REMOVE,
        }
        for behavior_name, behavior in (
            ("collision_behavior", collision_behavior),
            ("offroad_behavior", offroad_behavior),
            ("traffic_light_behavior", traffic_light_behavior),
        ):
            if behavior not in infraction_behavior_values:
                raise ValueError(f"{behavior_name} must be one of 'ignore', 'stop', or 'remove'. Got: {behavior}")
        self.collision_behavior = infraction_behavior_values[collision_behavior]
        self.offroad_behavior = infraction_behavior_values[offroad_behavior]
        self.traffic_light_behavior = infraction_behavior_values[traffic_light_behavior]
        if disable_red_light_infractions not in (0, 1):
            raise ValueError(f"disable_red_light_infractions must be 0 or 1. Got: {disable_red_light_infractions}")
        self.disable_red_light_infractions = disable_red_light_infractions
        if traffic_light_junction_phases not in (0, 1):
            raise ValueError(f"traffic_light_junction_phases must be 0 or 1. Got: {traffic_light_junction_phases}")
        self.traffic_light_junction_phases = traffic_light_junction_phases
        if replay_expert_agents not in (0, 1):
            raise ValueError(f"replay_expert_agents must be 0 or 1. Got: {replay_expert_agents}")
        self.replay_expert_agents = replay_expert_agents
        if use_map_cache not in (0, 1):
            raise ValueError(f"use_map_cache must be 0 (off) or 1 (on). Got: {use_map_cache}")
        self.use_map_cache = use_map_cache
        self.capture_replay = bool(capture_replay)
        self.replay_worker_idx = replay_worker_idx
        self._replay_captures = []
        self.human_agent_idx = human_agent_idx
        self.scenario_length = scenario_length
        self.resample_frequency = resample_frequency
        if use_neighbor_cache not in (0, 1):
            raise ValueError(f"use_neighbor_cache must be 0 (off) or 1 (on). Got: {use_neighbor_cache}")
        self.use_neighbor_cache = use_neighbor_cache
        self.dynamics_model = dynamics_model
        if dynamics_model == "classic":
            self.dynamics_model_flag = binding.DYNAMICS_MODEL_CLASSIC
        elif dynamics_model == "jerk":
            self.dynamics_model_flag = binding.DYNAMICS_MODEL_JERK
        else:
            raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        self.reset_accel_on_stop = reset_accel_on_stop
        self.eval_mode = eval_mode
        self.num_eval_scenarios = num_eval_scenarios
        if max_scenarios_per_batch is not None and max_scenarios_per_batch < 1:
            raise ValueError(f"max_scenarios_per_batch must be >= 1 or None. Got: {max_scenarios_per_batch}")
        self.max_scenarios_per_batch = max_scenarios_per_batch
        self.eval_map_indices = eval_map_indices
        self.eval_scenario_seeds = eval_scenario_seeds
        if self.eval_map_indices is not None:
            if self.eval_scenario_seeds is None or len(self.eval_scenario_seeds) != len(self.eval_map_indices):
                raise ValueError("eval_scenario_seeds must have one seed per eval_map_indices entry")
        if not isinstance(eval_training_render, bool):
            raise TypeError("eval_training_render must be a boolean")
        if eval_training_render and not eval_mode:
            raise ValueError("eval_training_render requires eval_mode")
        if eval_training_render and simulation_mode != "gigaflow":
            raise ValueError("eval_training_render only supports gigaflow simulation_mode")
        if eval_training_render and num_agents < max_agents_per_env:
            raise ValueError("eval_training_render requires num_agents >= max_agents_per_env")
        self.eval_training_render = eval_training_render
        self.use_exact_episode_seed = bool(eval_mode) and self.eval_scenario_seeds is not None
        self.termination_mode = termination_mode
        self.inactive_agent_threshold = inactive_agent_threshold
        self.terminate_on_goal = terminate_on_goal
        self.rng = np.random.default_rng(seed)
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
        self.obs_norm_speed_mps = float(obs_norm_speed_mps)
        if not np.isfinite(self.obs_norm_speed_mps) or self.obs_norm_speed_mps <= 0.0:
            raise ValueError(f"obs_norm_speed_mps must be finite and > 0. Got: {obs_norm_speed_mps}")
        self.obs_norm_goal_offset_m = float(obs_norm_goal_offset_m)
        self.obs_norm_xy_offset_m = float(obs_norm_xy_offset_m)
        self.obs_norm_veh_length_m = float(obs_norm_veh_length_m)
        self.obs_norm_veh_width_m = float(obs_norm_veh_width_m)
        self.obs_norm_road_seg_length_m = float(obs_norm_road_seg_length_m)
        self.obs_norm_road_seg_width_m = float(obs_norm_road_seg_width_m)
        self.obs_norm_z_m = float(obs_norm_z_m)
        self.eval_perceived_size_margin_m = float(eval_perceived_size_margin_m)
        self.eval_standstill_jerk_deadband_mps3 = float(eval_standstill_jerk_deadband_mps3)
        if self.eval_standstill_jerk_deadband_mps3 < 0:
            raise ValueError(
                f"eval_standstill_jerk_deadband_mps3 must be >= 0. Got: {eval_standstill_jerk_deadband_mps3}"
            )
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
        self.partner_blindness_duration_seconds = float(partner_blindness_duration_seconds)
        self.phantom_braking_prob = float(phantom_braking_prob)
        self.phantom_braking_trigger_prob = float(phantom_braking_trigger_prob)
        self.phantom_braking_duration_seconds = float(phantom_braking_duration_seconds)
        self.phantom_braking_freeze_steering = int(bool(phantom_braking_freeze_steering))
        self.partner_features = binding.PARTNER_FEATURES
        self.lane_features = binding.LANE_FEATURES
        self.boundary_features = binding.BOUNDARY_FEATURES
        self.traffic_control_features = binding.TRAFFIC_CONTROL_FEATURES
        self.obs_valid_count_features = binding.OBS_VALID_COUNT_FEATURES
        self.num_reward_coefs = binding.NUM_REWARD_COEFS if reward_conditioning else 0

        # One uniform target representation (ego-frame x, y, z) regardless of goal_regen_mode.
        self.goal_features = binding.GOAL_FEATURES
        self.goal_dim = self.num_goals * self.goal_features

        # GPS goal-distance (abs + rel) columns are lane-only (LANE_FEATURES); zero-filled when flag off.
        self.num_obs = (
            self.ego_features
            + self.num_reward_coefs
            + self.goal_dim
            + self.obs_slots_partners_n * self.partner_features
            + self.obs_slots_lane_kept * self.lane_features
            + self.obs_slots_boundary_kept * self.boundary_features
            + self.obs_slots_traffic_controls_n * self.traffic_control_features
            + self.obs_valid_count_features
        )

        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)

        # Marks dims expected in [-1, 1] for obs-stat logging; excludes raw enum/count dims (one-hot/mask in the policy)
        self.normalized_obs_mask = np.ones(self.num_obs, dtype=bool)
        traffic_control_base = (
            self.num_obs
            - self.obs_valid_count_features
            - self.obs_slots_traffic_controls_n * self.traffic_control_features
        )
        for slot_idx in range(self.obs_slots_traffic_controls_n):
            slot_end = traffic_control_base + (slot_idx + 1) * self.traffic_control_features
            self.normalized_obs_mask[slot_end - 2 : slot_end] = False
        self.normalized_obs_mask[self.num_obs - self.obs_valid_count_features :] = False

        self.init_step = init_step
        # Per C environment randomized start point. When on, each parallel environment
        # starts the episode at a randomized point.
        self.init_step_spread = bool(init_step_spread)
        # limit at which we set the starting point from the end of the total episode length
        self.init_step_min_horizon = int(init_step_min_horizon)
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
            if not os.path.isdir(map_dir):
                raise FileNotFoundError(map_dir_missing_message(map_dir))
            self.map_files = sorted(os.path.join(map_dir, f) for f in os.listdir(map_dir) if f.endswith(".bin"))

        if self.simulation_mode_str == "gigaflow":
            self.simulation_mode = binding.SIMULATION_MODE_GIGAFLOW
        elif self.simulation_mode_str == "replay":
            self.simulation_mode = binding.SIMULATION_MODE_REPLAY
        else:
            raise ValueError(f"simulation_mode must be one of 'gigaflow' or 'replay'. Got: {self.simulation_mode_str}")

        if self.goal_source == binding.GOAL_SOURCE_GT and self.simulation_mode != 1:
            raise ValueError(
                "goal_source 'gt' is only supported in replay simulation_mode (it reads the logged ground-truth trajectory)."
            )

        if self.init_step_spread:
            if self.simulation_mode != binding.SIMULATION_MODE_REPLAY:
                raise ValueError(
                    "init_step_spread is only supported in replay simulation_mode (it seeds each environment at a different expert timestep)."
                )
            if self.scenario_length - self.init_step_min_horizon <= 0:
                raise ValueError(
                    f"init_step_min_horizon ({self.init_step_min_horizon}) leaves no room to sample a start in a scenario of length {self.scenario_length}; it must be < scenario_length."
                )

        if self.control_mode_str == "control_vehicles":
            self.control_mode = binding.CONTROL_MODE_VEHICLES
        elif self.control_mode_str == "control_agents":
            self.control_mode = binding.CONTROL_MODE_AGENTS
        elif self.control_mode_str == "control_wosac":
            self.control_mode = binding.CONTROL_MODE_WOSAC
        elif self.control_mode_str == "control_sdc_only":
            self.control_mode = binding.CONTROL_MODE_SDC_ONLY
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
            self.init_mode = binding.INIT_MODE_CREATE_ALL_VALID
        elif self.init_mode_str == "create_only_controlled":
            self.init_mode = binding.INIT_MODE_CREATE_ONLY_CONTROLLED
        elif self.init_mode_str == "create_controllable_types":
            self.init_mode = binding.INIT_MODE_CREATE_CONTROLLABLE_TYPES
        else:
            raise ValueError(
                "init_mode must be one of 'create_all_valid', 'create_only_controlled', or "
                f"'create_controllable_types'. Got: {self.init_mode_str}"
            )

        if action_type == "discrete":
            self._action_type_flag = binding.ACTION_TYPE_DISCRETE
            if dynamics_model == "classic":
                self.single_action_space = gymnasium.spaces.Discrete(
                    len(binding.ACCELERATION_VALUES) * len(binding.STEERING_VALUES)
                )
            elif dynamics_model == "jerk":
                self.single_action_space = gymnasium.spaces.Discrete(len(binding.JERK_LONG) * len(binding.JERK_LAT))
            else:
                raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self._action_type_flag = binding.ACTION_TYPE_CONTINUOUS
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

        self.current_num_eval_scenarios = self._next_eval_batch_size()

        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs, maps_consumed = binding.shared(
            map_files=self.map_files,
            num_agents=num_agents,
            num_maps=num_maps,
            starting_map_counter=self.starting_map_counter,
            eval_mode=self.eval_mode,
            eval_training_render=self.eval_training_render,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            sdc_controller=self.sdc_controller,
            non_sdc_controller=self.non_sdc_controller,
            non_vehicle_controller=self.non_vehicle_controller,
            replay_expert_agents=self.replay_expert_agents,
            simulation_mode=self.simulation_mode,
            init_step=self.init_step,
            seed=self.random_seed,
            min_agents_per_env=self.min_agents_per_env,
            max_agents_per_env=self.max_agents_per_env,
            num_eval_scenarios=self.current_num_eval_scenarios,
            eval_map_indices=self.eval_map_indices,
            goal_radius=self.goal_radius,
        )
        # In eval mode, don't wrap counter - allows termination condition to work correctly
        self.starting_map_counter = self.starting_map_counter + maps_consumed
        # Set once a worker has evaluated its whole map window; a frozen worker
        # stops stepping and emitting so it can't re-process or double-count.
        self._eval_exhausted = self.eval_mode and self.current_num_eval_scenarios == 0

        self.num_agents = num_agents
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        super().__init__(buf=buf)
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            env_seed = self.eval_scenario_seeds[i] if self.eval_scenario_seeds is not None else self.random_seed
            env_id = binding.env_init(
                self.observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                self.masks[cur:nxt],
                env_seed,
                **self._env_init_kwargs(self.map_files[map_ids[i]], nxt - cur),
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

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
            # Absolute directory holding render assets (.glb models), so the C
            # renderer loads them regardless of the process CWD. Derived from
            # the installed package location, not a config knob.
            "resource_root": str(package_files("pufferlib") / "resources" / "drive"),
            "action_type": self._action_type_flag,
            "dynamics_model": self.dynamics_model_flag,
            "reset_accel_on_stop": self.reset_accel_on_stop,
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
            "disable_red_light_infractions": self.disable_red_light_infractions,
            "traffic_light_junction_phases": self.traffic_light_junction_phases,
            "use_map_cache": self.use_map_cache,
            "use_neighbor_cache": self.use_neighbor_cache,
            "goal_radius": self.goal_radius,
            "min_goal_spacing": self.min_goal_spacing,
            "max_goal_spacing": self.max_goal_spacing,
            "goal_heading_max_deg": self.goal_heading_max_deg,
            "num_goals": self.num_goals,
            "goal_regen_mode": self.goal_regen_mode,
            "goal_source": self.goal_source,
            "obs_goal_lane_distance": self.obs_goal_lane_distance,
            "obs_slots_lane_n": self.obs_slots_lane_n,
            "obs_slots_boundary_n": self.obs_slots_boundary_n,
            "obs_lane_stride": self.obs_lane_stride,
            "obs_boundary_stride": self.obs_boundary_stride,
            "obs_slots_partners_n": self.obs_slots_partners_n,
            "obs_slots_traffic_controls_n": self.obs_slots_traffic_controls_n,
            "traffic_control_scope": self.traffic_control_scope,
            "dt": self.dt,
            "base_max_speed_mps": self.base_max_speed_mps,
            "max_speed_mps": self.max_speed_mps,
            "spawn_initial_speed": self.spawn_initial_speed,
            "goal_speed": self.goal_speed,
            "scenario_length": int(self.scenario_length) if self.scenario_length is not None else None,
            "termination_mode": int(self.termination_mode),
            "inactive_agent_threshold": float(self.inactive_agent_threshold),
            "terminate_on_goal": int(self.terminate_on_goal),
            "map_file": map_file,
            "max_agents": max_agents,
            "max_agents_per_env": self.max_agents_per_env,
            "init_step": self._sample_init_step(),
            "init_mode": self.init_mode,
            "control_mode": self.control_mode,
            "sdc_controller": self.sdc_controller,
            "non_sdc_controller": self.non_sdc_controller,
            "non_vehicle_controller": self.non_vehicle_controller,
            "replay_expert_agents": self.replay_expert_agents,
            "simulation_mode": self.simulation_mode,
            "reward_conditioning": self.reward_conditioning,
            "reward_randomization": self.reward_randomization,
            "reward_log_sampling": self.reward_log_sampling,
            "compute_eval_metrics": self.compute_eval_metrics,
            "eval_mode": self.eval_mode,
            "eval_training_render": self.eval_training_render,
            "use_exact_episode_seed": int(self.use_exact_episode_seed),
            "obs_norm_speed_mps": self.obs_norm_speed_mps,
            "obs_norm_goal_offset_m": self.obs_norm_goal_offset_m,
            "obs_norm_xy_offset_m": self.obs_norm_xy_offset_m,
            "obs_norm_veh_length_m": self.obs_norm_veh_length_m,
            "obs_norm_veh_width_m": self.obs_norm_veh_width_m,
            "obs_norm_road_seg_length_m": self.obs_norm_road_seg_length_m,
            "obs_norm_road_seg_width_m": self.obs_norm_road_seg_width_m,
            "obs_norm_z_m": self.obs_norm_z_m,
            "eval_perceived_size_margin_m": self.eval_perceived_size_margin_m,
            "eval_standstill_jerk_deadband_mps3": self.eval_standstill_jerk_deadband_mps3,
            "obs_range_traffic_control_m": self.obs_range_traffic_control_m,
            "obs_range_partner_m": self.obs_range_partner_m,
            "obs_range_road_front_m": self.obs_range_road_front_m,
            "obs_range_road_behind_m": self.obs_range_road_behind_m,
            "obs_range_road_side_m": self.obs_range_road_side_m,
            "obs_slots_lane_kept": self.obs_slots_lane_kept,
            "obs_slots_boundary_kept": self.obs_slots_boundary_kept,
            "partner_blindness_prob": self.partner_blindness_prob,
            "partner_blindness_trigger_prob": self.partner_blindness_trigger_prob,
            "partner_blindness_duration_seconds": self.partner_blindness_duration_seconds,
            "phantom_braking_prob": self.phantom_braking_prob,
            "phantom_braking_trigger_prob": self.phantom_braking_trigger_prob,
            "phantom_braking_duration_seconds": self.phantom_braking_duration_seconds,
            "phantom_braking_freeze_steering": self.phantom_braking_freeze_steering,
        }

    def _sample_init_step(self):
        # randomizer for the initialization of the C environment
        if not self.init_step_spread:
            return self.init_step
        upper = self.scenario_length - self.init_step_min_horizon
        return int(self.rng.integers(0, upper))

    def _next_eval_batch_size(self):
        """Scenarios the next eval batch instantiates: whatever is left of this
        worker's map window, clamped by max_scenarios_per_batch. The clamp bounds
        peak memory, since each scenario in a batch is a live C env owning its
        map geometry (hundreds of MB on large maps)."""
        if not self.eval_mode:
            return self.num_eval_scenarios
        consumed = self.starting_map_counter - self.starting_map_counter_init
        remaining = self.num_eval_scenarios - consumed
        if self.max_scenarios_per_batch is not None and remaining > self.max_scenarios_per_batch:
            return self.max_scenarios_per_batch
        return remaining

    @property
    def random_seed(self):
        # 63-bit: stays exact through int64 (CSV, numpy) and the C binding's PyLong_AsLongLong
        return int(self.rng.integers(0, 2**63, dtype=np.int64))

    def reset(self, seed=None):
        if seed is not None and not self.use_exact_episode_seed:
            self.rng = np.random.default_rng(seed)
            binding.vec_reset(self.c_envs, [self.random_seed for _ in range(self.num_envs)])
        else:
            binding.vec_reset(self.c_envs)
        self.tick = 0
        self.truncations[:] = 0
        if self.capture_replay:
            self._initialize_replay_captures()
        return self.observations, []

    def step(self, actions):
        if self._eval_exhausted:
            self.rewards[:] = 0
            self.terminals[:] = 0
            self.truncations[:] = 0
            return (self.observations, self.rewards, self.terminals, self.truncations, [])
        if self.capture_replay:
            self._capture_replay_step()
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        # vec_log is the training aggregate; it resets env->log, which eval reads
        # per episode, so it must not run in eval mode.
        if not self.eval_mode and self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                info.append(log)
                # print(log)
        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            will_resample = 1
            if will_resample:
                # Read this batch's finished episodes before the envs are resampled/closed.
                if self.eval_mode:
                    for summary in binding.vec_per_episode_log(self.c_envs):
                        summary["summary_type"] = "evaluation_episode"
                        if self.capture_replay:
                            summary["replay_environment_bundle"] = self._build_replay_environment_bundle(summary)
                        info.append(summary)
                self.current_num_eval_scenarios = self._next_eval_batch_size()
                if self.current_num_eval_scenarios == 0:
                    self._eval_exhausted = True
                    return (self.observations, self.rewards, self.terminals, self.truncations, info)
                binding.vec_close(self.c_envs)
                # Pairs already replayed this sweep; slice the rest so a deferred
                # scene resumes exactly where the previous batch stopped.
                pair_start = self.starting_map_counter - self.starting_map_counter_init
                remaining_map_indices = (
                    self.eval_map_indices[pair_start:] if self.eval_map_indices is not None else None
                )
                agent_offsets, map_ids, num_envs, maps_consumed = binding.shared(
                    num_agents=self.num_agents,
                    num_maps=self.num_maps,
                    starting_map_counter=self.starting_map_counter,
                    eval_mode=self.eval_mode,
                    eval_training_render=self.eval_training_render,
                    init_mode=self.init_mode,
                    control_mode=self.control_mode,
                    sdc_controller=self.sdc_controller,
                    non_sdc_controller=self.non_sdc_controller,
                    non_vehicle_controller=self.non_vehicle_controller,
                    replay_expert_agents=self.replay_expert_agents,
                    simulation_mode=self.simulation_mode,
                    init_step=self.init_step,
                    map_files=self.map_files,
                    seed=self.random_seed,
                    min_agents_per_env=self.min_agents_per_env,
                    max_agents_per_env=self.max_agents_per_env,
                    num_eval_scenarios=self.current_num_eval_scenarios,  # Use the dynamic size here
                    eval_map_indices=remaining_map_indices,
                    goal_radius=self.goal_radius,
                )
                self.agent_offsets = agent_offsets
                self.map_ids = map_ids
                self.num_envs = num_envs
                # In eval mode, don't wrap counter - allows termination condition to work correctly
                self.starting_map_counter = self.starting_map_counter + maps_consumed
                env_ids = []
                for i in range(num_envs):
                    cur = agent_offsets[i]
                    nxt = agent_offsets[i + 1]
                    env_seed = (
                        self.eval_scenario_seeds[pair_start + i]
                        if self.eval_scenario_seeds is not None
                        else self.random_seed
                    )
                    env_id = binding.env_init(
                        self.observations[cur:nxt],
                        self.actions[cur:nxt],
                        self.rewards[cur:nxt],
                        self.terminals[cur:nxt],
                        self.truncations[cur:nxt],
                        self.masks[cur:nxt],
                        env_seed,
                        **self._env_init_kwargs(self.map_files[map_ids[i]], nxt - cur),
                    )
                    env_ids.append(env_id)
                self.c_envs = binding.vectorize(*env_ids)

                binding.vec_reset(self.c_envs)
                if self.capture_replay:
                    self._initialize_replay_captures()
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

    # ====== Replay capture (active when capture_replay=True) ======

    def _normalize_scenarios(self, state):
        if isinstance(state, list):
            return state
        if isinstance(state, dict):
            return [state]
        raise RuntimeError(f"Unexpected Drive state type for replay capture: {type(state).__name__}")

    def _create_replay_capture(self, scenario, active_agent_offset):
        map_path = scenario.get("map_name")
        if not isinstance(map_path, str) or not map_path:
            raise RuntimeError("Replay capture requires a non-empty scenario map_name")
        active_agent_count = int(scenario["active_agent_count"])
        return {
            "metadata": {
                "map_name": os.path.basename(map_path).split(".")[0],
                "map_path": map_path,
                "scenario_id": scenario.get("scenario_id"),
                "goal_source": self.goal_source,
                "goal_regen_mode": self.goal_regen_mode,
                "num_goals": self.num_goals,
                "dynamics_model": self.dynamics_model,
                "worker_idx": self.replay_worker_idx,
                "active_agent_offset": active_agent_offset,
                "active_agent_count": active_agent_count,
            },
            "scenario": scenario,
            "agent_capacity": len(scenario["agents"] or []),
            "traffic_capacity": len(scenario["traffic_elements"] or []),
            "frames": {
                key: [] for key in ("agent_f32", "agent_i32", "metrics_f32", "puffer_f32", "traffic_i16", "rewards_f32")
            },
        }

    def _initialize_replay_captures(self):
        scenarios = self._normalize_scenarios(self.get_state())
        active_agent_offset = 0
        self._replay_captures = []
        for scenario in scenarios:
            self._replay_captures.append(self._create_replay_capture(scenario, active_agent_offset))
            active_agent_offset += int(scenario["active_agent_count"])
        env_count = len(self._replay_captures)
        agent_capacity = max((capture["agent_capacity"] for capture in self._replay_captures), default=0)
        traffic_capacity = max(
            (max(capture["traffic_capacity"], 1) for capture in self._replay_captures),
            default=1,
        )
        self._replay_frame_arrays = {
            "agent_f32": np.empty(
                (env_count, agent_capacity, binding.AGENT_F32_FIELDS),
                dtype=np.float32,
            ),
            "agent_i32": np.empty(
                (env_count, agent_capacity, binding.AGENT_I32_FIELDS),
                dtype=np.int32,
            ),
            "metrics_f32": np.empty(
                (env_count, agent_capacity, binding.METRICS_F32_FIELDS),
                dtype=np.float32,
            ),
            "puffer_f32": np.empty(
                (env_count, agent_capacity, binding.SCORE_F32_FIELDS),
                dtype=np.float32,
            ),
            "traffic_i16": np.empty(
                (env_count, traffic_capacity, binding.TRAFFIC_I16_FIELDS),
                dtype=np.int16,
            ),
            "rewards_f32": np.empty(
                (env_count, agent_capacity, binding.REWARD_F32_FIELDS),
                dtype=np.float32,
            ),
        }

    def _capture_replay_step(self):
        self.get_obs_html_frame(
            self._replay_frame_arrays["agent_f32"],
            self._replay_frame_arrays["agent_i32"],
            self._replay_frame_arrays["metrics_f32"],
            self._replay_frame_arrays["puffer_f32"],
            self._replay_frame_arrays["traffic_i16"],
            self._replay_frame_arrays["rewards_f32"],
        )
        for env_idx, capture in enumerate(self._replay_captures):
            agent_capacity = capture["agent_capacity"]
            traffic_capacity = max(capture["traffic_capacity"], 1)
            for key in ("agent_f32", "agent_i32", "metrics_f32", "puffer_f32", "rewards_f32"):
                capture["frames"][key].append(self._replay_frame_arrays[key][env_idx, :agent_capacity].copy())
            capture["frames"]["traffic_i16"].append(
                self._replay_frame_arrays["traffic_i16"][env_idx, :traffic_capacity].copy()
            )

    def _build_replay_environment_bundle(self, summary):
        env_slot = int(summary["env_slot"])
        if env_slot < 0 or env_slot >= len(self._replay_captures):
            raise RuntimeError(f"Replay summary has invalid env_slot={env_slot}")
        capture = self._replay_captures[env_slot]
        episode_length = int(summary["episode_length"])
        captured_frame_count = len(capture["frames"]["agent_f32"])
        if episode_length <= 0 or episode_length > captured_frame_count:
            raise RuntimeError(
                f"Replay episode_length={episode_length} is incompatible with "
                f"captured_frame_count={captured_frame_count} for env_slot={env_slot}"
            )
        metadata = dict(capture["metadata"])
        metadata["episode_length"] = episode_length
        replay_environment_bundle = {
            "schema": "interactive_replay_environment_v1",
            "metadata": metadata,
            "scenario": capture["scenario"],
            "frames": {key: np.stack(frames[:episode_length], axis=0) for key, frames in capture["frames"].items()},
        }
        return zlib.compress(
            pickle.dumps(replay_environment_bundle, protocol=pickle.HIGHEST_PROTOCOL),
            level=3,
        )

    def close(self):
        binding.vec_close(self.c_envs)

    def get_state(self):
        try:
            return binding.vec_get(self.c_envs)
        except Exception:
            return binding.env_get(self.c_envs)

    def get_obs_html_frame(self, agent_f32, agent_i32, metrics_f32, puffer_f32, traffic_i16, rewards_f32):
        binding.vec_get_obs_html_frame(
            self.c_envs,
            agent_f32,
            agent_i32,
            metrics_f32,
            puffer_f32,
            traffic_i16,
            rewards_f32,
        )


def test_performance(timeout=10, atn_cache=1024, num_agents=1024):
    import time

    env = Drive(num_agents=num_agents)
    env.reset()
    tick = 0
    num_agents = 1024
    actions = np.random.randint(0, env.single_action_space.n, (atn_cache, num_agents))

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {num_agents * tick / (time.time() - start)}")
    env.close()
