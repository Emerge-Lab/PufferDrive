import numpy as np
import gymnasium
import json
import struct
import os
import math
import torch
import pufferlib
from pufferlib.ocean.drive import binding
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class Drive(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode=None,
        report_interval=1,
        width=1280,
        height=1024,
        human_agent_idx=0,
        reward_vehicle_collision=-0.1,
        reward_offroad_collision=-0.1,
        reward_goal=1.0,
        reward_goal_post_respawn=0.5,
        use_guided_autonomy=0,
        guidance_speed_weight=0.0,
        guidance_heading_weight=0.0,
        waypoint_reach_threshold=2.0,
        goal_behavior=0,
        goal_target_distance=10.0,
        goal_radius=2.0,
        goal_speed=20.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        episode_length=None,
        termination_mode=None,
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        action_type="discrete",
        dynamics_model="classic",
        max_controlled_agents=-1,
        buf=None,
        seed=1,
        bptt_horizon=32,
        human_data_dir="pufferlib/resources/drive/human_demonstrations",
        max_expert_sequences=128,
        init_steps=0,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        map_dir="resources/drive/binaries/training",
        use_all_maps=False,
    ):
        # env
        self.dt = dt
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.goal_radius = goal_radius
        self.goal_speed = goal_speed
        self.goal_behavior = goal_behavior
        self.goal_target_distance = goal_target_distance
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.use_guided_autonomy = use_guided_autonomy
        self.guidance_speed_weight = guidance_speed_weight
        self.guidance_heading_weight = guidance_heading_weight
        self.waypoint_reach_threshold = waypoint_reach_threshold
        self.human_agent_idx = human_agent_idx
        self.episode_length = episode_length
        self.termination_mode = termination_mode
        self.resample_frequency = resample_frequency
        self.dynamics_model = dynamics_model

        # Observation space calculation
        self.ego_features = {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}.get(
            dynamics_model
        )

        # Extract observation shapes from constants
        # These need to be defined in C, since they determine the shape of the arrays
        self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        self.max_partner_objects = binding.MAX_AGENTS - 1
        self.partner_features = binding.PARTNER_FEATURES
        self.road_features = binding.ROAD_FEATURES

        self.num_obs = (
            self.ego_features
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
        )
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)

        self.init_steps = init_steps
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        self.max_expert_sequences = int(max_expert_sequences)
        self.map_dir = map_dir

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
            if dynamics_model == "classic":
                # Joint action space (assume dependence)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
                # Multi discrete (assume independence)
                # self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
            elif dynamics_model == "jerk":
                # Joint action space (assume dependence) - 4 longitudinal × 3 lateral = 12
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4 * 3])
            else:
                raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"action_space must be 'discrete' or 'continuous'. Got: {action_type}")

        self._action_type_flag = 0 if action_type == "discrete" else 1

        # Check if resources directory exists
        binary_path = f"{map_dir}/map_000.bin"
        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"Required directory {binary_path} not found. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )

        # Check maps availability
        available_maps = len([name for name in os.listdir(map_dir) if name.endswith(".bin")])
        if num_maps > available_maps:
            raise ValueError(
                f"num_maps ({num_maps}) exceeds available maps in directory ({available_maps}). Please reduce num_maps or add more maps to resources/drive/binaries."
            )
        self.max_controlled_agents = int(max_controlled_agents)

        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs = binding.shared(
            map_dir=map_dir,
            num_agents=num_agents,
            num_maps=num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            use_all_maps=use_all_maps,
        )

        # agent_offsets[-1] works in both cases, just making it explicit that num_agents is ignored if use_all_maps
        self.num_agents = num_agents if not use_all_maps else agent_offsets[-1]
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
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=human_agent_idx,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                reward_goal=reward_goal,
                reward_goal_post_respawn=reward_goal_post_respawn,
                use_guided_autonomy=use_guided_autonomy,
                guidance_speed_weight=guidance_speed_weight,
                guidance_heading_weight=guidance_heading_weight,
                waypoint_reach_threshold=waypoint_reach_threshold,
                goal_radius=goal_radius,
                goal_speed=goal_speed,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                episode_length=(int(episode_length) if episode_length is not None else None),
                termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                max_controlled_agents=self.max_controlled_agents,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file="pufferlib/config/ocean/drive.ini",
                init_steps=init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=map_dir,
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

        # Human data storage
        self.bptt_horizon = bptt_horizon
        self.human_data_dir = human_data_dir
        if self.max_expert_sequences > 0:
            os.makedirs(self.human_data_dir, exist_ok=True)
            self._save_expert_data(bptt_horizon, self.max_expert_sequences)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.terminals[:] = 0
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs, self.num_agents)
            if log:
                info.append(log)
                # print(log)
        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            binding.vec_close(self.c_envs)
            agent_offsets, map_ids, num_envs = binding.shared(
                num_agents=self.num_agents,
                num_maps=self.num_maps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                init_steps=self.init_steps,
                max_controlled_agents=self.max_controlled_agents,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                goal_speed=self.goal_speed,
                map_dir=self.map_dir,
                use_all_maps=False,
            )
            self.agent_offsets = agent_offsets
            self.map_ids = map_ids
            self.num_envs = num_envs
            env_ids = []
            seed = np.random.randint(0, 2**32 - 1)
            for i in range(num_envs):
                cur = agent_offsets[i]
                nxt = agent_offsets[i + 1]
                env_id = binding.env_init(
                    self.observations[cur:nxt],
                    self.actions[cur:nxt],
                    self.rewards[cur:nxt],
                    self.terminals[cur:nxt],
                    self.truncations[cur:nxt],
                    seed,
                    action_type=self._action_type_flag,
                    human_agent_idx=self.human_agent_idx,
                    reward_vehicle_collision=self.reward_vehicle_collision,
                    reward_offroad_collision=self.reward_offroad_collision,
                    reward_goal=self.reward_goal,
                    reward_goal_post_respawn=self.reward_goal_post_respawn,
                    goal_radius=self.goal_radius,
                    goal_behavior=self.goal_behavior,
                    goal_target_distance=self.goal_target_distance,
                    goal_speed=self.goal_speed,
                    collision_behavior=self.collision_behavior,
                    offroad_behavior=self.offroad_behavior,
                    dt=self.dt,
                    episode_length=(int(self.episode_length) if self.episode_length is not None else None),
                    max_controlled_agents=self.max_controlled_agents,
                    map_id=map_ids[i],
                    max_agents=nxt - cur,
                    ini_file="pufferlib/config/ocean/drive.ini",
                    init_steps=self.init_steps,
                    init_mode=self.init_mode,
                    control_mode=self.control_mode,
                    map_dir=self.map_dir,
                )
                env_ids.append(env_id)
            self.c_envs = binding.vectorize(*env_ids)

            binding.vec_reset(self.c_envs, seed)
            self.terminals[:] = 1
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
            "x": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.float32),
            "valid": np.zeros((num_agents, self.episode_length - self.init_steps), dtype=np.int32),
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

    def _save_expert_data(self, bptt_horizon=32, max_expert_sequences=10000):
        """Collect and save expert trajectories with bptt_horizon length sequences."""
        trajectory_length = 91

        # Collect full expert trajectories - both discrete and continuous
        expert_actions_discrete = np.zeros((trajectory_length, self.num_agents, 2), dtype=np.float32)
        expert_actions_continuous = np.zeros((trajectory_length, self.num_agents, 2), dtype=np.float32)
        expert_observations_full = np.zeros((trajectory_length, self.num_agents, self.num_obs), dtype=np.float32)

        binding.vec_collect_expert_data(
            self.c_envs, expert_actions_discrete, expert_actions_continuous, expert_observations_full
        )

        # Determine how many sequences we can actually store
        num_sequences = min(self.num_agents, max_expert_sequences)

        # Preallocate sequences
        discrete_sequences = np.zeros((num_sequences, bptt_horizon, 2), dtype=np.float32)
        continuous_sequences = np.zeros((num_sequences, bptt_horizon, 2), dtype=np.float32)
        obs_sequences = np.zeros((num_sequences, bptt_horizon, self.num_obs), dtype=np.float32)

        # Take one sequence per agent (starting from timestep 0)
        for agent_idx in range(num_sequences):
            discrete_sequences[agent_idx] = expert_actions_discrete[:bptt_horizon, agent_idx, :]
            continuous_sequences[agent_idx] = expert_actions_continuous[:bptt_horizon, agent_idx, :]
            obs_sequences[agent_idx] = expert_observations_full[:bptt_horizon, agent_idx, :]

        self._cache_size = num_sequences

        torch.save(
            torch.from_numpy(discrete_sequences),
            os.path.join(self.human_data_dir, f"expert_actions_discrete_h{bptt_horizon}.pt"),
        )
        torch.save(
            torch.from_numpy(continuous_sequences),
            os.path.join(self.human_data_dir, f"expert_actions_continuous_h{bptt_horizon}.pt"),
        )
        torch.save(
            torch.from_numpy(obs_sequences),
            os.path.join(self.human_data_dir, f"expert_observations_h{bptt_horizon}.pt"),
        )

    def sample_expert_data(self, n_samples=512, return_both=False):
        """Sample a random batch of human (expert) sequences from disk.

        Args:
            n_samples: Number of sequences to randomly sample
            return_both: If True, return both continuous and discrete actions as a tuple.
                        If False, return only the action type matching the environment's action space.

        Returns:
            If return_both=True:
                (discrete_actions, continuous_actions, observations)
            If return_both=False:
                (actions, observations) where actions match the env's action type
        """
        discrete_path = os.path.join(self.human_data_dir, f"expert_actions_discrete_h{self.bptt_horizon}.pt")
        continuous_path = os.path.join(self.human_data_dir, f"expert_actions_continuous_h{self.bptt_horizon}.pt")
        observations_path = os.path.join(self.human_data_dir, f"expert_observations_h{self.bptt_horizon}.pt")

        observations_full = torch.load(observations_path, map_location="cpu")

        # Sample indices
        indices = torch.randint(0, self._cache_size, (n_samples,))
        sampled_obs = observations_full[indices]

        if return_both:
            discrete_actions = torch.load(discrete_path, map_location="cpu")[indices]
            continuous_actions = torch.load(continuous_path, map_location="cpu")[indices]
            return discrete_actions, continuous_actions, sampled_obs
        else:
            # Return only the action type matching the environment
            if self._action_type_flag == 1:  # continuous
                actions = torch.load(continuous_path, map_location="cpu")[indices]
            else:  # discrete
                actions = torch.load(discrete_path, map_location="cpu")[indices]
            return actions, sampled_obs

    def compute_realism_metrics(self, discrete_actions, continuous_actions):
        """Compute realism metrics from expert action samples.

        Args:
            discrete_actions: Tensor of shape (n_samples, bptt_horizon, 2) with discrete actions
            continuous_actions: Tensor of shape (n_samples, bptt_horizon, 2) with continuous actions

        Returns:
            Dictionary with realism metrics.
        """
        metrics = {}

        with torch.no_grad():
            # Flatten along the bptt_horizon
            continuous_accel = continuous_actions[:, :, 0].flatten()
            continuous_steer = continuous_actions[:, :, 1].flatten()

            metrics["expert_accel_histogram"] = continuous_accel.cpu().numpy()
            metrics["expert_steer_histogram"] = continuous_steer.cpu().numpy()

        return metrics
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


def infer_human_actions(obj):
    """Infer expert actions (steer, accel) using inverse bicycle model."""
    trajectory_length = 91

    # Initialize expert actions arrays
    expert_acceleration = []
    expert_steering = []

    positions = obj.get("position", [])
    velocities = obj.get("velocity", [])
    headings = obj.get("heading", [])
    valids = obj.get("valid", [])

    if len(positions) < 2 or len(velocities) < 2 or len(headings) < 2:
        return [0.0] * trajectory_length, [0.0] * trajectory_length

    dt = 0.1  # Discretization
    vehicle_length = obj.get("length", 4.5)  # Default vehicle length

    for t in range(trajectory_length - 1):
        if (
            t >= len(positions)
            or t >= len(velocities)
            or t >= len(headings)
            or t >= len(valids)
            or not valids[t]
            or not valids[t + 1]
            or t + 1 >= len(positions)
            or t + 1 >= len(velocities)
            or t + 1 >= len(headings)
        ):
            expert_acceleration.append(0.0)
            expert_steering.append(0.0)
            continue

        # Current state
        vel_t = velocities[t]
        heading_t = headings[t]
        speed_t = math.sqrt(vel_t.get("x", 0.0) ** 2 + vel_t.get("y", 0.0) ** 2)

        # Next state
        vel_t1 = velocities[t + 1]
        heading_t1 = headings[t + 1]
        speed_t1 = math.sqrt(vel_t1.get("x", 0.0) ** 2 + vel_t1.get("y", 0.0) ** 2)

        # Compute acceleration
        acceleration = (speed_t1 - speed_t) / dt

        # Normalize heading difference
        heading_diff = heading_t1 - heading_t
        while heading_diff > math.pi:
            heading_diff -= 2 * math.pi
        while heading_diff < -math.pi:
            heading_diff += 2 * math.pi

        # Compute yaw rate
        yaw_rate = heading_diff / dt

        # Compute steering using inverse bicycle model
        steering = 0.0
        if speed_t > 0.1:  # Avoid division by zero
            # From bicycle model: yaw_rate = (v * cos(beta) * tan(delta)) / L
            # Assuming beta ≈ 0: yaw_rate ≈ (v * tan(delta)) / L
            tan_steering = (yaw_rate * vehicle_length) / speed_t
            # Clamp tan_steering to avoid extreme values
            tan_steering = max(-10.0, min(10.0, tan_steering))
            steering = math.atan(tan_steering)

        # Clamp values to reasonable ranges
        acceleration = max(-4.0, min(4.0, acceleration))
        steering = max(-1.0, min(1.0, steering))

        expert_acceleration.append(acceleration)
        expert_steering.append(steering)

    # Handle last timestep
    expert_acceleration.append(0.0)
    expert_steering.append(0.0)

    # Ensure arrays are exactly trajectory_length
    expert_acceleration = expert_acceleration[:trajectory_length]
    expert_steering = expert_steering[:trajectory_length]

    # Pad if necessary
    while len(expert_acceleration) < trajectory_length:
        expert_acceleration.append(0.0)
    while len(expert_steering) < trajectory_length:
        expert_steering.append(0.0)

    return expert_acceleration, expert_steering


def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def dist(a, b):
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return dx * dx + dy * dy


def simplify_polyline(geometry, polyline_reduction_threshold, max_segment_length):
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
            if area < polyline_reduction_threshold and dist(point1, point3) <= max_segment_length:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def save_map_binary(map_data, output_file, unique_map_id):
    trajectory_length = 91
    """Saves map data in a binary format readable by C"""
    with open(output_file, "wb") as f:
        # Get metadata
        metadata = map_data.get("metadata", {})
        sdc_track_index = metadata.get("sdc_track_index", -1)  # -1 as default if not found
        tracks_to_predict = metadata.get("tracks_to_predict", [])

        # Write sdc_track_index
        f.write(struct.pack("i", sdc_track_index))

        # Write tracks_to_predict info (indices only)
        f.write(struct.pack("i", len(tracks_to_predict)))
        for track in tracks_to_predict:
            track_index = track.get("track_index", -1)
            f.write(struct.pack("i", track_index))

        # Count total entities
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get("objects", []):
            # Write unique map id
            f.write(struct.pack("i", unique_map_id))

            # Write base entity data
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))  # type
            f.write(struct.pack("i", obj.get("id", 0)))  # id
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

            # Infer and write human actions
            if obj_type == 1:  # Only for vehicles
                human_accel, human_steering = infer_human_actions(obj)

                # print(f"Human Acceleration: {human_accel}")
                # print(f"Human Steering: {human_steering}")

                f.write(struct.pack(f"{trajectory_length}f", *human_accel))
                f.write(struct.pack(f"{trajectory_length}f", *human_steering))
            else:
                # Write zeros for non-vehicles
                f.write(struct.pack(f"{trajectory_length}f", *[0.0] * trajectory_length))
                f.write(struct.pack(f"{trajectory_length}f", *[0.0] * trajectory_length))

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
            f.write(struct.pack("i", unique_map_id))

            geometry = road.get("geometry", [])
            road_type = road.get("map_element_id", 0)
            road_type_word = road.get("type", 0)
            if road_type_word == "lane":
                road_type = 2
            elif road_type_word == "road_edge":
                road_type = 15
            # breakpoint()
            if len(geometry) > 10 and road_type <= 16:
                geometry = simplify_polyline(geometry, 0.1, 250)
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
            f.write(struct.pack("i", road.get("id", 0)))  # id
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


def load_map(map_name, unique_map_id, binary_output=None):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, "r") as f:
        map_data = json.load(f)

    if binary_output:
        save_map_binary(map_data, binary_output, unique_map_id)


def _process_single_map(args):
    """Worker function to process a single map file"""
    i, map_path, binary_path = args
    try:
        load_map(str(map_path), i, str(binary_path))
        return (i, map_path.name, True, None)
    except Exception as e:
        return (i, map_path.name, False, str(e))


def process_all_maps(
    data_folder="data/processed/training",
    max_maps=10_000,
    num_workers=None,
):
    """Process all maps and save them as binaries using multiprocessing

    Args:
        data_folder: Path to the folder containing JSON map files
        max_maps: Maximum number of maps to process
        num_workers: Number of parallel workers (defaults to cpu_count())
    """
    from pathlib import Path

    if num_workers is None:
        num_workers = cpu_count()

    # Path to the training data
    data_dir = Path(data_folder)
    dataset_name = data_dir.name

    # Create the binaries directory if it doesn't exist
    binary_dir = Path(f"resources/drive/binaries/{dataset_name}")
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files in the training directory
    json_files = sorted(data_dir.glob("*.json"))

    # Prepare arguments for parallel processing
    tasks = []
    for i, map_path in enumerate(json_files[:max_maps]):
        binary_file = f"map_{i:03d}.bin"
        binary_path = binary_dir / binary_file
        tasks.append((i, map_path, binary_path))

    # Process maps in parallel with progress bar
    with Pool(num_workers) as pool:
        results = list(
            tqdm(pool.imap(_process_single_map, tasks), total=len(tasks), desc="Processing maps", unit="map")
        )

    # Collect statistics
    successful = sum(1 for _, _, success, _ in results if success)
    failed = sum(1 for _, _, success, _ in results if not success)

    if failed > 0:
        print(f"\nFailed {failed}/{len(results)} files:")
        for i, name, success, error in results:
            if not success:
                print(f"  {name}: {error}")


def test_performance(timeout=10, atn_cache=1024, num_agents=1024):
    import time

    env = Drive(
        num_agents=num_agents,
        num_maps=1,
        control_mode="control_vehicles",
        init_mode="create_all_valid",
        init_steps=0,
        episode_length=91,
    )

    env.reset()

    tick = 0
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
    # Process the train dataset
    process_all_maps(data_folder="data/processed/training")
    # Process the validation/test dataset
    # process_all_maps(data_folder="data/processed/validation")
    # # Process the validation_interactive dataset
    # process_all_maps(data_folder="data/processed/validation_interactive")
