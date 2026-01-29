import argparse
from pathlib import Path
import numpy as np
import gymnasium
import json
import struct
import os
import pufferlib
from pufferlib.ocean.drive import binding


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
        reward_traffic_light_violation=-0.5,
        goal_distance=40.0,
        waypoints_spacing=5.0,
        reward_goal=1.0,
        reward_goal_post_respawn=0.5,
        reward_ade=0.0,
        reach_goal_behavior=0,
        reward_speed=0.0,
        reward_comfort=-0.05,
        reward_velocity=0.0,
        reward_lane_align=-0.01,
        reward_lane_center=-0.005,
        reward_timestep=-0.000025,
        goal_radius=2.0,
        collision_behavior=0,
        offroad_behavior=0,
        traffic_light_behavior=0,
        end_sdc_path_behavior=0,
        dt=0.1,
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
        buf=None,
        seed=1,
        init_steps=0,
        eval_mode=0,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        map_dir=None,
        target_type="goal",
        reward_conditioning=False,
        reward_randomization=False,
    ):
        self.dt = dt
        self.reward_conditioning = reward_conditioning
        self.reward_randomization = reward_randomization
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_traffic_light_violation = reward_traffic_light_violation
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.goal_radius = goal_radius
        self.goal_distance = goal_distance
        self.waypoints_spacing = waypoints_spacing
        self.reach_goal_behavior = reach_goal_behavior
        self.target_type_str = target_type
        if target_type == "goal":
            self.target_type = binding.TARGET_GOAL
        elif target_type == "waypoints":
            self.target_type = binding.TARGET_WAYPOINTS
        elif target_type == "both":
            self.target_type = binding.TARGET_BOTH
        else:
            raise ValueError(f"target_type must be 'goal', 'waypoints', or 'both'. Got: {target_type}")
        self.reward_ade = reward_ade
        self.reward_speed = reward_speed
        self.reward_comfort = reward_comfort
        self.reward_velocity = reward_velocity
        self.reward_lane_align = reward_lane_align
        self.reward_lane_center = reward_lane_center
        self.reward_timestep = reward_timestep
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.traffic_light_behavior = traffic_light_behavior
        self.end_sdc_path_behavior = end_sdc_path_behavior
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
        self.termination_mode = termination_mode
        self.seed = seed
        self.min_agents_per_env = min_agents_per_env
        self.max_agents_per_env = max_agents_per_env

        # Observation space calculation based on target_type
        include_goal = self.target_type in (binding.TARGET_GOAL, binding.TARGET_BOTH)
        include_waypoints = self.target_type in (binding.TARGET_WAYPOINTS, binding.TARGET_BOTH)

        if include_goal:
            self.ego_features = {"classic": binding.EGO_FEATURES_CLASSIC, "jerk": binding.EGO_FEATURES_JERK}.get(
                dynamics_model
            )
        else:
            self.ego_features = {
                "classic": binding.EGO_FEATURES_CLASSIC_NO_GOAL,
                "jerk": binding.EGO_FEATURES_JERK_NO_GOAL,
            }.get(dynamics_model)

        # Extract observation shapes from constants
        # These need to be defined in C, since they determine the shape of the arrays
        self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        self.max_partner_objects = binding.MAX_AGENTS_OBSERVATIONS
        self.max_traffic_controls = binding.MAX_TRAFFIC_CONTROLS
        self.partner_features = binding.PARTNER_FEATURES
        self.road_features = binding.ROAD_FEATURES
        self.traffic_control_features = binding.TRAFFIC_CONTROL_FEATURES
        self.gps_features = binding.GPS_FEATURES
        self.max_gps_objects = binding.MAX_GPS_OBSERVATIONS if include_waypoints else 0
        self.num_reward_coefs = binding.NUM_REWARD_COEFS if reward_conditioning else 0
        self.num_obs = (
            self.ego_features
            + self.num_reward_coefs
            + self.max_partner_objects * self.partner_features
            + self.max_road_objects * self.road_features
            + self.max_traffic_controls * self.traffic_control_features
            + self.max_gps_objects * self.gps_features
        )

        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)

        self.init_steps = init_steps
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        self.simulation_mode_str = simulation_mode
        self.map_dir = map_dir

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
            raise ValueError(f"num_maps ({num_maps}) exceeds available maps in {map_dir} ({available_maps}).")
        self.starting_map_counter = 0
        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs = binding.shared(
            map_dir=map_dir,
            num_agents=num_agents,
            num_maps=num_maps,
            starting_map_counter=self.starting_map_counter,
            eval_mode=self.eval_mode,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            simulation_mode=self.simulation_mode,
            init_steps=self.init_steps,
            reach_goal_behavior=self.reach_goal_behavior,
            seed=self.seed,
            min_agents_per_env=self.min_agents_per_env,
            max_agents_per_env=self.max_agents_per_env,
        )
        self.starting_map_counter = (self.starting_map_counter + 1) % self.num_maps
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
                seed,
                action_type=self._action_type_flag,
                dynamics_model=self.dynamics_model_flag,
                human_agent_idx=self.human_agent_idx,
                reward_vehicle_collision=self.reward_vehicle_collision,
                reward_offroad_collision=self.reward_offroad_collision,
                reward_traffic_light_violation=self.reward_traffic_light_violation,
                reward_goal=self.reward_goal,
                reward_goal_post_respawn=self.reward_goal_post_respawn,
                reward_ade=self.reward_ade,
                reward_speed=self.reward_speed,
                reward_comfort=self.reward_comfort,
                reward_velocity=self.reward_velocity,
                reward_lane_align=self.reward_lane_align,
                reward_lane_center=self.reward_lane_center,
                reward_timestep=self.reward_timestep,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                traffic_light_behavior=self.traffic_light_behavior,
                end_sdc_path_behavior=self.end_sdc_path_behavior,
                goal_radius=self.goal_radius,
                goal_distance=self.goal_distance,
                waypoints_spacing=self.waypoints_spacing,
                reach_goal_behavior=self.reach_goal_behavior,
                target_type=self.target_type,
                dt=self.dt,
                scenario_length=(int(self.scenario_length) if self.scenario_length is not None else None),
                termination_mode=int(self.termination_mode),
                map_dir=self.map_dir,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                max_agents_per_env=self.max_agents_per_env,
                ini_file="pufferlib/config/ocean/drive.ini",
                init_steps=self.init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                simulation_mode=self.simulation_mode,
                reward_conditioning=self.reward_conditioning,
                reward_randomization=self.reward_randomization,
                eval_mode=self.eval_mode,
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        self.truncations[:] = 0
        return self.observations, []

    def step(self, actions):
        self.terminals[:] = 0
        self.truncations[:] = 0
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
            will_resample = 1
            if will_resample:
                binding.vec_close(self.c_envs)
                # np.random.seed(None)
                random_seed = self.seed + np.random.randint(0, 2**8 - 1)
                agent_offsets, map_ids, num_envs = binding.shared(
                    num_agents=self.num_agents,
                    num_maps=self.num_maps,
                    starting_map_counter=self.starting_map_counter,
                    eval_mode=self.eval_mode,
                    init_mode=self.init_mode,
                    control_mode=self.control_mode,
                    simulation_mode=self.simulation_mode,
                    init_steps=self.init_steps,
                    reach_goal_behavior=self.reach_goal_behavior,
                    map_dir=self.map_dir,
                    seed=random_seed,
                    min_agents_per_env=self.min_agents_per_env,
                    max_agents_per_env=self.max_agents_per_env,
                )

                self.starting_map_counter = (self.starting_map_counter + 1) % self.num_maps
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
                        random_seed,
                        action_type=self._action_type_flag,
                        human_agent_idx=self.human_agent_idx,
                        reward_vehicle_collision=self.reward_vehicle_collision,
                        reward_offroad_collision=self.reward_offroad_collision,
                        reward_traffic_light_violation=self.reward_traffic_light_violation,
                        reward_goal=self.reward_goal,
                        reward_goal_post_respawn=self.reward_goal_post_respawn,
                        reward_ade=self.reward_ade,
                        reward_speed=self.reward_speed,
                        reward_comfort=self.reward_comfort,
                        reward_velocity=self.reward_velocity,
                        reward_lane_align=self.reward_lane_align,
                        reward_lane_center=self.reward_lane_center,
                        reward_timestep=self.reward_timestep,
                        collision_behavior=self.collision_behavior,
                        offroad_behavior=self.offroad_behavior,
                        traffic_light_behavior=self.traffic_light_behavior,
                        end_sdc_path_behavior=self.end_sdc_path_behavior,
                        goal_radius=self.goal_radius,
                        goal_distance=self.goal_distance,
                        waypoints_spacing=self.waypoints_spacing,
                        reach_goal_behavior=self.reach_goal_behavior,
                        target_type=self.target_type,
                        dt=self.dt,
                        scenario_length=(int(self.scenario_length) if self.scenario_length is not None else None),
                        termination_mode=int(self.termination_mode),
                        map_dir=self.map_dir,
                        map_id=map_ids[i],
                        max_agents=nxt - cur,
                        max_agents_per_env=self.max_agents_per_env,
                        ini_file="pufferlib/config/ocean/drive.ini",
                        init_steps=self.init_steps,
                        init_mode=self.init_mode,
                        control_mode=self.control_mode,
                        simulation_mode=self.simulation_mode,
                        reward_conditioning=self.reward_conditioning,
                        reward_randomization=self.reward_randomization,
                        eval_mode=self.eval_mode,
                    )
                    env_ids.append(env_id)
                self.c_envs = binding.vectorize(*env_ids)

                binding.vec_reset(self.c_envs, random_seed)
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
            "x": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.float32),
            "valid": np.zeros((num_agents, self.scenario_length - self.init_steps), dtype=np.int32),
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
