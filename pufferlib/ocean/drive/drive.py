import numpy as np
import gymnasium
import json
import struct
import os
import configparser
from pathlib import Path
import pufferlib
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.curriculum import GoalCurriculum


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
        reward_ade=0.0,
        goal_behavior=0,
        goal_radius=2.0,
        collision_behavior=0,
        offroad_behavior=0,
        dt=0.1,
        scenario_length=None,
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        action_type="discrete",
        dynamics_model="classic",
        max_controlled_agents=-1,
        buf=None,
        seed=1,
        init_steps=0,
        init_mode="random_agents_init",
        control_mode="control_vehicles",
        num_agents_per_world=32,
        vehicle_width = 2.0,
        vehicle_length = 4.5,
        vehicle_height = 1.8,
        goal_sampling_mode="randomized_curriculum",
        max_distance_to_goal=100.0,
        goal_curriculum_start_distance=5.0,
        goal_curriculum_end_distance=None,
        goal_curriculum_steps=5,
        goal_curriculum_schedule="linear",
        goal_curriculum_total_timesteps=None,
        binary_dir=None,
        ini_file="pufferlib/config/ocean/drive.ini",
    ):
        # env
        self.dt = dt
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.goal_radius = goal_radius
        self.goal_behavior = goal_behavior
        self.collision_behavior = collision_behavior
        self.offroad_behavior = offroad_behavior
        self.reward_ade = reward_ade
        self.ini_file = ini_file
        self.human_agent_idx = human_agent_idx
        self.scenario_length = scenario_length
        self.resample_frequency = resample_frequency
        self.dynamics_model = dynamics_model
        self.max_distance_to_goal = max_distance_to_goal
        self.goal_curriculum_start_distance = goal_curriculum_start_distance
        self.goal_curriculum_end_distance = goal_curriculum_end_distance
        self.goal_curriculum_steps = goal_curriculum_steps
        self.goal_curriculum_schedule = goal_curriculum_schedule
        self.goal_curriculum_total_timesteps = goal_curriculum_total_timesteps
        self.goal_curriculum = None
        ini_scenario_length = None
        ini_resample_frequency = None
        if self.ini_file and os.path.exists(self.ini_file):
            config = configparser.ConfigParser()
            config.read(self.ini_file)
            ini_scenario_length = config.getint("env", "scenario_length", fallback=None)
            ini_resample_frequency = config.getint("env", "resample_frequency", fallback=None)

        self.scenario_length = self.scenario_length if self.scenario_length is not None else ini_scenario_length
        if self.resample_frequency is None:
            self.resample_frequency = (
                ini_resample_frequency
                if ini_resample_frequency is not None
                else (self.scenario_length if self.scenario_length is not None else 0)
            )

        # Observation space calculation
        if dynamics_model == "classic":
            ego_features = 7
        elif dynamics_model == "jerk":
            ego_features = 10
        else:
            raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")

        if goal_behavior == 1:
            # preview next goal + 1
            ego_features += 2 

        self.ego_features = ego_features
        partner_features = 7
        road_features = 7
        max_partner_objects = 63
        max_road_objects = 200
        self.num_obs = ego_features + max_partner_objects * partner_features + max_road_objects * road_features
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)
        self.init_steps = init_steps
        self.init_mode_str = init_mode
        self.control_mode_str = control_mode
        self.num_agents_per_world = -1
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.vehicle_height = vehicle_height

        # TODO: Convert all these to enums
        if self.control_mode_str == "control_vehicles":
            self.control_mode = 0
        elif self.control_mode_str == "control_agents":
            self.control_mode = 1
        elif self.control_mode_str == "control_tracks_to_predict":
            self.control_mode = 2
        elif self.control_mode_str == "control_sdc_only":
            self.control_mode = 3
        else:
            raise ValueError(
                f"control_mode must be one of 'control_vehicles', 'control_tracks_to_predict', or 'control_agents'. Got: {self.control_mode_str}"
            )
        if self.init_mode_str == "create_all_valid":
            self.init_mode = 0
        elif self.init_mode_str == "create_only_controlled":
            self.init_mode = 1
        elif self.init_mode_str == "random_agents_init":
            self.init_mode = 2
            self.num_agents_per_world = num_agents_per_world
            print(f"Using random_agents_init with {self.num_agents_per_world} agents per world.")
        else:
            raise ValueError(
                f"init_mode must be one of 'create_all_valid' or 'create_only_controlled'. Got: {self.init_mode_str}"
            )

        if goal_sampling_mode == "fixed_from_dataset":
            self.goal_sampling_mode = 0
        elif goal_sampling_mode == "random_within_radius":
            self.goal_sampling_mode = 1
        elif goal_sampling_mode == "randomized_curriculum":
            self.goal_sampling_mode = 2
            end_distance = goal_curriculum_end_distance if goal_curriculum_end_distance is not None else max_distance_to_goal
            self.goal_curriculum = GoalCurriculum(
                start_distance=goal_curriculum_start_distance,
                end_distance=end_distance,
                steps=goal_curriculum_steps,
                schedule=goal_curriculum_schedule,
                total_timesteps=goal_curriculum_total_timesteps,
            )
            self.max_distance_to_goal = self.goal_curriculum.current_distance
            self.goal_curriculum_step_interval = self.goal_curriculum.step_interval
            self.goal_curriculum_total_timesteps = self.goal_curriculum.total_timesteps

        else:
            raise ValueError(
                f"goal_sampling_mode must be one of 'fixed_from_dataset', 'random_within_radius', or 'randomized_curriculum'. Got: {goal_sampling_mode}"
            )

        if action_type == "discrete":
            if dynamics_model == "classic":
                # Joint action space (assume dependence)
                self.single_action_space = gymnasium.spaces.MultiDiscrete([7 * 13])
                # Multi discrete (assume independence)
                # self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
            elif dynamics_model == "jerk":
                self.single_action_space = gymnasium.spaces.MultiDiscrete([4, 3])
            else:
                raise ValueError(f"dynamics_model must be 'classic' or 'jerk'. Got: {dynamics_model}")
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"action_space must be 'discrete' or 'continuous'. Got: {action_type}")

        self._action_type_flag = 0 if action_type == "discrete" else 1

        binary_dir = resolve_binary_dir(binary_dir, ini_file)
        self.map_files = collect_map_files(binary_dir=binary_dir, limit=num_maps)
        if not self.map_files:
            raise FileNotFoundError("No map binaries found for the configured selection")

        # Check maps availability
        available_maps = len(self.map_files)
        if num_maps > available_maps:
            raise ValueError(
                f"num_maps ({num_maps}) exceeds available maps ({available_maps}). "
                "Reduce num_maps or add more binaries to the configured directory."
            )
        self.num_maps = len(self.map_files)
        self.max_controlled_agents = int(max_controlled_agents)

        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs = binding.shared(
            num_agents=num_agents,
            num_maps=self.num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_sampling_mode=self.goal_sampling_mode,
            max_distance_to_goal=self.max_distance_to_goal,
            num_agents_per_world=self.num_agents_per_world,
            map_files=self.map_files,
            ini_file="pufferlib/config/ocean/drive.ini",
        )

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
                human_agent_idx=human_agent_idx,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                reward_goal=reward_goal,
                reward_goal_post_respawn=reward_goal_post_respawn,
                reward_ade=reward_ade,
                goal_radius=goal_radius,
                goal_behavior=self.goal_behavior,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                scenario_length=(int(self.scenario_length) if self.scenario_length is not None else None),
                max_controlled_agents=self.max_controlled_agents,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file=self.ini_file,
                init_steps=init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                num_agents_per_world=self.num_agents_per_world,
                vehicle_height = self.vehicle_height,
                vehicle_length = self.vehicle_length,
                vehicle_width = self.vehicle_width,
                goal_sampling_mode=self.goal_sampling_mode,
                max_distance_to_goal=self.max_distance_to_goal,
                map_files=self.map_files,
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

    def _rebuild_envs(self, seed=None):
        """Reinitialize C envs so curriculum changes reach the simulator."""
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        if hasattr(self, "c_envs"):
            binding.vec_close(self.c_envs)

        agent_offsets, map_ids, num_envs = binding.shared(
            num_agents=self.num_agents,
            num_maps=self.num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            max_controlled_agents=self.max_controlled_agents,
            goal_behavior=self.goal_behavior,
            goal_sampling_mode=self.goal_sampling_mode,
            max_distance_to_goal=self.max_distance_to_goal,
            num_agents_per_world=self.num_agents_per_world,
            map_files=self.map_files,
            ini_file="pufferlib/config/ocean/drive.ini",
        )

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
                human_agent_idx=self.human_agent_idx,
                reward_vehicle_collision=self.reward_vehicle_collision,
                reward_offroad_collision=self.reward_offroad_collision,
                reward_goal=self.reward_goal,
                reward_goal_post_respawn=self.reward_goal_post_respawn,
                reward_ade=self.reward_ade,
                goal_radius=self.goal_radius,
                goal_behavior=self.goal_behavior,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=self.dt,
                scenario_length=(int(self.scenario_length) if self.scenario_length is not None else None),
                max_controlled_agents=self.max_controlled_agents,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file=self.ini_file,
                init_steps=self.init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                num_agents_per_world=self.num_agents_per_world,
                vehicle_height = self.vehicle_height,
                vehicle_length = self.vehicle_length,
                vehicle_width = self.vehicle_width,
                goal_sampling_mode=self.goal_sampling_mode,
                max_distance_to_goal=self.max_distance_to_goal,
                map_files=self.map_files,
            )
            env_ids.append(env_id)

        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        self.c_envs = binding.vectorize(*env_ids)
        binding.vec_reset(self.c_envs, seed)
        self.terminals[:] = 1
        self.tick = 0

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
            log = binding.vec_log(self.c_envs) or {}
            if self.goal_sampling_mode == 2 and self.goal_curriculum is not None:
                log.update(
                    {
                        "max_distance_to_goal": float(self.max_distance_to_goal),
                        "goal_curriculum_step": int(self.goal_curriculum.current_idx),
                    }
                )
            info.append(log)

        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            if self.goal_sampling_mode == 2 and self.goal_curriculum is not None:
                if self.goal_curriculum.step_interval is None:
                    self.goal_curriculum.advance_one_stage()
                    self.max_distance_to_goal = self.goal_curriculum.current_distance
                else:
                    self.goal_curriculum.agent_steps += self.resample_frequency * self.num_agents
                    target_idx = min(
                        self.goal_curriculum.agent_steps // self.goal_curriculum.step_interval,
                        len(self.goal_curriculum.distances) - 1,
                    )
                    self.goal_curriculum.current_idx = target_idx
                    self.max_distance_to_goal = self.goal_curriculum.current_distance

            seed = np.random.randint(0, 2**32 - 1)
            self._rebuild_envs(seed=seed)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def set_curriculum_progress(self, agent_steps):
        """Force curriculum progress based on global agent-step count. Returns True if distance changed."""
        if self.goal_sampling_mode != 2 or self.goal_curriculum is None:
            return False
        agent_steps_int = int(agent_steps)
        prev_distance = self.max_distance_to_goal
        self.goal_curriculum.agent_steps = max(0, agent_steps_int)
        if self.goal_curriculum.step_interval is not None:
            target_idx = min(
                self.goal_curriculum.agent_steps // self.goal_curriculum.step_interval,
                len(self.goal_curriculum.distances) - 1,
            )
            if target_idx != self.goal_curriculum.current_idx:
                self.goal_curriculum.current_idx = target_idx
                self.max_distance_to_goal = self.goal_curriculum.current_distance
        if self.max_distance_to_goal != prev_distance:
            self._rebuild_envs()
            return True
        return False

    def notify(self):
        """Called from vector workers when a curriculum change is broadcast."""
        self._rebuild_envs()

    def get_global_agent_state(self):
        """Get current global state of all active agents.

        Returns:
            dict with keys 'x', 'y', 'z', 'heading', 'id' containing numpy arrays
            of shape (num_active_agents,)
        """
        num_agents = self.num_agents

        states = {
            "x": np.zeros(num_agents, dtype=np.float32),
            "y": np.zeros(num_agents, dtype=np.float32),
            "z": np.zeros(num_agents, dtype=np.float32),
            "heading": np.zeros(num_agents, dtype=np.float32),
            "id": np.zeros(num_agents, dtype=np.int32),
        }

        binding.vec_get_global_agent_state(
            self.c_envs, states["x"], states["y"], states["z"], states["heading"], states["id"]
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

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


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


def process_maps(source_dir, output_dir, prefix="map", limit=None, start_index=0):
    """Process maps from source_dir into output_dir using the provided prefix."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(source_dir.glob("*.json"))
    if limit is not None:
        json_files = json_files[:limit]

    print(f"Found {len(json_files)} JSON files")

    # Process each JSON file
    for i, map_path in enumerate(json_files):
        map_idx = start_index + i
        binary_file = f"{prefix}_{map_idx:03d}.bin"
        binary_path = output_dir / binary_file

        print(f"Processing {map_path.name} -> {binary_file}")
        # try:
        load_map(str(map_path), map_idx, str(binary_path))
        # except Exception as e:
        #     print(f"Error processing {map_path.name}: {e}")


def process_carla_maps(
    source_dir="data_utils/carla/carla", output_dir="resources/drive/carla_binaries", prefix="carla_map", limit=None
):
    """Process Carla JSON maps into binaries."""
    process_maps(source_dir=source_dir, output_dir=output_dir, prefix=prefix, limit=limit)


def collect_map_files(binary_dir, limit):
    """Collect map binaries from a directory with an optional limit."""
    if not binary_dir:
        binary_dir = "resources/drive/binaries"
    path = Path(binary_dir)
    if not path.exists():
        return []
    files = sorted(str(p) for p in path.glob("*.bin"))
    if limit is not None:
        files = files[:limit]
    return files


def resolve_binary_dir(binary_dir, ini_path):
    """Resolve binary directory, preferring explicit arg, then ini, then default."""
    if binary_dir:
        return binary_dir
    if ini_path and os.path.exists(ini_path):
        config = configparser.ConfigParser()
        config.read(ini_path)
        ini_dir = config.get("env", "binary_dir", fallback=None)
        if ini_dir:
            return ini_dir
    return "resources/drive/carla_binaries"


def test_performance(timeout=10, atn_cache=1024, num_agents=32):
    import time

    # 0 = fixed_from_dataset, 1 = random_within_radius, 2 = randomized_curriculum

    env = Drive(
        num_agents=num_agents,
        num_maps=1,
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
    test_performance()
    # process_carla_maps()
