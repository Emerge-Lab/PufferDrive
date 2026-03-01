import numpy as np
import gymnasium
import json
import struct
import os

import torch
import pufferlib
import math
from enum import IntEnum
from pufferlib.ocean.drive import binding
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


class RenderView(IntEnum):
    FULL_SIM_STATE = 0  # Orthographic top-down, fully observable simulator state
    BEV_AGENT_OBS = 1  # Orthographic top-down, only show what the selected agent can observe
    AGENT_PERSP = 2  # Third-person perspective following selected agent


class RegMode(IntEnum):
    NONE = 0
    LOG_PROB_DIRECT = 1
    KL_ANCHOR = 2


class Drive(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode=RenderView.FULL_SIM_STATE,
        report_interval=1,
        width=1280,
        height=1024,
        human_agent_idx=0,
        reward_vehicle_collision=-0.1,
        reward_offroad_collision=-0.1,
        reward_goal=1.0,
        reward_goal_post_respawn=0.5,
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
        buf=None,
        seed=1,
        init_steps=0,
        init_mode="create_all_valid",
        control_mode="control_vehicles",
        max_controlled_agents=32,
        map_dir="resources/drive/binaries/training",
        ini_file_path="pufferlib/config/ocean/drive.ini",
        reg_mode=None,
        anchor_cpt_path=None,
        uses_memory=False,
        memory_size=0,
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
        self.human_agent_idx = human_agent_idx
        self.episode_length = episode_length
        self.termination_mode = termination_mode
        self.resample_frequency = resample_frequency
        self.dynamics_model = dynamics_model
        self.ini_file_path = ini_file_path
        self.uses_memory = uses_memory
        self.memory_size = memory_size
        self.total_num_samples = 0
        self.max_controlled_agents = max_controlled_agents
        self.anchor_cpt_path = anchor_cpt_path

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
        self.map_dir = map_dir
        str_to_reg_mode = {
            "None": RegMode.NONE,
            "log_prob_direct": RegMode.LOG_PROB_DIRECT,
            "kl_anchor": RegMode.KL_ANCHOR,
        }
        self.reg_mode = str_to_reg_mode.get(reg_mode.strip('"'), RegMode.NONE)

        if self.control_mode_str == "control_vehicles":
            self.control_mode = 0
        elif self.control_mode_str == "control_agents":
            self.control_mode = 1
        elif self.control_mode_str == "control_wosac":
            self.control_mode = 2
        elif self.control_mode_str == "control_sdc_only":
            self.control_mode = 3
        elif self.control_mode_str == "control_mixed_play":
            self.control_mode = 4
        else:
            raise ValueError(
                f"control_mode must be one of 'control_vehicles', 'control_wosac', 'control_agents' or 'control_mixed_play'. Got: {self.control_mode_str}"
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
                self.joint_action_space_size = binding.NUM_ACCEL_BINS * binding.NUM_STEER_BINS
                self.single_action_space = gymnasium.spaces.MultiDiscrete([self.joint_action_space_size])
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

        # Iterate through all maps to count total agents that can be initialized for each map
        agent_offsets, map_ids, num_envs = binding.shared(
            map_dir=map_dir,
            num_agents=num_agents,
            num_maps=num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            max_controlled_agents=self.max_controlled_agents,
        )

        self.num_agents = agent_offsets[-1]
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
                goal_radius=goal_radius,
                goal_speed=goal_speed,
                goal_behavior=self.goal_behavior,
                goal_target_distance=self.goal_target_distance,
                collision_behavior=self.collision_behavior,
                offroad_behavior=self.offroad_behavior,
                dt=dt,
                episode_length=(int(episode_length) if episode_length is not None else None),
                termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file=self.ini_file_path,
                init_steps=init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=map_dir,
                max_controlled_agents=self.max_controlled_agents,
                render_mode=render_mode,
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        self.truncations[:] = 0
        return self.observations, []

    def resample_maps(self):
        """Resample environment maps.
        Args:

        """
        self.tick = 0
        binding.vec_close(self.c_envs)
        agent_offsets, map_ids, num_envs = binding.shared(
            num_agents=self.num_agents,
            num_maps=self.num_maps,
            init_mode=self.init_mode,
            control_mode=self.control_mode,
            init_steps=self.init_steps,
            goal_behavior=self.goal_behavior,
            goal_target_distance=self.goal_target_distance,
            goal_speed=self.goal_speed,
            map_dir=self.map_dir,
            max_controlled_agents=self.max_controlled_agents,
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
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file=self.ini_file_path,
                init_steps=self.init_steps,
                init_mode=self.init_mode,
                control_mode=self.control_mode,
                map_dir=self.map_dir,
                termination_mode=(int(self.termination_mode) if self.termination_mode is not None else 0),
                max_controlled_agents=self.max_controlled_agents,
                render_mode=self.render_mode,
            )
            env_ids.append(env_id)
        self.c_envs = binding.vectorize(*env_ids)
        binding.vec_reset(self.c_envs, seed)
        self.truncations[:] = 1
        self.terminals[:] = 1

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

        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            # Resample batch of scenes used for training
            self.resample_maps()

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
            "is_vehicle": np.zeros(num_agents, dtype=bool),
            "is_track_to_predict": np.zeros(num_agents, dtype=bool),
            "scenario_id": np.zeros(num_agents, dtype="S16"),
        }

        binding.vec_get_global_ground_truth_trajectories(
            self.c_envs,
            trajectories["x"],
            trajectories["y"],
            trajectories["z"],
            trajectories["heading"],
            trajectories["valid"],
            trajectories["id"],
            trajectories["is_vehicle"],
            trajectories["is_track_to_predict"],
            trajectories["scenario_id"],
        )

        for key in trajectories:
            trajectories[key] = trajectories[key][:, None]

        trajectories["scenario_id"] = trajectories["scenario_id"].astype(str)

        return trajectories

    def _hash_pair(self, obs, act):
        return hash((obs.round(3).tobytes(), act.round(2).tobytes()))

    def _init_regularization_strategy(self, device="cuda", bc_hidden_size=1024):
        bc_anchor = None
        data = {}

        if self.reg_mode == RegMode.KL_ANCHOR:
            from examples.train_bc_policy import BCPolicy

            bc_policy = BCPolicy(
                input_size=self.num_obs,
                hidden_size=bc_hidden_size,
                output_size=self.joint_action_space_size,
            ).to(device)
            bc_policy.load_state_dict(torch.load(self.anchor_cpt_path, map_location=device))
            bc_policy.eval()
            for p in bc_policy.parameters():
                p.requires_grad = False
            bc_anchor = bc_policy
            print(f"Loaded BC anchor policy from {self.anchor_cpt_path}")

        elif self.reg_mode == RegMode.LOG_PROB_DIRECT:
            total_samples, unique_samples = self._prepare_human_data()
            data["data/total_human_samples"] = total_samples
            data["data/unique_human_samples"] = unique_samples
            data["data/perc_unique_human_samples"] = (unique_samples / total_samples) * 100
            print(f"Prepared {total_samples} human demonstrations ({unique_samples} unique)")

        return bc_anchor, data

    def _prepare_human_data(self, max_samples=16_384):
        """Prepare human demonstrations."""

        trajectory_length = 91

        if self.dynamics_model == "classic":
            expert_actions_discrete = np.zeros((trajectory_length, self.num_agents, 1), dtype=np.float32)
            expert_actions_continuous = np.zeros((trajectory_length, self.num_agents, 2), dtype=np.float32)
            expert_observations_full = np.zeros((trajectory_length, self.num_agents, self.num_obs), dtype=np.float32)

            binding.vec_collect_expert_data(
                self.c_envs, expert_actions_discrete, expert_actions_continuous, expert_observations_full
            )

            if np.all(expert_actions_discrete == -1):
                raise ValueError("No valid human demonstrations could be collected. Please check the data format.")

            if self.uses_memory:
                # LSTM is conditioned on the memory_size last observations.
                # Adapt the human data collection to chunk trajectories into
                # sequences of length memory_size: (feature_dim, memory_size)

                # Filter out invalid timesteps first
                invalid_action_mask = (expert_actions_discrete == -1.0).squeeze(-1)  # (T, N)

                sequences_obs = []
                sequences_actions_discrete = []
                sequences_actions_continuous = []

                # Calculate max sequences to collect
                max_sequences = max_samples // self.memory_size

                # Process each agent's trajectory
                for agent_idx in range(self.num_agents):
                    # Early exit if we have enough sequences
                    if len(sequences_obs) >= max_sequences:
                        break

                    # Get valid timesteps for this agent
                    valid_mask = ~invalid_action_mask[:, agent_idx]
                    valid_timesteps = np.where(valid_mask)[0]

                    if len(valid_timesteps) < self.memory_size:
                        # Skip agents with insufficient valid data
                        continue

                    # Extract valid data for this agent
                    agent_obs = expert_observations_full[valid_mask, agent_idx, :]
                    agent_actions_discrete = expert_actions_discrete[valid_mask, agent_idx, :]
                    agent_actions_continuous = expert_actions_continuous[valid_mask, agent_idx, :]

                    # Chunk into sequences of memory_size
                    num_sequences = len(agent_obs) - self.memory_size + 1
                    for start_idx in range(num_sequences):
                        # Check if we've collected enough
                        if len(sequences_obs) >= max_sequences:
                            break

                        end_idx = start_idx + self.memory_size
                        sequences_obs.append(agent_obs[start_idx:end_idx])
                        sequences_actions_discrete.append(agent_actions_discrete[start_idx:end_idx])
                        sequences_actions_continuous.append(agent_actions_continuous[start_idx:end_idx])

                if len(sequences_obs) == 0:
                    raise ValueError("No valid sequences could be created. Check memory_size and trajectory validity.")

                # Stack all sequences: [num_sequences, memory_size, feature_dim]
                self.expert_observations_full = torch.tensor(np.stack(sequences_obs, axis=0), dtype=torch.float32)
                self.expert_actions_discrete = torch.tensor(
                    np.stack(sequences_actions_discrete, axis=0), dtype=torch.float32
                )
                self.expert_actions_continuous = torch.tensor(
                    np.stack(sequences_actions_continuous, axis=0), dtype=torch.float32
                )

                self.total_num_samples = self.expert_actions_discrete.shape[0]
            else:
                # For classic dynamics without memory (no LSTM), we can directly
                # use the collected data as sequences. Just filter out the invalid timesteps
                invalid_action_mask = (expert_actions_discrete == -1.0).squeeze(-1)

                # Shape: [num_samples, feature_dim]
                self.expert_actions_discrete = torch.Tensor(expert_actions_discrete[~invalid_action_mask])
                self.expert_actions_continuous = torch.Tensor(expert_actions_continuous[~invalid_action_mask])
                self.expert_observations_full = torch.Tensor(expert_observations_full[~invalid_action_mask])

                if len(self.expert_observations_full) > max_samples:
                    # Limit storage if needed (keep most recent samples)
                    self.expert_observations_full = self.expert_observations_full[:max_samples:]
                    self.expert_actions_discrete = self.expert_actions_discrete[:max_samples:]
                    self.expert_actions_continuous = self.expert_actions_continuous[:max_samples:]

                # Count unique number of (observation, action) pairs. This gives
                # an idea of the diversity and coverage of the human demonstrations.
                obs_np = self.expert_observations_full.numpy()
                act_np = self.expert_actions_discrete.numpy()
                self.total_unique_samples = len({self._hash_pair(obs_np[i], act_np[i]) for i in range(len(obs_np))})
                self.total_num_samples = self.expert_actions_discrete.shape[0]

                return self.total_num_samples, self.total_unique_samples
        else:
            raise NotImplementedError(
                "Expert data collection is currently only implemented for the classic dynamics model. "
            )

    def sample_human_demonstrations(self, batch_size=512):
        # get random indices between min and max
        rand_idx = torch.randint(0, self.total_num_samples - 1, (batch_size,))
        return (
            self.expert_actions_discrete[rand_idx],
            self.expert_actions_continuous[rand_idx],
            self.expert_observations_full[rand_idx],
        )

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
            "scenario_id": np.zeros(num_polylines, dtype="S16"),
        }

        binding.vec_get_road_edge_polylines(
            self.c_envs,
            polylines["x"],
            polylines["y"],
            polylines["lengths"],
            polylines["scenario_id"],
        )

        polylines["scenario_id"] = polylines["scenario_id"].astype(str)

        return polylines

    def render(self, view_mode: RenderView = RenderView.FULL_SIM_STATE, draw_traces: bool = True, env_id: int = 0):
        binding.vec_render(self.c_envs, int(view_mode), draw_traces, env_id)

    def close(self):
        binding.vec_close(self.c_envs)

    @property
    def scenario_ids(self) -> list[str]:
        """Return scenario ID string for each env, stripping null padding."""
        return [s.rstrip("\x00") for s in binding.vec_get_scenario_ids(self.c_envs)]


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
        return [-1.0] * trajectory_length, [-1.0] * trajectory_length

    dt = 0.1  # Discretization
    vehicle_length = obj.get("length", 4.5)  # Default vehicle length

    for t in range(trajectory_length):
        # Check if we have enough data and both current and next timesteps are valid
        if (
            t >= len(positions)
            or t >= len(velocities)
            or t >= len(headings)
            or t >= len(valids)
            or not valids[t]
            or t + 1 >= len(positions)
            or t + 1 >= len(velocities)
            or t + 1 >= len(headings)
            or t + 1 >= len(valids)
            or not valids[t + 1]
        ):
            expert_acceleration.append(-1.0)
            expert_steering.append(-1.0)
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
        acceleration = max(-10.0, min(10.0, acceleration))
        steering = max(-2.0, min(2.0, steering))

        expert_acceleration.append(acceleration)
        expert_steering.append(steering)

    # Ensure arrays are exactly trajectory_length (should already be, but just in case)
    assert len(expert_acceleration) == trajectory_length, (
        f"Expected {trajectory_length}, got {len(expert_acceleration)}"
    )
    assert len(expert_steering) == trajectory_length, f"Expected {trajectory_length}, got {len(expert_steering)}"

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

        # Write original scenario_id with fallback to placeholder
        scenario_id = map_data.get("scenario_id", f"map_{unique_map_id:03d}")
        f.write(struct.pack("16s", scenario_id.encode("utf-8")))

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
            if obj_type in [1, 2, 3]:  # For vehicles, pedestrians, cyclists
                human_accel, human_steering = infer_human_actions(obj)
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
    max_maps=50_000,
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
    process_all_maps(data_folder="data/processed/interactive_data_training_100")
    # Process the validation/test dataset
    # process_all_maps(data_folder="data/processed/validation")
    # # Process the validation_interactive dataset
    # process_all_maps(data_folder="data/processed/validation_interactive")
