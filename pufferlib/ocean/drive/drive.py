import numpy as np
import gymnasium
import json
import struct
import os
import random
import pufferlib
from pufferlib.ocean.drive import binding
import numpy as np
import gymnasium
import json
import struct
import os
import random
import pufferlib
import torch 
from pufferlib.ocean.drive import binding
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import importlib


class Drive(pufferlib.PufferEnv):
    def __init__(self, render_mode=None, report_interval=1,
            width=1280, height=1024,
            human_agent_idx=0,
            reward_vehicle_collision=-0.1,
            reward_offroad_collision=-0.1,
            reward_goal_post_respawn=0.5,
            reward_vehicle_collision_post_respawn=-0.25,
            spawn_immunity_timer=30,
            resample_frequency = 91,
            num_maps=100,
            num_agents=512,
            buf = None,
            seed=1, 
            population_play = False,
            condition_type = "none",
            oracle_mode = False,
            collision_weight_lb = -1.0,
            collision_weight_ub = 0.0,
            offroad_weight_lb = -1.0,
            offroad_weight_ub = 0.0,
            goal_weight_lb = 0.0,
            goal_weight_ub = 1.0,
            entropy_weight_lb = 0.0,
            entropy_weight_ub = 1.0,
            co_player_policy_name = None,
            co_player_rnn_name = None, 
            co_player_policy = None,
            co_player_rnn = None, 
            ):

        # env
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.reward_vehicle_collision_post_respawn = reward_vehicle_collision_post_respawn
        self.spawn_immunity_timer = spawn_immunity_timer
        self.human_agent_idx = human_agent_idx
        self.resample_frequency = resample_frequency
        self.population_play = population_play

        self.condition_type = condition_type
        self.reward_conditioned = condition_type in ("reward", "all")
        self.entropy_conditioned = condition_type in ("entropy", "all")
        self.oracle_mode = oracle_mode

        self.collision_weight_lb = collision_weight_lb if self.reward_conditioned else self.reward_vehicle_collision
        self.collision_weight_ub = collision_weight_ub if self.reward_conditioned else self.reward_vehicle_collision
        self.offroad_weight_lb = offroad_weight_lb if self.reward_conditioned else self.reward_offroad_collision
        self.offroad_weight_ub = offroad_weight_ub if self.reward_conditioned else self.reward_offroad_collision
        self.goal_weight_lb = goal_weight_lb if self.reward_conditioned else 1.0
        self.goal_weight_ub = goal_weight_ub if self.reward_conditioned else 1.0
        self.entropy_weight_lb = entropy_weight_lb
        self.entropy_weight_ub = entropy_weight_ub

        conditioning_dims = (3 if self.reward_conditioned else 0) + (1 if self.entropy_conditioned else 0)
        self.oracle_dims = num_agents * conditioning_dims if self.oracle_mode else 0

        self.num_obs = 7 + conditioning_dims + 63*7 + 200*7 + self.oracle_dims

        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.num_agents_const = num_agents
        self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
        
        if self.population_play:
            if self.reward_conditioned or self.entropy_conditioned:
                raise ValueError("Population play with reward conditioning is not supported.")

        # self.single_action_space = gymnasium.spaces.Box(
        #     low=-1, high=1, shape=(2,), dtype=np.float32
        # )
        # Check if resources directory exists
        binary_path = "resources/drive/binaries/map_000.bin"
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Required directory {binary_path} not found. Please ensure the Drive maps are downloaded and installed correctly per docs.")
        
        binding_tuple =  binding.shared(num_agents=num_agents, num_maps=num_maps, population_play = self.population_play)
        
        
        if self.population_play:
            agent_offsets, map_ids, num_envs, self.ego_ids, co_player_ids = binding_tuple
            self.co_player_ids = [item for sublist in co_player_ids for item in sublist]
            self.num_ego_agents = num_agents

            env_co_player_ids = [[id - next(agent_offsets[i] for i in range(len(agent_offsets)-1) 
                                if agent_offsets[i] <= id < agent_offsets[i+1]) 
                     for id in subset] for subset in co_player_ids]
            
            env_ego_ids = [
                min(set(range(agent_offsets[i+1] - agent_offsets[i])) - set(subset)) 
                if subset and set(range(agent_offsets[i+1] - agent_offsets[i])) - set(subset)
                else 0
                for i, subset in enumerate(env_co_player_ids)
            ]
            
            self.num_co_players = len(self.co_player_ids)
            self.num_agents = self.total_agents = self.num_co_players + self.num_ego_agents
            co_player_atn_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_co_players)
            if isinstance(self.single_action_space, pufferlib.spaces.Box):
                self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=co_player_atn_space.dtype)
            else:
                self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=np.int32)
            self.co_player_policy = co_player_policy
            self.set_co_player_state()
        else:
            agent_offsets, map_ids, num_envs = binding_tuple
            self.num_agents = self.num_agents_const
            self.ego_ids = [i for i in range(agent_offsets[-1])]
            env_co_player_ids = [[] for i in range(num_envs)]
            env_ego_ids = [0 for i in range(num_envs)]
       
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        super().__init__()
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i+1]

            env_id = binding.env_init(
                self.observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                human_agent_idx=self.human_agent_idx,
                reward_vehicle_collision=self.reward_vehicle_collision,
                reward_offroad_collision=self.reward_offroad_collision,
                reward_goal_post_respawn=self.reward_goal_post_respawn,
                reward_vehicle_collision_post_respawn=self.reward_vehicle_collision_post_respawn,
                spawn_immunity_timer=self.spawn_immunity_timer,
                map_id=map_ids[i],
                use_rc=self.reward_conditioned,
                use_ec=self.entropy_conditioned,
                oracle_mode=self.oracle_mode,
                max_agents=nxt-cur,
                collision_weight_lb=self.collision_weight_lb,
                collision_weight_ub=self.collision_weight_ub,
                offroad_weight_lb=self.offroad_weight_lb,
                offroad_weight_ub=self.offroad_weight_ub,
                goal_weight_lb=self.goal_weight_lb,
                goal_weight_ub=self.goal_weight_ub,
                entropy_weight_lb=self.entropy_weight_lb,
                entropy_weight_ub=self.entropy_weight_ub,
                population_play = self.population_play,
                num_co_players = len(env_co_player_ids[i]),
                co_player_ids = env_co_player_ids[i],
                ego_agent_id = env_ego_ids[i],
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

        if self.population_play:
            self.num_agents = self.num_ego_agents

    # def map_to_local_indices(self, co_player_ids, agent_offsets):
    #     """
    #     Map global co_player_ids to local 0-indexed positions within their agent groups
    #     """
    #     env_co_player_ids = []
        
    #     for subset in co_player_ids:
    #         local_subset = []
    #         for global_id in subset:
    #             # Find which agent group this global_id belongs to
    #             for i in range(len(agent_offsets) - 1):
    #                 if agent_offsets[i] <= global_id < agent_offsets[i + 1]:
    #                     # Convert to local index within this agent group
    #                     local_id = global_id - agent_offsets[i]
    #                     local_subset.append(local_id)
    #                     break
    #         env_co_player_ids.append(local_subset)
        
    #     return env_co_player_ids
            

    def set_buffers(self):
        "Number of co players is non-stationary, resets env variable shape to account for this "

        obs_space = self.single_observation_space
        self.observations = np.zeros((self.total_agents, *obs_space.shape), dtype=obs_space.dtype)
        self.rewards = np.zeros(self.total_agents, dtype=np.float32)
        self.terminals = np.zeros(self.total_agents, dtype=bool)
        self.truncations = np.zeros(self.total_agents, dtype=bool)
        self.masks = np.ones(self.total_agents, dtype=bool)

        atn_space = pufferlib.spaces.joint_space(self.single_action_space, self.total_agents)
        if isinstance(self.single_action_space, pufferlib.spaces.Box):
            self.actions = np.zeros(atn_space.shape, dtype=atn_space.dtype)
        else:
            self.actions = np.zeros(atn_space.shape, dtype=np.int32)

        co_player_atn_space = pufferlib.spaces.joint_space(self.single_action_space, self.num_co_players)
        if isinstance(self.single_action_space, pufferlib.spaces.Box):
            self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=co_player_atn_space.dtype)
        else:
            self.co_player_actions = np.zeros(co_player_atn_space.shape, dtype=np.int32)

    def set_co_player_state(self):
        self.state = dict(
            lstm_h=torch.zeros(self.num_co_players, self.co_player_policy.hidden_size),
            lstm_c=torch.zeros(self.num_co_players, self.co_player_policy.hidden_size),
        )
    
    def get_co_player_actions(self):
        co_player_obs = self.observations[self.co_player_ids]
        co_player_obs = torch.as_tensor(co_player_obs)
        logits, value = self.co_player_policy.forward_eval(co_player_obs, self.state) 
        co_player_action, logprob, _ = pufferlib.pytorch.sample_logits(logits) 
        co_player_action = co_player_action.cpu().numpy().reshape(self.co_player_actions.shape)
        return co_player_action


    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []
    
    

    def step(self, ego_actions):
        self.terminals[:] = 0
        self.actions[self.ego_ids] = ego_actions

        if self.population_play:
            co_player_actions = self.get_co_player_actions()
            self.actions[self.co_player_ids] = co_player_actions
        
        binding.vec_step(self.c_envs)
     
        self.tick+=1
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
                #print(log)
        if(self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0):
            self.tick = 0
            will_resample = 1
            if will_resample:
                binding.vec_close(self.c_envs)
                binding_tuple =  binding.shared(num_agents=self.num_agents_const, num_maps=self.num_maps, population_play = self.population_play)
                if self.population_play:
                    agent_offsets, map_ids, num_envs, self.ego_ids, co_player_ids = binding_tuple
                    self.co_player_ids = [item for sublist in co_player_ids for item in sublist]
                    self.num_ego_agents = len(self.ego_ids)
                    self.num_co_players = len(self.co_player_ids)
                    self.num_agents = self.total_agents = self.num_co_players + self.num_ego_agents 
                    # self.co_player_policy = load_drivenet("resources/drive/puffer_drive_weights.bin", self.num_co_players )
                    self.set_co_player_state()
                    env_co_player_ids = [[id - next(agent_offsets[i] for i in range(len(agent_offsets)-1) 
                                        if agent_offsets[i] <= id < agent_offsets[i+1]) 
                            for id in subset] for subset in co_player_ids]
                    
                    env_ego_ids = [
                        min(set(range(agent_offsets[i+1] - agent_offsets[i])) - set(subset)) 
                        if subset and set(range(agent_offsets[i+1] - agent_offsets[i])) - set(subset)
                        else 0
                        for i, subset in enumerate(env_co_player_ids)
                    ]

                    self.set_buffers() 
                else:
                    agent_offsets, map_ids, num_envs = binding_tuple
                    self.num_agents = self.num_agents_const
                    self.ego_ids = [i for i in range(agent_offsets[-1])]
                    env_co_player_ids = [[] for i in range(num_envs)]
                    env_ego_ids = [0 for i in range(num_envs)]

                
                self.agent_offsets = agent_offsets
                self.map_ids = map_ids
                self.num_envs = num_envs
                
                env_ids = []
                seed = np.random.randint(0, 2**32-1)
                for i in range(num_envs):
                    cur = agent_offsets[i]
                    nxt = agent_offsets[i+1]

                    env_id = binding.env_init(
                        self.observations[cur:nxt],
                        self.actions[cur:nxt],
                        self.rewards[cur:nxt],
                        self.terminals[cur:nxt],
                        self.truncations[cur:nxt],
                        seed,
                        human_agent_idx=self.human_agent_idx,
                        reward_vehicle_collision=self.reward_vehicle_collision,
                        reward_offroad_collision=self.reward_offroad_collision,
                        reward_goal_post_respawn=self.reward_goal_post_respawn,
                        reward_vehicle_collision_post_respawn=self.reward_vehicle_collision_post_respawn,
                        spawn_immunity_timer=self.spawn_immunity_timer,
                        map_id=map_ids[i],
                        use_rc=self.reward_conditioned,
                        use_ec=self.entropy_conditioned,
                        oracle_mode=self.oracle_mode,
                        max_agents=nxt-cur,
                        collision_weight_lb=self.collision_weight_lb,
                        collision_weight_ub=self.collision_weight_ub,
                        offroad_weight_lb=self.offroad_weight_lb,
                        offroad_weight_ub=self.offroad_weight_ub,
                        goal_weight_lb=self.goal_weight_lb,
                        goal_weight_ub=self.goal_weight_ub,
                        entropy_weight_lb=self.entropy_weight_lb,
                        entropy_weight_ub=self.entropy_weight_ub,
                        population_play = self.population_play,
                        num_co_players = len(env_co_player_ids[i]),
                        co_player_ids = env_co_player_ids[i],
                        ego_agent_id = env_ego_ids[i],
                    )
                    env_ids.append(env_id)

                self.c_envs = binding.vectorize(*env_ids)

                binding.vec_reset(self.c_envs, seed)
                self.terminals[:] = 1

        if self.rewards.sum() !=0:
            pass
        return (self.observations[self.ego_ids], self.rewards[self.ego_ids],
                self.terminals[self.ego_ids], self.truncations[self.ego_ids], info)
        
   

    def render(self):
        binding.vec_render(self.c_envs, 0)
        
    def close(self):
        binding.vec_close(self.c_envs)

def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1['x'] - p3['x']) * (p2['y'] - p1['y']) - (p1['x'] - p2['x']) * (p3['y'] - p1['y']))

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
    with open(output_file, 'wb') as f:
        # Count total entities
        print(len(map_data.get('objects', [])))
        print(len(map_data.get('roads', [])))
        num_objects = len(map_data.get('objects', []))
        num_roads = len(map_data.get('roads', []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack('i', num_objects))
        f.write(struct.pack('i', num_roads))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get('objects', []):
            # Write base entity data
            obj_type = obj.get('type', 1)
            if(obj_type =='vehicle'):
                obj_type = 1
            elif(obj_type == 'pedestrian'):
                obj_type = 2;
            elif(obj_type == 'cyclist'):
                obj_type = 3;
            f.write(struct.pack('i', obj_type))  # type
            # f.write(struct.pack('i', obj.get('id', 0)))   # id  
            f.write(struct.pack('i', trajectory_length))                  # array_size
            # Write position arrays
            positions = obj.get('position', [])
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                f.write(struct.pack('f', float(pos.get('x', 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                f.write(struct.pack('f', float(pos.get('y', 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                f.write(struct.pack('f', float(pos.get('z', 0.0))))

            # Write velocity arrays
            velocities = obj.get('velocity', [])
            for arr, key in [(velocities, 'x'), (velocities, 'y'), (velocities, 'z')]:
                for i in range(trajectory_length):
                    vel = arr[i] if i < len(arr) else {'x': 0.0, 'y': 0.0, 'z': 0.0}
                    f.write(struct.pack('f', float(vel.get(key, 0.0))))
            
            # Write heading and valid arrays
            headings = obj.get('heading', [])
            f.write(struct.pack(f'{trajectory_length}f', *[float(headings[i]) if i < len(headings) else 0.0 for i in range(trajectory_length)]))
            
            valids = obj.get('valid', [])
            f.write(struct.pack(f'{trajectory_length}i', *[int(valids[i]) if i < len(valids) else 0 for i in range(trajectory_length)]))
            
            # Write scalar fields
            f.write(struct.pack('f', float(obj.get('width', 0.0))))
            f.write(struct.pack('f', float(obj.get('length', 0.0))))
            f.write(struct.pack('f', float(obj.get('height', 0.0))))
            goal_pos = obj.get('goalPosition', {'x': 0, 'y': 0, 'z': 0})  # Get goalPosition object with default
            f.write(struct.pack('f', float(goal_pos.get('x', 0.0))))  # Get x value
            f.write(struct.pack('f', float(goal_pos.get('y', 0.0))))  # Get y value
            f.write(struct.pack('f', float(goal_pos.get('z', 0.0))))  # Get z value
            f.write(struct.pack('i', obj.get('mark_as_expert', 0)))
        
        # Write roads
        for idx, road in enumerate(map_data.get('roads', [])):
            geometry = road.get('geometry', [])
            road_type = road.get('map_element_id', 0)
            road_type_word = road.get('type', 0)
            if(road_type_word == "lane"):
                road_type = 2
            elif(road_type_word == "road_edge"):
                road_type = 15
            # breakpoint()
            if(len(geometry) > 10 and road_type <=16):
                geometry = simplify_polyline(geometry, .1)
            size = len(geometry)
            # breakpoint()
            if(road_type >=0 and road_type <=3):
                road_type = 4
            elif(road_type >=5 and road_type <=13):
                road_type = 5
            elif(road_type >=14 and road_type <=16):
                road_type = 6
            elif(road_type == 17):
                road_type = 7
            elif(road_type == 18):
                road_type = 8
            elif(road_type == 19):
                road_type = 9
            elif(road_type == 20):
                road_type = 10
            # Write base entity data
            f.write(struct.pack('i', road_type))  # type
            # f.write(struct.pack('i', road.get('id', 0)))    # id
            f.write(struct.pack('i', size))                 # array_size
            
            # Write position arrays
            for coord in ['x', 'y', 'z']:
                for point in geometry:
                    f.write(struct.pack('f', float(point.get(coord, 0.0))))
            # Write scalar fields
            f.write(struct.pack('f', float(road.get('width', 0.0))))
            f.write(struct.pack('f', float(road.get('length', 0.0))))
            f.write(struct.pack('f', float(road.get('height', 0.0))))
            goal_pos = road.get('goalPosition', {'x': 0, 'y': 0, 'z': 0})  # Get goalPosition object with default
            f.write(struct.pack('f', float(goal_pos.get('x', 0.0))))  # Get x value
            f.write(struct.pack('f', float(goal_pos.get('y', 0.0))))  # Get y value
            f.write(struct.pack('f', float(goal_pos.get('z', 0.0))))  # Get z value
            f.write(struct.pack('i', road.get('mark_as_expert', 0)))

def load_map(map_name, binary_output=None):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, 'r') as f:
        map_data = json.load(f)
    
    if binary_output:
        save_map_binary(map_data, binary_output)

def process_all_maps():
    """Process all maps and save them as binaries"""
    import os
    from pathlib import Path

    # Create the binaries directory if it doesn't exist
    binary_dir = Path("resources/drive/binaries")
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Path to the training data
    data_dir = Path("data/processed_big/training")
    
    # Get all JSON files in the training directory
    json_files = sorted(data_dir.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    for i, map_path in enumerate(json_files[:10000]):
        binary_file = f"map_{i:03d}.bin"  # Use zero-padded numbers for consistent sorting
        binary_path = binary_dir / binary_file
        
        print(f"Processing {map_path.name} -> {binary_file}")
        # try:
        load_map(str(map_path), str(binary_path))
        # except Exception as e:
        #     print(f"Error processing {map_path.name}: {e}")

def test_performance(timeout=10, atn_cache=1024, num_agents=1024):
    import time

    env = Drive(num_agents=num_agents)
    env.reset()
    tick = 0
    num_agents = 1024
    actions = np.stack([
        np.random.randint(0, space.n + 1, (atn_cache, num_agents))
        for space in env.single_action_space
    ], axis=-1)

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]         
        env.step(atn)
        tick += 1

    print(f'SPS: {num_agents * tick / (time.time() - start)}')
    env.close()


if __name__ == '__main__':
    # test_performance()
    process_all_maps()
