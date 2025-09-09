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
            population_play = False):

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
        self.num_obs = 7 + 63*7 + 200*7
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1,
            shape=(self.num_obs,), dtype=np.float32)
        self.num_agents_const = num_agents
        self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])

        self.population_play = population_play
        # self.single_action_space = gymnasium.spaces.Box(
        #     low=-1, high=1, shape=(2,), dtype=np.float32
        # )
        # Check if resources directory exists
        binary_path = "resources/drive/binaries/map_000.bin"
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Required directory {binary_path} not found. Please ensure the Drive maps are downloaded and installed correctly per docs.")
        
        binding_tuple =  binding.shared(num_agents=num_agents, num_maps=num_maps, population_play = population_play)
        
        if self.population_play:
            agent_offsets, map_ids, num_envs, self.ego_ids, co_player_ids = binding_tuple
            self.co_player_ids = [item for sublist in co_player_ids for item in sublist]
            self.num_ego_agents = num_agents
            self.num_co_players = len(self.co_player_ids)
            self.num_agents = self.total_agents = self.num_co_players + self.num_ego_agents 
            self.co_player_policy = load_drivenet("/home/charliemolony/adaptive_driving_agent/PufferLib/pufferlib/resources/drive/puffer_drive_weights.bin", self.num_co_players )
        else:
            agent_offsets, map_ids, num_envs = binding_tuple
            self.num_agents = self.num_agents_const
            self.ego_ids = [i for i in range(agent_offsets[-1])]

                      
       
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
                human_agent_idx=human_agent_idx,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                reward_goal_post_respawn=reward_goal_post_respawn,
                reward_vehicle_collision_post_respawn=reward_vehicle_collision_post_respawn,
                spawn_immunity_timer=spawn_immunity_timer,
                map_id=map_ids[i],
                max_agents = nxt-cur
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

        if self.population_play:
            self.num_agents = self.num_ego_agents



    def set_buffers(self):

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


    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []
    
    def get_co_player_actions(self):
        co_player_obs = self.observations[self.co_player_ids]
        co_player_actions, co_player_values = self.co_player_policy.get_actions(torch.from_numpy(co_player_obs))
        return co_player_actions


    def step(self, ego_actions):
        self.terminals[:] = 0
        self.num_agents = self.total_agents
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
                if self.population_play:
                    agent_offsets, map_ids, num_envs, self.ego_ids, co_player_ids = binding.shared(num_agents=self.num_agents_const, num_maps=self.num_maps)
                    self.co_player_ids = [item for sublist in co_player_ids for item in sublist]
                    self.num_ego_agents = len(self.ego_ids)
                    self.num_co_players = len(self.co_player_ids)
                    self.num_agents = self.total_agents = self.num_co_players + self.num_ego_agents 
                else:
                    agent_offsets, map_ids, num_envs = binding.shared(num_agents=self.num_agents_const, num_maps=self.num_maps)
                    self.num_agents = self.num_agents_const
                    self.ego_ids = [i for i in range(agent_offsets[-1])]


                
                self.agent_offsets = agent_offsets
                self.map_ids = map_ids
                self.num_envs = num_envs
                self.set_buffers()
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
                        max_agents = nxt-cur
                    )
                    env_ids.append(env_id)

                self.c_envs = binding.vectorize(*env_ids)

                binding.vec_reset(self.c_envs, seed)
                self.terminals[:] = 1
                self.num_co_players = len(self.co_player_ids)
                self.co_player_policy = load_drivenet("/home/charliemolony/adaptive_driving_agent/PufferLib/pufferlib/resources/drive/puffer_drive_weights.bin", self.num_co_players )

        try:
            return (self.observations[self.ego_ids], self.rewards[self.ego_ids],
                self.terminals[self.ego_ids], self.truncations[self.ego_ids], info)
        
        except Exception as e:
            print(f"observations: {self.observations.shape}")
            print(f"ego ids: {self.ego_ids}")
            raise e

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct

class DriveNet(nn.Module):
    def __init__(self, weights_path, num_agents=1):
        super(DriveNet, self).__init__()
        self.num_agents = num_agents
        self.hidden_size = 256
        self.input_size = 64
        
        # Load weights from binary file
        self.weights_data = self.load_weights(weights_path)
        self.weight_idx = 0
        
        # Initialize layers
        self.ego_encoder = self._make_linear(7, self.input_size)
        self.ego_layernorm = self._make_layernorm(self.input_size)
        self.ego_encoder_two = self._make_linear(self.input_size, self.input_size)
        
        self.road_encoder = self._make_linear(13, self.input_size)
        self.road_layernorm = self._make_layernorm(self.input_size)
        self.road_encoder_two = self._make_linear(self.input_size, self.input_size)
        
        self.partner_encoder = self._make_linear(7, self.input_size)
        self.partner_layernorm = self._make_layernorm(self.input_size)
        self.partner_encoder_two = self._make_linear(self.input_size, self.input_size)
        
        self.shared_embedding = self._make_linear(self.input_size * 3, self.hidden_size)
        self.actor = self._make_linear(self.hidden_size, 20)
        self.value_fn = self._make_linear(self.hidden_size, 1)
        
        # LSTM layer
        self.lstm = self._make_lstm(self.hidden_size, 256)
        
        # Initialize LSTM hidden states
        self.register_buffer('lstm_h', torch.zeros(num_agents, 256))
        self.register_buffer('lstm_c', torch.zeros(num_agents, 256))
        
        # Multidiscrete action sizes
        self.action_sizes = [7, 13]
    
    def load_weights(self, filename):
        """Load weights from binary file"""
        with open(filename, 'rb') as f:
            data = f.read()
        
        # Assuming weights are stored as float32
        num_floats = len(data) // 4
        weights = struct.unpack(f'{num_floats}f', data)
        return torch.tensor(weights, dtype=torch.float32)
    
    def _get_weights(self, size):
        """Get next chunk of weights"""
        start_idx = self.weight_idx
        end_idx = start_idx + size
        weights = self.weights_data[start_idx:end_idx]
        self.weight_idx = end_idx
        return weights
    
    def _make_linear(self, in_features, out_features):
        """Create linear layer with loaded weights"""
        layer = nn.Linear(in_features, out_features, bias=True)
        
        # Load weights and bias
        weight_size = in_features * out_features
        bias_size = out_features
        
        weights = self._get_weights(weight_size).reshape(out_features, in_features)
        bias = self._get_weights(bias_size)
        
        layer.weight.data = weights
        layer.bias.data = bias
        
        return layer
    
    def _make_layernorm(self, normalized_shape):
        """Create layer norm with loaded weights"""
        layer = nn.LayerNorm(normalized_shape)
        
        # Load weight and bias
        weight = self._get_weights(normalized_shape)
        bias = self._get_weights(normalized_shape)
        
        layer.weight.data = weight
        layer.bias.data = bias
        
        return layer
    
    def _make_lstm(self, input_size, hidden_size):
        """Create LSTM with loaded weights"""
        lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # LSTM has 4 gates: input, forget, cell, output
        # Each gate has input-to-hidden and hidden-to-hidden weights
        for name, param in lstm.named_parameters():
            if 'weight_ih' in name:  # input to hidden weights
                weight_size = param.numel()
                weights = self._get_weights(weight_size).reshape(param.shape)
                param.data = weights
            elif 'weight_hh' in name:  # hidden to hidden weights  
                weight_size = param.numel()
                weights = self._get_weights(weight_size).reshape(param.shape)
                param.data = weights
            elif 'bias' in name:  # bias
                bias_size = param.numel()
                bias = self._get_weights(bias_size)
                param.data = bias
        
        return lstm
    
    def forward(self, observations):
        """
        Forward pass
        observations: tensor of shape [num_agents, obs_size] where obs_size = 7 + 63*7 + 200*7
        """
        batch_size = observations.shape[0]
        
        # Parse observations
        obs_self = observations[:, :7]  # [batch, 7]
        obs_partner = observations[:, 7:7+63*7].reshape(batch_size, 63, 7)  # [batch, 63, 7]
        obs_road_raw = observations[:, 7+63*7:].reshape(batch_size, 200, 7)  # [batch, 200, 7]
        
        # Process road observations - add one-hot encoding for last feature
        obs_road = torch.zeros(batch_size, 200, 13, device=observations.device)
        obs_road[:, :, :7] = obs_road_raw
        
        # One-hot encode the 7th feature (index 6) into positions 7-13
        road_categories = obs_road_raw[:, :, 6].long()  # [batch, 200]
        road_categories = torch.clamp(road_categories, 0, 6)  # Ensure valid range
        obs_road.scatter_(2, road_categories.unsqueeze(2) + 6, 1.0)
        
        # Process ego vehicle
        ego_encoded = self.ego_encoder(obs_self)  # [batch, 64]
        ego_encoded = self.ego_layernorm(ego_encoded)
        ego_encoded = self.ego_encoder_two(ego_encoded)  # [batch, 64]
        
        # Process partner vehicles
        partner_encoded = self.partner_encoder(obs_partner.reshape(-1, 7))  # [batch*63, 64]
        partner_encoded = partner_encoded.reshape(batch_size, 63, -1)  # [batch, 63, 64]
        partner_encoded = self.partner_layernorm(partner_encoded.reshape(-1, 64)).reshape(batch_size, 63, -1)
        partner_encoded = self.partner_encoder_two(partner_encoded.reshape(-1, 64))  # [batch*63, 64]
        partner_encoded = partner_encoded.reshape(batch_size, 63, -1)  # [batch, 63, 64]
        
        # Max pool over partner vehicles
        partner_max, _ = torch.max(partner_encoded, dim=1)  # [batch, 64]
        
        # Process road objects
        road_encoded = self.road_encoder(obs_road.reshape(-1, 13))  # [batch*200, 64]
        road_encoded = road_encoded.reshape(batch_size, 200, -1)  # [batch, 200, 64]
        road_encoded = self.road_layernorm(road_encoded.reshape(-1, 64)).reshape(batch_size, 200, -1)
        road_encoded = self.road_encoder_two(road_encoded.reshape(-1, 64))  # [batch*200, 64]
        road_encoded = road_encoded.reshape(batch_size, 200, -1)  # [batch, 200, 64]
        
        # Max pool over road objects
        road_max, _ = torch.max(road_encoded, dim=1)  # [batch, 64]
        
        # Concatenate all features
        combined = torch.cat([ego_encoded, road_max, partner_max], dim=1)  # [batch, 192]
        
        # Apply GELU activation
        combined = F.gelu(combined)
        
        # Shared embedding
        shared = self.shared_embedding(combined)  # [batch, 256]
        shared = F.relu(shared)
        
        # LSTM
        lstm_input = shared.unsqueeze(1)  # [batch, 1, 256]
        lstm_out, (h_new, c_new) = self.lstm(lstm_input, (self.lstm_h.unsqueeze(0), self.lstm_c.unsqueeze(0)))
        
        # Update hidden states
        self.lstm_h = h_new.squeeze(0)
        self.lstm_c = c_new.squeeze(0)
        
        # Actor and value outputs
        actor_logits = self.actor(self.lstm_h)  # [batch, 20]
        values = self.value_fn(self.lstm_h)  # [batch, 1]
        
        return actor_logits, values
    
    def get_actions(self, observations):
        """Get discrete actions using multidiscrete sampling"""
        with torch.no_grad():
            actor_logits, values = self.forward(observations)
            
            # Split logits for multidiscrete actions
            logit_splits = [7, 13]  # action_sizes
            action_logits = torch.split(actor_logits, logit_splits, dim=1)
            
            actions = []
            for logits in action_logits:
                # Sample from categorical distribution
                probs = F.softmax(logits, dim=1)
                action = torch.multinomial(probs, 1).squeeze(1)
                actions.append(action)
            
            return torch.stack(actions, dim=1), values
    
    def reset_lstm_state(self):
        """Reset LSTM hidden states"""
        self.lstm_h.zero_()
        self.lstm_c.zero_()

# Usage example
def load_drivenet(weights_path, num_agents=1):
    """Load DriveNet model"""
    model = DriveNet(weights_path, num_agents)
    model.eval()  # Set to evaluation mode
    return model



if __name__ == '__main__':
    # test_performance()
    process_all_maps()
