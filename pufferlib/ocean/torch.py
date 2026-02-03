from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_lane_objects = env.max_lane_objects
        self.max_boundary_objects = env.max_boundary_objects
        self.road_features = env.road_features
        self.traffic_control_features = env.traffic_control_features
        self.max_traffic_controls = env.max_traffic_controls
        # Ego = pure kinematic state only (no goal/gps)
        self.ego_dim = env.ego_features

        # Conditioning = reward_coefs + target waypoints (all adjacent in C obs layout)
        self.num_reward_coefs = getattr(env, "num_reward_coefs", 0)
        target_dim = getattr(env, "target_dim", 0)
        self.conditioning_dim = self.num_reward_coefs + target_dim

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.lane_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.boundary_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        # # Traffic light features: 2 position + 4 one-hot state (state normalized to -1,0,1,2 -> shifted to 0,1,2,3)
        # self.traffic_control_features_after_onehot = 2 + 4
        # self.traffic_light_encoder = nn.Sequential(
        #     pufferlib.pytorch.layer_init(nn.Linear(self.traffic_control_features_after_onehot, input_size)),
        #     nn.LayerNorm(input_size),
        #     pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        # )

        num_feature_sets = 4  # 5 if traffic lights are used
        if self.conditioning_dim > 0:
            self.conditioning_encoder = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(self.conditioning_dim, input_size)),
                nn.LayerNorm(input_size),
                # nn.ReLU(),
                pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
            )
            num_feature_sets += 1

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(num_feature_sets * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        partner_dim = self.max_partner_objects * self.partner_features
        lane_dim = self.max_lane_objects * self.road_features
        boundary_dim = self.max_boundary_objects * self.road_features
        # traffic_dim = self.max_traffic_controls * self.traffic_control_features

        # Slice obs: ego | conditioning | partners | lanes | boundaries | traffic_lights
        slide_idx = self.ego_dim
        ego_obs = observations[:, :slide_idx]

        if self.conditioning_dim > 0:
            conditioning_obs = observations[:, slide_idx : slide_idx + self.conditioning_dim]
        slide_idx += self.conditioning_dim

        partner_obs = observations[:, slide_idx : slide_idx + partner_dim]
        slide_idx += partner_dim
        lane_obs = observations[:, slide_idx : slide_idx + lane_dim]
        slide_idx += lane_dim
        boundary_obs = observations[:, slide_idx : slide_idx + boundary_dim]
        # slide_idx += boundary_dim
        # traffic_light_obs = observations[:, slide_idx : slide_idx + traffic_dim]

        # Reshape object observations
        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        lane_objects = lane_obs.view(-1, self.max_lane_objects, self.road_features)
        boundary_objects = boundary_obs.view(-1, self.max_boundary_objects, self.road_features)
        # traffic_light_raw = traffic_light_obs.view(-1, self.max_traffic_controls, self.traffic_control_features)

        # Traffic light one-hot encoding: features are [rel_x, rel_y, state]
        # state is normalized to -1, 0, 1, 2 -> shift to 0, 1, 2, 3 for one-hot
        # traffic_light_continuous = traffic_light_raw[:, :, :2]  # rel_x, rel_y
        # traffic_light_categorical = traffic_light_raw[:, :, 2]  # state
        # traffic_light_onehot = F.one_hot((traffic_light_categorical + 1).long(), num_classes=4).float()
        # traffic_light_objects = torch.cat([traffic_light_continuous, traffic_light_onehot], dim=2)

        # Encode each observation type
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        lane_features, _ = self.lane_encoder(lane_objects).max(dim=1)
        boundary_features, _ = self.boundary_encoder(boundary_objects).max(dim=1)
        # traffic_light_features, _ = self.traffic_light_encoder(traffic_light_objects).max(dim=1)

        feature_list = [ego_features, lane_features, boundary_features, partner_features]  # , traffic_light_features]

        if self.conditioning_dim > 0:
            conditioning_features = self.conditioning_encoder(conditioning_obs)
            feature_list.append(conditioning_features)

        concat_features = torch.cat(feature_list, dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value
