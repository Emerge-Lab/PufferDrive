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
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 3  # 6 is the number of one-hot encoded categorie
        self.traffic_control_features = env.traffic_control_features
        self.traffic_control_features_after_onehot = env.traffic_control_features + 3
        self.max_traffic_controls = env.max_traffic_controls
        self.target_type = env.target_type
        if self.target_type == 1 or self.target_type == 2:
            self.max_gps_objects = env.max_gps_objects
            self.gps_features = env.gps_features

        if self.target_type == 0 or self.target_type == 2:
            self.ego_dim = 8 if env.dynamics_model == "jerk" else 5
        else:
            self.ego_dim = 6 if env.dynamics_model == "jerk" else 3

        # Reward conditioning
        self.num_reward_coefs = getattr(env, "num_reward_coefs", 0)

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
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
        # self.traffic_light_encoder = nn.Sequential(
        #     pufferlib.pytorch.layer_init(nn.Linear(self.traffic_control_features_after_onehot, input_size)),
        #     nn.LayerNorm(input_size),
        #     # nn.ReLU(),
        #     pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        # )

        num_feature_sets = 3
        if self.num_reward_coefs > 0:
            self.conditioning_encoder = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(self.num_reward_coefs, input_size)),
                nn.LayerNorm(input_size),
                pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
            )
            num_feature_sets += 1
        if self.target_type == 1 or self.target_type == 2:
            self.gps_encoder = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(self.gps_features, input_size)),
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
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        # traffic_light_dim = self.max_traffic_controls * self.traffic_control_features

        # Unflatten observations
        if self.target_type in [1, 2]:
            gps_dim = self.max_gps_objects * self.gps_features
        else:
            gps_dim = 0

        slide_idx = ego_dim
        ego_obs = observations[:, :slide_idx]

        # Reward conditioning coefficients (after ego, before gps)
        if self.num_reward_coefs > 0:
            coef_obs = observations[:, slide_idx : slide_idx + self.num_reward_coefs]
            slide_idx += self.num_reward_coefs

        gps_obs = observations[:, slide_idx : slide_idx + gps_dim]
        slide_idx += gps_dim
        partner_obs = observations[:, slide_idx : slide_idx + partner_dim]
        slide_idx += partner_dim
        road_obs = observations[:, slide_idx : slide_idx + road_dim]
        slide_idx += road_dim
        # traffic_light_obs = observations[:, slide_idx : slide_idx + traffic_light_dim]

        # Reshape object observations
        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        # traffic_light_obs = traffic_light_obs.view(-1, self.max_traffic_controls, self.traffic_control_features)
        if self.target_type == 1 or self.target_type == 2:
            gps_objects = gps_obs.view(-1, self.max_gps_objects, self.gps_features)

        # Road object one-hot encoding
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot((road_categorical + 1).long(), num_classes=4)  # Shape: [batch, ROAD_MAX_OBJECTS, 4]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        # # Traffic light one-hot encoding
        # traffic_light_continuous = traffic_light_obs[:, :, : self.traffic_control_features - 1]
        # traffic_light_categorical = traffic_light_obs[:, :, self.traffic_control_features - 1]
        # # normalize_traffic_light_state returns -1, 0, 1, or 2. Shift to [0, 3] for one-hot
        # traffic_light_onehot = F.one_hot((traffic_light_categorical + 1).long(), num_classes=4)  # Shape: [batch, 10, 4]
        # traffic_light_objects = torch.cat([traffic_light_continuous, traffic_light_onehot], dim=2)

        # Encode each observation type
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)
        # traffic_light_features, _ = self.traffic_light_encoder(traffic_light_objects).max(dim=1)

        # Build feature list
        feature_list = [ego_features, road_features, partner_features]

        # Add conditioning features if enabled
        if self.num_reward_coefs > 0:
            coef_features = self.conditioning_encoder(coef_obs)
            feature_list.append(coef_features)

        # Add GPS features if enabled
        if self.target_type == 1 or self.target_type == 2:
            gps_features, _ = self.gps_encoder(gps_objects).max(dim=1)
            feature_list.append(gps_features)

        concat_features = torch.cat(feature_list, dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
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
