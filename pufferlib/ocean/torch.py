from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models
from pufferlib.ocean.drive import binding

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401

Recurrent = pufferlib.models.LSTMWrapper


class DriveBackbone(nn.Module):
    """
    Neural network backbone
    Architecture features:
      - Split Actor/Critic (configurable)
    """

    def _create_encoder(self, in_features, input_size, encoder_gigaflow, dropout=0.0):
        if encoder_gigaflow:
            return nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(in_features, input_size)),
                nn.LayerNorm(input_size),
                nn.Tanh(),
                nn.Dropout(dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
            )
        else:
            return nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(in_features, input_size)),
                nn.LayerNorm(input_size),
                pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
            )

    def __init__(
        self,
        env,
        input_size,
        backbone_hidden_size,
        backbone_num_layers,
        ego_dim,
        encoder_gigaflow,
        dropout,
        strip_last_partner_feature=False,
    ):
        super().__init__()

        # Observation dimensions from environment config
        self.max_partner_observations = env.max_partner_observations
        self.partner_features_count = env.partner_features
        self.strip_last_partner_feature = strip_last_partner_feature
        self.partner_encoder_features = (
            self.partner_features_count - 1 if strip_last_partner_feature else self.partner_features_count
        )
        # Road features size (lanes + boundaries)
        self.max_lane_segment_observations = env.max_lane_segment_observations
        self.max_boundary_segment_observations = env.max_boundary_segment_observations
        self.road_features_count = env.road_features
        # Traffic control size
        self.max_traffic_control_observations = env.max_traffic_control_observations
        self.traffic_control_features_count = env.traffic_control_features
        self.traffic_control_continuous_features = env.traffic_control_features - 2
        self.traffic_control_features_after_onehot = (
            self.traffic_control_continuous_features
            + binding.NUM_TRAFFIC_CONTROL_TYPES
            + binding.NUM_TRAFFIC_CONTROL_STATES
        )
        # Conditioning size (reward coefficients + target info)
        self.conditioning_dim = env.num_reward_coefs + env.target_dim

        num_feature_sets = 1

        # 1. observations Encoders
        # Each encoder projects raw features into a common input_size embedding space
        self.ego_encoder = self._create_encoder(ego_dim, input_size, encoder_gigaflow)
        if self.max_lane_segment_observations > 0:
            self.lane_encoder = self._create_encoder(
                self.road_features_count,
                input_size,
                encoder_gigaflow,
                dropout=dropout,
            )
            num_feature_sets += 1
        if self.max_boundary_segment_observations > 0:
            self.boundary_encoder = self._create_encoder(
                self.road_features_count,
                input_size,
                encoder_gigaflow,
                dropout=dropout,
            )
            num_feature_sets += 1
        if self.max_partner_observations > 0:
            self.partner_encoder = self._create_encoder(
                self.partner_encoder_features,
                input_size,
                encoder_gigaflow,
            )
            num_feature_sets += 1
        if self.max_traffic_control_observations > 0:
            self.traffic_control_encoder = self._create_encoder(
                self.traffic_control_features_after_onehot,
                input_size,
                encoder_gigaflow,
            )
            num_feature_sets += 1
        if self.conditioning_dim > 0:
            self.conditioning_encoder = self._create_encoder(self.conditioning_dim, input_size, encoder_gigaflow)
            num_feature_sets += 1

        # 2. Main Backbone MLP
        backbone_layers = []
        bb_in = num_feature_sets * input_size
        for _ in range(backbone_num_layers):
            backbone_layers.append(nn.GELU())
            backbone_layers.append(pufferlib.pytorch.layer_init(nn.Linear(bb_in, backbone_hidden_size)))
            bb_in = backbone_hidden_size
        # Add final GELU before heads
        backbone_layers.append(nn.GELU())
        self.backbone = nn.Sequential(*backbone_layers)
        self.out_dim = backbone_hidden_size if backbone_num_layers > 0 else num_feature_sets * input_size

    def forward(self, observations, ego_dim):
        # Extract and slice observations from the flat buffer
        partner_dim = self.max_partner_observations * self.partner_features_count
        lane_dim = self.max_lane_segment_observations * self.road_features_count
        boundary_dim = self.max_boundary_segment_observations * self.road_features_count
        traffic_control_dim = self.max_traffic_control_observations * self.traffic_control_features_count

        slide_idx = ego_dim
        ego_observations = observations[:, :slide_idx]

        conditioning_observations = observations[:, slide_idx : slide_idx + self.conditioning_dim]
        slide_idx += self.conditioning_dim

        partner_observations = observations[:, slide_idx : slide_idx + partner_dim]
        slide_idx += partner_dim

        lane_observations = observations[:, slide_idx : slide_idx + lane_dim]
        slide_idx += lane_dim

        boundary_observations = observations[:, slide_idx : slide_idx + boundary_dim]
        slide_idx += boundary_dim

        traffic_control_observations = observations[:, slide_idx : slide_idx + traffic_control_dim]

        # Encode Ego State
        ego_features = self.ego_encoder(ego_observations)

        feature_list = [ego_features]

        # Encode Lanes and Boundaries separately
        if self.max_lane_segment_observations > 0:
            lane_objects = lane_observations.view(-1, self.max_lane_segment_observations, self.road_features_count)
            lane_features, _ = self.lane_encoder(lane_objects).max(dim=1)
            feature_list.append(lane_features)
        if self.max_boundary_segment_observations > 0:
            boundary_objects = boundary_observations.view(
                -1, self.max_boundary_segment_observations, self.road_features_count
            )

            boundary_features, _ = self.boundary_encoder(boundary_objects).max(dim=1)
            feature_list.append(boundary_features)

        # Encode Partners
        if self.max_partner_observations > 0:
            partner_objects = partner_observations.view(-1, self.max_partner_observations, self.partner_features_count)
            if self.strip_last_partner_feature:
                partner_objects = partner_objects[..., :-1]
            partner_encoded = self.partner_encoder(partner_objects)
            partner_features, _ = partner_encoded.max(dim=1)
            feature_list.append(partner_features)

        # Encode Traffic Controls
        if self.max_traffic_control_observations > 0:
            traffic_control_objects = traffic_control_observations.view(
                -1, self.max_traffic_control_observations, self.traffic_control_features_count
            )
            traffic_control_continuous = traffic_control_objects[:, :, : self.traffic_control_continuous_features]
            traffic_control_type = traffic_control_objects[:, :, self.traffic_control_continuous_features]
            traffic_control_state = traffic_control_objects[:, :, self.traffic_control_continuous_features + 1]
            traffic_control_type_onehot = F.one_hot(
                traffic_control_type.long(),
                num_classes=binding.NUM_TRAFFIC_CONTROL_TYPES,
            ).float()
            traffic_control_state_onehot = F.one_hot(
                traffic_control_state.long(),
                num_classes=binding.NUM_TRAFFIC_CONTROL_STATES,
            ).float()
            traffic_control_objects = torch.cat(
                [traffic_control_continuous, traffic_control_type_onehot, traffic_control_state_onehot],
                dim=2,
            )
            traffic_control_features, _ = self.traffic_control_encoder(traffic_control_objects).max(dim=1)
            feature_list.append(traffic_control_features)

        # Add optional features if enabled
        if self.conditioning_dim > 0:
            conditioning_features = self.conditioning_encoder(conditioning_observations)
            feature_list.append(conditioning_features)

        # Concatenate all features and pass through main backbone
        concat_features = torch.cat(feature_list, dim=1)
        return self.backbone(concat_features)


class Drive(nn.Module):
    def __init__(
        self,
        env,
        input_size: int,
        backbone_hidden_size: int,
        backbone_num_layers: int,
        actor_hidden_size: int,
        actor_num_layers: int,
        critic_hidden_size: int,
        critic_num_layers: int,
        encoder_gigaflow: bool,
        dropout: int,
        split_network: bool,
    ):
        super().__init__()

        # Configuration flags from policy kwargs
        self.split_network = split_network
        self.ego_dim = env.ego_features

        # Prepare arguments for the Backbone
        backbone_args = {
            "env": env,
            "input_size": input_size,
            "backbone_hidden_size": backbone_hidden_size,
            "backbone_num_layers": backbone_num_layers,
            "ego_dim": self.ego_dim,
            "encoder_gigaflow": encoder_gigaflow,
            "dropout": dropout,
        }

        # Instantiate backbones
        self.actor_backbone = DriveBackbone(**backbone_args)

        # If split_network is True, create a separate backbone for the critic.
        # Otherwise, share the same backbone for both.
        if self.split_network:
            self.critic_backbone = DriveBackbone(**backbone_args)
        else:
            self.critic_backbone = self.actor_backbone

        # Setup action and value heads
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        # n-layer MLP for actor head (num_layers = number of hidden layers)
        backbone_out_dim = self.actor_backbone.out_dim
        actor_head_layers = []
        actor_in = backbone_out_dim
        for _ in range(actor_num_layers):
            actor_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(actor_in, actor_hidden_size)))
            actor_head_layers.append(nn.ReLU())
            actor_in = actor_hidden_size
        actor_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(actor_in, sum(self.atn_dim)), std=0.01))
        self.actor_head = nn.Sequential(*actor_head_layers)

        # n-layer MLP for critic head (num_layers = number of hidden layers)
        critic_head_layers = []
        critic_in = backbone_out_dim
        for _ in range(critic_num_layers):
            critic_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(critic_in, critic_hidden_size)))
            critic_head_layers.append(nn.ReLU())
            critic_in = critic_hidden_size
        critic_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(critic_in, 1), std=1))
        self.critic_head = nn.Sequential(*critic_head_layers)

    def forward(self, observations, state=None):
        """
        Forward pass handling both Actor and Critic inference.
        """
        # Forward pass for actor
        actor_hidden = self.actor_backbone(observations, self.ego_dim)

        # Forward pass for critic (may use separate backbone)
        if self.split_network:
            critic_hidden = self.critic_backbone(observations, self.ego_dim)
        else:
            critic_hidden = actor_hidden

        # Compute actions
        if self.is_continuous:
            params = self.actor_head(actor_hidden)
            loc, scale = torch.split(params, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            actions = torch.distributions.Normal(loc, std)
        else:
            actions = torch.split(self.actor_head(actor_hidden), self.atn_dim, dim=1)

        # Compute value
        value = self.critic_head(critic_hidden)

        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    # Required for PufferLib recurrent wrappers
    def encode_observations(self, observations, state=None):
        assert not self.split_network, "LSTM wrapper doesn't support split_network=True"
        return self.actor_backbone(observations, self.ego_dim)

    def decode_actions(self, hidden):
        """
        USE ONLY FOR LSTM WRAPPER.
        Decodes actions and value from the hidden state.
        Args:
            hidden: The hidden state for the actor (policy).
        """
        if self.is_continuous:
            parameters = self.actor_head(hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor_head(hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.critic_head(hidden)

        return action, value


class TargetDrive(nn.Module):
    def __init__(
        self,
        env,
        input_size: int,
        backbone_hidden_size: int,
        backbone_num_layers: int,
        actor_hidden_size: int,
        actor_num_layers: int,
        critic_hidden_size: int,
        critic_num_layers: int,
        encoder_gigaflow: bool,
        dropout: int,
        split_network: bool,
    ):
        super().__init__()

        self.split_network = split_network
        self.ego_dim = env.ego_features

        backbone_args = {
            "env": env,
            "input_size": input_size,
            "backbone_hidden_size": backbone_hidden_size,
            "backbone_num_layers": backbone_num_layers,
            "ego_dim": self.ego_dim,
            "encoder_gigaflow": encoder_gigaflow,
            "dropout": dropout,
            "strip_last_partner_feature": True,
        }

        self.actor_backbone = DriveBackbone(**backbone_args)

        if self.split_network:
            self.critic_backbone = DriveBackbone(**backbone_args)
        else:
            self.critic_backbone = self.actor_backbone

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        backbone_out_dim = self.actor_backbone.out_dim
        actor_head_layers = []
        actor_in = backbone_out_dim
        for _ in range(actor_num_layers):
            actor_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(actor_in, actor_hidden_size)))
            actor_head_layers.append(nn.ReLU())
            actor_in = actor_hidden_size
        actor_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(actor_in, sum(self.atn_dim)), std=0.01))
        self.actor_head = nn.Sequential(*actor_head_layers)

        critic_head_layers = []
        critic_in = backbone_out_dim
        for _ in range(critic_num_layers):
            critic_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(critic_in, critic_hidden_size)))
            critic_head_layers.append(nn.ReLU())
            critic_in = critic_hidden_size
        critic_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(critic_in, 1), std=1))
        self.critic_head = nn.Sequential(*critic_head_layers)

    def forward(self, observations, state=None):
        actor_hidden = self.actor_backbone(observations, self.ego_dim)

        if self.split_network:
            critic_hidden = self.critic_backbone(observations, self.ego_dim)
        else:
            critic_hidden = actor_hidden

        if self.is_continuous:
            params = self.actor_head(actor_hidden)
            loc, scale = torch.split(params, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            actions = torch.distributions.Normal(loc, std)
        else:
            actions = torch.split(self.actor_head(actor_hidden), self.atn_dim, dim=1)

        value = self.critic_head(critic_hidden)

        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        assert not self.split_network, "LSTM wrapper doesn't support split_network=True"
        return self.actor_backbone(observations, self.ego_dim)

    def decode_actions(self, hidden):
        if self.is_continuous:
            parameters = self.actor_head(hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor_head(hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.critic_head(hidden)

        return action, value
