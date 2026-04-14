from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper


class DriveBackbone(nn.Module):
    """GIGAFLOW-style backbone: per-group encoders, max-pool over set dims, GELU MLP."""

    def _create_encoder(self, in_features, input_size, encoder_gigaflow, dropout=0.0):
        if encoder_gigaflow:
            return nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(in_features, input_size)),
                nn.LayerNorm(input_size),
                nn.Tanh(),
                nn.Dropout(dropout),
                pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
            )
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
    ):
        super().__init__()

        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        # 3.0 road obs: last feature is a categorical type (7 classes)
        self.road_features_after_onehot = self.road_features + 6

        self.ego_encoder = self._create_encoder(ego_dim, input_size, encoder_gigaflow)
        self.partner_encoder = self._create_encoder(self.partner_features, input_size, encoder_gigaflow)
        self.road_encoder = self._create_encoder(
            self.road_features_after_onehot, input_size, encoder_gigaflow, dropout=dropout
        )

        num_feature_sets = 3  # ego, road, partner

        backbone_layers = []
        bb_in = num_feature_sets * input_size
        for _ in range(backbone_num_layers):
            backbone_layers.append(nn.GELU())
            backbone_layers.append(pufferlib.pytorch.layer_init(nn.Linear(bb_in, backbone_hidden_size)))
            bb_in = backbone_hidden_size
        backbone_layers.append(nn.GELU())
        self.backbone = nn.Sequential(*backbone_layers)
        self.out_dim = backbone_hidden_size if backbone_num_layers > 0 else num_feature_sets * input_size

    def forward(self, observations, ego_dim):
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features

        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7).float()
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        return self.backbone(concat_features)


class Drive(nn.Module):
    def __init__(
        self,
        env,
        input_size: int = 64,
        backbone_hidden_size: int = 512,
        backbone_num_layers: int = 4,
        actor_hidden_size: int = 512,
        actor_num_layers: int = 0,
        critic_hidden_size: int = 512,
        critic_num_layers: int = 0,
        encoder_gigaflow: bool = True,
        dropout: float = 0.0,
        split_network: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.split_network = split_network
        self.ego_dim = env.ego_features

        backbone_args = dict(
            env=env,
            input_size=input_size,
            backbone_hidden_size=backbone_hidden_size,
            backbone_num_layers=backbone_num_layers,
            ego_dim=self.ego_dim,
            encoder_gigaflow=encoder_gigaflow,
            dropout=dropout,
        )

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
        # LSTMWrapper reads policy.hidden_size
        self.hidden_size = backbone_out_dim

        actor_head_layers = []
        actor_in = backbone_out_dim
        for _ in range(actor_num_layers):
            actor_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(actor_in, actor_hidden_size)))
            actor_head_layers.append(nn.ReLU())
            actor_in = actor_hidden_size
        actor_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(actor_in, sum(self.atn_dim)), std=0.01))
        self.actor_head = nn.Sequential(*actor_head_layers)
        # Alias for LSTMWrapper compat (which reads policy.actor)
        self.actor = self.actor_head

        critic_head_layers = []
        critic_in = backbone_out_dim
        for _ in range(critic_num_layers):
            critic_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(critic_in, critic_hidden_size)))
            critic_head_layers.append(nn.ReLU())
            critic_in = critic_hidden_size
        critic_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(critic_in, 1), std=1))
        self.critic_head = nn.Sequential(*critic_head_layers)
        # Alias for LSTMWrapper compat (which reads policy.value_fn)
        self.value_fn = self.critic_head

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
