from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models
from pufferlib.ocean.drive import binding

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401

Recurrent = pufferlib.models.LSTMWrapper

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


def _activation(name):
    if name not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation {name!r}. Expected one of {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[name]


class DriveBackbone(nn.Module):
    def _create_encoder(self, in_features, out_size):
        layers = [pufferlib.pytorch.layer_init(nn.Linear(in_features, out_size))]
        if self.encoder_layer_norm:
            layers.append(nn.LayerNorm(out_size))
        layers.append(self.encoder_act_cls())
        layers.append(pufferlib.pytorch.layer_init(nn.Linear(out_size, out_size)))
        return nn.Sequential(*layers)

    def _encode_and_pool(self, objects, valid_counts, encoder, out_size):
        if not self.mask_padded_features:
            return encoder(objects).max(dim=1).values

        valid_mask = torch.arange(objects.shape[1], device=objects.device) < valid_counts.unsqueeze(1)
        encoded_objects = objects.new_full(
            (objects.shape[0], objects.shape[1], out_size),
            torch.finfo(objects.dtype).min,
        )
        encoded_objects[valid_mask] = encoder(objects[valid_mask])
        pooled = encoded_objects.amax(dim=1)
        return torch.where(valid_counts.unsqueeze(1) == 0, encoded_objects.new_zeros(pooled.shape), pooled)

    def __init__(
        self,
        env,
        ego_input_size,
        partner_input_size,
        lane_input_size,
        boundary_input_size,
        traffic_control_input_size,
        context_input_size,
        backbone_hidden_size,
        backbone_num_layers,
        ego_dim,
        encoder_activation,
        encoder_layer_norm,
        backbone_activation,
        backbone_layer_norm,
        mask_padded_features,
        strip_last_partner_features=0,
    ):
        super().__init__()
        self.encoder_act_cls = _activation(encoder_activation)
        self.encoder_layer_norm = encoder_layer_norm
        self.ego_dim = ego_dim
        self.ego_input_size = ego_input_size
        self.partner_input_size = partner_input_size
        self.lane_input_size = lane_input_size
        self.boundary_input_size = boundary_input_size
        self.traffic_control_input_size = traffic_control_input_size
        self.context_input_size = context_input_size

        self.max_partner_observations = env.max_partner_observations
        self.partner_features_count = env.partner_features
        self.strip_last_partner_features = strip_last_partner_features
        self.partner_encoder_features = self.partner_features_count - self.strip_last_partner_features
        self.obs_lane_segment_count = env.obs_lane_segment_count
        self.obs_boundary_segment_count = env.obs_boundary_segment_count
        self.road_features_count = env.road_features
        self.max_traffic_control_observations = env.max_traffic_control_observations
        self.traffic_control_features_count = env.traffic_control_features
        self.traffic_control_continuous_features = env.traffic_control_features - 2
        self.traffic_control_features_after_onehot = (
            self.traffic_control_continuous_features
            + binding.NUM_TRAFFIC_CONTROL_TYPES
            + binding.NUM_TRAFFIC_CONTROL_STATES
        )
        self.obs_valid_count_features = env.obs_valid_count_features
        self.mask_padded_features = mask_padded_features
        self.context_dim = env.num_reward_coefs + env.target_dim

        self.ego_encoder = self._create_encoder(ego_dim, ego_input_size)
        encoders_out = ego_input_size
        if self.obs_lane_segment_count > 0:
            self.lane_encoder = self._create_encoder(self.road_features_count, lane_input_size)
            encoders_out += lane_input_size
        if self.obs_boundary_segment_count > 0:
            self.boundary_encoder = self._create_encoder(self.road_features_count, boundary_input_size)
            encoders_out += boundary_input_size
        if self.max_partner_observations > 0:
            self.partner_encoder = self._create_encoder(self.partner_encoder_features, partner_input_size)
            encoders_out += partner_input_size
        if self.max_traffic_control_observations > 0:
            self.traffic_control_encoder = self._create_encoder(
                self.traffic_control_features_after_onehot,
                traffic_control_input_size,
            )
            encoders_out += traffic_control_input_size
        if self.context_dim > 0:
            self.context_encoder = self._create_encoder(self.context_dim, context_input_size)
            encoders_out += context_input_size

        backbone_act_cls = _activation(backbone_activation)
        backbone_layers = []
        bb_in = encoders_out
        for _ in range(backbone_num_layers):
            backbone_layers.append(backbone_act_cls())
            backbone_layers.append(pufferlib.pytorch.layer_init(nn.Linear(bb_in, backbone_hidden_size)))
            if backbone_layer_norm:
                backbone_layers.append(nn.LayerNorm(backbone_hidden_size))
            bb_in = backbone_hidden_size
        backbone_layers.append(backbone_act_cls())
        self.backbone = nn.Sequential(*backbone_layers)
        self.out_dim = backbone_hidden_size if backbone_num_layers > 0 else encoders_out

    def _split_observations(self, observations, ego_dim):
        partner_dim = self.max_partner_observations * self.partner_features_count
        lane_dim = self.obs_lane_segment_count * self.road_features_count
        boundary_dim = self.obs_boundary_segment_count * self.road_features_count
        traffic_control_dim = self.max_traffic_control_observations * self.traffic_control_features_count

        slide_idx = ego_dim
        ego_observations = observations[:, :slide_idx]
        context_observations = observations[:, slide_idx : slide_idx + self.context_dim]
        slide_idx += self.context_dim
        partner_observations = observations[:, slide_idx : slide_idx + partner_dim]
        slide_idx += partner_dim
        lane_observations = observations[:, slide_idx : slide_idx + lane_dim]
        slide_idx += lane_dim
        boundary_observations = observations[:, slide_idx : slide_idx + boundary_dim]
        slide_idx += boundary_dim
        traffic_control_observations = observations[:, slide_idx : slide_idx + traffic_control_dim]
        slide_idx += traffic_control_dim
        count_observations = observations[:, slide_idx : slide_idx + self.obs_valid_count_features]
        return (
            ego_observations,
            context_observations,
            partner_observations,
            lane_observations,
            boundary_observations,
            traffic_control_observations,
            count_observations,
        )

    def _valid_counts(self, count_observations):
        capacities = (
            self.obs_lane_segment_count,
            self.obs_boundary_segment_count,
            self.max_partner_observations,
            self.max_traffic_control_observations,
        )
        return [count_observations[:, i].long().clamp_(0, capacity) for i, capacity in enumerate(capacities)]

    def forward(self, observations, ego_dim):
        (
            ego_observations,
            context_observations,
            partner_observations,
            lane_observations,
            boundary_observations,
            traffic_control_observations,
            count_observations,
        ) = self._split_observations(observations, ego_dim)
        lane_counts, boundary_counts, partner_counts, traffic_control_counts = self._valid_counts(count_observations)

        feature_list = [self.ego_encoder(ego_observations)]

        if self.obs_lane_segment_count > 0:
            lane_objects = lane_observations.view(-1, self.obs_lane_segment_count, self.road_features_count)
            lane_features = self._encode_and_pool(lane_objects, lane_counts, self.lane_encoder, self.lane_input_size)
            feature_list.append(lane_features)
        if self.obs_boundary_segment_count > 0:
            boundary_objects = boundary_observations.view(-1, self.obs_boundary_segment_count, self.road_features_count)
            boundary_features = self._encode_and_pool(
                boundary_objects,
                boundary_counts,
                self.boundary_encoder,
                self.boundary_input_size,
            )
            feature_list.append(boundary_features)
        if self.max_partner_observations > 0:
            partner_objects = partner_observations.view(
                -1,
                self.max_partner_observations,
                self.partner_features_count,
            )
            if self.strip_last_partner_features > 0:
                partner_objects = partner_objects[..., : -self.strip_last_partner_features]
            partner_features = self._encode_and_pool(
                partner_objects,
                partner_counts,
                self.partner_encoder,
                self.partner_input_size,
            )
            feature_list.append(partner_features)
        if self.max_traffic_control_observations > 0:
            traffic_control_objects = traffic_control_observations.view(
                -1,
                self.max_traffic_control_observations,
                self.traffic_control_features_count,
            )
            traffic_control_continuous = traffic_control_objects[:, :, : self.traffic_control_continuous_features]
            traffic_control_type = traffic_control_objects[:, :, self.traffic_control_continuous_features]
            traffic_control_state = traffic_control_objects[:, :, self.traffic_control_continuous_features + 1]
            traffic_control_type_onehot = F.one_hot(
                traffic_control_type.long(),
                num_classes=binding.NUM_TRAFFIC_CONTROL_TYPES,
            ).to(traffic_control_continuous.dtype)
            traffic_control_state_onehot = F.one_hot(
                traffic_control_state.long(),
                num_classes=binding.NUM_TRAFFIC_CONTROL_STATES,
            ).to(traffic_control_continuous.dtype)
            traffic_control_objects = torch.cat(
                [traffic_control_continuous, traffic_control_type_onehot, traffic_control_state_onehot],
                dim=2,
            )
            traffic_control_features = self._encode_and_pool(
                traffic_control_objects,
                traffic_control_counts,
                self.traffic_control_encoder,
                self.traffic_control_input_size,
            )
            feature_list.append(traffic_control_features)
        if self.context_dim > 0:
            feature_list.append(self.context_encoder(context_observations))

        return self.backbone(torch.cat(feature_list, dim=1))

    def pool_slot_counts(self, observations, ego_dim):
        (
            _ego_observations,
            _context_observations,
            partner_observations,
            lane_observations,
            boundary_observations,
            traffic_control_observations,
            _count_observations,
        ) = self._split_observations(observations, ego_dim)

        counts = {}
        if self.obs_lane_segment_count > 0:
            lane_objects = lane_observations.view(-1, self.obs_lane_segment_count, self.road_features_count)
            lane_winners = self.lane_encoder(lane_objects).max(dim=1).indices
            lane_counts = torch.zeros(
                observations.shape[0],
                self.obs_lane_segment_count,
                device=observations.device,
                dtype=torch.int64,
            )
            counts["pool_lane"] = lane_counts.scatter_add(1, lane_winners, torch.ones_like(lane_winners))
        if self.obs_boundary_segment_count > 0:
            boundary_objects = boundary_observations.view(-1, self.obs_boundary_segment_count, self.road_features_count)
            boundary_winners = self.boundary_encoder(boundary_objects).max(dim=1).indices
            boundary_counts = torch.zeros(
                observations.shape[0],
                self.obs_boundary_segment_count,
                device=observations.device,
                dtype=torch.int64,
            )
            counts["pool_boundary"] = boundary_counts.scatter_add(
                1,
                boundary_winners,
                torch.ones_like(boundary_winners),
            )
        if self.max_partner_observations > 0:
            partner_objects = partner_observations.view(-1, self.max_partner_observations, self.partner_features_count)
            if self.strip_last_partner_features > 0:
                partner_objects = partner_objects[..., : -self.strip_last_partner_features]
            partner_winners = self.partner_encoder(partner_objects).max(dim=1).indices
            partner_counts = torch.zeros(
                observations.shape[0],
                self.max_partner_observations,
                device=observations.device,
                dtype=torch.int64,
            )
            counts["pool_partner"] = partner_counts.scatter_add(1, partner_winners, torch.ones_like(partner_winners))
        if self.max_traffic_control_observations > 0:
            traffic_control_objects = traffic_control_observations.view(
                -1,
                self.max_traffic_control_observations,
                self.traffic_control_features_count,
            )
            traffic_control_continuous = traffic_control_objects[:, :, : self.traffic_control_continuous_features]
            traffic_control_type = traffic_control_objects[:, :, self.traffic_control_continuous_features]
            traffic_control_state = traffic_control_objects[:, :, self.traffic_control_continuous_features + 1]
            traffic_control_type_onehot = F.one_hot(
                traffic_control_type.long(),
                num_classes=binding.NUM_TRAFFIC_CONTROL_TYPES,
            ).to(traffic_control_continuous.dtype)
            traffic_control_state_onehot = F.one_hot(
                traffic_control_state.long(),
                num_classes=binding.NUM_TRAFFIC_CONTROL_STATES,
            ).to(traffic_control_continuous.dtype)
            traffic_control_objects = torch.cat(
                [traffic_control_continuous, traffic_control_type_onehot, traffic_control_state_onehot],
                dim=2,
            )
            traffic_control_winners = self.traffic_control_encoder(traffic_control_objects).max(dim=1).indices
            traffic_control_counts = torch.zeros(
                observations.shape[0],
                self.max_traffic_control_observations,
                device=observations.device,
                dtype=torch.int64,
            )
            counts["pool_traffic"] = traffic_control_counts.scatter_add(
                1,
                traffic_control_winners,
                torch.ones_like(traffic_control_winners),
            )
        return counts


class Drive(nn.Module):
    def __init__(
        self,
        env,
        ego_input_size: int = 64,
        partner_input_size: int = 64,
        lane_input_size: int = 64,
        boundary_input_size: int = 64,
        traffic_control_input_size: int = 64,
        context_input_size: int = 64,
        backbone_hidden_size: int = 512,
        backbone_num_layers: int = 4,
        actor_hidden_size: int = 512,
        actor_num_layers: int = 0,
        critic_hidden_size: int = 512,
        critic_num_layers: int = 0,
        encoder_activation: str = "relu",
        encoder_layer_norm: bool = True,
        backbone_activation: str = "gelu",
        backbone_layer_norm: bool = False,
        shared_network: bool = True,
        mask_padded_features: bool = False,
        **legacy_kwargs,
    ):
        super().__init__()
        if "split_network" in legacy_kwargs:
            shared_network = not bool(legacy_kwargs.pop("split_network"))
        if legacy_kwargs:
            unknown = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected Drive policy kwargs: {unknown}")

        self.shared_network = shared_network
        self.ego_dim = env.ego_features
        backbone_args = {
            "env": env,
            "ego_input_size": ego_input_size,
            "partner_input_size": partner_input_size,
            "lane_input_size": lane_input_size,
            "boundary_input_size": boundary_input_size,
            "traffic_control_input_size": traffic_control_input_size,
            "context_input_size": context_input_size,
            "backbone_hidden_size": backbone_hidden_size,
            "backbone_num_layers": backbone_num_layers,
            "ego_dim": self.ego_dim,
            "encoder_activation": encoder_activation,
            "encoder_layer_norm": encoder_layer_norm,
            "backbone_activation": backbone_activation,
            "backbone_layer_norm": backbone_layer_norm,
            "mask_padded_features": mask_padded_features,
        }
        self.actor_backbone = DriveBackbone(**backbone_args)
        self.critic_backbone = self.actor_backbone if self.shared_network else DriveBackbone(**backbone_args)

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        self.atn_dim = (
            (env.single_action_space.shape[0],) * 2 if self.is_continuous else env.single_action_space.nvec.tolist()
        )

        backbone_out_dim = self.actor_backbone.out_dim
        self.actor_head = self._make_head(
            backbone_out_dim, actor_hidden_size, actor_num_layers, sum(self.atn_dim), 0.01
        )
        self.critic_head = self._make_head(backbone_out_dim, critic_hidden_size, critic_num_layers, 1, 1)

    @staticmethod
    def _make_head(input_dim, hidden_size, num_layers, output_dim, std):
        layers = []
        head_in = input_dim
        for _ in range(num_layers):
            layers.append(pufferlib.pytorch.layer_init(nn.Linear(head_in, hidden_size)))
            layers.append(nn.ReLU())
            head_in = hidden_size
        layers.append(pufferlib.pytorch.layer_init(nn.Linear(head_in, output_dim), std=std))
        return nn.Sequential(*layers)

    def forward(self, observations, state=None):
        actor_hidden = self.actor_backbone(observations, self.ego_dim)
        critic_hidden = actor_hidden if self.shared_network else self.critic_backbone(observations, self.ego_dim)

        if self.is_continuous:
            params = self.actor_head(actor_hidden)
            loc, scale = torch.split(params, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            actions = torch.distributions.Normal(loc, std)
        else:
            actions = torch.split(self.actor_head(actor_hidden), self.atn_dim, dim=1)

        return actions, self.critic_head(critic_hidden)

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def pool_slot_counts(self, observations, state=None):
        return self.actor_backbone.pool_slot_counts(observations, self.ego_dim)

    def encode_observations(self, observations, state=None):
        assert self.shared_network, "LSTM wrapper requires shared_network=True"
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


class TargetDrive(Drive):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        backbone_kwargs = self._target_backbone_kwargs(env, *args, **kwargs)
        self.actor_backbone = DriveBackbone(**backbone_kwargs)
        self.critic_backbone = self.actor_backbone if self.shared_network else DriveBackbone(**backbone_kwargs)

    def _target_backbone_kwargs(self, env, *args, **kwargs):
        params = {
            "env": env,
            "ego_input_size": kwargs.get("ego_input_size", 64),
            "partner_input_size": kwargs.get("partner_input_size", 64),
            "lane_input_size": kwargs.get("lane_input_size", 64),
            "boundary_input_size": kwargs.get("boundary_input_size", 64),
            "traffic_control_input_size": kwargs.get("traffic_control_input_size", 64),
            "context_input_size": kwargs.get("context_input_size", 64),
            "backbone_hidden_size": kwargs.get("backbone_hidden_size", 512),
            "backbone_num_layers": kwargs.get("backbone_num_layers", 4),
            "ego_dim": env.ego_features,
            "encoder_activation": kwargs.get("encoder_activation", "relu"),
            "encoder_layer_norm": kwargs.get("encoder_layer_norm", True),
            "backbone_activation": kwargs.get("backbone_activation", "gelu"),
            "backbone_layer_norm": kwargs.get("backbone_layer_norm", False),
            "mask_padded_features": kwargs.get("mask_padded_features", False),
            "strip_last_partner_features": 2,
        }
        return params
