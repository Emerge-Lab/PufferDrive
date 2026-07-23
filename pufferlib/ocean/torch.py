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


class DriveBackbone(nn.Module):
    """
    Neural network backbone
    Architecture features:
      - Split Actor/Critic (configurable)
    """

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
        return torch.where(valid_counts.unsqueeze(1) == 0, encoded_objects.new_zeros(()), pooled)

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
    ):
        super().__init__()
        self.encoder_act_cls = ACTIVATIONS[encoder_activation]
        self.encoder_layer_norm = encoder_layer_norm
        self.ego_dim = ego_dim
        self.ego_input_size = ego_input_size
        self.partner_input_size = partner_input_size
        self.lane_input_size = lane_input_size
        self.boundary_input_size = boundary_input_size
        self.traffic_control_input_size = traffic_control_input_size
        self.context_input_size = context_input_size

        # Observation dimensions from environment config
        self.obs_slots_partners_n = env.obs_slots_partners_n
        self.partner_features_count = env.partner_features
        # Road features size
        self.obs_slots_lane_kept = env.obs_slots_lane_kept
        self.obs_slots_boundary_kept = env.obs_slots_boundary_kept
        self.lane_features_count = env.lane_features
        self.boundary_features_count = env.boundary_features
        # Traffic control size
        self.obs_slots_traffic_controls_n = env.obs_slots_traffic_controls_n
        self.traffic_control_features_count = env.traffic_control_features
        self.traffic_control_continuous_features = env.traffic_control_features - 2
        self.traffic_control_features_after_onehot = (
            self.traffic_control_continuous_features
            + binding.NUM_TRAFFIC_CONTROL_TYPES
            + binding.NUM_TRAFFIC_CONTROL_STATES
        )
        self.obs_valid_count_features = binding.OBS_VALID_COUNT_FEATURES
        self.mask_padded_features = mask_padded_features
        # Context size (reward coefficients + target info)
        self.context_dim = env.num_reward_coefs + env.goal_dim

        # 1. observations Encoders
        # Each encoder projects raw features into its own embedding space
        self.ego_encoder = self._create_encoder(ego_dim, ego_input_size)
        encoders_out = ego_input_size
        if self.obs_slots_lane_kept > 0:
            self.lane_encoder = self._create_encoder(self.lane_features_count, lane_input_size)
            encoders_out += lane_input_size
        if self.obs_slots_boundary_kept > 0:
            self.boundary_encoder = self._create_encoder(self.boundary_features_count, boundary_input_size)
            encoders_out += boundary_input_size
        if self.obs_slots_partners_n > 0:
            self.partner_encoder = self._create_encoder(self.partner_features_count, partner_input_size)
            encoders_out += partner_input_size
        if self.obs_slots_traffic_controls_n > 0:
            self.traffic_control_encoder = self._create_encoder(
                self.traffic_control_features_after_onehot, traffic_control_input_size
            )
            encoders_out += traffic_control_input_size
        if self.context_dim > 0:
            self.context_encoder = self._create_encoder(self.context_dim, context_input_size)
            encoders_out += context_input_size

        # 2. Main Backbone MLP
        backbone_act_cls = ACTIVATIONS[backbone_activation]
        backbone_layers = []
        bb_in = encoders_out
        for _ in range(backbone_num_layers):
            backbone_layers.append(backbone_act_cls())
            backbone_layers.append(pufferlib.pytorch.layer_init(nn.Linear(bb_in, backbone_hidden_size)))
            if backbone_layer_norm:
                backbone_layers.append(nn.LayerNorm(backbone_hidden_size))
            bb_in = backbone_hidden_size
        # Add final activation before heads
        backbone_layers.append(backbone_act_cls())
        self.backbone = nn.Sequential(*backbone_layers)
        self.out_dim = backbone_hidden_size if backbone_num_layers > 0 else encoders_out

    def forward(self, observations, ego_dim):
        # Extract and slice observations from the flat buffer
        partner_dim = self.obs_slots_partners_n * self.partner_features_count
        lane_dim = self.obs_slots_lane_kept * self.lane_features_count
        boundary_dim = self.obs_slots_boundary_kept * self.boundary_features_count
        traffic_control_dim = self.obs_slots_traffic_controls_n * self.traffic_control_features_count

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
        count_observations = observations[
            :, slide_idx + traffic_control_dim : slide_idx + traffic_control_dim + self.obs_valid_count_features
        ]
        lane_counts, boundary_counts, partner_counts, traffic_control_counts = [
            count_observations[:, i].long().clamp_(0, capacity)
            for i, capacity in enumerate(
                (
                    self.obs_slots_lane_kept,
                    self.obs_slots_boundary_kept,
                    self.obs_slots_partners_n,
                    self.obs_slots_traffic_controls_n,
                )
            )
        ]

        # Encode Ego State
        ego_features = self.ego_encoder(ego_observations)

        feature_list = [ego_features]

        # Encode Lanes and Boundaries separately
        if self.obs_slots_lane_kept > 0:
            lane_objects = lane_observations.view(-1, self.obs_slots_lane_kept, self.lane_features_count)
            lane_features = self._encode_and_pool(lane_objects, lane_counts, self.lane_encoder, self.lane_input_size)
            feature_list.append(lane_features)
        if self.obs_slots_boundary_kept > 0:
            boundary_objects = boundary_observations.view(
                -1, self.obs_slots_boundary_kept, self.boundary_features_count
            )
            boundary_features = self._encode_and_pool(
                boundary_objects,
                boundary_counts,
                self.boundary_encoder,
                self.boundary_input_size,
            )
            feature_list.append(boundary_features)

        # Encode Partners
        if self.obs_slots_partners_n > 0:
            partner_objects = partner_observations.view(-1, self.obs_slots_partners_n, self.partner_features_count)
            partner_features = self._encode_and_pool(
                partner_objects,
                partner_counts,
                self.partner_encoder,
                self.partner_input_size,
            )
            feature_list.append(partner_features)

        # Encode Traffic Controls
        if self.obs_slots_traffic_controls_n > 0:
            traffic_control_objects = traffic_control_observations.view(
                -1, self.obs_slots_traffic_controls_n, self.traffic_control_features_count
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

        # Add optional features if enabled
        if self.context_dim > 0:
            context_features = self.context_encoder(context_observations)
            feature_list.append(context_features)

        # Concatenate all features and pass through main backbone
        concat_features = torch.cat(feature_list, dim=1)
        return self.backbone(concat_features)

    def pool_slot_counts(self, observations, ego_dim):
        partner_dim = self.obs_slots_partners_n * self.partner_features_count
        lane_dim = self.obs_slots_lane_kept * self.lane_features_count
        boundary_dim = self.obs_slots_boundary_kept * self.boundary_features_count
        traffic_control_dim = self.obs_slots_traffic_controls_n * self.traffic_control_features_count

        slide_idx = ego_dim + self.context_dim
        partner_observations = observations[:, slide_idx : slide_idx + partner_dim]
        slide_idx += partner_dim
        lane_observations = observations[:, slide_idx : slide_idx + lane_dim]
        slide_idx += lane_dim
        boundary_observations = observations[:, slide_idx : slide_idx + boundary_dim]
        slide_idx += boundary_dim
        traffic_control_observations = observations[:, slide_idx : slide_idx + traffic_control_dim]

        counts = {}
        if self.obs_slots_lane_kept > 0:
            lane_objects = lane_observations.view(-1, self.obs_slots_lane_kept, self.lane_features_count)
            lane_winners = self.lane_encoder(lane_objects).max(dim=1).indices
            lane_counts = torch.zeros(
                observations.shape[0], self.obs_slots_lane_kept, device=observations.device, dtype=torch.int64
            )
            counts["pool_lane"] = lane_counts.scatter_add(1, lane_winners, torch.ones_like(lane_winners))
        if self.obs_slots_boundary_kept > 0:
            boundary_objects = boundary_observations.view(
                -1, self.obs_slots_boundary_kept, self.boundary_features_count
            )
            boundary_winners = self.boundary_encoder(boundary_objects).max(dim=1).indices
            boundary_counts = torch.zeros(
                observations.shape[0], self.obs_slots_boundary_kept, device=observations.device, dtype=torch.int64
            )
            counts["pool_boundary"] = boundary_counts.scatter_add(
                1, boundary_winners, torch.ones_like(boundary_winners)
            )
        if self.obs_slots_partners_n > 0:
            partner_objects = partner_observations.view(-1, self.obs_slots_partners_n, self.partner_features_count)
            partner_winners = self.partner_encoder(partner_objects).max(dim=1).indices
            partner_counts = torch.zeros(
                observations.shape[0], self.obs_slots_partners_n, device=observations.device, dtype=torch.int64
            )
            counts["pool_partner"] = partner_counts.scatter_add(1, partner_winners, torch.ones_like(partner_winners))
        if self.obs_slots_traffic_controls_n > 0:
            traffic_control_objects = traffic_control_observations.view(
                -1, self.obs_slots_traffic_controls_n, self.traffic_control_features_count
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
                observations.shape[0], self.obs_slots_traffic_controls_n, device=observations.device, dtype=torch.int64
            )
            counts["pool_traffic"] = traffic_control_counts.scatter_add(
                1, traffic_control_winners, torch.ones_like(traffic_control_winners)
            )
        return counts


class Drive(nn.Module):
    def __init__(
        self,
        env,
        ego_input_size: int,
        partner_input_size: int,
        lane_input_size: int,
        boundary_input_size: int,
        traffic_control_input_size: int,
        context_input_size: int,
        backbone_hidden_size: int,
        backbone_num_layers: int,
        actor_hidden_size: int,
        actor_num_layers: int,
        critic_hidden_size: int,
        critic_num_layers: int,
        encoder_activation: str,
        encoder_layer_norm: bool,
        backbone_activation: str,
        backbone_layer_norm: bool,
        shared_network: bool,
        mask_padded_features: bool,
        action_type: str,
    ):
        super().__init__()
        
        self.register_buffer("JERK_LONG", torch.tensor((-15.0, -4.0, 0.0, 4.0), dtype=torch.float32, requires_grad=False), persistent=False)
        self.register_buffer("JERK_LAT", torch.tensor((-4.0, 0.0, 4.0), dtype=torch.float32, requires_grad=False), persistent=False)

        self.register_buffer("ACCELERATION_VALUES", torch.tensor((-4.0000, -2.6670, -1.3330, -0.0000, 1.3330, 2.6670, 4.0000), dtype=torch.float32, requires_grad=False), persistent=False)
        self.register_buffer("STEERING_VALUES", torch.tensor((-0.667, -0.500, -0.333, -0.167, 0.000, 0.167, 0.333, 0.500, 0.667), dtype=torch.float32, requires_grad=False), persistent=False)

        if env.dynamics_model == "jerk":
            action_long, action_lat = self.JERK_LONG, self.JERK_LAT
        elif env.dynamics_model == "classic":
            action_long, action_lat = self.ACCELERATION_VALUES, self.STEERING_VALUES
        else:
            raise ValueError(f"Unsupported dynamics model: {env.dynamics_model}")

        # Precompute the [-1, 1] continuous action per discrete choice by inverting the
        # sim's continuous scaling (drive.h). Constant → done once. Longitudinal is
        # asymmetric (braking / |t[0]|, accel / t[-1]); lateral symmetric. The symmetric
        # classic table collapses both branches to / t[-1].
        long_norm = torch.where(
            action_long < 0.0,
            action_long / -action_long[0],
            action_long / action_long[-1],
        )
        lat_norm = action_lat / action_lat[-1]
        self.register_buffer("action_long_norm", long_norm, persistent=False)
        self.register_buffer("action_lat_norm", lat_norm, persistent=False)

        # Joint continuous-action table: row k = the [-1,1] (long, lat) for discrete class k,
        # where k = long_idx * num_lat + lat_idx (matches drive.h decode).
        num_lat = lat_norm.numel()
        num_classes = long_norm.numel() * num_lat
        k = torch.arange(num_classes)
        action_table = torch.stack([long_norm[k // num_lat], lat_norm[k % num_lat]], dim=-1)  # [num_classes, 2]
        self.register_buffer("action_table", action_table, persistent=False)

        # Configuration flags from policy kwargs
        self.shared_network = shared_network
        self.ego_dim = env.ego_features

        # Prepare arguments for the Backbone
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

        # Instantiate backbones
        self.actor_backbone = DriveBackbone(**backbone_args)

        # If using shared network, critic backbone is the same as actor backbone.
        # Otherwise, create a separate critic backbone with the same architecture.
        if self.shared_network:
            self.critic_backbone = self.actor_backbone
        else:
            self.critic_backbone = DriveBackbone(**backbone_args)

        # Setup action and value heads
        self.is_continuous = action_type == "continuous" # TODO Check if what the `"trajectory"`, `"trajectory_frenet"`, `"trajectory_jerk" features do and if they are considered continuous.
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = [self.action_long_norm.numel() * self.action_lat_norm.numel()]

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
        if self.shared_network:
            critic_hidden = actor_hidden
        else:
            critic_hidden = self.critic_backbone(observations, self.ego_dim)

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

    def pool_slot_counts(self, observations, state=None):
        return self.actor_backbone.pool_slot_counts(observations, self.ego_dim)

    # Required for PufferLib recurrent wrappers
    def encode_observations(self, observations, state=None):
        assert self.shared_network, "LSTM wrapper requires shared_network=True"
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
    
    def discrete_actions_to_continuous(self, actions):
        return self.action_table[actions.long()]

    def discrete_probs_to_continuous_mean(self, probs):
        # probs: [..., num_classes] -> [..., 2]  (E[cont | probs])
        return probs @ self.action_table.to(probs.dtype)
