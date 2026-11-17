import contextlib

from torch import nn
import torch
import torch.nn.functional as F

import opengigaflow.pytorch
from opengigaflow.ocean.drive import binding
from opengigaflow.ocean.fused_slot_encoder import fused_encode_and_pool, fused_encoder_active, validate_fused_encoder

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
SLOT_COUNT_ATTRIBUTE_BY_ENCODER = {
    "lane_encoder": "obs_slots_lane_kept",
    "boundary_encoder": "obs_slots_boundary_kept",
    "partner_encoder": "obs_slots_partners_n",
    "traffic_control_encoder": "obs_slots_traffic_controls_n",
}


def build_mlp_head(in_features, hidden_size, num_layers, layer_norm, out_features, out_std):
    layers = []
    for _ in range(num_layers):
        layers.append(opengigaflow.pytorch.layer_init(nn.Linear(in_features, hidden_size)))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        in_features = hidden_size
    layers.append(opengigaflow.pytorch.layer_init(nn.Linear(in_features, out_features), std=out_std))
    return nn.Sequential(*layers)


class DriveBackbone(nn.Module):
    """Ego, context and max-pooled slot encoders feeding one MLP."""

    fused_slot_encoder = False  # class default: tests build instances without __init__

    def _create_encoder(self, in_features, out_size):
        layers = [opengigaflow.pytorch.layer_init(nn.Linear(in_features, out_size))]
        if self.encoder_layer_norm:
            layers.append(nn.LayerNorm(out_size))
        layers.append(self.encoder_act_cls())
        layers.append(opengigaflow.pytorch.layer_init(nn.Linear(out_size, out_size)))
        return nn.Sequential(*layers)

    def _encode_and_pool(self, objects, valid_counts, encoder):
        if self.fused_slot_encoder and fused_encoder_active(objects):
            return fused_encode_and_pool(objects, valid_counts, encoder)

        # Dense encode + masked fill keeps shapes static for torch.compile; padded rows are rare.
        valid_mask = torch.arange(objects.shape[1], device=objects.device) < valid_counts.unsqueeze(1)
        encoded_objects = encoder(objects)
        masked_objects = encoded_objects.masked_fill(~valid_mask.unsqueeze(2), torch.finfo(encoded_objects.dtype).min)
        pooled = masked_objects.max(dim=1).values
        return torch.where(valid_counts.unsqueeze(1) == 0, pooled.new_zeros(()), pooled)

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
        encoder_activation,
        encoder_layer_norm,
        backbone_activation,
        backbone_layer_norm,
        fused_slot_encoder,
    ):
        super().__init__()
        self.encoder_act_cls = ACTIVATIONS[encoder_activation]
        self.encoder_layer_norm = encoder_layer_norm
        self.ego_features_count = env.ego_features
        self.obs_slots_partners_n = env.obs_slots_partners_n
        self.partner_features_count = env.partner_features
        self.obs_slots_lane_kept = env.obs_slots_lane_kept
        self.obs_slots_boundary_kept = env.obs_slots_boundary_kept
        self.lane_features_count = env.lane_features
        self.boundary_features_count = env.boundary_features
        self.obs_slots_traffic_controls_n = env.obs_slots_traffic_controls_n
        self.traffic_control_features_count = env.traffic_control_features
        self.traffic_control_continuous_features = env.traffic_control_features - 2
        self.traffic_control_features_after_onehot = (
            self.traffic_control_continuous_features
            + binding.NUM_TRAFFIC_CONTROL_TYPES
            + binding.NUM_TRAFFIC_CONTROL_STATES
        )
        self.obs_valid_count_features = binding.OBS_VALID_COUNT_FEATURES
        self.context_dim = env.num_reward_coefs + env.goal_dim

        self.ego_encoder = self._create_encoder(self.ego_features_count, ego_input_size)
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

        self.fused_slot_encoder = fused_slot_encoder
        slot_encoder_names = ("lane_encoder", "boundary_encoder", "partner_encoder", "traffic_control_encoder")
        for encoder_name in slot_encoder_names:
            if fused_slot_encoder and hasattr(self, encoder_name):
                validate_fused_encoder(getattr(self, encoder_name))

        backbone_act_cls = ACTIVATIONS[backbone_activation]
        backbone_layers = []
        bb_in = encoders_out
        for _ in range(backbone_num_layers):
            backbone_layers.append(backbone_act_cls())
            backbone_layers.append(opengigaflow.pytorch.layer_init(nn.Linear(bb_in, backbone_hidden_size)))
            if backbone_layer_norm:
                backbone_layers.append(nn.LayerNorm(backbone_hidden_size))
            bb_in = backbone_hidden_size
        backbone_layers.append(backbone_act_cls())
        self.backbone = nn.Sequential(*backbone_layers)
        self.out_dim = backbone_hidden_size if backbone_num_layers > 0 else encoders_out

    def _slot_observations(self, observations):
        """Column blocks of the flat observations: ego, context, partner, lane, boundary, traffic control, counts."""
        block_widths = (
            self.ego_features_count,
            self.context_dim,
            self.obs_slots_partners_n * self.partner_features_count,
            self.obs_slots_lane_kept * self.lane_features_count,
            self.obs_slots_boundary_kept * self.boundary_features_count,
            self.obs_slots_traffic_controls_n * self.traffic_control_features_count,
            self.obs_valid_count_features,
        )
        blocks = []
        block_start = 0
        for block_width in block_widths:
            blocks.append(observations[:, block_start : block_start + block_width])
            block_start += block_width
        return blocks

    def _one_hot_traffic_controls(self, traffic_control_objects):
        continuous = traffic_control_objects[:, :, : self.traffic_control_continuous_features]
        control_type = traffic_control_objects[:, :, self.traffic_control_continuous_features]
        control_state = traffic_control_objects[:, :, self.traffic_control_continuous_features + 1]
        type_onehot = F.one_hot(control_type.long(), num_classes=binding.NUM_TRAFFIC_CONTROL_TYPES).to(continuous.dtype)
        state_onehot = F.one_hot(control_state.long(), num_classes=binding.NUM_TRAFFIC_CONTROL_STATES).to(
            continuous.dtype
        )
        return torch.cat([continuous, type_onehot, state_onehot], dim=2)

    def forward(self, observations):
        (
            ego_observations,
            context_observations,
            partner_observations,
            lane_observations,
            boundary_observations,
            traffic_control_observations,
            count_observations,
        ) = self._slot_observations(observations)
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

        feature_list = [self.ego_encoder(ego_observations)]
        if self.obs_slots_lane_kept > 0:
            lane_objects = lane_observations.view(-1, self.obs_slots_lane_kept, self.lane_features_count)
            feature_list.append(self._encode_and_pool(lane_objects, lane_counts, self.lane_encoder))
        if self.obs_slots_boundary_kept > 0:
            boundary_objects = boundary_observations.view(
                -1, self.obs_slots_boundary_kept, self.boundary_features_count
            )
            feature_list.append(self._encode_and_pool(boundary_objects, boundary_counts, self.boundary_encoder))
        if self.obs_slots_partners_n > 0:
            partner_objects = partner_observations.view(-1, self.obs_slots_partners_n, self.partner_features_count)
            feature_list.append(self._encode_and_pool(partner_objects, partner_counts, self.partner_encoder))
        if self.obs_slots_traffic_controls_n > 0:
            traffic_control_objects = self._one_hot_traffic_controls(
                traffic_control_observations.view(
                    -1, self.obs_slots_traffic_controls_n, self.traffic_control_features_count
                )
            )
            feature_list.append(
                self._encode_and_pool(traffic_control_objects, traffic_control_counts, self.traffic_control_encoder)
            )
        if self.context_dim > 0:
            feature_list.append(self.context_encoder(context_observations))
        return self.backbone(torch.cat(feature_list, dim=1))

    def pool_slot_counts(self, observations):
        _, _, partner_observations, lane_observations, boundary_observations, traffic_control_observations, _ = (
            self._slot_observations(observations)
        )
        pools = []
        if self.obs_slots_lane_kept > 0:
            lane_objects = lane_observations.view(-1, self.obs_slots_lane_kept, self.lane_features_count)
            pools.append(("pool_lane", self.lane_encoder, lane_objects))
        if self.obs_slots_boundary_kept > 0:
            boundary_objects = boundary_observations.view(
                -1, self.obs_slots_boundary_kept, self.boundary_features_count
            )
            pools.append(("pool_boundary", self.boundary_encoder, boundary_objects))
        if self.obs_slots_partners_n > 0:
            partner_objects = partner_observations.view(-1, self.obs_slots_partners_n, self.partner_features_count)
            pools.append(("pool_partner", self.partner_encoder, partner_objects))
        if self.obs_slots_traffic_controls_n > 0:
            traffic_control_objects = self._one_hot_traffic_controls(
                traffic_control_observations.view(
                    -1, self.obs_slots_traffic_controls_n, self.traffic_control_features_count
                )
            )
            pools.append(("pool_traffic", self.traffic_control_encoder, traffic_control_objects))

        counts = {}
        for pool_name, encoder, objects in pools:
            winners = encoder(objects).max(dim=1).indices
            slot_counts = torch.zeros(
                observations.shape[0], objects.shape[1], device=observations.device, dtype=torch.int64
            )
            counts[pool_name] = slot_counts.scatter_add(1, winners, torch.ones_like(winners))
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
        action_type: str,
        actor_head_layer_norm: bool = False,
        critic_head_layer_norm: bool = False,
        fp32_heads: bool = False,
        fused_slot_encoder: bool = False,
    ):
        super().__init__()

        action_long = torch.tensor(binding.JERK_LONG, dtype=torch.float32)
        action_lat = torch.tensor(binding.JERK_LAT, dtype=torch.float32)

        long_norm = torch.where(
            action_long < 0.0,
            action_long / -action_long[0],
            action_long / action_long[-1],
        )
        lat_norm = action_lat / action_lat[-1]

        # row k: the [-1, 1] (long, lat) of discrete class k = long_idx * num_lat + lat_idx, the env's decode order
        num_lat = lat_norm.numel()
        num_classes = long_norm.numel() * num_lat
        k = torch.arange(num_classes)
        action_table = torch.stack([long_norm[k // num_lat], lat_norm[k % num_lat]], dim=-1)
        self.register_buffer("action_table", action_table, persistent=False)
        # The env decode is piecewise, so expectations must be taken in physical space, not normalized space.
        action_table_physical = torch.stack([action_long[k // num_lat], action_lat[k % num_lat]], dim=-1)
        self.register_buffer("action_table_physical", action_table_physical, persistent=False)
        self.action_long_neg_scale = float(-action_long[0])
        self.action_long_pos_scale = float(action_long[-1])
        self.action_lat_scale = float(action_lat[-1])

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
            "encoder_activation": encoder_activation,
            "encoder_layer_norm": encoder_layer_norm,
            "backbone_activation": backbone_activation,
            "backbone_layer_norm": backbone_layer_norm,
            "fused_slot_encoder": fused_slot_encoder,
        }
        self.actor_backbone = DriveBackbone(**backbone_args)
        self.critic_backbone = DriveBackbone(**backbone_args)

        self.is_continuous = action_type == "continuous"
        self.action_dim = env.single_action_space.shape[0] if self.is_continuous else num_classes
        actor_output_dim = self.action_dim * 2 if self.is_continuous else self.action_dim
        backbone_out_dim = self.actor_backbone.out_dim
        self.actor_head = build_mlp_head(
            backbone_out_dim, actor_hidden_size, actor_num_layers, actor_head_layer_norm, actor_output_dim, 0.01
        )
        self.critic_head = build_mlp_head(
            backbone_out_dim, critic_hidden_size, critic_num_layers, critic_head_layer_norm, 1, 1
        )
        self.fp32_heads = fp32_heads

    def forward(self, observations):
        actor_hidden = self.actor_backbone(observations)
        critic_hidden = self.critic_backbone(observations)
        return self._run_heads(actor_hidden, critic_hidden)

    def _run_heads(self, actor_hidden, critic_hidden):
        head_context = contextlib.nullcontext()
        if self.fp32_heads:
            head_context = torch.autocast(device_type=actor_hidden.device.type, enabled=False)
            actor_hidden = actor_hidden.float()
            critic_hidden = critic_hidden.float()
        with head_context:
            if self.is_continuous:
                params = self.actor_head(actor_hidden)
                loc, scale = torch.split(params, self.action_dim, dim=1)
                std = torch.nn.functional.softplus(scale) + 1e-4
                actions = torch.distributions.Normal(loc, std)
            else:
                actions = self.actor_head(actor_hidden)
            value = self.critic_head(critic_hidden)
        return actions, value

    def forward_eval(self, x):
        return self.forward(x)

    def matmul_flops_per_transition(self):
        """Matmul FLOPs of one observation row through actor and critic: (forward, forward + backward)."""
        forward_flops = 0.0
        backward_flops = 0.0
        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            network_name, _, layer_path = name.partition(".")
            encoder_name = layer_path.partition(".")[0]
            rows_per_transition = 1
            if encoder_name in SLOT_COUNT_ATTRIBUTE_BY_ENCODER:
                network = getattr(self, network_name)
                rows_per_transition = getattr(network, SLOT_COUNT_ATTRIBUTE_BY_ENCODER[encoder_name])
            weight_count = module.in_features * module.out_features
            layer_flops = opengigaflow.pytorch.FLOPS_PER_MULTIPLY_ADD * rows_per_transition * weight_count
            # observations need no gradient, so an encoder's first layer only computes its weight gradient
            reads_observations = encoder_name.endswith("_encoder") and layer_path.endswith(".0")
            weight_gradient_flops = layer_flops
            input_gradient_flops = 0.0 if reads_observations else layer_flops
            forward_flops += layer_flops
            backward_flops += weight_gradient_flops + input_gradient_flops
        return forward_flops, forward_flops + backward_flops

    def pool_slot_counts(self, observations):
        return self.actor_backbone.pool_slot_counts(observations)

    def discrete_actions_to_continuous(self, actions):
        return self.action_table[actions.long()]

    def discrete_probs_to_continuous_mean(self, probs):
        mean_physical = probs @ self.action_table_physical.to(probs.dtype)
        mean_long = mean_physical[..., 0]
        mean_long_norm = torch.where(
            mean_long < 0.0,
            mean_long / self.action_long_neg_scale,
            mean_long / self.action_long_pos_scale,
        )
        # low-precision matmul can overshoot the [-1, 1] bounds by an epsilon
        return torch.stack([mean_long_norm, mean_physical[..., 1] / self.action_lat_scale], dim=-1).clamp_(-1.0, 1.0)
