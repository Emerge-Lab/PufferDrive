import math

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
LATTICE_MASK_ENCODER_SIZE = 64
LATTICE_MASKED_LOGIT = -1e9
LATTICE_SPECIAL_CELL_LOGIT_BIAS = -2.0
# structured head factors; the cell layout they index is init_lattice_config in lattice.h
LATTICE_KIND_OFFSET, LATTICE_KIND_LANE_RIGHT, LATTICE_KIND_LANE_LEFT, LATTICE_KIND_ONCOMING = range(4)
(
    LATTICE_MODE_CRUISE,
    LATTICE_MODE_STOP,
    LATTICE_MODE_STOP_LINE,
    LATTICE_MODE_EMERGENCY,
    LATTICE_MODE_BACKUP,
    LATTICE_MODE_TURN,
) = range(6)
LATTICE_BELL_MIN_SIGMA_SPACINGS = 0.35
LATTICE_BELL_MAX_SIGMA_RANGES = 2.0
LATTICE_BELL_INIT_SIGMA_RANGES = 0.4
LATTICE_BELL_CENTRE_MARGIN_SPACINGS = 2.0


def masked_log_softmax(logits, allowed, dim):
    """log_softmax over the allowed entries; a fully disallowed slice stays finite (its cells are masked later)."""
    return F.log_softmax(logits.masked_fill(~allowed, LATTICE_MASKED_LOGIT), dim=dim)


class LatticeBell:
    """Bell curve over an ascending menu: centre in [first, last] widened by a margin, spread log-bounded, both smooth."""

    def __init__(self, values):
        self.values = [float(value) for value in values]
        spacing = min((later - earlier for earlier, later in zip(self.values, self.values[1:])), default=1.0)
        span = max(self.values[-1] - self.values[0], spacing)
        margin = LATTICE_BELL_CENTRE_MARGIN_SPACINGS * spacing
        self.centre_low, self.centre_high = self.values[0] - margin, self.values[-1] + margin
        self.log_sigma_min = math.log(LATTICE_BELL_MIN_SIGMA_SPACINGS * spacing)
        self.log_sigma_max = math.log(LATTICE_BELL_MAX_SIGMA_RANGES * span)
        init_fraction = (math.log(LATTICE_BELL_INIT_SIGMA_RANGES * span) - self.log_sigma_min) / (
            self.log_sigma_max - self.log_sigma_min
        )
        self.init_sigma_raw = math.log(init_fraction / (1.0 - init_fraction))

    def centre_and_spread(self, centre_raw, spread_raw):
        centre = self.centre_low + (self.centre_high - self.centre_low) * torch.sigmoid(centre_raw)
        log_sigma = self.log_sigma_min + (self.log_sigma_max - self.log_sigma_min) * torch.sigmoid(spread_raw)
        return centre, log_sigma.exp()

    def logits(self, values, centre_raw, spread_raw):
        centre, sigma = self.centre_and_spread(centre_raw, spread_raw)
        z = (values.unsqueeze(0) - centre.unsqueeze(1)) / sigma.unsqueeze(1)
        return -0.5 * z * z


class LatticeStructuredHead(nn.Module):
    """Head outputs -> exact log-probabilities over the env's lattice cells (layout of init_lattice_config, lattice.h),
    chained level by level with the masks applied at every level: lateral kind -> in-lane offset (bell) -> duration;
    longitudinal mode -> speed (bell) -> comfort level/duration, stop distance (bell over ln m), back-up distance."""

    def __init__(self, env):
        super().__init__()
        offsets = [float(value) for value in env.lattice_lat_offsets_m]
        speeds = [float(value) for value in env.lattice_lon_speeds_mps]
        log_stops = [math.log(float(value)) for value in env.lattice_stop_distances_m]
        for name, values in (("lat_offsets_m", offsets), ("lon_speeds_mps", speeds), ("stop_distances_m", log_stops)):
            if any(later <= earlier for earlier, later in zip(values, values[1:])):
                raise ValueError(f"policy.lattice_structured_head needs strictly ascending env.lattice_{name}")
        self.nvec = [int(count) for count in env.lattice_nvec]
        self.offset_bell, self.speed_bell, self.stop_bell = (
            LatticeBell(offsets),
            LatticeBell(speeds),
            LatticeBell(log_stops),
        )
        self.register_buffer("offset_values", torch.tensor(offsets), persistent=False)
        self.register_buffer("speed_values", torch.tensor(speeds), persistent=False)
        self.register_buffer("stop_values", torch.tensor(log_stops), persistent=False)
        self.oncoming, self.turnaround = int(env.lattice_oncoming_overtake), int(env.lattice_turnaround)
        self.offset_count, self.lat_duration_count = len(offsets), len(env.lattice_lat_durations_s)
        self.choice_count = self.offset_count + 2 + self.oncoming
        self.speed_count, self.slot_count = len(speeds), int(env.lattice_lon_slot_count)
        self.stop_count, self.backup_count = len(log_stops), len(env.lattice_backup_distances_m)
        if self.choice_count * self.lat_duration_count != self.nvec[1]:
            raise ValueError(
                f"lateral cells {self.nvec[1]} != {self.choice_count} choices x {self.lat_duration_count} durations"
            )
        if (
            self.speed_count * self.slot_count + self.stop_count + 2 + self.backup_count + self.turnaround
            != self.nvec[3]
        ):
            raise ValueError(f"longitudinal cells {self.nvec[3]} do not match the menus")
        self.layout = [
            ("lat_gate", self.nvec[0]),
            ("lat_kind", LATTICE_KIND_ONCOMING + self.oncoming),
            ("offset_bell", 2),
            ("lat_duration", self.lat_duration_count),
            ("lon_gate", self.nvec[2]),
            ("lon_mode", LATTICE_MODE_TURN + self.turnaround),
            ("speed_bell", 2),
            ("lon_slot", self.slot_count),
            ("stop_bell", 2),
            ("backup", self.backup_count),
            ("exit", self.nvec[4]),
        ]
        self.param_count = sum(size for _, size in self.layout)
        # lateral choices (cell = choice_count * duration + choice): lane right, offsets, lane left, [oncoming]
        choice_kind = [LATTICE_KIND_LANE_RIGHT] + [LATTICE_KIND_OFFSET] * self.offset_count + [LATTICE_KIND_LANE_LEFT]
        choice_kind += [LATTICE_KIND_ONCOMING] * self.oncoming
        kind_count = LATTICE_KIND_ONCOMING + self.oncoming
        self.register_buffer("choice_kind", torch.tensor(choice_kind, dtype=torch.long), persistent=False)
        self.register_buffer("choice_kind_onehot", F.one_hot(self.choice_kind, kind_count).float(), persistent=False)
        self.register_buffer("prior", self.initial_prior(), persistent=False)

    def initial_prior(self):
        """Constant offsets added to the head output: unmasked factors start at the linear head's initial cell
        distribution for the same menu (uniform cells, special longitudinal cells at LATTICE_SPECIAL_CELL_LOGIT_BIAS),
        bells near-flat at the menu midpoint. Weight decay pulls the learned part back toward this prior."""
        kind_choices = [self.offset_count, 1, 1] + [1] * self.oncoming
        mode_cells = [self.speed_count * self.slot_count, self.stop_count, 1, 1, self.backup_count] + [
            1
        ] * self.turnaround
        mode_bias = [0.0] + [LATTICE_SPECIAL_CELL_LOGIT_BIAS] * (len(mode_cells) - 1)
        values = {
            "lat_kind": [math.log(count) for count in kind_choices],
            "lon_mode": [math.log(count) + bias for count, bias in zip(mode_cells, mode_bias)],
            "offset_bell": [0.0, self.offset_bell.init_sigma_raw],
            "speed_bell": [0.0, self.speed_bell.init_sigma_raw],
            "stop_bell": [0.0, self.stop_bell.init_sigma_raw],
        }
        return torch.cat(
            [torch.tensor(values.get(name, [0.0] * size), dtype=torch.float32) for name, size in self.layout]
        )

    def split(self, params):
        sizes = [size for _, size in self.layout]
        return dict(zip((name for name, _ in self.layout), torch.split(params.float() + self.prior, sizes, dim=1)))

    def lateral_cells(self, parts, allowed):
        batch = allowed.shape[0]
        cell_allowed = allowed.view(batch, self.lat_duration_count, self.choice_count)
        choice_allowed = cell_allowed.any(dim=1)
        kind_allowed = (choice_allowed.float() @ self.choice_kind_onehot) > 0.0
        kind_lp = masked_log_softmax(parts["lat_kind"], kind_allowed, dim=1)
        offset_logits = self.offset_bell.logits(
            self.offset_values, parts["offset_bell"][:, 0], parts["offset_bell"][:, 1]
        )
        offset_lp = masked_log_softmax(offset_logits, choice_allowed[:, 1 : 1 + self.offset_count], dim=1)
        lane_and_oncoming = offset_lp.new_zeros(batch, self.choice_count - 1 - self.offset_count)
        choice_lp = kind_lp[:, self.choice_kind] + torch.cat(
            [offset_lp.new_zeros(batch, 1), offset_lp, lane_and_oncoming], dim=1
        )
        duration_logits = parts["lat_duration"].unsqueeze(2).expand(-1, -1, self.choice_count)
        duration_lp = masked_log_softmax(duration_logits, cell_allowed, dim=1)
        return (choice_lp.unsqueeze(1) + duration_lp).reshape(batch, -1)

    def longitudinal_cells(self, parts, allowed):
        batch = allowed.shape[0]
        speed_cells = self.speed_count * self.slot_count
        stop_end = speed_cells + self.stop_count
        backup_start = stop_end + 2
        speed_allowed_cells = allowed[:, :speed_cells].view(batch, self.slot_count, self.speed_count)
        speed_allowed = speed_allowed_cells.any(dim=1)
        stop_allowed = allowed[:, speed_cells:stop_end]
        backup_allowed = allowed[:, backup_start : backup_start + self.backup_count]
        mode_allowed = [speed_allowed.any(1), stop_allowed.any(1), allowed[:, stop_end], allowed[:, stop_end + 1]]
        mode_allowed += [backup_allowed.any(1)] + ([allowed[:, -1]] if self.turnaround else [])
        mode_lp = masked_log_softmax(parts["lon_mode"], torch.stack(mode_allowed, dim=1), dim=1)
        speed_logits = self.speed_bell.logits(self.speed_values, parts["speed_bell"][:, 0], parts["speed_bell"][:, 1])
        speed_lp = masked_log_softmax(speed_logits, speed_allowed, dim=1)
        slot_logits = parts["lon_slot"].unsqueeze(2).expand(-1, -1, self.speed_count)
        slot_lp = masked_log_softmax(slot_logits, speed_allowed_cells, dim=1)
        cruise = mode_lp[:, LATTICE_MODE_CRUISE, None, None] + speed_lp.unsqueeze(1) + slot_lp
        stop_logits = self.stop_bell.logits(self.stop_values, parts["stop_bell"][:, 0], parts["stop_bell"][:, 1])
        stops = mode_lp[:, LATTICE_MODE_STOP, None] + masked_log_softmax(stop_logits, stop_allowed, dim=1)
        backups = mode_lp[:, LATTICE_MODE_BACKUP, None] + masked_log_softmax(parts["backup"], backup_allowed, dim=1)
        singles = [mode_lp[:, LATTICE_MODE_STOP_LINE, None], mode_lp[:, LATTICE_MODE_EMERGENCY, None]]
        turn = [mode_lp[:, LATTICE_MODE_TURN, None]] if self.turnaround else []
        return torch.cat([cruise.reshape(batch, -1), stops, *singles, backups, *turn], dim=1)

    def forward(self, params, lattice_masks):
        """Per-factor logits in env order; every allowed cell gets its exact chained log-probability."""
        parts = self.split(params)
        allowed = lattice_masks > 0.5
        nvec = self.nvec
        lat_start, lon_start = nvec[0], sum(nvec[:3])
        lat = self.lateral_cells(parts, allowed[:, lat_start : lat_start + nvec[1]])
        lon = self.longitudinal_cells(parts, allowed[:, lon_start : lon_start + nvec[3]])
        return torch.cat([parts["lat_gate"], lat, parts["lon_gate"], lon, parts["exit"]], dim=1)


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
        # Lattice action masks are appended after every other feature and also feed the encoder
        self.lattice_mask_features = getattr(env, "lattice_mask_features", 0)
        if self.lattice_mask_features > 0:
            self.lattice_mask_encoder = self._create_encoder(self.lattice_mask_features, LATTICE_MASK_ENCODER_SIZE)
            encoders_out += LATTICE_MASK_ENCODER_SIZE

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
        if self.lattice_mask_features > 0:
            feature_list.append(self.lattice_mask_encoder(observations[:, -self.lattice_mask_features :]))

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
        actor_head_layer_norm: bool = False,
        critic_head_layer_norm: bool = False,
        lattice_structured_head: bool = False,
    ):
        super().__init__()

        # spline dynamics is continuous-only; the jerk table is never indexed there, it only fills the unused buffers
        if env.dynamics_model_flag in (
            binding.DYNAMICS_MODEL_JERK,
            binding.DYNAMICS_MODEL_SPLINE,
            binding.DYNAMICS_MODEL_SPLINE_WERLING,
        ):
            action_long_values, action_lat_values = binding.JERK_LONG, binding.JERK_LAT
        elif env.dynamics_model_flag == binding.DYNAMICS_MODEL_CLASSIC:
            action_long_values, action_lat_values = binding.ACCELERATION_VALUES, binding.STEERING_VALUES
        else:
            raise ValueError(f"Unsupported dynamics model: {env.dynamics_model}")

        action_long = torch.tensor(action_long_values, dtype=torch.float32)
        action_lat = torch.tensor(action_lat_values, dtype=torch.float32)

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
        # The env decode is piecewise, so expectations must be taken in physical space, not normalized space.
        action_table_physical = torch.stack([action_long[k // num_lat], action_lat[k % num_lat]], dim=-1)
        self.register_buffer("action_table_physical", action_table_physical, persistent=False)
        self.action_long_neg_scale = float(-action_long[0])
        self.action_long_pos_scale = float(action_long[-1])
        self.action_lat_scale = float(action_lat[-1])

        # Configuration flags from policy kwargs
        self.shared_network = shared_network
        # The spline intent block sits directly after the ego block in the C layout, so it must be
        # counted here or every downstream slice (context/partner/road/traffic) shifts by its width.
        self.ego_dim = env.ego_features + env.spline_intent_features + getattr(env, "lattice_plan_features", 0)
        self.is_lattice = action_type == "lattice"
        # LSTMWrapper hands its state to decode_actions so the masks never live on self
        self.decode_uses_state = self.is_lattice
        if lattice_structured_head and not self.is_lattice:
            raise ValueError("policy.lattice_structured_head needs action_type 'lattice'")
        self.lattice_structured = LatticeStructuredHead(env) if lattice_structured_head else None
        if self.is_lattice:
            self.lattice_nvec = [int(n) for n in env.lattice_nvec]
            self.lattice_mask_features = sum(self.lattice_nvec)

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

        # Setup action and value heads. "continuous" and "spline" both parameterize a diagonal
        # Gaussian (Normal(loc, scale)) over env.single_action_space.shape[0] dims — spline just
        # has more dims (6, vs continuous's 2) and needs no distribution code of its own.
        # Deliberately kept named is_continuous (not e.g. is_gaussian) rather than renamed: this
        # attribute is a de facto shared contract with generic infra outside this file
        # (pufferlib.models.LSTMWrapper copies it at construction time; the WOSAC evaluator
        # reads it directly) that doesn't know about "spline" — widening what True covers here
        # is far lower blast-radius than renaming it everywhere those other consumers live.
        self.is_continuous = action_type in ("continuous", "spline")
        if self.is_continuous:
            self.action_dim = env.single_action_space.shape[0]
        elif self.is_lattice:
            self.action_dim = (
                self.lattice_structured.param_count
                if self.lattice_structured is not None
                else self.lattice_mask_features
            )
        else:
            self.action_dim = self.action_long_norm.numel() * self.action_lat_norm.numel()

        # n-layer MLP for actor head (num_layers = number of hidden layers)
        backbone_out_dim = self.actor_backbone.out_dim
        actor_head_layers = []
        actor_in = backbone_out_dim
        for _ in range(actor_num_layers):
            actor_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(actor_in, actor_hidden_size)))
            if actor_head_layer_norm:
                actor_head_layers.append(nn.LayerNorm(actor_hidden_size))
            actor_head_layers.append(nn.ReLU())
            actor_in = actor_hidden_size
        actor_output_dim = self.action_dim * 2 if self.is_continuous else self.action_dim
        actor_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(actor_in, actor_output_dim), std=0.01))
        self.actor_head = nn.Sequential(*actor_head_layers)
        if self.is_lattice and self.lattice_structured is None:
            # gates start unbiased; stop, stop-line, EMERGENCY and back-up cells start unlikely
            lon_cell_offset = sum(self.lattice_nvec[:3])
            first_special_cell = len(env.lattice_lon_speeds_mps) * env.lattice_lon_slot_count
            with torch.no_grad():
                self.actor_head[-1].bias[
                    lon_cell_offset + first_special_cell : lon_cell_offset + self.lattice_nvec[3]
                ] = LATTICE_SPECIAL_CELL_LOGIT_BIAS

        # n-layer MLP for critic head (num_layers = number of hidden layers)
        critic_head_layers = []
        critic_in = backbone_out_dim
        for _ in range(critic_num_layers):
            critic_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(critic_in, critic_hidden_size)))
            if critic_head_layer_norm:
                critic_head_layers.append(nn.LayerNorm(critic_hidden_size))
            critic_head_layers.append(nn.ReLU())
            critic_in = critic_hidden_size
        critic_head_layers.append(pufferlib.pytorch.layer_init(nn.Linear(critic_in, 1), std=1))
        self.critic_head = nn.Sequential(*critic_head_layers)

    def _decode_actions(self, hidden, lattice_masks=None):
        """Shared by forward() and decode_actions(): Normal(loc, scale) for a Gaussian action
        type (continuous or spline), masked per-factor logits for lattice, raw logits otherwise."""
        if self.is_lattice:
            if self.lattice_structured is None:
                logits = self.actor_head(hidden)
            else:
                # last layer and bells in fp32: bf16 rounding of a narrow bell's centre moves log-probs by ~0.03 nats
                head_hidden = self.actor_head[:-1](hidden)
                with torch.autocast(device_type=head_hidden.device.type, enabled=False):
                    params = self.actor_head[-1](head_hidden.float())
                    logits = self.lattice_structured(params, lattice_masks.to(params.device))
            masked = logits.masked_fill(lattice_masks.to(logits.device) < 0.5, LATTICE_MASKED_LOGIT)
            return pufferlib.pytorch.LatticeLogits(*torch.split(masked, self.lattice_nvec, dim=1))
        if self.is_continuous:
            params = self.actor_head(hidden)
            loc, scale = torch.split(params, self.action_dim, dim=1)
            # softplus is unbounded above; observed run away to std ~1e3 (entropy ~54 nats over
            # 6 dims, vs. ~6 nats at init) once a reward term was badly scaled, and never
            # recovered even after 300+ epochs. Actions are clipped to [-1,1] regardless, so a
            # std beyond that range buys no real exploration -- clamp comfortably above it.
            std = torch.nn.functional.softplus(scale).clamp(max=2.0) + 1e-4
            return torch.distributions.Normal(loc, std)
        return self.actor_head(hidden)

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
        lattice_masks = observations[:, -self.lattice_mask_features :] if self.is_lattice else None
        actions = self._decode_actions(actor_hidden, lattice_masks)

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
        if self.is_lattice:
            state["lattice_mask"] = observations[:, -self.lattice_mask_features :]
        return self.actor_backbone(observations, self.ego_dim)

    def decode_actions(self, hidden, state=None):
        """
        USE ONLY FOR LSTM WRAPPER.
        Decodes actions and value from the hidden state.
        Args:
            hidden: The hidden state for the actor (policy).
            state: the wrapper state; carries the lattice masks stored by encode_observations.
        """
        action = self._decode_actions(hidden, state["lattice_mask"] if self.is_lattice else None)
        value = self.critic_head(hidden)

        return action, value

    def discrete_actions_to_continuous(self, actions):
        return self.action_table[actions.long()]

    def discrete_probs_to_continuous_mean(self, probs):
        # probs: [..., num_classes] -> [..., 2]
        mean_physical = probs @ self.action_table_physical.to(probs.dtype)
        mean_long = mean_physical[..., 0]
        mean_long_norm = torch.where(
            mean_long < 0.0,
            mean_long / self.action_long_neg_scale,
            mean_long / self.action_long_pos_scale,
        )
        # low-precision matmul can overshoot the [-1, 1] bounds by an epsilon
        return torch.stack([mean_long_norm, mean_physical[..., 1] / self.action_lat_scale], dim=-1).clamp_(-1.0, 1.0)
