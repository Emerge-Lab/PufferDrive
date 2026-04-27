import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


_ACTIVATIONS = {
    "none": nn.Identity,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    def __init__(
        self,
        in_size,
        hidden_size,
        out_size,
        num_layers=1,
        final_std=0.01,
        activation="gelu",
    ):
        super().__init__()
        activation_cls = _ACTIVATIONS[activation]
        layers = []
        prev = in_size
        for _ in range(num_layers):
            layers += [_layer_init(nn.Linear(prev, hidden_size)), activation_cls()]
            prev = hidden_size
        layers.append(_layer_init(nn.Linear(prev, out_size), std=final_std))
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h)


class MinGRU(nn.Module):
    # https://arxiv.org/abs/2410.01201v1
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Linear(hidden_size, 3 * hidden_size, bias=False) for _ in range(num_layers)])

    def _g(self, x):
        return torch.where(x >= 0, x + 0.5, x.sigmoid())

    def _log_g(self, x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def _highway(self, x, out, proj):
        g = proj.sigmoid()
        return g * out + (1.0 - g) * x

    def _heinsen_scan(self, log_coeffs, log_values):
        a_star = log_coeffs.cumsum(dim=1)
        return (a_star + (log_values - a_star).logcumsumexp(dim=1)).exp()

    def initial_state(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),)

    def forward_eval(self, h, state):
        state = state[0]
        assert state.shape[1] == h.shape[0]
        h = h.unsqueeze(1)
        state_out = []
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            out = torch.lerp(state[i : i + 1].transpose(0, 1), self._g(hidden), gate.sigmoid())
            h = self._highway(h, out, proj)
            state_out.append(out[:, -1:])
        return h.squeeze(1), (torch.stack(state_out, 0).squeeze(2),)

    def forward_train(self, h):
        T = h.shape[1]
        for i in range(self.num_layers):
            hidden, gate, proj = self.layers[i](h).chunk(3, dim=-1)
            log_coeffs = -F.softplus(gate)
            log_values = -F.softplus(-gate) + self._log_g(hidden)
            out = self._heinsen_scan(log_coeffs, log_values)[:, -T:]
            h = self._highway(h, out, proj)
        return h


class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.LSTMCell(hidden_size, hidden_size) for _ in range(num_layers)])

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.lstm, f"weight_ih_l{i}")
            w_hh = getattr(self.lstm, f"weight_hh_l{i}")
            b_ih = getattr(self.lstm, f"bias_ih_l{i}")
            b_hh = getattr(self.lstm, f"bias_hh_l{i}")
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def forward_eval(self, h, state):
        assert state[0].shape[1] == state[1].shape[1] == h.shape[0]
        lstm_h, lstm_c = state
        for i in range(self.num_layers):
            h, c = self.cell[i](h, (lstm_h[i], lstm_c[i]))
            lstm_h[i] = h
            lstm_c[i] = c
        return h, (lstm_h, lstm_c)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h, _ = self.lstm(h)
        return h.transpose(0, 1)


class GRU(nn.Module):
    def __init__(self, hidden_size, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.cell = nn.ModuleList([torch.nn.GRUCell(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norm = torch.nn.RMSNorm(hidden_size)

        for i in range(num_layers):
            cell = self.cell[i]
            w_ih = getattr(self.gru, f"weight_ih_l{i}")
            w_hh = getattr(self.gru, f"weight_hh_l{i}")
            b_ih = getattr(self.gru, f"bias_ih_l{i}")
            b_hh = getattr(self.gru, f"bias_hh_l{i}")
            nn.init.orthogonal_(w_ih, 1.0)
            nn.init.orthogonal_(w_hh, 1.0)
            b_ih.data.zero_()
            b_hh.data.zero_()
            cell.weight_ih = w_ih
            cell.weight_hh = w_hh
            cell.bias_ih = b_ih
            cell.bias_hh = b_hh

    def initial_state(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h,)

    def forward_eval(self, h, state):
        assert state[0].shape[1] == h.shape[0]
        state = state[0]
        for i in range(self.num_layers):
            h_in = h
            h = self.cell[i](h, state[i])
            state[i] = h
            h = h + h_in
            h = self.norm(h)
        return h, (state,)

    def forward_train(self, h):
        # h: [B, T, H]
        h = h.transpose(0, 1)
        h_in = h
        h, _ = self.gru(h)
        h = h + h_in
        h = self.norm(h)
        return h.transpose(0, 1)


_RNN_CLS = {"LSTM": LSTM, "GRU": GRU, "MinGRU": MinGRU}
_STATE_LEN = {"MLP": 0, "LSTM": 2, "GRU": 1, "MinGRU": 1}


class Trunk(nn.Module):
    """MLP / LSTM / GRU / MinGRU. Takes [B(,T), in_size] -> [B(,T), hidden_size]."""

    def __init__(self, trunk_type, in_size, hidden_size, num_layers, activation):
        super().__init__()
        self.trunk_type = trunk_type
        if trunk_type == "MLP":
            self.mlp = nn.Sequential(
                MLP(
                    in_size,
                    hidden_size,
                    hidden_size,
                    num_layers=num_layers - 1,
                    final_std=np.sqrt(2),
                    activation=activation,
                ),
                _ACTIVATIONS[activation](),
            )
            self.rnn = None
        else:
            self.mlp = None
            self.input_projection = (
                nn.Identity() if in_size == hidden_size else _layer_init(nn.Linear(in_size, hidden_size))
            )
            self.rnn = _RNN_CLS[trunk_type](hidden_size, num_layers=num_layers)

    def initial_state(self, batch_size, device):
        return () if self.rnn is None else self.rnn.initial_state(batch_size, device)

    def forward_eval(self, hidden, state):
        if self.rnn is None:
            return self.mlp(hidden), state
        return self.rnn.forward_eval(self.input_projection(hidden), state)

    def forward_train(self, hidden):
        if self.rnn is None:
            return self.mlp(hidden)
        return self.rnn.forward_train(self.input_projection(hidden))


class VecEncoder(nn.Module):
    """Per-stream MLP + masked max-pool; concat to [B, sum_H]."""

    obs_stream_names = ["ego", "target", "partner", "lane", "boundary", "tc", "counts"]

    def __init__(self, env_constants, **policy_kwargs):
        super().__init__()
        self.env_constants = {k: int(v) for k, v in env_constants.items()}
        self.mask_padded = policy_kwargs["mask_padded_observations"]
        dropout = policy_kwargs["dropout"]
        self.activation_fn = _ACTIVATIONS[policy_kwargs["encoder_activation"]]

        is_active = lambda in_features, num_slots: in_features > 0 and (num_slots is None or num_slots > 0)
        self.encoder_specs = []  # (name, in_features, num_slots_or_None, hidden_size)
        self.encoders = nn.ModuleDict()
        for name, in_features, num_slots in self._encoder_descriptions:
            if not is_active(in_features, num_slots):
                continue
            hidden_size = policy_kwargs[f"{name}_hidden_size"]
            num_layers = policy_kwargs[f"{name}_num_layers"]
            self.encoders[name] = self._create_encoder(in_features, hidden_size, num_layers, dropout)
            self.encoder_specs.append((name, in_features, num_slots, hidden_size))

        target_total = self.env_constants["target_features"] * self.env_constants["num_target_waypoints"]
        self.obs_split_sizes = [
            self.env_constants["ego_features"],
            self.env_constants["num_reward_coefs"] + target_total,
            self.env_constants["obs_partner_slots"] * self.env_constants["partner_features"],
            self.env_constants["obs_lane_slots"] * self.env_constants["road_features"],
            self.env_constants["obs_boundary_slots"] * self.env_constants["road_features"],
            self.env_constants["obs_traffic_control_slots"] * self.env_constants["traffic_control_features"],
            self.env_constants["obs_count_features"],
        ]
        self._slot_caps = {
            "partner": self.env_constants["obs_partner_slots"],
            "target": self.env_constants["num_target_waypoints"],
            "lane": self.env_constants["obs_lane_slots"],
            "boundary": self.env_constants["obs_boundary_slots"],
            "tc": self.env_constants["obs_traffic_control_slots"],
        }
        self.register_buffer("_slot_arange", torch.arange(max(self._slot_caps.values(), default=0)), persistent=False)
        self.out_size = sum(spec[3] for spec in self.encoder_specs)

    @property
    def _encoder_descriptions(self):
        # (name, in_features, num_slots). num_slots is None for scalar streams (no pooling).
        target_total = self.env_constants["target_features"] * self.env_constants["num_target_waypoints"]
        return [
            ("ego", self.env_constants["ego_features"], None),
            ("target", target_total, None),
            ("partner", self.env_constants["partner_features"], self.env_constants["obs_partner_slots"]),
            ("lane", self.env_constants["road_features"], self.env_constants["obs_lane_slots"]),
            ("boundary", self.env_constants["road_features"], self.env_constants["obs_boundary_slots"]),
            ("tc", self.env_constants["traffic_control_features"], self.env_constants["obs_traffic_control_slots"]),
        ]

    def _create_encoder(self, in_features, hidden_size, num_layers, dropout):
        layers = [_layer_init(nn.Linear(in_features, hidden_size))]
        for _ in range(num_layers - 1):
            layers += [
                nn.LayerNorm(hidden_size),
                self.activation_fn(),
                nn.Dropout(dropout),
                _layer_init(nn.Linear(hidden_size, hidden_size)),
            ]
        layers.append(self.activation_fn())
        return nn.Sequential(*layers)

    def _pool(self, flat_slots, num_slots, num_features, hidden_size, encoder_layer, valid_count):
        slots = flat_slots.view(-1, num_slots, num_features)
        valid_mask = self._slot_arange[:num_slots] < valid_count.unsqueeze(1)
        if self.mask_padded:
            encoded = slots.new_full((slots.shape[0], num_slots, hidden_size), torch.finfo(slots.dtype).min)
            encoded[valid_mask] = encoder_layer(slots[valid_mask])
        else:
            encoded = encoder_layer(slots)
            encoded = encoded.masked_fill(~valid_mask.unsqueeze(-1), torch.finfo(slots.dtype).min)
        pooled = encoded.amax(dim=1)
        return torch.where(valid_count.unsqueeze(1) == 0, pooled.new_zeros(pooled.shape), pooled)

    def forward(self, obs, counts=None):
        obs = obs.float()
        obs_by_stream = dict(zip(self.obs_stream_names, torch.split(obs, self.obs_split_sizes, dim=1)))
        if counts is None:
            lane_count, boundary_count, partner_count, tc_count = obs_by_stream["counts"].long().unbind(dim=1)
        else:
            lane_count, boundary_count, partner_count, tc_count = counts
        counts_by_stream = {
            "lane": lane_count.clamp(0, self._slot_caps["lane"]),
            "boundary": boundary_count.clamp(0, self._slot_caps["boundary"]),
            "partner": partner_count.clamp(0, self._slot_caps["partner"]),
            "tc": tc_count.clamp(0, self._slot_caps["tc"]),
        }

        features = []
        for name, in_features, num_slots, hidden_size in self.encoder_specs:
            if num_slots is None:
                features.append(self.encoders[name](obs_by_stream[name]))
            else:
                features.append(
                    self._pool(
                        obs_by_stream[name],
                        num_slots,
                        in_features,
                        hidden_size,
                        self.encoders[name],
                        counts_by_stream[name],
                    )
                )
        return torch.cat(features, dim=1)


class Policy(nn.Module):
    """Encoder(s) -> Trunk(s) -> {actor_head, critic_head}.

    Sharing axes: shared_encoder, shared_trunk. Heads are always separate.
    (shared_encoder=False, shared_trunk=True) is rejected.
    """

    def __init__(self, vec, **policy_kwargs):
        super().__init__()
        self.shared_encoder = policy_kwargs["shared_encoder"]
        self.shared_trunk = policy_kwargs["shared_trunk"]
        assert self.shared_encoder or not self.shared_trunk, "shared_trunk requires shared_encoder"

        env_constants = vec.env_constants()
        self.encoder = VecEncoder(env_constants, **policy_kwargs)
        self.critic_encoder = self.encoder if self.shared_encoder else VecEncoder(env_constants, **policy_kwargs)

        trunk_type = policy_kwargs["trunk_type"]
        trunk_hidden_size = policy_kwargs["trunk_hidden_size"]
        trunk_num_layers = policy_kwargs["trunk_num_layers"]
        trunk_activation = policy_kwargs["trunk_activation"]
        trunk_in_size = self.encoder.out_size
        self.trunk = Trunk(trunk_type, trunk_in_size, trunk_hidden_size, trunk_num_layers, trunk_activation)
        self.critic_trunk = (
            self.trunk
            if self.shared_trunk
            else Trunk(trunk_type, trunk_in_size, trunk_hidden_size, trunk_num_layers, trunk_activation)
        )
        self.actor_state_len = 0 if self.shared_trunk else _STATE_LEN[trunk_type]

        action_sizes = tuple(int(size) for size in vec.act_sizes)
        self.action_sizes = action_sizes
        self.is_continuous = sum(action_sizes) == len(action_sizes)
        actor_out_size = len(action_sizes) if self.is_continuous else sum(action_sizes)
        self.actor_head = MLP(
            trunk_hidden_size,
            policy_kwargs["actor_hidden_size"],
            actor_out_size,
            num_layers=policy_kwargs["actor_num_layers"],
            final_std=0.01,
            activation=policy_kwargs["actor_activation"],
        )
        self.critic_head = MLP(
            trunk_hidden_size,
            policy_kwargs["critic_hidden_size"],
            1,
            num_layers=policy_kwargs["critic_num_layers"],
            final_std=1.0,
            activation=policy_kwargs["critic_activation"],
        )
        if self.is_continuous:
            self.actor_logstd = nn.Parameter(torch.zeros(1, len(action_sizes)))

    def initial_state(self, batch_size, device):
        actor_state = self.trunk.initial_state(batch_size, device)
        if self.shared_trunk:
            return actor_state
        critic_state = self.critic_trunk.initial_state(batch_size, device)
        return tuple(actor_state) + tuple(critic_state)

    def _heads(self, actor_hidden, critic_hidden):
        if self.is_continuous:
            action_mean = self.actor_head(actor_hidden)
            logits = torch.distributions.Normal(action_mean, self.actor_logstd.expand_as(action_mean).exp())
        else:
            actor_logits = self.actor_head(actor_hidden)
            logits = torch.split(actor_logits, self.action_sizes, dim=-1)
        return logits, self.critic_head(critic_hidden)

    def forward_eval(self, obs, state):
        actor_hidden = self.encoder(obs)
        critic_hidden = actor_hidden if self.shared_encoder else self.critic_encoder(obs)
        if self.shared_trunk:
            actor_hidden, state = self.trunk.forward_eval(actor_hidden, state)
            critic_hidden = actor_hidden
        else:
            actor_state = state[: self.actor_state_len]
            critic_state = state[self.actor_state_len :]
            actor_hidden, actor_state = self.trunk.forward_eval(actor_hidden, actor_state)
            critic_hidden, critic_state = self.critic_trunk.forward_eval(critic_hidden, critic_state)
            state = tuple(actor_state) + tuple(critic_state)
        logits, value = self._heads(actor_hidden, critic_hidden)
        return logits, value, state

    def forward(self, obs):
        batch_size, time_steps = obs.shape[:2]
        flat_obs = obs.reshape(batch_size * time_steps, *obs.shape[2:])
        actor_hidden = self.encoder(flat_obs)
        critic_hidden = actor_hidden if self.shared_encoder else self.critic_encoder(flat_obs)
        actor_hidden = self.trunk.forward_train(actor_hidden.reshape(batch_size, time_steps, -1))
        critic_hidden = (
            actor_hidden
            if self.shared_trunk
            else self.critic_trunk.forward_train(critic_hidden.reshape(batch_size, time_steps, -1))
        )
        logits, value = self._heads(
            actor_hidden.reshape(batch_size * time_steps, -1),
            critic_hidden.reshape(batch_size * time_steps, -1),
        )
        return logits, value.reshape(batch_size, time_steps)


class NatureEncoder(nn.Module):
    """NatureCNN encoder (Mnih et al. 2015). Returns [batch, hidden_size]."""

    def __init__(
        self, env, hidden_size=512, framestack=1, flat_size=64 * 7 * 7, channels_last=False, downsample=1, **kwargs
    ):
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample
        self.network = nn.Sequential(
            nn.Conv2d(framestack, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, observations):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, :: self.downsample, :: self.downsample]
        return self.network(observations.float() / 255.0)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(input_shape[0], out_channels, 3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ImpalaEncoder(nn.Module):
    """IMPALA ResNet encoder (Espeholt et al. 2018). Returns [batch, hidden_size]."""

    def __init__(self, env, hidden_size=256, cnn_width=16, **kwargs):
        super().__init__()
        h, w, c = env.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [cnn_width, 2 * cnn_width, 2 * cnn_width]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(shape[0] * shape[1] * shape[2], hidden_size),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, observations):
        return self.network(observations.permute(0, 3, 1, 2).float() / 255.0)
