import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces


class Default(nn.Module):
    """Default PyTorch policy. Flattens obs and applies a linear layer.

    PufferLib is not a framework. It does not enforce a base class.
    You can use any PyTorch policy that returns actions and values.
    We structure our forward methods as encode_observations and decode_actions
    to make it easier to wrap policies with LSTMs. You can do that and use
    our LSTM wrapper or implement your own. To port an existing policy
    for use with our LSTM wrapper, simply put everything from forward() before
    the recurrent cell into encode_observations and put everything after
    into decode_actions.
    """

    def __init__(self, env, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_multidiscrete = isinstance(env.single_action_space, pufferlib.spaces.MultiDiscrete)
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        try:
            self.is_dict_obs = isinstance(env.env.observation_space, pufferlib.spaces.Dict)
        except:
            self.is_dict_obs = isinstance(env.observation_space, pufferlib.spaces.Dict)

        if self.is_dict_obs:
            self.dtype = pufferlib.pytorch.nativize_dtype(env.emulated)
            input_size = int(sum(np.prod(v.shape) for v in env.env.observation_space.values()))
            self.encoder = nn.Linear(input_size, self.hidden_size)
        else:
            num_obs = np.prod(env.single_observation_space.shape)
            self.encoder = torch.nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
                nn.GELU(),
            )

        if self.is_multidiscrete:
            self.action_nvec = tuple(env.single_action_space.nvec)
            num_atns = sum(self.action_nvec)
            self.decoder = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, num_atns), std=0.01)
        elif not self.is_continuous:
            num_atns = env.single_action_space.n
            self.decoder = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, num_atns), std=0.01)
        else:
            self.decoder_mean = pufferlib.pytorch.layer_init(
                nn.Linear(hidden_size, env.single_action_space.shape[0]), std=0.01
            )
            self.decoder_logstd = nn.Parameter(torch.zeros(1, env.single_action_space.shape[0]))

        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    def encode_observations(self, observations, state=None):
        """Encodes a batch of observations into hidden states. Assumes
        no time dimension (handled by LSTM wrappers)."""
        batch_size = observations.shape[0]
        if self.is_dict_obs:
            observations = pufferlib.pytorch.nativize_tensor(observations, self.dtype)
            observations = torch.cat([v.view(batch_size, -1) for v in observations.values()], dim=1)
        else:
            observations = observations.view(batch_size, -1)
        return self.encoder(observations.float())

    def decode_actions(self, hidden):
        """Decodes a batch of hidden states into (multi)discrete actions.
        Assumes no time dimension (handled by LSTM wrappers)."""
        if self.is_multidiscrete:
            logits = self.decoder(hidden).split(self.action_nvec, dim=1)
        elif self.is_continuous:
            mean = self.decoder_mean(hidden)
            logstd = self.decoder_logstd.expand_as(mean)
            std = torch.exp(logstd)
            logits = torch.distributions.Normal(mean, std)
        else:
            logits = self.decoder(hidden)

        values = self.value(hidden)
        return logits, values


class LSTMWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128):
        """Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example."""
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous

        for name, param in self.named_parameters():
            if "layer_norm" in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        # self.pre_layernorm = nn.LayerNorm(hidden_size)
        # self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward_eval(self, observations, state):
        """Forward function for inference. 3x faster than using LSTM directly"""
        hidden = self.policy.encode_observations(observations, state=state)
        h = state["lstm_h"]
        c = state["lstm_c"]

        # TODO: Don't break compile
        if h is not None:
            assert h.shape[0] == c.shape[0] == observations.shape[0], (
                f"LSTM state must be (h, c), h shape {h.shape[0]}, c shape {c.shape[0]}, obs shape {observations.shape[0]}"
            )
            lstm_state = (h, c)
        else:
            lstm_state = None

        # hidden = self.pre_layernorm(hidden)
        hidden, c = self.cell(hidden, lstm_state)
        # hidden = self.post_layernorm(hidden)
        state["hidden"] = hidden
        state["lstm_h"] = hidden
        state["lstm_c"] = c
        logits, values = self.policy.decode_actions(hidden)
        return logits, values

    def forward(self, observations, state):
        """Forward function for training. Uses LSTM for fast time-batching"""
        x = observations
        lstm_h = state["lstm_h"]
        lstm_c = state["lstm_c"]

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        if lstm_h is not None:
            assert lstm_h.shape[1] == lstm_c.shape[1] == B, "LSTM state must be (h, c)"
            lstm_state = (lstm_h, lstm_c)
        else:
            lstm_state = None

        x = x.reshape(B * TT, *space_shape)
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B * TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)

        hidden = hidden.transpose(0, 1)
        # hidden = self.pre_layernorm(hidden)
        hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
        hidden = hidden.float()

        # hidden = self.post_layernorm(hidden)
        hidden = hidden.transpose(0, 1)

        flat_hidden = hidden.reshape(B * TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        # state.batch_logits = logits.reshape(B, TT, -1)
        state["hidden"] = hidden
        state["lstm_h"] = lstm_h.detach()
        state["lstm_c"] = lstm_c.detach()
        return logits, values


class Convolutional(nn.Module):
    def __init__(
        self,
        env,
        *args,
        framestack,
        flat_size,
        input_size=512,
        hidden_size=512,
        output_size=512,
        channels_last=False,
        downsample=1,
        **kwargs,
    ):
        """The CleanRL default NatureCNN policy used for Atari.
        It's just a stack of three convolutions followed by a linear layer

        Takes framestack as a mandatory keyword argument. Suggested default is 1 frame
        with LSTM or 4 frames without."""
        super().__init__()
        self.channels_last = channels_last
        self.downsample = downsample

        # TODO: Remove these from required params
        self.hidden_size = hidden_size
        self.is_continuous = False

        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, env.single_action_space.n), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(output_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, observations, state=None):
        if self.channels_last:
            observations = observations.permute(0, 3, 1, 2)
        if self.downsample > 1:
            observations = observations[:, :, :: self.downsample, :: self.downsample]
        return self.network(observations.float() / 255.0)

    def decode_actions(self, flat_hidden):
        action = self.actor(flat_hidden)
        value = self.value_fn(flat_hidden)
        return action, value


class ProcgenResnet(nn.Module):
    """Procgen baseline from the AICrowd NeurIPS 2020 competition
    Based on the ResNet architecture that was used in the Impala paper."""

    def __init__(self, env, cnn_width=16, mlp_width=256):
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
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=mlp_width),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(mlp_width, env.single_action_space.n), std=0.01)
        self.value = pufferlib.pytorch.layer_init(nn.Linear(mlp_width, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, observations, state=None):
        return self.forward(observations, state)

    def encode_observations(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return hidden

    def decode_actions(self, hidden):
        """linear decoder function"""
        action = self.actor(hidden)
        value = self.value(hidden)
        return action, value


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1
        )
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class TransformerWrapper(nn.Module):
    def __init__(self, env, policy, input_size=128, hidden_size=128, 
                 num_layers=2, num_heads=4, dropout=0.1, max_seq_len=128):
        """Wraps your policy with a Transformer without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example."""
        super().__init__()
        self.obs_shape = env.single_observation_space.shape

        self.policy = policy
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.is_continuous = self.policy.is_continuous

        for name, param in self.named_parameters():
            if "layer_norm" in name or "norm" in name:
                continue
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name and param.ndim >= 2:
                nn.init.orthogonal_(param, 1.0)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # For tracking sequence position during inference
        self.register_buffer('pos_encodings', self._generate_positional_encodings(max_seq_len, hidden_size))

    def _generate_positional_encodings(self, max_len, d_model):
        """Generate sinusoidal positional encodings"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward_eval(self, observations, state):
        """Forward function for inference. Maintains a rolling context window."""
        hidden = self.policy.encode_observations(observations, state=state)
        
        # Get past context from state
        past_hidden = state.get("transformer_hidden", None)
        seq_pos = state.get("seq_pos", 0)
        
        # Add current observation to context
        if past_hidden is not None:
            # Append new hidden state to past context
            context = torch.cat([past_hidden, hidden.unsqueeze(1)], dim=1)
            # Keep only last max_seq_len items
            if context.shape[1] > self.max_seq_len:
                context = context[:, -self.max_seq_len:, :]
                seq_pos = self.max_seq_len - 1
            else:
                seq_pos = min(seq_pos + 1, self.max_seq_len - 1)
        else:
            context = hidden.unsqueeze(1)
            seq_pos = 0
        
        # Add positional encoding
        batch_size, seq_len, _ = context.shape
        pos_enc = self.pos_encodings[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        context = context + pos_enc
        
        # Pass through transformer
        transformed = self.transformer(context)
        
        # Take the last position as output
        output = transformed[:, -1, :]
        
        # Update state
        state["hidden"] = output
        state["transformer_hidden"] = context.detach()
        state["seq_pos"] = seq_pos
        
        logits, values = self.policy.decode_actions(output)
        return logits, values

    def forward(self, observations, state):
        """Forward function for training. Processes full sequences efficiently."""
        x = observations
        past_hidden = state.get("transformer_hidden", None)

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        x = x.reshape(B * TT, *space_shape)
        hidden = self.policy.encode_observations(x, state)
        assert hidden.shape == (B * TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)
        
        # Add past context if available
        if past_hidden is not None:
            hidden = torch.cat([past_hidden, hidden], dim=1)
            # Trim to max sequence length
            if hidden.shape[1] > self.max_seq_len:
                hidden = hidden[:, -self.max_seq_len:, :]
        
        seq_len = hidden.shape[1]
        
        # Add positional encodings
        pos_enc = self.pos_encodings[:seq_len, :].unsqueeze(0).expand(B, -1, -1)
        hidden = hidden + pos_enc
        
        # Pass through transformer
        transformed = self.transformer(hidden)
        transformed = transformed.float()
        
        # Take only the new timesteps (not the past context)
        if past_hidden is not None:
            past_len = past_hidden.shape[1]
            transformed = transformed[:, past_len:, :]
        
        flat_hidden = transformed.reshape(B * TT, self.hidden_size)
        logits, values = self.policy.decode_actions(flat_hidden)
        values = values.reshape(B, TT)
        
        # Update state - store the full context for next iteration
        state["hidden"] = transformed
        state["transformer_hidden"] = hidden.detach()
        
        return logits, values