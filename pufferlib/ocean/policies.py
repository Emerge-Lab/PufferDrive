from torch import nn
import torch
import torch.nn.functional as F

import pufferlib

from pufferlib.policy import Policy
from pufferlib.samplers import DiscreteSampler, MultiDiscreteSampler, Sampler


class MultiDiscreteDriveMLP(Policy):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__(sampler=MultiDiscreteSampler())
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories
        # Determine ego dimension from environment's feature layout
        self.ego_dim = env.ego_features

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

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )

        self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward_eval(self, observations, state=None):
        assert observations.dim() == 2, "Expected input shape [batch_size, obs_dim]"
        hidden = self.encode_observations(observations)
        logits, value = self.decode_actions(hidden)
        actions, logprobs = self.sampler.sample_actions(logits)
        return actions, logprobs, value

    def forward_train(self, observations, actions, mask=None, state=None):
        assert observations.dim() == 3, "Expected input shape [batch_size, bptt, obs_dim]"
        flat_obs = observations.view(-1, observations.size(-1))
        flat_actions = actions.view(-1, actions.size(-1))
        if mask is not None:
            assert mask.dim() == 2, "Expected mask shape [batch_size, bptt]"
            flat_mask = mask.view(-1)
            flat_obs = flat_obs[flat_mask]
            flat_actions = flat_actions[flat_mask]
        logits, newvalue = self.decode_actions(self.encode_observations(flat_obs))
        newlogprob, entropy = self.sampler.compute_logprobs(logits, flat_actions)
        return newvalue, newlogprob, entropy

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, ROAD_MAX_OBJECTS, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        return embedding

    def decode_actions(self, flat_hidden):
        logits = self.actor(flat_hidden)
        logits = torch.split(logits, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)

        return logits, value


class MultiDiscreteDriveLSTM(Policy):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__(sampler=MultiDiscreteSampler())
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories
        # Determine ego dimension from environment's feature layout
        self.ego_dim = env.ego_features

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

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )

        self.atn_dim = env.single_action_space.nvec.tolist()

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

        # Per-agent hidden state for rollout
        # Not registered as parameters so they are excluded from the weight binary.
        self._eval_buffer_h = None  # [1, num_agents, hidden_size]
        self._eval_buffer_c = None  # [1, num_agents, hidden_size]

        # Not registered as parameters so they are excluded from the weight binary.
        self._train_buffer_h = None  # [1, num_agents, hidden_size]
        self._train_buffer_c = None  # [1, num_agents, hidden_size]

    def _init_buffers(self, batch_size: int, device: torch.device):
        if self._eval_buffer_h is None:
            self._eval_buffer_h = torch.zeros(1, batch_size, self.hidden_size, device=device)
            self._eval_buffer_c = torch.zeros(1, batch_size, self.hidden_size, device=device)
            self._train_buffer_h = torch.zeros(1, batch_size, self.hidden_size, device=device)
            self._train_buffer_c = torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward_eval(self, observations, state=None, truncations=None):
        assert observations.dim() == 2, "Expected input shape [batch_size, obs_dim]"
        assert "env_id" in state, "Expected state to contain 'env_id' for indexing recurrent buffer"
        batch_size, device = observations.shape[0], observations.device
        self._init_buffers(batch_size, device)

        embedding = self.encode_observations(observations)  # [batch_size, hidden_size]
        lstm_out, (h_new, c_new) = self.lstm(
            embedding.unsqueeze(0),  # [1, batch_size, hidden_size]
            (self.buffer_h.detach(), self.buffer_c.detach()),
        )
        self.buffer_h, self.buffer_c = h_new.detach(), c_new.detach()

        logits, value = self.decode_actions(lstm_out.squeeze(0))
        actions, logprobs = self.sampler.sample_actions(logits)
        return actions, logprobs, value

    def forward_train(self, observations, actions, mask=None, truncations=None):
        """
        inputs:
            observations - (batch_size, bptt, obs_dim)
            actions      - (batch_size, bptt, 1)
            mask         - (batch_size, bptt) boolean tensor indicating valid samples (optional)
            truncations  - (batch_size, bptt) boolean tensor indicating truncation points (optional)

        when mask = false it indicates the sample is invalid, we drop the sample and interrupt the LSTM recurring states
        when truncation = true, we interrupt the LSTM recurring states but keep the sample

        """
        assert observations.dim() == 3, "Expected input shape [batch_size, bptt, obs_dim]"
        batch_size, bptt, _ = observations.shape
        device = observations.device

        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)

        hiddens = []
        for t in range(bptt):
            if t > 0:
                reset = torch.zeros(batch_size, dtype=torch.bool, device=device)
                if mask is not None:
                    reset = reset | ~mask[:, t - 1].bool()
                if truncations is not None:
                    reset = reset | truncations[:, t - 1].bool()
                h[:, reset, :] = 0.0
                c[:, reset, :] = 0.0

            embedding = self.encode_observations(observations[:, t, :])
            out, (h, c) = self.lstm(embedding.unsqueeze(0), (h, c))
            hiddens.append(out.squeeze(0))

        hidden = torch.stack(hiddens, dim=1).reshape(batch_size * bptt, self.hidden_size)
        flat_actions = actions.view(-1, actions.size(-1))

        if mask is not None:
            flat_mask = mask.reshape(-1)
            hidden = hidden[flat_mask]
            flat_actions = flat_actions[flat_mask]

        logits, newvalue = self.decode_actions(hidden)
        newlogprob, entropy = self.sampler.compute_logprobs(logits, flat_actions)
        return newvalue, newlogprob, entropy

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, ROAD_MAX_OBJECTS, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        return embedding

    def decode_actions(self, flat_hidden):
        logits = self.actor(flat_hidden)
        logits = torch.split(logits, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)

        return logits, value
