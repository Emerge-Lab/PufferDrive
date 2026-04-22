from torch import nn
import torch

import pufferlib
import pufferlib.models
import pufferlib.pytorch

from pufferlib.models import Default as Policy  # noqa: F401

Recurrent = pufferlib.models.LSTMWrapper

# Signal obs layout constants (must match traffic.h)
_EGO_DIM = 5
_LANE_FEAT = 4
_N_PARTNERS = 4
_PARTNER_FEAT = 6


class TrafficBackbone(nn.Module):
    """Per-entity MLP encoders → max-pool → trunk MLP.

    Supports both obs layouts:
      per_lane=True  (61-dim): ego(5) | lane_stats(8×4=32) | partner_signals(4×6=24)
      per_lane=False (33-dim): ego(5) | lane_agg(1×4=4)    | partner_signals(4×6=24)

    n_lanes is derived from obs_dim: (obs_dim - 5 - 24) // 4
    """

    def __init__(self, obs_dim: int, input_size: int, backbone_hidden_size: int, backbone_num_layers: int):
        super().__init__()
        self.n_lanes = (obs_dim - _EGO_DIM - _N_PARTNERS * _PARTNER_FEAT) // _LANE_FEAT
        print(
            f"[TrafficBackbone.__init__] obs_dim={obs_dim}  n_lanes={self.n_lanes}  "
            f"input_size={input_size}  hidden={backbone_hidden_size}×{backbone_num_layers}"
        )

        def _enc(in_dim):
            return nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(in_dim, input_size)),
                nn.LayerNorm(input_size),
                nn.Tanh(),
                pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
            )

        self.ego_encoder = _enc(_EGO_DIM)
        self.lane_encoder = _enc(_LANE_FEAT)
        self.partner_encoder = _enc(_PARTNER_FEAT)

        trunk_in = 3 * input_size
        layers = []
        for _ in range(backbone_num_layers):
            layers += [nn.GELU(), pufferlib.pytorch.layer_init(nn.Linear(trunk_in, backbone_hidden_size))]
            trunk_in = backbone_hidden_size
        layers.append(nn.GELU())
        self.trunk = nn.Sequential(*layers)
        self.out_dim = backbone_hidden_size if backbone_num_layers > 0 else 3 * input_size
        print(f"[TrafficBackbone.__init__] trunk out_dim={self.out_dim}")

    def forward(self, obs):
        B = obs.shape[0]
        ego = obs[:, :_EGO_DIM]
        lanes = obs[:, _EGO_DIM : _EGO_DIM + self.n_lanes * _LANE_FEAT].reshape(B, self.n_lanes, _LANE_FEAT)
        partners = obs[:, _EGO_DIM + self.n_lanes * _LANE_FEAT :].reshape(B, _N_PARTNERS, _PARTNER_FEAT)

        ego_enc = self.ego_encoder(ego)
        lane_enc, _ = self.lane_encoder(lanes).max(dim=1)
        partner_enc, _ = self.partner_encoder(partners).max(dim=1)

        concat = torch.cat([ego_enc, lane_enc, partner_enc], dim=-1)
        return self.trunk(concat)


class Traffic(nn.Module):
    """Actor-critic policy for PufferTraffic signal control.

    Compatible with pufferlib.models.LSTMWrapper (set rnn_name = Recurrent in ini).
    Action space: MultiDiscrete([3]) → one logit head of size 3.
    """

    def __init__(
        self,
        env,
        input_size: int = 64,
        backbone_hidden_size: int = 256,
        backbone_num_layers: int = 2,
        actor_hidden_size: int = 256,
        actor_num_layers: int = 0,
        critic_hidden_size: int = 256,
        critic_num_layers: int = 0,
    ):
        super().__init__()

        obs_dim = env.single_observation_space.shape[0]
        print(
            f"[Traffic(policy).__init__] obs_dim={obs_dim}  atn_nvec={env.single_action_space.nvec.tolist()}  "
            f"actor_layers={actor_num_layers}  critic_layers={critic_num_layers}"
        )
        self.is_continuous = False
        self.backbone = TrafficBackbone(obs_dim, input_size, backbone_hidden_size, backbone_num_layers)
        self.atn_dim = env.single_action_space.nvec.tolist()  # [3]

        bb_dim = self.backbone.out_dim

        actor_layers, a_in = [], bb_dim
        for _ in range(actor_num_layers):
            actor_layers += [pufferlib.pytorch.layer_init(nn.Linear(a_in, actor_hidden_size)), nn.ReLU()]
            a_in = actor_hidden_size
        actor_layers.append(pufferlib.pytorch.layer_init(nn.Linear(a_in, sum(self.atn_dim)), std=0.01))
        self.actor_head = nn.Sequential(*actor_layers)

        critic_layers, c_in = [], bb_dim
        for _ in range(critic_num_layers):
            critic_layers += [pufferlib.pytorch.layer_init(nn.Linear(c_in, critic_hidden_size)), nn.ReLU()]
            c_in = critic_hidden_size
        critic_layers.append(pufferlib.pytorch.layer_init(nn.Linear(c_in, 1), std=1.0))
        self.critic_head = nn.Sequential(*critic_layers)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[Traffic(policy).__init__] Built. total_params={total_params:,}")

    def forward(self, observations, state=None):
        hidden = self.backbone(observations)
        actions = torch.split(self.actor_head(hidden), self.atn_dim, dim=1)
        value = self.critic_head(hidden)
        return actions, value

    def encode_observations(self, observations, state=None):
        """Required by LSTMWrapper — returns the backbone embedding."""
        return self.backbone(observations)

    def decode_actions(self, hidden, state=None):
        """Required by LSTMWrapper — returns (actions, value) from LSTM output."""
        actions = torch.split(self.actor_head(hidden), self.atn_dim, dim=1)
        value = self.critic_head(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def get_value(self, x, state=None):
        hidden = self.backbone(x)
        return self.critic_head(hidden)
