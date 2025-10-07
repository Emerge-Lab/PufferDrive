from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper


class Drive(nn.Module):
    def __init__(
        self,
        env,
        input_size=128,
        hidden_size=128,
        value_head="linear",
        masksembles_enable=False,
        masksembles_masks=4,
        masksembles_scale=1.5,
        masksembles_forward_passes=None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.value_head = str(value_head)
        self.masksembles_enabled = bool(masksembles_enable)
        self.masksembles_masks = int(masksembles_masks)
        self.masksembles_scale = float(masksembles_scale)
        self.masksembles_forward_passes = (
            int(masksembles_forward_passes)
            if masksembles_forward_passes is not None
            else self.masksembles_masks
        )
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(7, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        max_road_objects = 13
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(max_road_objects, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        max_partner_objects = 7
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(max_partner_objects, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)

        self._use_mlp_value = bool(self.masksembles_enabled or self.value_head.lower() == "mlp")
        if self._use_mlp_value:
            value_hidden_dim = 384
            self.value_hidden = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, value_hidden_dim))
            self.value_out = pufferlib.pytorch.layer_init(nn.Linear(value_hidden_dim, 1), std=1)
            if self.masksembles_enabled:
                self.register_buffer(
                    "_mask_matrix",
                    self._build_masks(value_hidden_dim, self.masksembles_masks, self.masksembles_scale),
                )
            else:
                self._mask_matrix = None
        else:
            self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)
            self._mask_matrix = None

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden, state)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        ego_dim = 7
        partner_dim = 63 * 7
        road_dim = 200 * 7
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, 63, 7)
        road_objects = road_obs.view(-1, 200, 7)
        road_continuous = road_objects[:, :, :6]  # First 6 features
        road_categorical = road_objects[:, :, 6]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, 200, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding

    def decode_actions(self, flat_hidden, state=None):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self._value_forward(flat_hidden, state)

        return action, value

    @torch.no_grad()
    def _build_masks(self, channels: int, K: int, scale: float) -> torch.Tensor:
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        phi = torch.rand((channels,), generator=g) * (2 * torch.pi)
        k_idx = torch.arange(K)
        anchors = 2 * torch.pi * k_idx / K
        diff = (phi.unsqueeze(0) - anchors.unsqueeze(1)).abs()
        dist = torch.minimum(diff, 2 * torch.pi - diff)
        m = max(1, int(round(K / max(scale, 1e-6))))
        topm = torch.topk(-dist, k=m, dim=0).indices
        masks = torch.zeros((K, channels), dtype=torch.bool)
        ar = torch.arange(channels)
        masks[topm, ar.expand_as(topm)] = True
        return masks

    def _apply_mask(self, x: torch.Tensor, mask_id: int | None) -> torch.Tensor:
        if not self.masksembles_enabled or self._mask_matrix is None or mask_id is None:
            return x
        mask = self._mask_matrix[mask_id].to(x.device)
        return x * mask

    def _value_forward(self, flat_hidden: torch.Tensor, state=None) -> torch.Tensor:
        if not self._use_mlp_value:
            return self.value_fn(flat_hidden)
        h = self.value_hidden(flat_hidden)
        mask_id = None
        if state is not None and isinstance(state, dict):
            mask_id = state.get("mask_id", None)
            if isinstance(mask_id, torch.Tensor):
                mask_id = int(mask_id.item())
        if mask_id is not None:
            h = self._apply_mask(h, mask_id)
        h = torch.relu(h)
        return self.value_out(h)

    def value_with_mask(self, flat_hidden: torch.Tensor, mask_id: int) -> torch.Tensor:
        if not self._use_mlp_value:
            return self.value_fn(flat_hidden)
        h = self.value_hidden(flat_hidden)
        h = self._apply_mask(h, int(mask_id))
        h = torch.relu(h)
        return self.value_out(h)

    @torch.no_grad()
    def value_uncertainty_from_hidden(self, flat_hidden: torch.Tensor, passes: int | None = None):
        if not self._use_mlp_value:
            v = self._value_forward(flat_hidden)
            return v, torch.zeros_like(v)
        if not self.masksembles_enabled or self._mask_matrix is None:
            v = self._value_forward(flat_hidden)
            return v, torch.zeros_like(v)

        K = self.masksembles_masks
        P = min(int(passes or self.masksembles_forward_passes or K), K)
        values = []
        for k in range(P):
            v = self.value_with_mask(flat_hidden, k)
            values.append(v)
        vs = torch.stack(values, dim=0)
        return vs.mean(dim=0), vs.std(dim=0)
