from torch.backends import cudnn
from torch import nn
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

import pufferlib
import pufferlib.models

from pufferlib import _C
from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401

Recurrent = pufferlib.models.LSTMWrapper

MAX_PARTNER_OBJECTS = 31
MAX_ROAD_OBJECTS = 128

ROAD_FEATURES = 7
ROAD_FEATURES_AFTER_ONEHOT = 13
PARTNER_FEATURES = 7


class LinearMaxKernels(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        out = torch.ops.pufferlib.linear_max_fused(x, weight, bias)
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = torch.ops.pufferlib.linear_max_fused_backward(grad_out.contiguous(), x, weight)
        return grad_x, grad_weight, grad_bias


class LinearMax(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, input_size).normal_(mean=0.0, std=input_size**-0.5))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return LinearMaxKernels.apply(x, self.weight, self.bias)


class LinearReluMaxKernel(torch.autograd.Function):
    """Custom kernel: Linear -> ReLU -> Max"""

    @staticmethod
    def forward(ctx, x, weight, bias):
        out = torch.ops.pufferlib.linear_relu_max(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias = ctx.saved_tensors
        grad_x, grad_weight, grad_bias = torch.ops.pufferlib.linear_relu_max_backward(
            grad_out.contiguous(), x, weight, bias
        )
        return grad_x, grad_weight, grad_bias


class FusedLinearReluMax(nn.Module):
    """Fused CUDA kernel: Linear -> ReLU -> Max"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size).normal_(mean=0.0, std=input_size**-0.5))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return LinearReluMaxKernel.apply(x, self.weight, self.bias)


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=256, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories

        # Determine ego dimension from environment's dynamics model
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        # self.road_encoder = nn.Sequential(
        #     FusedLinearReluMax(ROAD_FEATURES_AFTER_ONEHOT, input_size),  # Cuda kernel
        # )
        # self.partner_encoder = nn.Sequential(
        #     FusedLinearReluMax(PARTNER_FEATURES, input_size),  # Cuda kernel
        #     nn.LayerNorm(input_size),
        # )

        self.full_scene_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.observation_size, input_size)),
            nn.LayerNorm(input_size),
        )

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(4 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

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
        # partner_features = self.partner_encoder(partner_objects.contiguous())
        # road_features = self.road_encoder(road_objects.contiguous())

        full_scene_context = self.full_scene_encoder(observations)

        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        # Pass through shared embedding
        concat_features = torch.cat([ego_features, road_features, partner_features, full_scene_context], dim=1)
        embedding = F.relu(self.shared_embedding(concat_features))

        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value
