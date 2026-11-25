"""CUDA Drive encoder. Falls back to the PyTorch module when the op is unavailable."""

import os

import torch
from torch.autograd import Function

import pufferlib._C  # registers torch.ops.pufferlib


def _corner_stack(module, slot, name):
    return torch.stack([getattr(encoder[slot], name) for encoder in module.corner_encoders])


def _weights(module):
    ego = module.ego_encoder
    partner = module.partner_encoder
    road = module.road_encoder
    shared = module.shared_embedding[1]
    return (
        ego[0].weight, ego[0].bias, ego[1].weight, ego[1].bias, ego[2].weight, ego[2].bias,
        partner[0].weight, partner[0].bias, partner[1].weight, partner[1].bias, partner[2].weight, partner[2].bias,
        road[0].weight, road[0].bias, road[1].weight, road[1].bias, road[2].weight, road[2].bias,
        _corner_stack(module, 0, "weight"), _corner_stack(module, 0, "bias"),
        _corner_stack(module, 1, "weight"), _corner_stack(module, 1, "bias"),
        _corner_stack(module, 2, "weight"), _corner_stack(module, 2, "bias"),
        shared.weight, shared.bias,
    )


def supports_cuda_encoder(module, observations):
    if os.environ.get("DRIVE_ENCODER") == "pytorch":
        return False
    if not observations.is_cuda or observations.dtype != torch.float32 or observations.dim() != 2:
        return False
    if module.ego_encoder[0].out_features != 64 or module.hidden_size != 256 or module.ego_dim != 12:
        return False
    if module.max_partner_objects != 31 or module.partner_features != 7:
        return False
    if module.max_road_objects != 128 or module.road_features != 7:
        return False
    if getattr(module, "corner_views", 0) != 4 or getattr(module, "corner_partners", 0) != 16:
        return False
    if len(module.corner_encoders) != 4:
        return False
    # ego, road, center partners, and four corner pools.
    if module.shared_embedding[1].in_features != 7 * module.ego_encoder[0].out_features:
        return False
    expected = module.ego_dim + 31 * 7 + 4 * 16 * 7 + 128 * 7
    return observations.shape[1] == expected


class _DriveEncoder(Function):
    @staticmethod
    def forward(ctx, obs, *weights):
        # Function.forward runs under no_grad, so is_grad_enabled() is always false here.
        # needs_input_grad is false for rollout (evaluate uses no_grad) and true in the PPO update.
        needs = any(ctx.needs_input_grad)
        out = torch.ops.pufferlib.drive_encoder_forward(obs, *weights, needs)[0]
        if needs:
            ctx.save_for_backward(obs.contiguous(), *[w.contiguous() for w in weights])
        ctx.needs = needs
        return out

    @staticmethod
    def backward(ctx, grad):
        if not ctx.needs:
            return (None,) * 27
        obs, *weights = ctx.saved_tensors
        grads = torch.ops.pufferlib.drive_encoder_backward(grad.contiguous(), obs, *weights)
        return (None, *grads)


def encode(module, observations):
    flat = observations if observations.dim() == 2 else observations.reshape(observations.shape[0], -1)
    return _DriveEncoder.apply(flat, *_weights(module))
