"""Exponential moving average of policy weights.

Tracks parameters only, not buffers: the policy's buffers are constant action
tables, and LayerNorm keeps its affine terms as parameters, so there is no
running state that would freeze in the shadow copy.
"""

import copy

import torch


class EMA:
    def __init__(self, model, decay):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"ema decay must be in [0.0, 1.0). Got: {decay}")
        self.original_model = model
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.detach_()

    def update(self):
        with torch.no_grad():
            for ema_param, orig_param in zip(self.ema_model.parameters(), self.original_model.parameters()):
                ema_param.mul_(self.decay).add_((1.0 - self.decay) * orig_param)

    def apply_shadow(self):
        for ema_param, orig_param in zip(self.ema_model.parameters(), self.original_model.parameters()):
            orig_param.data.copy_(ema_param.data)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
