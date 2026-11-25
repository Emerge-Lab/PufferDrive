"""CUDA Drive encoder matches the PyTorch encoder, forward and backward."""

import unittest

import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete

from pufferlib.ocean.torch import Drive


def _env():
    ego, partners, roads, partner_f, road_f = 12, 31, 128, 7, 7
    corners, corner_n = 4, 16
    obs = ego + partners * partner_f + corners * corner_n * partner_f + roads * road_f

    class Env:
        single_observation_space = Box(low=-1, high=1, shape=(obs,), dtype=np.float32)
        single_action_space = MultiDiscrete([257, 257])
        max_partner_objects = partners
        partner_features = partner_f
        max_road_objects = roads
        road_features = road_f
        ego_features = ego
        corner_views = corners
        corner_partners = corner_n

    return Env()


def _obs(batch, ego=12, partners=31, roads=128, corners=4, corner_n=16, seed=0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    width = ego + partners * 7 + corners * corner_n * 7 + roads * 7
    obs = torch.randn(batch, width, generator=g)
    road_at = ego + partners * 7 + corners * corner_n * 7
    road = obs[:, road_at:].view(batch, roads, 7)
    road[:, :, 6] = torch.randint(0, 7, (batch, roads), generator=g).float()
    return obs


class DriveEncoderCudaTest(unittest.TestCase):
    def test_forward_and_backward_match_pytorch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        # The kernel is full fp32. Compare against the fp32 module, not TF32 tensor cores.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        policy = Drive(_env(), input_size=64, hidden_size=256).cuda()
        obs = _obs(8).cuda()
        with torch.no_grad():
            cuda_out = policy.encode_observations(obs)
            ref = policy.encode_observations_pytorch(obs)
        self.assertLess((cuda_out - ref).abs().max().item(), 2e-4)

        obs_g = obs.detach().clone()
        policy.encode_observations(obs_g).square().mean().backward()
        cuda_grads = {}
        for name, param in policy.named_parameters():
            if name.startswith("actor") or name.startswith("value_fn"):
                continue
            self.assertIsNotNone(param.grad, name)
            cuda_grads[name] = param.grad.detach().clone()
        policy.zero_grad(set_to_none=True)
        policy.encode_observations_pytorch(obs_g).square().mean().backward()
        for name, got in cuda_grads.items():
            exp = policy.get_parameter(name).grad
            self.assertIsNotNone(exp, name)
            self.assertLess((got - exp).abs().max().item(), 2e-3, name)


if __name__ == "__main__":
    unittest.main()
