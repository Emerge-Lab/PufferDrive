"""WOSAC evaluation class for PufferDrive."""

import torch
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import configparser
import os

import pufferlib
from pufferlib.ocean.benchmark import metrics
from pufferlib.ocean.benchmark import estimators




class HumanReplayEvaluator:
    """Evaluates policies against human replays in PufferDrive."""

    def __init__(self, config: Dict):
        self.config = config
        self.sim_steps = 91 - self.config["env"]["init_steps"]

    def rollout(self, args, puffer_env, policy):
        """Roll out policy in env with human replays. Store statistics.

        In human replay mode, only the SDC (self-driving car) is controlled by the policy
        while all other agents replay their human trajectories. This tests how compatible
        the policy is with (static) human partners.

        Args:
            args: Config dict with train settings (device, use_rnn, etc.)
            puffer_env: PufferLib environment wrapper
            policy: Trained policy to evaluate

        Returns:
            dict: Aggregated metrics including:
                - avg_collisions_per_agent: Average collisions per agent
                - avg_offroad_per_agent: Average offroad events per agent
        """
        import numpy as np
        import torch
        import pufferlib

        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]

        obs, info = puffer_env.reset()
        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        for time_idx in range(self.sim_steps):
            # Step policy
            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, value = policy.forward_eval(ob_tensor, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(puffer_env.action_space.shape)

            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, puffer_env.action_space.low, puffer_env.action_space.high)

            obs, rewards, dones, truncs, info_list = puffer_env.step(action_np)

            if len(info_list) > 0:  # Happens at the end of episode
                results = info_list[0]
                return results
