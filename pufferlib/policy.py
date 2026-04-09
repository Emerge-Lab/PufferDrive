import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple
from pufferlib.samplers import Sampler
from os import path


class Policy(nn.Module, ABC):
    def __init__(self, sampler: Sampler):
        super().__init__()
        self.sampler = sampler

    @abstractmethod
    def forward_eval(self, obs: torch.Tensor, state: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        given the observation and state, return the sampled action, its log probability, and value estimate.
        input: obs - (batch_size, obs_dim)
        state: dict of metadata for the policy (e.g. env id)

        returns action   - (batch_size, 1) sampled action
                logprob  - (batch_size, 1) log probability of the sampled action
                value    - (batch_size, 1) value estimate
        """
        actions = None
        logprobs = None
        values = None
        return actions, logprobs, values

    @abstractmethod
    def forward_train(
        self, obs: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor = None, truncations: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        given stored observations and actions, recompute values, log probabilities, and entropy.
        input: obs     - (batch_size, bptt, obs_dim)
               actions - (batch_size, bptt, 1) stored actions from the rollout buffer
               mask    - (batch_size, bptt) boolean tensor indicating valid samples (optional)
               state   - dict of metadata (optional)

        returns newvalue  - (num_valid_samples, 1)
                newlogprob - (num_valid_samples, 1)
                entropy    - (num_valid_samples, 1)
        """
        newvalue = None
        newlogprob = None
        entropy = None
        return newvalue, newlogprob, entropy
