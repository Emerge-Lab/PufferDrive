import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List


class Sampler(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample_actions(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        given the logits it samples actions and returns the sampled actions and their log probabilities
        input: logits - (batch_size, action_dim)

        returns action   - (batch_size, action_dim) sampled action
                logprobs - (batch_size,) log probability per sample
        """
        actions = None
        logprobs = None
        return actions, logprobs

    @abstractmethod
    def compute_logprobs(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        given the logits and actions, compute the log probabilities of the actions under the given logits
        input: logits   - (batch_size, action_dim)
               actions  - (batch_size, action_dim)

        returns logprobs - (batch_size,) log probability per sample
                entropy  - (batch_size,) entropy per sample
        """
        logprobs = None
        return logprobs


class DiscreteSampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample_actions(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_dist = torch.distributions.Categorical(logits=logits)
        actions = action_dist.sample().unsqueeze(-1)  # [batch_size, 1]
        logprobs = action_dist.log_prob(actions.squeeze(-1))  # [batch_size]
        return actions, logprobs

    def compute_logprobs(self, logits: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_dist = torch.distributions.Categorical(logits=logits)
        logprobs = action_dist.log_prob(actions.squeeze(-1))  # [batch_size]
        entropy = action_dist.entropy()  # [batch_size]
        return logprobs, entropy


class MultiDiscreteSampler(Sampler):
    def __init__(self):
        super().__init__()

    def _pad_logits(self, logits: List[torch.Tensor]) -> torch.Tensor:
        # logits: list of [batch, action_dim] → [batch, num_heads, max_action_dim] padded
        return torch.nn.utils.rnn.pad_sequence(
            [l.transpose(0, 1) for l in logits], batch_first=False, padding_value=-torch.inf
        ).permute(1, 2, 0)  # [num_heads, batch, max_action_dim]

    def sample_actions(self, logits: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = self._pad_logits(logits)  # [num_heads, batch, max_action_dim]
        action_dist = torch.distributions.Categorical(logits=padded)
        actions = action_dist.sample()  # [num_heads, batch]
        logprobs = action_dist.log_prob(actions).sum(0)  # [batch]
        return actions.T, logprobs  # [batch, num_heads], [batch]

    def compute_logprobs(self, logits: List[torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = self._pad_logits(logits)  # [num_heads, batch, max_action_dim]
        action_dist = torch.distributions.Categorical(logits=padded)
        logprobs = action_dist.log_prob(actions.T).sum(0)  # [batch]
        entropy = action_dist.entropy().sum(0)  # [batch]
        return logprobs, entropy
