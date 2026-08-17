import math

import torch
from torch import nn

VALUE_DISTRIBUTION_OFF = "off"
VALUE_DISTRIBUTION_TWO_HOT = "two_hot"
VALUE_DISTRIBUTION_HL_GAUSS = "hl_gauss"
VALUE_DISTRIBUTION_MODES = (VALUE_DISTRIBUTION_OFF, VALUE_DISTRIBUTION_TWO_HOT, VALUE_DISTRIBUTION_HL_GAUSS)
MIN_VALUE_BINS = 3
SQRT_TWO = math.sqrt(2.0)


class BinnedValueDistribution(nn.Module):
    def __init__(self, mode, num_bins, support_min, support_max, hl_gauss_sigma_ratio):
        super().__init__()
        self.mode = mode
        self.num_bins = num_bins
        self.bin_width = (support_max - support_min) / num_bins
        self.hl_gauss_sigma = hl_gauss_sigma_ratio * self.bin_width
        bin_edges = torch.linspace(support_min, support_max, num_bins + 1, dtype=torch.float32)
        self.register_buffer("bin_edges", bin_edges, persistent=False)
        self.register_buffer("bin_centers", 0.5 * (bin_edges[:-1] + bin_edges[1:]), persistent=False)
        if mode == VALUE_DISTRIBUTION_TWO_HOT:
            self.target_min = self.bin_centers[0].item()
            self.target_max = self.bin_centers[-1].item()
        else:
            self.target_min = float(support_min)
            self.target_max = float(support_max)

    def expectation(self, value_logits):
        probs = torch.softmax(value_logits.float(), dim=-1)
        # mul+sum stays float32 under autocast; matmul would be cast back to bfloat16
        return (probs * self.bin_centers).sum(dim=-1)

    def clipped_fraction(self, scalar_returns):
        outside = (scalar_returns < self.target_min) | (scalar_returns > self.target_max)
        return outside.float().mean()

    def target_probs(self, scalar_returns):
        # out-of-support targets saturate; HL-Gauss otherwise divides by an underflowed zero
        targets = scalar_returns.reshape(-1).float().clamp(self.target_min, self.target_max)
        if self.mode == VALUE_DISTRIBUTION_TWO_HOT:
            position = (targets - self.bin_centers[0]) / self.bin_width
            lower_idx = position.floor().clamp(0, self.num_bins - 2).long().unsqueeze(1)
            upper_weight = (position.unsqueeze(1) - lower_idx).clamp(0.0, 1.0)
            probs = torch.zeros(targets.shape[0], self.num_bins, dtype=targets.dtype, device=targets.device)
            probs.scatter_(1, lower_idx, 1.0 - upper_weight)
            probs.scatter_(1, lower_idx + 1, upper_weight)
            return probs
        cdf = 0.5 * (1.0 + torch.erf((self.bin_edges - targets.unsqueeze(1)) / (self.hl_gauss_sigma * SQRT_TWO)))
        return (cdf[:, 1:] - cdf[:, :-1]) / (cdf[:, -1:] - cdf[:, :1])

    def cross_entropy(self, value_logits, scalar_returns):
        log_probs = torch.log_softmax(value_logits.float(), dim=-1)
        return -(self.target_probs(scalar_returns) * log_probs).sum(dim=-1).mean()


def find_value_distribution(policy):
    """Recurrent and distributed wrappers hide the head, so search the whole tree."""
    for module in policy.modules():
        if isinstance(module, BinnedValueDistribution):
            return module
    return None


def make_binned_value_distribution(mode, num_bins, support_min, support_max, hl_gauss_sigma_ratio):
    if mode not in VALUE_DISTRIBUTION_MODES:
        raise ValueError(f"Invalid critic_distribution_mode {mode!r}: use one of {VALUE_DISTRIBUTION_MODES}")
    if mode == VALUE_DISTRIBUTION_OFF:
        return None
    if num_bins < MIN_VALUE_BINS:
        raise ValueError(f"critic_num_bins {num_bins} must be >= {MIN_VALUE_BINS}")
    if support_max <= support_min:
        raise ValueError(f"critic_support_max {support_max} must be > critic_support_min {support_min}")
    if hl_gauss_sigma_ratio <= 0.0:
        raise ValueError(f"critic_hl_gauss_sigma_ratio {hl_gauss_sigma_ratio} must be > 0")
    return BinnedValueDistribution(mode, num_bins, support_min, support_max, hl_gauss_sigma_ratio)
