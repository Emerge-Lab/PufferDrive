"""Simplified estimators to compute log-likelihood of simulated trajs based on https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/estimators.py"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

np.set_printoptions(suppress=True)


def histogram_estimate(
    log_samples: np.ndarray,
    sim_samples: np.ndarray,
    min_val: float,
    max_val: float,
    num_bins: int,
    additive_smoothing: float = 0.1,
) -> np.ndarray:
    """Computes log-likelihoods of samples based on histograms.

    Args:
        log_samples: Shape (n_agents, sample_size) - samples to evaluate
        sim_samples: Shape (n_agents, sample_size) - samples to build distribution from
        min_val: Minimum value for histogram bins
        max_val: Maximum value for histogram bins
        num_bins: Number of histogram bins
        additive_smoothing: Pseudocount for Laplace smoothing (default: 0.1)
        sanity_check: If True, plot visualization for debugging

    Returns:
        Shape (n_agents, sample_size) - log-likelihood of each log sample
        under the corresponding sim distribution
    """

    n_agents, sample_size = sim_samples.shape

    # Clip samples to valid range
    log_samples_clipped = np.clip(log_samples, min_val, max_val)
    sim_samples_clipped = np.clip(sim_samples, min_val, max_val)

    # Create bin edges
    edges = np.linspace(min_val, max_val, num_bins + 1)

    # Create histogram for each agent from sim samples
    sim_counts = np.array([np.histogram(sim_samples_clipped[i], bins=edges)[0] for i in range(n_agents)])

    # Apply smoothing and normalize to probabilities
    sim_counts = sim_counts.astype(float) + additive_smoothing
    sim_probs = sim_counts / sim_counts.sum(axis=1, keepdims=True)

    # Find which bin each log sample belongs to
    # digitize returns values in [1, num_bins], so subtract 1 for 0-indexing
    # right=False means bins are [left, right) except last bin which is [left, right]
    log_bins = np.digitize(log_samples_clipped, edges, right=False) - 1

    # Clip to valid bin indices (handles edge case where value == max_val)
    log_bins = np.clip(log_bins, 0, num_bins - 1)

    # Get log probabilities for each sample
    agent_indices = np.arange(n_agents)[:, None]
    log_probs = np.log(sim_probs[agent_indices, log_bins])

    return log_probs


def log_likelihood_estimate_timeseries(
    log_values: np.ndarray,
    sim_values: np.ndarray,
    min_val: float,
    max_val: float,
    num_bins: int,
    additive_smoothing: float = 0.1,
    treat_timesteps_independently: bool = True,
    sanity_check: bool = False,
    plot_agent_idx: int = 0,
) -> np.ndarray:
    """Computes log-likelihood estimates for time-series simulated features on a per-agent basis.

    Args:
        log_values: Shape (n_agents, 1, n_steps)
        sim_values: Shape (n_agents, n_rollouts, n_steps)
        min_val: Minimum value for histogram bins
        max_val: Maximum value for histogram bins
        num_bins: Number of histogram bins
        additive_smoothing: Pseudocount for Laplace smoothing
        treat_timesteps_independently: If True, treat each timestep independently
        sanity_check: If True, plot visualizations for debugging
        plot_agent_idx: Which agent to plot if sanity_check=True
        plot_rollouts: How many rollouts to show if sanity_check=True

    Returns:
        A tensor of shape (n_objects, n_steps) containing the log probability
        estimates of the log features under the simulated distribution of the same
        feature.
    """
    n_agents, n_rollouts, n_steps = sim_values.shape

    if treat_timesteps_independently:
        # Ignore temporal structure: We end up with (n_agents, n_rollouts * n_steps)
        log_flat = log_values.reshape(n_agents, n_steps)
        sim_flat = sim_values.reshape(n_agents, n_rollouts * n_steps)

    else:
        # If values in time are instead to be compared per-step, reshape:
        # - `sim_values` as (n_objects * n_steps, n_rollouts)
        # - `log_values` as (n_objects * n_steps, 1)
        log_flat = log_values.reshape(n_agents * n_steps, 1)
        sim_flat = sim_values.transpose(0, 2, 1).reshape(n_agents * n_steps, n_rollouts)

    # Compute log-likelihoods
    log_probs = histogram_estimate(log_flat, sim_flat, min_val, max_val, num_bins, additive_smoothing)

    # Depending on `independent_timesteps`, the likelihoods might be flattened, so
    # reshape back to the initial `log_values` shape.
    log_probs = log_probs.reshape(n_agents, n_steps)

    # Sanity check visualization
    if sanity_check:
        _plot_histogram_sanity_check(log_flat, sim_flat, log_probs, plot_agent_idx)

    return log_probs


def _plot_histogram_sanity_check(
    log_samples: np.ndarray,
    sim_samples: np.ndarray,
    log_probs: np.ndarray,
    idx: int,
):
    """Plot data as sanity check."""

    for idx in range(log_samples.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Histogram Log-Likelihood Sanity Check for Agent {idx}")

        # Plot 1: Simulated distribution (histogram)
        axes[0].hist(sim_samples[idx], density=True, alpha=0.7, color="blue")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Simulated distribution")
        axes[0].grid(alpha=0.3)

        # Plot 2: Ground-truth values overlaid on simulated
        axes[1].hist(sim_samples[idx], density=True, alpha=0.5, color="blue", label="Simulated")
        axes[1].scatter(
            log_samples[idx],
            np.zeros_like(log_samples[idx]),
            color="green",
            marker="|",
            s=200,
            linewidths=2,
            label="Ground-truth",
            zorder=5,
        )
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Ground-truth vs Simulated")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Plot 3: Log-likelihood values
        axes[2].hist(log_samples[idx], alpha=0.7, color="orange")
        axes[2].set_ylabel("Log-likelihood")
        axes[2].set_title("Log-likelihood of Ground-truth")
        axes[2].grid(alpha=0.3)
        axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"histogram_sanity_check_agent_{idx}.png")
