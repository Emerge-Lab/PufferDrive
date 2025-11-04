"""Simplified estimators to compute log-likelihood of simulated trajs based on https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/estimators.py"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


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
        plot_idx: Which batch index to plot if sanity_check=True

    Returns:
        Shape (n_agents, sample_size) - log-likelihood of each log sample
        under the corresponding sim distribution
    """

    n_agents, log_sample_size = log_samples.shape

    # Clip samples to valid range
    log_samples_clipped = np.clip(log_samples, min_val, max_val)
    sim_samples_clipped = np.clip(sim_samples, min_val, max_val)

    # Create bin edges
    edges = np.linspace(min_val, max_val, num_bins + 1)

    # Build histogram for each agent from sim samples
    # sim_counts shape: (n_agents, num_bins)
    sim_counts = np.array([np.histogram(sim_samples_clipped[i], bins=edges)[0] for i in range(n_agents)])

    # Apply smoothing and normalize to probabilities
    sim_counts = sim_counts.astype(float) + additive_smoothing
    sim_probs = sim_counts / sim_counts.sum(axis=1, keepdims=True)

    # For each log sample, create a histogram to identify which bin it belongs to
    # Flatten log samples: (n_agents * log_sample_size, 1)
    log_values_flat = log_samples_clipped.reshape(-1, 1)

    # Build histogram for each flattened log sample
    # log_counts shape: (n_agents * log_sample_size, num_bins)
    log_counts = np.array([np.histogram(log_values_flat[i], bins=edges)[0] for i in range(n_agents * log_sample_size)])

    # Find which bin each log sample belongs to (argmax over bins)
    log_bins = np.argmax(log_counts, axis=1)

    # Reshape to (n_agents, log_sample_size)
    log_bins = log_bins.reshape(n_agents, log_sample_size)

    # For each agent and log sample, get the log probability from the sim distribution
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
    """Computes log-likelihood estimates for time-series simulated features.

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
        Shape (n_agents, n_steps) - log-likelihood estimates
    """
    n_agents, n_rollouts, n_steps = sim_values.shape

    if treat_timesteps_independently:
        # Ignore temporal structure: We end up with (n_agents, n_rollouts * n_steps)
        log_flat = log_values.reshape(n_agents, n_steps)
        sim_flat = sim_values.reshape(n_agents, n_rollouts * n_steps)

        # Compute log-likelihoods
        log_probs = histogram_estimate(log_flat, sim_flat, min_val, max_val, num_bins, additive_smoothing)
        # Reshape back: (n_agents, n_steps)
        log_probs_reshaped = log_probs.reshape(n_agents, n_steps)
    else:
        raise NotImplementedError("Currently not supported.")

    # Sanity check visualization
    if sanity_check:
        _plot_histogram_sanity_check(log_flat, sim_flat, log_probs, plot_agent_idx)

    return log_probs_reshaped


def _plot_histogram_sanity_check(
    log_samples: np.ndarray,
    sim_samples: np.ndarray,
    log_probs: np.ndarray,
    idx: int,
):
    """Plot data as sanity check."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
        color="red",
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
    axes[2].scatter(log_samples[idx], log_probs[idx], alpha=0.7, color="green")
    axes[2].set_xlabel("Ground-truth Value")
    axes[2].set_ylabel("Log-likelihood")
    axes[2].set_title("Log-likelihood of Ground-truth")
    axes[2].grid(alpha=0.3)
    axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"histogram_sanity_check_agent_{idx}.png")
