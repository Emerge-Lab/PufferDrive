"""Simplified estimators to compute log-likelihood of simulated trajs based on https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/sim_agents_metrics/estimators.py"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

np.set_printoptions(suppress=True)


def histogram_estimate(
    log_samples: np.ndarray,
    sim_samples: np.ndarray,
    min_val: float,
    max_val: float,
    num_bins: int,
    additive_smoothing: float,
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
    additive_smoothing: float,
    treat_timesteps_independently: bool = True,
    sim_valid: Optional[np.ndarray] = None,
    sanity_check: bool = False,
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
        sim_valid: Optional boolean array of shape (n_agents, n_rollouts, n_steps) indicating valid sim values
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

        # Filter if sim_valid is provided
        if sim_valid is not None:
            sim_valid_flat = sim_valid.reshape(n_agents, n_rollouts * n_steps)

            # Diagnostic: show average number of valid samples per agent
            valid_counts = np.sum(sim_valid_flat, axis=1)
            avg_valid_samples = np.mean(valid_counts)
            total_samples = n_rollouts * n_steps
            print(
                f"Average valid samples per agent for histogram: {avg_valid_samples:.2f} / {total_samples} ({100 * avg_valid_samples / total_samples:.1f}%)"
            )

            # Set invalid samples to NaN (histogram_estimate will filter these out)
            sim_flat = np.where(sim_valid_flat, sim_flat, np.nan)

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
        _plot_histogram_sanity_check(log_flat, sim_flat, log_probs)

    return log_probs


def bernoulli_estimate(
    log_samples: np.ndarray,
    sim_samples: np.ndarray,
    additive_smoothing: float,
) -> np.ndarray:
    """Computes log probabilities of samples based on Bernoulli distributions.

    Args:
        log_samples: Boolean array of shape (n_agents, sample_size)
        sim_samples: Boolean array of shape (n_agents, sample_size)
        additive_smoothing: Pseudocount for Laplace smoothing

    Returns:
        Shape (n_agents, sample_size) - log-likelihood of each log sample
    """
    if log_samples.dtype != bool:
        raise ValueError("log_samples must be boolean array for Bernoulli estimate")
    if sim_samples.dtype != bool:
        raise ValueError("sim_samples must be boolean array for Bernoulli estimate")

    return histogram_estimate(
        log_samples.astype(float),
        sim_samples.astype(float),
        min_val=-0.5,
        max_val=1.5,
        num_bins=2,
        additive_smoothing=additive_smoothing,
    )


def log_likelihood_estimate_scenario_level(
    log_values: np.ndarray,
    sim_values: np.ndarray,
    min_val: float,
    max_val: float,
    num_bins: int,
    additive_smoothing: float | None = None,
    use_bernoulli: bool = False,
) -> np.ndarray:
    """Computes log-likelihood estimates for scenario-level features (no time dimension).

    Args:
        log_values: Shape (n_agents,)
        sim_values: Shape (n_agents, n_rollouts)
        min_val: Minimum value for histogram bins (ignored if use_bernoulli=True)
        max_val: Maximum value for histogram bins (ignored if use_bernoulli=True)
        num_bins: Number of histogram bins (ignored if use_bernoulli=True)
        additive_smoothing: Pseudocount for Laplace smoothing
        use_bernoulli: If True, use Bernoulli estimator for boolean features

    Returns:
        Shape (n_agents,) - log-likelihood of each log feature
    """
    if log_values.ndim != 1:
        raise ValueError(f"log_values must be 1D, got shape {log_values.shape}")
    if sim_values.ndim != 2:
        raise ValueError(f"sim_values must be 2D, got shape {sim_values.shape}")

    log_values_2d = log_values[:, np.newaxis]
    sim_values_2d = sim_values

    if use_bernoulli:
        log_likelihood_2d = bernoulli_estimate(
            log_values_2d.astype(bool),
            sim_values_2d.astype(bool),
            additive_smoothing=0.001,
        )
    else:
        log_likelihood_2d = histogram_estimate(
            log_values_2d,
            sim_values_2d,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
        )

    return log_likelihood_2d[:, 0]


def _plot_histogram_sanity_check(
    log_samples: np.ndarray,
    sim_samples: np.ndarray,
    log_probs: np.ndarray,
    num_rollouts: int = 32,
    num_steps: int = 81,
):
    """Plot data as sanity check."""

    sns.set_theme(style="ticks")

    for agent_idx in range(10):
        log_samples_agent = log_samples[agent_idx]
        sim_samples_agent = sim_samples[agent_idx]
        log_probs_agent = log_probs[agent_idx]

        valid_mask_logs = ~np.isnan(log_samples_agent)
        valid_mask_sims = ~np.isnan(sim_samples_agent)

        log_probs_agent = log_probs_agent[valid_mask_logs]
        sim_samples_agent_flat = sim_samples_agent[valid_mask_sims]
        log_samples_agent_flat = log_samples_agent[valid_mask_logs]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
        fig.suptitle(f"Agent {agent_idx}", fontsize=12)

        # Plot 1: Time-series of simulated samples
        sim_feat_timeseries = sim_samples_agent.reshape(num_rollouts, num_steps)
        for rollout_idx in range(sim_feat_timeseries.shape[0]):
            axes[0].plot(
                list(range(num_steps)), sim_feat_timeseries[rollout_idx, :], color=sns.color_palette()[0], alpha=0.5
            )

        axes[0].plot(list(range(num_steps)), log_samples_agent, color="green", alpha=1.0)
        axes[0].set_ylabel("Value")
        axes[0].set_xlabel("Timestep")
        axes[0].set_title("Feature time-series")

        # Plot 2: Ground-truth values overlaid on simulated
        sns.histplot(log_samples_agent_flat, alpha=0.5, color="green", label="Ground-truth", stat="density", ax=axes[1])
        sns.histplot(
            sim_samples_agent_flat,
            alpha=0.5,
            color=sns.color_palette()[0],
            label="Simulated",
            stat="density",
            ax=axes[1],
        )
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Ground-truth vs simulated")
        axes[1].legend()

        # Plot 3: GT value x likelihood score
        likelihood = np.exp(log_probs_agent)
        sns.scatterplot(x=log_samples_agent_flat, y=likelihood, color="purple", alpha=0.7, ax=axes[2])
        axes[2].set_title(f"Average likelihood for agent \n across [R, T]: {likelihood.mean():.3f}")
        axes[2].set_ylabel("Likelihood")
        axes[2].set_xlabel("Value")

        plt.tight_layout()
        plt.savefig(f"histogram_sanity_check_agent_{agent_idx}.png", dpi=150, bbox_inches="tight")
        plt.close()
