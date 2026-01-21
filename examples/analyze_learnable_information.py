import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.categorical import Categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pufferlib.pufferl import load_config, load_env


class BCPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, output_size)])

    def dist(self, obs):
        """Generate action distribution."""
        x_out = self.nn(obs.float())
        return [Categorical(logits=head(x_out)) for head in self.heads]

    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        action_dist = self.dist(obs)

        if deterministic:
            actions_idx = action_dist[0].logits.argmax(axis=-1)
        else:
            actions_idx = action_dist[0].sample()
        return actions_idx

    def _log_prob(self, obs, expert_actions):
        pred_action_dist = self.dist(obs)
        log_prob = pred_action_dist[0].log_prob(expert_actions).mean()
        return log_prob


def prepare_human_data(env, max_expert_sequences=512):
    """Step 1: Extract and process human demonstration data"""
    print("Preparing human data...")

    env._prep_human_data(
        bptt_horizon=1,
        max_expert_sequences=max_expert_sequences,
    )

    # Access the raw expert data collected by the environment
    expert_actions_discrete = torch.Tensor(env.expert_actions_discrete)  # Shape: (T, N, 1)
    expert_observations = torch.Tensor(env.expert_observations_full)  # Shape: (T, N, obs_dim)

    # Flatten to create a batch of samples
    action_labels = torch.flatten(expert_actions_discrete, 0, 1).squeeze()  # [B]
    observations = torch.flatten(expert_observations, 0, 1)  # [B, obs_dim]

    # Filter out invalid actions (-1)
    valid_mask = action_labels != -1
    action_labels = action_labels[valid_mask]
    observations = observations[valid_mask]

    return observations, action_labels


def compute_observation_coverage(obs, max_expert_sequences, subsample_for_viz=5000):
    """
    Compute observation space coverage metrics.

    Metrics:
    1. Statistical Dispersion: variance, std, range per dimension
    2. Effective Rank: PCA components needed to explain 95% variance
    3. Visualizations: PCA projections and distributions
    """
    print("\nAnalyzing observation coverage...")

    obs_np = obs.numpy() if isinstance(obs, torch.Tensor) else obs
    n_samples, obs_dim = obs_np.shape

    # === 1. Statistical Dispersion Metrics ===
    obs_std = np.std(obs_np, axis=0)
    obs_var = np.var(obs_np, axis=0)
    obs_min = np.min(obs_np, axis=0)
    obs_max = np.max(obs_np, axis=0)
    obs_range = obs_max - obs_min

    # Average metrics across dimensions
    avg_std = np.mean(obs_std)
    avg_var = np.mean(obs_var)
    avg_range = np.mean(obs_range)

    print(f"  Avg Std: {avg_std:.4f}")
    print(f"  Avg Variance: {avg_var:.4f}")
    print(f"  Avg Range: {avg_range:.4f}")

    # === 2. Effective Rank (PCA) ===
    pca = PCA()
    pca.fit(obs_np)
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)

    # Effective rank (number of dimensions to explain 95% variance)
    effective_rank = np.argmax(cumulative_var >= 0.95) + 1

    print(f"  Effective Rank (95% var): {effective_rank}/{obs_dim}")

    # === 3. Visualizations ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Variance per dimension
    axes[0, 0].bar(range(min(50, obs_dim)), obs_var[:50], color="steelblue", edgecolor="black")
    axes[0, 0].set_xlabel("Dimension", fontsize=11)
    axes[0, 0].set_ylabel("Variance", fontsize=11)
    axes[0, 0].set_title("Variance per Dimension (first 50)", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Range per dimension
    axes[0, 1].bar(range(min(50, obs_dim)), obs_range[:50], color="coral", edgecolor="black")
    axes[0, 1].set_xlabel("Dimension", fontsize=11)
    axes[0, 1].set_ylabel("Range", fontsize=11)
    axes[0, 1].set_title("Range per Dimension (first 50)", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: PCA explained variance
    axes[1, 0].plot(cumulative_var[: min(100, len(cumulative_var))], "b-", linewidth=2, label="Cumulative Variance")
    axes[1, 0].axhline(y=0.95, color="r", linestyle="--", linewidth=2, label="95% threshold")
    axes[1, 0].axvline(
        x=effective_rank - 1, color="g", linestyle="--", linewidth=2, label=f"Effective rank = {effective_rank}"
    )
    axes[1, 0].set_xlabel("Number of Components", fontsize=11)
    axes[1, 0].set_ylabel("Cumulative Explained Variance", fontsize=11)
    axes[1, 0].set_title("PCA Cumulative Variance", fontsize=12, fontweight="bold")
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: 2D PCA projection
    if n_samples > subsample_for_viz:
        viz_indices = np.random.choice(n_samples, subsample_for_viz, replace=False)
        obs_viz = obs_np[viz_indices]
    else:
        obs_viz = obs_np

    pca_2d = PCA(n_components=2)
    obs_2d = pca_2d.fit_transform(obs_viz)

    axes[1, 1].scatter(obs_2d[:, 0], obs_2d[:, 1], alpha=0.4, s=10, c="purple", edgecolors="none")
    axes[1, 1].set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} var)", fontsize=11)
    axes[1, 1].set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} var)", fontsize=11)
    axes[1, 1].set_title(f"2D PCA Projection (N={n_samples} samples)", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Observation Coverage Analysis (N={max_expert_sequences} sequences)", fontsize=14, fontweight="bold", y=1.00
    )
    plt.tight_layout()

    # Save and log
    plt.savefig(f"coverage_analysis_N{max_expert_sequences}.png", dpi=150, bbox_inches="tight")
    wandb.log({f"coverage_analysis": wandb.Image(fig)})
    plt.close()

    # Log metrics to wandb
    coverage_metrics = {
        "coverage/dataset_size": n_samples,
        "coverage/obs_dim": obs_dim,
        "coverage/avg_std": avg_std,
        "coverage/avg_variance": avg_var,
        "coverage/avg_range": avg_range,
        "coverage/effective_rank": effective_rank,
        "coverage/effective_rank_ratio": effective_rank / obs_dim,
        "coverage/pca_var_pc1": explained_var_ratio[0],
        "coverage/pca_var_pc2": explained_var_ratio[1] if len(explained_var_ratio) > 1 else 0,
    }

    wandb.log(coverage_metrics)

    return coverage_metrics


def train_bc_policy(obs, actions, config):
    """Step 2: Train behavioral cloning policy"""
    print("Training BC policy...")

    # Initialize wandb
    wandb.init(project="gsp_epiplexity", config=config)

    wandb.log({"dataset_size": obs.shape[0]})

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    obs_tensor = obs.float()
    actions_tensor = actions.long()

    dataset = TensorDataset(obs_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    data_iter = iter(dataloader)

    # Create model
    policy = BCPolicy(
        input_size=obs.shape[-1], hidden_size=config["hidden_size"], output_size=config["num_actions"]
    ).to(device)

    optimizer = Adam(policy.parameters(), lr=config["learning_rate"])

    # Training loop
    losses = []
    global_step = 0

    for epoch in range(config["epochs"]):
        epoch_losses = []

        for i in range(config["minibatches"]):
            try:
                batch_obs, batch_actions = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_obs, batch_actions = next(data_iter)

            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            # Forward pass
            log_prob = policy._log_prob(batch_obs, batch_actions.float())
            loss = -log_prob

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute accuracy
            with torch.no_grad():
                pred_action = policy(batch_obs, deterministic=True)
                accuracy = (batch_actions == pred_action).sum() / batch_actions.shape[0]

            # Log
            loss_val = loss.item()
            epoch_losses.append(loss_val)
            losses.append(loss_val)

            wandb.log({"global_step": global_step, "loss": loss_val, "accuracy": accuracy.item(), "epoch": epoch})

            global_step += 1

        avg_epoch_loss = np.mean(epoch_losses)

        if avg_epoch_loss < 0.001:
            print(f"Early stopping at epoch {epoch + 1} with loss {avg_epoch_loss:.6f}")
            break
        else:
            print(f"Epoch {epoch + 1}/{config['epochs']}: Loss = {avg_epoch_loss:.4f}")

    return losses, policy


def compute_epiplexity(losses, dataset_size):
    """Step 3: Compute area under loss curve above final loss (epiplexity)"""
    print("Computing epiplexity...")

    # Convert to numpy array
    losses_array = np.array(losses)

    # Take final loss as the last value (asymptotic loss)
    final_loss = losses_array[-1]

    # Epiplexity: area under the curve above the final loss
    losses_above_final = losses_array - final_loss
    epiplexity = np.trapezoid(losses_above_final)

    print(f"Final Loss: {final_loss:.4f}")
    print(f"Epiplexity (AUC above final loss): {epiplexity:.4f}")

    # Log to wandb
    wandb.log({"epiplexity": epiplexity, "final_loss": final_loss})

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot loss curve
    steps = np.arange(len(losses_array))
    ax.plot(steps, losses_array, linewidth=1.5, label="Training loss", zorder=3)

    # Draw horizontal line at final loss
    ax.axhline(y=final_loss, color="red", linestyle="--", linewidth=2, label=f"Final Loss = {final_loss:.4f}", zorder=2)

    # Fill the area between loss curve and final loss (epiplexity area)
    ax.fill_between(
        steps,
        losses_array,
        final_loss,
        where=(losses_array >= final_loss),
        color="red",
        alpha=0.3,
        label=f"Epiplexity = {epiplexity:.4f}",
        zorder=1,
    )

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"BC training loss curve (N={dataset_size} samples)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Log plot to wandb
    wandb.log({"loss_curve_with_epiplexity": wandb.Image(fig)})
    plt.close()

    return epiplexity, final_loss


if __name__ == "__main__":
    # Load configuration
    args = load_config("puffer_drive")
    args["vec"]["backend"] = "Serial"

    for max_expert_sequences in [256, 512, 1024, 2048, 4096]:
        args["env"]["num_agents"] = max_expert_sequences

        config = {
            "batch_size": 512,
            "hidden_size": 1024,
            "num_actions": 91,  # 7*13 for classic discrete action space
            "learning_rate": 1e-4,
            "epochs": 10_000,
            "minibatches": 4,
            "max_expert_sequences": max_expert_sequences,
        }

        env = load_env("puffer_drive", args)

        # Step 1: Prepare human data (o_t, a_t) tuples
        human_obs, human_actions = prepare_human_data(env.driver_env, max_expert_sequences=max_expert_sequences)

        print(f"Data shapes - Obs: {human_obs.shape}, Actions: {human_actions.shape}")

        # Step 2: Train BC policy
        losses, policy = train_bc_policy(human_obs, human_actions, config)

        # Analyze observation coverage
        # coverage_metrics = compute_observation_coverage(human_obs, max_expert_sequences)

        # Step 3: Compute epiplexity with visualization
        epiplexity, final_loss = compute_epiplexity(losses, human_obs.shape[0])

        print("\n" + "=" * 60)
        print(f"RESULTS (N={max_expert_sequences}):")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Epiplexity: {epiplexity:.4f}")
        print("=" * 60 + "\n")

        env.close()

        wandb.finish()
