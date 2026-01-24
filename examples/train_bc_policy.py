import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.categorical import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pufferlib.pufferl import load_config, load_env

CHECKPOINT_PATH = "models"


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


def train_bc_policy(obs, actions, config):
    """Step 2: Train behavioral cloning policy"""
    print("Training BC policy...")

    # Initialize wandb
    wandb.init(project="gsp_bc_policy", config=config)

    wandb.log({"dataset_size": obs.shape[0]})

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_tensor = obs.float()
    actions_tensor = actions.long()

    dataset = TensorDataset(obs_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    data_iter = iter(dataloader)

    # Create model
    policy = BCPolicy(
        input_size=obs.shape[-1],
        hidden_size=config["hidden_size"],
        output_size=config["num_actions"],
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


if __name__ == "__main__":
    # Load configuration
    args = load_config("puffer_drive")
    args["vec"]["backend"] = "Serial"
    args["env"]["num_maps"] = 229
    args["env"]["map_dir"] = "pufferlib/resources/drive/binaries/validation_interactive_small"

    for max_expert_sequences in [4096]:
        args["env"]["num_agents"] = max_expert_sequences

        config = {
            "batch_size": 512,
            "hidden_size": 1024,
            "num_actions": 21 * 31,  # TODO: Do not hardcode
            "learning_rate": 1e-4,
            "epochs": 100_000,
            "minibatches": 4,
            "max_expert_sequences": max_expert_sequences,
        }

        env = load_env("puffer_drive", args)

        # Step 1: Prepare human data (o_t, a_t) tuples
        human_obs, human_actions = prepare_human_data(env.driver_env, max_expert_sequences=max_expert_sequences)

        print(f"Data shapes - Obs: {human_obs.shape}, Actions: {human_actions.shape}")

        # Step 2: Train BC policy
        losses, policy = train_bc_policy(human_obs, human_actions, config)

        # Store the trained model
        torch.save(policy.state_dict(), f"{CHECKPOINT_PATH}/bc_policy.pt")

        env.close()

        wandb.finish()
