"""
BC Policy training script supporting both classic (joint) and delta_local (independent MultiDiscrete).

Classic: 1 head with NUM_ACCEL_BINS * NUM_STEER_BINS outputs
Delta local: 3 independent heads with NUM_DX_BINS, NUM_DY_BINS, NUM_YAW_BINS outputs

Usage:
    python examples/train_bc_policy.py classic
    python examples/train_bc_policy.py delta_local
"""

import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.categorical import Categorical
import numpy as np
from pufferlib.pufferl import load_config, load_env
from pufferlib.ocean.drive import binding

CHECKPOINT_PATH = "models"


class BCPolicy(nn.Module):
    """BC policy supporting both joint (single head) and independent (multi-head) action spaces."""

    def __init__(self, input_size, hidden_size, output_sizes):
        """
        Args:
            input_size: observation dimension
            hidden_size: hidden layer size
            output_sizes: list of ints. For classic: [651]. For delta_local: [21, 21, 127].
        """
        super().__init__()
        self.num_heads = len(output_sizes)
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
        self.heads = nn.ModuleList([nn.Linear(hidden_size, s) for s in output_sizes])

    def dist(self, obs):
        """Generate action distributions for all heads."""
        x_out = self.nn(obs.float())
        return [Categorical(logits=head(x_out)) for head in self.heads]

    def forward(self, obs, deterministic=False):
        """Generate actions from all heads."""
        dists = self.dist(obs)
        if deterministic:
            actions = [d.logits.argmax(dim=-1) for d in dists]
        else:
            actions = [d.sample() for d in dists]

        if self.num_heads == 1:
            return actions[0]
        return torch.stack(actions, dim=-1)

    def get_action_dist_logits(self, obs):
        """Get logits from all heads. Returns list of tensors."""
        return [d.logits for d in self.dist(obs)]

    def _log_prob(self, obs, expert_actions):
        """
        Compute mean log probability of expert actions.

        For single head: expert_actions shape (B, 1) or (B,)
        For multi head: expert_actions shape (B, num_heads)
        """
        dists = self.dist(obs)
        if self.num_heads == 1:
            log_prob = dists[0].log_prob(expert_actions.squeeze(-1).long()).mean()
        else:
            # Sum log probs across independent heads (product of independent distributions)
            log_prob = 0.0
            for i, d in enumerate(dists):
                log_prob = log_prob + d.log_prob(expert_actions[:, i].long()).mean()
            log_prob = log_prob / self.num_heads  # average across heads for comparable loss scale
        return log_prob


def load_data(driver_env):
    """Resample maps and prepare a fresh batch of human demonstrations."""
    driver_env.resample_maps()
    total_samples, unique_samples = driver_env._prepare_human_data()
    print(f"Resampled: {total_samples} samples ({unique_samples} unique)")
    wandb.log({"data/total_samples": total_samples, "data/unique_samples": unique_samples})

    obs = driver_env.expert_observations_full.float()
    actions = driver_env.expert_actions_discrete.long()

    # Zero out conditioning slots for BC training
    obs[:, driver_env.lambda_obs_idx] = 0.0
    obs[:, driver_env.reward_veh_obs_idx] = 0.0
    obs[:, driver_env.reward_offroad_obs_idx] = 0.0
    obs[:, driver_env.reward_goal_obs_idx] = 0.0

    return TensorDataset(obs, actions)

def compute_accuracy(policy, batch_obs, batch_actions):
    """Compute per-head and overall accuracy."""
    with torch.no_grad():
        pred = policy(batch_obs, deterministic=True)
        if policy.num_heads == 1:
            return (batch_actions.squeeze(-1) == pred).float().mean().item()
        else:
            correct = (batch_actions == pred).float()
            return correct.mean(dim=0).mean().item()


if __name__ == "__main__":
    import sys

    # Parse dynamics model before load_config eats sys.argv
    dynamics_model = "delta_local"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("classic", "delta_local"):
            dynamics_model = arg
            sys.argv.pop(i)
            break

    # Determine output head sizes
    if dynamics_model == "classic":
        output_sizes = [binding.NUM_ACCEL_BINS * binding.NUM_STEER_BINS]
    elif dynamics_model == "delta_local":
        output_sizes = [binding.NUM_DX_BINS, binding.NUM_DY_BINS, binding.NUM_YAW_BINS]
    else:
        raise ValueError(f"Unknown dynamics model: {dynamics_model}")

    args = load_config("puffer_drive")
    args["vec"]["backend"] = "Serial"
    args["env"]["num_maps"] = 100
    args["env"]["map_dir"] = "resources/drive/binaries/interactive_data_training_100"
    args["env"]["reg_mode"] = "log_prob_direct"
    args["env"]["dynamics_model"] = dynamics_model
    args["base"]["rnn_name"] = "none"
    args["env"]["fix_lambdas"] = True
    args["env"]["fix_rewards"] = True
    args["env"]["lambda_value"] = 0.0

    config = {
        "batch_size": 512,
        "hidden_size": 1024,
        "output_sizes": output_sizes,
        "learning_rate": 1e-4,
        "epochs": 1500,
        "minibatches": 64,
        "resample_every_n_epochs": 10,
        "num_maps": args["env"]["num_maps"],
        "dynamics_model": dynamics_model,
    }

    env = load_env("puffer_drive", args)
    driver_env = env.driver_env

    run_name = f"bc_{dynamics_model}_maps_{config['num_maps']}"
    wandb.init(project="kl_anchor", tags=["bc_policy", dynamics_model], name=run_name, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Dynamics model: {dynamics_model}")
    print(f"Action heads: {output_sizes} (total output nodes: {sum(output_sizes)})")

    policy = BCPolicy(
        input_size=driver_env.num_obs,
        hidden_size=config["hidden_size"],
        output_sizes=output_sizes,
    ).to(device)

    optimizer = Adam(policy.parameters(), lr=config["learning_rate"])

    dataset = load_data(driver_env)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    data_iter = iter(dataloader)

    global_step = 0
    for epoch in range(config["epochs"]):
        if epoch > 0 and epoch % config["resample_every_n_epochs"] == 0:
            dataset = load_data(driver_env)
            dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
            data_iter = iter(dataloader)

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

            accuracy = compute_accuracy(policy, batch_obs, batch_actions)

            epoch_losses.append(loss.item())
            wandb.log({"loss": loss.item(), "accuracy": accuracy, "epoch": epoch, "global_step": global_step})
            global_step += 1

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}/{config['epochs']}: Loss = {avg_loss:.4f}")

        if avg_loss < 0.001:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    save_path = f"{CHECKPOINT_PATH}/bc_{dynamics_model}_{config['num_maps']}.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Saved BC policy to {save_path}")

    env.close()
    wandb.finish()
