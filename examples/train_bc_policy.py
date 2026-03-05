import wandb
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.categorical import Categorical
import numpy as np
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
        # We map the observation to a single joint discrete action
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

    def get_action_dist_logits(self, obs):
        """Get the action distribution logits conditioned on the observation."""
        return self.dist(obs)[0].logits

    def _log_prob(self, obs, expert_actions):
        pred_action_dist = self.dist(obs)
        log_prob = pred_action_dist[0].log_prob(expert_actions.squeeze(-1).long()).mean()
        return log_prob


def load_data(driver_env):
    """Resample maps and prepare a fresh batch of human demonstrations."""
    driver_env.resample_maps()
    total_samples, unique_samples = driver_env._prepare_human_data()
    print(f"Resampled: {total_samples} samples ({unique_samples} unique)")
    wandb.log({"data/total_samples": total_samples, "data/unique_samples": unique_samples})

    obs = driver_env.expert_observations_full.float()
    actions = driver_env.expert_actions_discrete.long()

    # Zero out conditioning slots for BC training —
    # the BC/anchor policy should not depend on these values
    obs[:, driver_env.lambda_obs_idx] = 0.0
    obs[:, driver_env.reward_veh_obs_idx] = 0.0
    obs[:, driver_env.reward_offroad_obs_idx] = 0.0
    obs[:, driver_env.reward_goal_obs_idx] = 0.0

    return TensorDataset(obs, actions)


if __name__ == "__main__":
    args = load_config("puffer_drive")
    args["vec"]["backend"] = "Serial"
    args["env"]["num_maps"] = 100
    args["env"]["map_dir"] = "resources/drive/binaries/interactive_data_training_100"
    args["env"]["reg_mode"] = "log_prob_direct"  # To get the data
    args["base"]["rnn_name"] = "none"
    args["env"]["fix_lambdas"] = True
    args["env"]["fix_rewards"] = True
    args["env"]["lambda_value"] = 0.0

    config = {
        "batch_size": 512,
        "hidden_size": 1024,
        "num_actions": 21 * 31,
        "learning_rate": 1e-4,
        "epochs": 1500,
        "minibatches": 64,
        "resample_every_n_epochs": 10,
        "num_maps": args["env"]["num_maps"],
    }

    env = load_env("puffer_drive", args)
    driver_env = env.driver_env

    wandb.init(project="kl_anchor", tags=["bc_policy"], name=f"bc_policy_maps_{config['num_maps']}", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    policy = BCPolicy(
        input_size=driver_env.num_obs,
        hidden_size=config["hidden_size"],
        output_size=config["num_actions"],
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

            # Compute accuracy
            with torch.no_grad():
                pred_action = policy(batch_obs, deterministic=True)
                accuracy = (batch_actions.squeeze(-1) == pred_action).float().mean()

            epoch_losses.append(loss.item())
            wandb.log({"loss": loss.item(), "accuracy": accuracy.item(), "epoch": epoch, "global_step": global_step})
            global_step += 1

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1}/{config['epochs']}: Loss = {avg_loss:.4f}")

        if avg_loss < 0.001:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save(policy.state_dict(), f"{CHECKPOINT_PATH}/bc_policy_{config['num_maps']}.pt")
    print(f"Saved BC policy to {CHECKPOINT_PATH}/bc_policy_{config['num_maps']}.pt")

    env.close()
    wandb.finish()
