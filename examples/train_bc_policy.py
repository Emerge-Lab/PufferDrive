"""
BC Policy training script with wandb sweep support.

Usage:
    python examples/train_bc_policy.py train                        # single run, default config
    python examples/train_bc_policy.py train --dynamics classic     # single run, classic dynamics
    python examples/train_bc_policy.py sweep                        # launch sweep + agent
    python examples/train_bc_policy.py --dynamics classic
"""

import argparse
import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.categorical import Categorical
import numpy as np
import matplotlib.pyplot as plt

from pufferlib.pufferl import load_config, load_env
from pufferlib.ocean.drive import binding
import pufferlib
import pufferlib.models

CHECKPOINT_PATH = "models"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "best_avg_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 3e-5, "max": 1e-3},
        "input_size": {"values": [256, 512]},
        "hidden_size": {"values": [512, 1024, 2048]},
        "batch_size": {"values": [2048]},
        "resample_every_n_epochs": {"values": [5, 10]},
        "num_maps": {"values": [10000]},
    },
}

TRAIN_DEFAULTS = {
    "learning_rate": 1e-4,
    "input_size": 64,
    "hidden_size": 512,
    "batch_size": 2048,
    "resample_every_n_epochs": 2,  # Resample after k full passes through the dataset
    "epochs": 1000,
    "num_maps": 10000,
    "eval_frequency": 10,  # Validation dataset
}


class BCPolicy(nn.Module):
    """BC policy supporting both joint (single head) and independent (multi-head) action spaces."""

    def __init__(
        self,
        obs_dim,
        input_size,
        max_partner_objects,
        partner_features,
        max_road_objects,
        road_features,
        ego_dim,
        hidden_size,
        output_sizes,
    ):
        super().__init__()

        self.num_heads = len(output_sizes)
        self.hidden_size = hidden_size
        self.obs_dim = obs_dim

        self.max_partner_objects = max_partner_objects
        self.partner_features = partner_features
        self.max_road_objects = max_road_objects
        self.road_features = road_features
        self.road_features_after_onehot = road_features + 6
        self.ego_dim = ego_dim

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.ReLU(),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.ReLU(),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.ReLU(),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.shared_embedding = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, 512)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(512, hidden_size)),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, s) for s in output_sizes])

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim :]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        return F.relu(self.shared_embedding(concat_features))

    def dist(self, obs):
        hidden = self.encode_observations(obs.float())
        return [Categorical(logits=head(hidden)) for head in self.heads]

    def forward(self, obs, deterministic=False):
        dists = self.dist(obs)
        actions = [d.logits.argmax(dim=-1) if deterministic else d.sample() for d in dists]
        return actions[0] if self.num_heads == 1 else torch.stack(actions, dim=-1)

    def get_action_dist_logits(self, obs):
        """Get logits from all heads. Returns list of tensors."""
        return [d.logits for d in self.dist(obs)]

    def _log_prob(self, obs, expert_actions):
        dists = self.dist(obs)
        if self.num_heads == 1:
            return dists[0].log_prob(expert_actions.squeeze(-1).long()).mean()
        log_prob = sum(d.log_prob(expert_actions[:, i].long()).mean() for i, d in enumerate(dists))
        return log_prob / self.num_heads

    def _entropy(self, obs):
        dists = self.dist(obs)
        if self.num_heads == 1:
            return dists[0].entropy().mean().item()
        return {f"entropy_head_{i}": d.entropy().mean().item() for i, d in enumerate(dists)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_output_sizes(dynamics_model):
    if dynamics_model == "classic":
        return [binding.NUM_ACCEL_BINS * binding.NUM_STEER_BINS]
    if dynamics_model == "delta_local":
        return [binding.NUM_DX_BINS, binding.NUM_DY_BINS, binding.NUM_YAW_BINS]
    raise ValueError(f"Unknown dynamics model: {dynamics_model}")


def build_env_args(dynamics_model, num_maps):
    args = load_config("puffer_drive")
    args["vec"]["backend"] = "Serial"
    args["env"]["num_maps"] = num_maps
    args["env"]["map_dir"] = "resources/drive/binaries/training"
    args["env"]["reg_mode"] = "log_prob_direct"
    args["env"]["dynamics_model"] = dynamics_model
    args["base"]["rnn_name"] = "none"
    args["env"]["fix_lambdas"] = True
    args["env"]["fix_rewards"] = True
    args["env"]["lambda_value"] = 0.0
    return args


_cumulative_unique_hashes: set = set()


def load_data(driver_env):
    driver_env.resample_maps()
    total_samples, unique_samples = driver_env._prepare_human_data()

    obs_np = driver_env.expert_observations_full.numpy()
    act_np = driver_env.expert_actions_discrete.numpy()
    batch_hashes = {driver_env._hash_pair(obs_np[i], act_np[i]) for i in range(len(obs_np))}
    _cumulative_unique_hashes.update(batch_hashes)

    print(
        f"Resampled: {total_samples} samples ({unique_samples} unique, {len(_cumulative_unique_hashes)} cumulative unique)"
    )
    wandb.log(
        {
            "data/total_samples": total_samples,
            "data/unique_samples": unique_samples,
            "data/cumulative_unique_samples": len(_cumulative_unique_hashes),
        }
    )
    obs = driver_env.expert_observations_full.float()
    actions = driver_env.expert_actions_discrete.long()
    return TensorDataset(obs, actions)


def compute_accuracy(policy, batch_obs, batch_actions):
    with torch.no_grad():
        pred = policy(batch_obs, deterministic=True)
        if policy.num_heads == 1:
            return (batch_actions.squeeze(-1) == pred).float().mean().item()
        return (batch_actions == pred).float().mean(dim=0).mean().item()


def save_action_distribution_plot(policy, dataset, dynamics_model, num_maps, run_id, device):
    """Only implemented for classic dynamics (joint action head)."""
    if dynamics_model != "classic":
        return

    for batch_obs, batch_actions in DataLoader(dataset, batch_size=len(dataset)):
        all_actions = batch_actions.numpy().flatten()
        all_obs = batch_obs

    NUM_STEER = binding.NUM_STEER_BINS
    NUM_ACCEL = binding.NUM_ACCEL_BINS
    accel_step = 8.0 / (NUM_ACCEL - 1)
    steer_step = 2.0 / (NUM_STEER - 1)

    accel_idx = all_actions.astype(int) // NUM_STEER
    steer_idx = all_actions.astype(int) % NUM_STEER
    accel_vals = -4.0 + accel_idx * accel_step
    steer_vals = -1.0 + steer_idx * steer_step

    policy.eval()
    with torch.no_grad():
        pred = policy(all_obs.to(device), deterministic=True).cpu().numpy().flatten()
    pred_accel_idx = pred.astype(int) // NUM_STEER
    pred_steer_idx = pred.astype(int) % NUM_STEER
    pred_accel_vals = -4.0 + pred_accel_idx * accel_step
    pred_steer_vals = -1.0 + pred_steer_idx * steer_step

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"BC Training Data vs Learned Policy — {len(all_actions)} samples, {dynamics_model}, {num_maps} maps",
        fontsize=14,
    )
    for ax, vals, title, color in [
        (axes[0, 0], steer_vals, f"Expert Steering (non-zero: {(steer_idx != NUM_STEER // 2).sum()})", "steelblue"),
        (
            axes[0, 1],
            pred_steer_vals,
            f"Learned Steering (non-zero: {(pred_steer_idx != NUM_STEER // 2).sum()})",
            "orange",
        ),
        (axes[1, 0], accel_vals, "Expert Acceleration", "steelblue"),
        (axes[1, 1], pred_accel_vals, "Learned Acceleration", "orange"),
    ]:
        bins, rng = (NUM_STEER, (-1.0, 1.0)) if "Steer" in title else (NUM_ACCEL, (-4.0, 4.0))
        xlabel = "Steering (rad)" if "Steer" in title else "Acceleration (m/s²)"
        ax.hist(vals, bins=bins, range=rng, edgecolor="black", alpha=0.7, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = f"bc_action_distribution_{dynamics_model}_{num_maps}_{run_id}.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved action distribution plot to {path}")
    wandb.log({"action_distribution": wandb.Image(path)})


def evaluate(policy, dataloader, device):
    policy.eval()
    losses, accuracies = [], []
    entropy_accum = {}

    with torch.no_grad():
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            loss = -policy._log_prob(batch_obs, batch_actions.float())
            accuracy = compute_accuracy(policy, batch_obs, batch_actions)
            entropy = policy._entropy(batch_obs)

            losses.append(loss.item())
            accuracies.append(accuracy)
            if isinstance(entropy, dict):
                for k, v in entropy.items():
                    entropy_accum.setdefault(k, []).append(v)
            else:
                entropy_accum.setdefault("entropy", []).append(entropy)

    policy.train()
    metrics = {
        "loss": np.mean(losses),
        "accuracy": np.mean(accuracies),
        **{k: np.mean(v) for k, v in entropy_accum.items()},
    }
    return metrics


# ---------------------------------------------------------------------------
# Core training loop (called by both single-run and sweep agent)
# ---------------------------------------------------------------------------
def train(dynamics_model: str):
    output_sizes = get_output_sizes(dynamics_model)

    run = wandb.init(
        project="bc_anchor_policy",
        tags=["bc_policy", dynamics_model],
        config={**TRAIN_DEFAULTS, "dynamics_model": dynamics_model, "output_sizes": output_sizes},
    )
    config = wandb.config

    lr = config.learning_rate
    hidden_size = config.hidden_size
    batch_size = config.batch_size
    resample_every = config.resample_every_n_epochs
    epochs = config.epochs
    num_maps = config.num_maps
    eval_frequency = config.eval_frequency

    run.name = f"{dynamics_model}_maps{num_maps}"

    print(
        f"dynamics={dynamics_model}  lr={lr}  hidden={hidden_size}  num_maps={num_maps}"
        f"batch={batch_size}  resample_every={resample_every}"
    )

    # Train env
    args = build_env_args(dynamics_model, num_maps=num_maps)
    env = load_env("puffer_drive", args)
    driver_env = env.driver_env

    # Validation env: Same as train but uses different set of maps
    val_args = build_env_args(dynamics_model, num_maps=10000)
    val_args["env"]["map_dir"] = "resources/drive/binaries/validation"
    val_env = load_env("puffer_drive", val_args)
    val_driver_env = val_env.driver_env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    policy = BCPolicy(
        obs_dim=driver_env.num_obs,
        input_size=config.input_size,
        max_partner_objects=driver_env.max_partner_objects,
        partner_features=driver_env.partner_features,
        max_road_objects=driver_env.max_road_objects,
        road_features=driver_env.road_features,
        ego_dim=driver_env.ego_features,
        hidden_size=config.hidden_size,
        output_sizes=output_sizes,
    ).to(device)

    param_count = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {param_count:,}")
    wandb.log({"model/param_count": param_count})

    optimizer = Adam(policy.parameters(), lr=lr)
    # Train
    dataset = load_data(driver_env)
    minibatches = len(dataset) // batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)

    # Validation
    val_dataset = load_data(val_driver_env)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_avg_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):
        if epoch > 0 and epoch % resample_every == 0:
            dataset = load_data(driver_env)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            data_iter = iter(dataloader)

        epoch_losses = []
        for _ in range(minibatches):
            try:
                batch_obs, batch_actions = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_obs, batch_actions = next(data_iter)

            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            loss = -policy._log_prob(batch_obs, batch_actions.float())
            entropy = policy._entropy(batch_obs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = compute_accuracy(policy, batch_obs, batch_actions)
            epoch_losses.append(loss.item())

            # Log statistics
            wandb.log(
                {
                    f"train/{k}": v
                    for k, v in {
                        "loss": loss.item(),
                        "accuracy": accuracy,
                        "epoch": epoch,
                        "global_step": global_step,
                        **(entropy if isinstance(entropy, dict) else {"entropy": entropy}),
                    }.items()
                }
            )

            global_step += 1

        avg_loss = np.mean(epoch_losses)
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss

        wandb.log({"train/avg_loss": avg_loss, "train/best_avg_loss": best_avg_loss})

        if epoch % eval_frequency == 0:
            val_metrics = evaluate(policy, val_dataloader, device)
            wandb.log({"val/" + k: v for k, v in val_metrics.items()})
            print(f"  val: loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}")

        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}  best={best_avg_loss:.4f}")

        if avg_loss < 0.001:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    save_path = f"{CHECKPOINT_PATH}/bc_{dynamics_model}_{run.id}.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Saved checkpoint: {save_path}")
    wandb.summary["best_avg_loss"] = best_avg_loss

    save_action_distribution_plot(
        policy,
        dataset,
        dynamics_model,
        num_maps=args["env"]["num_maps"],
        run_id=run.id,
        device=device,
    )

    env.close()
    val_env.close()
    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="BC policy training")
    sub = parser.add_subparsers(dest="mode", required=True)

    for name, help_text in [
        ("train", "Single training run"),
        ("sweep", "Create a new sweep and attach an agent"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dynamics", choices=["classic", "delta_local"], default="delta_local")
        if name == "sweep":
            p.add_argument("--count", type=int, default=50, help="Number of sweep runs")

    return parser.parse_args()


if __name__ == "__main__":
    import sys

    args = parse_args()
    sys.argv = sys.argv[:1]  # strip before load_config sees them

    if args.mode == "train":
        train(dynamics_model=args.dynamics)

    elif args.mode == "sweep":
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="bc_anchor_policy")
        wandb.agent(sweep_id, function=lambda: train(args.dynamics), count=args.count)
