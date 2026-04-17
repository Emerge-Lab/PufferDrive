"""
BC Policy training script with wandb sweep support.

Usage:
    python examples/train_bc_policy.py train                        # single run, default config
    python examples/train_bc_policy.py train --dynamics classic     # single run, classic dynamics
    python examples/train_bc_policy.py train --map-dir path/to/maps --num-maps 25000
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

CHECKPOINT_PATH = "models/anchors"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "best_avg_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [1e-4]},
        "input_size": {"values": [128]},
        "hidden_size": {"values": [512]},
        "batch_size": {"values": [2048]},
        "resample_every_n_epochs": {"values": [1]},
        "num_maps": {"values": [67, 200, 1200, 12_000]},  # 10 min, 30 min, 3 hr, 30 hr
    },
}

TRAIN_DEFAULTS = {
    "learning_rate": 1e-4,
    "input_size": 128,
    "hidden_size": 512,
    "batch_size": 2048,
    "resample_every_n_epochs": 1,  # Resample after k full passes through the dataset
    "epochs": 4000,
    "num_maps": 12_000,
    "eval_frequency": 10,  # Validation dataset
    "val_patience": 40,  # Stop if val loss doesn't improve for this many eval checks
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
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(path: str, policy: BCPolicy, metrics: dict):
    """Save model weights and performance metrics together.

    The checkpoint dict contains:
        "model_state_dict" : policy.state_dict()
        "metrics"          : {
            "num_maps"         : int,
            "val_accuracy"     : float,
            "train_accuracy"   : float,
            "train_loss"       : float,
            "val_loss"         : float,
        }

    Metrics are explicitly cast to plain Python scalars so the checkpoint
    is safe to load with weights_only=True (numpy scalars are not allowlisted
    by default in PyTorch >= 2.6).
    """
    clean_metrics = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
    checkpoint = {
        "model_state_dict": policy.state_dict(),
        "metrics": clean_metrics,
    }
    torch.save(checkpoint, path)


def load_bc_policy(
    checkpoint_path: str,
    obs_dim: int,
    input_size: int,
    max_partner_objects: int,
    partner_features: int,
    max_road_objects: int,
    road_features: int,
    ego_dim: int,
    hidden_size: int,
    output_sizes: list,
    device: str = "cpu",
) -> tuple[BCPolicy, dict]:
    """Load a BCPolicy from a checkpoint file.

    Supports both the legacy format (bare state_dict) and the new format
    (dict with 'model_state_dict' and 'metrics' keys).

    Returns:
        policy  : BCPolicy with weights loaded, set to eval() and frozen.
        metrics : Performance metrics dict (empty dict for legacy checkpoints).
    """
    policy = BCPolicy(
        obs_dim=obs_dim,
        input_size=input_size,
        max_partner_objects=max_partner_objects,
        partner_features=partner_features,
        max_road_objects=max_road_objects,
        road_features=road_features,
        ego_dim=ego_dim,
        hidden_size=hidden_size,
        output_sizes=output_sizes,
    ).to(device)

    import numpy as np

    # weights_only=True is the safe default in PyTorch >= 2.6. Legacy checkpoints
    # may contain numpy scalars in the metrics dict, so we allowlist that type.
    with torch.serialization.safe_globals([np.core.multiarray.scalar]):
        raw = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        # New format
        policy.load_state_dict(raw["model_state_dict"])
        metrics = raw.get("metrics", {})
    else:
        # Legacy format: raw is the state_dict itself
        policy.load_state_dict(raw)
        metrics = {}

    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False

    if metrics:
        print(
            f"Loaded BC policy from {checkpoint_path} | "
            f"num_maps={metrics.get('num_maps', 'N/A')}  "
            f"val_acc={metrics.get('val_accuracy', float('nan')):.4f}  "
            f"val_loss={metrics.get('val_loss', float('nan')):.4f}"
        )
    else:
        print(f"Loaded BC policy from {checkpoint_path} (legacy checkpoint, no metrics)")

    return policy, metrics


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
    args["env"]["map_dir"] = "resources/drive/binaries/training_50k"
    args["env"]["reg_mode"] = "log_prob_direct"
    args["env"]["dynamics_model"] = dynamics_model
    args["base"]["rnn_name"] = "none"
    args["env"]["fix_lambdas"] = True
    args["env"]["fix_rewards"] = True
    args["env"]["lambda_value"] = 0.0
    args["env"]["control_mode"] = "control_sdc_only"  # "control_agents" #"control_sdc_only"
    args["env"]["num_agents"] = 128
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
    for batch_obs, batch_actions in DataLoader(dataset, batch_size=len(dataset)):
        all_obs = batch_obs
        all_actions = batch_actions.numpy()

    if dynamics_model == "classic":
        NUM_STEER = binding.NUM_STEER_BINS
        NUM_ACCEL = binding.NUM_ACCEL_BINS
        accel_step = 8.0 / (NUM_ACCEL - 1)
        steer_step = 2.0 / (NUM_STEER - 1)

        accel_idx = all_actions.flatten().astype(int) // NUM_STEER
        steer_idx = all_actions.flatten().astype(int) % NUM_STEER
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

    elif dynamics_model == "delta_local":
        NUM_DX = binding.NUM_DX_BINS
        NUM_DY = binding.NUM_DY_BINS
        NUM_YAW = binding.NUM_YAW_BINS
        MAX_DX = 2.0
        MAX_DY = 2.0
        MAX_DYAW = np.pi / 4.0

        dx_step = 2 * MAX_DX / (NUM_DX - 1)
        dy_step = 2 * MAX_DY / (NUM_DY - 1)
        yaw_step = 2 * MAX_DYAW / (NUM_YAW - 1)

        expert_dx_vals = -MAX_DX + all_actions[:, 0].astype(int) * dx_step
        expert_dy_vals = -MAX_DY + all_actions[:, 1].astype(int) * dy_step
        expert_yaw_vals = -MAX_DYAW + all_actions[:, 2].astype(int) * yaw_step

        policy.eval()
        with torch.no_grad():
            pred = policy(all_obs.to(device), deterministic=True).cpu().numpy()
        pred_dx_vals = -MAX_DX + pred[:, 0].astype(int) * dx_step
        pred_dy_vals = -MAX_DY + pred[:, 1].astype(int) * dy_step
        pred_yaw_vals = -MAX_DYAW + pred[:, 2].astype(int) * yaw_step

        fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        fig.suptitle(
            f"BC Training Data vs Learned Policy — {len(all_actions)} samples, {dynamics_model}, {num_maps} maps",
            fontsize=14,
        )
        plot_configs = [
            (axes[0, 0], expert_dx_vals, "Expert ΔX", "steelblue", NUM_DX, (-MAX_DX, MAX_DX), "ΔX (m)"),
            (axes[0, 1], pred_dx_vals, "Learned ΔX", "orange", NUM_DX, (-MAX_DX, MAX_DX), "ΔX (m)"),
            (axes[1, 0], expert_dy_vals, "Expert ΔY", "steelblue", NUM_DY, (-MAX_DY, MAX_DY), "ΔY (m)"),
            (axes[1, 1], pred_dy_vals, "Learned ΔY", "orange", NUM_DY, (-MAX_DY, MAX_DY), "ΔY (m)"),
            (axes[2, 0], expert_yaw_vals, "Expert ΔYaw", "steelblue", NUM_YAW, (-MAX_DYAW, MAX_DYAW), "ΔYaw (rad)"),
            (axes[2, 1], pred_yaw_vals, "Learned ΔYaw", "orange", NUM_YAW, (-MAX_DYAW, MAX_DYAW), "ΔYaw (rad)"),
        ]
        for ax, vals, title, color, bins, rng, xlabel in plot_configs:
            ax.hist(vals, bins=bins, range=rng, edgecolor="black", alpha=0.7, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.set_title(title)
            ax.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    else:
        return

    wandb.log({"action_distribution": wandb.Image(fig)})
    plt.close(fig)


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
def train(dynamics_model: str, map_dir: str = None, num_maps_override: int = None):
    output_sizes = get_output_sizes(dynamics_model)

    run = wandb.init(
        project="bc_anchor_policy",
        tags=["bc_policy", dynamics_model],
        config={**TRAIN_DEFAULTS, "dynamics_model": dynamics_model, "output_sizes": output_sizes},
    )

    # Apply CLI overrides to wandb config
    if num_maps_override is not None:
        wandb.config.update({"num_maps": num_maps_override}, allow_val_change=True)
    if map_dir is not None:
        wandb.config.update({"map_dir": map_dir}, allow_val_change=True)

    config = wandb.config

    lr = config.learning_rate
    hidden_size = config.hidden_size
    batch_size = config.batch_size
    resample_every = config.resample_every_n_epochs
    epochs = config.epochs
    num_maps = config.num_maps
    eval_frequency = config.eval_frequency
    val_patience = config.val_patience

    print(
        f"dynamics={dynamics_model}  lr={lr}  hidden={hidden_size}  num_maps={num_maps}  "
        f"batch={batch_size}  resample_every={resample_every}  val_patience={val_patience}"
    )

    # Train env
    args = build_env_args(dynamics_model, num_maps=num_maps)
    if hasattr(config, "map_dir") and config.map_dir is not None:
        args["env"]["map_dir"] = config.map_dir

    effective_map_dir = args["env"]["map_dir"]
    run.name = f"{dynamics_model}_{os.path.basename(effective_map_dir)}_{num_maps}maps"

    env = load_env("puffer_drive", args)
    driver_env = env.driver_env

    # Validation env: same config but uses a held-out set of maps
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

    # Train data
    dataset = load_data(driver_env)
    minibatches = len(dataset) // batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)

    # Validation data
    val_dataset = load_data(val_driver_env)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_avg_loss = float("inf")
    global_step = 0

    # Metrics tracked for the final checkpoint
    last_train_accuracy = float("nan")
    last_train_loss = float("nan")
    last_val_accuracy = float("nan")
    last_val_loss = float("nan")

    # Early stopping state (tracked in units of eval checks, not epochs)
    best_val_loss = float("inf")
    val_patience_counter = 0
    stop_training = False

    for epoch in range(epochs):
        if epoch > 0 and epoch % resample_every == 0:
            dataset = load_data(driver_env)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            data_iter = iter(dataloader)

        epoch_losses = []
        epoch_accuracies = []
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
            epoch_accuracies.append(accuracy)

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
        last_train_loss = avg_loss
        last_train_accuracy = np.mean(epoch_accuracies)

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss

        wandb.log({"train/avg_loss": avg_loss, "train/best_avg_loss": best_avg_loss})

        # ------------------------------------------------------------------
        # Validation + patience-based early stopping
        # ------------------------------------------------------------------
        if epoch % eval_frequency == 0:
            val_metrics = evaluate(policy, val_dataloader, device)
            last_val_loss = val_metrics["loss"]
            last_val_accuracy = val_metrics["accuracy"]

            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                val_patience_counter = 0
            else:
                val_patience_counter += 1

            wandb.log(
                {
                    **{"val/" + k: v for k, v in val_metrics.items()},
                    "early_stopping/patience_counter": val_patience_counter,
                    "early_stopping/best_val_loss": best_val_loss,
                }
            )
            print(
                f"  val: loss={last_val_loss:.4f}  acc={last_val_accuracy:.4f}  "
                f"patience={val_patience_counter}/{val_patience}"
            )

            if val_patience_counter >= val_patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: val loss has not improved for "
                    f"{val_patience} consecutive eval checks (best val loss={best_val_loss:.4f})"
                )
                stop_training = True

        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}  best={best_avg_loss:.4f}")

        if stop_training:
            break

        # Hard floor on train loss (original safety-net criterion)
        if avg_loss < 0.001:
            print(f"Early stopping at epoch {epoch + 1}: train loss below threshold")
            break

    # Final validation pass so saved metrics always reflect the last weights
    final_val_metrics = evaluate(policy, val_dataloader, device)
    last_val_loss = final_val_metrics["loss"]
    last_val_accuracy = final_val_metrics["accuracy"]
    wandb.log({"val/final_" + k: v for k, v in final_val_metrics.items()})

    # ------------------------------------------------------------------
    # Save checkpoint: weights + performance metrics
    # ------------------------------------------------------------------
    checkpoint_metrics = {
        "num_maps": num_maps,
        "val_accuracy": last_val_accuracy,
        "train_accuracy": last_train_accuracy,
        "train_loss": last_train_loss,
        "val_loss": last_val_loss,
    }

    save_path = f"{CHECKPOINT_PATH}/bc_{dynamics_model}_{num_maps}maps_{run.id}.pt"
    save_checkpoint(save_path, policy, checkpoint_metrics)
    wandb.save(save_path, base_path=".")
    print(
        f"Saved checkpoint: {save_path}\n"
        f"  num_maps={num_maps}  val_acc={last_val_accuracy:.4f}  val_loss={last_val_loss:.4f}  "
        f"  train_acc={last_train_accuracy:.4f}  train_loss={last_train_loss:.4f}"
    )
    wandb.summary["best_avg_loss"] = best_avg_loss
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary.update({"checkpoint_metrics/" + k: v for k, v in checkpoint_metrics.items()})

    save_action_distribution_plot(
        policy,
        dataset,
        dynamics_model,
        num_maps=num_maps,
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
        if name == "train":
            p.add_argument("--map-dir", type=str, default=None, help="Override map_dir")
            p.add_argument("--num-maps", type=int, default=None, help="Override num_maps")
            p.add_argument("--val-patience", type=int, default=None, help="Override val_patience")
        if name == "sweep":
            p.add_argument("--count", type=int, default=50, help="Number of sweep runs")

    return parser.parse_args()


if __name__ == "__main__":
    import sys

    args = parse_args()
    sys.argv = sys.argv[:1]  # strip before load_config sees them

    if args.mode == "train":
        train(
            dynamics_model=args.dynamics,
            map_dir=args.map_dir,
            num_maps_override=args.num_maps,
        )

    elif args.mode == "sweep":
        sweep_id = wandb.sweep(SWEEP_CONFIG, project="bc_anchor_policy")
        wandb.agent(sweep_id, function=lambda: train(args.dynamics), count=args.count)
