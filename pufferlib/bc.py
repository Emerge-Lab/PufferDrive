"""Behavior cloning for the Drive policy: `puffer bc puffer_drive [overrides]`.

Rolls replay logs with the SDC on the expert_tracking controller, which writes
the discrete action reproducing the next logged pose into the actions buffer,
then fits the actor head by cross-entropy on those (observation, action) pairs.
The checkpoint is saved like a training run so it can be loaded through
load_model_path or train.kl_ref_model_path.
"""

import copy
import importlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import pufferlib
from pufferlib.config_schema import (
    normalize_puffer_drive_config,
    validate_puffer_drive_config,
    validate_puffer_drive_resources,
)
from pufferlib.ocean.drive.drive import Drive


def collect_expert_dataset(env, num_steps):
    """Steps the env with placeholder actions and returns (observations, expert_actions) for valid samples."""
    num_agents = env.num_agents
    obs_dim = env.single_observation_space.shape[0]
    observations = np.empty((num_steps, num_agents, obs_dim), dtype=np.float32)
    expert_actions = np.empty((num_steps, num_agents), dtype=np.int64)
    valid = np.empty((num_steps, num_agents), dtype=bool)
    placeholder_actions = np.zeros_like(env.actions)

    env.reset()
    for step in range(num_steps):
        observations[step] = env.observations
        env.step(placeholder_actions)
        expert_actions[step] = env.actions.reshape(num_agents)
        valid[step] = env.masks.reshape(num_agents) != 0

    keep = valid.reshape(-1)
    return observations.reshape(-1, obs_dim)[keep], expert_actions.reshape(-1)[keep]


def fit_actor(policy, observations, expert_actions, bc_config, device):
    """Cross-entropy fit with early stopping on a held-out split; returns the best state dict and metrics."""
    num_samples = observations.shape[0]
    num_val = int(num_samples * bc_config["val_fraction"])
    if num_val < 1 or num_samples - num_val < bc_config["batch_size"]:
        raise pufferlib.APIUsageError(
            f"bc: {num_samples} samples is too few for val_fraction={bc_config['val_fraction']} "
            f"and batch_size={bc_config['batch_size']}"
        )
    permutation = torch.randperm(num_samples)
    val_idx, train_idx = permutation[:num_val], permutation[num_val:]
    observations = torch.from_numpy(observations)
    expert_actions = torch.from_numpy(expert_actions)
    val_obs = observations[val_idx].to(device)
    val_actions = expert_actions[val_idx].to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=bc_config["learning_rate"])
    batch_size = bc_config["batch_size"]
    best_val_loss = float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    best_epoch = 0
    history = []
    for epoch in range(bc_config["max_epochs"]):
        policy.train()
        train_loss_sum = 0.0
        train_batches = 0
        for start in range(0, train_idx.numel(), batch_size):
            batch = train_idx[start : start + batch_size]
            logits, _ = policy(observations[batch].to(device))
            loss = F.cross_entropy(logits, expert_actions[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_batches += 1

        policy.eval()
        with torch.no_grad():
            val_logits, _ = policy(val_obs)
            val_loss = F.cross_entropy(val_logits, val_actions).item()
            val_accuracy = (val_logits.argmax(-1) == val_actions).float().mean().item()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_batches, 1),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"bc epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(policy.state_dict())
            best_epoch = epoch
        elif epoch - best_epoch >= bc_config["patience"]:
            print(f"bc: early stop at epoch {epoch}, best epoch {best_epoch}")
            break

    return best_state, {
        "num_samples": num_samples,
        "num_val": num_val,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def bc(env_name, args=None):
    from pufferlib.pufferl import _save_experiment_config, load_config, torch_device

    args = args or load_config(env_name)
    args = normalize_puffer_drive_config(args, "behavior cloning")
    validate_puffer_drive_config(args, "behavior cloning")
    validate_puffer_drive_resources(args, "behavior cloning")
    if args["env"]["sdc_controller"] != "expert_tracking":
        raise pufferlib.APIUsageError("puffer bc requires env.sdc_controller=expert_tracking")
    if args["rnn_name"] is not None:
        raise pufferlib.APIUsageError("puffer bc does not support recurrent policies")
    if args["policy"]["action_type"] != "discrete":
        raise pufferlib.APIUsageError("puffer bc requires policy.action_type=discrete")

    torch.manual_seed(args["train"]["seed"])
    device = torch_device(args["train"]["device"])
    bc_config = args["bc"]

    env = Drive(**args["env"], seed=args["vec"]["seed"])
    collect_start = time.time()
    observations, expert_actions = collect_expert_dataset(env, bc_config["num_steps"])
    print(
        f"bc: collected {observations.shape[0]} samples from {bc_config['num_steps']} steps x {env.num_agents} envs "
        f"in {time.time() - collect_start:.1f}s"
    )

    env_module = importlib.import_module("pufferlib.ocean")
    policy_cls = getattr(env_module.torch, args["policy_name"])
    policy = policy_cls(env, **args["policy"]).to(device)
    env.close()

    best_state, metrics = fit_actor(policy, observations, expert_actions, bc_config, device)

    data_dir = args["train"]["data_dir"]
    models_dir = os.path.join(data_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"model_{env_name}_bc.pt")
    torch.save(best_state, model_path)
    _save_experiment_config(args, data_dir)
    with open(os.path.join(data_dir, "bc_metrics.json"), "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(f"bc: saved {model_path} (best val loss {metrics['best_val_loss']:.4f} at epoch {metrics['best_epoch']})")
    return model_path
