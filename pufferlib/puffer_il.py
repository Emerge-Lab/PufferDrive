from __future__ import annotations

import argparse
import ast
import configparser
import glob
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
import contextlib
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from rich_argparse import RichHelpFormatter

import pufferlib
import pufferlib.pufferl as pufferl
import pufferlib.pytorch
import pufferlib.spaces


IL_DEFAULTS = {
    "method": "dagger",
    "updates": 1,
    "expert_data": None,
    "save_expert_data": None,
    "teacher_load_model_path": None,
    "teacher_load_id": None,
    "bc_batch_size": 256,
    "bc_epochs": 4,
    "bc_rollout_steps": 8,
    "bc_teacher_target_samples": 131072,
    "dagger_iters": 10,
    "dagger_rollout_steps": 2048,
    "dagger_beta_start": 1.0,
    "dagger_beta_end": 0.0,
    "gail_updates": 10,
    "gail_rollout_steps": 2048,
    "gail_discriminator_hidden_size": 256,
    "gail_discriminator_lr": 3e-4,
    "gail_reward_scale": 1.0,
}


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor | None = None
    entropies: torch.Tensor | None = None
    dones: torch.Tensor | None = None


class TransitionBank:
    def __init__(self):
        self.observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []

    def __len__(self):
        return sum(batch.shape[0] for batch in self.observations)

    def add(self, observations, actions):
        obs = torch.as_tensor(observations).detach().cpu()
        act = torch.as_tensor(actions).detach().cpu()

        if obs.ndim == 0:
            obs = obs.unsqueeze(0)
        if act.ndim == 0:
            act = act.unsqueeze(0)

        if obs.shape[0] != act.shape[0]:
            raise pufferlib.APIUsageError(
                f"Observation batch size {obs.shape[0]} does not match action batch size {act.shape[0]}"
            )

        self.observations.append(obs)
        self.actions.append(act)

    def extend(self, observations, actions):
        self.add(observations, actions)

    def tensors(self):
        if not self.observations:
            return None, None

        return torch.cat(self.observations, dim=0), torch.cat(self.actions, dim=0)

    def sample(self, batch_size):
        observations, actions = self.tensors()
        if observations is None or actions is None:
            raise pufferlib.APIUsageError("No expert data available")

        if observations.shape[0] <= batch_size:
            indices = torch.arange(observations.shape[0])
        else:
            indices = torch.randperm(observations.shape[0])[:batch_size]

        return observations[indices], actions[indices]


class Discriminator(torch.nn.Module):
    def __init__(self, obs_space, action_space, hidden_size=256):
        super().__init__()
        self.obs_dim = int(np.prod(obs_space.shape))
        self.action_dim = action_feature_dim(action_space)
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(self.obs_dim + self.action_dim, hidden_size)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(hidden_size, hidden_size)),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )
        self.action_space = action_space

    def forward(self, observations, actions):
        obs = observations.reshape(observations.shape[0], -1).float()
        act = encode_actions(actions, self.action_space)
        inputs = torch.cat([obs, act], dim=-1)
        return self.net(inputs).squeeze(-1)


class ImitationTrainer:
    def __init__(self, config, vecenv, policy, teacher=None, logger=None):
        self.config = config
        self.vecenv = vecenv
        self.policy = policy
        self.teacher = teacher
        self.logger = logger or pufferl.NoLogger(config)

        self.device = normalize_device(config["train"]["device"])
        self.env_name = config["env_name"]
        self.method = config["il"]["method"].lower()
        self.data_dir = config["data_dir"]
        self.run_id = self.logger.run_id
        self.epoch = 0
        self.global_step = 0
        self.log_step = 0
        self.start_time = time.time()

        self.expert_bank = load_transition_bank(config["il"]["expert_data"])
        self.policy_bank = TransitionBank()

        if args_use_rnn(config):
            raise pufferlib.APIUsageError("puffer_il starter currently supports feedforward policies only")

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config["train"]["learning_rate"],
            betas=(config["train"]["adam_beta1"], config["train"]["adam_beta2"]),
            eps=config["train"]["adam_eps"],
        )
        self.discriminator = Discriminator(
            vecenv.single_observation_space,
            vecenv.single_action_space,
            hidden_size=config["il"]["gail_discriminator_hidden_size"],
        ).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=config["il"]["gail_discriminator_lr"]
        )

    @property
    def uptime(self):
        return time.time() - self.start_time

    def _forward_policy(self, model, observations, state=None):
        # Some policies expose forward_eval while others only implement forward.
        if hasattr(model, "forward_eval"):
            return model.forward_eval(observations, state)
        return model(observations, state)

    def _rollout_policy(self, steps, beta=0.0, teacher=None, collect_expert_labels=False):
        self.policy.eval()
        if teacher is not None:
            teacher.eval()

        self.vecenv.async_reset(seed=self.config["train"]["seed"])
        observations, _, terminals, truncations, infos, env_ids, masks = self.vecenv.recv()

        collected_obs = []
        collected_actions = []

        for _ in range(steps):
            observation_tensor = torch.as_tensor(observations, device=self.device)

            with torch.no_grad(), self._amp_context():
                logits, _ = self._forward_policy(self.policy, observation_tensor, None)
                student_actions, student_logprobs, student_entropy = pufferlib.pytorch.sample_logits(logits)

                if teacher is not None:
                    teacher_logits, _ = self._forward_policy(teacher, observation_tensor, None)
                    teacher_actions, _, _ = pufferlib.pytorch.sample_logits(teacher_logits)
                else:
                    teacher_actions = None

            env_actions = student_actions
            if teacher_actions is not None and beta > 0:
                use_teacher = torch.rand(student_actions.shape[0], device=self.device) < beta
                env_actions = student_actions.clone()
                env_actions[use_teacher] = teacher_actions[use_teacher]

            if collect_expert_labels and teacher_actions is not None:
                collected_actions.append(teacher_actions.detach().cpu())
            else:
                collected_actions.append(student_actions.detach().cpu())

            collected_obs.append(observation_tensor.detach().cpu())

            if isinstance(logits, torch.distributions.Normal):
                env_actions = torch.clamp(
                    env_actions,
                    torch.as_tensor(self.vecenv.action_space.low, device=env_actions.device),
                    torch.as_tensor(self.vecenv.action_space.high, device=env_actions.device),
                )

            self.vecenv.send(env_actions.cpu().numpy())
            observations, rewards, terminals, truncations, infos, env_ids, masks = self.vecenv.recv()
            self.global_step += int(np.asarray(masks).sum())

        observations = torch.cat(collected_obs, dim=0)
        actions = torch.cat(collected_actions, dim=0)
        return RolloutBatch(observations, actions)

    def _amp_context(self):
        if self.config["train"].get("amp", True) and str(self.device).startswith("cuda"):
            precision = self.config["train"]["precision"]
            return torch.amp.autocast(device_type="cuda", dtype=getattr(torch, precision))

        return contextlib.nullcontext()

    def _stack_bank(self, bank):
        observations, actions = bank.tensors()
        if observations is None or actions is None:
            raise pufferlib.APIUsageError("No data available")
        return observations, actions

    def _policy_batch_loss(self, observations, actions, weights=None):
        logits, _ = self._forward_policy(self.policy, observations, None)
        _, logprob, entropy = pufferlib.pytorch.sample_logits(logits, action=actions)

        loss = -logprob
        if weights is not None:
            loss = loss * weights

        return loss.mean(), entropy.mean()

    def train_bc(self):
        logs = {}
        if len(self.expert_bank) == 0:
            if self.teacher is None:
                raise pufferlib.APIUsageError("BC requires expert_data or a teacher policy/checkpoint")

            target_samples = int(self.config["il"].get("bc_teacher_target_samples", 0))
            if target_samples <= 0:
                raise pufferlib.APIUsageError("--il.bc-teacher-target-samples must be > 0 when using teacher-generated BC data")

            chunk_steps = int(self.config["il"].get("bc_rollout_steps", 0))
            if chunk_steps <= 0:
                raise pufferlib.APIUsageError("--il.bc-rollout-steps must be > 0")

            generated = 0
            chunks = 0
            while generated < target_samples:
                rollout = self._rollout_policy(
                    chunk_steps,
                    beta=1.0,
                    teacher=self.teacher,
                    collect_expert_labels=True,
                )
                self.expert_bank.extend(rollout.observations, rollout.actions)
                generated += int(len(rollout.actions))
                chunks += 1

            logs["bc/generated_samples"] = generated
            logs["bc/generated_chunks"] = chunks
            logs["bc/generated_target_samples"] = target_samples

        observations, actions = self._stack_bank(self.expert_bank)
        batch_size = self.config["il"]["bc_batch_size"]
        epochs = self.config["il"]["bc_epochs"]

        self.policy.train()
        for _ in range(epochs):
            permutation = torch.randperm(observations.shape[0])
            for start in range(0, observations.shape[0], batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch_obs = observations[batch_indices].to(self.device)
                batch_actions = actions[batch_indices].to(self.device)
                self.policy_optimizer.zero_grad(set_to_none=True)
                loss, entropy = self._policy_batch_loss(batch_obs, batch_actions)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["train"]["max_grad_norm"])
                self.policy_optimizer.step()
                logs["bc/loss"] = float(loss.detach().cpu())
                logs["bc/entropy"] = float(entropy.detach().cpu())

        self.epoch += 1
        return logs

    def train_dagger(self):
        if self.teacher is None:
            raise pufferlib.APIUsageError("DAgger requires a teacher policy or checkpoint")

        logs = {}
        total_iters = int(self.config["il"]["dagger_iters"])
        total_steps_per_iter = int(self.config["il"]["dagger_rollout_steps"])
        
        chunk_size = int(self.config["il"].get("bc_rollout_steps", total_steps_per_iter))

        for iteration in range(total_iters):
            beta = dagger_beta(
                iteration,
                total_iters,
                self.config["il"]["dagger_beta_start"],
                self.config["il"]["dagger_beta_end"],
            )

            generated = 0
            chunks = 0
            while generated < total_steps_per_iter:
                steps = min(chunk_size, total_steps_per_iter - generated)
                rollout = self._rollout_policy(
                    steps,
                    beta=beta,
                    teacher=self.teacher,
                    collect_expert_labels=True,
                )
                self.expert_bank.extend(rollout.observations, rollout.actions)
                generated += int(len(rollout.actions))
                chunks += 1

            logs["dagger/beta"] = beta
            logs["dagger/collected"] = generated
            logs["dagger/chunks"] = chunks
            logs.update(self.train_bc())

        return logs

    def _discriminator_loss(self, expert_obs, expert_actions, policy_obs, policy_actions):
        expert_logits = self.discriminator(expert_obs, expert_actions)
        policy_logits = self.discriminator(policy_obs, policy_actions)
        expert_targets = torch.ones_like(expert_logits)
        policy_targets = torch.zeros_like(policy_logits)
        loss = F.binary_cross_entropy_with_logits(expert_logits, expert_targets)
        loss = loss + F.binary_cross_entropy_with_logits(policy_logits, policy_targets)
        return loss, expert_logits, policy_logits

    def _gail_rewards(self, observations, actions):
        logits = self.discriminator(observations, actions)
        return F.softplus(logits) * self.config["il"]["gail_reward_scale"]

    def train_gail(self):
        if len(self.expert_bank) == 0:
            raise pufferlib.APIUsageError("GAIL requires expert_data")

        logs = {}
        for _ in range(self.config["il"]["gail_updates"]):
            rollout = self._rollout_policy(self.config["il"]["gail_rollout_steps"])
            self.policy_bank.extend(rollout.observations, rollout.actions)

            expert_obs, expert_actions = self.expert_bank.sample(min(self.config["il"]["bc_batch_size"], len(self.expert_bank)))
            policy_obs, policy_actions = self.policy_bank.sample(min(self.config["il"]["bc_batch_size"], len(self.policy_bank)))
            expert_obs = expert_obs.to(self.device)
            expert_actions = expert_actions.to(self.device)
            policy_obs = policy_obs.to(self.device)
            policy_actions = policy_actions.to(self.device)

            self.discriminator_optimizer.zero_grad(set_to_none=True)
            disc_loss, expert_logits, policy_logits = self._discriminator_loss(
                expert_obs, expert_actions, policy_obs, policy_actions
            )
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
            self.discriminator_optimizer.step()

            with torch.no_grad():
                rewards = self._gail_rewards(rollout.observations.to(self.device), rollout.actions.to(self.device))
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            self.policy.train()
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_logits, _ = self._forward_policy(self.policy, rollout.observations.to(self.device), None)
            _, logprob, entropy = pufferlib.pytorch.sample_logits(policy_logits, action=rollout.actions.to(self.device))
            loss = -(logprob * advantages.detach()).mean()
            loss = loss - self.config["train"].get("ent_coef", 0.0) * entropy.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["train"]["max_grad_norm"])
            self.policy_optimizer.step()

            logs = {
                "gail/discriminator_loss": float(disc_loss.detach().cpu()),
                "gail/policy_loss": float(loss.detach().cpu()),
                "gail/expert_logit": float(expert_logits.mean().detach().cpu()),
                "gail/policy_logit": float(policy_logits.mean().detach().cpu()),
                "gail/reward": float(rewards.mean().detach().cpu()),
            }

        self.epoch += 1
        return logs

    def save_checkpoint(self):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return None

        run_dir = os.path.join(self.data_dir, f"{self.env_name}_{self.run_id}")
        os.makedirs(run_dir, exist_ok=True)
        model_name = f"model_{self.env_name}_{self.epoch:06d}.pt"
        model_path = os.path.join(run_dir, model_name)
        if not os.path.exists(model_path):
            torch.save(self.policy.state_dict(), model_path)

        state = {
            "optimizer_state_dict": self.policy_optimizer.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "global_step": self.global_step,
            "update": self.epoch,
            "model_name": model_name,
            "run_id": self.run_id,
        }
        state_path = os.path.join(run_dir, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
        return model_path

    def close(self):
        self.vecenv.close()
        save_expert_path = self.config["il"].get("save_expert_data")
        if save_expert_path:
            save_transition_bank(self.expert_bank, save_expert_path)

        model_path = self.save_checkpoint()
        final_path = os.path.join(self.data_dir, f"{self.env_name}_{self.run_id}.pt")
        if model_path is not None:
            shutil.copy2(model_path, final_path)
        return final_path

    def train(self, updates=1):
        if updates <= 0:
            raise pufferlib.APIUsageError("--il.updates must be > 0")

        checkpoint_interval = int(self.config["train"].get("checkpoint_interval", 0))
        last_logs = {}
        for update in range(updates):
            if self.method == "bc":
                logs = self.train_bc()
            elif self.method == "dagger":
                logs = self.train_dagger()
            elif self.method in {"gail", "airl"}:
                logs = self.train_gail()
            else:
                raise pufferlib.APIUsageError(f"Unknown imitation method: {self.method}")

            logs["update"] = self.epoch
            logs["update_idx"] = update + 1
            logs["updates_total"] = updates
            logs["global_step"] = self.global_step

            if checkpoint_interval > 0 and self.epoch % checkpoint_interval == 0:
                checkpoint_path = self.save_checkpoint()
                if checkpoint_path is not None:
                    logs["checkpoint/model_path"] = checkpoint_path

            if self.logger is not None:
                self.log_step += 1
                self.logger.log({f"il/{k}": v for k, v in logs.items()}, self.log_step)

            last_logs = logs

        return last_logs


def train(env_name, args=None, vecenv=None, policy=None, logger=None):
    args = args or load_config(env_name)
    vecenv = vecenv or pufferl.load_env(env_name, args)
    policy = policy or pufferl.load_policy(args, vecenv, env_name)

    teacher = None
    if args["il"]["teacher_load_model_path"] is not None or args["il"]["teacher_load_id"] is not None:
        teacher_args = deepcopy(args)
        teacher_args["load_model_path"] = None
        teacher_args["load_id"] = None
        teacher = pufferl.load_policy(teacher_args, vecenv, env_name)
        _load_teacher_checkpoint(teacher, args, env_name)

    if args["neptune"]:
        logger = pufferl.NeptuneLogger(args)
    elif args["wandb"]:
        logger = pufferl.WandbLogger(args)

    trainer = ImitationTrainer(args, vecenv, policy, teacher=teacher, logger=logger)
    logs = trainer.train(updates=args["il"]["updates"])
    model_path = trainer.close()
    if trainer.logger is not None:
        trainer.logger.close(model_path)

    return logs


def _normalize_teacher_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, dict):
        raise pufferlib.APIUsageError("Unsupported teacher checkpoint format")

    return {k.replace("module.", ""): v for k, v in checkpoint.items()}


def _load_teacher_state_dict_into_policy(policy, checkpoint, source_name):
    state_dict = _normalize_teacher_state_dict(checkpoint)
    model_keys = set(policy.state_dict().keys())

    # Artifacts often save policy weights under a `policy.` prefix.
    stripped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("policy."):
            stripped_state_dict[key[7:]] = value
        else:
            stripped_state_dict[key] = value

    direct_matches = sum(1 for key in state_dict if key in model_keys)
    stripped_matches = sum(1 for key in stripped_state_dict if key in model_keys)
    if stripped_matches > direct_matches:
        state_dict = stripped_state_dict

    matched_keys = sum(1 for key in state_dict if key in model_keys)
    if matched_keys == 0:
        raise pufferlib.APIUsageError(
            f"Teacher checkpoint from {source_name} does not match current policy architecture"
        )

    incompat = policy.load_state_dict(state_dict, strict=False)
    if incompat.missing_keys:
        print(
            f"[puffer_il] Warning: missing {len(incompat.missing_keys)} teacher keys from {source_name}. "
            "Continuing with partial teacher initialization."
        )
    if incompat.unexpected_keys:
        print(f"[puffer_il] Warning: ignored {len(incompat.unexpected_keys)} unexpected teacher keys from {source_name}.")


def _resolve_teacher_checkpoint_paths(args, env_name):
    paths = []

    teacher_load_id = args["il"]["teacher_load_id"]
    if teacher_load_id is not None:
        if args["neptune"]:
            path = pufferl.NeptuneLogger(args, teacher_load_id, mode="read-only").download()
        elif args["wandb"]:
            path = pufferl.WandbLogger(args, teacher_load_id).download()
        else:
            raise pufferlib.APIUsageError("No tracker configured to resolve --il.teacher-load-id")
        paths.append(path)

    teacher_load_path = args["il"]["teacher_load_model_path"]
    if teacher_load_path == "latest":
        teacher_load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if teacher_load_path is not None:
        paths.append(teacher_load_path)

    return paths


def _load_teacher_checkpoint(teacher, args, env_name):
    device = args["train"]["device"]
    checkpoint_paths = _resolve_teacher_checkpoint_paths(args, env_name)
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        _load_teacher_state_dict_into_policy(teacher, checkpoint, checkpoint_path)


def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
    return pufferl.eval(env_name=env_name, args=args, vecenv=vecenv, policy=policy)


def sweep(args=None, env_name=None):
    """Run parameter sweep using wandb or neptune for hyperparameter optimization."""
    import pufferlib.sweep
    
    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"]:
        raise pufferlib.APIUsageError("Sweeps require either wandb or neptune")

    method = args["sweep"].pop("method")
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f"Invalid sweep method {method}. See pufferlib.sweep")

    sweep_obj = sweep_cls(args["sweep"])
    points_per_run = args["sweep"]["downsample"]
    # For IL, use method-specific loss metric as target (e.g., "il/bc/loss" for BC)
    method_name = args["il"]["method"].lower()
    target_key = f"il/{method_name}/loss"
    
    for i in range(args["max_runs"]):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sweep_obj.suggest(args)
        
        # Run IL training
        all_logs = []
        vecenv = pufferl.load_env(env_name, args)
        policy = pufferl.load_policy(args, vecenv, env_name)

        teacher = None
        if args["il"]["teacher_load_model_path"] is not None or args["il"]["teacher_load_id"] is not None:
            teacher_args = deepcopy(args)
            teacher_args["load_model_path"] = None
            teacher_args["load_id"] = None
            teacher = pufferl.load_policy(teacher_args, vecenv, env_name)
            _load_teacher_checkpoint(teacher, args, env_name)

        if args["neptune"]:
            logger = pufferl.NeptuneLogger(args)
        elif args["wandb"]:
            logger = pufferl.WandbLogger(args)
        else:
            logger = None

        trainer = ImitationTrainer(args, vecenv, policy, teacher=teacher, logger=logger)
        logs = trainer.train(updates=args["il"]["updates"])
        all_logs.append(logs)
        model_path = trainer.close()
        if trainer.logger is not None:
            trainer.logger.close(model_path)

        # Extract loss metric for sweep optimization
        if all_logs and target_key in all_logs[0]:
            loss_value = all_logs[0][target_key]
            # Observe: for IL, we typically want to minimize loss, so use negative as score
            sweep_obj.observe(args, -loss_value, all_logs[0].get("global_step", 1))
        else:
            print(f"Warning: Could not find {target_key} in logs for sweep observation")


def load_config(env_name, config_dir=None):
    parser = argparse.ArgumentParser(
        description=f":blowfish: PufferLib IL [bright_cyan]{pufferlib.__version__}[/] demo options",
        formatter_class=RichHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to a pretrained checkpoint")
    parser.add_argument("--load-id", type=str, default=None, help="Kickstart/eval from a finished run")
    parser.add_argument(
        "--render-mode", type=str, default="auto", choices=["auto", "human", "ansi", "rgb_array", "raylib", "None"]
    )
    parser.add_argument("--save-frames", type=int, default=0)
    parser.add_argument("--gif-path", type=str, default="eval.gif")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-runs", type=int, default=200)
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--wandb-group", type=str, default="debug")
    parser.add_argument("--neptune", action="store_true", help="Use neptune for logging")
    parser.add_argument("--neptune-name", type=str, default="pufferai")
    parser.add_argument("--neptune-project", type=str, default="ablations")
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--sanity-maps", nargs="*", default=None)

    parser.add_argument("--il.method", type=str, default=IL_DEFAULTS["method"], choices=["bc", "dagger", "gail", "airl"])
    parser.add_argument("--il.updates", type=int, default=IL_DEFAULTS["updates"])
    parser.add_argument("--il.expert-data", nargs="*", default=None)
    parser.add_argument("--il.save-expert-data", type=str, default=IL_DEFAULTS["save_expert_data"])
    parser.add_argument("--il.teacher-load-model-path", type=str, default=IL_DEFAULTS["teacher_load_model_path"])
    parser.add_argument("--il.teacher-load-id", type=str, default=IL_DEFAULTS["teacher_load_id"])
    parser.add_argument("--il.bc-batch-size", type=int, default=IL_DEFAULTS["bc_batch_size"])
    parser.add_argument("--il.bc-epochs", type=int, default=IL_DEFAULTS["bc_epochs"])
    parser.add_argument("--il.bc-rollout-steps", type=int, default=IL_DEFAULTS["bc_rollout_steps"])
    parser.add_argument(
        "--il.bc-teacher-target-samples", type=int, default=IL_DEFAULTS["bc_teacher_target_samples"]
    )
    parser.add_argument("--il.dagger-iters", type=int, default=IL_DEFAULTS["dagger_iters"])
    parser.add_argument("--il.dagger-rollout-steps", type=int, default=IL_DEFAULTS["dagger_rollout_steps"])
    parser.add_argument("--il.dagger-beta-start", type=float, default=IL_DEFAULTS["dagger_beta_start"])
    parser.add_argument("--il.dagger-beta-end", type=float, default=IL_DEFAULTS["dagger_beta_end"])
    parser.add_argument("--il.gail-updates", type=int, default=IL_DEFAULTS["gail_updates"])
    parser.add_argument("--il.gail-rollout-steps", type=int, default=IL_DEFAULTS["gail_rollout_steps"])
    parser.add_argument(
        "--il.gail-discriminator-hidden-size", type=int, default=IL_DEFAULTS["gail_discriminator_hidden_size"]
    )
    parser.add_argument("--il.gail-discriminator-lr", type=float, default=IL_DEFAULTS["gail_discriminator_lr"])
    parser.add_argument("--il.gail-reward-scale", type=float, default=IL_DEFAULTS["gail_reward_scale"])

    if config_dir is None:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
    else:
        puffer_dir = config_dir

    puffer_config_dir = os.path.join(puffer_dir, "config/**/*.ini")
    puffer_default_config = os.path.join(puffer_dir, "config/default.ini")
    if env_name == "default":
        config = configparser.ConfigParser()
        config.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            config = configparser.ConfigParser()
            config.read([puffer_default_config, path])
            if env_name in config["base"]["env_name"].split():
                break
        else:
            raise pufferlib.APIUsageError(f"No config for env_name {env_name}")

    def puffer_type(value):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value

    for section in config.sections():
        for key in config[section]:
            fmt = f"--{key}" if section == "base" else f"--{section}.{key}"
            parser.add_argument(fmt.replace("_", "-"), default=puffer_type(config[section][key]), type=puffer_type)

    parser.add_argument("-h", "--help", default=argparse.SUPPRESS, action="help", help="Show this help message and exit")

    parsed = vars(parser.parse_args())
    args = _nest_dict(parsed)

    args.setdefault("il", {})
    for key, value in IL_DEFAULTS.items():
        args["il"].setdefault(key, value)

    # Set up sweep defaults if not provided
    args.setdefault("sweep", {})
    if "method" not in args["sweep"]:
        args["sweep"]["method"] = "Protein"
    if "metric" not in args["sweep"]:
        args["sweep"]["metric"] = "loss"
    if "downsample" not in args["sweep"]:
        args["sweep"]["downsample"] = 10

    data_dir = args["train"].get("data_dir", args.get("data_dir", "experiments"))
    args["train"]["data_dir"] = data_dir
    args["data_dir"] = data_dir

    args["train"]["use_rnn"] = args["rnn_name"] is not None
    return args


def _nest_dict(parsed):
    args = {}
    for key, value in parsed.items():
        next_dict = args
        parts = key.split(".")
        for subkey in parts[:-1]:
            next_dict = next_dict.setdefault(subkey, {})
        next_dict[parts[-1]] = value
    return args


def load_transition_bank(paths):
    bank = TransitionBank()
    if paths is None:
        return bank

    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    for path in paths:
        path = Path(path)
        if path.is_dir():
            candidates = sorted([*path.glob("*.npz"), *path.glob("*.pt"), *path.glob("*.pth"), *path.glob("*.npy")])
        else:
            candidates = [path]

        for candidate in candidates:
            observations, actions = load_transition_file(candidate)
            bank.extend(observations, actions)

    return bank


def save_transition_bank(bank, path):
    observations, actions = bank.tensors()
    if observations is None or actions is None:
        raise pufferlib.APIUsageError("No expert data available to save")

    path = Path(path)
    if path.suffix == "":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "expert_data.npz"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".npz":
        np.savez(path, observations=observations.numpy(), actions=actions.numpy())
    elif path.suffix == ".npy":
        payload = np.array([observations.numpy(), actions.numpy()], dtype=object)
        np.save(path, payload, allow_pickle=True)
    elif path.suffix in {".pt", ".pth"}:
        torch.save({"observations": observations, "actions": actions}, path)
    else:
        raise pufferlib.APIUsageError(
            f"Unsupported expert data save file: {path}. Use .npz, .npy, .pt, or .pth"
        )


def load_transition_file(path):
    path = Path(path)
    if not path.exists():
        raise pufferlib.APIUsageError(f"Expert data file not found: {path}")

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        observations = data.get("observations", data.get("obs", data.get("states")))
        actions = data.get("actions", data.get("expert_actions", data.get("labels")))
    elif path.suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            observations = data.get("observations", data.get("obs", data.get("states")))
            actions = data.get("actions", data.get("expert_actions", data.get("labels")))
        else:
            raise pufferlib.APIUsageError(f"Unsupported expert data format in {path}")
    elif path.suffix == ".npy":
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object and len(data) == 2:
            observations, actions = data.tolist()
        else:
            raise pufferlib.APIUsageError(f"Unsupported .npy expert data format in {path}")
    else:
        raise pufferlib.APIUsageError(f"Unsupported expert data file: {path}")

    if observations is None or actions is None:
        raise pufferlib.APIUsageError(f"Expert data file {path} must contain observations and actions")

    return observations, actions


def args_use_rnn(config):
    return bool(config["train"].get("use_rnn", False))


def normalize_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def dagger_beta(iteration, total_iters, beta_start, beta_end):
    if total_iters <= 1:
        return beta_end

    progress = iteration / max(1, total_iters - 1)
    return beta_start + progress * (beta_end - beta_start)


def discounted_returns(rewards, dones, gamma):
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], device=rewards.device)
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[index] + gamma * running * (~dones[index]).float()
        returns[index] = running

    return returns


def action_feature_dim(action_space):
    if isinstance(action_space, pufferlib.spaces.Discrete):
        return action_space.n
    if isinstance(action_space, pufferlib.spaces.MultiDiscrete):
        return int(np.sum(action_space.nvec))
    if isinstance(action_space, pufferlib.spaces.Box):
        return int(np.prod(action_space.shape))
    raise pufferlib.APIUsageError(f"Unsupported action space for imitation learning: {action_space}")


def encode_actions(actions, action_space):
    actions = actions.detach()

    if isinstance(action_space, pufferlib.spaces.Discrete):
        if actions.ndim > 1:
            actions = actions.squeeze(-1)
        return F.one_hot(actions.long(), num_classes=action_space.n).float()

    if isinstance(action_space, pufferlib.spaces.MultiDiscrete):
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        columns = []
        for index, n in enumerate(action_space.nvec):
            columns.append(F.one_hot(actions[:, index].long(), num_classes=int(n)).float())
        return torch.cat(columns, dim=-1)

    if isinstance(action_space, pufferlib.spaces.Box):
        return actions.float().reshape(actions.shape[0], -1)

    raise pufferlib.APIUsageError(f"Unsupported action space for imitation learning: {action_space}")


def main():
    err = "Usage: puffer_il [train, eval, sweep] [env_name] [optional args]. --help for more info"
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    elif mode == "eval":
        eval(env_name=env_name)
    elif mode == "sweep":
        sweep(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)


if __name__ == "__main__":
    main()