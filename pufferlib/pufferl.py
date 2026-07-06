## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import copy
import numbers
import warnings

import pandas as pd


warnings.filterwarnings("error", category=RuntimeWarning)

import os
import sys
import traceback
import glob
import ast
import time
import random
import shutil
import subprocess
import argparse
import importlib
import configparser
from datetime import datetime
from threading import Thread
from collections import defaultdict, deque
import yaml

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record

import pufferlib
import pufferlib.sweep
import pufferlib.utils
import pufferlib.vector
import pufferlib.pytorch
import pufferlib.viz


try:
    from pufferlib import _C
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation"
    )

import rich
import rich.traceback
from rich.table import Table
from rich.console import Console
from rich_argparse import RichHelpFormatter
from tqdm import tqdm

rich.traceback.install(show_locals=False)

import signal  # Aggressively exit on ctrl+c

signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))

from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME  # noqa: E402

# Assume advantage kernel has been built if torch has been compiled with CUDA or HIP support
# and can find CUDA or HIP in the system
ADVANTAGE_CUDA = bool(CUDA_HOME or ROCM_HOME)
HIDDEN_DASHBOARD_METRICS = {
    "comfort_score",
    "driving_direction_score",
    "making_progress_rate",
    "multi_lane_score",
    "multi_lane_time",
    "speed_limit_compliance",
}


def torch_device(device):
    if isinstance(device, int):
        return torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")
    return device


def is_cuda_device(device):
    if isinstance(device, int):
        return torch.cuda.is_available()
    device = torch.device(device)
    return device.type == "cuda"


def base_policy(policy):
    return policy.module if hasattr(policy, "module") else policy


def clean_state_key(key):
    prefixes = ("module.", "_orig_mod.")
    while key.startswith(prefixes):
        key = key.split(".", 1)[1]
    return key


def clean_policy_state_dict(state_dict):
    return {clean_state_key(k): v for k, v in state_dict.items()}


def logits_to_float(logits):
    if isinstance(logits, torch.distributions.Normal):
        return torch.distributions.Normal(logits.loc.float(), logits.scale.float())
    if isinstance(logits, torch.Tensor):
        return logits.float()
    return tuple(l.float() for l in logits)


class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = config["torch_deterministic"]
        torch.backends.cudnn.benchmark = not config["torch_deterministic"]
        torch.use_deterministic_algorithms(config["torch_deterministic"], warn_only=True)

        # Reproducibility
        seed = config["seed"]

        # Vecenv info
        vecenv.async_reset(seed)
        obs_space = vecenv.single_observation_space
        atn_space = vecenv.single_action_space
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        # Experience
        if config["batch_size"] == "auto" and config["bptt_horizon"] == "auto":
            raise pufferlib.APIUsageError("Must specify batch_size or bptt_horizon")
        elif config["batch_size"] == "auto":
            config["batch_size"] = total_agents * config["bptt_horizon"]
        elif config["bptt_horizon"] == "auto":
            config["bptt_horizon"] = config["batch_size"] // total_agents

        batch_size = config["batch_size"]
        horizon = config["bptt_horizon"]
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(f"Total agents {total_agents} <= segments {segments}")

        device = config["device"]
        precision = config["precision"]
        if precision not in ("float32", "bfloat16"):
            raise pufferlib.APIUsageError(f"Invalid precision: {precision}: use float32 or bfloat16")
        use_cuda = is_cuda_device(device)
        if precision == "bfloat16" and use_cuda and not torch.cuda.is_bf16_supported():
            raise pufferlib.APIUsageError("bfloat16 precision requires a CUDA device with bf16 support")
        if precision == "bfloat16" and not config.get("amp", True):
            raise pufferlib.APIUsageError("bfloat16 precision requires train.amp=True")

        obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype]

        self.observations = torch.zeros(
            segments,
            horizon,
            *obs_space.shape,
            dtype=obs_dtype,
            pin_memory=device == "cuda" and config["cpu_offload"],
            device="cpu" if config["cpu_offload"] else device,
        )
        self.actions = torch.zeros(
            segments,
            horizon,
            *atn_space.shape,
            device=device,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[atn_space.dtype],
        )
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.importance = torch.ones(segments, horizon, device=device)
        self.masks = torch.zeros(segments, horizon, device=device, dtype=torch.bool)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents
        self.render = config["render"]
        self.render_interval = config["render_interval"]

        if self.render:
            ensure_drive_binary()

        # LSTM
        if config["use_rnn"]:
            n = vecenv.agents_per_batch
            h = policy.hidden_size
            self.lstm_h = {i * n: torch.zeros(n, h, device=device) for i in range(total_agents // n)}
            self.lstm_c = {i * n: torch.zeros(n, h, device=device) for i in range(total_agents // n)}

        # Minibatching & gradient accumulation
        minibatch_size = config["minibatch_size"]
        max_minibatch_size = config["max_minibatch_size"]
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        if minibatch_size > max_minibatch_size and minibatch_size % max_minibatch_size != 0:
            raise pufferlib.APIUsageError(
                f"minibatch_size {minibatch_size} > max_minibatch_size {max_minibatch_size} must divide evenly"
            )

        if batch_size < minibatch_size:
            raise pufferlib.APIUsageError(f"batch_size {batch_size} must be >= minibatch_size {minibatch_size}")

        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)
        self.total_minibatches = int(config["update_epochs"] * batch_size / self.minibatch_size)
        self.minibatch_segments = self.minibatch_size // horizon
        if self.minibatch_segments * horizon != self.minibatch_size:
            raise pufferlib.APIUsageError(
                f"minibatch_size {self.minibatch_size} must be divisible by bptt_horizon {horizon}"
            )

        # Torch compile
        self.uncompiled_policy = base_policy(policy)
        self.policy = policy
        if config["compile"]:
            compile_kwargs = {
                "mode": config["compile_mode"],
                "fullgraph": config["compile_fullgraph"],
            }
            self.policy = torch.compile(policy, **compile_kwargs)
            self.policy.forward_eval = torch.compile(self.uncompiled_policy.forward_eval, **compile_kwargs)
            pufferlib.pytorch.sample_logits = torch.compile(pufferlib.pytorch.sample_logits, **compile_kwargs)

        # Optimizer
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
            )
        elif config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
            )
        elif config["optimizer"] == "muon":
            import heavyball
            from heavyball import ForeachMuon

            warnings.filterwarnings(action="ignore", category=UserWarning, module=r"heavyball.*")
            heavyball.utils.compile_mode = "default"

            # # optionally a little bit better/faster alternative to newtonschulz iteration
            # import heavyball.utils
            # heavyball.utils.zeroth_power_mode = 'thinky_polar_express'

            # heavyball_momentum=True introduced in heavyball 2.1.1
            # recovers heavyball-1.7.2 behaviour - previously swept hyperparameters work well
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
                heavyball_momentum=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")

        self.optimizer = optimizer

        # Logging
        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        # Learning rate scheduler
        epochs = config["total_timesteps"] // config["batch_size"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.total_epochs = epochs

        # Automatic mixed precision
        self.amp_context = contextlib.nullcontext()
        if config.get("amp", True) and use_cuda and precision == "bfloat16":
            self.amp_context = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, precision))

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self.agent_steps = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = defaultdict(list)
        self.last_stats = defaultdict(list)
        self.losses = {}
        self.best_score = -float("inf")
        self.ema_max = 0.0
        # Set later via PuffeRL.attach_eval_manager (before evaluate() fires).
        self._eval_manager = None

        # Dashboard
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    def load_training_state(self, path):
        device = torch_device(self.config["device"])
        state = torch.load(path, map_location=device, weights_only=False)
        policy_state = state.get("policy_state_dict")
        if policy_state is None:
            model_name = state["model_name"]
            model_path = os.path.join(os.path.dirname(path), "models", model_name)
            policy_state = torch.load(model_path, map_location=device)

        policy_state = clean_policy_state_dict(policy_state)
        self.uncompiled_policy.load_state_dict(policy_state)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        self.epoch = state.get("epoch", state.get("update", 0))
        self.global_step = state.get("global_step", state.get("agent_step", 0))
        self.last_log_step = self.global_step
        self.best_score = state.get("best_score", self.best_score)
        self.ema_max = state.get("ema_max", self.ema_max)
        restore_rng_state(state)
        print(f"Resumed training from {path}: epoch={self.epoch}, global_step={self.global_step}")

    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def evaluate(self):
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)

        config = self.config
        device = config["device"]

        if config["use_rnn"]:
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            self.global_step += env_id.stop - env_id.start

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)  # , non_blocking=True)
            r = torch.as_tensor(r).to(device)  # , non_blocking=True)
            d = torch.as_tensor(d).to(device)  # , non_blocking=True)
            t = torch.as_tensor(t).to(device)  # , non_blocking=True)
            done_mask = (d + t).clamp(max=1.0)
            m = torch.as_tensor(mask).to(device)  # , non_blocking=True)

            # Obs distribution stats (max/min/mean across the batch and obs
            # dims, appended per env step). Surfaces clipping / unbounded
            # features / normalization regressions in wandb.
            self.stats["obs/max"].append(o_device.max().item())
            self.stats["obs/min"].append(o_device.min().item())
            self.stats["obs/mean"].append(o_device.mean().item())

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=done_mask,
                    env_id=env_id,
                    mask=mask,
                )

                if config["use_rnn"]:
                    state["lstm_h"] = self.lstm_h[env_id.start]
                    state["lstm_c"] = self.lstm_c[env_id.start]

                logits, value = self.policy.forward_eval(o_device, state)
                logits = logits_to_float(logits)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                if config["normalize_rewards"]:
                    r = torch.sign(r) * torch.log1p(torch.abs(r))

            profile("eval_copy", epoch)
            with torch.no_grad():
                if config["use_rnn"]:
                    self.lstm_h[env_id.start] = state["lstm_h"]
                    self.lstm_c[env_id.start] = state["lstm_c"]

                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1 + self.ep_indices[env_id.stop - 1].item())

                if config["cpu_offload"]:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob.float()
                # Truncation bootstrap hack for auto-reset envs.
                # Ideally we add `gamma * V(s_{t+1})` on truncation steps, but Drive resets in C so
                # the value at index `l` is post-reset. We use `values[..., l-1]` as a heuristic
                # proxy for the pre-reset terminal value (bootstrap term is not clipped).
                if l > 0:
                    trunc_mask = (t > 0) & (d == 0)
                    r = r + trunc_mask.to(r.dtype) * config["gamma"] * self.values[batch_rows, l - 1]
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = done_mask.float()
                self.truncations[batch_rows, l] = t.float()
                self.values[batch_rows, l] = value.flatten().float()
                self.masks[batch_rows, l] = m

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l + 1 >= config["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config["device"]).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile("eval_misc", epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)

            profile("env", epoch)
            self.vecenv.send(action)

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return self.stats

    @record
    def train(self):
        profile = self.profile
        epoch = self.epoch
        config = self.config
        profile("train", epoch)
        profile("train_misc", epoch, nest=True)
        losses = defaultdict(float)
        if config["use_rnn"]:
            self._train_ppo_trajectory(losses, profile, epoch)
        else:
            self._train_ppo_transition(losses, profile, epoch)

        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config["total_timesteps"]
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            self.losses = losses
            logs = self.mean_and_log()
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

            if self.render and self.epoch % self.render_interval == 0:
                model_dir = os.path.join(self.config["data_dir"], f"{self.config['env']}_{self.logger.run_id}")
                model_files = glob.glob(os.path.join(model_dir, "models", "model_*.pt"))

                if model_files:
                    # Take the latest checkpoint
                    latest_cpt = max(model_files, key=os.path.getctime)
                    bin_path = f"{model_dir}.bin"

                    # Export to .bin for rendering with raylib
                    try:
                        export_args = {"env_name": self.config["env"], "load_model_path": latest_cpt, **self.config}

                        export(
                            args=export_args,
                            env_name=self.config["env"],
                            vecenv=self.vecenv,
                            policy=self.uncompiled_policy,
                            path=bin_path,
                            silent=True,
                        )
                        pufferlib.utils.render_videos(
                            self.config, self.vecenv, self.logger, self.epoch, self.global_step, bin_path
                        )

                    except Exception as e:
                        print(f"Failed to export model weights: {e}")

        # All evaluation is now driven by the unified EvalManager. Each
        # [eval.<name>] section in drive.ini is one evaluator instance;
        # the manager fires any whose interval divides this epoch. See
        # docs/eval_unification.md for the design.
        # Under DDP, only rank 0 runs eval — every rank has identical
        # weights so duplicating the rollout wastes memory + compute,
        # and parallel mp4 writes from N ranks race on filenames. Other
        # ranks block on the next allreduce until rank 0 rejoins.
        is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
        if self._eval_manager is not None and is_rank0:
            # Subprocess evals load the policy from disk. Save the latest
            # checkpoint first so they see this epoch's weights, not the
            # last save_checkpoint() from `checkpoint_interval`.
            if self._eval_manager.has_subprocess_evals_at(self.epoch):
                self.save_checkpoint()
            self._eval_manager.maybe_run(
                epoch=self.epoch,
                policy=self.uncompiled_policy,
                env_name=self.config["env"],
                logger=self.logger,
                global_step=self.agent_steps,
            )

        return logs

    def _ppo_loss(self, mb_obs, mb_actions, mb_logprobs, mb_values, mb_returns, mb_adv, adv_weights=None):
        config = self.config
        state = dict(action=mb_actions, lstm_h=None, lstm_c=None)
        with self.amp_context:
            logits, newvalue = self.policy(mb_obs, state)
        logits = logits_to_float(logits)
        newvalue = newvalue.float()
        _, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

        newlogprob = newlogprob.float().view_as(mb_logprobs)
        newvalue = newvalue.view_as(mb_returns)
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
        if adv_weights is not None:
            mb_adv = adv_weights * mb_adv

        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        if config["vf_clip_coef"] is not None:
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -config["vf_clip_coef"], config["vf_clip_coef"])
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * (newvalue - mb_returns) ** 2
            v_loss = v_loss.mean()
        entropy_loss = entropy.mean()
        loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss

        return (
            loss,
            newvalue,
            ratio,
            {
                "policy_loss": pg_loss.item(),
                "value_loss": v_loss.item(),
                "entropy": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": clipfrac.item(),
            },
        )

    def _compute_advantages(self, ratio, rho_clip, c_clip):
        config = self.config
        device = config["device"]

        masks = self.masks.bool()
        terminals = torch.maximum(self.terminals, (~masks).float())
        advantages = compute_puff_advantage(
            self.values,
            self.rewards,
            terminals,
            ratio,
            torch.zeros_like(self.values, device=device),
            config["gamma"],
            config["gae_lambda"],
            rho_clip,
            c_clip,
        )
        advantages = advantages.masked_fill(~masks, 0.0)
        return advantages, advantages + self.values, masks

    def _train_ppo_trajectory(self, losses, profile, epoch):
        config = self.config

        b0 = config["adv_sampling_prio_beta0"]
        a = config["adv_sampling_prio_alpha"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1
        self.optimizer.zero_grad()
        for mb in range(self.total_minibatches):
            profile("train_misc", epoch)

            advantages, returns, masks = self._compute_advantages(
                self.ratio,
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta

            profile("train_copy", epoch)
            if config["cpu_offload"]:
                mb_obs = self.observations[idx.cpu()].to(config["device"], non_blocking=True)
            else:
                mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_values = self.values[idx]
            mb_returns = returns[idx]
            mb_adv = advantages[idx]

            profile("train_forward", epoch)
            loss, newvalue, ratio, stats = self._ppo_loss(
                mb_obs,
                mb_actions,
                mb_logprobs,
                mb_values,
                mb_returns,
                mb_adv,
                adv_weights=mb_prio,
            )
            self.ratio[idx] = ratio.detach()

            self.values[idx] = newvalue.detach().float()

            profile("train_misc", epoch)
            for key, value in stats.items():
                losses[key] += value / self.total_minibatches
            losses["importance"] += ratio.mean().item() / self.total_minibatches

            profile("learn", epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
                self.optimizer.step()
                self.optimizer.zero_grad()

        y_pred = self.values.flatten()
        y_true = returns.flatten()
        var_y = y_true.var()
        losses["explained_variance"] = (
            float("nan") if var_y == 0 else (1 - (y_true - y_pred).var(unbiased=False) / var_y).item()
        )

    def _train_ppo_transition(self, losses, profile, epoch):
        config = self.config
        device = config["device"]

        advantages, returns, masks = self._compute_advantages(
            torch.ones_like(self.values, device=device),
            1.0,
            1.0,
        )

        obs_shape = self.vecenv.single_observation_space.shape
        flat_obs = self.observations.reshape(-1, *obs_shape)
        flat_actions = self.actions.reshape(-1, *self.actions.shape[2:])
        flat_logprobs = self.logprobs.reshape(-1)
        flat_values = self.values.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_masks = masks.reshape(-1).bool()
        total_transitions = flat_masks.numel()
        valid_idx = torch.nonzero(flat_masks, as_tuple=False).flatten()
        valid_abs_adv = flat_advantages[valid_idx].abs()

        ewma_beta = config["adv_filter_ewma_beta"]
        current_max = valid_abs_adv.max().item() if valid_abs_adv.numel() > 0 else 0.0
        self.ema_max = current_max if epoch == 0 else ewma_beta * current_max + (1 - ewma_beta) * self.ema_max
        threshold = config["adv_filter_threshold_scale"] * self.ema_max

        keep_mask = valid_abs_adv >= threshold
        keep_idx = valid_idx[keep_mask]
        num_valid, num_kept = valid_idx.numel(), keep_idx.numel()

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            # Synchronize the number of kept transitions in multi-GPU setting to keep synchronization
            kept_tensor = torch.tensor([num_kept], device=device)
            torch.distributed.all_reduce(kept_tensor, op=torch.distributed.ReduceOp.MIN)
            min_num_kept = kept_tensor.item()
            if num_kept > min_num_kept:
                if min_num_kept == 0:
                    keep_idx = keep_idx[:0]
                else:
                    top_idx = torch.topk(valid_abs_adv[keep_mask], min_num_kept, largest=True, sorted=False).indices
                    keep_idx = keep_idx[top_idx]

        kept_fraction = num_kept / max(num_valid, 1)
        losses["filter_threshold"] = threshold
        losses["ema_max"] = self.ema_max
        losses["masked_fraction"] = 1.0 - (valid_idx.numel() / max(total_transitions, 1))
        losses["kept_fraction"] = kept_fraction
        losses["filtered_fraction"] = 1.0 - kept_fraction

        self.optimizer.zero_grad()
        total_minibatches = 0
        pending_minibatches = 0

        for _ in range(config["update_epochs"]):
            permutation = keep_idx[torch.randperm(keep_idx.numel(), device=keep_idx.device)]
            for start in range(0, permutation.numel(), self.minibatch_size):
                profile("train_copy", epoch)
                mb_idx = permutation[start : start + self.minibatch_size]
                if config["cpu_offload"]:
                    mb_obs = flat_obs[mb_idx.cpu()].to(device, non_blocking=True)
                else:
                    mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_logprobs = flat_logprobs[mb_idx]
                mb_values = flat_values[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_adv = flat_advantages[mb_idx]

                profile("train_forward", epoch)
                loss, _, _, stats = self._ppo_loss(
                    mb_obs,
                    mb_actions,
                    mb_logprobs,
                    mb_values,
                    mb_returns,
                    mb_adv,
                )

                profile("train_misc", epoch)
                for key, value in stats.items():
                    losses[key] += value

                profile("learn", epoch)
                loss.backward()
                total_minibatches += 1
                pending_minibatches += 1

                if pending_minibatches >= self.accumulate_minibatches:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pending_minibatches = 0

        if pending_minibatches > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
            self.optimizer.step()
            self.optimizer.zero_grad()

        if total_minibatches > 0:
            for key in ("policy_loss", "value_loss", "entropy", "old_approx_kl", "approx_kl", "clipfrac"):
                losses[key] /= total_minibatches

        y_pred = flat_values[valid_idx]
        y_true = flat_returns[valid_idx]
        var_y = y_true.var(unbiased=False)
        losses["explained_variance"] = (
            float("nan") if var_y == 0 else (1 - (y_true - y_pred).var(unbiased=False) / var_y).item()
        )

    def mean_and_log(self):
        config = self.config
        for k in list(self.stats.keys()):
            v = self.stats[k]
            try:
                v = np.mean(v)
            except:
                del self.stats[k]

            self.stats[k] = v

        device = config["device"]
        agent_steps = int(dist_sum(self.global_step, device))
        self.agent_steps = agent_steps
        logs = {
            "SPS": dist_sum(self.sps, device),
            "agent_steps": agent_steps,
            "uptime": time.time() - self.start_time,
            "epoch": int(dist_sum(self.epoch, device)),  # VB Why it is a sum ?
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{f"environment/{k}": v for k, v in self.stats.items()},
            **{f"losses/{k}": v for k, v in self.losses.items()},
            **{f"performance/{k}": v["elapsed"] for k, v in self.profile},
            # **{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            # **{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            # **{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        run_dir = os.path.join(self.config["data_dir"], f"{self.config['env']}_{run_id}")
        path = os.path.join(run_dir, f"{self.config['env']}_{run_id}.pt")
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        run_id = self.logger.run_id
        path = os.path.join(self.config["data_dir"], f"{self.config['env']}_{run_id}")
        if not os.path.exists(path):
            os.makedirs(path)

        models_dir = os.path.join(path, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_name = f"model_{self.config['env']}_{self.epoch:06d}.pt"
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            torch.save(self.uncompiled_policy.state_dict(), model_path)

        current_score = self.last_stats.get("puffer_score", self.last_stats.get("score", -float("inf")))
        new_best = current_score > self.best_score
        if new_best:
            self.best_score = current_score

        state = {
            "state_version": 2,
            "policy_state_dict": self.uncompiled_policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.global_step,
            "epoch": self.epoch,
            "update": self.epoch,
            "model_name": model_name,
            "run_id": run_id,
            "env": self.config["env"],
            "best_score": self.best_score,
            "ema_max": self.ema_max,
            "rng_state": capture_rng_state(),
        }
        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)

        if new_best:
            best_state_file = os.path.join(path, f"best_models/best_trainer_state_{self.epoch:06d}.pt")
            os.makedirs(os.path.dirname(best_state_file), exist_ok=True)
            shutil.copy(model_path, best_state_file)
            print(f"New best model saved at epoch {self.epoch} with puffer_score {self.best_score:.4f}")

        return model_path

    def print_dashboard(
        self, clear=False, idx=[0], c1="[cyan]", c2="[dim default]", b1="[bright_cyan]", b2="[default]"
    ):
        config = self.config
        sps = dist_sum(self.sps, config["device"])
        agent_steps = dist_sum(self.global_step, config["device"])
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        profile = self.profile
        console = Console()
        dashboard = Table(box=rich.box.ROUNDED, expand=True, show_header=False, border_style="bright_cyan")
        table = Table(box=None, expand=True, show_header=False)
        dashboard.add_row(table)

        table.add_column(justify="left", width=30)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=12)
        table.add_column(justify="center", width=13)
        table.add_column(justify="right", width=13)

        table.add_row(
            f"{b1}PufferLib {b2}3.0 {idx[0] * ' '}:blowfish:",
            f"{c1}CPU: {b2}{np.mean(self.utilization.cpu_util):.1f}{c2}%",
            f"{c1}GPU: {b2}{np.mean(self.utilization.gpu_util):.1f}{c2}%",
            f"{c1}DRAM: {b2}{np.mean(self.utilization.cpu_mem):.1f}{c2}%",
            f"{c1}VRAM: {b2}{np.mean(self.utilization.gpu_mem):.1f}{c2}%",
        )
        idx[0] = (idx[0] - 1) % 10

        s = Table(box=None, expand=True)
        remaining = f"{b2}A hair past a freckle{c2}"
        if sps != 0:
            remaining = duration((config["total_timesteps"] - agent_steps) / sps, b2, c2)

        s.add_column(f"{c1}Summary", justify="left", vertical="top", width=10)
        s.add_column(f"{c1}Value", justify="right", vertical="top", width=14)
        s.add_row(f"{b2}Env", f"{b2}{config['env']}")
        s.add_row(f"{b2}Params", abbreviate(self.model_size, b2, c2))
        s.add_row(f"{b2}Steps", abbreviate(agent_steps, b2, c2))
        s.add_row(f"{b2}SPS", abbreviate(sps, b2, c2))
        s.add_row(f"{b2}Epoch", f"{b2}{self.epoch}")
        s.add_row(f"{b2}Uptime", duration(self.uptime, b2, c2))
        s.add_row(f"{b2}Remaining", remaining)

        delta = profile.eval["buffer"] + profile.train["buffer"]
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf("Evaluate", b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf("  Forward", b2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf("  Env", b2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf("  Copy", b2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", b2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf("Train", b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf("  Forward", b2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf("  Learn", b2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf("  Copy", b2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", b2, delta, profile.train_misc, b2, c2))

        l = Table(
            box=None,
            expand=True,
        )
        l.add_column(f"{c1}Losses", justify="left", width=16)
        l.add_column(f"{c1}Value", justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f"{b2}{metric}", f"{b2}{value:.3f}")

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(s, p, l)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        left.add_column(f"{c1}User Stats", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=10)
        right.add_column(f"{c1}User Stats", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=10)
        i = 0

        if self.stats:
            self.last_stats = self.stats

        for metric, value in (self.stats or self.last_stats).items():
            if metric in HIDDEN_DASHBOARD_METRICS:
                continue

            try:  # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f"{b2}{metric}", f"{b2}{value:.3f}")
            i += 1
            if i == 30:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print("\033[0;0H" + capture.get())


def compute_puff_advantage(
    values, rewards, terminals, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
):
    """CUDA kernel for puffer advantage with automatic CPU fallback. You need
    nvcc (in cuda-dev-tools or in a cuda-dev docker base) for PufferLib to
    compile the fast version."""

    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
        ratio = ratio.cpu()
        advantages = advantages.cpu()

    torch.ops.pufferlib.compute_puff_advantage(
        values, rewards, terminals, ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip
    )

    if not ADVANTAGE_CUDA:
        return advantages.to(device)

    return advantages


def abbreviate(num, b2, c2):
    if num < 1e3:
        return f"{b2}{num}{c2}"
    elif num < 1e6:
        return f"{b2}{num / 1e3:.1f}{c2}K"
    elif num < 1e9:
        return f"{b2}{num / 1e6:.1f}{c2}M"
    elif num < 1e12:
        return f"{b2}{num / 1e9:.1f}{c2}B"
    else:
        return f"{b2}{num / 1e12:.2f}{c2}T"


def duration(seconds, b2, c2):
    if seconds < 0:
        return f"{b2}0{c2}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s" if m else f"{b2}{s}{c2}s"


def fmt_perf(name, color, delta_ref, prof, b2, c2):
    percent = 0 if delta_ref == 0 else int(100 * prof["buffer"] / delta_ref - 1e-5)
    return f"{color}{name}", duration(prof["elapsed"], b2, c2), f"{b2}{percent:2d}{c2}%"


def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()


def dist_mean(value, device):
    if not torch.distributed.is_initialized():
        return value

    return dist_sum(value, device) / torch.distributed.get_world_size()


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    rng_state = state.get("rng_state")
    if not rng_state:
        return

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in rng_state:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])


class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __iter__(self):
        return iter(self.profiles.items())

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        # Skip profiling the first few epochs, which are noisy due to setup
        if (epoch + 1) % self.frequency != 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]["start"] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile["start"]
        profile["delta"] += delta
        # Multiply delta by freq to account for skipped epochs
        profile["elapsed"] += delta * self.frequency

    def end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()
        for i in range(len(self.stack)):
            self.pop(end)

    def clear(self):
        for prof in self.profiles.values():
            if prof["delta"] > 0:
                prof["buffer"] = prof["delta"]
                prof["delta"] = 0


class Utilization(Thread):
    def __init__(self, delay=1, maxlen=20):
        super().__init__()
        self.cpu_mem = deque([0], maxlen=maxlen)
        self.cpu_util = deque([0], maxlen=maxlen)
        self.gpu_util = deque([0], maxlen=maxlen)
        self.gpu_mem = deque([0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            self.cpu_util.append(100 * psutil.cpu_percent() / psutil.cpu_count())
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100 * mem.active / mem.total)
            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                    time.sleep(self.delay)
                    continue

                try:
                    self.gpu_util.append(torch.cuda.utilization())
                    free, total = torch.cuda.mem_get_info()
                    self.gpu_mem.append(100 * (total - free) / total)
                except (ModuleNotFoundError, RuntimeError):
                    self.gpu_util.append(0)
                    self.gpu_mem.append(0)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def downsample(data_list, num_points):
    if not data_list or num_points <= 0:
        return []
    if num_points == 1:
        return [data_list[-1]]
    if len(data_list) <= num_points:
        return data_list

    last = data_list[-1]
    data_list = data_list[:-1]

    data_np = np.array(data_list)
    num_points -= 1  # one down for the last one

    n = (len(data_np) // num_points) * num_points
    data_np = data_np[-n:] if n > 0 else data_np
    downsampled = data_np.reshape(num_points, -1).mean(axis=1)

    return downsampled.tolist() + [last]


class NoLogger:
    def __init__(self, args, run_id=None):
        self.run_id = run_id or str(int(time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path, early_stop):
        pass


class NeptuneLogger:
    def __init__(self, args, load_id=None, mode="async"):
        import neptune as nept

        neptune_name = args["neptune_name"]
        neptune_project = args["neptune_project"]
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags=[args["tag"]] if args["tag"] is not None else [],
        )
        self.run_id = neptune._sys_id
        self.neptune = neptune
        for k, v in pufferlib.unroll_nested_dict(args):
            neptune[k].append(v)
        self.should_upload_model = not args["no_model_upload"]

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path):
        self.neptune["model"].track_files(model_path)

    def close(self, model_path, early_stop):
        self.neptune["early_stop"] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination="artifacts")
        return f"artifacts/{self.run_id}.pt"


class WandbLogger:
    def __init__(self, args, load_id=None, resume="allow"):
        import wandb

        wandb.init(
            id=load_id or wandb.util.generate_id(),
            name=args.get("run_name") or None,
            project=args["wandb_project"],
            group=args["wandb_group"],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args,
            tags=[args["tag"]] if args["tag"] is not None else [],
            settings=wandb.Settings(console="off"),  # stop sending dashboard to wandb
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args["no_model_upload"]

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path, early_stop):
        self.wandb.run.summary["early_stop"] = early_stop
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f"{self.run_id}:latest")
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f"{data_dir}/{model_file}"


class TensorBoardLogger:
    def __init__(self, run_id, experiment_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("TensorBoardLogger requires tensorboard.")

        self.run_id = run_id
        local_log_dir = experiment_dir
        os.makedirs(local_log_dir, exist_ok=True)
        print(f"[TensorBoardLogger] Logging locally to: {local_log_dir}")
        self.local_writer = SummaryWriter(log_dir=local_log_dir)

    def log(self, logs, step):
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.local_writer.add_scalar(key, value, step)

    def close(self, model_path, early_stop):
        self.local_writer.close()


def _get_git_metadata():
    git_metadata = {
        "commit_hash": os.environ.get("GITHUB_SHA") or os.environ.get("COMMIT_SHA"),
    }

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if shutil.which("git") is None:
            return git_metadata

        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        if git_metadata["commit_hash"] is None:
            git_metadata["commit_hash"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return git_metadata


def _save_experiment_config(args, path):
    import yaml
    import json

    experiment_dir = path
    os.makedirs(experiment_dir, exist_ok=True)

    # Save config as yaml
    config_yaml_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_yaml_path, "w") as f:
        # Convert defaultdict to dict for cleaner output
        config = json.loads(json.dumps(args))
        config["git"] = _get_git_metadata()
        yaml.dump(config, f)


def train(env_name, args=None, vecenv=None, policy=None, logger=None, early_stop_fn=None):
    args = args or load_config(env_name)

    # Fine-tuning: reload network, observation configuration from config.yaml and override the args --> only change new reward / new maps / new simulation mode
    if args["load_model_path"]:
        experiment_dir = os.path.dirname(args["load_model_path"])
        config_yaml_path = os.path.join(experiment_dir, "config.yaml")
        KEYS_OF_INTEREST = {
            "action_type",
            "dynamics_model",
            "target_type",
            "num_goals",
            "reward_conditioning",
            "reward_randomization",
            "trajectory_prediction_length",
            "num_trajectory_scaling_factors",
            "trajectory_scaling_factors",
            "obs_slots_boundary_n",
            "obs_slots_lane_n",
            "obs_boundary_stride",
            "obs_lane_stride",
            "obs_dropout_boundary",
            "obs_dropout_lane",
            "obs_slots_partners_n",
            "obs_slots_traffic_controls_n",
            "traffic_control_scope",
        }
        if os.path.exists(config_yaml_path):
            print(f"Found config.yaml at {config_yaml_path}. Merging with defaults...")
            with open(config_yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f)

            # Override Policy and RNN dimensions from model config
            for section in ["policy", "rnn"]:
                if section in yaml_config and isinstance(yaml_config[section], dict):
                    for k, v in yaml_config[section].items():
                        args[section][k] = v
            # Override ENV parameters for observation size from model config
            if "env" in yaml_config and isinstance(yaml_config["env"], dict):
                for k, v in yaml_config["env"].items():
                    if k in KEYS_OF_INTEREST:
                        args["env"][k] = v

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)

    train_seed = args["train"]["seed"]
    if train_seed is None:
        train_seed = time.time_ns() & 0xFFFFFFFF
    torch.manual_seed(train_seed)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    if "LOCAL_RANK" in os.environ:
        args["train"]["device"] = "cuda"
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank], output_device=local_rank)
        if hasattr(policy, "lstm"):
            # model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    # Under DDP only rank 0 owns the run logger; other ranks keep logger=None,
    # which PuffeRL wraps in a NoLogger. Without this gate every rank calls
    # wandb.init()/NeptuneLogger and you get world_size duplicate runs.
    is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    if is_rank0:
        if args["neptune"]:
            logger = NeptuneLogger(args)
        elif args["wandb"]:
            logger = WandbLogger(args)
        elif args["tb"]:
            date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_dir = os.path.join(args["train"]["data_dir"], rf"{env_name}_" + date_time)
            logger = TensorBoardLogger(
                run_id=date_time,
                experiment_dir=experiment_dir,
            )

    train_config = dict(**args["train"], env=env_name, eval=args.get("eval", {}))
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    if args["train"].get("resume_state_path"):
        pufferl.load_training_state(args["train"]["resume_state_path"])

    from pufferlib.ocean.benchmark.manager import EvalManager

    pufferl._eval_manager = EvalManager.from_config(args, run_id=logger.run_id if logger else None)

    # Restore optimizer state + step counters when resuming from a checkpoint.
    # save_checkpoint writes models/model_<env>_<epoch>.pt and trainer_state.pt
    # (sibling of models/) — so trainer_state.pt is one dir above the .pt path.
    if args.get("load_model_path"):
        trainer_state_path = os.path.join(os.path.dirname(os.path.dirname(args["load_model_path"])), "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            print(f"Resuming optimizer/step state from {trainer_state_path}")
            tstate = torch.load(trainer_state_path, map_location=train_config["device"])
            pufferl.optimizer.load_state_dict(tstate["optimizer_state_dict"])
            pufferl.global_step = tstate.get("global_step", pufferl.global_step)
            pufferl.epoch = tstate.get("update", pufferl.epoch)
            # Fast-forward the LR scheduler to the resumed epoch so the cosine
            # schedule continues where it left off.
            for _ in range(pufferl.epoch):
                pufferl.scheduler.step()
        else:
            print(f"No trainer_state.pt next to {args['load_model_path']}; starting optimizer fresh.")

    path = os.path.join(args["train"]["data_dir"], f"{env_name}_{pufferl.logger.run_id}")
    _save_experiment_config(args, path)

    # Sweep needs data for early stopped runs, so send data when steps > 100M
    logging_threshold = min(0.20 * train_config["total_timesteps"], 100_000_000)
    all_logs = []

    while pufferl.global_step < train_config["total_timesteps"]:
        if is_cuda_device(train_config["device"]):
            torch.compiler.cudagraph_mark_step_begin()
        try:
            pufferl.evaluate()
        except Exception:
            pufferl.vecenv.close()
            pufferl.utilization.stop()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            raise
        if is_cuda_device(train_config["device"]):
            torch.compiler.cudagraph_mark_step_begin()
        try:
            logs = pufferl.train()
        except Exception:
            pufferl.vecenv.close()
            pufferl.utilization.stop()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            raise

        if logs is not None:
            should_stop_early = False
            if early_stop_fn is not None:
                should_stop_early = early_stop_fn(logs)
                # This is hacky, but need to see if threshold looks reasonable
                if "early_stop_threshold" in logs:
                    pufferl.logger.log(
                        {"environment/early_stop_threshold": logs["early_stop_threshold"]}, logs["agent_steps"]
                    )

            if pufferl.global_step > logging_threshold:
                all_logs.append(logs)

            if should_stop_early:
                model_path = pufferl.close()
                pufferl.logger.close(model_path, early_stop=True)
                return all_logs

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    # i = 0
    # stats = {}
    # while i < 32 or not stats:
    #     stats = pufferl.evaluate()
    #     i += 1

    # Force every enabled evaluator to fire once at shutdown, regardless
    # of whether `epoch % interval == 0` lines up. Restores the
    # `epoch % interval == 0 or done_training` semantics from the legacy
    # eval pipeline — without this the final epoch's metrics get dropped
    # whenever total_timesteps lands off-cycle. Rank-0 only under DDP
    # for the same reasons as the in-loop call above.
    is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    if pufferl._eval_manager is not None and is_rank0:
        pufferl._eval_manager.maybe_run(
            epoch=pufferl.epoch,
            policy=pufferl.uncompiled_policy,
            env_name=pufferl.config["env"],
            logger=pufferl.logger,
            global_step=pufferl.agent_steps,
            force=True,
        )

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path, early_stop=False)
    return all_logs


# Env keys that define the observation / action layout a checkpoint was trained
# with. They must match at eval or the policy unpacks the obs at the wrong
# offsets, so they come from the checkpoint — unlike the eval-policy env config
# (sim mode, maps, rewards, behaviors), which the [eval.<name>] section owns.
_ARCH_ENV_KEYS = (
    # action / dynamics
    "action_type",
    "dynamics_model",
    "trajectory_prediction_length",
    "num_trajectory_scaling_factors",
    "trajectory_scaling_factors",
    # observation token counts + scope
    "obs_slots_partners_n",
    "obs_slots_lane_n",
    "obs_slots_boundary_n",
    "obs_lane_stride",
    "obs_boundary_stride",
    "obs_slots_traffic_controls_n",
    "obs_dropout_lane",
    "obs_dropout_boundary",
    "traffic_control_scope",
    "reward_conditioning",
    # target / goal representation
    "target_type",
    "num_goals",
    "min_goal_spacing",
    "max_goal_spacing",
    # observation normalization scales + spatial extent — the policy was
    # trained against these, so wrong values feed it mis-scaled / clipped obs.
    "obs_norm_xy_offset_m",
    "obs_norm_goal_offset_m",
    "obs_norm_veh_length_m",
    "obs_norm_veh_width_m",
    "obs_norm_road_seg_length_m",
    "obs_norm_road_seg_width_m",
    "obs_range_traffic_control_m",
    "obs_range_partner_m",
    "obs_range_road_front_m",
    "obs_range_road_behind_m",
    "obs_range_road_side_m",
)


def _merge_checkpoint_arch(args, model_path):
    """Adopt a checkpoint's architecture from its sibling config.yaml.

    A standalone eval may load a checkpoint whose network shape or observation
    layout differs from drive.ini. The training run writes config.yaml next to
    models/, so pull from it before the policy/env are built:
      - policy.*, rnn.*, policy_name, rnn_name (+ derived use_rnn) — the net,
        else load_state_dict mismatches.
      - the obs/action-layout env keys (_ARCH_ENV_KEYS) — else the eval env
        packs observations the policy can't unpack.
    The eval-policy env config (simulation_mode, map_dir, num_*, rewards,
    behaviors) is intentionally left to the [eval.<name>] section.
    """
    config_yaml_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), "config.yaml")
    if not os.path.exists(config_yaml_path):
        return args
    with open(config_yaml_path) as f:
        yaml_config = yaml.safe_load(f) or {}
    for section in ("policy", "rnn"):
        if isinstance(yaml_config.get(section), dict):
            args.setdefault(section, {}).update(yaml_config[section])
    for key in ("rnn_name", "policy_name"):
        if key in yaml_config:
            args[key] = yaml_config[key]
    args.setdefault("train", {})["use_rnn"] = args.get("rnn_name") is not None
    env_cfg = yaml_config.get("env", {})
    if isinstance(env_cfg, dict):
        args.setdefault("env", {})
        for key in _ARCH_ENV_KEYS:
            if key in env_cfg:
                args["env"][key] = env_cfg[key]
    print(f"[eval] merged policy/rnn + obs-layout config from {config_yaml_path}")
    return args


def eval(
    env_name,
    args=None,
    vecenv=None,
    policy=None,
    evaluator_name=None,
    out_path=None,
    global_step=None,
    epoch=None,
    eval_simulation=None,
    num_scenarios=None,
    render=None,
    render_backend=None,
    num_maps=None,
):
    """Run a single named evaluator from drive.ini.

    Standalone form: `puffer eval puffer_drive --evaluator <name>`. The
    evaluator's config (env/vec overrides, render flag, etc.) comes from
    the [eval.<name>] section. Loads the policy from `--load-model-path`.

    Ad-hoc form: instead of `--evaluator`, pass `--eval_simulation
    gigaflow|replay` to pick `validation_<sim>`. Either way, the simple
    flags `--num_scenarios`, `--render`, `--render-backend`, `--num_maps`
    override the chosen evaluator's config for this run (only when passed),
    so a checkpoint can be evaluated at an arbitrary scale from the CLI
    without editing drive.ini.

    Subprocess form: `--out <json>` writes the result dict to a JSON file
    so the parent EvalManager can read structured metrics back without
    parsing stdout. `--global-step` and `--epoch` flow through so render
    mp4 filenames carry the right `_epoch{E}_step{N}` tag (otherwise
    every subprocess invocation would write `_epoch0_step0.mp4` and
    successive epochs would silently overwrite each other on disk).
    """
    from pufferlib.ocean.benchmark.manager import EvalManager

    args = args or load_config(env_name)

    # When evaluating a checkpoint, adopt its network architecture from the
    # training run's sibling config.yaml so the policy is built to match the
    # weights regardless of what drive.ini currently says.
    if args.get("load_model_path"):
        _merge_checkpoint_arch(args, args["load_model_path"])

    if evaluator_name is None:
        evaluator_name = args.get("evaluator")
    if evaluator_name is None and eval_simulation:
        evaluator_name = f"validation_{eval_simulation}"
    if evaluator_name is None:
        raise pufferlib.APIUsageError(
            "puffer eval requires --evaluator <name> (or --eval_simulation gigaflow|replay); "
            "named [eval.<name>] sections live in drive.ini"
        )

    # Derive a default render output dir from the model path when none is set.
    # experiments/puffer_drive_e6guw2wv/models/model.pt → benchmark/puffer_e6guw2wv
    if not args.get("render_results_dir") and not args.get("eval_results_dir"):
        load_model_path = args.get("load_model_path")
        if load_model_path:
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(load_model_path))))
            run_id = exp_name.removeprefix(f"{env_name}_")
            args["render_results_dir"] = os.path.join("benchmark", f"puffer_{run_id}")

    manager = EvalManager.from_config(args)
    target = next((e for e in manager.evaluators if e.name == evaluator_name), None)
    if target is None:
        raise KeyError(f"No [eval.{evaluator_name}] section found. Known: {[e.name for e in manager.evaluators]}")

    # Ad-hoc CLI overrides applied to the chosen evaluator for this run.
    # The evaluator reads self.config / self.render at rollout time, so
    # mutating them here takes effect without touching drive.ini.
    if num_scenarios is not None:
        target.config.setdefault("eval", {})["num_scenarios"] = int(num_scenarios)
    if num_maps is not None:
        target.config.setdefault("env", {})["num_maps"] = int(num_maps)
    if render is not None:
        target.render = bool(render)
    if render_backend is not None:
        target.config["render_backend"] = render_backend

    # Build a fresh vecenv inside the manager via the evaluator's overrides.
    # Policy can come from a checkpoint (load_model_path) or be passed in.
    if policy is None:
        # Need a probe vecenv just to construct the policy with the right
        # obs/action spaces. Use the matching evaluator's env_overrides so
        # the obs shape matches what the rollout will see.
        probe_args = manager._build_eval_args(target, env_name=env_name, global_step=None)
        probe_vec = load_env(env_name, probe_args)
        policy = load_policy(probe_args, probe_vec, env_name)
        probe_vec.close()

    result = manager.run_one_by_name(
        evaluator_name,
        policy=policy,
        env_name=env_name,
        logger=None,
        global_step=global_step,
        epoch=epoch,
    )

    print("EVAL_RESULT_JSON_START")
    import json

    print(json.dumps({"name": evaluator_name, "metrics": result.metrics}))
    print("EVAL_RESULT_JSON_END")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(
                {"name": evaluator_name, "metrics": result.metrics, "frames": [str(p) for p in result.frames]},
                f,
            )

    return result.metrics


def mine_failures(env_name, args=None):
    """Roll out a trained policy against a fixed scenario suite, capture per-
    episode compact replays + summaries, and produce a sortable HTML index.

    Config keys (under `mine.` or via CLI flags):
      - mine.output_dir (default: f"./failure_mining/{env_name}")
      - mine.num_episodes (default: 100)
      - mine.score_threshold (default: -inf, i.e. capture every episode;
        episodes with `episode_return` strictly below this threshold are
        flagged as failures and have their replay bundle written to disk)
      - mine.render (default: True; render each captured replay to HTML and
        write a top-level index.html via mining_viz)

    Other args reused from training/eval: load_model_path, env.*, policy_name,
    train.device, vec.* (only num_envs is meaningful here; mining always uses
    a single vec env).
    """
    import csv
    import pandas as pd

    from pufferlib import mining_viz

    args = args or load_config(env_name)
    mine_cfg = args.get("mine") or {}
    output_dir = mine_cfg.get("output_dir") or f"./failure_mining/{env_name}"
    num_episodes = int(mine_cfg.get("num_episodes", 100))
    score_threshold = float(mine_cfg.get("score_threshold", float("-inf")))
    do_render = bool(mine_cfg.get("render", True))

    env_kwargs = dict(args["env"])
    env_kwargs["capture_compact_replay"] = True
    env_kwargs["emit_completed_episodes"] = True
    env_kwargs["eval_mode"] = env_kwargs.get("eval_mode", 1)
    env_kwargs["resample_frequency"] = 0
    # Mining is sequential: one vec env, walk episodes one batch at a time.
    vec_kwargs = dict(args["vec"])
    vec_kwargs.setdefault("num_envs", 1)
    vec_kwargs.setdefault("num_workers", 1)
    vec_kwargs.setdefault("batch_size", vec_kwargs["num_envs"])

    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    vecenv = pufferlib.vector.make(make_env, env_kwargs=env_kwargs, **vec_kwargs)

    policy = load_policy({**args, "env": env_kwargs}, vecenv, env_name)
    policy.eval()

    device = args["train"]["device"]
    if isinstance(device, int):
        device = torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")

    replay_dir = os.path.join(output_dir, "replays")
    render_dir = os.path.join(output_dir, "renders") if do_render else None
    os.makedirs(replay_dir, exist_ok=True)
    if render_dir is not None:
        os.makedirs(render_dir, exist_ok=True)

    rows = []
    next_episode_id = 0
    seed = args.get("train", {}).get("seed") or 0
    if hasattr(vecenv, "async_reset"):
        vecenv.async_reset(seed=seed)
    obs_arr, *_ = vecenv.recv()
    pbar_total = num_episodes
    pbar_done = 0
    print(f"[mine_failures] target episodes={num_episodes} output={output_dir} score_threshold={score_threshold}")
    while pbar_done < num_episodes:
        with torch.no_grad():
            o_t = torch.as_tensor(obs_arr).to(device)
            state = {"reward": None, "done": None, "env_id": None, "mask": None}
            logits, _ = policy.forward_eval(o_t, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy()
            if action.ndim == 1 and len(vecenv.single_action_space.shape) >= 1:
                action = action.reshape(-1, *vecenv.single_action_space.shape)
        vecenv.send(action)
        obs_arr, _, _, _, infos, *_ = vecenv.recv()
        for info in infos:
            if not isinstance(info, dict):
                continue
            if info.get("summary_type") != "completed_episode":
                continue
            episode_id = next_episode_id
            next_episode_id += 1
            bundle_bytes = info.pop("compact_replay_bundle", None)
            row = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in info.items()}
            row["episode_id"] = episode_id
            row["avg_distance_per_infraction"] = float(row.get("total_distance_travelled", 0.0)) / max(
                1.0, float(row.get("total_infractions", 0.0))
            )
            row["failed"] = 1 if row.get("episode_return", 0.0) < score_threshold else 0
            row["has_replay"] = 0
            row["replay_path"] = None
            if bundle_bytes is not None and row["failed"]:
                replay_path = os.path.join(replay_dir, f"episode_{episode_id:06d}.replay.zlib")
                with open(replay_path, "wb") as f:
                    f.write(bundle_bytes)
                row["has_replay"] = 1
                row["replay_path"] = replay_path
            rows.append(row)
            pbar_done += 1
            if pbar_done >= num_episodes:
                break

    vecenv.close()

    episodes_df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "episodes.csv")
    episodes_df.to_csv(csv_path, index=False)
    print(
        f"[mine_failures] wrote {csv_path} ({len(rows)} episodes, {int(episodes_df['failed'].sum())} failures captured)"
    )

    if do_render and render_dir is not None:
        render_lookup = {}
        rendered = 0
        for row in rows:
            if not row.get("has_replay"):
                continue
            ep_id = int(row["episode_id"])
            out_html = os.path.join(render_dir, f"episode_{ep_id:06d}.html")
            mining_viz.render_compact_replay_html(row["replay_path"], out_html, render_context={"summary": row})
            render_lookup[ep_id] = os.path.relpath(out_html, render_dir)
            rendered += 1
        index_path = os.path.join(render_dir, "index.html")
        mining_viz.generate_failure_index(episodes_df, render_lookup, index_path)
        print(f"[mine_failures] rendered {rendered} replays + index at {index_path}")

    return episodes_df


def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"] and not args["tb"]:
        raise pufferlib.APIUsageError("Sweeps require either wandb, neptune, or tb")

    method = args["sweep"].pop("method")
    try:
        sweep_cls = getattr(pufferlib.sweep, method)
    except:
        raise pufferlib.APIUsageError(f"Invalid sweep method {method}. See pufferlib.sweep")

    sweep = sweep_cls(args["sweep"])
    points_per_run = args["sweep"]["downsample"]
    target_key = f"environment/{args['sweep']['metric']}"

    for i in range(args["max_runs"]):
        seed = time.time_ns() & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        sweep.suggest(args)
        total_timesteps = args["train"]["total_timesteps"]
        all_logs = train(env_name, args=args)
        all_logs = [e for e in all_logs if target_key in e]
        scores = downsample([log[target_key] for log in all_logs], points_per_run)
        costs = downsample([log["uptime"] for log in all_logs], points_per_run)
        timesteps = downsample([log["agent_steps"] for log in all_logs], points_per_run)
        for score, cost, timestep in zip(scores, costs, timesteps):
            args["train"]["total_timesteps"] = timestep
            sweep.observe(args, score, cost)

        # Prevent logging final eval steps as training steps
        args["train"]["total_timesteps"] = total_timesteps


def controlled_exp(env_name, args=None):
    """Run experiments with all combinations of specified parameter values."""
    import itertools
    from copy import deepcopy

    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"]:
        raise pufferlib.APIUsageError("Targeted experiments require either wandb or neptune")

    # Check if controlled_exp config exists
    if "controlled_exp" not in args:
        raise pufferlib.APIUsageError("No [controlled_exp.*] sections found in config")

    # Extract parameters from controlled_exp namespace
    params = {}
    for section, section_config in args["controlled_exp"].items():
        if isinstance(section_config, dict):
            for param, param_config in section_config.items():
                if isinstance(param_config, dict) and "values" in param_config:
                    params[f"{section}.{param}"] = param_config["values"]

    if not params:
        raise pufferlib.APIUsageError("No parameters with 'values' lists found in [controlled_exp.*] sections")

    # Generate all combinations
    keys = list(params.keys())
    combinations = list(itertools.product(*[params[k] for k in keys]))

    print(f"Running a total of {len(combinations)} experiments with parameters: {keys}")

    # Run each combination
    for i, combo in enumerate(combinations, 1):
        exp_args = deepcopy(args)

        # Set parameters
        for key, value in zip(keys, combo):
            section, param = key.split(".")
            exp_args[section][param] = value

        print(f"\nExperiment {i}/{len(combinations)}: {dict(zip(keys, combo))}")

        # Train
        train(env_name, args=exp_args)

    print(f"\n✓ Completed all {len(combinations)} experiments")


def profile(args=None, env_name=None, vecenv=None, policy=None):
    args = load_config()
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    train_config = dict(**args["train"], env=args["env_name"], tag=args["tag"])
    pufferl = PuffeRL(train_config, vecenv, policy, neptune=args["neptune"], wandb=args["wandb"])

    from torch.profiler import profile, record_function, ProfilerActivity

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                stats = pufferl.evaluate()
                pufferl.train()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")


def export(args=None, env_name=None, vecenv=None, policy=None, path=None, silent=False):
    args = args or load_config(env_name)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)

    weights = []
    for name, param in policy.named_parameters():
        weights.append(param.data.cpu().numpy().flatten())
        if not silent:
            print(name, param.shape, param.data.cpu().numpy().ravel()[0])

    weights = np.concatenate(weights)
    if path is None:
        path = f"pufferlib/resources/drive/{args['env_name']}_weights.bin"

    weights.tofile(path)

    if not silent:
        print(f"Saved {len(weights)} weights to {path}")


def ensure_drive_binary():
    """Delete existing visualize binary and rebuild it. This ensures the
    binary is always up-to-date with the latest code changes.
    """
    if os.path.exists("./visualize"):
        print("Removing existing visualize binary...")
        os.remove("./visualize")

    print("Building visualize binary...")
    try:
        result = subprocess.run(
            ["bash", "scripts/build_ocean.sh", "visualize", "local"], capture_output=True, text=True, timeout=300
        )

        if result.returncode == 0:
            print("Successfully built visualize binary")
        else:
            print(f"Build failed: {result.stderr}")
            raise RuntimeError("Failed to build visualize binary for rendering")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Build timed out")
    except Exception as e:
        raise RuntimeError(f"Build error: {e}")


def autotune(args=None, env_name=None, vecenv=None, policy=None):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    env_name = args["env_name"]
    make_env = env_module.env_creator(env_name)
    pufferlib.vector.autotune(make_env, batch_size=args["train"]["env_batch_size"])


def load_env(env_name, args):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    return pufferlib.vector.make(make_env, env_kwargs=args["env"], **args["vec"])


def load_policy(args, vecenv, env_name=""):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)

    device = torch_device(args["train"]["device"])
    policy_cls = getattr(env_module.torch, args["policy_name"])
    policy = policy_cls(vecenv.driver_env, **args["policy"])

    rnn_name = args["rnn_name"]
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args["rnn_name"])
        policy = rnn_cls(vecenv.driver_env, policy, **args["rnn"])

    policy = policy.to(device)

    load_id = args["load_id"]
    if load_id is not None:
        if args["neptune"]:
            path = NeptuneLogger(args, load_id, mode="read-only").download()
        elif args["wandb"]:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError("No run id provided for eval")

        state_dict = torch.load(path, map_location=device)
        policy.load_state_dict(clean_policy_state_dict(state_dict))

    load_path = args["load_model_path"]
    if load_path == "latest":
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        policy.load_state_dict(clean_policy_state_dict(state_dict))
        # state_path = os.path.join(*load_path.split('/')[:-1], 'state.pt')
        # optim_state = torch.load(state_path)['optimizer_state_dict']
        # pufferl.optimizer.load_state_dict(optim_state)

    return policy


def load_config(env_name, config_dir=None):
    parser = argparse.ArgumentParser(
        description=f":blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]"
        " demo options. Shows valid args for your env and policy",
        formatter_class=RichHelpFormatter,
        add_help=False,
    )
    parser.add_argument("--load-model-path", type=str, default=None, help="Path to a pretrained checkpoint")
    parser.add_argument(
        "--load-id", type=str, default=None, help="Kickstart/eval from from a finished Wandb/Neptune run"
    )
    parser.add_argument(
        "--render-mode", type=str, default="auto", choices=["auto", "human", "ansi", "rgb_array", "raylib", "None"]
    )
    parser.add_argument("--video-path", type=str, default="videos", help="Path to save videos")
    parser.add_argument("--num_scenarios", type=int, default=3, help="Number of scenarios to eval")
    parser.add_argument("--render", type=int, default=0, help="Rendering the evaluation")
    parser.add_argument("--agent_index", nargs="*", type=int, default=None, help="Agent index to plot the observation")
    parser.add_argument("--save-frames", type=int, default=0)
    parser.add_argument("--gif-path", type=str, default="eval.gif")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-runs", type=int, default=200, help="Max number of sweep runs")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--wandb-group", type=str, default="debug")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Wandb run display name. Unset → wandb auto-generates one.",
    )
    parser.add_argument("--neptune", action="store_true", help="Use neptune for logging")
    parser.add_argument("--neptune-name", type=str, default="pufferai")
    parser.add_argument("--neptune-project", type=str, default="ablations")
    parser.add_argument("--tb", action="store_true", help="Use tensorboard for logging")
    parser.add_argument("--local-rank", type=int, default=0, help="Used by torchrun for DDP")
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment")
    parser.add_argument(
        "--eval_simulation", type=str, default=None, help="Simulation mode for evaluation - gigaflow/replay"
    )
    args = parser.parse_known_args()[0]

    if config_dir is None:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
    else:
        print("Using custom config dir:", config_dir)
        puffer_dir = config_dir

    # Load defaults and config
    puffer_config_dir = os.path.join(puffer_dir, "config/**/*.ini")
    puffer_default_config = os.path.join(puffer_dir, "config/default.ini")
    if env_name == "default":
        p = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
            p.read([puffer_default_config, path])
            if env_name in p["base"]["env_name"].split():
                break
        else:
            raise pufferlib.APIUsageError("No config for env_name {}".format(env_name))

    # Dynamic help menu from config
    def puffer_type(value):
        try:
            return ast.literal_eval(value)
        except:
            return value

    for section in p.sections():
        for key in p[section]:
            fmt = f"--{key}" if section == "base" else f"--{section}.{key}"
            parser.add_argument(fmt.replace("_", "-"), default=puffer_type(p[section][key]), type=puffer_type)

    parser.add_argument(
        "-h", "--help", default=argparse.SUPPRESS, action="help", help="Show this help message and exit"
    )

    # Unpack to nested dict
    parsed = vars(parser.parse_args())
    args = defaultdict(dict)
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            prev = next
            next = next.setdefault(subkey, {})

        prev[subkey] = value

    args["train"]["use_rnn"] = args["rnn_name"] is not None

    # Use World size to divide Num_Agents / minibatch size in DDP
    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        args["env"]["num_agents"] = args["env"]["num_agents"]
        args["train"]["minibatch_size"] = args["train"]["minibatch_size"]
        args["train"]["max_minibatch_size"] = args["train"]["max_minibatch_size"]
        args["train"]["total_timesteps"] = args["train"]["total_timesteps"] // world_size

    return args


def main():
    err = "Usage: puffer [train, eval, mine_failures, sweep, controlled_exp, autotune, profile, export] [env_name] [optional args]. --help for more info"
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    elif mode == "eval":
        # Pull eval-specific argv before load_config consumes them. These
        # aren't registered as configparser-style dotted keys because
        # they're per-invocation, not per-config-section.
        evaluator_name = None
        out_path = None
        global_step = None
        epoch = None
        # Ad-hoc overrides for the chosen evaluator (None = not passed, so the
        # [eval.<name>] section value stands). Pulled from argv here rather
        # than registered in load_config so we can tell "passed" from
        # "default" and only override when the user actually set them.
        eval_simulation = None
        render_backend = None
        num_scenarios = None
        render = None
        num_maps = None
        scalar_flags = {
            "--num-scenarios": "num_scenarios",
            "--num_scenarios": "num_scenarios",
            "--render": "render",
            "--num-maps": "num_maps",
            "--num_maps": "num_maps",
        }
        str_flags = {
            "--eval-simulation": "eval_simulation",
            "--eval_simulation": "eval_simulation",
            "--render-backend": "render_backend",
            "--render_backend": "render_backend",
        }
        str_overrides = {}
        overrides = {}
        i = 0
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--evaluator" and i + 1 < len(sys.argv):
                evaluator_name = sys.argv[i + 1]
                del sys.argv[i : i + 2]
                continue
            if arg == "--out" and i + 1 < len(sys.argv):
                out_path = sys.argv[i + 1]
                del sys.argv[i : i + 2]
                continue
            if arg == "--global-step" and i + 1 < len(sys.argv):
                global_step = int(sys.argv[i + 1])
                del sys.argv[i : i + 2]
                continue
            if arg == "--epoch" and i + 1 < len(sys.argv):
                epoch = int(sys.argv[i + 1])
                del sys.argv[i : i + 2]
                continue
            if arg in str_flags and i + 1 < len(sys.argv):
                str_overrides[str_flags[arg]] = sys.argv[i + 1]
                del sys.argv[i : i + 2]
                continue
            if arg in scalar_flags and i + 1 < len(sys.argv):
                overrides[scalar_flags[arg]] = int(sys.argv[i + 1])
                del sys.argv[i : i + 2]
                continue
            i += 1
        eval_simulation = str_overrides.get("eval_simulation")
        render_backend = str_overrides.get("render_backend")
        num_scenarios = overrides.get("num_scenarios")
        render = overrides.get("render")
        num_maps = overrides.get("num_maps")
        eval(
            env_name=env_name,
            evaluator_name=evaluator_name,
            out_path=out_path,
            global_step=global_step,
            epoch=epoch,
            eval_simulation=eval_simulation,
            num_scenarios=num_scenarios,
            render=render,
            render_backend=render_backend,
            num_maps=num_maps,
        )
    elif mode == "sweep":
        sweep(env_name=env_name)
    elif mode == "controlled_exp":
        controlled_exp(env_name=env_name)
    elif mode == "autotune":
        autotune(env_name=env_name)
    elif mode == "profile":
        profile(env_name=env_name)
    elif mode == "export":
        export(env_name=env_name)
    elif mode in ("mine_failures", "mine-failures"):
        mine_failures(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
