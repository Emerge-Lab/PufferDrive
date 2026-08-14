## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import copy
import warnings

import pandas as pd


warnings.filterwarnings("error", category=RuntimeWarning)

import os
import sys
import traceback
import glob
import time
import random
import shutil
import subprocess
import importlib
from datetime import datetime
from threading import Thread
from collections import defaultdict, deque
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record

import pufferlib
from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark
from pufferlib.ocean.evaluation_utils import eval_replay as drive_eval_replay
import pufferlib.sweep
import pufferlib.utils
import pufferlib.vector
import pufferlib.pytorch
from pufferlib.config_schema import ENV_SCHEMAS


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
    "total_distance_travelled_sum",
    "total_infraction_count",
}

# Metric key prefixes for benchmark results. Training evaluation logs a step series;
# a standalone eval writes run-level summaries, so the two never share a key.
TRAINING_EVAL_KEY_PREFIX = "eval_"


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
        # Decorrelate reset streams across DDP ranks
        if seed is not None and torch.distributed.is_initialized():
            seed = seed * torch.distributed.get_world_size() + torch.distributed.get_rank()

        # Vecenv info
        vecenv.async_reset(seed)

        self.env_continuous = isinstance(vecenv.single_action_space, pufferlib.spaces.Box)
        obs_space = vecenv.single_observation_space
        # Custom policy attributes live on the base module, not the DDP/compile wrapper.
        unwrapped_policy = base_policy(policy)
        if self.env_continuous and not unwrapped_policy.is_continuous:
            action_shape = (len(unwrapped_policy.atn_dim),)
            action_dtype = torch.int32
        else:
            action_shape = vecenv.single_action_space.shape
            action_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[vecenv.single_action_space.dtype]
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
            *action_shape,
            device=device,
            dtype=action_dtype,
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
                weight_decay=config["adam_weight_decay"],
            )
        elif config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
                weight_decay=config["adam_weight_decay"],
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
                action, logprob, _, cont_action = pufferlib.pytorch.sample_logits(
                    logits, env_continuous=self.env_continuous, policy=self.uncompiled_policy
                )
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

            if self.env_continuous and not self.uncompiled_policy.is_continuous:
                cont_action = cont_action.cpu().numpy()
                self.vecenv.send(cont_action.squeeze(0))
            else:
                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)
                self.vecenv.send(action)

        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        return pufferlib.utils.reduce_environment_metrics(self.stats)

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
                model_dir = self.config["data_dir"]
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

        return logs

    def _ppo_loss(self, mb_obs, mb_actions, mb_logprobs, mb_values, mb_returns, mb_adv, adv_weights=None):
        config = self.config
        state = dict(action=mb_actions, lstm_h=None, lstm_c=None)
        with self.amp_context:
            logits, newvalue = self.policy(mb_obs, state)
        logits = logits_to_float(logits)
        newvalue = newvalue.float()
        _, newlogprob, entropy, _ = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

        newlogprob = newlogprob.float().view_as(mb_logprobs)
        newvalue = newvalue.view_as(mb_returns)
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

        with torch.no_grad():
            if torch.distributed.is_initialized():
                # This mean computation assumes that all GPUs use the same batch size. This is currently guaranteed.
                world_size = torch.distributed.get_world_size()
                # Distributed mean
                advantage_mean = mb_adv.mean()
                torch.distributed.all_reduce(advantage_mean, op=torch.distributed.ReduceOp.SUM)
                advantage_mean = advantage_mean / world_size

                advantage_std = torch.sum(torch.square(mb_adv - advantage_mean))
                torch.distributed.all_reduce(advantage_std, op=torch.distributed.ReduceOp.SUM)
                advantage_std = advantage_std / (world_size * torch.numel(mb_adv) - 1)  # -1 is bessel's correction
                advantage_std = torch.sqrt(advantage_std)
            else:
                advantage_mean = mb_adv.mean()
                advantage_std = mb_adv.std()

            mb_adv = (mb_adv - advantage_mean) / (advantage_std + 1e-8)
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

        losses["masked_fraction"] = 1.0 - (valid_idx.numel() / max(total_transitions, 1))

        if config["adv_filter_enabled"]:
            ewma_beta = config["adv_filter_ewma_beta"]
            current_max = valid_abs_adv.max().item() if valid_abs_adv.numel() > 0 else 0.0
            self.ema_max = current_max if epoch == 0 else ewma_beta * current_max + (1 - ewma_beta) * self.ema_max
            threshold = config["adv_filter_threshold_scale"] * self.ema_max

            keep_mask = valid_abs_adv >= threshold
            keep_idx = valid_idx[keep_mask]
            num_valid, num_kept = valid_idx.numel(), keep_idx.numel()

            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                # Filtering keeps a different count per rank, so trim to the global
                # minimum to keep the number of minibatches synchronized
                kept_tensor = torch.tensor([num_kept], device=device)
                torch.distributed.all_reduce(kept_tensor, op=torch.distributed.ReduceOp.MIN)
                min_num_kept = kept_tensor.item()
                if num_kept > min_num_kept:
                    if min_num_kept == 0:
                        keep_idx = keep_idx[:0]
                    else:
                        top_idx = torch.topk(valid_abs_adv[keep_mask], min_num_kept, largest=True, sorted=False).indices
                        keep_idx = keep_idx[top_idx]

            kept_fraction = keep_idx.numel() / max(num_valid, 1)
            losses["filter_threshold"] = threshold
            losses["ema_max"] = self.ema_max
            losses["kept_fraction"] = kept_fraction
            losses["filtered_fraction"] = 1.0 - kept_fraction
        else:
            keep_idx = valid_idx

        if config["min_batch_size"] is not None:
            num_missing = config["min_batch_size"] - keep_idx.numel()
            if num_missing > 0 and keep_idx.numel() > 0:
                # Repeat random elements of keep_idx until min_batch_size is reached
                pad_idx = torch.randint(keep_idx.numel(), (num_missing,), device=keep_idx.device)
                keep_idx = torch.cat([keep_idx, keep_idx[pad_idx]])

        self.optimizer.zero_grad()
        total_minibatches = 0
        pending_minibatches = 0

        # Disabled for now: dropping the partial final minibatch means zero optimizer
        # steps (silently) whenever fewer than minibatch_size transitions survive the
        # advantage filter, which permanently freezes a plateaued policy.
        # full_minibatch_transitions = (keep_idx.numel() // self.minibatch_size) * self.minibatch_size
        full_minibatch_transitions = keep_idx.numel()

        for _ in range(config["update_epochs"]):
            permutation = keep_idx[torch.randperm(keep_idx.numel(), device=keep_idx.device)]
            for start in range(0, full_minibatch_transitions, self.minibatch_size):
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
        self.stats = pufferlib.utils.reduce_environment_metrics(self.stats)

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
        # Fixed, configurable filename so a follow-up eval job can reference the final
        # model without knowing which epoch training stopped at.
        path = os.path.join(self.config["data_dir"], self.config["final_model_name"])
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        run_id = self.logger.run_id
        path = self.config["data_dir"]
        if not os.path.exists(path):
            os.makedirs(path)

        models_dir = os.path.join(path, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_name = f"model_{self.config['env']}_{self.epoch:06d}.pt"
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            torch.save(self.uncompiled_policy.state_dict(), model_path)
        for old_model_path in glob.glob(os.path.join(models_dir, "model_*.pt")):
            if old_model_path != model_path:
                os.remove(old_model_path)

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
            for old_best_path in glob.glob(os.path.join(path, "best_models", "best_trainer_state_*.pt")):
                if old_best_path != best_state_file:
                    os.remove(old_best_path)
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
        self.run_id = run_id or args["run_name"]

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
            # Neptune's resume-by-name key, mirroring the wandb id above.
            custom_run_id=None if load_id else args["run_name"],
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
    def __init__(self, args, load_id=None, resume="allow", upload_config=True, disable_meta=False):
        import wandb

        # run_name is the run id: with resume="allow" wandb attaches to the
        # existing run when one already carries this id, and creates it otherwise.
        wandb.init(
            id=load_id or args["run_name"],
            name=args["run_name"],
            project=args["wandb_project"],
            group=args["wandb_group"],
            allow_val_change=True,
            save_code=False,
            resume=resume,
            config=args if upload_config else None,
            tags=[args["tag"]] if args["tag"] is not None else [],
            settings=wandb.Settings(console="off", x_disable_meta=disable_meta),
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args["no_model_upload"]

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def finish(self):
        """End the wandb session without the model upload and early_stop that close() adds."""
        self.wandb.finish()

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        # Ship the config with the weights; load_checkpoint_architecture reads it off disk.
        config_path = os.path.join(drive_benchmark.resolve_run_dir(model_path), "config.yaml")
        if os.path.isfile(config_path):
            artifact.add_file(config_path)
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
        yaml.dump(config, f)


def _global_agent_steps(pufferl):
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return int(pufferl.global_step * world_size)


def derive_rank_seeds(vec_seed, train_seed, world_size, global_rank):
    """Deterministic per-rank (torch_seed, env_seed): DDP ranks share weights, so identical seeds
    would collect duplicate experience. global_rank is torchrun's global RANK, not LOCAL_RANK."""
    torch_seed = train_seed * world_size + global_rank
    env_seed = vec_seed
    if env_seed is not None:
        env_seed = int(np.random.SeedSequence([env_seed, train_seed, global_rank]).generate_state(1)[0])
    return torch_seed, env_seed


def train(env_name, args=None, vecenv=None, policy=None, logger=None, early_stop_fn=None):
    args = args or load_config(env_name)
    training_evaluation_scheduled = drive_benchmark.validate_training_evaluation_config(args)

    # Fine-tuning: reload network, observation configuration from config.yaml and override the args --> only change new reward / new maps / new simulation mode
    if args["load_model_path"]:
        experiment_dir = drive_benchmark.resolve_run_dir(args["load_model_path"])
        config_yaml_path = os.path.join(experiment_dir, "config.yaml")
        KEYS_OF_INTEREST = {
            "action_type",
            "dynamics_model",
            "goal_source",
            "goal_regen_mode",
            "num_goals",
            "min_goal_spacing",
            "max_goal_spacing",
            "obs_goal_lane_distance",
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
        else:
            print(
                f"No config.yaml at {config_yaml_path}; fine-tuning with the configured "
                "policy/observation architecture instead of the checkpoint's."
            )

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if "LOCAL_RANK" in os.environ:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {global_rank} (local {local_rank}), MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)

    train_seed = args["train"]["seed"]
    if train_seed is None:
        train_seed = time.time_ns() & 0xFFFFFFFF

    torch_seed, env_seed = derive_rank_seeds(args["vec"]["seed"], train_seed, world_size, global_rank)
    torch.manual_seed(torch_seed)
    vecenv = vecenv or load_env(env_name, args, seed=env_seed)
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

    # Set before the logger so the run config the logger uploads carries it too.
    args["git"] = _get_git_metadata()

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
            logger = TensorBoardLogger(
                run_id=args["run_name"],
                experiment_dir=args["train"]["data_dir"],
            )

    train_config = dict(**args["train"], env=env_name, eval=args.get("eval", {}), run_name=args["run_name"])
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    # A run is identified by its name, and its directory is train.data_dir. Relaunching
    # the same run therefore finds its own trainer_state.pt and continues from it rather
    # than overwriting the checkpoints already in that directory. An explicit
    # resume_state_path or load_model_path still wins.
    resume_state_path = args["train"].get("resume_state_path")
    if not resume_state_path and not args.get("load_model_path"):
        run_state_path = os.path.join(args["train"]["data_dir"], "trainer_state.pt")
        if os.path.exists(run_state_path):
            print(f"Run '{args['run_name']}' already exists in {args['train']['data_dir']}; resuming it.")
            resume_state_path = run_state_path

    if resume_state_path:
        pufferl.load_training_state(resume_state_path)
        # The checkpoint carries rank 0's RNG streams (save_checkpoint is rank-0 only);
        # re-decorrelate the other ranks or post-resume action sampling re-synchronizes.
        if global_rank != 0:
            torch.manual_seed(int(np.random.SeedSequence([torch_seed, pufferl.epoch]).generate_state(1)[0]))

    # Restore optimizer state + step counters when resuming from a checkpoint.
    # save_checkpoint writes models/model_<env>_<epoch>.pt and trainer_state.pt
    # (sibling of models/) — so trainer_state.pt is one dir above the .pt path.
    if args.get("load_model_path"):
        trainer_state_path = os.path.join(drive_benchmark.resolve_run_dir(args["load_model_path"]), "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            print(f"Resuming optimizer/step state from {trainer_state_path}")
            # weights_only=False as in load_training_state: the state carries the
            # numpy scalars of the saved RNG state, not just tensors.
            tstate = torch.load(trainer_state_path, map_location=train_config["device"], weights_only=False)
            pufferl.optimizer.load_state_dict(tstate["optimizer_state_dict"])
            pufferl.global_step = tstate.get("global_step", pufferl.global_step)
            pufferl.epoch = tstate.get("update", pufferl.epoch)
            # Fast-forward the LR scheduler to the resumed epoch so the cosine
            # schedule continues where it left off.
            for _ in range(pufferl.epoch):
                pufferl.scheduler.step()
        else:
            print(f"No trainer_state.pt next to {args['load_model_path']}; starting optimizer fresh.")

    path = args["train"]["data_dir"]
    if is_rank0:
        _save_experiment_config(args, path)

    # Sweep needs data for early stopped runs, so send data when steps > 100M
    logging_threshold = min(0.20 * train_config["total_timesteps"], 100_000_000)
    all_logs = []
    last_training_evaluation_epoch = None

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

        if training_evaluation_scheduled and pufferl.epoch % args["train"]["evaluation_interval_epochs"] == 0:
            last_training_evaluation_epoch = pufferl.epoch
            if is_rank0:
                run_training_evaluation(
                    env_name=env_name,
                    args=args,
                    policy=pufferl.uncompiled_policy,
                    logger=pufferl.logger,
                    epoch=pufferl.epoch,
                    global_step=_global_agent_steps(pufferl),
                    run_dir=path,
                )

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

    if training_evaluation_scheduled and last_training_evaluation_epoch != pufferl.epoch and is_rank0:
        run_training_evaluation(
            env_name=env_name,
            args=args,
            policy=pufferl.uncompiled_policy,
            logger=pufferl.logger,
            epoch=pufferl.epoch,
            global_step=_global_agent_steps(pufferl),
            run_dir=path,
        )

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path, early_stop=False)
    return all_logs


def eval(
    env_name,
    args=None,
    policy=None,
    eval_output_dir=None,
    eval_output_subdir=None,
    use_training_config=False,
    benchmark_names=None,
):
    """Run configured benchmarks or replay failures from an existing CSV."""
    cli_overrides = list(sys.argv[1:]) if args is None else []
    if any(override.split("=", 1)[0] == "env.num_agents" for override in cli_overrides):
        raise pufferlib.APIUsageError("Use eval.num_agents to configure evaluation agent count")
    args = args or load_config(env_name)
    eval_config = args["eval"]
    benchmark_config_path = eval_config["benchmark_config"]
    selected_benchmarks = benchmark_names if benchmark_names is not None else eval_config["benchmarks"]
    eval_config["benchmarks"] = selected_benchmarks
    output_name = eval_config["output_name"]
    render_scenarios = eval_config["render_scenarios"]
    render_filter = eval_config["render_filter"]
    max_rendered_failures = eval_config["max_rendered_failures"]
    failure_replay_csv = eval_config["failure_replay_csv"]
    max_sdc_replay_workers = eval_config["max_sdc_replay_workers"]
    valid_action_selections = (
        pufferlib.pytorch.ACTION_SELECT_SAMPLE,
        pufferlib.pytorch.ACTION_SELECT_MODE,
        pufferlib.pytorch.ACTION_SELECT_MEAN,
    )
    if eval_config["action_selection"] not in valid_action_selections:
        raise pufferlib.APIUsageError(
            f"eval.action_selection='{eval_config['action_selection']}' must be one of {valid_action_selections}"
        )
    if render_scenarios and failure_replay_csv is not None:
        raise pufferlib.APIUsageError(
            "eval.render_scenarios requires a standard benchmark pass and cannot be combined "
            "with eval.failure_replay_csv"
        )
    if failure_replay_csv is not None and render_filter is None:
        raise pufferlib.APIUsageError("eval.failure_replay_csv requires eval.render_filter")

    report_to_wandb = bool(args["wandb"]) and not use_training_config
    environment_config, benchmarks = drive_benchmark.load_benchmark_config(benchmark_config_path, selected_benchmarks)
    if use_training_config:
        if policy is None:
            raise pufferlib.APIUsageError("Training evaluation requires the live policy")
        base_args = copy.deepcopy(args)
        environment_config["obs_dropout_lane"] = base_args["env"]["obs_dropout_lane"]
        environment_config["obs_dropout_boundary"] = base_args["env"]["obs_dropout_boundary"]
        checkpoint_config_path = None
    else:
        base_args, checkpoint_config_path = drive_benchmark.load_checkpoint_architecture(args)

    wandb_run_identity = (
        drive_benchmark.load_checkpoint_run_identity(checkpoint_config_path) if report_to_wandb else None
    )
    if eval_output_dir is None:
        run_dir = drive_benchmark.resolve_run_dir(base_args["load_model_path"])
        eval_output_dir = os.path.join(run_dir, eval_config["output_dir_name"])
    if eval_output_subdir is None:
        eval_output_subdir = datetime.now().strftime("%Y%m%d-%H%M%S")
    failure_replay_output_dir = None
    if failure_replay_csv is not None:
        failure_replay_csv = os.path.abspath(failure_replay_csv)
        if not os.path.isfile(failure_replay_csv):
            raise pufferlib.APIUsageError(
                f"eval.failure_replay_csv must reference an existing CSV file: {failure_replay_csv}"
            )
        failure_replay_output_dir = os.path.dirname(failure_replay_csv)
    benchmark_results = {}
    evaluation_policy_cache = {"policy": policy}
    cli_override_config = OmegaConf.from_dotlist(cli_overrides)
    for benchmark in benchmarks:
        run_args = drive_benchmark.build_benchmark_args(base_args, benchmark, environment_config)
        run_args = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(dict(run_args)), cli_override_config),
            resolve=True,
        )
        run_args["env"]["num_agents"] = run_args["eval"]["num_agents"]
        if run_args["env"]["simulation_mode"] == "replay" and run_args["env"]["control_mode"] == "control_sdc_only":
            run_args["vec"]["num_envs"] = min(run_args["vec"]["num_envs"], max_sdc_replay_workers)
        output_directory_name = benchmark["name"]
        if output_name is not None:
            output_directory_name = f"{output_directory_name}_{output_name}"
        if failure_replay_output_dir is not None:
            benchmark_output_dir = failure_replay_output_dir
            resolved_benchmark_output_dir = os.path.join(benchmark_output_dir, "failures")
            os.makedirs(resolved_benchmark_output_dir, exist_ok=True)
        else:
            benchmark_output_dir = os.path.join(eval_output_dir, output_directory_name)
            benchmark_output_dir = os.path.join(benchmark_output_dir, eval_output_subdir)
            os.makedirs(benchmark_output_dir)
            resolved_benchmark_output_dir = benchmark_output_dir
        drive_benchmark.write_resolved_benchmark_config(
            run_args,
            benchmark,
            benchmark_config_path,
            checkpoint_config_path,
            os.path.join(resolved_benchmark_output_dir, "resolved_benchmark.yaml"),
        )

        np.random.seed(run_args["train"]["seed"])
        if failure_replay_csv is not None:
            benchmark_results[benchmark["name"]] = _render_eval_failures(
                env_name,
                run_args,
                benchmark,
                failure_replay_csv,
                benchmark_output_dir,
                policy,
                eval_config["capture_observations"],
                max_rendered_failures,
                evaluation_policy_cache=evaluation_policy_cache,
            )
            continue

        num_scenarios = run_args["num_scenarios"]
        num_workers = min(run_args["vec"]["num_envs"], num_scenarios)
        worker_env_kwargs, total_steps = drive_benchmark._plan_benchmark_eval_workers(
            run_args,
            num_scenarios,
            num_workers,
            run_args["env"]["scenario_length"],
            capture_replay=render_scenarios,
        )
        print(f"Evaluation {benchmark['name']}: {num_scenarios} scenarios across {num_workers} workers")
        replay_output_dir = os.path.join(benchmark_output_dir, "replays") if render_scenarios else None
        summaries = _run_eval_rollout(
            run_args,
            env_name,
            worker_env_kwargs,
            total_steps,
            f"Evaluating {benchmark['name']}",
            num_scenarios,
            policy=policy,
            replay_output_dir=replay_output_dir,
            capture_observations=render_scenarios and eval_config["capture_observations"],
            evaluation_policy_cache=evaluation_policy_cache,
        )
        summary = drive_benchmark._write_eval_reports(summaries, benchmark_output_dir, num_scenarios)
        benchmark_results[benchmark["name"]] = {
            "episodes": summaries,
            "summary": summary,
        }

        if render_scenarios:
            drive_eval_replay._render_eval_replays(summaries, benchmark_output_dir)
        elif render_filter is not None:
            _render_eval_failures(
                env_name,
                run_args,
                benchmark,
                os.path.join(benchmark_output_dir, "episode_metrics.csv"),
                benchmark_output_dir,
                policy,
                eval_config["capture_observations"],
                max_rendered_failures,
                evaluation_policy_cache=evaluation_policy_cache,
            )

    if wandb_run_identity is not None:
        report_eval_to_wandb(args, benchmark_results, wandb_run_identity, eval_config["output_dir_name"])
    return benchmark_results


def report_eval_to_wandb(args, benchmark_results, wandb_run_identity, output_dir_name):
    """Attach standalone eval results to the wandb run that trained the checkpoint."""
    metrics = drive_benchmark.summarize_benchmark_metrics(benchmark_results, f"final_{output_dir_name}_")
    if not metrics:
        print("No evaluation metrics to report to wandb.")
        return

    run_args = copy.copy(args)
    run_args.update(wandb_run_identity)

    logger = WandbLogger(run_args, resume="must", upload_config=False, disable_meta=True)
    try:
        logger.log(metrics, step=None)
    finally:
        logger.finish()
    print(f"Reported {len(metrics)} eval metrics to wandb run {wandb_run_identity['run_name']}")


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


def _run_eval_rollout(
    args,
    env_name,
    worker_env_kwargs,
    total_steps,
    desc,
    expected_episodes,
    policy=None,
    recorded_agents_per_batch=None,
    replay_output_dir=None,
    capture_observations=False,
    episode_id_offset=0,
    evaluation_policy_cache=None,
):
    """Roll out a deterministic policy over the workers and gather evaluation episode summaries."""
    num_workers = len(worker_env_kwargs)
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    vecenv = pufferlib.vector.make(
        [make_env] * num_workers,
        env_args=[[]] * num_workers,
        env_kwargs=worker_env_kwargs,
        backend="Multiprocessing",
        num_envs=num_workers,
        num_workers=num_workers,
        batch_size=num_workers,
        seed=args["vec"]["seed"],
    )
    scenario_progress = None
    try:
        agents_per_batch = vecenv.agents_per_batch
        inference_agents_per_batch = recorded_agents_per_batch or agents_per_batch
        if agents_per_batch > inference_agents_per_batch:
            raise pufferlib.APIUsageError(
                f"Replay environment batch has {agents_per_batch} agents, which exceeds the "
                f"CSV policy batch of {inference_agents_per_batch}. Reduce num_agents or the replay worker count."
            )
        if agents_per_batch < inference_agents_per_batch:
            print(
                f"Padding policy inference from {agents_per_batch} to "
                f"{inference_agents_per_batch} agents to preserve the recorded batch shape"
            )

        rollout_seed = args["train"]["seed"]
        torch.manual_seed(rollout_seed)
        if evaluation_policy_cache is None:
            evaluation_policy_cache = {"policy": policy}
        policy = evaluation_policy_cache["policy"]
        if policy is None:
            policy = load_policy(args, vecenv, env_name)
            evaluation_policy_cache["policy"] = policy
        policy.eval()
        if "policy_forward_eval" not in evaluation_policy_cache:
            policy_forward_eval = policy.forward_eval
            eval_sample_logits = pufferlib.pytorch.sample_logits
            if args["train"]["compile"]:
                compile_kwargs = {
                    "mode": args["train"]["compile_mode"],
                    "fullgraph": args["train"]["compile_fullgraph"],
                }
                policy_forward_eval = torch.compile(policy_forward_eval, **compile_kwargs)
                eval_sample_logits = torch.compile(eval_sample_logits, **compile_kwargs)
            evaluation_policy_cache["policy_forward_eval"] = policy_forward_eval
            evaluation_policy_cache["sample_logits"] = eval_sample_logits
        policy_forward_eval = evaluation_policy_cache["policy_forward_eval"]
        eval_sample_logits = evaluation_policy_cache["sample_logits"]
        # A discrete policy on a continuous env emits a discrete class that the
        # policy's own table maps back to the continuous action the env expects.
        action_selection = args["eval"]["action_selection"]
        uncompiled_policy = base_policy(policy)
        env_continuous = isinstance(vecenv.single_action_space, pufferlib.spaces.Box)
        discrete_policy_on_continuous_env = env_continuous and not uncompiled_policy.is_continuous
        device = torch_device(args["train"]["device"])
        use_bfloat16 = args["train"]["amp"] and args["train"]["precision"] == "bfloat16" and is_cuda_device(device)
        if use_bfloat16 and not torch.cuda.is_bf16_supported():
            raise pufferlib.APIUsageError("bfloat16 evaluation requires CUDA BF16 support")
        eval_amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16)
        obs, _ = vecenv.reset(rollout_seed)
        padding_agent_count = inference_agents_per_batch - agents_per_batch
        policy_obs_tensor = None
        if padding_agent_count:
            policy_obs_tensor = torch.zeros(
                (inference_agents_per_batch, *obs.shape[1:]),
                dtype=torch.as_tensor(obs).dtype,
                device=device,
            )
        recurrent_state = None
        if args["train"].get("use_rnn", False):
            recurrent_state = {
                "lstm_h": torch.zeros(inference_agents_per_batch, policy.hidden_size, device=device),
                "lstm_c": torch.zeros(inference_agents_per_batch, policy.hidden_size, device=device),
            }

        capture_batch_steps = worker_env_kwargs[0]["resample_frequency"]
        replay_capture = None
        if replay_output_dir is not None:
            replay_capture = drive_eval_replay.EvalReplayCapture(
                args,
                policy,
                replay_output_dir,
                capture_observations,
                num_workers,
                agents_per_batch,
                capture_batch_steps,
                episode_id_offset,
            )

        episode_summaries = []
        scenario_progress = tqdm(total=expected_episodes, desc=desc, unit="scenario")
        for _ in range(total_steps):
            with torch.no_grad(), eval_amp_context:
                environment_obs_tensor = torch.as_tensor(obs, device=device)
                if padding_agent_count:
                    policy_obs_tensor[:agents_per_batch].copy_(environment_obs_tensor)
                else:
                    policy_obs_tensor = environment_obs_tensor
                if recurrent_state is None:
                    logits, value = policy_forward_eval(policy_obs_tensor)
                else:
                    logits, value = policy_forward_eval(policy_obs_tensor, recurrent_state)
                action, logprob, entropy, cont_action = eval_sample_logits(
                    logits,
                    action_selection=action_selection,
                    env_continuous=env_continuous,
                    policy=uncompiled_policy,
                )
                if discrete_policy_on_continuous_env:
                    # raw_action stays the discrete class (what the replay logs record),
                    # while the env is stepped with its continuous counterpart.
                    raw_action = action[:agents_per_batch].cpu().numpy()
                    continuous_actions = cont_action.reshape(-1, *vecenv.single_action_space.shape)
                    action = continuous_actions[:agents_per_batch].float().cpu().numpy()
                else:
                    raw_action = action[:agents_per_batch].cpu().numpy().reshape(vecenv.action_space.shape)
                    action = raw_action
            if isinstance(logits, torch.distributions.Normal):
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

            if replay_capture is not None:
                replay_capture.capture_frame(
                    obs,
                    policy_obs_tensor,
                    raw_action,
                    action,
                    logits,
                    value,
                    logprob,
                    entropy,
                )

            obs, _, terminals, truncations, infos = vecenv.step(action)
            if recurrent_state is not None:
                finished_agent_mask = torch.as_tensor(
                    np.logical_or(terminals, truncations),
                    dtype=torch.bool,
                    device=device,
                ).reshape(agents_per_batch, 1)
                recurrent_state["lstm_h"][:agents_per_batch].masked_fill_(finished_agent_mask, 0)
                recurrent_state["lstm_c"][:agents_per_batch].masked_fill_(finished_agent_mask, 0)
            for worker_info in infos:
                worker_items = worker_info if isinstance(worker_info, list) else [worker_info]
                for item in worker_items:
                    if not isinstance(item, dict) or item.get("summary_type") != "evaluation_episode":
                        continue
                    if replay_capture is not None:
                        replay_capture.queue_replay(item, len(episode_summaries))
                    else:
                        item.pop("replay_environment_bundle", None)
                    item["agents_per_batch"] = inference_agents_per_batch
                    episode_summaries.append(item)
                    if len(episode_summaries) <= expected_episodes:
                        scenario_progress.update(1)

            if replay_capture is not None and replay_capture.pending_count:
                scenario_progress.set_postfix_str(f"writing {replay_capture.pending_count} replays")
                replay_capture.write_pending()
                scenario_progress.set_postfix_str("")

            if len(episode_summaries) >= expected_episodes:
                break
    finally:
        if scenario_progress is not None:
            scenario_progress.close()
        vecenv.close()
    if len(episode_summaries) != expected_episodes:
        print(
            f"WARNING: Evaluation expected {expected_episodes} episode summaries, "
            f"but received {len(episode_summaries)}. Writing the available results.",
            file=sys.stderr,
        )
    return episode_summaries


def run_training_evaluation(env_name, args, policy, logger, epoch, global_step, run_dir):
    """Run the configured evaluator and log its means on the training run."""
    eval_args = copy.deepcopy(args)
    eval_args["eval"]["benchmarks"] = eval_args["train"]["evaluation_benchmarks"]
    eval_args["eval"]["render_scenarios"] = False
    eval_args["eval"]["render_filter"] = None
    eval_args["eval"]["failure_replay_csv"] = None
    eval_output_dir = os.path.join(run_dir, "eval", "training")
    eval_output_subdir = f"epoch_{epoch:06d}_step_{global_step}"

    rng_state = capture_rng_state()
    policy_was_training = bool(getattr(policy, "training", False))
    try:
        benchmark_results = eval(
            env_name=env_name,
            args=eval_args,
            policy=policy,
            eval_output_dir=eval_output_dir,
            eval_output_subdir=eval_output_subdir,
            use_training_config=True,
        )
        metrics = drive_benchmark.summarize_benchmark_metrics(benchmark_results, TRAINING_EVAL_KEY_PREFIX)
        if metrics:
            logger.log(metrics, global_step)
        return benchmark_results
    except Exception:
        print(f"\n[training eval] Evaluation failed at epoch {epoch}; continuing training:")
        traceback.print_exc()
        return {}
    finally:
        if hasattr(policy, "train"):
            policy.train(policy_was_training)
        restore_rng_state({"rng_state": rng_state})


def _render_eval_failures(
    env_name,
    run_args,
    benchmark,
    metrics_path,
    benchmark_output_dir,
    policy,
    capture_observations,
    max_rendered_failures,
    evaluation_policy_cache=None,
):
    configured_render_filter = run_args["eval"]["render_filter"]
    selected_rows = drive_benchmark.select_render_rows(metrics_path, configured_render_filter)
    if max_rendered_failures is not None:
        selected_rows = selected_rows.head(max_rendered_failures).copy()
    failures_dir = os.path.join(benchmark_output_dir, "failures")
    os.makedirs(failures_dir, exist_ok=True)
    selected_path = os.path.join(failures_dir, "selected_failures.csv")
    selected_rows.to_csv(selected_path, index=False)
    if selected_rows.empty:
        print(f"No failures matched for benchmark {benchmark['name']}; wrote {selected_path}")
        return {"episodes": [], "summary": None}

    map_indices = drive_benchmark._resolve_map_indices(
        run_args["env"]["map_dir"],
        selected_rows["map_name"].tolist(),
    )
    seeds = pd.to_numeric(selected_rows["seed"], errors="raise").astype(np.int64).tolist()
    pairs = list(zip(map_indices, seeds))
    failure_args = copy.deepcopy(run_args)
    configured_worker_count = failure_args["vec"]["num_envs"]
    if configured_worker_count <= 0:
        raise pufferlib.APIUsageError("Failure rendering requires at least one worker")
    replay_wave_size = len(pairs)
    if capture_observations:
        observation_replay_wave_size = run_args["eval"]["observation_replay_wave_size"]
        if (
            isinstance(observation_replay_wave_size, bool)
            or not isinstance(observation_replay_wave_size, int)
            or observation_replay_wave_size <= 0
        ):
            raise pufferlib.APIUsageError(
                "eval.observation_replay_wave_size must be a positive integer when rendering observations"
            )
        replay_wave_size = min(
            len(pairs),
            configured_worker_count,
            observation_replay_wave_size,
        )
        replay_agent_capacity = failure_args["env"]["max_agents_per_env"]
        if replay_agent_capacity <= 0:
            raise pufferlib.APIUsageError("Failure rendering requires max_agents_per_env > 0")
        failure_args["env"]["num_agents"] = replay_agent_capacity
    replay_output_dir = os.path.join(failures_dir, "replays")
    os.makedirs(replay_output_dir, exist_ok=True)
    agents_per_batch_values = selected_rows["agents_per_batch"].unique()
    if len(agents_per_batch_values) != 1:
        raise pufferlib.APIUsageError("Benchmark failure rows must contain exactly one agents_per_batch value")
    recorded_agents_per_batch = int(agents_per_batch_values[0])
    summaries = []
    replay_wave_count = (len(pairs) + replay_wave_size - 1) // replay_wave_size
    for replay_wave_idx, replay_pair_start in enumerate(range(0, len(pairs), replay_wave_size)):
        replay_pairs = pairs[replay_pair_start : replay_pair_start + replay_wave_size]
        num_workers = min(configured_worker_count, len(replay_pairs))
        failure_args["vec"]["num_envs"] = num_workers
        worker_env_kwargs, total_steps = drive_benchmark._plan_failure_replay_workers(
            failure_args,
            replay_pairs,
            num_workers,
            failure_args["env"]["scenario_length"],
        )
        replay_desc = f"Rendering {benchmark['name']} failures"
        if replay_wave_count > 1:
            replay_desc += f" (wave {replay_wave_idx + 1}/{replay_wave_count})"
        wave_summaries = _run_eval_rollout(
            failure_args,
            env_name,
            worker_env_kwargs,
            total_steps,
            replay_desc,
            len(replay_pairs),
            policy=policy,
            recorded_agents_per_batch=recorded_agents_per_batch,
            replay_output_dir=replay_output_dir,
            capture_observations=capture_observations,
            episode_id_offset=len(summaries),
            evaluation_policy_cache=evaluation_policy_cache,
        )
        summaries.extend(wave_summaries)
    summary = drive_benchmark._write_eval_reports(summaries, failures_dir, len(pairs))
    drive_eval_replay._render_eval_replays(summaries, failures_dir)
    return {
        "episodes": summaries,
        "summary": summary,
    }


def load_env(env_name, args, seed=None):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)
    make_env = env_module.env_creator(env_name)
    vec_kwargs = dict(args["vec"])
    if seed is not None:
        vec_kwargs["seed"] = seed
    return pufferlib.vector.make(make_env, env_kwargs=args["env"], **vec_kwargs)


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
    if config_dir is None:
        puffer_dir = os.path.dirname(os.path.realpath(__file__))
        config_dir = os.path.join(puffer_dir, "config")

    # Everything left in argv after main() pops mode/env_name is a Hydra
    # override: train.learning_rate=1e-4, env.num_agents=512, wandb=true
    overrides = []
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            hint = arg.lstrip("-").replace("-", "_").split("=")[0]
            raise pufferlib.APIUsageError(f"'{arg}' uses the old flag syntax. Use Hydra overrides: {hint}=<value>")
        overrides.append(arg)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=env_name, overrides=overrides)

    # Structured-schema validation (types, enum names, unknown keys) for envs
    # that declare one. Overrides are already composed in, so CLI typos fail
    # here too — at load time, not deep in env construction.
    env_schema = ENV_SCHEMAS.get(env_name)
    if env_schema is not None:
        cfg["env"] = OmegaConf.merge(OmegaConf.structured(env_schema), cfg["env"])

    # Plain nested dict — the contract every downstream consumer relies on.
    # Protein's sweep.suggest() writes arbitrary keys into it, so no
    # struct-mode OmegaConf objects may leak past this point. enum_to_str
    # converts validated enum members back to their names; throw_on_missing
    # rejects schema keys the YAML no longer provides.
    args = defaultdict(dict, OmegaConf.to_container(cfg, resolve=True, enum_to_str=True, throw_on_missing=True))

    args["train"]["use_rnn"] = args["rnn_name"] is not None

    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        args["train"]["total_timesteps"] //= world_size

    return args


def main():
    err = "Usage: puffer [train, eval, sweep, controlled_exp, autotune, profile, export] [env_name] [optional args]. --help for more info"
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    elif mode == "eval":
        if len(sys.argv) < 2:
            raise pufferlib.APIUsageError("Usage: puffer eval [env_name] [benchmark_name] [optional args]")
        benchmark_name = sys.argv.pop(1)
        eval(env_name=env_name, benchmark_names=benchmark_name)
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
    else:
        raise pufferlib.APIUsageError(err)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
