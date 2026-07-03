## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import copy
import warnings

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
import pufferlib.evaluate
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch

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
        return pufferlib.evaluate._reduce_environment_metrics(self.stats)

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

        do_eval = self.epoch % config["eval"]["eval_interval"] == 0 or done_training
        if config["eval"]["wosac_realism_eval"] and do_eval:
            pufferlib.utils.run_wosac_eval_in_subprocess(config, self.logger, self.global_step)
        if config["eval"]["human_replay_eval"] and do_eval:
            pufferlib.utils.run_human_replay_eval_in_subprocess(config, self.logger, self.global_step)
        if config["eval"]["wosac_realism_eval"] and do_eval:
            pufferlib.utils.run_wosac_eval_in_subprocess(config, self.logger, self.global_step)
        if config["eval"]["human_replay_eval"] and do_eval:
            pufferlib.utils.run_human_replay_eval_in_subprocess(config, self.logger, self.global_step)

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
            self.amp_context.__enter__()  # TODO: AMP needs some debugging

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
                # Skip tiny permutation tail: Adam steps at full magnitude regardless of sample
                # count, so a near-empty minibatch is a noise step. Rank-symmetric under DDP
                # because num_kept is synced across ranks.
                if mb_idx.numel() < self.minibatch_size // 4:
                    continue
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
        self.stats = pufferlib.evaluate._reduce_environment_metrics(self.stats)

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
            "num_target_waypoints",
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
        visible_gpu_count = torch.cuda.device_count()
        if local_rank >= visible_gpu_count:
            raise RuntimeError(
                f"local_rank {local_rank} >= visible GPUs {visible_gpu_count}: "
                "node has fewer GPUs than processes (degraded node or bad nproc-per-node)"
            )
        torch.cuda.set_device(local_rank)

    train_seed = args["train"]["seed"]
    if train_seed is None:
        train_seed = time.time_ns() & 0xFFFFFFFF
    # DDP: identical seeds would make every rank roll out identical trajectories.
    # Wide stride so per-env seeds (env_idx + seed in vec_reset) never collide across ranks.
    train_seed += int(os.environ.get("LOCAL_RANK", 0)) * 100_000
    args["train"]["seed"] = train_seed
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

    if args["eval"]["benchmark"]:
        benchmark(
            env_name=env_name,
            args=args,
            policy=pufferl.uncompiled_policy,
            logger=pufferl.logger,
            quiet=True,
        )

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path, early_stop=False)
    return all_logs


def load_eval_config(env_name, model_path=None, eval_overrides=None):
    """Load config for evaluation, merging experiment YAML with defaults."""
    args = load_config(env_name)

    if model_path:
        experiment_dir = os.path.dirname(os.path.dirname(model_path))
        config_yaml_path = os.path.join(experiment_dir, "config.yaml")

        if os.path.exists(config_yaml_path):
            print(f"Found config.yaml at {config_yaml_path}. Merging with defaults...")
            with open(config_yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f)

            for section in ["env", "train", "policy", "rnn"]:
                if section in yaml_config and isinstance(yaml_config[section], dict):
                    for k, v in yaml_config[section].items():
                        args[section][k] = v

    if eval_overrides:
        for section, section_overrides in eval_overrides.items():
            if isinstance(section_overrides, dict):
                for k, v in section_overrides.items():
                    args[section][k] = v
            else:
                args[section] = section_overrides

    return args


def eval(env_name, args=None, vecenv=None, policy=None):
    """Evaluate a policy."""

    if args is None:
        tmp_args = load_config(env_name)
        model_path = tmp_args.get("load_model_path")
        eval_overrides = {
            "vec": {
                "num_envs": 1,
                "num_workers": 1,
                "batch_size": 1,
            },
            "env": {
                "dt": 0.1,
                "eval_mode": 1,
                "num_agents": 50,
                "min_agents_per_env": 50,
                "max_agents_per_env": 50,
                "reward_randomization": False,
                "scenario_length": 100,
                "resample_frequency": 100,
                "num_maps": 6,
            },
        }
        args = load_eval_config(env_name, model_path, eval_overrides)

    wosac_enabled = args["eval"]["wosac_realism_eval"]
    human_replay_enabled = args["eval"]["human_replay_eval"]
    args["env"]["map_dir"] = args["eval"]["map_dir"]
    dataset_name = args["env"]["map_dir"].split("/")[-1]

    if wosac_enabled:
        print(f"Running WOSAC realism evaluation with {dataset_name} dataset. \n")
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

        backend = args["eval"]["backend"]
        assert backend == "PufferEnv" or not wosac_enabled, "WOSAC evaluation only supports PufferEnv backend."
        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["num_agents"] = args["eval"]["wosac_num_agents"]
        args["env"]["init_mode"] = args["eval"]["wosac_init_mode"]
        args["env"]["control_mode"] = args["eval"]["wosac_control_mode"]
        args["env"]["init_step"] = args["eval"]["wosac_init_steps"]
        args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]

        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        evaluator = WOSACEvaluator(args)

        # Collect ground truth trajectories from the dataset
        gt_trajectories = evaluator.collect_ground_truth_trajectories(vecenv)

        # Roll out trained policy in the simulator
        simulated_trajectories = evaluator.collect_simulated_trajectories(args, vecenv, policy)

        print(f"\nCollected trajectories on {len(np.unique(gt_trajectories['scenario_id']))} scenarios.")

        if args["eval"]["wosac_sanity_check"]:
            evaluator._quick_sanity_check(gt_trajectories, simulated_trajectories)

        # Analyze and compute metrics
        agent_state = vecenv.driver_env.get_global_agent_state()
        road_edge_polylines = vecenv.driver_env.get_road_edge_polylines()
        results = evaluator.compute_metrics(
            gt_trajectories,
            simulated_trajectories,
            agent_state,
            road_edge_polylines,
            args["eval"]["wosac_aggregate_results"],
        )

        if args["eval"]["wosac_aggregate_results"]:
            import json

            print("WOSAC_METRICS_START")
            print(json.dumps(results))
            print("WOSAC_METRICS_END")

        return results

    elif human_replay_enabled:
        print(f"Running human replay evaluation with {dataset_name} dataset.\n")
        from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator

        backend = args["eval"].get("backend", "PufferEnv")
        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["num_agents"] = args["eval"]["human_replay_num_agents"]
        args["env"]["control_mode"] = args["eval"]["human_replay_control_mode"]
        args["env"]["scenario_length"] = 91  # Standard scenario length

        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        evaluator = HumanReplayEvaluator(args)

        # Run rollouts with human replays
        results = evaluator.rollout(args, vecenv, policy)

        import json

        print("HUMAN_REPLAY_METRICS_START")
        print(json.dumps(results))
        print("HUMAN_REPLAY_METRICS_END")

        return results
    else:  # Standard evaluation: Render
        backend = args["vec"]["backend"]
        if backend != "PufferEnv":
            backend = "Serial"

        args["vec"] = dict(backend=backend, num_envs=1)
        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        driver = vecenv.driver_env
        num_agents = vecenv.observation_space.shape[0]
        device = args["train"]["device"]

        # Rebuild visualize binary if saving frames (for C-based rendering)
        if args["save_frames"] > 0:
            ensure_drive_binary()

        if args["render_mode"] == "matplotlib":
            import mediapy
            import pufferlib.viz

            os.makedirs(args["video_path"], exist_ok=True)

            for i in range(args["num_scenarios"]):
                state = {}
                if args["train"]["use_rnn"]:
                    state = dict(
                        lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                        lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                    )

                ob, _ = vecenv.reset()
                sim_frames = []
                if args["render_obs"]:
                    frames_obs = []

                scenario = vecenv.get_state()[0]
                map_name = scenario["map_name"].split("/")[-1].split(".")[0]

                sim_video_path = f"{args['video_path']}/sim_{i}_{map_name}.mp4"
                if args["render_obs"]:
                    video_path_obs = f"{args['video_path']}/obs_{i}_{map_name}.mp4"

                print(f"Rendering episode {i} - map {map_name} to {sim_video_path}")

                for t in range(args["env"]["scenario_length"]):
                    scenario = vecenv.get_state()[0]  # TODO make env_indices configurable

                    sim_img = pufferlib.viz.plot_simulator_state(scenario, timestep=t, reuse_key=f"video_{i}")
                    if args["render_obs"]:
                        obs_img = pufferlib.viz.plot_observation(
                            ob,
                            target_type=args["env"]["target_type"],
                            reward_conditioning=args["env"]["reward_conditioning"],
                            num_target_waypoints=args["env"]["num_target_waypoints"],
                            max_partners=args["env"]["obs_slots_partners"],
                            max_lane_segments=args["env"]["obs_slots_lane"],
                            max_boundary_segments=args["env"]["obs_slots_boundary"],
                            obs_slots_traffic_controls=args["env"]["obs_slots_traffic_controls"],
                            obs_dropout_lane=args["env"].get("obs_dropout_lane", 0.0),
                            obs_dropout_boundary=args["env"].get("obs_dropout_boundary", 0.0),
                            traffic_control_scope=args["env"].get("traffic_control_scope", 0),
                        )

                    with torch.no_grad():
                        ob = torch.as_tensor(ob).to(device)
                        logits, _ = policy.forward_eval(ob, state)
                        action, _, _ = pufferlib.pytorch.sample_logits(logits)
                        action = action.cpu().numpy().reshape(vecenv.action_space.shape)

                    if isinstance(logits, torch.distributions.Normal):
                        action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

                    ob = vecenv.step(action)[0]

                    sim_frames.append(sim_img)
                    if args["render_obs"]:
                        frames_obs.append(obs_img)

                mediapy.write_video(sim_video_path, np.array(sim_frames), fps=20)
                if args["render_obs"]:
                    mediapy.write_video(video_path_obs, np.array(frames_obs), fps=20)
                pufferlib.viz.close_figure(f"video_{i}")
        else:
            frames = []
            driver = vecenv.driver_env
            ob, _ = vecenv.reset()

            while True:
                render = driver.render()
                if len(frames) < args["save_frames"]:
                    frames.append(render)
            while True:
                render = driver.render()
                if len(frames) < args["save_frames"]:
                    frames.append(render)

                # Screenshot Ocean envs with F12, gifs with control + F12
                if driver.render_mode == "ansi":
                    print("\033[0;0H" + render + "\n")
                    time.sleep(1 / args["fps"])
                elif driver.render_mode == "rgb_array":
                    pass
                    # import cv2
                    # render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
                    # cv2.imshow('frame', render)
                    # cv2.waitKey(1)
                    # time.sleep(1/args['fps'])

                with torch.no_grad():
                    ob = torch.as_tensor(ob).to(device)
                    logits, value = policy.forward_eval(ob, state)
                    action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                    action = action.cpu().numpy().reshape(vecenv.action_space.shape)

                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

                ob = vecenv.step(action)[0]

                if len(frames) > 0 and len(frames) == args["save_frames"]:
                    import imageio

                    imageio.mimsave(args["gif_path"], frames, fps=args["fps"], loop=0)
                    frames.append("Done")


def load_eval_multi_scenarios_config(env_name, model_path=None, eval_overrides=None):
    """Load configuration and merge with config.yaml and overrides."""
    args = load_config(env_name)
    if model_path:
        args["load_model_path"] = model_path
        config_yaml_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), "config.yaml")

        if os.path.exists(config_yaml_path):
            with open(config_yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}

            exclude_keys = set(eval_overrides["env"].keys()) if eval_overrides else set()
            for section in ["env", "policy", "rnn"]:
                if section in yaml_config and isinstance(yaml_config[section], dict):
                    args[section].update({k: v for k, v in yaml_config[section].items() if k not in exclude_keys})

            for key in ["rnn_name", "policy_name"]:
                if key in yaml_config:
                    args[key] = yaml_config[key]

            args["train"]["use_rnn"] = args["rnn_name"] is not None

    if eval_overrides:
        for section, overrides in eval_overrides.items():
            if isinstance(overrides, dict):
                args[section].update(overrides)
            else:
                args[section] = overrides

    return args


def _build_eval_run_args(env_name, model_path, suite, replay_expert_actions=False):
    """Build run arguments from a benchmark suite."""
    overrides = pufferlib.evaluate.build_eval_overrides(
        mode=suite["simulation_mode"],
        num_agents=suite["num_agents"],
        num_scenarios=suite["num_scenarios"],
        map_dir=suite["map_dir"],
        num_maps=suite["num_maps"],
        scenario_length=suite["scenario_length"],
        max_agents=suite["max_agents_per_env"],
        control_mode=suite["control_mode"],
    )
    args = load_eval_multi_scenarios_config(env_name, model_path, overrides)
    args["env"]["replay_expert_actions"] = replay_expert_actions
    args["num_scenarios"] = suite["num_scenarios"]
    args["eval_env_overrides"] = overrides["env"]
    return args


def _metrics_env_count(run_args, suite):
    # Failure replay sizes its batch from this same formula: replay logits are only
    # bit-identical to the metrics run if the two counts stay in sync.
    return min(run_args["vec"]["num_envs"], suite["num_scenarios"])


def _append_matching_failure_infos(all_infos, infos, failure_row):
    if not infos:
        return False

    map_index = int(failure_row["map_index"])
    seed = int(failure_row["Seed"])
    # Identity columns come from the original metrics run, not the replay batch.
    overrides = {"scenario_index": int(failure_row["scenario_index"])}
    episode_id = failure_row.get("episode_id")
    if episode_id is not None:
        overrides["episode_id"] = int(episode_id)

    map_values = infos.get("map_index")
    seed_values = infos.get("Seed")
    if map_values is None or seed_values is None:
        return False

    matched = False
    for row_idx in range(len(map_values)):
        if int(map_values[row_idx]) != map_index or int(seed_values[row_idx]) != seed:
            continue
        for key, values in infos.items():
            if key == "env_id":
                continue
            all_infos.setdefault(key, []).append(overrides[key] if key in overrides else values[row_idx])
        for key, value in overrides.items():
            if key not in infos:
                all_infos.setdefault(key, []).append(value)
        matched = True
    return matched


def _render_suite_failures(
    env_name, run_args, suite, eval_cfg, res_dir, summary_entry, policy, replay_expert_actions, quiet
):
    failure_csv = eval_cfg["failure_csv"] or os.path.join(res_dir, "episode_metrics.csv")
    failure_rows = pufferlib.evaluate.load_failure_rows(failure_csv)
    summary_entry["failure_csv"] = failure_csv
    summary_entry["num_failure_renders"] = len(failure_rows)

    if not quiet:
        print(f"🎥 Rendering {len(failure_rows)} failure scenario(s) from {failure_csv}...")

    failure_records = list(failure_rows.to_dict("records"))
    if not failure_records:
        return

    # Size the replay env to the metrics-eval policy batch (n_work envs * num_agents):
    # cuda kernel selection depends on batch rows, so matching the shape keeps logits
    # bit-identical to the original run. Pack as many failure worlds per pass as fit.
    metrics_batch_rows = _metrics_env_count(run_args, suite) * int(run_args["env"]["num_agents"])
    max_agents = int(run_args["env"]["max_agents_per_env"])
    chunk_size = max(1, metrics_batch_rows // max_agents)
    all_failure_infos = {}
    pol_inst = None
    for chunk_start in range(0, len(failure_records), chunk_size):
        chunk = failure_records[chunk_start : chunk_start + chunk_size]
        r_args = copy.deepcopy(run_args)
        r_args.update(
            {
                "render_obs": eval_cfg["render_obs"],
                "num_scenarios": len(chunk),
                "render_episode_id_map": [int(row["scenario_index"]) for row in chunk],
            }
        )
        r_args["env"]["num_agents"] = metrics_batch_rows
        r_args["env"]["num_eval_scenarios"] = len(chunk)
        r_args["env"]["eval_map_indices"] = [int(row["map_index"]) for row in chunk]
        r_args["env"]["eval_scenario_seeds"] = [int(row["Seed"]) for row in chunk]
        r_args["vec"].update(
            {
                "backend": "Serial",
                "num_envs": 1,
                "num_workers": 1,
                "batch_size": 1,
            }
        )

        vecenv_render = load_env(env_name, r_args)
        if pol_inst is None and not replay_expert_actions:
            pol_inst = policy or load_policy(r_args, vecenv_render, env_name)
        infos = pufferlib.evaluate.evaluation_render(
            r_args,
            vecenv_render,
            pol_inst,
            quiet=False,
            dump_metrics=False,
        )
        vecenv_render.close()
        for row in chunk:
            if not _append_matching_failure_infos(all_failure_infos, infos, row):
                raise pufferlib.APIUsageError(
                    f"Failure replay did not produce map_index={int(row['map_index'])}, Seed={int(row['Seed'])}."
                )

    if all_failure_infos:
        pufferlib.evaluate._export_metrics(
            all_failure_infos,
            res_dir,
            len(failure_rows),
            quiet=quiet,
            verify_coverage=False,
            filename="episode_metrics_failures.csv",
        )

    htmls = sorted(glob.glob(os.path.join(res_dir, "gif", "*.html")))
    if htmls:
        summary_entry["render_html"] = htmls[0]


def benchmark(env_name, args=None, policy=None, logger=None, quiet=False):
    """
    Run the final evaluation suite.
    - Normal eval uses catalog num_scenarios.
    - Render uses catalog num_scenarios_to_render.
    - Render-only uses catalog num_scenarios_to_render.
    """
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return []

    args = args or load_config(env_name)
    eval_cfg = args["eval"]
    render = eval_cfg["render"]
    render_only = eval_cfg["render_only"]
    render_obs = eval_cfg["render_obs"]
    render_failures_only = eval_cfg["render_failures_only"]

    # Resolve paths and suites
    _, bench_dir, model_path = pufferlib.evaluate._resolve_benchmark_context(env_name, args, logger, policy)
    replay_expert_actions = model_path is None and policy is None
    suites = pufferlib.evaluate._build_final_master_eval_suites(args)
    if replay_expert_actions:
        skipped = [s["suite_id"] for s in suites if s["simulation_mode"] != "replay"]
        suites = [s for s in suites if s["simulation_mode"] == "replay"]
        if skipped and not quiet:
            print(f"Skipping non-replay suites for expert benchmark: {', '.join(skipped)}")
    if not suites:
        raise pufferlib.APIUsageError("No final evaluation datasets configured under [eval]")

    def_agents = eval_cfg["num_agents"]
    sdc_envs = max(1, eval_cfg["benchmark_sdc_num_envs"])
    all_summaries = []

    if not quiet:
        print("\n🚀 Starting Final Master Evaluation Suite...")

    for suite in suites:
        if not quiet:
            print(f"\n📊 Processing Suite: {suite['suite_id']}")
        # Prepare Run Arguments
        suite["num_agents"] = def_agents
        run_args = _build_eval_run_args(env_name, model_path, suite, replay_expert_actions)
        res_dir = os.path.join(bench_dir, suite["suite_id"])
        os.makedirs(res_dir, exist_ok=True)
        run_args["eval_results_dir"] = res_dir

        # Throttle CPUs for SDC mode
        if suite["simulation_mode"] == "replay" and suite["control_mode"] == "control_sdc_only":
            max_envs = min(sdc_envs, psutil.cpu_count(logical=False) or sdc_envs)
            run_args["vec"]["num_envs"] = min(run_args["vec"]["num_envs"], max_envs)

        summary_entry = {
            "suite_id": suite["suite_id"],
            "suite_name": suite["name"],
            "mode": suite["simulation_mode"],
            "map_dir": suite["map_dir"],
            "results_dir": res_dir,
        }

        if render_failures_only:
            _render_suite_failures(
                env_name, run_args, suite, eval_cfg, res_dir, summary_entry, policy, replay_expert_actions, quiet
            )
            all_summaries.append(summary_entry)
            continue

        if render_only:
            run_args["env"]["num_eval_scenarios"] = suite["num_scenarios_to_render"]
            print(f"🎥 Render-Only mode for {run_args['env']['num_eval_scenarios']} scenarios...")

            run_args["env"]["starting_map"] = 0
            run_args["render_obs"] = render_obs
            run_args["num_scenarios"] = suite["num_scenarios_to_render"]
            run_args["vec"].update(
                {
                    "backend": "Serial" if run_args["vec"]["backend"] != "PufferEnv" else run_args["vec"]["backend"],
                    "num_envs": 1,
                    "num_workers": 1,
                    "batch_size": 1,
                }
            )
            vecenv = load_env(env_name, run_args)
            pol_inst = None if replay_expert_actions else policy or load_policy(run_args, vecenv, env_name)
            pufferlib.evaluate.evaluation_render(
                run_args,
                vecenv,
                pol_inst,
                quiet=False,
                dump_metrics=True,
            )
            vecenv.close()

        else:
            if not quiet:
                print(f"⚡ Computing metrics for {suite['num_scenarios']} scenarios...")
            pkg = run_args["package"]
            make_env = importlib.import_module(
                "pufferlib.ocean" if pkg == "ocean" else f"pufferlib.environments.{pkg}"
            ).env_creator(env_name)

            # Cap workers to num_scenarios and build kwargs list
            n_work = _metrics_env_count(run_args, suite)
            rem, spw = suite["num_scenarios"] % n_work, suite["num_scenarios"] // n_work

            kwargs_list, curr_st = [], 0
            for j in range(n_work):
                c = spw + (1 if j < rem else 0)
                kw = copy.deepcopy(run_args["env"])
                kw.update({"starting_map": curr_st, "num_eval_scenarios": c})
                kwargs_list.append(kw)
                curr_st += c

            # Update vec configuration for safety
            run_args["vec"].update({"num_envs": n_work, "num_workers": n_work, "batch_size": n_work})
            vecenv_metrics = pufferlib.vector.make(
                [make_env] * n_work, env_args=[[]] * n_work, env_kwargs=kwargs_list, **run_args["vec"]
            )
            pol_inst = None if replay_expert_actions else policy or load_policy(run_args, vecenv_metrics, env_name)

            avg_infos = pufferlib.evaluate.evaluation_metrics(run_args, vecenv_metrics, pol_inst, quiet=quiet)
            vecenv_metrics.close()

            if avg_infos:
                summary_entry.update(avg_infos)

            if render:
                n_render = suite["num_scenarios_to_render"]
                print(f"🎥 Rendering {n_render} scenario(s) for visualization of agent's behavior ...")
                r_args = copy.deepcopy(run_args)
                r_args.update({"render_obs": render_obs, "num_scenarios": n_render})

                r_args["vec"].update(
                    {
                        "backend": "Serial" if r_args["vec"]["backend"] != "PufferEnv" else r_args["vec"]["backend"],
                        "num_envs": 1,
                        "num_workers": 1,
                        "batch_size": 1,
                    }
                )
                r_args["env"]["num_eval_scenarios"] = n_render

                vecenv_render = load_env(env_name, r_args)
                pufferlib.evaluate.evaluation_render(
                    r_args,
                    vecenv_render,
                    pol_inst,
                    quiet=False,  # TODO - true in GCP only
                    dump_metrics=False,
                )
                vecenv_render.close()

                htmls = sorted(glob.glob(os.path.join(res_dir, "gif", "*.html")))
                if htmls:
                    summary_entry["render_html"] = htmls[0]

        all_summaries.append(summary_entry)

    # Master Summary
    if all_summaries:
        csv_path = os.path.join(bench_dir, "master_evaluation_summary.csv")
        df = pufferlib.evaluate._merge_master_benchmark_summary(csv_path, all_summaries)
        df.to_csv(csv_path, index=False)
        if not quiet:
            print(f"\n✅ Final evaluation complete! Master summary saved to: {csv_path}")


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


def export_onnx(args=None, env_name=None, vecenv=None, policy=None, path=None, silent=False):
    tmp_args = load_config(env_name)
    model_path = tmp_args.get("load_model_path")
    eval_overrides = {
        "env": {
            "simulation_mode": "replay",
            "scenario_length": 91,
            "map_dir": "pufferlib/resources/drive/binaries/1000",
        },
    }
    args = load_eval_multi_scenarios_config(env_name, model_path, eval_overrides)
    # args = args or load_config(env_name)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)
    device = "cpu"  # args["train"]["device"]

    # Set the model to evaluation mode
    policy.eval()
    policy = policy.to(device)

    # Create a dummy input from the observation space
    obs_space = vecenv.single_observation_space
    dummy_obs = torch.ones(
        (1,) + obs_space.shape,
        dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict.get(obs_space.dtype, torch.float32),
    ).to(device)

    if path is None:
        path = args["load_model_path"].replace(".pt", ".onnx")

    if not args["train"]["use_rnn"]:
        # Non-recurrent model
        class OnnxWrapper(torch.nn.Module):
            """ONNX export wrapper that bypasses torch.distributions.Normal.

            Why this is needed:
            - torch.distributions.Normal isn't ONNX-exportable because it has
              data-dependent validation guards (checking std > 0)
            - During ONNX tracing, these guards fail with:
              "GuardOnDataDependentSymNode: Could not guard on data-dependent expression"
            - We only need loc and std tensors for inference anyway, so we compute
              them directly without creating the distribution object
            """

            def __init__(self, policy):
                super().__init__()
                super().eval()
                self.policy = policy

            def forward(self, obs):
                logits, value = self.policy.forward(obs, state=None)
                return logits.loc, logits.scale, value

        model_to_export = OnnxWrapper(policy)
        dummy_input = (dummy_obs,)
        input_names = ["input"]
        output_names = ["logits_loc", "logits_scale", "value"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "logits_loc": {0: "batch_size"},
            "logits_scale": {0: "batch_size"},
            "value": {0: "batch_size"},
        }
    else:
        # Recurrent model (LSTM)
        class OnnxRnnWrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy  # This is an LSTMWrapper

            def forward(self, obs, h_in, c_in):
                state = {"lstm_h": h_in, "lstm_c": c_in}
                # Use forward_eval for inference
                logits, value = self.policy.forward_eval(obs, state)
                h_out = state["lstm_h"]
                c_out = state["lstm_c"]
                return logits.loc, logits.scale, value, h_out, c_out

        model_to_export = OnnxRnnWrapper(policy)
        hidden_size = policy.hidden_size
        dummy_h = torch.ones(1, hidden_size).to(device)
        dummy_c = torch.ones(1, hidden_size).to(device)
        dummy_input = (dummy_obs, dummy_h, dummy_c)
        input_names = ["input", "h_in", "c_in"]
        output_names = ["logits_loc", "logits_scale", "value", "h_out", "c_out"]
        dynamic_axes = {
            "input": {0: "batch_size"},
            "h_in": {0: "batch_size"},
            "c_in": {0: "batch_size"},
            "logits_loc": {0: "batch_size"},
            "logits_scale": {0: "batch_size"},
            "value": {0: "batch_size"},
            "h_out": {0: "batch_size"},
            "c_out": {0: "batch_size"},
        }

    # Export the model using legacy exporter for opset 12 compatibility
    # dynamo=False forces the legacy TorchScript-based exporter instead of
    # the newer dynamo-based one which may require higher opset versions
    torch.onnx.export(
        model_to_export,
        dummy_input,
        path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    if not silent:
        print(f"Model exported to {path}")

    # Check that the ONNX model output matches the PyTorch model output
    check_valid_onnx(path, model_to_export, dummy_input, silent)

    # Clean up the vectorized environment (terminates worker processes)
    vecenv.close()

    return None


def check_valid_onnx(path, model_to_export, dummy_input, silent):
    try:
        import onnxruntime
        import numpy as np

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])

        with torch.no_grad():
            if isinstance(dummy_input, tuple):
                pytorch_outs = model_to_export(*dummy_input)
            else:
                pytorch_outs = model_to_export(dummy_input)

        if isinstance(dummy_input, tuple):
            ort_inputs = {ort_session.get_inputs()[i].name: to_numpy(dummy_input[i]) for i in range(len(dummy_input))}
        else:
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}

        ort_outs = ort_session.run(None, ort_inputs)

        for i, (pytorch_out, ort_out) in enumerate(zip(pytorch_outs, ort_outs)):
            np.testing.assert_allclose(to_numpy(pytorch_out), ort_out, rtol=1e-03, atol=1e-05)

        if not silent:
            print("ONNX model verification successful: outputs match PyTorch model outputs.")

    except ImportError:
        if not silent:
            print("Skipping ONNX model verification: onnxruntime is not installed.")
    except Exception as e:
        if not silent:
            print(f"ONNX model verification failed: {e}")


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
        "--render-mode",
        type=str,
        default="matplotlib",
        choices=["auto", "human", "ansi", "rgb_array", "raylib", "matplotlib", "None"],
    )
    parser.add_argument("--video-path", type=str, default="videos", help="Path to save videos")
    parser.add_argument("--num_scenarios", type=int, default=3, help="Number of scenarios to eval")
    parser.add_argument("--num_maps", type=int, default=1, help="Number of maps to use in gigaflow mode")
    parser.add_argument("--render", type=int, default=0, help="Rendering the evaluation")
    parser.add_argument(
        "--render_obs", type=int, default=0, help="Rendering the observation of first agent in evaluation"
    )
    parser.add_argument("--agent_index", nargs="*", type=int, default=None, help="Agent index to plot the observation")
    parser.add_argument("--save-frames", type=int, default=0)
    parser.add_argument("--gif-path", type=str, default="eval.gif")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-runs", type=int, default=200, help="Max number of sweep runs")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--wandb-group", type=str, default="debug")
    parser.add_argument("--neptune", action="store_true", help="Use neptune for logging")
    parser.add_argument("--neptune-name", type=str, default="pufferai")
    parser.add_argument("--neptune-project", type=str, default="ablations")
    parser.add_argument("--tb", action="store_true", help="Use tensorboard for logging")
    parser.add_argument("--local-rank", type=int, default=0, help="Used by torchrun for DDP")
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment")
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
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
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
    err = "Usage: puffer [train, eval, benchmark, sweep, controlled_exp, autotune, profile, export] [env_name] [optional args]. --help for more info"
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    elif mode == "eval":
        eval(env_name=env_name)
    elif mode == "benchmark":
        benchmark(env_name=env_name)
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
    elif mode == "export_onnx":
        export_onnx(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
