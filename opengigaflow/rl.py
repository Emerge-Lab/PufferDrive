"""PPO training and benchmarks: python -m opengigaflow.rl train | eval <benchmark> [overrides]; DDP via torchrun."""

import contextlib
import copy
import json
import math
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

import opengigaflow
from opengigaflow.ocean import torch as drive_torch
from opengigaflow.ocean.cosim import arch as drive_arch
from opengigaflow.ocean.drive.drive import Drive
from opengigaflow.ocean.evaluation_utils import evaluation_utils as drive_benchmark
from opengigaflow.ocean.evaluation_utils import eval_replay as drive_eval_replay
from opengigaflow.ocean.evaluation_utils import eval_wandb_report as drive_eval_report
import opengigaflow.utils
import opengigaflow.vector
import opengigaflow.pytorch
from opengigaflow.pytorch import clean_policy_state_dict
from opengigaflow.config_schema import (
    MAX_C_SEED,
    normalize_opengigaflow_config,
    validate_opengigaflow_config,
    validate_opengigaflow_resources,
)


try:
    from opengigaflow import _C
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

# assumes the op was built with its CUDA kernel whenever torch finds a CUDA / HIP toolkit at import (as setup.py does)
ADVANTAGE_CUDA = bool(CUDA_HOME or ROCM_HOME)
TRAINER_STATE_VERSION = 2
HIDDEN_DASHBOARD_METRICS = {
    "comfort_score",
    "driving_direction_score",
    "making_progress_rate",
    "multi_lane_score",
    "multi_lane_time",
    "speed_limit_compliance",
}
# sums over one logging window: their size follows the logging cadence, so they are not charted
UNCHARTED_WINDOW_SUM_METRICS = {"n", "total_distance_travelled", "total_infractions"}
FLOPS_PER_TFLOP = 1e12
ENV_NAME = "opengigaflow"
LOG_INTERVAL_SECONDS = 0.25
MAX_DASHBOARD_USER_STATS = 30
LABEL_COLOR = "[cyan]"
UNIT_COLOR = "[dim default]"
TITLE_COLOR = "[bright_cyan]"
VALUE_COLOR = "[default]"
# env settings a train-mode load_model_path run takes from the checkpoint's config.yaml with the network
FINE_TUNE_ENV_KEYS = {
    "goal_source",
    "goal_regen_mode",
    "num_goals",
    "min_goal_spacing",
    "max_goal_spacing",
    "reward_conditioning",
    "reward_randomization",
    "obs_slots_boundary_n",
    "obs_slots_lane_n",
    "obs_boundary_stride",
    "obs_lane_stride",
    "obs_stride_merge_segments",
    "obs_dropout_boundary",
    "obs_dropout_lane",
    "obs_slots_partners_n",
    "obs_slots_traffic_controls_n",
    "obs_traffic_light_yellow_remaining",
    "traffic_lights_enabled",
    "stop_signs_enabled",
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


def logits_to_float(logits):
    if isinstance(logits, torch.distributions.Normal):
        return torch.distributions.Normal(logits.loc.float(), logits.scale.float())
    return logits.float()


class PPOTrainer:
    def __init__(self, config, vecenv, policy, logger=None):
        allow_tf32 = config["tf32"]
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.deterministic = config["torch_deterministic"]
        torch.backends.cudnn.benchmark = not config["torch_deterministic"]
        torch.use_deterministic_algorithms(config["torch_deterministic"], warn_only=True)

        seed = config["seed"]
        # Decorrelate reset streams across DDP ranks
        if seed is not None and torch.distributed.is_initialized():
            seed = seed * torch.distributed.get_world_size() + torch.distributed.get_rank()

        vecenv.async_reset(seed)

        obs_space = vecenv.single_observation_space
        # Custom policy attributes live on the base module, not the DDP/compile wrapper.
        unwrapped_policy = base_policy(policy)
        if not unwrapped_policy.is_continuous:
            action_shape = ()
            action_dtype = torch.int32
        else:
            action_shape = vecenv.single_action_space.shape
            action_dtype = opengigaflow.pytorch.numpy_to_torch_dtype_dict[vecenv.single_action_space.dtype]
        total_agents = vecenv.num_agents
        self.total_agents = total_agents

        if config["batch_size"] == "auto":
            config["batch_size"] = total_agents * config["bptt_horizon"]
        elif config["bptt_horizon"] == "auto":
            config["bptt_horizon"] = config["batch_size"] // total_agents

        batch_size = config["batch_size"]
        horizon = config["bptt_horizon"]
        segments = batch_size // horizon
        self.segments = segments

        device = config["device"]
        precision = config["precision"]
        use_cuda = is_cuda_device(device)
        if precision == "bfloat16" and use_cuda and not torch.cuda.is_bf16_supported():
            raise opengigaflow.APIUsageError("bfloat16 precision requires a CUDA device with bf16 support")
        fp32_heads = getattr(unwrapped_policy, "fp32_heads", False)
        if precision == "bfloat16" and not allow_tf32 and not fp32_heads:
            raise opengigaflow.APIUsageError(
                "train.tf32=False with train.precision=bfloat16 requires policy.fp32_heads=true"
            )

        obs_dtype = getattr(torch, config["rollout_dtype"])
        compress_observations = config["rollout_dtype"] != "float32"
        transport_dtype = getattr(vecenv, "observation_transport_dtype", obs_space.dtype)
        if transport_dtype != obs_space.dtype and not compress_observations:
            raise opengigaflow.APIUsageError(
                f"Observations arrive as {transport_dtype}; this needs train.rollout_dtype={transport_dtype}"
            )

        self.observations = torch.zeros(
            segments,
            horizon,
            *obs_space.shape,
            dtype=obs_dtype,
            device=device,
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
        self.terminals = torch.zeros(segments, horizon, device=device, dtype=torch.bool)
        self.masks = torch.zeros(segments, horizon, device=device, dtype=torch.bool)
        self.ep_lengths = torch.zeros(total_agents, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, dtype=torch.int32)
        self.free_idx = total_agents

        minibatch_size = config["minibatch_size"]
        max_minibatch_size = config["max_minibatch_size"]
        self.minibatch_size = min(minibatch_size, max_minibatch_size)
        self.accumulate_minibatches = max(1, minibatch_size // max_minibatch_size)

        self.uncompiled_policy = unwrapped_policy
        self.policy = policy
        if config["compile"]:
            compile_kwargs = {
                "mode": config["compile_mode"],
                "fullgraph": config["compile_fullgraph"],
            }
            self.policy = torch.compile(policy, **compile_kwargs)
            self.policy.forward_eval = torch.compile(self.uncompiled_policy.forward_eval, **compile_kwargs)
            opengigaflow.pytorch.sample_logits = torch.compile(opengigaflow.pytorch.sample_logits, **compile_kwargs)

        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config["learning_rate"],
            betas=(config["adam_beta1"], config["adam_beta2"]),
            eps=config["adam_eps"],
            weight_decay=config["adam_weight_decay"],
        )
        self.optimizer = optimizer

        critic_params = [param for name, param in self.policy.named_parameters() if "critic" in name]
        critic_param_ids = {id(param) for param in critic_params}
        actor_params = [param for param in self.policy.parameters() if id(param) not in critic_param_ids]
        self.separate_grad_clip = bool(config["separate_grad_clip"] and critic_params and actor_params)
        self.actor_params = actor_params
        self.critic_params = critic_params

        self.logger = logger
        if logger is None:
            self.logger = NoLogger(config)

        self.forward_flops_per_transition, self.update_flops_per_transition = (
            unwrapped_policy.matmul_flops_per_transition()
        )
        self.peak_flops_per_second = None
        if use_cuda:
            from opengigaflow.tensor_core_peak import measure_tensor_core_peak_flops_per_second

            self.peak_flops_per_second = measure_tensor_core_peak_flops_per_second(device)
            distributed = torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if distributed else 0
            world_size = torch.distributed.get_world_size() if distributed else 1
            peak_tflops = self.peak_flops_per_second / FLOPS_PER_TFLOP
            gpu_name = torch.cuda.get_device_name(device)
            print(f"rank {rank}: {gpu_name} tensor-core peak {peak_tflops:.1f} TFLOP/s (bf16, fp32 accumulation)")
            self.logger.log_config({"peak_tensor_core_tflops": dist_sum(peak_tflops, device) / world_size})
        self.rollout_seconds = 0.0
        self.update_seconds = 0.0
        self.trained_transition_count = 0

        epochs = config["total_timesteps"] // config["batch_size"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.amp_context = contextlib.nullcontext()
        if use_cuda and precision == "bfloat16":
            self.amp_context = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, precision))

        self.config = config
        self.vecenv = vecenv
        # Workers already exist; registration belongs to this rank's CUDA context.
        if use_cuda and hasattr(vecenv, "pin_observations"):
            with torch.cuda.device(device):
                vecenv.pin_observations()
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.utilization = Utilization()
        self.profile = Profile()
        self.stats = {}
        self.env_metric_sums = {}
        self.last_stats = {}
        self.losses = {}
        self.best_score = -float("inf")
        self.ema_max = 0.0

        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        self.print_dashboard(clear=True)

    def load_training_state(self, path):
        device = torch_device(self.config["device"])
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get("state_version") != TRAINER_STATE_VERSION:
            raise ValueError(f"{path} is not a state_version {TRAINER_STATE_VERSION} trainer state")
        self.uncompiled_policy.load_state_dict(clean_policy_state_dict(state["policy_state_dict"]))
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.last_log_step = self.global_step
        self.best_score = state["best_score"]
        self.ema_max = state["ema_max"]
        restore_rng_state(state["rng_state"])
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
        rollout_start_time = time.perf_counter()
        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)

        config = self.config
        device = config["device"]
        if hasattr(self.vecenv, "bind_driver_thread"):
            self.vecenv.bind_driver_thread()

        full_rows = 0
        while full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            self.global_step += env_id.stop - env_id.start

            profile("eval_copy", epoch)
            # .float() is a no-op for float32 rollouts and widens float16 ones
            o_device = torch.as_tensor(o).to(device=device, dtype=self.observations.dtype, non_blocking=True).float()
            r = torch.as_tensor(r).to(device)
            d = torch.as_tensor(d).to(device)
            t = torch.as_tensor(t).to(device)
            m = torch.as_tensor(mask).to(device)
            done_mask = (d + t).clamp(max=1.0)

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                logits, value = self.policy.forward_eval(o_device)
                logits = logits_to_float(logits)
                action, logprob, _, cont_action = opengigaflow.pytorch.sample_logits(
                    logits, env_continuous=True, policy=self.uncompiled_policy
                )
                if config["normalize_rewards"]:
                    r = torch.sign(r) * torch.log1p(torch.abs(r))

            profile("eval_copy", epoch)
            with torch.no_grad():
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1 + self.ep_indices[env_id.stop - 1].item())

                self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob.float()
                # Truncated envs expose the pre-reset state here, so V(o) is the bootstrap value (unclipped).
                if config["use_value_bootstrapping"]:
                    trunc_mask = (t > 0) & (d == 0)
                    r = r + trunc_mask.to(r.dtype) * config["gamma"] * value.flatten().float()
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = done_mask.bool()
                self.values[batch_rows, l] = value.flatten().float()
                self.masks[batch_rows, l] = m

                self.ep_lengths[env_id] += 1
                if l + 1 >= config["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, dtype=torch.int32)
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    full_rows += num_full

            profile("eval_misc", epoch)
            for i in info:
                # every completed episode counts once: weight each env log by its episode count
                weight = float(i.get(opengigaflow.utils.EPISODE_COUNT_KEY, 1.0)) if isinstance(i, dict) else 1.0
                for k, v in opengigaflow.unroll_nested_dict(i):
                    opengigaflow.utils.accumulate_environment_metric(self.env_metric_sums, k, v, weight)

            profile("env", epoch)

            if not self.uncompiled_policy.is_continuous:
                cont_action = cont_action.cpu().numpy()
                self.vecenv.send(cont_action)
            else:
                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)
                self.vecenv.send(action)

        profile("eval_misc", epoch)
        if hasattr(self.vecenv, "release_driver_thread"):
            self.vecenv.release_driver_thread()
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, dtype=torch.int32)
        self.ep_lengths.zero_()
        profile.end()
        # no synchronize needed: the last action copy to the CPU already waited for the GPU
        self.rollout_seconds += time.perf_counter() - rollout_start_time

    @record
    def train(self):
        update_start_time = time.perf_counter()
        profile = self.profile
        epoch = self.epoch
        config = self.config
        profile("train", epoch)
        profile("train_misc", epoch, nest=True)
        losses = defaultdict(float)
        self.trained_transition_count += self._train_ppo_transition(losses, profile, epoch)

        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()

        profile.end()
        # no synchronize needed: the explained-variance read already waited for the GPU
        self.update_seconds += time.perf_counter() - update_start_time
        self.epoch += 1
        done_training = self.global_step >= config["total_timesteps"]
        should_log = done_training or time.time() > self.last_log_time + LOG_INTERVAL_SECONDS
        if torch.distributed.is_initialized():
            should_log_flag = torch.tensor([1 if should_log else 0], device=config["device"], dtype=torch.int32)
            torch.distributed.broadcast(should_log_flag, src=0)
            should_log = bool(should_log_flag.item())
        if should_log:
            self.losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            self.mean_and_log()
            self.print_dashboard()
            self.stats = {}
            self.env_metric_sums = {}
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            self.rollout_seconds = 0.0
            self.update_seconds = 0.0
            self.trained_transition_count = 0
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()

    def _current_ent_coef(self):
        config = self.config
        ent_coef = config["ent_coef"]
        if not config["ent_coef_anneal"]:
            return ent_coef
        progress = min(self.global_step / max(config["total_timesteps"], 1), 1.0)
        start_frac = config["ent_coef_anneal_start_frac"]
        if progress <= start_frac:
            return ent_coef
        anneal_frac = (progress - start_frac) / max(1.0 - start_frac, 1e-8)
        return ent_coef + anneal_frac * (config["ent_coef_final"] - ent_coef)

    def _ppo_loss(self, mb_obs, mb_actions, mb_logprobs, mb_values, mb_returns, mb_adv):
        config = self.config
        mb_obs = mb_obs.float()
        with self.amp_context:
            logits, newvalue = self.policy(mb_obs)
        logits = logits_to_float(logits)
        newvalue = newvalue.float()
        _, newlogprob, entropy, _ = opengigaflow.pytorch.sample_logits(logits, action=mb_actions)

        newlogprob = newlogprob.float().view_as(mb_logprobs)
        newvalue = newvalue.view_as(mb_returns)
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()
            if torch.distributed.is_initialized():
                # This mean computation assumes that all GPUs use the same batch size. This is currently guaranteed.
                world_size = torch.distributed.get_world_size()
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

        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        if config["vf_clip_coef"] is not None:
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -config["vf_clip_coef"], config["vf_clip_coef"])
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = (0.5 * (newvalue - mb_returns) ** 2).mean()
        entropy_loss = entropy.mean()
        ent_coef = self._current_ent_coef()
        loss = pg_loss + config["vf_coef"] * v_loss - ent_coef * entropy_loss

        return loss, {
            "policy_loss": pg_loss.detach(),
            "value_loss": v_loss.detach(),
            "entropy": entropy_loss.detach(),
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
        }

    def _clip_gradients(self, losses):
        # train() converts these logging-only scalars after the optimizer updates.
        max_grad_norm = self.config["max_grad_norm"]
        if self.separate_grad_clip:
            losses["actor_grad_norm"] = torch.nn.utils.clip_grad_norm_(self.actor_params, max_grad_norm).detach()
            losses["critic_grad_norm"] = torch.nn.utils.clip_grad_norm_(self.critic_params, max_grad_norm).detach()
        else:
            losses["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm).detach()

    def _compute_advantages(self):
        config = self.config
        masks = self.masks
        terminals = (self.terminals | ~masks).float()
        advantages = compute_puff_advantage(self.values, self.rewards, terminals, config["gamma"], config["gae_lambda"])
        advantages = advantages.masked_fill(~masks, 0.0)
        return advantages, advantages + self.values, masks

    def _train_ppo_transition(self, losses, profile, epoch):
        config = self.config
        device = config["device"]

        advantages, returns, masks = self._compute_advantages()

        obs_shape = self.vecenv.single_observation_space.shape
        flat_obs = self.observations.reshape(-1, *obs_shape)
        flat_actions = self.actions.reshape(-1, *self.actions.shape[2:])
        flat_logprobs = self.logprobs.reshape(-1)
        flat_values = self.values.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_masks = masks.reshape(-1)
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
            if config["adv_filter_leak_fraction"] > 0.0:
                keep_mask |= torch.rand_like(valid_abs_adv) < config["adv_filter_leak_fraction"]
            keep_idx = valid_idx[keep_mask]
            num_valid, num_kept = valid_idx.numel(), keep_idx.numel()

            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                # Ranks keep different counts: trim to the global minimum so minibatch counts stay in sync
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
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                # Ranks mask different row counts: trim to the global minimum so minibatch counts stay in sync
                kept_tensor = torch.tensor([keep_idx.numel()], device=device)
                torch.distributed.all_reduce(kept_tensor, op=torch.distributed.ReduceOp.MIN)
                keep_idx = keep_idx[torch.randperm(keep_idx.numel(), device=device)[: kept_tensor.item()]]

        if config["min_batch_size"] is not None:
            num_missing = config["min_batch_size"] - keep_idx.numel()
            if num_missing > 0 and keep_idx.numel() > 0:
                pad_idx = torch.randint(keep_idx.numel(), (num_missing,), device=keep_idx.device)
                keep_idx = torch.cat([keep_idx, keep_idx[pad_idx]])

        self.optimizer.zero_grad()
        total_minibatches = 0
        retained_transition_count = keep_idx.numel()
        # A lone trailing transition has no sample std to normalize its advantage with (NaN), so it sits out this update
        if retained_transition_count % self.minibatch_size == 1:
            retained_transition_count -= 1
        optimizer_batch_size = self.minibatch_size * self.accumulate_minibatches
        if config["discard_partial_minibatch"] and retained_transition_count > optimizer_batch_size:
            retained_transition_count -= retained_transition_count % optimizer_batch_size

        for _ in range(config["update_epochs"]):
            permutation = keep_idx[torch.randperm(keep_idx.numel(), device=keep_idx.device)]
            for start in range(0, retained_transition_count, self.minibatch_size):
                profile("train_copy", epoch)
                mb_idx = permutation[start : start + self.minibatch_size]
                mb_obs = flat_obs[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_logprobs = flat_logprobs[mb_idx]
                mb_values = flat_values[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_adv = flat_advantages[mb_idx]

                profile("train_forward", epoch)
                loss, stats = self._ppo_loss(
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
                accumulation_start = (start // optimizer_batch_size) * optimizer_batch_size
                accumulation_end = min(accumulation_start + optimizer_batch_size, retained_transition_count)
                # Mean losses contribute in proportion to their samples, including a partial final group.
                if self.accumulate_minibatches > 1:
                    loss = loss * (mb_idx.numel() / (accumulation_end - accumulation_start))
                loss.backward()
                total_minibatches += 1

                # Flush at the end of each PPO epoch so updates never span two permutations.
                if start + self.minibatch_size < accumulation_end:
                    continue
                self._clip_gradients(losses)
                self.optimizer.step()
                self.optimizer.zero_grad()

        if total_minibatches > 0:
            for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"):
                losses[key] /= total_minibatches

        y_pred = flat_values[valid_idx]
        y_true = flat_returns[valid_idx]
        var_y = y_true.var(unbiased=False)
        losses["explained_variance"] = (
            float("nan") if var_y == 0 else (1 - (y_true - y_pred).var(unbiased=False) / var_y).item()
        )
        return config["update_epochs"] * retained_transition_count

    def flop_window(self):
        """This rank's (model FLOPs, FLOPs at its measured peak rate) since the last log, per logged MFU key."""
        if self.peak_flops_per_second is None or self.global_step == self.last_log_step:
            return {}
        rollout_flops = (self.global_step - self.last_log_step) * self.forward_flops_per_transition
        update_flops = self.trained_transition_count * self.update_flops_per_transition
        window_seconds = time.time() - self.last_log_time
        return {
            "MFU": (rollout_flops + update_flops, self.peak_flops_per_second * window_seconds),
            "MFU_rollout": (rollout_flops, self.peak_flops_per_second * self.rollout_seconds),
            "MFU_train": (update_flops, self.peak_flops_per_second * self.update_seconds),
        }

    def mean_and_log(self):
        config = self.config
        env_metric_sums = self.env_metric_sums
        losses = {k: float(v) for k, v in self.losses.items()}
        flop_windows = [self.flop_window()]
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank_payloads = [None] * world_size
            torch.distributed.all_gather_object(rank_payloads, (env_metric_sums, losses, flop_windows[0]))
            flop_windows = [rank_flop_window for _, _, rank_flop_window in rank_payloads]
            merged_env_sums = {}
            loss_sums = {}
            for rank_env_sums, rank_losses, _ in rank_payloads:
                for key, (value_sum, value_count) in rank_env_sums.items():
                    entry = merged_env_sums.setdefault(key, [0.0, 0])
                    entry[0] += value_sum
                    entry[1] += value_count
                for key, value in rank_losses.items():
                    if value != value:  # skip NaN so one degenerate rank can't poison the mean
                        continue
                    entry = loss_sums.setdefault(key, [0.0, 0])
                    entry[0] += value
                    entry[1] += 1
            env_metric_sums = merged_env_sums
            losses = {key: total / count for key, (total, count) in loss_sums.items()}
        self.stats = opengigaflow.utils.finalize_environment_metrics(env_metric_sums)

        device = config["device"]
        agent_steps = int(dist_sum(self.global_step, device))
        logs = {
            "SPS": dist_sum(self.sps, device),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **pooled_flop_utilization_percent(flop_windows),
            **{f"environment/{k}": v for k, v in self.stats.items() if k not in UNCHARTED_WINDOW_SUM_METRICS},
            **{f"losses/{k}": v for k, v in losses.items()},
        }

        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        self.logger.log(logs, agent_steps)

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        model_path = self.save_checkpoint()
        # fixed name: a chained eval job finds the final model without knowing the last epoch
        path = os.path.join(self.config["data_dir"], self.config["final_model_name"])
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        run_id = self.logger.run_id
        path = self.config["data_dir"]
        models_dir = os.path.join(path, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_name = f"model_{self.config['env']}_{self.epoch:06d}.pt"
        model_path = os.path.join(models_dir, model_name)
        if not os.path.exists(model_path):
            torch.save(self.uncompiled_policy.state_dict(), model_path)
        for old_model_path in glob.glob(os.path.join(models_dir, "model_*.pt")):
            if old_model_path != model_path:
                os.remove(old_model_path)

        current_score = self.last_stats.get("score", -float("inf"))
        new_best = current_score > self.best_score
        if new_best:
            self.best_score = current_score

        state = {
            "state_version": TRAINER_STATE_VERSION,
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
            print(f"New best model saved at epoch {self.epoch} with score {self.best_score:.4f}")

        return model_path

    def print_dashboard(self, clear=False):
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
            f"{TITLE_COLOR}OpenGigaFlow",
            f"{LABEL_COLOR}CPU: {VALUE_COLOR}{np.mean(self.utilization.cpu_util):.1f}{UNIT_COLOR}%",
            f"{LABEL_COLOR}GPU: {VALUE_COLOR}{np.mean(self.utilization.gpu_util):.1f}{UNIT_COLOR}%",
            f"{LABEL_COLOR}DRAM: {VALUE_COLOR}{np.mean(self.utilization.cpu_mem):.1f}{UNIT_COLOR}%",
            f"{LABEL_COLOR}VRAM: {VALUE_COLOR}{np.mean(self.utilization.gpu_mem):.1f}{UNIT_COLOR}%",
        )

        summary = Table(box=None, expand=True)
        remaining = f"{VALUE_COLOR}A hair past a freckle{UNIT_COLOR}"
        if sps != 0:
            # total_timesteps is this rank's share; agent_steps and sps are summed over the ranks
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            remaining = duration((config["total_timesteps"] * world_size - agent_steps) / sps)

        summary.add_column(f"{LABEL_COLOR}Summary", justify="left", vertical="top", width=10)
        summary.add_column(f"{LABEL_COLOR}Value", justify="right", vertical="top", width=14)
        summary.add_row(f"{VALUE_COLOR}Env", f"{VALUE_COLOR}{config['env']}")
        summary.add_row(f"{VALUE_COLOR}Params", abbreviate(self.model_size))
        summary.add_row(f"{VALUE_COLOR}Steps", abbreviate(agent_steps))
        summary.add_row(f"{VALUE_COLOR}SPS", abbreviate(sps))
        summary.add_row(f"{VALUE_COLOR}Epoch", f"{VALUE_COLOR}{self.epoch}")
        summary.add_row(f"{VALUE_COLOR}Uptime", duration(self.uptime))
        summary.add_row(f"{VALUE_COLOR}Remaining", remaining)

        delta = profile.eval["buffer"] + profile.train["buffer"]
        performance = Table(box=None, expand=True, show_header=False)
        performance.add_column(f"{LABEL_COLOR}Performance", justify="left", width=10)
        performance.add_column(f"{LABEL_COLOR}Time", justify="right", width=8)
        performance.add_column(f"{LABEL_COLOR}%", justify="right", width=4)
        performance.add_row(*fmt_perf("Evaluate", TITLE_COLOR, delta, profile.eval))
        performance.add_row(*fmt_perf("  Forward", VALUE_COLOR, delta, profile.eval_forward))
        performance.add_row(*fmt_perf("  Env", VALUE_COLOR, delta, profile.env))
        performance.add_row(*fmt_perf("  Copy", VALUE_COLOR, delta, profile.eval_copy))
        performance.add_row(*fmt_perf("  Misc", VALUE_COLOR, delta, profile.eval_misc))
        performance.add_row(*fmt_perf("Train", TITLE_COLOR, delta, profile.train))
        performance.add_row(*fmt_perf("  Forward", VALUE_COLOR, delta, profile.train_forward))
        performance.add_row(*fmt_perf("  Learn", VALUE_COLOR, delta, profile.learn))
        performance.add_row(*fmt_perf("  Copy", VALUE_COLOR, delta, profile.train_copy))
        performance.add_row(*fmt_perf("  Misc", VALUE_COLOR, delta, profile.train_misc))

        losses_table = Table(box=None, expand=True)
        losses_table.add_column(f"{LABEL_COLOR}Losses", justify="left", width=16)
        losses_table.add_column(f"{LABEL_COLOR}Value", justify="right", width=8)
        for metric, value in self.losses.items():
            losses_table.add_row(f"{VALUE_COLOR}{metric}", f"{VALUE_COLOR}{value:.3f}")

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(summary, performance, losses_table)
        dashboard.add_row(monitor)

        table = Table(box=None, expand=True, pad_edge=False)
        dashboard.add_row(table)
        left = Table(box=None, expand=True)
        right = Table(box=None, expand=True)
        table.add_row(left, right)
        for user_stats_table in (left, right):
            user_stats_table.add_column(f"{LABEL_COLOR}User Stats", justify="left", width=20)
            user_stats_table.add_column(f"{LABEL_COLOR}Value", justify="right", width=10)

        if self.stats:
            self.last_stats = self.stats

        shown_count = 0
        for metric, value in (self.stats or self.last_stats).items():
            if metric in HIDDEN_DASHBOARD_METRICS or not math.isfinite(value):
                continue
            stats_half = left if shown_count % 2 == 0 else right
            stats_half.add_row(f"{VALUE_COLOR}{metric}", f"{VALUE_COLOR}{value:.3f}")
            shown_count += 1
            if shown_count == MAX_DASHBOARD_USER_STATS:
                break

        if clear:
            console.clear()

        with console.capture() as capture:
            console.print(dashboard)

        print("\033[0;0H" + capture.get())


def compute_puff_advantage(values, rewards, terminals, gamma, gae_lambda):
    """GAE advantages; CPU fallback when torch finds no CUDA / HIP toolkit at import (ADVANTAGE_CUDA false)."""
    device = values.device
    if not ADVANTAGE_CUDA:
        values = values.cpu()
        rewards = rewards.cpu()
        terminals = terminals.cpu()
    advantages = torch.zeros_like(values)
    torch.ops.opengigaflow.compute_puff_advantage(values, rewards, terminals, advantages, gamma, gae_lambda)
    return advantages.to(device)


def abbreviate(num):
    if num < 1e3:
        return f"{VALUE_COLOR}{num}{UNIT_COLOR}"
    elif num < 1e6:
        return f"{VALUE_COLOR}{num / 1e3:.1f}{UNIT_COLOR}K"
    elif num < 1e9:
        return f"{VALUE_COLOR}{num / 1e6:.1f}{UNIT_COLOR}M"
    elif num < 1e12:
        return f"{VALUE_COLOR}{num / 1e9:.1f}{UNIT_COLOR}B"
    else:
        return f"{VALUE_COLOR}{num / 1e12:.2f}{UNIT_COLOR}T"


def duration(seconds):
    if seconds < 0:
        return f"{VALUE_COLOR}0{UNIT_COLOR}s"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{VALUE_COLOR}{h}{UNIT_COLOR}h {VALUE_COLOR}{m}{UNIT_COLOR}m {VALUE_COLOR}{s}{UNIT_COLOR}s"
    if m:
        return f"{VALUE_COLOR}{m}{UNIT_COLOR}m {VALUE_COLOR}{s}{UNIT_COLOR}s"
    return f"{VALUE_COLOR}{s}{UNIT_COLOR}s"


def fmt_perf(name, color, delta_ref, prof):
    percent = 0 if delta_ref == 0 else int(100 * prof["buffer"] / delta_ref - 1e-5)
    return f"{color}{name}", duration(prof["elapsed"]), f"{VALUE_COLOR}{percent:2d}{UNIT_COLOR}%"


def dist_sum(value, device):
    if not torch.distributed.is_initialized():
        return value

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()


def shutdown_distributed():
    if not torch.distributed.is_initialized():
        return

    # Every rank must be done with the shared buffers before any rank tears them down.
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(rng_state):
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in rng_state:
        # a state saved on a node with more GPUs restores the generators of the devices that exist here
        device_states = rng_state["cuda"][: torch.cuda.device_count()]
        torch.cuda.set_rng_state_all([state.cpu() for state in device_states])


class Profile:
    def __init__(self, frequency=5):
        self.profiles = defaultdict(lambda: defaultdict(float))
        self.frequency = frequency
        self.stack = []

    def __getattr__(self, name):
        return self.profiles[name]

    def __call__(self, name, epoch, nest=False):
        # only every frequency-th epoch is timed; pop() scales its time up to stand for the skipped ones
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
            self.cpu_util.append(psutil.cpu_percent())
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


class NoLogger:
    def __init__(self, args):
        self.run_id = args["run_name"]

    def log(self, logs, step):
        pass

    def log_config(self, values):
        pass

    def close(self, model_path):
        pass


class WandbLogger:
    def __init__(self, args):
        import wandb

        # run_name is the run id: resume="allow" attaches to the run that carries it or creates one
        wandb.init(
            id=args["run_name"],
            name=args["run_name"],
            project=args["wandb_project"],
            group=args["wandb_group"],
            allow_val_change=True,
            save_code=False,
            resume="allow",
            config=args,
            tags=[args["tag"]] if args["tag"] is not None else [],
            settings=wandb.Settings(console="off", x_disable_stats=True),
        )
        self.wandb = wandb
        self.run_id = wandb.run.id
        self.should_upload_model = not args["no_model_upload"]

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def log_config(self, values):
        self.wandb.config.update(values, allow_val_change=True)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        # evaluation reads the architecture from the config.yaml next to the weights
        config_path = os.path.join(drive_benchmark.resolve_run_dir(model_path), "config.yaml")
        if os.path.isfile(config_path):
            artifact.add_file(config_path)
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path):
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()


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

    def log_config(self, values):
        pass

    def close(self, model_path):
        self.local_writer.close()


class TensorBoardMirror:
    """A remote run logger that also writes local TensorBoard event files."""

    def __init__(self, remote_logger, tensorboard_logger):
        self.remote_logger = remote_logger
        self.tensorboard_logger = tensorboard_logger

    def __getattr__(self, name):
        return getattr(self.remote_logger, name)

    def log(self, logs, step):
        self.remote_logger.log(logs, step)
        self.tensorboard_logger.log(logs, step)

    def close(self, model_path):
        self.tensorboard_logger.close(model_path)
        self.remote_logger.close(model_path)


def _get_git_metadata():
    git_metadata = {
        "commit_hash": os.environ.get("GITHUB_SHA") or os.environ.get("COMMIT_SHA"),
    }
    if git_metadata["commit_hash"] is not None:
        return git_metadata

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
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
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.yaml"), "w") as f:
        # json round trip turns defaultdicts into plain dicts for yaml.dump
        yaml.dump(json.loads(json.dumps(args)), f)


def _global_agent_steps(trainer):
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return int(trainer.global_step * world_size)


def derive_rank_seeds(vec_seed, train_seed, world_size, global_rank):
    """Per-rank (torch_seed, env_seed) from torchrun's global RANK: equal seeds would collect duplicate experience."""
    torch_seed = train_seed * world_size + global_rank
    env_seed = vec_seed
    if env_seed is not None:
        env_seed = int(np.random.SeedSequence([env_seed, train_seed, global_rank]).generate_state(1)[0])
    return torch_seed, env_seed


def parse_cpulist(cpulist):
    """sysfs cpulist ("0-119,256-375") to a set of CPU ids."""
    cpus = set()
    for part in cpulist.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def pooled_flop_utilization_percent(flop_windows):
    """MFU per phase over all ranks: summed model FLOPs over the summed FLOPs each rank's GPU does at its peak rate."""
    utilization_percent = {}
    for key in flop_windows[0]:
        model_flops = sum(window[key][0] for window in flop_windows)
        peak_rate_flops = sum(window[key][1] for window in flop_windows)
        utilization_percent[key] = 100.0 * model_flops / peak_rate_flops
    return utilization_percent


def gpu_numa_node(device_index):
    """NUMA node of a CUDA device; None when sysfs does not report one."""
    props = torch.cuda.get_device_properties(device_index)
    pci_dir = f"/sys/bus/pci/devices/{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    try:
        numa_node = int(open(f"{pci_dir}/numa_node").read())
    except (OSError, ValueError):
        return None
    return numa_node if numa_node >= 0 else None


def rank_core_share(cores, share_count, share_idx):
    """Contiguous slice of a NUMA node's cores for one of the ranks whose GPUs sit on that node."""
    cores_per_share = len(cores) // share_count
    return cores[share_idx * cores_per_share : (share_idx + 1) * cores_per_share]


def bind_process_to_gpu_numa_node(device_index, local_devices=None):
    """Pin this process and its env workers to the GPU's NUMA node, shared with `local_devices`; True if pinned."""
    if os.environ.get("OPENGIGAFLOW_NUMA_BIND", "1") == "0" or not torch.cuda.is_available():
        return False
    numa_node = gpu_numa_node(device_index)
    if numa_node is None:
        return False
    try:
        node_cpus = parse_cpulist(open(f"/sys/devices/system/node/node{numa_node}/cpulist").read())
    except (OSError, ValueError):
        return False
    cpus = node_cpus & os.sched_getaffinity(0)
    if not cpus:
        print(
            f"NUMA bind skipped: GPU {device_index} is on node {numa_node} but none of its CPUs are in this process's affinity mask"
        )
        return False
    node_devices = [device for device in local_devices or [device_index] if gpu_numa_node(device) == numa_node]
    share_cpus = cpus
    if len(node_devices) > 1:
        node_cores = opengigaflow.vector.read_cpu_cores(cpus)
        share = rank_core_share(node_cores, len(node_devices), node_devices.index(device_index))
        share_cpus = {cpu for _, core_threads in share for cpu in core_threads}
    cpus = share_cpus or cpus
    os.sched_setaffinity(0, cpus)
    print(
        f"GPU {device_index} on NUMA node {numa_node}: process and env workers bound to {len(cpus)} CPUs "
        f"({len(node_devices)} local ranks on this node)"
    )
    return bool(share_cpus)


def train(env_name, args=None, vecenv=None, policy=None, logger=None):
    args = args or load_config(env_name)

    # fine-tuning keeps the checkpoint's network and observation layout; rewards, maps and sim mode are this run's
    if args["load_model_path"]:
        config_yaml_path = os.path.join(drive_benchmark.resolve_run_dir(args["load_model_path"]), "config.yaml")
        if os.path.exists(config_yaml_path):
            print(f"Found config.yaml at {config_yaml_path}. Merging with defaults...")
            yaml_config = drive_arch.load_checkpoint_config(config_yaml_path)
            if isinstance(yaml_config.get("policy"), dict):
                args["policy"].update(yaml_config["policy"])
            if isinstance(yaml_config.get("env"), dict):
                checkpoint_env = yaml_config["env"]
                # a config saved before this obs column existed carries no key: that policy has no such column
                args["env"]["obs_traffic_light_yellow_remaining"] = False
                args["env"].update({key: checkpoint_env[key] for key in checkpoint_env if key in FINE_TUNE_ENV_KEYS})
        else:
            print(
                f"No config.yaml at {config_yaml_path}; fine-tuning with the configured "
                "policy/observation architecture instead of the checkpoint's."
            )

    args = normalize_opengigaflow_config(args, "training")
    validate_opengigaflow_config(args, "training")
    if vecenv is None:
        validate_opengigaflow_resources(args, "training")

    cpus_bound = False
    # Assume TorchRun DDP is used if LOCAL_RANK is set
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if "LOCAL_RANK" in os.environ:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {global_rank} (local {local_rank}), MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        local_rank_count = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
        cpus_bound = bind_process_to_gpu_numa_node(local_rank, range(local_rank_count))
    elif is_cuda_device(args["train"]["device"]) and torch.cuda.is_available():
        cpus_bound = bind_process_to_gpu_numa_node(torch.device(args["train"]["device"]).index or 0)

    train_seed = args["train"]["seed"]
    if train_seed is None:
        train_seed = time.time_ns() & 0xFFFFFFFF

    torch_seed, env_seed = derive_rank_seeds(args["vec"]["seed"], train_seed, world_size, global_rank)
    torch.manual_seed(torch_seed)
    vecenv = vecenv or load_env(args, seed=env_seed, pin_workers=cpus_bound)
    policy = policy or load_policy(args, vecenv)

    if "LOCAL_RANK" in os.environ:
        args["train"]["device"] = "cuda"
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank], output_device=local_rank)
        model.forward_eval = policy.forward_eval
        policy = model

    # Set before the logger so the run config the logger uploads carries it too.
    args["git"] = _get_git_metadata()

    # only rank 0 owns the run logger; the others get PPOTrainer's NoLogger instead of duplicate wandb runs
    is_rank0 = (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    if is_rank0:
        if args["wandb"]:
            logger = WandbLogger(args)
        if args["tb"]:
            tensorboard_logger = TensorBoardLogger(
                run_id=args["run_name"],
                experiment_dir=args["train"]["data_dir"],
            )
            logger = TensorBoardMirror(logger, tensorboard_logger) if args["wandb"] else tensorboard_logger

    train_config = dict(**args["train"], env=env_name, run_name=args["run_name"])
    if torch.distributed.is_initialized():
        train_config["total_timesteps"] //= torch.distributed.get_world_size()
    trainer = PPOTrainer(train_config, vecenv, policy, logger)

    # relaunching a run name resumes its own trainer_state.pt unless resume_state_path or load_model_path is set
    resume_state_path = args["train"]["resume_state_path"]
    if not resume_state_path and not args["load_model_path"]:
        run_state_path = os.path.join(args["train"]["data_dir"], "trainer_state.pt")
        if os.path.exists(run_state_path):
            print(f"Run '{args['run_name']}' already exists in {args['train']['data_dir']}; resuming it.")
            resume_state_path = run_state_path

    if resume_state_path:
        trainer.load_training_state(resume_state_path)
        # the checkpoint holds rank 0's RNG streams: re-decorrelate the other ranks' action sampling
        if global_rank != 0:
            torch.manual_seed(int(np.random.SeedSequence([torch_seed, trainer.epoch]).generate_state(1)[0]))

    if args["load_model_path"]:
        trainer_state_path = os.path.join(drive_benchmark.resolve_run_dir(args["load_model_path"]), "trainer_state.pt")
        if os.path.exists(trainer_state_path):
            print(f"Resuming optimizer/step state from {trainer_state_path}")
            # weights_only=False: the state carries the numpy RNG state, not just tensors
            tstate = torch.load(trainer_state_path, map_location=train_config["device"], weights_only=False)
            trainer.optimizer.load_state_dict(tstate["optimizer_state_dict"])
            # The saved groups carry the old run's annealed lr; train.learning_rate is this run's base.
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = train_config["learning_rate"]
                param_group["initial_lr"] = train_config["learning_rate"]
            trainer.global_step = tstate["global_step"]
            trainer.last_log_step = trainer.global_step
            trainer.epoch = tstate["update"]
            trainer.ema_max = tstate["ema_max"]
            # fast-forward the cosine schedule to the resumed epoch
            for _ in range(trainer.epoch):
                trainer.scheduler.step()
        else:
            print(f"No trainer_state.pt next to {args['load_model_path']}; starting optimizer fresh.")

    path = args["train"]["data_dir"]
    if is_rank0:
        _save_experiment_config(args, path)

    stop_after_epochs = args["train"]["stop_after_epochs"]
    stop_epoch = float("inf") if stop_after_epochs is None else trainer.epoch + stop_after_epochs

    use_cuda = is_cuda_device(train_config["device"])
    while trainer.global_step < train_config["total_timesteps"] and trainer.epoch < stop_epoch:
        try:
            if use_cuda:
                torch.compiler.cudagraph_mark_step_begin()
            trainer.evaluate()
            if use_cuda:
                torch.compiler.cudagraph_mark_step_begin()
            trainer.train()
        except Exception:
            trainer.vecenv.close()
            trainer.utilization.stop()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            raise

        if is_rank0 and args["train"]["render"] and trainer.epoch % args["train"]["render_interval"] == 0:
            render_training_replays(
                args=args,
                policy=trainer.uncompiled_policy,
                epoch=trainer.epoch,
                global_step=_global_agent_steps(trainer),
                run_dir=path,
            )

    # the last epoch of a finished run has logged already; a run stopped early still holds an unlogged window
    if trainer.global_step != trainer.last_log_step:
        trainer.mean_and_log()
    trainer.print_dashboard()
    model_path = trainer.close()
    shutdown_distributed()
    trainer.logger.close(model_path)


def eval(
    env_name,
    args=None,
    policy=None,
    eval_output_dir=None,
    eval_output_subdir=None,
    benchmark_names=None,
):
    """Run configured benchmarks or replay failures from an existing CSV."""
    cli_overrides = known_cli_overrides(env_name) if args is None else []
    if any(override.split("=", 1)[0] == "env.num_agents" for override in cli_overrides):
        raise opengigaflow.APIUsageError("Use eval.num_agents to configure evaluation agent count")
    args = args or load_config(env_name, overrides=cli_overrides)
    eval_config = args["eval"]
    benchmark_config_path = eval_config["benchmark_config"]
    selected_benchmarks = benchmark_names if benchmark_names is not None else eval_config["benchmarks"]
    eval_config["benchmarks"] = selected_benchmarks
    output_name = eval_config["output_name"]
    render_filter = eval_config["render_filter"]
    max_rendered_failures = eval_config["max_rendered_failures"]
    failure_replay_csv = eval_config["failure_replay_csv"]
    scenario_offset = eval_config["scenario_offset"]
    configured_output_subdir = eval_config["output_subdir"]
    eval_training_render = args["env"]["eval_training_render"]
    eval_device = torch.device(args["train"]["device"])
    if eval_device.type == "cuda":
        bind_process_to_gpu_numa_node(eval_device.index or 0)

    report_to_wandb = bool(args["wandb"]) and failure_replay_csv is None
    environment_config, benchmarks = drive_benchmark.load_benchmark_config(
        benchmark_config_path, selected_benchmarks, eval_config["map_dir"], eval_config["num_scenarios"]
    )
    base_args, checkpoint_config_path = drive_benchmark.load_checkpoint_architecture(args)
    base_args["env"]["eval_training_render"] = eval_training_render

    if report_to_wandb:
        drive_eval_report.load_run_identity(checkpoint_config_path)
    if eval_output_dir is None:
        run_dir = drive_benchmark.resolve_run_dir(base_args["load_model_path"])
        eval_output_dir = os.path.join(run_dir, eval_config["output_dir_name"])
    if eval_output_subdir is None:
        eval_output_subdir = configured_output_subdir or datetime.now().strftime("%Y%m%d-%H%M%S")
    failure_replay_output_dir = None
    if failure_replay_csv is not None:
        failure_replay_csv = os.path.abspath(failure_replay_csv)
        failure_replay_output_dir = os.path.dirname(failure_replay_csv)
    benchmark_results = {}
    selfplay_summary_path = None
    evaluation_policy_cache = {"policy": policy}
    for benchmark in benchmarks:
        run_args = drive_benchmark.build_benchmark_args(
            base_args,
            benchmark,
            environment_config,
            cli_overrides,
        )
        render_scenarios = eval_config["render_scenarios"] or run_args["env"]["eval_training_render"]
        if scenario_offset:
            if run_args["env"]["simulation_mode"] == "replay":
                map_dir = run_args["env"]["map_dir"]
                available_map_count = (
                    1 if os.path.isfile(map_dir) else len([f for f in os.listdir(map_dir) if f.endswith(".bin")])
                )
                if scenario_offset + run_args["num_scenarios"] > available_map_count:
                    raise opengigaflow.APIUsageError(
                        f"eval.scenario_offset ({scenario_offset}) plus num_scenarios "
                        f"({run_args['num_scenarios']}) exceeds the {available_map_count} replay maps in {map_dir}"
                    )
            # Shard-unique seed: identical seeds make every shard re-draw the same scenarios.
            shard_seed = (run_args["train"]["seed"] + scenario_offset) % (MAX_C_SEED + 1)
            run_args["train"]["seed"] = shard_seed
            run_args["vec"]["seed"] = shard_seed
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
            scenario_offset=scenario_offset,
        )
        print(f"Evaluation {benchmark['name']}: {num_scenarios} scenarios across {num_workers} workers")
        replay_output_dir = (
            os.path.join(benchmark_output_dir, drive_eval_replay.ZLIB_REPLAY_DIR_NAME) if render_scenarios else None
        )
        summaries = _run_eval_rollout(
            run_args,
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
        if benchmark["name"] == drive_eval_report.SELFPLAY_BENCHMARK_NAME and summary is not None:
            selfplay_summary_path = os.path.join(benchmark_output_dir, drive_eval_report.SELFPLAY_SUMMARY_FILE_NAME)

        if render_scenarios:
            max_renderer_count = (
                eval_config["observation_replay_writer_count"] if eval_config["capture_observations"] else None
            )
            drive_eval_replay._render_eval_replays(
                summaries, benchmark_output_dir, eval_config["keep_zlib_replays"], max_renderer_count
            )
        elif render_filter is not None:
            _render_eval_failures(
                run_args,
                benchmark,
                os.path.join(benchmark_output_dir, "episode_metrics.csv"),
                benchmark_output_dir,
                policy,
                eval_config["capture_observations"],
                max_rendered_failures,
                evaluation_policy_cache=evaluation_policy_cache,
            )

    if report_to_wandb and selfplay_summary_path is not None:
        drive_eval_report.report_headline_metric(
            drive_eval_report.EVAL_TYPE_SELFPLAY, selfplay_summary_path, os.path.dirname(checkpoint_config_path)
        )
    elif report_to_wandb:
        print(
            f"wandb: nothing reported; only the {drive_eval_report.SELFPLAY_BENCHMARK_NAME} benchmark has a headline metric."
        )
    return benchmark_results


def _require_finite_eval_batch(batch_array, what, num_workers, worker_env_kwargs):
    if np.isfinite(batch_array).all():
        return
    rows_per_worker = batch_array.shape[0] // num_workers
    bad_rows = np.unique(np.where(~np.isfinite(batch_array))[0])
    bad_workers = sorted({int(row) // rows_per_worker for row in bad_rows})
    details = ", ".join(
        f"worker {w} (starting_map={worker_env_kwargs[w].get('starting_map')},"
        f" num_eval_scenarios={worker_env_kwargs[w].get('num_eval_scenarios')})"
        for w in bad_workers
    )
    raise opengigaflow.APIUsageError(f"Non-finite {what} from {details}")


def _run_eval_rollout(
    args,
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
    vecenv = opengigaflow.vector.make(
        [Drive] * num_workers,
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
            raise opengigaflow.APIUsageError(
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
            policy = load_policy(args, vecenv)
            evaluation_policy_cache["policy"] = policy
        policy.eval()
        if "policy_forward_eval" not in evaluation_policy_cache:
            policy_forward_eval = policy.forward_eval
            eval_sample_logits = opengigaflow.pytorch.sample_logits
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
        action_selection = args["eval"]["action_selection"]
        uncompiled_policy = base_policy(policy)
        discrete_policy = not uncompiled_policy.is_continuous
        device = torch_device(args["train"]["device"])
        use_bfloat16 = args["train"]["precision"] == "bfloat16" and is_cuda_device(device)
        if use_bfloat16 and not torch.cuda.is_bf16_supported():
            raise opengigaflow.APIUsageError("bfloat16 evaluation requires CUDA BF16 support")
        allow_tf32 = args["train"]["tf32"]
        if not allow_tf32 and use_bfloat16 and not getattr(uncompiled_policy, "fp32_heads", False):
            raise opengigaflow.APIUsageError(
                "train.tf32=False with train.precision=bfloat16 requires policy.fp32_heads=true"
            )
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        eval_amp_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16)
        obs, _ = vecenv.reset(rollout_seed)
        _require_finite_eval_batch(obs, "observations after eval reset", num_workers, worker_env_kwargs)
        padding_agent_count = inference_agents_per_batch - agents_per_batch
        policy_obs_tensor = None
        if padding_agent_count:
            policy_obs_tensor = torch.zeros(
                (inference_agents_per_batch, *obs.shape[1:]),
                dtype=torch.as_tensor(obs).dtype,
                device=device,
            )
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
                logits, value = policy_forward_eval(policy_obs_tensor)
                action, logprob, entropy, cont_action = eval_sample_logits(
                    logits,
                    action_selection=action_selection,
                    env_continuous=True,
                    policy=uncompiled_policy,
                )
                if discrete_policy:
                    # replay logs record the discrete class; the env steps on its continuous counterpart
                    raw_action = action[:agents_per_batch].cpu().numpy()
                    continuous_actions = cont_action.reshape(-1, *vecenv.single_action_space.shape)
                    action = continuous_actions[:agents_per_batch].float().cpu().numpy()
                else:
                    raw_action = action[:agents_per_batch].cpu().numpy().reshape(vecenv.action_space.shape)
                    action = raw_action
            action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)
            _require_finite_eval_batch(action, "policy actions", num_workers, worker_env_kwargs)

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

            obs, _, _, _, infos = vecenv.step(action)
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


def render_training_replays(args, policy, epoch, global_step, run_dir):
    """Render one HTML replay per selected training map with the live policy, observations included."""
    run_args = copy.deepcopy(args)
    env_config = run_args["env"]
    # Fixed-length episodes: eval-mode summaries flush on the resample boundary.
    env_config["termination_mode"] = False
    env_config["compute_eval_metrics"] = True
    # Without it eval mode spawns max_agents_per_env eval-sized cars per map, not the training distribution.
    env_config["eval_training_render"] = True
    env_config["num_agents"] = env_config["max_agents_per_env"]
    run_args["eval"]["action_selection"] = opengigaflow.pytorch.ACTION_SELECT_SAMPLE
    capture_observations = run_args["eval"]["capture_observations"]
    # Observation replays render at multi-GB RSS each; the obs writer count bounds that too.
    max_renderer_count = run_args["eval"]["observation_replay_writer_count"] if capture_observations else None
    # Same maps and episode seeds at every render of a run, so renders across epochs compare like for like.
    render_seed = run_args["train"]["seed"] if run_args["train"]["seed"] is not None else 0
    render_rng = np.random.default_rng(render_seed)
    render_num_maps = run_args["train"]["render_num_maps"]
    if render_num_maps is None:
        map_indices = list(range(env_config["num_maps"]))
    else:
        map_indices = sorted(render_rng.choice(env_config["num_maps"], size=render_num_maps, replace=False).tolist())
    map_seed_pairs = [(map_idx, int(render_rng.integers(0, 2**63, dtype=np.int64))) for map_idx in map_indices]
    num_scenarios = len(map_seed_pairs)
    num_workers = min(run_args["vec"]["num_envs"], num_scenarios)
    worker_env_kwargs, total_steps = drive_benchmark._plan_failure_replay_workers(
        run_args, map_seed_pairs, num_workers, env_config["scenario_length"]
    )
    output_dir = os.path.join(run_dir, "renders", f"epoch_{epoch:06d}_step_{global_step}")

    rng_state = capture_rng_state()
    policy_was_training = bool(getattr(policy, "training", False))
    try:
        summaries = _run_eval_rollout(
            run_args,
            worker_env_kwargs,
            total_steps,
            "Rendering training replays",
            num_scenarios,
            policy=policy,
            replay_output_dir=os.path.join(output_dir, drive_eval_replay.ZLIB_REPLAY_DIR_NAME),
            capture_observations=capture_observations,
        )
        drive_eval_replay._render_eval_replays(
            summaries, output_dir, run_args["eval"]["keep_zlib_replays"], max_renderer_count
        )
    except Exception:
        print(f"\n[training render] Replay rendering failed at epoch {epoch}; continuing training:")
        traceback.print_exc()
    finally:
        if hasattr(policy, "train"):
            policy.train(policy_was_training)
        restore_rng_state(rng_state)


def _render_eval_failures(
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
    replay_wave_size = len(pairs)
    if capture_observations:
        observation_replay_wave_size = run_args["eval"]["observation_replay_wave_size"]
        replay_wave_size = min(
            len(pairs),
            configured_worker_count,
            observation_replay_wave_size,
        )
        replay_agent_capacity = failure_args["env"]["max_agents_per_env"]
        failure_args["env"]["num_agents"] = replay_agent_capacity
    replay_output_dir = os.path.join(failures_dir, drive_eval_replay.ZLIB_REPLAY_DIR_NAME)
    os.makedirs(replay_output_dir, exist_ok=True)
    agents_per_batch_values = selected_rows["agents_per_batch"].unique()
    if len(agents_per_batch_values) != 1:
        raise opengigaflow.APIUsageError("Benchmark failure rows must contain exactly one agents_per_batch value")
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
    max_renderer_count = run_args["eval"]["observation_replay_writer_count"] if capture_observations else None
    drive_eval_replay._render_eval_replays(
        summaries, failures_dir, run_args["eval"]["keep_zlib_replays"], max_renderer_count
    )
    return {
        "episodes": summaries,
        "summary": summary,
    }


def load_env(args, seed=None, pin_workers=False):
    vec_kwargs = dict(args["vec"])
    if seed is not None:
        vec_kwargs["seed"] = seed
    if vec_kwargs["backend"] == "Multiprocessing":
        vec_kwargs["pin_workers"] = pin_workers
        if args["train"]["rollout_dtype"] == "float16":
            vec_kwargs["observation_transport_dtype"] = "float16"
    return opengigaflow.vector.make(Drive, env_kwargs=args["env"], **vec_kwargs)


def load_policy(args, vecenv):
    device = torch_device(args["train"]["device"])
    policy_cls = getattr(drive_torch, args["policy_name"])
    policy = policy_cls(vecenv.driver_env, **args["policy"])

    policy = policy.to(device)

    load_path = args["load_model_path"]
    if load_path is not None:
        checkpoint = torch.load(load_path, map_location=device)
        policy = opengigaflow.pytorch.load_policy_weights(policy, checkpoint)

    return policy


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")


def config_has_setting(config, setting_path):
    node = config
    for name in setting_path.split("."):
        if not isinstance(node, dict) or name not in node:
            return False
        node = node[name]
    return True


def known_cli_overrides(env_name):
    """Hydra overrides from argv; one that names a setting the config lacks is dropped with a warning."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        settings = OmegaConf.to_container(compose(config_name=env_name))
    overrides = []
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            hint = arg.lstrip("-").replace("-", "_").split("=")[0]
            raise opengigaflow.APIUsageError(f"'{arg}' uses the old flag syntax. Use Hydra overrides: {hint}=<value>")
        setting_path, assignment, _ = arg.partition("=")
        is_plain_setting = bool(assignment) and all(name.isidentifier() for name in setting_path.split("."))
        if is_plain_setting and not config_has_setting(settings, setting_path):
            print(
                f"WARNING: ignoring override '{arg}': the {env_name} config has no setting '{setting_path}' "
                "(removed or misspelled)",
                file=sys.stderr,
            )
            continue
        overrides.append(arg)
    return overrides


def load_config(env_name, overrides=None):
    if overrides is None:
        overrides = known_cli_overrides(env_name)

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=env_name, overrides=overrides)

    args = defaultdict(dict, OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    return defaultdict(dict, normalize_opengigaflow_config(args, "load"))


def main():
    err = "Usage: opengigaflow [train | eval <benchmark>] [optional args]"
    if len(sys.argv) < 2:
        raise opengigaflow.APIUsageError(err)

    mode = sys.argv.pop(1)
    if mode == "train":
        train(env_name=ENV_NAME)
    elif mode == "eval":
        if len(sys.argv) < 2:
            raise opengigaflow.APIUsageError("Usage: opengigaflow eval <benchmark> [optional args]")
        benchmark_name = sys.argv.pop(1)
        eval(env_name=ENV_NAME, benchmark_names=benchmark_name)
    else:
        raise opengigaflow.APIUsageError(err)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
