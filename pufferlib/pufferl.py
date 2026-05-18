## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import copy
import numbers
import warnings
from dataclasses import dataclass
import pickle

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
import json
from datetime import datetime
from threading import Thread
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import yaml
import zlib

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record

import gymnasium

import pufferlib
import pufferlib.sweep
import pufferlib.vector
import pufferlib.pytorch
import pufferlib.viz
import pufferlib.mining_viz

import mediapy

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
    "ttc_within_bound_rate",
}


def metric_log_key(metric_name):
    if "/" in metric_name:
        return metric_name

    policy_metrics = {
        "episode_return",
        "episode_return_collision",
        "episode_return_offroad",
        "episode_return_drive",
        "episode_return_adversarial",
        "mean_reward",
        "mean_reward_collision",
        "mean_reward_offroad",
        "mean_reward_drive",
        "mean_reward_adversarial",
    }
    target_metrics = {
        "target_episode_return": "episode_return",
        "target_episode_return_collision": "episode_return_collision",
        "target_episode_return_offroad": "episode_return_offroad",
        "target_episode_return_drive": "episode_return_drive",
        "target_mean_reward": "mean_reward",
        "target_mean_reward_collision": "mean_reward_collision",
        "target_mean_reward_offroad": "mean_reward_offroad",
        "target_mean_reward_drive": "mean_reward_drive",
    }
    overview_metrics = {
        "episode_length",
        "target_episode_length",
        "did_target_fail",
        "did_target_collide",
        "did_target_offroad",
        "target_hit_count",
        "target_hit_responsibility",
        "target_hit_low_responsibility_rate",
        "collision_rate",
        "offroad_rate",
        "red_light_violation_rate",
    }

    if metric_name in policy_metrics:
        return f"policy/{metric_name}"
    if metric_name in target_metrics:
        return f"target/{target_metrics[metric_name]}"
    if metric_name in overview_metrics:
        return f"overview/{metric_name}"
    return f"environment/{metric_name}"


@dataclass
class ActorOutput:
    action: torch.Tensor
    logprob: torch.Tensor | None = None
    value: torch.Tensor | None = None
    rollout_observation: torch.Tensor | None = None
    clip_actions: bool = False


class TrainableTorchActor:
    def __init__(self, policy, observation_space):
        self.policy = policy
        self.rollout_observation_space = observation_space
        self.hidden_size = getattr(policy, "hidden_size", None)

    def prepare_observation(self, raw_observation):
        return raw_observation

    def act(self, raw_observation, state=None, deterministic=False):
        observation = self.prepare_observation(raw_observation)
        logits, value = self.policy.forward_eval(observation, state)
        action, logprob, _ = pufferlib.pytorch.sample_logits(logits, deterministic=deterministic)
        return ActorOutput(
            action=action,
            logprob=logprob,
            value=value.flatten(),
            rollout_observation=observation,
            clip_actions=isinstance(logits, torch.distributions.Normal),
        )


class TargetTorchActor:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env
        self.hidden_size = getattr(policy, "hidden_size", None)
        self.target_max_partner_obs_distance = float(getattr(env, "target_max_partner_obs_distance", 0.0))
        self.adv_reward_weight_drive_conditioning = bool(getattr(env, "adv_reward_weight_drive_conditioning", False))
        self.adv_reward_weight_drive_feature_start = env.ego_features + env.num_reward_coefs
        self.partner_start = env.ego_features + env.num_reward_coefs + env.target_dim
        self.max_partner_observations = env.max_partner_observations
        self.partner_features = env.partner_features
        self.max_position = env.max_position

    def prepare_observation(self, raw_observation):
        observations = raw_observation.clone()
        if self.adv_reward_weight_drive_conditioning:
            observations = torch.cat(
                [
                    observations[:, : self.adv_reward_weight_drive_feature_start],
                    observations[:, self.adv_reward_weight_drive_feature_start + 1 :],
                ],
                dim=-1,
            )

        if self.target_max_partner_obs_distance <= 0.0 or self.max_partner_observations <= 0:
            return observations

        partner_dim = self.max_partner_observations * self.partner_features
        if partner_dim <= 0:
            return observations

        partner_observations = observations[:, self.partner_start : self.partner_start + partner_dim].view(
            -1, self.max_partner_observations, self.partner_features
        )

        dx = partner_observations[..., 0] * self.max_position
        dy = partner_observations[..., 1] * self.max_position
        far_mask = dx.square() + dy.square() > self.target_max_partner_obs_distance**2
        partner_observations[far_mask] = 0.0
        return observations

    def act(self, raw_observation, state=None, deterministic=False):
        observation = self.prepare_observation(raw_observation)
        logits, _ = self.policy.forward_eval(observation, state)
        action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=deterministic)
        return ActorOutput(
            action=action,
            clip_actions=isinstance(logits, torch.distributions.Normal),
        )


def _make_target_policy_env_view(env):
    target_env = copy.copy(env)
    removed_features = int(getattr(env, "num_adv_reward_weight_drive_features", 0) or 0)
    target_env.adv_reward_weight_drive_conditioning = False
    target_env.num_adv_reward_weight_drive_features = 0
    if hasattr(env, "num_obs"):
        target_env.num_obs = int(env.num_obs) - removed_features
    if hasattr(env, "single_observation_space") and hasattr(env.single_observation_space, "dtype"):
        target_env.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(target_env.num_obs,),
            dtype=env.single_observation_space.dtype,
        )
    return target_env


def _index_batch(batch, indices):
    if batch is None:
        return None
    if torch.is_tensor(batch):
        return batch.index_select(0, indices)
    if isinstance(batch, np.ndarray):
        return batch[indices.detach().cpu().numpy()]
    return batch


def _run_actor_subset(actor, raw_observation, indices, step_context, recurrent_state=None, deterministic=False):
    if actor is None or indices.numel() == 0:
        return None

    state = {
        "reward": _index_batch(step_context.get("reward"), indices),
        "done": _index_batch(step_context.get("done"), indices),
        "env_id": step_context.get("env_id"),
        "mask": _index_batch(step_context.get("mask"), indices),
    }

    if recurrent_state is not None:
        state["lstm_h"] = recurrent_state["lstm_h"].index_select(0, indices)
        state["lstm_c"] = recurrent_state["lstm_c"].index_select(0, indices)

    actor_output = actor.act(
        _index_batch(raw_observation, indices),
        state=state,
        deterministic=deterministic,
    )

    if recurrent_state is not None:
        recurrent_state["lstm_h"].index_copy_(0, indices, state["lstm_h"])
        recurrent_state["lstm_c"].index_copy_(0, indices, state["lstm_c"])

    return actor_output


def _allocate_rollout_observation(policy_actor, batch_size, device):
    obs_space = policy_actor.rollout_observation_space
    obs_dtype = pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype]
    return torch.zeros(batch_size, *obs_space.shape, dtype=obs_dtype, device=device)


def _route_actor_actions(
    raw_observation,
    target_mask,
    policy_actor,
    step_context,
    policy_recurrent_state=None,
    target_actor=None,
    target_recurrent_state=None,
    deterministic=False,
):
    batch_size = raw_observation.shape[0]
    device = raw_observation.device
    adv_idx = torch.nonzero(~target_mask, as_tuple=False).flatten()
    target_idx = torch.nonzero(target_mask, as_tuple=False).flatten()

    full_action = None
    full_logprob = torch.zeros(batch_size, device=device)
    full_value = torch.zeros(batch_size, device=device)
    # PPO only trains the adversarial actor, so target rows stay zero-padded in rollout observations.
    rollout_observation = _allocate_rollout_observation(policy_actor, batch_size, device)
    clip_actions = False

    adv_output = _run_actor_subset(
        policy_actor,
        raw_observation,
        adv_idx,
        step_context,
        recurrent_state=policy_recurrent_state,
        deterministic=deterministic,
    )
    if adv_output is not None:
        full_action = torch.zeros(
            batch_size, *adv_output.action.shape[1:], dtype=adv_output.action.dtype, device=device
        )
        full_action.index_copy_(0, adv_idx, adv_output.action)
        if adv_output.logprob is not None:
            full_logprob.index_copy_(0, adv_idx, adv_output.logprob)
        if adv_output.value is not None:
            full_value.index_copy_(0, adv_idx, adv_output.value)
        if adv_output.rollout_observation is not None:
            rollout_observation.index_copy_(0, adv_idx, adv_output.rollout_observation)
        clip_actions = clip_actions or adv_output.clip_actions

    target_output = _run_actor_subset(
        target_actor,
        raw_observation,
        target_idx,
        step_context,
        recurrent_state=target_recurrent_state,
        deterministic=deterministic,
    )
    if target_output is not None:
        if full_action is None:
            full_action = torch.zeros(
                batch_size,
                *target_output.action.shape[1:],
                dtype=target_output.action.dtype,
                device=device,
            )
        full_action.index_copy_(0, target_idx, target_output.action)
        clip_actions = clip_actions or target_output.clip_actions

    return ActorOutput(
        action=full_action,
        logprob=full_logprob,
        value=full_value,
        rollout_observation=rollout_observation,
        clip_actions=clip_actions,
    )


class PuffeRL:
    def __init__(self, config, vecenv, policy, target_policy=None, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = config["torch_deterministic"]
        torch.backends.cudnn.benchmark = not config["torch_deterministic"]
        torch.use_deterministic_algorithms(config["torch_deterministic"], warn_only=True)

        # Reproducibility
        seed = config["seed"]

        # Vecenv info
        vecenv.async_reset(seed)
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

        # Torch compile
        self.uncompiled_policy = policy
        self.policy = policy
        self.uncompiled_target_policy = target_policy
        self.target_policy = target_policy
        if config["compile"]:
            self.policy = torch.compile(policy, mode=config["compile_mode"])
            self.policy.forward_eval = torch.compile(policy, mode=config["compile_mode"])
            pufferlib.pytorch.sample_logits = torch.compile(
                pufferlib.pytorch.sample_logits, mode=config["compile_mode"]
            )
            if target_policy is not None:
                self.target_policy = torch.compile(target_policy, mode=config["compile_mode"])
                self.target_policy.forward_eval = torch.compile(target_policy, mode=config["compile_mode"])

        self.policy_actor = TrainableTorchActor(self.policy, vecenv.single_observation_space)
        self.target_actor = (
            TargetTorchActor(self.target_policy, vecenv.driver_env) if self.target_policy is not None else None
        )
        obs_space = self.policy_actor.rollout_observation_space

        self.observations = torch.zeros(
            segments,
            horizon,
            *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
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
        self.target_masks = torch.zeros(segments, horizon, device=device, dtype=torch.bool)
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
            h = self.policy_actor.hidden_size
            self.lstm_h = {i * n: torch.zeros(n, h, device=device) for i in range(total_agents // n)}
            self.lstm_c = {i * n: torch.zeros(n, h, device=device) for i in range(total_agents // n)}
            if self.target_actor is not None:
                target_h = self.target_actor.hidden_size
                self.target_lstm_h = {i * n: torch.zeros(n, target_h, device=device) for i in range(total_agents // n)}
                self.target_lstm_c = {i * n: torch.zeros(n, target_h, device=device) for i in range(total_agents // n)}

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
        epochs = (config["total_timesteps"] + config["batch_size"] - 1) // config["batch_size"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.total_epochs = epochs

        # Automatic mixed precision
        precision = config["precision"]
        self.amp_context = contextlib.nullcontext()
        if config.get("amp", True) and config["device"] == "cuda":
            self.amp_context = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, precision))
        if precision not in ("float32", "bfloat16"):
            raise pufferlib.APIUsageError(f"Invalid precision: {precision}: use float32 or bfloat16")

        # Initializations
        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
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

    def _build_target_mask(self, info, env_id, device):
        num_agents_per_batch = self.vecenv.agents_per_batch
        sdc_controller = getattr(self.vecenv.driver_env, "sdc_controller_str", "policy")
        should_mask_sdc = self.target_actor is not None or sdc_controller != "policy"
        if not should_mask_sdc:
            return torch.zeros(num_agents_per_batch, dtype=torch.bool, device=device)

        num_agents_per_worker = self.vecenv.driver_env.num_agents
        target_mask = torch.zeros(num_agents_per_batch, dtype=torch.bool, device=device)

        env_counter = 0
        for information in info:
            agent_offsets = information.get("agent_offsets")
            if agent_offsets is None:
                continue

            agent_offsets = torch.as_tensor(agent_offsets, dtype=torch.int64, device=device)
            target_mask[agent_offsets[:-1] + env_counter * num_agents_per_worker] = True
            env_counter += 1

        return target_mask

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
                if self.target_actor is not None:
                    self.target_lstm_h[k].zero_()
                    self.target_lstm_c[k].zero_()

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            self.global_step += int(mask.sum())

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)  # , non_blocking=True)
            r = torch.as_tensor(r).to(device)  # , non_blocking=True)
            d = torch.as_tensor(d).to(device)  # , non_blocking=True)
            t = torch.as_tensor(t).to(device)  # , non_blocking=True)
            done_mask = (d + t).clamp(max=1.0)
            m = torch.as_tensor(mask).to(device)  # , non_blocking=True)
            target_mask = self._build_target_mask(info, env_id, device)

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                step_context = dict(
                    reward=r,
                    done=done_mask,
                    env_id=env_id,
                    mask=mask,
                )
                policy_recurrent_state = None
                target_recurrent_state = None
                if config["use_rnn"]:
                    policy_recurrent_state = {
                        "lstm_h": self.lstm_h[env_id.start],
                        "lstm_c": self.lstm_c[env_id.start],
                    }
                    if self.target_actor is not None:
                        target_recurrent_state = {
                            "lstm_h": self.target_lstm_h[env_id.start],
                            "lstm_c": self.target_lstm_c[env_id.start],
                        }

                actor_output = _route_actor_actions(
                    o_device,
                    target_mask,
                    self.policy_actor,
                    step_context,
                    policy_recurrent_state=policy_recurrent_state,
                    target_actor=self.target_actor,
                    target_recurrent_state=target_recurrent_state,
                    deterministic=False,
                )

                if config["normalize_rewards"]:
                    r = torch.sign(r) * torch.log1p(torch.abs(r))

            profile("eval_copy", epoch)
            with torch.no_grad():
                # Fast path for fully vectorized envs
                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(self.ep_indices[env_id.start].item(), 1 + self.ep_indices[env_id.stop - 1].item())

                if config["cpu_offload"]:
                    self.observations[batch_rows, l] = actor_output.rollout_observation.cpu()
                else:
                    self.observations[batch_rows, l] = actor_output.rollout_observation

                self.actions[batch_rows, l] = actor_output.action
                self.logprobs[batch_rows, l] = actor_output.logprob
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
                self.values[batch_rows, l] = actor_output.value
                self.masks[batch_rows, l] = m
                self.target_masks[batch_rows, l] = target_mask

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l + 1 >= config["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config["device"]).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full

                action = actor_output.action.cpu().numpy()
                if actor_output.clip_actions:
                    action = np.clip(action, self.vecenv.action_space.low, self.vecenv.action_space.high)

            profile("eval_misc", epoch)
            for i in info:
                if i.keys() == {"agent_offsets"}:
                    continue
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
        profile("train", epoch)
        profile("train_misc", epoch, nest=True)
        losses = defaultdict(float)
        config = self.config
        ppo_granularity = config["ppo_granularity"]
        if ppo_granularity == "auto":
            ppo_granularity = "trajectory" if config["use_rnn"] else "transition"
        if config["use_rnn"] and ppo_granularity == "transition":
            raise ValueError("RNN requires trajectory-level training")

        if ppo_granularity == "trajectory":
            explained_var = self._train_ppo_trajectory(losses, profile, epoch)
        else:
            explained_var = self._train_ppo_transition(losses, profile, epoch)

        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()

        losses["explained_variance"] = explained_var

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.epoch >= self.total_epochs
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            self.losses = losses
            logs = self.mean_and_log()
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        # The final checkpoint is written by close(), after all ranks have
        # finished the last distributed logging/dashboard reductions. Saving
        # here can leave nonzero ranks waiting in the next all_reduce while
        # rank 0 is still doing slow filesystem or artifact work.
        if not done_training and self.epoch % config["checkpoint_interval"] == 0:
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

        if self.config["eval"]["wosac_realism_eval"] and (
            self.epoch % self.config["eval"]["eval_interval"] == 0 or done_training
        ):
            pufferlib.utils.run_wosac_eval_in_subprocess(self.config, self.logger, self.global_step)

        if self.config["eval"]["human_replay_eval"] and (
            self.epoch % self.config["eval"]["eval_interval"] == 0 or done_training
        ):
            pufferlib.utils.run_human_replay_eval_in_subprocess(self.config, self.logger, self.global_step)

        if self.config["eval"]["wosac_realism_eval"] and (
            self.epoch % self.config["eval"]["eval_interval"] == 0 or done_training
        ):
            pufferlib.utils.run_wosac_eval_in_subprocess(self.config, self.logger, self.global_step)

        if self.config["eval"]["human_replay_eval"] and (
            self.epoch % self.config["eval"]["eval_interval"] == 0 or done_training
        ):
            pufferlib.utils.run_human_replay_eval_in_subprocess(self.config, self.logger, self.global_step)

        if self.config["eval"]["multi_scenario_eval"] and (
            self.epoch % self.config["eval"]["eval_interval"] == 0 or done_training
        ):
            # Get evaluation settings from config
            eval_simulation_mode = self.config["eval"]["multi_scenario_simulation_mode"]
            num_agents_eval = self.config["eval"]["num_agents"]
            map_dir = self.config["eval"]["map_dir"]

            # Build eval_overrides using helper function
            eval_overrides = build_eval_overrides(
                simulation_mode=eval_simulation_mode,
                num_agents=num_agents_eval,
                num_scenarios=self.config["eval"]["multi_scenario_num_scenarios"],
                map_dir=map_dir,
                num_carla_maps=self.config["eval"].get("num_carla_maps", 8),
                agents_per_scene=self.config["eval"].get("agents_per_scene", 30),
                scenario_length=self.config["eval"].get("scenario_length"),
            )

            # Build eval args by applying overrides to training config
            eval_args = load_eval_multi_scenarios_config(
                env_name=self.config["env"],
                model_path=None,  # No saved model - using current policy in memory
                eval_overrides=eval_overrides,
            )
            # Add inline-specific settings
            eval_args["global_step"] = self.global_step  # Log by global step for TensorBoard
            eval_args["num_scenarios"] = self.config["eval"]["multi_scenario_num_scenarios"]
            eval_args["eval_simulation"] = eval_simulation_mode

            # Mark this as inline evaluation and set results folder in experiments
            eval_args["inline_eval"] = True  # Flag to indicate inline evaluation during training
            experiment_name = f"{self.config['env']}_{self.logger.run_id}"
            eval_args["load_model_path"] = os.path.join(
                self.config["data_dir"], experiment_name, "models", f"inline_epoch_{self.epoch}.pt"
            )
            # For inline eval, results go in experiments folder instead of benchmark
            eval_args["eval_results_dir"] = os.path.join(
                self.config["data_dir"],
                experiment_name,
                "validation",
                f"epoch_{self.epoch}",
                self.config["eval"]["multi_scenario_simulation_mode"],
            )

            # Call eval_multi_scenarios inline with current policy and logger
            print(f"\n🔄 Running multi-scenario evaluation at step {self.global_step}...")
            eval_multi_scenarios(
                env_name=self.config["env"],
                args=eval_args,
                vecenv=None,  # Let it create its own eval environment
                policy=self.uncompiled_policy,  # Pass current policy
                logger=self.logger,  # Pass logger for TensorBoard logging
                metric_prefix="validation",  # Use validation_ prefix
                quiet=True,  # Suppress verbose output during inline eval
            )

        return logs

    def _ppo_loss(
        self,
        mb_obs,
        mb_actions,
        mb_logprobs,
        mb_values,
        mb_returns,
        mb_adv,
        clip_coef,
        vf_clip,
        adv_weights=None,
        unbiased_std=False,
        loss_mask=None,
    ):
        state = dict(action=mb_actions, lstm_h=None, lstm_c=None)
        logits, newvalue = self.policy(mb_obs, state)
        _, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

        newlogprob = newlogprob.view_as(mb_logprobs)
        newvalue = newvalue.view_as(mb_returns)
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

        if loss_mask is None:
            loss_mask = torch.ones_like(mb_adv, dtype=torch.bool)
        else:
            loss_mask = loss_mask.to(torch.bool)

        filtered_adv = mb_adv[loss_mask]
        if filtered_adv.numel() > 1:
            f_mean = filtered_adv.mean()
            f_std = filtered_adv.std(unbiased=unbiased_std)
        elif filtered_adv.numel() == 1:
            f_mean = filtered_adv[0]
            f_std = torch.ones((), device=mb_adv.device, dtype=mb_adv.dtype)
        else:
            # This minibatch contains only target transitions, so the loss mask will zero it out.
            f_mean = torch.zeros((), device=mb_adv.device, dtype=mb_adv.dtype)
            f_std = torch.ones((), device=mb_adv.device, dtype=mb_adv.dtype)

        mb_adv = (mb_adv - f_mean) / (f_std + 1e-8)
        if adv_weights is not None:
            mb_adv = adv_weights * mb_adv

        loss_mask_f = loss_mask.to(mb_adv.dtype)
        denominator = loss_mask_f.sum().clamp_min(1.0)

        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = (torch.max(pg_loss1, pg_loss2) * loss_mask_f).sum() / denominator

        if vf_clip is not None:
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * (torch.max(v_loss_unclipped, v_loss_clipped) * loss_mask_f).sum() / denominator
        else:
            v_loss = 0.5 * (((newvalue - mb_returns) ** 2) * loss_mask_f).sum() / denominator

        entropy = entropy.view_as(mb_logprobs)
        entropy_loss = (entropy * loss_mask_f).sum() / denominator
        loss = pg_loss + self.config["vf_coef"] * v_loss - self.config["ent_coef"] * entropy_loss

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

    def _train_ppo_trajectory(self, losses, profile, epoch):
        config = self.config
        device = config["device"]

        b0 = config["adv_sampling_prio_beta0"]
        a = config["adv_sampling_prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile("train_misc", epoch)
            self.amp_context.__enter__()

            masks = self.masks.bool()
            terminals = torch.maximum(self.terminals, (~masks).float())
            advantages = torch.zeros_like(self.values, device=device)
            advantages = compute_puff_advantage(
                self.values,
                self.rewards,
                terminals,
                self.ratio,
                advantages,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            advantages.masked_fill_(~masks, 0.0)

            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta

            profile("train_copy", epoch)
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_adv = advantages[idx]
            mb_target_masks = self.target_masks[idx]
            loss_mask = self.masks[idx] & (~mb_target_masks)

            if not config["use_rnn"]:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            profile("train_forward", epoch)
            loss, newvalue, ratio, stats = self._ppo_loss(
                mb_obs,
                mb_actions,
                mb_logprobs,
                mb_values,
                mb_returns,
                mb_adv,
                clip_coef,
                vf_clip,
                adv_weights=mb_prio,
                unbiased_std=True,
                loss_mask=loss_mask,
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

        valid_idx = torch.nonzero((self.masks & (~self.target_masks)).reshape(-1), as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            return float("nan")

        y_pred = self.values.reshape(-1)[valid_idx]
        y_true = (advantages + self.values).reshape(-1)[valid_idx]
        var_y = y_true.var()
        return float("nan") if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()

    def _train_ppo_transition(self, losses, profile, epoch):
        config = self.config
        device = config["device"]

        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]

        masks = self.masks.bool()
        terminals = torch.maximum(self.terminals, (~masks).float())
        advantages = compute_puff_advantage(
            self.values,
            self.rewards,
            terminals,
            torch.ones_like(self.values, device=device),
            torch.zeros_like(self.values, device=device),
            config["gamma"],
            config["gae_lambda"],
            1.0,
            1.0,
        )
        advantages = advantages.masked_fill(~masks, 0.0)
        returns = advantages + self.values

        flat_advantages_f = advantages.reshape(-1)
        flat_masks_f = masks.reshape(-1).bool()
        flat_target_masks_f = self.target_masks.reshape(-1).bool()
        total_transitions = flat_masks_f.numel()
        valid_idx = torch.nonzero(flat_masks_f & (~flat_target_masks_f), as_tuple=False).flatten()

        filter_metrics = {
            "masked_fraction": 1.0 - (valid_idx.numel() / max(total_transitions, 1)),
            "kept_fraction": 0.0,
            "filtered_fraction": 1.0,
        }

        ewma_beta = config["adv_filter_ewma_beta"]
        threshold_scale = config["adv_filter_threshold_scale"]
        valid_abs_adv = flat_advantages_f[valid_idx].abs()
        current_max = valid_abs_adv.max().item() if valid_abs_adv.numel() > 0 else 0.0
        self.ema_max = current_max if epoch == 0 else ewma_beta * current_max + (1 - ewma_beta) * self.ema_max
        threshold = threshold_scale * self.ema_max

        keep_mask = valid_abs_adv >= threshold
        keep_idx = valid_idx[keep_mask]
        num_valid, num_kept = valid_idx.numel(), keep_idx.numel()

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            # Synchronize the number of kept transitions in multi-GPU setting to keep synchronization
            kept_tensor = torch.tensor([num_kept], device=device)
            torch.distributed.all_reduce(kept_tensor, op=torch.distributed.ReduceOp.MIN)
            _min_num_kept = kept_tensor.item()
            if num_kept > _min_num_kept:
                if _min_num_kept == 0:
                    keep_idx = keep_idx[:0]
                else:
                    top_idx = torch.topk(valid_abs_adv[keep_mask], _min_num_kept, largest=True, sorted=False).indices
                    keep_idx = keep_idx[top_idx]

        filter_metrics["kept_fraction"] = num_kept / max(num_valid, 1)
        filter_metrics["filtered_fraction"] = 1.0 - filter_metrics["kept_fraction"]

        losses["filter_threshold"] = threshold
        losses["ema_max"] = self.ema_max
        losses.update(filter_metrics)

        if valid_idx.numel() == 0:
            return float("nan")

        obs_shape = self.vecenv.single_observation_space.shape
        flat_obs = self.observations.reshape(-1, *obs_shape)
        flat_actions = self.actions.reshape(-1, *self.actions.shape[2:])
        flat_logprobs = self.logprobs.reshape(-1)
        flat_values = self.values.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)

        self.optimizer.zero_grad()
        total_minibatches = 0
        pending_minibatches = 0

        for _ in range(config["update_epochs"]):
            permutation = keep_idx[torch.randperm(keep_idx.numel(), device=keep_idx.device)]
            for start in range(0, permutation.numel(), self.minibatch_size):
                profile("train_copy", epoch)
                mb_idx = permutation[start : start + self.minibatch_size]
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
                    clip_coef,
                    vf_clip,
                    unbiased_std=False,
                )
                self.amp_context.__enter__()  # TODO: AMP needs some debugging

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
        return float("nan") if var_y == 0 else (1 - (y_true - y_pred).var(unbiased=False) / var_y).item()

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
        logs = {
            "SPS": dist_sum(self.sps, device),
            "agent_steps": agent_steps,
            "uptime": time.time() - self.start_time,
            "epoch": int(dist_sum(self.epoch, device)),  # VB Why it is a sum ?
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{metric_log_key(k): v for k, v in self.stats.items()},
            **{f"losses/{k}": v for k, v in self.losses.items()},
            **{f"performance/{k}": v["elapsed"] for k, v in self.profile},
            # **{metric_log_key(k): dist_mean(v, device) for k, v in self.stats.items()},
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
        distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        if distributed:
            torch.distributed.destroy_process_group()
            if rank != 0:
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
        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "agent_step": self.global_step,
            "update": self.epoch,
            "model_name": model_name,
            "run_id": run_id,
        }
        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.rename(state_path + ".tmp", state_path)

        best_metric = self.config.get("best_model_metric", "puffer_score")
        if best_metric in self.last_stats:
            current_score = self.last_stats[best_metric]
        elif best_metric != "puffer_score" and "puffer_score" in self.last_stats:
            current_score = self.last_stats["puffer_score"]
        else:
            current_score = self.last_stats.get("score", -float("inf"))

        if current_score > self.best_score:
            self.best_score = current_score

            best_state_file = os.path.join(path, f"best_models/best_trainer_state_{self.epoch:06d}.pt")
            os.makedirs(os.path.dirname(best_state_file), exist_ok=True)
            shutil.copy(model_path, best_state_file)
            print(f"New best model saved at epoch {self.epoch} with {best_metric} {self.best_score:.4f}")

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
        total_timesteps = config.get("global_total_timesteps", config["total_timesteps"])
        if sps != 0:
            remaining = duration((total_timesteps - agent_steps) / sps, b2, c2)

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


def _is_nonzero_distributed_rank():
    return torch.distributed.is_initialized() and torch.distributed.get_rank() != 0


def _get_shared_wandb_run_id():
    run_id = os.environ.get("WANDB_RUN_ID")
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0 and run_id is None:
            import wandb

            run_id = wandb.util.generate_id()
        shared = [run_id]
        torch.distributed.broadcast_object_list(shared, src=0)
        return shared[0]

    if run_id is not None:
        return run_id

    import wandb

    return wandb.util.generate_id()


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

        run_id = load_id or os.environ.get("WANDB_RUN_ID") or wandb.util.generate_id()
        wandb.init(
            id=run_id,
            name=args.get("wandb_name"),
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
        "commit_hash": None,
        "commit_message": None,
        "is_dirty": None,
    }

    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        git_metadata["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_metadata["commit_message"] = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        git_metadata["is_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        pass

    return git_metadata


def _save_experiment_config(args, path):
    import yaml
    import json

    if _is_nonzero_distributed_rank():
        return

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
            "adv_reward_weight_drive_conditioning",
            "trajectory_prediction_length",
            "num_trajectory_scaling_factors",
            "trajectory_scaling_factors",
            "max_boundary_segment_observations",
            "max_lane_segment_observations",
            "boundary_segment_dropout",
            "lane_segment_dropout",
            "max_partner_observations",
            "target_max_partner_obs_distance",
            "max_traffic_control_observations",
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
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    train_seed = args["train"]["seed"]
    if train_seed is None:
        train_seed = time.time_ns() & 0xFFFFFFFF
    torch.manual_seed(train_seed)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)

    target_policy = None
    target_policy_path = args["train"].get("target_policy")
    sdc_controller = str(args["env"].get("sdc_controller", "policy")).lower()
    if sdc_controller != "policy":
        if target_policy_path is not None and str(target_policy_path).lower() != "none":
            print(f"Skipping train.target_policy because env.sdc_controller={sdc_controller!r}")
    elif target_policy_path is not None and str(target_policy_path).lower() != "none":
        target_args = copy.deepcopy(args)
        target_args["load_model_path"] = target_policy_path
        target_args["policy_name"] = "TargetDrive"
        target_env = _make_target_policy_env_view(vecenv.driver_env)
        target_policy = load_policy(target_args, vecenv, env_name, policy_env=target_env)
        for param in target_policy.parameters():
            param.requires_grad = False
        target_policy.eval()

    if "LOCAL_RANK" in os.environ:
        args["train"]["device"] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        policy = policy.to(local_rank)
        if target_policy is not None:
            target_policy = target_policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank], output_device=local_rank)
        if hasattr(policy, "lstm"):
            # model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

    if args["neptune"]:
        logger = NeptuneLogger(args)
    elif args["wandb"]:
        run_id = _get_shared_wandb_run_id()
        if _is_nonzero_distributed_rank():
            logger = NoLogger(args, run_id=run_id)
        else:
            logger = WandbLogger(args, load_id=run_id)
    elif args["tb"]:
        date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = os.path.join(args["train"]["data_dir"], rf"{env_name}_" + date_time)
        logger = TensorBoardLogger(
            run_id=date_time,
            experiment_dir=experiment_dir,
        )

    train_config = dict(**args["train"], env=env_name, eval=args.get("eval", {}))
    pufferl = PuffeRL(train_config, vecenv, policy, target_policy, logger)

    path = os.path.join(args["train"]["data_dir"], f"{env_name}_{pufferl.logger.run_id}")
    _save_experiment_config(args, path)

    # Sweep needs data for early stopped runs, so send data when steps > 100M.
    # When using DDP, logs report globally aggregated agent_steps.
    logging_total_timesteps = train_config.get("global_total_timesteps", train_config["total_timesteps"])
    logging_threshold = min(0.20 * logging_total_timesteps, 100_000_000)
    all_logs = []

    while pufferl.epoch < pufferl.total_epochs:
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        try:
            pufferl.evaluate()
        except Exception:
            pufferl.vecenv.close()
            pufferl.utilization.stop()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            raise
        if train_config["device"] == "cuda":
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

            if logs["agent_steps"] > logging_threshold:
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

    # train() already logs and prints on the update that reaches done_training.
    # Avoid another final distributed logging pass before rank 0 does the
    # shutdown checkpoint and artifact handling.
    model_path = pufferl.close()
    pufferl.logger.close(model_path, early_stop=False)
    return all_logs


def eval(env_name, args=None, vecenv=None, policy=None):
    """Evaluate a policy."""

    args = args or load_config(env_name)
    args["env"]["termination_mode"] = 0

    wosac_enabled = args["eval"]["wosac_realism_eval"]
    human_replay_enabled = args["eval"]["human_replay_eval"]

    if wosac_enabled:
        args["env"]["map_dir"] = args["eval"]["map_dir"]
        dataset_name = args["env"]["map_dir"].split("/")[-1]

        print(f"Running WOSAC realism evaluation with {dataset_name} dataset.\n")
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

        backend = args["eval"]["backend"]
        assert backend == "PufferEnv" or not wosac_enabled, "WOSAC evaluation only supports PufferEnv backend."

        # Configure environment for WOSAC
        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["init_mode"] = args["eval"]["wosac_init_mode"]
        args["env"]["control_mode"] = args["eval"]["wosac_control_mode"]
        args["env"]["init_steps"] = args["eval"]["wosac_init_steps"]
        args["env"]["goal_behavior"] = args["eval"]["wosac_goal_behavior"]
        args["env"]["goal_radius"] = args["eval"]["wosac_goal_radius"]

        # Batch size configuration
        num_scenes_per_batch = args["eval"]["wosac_batch_size"]
        args["env"]["num_agents"] = num_scenes_per_batch * 10
        args["env"]["num_maps"] = args["eval"]["wosac_scenario_pool_size"]

        # Create environment and policy
        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        # Make eval class instance
        evaluator = WOSACEvaluator(args)

        # Obtain scores
        df_results = evaluator.evaluate(args, vecenv, policy)

        # Average results over scenarios
        results_dict = df_results.mean().to_dict()
        results_dict["total_num_agents"] = df_results["num_agents_per_scene"].sum()
        results_dict["total_unique_scenarios"] = df_results.index.unique().shape[0]
        results_dict["realism_meta_score_std"] = df_results["realism_meta_score"].std()
        results_dict = {k: v.item() if hasattr(v, "item") else v for k, v in results_dict.items()}

        import json

        print("\nWOSAC_METRICS_START")
        print(json.dumps(results_dict))
        print("WOSAC_METRICS_END")
        vecenv.close()
        return results_dict

    elif human_replay_enabled:
        args["env"]["map_dir"] = args["eval"]["map_dir"]
        dataset_name = args["env"]["map_dir"].split("/")[-1]
        print(f"Running human replay evaluation with {dataset_name} dataset.\n")
        from pufferlib.ocean.benchmark.evaluator import HumanReplayEvaluator

        backend = args["eval"].get("backend", "PufferEnv")
        args["env"]["map_dir"] = args["eval"]["map_dir"]
        args["env"]["num_agents"] = args["eval"]["human_replay_num_agents"]

        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["control_mode"] = args["eval"]["human_replay_control_mode"]
        args["env"]["episode_length"] = 91  # WOMD scenario length

        vecenv = vecenv or load_env(env_name, args)
        policy = policy or load_policy(args, vecenv, env_name)

        print(f"Effective number of scenarios used: {len(vecenv.driver_env.agent_offsets) - 1}")

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

        ob, info = vecenv.reset()
        driver = vecenv.driver_env
        num_agents = vecenv.observation_space.shape[0]
        device = args["train"]["device"]

        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        frames = []
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
    """Load config for evaluation, merging experiment YAML with defaults."""
    args = load_config(env_name)
    if model_path:
        experiment_dir = os.path.dirname(os.path.dirname(model_path))
        config_yaml_path = os.path.join(experiment_dir, "config.yaml")
        EXCLUDE_KEYS = eval_overrides["env"].keys()
        # Override Policy and RNN dimensions from training config
        if os.path.exists(config_yaml_path):
            print(f"Found config.yaml at {config_yaml_path}. Merging with defaults...")
            with open(config_yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f)

            for section in ["env", "policy", "rnn"]:
                if section in yaml_config and isinstance(yaml_config[section], dict):
                    for k, v in yaml_config[section].items():
                        if k not in EXCLUDE_KEYS:
                            args[section][k] = v

            # Also copy root-level keys like rnn_name, policy_name
            for key in ["rnn_name", "policy_name"]:
                if key in yaml_config:
                    args[key] = yaml_config[key]

            # Update use_rnn based on rnn_name
            args["train"]["use_rnn"] = args["rnn_name"] is not None

    # Override env parameters from evaluation config
    if eval_overrides:
        for section, section_overrides in eval_overrides.items():
            if isinstance(section_overrides, dict):
                for k, v in section_overrides.items():
                    args[section][k] = v
            else:
                args[section] = section_overrides

    return args


def build_eval_overrides(
    simulation_mode,
    num_agents,
    num_scenarios,
    map_dir=None,
    maps=None,
    num_carla_maps=8,
    agents_per_scene=30,
    scenario_length=None,
):
    """Build evaluation overrides for a given simulation mode.

    Args:
        simulation_mode: "gigaflow" or "replay"
        num_agents: agent slot budget for evaluation
        map_dir: replay dataset directory, required for replay mode
    """
    # Common reward coefficients (same for both modes)
    if scenario_length is None:
        scenario_length = 500 if simulation_mode == "gigaflow" else 91

    common_env = {
        "eval_mode": 1,
        "collision_behavior": 1,
        "offroad_behavior": 1,
        "traffic_light_behavior": 0,
        "reward_randomization": False,
        "min_agents_per_env": agents_per_scene,
        "max_agents_per_env": agents_per_scene,
        "reward_vehicle_collision": 3.0,
        "reward_offroad_collision": 3.0,
        "reward_stop_line": 1.0,
        "reward_ade": 0.0,
        "reward_goal": 1.0,
        "reward_overspeed": 0.05,
        "reward_comfort": 0.05,
        "reward_velocity": 0.0025,
        "reward_lane_align": 0.025,
        "reward_lane_center": 0.0038,
        "reward_timestep": 0.000025,
        "adversarial_termination_mode": 2,
        "reward_reverse": 0.005,
        "goal_speed": 20.0,
        "num_target_waypoints": 3,
        "min_waypoint_spacing": 30.0,
        "max_waypoint_spacing": 30.0,
        "lane_segment_dropout": 0.0,
        "boundary_segment_dropout": 0.0,
        "max_lane_segment_observations": 80,
        "max_boundary_segment_observations": 80,
    }

    if simulation_mode == "gigaflow":
        env_overrides = {
            **common_env,
            "simulation_mode": "gigaflow",
            "resample_frequency": scenario_length,
            "scenario_length": scenario_length,
            "map_dir": map_dir or "pufferlib/resources/drive/binaries/carla",
            "num_maps": num_carla_maps,
            "num_agents": num_agents,
            "termination_mode": 0.0,
        }
        if maps is not None:
            env_overrides["maps"] = maps

        eval_overrides = {"env": env_overrides}
    elif simulation_mode == "replay":
        eval_overrides = {
            "env": {
                **common_env,
                "simulation_mode": "replay",
                "resample_frequency": scenario_length,
                "scenario_length": scenario_length,
                "map_dir": map_dir or "pufferlib/resources/drive/binaries/womd",
                "maps": None,
                "num_maps": num_scenarios,
                "num_agents": num_agents,
                "termination_mode": 0.0,
                # "control_mode": "control_sdc_only",
            },
        }
    else:
        raise ValueError(f"Invalid simulation_mode: {simulation_mode}. Must be 'gigaflow' or 'replay'.")

    return eval_overrides


def verify_scenario_coverage(csv_path: str, num_scenarios: int) -> dict:
    """
    Verify that episode_metrics.csv contains all expected scenarios.

    Args:
        csv_path: Path to episode_metrics.csv
        num_scenarios: Expected number of scenarios (e.g., 1000)

    Returns:
        dict with keys:
            - complete: bool - True if all scenarios present
            - expected_count: number of expected scenarios
            - found_count: number of unique scenarios found
            - missing: sorted list of missing map names
            - extra: sorted list of unexpected map names
            - duplicates: dict mapping map_name -> count (if >1)
    """
    df = pd.read_csv(csv_path)

    # Expected: map_000, map_001, ..., map_{num_scenarios-1}
    expected = {f"map_{i:03d}" for i in range(num_scenarios)}
    found = set(df["map_name"].unique())

    missing = expected - found
    extra = found - expected

    # Check for duplicates
    counts = df["map_name"].value_counts()
    duplicates = {name: count for name, count in counts.items() if count > 1}

    complete = len(missing) == 0

    return {
        "complete": complete,
        "expected_count": num_scenarios,
        "found_count": len(found),
        "missing": sorted(missing),
        "extra": sorted(extra),
        "duplicates": duplicates,
    }


def verify_scenario_coverage_gigaflow(csv_path: str, num_scenarios: int) -> dict:
    """
    Verify gigaflow evaluation CSV: maps repeat across scenarios, so check total
    row count rather than unique map names.
    """
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    complete = total_rows == num_scenarios
    return {
        "complete": complete,
        "expected_count": num_scenarios,
        "found_count": total_rows,
    }


# Helper functions for eval_multi_scenarios and eval_multi_scenarios_render
def _export_metrics(
    global_infos,
    eval_folder,
    num_scenarios,
    quiet,
    verify_coverage=False,
    simulation_mode="replay",
    filename_suffix="",
):
    """Export episode and summary CSVs, return avg_infos dict."""
    # Episode Metrics
    try:
        df_episodes = pd.DataFrame(global_infos)
        first_cols = ["episode_id", "map_name"]
        other_cols = [col for col in df_episodes.columns if col not in first_cols]
        new_col_order = first_cols + other_cols
        df_episodes = df_episodes[new_col_order]

        if verify_coverage:
            df_episodes = df_episodes.sort_values(by=["map_name", "episode_id"])

        episode_csv_path = os.path.join(eval_folder, f"episode_metrics{filename_suffix}.csv")
        df_episodes.to_csv(episode_csv_path, index=False)
        if not quiet:
            print(f"\n✅ Per-episode metrics exported to {episode_csv_path}")

        if verify_coverage:
            if simulation_mode == "gigaflow":
                result = verify_scenario_coverage_gigaflow(episode_csv_path, num_scenarios)
                if not quiet:
                    if result["complete"]:
                        print(f"✅ All {num_scenarios} episodes present in CSV")
                    else:
                        print(
                            f"⚠️ Episode count mismatch: expected {result['expected_count']}, found {result['found_count']}"
                        )
            else:
                result = verify_scenario_coverage(episode_csv_path, num_scenarios)
                if not quiet:
                    if result["complete"]:
                        print(f"✅ All {num_scenarios} scenarios present in CSV")
                    else:
                        print(f"⚠️ Scenario coverage incomplete:")
                        print(f"   Expected: {result['expected_count']}, Found: {result['found_count']}")
                        if result["missing"]:
                            print(f"   Missing ({len(result['missing'])}): {result['missing']}")
                        if result["extra"]:
                            print(f"   Extra: {result['extra'][:10]}...")
                    if result["duplicates"]:
                        print(f"   Duplicates: {len(result['duplicates'])} scenarios have multiple entries")
                        for name, count in sorted(result["duplicates"].items()):
                            print(f"      {name}: {count} entries")
    except Exception as e:
        print(f"\n⚠️ Could not export per-episode CSV. Error: {e}")
        print("Global infos data:", global_infos)

    # Evaluation average metrics
    avg_infos = {}
    for k, v in global_infos.items():
        if k == "num_scenarios":
            avg_infos[k] = np.sum(v)
        elif v and isinstance(v[0], numbers.Number):
            avg_infos[k] = np.mean(v)
    df_summary = pd.DataFrame(list(avg_infos.items()), columns=["Metric", "Average"])
    summary_csv_path = os.path.join(eval_folder, f"evaluation_summary{filename_suffix}.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    if not quiet:
        print(f"\n✅ Average results exported to {summary_csv_path}")
        print(df_summary.to_string(index=False))

    return avg_infos


def _log_eval_metrics(logger, avg_infos, args, metric_prefix, quiet):
    """Log metrics to TensorBoard/wandb if logger is provided."""
    if logger is None or args.get("global_step") is None:
        return

    global_step = args["global_step"]

    # Create log dict with metric prefix (use / for TensorBoard grouping)
    log_dict = {}
    for metric_key, metric_value in avg_infos.items():
        if isinstance(metric_value, (int, float)):
            log_dict[f"{metric_prefix}/{metric_key}"] = float(metric_value)

    # Log to TensorBoard if available
    if hasattr(logger, "local_writer") and logger.local_writer:
        for key, value in log_dict.items():
            logger.local_writer.add_scalar(key, value, global_step)
        if not quiet:
            print(f"✅ Logged {len(log_dict)} validation metrics to TensorBoard at step {global_step}")

    # Also log to wandb/neptune if available
    if hasattr(logger, "log"):
        logger.log(log_dict, global_step)


def _load_target_policy_for_eval(args, vecenv, env_name, target_policy=None):
    sdc_controller = str(args["env"].get("sdc_controller", "policy")).lower()
    if sdc_controller != "policy":
        return None

    if target_policy is not None:
        target_policy.eval()
        return target_policy

    target_policy_path = args.get("target_policy_path") or args["train"].get("target_policy")
    if target_policy_path is None or str(target_policy_path).lower() == "none":
        raise pufferlib.APIUsageError("Adversarial eval requires train.target_policy")

    target_args = copy.deepcopy(args)
    target_args["load_model_path"] = target_policy_path
    target_args["policy_name"] = "TargetDrive"
    target_env = _make_target_policy_env_view(vecenv.driver_env)
    target_policy = load_policy(target_args, vecenv, env_name, policy_env=target_env)
    target_policy.eval()
    return target_policy


def _build_eval_target_mask(infos, vecenv, device):
    num_agents_per_worker = vecenv.driver_env.num_agents
    num_agents_per_batch = vecenv.agents_per_batch
    target_mask = torch.zeros(num_agents_per_batch, dtype=torch.bool, device=device)

    if infos is None:
        return target_mask

    def _iter_info_dicts(payload):
        if payload is None:
            return
        if isinstance(payload, dict):
            yield payload
            return
        if isinstance(payload, (list, tuple)):
            for item in payload:
                yield from _iter_info_dicts(item)

    infos = list(_iter_info_dicts(infos))

    env_counter = 0
    for information in infos:
        if not isinstance(information, dict):
            continue
        agent_offsets = information.get("agent_offsets")
        if agent_offsets is None:
            continue

        agent_offsets = torch.as_tensor(agent_offsets, dtype=torch.int64, device=device)
        target_mask[agent_offsets[:-1] + env_counter * num_agents_per_worker] = True
        env_counter += 1

    return target_mask


def _extract_episode_summaries(infos):
    if not infos:
        return []
    if isinstance(infos, dict):
        return [infos] if "map_name" in infos else []
    if isinstance(infos, list):
        if not infos:
            return []
        if isinstance(infos[0], dict):
            return [summary for summary in infos if isinstance(summary, dict) and "map_name" in summary]
        if isinstance(infos[0], list):
            summaries = []
            for sub_infos in infos:
                if isinstance(sub_infos, dict):
                    if "map_name" in sub_infos:
                        summaries.append(sub_infos)
                elif isinstance(sub_infos, list):
                    for summary in sub_infos:
                        if isinstance(summary, dict) and "map_name" in summary:
                            summaries.append(summary)
            return summaries
    return []


def _extract_completed_episode_summaries(infos):
    if not infos:
        return []
    if isinstance(infos, dict):
        return [infos] if infos.get("summary_type") == "completed_episode" else []
    if isinstance(infos, list):
        summaries = []
        for item in infos:
            summaries.extend(_extract_completed_episode_summaries(item))
        return summaries
    return []


def _render_compact_replay_job(job):
    replay_path, output_path, render_context = job
    return pufferlib.mining_viz.render_compact_replay_html(replay_path, output_path, render_context=render_context)


def _sanitize_path_component(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _get_no_model_run_names(args):
    env_args = args.get("env", {})
    sdc_controller = _sanitize_path_component(env_args.get("sdc_controller", "policy"))
    non_sdc_controller = _sanitize_path_component(env_args.get("non_sdc_controller", "policy"))
    return "controller_baselines", f"sdc_{sdc_controller}_non_sdc_{non_sdc_controller}"


def _get_eval_folder(args, adversarial=False):
    if "inline_eval" in args and args["inline_eval"] and "eval_results_dir" in args:
        return args["eval_results_dir"]

    model_path = args.get("load_model_path")
    if model_path is None:
        experiment_name, model_name = _get_no_model_run_names(args)
    else:
        model_filename_with_ext = os.path.basename(model_path)
        model_name = os.path.splitext(model_filename_with_ext)[0]
        models_dir = os.path.dirname(model_path)
        experiment_dir = os.path.dirname(models_dir)
        experiment_name = os.path.basename(experiment_dir)

    suffix = f"{args['eval_simulation']}_adversarial" if adversarial else args["eval_simulation"]
    return os.path.join("benchmark", experiment_name, model_name, suffix)


def _get_failure_mining_folder(args):
    model_path = args.get("load_model_path")
    if model_path is None:
        experiment_name, model_name = _get_no_model_run_names(args)
    else:
        model_filename_with_ext = os.path.basename(model_path)
        model_name = os.path.splitext(model_filename_with_ext)[0]
        models_dir = os.path.dirname(model_path)
        experiment_dir = os.path.dirname(models_dir)
        experiment_name = os.path.basename(experiment_dir)

    return os.path.join("failure_runs", experiment_name, model_name, args["eval_simulation"])


def _resolve_gigaflow_mining_maps(args):
    map_dir = args["env"]["map_dir"]
    all_map_files = sorted(os.path.join(map_dir, f) for f in os.listdir(map_dir) if f.endswith(".bin"))
    if not all_map_files:
        raise FileNotFoundError(f"No .bin files found in {map_dir}")

    selectors = args.get("eval_maps")
    if selectors is None:
        selectors = args.get("env", {}).get("maps")
    if isinstance(selectors, str) and selectors.strip().lower() in ("", "none", "null"):
        selectors = None
    if selectors is None:
        num_carla_maps = args.get("num_carla_maps", len(all_map_files))
        return [os.path.basename(path) for path in all_map_files[:num_carla_maps]]

    selected = []
    if isinstance(selectors, str):
        raw_selectors = [part.strip() for part in selectors.split(",") if part.strip()]
    elif isinstance(selectors, (list, tuple, set)):
        raw_selectors = list(selectors)
    else:
        raw_selectors = [selectors]

    available = {os.path.basename(path): path for path in all_map_files}
    for selector in raw_selectors:
        if isinstance(selector, int) or (isinstance(selector, str) and selector.isdigit()):
            target_name = f"opendrive__Town{int(selector):02d}.bin"
        else:
            selector = str(selector).strip()
            if selector.lower().startswith("town") and selector[4:].split(".")[0].isdigit():
                target_name = f"opendrive__Town{int(selector[4:].split('.')[0]):02d}.bin"
            else:
                target_name = os.path.basename(selector)
        if target_name not in available:
            raise ValueError(f"Unknown gigaflow mining map selector {selector!r}")
        if target_name not in selected:
            selected.append(target_name)

    return selected


def _get_random_eval_filename_suffix(args):
    parts = []

    agents_per_scene = args.get("eval_agents_per_scene") or args["eval"].get("agents_per_scene")
    if agents_per_scene is None:
        agents_per_scene = args["env"].get("min_agents_per_env")

    if agents_per_scene is not None:
        parts.append(f"agents{agents_per_scene}")

    if args.get("seed") is not None:
        parts.append(f"seed{args['seed']}")

    if args.get("target_policy_path") is not None:
        target_name = os.path.splitext(os.path.basename(args["target_policy_path"]))[0]
        parts.append(f"target_{target_name}")

    return f"_{'_'.join(parts)}" if parts else ""


def eval_multi_scenarios(
    env_name, args=None, vecenv=None, policy=None, logger=None, metric_prefix="validation", quiet=False
):
    t0 = time.time()

    if args is None:
        tmp_args = load_config(env_name)
        model_path = tmp_args.get("load_model_path")
        num_agents_eval = tmp_args["eval"]["num_agents"]
        map_dir = tmp_args["eval"]["map_dir"]

        eval_overrides = build_eval_overrides(
            simulation_mode=tmp_args["eval_simulation"],
            num_agents=num_agents_eval,
            num_scenarios=tmp_args["num_scenarios"],
            map_dir=map_dir,
            maps=tmp_args.get("eval_maps"),
            num_carla_maps=tmp_args.get("num_carla_maps", 8),
            agents_per_scene=tmp_args.get("eval_agents_per_scene") or tmp_args["eval"].get("agents_per_scene", 30),
            scenario_length=tmp_args.get("eval_scenario_length") or tmp_args["eval"].get("scenario_length"),
        )
        args = load_eval_multi_scenarios_config(env_name, model_path, eval_overrides)

    # Reproducibility — same approach as training
    seed = args["train"]["seed"] or 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    backend = args["vec"]["backend"]
    if backend == "PufferEnv":
        backend = "Multiprocessing"
    num_scenarios = args["num_scenarios"]

    num_workers = min(args["vec"]["num_envs"], num_scenarios)

    # Distribute scenarios across workers
    scenarios_per_worker = num_scenarios // num_workers
    remainder = num_scenarios % num_workers
    current_start = 0
    env_kwargs_list = []
    for j in range(num_workers):
        worker_kwargs = copy.deepcopy(args["env"])
        worker_num_scenario = scenarios_per_worker + (1 if j < remainder else 0)
        worker_kwargs["starting_map"] = current_start
        worker_kwargs["num_eval_scenarios"] = worker_num_scenario
        env_kwargs_list.append(worker_kwargs)
        current_start += worker_num_scenario

    print(f"Distributing {num_scenarios} scenarios across {num_workers} workers:")
    for j, w in enumerate(env_kwargs_list):
        start = w["starting_map"]
        count = w["num_eval_scenarios"]
        print(f"  Worker {j}: maps {start}-{start + count - 1} ({count} scenarios)")

    args["vec"] = dict(backend=backend, num_envs=num_workers, num_workers=num_workers, batch_size=num_workers)

    if vecenv is None:
        package = args["package"]
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(env_name)
        # Pass as lists to preserve per-worker env_kwargs
        env_creators = [make_env] * num_workers
        env_args = [[]] * num_workers
        vecenv = pufferlib.vector.make(env_creators, env_args=env_args, env_kwargs=env_kwargs_list, **args["vec"])

    policy = policy or load_policy(args, vecenv, env_name)
    policy.eval()
    num_agents = vecenv.observation_space.shape[0]
    device = args["train"]["device"]

    state = {}
    if args["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    eval_folder = _get_eval_folder(args, adversarial=False)
    os.makedirs(eval_folder, exist_ok=True)

    filename_suffix = _get_random_eval_filename_suffix(args)
    global_infos = {}
    scenarios_processed = 0
    vecenv.async_reset(42)

    ob, _, _, _, infos, _, _ = vecenv.recv()
    with tqdm(total=num_scenarios, desc="Processing scenarios", disable=quiet) as pbar:
        while scenarios_processed < num_scenarios:
            # Reset LSTM
            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                )

            for _ in range(args["env"]["scenario_length"]):
                with torch.no_grad():
                    ob = torch.as_tensor(ob).to(device)
                    logits, _ = policy.forward_eval(ob, state)
                    action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                    action = action.cpu().numpy().reshape(vecenv.action_space.shape)

                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

                ob, _, _, _, infos = vecenv.step(action)

                # Multi-worker backend returns infos as list of lists (one per worker)
                summaries = _extract_episode_summaries(infos)
                if summaries:
                    for env_idx, summary in enumerate(summaries):
                        env_map_name = summary["map_name"].split("/")[-1].split(".")[0]
                        summary["episode_id"] = env_idx
                        summary["map_name"] = env_map_name
                        scenarios_processed += 1
                        pbar.update(1)

                        for k, v in summary.items():
                            if k not in global_infos:
                                global_infos[k] = []
                            global_infos[k].append(v)

    avg_infos = _export_metrics(
        global_infos,
        eval_folder,
        num_scenarios,
        quiet,
        verify_coverage=True,
        simulation_mode=args["env"]["simulation_mode"],
        filename_suffix=filename_suffix,
    )
    print(f"\nTotal evaluation time: {time.time() - t0:.2f} seconds for {num_scenarios} scenarios.")
    _log_eval_metrics(logger, avg_infos, args, metric_prefix, quiet)

    # Close vectorized environment to avoid file descriptor leaks
    vecenv.close()


def eval_multi_scenarios_render(
    env_name, args=None, vecenv=None, policy=None, logger=None, metric_prefix="validation", quiet=False
):
    # Set fixed seed for reproducible evaluation
    np.random.seed(42)
    torch.manual_seed(42)

    if args is None:
        tmp_args = load_config(env_name)
        model_path = tmp_args.get("load_model_path")
        num_agents_eval = tmp_args["eval"]["num_agents"]
        map_dir = tmp_args["eval"]["map_dir"]
        eval_overrides = build_eval_overrides(
            simulation_mode=tmp_args["eval_simulation"],
            num_agents=num_agents_eval,
            num_scenarios=tmp_args["num_scenarios"],
            map_dir=map_dir,
            maps=tmp_args.get("eval_maps"),
            num_carla_maps=tmp_args.get("num_carla_maps", 8),
            agents_per_scene=tmp_args.get("eval_agents_per_scene") or tmp_args["eval"].get("agents_per_scene", 30),
            scenario_length=tmp_args.get("eval_scenario_length") or tmp_args["eval"].get("scenario_length"),
        )
        args = load_eval_multi_scenarios_config(env_name, model_path, eval_overrides)

    backend = args["vec"]["backend"]
    if backend != "PufferEnv":
        backend = "Serial"

    args["vec"] = {**args["vec"], "backend": backend, "num_envs": 1}
    args["env"]["num_eval_scenarios"] = args["num_scenarios"]  # first batch: fill as many scenarios as fit

    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    policy.eval()
    num_agents = vecenv.observation_space.shape[0]
    device = args["train"]["device"]

    state = {}
    if args["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    eval_folder = _get_eval_folder(args, adversarial=False)
    os.makedirs(eval_folder, exist_ok=True)

    if args["render"]:
        gif_folder = eval_folder + "/gif"
        os.makedirs(gif_folder, exist_ok=True)

    filename_suffix = _get_random_eval_filename_suffix(args)
    global_infos = {}
    num_scenarios = args["num_scenarios"]

    scenarios_processed = 0
    with tqdm(total=num_scenarios, desc="Processing scenarios", disable=quiet) as pbar:
        while scenarios_processed < num_scenarios:
            ob, _ = vecenv.reset()

            # Get initial states for all environments in the batch
            scenarios = vecenv.get_state()
            num_envs_in_batch = len(scenarios)
            batch_start = scenarios_processed

            # Prepare batch_size_eval for the resample that fires at end of the step loop.
            # That resample will load the NEXT batch, so cap it at remaining_after_this.
            remaining_after_this = num_scenarios - scenarios_processed - num_envs_in_batch
            vecenv.envs[0].batch_size_eval = max(1, remaining_after_this)

            map_names = []
            for env_idx in range(num_envs_in_batch):
                map_names.append(scenarios[env_idx]["map_name"].split("/")[-1].split(".")[0])

            # Reset LSTM
            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                )

            # Initialize histories as lists of lists (one list per environment)
            if args["render"]:
                agent_histories = [[] for _ in range(num_envs_in_batch)]
                traffic_histories = [[] for _ in range(num_envs_in_batch)]
                trajectory_histories = [[] for _ in range(num_envs_in_batch)]
                all_agents_obs_histories = [[] for _ in range(num_envs_in_batch)]

            for t in range(args["env"]["scenario_length"]):
                if args["render"]:
                    current_scenarios = vecenv.get_state()
                    start_obs_index = 0

                    # Loop through every environment in the batch to record its history
                    for env_idx in range(num_envs_in_batch):
                        env_scenario = current_scenarios[env_idx]

                        agent_histories[env_idx].append(
                            pufferlib.viz.fill_agents_state(
                                env_scenario, use_trajectory="trajectory" in args["env"]["action_type"]
                            )
                        )
                        traffic_histories[env_idx].append(pufferlib.viz.fill_traffics_state(env_scenario, t))

                        if "trajectory" in args["env"]["action_type"]:
                            trajectory_histories[env_idx].append(pufferlib.viz.fill_trajectories(env_scenario, t))

                        # Collect observation dictionaries for ALL active agents in THIS environment at timestep t
                        if args["render_obs"]:
                            step_obs_dict = {}
                            if env_idx > 0:
                                start_obs_index += current_scenarios[env_idx - 1]["active_agent_count"]
                            for agent_idx in range(env_scenario["active_agent_count"]):
                                agent_id = env_scenario["active_agent_indices"][agent_idx]
                                step_obs_dict[int(agent_id)] = pufferlib.viz.extract_obs_frame(
                                    ob,
                                    env_scenario,
                                    args,
                                    timestep=t,
                                    obs_index=start_obs_index + agent_idx,
                                    agent_idx=agent_idx,
                                    head_north=True,
                                )
                            all_agents_obs_histories[env_idx].append(step_obs_dict)

                with torch.no_grad():
                    ob = torch.as_tensor(ob).to(device)
                    logits, _ = policy.forward_eval(ob, state)
                    action, _, _ = pufferlib.pytorch.sample_logits(logits, deterministic=True)
                    action = action.cpu().numpy().reshape(vecenv.action_space.shape)

                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

                ob, _, _, _, infos = vecenv.step(action)

                # Serial backend returns infos as single list (infos[0] is the env's info list)
                summaries = _extract_episode_summaries(infos)
                if summaries:
                    for env_idx, summary in enumerate(summaries):
                        env_map_name = summary["map_name"].split("/")[-1].split(".")[0]
                        summary["episode_id"] = batch_start + env_idx
                        summary["env_id"] = env_idx
                        summary["map_name"] = env_map_name

                        for k, v in summary.items():
                            if k not in global_infos:
                                global_infos[k] = []
                            global_infos[k].append(v)

            if args["render"]:
                # Loop through every environment to generate its specific HTML replay
                for env_idx in range(num_envs_in_batch):
                    global_episode_id = batch_start + env_idx
                    # Ensure we don't render padding environments if num_scenarios isn't perfectly divisible by batch_size
                    if global_episode_id >= num_scenarios:
                        break
                    env_map_name = map_names[env_idx]

                    pufferlib.viz.generate_interactive_replay(
                        current_scenarios[env_idx],
                        agent_histories[env_idx],
                        traffic_histories[env_idx],
                        trajectory_histories[env_idx],
                        all_agents_obs_histories[env_idx],
                        f"{gif_folder}/{env_map_name}{filename_suffix}_{global_episode_id:03d}.html",
                        head_north=True,
                    )

            scenarios_processed += num_envs_in_batch
            pbar.update(num_envs_in_batch)

    if args["render"]:
        pufferlib.viz.build_gallery_index(gif_folder)

    avg_infos = _export_metrics(
        global_infos,
        eval_folder,
        num_scenarios,
        quiet,
        verify_coverage=False,
        filename_suffix=filename_suffix,
    )
    _log_eval_metrics(logger, avg_infos, args, metric_prefix, quiet)

    # Close vectorized environment to avoid file descriptor leaks
    vecenv.close()


def _render_adversarial_serial(
    env_name,
    args=None,
    vecenv=None,
    policy=None,
    target_policy=None,
    logger=None,
    metric_prefix="validation",
    quiet=False,
):
    if args is None:
        tmp_args = load_config(env_name)
        model_path = tmp_args.get("load_model_path")
        num_agents_eval = tmp_args["eval"]["num_agents"]
        map_dir = tmp_args["eval"]["map_dir"]
        eval_overrides = build_eval_overrides(
            simulation_mode=tmp_args["eval_simulation"],
            num_agents=num_agents_eval,
            num_scenarios=tmp_args["num_scenarios"],
            map_dir=map_dir,
            maps=tmp_args.get("eval_maps"),
            num_carla_maps=tmp_args.get("num_carla_maps", 8),
            agents_per_scene=tmp_args.get("eval_agents_per_scene") or tmp_args["eval"].get("agents_per_scene", 30),
            scenario_length=tmp_args.get("eval_scenario_length") or tmp_args["eval"].get("scenario_length"),
        )
        args = load_eval_multi_scenarios_config(env_name, model_path, eval_overrides)

    if args.get("seed") is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        args["vec"]["seed"] = args["seed"]
    else:
        args["vec"]["seed"] = None

    backend = args["vec"]["backend"]
    if backend != "PufferEnv":
        backend = "Serial"

    args["vec"] = {**args["vec"], "backend": backend, "num_envs": 1}
    args["env"]["num_eval_scenarios"] = args["num_scenarios"]

    vecenv = vecenv or load_env(env_name, args)

    policy = policy or load_policy(args, vecenv, env_name)
    target_policy = _load_target_policy_for_eval(args, vecenv, env_name, target_policy)
    policy_actor = TrainableTorchActor(policy, vecenv.single_observation_space)
    target_actor = TargetTorchActor(target_policy, vecenv.driver_env) if target_policy is not None else None
    num_agents = vecenv.observation_space.shape[0]
    device = args["train"]["device"]

    state = None
    target_state = None
    if args["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy_actor.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy_actor.hidden_size, device=device),
        )
        if target_actor is not None:
            target_state = dict(
                lstm_h=torch.zeros(num_agents, target_actor.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, target_actor.hidden_size, device=device),
            )

    eval_folder = _get_eval_folder(args, adversarial=True)
    os.makedirs(eval_folder, exist_ok=True)

    if args["render"]:
        gif_folder = eval_folder + "/gif"
        os.makedirs(gif_folder, exist_ok=True)

    filename_suffix = _get_random_eval_filename_suffix(args)
    global_infos = {}
    num_scenarios = args["num_scenarios"]

    scenarios_processed = 0
    with tqdm(total=num_scenarios, desc="Processing scenarios", disable=quiet) as pbar:
        while scenarios_processed < num_scenarios:
            reset_seed = None if args.get("seed") is None else args["seed"] + scenarios_processed
            ob, infos = vecenv.reset(seed=reset_seed)

            scenarios = vecenv.get_state()
            num_envs_in_batch = len(scenarios)
            batch_start = scenarios_processed

            remaining_after_this = num_scenarios - scenarios_processed - num_envs_in_batch
            vecenv.envs[0].batch_size_eval = max(1, remaining_after_this)

            map_names = []
            for env_idx in range(num_envs_in_batch):
                map_names.append(scenarios[env_idx]["map_name"].split("/")[-1].split(".")[0])

            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy_actor.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy_actor.hidden_size, device=device),
                )
                target_state = (
                    dict(
                        lstm_h=torch.zeros(num_agents, target_actor.hidden_size, device=device),
                        lstm_c=torch.zeros(num_agents, target_actor.hidden_size, device=device),
                    )
                    if target_actor is not None
                    else None
                )

            if args["render"]:
                agent_histories = [[] for _ in range(num_envs_in_batch)]
                traffic_histories = [[] for _ in range(num_envs_in_batch)]
                trajectory_histories = [[] for _ in range(num_envs_in_batch)]
                all_agents_obs_histories = [[] for _ in range(num_envs_in_batch)]

            for t in range(args["env"]["scenario_length"]):
                if args["render"]:
                    current_scenarios = vecenv.get_state()
                    start_obs_index = 0

                    for env_idx in range(num_envs_in_batch):
                        env_scenario = current_scenarios[env_idx]

                        agent_histories[env_idx].append(
                            pufferlib.viz.fill_agents_state(
                                env_scenario, use_trajectory="trajectory" in args["env"]["action_type"]
                            )
                        )
                        traffic_histories[env_idx].append(pufferlib.viz.fill_traffics_state(env_scenario, t))

                        if "trajectory" in args["env"]["action_type"]:
                            trajectory_histories[env_idx].append(pufferlib.viz.fill_trajectories(env_scenario, t))

                        if args["render_obs"]:
                            step_obs_dict = {}
                            if env_idx > 0:
                                start_obs_index += current_scenarios[env_idx - 1]["active_agent_count"]
                            for agent_idx in range(env_scenario["active_agent_count"]):
                                agent_id = env_scenario["active_agent_indices"][agent_idx]
                                step_obs_dict[int(agent_id)] = pufferlib.viz.extract_obs_frame(
                                    ob,
                                    env_scenario,
                                    args,
                                    timestep=t,
                                    obs_index=start_obs_index + agent_idx,
                                    agent_idx=agent_idx,
                                    head_north=True,
                                )
                            all_agents_obs_histories[env_idx].append(step_obs_dict)

                with torch.no_grad():
                    ob_tensor = torch.as_tensor(ob).to(device)
                    target_mask = _build_eval_target_mask(infos, vecenv, device)
                    step_context = dict(reward=None, done=None, env_id=None, mask=None)
                    actor_output = _route_actor_actions(
                        ob_tensor,
                        target_mask,
                        policy_actor,
                        step_context,
                        policy_recurrent_state=state if args["train"]["use_rnn"] else None,
                        target_actor=target_actor,
                        target_recurrent_state=target_state if args["train"]["use_rnn"] else None,
                        deterministic=True,
                    )
                    action = actor_output.action.cpu().numpy().reshape(vecenv.action_space.shape)

                if actor_output.clip_actions:
                    action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

                ob, _, _, _, infos = vecenv.step(action)

                summaries = _extract_episode_summaries(infos)
                if summaries:
                    for env_idx, summary in enumerate(summaries):
                        env_map_name = summary["map_name"].split("/")[-1].split(".")[0]
                        summary["episode_id"] = batch_start + env_idx
                        summary["env_id"] = env_idx
                        summary["map_name"] = env_map_name

                        for k, v in summary.items():
                            if k not in global_infos:
                                global_infos[k] = []
                            global_infos[k].append(v)

            if args["render"]:
                for env_idx in range(num_envs_in_batch):
                    global_episode_id = batch_start + env_idx
                    if global_episode_id >= num_scenarios:
                        break
                    env_map_name = map_names[env_idx]

                    pufferlib.viz.generate_interactive_replay(
                        current_scenarios[env_idx],
                        agent_histories[env_idx],
                        traffic_histories[env_idx],
                        trajectory_histories[env_idx],
                        all_agents_obs_histories[env_idx],
                        f"{gif_folder}/{env_map_name}{filename_suffix}_{global_episode_id:03d}.html",
                        head_north=True,
                    )

            scenarios_processed += num_envs_in_batch
            pbar.update(num_envs_in_batch)

    if args["render"]:
        pufferlib.viz.build_gallery_index(gif_folder)

    avg_infos = _export_metrics(
        global_infos,
        eval_folder,
        num_scenarios,
        quiet,
        verify_coverage=False,
        filename_suffix=filename_suffix,
    )
    _log_eval_metrics(logger, avg_infos, args, metric_prefix, quiet)
    vecenv.close()


def _decompress_replay_bundle(payload):
    return pickle.loads(zlib.decompress(payload))


def _render_replay_bundle_job(job):
    bundle_payload, output_path = job
    bundle = _decompress_replay_bundle(bundle_payload)
    pufferlib.viz.generate_interactive_replay(
        bundle["static_scenario"],
        bundle["agent_history"],
        bundle["traffic_history"],
        bundle["trajectory_history"],
        bundle["all_agents_obs_history"],
        output_path,
        head_north=bundle.get("head_north", True),
    )
    return output_path


def _load_adversarial_render_args(env_name, args=None):
    if args is not None:
        return args

    tmp_args = load_config(env_name)
    model_path = tmp_args.get("load_model_path")
    num_agents_eval = tmp_args["eval"]["num_agents"]
    map_dir = tmp_args["eval"]["map_dir"]
    eval_overrides = build_eval_overrides(
        simulation_mode=tmp_args["eval_simulation"],
        num_agents=num_agents_eval,
        num_scenarios=tmp_args["num_scenarios"],
        map_dir=map_dir,
        maps=tmp_args.get("eval_maps"),
        num_carla_maps=tmp_args.get("num_carla_maps", 8),
        agents_per_scene=tmp_args.get("eval_agents_per_scene") or tmp_args["eval"].get("agents_per_scene", 30),
        scenario_length=tmp_args.get("eval_scenario_length") or tmp_args["eval"].get("scenario_length"),
    )
    return load_eval_multi_scenarios_config(env_name, model_path, eval_overrides)


def _render_adversarial_buffered(
    env_name,
    args=None,
    vecenv=None,
    policy=None,
    target_policy=None,
    logger=None,
    metric_prefix="validation",
    quiet=False,
):
    args = _load_adversarial_render_args(env_name, args)

    if args.get("seed") is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        args["vec"]["seed"] = args["seed"]
    else:
        args["vec"]["seed"] = None

    backend = args["vec"]["backend"]
    if backend == "PufferEnv":
        backend = "Multiprocessing"
    num_scenarios = args["num_scenarios"]
    num_workers = min(args["vec"]["num_envs"], num_scenarios)
    agents_per_scene = args.get("eval_agents_per_scene") or args["eval"].get("agents_per_scene")
    if agents_per_scene is None:
        agents_per_scene = args["env"].get("max_agents_per_env")

    scenarios_per_worker = num_scenarios // num_workers
    remainder = num_scenarios % num_workers
    current_start = 0
    env_kwargs_list = []
    for worker_idx in range(num_workers):
        worker_kwargs = copy.deepcopy(args["env"])
        worker_kwargs["capture_replay"] = bool(args["render"])
        worker_kwargs["capture_replay_keep_failed_only"] = bool(args.get("render_failures_only", 1))
        worker_kwargs["capture_replay_always_keep_first"] = bool(args.get("always_render_first", 1))
        worker_kwargs["num_agents"] = int(agents_per_scene)
        worker_num_scenarios = scenarios_per_worker + (1 if worker_idx < remainder else 0)
        worker_kwargs["starting_map"] = current_start
        worker_kwargs["num_eval_scenarios"] = worker_num_scenarios
        env_kwargs_list.append(worker_kwargs)
        current_start += worker_num_scenarios

    args["vec"] = dict(
        backend=backend,
        num_envs=num_workers,
        num_workers=num_workers,
        batch_size=num_workers,
        seed=args["vec"].get("seed"),
    )

    if not quiet:
        print("Buffered adversarial render configuration:")
        print(f"  SDC controller: {args['env'].get('sdc_controller', 'policy')}")
        print(f"  Non-SDC controller: {args['env'].get('non_sdc_controller', 'policy')}")
        print(f"  Workers: {num_workers}")
        print(f"  Scenarios: {num_scenarios}")
        print(f"  Agents per scene: {agents_per_scene}")
        print(f"  Eval agent budget from config: {args['eval']['num_agents']}")
        print(f"  Effective buffered worker agent budget: {agents_per_scene}")
        print("  Internal scenarios per worker: 1")
        print(f"  Replay capture enabled: {bool(args['render'])}")
        print(f"  Replay obs capture enabled: {bool(args.get('render_obs'))}")
        print(f"  Failure-only keep rule: {bool(args.get('render_failures_only', 1))}")
        print(f"  Always keep first scenario: {bool(args.get('always_render_first', 1))}")

    if vecenv is None:
        package = args["package"]
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(env_name)
        env_creators = [make_env] * num_workers
        env_args = [[]] * num_workers
        vecenv = pufferlib.vector.make(env_creators, env_args=env_args, env_kwargs=env_kwargs_list, **args["vec"])

    policy = policy or load_policy(args, vecenv, env_name)
    target_policy = _load_target_policy_for_eval(args, vecenv, env_name, target_policy)
    policy_actor = TrainableTorchActor(policy, vecenv.single_observation_space)
    target_actor = TargetTorchActor(target_policy, vecenv.driver_env) if target_policy is not None else None
    num_agents = vecenv.observation_space.shape[0]
    device = args["train"]["device"]

    state = None
    target_state = None
    if args["train"]["use_rnn"]:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy_actor.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy_actor.hidden_size, device=device),
        )
        if target_actor is not None:
            target_state = dict(
                lstm_h=torch.zeros(num_agents, target_actor.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, target_actor.hidden_size, device=device),
            )

    eval_folder = _get_eval_folder(args, adversarial=True)
    os.makedirs(eval_folder, exist_ok=True)
    gif_folder = None
    if args["render"]:
        gif_folder = os.path.join(eval_folder, "gif")
        os.makedirs(gif_folder, exist_ok=True)

    filename_suffix = _get_random_eval_filename_suffix(args)
    global_infos = {}
    replay_jobs = []

    vecenv.async_reset(args.get("seed"))
    ob, _, _, _, infos, _, _ = vecenv.recv()
    scenarios_processed = 0
    with tqdm(total=num_scenarios, desc="Processing scenarios", disable=quiet) as pbar:
        while scenarios_processed < num_scenarios:
            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy_actor.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy_actor.hidden_size, device=device),
                )
                target_state = None
                if target_actor is not None:
                    target_state = dict(
                        lstm_h=torch.zeros(num_agents, target_actor.hidden_size, device=device),
                        lstm_c=torch.zeros(num_agents, target_actor.hidden_size, device=device),
                    )

            for _ in range(args["env"]["scenario_length"]):
                with torch.no_grad():
                    ob_tensor = torch.as_tensor(ob).to(device)
                    target_mask = _build_eval_target_mask(infos, vecenv, device)
                    step_context = dict(reward=None, done=None, env_id=None, mask=None)
                    actor_output = _route_actor_actions(
                        ob_tensor,
                        target_mask,
                        policy_actor,
                        step_context,
                        policy_recurrent_state=state if args["train"]["use_rnn"] else None,
                        target_actor=target_actor,
                        target_recurrent_state=target_state if args["train"]["use_rnn"] else None,
                        deterministic=True,
                    )
                    action = actor_output.action.cpu().numpy().reshape(vecenv.action_space.shape)

                if actor_output.clip_actions:
                    action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

                ob, _, _, _, infos = vecenv.step(action)
                summaries = [
                    summary
                    for summary in _extract_episode_summaries(infos)
                    if summary.get("summary_type") == "replay_window"
                ]
                if not summaries:
                    continue

                for summary in summaries:
                    summary = dict(summary)
                    replay_bundle = summary.pop("replay_bundle", None)
                    env_map_name = os.path.basename(summary["map_name"]).split(".")[0]
                    summary["map_name"] = env_map_name

                    for key, value in summary.items():
                        if key not in global_infos:
                            global_infos[key] = []
                        global_infos[key].append(value)

                    if replay_bundle is not None and gif_folder is not None:
                        global_episode_id = int(summary["episode_id"])
                        failure_flag = int(summary.get("did_target_fail", 0))
                        output_path = os.path.join(
                            gif_folder,
                            f"{env_map_name}{filename_suffix}_fail{failure_flag}_{global_episode_id:03d}.html",
                        )
                        replay_jobs.append((replay_bundle, output_path))

                    scenarios_processed += 1
                    pbar.update(1)

                if scenarios_processed >= num_scenarios:
                    break

    vecenv.close()

    if replay_jobs:
        render_workers = int(args.get("replay_render_workers", 0) or 0)
        if render_workers > 1:
            with ProcessPoolExecutor(max_workers=render_workers) as executor:
                list(
                    tqdm(
                        executor.map(_render_replay_bundle_job, replay_jobs),
                        total=len(replay_jobs),
                        desc="Rendering replays",
                        disable=quiet,
                    )
                )
        else:
            for replay_job in tqdm(replay_jobs, desc="Rendering replays", disable=quiet):
                _render_replay_bundle_job(replay_job)
        pufferlib.viz.build_gallery_index(gif_folder)

    if replay_jobs:
        rendered_rows = []
        for _, output_path in replay_jobs:
            filename = os.path.basename(output_path)
            rendered_rows.append({"filename": filename, "path": output_path})
        rendered_df = pd.DataFrame(rendered_rows)
        rendered_df.to_csv(os.path.join(eval_folder, f"rendered_replays{filename_suffix}.csv"), index=False)

    avg_infos = _export_metrics(
        global_infos,
        eval_folder,
        num_scenarios,
        quiet,
        verify_coverage=False,
        simulation_mode=args["env"]["simulation_mode"],
        filename_suffix=filename_suffix,
    )
    _log_eval_metrics(logger, avg_infos, args, metric_prefix, quiet)


def _prepare_mine_failures_args(env_name, tmp_args):
    model_path = tmp_args.get("load_model_path")
    num_agents_eval = tmp_args["eval"]["num_agents"]
    map_dir = tmp_args["eval"]["map_dir"]
    target_num_episodes = tmp_args.get("num_episodes") or tmp_args["num_scenarios"]
    requested_agents_per_scene = tmp_args.get("eval_agents_per_scene")
    requested_min_agents_per_env = tmp_args["env"].get("min_agents_per_env")
    requested_max_agents_per_env = tmp_args["env"].get("max_agents_per_env")
    capture_mining_replay = bool(tmp_args.get("capture_mining_replay", 0))
    capture_mining_replay_failures_only = bool(tmp_args.get("capture_mining_replay_failures_only", 1))
    eval_overrides = build_eval_overrides(
        simulation_mode=tmp_args["eval_simulation"],
        num_agents=num_agents_eval,
        num_scenarios=target_num_episodes,
        map_dir=map_dir,
        maps=tmp_args.get("eval_maps"),
        num_carla_maps=tmp_args.get("num_carla_maps", 8),
        agents_per_scene=requested_agents_per_scene or tmp_args["eval"].get("agents_per_scene", 30),
        scenario_length=tmp_args.get("eval_scenario_length") or tmp_args["eval"].get("scenario_length"),
    )
    args = load_eval_multi_scenarios_config(env_name, model_path, eval_overrides)
    args["num_episodes"] = target_num_episodes
    args["capture_mining_replay"] = capture_mining_replay
    args["capture_mining_replay_failures_only"] = capture_mining_replay_failures_only
    args["append_mining_run"] = bool(tmp_args.get("append_mining_run", 0))
    fixed_adv_reward_weight_drive = tmp_args.get("adv_reward_weight_drive")
    args["adv_reward_weight_drive"] = fixed_adv_reward_weight_drive
    args["adv_reward_weight_drive_bin"] = tmp_args.get("adv_reward_weight_drive_bin")
    if requested_agents_per_scene is None:
        args["env"]["min_agents_per_env"] = requested_min_agents_per_env
        args["env"]["max_agents_per_env"] = requested_max_agents_per_env
    return args


def mine_failures(env_name, args=None, vecenv=None, policy=None, target_policy=None, quiet=False):
    t0 = time.time()

    if args is None:
        args = _prepare_mine_failures_args(env_name, load_config(env_name))

    if args["eval_simulation"] != "gigaflow":
        raise pufferlib.APIUsageError("mine_failures currently supports gigaflow only")

    if args["train"]["use_rnn"]:
        raise pufferlib.APIUsageError("mine_failures does not support RNN policies yet")

    if args.get("seed") is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        args["vec"]["seed"] = args["seed"]
    else:
        args["vec"]["seed"] = None

    target_num_episodes = args.get("num_episodes") or args["num_scenarios"]
    backend = args["vec"]["backend"]
    if backend == "PufferEnv":
        backend = "Multiprocessing"

    num_workers = min(args["vec"]["num_envs"], target_num_episodes)
    agents_per_scene = args.get("eval_agents_per_scene") or args["eval"].get("agents_per_scene")
    min_agents_per_env = args["env"].get("min_agents_per_env")
    max_agents_per_env = args["env"].get("max_agents_per_env")
    worker_agent_budget = args["env"].get("num_agents")
    capture_mining_replay = bool(args.get("capture_mining_replay", 0))
    capture_mining_replay_failures_only = bool(args.get("capture_mining_replay_failures_only", 1))
    append_mining_run = bool(args.get("append_mining_run", 0))
    fixed_adv_reward_weight_drive = args.get("adv_reward_weight_drive")
    fixed_adv_reward_weight_drive_bin = args.get("adv_reward_weight_drive_bin")
    if fixed_adv_reward_weight_drive is not None:
        fixed_adv_reward_weight_drive = float(fixed_adv_reward_weight_drive)
        if not 0.0 <= fixed_adv_reward_weight_drive <= 1.0:
            raise pufferlib.APIUsageError("--adv-reward-weight-drive must be in [0, 1]")
    args["env"]["compute_eval_metrics"] = True

    selected_map_names = _resolve_gigaflow_mining_maps(args)
    if not selected_map_names:
        raise pufferlib.APIUsageError("mine_failures requires at least one gigaflow map")

    env_kwargs_list = []
    for worker_idx in range(num_workers):
        worker_kwargs = copy.deepcopy(args["env"])
        worker_kwargs["emit_completed_episodes"] = True
        worker_kwargs["capture_replay"] = False
        worker_kwargs["capture_compact_replay"] = capture_mining_replay
        worker_kwargs["capture_compact_replay_failures_only"] = capture_mining_replay_failures_only
        worker_kwargs["compute_eval_metrics"] = True
        worker_kwargs["deterministic_traffic_lights"] = True
        if fixed_adv_reward_weight_drive is not None:
            worker_kwargs["adv_reward_weight_drive_conditioning"] = True
            worker_kwargs["adv_reward_weight_drive_override"] = fixed_adv_reward_weight_drive
        worker_kwargs["eval_mode"] = 0
        worker_kwargs["resample_frequency"] = 0
        worker_kwargs["starting_map"] = 0
        worker_kwargs["num_eval_scenarios"] = 1
        worker_kwargs["num_maps"] = 1
        worker_kwargs["maps"] = selected_map_names[worker_idx % len(selected_map_names)]
        env_kwargs_list.append(worker_kwargs)

    args["vec"] = dict(
        backend=backend,
        num_envs=num_workers,
        num_workers=num_workers,
        batch_size=num_workers,
        seed=args["vec"].get("seed"),
    )

    if not quiet:
        print("Failure mining configuration:")
        print(f"  SDC controller: {args['env'].get('sdc_controller', 'policy')}")
        print(f"  Non-SDC controller: {args['env'].get('non_sdc_controller', 'policy')}")
        print(f"  Target episodes: {target_num_episodes}")
        print(f"  Worker count: {num_workers}")
        print(f"  Worker agent budget: {worker_agent_budget}")
        print(f"  Min agents per env: {min_agents_per_env}")
        print(f"  Max agents per env: {max_agents_per_env}")
        print("  Compute eval metrics: True (forced in mine_failures)")
        print("  Deterministic traffic lights: True (forced in mine_failures)")
        if agents_per_scene is not None:
            print(f"  Eval agents per scene override: {agents_per_scene}")
        print(f"  Capture compact replay: {capture_mining_replay}")
        if capture_mining_replay:
            print(f"  Capture failures only: {capture_mining_replay_failures_only}")
        if fixed_adv_reward_weight_drive is not None:
            print(f"  Adv reward weight drive: {fixed_adv_reward_weight_drive:.6g}")
        print(f"  Eval simulation: {args['eval_simulation']}")
        print(f"  Worker map assignment: {', '.join(selected_map_names)}")

    if vecenv is None:
        package = args["package"]
        module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
        env_module = importlib.import_module(module_name)
        make_env = env_module.env_creator(env_name)
        env_creators = [make_env] * num_workers
        env_args = [[]] * num_workers
        vecenv = pufferlib.vector.make(env_creators, env_args=env_args, env_kwargs=env_kwargs_list, **args["vec"])

    policy = policy or load_policy(args, vecenv, env_name)
    target_policy = _load_target_policy_for_eval(args, vecenv, env_name, target_policy)
    policy_actor = TrainableTorchActor(policy, vecenv.single_observation_space)
    target_actor = TargetTorchActor(target_policy, vecenv.driver_env) if target_policy is not None else None
    num_agents = vecenv.observation_space.shape[0]
    device = args["train"]["device"]

    state = None
    target_state = None

    output_folder = _get_failure_mining_folder(args)
    os.makedirs(output_folder, exist_ok=True)
    filename_suffix = _get_random_eval_filename_suffix(args)
    episodes_csv_path = os.path.join(output_folder, f"episodes{filename_suffix}.csv")
    summary_path = os.path.join(output_folder, f"summary{filename_suffix}.json")
    replay_output_folder = os.path.join(output_folder, f"replays{filename_suffix}")
    if capture_mining_replay:
        os.makedirs(replay_output_folder, exist_ok=True)

    existing_episodes_df = None
    existing_episode_count = 0
    next_episode_id = 0
    if append_mining_run and os.path.exists(episodes_csv_path):
        existing_episodes_df = pd.read_csv(episodes_csv_path)
        existing_episode_count = int(len(existing_episodes_df))
        if "episode_id" not in existing_episodes_df.columns:
            raise pufferlib.APIUsageError(f"Cannot append to {episodes_csv_path}: missing required 'episode_id' column")
        existing_episode_ids = pd.to_numeric(existing_episodes_df["episode_id"], errors="coerce").dropna()
        if not existing_episode_ids.empty:
            next_episode_id = int(existing_episode_ids.max()) + 1
        else:
            next_episode_id = existing_episode_count

    vecenv.async_reset(args.get("seed"))
    ob, _, _, _, infos, _, _ = vecenv.recv()
    completed_episode_rows = []
    replay_files_written = 0
    replay_bytes_written = 0
    with tqdm(total=target_num_episodes, desc="Mining episodes", disable=quiet) as pbar:
        while len(completed_episode_rows) < target_num_episodes:
            with torch.no_grad():
                ob_tensor = torch.as_tensor(ob).to(device)
                target_mask = _build_eval_target_mask(infos, vecenv, device)
                step_context = dict(reward=None, done=None, env_id=None, mask=None)
                actor_output = _route_actor_actions(
                    ob_tensor,
                    target_mask,
                    policy_actor,
                    step_context,
                    policy_recurrent_state=state,
                    target_actor=target_actor,
                    target_recurrent_state=target_state,
                    deterministic=True,
                )
                action = actor_output.action.cpu().numpy().reshape(vecenv.action_space.shape)

            if actor_output.clip_actions:
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

            ob, _, _, _, infos = vecenv.step(action)
            summaries = _extract_completed_episode_summaries(infos)
            if not summaries:
                continue

            for summary in summaries:
                summary = dict(summary)
                summary.pop("summary_type", None)
                replay_bundle = summary.pop("compact_replay_bundle", None)
                summary["episode_id"] = next_episode_id
                next_episode_id += 1
                map_name = summary.get("map_name")
                if isinstance(map_name, str):
                    summary["map_name"] = os.path.basename(map_name).split(".")[0]
                if fixed_adv_reward_weight_drive is not None:
                    summary["adv_reward_weight_drive"] = fixed_adv_reward_weight_drive
                    if fixed_adv_reward_weight_drive_bin is not None:
                        summary["adv_reward_weight_drive_bin"] = int(fixed_adv_reward_weight_drive_bin)
                summary["has_replay"] = 0
                summary["replay_path"] = None
                if replay_bundle is not None:
                    replay_path = os.path.join(replay_output_folder, f"episode_{summary['episode_id']:06d}.replay.zlib")
                    with open(replay_path, "wb") as replay_file:
                        replay_file.write(replay_bundle)
                    summary["has_replay"] = 1
                    summary["replay_path"] = replay_path
                    replay_files_written += 1
                    replay_bytes_written += len(replay_bundle)
                completed_episode_rows.append(summary)
                pbar.update(1)
                if len(completed_episode_rows) >= target_num_episodes:
                    break

    vecenv.close()

    completed_episode_rows = completed_episode_rows[:target_num_episodes]
    new_episodes_df = pd.DataFrame(completed_episode_rows)
    if existing_episodes_df is not None:
        episodes_df = pd.concat([existing_episodes_df, new_episodes_df], ignore_index=True, sort=False)
    else:
        episodes_df = new_episodes_df
    episodes_df.to_csv(episodes_csv_path, index=False)

    numeric_means = episodes_df.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
    with open(summary_path, "w") as f:
        json.dump(
            {
                "append_mode": bool(append_mining_run),
                "existing_episodes_before_append": int(existing_episode_count),
                "new_episodes_written": int(len(new_episodes_df)),
                "num_episodes": int(len(episodes_df)),
                "elapsed_seconds": time.time() - t0,
                "replay_files_written": int(replay_files_written),
                "replay_megabytes_written": float(replay_bytes_written / (1024 * 1024)),
                "metrics_mean": {k: float(v) for k, v in numeric_means.items()},
            },
            f,
            indent=2,
        )

    if not quiet:
        if append_mining_run and existing_episode_count > 0:
            print(
                f"\nAppended {len(new_episodes_df)} completed episodes to {episodes_csv_path} "
                f"(total={len(episodes_df)})"
            )
        else:
            print(f"\nWrote {len(episodes_df)} completed episodes to {episodes_csv_path}")
        print(f"Wrote summary metrics to {summary_path}")
        print(f"Total mining time: {time.time() - t0:.2f} seconds.")

    return episodes_df


def mine_adv_reward_weight_drive_sweep(env_name, args=None, quiet=False):
    base_args = args or load_config(env_name)
    n_bins = base_args.get("adv_reward_weight_drive_bins")
    if n_bins is None:
        n_bins = 10
    n_bins = int(n_bins)
    episodes_per_bin = int(base_args.get("episodes_per_lambda_bin") or base_args.get("num_episodes") or 1000)
    if n_bins < 1:
        raise pufferlib.APIUsageError("--adv-reward-weight-drive-bins must be >= 1")
    if episodes_per_bin < 1:
        raise pufferlib.APIUsageError("--episodes-per-lambda-bin must be >= 1")

    drive_weight_values = [idx / n_bins for idx in range(n_bins + 1)]
    merged_df = None
    if not quiet:
        print("Adv reward weight drive sweep:")
        print(f"  Bins: {n_bins} ({len(drive_weight_values)} values)")
        print(f"  Episodes per value: {episodes_per_bin}")
        print(f"  Total episodes: {episodes_per_bin * len(drive_weight_values)}")

    for bin_idx, drive_weight in enumerate(drive_weight_values):
        run_args = copy.deepcopy(base_args)
        run_args["num_episodes"] = episodes_per_bin
        run_args["adv_reward_weight_drive"] = float(drive_weight)
        run_args["adv_reward_weight_drive_bin"] = bin_idx
        run_args["append_mining_run"] = bool(base_args.get("append_mining_run", 0)) or bin_idx > 0
        prepared_args = _prepare_mine_failures_args(env_name, run_args)
        if not quiet:
            print(f"\nAdv reward weight drive bin {bin_idx}/{n_bins}: value={drive_weight:.6g}")
        merged_df = mine_failures(env_name, args=prepared_args, quiet=quiet)

    return merged_df


def render_mined_failures(env_name, args=None, quiet=False):
    args = args or load_config(env_name)
    episodes_csv_path = args.get("episodes_csv_path")
    if not episodes_csv_path:
        raise pufferlib.APIUsageError("render_mined_failures requires --episodes-csv-path")
    if not os.path.exists(episodes_csv_path):
        raise FileNotFoundError(f"Episodes CSV not found: {episodes_csv_path}")

    episodes_df = pd.read_csv(episodes_csv_path)
    if "has_replay" not in episodes_df.columns or "replay_path" not in episodes_df.columns:
        raise pufferlib.APIUsageError(
            "Episodes CSV does not contain replay columns. Re-run mine_failures with replay capture."
        )

    render_dir = os.path.join(
        os.path.dirname(episodes_csv_path),
        f"rendered_{os.path.splitext(os.path.basename(episodes_csv_path))[0]}",
    )
    os.makedirs(render_dir, exist_ok=True)

    replay_rows = episodes_df[
        episodes_df["has_replay"].fillna(0).astype(int).eq(1) & episodes_df["replay_path"].notna()
    ].copy()
    replay_rows = replay_rows[replay_rows["replay_path"].map(os.path.exists)]

    jobs = []
    render_lookup = {}
    for row in replay_rows.to_dict(orient="records"):
        episode_id = int(row["episode_id"])
        output_path = os.path.join(render_dir, f"episode_{episode_id:06d}.html")
        render_lookup[episode_id] = os.path.relpath(output_path, render_dir)

    replay_records = []
    for row in replay_rows.to_dict(orient="records"):
        episode_id = int(row["episode_id"])
        record = dict(row)
        record["episode_id"] = episode_id
        record["replay_path"] = row["replay_path"]
        record["href"] = render_lookup[episode_id]
        record["did_target_fail"] = int(bool(row.get("did_target_fail", 0)))
        replay_records.append(record)
    replay_records.sort(key=lambda item: item["episode_id"])

    for idx, record in enumerate(replay_records):
        output_path = os.path.join(render_dir, f"episode_{record['episode_id']:06d}.html")
        render_context = {
            "navigation": {
                "index_html": "index.html",
                "prev_html": replay_records[idx - 1]["href"] if idx > 0 else None,
                "next_html": replay_records[idx + 1]["href"] if idx + 1 < len(replay_records) else None,
                "episodes": [
                    {
                        "episode_id": item["episode_id"],
                        "href": item["href"],
                        "map_name": item.get("map_name"),
                        "scenario_id": item.get("scenario_id"),
                        "did_target_fail": item.get("did_target_fail", 0),
                        "adv_reward_weight_drive": item.get("adv_reward_weight_drive"),
                        "adv_drive_weight": item.get("adv_drive_weight"),
                    }
                    for item in replay_records
                ],
            },
            "summary": dict(record),
        }
        jobs.append((record["replay_path"], output_path, render_context))

    render_workers = int(args.get("replay_render_workers", 0) or 0)
    if jobs:
        if render_workers > 0:
            with ProcessPoolExecutor(max_workers=render_workers) as executor:
                list(
                    tqdm(
                        executor.map(_render_compact_replay_job, jobs),
                        total=len(jobs),
                        desc="Rendering mining replays",
                        disable=quiet,
                    )
                )
        else:
            for job in tqdm(jobs, total=len(jobs), desc="Rendering mining replays", disable=quiet):
                _render_compact_replay_job(job)

    index_path = os.path.join(render_dir, "index.html")
    pufferlib.mining_viz.generate_failure_index(episodes_df, render_lookup, index_path)

    if not quiet:
        print(f"Rendered {len(jobs)} replay pages into {render_dir}")
        print(f"Wrote failure index to {index_path}")

    return render_dir


def render_adversarial(
    env_name,
    args=None,
    vecenv=None,
    policy=None,
    target_policy=None,
    logger=None,
    metric_prefix="validation",
    quiet=False,
):
    args = _load_adversarial_render_args(env_name, args)
    if args.get("render_obs"):
        return _render_adversarial_serial(
            env_name,
            args=args,
            vecenv=vecenv,
            policy=policy,
            target_policy=target_policy,
            logger=logger,
            metric_prefix=metric_prefix,
            quiet=quiet,
        )

    return _render_adversarial_buffered(
        env_name,
        args=args,
        vecenv=vecenv,
        policy=policy,
        target_policy=target_policy,
        logger=logger,
        metric_prefix=metric_prefix,
        quiet=quiet,
    )


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
    target_key = metric_log_key(args["sweep"]["metric"])

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
    pufferl = PuffeRL(train_config, vecenv, policy)

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


def load_policy(args, vecenv, env_name="", policy_env=None):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)

    device = args["train"]["device"]
    if isinstance(device, int):
        device = torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")
    policy_env = vecenv.driver_env if policy_env is None else policy_env
    policy_cls = getattr(env_module.torch, args["policy_name"])
    policy = policy_cls(policy_env, **args["policy"])

    rnn_name = args["rnn_name"]
    if rnn_name is not None:
        rnn_cls = getattr(env_module.torch, args["rnn_name"])
        policy = rnn_cls(policy_env, policy, **args["rnn"])

    policy = policy.to(device)

    load_id = args["load_id"]
    if load_id is not None:
        if args["neptune"]:
            path = NeptuneLogger(args, load_id, mode="read-only").download()
        elif args["wandb"]:
            path = WandbLogger(args, load_id).download()
        else:
            raise pufferlib.APIUsageError("No run id provided for eval")

        # Load checkpoints on CPU first so torchrun ranks do not depend on the
        # source CUDA device ids embedded in the serialized tensors.
        state_dict = torch.load(path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args["load_model_path"]
    if load_path == "latest":
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        # Load checkpoints on CPU first so torchrun ranks do not depend on the
        # source CUDA device ids embedded in the serialized tensors.
        state_dict = torch.load(load_path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)
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
        "--target-policy-path",
        type=str,
        default=None,
        help="Optional explicit target policy checkpoint for adversarial evaluation/rendering",
    )
    parser.add_argument(
        "--load-id", type=str, default=None, help="Kickstart/eval from from a finished Wandb/Neptune run"
    )
    parser.add_argument(
        "--render-mode", type=str, default="auto", choices=["auto", "human", "ansi", "rgb_array", "raylib", "None"]
    )
    parser.add_argument("--video-path", type=str, default="videos", help="Path to save videos")
    parser.add_argument("--seed", type=int, default=None, help="Optional explicit seed for evaluation/render runs")
    parser.add_argument("--num-scenarios", type=int, default=3, help="Number of scenarios to eval")
    parser.add_argument("--num-episodes", type=int, default=None, help="Number of completed episodes to mine")
    parser.add_argument(
        "--adv-reward-weight-drive",
        type=float,
        default=None,
        help="Fixed adversarial drive reward coefficient for mine_failures. If unset, adversaries use env sampling.",
    )
    parser.add_argument(
        "--adv-reward-weight-drive-bins",
        type=int,
        default=None,
        help="Number of intervals for mine_adv_reward_weight_drive_sweep; values include both 0 and 1.",
    )
    parser.add_argument(
        "--episodes-per-lambda-bin",
        type=int,
        default=None,
        help="Completed episodes to mine for each value in mine_adv_reward_weight_drive_sweep.",
    )
    parser.add_argument(
        "--append-mining-run",
        type=int,
        default=0,
        help="Append new mined episodes to an existing mining CSV instead of overwriting it",
    )
    parser.add_argument("--episodes-csv-path", type=str, default=None, help="Path to a mined episodes CSV")
    parser.add_argument(
        "--capture-mining-replay",
        type=int,
        default=0,
        help="Capture compact replay files during mine_failures",
    )
    parser.add_argument(
        "--capture-mining-replay-failures-only",
        type=int,
        default=1,
        help="When capturing compact mining replay, only keep episodes where the target failed",
    )
    parser.add_argument(
        "--eval-agents-per-scene",
        type=int,
        default=None,
        help="Fixed number of agents per scenario for evaluation overrides",
    )
    parser.add_argument(
        "--eval-scenario-length",
        type=int,
        default=None,
        help="Scenario length for evaluation overrides",
    )
    parser.add_argument(
        "--num-carla-maps", type=int, default=8, help="Number of CARLA maps to use in gigaflow mode (max 8)"
    )
    parser.add_argument(
        "--eval-maps",
        type=str,
        default=None,
        help="Gigaflow-only explicit CARLA map selection for evaluation, e.g. 2,5,10",
    )
    parser.add_argument("--render", type=int, default=0, help="Rendering the evaluation")
    parser.add_argument(
        "--render-obs", type=int, default=0, help="Rendering the observation of first agent in evaluation"
    )
    parser.add_argument(
        "--render-failures-only",
        type=int,
        default=1,
        help="For buffered adversarial rendering, only keep episodes where the target failed",
    )
    parser.add_argument(
        "--always-render-first",
        type=int,
        default=1,
        help="For buffered adversarial rendering, always keep episode 0 even if the target does not fail",
    )
    parser.add_argument(
        "--replay-render-workers",
        type=int,
        default=0,
        help="Number of offline workers to use when materializing buffered replay HTML files",
    )
    parser.add_argument("--agent-index", nargs="*", type=int, default=None, help="Agent index to plot the observation")
    parser.add_argument("--save-frames", type=int, default=0)
    parser.add_argument("--gif-path", type=str, default="eval.gif")
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--max-runs", type=int, default=200, help="Max number of sweep runs")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--wandb-group", type=str, default="debug")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--neptune", action="store_true", help="Use neptune for logging")
    parser.add_argument("--neptune-name", type=str, default="pufferai")
    parser.add_argument("--neptune-project", type=str, default="ablations")
    parser.add_argument("--tb", action="store_true", help="Use tensorboard for logging")
    parser.add_argument("--local-rank", type=int, default=0, help="Used by torchrun for DDP")
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment")
    parser.add_argument(
        "--eval-simulation", type=str, default=None, help="Simulation mode for evaluation - gigaflow/replay"
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
    args["train"]["global_total_timesteps"] = args["train"]["total_timesteps"]
    # Under DDP, keep the per-rank batch geometry unchanged so launching on
    # N GPUs gives an N-times larger effective batch. Only divide the local
    # training horizon so the globally aggregated agent_steps still reach the
    # user-requested total.
    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        args["train"]["world_size"] = world_size
        args["train"]["total_timesteps"] = args["train"]["total_timesteps"] // world_size

    return args


def main():
    err = "Usage: puffer [train, eval, eval_adversarial, eval_multi_scenarios, eval_multi_scenarios_render, render_adversarial, mine_failures, mine_adv_reward_weight_drive_sweep, render_mined_failures, sweep, controlled_exp, autotune, profile, export] [env_name] [optional args]. --help for more info"
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    elif mode == "eval":
        eval(env_name=env_name)
    elif mode == "eval_multi_scenarios":
        eval_multi_scenarios(env_name=env_name)
    elif mode == "eval_multi_scenarios_render":
        eval_multi_scenarios_render(env_name=env_name)
        print("")
    elif mode == "render_adversarial":
        render_adversarial(env_name=env_name)
        print("")
    elif mode == "mine_failures":
        mine_failures(env_name=env_name)
        print("")
    elif mode == "mine_adv_reward_weight_drive_sweep":
        mine_adv_reward_weight_drive_sweep(env_name=env_name)
        print("")
    elif mode == "render_mined_failures":
        render_mined_failures(env_name=env_name)
        print("")
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
