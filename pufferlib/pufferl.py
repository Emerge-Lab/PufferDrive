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
        self.uncompiled_policy = policy
        self.policy = policy
        if config["compile"]:
            self.policy = torch.compile(policy, mode=config["compile_mode"])
            self.policy.forward_eval = torch.compile(policy, mode=config["compile_mode"])
            pufferlib.pytorch.sample_logits = torch.compile(
                pufferlib.pytorch.sample_logits, mode=config["compile_mode"]
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
        epochs = config["total_timesteps"] // config["batch_size"]
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

            self.global_step += int(mask.sum())

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
                self.logprobs[batch_rows, l] = logprob
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
                self.values[batch_rows, l] = value.flatten()
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

        behaviours_eval_enabled = self.config["eval"].get("driving_behaviours_eval", False)
        behaviours_eval_interval = int(self.config["eval"].get("driving_behaviours_eval_interval", 25))
        behaviours_config = self.config.get("driving_behaviours_eval")
        if (
            behaviours_eval_enabled
            and behaviours_config
            and behaviours_eval_interval > 0
            and (self.epoch % behaviours_eval_interval == 0 or done_training)
        ):
            self.save_checkpoint()
            pufferlib.utils.run_driving_behaviours_eval_in_subprocess(
                self.config, self.logger, self.global_step, behaviours_config
            )
            if self.config["eval"].get("render_driving_behaviours"):
                self._render_driving_behaviours(behaviours_config)

        if self.config["eval"]["multi_scenario_eval"] and (
            self.epoch % self.config["eval"]["eval_interval"] == 0 or done_training
        ):
            # Get evaluation settings from config
            eval_simulation_mode = self.config["eval"]["multi_scenario_simulation_mode"]
            num_agents_eval = self.config["eval"]["num_agents"]
            map_dir = self.config["eval"]["map_dir"]

            # Inline eval runs "clean" by default — perturbations + dropout off,
            # red-light stops enforced — so the logged validation metrics
            # track progress under controlled conditions rather than noisy
            # training perturbations. The live training policy's road slicing
            # is re-aligned to the clean env at eval time via
            # _swap_policy_obs_counts inside eval_multi_scenarios.
            clean_eval = self.config["eval"].get("clean_eval", True)
            eval_overrides = build_eval_overrides(
                simulation_mode=eval_simulation_mode,
                num_agents=num_agents_eval,
                num_scenarios=self.config["eval"]["multi_scenario_num_scenarios"],
                map_dir=map_dir,
                num_carla_maps=self.config["eval"].get("num_carla_maps", 8),
                clean=clean_eval,
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
                clean=clean_eval,
            )

        # Replay eval — sibling of multi_scenario_eval. Runs inline eval on
        # replay scenarios (e.g. nuPlan mini train bins) with control_sdc_only
        # so the policy drives the SDC while other agents follow their logged
        # trajectories. Metrics logged under "validation_replay/".
        if self.config["eval"].get("replay_eval") and (
            self.epoch % self.config["eval"]["eval_interval"] == 0 or done_training
        ):
            clean_eval = self.config["eval"].get("clean_eval", True)
            replay_num_scenarios = int(self.config["eval"]["replay_num_scenarios"])
            replay_scenario_length = int(self.config["eval"]["replay_scenario_length"])
            replay_overrides = build_eval_overrides(
                simulation_mode="replay",
                num_agents=self.config["eval"]["num_agents"],
                num_scenarios=replay_num_scenarios,
                map_dir=self.config["eval"]["replay_map_dir"],
                clean=clean_eval,
            )
            # Override defaults that build_eval_overrides sets for WOMD replay
            # (scenario_length=91) — nuPlan with duration_s=20 needs 201.
            replay_overrides["env"]["scenario_length"] = replay_scenario_length
            replay_overrides["env"]["resample_frequency"] = replay_scenario_length
            replay_overrides["env"]["control_mode"] = self.config["eval"]["replay_control_mode"]
            replay_overrides["env"]["init_steps"] = int(self.config["eval"]["replay_init_steps"])

            replay_args = load_eval_multi_scenarios_config(
                env_name=self.config["env"],
                model_path=None,
                eval_overrides=replay_overrides,
            )
            replay_args["global_step"] = self.global_step
            replay_args["num_scenarios"] = replay_num_scenarios
            replay_args["eval_simulation"] = "replay"
            replay_args["inline_eval"] = True
            experiment_name = f"{self.config['env']}_{self.logger.run_id}"
            replay_args["load_model_path"] = os.path.join(
                self.config["data_dir"], experiment_name, "models", f"inline_epoch_{self.epoch}.pt"
            )
            replay_args["eval_results_dir"] = os.path.join(
                self.config["data_dir"],
                experiment_name,
                "validation_replay",
                f"epoch_{self.epoch}",
                "replay",
            )
            print(f"\n🔄 Running replay eval at step {self.global_step}...")
            eval_multi_scenarios(
                env_name=self.config["env"],
                args=replay_args,
                vecenv=None,
                policy=self.uncompiled_policy,
                logger=self.logger,
                metric_prefix="validation_replay",
                quiet=True,
                clean=clean_eval,
            )

        # Multi-scenario render — independent interval so the heavier render
        # path doesn't have to fire every eval_interval. Mirrors the block
        # above but calls eval_multi_scenarios_render with render=True and
        # the configured backend ("egl" by default on this branch, writes
        # one mp4 per scenario via the C render.h pipeline).
        if self.config["eval"]["multi_scenario_render"] and (
            self.epoch % self.config["eval"]["multi_scenario_render_interval"] == 0 or done_training
        ):
            render_simulation_mode = self.config["eval"]["multi_scenario_simulation_mode"]
            num_agents_render = self.config["eval"]["num_agents"]
            render_map_dir = self.config["eval"]["map_dir"]

            render_overrides = build_eval_overrides(
                simulation_mode=render_simulation_mode,
                num_agents=num_agents_render,
                num_scenarios=self.config["eval"]["multi_scenario_num_scenarios"],
                map_dir=render_map_dir,
                num_carla_maps=self.config["eval"].get("num_carla_maps", 8),
            )

            render_args = load_eval_multi_scenarios_config(
                env_name=self.config["env"],
                model_path=None,
                eval_overrides=render_overrides,
            )
            render_args["global_step"] = self.global_step
            render_args["num_scenarios"] = self.config["eval"]["multi_scenario_num_scenarios"]
            render_args["eval_simulation"] = render_simulation_mode
            render_args["render"] = True  # master on/off for the render branch
            render_args["render_obs"] = False  # HTML-only; EGL path ignores

            render_args["inline_eval"] = True
            experiment_name = f"{self.config['env']}_{self.logger.run_id}"
            render_args["load_model_path"] = os.path.join(
                self.config["data_dir"], experiment_name, "models", f"inline_epoch_{self.epoch}.pt"
            )
            render_args["eval_results_dir"] = os.path.join(
                self.config["data_dir"],
                experiment_name,
                "renders",
                f"epoch_{self.epoch:08d}",
                self.config["eval"]["multi_scenario_simulation_mode"],
            )

            backend_name = self.config["eval"]["multi_scenario_render_backend"]
            print(f"\n🎬 Running multi-scenario {backend_name} render at step {self.global_step}...")
            # Render failures (missing map dir, corrupted .bin files, ffmpeg
            # absent, EGL unavailable, etc.) should NEVER crash training — the
            # render is a logging side-channel. Catch any exception here, log
            # it, and let training keep going. The upstream eval_multi_scenarios
            # metric call is separate and already ran, so metric eval continues
            # to work even if video rendering is broken.
            # Multi-view EGL render: run the full render fn once per view
            # (sim_state then bev). Each call creates a fresh vecenv that
            # starts at scenario 0, runs all scenarios with one camera, and
            # tears down. Doing both views in ONE rollout would not work
            # because Drive.step's resample fires at the last step and
            # advances starting_map_counter — a re-reset would replay the
            # NEXT batch instead of the original one.
            _bev_views = [(0, "", "sim_state"), (1, "_bev", "bev")] if backend_name == "egl" else [(0, "", "sim_state")]
            for _vmode, _vsuffix, _vlabel in _bev_views:
                try:
                    eval_multi_scenarios_render(
                        env_name=self.config["env"],
                        args=dict(render_args),
                        vecenv=None,
                        policy=self.uncompiled_policy,
                        logger=self.logger,
                        metric_prefix=f"render_{_vlabel}",
                        quiet=True,
                        render_backend=backend_name,
                        view_mode=_vmode,
                        video_suffix=_vsuffix,
                        log_view_label=_vlabel,
                        # Configurable cap: eval.render_max_steps. Default 50 until
                        # the mystery ~500-c_render-call abort is properly diagnosed.
                        # Set to 0/negative to disable the cap entirely.
                        render_max_steps=(self.config["eval"].get("render_max_steps", 50) or None),
                    )
                except Exception as e:
                    import traceback

                    print(
                        f"\n⚠️  multi_scenario_render failed (view={_vlabel}) at step {self.global_step}: "
                        f"{type(e).__name__}: {e}"
                    )
                    traceback.print_exc()
                    print("Training continues.")

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

        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=unbiased_std) + 1e-8)
        if adv_weights is not None:
            mb_adv = adv_weights * mb_adv

        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        if vf_clip is not None:
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * (newvalue - mb_returns) ** 2
            v_loss = v_loss.mean()
        entropy_loss = entropy.mean()
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
        y_true = advantages.flatten() + self.values.flatten()
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
        total_transitions = flat_masks_f.numel()
        valid_idx = torch.nonzero(flat_masks_f, as_tuple=False).flatten()

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

    def _render_driving_behaviours(self, behaviours_config):
        """Render one scenario per driving behaviour class using eval_multi_scenarios_render."""
        EVAL_SECTIONS_PREFIX = "eval_"
        backend_name = self.config["eval"].get("multi_scenario_render_backend", "egl")
        bev_views = [(0, "", "sim_state"), (1, "_bev", "bev")] if backend_name == "egl" else [(0, "", "sim_state")]

        for class_name, class_cfg in behaviours_config.items():
            if not class_name.startswith(EVAL_SECTIONS_PREFIX):
                continue
            map_dir = class_cfg.get("map_dir", "")
            if isinstance(map_dir, str):
                map_dir = map_dir.strip('"').strip("'")
            if not os.path.isdir(map_dir) or not any(f.endswith(".bin") for f in os.listdir(map_dir)):
                continue

            short = class_name[len(EVAL_SECTIONS_PREFIX) :]
            num_maps = len([f for f in os.listdir(map_dir) if f.endswith(".bin")])
            render_overrides = build_eval_overrides(
                simulation_mode="replay",
                num_agents=1,
                num_scenarios=1,
                map_dir=map_dir,
            )
            render_overrides["env"]["control_mode"] = "control_sdc_only"
            render_overrides["env"]["num_maps"] = num_maps
            render_overrides["env"]["scenario_length"] = class_cfg.get("scenario_length", 91)

            render_args = load_eval_multi_scenarios_config(
                env_name=self.config["env"],
                model_path=None,
                eval_overrides=render_overrides,
            )
            experiment_name = f"{self.config['env']}_{self.logger.run_id}"
            render_args["global_step"] = self.global_step
            render_args["num_scenarios"] = 1
            render_args["eval_simulation"] = "replay"
            render_args["render"] = True
            render_args["inline_eval"] = True
            render_args["eval_results_dir"] = os.path.join(
                self.config["data_dir"],
                experiment_name,
                "renders",
                f"epoch_{self.epoch:08d}",
                "driving_behaviours",
                short,
            )

            for vmode, vsuffix, vlabel in bev_views:
                try:
                    eval_multi_scenarios_render(
                        env_name=self.config["env"],
                        args=dict(render_args),
                        vecenv=None,
                        policy=self.uncompiled_policy,
                        logger=self.logger,
                        metric_prefix=f"driving_behaviours/{short}",
                        render_key_prefix=f"driving_behaviours/{short}/render/{vlabel}",
                        quiet=True,
                        render_backend=backend_name,
                        view_mode=vmode,
                        video_suffix=vsuffix,
                        log_view_label=vlabel,
                        render_max_steps=(self.config["eval"].get("render_max_steps", 50) or None),
                    )
                except Exception as e:
                    import traceback

                    print(f"DrivingBehavioursRender [{short}] view={vlabel}: {type(e).__name__}: {e}")
                    traceback.print_exc()

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

        current_score = self.last_stats.get("puffer_score", self.last_stats.get("score", -float("inf")))

        if current_score > self.best_score:
            self.best_score = current_score

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
            "max_boundary_segment_observations",
            "max_lane_segment_observations",
            "max_partner_observations",
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

    if "LOCAL_RANK" in os.environ:
        args["train"]["device"] = torch.cuda.current_device()
        torch.distributed.init_process_group(backend="nccl", world_size=world_size)
        policy = policy.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank], output_device=local_rank)
        if hasattr(policy, "lstm"):
            # model.lstm = policy.lstm
            model.hidden_size = policy.hidden_size

        model.forward_eval = policy.forward_eval
        policy = model.to(local_rank)

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

    train_config = dict(
        **args["train"],
        env=env_name,
        eval=args.get("eval", {}),
        driving_behaviours_eval=args.get("driving_behaviours_eval"),
    )
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    path = os.path.join(args["train"]["data_dir"], f"{env_name}_{pufferl.logger.run_id}")
    _save_experiment_config(args, path)

    # Sweep needs data for early stopped runs, so send data when steps > 100M
    logging_threshold = min(0.20 * train_config["total_timesteps"], 100_000_000)
    all_logs = []

    while pufferl.global_step < train_config["total_timesteps"]:
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

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
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
        args["env"]["num_maps"] = len([f for f in os.listdir(args["env"]["map_dir"]) if f.endswith(".bin")])

        args["vec"] = dict(backend=backend, num_envs=1)
        args["env"]["control_mode"] = args["eval"]["human_replay_control_mode"]
        args["env"]["scenario_length"] = 91  # WOMD scenario length

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


def build_eval_overrides(simulation_mode, num_agents, num_scenarios, map_dir=None, num_carla_maps=8, clean=False):
    """Build evaluation overrides for a given simulation mode.

    Args:
        simulation_mode: "gigaflow" or "replay"
        num_agents: agent slot budget for evaluation
        map_dir: replay dataset directory, required for replay mode
        clean: if True, run a "clean" eval — zero road-segment dropout and
            enforce red-light stops. Only safe when the policy is rebuilt
            from the eval env (standalone eval / render_scenario.py). Inline
            eval during training reuses the live training policy, whose
            encoder was built for the training obs shape; zeroing dropout
            there changes the obs shape and triggers a CUDA device-side
            assert. Perturbation probabilities (partner_blindness,
            phantom_braking) are always forced to zero at eval — they're
            pure randomness, they don't change the obs shape, and eval
            should be deterministic regardless of clean mode.
    """
    # Common reward coefficients (same for both modes)
    common_env = {
        "eval_mode": 1,
        # Eval always runs at dt=0.1 (10Hz), the rate the logged trajectories
        # were sampled at. Training may override dt for curriculum / speed
        # experiments, but eval metrics need the logged-rate physics or
        # replay positions drift against the real waypoints.
        "dt": 0.1,
        "collision_behavior": 1,
        "offroad_behavior": 1,
        "traffic_light_behavior": 1 if clean else 0,
        "reward_randomization": False,
        "reward_vehicle_collision": 3.0,
        "reward_offroad_collision": 3.0,
        "reward_ade": 0.0,
        "reward_goal": 1.0,
        "reward_overspeed": 0.05,
        "reward_comfort": 0.05,
        "reward_velocity": 0.0025,
        "reward_lane_align": 0.025,
        "reward_lane_center": 0.0038,
        "reward_timestep": 0.000025,
        # Always zero perturbations at eval. These don't change obs shape so
        # it's safe to force even for inline eval, and a deterministic eval
        # is what we want for tracking progress.
        "partner_blindness_prob": 0.0,
        "phantom_braking_prob": 0.0,
        "phantom_braking_trigger_prob": 0.0,
    }

    if clean:
        # Dropout changes the obs shape. Only safe when the policy is
        # rebuilt from the eval env (standalone eval / render_scenario),
        # OR when the inline caller uses _swap_policy_obs_counts.
        common_env["lane_segment_dropout"] = 0.0
        common_env["boundary_segment_dropout"] = 0.0
        # Clean eval gets a bigger partner budget too — mirrors the lane
        # story. The partner encoder is shared-MLP + max-pool, so the same
        # policy weights handle the larger count via _swap_policy_obs_counts.
        common_env["max_partner_observations"] = 32

    if simulation_mode == "gigaflow":
        eval_overrides = {
            "env": {
                **common_env,
                "simulation_mode": "gigaflow",
                "min_agents_per_env": 50,
                "max_agents_per_env": 50,
                "resample_frequency": 3000,
                "scenario_length": 3000,
                # Point at the py123d-converted CARLA towns added to this branch.
                # The older binaries/carla dir predates the 123Drive pipeline and
                # is not populated on emerge/temp_training.
                "map_dir": map_dir or "pufferlib/resources/drive/binaries/carla_py123d",
                "num_maps": num_carla_maps,
                "num_agents": num_agents,
                "termination_mode": 0.0,
            }
        }
    elif simulation_mode == "replay":
        eval_overrides = {
            "env": {
                **common_env,
                "simulation_mode": "replay",
                "resample_frequency": 91,
                "scenario_length": 91,
                "max_agents_per_env": 64,
                "map_dir": map_dir or "pufferlib/resources/drive/binaries/womd",
                "num_maps": num_scenarios,
                "num_agents": num_agents,
                "min_agents_per_env": 1,
                "termination_mode": 0.0,
                # "control_mode": "control_sdc_only",
            },
        }
    else:
        raise ValueError(f"Invalid simulation_mode: {simulation_mode}. Must be 'gigaflow' or 'replay'.")

    return eval_overrides


_SWAPPABLE_OBS_COUNTS = (
    "obs_lane_segment_count",
    "obs_boundary_segment_count",
    "max_partner_observations",
    "max_traffic_control_observations",
)


@contextlib.contextmanager
def _swap_policy_obs_counts(policy, vecenv):
    """Temporarily align the policy's obs slicing with the eval env.

    Training may use dropout > 0 or tighter partner/traffic-control caps,
    giving smaller counts. Clean eval (or an eval with --max-partner-observations
    bumped) produces a larger obs buffer. The GigaFlow encoders for lanes,
    boundaries, partners, and traffic controls are all shared MLPs with
    max-pool over their element dimension — weights are count-invariant.
    Only the obs-buffer slicing in DriveBackbone.forward depends on these
    counts, so we swap them for the duration of the eval and the same
    training policy works on the larger clean obs.
    """
    try:
        eval_env = vecenv.driver_env
        new_counts = {name: int(getattr(eval_env, name)) for name in _SWAPPABLE_OBS_COUNTS}
    except AttributeError:
        # If the eval env doesn't expose these (unknown wrapper), skip the
        # swap — forward will still work when training and eval obs shapes
        # coincide (clean=False or no overrides).
        yield
        return

    targets = [m for m in policy.modules() if all(hasattr(m, n) for n in _SWAPPABLE_OBS_COUNTS)]
    saved = [{n: getattr(m, n) for n in _SWAPPABLE_OBS_COUNTS} for m in targets]
    try:
        for m in targets:
            for n, v in new_counts.items():
                setattr(m, n, v)
        yield
    finally:
        for m, orig in zip(targets, saved):
            for n, v in orig.items():
                setattr(m, n, v)


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
def _export_metrics(global_infos, eval_folder, num_scenarios, quiet, verify_coverage=False, simulation_mode="replay"):
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

        episode_csv_path = os.path.join(eval_folder, "episode_metrics.csv")
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
    summary_csv_path = os.path.join(eval_folder, "evaluation_summary.csv")
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


def eval_multi_scenarios(
    env_name,
    args=None,
    vecenv=None,
    policy=None,
    logger=None,
    metric_prefix="validation",
    quiet=False,
    clean=False,
):
    t0 = time.time()

    if args is None:
        tmp_args = load_config(env_name)
        model_path = tmp_args.get("load_model_path")
        num_agents_eval = tmp_args["eval"]["num_agents"]
        map_dir = tmp_args["eval"]["map_dir"]

        # CLI standalone entry point: read clean_eval from the eval section
        # so users can enable it via --eval.clean-eval. Inline callers pass
        # clean= directly and come in through the args-provided branch.
        clean_from_config = tmp_args["eval"].get("clean_eval", False)
        eval_overrides = build_eval_overrides(
            simulation_mode=tmp_args["eval_simulation"],
            num_agents=num_agents_eval,
            num_scenarios=tmp_args["num_scenarios"],
            map_dir=map_dir,
            num_carla_maps=tmp_args.get("num_carla_maps", 8),
            clean=clean_from_config,
        )
        args = load_eval_multi_scenarios_config(env_name, model_path, eval_overrides)
        clean = clean or clean_from_config

    # Reproducibility — same approach as training
    seed = args["train"]["seed"] or 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    backend = args["vec"]["backend"]
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

    # Folder for evaluation results
    # For inline evaluation during training, use eval_results_dir in experiments folder
    # For standalone evaluation, use benchmark folder
    if "inline_eval" in args and args["inline_eval"] and "eval_results_dir" in args:
        eval_folder = args["eval_results_dir"]
    else:
        # Standalone evaluation path (in benchmark folder)
        model_path = args["load_model_path"]
        if model_path is None:
            eval_folder = os.path.join("benchmark", "no_policy", args["eval_simulation"])
        else:
            model_filename_with_ext = os.path.basename(model_path)
            model_name = os.path.splitext(model_filename_with_ext)[0]
            models_dir = os.path.dirname(model_path)
            experiment_dir = os.path.dirname(models_dir)
            experiment_name = os.path.basename(experiment_dir)
            eval_folder = os.path.join("benchmark", experiment_name, model_name, args["eval_simulation"])
    os.makedirs(eval_folder, exist_ok=True)

    global_infos = {}
    scenarios_processed = 0
    vecenv.async_reset(42)

    ob, _, _, _, infos, _, _ = vecenv.recv()
    # Clean eval may use different road-dropout than training. The shared
    # training policy's obs slicing needs to be aligned with this env; see
    # _swap_policy_obs_counts.
    swap_ctx = _swap_policy_obs_counts(policy, vecenv) if clean else contextlib.nullcontext()
    with swap_ctx, tqdm(total=num_scenarios, desc="Processing scenarios", disable=quiet) as pbar:
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
                if infos and infos[0]:
                    for sub_env in infos:
                        for env_idx, summary in enumerate(sub_env):
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
    )
    print(f"\nTotal evaluation time: {time.time() - t0:.2f} seconds for {num_scenarios} scenarios.")
    _log_eval_metrics(logger, avg_infos, args, metric_prefix, quiet)

    # Close vectorized environment to avoid file descriptor leaks
    vecenv.close()


def eval_multi_scenarios_render(
    env_name,
    args=None,
    vecenv=None,
    policy=None,
    logger=None,
    metric_prefix="validation",
    quiet=False,
    render_backend="html",
    view_mode=0,
    video_suffix="",
    log_view_label="render",
    render_max_steps=None,
    render_key_prefix=None,
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
            num_carla_maps=tmp_args.get("num_carla_maps", 8),
        )
        args = load_eval_multi_scenarios_config(env_name, model_path, eval_overrides)

    backend = args["vec"]["backend"]
    if backend != "PufferEnv":
        backend = "Serial"

    args["vec"] = dict(backend=backend, num_envs=1)
    args["env"]["num_eval_scenarios"] = args["num_scenarios"]  # first batch: fill as many scenarios as fit

    # Backend selection.
    #   "html" — the existing viz.generate_interactive_replay path (CPU-only,
    #            self-contained HTML per scenario).
    #   "egl"  — the C-side render.h → make_client → client_record_frame
    #            pipeline (EGL GPU context, PBO double-buffer readback,
    #            writev → ffmpeg libx264, one mp4 per scenario).
    egl_mode = bool(args.get("render")) and render_backend == "egl"
    html_mode = bool(args.get("render")) and not egl_mode
    if egl_mode:
        # Force the C env to RENDER_HEADLESS so make_client spawns ffmpeg and
        # (under DRIVE_HAS_EGL) switches the active GL context to the GPU.
        args["env"]["render_mode"] = "headless"

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

    # Folder for evaluation results
    # For inline evaluation during training, use eval_results_dir in experiments folder
    # For standalone evaluation, use benchmark folder
    if "inline_eval" in args and args["inline_eval"] and "eval_results_dir" in args:
        eval_folder = args["eval_results_dir"]
    else:
        # Standalone evaluation path (in benchmark folder)
        model_path = args["load_model_path"]
        if model_path is None:
            eval_folder = os.path.join("benchmark", "no_policy", args["eval_simulation"])
        else:
            model_filename_with_ext = os.path.basename(model_path)
            model_name = os.path.splitext(model_filename_with_ext)[0]
            models_dir = os.path.dirname(model_path)
            experiment_dir = os.path.dirname(models_dir)
            experiment_name = os.path.basename(experiment_dir)
            eval_folder = os.path.join("benchmark", experiment_name, model_name, args["eval_simulation"])
    os.makedirs(eval_folder, exist_ok=True)

    saved_cwd = None
    mp4_folder = None
    gif_folder = None
    if html_mode:
        gif_folder = eval_folder + "/gif"
        os.makedirs(gif_folder, exist_ok=True)
    if egl_mode:
        mp4_folder = os.path.join(eval_folder, "mp4")
        os.makedirs(mp4_folder, exist_ok=True)
        # C-side make_client writes <scenario_id>.mp4 into the process cwd. We
        # chdir into mp4_folder so every scenario's file lands in the right
        # place, then restore cwd after the rollout loop.
        saved_cwd = os.getcwd()
        os.chdir(mp4_folder)

    global_infos = {}
    num_scenarios = args["num_scenarios"]

    # Apply per-env video suffix once before any render. make_client reads
    # env->video_suffix on the first render to build the ffmpeg filename, so
    # this must fire before any step. We don't yet know how many internal
    # envs are in the vecenv (vecenv.get_state() only works after reset),
    # so set on a generous prefix and let extras be no-ops.
    if egl_mode and video_suffix:
        _target_env_pre = vecenv if not hasattr(vecenv, "envs") else vecenv.envs[0]
        # Drive exposes its internal C-level vec env count via num_envs.
        # Use it as the loop bound so we never call set_video_suffix on an
        # out-of-range env_id (which would corrupt memory before the C
        # bounds check landed).
        _internal_num_envs = getattr(_target_env_pre, "num_envs", 1)
        for _e in range(_internal_num_envs):
            try:
                _target_env_pre.set_video_suffix(video_suffix, env_idx=_e)
            except Exception:
                break

    scenarios_processed = 0
    # PufferEnv native backend: vecenv IS the Drive env (no .envs list).
    # Serial/Multiprocessing: need vecenv.envs[0] to reach the underlying env.
    target_env = vecenv if not hasattr(vecenv, "envs") else vecenv.envs[0]

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
            target_env.batch_size_eval = max(1, remaining_after_this)

            map_names = []
            for env_idx in range(num_envs_in_batch):
                map_names.append(scenarios[env_idx]["map_name"].split("/")[-1].split(".")[0])

            # Reset LSTM
            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                )

            # Initialize histories as lists of lists (one list per environment).
            # Only needed for the HTML replay path — EGL writes mp4 frames
            # directly to ffmpeg via c_render each step.
            if html_mode:
                agent_histories = [[] for _ in range(num_envs_in_batch)]
                traffic_histories = [[] for _ in range(num_envs_in_batch)]
                trajectory_histories = [[] for _ in range(num_envs_in_batch)]
                all_agents_obs_histories = [[] for _ in range(num_envs_in_batch)]

            _render_steps = args["env"]["scenario_length"]
            if render_max_steps is not None:
                _render_steps = min(_render_steps, render_max_steps)
            for t in range(_render_steps):
                if html_mode:
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

                if egl_mode:
                    # Flush one frame per env through c_render → client_record_frame
                    # → PBO async readback → writev → ffmpeg pipe. make_client is
                    # called lazily on the first render per env (sets up ffmpeg +
                    # GPU context) and close_client at scenario end flushes the
                    # trailing PBO frame.
                    for e in range(num_envs_in_batch):
                        target_env.render(env_idx=e, view_mode=view_mode)

                # Serial backend returns infos as single list (infos[0] is the env's info list)
                if infos and infos[0]:
                    for env_idx, summary in enumerate(infos[0]):
                        env_map_name = summary["map_name"].split("/")[-1].split(".")[0]
                        summary["episode_id"] = batch_start + env_idx
                        summary["env_id"] = env_idx
                        summary["map_name"] = env_map_name

                        for k, v in summary.items():
                            if k not in global_infos:
                                global_infos[k] = []
                            global_infos[k].append(v)

            if html_mode:
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
                        f"{gif_folder}/{env_map_name}_{global_episode_id:03d}.html",
                        head_north=True,
                    )

            if egl_mode:
                # Close every env's Client so ffmpeg gets EOF on its input pipe,
                # the trailing PBO frame is flushed, and libx264 writes the mp4
                # trailer. Without this, the mp4 files are either empty or one
                # frame short.
                import sys as _sys_cc

                _sys_cc.stderr.write(
                    f"[render-instr] starting close_client loop num_envs_in_batch={num_envs_in_batch}\n"
                )
                _sys_cc.stderr.flush()
                for e in range(num_envs_in_batch):
                    _sys_cc.stderr.write(f"[render-instr] close_client(env_idx={e}) calling\n")
                    _sys_cc.stderr.flush()
                    target_env.close_client(env_idx=e)
                    _sys_cc.stderr.write(f"[render-instr] close_client(env_idx={e}) returned\n")
                    _sys_cc.stderr.flush()

            scenarios_processed += num_envs_in_batch
            pbar.update(num_envs_in_batch)

    import sys as _sys_instr

    _sys_instr.stderr.write("[render-instr] rollout loop done\n")
    _sys_instr.stderr.flush()

    # render_key_prefix overrides metric_prefix for wandb media uploads only.
    # This lets callers keep metric_prefix for scalar metrics while using a
    # different namespace for renders (e.g. driving_behaviours/<class>/render/<view>).
    _upload_prefix = render_key_prefix if render_key_prefix is not None else metric_prefix

    if html_mode:
        pufferlib.viz.build_gallery_index(gif_folder)
        if logger is not None:
            try:
                import wandb

                html_paths = sorted(os.path.join(gif_folder, f) for f in os.listdir(gif_folder) if f.endswith(".html"))
                if html_paths:
                    step = args.get("global_step")
                    html_log = {
                        f"{_upload_prefix}/{os.path.splitext(os.path.basename(p))[0]}": wandb.Html(p)
                        for p in html_paths
                    }
                    if hasattr(logger, "log"):
                        logger.log(html_log, step) if step is not None else logger.log(html_log)
                    if not quiet:
                        print(f"Uploaded {len(html_paths)} render HTML(s) to wandb")
            except Exception as e:
                if not quiet:
                    print(f"Failed to upload render HTMLs to wandb: {e}")

    if saved_cwd is not None:
        os.chdir(saved_cwd)
    _sys_instr.stderr.write("[render-instr] chdir restored\n")
    _sys_instr.stderr.flush()

    avg_infos = _export_metrics(global_infos, eval_folder, num_scenarios, quiet, verify_coverage=False)
    _sys_instr.stderr.write("[render-instr] _export_metrics done\n")
    _sys_instr.stderr.flush()
    _log_eval_metrics(logger, avg_infos, args, metric_prefix, quiet)
    _sys_instr.stderr.write("[render-instr] _log_eval_metrics done\n")
    _sys_instr.stderr.flush()

    if egl_mode and mp4_folder and logger is not None:
        try:
            import wandb

            mp4_paths = sorted(os.path.join(mp4_folder, f) for f in os.listdir(mp4_folder) if f.endswith(".mp4"))
            if mp4_paths:
                video_log = {
                    f"{_upload_prefix}/{os.path.splitext(os.path.basename(p))[0]}": wandb.Video(p, fps=30, format="mp4")
                    for p in mp4_paths
                }
                step = args.get("global_step")
                if hasattr(logger, "log"):
                    logger.log(video_log, step) if step is not None else logger.log(video_log)
                if not quiet:
                    print(f"Uploaded {len(mp4_paths)} render mp4(s) to wandb")
        except Exception as e:
            if not quiet:
                print(f"Failed to upload render mp4s to wandb: {e}")

    # Close vectorized environment to avoid file descriptor leaks
    vecenv.close()


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

    device = args["train"]["device"]
    if isinstance(device, int):
        device = torch.device("cuda", device) if torch.cuda.is_available() else torch.device("cpu")
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
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args["load_model_path"]
    if load_path == "latest":
        load_path = max(glob.glob(f"experiments/{env_name}*.pt"), key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
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
        "--load-id", type=str, default=None, help="Kickstart/eval from from a finished Wandb/Neptune run"
    )
    parser.add_argument(
        "--render-mode", type=str, default="auto", choices=["auto", "human", "ansi", "rgb_array", "raylib", "None"]
    )
    parser.add_argument("--video-path", type=str, default="videos", help="Path to save videos")
    parser.add_argument("--num_scenarios", type=int, default=3, help="Number of scenarios to eval")
    parser.add_argument(
        "--num_carla_maps", type=int, default=8, help="Number of CARLA maps to use in gigaflow mode (max 8)"
    )
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

    # Load driving behaviours eval config if specified
    behaviours_config_path = args.get("eval", {}).get("driving_behaviours_eval_config")
    if behaviours_config_path:
        if isinstance(behaviours_config_path, str):
            behaviours_config_path = behaviours_config_path.strip('"').strip("'")
        if os.path.exists(behaviours_config_path):
            print(f"Loading driving behaviours eval config from {behaviours_config_path}")
            bp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
            bp.read(behaviours_config_path)
            behaviours = {}
            for section in bp.sections():
                behaviours[section] = {k: puffer_type(v) for k, v in bp[section].items()}
            args["driving_behaviours_eval"] = behaviours
        else:
            print(f"Warning: driving_behaviours_eval_config not found: {behaviours_config_path}")

    # Use World size to divide Num_Agents / minibatch size in DDP
    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        args["env"]["num_agents"] = args["env"]["num_agents"] // world_size
        args["train"]["minibatch_size"] = args["train"]["minibatch_size"] // world_size
        args["train"]["max_minibatch_size"] = args["train"]["max_minibatch_size"] // world_size
        args["train"]["total_timesteps"] = args["train"]["total_timesteps"] // world_size

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
        eval(env_name=env_name)
    elif mode == "eval_multi_scenarios":
        eval_multi_scenarios(env_name=env_name)
    elif mode == "eval_multi_scenarios_render":
        eval_multi_scenarios_render(env_name=env_name)
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
