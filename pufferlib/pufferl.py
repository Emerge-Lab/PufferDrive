## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import contextlib
import warnings

warnings.filterwarnings("error", category=RuntimeWarning)

import os
import sys
import glob
import ast
import time
import math
import copy
import random
import shutil
import numbers
import subprocess
import argparse
import importlib
import configparser
from threading import Thread
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import psutil

import torch
import torch.distributed
from torch.distributed.elastic.multiprocessing.errors import record
import torch.utils.cpp_extension

import pufferlib
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

# Assume advantage kernel has been built if CUDA compiler is available
ADVANTAGE_CUDA = shutil.which("nvcc") is not None


class PuffeRL:
    def __init__(self, config, vecenv, policy, logger=None):
        # Backend perf optimization
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = config["torch_deterministic"]
        torch.backends.cudnn.benchmark = True

        # Reproducibility
        seed = config["seed"]
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

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
        elif config["optimizer"] == "muon":
            from heavyball import ForeachMuon

            warnings.filterwarnings(action="ignore", category=UserWarning, module=r"heavyball.*")
            import heavyball.utils

            heavyball.utils.compile_mode = config["compile_mode"] if config["compile"] else None
            optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_eps"],
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

        self._mask_id_map = {}
        self.masksembles_log_interval = int(config.get("masksembles_log_interval", 0))
        self._last_uncertainty_log_step = 0

        self.masksembles_episode_logging = bool(config.get("masksembles_episode_logging", False))
        self._ep_return_sum = torch.zeros(total_agents, device=device)
        self._ep_uncert_sum = torch.zeros(total_agents, device=device)
        self._ep_uncert_count = torch.zeros(total_agents, device=device)

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
                self.lstm_h[k] = torch.zeros(self.lstm_h[k].shape, device=device)
                self.lstm_c[k] = torch.zeros(self.lstm_c[k].shape, device=device)

        self.full_rows = 0
        while self.full_rows < self.segments:
            profile("env", epoch)
            o, r, d, t, info, env_id, mask = self.vecenv.recv()

            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)

            done_mask = d + t  # TODO: Handle truncations separately
            self.global_step += int(mask.sum())

            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)  # , non_blocking=True)
            r = torch.as_tensor(r).to(device)  # , non_blocking=True)
            d = torch.as_tensor(d).to(device)  # , non_blocking=True)
            t_torch = torch.as_tensor(t).to(device)

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config["use_rnn"]:
                    state["lstm_h"] = self.lstm_h[env_id.start]
                    state["lstm_c"] = self.lstm_c[env_id.start]

                module = getattr(self.policy, "module", self.policy)
                ms_enabled = getattr(module, "masksembles_enabled", False)
                if ms_enabled:
                    K = int(getattr(module, "masksembles_masks", 4))
                    l = self.ep_lengths[env_id.start].item()
                    if env_id.start not in self._mask_id_map or l == 0:
                        self._mask_id_map[env_id.start] = random.randint(0, K - 1)
                    state["mask_id"] = self._mask_id_map[env_id.start]

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                raw_r = r
                r = torch.clamp(r, -1, 1)

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
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                # Note: We are not yet handling masks in this version
                self.ep_lengths[env_id] += 1
                if l + 1 >= config["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(num_full, device=config["device"]).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full
                    if getattr(module, "masksembles_enabled", False):
                        for start in range(env_id.start, env_id.stop):
                            self._mask_id_map.pop(start, None)

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

            if (
                getattr(module, "masksembles_enabled", False)
                and self.masksembles_log_interval > 0
                and (self.global_step - self._last_uncertainty_log_step) >= self.masksembles_log_interval
            ):
                try:
                    passes = int(getattr(module, "masksembles_forward_passes", getattr(module, "masksembles_masks", 4)))
                    if config["use_rnn"] and "hidden" in state and hasattr(module, "value_uncertainty_from_hidden"):
                        _, std = module.value_uncertainty_from_hidden(state["hidden"], passes=passes)
                        val = float(std.mean().detach().cpu().item())
                        self.logger.log({"uncertainty/value_std_mean": val}, step=self.global_step)
                        self._last_uncertainty_log_step = self.global_step
                except Exception:
                    pass

            if getattr(module, "masksembles_enabled", False) and self.masksembles_episode_logging:
                try:
                    passes = int(getattr(module, "masksembles_forward_passes", getattr(module, "masksembles_masks", 4)))
                    if config["use_rnn"] and "hidden" in state and hasattr(module, "value_uncertainty_from_hidden"):
                        _, std = module.value_uncertainty_from_hidden(state["hidden"], passes=passes)
                        std = std.view(-1)
                        agent_ids = torch.arange(env_id.start, env_id.stop, device=device)
                        self._ep_uncert_sum[agent_ids] += std
                        self._ep_uncert_count[agent_ids] += 1
                        self._ep_return_sum[agent_ids] += raw_r.view(-1)
                        done_bool = torch.logical_or(d.bool(), t_torch.bool())
                        if torch.any(done_bool):
                            done_idx = torch.nonzero(done_bool).view(-1)
                            finished = agent_ids[done_idx]
                            if done_idx.numel() > 0 and hasattr(self.logger, "wandb") and self.logger.wandb:
                                import wandb
                                ret = self._ep_return_sum[finished].detach().cpu().numpy().tolist()
                                cnt = torch.clamp(self._ep_uncert_count[finished], min=1)
                                unc = (self._ep_uncert_sum[finished] / cnt).detach().cpu().numpy().tolist()
                                tbl = wandb.Table(columns=["episode_return", "uncertainty_mean"])
                                for r_v, u_v in zip(ret, unc):
                                    tbl.add_data(float(r_v), float(u_v))
                                self.logger.wandb.log(
                                    {
                                        "uncertainty/episode_table": tbl,
                                        "uncertainty/episode_scatter": wandb.plot.scatter(
                                            tbl,
                                            x="episode_return",
                                            y="uncertainty_mean",
                                            title="Episode Uncertainty vs Return",
                                        ),
                                    },
                                    step=self.global_step,
                                )
                            # Reset finished accumulators
                            self._ep_return_sum[finished] = 0
                            self._ep_uncert_sum[finished] = 0
                            self._ep_uncert_count[finished] = 0
                except Exception:
                    pass

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
        losses = defaultdict(float)
        config = self.config
        device = config["device"]

        b0 = config["prio_beta0"]
        a = config["prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile("train_misc", epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            advantages = torch.zeros(shape, device=device)
            advantages = compute_puff_advantage(
                self.values,
                self.rewards,
                self.terminals,
                self.ratio,
                advantages,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )

            profile("train_copy", epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_rewards = self.rewards[idx]
            mb_terminals = self.terminals[idx]
            mb_truncations = self.truncations[idx]
            mb_ratio = self.ratio[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile("train_forward", epoch)
            if not config["use_rnn"]:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            logits, newvalue = self.policy(mb_obs, state)
            actions, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)

            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config["clip_coef"]).float().mean()

            adv = advantages[idx]
            adv = compute_puff_advantage(
                mb_values,
                mb_rewards,
                mb_terminals,
                ratio,
                adv,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            # Losses
            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss
            self.amp_context.__enter__()  # TODO: AMP needs some debugging

            # This breaks vloss clipping?
            self.values[idx] = newvalue.detach().float()

            # Logging
            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item() / self.total_minibatches
            losses["value_loss"] += v_loss.item() / self.total_minibatches
            losses["entropy"] += entropy_loss.item() / self.total_minibatches
            losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
            losses["approx_kl"] += approx_kl.item() / self.total_minibatches
            losses["clipfrac"] += clipfrac.item() / self.total_minibatches
            losses["importance"] += ratio.mean().item() / self.total_minibatches

            # Learn on accumulated minibatches
            profile("learn", epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Reprioritize experience
        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()

        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
        losses["explained_variance"] = explained_var.item()

        profile.end()
        logs = None
        self.epoch += 1
        done_training = self.global_step >= config["total_timesteps"]
        if done_training or self.global_step == 0 or time.time() > self.last_log_time + 0.25:
            logs = self.mean_and_log()
            self.losses = losses
            self.print_dashboard()
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            self.save_checkpoint()
            self.msg = f"Checkpoint saved at update {self.epoch}"

        if self.render and self.epoch % self.render_interval == 0:
            run_id = self.logger.run_id
            model_dir = os.path.join(self.config["data_dir"], f"{self.config['env']}_{run_id}")

            if os.path.exists(model_dir):
                model_files = glob.glob(os.path.join(model_dir, "model_*.pt"))
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
                        )

                    except Exception as e:
                        print(f"Failed to export model weights: {e}")
                        return logs

                    # Now call the C rendering function
                    try:
                        # Create output directory for GIFs
                        gif_output_dir = os.path.join(model_dir, "gifs")
                        os.makedirs(gif_output_dir, exist_ok=True)

                        # Copy the binary weights to the expected location
                        expected_weights_path = "resources/drive/puffer_drive_weights.bin"
                        os.makedirs(os.path.dirname(expected_weights_path), exist_ok=True)
                        shutil.copy2(bin_path, expected_weights_path)

                        cmd = ["xvfb-run", "-s", "-screen 0 1280x720x24", "./drive"]

                        # Add render configurations
                        if config["show_grid"]:
                            cmd.append("--show-grid")
                        if config["obs_only"]:
                            cmd.append("--obs-only")
                        if config["show_lasers"]:
                            cmd.append("--lasers")
                        if config["show_human_logs"]:
                            cmd.append("--log-trajectories")

                        env_vars = os.environ.copy()
                        env_vars["ASAN_OPTIONS"] = "exitcode=0"
                        result = subprocess.run(
                            cmd,
                            cwd=os.getcwd(),
                            capture_output=True,
                            text=True,
                            timeout=120,
                            env=env_vars,
                        )

                        # Treat normal success (0) as success; keep 1 tolerant if toolchains differ
                        if result.returncode in (0, 1):
                            # Move both generated GIFs to the model directory
                            gifs = [
                                ("resources/drive/output_topdown.gif", f"epoch_{self.epoch:06d}_topdown.gif"),
                                ("resources/drive/output_agent.gif", f"epoch_{self.epoch:06d}_agent.gif"),
                            ]

                            found_any = False
                            for source_gif, target_filename in gifs:
                                if os.path.exists(source_gif):
                                    found_any = True
                                    target_gif = os.path.join(gif_output_dir, target_filename)
                                    shutil.move(source_gif, target_gif)

                                    # Log to wandb if available
                                    if hasattr(self.logger, "wandb") and self.logger.wandb:
                                        import wandb

                                        view_type = "world_state" if "topdown" in target_filename else "agent_view"
                                        self.logger.wandb.log(
                                            {f"render/{view_type}": wandb.Video(target_gif, format="gif")},
                                            step=self.global_step,
                                        )
                                else:
                                    print(f"GIF generation completed but {source_gif} not found")

                            if not found_any:
                                print("GIF generation completed but file not found")
                        else:
                            print(f"C rendering failed: {result.stderr}")

                    except subprocess.TimeoutExpired:
                        print("C rendering timed out")
                    except Exception as e:
                        print(f"Failed to generate GIF: {e}")

                    finally:
                        # Clean up bin weights file
                        if os.path.exists(expected_weights_path):
                            os.remove(expected_weights_path)

        return logs

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
            "epoch": int(dist_sum(self.epoch, device)),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{f"environment/{k}": v for k, v in self.stats.items()},
            **{f"losses/{k}": v for k, v in self.losses.items()},
            **{f"performance/{k}": v["elapsed"] for k, v in self.profile},
            # **{f'environment/{k}': dist_mean(v, device) for k, v in self.stats.items()},
            # **{f'losses/{k}': dist_mean(v, device) for k, v in self.losses.items()},
            # **{f'performance/{k}': dist_sum(v['elapsed'], device) for k, v in self.profile},
        }

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                self.logger.log(logs, agent_steps)
                return logs
            else:
                return None

        self.logger.log(logs, agent_steps)
        return logs

    def close(self):
        self.vecenv.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()
        run_id = self.logger.run_id
        path = os.path.join(self.config["data_dir"], f"{self.config['env']}_{run_id}.pt")
        shutil.copy(model_path, path)
        return path

    def save_checkpoint(self):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        run_id = self.logger.run_id
        path = os.path.join(self.config["data_dir"], f"{self.config['env']}_{run_id}")
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f"model_{self.config['env']}_{self.epoch:06d}.pt"
        model_path = os.path.join(path, model_name)
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
        return model_path

    def print_dashboard(self, clear=False, idx=[0], c1="[cyan]", c2="[white]", b1="[bright_cyan]", b2="[bright_white]"):
        config = self.config
        sps = dist_sum(self.sps, config["device"])
        agent_steps = dist_sum(self.global_step, config["device"])
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
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
        remaining = "A hair past a freckle"
        if sps != 0:
            remaining = duration((config["total_timesteps"] - agent_steps) / sps, b2, c2)

        s.add_column(f"{c1}Summary", justify="left", vertical="top", width=10)
        s.add_column(f"{c1}Value", justify="right", vertical="top", width=14)
        s.add_row(f"{c2}Env", f"{b2}{config['env']}")
        s.add_row(f"{c2}Params", abbreviate(self.model_size, b2, c2))
        s.add_row(f"{c2}Steps", abbreviate(agent_steps, b2, c2))
        s.add_row(f"{c2}SPS", abbreviate(sps, b2, c2))
        s.add_row(f"{c2}Epoch", f"{b2}{self.epoch}")
        s.add_row(f"{c2}Uptime", duration(self.uptime, b2, c2))
        s.add_row(f"{c2}Remaining", remaining)

        delta = profile.eval["buffer"] + profile.train["buffer"]
        p = Table(box=None, expand=True, show_header=False)
        p.add_column(f"{c1}Performance", justify="left", width=10)
        p.add_column(f"{c1}Time", justify="right", width=8)
        p.add_column(f"{c1}%", justify="right", width=4)
        p.add_row(*fmt_perf("Evaluate", b1, delta, profile.eval, b2, c2))
        p.add_row(*fmt_perf("  Forward", c2, delta, profile.eval_forward, b2, c2))
        p.add_row(*fmt_perf("  Env", c2, delta, profile.env, b2, c2))
        p.add_row(*fmt_perf("  Copy", c2, delta, profile.eval_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", c2, delta, profile.eval_misc, b2, c2))
        p.add_row(*fmt_perf("Train", b1, delta, profile.train, b2, c2))
        p.add_row(*fmt_perf("  Forward", c2, delta, profile.train_forward, b2, c2))
        p.add_row(*fmt_perf("  Learn", c2, delta, profile.learn, b2, c2))
        p.add_row(*fmt_perf("  Copy", c2, delta, profile.train_copy, b2, c2))
        p.add_row(*fmt_perf("  Misc", c2, delta, profile.train_misc, b2, c2))

        l = Table(
            box=None,
            expand=True,
        )
        l.add_column(f"{c1}Losses", justify="left", width=16)
        l.add_column(f"{c1}Value", justify="right", width=8)
        for metric, value in self.losses.items():
            l.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")

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
            try:  # Discard non-numeric values
                int(value)
            except:
                continue

            u = left if i % 2 == 0 else right
            u.add_row(f"{c2}{metric}", f"{b2}{value:.3f}")
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
        return str(num)
    elif num < 1e6:
        return f"{num / 1e3:.1f}K"
    elif num < 1e9:
        return f"{num / 1e6:.1f}M"
    elif num < 1e12:
        return f"{num / 1e9:.1f}B"
    else:
        return f"{num / 1e12:.2f}T"


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
        if epoch % self.frequency != 0:
            return

        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self.pop(tick)

        self.stack.append(name)
        self.profiles[name]["start"] = tick

    def pop(self, end):
        profile = self.profiles[self.stack.pop()]
        delta = end - profile["start"]
        profile["elapsed"] += delta
        profile["delta"] += delta

    def end(self):
        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

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

                self.gpu_util.append(torch.cuda.utilization())
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100 * (total - free) / total)
            else:
                self.gpu_util.append(0)
                self.gpu_mem.append(0)

            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def downsample(arr, m):
    if len(arr) < m:
        return arr

    if m == 0:
        return [arr[-1]]

    orig_arr = arr
    last = arr[-1]
    arr = arr[:-1]
    arr = np.array(arr)
    n = len(arr)
    n = (n // m) * m
    arr = arr[-n:]
    downsampled = arr.reshape(m, -1).mean(axis=1)
    return np.concatenate([downsampled, [last]])


class NoLogger:
    def __init__(self, args):
        self.run_id = str(int(100 * time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path):
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

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def close(self, model_path):
        self.neptune["model"].track_files(model_path)
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
        )
        self.wandb = wandb
        self.run_id = wandb.run.id

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def close(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        self.wandb.run.log_artifact(artifact)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f"{self.run_id}:latest")
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f"{data_dir}/{model_file}"


class ReportAggregator:
    """Tracks weighted statistics emitted by Drive logs across evaluation rollouts."""

    def __init__(self):
        self.sums = defaultdict(float)
        self.sums_sq = defaultdict(float)
        self.weights = defaultdict(float)
        self.total_weight = 0.0
        self.history = []

    @staticmethod
    def _to_float(value):
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, np.ndarray) and value.size == 1:
            return float(value.item())
        return None

    def update(self, metrics):
        weight = self._to_float(metrics.get("n", 1.0))
        if weight is None or weight <= 0:
            return

        numeric = {}
        for key, value in metrics.items():
            if key == "n":
                continue
            scalar = self._to_float(value)
            if scalar is None:
                continue
            self.sums[key] += scalar * weight
            self.sums_sq[key] += (scalar ** 2) * weight
            self.weights[key] += weight
            numeric[key] = scalar

        self.total_weight += weight
        numeric["n"] = weight
        numeric["episodes"] = self.total_weight
        self.history.append(numeric)

    def mean(self):
        means = {}
        for key, total in self.sums.items():
            weight = self.weights.get(key, 0.0)
            if weight > 0:
                means[key] = total / weight
        return means

    def std(self):
        stds = {}
        means = self.mean()
        for key, total_sq in self.sums_sq.items():
            weight = self.weights.get(key, 0.0)
            if weight > 0:
                mean_val = means.get(key, 0.0)
                variance = max(total_sq / weight - mean_val ** 2, 0.0)
                stds[key] = variance ** 0.5
        return stds

    def to_table(self, keys=None):
        if keys is None:
            keys = sorted({k for entry in self.history for k in entry.keys() if k not in {"n"}})
        rows = []
        for entry in self.history:
            row = [entry.get("episodes", 0.0)]
            for key in keys:
                row.append(entry.get(key))
            rows.append(row)
        return keys, rows


def save_policy_weights(policy, path):
    """Write the flattened policy parameters to a binary file for the renderer."""

    weights = []
    for _, param in policy.named_parameters():
        weights.append(param.data.detach().cpu().numpy().ravel())

    if not weights:
        raise RuntimeError("Policy has no parameters to export")

    flat = np.concatenate(weights)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat.tofile(path)
    return path


def log_wandb_report(
    wandb_logger,
    means,
    stds,
    avg_entropy,
    collisions_total,
    runtime,
    aggregator,
    collisions_cumulative,
    gif_assets,
    avg_uncertainty=None,
):
    import wandb

    run = wandb_logger.wandb.run
    prev_step = int(run.summary.get("report/step_index", 0) or 0)
    report_step = prev_step + 1

    # Define a unified step metric for scalar history and
    # a dedicated axis for the cumulative collisions series
    wandb_logger.wandb.define_metric("report/step")
    # Explicitly map known scalars to report/step to avoid glob conflicts
    meta_scalar_keys = [
        "report/episodes",
        "report/collisions_total",
        "report/runtime_seconds",
        "report/policy_entropy",
    ]
    for k in meta_scalar_keys:
        wandb_logger.wandb.define_metric(k, step_metric="report/step")

    # The metric names depend on what was observed; map per-key
    metric_keys = sorted(means.keys())
    for k in metric_keys:
        wandb_logger.wandb.define_metric(f"report/{k}_mean", step_metric="report/step")
        wandb_logger.wandb.define_metric(f"report/{k}_std", step_metric="report/step")

    # Cumulative collisions use episode count as the x-axis. Support a fresh key
    # to avoid legacy step mappings on resumed runs.
    wandb_logger.wandb.define_metric(
        "report/cumulative_collisions", step_metric="report/collision_episode"
    )
    wandb_logger.wandb.define_metric(
        "report/cumulative_collisions_by_episode", step_metric="report/collision_episode"
    )

    log_payload = {
        "report/step": report_step,
        "report/episodes": aggregator.total_weight,
        "report/collisions_total": collisions_total,
        "report/runtime_seconds": runtime,
    }

    log_payload.update({f"report/{k}_mean": v for k, v in means.items()})
    log_payload.update({f"report/{k}_std": v for k, v in stds.items()})

    if avg_entropy is not None:
        log_payload["report/policy_entropy"] = avg_entropy
    if avg_uncertainty is not None:
        log_payload["report/value_uncertainty_mean"] = avg_uncertainty

    # Build a compact metrics table to browse per-episode stats
    metric_keys = sorted(means.keys())
    metrics_table = wandb.Table(columns=["episodes"] + [f"report/{k}" for k in metric_keys])
    for entry in aggregator.history:
        row = [entry.get("episodes", 0.0)]
        for key in metric_keys:
            row.append(entry.get(key))
        metrics_table.add_data(*row)

    log_payload["report/metrics_table"] = metrics_table
    log_payload["report/tables/metrics"] = metrics_table

    # Summary table with metrics vertically (read top-to-bottom)
    summary_cols = ["metric", "mean", "std"]
    summary_table = wandb.Table(columns=summary_cols)
    for key in metric_keys:
        summary_table.add_data(key, means.get(key), stds.get(key))
    if avg_entropy is not None:
        summary_table.add_data("policy_entropy", avg_entropy, None)
    if avg_uncertainty is not None:
        summary_table.add_data("value_uncertainty", avg_uncertainty, None)
    log_payload["report/summary_table"] = summary_table
    log_payload["report/tables/summary"] = summary_table

    if collisions_cumulative:
        # Also push as scalar time-series for easy default line charts
        for episode_count, total in collisions_cumulative:
            wandb_logger.wandb.log(
                {
                    "report/cumulative_collisions": total,
                    "report/cumulative_collisions_by_episode": total,
                    "report/collision_episode": episode_count,
                }
            )

        collision_table = wandb.Table(columns=["episodes", "cumulative_collisions"])
        for episode_count, total in collisions_cumulative:
            collision_table.add_data(episode_count, total)

        log_payload["report/collision_table"] = collision_table
        log_payload["report/tables/collisions"] = collision_table
        log_payload["report/collision_chart"] = wandb.plot.line(
            table=collision_table,
            x="episodes",
            y="cumulative_collisions",
            title="Cumulative Collisions",
        )

    wandb_logger.wandb.log(log_payload)

    # Keep summary strictly JSON-serializable (numbers/strings/lists of strings)
    summary_updates = {
        "report/latest_step": report_step,
        "report/episodes": aggregator.total_weight,
        "report/collisions_total": collisions_total,
        "report/runtime_seconds": runtime,
    }

    summary_updates.update({f"report/{k}_mean": v for k, v in means.items()})
    summary_updates.update({f"report/{k}_std": v for k, v in stds.items()})

    if avg_entropy is not None:
        summary_updates["report/policy_entropy"] = avg_entropy
    if avg_uncertainty is not None:
        summary_updates["report/value_uncertainty_mean"] = avg_uncertainty

    # Log GIFs as per-view sliders under stable keys. Expect gif_assets to contain lists.
    if gif_assets:
        topdown_list = gif_assets.get("topdown", [])
        agent_list = gif_assets.get("agent", [])
        gif_log = {"report/step": report_step}
        if topdown_list:
            gif_log["report/gifs/topdown"] = topdown_list
        if agent_list:
            gif_log["report/gifs/agent"] = agent_list
        wandb_logger.wandb.log(gif_log)

        # Summary metadata for quick browsing in workspace
        summary_updates["report/gif_count_topdown"] = len(topdown_list)
        summary_updates["report/gif_count_agent"] = len(agent_list)
        summary_updates["report/gif_total"] = len(topdown_list) + len(agent_list)
        # Capture simple labels if present
        labels = gif_assets.get("labels", [])
        if labels:
            # Ensure labels is JSON-serializable (list of strings)
            summary_updates["report/gif_labels"] = [str(x) for x in labels]

    run.summary.update(summary_updates)
    run.summary["report/step_index"] = report_step
    wandb_logger.wandb.finish()


def train(env_name, args=None, vecenv=None, policy=None, logger=None):
    args = args or load_config(env_name)

    # Assume TorchRun DDP is used if LOCAL_RANK is set
    if "LOCAL_RANK" in os.environ:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print("World size", world_size)
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"rank: {local_rank}, MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
        torch.cuda.set_device(local_rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

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

    train_config = dict(**args["train"], env=env_name)
    pufferl = PuffeRL(train_config, vecenv, policy, logger)

    all_logs = []
    while pufferl.global_step < train_config["total_timesteps"]:
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        pufferl.evaluate()
        if train_config["device"] == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        logs = pufferl.train()

        if logs is not None:
            if pufferl.global_step > 0.20 * train_config["total_timesteps"]:
                all_logs.append(logs)

    # Final eval. You can reset the env here, but depending on
    # your env, this can skew data (i.e. you only collect the shortest
    # rollouts within a fixed number of epochs)
    i = 0
    stats = {}
    while i < 32 or not stats:
        stats = pufferl.evaluate()
        i += 1

    logs = pufferl.mean_and_log()
    if logs is not None:
        all_logs.append(logs)

    pufferl.print_dashboard()
    model_path = pufferl.close()
    pufferl.logger.close(model_path)
    return all_logs


def eval(env_name, args=None, vecenv=None, policy=None):
    args = args or load_config(env_name)
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


def report(env_name, args=None, vecenv=None, policy=None):
    start_time = time.time()
    args = args or load_config(env_name)

    report_cfg = dict(args.get("report", {}))
    target_episodes = int(report_cfg.get("num_episodes", 0))
    if target_episodes <= 0:
        raise pufferlib.APIUsageError("report.num_episodes must be greater than 0")

    # Vector configuration overrides for evaluation
    eval_vec_cfg = dict(args["vec"])
    backend = report_cfg.get("backend", eval_vec_cfg.get("backend", "Serial"))
    if backend != "PufferEnv":
        eval_vec_cfg["backend"] = backend
    else:
        eval_vec_cfg["backend"] = "Serial"

    eval_vec_cfg["num_envs"] = int(report_cfg.get("num_envs", eval_vec_cfg.get("num_envs", 1)))
    if "num_workers" in eval_vec_cfg:
        eval_vec_cfg["num_workers"] = int(report_cfg.get("num_workers", min(eval_vec_cfg["num_workers"], eval_vec_cfg["num_envs"])))
    eval_vec_cfg["batch_size"] = eval_vec_cfg["num_envs"]

    args_for_env = copy.deepcopy(args)
    args_for_env["vec"] = eval_vec_cfg

    external_vecenv = vecenv is not None
    vecenv = vecenv or load_env(env_name, args_for_env)

    wandb_logger = None
    if args["wandb"]:
        load_id = args.get("load_id")
        if load_id is None:
            raise pufferlib.APIUsageError("report mode with wandb enabled requires --load-id to resume the run")
        wandb_logger = WandbLogger(args, load_id, resume="allow")

    policy = policy or load_policy(args, vecenv, env_name, logger=wandb_logger)
    policy.eval()

    device = args["train"]["device"]
    use_rnn = args["train"].get("use_rnn", False)
    num_agents = vecenv.observation_space.shape[0]
    state = None
    if use_rnn:
        hidden_size = getattr(policy, "hidden_size", args["rnn"].get("hidden_size", 256))
        state = dict(
            lstm_h=torch.zeros(num_agents, hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, hidden_size, device=device),
        )

    entropy_enabled = bool(report_cfg.get("entropy_metric", True))
    entropy_sum = 0.0
    entropy_count = 0.0
    base_policy = getattr(policy, "module", policy)
    uncertainty_enabled = bool(report_cfg.get("uncertainty", True)) and bool(
        getattr(base_policy, "masksembles_enabled", False)
    )
    uncertainty_passes = int(
        report_cfg.get(
            "uncertainty_forward_passes",
            getattr(base_policy, "masksembles_forward_passes", getattr(base_policy, "masksembles_masks", 4)),
        )
    )
    uncertainty_sum = 0.0
    uncertainty_count = 0.0
    uncertainty_scatter = bool(report_cfg.get("uncertainty_episode_scatter", True))
    base_policy = getattr(policy, "module", policy)
    uncertainty_enabled = bool(report_cfg.get("uncertainty", True)) and bool(
        getattr(base_policy, "masksembles_enabled", False)
    )
    uncertainty_passes = int(
        report_cfg.get(
            "uncertainty_forward_passes",
            getattr(base_policy, "masksembles_forward_passes", getattr(base_policy, "masksembles_masks", 4)),
        )
    )
    uncertainty_sum = 0.0
    uncertainty_count = 0.0
    max_steps = int(report_cfg.get("max_steps", 0))
    steps = 0
    runtime = 0.0

    aggregator = ReportAggregator()
    collisions_cumulative = []
    collisions_total = 0.0
    # Episode-level tracking arrays
    ep_return_sum = np.zeros(num_agents, dtype=np.float32)
    ep_unc_sum = np.zeros(num_agents, dtype=np.float32)
    ep_unc_count = np.zeros(num_agents, dtype=np.int32)
    episode_points = []

    seed_value = report_cfg.get("seed", None)
    if seed_value in (None, -1):
        observations, _ = pufferlib.vector.reset(vecenv)
    else:
        observations, _ = pufferlib.vector.reset(vecenv, seed=int(seed_value))

    try:
        while aggregator.total_weight < target_episodes:
            obs_tensor = torch.as_tensor(observations, device=device)

            with torch.no_grad():
                logits, values = policy.forward_eval(obs_tensor, state)
                actions, _, entropy = pufferlib.pytorch.sample_logits(logits)
                if entropy_enabled and entropy is not None:
                    entropy_sum += entropy.sum().item()
                    entropy_count += float(entropy.numel())

                std_vec_np = None
                if uncertainty_enabled:
                    try:
                        mod = getattr(policy, "module", policy)
                        if use_rnn and state is not None and "hidden" in state and hasattr(
                            mod, "value_uncertainty_from_hidden"
                        ):
                            _, std = mod.value_uncertainty_from_hidden(state["hidden"], passes=uncertainty_passes)
                        elif not use_rnn and hasattr(mod, "encode_observations") and hasattr(
                            mod, "value_uncertainty_from_hidden"
                        ):
                            h = mod.encode_observations(obs_tensor)
                            _, std = mod.value_uncertainty_from_hidden(h, passes=uncertainty_passes)
                        else:
                            std = None

                        if std is not None:
                            uncertainty_sum += float(std.mean().detach().cpu().item())
                            uncertainty_count += 1.0
                            std_vec_np = std.detach().cpu().view(-1).numpy()
                    except Exception:
                        pass

                if uncertainty_enabled:
                    try:
                        mod = getattr(policy, "module", policy)
                        if use_rnn and state is not None and "hidden" in state and hasattr(
                            mod, "value_uncertainty_from_hidden"
                        ):
                            _, std = mod.value_uncertainty_from_hidden(state["hidden"], passes=uncertainty_passes)
                            uncertainty_sum += float(std.mean().detach().cpu().item())
                            uncertainty_count += 1.0
                        elif not use_rnn and hasattr(mod, "encode_observations") and hasattr(
                            mod, "value_uncertainty_from_hidden"
                        ):
                            h = mod.encode_observations(obs_tensor)
                            _, std = mod.value_uncertainty_from_hidden(h, passes=uncertainty_passes)
                            uncertainty_sum += float(std.mean().detach().cpu().item())
                            uncertainty_count += 1.0
                    except Exception:
                        pass

            actions_np = actions.detach().cpu().numpy().reshape(vecenv.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                actions_np = np.clip(actions_np, vecenv.action_space.low, vecenv.action_space.high)

            observations, rewards, terminals, truncations, infos = pufferlib.vector.step(vecenv, actions_np)
            steps += 1

            if uncertainty_enabled and std_vec_np is not None:
                ep_unc_sum += std_vec_np
                ep_unc_count += 1
            ep_return_sum += rewards.astype(np.float32)

            if use_rnn:
                done = np.logical_or(terminals, truncations).astype(np.float32)
                if done.any():
                    keep = torch.as_tensor(1.0 - done, device=device).unsqueeze(1)
                    state["lstm_h"] = state["lstm_h"] * keep
                    state["lstm_c"] = state["lstm_c"] * keep

            done_ep = np.logical_or(terminals, truncations)
            if np.any(done_ep):
                idxs = np.where(done_ep)[0]
                for i in idxs.tolist():
                    unc_mean = float(ep_unc_sum[i] / max(1, int(ep_unc_count[i])))
                    episode_points.append((float(ep_return_sum[i]), unc_mean))
                ep_return_sum[idxs] = 0.0
                ep_unc_sum[idxs] = 0.0
                ep_unc_count[idxs] = 0

            for log in infos or []:
                if not log:
                    continue
                aggregator.update(log)
                weight = ReportAggregator._to_float(log.get("n", 1.0)) or 1.0
                collisions = ReportAggregator._to_float(log.get("collision_rate"))
                # Expand aggregated logs into per-episode increments for chart readability
                episodes_n = int(round(weight)) if weight is not None else 0
                per_ep = float(collisions) if collisions is not None else 0.0
                for _ in range(max(episodes_n, 0)):
                    collisions_total += per_ep
                    episode_axis = len(collisions_cumulative) + 1
                    collisions_cumulative.append((episode_axis, collisions_total))

            if max_steps > 0 and steps >= max_steps:
                break

    finally:
        runtime = time.time() - start_time
        if not external_vecenv:
            vecenv.close()

    if aggregator.total_weight < target_episodes:
        print(
            f"[report] Collected {aggregator.total_weight:.0f} agent episodes (< target {target_episodes})."
            " Increase report.max_steps or episodes if you need more coverage."
        )

    means = aggregator.mean()
    stds = aggregator.std()
    avg_entropy = entropy_sum / entropy_count if entropy_count > 0 else None
    avg_uncertainty = uncertainty_sum / uncertainty_count if uncertainty_count > 0 else None

    interesting_keys = [
        "score",
        "episode_return",
        "collision_rate",
        "clean_collision_rate",
        "offroad_rate",
        "completion_rate",
        "dnf_rate",
        "lane_alignment_rate",
        "avg_displacement_error",
        "episode_length",
        "perf",
    ]

    summary_lines = [
        f"Report completed in {runtime:.1f}s",
        f"Agent episodes: {aggregator.total_weight:.0f}",
        f"Cumulative collisions: {collisions_total:.2f}",
    ]

    if avg_entropy is not None:
        summary_lines.append(f"Average policy entropy: {avg_entropy:.4f}")
    if avg_uncertainty is not None:
        summary_lines.append(f"Average value uncertainty: {avg_uncertainty:.4f}")

    for key in interesting_keys:
        if key in means:
            mean_val = means[key]
            std_val = stds.get(key)
            if std_val is not None:
                summary_lines.append(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
            else:
                summary_lines.append(f"{key}: {mean_val:.4f}")

    print("\n".join(summary_lines))

    if wandb_logger is not None:
        gif_assets = generate_report_gifs(env_name, args, policy, wandb_logger, report_cfg)
        log_wandb_report(
            wandb_logger,
            means,
            stds,
            avg_entropy,
            collisions_total,
            runtime,
            aggregator,
            collisions_cumulative,
            gif_assets,
            avg_uncertainty=avg_uncertainty,
        )
        if uncertainty_enabled and uncertainty_scatter and episode_points:
            import wandb
            tbl = wandb.Table(columns=["episode_return", "uncertainty_mean"])
            for r_v, u_v in episode_points:
                tbl.add_data(float(r_v), float(u_v))
            wandb_logger.wandb.log(
                {
                    "report/uncertainty/episode_table": tbl,
                    "report/uncertainty/episode_scatter": wandb.plot.scatter(
                        tbl,
                        x="episode_return",
                        y="uncertainty_mean",
                        title="Report: Episode Uncertainty vs Return",
                    ),
                }
            )

    return dict(
        mean=means,
        std=stds,
        entropy=avg_entropy,
        episodes=aggregator.total_weight,
        collisions_total=collisions_total,
        runtime=runtime,
    )


def generate_report_gifs(env_name, args, policy, wandb_logger, report_cfg):
    if wandb_logger is None:
        return {}

    ensure_drive_binary()

    import wandb

    train_cfg = args.get("train", {})
    env_cfg = args.get("env", {})

    gif_root = report_cfg.get("gif_output_dir") or Path(train_cfg.get("data_dir", "experiments")) / "report_gifs"
    gif_root = Path(gif_root)
    if not gif_root.is_absolute():
        gif_root = Path.cwd() / gif_root

    output_dir = gif_root / env_name / wandb_logger.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / f"{env_name}_{wandb_logger.run_id}.bin"
    save_policy_weights(policy, weights_path)

    expected_weights_path = Path("resources/drive/puffer_drive_weights.bin")
    expected_weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, expected_weights_path)

    available_maps = sorted(glob.glob("resources/drive/binaries/map_*.bin"))
    map_count = max(int(report_cfg.get("gif_map_count", 0)), 0)
    if map_count == 0 or not available_maps:
        expected_weights_path.unlink(missing_ok=True)
        return {}

    rng_seed = report_cfg.get("seed", None)
    rng = random.Random(rng_seed if rng_seed not in (None, -1) else time.time())
    # Randomize selection order always; cap to requested count
    shuffled = list(available_maps)
    rng.shuffle(shuffled)
    selected_maps = shuffled[: min(map_count, len(shuffled))]

    variants = max(int(report_cfg.get("gif_variants_per_map", 1)), 1)
    timeout = int(report_cfg.get("gif_timeout", 180))

    render_opts = {
        "show_grid": bool(train_cfg.get("show_grid", False)),
        "obs_only": bool(train_cfg.get("obs_only", True)),
        "show_lasers": bool(train_cfg.get("show_lasers", False)),
        "show_human_logs": bool(train_cfg.get("show_human_logs", True)),
    }

    control_all = bool(env_cfg.get("control_all_agents", False))
    try:
        n_policy = int(env_cfg.get("num_policy_controlled_agents", -1))
    except (TypeError, ValueError):
        n_policy = -1
    deterministic = bool(env_cfg.get("deterministic_agent_selection", False))

    # Build lists for a per-view slider; also capture human-readable labels
    gif_topdown = []
    gif_agent = []
    gif_labels = []

    try:
        for map_path in selected_maps:
            map_name = Path(map_path).stem
            for variant in range(variants):
                topdown_out = Path("resources/drive/output_topdown.gif")
                agent_out = Path("resources/drive/output_agent.gif")
                if topdown_out.exists():
                    topdown_out.unlink()
                if agent_out.exists():
                    agent_out.unlink()

                cmd = [
                    "xvfb-run",
                    "-a",
                    "-s",
                    "-screen 0 1280x720x24",
                    "./drive",
                ]

                if render_opts["show_grid"]:
                    cmd.append("--show-grid")
                if render_opts["obs_only"]:
                    cmd.append("--obs-only")
                if render_opts["show_lasers"]:
                    cmd.append("--lasers")
                if render_opts["show_human_logs"]:
                    cmd.append("--log-trajectories")
                if control_all:
                    cmd.append("--pure-self-play")
                if n_policy > 0:
                    cmd += ["--num-policy-controlled-agents", str(n_policy)]
                if deterministic:
                    cmd.append("--deterministic-selection")

                cmd.extend(["--map-name", map_path])

                env_vars = os.environ.copy()
                env_vars["ASAN_OPTIONS"] = "exitcode=0"

                try:
                    result = subprocess.run(
                        cmd,
                        cwd=os.getcwd(),
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        env=env_vars,
                    )
                except subprocess.TimeoutExpired:
                    print(f"[report] GIF generation timed out for {map_name} variant {variant}")
                    continue

                gifs_exist = topdown_out.exists() or agent_out.exists()
                if result.returncode not in (0, 1) and not gifs_exist:
                    print(
                        f"[report] Renderer failed for {map_name} variant {variant}: returncode={result.returncode}\n{result.stderr}"
                    )
                    continue

                dest_files = []
                if topdown_out.exists():
                    dest = output_dir / f"{map_name}_variant{variant}_topdown.gif"
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(topdown_out), dest)
                    dest_files.append((dest, "topdown"))

                if agent_out.exists():
                    dest = output_dir / f"{map_name}_variant{variant}_agent.gif"
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(agent_out), dest)
                    dest_files.append((dest, "agent"))

                for dest, view in dest_files:
                    label = f"{map_name}/variant{variant} - {view}"
                    # Use Image for consistent slider captions; GIFs animate in UI
                    img = wandb.Image(str(dest), caption=label)
                    if view == "topdown":
                        gif_topdown.append(img)
                    elif view == "agent":
                        gif_agent.append(img)
                    gif_labels.append(label)
    finally:
        expected_weights_path.unlink(missing_ok=True)

    return {"topdown": gif_topdown, "agent": gif_agent, "labels": gif_labels}


def sweep(args=None, env_name=None):
    args = args or load_config(env_name)
    if not args["wandb"] and not args["neptune"]:
        raise pufferlib.APIUsageError("Sweeps require either wandb or neptune")

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


def export(args=None, env_name=None, vecenv=None, policy=None, path=None):
    args = args or load_config(env_name)
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv)


    skip_prefixes = ("value_hidden.", "value_out.")
    has_mlp = False
    has_linear_value = False
    for name, _ in policy.named_parameters():
        if ".value_fn." in name:
            has_linear_value = True
        if (".value_hidden." in name) or (".value_out." in name):
            has_mlp = True

    hidden_size = getattr(policy, "hidden_size", None)
    if hidden_size is None and hasattr(policy, "policy"):
        hidden_size = getattr(policy.policy, "hidden_size", None)

    weights = []
    total = 0
    skipped = 0
    inserted_placeholder = False
    need_placeholder = has_mlp and not has_linear_value and hidden_size is not None

    for name, param in policy.named_parameters():
        if need_placeholder and not inserted_placeholder and name.startswith("lstm."):
            import numpy as _np
            placeholder = _np.zeros(hidden_size * 1 + 1, dtype=_np.float32)
            weights.append(placeholder)
            total += placeholder.size
            inserted_placeholder = True

        if (".value_hidden." in name) or (".value_out." in name):
            skipped += param.numel()
            continue

        arr = param.data.detach().cpu().numpy().ravel()
        weights.append(arr)
        total += arr.size

    if need_placeholder and not inserted_placeholder and hidden_size is not None:
        placeholder = np.zeros(hidden_size * 1 + 1, dtype=np.float32)
        weights.append(placeholder)

    weights = np.concatenate(weights) if weights else np.array([], dtype=np.float32)
    if path is None:
        path = f"{args['env_name']}_weights.bin"

    weights.tofile(path)
    try:
        import os
        print(f"Saved {len(weights)} weights to {path}; skipped {skipped} MLP-only params")
    except Exception:
        pass


def ensure_drive_binary():
    """Ensure the drive binary exists, build it once if necessary. This
    is required for rendering with raylib.
    """
    if not os.path.exists("./drive"):
        print("Drive binary not found, building...")
        try:
            result = subprocess.run(
                ["bash", "scripts/build_ocean.sh", "drive", "local"], capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                print("Successfully built drive binary")
            else:
                print(f"Build failed: {result.stderr}")
                raise RuntimeError("Failed to build drive binary for rendering")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Build timed out")
        except Exception as e:
            raise RuntimeError(f"Build error: {e}")
    else:
        print("Drive binary found, ready for rendering")


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


def load_policy(args, vecenv, env_name="", logger=None):
    package = args["package"]
    module_name = "pufferlib.ocean" if package == "ocean" else f"pufferlib.environments.{package}"
    env_module = importlib.import_module(module_name)

    device = args["train"]["device"]
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
            if logger is not None:
                path = logger.download()
            else:
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


def load_config(env_name):
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
    parser.add_argument("--local-rank", type=int, default=0, help="Used by torchrun for DDP")
    parser.add_argument("--tag", type=str, default=None, help="Tag for experiment")
    args = parser.parse_known_args()[0]

    # Load defaults and config
    puffer_dir = os.path.dirname(os.path.realpath(__file__))
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
    return args


def main():
    err = (
        "Usage: puffer [train, eval, report, sweep, autotune, profile, export] [env_name] "
        "[optional args]. --help for more info"
    )
    if len(sys.argv) < 3:
        raise pufferlib.APIUsageError(err)

    mode = sys.argv.pop(1)
    env_name = sys.argv.pop(1)
    if mode == "train":
        train(env_name=env_name)
    elif mode == "eval":
        eval(env_name=env_name)
    elif mode == "report":
        report(env_name=env_name)
    elif mode == "sweep":
        sweep(env_name=env_name)
    elif mode == "autotune":
        autotune(env_name=env_name)
    elif mode == "profile":
        profile(env_name=env_name)
    elif mode == "export":
        export(env_name=env_name)
    else:
        raise pufferlib.APIUsageError(err)


if __name__ == "__main__":
    main()
