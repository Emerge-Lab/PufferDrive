"""Minimal continuous-action PPO trainer for PufferDrive.

This intentionally keeps the full RL loop in one file so the relationship
between collection, GAE, PPO updates, and checkpoints is easy to inspect.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class TrainConfig:
    map_dir: str
    num_maps: int
    num_envs: int
    controlled_agents_per_env: int
    episode_length: int
    resample_frequency: int
    goal_behavior: int
    goal_target_distance: float
    total_timesteps: int
    rollout_steps: int
    update_epochs: int
    minibatch_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_coef: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    hidden_size: int
    seed: int
    device: str
    checkpoint_dir: str
    checkpoint_interval: int
    eval_steps: int
    resume: str | None


class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.critic = nn.Linear(hidden_size, 1)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    def distribution_and_value(self, observations):
        hidden = self.encoder(observations)
        mean = self.actor_mean(hidden)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), self.critic(hidden).squeeze(-1)

    @torch.no_grad()
    def sample_action(self, observations):
        distribution, value = self.distribution_and_value(observations)
        raw_action = distribution.sample()
        env_action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        return env_action, raw_action, log_prob, value

    @torch.no_grad()
    def deterministic_action(self, observations):
        distribution, value = self.distribution_and_value(observations)
        return torch.tanh(distribution.mean), value

    def evaluate_raw_action(self, observations, raw_actions):
        distribution, value = self.distribution_and_value(observations)
        log_prob = distribution.log_prob(raw_actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, steps, agents, observation_dim, action_dim, device):
        self.observations = torch.zeros((steps, agents, observation_dim), dtype=torch.float32, device=device)
        self.raw_actions = torch.zeros((steps, agents, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((steps, agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((steps, agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((steps, agents), dtype=torch.float32, device=device)
        self.values = torch.zeros((steps, agents), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((steps, agents), dtype=torch.float32, device=device)
        self.returns = torch.zeros((steps, agents), dtype=torch.float32, device=device)

    def compute_gae(self, last_value, gamma, gae_lambda):
        last_advantage = torch.zeros_like(last_value)
        for step in reversed(range(self.rewards.shape[0])):
            if step == self.rewards.shape[0] - 1:
                next_value = last_value
            else:
                next_value = self.values[step + 1]
            next_nonterminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_value * next_nonterminal - self.values[step]
            last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
            self.advantages[step] = last_advantage
        self.returns.copy_(self.advantages + self.values)

    def flatten(self):
        return {
            "observations": self.observations.flatten(0, 1),
            "raw_actions": self.raw_actions.flatten(0, 1),
            "log_probs": self.log_probs.flatten(),
            "values": self.values.flatten(),
            "advantages": self.advantages.flatten(),
            "returns": self.returns.flatten(),
        }


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_env(config):
    from pufferlib.ocean.drive.drive import Drive

    map_dir = Path(config.map_dir)
    if not map_dir.exists():
        raise FileNotFoundError(f"Map directory does not exist: {map_dir}")

    num_agents = config.num_envs * config.controlled_agents_per_env
    return Drive(
        map_dir=str(map_dir),
        num_maps=config.num_maps,
        num_agents=num_agents,
        control_mode="control_mixed_play",
        init_mode="create_all_valid",
        goal_behavior=config.goal_behavior,
        goal_target_distance=config.goal_target_distance,
        action_type="continuous",
        episode_length=config.episode_length,
        termination_mode=0,
        resample_frequency=config.resample_frequency,
        render_mode=1,
        max_controlled_agents=config.controlled_agents_per_env,
    )


def save_checkpoint(path, model, optimizer, config, global_step, update):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "global_step": global_step,
            "update": update,
        },
        path,
    )


def ppo_update(model, optimizer, buffer, config):
    batch = buffer.flatten()
    advantages = batch["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    batch_size = advantages.shape[0]
    indices = torch.arange(batch_size, device=advantages.device)

    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    for _ in range(config.update_epochs):
        permutation = indices[torch.randperm(batch_size, device=indices.device)]
        for start in range(0, batch_size, config.minibatch_size):
            mb = permutation[start : start + config.minibatch_size]
            new_log_prob, entropy, new_value = model.evaluate_raw_action(
                batch["observations"][mb],
                batch["raw_actions"][mb],
            )

            log_ratio = new_log_prob - batch["log_probs"][mb]
            ratio = log_ratio.exp()
            mb_advantages = advantages[mb]
            policy_loss = -torch.min(
                ratio * mb_advantages,
                torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * mb_advantages,
            ).mean()

            value_loss = 0.5 * (new_value - batch["returns"][mb]).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["policy_loss"].append(policy_loss.item())
                metrics["value_loss"].append(value_loss.item())
                metrics["entropy"].append(entropy_loss.item())
                metrics["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                metrics["clip_fraction"].append(((ratio - 1).abs() > config.clip_coef).float().mean().item())

    return {name: float(np.mean(values)) for name, values in metrics.items()}


@torch.no_grad()
def evaluate(model, env, observation, device, steps):
    rewards = []
    for _ in range(steps):
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        actions, _ = model.deterministic_action(obs_tensor)
        observation, reward, terminal, truncation, info = env.step(actions.cpu().numpy().astype(np.float32))
        rewards.append(float(np.mean(reward)))
    return observation, float(np.mean(rewards)), float(np.sum(rewards))


def train(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)

    print("Creating PufferDrive environment...")
    env = make_env(config)
    observation, _ = env.reset(seed=config.seed)
    num_agents, observation_dim = observation.shape
    action_dim = env.single_action_space.shape[0]

    print(f"device={device}")
    print(f"agents={num_agents}, envs={env.num_envs}, obs_dim={observation_dim}, action_dim={action_dim}")
    print(f"map_ids={env.map_ids}")
    print(f"scenario_ids={env.scenario_ids}")

    model = ActorCritic(observation_dim, action_dim, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    global_step = 0
    first_update = 1

    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        first_update = int(checkpoint.get("update", 0)) + 1
        print(f"Resumed from {config.resume} at step {global_step}")

    steps_per_update = config.rollout_steps * num_agents
    total_updates = max(1, (config.total_timesteps - global_step + steps_per_update - 1) // steps_per_update)
    recent_episode_returns = deque(maxlen=100)
    recent_episode_lengths = deque(maxlen=100)
    running_returns = np.zeros(num_agents, dtype=np.float32)
    running_lengths = np.zeros(num_agents, dtype=np.int32)
    started_at = time.time()

    try:
        for update_offset in range(total_updates):
            update = first_update + update_offset
            buffer = RolloutBuffer(
                config.rollout_steps,
                num_agents,
                observation_dim,
                action_dim,
                device,
            )

            for step in range(config.rollout_steps):
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
                env_action, raw_action, log_prob, value = model.sample_action(obs_tensor)
                next_observation, reward, terminal, truncation, info = env.step(
                    env_action.cpu().numpy().astype(np.float32)
                )
                done = np.logical_or(terminal, truncation)

                buffer.observations[step].copy_(obs_tensor)
                buffer.raw_actions[step].copy_(raw_action)
                buffer.log_probs[step].copy_(log_prob)
                buffer.values[step].copy_(value)
                buffer.rewards[step].copy_(torch.as_tensor(reward, dtype=torch.float32, device=device))
                buffer.dones[step].copy_(torch.as_tensor(done, dtype=torch.float32, device=device))

                running_returns += reward
                running_lengths += 1
                for agent_index in np.flatnonzero(done):
                    recent_episode_returns.append(float(running_returns[agent_index]))
                    recent_episode_lengths.append(int(running_lengths[agent_index]))
                    running_returns[agent_index] = 0
                    running_lengths[agent_index] = 0

                observation = next_observation
                global_step += num_agents

            with torch.no_grad():
                last_obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
                _, last_value = model.distribution_and_value(last_obs)
            buffer.compute_gae(last_value, config.gamma, config.gae_lambda)
            metrics = ppo_update(model, optimizer, buffer, config)

            elapsed = max(time.time() - started_at, 1e-6)
            sps = int(global_step / elapsed)
            mean_reward = float(buffer.rewards.mean().item())
            episode_return = float(np.mean(recent_episode_returns)) if recent_episode_returns else float("nan")
            episode_length = float(np.mean(recent_episode_lengths)) if recent_episode_lengths else float("nan")
            print(
                f"update={update:04d} step={global_step:09d} sps={sps:6d} "
                f"reward/step={mean_reward:+.4f} episode_return={episode_return:+.3f} "
                f"episode_len={episode_length:.1f} policy_loss={metrics['policy_loss']:+.4f} "
                f"value_loss={metrics['value_loss']:.4f} entropy={metrics['entropy']:.3f} "
                f"kl={metrics['approx_kl']:.5f} clipfrac={metrics['clip_fraction']:.3f}"
            )

            if update % config.checkpoint_interval == 0 or global_step >= config.total_timesteps:
                checkpoint_path = Path(config.checkpoint_dir) / f"ppo_step_{global_step}.pt"
                save_checkpoint(checkpoint_path, model, optimizer, config, global_step, update)
                print(f"saved checkpoint: {checkpoint_path}")

            if global_step >= config.total_timesteps:
                break

        observation, eval_mean_reward, eval_return = evaluate(
            model,
            env,
            observation,
            device,
            config.eval_steps,
        )
        print(f"deterministic_eval mean_reward={eval_mean_reward:+.4f} return={eval_return:+.3f}")

        final_path = Path(config.checkpoint_dir) / "ppo_final.pt"
        save_checkpoint(final_path, model, optimizer, config, global_step, update)
        print(f"saved final checkpoint: {final_path}")
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal PPO trainer based on parallel_data_collect.py")
    parser.add_argument("--map-dir", default="resources/drive/binaries")
    parser.add_argument("--num-maps", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--controlled-agents-per-env", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=91)
    parser.add_argument("--resample-frequency", type=int, default=910)
    parser.add_argument("--goal-behavior", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--goal-target-distance", type=float, default=30.0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints/minimal_ppo")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=91)
    parser.add_argument("--resume")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    print(json.dumps(asdict(config), indent=2))
    train(config)
