"""Standalone evaluator for checkpoint benchmarking."""

import torch
import numpy as np
import pufferlib


class CheckpointEvaluator:
    """Runs policy rollouts in a puffer environment.

    The caller is responsible for constructing the config and environment
    with the desired control_mode, goal_behavior, etc. This class just
    handles the rollout loop.
    """

    def __init__(self, configs):
        self.configs = configs

    def rollout(self, env, policy=None, render_env_idx=None, view_mode=None, deterministic=False):
        """Run a single rollout and return the per-env info logs.

        Args:
            policy: The policy to evaluate.
            env: A puffer vecenv, already configured.
            render_env_idx: If not None, render this env index each step.
                The env's render_mode should be set to 1 (headless/ffmpeg)
                or 0 (window) accordingly.
            view_mode: RenderView enum value. Defaults to FULL_SIM_STATE.
        """
        from pufferlib.ocean.drive.drive import RenderView

        if view_mode is None:
            view_mode = RenderView.FULL_SIM_STATE

        driver = env.driver_env
        num_agents = env.observation_space.shape[0]
        device = self.configs["train"]["device"]

        obs, info = env.reset()
        terminals = np.zeros((num_agents, 1), dtype=bool)

        if policy:
            state = {}
            if self.configs["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                )

        info_list = []
        for time_idx in range(env.episode_length):
            if render_env_idx is not None:
                cur = driver.agent_offsets[render_env_idx]
                nxt = driver.agent_offsets[render_env_idx + 1]
                render_env_done = terminals[cur:nxt].all()
                if not render_env_done:
                    driver.render(view_mode=view_mode, env_idx=render_env_idx)

            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                if policy:
                    logits, value = policy.forward_eval(ob_tensor, state)

                    if deterministic:
                        # Greedy: pick highest-probability action per head
                        if isinstance(logits, torch.Tensor):
                            action = logits.argmax(dim=-1, keepdim=True).int()
                        elif isinstance(logits, torch.distributions.Normal):
                            action = logits.loc  # mean of the distribution
                        else:  # multi-discrete (list of logit tensors)
                            action = torch.stack([l.argmax(dim=-1) for l in logits], dim=-1).int()
                    else:
                        action, logprob, _ = pufferlib.pytorch.sample_logits(logits)

                    if isinstance(logits, torch.distributions.Normal):
                        action = np.clip(action, driver.action_space.low, driver.action_space.high)

                    action_np = action.cpu().numpy().reshape(env.action_space.shape)
                else:  # Take random actions
                    if driver._action_type_flag == 1:  # Continuous
                        action_np = np.random.uniform(low=-1.0, high=1.0, size=env.action_space.shape).astype(
                            env.action_space.dtype
                        )
                    else:  # Discrete
                        action_np = np.random.randint(
                            low=0,
                            high=env.action_space.shape[0],
                            size=env.action_space.shape,
                            dtype=env.action_space.dtype,
                        )

                # print(action_np[0])
                # print(action_np.shape)

            obs, rewards, terminals, truncated, info_list = env.step(action_np, per_env_logs=True)

            if truncated.all():
                break

        return info_list
