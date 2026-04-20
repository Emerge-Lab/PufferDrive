"""Shared rollout loop for Drive evaluation and rendering.

Single source of truth for the forward-sample-step-break cycle. Used by:
  - ``pufferl.render`` — offline batch rendering, one video per map
  - ``eval_multi_scenarios_render`` — inline training render path

Callers pass a ``RenderContext`` to turn on rendering; pass ``None`` for a
pure stats rollout.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import pufferlib.pytorch


@dataclass
class RenderContext:
    """Enables rendering inside :func:`rollout_loop`.

    Attributes:
        view_mode: ``RenderView`` enum value passed to ``driver.render``.
        env_id: which sub-env in the vecenv to record from (default 0).
        video_suffix: appended to the mp4 filename; applied once before the
            first render via ``set_video_suffix`` so multi-view rollouts don't
            collide on output paths.
    """

    view_mode: int
    env_id: int = 0
    video_suffix: str = ""


def rollout_loop(
    policy,
    env,
    device,
    use_rnn: bool,
    max_steps: Optional[int] = None,
    render_ctx: Optional[RenderContext] = None,
    per_env_logs: bool = False,
):
    """Run a single policy rollout in a Drive vecenv.

    Args:
        policy: the policy to run. Caller is responsible for calling ``.eval()``.
        env: a ``PufferEnv``-compatible vecenv wrapping one or more Drive sub-envs.
        device: torch device for observation / state tensors.
        use_rnn: whether to allocate and carry LSTM hidden state.
        max_steps: loop iteration cap. Defaults to ``env.driver_env.episode_length``.
        render_ctx: if set, render the specified env/view every step before
            sampling actions.
        per_env_logs: passed through to ``env.step`` for unaggregated per-env
            logs (only supported on PufferEnv native backend).

    Returns:
        The last ``info`` returned by ``env.step``.
    """
    driver = env.driver_env
    num_agents = env.observation_space.shape[0]

    if render_ctx is not None:
        driver.set_video_suffix(render_ctx.video_suffix, env_id=render_ctx.env_id)

    obs, _ = env.reset()

    state = {}
    if use_rnn:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    if max_steps is None:
        max_steps = driver.episode_length

    info = []
    for _ in range(max_steps):
        if render_ctx is not None:
            driver.render(
                view_mode=render_ctx.view_mode,
                env_id=render_ctx.env_id,
            )

        with torch.no_grad():
            ob_t = torch.as_tensor(obs).to(device)
            logits, _ = policy.forward_eval(ob_t, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(env.action_space.shape)

        if isinstance(logits, torch.distributions.Normal):
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        if per_env_logs:
            obs, _, _, truncs, info = env.step(action_np, per_env_logs=True)
        else:
            obs, _, _, truncs, info = env.step(action_np)

        if truncs.all():
            break

    return info
