"""Shared rollout loop for Drive evaluation and rendering.

Single source of truth for the forward-sample-step-break cycle. Used by:
  - ``Evaluator._run_rollout`` — periodic training eval, optional env-selective render
  - ``pufferl.render`` — offline batch rendering, one video per map
  - ``SafeEvaluator.render`` — safe-eval-time rendering

Extracting this eliminates three hand-maintained copies of the same loop that
had drifted in subtle ways:
  * ``render_one_map`` was missing continuous-action clipping
  * ``render_one_map`` used ``done.all() or truncated.all()`` while the others used ``truncs.all()``
  * ``render_one_map`` inferred video filenames via a ``glob`` diff instead of calling ``set_video_suffix``
  * ``SafeEvaluator`` was using ``hasattr(policy, "hidden_size")`` as a proxy
    for "is an RNN policy", which is fragile (many non-RNN policies also
    expose ``hidden_size``). Callers now pass the authoritative ``use_rnn``
    flag from the training config.

Callers pass a ``RenderContext`` to turn on rendering; pass ``None`` for a pure
stats rollout.
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
        draw_traces: whether the C renderer should overlay trajectory traces.
        video_suffix: appended to the mp4 filename; applied once before the
            first render via ``set_video_suffix`` so multi-view rollouts don't
            collide on output paths.
    """

    view_mode: int
    env_id: int = 0
    draw_traces: bool = True
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
            sampling actions. Filename suffix is applied via ``set_video_suffix``.
        per_env_logs: passed through to ``env.step`` so callers can get
            unaggregated per-env logs instead of the default ``vec_log`` aggregate.

    Returns:
        The last ``info`` returned by ``env.step``. For stats rollouts this is
        the aggregated ``vec_log`` dict or per-env logs list; for render
        rollouts callers typically ignore it.
    """
    driver = env.driver_env
    num_agents = env.observation_space.shape[0]

    # Set the video filename suffix before the first render call so the C
    # binding names the mp4 correctly up front (e.g. {scenario_id}_bev.mp4).
    # Without this, callers have to glob-diff cwd and rename post hoc.
    if render_ctx is not None and render_ctx.video_suffix:
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
        # Render BEFORE the step so each frame shows the state the policy was
        # conditioned on.
        if render_ctx is not None:
            driver.render(
                view_mode=render_ctx.view_mode,
                draw_traces=render_ctx.draw_traces,
                env_id=render_ctx.env_id,
            )

        with torch.no_grad():
            ob_t = torch.as_tensor(obs).to(device)
            logits, _ = policy.forward_eval(ob_t, state)
            action, _, _ = pufferlib.pytorch.sample_logits(logits)
            action_np = action.cpu().numpy().reshape(env.action_space.shape)

        # Clip continuous actions to the valid range. This was previously
        # missing in render_one_map, so continuous policies could emit OOB
        # actions during offline rendering.
        if isinstance(logits, torch.distributions.Normal):
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        obs, _, _, truncs, info = env.step(action_np, per_env_logs=per_env_logs)

        # truncs.all() is set in the single c_step where the env auto-resets
        # (time limit or early termination). Breaking here exits the loop
        # exactly when the current episode wraps.
        if truncs.all():
            break

    return info
