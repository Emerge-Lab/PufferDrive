# Visualizer

PufferDrive uses [Raylib](https://www.raylib.com/) for rendering the environment. Rendering is driven from Python using the torch policy directly. No separate binary or weight export is required.

## Dependencies

For headless rendering, we need ffmpeg and xvfb.
```bash
sudo apt update && sudo apt install ffmpeg xvfb
```

## Render Modes

Configure `render_mode` in `pufferlib/config/ocean/drive.ini`:

```ini
; 0 = pop-up window (requires display)
; 1 = headless (pipes frames to ffmpeg, recommended for servers/training)
render_mode = 1
```

## Rendering once

```bash
puffer eval puffer_drive
```

This runs a short rollout, calls `env.render()` each step, and finalizes the video on `vecenv.close()`. Use `render_mode` to determine whether the video shows up as a pop-up window, or whether it is stored as an mp4.

## View modes

Control what is rendered via the `view_mode` argument to `env.render()`:

```python
class RenderView(IntEnum):
    FULL_SIM_STATE = 0  # Top-down, fully observable
    BEV_AGENT_OBS  = 1  # Top-down, selected agent's observations only
    AGENT_PERSP    = 2  # Third-person perspective following selected agent

env.render(view_mode=RenderView.FULL_SIM_STATE, draw_traces=True, env_id=0)
```

## Training-time evaluation

Rendering during training is controlled by the `[eval]` section of `drive.ini`. See that file for available options (`human_replay_eval`, `self_play_eval`, `eval_interval`, etc.).

## Sharp edges

- **Raylib is not thread-safe.** If you create two separate render envs, always call `env1.close()` before calling `env2.render()`.
- Headless mode derives window dimensions from map bounds automatically; no manual resolution configuration is needed.