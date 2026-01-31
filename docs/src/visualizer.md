# Visualizer

PufferDrive ships a Raylib-based visualizer for replaying scenes, exporting videos, and debugging policies.

## Dependencies
Install the minimal system packages for headless render/export:

```bash
sudo apt update
sudo apt install ffmpeg xvfb
```

On environments without sudo, install them into your conda/venv:

```bash
conda install -c conda-forge xorg-x11-server-xvfb-cos6-x86_64 ffmpeg
```

## Build
Compile the visualizer binary from the repo root:

```bash
bash scripts/build_ocean.sh visualize local
```

If you need to force a rebuild, remove the cached binary first (`rm ./visualize`).

## Rendering a Video
Launch the visualizer with a virtual display and export an `.mp4` for the binary scenario:

```bash
xvfb-run -s "-screen 0 1280x720x24" ./visualize
```

Adjust the screen size and color depth as needed. The `xvfb-run` wrapper allows Raylib to render without an attached display, which is convenient for servers and CI jobs.

## Arguments & Configuration

The `visualize` tool supports several CLI arguments to control the rendering output. It also reads the `pufferlib/config/ocean/drive.ini` file for default environment settings(For more details on these settings, refer to [Configuration](simulator.md#configuration)).

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--map-name <path>` | Path to the map binary file (e.g., `resources/drive/binaries/training/map_000.bin`). If omitted, picks a random map out of `num_maps` from `map_dir` in `drive.ini`. | Random |
| `--policy-name <path>` | Path to the policy weights file (`.bin`). | `resources/drive/puffer_drive_weights.bin` |
| `--view <mode>` | Selects which views to render: `agent`, `topdown`, or `both`. | `both` |
| `--output-agent <path>` | Output filename for agent view video. | `<policy>_agent.mp4` |
| `--output-topdown <path>` | Output filename for top-down view video. | `<policy>_topdown.mp4` |
| `--frame-skip <n>` | Renders every Nth frame to speed up generation (framerate remains 30fps). | `1` |
| `--num-maps <n>` | Overrides the number of maps to sample from if `--map-name` is not set. | `drive.ini` value |

### Visualization Flags

| Flag | Description |
| :--- | :--- |
| `--show-grid` | Draws the underlying nav-graph/grid on the map. |
| `--obs-only` | Hides objects not currently visible to the agent's sensors (fog of war). |
| `--lasers` | Visualizes the raycast sensor lines from the agent. |
| `--log-trajectories` | Draws the ground-truth "human" expert trajectories as green lines. |
| `--zoom-in` | Zooms the camera mainly on the active region rather than the full map bounds. |

### Key `drive.ini` Settings
The visualizer initializes the environment using `pufferlib/config/ocean/drive.ini`. Important settings include:

- `[env] dynamics_model`: `classic` or `jerk`. Must match the trained policy.
- `[env] episode_length`: Duration of the playback. defaults to 91 if set to 0.
- `[env] control_mode`: Determines which agents are active (`control_vehicles` vs `control_sdc_only`).
- `[env] goal_behavior`: Defines agent behavior upon reaching goals (respawn vs stop).
