## Drive with trained agents

You can take manual control of an agent in the simulator by holding **LEFT SHIFT** and using the keyboard controls. When you're in control, the action values displayed on screen will turn **yellow**.

### Local rendering

To launch an interactive renderer, first build:

```bash
bash scripts/build_ocean.sh drive local
```

then launch:

```bash
./drive
```

This will run `demo()` with an existing model checkpoint.

## Arguments & Configuration

The `drive` tool supports similar CLI arguments as the visualizer to control the environment and rendering. It also reads the `pufferlib/config/ocean/drive.ini` file for default environment settings.

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--map-name <path>` | Path to the map binary file (e.g., `resources/drive/binaries/training/map_000.bin`). If omitted, picks a random map out of `num_maps` from `map_dir` in `drive.ini`. | Random |
| `--policy-name <path>` | `Path to the policy weights file (.bin).` | `resources/drive/puffer_drive_weights.bin` |
| `--view <mode>` | Selects which views to render: `agent`, `topdown`, or `both`. | `both` |
| `--frame-skip <n>` | Renders every Nth frame to speed up simulation (framerate remains 30fps). | `1` |
| `--num-maps <n>` | Overrides the number of maps to sample from if `--map-name` is not set. | `drive.ini` value |

### Visualization Flags

| Flag | Description |
| :--- | :--- |
| `--show-grid` | Draws the underlying nav-graph/grid on the map. |
| `--obs-only` | Hides objects not currently visible to the agent's sensors (fog of war). |
| `--lasers` | Visualizes the raycast sensor lines from the agent. |
| `--log-trajectories` | Draws the ground-truth "human" expert trajectories as green lines. |
| `--zoom-in` | Zooms the camera mainly on the active region rather than the full map bounds. |

### Controls

**General:**

- **LEFT SHIFT + Arrow Keys/WASD** - Take manual control
- **SPACE** - First-person camera view
- **Mouse Drag** - Pan camera
- **Mouse Wheel** - Zoom

**Classic dynamics model**

- **SHIFT + UP/W** - Increase acceleration
- **SHIFT + DOWN/S** - Decrease acceleration (brake)
- **SHIFT + LEFT/A** - Steer left
- **SHIFT + RIGHT/D** - Steer right

Each key press increments or decrements the action level. For example, tapping W multiple times increases acceleration from neutral (index 3) → 5 → 6 (maximum acceleration). We assume **no friction**, so releasing all keys maintains constant speed and heading.

**Jerk dynamics model**

- **SHIFT + UP/W** - Accelerate (+4.0 m/s³ jerk)
- **SHIFT + DOWN/S** - Brake (-15.0 m/s³ jerk)
- **SHIFT + LEFT/A** - Turn left (+4.0 m/s³ lateral jerk)
- **SHIFT + RIGHT/D** - Turn right (-4.0 m/s³ lateral jerk)

Actions are applied directly when keys are pressed. Pressing W always applies +4.0 m/s³ longitudinal jerk, regardless of how long the key is held.
