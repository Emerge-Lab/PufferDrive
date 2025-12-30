## Training

### Basic training

Launch a training run with Weights & Biases logging:

```bash
puffer train puffer_drive --wandb --wandb-project "pufferdrive"
```

### Environment configurations

**Default configuration (Waymo maps)**

The default settings in `drive.ini` are optimized for:

- Training in thousands of Waymo maps
- Short episodes (91 steps)

**Carla maps configuration**

For training agents to drive indefinitely in larger Carla maps, we recommend modifying `drive.ini` as follows:

```ini
[env]
goal_speed = 10.0  # Target speed in m/s at the goal. Lower values discourage excessive speeding
goal_behavior = 1  # 0: respawn, 1: generate_new_goals, 2: stop
goal_target_distance = 30.0  # Distance to new goal when using generate_new_goals

# Episode settings
episode_length = 300 # Increase for longer episode horizon
resample_frequency = 100000 # No resampling needed (there are only a few Carla maps)
termination_mode = 0  # 0: terminate at episode_length, 1: terminate after all agents reset

# Map settings
map_dir = "resources/drive/binaries/carla"
num_maps = 2
```

this should give a good starting point. With these settings, you'll need about 2-3 billion steps to get an agent that reaches most of it's goals (> 95%) and has a combined collsion / off-road rate of 3 % per episode of 300 steps.

> [!Note]
> The default training hyperparameters work well for both configurations and typically don't need adjustment.

> [!Note]
> The checkpoint at `resources/drive/puffer_drive_weights_carla_town12.bin` is an agent trained on Carla town 01 and 02 with these settings. This is the one used in the interactive demo.

## Controlled experiments

Aside from `train` and `sweep`, we support a third mode for running controlled experiments over lists of values:

```bash
puffer controlled_exp puffer_drive --wandb --wandb-project "pufferdrive2.0_carla" --tag speed
```

Define parameter sweeps in `drive.ini`:

```ini
[controlled_exp.env.goal_speed]
values = [10, 20, 30]
```

This will launch separate training runs for each value in the list, which cab be useful for:

- Hyperparameter tuning
- Architecture search
- Running multiple random seeds
- Ablation studies
