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
goal_speed = 15.0  # Target speed in m/s at the goal. Lower values discourage excessive speeding
goal_behavior = 1  # 0: respawn, 1: generate_new_goals, 2: stop
goal_target_distance = 30.0  # Distance to new goal when using generate_new_goals

# Episode settings
episode_length = 200 # Increase for longer episode horizon
resample_frequency = 100000 # No resampling needed (there are only a few Carla maps)
termination_mode = 0  # 0: terminate at episode_length, 1: terminate after all agents reset

# Map settings
map_dir = "resources/drive/binaries/carla"
num_maps = 2
```

this should give a good starting point.

> [!Note]
> The default training hyperparameters work well for both configurations and typically don't need adjustment.


## Controlled experiments

Run parameter sweeps for architecture search or multi-seed experiments:
```bash
puffer controlled_exp puffer_drive --wandb --wandb-project "pufferdrive2.0_carla" --tag speed
```

Define parameter sweeps in `drive.ini`:
```ini
[controlled_exp.env.goal_speed]
values = [10, 20, 30]
```

This will launch separate training runs for each value in the list, useful for:
- Hyperparameter tuning
- Architecture search
- Running multiple random seeds
- Ablation studies

You can specify multiple controlled experiment parameters, and the system will iterate through all combinations.
