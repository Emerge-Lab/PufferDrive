# PufferDrive simulator guide

A high-performance autonomous driving simulator in C with Python bindings.

**Entry point:** `pufferlib/ocean/drive/drive.py` wraps `pufferlib/ocean/drive/drive.h`

## Configuration

### Basic settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_maps` | - | Map binaries to load |
| `num_agents` | 32 | Policy-controlled agents (max 64) |
| `episode_length` | 91 | Steps per episode |
| `resample_frequency` | 910 | Steps between map resampling |

> [!TIP]
> Set `episode_length = 91` to match Waymo log length for single-goal tasks. Use longer episodes (e.g., 200+) with `goal_behavior=1` for multi-goal driving.

### Control modes

- `control_vehicles`: Only vehicles
- `control_agents`: All agent types (vehicles, cyclists, pedestrians)
- `control_tracks_to_predict`: WOMD evaluation mode
- `control_sdc_only`: Self-driving car only

> [!NOTE]
> `control_vehicles` filters out agents marked as "expert" and those too close to their goal (<2m). For full WOMD evaluation, use `control_tracks_to_predict`.

### Goal behaviors

Three modes determine what happens when an agent reaches its goal:

**Mode 0 (Respawn) - Default:**
- Agent teleports back to starting position
- Other agents removed from environment (prevents post-respawn collisions)
- Useful for maximizing environment interaction per episode

**Mode 1 (Generate new) - Multi-goal:**
- Agent receives a new goal sampled from the road network
- Can complete multiple goals per episode
- Tests long-horizon driving competence

**Mode 2 (Stop):**
- Agent stops in place after reaching goal
- Episode continues until `episode_length`
- Simplest setting for evaluation

> [!IMPORTANT]
> Goal behavior fundamentally changes what "success" means:
> - **Mode 0/2 (single goal):** Success = reaching the one goal without collision/off-road
> - **Mode 1 (multi-goal):** Success = completing ≥X% of sampled goals cleanly

**Config files:** `pufferlib/config/ocean/drive.ini` (loaded first), then `pufferlib/config/default.ini`

## Episode flow

1. **Initialize**: Load maps, select agents, set start positions
2. **Step loop** (until `episode_length`):
   - Move expert replay agents (if they exist)
   - Apply policy actions to controlled agents
   - Update simulator
   - Check collisions
   - Assign rewards
   - Handle goal completion/respawns
   - Compute observations
3. **End**: Log metrics, reset

> [!NOTE]
> Maps are resampled every `resample_frequency` steps (~10 episodes with default settings) to increase map diversity.

> [!CAUTION]
> No early termination - episodes always run to `episode_length` regardless of goal completion or collisions with the default settings.

## Actions

### Discrete actions
- **Classic**: 91 options (7 accel × 13 steer)
  - Accel: `[-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0]` m/s²
  - Steer: 13 values from -1.0 to 1.0
- **Jerk**: 12 options (4 long × 3 lat)
  - Long jerk: `[-15, -4, 0, 4]` m/s³
  - Lat jerk: `[-4, 0, 4]` m/s³

> [!NOTE]
> Discrete actions are decoded as: `action_idx → (accel_idx, steer_idx)` using division and modulo.

### Continuous actions
- 2D Box `[-1, 1]`
- **Classic**: Scaled to ±4 m/s² accel, ±1 steer
- **Jerk**: Asymmetric long (brake -15, accel +4), symmetric lat (±4)

### Dynamics models

**Classic (bicycle model):**
- Integrates accel/steer with dt=0.1s
- Wheelbase = 60% of vehicle length
- Standard kinematic bicycle model

**Jerk (physics-based):**
- Integrates jerk → accel → velocity → pose
- Steering limited to ±0.55 rad
- Speed clipped to [0, 20] m/s
- More realistic comfort and control constraints

> [!IMPORTANT]
> Jerk dynamics adds 3 extra observation features (steering angle, long accel, lat accel) compared to classic.

## Observations

### Size
- **Classic**: 1848 floats = 7 (ego) + 217 (partners) + 1624 (roads)
- **Jerk**: 1851 floats = 10 (ego) + 217 (partners) + 1624 (roads)

Where partners = `MAX_AGENTS - 1` agents × 7 features, roads = 232 segments × 7 features

> [!IMPORTANT]
> All observations are in the **ego vehicle's reference frame** (agent-centric) and are normalized. Positions rotate with the agent's heading.

### Ego features (ego frame)

**Classic (7):** goal_x, goal_y, speed, width, length, collision_flag, respawn_flag

**Jerk adds (3):** steering_angle, long_accel, lat_accel


### Partner features (up to `MAX_AGENTS - 1` agents, 7 each)
rel_x, rel_y, width, length, heading_cos, heading_sin, speed

- Within 50m of ego
- Active agents first, then static experts
- Zero-padded if fewer agents

> [!TIP]
> Partner heading is encoded as `(cos, sin)` of relative angle to avoid discontinuities at ±π.

### Road features (up to 232 segments, 7 each)
mid_x, mid_y, length, width, dir_cos, dir_sin, type

- Retrieved from 21×21 grid (5m cells, ~105m × 105m area)
- Types: ROAD_LANE=0, ROAD_LINE=1, ROAD_EDGE=2
- Pre-cached for efficiency

> [!NOTE]
> Road observations use a spatial grid with 5m cells. The 21×21 vision range gives ~105m visibility in all directions.


## Rewards & metrics

### Per-step rewards
- Vehicle collision: -1.0
- Off-road: -1.0
- Goal reached: +1.0 (or +0.25 after respawn in mode 0)
- Jerk penalty (classic only): -0.0002 × Δv/dt

> [!TIP]
> Goal completion requires both distance < `goal_radius` (default 2m) AND speed ≤ `goal_speed`.

### Episode metrics

**Core metrics**

- **`score`** - Aggregate success metric (threshold-based):
  - **Single-goal setting (modes 0, 2):** Binary 1.0 if goal reached cleanly
    - **Mode 0 (respawn):** No collision/off-road before first goal (post-respawn collisions ignored)
    - **Mode 2 (stop):** No collision/off-road throughout entire episode
  - **Multi-goal setting (mode 1):** Fractional based on completion rate with no collisions throughout episode:
    - 1 goal: ≥99% required
    - 2 goals: ≥50% required
    - 3-4 goals: ≥80% required
    - 5+ goals: ≥90% required

- **`collision_rate`** - Fraction of agents with ≥1 vehicle collision this episode

- **`offroad_rate`** - Fraction of agents with ≥1 off-road event this episode

- **`completion_rate`** - Fraction of goals reached this episode

- **`lane_alignment_rate`** - Fraction of time agents spent aligned with lane headings

**In-depth metrics**

- **`avg_collisions_per_agent`** - Mean collision count per agent (captures repeated collisions)

- **`avg_offroad_per_agent`** - Mean off-road count per agent (captures repeated off-road events)

> [!NOTE]
> The "rate" metrics are binary flags (did it happen?), while "avg_per_agent" metrics count total occurrences. An agent can have `collision_rate=1` but `avg_collisions_per_agent=3` if they collided three times.

- **`goals_reached_this_episode`** - Total goals completed across all agents

- **`goals_sampled_this_episode`** - Total goals assigned (>1 in multi-goal mode)

#### Metrics interpretation by goal behavior

| Metric | Respawn (0) | Multi-Goal (1) | Stop (2) |
|--------|-------------|----------------|----------|
| `score` | Reached goal before any collision/off-road? | Reached X% of goals with no collisions? | Reached goal with no collisions? |
| `completion_rate` | Reached the goal? | Fraction of sampled goals reached | Reached the goal? |
| `goals_reached` | Always ≤1 | Can be >1 | Always ≤1 |
| `collision_rate` | Any collision before first goal? | Any collision in episode? | Any collision in episode? |

> [!WARNING]
> **Respawn mode (0) scoring:** Score only considers collisions/off-road events that occurred before reaching the first goal. Post-respawn collisions do not disqualify the agent from receiving a score of 1.0.

> [!WARNING]
> **Respawn mode (0) side effect:** After respawn, all other agents are removed from the environment. This means vehicle collisions become impossible post-respawn, but off-road collisions can still occur.

## Source files

### C core
- `drive.h`: Main simulator (stepping, observations, collisions)
- `drive.c`: Demo and testing
- `binding.c`: Python interface
- `visualize.c`: Raylib renderer
- `drivenet.h`: C inference network

### Python
- `drive.py`: Gymnasium wrapper
- `torch.py`: Neural network (ego/partner/road encoders → actor/critic)

## Neural network

Three MLP encoders (ego, partners, roads) → concatenate → actor/critic heads

- Partner and road outputs are max-pooled (permutation invariant)
- Discrete actions: logits per dimension
- Continuous actions: Gaussian (mean + std)
- Optional LSTM wrapper for recurrence

> [!TIP]
> The architecture is modular - you can easily swap out encoders or add new observation types without changing the policy head.

## Constants reference

> [!WARNING]
> These constants are hardcoded in the C implementation. Changing them requires recompiling.

### Limits
- `MAX_AGENTS = 32` (compile-time, can be overridden with `-DMAX_AGENTS=64`)
- `MAX_ROAD_OBSERVATIONS = 232`
- `TRAJECTORY_LENGTH = 91`
- `MIN_DISTANCE_TO_GOAL = 2.0` m (agents closer than this won't be controlled)

### Spatial
- `GRID_CELL_SIZE = 5.0` m
- `VISION_RANGE = 21` cells (~105m × 105m)
- Partner observation range: 50m

### Physics
- `DEFAULT_DT = 0.1` s
- Jerk long clip: `[-15, 4]` m/s³
- Jerk lat clip: `[-4, 4]` m/s³
- Steering limit: `[-0.55, 0.55]` rad (~31.5°)
- Speed clip (jerk): `[0, 20]` m/s

### Normalization
- `MAX_SPEED = 100` m/s
- `MAX_VEH_LEN = 30` m
- `MAX_VEH_WIDTH = 15` m
- `MAX_ROAD_SEGMENT_LENGTH = 100` m

> [!NOTE]
> Normalization scales are chosen to map reasonable driving scenarios to ~[-1, 1] range for neural network stability.

---

**Version:** PufferDrive v2.0
