# Simulator Guide

Deep dive into how the Drive environment is wired, what it expects as inputs, and how observations/actions/configs are shaped. The environment entrypoint is `pufferlib/ocean/drive/drive.py`, which wraps the C core in `pufferlib/ocean/drive/drive.h` via `binding.c`.

## Runtime inputs and lifecycle
- **Map binaries**: The environment scans `resources/drive/binaries` for `map_*.bin` files and requires at least one to load. Keep `num_maps` no larger than what is present on disk. During vectorized setup, `binding.shared` samples maps until it accumulates at least `num_agents` controllable entities, skipping maps with no valid agents (`set_active_agents` in `drive.h`).
- **Episode length**: Default `scenario_length = 91` to match the Waymo logs (trajectory data is 91 steps), but you can set `env.scenario_length` (CLI or `.ini`) to any positive value. Metrics are logged and `c_reset` is called when `timestep == scenario_length`.
- **Resampling maps**: Python-side `Drive.step` reinitializes the vectorized environments every `resample_frequency` steps (default `910`, ~10 episodes) with fresh map IDs and seeds.
- **Initialization controls**:
  - `init_steps` starts agents from a later timestep in the logged trajectory.
  - `init_mode` (`create_all_valid` vs `create_only_controlled`) decides which logged actors are instantiated at reset.
  - `control_mode` (`control_vehicles`, `control_agents`, `control_tracks_to_predict`, `control_sdc_only`) selects which instantiated actors are policy-controlled. Non-controlled actors can still appear as static or expert replay agents.
  - `goal_behavior` chooses what happens on goal reach (`0` respawn at start pose, `1` sample new lane-following goals via the lane topology graph, `2` stop in place). `goal_radius` sets the completion threshold in meters.

See [Data](data.md) for how to produce the `.bin` inputs, including the binary layout.

## Actions and dynamics
- **Action types** (`env.action_type`):
  - `discrete` (default): classic dynamics use a single `MultiDiscrete([7*13])` index decoded into acceleration (`ACCELERATION_VALUES`) and steering (`STEERING_VALUES`); jerk dynamics use `MultiDiscrete([4, 3])` over `JERK_LONG`/`JERK_LAT`.
  - `continuous`: a 2-D Box in `[-1, 1]`. Classic scales to the max accel/steer magnitudes used in the discrete table. Jerk scales asymmetrically: negative values reach up to `-15 m/s^3` braking, positives up to `4 m/s^3` acceleration, lateral jerk up to `±4 m/s^3`.
- **Dynamics models** (`env.dynamics_model`):
  - `classic`: bicycle model integrating accel/steer with `dt` (default `0.1`).
  - `jerk`: integrates longitudinal/lateral jerk into accel, then into velocity/pose with steering limited to `±0.55 rad`. Speeds are clipped to `[0, 20] m/s`.

## Observation space
Shape is `ego_features + 63 * 7 + 200 * 7` = `1848` for classic dynamics (`ego_features = 7`) or `1851` for jerk dynamics (`ego_features = 10`). Computed in `compute_observations` (`drive.h`):

- **Ego block** (classic):
  1. Goal position in ego frame (x, y) scaled by `0.005` (~200 m range to 1.0)
  2. Ego speed / `MAX_SPEED` (100 m/s)
  3. Width / `MAX_VEH_WIDTH` (15 m)
  4. Length / `MAX_VEH_LEN` (30 m)
  5. Collision flag (1 if the agent collided this step)
  6. Respawn flag (1 if the agent was respawned this episode)
- **Ego block additions (jerk dynamics model only)**:
  - Steering angle / π
  - Longitudinal acceleration normalized to `[-15, 4]`
  - Lateral acceleration normalized to `[-4, 4]`
  - Respawn flag (index 9)
- **Partner blocks**: Up to 63 other agents (active first, then static experts) within 50 m. Each uses 7 values: relative (x, y) in ego frame scaled by `0.02`, width/length normalized as above, relative heading encoded as `(cos Δθ, sin Δθ)`, and speed / `MAX_SPEED`. Zero-padded when fewer neighbors are present or when agents are in respawn.
- **Road blocks**: Up to 200 nearby road segments pulled from a precomputed grid (`vision_range = 21`). Each entry stores relative midpoint (x, y) scaled by `0.02`, segment length / `MAX_ROAD_SEGMENT_LENGTH` (100 m), width / `MAX_ROAD_SCALE` (100), `(cos, sin)` of the segment direction in ego frame, and a type ID (`ROAD_LANE`..`DRIVEWAY` stored as `0..6`). Remaining slots are zero-padded.

## Rewards, termination, and metrics
- **Per-step rewards** (`c_step`):
  - Collision with another actor: `reward_vehicle_collision` (default `-0.5`)
  - Off-road (road-edge intersection): `reward_offroad_collision` (default `-0.2`)
  - Goal reached: `reward_goal` (default `1.0`) or `reward_goal_post_respawn` after a respawn
  - Optional ADE shaping: `reward_ade * avg_displacement_error`, where ADE is accumulated in `compute_agent_metrics`
- **Termination**: No early truncation; episodes roll to `scenario_length` steps. If `goal_behavior` is respawn, `respawn_agent` resets the pose and marks `respawn_timestep` so the respawn flag shows up in observations.
- **Logged metrics** (`add_log` aggregates over all active agents across envs):
  - `score`: reached goal without collision/off-road
  - `collision_rate` / `offroad_rate`: fraction of agents with ≥1 event in the episode
  - `avg_collisions_per_agent` / `avg_offroad_per_agent`: counts per agent, capturing repeated events
  - `completion_rate`: reached goal (even if collided/off-road); `dnf_rate`: clean but never reached goal
  - `lane_alignment_rate`, `avg_displacement_error`, `num_goals_reached`, plus counts of active/static/expert agents

`collision_behavior`, `offroad_behavior`, `reward_vehicle_collision_post_respawn`, and `spawn_immunity_timer` are parsed from the INI but currently unused in the stepping logic.

## Configuration files (`.ini`)
`pufferlib/config/default.ini` supplies global defaults. Environment-specific overrides live in `pufferlib/config/ocean/drive.ini` and are loaded first when you run `puffer train puffer_drive`; CLI flags (e.g., `--env.num-maps 128`) override both.

Key sections in `pufferlib/config/ocean/drive.ini`:
- **[env]**: Simulator knobs: `num_agents` (policy slots, C core cap 64), `num_maps`, `scenario_length`, `resample_frequency`, `action_type`, `dynamics_model`, rewards, `goal_radius`, `goal_behavior`, `init_steps`, `init_mode`, `control_mode`; rendering toggles `render`, `render_interval`, `obs_only`, `show_grid`, `show_lasers`, `show_human_logs`, `render_map`.
- **[vec]**: Vectorization sizing (`num_envs`, `num_workers`, `batch_size`; backend defaults to multiprocessing).
- **[policy]/[rnn]**: Model widths for the Torch policy (`input_size`, `hidden_size`) and optional LSTM wrapper.
- **[train]**: PPO-style hyperparameters (timesteps, learning rate, clipping, batch/minibatch, BPTT horizon, optimizer choice) merged with any unspecified defaults from `pufferlib/config/default.ini`.
- **[eval]**: WOSAC/human-replay switches and sizing (`eval.wosac_*`, `eval.human_replay_*`) mapped directly to the `Drive` kwargs in evaluation subprocesses.

## Model overview
Defined in `pufferlib/ocean/torch.py:Drive`:
- Three MLP encoders (ego, partners, roads) with LayerNorm. Partner and road encodings are max-pooled across instances.
- Concatenated embedding → GELU → linear to `hidden_size`, then split into actor/value heads.
- Discrete actions are emitted as logits per dimension (`MultiDiscrete`), continuous actions as Gaussian parameters (`softplus` std). Value head is a single linear output.
- `Recurrent = pufferlib.models.LSTMWrapper` can wrap the policy using the `rnn` config entries; otherwise the policy is feed-forward.

## Drive source files (what lives where)
- `pufferlib/ocean/drive/drive.py`: Python Gymnasium-style wrapper that sets up buffers, validates map availability, seeds the C core via `binding.env_init`, and handles map resampling.
- `pufferlib/ocean/drive/drive.h`: Main C implementation of stepping, observations, rewards/metrics, grid map, lane graph, and collision checking.
- `pufferlib/ocean/drive/binding.c`: Python C-extension glue that exposes `Drive` to Python, handles shared buffer setup, logging, and reading the `.ini` config.
- `pufferlib/ocean/drive/visualize.c`: Raylib-based renderer used by the `visualize` binary and training video exports.
- `pufferlib/ocean/drive/drive.c`: Small C demo/perf harness and network parity test runner for the C policy head.
- `pufferlib/ocean/drive/drivenet.h`: Lightweight C inference network used by the visualizer/demo to mirror the Torch policy outputs.

## Drive README (C core notes)

### Agent initialization and control

#### `init_mode`

Determines which agents are **created** in the environment.

| Option | Description |
| --- | --- |
| `create_all_valid` | Create all entities valid at initialization (`traj_valid[init_steps] == 1`). |
| `create_only_controlled` | Create only those agents that are controlled by the policy. |

#### `control_mode`

Determines which created agents are **controlled** by the policy.

| Option | Description |
| --- | --- |
| `control_vehicles` (default) | Control only valid **vehicles** (not experts, beyond `MIN_DISTANCE_TO_GOAL`, under `MAX_AGENTS`). |
| `control_agents` | Control all valid **agent types** (vehicles, cyclists, pedestrians). |
| `control_tracks_to_predict` *(WOMD only)* | Control agents listed in the `tracks_to_predict` metadata. |

### Termination conditions (`done`)

Episodes are never truncated before reaching `episode_len`. The `goal_behavior` argument controls agent behavior after reaching a goal early:

- **`goal_behavior=0` (default):** Agents respawn at their initial position after reaching their goal (last valid log position).
- **`goal_behavior=1`:** Agents receive new goals indefinitely after reaching each goal.
- **`goal_behavior=2`:** Agents stop after reaching their goal.

### Logged performance metrics

We record multiple performance metrics during training, aggregated over all *active agents* (alive and controlled). Key metrics include:

- `score`: Goals reached cleanly (goal was achieved without collision or going off-road)
- `collision_rate`: Binary flag (0 or 1) if agent hit another vehicle.
- `offroad_rate`: Binary flag (0 or 1) if agent left road bounds.
- `completion_rate`: Whether the agent reached its goal in this episode (even if it collided or went off-road).

#### Metric aggregation

The `num_agents` parameter in `drive.ini` defines the total number of agents used to collect experience. At runtime, Puffer uses `num_maps` to create enough environments to populate the buffer with `num_agents`, distributing them evenly across `num_envs`.

Because agents are respawned immediately after reaching their goal, they remain active throughout the episode.

At the end of each episode (i.e., when `timestep == TRAJECTORY_LENGTH`), metrics are logged once via:

```c
if (env->timestep == TRAJECTORY_LENGTH) {
    add_log(env);
    c_reset(env);
    return;
}
```

Metrics are normalized and aggregated in `vec_log` (`pufferlib/ocean/env_binding.h`). They are averaged over all active agents across all environments. For example, the aggregated collision rate is computed as:

$$
r^{agg}_{\text{collision}} = \frac{\mathbb{I}[\text{collided in episode}]}{N}
$$

where $N$ is the number of controlled agents.

Since these metrics do not capture *multiple* events per agent, we additionally log the **average number of collision and off-road events per episode**. This is computed as:

$$
c^{avg}_{\text{collision}} = \frac{\text{total number of collision events across all agents and environments}}{N}
$$

where $N$ is the total number of controlled agents. For example, an `avg_collisions_per_agent` value of 4 indicates that, on average, each agent collides four times per episode.

![Collision/off-road aggregation examples](images/examples_a_b.png)

#### Effect of respawning on metrics

By default, agents are reset to their initial position when they reach their goal before the episode ends. Upon respawn, `respawn_timestep` is updated from `-1` to the current step index. After an agent respawns, all other agents are removed from the environment, so collisions with other agents cannot occur post-respawn.

![Pre- and post-respawn environment](images/pre_and_post_respawn.png)

![Example respawn collision case](images/realistic_collision_event_post_respawn.png)
