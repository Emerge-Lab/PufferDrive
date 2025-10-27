# PufferDrive

This readme contains several important assumptions and definions about the `PufferDrive` environment.

## Assumptions for initializating agents

### Waymo Open Motion Dataset (WOMD)

By default, the environment only creates and controls **vehicles** that meet the following conditions:

- Their `valid` flag is `True` at initialization (as determined by `init_steps`).
- Their initial position is more than `MIN_DISTANCE_TO_GOAL` away from the goal.
- They are **not** marked as experts in the scenario file.
- The total number of agents has **not** yet reached `MAX_AGENTS`.

When `control_non_vehicles=True`, these same conditions apply, but the environment will also include **non-vehicle agents**, such as cyclists and pedestrians.

## Termination conditions (`done`)

Episodes are never truncated before reaching `episode_len`. The `use_goal_generation` argument controls agent behavior after reaching a goal early:

* **`use_goal_generation=False` (default):** Agents respawn at their initial position after reaching their goal (last valid log position).
* **`use_goal_generation=True`:** Agents receive new goals indefinitely after reaching each goal.

## Logged performance metrics

We record multiple performance metrics during training, aggregated over all *active agents* (alive and controlled). Key metrics include:

- `score`: Goals reached cleanly (goal was achieved without collision or going off-road)
- `collision_rate`: Binary flag (0 or 1) if agent hit another vehicle.
- `offroad_rate`: Binary flag (0 or 1) if agent left road bounds.
- `completion_rate`: Whether the agent reached its goal in this episode (even if it collided or went off-road).


### Metric aggregation

The `num_agents` parameter in `drive.ini` defines the total number of agents used to collect experience.
At runtime, **Puffer** uses `num_maps` to create enough environments to populate the buffer with `num_agents`, distributing them evenly across `num_envs`.

Because agents are respawned immediately after reaching their goal, they remain active throughout the episode.

At the end of each episode (i.e., when `timestep == TRAJECTORY_LENGTH`), metrics are logged once via:

```C
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
This value represents the fraction of agents that collided at least once during the episode. So, cases **A** and **B** below would yield identical off-road and collision rates:

![alt text](../../resources/drive/examples_a_b.png)

Since these metrics do not capture *multiple* events per agent, we additionally log the **average number of collision and off-road events per episode**. This is computed as:

$$
c^{avg}_{\text{collision}} = \frac{\text{total number of collision events across all agents and environments}}{N}
$$

where $N$ is the total number of controlled agents.
For example, an `avg_collisions_per_agent` value of 4 indicates that, on average, each agent collides four times per episode.

### Effect of respawning on metrics

By default, agents are reset to their initial position when they reach their goal before the episode ends. Upon respawn, `respawn_timestep` is updated from `-1` to the current step index.

This raises the question: **how does repeated respawning affect aggregated metrics?**

To begin, note that the environment is a bit different before and after respawn. After an agent respawns, all other agents are "removed" from the environment. As a result, collisions with other agents cannot occur post-respawn.

This effectively transforms the scenario into a single-agent environment, simplifying the task since the agent no longer needs to coordinate with others.

![alt text](../../resources/drive/pre_and_post_respawn.png)

#### `score`

Consider an episode of 91 steps where an agent is initialized relatively close to the goal position and reaches its goal three times:

1. **First attempt:** reaches the goal without collisions
2. **Second attempt:** reaches the goal without collisions
3. **Third attempt:** reaches the goal but goes off-road along the way

![alt text](../../resources/drive/realistic_collision_event_post_respawn.png)

The highlighted trajectory shows the first attempt. In this case, the recorded score is `0.0` — a single off-road event invalidates the score for the entire episode. This behavior is desired: the score metric is unforgiving.

#### `offroad_rate` and `collision_rate`

Same logic holds as above.
