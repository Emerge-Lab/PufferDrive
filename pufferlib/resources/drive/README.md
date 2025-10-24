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

We record multiple performance metrics during training, aggregated over all *active agents* (alive and controlled). Since agents are respawned upon reaching their goal by default (see section above), and many agents in the Waymo Open Dataset (WOMD) are initialized near their goals, metrics are computed **only for each agent’s first attempt** at completing the scene to ensure an unbiased performance estimate.

- `score`: Goals reached cleanly (goal was achieved without collision or going off-road)
- `goal_rate`: Measures whether agent got within `goal_radius` of goal position before the end of the episode.
- `collision_rate`: Binary flag (0 or 1) if agent hit another vehicle.
- `offroad_rate`: Binary flag (0 or 1) if agent left road bounds.
