## Logged performance metrics

### Environment

We log several performance metrics during training:

- `score`: Goals reached cleanly (no collision before goal)
- `perf`: Same as score
- `collision_rate`: Binary flag (0 or 1) if agent hit another vehicle. Averaged across all controlled agents.
- `offroad_rate`: Binary flag (0 or 1) if agent left road bounds. Averaged across all controlled agents.

### Questions
- Which agents are included in computing the metrics above?
    - Only the first attempt at solving the scene. Respawned agents are excluded from logs.

## Assumptions for initializating agents

### Waymo Open Motion Dataset (WOMD)

By default, the environment only creates and controls **vehicles** that meet the following conditions:

- Their `valid` flag is `True` at initialization (as determined by `init_steps`).
- Their initial position is more than `MIN_DISTANCE_TO_GOAL` away from the goal.
- They are **not** marked as experts in the scenario file.
- The total number of agents has **not** yet reached `MAX_AGENTS`.

When `control_non_vehicles=True`, these same conditions apply, but the environment will also include **non-vehicle agents**, such as cyclists and pedestrians.
