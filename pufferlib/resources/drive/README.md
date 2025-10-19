## Assumptions for initializating agents

### Waymo Open Motion Dataset (WOMD)

By default, the environment only creates and controls **vehicles** that meet the following conditions:

- Their `valid` flag is `True` at initialization.
- Their initial position is more than `MIN_DISTANCE_TO_GOAL` away from the goal.
- They are **not** marked as experts in the scenario file.
- The total number of agents has **not** yet reached `MAX_AGENTS`.

When `control_non_vehicles=True`, these same conditions apply, but the environment will also include **non-vehicle agents**, such as cyclists and pedestrians.
