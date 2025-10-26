## Assumptions for initializating agents

### Waymo Open Motion Dataset (WOMD)

The `init_mode` argument determines which agents are controlled by the policy.It has the following options: `default`, `tracks_to_predict`, and `all_agents`.


> `default`: The environment only creates and controls **vehicles** that meet the following conditions:
- Their `valid` flag is `True` at initialization (as determined by `init_steps`).
- Their initial position is more than `MIN_DISTANCE_TO_GOAL` away from the goal.
- They are **not** marked as experts in the scenario file.
- The total number of agents has **not** yet reached `MAX_AGENTS`.

When `control_non_vehicles=True`, these same conditions apply, but the environment will also include **non-vehicle agents**, such as cyclists and pedestrians.


> `tracks_to_predict`: The controlled agents are directly read from the `tracks_to_predict` metadata flag. Works only for WOMD.

> `all_agents`: Controls all possible agents in the dataset.
