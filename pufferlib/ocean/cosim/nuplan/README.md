# PufferDrive policy under nuPlan's closed-loop evaluation

## Goal

Evaluate a PufferDrive-trained policy with nuPlan's **unmodified** `run_simulation.py` closed-loop pipeline — the same simulator, metrics, and scoring CaRL's own nuPlan planners are evaluated with — so scores are directly comparable. The nuPlan analog of `pufferlib/ocean/cosim/carla/`.

## Design

Zero changes to nuplan-devkit or CaRL's `carl_nuplan`. Hydra loads any planner via `planner=pufferdrive_planner` (a config pointing at this package's `PufferDrivePlanner`), so the whole integration is this package:

- `planner.py` — `PufferDrivePlanner(AbstractPlanner)`. The shadow `Drive`
  env holds ONE policy agent (the ego) plus `num_agents - 1` static partner
  slots (`cosim_partner_slots`) that are never spawned or stepped by
  PufferDrive. Per planning iteration (dt = the scenario's
  `database_interval`): stream nuPlan's `PlannerInput` (`DetectionsTracks`,
  traffic lights) into the partner slots (`nuplan_bridge` transforms),
  recompute observations, run the policy on the ego row, `env.step`
  (PufferDrive integrates the ego one dt), and return the integrated pose as
  the trajectory. The ego is synced from nuPlan telemetry only at the first
  iteration; afterwards the shadow env owns it and the planner raises if
  nuPlan's ego pose ever diverges from the integrated one (the configs pin
  `perfect_tracking_controller`).
- Route goals (`goal_source: external`): PDM-style lane-graph route ->
  goals every `goal_spacing` m. The shadow env consumes the goals of its
  `num_goals` window itself exactly like training (`goal_regen_mode: finite`,
  consumed slots zeroed in the obs); the planner pushes the next window only
  when the current one is exhausted or its current goal is clearly behind the
  ego. Lights nuPlan does not report are GREEN (training never produces
  UNKNOWN).
- `pufferlib/ocean/cosim/nuplan_bridge.py`
- `config/` — the Hydra configs that plug the planner in without touching
  nuplan-devkit or `carl_nuplan`: `planner/pufferdrive_planner.yaml` and two
  challenge presets, `closed_loop_{nonreactive,reactive}_agents_pufferdrive`
  (differing only in the `observation` override: log-replay vs IDM agents).
  Found via `hydra.searchpath=[pkg://pufferlib.ocean.cosim.nuplan.config, ...]`.
  Use city bins

## How to run

```
JOBID=13838103
export CKPT=/scratch/yw4142/runs/2026-07-07_multi_agent_nightly_best_seed2_run_name2026-07-07_seed2_499f818/puffer_drive_nlemvgc0/models/model_puffer_drive_000954.pt
export LIMIT_TOTAL_SCENARIOS=10

srun --jobid=$JOBID --overlap \
  bash /scratch/yw4142/PufferDrive_nightly/pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh
```

## Lane graph / GPS goal features

The deployed city bins (`/scratch/yw4142/datasets/ad/nuplan/maps/nuplan__*.bin`,
converted 2026-07-09) DO carry lane graphs — built by 123Drive's bin_factory
(`src/bin_factory/transforms/graph.py`: directed scipy Dijkstra over lane
entry/exit connectivity; the n×n distance matrix dominates bin size, e.g.
Vegas 4583 lanes → 84 MB). With `set_agent_goals`' nearest-lane snapping, the
GPS lane-distance goal columns (`obs_goal_lane_distance`) are therefore live
here, same as the CARLA side.

To (re)generate city bins WITH lane graphs, use the 123Drive converter (from
`/scratch/ev2237/123Drive`):

```bash
uv run convert --preset nuplan --map_only \
  --py123d_path /scratch/ev2237/data/nuplan/py123d_output \
  --output /scratch/yw4142/datasets/ad/nuplan/maps
```

Do NOT use `nuplan_bridge.write_drive_bin` for city bins — it writes an empty
lane graph (`n: 0`), which silently zeroes the GPS goal columns.

## Features not obtainable from nuPlan

- Background (DetectionsTracks) agents carry no acceleration/angular velocity —
  perception tracks, not telemetry. Zeros are exact for partner observations
  (never read for non-ego agents); ego telemetry comes from
  `EgoState.dynamic_car_state`.
- Partner `seconds_stopped` is approximated at the planner's 10 Hz cadence from
  the synced speeds (same caveat as the CARLA side).

## Remaining TODOs

- [ ] The full Val14 split (both challenges) has never been run to
  completion, so total wall time are unmeasured.
- [ ] Double check static variables
- [ ] `nuplan_bridge.write_drive_bin` should build a lane graph (or refuse
  `--map_only`-style use) so a regenerated bin can't silently lose the GPS
  goal features (deployed city bins already have graphs; see above)
