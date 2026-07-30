# PufferDrive policy under nuPlan's closed-loop evaluation

## Goal

Evaluate a PufferDrive-trained policy with nuPlan's **unmodified** `run_simulation.py` closed-loop pipeline — the same simulator, metrics, and scoring CaRL's own nuPlan planners are evaluated with — so scores are directly comparable. The nuPlan analog of `pufferlib/ocean/cosim/carla/`.

## Design

Zero changes to nuplan-devkit or CaRL's `carl_nuplan`. Hydra loads any planner via `planner=pufferdrive_planner` (a config pointing at this package's `PufferDrivePlanner`), so the whole integration is this package:

- `planner.py` — `PufferDrivePlanner(AbstractPlanner)`. Per 10 Hz planning
  iteration: read nuPlan's `PlannerInput` (ego, `DetectionsTracks`, traffic
  lights), overwrite the shadow `Drive` env with it (`nuplan_bridge`
  transforms), recompute observations, run the policy, `env.step` (PufferDrive
  integrates the ego one dt), and return the integrated pose as the
  trajectory
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

## Remaining TODOs

- [ ] The full Val14 split (both challenges) has never been run to
  completion, so total wall time are unmeasured.
- [ ] Route goals is hacky
- [ ] Double check static variables
