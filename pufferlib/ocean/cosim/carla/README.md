# PufferDrive policy under CaRL's CARLA leaderboard

## Goal

Evaluate a PufferDrive-trained policy with **CaRL's unmodified `original_leaderboard`** pipeline — the same evaluator, scenarios, scoring, and statistics that produced the CaRL longest6-v2 numbers

## Design

Zero changes to CaRL. The leaderboard imports any agent file via `--agent`, so the whole integration is this package:

- `leaderboard_agent.py` — `AutonomousAgent` subclass (entry point `PufferAgent`). 
  Loads a PufferDrive checkpoint, runs the policy every `dt / tick_dt` ticks, returns a `carla.VehicleControl` every tick. Get `sensor.camera.rgb` chase cam via `sensors()` — the leaderboard's own standard sensor mechanism, so this doesn't touch the read-only-w.r.t.-CARLA contract; frames stream straight to an mp4 writer rather than buffering in memory, unlike the PufferDrive-side `COSIM_DEBUG_BEV` video.
- `world_sync.py` — `WorldSync`: read-only CARLA -> shadow PufferDrive `Drive` env bridge. 
  Every policy step it overwrites ALL shadow agents (ego included) from CARLA ground truth (ego pose/velocity, nearest vehicles+walkers with true bounding boxes, traffic-light states, route goals every 20 m from the
  leaderboard's dense global plan), recomputes observations, then integrates the policy action one dt to produce a target (speed, yaw).
- `controller.py` — `TrackingController`: 
  convert that shadow-env kinematic target into actual throttle/brake/steer at CARLA's native tick rate

## How to run

CARLA needs the GPU/Vulkan container
`carl` conda env (cp310 — CARLA 0.9.15 ships no cp312 wheel)
  - look into compatability with our env

```bash
JOBID=13552620
export CKPT=/scratch/yw4142/runs/2026-07-07_multi_agent_nightly_best_seed2_run_name2026-07-07_seed2_499f818/puffer_drive_nlemvgc0/models/model_puffer_drive_000954.pt

srun --jobid=$JOBID --overlap \
  bash pufferlib/ocean/cosim/carla/run_leaderboard.sh
```

## Remaining TODOs

- [ ] Town06 / Town07 missing inside CARLA
- [ ] Goal is calculated manually
- [ ] `DynamicObjectCrossing`, `SignalizedJunctionLeftTurn`, etc. — with a `KeyError` printed as `"Skipping scenario 'X' due to setup error: 'X'"`.
- [ ] Run all 36 routes in parallel
- [ ] Double check static variables
