# PufferDrive policy under CaRL's CARLA leaderboard

## Goal

Evaluate a PufferDrive-trained policy with **CaRL's unmodified `original_leaderboard`** pipeline — the same evaluator, scenarios, scoring, and statistics that produced the CaRL longest6-v2 numbers

## Design

Zero changes to CaRL. The leaderboard imports any agent file via `--agent`, so the whole integration is this package:

- `leaderboard_agent.py` — `AutonomousAgent` subclass (entry point `PufferAgent`). 
  Loads a PufferDrive checkpoint, builds the shadow `Drive` env from the checkpoint's config, and every `dt / tick_dt` ticks: overwrites ALL shadow agents (ego included) from CARLA ground truth (ego pose/velocity, nearest vehicles+walkers with true bounding boxes, traffic-light states, route goals from the leaderboard's dense global plan), recomputes observations, runs the policy, integrates one dt to produce a target (speed, yaw), and returns a `carla.VehicleControl` every tick. Get `sensor.camera.rgb` chase cam via `sensors()` — the leaderboard's own standard sensor mechanism, so this doesn't touch the read-only-w.r.t.-CARLA contract; frames stream straight to an mp4 writer rather than buffering in memory, unlike the PufferDrive-side `COSIM_DEBUG_BEV` video.
- `controller.py` — `TrackingController`: 
  convert that shadow-env kinematic target into actual throttle/brake/steer at CARLA's native tick rate

All shadow-env assumptions about the model (observation layout, dynamics model, dt, goal
count/radius, reward conditioning) are adopted from the checkpoint's sibling `config.yaml`
(`cosim/arch.py`'s `resolve_arch`); the config keys that contract depends on are pinned by
`tests/unit_tests/test_cosim_config_contract.py`.

## How to run

CARLA needs the GPU/Vulkan container
`carl` conda env (cp310 — CARLA 0.9.15 ships no cp312 wheel)
  - look into compatability with our env

`run_leaderboard.sh` exports `SCENARIO_RUNNER_ROOT` itself. When invoking the
evaluator manually you MUST export it (route scenarios silently fail to load
without it; `leaderboard_agent.py` refuses to start when it is unset):

```bash
JOBID=14401249
export CKPT=/scratch/yw4142/PufferDrive_nightly/weights/mimolette/models/model_puffer_drive_003815.pt
export SCENARIO_RUNNER_ROOT=/scratch/yw4142/CaRL/CARLA/original_leaderboard/scenario_runner

srun --jobid=$JOBID --overlap \
  bash pufferlib/ocean/cosim/carla/run_leaderboard.sh
```

Optional env vars (see `leaderboard_agent.py` docstring): `COSIM_DEVICE`,
`COSIM_DYNAMICS_SOURCE`, `COSIM_DT`, `COSIM_NUM_AGENTS`, `COSIM_GOAL_SPACING`,
`COSIM_DEBUG_BEV`, `COSIM_DEBUG_CARLA_VIEW`, `COSIM_RECORD_INFRACTIONS`.

### Ego dynamics source

`COSIM_DYNAMICS_SOURCE=carla` (default) converts the shadow env's target
(speed, yaw) into `carla.VehicleControl` via `controller.py`'s
`TrackingController`, and CARLA's own vehicle physics moves the ego — subject
to real tracking lag against a jerk-model target the policy never trained
against (measured: `yaw_err_max` up to 90° on a real route, and even a
zero-CARLA-physics bicycle+PI proxy of the same controller shows a real
lane_dist/lane_angle drift within ~1s of a sustained turn).

`COSIM_DYNAMICS_SOURCE=pufferdrive` skips the controller entirely: PufferDrive's
own dynamics (the ones the policy was trained on) move the ego, and the CARLA
actor is teleported to match every policy step. Physics is left ON — disabling
it on this already-active, leaderboard-managed hero actor segfaults the UE4
engine (tried at several points in the agent's init sequence, same crash every
time) — but `set_transform()` overrides it every step regardless, so the net
effect matches the standalone `carla_cosim.py` loop's design (which disables
physics, but only because it spawns its own ego before any tick happens).
Target linear/angular velocity is also pushed each step so CARLA's own
`get_velocity()` (read by CaRL's own criteria, e.g. `MinSpeedTest`) reflects
the actual motion. This removes the dynamics-mismatch tracking lag at the cost
of the ego no longer being a physically-simulated CARLA vehicle; the
leaderboard's route/collision/infraction criteria still see it correctly since
they only read its pose.

### Infraction clips

`COSIM_RECORD_INFRACTIONS=/dir` keeps a rolling ~5 s chase-cam buffer and dumps
one short mp4 whenever the shadow env's own detectors flag an ego
collision/offroad/red-light (at most one clip per 10 m of ego travel) — the
co-sim analog of CaRL `eval_agent.py`'s `RECORD=1` infraction videos. The
trigger is the shadow env's proxy detection, not the leaderboard's official
infraction ledger; use it for debugging, not scoring.

### Route goals and the GPS lane-distance feature

Goals are cut from the leaderboard's dense global plan (`set_global_plan`'s
`dense_global_plan_world_coord`, ~1 m spacing, already lane-centered) every
`COSIM_GOAL_SPACING` meters (default 20). On the C side, `set_agent_goals`
snaps each externally-set goal to its nearest direction-matched drivable lane
(`find_goal_lane` in drive.h), so the GPS lane-distance observation columns
(`obs_goal_lane_distance`) are live, matching how `goal_source=map` training
goals always carry a lane. Because inter-goal navigation is led by those GPS
features (lane-graph distance to the goal's lane, not the euclidean direction
to the goal), larger spacings — training samples
`min_goal_spacing`..`max_goal_spacing`, e.g. 20–200 m — are in-distribution
and do not cause wrong turns at forks. Requires the town bin to carry a lane
graph (all `carla/opendrive__Town*.bin` do).

## Slurm launch (verified 2026-08-06)

One self-contained sbatch per route: the job gets its own GPU, starts its own
CARLA server, runs the evaluator, and writes `result.json` + videos under
`$RUNDIR`. Single route (e.g. longest6 route 5, the shortest at ~686 m):

```bash
PD=/scratch/yw4142/PufferDrive_cosim
CKPT=$PD/weights/mimolette/models/model_puffer_drive_003815.pt
RUNDIR=/scratch/yw4142/runs/cosim_lb_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUNDIR"
sbatch -A torch_pr_355_tandon_advanced -p l40s_public --gres=gpu:1 -c 8 --mem=64G -t 02:00:00 \
  --job-name=cosim_r5 --output="$RUNDIR/slurm.out" --error="$RUNDIR/slurm.out" \
  --wrap "CKPT=$CKPT ROUTES_SUBSET=5 OUT=$RUNDIR/result.json \
bash $PD/pufferlib/ocean/cosim/carla/run_leaderboard.sh"
```

### All 36 longest6 routes in parallel

Same pattern as CaRL's `evaluate_routes_slurm.py` (one job per split route,
one CARLA server per job, aggregate at the end). Distinct `CARLA_PORT`s guard
against two jobs landing on the same node:

```bash
PD=/scratch/yw4142/PufferDrive_cosim
CKPT=$PD/weights/mimolette/models/model_puffer_drive_003815.pt
SPLIT=/scratch/yw4142/CaRL/CARLA/custom_leaderboard/leaderboard/data/longest6_split
BASE=/scratch/yw4142/runs/cosim_lb_parallel_$(date +%Y%m%d_%H%M%S)
for i in $(seq -w 0 35); do
  RUNDIR=$BASE/route_$i; mkdir -p "$RUNDIR"
  sbatch -A torch_pr_355_tandon_advanced -p l40s_public --gres=gpu:1 -c 8 --mem=64G -t 04:00:00 \
    --job-name=cosim_r$i --output="$RUNDIR/slurm.out" --error="$RUNDIR/slurm.out" \
    --wrap "CKPT=$CKPT ROUTES=$SPLIT/longest6_$i.xml ROUTES_SUBSET= \
CARLA_PORT=$((2000 + 10#$i * 4)) OUT=$RUNDIR/result.json \
bash $PD/pufferlib/ocean/cosim/carla/run_leaderboard.sh"
done
# aggregate: CaRL's tools/result_parser.py over the per-route result.json files
```

CaRL's script additionally monitors and resubmits crashed routes — worth
porting if crash rates matter.

## Features not obtainable from CARLA

- Partner (background-agent) `seconds_stopped` is approximated: it accumulates
  from the shadow env's own step cadence (one update per policy dt from the
  synced speeds), not from CARLA's per-tick velocities. Stopped CARLA vehicles
  read as stopped, but the duration resolution is one policy dt.
- Background yaw-rate / longitudinal acceleration come from CARLA's physics
  (`get_angular_velocity`/`get_acceleration`); they feed the ego's steering
  and acceleration observation features only for the ego itself — partner
  observations never read them, so nothing is lost there.

## Remaining TODOs

- [ ] Double check Town 6 offset
- [ ] handle 2 kinds of goal, different kind of goals
- [ ] per tick smoothness
- [ ] some town might have routing issue, off road error