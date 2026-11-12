# PufferDrive policy under CaRL's CARLA leaderboard

## Goal

Evaluate a PufferDrive-trained policy with **CaRL's unmodified `original_leaderboard`** pipeline — the same evaluator, scenarios, scoring, and statistics that produced the CaRL longest6-v2 numbers

## Design

Zero changes to CaRL. The leaderboard imports any agent file via `--agent`, so the whole integration is this package:

- `shadow_ego.py` — `ShadowEgo`, the carla-free core shared with the Leaderboard 1.0 integration (`lb1/`).
  Loads a PufferDrive checkpoint, builds the shadow `Drive` env from the checkpoint's config, calibrates the bin frame, maps lights/stop signs and route goals, and every CARLA tick (the shadow env runs at the 0.05 s tick, no action repeat): overwrites ALL shadow agents (ego included) from the CARLA-frame snapshot (ego pose/velocity, nearest vehicles+walkers with true bounding boxes, traffic-light states, route goals from the leaderboard's target points), recomputes observations, runs the policy and integrates one dt into the ego's next pose (or into pedals and steer for CARLA's own physics). All per-route logs (telemetry, world log, obs replay) are written here.
- `leaderboard_agent.py` — `AutonomousAgent` subclass (entry point `PufferAgent`), the CARLA side: reads the ground truth into the snapshot, teleports the ego to the returned pose (resting it on the road mesh), runs CaRL's criteria for the `carla_*` telemetry columns, and returns a `carla.VehicleControl` every tick. Get `sensor.camera.rgb` chase cam via `sensors()` — the leaderboard's own standard sensor mechanism, so this doesn't touch the read-only-w.r.t.-CARLA contract; frames stream straight to an mp4 writer. `COSIM_OBS_HTML` records the interactive observation replay (exact policy input/outputs).
- `controller.py` — `PedalMap`:
  turns the jerk integrator's acceleration and steering into throttle/brake/steer for CARLA's own vehicle physics (`COSIM_DYNAMICS_SOURCE=carla_pedals`)
- `lb1/` — the same policy under carla_garage's Leaderboard 1.0 (CARLA 0.9.10.1, longest6 v1): a Python 3.7 client agent plus `policy_server.py` running `ShadowEgo` in the PufferDrive venv; see `lb1/README.md`.

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
export CKPT=/scratch/yw4142/PufferDrive_nightly/experiments/<run>/final_model.pt
export SCENARIO_RUNNER_ROOT=/scratch/yw4142/CaRL/CARLA/original_leaderboard/scenario_runner

srun --jobid=$JOBID --overlap \
  bash pufferlib/ocean/cosim/carla/run_leaderboard.sh
```

Optional env vars (see `leaderboard_agent.py` docstring): `COSIM_DEVICE`,
`COSIM_DYNAMICS_SOURCE`, `COSIM_NUM_AGENTS`, `COSIM_MAX_SPEED_MPS` (ego speed cap via
C_vel; default = the checkpoint's `base_max_speed_mps`, must stay inside the trained range),
`COSIM_ZERO_PARTNER_STOPPED_TIME` (1 = vehicle partners' `seconds_stopped` obs feature held at 0, ego
and walkers untouched; ablation for the parked-car / red-light-queue avoidance), `COSIM_VRU_SIZE_PAD_M`
(added to the length and width of every walker and two-wheeler box in the policy's obs; the
longest6 scripts set 0.8 m), `COSIM_OBS_SLOTS_PARTNERS_N` (partner slots in the policy's obs;
default = the checkpoint's training value, the slots are max-pooled so any count transfers; the
longest6 v1 script sets 40, GIGAFLOW's eval value), `COSIM_OBS_HTML`,
`COSIM_DEBUG_CARLA_VIEW`.

### Ego dynamics source

`COSIM_DYNAMICS_SOURCE=pufferdrive` (default): PufferDrive's
own dynamics (the ones the policy was trained on) move the ego, and the CARLA
actor is teleported to the post-step pose every tick, where the next tick's
observation reads it. The teleport keeps the body flat on CARLA's road plane
(pitch/roll from the waypoint's up vector rotated into the policy's heading,
velocity along the pitched forward vector): a yaw-only pose on Town03/04
grades of 5-15 deg buried the wheels in the mesh and logged 20-30
`static.road` collisions per route. Physics is left ON — disabling
it on this already-active, leaderboard-managed hero actor segfaults the UE4
engine (tried at several points in the agent's init sequence, same crash every
time) — but `set_transform()` overrides it every step regardless.
Target linear/angular velocity is also pushed each step so CARLA's own
`get_velocity()` (read by CaRL's own criteria, e.g. `MinSpeedTest`) reflects
the actual motion. The ego is then no longer a physically-simulated CARLA
vehicle; the leaderboard's route/collision/infraction criteria still see it
correctly since they only read its pose.

`COSIM_DYNAMICS_SOURCE=carla_pedals` drives the CARLA vehicle the way the leaderboard
prescribes: the agent only returns a `carla.VehicleControl` (steer, throttle, brake, reverse) from
`run_step`, nothing else touches the ego, and the gearbox stays automatic: the leaderboard resets
`manual_gear_shift` on every control, so the agent cannot select gears. Like the nuPlan planner's
bicycle model, CARLA is the plant: pose, speed, yaw rate and the current gear are
re-read every tick, while acceleration and steering angle stay the jerk integrator's own state. The
jerk model moves the box centre along its heading, whereas the CARLA car turns about its rear axle,
so its centre travels up to 13 degrees inside its heading in a tight turn. The policy is therefore
given the direction of travel as its heading (body heading + the slip the commanded steering gives
at `rear_axle_to_center_m` ahead of the rear axle); with the body heading it cut every corner
(0.6 m at a 7 m radius). `controller.py`'s `PedalMap` turns acceleration and steering angle
into pedals and steer with constant gains and no feedback (the policy closes the loop). The gains
are per ego blueprint and per gear (`PEDAL_GAINS_BY_BLUEPRINT`: `mkz_2020` for longest6 v2,
`mkz2017` for v1, measured open loop on a straight road with each gear held); an unmeasured car is
refused. For the gear CARLA reports, throttle = the throttle that holds the current speed + the
target acceleration divided by that gear's acceleration per unit throttle, clipped at zero: CARLA's
engine braking makes the throttle the main brake too. The brake pedal supplies what exceeds the
gear's engine braking, capped at `max_engine_braking_mps2` so that hard braking acts at once, and
the whole deceleration while the gearbox shifts (gear 0: the engine is disconnected). Steer = path
curvature divided by the curvature one unit of steer gives, which falls with speed. A car that
has braked to below `STANDSTILL_SPEED_MPS` is held on the full brake, reads exactly stopped and,
like the jerk model, starts its acceleration again from zero. From there a positive acceleration
launches it forward; on a negative one it stays on the brake and never backs up
(`COSIM_PEDAL_REVERSE=1`: it launches backward with `VehicleControl.reverse`, the throttle from
the reverse gains, and the brake for a positive acceleration while it rolls backward; on longest6
v1 that scored 88.9 against 91.3 driving score). The route
summary line prints how far the gains are off (`accel_bias_*`, per gear
`accel_bias_by_gear_mps2` = (ticks, bias) with reverse as gear -1, `yaw_rate_gain`). An automatic
shift leaves the car without drive for `gear_shift_seconds` (0.5 s on the v1 car), which no pedal
value can make up for and the policy has never seen: it would sprint through a yellow and arrive
late. So in this mode the policy sees that much less remaining yellow than CARLA reports and stops
at those lights instead (`COSIM_PEDAL_YELLOW_MARGIN=0`: it sees CARLA's value and sprints). Scripts
11 and 14 pass both switches through and tag the result folder `_reverse` / `_noyellowmargin`.
Not handled: road grade (gravity is not compensated).

### Bin frame calibration

The exported bins sit a constant fraction of a metre off the CARLA frame (2026-09-23, stored
offsets: Town01 0.28 m, Town02 0.62 m, Town06 1.2 m along x), so the ego and every streamed
actor drove that far right of CARLA's lane centre and clipped curbs the shadow env could not
see. `carla_bridge.calibrate_town_offset` fits the translation against CARLA's driving
waypoints at init (median lane residual -> 0.00 m on Town01-06) and refuses a bin whose lanes do
not match the loaded map; the agent logs the applied shift.

### HTML report (nuPlan-style)

`COSIM_WORLD_LOG=/dir` makes the agent write one `.npz` per route with the bin-frame world
state at every policy step (ego pose/speed, every streamed partner, light states, route goals,
dense route). `scripts/eval/analyze_carla_cosim.py <run_dir> [...] <report_dir>` then builds
the same report as `analyze_nuplan_cosim.py`: score table worst first, failure categories,
per-route diagnosis (standstills with the ego's own light state, infractions located in time),
six-frame top-down strips over the bin map, speed plots and the chase-cam / top-down videos.
`<run_dir>` is the directory holding `result.json`, `world_log/`, `telemetry/` and `carla_view/`.

### Route goals and the GPS lane-distance feature

Goals are the leaderboard's own target points (`set_global_plan`'s sparse
`_global_plan_world_coord`: junction entries/exits, lane changes and one point
every 200 m, the same list every leaderboard sensor agent gets). The ego always
sees the next `num_goals` of them as a sliding window, refilled after every
consumed point, so the window shrinks to 2 and then 1 only at the route end
(the shadow env's speed-gated window-final goal is then the true end). The
route's first point sits under the ego and is skipped. Each goal's direction
is the dense plan's local direction there; on the C side `set_agent_goals`
snaps each goal to its nearest direction-matched drivable lane
(`find_goal_lane` in drive.h), so the GPS lane-distance observation columns
are live, matching how `goal_source=map` training
goals always carry a lane. Requires the town bin to carry a lane graph (all
`carla/opendrive__Town*.bin` do).

## Slurm launch (verified 2026-08-06)

One self-contained sbatch per route: the job gets its own GPU, starts its own
CARLA server, runs the evaluator, and writes `result.json` + videos under
`$RUNDIR`. Single route (e.g. longest6 route 5, the shortest at ~686 m):

```bash
PD=/scratch/yw4142/PufferDrive_cosim
CKPT=$PD/experiments/<run>/final_model.pt
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
CKPT=$PD/experiments/<run>/final_model.pt
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
