# OpenGigaFlow policy under CaRL's CARLA leaderboard

## Goal

Evaluate an OpenGigaFlow-trained policy with **CaRL's unmodified `original_leaderboard`** pipeline — the same evaluator, scenarios, scoring, and statistics that produced the CaRL longest6-v2 numbers

## Design

Zero changes to CaRL. The leaderboard imports any agent file via `--agent`, so the whole integration is this package:

- `shadow_ego.py` — `ShadowEgo`, the carla-free core shared with the Leaderboard 1.0 integration (`lb1/`).
  Loads an OpenGigaFlow checkpoint, builds the shadow `Drive` env from the checkpoint's config, calibrates the bin frame, maps lights/stop signs and route goals, and every CARLA tick (the shadow env runs at the 0.05 s tick, no action repeat): overwrites the ego and the partners from the CARLA-frame snapshot (ego pose/velocity, nearest vehicles+walkers with true bounding boxes, traffic-light states, route goals from the leaderboard's target points), recomputes observations, runs the policy and integrates one dt into pedals and steer for CARLA's own physics. All per-route logs (telemetry, world log, obs replay) are written here.
- `leaderboard_agent.py` — `AutonomousAgent` subclass (entry point `OpenGigaFlowAgent`), the CARLA side: reads the ground truth into the snapshot, runs CaRL's criteria for the `carla_*` telemetry columns, and returns the policy's `carla.VehicleControl` every tick. Get `sensor.camera.rgb` chase cam via `sensors()` — the leaderboard's own standard sensor mechanism, so this doesn't touch the read-only-w.r.t.-CARLA contract; frames stream straight to an mp4 writer. `COSIM_OBS_HTML` records the interactive observation replay (exact policy input/outputs).
- `controller.py` — `PedalMap`:
  turns the jerk integrator's acceleration and steering into throttle/brake/steer for CARLA's own vehicle physics
- `lb1/` — the same policy under carla_garage's Leaderboard 1.0 (CARLA 0.9.10.1, longest6 v1): a Python 3.7 client agent plus `policy_server.py` running `ShadowEgo` in the OpenGigaFlow venv; see `lb1/README.md`.

All shadow-env assumptions about the model (observation layout, dynamics model, dt, goal
count/radius, reward conditioning) are adopted from the checkpoint's sibling `config.yaml`
(`cosim/arch.py`'s `shadow_env_kwargs`); the config keys that contract depends on are pinned by
`tests/unit_tests/test_cosim_config_contract.py`.

## How to run

All 36 longest6 v2 routes on an 8-GPU node: `scripts/slurm/eval_longest6_v2.sh`. Locally, against a CARLA 0.9.15
server you started yourself, in a cp310 env (CARLA 0.9.15 ships no cp312 wheel): `run_evaluator.py --model-dir <dir>
--routes <xml> ...` (VS Code launch config "Cosim: CARLA leaderboard agent") with `CARL_WORK_DIR` and
`SCENARIO_RUNNER_ROOT` exported (route scenarios silently fail to load without the latter, so `leaderboard_agent.py`
refuses to start). The agent's environment variables are listed in `leaderboard_agent.py`, the shadow env's (`COSIM_*`)
in `shadow_ego.py`; its `COSIM_EVAL_DEFAULTS` are the eval settings behind our scores, except that
`eval_longest6_v2.sh` keeps the checkpoint's partner slots.

### Ego dynamics

The agent drives the CARLA vehicle the way the leaderboard
prescribes: it only returns a `carla.VehicleControl` (steer, throttle, brake) from
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
launches it forward; on a negative one it stays on the brake and never backs up (reversing scored
88.9 against 91.3 driving score on longest6 v1). The route
summary line prints how far the gains are off (`accel_bias_*`, per gear
`accel_bias_by_gear_mps2` = (ticks, bias), `yaw_rate_gain`). An automatic
shift leaves the car without drive for `gear_shift_seconds` (0.5 s on the v1 car), which no pedal
value can make up for and the policy has never seen: it would sprint through a yellow and arrive
late. So the policy sees that much less remaining yellow than CARLA reports and stops
at those lights instead (`COSIM_PEDAL_YELLOW_MARGIN=0`: it sees CARLA's value and sprints). The longest6
scripts pass that switch through and tag the result folder `_noyellowmargin`.
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

## Features not obtainable from CARLA

- Partner (background-agent) `seconds_stopped` is approximated: it accumulates
  from the shadow env's own step cadence (one update per policy dt from the
  synced speeds), not from CARLA's per-tick velocities. Stopped CARLA vehicles
  read as stopped, but the duration resolution is one policy dt.
- Background yaw-rate / longitudinal acceleration come from CARLA's physics
  (`get_angular_velocity`/`get_acceleration`); they feed the ego's steering
  and acceleration observation features only for the ego itself — partner
  observations never read them, so nothing is lost there.
