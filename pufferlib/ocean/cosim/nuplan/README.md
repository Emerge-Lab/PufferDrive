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
- Ego plant `ego_dynamics: nuplan_bicycle` (env `EGO_DYNAMICS=nuplan_bicycle`, script 9's default since
  2026-10-30 together with floor 0, the model's own actuator lag, the center cap at margin 0.95 and the floor
  clamp: run 431 = 93.01 vs 93.12 for perfect tracking, driven exactly as CaRL drives the model; the
  lag-compensated variant scores 93.48 (run 428) but inverts the plant's actuator model and is kept off;
  `EGO_DYNAMICS=pufferdrive` is the old eval; script 9 tags the run
  `_bicycle_vmin<floor>[_nolag][_centercap<m>][_clamp]`): nuPlan's
  `KinematicBicycleModel` is the plant, driven through
  CaRL's `one_stage_controller` exactly like CaRL. Every step the shadow ego takes nuPlan's pose, speed
  and yaw rate but keeps its own jerk-integrated accel/steering (the state the policy was trained to
  observe; restamping the lagged plant's values made Drive's zero-crossing snaps fire every step and
  swallow the policy's steering), and its post-step accel/steering become the model's acceleration /
  steering-rate commands (`cosim/nuplan_bicycle.py`). The model low-passes both (gain dt/(dt+tau), tau
  0.2 s / 0.05 s) and has neither a speed floor nor a stop snap: `bicycle_min_speed_mps`
  (`BICYCLE_MIN_SPEED_MPS`; 0 = CaRL's no-reverse driving, -2 = Drive's floor) is enforced on the command,
  a shadow stop snap commands the model to exactly zero speed, and `bicycle_undo_low_pass`
  (`BICYCLE_UNDO_LOW_PASS`) pre-compensates the lag so one model step lands on the targets (off = CaRL's
  lag, the model realises a third of each accel step). The planner checks every step that nuPlan's accel and speed are the
  model's response to its last command, so a perfect-tracking controller or mismatched time constants
  abort the scenario. Comfort under the bicycle model: nuPlan scores the acceleration at the vehicle
  center, which the devkit computes as rear-axle accel + d*yaw_rate^2 + d*yaw_accel (d = 1.46 m), so
  pulling away at the 2.30 m/s^2 cap into a left turn reads above the 2.40 bound (Val14: ~250 scenarios;
  perfect tracking drops the angular velocity and never sees it). `bicycle_center_accel_cap`
  (`BICYCLE_CENTER_ACCEL_CAP`, tag `_centercap`) is CaRL's clip_angular_adjustment: the plant's accel is
  reduced while turning so that reading stays under the cap (`bicycle_center_accel_cap_margin`,
  `BICYCLE_CENTER_ACCEL_CAP_MARGIN`, leaves headroom for nuPlan's Savitzky-Golay overshoot);
  `bicycle_max_yaw_jerk_radps3` (`BICYCLE_MAX_YAW_JERK`, tag `_yawjerk<v>`) slows the steering command so the
  yaw acceleration, whose derivative enters nuPlan's longitudinal jerk through the same formula, cannot jump
  within a step; `accel_cap_mps2` (`EGO_ACCEL_CAP_MPS2`, tag `_acap<v>`, any ego dynamics) lowers the ego's
  accel cap itself (Val14: no gain over the center cap, costs progress). Run-426 anatomy knobs (2026-10-30,
  default off): `partner_split_max_length_m` (`PARTNER_SPLIT_MAX_LENGTH_M`, tag
  `_split<v>`): partners longer than this are shown as equal boxes along their heading (training spawns at
  most 7 m, nuPlan buses reach 12.7 m). `STANDSTILL_JERK_DEADBAND` (tag `_deadband<v>`) overrides the shadow
  env's standstill jerk deadband (arch.py 1.5 m/s^3; it was an anti-reverse-creep hack for perfect tracking
  and froze five run-426 stalls whose start intent was below it; the bicycle floor does that job now).
  `bicycle_floor_accel_clamp` (`BICYCLE_FLOOR_ACCEL_CLAMP`, default true, tag `_clamp`): the policy's
  integrated acceleration is clamped to zero while the plant sits on the speed floor, mirroring Drive's stop
  reset (nine run-426 stalls had it wound down to -5 m/s^2 at standstill); false reproduces run 426. Known
  and left to training: Drive's jerk model keeps its kinematic reference at the box center while nuPlan's
  bicycle rotates about the rear axle 1.46 m behind it, so the same steering cuts the nuPlan center to the
  inside of a turn (four inside-rear corner collisions in run 426); no eval-side conversion.
- Ego plant `ego_dynamics: nuplan_tracker` (env `EGO_DYNAMICS=nuplan_tracker`, script 9 tag
  `_tracker_<lqr|ilqr>_step<s>_veh<n>[_reverse]`): GIGAFLOW's nuPlan evaluation ("we roll out an entire scenario
  using GIGAFLOW policies for all agents ... and use the subsequent observed trajectory from the driver as its
  trajectory"). The planner returns a real trajectory and nuPlan's own `two_stage_controller` drives the
  `KinematicBicycleModel` along it, the setup every trajectory planner on the benchmark is scored with.
  - Rollout: every planning step the shadow env takes the whole scene from nuPlan and is rolled out for
    `horizon_seconds` (8 s) in `rollout_step_seconds` steps (`ROLLOUT_STEP_SECONDS`, 0.3 s = the training step, 27
    steps). The ego starts from the plant's full state (pose, speed, yaw rate, acceleration, steering angle) and is
    driven by the policy; its rolled-out states, one every 0.3 s, are the trajectory. The rollout is thrown away
    afterwards: a goal the ego consumes in its first step stays consumed (the other modes consume on their one
    step too), later ones are undone (`RouteGoalWindow.snapshot` / `restore`).
  - Other agents (`rollout_policy_vehicles`, `ROLLOUT_POLICY_VEHICLES`, default 0): every other agent keeps the
    velocity observed at this step (a forecast, nothing is read from the log ahead); static objects stay where they
    are; lights hold their state, a yellow runs out into red. With a count above 0 the nearest that many vehicles
    and bicycles are driven by the policy instead. nuPlan gives no route for them, so their goals are invented: one
    every `goal_spacing` along the lane they are on and, at a fork, the exit that turns least
    (`Drive.set_agent_lane_goals`, renewed by `c_step` whenever one is reached); they start with zero acceleration
    and steering, which nuPlan's tracks do not carry. Pedestrians always keep their velocity: nuPlan replays them,
    so they do not yield to the ego, which a policy-driven one would. The reactive challenge simulates no bicycles
    at all (IDM vehicles, replayed pedestrians and static objects). Why 0: with 32 (Val14 runs 2566 / 2567,
    k_scaled_0072) a policy-driven queue leader pulls up ~6 m to the bin's stop line while nuPlan's vehicle stays
    where it stopped, the ego's plan follows it a second later and both trackers act on that plan: LQR creeps into
    the standing lead at 0.05 m/s, iLQR drives into it at 0.5 m/s (about 10 of the 31 and 10 of the 22 at-fault
    collisions those runs added over the bicycle plant; 5 of 10 and 5 of 5 rerun locally with 0 are gone).
  - What the trackers cost (k_scaled_0072, Val14 reactive, policy-driven forecast): LQR 0.897, iLQR 0.870, bicycle
    plant 0.923. nuPlan's LQR picks one acceleration and one steering rate that reach the trajectory's state 1.0 s
    ahead (accel = 0.91 x (speed in 1 s - speed now)). Once the plan stops within that second it brakes at 0.91 x
    speed per second (0.9 m/s^2 at 1 m/s) and below 0.2 m/s with a 2 s time constant: a stop the policy plans
    within 0.6 s takes seconds, never reaches rest and ends in the pedestrian or lead it was for (nuPlan counts a
    moving ego as at fault). It also turns in before the plan does and drops a sub-second swerve. iLQR follows the
    poses, but reaches 2.7 m/s^2 for a planned 2.3 on a straight launch (251 scenarios above nuPlan's 2.40 comfort
    bound; `accel_cap_mps2: 2.0` still gives 2.45).
  - No reversing (`rollout_allow_reverse: false`, `ROLLOUT_ALLOW_REVERSE`): an agent the policy would back up stays
    where it stood, at rest, like `bicycle_min_speed_mps: 0`. Without it 3 of 40 sampled Val14 scenarios backed up
    by up to 4.5 m and one left the drivable area.
  - Tracker: `ego_controller=two_stage_controller` is nuPlan's default LQR tracker, `two_stage_ilqr_controller`
    (this package) its iLQR tracker (`EGO_TRACKER=lqr|ilqr`). The iLQR preset drops nuPlan's `max_solve_time: 0.05`:
    the full 20 iterations take ~45 ms on an idle workstation, so on a loaded node the wall-clock cap cuts them
    short and the command depends on the hardware (CaRL's `LogActionPlanner` keeps the cap). The LQR tracker reads
    the trajectory's speed 1.0 s ahead and iLQR tracks its poses for the whole horizon, which is why the rollout
    covers the horizon instead of one step: a trajectory that ends after 0.3 s makes the LQR tracker accelerate
    and brake at a fifth of what the policy asked for and makes the iLQR warm start raise.
  - Cost: 27 policy calls per planning step instead of 1 (batched over the policy-driven vehicles, if any); 20 s to
    1.5 min per scenario on one core.
- Route goals (`goal_source: external`): PDM-style lane-graph route ->
  goals every `goal_spacing` m. Planner `goal_source: roadblock` (env
  `GOAL_SOURCE=roadblock`) uses the centroids of the CaRL-corrected route
  roadblocks instead (off-polygon centroids moved onto a lane baseline, thinned
  to `goal_spacing`), i.e. the route information CaRL rendered, without a lane
  choice. `goal_source: roadblock_lane` keeps the roadblock goals but puts each
  on the lane that continues the ego's own lane through the route (a lane
  change only where the lane graph forces one), so multi-lane roads stop
  pulling the ego toward the road centre. All come from the challenge's route
  roadblock ids only; the planner
  never reads the logged ego. Eval hack `vru_size_pad_m` (env `VRU_SIZE_PAD_M`): this
  much is added to every pedestrian and bicycle partner box's length and width in the
  policy's observation (training spawns nothing below 0.8 x 0.8 m, nuPlan pedestrians
  are 0.4-0.8 m, bicycles ~0.5 m wide), so they read strictly larger than the smallest
  training vehicle; nuPlan still scores the true boxes. Off by default, script 9 sets
  0.8 m. Eval hack `obs_lane_width_clip_min_m` / `obs_lane_width_clip_max_m` (env
  `LANE_WIDTH_CLIP_MIN_M` / `LANE_WIDTH_CLIP_MAX_M`, Drive env keys via `env_overrides`): the
  lane widths the policy observes are clipped to this range (the CARLA training maps span
  2.85-4.4 m, the nuPlan city bins reach 50 m at junction connectors and multi-lane
  roadblocks). Off by default, script 9 sets 2.85-4.4 m. The shadow env consumes the goals of its
  `num_goals` window itself exactly like training (`goal_regen_mode: finite`,
  consumed slots zeroed in the obs); the planner pushes the next window only
  when the current one is exhausted or its current goal is clearly behind the
  ego. `sliding_goal_window: true` (env `SLIDING_GOAL_WINDOW=true`) instead
  refills the window after every consumed goal, so the policy sees a
  window-final speed-gated goal (which it parks at) only at the true route
  end. Lights nuPlan does not report are GREEN (training never produces
  UNKNOWN).
- Start-up jerk caps (`startup_accel_jerk_cap_mps3` /
  `startup_brake_jerk_cap_mps3`, env `STARTUP_ACCEL_JERK_CAP` /
  `STARTUP_BRAKE_JERK_CAP`, 0 = off) clip the ego's continuous jerk action for
  the first `startup_jerk_cap_seconds` (1.5) of each scenario. nuPlan's
  `ego_lon_jerk` is a Savitzky-Golay derivative that extrapolates at the
  trajectory edges: the policy's full +4 m/s³ ramp reads 2.2 m/s³ mid-scenario
  but 5.0 m/s³ when it starts at t=0 (bound 4.13); a 2 m/s³ cap reads 2.9.
  In the 2026-09-03 val14 run 199 of the 232 lon-jerk comfort failures were in
  the first 1.5 s.
- `pufferlib/ocean/cosim/nuplan_bridge.py`
- `config/` — the Hydra configs that plug the planner in without touching
  nuplan-devkit or `carl_nuplan`: `planner/pufferdrive_planner.yaml` and two
  challenge presets, `closed_loop_{nonreactive,reactive}_agents_pufferdrive`
  (differing only in the `observation` override: log-replay vs IDM agents),
  plus `ego_controller/two_stage_ilqr_controller.yaml` and its tracker
  `ego_controller/tracker/ilqr_tracker_deterministic.yaml` for `nuplan_tracker`.
  Found via `hydra.searchpath=[pkg://pufferlib.ocean.cosim.nuplan.config, ...]`.
  Use city bins

## Visualization and run layout

- `COSIM_OBS_HTML=all|failures|infractions|0` (script 9: `infractions`): one interactive pufferlib.viz
  replay per scenario -> `$GROUP/obs_html/<token>.html`, showing the exact observation
  vector the policy received each step (partner/lane/boundary/light slots, goals), its
  action/value/entropy/probabilities and the encoder max-pool winners. This is what the
  agent saw; road geometry is cropped to 250 m around the driven path. `failures` renders
  only scenarios scoring below `COSIM_OBS_HTML_MAX_SCORE` (default 0.9), `all` every
  scenario, `infractions` only collision / drivable-area / direction / no-progress failures.
  `obs_html/index.html` is the same navigator as the self-play eval replays (pages named
  `s<score>_<flags>_<type>_<token>.html`, worst first). Every `.replay.zlib` is deleted after
  rendering: a page embeds its replay, `scripts/eval/render_obs_html.py` re-renders pages from
  themselves and `pufferlib.viz.read_replay_zlib` reads a page like a `.replay.zlib`.
- nuPlan ground-truth videos are opt-in: `CALLBACKS="[simulation_log_callback, carl_visualization_callback]"`
  renders one `.avi` per scenario (every step of all 1118 scenarios through matplotlib, 14 GB, plus a frame-shape
  print per step in the job log) for the few the tidy keeps. Off since 2026-11-09: the kept simulation logs
  render the same view offline (`analyze_nuplan_cosim.py`, CaRL's `render_simulation_log`).
- After the run `scripts/eval/tidy_nuplan_run.py` (runner env `TIDY`, default on) flattens `$GROUP` to
  `overview.csv` (CLS, sub-metrics, loss importance), `scenarios.csv` (per scenario, worst first), `obs_html/`,
  `simulation_log/<token>.msgpack.xz` of the scenarios `COSIM_OBS_HTML` selects, and `details/`
  (`aggregated_metrics.csv`, `metrics/` parquets, `config.yaml`, `overrides.yaml`, `runner_report.parquet`, the
  SLURM log, and `videos/<token>.mp4` of the kept scenarios when the video callback ran). A Val14 run shrinks
  from 2300 files in 2000 folders (45 GB with videos) to ~220 files in 4 folders and ~0.2 GB for ~100 flagged
  scenarios. The planner no longer pickles its policy into the simulation logs (28 MB -> 0.2 MB each);
  `--strip-planner` rewrites the kept logs of older runs. `scripts/eval/nuplan_run_layout.py` resolves both
  layouts (and the `details/simulation_log/` of runs tidied before 2026-11-09) for `compare_nuplan_evals.py`,
  `nuplan_comfort_breakdown.py`, `analyze_nuplan_cosim.py` and the renderer.

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

Lane widths: the converter writes left + right boundary distance, and nuPlan lanes that include a
parking strip have their baseline 1.7 m from one boundary and 4.3 m from the other, so the policy
believed it had 3 m of room towards the oncoming lane. Eval hack `symmetric_lane_widths` (env
`SYMMETRIC_LANE_WIDTHS`, script 9 sets it): the planner rewrites every lane width as twice the
centerline's distance to the nearer GPKG boundary (`nuplan_bridge.symmetric_lane_widths`) and the
shadow env loads that copy from `bin_cache_dir` (`<city>.<size-mtime>.symw.bin`, one rewrite per city). No bin
regeneration needed; 123Drive's `compute_lane_widths` writes the same definition since 2026-10-29.

Do NOT use `nuplan_bridge.write_drive_bin` for city bins — it writes an empty
lane graph (`n: 0`), which silently zeroes the GPS goal columns.

## Features not obtainable from nuPlan

- Background (DetectionsTracks) agents carry no acceleration/angular velocity —
  perception tracks, not telemetry. Zeros are exact for partner observations
  (never read for non-ego agents); ego telemetry comes from
  `EgoState.dynamic_car_state`.
- Partner `seconds_stopped` is counted per nuPlan track token at the planner's
  10 Hz cadence from the synced speeds and injected with the partner states
  (shadow slots change owner whenever a track appears or ends). It starts from
  the planner's history buffer, about 2 s before the scenario; a track that
  leaves the detections starts over.

## Remaining TODOs

- [ ] The full Val14 split (both challenges) has never been run to
  completion, so total wall time are unmeasured.
- [ ] Double check static variables
- [ ] `nuplan_bridge.write_drive_bin` should build a lane graph (or refuse
  `--map_only`-style use) so a regenerated bin can't silently lose the GPS
  goal features (deployed city bins already have graphs; see above)
