# PufferDrive policy under nuPlan's closed-loop evaluation

## Goal

Evaluate a PufferDrive-trained policy with nuPlan's **unmodified** `run_simulation.py` closed-loop pipeline — the same simulator, metrics, and scoring CaRL's own nuPlan planners are evaluated with — so scores are directly comparable. The nuPlan analog of `pufferlib/ocean/cosim/carla/`.

## Design

Zero changes to nuplan-devkit or CaRL's `carl_nuplan`. Hydra loads any planner via `planner=pufferdrive_planner` (a config pointing at this package's `PufferDrivePlanner`), so the whole integration is this package:

- `planner.py` — `PufferDrivePlanner(AbstractPlanner)`. The shadow `Drive`
  env holds ONE policy agent (the ego) plus `num_agents - 1` static partner
  slots (`cosim_partner_slots`) that are never spawned or stepped by
  PufferDrive. Per planning iteration (dt = the scenario's
  `database_interval`): stream nuPlan's `PlannerInput` (ego state,
  `DetectionsTracks`, traffic lights) into the shadow env (`nuplan_bridge`
  transforms), recompute observations, run the policy on the ego row,
  `env.step` (the jerk model integrates the ego's accel/steering intent one
  dt), and return that intent as an action trajectory for nuPlan's ego plant.
- Ego plant (script 9 runs it with floor 0 and the center cap at margin 0.95: run 431 = 93.01; it tags the
  run `_bicycle_vmin<floor>[_centercap<m>]_clamp`): nuPlan's `KinematicBicycleModel`, driven through
  CaRL's `one_stage_controller` exactly like CaRL, with the model's own actuator lag (the challenge presets
  pin the controller). Every step the shadow ego takes nuPlan's pose, speed
  and yaw rate but keeps its own jerk-integrated accel/steering (the state the policy was trained to
  observe; restamping the lagged plant's values made Drive's zero-crossing snaps fire every step and
  swallow the policy's steering), and its post-step accel/steering become the model's acceleration /
  steering-rate commands (`cosim/nuplan_bicycle.py`). The model low-passes both (gain dt/(dt+tau), tau
  0.2 s / 0.05 s) and has neither a speed floor nor a stop snap: `bicycle_min_speed_mps`
  (`BICYCLE_MIN_SPEED_MPS`; 0 = CaRL's no-reverse driving, -2 = Drive's floor) is enforced on the command
  and a shadow stop snap commands the model to exactly zero speed (the model realises a third of each accel
  step). The planner checks every step that nuPlan's accel and speed are the model's response to its last
  command, so a perfect-tracking controller or mismatched time constants abort the scenario. Comfort under
  the bicycle model: nuPlan scores the acceleration at the vehicle center, which the devkit computes as
  rear-axle accel + d*yaw_rate^2 + d*yaw_accel (d = 1.46 m), so pulling away at the 2.30 m/s^2 cap into a
  left turn reads above the 2.40 bound (Val14: ~250 scenarios). `bicycle_center_accel_cap`
  (`BICYCLE_CENTER_ACCEL_CAP`, tag `_centercap`) is CaRL's clip_angular_adjustment: the plant's accel is
  reduced while turning so that reading stays under the cap (`bicycle_center_accel_cap_margin`,
  `BICYCLE_CENTER_ACCEL_CAP_MARGIN`, leaves headroom for nuPlan's Savitzky-Golay overshoot).
  `STANDSTILL_JERK_DEADBAND` (tag `_deadband<v>`) overrides the shadow env's standstill jerk deadband
  (arch.py 1.5 m/s^3; it was an anti-reverse-creep hack and froze five run-426 stalls whose start intent
  was below it; the bicycle floor does that job now). The policy's integrated acceleration is clamped to
  zero while the plant sits on the speed floor, mirroring Drive's stop reset (nine run-426 stalls had it
  wound down to -5 m/s^2 at standstill). Known
  and left to training: Drive's jerk model keeps its kinematic reference at the box center while nuPlan's
  bicycle rotates about the rear axle 1.46 m behind it, so the same steering cuts the nuPlan center to the
  inside of a turn (four inside-rear corner collisions in run 426); no eval-side conversion.
- Route goals (`goal_source: external`): the centroids of the CaRL-corrected
  route roadblocks (off-polygon centroids moved onto a lane baseline, thinned
  to `goal_spacing`), i.e. the route information CaRL rendered, without a lane
  choice. They come from the challenge's route roadblock ids only; the planner
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
  (log-replay vs IDM agents, each with its metrics; both pin `one_stage_controller`).
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
GPS lane-distance goal columns are therefore live
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
