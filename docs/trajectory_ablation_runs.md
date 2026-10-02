# spline_werling trajectory ablation (branch `aditya/trajectory_ablation`)

Code: commit `b39be70d` on top of `731e7117` ("Adding Werling Spline Dynamics"). W&B project
`emerge_/pufferdrive`, group `spline_werling`, tag `s30bstraj`. Launched 2026-10-01 with
`scripts/launch_fairshare.py` (30B steps, 40 h + `afterany` continuation, a100|h100|h200, **24 CPUs**).

## 1. What was slow or not improving (measured 2026-10-01, running runs untouched)

Comparisons are at equal agent steps against `spline_baseline` (jerk dynamics, same rewards).

| # | Issue | Evidence |
|---|-------|----------|
| 1 | Goal progress stalls | v1 goals/episode 0.155 vs 0.094 at 1B, 0.183 vs 0.215 at 2.3B, 0.25 vs 0.59 at 5B; DNF 0.61 vs 0.39 |
| 2 | Goals the lattice cannot reach | `goal_source: map` drops the first goal on a uniformly random map lane. Along lane links (no U-turns) 35/63 goals are unreachable in Town04, 29/64 in Town05, 18/64 in Town06, 14/63 in Town10HD. Reachable routes average 364-3531 m for a 129-460 m straight-line distance |
| 3 | Route features carry no information | car->goal lane distance is capped at 500 m for 40-63 of 64 agents per town; per-exit distances are capped at the first split in 70/120 (Town01) to 71/72 (Town04); unreachable and far both read 1.0, so goal exit mode ties to slot 0 |
| 4 | Queuing behind a stopped car | harness, 22 straight 2-lane sites (Town03/04/06), lead car stopped 12-30 m ahead, ego at 2-5 m/s, v1 checkpoint 002250: p(lane change) per decision 0.001-0.005, lower than on an empty road (0.006-0.008); p(stop cell) rises to 0.44-0.88. Masks allow a lane change at all 22 sites, so the policy, not the lattice, refuses |
| 5 | Waiting is nearly free | velocity reward is binary above 2.5 m/s (7.5e-4 per step); waiting 10 s costs ~0.025, a 2 % crash risk costs 0.03 |
| 6 | Slow to move | time to first motion 15-88 s, moving fraction 0.36-0.46, speed 0.7-1.05 m/s vs 1.2-1.8 for the baseline |
| 7 | Plans re-chosen about every other decision | lat_new_rate 0.43 -> 0.50, lon_new_rate 0.44 -> 0.55 over training, all runs |
| 8 | Red-light violations rising | v1 0.086 -> 0.134 while baseline falls 0.115 -> 0.099; the plan block had stop-line distance but no light state |
| 9 | Velocity scaling trades off | vel5x: first motion 1.8 s, but goals 0.171 vs 0.183 at 2.3B, red light 0.18, collisions 0.156; vel50x collisions 0.28-0.31 |
| 10 | Low SPS | 60-100 K vs 134-177 K for the baseline. Jobs keep 14.2-14.7 of 16 cores busy; 90 % of wall time is rollout, of which 62 % is waiting on CPU envs. Callgrind: lattice masks are 65 % of `c_step` (longitudinal cells 42 %) |
| 11 | No overtaking lanes in 3 towns | same-direction neighbour lanes cover 0 % of Town01/Town02 and 0.9 % of Town07 lane length (structural, not fixed here) |

## 2. Fixes (each is a separate, switchable change)

| Fix | Change | Addresses | Why it should not regress |
|-----|--------|-----------|---------------------------|
| F2 route features | Per-exit feature = extra route over the best exit / 200 m (1 = unreachable or absent); car->goal feature log-scaled to 5 km; goal exit mode uses uncapped distances | 1, 3 | Same observation size; only replaces saturated values |
| F3 light state | Plan block +2: red / yellow at the next stop line on the chain (58 -> 60 plan features) | 8 | Information only |
| Preview fix | Preview advanced rail arc by d_sigma*factor/sqrt(factor^2+d'^2); now d_sigma/sqrt(factor^2+d'^2) (checks already used the exact relation) | obs accuracy on curves | Identical on straight rails or at d = 0 |
| G route goals | `env.goal_source=route` (existing option): goals 20-200 m apart along a random lane walk; 512/512 reachable, mean route 95-126 m | 1, 2 | Task change, see caveats |
| F6 route progress | `env.reward_route_progress` per meter of lane-route distance gained toward the current goal; potential capped at 1000 m; unmeasurable stretches keep the last baseline, so each goal's total is k x (start - end distance) | 4, 5, 6 | Potential-style shaping: telescopes, nothing to farm; pays for passing a blocker, never for detours |
| F1 plan consistency | `env.reward_trajectory_consistency` x RMS world distance between the path observed before a decision and the newly committed one (5 preview samples, 0.6-3.0 s at dt 0.3), only when a plan changes; logged as `lattice/plan_change_rms_m` (all lattice runs) | 7 | At 2e-4/m a hard emergency brake (RMS 8-15 m) costs <= 3e-3 vs collision 1.5; random-policy RMS median 1.0 m |
| SPS | 24 CPUs per run (`scripts/cluster_configs/ag11023_priority_anygpu_cpu24.yaml`), num_envs unchanged at 20 | 10 | No code or learning change; extra code costs <= 5 % per agent-step (bench 85.9-87.8 vs 83.6-84.9 us) |

## 3. Runs (each adds one feature to the previous)

All: `puffer train puffer_drive_spline_werling env.lattice_exit_mode=goal` plus the code fixes F2, F3 and the preview fix.

| Run (wandb name) | Main / continuation | Adds | Overrides beyond the base |
|------------------|---------------------|------|---------------------------|
| `spline_werling_trajA_info` | 18935213 / 18935215 | F2, F3, preview fix, 24 CPUs | none |
| `spline_werling_trajB_routegoals` | 18935221 / 18935224 | + reachable route goals | `env.goal_source=route` |
| `spline_werling_trajC_progress` | 18935229 / 18935231 | + route-progress reward | `+ env.reward_route_progress=1e-3` |
| `spline_werling_trajD_consistency` | 18935235 / 18935236 | + plan-consistency penalty | `+ env.reward_trajectory_consistency=2e-4` |

| `spline_werling_trajA_info_wait_pen` | 18956052 / 18956053 | A + waiting penalty | `env.reward_wait_penalty_frac=5e-4 env.lattice_light_in_view=false` |
| `spline_werling_trajC2_routegoals_wait_pen` | 18956054 / 18956055 | B + waiting penalty | `env.goal_source=route env.reward_wait_penalty_frac=5e-4 env.lattice_light_in_view=false` |
| `spline_werling_trajD2_mapgoals_wait_pen` | 18956057 / 18956058 | A + waiting penalty + consistency | `env.reward_wait_penalty_frac=5e-4 env.reward_trajectory_consistency=2e-4 env.lattice_light_in_view=false` |
| `spline_werling_trajE_overtake_wait_pen` | 18986342 / 18986344 | A + waiting penalty + borrowing the oncoming lane (section 7) | `env.reward_wait_penalty_frac=5e-4 env.lattice_light_in_view=false env.lattice_oncoming_overtake=true env.reward_oncoming_penalty_frac=5e-4` |
| `spline_werling_trajE_overtake_wait_pen_consist10x` | 19026614 / 19026615 | trajE + plan consistency at 10x D's coefficient | trajE's + `env.reward_trajectory_consistency=2e-3` |
| `spline_werling_trajE_overtake_wait_pen_consist50x` | 19026638 / 19026639 | trajE + plan consistency at 50x D's coefficient | trajE's + `env.reward_trajectory_consistency=1e-2` |

The three `*_wait_pen` runs use commit `d9f1e2ee` (waiting penalty) on top of `37157e4c` (view-gated light features,
switched off for them so they differ from A only as listed). Waiting penalty: each step a car pays
0.0005 x min(its collision, offroad, stop-line coefficient) x max(0, 1 - speed / 1 m/s), except while its reported next
light is red or yellow; a whole standstill episode costs at most 0.46 of that cheapest infraction (gamma 0.999, 2560 steps).

Cancelled 2026-10-01: `spline_werling_vel50x` (18867873 after 24 h, continuation 18867876), and before they started
`trajC_progress_mapgoals` (18953137 / 138), `trajD_consistency_mapgoals` (18953142 / 143) and the
first `trajE_overtake_wait_pen` submission (18976365 / 366). Also cancelled 2026-10-01 to free a GPU:
`spline_werling_vel10x` (18867871 after 34 h 17 m, continuation 18867872). Cancelled 2026-10-02 to free GPUs:
`spline_baseline` (18811661 after 23 h 08 m) and `spline_werling_vel5x` (continuation 18867868 after 6 h 46 m).

Compare A with `spline_werling_goal` (same exit mode, old code). With map goals the progress potential is flat beyond
1000 m of route and pays nothing toward an unreachable goal, so in C_map / D_map it acts on 60 % of first goals. B-D change the goal task, so compare them with each other
and with A, not on raw goal counts against map-goal runs.

What to watch:
- A: `exit_nonstraight_rate`, goals and DNF vs `spline_werling_goal`; red-light rate vs v1; SPS.
- B: goals per episode, DNF, collisions.
- C: `reward_components/route_progress`, moving fraction, time to first motion, `chosen_change_rate`, avg speed; collisions must not climb like vel5x.
- D: `lattice/plan_change_rms_m` and lat/lon new rates should fall, with goals and collisions flat vs C.

## 4. Caveats

- Observation size grows from 1072 to 1074, so checkpoints from earlier runs do not load into these runs. With
  `lattice_oncoming_overtake` it grows by 5 more (plan flag + 4 lateral cells) and the lateral head has 24 cells.
- Route goals change what "goal reached" means. Evaluation (WOSAC, human replay) still uses logged goals.
- `goal_source: route` follows the agent's random route, so the first goal can sit on a lane the policy has not chosen yet; goal exit mode steers to it by lane-graph distance.

## 5. Later measurements (2026-10-01, corrected)

- No map goal is unreachable once lane changes count (0 of 510 first goals). An earlier version of this section followed
  lane links only and reported "sealed lane sets"; those are links-only artefacts (e.g. Town10HD's 16-lane ring, where 13
  lanes have a same-direction neighbour outside it).
- The map file's lane-graph distance follows links only. It feeds `g`, `c`, goal-mode exit choice and the progress
  reward, and it scores 167 of 234 (map goals) and 240 of 242 (route goals) lane changes on offer at reset as a >=100 m
  detour or a dead end while the true route changes by <30 m. Goal-mode exit picks differ from the lane-change-aware
  shortest at 58 of 351 (map) and 11 of 354 (route) first junctions. This is why the progress reward (C, D) can penalise
  a harmless lane change; the waiting penalty does not use route distance.
- Lane-change-aware first-goal routes (map goals): 368-1743 m mean per town, median 2.1-3.7x the straight line; 354 of
  510 within 768 m. Route goals: 104-124 m.
- Old goal exit mode (500 m cap, tie -> slot 0) picked a different exit from the uncapped lane-graph rule at 82 of 351
  first junctions (23 %).
- Light features: the rail finds a stop line up to ~222 m ahead; with `lattice_light_in_view` they appear only once it is
  among the 4 nearest observed stop lines within 200 m (162 m on the measured Town10HD approach).
- Explainer: https://claude.ai/artifact/UmKvFJcedLxoYEB35dchb2 ("Where the Goals Go").

## 6. Not done (candidates for the next batch)

- Partner-conflict feature along the committed preview (collisions are flat at 0.13-0.15).
- Exact mask speedups (longitudinal cells are 42 % of `c_step`); a 0.6 s decision period would halve mask cost but adds reaction latency.
- EMERGENCY use stays at 0.10-0.15 of steps.
- Town01/02/07 cannot be overtaken in (no same-direction neighbour lanes): addressed by section 7 behind a flag, one run so far.

## 7. Borrowing the oncoming lane (commit `03889a6a`, `env.lattice_oncoming_overtake`)

Addresses issue 11: on roads with one lane each way (most of Town01/02/07) a lattice car could only queue behind a
stopped car, because its widest lateral move is +0.9 m.

| Part | Rule |
|------|------|
| Menu | A sixth lateral choice per duration, "oncoming lane": 20 -> 24 lateral cells, 89 -> 93 mask features. Off by default (old menu, old checkpoints load) |
| Where | Lane profiles record an opposite-heading lane 2.5-4.5 m to the left only when no same-direction lane runs on either side and no road edge lies between. Target = that lane's offset at the car (4.0 m on Town01) |
| Offered | Forward gear, not reversing, and the next max(40 m, 6 s x speed) of rail keeps the oncoming lane beside it with no connector (junction, light, merge) |
| Kept | Once the plan ends in the oncoming lane, keep needs max(20 m, 4 s x speed); otherwise keep is masked and the car must pick a return. Lane changes are masked while borrowing |
| Observation | Plan block +1: borrowing flag (60 -> 61) |
| Reward | While borrowing (past half the oncoming offset, facing along the rail) the lane metrics use the car's own lane: its direction, its lane index (lights, speed limit, route distance) and the distance to the borrowed lane's centre. So the wrong-way alignment penalty and the zero velocity reward of the old scoring no longer apply. `env.reward_oncoming_penalty_frac` charges frac x min(collision, offroad, stop-line coefficient) per borrowing step instead; the schema keeps (wait + oncoming) x discounted episode below one infraction (0.92 at 5e-4 + 5e-4) |
| Logs | `lattice/oncoming_rate` (share of steps borrowing), `lattice/oncoming_starts` (per agent-episode), `reward_components/oncoming` |

Measured (C tests and `harness/overtake_share.c`, default menu, dt 0.3):
- Share of lane length where the move is offered at <= 6.7 m/s (40 m window): Town01 55.9 %, Town02 42.9 %, Town07
  22.6 %, Town03 5.1 %, Town04 3.1 %, Town05 3.0 %, Town10HD 0.5 %, Town06 0 %. At 10 m/s (60 m): 49.0 / 35.3 / 13.1 / 3.8 / 1.9 / 2.6 / 0.0 / 0 %.
- Town01, stopped car 35 m ahead at 6 m/s: the 3.6/4.8/6.0 s cells are feasible (2.4 s is not); the car reaches
  d = 4.07 m, passes with no collision and returns to d = 0.000. The reward's lane stays the car's own lane on every step.
  27 borrowing steps cost 0.0135 in total. The old scoring charged about 0.0525 per borrowing step and paid no velocity
  reward (1.27 over the prototype's 7.2 s borrow); a crash at that speed costs 2.1.
- Borrowing toward a split: keep is masked 22.4 m before the junction (window 24 m at 6 m/s); the car enters the
  junction at d = -0.06 m.
- Random valid actions, 16 agents x 300 steps x 8 towns: 410 borrow starts, 2.4 % of steps borrowing, 0 invalid
  actions, bit-identical twin runs.

Run: `spline_werling_trajE_overtake_wait_pen` = `trajA_info_wait_pen` + `env.lattice_oncoming_overtake=true
env.reward_oncoming_penalty_frac=5e-4`. First submitted as 18976365 / 18976366 and cancelled
unstarted (a misread request); relaunched 2026-10-01 as 18986342 / 18986344 after `spline_werling_vel10x` was cancelled
to free a GPU. Compare with `trajA_info_wait_pen`.

Consistency variants (submitted 2026-10-02): trajE plus `env.reward_trajectory_consistency` at 10x (2e-3) and 50x (1e-2)
D's 2e-4. They run trajE's exact code: the working tree then held uncommitted checkpoint-render work, so their snapshots
were replaced with trajE's `code_v1` copy (identical to HEAD `64e271e0`) before `sbatch`. At 1e-2 per metre RMS, a hard
emergency brake (RMS 8-15 m) costs 0.08-0.15 against a 1.5 collision; keeping the committed plans costs nothing.
Compare both with trajE: `lattice/plan_change_rms_m`, lat/lon new rates, goals, collisions, oncoming rate.

What to watch: `lattice/oncoming_rate` and `oncoming_starts` should rise above the random-policy level without collisions
climbing; goals, DNF, moving fraction and time to first motion vs `trajA_info_wait_pen`; `reward_components/oncoming`.

Harnesses behind the numbers: `/scratch/ag11023/tmp/claude/ablation/harness/` (`queue.c` + `queue_policy.py`, `lanestats.c`,
`routecheck.c`, `bench.c`, `newfeat.c`, `overtake.c`, `overtake_share.c`); W&B pulls in `/scratch/ag11023/tmp/claude/ablation/`.
