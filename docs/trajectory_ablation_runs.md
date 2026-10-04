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
| `spline_werling_trajF_overtake_turnaround_wait_pen` | 19032829 / 19032830 | trajE + turning around (section 8) | trajE's + `env.lattice_turnaround=true` |

The three `*_wait_pen` runs use commit `d9f1e2ee` (waiting penalty) on top of `37157e4c` (view-gated light features,
switched off for them so they differ from A only as listed). Waiting penalty: each step a car pays
0.0005 x min(its collision, offroad, stop-line coefficient) x max(0, 1 - speed / 1 m/s), except while its reported next
light is red or yellow; a whole standstill episode costs at most 0.46 of that cheapest infraction (gamma 0.999, 2560 steps).

Cancelled 2026-10-01: `spline_werling_vel50x` (18867873 after 24 h, continuation 18867876), and before they started
`trajC_progress_mapgoals` (18953137 / 138), `trajD_consistency_mapgoals` (18953142 / 143) and the
first `trajE_overtake_wait_pen` submission (18976365 / 366). Also cancelled 2026-10-01 to free a GPU:
`spline_werling_vel10x` (18867871 after 34 h 17 m, continuation 18867872). Cancelled 2026-10-02 to free GPUs:
`spline_baseline` (18811661 after 23 h 08 m) and `spline_werling_vel5x` (continuation 18867868 after 6 h 46 m).
Also cancelled 2026-10-02 at the user's request: `spline_werling_halfpen` (18878816); the `halfpen_goal` run (18881663) kept.
`trajE_overtake_wait_pen_consist50x` (19026638 after 3 h 28 m, continuation 19026639 unstarted) was cancelled
2026-10-02 at 11:47 EDT from outside this session (sacct: CANCELLED, not a failure).
Also cancelled 2026-10-02 at 16:16 EDT from outside this session, each continuation first and then its main job:
trajA_info (18935213 / 215), trajB_routegoals (18935221 / 224), trajC_progress (18935229 / 231), trajA_info_wait_pen
(18956052 / 053) and trajC2_routegoals_wait_pen (18956054 / 055). At 16:26 EDT the three `s30bswerlgoal` runs were also cancelled
(18881653, 18881656 and the `halfpen_goal` run 18881663).

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

## 8. Turning around (`env.lattice_turnaround`)

Lattice rails follow lane links forward, so a car that wants to go back the way it came has had to drive on to a
junction and loop round. With lane changes every first map goal is reachable, but often only by a long loop: the
median route to the first map goal is 2.1-3.7x the straight-line distance (`benefit2.c`, 510 goals over 8 towns).

| Part | Rule |
|------|------|
| Menu | One more longitudinal cell after the back-ups, "turn around": 60 -> 61 longitudinal cells (+1 mask feature). Off by default (old menu, old checkpoints load) |
| Planner | Rest-to-rest legs at constant curvature, alternating gear and all rotating one way, each stopping where the box plus 0.3 m would touch a road edge (start may sit 0.15 m close). It tries both senses x both first gears x three arc radii (1.0 / 0.7 / 0.5 x 0.9 of the tightest curvature the car can steer to), re-aiming at the landing lane's heading on curves. A plan must land on another non-connector lane, within 0.3 rad of its heading and 1.5 lane widths of its centre, with no road edge between the car and that lane and a car-width corridor clear 10 m ahead. Ranking: legs, with a landing beyond half a lane counting 2 extra legs; ties go to fewer legs, then centred, then shorter |
| Offered | Stopped exactly, on a lane rail, not on a connector, not borrowing, not a phantom braker; no stop line, light or yield line within the plan's reach (2 x radius + box corner) in any direction; no vehicle within the reach or closing in on it before the turn ends; a plan exists. Cached while the car does not move |
| Execution | Each leg runs on a lane-less arc rail with d held at 0: the wheel swings at rest, then a rest-to-rest STOP/BACKUP plan at half the accel and jerk limits, at most 2 m/s. At every cusp the rest is re-planned from the actual pose (sweeps 0.3 / 0.15 / 0.08 m, either sense, no landing limits); a car the sim stops mid-turn aborts it. While turning, every factor is masked to index 0 |
| Landing | Back on the lane rail of the base-lane search with gear 1. A last leg can end up to 1.5 lanes off the landing lane's centre: the lateral plan then merges onto the centre over max(curvature-safe distance, the longest low-speed distance). Until the car is within half a lane of that centre, its lane metrics (lane index, direction, centre distance) use the landing lane, not the nearest lane, which can be the old one faced against |
| Observation | Plan block +3 after the borrowing flag: turning flag, rotation still to go / pi (clipped to [0, 1]), and the route gap (route after turning minus route ahead, / 200 m, in [-1, 1]; 1 = no gain). The gap is shown only where the turn is feasible: at rest from the offer, while moving from a probe every 10 decisions, and only when it is negative |
| Reward | While turning, the lane-align, lane-centre and reverse terms are 0; all others (collision, offroad, goal, waiting, comfort) are unchanged |
| Logs | `lattice/turn_rate` (share of steps turning), `lattice/turn_starts`, `turn_completions`, `turn_aborts` (per agent-episode) |

Measured (C tests, harnesses in `/scratch/ag11023/tmp/claude/uturn2/`, default menu, dt 0.3, training sizes and
coefficients):
- Offered share of non-connector lane length (every 4 m, `landing.c`, empty road):

  | Car | Town01 | Town02 | Town03 | Town04 | Town05 | Town06 | Town07 | Town10HD |
  |-----|--------|--------|--------|--------|--------|--------|--------|----------|
  | 2.0 x 1.5 m | 100 % | 100 % | 78 % | 9 % | 51 % | 0 % | 88 % | 84 % |
  | 3.5 x 1.6 m | 100 % | 100 % | 78 % | 10 % | 51 % | 0 % | 89 % | 85 % |
  | 4.5 x 1.8 m | 100 % | 100 % | 77 % | 9 % | 52 % | 0 % | 40 % | 85 % |
  | 5.5 x 2.5 m | 0 % | 0 % | 28 % | 0.1 % | 24 % | 0.2 % | 0 % | 41 % |

  Landings within half a lane of the landing lane's centre: 73-100 % of plans, except the 4.5 m car on Town01/02/04/07
  (4-7 %; it lands 1.85-3.7 m off with 4 legs on Town01/02, 6 on Town04/07) and the 3.5 m car on Town07 (46 %). Mean
  legs 1.0-6.0 per town and size. Wider arcs are used in up to 90 % of plans (the 3.5 m car on Town01/02).
- Route benefit (`benefit2.c`, every first map goal, turn-arounds as extra graph edges at +30 m): a turn-around is on the
  best route for 171 of 510 goals (median saving 132-274 m per town; none in Town06); goals within an episode's driving
  range rise from 354 to 390 of 510.
- Randomised execution (`sweep12.c`, 300 random placements per town, sizes L U(0.8, 7) x W U(0.8, 2.7), randomised
  c_steer / c_throttle / c_acc): 715 turns offered, 715 completed, 0 aborted, 0 offroad steps while turning, min box
  clearance to an edge 0.108 m (p1 0.135-0.270 m per town), median turn 11.4-15.9 s (p90 20.1-37.8 s), 1.4-2.6 legs on
  average. Landing |d| median 0.92-1.36 m, beyond half a lane in 143 of 715. In the 40 steps after landing (greedy
  speed, lateral plan kept) 3 of 715 cars clipped an edge (one from a centred landing) and the lane-align term went
  negative for 2, both next to near-perpendicular lanes (cos -0.16 and -0.04).
- Town01 sedan K-turn (C test): 4 legs, 27.3 s, every step clear of edges, lands 2.36 m off the landing lane, uses the
  landing override for 11 steps and is at d = 0.00 after pulling away 81 m.
- Random valid actions (C test, 16 agents x 300 steps x 8 towns): 0 invalid actions, bit-identical twin runs.
- Cost: 0.2-1.0 ms per plan (mean over every 4 m of lane); 2-7 % more time per agent-step on 64-agent random-action
  benchmarks (Town01/03/05/10HD, noise about 3 %), nearly all from offers at rest, not from the moving-car probes.
- Independent black-box suite (written by an agent that had not seen the implementation, 34 groups; kept outside the
  repo in `/scratch/ag11023/tmp/claude/uturn2/blackbox/`): 32 pass. It found two real defects that are fixed here (the
  remaining-rotation feature read 0 for turns of more than 180 degrees; landings up to 7.1 m off, now capped and
  merged). The two groups that still fail are by design: landings up to 3.9 m off the landing lane (its spec said within
  half a lane) and a 157 degree rotation onto an angled landing lane (its spec said 180 degrees). Its random rollouts:
  113 turns started, 96 completed, 0 aborted, 17 still turning when the rollout ended, 0 invalid actions.

Run: `spline_werling_trajF_overtake_turnaround_wait_pen` = `trajE_overtake_wait_pen` + `env.lattice_turnaround=true`,
same budget and settings (30B steps, 24 CPUs, 40 h + continuation). Code commit `d6119098`; submitted 2026-10-02 as
19032829 / continuation 19032830 (torch_pr_355_tandon_priority). Compare with trajE: goals per episode
and DNF first, then `lattice/turn_starts` (should rise above the random-policy level), `turn_aborts` (should stay near
0), collisions and offroad (should not climb), time to first motion and moving fraction.

## 9. Tuning the waiting, borrowing and slow-driving penalties (2026-10-02)

Seen in the trajE / trajF renders: cars queue behind slow or stopped cars instead of overtaking or turning around.

### Measured first (consist10x settings: wait 5e-4, oncoming 5e-4, waiting penalty fades out by 1 m/s)

- Masks are not the cause. Behind a stopped car the oncoming choice is offered from every gap and speed, and a borrow can
  be turned back at any step with the 4.8 / 6.0 s lane-centre cells. A borrow turned back before the car is half way into
  the oncoming lane never reads as borrowing, so it pays no oncoming penalty (C test: 1 start, 0 borrowing steps).
- The policies start 8-14 borrows per agent-episode on W&B but almost never complete one.
- Reward over a 24 s window, mean per-agent coefficients (cheapest infraction m = 0.398, `costs2.c` with Log deltas from
  `c_step`; lane-centre bias 0, the best case):

  | Choice behind the blocker | old (control) | tuned (base) |
  |---|---|---|
  | overtake the stopped car | +0.0316 | +0.0342 |
  | wait behind it | -0.0075 | -0.0139 |
  | follow at 2.5 m/s | **+0.0176** (free) | +0.0064 |
  | overtake minus follow | 0.0140 | 0.0278 |
  | overtake minus wait | 0.0391 | 0.0481 |

  With the old settings, following at 2.5 m/s paid nothing (the velocity reward is binary above 2.5 m/s and the waiting
  penalty ended at 1 m/s). The turn-around legs are capped at 2 m/s, so before the exemption below a U-turn paid about
  what standing still pays.

### Code changes (this commit)

- `env.reward_wait_full_speed_mps` (default 1.0 = the old behaviour): the waiting penalty fades out linearly up to this
  speed, so slow following is charged too.
- No waiting penalty while turning around (it charged the manoeuvre that ends the wait).
- Guard horizon: with `train.use_value_bootstrapping` truncations bootstrap from the value, so the schema now uses
  1 / (1 - gamma) = 1000 steps instead of the 2560-step discounted sum (922.8). The old trajE values sit exactly at 1.0.
- `env.reward_wait_guard` (default true): false skips that guard, only for the two x10 probes below. It is a `Drive`
  kwarg so evaluation of those checkpoints passes the same check.
- Turn-offer fix: the offer was cached by bit-exact pose, including the traffic check. A car at rest kept the offer while
  a car closed in (43.9 m away at 8 m/s, reproduced), and never got it back after a parked car left. Now the plan is
  cached by pose, and the rail state, stop lines and traffic are re-checked at every decision.
- New W&B keys (per agent-episode, like the other `lattice/` keys):
  - `lattice/oncoming_passes`: cars passed while borrowing (the car's rear ahead of the other car's front, which is still
    in the own lane); each car of a platoon counts.
  - `lattice/oncoming_collisions`: collisions while borrowing or within 10 steps after a borrow.
  - `lattice/turn_legs`: planned legs of each turn-around; legs / `turn_starts` = 1 for a U-turn, 3+ for a K-turn.
  - `lattice/turn_route_gap`: the route-gap feature at each turn start, summed; / `turn_starts` < 0 means turning
    shortened the route to the goal, 1 means no route.
  - `lattice/queued_rate`: share of steps below 1 m/s with a car ahead within 15 m (half a lane sideways, not facing
    the car), not at a reported red / yellow; `queued_frozen_rate` the part where that car is frozen after an infraction.
  - `lattice/slow_follow_rate`: share of steps at 1-5 m/s with a car ahead within 30 m.
  - `lattice/wait_exempt_rate`: share of steps that would pay the waiting penalty but a red / yellow is reported ahead.
  - Already there: `oncoming_starts`, `oncoming_rate`, `turn_starts`, `turn_completions`, `turn_aborts`, `turn_rate`.
- Cost: `c_step` takes 2.6-6.6 % longer per agent-step than at `52bdd8af` (25th percentile of 10 runs, 64 agents, random
  valid actions, Town01 / 03 / 10HD; 3.5-6.0 % with full speed 50 m/s, where every car below 50 m/s runs the counters).

### Runs (prefix `s30btune`, nice 1000-4000 so they never outrank the pending trajF or any continuation)

All: `puffer train puffer_drive_spline_werling env.lattice_exit_mode=goal env.lattice_light_in_view=false
env.lattice_oncoming_overtake=true env.reward_trajectory_consistency=2e-3 env.lattice_turnaround=true` (consist10x plus
the turn-around), from scratch. The turn-around changes the observation and action spaces, and fine-tuning would also
drop the flag (`KEYS_OF_INTEREST`). Each sweep run changes one knob of the base by x0.1 or x10.

| Run | wait | oncoming | full speed (m/s) | guard x 1000 steps | nice | main / continuation |
|---|---|---|---|---|---|---|
| `spline_werling_tune_base` | 7e-4 | 2.5e-4 | 5 | 0.95 | 1000 | 19064424 / 19064426 |
| `spline_werling_tune_control` | 5e-4 | 5e-4 | 1 | 1.00 | 1000 | 19064431 / 19064432 |
| `spline_werling_tune_wait7e-5` | 7e-5 | 2.5e-4 | 5 | 0.32 | 2000 | 19064435 / 19064436 |
| `spline_werling_tune_wait7e-3` | 7e-3 | 2.5e-4 | 5 | 7.25 (guard off) | 2000 | 19064438 / 19064439 |
| `spline_werling_tune_onc2.5e-5` | 7e-4 | 2.5e-5 | 5 | 0.73 | 3000 | 19064449 / 19064452 |
| `spline_werling_tune_onc2.5e-3` | 7e-4 | 2.5e-3 | 5 | 3.20 (guard off) | 3000 | 19064460 / 19064464 |
| `spline_werling_tune_vfull0.5` | 7e-4 | 2.5e-4 | 0.5 | 0.95 | 4000 | 19064469 / 19064472 |
| `spline_werling_tune_vfull50` | 7e-4 | 2.5e-4 | 50 | 0.95 | 4000 | 19064735 / 19064738 |

Submitted 2026-10-02 16:31 EDT from commit `2f4d4aa3` (torch_pr_355_tandon_priority); all 8 snapshots diff clean
against it.

Why the base:
- wait 7e-4 + oncoming 2.5e-4 keeps a whole stuck episode in the oncoming lane at 0.95 of one infraction.
- Oncoming 5e-5 was rejected: that makes cruising in the oncoming lane nearly free, and no traffic mask forces a return.
- Full speed 5 m/s charges slow following without charging free driving.

Predictions, recorded before any result:
- **base vs control**: lower `queued_rate` and `slow_follow_rate`, higher `oncoming_passes` per `oncoming_starts`,
  earlier first motion and more goals. Risk: more `oncoming_collisions`.
- **wait7e-5**: the most queuing of the guarded runs (waiting is almost free again).
- **wait7e-3**: fastest to move, but for small-m agents crashing out pays after ~140 stuck steps, so collisions,
  offroad and red-light running should rise.
- **onc2.5e-5**: the highest `oncoming_rate`, with long stays in the oncoming lane and more `oncoming_collisions`.
- **onc2.5e-3**: for m = 1, overtaking (-0.031 per 24 s) loses to following at 2.5 m/s (-0.010) but still beats waiting
  (-0.048). So fewer passes, slow following kept, and turn-arounds relatively more attractive.
- **vfull0.5**: like the control for slow following (free from 0.5 m/s), so the highest `slow_follow_rate`.
- **vfull50**: the waiting penalty turns into a time penalty (free driving pays 0.020 per 24 s at m = 0.398). Expect
  higher speeds, possibly more collisions and overspeed.

Limits that tuning cannot change:
- A turn-around is never offered with another car within about 19 m (a 2.7 m-wheelbase car's turn reach of 16.4 m,
  about three turning radii, plus 2.4 m for the other car), and a back-up moves at most 13.4 m. So a car stacked right behind a stopped car can only get out by borrowing or
  changing lanes.
- With `light_in_view=false` a red light reported anywhere ahead on the chain exempts the wait; `wait_exempt_rate`
  shows how often.
- The lane-centre term is a penalty for about 85 % of agents (random centre bias). It is not scaled by m, so the guard
  does not cover it.
- Passes are judged in the move stage: lower-indexed cars have already moved that step (no effect for stopped cars).
  A pass on the episode's last step is not counted. `turn_legs` counts planned legs, and re-plans at cusps can change
  that count.

## 10. Why `trajE_overtake_wait_pen_consist10x` stalls against `spline_baseline` (measured 2026-10-03)

Harnesses and raw results: `/scratch/ag11023/tmp/claude/flat/`.
- `diag_rollout2.py` replays a checkpoint for a full episode exactly as `PuffeRL.evaluate` samples: 1024 agents, the
  run's own code snapshot.
- `analyze_npz.py` rebuilds the GAE advantages and applies the trainer's filter.
- `start_cost.c` prices a start from rest.

Checkpoints used:
- consist10x epoch 6850 (17.9B steps). It replays bit-identically on its own snapshot and on `2f4d4aa3`.
- spline_baseline epoch 11150.

The replayed episodes match W&B:

| | consist10x replay | consist10x W&B | baseline replay | baseline W&B |
|---|---|---|---|---|
| goals | 0.63 | 0.6-1.1 | 6.5 | 7.4 |
| DNF | 0.58 | ~0.5 | 0.02 | 0.03 |
| speed (m/s) | 0.98 | 1.2-1.6 | 4.1 | 4.9 |

### What "flatlined" is

| At equal steps | 4B | 8B | 12B | 16B |
|---|---|---|---|---|
| consist10x goals / DNF | 0.48 / 0.48 | 0.63 / 0.48 | 0.80 / 0.48 | 0.96 / 0.47 |
| spline_baseline goals / DNF | 0.55 / 0.42 | 1.11 / 0.29 | 1.43 / 0.23 | 2.41 / 0.12 |

- consist10x still improves slowly (1.16 goals at 17.9B). It is not collapsing.
- Every map-goal lattice run plateaus the same way: v1, goal, trajA, trajE, consist10x and v2_h100 sit at 0.6-1.3
  goals with DNF 0.4-0.6.
- The route-goal lattice runs do not: trajB / C / D reach 3.5 / 5.9 / 5.3 goals at 12B with DNF 0.07-0.08.
- So the lattice can learn to drive. It stalls on the baseline's task, map goals.

### What the cars do (replay of consist10x)

- **Stationary:** 74 % of live decisions are made at rest (< 0.3 m/s).
- **Masks are not the cause:** a cell of 2.5 m/s or faster is valid at 99.98 % of those decisions.
- **What the policy picks:** half the time it keeps its plan. When it picks a new speed plan at rest, it chooses
  0 m/s 87 % of the time, EMERGENCY 12 % and any moving cell 0.9 %.
- **Where the time goes** (share of all steps):
  - queued behind a car within 15 m: 25.4 %, of which 4.4 % behind a car frozen after a crash for the rest of the episode;
  - slow while a red / yellow light is reported somewhere ahead, which exempts the wait penalty: 22.1 %;
  - slow-following: 2.8 %.
- **Overtakes:** 23.5 borrow starts per agent-episode, 0 borrowing steps, 0 passes.
- **Gates:** keep and new are a coin flip at every 0.3 s decision (lat gate entropy 0.52 of 0.69, lon gate 0.65 of
  0.69, p(new) 0.45 / 0.50).

### Why the policy never learns to start: the advantage filter

`train.adv_filter_threshold_scale=0.01` drops every sample with |advantage| < 1 % of the EWMA batch-max |advantage|.
That max is set by rare crash events (4.6), so the threshold is 0.046. The value loss uses only the kept samples too.

| Live steps (frozen cars excluded) | consist10x | spline_baseline |
|---|---|---|
| share of steps at rest | 74 % | 11 % |
| median abs advantage at rest / moving | 0.0069 / 0.027 | 0.068 / 0.082 |
| kept at rest / moving | **2.7 %** / 33 % | 29 % / 36 % |
| kept overall | 10.6 % | 35 % |

- The decision to pull away is almost never in the gradient, so the stop habit is never corrected.
- This matches W&B: `kept_fraction` stays at 0.11-0.16 for every map-goal lattice run, against 0.2-0.32 for the
  route-goal runs and up to 0.44 for the baseline.
- Why lattice advantages are small:
  - with about 0.9 goals per episode the value function sees almost no big rewards (explained variance 0.99);
  - the dense terms are tiny, about 7.5e-4 per step;
  - so most advantages are 0.005-0.03.
- Why the baseline keeps more: it collects 6-7 goals per episode, so its returns and value errors are 10x larger.
- The trap is a loop: few goals, then tiny advantages, then filtered samples, then no learning to move, then few goals.

### Why the map-goal task stays hard for the lattice

- Goal-mode exits (`lattice_goal_exit_slot`) and the route features use `lane_graph.distances`, which follows lane
  links only.
- Following links from the spawn lane, 14-35 of ~64 first goals per town are unreachable (Town04 / 05 / 06 / 10HD,
  section 1, item 2); with lane changes, 0 of 510 are.
- For an unreachable goal, every exit is INFINITY, so the car takes exit 0, the straightest branch.
- Measured effect (section 5): goal mode picks a different exit from the lane-change-aware shortest route at 58 of 351
  first junctions (17 %) with map goals and 11 of 354 with route goals.
- The route features and the progress reward read 167 of 234 lane changes on offer as a >= 100 m detour or a dead
  end, although the true route changes by < 30 m.
- Correction (same day): the per-lane goal-distance columns use the same table for every policy. Moving into the right
  lane drops the observed distance (median 409 m over the 192 of 510 spawns where a lane change shortens the route;
  96 more read unreachable until the change), so 'take the lane with the smaller number' is a clear local hint for
  both policies. Town01 / 02 / 07 have no same-direction neighbour lanes, so there routes are long because the lane
  graph has no U-turns.
- The baseline follows lanes too. In a replay of 2.0M moving agent-steps it is lane-aligned (cos > 0.5) 96.1 % of the
  time, across a lane or off any lane 3.3 %, wrong way 0.6 %, more than 2 m off centre 0.6 %, and reversing 3.3 %.
  The reversing and crossing are consistent with occasional turn-arounds, which the lattice lacked before trajF.
- So links-only routing explains only part of the gap: goal-mode exit picks (17 % of first junctions) and the progress
  reward (off in consist10x). Runs where the policy picks exits on map goals are no faster (v2_h100: 0.60 goals at 12B,
  1.25 at 25B).

### Other findings

- **Consistency penalty at a start:** a start from rest costs 0.003-0.0065 at the 2e-3 coefficient (`start_cost.c`).
  Going is still +0.026 better than staying over 12 s.
  - Time to first motion is 52 s against 9 s for trajE, the same run without the penalty, and consist50x froze completely.
  - But consist10x reaches more goals than trajE at equal steps (0.96 vs 0.76 at 16B) with fewer collisions, so the
    penalty helps once moving.
- **Red-light wait exemption:** with `light_in_view=false` any red / yellow reported ahead on the chain switches the
  wait penalty off, at any distance. That covers 22 % of all steps.
- **Wrecks stay in the lane:** `collision_behavior=stop` leaves crashed cars in place for the rest of the 2560-step
  episode, and with no completed overtakes the cars behind them never move again.
- **Dead run:** `trajD2_mapgoals_wait_pen` (18956057) went NaN between 12.0B and 12.5B and kept running until it finished.
  - Its entropy, KL and value loss are NaN, it keeps 0 % of samples, gets 0.015 goals and collides at 0.34.
  - `vel5x_goal` shows the same signature.
  - Nothing aborts a run on a non-finite loss.
  - Its main job timed out at 03:35 and the continuation (18956058) ran to completion at 21:09 on 2026-10-03.

### What should help (ranked by expected impact / cost)

1. **Config: stop the filter starving rest decisions.** `train.adv_filter_threshold_scale=1e-3` keeps about 70 % of
   samples, or use `train.adv_filter_enabled=false`.
   - Cost: the learn phase (about 22 % of wall time) grows with the kept share, so SPS may fall 30-50 %.
   - Better long-term: a threshold from a quantile of |adv| rather than the max.
2. **Turn-arounds on map goals:** trajF is running. Town01 / 02 / 07 routes are long because there are no U-turns,
   and the baseline reverses or crosses lanes about 3 % of the time it moves.
3. **Code, lower priority: a lane-change-aware route distance** for goal-mode exits, route features and
   `reward_route_progress`. It matters for goal-mode exits and before the progress reward is turned on for map
   goals; the lane columns already point to the right lane.
   - At 12B, route goals plus progress reach 5.9 goals against 3.5 without progress (trajC vs trajB).
4. **Code, small: exempt starts from rest from the consistency penalty** (old plan at a standstill).
5. **Code, small: limit the red-light wait exemption to a stop line within braking reach.**
6. **Code, small: fail fast on a non-finite loss**, and find why trajD2 and vel5x_goal went NaN.
7. **Revisit `ent_coef`** after item 1. The gates sit at maximum entropy, so it may be too high for the factored lattice.

### What the queued runs can and cannot show

- **s30btune:** wait, oncoming and full-speed shaping.
  - The base wait term is about 2.8e-4 per step, so its advantages stay below the filter threshold.
  - Only `wait7e-3` (about 2.8e-3 per step) clears it, so expect the guarded sweep to look like the control.
  - None of them change the filter or the routing.
- **trajF:** turn-arounds shorten some map-goal routes. At 4B it looks like trajE (0.37 goals) with a 2.5 s first motion.
- **v2_h100 (50B):** a map-goal control for a longer horizon. It had 1.25 goals at 25B.
- **Gap:** no queued run tests items 1, 3, 4 or 5.

## 11. Fixes for the stall, and the 20B run (2026-10-04)

Requested by the user after section 10: map goals stay, exits are chosen by the policy and can be changed, the car
gets no light information it could not see, the advantage filter is relaxed, turn-arounds are on, and the smaller
items are fixed. Lane-change-aware routing is dropped for now.

Code (one commit), after an independent code review (`/scratch/ag11023/tmp/claude/review_fix/`):
- **Exits** (`lattice_exit_mode=policy`):
  - The nearest split within 160 m stays live until the car's centre passes it.
  - The exit is part of a new lateral plan: it is applied on decisions whose lateral gate is new and is not a lane
    change. It counts in the log-prob and entropy with the lateral gate, like the cells, so keep decisions never flip
    the route. A switch therefore goes through a new lateral plan, which the consistency term prices.
  - The exit already on the chain stays offered; feasibility applies only to switching. The review found the old
    every-decision check revoked a committed turn in 181 of 194 runs at mask-allowed speeds; 0 now.
  - A switch must leave room to brake at 1.5 m/s^2 (`LATTICE_EXIT_SWITCH_BRAKE_MPS2`) after one decision period of
    coasting. Late switches the mask allows (6-30 m, 4-13 m/s): emergencies 7 of 1386, none over the speed envelope
    by > 0.5 m/s. At 2.5 m/s^2 it was 23 of 1692, with 7 over (worst 3.05 m/s).
  - Lane changes are no longer masked while an exit is live. New counter `lattice/exit_switches`. Goal mode is
    unchanged (bit-identical trajectories in the review). `late_exit_pending` clears once no undecided split is near.
  - Cost with random actions: `c_step` +10-12 % per agent-step (330-398 switches per 64 agents x 400 steps).
- **Lights** (`env.obs_light_facing_only`, new, default false; the run sets it true together with
  `lattice_light_in_view=true`):
  - A light's state is observed only when the nearest point of its stop line is ahead of the car and the car is
    within 60 deg of the way its face points (against its first controlled lane, a junction connector). A degenerate
    lane hides it.
  - Visible lights rank before hidden ones in the 4 traffic-control slots and in the in-view test. The review found
    the own light outside the slots for 20-43 % of cars 30-80 m back; ranking visible first puts it in for 98-100 %.
  - This applies to the slots and to the plan block's red / yellow flags, which also drive the wait exemption.
  - The stop-line cell needs the stop line in view. Controlled-lane ids are validated at init when the flag is on.
  - Partner `seconds_stopped` (an audit leak unrelated to lights) is unchanged.
- **Wait exemption:** a reported red / yellow waives the waiting penalty only within 30 m of its stop line or when
  queued behind a car within 15 m. Before, any reported red anywhere ahead waived it (22 % of steps in the replay).
- **Consistency:** a plan committed below 0.5 m/s is a start, not a plan change, and costs nothing. The RMS metric
  still records it.
- **NaN root cause:**
  - `_train_ppo_transition` kept the remainder minibatch; with exactly k * 65536 + 1 kept samples it held one sample,
    whose advantage std (Bessel) is NaN.
  - W&B confirms both dead runs: trajD2 at epoch 4670 with 327681 = 5 * 65536 + 1 kept, and vel5x_goal at epoch 3648
    with 393217 = 6 * 65536 + 1.
  - Minibatches are now near-equal chunks (`minibatch_chunks`), and `_ppo_loss` raises on a non-finite loss.
- `obs_light_facing_only` and `lattice_light_in_view` join the fine-tune keys copied from a checkpoint's config.
- `scripts/launch_fairshare.py` (untracked) gained `--total-timesteps`.

Hyperparameters, reviewed for balance (`/scratch/ag11023/tmp/claude/review_hp/`):
- **`train.adv_filter_threshold_scale=1e-3`:**
  - In the replay it keeps 71 % of live samples and 64 % of samples at rest, against 10.8 % / 2.8 % at 0.01.
  - 3e-3 keeps rest samples at only 0.31x the moving rate.
  - Turning the filter off adds samples with no signal and about 18 optimizer steps per epoch.
- **`train.update_epochs=2`:** about 46 optimizer steps per epoch, against the baseline's 45 (44 % kept x 3 epochs).
- **`train.learning_rate=4e-4`:** per optimizer step the lattice's approx_kl is about 1.5x the baseline's
  (0.0011-0.0015 vs 0.00086). At 5e-4 with 46 steps the peak KL would reach about 0.05-0.06; at 4e-4 about
  0.03-0.04, like trajB / C / D.
- **`ent_coef=0.01` kept:**
  - Normalised advantages average about 0.39 among kept samples, so entropy is about 2.6 % of a typical sample's
    policy-gradient weight (8 % at rest).
  - Stop rule: drop to 0.005 if entropy keeps rising above about 3.5 nats after 3B steps while goals stay flat.
- **Rewards:** wait 7e-4, oncoming 2.5e-4, full speed 5 m/s, consistency 2e-3 (guard 0.95). The lane-centre term is
  not covered by the guard.
- **20B steps, cosine to zero.** Read whether the stall is fixed at 4-8B: kept fraction at rest, time to first
  motion, goals against consist10x and the baseline.
- **Expected time:** about 55-61 h on H100 and 73-80 h on A100, i.e. one main job plus one continuation.

Run: `spline_werling_fix20b` (prefix `s20bfix`, 20B steps), main 19142026 / continuation 19142027, submitted 2026-10-04
from commit `8449f8e2`; the snapshot diffs clean against it. Overrides: `env.goal_source=map
env.lattice_exit_mode=policy env.lattice_light_in_view=true env.obs_light_facing_only=true
env.lattice_oncoming_overtake=true env.lattice_turnaround=true env.reward_trajectory_consistency=2e-3
env.reward_wait_penalty_frac=7e-4 env.reward_oncoming_penalty_frac=2.5e-4 env.reward_wait_full_speed_mps=5.0
train.adv_filter_threshold_scale=1e-3 train.update_epochs=2 train.learning_rate=4e-4`.

Cancelled 2026-10-04 at the user's request:
- the running continuations of `trajE_overtake_wait_pen` (18986344) and `trajD_consistency` (18935236);
- every queued continuation: consist10x 19026615, trajF 19032830, the v2_h100 chain 19053277 / 278 / 280, and the
  eight s30btune continuations 19064426 / 432 / 436 / 439 / 452 / 464 / 472 / 738.

The s30btune main jobs and trajF's main job were kept.

What to read first, at 4-8B against consist10x and the baseline:
- `losses/kept_fraction` (expect about 0.7);
- `lattice/time_to_first_motion_s` and `moving_fraction`;
- `lattice/queued_rate`;
- goals and DNF;
- `losses/approx_kl` (expect at most about 0.04);
- `lattice/exit_switches`.

Cancelled 2026-10-04 08:20 EDT at the user's request, because they ran the pre-fix code:
- the seven pending s30btune main jobs: control 19064431, wait7e-5 19064435, wait7e-3 19064438, onc2.5e-5 19064449,
  onc2.5e-3 19064460, vfull0.5 19064469, vfull50 19064735;
- their continuations had already been cancelled.

`spline_werling_tune_base` (19064424, running) was kept.

## 12. Committing to an overtake (`env.lattice_overtake_commit`, 2026-10-04)

Why: in fix20b no borrow ever finished an overtake (`lattice/oncoming_passes` 0 at 1.43B). The lateral gate picks a
new plan on about 40 % of decisions, and passing a stopped car takes about 35 decisions, so a borrow survives
start-to-pass with probability about 0.6^35 = 4e-6. Turn-arounds, which are committed manoeuvres, complete 95 %.

What it does (default false; needs `lattice_oncoming_overtake`):
- **Start:** a new lateral plan into the oncoming lane commits when a car heading the same way (within 60 deg of the
  rail, `LATTICE_PASS_MIN_COS`) is ahead in the own lane within 60 m and nothing in the oncoming lane is already a
  threat. That car is the target.
- **While committed:** the lateral gate's "new" is masked, so the borrow keeps its plan. It is unmasked whenever
  `must_return` holds (the oncoming lane ends or stops being clear ahead), so the forced return always works.
  Longitudinal control and the exit slot's committed exit are unchanged.
- **Release**, checked every step; each release is counted once:
  - `overtake_completions`: the target is passed (its front behind the car's rear).
  - `overtake_yields`: a car ahead in the oncoming lane would be reached within 5 s at the current closing speed, or a
    standing (< 1 m/s along the rail) or oncoming car is within 20 m.
  - `overtake_abandons`: the lateral plan left the oncoming lane (a stop request before the car is fully across
    re-targets the lateral plan), or 15 s passed while the car's front has not reached the target's rear.
  - Not counted: the target is removed or leaves the lane.
- The pass counter (`oncoming_passes`) uses the same target rule. Before, it also counted cars driving head-on in the
  own lane, which the review found to be 85 % of counted passes in random rollouts. It is not comparable with runs
  before this commit.

Independent review (`/scratch/ag11023/tmp/claude/review_commit/`; recheck `/scratch/ag11023/tmp/claude/recheck/`):
- **Found, fixed and tested:**
  - Targets driving head-on: 299 of 661 commitments before, 0 now.
  - Stopped in the oncoming lane short of a parked car with "new" masked for up to 11 s: fixed by the 20 m standing rule.
  - 20 % of commitments released in the step they started: fixed by requiring no threat at the start (161 down to 6).
  - The 15 s cap firing mid-pass and turning the car into its target: no cap while the car is level with the target.
    Longitudinal control (pass, or fall back) always ends such a commitment.
- **No regressions found:** no invalid actions, empty masks or determinism breaks. Threat detection found 1755 of
  1757 oncoming cars placed 10-100 m ahead (curves included), with no false positive for a same-way car.
- **Collisions with random actions** (8 towns x 2 seeds x 600 steps x 24 cars):
  - while committed: 16 (1 head-on) uniform and 9 (0) borrow-happy, against 32 (15) and 45 (27) before the fixes;
  - all collisions per 1k agent-steps, off vs on: 3.13 vs 3.27 uniform, 3.52 vs 3.46 borrow-happy. Committed
    borrows last longer, so more collisions happen while borrowing (49 vs 103 uniform).
- **Known and accepted:**
  - A faster same-way car coming from behind in the oncoming lane is not a threat; it is the pre-existing borrowing
    blind spot.
  - While committed the exit slot is not read. `must_return` forces a decision before the split, but at 14 m/s or
    more a turning exit may no longer be feasible by then.
  - A commitment level with a target that matches its speed holds until longitudinal control changes, a threat
    appears or `must_return`.
  - Cost: one threat check is about 1.5 % of a `c_step` per committed car.

Tests (`tests/drive/test_drive_spline_werling.c`):
- held until passed against a turn-back-every-decision policy (23 steps, no collision; without the flag: hits the car);
- yields to an oncoming car after 2 steps;
- no commitment toward an oncoming car 50 m ahead, or with nothing ahead;
- yields 19.6 m short of a parked oncoming-lane car at 2.7 m/s;
- standing vs same-way car within 20 m;
- a slow car: held 18.3 s until passed, and abandoned at 14.7 s after falling back;
- a head-on car in the own lane is neither target nor pass;
- reset clears; init rejects bad values;
- random rollouts in 8 towns, off and on.

Runs (prefix `s20bcommit`, 20B steps, submitted 2026-10-04 from commit `9bace7a3`; every snapshot matches it).
Each run's resolved config equals fix20b's plus `env.lattice_overtake_commit=true`, except for the one listed change
(checked key by key, 248 keys). All are at nice 1000 so they do not outrank fix20b's continuation.

| run | change from fix20b + commit | main | continuation |
|---|---|---|---|
| `spline_werling_fix20b_commit` | none (the direct comparison with fix20b) | 19160082 | 19160083 |
| `spline_werling_fix20b_commit_nofilter` | `train.adv_filter_enabled=false` | 19160086 | 19160087 |
| `spline_werling_fix20b_commit_filter3e-3` | `train.adv_filter_threshold_scale=3e-3` | 19160089 | 19160090 |
| `spline_werling_fix20b_commit_lr5e-4` | `train.learning_rate=5e-4` | 19160091 | 19160095 |
| `spline_werling_fix20b_commit_ep3` | `train.update_epochs=3` | 19160124 | 19160125 |
| `spline_werling_fix20b_commit_ent5e-3` | `train.ent_coef=0.005` | 19160127 | 19160128 |
| `spline_werling_fix20b_commit_noconsist` | `env.reward_trajectory_consistency=0` | 19160135 | 19160136 |

What to read first:
- `lattice/overtake_completions` per commitment (fix20b had no passes at all);
- yields and abandons;
- `lattice/oncoming_steps`;
- `lattice/oncoming_collisions`;
- goals and speed against fix20b at the same step count.

## 13. A speed bonus (`env.reward_speed_bonus`, 2026-10-04)

Why: every lattice run cruises at exactly 5 m/s, in renders of fix20b (2.6B) and consist10x (5.2B, 7.6B):
- 53-75 % of moving time is at 4.5-5.5 m/s;
- the policy puts 60-76 % of its target-speed probability on the 5 m/s cell and about 1 % on 7.5 m/s, although 7.5 m/s is offered on 85 % of moving frames;
- the flat velocity reward pays the same at any speed above 2.5 m/s, and 5 m/s is the first menu speed above that;
- spline_baseline, which picks continuous speeds and has no such plateau, sped up with training: 74 % of its cars exceed 10 m/s at 25.6B.

Rejected options:
- a reward normalised by the lane speed limit: the user does not want a lane-limit-bound bonus;
- per-metre route progress: its total per trip is fixed, so it does not pay for speed (trajC vs trajB moved at about the same speed);
- the user picked a speed bonus measured against a fixed top speed, on top of today's flat bonus.

What it does (default off, so existing runs are unchanged):
- Per step: `reward_speed_bonus x dt x max(lane cos, 0) x min(max(v - from, 0) / (base_max_speed_mps - from), 1)`, with `from` = `env.reward_speed_bonus_from_mps`.
- No pay while reversing, facing the wrong way, off the lane, or on turn-around legs.
- Lane speed limits bind only through the existing overspeed penalty.
- Logged as `reward_components/speed_bonus`.
- `binding.c` now also rejects `base_max_speed_mps <= 0` and a `from` outside [0, base_max_speed_mps).

Why count from 5 m/s (review finding): with `from = 0`, a car at today's 5 m/s would earn up to 60 % more for just moving. That makes stopping at lights, queues and turn-arounds relatively costlier, which the waiting-penalty exemptions deliberately avoid. From 5 m/s:
- below 5 m/s every incentive is identical to the base run, and the waiting penalty keeps covering 0-5 m/s;
- the bonus covers 5-20 m/s.

Tuning (measured; `/scratch/ag11023/tmp/claude/speedtune/`, reviewer `/scratch/ag11023/tmp/claude/review_speed/`):
- The landscape harness: 64 scripted lattice cars x 8 towns x 2500 steps with training rewards (randomised weights, lights, map goals, wait, consistency, oncoming). Cars obey the current lane's limit and slow for curves; collisions are not modelled.
- Its per-step reward excluding crashes, off-road and goals: at weight 0 the best target is 5 m/s, which matches training.
- Counted from 5 m/s:

  | target V (m/s) | w = 3e-3 | 4e-3 | 4.5e-3 | 5e-3 | 8e-3 |
  |---|---|---|---|---|---|
  | 5 | 1.31e-4 | 1.31e-4 | 1.31e-4 | 1.31e-4 | 1.32e-4 |
  | 7.5 | 1.34e-4 | 1.70e-4 | 1.89e-4 | 2.07e-4 | 3.16e-4 |
  | 10 | 1.08e-4 | 1.68e-4 | 1.98e-4 | 2.28e-4 | 4.08e-4 |
  | 12.5 | 0.18e-4 | 0.93e-4 | 1.31e-4 | 1.68e-4 | 3.93e-4 |
  | 15 | 0.22e-4 | 1.02e-4 | 1.42e-4 | 1.83e-4 | 4.23e-4 |
  | 20 | -0.07e-4 | 0.75e-4 | 1.17e-4 | 1.58e-4 | 4.06e-4 |

- **Chosen weight 4.5e-3.**
  - 7.5-10 m/s beats 5 m/s by 44-51 %, while 12.5-20 m/s stays at or below the 5 m/s level, which leaves room for crash risk.
  - 3e-3 only ties with 5 m/s. From 8e-3 up, 10-20 m/s is flat, so speed would be set by crash risk alone.
- **Pay per extra m/s:** 9e-5 per step, about 1.6x the waiting penalty's relief below 5 m/s (5.6e-5 for the mean car).
- **Payback:** a +2.5 m/s speed-up recovers its plan-change cost (5.1 / 3.3 / 2.1 / 1.5e-3 for 2.4 / 3.6 / 4.8 / 6.0 s plans) in 6.8 / 4.3 / 2.8 / 2.0 s.
- **Scale:** about +0.08 per episode at a 7.5 m/s cruise and +0.17 at 10 m/s. Compare fix20b's goals +0.52, consistency -0.17, lane centre -0.18 and collisions -0.23.
- **Overspeed:** above limit + 2 m/s it pays only for cars whose overspeed weight is below 6e-4 (0.06 %). Comfort (lateral acceleration above 3 m/s^2 at speed) outweighs the bonus.
- **Only the bonus changes;** everything else stays at the base run's values, for a clean A/B.

Watch items from the review:
- **Grace band:** the bonus pays up to the overspeed threshold (limit + 2 m/s). On Town01/02 (11.2 m/s limits), 12.5 m/s cruising scores 0.59 on the eval's speed-limit compliance, which has no tolerance. This was left unbounded by choice. A stop-line plan issued far from the line can also accelerate before braking, which crosses limit + 2 occasionally.
- **Overtaking moving traffic now pays:** passing a 5 m/s car to drive at 10 m/s earns 4.5e-4 per step, against the oncoming penalty of at most 2.5e-4. Watch `oncoming_starts`, `overtake_commits` and `oncoming_collisions`.
- **Final-goal arrival-speed gate:** watch goals forfeited at the final waypoint.

Tests:
- C: the formula with and without the start point, alignment, reversing, the clamp, and the turn-around gate (`test_drive_observations_rewards.c`).
- C: a lattice car at 10 m/s earns 4.0e-4 per step counted from 5 m/s (`test_drive_spline_werling.c`).
- Python: config bounds.
- Python rollout, weight 0 vs on: identical driving, rewards differ by exactly the formula, and the episode log reports it (`tests/smoke_tests/test_drive_speed_bonus.py`).
- The reviewer's 12 deliberately broken variants are all caught by the C test.
