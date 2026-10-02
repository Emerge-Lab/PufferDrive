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

| Run | wait | oncoming | full speed (m/s) | guard x 1000 steps | nice |
|---|---|---|---|---|---|
| `spline_werling_tune_base` | 7e-4 | 2.5e-4 | 5 | 0.95 | 1000 |
| `spline_werling_tune_control` | 5e-4 | 5e-4 | 1 | 1.00 | 1000 |
| `spline_werling_tune_wait7e-5` | 7e-5 | 2.5e-4 | 5 | 0.32 | 2000 |
| `spline_werling_tune_wait7e-3` | 7e-3 | 2.5e-4 | 5 | 7.25 (guard off) | 2000 |
| `spline_werling_tune_onc2.5e-5` | 7e-4 | 2.5e-5 | 5 | 0.73 | 3000 |
| `spline_werling_tune_onc2.5e-3` | 7e-4 | 2.5e-3 | 5 | 3.20 (guard off) | 3000 |
| `spline_werling_tune_vfull0.5` | 7e-4 | 2.5e-4 | 0.5 | 0.95 | 4000 |
| `spline_werling_tune_vfull50` | 7e-4 | 2.5e-4 | 50 | 0.95 | 4000 |

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
