# Submission experiments checklist

Last verified: 2026-09-14. Main result: `farther_goals`, branch `wawa/adversarial_3.0`.

Maintain this document after submissions, completions and downloads. Mark an evaluation done only when its results and resolved config have been inspected; a completed training is not an evaluated model. Five 20B CARLA matrix jobs submitted on 2026-09-14; results pending.

## Decisions and scope

- [x] Keep `farther_goals` as the main experiment family.
- [x] Prioritize numbers: no new rendering or observation capture.
- [ ] Decide whether 20B can replace 100B after checking genuine failures, traffic goals reached, and cross-target specialization for all five baselines.
- [ ] Confirm comparison scope: proposed IDM traffic and ordinary no-conditioning policy traffic against each of the five targets, on both CARLA and WOMD.
- [x] Allocate GPUs for the 20B CARLA matrix: five NOA single-GPU nodes, reservation requested. Resources for other evaluations remain unassigned.

`ignore_follower`, map-goal and at-fault experiments remain separate ablations, not replacements for the main results. The strongest-planner integration was declined; no further work planned on it.

## Verified 20B models

All five training jobs succeeded. Each final checkpoint and config exists at `gs://puffer-drive/experiments/<run>/`. Read configs confirm 20,000,000,000 steps, training seed 29, environment seed 42, route goals 25–75m, outer spawn radius 35m, guaranteed-near radius 10m, unavoidable reward -0.1.

| Target | Run | NOA training job | Final weights | CARLA diagonal, 10k | Cross-eval |
|---|---|---|---|---|---|
| PDM | `pdm_farther_goals_20B` | `2638381573660999680` | Verified | Downloaded | Submitted |
| IDM | `idm_farther_goals_20B` | `5079244610765586432` | Verified | Submitted | Submitted |
| Corridor IDM | `corridor_idm_farther_goals_20B` | `7385087619979280384` | Verified | Submitted | Submitted |
| Conditioning | `conditionning_farther_goals_20B` | `2377084834343288832` | Verified | Submitted | Submitted |
| No conditioning | `no_conditionning_farther_goals_20B` | `1620480096945045504` | Verified | Submitted | Submitted |

Weights are `final_model.pt`; configs are `config.yaml`. The learned targets are respectively `experiments/3_0_conditionning_target.pt` and `experiments/3_0_no_conditionning_target.pt`, not their adversarially trained counterparts.

Only PDM's 20B evaluation was found locally and under the five runs' usual remote `eval/` directories. The other four models' desired evaluation characteristics are therefore NOT established.

## Existing results and canonical sources

- [x] 100B CARLA diagonals: five targets, 10k episodes each.
- [x] 100B CARLA cross-eval: all 20 off-diagonal cells, 10k episodes each.
- [x] Corrected conditioning and no-conditioning 100B diagonals downloaded.
- [x] 100B LARGE diagonals: five targets, 10k episodes each.
- [x] 100B WOMD diagonals: five targets, 10k episodes each.
- [x] PDM budget evaluations: 10B, 20B, 50B and 100B, 10k CARLA episodes each.

Local source directories:

- [Initial diagonals and PDM budget sweep](failure_runs/farther_goals_evals_20260912/).
- [100B cross-evals and corrected learned-target diagonals](failure_runs/farther_goals_cross_eval_20260912/).
- [100B LARGE and WOMD diagonals](failure_runs/farther_goals_large_womd_20260913/).

Use each evaluation's top-level `evaluation_summary.json`, `episode_metrics.csv`, and `resolved_benchmark.yaml`. Do not count duplicate summaries under `failures/` as additional experiments or average only selected failures.

**Obsolete results:** exclude the initial `conditionning/` and `no_conditionning/` CARLA diagonals in `farther_goals_evals_20260912`. Those preceded the target-policy routing fix. Use `adversarial_carla_cross_conditionning_10k` under the conditioning cross-eval directory and `adversarial_carla_cross_no_conditionning_10k` under the no-conditioning directory instead.

### Current CARLA reference numbers

Rates below are percentages of all evaluated episodes, not percentages conditional on collision. Goals and score are means from the summary. Each row has 10,000 episodes.

| Adversary training target / budget | SDC collision % | Genuine % | Forced % | Unavoidable % | SDC at-fault % | Traffic goals | Traffic score |
|---|---:|---:|---:|---:|---:|---:|---:|
| PDM 100B, seed 30 | 85.33 | 63.29 | 14.55 | 7.49 | 3.40 | 5.506 | 0.2905 |
| IDM 100B | 93.15 | 59.67 | 24.53 | 8.95 | 44.33 | 2.867 | 0.1911 |
| Corridor IDM 100B | 92.98 | 70.69 | 11.19 | 11.10 | 28.02 | 2.267 | 0.1711 |
| Conditioning 100B, corrected | 90.93 | 76.98 | 9.42 | 4.55 | 36.08 | 2.282 | 0.1778 |
| No conditioning 100B, corrected | 89.78 | 74.77 | 10.42 | 4.60 | 31.35 | 2.672 | 0.1957 |
| PDM 20B, seed 29 | 82.98 | 58.79 | 16.11 | 8.08 | 2.20 | 6.765 | 0.3210 |

The assembled 100B genuine-failure matrix has its row maximum on the diagonal in all five rows. This is not a claim that every total-collision-rate row is diagonally maximal. PDM 20B is promising on its diagonal, but has no cross-eval yet. PDM's 20B/100B comparison also changes seed (29 versus 30); do not attribute the entire difference to budget.

## P0 — Complete the 20B CARLA matrix

- [ ] Freeze evaluation version and settings against the existing 100B resolved configs; record image/commit and exact target/adversary checkpoints.
- [x] Submit the 24 evaluations through `mlops/eval.sh` with sequential `--next` groups. Hydra composition, command routing, checkpoint availability and GPU/reservation manifests checked before submission; no local inference test performed this launch.
- [ ] Preflight controller routing and checkpoint loading for all five targets using the existing evaluation launcher.
- [ ] Evaluate each adversarial policy against the other four targets: 20 off-diagonal evaluations × 10k episodes.
- [ ] Evaluate the four missing diagonals: IDM, corridor IDM, conditioning, no conditioning; 10k episodes each.
- [ ] Reuse PDM's existing 20B diagonal only if the frozen protocol matches; otherwise rerun it explicitly.
- [ ] Download and inspect all results/configs, with rendering and observation capture disabled.
- [ ] Assemble 5×5 matrices for total collisions, genuine, forced, unavoidable and at-fault rates.
- [ ] Check row-wise diagonal rank and gap to the strongest off-diagonal genuine-failure rate; compare with 100B.
- [ ] Compare all five 20B/100B diagonals, including traffic and SDC goals reached and traffic score.
- [ ] Decide the primary training budget with the user; do not silently substitute 20B for 100B.

Submitted work: 24 evaluations / 240k episodes, five sequential-per-adversary jobs (four evaluations for PDM, five for each other model).

### Submitted CARLA matrix jobs — 2026-09-14

Project `valeo-cp2879-dev`, region `europe-west4`; each job uses one RTX PRO 6000 GPU with `ANY_RESERVATION`. Submitted successfully; completion not yet checked.

| Job name | Adversary baseline | Evaluations | Vertex job ID |
|---|---|---:|---|
| `xm20-pdm` | PDM | 4 | `8519326223006957568` |
| `xm20-idm` | IDM | 5 | `4767827733407334400` |
| `xm20-cidm` | Corridor IDM | 5 | `8679204009778610176` |
| `xm20-cond` | Conditioning | 5 | `3891877605883772928` |
| `xm20-nc` | No conditioning | 5 | `3753391917342130176` |

Reused original corrected 100B matrix image, without rebuild:
`europe-west4-docker.pkg.dev/valeo-cp2879-dev/driving-policy/puffer-drive-dev/puffer-drive:cross-eval-c563dd12`.

Benchmark `adversarial_carla`, 10,000 episodes per cell, rendering/observation capture disabled. Original cross-eval arguments retained except adversary checkpoint and output name. Explicit target controller/path per cell; controller targets use `train.target_policy=null`. PDM diagonal reused from the previous evaluation. Evaluation W&B disabled as in the original matrix; JSON/CSV and resolved configs go to GCS.

Expected result paths:
`gs://puffer-drive/experiments/<baseline>_farther_goals_20B/eval/adversarial_carla_cross20_<target>_10k/<timestamp>/`.

- [ ] Confirm all five jobs completed and all 24 result summaries contain 10,000 episodes.

## P1 — Ordinary-traffic comparison on CARLA

Proposed design, pending scope confirmation: keep each target fixed; replace adversarial traffic with (a) IDM and (b) the ordinary no-conditioning driving policy. Five targets × two traffic controllers = ten evaluations, 10k episodes each. Reuse matching adversarial diagonals as the third traffic condition.

- [ ] Confirm all five targets should be included.
- [ ] Validate ordinary no-conditioning traffic observation compatibility and checkpoint/config loading; it is not an adversarial policy and must not inherit adversarial-only input features.
- [ ] Verify controlled IDM traffic remains in SDC/traffic metrics and collision classification, with identical episode-end rules.
- [ ] Keep maps, scene seeds, population, spawning, goals, dynamics and termination identical to the adversarial evaluation; change only traffic controller/policy and required observation adaptation.
- [ ] Run the ten evaluations, no rendering/observation capture.
- [ ] Download results and compare absolute collision rates and percentage-point increases over ordinary traffic, including genuine failure rates.

The controller changes trajectories. Verify matched initial scenes; a shared seed alone is not evidence that all scene initializations match.

## P2 — WOMD comparison, 10k scenes

- [x] Existing five 100B adversarial diagonals downloaded: each summary reports 10k episodes; resolved WOMD configs use `num_maps: 10000`.
- [ ] Freeze the same explicit 10k WOMD scene selection for every comparison; do not accidentally evaluate the full corpus through `num_maps=-1`.
- [ ] If selecting 20B as primary, evaluate its five diagonals on those same 10k scenes; currently missing.
- [ ] Run IDM-traffic and ordinary no-conditioning-traffic controls against the five targets on the same scenes (ten evaluations, subject to scope confirmation).
- [ ] Compare collision rates/types, SDC at-fault, goals and traffic score; retain episode-level data and report classification coverage.

No WOMD cross-target matrix requested. Existing 100B LARGE results remain available; a new 20B LARGE campaign is not currently prioritized.

## Protocol and reporting checks

- [ ] Use saved **evaluation** settings, not today's training defaults. Existing CARLA results use dt=0.1, eight maps, 2–8 agents, targeted spawning at 50m/25m, route goals 30–80m, 2560 steps, target-inactive ending with 10s collision continuation, traffic-light synchronization on and no proximity early reset. Training uses 35m/10m spawning and 25–75m goals instead.
- [ ] Preserve WOMD's own replay protocol: 91 steps, route goals 30m, zero collision continuation, controlled vehicles and recorded initial scenes. Do not compare its absolute goals/score directly with long CARLA episodes without noting horizon differences.
- [ ] Explicitly distinguish adversary weights (`load_model_path`) from learned target weights (`train.target_policy`); verify the resolved target controller for every cell.
- [ ] Record checkpoint identities, training budget/seed, evaluation image/commit, resolved config, scene selection, job ID and result URI for each new run. Saved 20B training configs have `git.commit_hash: null`; recover image provenance from job metadata.
- [ ] Keep metric names explicit: `sdc_collision_rate`, `sdc_target_collision_genuine_failure_rate`, `sdc_target_collision_adversary_forced_rate`, `sdc_target_collision_unavoidable_rate`, `sdc_at_fault_collision_rate`, `traffic_num_goals_reached`, `sdc_num_goals_reached`, `traffic_score`.
- [ ] Report at-fault separately: it overlaps avoidability classes, not a fourth mutually exclusive type. Classified collision rates need not sum to every SDC collision, particularly on WOMD; report unclassified/other coverage separately.
- [ ] Add uncertainty estimates from episode-level data before paper tables; state whether variation is over evaluation scenes or training seeds. A single training seed does not measure training variability.
- [ ] Produce final tables: 20B versus 100B diagonals; cross-target specialization; adversarial versus ordinary traffic on CARLA; same comparison on WOMD.

## Update log

- 2026-09-14: Created checklist. Verified five successful 20B trainings, final weights/configs on NOA, PDM-only 20B diagonal, full 100B CARLA matrix and five LARGE/WOMD diagonals. Marked obsolete learned-target diagonals and seed/protocol caveats. No cloud mutations or evaluation submissions.
- 2026-09-14: Submitted five reserved single-GPU NOA jobs for the 20B CARLA matrix. Added sequential support to the existing generic eval launcher; all 24 Hydra preflights passed. Job IDs and expected output locations recorded above. Results remain pending.
