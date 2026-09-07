# ReGentS integration plan

## Goal and scope

Build a deterministic offline workflow that generates adversarial vehicle scenarios, saves them, and compares fixed IDM and learned-policy ego controllers on the same scenarios. C remains authoritative for controller execution, collisions, off-road events, and metrics; PyTorch only optimizes temporary adversary actions. The local paper and `ReGentS/` checkout resolve ambiguous behavior.

Initial scope is one scenario per job, `classic` dynamics, continuous acceleration/target steering, vehicle adversaries, and a frozen logged/IDM/policy ego trajectory within each Torch block. Excluded for now: PPO updates, gradients through the ego, online training integration, jerk dynamics, pedestrians/cyclists, and batched optimization.

## Contracts and invariants

- Loss weights (2026-09-07): adopt V-Max `origin/dev/Regents_is_back` defaults from `vmax/scripts/evaluate/regents/evaluate.py`, explicitly labeled there as `WAWA TUNING`: ego collision `1.0`, background collision `20.0`, and road deviation `10.0`. This replaces released ReGentS `1/5/20`; the squared-center cutoff and its zero-gradient limitation are unchanged. The source reminder lives beside the constants in `losses.py`.
- Full-log selection can retain candidates absent from a shorter optimization window. Such scenes retain the full-log candidate mask but return `no_candidate_in_optimization_horizon` without evaluating an undefined ego loss. The pinned map-8/16-transition reactive-IDM fixture now exercises this outcome: moving candidate 15 first becomes valid at transition 33, while prior candidates 20 and 22 are stationary at zero logged speed and are filtered. Such a scene writes no loss-history CSV, because it records no cost history. Map 3 supplies the live real-scenario optimizer regression. The map-7 open-loop replay explicitly disables the speed gate only inside its test fixture because its purpose is to exercise baseline-relative collision accounting with a stationary overlapping actor; production generation retains the gate.
- Candidate filtering (updated 2026-09-07): filter over the complete exported log, independently of the optimization horizon. Exclude ego, non-vehicles, agents with valid-state fraction below 0.5, first-to-last displacement below 0.2 m, maximum absolute logged speed below 0.2 m/s, or rear bearing outside +/-7pi/8 for more than 80% of logged timesteps. The speed gate intentionally departs from released ReGentS: NuPlan parked tracks can accumulate enough localization jitter to pass the displacement-only test and then receive adversarial acceleration. Rear occupancy includes invalid timesteps as in the reference. Exclude collation-only time padding using the scenario's maximum exported trajectory length. Minimum valid-transition gates remain removed; logged ego overlaps are diagnostics only. Valid-transition masking still controls which actions exist. Retain adapter presence/metadata checks and explicit caller/invalid-ego scene constraints. `minimum_valid_state_fraction` replaces the obsolete valid-transition filter fields; `static_speed_threshold_mps` is restored and recorded explicitly in generation configs. `valid_state_fraction` and peak speed accompany transition diagnostics. Acceleration units are unchanged.
- Static-track handling (2026-09-07): measure displacement between an agent's first and last *valid* logged states, deviating from the reference's raw first/last storage samples. State storage is zero filled outside an agent's logged frames, so raw endpoints measured distance to the map origin for every agent entering or leaving mid scene: a parked car absent at t=0 reported tens of metres, while a car absent at both endpoints reported exactly 0 m and was wrongly filtered as static. On `regents_nuplan` the endpoint correction moved candidates 536 -> 572, but still admitted 190 candidates that never exceeded 0.1 m/s. The restored speed gate closes that gap. In the fifth saved gallery scenario, parked agents 3 and 5 have zero logged speed but 0.539 m and 0.294 m endpoint jitter; displacement-only selection let optimization move them 11.924 m and 16.345 m. They are now static, remain replay-controlled, and cannot receive optimized actions. Existing artifacts must be regenerated to reflect this policy.
- Experiment overrides (2026-09-07): `puffer regents` accepts `--experiment-name`/`--exp-name` and `--road-weight`/`--drivable-area-weight`. A named run is written below `<configured-output>/<generation-name>/<experiment-name>`; names are restricted to portable path-segment characters. The road weight must be finite and non-negative, may be zero to disable the differentiable road term, and is applied to the resolved optimizer cost config before worker launch. Both overrides are persisted in artifact source configuration and therefore participate in its source hash. C off-road detection and reporting remain enabled when the loss weight is zero.
- Loss/update decision (2026-09-05): use the released code's squared-center collision costs: ego `min_i mean_valid_t ||xy_i-xy_ego||²`, background `-min_pair,t min(1.25², ||xy_i-xy_j||²)`. That background minimum runs over candidate-to-candidate pairs only, as `calculate_distance_adv_col` pairs the adversary trajectories with each other; a pair with one frozen endpoint is a diagnostic, never a loss term. Signed boxes still define actual collisions and the reported box clearance of the center-distance winning pair.
- Adversary-adversary collisions (2026-09-07): the method does not prevent them and we keep it that way. `tau=1.25 m` truncates a *center* distance, so two vehicles overlap long before their centers close to `tau`; the term saturates at `-tau²` and contributes no gradient. Measured on `regents_nuplan`: saturated in 61.1% of iterations, saturated in every iteration for 45.8% of scenarios (median 6 active iterations of ~500), with background collisions in 54% of C replays. Raising `tau` to vehicle scale, or barriering on signed box clearance, would both suppress this but depart from the released constants, so neither is canonical. The released `calculate_distance_bounding_boxes` is `pass`, so the reference offers no box-distance variant to match. Road cost sums all valid candidate corners and timesteps without baseline subtraction. Its grid kernel uses the reference amplitude `exp(-r²/(2 sigma²))/(sigma sqrt(2 pi))`, without discrete normalization or pixel-area weighting. We retain a three-sigma grid convolution, bilinear interpolation, correct x/y indexing and out-of-map padding, rather than copying the reference's defective 32-pixel crop. There is no reference road gradient to be in parity with: see the deviation-term audit below. Weights are V-Max `1/20/10` as of 2026-09-07.
- Deviation-term audit (2026-09-07): the released deviation term never fires, so its coefficient carries no parity constraint. `cost.py:96` crops with `jnp.minimum(x_center - cropsize//2, 0)`, which is at most zero, and `dynamic_slice` clamps a negative start to zero, so every agent is scored against the same 32x32 patch at the map's origin corner; the map is also built `[y, x]` by `meshgrid`/`lax.map` but sliced `(x_center, y_center)`. The Gaussian therefore evaluates hundreds of metres from the agent and returns zero. Released ReGentS is effectively `ego_col + 5*adv_col`. Measured on scenario 0 with our working term at weight 10 versus weight 0: ego cost 25.72 -> 34.89 (failed) versus 25.72 -> 13.47 (succeeded), and the non-argmin agent 1 moved 14.58 m versus 0.19 m. The road term is therefore ours to validate and tune, not to match.
- Drivable-area raster (2026-09-07): build the surface from road-edge polylines (`ROAD_EDGE_UNKNOWN/BOUNDARY/MEDIAN`, newly exported as `binding.ROAD_TYPE_ROAD_EDGE_*`) with the reference signed-distance test, joined with the previous half-lane-width centerline corridor. Two corrections were needed. The export preserves no edge winding, unlike the reference dataset: on map 0, 737 edge points face their nearest lane negatively and 337 positively, whole polylines disagreeing, so each polyline is oriented by majority vote against the nearest lane centerline before the sign test runs unchanged. The nearest-point test then still mislabels wedges where two polylines meet, which alone left the map-0 ego off-road for 89% of its logged positions, so the lane corridor is unioned in: a mapped centerline is drivable by construction and repairs exactly those wedges. Across the 50 `regents_nuplan` maps the logged ego's road cost is now a median 0.05 of a 404 per-agent maximum (mean 7.54, max 46.42), moving candidates a median 0.00, and only 5 parked candidates sit fully off the road surface; raster build costs a mean 1.9 s per map. A map without road edges now aborts rather than falling back to the corridor alone. Re-running the 50-scenario `regents_nuplan` cohort with both corrections and the unchanged 1/20/10 weights: generation success 40% -> 56%, off-road 14% -> 8%, `iteration_limit` failures 22 -> 14, background collisions 54% -> 62% (the saturated term still shapes nothing), C/torch parity unchanged at 3.4e-05, mean optimization 38.9 s per scenario against 29.4 s. Scenario 0 still fails because its 100-transition window holds a stationary ego, which is a horizon question, not a road-cost one.
- Deviation kernel normalization (2026-09-07): the convolution kernel carries unit mass, so a fully out-of-bounds corner costs exactly one at any raster resolution. The released amplitude has neither discrete mass normalization nor a pixel-area factor, which makes its potential scale with the inverse square of the resolution: a saturated corner is 1.290 at the reference's 1.0 m default and 5.011 at our 0.5 m raster. Since the released term never evaluates at the sampled position, that scale is not worth reproducing.
- Steering now defaults to curvature with a 0.5 post-Adam update scale. Divergence cancels only the applied steering update while preserving raw gradients and Adam moments. Projection follows scaling/cancellation. Wheel-angle mode remains explicit, also with default scale 0.5. Acceleration remains normalized and all PufferDrive physical limits remain in force. Historical sweeps below predate these loss/update changes.
- Stopping/result selection (2026-09-05): follow the released ReGentS optimizer by stopping on any valid candidate/ego overlap (including iteration zero) and returning the current iterate, or the last evaluated iterate at the update limit. Background collisions and off-road events are diagnostics, not acceptance gates. C replay still verifies trajectory parity and a new actionable ego collision; other infractions do not invalidate success. Optional patience and collision-stop overrides remain available, with patience disabled by default. Loss geometry, weights, filters, and frozen-ego dynamics are unchanged. We evaluate the final Adam update, unlike the reference's inner loop.
- Legacy artifact fields `best_iteration` and `*_rejection_count` now mean returned iteration and counts of iterates with new infractions, respectively. They no longer imply selection or rejection; artifacts identify the stopping policy explicitly. Prior acceptance results below used the former strict feasibility policy and are not directly comparable.
- State is `[batch, agent, time, feature]`: centered `x/y`, wrapped heading, signed speed, and steering angle. Action is `[batch, agent, time - 1, 2]`: normalized acceleration and target steering in `[-1, 1]`.
- Preserve stable agent identity, one documented coordinate frame, type, dimensions, wheelbase, maximum speed, SDC and validity masks, scenario ID, timestep, and map transform. Transition validity is `valid[t] & valid[t + 1]`; mask invalid values rather than repairing them.
- Validate external scenario data, tensor shapes/ranges, and artifact schemas before use. Controller routing remains native through `resolve_agent_controller()`; do not add a Python controller state machine.
- Logged/Torch speed is longitudinal velocity projected onto heading. C state uses speed magnitude signed by heading agreement. Compare speed/steering only after injection, compare position/heading only while jointly valid, exclude the shared initial state from scoring, and stop at the first C collision response.
- Judge optimized C events relative to a baseline C rollout so logged overlaps and raster errors are not attributed to optimization.
- The reactive loop alternates a frozen C ego rollout, Torch adversary optimization, and C replay for a bounded number of iterations. No gradient crosses C or the ego controller.
- Artifacts carry schema/version, dataset/scenario identity, exact config and map hash, masks, initial/optimized actions, Torch/C trajectories, controller actions, metadata, and metrics. Writes are atomic and pickle-free; loads reject unknown or inconsistent data.
- Logged and adversarial HTML replays label every post-filter adversary candidate. Ordinary `ADV` candidates render in dark yellow; the candidate supplying the final hard-min ego-collision loss renders in magenta, is labeled `LOSS ADV`, and is printed in scenario metadata. The selected adversary remains separate metadata for outcome attribution.
- Captured ReGentS HTML omits observations by default, including the ego observation. A generation may explicitly opt in with `capture_observations: true`; this is valid only when HTML frames are also captured. Replay `active_count` remains the single controlled ego and never derives from the rollout horizon.

## Status: offline POC complete

Stages 0–6 were completed on 2026-09-03 in `pufferlib/ocean/regents/`, with tests in `tests/regents/`:

| Stage | Result |
| --- | --- |
| 0–1 | Deterministic native IDM benchmark; canonical padded Torch export with stable indexing, centered coordinates, metadata, masks, and drivable raster transforms. |
| 2–3 | Differentiable classic dynamics and deterministic inverse dynamics, including validity gaps and residual reporting. C/Torch parity gate is `<=1e-4` through 64 transitions; measured maximum is `3.815e-5`. Longer horizons need a separate gate. |
| 4–5 | Exact oriented-box geometry, differentiable collision/drivable costs, deterministic filtering, and frozen-ego Adam optimization. Only valid candidate actions change. C success requires parity and a new actionable ego/adversary collision; background collisions and off-road events are reported independently. |
| 6 | Stable-index C injection, authoritative replay/events, reactive IDM loop, validated artifacts, `puffer regents`, metrics CSV, and HTML galleries. |

Candidate-filter verification (2026-09-07): all 16 filter/optimizer/C-replay tests pass. The full suite reports 33 passed, one skipped, and one unrelated loss-diagnostics failure: its fixture supplies only one optimized agent while expecting a pair after the separate candidate-pair scope change. Filter boundaries, full-log versus short-horizon behavior, and reference-expression agreement are covered.

`tests/regents/` deliberately holds a few broad regression tests rather than many narrow ones - roughly two to four per module, each covering one behaviour end to end. Shared scenario builders live in `conftest.py`, and all CUDA parity sits in `test_cpu_gpu_consistency.py`, which skips wholesale without a GPU. Add assertions to the existing test that owns a behaviour rather than adding a test function; the trade-off is that a failure early in a test hides the assertions after it.

### Action-space decision

Reference ReGentS/Waymax uses raw acceleration (`+-6.0 m/s^2`) and instantaneous curvature (`+-0.3 1/m`), with a `0.6 m/s` inverse-dynamics steering guard. Its KING ego also writes PID throttle/steer into those physical-model slots. PufferDrive uses normalized actions over `+-4.0 m/s^2` and a `+-0.667 rad` target wheel angle rate-limited to `0.6 rad/s`.

`waymax_actions.py` provides differentiable conversion using

    curvature = cos(atan(rear_axle_ratio * tan(wheel_angle))) * tan(wheel_angle) / wheelbase

and its closed-form inverse. Reference authority exceeds PufferDrive authority: the latter reaches about `0.735 / wheelbase` curvature. Conversion therefore clamps unreachable values and reports saturation. Converted steering remains a rate-limited target, so it does not reproduce the requested yaw rate on the first step.

`ReGentSOptimizationConfig.steering_parameterization` selects:

- `wheel_angle`: normalized target angle with `steering_update_scale=0.5`.
- `curvature` (default): curvature in `1/m`, clamped per agent to `+-0.735 / wheelbase`, converted at rollout, with `curvature_steering_update_scale=0.5`.

Adam and post-update front-divergence cancellation operate on the selected parameter; moments are preserved. Frozen entries are copied verbatim from the post-conversion baseline. Artifact `initial_actions` and `optimized_actions` always remain normalized simulator actions, so downstream replay is parameterization-independent. Curvature round-trip error is about `1e-7` at iteration zero, so the curvature parameter is projected to `(1 - 1e-6)` of the achievable limit, before the first forward pass as well as after every Adam step: projecting onto the limit itself round trips, in float32, to a normalized wheel angle one ulp outside the action box for about 13% of wheelbases, which the C injector rejects.

A 12-map NuPlan comparison (seeds `42..53`, 50 transitions, 200 iterations, `lr=1e-3`; two filtered) tied at `7/10` successes with no safety rejections. Peak steering was `18.7` vs `19.4` degrees; curvature took 58 vs 45 iterations per success. This supports transfer of the reference scale but not changing the default; wall times were not comparable.

### Historical background non-collision loss decision (superseded 2026-09-05)

The paper defines the background-agent regularizer inherited from KING as

    C_adv_col(s) = -min_{i != j, i,j in backgrounds, k in [0,T-1]} min(tau, d_BB(i,j,k))

where `d_BB` is the closest-point distance between oriented bounding polygons. The prose calls it Euclidean while the collision condition uses `d_BB <= 0`; PufferDrive resolves that ambiguity with signed oriented-box distance in meters: it equals closest-point distance while separated and becomes negative under penetration. The paper uses one global hard minimum, so only the closest pair/timestep receives a gradient. This sparse gradient is paper-faithful, not an implementation accident.

The released `ReGentS/cost.py` is not the authority for geometry: its bounding-box distance function is unimplemented and its default path uses squared center distance with `tau**2`. It does confirm removal of self/duplicate pairs, truncation before negation, weights `5.0` with `tau=1.25`, and that the filtered `adv_idx` is the agent set supplied to both collision costs.

The canonical PufferDrive loss therefore uses post-filter candidate-to-candidate pairs, matching the released ReGentS method. Candidate-to-frozen-background contacts remain part of baseline-relative diagnostics, not the differentiable paper loss or an acceptance gate. A trial that widened the differentiable pair set made all 30 map-8 iterates off-road and removed the established collision, so wider pair scope is experimental rather than canonical.

### Recorded acceptance run

`regents_nuplan`: 16 NuPlan maps, seeds `42..57`, `dt=0.1`, 16 transitions, 500 Adam updates at `1e-3`, and at most three outer iterations. Maximum C/Torch error was `1.526e-5`; success was `1/16` (map 8, seed 50, collision at timestep 11), with 13 filtered, one initial-reconstruction collision, and one iteration-limit rejection. Optimization took `591.6 s` and produced 32 replay pages plus a gallery.

These results are configuration- and horizon-specific. The current `regents_nuplan` config has drifted; restore the saved experiment config under a distinct name before citing them.

## Constraints on future work

1. **Filter yield:** the historical overlap filter rejected `13/16`; the reference filter no longer excludes logged overlaps. Re-measure yield with full-log selection and report the optimization horizon separately, including candidates that enter after it.
2. **Drivable raster/loss:** centerline tubes under-cover NuPlan. Current loss uses the reference sum and absolute potential as specified above; raster resolution now affects its scale. Do not copy the reference crop literally: it uses `minimum(..., 0)` and x/y indices in `[y, x]` order. Sampled NuPlan edge-side conversions marked 22–58% of logged vehicle centers off-road because edge direction lacks Waymo's inside/outside convention. Drivable topology remains an independent limitation.
3. **Adversary selection/loss:** the paper uses hard minima for both ego collision induction and background collision avoidance. The ego term sent gradients to one candidate without argmin switching; the background term likewise exposes only one candidate pair/timestep. If joint optimization remains poor after the canonical loss is fully instrumented, compare one job per candidate and an explicitly non-paper dense pairwise surrogate rather than silently changing the canonical loss.
4. **Optimizer scale:** use curvature with reference steering scale 0.5 by default. Earlier wheel-angle sweeps used different losses and acceptance rules and do not establish suitable settings for the current objective.
5. **Regression invariants:** retain the straight-through signed-speed derivative at zero, per-agent reset before validity exits, separate front-bearing (`pi/8`) and yaw (`pi/2`) windows, and post-Adam steering scaling/clamping. The reset fix reduced worst parity error from `67.118` to `3.815e-5`.
6. **Full horizon:** the 200-transition/20-second run did not complete. Do not extrapolate parity, yield, runtime, or success beyond tested 16- and 50-transition horizons.

## Remaining work

### Stage 7 — Correct background non-collision loss (historical, superseded by reference-code objective)

1. Make the pair set explicit and deterministic: unique `i < j` post-filter candidate pairs with at least one jointly valid timestep. Keep separate partner/optimized masks in the loss API so wider experimental scopes cannot silently alter the canonical call.
2. Preserve the paper reduction: signed oriented-box distance in meters, global pair/time hard minimum, upper truncation at `tau=1.25 m`, negation, then weight `5.0`. No eligible pair returns zero; chunking must preserve the value.
3. Record the winning pair indices and stable IDs, timestep, raw signed distance, and truncation state in every loss snapshot, artifact, and loss CSV. Also report pre-existing background-collision pair counts and rejected iterates.
4. Cover separation, contact, penetration, truncation, validity, masking, multi-candidate gradient sparsity, mixed frozen actors, optimizer behavior, artifacts, CSV, determinism, and CPU/GPU parity in the existing regression tests.
5. Re-run the established C fixture and a fixed multi-candidate NuPlan comparison. If the canonical hard minimum still starves optimization, evaluate a separately named non-paper per-pair margin loss; never relabel it as the paper objective.

**Exit:** loss values match the equation, only canonical candidate pairs contribute, diagnostics round-trip, deterministic tests pass, and the established C replay introduces no new background collision.

**Status (2026-09-04):** complete. Loss scope, diagnostics, artifact v2 output with v1 read compatibility, CSV reporting, and focused regressions are implemented. The three apparent fixture failures came from selecting a mutable map directory by position: newly downloaded files changed the Stage 3 audit cohort and replaced the recorded map-8 scenario. The regressions now resolve the original 16 scenarios by stable ID without changing metrics or thresholds; the recovered cohort reproduces the 67,111-transition audit exactly, and both pinned C replay gates pass. The full ReGentS suite passes with 27 tests and one CUDA-only skip.

### Stage 8 — Evaluate artifacts

Add a validated artifact-set reader; reconstruct scenario, seed, horizon, and adversary plan; verify the map hash; and replay in C under a configured ego controller without Torch optimization. Emit per-scenario and aggregate collision, actionable-collision, off-road, and infraction metrics from the generation event source.

**Exit:** replaying the saved IDM set exactly reproduces deterministic C collision timesteps and event flags.

### Stage 9 — Policy ego comparison

Load a checkpoint for gradient-free inference through `CONTROLLER_POLICY`. Allow optimization/reactive replay to use fixed IDM or policy ego without changing stop-gradient contracts. Generate against a named controller, record it, then compare IDM and policy on the identical saved set; an IDM-targeted set is not controller-neutral.

**Exit:** policy-targeted generation completes and one table compares both controllers on a fixed artifact set.

### Scenario batching (implemented, off by default)

`optimize_frozen_ego_scenarios` runs any number of scenarios in one Adam loop;
`collate_scenarios` pads them to a shared agent and time count. Generation schedules
one batch per pool task via `batch_size`, ordering scenarios by agent count first.
Capture and C replay stay per-Drive, so no C contract changed.

**Correctness.** Batch size 1 reproduces the previous implementation exactly. Across a
240-scenario NuPlan run, batch 2 matched batch 1 on 239 of 240 scenarios; the one
difference was a `collision_timestep` of 82 against 83, with identical success,
adversary, and infraction verdicts. Scenarios stay independent because each loss reads
only its own actions, so the summed batch loss gives each row the gradient it would get
alone; a stopped scenario is masked out and its parameters restored after each step.

**Measured effect.** Batching removes per-iteration Python and dispatch overhead
(`_compose_rollout` is 30.8 ms of a ~67 ms iteration) but raises each worker's working
set. Those pull in opposite directions:

| setting | result |
| --- | --- |
| 1 worker, 8 scenarios, 100 iterations | 99 s at batch 1, 73 s at batch 8 (1.36x faster) |
| 24 workers, 240 scenarios, 100 iterations | 313 s at batch 1, 449 s at batch 2 (1.43x slower) |
| in-worker CPU, same 240 scenarios | 6973 s at batch 1, 8888 s at batch 2 (+27%) |

The regression is contention, not scheduling: per-scenario CPU time itself rises once 24
workers compete. `batch_size` therefore defaults to 1, which is the fastest setting for a
24-worker run on this machine; raise it only when workers sit well below core count.

An earlier synthetic benchmark predicted 2.09x. It was wrong twice over: it replicated one
scenario, so every batch member shared an agent count and a drivable-area raster. Real
agent counts run from 3 to 326, and padding in file order wastes 33% of the work at batch
2 and 54% at batch 8. Agent-count ordering recovers that to 96% and 84%, and is why
`_agent_count_ordered_batches` exists, but it does not recover the contention cost.

Two levers remain if batching is wanted at full worker count. `_background_collision_signature`
costs 9.8 ms per iteration and grows with the square of the background agent count, which is
what makes large scenes expensive to hold in a batch. Batches also run until every member
stops, worth about 77% efficiency at batch 8 given the observed iteration spread (min 1,
median 501, p75 501). Neither is addressed.

### Stage 10 — Full validation and scale

Set long-horizon parity criteria; complete the 200-transition run; report parity, yield, success, and runtime; resolve original-collision filtering; rebuild drivable topology; and add headless EGL MP4 beside HTML. Batching within workers is implemented and verified but is off by default; see the batching section above for why it does not pay at 24 workers. A measured CPU trial found simple multiprocessing slower than serial.

**Exit:** a reproducible full-horizon set can be generated efficiently, rendered, and evaluated under both controllers.

Stages 8–10 depend on the Stage 7 loss gate, and Stage 9 depends on Stage 8. Finish Stage 10 correctness work before benchmark claims and batching before benchmark-scale generation. Keep ReGentS out of the PPO hot path until the offline workflow and controller comparison are accepted.
