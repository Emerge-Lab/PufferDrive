# ReGentS integration plan

## Goal and scope

Build a deterministic offline workflow that generates adversarial vehicle scenarios, saves them, and compares fixed IDM and learned-policy ego controllers on the same scenarios. C remains authoritative for controller execution, collisions, off-road events, and metrics; PyTorch only optimizes temporary adversary actions. The local paper and `ReGentS/` checkout resolve ambiguous behavior.

Initial scope is one scenario per job, `classic` dynamics, continuous acceleration/target steering, vehicle adversaries, and a frozen logged/IDM/policy ego trajectory within each Torch block. Excluded for now: PPO updates, gradients through the ego, online training integration, jerk dynamics, pedestrians/cyclists, and batched optimization.

## Contracts and invariants

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
| 4–5 | Exact oriented-box geometry, differentiable collision/drivable costs, deterministic filtering, and frozen-ego Adam optimization. Only valid candidate actions change. C feasibility requires a new actionable ego/adversary collision, no new candidate/background collision, and no new adversary off-road event. |
| 6 | Stable-index C injection, authoritative replay/events, reactive IDM loop, validated artifacts, `puffer regents`, metrics CSV, and HTML galleries. |

Current review: 28 tests pass in about 25 s.

`tests/regents/` deliberately holds a few broad regression tests rather than many narrow ones - roughly two to four per module, each covering one behaviour end to end. Shared scenario builders live in `conftest.py`, and all CUDA parity sits in `test_cpu_gpu_consistency.py`, which skips wholesale without a GPU. Add assertions to the existing test that owns a behaviour rather than adding a test function; the trade-off is that a failure early in a test hides the assertions after it.

### Action-space decision

Reference ReGentS/Waymax uses raw acceleration (`+-6.0 m/s^2`) and instantaneous curvature (`+-0.3 1/m`), with a `0.6 m/s` inverse-dynamics steering guard. Its KING ego also writes PID throttle/steer into those physical-model slots. PufferDrive uses normalized actions over `+-4.0 m/s^2` and a `+-0.667 rad` target wheel angle rate-limited to `0.6 rad/s`.

`waymax_actions.py` provides differentiable conversion using

    curvature = cos(atan(rear_axle_ratio * tan(wheel_angle))) * tan(wheel_angle) / wheelbase

and its closed-form inverse. Reference authority exceeds PufferDrive authority: the latter reaches about `0.735 / wheelbase` curvature. Conversion therefore clamps unreachable values and reports saturation. Converted steering remains a rate-limited target, so it does not reproduce the requested yaw rate on the first step.

`ReGentSOptimizationConfig.steering_parameterization` selects:

- `wheel_angle` (default): normalized target angle with `steering_update_scale`; byte-identical to the earlier optimizer on maps 3, 5, and 8 over 40 iterations.
- `curvature`: curvature in `1/m`, clamped per agent to `+-0.735 / wheelbase`, converted at rollout, with `curvature_steering_update_scale=0.5` by default.

Adam, front-divergence cancellation, and momentum reset operate on the selected parameter. Frozen entries are copied verbatim from the post-conversion baseline. Artifact `initial_actions` and `optimized_actions` always remain normalized simulator actions, so downstream replay is parameterization-independent. Curvature round-trip error is about `1e-7` at iteration zero.

A 12-map NuPlan comparison (seeds `42..53`, 50 transitions, 200 iterations, `lr=1e-3`; two filtered) tied at `7/10` successes with no safety rejections. Peak steering was `18.7` vs `19.4` degrees; curvature took 58 vs 45 iterations per success. This supports transfer of the reference scale but not changing the default; wall times were not comparable.

### Background non-collision loss decision

The paper defines the background-agent regularizer inherited from KING as

    C_adv_col(s) = -min_{i != j, i,j in backgrounds, k in [0,T-1]} min(tau, d_BB(i,j,k))

where `d_BB` is the closest-point distance between oriented bounding polygons. The prose calls it Euclidean while the collision condition uses `d_BB <= 0`; PufferDrive resolves that ambiguity with signed oriented-box distance in meters: it equals closest-point distance while separated and becomes negative under penetration. The paper uses one global hard minimum, so only the closest pair/timestep receives a gradient. This sparse gradient is paper-faithful, not an implementation accident.

The released `ReGentS/cost.py` is not the authority for geometry: its bounding-box distance function is unimplemented and its default path uses squared center distance with `tau**2`. It does confirm removal of self/duplicate pairs, truncation before negation, weights `5.0` with `tau=1.25`, and that the filtered `adv_idx` is the agent set supplied to both collision costs.

The canonical PufferDrive loss therefore uses post-filter candidate-to-candidate pairs, matching the released ReGentS method. Candidate-to-frozen-background contacts remain part of the stricter baseline-relative feasibility check, not the differentiable paper loss. A trial that widened the differentiable pair set made all 30 map-8 iterates off-road and removed the established collision, so wider pair scope is experimental rather than canonical.

### Recorded acceptance run

`regents_nuplan`: 16 NuPlan maps, seeds `42..57`, `dt=0.1`, 16 transitions, 500 Adam updates at `1e-3`, and at most three outer iterations. Maximum C/Torch error was `1.526e-5`; success was `1/16` (map 8, seed 50, collision at timestep 11), with 13 filtered, one initial-reconstruction collision, and one iteration-limit rejection. Optimization took `591.6 s` and produced 32 replay pages plus a gallery.

These results are configuration- and horizon-specific. The current `regents_nuplan` config has drifted; restore the saved experiment config under a distinct name before citing them.

## Constraints on future work

1. **Filter yield:** exact logged ego overlap rejected `13/16`. Evaluate penetration-depth or coverage alternatives and always report horizon; moving from 16 to 50 transitions changed which scenes passed.
2. **Drivable raster/loss:** centerline tubes under-cover NuPlan. Reporting subtracts detached baseline potential, which shifts values but not gradients or iterate ranking. The per-vehicle valid-time mean matches the paper's `1/T` reduction for full trajectories; the reference code omits that normalization. Do not copy its crop literally either: it uses `minimum(..., 0)` and x/y indices in `[y, x]` order. Independently, sampled NuPlan edge-side conversions mark 22–58% of logged vehicle centers off-road because edge direction lacks Waymo's inside/outside convention. Reconstruct NuPlan drivable topology, then compare against the paper-intended loss rather than reference-code defects.
3. **Adversary selection/loss:** the paper uses hard minima for both ego collision induction and background collision avoidance. The ego term sent gradients to one candidate without argmin switching; the background term likewise exposes only one candidate pair/timestep. If joint optimization remains poor after the canonical loss is fully instrumented, compare one job per candidate and an explicitly non-paper dense pairwise surrogate rather than silently changing the canonical loss.
4. **Optimizer scale:** keep `learning_rate * steering_update_scale` near `0.001 * 4.0 = 0.004` in wheel-angle mode; higher learning rate caused off-road iterates. A 25-scene sweep favored scale `4.0` (`8/25`) over `0.5` (`4/25`), provisionally. Curvature mode removes this unit mismatch by using the reference variable.
5. **Regression invariants:** retain the straight-through signed-speed derivative at zero, per-agent reset before validity exits, separate front-bearing (`pi/8`) and yaw (`pi/2`) windows, and post-Adam steering scaling/clamping. The reset fix reduced worst parity error from `67.118` to `3.815e-5`.
6. **Full horizon:** the 200-transition/20-second run did not complete. Do not extrapolate parity, yield, runtime, or success beyond tested 16- and 50-transition horizons.

## Remaining work

### Stage 7 — Correct background non-collision loss (completed)

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
