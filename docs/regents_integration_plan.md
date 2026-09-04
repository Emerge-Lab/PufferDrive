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
- Logged and adversarial HTML replays label every post-filter adversary candidate and render each in dark brown. The candidate supplying the final hard-min ego-collision loss is labeled `LOSS ADV` and printed in scenario metadata; the selected adversary remains separate metadata for outcome attribution.

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

### Recorded acceptance run

`regents_nuplan`: 16 NuPlan maps, seeds `42..57`, `dt=0.1`, 16 transitions, 500 Adam updates at `1e-3`, and at most three outer iterations. Maximum C/Torch error was `1.526e-5`; success was `1/16` (map 8, seed 50, collision at timestep 11), with 13 filtered, one initial-reconstruction collision, and one iteration-limit rejection. Optimization took `591.6 s` and produced 32 replay pages plus a gallery.

These results are configuration- and horizon-specific. The current `regents_nuplan` config has drifted; restore the saved experiment config under a distinct name before citing them.

## Constraints on future work

1. **Filter yield:** exact logged ego overlap rejected `13/16`. Evaluate penetration-depth or coverage alternatives and always report horizon; moving from 16 to 50 transitions changed which scenes passed.
2. **Drivable raster/loss:** centerline tubes under-cover NuPlan. Reporting subtracts detached baseline potential, which shifts values but not gradients or iterate ranking. The per-vehicle valid-time mean matches the paper's `1/T` reduction for full trajectories; the reference code omits that normalization. Do not copy its crop literally either: it uses `minimum(..., 0)` and x/y indices in `[y, x]` order. Independently, sampled NuPlan edge-side conversions mark 22–58% of logged vehicle centers off-road because edge direction lacks Waymo's inside/outside convention. Reconstruct NuPlan drivable topology, then compare against the paper-intended loss rather than reference-code defects.
3. **Adversary selection:** hard-min ego collision cost sent gradients to one candidate without argmin switching. Test one job per candidate or freeze non-argmin actors before multi-agent optimization.
4. **Optimizer scale:** keep `learning_rate * steering_update_scale` near `0.001 * 4.0 = 0.004` in wheel-angle mode; higher learning rate caused off-road iterates. A 25-scene sweep favored scale `4.0` (`8/25`) over `0.5` (`4/25`), provisionally. Curvature mode removes this unit mismatch by using the reference variable.
5. **Regression invariants:** retain the straight-through signed-speed derivative at zero, per-agent reset before validity exits, separate front-bearing (`pi/8`) and yaw (`pi/2`) windows, and post-Adam steering scaling/clamping. The reset fix reduced worst parity error from `67.118` to `3.815e-5`.
6. **Full horizon:** the 200-transition/20-second run did not complete. Do not extrapolate parity, yield, runtime, or success beyond tested 16- and 50-transition horizons.

## Remaining work

### Stage 7 — Evaluate artifacts (next blocker)

Add a validated artifact-set reader; reconstruct scenario, seed, horizon, and adversary plan; verify the map hash; and replay in C under a configured ego controller without Torch optimization. Emit per-scenario and aggregate collision, actionable-collision, off-road, and infraction metrics from the generation event source.

**Exit:** replaying the saved IDM set exactly reproduces deterministic C collision timesteps and event flags.

### Stage 8 — Policy ego comparison

Load a checkpoint for gradient-free inference through `CONTROLLER_POLICY`. Allow optimization/reactive replay to use fixed IDM or policy ego without changing stop-gradient contracts. Generate against a named controller, record it, then compare IDM and policy on the identical saved set; an IDM-targeted set is not controller-neutral.

**Exit:** policy-targeted generation completes and one table compares both controllers on a fixed artifact set.

### Stage 9 — Full validation and scale

Set long-horizon parity criteria; complete the 200-transition run; report parity, yield, success, and runtime; resolve original-collision filtering; rebuild drivable topology; add headless EGL MP4 beside HTML; and batch scenarios within workers while preserving per-scenario determinism and reductions. A measured CPU trial found simple multiprocessing slower than serial.

**Exit:** a reproducible full-horizon set can be generated efficiently, rendered, and evaluated under both controllers.

Stage 8 depends on Stage 7. Finish Stage 9 correctness work before benchmark claims and batching before benchmark-scale generation. Keep ReGentS out of the PPO hot path until the offline workflow and controller comparison are accepted.
