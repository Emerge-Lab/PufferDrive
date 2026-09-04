# ReGentS integration plan for PufferDrive 3.0

## Goal

Generate deterministic adversarial driving scenarios, then benchmark fixed IDM and learned-policy ego controllers on the same saved scenarios. The C simulator remains authoritative for controller execution, collision/off-road outcomes, and final metrics; PyTorch is used only for differentiable adversary optimization. The local paper and `ReGentS/` checkout are the reference when behavior is ambiguous.

Scope:

- continuous acceleration and target-steering actions with `classic` dynamics;
- vehicle adversaries only;
- one scenario per optimization job until batching is implemented;
- frozen logged, IDM, or policy ego trajectory within each Torch block;
- deterministic offline generation and evaluation;
- no PPO updates or gradients through the ego controller;
- no jerk dynamics, pedestrians/cyclists, or online training integration yet.

## Contracts

- State: `[batch, agent, time, feature]` containing centered `x`, `y`, wrapped heading, signed speed, and steering angle.
- Action: `[batch, agent, time - 1, 2]`, normalized acceleration and target steering in `[-1, 1]`.
- Preserve stable agent identity and one documented coordinate frame. Carry type, dimensions, wheelbase, maximum speed, SDC mask, validity masks, scenario ID, timestep, and map transform.
- Transition validity is `valid[t] & valid[t + 1]`. Invalid values are masked, never repaired.
- Validate all external scenario data, tensor shapes, action ranges, and artifact schemas before optimization or replay.
- Keep controller selection native: `resolve_agent_controller()` routes IDM, policy, and replay from configuration. Do not add a Python controller state machine.

## Delivered: Stages 0–6

All were completed on 2026-09-03. Implementation is in `pufferlib/ocean/regents/`; tests are in `tests/regents/`.

| Stage | Delivered result |
| --- | --- |
| 0 | Native deterministic IDM ego benchmark with logged backgrounds. |
| 1 | Canonical padded Torch export, stable indexing, centered coordinates, signed speeds, dimensions, masks, and deterministic drivable raster transforms. |
| 2 | Differentiable classic bicycle dynamics matching C operation order. Required parity is `<=1e-4` through 64 transitions; measured maximum was `3.815e-5`. Longer horizons require a separately justified threshold. |
| 3 | Deterministic inverse dynamics for acceleration and rate-limited target steering, with validity-gap handling and residual reporting. Real-data P95 gates pass. |
| 4 | Exact oriented-box geometry and differentiable ego-collision, background-collision, and drivable-area costs with deterministic chunking. |
| 5 | Deterministic candidate filters and frozen-ego Adam optimization. Only valid candidate actions change; all others are restored exactly after each update. C feasibility requires a new actionable ego/adversary collision, no new candidate-involved background collision, and no new adversary off-road event. |
| 6 | Stable-index C action injection, C-authoritative replay/events, reactive IDM outer loop, atomic pickle-free artifacts, `puffer regents`, metrics CSV, and HTML replay galleries. |

The initial offline POC is complete. As of the review, 95 tests pass and four CUDA-only tests skip without CUDA.

### Important implementation decisions

- Logged/Torch speed is longitudinal velocity projected onto heading. C-integrated state uses magnitude signed by heading agreement. Compare speed and steering only after injection has produced the state.
- Compare C and Torch position/heading while jointly valid, verify but do not score the shared initial state, and stop parity comparison at the first C collision response.
- Evaluate optimized events relative to a baseline C rollout so logged overlaps and raster mismatches are not attributed to optimization.
- The reactive loop freezes each C ego rollout, optimizes adversaries in Torch, replays in C, and repeats to a bounded outer iteration count. No gradient crosses C or the ego controller.
- Artifacts include schema/version, dataset and scenario identity, exact config and map hash, masks, initial/optimized actions, Torch/C trajectories, controller actions, metadata, and metrics. Loading rejects unknown or inconsistent data.

### Recorded acceptance run

`regents_nuplan`: 16 NuPlan maps, seeds `42..57`, `dt=0.1`, 16 transitions, 500 Adam updates at `1e-3`, and at most three outer iterations.

- maximum C/Torch error: `1.526e-5` against the `1e-4` gate;
- success: `1/16` (map 8, seed 50, collision at timestep 11);
- rejections: 13 filtered, one initial-reconstruction collision, one iteration limit;
- optimization time: `591.6 s`;
- output: 32 replay pages and a gallery.

These numbers are horizon- and configuration-specific. The current `regents_nuplan` config has drifted and does not reproduce this run; only its saved experiment directory records it. Restore the old config under a distinct name before citing the result.

## Findings that constrain future work

1. **Low filter yield.** Exact logged ego overlap rejects most candidates (`13/16` in the acceptance run). Measure penetration-depth or coverage-based alternatives before reporting method-level success. Always report the horizon: changing from 16 to 50 transitions changed which scenes passed without a filter-code change.

2. **Drivable raster mismatch.** Centerline tubes under-cover NuPlan compared with ReGentS road-edge geometry and can mark logged vehicles off-road. Charging only `relu(optimized_potential - baseline_potential)` removed the constant initial penalty and aligns loss with C feasibility, but the raster gradient remains wrong where the map is wrong. Rebuild it from road edges.

3. **Winner-take-all adversary selection.** The hard minimum in ego collision cost sends attraction gradient to one candidate, and the argmin did not switch in the measured run. Test one job per candidate or freeze non-argmin actors before paying for multi-agent optimization.

4. **Optimization scale coupling.** Preserve `learning_rate * steering_update_scale` near the tuned `0.001 * 4.0 = 0.004`. PufferDrive rate-limits steering, so raising the learning rate without lowering the steering scale caused every iterate in a measured two-map run to go off-road. Current generation configs therefore use `learning_rate: 0.001`. The 25-scenario steering-scale sweep favored `4.0` (`8/25`) over the reference `0.5` (`4/25`), but the sample is small and the optimum is provisional.

5. **Correctness fixes already landed.** Preserve the straight-through signed-speed derivative at zero speed, per-agent episode reset before validity early exits, the separate front-bearing (`pi/8`) and yaw (`pi/2`) windows, and post-Adam steering scaling/clamping. The reset fix reduced worst C/Torch error from `67.118` to `3.815e-5`; these are regression requirements.

6. **Full horizon is unvalidated.** The 200-transition/20-second run never completed. Current results cover only 16 or 50 transitions, while cost grows with horizon and Adam iterations. Do not extrapolate parity, yield, runtime, or success.

## Remaining plan

### Stage 7 — Evaluate saved artifacts

- Add an artifact-set reader that reconstructs scenario, seed, horizon, and adversary plan and verifies the stored map hash.
- Replay a fixed artifact set in C under a configured ego controller, without Torch optimization.
- Emit per-scenario and aggregate collision, actionable-collision, off-road, and infraction metrics from the same C event source used by generation.

Exit gate: replaying the saved IDM generation set reproduces its C collision timesteps and event flags exactly wherever C is deterministic.

### Stage 8 — Learned-policy ego and comparison

- Load a checkpoint in inference mode with gradients disabled and use existing `CONTROLLER_POLICY` routing.
- Relax IDM-only guards so optimization and reactive replay accept either fixed IDM or policy controllers without changing stop-gradient contracts.
- Generate against one named controller, then evaluate IDM and policy on that exact saved set. Clearly record the generation controller; an IDM-targeted set is not controller-neutral.

Exit gate: policy-targeted generation completes, and one table compares IDM and policy over a fixed artifact set.

### Stage 9 — Validation, rendering, and scale

- Establish long-horizon parity criteria, complete the 200-transition run, and report parity, yield, success, and runtime.
- Resolve original-collision filtering and rebuild drivable area from road edges.
- Add headless EGL `mp4` output alongside HTML replays.
- Batch scenarios inside each worker while preserving per-scenario determinism and reduction semantics. Simple multi-process execution was slower than serial in the measured CPU trial.

Exit gate: a reproducible, full-horizon artifact set can be generated efficiently, rendered, and evaluated under both controllers.

## Execution order

Stage 7 is the immediate blocker: artifacts currently have no production evaluation consumer. Stage 8 depends on Stage 7. Complete the correctness parts of Stage 9 before publishing benchmark claims; batching is required before producing a benchmark-sized full-horizon set.

Keep ReGentS out of the PPO hot path until the offline pipeline and controller comparison are accepted.
