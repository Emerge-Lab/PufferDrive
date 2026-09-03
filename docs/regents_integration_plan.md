# ReGentS integration plan for PufferDrive 3.0

## Goal and scope

Implement support for ReGentS in PufferDrive in six stages. The existing C simulator remains the source of truth for dynamics, collision outcomes, and final evaluation. A separate PyTorch path provides differentiable dynamics and costs for adversarial action optimization.

The initial scope is deliberately narrow:

- replay scenarios with continuous actions and `classic` dynamics;
- vehicle adversaries only;
- one scenario per optimization job before adding batching;
- a frozen logged or IDM ego rollout during each Torch optimization block;
- IDM as the first reactive ego controller, before a learned ego policy;
- deterministic, offline scenario generation and evaluation;
- no PPO loop, policy updates, rollout buffer, or training command in the ReGentS POC;
- **fixed ego policy / IDM:** we do not train or update any ego policy while using ReGentS; the ego policy and IDM are completely fixed (frozen) and run in evaluation/inference mode only.

Pedestrians/cyclists, joint policy gradients through the ego policy, and online generation during PPO training are follow-up work. To ensure correct physical model representation and sample-efficient gradients, adversarial actions are strictly defined as acceleration and steering (classic dynamics); jerk dynamics are explicitly excluded from the adversarial planning and optimization process. The method reference is the local [ReGentS paper](../2409.07830v1.pdf). Additionally, the original ReGentS reference codebase has been added directly to this workspace in the `ReGentS/` directory to help guide and ensure the correct integration. If any doubt or ambiguity arises during the implementation of these stages, developers should consult the reference files in `ReGentS/`.

### Ultimate Goal: Policy & IDM Benchmarking and Visual Rendering

Beyond the initial POC, the ultimate objective of the ReGentS integration is to systematically evaluate and compare different ego controllers under synthesized adversarial conditions:

- **Benchmark against IDM:** Evaluate how an IDM-controlled SDC responds when subjected to ReGentS-optimized adversarial scenarios.
- **Benchmark against the Policy:** Evaluate the learned neural network policy (controlling the ego agent) against these same adversarial scenarios to assess its resilience and safety performance under high-stress conditions.
- **Comparative Analysis:** Contrast the final evaluation metrics (collision rates, off-road events, infraction rates) of the IDM controller versus the trained ego policy to benchmark performance improvements.
- **Visual Rendering:** Generate high-fidelity rendered videos (`mp4` via the headless EGL pipeline or interactive replays) of these evaluations as soon as rendering capabilities become available for the ReGentS pipeline, ensuring qualitative validation of both adversarial maneuvers and ego reactions.

## Step 0 — Validate IDM as the ego controller (Completed)

Completed (2026-09-03, Commit `a68e49f7`): Integrated native C IDM controller benchmark scenario config (`regents_idm` in `benchmark.yaml`) and verified deterministic routing, logged background motion, and evaluation metrics without requiring any Torch policy or checkpoint. The NuPlan baseline rollout on 100 scenarios using seed `42` with `init_step=0` was documented, and regression tests were added in `tests/regents/test_stage0_idm.py`.

## Core contracts

Use one explicit Torch representation throughout the differentiable path:

- state tensors: `[batch, agent, time, feature]`, with at least scenario-local centered `x`, `y`, wrapped heading, signed speed, and steering angle;
- action tensors: `[batch, agent, time - 1, 2]`, containing normalized PufferDrive acceleration and target-steering actions in `[-1, 1]`;
- metadata: agent type, stable agent ID/index, SDC mask, length, width, wheelbase, per-agent maximum speed, scenario ID, and timestep;
- masks: state validity, transition validity (`valid[t] & valid[t + 1]`), vehicle, ego, candidate adversary, and optimized action;
- map data: a documented world-to-grid transform and either a drivable-area raster or enough geometry to construct it deterministically.

All APIs must preserve agent identity and use one documented coordinate frame. Invalid entries are represented by masks, not silently repaired. External scenario data and tensor shapes are validated before optimization starts.

## Stage 1 — Expose complete scenarios and build the Torch adapter (Completed)

Completed (2026-09-03, Commit `80d7b938`): Implemented scenario exporting and canonical padding directly from `Drive.get_state()` payloads via `export_drive_scenarios()` in `pufferlib/ocean/regents/adapter.py`.
- **Schema & Normalization:** State dataclasses defined in `state.py` represent centered Cartesian frames, stable agent indexing, SDC masking (`EGO_IDX == 0`), signed speed derivation from projections, and time-varying/first-valid agent dimensions using a shared `WHEELBASE_LENGTH_RATIO`.
- **Drivable Area Rasterization:** Built raster-transform structures (`DrivableAreaRaster`, `RasterTransform`) mapping centered simulation space to grids.
- **Verification:** Exhaustive tests implemented in `tests/regents/test_adapter.py` validating shapes, vectorized padding, dtypes, centered coordinate transform consistency, and byte-identical repeated exports.

## Stage 2 — Implement differentiable classic dynamics and prove C parity (Completed)

Completed (2026-09-03, Commit `7de36bb5c`): Implemented pure Torch `classic_step()` and masked `classic_rollout()` in `pufferlib/ocean/regents/dynamics.py`. The CPU `float32` implementation follows the C `move_dynamics()` operation order, including continuous action scaling, steering-rate and angle limits, signed speed clipping, slip angle, position integration at `old_heading + beta`, heading wrapping, and carried actual steering state. A diagnostic binding runs the authoritative C step in isolation, with stochastic and infraction behavior disabled, and returns all five state components.

The agreed strict parity horizon for the initial POC is 64 transitions, matching its scenario horizon. Tests compare every state component at every step and measured these maximum absolute errors:

- `1.7882e-7` across the one-step neutral, braking, acceleration, reverse, steering-limit, speed-limit, and heading-wrap cases;
- `3.8147e-5` over deterministic random actions for 64 transitions across two batches and three agents;
- `3.0923e-11` over 32 approximately inverse-derived actions from a real replay scenario.

All results pass the mandatory `1e-4` threshold. An exploratory 256-transition random-action stress test reached `1.6404e-4` from accumulated C-libm versus Torch transcendental rounding; any future horizon longer than 64 transitions must establish an appropriate parity approach and acceptance threshold before use. Stage 3 inverse/forward reconstruction remains required before M1 is complete or loss/optimization work may begin.

## Stage 3 — Estimate expert actions with inverse dynamics (Completed)

Completed (2026-09-03): Implemented `estimate_expert_actions()` in `pufferlib/ocean/regents/inverse_dynamics.py`. It estimates normalized acceleration from consecutive signed speeds and analytically seeds steering from wrapped heading change, then performs a deterministic bounded steering search. Acceleration first minimizes reachable speed error; conditional on that result, steering minimizes squared position error plus wheelbase-scaled squared heading error. Every inconsistent transition retains its component errors and combined meter-scaled residual. Steering is carried within each contiguous run, reset to neutral at an unobserved run start, and never inferred across a validity gap.

The implementation was compared with the paper and its bundled reference code. ReGentS delegates background action estimation to Waymax's expert actor and `InvertibleBicycleModel`; it does not contain a separate inverse implementation. The shared semantics are consecutive-state local derivatives, wrapped yaw differences, action bounds, validity masking, and Waymax's `0.6 m/s` low-speed noise guard. The equations intentionally differ because Waymax controls unsigned acceleration and curvature with trapezoidal position integration, whereas PufferDrive controls signed acceleration and a rate-limited target wheel angle, updates speed before Euler position integration, and uses wheelbase plus slip angle. Copying Waymax's curvature inverse would therefore fail the Stage 2 reference dynamics.

Exact forward/inverse tests cover Torch- and C-generated trajectories, yaw wrapping, acceleration and steering limits, and reverse motion at the Stage 2 `1e-4` tolerance. Gap and low-speed tests prove that invalid actions remain masked and low-speed heading residuals are excluded.

The fixed initial real-data audit uses the lexicographically first 16 local NuPlan scenarios, seeds `42..57`, `init_step=0`, `dt=0.1`, and all 67,111 valid vehicle transitions. There are 27,339 normal-speed and 39,772 low-speed transitions. Percentiles below are `P50 / P90 / P95 / P99 / max`:

| Split | Position error (m) | Heading error (rad) | Speed error (m/s) |
| --- | --- | --- | --- |
| Normal speed | `0.04252 / 0.16211 / 0.23694 / 0.47635 / 2.10607` | `0.001442 / 0.006005 / 0.009119 / 0.020494 / 0.440930` | `0 / 9.54e-7 / 9.54e-7 / 0.09736 / 4.25930` |
| Low speed | `0.01684 / 0.05301 / 0.07534 / 0.17265 / 3.18633` | excluded; diagnostic only | `0 / 0 / 0 / 1.16e-10 / 8.53201` |

The per-timestep mean ranges are `0.03525..0.05448 m` position, `0.001934..0.006410 rad` normal-speed heading, and `4.69e-8..0.03005 m/s` speed. The recorded acceptance gates are: normal-speed P95 position `<= 0.25 m`, heading `<= 0.012 rad`, speed `<= 1e-5 m/s`; low-speed P95 position `<= 0.08 m` and speed `<= 1e-5 m/s`; maximum per-timestep means `<= 0.06 m`, `<= 0.007 rad`, and `<= 0.04 m/s`, respectively. Maxima are diagnostic rather than acceptance gates because logged transitions can violate acceleration and steering constraints; those outliers remain reported and masked only by true trajectory validity.

Both M1 hard gates now pass. Stage 4 loss/geometry work is unlocked, but no optimization work has started.

## Stage 4 — Implement differentiable geometry and KING/ReGentS costs (Completed)

Completed (2026-09-03): Implemented exact differentiable oriented-box geometry in `pufferlib/ocean/regents/geometry.py` and the three independent paper costs plus their weighted composition in `pufferlib/ocean/regents/losses.py`.

The WOSAC audit retained only its low-level box-corner, Minkowski-sum, and convex-polygon distance primitives. ReGentS does not use WOSAC's rounded-box approximation, evaluation-shaped masking, in-place operations, or all-pairs allocation. Its sharp-box distance is positive for separation, zero at contact, and negative for penetration; randomized tests match the C simulator's four-axis SAT collision sign away from numerical boundaries. Background pairs are evaluated in deterministic chunks of 4,096 pairs, bounding temporary geometry memory by `O(chunk_size * timesteps)` while preserving the exact hard-min reduction.

The cost contracts are:

1. ego collision induction computes each candidate's signed box distance to the sole ego at jointly valid timesteps, averages using that candidate's valid denominator, then takes the hard minimum over candidates;
2. background collision avoidance computes `-min(min(1.25 m, signed_distance))` over distinct jointly valid background pairs and timesteps, returning the neutral value zero when fewer than two backgrounds have a jointly valid pair;
3. drivable-area deviation sums the four sampled corner potentials for each optimized vehicle, averages each vehicle over only its valid timesteps, then sums optimized vehicles as in the paper's actor/corner sums.

Masked storage is replaced with non-degenerate internal geometry placeholders before box operations, then excluded before every reduction; placeholders never contribute a value or denominator. This prevents invalid padded boxes from producing NaN gradients without repairing or treating invalid scenario state as observed.

The map-static potential starts from `~drivable_mask`, applies a normalized Gaussian with configurable sigma and cutoff, and adds an out-of-bounds frame. Sampling uses the Stage 1 centered world transform, `grid_sample(..., align_corners=True)`, bilinear interpolation, and border padding, so points beyond raster coverage are explicitly out of bounds. `prepare_out_of_bounds_rasters()` constructs these fields once before optimization and transfers them to the optimization device/dtype.

Named reference defaults follow the bundled ReGentS implementation: ego/background/drivable weights `1 / 5 / 20`, background truncation `1.25 m`, and Gaussian sigma `0.5 m`. The finite Gaussian support is explicitly set to `3 sigma`; this replaces the reference code's crop-based approximation with deterministic normalized convolution and is part of the cost configuration.

Tests cover hand-computed separation, penetration and contact; rotated and randomized SAT cases; validity and denominator behavior; world/grid transforms and outside-map sampling; Gaussian bounds and spatial gradients; geometry and raster `gradcheck`; finite differences; independent finite nonzero gradients for all three costs; acceleration and steering gradients through `classic_rollout`; CPU/GPU consistency checks; and a 20-step synthetic Adam fixture whose combined loss decreases. A canonical real replay scenario also produces finite values through the complete adapter/raster/cost path. The Stage 4 focused suite passes on CPU; CUDA consistency tests are present and skip when CUDA is unavailable. M2 is complete, unlocking Stage 5 frozen-ego generation.

## Stage 5 — Add ReGentS selection, constraints, and frozen-ego optimization (Completed)

Completed (2026-09-03): Implemented deterministic candidate selection in `pufferlib/ocean/regents/filters.py` and one-scenario frozen-ego Adam optimization in `pufferlib/ocean/regents/optimizer.py`. `capture_frozen_idm_trajectory()` resets a directly instantiated Drive configured with native C IDM for stable agent zero and replay for backgrounds, captures detached centered ego states, and feeds the same `ScenarioBatch` and frozen trajectory into the optimizer. Logged ego states remain an explicit `logged_fixture` source for deterministic tests.

Candidate selection records reason bits rather than silently compacting agents. It excludes the SDC, non-vehicles, insufficient transition coverage, static actors, the paper's non-actionable rear sector, caller-labeled unsuitable scenes, and original collisions. Original collision labels use exact oriented-box overlap for every jointly valid metadata-bearing actor pair. The named reference defaults are at least `50%` and one valid transition, static first-to-last displacement below `0.2 m` or maximum absolute speed below `0.2 m/s`, and rear occupancy strictly greater than `80%` within `pi/8` of directly behind the ego. The optimization horizon is explicit and all fractions use only the applicable jointly valid states or transitions.

Stage 3 actions initialize the result for every vehicle, while only selected, valid action entries enter the differentiable graph. The frozen ego and non-selected actors stay on their captured/logged reference trajectories during an optimization block. Adam defaults to learning rate `1e-3`, betas `0.9 / 0.999`, epsilon `1e-8`, and 500 updates; actions are projected to `[-1, 1]` and non-candidate or invalid entries are restored byte-for-byte after every update. The zero-speed magnitude calculation in `classic_step()` now uses a forward-equivalent clamp at the dtype's smallest normal value, defining a finite derivative at an exactly stationary state without changing Stage 2 C parity.

Front divergence uses wrapped ego-relative bearing and yaw. Both must lie strictly inside the paper's `(-pi/8, pi/8)` applicability bounds, the bearing and yaw must be on the same side, and the bearing magnitude must be smaller than the yaw magnitude. Steering updates are canceled when red-zone occupancy is strictly greater than the named `tau_front=0.5`; acceleration updates remain active. Adam steering moments are cleared for canceled entries so momentum cannot bypass the rule.

Success requires a discrete-timestep ego/candidate oriented-box overlap with no background/background overlap and no newly introduced off-road vehicle corner. Existing raster mismatch on the logged/reference trajectory is retained as the feasibility baseline rather than retroactively declaring valid source data off-road; only new corner violations are rejected. The optimizer stops on the first feasible collision by default, supports named collision and stagnation early-stop settings, and otherwise returns the lowest-total-cost feasible iterate. It records selection/filter reasons, stable adversary index and ID, initial/final component costs, total/acceleration/steering gradient norms, front-divergence iterations, action saturation, maximum inverse reconstruction residual, collision timestep, constraint rejection counts, iteration counts, frozen-ego source, failure reason, and deterministic seed.

Tests cover every filter, strict positive and negative angular boundaries, the front-divergence steering/acceleration split, native C IDM capture, braking and merging collision generation, rejection of background-collision and newly off-road iterates, exact preservation of frozen actions, and repeat determinism. The fixed NuPlan scenarios at map indices 5 and 8 with seeds 47 and 50, a 16-transition horizon, five Adam updates, and learning rate `1e-3` remain finite and deterministic. Their ego collision costs decrease from `2.0703814` to `1.9595402` and `5.6210895` to `5.5688214`; total costs decrease from `69.4004211` to `68.1501007` and `19.9497566` to `18.9804382`. The complete ReGentS suite passes with 75 tests and two expected CUDA skips. M3 is complete, unlocking Stage 6 C replay.

## Stage 6 — Replay in C, add reactive ego iteration, and integrate the pipeline (Completed)

Completed (2026-09-03): Implemented stable-index action injection in the C simulator, authoritative C replay and the reactive IDM loop in `pufferlib/ocean/regents/rollout.py`, portable artifacts in `pufferlib/ocean/regents/artifacts.py`, and an offline generation entry point in `pufferlib/ocean/regents/generation.py` behind `puffer regents`.

`regents_set_action_plan` installs one `[stable_agent, transition, 2]` plan plus its mask, rejecting non-replay, non-vehicle, ego, non-finite, and out-of-range entries, and requiring replay mode with continuous classic dynamics on exactly one environment. `c_step` applies the plan in the expert-static loop by stable simulator index, never by a compacted batch position. `regents_get_events` returns the timestep, the simulator's own moving-OBB collision pairs, and per-agent off-road flags; injected actors get `compute_metrics` against a scratch log so they carry authoritative infraction flags without entering policy episode logs or rewards.

`start_regents_injection` seeds an adversary's first injected transition from its logged state: neutral wheel steering, the logged dimensions and wheelbase, the logged yaw rate, and the logged longitudinal velocity projected on the agent heading. Three C-side gaps that only surface under injection were found and fixed here. `c_reset` returns early whenever `timestep == init_step`, so `set_start_position` cannot be relied on to prepare an injected agent; seeding at the transition is the only order-independent mechanism. Agents invalid at `init_step` never reach `generate_reward_coefs`, leaving `reward_coefs[REWARD_COEF_SPEED]` at zero and clipping the adversary to a zero speed limit, so injection sets the neutral coefficient the exported `maximum_speed_mps` already assumes. Replay actors carry no wheel steering or classic-dynamics speed state at all.

The speed conventions are deliberately different on the two sides and must stay that way. Logged and Torch states use the bicycle model's longitudinal speed, the heading projection of the logged velocity; this is what Stage 1 exports and what the Stage 3 gates were measured against. C's `sim_speed_signed` is the velocity magnitude signed by heading agreement, which is exact for states C integrated itself, so `signed_speed_from_c_velocity()` in `state.py` is used only when reading C state back. The two agree exactly once injection integrates a state and differ by the logged slip angle before it, so speed and steering are compared only at states injection actually produced. Position and heading are compared for every jointly valid agent, the shared initial state is verified rather than scored, and the comparison truncates at the first C collision because collision and infraction responses intentionally change C state.

C is the success oracle, measured against a baseline C rollout of the same scenario, seed, and horizon driven by the Stage 3 initial actions under the same mask. Only collision pairs and off-road flags absent from that baseline are attributed to the optimization, so pre-existing logged overlaps neither fail generation nor inflate the background-collision rate. Success requires a new actionable ego/adversary collision with no new background collision and no new adversary off-road. A reactive ego answers the new adversary, so its divergence from the frozen reference is reported as `maximum_ego_reference_error` and excluded from the parity gate whenever the ego is not replay-controlled.

`run_reactive_idm_generation()` captures a native C IDM ego, detaches it, optimizes adversaries in Torch, reruns C for IDM's response, and repeats to a fixed maximum outer iteration count, stopping on the first C-confirmed success or on unchanged actions. No gradient crosses the C simulator or the ego controller. Replacing IDM with a learned policy needs no change to this interface.

Artifacts are pickle-free `npz` files carrying a schema tag, scenario and dataset identity, the source map path and a SHA-256 hash over the canonical configuration plus the exact map bytes, masks, initial and optimized actions, Torch and C trajectories, C ego actions, both replay metadata blocks, and every optimization and replay metric. `save_generation_artifact()` writes atomically and returns exactly the persisted metadata; `load_generation_artifact()` rejects unknown schemas, mismatched fields, and oversized arrays.

`puffer regents <env_name> <generation_name>` reads `pufferlib/config/evaluation/regents.yaml`, validates it against the `Drive` signature, instantiates `Drive` directly, and never initializes PPO, a policy optimizer, a rollout buffer, or training logging. It writes one artifact per scenario plus `generation_metrics.csv`, and reports generation success, ego, actionable, background-collision and off-road rates, maximum C/Torch trajectory error, optimization runtime, and rejection reasons.

Visualization reuses the existing viewer rather than adding one. With `render_replays: true`, both C rollouts also capture the simulator's own HTML replay frames through `Drive.get_obs_html_frame()` plus the ego observations, and `render_scenario_replays()` hands them to `pufferlib.viz.save_interactive_replay_zlib()` and `render_interactive_replay_zlib()`. Each scenario yields a `.logged` and an `.adversarial` page under `rendered_replays/`, indexed by `pufferlib.viz.build_gallery_index()`. The ego's captured C actions fill the viewer's action channel; the pages are self-contained and reference no external resource. Capture is off by default so a generation run pays nothing for it.

The recorded acceptance run is `regents_nuplan`: the first 16 local NuPlan maps, seeds `42..57`, `init_step=0`, `dt=0.1`, a 16-transition horizon, 500 Adam updates at learning rate `1e-3`, and at most 3 outer iterations. Maximum C/Torch trajectory error was `1.526e-5` across all 16 scenarios, against the mandatory `1e-4` Stage 2 tolerance; the per-scenario maximum never exceeded `1.53e-5` and eleven scenarios were below `1.2e-7`. Total optimization time was 591.6 s. Generation success was `1/16`. Map index 8 with seed 50 produced a C-confirmed actionable ego collision at timestep 11 against adversary index 11, with no background collision and no new off-road corner. The rejections were 13 scenes filtered as `original_collision,no_candidate`, one `initial_reconstruction_collision`, and one `iteration_limit`. The run also rendered 32 replay pages and a gallery index under `experiments/regents/regents_nuplan/rendered_replays/`.

The exit gate passes: optimized actions reproduce in C within the Stage 2 tolerance until a collision response changes the state, C confirms the intended collision, and original and adversarial replays plus every metric regenerate from the saved artifact. M4 and M5's IDM half are complete; learned-policy parity remains.

**Open issue for Stage 5, not a Stage 6 blocker.** Thirteen of sixteen NuPlan scenes are filtered as having an original collision with no remaining candidate, so the offline success rate is bounded by the filter rather than by the optimizer. The C baseline shows the same maps carry many jointly valid overlapping logged boxes, so exact oriented-box overlap on logged data is likely too strict for this dataset. Decide whether the original-collision filter should use a penetration-depth threshold, exclude only the specific colliding pair rather than the scene, or keep exact overlap and accept the yield, before quoting a generation success rate as a method result.

### Original Stage 6 scope


The optimized trajectory is only accepted if it reproduces in the C simulator.

Open-loop replay:

- Add the minimum mixed-controller/action-injection path needed to keep ego on logged replay while applying optimized continuous actions to selected background vehicles. Map actions by stable simulator index, never by an incidental compacted batch position.
- Reset to the exact scenario, seed, and initial state used by Torch, replay the action sequence, and capture complete C states after every step.
- Compare C and Torch trajectories, collision timestep/pair, off-road events, and background-background collisions. Use C results as the authoritative success label, including its swept-collision behavior.
- Save a compact generation artifact containing scenario identity, source configuration/hash, masks, initial and optimized actions, Torch and C metrics, and failure reason. Save original and adversarial replay metadata for existing visualization tools.

Reactive ego support:

- First configure PufferDrive with `sdc_controller=idm`, run the IDM-controlled ego in C, and capture its actions/trajectory.
- Detach that ego trajectory and optimize adversaries in Torch for an iteration block.
- Reset and rerun C with the new adversary actions to obtain IDM's response.
- Repeat the alternating C-rollout/Torch-optimization loop with a fixed maximum number of outer iterations and deterministic stopping criteria. Do not backpropagate through the C simulator or the ego policy.
- Once the IDM loop is validated, replace IDM with the learned PufferDrive ego policy without changing the stop-gradient interface or optimizer contracts.

Pipeline integration:

- Begin with an offline generation entry point and a dedicated config under the evaluation configuration tree. It must instantiate `Drive` directly and must not initialize PPO, an optimizer for policy parameters, a training rollout buffer, or training logging.
- Adam in this pipeline optimizes only temporary adversary action tensors for the current scenario; it does not train or update an ego policy.
- When a learned ego policy is added later, load it in inference/evaluation mode and run it without gradients. The only autograd graph is the Torch adversary dynamics/loss path.
- Add generated artifacts as an explicit replay/evaluation dataset input.
- Report generation success rate, ego collision rate, actionable collision rate, off-road rate, background-background collision rate, C/Torch trajectory error, optimization runtime, and rejection reasons.
- Consumption by PPO training jobs or online generation during training is outside this plan and requires a separate explicit design decision after the evaluation pipeline is stable.

Exit gate: optimized actions reproduce in C within the Stage 2 tolerance until any intentional collision/infraction response changes the state, C confirms the intended collision, and original/adversarial replays plus metrics can be regenerated from the saved artifact.

### Design Principle: Native Simulator Integration & Minimal External Files

To maintain simplicity, reliability, and ease of maintenance, we prioritize running all ego controller execution directly inside the existing C simulator, avoiding the creation of unnecessary python files or layers:

- **Unified Controller Routing:** The simulator already natively supports routing controllers like `CONTROLLER_IDM`, `CONTROLLER_POLICY`, and `CONTROLLER_REPLAY` via `env->sdc_controller` and `env->non_sdc_controller`. This is processed inside the C simulator's `resolve_agent_controller()` and step loops (`drive.h`).
- **No External Wrapper Overhead:** Instead of constructing external Python-side wrappers or custom state machines to switch between IDM and policy models, we configure the existing variables in the simulation environment (e.g. configuring `sdc_controller` directly as `"idm"` or `"policy"`) and let the C simulator execute the controllers internally.
- **Minimalistic Code Footprint:** Avoid generating supplementary python classes or controller files. Keep the integration centered on the existing `pufferlib/ocean/drive` infrastructure, reusing the simulator's built-in hooks.

## Proposed code organization

```text
pufferlib/ocean/regents/
    __init__.py
    state.py
    adapter.py
    dynamics.py
    inverse_dynamics.py
    geometry.py
    losses.py
    filters.py
    optimizer.py
    rollout.py
    artifacts.py
    generation.py

tests/regents/
    test_stage0_idm.py
    test_adapter.py
    test_dynamics_parity.py
    test_inverse_dynamics.py
    test_geometry.py
    test_losses.py
    test_filters.py
    test_optimizer.py
    test_c_replay.py     # Stage 6 replay, reactive loop, artifacts, entry point

pufferlib/config/evaluation/
    benchmark.yaml  # regents_idm Stage 0 benchmark
    regents.yaml    # Stage 6 offline generation config, run by `puffer regents`
```

Keep binding changes in the existing Drive binding files and keep shared mathematical constants synchronized from one documented source. Avoid importing ReGentS into the PPO hot path until the offline pipeline is accepted.

## Milestones and stop/go order

| Milestone | Required result | Unlocks |
| --- | --- | --- |
| S0 | Deterministic replay rollout with an IDM-controlled SDC and logged backgrounds | Scenario extraction |
| M0 | Complete, deterministic scenario export and Torch adapter | Dynamics work |
| M1 | `C classic rollout ~= Torch rollout` and inverse/forward reconstruction validated on real scenarios | Loss/optimization work |
| M2 | Independently validated differentiable costs and geometry | Fixed-ego generation |
| M3 | Stable ReGentS optimization against a frozen logged/IDM ego rollout | C replay |
| M4 | C reproduces optimized scenarios and authoritative metrics | Reactive ego loop |
| M5 | Deterministic reactive IDM generation and saved artifacts; then learned-policy parity | Training-pipeline experiments |
| M6 | Batching and parallelization of optimization jobs | Multi-scenario generation |

M1 is the first ReGentS model milestone and is a hard gate. In particular, do not tune loss weights to compensate for a dynamics or inverse-dynamics mismatch.

S0 through M4 are complete. M5's reactive IDM half is complete; its learned-policy parity half and M6 batching remain.

## Known risks to resolve early

- `get_state()` is intentionally broad and Python-list based. It is suitable for initial offline extraction, but a compact NumPy binding may be needed later if profiling shows serialization to be a bottleneck.
- `simulation dt` and logged trajectory `log_dt` are distinct fields. A mismatch invalidates inverse dynamics unless explicitly resampled.
- Existing accessors return world coordinates by adding map means, while C dynamics operate in centered coordinates. The adapter must choose one frame and transform state and map data together.
- The current road-edge accessor has incomplete vectorized scenario IDs, and road edges may not encode the same drivable-area semantics as the ReGentS out-of-bounds raster.
- Logged trajectories need not be exactly realizable by PufferDrive classic dynamics. Real-data reconstruction thresholds must be measured and reported rather than assumed to be `1e-4`; the `1e-4` target remains mandatory for C-generated parity fixtures.
- Hard `min`, clipping, action saturation, and collision boundaries are only piecewise differentiable. Tests must cover useful gradient flow near, but not exactly on, these boundaries.
- Torch uses discrete-timestep box distances while C also checks swept OBB collisions. Optimization loss and authoritative acceptance therefore have intentionally different roles.
- Pairwise geometry is potentially `O(agents^2 * timesteps)` in memory. Establish correctness with one scenario first, then profile and chunk without changing reduction semantics.
