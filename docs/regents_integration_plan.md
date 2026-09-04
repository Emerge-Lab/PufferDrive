# ReGentS integration plan for PufferDrive 3.0

## Goal and scope

Implement support for ReGentS in PufferDrive. Stages 0-6 delivered the offline POC; stages 7-9 carry it to the ultimate goal below. The existing C simulator remains the source of truth for dynamics, collision outcomes, and final evaluation. A separate PyTorch path provides differentiable dynamics and costs for adversarial action optimization.

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

## Core contracts

One explicit Torch representation throughout the differentiable path:

- state tensors `[batch, agent, time, feature]`: scenario-local centered `x`, `y`, wrapped heading, signed speed, steering angle;
- action tensors `[batch, agent, time - 1, 2]`: normalized acceleration and target steering in `[-1, 1]`;
- metadata: agent type, stable agent ID/index, SDC mask, length, width, wheelbase, per-agent maximum speed, scenario ID, timestep;
- masks: state validity, transition validity (`valid[t] & valid[t + 1]`), vehicle, ego, candidate adversary, optimized action;
- map data: a documented world-to-grid transform plus a drivable-area raster or the geometry to build one deterministically.

APIs preserve agent identity and use one documented coordinate frame. Invalid entries are masked, never silently repaired. External scenario data and tensor shapes are validated before optimization starts.

## Design principle: native simulator integration

Ego controller execution stays inside the C simulator. `resolve_agent_controller()` already routes `CONTROLLER_IDM`, `CONTROLLER_POLICY`, and `CONTROLLER_REPLAY` from `env->sdc_controller` / `env->non_sdc_controller`, so switching controllers is a config value, never a Python wrapper or state machine. Keep the footprint inside the existing `pufferlib/ocean/drive` infrastructure, keep binding changes in the existing Drive binding files, and keep shared mathematical constants synchronized from one documented source.

## Completed stages 0-6

All completed 2026-09-03. Code lives in `pufferlib/ocean/regents/`, tests in `tests/regents/`.

### Stage 0 — IDM as the ego controller (`a68e49f7`)

Native C IDM benchmark config `regents_idm` in `benchmark.yaml`. Deterministic routing, logged background motion, and evaluation metrics verified without any Torch policy or checkpoint. NuPlan baseline: 100 scenarios, seed `42`, `init_step=0`. Tests: `test_stage0_idm.py`.

### Stage 1 — Scenario export and Torch adapter (`80d7b938`)

`export_drive_scenarios()` in `adapter.py` builds canonically padded batches from `Drive.get_state()`. `state.py` holds the dataclasses: centered Cartesian frame, stable agent indexing, `EGO_IDX == 0` SDC masking, signed speed from heading projection, first-valid agent dimensions via a shared `WHEELBASE_LENGTH_RATIO`. `DrivableAreaRaster` / `RasterTransform` map centered simulation space to grids. Tests cover shapes, padding, dtypes, transform consistency, and byte-identical repeated exports.

### Stage 2 — Differentiable classic dynamics and C parity (`7de36bb5c`)

`classic_step()` / `classic_rollout()` in `dynamics.py` follow the C `move_dynamics()` operation order exactly: action scaling, steering rate and angle limits, signed speed clipping, slip angle, position integration at `old_heading + beta`, heading wrap, carried actual steering. A diagnostic binding runs the authoritative C step in isolation with stochastic and infraction behavior disabled.

Strict parity horizon is 64 transitions. Maximum absolute error over every state component at every step:

| Case | Error |
| --- | --- |
| One-step neutral / brake / accelerate / reverse / steering limit / speed limit / heading wrap | `1.7882e-7` |
| 64 transitions, random actions, 2 batches x 3 agents | `3.8147e-5` |
| 32 inverse-derived actions from a real replay scenario | `3.0923e-11` |

All pass the mandatory `1e-4` threshold. A 256-transition stress test reached `1.6404e-4` from accumulated C-libm versus Torch transcendental rounding: **any horizon beyond 64 transitions must establish its own parity approach and threshold before use.**

### Stage 3 — Inverse dynamics (`b6f93aa7b`)

`estimate_expert_actions()` in `inverse_dynamics.py`. Acceleration comes from consecutive signed speeds and first minimizes reachable speed error; steering is analytically seeded from wrapped heading change, then refined by a deterministic bounded search minimizing squared position error plus wheelbase-scaled squared heading error. Steering carries within a contiguous run, resets to neutral at an unobserved run start, and is never inferred across a validity gap. Every inconsistent transition retains its component errors and a combined meter-scaled residual. Offline generation bounds reconstruction to its configured optimization horizon, and the reactive loop reuses that exact prefix result across outer blocks; reconstructing unused logged suffixes cannot affect the prefix and only adds work.

ReGentS itself delegates this to Waymax's expert actor, so only the semantics are shared (consecutive-state local derivatives, wrapped yaw differences, action bounds, validity masking, the `0.6 m/s` low-speed noise guard). The equations must differ: Waymax controls unsigned acceleration and curvature with trapezoidal integration; PufferDrive controls signed acceleration and a rate-limited target wheel angle, updates speed before Euler integration, and uses wheelbase plus slip angle. Copying Waymax's curvature inverse would fail Stage 2.

Real-data audit: first 16 local NuPlan scenarios, seeds `42..57`, `init_step=0`, `dt=0.1`, 67,111 valid vehicle transitions (27,339 normal-speed, 39,772 low-speed). `P50 / P90 / P95 / P99 / max`:

| Split | Position (m) | Heading (rad) | Speed (m/s) |
| --- | --- | --- | --- |
| Normal speed | `0.04252 / 0.16211 / 0.23694 / 0.47635 / 2.10607` | `0.001442 / 0.006005 / 0.009119 / 0.020494 / 0.440930` | `0 / 9.54e-7 / 9.54e-7 / 0.09736 / 4.25930` |
| Low speed | `0.01684 / 0.05301 / 0.07534 / 0.17265 / 3.18633` | excluded; diagnostic only | `0 / 0 / 0 / 1.16e-10 / 8.53201` |

Per-timestep mean ranges: `0.03525..0.05448 m`, `0.001934..0.006410 rad`, `4.69e-8..0.03005 m/s`. Acceptance gates — normal-speed P95 `<= 0.25 m` / `<= 0.012 rad` / `<= 1e-5 m/s`; low-speed P95 `<= 0.08 m` / `<= 1e-5 m/s`; maximum per-timestep means `<= 0.06 m` / `<= 0.007 rad` / `<= 0.04 m/s`. Maxima are diagnostic, not gates: logged transitions can violate acceleration and steering constraints, and those outliers stay reported and masked only by true validity.

Both M1 hard gates pass.

### Stage 4 — Differentiable geometry and costs (`2cbd9322c`)

Exact oriented-box geometry in `geometry.py`, the three paper costs and their weighted composition in `losses.py`. The WOSAC audit kept only its box-corner, Minkowski-sum, and convex-polygon distance primitives; ReGentS uses none of its rounded-box approximation, evaluation-shaped masking, in-place ops, or all-pairs allocation. Signed distance is positive for separation, zero at contact, negative for penetration, and matches the C four-axis SAT collision sign away from numerical boundaries. Background pairs evaluate in deterministic 4,096-pair chunks, bounding temporary memory at `O(chunk_size * timesteps)` without changing the exact hard-min reduction.

Cost contracts:

1. **Ego collision induction** — each candidate's signed box distance to the sole ego at jointly valid timesteps, averaged over that candidate's valid denominator, then hard minimum over candidates.
2. **Background collision avoidance** — `-min(min(1.25 m, signed_distance))` over distinct jointly valid selected-adversary pairs and timesteps; neutral zero when fewer than two selected adversaries share a valid pair. This matches the reference implementation's `adv_idx` input rather than regularizing replay-only actors.
3. **Drivable-area deviation** — each optimized vehicle's four sampled corner potentials, summed over corners, averaged over that vehicle's valid timesteps, then summed over optimized vehicles. The reference `calculate_potential_adv_dev` reduces with `sum`, but its cropped Gaussian density is not normalized; PufferDrive uses a normalized full-raster convolution and keeps this term horizon-invariant and comparable to the time-averaged ego cost.

Masked storage becomes non-degenerate internal placeholders before box operations and is excluded before every reduction, so invalid padded boxes never produce NaN gradients and never contribute a value or denominator.

The map-static potential starts from `~drivable_mask`, applies a normalized Gaussian, and adds an out-of-bounds frame; sampling uses the Stage 1 centered transform with `grid_sample(..., align_corners=True)`, bilinear interpolation, and border padding, so points beyond raster coverage are explicitly out of bounds. `prepare_out_of_bounds_rasters()` builds these once per optimization.

Reference defaults: ego/background/drivable weights `1 / 5 / 20`, background truncation `1.25 m`, Gaussian sigma `0.5 m`, finite support `3 sigma` (deterministic normalized convolution replacing the reference code's crop approximation, and part of the cost configuration). The transferred `20` coefficient is not yet calibrated for this raster. On smoke scenario 1 at 50 transitions, the initial boundary cost is `2.483`, or `0.02587` mean potential per corner across 24 selected vehicles; its weighted contribution is still `49.667`, versus `0.802` ego collision and `-1.879` background collision. Report both the configured raw reduction and a normalized per-corner diagnostic when tuning this coefficient.

Tests: hand-computed separation/penetration/contact, rotated and randomized SAT, denominators, transforms and outside-map sampling, Gaussian bounds and spatial gradients, `gradcheck` and finite differences, independent nonzero gradients for all three costs, gradients through `classic_rollout`, CPU/GPU consistency, a 20-step Adam fixture with decreasing loss, and a real replay scenario through the full adapter/raster/cost path. CUDA tests skip when unavailable.

### Stage 5 — Selection, constraints, frozen-ego optimization (`389fc411d`)

`filters.py` selects candidates deterministically; `optimizer.py` runs one-scenario frozen-ego Adam. `capture_frozen_idm_trajectory()` resets a directly instantiated Drive with either native C IDM or replay on stable agent zero and replay backgrounds, then captures detached centered ego states. `logged_fixture` remains an explicit source for deterministic tests.

Selection records reason bits rather than compacting agents, excluding: SDC, non-vehicles, insufficient transition coverage, static actors, the paper's non-actionable rear sector, caller-labeled unsuitable scenes, and original collisions. Original-collision labels mirror the reference `overlap_with_ego` — per metadata-bearing agent, exact oriented-box overlap against the sole ego at jointly valid timesteps. Background/background overlap is not labeled. A labeled agent is dropped as a candidate; the scene is filtered `ORIGINAL_COLLISION` only when every candidate-masked agent carries the label. Defaults: at least `50%` and one valid transition; static below `0.2 m` first-to-last displacement or `0.2 m/s` maximum absolute speed; rear occupancy strictly above `80%` within `pi/8` of directly behind the ego. All fractions use only applicable jointly valid states or transitions.

Stage 3 actions initialize every vehicle; only selected valid entries enter the graph. The frozen ego and non-selected actors hold their reference trajectories. Adam defaults: learning rate `1e-3`, betas `0.9 / 0.999`, epsilon `1e-8`, 500 updates. Actions project to `[-1, 1]`, and non-candidate or invalid entries are restored byte-for-byte after every update. `classic_step()`'s zero-speed magnitude uses a forward-equivalent clamp at the dtype's smallest normal value, giving a finite derivative at an exactly stationary state without changing Stage 2 parity.

Front divergence uses wrapped ego-relative bearing and yaw with separate windows, as in the reference `opt()`: bearing strictly inside `(-pi/8, pi/8)`, yaw strictly inside `(-pi/2, pi/2)`, on the same side, with bearing magnitude below yaw magnitude. Steering updates cancel when red-zone occupancy exceeds `tau_front=0.5`; acceleration updates continue, and Adam steering moments are cleared for canceled entries so momentum cannot bypass the rule. Every surviving steering update is scaled by `steering_update_scale`, applied to the post-Adam update rather than the gradient because Adam's per-parameter normalization makes gradient scaling a no-op. The reference uses `0.5`, damping steering relative to acceleration; PufferDrive defaults to `4.0`.

The reference value is not transferable because the action spaces carry different units: Waymax controls curvature bounded at `0.3 1/m`, PufferDrive a wheel angle bounded at `0.667 rad`. The scale is therefore a steering-only step size, equivalent to a per-channel learning rate that leaves acceleration at `learning_rate`. A scale above one extrapolates past the Adam step, so the update is re-clamped to `[-1, 1]` after the blend; without that clamp `4.0` drives actions out of contract.

Measured over the 25-scenario NuPlan set at `lr=1e-3`, 500 iterations, everything else fixed:

| `steering_update_scale` | Success | `iteration_limit` | Background collisions | Off-road |
| --- | --- | --- | --- | --- |
| `0.5` (reference) | `4/25` | 15 | 4 | 2 |
| `1.0` | `5/25` | 14 | 4 | 1 |
| `2.0` | `5/25` | 14 | 4 | 1 |
| **`4.0`** | **`8/25`** | **11** | 4 | 1 |
| `8.0` | `7/25` | 12 | 4 | 1 |

Success doubles at `4.0` with no increase in constraint violations, and the curve is unimodal with a mild decline at `8.0`. At `0.5` the peak steering excursion was only `3.5-15.8` degrees of wheel angle over a full 8 s horizon; at `4.0` it is `11.3-34.4` degrees. Caveat: `n = 25`, so treat the exact optimum as provisional.

Success requires a discrete-timestep ego/candidate box overlap with no newly introduced collision in a candidate-involved background pair and no newly introduced off-road corner. Existing background overlaps and raster mismatch on the reference trajectory are pair-level and agent-level feasibility baselines, matching the C oracle, so valid source data is never retroactively rejected. Pair signatures are evaluated in deterministic chunks without Python scalar geometry loops. The optimizer stops on the first feasible collision by default, supports named collision and stagnation early-stops, and otherwise returns the lowest-total-cost feasible iterate. It records filter reasons, adversary index and ID, initial/final component costs, gradient norms, front-divergence iterations, action saturation, maximum inverse reconstruction residual, collision timestep, rejection counts, iteration counts, frozen-ego source, failure reason, and seed.

Determinism fixture — NuPlan map indices 5 and 8, seeds 47 and 50, 16-transition horizon, five Adam updates at `1e-3`. The recorded totals predate the `0.5` steering damping and must be re-measured before being used as a gate.

### Stage 6 — C replay, reactive ego, pipeline (`3734f6373`)

Stable-index action injection in C, authoritative replay and the reactive loop in `rollout.py`, artifacts in `artifacts.py`, offline generation in `generation.py` behind `puffer regents`.

**C injection.** `regents_set_action_plan` installs one `[stable_agent, transition, 2]` plan plus mask, rejecting non-replay, non-vehicle, ego, non-finite, and out-of-range entries, and requiring replay mode with continuous classic dynamics on exactly one environment. `c_step` applies it in the expert-static loop by stable simulator index, never a compacted batch position. `regents_get_events` returns the timestep, the simulator's moving-OBB collision pairs, and per-agent off-road flags; injected actors run `compute_metrics` against a scratch log so they carry authoritative infraction flags without entering policy episode logs or rewards.

`start_regents_injection` seeds an adversary's first injected transition from its logged state: neutral wheel steering, logged dimensions and wheelbase, logged yaw rate, and logged longitudinal velocity projected on the agent heading. Three C-side gaps surface only under injection and were fixed here — `c_reset` returns early when `timestep == init_step`, so `set_start_position` cannot prepare an injected agent and seeding at the transition is the only order-independent mechanism; agents invalid at `init_step` never reach `generate_reward_coefs`, leaving `reward_coefs[REWARD_COEF_SPEED]` zero and clipping the adversary to a zero speed limit, so injection sets the neutral coefficient the exported `maximum_speed_mps` already assumes; replay actors carry no wheel steering or classic-dynamics speed state at all.

**Speed conventions differ by design and must stay that way.** Logged and Torch states use the bicycle model's longitudinal speed (the heading projection of logged velocity) — what Stage 1 exports and what the Stage 3 gates measured. C's `sim_speed_signed` is velocity magnitude signed by heading agreement, exact for states C integrated itself, so `signed_speed_from_c_velocity()` is used only when reading C state back. The two agree once injection integrates a state and differ by the logged slip angle before it, so speed and steering are compared only at injection-produced states. Position and heading are compared for every jointly valid agent, the shared initial state is verified rather than scored, and comparison truncates at the first C collision because collision and infraction responses intentionally change C state.

**C is the success oracle**, measured against a baseline C rollout of the same scenario, seed, and horizon driven by the Stage 3 initial actions under the same mask. Only collision pairs and off-road flags absent from that baseline are attributed to the optimization, so pre-existing logged overlaps neither fail generation nor inflate the background-collision rate. Success requires a new actionable ego/adversary collision, no new background collision, and no new adversary off-road. A reactive ego's divergence from the frozen reference is reported as `maximum_ego_reference_error` and excluded from the parity gate whenever the ego is not replay-controlled.

`run_reactive_idm_generation()` captures the configured C IDM or replay ego and detaches it before optimizing adversaries in Torch. With IDM it reruns C for the reactive response and repeats to a fixed maximum outer iteration count, stopping on the first C-confirmed success or unchanged actions. Replay is an explicit open-loop diagnostic mode and is forced to one outer block. No gradient crosses the C simulator or the ego controller; swapping in a learned policy needs no interface change.

Artifacts are pickle-free `npz` files carrying a schema tag, scenario and dataset identity, source map path, a SHA-256 hash over the canonical configuration plus exact map bytes, masks, initial and optimized actions, Torch and C trajectories, C ego actions, both replay metadata blocks, and every metric. `save_generation_artifact()` writes atomically and returns exactly the persisted metadata; `load_generation_artifact()` rejects unknown schemas, mismatched fields, and oversized arrays.

`puffer regents <env_name> <generation_name>` reads `pufferlib/config/evaluation/regents.yaml`, validates it against the `Drive` signature, instantiates `Drive` directly, and never initializes PPO, a policy optimizer, a rollout buffer, or training logging. It writes one artifact per scenario plus `generation_metrics.csv`. The configured raster resolution is validated as finite and positive and passed through frozen-trajectory export; its default is the adapter's `0.5 m/pixel`, so configuration and actual loss geometry cannot silently differ.

Visualization reuses the existing viewer. With `render_replays: true` (off by default, so generation pays nothing for it), both C rollouts capture the simulator's HTML replay frames via `Drive.get_obs_html_frame()` plus ego observations, and `render_scenario_replays()` passes them to `pufferlib.viz`. Each scenario yields self-contained `.logged` and `.adversarial` pages under `rendered_replays/`, indexed by `build_gallery_index()`, with the ego's captured C actions in the viewer's action channel.

**Acceptance run `regents_nuplan`** — first 16 local NuPlan maps, seeds `42..57`, `init_step=0`, `dt=0.1`, 16-transition horizon, 500 Adam updates at `1e-3`, at most 3 outer iterations:

| Result | Value |
| --- | --- |
| Maximum C/Torch trajectory error | `1.526e-5` (gate `1e-4`; 11 of 16 below `1.2e-7`) |
| Generation success | `1/16` — map 8, seed 50, actionable ego collision at timestep 11 against adversary 11 |
| Rejections | 13 `scene_filtered:original_collision,no_candidate`, 1 `initial_reconstruction_collision`, 1 `iteration_limit` |
| Total optimization time | `591.6 s` |
| Rendered | 32 replay pages plus gallery index |

Exit gate passes: optimized actions reproduce in C within Stage 2 tolerance until a collision response changes state, C confirms the intended collision, and replays plus every metric regenerate from the saved artifact.

## Open issues carried out of stages 0-6

- **Filter yield bounds the result, not the optimizer.** 13 of 16 scenes filter as original-collision with no remaining candidate. The C baseline shows these maps carry many jointly valid overlapping logged boxes, so exact oriented-box overlap on logged data is likely too strict for NuPlan. Per-agent exclusion is already implemented; undecided is whether the ego-overlap test needs a penetration-depth threshold or a jointly-valid-coverage requirement, or whether the yield is accepted. Resolve before quoting a generation success rate as a method result.
- **Yield is horizon-specific.** The later `regents_nuplan_smoke` run at a 50-transition horizon selected adversaries on maps 0 and 1, which the 16-transition acceptance run filtered as `original_collision,no_candidate`, with no filter code change between them. Never quote a yield without its horizon.
- **Config drift.** `regents.yaml`'s `regents_nuplan` entry was edited after the acceptance run; its scenario count, horizon, Adam iteration count, and SDC controller differ from the acceptance settings. It is now a replay-ego open-loop run rather than a reactive-IDM run, so that name no longer reproduces the recorded result — only `experiments/regents/regents_nuplan/` holds it. Restore the settings under that name or move the record to a separate entry before citing the numbers again.
- **Full horizon unvalidated.** The `regents_nuplan_20s` entry (16 scenarios, 200 transitions, the complete 20 s window) exists but `experiments/regents/regents_nuplan_20s/` is empty — started, never completed. Every recorded result is at 16 or 50 transitions, at most 5 s of a 20 s scenario. Parity, yield, runtime, and success rate at full horizon are unmeasured.

### The drivable-area term was charging the map, not the optimization (2026-09-03)

`_rasterize_drivable_area()` paints drivable area as tubes of half `LANE_WIDTH_METERS` (`1.85 m`) around the centerlines of `LANE_FREEWAY` and `LANE_SURFACE_STREET` only. The reference builds its map from **road edges** (`datatypes.is_road_edge`) with a cross-product inside/outside test, which is a different and better-covering definition. The local NuPlan maps do carry `ROAD_EDGE_BOUNDARY` (type 21) geometry that is currently unused.

The centerline-tube definition systematically under-covers, and it charges the logged data. Measured on logged trajectories, per cent of valid box corners the raster calls off-road:

| Scenario | Drivable pixels | Ego corners off-road | Candidate corners off-road |
| --- | --- | --- | --- |
| 6 | `8.6%` | `75.0%` | `100.0%` |
| 4 | `16.1%` | `10.5%` | `25.2%` |
| 23 | `10.5%` | `0.0%` | `7.2%` |
| 0, 1, 5, 10, 17 | `10-18%` | `0.0%` | `0-2.5%` |

In scenario 6 the ego's own recorded path sits a median `3.93 m` from the nearest drivable lane centerline, more than twice the tube half-width. These are real recorded vehicles, so this is raster error, not agent behaviour.

Consequence: at weight `20` the drivable term was `52%` of the loss magnitude at iteration 0 on average and over `90%` in six of 25 scenarios, all of it a constant map-mismatch penalty present before any optimization. Its gradient pulls candidates toward lane centerlines rather than toward the ego, which starves the adversarial objective.

**Fix applied - charge only what the optimization introduces.** `drivable_area_deviation_cost()` takes an optional `baseline_corner_potential` and charges `relu(potential - baseline)`. The optimizer samples the baseline once from the reference rollout. This makes the loss agree with the success rule, which already counted only `offroad_signature & ~baseline_offroad_signature`; the two previously disagreed. Iteration-0 drivable cost is now exactly `0` in every scenario.

Result over the 25-scenario set: success `3/25 -> 4/25`, `iteration_limit` `18 -> 15`. Scenario 1 becomes a success; scenarios 11 and 12 move from `iteration_limit` to `c_background_collision`, so background collisions rose `1 -> 3` as the optimizer began pushing harder. The background weight is the next coefficient to look at.

**This is mitigation, not the root fix.** The raster still mislabels drivable area, which continues to distort the term's gradient wherever the map is wrong, and still drives the C off-road oracle. Rebuilding the raster from road edges as the reference does remains open, and is the risk already recorded under "road edges may not encode the same drivable-area semantics".

### Gradient audit and the zero-speed BPTT truncation (2026-09-03)

**Direction is correct, verified against finite differences.** On scenario 0 (map 0, seed 42) the ego is stationary for the whole episode and adversary 6 sits `8.9 m` directly ahead, same heading, also parked and drifting away. The only way to collide is to reverse into the ego. `dL/d(accel) > 0` at all 80 transitions, so gradient descent lowers acceleration and backs the adversary into the ego. Steering gradient is zero while the adversary is parked, which is correct in both formulations: yaw rate is proportional to speed, so steering cannot act at zero speed.

**Bug found and fixed.** `_classic_step()` mirrored C's `update_agent_speed()`, recovering signed speed as `sqrt(vx^2 + vy^2)` resigned onto the heading. Both `clamp_min` and the `where` guarding it are flat at an exactly stationary agent, so `d(signed_speed)/d(speed) = 0` there and the state recurrence was cut: the rollout gradient collapsed to its single-step direct effect. Measured on an isolated 20-transition rollout from `v0 = 0`, `d(final_x)/d(a[0])` was `0.04` against a true `0.80` — short by exactly the horizon. On scenario 0 the accel gradient was `41x` too small, `(N+1)/2` for `N = 80`.

That round trip is the identity on speed, so the fix carries the C value forward and routes the derivative through `speed`: `signed_speed = speed + (c_signed_speed - speed).detach()`. Forward values are bit-identical and all 16 C-parity tests still pass. Scenario 0 analytic-versus-finite-difference agreement went from `97%` error to under `0.5%` at every acceleration transition. Regression test: `test_stationary_agent_backpropagates_through_the_whole_horizon`.

**Impact was small, and the reason matters.** Success stayed at `3/25` and no scenario changed outcome. Adam normalizes per parameter, so a uniformly mis-scaled gradient produces nearly the same update; the truncation preserved sign and roughly the shape. The bug was invisible end-to-end precisely because Adam hides gradient magnitude. It still matters: reported `gradient_norms` were wrong, and any non-Adam optimizer, line search, or gradient-norm stopping rule would have been wrong too.

Reference comparison: Waymax's `InvertibleBicycleModel` integrates `new_vel = speed + accel * dt` the same way, so the acceleration gradient has the same double-integrator structure. It differs in two ways that are already accepted in Stage 3 - it uses curvature rather than a rate-limited wheel angle, and second-order position integration rather than C's Euler step. It also carries unsigned `speed = sqrt(vel_x^2 + vel_y^2)`, so it cannot represent reverse at all; scenario 0's solution requires reverse and is only expressible in the PufferDrive formulation. The reference's `opt()` nudges zero-valued actions to `1e-6` and aborts on non-finite gradients, which is the same zero-speed degeneracy surfacing as NaN there and as silent truncation here.

### The optimization is single-adversary in practice (2026-09-03)

Actions are parameterized for every candidate, but `ego_background_collision_cost()` reduces over candidates with a hard `min`, exactly as the reference `calculate_distance_ego_col` does. A hard min routes subgradient only to the argmin, so **exactly one candidate is ever pulled toward the ego**. Measured at the initial iterate:

| Scenario | Candidates | Receive ego-attraction gradient | Receive any gradient |
| --- | --- | --- | --- |
| 1 | 24 | 1 | 16 |
| 3 | 14 | 1 | 9 |
| 0 | 2 | 1 | 1 |

Every other candidate receives only constraint gradient from background-collision repulsion and drivable-area deviation, which never points at the ego.

The design intent is that the argmin is a soft selection that can switch as trajectories change, so the adversary need not be chosen up front. That does not happen: over 300 iterations on scenario 3 the argmin never moved off agent 4. Pulling the closest candidate closer only reinforces its argmin status, so the choice is settled at iteration 0 by the initial geometry and is winner-take-all thereafter.

Consequences to decide on, none of them yet acted on:

- The multi-agent parameterization costs optimizer work and perturbs background vehicles off their logged trajectories with no adversarial benefit. Freezing non-argmin candidates, or running one job per candidate, would be cheaper and more faithful to the logged scene.
- Because selection is fixed by initial geometry, a scenario whose only viable adversary is not the initially closest vehicle can never be found. This is a plausible contributor to the `18/25 iteration_limit` rate and should be tested by seeding one job per candidate.
- `actionable_collision` requires the C collision to involve `selected_adversary_idx`, which is stricter than the reference's "any ego collision counts". It cost nothing in the current run - every C ego collision was also actionable - but it can discard valid scenarios.

### C fix: per-agent episode state leaked across resets (2026-09-03)

`set_start_position()` reset metrics and agent state only after its per-agent early `continue`s, so an agent invalid at `init_step` never reached `reset_agent_state()`. Its `stopped` / `removed` flags survived `c_reset()` and outlived the episode. `move_dynamics()` freezes a `stopped` agent with `clear_agent_motion()`, so a late-appearing adversary that was stopped in one rollout stayed frozen at zero speed in every later one.

This surfaced as the ReGentS parity gate: `replay_optimized_scenario_in_c()` runs the baseline rollout before the adversarial rollout on the same Drive. On map 18 seed 60, adversary 9 first becomes valid at `t=16`, was stopped at `t=78` of the baseline, and then stayed frozen for the whole adversarial rollout, giving `c_torch_trajectory_error = 67.1 m` against `~1e-5` elsewhere.

Fix: hoist `reset_agent_metrics()` and `reset_agent_state()` above the early `continue`s so every agent resets regardless of validity at `init_step`. Both are pure state clears and consume no RNG, so trajectories stay bit-identical for agents that already reset. `generate_reward_coefs()` deliberately stays where it is: it draws from `env->rng_state` under `reward_randomization`, and moving it would shift the RNG stream for existing runs. The related reward-coef half of this gap remains covered by `start_regents_injection()` setting the neutral speed coefficient.

Measured over the 25-scenario NuPlan set: maximum `c_torch_trajectory_error` `67.118 -> 3.815e-05`, and scenarios over the `1e-4` gate `3 (18, 5, 17) -> 0`. Success rate is unchanged at `3/25`; the three parity rejections become honest optimizer outcomes instead of masking them.

This was not ReGentS-specific. Any replay episode where a late-appearing agent commits an infraction would leave that agent permanently frozen in every subsequent episode of the same env.

### Reference-fidelity audit against `ReGentS/` (2026-09-03)

Three divergences from `ReGentS/method/optim_scenario.py` were found and two were kept:

- **Front-divergence yaw window** used `pi/8` for both bearing and yaw; the reference uses `pi/8` for bearing and `pi/2` for yaw. The whole `[pi/8, pi/2)` yaw band — agents actually turning into the ego path — was misclassified as non-divergent, so the steering-cancellation rule was largely inert. Fixed via `front_yaw_half_angle_radians`.
- **Steering update damping** was missing. The reference scales every surviving steering update by `0.5`. Fixed via `steering_update_scale`, applied to the post-Adam update because Adam's normalization makes gradient scaling a no-op.
- **Drivable-area reduction** was changed to the reference's `sum` and then reverted. See cost contract 3: the reference's potential is an unnormalized density, ours is a normalized convolution, so the reduction is not transferable without also transferring the potential scale. Measured: with `sum`, generation collapsed to `0/25` because the off-road term outweighed the ego term by ~50x and the optimizer only polished off-road compliance (scenario 3: total `407 -> 77` while ego cost moved `4.825 -> 4.797`).

A/B over 25 NuPlan scenarios, seed 42, 80-transition horizon, 500 Adam updates, mean drivable reduction:

| Variant | Success | Background-collision rejections |
| --- | --- | --- |
| `steering_update_scale=1.0`, yaw `pi/8` (pre-audit) | `2/25` | 1 |
| `steering_update_scale=0.5`, yaw `pi/2` (reference) | `3/25` | 0 |

Scenario 23 flips to success and scenario 4's background collision disappears. Both fidelity fixes are net positive; neither is tuned.

## Status review against the general goal (2026-09-03)

Stages 0-6 are implemented and committed as `pufferlib/ocean/regents/` (12 modules) with `tests/regents/` (9 suites, 95 tests passing and 4 CUDA-dependent tests skipped as of this review), the two C binding entry points, and the `puffer regents` command. This section audits that result against the goal statement at the top of this document rather than against each stage's own exit gate.

### Initial POC scope

| Scope item | State | Evidence |
| --- | --- | --- |
| Replay with continuous actions and `classic` dynamics | Done | `dynamics.py`; Stage 2 parity `<= 3.8e-5` at 64 transitions |
| Vehicle adversaries only | Done | `NON_VEHICLE` filter in `filters.py` |
| One scenario per optimization job | Done, and now a constraint | `optimizer.py` rejects `batch_size != 1`; `generation.py` loops scenarios serially |
| Frozen logged or IDM ego during a Torch block | Done | `capture_frozen_idm_trajectory()`, `logged_fixture` source |
| IDM as the first reactive ego controller | Done | `run_reactive_idm_generation()`, native `CONTROLLER_IDM` |
| Deterministic offline generation and evaluation | Done | seeded generation, artifact config hash, repeat-determinism tests |
| No PPO loop, policy updates, rollout buffer, or training command | Done | `puffer regents` instantiates `Drive` directly |
| Fixed, frozen ego policy / IDM | Done | no optimizer touches controller state; Adam holds only action tensors |

The POC as scoped is complete. Its one measured end-to-end result is `1/16` generation success at a 16-transition horizon, with the caveats recorded under Stage 6.

### Ultimate goal

| Objective | State | Gap |
| --- | --- | --- |
| Benchmark against IDM | Partial | Generation runs IDM and records C-authoritative collision/off-road/infraction outcomes per scenario, but only for the scenario it just generated. There is no evaluation pass over a fixed generated set. |
| Benchmark against the policy | Not started | Nothing in `pufferlib/ocean/regents/` loads a checkpoint or sets `sdc_controller='policy'`. `run_reactive_idm_generation()` and `capture_frozen_idm_trajectory()` accept IDM and replay diagnostics but still reject a learned-policy SDC. The stop-gradient interface is ready for the swap; the swap is unwritten. |
| Comparative analysis IDM vs policy | Not started | `generation.py` reports one controller's rates. No harness runs two controllers over the same adversarial set and contrasts them. |
| Visual rendering | Partial | Interactive HTML replays and a gallery index ship and were produced for all 16 acceptance scenarios. The `mp4` half is not wired: the headless EGL pipeline in `scripts/render_scenario.py` is never invoked from the ReGentS path. |

### Structural gaps found in this review

- **Artifacts have no consumer.** `save_generation_artifact()` writes complete, self-describing `npz` files, and `load_generation_artifact()` only ever runs inside `tests/regents/test_c_replay.py`. The Stage 6 line item "add generated artifacts as an explicit replay/evaluation dataset input" was not implemented, and it is the single missing piece that blocks every remaining benchmarking objective.
- **Optimization is single-scenario by construction.** M6 batching is untouched, and generation cost is already `591.6 s` for 16 scenarios at a 16-transition horizon with 500 Adam updates. A full-horizon, checkpoint-benchmark-sized set is not reachable at this cost.
- **Filter yield, not optimizer strength, currently bounds the method result**, per the Stage 5 open issue above.

## Stage 7 — Evaluate a generated set with a fixed controller

Turn saved artifacts into an evaluation input, which unblocks both remaining benchmark objectives.

- Add an artifact-set reader that rebuilds the exact scenario, seed, horizon, and adversary action plan from an `npz` and installs it through the existing `regents_set_action_plan` path. It must verify the artifact's map hash against the map binary it loads and abort on mismatch.
- Run one configured ego controller over the whole set in C, with the adversary plan fixed. No Torch optimization runs in this mode.
- Report per-set collision, actionable-collision, off-road, and infraction rates plus per-scenario rows, using the same C event source as generation so numbers are comparable across controllers.
- Keep the controller a config value (`sdc_controller`), not a code path, per the native-integration design principle.

Exit gate: replaying the `regents_nuplan` artifacts with `sdc_controller=idm` reproduces the collision timesteps and event flags recorded at generation time, bit-for-bit where C is deterministic.

## Stage 8 — Learned-policy ego and comparative benchmark

Completes M5 and the ultimate goal's comparative half.

- Load a checkpoint in inference/evaluation mode with gradients disabled, and drive the ego through the simulator's existing `CONTROLLER_POLICY` routing. Relax the IDM-only guards in `optimizer.py` and `rollout.py` to accept either controller without changing the frozen-ego or stop-gradient contracts.
- Confirm the reactive outer loop behaves identically with a policy ego: no gradient crosses C or the controller, and `maximum_ego_reference_error` stays a reported diagnostic rather than a gate.
- Generate against one controller, then evaluate both controllers on that same set through Stage 7, and report the contrast. State explicitly which controller a set was generated against; a set generated against IDM is not a neutral benchmark for a policy and vice versa.

Exit gate: the same generation run completes with `sdc_controller=policy`, and a single comparative table reports IDM and policy rates over one fixed adversarial set.

## Stage 9 — Full-horizon validation, mp4 rendering, and batching

- Complete the `regents_nuplan_20s` run and record parity, yield, success rate, and runtime at 200 transitions. Establish a parity threshold for horizons beyond 64 transitions first, as Stage 2 requires.
- Resolve the Stage 5 original-collision open issue with a measured decision, then requote yield.
- Route the ReGentS replay path through the headless EGL pipeline for `mp4` output alongside the existing HTML replays.
- Batch the optimizer across scenarios (M6), preserving per-scenario determinism and the existing reduction semantics.

Retain the vectorized candidate-involved feasibility signature, candidate-only differentiable regularizer, horizon-bounded inverse dynamics, and inverse-prefix reuse as prerequisites for true batching. In the 2026-09-03 two-scenario, 50-transition smoke run, these corrections reduced reported optimization time from `23.4 s` to `6.0 s` while preserving maximum C/Torch error `3.815e-6`. A two-process trial was slower than serial execution on this CPU workload; M6 should stack scenarios inside each worker to amortize Torch work rather than only spawn more one-scenario processes.

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

Stages 7-9 add an artifact-set evaluation module beside `generation.py` and its test suite; the controller swap is a config value plus guard relaxation in `optimizer.py` and `rollout.py`, not a new controller layer.

Avoid importing ReGentS into the PPO hot path until the offline pipeline is accepted.

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
| M7 | Saved artifacts replay as an evaluation set under a configured controller | Controller benchmarking |
| M8 | IDM and learned policy compared over one fixed adversarial set | The stated ultimate goal |

M1 is the first ReGentS model milestone and is a hard gate. In particular, do not tune loss weights to compensate for a dynamics or inverse-dynamics mismatch.

S0 through M4 are complete. M5's reactive IDM half is complete; its learned-policy parity half remains, along with M6 batching and the new M7/M8. The initial POC scope is fully delivered; the ultimate goal is roughly half delivered, gated on the artifact-set evaluation input described in Stage 7.

## Known risks to resolve early

- `get_state()` is intentionally broad and Python-list based. It is suitable for initial offline extraction, but a compact NumPy binding may be needed later if profiling shows serialization to be a bottleneck.
- `simulation dt` and logged trajectory `log_dt` are distinct fields. A mismatch invalidates inverse dynamics unless explicitly resampled.
- Existing accessors return world coordinates by adding map means, while C dynamics operate in centered coordinates. The adapter must choose one frame and transform state and map data together.
- The current road-edge accessor has incomplete vectorized scenario IDs, and road edges may not encode the same drivable-area semantics as the ReGentS out-of-bounds raster.
- Logged trajectories need not be exactly realizable by PufferDrive classic dynamics. Real-data reconstruction thresholds must be measured and reported rather than assumed to be `1e-4`; the `1e-4` target remains mandatory for C-generated parity fixtures.
- Hard `min`, clipping, action saturation, and collision boundaries are only piecewise differentiable. Tests must cover useful gradient flow near, but not exactly on, these boundaries.
- Torch uses discrete-timestep box distances while C also checks swept OBB collisions. Optimization loss and authoritative acceptance therefore have intentionally different roles.
- Pairwise geometry is potentially `O(agents^2 * timesteps)` in memory. Establish correctness with one scenario first, then profile and chunk without changing reduction semantics.
- Optimization cost scales with horizon and Adam iterations. `591.6 s` for 16 scenarios at a 16-transition horizon sets the floor; a benchmark-sized set at the full 200-transition horizon needs M6 batching before it is affordable.
- A generated set is adversarial *against the controller it was generated with*. Any IDM-versus-policy comparison must state which controller drove generation, or the comparison measures the generation target rather than controller quality.
