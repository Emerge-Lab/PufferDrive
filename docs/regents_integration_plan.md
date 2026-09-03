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

`estimate_expert_actions()` in `inverse_dynamics.py`. Acceleration comes from consecutive signed speeds and first minimizes reachable speed error; steering is analytically seeded from wrapped heading change, then refined by a deterministic bounded search minimizing squared position error plus wheelbase-scaled squared heading error. Steering carries within a contiguous run, resets to neutral at an unobserved run start, and is never inferred across a validity gap. Every inconsistent transition retains its component errors and a combined meter-scaled residual.

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
2. **Background collision avoidance** — `-min(min(1.25 m, signed_distance))` over distinct jointly valid background pairs and timesteps; neutral zero when fewer than two backgrounds share a valid pair.
3. **Drivable-area deviation** — four sampled corner potentials per optimized vehicle, averaged over that vehicle's valid timesteps, summed over optimized vehicles.

Masked storage becomes non-degenerate internal placeholders before box operations and is excluded before every reduction, so invalid padded boxes never produce NaN gradients and never contribute a value or denominator.

The map-static potential starts from `~drivable_mask`, applies a normalized Gaussian, and adds an out-of-bounds frame; sampling uses the Stage 1 centered transform with `grid_sample(..., align_corners=True)`, bilinear interpolation, and border padding, so points beyond raster coverage are explicitly out of bounds. `prepare_out_of_bounds_rasters()` builds these once per optimization.

Reference defaults: ego/background/drivable weights `1 / 5 / 20`, background truncation `1.25 m`, Gaussian sigma `0.5 m`, finite support `3 sigma` (deterministic normalized convolution replacing the reference code's crop approximation, and part of the cost configuration).

Tests: hand-computed separation/penetration/contact, rotated and randomized SAT, denominators, transforms and outside-map sampling, Gaussian bounds and spatial gradients, `gradcheck` and finite differences, independent nonzero gradients for all three costs, gradients through `classic_rollout`, CPU/GPU consistency, a 20-step Adam fixture with decreasing loss, and a real replay scenario through the full adapter/raster/cost path. CUDA tests skip when unavailable.

### Stage 5 — Selection, constraints, frozen-ego optimization (`389fc411d`)

`filters.py` selects candidates deterministically; `optimizer.py` runs one-scenario frozen-ego Adam. `capture_frozen_idm_trajectory()` resets a directly instantiated Drive with native C IDM on stable agent zero and replay backgrounds, then captures detached centered ego states. `logged_fixture` remains an explicit source for deterministic tests.

Selection records reason bits rather than compacting agents, excluding: SDC, non-vehicles, insufficient transition coverage, static actors, the paper's non-actionable rear sector, caller-labeled unsuitable scenes, and original collisions. Original-collision labels mirror the reference `overlap_with_ego` — per metadata-bearing agent, exact oriented-box overlap against the sole ego at jointly valid timesteps. Background/background overlap is not labeled. A labeled agent is dropped as a candidate; the scene is filtered `ORIGINAL_COLLISION` only when every candidate-masked agent carries the label. Defaults: at least `50%` and one valid transition; static below `0.2 m` first-to-last displacement or `0.2 m/s` maximum absolute speed; rear occupancy strictly above `80%` within `pi/8` of directly behind the ego. All fractions use only applicable jointly valid states or transitions.

Stage 3 actions initialize every vehicle; only selected valid entries enter the graph. The frozen ego and non-selected actors hold their reference trajectories. Adam defaults: learning rate `1e-3`, betas `0.9 / 0.999`, epsilon `1e-8`, 500 updates. Actions project to `[-1, 1]`, and non-candidate or invalid entries are restored byte-for-byte after every update. `classic_step()`'s zero-speed magnitude uses a forward-equivalent clamp at the dtype's smallest normal value, giving a finite derivative at an exactly stationary state without changing Stage 2 parity.

Front divergence uses wrapped ego-relative bearing and yaw: both strictly inside `(-pi/8, pi/8)`, on the same side, with bearing magnitude below yaw magnitude. Steering updates cancel when red-zone occupancy exceeds `tau_front=0.5`; acceleration updates continue, and Adam steering moments are cleared for canceled entries so momentum cannot bypass the rule.

Success requires a discrete-timestep ego/candidate box overlap with no background/background overlap and no newly introduced off-road corner. Existing raster mismatch on the reference trajectory is the feasibility baseline, so valid source data is never retroactively declared off-road. The optimizer stops on the first feasible collision by default, supports named collision and stagnation early-stops, and otherwise returns the lowest-total-cost feasible iterate. It records filter reasons, adversary index and ID, initial/final component costs, gradient norms, front-divergence iterations, action saturation, maximum inverse reconstruction residual, collision timestep, rejection counts, iteration counts, frozen-ego source, failure reason, and seed.

Determinism fixture — NuPlan map indices 5 and 8, seeds 47 and 50, 16-transition horizon, five Adam updates at `1e-3`: ego collision cost `2.0703814 -> 1.9595402` and `5.6210895 -> 5.5688214`; total cost `69.4004211 -> 68.1501007` and `19.9497566 -> 18.9804382`.

### Stage 6 — C replay, reactive ego, pipeline (`3734f6373`)

Stable-index action injection in C, authoritative replay and the reactive loop in `rollout.py`, artifacts in `artifacts.py`, offline generation in `generation.py` behind `puffer regents`.

**C injection.** `regents_set_action_plan` installs one `[stable_agent, transition, 2]` plan plus mask, rejecting non-replay, non-vehicle, ego, non-finite, and out-of-range entries, and requiring replay mode with continuous classic dynamics on exactly one environment. `c_step` applies it in the expert-static loop by stable simulator index, never a compacted batch position. `regents_get_events` returns the timestep, the simulator's moving-OBB collision pairs, and per-agent off-road flags; injected actors run `compute_metrics` against a scratch log so they carry authoritative infraction flags without entering policy episode logs or rewards.

`start_regents_injection` seeds an adversary's first injected transition from its logged state: neutral wheel steering, logged dimensions and wheelbase, logged yaw rate, and logged longitudinal velocity projected on the agent heading. Three C-side gaps surface only under injection and were fixed here — `c_reset` returns early when `timestep == init_step`, so `set_start_position` cannot prepare an injected agent and seeding at the transition is the only order-independent mechanism; agents invalid at `init_step` never reach `generate_reward_coefs`, leaving `reward_coefs[REWARD_COEF_SPEED]` zero and clipping the adversary to a zero speed limit, so injection sets the neutral coefficient the exported `maximum_speed_mps` already assumes; replay actors carry no wheel steering or classic-dynamics speed state at all.

**Speed conventions differ by design and must stay that way.** Logged and Torch states use the bicycle model's longitudinal speed (the heading projection of logged velocity) — what Stage 1 exports and what the Stage 3 gates measured. C's `sim_speed_signed` is velocity magnitude signed by heading agreement, exact for states C integrated itself, so `signed_speed_from_c_velocity()` is used only when reading C state back. The two agree once injection integrates a state and differ by the logged slip angle before it, so speed and steering are compared only at injection-produced states. Position and heading are compared for every jointly valid agent, the shared initial state is verified rather than scored, and comparison truncates at the first C collision because collision and infraction responses intentionally change C state.

**C is the success oracle**, measured against a baseline C rollout of the same scenario, seed, and horizon driven by the Stage 3 initial actions under the same mask. Only collision pairs and off-road flags absent from that baseline are attributed to the optimization, so pre-existing logged overlaps neither fail generation nor inflate the background-collision rate. Success requires a new actionable ego/adversary collision, no new background collision, and no new adversary off-road. A reactive ego's divergence from the frozen reference is reported as `maximum_ego_reference_error` and excluded from the parity gate whenever the ego is not replay-controlled.

`run_reactive_idm_generation()` captures a native C IDM ego, detaches it, optimizes adversaries in Torch, reruns C for IDM's response, and repeats to a fixed maximum outer iteration count, stopping on the first C-confirmed success or unchanged actions. No gradient crosses the C simulator or the ego controller; swapping in a learned policy needs no interface change.

Artifacts are pickle-free `npz` files carrying a schema tag, scenario and dataset identity, source map path, a SHA-256 hash over the canonical configuration plus exact map bytes, masks, initial and optimized actions, Torch and C trajectories, C ego actions, both replay metadata blocks, and every metric. `save_generation_artifact()` writes atomically and returns exactly the persisted metadata; `load_generation_artifact()` rejects unknown schemas, mismatched fields, and oversized arrays.

`puffer regents <env_name> <generation_name>` reads `pufferlib/config/evaluation/regents.yaml`, validates it against the `Drive` signature, instantiates `Drive` directly, and never initializes PPO, a policy optimizer, a rollout buffer, or training logging. It writes one artifact per scenario plus `generation_metrics.csv`.

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
- **Config drift.** `regents.yaml`'s `regents_nuplan` entry was edited after the acceptance run; its scenario count, horizon, and Adam iteration count all differ from the acceptance settings, so that name no longer reproduces the recorded result — only `experiments/regents/regents_nuplan/` holds it. Restore the settings under that name or move the record to a separate entry before citing the numbers again.
- **Full horizon unvalidated.** The `regents_nuplan_20s` entry (16 scenarios, 200 transitions, the complete 20 s window) exists but `experiments/regents/regents_nuplan_20s/` is empty — started, never completed. Every recorded result is at 16 or 50 transitions, at most 5 s of a 20 s scenario. Parity, yield, runtime, and success rate at full horizon are unmeasured.

## Status review against the general goal (2026-09-03)

Stages 0-6 are implemented and committed as `pufferlib/ocean/regents/` (12 modules) with `tests/regents/` (9 suites, 96 tests passing as of this review), the two C binding entry points, and the `puffer regents` command. This section audits that result against the goal statement at the top of this document rather than against each stage's own exit gate.

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
| Benchmark against the policy | Not started | Nothing in `pufferlib/ocean/regents/` loads a checkpoint or sets `sdc_controller='policy'`. `run_reactive_idm_generation()` and `capture_frozen_idm_trajectory()` both hard-reject a non-IDM SDC. The stop-gradient interface is ready for the swap; the swap is unwritten. |
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
