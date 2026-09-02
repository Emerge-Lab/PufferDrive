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
- no PPO loop, policy updates, rollout buffer, or training command in the ReGentS POC.

Jerk dynamics, pedestrians/cyclists, joint policy gradients through the ego policy, and online generation during PPO training are follow-up work. The method reference is the local [ReGentS paper](../2409.07830v1.pdf).

### Ultimate Goal: Policy & IDM Benchmarking and Visual Rendering

Beyond the initial POC, the ultimate objective of the ReGentS integration is to systematically evaluate and compare different ego controllers under synthesized adversarial conditions:

- **Benchmark against IDM:** Evaluate how an IDM-controlled SDC responds when subjected to ReGentS-optimized adversarial scenarios.
- **Benchmark against the Policy:** Evaluate the learned neural network policy (controlling the ego agent) against these same adversarial scenarios to assess its resilience and safety performance under high-stress conditions.
- **Comparative Analysis:** Contrast the final evaluation metrics (collision rates, off-road events, infraction rates) of the IDM controller versus the trained ego policy to benchmark performance improvements.
- **Visual Rendering:** Generate high-fidelity rendered videos (`mp4` via the headless EGL pipeline or interactive replays) of these evaluations as soon as rendering capabilities become available for the ReGentS pipeline, ensuring qualitative validation of both adversarial maneuvers and ego reactions.

## Step 0 — Validate IDM as the ego controller

Before implementing any ReGentS module, validate the existing IDM controller path for the SDC. PufferDrive already accepts `sdc_controller=idm` and dispatches an IDM-controlled active agent through `move_idm()`, so this step should add a supported configuration and regression coverage rather than another controller implementation.

Initial configuration:

- `simulation_mode=replay`;
- `eval_mode=true` and `compute_eval_metrics=true`;
- `control_mode=control_sdc_only`;
- `sdc_controller=idm`;
- background vehicles and non-vehicles on logged replay;
- `reward_conditioning=false` and `reward_randomization=false`;
- one fixed scenario, `init_step`, and exact evaluation seed;
- replay capture enabled when producing visual artifacts.

Work and exit gate:

- Add a focused integration test that constructs `Drive` directly in evaluation mode, resets a real replay scenario, and verifies agent index 0 is active with `CONTROLLER_IDM` while background actors use `CONTROLLER_REPLAY`.
- Step through a short rollout without meaningful policy actions and verify that the SDC is moved by IDM, background actors follow their logged states, and `get_state()` captures the resulting trajectory.
- Repeat with the same seed and assert identical states and controller assignments.
- Confirm collision, off-road, and termination handling remain available for the IDM SDC.
- Record the exact configuration overrides used by the ReGentS POC. Any controller-routing or reset bug discovered here is fixed before Stage 1.

Do not invoke `puffer train` for this workflow. The eventual user-facing command should be a dedicated offline ReGentS generation/evaluation entry point backed by an evaluation config.


### Step 0 validation record (2026-09-02)

S0 passed on the first 100 unique, lexicographically sorted `.bin` files in `pufferlib/resources/drive/binaries/nuplan` using seed `42`, `init_step=0`, and a 200-step horizon. The selected filenames have SHA-256 `d8e98b2c88965044d4074ec47a667638b95d2975e2390e275510364421643ea9`; the directory contained 101 files at evaluation time, so the resolved output records the exact 100-file selection rather than cycling or sampling with replacement.

The supported entry point is `puffer regents puffer_drive stage0 [key=value overrides]`, backed by `pufferlib/config/evaluation/regents.yaml`; `python -m pufferlib.ocean.regents.stage0` is also supported. That config records the exact POC overrides: replay/evaluation mode, evaluation metrics enabled, SDC-only control with IDM at index 0, replay controllers for every background type, continuous actions with classic dynamics, fixed `dt=0.1`, disabled reward conditioning/randomization and init-step spread, and stop handling for collision/off-road/traffic-light infractions. Replay capture is optional and disabled for the metrics-only baseline.

The validation repeats an eight-step `get_state()` trajectory on one pinned scenario and exact seed, compares serialized simulator states and controller assignments byte-for-byte, verifies arbitrary policy-buffer actions do not affect the IDM SDC, and checks valid background state fields against their logged values after each step. All 100 scenarios then passed controller routing, logged-background, metric-availability, and truncation checks. Baseline outcomes were 22% collision, 5% at-fault collision, 2% off-road, and 1% red-light violation. These rates are recorded baselines, not S0 pass thresholds. No controller-routing or reset bug was found.

## Core contracts

Use one explicit Torch representation throughout the differentiable path:

- state tensors: `[batch, agent, time, feature]`, with at least scenario-local centered `x`, `y`, wrapped heading, signed speed, and steering angle;
- action tensors: `[batch, agent, time - 1, 2]`, containing normalized PufferDrive acceleration and target-steering actions in `[-1, 1]`;
- metadata: agent type, stable agent ID/index, SDC mask, length, width, wheelbase, per-agent maximum speed, scenario ID, and timestep;
- masks: state validity, transition validity (`valid[t] & valid[t + 1]`), vehicle, ego, candidate adversary, and optimized action;
- map data: a documented world-to-grid transform and either a drivable-area raster or enough geometry to construct it deterministically.

All APIs must preserve agent identity and use one documented coordinate frame. Invalid entries are represented by masks, not silently repaired. External scenario data and tensor shapes are validated before optimization starts.

## Stage 1 — Expose complete scenarios and build the Torch adapter

Use `Drive.get_state()` as the primary scenario source rather than introducing a second scenario loader. The render/replay state already serializes all C agents, their complete logged positions/headings/velocities/validity, current dynamics state, type and controller metadata, wheelbase and dimensions, road geometry, traffic controls, active/static mappings, and map bounds. The smaller `get_global_*` helpers are not needed for the first implementation.

Work:

- Audit and document the `get_state()` schema, especially the centered coordinate frame, per-agent trajectory lengths, vectorized return format, and the fact that serialized agent `id` is the stable C agent-array index. Derive the SDC mask from PufferDrive's `EGO_IDX == 0` invariant.
- Build `state.py` dataclasses and `adapter.py` conversion/validation directly from this payload. Convert its Python lists to contiguous NumPy arrays and then perform one explicit conversion to Torch.
- Derive logged signed speed from `log_velocity_x/y` projected onto `log_heading`. Derive current signed speed from `sim_vx/y` and `sim_heading`; no new field is required for that value.
- Use the serialized road elements, their types, and `map_corners` to build a deterministic drivable-area raster with a documented origin, resolution, dimensions, and axis convention. Validate the result against existing off-road behavior.
- Obtain simulation `dt`, configured maximum speed, `init_step`, and `scenario_length` from the Python `Drive` instance. Initially disable reward conditioning/randomization so the effective maximum speed is the configured base maximum.
- Add fields to `get_state()` only for information that cannot be derived or read from the Python environment. The only expected initial gap is `log_dt`, which should be exposed so temporal alignment can be checked. If variable per-environment `init_step` is enabled later, expose that value per serialized scenario as well.
- Initially require `simulation_mode=replay`, `action_type=continuous`, `dynamics_model=classic`, and a fixed `init_step`. Fail clearly if `dt` and `log_dt` differ; deterministic trajectory resampling can be added later as a separate feature.

Tests and exit gate:

- Adapter/schema tests cover shapes, dtypes, all-agent indexing, SDC identity, the centered coordinate frame, vectorized scenarios, and invalid timesteps.
- Resetting the same scenario and seed produces byte-identical exported inputs.
- A real scenario can be converted into the Torch representation with all contract checks enabled and no identity or timestep ambiguity.

## Stage 2 — Implement differentiable classic dynamics and prove C parity

Implement a pure Torch `classic_step` and a masked multi-step `classic_rollout`. Match `move_dynamics()` in `drive.h` in the same operation order and initially use CPU `float32` for the strictest comparison with C.

The implementation must include:

- normalized continuous action scaling (`acceleration * 4.0`, target steering `* 0.667`);
- target-steering rate limiting at `0.6 rad/s`, followed by the steering limit;
- speed update and clipping to `[-2.0, effective_max_speed_mps]`;
- `beta = atan(0.5 * tan(steering))`;
- yaw rate from updated speed, wheelbase, `beta`, and steering;
- position update using the old heading and updated speed;
- heading wrapping with the same interval convention as C;
- state carried across time, including signed speed and actual steering angle.

Parity tests must use a small diagnostic binding or fixture that exposes the complete C state after each step. Disable unrelated stochastic features and infraction side effects during these tests.

Tests and exit gate:

- One-step parity across neutral, braking, acceleration, reversing, steering-rate saturation, steering clipping, speed clipping, and heading wrap cases.
- Full-rollout parity on deterministic random action sequences and inverse-derived actions from real scenarios.
- Compare every state component at every valid step, not only final `x/y`.
- Target maximum absolute error is `1e-4` or better over the agreed rollout horizon. Any systematic mismatch blocks later stages.

## Stage 3 — Estimate expert actions with inverse dynamics

Implement inverse dynamics for valid consecutive logged vehicle states. The result initializes the adversarial action sequence; it is not treated as exact ground-truth control.

Work:

- Compute signed speed from logged velocity projected onto heading. Use finite differences only as an explicitly tested fallback.
- Compute heading differences with `atan2(sin(delta), cos(delta))` so yaw wrapping is safe.
- Estimate acceleration from consecutive signed speeds and recover the steering needed by the PufferDrive bicycle equations. Account for the fact that PufferDrive applies the updated speed before position and heading integration.
- Carry actual steering through the sequence, apply the same steering-rate and angle limits, and convert the recovered physical commands back to normalized actions.
- Mark actions invalid across gaps. Do not interpolate, pad, or infer through invalid states in the first implementation.
- Handle near-zero speed as an underdetermined case: keep a deterministic neutral or previous feasible steering estimate and exclude its heading residual from the inverse validation metric.
- If the logged transition is inconsistent with the classic model, choose the bounded action minimizing a documented one-step state residual and report that residual. Never hide it with clipping or a loose mask.

Tests and exit gate:

- Exact recovery tests on trajectories generated by the Torch model and by C, including limit cases and reverse motion.
- Real-data reconstruction reports position, heading, and speed errors by timestep and percentile, split by low-speed and normal-speed transitions.
- Required milestone: `forward(state_t, inverse(state_t, state_t+1)) ~= state_t+1` on valid real PufferDrive transitions, with acceptance thresholds recorded from an initial dataset audit.
- Do not start ReGentS optimization until both this gate and the Stage 2 C/Torch parity gate pass.

## Stage 4 — Implement differentiable geometry and KING/ReGentS costs

Implement the three paper costs independently before combining them:

1. ego-background collision induction: minimum over candidate adversaries of their validity-masked, time-averaged signed bounding-box distance to ego;
2. background-background collision avoidance: penalize the minimum truncated distance between distinct background actors;
3. drivable-area deviation: sample a Gaussian-smoothed out-of-bounds raster at the four corners of each optimized vehicle box and average over valid steps.

The existing WOSAC Torch geometry in `pufferlib/ocean/evaluation_utils/wosac/` is a strong reuse candidate. Before reuse, verify that its rounded-box distance, validity behavior, sign convention, in-place operations, and memory scaling match the optimization requirements. Put any ReGentS-specific semantics behind the new `geometry.py` API rather than importing evaluation details throughout the optimizer.

Work and tests:

- Implement oriented box corners and a signed separation/penetration distance with a collision sign that agrees with the C OBB check away from numerical boundaries.
- Construct the out-of-bounds raster once per map. Use `torch.nn.functional.grid_sample` with an explicit world-to-normalized-grid transform and tested boundary behavior.
- Apply masks before reductions so invalid actors/timesteps cannot win a `min` or alter a mean denominator.
- Keep the three weights and the KING truncation/Gaussian parameters named and configurable. Optional plausibility or action-deviation regularizers must be labeled as extensions, not paper-equivalent ReGentS.
- Add hand-computed unit cases, transform tests, CPU/GPU consistency checks, `torch.autograd.gradcheck` where practical, finite-difference comparisons, and tests that useful gradients reach acceleration and steering actions.

Exit gate: each cost has correct values, masks, and finite nonzero gradients in representative isolated scenes, and the combined loss decreases in a small synthetic optimization.

## Stage 5 — Add ReGentS selection, constraints, and frozen-ego optimization

First optimize against a frozen ego trajectory. Use a C rollout with the ego controlled by IDM for the first POC; the logged ego trajectory remains a useful deterministic fixture for unit and regression tests. Freezing the captured IDM trajectory within an optimization block isolates adversary optimization from controller feedback while exercising the controller intended for the first end-to-end demonstration.

Candidate filtering:

- exclude the SDC and non-vehicles;
- require a configurable fraction/count of valid transitions over the optimization horizon;
- exclude already invalid or unsuitable scenes, and separately label scenes that already contain a collision;
- exclude static vehicles using a named displacement/speed threshold;
- exclude rear/convergent adversaries that remain in the paper's non-actionable rear sector for more than a configurable fraction of the original scenario.

Optimization:

- Initialize background actions from Stage 3 and optimize only candidate, valid actions with Adam.
- Clamp/project normalized actions to `[-1, 1]` after each update. Preserve non-candidate and invalid actions exactly.
- Implement the ReGentS front-divergence rule explicitly: for a candidate that spends more than `tau_front` of applicable timesteps in the front red zone, cancel its steering-action update while retaining its acceleration update. Use wrapped ego-relative yaw/position angles and the paper's `pi/8` applicability bounds.
- Make all thresholds, cost weights, learning-rate settings, iteration count, and early-stop rules named configuration values.
- Stop on a verified ego-adversary box overlap, but also retain the best feasible iterate so a failed generation job has a useful diagnostic result.
- Record the selected adversary, filter reasons, initial/final costs, gradient norms, action saturation, reconstruction error, collision timestep, and deterministic seed.

Tests and exit gate:

- Unit tests cover every filter and both sides of each angular boundary.
- A front-divergence fixture proves steering gradients are masked and acceleration gradients remain active.
- End-to-end synthetic scenes cover braking, merging, background-collision rejection, and off-road rejection.
- On a small fixed real-scenario suite, optimization is deterministic, reduces the intended collision cost, and produces no NaN/Inf values.

## Stage 6 — Replay in C, add reactive ego iteration, and integrate the pipeline

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

tests/regents/
    test_adapter.py
    test_dynamics_parity.py
    test_inverse_dynamics.py
    test_geometry.py
    test_losses.py
    test_filters.py
    test_optimizer.py
    test_c_replay.py

pufferlib/config/evaluation/
    regents.yaml
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
| M6 | Separate C/Torch/inverse parity for jerk dynamics | Optional jerk support |

M1 is the first ReGentS model milestone and is a hard gate. In particular, do not tune loss weights to compensate for a dynamics or inverse-dynamics mismatch.

## Known risks to resolve early

- `get_state()` is intentionally broad and Python-list based. It is suitable for initial offline extraction, but a compact NumPy binding may be needed later if profiling shows serialization to be a bottleneck.
- `simulation dt` and logged trajectory `log_dt` are distinct fields. A mismatch invalidates inverse dynamics unless explicitly resampled.
- Existing accessors return world coordinates by adding map means, while C dynamics operate in centered coordinates. The adapter must choose one frame and transform state and map data together.
- The current road-edge accessor has incomplete vectorized scenario IDs, and road edges may not encode the same drivable-area semantics as the ReGentS out-of-bounds raster.
- Logged trajectories need not be exactly realizable by PufferDrive classic dynamics. Real-data reconstruction thresholds must be measured and reported rather than assumed to be `1e-4`; the `1e-4` target remains mandatory for C-generated parity fixtures.
- Hard `min`, clipping, action saturation, and collision boundaries are only piecewise differentiable. Tests must cover useful gradient flow near, but not exactly on, these boundaries.
- Torch uses discrete-timestep box distances while C also checks swept OBB collisions. Optimization loss and authoritative acceptance therefore have intentionally different roles.
- Pairwise geometry is potentially `O(agents^2 * timesteps)` in memory. Establish correctness with one scenario first, then profile and chunk without changing reduction semantics.
