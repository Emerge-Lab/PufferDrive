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
    test_stage0_idm.py
    test_adapter.py
    test_dynamics_parity.py
    test_inverse_dynamics.py
    test_geometry.py
    test_losses.py
    test_filters.py
    test_optimizer.py
    test_c_replay.py

pufferlib/config/evaluation/
    benchmark.yaml  # regents_idm Stage 0 benchmark
    regents.yaml    # Stage 1+ offline generation config
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

## Known risks to resolve early

- `get_state()` is intentionally broad and Python-list based. It is suitable for initial offline extraction, but a compact NumPy binding may be needed later if profiling shows serialization to be a bottleneck.
- `simulation dt` and logged trajectory `log_dt` are distinct fields. A mismatch invalidates inverse dynamics unless explicitly resampled.
- Existing accessors return world coordinates by adding map means, while C dynamics operate in centered coordinates. The adapter must choose one frame and transform state and map data together.
- The current road-edge accessor has incomplete vectorized scenario IDs, and road edges may not encode the same drivable-area semantics as the ReGentS out-of-bounds raster.
- Logged trajectories need not be exactly realizable by PufferDrive classic dynamics. Real-data reconstruction thresholds must be measured and reported rather than assumed to be `1e-4`; the `1e-4` target remains mandatory for C-generated parity fixtures.
- Hard `min`, clipping, action saturation, and collision boundaries are only piecewise differentiable. Tests must cover useful gradient flow near, but not exactly on, these boundaries.
- Torch uses discrete-timestep box distances while C also checks swept OBB collisions. Optimization loss and authoritative acceptance therefore have intentionally different roles.
- Pairwise geometry is potentially `O(agents^2 * timesteps)` in memory. Establish correctness with one scenario first, then profile and chunk without changing reduction semantics.
