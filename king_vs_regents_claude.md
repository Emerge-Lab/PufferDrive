# KING vs ReGentS: loss and adversary selection

Sources compared (2026-09-15):

- **Original KING**: `king/` (autonomousvision/king). CARLA 0.9.10 towns + differentiable PyTorch proxy simulator.
- **ReGentS**: `ReGentS/` (valeoai/ReGentS; identical to `../ReGentS`). Waymax + JAX on the Waymo Open Motion Dataset (WOMD).
- **ReGentS paper**: `2409.07830v1.pdf` (§4.2–§5).

Claims marked **(verified)** were checked by running `ReGentS/cost.py` unmodified under JAX 0.5.3 on CPU (Appendix A). KING's PyTorch cost needs CUDA and CARLA, so its behavior comes from reading the code. Neither full optimization loop was run: the WOMD data and AIM-BEV checkpoint for ReGentS aren't available, and there is no CARLA server for KING.

## TL;DR

- **The ReGentS method does not change the loss.** `method.type: king` and `method.type: regents` call the same `loss_king` (`ReGentS/method/optim_scenario.py:90-100`). ReGentS adds two things: (1) filtering which vehicles may be adversaries, and (2) zeroing or halving steering updates after Adam.
- **The loss in the ReGentS repository still differs from original KING.** These differences come from the Waymax port, not from the ReGentS method:
  - squared center-to-center distance instead of box distance;
  - "min over adversaries of the time average" instead of "time average of the per-step min";
  - different weights and optimizer settings.
- **Two of the three ReGentS loss terms barely act (verified).**
  - The adversary–adversary repulsion only has a gradient once two vehicle centers are within 1.25 m of each other, i.e. deep overlap.
  - The off-road term is ≈ 0 almost everywhere because of a crop-index bug.
  - So in most scenes the released optimizer is driven almost entirely by the ego-attraction term, plus the update masks.
- **Adversary selection differs.**
  - KING has no explicit selection. It jointly optimizes 1, 2 or 4 pre-spawned vehicles, and its per-timestep min lets several of them share the pull.
  - ReGentS pools every valid vehicle of a WOMD scene (up to 31 besides the ego). Its min-of-mean puts the whole pull on a single vehicle, so it relies on filters (static, persistent rear) and a steering freeze for front-divergent vehicles to keep that vehicle a sensible choice.
- **The ReGentS code and paper disagree** on three points:
  - the yaw window of the divergence rule: ±90° in code vs ±22.5° in the paper;
  - an undocumented ×0.5 steering step for every non-flagged vehicle;
  - center distance in code vs bounding-box distance in the paper.
- **The "KING" column of ReGentS Table 1 is not the original KING code.** Its 48.99 % success rate (vs 60.40 % for ReGentS) comes from ReGentS's own Waymax port: the shared loss, without the filters and masks.

## 1. Three things to compare

| | Original KING | ReGentS repo, `method.type: king` | ReGentS repo, `method.type: regents` |
| --- | --- | --- | --- |
| Entry point | `king/generate_scenarios.py` | `ReGentS/method/optim_scenario.py` | same file |
| Simulator / data | CARLA towns, PyTorch proxy sim | Waymax (JAX), WOMD | Waymax (JAX), WOMD |
| Loss | KING loss (§3.1) | port of the KING loss (§3.2) | **same as `king` mode** |
| Adversary candidates | all N spawned vehicles | valid vehicles except the ego | `king` candidates minus static and persistent-rear vehicles |
| Update rule | Adam | Adam | Adam, then steering ×0 (front-divergent) or ×0.5 (all others) |
| Reported success | ≈ 80 % on CARLA (as quoted in ReGentS §5.2) | 48.99 % on 200 WOMD validation scenes (Table 1) | 60.40 % (Table 1) |

## 2. Pipeline differences that shape the gradients

| | Original KING | ReGentS repo (both modes) |
| --- | --- | --- |
| Step and horizon | 0.25 s × 80 steps = 20 s (`--sim_tickrate 4 --sim_horizon 80`) | 0.1 s steps over the WOMD log, ≈ 9 s (91 frames, `init_steps: 1`) |
| Vehicles | ego + N ∈ {1, 2, 4} (`run_generation.sh`) | ego + up to 31 objects (`max_num_objects: 32`) |
| Adversary initial state | spawn point, route and 80-step action sequence from `king_initializations/.../{N}_agents/RouteScenario_*.json`; start speed 4 m/s (`simulator.py:154,396`) | logged WOMD state; actions inferred from the log with Waymax's inverse bicycle model (`simulation.py:36-55`) |
| Adversary behavior | open-loop action sequence, non-reactive (`bm_policy.py`) | open-loop action replay for every non-ego object, non-reactive (`agent/action_actor.py`) |
| Optimized parameters | pre-tanh `steer` and `throttle` per adversary per step, shape `B×N×T×1` (`bm_policy.py:26-36,51-52`); negative throttle means braking | raw `(accel, curvature)` per object per step (T arrays of `num_objects×2`); Waymax hard-clips to ±6 m/s² and ±0.3 m⁻¹, so the gradient is 0 beyond the bound |
| Dynamics | World-on-Rails bicycle model; speed = softplus(·) ≥ 0 (`motion_model.py:33-55`) | Waymax `InvertibleBicycleModel`; speed can become negative |
| Ego planner | AIM-BEV trained on CARLA | AIM-BEV retrained on WOMD (`ReGentS/agent/`) |
| Gradient through the ego's reaction | off by default (`--detach_ego_path 1`, `generate_scenarios.py:367-370`). "Both paths" mode (`run_generation_both_paths.sh`) backpropagates through the BEV renderer, AIM-BEV and the PID controller, with the gradient entering the BEV image norm-clipped to 0.05 (`proxy_simulator/utils.py:104-117`, `renderer.py:315-316`) | always off: `jax.lax.stop_gradient(state)` and `torch2jax_with_vjp(..., nondiff_argnums=(0, 1, 2))` (`agent/aim_bev_actor.py:49,62`) |
| Termination inside a rollout | the whole scene freezes (no state updates) on ego collision, adversary–adversary collision, any adversary corner off-road, or route completion (`simulator.py:507,582,616,651`; `motion_model.py:50`) | none; the full log length is always simulated |
| Optimizer | Adam, lr 5e-3, β₁ 0.8, β₂ 0.99 or 0.999, 100–150 iterations (`run_generation.sh`); both-paths mode uses β (0.9, 0.999) and 75 iterations; Adam state reset per route | Adam, lr 1e-3, default β, at most 500 iterations (`conf/config_scenario_opt.yaml`); fresh solver per scenario |
| Stop criterion | ego collision (batch size 1): no update on that iteration, then break (`generate_scenarios.py:159-161,200-201`) | ego overlaps a *candidate* adversary: break before the update (`optim_scenario.py:65-67`); also stops on NaN/Inf gradients |

## 3. Loss

### 3.1 Original KING

`BatchedPolygonCollisionCost` (`king/proxy_simulator/driving_costs.py:123-245`) computes the following for every ordered pair of boxes $(a, b)$ at each timestep:

$$d_{c\to e}(a,b,t) = \min_{\text{corner } c \text{ of } a}\ \min_{\text{edge } e \text{ of } b}\ \operatorname{dist}(c, e)$$

This is an unsigned, one-directional gap in meters: corners of $a$ against edges of $b$ only. The terms are aggregated in `generate_scenarios.py:132-154`:

$$C^{ego}_{col} = \frac{1}{T}\sum_{t} \min_{i\in[1,N]} d_{c\to e}(0,i,t)$$

$$C^{adv}_{col} = \min_{t}\ \min_{i}\ \min\Big(\tau,\ \min_{j\ne i,\ j\ge 1} d_{c\to e}(i,j,t)\Big),\qquad \tau = 1.25\ \text{m}$$

$$C^{adv}_{rd} = \frac{1}{T}\sum_{t}\sum_{i}\sum_{c=1}^{4}\ \sum_{p\,\in\,\text{crop}(c)} (1-\text{road}_p)\ \mathcal{N}(p;\,x_c,\,\sigma = 1\ \text{m})$$

$$\mathcal{L}_{KING} = w_{ego}\,C^{ego}_{col} + w_{rd}\,C^{adv}_{rd} - w_{adv}\,C^{adv}_{col}$$

Notes:

- **The min over adversaries sits inside the time average.** `driving_costs.py:237` returns the per-step min, and `generate_scenarios.py:132-138` averages it over time; the outer `torch.min` there runs over a singleton axis.
- **Adversary–adversary term.** Each adversary's per-step nearest-neighbor gap is clamped at τ (`generate_scenarios.py:411-413`), then the global min over time and adversaries is taken (`:139-145`). It is 0 when N = 1.
- **Off-road term.**
  - The crop is ±32 px (±6.4 m at 5 px/m), sampled every 2 px around each corner of a 2 m × 5 m box (`driving_costs.py:25-27,49-53,71-79`).
  - Crop centers go through `.item()`, so only the Gaussian itself is differentiable.
- **Weights** (`run_generation.sh`): $w_{ego} = 1$, and $(w_{adv}, w_{rd})$ is (3, 20) for 4 agents, (5, 23) for 2 and (0, 20) for 1. The argparse defaults are $w_{adv} = 0$ and $w_{rd} = 1$.
- **Frozen states.** The scene freezes on termination (§2), so costs after an off-road or adversary–adversary event are evaluated on frozen states.

### 3.2 ReGentS repository (identical in `king` and `regents` modes)

The terms live in `ReGentS/cost.py` and are assembled in `method/optim_scenario.py:82-100`. Notation:

- $\mathcal{A}$ is the adversary candidate set (§4.2).
- $p$ denotes box centers.
- $V$ is the set of timesteps where the objects involved are valid **in the log**.

$$C^{ego}_{col} = \min_{i\in\mathcal{A}}\ \frac{1}{|V_{0i}|}\sum_{t\in V_{0i}} \lVert p_{0,t}-p_{i,t}\rVert^2$$

$$C^{adv}_{col} = \min\Big(\tau^2,\ \min_{i<j\in\mathcal{A},\ t\in V_{ij}} \lVert p_{i,t}-p_{j,t}\rVert^2\Big),\qquad \tau = 1.25$$

$$C^{adv}_{dev} = \sum_{i\in\mathcal{A}}\sum_{t\in V_i}\sum_{c=1}^{4}\ \sum_{q\,\in\,\text{crop}} m^{\text{off}}_q\ \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\lVert q-x_c\rVert^2/2\sigma^2},\qquad \sigma = 0.5\ \text{m}$$

$$\mathcal{L}_{ReGentS} = C^{ego}_{col} - 5\,C^{adv}_{col} + 20\,C^{adv}_{dev}$$

Notes:

- `calculate_distance_bounding_boxes` is a `pass` stub (`cost.py:30-31`); only the `'center'` distance exists (`cost.py:6-7`).
- $m^{\text{off}}$ is a 1 m raster rebuilt from WOMD road-edge points (`method/utils.py:61-108`), where 1 means off-road.
- The weights come from `conf/config_scenario_opt.yaml:23-29` and are the same for every scene size.

### 3.3 Side by side

| | Original KING | ReGentS repo |
| --- | --- | --- |
| Distance | box corner→edge gap, m | squared center distance, m² |
| Ego attraction | mean over t of min over adversaries | min over adversaries of mean over valid t |
| Adversary–adversary repulsion | min over t and adversaries of min(1.25 m, gap) | min over t and pairs of min(1.25², center distance²) |
| Off-road | Gaussian (σ = 1 m) × non-road mask, averaged over t | Gaussian (σ = 0.5 m) × off-road mask, summed over t; crop is broken (§3.4d) |
| Weights (ego / adv–adv / off-road) | 1 / 3–5 / 20–23 depending on density | 1 / 5 / 20 |
| Vehicles in the loss | all N spawned | candidate set $\mathcal{A}$ only |

### 3.4 Consequences (verified)

#### a) Where the min sits decides which vehicle gets pulled

Toy setup:

- The ego is static at the origin.
- Adversary 0 moves from 3 m to 30 m ahead, with a lateral offset of +3.5 m.
- Adversary 1 moves from 32 m to 4 m, with a lateral offset of −3.5 m, so it is slightly farther on average.

The table shows which adversaries receive gradient from the ego-attraction term:

| | ReGentS `calculate_distance_ego_col` | KING reduction (mean over t of min, center distance) |
| --- | --- | --- |
| Adversary 0 (close early) | gradient on 20/20 steps | gradient on 11/20 steps |
| Adversary 1 (close late) | **0** | gradient on 9/20 steps |

- **KING's form** lets different vehicles be "the closest" at different times, and pulls each one during the steps where it is closest.
- **ReGentS's form** sends the whole pull to one vehicle per iteration: the candidate with the smallest average distance.

This is the paper's *minimum trap* (§4.3): whichever vehicle wins the first iterations tends to keep winning. It is also why ReGentS has to prune the candidate set, since the loss itself cannot escape a bad argmin.

#### b) Squaring moves the pull to far timesteps

For the selected adversary above, the gradient at the last step (≈ 30 m away) is **6.6×** the gradient at the first step (≈ 4.6 m away), because $\nabla\lVert\Delta\rVert^2 = 2\Delta$. KING's unsquared gap has the same gradient magnitude at every step.

Squared distances therefore amplify the paper's *time-averaged distance bias* (Fig. 3b): the optimizer works hardest on the part of the trajectory where the vehicles are far apart.

#### c) The adversary–adversary term is almost never active in ReGentS

Two same-lane vehicles, each 4.5 m long, evaluated with `calculate_distance_adv_col` (τ = 1.25):

| Center gap | Bumper gap | Value | Gradient norm |
| --- | --- | --- | --- |
| 5.0 m | +0.5 m | 1.5625 (clipped) | 0 |
| 3.0 m | −1.5 m (overlapping) | 1.5625 (clipped) | 0 |
| 1.0 m | −3.5 m | 1.0 | 4.0 |

In KING, τ applies to the box gap, so the 5.0 m case (0.5 m gap) is already repelled. In ReGentS the term only reacts once two centers are within 1.25 m, when the vehicles already overlap by more than 3 m.

In both codebases the term is a single global min, so at most one pair at one timestep is pushed apart per iteration.

#### d) The ReGentS off-road term is effectively zero (bug)

`cost.py:96`:

```python
cropped_binary_map = jax.lax.dynamic_slice(binary_map, (jnp.minimum(x_center-cropsize//2, 0), jnp.minimum(y_center-cropsize//2, 0)), (cropsize, cropsize))
```

Two things go wrong:

1. `jnp.minimum(·, 0)` makes every start index ≤ 0; `jnp.maximum` was presumably intended.
2. `lax.dynamic_slice` wraps negative starts around from the end and then clamps, so the crop always lands on one of the map's corners:

| Start passed | Cells actually sliced (200-cell axis) |
| --- | --- |
| 0 | 0…31 |
| −6 | 168…199 |
| 84 (what `jnp.maximum` would give for cell 100) | 84…115 |

The x index is also applied to the row (y) axis of the map. Measured on a 200 m × 200 m map with a 10 m wide road (20 steps, 4 corners):

| Adversary | Released code: value / gradient norm (x, y) | With `jnp.maximum` and rows = y |
| --- | --- | --- |
| straddling the road edge (100, 104.5) | 0 / 0 | 19.7 / 48.3 |
| deep off-road (100, 150) | 0 / 0 | 101.4 / 0 |

On a map that is off-road everywhere, the released term is non-zero only for vehicles about 16–31 m from the map's minimum corner:

- non-zero: (20, 20) gives 25.4, (28, 28) gives 25.3;
- zero: (10, 10), (34, 34) and (100, 100).

So in the released code, neither mode has a working off-road penalty.

The corrected version also exposes a limitation shared by both methods: deep inside a uniform off-road region the Gaussian potential is flat (gradient 0), so it only steers vehicles that are near the road edge.

## 4. Adversary selection

### 4.1 KING: a fixed set, chosen implicitly

- **Where adversaries come from.** They are the N vehicles in the route's initialization file (`simulator.py:154-221`).
  - In `initializations_subset/`, each of the 120 routes per density has `adv_spawn_points`, `adv_routes` and an 80-step `action_seq` per vehicle.
  - Spawn points lie a median 33 m (N = 1) to 62 m (N = 2 or 4) from the ego route start, with a minimum of 3.8 m and a maximum of 158 m.
  - The script that produced these files is not in the repository.
- **Nothing is excluded.** Every spawned vehicle is optimized, whether it is static, behind the ego, or anything else.
- **The collider emerges from optimization.** At each timestep the pull goes to the currently closest vehicle (§3.4a).
- **Success** means the ego overlaps any spawned vehicle, using a separating-axis check (`simulator.py:573-578`).
- **Avoidability is only assessed afterwards.** `tools/determine_solvability.py` replays each final scenario with the privileged `AutoPilot` expert as ego (it is given the adversary actions) and reports the fraction it survives. No script in the repository uses this result to filter scenarios.

### 4.2 ReGentS: candidate filtering (`optim_scenario.py:15-36`)

| Rule | Mode | Computed from | Excluded when |
| --- | --- | --- | --- |
| Ego | both | metadata | `is_sdc` |
| Non-vehicle | both | metadata | `object_types != 1` (pedestrians, cyclists, …) |
| Rarely valid | both | log | valid in less than 50 % of frames |
| Static | `regents` | log, once | ‖p_last − p_first‖ < 0.2 m (net displacement only) |
| Persistent rear | `regents` | log, once | bearing from the **logged** ego within ±22.5° of its rear axis in more than 80 % of frames |

$$\text{rear}_i = \frac{1}{T}\sum_{k} \mathbb{1}\Big[\big|\operatorname{wrap}(\alpha^{log}_{i,k} - \psi^{log}_{0,k})\big| > \tfrac{7\pi}{8}\Big] > 0.8,\qquad \alpha_{i,k} = \operatorname{atan2}(p_{i,k} - p_{0,k})$$

Remarks:

- **The rear rule is purely positional (verified).** Both a vehicle following the ego and an oncoming vehicle that has already passed (bearing 160°, yaw 180°) are excluded. A vehicle at bearing 150° is kept.
- It uses the logged ego trajectory, not the AIM-BEV ego that is actually simulated.
- Neither rule applies a validity mask when averaging over frames.
- There is no danger score and no explicit pick. Candidates stay in the pool, and the min in $C^{ego}_{col}$ does the choosing (§3.4a).

### 4.3 ReGentS: steering-update masking (`optim_scenario.py:53-59,69-77`)

The flag is recomputed at every iteration from the **current simulated** trajectories. With $\Delta\psi = \operatorname{wrap}(\psi_i-\psi_0)$ and $\Delta\alpha = \operatorname{wrap}(\alpha_i-\psi_0)$:

$$\text{divergent}_i = \frac{1}{T}\sum_{k} \mathbb{1}\Big[\,0 < \tfrac{\Delta\alpha}{\Delta\psi} < 1\ \wedge\ |\Delta\alpha| < \tfrac{\pi}{8}\ \wedge\ |\Delta\psi| < \tfrac{\pi}{2}\,\Big] > 0.5$$

In words: the vehicle is ahead of the ego within ±22.5°, and its heading offset is on the same side as its position offset and larger than it. The vehicle points away from the ego's path (the red zone in paper Figs. 4b and 5).

The masks are applied to Adam's output (`updates`), not to the gradient:

| Update | `king` mode | `regents` mode |
| --- | --- | --- |
| Ego, all actions | ×0 | ×0 |
| Acceleration, any candidate | ×1 | ×1 |
| Steering, divergent vehicle | ×1 | **×0** |
| Steering, every other vehicle | ×1 | **×0.5** |

Toy geometries (verified): ego at the origin heading +x, vehicle 20 m away.

| Bearing | Yaw | Divergent |
| --- | --- | --- |
| +5° | +10° | yes |
| +10° | +5° | no (points back toward the ego's path) |
| −5° | +10° | no (opposite sides) |
| +5° | 0° | no (Δψ = 0) |
| +5° | +60° | **yes in code**; no under the paper's ±22.5° yaw window |

Remarks:

- **Momentum keeps building while frozen.** Masking happens after `solver.update`, so Adam's moment estimates keep accumulating the raw steering gradient while a vehicle is frozen. That accumulated momentum is applied as soon as the vehicle leaves the zone.
- **Freezing does not reset steering.** A frozen vehicle keeps its current steering values; only the update is zeroed.
- **Excluded vehicles are not explicitly masked, but they still get no update.** Two facts make their gradient exactly 0:
  - each object's rollout depends only on its own actions (Waymax applies the bicycle update per object);
  - the ego path is `stop_gradient`.

  Adam's update from all-zero moments is then 0. This is reasoned from the code, not run end to end.

### 4.4 Which collisions count, and which are avoided

| | Original KING | ReGentS repo |
| --- | --- | --- |
| A collision counts as success when the ego hits | any spawned vehicle | a candidate only (`method/utils.py:47-59`) |
| Adversaries are repelled from | each other | other candidates only |
| Adversary hitting an excluded vehicle (parked car, pedestrian, rear vehicle) | n/a: every vehicle is an adversary | not rewarded, not penalized, not detected |

The same candidate set answers both "who may attack" and "who must be avoided". An adversary can therefore be optimized straight through a parked car at no cost.

## 5. ReGentS paper vs ReGentS code

| Topic | Paper | Code | Effect |
| --- | --- | --- | --- |
| Vehicle distance | bounding-box distance $d_{BB}$ (Eq. 1) | squared center distance | §3.4b, §3.4c |
| KING's ego cost | min over adversaries of the time mean (Eq. 4) | same in the ReGentS code; original KING code uses the mean of the per-step min | §3.4a |
| Adversary–adversary threshold | τ on $d_{BB}$ | τ² on the squared center distance | repulsion only in deep overlap |
| Off-road cost | Gaussian × drivable-area map | crop always lands on a map corner | term ≈ 0 |
| Divergence yaw window | ψᵢ − ψ₀ within ±22.5° (§4.3) | within ±90° | more vehicles have their steering frozen |
| Divergence persistence τ_front | value not given | 0.5 | — |
| Steering of non-flagged vehicles | unchanged | update ×0.5 | `regents` runs at half of `king`'s steering step size, a confound in Table 1 |
| Rear-end zone | 45° sector behind the ego, "most time steps" | ±22.5°, more than 80 % of logged frames | matches |
| Static vehicles | "static adversaries" | net displacement < 0.2 m | a vehicle that loops back to its start counts as static |
| Gradient | "exact gradient … in contrast to [KING]" (§5.1) | the ego's reaction is still detached (`stop_gradient`), like KING's default; KING's optional both-paths mode goes further | — |

## 6. Implications for the PufferDrive port

For each item below, decide explicitly whether to follow the paper or the released code:

1. **Distance.** Box gap (paper, KING) or squared center distance (ReGentS code). Squared center distance overweights far timesteps and makes τ = 1.25 nearly meaningless.
2. **Ego-attraction reduction.** Min-of-mean (paper Eq. 4 and ReGentS code; the filters and masks were designed around it) or KING's mean-of-min (spreads the pull, weaker minimum trap).
3. **Adversary–adversary threshold.** Define τ on the box gap; otherwise the term is inactive.
4. **Off-road term.** Do not copy the crop bug. Note that reproducing the *released* numbers implies an effectively disabled off-road term.
5. **Obstacle set.** Consider penalizing adversary collisions with *excluded* vehicles, which the loss currently cannot see.
6. **Divergence rule.** Choose the yaw window (±22.5° paper, ±90° code), whether to keep the ×0.5 steering scale, and whether to mask before or after the optimizer's moment update.
7. **Rear rule.** Logged ego (code) or simulated ego, and whether it should stay position-only.

## Appendix A: how the numbers were produced

- **Environment.** `ReGentS/cost.py` was imported unmodified, using JAX 0.5.3 from `~/Workspace/v-max/.venv_3.10_v_max` with `JAX_PLATFORMS=cpu`.
- **Divergence and rear rules.** The rules in §4.2–4.3 were copied verbatim from `method/optim_scenario.py` into the test script. Importing that module directly would require WOMD data and the AIM-BEV checkpoint.
- **KING's reduction.** For §3.4a it was re-implemented in JAX with center distances, to isolate the effect of the reduction order. KING's own PyTorch cost needs CUDA and was not run.
- **KING's diagonal-removal indexing** (`driving_costs.py:235`) was checked with a labeled 4×4 matrix:
  - row 0 is ego corners → adversary edges;
  - block `[1:, 1:]` is adversary corners → other adversaries' edges.
- **KING initialization statistics** come from parsing `king/driving_agents/king/aim_bev/king_initializations/initializations_subset/*/RouteScenario_*.json` against the first waypoint of each route in `king/leaderboard/data/routes/subset_20perTown.xml`.

Minimal reproduction of the off-road crop bug:

```python
import jax, jax.numpy as jnp

axis_cells = jnp.arange(200)
print(jax.lax.dynamic_slice(axis_cells, (jnp.minimum(100 - 16, 0),), (32,))[jnp.array([0, -1])])  # [ 0 31]    agent at cell 100
print(jax.lax.dynamic_slice(axis_cells, (jnp.minimum(10 - 16, 0),), (32,))[jnp.array([0, -1])])   # [168 199]  agent at cell 10
```
