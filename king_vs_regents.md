# KING vs ReGentS

## Main finding

ReGentS keeps KING's objective structure and adds rules for adversary eligibility and steering updates. In the ReGentS repository, `method.type: king` and `method.type: regents` call the same loss function. Differences between original KING and that shared implementation should therefore be distinguished from ReGentS's added rules.

The paper's Table 1 "KING" baseline (48.99% success vs 60.40% for ReGentS) was produced on WOMD, so it comes from a Waymax re-implementation (presumably this `king` mode), not from the original CARLA-based KING code.

This comparison is based on static inspection of:

- [Original KING](king/), included in this repository.
- [ReGentS](../ReGentS/), the neighboring repository, which also implements a KING baseline mode.
- [The local ReGentS paper](2409.07830v1.pdf), especially §4.3.

Neither optimizer was run for this comparison. The ReGentS cost functions were later run on toy inputs (JAX 0.5.3, CPU) to check section 5; see [king_vs_regents_claude.md](king_vs_regents_claude.md) for details.

## 1. Loss

Both implementations minimize an objective with three purposes:

```text
loss = ego_collision_weight × ego_distance
     - adversary_separation_weight × capped_adversary_separation
     + road_weight × offroad_penalty
```

Minimization encourages an adversary to approach the ego, discourages collisions between adversaries, and penalizes leaving the road. These are soft penalties, not guarantees.

| Detail | Original KING | ReGentS repository: both KING and ReGentS modes |
| --- | --- | --- |
| Vehicle distance | Distance computed from box corners and edges | Squared distance between vehicle centers |
| Ego collision reduction | Find the closest adversary at each timestep, then average over time | Average distance over time for each adversary, then choose the minimum |
| Adversary separation | Minimum separation over adversaries and time, capped at a threshold | Same broad structure, using squared center distance and a squared threshold |
| Road penalty | Gaussian overlap with non-drivable raster near vehicle corners, averaged over time | Gaussian potential near vehicle corners, summed over valid trajectory samples |

The ego collision reductions are different:

```text
Original KING:    mean_over_time(min_over_adversaries(box_distance))
ReGentS code:    min_over_adversaries(mean_over_time(squared_center_distance))
```

Original KING can follow different closest adversaries at different timesteps. The ReGentS implementation favors the candidate with the smallest average distance across the trajectory; this candidate can change between optimization iterations.

Squaring also shifts the attraction toward timesteps where the vehicles are far apart: the gradient grows with distance (6.6× larger at 30 m than at 4.6 m in a toy check), whereas KING's unsquared distance gives the same gradient magnitude at every timestep.

The ReGentS default configuration uses:

```text
loss = ego_distance - 5 × capped_adversary_separation + 20 × offroad_penalty
separation_threshold = 1.25 meters
```

There is no additional loss term enabled by switching from `king` to `regents`. Weights are not directly comparable across repositories because the distance definitions and reductions differ.

Sources: [KING loss aggregation](king/generate_scenarios.py), [KING geometry and road costs](king/proxy_simulator/driving_costs.py), [ReGentS costs](../ReGentS/cost.py), [shared loss construction](../ReGentS/method/optim_scenario.py), [ReGentS configuration](../ReGentS/conf/config_scenario_opt.yaml).

## 2. Adversary selection

Original KING loads a configured set of adversaries from scenario initialization files and optimizes their action sequences. Its default adversary count is four.

In the ReGentS repository, both modes exclude the ego, non-vehicle objects, and objects valid for less than 50% of the recorded trajectory. ReGentS mode adds two exclusions:

| Exclusion | Implementation | Motivation |
| --- | --- | --- |
| Static vehicles | Start-to-end displacement below 0.2 m | Avoid optimization getting stuck on stopped vehicles |
| Persistent rear vehicles | Within ±22.5° of directly behind the ego for more than 80% of recorded timesteps | Avoid rear-end attacks that give the ego little opportunity to respond |

These filters are computed once from the original recorded trajectories. Multiple eligible candidates remain; there is no separate danger score that selects one adversary. The minimum in the ego collision loss implicitly selects which candidate receives the direct attraction gradient, while regularization can affect other candidates.

Sources: [KING initialization](king/proxy_simulator/simulator.py), [KING defaults](king/generate_scenarios.py), [ReGentS candidate selection](../ReGentS/method/optim_scenario.py).

## 3. Steering updates

ReGentS addresses a gradient issue where a vehicle ahead of the ego makes unnecessary turns instead of creating a conflict through deceleration.

At each optimization iteration, it checks the simulated trajectories. Let `bearing` be the angle of the adversary's position relative to the ego's heading, and `relative_yaw` its heading relative to the ego. A vehicle is flagged when all these conditions hold for more than 50% of timesteps:

- Its bearing is within ±22.5° ahead of the ego.
- Its relative yaw is within ±90° (the paper states ±22.5°).
- `0 < bearing / relative_yaw < 1`.

After Adam computes the action updates, ReGentS modifies them:

| Update | KING mode | ReGentS mode |
| --- | --- | --- |
| Acceleration | Normal update | Normal update |
| Steering of flagged vehicle | Normal update | Zero update |
| Steering of other vehicles | Normal update | Half-sized update |

The flag applies to the vehicle's entire action sequence for that iteration. Zeroing the update preserves its current steering values; it does not set steering to zero or restore the original trajectory. Acceleration remains adjustable. The flag is recomputed on the next iteration.

The paper does not describe the ×0.5 factor for non-flagged vehicles, so ReGentS mode also takes half of KING mode's steering step, which confounds the Table 1 comparison. Because the mask is applied after Adam, Adam's moment estimates keep accumulating the raw steering gradient while a vehicle is flagged, and that momentum is applied once the flag clears.

Source: [ReGentS optimization loop](../ReGentS/method/optim_scenario.py).

## 4. Simulation and optimization setup

These differences concern the original KING repository versus the ReGentS repository. They are shared by the latter's `king` and `regents` modes, so they are not additional effects of enabling ReGentS's rules.

| Aspect | Original KING | ReGentS repository |
| --- | --- | --- |
| Scenario source | Synthetic CARLA scenarios | Recorded Waymo Open Motion Dataset scenarios |
| Differentiable simulation | Separate PyTorch proxy simulator for the CARLA setting | JAX-based Waymax simulation |
| Scene size | Supplied generation script runs 1, 2, or 4 adversaries | Configuration allows up to 32 objects total, including ego and objects that may be excluded |
| Initial background actions | Loaded non-critical action sequences from scenario initialization files | Expert actions obtained from recorded trajectories through Waymax's dynamics |
| Ego planner | Supports AIM-BEV and TransFuser options | Uses AIM-BEV adapted to the Waymo setting |
| Gradient through the ego's reaction | Detached by default; an optional mode backpropagates through the ego planner and its BEV rendering | Always detached (`jax.lax.stop_gradient`) |
| Rollout termination | The scene freezes after an ego collision, a collision between adversaries, an adversary leaving the road, or route completion | None; the full recorded horizon is always simulated |
| Optimizer | PyTorch Adam; default learning rate 0.005 | Optax Adam; configured learning rate 0.001 |
| Optimization budget | CLI default 151 iterations; supplied script uses 100–150 depending on adversary count | Configured maximum 500 iterations, with an early exit on detected ego–candidate collision |

Both approaches optimize action sequences through vehicle dynamics; the change to Waymax does not turn ReGentS into an RL-trained adversary policy.

Loss weights also depend on the supplied configuration. For example, KING's four-adversary generation command uses ego/separation/road weights of 1/3/20, while ReGentS's configuration uses 1/5/20. KING's other commands use different weights. These are experiment settings, not a universal mathematical distinction between the approaches.

Sources: [KING generation script](king/run_generation.sh), [KING initialization](king/proxy_simulator/simulator.py), [KING options and optimizer](king/generate_scenarios.py), [ReGentS scenario setup](../ReGentS/generate_scenario.py), [ReGentS simulation](../ReGentS/simulation.py), [ReGentS configuration](../ReGentS/conf/config_scenario_opt.yaml), and the [paper, §5.1](2409.07830v1.pdf).

## 5. Checked numerically

These checks ran the unmodified ReGentS cost functions on toy inputs. They concern the released implementation shared by both modes.

### 5.1. The road penalty is effectively zero

In [`cost.py`](../ReGentS/cost.py), the crop around each vehicle corner starts at `jnp.minimum(x_center-cropsize//2, 0)`, which is never positive. `jax.lax.dynamic_slice` wraps negative starts around from the end of the map and then clamps them, so the crop always lands on a map corner instead of around the vehicle. The x index is also applied to the map's row axis.

For a vehicle straddling a road edge, the released code returns a cost of 0 and a gradient of 0. With `jnp.maximum` and rows indexed by y, the same case returns 19.7 with a gradient norm of about 48. The term is non-zero only for vehicles about 16–31 m from the map's minimum corner, so neither mode has a working road penalty.

### 5.2. The separation penalty only acts once vehicles deeply overlap

The 1.25 m threshold is applied to the squared center distance. Two 4.5 m vehicles in the same lane with centers 5 m apart (0.5 m bumper gap) or 3 m apart (already overlapping) receive zero gradient; the term only pushes them apart at a 1 m center distance. In original KING, the threshold applies to the box gap, so the 5 m case is already repelled.

## Summary: what ReGentS adds to KING

The comparison has two levels:

1. **Method changes:** ReGentS excludes static and persistent rear vehicles from adversary candidates, cancels steering updates for vehicles with the front-divergence geometry, and halves the other steering updates. The objective formula stays the same between the two modes.
2. **Changes in the released implementation and experimental setting:** the ReGentS repository uses Waymax and recorded Waymo scenarios, changes the distance geometry and loss reductions, and supplies different optimization settings.

For PufferDrive, the first group identifies the ReGentS-specific rules. The second group identifies additional choices needed to match its released implementation. Two released behaviors should not be copied as-is: the broken road-penalty crop and the separation threshold on center distance (section 5).
