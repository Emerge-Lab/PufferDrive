# Merge Plan: `vcha/turbostream` → `ev/merge_turbostream` (off `3.0`)

Working branch: `ev/merge_turbostream`, forked from `origin/3.0` at `2b93c149`.

Target: integrate the turbostream feature additions while preserving the
3.0 changes we rely on (GPU renderer from PR #400, libx264 threads cap
from PR #403, goal spawn outside radius from PR #399, variable-agent
spawning, training loop, etc.).

## Strategy

**Piecemeal port onto 3.0 as the base.** A direct `git merge` would
produce hundreds of conflicts and force us to redo the renderer work.
Instead, land turbostream features as individual commits on this branch,
each small enough to audit on its own. Each phase below is intended to
land as a single commit (or a tight sequence of commits) with a clear
description.

## Things to preserve from 3.0 (do NOT pull from turbostream)

These are features on 3.0 that are either absent on turbostream or
present in a worse form. Guard against accidentally reverting them
during the merge.

| Feature | Where it lives on 3.0 | Why keep it |
|---|---|---|
| **GPU/PBO headless rendering** | `pufferlib/ocean/drive/egl_headless.h`, `make_client` and the PBO readback/writev loop in `drive.h`, `polyline_max_segment_length` and `road_cache` | PR #400. Turbostream has no EGL path. Regressing here would take eval render from ~30 fps back to ~1 fps software rendering. |
| **libx264 `-threads 4` cap** | `drive.h` `make_client` execlp | PR #403. Without this, eval renders hang on multi-core nodes (SLURM cgroup oversubscription). |
| **`active_step_count` metric fix** | `pufferlib/ocean/drive/drive.h` Log struct + `add_log` + `c_step` reward loop | PR #402. Fixes the stopped-agent dilution bug. Port this onto turbostream's metric indices. |
| **Partner obs velocity in ego frame** | `pufferlib/ocean/drive/drive.h` `compute_partner_observations` | PR #404. Emits `(rel_vx_ego, rel_vy_ego)` instead of turbostream's scalar `sim_speed`. More information for the policy. |
| **Goal spawn outside radius (PR #399)** | `pufferlib/ocean/drive/drive.h` goal generation | PR #399 merged to 3.0. Keep 3.0's version. |
| **Variable-agent spawning** (`init_variable_agent_number`) | `set_active_agents`, `spawn_agents_with_counts` | Current training config uses this. Optional to keep — see open question below. |
| **Current reward randomization bounds** | `drive.ini` `reward_bound_*` | Already aligned to GIGAFLOW spec via PR #401-ish effort. Keep 3.0's values. |
| **`rebuild_on_cluster.py` `TORCH_CUDA_ARCH_LIST`** | `scripts/rebuild_on_cluster.py` | Already ported in commit `11bc54ca` on this branch. |

## Feature ports (ordered, each a separate commit)

### Phase 1 — Build script multi-arch fix ✅

**Status**: done in `11bc54ca`.

Ports the `TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"` export into
`rebuild_on_cluster.py` so multi-arch builds cover A100, L40S,
H100/H200 instead of only the build node's GPU type.

Not a turbostream port — this is a standalone fix that belongs on
this branch before anything else so subsequent cluster rebuilds
work on every node type.

### Phase 2 — `compute_metrics` / `compute_rewards` split

**What it does**: separates event detection from reward application in
the per-step loop. Gives us a single function that audits rewards
independently of metrics, which is a prerequisite for any of the later
reward-logic changes.

**Turbostream files**: `pufferlib/ocean/drive/drive.h`

On turbostream, `compute_agent_metrics` (from 3.0) has been replaced by
`compute_metrics(env, i)` which writes per-agent state (metrics_array,
collision_state, etc.) and `compute_rewards(env, i)` which reads that
state and applies `env->rewards[i] +=` with explicit leading `-` on
penalty terms.

**Port targets**:
- Split 3.0's `compute_agent_metrics` at `drive.h:~2622` into `compute_metrics` + `compute_rewards`
- Keep 3.0's ini semantics (coefs stored with negative sign, no leading `-`)
- Preserve 3.0's `if (agent->stopped) continue;` at the top of the reward block
- Preserve 3.0's `active_step_count += 1` increment (from PR #402)

**Dependencies**: none (can land first after Phase 1)

**Risk**: low. Pure refactor of an existing function. No behavior change.

**Verification**: rebuild, launch a 1-epoch run, confirm rewards + metrics
match the pre-split baseline bit-for-bit on a fixed seed.

### Phase 3 — OBB collision detection

**What it does**: replaces `check_aabb_collision` (3.0) with
`check_obb_collision` (turbostream). Oriented bounding boxes handle cars
at arbitrary headings correctly, where AABB either over-rejects or
under-rejects collisions for rotated vehicles.

**Turbostream files**: `pufferlib/ocean/drive/drive.h` collision check
region (roughly `drive.h:~2400` on 3.0)

**Port targets**:
- Replace `check_aabb_collision` call sites in `compute_metrics`
- Also pull in `check_z_collision_possibility` to replace `check_z_collision`

**Dependencies**: Phase 2 (so we can cleanly edit the metric detection
path without also editing reward application)

**Risk**: medium. Collision detection determines when events fire, which
affects rewards and terminations. Validate against 3.0's behavior on a
deterministic run (same seed, same actions, compare collision counts).

**Verification**: launch one job, compare collision_rate and
offroad_rate vs a 3.0 baseline. Expect slightly different absolute
values (OBB is more accurate) but same order of magnitude.

### Phase 4 — Traffic control (red lights, stop lines, stop signs)

**What it does**: implements the traffic control state machine that 3.0
has scaffolding for but no working code path. turbostream fires
`RED_LIGHT_IDX` and applies a reward penalty when an agent crosses a red
light.

**Turbostream files**:
- `drive.h`: `generate_traffic_light_states`, `check_lane_change_red_light`,
  `check_red_light_violation`, `check_spawn_red_light_violation`,
  `check_stop_line_crossing`, `traffic_control_in_scope`
- `datatypes.h`: `NUM_TRAFFIC_CONTROL_STATES`, `NUM_TRAFFIC_CONTROL_TYPES`,
  `RED_LIGHT_IDX`, Agent `stop_line[6]`
- `drive.h` Drive struct: `max_traffic_control_observations`,
  `traffic_control_scope`, `traffic_light_behavior`
- `binding.c`: new kwargs unpacking for the traffic control fields

**Port targets**:
- Add the traffic control functions, state, and reward wiring
- Add a new observation block for traffic control entities (distinct
  from road observations)
- Update `drive.ini` with the new config keys

**Dependencies**: Phase 2 (`compute_rewards` needs to exist), Phase 3
(OBB collision interacts with traffic control via stop-line geometry)

**Risk**: medium-high. Introduces a new observation block that changes
obs layout (breaks checkpoint compatibility). Also interacts with
training metrics — need to verify `red_light_violation_rate` gets
populated correctly.

**Verification**: launch one job. Check that `red_light_violation_rate`
is no longer always zero in wandb logs, and that episode_return
decreases slightly in scenarios with red lights (due to the new
penalty firing).

### Phase 5 — Time-to-collision (TTC) subsystem

**What it does**: introduces a TTC estimator that computes the closest
approach time between each pair of agents using a circle-circle
intersection with relative velocity. Gives the policy a direct
"seconds until we hit" signal.

**Turbostream files**:
- `drive.h`: `compute_agent_ttc`, `compute_pairwise_ttc`,
  `default_ttc_result`, `ttc_update_min_result`, `is_at_fault_collision`
- `datatypes.h`: `struct ttc_result`, `MIN_TTC_IDX`,
  `AT_FAULT_COLLISION_IDX`, Agent fields `min_ttc`, `ttc_samples`,
  `ttc_violations`, `closing_speed`, `distance_to_collision`,
  `other_idx`, `cached_ttc`

**Port targets**:
- Add the TTC struct and computation functions
- Wire `compute_agent_ttc` into the per-step loop before `compute_metrics`
- Expose `min_ttc` and `at_fault_collision` as new metric slots
- Optionally emit TTC in the observation for partners

**Dependencies**: Phase 2 (`compute_metrics` split), Phase 3 (OBB
collision provides the pairwise geometry used in TTC)

**Risk**: medium. TTC computation is O(N²) per step (but N is ~100,
so tractable). Validate CPU cost doesn't regress SPS more than ~5%.

**Verification**: launch one job, confirm `min_ttc` appears as a new
wandb metric, confirm SPS is within 5% of baseline, confirm
`at_fault_collision` count is < `collision_count`.

### Phase 6 — Waypoint / path / progression system

**The big one.** This is the deepest architectural change in turbostream
and has the biggest merge surface.

**What it does**: replaces 3.0's single-point goal (`goal_position_x/y/z`,
`sample_new_goal`, `respawn_agent`) with a route of waypoints along a
planned lane path (`path_progression`, `num_target_waypoints`,
`goal_positions_z[MAX_TARGET_WAYPOINTS]`). The agent progresses along a
route and gets a per-waypoint reward, with a final terminal bonus for
reaching the end of the route (gated by `goal_speed_threshold`).

**Turbostream files**:
- `drive.h`: `build_path`, `compute_new_route`, `generate_random_route`,
  `compute_progression`, `compute_remaining_lane_distance`,
  `compute_lane_length`, `compute_lane_end_distance_sq`,
  `get_closest_waypoint_index_on_path`, `initialize_agent_progression`,
  `reset_agent_path_progression`, `score_lane_candidate`,
  `compute_multi_segment_alignment`, `find_closest_segment_on_lane`
- `datatypes.h`: `struct LaneGraph`, Agent fields `path_progression`,
  `multi_lane_time`, `route_gt_len`, `num_target_waypoints`,
  `current_lane_idx`, `previous_lane_idx`, `n_lanes`, `lane_ids`,
  `lane_lengths`, `headings`, `distances`, `goal_positions_z[]`
- `drive.h` reward path: the waypoint disjunction `(1_waypoint ∨ |v|<v_goal)`
  I traced earlier

**Port targets**:
- Add the LaneGraph struct and the lane-indexing machinery
- Add the path building / progression tracking functions
- Replace 3.0's `sample_new_goal` / `respawn_agent` with
  `compute_new_route` / `reset_agent_path_progression`
- Update the goal-reward check in `compute_rewards` to handle waypoints
  correctly (fire `reward_goal` for each intermediate waypoint regardless
  of speed, fire the terminal bonus only when `|v| < goal_speed_threshold`)
- Remove 3.0's `goal_behavior`, `min_goal_distance`, `max_goal_distance`,
  `min_goal_speed`, `max_goal_speed` config keys

**Dependencies**: Phase 2 (reward split), Phase 3 (OBB — used in path
scoring), Phase 4 (traffic control — route awareness of lights/stops),
Phase 5 (TTC — interacts with path progression for stuck detection)

**Risk**: high. This reshapes the goal lifecycle end-to-end and is
incompatible with 3.0's `sample_new_goal` / `goal_behavior` paths.
Trained policies from 3.0 won't transfer directly (different reward
surface). Requires a retraining cycle to validate.

**Open question**: do we keep 3.0's `sample_new_goal` as a fallback
`simulation_mode` alongside turbostream's waypoint path, or cut over
entirely? See the open question section below.

**Verification**: launch 2 runs (new layout) and compare episode_return
+ goals_reached trajectories vs a 3.0 baseline. Expect different
absolute values but similar learning curve shape.

### Phase 7 — Multi-scenario eval pipeline

**What it does**: replaces 3.0's SafeEvaluator (in-process) with
turbostream's trajectory-based eval that collects ground-truth and
simulated trajectories, computes histogram-based metrics (distance,
heading, speed distributions), and produces a metametric score.

**Turbostream files**:
- `pufferl.py`: `eval_multi_scenarios`, `eval_multi_scenarios_render`,
  `build_eval_overrides`, `_export_metrics`, `_log_eval_metrics`,
  `verify_scenario_coverage`, `verify_scenario_coverage_gigaflow`,
  `load_eval_multi_scenarios_config`, `_save_experiment_config`,
  `_get_git_metadata`, `upload_model`
- `pufferlib/ocean/benchmark/evaluator.py`: completely new class with
  `collect_ground_truth_trajectories`, `collect_simulated_trajectories`,
  `compute_metrics`, `_compute_metametric`, `_get_histogram_params`,
  `rollout`, `_quick_sanity_check`

**Port targets**:
- Import the new evaluator module as a sibling to `SafeEvaluator`
- Add the new pufferl entry points
- Decide whether SafeEvaluator stays or gets replaced
- Add CLI flags: `--num_scenarios`, `--render`, `--video-path`

**Dependencies**: Phase 6 (waypoint system — the multi_scenarios eval
path assumes waypoint-based scenarios)

**Risk**: medium. Changes the eval workflow and the metrics shown in
wandb during eval. Training itself is unaffected. Backward compat on
the CLI side is required since existing SLURM configs reference
`--sanity-maps` and similar.

**Verification**: run a single eval from a trained checkpoint and
confirm the metametric output matches expected WOSAC-style metrics.

### Phase 8 — PPO train loop split

**What it does**: turbostream splits the single `train()` into two
class methods — `_train_ppo_trajectory` (episode-based) and
`_train_ppo_transition` (step-based) — with a shared `_ppo_loss`.
Also adds `early_stop_fn` hook to `train`.

**Turbostream files**:
- `pufferl.py`: `_ppo_loss`, `_train_ppo_trajectory`, `_train_ppo_transition`,
  `train` signature change

**Port targets**:
- Refactor 3.0's monolithic `train` loop into the split layout
- Preserve 3.0's `clamp_reward`, `is_invalid_step` masking, and wandb
  logging paths

**Dependencies**: Phase 7 (eval pipeline — the train loop references
the new eval functions)

**Risk**: medium-high. This is the hot loop — any subtle bug reshapes
the gradient signal. Needs a direct A/B against a pre-split baseline.

**Verification**: launch 2 runs on the split loop, 2 on the pre-split
loop, same seed, confirm loss curves match within 5% over the first
1B steps. If they diverge, the split has a bug.

### Phase 9 — Additional turbostream features (optional)

These are smaller additions that can land individually:

- `update_agent_speed` maintaining `Agent.sim_speed` (currently dead
  on 3.0) — cleanup, unblocks future refactors
- `invalidate_agent` for cleaner deactivation than STOP_AGENT
- Cached `cos_heading` / `sin_heading` on Agent (perf win,
  fewer per-frame trig calls)
- `compute_displacement_error` for `avg_displacement_error_rate` metric
- `compute_euclidean_distance` helper (replaces scattered 3D distance
  calls)
- Lane-aware observation ordering (via `LaneGraph`)
- `simulation_mode` dispatch (gigaflow vs replay) — useful for the
  eval path but not required for training

Each of these is a few-file change with low risk.

## Things we won't port

| Feature | Reason |
|---|---|
| **Turbostream's `render.h`** | 3.0's GPU-accelerated renderer (PR #400) is strictly better. Turbostream uses raylib's default software path. |
| **Scalar `sim_speed` partner obs** | Our PR #404 already emits `(rel_vx_ego, rel_vy_ego)` which is strictly more information. |
| **`num_agents_per_env` fixed-count mode** (no variable spawning) | Current training uses `init_variable_agent_number`. Keep 3.0's variable spawn path. See open question. |
| **Metametric eval only** | Keep SafeEvaluator as well so in-process eval stays available during training runs. |
| **Turbostream's polyline-not-simplified approach** | Keep 3.0's `simplify_polyline` + `create_sparse_lane_points` since they're needed for the VBO road cache in PR #400. |

## Open questions

1. **Variable-agent-number spawning**: turbostream doesn't have it, but
   all current training uses it. Options:
   - (a) Keep 3.0's variable spawn path as an additional `init_mode` alongside turbostream's fixed-count paths
   - (b) Drop variable spawning entirely and switch training to fixed-count
   - (c) Make variable spawning a turbostream `simulation_mode` variant

   Decision needed before Phase 6 (waypoint system) since the route-building
   code assumes known agent count at init time.

2. **`goal_behavior` replacement**: turbostream assumes waypoint paths
   always. 3.0 supports `GOAL_RESPAWN`, `GOAL_GENERATE_NEW`, `GOAL_STOP`.
   Should we:
   - (a) Keep 3.0's `goal_behavior` modes alongside turbostream's waypoint path (one as a different `simulation_mode`)
   - (b) Collapse to waypoint-only, retraining from scratch

   Affects Phase 6.

3. **Observation layout change**: turbostream splits road observations
   into lane-segment and boundary-segment blocks with independent caps.
   Adopting this breaks checkpoint compatibility.
   - (a) Keep 3.0's single `MAX_ROAD_SEGMENT_OBSERVATIONS` block
   - (b) Migrate to turbostream's split blocks (fresh training required)

   Affects Phase 4 and Phase 6.

4. **`PARTNER_FEATURES`**: we pushed to 9 in PR #404 for 2D rel-v in
   ego frame. Turbostream uses 8 with scalar speed. Confirm we want to
   keep the 9-feature layout through the merge (it's strictly more
   informative, but means every new feature we port from turbostream
   that touches partner obs has to be adjusted for the extra slot).

5. **Reward encoding sign convention**: turbostream uses positive α with
   explicit leading `-` in penalty expressions. 3.0 uses negative α with
   no leading `-`. Stay on 3.0's convention since we've verified the
   ranges match the GIGAFLOW spec via the reward bounds PR. Document
   this explicitly so ported code from turbostream doesn't accidentally
   mix conventions.

## Merge hazard map

Quick reference for "if you're merging turbostream code that touches X,
watch out for Y":

| If you're porting... | ...watch out for |
|---|---|
| Anything in `compute_agent_metrics` / `c_step` | 3.0 has `active_step_count` increment (PR #402), `is_invalid_step` masking, `if (agent->stopped) continue` skip — preserve all of these |
| Anything in `make_client` / rendering | 3.0 has `egl_headless_init`, PBO double-buffer, `writev`, `-threads 4` — preserve all |
| Anything in `compute_partner_observations` | 3.0 has 2D rel-v in ego frame (PR #404), `PARTNER_FEATURES = 9` — preserve |
| Anything touching the `Log` struct | 3.0 has `active_step_count` field, `dist_since_infraction`, etc. — merge carefully |
| Anything in `pufferl.py` train loop | 3.0 has `clamp_reward` gating, `heavyball` optimizer integration — preserve |
| Anything in `drive.ini` | 3.0 has GIGAFLOW-spec reward bounds from PR #401 effort — don't revert the ranges |
| `binding.c` kwargs | Additive only — adding turbostream kwargs on top of 3.0's is fine, but don't remove 3.0 kwargs without checking they're dead |

## Commit-by-commit plan (short form)

1. `WIP: rebuild_on_cluster: multi-arch TORCH_CUDA_ARCH_LIST` ✅ (done, `11bc54ca`)
2. `WIP: split compute_agent_metrics into compute_metrics + compute_rewards`
3. `WIP: OBB collision detection (check_obb_collision / check_z_collision_possibility)`
4. `WIP: traffic control subsystem (red lights, stop lines, stop signs)`
5. `WIP: time-to-collision subsystem (ttc_result, compute_pairwise_ttc)`
6. `WIP: waypoint/path/progression system (replaces sample_new_goal)`
7. `WIP: multi-scenario eval pipeline (eval_multi_scenarios + new evaluator)`
8. `WIP: PPO train loop split (_train_ppo_trajectory + _train_ppo_transition)`
9. `WIP: agent speed caching + invalidate_agent` (optional, small cleanup)

Each commit stays WIP until it's been launched + verified on the
cluster. After Phase 6, we'll have a functioning turbostream-ported
branch that can be opened as a real PR to 3.0.
