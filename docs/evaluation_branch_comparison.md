# Evaluation Branch Comparison

How evaluation differs between:

- `3.0` — `88a4d3c58`
- failure-replay branch — `8ef4f94b4`

Two questions drove this note: do the branches differ in core per-agent state
access, and can the evaluators be unified?

## TL;DR

- **Per-agent state access is effectively equivalent.** `3.0` already exposes
  every agent's simulated state and logged trajectories via `get_state()` and the
  `obs_html` render path, so no new C path is needed to inspect agent x/y. It just
  doesn't export a per-agent *final metrics* CSV (only aggregate `vec_log` +
  optional per-episode summaries).
- **The real differences are in how eval is configured, parallelized,
  aggregated, identified, and exported** — these can change reported scores even
  when rollout behavior matches.
- **`3.0` is the unified architecture; the failure-replay branch uses the
  benchmark-suite path.** Unification means porting the benchmark behavior into
  `3.0`'s `EvalManager`, not the reverse.

## Two architectures

| | failure-replay branch | `3.0` |
|---|---|---|
| Entry | `run_all_eval.sh` → `puffer benchmark` | `puffer eval --evaluator <name>` + inline during training |
| Config | `[eval]` + `benchmark_catalog.yaml` suites | `[eval.<name>]` sections |
| Core code | `pufferlib/evaluate.py` | `benchmark/evaluators/` + `manager.py` |
| Orchestrator | `benchmark()` loop | `EvalManager` |

`pufferlib/evaluate.py` does not exist on `3.0`; that branch uses the evaluator
package instead of the `benchmark()` suite flow.

## What each eval measures

Both use logged scenarios, but they answer different questions.

**Replay / benchmark (current setup)** — *"Did the policy survive this scenario
and avoid infractions?"* An operational/safety regression benchmark. One outcome
per scenario:

- collision / offroad / red-light rates, return, score / DNF, distance,
  infractions
- per-scenario CSV rows
- optional exact failure rerender from `Seed`, `map_index`, `scenario_index`

**WOSAC** — *"Are the policy's sampled futures statistically plausible vs logged
human behavior?"* A generative realism benchmark. It collects ground-truth
trajectories + **multiple** rollouts per scene and scores a distribution:

- ADE / minADE, speed/accel likelihood, angular speed/accel likelihood
- distance-to-nearest-object, time-to-collision, collision/offroad,
  distance-to-road-edge likelihoods
- `realism_meta_score`

**Why keep them separate.** A replay failure with one collision is a concrete
failure. A low WOSAC score means the rollout *distribution* is unrealistic even
if the policy didn't crash in that run. WOSAC also needs many rollouts per scene,
so it is slower and heavier. It should be its own evaluator type/suite, not
another column on the failure-replay row.

On `3.0`, WOSAC is already a first-class evaluator (`[eval.wosac]`, `type =
"wosac"`) but `enabled = false` by default; its adapter is intended to reuse the
realism math in `benchmark/evaluator.py`. Caveat: the adapter calls an inner
`evaluate(...)` that the inspected commit doesn't define (it exposes
`collect_ground_truth_trajectories` / `collect_simulated_trajectories` /
`compute_metrics`), so a small shim may be needed.

## Differences that move the score

**Aggregation.** `3.0` takes a weighted mean of `vec_log` emissions by `n`
(`sum(metric*n)/sum(n)`, `n` = agent trajectories). The failure-replay branch
collects per-env rows into `episode_metrics.csv` and averages columns,
special-casing totals (distance, infraction counts). Identical rollouts can
report different numbers when batch / env / agent counts differ.

**CARLA config.** The inspected command resolves to the catalog `carla` suite
(`gigaflow`, 80 scenarios, 8 maps, `scenario_length=3000`,
`control_mode=control_vehicles`, `traffic_light_behavior=1`, `goal_speed=20`).
The closest `3.0` evaluator, `validation_gigaflow`, defaults to 250 scenarios,
`scenario_length=500`, `num_agents=1024`, `traffic_light_behavior=0`. Raw scores
are not comparable without normalizing these. One more `3.0` caveat: the
`multi_scenario` evaluator auto-sets `env.num_eval_scenarios` from
`eval.num_scenarios` for replay, but not for gigaflow. A faithful CARLA
comparison should set or derive that explicitly.

**Identity metadata.** The failure-replay branch stamps rows with `Seed`,
`map_index`, `scenario_index`, and replay-batch metadata for exact failure
replay. `3.0` has `map_name` / `scenario_id` in completed-episode summaries but
not the full identity scheme — port it if exact row matching is required.

## Multi-worker eval and scenario sharding

How each branch parallelizes a sweep over a fixed scenario set differs, and `3.0`
cannot reproduce the failure-replay per-worker partition via config.

**Failure-replay branch — explicit per-worker partition.** `evaluation_metrics`
splits `num_scenarios` across `n_work` workers, giving each a *distinct*
`starting_map` + `num_eval_scenarios` chunk (heterogeneous `env_kwargs`), so
coverage is non-overlapping:

```python
for j in range(n_work):
    kw = {..., "starting_map": curr_st, "num_eval_scenarios": c}  # unique per worker
    curr_st += c
```

**`3.0` default — single process, C-side sweep.** `vec_overrides()` =
`{backend: "PufferEnv", num_envs: 1}`. One `Drive` object distributes maps across
internal C envs via `binding.shared(...)` and advances
`starting_map_counter += num_envs` across resets. This avoids IPC and gives a
sequential sweep from a single `starting_map`, but exact coverage depends on
`env.num_eval_scenarios` being set consistently with the evaluator target.

**`3.0` multiprocessing — not sharded.** Opt-in via `[eval.<name>.vec] backend =
"Multiprocessing"`, but `EvalManager._run_inline` passes the *same* dict to every
worker (`env_kwargs=[args["env"]]*num_envs`). All workers start at the same map →
redundant duplicate sweeps, not a partition.

To get the failure-replay sharding, `_run_inline` (or a new hook) must build a
heterogeneous `env_kwargs` list computing per-worker
`starting_map`/`num_eval_scenarios`. A single-dict `vec_overrides()` can't express
it. (Separately, gigaflow env-count math also differs: `3.0` keeps a partial
final env, the failure-replay branch allocates only full `max_agents_per_env`
chunks — minor, but it changes how many internal envs are active for some
configs.)

## Eval path decision

`3.0` is the going-forward branch, so the eval code lives there. The remaining
choice is whether to keep `3.0`'s eval *path* (the `EvalManager` framework) or
bring the failure-replay branch's `benchmark()` path onto `3.0`.

**Keep `3.0`'s path; integrate the failure-replay branch's features into it.**

Because `3.0` owns the training loop, eval must run **inline during training**,
and only `EvalManager` is wired for that — a standalone `benchmark()` cannot be
driven from inside the loop. Bringing the `benchmark()` path onto `3.0` would
leave two eval paths on the mainline (manager + standalone), the "second eval
path" `3.0`'s design deliberately removed. So the path stays `3.0`'s and the
failure-replay value is carried in as features.

- **Path (architecture) — `3.0`'s.** Evaluator registry + per-evaluator classes,
  `[eval.<name>]` config discovery/inheritance, inline + subprocess execution,
  per-evaluator logging/render, `egl` / `triage_html` / `obs_html` backends,
  completed-episode CSV + coverage checks.
- **Features (from failure-replay, additive on that path).** Exact failure replay
  (`Seed` / `map_index` / `scenario_index` rerun), per-worker `starting_map` /
  chunk sharding, identity metadata on the info / `completed_episode` stream,
  catalog-suite translation, the WOSAC orchestration fix, and the master
  multi-suite summary CSV.

The reverse — failure-replay branch as the path — would re-introduce
`pufferlib/evaluate.py` and `benchmark()` on `3.0` and still require rebuilding
the manager to get inline eval, ending with a duplicate path. Not worth it once
`3.0` is the mainline. The branches also share no git history, so this is a port,
not a merge: the failure-replay logic is re-expressed in `3.0`'s evaluator idiom.
Each feature is small and self-contained, so the port is bounded.

## Unification direction

Concrete steps to carry the failure-replay features onto `3.0`'s path:

1. Add a catalog-backed evaluator/adapter that maps `benchmark_catalog.yaml`
   suites onto `[eval.<name>]` config.
2. Add a `failure_replay` evaluator type for the CSV-driven exact rerun
   (`Seed` / `map_index` / `scenario_index`); reuse
   `3.0`'s `dnf_triage` + `triage_html` for the render half.
3. Port the identity metadata onto the info / `completed_episode` stream so exact
   row matching works.
4. Port per-worker `starting_map` / chunk sharding into `_run_inline`
   (heterogeneous `env_kwargs`) for multi-process sweeps.
5. Fix the WOSAC adapter's `inner.evaluate(...)` gap, reusing the existing
   collect/compute methods in `benchmark/evaluator.py`. Keep WOSAC its own
   evaluator type, never a column on the safety-metric row.
6. Add the master multi-suite summary CSV as a manager-level rollup.
7. Make aggregation explicit: report scenario-row means and agent-weighted means
   separately, or pick one as the scoreboard. Add a new C binding only if exact
   per-agent final metric rows from `env->logs[i]` are required — agent x/y is
   already available via `get_state()` / `obs_html`.

## Takeaway

Same practical per-agent state access on both branches. The meaningful work is
evaluator/reporting design — config, parallelism, aggregation, identity, and
keeping safety-regression (replay) separate from realism (WOSAC). Decision: `3.0`
is the going-forward branch; keep its `EvalManager` eval path and integrate the
failure-replay branch's features into it — one eval path on the mainline, with the
failure-replay value carried in additively.
