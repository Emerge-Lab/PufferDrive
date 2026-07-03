# Evaluation Pipeline Trace: `puffer benchmark` vs 3.0 `puffer eval`

Function-level trace of the two CARLA evaluation paths:

- **This branch** (`vb/failures_met2`) — `puffer benchmark` + `benchmark_catalog.yaml`
- **3.0** (`emergelab/3.0`, inspected at `38b2b02f1`) — `puffer eval --evaluator` + `EvalManager`

Companion to `evaluation_branch_comparison.md`, which covers the design
rationale; this doc covers the concrete call chain and config layering.

## Reference commands

```bash
# This branch
puffer benchmark puffer_drive \
    --load-model-path experiments/<run>/models/model_puffer_drive_001491.pt \
    --eval.benchmark-datasets carla \
    --eval.render 0 --eval.render-obs 0 --eval.render-only 0

# 3.0 (rebuild C after checkout: python setup.py build_ext --inplace --force)
puffer eval puffer_drive \
    --evaluator validation_gigaflow \
    --load-model-path <path/to/checkpoint.pt>
```

On this branch the three render flags are already `False` by default, so the
command above is a pure metrics run: no gif/html output, only CSVs.

---

## 1. This branch: `puffer benchmark` trace

### 1.1 Entry and configuration

| Step | Function | File |
|---|---|---|
| CLI dispatch | `main()` → `benchmark(env_name)` | `pufferl.py:2574`, `pufferl.py:1939` |
| Config load | `load_config()` | `pufferl.py:2473` |

`load_config` merges `config/default.ini` + `config/ocean/drive.ini` + CLI
dotted overrides (`--eval.benchmark-datasets carla` →
`args["eval"]["benchmark_datasets"]`).

### 1.2 Context and suite resolution (`pufferlib/evaluate.py`)

- **`_resolve_benchmark_context()`** (`evaluate.py:128`) — path ends in `.pt`
  → used directly. Output root:
  `experiments/<run>/final_evaluation/model_puffer_drive_001491/`.
  A model path exists → `replay_expert_actions = False` (policy drives, not
  logged experts).
- **`_build_final_master_eval_suites()`** (`evaluate.py:159`) — parses
  `pufferlib/ocean/competition/benchmark_catalog.yaml`, keeps only the
  suites named in `eval.benchmark_datasets`. The `carla` suite: gigaflow
  mode, 160 scenarios, 8 maps, `max_agents_per_env=50`,
  `scenario_length=6000`, `control_mode=control_vehicles`, maps from
  `pufferlib/resources/drive/binaries/carla` (local; GCS path when running
  in cloud, detected by `_is_cloud()`).

### 1.3 Run-args construction (`pufferl.py:1807`)

`_build_eval_run_args()` layers the env config. Later layers win:

1. **Base ini** — `default.ini` + `drive.ini` (`suite["num_agents"]` is set
   from `eval.num_agents = 101`).
2. **Checkpoint `config.yaml`** — `load_eval_multi_scenarios_config()`
   (`pufferl.py:1774`) merges the *full* `env`, `policy`, `rnn` sections plus
   `policy_name`/`rnn_name` from the yaml sitting next to `models/`, **except**
   env keys the eval overrides own. Network arch therefore matches the
   trained weights.
3. **Canonical eval overrides** — `build_eval_overrides()` (`evaluate.py:66`)
   applied last: `eval_mode=1`, collision/offroad behavior 1,
   `traffic_light_behavior=1`, `goal_speed=20`, fixed eval reward
   coefficients, zero obs dropout, gigaflow map settings (`num_maps` clamped
   to the `.bin` count actually present in `map_dir`).

Per-suite results dir: `final_evaluation/<model>/carla/`.

### 1.4 Sharded metrics rollout (`pufferl.py:2035-2058`)

- `n_work = _metrics_env_count()` (`pufferl.py:1825`) =
  `min(vec.num_envs, num_scenarios)` = `min(16, 160)` = **16**.
- The 160 scenarios are **partitioned**: each worker gets a distinct
  `starting_map` + `num_eval_scenarios` chunk in its own `env_kwargs`
  (non-overlapping coverage).
- `pufferlib.vector.make()` spawns 16 envs (Multiprocessing backend from
  `default.ini`). Each worker builds `Drive`
  (`pufferlib/ocean/drive/drive.py`) → `binding.c` → the C sim in
  `drive.h`/`drive.c`, loading the carla `.bin` maps.
- `load_policy()` (`pufferl.py:2431`) builds the net from the merged arch and
  loads the `.pt` weights.
- SDC-replay suites additionally cap workers to
  `eval.benchmark_sdc_num_envs` (not applicable to carla/gigaflow).

### 1.5 Eval loop — `evaluation_metrics()` (`evaluate.py:284`)

- Seeds numpy/torch from `train.seed`; `vecenv.async_reset(seed)`.
- Loops until 160 scenario summaries collected. Inner loop runs exactly
  `scenario_length` steps; RNN state is reset per scenario
  (`_reset_rnn_state`).
- Per step: `policy.forward_eval()` under `no_grad`, **deterministic**
  `sample_logits`, continuous actions clipped to the action-space bounds.
- At scenario end the C env emits a per-episode summary — `map_name`, `Seed`,
  `map_index`, `scenario_index`, collision/offroad/red-light rates, distance,
  infraction counts, score. `Seed`/`map_index`/`scenario_index` are the
  identity metadata that later enables exact failure replay
  (`--eval.render-failures-only`).

### 1.6 Export — `_export_metrics()` (`evaluate.py:247`)

Written to `final_evaluation/<model>/carla/`:

- `episode_metrics.csv` — one row per scenario, identity columns first,
  with a coverage check ("Exported N/160 episodes").
- `evaluation_summary.csv` — column averages via
  `_reduce_environment_metrics` (`num_scenarios` summed, the rest averaged).

Back in `benchmark()`: the render pass is skipped (`render=0`), and
`_merge_master_benchmark_summary()` (`evaluate.py:199`) upserts one row per
`suite_id` into `final_evaluation/<model>/master_evaluation_summary.csv`.

### 1.7 File map (call order)

```
pufferl.py        main → benchmark → _build_eval_run_args
                  → load_eval_multi_scenarios_config / load_policy / _metrics_env_count
evaluate.py       _resolve_benchmark_context, _build_final_master_eval_suites,
                  build_eval_overrides, evaluation_metrics, _export_metrics,
                  _merge_master_benchmark_summary
config sources    benchmark_catalog.yaml, default.ini, drive.ini,
                  checkpoint config.yaml
simulation        drive.py → binding.c → drive.h / drive.c
```

---

## 2. 3.0: `puffer eval --evaluator validation_gigaflow` trace

### 2.1 Entry and checkpoint arch adoption

- `main()` → mode `eval` → `eval()` (`pufferl.py:1649` at `38b2b02f1`).
- **`_merge_checkpoint_arch()`** (`pufferl.py:1614`) reads the checkpoint's
  sibling `config.yaml` but adopts **only** `policy.*`, `rnn.*`,
  `policy_name`/`rnn_name` (→ `use_rnn`), and the obs/action-layout env keys
  in `_ARCH_ENV_KEYS`. The eval *environment* (sim mode, maps, rewards) is
  deliberately left to the `[eval.<name>]` section — narrower than this
  branch's full-`env` merge.
- Evaluator name: `--evaluator <name>`, or `--eval_simulation gigaflow` →
  `validation_gigaflow`. A default render output dir
  `benchmark/puffer_<run_id>` is derived from the model path.

### 2.2 Manager construction — `EvalManager.from_config()` (`manager.py:57`)

- `_discover_eval_sections()` — every `[eval.<name>]` dict in the parsed
  config.
- `_build_section_config()` (`manager.py:299`) — resolves the `inherits`
  chain by deep merge (dotted keys expanded, cycle-checked), then the
  `clean = true` macro `setdefault`s `CLEAN_EVAL_OVERRIDES` into `env`.
- For `validation_gigaflow` the chain is:

  ```
  [eval.validation_defaults]      (template)
      num_agents=1024, traffic_light_behavior=0, goal_speed=3.0,
      eval reward set, interval=250, mode=inline, verify_coverage
  └── [eval.validation_gigaflow]
      type=multi_scenario, simulation_mode=gigaflow,
      map_dir=.../binaries/carla, num_maps=8,
      min/max_agents_per_env=40, scenario_length=500,
      resample_frequency=500, render_backend=egl
  ```

- Sections with a `type` are instantiated from `EVALUATOR_REGISTRY` →
  `MultiScenarioEvaluator`.

### 2.3 Ad-hoc CLI overrides and policy

- `--num_scenarios`, `--num_maps`, `--render`, `--render-backend` mutate the
  target evaluator's `config`/`render` in place, for this run only.
- A throwaway *probe* vecenv (`manager._build_eval_args()` + `load_env()`)
  provides the obs/action spaces so `load_policy()` can build the net and
  load the `.pt`.

### 2.4 Run — `run_one_by_name()` → `_run_one()` → `_run_inline()` (`manager.py:155`)

- **`_build_eval_args()`** (`manager.py:227`) — deep-copies the full train
  config, then:
  - `args["env"].update(ev.env_overrides())`
  - `args["vec"].update(ev.vec_overrides())`
  - seed from `train.seed`; forwards `eval.*` knobs; enables
    `env.emit_completed_episodes` when `export_episode_csv` /
    `verify_coverage` is set.
- **`MultiScenarioEvaluator.env_overrides()`** — baseline `eval_mode=1`,
  `termination_mode=0`, `reward_randomization=False`, then the section's
  `env.*`. **Replay-only** auto-derive:
  `num_eval_scenarios = eval.num_scenarios`. Gigaflow leaves the C default
  alone — set `env.num_eval_scenarios` explicitly for a faithful fixed-size
  sweep.
- **`vec_overrides()`** default: `{backend: "PufferEnv", num_envs: 1}` — a
  single `Drive` object whose C kernel batches all internal envs. No
  multiprocessing, no scenario sharding. (Opting into Multiprocessing gives
  every worker the *same* kwargs — duplicate sweeps, not a partition.)

### 2.5 Rollout — `Evaluator.rollout()` (`base.py:92`) → `_run_rollout_loop()` (`base.py:126`)

- `policy.eval()` for the duration; metric and render passes timed
  separately (`metric_seconds` / `render_seconds` / `eval_seconds`).
- Open step loop (no fixed scenario-length inner loop): deterministic
  `sample_logits`, continuous actions clipped.
- **LSTM state masked per-agent** on `terminals | truncations` (state carries
  across steps otherwise, unlike this branch's per-scenario reset).
- Infos split: `summary_type == "completed_episode"` rows vs `my_log`
  emissions.
- Stop condition (`multi_scenario.py:_should_stop`): collected episode rows
  (when CSV/coverage is on) or log emissions ≥ `eval.num_scenarios`.
- Stall backstop: abort after 3×`scenario_length` steps with no new
  episodes/emissions.

### 2.6 Aggregation and output

- **`_aggregate_infos()`** (`base.py:231`) — agent-weighted mean over
  `vec_log` emissions: `sum(metric·n) / sum(n)`, `n` = agent trajectories
  behind each emission. Also reports `num_log_cycles` and
  `num_agents_evaluated`. Caveat: ratio fields (e.g.
  `avg_distance_per_infraction`) are only approximated by a weighted mean of
  per-emission ratios.
- **`_maybe_export_episodes()`** — opt-in per-episode CSV + coverage report.
- **`_render_pass()`** if `render` (egl / `triage_html` / `obs_html`
  backends).
- Back in `eval()`: metrics printed as JSON between
  `EVAL_RESULT_JSON_START/END` markers; `--out <json>` writes a result file —
  this is how `mode = "subprocess"` evaluators report back to a training
  run's `EvalManager.maybe_run()`, which fires every `interval` epochs
  inline during training.

### 2.7 File map (call order)

```
pufferl.py                     main → eval → _merge_checkpoint_arch
benchmark/manager.py           EvalManager.from_config → _discover_eval_sections
                               → _build_section_config → run_one_by_name
                               → _run_one → _run_inline → _build_eval_args
benchmark/evaluators/base.py   rollout → _run_rollout_loop → _aggregate_infos
                               → _maybe_export_episodes → _render_pass
benchmark/evaluators/multi_scenario.py   env_overrides, _should_stop
config sources                 drive.ini [eval.*] sections,
                               checkpoint config.yaml (arch keys only)
simulation                     drive.py → binding.c → drive.h / drive.c
```

---

## 3. Side-by-side comparison

| | This branch — `puffer benchmark` | 3.0 — `puffer eval --evaluator validation_gigaflow` |
|---|---|---|
| **Entry** | `benchmark()` `pufferl.py:1939` | `eval()` `pufferl.py:1649` → `EvalManager` |
| **Suite/eval definition** | `benchmark_catalog.yaml` `carla` suite (`_build_final_master_eval_suites`) | `[eval.validation_gigaflow]` ini section + `inherits` chain + `clean` macro (`_build_section_config`) |
| **CARLA settings** | 160 scenarios, 8 maps, 50 agents/env, `scenario_length=6000`, `traffic_light_behavior=1`, `goal_speed=20`, `num_agents=101` | `eval.num_scenarios` (section default), 8 maps, 40 agents/env, `scenario_length=500`, `traffic_light_behavior=0`, `goal_speed=3.0`, `num_agents=1024` |
| **Env-config layering (last wins)** | ini → checkpoint `config.yaml` (full `env`+`policy`+`rnn`, minus override-owned keys) → `build_eval_overrides` + catalog | ini → checkpoint `config.yaml` (**arch keys only**) → evaluator baseline + section `env.*` → ad-hoc CLI flags |
| **Vectorization** | Multiprocessing, `n_work = min(vec.num_envs=16, num_scenarios)` workers | `PufferEnv`, `num_envs=1` — single process, C batches internal envs |
| **Scenario sharding** | Explicit partition: per-worker distinct `starting_map` + `num_eval_scenarios` chunk | None — sequential C-side map sweep from one `starting_map`; gigaflow does not auto-set `num_eval_scenarios` |
| **Rollout loop** | `evaluation_metrics()` `evaluate.py:284` — fixed `scenario_length` inner loop, RNN reset per scenario | `_run_rollout_loop()` `base.py:126` — open loop with `_should_stop` target + stall backstop, per-agent LSTM masking on done |
| **Action sampling** | deterministic `sample_logits` | deterministic `sample_logits` (same) |
| **Aggregation** | Per-scenario CSV rows, column means (`_reduce_environment_metrics`; totals summed) | Agent-weighted mean of `vec_log` emissions: `sum(m·n)/sum(n)` (`_aggregate_infos`) |
| **Identity metadata** | `Seed`/`map_index`/`scenario_index` on every row → exact failure replay | `map_name`/`scenario_id` in episode summaries only; no exact-replay scheme |
| **Outputs** | `episode_metrics.csv`, `evaluation_summary.csv`, `master_evaluation_summary.csv` under `final_evaluation/<model>/` | JSON to stdout/`--out`; opt-in episode CSV + coverage report; wandb/tb logging via `_log` |
| **Inline during training** | No (standalone; `eval.benchmark=True` hook at train end) | Yes — `maybe_run()` every `interval` epochs, inline or subprocess |

### Score-moving differences

Two differences dominate any raw-score comparison (see
`evaluation_branch_comparison.md`):

1. **Aggregation semantics** — scenario-row mean vs agent-weighted mean.
   Identical rollouts report different numbers when batch/env/agent counts
   differ.
2. **Suite config mismatch** — scenario length 6000 vs 500, traffic-light
   behavior 1 vs 0, goal speed 20 vs 3, 101 vs 1024 agents.

For a faithful CARLA comparison on 3.0: override the mismatched knobs on the
evaluator (`--num_scenarios` / `--num_maps` exist as CLI flags; the rest need
`env.*` edits in the `[eval.validation_gigaflow]` section) and set
`env.num_eval_scenarios` explicitly, since the gigaflow path won't derive it
from `eval.num_scenarios`.
