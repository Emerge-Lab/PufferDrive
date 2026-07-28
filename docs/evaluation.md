# Evaluation

PufferDrive evaluation is config-driven. A named benchmark defines which
scenarios to run, while the shared `env` section defines the environment
settings used for every benchmark. The evaluator writes deterministic per-episode
metrics and can replay only the failed episodes as interactive HTML.

Failure selection, replay capture, and HTML rendering are all handled by
`puffer eval`; there is no separate failure-mining workflow.

## Configuration

The evaluation files live outside `pufferlib/ocean` because they configure the
evaluation application, not the simulator:

- `pufferlib/config/evaluation/benchmark.yaml` defines the shared deterministic
  environment and named benchmarks.
- `pufferlib/config/puffer_drive.yaml` selects that file and configures the
  evaluator under `eval`.

Each configured benchmark contains:

- `name`: positional benchmark name passed after `puffer_drive`.
- `simulation_mode`: `gigaflow` for generated scenarios or `replay` for recorded ones.
- `num_scenarios`: number of episode summaries expected in the report.
- `num_maps`: number of sorted map files available to the benchmark.
- `max_agents_per_env`: maximum active agents in one simulator environment.
- `scenario_length`: maximum number of simulator steps per scenario.
- `control_mode`: which agents the policy controls.
- `map_dir`: local map file or directory containing `.bin` files.

Each benchmark seed must be a non-negative integer. Map files, requested counts,
modes, and environment override keys are validated before environments are
created.

`eval.num_agents` is the agent capacity of each evaluation worker and must be at
least the benchmark's `max_agents_per_env`. The policy inference batch grows with
the number of workers, so reduce both `eval.num_agents` and `vec.num_envs` for a
small local CPU check. `env.num_agents` configures training and is rejected by
the direct evaluation command; use `eval.num_agents` for evaluation.

Replay benchmarks using `control_sdc_only` additionally cap their worker count with
`eval.max_sdc_replay_workers` (default `4`). Other replay and gigaflow benchmarks
continue to use `vec.num_envs`.

## Running evaluation

Benchmark selection and a 3.0 checkpoint are required. The checkpoint must be in
a run's `models` directory, with the matching `config.yaml` in the run directory.

```bash
puffer eval puffer_drive carla_fast \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  train.device=cpu
```

Select multiple benchmarks with a comma-separated value:

```bash
puffer eval puffer_drive carla_fast,womd_single \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt
```

For a smaller CPU run using the committed `carla_fast` benchmark:

```bash
puffer eval puffer_drive carla_fast \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.num_agents=50 \
  vec.num_envs=2 \
  train.device=cpu
```

Hydra overrides are applied after the selected benchmark. This supports
parameter experiments without editing `benchmark.yaml`:

```bash
puffer eval puffer_drive carla_fast \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  env.goal_speed=10 \
  eval.output_name=goal_speed_10
```

Use `eval.output_name` when comparing runs so the folder identifies the
experiment. The resolved configuration saved with every result records the
effective values.

## Evaluation flow

For each selected benchmark, the evaluator:

1. Loads the checkpoint's policy, RNN, and accepted 3.0 environment settings.
2. Applies the benchmark and shared evaluation overrides.
3. Splits a deterministic scenario window across evaluation workers.
4. Runs deterministic policy inference and gathers one `evaluation_episode`
   summary per scenario.
5. Writes per-episode metrics and aggregate numeric means.

The resolved configuration is written with the report so every run records its
benchmark config, checkpoint configuration, worker arguments, maps, and seeds.

## Scenario replay and rendering

Capture and render every scenario from the standard benchmark pass with:

```bash
puffer eval puffer_drive carla_fast \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.render_scenarios=true \
  eval.capture_observations=false
```

`eval.render_scenarios=true` records each of the benchmark's configured
`num_scenarios` during the metrics rollout. It writes the completed
`.replay.zlib` files incrementally, then renders one interactive HTML page per
scenario and builds a navigable `index.html`. The benchmark is not rerun, and
the benchmark seed and worker configuration continue to determine the evaluated
map/seed rows. Capturing every scenario increases CPU, memory, and disk usage.

`eval.capture_observations=true` also stores policy observations. Periodic training
evaluation always disables scenario rendering. If `eval.render_filter` is
also set, the all-scenario gallery is produced and the redundant filtered
replay pass is skipped. `eval.render_scenarios` cannot be combined with
`eval.failure_replay_csv`, which skips the standard benchmark pass.

## Filtered replay and rendering

Set `eval.render_filter` to render scenarios where a selected metric is greater
than zero:

```bash
puffer eval puffer_drive carla_fast \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.render_filter=offroad_rate \
  eval.max_rendered_failures=10 \
  eval.capture_observations=false
```

The default `eval.render_filter: null` disables filtered rendering. Any numeric
column in `episode_metrics.csv` can be used. Multiple comma-separated columns
use OR: `collision_rate,offroad_rate` selects scenarios where either metric is
greater than zero. Unknown columns raise an error when the metrics CSV is read.

The filtered pass replays the selected map/seed pairs, captures standard
interactive `.replay.zlib` files, renders one HTML page per replay, and builds a
navigable `index.html`. `eval.max_rendered_failures` limits each selected benchmark
to its first N matching scenarios in metrics-file order; the default `null`
renders every match. `eval.capture_observations=true` also stores policy observations;
`eval.observation_replay_wave_size` and
`eval.observation_replay_writer_count` bound its peak memory and writer
parallelism.

To filter and replay an existing metrics CSV without rerunning the standard
benchmark pass:

```bash
puffer eval puffer_drive carla \
  load_model_path=experiments/mimolette/models/model_puffer_drive_003815.pt \
  eval.failure_replay_csv=experiments/mimolette/eval/carla/episode_metrics.csv \
  eval.render_filter=offroad_rate \
  eval.max_rendered_failures=10 \
  eval.capture_observations=false
```

The selected benchmark supplies the replay environment settings, so it should
match the benchmark that produced the CSV. `eval.failure_replay_csv` requires a
non-null `eval.render_filter`.

## Outputs

Direct checkpoint evaluation writes below the checkpoint run directory:

```text
eval/<benchmark>[_<output_name>]/<timestamp>/
├── resolved_benchmark.yaml
├── episode_metrics.csv
├── evaluation_summary.json
├── replays/                        # only when render_scenarios=true
│   └── *.replay.zlib
├── rendered_replays/               # only when render_scenarios=true
│   ├── *.html
│   └── index.html
└── failures/                       # render_filter set without render_scenarios
    ├── selected_failures.csv
    ├── episode_metrics.csv
    ├── evaluation_summary.json
    ├── replays/
    │   └── *.replay.zlib
    └── rendered_replays/
        ├── *.html
        └── index.html
```

Without `eval.output_name`, the first run uses `<benchmark>`. With a name, it
uses `<benchmark>_<output_name>`. If that directory exists, evaluation preserves
it and appends `_0`, `_1`, and so on to the new directory.

`episode_metrics.csv` contains map and scenario identifiers, the episode seed,
agent batch size, infractions, progress, rewards, and score metrics.
`evaluation_summary.json` contains the requested scenario count, emitted episode
count, and means for every numeric metric.

## Evaluation during training

Training uses the same configured evaluator and the live policy:

```yaml
env:
  num_agents: 1024
eval:
  num_agents: 128
train:
  evaluation_interval_epochs: 100
  evaluation_benchmarks: carla_fast
```

The default `evaluation_interval_epochs: null` disables evaluation during
training. With the configuration above, the selected benchmarks run every 100
epochs in a separate evaluation environment with 128 agents; the training
environment remains at 1024. Mid-training evaluation currently shares
`vec.num_envs` with training. If training ends between scheduled intervals, one
final evaluation runs at the last epoch. Training evaluation logs benchmark
metric means to the active logger and writes reports under the training run's
`eval/training` hierarchy.
