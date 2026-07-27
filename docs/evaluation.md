# Evaluation

PufferDrive evaluation is catalog-driven. A named benchmark defines which
scenarios to run, while one shared evaluation file defines the environment
settings used for every benchmark. The evaluator writes deterministic per-episode
metrics and can replay only the failed episodes as interactive HTML.

Failure selection, replay capture, and HTML rendering are all handled by
`puffer eval`; there is no separate failure-mining workflow.

## Configuration

The evaluation files live outside `pufferlib/ocean` because they configure the
evaluation application, not the simulator:

- `pufferlib/config/evaluation/benchmark_catalog.yaml` defines named benchmarks.
- `pufferlib/config/evaluation/benchmark_evaluation.yaml` defines shared,
  deterministic environment overrides.
- `pufferlib/config/puffer_drive.yaml` selects those files and configures the
  evaluator under `eval`.

Each catalog benchmark contains:

- `name`: value passed to `eval.benchmarks`.
- `mode`: `gigaflow` for generated scenarios or `replay` for recorded ones.
- `num_scenarios`: number of episode summaries expected in the report.
- `num_maps`: number of sorted map files available to the benchmark.
- `max_agents_per_env`: maximum active agents in one simulator environment.
- `scenario_length`: maximum number of simulator steps per scenario.
- `control_mode`: which agents the policy controls.
- `paths.local`: local map file or directory containing `.bin` files.

The catalog seed and optional benchmark seed must be non-negative integers. Map
files, requested counts, modes, and environment override keys are validated
before environments are created.

`eval.num_agents` is the agent capacity of each evaluation worker and must be at
least the benchmark's `max_agents_per_env`. The policy inference batch grows with
the number of workers, so reduce both `eval.num_agents` and `vec.num_envs` for a
small local CPU check.

Replay benchmarks using `control_sdc_only` additionally cap their worker count with
`eval.max_sdc_replay_workers` (default `4`). Other replay and gigaflow benchmarks
continue to use `vec.num_envs`.

## Running evaluation

Benchmark selection and a 3.0 checkpoint are required. The checkpoint must be in
a run's `models` directory, with the matching `config.yaml` in the run directory.

```bash
source .venv/bin/activate

puffer eval puffer_drive \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.benchmarks=carla_fast \
  eval.render_failures=false \
  train.device=cpu
```

Select multiple benchmarks with a comma-separated value:

```bash
puffer eval puffer_drive \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.benchmarks=carla_fast,womd_single
```

For a smaller CPU run using the committed `carla_fast` benchmark:

```bash
puffer eval puffer_drive \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.benchmarks=carla_fast \
  eval.num_agents=50 \
  vec.num_envs=2 \
  train.device=cpu
```

## Evaluation flow

For each selected benchmark, the evaluator:

1. Loads the checkpoint's policy, RNN, and accepted 3.0 environment settings.
2. Applies the benchmark and shared evaluation overrides.
3. Splits a deterministic scenario window across evaluation workers.
4. Runs deterministic policy inference and gathers one `evaluation_episode`
   summary per scenario.
5. Writes per-episode metrics and aggregate numeric means.

The resolved configuration is written with the report so every run records its
catalog, checkpoint configuration, worker arguments, maps, and seeds.

## Scenario replay and rendering

Capture and render every scenario from the standard benchmark pass with:

```bash
puffer eval puffer_drive \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.benchmarks=carla_fast \
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
evaluation always disables scenario rendering. If `eval.render_failures` is
also true, the all-scenario gallery is produced and the redundant failure
replay pass is skipped. `eval.render_scenarios` cannot be combined with
`eval.failure_replay_csv`, which skips the standard benchmark pass.

## Failure replay and rendering

Enable the integrated failure pass with:

```bash
puffer eval puffer_drive \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.benchmarks=carla_fast \
  eval.render_failures=true \
  eval.max_rendered_failures=10 \
  eval.capture_observations=false
```

An episode is selected when any configured `eval.failure_metrics` value is
greater than zero. The supported metrics are:

- `collision_rate`
- `at_fault_collision_rate`
- `offroad_rate`
- `red_light_violation_rate`

The failure pass replays the selected map/seed pairs, captures standard
interactive `.replay.zlib` files, renders one HTML page per replay, and builds a
navigable `index.html`. `eval.max_rendered_failures` limits each selected benchmark
to its first N failures in metrics-file order; the default `null` renders all
failures. `eval.capture_observations=true` also stores policy observations;
`eval.observation_replay_wave_size` and
`eval.observation_replay_writer_count` bound its peak memory and writer
parallelism.

To replay failures from an existing metrics CSV without rerunning the standard
benchmark pass:

```bash
puffer eval puffer_drive \
  load_model_path=experiments/mimolette/models/model_puffer_drive_003815.pt \
  eval.benchmarks=carla \
  eval.failure_replay_csv=experiments/mimolette/eval/carla/episode_metrics.csv \
  eval.max_rendered_failures=10 \
  eval.capture_observations=false
```

The selected benchmark supplies the replay environment settings, so it should
match the benchmark that produced the CSV.

## Outputs

Direct checkpoint evaluation writes below the checkpoint run directory:

```text
eval/<benchmark>/
├── resolved_benchmark.yaml
├── episode_metrics.csv
├── evaluation_summary.json
├── replays/                        # only when render_scenarios=true
│   └── *.replay.zlib
├── rendered_replays/               # only when render_scenarios=true
│   ├── *.html
│   └── index.html
└── failures/                       # render_failures=true without render_scenarios
    ├── selected_failures.csv
    ├── episode_metrics.csv
    ├── evaluation_summary.json
    ├── replays/
    │   └── *.replay.zlib
    └── rendered_replays/
        ├── *.html
        └── index.html
```

`episode_metrics.csv` contains map and scenario identifiers, the episode seed,
agent batch size, infractions, progress, rewards, and score metrics.
`evaluation_summary.json` contains the requested scenario count, emitted episode
count, and means for every numeric metric.

## Evaluation during training

Training uses the same catalog evaluator and the live policy:

```yaml
train:
  evaluation_interval_epochs: 100
  evaluation_benchmarks: carla_fast
```

The configured benchmarks run every 100 training epochs. If training ends between
scheduled intervals, one final evaluation runs at the last epoch. Training
evaluation logs benchmark metric means to the active logger and writes reports under
the training run's `eval/training` hierarchy.
