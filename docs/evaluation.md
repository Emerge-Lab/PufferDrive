# Evaluation

PufferDrive evaluation is catalog-driven. A named benchmark suite defines which
scenarios to run, while one shared evaluation file defines the environment
settings used for every suite. The evaluator writes deterministic per-episode
metrics and can replay only the failed episodes as interactive HTML.

Failure selection, replay capture, and HTML rendering are all handled by
`puffer eval`; there is no separate failure-mining workflow.

## Configuration

The evaluation files live outside `pufferlib/ocean` because they configure the
evaluation application, not the simulator:

- `pufferlib/config/evaluation/benchmark_catalog.yaml` defines named suites.
- `pufferlib/config/evaluation/benchmark_evaluation.yaml` defines shared,
  deterministic environment overrides.
- `pufferlib/config/puffer_drive.yaml` selects those files and configures the
  evaluator under `eval`.

Each catalog suite contains:

- `name`: value passed to `eval.datasets`.
- `mode`: `gigaflow` for generated scenarios or `replay` for recorded ones.
- `num_scenarios`: number of episode summaries expected in the report.
- `num_maps`: number of sorted map files available to the suite.
- `max_agents_per_env`: maximum active agents in one simulator environment.
- `scenario_length`: maximum number of simulator steps per scenario.
- `control_mode`: which agents the policy controls.
- `paths.local`: local map file or directory containing `.bin` files.

The catalog seed and optional suite seed must be non-negative integers. Map
files, requested counts, modes, and environment override keys are validated
before environments are created.

`eval.num_agents` is the agent capacity of each evaluation worker and must be at
least the suite's `max_agents_per_env`. The policy inference batch grows with
the number of workers, so reduce both `eval.num_agents` and `vec.num_envs` for a
small local CPU check.

Replay suites using `control_sdc_only` additionally cap their worker count with
`eval.benchmark_sdc_num_envs` (default `8`). Other replay and gigaflow suites
continue to use `vec.num_envs`.

## Running evaluation

Dataset selection and a 3.0 checkpoint are required. The checkpoint must be in
a run's `models` directory, with the matching `config.yaml` in the run directory.

```bash
source .venv/bin/activate

puffer eval puffer_drive \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.datasets=carla_fast \
  eval.render_failures=false \
  train.device=cpu
```

Select multiple suites with a comma-separated value:

```bash
puffer eval puffer_drive \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.datasets=carla_fast,womd_single
```

For a smaller CPU run using the committed `carla_fast` suite:

```bash
puffer eval puffer_drive \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.datasets=carla_fast \
  eval.num_agents=50 \
  vec.num_envs=2 \
  train.device=cpu
```

## Evaluation flow

For each selected suite, the evaluator:

1. Loads the checkpoint's policy, RNN, and accepted 3.0 environment settings.
2. Applies the suite and shared evaluation overrides.
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
  eval.datasets=carla_fast \
  eval.render_scenarios=true \
  eval.render_obs=false
```

`eval.render_scenarios=true` records each of the suite's configured
`num_scenarios` during the metrics rollout. It writes the completed
`.replay.zlib` files incrementally, then renders one interactive HTML page per
scenario and builds a navigable `index.html`. The benchmark is not rerun, and
the suite seed and worker configuration continue to determine the evaluated
map/seed rows. Capturing every scenario increases CPU, memory, and disk usage.

`eval.render_obs=true` also stores policy observations. Periodic training
evaluation always disables scenario rendering. If `eval.render_failures` is
also true, the all-scenario gallery is produced and the redundant failure
replay pass is skipped. `eval.render_scenarios` cannot be combined with
`eval.replay_failures_csv`, which skips the standard benchmark pass.

## Failure replay and rendering

Enable the integrated failure pass with:

```bash
puffer eval puffer_drive \
  load_model_path=weights/mimolette/models/model_puffer_drive_003815.pt \
  eval.datasets=carla_fast \
  eval.render_failures=true \
  eval.render_failures_number=10 \
  eval.render_obs=false
```

An episode is selected when any configured `eval.failure_metrics` value is
greater than zero. The supported metrics are:

- `collision_rate`
- `at_fault_collision_rate`
- `offroad_rate`
- `red_light_violation_rate`

The failure pass replays the selected map/seed pairs, captures standard
interactive `.replay.zlib` files, renders one HTML page per replay, and builds a
navigable `index.html`. `eval.render_failures_number` limits each selected suite
to its first N failures in metrics-file order; the default `null` renders all
failures. `eval.render_obs=true` also stores policy observations;
`eval.observation_replay_wave_size` and
`eval.observation_replay_writer_count` bound its peak memory and writer
parallelism.

To replay failures from an existing metrics CSV without rerunning the standard
benchmark pass:

```bash
puffer eval puffer_drive \
  load_model_path=experiments/mimolette/models/model_puffer_drive_003815.pt \
  eval.datasets=carla \
  eval.replay_failures_csv=experiments/mimolette/eval/carla/episode_metrics.csv \
  eval.render_failures_number=10 \
  eval.render_obs=false
```

The selected dataset supplies the replay environment settings, so it should
match the suite that produced the CSV.

## Outputs

Direct checkpoint evaluation writes below the checkpoint run directory:

```text
eval/<suite>/
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
eval:
  training_enabled: true
  training_interval: 100
  training_datasets: carla_fast
```

The configured suites run every 100 training epochs. If training ends between
scheduled intervals, one final evaluation runs at the last epoch. Training
evaluation logs suite metric means to the active logger and writes reports under
the training run's `eval/training` hierarchy.
