# Evaluation — operational guide

How evaluation works in PufferDrive and how to run it. All evaluation goes
through one system: the `Evaluator` classes in
`pufferlib/ocean/benchmark/evaluators/` orchestrated by the `EvalManager` in
`pufferlib/ocean/benchmark/manager.py`. The same evaluators run inline during
training and standalone from the CLI — there is no second eval path.

## Concepts

- **Evaluator** — one evaluation, defined by an `[eval.<name>]` section in
  `pufferlib/config/ocean/drive.ini`. It owns an env config, a rollout, a set
  of metrics, and optional rendering.
- **EvalManager** — discovers every `[eval.<name>]` section, instantiates the
  evaluators, and runs the ones whose `interval` is due (inline) or that you
  name (standalone). One evaluator failing doesn't stop the rest.
- **Evaluator types** (the `type` field): `multi_scenario` (sweep a scenario
  set in one batched rollout), `behavior_class` (a labelled nuPlan scene
  bucket), `human_replay`, `wosac`.

## Config schema

```ini
[eval.<name>]
type          = "multi_scenario"   ; registered evaluator class (omit for a template)
enabled       = true               ; skip when false
interval      = 250                ; run every N epochs inline (0 disables inline)
mode          = "inline"           ; "inline" (block training) | "subprocess"
inherits      = "<other_section>"  ; optional: pull defaults from another section
clean         = true               ; zero perturbations/dropout + enforce red lights
render        = true               ; capture renders during the rollout
render_views  = ["sim_state","bev"]; camera views for the egl backend
render_backend = "egl"             ; egl | triage_html | obs_html (see Render backends)
env.<key>     = <value>            ; any [env] override (dotted)
eval.<key>    = <value>            ; evaluator-specific knob (see below)
vec.<key>     = <value>            ; any [vec] override
```

A section **without** a `type` is a *template*: it is never instantiated, only
pulled in via `inherits`. `validation_defaults` is a template.

`eval.*` knobs read by `multi_scenario`:

| Key | Meaning |
|---|---|
| `eval.num_scenarios` | how many episodes to evaluate (loop target) |
| `eval.export_episode_csv` | write one CSV row per finished episode |
| `eval.verify_coverage` | report expected-vs-evaluated counts + duplicate maps |
| `eval.render_num_scenarios` | how many scenarios to render (caps render cost) |
| `eval.render_max_steps` | steps per rendered clip |

The `clean` macro zeros `obs_dropout_lane`, `obs_dropout_boundary`,
`partner_blindness_prob`, `phantom_braking_prob`,
`phantom_braking_trigger_prob` and sets `traffic_light_behavior=1`. A value set
explicitly in the section wins over the macro (e.g. `env.traffic_light_behavior
= 0` keeps red lights ignored even with `clean = true`).

## Running evaluation

### Inline during training

Any `enabled` evaluator with `interval > 0` runs automatically every `interval`
epochs (and once at shutdown). Nothing extra to do — the metrics land in
wandb/TensorBoard under `<name>/<metric>` and renders under `<name>/render`.

### Standalone, by name

```bash
puffer eval puffer_drive --evaluator validation_gigaflow \
    --load-model-path experiments/puffer_drive_xxxx/models/model_000500.pt
```

Runs that one evaluator with its `[eval.validation_gigaflow]` config. The
checkpoint's network architecture is read from the sibling `config.yaml` (next
to `models/`), so a checkpoint loads even if its policy/rnn dims differ from
`drive.ini`. With no `--load-model-path`, a fresh (random) policy is used —
useful for smoke-testing the eval path itself.

### Standalone, ad-hoc

Same as by-name, except instead of naming an evaluator you select one of the two
built-in `validation_*` evaluators by simulation and override its config from the
CLI — no `drive.ini` edit needed:

- `--eval_simulation gigaflow` → runs the `validation_gigaflow` section
- `--eval_simulation replay` → runs the `validation_replay` section

The flags below override that evaluator's config for this run, and each applies
**only when passed** — omit one and the evaluator's own `[eval.*]` value stands:

```bash
puffer eval puffer_drive --eval_simulation gigaflow \
    --load-model-path <ckpt> \
    --num_scenarios 50 --render 1 --render-backend obs_html --num_maps 4
```

| Flag | Effect |
|---|---|
| `--eval_simulation gigaflow\|replay` | selects `validation_<sim>` when `--evaluator` is absent |
| `--num_scenarios N` | override the evaluator's `eval.num_scenarios` |
| `--render 0\|1` | toggle rendering on/off |
| `--render-backend egl\|triage_html\|obs_html` | choose the renderer (see Render backends) |
| `--num_maps N` | override `env.num_maps` (CARLA maps for gigaflow, bin count for replay) |

Any other section value can be overridden with the generic dotted form, e.g.
`--eval.validation-replay.env.scenario-length 91`.

### Subprocess mode

`mode = "subprocess"` runs the evaluator in a fresh `python -m pufferlib.pufferl
eval … --out <json>` process that loads the latest checkpoint from disk; the
parent reads metrics back from the JSON. Use it to isolate a heavy/leaky eval
from the training process.

## Outputs

- **Aggregate metrics** — a weighted per-agent mean of the env's `vec_log`
  emissions, logged to wandb/TensorBoard. Always produced.
- **Per-episode CSV** (`eval.export_episode_csv = true`) — one row per finished
  episode in `episode_metrics/<name>_epoch{E}_step{N}.csv`, including
  `map_name`/`scenario_id` and the per-episode metrics. Drains the env's
  `completed_episode` summaries, which the manager enables automatically for
  evaluators that opt in.
- **Coverage** (`eval.verify_coverage = true`) — folds `coverage_expected`,
  `coverage_found`, `coverage_unique_maps`, `coverage_complete` into the
  metrics and logs any maps evaluated more than once. For a unique-scenario
  sweep (replay) duplicates flag a problem; for cycling maps (gigaflow) they
  are expected.
- **Renders** — selected by `render_backend` (see below). `render_num_scenarios`
  caps how many scenarios are rendered, so render cost stays bounded regardless
  of `num_scenarios`.

### Render backends

`render_backend` picks one renderer (it is not a stack — exactly one runs):

| `render_backend` | Output | Shows | Built from | Use it to |
|---|---|---|---|---|
| `egl` (default) | mp4 per (scenario, view) | top-down sim camera | GPU EGL → ffmpeg | get a shareable video clip |
| `triage_html` | one HTML per episode → `gif/<name>/` | scene playback **+ per-episode metrics** | the captured compact-replay bundle (no re-sim) | triage *which* episodes failed |
| `obs_html` | one HTML per scenario (+ gallery `index.html`) → `obs/<name>/` | interactive scene **+ each agent's NN observation** | a CPU re-roll capturing state + obs | inspect *what the policy sees* |

Both HTML backends are CPU-only (no EGL/ffmpeg). `triage_html` is lighter (it
reuses data already captured during the rollout); `obs_html` re-simulates to
record the observation, so it's heavier but shows the policy's actual inputs.

## The built-in evaluators

| Section | Type | What it runs |
|---|---|---|
| `validation_replay` | multi_scenario | replay sweep over a nuPlan bin directory, `control_sdc_only` |
| `validation_gigaflow` | multi_scenario | gigaflow sweep over the CARLA maps |
| `wosac` | wosac | Waymo open sim agents challenge metrics |

`validation_replay` and `validation_gigaflow` inherit shared eval reward
weights and clean-eval knobs from the `validation_defaults` template.

## Adding an evaluator

1. Subclass `Evaluator` (or an existing type) in
   `pufferlib/ocean/benchmark/evaluators/`, set `type_name`, and register it in
   `evaluators/__init__.py`. Most subclasses only override `env_overrides`,
   `_should_stop`, and optionally `_render_env_overrides`; the base `rollout`
   handles the step loop, metric aggregation, CSV, and coverage.
2. Add an `[eval.<name>]` section with `type = "<your_type_name>"`.

## Scripts

`scripts/eval/` drives the unified pipeline over many checkpoints:

- `run_all_eval.sh` — eval the latest checkpoint in every `experiments/*/`.
- `run_all_latest_eval.py` — eval the latest checkpoint in every `runs/*/`, with rendering.
- `run_failure_scenarios.py` — re-eval from a failure CSV.

All call `puffer eval puffer_drive` with the flags above.
