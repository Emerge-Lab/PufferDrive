# nuPlan data — download, conversion, usage

How to obtain the nuPlan dataset and turn it into the `.bin` scenario files
that PufferDrive's replay mode and the nuPlan evaluators consume.

## Default path — fetch pre-converted bins

Pre-converted bins are publicly downloadable:

```bash
python data_utils/fetch_data.py   # the ~10 GB nuplan_mini_train + nuplan_mini_val sets
```

## Converting yourself

The pipeline has two external stages:

```
raw nuPlan  →  [py123d]  →  .arrow  →  [123Drive]  →  .bin
```

- [py123d](https://github.com/kesai-labs/py123d) downloads nuPlan and parses
  it into its arrow format ([nuPlan guide](https://github.com/kesai-labs/py123d/blob/main/docs/datasets/nuplan.rst)).
- [123Drive](https://github.com/vcharraut/123Drive) converts py123d arrow
  data into PufferDrive `.bin` files.

## Stage 1 — download and parse nuPlan with py123d

nuPlan is Motional's planning dataset, distributed from a public AWS bucket
under a non-commercial license
([terms](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE)).
The sensor blobs (camera/lidar) are not needed — replay only uses the logged
agent trajectories and map geometry, which the defaults below fetch.

```bash
pip install "py123d[nuplan]"
pip install "nuplan-devkit @ git+https://github.com/motional/nuplan-devkit/@nuplan-devkit-v1.2"

# Both stages read these env vars — export them before any py123d command
# (details in the py123d nuPlan guide linked above):
export NUPLAN_DATA_ROOT=/path/to/nuplan_raw       # raw logs + maps land here
export NUPLAN_MAPS_ROOT=$NUPLAN_DATA_ROOT/maps
export PY123D_DATA_ROOT=/path/to/py123d_out       # arrow output root

# Mini set (~11 GB) — enough for replay training and the nuPlan evals:
py123d-download dataset=nuplan \
    'dataset.downloader.splits=[nuplan-mini_train, nuplan-mini_val, nuplan-mini_test]'

# Parse the downloaded logs + maps into py123d's arrow format:
py123d-conversion dataset=nuplan-mini
```

Alternatively, use the streaming mode described in the py123d guide
(`py123d-conversion dataset=nuplan-mini-stream`), which downloads to a
temporary directory and cleans up after itself. The full (non-mini) set is
~135 GB: `py123d-download dataset=nuplan` / `py123d-conversion dataset=nuplan`.

## Stage 2 — convert arrow to .bin with 123Drive

```bash
git clone https://github.com/vcharraut/123Drive && cd 123Drive
uv sync

# PY123D_DATA_ROOT must still be exported (per stage 1): the converter
# resolves map data through it, not through --py123d_path.
uv run convert --preset nuplan --datasets nuplan-mini \
    --py123d_path $PY123D_DATA_ROOT --output ./nuplan_bins
```

`--py123d_path` points at the py123d dataset root (the directory containing
`logs/` and `maps/` from stage 1). `--datasets` must match the converted
dataset's name: mini-set scenes carry the `nuplan-mini` prefix, and without
the flag the scan defaults to `nuplan` and finds no scenarios. Drop the flag
when converting the full set.

**RAM notes:**

- `--workers` defaults to `0` = 80% of CPU cores, and each worker holds a
  parsed log in memory. If conversion OOMs, pass an explicit low worker
  count (e.g. `--workers 4`) instead of the default.
- Keep the `--preset nuplan` flag: among other settings it pins
  `--duration_s 20`, chunking nuPlan's minute-long logs into 20 s scenarios
  precisely to avoid RAM exhaustion.

Useful extras: `--num_scenes N` limits output for a quick smoke test, and
`uv run web` serves a browser viewer for the produced bins at
`http://localhost:8080`.

## Expected output layout

The converter emits a flat directory of one `.bin` per scenario, named by the
nuPlan scenario token:

```
nuplan_mini_train_bins/
├── nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin
├── nuplan__...bin
└── ...
```

A checked-in sample lives at
`pufferlib/resources/drive/binaries/nuplan/nuplan__00018a38-0063-54d1-a3c1-1ab931a4a1e5.bin`.

## Using the bins

Replay training on nuPlan bins, controlling only the SDC:

```bash
puffer train puffer_drive \
    env.map_dir=data/nuplan_mini_train \
    env.num_maps=250 \
    env.simulation_mode=replay \
    env.control_mode=control_sdc_only \
    env.scenario_length=200
```

## Enabling the nuPlan evals

The `validation_replay` evaluator in `pufferlib/config/puffer_drive.yaml`
ships **disabled**, with `env.map_dir` preset to `data/nuplan_mini_val` —
where the default fetch lands.

- **Inline during training:** after fetching, pass
  `eval.validation_replay.enabled=true` (or flip it in the yaml), and point
  `env.map_dir` elsewhere if your bins live elsewhere.
- **Standalone:** `puffer eval --evaluator <name>` runs a named evaluator
  even when it is disabled; only `env.map_dir` needs overriding, via the
  generic dotted Hydra form:

```bash
puffer eval puffer_drive --evaluator validation_replay \
    eval.validation_replay.env.map_dir=data/nuplan_mini_val
```

See `docs/evaluation.md` for the full evaluator config schema.
