# nuPlan data — download, conversion, usage

How to obtain the nuPlan dataset and turn it into the `.bin` scenario files
that PufferDrive's replay mode and the nuPlan evaluators consume.

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

# Mini set (~11 GB) — enough for replay training and the nuPlan evals:
py123d-download dataset=nuplan \
    'dataset.downloader.splits=[nuplan-mini_train, nuplan-mini_val, nuplan-mini_test]'

# Parse the downloaded logs + maps into py123d's arrow format:
py123d-conversion datasets=["nuplan-mini"]
```

Set `NUPLAN_DATA_ROOT` (and `NUPLAN_MAPS_ROOT`) per the
[py123d nuPlan guide](https://github.com/kesai-labs/py123d/blob/main/docs/datasets/nuplan.rst)
before converting, or use the streaming mode described there
(`py123d-conversion dataset=nuplan-mini-stream`), which downloads to a
temporary directory and cleans up after itself. The full (non-mini) set is
~135 GB: `py123d-download dataset=nuplan` / `py123d-conversion datasets=["nuplan"]`.

## Stage 2 — convert arrow to .bin with 123Drive

```bash
git clone https://github.com/vcharraut/123Drive && cd 123Drive
uv sync
uv run convert --preset nuplan --py123d_path /path/to/py123d/data --output ./nuplan_bins
```

`--py123d_path` points at the py123d dataset root (the directory containing
`logs/` and `maps/` from stage 1).

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

- `env.map_dir` may point either at a directory of `.bin` files or at a
  single `.bin` file (`pufferlib/ocean/drive/drive.py:268-273`).
- `env.num_maps` must not exceed the number of `.bin` files in `map_dir`;
  the env aborts otherwise (`pufferlib/ocean/drive/drive.py:368-369`).

## Using the bins

Replay training on nuPlan bins, controlling only the SDC:

```bash
puffer train puffer_drive \
    --env.map-dir /path/to/nuplan_mini_train_bins \
    --env.num-maps 250 \
    --env.simulation-mode replay \
    --env.control-mode control_sdc_only \
    --env.scenario-length 200
```

This mirrors the `[eval.validation_replay]` setup in
`pufferlib/config/ocean/drive.ini`. nuPlan mini logs are ~201 steps at 10 Hz,
so `scenario_length` of 200 covers one full log.

## Enabling the nuPlan evals

`[eval.validation_replay]` and `[eval.behaviors_full_dir]` in
`pufferlib/config/ocean/drive.ini` ship **disabled** with placeholder
`env.map_dir` values, because they require a local nuPlan bin directory.

- **Inline during training:** edit the section — point `env.map_dir` at your
  bin directory and set `enabled = true`.
- **Standalone:** `puffer eval --evaluator <name>` runs a named evaluator
  even when it is disabled; only `env.map_dir` needs overriding, via the
  generic dotted CLI form:

```bash
puffer eval puffer_drive --evaluator validation_replay \
    --eval.validation-replay.env.map-dir /path/to/nuplan_mini_train_bins
```

Per-category behavior evals (`[eval.behaviors_<category>]`) are user-defined
sections: each inherits `behaviors_defaults`, sets `type = "behavior_class"`,
and points `env.map_dir` at a directory of bins for one labelled scene
category (hard_stop, merge, ...). See `docs/evaluation.md` for the full
evaluator config schema.
