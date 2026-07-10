# nuPlan data — download, conversion, usage

How to obtain the nuPlan dataset and turn it into the `.bin` scenario files
that PufferDrive's replay mode and the nuPlan evaluators consume.

## Getting nuPlan

nuPlan is Motional's planning dataset. Register and download it from the
official page: <https://www.nuscenes.org/nuplan>.

For PufferDrive replay you need:

- the **maps** package, and
- the **mini split** log databases (`.db` files).

The sensor blobs (camera/lidar) are not needed — replay only uses the logged
agent trajectories and map geometry.

## Converting to PufferDrive .bin

Conversion is done by the external **py123d** converter; it is not part of
this repository. Prior converted datasets in this repo's history: the CARLA
town bins were converted via py123d (commit `c2667356`), and the nuPlan
behavior-category eval bins referenced by the config were a py123d v0.2.1
reconvert (commit `2b1d3ecb`).

<!-- TODO(maintainers): converter repo URL + exact invocation -->
- Converter repository: *(to be filled by maintainers)*
- Invocation: *(to be filled by maintainers)*

**RAM note:** convert with reduced parallelism. Each converter worker holds a
fully parsed nuPlan log database in memory, and the default worker count
assumes a large-RAM machine; on a laptop or small node the defaults will OOM.

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
