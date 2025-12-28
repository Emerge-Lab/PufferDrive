# Data

PufferDrive consumes map binaries generated from multiple data sources, including the [Waymo Open Motion Dataset (WOMD)](https://github.com/waymo-research/waymo-open-dataset) JSON files, [ScenarioMax](https://github.com/valeoai/V-Max), and [CARLA](https://carla.org/). This page covers how to obtain data and convert it into the binary format expected by the simulator.

## Download options

- [`pufferdrive_womd_train`](https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_train): **10k scenarios** from the Waymo Open Motion _training_ dataset.
- [`pufferdrive_womd_val`](https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_val): **10k scenarios** from the Waymo Open Motion _validation_ dataset.
- [`pufferdrive_mixed`](https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_train_carla_mixed): **10,200** scenarios. The 10K from the WOMD train set above + Towns 1 and 2 duplicated 100x each.
- Additional compatible sources: [ScenarioMax](https://github.com/valeoai/ScenarioMax) exports JSON in the same format.
- Included [CARLA](https://carla.org/) maps: Readily available CARLA maps live in `data_utils/carla/carla_data`.

### Download via Hugging Face

Install the CLI once:

```bash
uv pip install -U "huggingface_hub[cli]"
```

Download:
```bash
huggingface-cli download daphne-cornelisse/pufferdrive_womd_train \
  --repo-type dataset \
  --local-dir data/processed/training
```

Place raw JSON files under `data/processed/training` (default location read by the conversion script).

## Convert JSON to map binaries

The conversion script writes compact `.bin` maps to `resources/drive/binaries`:

```bash
python pufferlib/ocean/drive/drive.py
```

Notes:

- The script iterates every JSON file in `data/processed/training` and emits `map_XXX.bin` files.
- `resources/drive/binaries/map_000.bin` ships with the repo for quick smoke tests; generate additional bins for training/eval.
- If you want to point at a different dataset location or limit the number of maps, adjust `process_all_maps` in `pufferlib/ocean/drive/drive.py` before running.

## Map binary format reference

The simulator reads the compact binary layout produced by `save_map_binary` in `pufferlib/ocean/drive/drive.py` and parsed by `load_map_binary` in `pufferlib/ocean/drive/drive.h`:

- **Header**: `sdc_track_index` (int), `num_tracks_to_predict` (int) followed by that many `track_index` ints, `num_objects` (int), `num_roads` (int).
- **Objects (vehicles/pedestrians/cyclists)**: For each object, the writer stores `scenario_id` (`unique_map_id` passed to `load_map`), `type` (`1` vehicle, `2` pedestrian, `3` cyclist), `id`, `array_size` (`TRAJECTORY_LENGTH = 91`), positions `x/y/z[91]`, velocities `vx/vy/vz[91]`, `heading[91]`, `valid[91]`, and scalars `width/length/height`, `goalPosition (x, y, z)`, `mark_as_expert` (int). Missing trajectory entries are zero-padded by the converter.
- **Road elements**: Each road entry stores `scenario_id`, a remapped `type` (`4` lane, `5` road line, `6` road edge, `7` stop sign, `8` crosswalk, `9` speed bump, `10` driveway), `id`, `array_size` (#points), then `x/y/z` arrays of that length and scalars `width/length/height`, `goalPosition`, `mark_as_expert`. `save_map_binary` also simplifies long polylines (`len(geometry) > 10` and `type <= 16`) with a 0.1 area threshold to keep files small.
- **Control hints**: `tracks_to_predict` and `mark_as_expert` influence which agents are controllable (`control_mode` in the simulator) versus replayed as experts or static actors (`set_active_agents` in `drive.h`).

Refer to [Simulator](simulator.md) for how the binaries are consumed during resets, observation construction, and reward logging.

## Verifying data availability

- After conversion, `ls resources/drive/binaries | head` should show numbered `.bin` files.
- If you see `Required directory resources/drive/binaries/map_000.bin not found` during training, rerun the conversion or check paths.
- With binaries in place, run `puffer train puffer_drive` from [Getting Started](getting-started.md) as a smoke test that the build, data, and bindings are wired together.
- To inspect the binary output, convert a single JSON file with `load_map(<json>, <id>, <output_path>)` inside `drive.py`.

## Interactive scenario editor

See [Interactive scenario editor](scene-editor.md) for a browser-based workflow to inspect, edit, and export Waymo/ScenarioMax JSON into the `.bin` format consumed by the simulator.

## Generate CARLA agent trajectories

The agent trajectories in the provided CARLA maps are procedurally generated assuming a general velocity range without a valid initial state (no collision/offroad). The repository uses an external submodule for CARLA XODR processing (`pyxodr`).

To generate your own CARLA agent trajectories, install the submodules and developer requirements (editable install) before running the generator:

```bash
git submodule update --init --recursive

python -m pip install -e . -r requirements-dev.txt
```

Run the generator script. Important optional args:

- `--num_objects`: how many agents to initialize in a map (default: map-dependent)
- `--num_data_per_map`: number of data files to generate per map
- `--avg_speed`: controls the gap between subsequent points in the trajectory

```bash
python data_utils/carla/generate_carla_agents.py --num_objects 32 --num_data_per_map 8 --avg_speed 2
```

There is also a visualizer for inspecting initial agent positions on the map:

```bash
python data_utils/carla/plot.py
```

Notes:

- Base Carla maps that agents are spawned live under `data_utils/carla/carla_py123d` and the Carla XODRs are at `data/CarlaXODRs` to interact with the `pyxodr` submodule for XODR parsing and agent traj generation.
- If you encounter missing binary or map errors, ensure the submodule was initialized and the required packages from `requirements-dev.txt` are installed.
