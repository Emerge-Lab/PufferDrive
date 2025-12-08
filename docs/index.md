# PufferDrive

PufferDrive is a high-throughput autonomous driving simulator built on [PufferLib](https://puffer.ai). It pairs a fast vectorized simulator with data conversion and benchmarking scripts so you can train and evaluate agents with minimal setup.

## Highlights
- Multi-agent drive environment with batched stepping for speed.
- Scripts to convert Waymo Open Motion Dataset JSON into lightweight binaries.
- Benchmarks for distributional realism and human compatibility.
- Raylib-based visualizer for local or headless render/export.

## Quick start
- Follow [Getting Started](getting-started.md) to install, build the C extensions, and run `puffer train puffer_drive`.
- Consult [Simulator](simulator.md) for how actions/observations, rewards, and `.ini` settings map to the underlying C environment and Torch policy.
- Prepare drive map binaries with the steps in [Data](data.md).
- Evaluate a policy with the commands in [Evaluation](evaluation.md) and preview runs with the [Visualizer](visualizer.md).

## Repository layout
- `pufferlib/ocean/drive`: Drive environment implementation and map processing utilities.
- `resources/drive/binaries`: Expected location for compiled map binaries (outputs of the data conversion step).
- `scripts/build_ocean.sh`: Helper for building the Raylib visualizer and related binaries.
- `examples`, `tests`, `experiments`: Reference usage, checks, and research scripts that pair with the docs pages.
