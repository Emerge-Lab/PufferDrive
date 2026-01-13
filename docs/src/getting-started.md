# Getting started

This page walks through installing PufferDrive from source, building the native extensions, and running a first training job.

## Prerequisites
- Python 3.9+ with a virtual environment manager (`uv`, `venv`, or `conda`).
- A C/C++ toolchain for building the bundled extensions (GCC/Clang + make).
- [PyTorch](https://pytorch.org/) installed inside your environment (pick the CPU/GPU wheel that matches your setup).

## Installation
Clone and set up an isolated environment:

```bash
git clone https://github.com/Emerge-Lab/PufferDrive.git
cd PufferDrive
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

Build the C extensions in place:

```bash
python setup.py build_ext --inplace --force
```
Run this with your virtual environment activated so the compiled extension links against the correct Python.

### When to rebuild the extension
- Re-run `python setup.py build_ext --inplace --force` after changing any C/Raylib sources in `pufferlib/ocean/drive` (e.g., `drive.c`, `drive.h`, `binding.c`, `visualize.c`) or after pulling upstream changes that touch those files. This regenerates the `binding.cpython-*.so` used by `Drive`.
- Pure Python edits (training scripts, docs, data utilities) do not require a rebuild; just restart your Python process.

## Verify the setup
Once map binaries are available (see [Data](data.md)), launch a quick training run to confirm the environment, data, and bindings are wired up correctly:

```bash
puffer train puffer_drive
```

For multi-node training (only uses Data Parallelism with torch ddp)
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=6 -m puffer train puffer_drive
```

If map binaries are missing, follow the steps in [Data](data.md) to generate them before training. See [Visualizer](visualizer.md) for rendering runs and [Evaluation](evaluation.md) for benchmark commands.


## Logging with Weights & Biases

Enable W&B logging with the built-in CLI flags (the package is already a dependency in `setup.py`):

```bash
puffer train puffer_drive --wandb --wandb-project pufferdrive --wandb-group local-dev
```

- Add `--wandb` to turn on logging; `--wandb-project` and `--wandb-group` set the destination in W&B.
- Checkpoint uploads and evaluation helpers (`pufferlib/utils.py`) will log WOSAC/human-replay metrics and rendered videos when W&B is enabled.
