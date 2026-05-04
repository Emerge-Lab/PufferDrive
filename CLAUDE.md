# CLAUDE.md

Operational notes for working on PufferDrive's `puffer-4` branch — local builds and the NYU Greene cluster setup.

## Workflow

Claude Code runs on the user's Mac. The user pushes code-edit requests through that local session, but **all cluster operations are expected to run *on* the cluster login node**, not from the Mac. Tooling that touches SLURM (`scripts/rebuild_on_cluster.py`, `scripts/submit_cluster.py`, `setup_container.sh`) calls `sbatch` / `subprocess.run` directly — no `ssh torch '...'` round-trips.

To execute commands on the cluster from this local session, prefix with `ssh torch '<cmd>'`.

## Quick reference

```bash
# Local Mac CPU build (after brew install libomp)
uv venv && source .venv/bin/activate
uv pip install -e .
python setup.py build_ext --inplace --force

# Cluster rebuild (run ON the cluster login node — submits a SLURM job that builds inside the overlay)
python scripts/rebuild_on_cluster.py --account $ACCOUNT --wait
```

## Repo layout

- `sim/` — C simulation engine. `binding.c` + `drive.h`/`drive.c`/`datatypes.h`/`env_fields.h`/`render.h`. Compiled as **C** (uses C11 `<stdatomic.h>`, designated initializers, void* implicit conversions — won't compile cleanly as C++).
- `src/` — CUDA training backend + Python entry points. `bindings.cu` (CUDA, default torch backend), `bindings_cpu.cpp` (CPU pybind11 fallback), `pufferlib.cu`, `kernels.cu`, `models.cu`, `vecenv.h` (vec-env helper, OpenMP).
- `pufferlib/` — Python package. The compiled extension lands here as `_C${EXT_SUFFIX}`.
- `config/drive.ini` — env/training config (see `sim/env_fields.h` for the wired knobs).
- `scripts/` — cluster tooling.
- `vendor/` — bundled raylib, cJSON, ini.c. `vendor/raylib-5.5_*` is gitignored and downloaded by `setup.py` on demand.
- `build.sh` — canonical compile script. Two-step (static lib from `binding.c` → link with `bindings.cu`/`bindings_cpu.cpp`).

## Building

`setup.py` picks the right path automatically:

- **macOS / no CUDA** → native `setuptools.Extension`. Compiles `sim/binding.c` (as C) + `src/bindings_cpu.cpp` (as C++) into `pufferlib._C` via plain pybind11. Requires `brew install libomp`.
- **Linux + CUDA** (`$CUDA_HOME/bin/nvcc` exists) → custom `build_ext` shells out to `./build.sh`, which writes `pufferlib/_C${EXT_SUFFIX}.so` directly. The `.cpp_extension.CUDAExtension` path doesn't work for puffer-4 because it forces `.c` files through `c++`, breaking the C-only headers.
- `PUFFER_CPU=1` forces the CPU path on Linux.

`build.sh` flags worth knowing:
- `./build.sh` — torch backend (`bindings.cu` with `PUFFER_NATIVE_PUFFERL=0`).
- `./build.sh --cuda` — native PuffeRL backend.
- `./build.sh --cpu` — `bindings_cpu.cpp` only (no CUDA).
- `./build.sh --fast` / `--local` — standalone executable.
- `OMP_LIB="-L/path/lib -lomp"` — env-var overridable. Default `-lomp5` (Linux) / `-lomp` (Mac). conda-forge `llvm-openmp` ships `libomp` (not `libomp5`).

## Cluster (NYU Greene)

### SSH

`ssh torch` is the assumed alias for the Greene login node, set up in `~/.ssh/config`. From this local Claude Code session it's the only way to run anything on the cluster — but the cluster-side scripts themselves (`rebuild_on_cluster.py`, `submit_cluster.py`) are designed to run *on* the cluster, so the typical pattern is `ssh torch 'cd /scratch/$USER/code/PufferDrive && python scripts/<tool>.py ...'`.

### Paths

| Thing | Path |
|---|---|
| Project root | `/scratch/$USER/code/PufferDrive` |
| Overlay (writable ext3) | `/scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3` |
| Singularity image | `/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif` |
| Overlay template | `/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz` |
| Job logs | `/scratch/$USER/rebuild_logs/` |
| miniforge inside overlay | `/ext3/miniforge3/` (sourced via `/ext3/env.sh`) |

These match `submit_cluster.py`'s `--container_image` / `--container_overlay` defaults and `setup_container.sh`'s defaults. `setup_container.sh` is env-var overridable (`OVERLAY_PATH`, `IMAGE_PATH`, `CONTAINER_DIR`).

### SLURM accounts

The user has multiple `torch_pr_*` allocations (e.g. `torch_pr_102_tandon_advanced`, `torch_pr_355_general`, `torch_pr_924_general`, `torch_pr_975_general`, `torch_pr_1009_*`, `torch_pr_1010_*`). Default account hits `QOSGrpGRE` if a long training job is using its GPU quota — switch to a different `torch_pr_*` family. Use `sacctmgr show assoc user=$USER --parsable2 -n format=Account` to list them.

### Container invocation patterns

**Read-only** (training, rebuild — what `submit_cluster.py` and `rebuild_on_cluster.py` use):
```bash
singularity exec --nv \
    --overlay /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    bash -c 'source /ext3/env.sh && export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" && cd $PROJECT && <cmd>'
```

**Writable** (installs):
```bash
singularity exec --nv --fakeroot \
    --overlay /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3 \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    bash -c 'source /ext3/env.sh && <install cmds>'
```

### Why writes are constrained on Greene

The user is **not in `/etc/subuid`**, so `--fakeroot` falls back to `apptainer`'s "root-mapped namespace + fakeroot command" mode using fuse-overlayfs. Effects to be aware of:

- Writes intended for paths under `/usr`, `/root`, `/tmp` etc. land in the overlay's `upper/usr/`, `upper/root/`, `upper/tmp/` directories. Apptainer surfaces only `upper/ext3/*` as `/ext3`, so writes to other paths are **invisible from inside the container** but **still consume blocks in the ext3 file**.
- Failed pip installs that wrote to `/usr/local/lib/python3.X/site-packages/...` show up as `upper/usr/local/...` in the raw ext3 — `du /ext3` won't see them. Use `debugfs -R "ls /upper" overlay.ext3` from a login node to inspect.
- `--overlay X:rw` (no fakeroot) returns `permission denied` on apptainer's session upper dir. There's no way to mount the overlay genuinely-writable as a regular user without subuid.

### Cluster overlay maintenance

Cleanup invisible cruft (e.g. after a partial install ENOSPC'd):

```bash
# On login node — overlay must NOT be in active use by any job
MOUNT=$HOME/.overlay_mount
mkdir -p $MOUNT
fuse2fs -o rw,fakeroot /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3 $MOUNT
ls $MOUNT/upper                     # see what's there outside upper/ext3
rm -rf $MOUNT/upper/usr             # or whatever's bloating
fusermount3 -u $MOUNT && rmdir $MOUNT
e2fsck -f -p /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3
```

Inspect from outside without mounting:

```bash
debugfs -R "ls /upper" /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3
debugfs -R "ls /upper/ext3/miniforge3" overlay.ext3
```

### Adding a dependency to the overlay

Submit a writable-mount job that installs into `/ext3/miniforge3` (which **is** under `upper/ext3/`, so it persists where setup expects). Example for adding `clang` and `ccache`:

```bash
sbatch --account=$ACCOUNT --gres=gpu:1 --cpus-per-task=2 --mem=4gb --time=15 \
  -o /scratch/$USER/rebuild_logs/install_%j.log --wrap "
singularity exec --nv --fakeroot \
    --overlay /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3 \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif bash -c '
        source /ext3/env.sh
        mamba install -n base -c conda-forge -y clang clangxx ccache
'"
```

`mamba install` writes to `/ext3/miniforge3/...` (visible inside the container, persistent). `pip install pkg` from inside the env also lands there.

`pip install --user pkg` writes to `~/.local/lib/...` which lives on home, **not** the overlay — useful when you can't get a writable mount.

### Rebuild after a code change

```bash
# Push your branch from the Mac first, then on the cluster login node:
ssh torch 'cd /scratch/$USER/code/PufferDrive && git pull --ff-only origin <branch> && \
    python scripts/rebuild_on_cluster.py --account $ACCOUNT --wait'
```

`rebuild_on_cluster.py` runs on the cluster (uses `subprocess.run`, not ssh). It writes a self-contained sbatch script to `/scratch/$USER/rebuild_logs/`, submits it, optionally polls until it finishes, and prints the log. Sets `TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"` so the rebuilt `_C.so` runs on A100/L40S/H100/H200 without "no kernel image" crashes when training jobs land on a different GPU type. `--user` defaults to the cluster `$USER` and rarely needs setting.

## Common pitfalls

- **`undefined symbol: ncclCommShrink` on `from torch._C import *`** — the cuda12.8.1 sif at `/share/apps/...` ships libnccl 2.25.1 in `/usr/lib/x86_64-linux-gnu/`. torch ≥ 2.10 (cu128 wheels) calls `ncclCommShrink`, added in NCCL 2.27.5. torch ships its own bundled NCCL 2.27.5 in `site-packages/nvidia/nccl/lib/`, but in `torchrun`-spawned child processes the loader sometimes resolves libtorch_cuda's `libnccl.so.2` from the sif's old `/usr/lib` instead, missing the symbol. Fix: prepend torch's NCCL dir to `LD_LIBRARY_PATH` so it wins. `setup_container.sh install`/`rebuild` patches `/ext3/env.sh` to do this automatically; `submit_cluster.py` and `rebuild_on_cluster.py` also prepend at launch as belt-and-suspenders. Existing overlays built before the env.sh patch can be retrofitted by mounting writable and appending the same `NCCL_DIR=$(compgen -G ...); export LD_LIBRARY_PATH=...` block to `/ext3/env.sh`.
- **`CONDA_PREFIX` is empty** even after `source /ext3/env.sh` from `/ext3/miniforge3/bin/python3`. Use `sys.prefix` to find the active conda env's lib dir.
- **`-lomp5` on Linux fails** with conda-forge openmp; default is for older Intel OpenMP packagings. setup.py overrides via `OMP_LIB`.
- **clang's default library search path doesn't include `$prefix/lib`** even when running from a conda env. Pass `-L$sys.prefix/lib` explicitly.
- **`du /ext3` undercounts** when the overlay has cruft outside `upper/ext3/`. Compare `df` vs `du`; `debugfs -R "ls /upper"` reveals hidden directories.
- **`e2fsck` on a clean overlay won't recover space** if the issue is hidden directories rather than orphan inodes. The space is "in use" by visible-to-debugfs files at non-`/ext3` paths.
- **build.sh assumes `clang` and `ccache` in PATH.** Both are installable via `mamba install -c conda-forge clang clangxx ccache`.
- **Squash-merging stacked PRs**: `git push --force-with-lease` may report "stale info" when pushing through a token URL different from `origin`. Either fetch first, or use `--force` with care.

## Branches of note

- `puffer-4` — active development branch, base for new PRs.
- `2.0` — older release branch; `puffer-4`'s PRs sometimes target it for the diff.
- `emerge/temp_training` — different layout (`pufferlib/ocean/drive/...`). Uses raw CPython API in env bindings (no pybind11) and `torch.utils.cpp_extension.CUDAExtension` for the training side. Useful reference but **don't copy paths verbatim**.
- `vcha/turbostream` — source for the blind-agent feature port.
- `ev/merge_turbostream` — source for `rebuild_on_cluster.py`.
