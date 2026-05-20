# Cluster training — operational guide

How to run PufferDrive training on a SLURM cluster. This is written with the NYU cluster in mind but it should mostly hold for any SLURM cluster. 

## A quick overview of the setup and launch process

```bash
# One-time per cluster:
#   (a) create the singularity overlay and install deps into the venv
./scripts/setup_container.sh create-overlay
sbatch --account=<acct> --gres=gpu:1 --cpus-per-task=8 --mem=32gb --time=60 \
    --wrap "./scripts/setup_container.sh install"
#   (b) install submitit on the login-node system python (used to compose
#       the submission; the in-container venv python runs the actual job)
python3 -m ensurepip --user
python3 -m pip install --user submitit pyyaml cloudpickle

# If code changes, or we haven't built before, rebuild the C code in the container
sbatch --account=<acct> --partition=cpu_short --cpus-per-task=8 --mem=16gb --time=20 \
    --chdir=$PWD -o $LOGDIR/rebuild_%j.log \
    --wrap "./scripts/setup_container.sh rebuild"

# Training: submit_cluster.py from the login node with --container --heartbeat.
# By default launches RL training but can be modified through the --main argument
# to launch other modes
python3 scripts/submit_cluster.py \
    --save_dir /scratch/$USER/runs \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --container --heartbeat \
    --account <acct> --partition <gpu-partition> --time 2880 \
    --args train.checkpoint_interval=250 env.simulation_mode=gigaflow # use this to override config args
```

## Container model

PufferDrive on Greene runs inside a singularity container. The container provides
a modern glibc + CUDA toolkit; the project's Python environment lives in a venv
on `/scratch` so installs aren't bottlenecked by the slow process of building a venv inside a container.

The container is invoked with a **read-only** overlay mount for the miniforge3
base interpreter, plus the on-disk venv for project packages. As an example of running such a command:
```bash
singularity exec --nv \
    --overlay /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    bash -c '
        source /scratch/$USER/venvs/pufferdrive/bin/activate
        export PYTHONNOUSERSITE=1
        cd /scratch/$USER/code/PufferDrive
        <your command>
    '
```

## Submitting training — `submit_cluster.py`

`scripts/submit_cluster.py` is the canonical submission path. It composes: 
- a `compute_config` YAML (SLURM settings)
- a `program_config` YAML (pufferl training args)
- `--args` CLI overrides
- wraps the inner train command in `singularity exec` when `--container` is set
- optionally injects the GPU heartbeat when `--heartbeat` is set. WARNING: this is specifically for the torch cluster to prevent our jobs being killed. No one else should use this.

It performs code isolation (symlinks the
top-level entries + hard-copies `pufferlib/` into a per-run sandbox), and
hands the package to `submitit` for `sbatch`-submission.

### WARNING: two python installation are being used here

A `submit_cluster.py --container` submission uses two distinct python
environments:

- **Login-side composer**: the python that runs `submit_cluster.py` itself.
  Only needs `submitit`, `pyyaml`, `cloudpickle` importable. Used purely to
  build the sbatch script and submit it to SLURM. On Greene this is
  `/usr/bin/python3` (system python) and you can run `pip install --user submitit pyyaml
  cloudpickle` to provide those deps.
- **Compute-side executor**: the python that runs the training job on the
  compute node. This is the **venv python** inside the singularity overlay. submitit's
  outer launcher is wrapped in `singularity exec` so it lands in this
  environment; `launch_training` then runs `torchrun` inside the same
  container.

### One-time login-side setup

```bash
# Greene's /usr/bin/python3 ships without pip; bootstrap it:
python3 -m ensurepip --user
python3 -m pip install --user --upgrade pip
python3 -m pip install --user submitit pyyaml cloudpickle
```

After this, `python3 -c 'import submitit'` works on the login node.

### Run from the login node

```bash
python3 scripts/submit_cluster.py \
    --save_dir /scratch/$USER/runs \
    --prefix mytrain \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --account <acct> --partition <gpu-partition> --time 2880 \
    --container \
    --heartbeat \
    --args \
        train.total_timesteps=10000000000 \
        train.checkpoint_interval=250
```

Key flags:

| Flag | Effect |
|---|---|
| `--container` | wraps both submitit's outer launcher and the inner train command in `singularity exec --nv --overlay $OVERLAY:ro $IMAGE` |
| `--heartbeat` | wraps the train command in a brace group that backgrounds `python scripts/gpu_heartbeat.py` preventing the cluster from killing your job due to low GPU usage |
| `--args key=value ...` | passes nested config keys (underscores converted to dashes) as `--key value` on the torchrun line; e.g. `env.simulation_mode=replay` becomes `--env.simulation-mode replay` |
| `--account` / `--partition` / `--time` | override `compute_config` SLURM settings |

### GPU heartbeat — required for long runs

`--heartbeat` is not optional for jobs over ~2 hours. Without it, the
cluster's idle-GPU reclaimer issues a `scancel` from `uid 0` (root) during
the first eval / checkpoint dip in GPU utilization.

`scripts/gpu_heartbeat.py` monitors `nvidia-smi` and runs short matmul bursts
when utilization drops below 65%, so the cluster always sees the GPU as
active. It cooperates with training and steps aside when training is busy.

### Environment knobs the container path sets

When `--container` is on, the inner bash command has these env vars set
before `cd $PROJECT_ROOT && <train>`:

```bash
source /scratch/$USER/venvs/pufferdrive/bin/activate
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=/scratch/$USER/cache
export WANDB_CACHE_DIR=/scratch/$USER/wandb_cache
export WANDB_CONFIG_DIR=/scratch/$USER/wandb_config
export WANDB_DATA_DIR=/scratch/$USER/wandb_data
export WANDB_DIR=/scratch/$USER/wandb_data
```

## CPU rebuild path

GPU partitions are routinely saturated by training jobs. `setup_container.sh
rebuild` doesn't need a GPU even though it compiles CUDA code: `nvcc` is a
cross-compiler. It generates PTX/SASS for each architecture in
`TORCH_CUDA_ARCH_LIST` without needing matching hardware on the build host,
the same way a C compiler can target ARM from an x86 host. The CUDA toolkit
itself (`nvcc`, headers, libs) lives in the cuda12.8.1 `.sif` image, so any
node that can mount the image can run the build — CPU partitions included.
The rebuild script exports `TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"` upfront, so
the resulting `.so` is a fat binary that runs on every GPU type at job time.
Submit to a CPU partition for fast turnaround:

```bash
sbatch --account=<general-account> --partition=cpu_short \
    --cpus-per-task=8 --mem=16gb --time=20 \
    --chdir=$PWD \
    -o /scratch/$USER/rebuild_logs/rebuild_%j.log \
    --wrap "./scripts/setup_container.sh rebuild"
```

`--chdir=$PWD` is required because the script uses `./scripts/`. Takes ~40s.

### Common pitfalls

- **`ncclCommShrink` undefined symbol** at `from torch._C import *`. Greene's
  cuda12.8.1 sif ships `libnccl 2.25.1` in `/usr/lib`, but torch ≥ 2.10 calls
  `ncclCommShrink` from NCCL ≥ 2.27.5. torch's own NCCL 2.27.5 sits in
  `site-packages/nvidia/nccl/lib/` and needs to win the loader search.
  `setup_container.sh install`/`rebuild` patches `/ext3/env.sh` to prepend that
  dir to `LD_LIBRARY_PATH`; existing overlays from before that patch need the
  same line appended to `/ext3/env.sh`.
- **`-lomp5` link errors on Linux** with conda-forge openmp. The default is for
  older Intel OpenMP packaging. `setup.py` honors `OMP_LIB="-L$prefix/lib -lomp"`.
- **`du /ext3` undercounts** when the overlay has cruft outside `upper/ext3/`
  (e.g. failed pip installs that wrote to `/usr/local/lib/...` end up in
  `upper/usr/local/` and aren't visible to apptainer's view). Use
  `debugfs -R "ls /upper" overlay.ext3` from a login node to inspect.

### `TORCH_CUDA_ARCH_LIST`: a warning that you can skip

PufferDrive's C extension contains CUDA kernels. When `setup.py build_ext`
compiles them, `nvcc` emits machine code for each architecture listed in
the `TORCH_CUDA_ARCH_LIST` env var (and only those); the result is a large binary containing one variant per arch. If the env var is unset, the build
defaults to whatever GPU was visible to the compiler at build time which is often
just one architecture.

On Greene, you frequently don't get to
choose which GPU you land on. `_general` accounts queue across L40S
(sm_89), H100 (sm_90), and H200 (sm_90); `_tandon_*` partitions add A100
(sm_80). If the `_C.so` was built against only sm_80 and your job lands on
an H100, every CUDA call into the extension dies with
`no kernel image is available for execution on the device`.

Setting `TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"` covers A100 / L40S+H100 / H200
in one fat binary — the build is a bit slower (three variants instead of
one) and the `.so` is a bit larger, but the resulting binary runs on every
GPU Greene routes you to.

`setup_container.sh rebuild` exports this automatically for the build step,
so a fresh rebuild on the cluster is already multi-arch. The env var only
matters when you build the C extension **outside** the rebuild wrapper —
e.g. an interactive `python setup.py build_ext --inplace --force` inside a
hand-launched singularity exec. Adding the export to your shell profile
(or sourcing it before any manual build) saves you from hitting the "no
kernel image" error after a quick fix-and-rebuild loop.