# Cluster training — operational guide

How to run PufferDrive training on a SLURM cluster. Written against the NYU
Greene workflow but the patterns generalize. Pairs with `scripts/setup_container.sh`,
`scripts/gpu_heartbeat.py`, and `scripts/submit_cluster.py`.

## TL;DR

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

# Per code change to C extensions: rebuild on a CPU partition (no GPU needed).
sbatch --account=<acct> --partition=cpu_short --cpus-per-task=8 --mem=16gb --time=20 \
    --chdir=$PWD -o $LOGDIR/rebuild_%j.log \
    --wrap "./scripts/setup_container.sh rebuild"

# Training: submit_cluster.py from the login node with --container --heartbeat.
python3 scripts/submit_cluster.py \
    --save_dir /scratch/$USER/runs \
    --compute_config scripts/cluster_configs/nyu_greene.yaml \
    --program_config scripts/cluster_configs/train_base.yaml \
    --container --heartbeat \
    --account <acct> --partition <gpu-partition> --time 2880 \
    --args train.checkpoint_interval=250 env.simulation_mode=gigaflow
```

## Container model

PufferDrive on Greene runs inside a singularity container. The container provides
a modern glibc + CUDA toolkit; the project's Python environment lives in a venv
on `/scratch` (not in the overlay) so installs aren't bottlenecked by fuse2fs.

The container is invoked with a **read-only** overlay mount for the miniforge3
base interpreter, plus the on-disk venv for project packages:

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

`source venv/activate` is required — sourcing `/ext3/env.sh` alone gives you
a torch-less base interpreter (it imports as a namespace-package stub with
`torch.__file__ == None`).

## Submitting training — `submit_cluster.py`

`scripts/submit_cluster.py` is the canonical submission path. It composes a
`compute_config` YAML (SLURM settings) + a `program_config` YAML (pufferl
training args) + `--args` CLI overrides, wraps the inner train command in
`singularity exec` when `--container` is set, optionally injects the GPU
heartbeat when `--heartbeat` is set, performs code isolation (symlinks the
top-level entries + hard-copies `pufferlib/` into a per-run sandbox), and
hands the package to `submitit` for `sbatch`-submission.

### Two pythons in play

A `submit_cluster.py --container` submission uses two distinct python
environments:

- **Login-side composer**: the python that runs `submit_cluster.py` itself.
  Only needs `submitit`, `pyyaml`, `cloudpickle` importable. Used purely to
  build the sbatch script and submit it to SLURM. On Greene this is
  `/usr/bin/python3` (system python) with `pip install --user submitit pyyaml
  cloudpickle` to provide those deps.
- **Compute-side executor**: the python that runs the training job on the
  compute node. This is the **venv python** inside the singularity overlay
  — same on every node because the overlay is content-identical. submitit's
  outer launcher is wrapped in `singularity exec` so it lands in this
  environment; `launch_training` then runs `torchrun` inside the same
  container.

`submit_cluster.py` handles the wrap automatically when `--container` is set
— you don't need to think about it. The only setup step is installing the
three login-side deps once.

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
| `--heartbeat` | wraps the train command in a brace group that backgrounds `python scripts/gpu_heartbeat.py` and kills it on train exit, preserving the train exit code |
| `--args key=value ...` | passes nested config keys (underscores converted to dashes) as `--key value` on the torchrun line; e.g. `env.simulation_mode=replay` becomes `--env.simulation-mode replay` |
| `--account` / `--partition` / `--time` | override `compute_config` SLURM settings |

### GPU heartbeat — required for long runs

`--heartbeat` is not optional for jobs over ~2 hours. Without it, the
cluster's idle-GPU reclaimer issues a `scancel` from `uid 0` (root) during
the first eval / checkpoint dip in GPU utilization.

`scripts/gpu_heartbeat.py` monitors `nvidia-smi` and runs short matmul bursts
when utilization drops below 65%, so the cluster always sees the GPU as
active. It cooperates with real training (steps aside when training is busy).

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

You may want to set `TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"` in your shell
profile if you build C extensions across the different GPU types on Greene
(A100 sm_80, L40S/H100 sm_89/90, H200 sm_90).

## CPU rebuild path

GPU partitions are routinely saturated by training jobs. `setup_container.sh
rebuild` doesn't actually need a GPU — it just runs `python setup.py
build_ext --inplace --force` plus a smoke import. Submit to a CPU partition
for fast turnaround:

```bash
sbatch --account=<general-account> --partition=cpu_short \
    --cpus-per-task=8 --mem=16gb --time=20 \
    --chdir=$PWD \
    -o /scratch/$USER/rebuild_logs/rebuild_%j.log \
    --wrap "./scripts/setup_container.sh rebuild"
```

`--chdir=$PWD` is required because the script uses `./scripts/`. Takes ~40s.

## Account / partition strategy

NYU Greene exposes `_general` and `_tandon_priority` account tiers, each with
their own QOS pool per partition. When `squeue` shows your job pending on
`QOSGrpGRES`, the issue is partition-level pool saturation — switching
accounts within the same tier doesn't help, but switching partitions does.

`QOSMaxGRESPerUser` is different: you're over your own concurrent-GPU cap.
Cancel a pending job or wait.

Practical recipe:

- For short jobs (rebuilds, eval, mining): try `cpu_short` first when no GPU
  is needed, else `h200_public + <general-account>`. Often the fastest GPU
  slot.
- For long training: `_tandon_priority` accounts have their own QOS pools
  separate from `_general`, so they unblock when `_general` pools are
  pinned. Race 2–3 partitions in parallel and cancel the losers as soon as
  one starts. `l40s_public` typically has multi-hour queues and is the last
  resort.

Quick test-only across combos:

```bash
for combo in \
    "<acct-priority> a100_tandon" \
    "<acct-priority> h100_tandon" \
    "<acct-general>  h200_public"; do
  read ACCT PART <<< "$combo"
  RES=$(sbatch --test-only --account=$ACCT --partition=$PART \
        --gres=gpu:1 --cpus-per-task=16 --mem=96gb --time=2880 \
        --wrap "echo test" 2>&1 | head -1)
  echo "$ACCT $PART -> $RES"
done
```

`--test-only` prints an estimated start time without actually submitting.

## Memory sizing — replay mode is heavier than gigaflow

Gigaflow training with `num_agents=1024` fits comfortably in 96 GB on Greene.
Replay-mode training on nuPlan does not — each sub-env loads its own bin file
(parsed lane graph + per-agent trajectories), so `--mem=96gb` OOMs.

Levers, in order of impact:

- `--vec.num-envs N` (drive.ini default `20`). Each vec worker is a fork; each
  worker holds copy-on-write-divergent state proportional to `num_agents/num_envs`
  + the loaded map data. Halving from 20→10 saves ~25 GB.
- Disable subsets of `[eval.*]` evaluators via CLI overrides. The 14 enabled
  evaluators in `drive.ini` all spin up their own `pufferlib.vector.make` envs
  at the first eval cycle and can collectively cost 30–50 GB at peak.
  `[eval.validation_gigaflow]` specifically renders 8 × 1080p MP4s in parallel.
- `--mem=128gb` or `--mem=192gb` if you need the eval signal in wandb.

`vec.*` keys are not in pufferl's `KEYS_OF_INTEREST` auto-merge, so a sibling
`config.yaml` next to a `load_model_path` won't override them. They come from
`drive.ini` or the CLI.

## Common pitfalls

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

## Don't chain `sleep` to wait on background jobs

A bare `sleep N` to poll on a submitted job's state is hard on the SLURM
controller and brittle. Patterns that work:

- **One-shot wait**: a single `sacct -j $JOBID --format=State -n -P` after a
  generous initial sleep tuned to expected runtime.
- **Conditional wait**: a `Monitor`-style `until` loop in a single background
  shell, with a sane upper bound.
- **Wall-clock interval**: schedule a wake-up rather than long-running `sleep`.

Hammering `squeue` in a tight loop is bad cluster citizenship — the controller
is shared across all users. Sleep at least 60 s between checks.
