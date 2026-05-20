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
#   (b) install submitit on the login-node system python (see "Why" below)
python3 -m ensurepip --user
python3 -m pip install --user submitit pyyaml cloudpickle

# Per code change to C extensions: rebuild on a CPU partition (no GPU needed).
sbatch --account=<acct> --partition=cpu_short --cpus-per-task=8 --mem=16gb --time=20 \
    --chdir=$PWD -o $LOGDIR/rebuild_%j.log \
    --wrap "./scripts/setup_container.sh rebuild"

# Training: submit_cluster.py from the login node (NOT inside singularity)
# with --container --heartbeat. Heartbeat is required for runs > ~2h.
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

`source venv/activate` is **required** — sourcing `/ext3/env.sh` alone gives you
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

### Why submitit needs the system python

`submitit` serializes the launch function via `cloudpickle` and writes an
sbatch script that, on the compute node, runs

```
srun <python-path> -m submitit.<launcher> <pkl>
```

`<python-path>` is `sys.executable` of the python that ran
`submit_cluster.py`. That python must:

1. Have `submitit` importable.
2. Be invocable from the compute node *outside* singularity (because the
   `srun` wrapper itself isn't inside the container — only the inner train
   command is).

The venv python on `/scratch/$USER/venvs/pufferdrive/bin/python` does **not**
qualify: it's a symlink to `/ext3/miniforge3/bin/python3`, which only exists
inside the singularity overlay. On the compute node `srun` tries to invoke
that path outside the container and fails with
`execve(): /scratch/.../python: No such file or directory`.

The system `/usr/bin/python3` does qualify: it's on every node, no overlay
symlinks, and the `~/.local` user site is on a shared filesystem so packages
installed via `pip install --user` are visible from compute nodes.

### One-time setup of submitit on system python

```bash
# Greene's /usr/bin/python3 is stripped of pip. Bootstrap with ensurepip:
python3 -m ensurepip --user
python3 -m pip install --user --upgrade pip
python3 -m pip install --user submitit pyyaml cloudpickle
```

`submitit` is pure-python and the deps are too, so `--user` install works
without needing a compiler. After this, `python3 -c 'import submitit'` works
on the login node and all compute nodes.

### Run submit_cluster.py from the *login node*, not from inside the container

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
| `--container` | wraps the inner train command in `singularity exec --nv --overlay $OVERLAY:ro $IMAGE_PATH ...` and prepends `source $VENV/bin/activate && export PYTHONNOUSERSITE=1` |
| `--heartbeat` | wraps the inner train command in a brace group that backgrounds `python scripts/gpu_heartbeat.py` and kills it on train exit, preserving the train exit code |
| `--args key=value key2=value2 ...` | passes nested config keys (underscores converted to dashes) as `--key value` on the torchrun line; e.g. `env.simulation_mode=replay` becomes `--env.simulation-mode replay` |
| `--account` / `--partition` / `--time` | override `compute_config` SLURM settings |

`AutoExecutor` (inside submit_cluster.py) probes for `sbatch` on `$PATH`. The
login-node `$PATH` includes `/opt/slurm/bin`, so submitit picks
`SlurmExecutor` automatically — no `cluster="slurm"` hint needed.

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

### Fallback: direct sbatch (if submitit setup is skipped)

Sometimes you can't or don't want to install submitit on the system python
(restricted environment, fast smoke test, etc.). A direct sbatch with the
same singularity-exec + heartbeat pattern is fine. The translation from
`submit_cluster.py --container --heartbeat` to a hand-written script is
straightforward:

```bash
#!/bin/bash
#SBATCH --job-name=mytrain
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1 --cpus-per-task=16 --mem=96gb --time=2880
#SBATCH -o /scratch/$USER/runs/logs/train_%j.log

singularity exec --nv \
  --overlay /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  bash -c "
    source /scratch/$USER/venvs/pufferdrive/bin/activate
    export PYTHONNOUSERSITE=1
    export TORCH_CUDA_ARCH_LIST=\"8.0;8.9;9.0\"
    export XDG_CACHE_HOME=/scratch/$USER/cache
    cd /scratch/$USER/code/PufferDrive
    python scripts/gpu_heartbeat.py > /tmp/gpu_heartbeat.log 2>&1 &
    HB_PID=\$!
    torchrun --standalone --nproc_per_node 1 -m pufferlib.pufferl train puffer_drive \
        --train.total-timesteps 10000000000 \
        --train.checkpoint-interval 250 \
        --wandb --wandb-project pufferdrive \
        --train.data-dir /scratch/$USER/runs/mytrain
    TRAIN_EXIT=\$?
    kill \$HB_PID 2>/dev/null
    exit \$TRAIN_EXIT
"
```

This skips submit_cluster.py's code isolation and YAML composition but gets
the job running. Prefer `submit_cluster.py` once the one-time submitit
install is done.

## CPU rebuild path

GPU partitions are routinely saturated by training jobs of this same project.
`setup_container.sh rebuild` doesn't actually need a GPU — it just runs
`python setup.py build_ext --inplace --force` plus a smoke import. Submit to a
CPU partition for fast turnaround:

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
`QOSGrpGRES`, the issue is partition-level pool saturation — **switching
accounts within the same tier doesn't help**, but switching partitions does.

`QOSMaxGRESPerUser` is different: you're over your own concurrent-GPU cap.
Cancel a pending job or wait.

Practical recipe for long training:

- For short jobs (rebuilds, eval, mining): try `cpu_short` if CPU-only; else
  `h200_public + *_general`. Often the fastest GPU slot.
- For long training: race 2–3 GPU partitions in parallel and cancel the
  losers as soon as one starts. `tandon_priority` accounts often unblock when
  `_general` pools are pinned. `l40s_public` typically has multi-hour queues
  and is the last resort.

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

`vec.*` keys are **not** in pufferl's `KEYS_OF_INTEREST` auto-merge, so a
sibling `config.yaml` next to a `load_model_path` won't override them. They
come from `drive.ini` or the CLI.

## Submission pitfalls to avoid

A few mistakes that look reasonable but break the submission flow:

- **Don't run `submit_cluster.py` from inside the container.** It works at the
  AutoExecutor level (sbatch is reachable; the submission goes through), but
  the submitted job inherits the in-container venv python as `sys.executable`.
  On the compute node `srun` tries to invoke that path *outside* singularity
  and fails with `execve(): /scratch/.../python: No such file or directory`.
  submit_cluster.py wraps the *inner* train command in singularity-exec but
  the *outer* submitit launcher is not wrapped.

  The fix is the layout described above: install submitit on the system
  `/usr/bin/python3` via `pip install --user`, run `submit_cluster.py` from
  the login node directly (no container, no venv activate).

- **Don't `pip install submitit` into the venv expecting it to work from the
  login node.** The venv's `pip` and `python` shebangs point at
  `/ext3/miniforge3/bin/python3` (overlay-internal). Running them outside the
  container errors with "required file not found". The venv is *runtime*
  only — its packages are invisible to login-node tooling.

- **Don't bind `/opt/slurm` + `/run/munge` + `/etc/passwd` into the container
  as a workaround.** It does make `sbatch` callable from inside the container
  (you'll see "slurm 25.05.4" if you run `sbatch --version`), but you're then
  back to pitfall #1: the submitted job's outer python is still the venv
  python. The bindings buy you the submission but not the execution.

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
- **Squash-merging stacked PRs** can hit "stale info" on `--force-with-lease`
  when the token URL differs from `origin`. Either fetch first or use
  `--force` with care.

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
