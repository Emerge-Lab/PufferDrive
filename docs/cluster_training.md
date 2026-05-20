# Cluster training — operational guide

How to run PufferDrive training on a SLURM cluster. Written against the NYU
Greene workflow but the patterns generalize. Pairs with `scripts/setup_container.sh`,
`scripts/gpu_heartbeat.py`, and `scripts/submit_cluster.py`.

## TL;DR

```bash
# One-time per cluster: create the singularity overlay and install deps.
./scripts/setup_container.sh create-overlay
sbatch --account=<acct> --gres=gpu:1 --cpus-per-task=8 --mem=32gb --time=60 \
    --wrap "./scripts/setup_container.sh install"

# Per code change to C extensions: rebuild on a CPU partition (no GPU needed).
sbatch --account=<acct> --partition=cpu_short --cpus-per-task=8 --mem=16gb --time=20 \
    --chdir=$PWD -o $LOGDIR/rebuild_%j.log \
    --wrap "./scripts/setup_container.sh rebuild"

# Training: direct sbatch with inline singularity-exec + heartbeat.
#   (`submit_cluster.py` has known limitations on this branch lineage —
#   see "submit_cluster.py" below.)
sbatch /path/to/my_train.sh   # template in this doc
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

## Training sbatch template

The minimal template below uses a direct `sbatch` (no `submit_cluster.py`),
includes the GPU heartbeat to prevent idle-reclamation, and wraps everything in
a singularity-exec. Adapt the `--account`, `--partition`, paths, and CLI args:

```bash
#!/bin/bash
#SBATCH --job-name=mytrain
#SBATCH --account=<your-account>
#SBATCH --partition=<gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=2880          # 48h
#SBATCH -o /scratch/$USER/runs/logs/train_%j.log

singularity exec --nv \
  --overlay /scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3:ro \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  bash -c "
    source /scratch/$USER/venvs/pufferdrive/bin/activate
    export PYTHONNOUSERSITE=1
    export TORCH_CUDA_ARCH_LIST=\"8.0;8.9;9.0\"
    export XDG_CACHE_HOME=/scratch/$USER/cache
    export WANDB_DIR=/scratch/$USER/wandb_data
    cd /scratch/$USER/code/PufferDrive

    # GPU heartbeat: keeps utilization above 65% during eval/checkpoint dips
    # so the cluster's idle-GPU reclaimer doesn't kill the job (root scancel
    # at ~2h is the symptom).
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

`TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"` covers A100 (sm_80), L40S/H100 (sm_89/90),
and H200 (sm_90). Without it the C extension is compiled only for the build
host's GPU type and crashes on different hardware with `no kernel image is available`.

## GPU heartbeat — required for long runs

Without `scripts/gpu_heartbeat.py` backgrounded alongside training, jobs lasting
~2 hours risk **CANCELLED by uid 0** from the cluster's idle-GPU reclaimer.
Eval / checkpoint / map-load phases dip GPU utilization briefly, and the
reclaimer interprets those dips as "idle".

The heartbeat monitors `nvidia-smi` and runs short matmul bursts when
utilization drops below 65%, so the cluster always sees the GPU as active.
It cooperates with real training (steps aside when training is active).

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

## submit_cluster.py — known limitations

`scripts/submit_cluster.py` wraps the training launch in submitit + a heartbeat
wrapper. On the `emerge/temp_training`-derived branch lineage it doesn't work
end-to-end:

1. Login-node `/usr/bin/python3` lacks `pip` → can't `pip install submitit`
   on the login node. The venv's `pip` shebang points at
   `/ext3/miniforge3/bin/python3` (overlay-internal) so `pip install` outside
   the container errors with "required file not found".
2. Running `submit_cluster.py` *inside* the container makes submitit's `srun`
   launcher inherit the venv python path (`/scratch/.../venvs/.../python`).
   On the compute node `srun` tries to invoke that path *outside* singularity
   and fails with `execve(): No such file or directory`. submit_cluster.py
   wraps the *inner* train command in singularity-exec but the *outer* launcher
   is not wrapped.

Workaround if you really want submitit + sbatch: bind the slurm dirs into the
container so the in-container python can see sbatch and call it directly:

```bash
singularity exec --nv \
    --bind /opt/slurm:/opt/slurm \
    --bind /run/munge:/run/munge \
    --bind /etc/passwd:/etc/passwd \
    --bind /etc/group:/etc/group \
    --overlay overlay.ext3:ro \
    $SIF bash -c 'PATH=/opt/slurm/bin:$PATH ...submit_cluster.py...'
```

This gets the submission through (real SLURM job ID), but the **submitted job
itself** still hits (2) above unless you also bind those dirs into the launched
container, which submit_cluster.py doesn't do.

**Recommended**: use the direct-sbatch template from this doc. The heartbeat
is a 4-line bash addition; you don't need submitit for that.

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
