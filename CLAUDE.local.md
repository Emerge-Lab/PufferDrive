# NYU Torch Cluster Setup

## SSH Access
- Host: `torch` (login.torch.hpc.nyu.edu)
- User: `ev2237`
- SSH multiplexing enabled (ControlMaster auto, ControlPersist 600)
- Must authenticate manually first, then multiplexed connections work from Claude

## Paths on Cluster
- **Code**: `/scratch/ev2237/code/PufferDrive`
- **Experiments**: `/scratch/ev2237/experiments/`
- **Singularity overlay**: `/scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3`
- **SIF image**: `/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif`

## Singularity Container
All training runs use Singularity with the overlay (contains miniforge3, Python 3.12, torch 2.10.0+cu128, xvfb, ffmpeg).

**Inner script** (`/scratch/ev2237/train_singularity.sh`) — accepts all training args via `"$@"`:
```bash
#!/bin/bash
set -e
SCRATCH_DIR="${SCRATCH_DIR:-/scratch/ev2237}"
CODE_DIR="${SCRATCH_DIR}/code/PufferDrive"
source /ext3/env.sh
cd "$CODE_DIR"
export XDG_CACHE_HOME="${SCRATCH_DIR}/cache"
export WANDB_CACHE_DIR="${SCRATCH_DIR}/wandb_cache"
export WANDB_CONFIG_DIR="${SCRATCH_DIR}/wandb_config"
export WANDB_DATA_DIR="${SCRATCH_DIR}/wandb_data"
export WANDB_DIR="${SCRATCH_DIR}/wandb_data"
mkdir -p "$XDG_CACHE_HOME"
torchrun --standalone --nproc_per_node 1 -m pufferlib.pufferl train puffer_drive "$@"
```

**Helper variables** (used across all launch templates):
```bash
SING="singularity exec --nv --overlay /scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
SCRIPT="/scratch/ev2237/train_singularity.sh"
ACCOUNTS="torch_pr_355_general torch_pr_355_tandon_advanced torch_pr_104_tandon_advanced torch_pr_102_tandon_advanced torch_pr_45_tandon_advanced"
```

## SLURM Accounts
Use multi-account shotgun approach (submit to all, cancel losers):
- `torch_pr_355_general`, `torch_pr_355_tandon_advanced`, `torch_pr_355_tandon_priority`
- `torch_pr_104_tandon_advanced`, `torch_pr_104_general`
- `torch_pr_102_tandon_advanced`, `torch_pr_102_general`
- `torch_pr_45_tandon_advanced`, `torch_pr_45_general`

No need to specify partitions explicitly.

## Quick Test Launch (All Eval Types)
To test render, safe eval metrics, WOSAC, and human replay eval with fast intervals:
```bash
SING="singularity exec --nv --overlay /scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
SCRIPT="/scratch/ev2237/train_singularity.sh"
COMMON="--gres=gpu:1 --cpus-per-task=4 --mem=32gb --time=30 --export=ALL"
JOB_NAME=test_eval

for ACCT in torch_pr_355_general torch_pr_355_tandon_advanced torch_pr_104_tandon_advanced torch_pr_102_tandon_advanced torch_pr_45_tandon_advanced; do
sbatch --account=$ACCT $COMMON --job-name=$JOB_NAME \
  --output=/scratch/ev2237/experiments/$JOB_NAME-%j.out \
  --error=/scratch/ev2237/experiments/$JOB_NAME-%j.out \
  --wrap "$SING bash $SCRIPT \
    --vec.num-workers 4 --vec.num-envs 4 --env.num-agents 64 \
    --env.map-dir resources/drive/binaries/training --env.num-maps 100 \
    --train.total-timesteps 2000000000 --train.checkpoint-interval 3 \
    --train.render True --train.render-interval 3 \
    --safe-eval.enabled True --safe-eval.interval 3 --safe-eval.num-episodes 10 --safe-eval.num-agents 16 \
    --eval.eval-async False --eval.wosac-realism-eval True --eval.human-replay-eval True --eval.eval-interval 3 \
    --wandb --wandb-project pufferdrive --wandb-group $JOB_NAME \
    --train.data-dir /scratch/ev2237/experiments/$JOB_NAME"
done
# Then cancel all but one running job
```
- **Evals trigger at different epochs**: render/safe-eval at epoch 3 (interval=3), WOSAC/human-replay at epoch 4 (`(epoch-1) % 3 == 0`)
- **32GB RAM required**: Training + eval subprocess can OOM with 16GB
- **Safe eval takes ~2-3 min**: 1000-step episodes with 16 agents

## Full Training Launch (Production)
```bash
SING="singularity exec --nv --overlay /scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
SCRIPT="/scratch/ev2237/train_singularity.sh"
COMMON="--gres=gpu:1 --cpus-per-task=16 --mem=32gb --time=360 --export=ALL"

for ACCT in torch_pr_355_general torch_pr_355_tandon_advanced torch_pr_104_tandon_advanced torch_pr_102_tandon_advanced torch_pr_45_tandon_advanced; do
sbatch --account=$ACCT $COMMON --job-name=train_waymo \
  --output=/scratch/ev2237/experiments/train_waymo-%j.out \
  --error=/scratch/ev2237/experiments/train_waymo-%j.out \
  --wrap "$SING bash $SCRIPT \
    --vec.num-workers 16 --vec.num-envs 16 --env.num-agents 1024 \
    --env.map-dir resources/drive/binaries/training --env.num-maps 10000 \
    --train.total-timesteps 2000000000 --train.checkpoint-interval 250 \
    --train.render True --train.render-interval 250 \
    --safe-eval.enabled True --safe-eval.interval 250 \
    --eval.eval-async False \
    --wandb --wandb-project pufferdrive --wandb-group cluster \
    --train.data-dir /scratch/ev2237/experiments/train_waymo"
done
# Then cancel all but one running job
```

## Key Gotchas
- **Overlay `:ro` for training, `:rw` + `--fakeroot` for installs**: The overlay must be `:rw` with `--fakeroot` when installing packages. Use `:ro` for running training (allows concurrent jobs).
- **C extension must be built inside Singularity**: Run `python setup.py build_ext --inplace --force` inside the container. The `.so` goes to the source tree on `/scratch`, not the overlay.
- **TORCH_CUDA_ARCH_LIST for multi-GPU support**: Set `export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"` before building the C extension to support A100 (sm_80), L40S (sm_89), and H200 (sm_90). Without this, it auto-detects only the current GPU.
- **CUDA_HOME must be set**: `export CUDA_HOME=/usr/local/cuda` before building extensions.
- **checkpoint-interval must be <= safe-eval interval**: Safe eval needs a saved checkpoint to export the model. If checkpoint-interval > safe-eval interval, safe eval silently skips.
- **Empty --exclude causes sbatch error**: Don't pass `--exclude=""` to sbatch.
- **render=True triggers ensure_drive_binary()**: This runs `bash scripts/build_ocean.sh visualize fast` at init, requires bash on PATH.
- **xvfb-run needed for headless rendering**: The `./visualize` binary uses Raylib and needs a display. Container provides `/usr/bin/xvfb-run`.
- **device config returns int not str**: `config.get("device", "cuda")` returns an int (GPU index) during training. Must wrap with `str()` in subprocess command building.
- **XDG_CACHE_HOME must be on scratch**: Mesa shader cache and other tools write to `~/.cache`, which hits home disk quota. `train_singularity.sh` sets `XDG_CACHE_HOME` to scratch automatically.
- **Visualize binary uses llvmpipe (CPU software renderer)**: On the cluster, the visualize binary falls back to CPU rendering. Large CARLA maps (3500x3500px) can timeout at 600s. Waymo maps are smaller and render fine.
- **10x envs (160 envs) OOM at 32GB**: Each env_init loads a map and builds grid_map/neighbor_cache (~115ms, significant memory). 160 envs × 16 sub-envs = 2560 env_inits. Needs 48-64GB or fewer envs.
