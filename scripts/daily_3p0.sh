#!/bin/bash
# Daily 3.0 training jobs: rebuild from 3.0 and launch 3 seeds.
# Intended to be run via cron on the torch cluster login node.
#
# Crontab entry (runs at 6am daily):
#   0 6 * * * /scratch/ev2237/code/PufferDrive/scripts/daily_3p0.sh >> /scratch/ev2237/logs/daily_3p0.log 2>&1

set -euo pipefail

MAIN_REPO=/scratch/ev2237/code/PufferDrive
OVERLAY=/scratch/ev2237/images/PufferDrive/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
SAVE_DIR=/scratch/ev2237/experiments
WANDB_PROJECT=Daily3p0Runs
DATE=$(date +%Y-%m-%d)
SEEDS="42 55 1"
ACCOUNT=torch_pr_355_tandon_advanced
WORKTREE=/scratch/ev2237/daily_builds/${DATE}

echo "=== Daily 3.0 build: $DATE ==="

# Create an isolated worktree so we never touch the main repo's state
cd $MAIN_REPO
git fetch origin
if [ -d "$WORKTREE" ]; then
  echo "Worktree $WORKTREE already exists, removing stale one"
  git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
fi
git worktree add "$WORKTREE" origin/3.0 --detach

REPO=$WORKTREE
COMPUTE_CONFIG=$REPO/scripts/cluster_configs/nyu_greene.yaml
PROGRAM_CONFIG=$REPO/scripts/cluster_configs/train_base.yaml

# Replace worktree's pufferlib/resources with symlink to main repo's copy
# (map binaries are .gitignored so the worktree doesn't have them)
rm -rf "$REPO/pufferlib/resources"
ln -s "$MAIN_REPO/pufferlib/resources" "$REPO/pufferlib/resources"

# Rebuild C extension on a compute node
echo "Rebuilding C extension in worktree..."
srun --account=torch_pr_355_general --gres=gpu:1 --cpus-per-task=4 --mem=16gb --time=15 \
  singularity exec --nv --overlay ${OVERLAY}:ro $IMAGE \
  bash -c "source /ext3/env.sh && cd $REPO && python setup.py build_ext --inplace --force"

echo "Build complete. Launching jobs..."

# Activate login venv for submitit
source /scratch/ev2237/login_venv/bin/activate

# Launch 3 seeds
for seed in $SEEDS; do
  PREFIX="${DATE}-3p0-seed${seed}"

  JOB_OUTPUT=$(python $REPO/scripts/submit_cluster.py \
    --save_dir $SAVE_DIR \
    --compute_config $COMPUTE_CONFIG \
    --program_config $PROGRAM_CONFIG \
    --prefix "$PREFIX" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-group "$DATE" \
    --container \
    --container_overlay $OVERLAY \
    --account $ACCOUNT \
    --args "env.map_dir=resources/drive/binaries/carla_3D" "env.num_maps=3" "train.seed=${seed}" "train.render_interval=250" "train.checkpoint_interval=250" 2>&1)

  JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted job \K\d+')
  echo "Seed $seed: submitted job $JOB_ID"
done

# Clean up worktrees older than 7 days
find /scratch/ev2237/daily_builds -maxdepth 1 -mindepth 1 -type d -mtime +7 -exec bash -c \
  'cd '"$MAIN_REPO"' && git worktree remove --force "$1" 2>/dev/null || rm -rf "$1"' _ {} \;

echo "=== Daily 3.0 launch complete: $DATE ==="
