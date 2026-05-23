#!/bin/bash
# Single-agent "speed run" training -- nightly 3-seed sweep.
#
# One controlled agent per environment on a single CARLA map (Town02), with many
# parallel environments, and goals sampled at random drivable points anywhere on
# the map (goal_placement=1) -- so the goal can be far away and in any direction,
# including behind the agent. On reaching a goal a new random one is drawn, so the
# agent does continuous point-to-point runs across the town.
#
# Launches 3 seeds as a SLURM job array, all logged to wandb group "Nightly Test".
#
# Launch:
#   sbatch scripts/single_agent_speed_run.sbatch
#
# Override scheduling / scale / seeds without editing this file:
#   sbatch --partition=a100_tandon --account=torch_pr_924_tandon_advanced scripts/single_agent_speed_run.sbatch
#   NUM_AGENTS=2048 sbatch scripts/single_agent_speed_run.sbatch
#   SEED_BASE=100 sbatch scripts/single_agent_speed_run.sbatch   # seeds 100,101,102
#
#SBATCH --job-name=single_agent_speed_run
#SBATCH --account=torch_pr_924_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=12:00:00

set -euo pipefail

# Tunables (override via the environment, e.g. NUM_AGENTS=2048 sbatch ...).
NUM_AGENTS="${NUM_AGENTS:-1024}"                # number of single-agent environments
MAX_GOAL_POSITION="${MAX_GOAL_POSITION:-1000}"  # obs goal-vector normalization (m); cover Town02's extent
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000000}"  # cap the run at 1B steps
WANDB_GROUP="${WANDB_GROUP:-Nightly Test}"
SEED_BASE="${SEED_BASE:-0}"
SEED=$((SEED_BASE + SLURM_ARRAY_TASK_ID))      # one seed per array task: SEED_BASE .. SEED_BASE+2
LAUNCH_DATE="$(date +%Y-%m-%d)"                 # stamp the wandb run name with the launch date
WANDB_NAME="${LAUNCH_DATE}_single_agent_seed${SEED}"

# Disable the nuPlan-based evaluators (replay + behavior buckets); they are
# irrelevant to single-map CARLA training and spawn their own envs. validation_gigaflow
# (CARLA) stays on. enabled 0 -> bool(0)=False in the eval manager. Note: dotted CLI
# flags use hyphens (the argparser maps _ -> -), so section names are hyphenated here.
DISABLE_EVALS=""
for ev in validation-replay behaviors-full-dir behaviors-hard-stop behaviors-highway-straight \
          behaviors-lane-change behaviors-merge behaviors-parked-cars behaviors-roundabout \
          behaviors-stopped-traffic behaviors-traffic-light-green behaviors-traffic-light-stop \
          behaviors-unprotected-left behaviors-unprotected-right; do
    DISABLE_EVALS="$DISABLE_EVALS --eval.$ev.enabled 0"
done

OVERLAY="/scratch/$USER/images/PufferDrive/overlay-15GB-500K.ext3"
IMAGE="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"

singularity exec --nv --overlay "${OVERLAY}:ro" "$IMAGE" bash -c '
    source /scratch/'"$USER"'/venvs/pufferdrive/bin/activate
    export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
    export PYTHONNOUSERSITE=1
    export WANDB_NAME='"$WANDB_NAME"'  # wandb run name (no name= in WandbLogger -> env var wins)
    NCCL_DIR=$(compgen -G "/scratch/'"$USER"'/venvs/pufferdrive/lib/python3.12/site-packages/nvidia/nccl/lib" 2>/dev/null) \
        && export LD_LIBRARY_PATH="$NCCL_DIR:$LD_LIBRARY_PATH"
    cd /scratch/'"$USER"'/code/PufferDrive

    # Single-map dir holding only Town02, derived from the bundled carla maps.
    # Atomic create (temp + mv) so concurrent array tasks never read a partial file.
    DEST=pufferlib/resources/drive/binaries/carla_town02/opendrive__Town02.bin
    if [ ! -e "$DEST" ]; then
        mkdir -p pufferlib/resources/drive/binaries/carla_town02
        TMP=$(mktemp pufferlib/resources/drive/binaries/carla_town02/Town02.XXXXXX)
        cp pufferlib/resources/drive/binaries/carla/opendrive__Town02.bin "$TMP" && mv -f "$TMP" "$DEST"
    fi

    python scripts/gpu_heartbeat.py > /tmp/gpu_heartbeat_'"$SLURM_JOB_ID"'.log 2>&1 &
    HEARTBEAT_PID=$!

    python -m pufferlib.pufferl train puffer_drive \
        --env.map-dir pufferlib/resources/drive/binaries/carla_town02 \
        --env.num-maps 1 \
        --env.max-agents-per-env 1 \
        --env.num-agents '"$NUM_AGENTS"' \
        --env.use-map-cache 1 \
        --env.traffic-light-behavior 0 \
        --env.max-traffic-control-observations 0 \
        --env.goal-placement 1 \
        --env.num-target-waypoints 1 \
        --env.min-waypoint-spacing 0 \
        --env.max-goal-position '"$MAX_GOAL_POSITION"' \
        --train.total-timesteps '"$TOTAL_TIMESTEPS"' \
        --train.seed '"$SEED"' '"$DISABLE_EVALS"' \
        --wandb --wandb-project puffer_drive --wandb-group "'"$WANDB_GROUP"'" --tag seed'"$SEED"'
    TRAIN_EXIT=$?

    kill $HEARTBEAT_PID 2>/dev/null || true
    exit $TRAIN_EXIT
'
