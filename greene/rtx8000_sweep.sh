#!/bin/bash
#SBATCH --array=0-6
#SBATCH --job-name=pd_sweep_rtx8000
#SBATCH --output=/scratch/%u/PufferDrive/logs/output_%A_%a.txt
#SBATCH --error=/scratch/%u/PufferDrive/logs/error_%A_%a.txt
#SBATCH --mem=48G
#SBATCH --time=0-24:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --account=pr_100_tandon_priority

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "User: $USER"
echo "Running on node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
date

# Set project directory
PROJECT_DIR="/scratch/${USER}/PufferDrive"
echo "Project directory: $PROJECT_DIR"

# Create logs directory if it doesn't exist
mkdir -p ${PROJECT_DIR}/logs

# Navigate to project directory
cd ${PROJECT_DIR}

# Check if directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory does not exist: $PROJECT_DIR"
    exit 1
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "ERROR: Virtual environment not found at ${PROJECT_DIR}/.venv"
    exit 1
fi

# RTX8000 optimized settings
NUM_WORKERS=32
NUM_ENVS=32
VEC_BATCH_SIZE=4
NUM_AGENTS=1024 
BPTT_HORIZON=32 
HUMAN_REG_COEF=0.0
USE_GUIDANCE_REWARDS=1

# Calculate the segment size (minimum train batch size)
SEGMENT_SIZE=$((NUM_AGENTS * NUM_WORKERS * BPTT_HORIZON))

# Target batch size (aim for 1-2M on H100)
TARGET_BATCH_SIZE=2097152  # 2M = 2^21

# Calculate train batch size: largest multiple of SEGMENT_SIZE that doesn't exceed target
TRAIN_BATCH_SIZE=$(( (TARGET_BATCH_SIZE / SEGMENT_SIZE) * SEGMENT_SIZE ))

# If calculated batch size is 0 or too small, use the segment size
if [ $TRAIN_BATCH_SIZE -lt $SEGMENT_SIZE ]; then
    TRAIN_BATCH_SIZE=$SEGMENT_SIZE
fi

echo "Starting sweep with settings:"
echo "  NUM_WORKERS: $NUM_WORKERS"
echo "  NUM_ENVS: $NUM_ENVS"
echo "  VEC_BATCH_SIZE: $VEC_BATCH_SIZE"
echo "  NUM_AGENTS: $NUM_AGENTS"
echo "  BPTT_HORIZON: $BPTT_HORIZON"
echo "  SEGMENT_SIZE (min required): $SEGMENT_SIZE"
echo "  TRAIN_BATCH_SIZE (calculated): $TRAIN_BATCH_SIZE"
echo "  Divisibility check: $TRAIN_BATCH_SIZE / $SEGMENT_SIZE = $(($TRAIN_BATCH_SIZE / $SEGMENT_SIZE))"

# Launch sweep with optimized parameters
puffer sweep puffer_drive \
  --wandb \
  --wandb-project "gsp" \
  --tag "guidance_rewards" \
  --vec.num-workers $NUM_WORKERS \
  --vec.num-envs $NUM_ENVS \
  --vec.batch-size $VEC_BATCH_SIZE \
  --env.num-agents $NUM_AGENTS \
  --env.use-guided-autonomy $USE_GUIDANCE_REWARDS \
  --train.batch-size $TRAIN_BATCH_SIZE \
  --train.bptt-horizon $BPTT_HORIZON \

# Print completion info
echo "Sweep completed"
date