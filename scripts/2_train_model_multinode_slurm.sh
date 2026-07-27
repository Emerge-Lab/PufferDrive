#!/bin/bash
#SBATCH --job-name train_puffer
#SBATCH --nodes 8                        # Number of nodes requested
#SBATCH --ntasks-per-node 1              # Run 1 srun task per node (which fires up torchrun)
#SBATCH --gres gpu:8                     # GPUs per node
#SBATCH --cpus-per-task 144
#SBATCH --mem=1007G
#SBATCH --time 3-00:00
#SBATCH --output /home/bjaeger/PufferDrive/experiments/k_scaled_0003/log_%a_%A.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/k_scaled_0003/log_%a_%A.err
#SBATCH --partition dev

# Set up PyTorch Distributed Rendezvous parameters from Slurm variables
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

echo "START TIME: $(date)"
echo "Master node: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Total nodes: ${WORLD_SIZE}"
start=$(date +%s)

export RUN_NAME=k_scaled_0003
echo ${RUN_NAME}

# Thread limit limits CPU thrashing across worker environments
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source .venv/bin/activate

# Execute torchrun across all nodes using srun
srun torchrun \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --nproc-per-node=8 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --max_restarts=0 \
    --start-method spawn \
    -m pufferlib.pufferl train puffer_drive \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group=emerge_ \
    train.data_dir=/home/bjaeger/PufferDrive/experiments/${RUN_NAME} \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla \
    train.name=${RUN_NAME} \
    train.total_timesteps=100000000000 \
    train.learning_rate=0.01 \
    vec.num_envs=16 \
    +eval.map_dir=/home/bjaeger/data/nuPlan/PufferDrive \
    train.compile=True \
    train.max_minibatch_size=131072 \
    train.minibatch_size=131072 \
    train.precision=bfloat16 \
    eval.validation_gigaflow.render_backend=obs_html \
    train.seed=0 \
    tb=True

end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
