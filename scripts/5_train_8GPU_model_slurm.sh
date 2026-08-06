#!/bin/bash
#SBATCH --job-name train_puffer
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 2-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/PufferDrive/experiments/k_exp_0000/log_%a_%A.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/k_exp_0000/log_%a_%A.err
#SBATCH --partition dev

# print info about current job
echo "START TIME: $(date)"
start=$(date +%s)

export RUN_NAME=k_exp_0000
echo ${RUN_NAME}

# Seed each array task deterministically: 1000 * array task id
SEED=$((1000 * SLURM_ARRAY_TASK_ID))
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> train.seed=${SEED}"

# TODO could try to tune these. 1 Is probably best since Puffer parallelizes across all cores.
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source .venv/bin/activate
torchrun --standalone --nnodes=1 --nproc-per-node=8 --max_restarts=0 --start-method spawn \
    -m pufferlib.pufferl train puffer_drive \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group=emerge_ \
    train.data_dir=/home/bjaeger/PufferDrive/experiments/${RUN_NAME} \
    env.map_dir=/home/bjaeger/PufferDrive/pufferlib/resources/drive/binaries/carla \
    train.name=${RUN_NAME}_${SEED} \
    run_name=${RUN_NAME}_${SEED} \
    train.total_timesteps=100000000000 \
    vec.num_envs=16 \
    +eval.map_dir=/home/bjaeger/data/nuPlan/PufferDrive \
    train.compile=True \
    train.max_minibatch_size=131072 \
    train.minibatch_size=131072 \
    train.precision=bfloat16 \
    env.num_agents=256 \
    train.min_batch_size=524288 \
    policy.action_type=discrete \
    env.action_type=continuous \
    train.adv_filter_enabled=False \
    train.seed=${SEED} \
    tb=True

# 524288 = 256 * 16 * 128
# 131072 = 524288 / 4

end=$(date +%s)
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
