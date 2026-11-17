#!/bin/bash
#SBATCH --job-name train_opengigaflow
#SBATCH --nodes 4                        # Number of nodes requested
#SBATCH --ntasks-per-node 1              # Run 1 srun task per node (which fires up torchrun)
#SBATCH --gres gpu:8                     # GPUs per node
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --mem=0
#SBATCH --time 6-00:00
#SBATCH --output experiments/logs/train_multinode_%j.out
#SBATCH --error experiments/logs/train_multinode_%j.err
#SBATCH --partition dev
# Multi-node training on our cluster, then the evaluations: our values for scripts/slurm/train.sh, which documents the settings.
# k_scaled_0080 = opengigaflow.yaml's training settings (minibatch 393216 per rank = 12.6M rows per optimizer step on 32 ranks)
# at lr 1.5e-3. GPU memory ~21 GB + 0.22 GB per 1024-agent worker (fp16).
# Another run: sbatch --export=ALL,RUN_NAME=<name> scripts/kesai/1_train_multinode.sh
source scripts/kesai/cluster_env.sh || exit 1

SEED=1000

set -a  # every assignment up to "set +a" reaches scripts/slurm/train.sh
RUN_NAME=${RUN_NAME:-k_scaled_0080_${SEED}}
SELFPLAY_EVAL_MAP_DIR=$PD/opengigaflow/resources/carla
SELFPLAY_EVAL_WANDB_REPORT=1
WORKERS_PER_GPU=60  # self-play eval: 8 shards x 60 env workers = the node's 480 CPUs (x 5 scenarios = 300 per 9000-step episode)
# the co-sim evals need one 8-GPU node each
FOLLOW_UP_EVALS="scripts/kesai/3_eval_nuplan_reactive.sh scripts/kesai/4_eval_nuplan_nonreactive.sh scripts/kesai/6_eval_longest6_v2.sh scripts/kesai/5_eval_longest6_v1.sh scripts/kesai/7_eval_alpasim.sh"
set +a

# Higher learning rate: with 4 nodes the total minibatch size increases by 4x, reducing the total amount of optimizer steps.
exec bash scripts/slurm/train.sh \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group=emerge_ \
    train.learning_rate=0.0015 \
    train.seed="$SEED" \
    tb=True
