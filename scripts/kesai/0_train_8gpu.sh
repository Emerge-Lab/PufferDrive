#!/bin/bash
#SBATCH --job-name train_opengigaflow
#SBATCH --nodes 1                        # Number of nodes requested
#SBATCH --ntasks-per-node 1              # Run 1 srun task per node (which fires up torchrun)
#SBATCH --gres gpu:8                     # GPUs per node
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --mem=0
#SBATCH --time 6-00:00
#SBATCH --output experiments/logs/train_3seeds_%a_%A.out
#SBATCH --error experiments/logs/train_3seeds_%a_%A.err
#SBATCH --partition dev
#SBATCH --array=0
# Training on our cluster, one 8-GPU node per seed (array task = SEEDS index), then the evaluations: our values for
# scripts/slurm/train.sh, which documents the settings. k_scaled_0085 = opengigaflow.yaml's training settings on the v21 maps.
source scripts/kesai/cluster_env.sh || exit 1

SEEDS=(1000 2000 3000 4000)
RUN_PREFIX=k_scaled_0085
TRAIN_MAP_DIR=$PD/opengigaflow/resources/carla_128_affine_aug_v21
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
[ -n "${SEED}" ] || { echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} has no run configured; aborting."; exit 1; }
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} -> train.seed=${SEED} maps=${TRAIN_MAP_DIR}"

set -a  # every assignment up to "set +a" reaches scripts/slurm/train.sh
RUN_NAME=${RUN_PREFIX}_${SEED}
SELFPLAY_EVAL_MAP_DIR=$PD/opengigaflow/resources/carla_v21  # the 8 towns: the benchmark reads only the folder's first 8 sorted maps
COSIM_TOWN_BIN_DIR=$SELFPLAY_EVAL_MAP_DIR  # the CARLA co-sim evals below drive on the same towns
SELFPLAY_EVAL_WANDB_REPORT=1
WORKERS_PER_GPU=60  # self-play eval: 8 shards x 60 env workers = the node's 480 CPUs (x 5 scenarios = 300 per 9000-step episode)
# the co-sim evals need one 8-GPU node each
FOLLOW_UP_EVALS="scripts/kesai/3_eval_nuplan_reactive.sh scripts/kesai/4_eval_nuplan_nonreactive.sh scripts/kesai/6_eval_longest6_v2.sh scripts/kesai/5_eval_longest6_v1.sh scripts/kesai/7_eval_alpasim.sh"
set +a
for map_dir in "$TRAIN_MAP_DIR" "$SELFPLAY_EVAL_MAP_DIR"; do
    [ -d "$map_dir" ] || { echo "missing map dir $map_dir (git-ignored, copy it to the cluster)"; exit 1; }
done

exec bash scripts/slurm/train.sh \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group=emerge_ \
    env.map_dir="$TRAIN_MAP_DIR" \
    train.seed="$SEED" \
    tb=True
