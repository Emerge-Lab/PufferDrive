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

source scripts/kesai/cluster_env.sh || exit 1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_JOB_NUM_NODES

printf 'START TIME: %s\nMaster node: %s\nTotal nodes: %s\n' "$(date)" "${MASTER_ADDR}:${MASTER_PORT}" "${WORLD_SIZE}"
start=$(date +%s)

export SEED=1000

export RUN_NAME=${RUN_NAME:-k_scaled_0080_${SEED}}
echo ${RUN_NAME}

export DATA_DIR=$PD/experiments/${RUN_NAME}
echo ${DATA_DIR}

# passed to train as train.final_model_name, so the evals below find the final model without knowing its epoch
export FINAL_MODEL_NAME=final_model.pt
export MODEL_PATH=${DATA_DIR}/${FINAL_MODEL_NAME}
echo ${MODEL_PATH}

# one BLAS/OpenMP thread per process: the env workers would otherwise thrash the CPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh $PD || exit 1
# k_scaled_0080: the opengigaflow.yaml defaults with 128 env workers and minibatch 393216 per rank (12.6M rows per optimizer
# step on 32 ranks) at lr 1.5e-3 for 1e12 steps; the other overrides are run identity, logging and hardware. GPU memory ~21 GB + 0.22 GB per
# 1024-agent worker (fp16). Another run: sbatch --export=ALL,RUN_NAME=<name> scripts/kesai/3_train_64GPU_multinode_slurm.sh
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" torchrun \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --nproc-per-node=8 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --max_restarts=0 \
    --start-method spawn \
    -m opengigaflow.rl train \
    wandb=True \
    wandb_project=nightly-multi-long \
    wandb_group=emerge_ \
    train.data_dir=${DATA_DIR} \
    run_name=${RUN_NAME} \
    train.total_timesteps=1000000000000 \
    vec.num_envs=128 \
    train.max_minibatch_size=393216 \
    train.minibatch_size=393216 \
    train.learning_rate=0.0015 \
    train.final_model_name=${FINAL_MODEL_NAME} \
    train.seed=${SEED} \
    tb=True

[ -f "${MODEL_PATH}" ] || { echo "Training did not produce ${MODEL_PATH}; skipping evaluation."; exit 1; }

# the co-sim evals need one 8-GPU node each: submitted before the long self-play eval below, so they queue while it runs
for EVAL_SCRIPT in 9_nuPlan.sh 13_nuPlan_nonreactive.sh 11_carla_longest6_v2.sh 14_carla_longest6_v1.sh 12_alpasim.sh; do
    echo "Submitting scripts/kesai/${EVAL_SCRIPT} for ${MODEL_PATH}"
    RUN_DIR=${DATA_DIR} sbatch scripts/kesai/${EVAL_SCRIPT} \
        || echo "submission failed; run by hand: RUN_DIR=${DATA_DIR} sbatch scripts/kesai/${EVAL_SCRIPT}"
done

# 8 srun shards per node, each one GPU, an eighth of the CPUs and 60 env workers (x 5 scenarios = 300 per 9000-step episode)
echo "Follow-up evals submitted, evaluating ${MODEL_PATH}"
.venv/bin/python scripts/parallel_eval.py carla \
    --total-scenarios 80000 \
    --num-nodes 4 \
    --shards-per-node 8 \
    env.map_dir=$PD/opengigaflow/resources/carla \
    vec.num_envs=60 \
    eval.render_filter=all_infractions \
    eval.capture_observations=true \
    eval.output_name=${RUN_NAME} \
    load_model_path=${MODEL_PATH} \
    wandb=True

end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start))"
