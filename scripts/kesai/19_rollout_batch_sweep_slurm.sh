#!/bin/bash
#SBATCH --job-name rollout_sweep
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 2:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task 60
#SBATCH --output experiments/logs/rollout_sweep_%j.out
#SBATCH --error experiments/logs/rollout_sweep_%j.err
#SBATCH --partition low

source scripts/kesai/cluster_env.sh || exit 1

# Rollout-only timing of one training rank (60 blocking workers on 60 threads, env.num_agents per worker, float16 obs buffer)
# as a function of vec.batch_size, the number of workers one recv waits for: 30 (the "auto" default) means two buffers
# alternate between simulation and GPU forward; smaller values keep more buffers in flight at the cost of more recv calls.
# Values must divide NUM_ENVS (zero_copy). Prints the trainer's section split (env wait, copy, forward) per batch size.
# Usage: sbatch scripts/kesai/19_rollout_batch_sweep_slurm.sh   # override with NUM_ENVS=.. NUM_AGENTS=.. BATCH_SIZES=".." PUFFER_NUMA_BIND=0
# NUMA A/B, 61 processes on 64 threads in both arms: sbatch -m block:fcyclic -c 128 (binds to the GPU socket half) vs PUFFER_NUMA_BIND=0 with -m block:fcyclic -c 64 (32 per socket)
echo "START TIME: $(date)"
start=$(date +%s)
cd $PD || exit 1
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
export OMP_NUM_THREADS=1
source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh $PD || exit 1

NUM_ENVS=${NUM_ENVS:-60}
NUM_AGENTS=${NUM_AGENTS:-4096}
BATCH_SIZES=${BATCH_SIZES:-"60 30 20 15 12 10"}
RESULT_FILE=$PD/experiments/logs/rollout_sweep_${SLURM_JOB_ID}_result.txt
echo "=== rollout sweep: ${NUM_ENVS} workers x ${NUM_AGENTS} agents, float16 obs, fused+cuda encoder ===" | tee ${RESULT_FILE}
for B in ${BATCH_SIZES}; do
    echo "--- vec.batch_size=${B} ---" | tee -a ${RESULT_FILE}
    env PUFFER_SLOT_ENCODER_BACKEND=cuda python scripts/mfu_bench.py --num-envs ${NUM_ENVS} --minibatch 32768 --fused \
        --rollout-only --profile --warmup 1 --epochs 3 \
        env.num_agents=${NUM_AGENTS} train.rollout_dtype=float16 env.scenario_length=1200 vec.batch_size=${B} 2>&1 \
        | grep -E "ROLLOUT_RESULT|NUMA|^  (env|eval_copy|eval_forward|eval_misc|eval) |^epoch|Error|error" | tee -a ${RESULT_FILE}
done
echo "=== end of rollout sweep ===" | tee -a ${RESULT_FILE}
echo
cat ${RESULT_FILE}
end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start)) s"
