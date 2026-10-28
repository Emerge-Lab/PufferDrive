#!/bin/bash
#SBATCH --job-name rollout_kernels
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 0:30:00
#SBATCH --gres gpu:1
#SBATCH --gres-flags=enforce-binding   # CPUs on the GPU's socket
#SBATCH --mem=120G
#SBATCH --cpus-per-task 18
#SBATCH --output experiments/logs/rollout_kernels_%j.out
#SBATCH --error experiments/logs/rollout_kernels_%j.err
#SBATCH --partition low

source scripts/kesai/cluster_env.sh || exit 1

# Rollout kernel table of one trainer rank in the script-16 configuration (32 workers x 1024 agents), ~3 minutes.
# Compare the fused_slot_encoder_fwd_cuda<256, 16, false> row across code versions; the cutlass 256x128 GEMM, the
# simt sgemm and the Memcpy rows are clock anchors that must not move.
# Usage: sbatch -p low -w rtx-00 scripts/kesai/20_rollout_kernel_table_slurm.sh   # BACKEND=triton NUM_ENVS=.. EXTRA_ARGS=..
echo "START TIME: $(date)"
start=$(date +%s)
cd $PD || exit 1
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
echo "node $(hostname)"; nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active --format=csv
export OMP_NUM_THREADS=1
source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh $PD || exit 1

NUM_ENVS=${NUM_ENVS:-32}
MINIBATCH=${MINIBATCH:-131072}
BACKEND=${BACKEND:-cuda}
EXTRA_ARGS=${EXTRA_ARGS:-}
RESULT_FILE=$PD/experiments/logs/rollout_kernels_${SLURM_JOB_ID}_result.txt
echo "=== rollout kernel table, fused+${BACKEND}, ${NUM_ENVS} envs, git $(git rev-parse --short HEAD) ===" | tee ${RESULT_FILE}
python -c "from pufferlib.ocean.cuda_slot_encoder import _compile; print('fwd no-store kernel on this GPU: regs/local', _compile(256, 16, False, 0)[3:5])" 2>&1 | tail -1 | tee -a ${RESULT_FILE}
env PUFFER_SLOT_ENCODER_BACKEND=${BACKEND} python scripts/mfu_bench.py --num-envs ${NUM_ENVS} --minibatch ${MINIBATCH} --fused \
    --rollout-only --kernel-table --warmup 1 --epochs 1 ${EXTRA_ARGS} 2>&1 \
    | grep -v "^MFU_RESULT" | grep "capability\|kernel table, rollout\|^ *share\|^ *[0-9][0-9.]*%\|ROLLOUT_RESULT\|^epoch\|Error" | tee -a ${RESULT_FILE}
echo "node $(hostname)"; nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active --format=csv
end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start)) s"
