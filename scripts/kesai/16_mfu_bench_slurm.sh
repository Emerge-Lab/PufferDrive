#!/bin/bash
#SBATCH --job-name mfu_bench
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task 18
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/mfu_bench_%j.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/mfu_bench_%j.err
#SBATCH --partition dev

# Measures MFU of the real PPO loop (rollout, update, whole epoch) and SPS on this node's GPU, with and without
# policy.fused_slot_encoder, and adds hardware-counter utilization of one step if Nsight Compute is available.
# 32 envs on 18 threads, like the multinode training script (blocking workers allow 2 per hardware thread).

echo "START TIME: $(date)"
start=$(date +%s)
cd /home/bjaeger/PufferDrive || exit 1
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

export OMP_NUM_THREADS=1
source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh /home/bjaeger/PufferDrive || exit 1

RESULT_FILE=/home/bjaeger/PufferDrive/experiments/logs/mfu_bench_${SLURM_JOB_ID}_result.txt
NUM_ENVS=${NUM_ENVS:-32}      # override: NUM_ENVS=48 sbatch scripts/kesai/16_mfu_bench_slurm.sh
MINIBATCH=${MINIBATCH:-131072}
echo "=== MFU result (paste this block back) ===" | tee ${RESULT_FILE}
python scripts/mfu_bench.py --num-envs ${NUM_ENVS} --minibatch ${MINIBATCH} 2>&1 | grep -v "^MFU_RESULT" | grep "reference\|capability" | tee -a ${RESULT_FILE}
python scripts/mfu_bench.py --num-envs ${NUM_ENVS} --minibatch ${MINIBATCH} --fused 2>&1 | grep -v "^MFU_RESULT" | grep "fused" | tee -a ${RESULT_FILE}

NCU=$(command -v ncu || ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | tail -1)
if [ -n "${NCU}" ]; then
    echo "--- hardware counters (ncu range replay, one update minibatch of ${MINIBATCH} rows + one rollout forward) ---" | tee -a ${RESULT_FILE}
    METRICS=gpu__time_duration.sum,sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32.sum,dram__bytes.sum
    for variant in reference fused; do
        FLAG=""; [ ${variant} = fused ] && FLAG="--fused"
        CSV=/home/bjaeger/PufferDrive/experiments/logs/mfu_bench_${SLURM_JOB_ID}_${variant}.csv
        timeout 600 ${NCU} --replay-mode range --target-processes application-only --clock-control none --nvtx \
            --nvtx-include "update" --nvtx-include "rollout" --nvtx-include "calibrate" --print-nvtx-rename kernel \
            --metrics ${METRICS} --csv --page raw --print-units base \
            python scripts/profile_policy_step.py --rows ${MINIBATCH} ${FLAG} > ${CSV} 2>&1 \
            && { echo "${variant}:" | tee -a ${RESULT_FILE}; python scripts/profile_policy_step.py --parse ${CSV} --rows ${MINIBATCH} | tee -a ${RESULT_FILE}; } \
            || echo "${variant}: ncu failed (counters restricted or unsupported), see ${CSV}" | tee -a ${RESULT_FILE}
    done
else
    echo "Nsight Compute not found on this node; hardware-counter utilization skipped." | tee -a ${RESULT_FILE}
fi
echo "=== end of MFU result ===" | tee -a ${RESULT_FILE}

echo
echo "Result block (also in ${RESULT_FILE}):"
cat ${RESULT_FILE}
end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start)) s"
