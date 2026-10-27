#!/bin/bash
#SBATCH --job-name mfu_bench
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task 18
#SBATCH --output experiments/logs/mfu_bench_%j.out
#SBATCH --error experiments/logs/mfu_bench_%j.err
#SBATCH --partition dev

source scripts/kesai/cluster_env.sh || exit 1

# Measures MFU of the real PPO loop (rollout, update, whole epoch) and SPS on this node's GPU: reference encoders,
# policy.fused_slot_encoder with the Triton kernels, and the same with the CUDA kernels (PUFFER_SLOT_ENCODER_BACKEND=cuda),
# and adds hardware-counter utilization of one step if Nsight Compute is available.
# 32 envs on 18 threads, like the multinode training script (blocking workers allow 2 per hardware thread).

echo "START TIME: $(date)"
start=$(date +%s)
cd $PD || exit 1
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

export OMP_NUM_THREADS=1
source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh $PD || exit 1

RESULT_FILE=$PD/experiments/logs/mfu_bench_${SLURM_JOB_ID}_result.txt
NUM_ENVS=${NUM_ENVS:-32}      # override: NUM_ENVS=48 sbatch scripts/kesai/16_mfu_bench_slurm.sh
MINIBATCH=${MINIBATCH:-131072}
EXTRA_ARGS=${EXTRA_ARGS:-}    # e.g. EXTRA_ARGS="--kernel-table train.compile_mode=max-autotune"
echo "=== MFU result (paste this block back) ===" | tee ${RESULT_FILE}
python scripts/mfu_bench.py --num-envs ${NUM_ENVS} --minibatch ${MINIBATCH} ${EXTRA_ARGS} 2>&1 | grep -v "^MFU_RESULT" | grep "reference\|capability\|kernel table\|^ *[0-9][0-9.]*%\|^ *share" | tee -a ${RESULT_FILE}
python scripts/mfu_bench.py --num-envs ${NUM_ENVS} --minibatch ${MINIBATCH} --fused ${EXTRA_ARGS} 2>&1 | grep -v "^MFU_RESULT" | grep "fused\|kernel table\|^ *[0-9][0-9.]*%\|^ *share" | sed 's/^fused/fused+triton/' | tee -a ${RESULT_FILE}
env PUFFER_SLOT_ENCODER_BACKEND=cuda python scripts/mfu_bench.py --num-envs ${NUM_ENVS} --minibatch ${MINIBATCH} --fused ${EXTRA_ARGS} 2>&1 | grep -v "^MFU_RESULT" | grep "fused\|Error\|kernel table\|^ *[0-9][0-9.]*%\|^ *share" | sed 's/^fused/fused+cuda/' | tee -a ${RESULT_FILE}
echo "--- rollout split, fused+triton (env = waiting for the simulation workers, the rest is the trainer process) ---" | tee -a ${RESULT_FILE}
python scripts/mfu_bench.py --num-envs ${NUM_ENVS} --minibatch ${MINIBATCH} --fused --rollout-only --profile --warmup 1 --epochs 2 ${EXTRA_ARGS} 2>&1 | grep -A5 "section split" | tee -a ${RESULT_FILE}
echo "--- rollout split, fused+cuda ---" | tee -a ${RESULT_FILE}
env PUFFER_SLOT_ENCODER_BACKEND=cuda python scripts/mfu_bench.py --num-envs ${NUM_ENVS} --minibatch ${MINIBATCH} --fused --rollout-only --profile --warmup 1 --epochs 2 ${EXTRA_ARGS} 2>&1 | grep -A5 "section split" | tee -a ${RESULT_FILE}

NCU=$(command -v ncu || ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | tail -1)
if [ -n "${NCU}" ]; then
    echo "--- hardware counters (ncu range replay, one update minibatch of ${MINIBATCH} rows + one rollout forward) ---" | tee -a ${RESULT_FILE}
    METRICS=gpu__time_duration.sum,sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32.sum,dram__bytes.sum
    for variant in reference fused fused_cuda; do
        FLAG=""; [ ${variant} != reference ] && FLAG="--fused"
        BACKEND=triton; [ ${variant} = fused_cuda ] && BACKEND=cuda
        CSV=$PD/experiments/logs/mfu_bench_${SLURM_JOB_ID}_${variant}.csv
        ( export PUFFER_SLOT_ENCODER_BACKEND=${BACKEND}; timeout 600 ${NCU} --replay-mode range --target-processes application-only --clock-control none --nvtx \
            --nvtx-include "update" --nvtx-include "rollout" --nvtx-include "calibrate" --print-nvtx-rename kernel \
            --metrics ${METRICS} --csv --page raw --print-units base \
            python scripts/profile_policy_step.py --rows ${MINIBATCH} ${FLAG} > ${CSV} 2>&1 ) \
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
