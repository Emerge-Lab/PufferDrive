#!/bin/bash
#SBATCH --job-name tune_fused_encoder
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 3:00:00
#SBATCH --gres gpu:1
#SBATCH --mem=125G
#SBATCH --cpus-per-task 18
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/tune_fused_%j.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/tune_fused_%j.err
#SBATCH --partition dev

# Times the fused slot-encoder kernels on this node's GPU and prints the KernelConfig to paste into
# CONFIGS_BY_CAPABILITY in pufferlib/ocean/fused_slot_encoder.py. Submit once per GPU type (H100, B200, ...);
# pick the node type with --constraint or --gres gpu:<type>:1 if the partition mixes GPUs.

echo "START TIME: $(date)"
start=$(date +%s)
cd /home/bjaeger/PufferDrive || exit 1
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

export OMP_NUM_THREADS=1
source .venv/bin/activate
bash scripts/kesai/build_ext_if_changed.sh /home/bjaeger/PufferDrive || exit 1

RESULT_FILE=/home/bjaeger/PufferDrive/experiments/logs/tune_fused_${SLURM_JOB_ID}_result.txt
CONFIG_JSON=/home/bjaeger/PufferDrive/experiments/logs/tune_fused_${SLURM_JOB_ID}_config.json
python scripts/tune_fused_slot_encoder.py --batch 131072 --shapes all --write-config ${CONFIG_JSON} 2>&1 | tee ${RESULT_FILE}.full
# The pasteable block is the tail of the full log.
sed -n '/=== fused_slot_encoder tuning result/,/=== end of tuning result ===/p' ${RESULT_FILE}.full > ${RESULT_FILE}

# Real policy update + rollout step on this GPU: compiled reference vs fused kernels with the config just tuned.
echo "=== step timing (paste this block back too) ===" | tee -a ${RESULT_FILE}
python scripts/profile_policy_step.py --rows 131072 2>&1 | grep "^RESULT" | tee -a ${RESULT_FILE}
python scripts/profile_policy_step.py --rows 131072 --fused --kernel-config ${CONFIG_JSON} 2>&1 | grep "^RESULT" | tee -a ${RESULT_FILE}
awk '/^RESULT fused=0/ {for (i=1;i<=NF;i++) if ($i ~ /^update_ms=/) {split($i,a,"="); base=a[2]}}
     /^RESULT fused=1/ {for (i=1;i<=NF;i++) if ($i ~ /^update_ms=/) {split($i,a,"="); fused=a[2]}}
     END {if (base && fused) printf "update speedup fused vs reference: %.2fx\n", base / fused}' ${RESULT_FILE} | tee -a ${RESULT_FILE}
echo "=== end of step timing ===" | tee -a ${RESULT_FILE}

echo
echo "Result block (also in ${RESULT_FILE}):"
cat ${RESULT_FILE}

end=$(date +%s)
echo "END TIME: $(date)"
echo "Runtime: $((end-start)) s"
