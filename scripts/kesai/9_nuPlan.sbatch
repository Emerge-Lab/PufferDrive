#!/bin/bash
#SBATCH --job-name eval_nuplan
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 2-00:00
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/cosim_Puffer/experiments/logs/log_%j.out
#SBATCH --error /home/bjaeger/cosim_Puffer/experiments/logs/log_%j.err
#SBATCH --partition dev
set -u

export PD=/home/bjaeger/cosim_Puffer
export PY=$(conda info --base)/envs/carl_nuplan/bin/python
export CKPT=$PD/weights/mimolette/models/model_puffer_drive_003815.pt
export CITY_BIN_DIR=/home/shared/data/nuPlan/PufferDrive
export PYTHONPATH=$PD:$CARL_DEVKIT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}
export WORKER=sequential
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export COSIM_DEBUG_BEV=0
export CALLBACKS="[simulation_log_callback]"

RUNNER=$PD/pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh
NPAR=80
NUM_UNITS=112
EVAL_ROOT=/home/bjaeger/cosim_Puffer/experiments/nuplan_val14_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}
mkdir -p "$EVAL_ROOT"

for i in $(seq -w 00 55); do
    for REACTIVITY in nonreactive reactive; do
        while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done
        (
            UNIT=shard_${i}_${REACTIVITY}
            export SPLIT=nuplan_shard_$i GROUP=$EVAL_ROOT/shard_$i
            export CHALLENGES=closed_loop_${REACTIVITY}_agents_pufferdrive
            bash "$RUNNER" > "$EVAL_ROOT/$UNIT.log" 2>&1
            echo $? > "$EVAL_ROOT/$UNIT.exit"
            done_count=$(ls "$EVAL_ROOT"/*.exit | wc -l)
            bar=$(printf '%*s' $((done_count * 40 / NUM_UNITS)) '' | tr ' ' '#')
            printf '%s [%-40s] %3d/%d units  (%s exit %s)\n' \
                "$(date +%H:%M)" "$bar" "$done_count" "$NUM_UNITS" "$UNIT" "$(cat "$EVAL_ROOT/$UNIT.exit")"
        ) &
    done
done
wait

echo "eval_root -> $EVAL_ROOT"
FAILED=$(grep -L '^0$' "$EVAL_ROOT"/shard_*.exit || true)
[ -z "$FAILED" ] || { echo "FAILED shards:"; echo "$FAILED"; exit 1; }
