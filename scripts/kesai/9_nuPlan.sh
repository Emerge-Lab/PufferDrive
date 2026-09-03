#!/bin/bash
#SBATCH --job-name eval_nuplan
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/log_%j.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/log_%j.err
#SBATCH --partition dev
set -u

export PD=/home/bjaeger/PufferDrive
export PY=$(conda info --base)/envs/carl_nuplan/bin/python
RUN_DIR=/home/bjaeger/PufferDrive/experiments/k_scaled_0036_1000
# the planner finds config.yaml next to final_model.pt (or one level above a models/*.pt)
export CKPT=$RUN_DIR/final_model.pt
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
echo "Evaluating checkpoint: $CKPT"
export CITY_BIN_DIR=/home/shared/data/nuplan/PufferDrive
# shell env still exports the pre-rename nuPlan casing; pin the renamed paths here
export NUPLAN_DATA_ROOT=/home/shared/data/nuplan
export NUPLAN_MAPS_ROOT=/home/shared/data/nuplan/maps
# PD first, and drop inherited entries from other PufferDrive checkouts (the login shell still exports
# /home/bjaeger/cosim_Puffer): Python would otherwise import that checkout's pufferlib package.
INHERITED_PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v "cosim_Puffer\|/PufferDrive" | paste -sd: -)
export PYTHONPATH=$PD:$CARL_DEVKIT_ROOT:$NUPLAN_DEVKIT_ROOT${INHERITED_PYTHONPATH:+:$INHERITED_PYTHONPATH}
cd "$PD" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export COSIM_OBS_HTML=infractions  # obs replays (what the policy saw) only for collision/offroad/wrong-way/no-progress scenarios; all|failures|infractions|0
# carl_visualization_callback: nuPlan ground-truth video per scenario -> $GROUP/simulation/<challenge>/<ts>/visualization
export CALLBACKS="[simulation_log_callback, carl_visualization_callback]"
export CHALLENGES=closed_loop_reactive_agents_pufferdrive
export WORKER=ray_distributed
# Per-worker memory is small now (1 policy agent + partner slots, light buffers sized to the scenario), so this is CPU-bound.
export THREADS_PER_NODE=128
# route: lane-graph route goals; roadblock: route roadblock centroids
export GOAL_SOURCE=roadblock
# Ablations (see run_nuplan_planner.sh): refill the goal window after every consumed goal; cap the ego's
# accelerating / braking jerk [m/s^3] for the first 1.5 s of each scenario (0 = off).
export SLIDING_GOAL_WINDOW=true
export STARTUP_ACCEL_JERK_CAP=2
export STARTUP_BRAKE_JERK_CAP=2
# Speed cap in slow zones: below 30 km/h (8.33 m/s) the ego may not exceed the nuPlan lane limit (+ margin); 0 = off.
export LANE_SPEED_CAP_BELOW_MPS=8.33
export LANE_SPEED_CAP_MARGIN_MPS=0
# pedestrian/bicycle partner boxes grown to the training minimum agent size in the policy's obs (0 = off)
export PEDESTRIAN_MIN_SIZE_M=0.8
ABLATION_TAG=""
[ "$SLIDING_GOAL_WINDOW" = "true" ] && ABLATION_TAG="${ABLATION_TAG}_slide"
[ "$STARTUP_ACCEL_JERK_CAP" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_jacc${STARTUP_ACCEL_JERK_CAP}"
[ "$STARTUP_BRAKE_JERK_CAP" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_jbrk${STARTUP_BRAKE_JERK_CAP}"
[ "$LANE_SPEED_CAP_BELOW_MPS" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_cap${LANE_SPEED_CAP_BELOW_MPS}mps"
[ "$PEDESTRIAN_MIN_SIZE_M" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_ped${PEDESTRIAN_MIN_SIZE_M}m"
# results live in the model's own eval folder, next to the PufferDrive benchmark evals
export GROUP=$RUN_DIR/eval/nuplan_val14_${GOAL_SOURCE}${ABLATION_TAG}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}

PATH="$(dirname "$PY"):$PATH" bash "$PD/scripts/kesai/build_ext_if_changed.sh" "$PD" || exit 1

# The planner must come from this checkout (needs the freshly built C extension above).
IMPORTED_PLANNER=$("$PY" -c "import pufferlib.ocean.cosim.nuplan.planner as p; print(p.__file__)") || { echo "ERROR: planner import failed (see traceback above)"; exit 1; }
case "$IMPORTED_PLANNER" in
    "$PD"/*) echo "planner: $IMPORTED_PLANNER" ;;
    *) echo "ERROR: planner imported from '$IMPORTED_PLANNER', expected $PD"; exit 1 ;;
esac

bash "$PD/pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh"
