#!/bin/bash
#SBATCH --job-name eval_nuplan
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --output experiments/logs/nuPlan_reactive_%j.out
#SBATCH --error experiments/logs/nuPlan_reactive_%j.err
#SBATCH --partition dev
set -u
source scripts/kesai/cluster_env.sh || exit 1

export PY=$(conda info --base)/envs/carl_nuplan/bin/python
# Overridable env (default):
#   RUN_DIR           the training run (0_train_8gpu and 1_train_multinode submit this job with RUN_DIR=<its run dir>)
#   CHALLENGES        closed_loop_reactive_agents_opengigaflow | closed_loop_nonreactive_agents_opengigaflow
#   WANDB_REPORT (1)  the CLS -> the run's final_eval/nuplan_{reactive,nonreactive}_CLS in wandb; 0 for ablations
RUN_DIR=${RUN_DIR:-$PD/experiments/k_scaled_0059_1000}
case "$RUN_DIR" in /*) ;; *) RUN_DIR=$PD/$RUN_DIR ;; esac # relative to the checkout: $PD is only defined inside the job
# the planner finds config.yaml next to final_model.pt (or one level above a models/*.pt)
export CKPT=$RUN_DIR/final_model.pt
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
# reactive = IDM agents (CLS-R), nonreactive = log-replay agents (CLS-NR)
export CHALLENGES=${CHALLENGES:-closed_loop_reactive_agents_opengigaflow}
case "$CHALLENGES" in
    closed_loop_reactive_agents_opengigaflow) CHALLENGE_TAG=""; WANDB_EVAL_TYPE=nuplan_reactive ;;
    closed_loop_nonreactive_agents_opengigaflow) CHALLENGE_TAG="_nonreactive"; WANDB_EVAL_TYPE=nuplan_nonreactive ;;
    *) echo "CHALLENGES must be closed_loop_reactive_agents_opengigaflow or closed_loop_nonreactive_agents_opengigaflow"; exit 1 ;;
esac
WANDB_REPORT=${WANDB_REPORT:-1}
VAL14_SCENARIO_COUNT=1118  # a run that scored fewer scenarios (LIMIT_TOTAL_SCENARIOS, failed simulations) is not reported
echo "Evaluating checkpoint: $CKPT"
export CITY_BIN_DIR=${CITY_BIN_DIR:-$SHARED_DATA/nuplan/PufferDrive}
# shell env still exports the pre-rename nuPlan casing; pin the renamed paths here
export NUPLAN_DATA_ROOT=$SHARED_DATA/nuplan
export NUPLAN_MAPS_ROOT=$SHARED_DATA/nuplan/maps
# the runner's default would mkdir a /scratch path of the old cluster (harmless, but a Permission denied line in every kesai log)
export NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-$HOME/nuplan_exp}
# PD first; drop inherited entries of other OpenGigaFlow checkouts, whose opengigaflow Python would import instead
INHERITED_PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v "cosim_Puffer\|/PufferDrive" | paste -sd: -)
export PYTHONPATH=$PD:$CARL_DEVKIT_ROOT:$NUPLAN_DEVKIT_ROOT${INHERITED_PYTHONPATH:+:$INHERITED_PYTHONPATH}
cd "$PD" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export COSIM_OBS_HTML=infractions  # obs replays (what the policy saw) only for collision/offroad/wrong-way/no-progress scenarios; all|failures|infractions|0
# + carl_visualization_callback: ground-truth videos of every scenario (14 GB; set OPENCV_FOR_THREADS_NUM=1)
export CALLBACKS="[simulation_log_callback]"
export WORKER=ray_distributed
# CPU-bound; a worker plateaus at ~2 GB with every city map cached: 384 stay under ~770 GB of a kesai node's 1 TB
export THREADS_PER_NODE=${THREADS_PER_NODE:-$([ "$CLUSTER" = kesai ] && echo 384 || echo 128)}
# the planner's eval settings are opengigaflow_planner.yaml's; an override variable of run_nuplan_planner.sh set here tags the folder
ABLATION_TAG=""
for override in SLIDING_GOAL_WINDOW:slide STARTUP_JERK_CAP_SECONDS:jsec STARTUP_ACCEL_JERK_CAP:jacc STARTUP_BRAKE_JERK_CAP:jbrk \
    VRU_SIZE_PAD_M:vrupad OBS_SLOTS_PARTNERS_N:slots EVAL_PERCEIVED_SIZE_MARGIN_M:margin REWARD_OVERSPEED:overspeed \
    LANE_WIDTH_CLIP_MIN_M:lanewmin LANE_WIDTH_CLIP_MAX_M:lanewmax BICYCLE_MIN_SPEED_MPS:vmin \
    BICYCLE_CENTER_ACCEL_CAP:centercap BICYCLE_CENTER_ACCEL_CAP_MARGIN:centercapmargin STANDSTILL_JERK_DEADBAND:deadband; do
    override_name=${override%%:*}
    [ -n "${!override_name:-}" ] || continue
    export "$override_name"  # a plain assignment above would otherwise tag the folder without reaching the runner
    ABLATION_TAG="${ABLATION_TAG}_${override##*:}${!override_name}"
done
[ "$CITY_BIN_DIR" != "$SHARED_DATA/nuplan/PufferDrive" ] && ABLATION_TAG="${ABLATION_TAG}_bins$(basename "$CITY_BIN_DIR")"
# results: the model's own eval folder, next to the OpenGigaFlow benchmark evals (overview.csv, scenarios.csv, obs_html/, simulation_log/, details/)
export GROUP=$RUN_DIR/eval/nuplan_val14${CHALLENGE_TAG}${ABLATION_TAG}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}

# NO_TRAIN: the cp39 nuPlan env builds only the C binding, so nvcc never meets this env's cu128 torch
NO_TRAIN=1 PATH="$(dirname "$PY"):$PATH" bash "$PD/scripts/kesai/build_ext_if_changed.sh" "$PD" || exit 1

# The planner must come from this checkout (needs the freshly built C extension above).
IMPORTED_PLANNER=$("$PY" -c "import opengigaflow.ocean.cosim.nuplan.planner as p; print(p.__file__)") || { echo "ERROR: planner import failed (see traceback above)"; exit 1; }
case "$IMPORTED_PLANNER" in
    "$PD"/*) echo "planner: $IMPORTED_PLANNER" ;;
    *) echo "ERROR: planner imported from '$IMPORTED_PLANNER', expected $PD"; exit 1 ;;
esac

bash "$PD/opengigaflow/ocean/cosim/nuplan/run_nuplan_planner.sh"
RUNNER_STATUS=$?
# the repo .venv holds wandb; the script reads overview.csv and never imports opengigaflow
if [ "$WANDB_REPORT" = "1" ] && [ $RUNNER_STATUS -eq 0 ]; then
    env -u PYTHONPATH "$PD/.venv/bin/python" "$PD/opengigaflow/ocean/evaluation_utils/eval_wandb_report.py" \
        "$WANDB_EVAL_TYPE" "$GROUP/overview.csv" --run-dir "$RUN_DIR" --expected-episode-count "$VAL14_SCENARIO_COUNT" \
        || echo "wandb report failed (results above are unaffected)"
fi
# the job's own log next to the results (the SBATCH --output of this script or of 4_eval_nuplan_nonreactive.sh)
JOB_LOGS=(experiments/logs/nuPlan_*_"${SLURM_JOB_ID:-none}".out experiments/logs/nuPlan_*_"${SLURM_JOB_ID:-none}".err)
for job_log in "${JOB_LOGS[@]}"; do
    [ -f "$job_log" ] && [ -d "$GROUP/details" ] && cp "$job_log" "$GROUP/details/"
done
exit $RUNNER_STATUS
