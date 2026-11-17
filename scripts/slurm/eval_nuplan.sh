#!/bin/bash
#SBATCH --job-name eval_nuplan
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --exclusive                      # whole node: the simulations are CPU-bound
#SBATCH --output experiments/logs/eval_nuplan_%j.out
#SBATCH --error experiments/logs/eval_nuplan_%j.err
##SBATCH --partition <partition>         # site-specific options: uncomment and set, or pass them to sbatch
# nuPlan Val14 closed-loop eval of a run's final_model.pt on one node (opengigaflow/ocean/cosim/nuplan/run_nuplan_planner.sh);
# results go to $RUN_DIR/eval/nuplan_val14[_nonreactive]_<date>_<job>. Submit from the checkout root, once experiments/logs exists:
#   RUN_DIR=experiments/<run> sbatch scripts/slurm/eval_nuplan.sh
# Settings, from the environment or edited here:
#   RUN_DIR             the training run (holds final_model.pt and config.yaml)
#   PY                  python of the nuPlan environment (CaRL's carl_nuplan conda env)
#   NUPLAN_DEVKIT_ROOT, CARL_DEVKIT_ROOT  the nuplan-devkit checkout and CaRL's nuPlan folder (holds carl_nuplan)
#   NUPLAN_DATA_ROOT, NUPLAN_MAPS_ROOT, NUPLAN_EXP_ROOT  as nuplan-devkit reads them
#   CITY_BIN_DIR        the nuPlan city maps as .bin files (nuplan__<map>.bin)
#   CHALLENGES          closed_loop_reactive_agents_opengigaflow (IDM agents, CLS-R, the default) or
#                       closed_loop_nonreactive_agents_opengigaflow (log-replay agents, CLS-NR)
#   THREADS_PER_NODE    simulation workers (default: ray's, one per CPU); each plateaus at ~2 GB with every city map cached
#   COSIM_OBS_HTML (infractions)  replay pages of what the policy saw: all | failures | infractions | 0
#   CALLBACKS           + carl_visualization_callback: ground-truth videos of every scenario (14 GB; set OPENCV_FOR_THREADS_NUM=1)
#   WANDB_REPORT (0)    1: the CLS -> the run's final_eval/nuplan_{reactive,nonreactive}_CLS in wandb
#   EVAL_TAG            appended to the result folder's name
# The planner's eval settings are opengigaflow_planner.yaml's; an override variable of run_nuplan_planner.sh set here tags the folder.
set -u
RUN_DIR=${RUN_DIR:-experiments/my_run}
PY=${PY:-/path/to/envs/carl_nuplan/bin/python}
NUPLAN_DEVKIT_ROOT=${NUPLAN_DEVKIT_ROOT:-/path/to/nuplan-devkit}
CARL_DEVKIT_ROOT=${CARL_DEVKIT_ROOT:-/path/to/CaRL/nuPlan}
NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT:-/path/to/nuplan}
NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-$NUPLAN_DATA_ROOT/maps}
NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT:-$HOME/nuplan_exp}
CITY_BIN_DIR=${CITY_BIN_DIR:-/path/to/nuplan_city_bins}
CHALLENGES=${CHALLENGES:-closed_loop_reactive_agents_opengigaflow}
THREADS_PER_NODE=${THREADS_PER_NODE:-}
COSIM_OBS_HTML=${COSIM_OBS_HTML:-infractions}
CALLBACKS=${CALLBACKS:-"[simulation_log_callback]"}
WORKER=${WORKER:-ray_distributed}
WANDB_REPORT=${WANDB_REPORT:-0}
EVAL_TAG=${EVAL_TAG:-}

source scripts/slurm/job_env.sh || exit 1

case "$RUN_DIR" in /*) ;; *) RUN_DIR=$PD/$RUN_DIR ;; esac
# the planner finds config.yaml next to final_model.pt (or one level above a models/*.pt)
export CKPT=$RUN_DIR/final_model.pt
for path in "$CKPT" "$PY" "$NUPLAN_DEVKIT_ROOT" "$CARL_DEVKIT_ROOT" "$NUPLAN_DATA_ROOT" "$NUPLAN_MAPS_ROOT" "$CITY_BIN_DIR"; do
    [ -e "$path" ] || { echo "missing $path (the settings are listed at the top of scripts/slurm/eval_nuplan.sh)"; exit 1; }
done
case "$CHALLENGES" in
    closed_loop_reactive_agents_opengigaflow) CHALLENGE_TAG=""; WANDB_EVAL_TYPE=nuplan_reactive ;;
    closed_loop_nonreactive_agents_opengigaflow) CHALLENGE_TAG="_nonreactive"; WANDB_EVAL_TYPE=nuplan_nonreactive ;;
    *) echo "CHALLENGES must be closed_loop_reactive_agents_opengigaflow or closed_loop_nonreactive_agents_opengigaflow"; exit 1 ;;
esac
VAL14_SCENARIO_COUNT=1118  # a run that scored fewer scenarios (LIMIT_TOTAL_SCENARIOS, failed simulations) is not reported
echo "Evaluating checkpoint: $CKPT"
export PY CHALLENGES CITY_BIN_DIR NUPLAN_DATA_ROOT NUPLAN_MAPS_ROOT NUPLAN_EXP_ROOT NUPLAN_DEVKIT_ROOT COSIM_OBS_HTML CALLBACKS WORKER
[ -n "$THREADS_PER_NODE" ] && export THREADS_PER_NODE
# this checkout first: another OpenGigaFlow checkout on the inherited path would import instead
export PYTHONPATH=$PD:$CARL_DEVKIT_ROOT:$NUPLAN_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}
cd "$PD" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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
# results: the model's own eval folder, next to the OpenGigaFlow benchmark evals (overview.csv, scenarios.csv, obs_html/, simulation_log/, details/)
export GROUP=$RUN_DIR/eval/nuplan_val14${CHALLENGE_TAG}${ABLATION_TAG}${EVAL_TAG}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}

# NO_TRAIN: the nuPlan env builds only the C binding, without the torch training extension
NO_TRAIN=1 PATH="$(dirname "$PY"):$PATH" bash "$PD/scripts/slurm/build_ext_if_changed.sh" "$PD" || exit 1

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
# the job's own log next to the results
JOB_LOGS=(experiments/logs/*_"${SLURM_JOB_ID:-none}".out experiments/logs/*_"${SLURM_JOB_ID:-none}".err)
for job_log in "${JOB_LOGS[@]}"; do
    [ -f "$job_log" ] && [ -d "$GROUP/details" ] && cp "$job_log" "$GROUP/details/"
done
exit $RUNNER_STATUS
