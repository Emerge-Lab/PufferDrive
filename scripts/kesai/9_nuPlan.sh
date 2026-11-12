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
# Overridable env: RUN_DIR (3_train_64GPU_multinode_slurm.sh submits this job with RUN_DIR=<its run dir>), CHALLENGES,
# WANDB_REPORT (1, default: the CLS becomes the training run's final_eval/nuplan_{reactive,nonreactive}_CLS in wandb,
# replacing an earlier report of that challenge; 0 for ablations, so the run keeps the number of its standard eval)
RUN_DIR=${RUN_DIR:-$PD/experiments/k_scaled_0059_1000}
case "$RUN_DIR" in /*) ;; *) RUN_DIR=$PD/$RUN_DIR ;; esac # relative to the checkout: $PD is only defined inside the job
# the planner finds config.yaml next to final_model.pt (or one level above a models/*.pt)
export CKPT=$RUN_DIR/final_model.pt
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
# reactive = IDM agents (CLS-R), nonreactive = log-replay agents (CLS-NR)
export CHALLENGES=${CHALLENGES:-closed_loop_reactive_agents_pufferdrive}
case "$CHALLENGES" in
    closed_loop_reactive_agents_pufferdrive) CHALLENGE_TAG=""; WANDB_EVAL_TYPE=nuplan_reactive ;;
    closed_loop_nonreactive_agents_pufferdrive) CHALLENGE_TAG="_nonreactive"; WANDB_EVAL_TYPE=nuplan_nonreactive ;;
    *) echo "CHALLENGES must be closed_loop_reactive_agents_pufferdrive or closed_loop_nonreactive_agents_pufferdrive"; exit 1 ;;
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
# PD first, and drop inherited entries from other PufferDrive checkouts (the login shell still exports
# $HOME/cosim_Puffer): Python would otherwise import that checkout's pufferlib package.
INHERITED_PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v "cosim_Puffer\|/PufferDrive" | paste -sd: -)
export PYTHONPATH=$PD:$CARL_DEVKIT_ROOT:$NUPLAN_DEVKIT_ROOT${INHERITED_PYTHONPATH:+:$INHERITED_PYTHONPATH}
cd "$PD" || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export COSIM_OBS_HTML=infractions  # obs replays (what the policy saw) only for collision/offroad/wrong-way/no-progress scenarios; all|failures|infractions|0
# one simulation log per scenario; the tidy keeps those of the COSIM_OBS_HTML scenarios in simulation_log/. Adding
# carl_visualization_callback renders nuPlan's ground-truth video of every scenario (1118 .avi, 14 GB, a matplotlib
# frame per step; set OPENCV_FOR_THREADS_NUM=1 against its thread pool) for the few the tidy keeps as details/videos/.
export CALLBACKS="[simulation_log_callback]"
export WORKER=ray_distributed
# Per-worker memory is small now (1 policy agent + partner slots, light buffers sized to the scenario), so this is CPU-bound.
# 128 fits the 144-CPU scaleway nodes. A kesai RTX node gives the job 480 CPUs on 256 physical cores and a worker plateaus
# at ~2.0 GB once it has cached all four city maps (256 workers = 509 GB, 30 Oct): 384 fills half the hyperthreads and
# stays under ~770 GB of the 1 TB, leaving the rest to the page cache of the logs.
export THREADS_PER_NODE=${THREADS_PER_NODE:-$([ "$CLUSTER" = kesai ] && echo 384 || echo 128)}
# Ablations (see run_nuplan_planner.sh): refill the goal window after every consumed goal; cap the ego's
# accelerating / braking jerk [m/s^3] for the first 1.5 s of each scenario (0 = off).
export SLIDING_GOAL_WINDOW=true
export STARTUP_ACCEL_JERK_CAP=2
export STARTUP_BRAKE_JERK_CAP=2
# pedestrian and bicycle partner boxes grow by this much in length and width in the policy's obs (0 = off): 0.8 puts them strictly above the smallest training vehicle
export VRU_SIZE_PAD_M=0.8
# Shadow-env obs overrides: partner slots (training 20, GIGAFLOW eval 40; 40 real partners max-pooled is OOD in
# sidewalk crowds) and the ego perceived-size margin [m per side] (0.2 = native eval; nuPlan's 2.3 x 5.2 m ego
# reads as 2.7 x 5.6 m). Unset = planner yaml / arch.py defaults (40 / 0.2).
export OBS_SLOTS_PARTNERS_N=40
export EVAL_PERCEIVED_SIZE_MARGIN_M=0.2
# overspeed conditioning alpha (unset = arch.py 0.1 = max of the U(0, 0.1) training range; 0.05 kept the ego at the limit, 0.005 let it speed)
export REWARD_OVERSPEED=0.1
# Lane-width clip [m]: the policy observes lane widths clipped to the training range (CARLA maps 2.85-4.4 m; nuPlan
# city bins have 22% of segments wider than 5 m, up to 50 m). LANE_WIDTH_CLIP_MAX_M=0 turns it off (pre-2026-10-29 evals).
export LANE_WIDTH_CLIP_MIN_M=${LANE_WIDTH_CLIP_MIN_M:-2.85}
export LANE_WIDTH_CLIP_MAX_M=${LANE_WIDTH_CLIP_MAX_M:-4.4}
# Ego plant: nuPlan's KinematicBicycleModel via CaRL's one_stage_controller, driven exactly as CaRL drives it, with the
# model's own actuator lag (it realises dt/(dt+tau) of each command; run 431 = 93.01). The policy's integrated accel is
# zeroed while the plant sits on the speed floor. Ablated: BICYCLE_MIN_SPEED_MPS -2 (Drive's floor) costs ~0.9 on every
# safety metric.
export BICYCLE_MIN_SPEED_MPS=${BICYCLE_MIN_SPEED_MPS:-0}
# Comfort: nuPlan scores the center accel (rear + d*yaw_rate^2 + d*yaw_accel), which the bicycle plant exposes.
# BICYCLE_CENTER_ACCEL_CAP=true reduces the plant's accel while turning (CaRL clip_angular_adjustment; 248 -> 20
# positive lon-accel fails), MARGIN 0.95 takes the Savitzky-Golay overshoot residual (+0.2 CLS, run 426).
export BICYCLE_CENTER_ACCEL_CAP=${BICYCLE_CENTER_ACCEL_CAP:-true}
export BICYCLE_CENTER_ACCEL_CAP_MARGIN=${BICYCLE_CENTER_ACCEL_CAP_MARGIN:-0.95}
# Run-426 anatomy arm: STANDSTILL_JERK_DEADBAND=0 (five stalls were the policy's 0-1.5 m/s^3 start intent being zeroed;
# the floor now does the deadband's anti-reverse job). Drive's jerk model keeps its axle at the box centre while
# nuPlan's sits 1.46 m behind (four inside-rear corner collisions in run 426): a training-side item, no eval-side
# conversion.
export STANDSTILL_JERK_DEADBAND=${STANDSTILL_JERK_DEADBAND:-}
ABLATION_TAG=""
[ "$SLIDING_GOAL_WINDOW" = "true" ] && ABLATION_TAG="${ABLATION_TAG}_slide"
[ "$STARTUP_ACCEL_JERK_CAP" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_jacc${STARTUP_ACCEL_JERK_CAP}"
[ "$STARTUP_BRAKE_JERK_CAP" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_jbrk${STARTUP_BRAKE_JERK_CAP}"
[ "$VRU_SIZE_PAD_M" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_vrupad${VRU_SIZE_PAD_M}m"
[ "$CITY_BIN_DIR" != "$SHARED_DATA/nuplan/PufferDrive" ] && ABLATION_TAG="${ABLATION_TAG}_bins$(basename "$CITY_BIN_DIR")"
[ -n "${OBS_SLOTS_PARTNERS_N:-}" ] && ABLATION_TAG="${ABLATION_TAG}_slots${OBS_SLOTS_PARTNERS_N}"
[ -n "${EVAL_PERCEIVED_SIZE_MARGIN_M:-}" ] && ABLATION_TAG="${ABLATION_TAG}_margin${EVAL_PERCEIVED_SIZE_MARGIN_M}"
[ -n "${REWARD_OVERSPEED:-}" ] && ABLATION_TAG="${ABLATION_TAG}_overspeed${REWARD_OVERSPEED}"
[ "$LANE_WIDTH_CLIP_MAX_M" != "0" ] && ABLATION_TAG="${ABLATION_TAG}_lanew${LANE_WIDTH_CLIP_MIN_M}-${LANE_WIDTH_CLIP_MAX_M}"
# _roadblock (below), _bicycle and _clamp name what every run uses now: kept so folder names match the earlier evals
ABLATION_TAG="${ABLATION_TAG}_bicycle_vmin${BICYCLE_MIN_SPEED_MPS}"
if [ "$BICYCLE_CENTER_ACCEL_CAP" = "true" ]; then
    ABLATION_TAG="${ABLATION_TAG}_centercap"
    [ "$BICYCLE_CENTER_ACCEL_CAP_MARGIN" != "1.0" ] && ABLATION_TAG="${ABLATION_TAG}${BICYCLE_CENTER_ACCEL_CAP_MARGIN}"
fi
ABLATION_TAG="${ABLATION_TAG}_clamp"
[ -n "$STANDSTILL_JERK_DEADBAND" ] && ABLATION_TAG="${ABLATION_TAG}_deadband${STANDSTILL_JERK_DEADBAND}"
# results: the model's own eval folder, next to the PufferDrive benchmark evals (overview.csv, scenarios.csv, obs_html/, simulation_log/, details/)
export GROUP=$RUN_DIR/eval/nuplan_val14${CHALLENGE_TAG}_roadblock${ABLATION_TAG}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}

# NO_TRAIN skips the training-only CUDA extension (_C, never imported by the planner): the cp39 nuPlan env builds only
# the C binding, so nvcc 13.4 on kesai never meets this env's cu128 torch (script 12 builds the same way)
NO_TRAIN=1 PATH="$(dirname "$PY"):$PATH" bash "$PD/scripts/kesai/build_ext_if_changed.sh" "$PD" || exit 1

# The planner must come from this checkout (needs the freshly built C extension above).
IMPORTED_PLANNER=$("$PY" -c "import pufferlib.ocean.cosim.nuplan.planner as p; print(p.__file__)") || { echo "ERROR: planner import failed (see traceback above)"; exit 1; }
case "$IMPORTED_PLANNER" in
    "$PD"/*) echo "planner: $IMPORTED_PLANNER" ;;
    *) echo "ERROR: planner imported from '$IMPORTED_PLANNER', expected $PD"; exit 1 ;;
esac

bash "$PD/pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh"
RUNNER_STATUS=$?
# the repo .venv holds wandb; the script reads overview.csv and never imports pufferlib
if [ "$WANDB_REPORT" = "1" ] && [ $RUNNER_STATUS -eq 0 ]; then
    env -u PYTHONPATH "$PD/.venv/bin/python" "$PD/pufferlib/ocean/evaluation_utils/eval_wandb_report.py" \
        "$WANDB_EVAL_TYPE" "$GROUP/overview.csv" --run-dir "$RUN_DIR" --expected-episode-count "$VAL14_SCENARIO_COUNT" \
        || echo "wandb report failed (results above are unaffected)"
fi
# the job's own log next to the results (the SBATCH --output of this script or of 13_nuPlan_nonreactive.sh)
JOB_LOGS=(experiments/logs/nuPlan_*_"${SLURM_JOB_ID:-none}".out experiments/logs/nuPlan_*_"${SLURM_JOB_ID:-none}".err)
for job_log in "${JOB_LOGS[@]}"; do
    [ -f "$job_log" ] && [ -d "$GROUP/details" ] && cp "$job_log" "$GROUP/details/"
done
exit $RUNNER_STATUS
