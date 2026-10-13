#!/bin/bash
# nuPlan co-sim on each <challenge>:<checkpoint> job with the HIFT-PPO eval conditioning, scored with
# nuplan_paper_metrics.py. One challenge per job: simulation logs pickle the planner (~5 GB per 112
# scenarios), so each job's logs are scored and deleted before the next. CKPT "expert" runs nuPlan's
# log-replay planner through the same scorer as a calibration.
#
#   scripts/eval/run_nuplan_paper_eval.sh nonreactive:/path/final_model.pt reactive:/path/final_model.pt nonreactive:expert
#
# Required env: PY (co-sim python), PD (co-sim PufferDrive checkout), NUPLAN_DATA_ROOT, NUPLAN_MAPS_ROOT,
# NUPLAN_EXP_ROOT, NUPLAN_DEVKIT_ROOT, CITY_BIN_DIR. Optional: SPLIT (reduced_val14_split), OUT_PREFIX
# (paper_eval), CONDITIONING_FILE (hift_eval_conditioning.txt; empty keeps the clean-eval vector),
# THREADS_PER_NODE (16).
set -u
: "${PY:?}" "${PD:?}" "${NUPLAN_DATA_ROOT:?}" "${NUPLAN_MAPS_ROOT:?}" "${NUPLAN_EXP_ROOT:?}" "${NUPLAN_DEVKIT_ROOT:?}" "${CITY_BIN_DIR:?}"
export NUPLAN_DATA_ROOT NUPLAN_MAPS_ROOT NUPLAN_EXP_ROOT NUPLAN_DEVKIT_ROOT CITY_BIN_DIR PY PD
export SPLIT=${SPLIT:-reduced_val14_split} CALLBACKS="[simulation_log_callback]" COSIM_OBS_HTML=0
export WORKER=ray_distributed THREADS_PER_NODE=${THREADS_PER_NODE:-16}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDITIONING_FILE=${CONDITIONING_FILE-$SCRIPT_DIR/hift_eval_conditioning.txt}
OUT_PREFIX=${OUT_PREFIX:-paper_eval}
EXTRA_HYDRA=""
if [ -n "$CONDITIONING_FILE" ]; then
    for kv in $(cat "$CONDITIONING_FILE"); do EXTRA_HYDRA+="++planner.pufferdrive_planner.env_overrides.$kv "; done
fi
export EXTRA_HYDRA

for JOB in "$@"; do
    MODE=${JOB%%:*}
    export CKPT=${JOB#*:}
    NAME=$([ "$CKPT" = expert ] && echo expert || basename "$(dirname "$CKPT")")
    export GROUP=$NUPLAN_EXP_ROOT/${OUT_PREFIX}_$NAME CHALLENGES=closed_loop_${MODE}_agents_pufferdrive
    echo "== $NAME $MODE $(date +%H:%M)"
    if [ "$CKPT" = expert ]; then
        (cd "$PD" && PYTHONPATH=$PD "$PY" "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
            +simulation=$CHALLENGES scenario_builder=nuplan \
            scenario_builder.data_root="$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/val" scenario_filter=$SPLIT \
            planner=log_future_planner callback="$CALLBACKS" \
            main_callback="[time_callback, metric_file_callback, metric_aggregator_callback, csv_main_callback]" \
            hydra.searchpath="[pkg://pufferlib.ocean.cosim.nuplan.config, pkg://carl_nuplan.planning.script.config.common, pkg://carl_nuplan.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
            group="$GROUP" worker=$WORKER worker.threads_per_node=$THREADS_PER_NODE > "$GROUP.$MODE.log" 2>&1)
    else
        (cd "$PD" && bash pufferlib/ocean/cosim/nuplan/run_nuplan_planner.sh > "$GROUP.$MODE.log" 2>&1)
    fi
    echo "   exit $?"
    PYTHONPATH=$PD "$PY" "$SCRIPT_DIR/nuplan_paper_metrics.py" score-logs "$GROUP"
    rm -rf "$GROUP"/simulation/*/*/simulation_log
done
