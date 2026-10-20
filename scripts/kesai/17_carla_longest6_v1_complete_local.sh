#!/bin/bash
# Complete a longest6 v1 run that lost routes on the cluster (CARLA 0.9.10 render-thread crashes, the job time limit) on
# this desktop: scripts/kesai/14_carla_longest6_v1.sh in its COMPLETE_RUN mode with the local CARLA 0.9.10, carla_garage,
# the `garage` conda env and the repo .venv as policy server. Sync the run folder from the cluster as is (it lives in the
# model's eval/ folder); only its routes without a valid record are driven, in place, WORKERS servers in parallel on the
# one GPU, then results.csv, the HTML report and the obs replay gallery are rebuilt over all routes. Nothing from the
# cluster is overwritten: the replaced route folders and worker logs move to <run>/pre_completion_<stamp>/.
#
# Usage: bash scripts/kesai/17_carla_longest6_v1_complete_local.sh experiments/k_scaled_0050_1000/eval/carla_longest6_v1_vrupad0.8_<date>_<jobid>
#
# Overridable env: CKPT (default: the model folder the run sits in, else the model named in its run_info.txt under
# $PD/experiments), WORKERS (8: ~2 cores and 2.5-5 GB each, the GPU stays idle), PORT_BASE (2200; never 2000, the
# interactive server), STAGGER_S, ROUTE_SUBSET, and every other variable of script 14 (SCENARIOS, REPETITIONS and
# COSIM_VRU_SIZE_PAD_M default to the run's own values). Expect ~1 h of wall time per hour of game time per worker.
set -u

RUN=${1:?usage: $0 <synced longest6 v1 run folder (holds routes/ and run_info.txt)>}
RUN=$(cd "$RUN" 2>/dev/null && pwd) || { echo "no such folder: $1"; exit 1; }
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PD=${PD:-$(cd "$SCRIPT_DIR/../.." && pwd)}
export GARAGE_PY=${GARAGE_PY:-$HOME/miniconda3/envs/garage/bin/python}
export SERVER_PY=${SERVER_PY:-$PD/.venv/bin/python}
export CARLA_ROOT=${CARLA_ROOT:-$HOME/ordnung/internal/CARLA_0.9.10}
export GARAGE_WORK_DIR=${GARAGE_WORK_DIR:-$HOME/ordnung/internal/carla_garage_1}
export WORKERS=${WORKERS:-8} PORT_BASE=${PORT_BASE:-2200} STAGGER_S=${STAGGER_S:-20}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
# garage's torch 1.12 needs an executable stack that glibc 2.41 refuses by default (result_parser.py)
export GLIBC_TUNABLES=${GLIBC_TUNABLES:-glibc.rtld.execstack=2}
unset WAYLAND_DISPLAY  # DISPLAY= alone still opens one window per server on this desktop

if [ -z "${CKPT:-}" ]; then
    MODEL_DIR=$(cd "$RUN/../.." && pwd)
    if [ ! -f "$MODEL_DIR/final_model.pt" ]; then
        PREVIOUS_CKPT=$(awk '$1 == "checkpoint" {print $2; exit}' "$RUN/run_info.txt" 2>/dev/null)
        MODEL_DIR=$PD/experiments/$(basename "$(dirname "$PREVIOUS_CKPT")")
    fi
    CKPT=$MODEL_DIR/final_model.pt
fi
export CKPT
[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT (set CKPT=/path/to/final_model.pt)"; exit 1; }
[ -f "$(dirname "$CKPT")/config.yaml" ] || { echo "no config.yaml next to $CKPT"; exit 1; }

export COMPLETE_RUN=$RUN
exec bash "$SCRIPT_DIR/14_carla_longest6_v1.sh"
