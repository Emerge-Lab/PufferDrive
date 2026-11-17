#!/bin/bash
# Complete a longest6 v1 run folder synced from the cluster on this desktop: 14_carla_longest6_v1.sh's COMPLETE_RUN mode
# with the local CARLA 0.9.10, carla_garage, the `garage` conda env and the repo .venv as policy server. Only the route
# runs without a valid record are driven, in place; the replaced files move to <run>/pre_completion_<stamp>/.
#   bash scripts/kesai/17_carla_longest6_v1_complete_local.sh experiments/<run>/eval/carla_longest6_v1_<date>_<job>
# Overridable env (default):
#   CKPT           final_model.pt of the model folder the run sits in, else of the run_info.txt model in experiments/
#   WORKERS (8)    ~2 cores and 2.5-5 GB each
#   PORT_BASE (2200; never 2000, the interactive server), STAGGER_S (20)
#   the variables of 14_carla_longest6_v1.sh; the settings its run_info.txt records default to the run's own
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
