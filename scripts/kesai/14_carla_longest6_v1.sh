#!/bin/bash
#SBATCH --job-name eval_longest6_v1
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=0
#SBATCH --exclusive                      # whole node on either cluster, --mem=0 = all of its memory
#SBATCH --output experiments/logs/carla_longest6_v1_%j.out
#SBATCH --error experiments/logs/carla_longest6_v1_%j.err
#SBATCH --open-mode=append               # a requeued job keeps the log of the instance it resumes
#SBATCH --partition dev
# CARLA longest6 v1 eval of an OpenGigaFlow checkpoint through carla_garage's unmodified Leaderboard 1.0 (CARLA 0.9.10)
# and opengigaflow/ocean/cosim/carla/lb1/leaderboard_agent.py: the evaluator runs in the `garage` env (Python 3.7), the
# agent starts lb1/policy_server.py in $SERVER_PY per route. WORKERS workers pull the route runs (routes x SEEDS) from
# one queue, longest first; results.csv from garage's tools/result_parser.py, the replay gallery in $OUT/obs_html, then
# scripts/eval/tidy_carla_run.py flattens $OUT to results.csv, run_info.txt, obs_html/, routes/ and details/.
# Overridable env (default):
#   RUN_DIR, CKPT, CARLA_ROOT, GARAGE_WORK_DIR, GARAGE_PY, SERVER_PY, PD  checkpoint, checkouts, pythons
#   ROUTE_DIR, ROUTES_XML      the split route xmls; the combined longest6.xml for the parser and the gallery
#   SCENARIOS (1)              1 = eval_scenarios.json, 0 = no_scenarios.json
#   SEEDS ("0 1 2")            traffic-manager seeds, one pass over the routes each; empty = seed 100, legacy layout
#   REPETITIONS (1)            leaderboard repetitions within a pass
#   ROUTE_SUBSET               "0 5 17": only these route ids
#   WORKERS (32), NUM_GPUS (the allocation's), STAGGER_S (5 s between worker launches), MAX_ATTEMPTS (3)
#   PORT_BASE (12000)          worker w: rpc port PORT_BASE+50w, TM PORT_BASE+6000+50w; run_info.txt: the base used
#   OMP_NUM_THREADS            policy server threads (2 from 16 workers on, else 4)
#   LOGGING (1)                telemetry, world log and obs replay per route; 0 = scores only
#   CARLA_VIEW (0)             1 = chase-cam mp4 per route (needs LOGGING=1)
#   OBS_HTML (1 = failures)    replay pages of the route runs with a flagged infraction or failed status; all; 0 = none
#   REPORT (0)                 1 = scripts/eval/analyze_carla_cosim.py's HTML report in $OUT/report (needs LOGGING=1)
#   TIDY (1)                   0 = keep every file
#   WANDB_REPORT (1)           results.csv's DS becomes the run's final_eval/longest6_v1_DS in wandb; 0 for ablations
#   LIBTIFF5_COMPAT (1)        no libtiff.so.5 for the egg: the garage env's, else a link to the system .so.6; 0 = never link
#   COSIM_MAX_SPEED_MPS (30)   ego speed cap
#   COSIM_ZERO_PARTNER_STOPPED_TIME (1)  vehicles' stopped-time obs held at 0, walkers keep theirs; 0 = all real
#   COSIM_VRU_SIZE_PAD_M (0.8) added to the length and width of walker and two-wheeler boxes; 0 = true boxes
#   COSIM_OBS_SLOTS_PARTNERS_N (40)  partner slots in the policy's obs; empty = the checkpoint's training value
#   COSIM_DYNAMICS_SOURCE (carla_pedals: CARLA's physics moves the ego; opengigaflow: teleported by OpenGigaFlow's)
#   COSIM_PEDAL_YELLOW_MARGIN (1)  carla_pedals: 0 = the policy sees CARLA's remaining yellow
#   COMPLETE_RUN               an existing $OUT: reruns in place its route runs without a valid record with the settings
#                              of its run_info.txt (CKPT must match); a requeued job completes its own folder this way
set -u
source scripts/kesai/cluster_env.sh || exit 1

export GARAGE_PY=${GARAGE_PY:-$HOME/miniconda3/envs/garage/bin/python}   # cp37: the CARLA 0.9.10 egg
export SERVER_PY=${SERVER_PY:-$HOME/miniconda3/envs/carl/bin/python}     # any OpenGigaFlow-capable python
export CARLA_ROOT=${CARLA_ROOT:-$HOME/CARLA_0.9.10}
export GARAGE_WORK_DIR=${GARAGE_WORK_DIR:-$HOME/carla_garage_1}
RUN_DIR=${RUN_DIR:-$PD/experiments/k_scaled_0071_1000}
# the agent finds config.yaml next to final_model.pt (or one level above a models/*.pt)
export CKPT=${CKPT:-$RUN_DIR/final_model.pt}
ROUTE_DIR=${ROUTE_DIR:-$GARAGE_WORK_DIR/leaderboard/data/longest6_split}
ROUTES_XML=${ROUTES_XML:-$GARAGE_WORK_DIR/leaderboard/data/longest6.xml}
COMPLETE_RUN=${COMPLETE_RUN:-}
# a requeued job (preemption, node failure) resumes its own run folder instead of starting over
if [ -z "$COMPLETE_RUN" ] && [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    for candidate in "$RUN_DIR"/eval/carla_longest6_v1*_"${SLURM_JOB_ID:-none}"; do   # sorted by start time: the newest wins
        [ -d "$candidate/routes" ] && [ -f "$candidate/run_info.txt" ] && COMPLETE_RUN=$candidate
    done
    [ -n "$COMPLETE_RUN" ] && echo "requeued job $SLURM_JOB_ID (restart $SLURM_RESTART_COUNT): resuming $COMPLETE_RUN"
fi
if [ -n "$COMPLETE_RUN" ]; then
    [ -d "$COMPLETE_RUN/routes" ] && [ -f "$COMPLETE_RUN/run_info.txt" ] || { echo "COMPLETE_RUN=$COMPLETE_RUN is not a run folder (needs routes/ and run_info.txt)"; exit 1; }
    run_info_value() { awk -v key="$1" '$1 == key {print $2; exit}' "$COMPLETE_RUN/run_info.txt"; }
    # the run's own settings, so the completed routes are scored like the rest
    PREVIOUS_SCENARIO_FILE=$(run_info_value scenarios)
    [ -n "$PREVIOUS_SCENARIO_FILE" ] && SCENARIOS=${SCENARIOS:-$([ "$(basename "$PREVIOUS_SCENARIO_FILE")" = "no_scenarios.json" ] && echo 0 || echo 1)}
    REPETITIONS=${REPETITIONS:-$(run_info_value repetitions)}
    # the rest of the line: several seeds; a run from before the seeds layout has no such line -> SEEDS stays empty
    SEEDS=${SEEDS-$(awk '$1 == "seeds" {$1 = ""; print substr($0, 2); exit}' "$COMPLETE_RUN/run_info.txt")}
    COSIM_VRU_SIZE_PAD_M=${COSIM_VRU_SIZE_PAD_M:-$(run_info_value vru_size_pad_m)}
    COSIM_OBS_SLOTS_PARTNERS_N=${COSIM_OBS_SLOTS_PARTNERS_N-$(run_info_value obs_slots_partners_n)}   # a run from before the knob: empty = the checkpoint's value
    PREVIOUS_WALKER_SPEED_FLOOR_MPS=$(run_info_value walker_speed_floor_mps)   # written by runs from before the floor was removed
    [[ "${PREVIOUS_WALKER_SPEED_FLOOR_MPS:-0}" =~ ^0([.]0+)?$ ]] || { echo "COMPLETE_RUN=$COMPLETE_RUN was driven with the removed walker speed floor ($PREVIOUS_WALKER_SPEED_FLOOR_MPS m/s) and cannot be completed"; exit 1; }
    PREVIOUS_PEDAL_REVERSE=$(run_info_value pedal_reverse)   # written by runs from before reverse was removed; only carla_pedals runs used it
    [ "$(run_info_value dynamics_source)" = "carla_pedals" ] && [ "${PREVIOUS_PEDAL_REVERSE:-0}" != "0" ] && { echo "COMPLETE_RUN=$COMPLETE_RUN was driven with the removed pedal reverse (pedal_reverse $PREVIOUS_PEDAL_REVERSE) and cannot be completed"; exit 1; }
    COSIM_DYNAMICS_SOURCE=${COSIM_DYNAMICS_SOURCE:-$(run_info_value dynamics_source)}
    COSIM_DYNAMICS_SOURCE=${COSIM_DYNAMICS_SOURCE:-opengigaflow}   # a run from before the knob was driven with opengigaflow
    COSIM_PEDAL_YELLOW_MARGIN=${COSIM_PEDAL_YELLOW_MARGIN:-$(run_info_value pedal_yellow_margin)}   # a pedal run from before the knob has no such line: set it by hand
fi
SCENARIOS=${SCENARIOS:-1}
if [ "$SCENARIOS" = "1" ]; then
    SCENARIO_FILE=$GARAGE_WORK_DIR/leaderboard/data/scenarios/eval_scenarios.json
else
    SCENARIO_FILE=$GARAGE_WORK_DIR/leaderboard/data/scenarios/no_scenarios.json
fi
REPETITIONS=${REPETITIONS:-1}
SEEDS=${SEEDS-0 1 2}   # traffic-manager seeds, one pass over the routes each (the carla_garage / CaRL protocol); empty = legacy
for seed in $SEEDS; do [[ "$seed" =~ ^[0-9]+$ ]] || { echo "SEEDS=\"$SEEDS\" must be integers"; exit 1; }; done
if [ -n "$SEEDS" ]; then read -r -a SEED_LIST <<< "$SEEDS"; else SEED_LIST=(""); fi   # "" = the evaluator's default seed, route_<id> folders
unit_dir() { printf "route_%02d%s" "$1" "${2:+_s$2}"; }                                      # $1 route id, $2 seed
unit_json() { printf "%s%s/longest_weathers_%s.json" "$OUT/aggregate" "${2:+/s$2}" "$1"; }   # the parser's copy of the record
unit_record() {  # the route run's record: in its folder while it drives, flat once tidied
    local nested=$OUT/routes/$(unit_dir "$1" "$2")/result.json
    [ -f "$nested" ] && echo "$nested" || echo "$OUT/routes/$(unit_dir "$1" "$2").result.json"
}
ROUTE_SUBSET=${ROUTE_SUBSET:-}
PORT_BASE=${PORT_BASE:-12000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
WORKERS=${WORKERS:-32}
STAGGER_S=${STAGGER_S:-5}
LOGGING=${LOGGING:-1}
CARLA_VIEW=${CARLA_VIEW:-0}
[ "$CARLA_VIEW" = "1" ] && [ "$LOGGING" != "1" ] && { echo "CARLA_VIEW=1 needs LOGGING=1"; exit 1; }
OBS_HTML=${OBS_HTML:-1}   # 1|failures: replay pages of the failed route runs; all: every route run; 0: none
case "$OBS_HTML" in 1) OBS_HTML=failures ;; 0|failures|all) ;; *) echo "OBS_HTML=\"$OBS_HTML\" must be 0, 1, failures or all"; exit 1 ;; esac
REPORT=${REPORT:-0}
TIDY=${TIDY:-1}
WANDB_REPORT=${WANDB_REPORT:-1}
export COSIM_VRU_SIZE_PAD_M=${COSIM_VRU_SIZE_PAD_M:-0.8}
export COSIM_OBS_SLOTS_PARTNERS_N=${COSIM_OBS_SLOTS_PARTNERS_N-40}   # GIGAFLOW's eval value; empty = the checkpoint's training value
[[ -z "$COSIM_OBS_SLOTS_PARTNERS_N" || "$COSIM_OBS_SLOTS_PARTNERS_N" =~ ^[1-9][0-9]*$ ]] || { echo "COSIM_OBS_SLOTS_PARTNERS_N=\"$COSIM_OBS_SLOTS_PARTNERS_N\" must be a positive integer or empty"; exit 1; }
export COSIM_DYNAMICS_SOURCE=${COSIM_DYNAMICS_SOURCE:-carla_pedals}
[[ "$COSIM_DYNAMICS_SOURCE" =~ ^(opengigaflow|carla_pedals)$ ]] || { echo "COSIM_DYNAMICS_SOURCE=\"$COSIM_DYNAMICS_SOURCE\" must be opengigaflow or carla_pedals"; exit 1; }
export COSIM_PEDAL_YELLOW_MARGIN=${COSIM_PEDAL_YELLOW_MARGIN:-1}
[[ "$COSIM_PEDAL_YELLOW_MARGIN" =~ ^[01]$ ]] || { echo "COSIM_PEDAL_YELLOW_MARGIN=\"$COSIM_PEDAL_YELLOW_MARGIN\" must be 0 or 1"; exit 1; }
EVALUATOR=$GARAGE_WORK_DIR/leaderboard/leaderboard/leaderboard_evaluator_local.py
CARLA_EGG=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
AGENT=$PD/opengigaflow/ocean/cosim/carla/lb1/leaderboard_agent.py
for path in "$CKPT" "$GARAGE_PY" "$SERVER_PY" "$CARLA_ROOT/CarlaUE4.sh" "$CARLA_EGG" "$EVALUATOR" "$ROUTE_DIR" "$ROUTES_XML" \
            "$SCENARIO_FILE" "$AGENT" "$GARAGE_WORK_DIR/tools/result_parser.py"; do
    [ -e "$path" ] || { echo "missing $path"; exit 1; }
done
for path in "$GARAGE_PY" "$SERVER_PY" "$CARLA_ROOT/CarlaUE4.sh"; do
    [ -x "$path" ] || { echo "not executable: $path (chmod +x, or the filesystem is mounted noexec)"; exit 1; }
done
echo "Evaluating checkpoint: $CKPT"

# GPUs of this allocation; the workers share them round-robin
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")}
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 8 ]; then echo "NUM_GPUS=$NUM_GPUS must be 1..8"; exit 1; fi
if [ "$WORKERS" -lt 1 ] || [ "$WORKERS" -gt 64 ]; then echo "WORKERS=$WORKERS must be 1..64"; exit 1; fi
command -v flock >/dev/null 2>&1 || { echo "flock (util-linux) is needed for the route queue"; exit 1; }

# bases below the ephemeral range (32768+) and clear of node_exporter (9100) and dcgm-exporter (9400)
PORT_BASE=$(pick_port_base "$WORKERS" "$PORT_BASE" 12000 22000) || { echo "no free port range for $WORKERS workers (ss -ltun)"; exit 1; }
echo "port base $PORT_BASE: rpc $PORT_BASE+50w, TM $((PORT_BASE + 6000))+50w"

# The standard eval's folder is carla_longest6_v1_<date>_<job>; a setting that differs from its default adds its tag.
TAG=longest6_v1$([ "$SCENARIOS" = "1" ] || echo "_noscen")$([ "$REPETITIONS" = "1" ] || echo "_rep$REPETITIONS")
case "$SEEDS" in "0 1 2") ;; "") TAG=${TAG}_legacyseed ;; *) TAG=${TAG}_seeds$(echo "$SEEDS" | tr ' ' '-') ;; esac
TAG=$TAG$([ "$COSIM_VRU_SIZE_PAD_M" = "0.8" ] || echo "_vrupad$COSIM_VRU_SIZE_PAD_M")
TAG=$TAG$([ "$COSIM_OBS_SLOTS_PARTNERS_N" = "40" ] || echo "_partners${COSIM_OBS_SLOTS_PARTNERS_N:-trained}")
TAG=$TAG$([ "$COSIM_DYNAMICS_SOURCE" = "carla_pedals" ] || echo "_$COSIM_DYNAMICS_SOURCE")
if [ "$COSIM_DYNAMICS_SOURCE" = "carla_pedals" ]; then
    TAG=$TAG$([ "$COSIM_PEDAL_YELLOW_MARGIN" = "1" ] || echo "_noyellowmargin")
fi
# results live in the model's own eval folder, next to the OpenGigaFlow benchmark evals
OUT=${COMPLETE_RUN:-$RUN_DIR/eval/carla_${TAG}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}
mkdir -p "$OUT/routes" "$OUT/aggregate"
: > "$OUT/server_pids"
GIT_STATE=$(git -C "$PD" rev-parse --short HEAD 2>/dev/null)$(git -C "$PD" diff --quiet 2>/dev/null || echo " dirty")
if [ -z "$COMPLETE_RUN" ]; then
    {
        echo "checkpoint $CKPT"; echo "routes $ROUTE_DIR ($ROUTES_XML)"; echo "scenarios $SCENARIO_FILE"
        echo "seeds $SEEDS"; echo "repetitions $REPETITIONS"; echo "gpus $NUM_GPUS"; echo "vru_size_pad_m $COSIM_VRU_SIZE_PAD_M"
        echo "obs_slots_partners_n $COSIM_OBS_SLOTS_PARTNERS_N"
        echo "dynamics_source $COSIM_DYNAMICS_SOURCE"
        echo "pedal_yellow_margin $COSIM_PEDAL_YELLOW_MARGIN"
        echo "port_base $PORT_BASE"
        echo "git $GIT_STATE"
    } > "$OUT/run_info.txt"
else
    PREVIOUS_CKPT=$(run_info_value checkpoint)
    PREVIOUS_GIT=$(awk '$1 == "git" {$1 = ""; print substr($0, 2); exit}' "$COMPLETE_RUN/run_info.txt")
    [ "$(basename "$(dirname "$PREVIOUS_CKPT")")" = "$(basename "$(dirname "$CKPT")")" ] \
        || echo "WARNING: the run was driven with $PREVIOUS_CKPT, this completion uses $CKPT"
    [ "$PREVIOUS_GIT" = "$GIT_STATE" ] \
        || echo "WARNING: the run used git '$PREVIOUS_GIT', this checkout is '$GIT_STATE' (PD=<checkout of that commit> keeps the routes comparable)"
fi
echo "Results -> $OUT"

carla_imports() { PYTHONPATH="$CARLA_EGG" "$GARAGE_PY" -c "import carla" >/dev/null 2>&1; }
# the 0.9.10 egg links libtiff.so.5, which newer distros no longer ship
if ! carla_imports; then
    GARAGE_LIB=$(cd "$(dirname "$GARAGE_PY")/../lib" 2>/dev/null && pwd)
    if [ -n "$GARAGE_LIB" ] && [ -e "$GARAGE_LIB/libtiff.so.5" ]; then
        export LD_LIBRARY_PATH="$GARAGE_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "libtiff.so.5 from the garage env: $GARAGE_LIB"
    fi
fi
if ! carla_imports && [ "${LIBTIFF5_COMPAT:-1}" = "1" ]; then
    SYSTEM_LIBTIFF=$( (ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null) | awk '/libtiff.so.6 /{print $NF; exit}')
    if [ -n "$SYSTEM_LIBTIFF" ]; then
        mkdir -p "$OUT/compat_lib" && ln -sf "$SYSTEM_LIBTIFF" "$OUT/compat_lib/libtiff.so.5"
        export LD_LIBRARY_PATH="$OUT/compat_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "libtiff.so.5 missing: using $SYSTEM_LIBTIFF via $OUT/compat_lib"
    fi
fi
carla_imports || { echo "ERROR: $GARAGE_PY cannot import the CARLA egg:"; PYTHONPATH="$CARLA_EGG" "$GARAGE_PY" -c "import carla"; exit 1; }

# the garage evaluator and the CARLA egg only; the policy server gets PYTHONPATH=$PD from the agent
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG:$GARAGE_WORK_DIR/scenario_runner:$GARAGE_WORK_DIR/leaderboard"
export SCENARIO_RUNNER_ROOT=$GARAGE_WORK_DIR/scenario_runner
export LEADERBOARD_ROOT=$GARAGE_WORK_DIR/leaderboard
export BENCHMARK=longest6 DATAGEN=0
cd "$PD" || exit 1
OMP_THREADS=${OMP_NUM_THREADS:-$([ "$WORKERS" -ge 16 ] && echo 2 || echo 4)}
export OMP_NUM_THREADS=$OMP_THREADS MKL_NUM_THREADS=$OMP_THREADS OPENBLAS_NUM_THREADS=$OMP_THREADS NUMEXPR_NUM_THREADS=$OMP_THREADS
export COSIM_SERVER_PYTHON=$SERVER_PY
export COSIM_DEVICE=${COSIM_DEVICE:-cpu}
export COSIM_MAX_SPEED_MPS=${COSIM_MAX_SPEED_MPS:-30}
export COSIM_ZERO_PARTNER_STOPPED_TIME=${COSIM_ZERO_PARTNER_STOPPED_TIME:-1}
export COSIM_OBS_HTML_MAX_STEPS=${COSIM_OBS_HTML_MAX_STEPS:-20000}
export COSIM_OBS_HTML_RENDER=0  # routes save the compact replay only; render_carla_obs_html.py renders all pages into $OUT/obs_html

PATH="$(dirname "$SERVER_PY"):$PATH" bash "$PD/scripts/kesai/build_ext_if_changed.sh" "$PD" || exit 1
IMPORTED_CORE=$(env -u PYTHONPATH PYTHONPATH="$PD" "$SERVER_PY" -c "import opengigaflow.ocean.cosim.carla.shadow_ego as s; print(s.__file__)") \
    || { echo "ERROR: shadow env import failed in $SERVER_PY (see traceback above)"; exit 1; }
case "$IMPORTED_CORE" in
    "$PD"/*) echo "shadow env: $IMPORTED_CORE" ;;
    *) echo "ERROR: shadow env imported from '$IMPORTED_CORE', expected $PD"; exit 1 ;;
esac
"$GARAGE_PY" -c "import carla, leaderboard.leaderboard_evaluator_local" || { echo "ERROR: the garage python cannot import the evaluator (see traceback above)"; exit 1; }

route_done() {  # 0 when the route json holds a record that is not one of the crash states garage resubmits
    "$SERVER_PY" - "$1" <<'EOF'
import json, sys
try:
    records = json.load(open(sys.argv[1]))["_checkpoint"]["records"]
except Exception:
    sys.exit(1)
crashed = ("Failed - Agent couldn't be set up", "Failed - Simulation crashed", "Failed - Agent crashed")
sys.exit(0 if records and all(r["status"] not in crashed for r in records) else 1)
EOF
}

# route ids = the split files longest_weathers_<id>.xml
mapfile -t ROUTE_IDS < <(ls "$ROUTE_DIR"/longest_weathers_*.xml | sed 's/.*longest_weathers_\([0-9]*\)\.xml/\1/' | sort -n)
TOTAL_ROUTES=${#ROUTE_IDS[@]}
[ -n "$ROUTE_SUBSET" ] && read -r -a ROUTE_IDS <<< "$ROUTE_SUBSET"
# work units = route x seed ("<route id>:<seed>"), each its own server run and result folder
UNITS=()
for seed in "${SEED_LIST[@]}"; do for route_id in "${ROUTE_IDS[@]}"; do UNITS+=("$route_id:$seed"); done; done
TOTAL_UNITS=$((TOTAL_ROUTES * ${#SEED_LIST[@]}))
if [ -n "$COMPLETE_RUN" ]; then
    PENDING=()
    for unit in "${UNITS[@]}"; do
        route_done "$(unit_record "${unit%%:*}" "${unit##*:}")" || PENDING+=("$unit")
    done
    UNITS=(${PENDING[@]+"${PENDING[@]}"})
    COMPLETION_STAMP=$(date +%Y%m%d_%H%M%S)
    if [ "${#UNITS[@]}" -eq 0 ]; then
        echo "every route run of $OUT has a valid record: only results.csv, the report and the gallery are rebuilt"
    else
        echo "completing $OUT: ${#UNITS[@]} of $TOTAL_UNITS route runs without a valid record: ${UNITS[*]}"
        # what the completion replaces stays readable next to the live results
        PREVIOUS=$OUT/pre_completion_$COMPLETION_STAMP
        mkdir -p "$PREVIOUS/routes"
        for unit in "${UNITS[@]}"; do
            route_name=$(unit_dir "${unit%%:*}" "${unit##*:}")
            [ -d "$OUT/routes/$route_name" ] && mv "$OUT/routes/$route_name" "$PREVIOUS/routes/"
            for f in "$OUT/routes/$route_name".*; do [ -e "$f" ] && mv "$f" "$PREVIOUS/routes/"; done   # a tidied run's flat files
            rm -f "$(unit_json "${unit%%:*}" "${unit##*:}")"
        done
        for f in "$OUT"/worker_w*.log; do [ -e "$f" ] && mv "$f" "$PREVIOUS/"; done
    fi
    echo "completion $COMPLETION_STAMP on $(hostname): route runs ${UNITS[*]:-none}; checkpoint $CKPT; scenarios $SCENARIO_FILE; seeds $SEEDS; repetitions $REPETITIONS; vru_size_pad_m $COSIM_VRU_SIZE_PAD_M; obs_slots_partners_n $COSIM_OBS_SLOTS_PARTNERS_N; dynamics_source $COSIM_DYNAMICS_SOURCE; pedal_yellow_margin $COSIM_PEDAL_YELLOW_MARGIN; gpus $NUM_GPUS; git $GIT_STATE" >> "$OUT/run_info.txt"
fi
[ "$WORKERS" -gt "${#UNITS[@]}" ] && WORKERS=${#UNITS[@]}
# one shared queue, longest route first; length = the xml's polyline (a failed run's duration would read short)
QUEUE=$OUT/queue
"$SERVER_PY" - "$ROUTE_DIR" "${UNITS[@]}" > "$QUEUE" <<'EOF'
import math, sys, xml.etree.ElementTree as ET
route_dir, units = sys.argv[1], sys.argv[2:]  # unit = "<route id>:<seed>"
def length(rid):
    pts = [(float(w.get("x")), float(w.get("y"))) for w in ET.parse(f"{route_dir}/longest_weathers_{rid}.xml").iter("waypoint")]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
lengths = {u.split(":")[0]: length(int(u.split(":")[0])) for u in units}
for unit in sorted(units, key=lambda u: -lengths[u.split(":")[0]]):
    print(unit)
EOF
[ "$(wc -l < "$QUEUE")" -eq "${#UNITS[@]}" ] || { echo "ERROR: route queue $QUEUE holds $(wc -l < "$QUEUE") of ${#UNITS[@]} route runs"; exit 1; }
: > "$QUEUE.lock"
echo "${#ROUTE_IDS[@]} routes x ${#SEED_LIST[@]} seeds = ${#UNITS[@]} route runs over $WORKERS workers on $NUM_GPUS GPUs, policy server threads $OMP_THREADS"
echo "  queue, longest first: $(tr '\n' ' ' < "$QUEUE")"
[ -f "$OUT/timing.csv" ] || echo "route,worker,gpu,attempt,start,wall_s,game_s,pace,status,driving_score,seed" > "$OUT/timing.csv"
TIMING_ROWS_BEFORE=$(($(wc -l < "$OUT/timing.csv") - 1))

log_route_timing() {  # $1 worker, $2 gpu, $3 route id, $4 attempt, $5 start epoch, $6 result json, $7 seed
    "$SERVER_PY" - "$@" "$(date +%s)" "$OUT/timing.csv" <<'EOF'
import json, sys, time
w, gpu, route_id, attempt, t0, path, seed, t1, csv_path = sys.argv[1:10]
rec = json.load(open(path))["_checkpoint"]["records"][0]
wall = int(t1) - int(t0); game = float(rec["meta"]["duration_game"]); ds = float(rec["scores"]["score_composed"])
pace = game / wall if wall > 0 else 0.0
with open(csv_path, "a") as f:
    f.write(f"{route_id},{w},{gpu},{attempt},{time.strftime('%H:%M:%S', time.localtime(int(t0)))},{wall},{game:.1f},{pace:.3f},{rec['status']},{ds:.2f},{seed}\n")
print(f"[w{w} gpu{gpu} route{route_id}{' s' + seed if seed else ''}] done: {wall} s wall, {game:.0f} s game, {pace:.2f}x real time, DS {ds:.1f}, {rec['status']}")
EOF
}

run_route() {  # $1 worker, $2 gpu, $3 route id, $4 attempt, $5 seed: one CARLA server + one evaluator run, into $OUT/routes/route_<id>_s<seed>
    local w=$1 gpu=$2 route_id=$3 attempt=$4 seed=$5
    local route_dir=$OUT/routes/$(unit_dir "$route_id" "$seed")
    local seed_args=()
    [ -n "$seed" ] && seed_args=(--trafficManagerSeed "$seed")   # leaderboard 1.0 spelling; empty seed = the evaluator's default (runs before the seeds layout)
    local routes_xml=$ROUTE_DIR/longest_weathers_$route_id.xml
    local port=$((PORT_BASE + w * 50)) tm_port=$((PORT_BASE + 6000 + w * 50))
    local server_log=$route_dir/carla_server_attempt$attempt.log evaluator_log=$route_dir/evaluator_attempt$attempt.log
    local t_start
    t_start=$(date +%s)
    mkdir -p "$route_dir"
    rm -f "$route_dir/result.json"
    local pin
    pin=$(numa_pin_for_gpu "$gpu")
    # offscreen OpenGL on the worker's GPU (UE 4.24 has no -RenderOffScreen/-nullrhi/-graphicsadapter)
    $pin env DISPLAY= SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE="$gpu" "$CARLA_ROOT/CarlaUE4.sh" -opengl -nosound \
        -carla-rpc-port="$port" -carla-streaming-port=$((port + 1)) > "$server_log" 2>&1 &
    local server_pid=$! up=1
    echo "$server_pid" >> "$OUT/server_pids"
    for _ in $(seq 1 60); do
        kill -0 "$server_pid" 2>/dev/null || { echo "[w$w gpu$gpu route$route_id${seed:+ s$seed}] CARLA server died"; break; }
        "$GARAGE_PY" -c "
import carla, sys
try:
    c = carla.Client('localhost', $port); c.set_timeout(10.0); c.get_world().get_map()
except Exception:
    sys.exit(1)" 2>/dev/null && up=0 && break
        sleep 5
    done
    if [ $up -ne 0 ]; then
        # reaped here: a dead unreaped server makes the next attempt's wait -n return at once and kill its evaluator
        pkill -9 -P "$server_pid" 2>/dev/null; kill -9 "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null
        echo "[w$w gpu$gpu route$route_id${seed:+ s$seed}] last lines of $server_log:"; tail -n 20 "$server_log"
        return 1
    fi
    local log_env=(ROUTES="$routes_xml")
    if [ "$LOGGING" = "1" ]; then
        log_env+=(COSIM_TELEMETRY="$route_dir/telemetry" COSIM_WORLD_LOG="$route_dir/world_log")
        [ "$CARLA_VIEW" = "1" ] && log_env+=(COSIM_DEBUG_CARLA_VIEW="$route_dir/carla_view")
        [ "$OBS_HTML" != "0" ] && log_env+=(COSIM_OBS_HTML="$route_dir/obs_html")
    fi
    $pin env CUDA_VISIBLE_DEVICES="$gpu" "${log_env[@]}" \
        "$GARAGE_PY" -u "$EVALUATOR" --routes "$routes_xml" --scenarios "$SCENARIO_FILE" --repetitions "$REPETITIONS" \
        --agent "$AGENT" --agent-config "$CKPT" --checkpoint "$route_dir/result.json" --track MAP \
        --port "$port" --trafficManagerPort "$tm_port" --timeout 600 "${seed_args[@]}" \
        > "$evaluator_log" 2>&1 &
    local evaluator_pid=$!
    wait -n 2>/dev/null  # the evaluator, or the server when it dies first (render-thread crash a minute in)
    if kill -0 "$evaluator_pid" 2>/dev/null; then
        echo "[w$w gpu$gpu route$route_id${seed:+ s$seed}] CARLA server died mid-route; stopping the evaluator (600 s client timeouts otherwise)"
        pkill -9 -P "$evaluator_pid" 2>/dev/null; kill -9 "$evaluator_pid" 2>/dev/null; wait "$evaluator_pid" 2>/dev/null
    fi
    # children first: killing the wrapper shell first reparents the UE4 binary and leaves it running
    pkill -9 -P "$server_pid" 2>/dev/null; kill -9 "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null
    if ! route_done "$route_dir/result.json"; then
        echo "[w$w gpu$gpu route$route_id${seed:+ s$seed}] no valid route record; last lines of $evaluator_log:"; tail -n 20 "$evaluator_log"
        return 1
    fi
    log_route_timing "$w" "$gpu" "$route_id" "$attempt" "$t_start" "$route_dir/result.json" "$seed"
    return 0
}

next_unit() {  # pops the longest remaining route run off $QUEUE under its lock; status 1 once the queue is empty
    (
        flock 9 || exit 1
        unit=$(head -n 1 "$QUEUE")
        [ -n "$unit" ] || exit 1
        tail -n +2 "$QUEUE" > "$QUEUE.next" && mv "$QUEUE.next" "$QUEUE" || exit 1
        echo "$unit"
    ) 9>>"$QUEUE.lock"
}

run_worker() {  # $1 worker: pulls route runs from the queue until it is empty, each retried up to MAX_ATTEMPTS on the next GPU
    local w=$1 unit route_id seed attempt gpu
    while unit=$(next_unit); do
        route_id=${unit%%:*}; seed=${unit##*:}
        for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
            gpu=$(( (w + attempt - 1) % NUM_GPUS ))
            echo "[w$w] route $route_id${seed:+ seed $seed} attempt $attempt gpu $gpu $(date +%H:%M:%S)"
            run_route "$w" "$gpu" "$route_id" "$attempt" "$seed" && break
            echo "[w$w] route $route_id${seed:+ seed $seed} attempt $attempt failed"
        done
    done
    echo "[w$w] queue empty $(date +%H:%M:%S)"
}

# only the servers this job started (children first: the UE4 binary outlives its wrapper shell otherwise)
trap 'for p in $(cat "$OUT/server_pids" 2>/dev/null); do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done' EXIT
WORKER_PIDS=()
ROUTES_T0=$(date +%s)
for ((w = 0; w < WORKERS; w++)); do
    [ "$w" -gt 0 ] && sleep "$STAGGER_S"  # simultaneous engine starts are the one thing the crashed runs shared
    run_worker "$w" > "$OUT/worker_w$w.log" 2>&1 &
    WORKER_PIDS+=($!)
done
wait ${WORKER_PIDS[@]+"${WORKER_PIDS[@]}"}
ROUTES_WALL_S=$(( $(date +%s) - ROUTES_T0 ))
rm -f "$QUEUE" "$QUEUE.lock"
cat "$OUT"/worker_w*.log 2>/dev/null
"$SERVER_PY" - "$OUT/timing.csv" "$ROUTES_WALL_S" "$WORKERS" "$NUM_GPUS" "${#UNITS[@]}" "$TIMING_ROWS_BEFORE" <<'EOF' | tee -a "$OUT/run_info.txt"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]))); wall, workers, gpus, n_routes, rows_before = (int(x) for x in sys.argv[2:7])
rows = rows[rows_before:]  # a completion appends to the previous run's timing.csv
if not rows:
    print(f"speed: no route finished in {wall / 3600:.2f} h wall ({workers} workers on {gpus} GPUs)"); sys.exit(0)
game = sum(float(r["game_s"]) for r in rows); route_wall = sum(int(r["wall_s"]) for r in rows)
per_worker = {}
for r in rows:
    per_worker[r["worker"]] = per_worker.get(r["worker"], 0) + int(r["wall_s"])
print(f"speed: {len(rows)}/{n_routes} route runs in {wall / 3600:.2f} h wall, {workers} workers on {gpus} GPUs; game time {game / 3600:.2f} h; "
      f"per-route pace {game / route_wall:.2f}x real time; busiest worker {max(per_worker.values()) / 3600:.2f} h, "
      f"idlest {min(per_worker.values()) / 3600:.2f} h; route runs/h {len(rows) / (wall / 3600):.1f}")
EOF

# garage's parser over the valid route records, one subfolder per seed -> $OUT/aggregate/results.csv (crash records abort it)
for f in "$OUT"/routes/route_*/result.json "$OUT"/routes/route_*.result.json; do   # route folders, or the flat records of a tidied run
    [ -e "$f" ] || continue
    name=$(basename "$f" .result.json); [ "$name" = "result.json" ] && name=$(basename "$(dirname "$f")")
    case $name in *_s*) seed=${name##*_s}; name=${name%_s*} ;; *) seed= ;; esac
    route_id=$(echo "$name" | sed 's/route_0*//; s/^$/0/')
    dest=$(unit_json "$route_id" "$seed"); mkdir -p "$(dirname "$dest")"
    route_done "$f" && cp "$f" "$dest" && continue
    echo "route $route_id${seed:+ seed $seed} has no valid record (all attempts crashed): left out of results.csv"
done
# the garage python with the egg; the fallback server python has its own carla and must not see the py3.7 egg
PARSER_ARGS=(--xml "$ROUTES_XML" --results "$OUT/aggregate" --log_dir "$OUT/aggregate")
env PYTHONPATH="$CARLA_EGG" "$GARAGE_PY" "$GARAGE_WORK_DIR/tools/result_parser.py" "${PARSER_ARGS[@]}" \
    || env -u PYTHONPATH "$SERVER_PY" "$GARAGE_WORK_DIR/tools/result_parser.py" "${PARSER_ARGS[@]}" \
    || echo "result_parser.py failed with $GARAGE_PY and $SERVER_PY (per-route jsons in $OUT/aggregate are unaffected)"
"$SERVER_PY" - "$OUT/aggregate" "$TOTAL_UNITS" "**/longest_weathers_*.json" <<'EOF'
import glob, json, os, re, statistics, sys
rows = []  # (seed, route_id, status, DS, RC, IP); seed "" = a run from before the seeds layout
for f in glob.glob(os.path.join(sys.argv[1], sys.argv[3]), recursive=True):
    parent = os.path.basename(os.path.dirname(f))
    seed = parent[1:] if parent.startswith("s") and parent[1:].isdigit() else ""
    for r in json.load(open(f))["_checkpoint"]["records"]:
        rows.append((seed, r["route_id"], r["status"], r["scores"]["score_composed"], r["scores"]["score_route"], r["scores"]["score_penalty"]))
def route_index(route_id):
    m = re.search(r"_(\d+)", route_id)
    return int(m.group(1)) if m else 0
rows.sort(key=lambda r: (int(r[0]) if r[0] else -1, route_index(r[1])))
print(f"\n{len(rows)} route records of {sys.argv[2]} route runs")
for seed, route_id, status, ds, rc, ip in rows:
    print(f"  {('s' + seed) if seed else '-':>3s} {route_id:22s} {status:34s} DS {ds:6.2f} RC {rc:6.2f} IP {ip:5.3f}")
if rows:
    n = len(rows)
    print(f"mean DS {sum(r[3] for r in rows)/n:.2f}  RC {sum(r[4] for r in rows)/n:.2f}  IP {sum(r[5] for r in rows)/n:.3f}  (all {n} route runs)")
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r[0], []).append(r)
    if len(by_seed) > 1:  # the paper-style number: the benchmark DS per seed, mean +- std over the seeds
        ds = {s: sum(r[3] for r in v) / len(v) for s, v in by_seed.items()}
        rc = {s: sum(r[4] for r in v) / len(v) for s, v in by_seed.items()}
        for s in sorted(by_seed, key=lambda s: int(s) if s else -1):
            print(f"seed {s}: DS {ds[s]:.2f}  RC {rc[s]:.2f}  ({len(by_seed[s])} routes)")
        print(f"over seeds: DS {statistics.mean(ds.values()):.2f} +- {statistics.pstdev(ds.values()):.2f}  "
              f"RC {statistics.mean(rc.values()):.2f} +- {statistics.pstdev(rc.values()):.2f}  (population std over {len(by_seed)} seeds, as in results.csv)")
EOF
# the repo .venv holds wandb; the script reads results.csv and the route records and never imports opengigaflow
if [ "$WANDB_REPORT" = "1" ] && [ "$SCENARIOS" = "1" ]; then
    env -u PYTHONPATH "$PD/.venv/bin/python" "$PD/opengigaflow/ocean/evaluation_utils/eval_wandb_report.py" \
        longest6_v1 "$OUT/aggregate/results.csv" --run-dir "$(dirname "$CKPT")" --expected-episode-count $((TOTAL_UNITS * REPETITIONS)) \
        || echo "wandb report failed (results above are unaffected)"
fi

REPORT_ARGS=()
[ "$OBS_HTML" != "0" ] && REPORT_ARGS+=(--obs-html-dir "$OUT/obs_html")
if [ "$REPORT" = "1" ] && [ "$LOGGING" = "1" ]; then
    env -u PYTHONPATH "$SERVER_PY" "$PD/scripts/eval/analyze_carla_cosim.py" "$OUT"/routes/route_* "$OUT/report" "${REPORT_ARGS[@]}" \
        || echo "HTML report failed (results above are unaffected)"
fi
# after the report: the gallery links to it when it exists
if [ "$OBS_HTML" != "0" ] && [ "$LOGGING" = "1" ]; then
    env -u PYTHONPATH "$SERVER_PY" "$PD/scripts/eval/render_carla_obs_html.py" "$OUT" --routes "$ROUTES_XML" --select "$OBS_HTML" \
        || echo "obs replay gallery failed (results above are unaffected)"
fi
# the pages are rendered: flatten the route folders, keep the logs of the failed (OBS_HTML=all: every) route runs
if [ "$TIDY" = "1" ]; then
    env -u PYTHONPATH "$SERVER_PY" "$PD/scripts/eval/tidy_carla_run.py" "$OUT" --keep "$([ "$OBS_HTML" = "all" ] && echo all || echo failures)" \
        || echo "tidy failed (results above are unaffected)"
fi
[ -f "$OUT/report/index.html" ] && echo "report -> $OUT/report/index.html"
[ -f "$OUT/obs_html/index.html" ] && echo "obs replays -> $OUT/obs_html/index.html"
for results in "$OUT/results.csv" "$OUT/aggregate/results.csv"; do [ -f "$results" ] && { echo "results -> $results"; break; }; done
echo "Done -> $OUT"
