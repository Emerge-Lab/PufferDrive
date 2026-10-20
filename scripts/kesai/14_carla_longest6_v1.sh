#!/bin/bash
#SBATCH --job-name eval_longest6_v1
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/log_%j.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/log_%j.err
#SBATCH --partition dev
# CARLA longest6 v1 evaluation of a PufferDrive checkpoint through carla_garage's unmodified Leaderboard 1.0
# (CARLA 0.9.10.1, pufferlib/ocean/cosim/carla/lb1/leaderboard_agent.py), the Leaderboard 1.0 counterpart of
# 11_carla_longest6_v2.sh. The garage evaluator runs in the `garage` conda env (Python 3.7, the CARLA 0.9.10 egg);
# the agent starts pufferlib/ocean/cosim/carla/lb1/policy_server.py in a PufferDrive python ($SERVER_PY) per route.
#
# WORKERS parallel workers (default 16), each one CARLA server + one evaluator at a time, spread round-robin over the
# GPUs of the allocation (1-8, e.g. `sbatch --gres gpu:2`): a server needs ~2 cores and no GPU once the agent has
# turned the spectator render off, so several share a GPU. The 36 routes (one split xml each) go longest-first to
# the least loaded worker (route length from the xml), every route gets its own server (restarted per route),
# worker launches are staggered, a crashed route is retried on the next GPU, and the per-route result jsons are
# aggregated at the end with garage's tools/result_parser.py (results.csv) plus the nuPlan-style HTML report
# ($OUT/report/index.html) and the obs replay gallery of all routes ($OUT/obs_html/index.html, as in
# 11_carla_longest6_v2.sh). Speed: wall and game time of every route go to $OUT/timing.csv, the run totals to
# run_info.txt and stdout.
#
# Overridable env: RUN_DIR, ROUTE_DIR (the split xmls), ROUTES_XML (the combined longest6.xml, for the parser
# and the gallery), SCENARIOS (1 = eval_scenarios.json, 0 = no_scenarios.json), REPETITIONS, ROUTE_SUBSET
# ("0 5 17": only these route ids), WORKERS, NUM_GPUS, STAGGER_S (15: seconds between worker launches), PORT_BASE
# (CARLA rpc port of worker 0; worker w uses PORT_BASE+50w, TM PORT_BASE+6000+50w), OMP_NUM_THREADS (policy server
# threads; 2 from 16 workers on, else 4), LOGGING (1, default: telemetry, world log and the HTML report per route; 0: scores
# only), CARLA_VIEW (0, default: no chase-cam; 1: chase-cam mp4 per route, needs LOGGING=1), OBS_HTML (1 = also
# the interactive obs replay per route, rendered into $OUT/obs_html at the end, large; needs LOGGING=1), REPORT
# (0 = skip the HTML report), MAX_ATTEMPTS, CARLA_ROOT, GARAGE_WORK_DIR, GARAGE_PY, SERVER_PY, PD,
# COSIM_MAX_SPEED_MPS (ego speed cap, default 30), COSIM_ZERO_PARTNER_STOPPED_TIME (default 1),
# COSIM_VRU_SIZE_PAD_M (default 0.8: added to the length and width of every walker and two-wheeler box, so they read
# strictly larger than the smallest training vehicle; 0 = true boxes), LIBTIFF5_COMPAT (1: when `import carla` fails for
# the egg's missing libtiff.so.5, try the garage env's lib dir (`conda install -n garage "libtiff<4.5"`), then link the
# system libtiff.so.6 under that name; 0: never link).
#
# Completing a partial run (routes lost to CARLA crashes or the job time limit): COMPLETE_RUN=<existing $OUT folder>
# reruns, in place, only its routes without a valid record (no result.json, or a crash status the leaderboard would
# resubmit; ROUTE_SUBSET narrows the candidates, deleting a route's result.json forces it). SCENARIOS, REPETITIONS and
# COSIM_VRU_SIZE_PAD_M default to the run's own values from run_info.txt, CKPT must be the run's checkpoint. The replaced
# route folders and worker logs move to $OUT/pre_completion_<stamp>/, timing.csv grows, run_info.txt gets a completion
# line, and results.csv, the report and the gallery are rebuilt over all routes.
# scripts/kesai/17_carla_longest6_v1_complete_local.sh does this on a desktop for a run folder synced from the cluster.
set -u

export PD=${PD:-/home/bjaeger/PufferDrive}
export GARAGE_PY=${GARAGE_PY:-/home/bjaeger/miniconda3/envs/garage/bin/python}   # cp37: the CARLA 0.9.10 egg
export SERVER_PY=${SERVER_PY:-/home/bjaeger/miniconda3/envs/carl/bin/python}     # any PufferDrive-capable python
export CARLA_ROOT=${CARLA_ROOT:-/home/bjaeger/CARLA_0.9.10}
export GARAGE_WORK_DIR=${GARAGE_WORK_DIR:-/home/bjaeger/carla_garage_1}
RUN_DIR=${RUN_DIR:-/home/bjaeger/PufferDrive/experiments/k_scaled_0045_1000}
# the agent finds config.yaml next to final_model.pt (or one level above a models/*.pt)
export CKPT=${CKPT:-$RUN_DIR/final_model.pt}
ROUTE_DIR=${ROUTE_DIR:-$GARAGE_WORK_DIR/leaderboard/data/longest6_split}
ROUTES_XML=${ROUTES_XML:-$GARAGE_WORK_DIR/leaderboard/data/longest6.xml}
COMPLETE_RUN=${COMPLETE_RUN:-}
if [ -n "$COMPLETE_RUN" ]; then
    [ -d "$COMPLETE_RUN/routes" ] && [ -f "$COMPLETE_RUN/run_info.txt" ] || { echo "COMPLETE_RUN=$COMPLETE_RUN is not a run folder (needs routes/ and run_info.txt)"; exit 1; }
    run_info_value() { awk -v key="$1" '$1 == key {print $2; exit}' "$COMPLETE_RUN/run_info.txt"; }
    # the run's own settings, so the completed routes are scored like the rest
    PREVIOUS_SCENARIO_FILE=$(run_info_value scenarios)
    [ -n "$PREVIOUS_SCENARIO_FILE" ] && SCENARIOS=${SCENARIOS:-$([ "$(basename "$PREVIOUS_SCENARIO_FILE")" = "no_scenarios.json" ] && echo 0 || echo 1)}
    REPETITIONS=${REPETITIONS:-$(run_info_value repetitions)}
    COSIM_VRU_SIZE_PAD_M=${COSIM_VRU_SIZE_PAD_M:-$(run_info_value vru_size_pad_m)}
fi
SCENARIOS=${SCENARIOS:-1}
if [ "$SCENARIOS" = "1" ]; then
    SCENARIO_FILE=$GARAGE_WORK_DIR/leaderboard/data/scenarios/eval_scenarios.json
else
    SCENARIO_FILE=$GARAGE_WORK_DIR/leaderboard/data/scenarios/no_scenarios.json
fi
REPETITIONS=${REPETITIONS:-1}
ROUTE_SUBSET=${ROUTE_SUBSET:-}
PORT_BASE=${PORT_BASE:-2000}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
WORKERS=${WORKERS:-16}
STAGGER_S=${STAGGER_S:-15}
LOGGING=${LOGGING:-1}
CARLA_VIEW=${CARLA_VIEW:-0}
[ "$CARLA_VIEW" = "1" ] && [ "$LOGGING" != "1" ] && { echo "CARLA_VIEW=1 needs LOGGING=1"; exit 1; }
OBS_HTML=${OBS_HTML:-1}
REPORT=${REPORT:-$LOGGING}
export COSIM_VRU_SIZE_PAD_M=${COSIM_VRU_SIZE_PAD_M:-0.8}
EVALUATOR=$GARAGE_WORK_DIR/leaderboard/leaderboard/leaderboard_evaluator_local.py
CARLA_EGG=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
AGENT=$PD/pufferlib/ocean/cosim/carla/lb1/leaderboard_agent.py
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

TAG=longest6_v1$([ "$SCENARIOS" = "1" ] || echo "_noscen")$([ "$REPETITIONS" = "1" ] || echo "_rep$REPETITIONS")
TAG=$TAG$([ "$COSIM_VRU_SIZE_PAD_M" = "0" ] || echo "_vrupad$COSIM_VRU_SIZE_PAD_M")
# results live in the model's own eval folder, next to the PufferDrive benchmark evals
OUT=${COMPLETE_RUN:-$RUN_DIR/eval/carla_${TAG}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}}
mkdir -p "$OUT/routes" "$OUT/aggregate"
: > "$OUT/server_pids"
GIT_STATE=$(git -C "$PD" rev-parse --short HEAD 2>/dev/null)$(git -C "$PD" diff --quiet 2>/dev/null || echo " dirty")
if [ -z "$COMPLETE_RUN" ]; then
    {
        echo "checkpoint $CKPT"; echo "routes $ROUTE_DIR ($ROUTES_XML)"; echo "scenarios $SCENARIO_FILE"
        echo "repetitions $REPETITIONS"; echo "gpus $NUM_GPUS"; echo "vru_size_pad_m $COSIM_VRU_SIZE_PAD_M"
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
export COSIM_DYNAMICS_SOURCE=pufferdrive
export COSIM_MAX_SPEED_MPS=${COSIM_MAX_SPEED_MPS:-30}
export COSIM_ZERO_PARTNER_STOPPED_TIME=${COSIM_ZERO_PARTNER_STOPPED_TIME:-1}
export COSIM_OBS_HTML_MAX_STEPS=${COSIM_OBS_HTML_MAX_STEPS:-20000}
export COSIM_OBS_HTML_RENDER=0  # routes save the compact replay only; render_carla_obs_html.py renders all pages into $OUT/obs_html

mkdir -p "$PD/experiments/logs"  # the build helper keeps its source hash there
PATH="$(dirname "$SERVER_PY"):$PATH" bash "$PD/scripts/kesai/build_ext_if_changed.sh" "$PD" || exit 1
IMPORTED_CORE=$(env -u PYTHONPATH PYTHONPATH="$PD" "$SERVER_PY" -c "import pufferlib.ocean.cosim.carla.shadow_ego as s; print(s.__file__)") \
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
if [ -n "$COMPLETE_RUN" ]; then
    PENDING=()
    for route_id in "${ROUTE_IDS[@]}"; do
        route_done "$OUT/routes/route_$(printf "%02d" "$route_id")/result.json" || PENDING+=("$route_id")
    done
    [ "${#PENDING[@]}" -gt 0 ] || { echo "nothing to complete: every route of $OUT has a valid record"; exit 0; }
    echo "completing $OUT: ${#PENDING[@]} of $TOTAL_ROUTES routes without a valid record: ${PENDING[*]}"
    ROUTE_IDS=("${PENDING[@]}")
    # what the completion replaces stays readable next to the live results
    COMPLETION_STAMP=$(date +%Y%m%d_%H%M%S)
    PREVIOUS=$OUT/pre_completion_$COMPLETION_STAMP
    mkdir -p "$PREVIOUS/routes"
    for route_id in "${ROUTE_IDS[@]}"; do
        route_name=route_$(printf "%02d" "$route_id")
        [ -d "$OUT/routes/$route_name" ] && mv "$OUT/routes/$route_name" "$PREVIOUS/routes/"
        rm -f "$OUT/aggregate/longest_weathers_$route_id.json"
    done
    for f in "$OUT"/worker_w*.log; do [ -e "$f" ] && mv "$f" "$PREVIOUS/"; done
    echo "completion $COMPLETION_STAMP on $(hostname): routes ${ROUTE_IDS[*]}; checkpoint $CKPT; scenarios $SCENARIO_FILE; repetitions $REPETITIONS; vru_size_pad_m $COSIM_VRU_SIZE_PAD_M; gpus $NUM_GPUS; git $GIT_STATE" >> "$OUT/run_info.txt"
fi
[ "$WORKERS" -gt "${#ROUTE_IDS[@]}" ] && WORKERS=${#ROUTE_IDS[@]}
# longest route first to the least loaded worker; length = the xml's waypoint polyline (an earlier run's duration would
# read a route that failed early as short)
mapfile -t ASSIGNMENT < <("$SERVER_PY" - "$ROUTE_DIR" "$WORKERS" "${ROUTE_IDS[@]}" <<'EOF'
import math, sys, xml.etree.ElementTree as ET
route_dir, workers, ids = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3:]]
SETUP_M = 300.0  # server start + world load per route, in route-length-equivalent metres
def length(rid):
    pts = [(float(w.get("x")), float(w.get("y"))) for w in ET.parse(f"{route_dir}/longest_weathers_{rid}.xml").iter("waypoint")]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
loads = [0.0] * workers; assign = [[] for _ in range(workers)]
for rid in sorted(ids, key=lambda r: -length(r)):
    w = min(range(workers), key=lambda i: loads[i]); assign[w].append(rid); loads[w] += length(rid) + SETUP_M
for w in range(workers):
    print(w, " ".join(map(str, assign[w])), f"# {loads[w] / 1000:.1f} km")
EOF
)
echo "${#ROUTE_IDS[@]} routes over $WORKERS workers on $NUM_GPUS GPUs, policy server threads $OMP_THREADS"
printf "  worker %s\n" "${ASSIGNMENT[@]}"
[ -f "$OUT/timing.csv" ] || echo "route,worker,gpu,attempt,start,wall_s,game_s,pace,status,driving_score" > "$OUT/timing.csv"
TIMING_ROWS_BEFORE=$(($(wc -l < "$OUT/timing.csv") - 1))

log_route_timing() {  # $1 worker, $2 gpu, $3 route id, $4 attempt, $5 start epoch, $6 result json
    "$SERVER_PY" - "$@" "$(date +%s)" "$OUT/timing.csv" <<'EOF'
import json, sys, time
w, gpu, route_id, attempt, t0, path, t1, csv_path = sys.argv[1:9]
rec = json.load(open(path))["_checkpoint"]["records"][0]
wall = int(t1) - int(t0); game = float(rec["meta"]["duration_game"]); ds = float(rec["scores"]["score_composed"])
pace = game / wall if wall > 0 else 0.0
with open(csv_path, "a") as f:
    f.write(f"{route_id},{w},{gpu},{attempt},{time.strftime('%H:%M:%S', time.localtime(int(t0)))},{wall},{game:.1f},{pace:.3f},{rec['status']},{ds:.2f}\n")
print(f"[w{w} gpu{gpu} route{route_id}] done: {wall} s wall, {game:.0f} s game, {pace:.2f}x real time, DS {ds:.1f}, {rec['status']}")
EOF
}

run_route() {  # $1 worker, $2 gpu, $3 route id, $4 attempt: one CARLA server + one evaluator run, into $OUT/routes/route_<id>
    local w=$1 gpu=$2 route_id=$3 attempt=$4
    local route_dir=$OUT/routes/route_$(printf "%02d" "$route_id")
    local routes_xml=$ROUTE_DIR/longest_weathers_$route_id.xml
    local port=$((PORT_BASE + w * 50)) tm_port=$((PORT_BASE + 6000 + w * 50))
    local server_log=$route_dir/carla_server_attempt$attempt.log evaluator_log=$route_dir/evaluator_attempt$attempt.log
    local t_start
    t_start=$(date +%s)
    mkdir -p "$route_dir"
    rm -f "$route_dir/result.json"
    # offscreen OpenGL on the worker's GPU (UE 4.24 has no -RenderOffScreen/-nullrhi/-graphicsadapter)
    DISPLAY= SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE="$gpu" "$CARLA_ROOT/CarlaUE4.sh" -opengl -nosound \
        -carla-rpc-port="$port" -carla-streaming-port=$((port + 1)) > "$server_log" 2>&1 &
    local server_pid=$! up=1
    echo "$server_pid" >> "$OUT/server_pids"
    for _ in $(seq 1 60); do
        kill -0 "$server_pid" 2>/dev/null || { echo "[w$w gpu$gpu route$route_id] CARLA server died"; break; }
        "$GARAGE_PY" -c "
import carla, sys
try:
    c = carla.Client('localhost', $port); c.set_timeout(10.0); c.get_world().get_map()
except Exception:
    sys.exit(1)" 2>/dev/null && up=0 && break
        sleep 5
    done
    if [ $up -ne 0 ]; then
        pkill -9 -P "$server_pid" 2>/dev/null; kill -9 "$server_pid" 2>/dev/null
        echo "[w$w gpu$gpu route$route_id] last lines of $server_log:"; tail -n 20 "$server_log"
        return 1
    fi
    local log_env=(ROUTES="$routes_xml")
    if [ "$LOGGING" = "1" ]; then
        log_env+=(COSIM_TELEMETRY="$route_dir/telemetry" COSIM_WORLD_LOG="$route_dir/world_log")
        [ "$CARLA_VIEW" = "1" ] && log_env+=(COSIM_DEBUG_CARLA_VIEW="$route_dir/carla_view")
        [ "$OBS_HTML" = "1" ] && log_env+=(COSIM_OBS_HTML="$route_dir/obs_html")
    fi
    env CUDA_VISIBLE_DEVICES="$gpu" "${log_env[@]}" \
        "$GARAGE_PY" -u "$EVALUATOR" --routes "$routes_xml" --scenarios "$SCENARIO_FILE" --repetitions "$REPETITIONS" \
        --agent "$AGENT" --agent-config "$CKPT" --checkpoint "$route_dir/result.json" --track MAP \
        --port "$port" --trafficManagerPort "$tm_port" --timeout 600 \
        > "$evaluator_log" 2>&1 &
    local evaluator_pid=$!
    wait -n 2>/dev/null  # the evaluator, or the server when it dies first (render-thread crash a minute in)
    if kill -0 "$evaluator_pid" 2>/dev/null; then
        echo "[w$w gpu$gpu route$route_id] CARLA server died mid-route; stopping the evaluator (600 s client timeouts otherwise)"
        pkill -9 -P "$evaluator_pid" 2>/dev/null; kill -9 "$evaluator_pid" 2>/dev/null; wait "$evaluator_pid" 2>/dev/null
    fi
    # children first: killing the wrapper shell first reparents the UE4 binary and leaves it running
    pkill -9 -P "$server_pid" 2>/dev/null; kill -9 "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null
    if ! route_done "$route_dir/result.json"; then
        echo "[w$w gpu$gpu route$route_id] no valid route record; last lines of $evaluator_log:"; tail -n 20 "$evaluator_log"
        return 1
    fi
    log_route_timing "$w" "$gpu" "$route_id" "$attempt" "$t_start" "$route_dir/result.json"
    return 0
}

run_worker() {  # $1 worker: its assigned routes, sequentially, each retried up to MAX_ATTEMPTS on the next GPU
    local w=$1 ids route_id attempt gpu
    ids=${ASSIGNMENT[w]#* }
    ids=${ids%%#*}
    for route_id in $ids; do
        for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
            gpu=$(( (w + attempt - 1) % NUM_GPUS ))
            echo "[w$w] route $route_id attempt $attempt gpu $gpu $(date +%H:%M:%S)"
            run_route "$w" "$gpu" "$route_id" "$attempt" && break
            echo "[w$w] route $route_id attempt $attempt failed"
        done
    done
}

# never attach to a server that is not ours (a peer's test server, a leftover of an earlier job)
if command -v ss >/dev/null 2>&1; then
    for ((w = 0; w < WORKERS; w++)); do
        for port in $((PORT_BASE + w * 50)) $((PORT_BASE + w * 50 + 1)) $((PORT_BASE + 6000 + w * 50)); do
            [ -z "$(ss -Hltn "sport = :$port")" ] || { echo "port $port is already in use (ss -ltn); pick another PORT_BASE"; exit 1; }
        done
    done
fi
# only the servers this job started (children first: the UE4 binary outlives its wrapper shell otherwise)
trap 'for p in $(cat "$OUT/server_pids" 2>/dev/null); do pkill -9 -P "$p" 2>/dev/null; kill -9 "$p" 2>/dev/null; done' EXIT
WORKER_PIDS=()
ROUTES_T0=$(date +%s)
for ((w = 0; w < WORKERS; w++)); do
    [ "$w" -gt 0 ] && sleep "$STAGGER_S"  # simultaneous engine starts are the one thing the crashed runs shared
    run_worker "$w" > "$OUT/worker_w$w.log" 2>&1 &
    WORKER_PIDS+=($!)
done
wait "${WORKER_PIDS[@]}"
ROUTES_WALL_S=$(( $(date +%s) - ROUTES_T0 ))
cat "$OUT"/worker_w*.log
"$SERVER_PY" - "$OUT/timing.csv" "$ROUTES_WALL_S" "$WORKERS" "$NUM_GPUS" "${#ROUTE_IDS[@]}" "$TIMING_ROWS_BEFORE" <<'EOF' | tee -a "$OUT/run_info.txt"
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]))); wall, workers, gpus, n_routes, rows_before = (int(x) for x in sys.argv[2:7])
rows = rows[rows_before:]  # a completion appends to the previous run's timing.csv
if not rows:
    print(f"speed: no route finished in {wall / 3600:.2f} h wall ({workers} workers on {gpus} GPUs)"); sys.exit(0)
game = sum(float(r["game_s"]) for r in rows); route_wall = sum(int(r["wall_s"]) for r in rows)
per_worker = {}
for r in rows:
    per_worker[r["worker"]] = per_worker.get(r["worker"], 0) + int(r["wall_s"])
print(f"speed: {len(rows)}/{n_routes} routes in {wall / 3600:.2f} h wall, {workers} workers on {gpus} GPUs; game time {game / 3600:.2f} h; "
      f"per-route pace {game / route_wall:.2f}x real time; busiest worker {max(per_worker.values()) / 3600:.2f} h, "
      f"idlest {min(per_worker.values()) / 3600:.2f} h; routes/h {len(rows) / (wall / 3600):.1f}")
EOF

# aggregate: garage's parser over the per-route jsons -> $OUT/aggregate/results.csv, plus a per-route table; a crash
# record has no global_record and would abort the parser
for f in "$OUT"/routes/route_*/result.json; do
    route_id=$(basename "$(dirname "$f")" | sed 's/route_0*//; s/^$/0/')
    route_done "$f" && cp "$f" "$OUT/aggregate/longest_weathers_$route_id.json" && continue
    echo "route $route_id has no valid record (all attempts crashed): left out of results.csv"
done
# garage's parser imports torch/pygame and carla: the garage python gets the egg; the server python (the fallback where
# the garage env's torch does not load) has its own carla and must not see the py3.7 egg
PARSER_ARGS=(--xml "$ROUTES_XML" --results "$OUT/aggregate" --log_dir "$OUT/aggregate")
env PYTHONPATH="$CARLA_EGG" "$GARAGE_PY" "$GARAGE_WORK_DIR/tools/result_parser.py" "${PARSER_ARGS[@]}" \
    || env -u PYTHONPATH "$SERVER_PY" "$GARAGE_WORK_DIR/tools/result_parser.py" "${PARSER_ARGS[@]}" \
    || echo "result_parser.py failed with $GARAGE_PY and $SERVER_PY (per-route jsons in $OUT/aggregate are unaffected)"
"$SERVER_PY" - "$OUT/aggregate" "$TOTAL_ROUTES" <<'EOF'
import glob, json, sys
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/longest_weathers_*.json"), key=lambda p: int(p.rsplit("_", 1)[1][:-5])):
    for r in json.load(open(f))["_checkpoint"]["records"]:
        rows.append((r["route_id"], r["status"], r["scores"]["score_composed"], r["scores"]["score_route"], r["scores"]["score_penalty"]))
print(f"\n{len(rows)} route records of {sys.argv[2]} routes")
for route_id, status, ds, rc, ip in rows:
    print(f"  {route_id:22s} {status:34s} DS {ds:6.2f} RC {rc:6.2f} IP {ip:5.3f}")
if rows:
    n = len(rows)
    print(f"mean DS {sum(r[2] for r in rows)/n:.2f}  RC {sum(r[3] for r in rows)/n:.2f}  IP {sum(r[4] for r in rows)/n:.3f}")
EOF

REPORT_ARGS=()
[ "$OBS_HTML" = "1" ] && REPORT_ARGS+=(--obs-html-dir "$OUT/obs_html")
if [ "$REPORT" = "1" ] && [ "$LOGGING" = "1" ]; then
    env -u PYTHONPATH "$SERVER_PY" "$PD/scripts/eval/analyze_carla_cosim.py" "$OUT"/routes/route_* "$OUT/report" "${REPORT_ARGS[@]}" \
        || echo "HTML report failed (results above are unaffected)"
fi
# after the report: the gallery links to it when it exists
if [ "$OBS_HTML" = "1" ] && [ "$LOGGING" = "1" ]; then
    env -u PYTHONPATH "$SERVER_PY" "$PD/scripts/eval/render_carla_obs_html.py" "$OUT" --routes "$ROUTES_XML" \
        || echo "obs replay gallery failed (results above are unaffected)"
fi
[ -f "$OUT/report/index.html" ] && echo "report -> $OUT/report/index.html"
[ -f "$OUT/obs_html/index.html" ] && echo "obs replays -> $OUT/obs_html/index.html"
echo "Done -> $OUT"
