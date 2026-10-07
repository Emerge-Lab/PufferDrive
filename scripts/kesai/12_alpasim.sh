#!/bin/bash
#SBATCH --job-name eval_alpasim
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --gres gpu:8
#SBATCH --mem=1007G
#SBATCH --cpus-per-task 144
#SBATCH --output /home/bjaeger/PufferDrive/experiments/logs/log_%j.out
#SBATCH --error /home/bjaeger/PufferDrive/experiments/logs/log_%j.err
#SBATCH --partition dev
# AlpaSim closed-loop evaluation of a PufferDrive checkpoint on NuRec scenes, the AlpaSim counterpart of 9_nuPlan.sh:
# alpasim-training's teacher driver runs the policy in a shadow env of this checkout (PUFFERDRIVE_ROOT=$PD) on the
# scenes converted to PufferDrive bins; AlpaSim renders, simulates the ego and replays the logged traffic.
#
# One-time setup: alpasim-training at ALPAGYM_ROOT (`uv sync --all-packages`), the AlpaSim fork at ALPASIM_ROOT, the
# NuRec scenes and bins with 130 km/h lane limits under NUREC_ROOT (alpasim-training README steps 3-4; all-usdzs/ must hold real files),
# and the AlpaSim base image archived to ALPASIM_IMAGE_TAR (REPO=$ALPASIM_ROOT OUT_DIR=<dir of ALPASIM_IMAGE_TAR> sbatch
# --output=<log> --error=<log> $ALPAGYM_ROOT/scripts/build_alpasim_image.sbatch).
#
# Overridable env: RUN_DIR, CKPT (default RUN_DIR/final_model.pt, else the newest RUN_DIR/models/model_*.pt), PD,
# ALPAGYM_ROOT, ALPASIM_ROOT, NUREC_ROOT, BINS, USDZ_DIR, ALPASIM_IMAGE_TAR, TOPOLOGY (8gpu_64rollouts = whole node;
# 1gpu for a quick check), PARALLEL (scenes in flight: 32 on 8gpu, at most 4 on 1gpu), N_SCENES (0 = every scene that
# passed the conversion checks), SCENE_LIST (file of scene ids), GOAL_MODE (gt_time:<s> | gt | route), CAM_W/CAM_H (16:9 render size;
# the teacher never sees the cameras), RENDER_VIDEO (1 = one mp4 per scene), SIM_TIMEOUT_S (per batch of PARALLEL scenes).
set -u

export PD=${PD:-/home/bjaeger/PufferDrive}
ALPAGYM_ROOT=${ALPAGYM_ROOT:-/home/bjaeger/alpasim-training}
export ALPASIM_ROOT=${ALPASIM_ROOT:-/home/bjaeger/alpasim}
NUREC_ROOT=${NUREC_ROOT:-/home/shared/data/nurec}
BINS=${BINS:-$NUREC_ROOT/bins_130kmh}
USDZ_DIR=${USDZ_DIR:-$NUREC_ROOT/all-usdzs}
RUN_DIR=${RUN_DIR:-/home/bjaeger/PufferDrive/experiments/k_scaled_0045_1000}
TOPOLOGY=${TOPOLOGY:-8gpu_64rollouts}
case "$TOPOLOGY" in
    8gpu*) PARALLEL=${PARALLEL:-32} ;;
    *) PARALLEL=${PARALLEL:-4} ;;
esac
N_SCENES=${N_SCENES:-0}
SCENE_LIST=${SCENE_LIST:-}
GOAL_MODE=${GOAL_MODE:-gt_time:5}
CAM_W=${CAM_W:-640}
CAM_H=${CAM_H:-360}
RENDER_VIDEO=${RENDER_VIDEO:-1}
SIM_TIMEOUT_S=${SIM_TIMEOUT_S:-3600}
ALPAGYM_PY=$ALPAGYM_ROOT/.venv/bin/python
# uv: the host syncs the AlpaSim checkout's venv before it starts the Wizard
export PATH="$HOME/.local/bin:$PATH"

if [ -z "${CKPT:-}" ]; then
    CKPT=$RUN_DIR/final_model.pt
    [ -f "$CKPT" ] || CKPT=$(ls "$RUN_DIR"/models/model_*.pt 2>/dev/null | sort | tail -n 1)
fi
[ -n "$CKPT" ] || { echo "no checkpoint in $RUN_DIR (final_model.pt or models/model_*.pt)"; exit 1; }
for path in "$CKPT" "$ALPAGYM_PY" "$ALPASIM_ROOT/pyproject.toml" "$BINS/scene_manifest.jsonl" "$USDZ_DIR"; do
    [ -e "$path" ] || { echo "missing $path"; exit 1; }
done
[ -f "$(dirname "$CKPT")/config.yaml" ] || [ -f "$(dirname "$(dirname "$CKPT")")/config.yaml" ] \
    || { echo "no config.yaml next to or one level above $CKPT"; exit 1; }
[ -z "$(find "$USDZ_DIR" -maxdepth 1 -type l -print -quit)" ] \
    || { echo "$USDZ_DIR holds symlinks, which dangle inside the renderer container; point USDZ_DIR at the real files"; exit 1; }
for tool in docker nvidia-smi uv; do
    command -v "$tool" > /dev/null || { echo "missing tool: $tool"; exit 1; }
done
case "$TOPOLOGY" in
    8gpu*)
        NUM_GPUS=$(nvidia-smi -L | grep -c "^GPU")
        [ "$NUM_GPUS" -eq 8 ] || { echo "$TOPOLOGY pins its services to GPUs 0-7 and needs all 8, got $NUM_GPUS"; exit 1; }
        ;;
esac
ALPASIM_VERSION=$(grep -m1 '^version' "$ALPASIM_ROOT/pyproject.toml" | sed -E 's/.*"(.*)".*/\1/')
BASE_TAG=alpasim-base:$ALPASIM_VERSION
ALPASIM_IMAGE_TAR=${ALPASIM_IMAGE_TAR:-/home/bjaeger/docker_images/alpasim-base-$ALPASIM_VERSION.tar.zst}
NRE_TAG=$(grep -m1 -oE 'nvcr.io/nvidia/nre/[^[:space:]"]+' "$ALPASIM_ROOT/src/wizard/configs/base_config.yaml")
[ -n "$NRE_TAG" ] || { echo "no NRE renderer image in $ALPASIM_ROOT/src/wizard/configs/base_config.yaml"; exit 1; }
echo "Evaluating checkpoint: $CKPT"

# drop inherited entries from other PufferDrive checkouts: the teacher would import their pufferlib instead of $PD's
export PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v "cosim_Puffer\|/PufferDrive" | paste -sd: -)
export PUFFERDRIVE_ROOT=$PD
mkdir -p "$PD/experiments/logs"  # the build helper keeps its source hash there
# cp312 binding for the alpasim-training venv; NO_TRAIN skips the training-only CUDA extension
NO_TRAIN=1 PATH="$ALPAGYM_ROOT/.venv/bin:$PATH" bash "$PD/scripts/kesai/build_ext_if_changed.sh" "$PD" || exit 1
IMPORTED_PUFFERLIB=$("$ALPAGYM_PY" -c "import alpagym_distill.teacher, pufferlib; from pufferlib.ocean.drive import binding; print(pufferlib.__file__)") \
    || { echo "ERROR: teacher import failed (see traceback above)"; exit 1; }
case "$IMPORTED_PUFFERLIB" in
    "$PD"/*) echo "pufferlib: $IMPORTED_PUFFERLIB" ;;
    *) echo "ERROR: pufferlib imported from '$IMPORTED_PUFFERLIB', expected $PD"; exit 1 ;;
esac

mapfile -t SCENES < <("$ALPAGYM_PY" - "$BINS/scene_manifest.jsonl" "$N_SCENES" "$SCENE_LIST" <<'EOF'
import json, sys
manifest, n_scenes, scene_list = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = [json.loads(line) for line in open(manifest).read().splitlines()[1:]]
scenes = [r["alpasim_scene_id"] for r in rows if r["source_parity_pass"] and r["gt_replay_pass"]]
print(f"{len(scenes)} of {len(rows)} converted scenes passed the conversion checks", file=sys.stderr)
if scene_list:
    usable = set(scenes)
    listed = [s.strip() for s in open(scene_list) if s.strip() and not s.startswith("#")]
    scenes = [s for s in (s if s.startswith("clipgt-") else "clipgt-" + s for s in listed) if s in usable]
    print(f"{len(scenes)} of {len(listed)} scenes in {scene_list} usable", file=sys.stderr)
sys.stdout.write("".join(s + "\n" for s in (scenes if n_scenes == 0 else scenes[:n_scenes])))
EOF
)
[ ${#SCENES[@]} -gt 0 ] || { echo "no scenes to evaluate"; exit 1; }

# results live in the model's own eval folder, next to the PufferDrive benchmark evals
OUT=$RUN_DIR/eval/alpasim_${GOAL_MODE//:/}_$(basename "$CKPT" .pt)_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
{
    echo "checkpoint $CKPT"; echo "scenes ${#SCENES[@]}"; echo "topology $TOPOLOGY parallel $PARALLEL goal_mode $GOAL_MODE"
    for repo in "$PD" "$ALPAGYM_ROOT" "$ALPASIM_ROOT"; do
        echo "git $repo $(git -C "$repo" rev-parse --short HEAD 2>/dev/null) $(git -C "$repo" diff --quiet 2>/dev/null || echo dirty)"
    done
} > "$OUT/run_info.txt"
echo "Results -> $OUT"

docker info > /dev/null 2>&1 || { echo "docker is not usable on $(hostname)"; exit 1; }
# own compose project and port window per job, so evals sharing a node never touch each other's services
export COMPOSE_PROJECT_NAME=alpasim_${SLURM_JOB_ID:-$$}
BASEPORT=$((6000 + (${SLURM_JOB_ID:-$$} % 90) * 20))
LOADED_BASE=0
PULLED_NRE=0
cleanup() {
    local compose_file=$OUT/alpagym/alpasim/wizard_0/docker-compose.yaml
    [ -f "$compose_file" ] && docker compose -f "$compose_file" down --volumes --remove-orphans > /dev/null 2>&1
    for container in $(docker ps -aq --filter "label=com.docker.compose.project=$COMPOSE_PROJECT_NAME"); do
        docker rm -f "$container" > /dev/null 2>&1
    done
    # /var/lib/docker is tmpfs on the compute nodes: an image left behind keeps holding ~40 GB of node RAM
    [ "$LOADED_BASE" = 1 ] && docker rmi "$BASE_TAG" > /dev/null 2>&1
    [ "$PULLED_NRE" = 1 ] && docker rmi "$NRE_TAG" > /dev/null 2>&1
}
trap cleanup EXIT
if ! docker image inspect "$BASE_TAG" > /dev/null 2>&1; then
    [ -f "$ALPASIM_IMAGE_TAR" ] || { echo "$BASE_TAG is not on $(hostname) and $ALPASIM_IMAGE_TAR is missing"; exit 1; }
    echo "Loading $BASE_TAG from $ALPASIM_IMAGE_TAR"
    LOADED_BASE=1
    zstd -dc "$ALPASIM_IMAGE_TAR" | docker load || exit 1
fi
if ! docker image inspect "$NRE_TAG" > /dev/null 2>&1; then
    echo "Pulling $NRE_TAG"
    PULLED_NRE=1
    docker pull "$NRE_TAG" || exit 1
fi

# the teacher is an external driver, but the 8gpu topologies re-assert driver.skip=false and abort at startup
OVERRIDES="wizard.baseport=$BASEPORT runtime.endpoints.driver.skip=true"
[ "$RENDER_VIDEO" = "1" ] && RENDER_VIDEO_FLAG=true || RENDER_VIDEO_FLAG=false
echo "${#SCENES[@]} scenes, $TOPOLOGY with $PARALLEL in flight; full log: $OUT/evaluate.log"
cd "$ALPAGYM_ROOT" || exit 1
"$ALPAGYM_PY" -m alpagym_host.evaluate \
    --model_kind teacher \
    --teacher_checkpoint "$CKPT" \
    --teacher_bins_dir "$BINS" \
    --teacher_goal_mode "$GOAL_MODE" \
    --render_video "$RENDER_VIDEO_FLAG" \
    --local_scene_dir "$USDZ_DIR" \
    --scene_ids "${SCENES[@]}" \
    --topology "$TOPOLOGY" \
    --num_parallel_scenes "$PARALLEL" \
    --camera_width "$CAM_W" --camera_height "$CAM_H" \
    --simulation_timeout_s "$SIM_TIMEOUT_S" \
    --run_dir "$OUT/alpagym" \
    --extra_overrides "$OVERRIDES" \
    > "$OUT/evaluate.log" 2>&1
EVAL_STATUS=$?

if [ -f "$OUT/alpagym/evaluation.yaml" ]; then
    "$ALPAGYM_PY" - "$OUT/alpagym" <<'EOF'
import sys
from pathlib import Path
import yaml
run = Path(sys.argv[1])
summary = yaml.safe_load((run / "evaluation.yaml").read_text())
print(f"\n{len(summary['scenes'])} scenes evaluated, complete: {summary['complete']}")
aggregate = run / "metrics" / "aggregate.yaml"
if aggregate.exists():
    for name, stats in yaml.safe_load(aggregate.read_text())["metrics"].items():
        print(f"  {name:34s} mean {stats['mean']:9.4f}  median {stats['median']:9.4f}  n {stats['num_finite']}")
EOF
fi
if [ $EVAL_STATUS -ne 0 ]; then
    echo "AlpaSim evaluation failed (exit $EVAL_STATUS); last host lines of $OUT/evaluate.log:"
    grep -vE '^[a-z0-9_-]+-[0-9]+-[0-9]+ +\|' "$OUT/evaluate.log" | tail -n 30
fi
echo "per-scene metrics -> $OUT/alpagym/metrics/per_scene.csv; videos -> $OUT/alpagym/alpasim/wizard_0/rollouts"
echo "Done -> $OUT"
exit $EVAL_STATUS
