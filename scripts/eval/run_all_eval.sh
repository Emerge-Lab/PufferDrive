#!/bin/bash

BASE_DIR="experiments"
# Unified eval pipeline: --eval-simulation picks validation_<sim>; the simple
# flags below override that evaluator's [eval.*] section for this run.
COMMAND_PREFIX="puffer eval puffer_drive"

# Defaults
NUM_SCENARIOS=50
RENDER=0
RENDER_BACKEND=""   # empty -> use the evaluator's section default (egl|triage_html|obs_html)
EVAL_SIMULATION="replay"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-scenarios)
            NUM_SCENARIOS="$2"
            shift 2
            ;;
        --eval-simulation)
            EVAL_SIMULATION="$2"
            shift 2
            ;;
        --render)
            RENDER="$2"
            shift 2
            ;;
        --render-backend)
            RENDER_BACKEND="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown argument: $1"
            exit 1
            ;;
    esac
done

COMMAND_SUFFIX="--num-scenarios ${NUM_SCENARIOS} --render ${RENDER} --eval-simulation ${EVAL_SIMULATION}"
if [ -n "${RENDER_BACKEND}" ]; then
    COMMAND_SUFFIX="${COMMAND_SUFFIX} --render-backend ${RENDER_BACKEND}"
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' not found."
    echo "Please run this script from the directory containing '$BASE_DIR/'"
    exit 1
fi

echo "🚀 Starting evaluation for all experiments in '$BASE_DIR'..."
echo "   num_scenarios = ${NUM_SCENARIOS}"
echo "   eval_simulation = ${EVAL_SIMULATION}"
echo "---"

for exp_path in ${BASE_DIR}/*/; do
    MODELS_DIR="${exp_path}/models/"
    echo "Processing experiment: ${exp_path}"

    LATEST_MODEL=$(ls -1 ${MODELS_DIR}*.pt 2>/dev/null | grep -v "trainer_state.pt" | sort -V | tail -n 1)

    if [ -z "${LATEST_MODEL}" ]; then
        echo "  [SKIP] No '.pt' model files found in ${MODELS_DIR}."
        echo "---"
        continue
    fi

    echo "  ✅ Found latest model: ${LATEST_MODEL}"
    FULL_COMMAND="${COMMAND_PREFIX} --load-model-path ${LATEST_MODEL} ${COMMAND_SUFFIX}"
    echo "  ▶️  Executing:"
    echo "      ${FULL_COMMAND}"

    ${FULL_COMMAND}
    echo "---"
done

echo "🎉 All experiments processed."
