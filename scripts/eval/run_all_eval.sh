#!/bin/bash

BASE_DIR="experiments"
COMMAND_PREFIX="puffer benchmark puffer_drive"

# Defaults
BENCHMARK_DATASETS="carla,womd_multi,womd_single,nuplan_multi,nuplan_single"
RENDER="False"
RENDER_OBS="False"
RENDER_ONLY="True"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark-datasets)
            BENCHMARK_DATASETS="$2"
            shift 2
            ;;
        --render)
            RENDER="$2"
            shift 2
            ;;
        --render-obs)
            RENDER_OBS="$2"
            shift 2
            ;;
        --render-only)
            RENDER_ONLY="$2"
            shift 2
            ;;
        *)
            echo "❌ Unknown argument: $1"
            exit 1
            ;;
    esac
done

COMMAND_SUFFIX="--eval.benchmark-datasets ${BENCHMARK_DATASETS} --eval.render ${RENDER} --eval.render-obs ${RENDER_OBS} --eval.render-only ${RENDER_ONLY}"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' not found."
    echo "Please run this script from the directory containing '$BASE_DIR/'"
    exit 1
fi

echo "🚀 Starting evaluation for all experiments in '$BASE_DIR'..."
echo "   benchmark_datasets = ${BENCHMARK_DATASETS}"
echo "   render = ${RENDER}"
echo "   render_obs = ${RENDER_OBS}"
echo "   render_only = ${RENDER_ONLY}"
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
