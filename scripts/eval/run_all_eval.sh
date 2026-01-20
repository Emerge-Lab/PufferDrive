#!/bin/bash
BASE_DIR="experiments"
COMMAND_PREFIX="puffer eval_multi_scenarios puffer_drive"
COMMAND_SUFFIX="--num_scenarios 500 --render 0 --render_obs 0"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' not found."
    echo "Please run this script from the directory containing '$BASE_DIR/'"
    exit 1
fi

echo "🚀 Starting evaluation for all experiments in '$BASE_DIR'..."
echo "---"

for exp_path in ${BASE_DIR}/*/; do
    MODELS_DIR="${exp_path}best_models/"
    echo "Processing experiment: ${exp_path}"
    if [ ! -d "${MODELS_DIR}" ]; then
        echo "  [SKIP] 'best_models/' directory not found in ${exp_path}."
        echo "---"
        continue
    fi
    LATEST_MODEL=$(ls -1 ${MODELS_DIR}*.pt 2>/dev/null | sort -V | tail -n 1)
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
