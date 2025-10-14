#!/bin/bash
#
# This script runs the training process for PufferDrive.
# It first prepares the dataset by converting JSON maps to a binary format,
# and then starts the reinforcement learning training, saving artifacts to a
# specified output directory.
#
# Usage:
#   ./run_training.sh [--dataset-path DATASET_PATH] [PUFFERL_ARGS...]
#
# Arguments:
#   --dataset-path (optional): Path to the dataset (e.g., gs://my-bucket/data).
#                              If not provided, the default in drive.py is used.
#   PUFFERL_ARGS   (optional): Additional arguments passed to `pufferl train`.

set -e # Exit on error

# --- Argument Parsing ---
DATASET_PATH=""
PUFFERL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    *)
      PUFFERL_ARGS+=("$1")
      shift
      ;;
  esac
done

# --- Data Pre-processing ---
DATA_PROCESSING_ARGS=()
if [ -n "$DATASET_PATH" ]; then
  echo "Using dataset for pre-processing from: $DATASET_PATH"
  DATA_PROCESSING_ARGS+=(--data_dir "$DATASET_PATH")
else
  echo "No dataset path provided. The default path inside drive.py will be used."
fi

# Prepare JSONs by converting them to the binary format required for training.
python /pufferdrive/pufferlib/ocean/drive/drive.py "${DATA_PROCESSING_ARGS[@]}"

# Always write artifacts to a local directory first. This is the most robust
# pattern for cloud jobs, avoiding issues with direct GCS writes (e.g., append mode).
# The entire directory will be uploaded to GCS at the end of the job.
OUTPUT_DIR="/pufferdrive/training_output"
mkdir -p "$OUTPUT_DIR"
echo "Training artifacts will be saved locally to: $OUTPUT_DIR"

# The GCPLogger in pufferl.py will automatically pick up project and job IDs from env vars.
if [ -n "$CLOUD_ML_JOB_ID" ]; then
  echo "GCP environment detected (via CLOUD_ML_JOB_ID). Enabling GCP Cloud Monitoring logger."
  echo "AJE bypass for test: use a tensorboard"
  PUFFERL_ARGS+=(--tb)
fi

# --- Training ---
# Run the PufferLib training command for the 'puffer_drive' environment.
# PufferLib saves checkpoints to the directory specified by --train.data-dir.
echo "Starting PufferDrive training... Artifacts will be saved to: $OUTPUT_DIR"
python -m pufferlib.pufferl train puffer_drive --train.data-dir "$OUTPUT_DIR" "${PUFFERL_ARGS[@]}"

# After training, if running on Vertex AI, copy the artifacts to the GCS bucket
# provided by the AIP_MODEL_DIR environment variable. This makes the model
# available in the Vertex AI Model Registry.
if [ -n "$AIP_MODEL_DIR" ]; then
  echo "AIP_MODEL_DIR is set to $AIP_MODEL_DIR"
  echo "Copying final training artifacts from $OUTPUT_DIR to $AIP_MODEL_DIR..."
  # Use a helper script with gcsfs since gsutil is not in the image.
  python /pufferdrive/automation/gcs_sync.py "${OUTPUT_DIR}/" "$AIP_MODEL_DIR"
else
  echo "AIP_MODEL_DIR is not set. Skipping final copy. Artifacts are in $OUTPUT_DIR."
fi
