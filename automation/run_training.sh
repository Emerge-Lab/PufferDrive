#!/bin/bash
#
# This script runs the training process for PufferDrive.
# It first prepares the dataset by converting JSON maps to a binary format,
# and then starts the reinforcement learning training, saving artifacts to a
# specified output directory.
#
# Usage:
#   ./run_training.sh [DATASET_PATH] [PUFFERL_ARGS...]
#
# Arguments:
#   DATASET_PATH (optional): Path to the dataset (e.g., gs://my-bucket/data).
#                            If not provided, the default in drive.py is used.
#   PUFFERL_ARGS (optional): Additional arguments passed to `pufferl train`.

set -e # Exit on error

# Prepare arguments for the data processing script
DATA_PROCESSING_ARGS=()
if [ -n "$1" ]; then
  # The first argument is assumed to be the dataset path for pre-processing.
  # All subsequent arguments are passed to the training command.
  echo "Using dataset for pre-processing from: $1"
  DATA_PROCESSING_ARGS+=(--data_dir "$1")
  shift # Remove the dataset path from the list of arguments
else
  # If no path is provided, the Python script will use its default
  echo "No dataset path provided. The default path inside drive.py will be used."
fi

# Define a local directory for training outputs. Pufferl will save checkpoints here.
LOCAL_OUTPUT_DIR="/pufferdrive/training_output"
mkdir -p "$LOCAL_OUTPUT_DIR"
echo "Local training outputs will be saved to: $LOCAL_OUTPUT_DIR"

# Prepare JSONs by converting them to the binary format required for training.
python /pufferdrive/pufferlib/ocean/drive/drive.py "${DATA_PROCESSING_ARGS[@]}"

# Run the PufferLib training command for the 'puffer_drive' environment.
# PufferLib saves checkpoints to the directory specified by --train.data-dir.
echo "Starting PufferDrive training..."
python -m pufferlib.pufferl train puffer_drive --train.data-dir "$LOCAL_OUTPUT_DIR" "$@"

# After training, if running on Vertex AI, copy the artifacts to the GCS bucket
# provided by the AIP_MODEL_DIR environment variable. This makes the model
# available in the Vertex AI Model Registry.
if [ -n "$AIP_MODEL_DIR" ]; then
  echo "AIP_MODEL_DIR is set to $AIP_MODEL_DIR"
  echo "Copying training artifacts from $LOCAL_OUTPUT_DIR to $AIP_MODEL_DIR..."
  # Use a helper script with gcsfs since gsutil is not in the image.
  python /pufferdrive/automation/gcs_sync.py "$LOCAL_OUTPUT_DIR" "$AIP_MODEL_DIR"
else
  echo "AIP_MODEL_DIR is not set. Skipping copy to GCS. Artifacts are in $LOCAL_OUTPUT_DIR."
fi
