#!/bin/bash
#
# This script runs the training process for PufferDrive.
# It first prepares the dataset by converting JSON maps to a binary format and
# then starts the reinforcement learning training.
#
# Usage:
#   ./run_training.sh [DATASET_PATH]
#
# Arguments:
#   DATASET_PATH (optional): The path to the directory containing the training
#                            dataset (JSON map files). This can be a local path
#                            (e.g., /path/to/data) or a GCS bucket path
#                            (e.g., gs://my-bucket/data/train). If not provided,
#                            the script will use the default path specified in
#                            the drive.py script.

set -e # Exit on error

# Prepare arguments for the data processing script
ARGS=()
if [ -n "$1" ]; then
  # If a dataset path is provided, add it to the arguments
  echo "Using dataset from: $1"
  ARGS+=(--dataset-path "$1")
else
  # If no path is provided, the Python script will use its default
  echo "No dataset path provided. The default path inside drive.py will be used."
fi

# Prepare Jsons by converting them to the binary format required for training.
python /pufferdrive/pufferlib/ocean/drive/drive.py "${ARGS[@]}"

# Run the PufferLib training command for the 'puffer_drive' environment.
echo "Starting PufferDrive training..."
python -m pufferlib.pufferl train puffer_drive
