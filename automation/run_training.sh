#!/bin/bash
#
# This script runs the training process for PufferDrive.
# It first prepares the dataset by converting JSON maps to a binary format,
# and then starts the reinforcement learning training.
#
# Usage:
#   ./run_training.sh [DATASET_PATH]
#
# Arguments:
#   DATASET_PATH (optional): The path to the directory containing the training
#                            dataset (JSON map files). If not provided, the
#                            script will use the default path specified in
#                            the drive.py script (e.g., 'data/train').

# Prepare Jsons by converting them to the binary format required for training.
if [ -z "$1" ]; then
  echo "No dataset path provided. The default path inside drive.py will be used."
  python /puffertank/pufferlib/ocean/drive/drive.py
else
  DATA_DIR=$1
  echo "Using dataset from: $DATA_DIR"
  python /puffertank/pufferlib/ocean/drive/drive.py --data_dir "$DATA_DIR"
fi

# Run the PufferLib training command for the 'puffer_drive' environment.
echo "Starting PufferDrive training..."
python -m pufferlib.pufferl train puffer_drive
