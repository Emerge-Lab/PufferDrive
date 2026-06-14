#!/usr/bin/env bash
set -euo pipefail

WINDOWS_REPO="$(cd "$(dirname "$0")/.." && pwd)"
NATIVE_REPO="${HOME}/PufferDrive-native"
VENV_DIR="${HOME}/.venvs/pufferdrive-wsl"

if [[ ! -f "$NATIVE_REPO/checkpoints/minimal_ppo/ppo_final.pt" ]]; then
  echo "Checkpoint not found: $NATIVE_REPO/checkpoints/minimal_ppo/ppo_final.pt"
  exit 1
fi

cp -f "$WINDOWS_REPO/scripts/minimal_ppo_train.py" "$NATIVE_REPO/scripts/minimal_ppo_train.py"
cp -f "$WINDOWS_REPO/scripts/visualize_minimal_ppo.py" "$NATIVE_REPO/scripts/visualize_minimal_ppo.py"

cd "$NATIVE_REPO"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if ! cmp -s "$WINDOWS_REPO/pufferlib/ocean/drive/drive.h" "$NATIVE_REPO/pufferlib/ocean/drive/drive.h"; then
  echo "Native renderer source changed; rebuilding Ocean binding..."
  cp -f "$WINDOWS_REPO/pufferlib/ocean/drive/drive.h" "$NATIVE_REPO/pufferlib/ocean/drive/drive.h"
  NO_TRAIN=1 python setup.py build_ext --inplace --force
fi

python scripts/visualize_minimal_ppo.py "$@"

mkdir -p "$WINDOWS_REPO/training_visualizations"
cp -f training_visualizations/*.mp4 "$WINDOWS_REPO/training_visualizations/"
cp -f training_visualizations/*.json "$WINDOWS_REPO/training_visualizations/"

echo "Copied outputs to $WINDOWS_REPO/training_visualizations"
