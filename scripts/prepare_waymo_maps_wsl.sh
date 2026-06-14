#!/usr/bin/env bash
set -euo pipefail

WINDOWS_REPO="$(cd "$(dirname "$0")/.." && pwd)"
NATIVE_REPO="${HOME}/PufferDrive-native"
VENV_DIR="${HOME}/.venvs/pufferdrive-wsl"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/prepare_waymo_maps_wsl.sh SCENARIO.json [SCENARIO.json ...]"
  exit 2
fi

if [[ ! -x "$VENV_DIR/bin/python" || ! -d "$NATIVE_REPO/pufferlib" ]]; then
  echo "Native WSL environment is not ready."
  echo "Run bash scripts/wsl_native_3d_setup.sh first."
  exit 1
fi

json_files=()
for path in "$@"; do
  if [[ ! -f "$path" ]]; then
    echo "Scenario JSON not found: $path"
    exit 1
  fi
  json_files+=("$(realpath "$path")")
done

cp -f "$WINDOWS_REPO/scripts/prepare_waymo_maps.py" "$NATIVE_REPO/scripts/prepare_waymo_maps.py"

cd "$NATIVE_REPO"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/prepare_waymo_maps.py "${json_files[@]}"

echo "Prepared maps in $NATIVE_REPO/resources/drive/binaries/training"
