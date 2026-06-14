#!/usr/bin/env bash
set -euo pipefail

WINDOWS_REPO="$(cd "$(dirname "$0")/.." && pwd)"
NATIVE_REPO="${HOME}/PufferDrive-native"
VENV_DIR="${HOME}/.venvs/pufferdrive-wsl"

if [[ ! -d "$NATIVE_REPO/pufferlib" ]]; then
  echo "Native repository not found: $NATIVE_REPO"
  echo "Run bash scripts/wsl_native_3d_setup.sh first."
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "WSL Python environment not found: $VENV_DIR"
  echo "Run bash scripts/wsl_native_3d_setup.sh first."
  exit 1
fi

cp -f "$WINDOWS_REPO/scripts/minimal_ppo_train.py" "$NATIVE_REPO/scripts/minimal_ppo_train.py"

cd "$NATIVE_REPO"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/minimal_ppo_train.py "$@"
