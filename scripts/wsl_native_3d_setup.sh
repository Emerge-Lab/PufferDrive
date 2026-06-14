#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ORIGINAL_REPO_DIR="${PUFFERDRIVE_NATIVE_ORIGIN:-}"

echo "==> Installing Linux build and render dependencies"
sudo apt update
sudo apt install -y \
  build-essential \
  clang \
  curl \
  ffmpeg \
  libgl1 \
  libglib2.0-0 \
  libx11-dev \
  libxcursor-dev \
  libxi-dev \
  libxinerama-dev \
  libxrandr-dev \
  libxxf86vm-dev \
  mesa-utils \
  python3-dev \
  python3-pip \
  python3-venv \
  rsync \
  xvfb

if [[ "$PWD" == /mnt/* && -z "$ORIGINAL_REPO_DIR" ]]; then
  NATIVE_REPO_DIR="${HOME}/PufferDrive-native"
  echo "==> Repository is on a Windows mount; mirroring to ${NATIVE_REPO_DIR} for native Linux build"
  mkdir -p "$NATIVE_REPO_DIR"
  rsync -a --delete \
    --exclude ".git/" \
    --exclude ".venv/" \
    --exclude ".venv-wsl/" \
    --exclude ".venv-wsl-build/" \
    --exclude ".venv-wsl-native/" \
    --exclude "build/" \
    --exclude "dist/" \
    --exclude "*.egg-info/" \
    "$PWD/" "$NATIVE_REPO_DIR/"
  echo "==> Restarting setup from Linux filesystem"
  PUFFERDRIVE_NATIVE_ORIGIN="$PWD" bash "$NATIVE_REPO_DIR/scripts/wsl_native_3d_setup.sh" "$@"
  exit $?
fi

if [[ ! -d resources ]]; then
  echo "==> Creating Linux resources symlink"
  rm -f resources
  ln -s pufferlib/resources resources
fi

echo "==> Creating WSL Python environment"
VENV_DIR="${HOME}/.venvs/pufferdrive-wsl"
rm -rf "$VENV_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv --copies "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing Python dependencies"
python -m pip install \
  "numpy<2" \
  Cython \
  "gym==0.23" \
  "gymnasium==0.29.1" \
  "pettingzoo==1.24.1" \
  "shimmy[gym-v21]" \
  tqdm \
  imageio \
  imageio-ffmpeg \
  "torch"

echo "==> Building PufferDrive native Ocean binding"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
NO_TRAIN=1 python setup.py build_ext --inplace --force

echo "==> Done. Native build is ready in ${PWD}"
echo "Next:"
echo "  bash scripts/prepare_waymo_maps_wsl.sh /path/to/scenario.json"
echo "  bash scripts/run_minimal_ppo_wsl.sh --map-dir resources/drive/binaries/training --num-maps 1"
