#!/usr/bin/env bash
# Starts the CARLA server the demo renders through. CARLA_ROOT must point at a CARLA 0.9.16 install.
set -euo pipefail

: "${CARLA_ROOT:?set CARLA_ROOT to the CARLA install directory (the one containing CarlaUE4.sh)}"
CARLA_PORT="${CARLA_PORT:-2000}"
NVIDIA_VULKAN_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json

if [[ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]]; then
    echo "no CarlaUE4.sh in $CARLA_ROOT" >&2
    exit 1
fi
# CARLA renders on Vulkan device 0, which on a CPU with integrated graphics is not the discrete GPU.
if [[ -f "$NVIDIA_VULKAN_ICD" && -z "${VK_ICD_FILENAMES:-}" ]]; then
    export VK_ICD_FILENAMES="$NVIDIA_VULKAN_ICD"
fi
exec "$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -carla-rpc-port="$CARLA_PORT" -quality-level=Epic "$@"
