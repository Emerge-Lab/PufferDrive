#!/bin/bash

# Usage: ./build_ocean.sh visualize [local|fast]

ENV=$1
MODE=${2:-local}
PLATFORM="$(uname -s)"

if [ "$ENV" = "visualize" ]; then
    SRC_DIR="pufferlib/ocean/drive"
else
    SRC_DIR="pufferlib/ocean/$ENV"
fi

RAYLIB_NAME='raylib-5.5_macos'
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
fi

LINK_ARCHIVES="./extern/$RAYLIB_NAME/lib/libraylib.a"

# Detect available compiler and set compiler-specific flags
if command -v clang >/dev/null 2>&1; then
    COMPILER="clang"
    ERROR_LIMIT_FLAG="-ferror-limit=3"
    echo "Using clang compiler"
else
    COMPILER="gcc"
    ERROR_LIMIT_FLAG="-fmax-errors=3"
    echo "Using gcc compiler (clang not found)"
fi

FLAGS=(
    -Wall
    -I./extern/$RAYLIB_NAME/include
    -I./pufferlib/extensions
    -I./extern/inih-r62
    "$SRC_DIR/$ENV.c" -o "$ENV"
    ./extern/inih-r62/ini.c
    $LINK_ARCHIVES
    -lm
    -lpthread
    $ERROR_LIMIT_FLAG
    -DPLATFORM_DESKTOP
)

if [ "$PLATFORM" = "Darwin" ]; then
    FLAGS+=(
        -framework Cocoa
        -framework IOKit
        -framework CoreVideo
    )
fi

echo ${FLAGS[@]}

if [ "$MODE" = "local" ]; then
    echo "Building $ENV for local testing..."
    if [ "$PLATFORM" = "Linux" ]; then
        FLAGS+=(
            -fsanitize=address,undefined,bounds,pointer-overflow,leak
            -fno-omit-frame-pointer
        )
    fi
    $COMPILER -g -O0 ${FLAGS[@]}
elif [ "$MODE" = "fast" ]; then
    echo "Building optimized $ENV for local testing..."
    $COMPILER -pg -O2 -DNDEBUG ${FLAGS[@]}
    echo "Built to: $ENV"
else
    echo "Invalid mode: $MODE (use local|fast)"
    exit 1
fi
