#!/bin/bash
set -e

# Usage: ./test_gigaflow_dynamics
# TODO: Make compatible for Linux
MODE=${1:-release}
PLATFORM=$(uname -s)

# Config
RAYLIB_DIR="raylib-5.5_macos"
BOX2D_DIR="box2d-macos-arm64"
SRC="tests/test_gigaflow_dynamics.c"
OUTPUT="test_gigaflow_dynamics"

# Flags
FLAGS=(
    -Wall
    -I"$RAYLIB_DIR/include"
    -I"$BOX2D_DIR/include"
    -I./pufferlib/extensions
    -I./pufferlib/ocean/drive
    -DPLATFORM_DESKTOP
)

# Linker
LIBS=(
    "$RAYLIB_DIR/lib/libraylib.a"
    -lm
    -lpthread
)

if [[ "$PLATFORM" == "Darwin" ]]; then
    LIBS+=(
        -framework Cocoa
        -framework IOKit
        -framework CoreVideo
    )
fi


  echo "Building for release..."
  FLAGS+=(-O2)
# Compile
clang "${FLAGS[@]}" -o "$OUTPUT" "$SRC" "${LIBS[@]}"

echo "Success! Run with: ./$OUTPUT"