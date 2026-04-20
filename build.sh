#!/bin/bash
set -e

# Usage:
#   ./build.sh                  # Build torch-compatible GPU _C.so with drive statically linked
#   ./build.sh --cuda           # Build native CUDA trainer _C.so
#   ./build.sh --cpu            # CPU fallback, torch only
#   ./build.sh --debug          # Debug build
#   ./build.sh --local          # Standalone executable (debug, sanitizers)
#   ./build.sh --fast           # Standalone executable (optimized)
#   ./build.sh --web            # Emscripten web build
#   ./build.sh --profile        # Kernel profiling binary
#   ./build.sh clean            # Remove compiled artifacts and Python bytecode

ENV=drive
BACKEND=torch

for arg in "$@"; do
    case $arg in
        clean) CLEAN=1 ;;
        --cuda) BACKEND=cuda ;;
        --debug) DEBUG=1 ;;
        --local) MODE=local ;;
        --fast)  MODE=fast ;;
        --web)   MODE=web ;;
        --profile) MODE=profile ;;
        --cpu)   MODE=cpu ;;
        *) echo "Error: unknown argument '$arg'" && exit 1 ;;
    esac
done

if [ -z "$MODE" ] && [ "$BACKEND" = "torch" ] && [ -z "$PRECISION" ]; then
    PRECISION="-DPRECISION_FLOAT"
fi

if [ -n "$CLEAN" ]; then
    rm -rf build drive profile
    rm -f pufferlib/_C*.so pufferlib/_C*.pyd pufferlib/_C*.dylib
    find . -type d -name '__pycache__' -exec rm -rf {} +
    find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
    exit 0
fi

# Linux/mac
PLATFORM="$(uname -s)"
if [ "$PLATFORM" = "Linux" ]; then
    RAYLIB_NAME='raylib-5.5_linux_amd64'
    OMP_LIB=-lomp5
    SANITIZE_FLAGS=(-fsanitize=address,undefined,bounds,pointer-overflow,leak -fno-omit-frame-pointer)
    STANDALONE_LDFLAGS=(-lGL)
    SHARED_LDFLAGS=(-Bsymbolic-functions)
else
    RAYLIB_NAME='raylib-5.5_macos'
    OMP_LIB=-lomp
    SANITIZE_FLAGS=()
    STANDALONE_LDFLAGS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
    SHARED_LDFLAGS=(-framework Cocoa -framework OpenGL -framework IOKit -undefined dynamic_lookup)
fi

CLANG_WARN=(
    -Wall
    -ferror-limit=3
    -Werror=incompatible-pointer-types
    -Werror=return-type
    -Wno-error=incompatible-pointer-types-discards-qualifiers
    -Wno-incompatible-pointer-types-discards-qualifiers
    -Wno-error=array-parameter
)

download() {
    local name=$1 url=$2
    local target="vendor/$name"
    local archive=""
    [ -d "$target" ] && return
    if [ -d "$name" ]; then
        mkdir -p vendor
        mv "$name" "$target"
        return
    fi
    echo "Downloading $name..."
    mkdir -p vendor
    case "$url" in
        *.zip)
            archive="$target.zip"
            curl -sL "$url" -o "$archive"
            unzip -q "$archive" -d vendor
            rm "$archive"
            ;;
        *)
            archive="$target.tar.gz"
            curl -sL "$url" -o "$archive"
            tar xf "$archive" -C vendor
            rm "$archive"
            ;;
    esac
}

RAYLIB_URL="https://github.com/raysan5/raylib/releases/download/5.5"
if [ "$MODE" = "web" ]; then
    RAYLIB_NAME='raylib-5.5_webassembly'
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.zip"
else
    download "$RAYLIB_NAME" "$RAYLIB_URL/$RAYLIB_NAME.tar.gz"
fi

RAYLIB_DIR="vendor/$RAYLIB_NAME"
RAYLIB_A="$RAYLIB_DIR/lib/libraylib.a"
INCLUDES=(-I./$RAYLIB_DIR/include -I./src -I./vendor)
LINK_ARCHIVES=("$RAYLIB_A")
EXTRA_SRC=""

[ -d "sim" ] || { echo "Error: sim/ not found"; exit 1; }
SRC_DIR="sim"
OUTPUT_NAME="$ENV"

# Standalone environment build
if [ -n "$DEBUG" ] || [ "$MODE" = "local" ]; then
    CLANG_OPT=(-g -O0 "${CLANG_WARN[@]}" "${SANITIZE_FLAGS[@]}")
    NVCC_OPT="-O0 -g"
    LINK_OPT="-g"
else
    CLANG_OPT=(-O2 -DNDEBUG "${CLANG_WARN[@]}")
    NVCC_OPT="-O2 --threads 0"
    LINK_OPT="-O2"
fi
if [ "$MODE" = "local" ] || [ "$MODE" = "fast" ]; then
    FLAGS=(
        "${INCLUDES[@]}"
        "$SRC_DIR/$ENV.c" $EXTRA_SRC -o "$OUTPUT_NAME"
        "${LINK_ARCHIVES[@]}"
        "${STANDALONE_LDFLAGS[@]}"
        -lm -lpthread -fopenmp
        -DPLATFORM_DESKTOP
    )
    echo "Compiling $ENV..."
    ${CC:-clang} "${CLANG_OPT[@]}" "${FLAGS[@]}"
    echo "Built: ./$OUTPUT_NAME"
    exit 0
elif [ "$MODE" = "web" ]; then
    mkdir -p "build/web/$ENV"
    echo "Compiling $ENV for web..."
    emcc \
        -o "build/web/$ENV/game.html" \
        "$SRC_DIR/$ENV.c" $EXTRA_SRC \
        -O3 -Wall \
        "${LINK_ARCHIVES[@]}" \
        "${INCLUDES[@]}" \
        -L. -L./$RAYLIB_DIR/lib \
        -sASSERTIONS=2 -gsource-map \
        -sUSE_GLFW=3 -sUSE_WEBGL2=1 -sASYNCIFY -sFILESYSTEM -sFORCE_FILESYSTEM=1 \
        --shell-file vendor/minshell.html \
        -sINITIAL_MEMORY=512MB -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=512KB \
        -DNDEBUG -DPLATFORM_WEB -DGRAPHICS_API_OPENGL_ES3 \
        --preload-file resources/$ENV@resources/$ENV \
        --preload-file resources/shared@resources/shared
    echo "Built: build/web/$ENV/game.html"
    exit 0
fi

# Find cuDNN path
CUDA_HOME=${CUDA_HOME:-${CUDA_PATH:-$(dirname "$(dirname "$(which nvcc)")")}}
CUDNN_IFLAG=""
CUDNN_LFLAG=""
for dir in /usr/local/cuda/include /usr/include; do
    if [ -f "$dir/cudnn.h" ]; then
        CUDNN_IFLAG="-I$dir"
        break
    fi
done
for dir in /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
    if [ -f "$dir/libcudnn.so" ]; then
        CUDNN_LFLAG="-L$dir"
        break
    fi
done
if [ -z "$CUDNN_IFLAG" ]; then
    CUDNN_IFLAG=$(python -c "import nvidia.cudnn, os; print('-I' + os.path.join(nvidia.cudnn.__path__[0], 'include'))" 2>/dev/null || echo "")
fi
if [ -z "$CUDNN_LFLAG" ]; then
    CUDNN_LFLAG=$(python -c "import nvidia.cudnn, os; print('-L' + os.path.join(nvidia.cudnn.__path__[0], 'lib'))" 2>/dev/null || echo "")
fi

# NCCL include/lib fallback (mirrors the cuDNN fallback above).
# Needed when NCCL is provided by the nvidia-nccl-cu12 wheel in the active venv.
NCCL_IFLAG=""
NCCL_LFLAG=""
for dir in /usr/include /usr/local/cuda/include; do
    if [ -f "$dir/nccl.h" ]; then NCCL_IFLAG="-I$dir"; break; fi
done
for dir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
    if [ -f "$dir/libnccl.so" ] || [ -f "$dir/libnccl.so.2" ]; then NCCL_LFLAG="-L$dir"; break; fi
done
if [ -z "$NCCL_IFLAG" ]; then
    NCCL_IFLAG=$(python -c "import nvidia.nccl, os; print('-I' + os.path.join(nvidia.nccl.__path__[0], 'include'))" 2>/dev/null || echo "")
fi
if [ -z "$NCCL_LFLAG" ]; then
    NCCL_LFLAG=$(python -c "import nvidia.nccl, os; print('-L' + os.path.join(nvidia.nccl.__path__[0], 'lib'))" 2>/dev/null || echo "")
fi

export CCACHE_DIR="${CCACHE_DIR:-$HOME/.ccache}"
export CCACHE_BASEDIR="$(pwd)"
export CCACHE_COMPILERCHECK=content
NVCC="ccache $CUDA_HOME/bin/nvcc"
CC="${CC:-$(command -v ccache >/dev/null && echo 'ccache clang' || echo 'clang')}"
ARCH=${NVCC_ARCH:-native}

PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python -c "import pybind11; print(pybind11.get_include())")
NUMPY_INCLUDE=$(python -c "import numpy; print(numpy.get_include())")
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT="pufferlib/_C${EXT_SUFFIX}"

BINDING_SRC="$SRC_DIR/binding.c"
mkdir -p build
STATIC_OBJ="build/libstatic_${ENV}.o"
STATIC_LIB="build/libstatic_${ENV}.a"

if [ ! -f "$BINDING_SRC" ]; then
    echo "Error: $BINDING_SRC not found"
    exit 1
fi

echo "Compiling static library for $ENV..."
${CC:-clang} -c "${CLANG_OPT[@]}" \
    -I. -Isrc -I$SRC_DIR -Ivendor \
    -I./$RAYLIB_DIR/include -I$CUDA_HOME/include \
    -DPLATFORM_DESKTOP \
    -fno-semantic-interposition -fvisibility=hidden \
    -fPIC -fopenmp \
    "$BINDING_SRC" -o "$STATIC_OBJ"
ar rcs "$STATIC_LIB" "$STATIC_OBJ"

# Brittle hack: have to extract the tensor type from the static lib to build trainer
OBS_TENSOR_T=$(awk '/^#define OBS_TENSOR_T/{print $3}' "$BINDING_SRC")
if [ -z "$OBS_TENSOR_T" ]; then
    echo "Error: Could not find OBS_TENSOR_T in $BINDING_SRC"
    exit 1
fi

if [ -z "$MODE" ]; then
    PUFFER_NATIVE_PUFFERL=0
    if [ "$BACKEND" = "cuda" ]; then
        PUFFER_NATIVE_PUFFERL=1
    fi
    echo "Compiling CUDA ($ARCH) ${BACKEND} backend..."
    $NVCC -c -arch=$ARCH -Xcompiler -fPIC \
        -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1 \
        -Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
        -Xcompiler=-DPLATFORM_DESKTOP \
        -std=c++17 \
        -I. -Isrc \
        -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE -I$NUMPY_INCLUDE \
        -I$CUDA_HOME/include $CUDNN_IFLAG -I$RAYLIB_DIR/include \
        -Xcompiler=-fopenmp \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        -DPUFFER_NATIVE_PUFFERL=$PUFFER_NATIVE_PUFFERL \
        $PRECISION $NVCC_OPT \
        src/bindings.cu -o build/bindings.o

    LINK_CMD=(
        ${CXX:-g++} -shared -fPIC -fopenmp
        build/bindings.o "$STATIC_LIB" "$RAYLIB_A"
        -L$CUDA_HOME/lib64 $CUDNN_LFLAG
        -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand -lcudnn
        $OMP_LIB $LINK_OPT
        "${SHARED_LDFLAGS[@]}"
        -o "$OUTPUT"
    )
    "${LINK_CMD[@]}"
    echo "Built: $OUTPUT"

elif [ "$MODE" = "cpu" ]; then
    echo "Compiling CPU training backend..."
    ${CXX:-g++} -c -fPIC -fopenmp \
        -D_GLIBCXX_USE_CXX11_ABI=1 \
        -DPLATFORM_DESKTOP \
        -std=c++17 \
        -I. -Isrc \
        -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        $PRECISION $LINK_OPT \
        src/bindings_cpu.cpp -o build/bindings_cpu.o
    LINK_CMD=(
        ${CXX:-g++} -shared -fPIC -fopenmp
        build/bindings_cpu.o "$STATIC_LIB" "$RAYLIB_A"
        -lm -lpthread $OMP_LIB $LINK_OPT
        "${SHARED_LDFLAGS[@]}"
        -o "$OUTPUT"
    )
    "${LINK_CMD[@]}"
    echo "Built: $OUTPUT"

elif [ "$MODE" = "profile" ]; then
    echo "Compiling profile binary ($ARCH)..."
    $NVCC $NVCC_OPT -arch=$ARCH -std=c++17 \
        -I. -Isrc -I$SRC_DIR -Ivendor \
        -I$CUDA_HOME/include $CUDNN_IFLAG -I$RAYLIB_DIR/include \
        -DOBS_TENSOR_T=$OBS_TENSOR_T \
        -DENV_NAME=$ENV \
        -Xcompiler=-DPLATFORM_DESKTOP \
        $PRECISION \
        -Xcompiler=-fopenmp \
        tests/profile_kernels.cu vendor/ini.c \
        "$STATIC_LIB" "$RAYLIB_A" \
        -lnccl -lnvidia-ml -lcublas -lcurand -lcudnn \
        -lGL -lm -lpthread $OMP_LIB \
        -o profile
    echo "Built: ./profile"
fi
