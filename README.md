# PufferDrive

## Install

```bash
# macOS only: libomp is required.
brew install libomp

# Python env
uv venv && source .venv/bin/activate
uv pip install -e .

# Build the C extension (re-run after any sim/ or src/ change)
python setup.py build_ext --inplace --force
```

`setup.py` auto-detects CUDA on Linux (via `torch.utils.cpp_extension.CUDA_HOME`) and builds the GPU backend (`src/bindings.cu`); on macOS or without CUDA it builds the CPU backend (`src/bindings_cpu.cpp`). Set `PUFFER_CPU=1` to force the CPU build.

`build.sh` is still available for CUDA-native PuffeRL builds and the standalone executable.
