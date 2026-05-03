# PufferDrive

## Install

```bash
# macOS only: libomp is required for the env build.
brew install libomp

# Python env
uv venv && source .venv/bin/activate
uv pip install -e .

# Build the drive C extension (re-run after any sim/ or src/ change)
python setup.py build_ext --inplace --force
```

`setup.py` only builds the env-side `pufferlib._C` (CPU pybind11 bindings) so you can iterate on the C sim without CUDA. For full GPU/CUDA training, use `build.sh`.
