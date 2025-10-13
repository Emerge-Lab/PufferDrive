#!/bin/bash

set -e
echo "=== Building C extensions ==="
python setup.py build_ext --inplace --force

echo "=== Submitting job array ==="
sbatch scripts/run.sh
