#!/usr/bin/env bash
# Regenerate the smoke goldens inside the pinned QEMU image, the only place they are bit-reproducible:
#   tests/smoke_tests/update_goldens.sh [test file ...]   (default: both smoke tests)
# then review `git diff tests/smoke_tests/data` and commit the goldens.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

TESTS=("$@")
if [ ${#TESTS[@]} -eq 0 ]; then
  TESTS=(
    tests/smoke_tests/test_drive_train.py
    tests/smoke_tests/test_drive_rollout.py
  )
fi

echo ">>> Building pinned image (picks up C/sim changes)"
docker build -f tests/smoke_tests/Dockerfile -t opengigaflow-smoke .

echo ">>> Regenerating goldens under QEMU: ${TESTS[*]}"
docker run --rm -e SMOKE_UPDATE_GOLDEN=1 \
  -v "$PWD/tests/smoke_tests/data:/app/tests/smoke_tests/data" \
  opengigaflow-smoke "${TESTS[@]}"

echo ">>> Done. Updated goldens:"
git -c core.fileMode=false status --short tests/smoke_tests/data/
