#!/usr/bin/env bash
# Regenerate the smoke goldens inside the pinned QEMU image so the values match CI.
# Run from the repo root. Optionally pass specific test files; defaults to all.
#
#   tests/smoke_tests/update_goldens.sh
#   tests/smoke_tests/update_goldens.sh tests/smoke_tests/test_drive_rollout.py
#
# Review `git diff` afterwards, then commit (rollout golden needs -f, it's under
# the gitignored data/):
#   git add tests/smoke_tests/data/drive_smoke_golden.json
#   git add -f tests/smoke_tests/data/drive_rollout_golden.json
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
docker build -f tests/smoke_tests/Dockerfile -t pufferdrive-smoke .

echo ">>> Regenerating goldens under QEMU: ${TESTS[*]}"
docker run --rm -e SMOKE_UPDATE_GOLDEN=1 \
  -v "$PWD/tests/smoke_tests/data:/app/tests/smoke_tests/data" \
  pufferdrive-smoke "${TESTS[@]}"

echo ">>> Done. Updated goldens:"
git -c core.fileMode=false status --short tests/smoke_tests/data/
