#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source .venv/bin/activate

for generation_name in regents_wod_motion_val_pdm regents_wod_motion_val; do
    python -m pufferlib.pufferl regents puffer_drive "$generation_name" \
        --scenario-count 1000 --experiment-name first_1000
done
