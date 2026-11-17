#!/bin/bash
# Moved to scripts/slurm; jobs submitted before the move still start it from here.
exec bash "$(dirname "${BASH_SOURCE[0]}")/../slurm/build_ext_if_changed.sh" "$@"
