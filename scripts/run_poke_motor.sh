#!/usr/bin/env bash
set -euo pipefail
# Wrapper to run scripts/poke_motor.py with the repository root on PYTHONPATH
# Usage: ./scripts/run_poke_motor.sh [args...]

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT"

# Force headless GUI so Qt doesn't abort in WSL/CI when no display is available.
# This uses the offscreen Qt plugin and Matplotlib Agg backend.
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
export MPLBACKEND=${MPLBACKEND:-Agg}

python3 "${REPO_ROOT}/scripts/poke_motor.py" "$@"
