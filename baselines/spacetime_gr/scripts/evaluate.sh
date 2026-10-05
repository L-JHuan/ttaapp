#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT}"
BASELINE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "$BASELINE_ROOT/../.." && pwd)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SEMANTIC_CODES="$RUN_ROOT/codebook/spacetime_gr_sid.csv"
exec bash "$REPO_ROOT/scripts/evaluate.sh"
