#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "$BASELINE_ROOT/../.." && pwd)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec bash "$REPO_ROOT/scripts/train.sh"
