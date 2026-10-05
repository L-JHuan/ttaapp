#!/usr/bin/env bash
set -euo pipefail

: "${PROCESSED_ROOT:?Set PROCESSED_ROOT to an existing TAP preprocessing directory}"
: "${RUN_ROOT:?Set RUN_ROOT}"

BASELINE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "$BASELINE_ROOT/../.." && pwd)
export PYTHONPATH="$BASELINE_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BLOCK_SIZE_KM=${SPACETIME_BLOCK_SIZE_KM:-5.0}
KEEP_LAST_K_TRAIN=${KEEP_LAST_K_TRAIN:-5}
SID_CSV="$RUN_ROOT/codebook/spacetime_gr_sid.csv"
mkdir -p "$RUN_ROOT/codebook" "$RUN_ROOT/data"

python -m spacetime_gr.build_spacetime_sid \
  --poi_info "$PROCESSED_ROOT/poi_info.csv" \
  --output_csv "$SID_CSV" \
  --report_json "$RUN_ROOT/codebook/spacetime_gr_report.json" \
  --block_size_km "$BLOCK_SIZE_KM"

LLM_ARGS=()
if [[ "${NO_VALIDATION:-0}" == "1" ]]; then
  LLM_ARGS+=(--no_validation)
fi
python -m tap_sid.build_llm_data \
  --sid_csv "$SID_CSV" \
  --split_dir "$PROCESSED_ROOT/v1_sequence" \
  --output_dir "$RUN_ROOT/data" \
  --keep_last_k_train "$KEEP_LAST_K_TRAIN" \
  "${LLM_ARGS[@]}"

echo "Spacetime-GR adapted codebook and SFT data completed: $RUN_ROOT"
