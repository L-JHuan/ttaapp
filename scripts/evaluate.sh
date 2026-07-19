#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL}"
: "${RUN_ROOT:?Set RUN_ROOT}"

DEVICE=${DEVICE:-cuda:0}
EVAL_DIR="$RUN_ROOT/eval"
mkdir -p "$EVAL_DIR"

TQDM_MININTERVAL=60 TQDM_MINITERS=20 \
python -m tap_sid.evaluate_tap_sid \
  --base_model "$BASE_MODEL" \
  --adapter_dir "$RUN_ROOT/checkpoint/final_sft" \
  --dataset "$RUN_ROOT/data/llm_test.json" \
  --semantic_codes "$RUN_ROOT/codebook/tap_sid.csv" \
  --output_predictions "$EVAL_DIR/test_predictions.json" \
  --output_metrics "$EVAL_DIR/test_metrics.json" \
  --cutoff_len 2048 \
  --max_new_tokens 32 \
  --num_beams 10 \
  --k 10 \
  --device "$DEVICE" \
  --seed 42

python -m json.tool "$EVAL_DIR/test_predictions.json" >/dev/null
python -m json.tool "$EVAL_DIR/test_metrics.json" >/dev/null

