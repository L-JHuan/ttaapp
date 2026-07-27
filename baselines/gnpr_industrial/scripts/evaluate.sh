#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL}"
: "${RUN_ROOT:?Set RUN_ROOT}"

EVAL_DIR="$RUN_ROOT/eval"
TEST_DATASET=${TEST_DATASET:-"$RUN_ROOT/data/llm_test.json"}
if [[ ! -e "$TEST_DATASET" && -d "$RUN_ROOT/data/llm_test.jsonl" ]]; then
  TEST_DATASET="$RUN_ROOT/data/llm_test.jsonl"
fi
test -e "$TEST_DATASET"
test -f "$RUN_ROOT/codebook/gnpr_sid.csv"
test -d "$RUN_ROOT/checkpoint/final_sft"

EVAL_GPUS=${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-}}
if [[ -z "$EVAL_GPUS" ]]; then
  echo "Set EVAL_GPUS, for example 0,1,2,3" >&2
  exit 2
fi
IFS=',' read -r -a RAW_GPU_IDS <<< "$EVAL_GPUS"
GPU_IDS=()
declare -A SEEN_GPUS=()
for raw_gpu in "${RAW_GPU_IDS[@]}"; do
  gpu="${raw_gpu//[[:space:]]/}"
  if [[ -z "$gpu" || -n "${SEEN_GPUS[$gpu]:-}" ]]; then
    echo "Invalid or duplicate GPU in EVAL_GPUS=$EVAL_GPUS" >&2
    exit 2
  fi
  SEEN_GPUS[$gpu]=1
  GPU_IDS+=("$gpu")
done

NUM_SHARDS=${#GPU_IDS[@]}
EVAL_RUN_ID=${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
SHARD_DIR="$EVAL_DIR/shards/$EVAL_RUN_ID"
if [[ -e "$SHARD_DIR" ]]; then
  echo "Evaluation run already exists: $SHARD_DIR" >&2
  exit 2
fi
mkdir -p "$SHARD_DIR/logs"

EVAL_CUTOFF_LEN=${EVAL_CUTOFF_LEN:-2048}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-32}
EVAL_NUM_BEAMS=${EVAL_NUM_BEAMS:-10}
EVAL_K=${EVAL_K:-10}
EVAL_SEED=${EVAL_SEED:-42}
NO_CACHE_ARGS=()
if [[ "${EVAL_NO_CACHE:-0}" == "1" ]]; then
  NO_CACHE_ARGS+=(--no_cache)
fi

python -m gnpr_baseline.split_eval_shards \
  --dataset "$TEST_DATASET" \
  --output_dir "$SHARD_DIR" \
  --num_shards "$NUM_SHARDS"

PIDS=()
for ((shard_index = 0; shard_index < NUM_SHARDS; shard_index++)); do
  gpu="${GPU_IDS[$shard_index]}"
  shard_tag=$(printf "%05d" "$shard_index")
  shard_log="$SHARD_DIR/logs/shard_${shard_tag}.log"
  CUDA_VISIBLE_DEVICES="$gpu" TQDM_MININTERVAL=60 TQDM_MINITERS=20 \
    python -m gnpr_baseline.evaluate_generative_sid \
      --base_model "$BASE_MODEL" \
      --adapter_dir "$RUN_ROOT/checkpoint/final_sft" \
      --dataset "$SHARD_DIR/dataset_${shard_tag}.json" \
      --semantic_codes "$RUN_ROOT/codebook/gnpr_sid.csv" \
      --output_predictions "$SHARD_DIR/predictions_${shard_tag}.json" \
      --output_metrics "$SHARD_DIR/metrics_${shard_tag}.json" \
      --cutoff_len "$EVAL_CUTOFF_LEN" \
      --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
      --num_beams "$EVAL_NUM_BEAMS" \
      --k "$EVAL_K" \
      --device cuda:0 \
      --seed "$EVAL_SEED" \
      --shard_index "$shard_index" \
      --num_shards "$NUM_SHARDS" \
      "${NO_CACHE_ARGS[@]}" \
      >"$shard_log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for ((shard_index = 0; shard_index < NUM_SHARDS; shard_index++)); do
  if ! wait "${PIDS[$shard_index]}"; then
    echo "Evaluation shard $shard_index failed" >&2
    FAILED=1
  fi
done
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi

python -m gnpr_baseline.merge_eval_shards \
  --dataset "$TEST_DATASET" \
  --shard_dir "$SHARD_DIR" \
  --num_shards "$NUM_SHARDS" \
  --output_predictions "$EVAL_DIR/test_predictions.json" \
  --output_metrics "$EVAL_DIR/test_metrics.json" \
  --k "$EVAL_K"

python -m json.tool "$EVAL_DIR/test_predictions.json" >/dev/null
python -m json.tool "$EVAL_DIR/test_metrics.json" >/dev/null
echo "GNPR evaluation completed with $NUM_SHARDS GPU shard(s)."
