#!/usr/bin/env bash
set -euo pipefail

: "${PROCESSED_ROOT:?Set PROCESSED_ROOT}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${TRAIN_END:?Set TRAIN_END}"
: "${CATEGORY_MODEL:?Set CATEGORY_MODEL}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EMBED_DIR="$RUN_ROOT/embeddings"
CODEBOOK_DIR="$RUN_ROOT/codebook"
RQ_DIR="$CODEBOOK_DIR/rqvae"
SID_CSV="$CODEBOOK_DIR/gnpr_sid.csv"
mkdir -p "$EMBED_DIR" "$RQ_DIR" "$RUN_ROOT/data"

CATEGORY_DEVICE=${CATEGORY_DEVICE:-cpu}
CATEGORY_DIM=${CATEGORY_DIM:-64}
CODEBOOK_DEVICE=${CODEBOOK_DEVICE:-cuda:0}
CODEBOOK_EPOCHS=${CODEBOOK_EPOCHS:-3000}
CODEBOOK_BATCH_SIZE=${CODEBOOK_BATCH_SIZE:-128}
CODEBOOK_NUM_WORKERS=${CODEBOOK_NUM_WORKERS:-4}
CODEBOOK_EVAL_STEP=${CODEBOOK_EVAL_STEP:-10}
CODEBOOK_MIN_SELECT_EPOCH=${CODEBOOK_MIN_SELECT_EPOCH:-200}
DEFAULT_TIMEZONE_OFFSET_MINUTES=${DEFAULT_TIMEZONE_OFFSET_MINUTES:-480}
KEEP_LAST_K_TRAIN=${KEEP_LAST_K_TRAIN:-5}

if [[ -d "$PROCESSED_ROOT/sequence_parquet" ]]; then
  SPARK_SUBMIT_BIN=${SPARK_SUBMIT_BIN:-spark-submit}
  SPARK_DRIVER_MEMORY=${SPARK_DRIVER_MEMORY:-8g}
  SPARK_SHUFFLE_PARTITIONS=${SPARK_SHUFFLE_PARTITIONS:-2000}
  SPARK_OUTPUT_PARTITIONS=${SPARK_OUTPUT_PARTITIONS:-256}
  SPARK_ARGS=(--driver-memory "$SPARK_DRIVER_MEMORY")
  if [[ -n "${SPARK_MASTER:-}" ]]; then
    SPARK_ARGS+=(--master "$SPARK_MASTER")
  fi
  if [[ -n "${SPARK_SUBMIT_OPTIONS:-}" ]]; then
    read -r -a EXTRA_SPARK_ARGS <<< "$SPARK_SUBMIT_OPTIONS"
    SPARK_ARGS+=("${EXTRA_SPARK_ARGS[@]}")
  fi

  "$SPARK_SUBMIT_BIN" "${SPARK_ARGS[@]}" \
    "$REPO_ROOT/gnpr_baseline/build_poi_embeddings_spark.py" \
    --processed_root "$PROCESSED_ROOT" \
    --train_end "$TRAIN_END" \
    --category_model "$CATEGORY_MODEL" \
    --output_dir "$EMBED_DIR" \
    --timezone_offset_minutes "$DEFAULT_TIMEZONE_OFFSET_MINUTES" \
    --category_dim "$CATEGORY_DIM" \
    --device "$CATEGORY_DEVICE" \
    --shuffle_partitions "$SPARK_SHUFFLE_PARTITIONS"
  VALIDATE_ARGS=(--spark_catalog "$PROCESSED_ROOT/metadata/catalog")
  DATA_BUILD_MODE=spark
else
  if [[ -z "${EVENTS:-}" ]]; then
    EVENTS="$PROCESSED_ROOT/converted_raw/events.csv"
  fi
  if [[ -z "${POIS:-}" ]]; then
    POIS="$PROCESSED_ROOT/converted_raw/pois.csv"
  fi
  test -f "$EVENTS"
  test -f "$POIS"
  test -f "$PROCESSED_ROOT/id_mappings.json"
  python -m gnpr_baseline.build_poi_embeddings \
    --events "$EVENTS" \
    --pois "$POIS" \
    --id_mappings "$PROCESSED_ROOT/id_mappings.json" \
    --train_end "$TRAIN_END" \
    --category_model "$CATEGORY_MODEL" \
    --output_dir "$EMBED_DIR" \
    --timezone_offset_minutes "$DEFAULT_TIMEZONE_OFFSET_MINUTES" \
    --category_dim "$CATEGORY_DIM" \
    --device "$CATEGORY_DEVICE"
  VALIDATE_ARGS=(--id_mappings "$PROCESSED_ROOT/id_mappings.json")
  DATA_BUILD_MODE=csv
fi

python -m gnpr_baseline.train_codebook \
  --data_path "$EMBED_DIR/poi_Emb_dict.pkl" \
  --output_dir "$RQ_DIR" \
  --device "$CODEBOOK_DEVICE" \
  --epochs "$CODEBOOK_EPOCHS" \
  --batch_size "$CODEBOOK_BATCH_SIZE" \
  --num_workers "$CODEBOOK_NUM_WORKERS" \
  --eval_step "$CODEBOOK_EVAL_STEP" \
  --min_select_epoch "$CODEBOOK_MIN_SELECT_EPOCH"

python -m gnpr_baseline.export_codebook \
  --data_path "$EMBED_DIR/poi_Emb_dict.pkl" \
  --checkpoint "$RQ_DIR/best_loss_model.pth" \
  --output_csv "$SID_CSV" \
  --device "$CODEBOOK_DEVICE" \
  --batch_size "$CODEBOOK_BATCH_SIZE" \
  --num_workers "$CODEBOOK_NUM_WORKERS"

if [[ "$DATA_BUILD_MODE" == "spark" ]]; then
  "$SPARK_SUBMIT_BIN" "${SPARK_ARGS[@]}" \
    "$REPO_ROOT/gnpr_baseline/validate_codebook.py" \
    --sid_csv "$SID_CSV" \
    "${VALIDATE_ARGS[@]}"
  "$SPARK_SUBMIT_BIN" "${SPARK_ARGS[@]}" \
    "$REPO_ROOT/gnpr_baseline/build_llm_data_spark.py" \
    --sid_csv "$SID_CSV" \
    --sequence_root "$PROCESSED_ROOT/sequence_parquet" \
    --output_dir "$RUN_ROOT/data" \
    --shuffle_partitions "$SPARK_SHUFFLE_PARTITIONS" \
    --output_partitions "$SPARK_OUTPUT_PARTITIONS"
else
  python -m gnpr_baseline.validate_codebook \
    --sid_csv "$SID_CSV" \
    "${VALIDATE_ARGS[@]}"
  LLM_ARGS=()
  if [[ "${NO_VALIDATION:-0}" == "1" ]]; then
    LLM_ARGS+=(--no_validation)
  fi
  python -m gnpr_baseline.build_llm_data \
    --sid_csv "$SID_CSV" \
    --split_dir "$PROCESSED_ROOT/v1_sequence" \
    --output_dir "$RUN_ROOT/data" \
    --keep_last_k_train "$KEEP_LAST_K_TRAIN" \
    "${LLM_ARGS[@]}"
fi

echo "GNPR codebook and SFT data completed: $RUN_ROOT"
