#!/usr/bin/env bash
set -euo pipefail

: "${INDUSTRIAL_JSONL:?Set INDUSTRIAL_JSONL}"
: "${PROCESSED_ROOT:?Set PROCESSED_ROOT}"
: "${TRAIN_END:?Set TRAIN_END}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SPARK_SUBMIT_BIN=${SPARK_SUBMIT_BIN:-spark-submit}
SPARK_MASTER=${SPARK_MASTER:-}
SPARK_DRIVER_MEMORY=${SPARK_DRIVER_MEMORY:-8g}
SPARK_SHUFFLE_PARTITIONS=${SPARK_SHUFFLE_PARTITIONS:-2000}
SPARK_OUTPUT_PARTITIONS=${SPARK_OUTPUT_PARTITIONS:-256}
SPARK_MAPPING_PARTITIONS=${SPARK_MAPPING_PARTITIONS:-256}
DEFAULT_TIMEZONE_OFFSET_MINUTES=${DEFAULT_TIMEZONE_OFFSET_MINUTES:-480}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-50}
MIN_HISTORY_LENGTH=${MIN_HISTORY_LENGTH:-1}
KEEP_LAST_K_TRAIN=${KEEP_LAST_K_TRAIN:-5}
VALIDATION_ARGS=()

if [[ "${REUSE_PREPROCESSED:-0}" == "1" ]]; then
  test -f "$PROCESSED_ROOT/spark_protocol_report.json"
  test -d "$PROCESSED_ROOT/sequence_parquet"
  test -d "$PROCESSED_ROOT/metadata/catalog"
  echo "Reusing existing TAP Spark preprocessing: $PROCESSED_ROOT"
  exit 0
fi

if [[ "${NO_VALIDATION:-0}" == "1" ]]; then
  if [[ -n "${VALIDATION_END:-}" ]]; then
    echo "NO_VALIDATION=1 requires empty VALIDATION_END" >&2
    exit 2
  fi
elif [[ -n "${VALIDATION_END:-}" ]]; then
  VALIDATION_ARGS+=(--validation_end "$VALIDATION_END")
else
  echo "Set VALIDATION_END or NO_VALIDATION=1" >&2
  exit 2
fi

SPARK_ARGS=(--driver-memory "$SPARK_DRIVER_MEMORY")
if [[ -n "$SPARK_MASTER" ]]; then
  SPARK_ARGS+=(--master "$SPARK_MASTER")
fi
if [[ -n "${SPARK_SUBMIT_OPTIONS:-}" ]]; then
  read -r -a EXTRA_SPARK_ARGS <<< "$SPARK_SUBMIT_OPTIONS"
  SPARK_ARGS+=("${EXTRA_SPARK_ARGS[@]}")
fi

"$SPARK_SUBMIT_BIN" "${SPARK_ARGS[@]}" \
  "$REPO_ROOT/gnpr_baseline/prepare_industrial_spark.py" \
  --input "$INDUSTRIAL_JSONL" \
  --output_dir "$PROCESSED_ROOT" \
  --train_end "$TRAIN_END" \
  "${VALIDATION_ARGS[@]}" \
  --timezone_offset_minutes "$DEFAULT_TIMEZONE_OFFSET_MINUTES" \
  --max_sequence_length "$MAX_SEQUENCE_LENGTH" \
  --min_history_length "$MIN_HISTORY_LENGTH" \
  --keep_last_k_train "$KEEP_LAST_K_TRAIN" \
  --shuffle_partitions "$SPARK_SHUFFLE_PARTITIONS" \
  --output_partitions "$SPARK_OUTPUT_PARTITIONS" \
  --mapping_partitions "$SPARK_MAPPING_PARTITIONS"

echo "Prepared shared Spark data: $PROCESSED_ROOT"
