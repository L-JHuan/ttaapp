#!/usr/bin/env bash
set -euo pipefail

: "${INDUSTRIAL_JSONL:?Set INDUSTRIAL_JSONL}"
: "${PROCESSED_ROOT:?Set PROCESSED_ROOT}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${TRAIN_END:?Set TRAIN_END}"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SPARK_SUBMIT_BIN=${SPARK_SUBMIT_BIN:-spark-submit}
SPARK_MASTER=${SPARK_MASTER:-}
SPARK_SHUFFLE_PARTITIONS=${SPARK_SHUFFLE_PARTITIONS:-2000}
SPARK_OUTPUT_PARTITIONS=${SPARK_OUTPUT_PARTITIONS:-256}
SPARK_MAPPING_PARTITIONS=${SPARK_MAPPING_PARTITIONS:-256}
SPARK_DRIVER_MEMORY=${SPARK_DRIVER_MEMORY:-8g}
INDUSTRIAL_TIMEZONE_OFFSET_MINUTES=${INDUSTRIAL_TIMEZONE_OFFSET_MINUTES:-480}
N_COARSE_REGIONS=${N_COARSE_REGIONS:-64}
N_FINE_REGIONS=${N_FINE_REGIONS:-256}
SID_CSV="$RUN_ROOT/codebook/tap_sid.csv"

mkdir -p "$RUN_ROOT/codebook" "$RUN_ROOT/data"

SPARK_ARGS=(--driver-memory "$SPARK_DRIVER_MEMORY")
if [[ -n "$SPARK_MASTER" ]]; then
  SPARK_ARGS+=(--master "$SPARK_MASTER")
fi
if [[ -n "${SPARK_SUBMIT_OPTIONS:-}" ]]; then
  read -r -a EXTRA_SPARK_ARGS <<< "$SPARK_SUBMIT_OPTIONS"
  SPARK_ARGS+=("${EXTRA_SPARK_ARGS[@]}")
fi

"$SPARK_SUBMIT_BIN" "${SPARK_ARGS[@]}" \
  "$REPO_ROOT/tap_sid/prepare_industrial_spark.py" \
  --input "$INDUSTRIAL_JSONL" \
  --output_dir "$PROCESSED_ROOT" \
  --train_end "$TRAIN_END" \
  --timezone_offset_minutes "$INDUSTRIAL_TIMEZONE_OFFSET_MINUTES" \
  --max_sequence_length 50 \
  --keep_last_k_train 5 \
  --shuffle_partitions "$SPARK_SHUFFLE_PARTITIONS" \
  --output_partitions "$SPARK_OUTPUT_PARTITIONS" \
  --mapping_partitions "$SPARK_MAPPING_PARTITIONS"

"$SPARK_SUBMIT_BIN" "${SPARK_ARGS[@]}" \
  "$REPO_ROOT/tap_sid/build_tap_sid.py" \
  --catalog_parquet "$PROCESSED_ROOT/metadata/catalog" \
  --output_csv "$SID_CSV" \
  --report_json "$RUN_ROOT/codebook/tap_sid_report.json" \
  --n_coarse_regions "$N_COARSE_REGIONS" \
  --n_fine_regions "$N_FINE_REGIONS" \
  --seed 2024

"$SPARK_SUBMIT_BIN" "${SPARK_ARGS[@]}" \
  "$REPO_ROOT/tap_sid/build_llm_data_spark.py" \
  --sid_csv "$SID_CSV" \
  --sequence_root "$PROCESSED_ROOT/sequence_parquet" \
  --output_dir "$RUN_ROOT/data" \
  --shuffle_partitions "$SPARK_SHUFFLE_PARTITIONS" \
  --output_partitions "$SPARK_OUTPUT_PARTITIONS"

echo "Spark preprocessing completed."
echo "Protocol report: $PROCESSED_ROOT/spark_protocol_report.json"
echo "Training shards: $RUN_ROOT/data/llm_train.jsonl"
echo "Test shards: $RUN_ROOT/data/llm_test.jsonl"
