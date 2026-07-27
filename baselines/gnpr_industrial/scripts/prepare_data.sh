#!/usr/bin/env bash
set -euo pipefail

: "${PROCESSED_ROOT:?Set PROCESSED_ROOT}"
: "${TRAIN_END:?Set TRAIN_END}"

NO_VALIDATION=${NO_VALIDATION:-0}
DEFAULT_TIMEZONE_OFFSET_MINUTES=${DEFAULT_TIMEZONE_OFFSET_MINUTES:-480}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-50}
MIN_HISTORY_LENGTH=${MIN_HISTORY_LENGTH:-1}
PREPARE_ARGS=()

if [[ "$NO_VALIDATION" == "1" ]]; then
  PREPARE_ARGS+=(--no_validation)
else
  : "${VALIDATION_END:?Set VALIDATION_END or set NO_VALIDATION=1}"
  PREPARE_ARGS+=(--validation_end "$VALIDATION_END")
fi

if [[ "${REUSE_PREPROCESSED:-0}" == "1" ]]; then
  test -f "$PROCESSED_ROOT/protocol_report.json"
  test -f "$PROCESSED_ROOT/id_mappings.json"
  test -d "$PROCESSED_ROOT/v1_sequence"
  echo "Reusing existing chronological preprocessing: $PROCESSED_ROOT"
  exit 0
fi

if [[ -n "${INDUSTRIAL_JSONL:-}" ]]; then
  CONVERTED_DIR="$PROCESSED_ROOT/converted_raw"
  EVENTS="$CONVERTED_DIR/events.csv"
  POIS="$CONVERTED_DIR/pois.csv"
  mkdir -p "$CONVERTED_DIR"
  python -m gnpr_baseline.convert_industrial_jsonl \
    --input "$INDUSTRIAL_JSONL" \
    --events_output "$EVENTS" \
    --pois_output "$POIS" \
    --report_output "$CONVERTED_DIR/conversion_report.json" \
    --timezone_offset_minutes "$DEFAULT_TIMEZONE_OFFSET_MINUTES"
else
  : "${EVENTS:?Set EVENTS or INDUSTRIAL_JSONL}"
  : "${POIS:?Set POIS or INDUSTRIAL_JSONL}"
fi

python -m gnpr_baseline.prepare_realworld_data \
  --events "$EVENTS" \
  --pois "$POIS" \
  --output_dir "$PROCESSED_ROOT" \
  --train_end "$TRAIN_END" \
  --default_timezone_offset_minutes "$DEFAULT_TIMEZONE_OFFSET_MINUTES" \
  --max_sequence_length "$MAX_SEQUENCE_LENGTH" \
  --min_history_length "$MIN_HISTORY_LENGTH" \
  --collapse_consecutive_same_poi \
  --catalog_scope train_seen \
  "${PREPARE_ARGS[@]}"

echo "Prepared shared chronological data: $PROCESSED_ROOT"
