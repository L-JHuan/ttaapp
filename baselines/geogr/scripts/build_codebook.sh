#!/usr/bin/env bash
set -euo pipefail

: "${PROCESSED_ROOT:?Set PROCESSED_ROOT to an existing TAP preprocessing directory}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${GEOGR_CODEBOOK_SIZE:?Set GEOGR_CODEBOOK_SIZE (paper: NYC=32, TKY=64)}"

BASELINE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_ROOT=$(cd "$BASELINE_ROOT/../.." && pwd)
export PYTHONPATH="$BASELINE_ROOT:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

SID_CSV="$RUN_ROOT/codebook/geogr_sid.csv"
INITIAL_EMBEDDINGS="$RUN_ROOT/embeddings/geogr_initial_embeddings.npz"
mkdir -p "$RUN_ROOT/codebook" "$RUN_ROOT/data" "$RUN_ROOT/embeddings"

EMBEDDING_ARGS=()
if [[ -n "${GEOGR_INITIAL_EMBEDDINGS_NPZ:-}" ]]; then
  EMBEDDING_ARGS+=(--initial_embeddings_npz "$GEOGR_INITIAL_EMBEDDINGS_NPZ")
else
  : "${GEOGR_ENCODER_MODEL:?Set GEOGR_ENCODER_MODEL or GEOGR_INITIAL_EMBEDDINGS_NPZ}"
  EMBEDDING_ARGS+=(
    --encoder_model "$GEOGR_ENCODER_MODEL"
    --encoder_device "${GEOGR_ENCODER_DEVICE:-cuda:0}"
    --encoder_backend "${GEOGR_ENCODER_BACKEND:-sentence_transformers}"
    --encoder_dtype "${GEOGR_ENCODER_DTYPE:-auto}"
    --encoder_batch_size "${GEOGR_ENCODER_BATCH_SIZE:-16}"
    --encoder_max_length "${GEOGR_ENCODER_MAX_LENGTH:-128}"
    --save_initial_embeddings_npz "$INITIAL_EMBEDDINGS"
  )
fi

CATALOG_ARGS=()
if [[ -n "${GEOGR_ID_MAPPINGS:-}" ]]; then
  CATALOG_ARGS+=(--id_mappings "$GEOGR_ID_MAPPINGS")
elif [[ -f "$PROCESSED_ROOT/id_mappings.json" ]]; then
  CATALOG_ARGS+=(--id_mappings "$PROCESSED_ROOT/id_mappings.json")
fi

python -m geogr.build_geogr_sid \
  --poi_info "$PROCESSED_ROOT/poi_info.csv" \
  --role_priors "$PROCESSED_ROOT/role_priors.csv" \
  --train_sequences "$PROCESSED_ROOT/v1_sequence/train_poi_sequence.csv" \
  --output_csv "$SID_CSV" \
  --report_json "$RUN_ROOT/codebook/geogr_report.json" \
  --max_distance_km "${GEOGR_MAX_DISTANCE_KM:-3.0}" \
  --min_common_users "${GEOGR_MIN_COMMON_USERS:-2}" \
  --swing_alpha "${GEOGR_SWING_ALPHA:-1.0}" \
  --max_pairs_per_poi "${GEOGR_MAX_PAIRS_PER_POI:-50}" \
  --contrastive_device "${GEOGR_CONTRASTIVE_DEVICE:-cuda:0}" \
  --contrastive_epochs "${GEOGR_CONTRASTIVE_EPOCHS:-10}" \
  --contrastive_batch_size "${GEOGR_CONTRASTIVE_BATCH_SIZE:-256}" \
  --contrastive_learning_rate "${GEOGR_CONTRASTIVE_LEARNING_RATE:-1e-4}" \
  --contrastive_temperature "${GEOGR_CONTRASTIVE_TEMPERATURE:-0.07}" \
  --codebook_size "$GEOGR_CODEBOOK_SIZE" \
  --num_layers 3 \
  --seed "${GEOGR_SEED:-2024}" \
  "${CATALOG_ARGS[@]}" \
  "${EMBEDDING_ARGS[@]}"

LLM_ARGS=()
if [[ "${NO_VALIDATION:-0}" == "1" ]]; then
  LLM_ARGS+=(--no_validation)
fi
python -m tap_sid.build_llm_data \
  --sid_csv "$SID_CSV" \
  --split_dir "$PROCESSED_ROOT/v1_sequence" \
  --output_dir "$RUN_ROOT/data" \
  --keep_last_k_train "${KEEP_LAST_K_TRAIN:-5}" \
  "${LLM_ARGS[@]}"

echo "GeoGR adapted codebook and SFT data completed: $RUN_ROOT"
