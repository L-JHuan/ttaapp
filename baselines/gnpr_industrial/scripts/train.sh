#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL}"
: "${RUN_ROOT:?Set RUN_ROOT}"

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
TRAIN_EVAL_BATCH_SIZE=${TRAIN_EVAL_BATCH_SIZE:-1}
TRAIN_GRAD_ACCUM=${TRAIN_GRAD_ACCUM:-8}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-3}
TRAIN_LEARNING_RATE=${TRAIN_LEARNING_RATE:-1e-5}
TRAIN_CUTOFF_LEN=${TRAIN_CUTOFF_LEN:-2048}
TRAIN_LORA_R=${TRAIN_LORA_R:-16}
TRAIN_LORA_ALPHA=${TRAIN_LORA_ALPHA:-32}
TRAIN_LORA_DROPOUT=${TRAIN_LORA_DROPOUT:-0.1}
TRAIN_SEED=${TRAIN_SEED:-42}
TRAIN_LOGGING_STEPS=${TRAIN_LOGGING_STEPS:-50}
CHECKPOINT_DIR="$RUN_ROOT/checkpoint"
TRAIN_DATASET=${TRAIN_DATASET:-"$RUN_ROOT/data/llm_train.json"}
VALID_DATASET=${VALID_DATASET:-"$RUN_ROOT/data/llm_val.json"}
VALID_ARGS=()

if [[ ! -e "$TRAIN_DATASET" && -d "$RUN_ROOT/data/llm_train.jsonl" ]]; then
  TRAIN_DATASET="$RUN_ROOT/data/llm_train.jsonl"
fi
if [[ ! -e "$VALID_DATASET" && -d "$RUN_ROOT/data/llm_val.jsonl" ]]; then
  VALID_DATASET="$RUN_ROOT/data/llm_val.jsonl"
fi
test -e "$TRAIN_DATASET"
if [[ -e "$VALID_DATASET" ]]; then
  VALID_ARGS+=(--valid_dataset "$VALID_DATASET" --eval_during_train)
elif [[ "${NO_VALIDATION:-0}" != "1" ]]; then
  echo "Validation dataset not found: $VALID_DATASET" >&2
  exit 2
fi

mkdir -p "$CHECKPOINT_DIR"
TQDM_MININTERVAL=60 TQDM_MINITERS=50 \
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m gnpr_baseline.train_generative_sid \
  --base_model "$BASE_MODEL" \
  --train_dataset "$TRAIN_DATASET" \
  --output_dir "$CHECKPOINT_DIR" \
  --batch_size "$TRAIN_BATCH_SIZE" \
  --eval_batch_size "$TRAIN_EVAL_BATCH_SIZE" \
  --grad_accum "$TRAIN_GRAD_ACCUM" \
  --num_train_epochs "$TRAIN_EPOCHS" \
  --learning_rate "$TRAIN_LEARNING_RATE" \
  --cutoff_len "$TRAIN_CUTOFF_LEN" \
  --lm_loss_weight 1.0 \
  --alpha_prefix 0.0 \
  --head_dropout 0.1 \
  --lora_r "$TRAIN_LORA_R" \
  --lora_alpha "$TRAIN_LORA_ALPHA" \
  --lora_dropout "$TRAIN_LORA_DROPOUT" \
  --gradient_checkpointing \
  --seed "$TRAIN_SEED" \
  --logging_steps "$TRAIN_LOGGING_STEPS" \
  "${VALID_ARGS[@]}"
