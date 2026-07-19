#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?Set BASE_MODEL}"
: "${RUN_ROOT:?Set RUN_ROOT}"

NPROC_PER_NODE=${NPROC_PER_NODE:-1}
CHECKPOINT_DIR="$RUN_ROOT/checkpoint"
mkdir -p "$CHECKPOINT_DIR"

TQDM_MININTERVAL=60 TQDM_MINITERS=50 \
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m tap_sid.train_tap_sid \
  --base_model "$BASE_MODEL" \
  --train_dataset "$RUN_ROOT/data/llm_train.json" \
  --valid_dataset "$RUN_ROOT/data/llm_val.json" \
  --output_dir "$CHECKPOINT_DIR" \
  --batch_size 1 \
  --eval_batch_size 1 \
  --grad_accum 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --cutoff_len 2048 \
  --lm_loss_weight 1.0 \
  --alpha_prefix 0.0 \
  --head_dropout 0.1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --gradient_checkpointing \
  --seed 42 \
  --logging_steps 50

