#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${1:?Usage: run_matched_pipeline.sh /absolute/path/to/geogr.env}
source "$ENV_FILE"

required=(REPO_ROOT PROCESSED_ROOT RUN_ROOT BASE_MODEL DATASET CODEBOOK_SIZE GPUS)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing required variable: $name" >&2; exit 2; }
done

PYTHON_BIN=${PYTHON_BIN:-python}
WAIT_SECONDS=${WAIT_SECONDS:-10}
GPU_MEMORY_THRESHOLD_MIB=${GPU_MEMORY_THRESHOLD_MIB:-300}
GPU_UTIL_THRESHOLD=${GPU_UTIL_THRESHOLD:-5}
SFT_EPOCHS=${SFT_EPOCHS:-3}
SFT_GRAD_ACCUM=${SFT_GRAD_ACCUM:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SEED=${SEED:-42}
KEEP_LAST_K_TRAIN=${KEEP_LAST_K_TRAIN:-5}
TEST_BEAMS=${TEST_BEAMS:-10}
EVAL_RUN_SUFFIX=${EVAL_RUN_SUFFIX:-}
P2P_BATCH_SIZE=${P2P_BATCH_SIZE:-4}
P2P_GRAD_ACCUM=${P2P_GRAD_ACCUM:-2}
P2P_EPOCHS=${P2P_EPOCHS:-3}
P2P_MAX_PAIRS_PER_POI=${P2P_MAX_PAIRS_PER_POI:-50}
EM_ITERATIONS=${EM_ITERATIONS:-2}
EM_EPOCHS=${EM_EPOCHS:-2}
EM_BEAMS=${EM_BEAMS:-20}
CPT_EPOCHS=${CPT_EPOCHS:-2}
CPT_GRAD_ACCUM=${CPT_GRAD_ACCUM:-8}

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
[[ ${#GPU_IDS[@]} -eq 2 ]] || { echo "GPUS must contain exactly two GPU IDs" >&2; exit 2; }
for index in 0 1; do GPU_IDS[$index]=${GPU_IDS[$index]//[[:space:]]/}; done

POI_INFO="$PROCESSED_ROOT/poi_info.csv"
ROLE_PRIORS="$PROCESSED_ROOT/role_priors.csv"
SPLIT_DIR="$PROCESSED_ROOT/v1_sequence"
ID_MAPPINGS="$PROCESSED_ROOT/id_mappings.json"
ID_MAPPING_ARGS=()
[[ -s "$ID_MAPPINGS" ]] && ID_MAPPING_ARGS=(--id_mappings "$ID_MAPPINGS")

for path in "$POI_INFO" "$ROLE_PRIORS" "$SPLIT_DIR/train_poi_sequence.csv" \
  "$SPLIT_DIR/validation_poi_sequence.csv" "$SPLIT_DIR/test_poi_sequence.csv"; do
  [[ -s "$path" ]] || { echo "Required input missing: $path" >&2; exit 2; }
done
[[ -d "$BASE_MODEL" ]] || { echo "Base model missing: $BASE_MODEL" >&2; exit 2; }
[[ "$RUN_ROOT" != "/" && "$RUN_ROOT" != "$REPO_ROOT" ]] || { echo "Unsafe RUN_ROOT" >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/sid" "$RUN_ROOT/em" "$RUN_ROOT/data"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/geogr_full_pipeline:$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_MININTERVAL=60
export TQDM_MINITERS=50

timestamp() { date '+%Y-%m-%d %H:%M:%S %z'; }
log() { echo "[$(timestamp)] $*"; }
gpu_is_free() {
  local gpu=$1 memory utilization
  IFS=',' read -r memory utilization < <(
    nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
  )
  memory=${memory//[[:space:]]/}; utilization=${utilization//[[:space:]]/}
  (( memory <= GPU_MEMORY_THRESHOLD_MIB && utilization <= GPU_UTIL_THRESHOLD ))
}
wait_for_pair() {
  log "每 ${WAIT_SECONDS}s 检查 GPU ${GPU_IDS[*]}。"
  until gpu_is_free "${GPU_IDS[0]}" && gpu_is_free "${GPU_IDS[1]}"; do
    sleep "$WAIT_SECONDS"
  done
  log "GPU ${GPU_IDS[*]} 已空闲。"
}
json_ok() { "$PYTHON_BIN" -m json.tool "$1" >/dev/null; }
require_new_dir() {
  local path=$1
  if [[ -d "$path" && -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Incomplete or existing output requires a new path: $path" >&2
    exit 2
  fi
}

wait_for_pair

P2P_ROOT="$RUN_ROOT/p2p"
if [[ ! -s "$P2P_ROOT/p2p_encoder_report.json" || ! -s "$P2P_ROOT/refined_embeddings.npz" ]]; then
  require_new_dir "$P2P_ROOT"
  log "P2P: 两卡 Llama-3-8B LoRA 对比微调。"
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node=2 -m geogr_full_pipeline.train_p2p_encoder \
    --poi_info "$POI_INFO" --role_priors "$ROLE_PRIORS" "${ID_MAPPING_ARGS[@]}" \
    --train_sequences "$SPLIT_DIR/train_poi_sequence.csv" \
    --encoder_model "$BASE_MODEL" --output_dir "$P2P_ROOT" \
    --batch_size "$P2P_BATCH_SIZE" --grad_accum "$P2P_GRAD_ACCUM" \
    --encode_batch_size 8 --epochs "$P2P_EPOCHS" \
    --learning_rate "$LEARNING_RATE" --temperature 0.07 \
    --max_distance_km 3.0 --min_common_users 2 --swing_alpha 1.0 \
    --max_pairs_per_poi "$P2P_MAX_PAIRS_PER_POI" --seed 2024 \
    >"$RUN_ROOT/logs/p2p_train_gpu_pair.log" 2>&1
fi
json_ok "$P2P_ROOT/p2p_encoder_report.json"

INITIAL_SID="$RUN_ROOT/sid/initial_rq_sid.csv"
if [[ ! -s "$INITIAL_SID" ]]; then
  log "RQ: 构建三层初始码本。"
  "$PYTHON_BIN" -m geogr_full_pipeline.build_initial_rq_sid \
    --embeddings_npz "$P2P_ROOT/refined_embeddings.npz" \
    --output_csv "$INITIAL_SID" --report_json "$RUN_ROOT/sid/initial_rq_report.json" \
    --codebook_size "$CODEBOOK_SIZE" --seed 2024 \
    >"$RUN_ROOT/logs/initial_rq.log" 2>&1
fi
json_ok "$RUN_ROOT/sid/initial_rq_report.json"

CURRENT_SID="$INITIAL_SID"
PREVIOUS_EM_ADAPTER=""
for ((iteration=1; iteration<=EM_ITERATIONS; iteration++)); do
  ITER_ROOT="$RUN_ROOT/em/iteration_${iteration}"
  mkdir -p "$ITER_ROOT"
  EM_DATA="$ITER_ROOT/em_train.json"
  if [[ ! -s "$EM_DATA" ]]; then
    "$PYTHON_BIN" -m geogr_full_pipeline.build_em_data \
      --poi_info "$POI_INFO" --role_priors "$ROLE_PRIORS" "${ID_MAPPING_ARGS[@]}" \
      --sid_csv "$CURRENT_SID" --output_json "$EM_DATA" \
      --report_json "$ITER_ROOT/em_data_report.json"
  fi
  EM_CHECKPOINT="$ITER_ROOT/checkpoint"
  if [[ ! -s "$EM_CHECKPOINT/final_sft/adapter_config.json" ]]; then
    require_new_dir "$EM_CHECKPOINT"
    INIT_ARGS=()
    if [[ -n "$PREVIOUS_EM_ADAPTER" ]]; then
      INIT_ARGS=(--init_adapter_dir "$PREVIOUS_EM_ADAPTER" --tokenizer_path "$PREVIOUS_EM_ADAPTER")
    fi
    log "EM iteration $iteration: 描述到 SID 的两卡 LoRA 训练。"
    CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
      --standalone --nproc_per_node=2 -m geogr_full_pipeline.train_matched_sft \
      --base_model "$BASE_MODEL" "${INIT_ARGS[@]}" \
      --train_dataset "$EM_DATA" --output_dir "$EM_CHECKPOINT" \
      --batch_size 1 --eval_batch_size 1 --grad_accum 8 \
      --num_train_epochs "$EM_EPOCHS" --learning_rate "$LEARNING_RATE" \
      --cutoff_len 512 --lm_loss_weight 1.0 --alpha_prefix 0.0 \
      --head_dropout 0.1 --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 \
      --gradient_checkpointing --seed "$SEED" --logging_steps 120 \
      --attn_implementation sdpa \
      >"$RUN_ROOT/logs/em_${iteration}_train_gpu_pair.log" 2>&1
  fi
  EM_ADAPTER="$EM_CHECKPOINT/final_sft"

  SMOKE_REPORT="$ITER_ROOT/beam20_smoke_report.json"
  if [[ ! -s "$SMOKE_REPORT" ]]; then
    log "EM iteration $iteration: 单样本 beam-$EM_BEAMS 显存 smoke。"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PYTHON_BIN" -m geogr_full_pipeline.em_refine_sid \
      --poi_info "$POI_INFO" --role_priors "$ROLE_PRIORS" "${ID_MAPPING_ARGS[@]}" \
      --input_sid_csv "$CURRENT_SID" --base_model "$BASE_MODEL" \
      --adapter_dir "$EM_ADAPTER" --tokenizer_path "$EM_ADAPTER" \
      --output_sid_csv "$ITER_ROOT/unused_smoke_sid.csv" \
      --candidates_json "$ITER_ROOT/beam20_smoke_candidates.json" \
      --report_json "$SMOKE_REPORT" --codebook_size "$CODEBOOK_SIZE" \
      --num_beams "$EM_BEAMS" --device cuda:0 --limit 1 --candidates_only \
      >"$RUN_ROOT/logs/em_${iteration}_beam20_smoke_gpu${GPU_IDS[0]}.log" 2>&1
  fi
  json_ok "$SMOKE_REPORT"

  SHARD_PIDS=()
  for shard_index in 0 1; do
    SHARD_JSON="$ITER_ROOT/candidates_shard_${shard_index}.json"
    SHARD_REPORT="$ITER_ROOT/candidates_shard_${shard_index}_report.json"
    if [[ ! -s "$SHARD_JSON" || ! -s "$SHARD_REPORT" ]]; then
      gpu=${GPU_IDS[$shard_index]}
      log "EM iteration $iteration: GPU$gpu 生成候选分片 $shard_index/2。"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m geogr_full_pipeline.em_refine_sid \
        --poi_info "$POI_INFO" --role_priors "$ROLE_PRIORS" "${ID_MAPPING_ARGS[@]}" \
        --input_sid_csv "$CURRENT_SID" --base_model "$BASE_MODEL" \
        --adapter_dir "$EM_ADAPTER" --tokenizer_path "$EM_ADAPTER" \
        --output_sid_csv "$ITER_ROOT/unused_shard_${shard_index}_sid.csv" \
        --candidates_json "$SHARD_JSON" --report_json "$SHARD_REPORT" \
        --codebook_size "$CODEBOOK_SIZE" --num_beams "$EM_BEAMS" \
        --device cuda:0 --shard_index "$shard_index" --num_shards 2 \
        --candidates_only \
        >"$RUN_ROOT/logs/em_${iteration}_candidates_shard_${shard_index}_gpu${gpu}.log" 2>&1 &
      SHARD_PIDS+=("$!")
    fi
  done
  for pid in "${SHARD_PIDS[@]}"; do wait "$pid"; done
  for shard_index in 0 1; do
    json_ok "$ITER_ROOT/candidates_shard_${shard_index}.json"
    json_ok "$ITER_ROOT/candidates_shard_${shard_index}_report.json"
  done

  MERGED_CANDIDATES="$ITER_ROOT/candidates_merged.json"
  if [[ ! -s "$MERGED_CANDIDATES" ]]; then
    "$PYTHON_BIN" -m geogr_full_pipeline.merge_em_candidates \
      --input_sid_csv "$CURRENT_SID" \
      --shard "$ITER_ROOT/candidates_shard_0.json" \
      --shard "$ITER_ROOT/candidates_shard_1.json" \
      --output_json "$MERGED_CANDIDATES"
  fi
  NEXT_SID="$RUN_ROOT/sid/em_iteration_${iteration}_sid.csv"
  if [[ ! -s "$NEXT_SID" ]]; then
    "$PYTHON_BIN" -m geogr_full_pipeline.em_refine_sid \
      --poi_info "$POI_INFO" --role_priors "$ROLE_PRIORS" "${ID_MAPPING_ARGS[@]}" \
      --input_sid_csv "$CURRENT_SID" --base_model "$BASE_MODEL" \
      --adapter_dir "$EM_ADAPTER" --tokenizer_path "$EM_ADAPTER" \
      --output_sid_csv "$NEXT_SID" --candidates_json "$MERGED_CANDIDATES" \
      --report_json "$ITER_ROOT/em_refinement_report.json" \
      --codebook_size "$CODEBOOK_SIZE" --num_beams "$EM_BEAMS" \
      --reuse_candidates
  fi
  json_ok "$ITER_ROOT/em_refinement_report.json"
  CURRENT_SID="$NEXT_SID"
  PREVIOUS_EM_ADAPTER="$EM_ADAPTER"
done

FINAL_SID="$RUN_ROOT/sid/geogr_em_final_sid.csv"
if [[ ! -s "$FINAL_SID" ]]; then cp "$CURRENT_SID" "$FINAL_SID"; fi

if [[ ! -s "$RUN_ROOT/data/llm_json_report.json" ]]; then
  "$PYTHON_BIN" -m tap_sid.build_llm_data \
    --sid_csv "$FINAL_SID" --split_dir "$SPLIT_DIR" --output_dir "$RUN_ROOT/data" \
    --keep_last_k_train "$KEEP_LAST_K_TRAIN"
fi
if [[ ! -s "$RUN_ROOT/cpt/cpt_data_report.json" ]]; then
  "$PYTHON_BIN" -m geogr_full_pipeline.build_cpt_data \
    --poi_info "$POI_INFO" --role_priors "$ROLE_PRIORS" "${ID_MAPPING_ARGS[@]}" \
    --train_sequences "$SPLIT_DIR/train_poi_sequence.csv" --sid_csv "$FINAL_SID" \
    --output_json "$RUN_ROOT/cpt/cpt_train.json" \
    --report_json "$RUN_ROOT/cpt/cpt_data_report.json"
fi

train_sft_variant() {
  local variant_root=$1 init_adapter=${2:-}
  if [[ -s "$variant_root/checkpoint/final_sft/adapter_config.json" ]]; then return; fi
  require_new_dir "$variant_root/checkpoint"
  local init_args=()
  if [[ -n "$init_adapter" ]]; then
    init_args=(--init_adapter_dir "$init_adapter" --tokenizer_path "$init_adapter")
  fi
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node=2 -m geogr_full_pipeline.train_matched_sft \
    --base_model "$BASE_MODEL" "${init_args[@]}" \
    --train_dataset "$RUN_ROOT/data/llm_train.json" \
    --valid_dataset "$RUN_ROOT/data/llm_val.json" \
    --output_dir "$variant_root/checkpoint" \
    --batch_size 1 --eval_batch_size 1 --grad_accum "$SFT_GRAD_ACCUM" \
    --num_train_epochs "$SFT_EPOCHS" --learning_rate "$LEARNING_RATE" \
    --cutoff_len 2048 --lm_loss_weight 1.0 --alpha_prefix 0.0 \
    --head_dropout 0.1 --lora_r 16 --lora_alpha 32 --lora_dropout 0.1 \
    --gradient_checkpointing --seed "$SEED" --logging_steps 120 \
    --attn_implementation sdpa \
    >"$RUN_ROOT/logs/$(basename "$variant_root")_train_gpu_pair.log" 2>&1
}

SFT_ONLY_ROOT="$RUN_ROOT/sft_only"
train_sft_variant "$SFT_ONLY_ROOT"

if [[ ! -s "$RUN_ROOT/cpt/checkpoint/final_cpt/adapter_config.json" ]]; then
  require_new_dir "$RUN_ROOT/cpt/checkpoint"
  log "CPT: 四模板两卡 LoRA 继续预训练。"
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node=2 -m geogr_full_pipeline.train_cpt \
    --base_model "$BASE_MODEL" --dataset "$RUN_ROOT/cpt/cpt_train.json" \
    --output_dir "$RUN_ROOT/cpt/checkpoint" --batch_size 1 \
    --grad_accum "$CPT_GRAD_ACCUM" --num_train_epochs "$CPT_EPOCHS" \
    --learning_rate "$LEARNING_RATE" --warmup_steps 20 --cutoff_len 512 \
    --seed "$SEED" --logging_steps 120 \
    >"$RUN_ROOT/logs/cpt_train_gpu_pair.log" 2>&1
fi
CPT_SFT_ROOT="$RUN_ROOT/cpt_sft"
train_sft_variant "$CPT_SFT_ROOT" "$RUN_ROOT/cpt/checkpoint/final_cpt"

PIPELINE_RUN_ROOT="$RUN_ROOT"
evaluate_variant() {
  local variant_root=$1 run_id=$2
  if [[ -s "$variant_root/eval/test_metrics.json" ]]; then return; fi
  PYTHON_BIN="$PYTHON_BIN" BASE_MODEL="$BASE_MODEL" RUN_ROOT="$variant_root" \
    SEMANTIC_CODES="$FINAL_SID" \
    TEST_DATASET="$PIPELINE_RUN_ROOT/data/llm_test.json" EVAL_GPUS="$GPUS" \
    EVAL_RUN_ID="$run_id" EVAL_NUM_BEAMS="$TEST_BEAMS" EVAL_K=10 EVAL_SEED="$SEED" \
    CUDA_VISIBLE_DEVICES="$GPUS" bash scripts/evaluate.sh \
    >"$PIPELINE_RUN_ROOT/logs/$(basename "$variant_root")_eval_${run_id}.log" 2>&1
  json_ok "$variant_root/eval/test_predictions.json"
  json_ok "$variant_root/eval/test_metrics.json"
}

evaluate_variant "$SFT_ONLY_ROOT" "${DATASET}_geogr_em_sft_beam10${EVAL_RUN_SUFFIX}"
evaluate_variant "$CPT_SFT_ROOT" "${DATASET}_geogr_em_cpt_sft_beam10${EVAL_RUN_SUFFIX}"
log "GeoGR matched-protocol pipeline completed: $RUN_ROOT"
