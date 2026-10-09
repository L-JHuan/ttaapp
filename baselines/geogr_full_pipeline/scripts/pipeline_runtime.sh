#!/usr/bin/env bash
# 每次调用单独留日志，静默 CPU 阶段也保留心跳；不伪造算法进度。
run_stage() {
  local name=$1 status=0 child heartbeat started=$SECONDS
  shift
  CURRENT_STAGE_LOG="$RUN_ROOT/logs/${name}_${PIPELINE_RUN_ID}.log"
  mkdir -p "$RUN_ROOT/logs"
  echo "[$(date -Is)] START $name" >>"$CURRENT_STAGE_LOG"
  printf 'Command:' >>"$CURRENT_STAGE_LOG"
  printf ' %q' "$@" >>"$CURRENT_STAGE_LOG"
  printf '\n' >>"$CURRENT_STAGE_LOG"
  echo "[$(date -Is)] $name: Log: $CURRENT_STAGE_LOG"
  (trap - ERR; exec "$@") >>"$CURRENT_STAGE_LOG" 2>&1 &
  child=$!
  (
    trap - ERR
    sleeper=""
    trap '[[ -z "$sleeper" ]] || kill "$sleeper" 2>/dev/null; exit 0' TERM INT
    while true; do
      sleep "${STAGE_HEARTBEAT_SECONDS:-60}" &
      sleeper=$!
      wait "$sleeper" || break
      kill -0 "$child" 2>/dev/null || break
      echo "[$(date -Is)] RUNNING $name elapsed=$((SECONDS-started))s pid=$child"
    done
  ) >>"$CURRENT_STAGE_LOG" 2>&1 &
  heartbeat=$!
  # 分片启动时本函数运行于独立子 shell；仅终止它自己的子进程。
  trap 'kill "$child" "$heartbeat" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 143' TERM INT
  if wait "$child"; then status=0; else status=$?; fi
  kill "$heartbeat" 2>/dev/null || true
  wait "$heartbeat" 2>/dev/null || true
  trap - TERM INT
  if (( status != 0 )); then
    echo "[$(date -Is)] FAILED exit=$status elapsed=$((SECONDS-started))s" >>"$CURRENT_STAGE_LOG"
    echo "Stage log: $CURRENT_STAGE_LOG" >&2
    tail -n 60 "$CURRENT_STAGE_LOG" >&2
    return "$status"
  fi
  echo "[$(date -Is)] DONE exit=0 elapsed=$((SECONDS-started))s" >>"$CURRENT_STAGE_LOG"
  echo "[$(date -Is)] $name completed."
}

check_gpu_communication() {
  # 保留原 GPU 数量，先测原生通信；仅失败后尝试兼容传输，不自动降低卡数。
  if run_stage nccl_preflight timeout --signal=TERM --kill-after=15s 180s env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -u \
      -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
      -m geogr_full_pipeline.nccl_preflight --report_dir "$RUN_ROOT/preflight/$PIPELINE_RUN_ID/native"; then
    return 0
  fi
  [[ "${NCCL_COMPAT_RETRY:-1}" == "1" ]] || return 1
  echo "Native NCCL preflight failed; testing socket compatibility on the same $GPU_COUNT GPUs."
  export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
  export NCCL_CUMEM_HOST_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
  if run_stage nccl_preflight_compat timeout --signal=TERM --kill-after=15s 180s env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -u \
      -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
      -m geogr_full_pipeline.nccl_preflight --report_dir "$RUN_ROOT/preflight/$PIPELINE_RUN_ID/compat"; then
    echo "NCCL socket compatibility passed; training will use this profile (communication may be slower)."
    return 0
  fi
  echo "Both communication profiles failed; no model training started. Provide both preflight logs." >&2
  return 1
}
