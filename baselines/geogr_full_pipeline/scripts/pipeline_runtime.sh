#!/usr/bin/env bash
# 每次调用单独留日志，静默 CPU 阶段也保留心跳；不伪造算法进度。
show_stage_error() {
  local file=$1 root=${GEOGR_SCRIPT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
  echo "Stage log: $file" >&2
  # 不依赖 torch 导入；优先展示底层根因和第一个原始堆栈。
  "${PYTHON_BIN:-python}" "$root/geogr_full_pipeline/summarize_stage_error.py" "$file" >&2 || tail -n 60 "$file" >&2
}

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
  (trap - ERR; exec env GEOGR_STAGE="$name" "$@") >>"$CURRENT_STAGE_LOG" 2>&1 &
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
    show_stage_error "$CURRENT_STAGE_LOG"
    return "$status"
  fi
  echo "[$(date -Is)] DONE exit=0 elapsed=$((SECONDS-started))s" >>"$CURRENT_STAGE_LOG"
  echo "[$(date -Is)] $name completed."
}

enable_compat_transport() {
  export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
  export NCCL_CUMEM_HOST_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0
}

check_gpu_communication() {
  # 小模型预检不覆盖加载大模型后的内存状态，默认直接验证兼容配置。
  local profile=${GEOGR_NCCL_PROFILE:-socket}
  case "$profile" in
    socket)
      enable_compat_transport
      echo "Validate explicit socket compatibility on the same $GPU_COUNT GPUs; model/optimizer settings unchanged."
      if run_stage nccl_preflight_compat timeout --signal=TERM --kill-after=15s 180s env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -u \
          -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
          -m geogr_full_pipeline.nccl_preflight --report_dir "$RUN_ROOT/preflight/$PIPELINE_RUN_ID/compat"; then
        echo "Socket compatibility preflight passed; this profile stays enabled for subsequent training."
        return 0
      fi
      echo "Socket compatibility preflight failed; no training started." >&2
      return 1
      ;;
    auto|native) ;;
    *) echo "Unknown GEOGR_NCCL_PROFILE=$profile; expected socket, auto or native." >&2; return 2 ;;
  esac
  if run_stage nccl_preflight timeout --signal=TERM --kill-after=15s 180s env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -u \
      -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
      -m geogr_full_pipeline.nccl_preflight --report_dir "$RUN_ROOT/preflight/$PIPELINE_RUN_ID/native"; then
    return 0
  fi
  [[ "$profile" != "native" ]] || return 1
  [[ "${NCCL_COMPAT_RETRY:-1}" == "1" ]] || return 1
  echo "Native NCCL preflight failed; testing socket compatibility on the same $GPU_COUNT GPUs."
  enable_compat_transport
  if run_stage nccl_preflight_compat timeout --signal=TERM --kill-after=15s 180s env CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -u \
      -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
      -m geogr_full_pipeline.nccl_preflight --report_dir "$RUN_ROOT/preflight/$PIPELINE_RUN_ID/compat"; then
    echo "NCCL socket compatibility passed; training will use this profile (communication may be slower)."
    return 0
  fi
  echo "Both communication profiles failed; no model training started. Provide both preflight logs." >&2
  return 1
}
